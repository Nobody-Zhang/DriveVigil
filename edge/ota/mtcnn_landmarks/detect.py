import argparse
import logging

import cv2
import torch
from utils.cal import cal_euler_angles
from utils.utils import calibrate_box, convert_to_square, generate_bbox, pad, processed_image, py_nms
from utils.visualization import *

logger = logging.getLogger(__name__)

"""
参数解析
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default="infer_models", help="PNet、RNet、ONet三个模型文件存在的文件夹路径"
)
parser.add_argument("--image_path", type=str, default="/home/huawei/DATA", help="需要预测图像的根目录")
parser.add_argument("--save_path", type=str, default="./output/exp", help="需要保存图像的根目录")
parser.add_argument("--device", type=str, default="0", help="0,1 or cpu")
args = parser.parse_args()

output_dir = args.save_path + args.model_path
test_image_rootpath = args.image_path


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取P模型
pnet = torch.jit.load(os.path.join(args.model_path, "PNet.pth"))
pnet.to(device)
softmax_p = torch.nn.Softmax(dim=0)
pnet.eval()

# 获取R模型
rnet = torch.jit.load(os.path.join(args.model_path, "RNet.pth"))
rnet.to(device)
softmax_r = torch.nn.Softmax(dim=-1)
rnet.eval()

# 获取O模型
onet = torch.jit.load(os.path.join(args.model_path, "ONet.pth"))
onet.to(device)
softmax_o = torch.nn.Softmax(dim=-1)
onet.eval()


# 使用PNet模型预测
def predict_pnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    # 执行预测
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用RNet模型预测
def predict_rnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用ONet模型预测
def predict_onet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, landmark_pred = onet(infer_data)
    cls_prob = softmax_o(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()


# 获取PNet网络输出结果
def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    """
    net_size = 12
    # 人脸和输入图像的比率
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    # 图像金字塔
    while min(current_height, current_width) > net_size:
        # 类别和box
        cls_cls_map, reg = predict_pnet(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor  # 继续缩小图像做金字塔
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape

        if boxes.size == 0:
            continue
        # 非极大值抑制留下重复低的box
        keep = py_nms(boxes[:, :5], 0.5, mode="Union")
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    # 将金字塔之后的box也进行非极大值抑制
    keep = py_nms(all_boxes[:, 0:5], 0.7, mode="Union")
    all_boxes = all_boxes[keep]
    # box的长宽
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    # 对应原图的box坐标和分数
    boxes_c = np.vstack(
        [
            all_boxes[:, 0] + all_boxes[:, 5] * bbw,
            all_boxes[:, 1] + all_boxes[:, 6] * bbh,
            all_boxes[:, 2] + all_boxes[:, 7] * bbw,
            all_boxes[:, 3] + all_boxes[:, 8] * bbh,
            all_boxes[:, 4],
        ]
    )
    boxes_c = boxes_c.T

    return boxes_c


# 获取RNet网络输出结果
def detect_rnet(im, dets, thresh):
    """通过rent选择box
    参数：
      im：输入图像
      dets:pnet选择的box，是相对原图的绝对坐标
    返回值：
      box绝对坐标
    """
    h, w, c = im.shape
    # 将pnet的box变成包含它的正方形，可以避免信息损失
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    # 调整超出图像的box
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
    for i in range(int(num_boxes)):
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i] : edy[i] + 1, dx[i] : edx[i] + 1, :] = im[y[i] : ey[i] + 1, x[i] : ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        except Exception:
            continue
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None

    keep = py_nms(boxes, 0.4, mode="Union")
    boxes = boxes[keep]
    # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes_c


# 获取ONet模型预测结果
def detect_onet(im, dets, thresh):
    """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    num_boxes = dets.shape[0]
    cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i] : edy[i] + 1, dx[i] : edx[i] + 1, :] = im[y[i] : ey[i] + 1, x[i] : ex[i] + 1, :]
        img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) / 128
        cropped_ims[i, :, :, :] = img
    cls_scores, reg, landmark = predict_onet(cropped_ims)

    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        landmark = landmark[keep_inds]
    else:
        return None, None

    w = boxes[:, 2] - boxes[:, 0] + 1

    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
    landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
    boxes_c = calibrate_box(boxes, reg)

    keep = py_nms(boxes_c, 0.6, mode="Minimum")
    boxes_c = boxes_c[keep]
    landmark = landmark[keep]
    return boxes_c, landmark


# 预测图片
"""
image_path:图片路径
boxes_c:人脸框,从文件中读入(测试阶段，实际阶段应该从yolov5模型中检测获得)
"""


def infer_image(image_path):
    im = cv2.imread(image_path)
    # 调用第一个模型预测
    boxes_c = detect_pnet(im, 20, 0.79, 0.9)
    if boxes_c is None:
        return None, None
    # 筛选在图像右半边的人脸，即boxes_c[:,0]>im.shape[1]/2
    boxes_c = boxes_c[boxes_c[:, 0] > im.shape[1] / 2]
    if boxes_c is None:
        return None, None
    # 调用第二个模型预测
    boxes_c = detect_rnet(im, boxes_c, 0.0)
    # 筛选在图像右半边的人脸，即boxes_c[:,0]>im.shape[1]/2
    if boxes_c is None:
        return None, None
    boxes_c = boxes_c[boxes_c[:, 0] > im.shape[1] / 2]
    if boxes_c is None:
        return None, None
    # 调用第三个模型预测
    boxes_c, landmark = detect_onet(im, boxes_c, 0.5)
    # 筛选在图像右半边的人脸，即boxes_c[:,0]>im.shape[1]/2
    boxes_c = boxes_c[boxes_c[:, 0] > im.shape[1] / 2]
    if boxes_c is None:
        return None, None

    return boxes_c, landmark


def infer_image_with_Onet(image_path, boxes_c=None):
    im = cv2.imread(image_path)
    if boxes_c is None:
        return None, None
    # 调用第三个模型预测
    boxes_c, landmark = detect_onet(im, boxes_c, 0.5)
    if boxes_c is None:
        return None, None
    return boxes_c, landmark


def detect_video(video_path, output_path):
    """
    video_path:视频路径
    output_path:输出视频路径
    对.mp4视频进行人脸检测，输出.mp4视频到output_path
    """

    c_roll = None
    c_yaw = None
    c_pitch = None  # 初始值设置为None，后续用来计算相对欧拉角

    cap = cv2.VideoCapture(video_path)  # 读取视频
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    # # 人工设置帧率
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频尺寸
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 设置视频编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, size)  # 输出视频
    while True:
        ret, frame = cap.read()  # 读取视频帧
        if ret:
            # 调用第一个模型预测
            boxes_c = detect_pnet(im=frame, min_face_size=20, scale_factor=0.79, thresh=0.9)
            if boxes_c is None:
                out.write(frame)
                continue
            # 调用第二个模型预测
            boxes_c = detect_rnet(im=frame, dets=boxes_c, thresh=0.5)
            if boxes_c is None:
                out.write(frame)
                continue
            # 调用第三个模型预测
            # 筛选在图像右半边的人脸，即boxes_c[:,0]>frame.shape[1]/2
            boxes_c = boxes_c[boxes_c[:, 0] > frame.shape[1] / 2]
            if boxes_c is None:
                out.write(frame)
                continue
            boxes_c, landmark = detect_onet(im=frame, dets=boxes_c, thresh=0.5)
            if boxes_c is None:
                out.write(frame)
                continue
            boxes_c = boxes_c[0:1, :]
            landmark = landmark[0:1, :]
            # 画出人脸框和关键点
            draw_face_video(frame, boxes_c, landmark)
            # 计算欧拉角
            points = landmark[0]
            # 将关键点坐标0-4为x,5-9为y
            points_x_y = np.array(
                [
                    points[0],
                    points[2],
                    points[4],
                    points[6],
                    points[8],
                    points[1],
                    points[3],
                    points[5],
                    points[7],
                    points[9],
                ]
            )
            roll, yaw, pitch = cal_euler_angles(points_x_y)
            if c_roll is None:
                c_roll = roll
                c_yaw = yaw
                c_pitch = pitch
            roll = roll - c_roll
            yaw = yaw - c_yaw
            pitch = pitch - c_pitch
            euler_angles = [roll, yaw, pitch]
            # 标出欧拉角
            draw_euler_angles(frame, euler_angles, axis_length=70, points=points)
            out.write(frame)
        else:
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # 预测图片获取人脸的box和关键点
    for filename in os.listdir(test_image_rootpath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            """
                对图片进行推理
            """
            image_path = os.path.join(test_image_rootpath, filename)
            boxes_c, landmarks = infer_image(image_path)
            # 把关键画出来
            if boxes_c is not None:
                draw_face(image_path=image_path, boxes_c=boxes_c, landmarks=landmarks, output_path=output_dir)
                logger.info("image_name: %s", filename)
            else:
                logger.info("image_name: %s not have face", filename)
        elif filename.endswith(".mp4"):
            video_path = os.path.join(test_image_rootpath, filename)
            output_path = os.path.join(output_dir, filename)
            detect_video(video_path, output_path)
            logger.info("video_name: %s", filename)
