import logging
import os
import tempfile
import time
import warnings

import cv2
import numpy as np
import torch
from model_service.pytorch_model_service import PTServingBaseService

logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from models.ONet import ONet
from models.PNet import PNet
from models.RNet import RNet
from utils.cal import cal_euler_angles
from utils.utils import calibrate_box, convert_to_square, generate_bbox, pad, processed_image, py_nms

pnet = None
rnet = None
onet = None
softmax_p = None
softmax_r = None
softmax_o = None


def load_model(model_dir, device):
    pNet = PNet()
    pNet.load_state_dict(torch.load(os.path.join(model_dir, "PNet.pt")))
    pNet.to(device)
    pNet.eval()
    rNet = RNet()
    rNet.load_state_dict(torch.load(os.path.join(model_dir, "RNet.pt")))
    rNet.to(device)
    rNet.eval()
    oNet = ONet()
    oNet.load_state_dict(torch.load(os.path.join(model_dir, "ONet.pt")))
    oNet.to(device)
    oNet.eval()

    return pNet, rNet, oNet


def load_models(models_dir, device):
    """
    param:
        models_dir: 模型文件夹路径
        device
    """
    global pnet, rnet, onet, softmax_p, softmax_r, softmax_o
    pnet, rnet, onet = load_model(models_dir, device)

    softmax_p = torch.nn.Softmax(dim=0)
    softmax_r = torch.nn.Softmax(dim=-1)
    softmax_o = torch.nn.Softmax(dim=-1)


# 使用PNet模型预测
def predict_pnet(infer_data, device):
    """
    detect_pnet函数的子函数
    """
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
def predict_rnet(infer_data, device):
    """
    detect_rnet函数的子函数
    """
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用ONet模型预测
def predict_onet(infer_data, device):
    """
    detect_onet函数的子函数
    """
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, landmark_pred = onet(infer_data)
    cls_prob = softmax_o(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()


# 获取PNet网络输出结果
def detect_pnet(im, min_face_size, scale_factor, thresh, device):
    """
    param:
        im: 图片
        min_face_size: 最小人脸尺寸
        scale_factor: 缩放因子
        thresh: 阈值
        device: 设备
    return:
        boxes_c: 人脸检测框坐标
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
        cls_cls_map, reg = predict_pnet(im_resized, device=device)
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
def detect_rnet(im, dets, thresh, device):
    """
    通过rent选择box
        params:
            im：输入图像
            dets:pnet选择的box，是相对原图的绝对坐标
        返回值:
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
    cls_scores, reg = predict_rnet(cropped_ims, device=device)
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
def detect_onet(im, dets, thresh, device):
    """
    将onet的选框继续筛选基本和rnet差不多但多返回了landmark
    """
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
    cls_scores, reg, landmark = predict_onet(cropped_ims, device=device)

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


def infer_image(image_path, device):
    """
    对单张图片进行人脸检测，输出欧拉角
    param
        image_path: 图片路径
        device: 设备
    return:
        roll,yaw,pitch: 欧拉角
    """
    im = cv2.imread(image_path)
    start = time.time()
    # 调用PNet模型预测
    boxes_c = detect_pnet(im=im, min_face_size=20, scale_factor=0.79, thresh=0.9, device=device)
    # 筛选在图像右半边的人脸，即boxes_c[:,0]>im.shape[1]/2
    boxes_c = boxes_c[boxes_c[:, 0] > im.shape[1] / 2]
    if boxes_c is None:
        return None, None, None
    # 调用RNet模型预测
    boxes_c = detect_rnet(im, boxes_c, 0.6, device)
    # 筛选在图像右半边的人脸，即boxes_c[:,0]>im.shape[1]/2
    boxes_c = boxes_c[boxes_c[:, 0] > im.shape[1] / 2]
    if boxes_c is None:
        return None, None, None

    # 调用第三个模型预测
    boxes_c, landmark = detect_onet(im, boxes_c, 0.5, device)
    # 筛选在图像右半边的人脸，即boxes_c[:,0]>im.shape[1]/2
    boxes_c = boxes_c[boxes_c[:, 0] > im.shape[1] / 2]
    if boxes_c is None:
        return None, None, None
    # 计算欧拉角
    points = landmark[0]
    # 将关键点坐标0-4为x,5-9为y
    points_x_y = np.array(
        [points[0], points[2], points[4], points[6], points[8], points[1], points[3], points[5], points[7], points[9]]
    )
    roll, yaw, pitch = cal_euler_angles(points_x_y)

    result = {"result": {"Roll": 0.0, "Yaw": 0.0, "Pitch": 0.0, "duration": 6000}}

    result["result"]["duration"] = int((time.time() - start) * 1000)
    result["result"]["Roll"] = roll
    result["result"]["Yaw"] = yaw
    result["result"]["Pitch"] = pitch
    return result


def get_column(matrix, column_number):
    column = [row[column_number] for row in matrix]
    return column


def infer_video(video_path, device, fps=None):
    """
    对.mp4视频进行人脸检测，输出.mp4视频到output_path
    param:
        video_path:视频路径
        output_path:输出视频路径
        fps:帧率
        device:设备
    return:
        euler_angles_per_frame: 每一帧的欧拉角,若该帧没有检测到人脸，则返回[-1,-1,-1]

    """
    start = time.time()
    cap = cv2.VideoCapture(video_path)  # 读取视频
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    else:
        cap.set(cv2.CAP_PROP_FPS, fps)

    c_roll = None
    c_yaw = None
    c_pitch = None  # 初始值设置为None，后续用来计算相对欧拉角

    euler_angles_per_frame = []
    while True:
        ret, frame = cap.read()  # 读取视频帧
        if ret:
            # 调用第一个模型预测
            boxes_c = detect_pnet(frame, 20, 0.79, 0.9, device=device)
            if boxes_c is None:
                euler_angles_per_frame.append([-1.0, -1.0, -1.0])
                continue
            # 调用第二个模型预测
            boxes_c = detect_rnet(frame, boxes_c, 0.5, device=device)
            if boxes_c is None:
                euler_angles_per_frame.append([-1.0, -1.0, -1.0])
                continue
            # 调用第三个模型预测
            # 筛选在图像右半边的人脸，即boxes_c[:,0]>frame.shape[1]/2
            boxes_c = boxes_c[boxes_c[:, 0] > frame.shape[1] / 2]
            if boxes_c is None:
                euler_angles_per_frame.append([-1.0, -1.0, -1.0])
                continue
            boxes_c, landmark = detect_onet(frame, boxes_c, 0.5, device=device)
            if boxes_c is None:
                euler_angles_per_frame.append([-1.0, -1.0, -1.0])
                continue

            # 选择人脸置信度最大的人脸
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
            euler_angles_per_frame.append(euler_angles)
        else:
            break

    result = {"result": {"Roll": [], "Yaw": [], "Pitch": [], "duration": 6000}}

    result["result"]["duration"] = int((time.time() - start) * 1000)
    result["result"]["Roll"] = get_column(euler_angles_per_frame, 0)
    result["result"]["Yaw"] = get_column(euler_angles_per_frame, 1)
    result["result"]["Pitch"] = get_column(euler_angles_per_frame, 2)

    return result


def infer_image_with_Onet(image_path, boxes_c=None, device=None):
    """
    param:
        image_path: 图片路径
        boxes_c: 人脸检测框[x1,y1,x2,y2,score]
    return:
        roll,yaw,pitch: 欧拉角

    """
    im = cv2.imread(image_path)
    if boxes_c is None:
        return None, None, None
    # 调用第三个模型预测
    boxes_c, landmark = detect_onet(im, boxes_c, 0.5, device=device)
    if boxes_c is None:
        return None, None, None
    # 计算欧拉角
    points = landmark[0]
    # 将关键点坐标0-4为x,5-9为y
    points_x_y = np.array(
        [points[0], points[2], points[4], points[6], points[8], points[1], points[3], points[5], points[7], points[9]]
    )
    roll, yaw, pitch = cal_euler_angles(points_x_y)
    return roll, yaw, pitch


class MTCNN_model:
    def __init__(self):
        self.device = torch.device("cuda")
        load_models("/home/ma-user/infer/model/1/infer_models", self.device)

    def inference(self, source):
        res = infer_video(source, self.device)

        return res


class PTVisionService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        # super(PTVisionService, self).__init__(model_name, model_path)
        # 调用自定义函数加载模型
        self.capture = None  # Fixed: set via tempfile in _preprocess to avoid race condition
        self.model_name = model_name
        self.model_path = model_path
        self.model = MTCNN_model()

    def _inference(self, data):
        result = self.model.inference(source=self.capture)

        return result

    def _preprocess(self, data):
        # 这个函数把data写到test.mp4里面了
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        tmp = tempfile.NamedTemporaryFile(
                            suffix=".mp4", delete=False
                        )  # Fixed: use tempfile to avoid race condition
                        self.capture = tmp.name
                        tmp.close()
                        with open(self.capture, "wb") as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                except Exception:
                    return {"message": "There was an error processing the file"}
        return "ok"

    def _postprocess(self, data):
        if self.capture and os.path.exists(self.capture):  # Fixed: clean up temp file
            os.remove(self.capture)
        return data
