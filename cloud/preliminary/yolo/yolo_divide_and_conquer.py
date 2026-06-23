# import argparse
import logging
import os
import sys
from pathlib import Path

# import gc
import cv2
import torch

logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from status import YOLO_Status  # extracted pure classifier (see status.py)
from utils.datasets import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from utils.plots import *
from utils.torch_utils import select_device, time_sync


# 将字节转换为GB
def bytes_to_gigabytes(bytes_value):
    return bytes_value / (1024 * 1024 * 1024)


def load_imgs(dataset, half, device):
    il = []
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        il.append((im, im0s))  # save every frame
    return il


@torch.no_grad()
def yolo_run(
    weights=ROOT / "fine_tune_openvino_model/best.xml",  # model.pt path(s)
    source="",  # file/dir/URL/glob, 0 for webcam
    data=ROOT / "fine_tune_openvino_model/best.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.20,  # confidence threshold
    iou_thres=0.40,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    iou_presice_b_search=0.05,  # 二分时间误差系数，准确率优先，给到0.05
):
    source = str(source)
    # ------------------------- Init model -------------------------
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, pt, jit, onnx, engine = model.stride, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half &= (pt or jit or onnx or engine) and device.type != "cpu"  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    fps = dataset.cap.get(cv2.CAP_PROP_FPS)
    im_lis = load_imgs(dataset, half, device)  # 保存所有的帧便于后续分治
    tmp = []
    sta_tmp = {}

    YOLO_determin = YOLO_Status()

    def f(probe_im_0):
        # 得到probe_im_0的状态（均以帧为单位）
        if probe_im_0 in sta_tmp:
            return sta_tmp[probe_im_0]
        if probe_im_0 >= len(im_lis) or probe_im_0 < 0:
            return 0
        im = im_lis[probe_im_0][0]
        im0s = im_lis[probe_im_0][1]

        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        sta = 0
        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                sta = YOLO_determin.determin(im0, det.numpy())
            else:
                # Nothing detected, assume the status if "turning"
                sta = 4
            # if sta == 1:
            #     cv2.imshow(f"{sta}", im0)
            #     cv2.waitKey(1000)
        sta_tmp[probe_im_0] = sta
        return sta

    def b_search(l1, r1, l2, r2, n, goal_n, k, is_3=False):
        """
        注意：输入的所有的单位为帧数！
        l1: left bound1 左区间的左边界
        r1: right bound1 左区间的右边界
        l2: left bound2 右区间的左边界
        r2: right bound2 右区间的右边界
        n: 现有可能产生的误差之和
        goal_n: 目标可能产生的误差之和
        k: the target status to find 目标状态
        """
        if n <= goal_n:
            # if is_3: # 递归到这个地方，一直都是01, 10, 10, 01这样的，无法更加精确的判断结果到底的状态，一律返回真
            lef_frame_ans = max((l1 + r1) / 2, 0)
            rig_frame_ans = min((l2 + r2) / 2, len(im_lis) - 1)

            if rig_frame_ans - lef_frame_ans < 3 * fps:
                return [False]  # 可能出现的边界条件的判断
            return [True, lef_frame_ans / fps, rig_frame_ans / fps]  # 表示可行，并且返回边界的值

        mid1 = int((l1 + r1) / 2)
        mid2 = int((l2 + r2) / 2)
        sta1 = f(mid1)
        sta2 = f(mid2)

        # 1 1
        if sta1 == k and sta2 == k:
            return b_search(
                l1, mid1, mid2, r2, n / 2, iou_presice_b_search * (mid2 - mid1) / fps, k
            )  # 就算是需要判断的3s，无论如何都是可行的

        # 0 0
        if sta1 != k and sta2 != k:
            if is_3:
                return [False]  # 如果是需要判断的3s，则无论如何都是不可行的
            return b_search(
                mid1, r1, l2, mid2, n / 2, iou_presice_b_search * (l2 - r1) / fps, k
            )  # 继续搜索边界，提升精度

        # 1 0
        if sta1 == k and sta2 != k:
            if is_3:  # 固定时长，多迭代一轮
                return b_search(l1, mid1, l2, mid2, n / 2, iou_presice_b_search * 3 * 0.25, k, True)
            return b_search(l1, mid1, l2, mid2, n / 2, iou_presice_b_search * (l2 - mid1) / fps, k)

        # 0 1
        if sta1 != k and sta2 == k:
            if is_3:
                return b_search(mid1, r1, mid2, r2, n / 2, iou_presice_b_search * 3 * 0.25, k, True)
            return b_search(mid1, r1, mid2, r2, n / 2, iou_presice_b_search * (mid2 - r1) / fps, k)

        return [False]  # Fixed: defensive fallback for theoretically unreachable state

    def divide_and_conquer(lo, hi):
        # 分治算法，lo和hi表示的是左右的边界, [lo, hi]，且左右的状态和lo - 0.5 * fps, hi + 0.5 * fps的状态不一样
        if hi - lo < 3 * fps:  # 区间小于3s
            return
        mid = int((lo + hi) / 2)  # 选中间的帧
        sta_mid = f(mid)
        i = 1
        j = 1
        if sta_mid != 0:
            while int(mid - 0.375 * i * fps) >= lo and f(int(mid - 0.375 * i * fps)) == sta_mid:
                i += 1
            while int(mid + 0.375 * j * fps) <= hi and f(int(mid + 0.375 * j * fps)) == sta_mid:
                j += 1
            if i + j >= 9:  # 表示当前已经有2.625s，但是需要更进一步二分判断
                # 注意保存的是l1，r2的帧，因为这俩都判断是不可行的
                tmp.append([i + j == 9, int(mid - 0.375 * i * fps), int(mid + 0.375 * j * fps), sta_mid])
        divide_and_conquer(lo, int(mid - fps * i * 0.375))
        divide_and_conquer(int(mid + fps * j * 0.375), hi)
        return

    # ------------------------- Run inference -------------------------
    t_start = time_sync()  # Start_time
    divide_and_conquer(0, len(im_lis) - 1)
    # ------------------- Attention! tot_status be like [0, 0, 2, ...] type: int--------------------------
    # for i in range(5):  # Just in case, time of the vidio isn't enouth, append 0
    #     tot_status.append(0)
    # tot_status.append(0)  # 为了最后一个状态的判断，需要多加一个0
    # Post process, using the sliding window algorithm to judge the final status
    res = []
    # 每一帧（抽帧之后的）遍历
    tmp.sort(key=lambda x: x[1])
    for i in tmp:
        min_t = (i[2] - i[1]) / fps - 0.75
        _ = b_search(
            i[1], i[1] + fps * 0.375, i[2] - fps * 0.375, i[2], 0.375, min_t * iou_presice_b_search, i[3], is_3=i[0]
        )
        if _[0]:  # 表示当前出现了大于3s的
            res.append({"periods": [int(_[1] * 1000), int(_[2] * 1000)], "category": i[3]})
    # -------------------- Suit the output format --------------------
    t_end = time_sync()  # End_time
    duration = t_end - t_start

    result = {"result": {"duration": 6000, "drowsy": 0}}
    result["result"]["drowsy"] = res

    result["result"]["duration"] = int(duration * 1000)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    list = yolo_run(source=ROOT / "zipped.mp4")
    logger.info(list)
