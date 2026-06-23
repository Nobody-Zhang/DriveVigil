from __future__ import annotations

import logging

# import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# import argparse
import torch

logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from utils.plots import *
from utils.torch_utils import select_device, time_sync


# 将字节转换为GB
def bytes_to_gigabytes(bytes_value: int) -> float:
    """Convert a byte count to gigabytes.

    Args:
        bytes_value: Number of bytes to convert.

    Returns:
        The equivalent value in gigabytes.
    """
    return bytes_value / (1024 * 1024 * 1024)


def load_imgs(dataset: LoadImages, half: bool, device: torch.device) -> List[Tuple[torch.Tensor, np.ndarray]]:
    """Load all video frames into memory as preprocessed tensors.

    Iterates through every frame in the dataset, converts each to a
    normalized float tensor (optionally FP16), and pairs it with the
    original numpy image for later use by the divide-and-conquer
    algorithm.

    Args:
        dataset: A YOLOv5 ``LoadImages`` iterator over video frames.
        half: If True, convert tensors to FP16 half-precision.
        device: The torch device to place tensors on.

    Returns:
        A list of ``(tensor, original_image)`` tuples for every frame,
        where *tensor* has shape ``(1, C, H, W)`` and *original_image*
        is the unmodified ``(H, W, C)`` numpy array.
    """
    il: List[Tuple[torch.Tensor, np.ndarray]] = []
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        il.append((im, im0s))  # save every frame
    return il


# write by llr
# transform xyxy loacationn to xywh loacation, scale in (0, 1)
def xyxy2xywh_normalized(
    xmin: int, ymin: int, xmax: int, ymax: int, wide: int, height: int
) -> Tuple[float, float, float, float]:  # Fixed: renamed to avoid shadowing utils.general.xyxy2xywh
    """Convert bounding box from xyxy to normalized xywh format.

    Transforms corner coordinates ``(xmin, ymin, xmax, ymax)`` to
    center-based coordinates ``(x, y, w, h)`` normalized to ``[0, 1]``
    by image dimensions.

    Args:
        xmin: Left edge x-coordinate in pixels.
        ymin: Top edge y-coordinate in pixels.
        xmax: Right edge x-coordinate in pixels.
        ymax: Bottom edge y-coordinate in pixels.
        wide: Image width in pixels (normalization denominator).
        height: Image height in pixels (normalization denominator).

    Returns:
        A 4-tuple ``(x, y, w, h)`` where ``x, y`` are the normalized
        center coordinates and ``w, h`` are the normalized width and
        height, all in ``[0, 1]``.
    """
    x = ((xmin + xmax) // 2) / wide
    y = ((ymin + ymax) // 2) / height
    w = (xmax - xmin) / wide
    h = (ymax - ymin) / height
    return x, y, w, h


class YOLO_Status:
    """Classifies driver behavioral status from YOLO detection results.

    Analyzes bounding boxes detected by the YOLO model (face, sideface,
    eyes, mouth, phone) to classify each video frame into one of five
    driver states. Uses spatial heuristics and a priority system to
    resolve conflicting detections.

    Attributes:
        cls_: Mapping from detection class names to YOLO class indices.
        status_prior: Priority values for each behavioral status.
        condition: Maps internal priority indices to output category codes
            (0=normal, 1=eyes closed, 2=yawn, 3=calling, 4=turning).
    """

    def __init__(self) -> None:
        self.cls_: Dict[str, int] = {
            "close_eye": 0,
            "close_mouth": 1,
            "face": 2,
            "open_eye": 3,
            "open_mouth": 4,
            "phone": 5,
            "sideface": 6,
        }
        self.status_prior: Dict[str, int] = {"normal": 0, "closeeye": 1, "yawn": 3, "calling": 4, "turning": 2}
        self.condition: List[int] = [0, 1, 4, 2, 3]

    def determin(self, img: np.ndarray, dets: np.ndarray) -> int:
        """Determine the driver's behavioral status for a single frame.

        Processes all detection boxes to identify the driver's face (front
        or side), eyes (open/closed), mouth (open/closed), and phone. Uses
        spatial heuristics to associate body parts with the driver (rightmost,
        lowest face) and resolves conflicts via the priority system.

        Args:
            img: The original frame as a numpy array of shape ``(H, W, C)``,
                used to obtain image dimensions for coordinate normalization.
            dets: Detection results as a numpy array of shape ``(N, 6)``,
                where each row is ``[x1, y1, x2, y2, confidence, class_id]``.

        Returns:
            An integer status code:
                - 0: normal driving
                - 1: eyes closed
                - 2: yawning
                - 3: calling (phone use)
                - 4: turning around / looking away
        """
        wide, height = img.shape[1], img.shape[0]  # 输入图片宽、高
        status = 0  # 最终状态，默认为0
        driver = (0, 0, 0, 0)  # 司机正脸xywh坐标
        driver_xyxy = (0, 0, 0, 0)  # 司机正脸xyxy坐标
        driver_conf = 0  # 正脸可信度
        sideface = (0, 0, 0, 0)  # 司机侧脸xywh坐标
        sideface_xyxy = (0, 0, 0, 0)  # 侧脸xyxy坐标
        sideface_conf = 0  # 侧脸可信度
        face = (0, 0, 0, 0)  # 司机的脸，不管正侧
        face_xyxy = (0, 0, 0, 0)  # 司机的脸xyxy坐标
        phone = (0, 0, 0, 0)  # 手机xywh坐标
        openeye = (0, 0, 0, 0)  # 睁眼xywh坐标
        closeeye = (0, 0, 0, 0)  # 闭眼xywh坐标， 以防两只眼睛识别不一样
        openeye_score = 0  # 睁眼可信度
        closeeye_score = 0  # 闭眼可信度
        eyes = []  # 第一遍扫描眼睛列表
        mouth = (0, 0, 0, 0)  # 嘴xywh坐标
        mouth_status = 0  # 嘴状态，0 为闭， 1为张
        mouths = []  # 第一遍扫描嘴列表
        phone_flag = False
        face_flag = False

        # 处理boxes
        bboxes = dets
        for box in bboxes:  # 遍历每个box
            xyxy = tuple(box[:4])  # xyxy坐标
            xywh = xyxy2xywh_normalized(*xyxy, wide, height)  # xywh坐标
            conf = box[4]  # 可信度
            cls = box[5]  # 类别
            if cls == self.cls_["face"]:  # 正脸
                if 0.5 < xywh[0] and xywh[1] > driver[1]:
                    # box中心在右侧0.5 并且 在司机下侧
                    driver = xywh  # 替换司机
                    driver_xyxy = xyxy
                    driver_conf = conf
                    face_flag = True
            elif cls == self.cls_["sideface"]:  # 侧脸
                if 0.5 < xywh[0] and xywh[1] > sideface[1]:  # box位置，与face一致
                    sideface = xywh  # 替换侧脸
                    sideface_xyxy = xyxy
                    sideface_conf = conf
                    face_flag = True
            elif cls == self.cls_["phone"]:  # 手机
                if 0.4 < xywh[0] and 0.2 < xywh[1] and xywh[1] > phone[1] and xywh[0] > phone[0]:
                    # box位置在右0.4, 下0.2, 原手机右下
                    phone = xywh  # 替换手机
                    phone_flag = True  # 表示当前其实有手机
            elif cls == self.cls_["open_eye"] or cls == self.cls_["close_eye"]:  # 眼睛，先存着
                eyes.append((cls, xywh, conf))
            elif cls == self.cls_["open_mouth"] or cls == self.cls_["close_mouth"]:  # 嘴，先存着
                mouths.append((cls, xywh))

        if not face_flag:  # 没有检测到脸
            return 4  # 4 -> turning around

        # 判断状态
        face = driver
        face_xyxy = driver_xyxy
        if (
            abs(driver[0] - sideface[0]) < 0.1 and abs(driver[1] - sideface[1]) < 0.1
        ):  # 正脸与侧脸很接近，说明同时检测出了正脸和侧脸
            if driver_conf > sideface_conf:  # 正脸可信度更高
                status = max(status, self.status_prior["normal"])
                face = driver
                face_xyxy = driver_xyxy
            else:  # 侧脸可信度更高
                status = max(status, self.status_prior["turning"])
                face = sideface
                face_xyxy = sideface_xyxy
        elif sideface[0] > driver[0]:  # 正侧脸不重合，并且侧脸在正脸右侧，说明司机是侧脸
            status = max(status, self.status_prior["turning"])
            face = sideface
            face_xyxy = sideface_xyxy

        if face[2] == 0:  # 司机躲猫猫捏
            status = max(status, self.status_prior["turning"])

        if abs(face[0] - phone[0]) < 0.3 and abs(face[1] - phone[1]) < 0.3 and phone_flag:
            status = max(status, self.status_prior["calling"])  # 判断状态为打电话

        for eye_i in eyes:
            if (
                eye_i[1][0] < face_xyxy[0] / wide
                or eye_i[1][0] > face_xyxy[2] / wide
                or eye_i[1][1] < face_xyxy[1] / height
                or eye_i[1][1] > face_xyxy[3] / height
            ):
                continue
            if eye_i[0] == self.cls_["open_eye"]:  # 睁眼
                if eye_i[1][0] > openeye[0]:  # 找最右边的，下面的同理
                    openeye = eye_i[1]
                    openeye_score = eye_i[2]
            elif eye_i[0] == self.cls_["close_eye"]:  # 睁眼
                if eye_i[1][0] > closeeye[0]:  # 找最右边的，下面的同理
                    closeeye = eye_i[1]
                    closeeye_score = eye_i[2]

        for mouth_i in mouths:
            if (
                mouth_i[1][0] < face_xyxy[0] / wide
                or mouth_i[1][0] > face_xyxy[2] / wide
                or mouth_i[1][1] < face_xyxy[1] / height
                or mouth_i[1][1] > face_xyxy[3] / height
            ):
                continue
            if mouth_i[0] == self.cls_["open_mouth"]:  # 张嘴
                if mouth_i[1][0] > mouth[0]:
                    mouth = mouth_i[1]
                    mouth_status = 1
            elif mouth_i[0] == self.cls_["close_mouth"]:  # 闭嘴
                if mouth_i[1][0] > mouth[0]:
                    mouth = mouth_i[1]
                    mouth_status = 0

        if mouth_status == 1:  # 嘴是张着的
            status = max(status, self.status_prior["yawn"])

        if abs(closeeye[0] - openeye[0]) < 0.2:  # 睁眼和闭眼离得很近， 说明是同一个人两只眼睛判断得不一样
            if closeeye_score > openeye_score:  # 闭眼可信度比睁眼高
                status = max(status, self.status_prior["closeeye"])
            else:
                status = max(status, self.status_prior["normal"])
        else:  # 说明是两个人的眼睛，靠右边的是司机的眼睛
            if closeeye[0] > openeye[0]:  # 司机是闭眼
                status = max(status, self.status_prior["closeeye"])
            else:  # 司机是睁眼
                status = max(status, self.status_prior["normal"])

        return self.condition[status]


@torch.no_grad()
def yolo_run(
    weights: Union[str, Path] = ROOT / "fine_tune_openvino_model/best.xml",  # model.pt path(s)
    source: str = "",  # file/dir/URL/glob, 0 for webcam
    data: Union[str, Path] = ROOT / "fine_tune_openvino_model/best.yaml",  # dataset.yaml path
    imgsz: Tuple[int, int] = (640, 640),  # inference size (height, width)
    conf_thres: float = 0.20,  # confidence threshold
    iou_thres: float = 0.40,  # NMS IOU threshold
    max_det: int = 1000,  # maximum detections per image
    device: str = "cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes: Optional[List[int]] = None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms: bool = False,  # class-agnostic NMS
    augment: bool = False,  # augmented inference
    visualize: bool = False,  # visualize features
    half: bool = False,  # use FP16 half-precision inference
    dnn: bool = False,  # use OpenCV DNN for ONNX inference
    iou_presice_b_search: float = 0.05,  # 二分时间误差系数，准确率优先，给到0.05
) -> Dict[str, Any]:
    """Run YOLO inference with divide-and-conquer temporal localization.

    Instead of classifying every frame, this function uses a
    divide-and-conquer strategy to efficiently locate temporal segments
    of dangerous driving behaviors in dashcam videos:

    1. **Load** all frames into memory as preprocessed tensors.
    2. **Divide-and-conquer**: recursively probe the midpoint of each
       time segment, expand outward to estimate the extent of any
       detected behavior, then partition the remaining timeline and
       recurse on both sides.
    3. **Binary search** (``b_search``): refine the start and end frame
       boundaries of each candidate behavior to sub-second precision,
       controlled by ``iou_presice_b_search``.
    4. **Filter**: discard behaviors shorter than 3 seconds (competition
       requirement: ``fps * 3`` frames minimum).

    Args:
        weights: Path to the model weights file (OpenVINO ``.xml``,
            PyTorch ``.pt``, or ONNX ``.onnx``).
        source: Path to the input video file.
        data: Path to the dataset YAML config file.
        imgsz: Inference input size as ``(height, width)``.
        conf_thres: Minimum confidence threshold for detections.
        iou_thres: IoU threshold for Non-Maximum Suppression.
        max_det: Maximum number of detections per frame.
        device: Inference device (``'cpu'`` or CUDA device id).
        classes: Optional list of class indices to filter detections.
        agnostic_nms: If True, use class-agnostic NMS.
        augment: If True, use augmented inference.
        visualize: If True, visualize feature maps.
        half: If True, use FP16 half-precision inference.
        dnn: If True, use OpenCV DNN backend for ONNX.
        iou_presice_b_search: Binary search error coefficient — lower
            values give tighter temporal boundaries at the cost of more
            inference calls. 0.05 prioritizes accuracy.

    Returns:
        A dict with structure::

            {
                "result": {
                    "duration": <int>,   # total inference time in ms
                    "drowsy": [          # detected behavior periods
                        {
                            "periods": [start_ms, end_ms],
                            "category": <int>
                        },
                        ...
                    ]
                }
            }

        Category codes: 1=eyes closed, 2=yawn, 3=calling, 4=turning.
    """
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
    tmp: List[List[Any]] = []
    sta_tmp: Dict[int, int] = {}

    YOLO_determin = YOLO_Status()

    def f(probe_im_0: int) -> int:
        """Get the driver status for a specific frame index.

        Uses memoization (``sta_tmp``) to avoid redundant inference on
        already-classified frames. Out-of-bounds indices return 0
        (normal).

        Args:
            probe_im_0: Zero-based frame index to classify.

        Returns:
            Integer status code (0-4) for the frame.
        """
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

    def b_search(
        l1: float, r1: float, l2: float, r2: float, n: float, goal_n: float, k: int, is_3: bool = False
    ) -> List[Any]:
        """Binary search to refine temporal boundaries of a behavior.

        Given a candidate behavior segment, this function narrows down
        the exact start and end frame boundaries using binary search.
        The search space is defined by two intervals:

        - ``[l1, r1]``: contains the left (start) boundary
        - ``[l2, r2]``: contains the right (end) boundary

        At each step, it probes the midpoints of both intervals and
        adjusts the search space based on whether each midpoint matches
        the target behavior ``k``:

        - **Both match** (1, 1): narrow inward (the behavior extends
          across both midpoints, so the true boundaries are outside).
        - **Neither match** (0, 0): narrow outward (the behavior is
          contained between the midpoints, if ``is_3`` then reject).
        - **Mixed** (1, 0) or (0, 1): narrow the non-matching side.

        Recursion terminates when accumulated error ``n`` falls within
        ``goal_n``, at which point the midpoints of the remaining
        intervals are taken as the boundary estimates. Segments shorter
        than 3 seconds are rejected.

        Args:
            l1: Left bound of the left (start) search interval, in frames.
            r1: Right bound of the left (start) search interval, in frames.
            l2: Left bound of the right (end) search interval, in frames.
            r2: Right bound of the right (end) search interval, in frames.
            n: Current accumulated error (halved at each recursion).
            goal_n: Target error threshold — search stops when ``n <= goal_n``.
            k: The target behavior status code to search for.
            is_3: If True, this segment is borderline (~3s) and requires
                stricter validation with extra iterations.

        Returns:
            A list where the first element is a bool indicating success:
                - ``[True, start_sec, end_sec]``: valid behavior found with
                  boundary timestamps in seconds.
                - ``[False]``: no valid behavior of sufficient duration.
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

    def divide_and_conquer(lo: int, hi: int) -> None:
        """Recursively partition the video timeline to find behavior segments.

        This is the core temporal localization algorithm. It works by:

        1. **Probe**: classify the middle frame of the segment ``[lo, hi]``.
        2. **Expand**: if the middle frame shows a dangerous behavior,
           step outward in 0.375-second increments to estimate how far
           the behavior extends in both directions.
        3. **Record**: if the expansion covers >= 9 steps (~2.625s), the
           candidate is recorded for later binary search refinement.
        4. **Recurse**: process the remaining segments on both sides of
           the expanded region.

        Segments shorter than 3 seconds (``3 * fps`` frames) are pruned
        as they cannot contain a valid behavior period.

        Args:
            lo: Left boundary frame index (inclusive).
            hi: Right boundary frame index (inclusive).
        """
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
    res: List[Dict[str, Any]] = []
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

    result: Dict[str, Any] = {"result": {"duration": 6000, "drowsy": 0}}
    result["result"]["drowsy"] = res

    result["result"]["duration"] = int(duration * 1000)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    list = yolo_run(source=ROOT / "zipped.mp4")
    logger.info(list)
