"""Pure, dependency-free per-frame status classification for the cloud detector.

This module holds the core intellectual property of the cloud-side detector: the
geometry heuristics that classify a *single* video frame into one driver-behavior
category, given the YOLO detections for that frame. It deliberately has **no heavy
dependencies** (no ``torch``, ``cv2``, or OpenVINO), so it can be imported and
unit-tested without a GPU, model weights, or cloud credentials.

YOLO class indices are in **alphabetical order** -- this is what the trained model
emits, so keep it in sync with any retraining::

    close_eye=0, close_mouth=1, face=2, open_eye=3, open_mouth=4, phone=5, sideface=6

Output categories returned by :meth:`YOLO_Status.determin`::

    0 = normal, 1 = close eye, 2 = yawn, 3 = calling, 4 = turning around

The internal ``status_prior`` levels and the ``condition`` lookup implement a
priority scheme: a frame is assigned the highest-priority behavior it matches, and
``condition`` then maps that priority level back to the output category above.
"""


# write by llr
# transform xyxy loacationn to xywh loacation, scale in (0, 1)
def xyxy2xywh_normalized(
    xmin: int, ymin: int, xmax: int, ymax: int, wide: int, height: int
) -> tuple:  # Fixed: renamed to avoid shadowing utils.general.xyxy2xywh
    """
    tranform xyxy location to xywh location

    :param xmin: xmin
    :param ymin: ymin
    :param xmax: xmax
    :param ymax: ymax
    :param wide: wide
    :param height: height
    :return: tuple(x,y,w,h)
    """
    x = ((xmin + xmax) // 2) / wide
    y = ((ymin + ymax) // 2) / height
    w = (xmax - xmin) / wide
    h = (ymax - ymin) / height
    return x, y, w, h


class YOLO_Status:
    def __init__(self):
        self.cls_ = {
            "close_eye": 0,
            "close_mouth": 1,
            "face": 2,
            "open_eye": 3,
            "open_mouth": 4,
            "phone": 5,
            "sideface": 6,
        }
        self.status_prior = {"normal": 0, "closeeye": 1, "yawn": 3, "calling": 4, "turning": 2}
        self.condition = [0, 1, 4, 2, 3]

    def determin(self, img, dets) -> int:
        """
        determin which status this frame belongs to\n
        0 -> normal status\n
        1 -> close eye\n
        2 -> yawn\n
        3 -> calling\n
        4 -> turning around\n

        :param img: input image, format the same as detect function
        :param dets: to detect boxes
        :returns: an int status symbol
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
                if 0.5 < xywh[0] and xywh[1] > driver[1] and conf > 0.4:
                    # box中心在右侧0.5 并且 在司机下侧
                    driver = xywh  # 替换司机
                    driver_xyxy = xyxy
                    driver_conf = conf
                    face_flag = True
            elif cls == self.cls_["sideface"]:  # 侧脸
                if 0.5 < xywh[0] and xywh[1] > sideface[1] and conf > 0.4:  # box位置，与face一致
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
                if conf > 0.4:
                    eyes.append((cls, xywh, conf))
            elif cls == self.cls_["open_mouth"] or cls == self.cls_["close_mouth"]:  # 嘴，先存着
                if conf > 0.4:
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
