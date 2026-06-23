"""Behavioral spec for the per-frame driver-status classifier (``YOLO_Status``).

These tests pin down the documented contract that the divide-and-conquer
temporal localizer depends on:

* the 7 YOLO class indices are in **alphabetical order** (what the trained model
  emits -- any retraining must keep this ordering), and
* ``determin()`` maps a single frame's detections to one of five categories::

      0 = normal, 1 = close eye, 2 = yawn, 3 = calling, 4 = turning

They use only synthetic detection boxes -- no GPU, model weights, or cloud.
"""

import types

import pytest
from status import YOLO_Status

WIDE = HEIGHT = 1000


def img(w=WIDE, h=HEIGHT):
    """Stand-in for an OpenCV frame: ``determin()`` only reads ``.shape``."""
    return types.SimpleNamespace(shape=(h, w))


def box(cls, x1, y1, x2, y2, conf=0.9):
    """Build one detection row: ``[xmin, ymin, xmax, ymax, conf, class]``."""
    return [x1, y1, x2, y2, conf, cls]


# A frontal face whose center sits in the lower-right -> selected as the driver.
DRIVER_FACE = box(2, 600, 400, 800, 700)


@pytest.fixture
def clf():
    return YOLO_Status()


def test_class_index_ordering_is_alphabetical(clf):
    assert clf.cls_ == {
        "close_eye": 0,
        "close_mouth": 1,
        "face": 2,
        "open_eye": 3,
        "open_mouth": 4,
        "phone": 5,
        "sideface": 6,
    }


def test_priority_to_category_mapping(clf):
    # `condition` maps an internal priority level to the public category:
    # normal(0)->0, closeeye(1)->1, turning(2)->4, yawn(3)->2, calling(4)->3.
    assert clf.status_prior == {"normal": 0, "closeeye": 1, "yawn": 3, "calling": 4, "turning": 2}
    assert clf.condition == [0, 1, 4, 2, 3]


def test_no_detections_is_turning(clf):
    assert clf.determin(img(), []) == 4


def test_no_face_is_turning(clf):
    # Eyes/mouth present but no face box -> driver not found -> turning.
    dets = [box(3, 650, 450, 710, 490), box(1, 650, 600, 720, 650)]
    assert clf.determin(img(), dets) == 4


def test_open_eye_outranks_closed_eye_is_normal(clf):
    # Same driver, both eyes detected near each other; the more-confident OPEN
    # eye wins -> normal. Load-bearing: with the open_eye removed, the lone
    # close_eye yields category 1 (so this is not just re-asserting the default).
    dets = [
        DRIVER_FACE,
        box(0, 640, 450, 690, 490, conf=0.5),  # close_eye, lower confidence
        box(3, 700, 450, 750, 490, conf=0.9),  # open_eye, higher confidence
        box(1, 650, 600, 720, 650),  # close_mouth inside the face
    ]
    assert clf.determin(img(), dets) == 0


def test_closed_eye_is_closeeye(clf):
    dets = [
        DRIVER_FACE,
        box(0, 650, 450, 710, 490),  # close_eye inside the face
        box(1, 650, 600, 720, 650),  # close_mouth
    ]
    assert clf.determin(img(), dets) == 1


def test_open_mouth_is_yawn(clf):
    dets = [
        DRIVER_FACE,
        box(3, 650, 450, 710, 490),  # open_eye (eyes look normal)
        box(4, 650, 600, 720, 650),  # open_mouth inside the face -> yawn
    ]
    assert clf.determin(img(), dets) == 2


def test_phone_near_face_is_calling(clf):
    dets = [
        DRIVER_FACE,
        box(5, 700, 450, 800, 620),  # phone overlapping the face region
    ]
    assert clf.determin(img(), dets) == 3


def test_sideface_right_of_frontal_face_is_turning(clf):
    dets = [
        box(2, 500, 400, 700, 700),  # frontal face (driver), center x = 0.6
        box(6, 750, 400, 900, 700),  # sideface further right -> turning
    ]
    assert clf.determin(img(), dets) == 4


def test_calling_beats_normal_eyes(clf):
    # A phone near the face (calling, priority 4) wins over open eyes (normal).
    dets = [
        DRIVER_FACE,
        box(3, 650, 450, 710, 490),  # open_eye
        box(5, 700, 450, 800, 620),  # phone
    ]
    assert clf.determin(img(), dets) == 3
