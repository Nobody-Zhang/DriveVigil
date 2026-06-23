"""Unit tests for the pure geometry helper in the cloud detector."""

import pytest
from status import xyxy2xywh_normalized


def test_full_frame_box_is_centered_and_unit_sized():
    # A box covering the whole frame -> center (0.5, 0.5), size (1.0, 1.0).
    x, y, w, h = xyxy2xywh_normalized(0, 0, 100, 200, 100, 200)
    assert (x, y, w, h) == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_top_left_quadrant_box():
    x, y, w, h = xyxy2xywh_normalized(0, 0, 500, 500, 1000, 1000)
    assert (x, y, w, h) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_offset_box_normalization():
    # The canonical driver-face box used throughout the status tests.
    x, y, w, h = xyxy2xywh_normalized(600, 400, 800, 700, 1000, 1000)
    assert (x, y, w, h) == pytest.approx((0.7, 0.55, 0.2, 0.3))


def test_odd_coordinate_sum_uses_floor_division():
    # The helper uses integer floor division: (0 + 101) // 2 == 50 (not 50.5),
    # so center-x is 50/1000 = 0.05. This pins the floor semantics.
    x, y, w, h = xyxy2xywh_normalized(0, 0, 101, 201, 1000, 1000)
    assert (x, y, w, h) == pytest.approx((0.05, 0.1, 0.101, 0.201))
