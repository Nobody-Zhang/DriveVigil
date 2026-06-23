"""Pytest configuration for the test suite.

``status.py`` (the pure per-frame classifier) lives next to the vendored YOLOv5
package inside ``cloud/preliminary/yolo``. We load it directly by file path and
register it as the top-level module ``status`` so the tests can ``import status``
**without** putting that directory on ``sys.path`` -- doing so would shadow the
repo's own top-level ``utils``/``models`` packages with the vendored YOLOv5
copies. ``status.py`` imports no torch/cv2/OpenVINO, so the tests run anywhere
with just pytest installed.
"""

import importlib.util
import sys
from pathlib import Path

_STATUS_PATH = Path(__file__).resolve().parent.parent / "cloud" / "preliminary" / "yolo" / "status.py"

if "status" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("status", _STATUS_PATH)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    sys.modules["status"] = _module
