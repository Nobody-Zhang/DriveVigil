# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
loosely follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Open-source hardening of the original competition code — no changes to the
detection algorithm or leaderboard behavior.

### Added

- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, a pull-request template, an issue-template chooser config, and
  a `CITATION.cff` (enables GitHub's "Cite this repository" widget).
- Automated test suite (`tests/`, pytest) covering the pure per-frame classifier
  `YOLO_Status` and its geometry helpers — runs with no GPU, cloud, or assets.
- `requirements-dev.txt` for the test/dev toolchain.
- `edge/deepstream/COMPETITION.md` distinguishing real entry points from
  competition scratch scripts.
- This changelog.

### Changed

- Extracted the core per-frame classifier `YOLO_Status` and the
  `xyxy2xywh_normalized` geometry helper from
  `cloud/preliminary/yolo/yolo_divide_and_conquer.py` into a dependency-light
  `cloud/preliminary/yolo/status.py`, so the core IP can be imported and unit
  tested without `torch`/`cv2`/OpenVINO. Behavior is unchanged.
- CI (`.github/workflows/lint.yml`) now runs the test suite in addition to the
  ruff lint/format gate.

### Earlier open-source preparation (already in history)

- Rewrote `README.md` (English) and added `README_zh.md` (Chinese).
- Added ruff configuration, a lint CI workflow, and pre-commit hooks; formatted
  all hand-maintained files.
- Replaced all `print()` calls with the `logging` module.
- Added type hints and Google-style docstrings to core files.
- Moved all credentials to `.env` (`.env.example` template) and added a
  comprehensive `.gitignore`.

## [1.0.0] - 2023

Original competition release.

- Cloud: YOLOv5 + OpenVINO divide-and-conquer temporal localizer on Huawei Cloud
  ModelArts. Preliminary score **0.9741**, semi-final score 0.8807.
- Edge (Jetson TX2 NX): DeepStream + TensorRT real-time pipeline, OTA model
  update loop, voice assistant (ASR + LLM + TTS), and a C++ watchdog daemon.
- **Second Prize**, 18th Challenge Cup National College Students' Extracurricular
  Academic Science and Technology Works Competition — Huawei Cloud Track.
- Large binary assets published under [Releases v1.0](../../releases).

[Unreleased]: https://github.com/Nobody-Zhang/DriveVigil/compare/v1.0...HEAD
[1.0.0]: https://github.com/Nobody-Zhang/DriveVigil/releases/tag/v1.0
