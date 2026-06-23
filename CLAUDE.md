# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Award-winning **fatigue/distracted-driving detection** system built for the Huawei Cloud track of the 18th Challenge Cup. It is a **cloud-edge collaboration**: heavy video analysis runs on Huawei Cloud ModelArts (`cloud/`), while real-time inference, model OTA, voice interaction, and monitoring run on an NVIDIA Jetson TX2 NX (`edge/`). This is competition code, not a maintained product — expect leaderboard-tuned heuristics and, under `edge/deepstream/`, many one-off experiment scripts.

## First-time setup (required before anything runs)

```bash
bash scripts/setup_env.sh         # creates .venv, installs requirements.txt, copies .env.example -> .env
bash scripts/download_assets.sh   # REQUIRED: pulls ~40 large binaries (weights, TensorRT engines,
                                  # OpenVINO IR, videos, wheels) from GitHub Releases v1.0 to correct paths
```

Large binaries are **not in git** — they live in GitHub Releases v1.0 (see `docs/ASSETS.md`). Nothing (cloud inference, edge pipeline, OTA) works until `download_assets.sh` has run. Then fill credentials into `.env` (see below).

## Common commands

```bash
# Lint / format (CI gate — .github/workflows/lint.yml; ruff pinned to 0.15.17)
ruff check .            # or: ruff check --fix .
ruff format --check .   # or: ruff format .
pre-commit run --all-files   # ruff + ruff-format, pinned v0.15.17

# Tests (CI gate). Pure, hardware-free logic — needs no GPU, cloud, or assets.
pip install -r requirements-dev.txt
pytest                  # only collects tests/ (see [tool.pytest.ini_options])

# Some modules also self-validate via __main__ blocks against sample videos from
# download_assets.sh, e.g. (needs the OpenVINO model + a video):
cd cloud/preliminary/yolo && python yolo_divide_and_conquer.py

# Edge — DeepStream (on Jetson only)
CUDA_VER=10.2 make -C edge/deepstream/nvdsinfer_custom_impl_Yolo   # build custom YOLO TensorRT parser
cd edge/deepstream && make && deepstream-app -c deepstream_app_config.txt

# Edge — Watchdog daemon (needs OpenCV)
cd edge/watchdog && mkdir build && cd build && cmake .. && make    # -> ./WatchDog

# Edge — OTA retraining loop / voice assistant (need .env credentials)
cd edge/ota && python main.py
cd edge/voice/scripts && python recognize_generate4.py
```

`test_signer.py` (apigw/ota) and `edge/deepstream/gpt_test.py` are SDK demos / experiments, not unit tests.

## Architecture (the parts that span multiple files)

### Cloud: ModelArts "custom AI application" pattern (`cloud/`)
Each cloud variant is a deployable ModelArts app defined by a **two-file contract**:
- `config.json` — declares runtime, the HTTP API schema, and pip dependencies (including bundled `.whl` files).
- `customize_service.py` — a `PTServingBaseService` subclass with `_preprocess` → `_inference` → `_postprocess`. It is deliberately **thin**: it writes the uploaded video to a temp file and delegates *all* logic to `yolo/yolo_divide_and_conquer.py::yolo_run(source=...)`. Model loading happens inside `yolo_run`, not in `__init__` — that is why inference timing starts at "Run inference".

Three variants:
- `cloud/baseline/` — dlib EAR/MAR (eye/mouth aspect ratio) approach; different lineage from the YOLO variants.
- `cloud/preliminary/` — **best score 0.9741**, the canonical implementation.
- `cloud/semifinal/` — score 0.8807; same divide-and-conquer algorithm as preliminary, only thresholds differ.

> **Cloud runs Python 3.7** (runtime `pytorch_1.8.0-cuda_10.2-py_3.7-ubuntu_18.04-x86_64`), even though the repo's ruff/CI target 3.8. Do not introduce 3.8+ syntax into code that deploys to ModelArts.

### The divide-and-conquer temporal localizer (core IP — `cloud/preliminary/yolo/yolo_divide_and_conquer.py`)
Finds time periods (≥3s) where the driver shows a fatigue/distraction behavior **without running detection on every frame**:
- `YOLO_Status.determin(img, dets)` classifies a *single* frame. It lives in the dependency-light `cloud/preliminary/yolo/status.py` (no torch/cv2/OpenVINO imports) and is imported back into `yolo_divide_and_conquer.py`; this split is what lets the `tests/` suite exercise it without hardware. The 7 YOLO classes are in **alphabetical index order**: `close_eye=0, close_mouth=1, face=2, open_eye=3, open_mouth=4, phone=5, sideface=6` — this ordering is what the trained model emits, keep it in sync with any retraining. Geometric heuristics select the driver among multiple people (rightmost/lowest face; eyes/mouth must fall inside the face bbox; phone near face ⇒ calling). It returns a category through the priority indirection `status_prior` → `condition=[0,1,4,2,3]`: **0=normal, 1=closeeye, 2=yawn, 3=calling, 4=turning**. (The semifinal copy keeps its own `YOLO_Status` inline.)
- `f(frame_idx)` runs the model on one frame, **memoized** in `sta_tmp` (each frame inferred at most once).
- `divide_and_conquer(lo, hi)` recursively bisects the frame range; at each midpoint it probes outward in 0.375s steps to measure how long the state persists, recording candidates (`tmp`) when a state lasts long enough.
- `b_search(...)` refines each candidate's exact start/end boundary to a precision set by `iou_presice_b_search` (0.05 = accuracy-first).
- **Output** (the code is authoritative; the per-folder READMEs show an older single-`category` contract): `{"result": {"drowsy": [{"periods": [start_ms, end_ms], "category": k}, ...], "duration": infer_ms}}`.
- Weights load as **OpenVINO IR** (`fine_tune_openvino_model/best.xml` + `.bin`) via vendored YOLOv5's `DetectMultiBackend`.

### Cloud↔Edge OTA loop (`edge/ota/main.py`)
Closes the data loop: edge records videos → once ≥3 accumulate (polls `smart_record.log`) → uploads weights + videos to OBS → POSTs a ModelArts training job (AK/SK-signed via `apig_sdk.signer`, free GPU flavor) → polls until complete → downloads retrained `best.pt`/`best.onnx` back to the edge → clears the OBS dataset. Bucket/paths are hardcoded to the team's OBS; credentials come from env.

### Edge real-time pipeline (`edge/deepstream/`)
Vendored fork of marcoslucianops/DeepStream-Yolo. C/GStreamer app + a custom CUDA YOLO parser in `nvdsinfer_custom_impl_Yolo/`. `config_infer_primary_yoloV5.txt` points at the ONNX/engine (7 classes, NMS thresholds). Flow: camera `/dev/video1` → TensorRT detect → RTSP out (`rtsp://localhost:8554/ds-test`). `app.py` is a small Flask MJPEG bridge that re-serves that RTSP at `/video_feed` for the Vue frontend (`edge/frontend/`, a prebuilt SPA). The many `test*.py`, `try*.py`, `gpt*.py` files here are competition scratch — the real entry points are the `camera_*_rtsp_*.py` / `deepstream_yolov5*.py` scripts and the C app.

### Edge voice assistant (`edge/voice/scripts/recognize_generate4.py`)
Three stages: Huawei SIS **ASR** (websocket) → text generation → Huawei SIS **TTS**. The generation backend is switchable between a self-hosted LLaMA and DashScope/Qwen (通义千问, `DASHSCOPE_API_KEY`). Specific safety warnings are served from pre-recorded clips in `edge/voice/audio_bags/*.wav`; general queries go to the LLM. Coordinates with a record-controller socket on `localhost:5333`.

### Edge watchdog (`edge/watchdog/monitoring_threads.cpp`)
C++ OpenCV + pthreads daemon. Three threads watch the camera (`/dev/video1`), the network (ping `huaweicloud.com`), and the capture loop; any failure flips a shared atomic + `condition_variable` to stop everything.

## Conventions

- **Secrets** never go in code. All credentials/endpoints are read via `os.environ.get` from `.env` (template: `.env.example`): `HUAWEICLOUD_AK`/`SK`, `HUAWEICLOUD_IMA_ID`, `HUAWEICLOUD_PROJECT_ID`, `HUAWEICLOUD_REGION`, the `HUAWEICLOUD_*_URL` endpoints, and `DASHSCOPE_API_KEY`.
- **Logging, not print.** Use the `logging` module (a refactor removed all `print()`s). Shared helper: `utils/logging_config.py::setup_logging()`. Pattern: entry scripts call `logging.basicConfig(...)`; modules use `logger = logging.getLogger(__name__)`.
- **Vendored / archival trees — do not edit or lint.** `pyproject.toml`'s ruff `exclude` lists them: every `**/yolo/models/`, `**/yolo/utils/`, and `**/yolo/export.py` (vendored Ultralytics YOLOv5), `edge/cloud_finetune/yolov5/`, `edge/voice/huaweicloud-python-sdk-sis-1.8.1/` (Huawei SIS SDK), every `apig_sdk/` (Huawei API Gateway SDK), the mtcnn `utils/`, `edge/frontend/`, and the **entire `edge/deepstream/`** tree (a vendored DeepStream-Yolo fork plus one-off competition experiment scripts — hardware-bound and archival; see `edge/deepstream/COMPETITION.md`). ruff also ignores `E402`/`F403`/`F405` repo-wide (the intentional `sys.path`-then-import and vendored star-import patterns). ~257 `.py` files are tracked but only ~60 are hand-maintained and linted.
- **Same code in two places.** `cloud/preliminary` and `cloud/semifinal` (and the mtcnn copies under `edge/`) are near-duplicates; a change to the algorithm usually needs mirroring across variants.
