# edge/deepstream — competition file map

> The upstream **DeepStream-Yolo** documentation is in [README.md](README.md).
> This companion file maps the competition-specific files: which are real entry
> points and which are one-off experiment scripts.

This directory is a **vendored fork of
[marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)**
adapted for the competition's 7-class YOLOv5 model, **plus many one-off
experiment scripts** written during the competition.

Because it is hardware-bound (NVIDIA DeepStream + TensorRT + GStreamer, Jetson
only) and largely archival, the whole `edge/deepstream/` tree is **excluded from
the repo's ruff lint/format gate** (see `pyproject.toml`). Treat the scratch
scripts below as historical reference, not as a foundation to build on.

Pipeline at a glance: camera `/dev/video1` → TensorRT YOLOv5 detect → RTSP out
(`rtsp://localhost:8554/ds-test`); `app.py` re-serves that stream as MJPEG at
`/video_feed` for the Vue frontend in `edge/frontend/`.

## Build (Jetson)

```bash
# Build the custom CUDA YOLO TensorRT output parser
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo

# Build and run the C/GStreamer app
make
deepstream-app -c deepstream_app_config.txt
```

## Real entry points

| File | What it does |
| ---- | ------------ |
| `deepstream_yolov5.py` | Main Python DeepStream YOLOv5 pipeline → RTSP out. |
| `deepstream_yolov5_show.py` | Same pipeline with an on-screen display sink. |
| `camera_in_rtsp_out.py`, `camera_input_rtsp_output.py` | Camera → TensorRT → RTSP bridges. |
| `deepstream_test1_rtsp_out.py`, `deepstream_test_1_usb.py` | DeepStream-sample-derived variants (RTSP / USB camera). |
| `app.py` | Small Flask app: re-serves the RTSP stream as MJPEG at `/video_feed`. |

### C / GStreamer application

`deepstream_app_main.c`, `deepstream_app.c` / `.h`,
`deepstream_app_config_parser.c`, `csi_input_rtsp_output.c`, `deepstream_utc.c`,
`deepstream_test5_app_main.c` / `.h`, and the `Makefile`, with the custom parser
in `nvdsinfer_custom_impl_Yolo/`.

### Configuration

| File | Purpose |
| ---- | ------- |
| `config_infer_primary_yoloV5.txt` | Inference config: model/engine path, 7 classes, NMS thresholds. |
| `deepstream_app_config.txt` | DeepStream application config (sources, sinks, OSD). |
| `labels.txt` | Class labels. |
| `ota_override_config.txt` | Config overrides applied after an OTA model update. |

## Competition scratch / experiments (not maintained)

Kept for historical reference only:

- **Python:** `test.py`, `test0.py`, `test1.py`, `gpt_test.py`, `gptfor.py`,
  `trysocket.py`, `trysocket2.py`, `trysocket3queue.py`, `trysocket4both.py`,
  `trysocket4int.py`, `trysocket4smart_record.py`, `trynew.py`, `trysend.py`,
  `and.py`, `send_int.py`, `make_element.py`, `MP4_in.py`, `MP4_in_gsm.py`,
  `mp4_directly_rtsp.py`, `camera_in.py`, `cloudinfer.py`
- **C++:** `test.cpp`, `test_97.cpp`, `test_bk93.cpp`, `test_origin.cpp`
- **Stray logs:** `ereror.txt`, `failed.txt`

## Vendored / third-party subdirectories

`nvdsinfer_custom_impl_Yolo/` (custom TensorRT parser), `utils/` (DeepStream-Yolo
helpers), `apig_sdk/` (Huawei API Gateway SDK), `vuertsp-master/` (Vue RTSP
player), `templates/`, `licenses/`, `docs/`.
