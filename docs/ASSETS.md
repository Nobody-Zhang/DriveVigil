# Large Binary Assets

This repository uses [GitHub Releases](https://github.com/Nobody-Zhang/DriveVigil/releases/tag/v1.0) to host large binary files that are not tracked in git.

## Quick Start

```bash
bash scripts/download_assets.sh
```

The script downloads all assets to their correct locations, skipping files that already exist.

## Asset Inventory

### Model Weights (186 MB)

| File | Size | Location |
|------|------|----------|
| `shape_predictor_68_face_landmarks.dat` | 95 MB | `edge/cloud_finetune/yolov5/` |
| `yolov5s.pt` | 14 MB | `edge/ota/` |
| `best (1).pt` | 4 MB | `edge/ota/` |
| `ONet.pt` | 1.5 MB | `edge/ota/mtcnn_landmarks/infer_models/` |
| `yolov5s_best.onnx` | 27 MB | `edge/deepstream/` |
| `best.onnx` | 27 MB | `edge/ota/` |
| `mixed_n.onnx` | 7 MB | `edge/deepstream/` |
| `yolov5n_best.onnx` | 7 MB | `edge/deepstream/` |
| `best.bin` (OpenVINO) | 27 MB | `cloud/preliminary/yolo/fine_tune_openvino_model/` |
| `best.bin` (OpenVINO) | 27 MB | `cloud/semifinal/yolo/fine_tune_openvino_model/` |

### TensorRT Engines (88 MB)

> These are platform-specific to Jetson TX2 NX. If your GPU architecture differs, regenerate from ONNX.

| File | Size | Location |
|------|------|----------|
| `v5s_mixes.engine` | 35 MB | `edge/deepstream/` |
| `model_b1_gpu0_fp32_1.engine` | 13 MB | `edge/deepstream/` |
| `model_b1_gpu0_fp32.engine` | 13 MB | `edge/deepstream/` |
| `zgb_n.engine` | 13 MB | `edge/deepstream/` |
| `model_b1_gpu0_v5n_mixed.engine` | 13 MB | `edge/deepstream/` |

### Python Wheels (77 MB unique, 4 copies each)

| File | Size | Locations |
|------|------|-----------
| `scikit_learn-1.0.2-*.whl` | 24 MB | cloud/preliminary, cloud/semifinal, edge/ota/mtcnn, edge/mtcnn |
| `scikit_image-0.19.3-*.whl` | 13 MB | cloud/preliminary, cloud/semifinal, edge/ota/mtcnn, edge/mtcnn |

### Videos (26 MB)

| File | Size | Location(s) |
|------|------|-------------|
| `SmartRecord_00000_20231031-043454_8868.mp4` | 4 MB | `edge/deepstream/` |
| `SmartRecord_00000_20231031-072426_8890.mp4` | 4 MB | `edge/deepstream/` |
| `SmartRecord_00000_20231028-065356_8625.mp4` | 4 MB | `edge/deepstream/` |
| `SmartRecord_00001_20230830-080226_2871.mp4` | 4 MB | `edge/ota/videos/`, `edge/apigw/videos/` |
| `SmartRecord_00000_20230830-080212_2871.mp4` | 3 MB | `edge/ota/videos/`, `edge/apigw/videos/` |
| `day_man_053_31_1.mp4` | 1 MB | `edge/ota/videos/`, `edge/apigw/videos/` |
| `1.mp4` | 1 MB | `edge/ota/mtcnn_landmarks/test/` |

### Audio (36 MB)

| File | Size | Location |
|------|------|----------|
| `NWGYU.wav` | 36 MB | `edge/deepstream/vuertsp-master/src/assets/audio/` |

### Archives (103 MB)

| File | Size | Location |
|------|------|----------|
| `vuertsp-master.zip` | 68 MB | `edge/deepstream/` |
| `vue.zip` | 34 MB | `edge/deepstream/vuertsp-master/` |

### Documents (70 MB)

| File | Size | Location |
|------|------|----------|
| `presentation.pptx` | 66 MB | `docs/` |
| `technical_report.pdf` | 4 MB | `docs/` |

### Other (9 MB)

| File | Size | Location |
|------|------|----------|
| `certificate.png` | 3 MB | `docs/` |
| `tmp.npy` | 6 MB | `edge/deepstream/` |
| `chunk-vendors.*.js.map` | 1 MB x2 | `edge/frontend/js/` |

## Total

- **40 tracked files** removed from git
- **~595 MB** total (with duplicates), **~380 MB** unique downloads
