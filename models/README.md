# Local ONNX models

The app does not upload images or download models on startup. All model files, shapes, color orders, normalization parameters, SHA-256 checksums, and upstream licenses are defined canonically in [manifest.json](manifest.json).

## Roles in `models/manifest.json`

| Role | ID | Filename | Purpose |
| --- | --- | --- | --- |
| `face_detector` | `anime_face_yolov3` | `anime_face_yolov3.onnx` | Anime/illustration face bounding boxes |
| `landmark_28` | `anime_face_hrnetv2` | `anime_face_hrnetv2.onnx` | 28-point facial landmarks (mapped to 5 rig landmarks; `crown` is geometrically estimated) |
| `background` | `u2netp` | `u2netp.onnx` | Foreground cutout / background removal (`[1,3,320,320]` RGB NCHW, ImageNet norm) |
| `part_encoder` | `mobilesam_encoder` | `mobilesam_encoder.onnx` | MobileSAM image encoder for interactive part segmentation |
| `part_decoder` | `mobilesam_decoder` | `mobilesam_decoder.onnx` | MobileSAM prompt decoder for candidate masks and click refinement |

## Setup

```powershell
# 1) Download U2NetP background model (4.7 MB, MD5/SHA-256 verified)
powershell -ExecutionPolicy Bypass -File scripts/download-model.ps1

# 2) Export YOLOv3, HRNetV2, and MobileSAM encoder/decoder into models/
python tools/model-export/export_models.py
```

You can also link local `.onnx` files or download `U2NetP` inside the app's **ONNX 모델 관리** modal, where each role's SHA-256 status is verified before inference. Model files are excluded from Git and are not redistributed with the repository; review upstream distribution terms (`MIT`, `Apache-2.0`) before redistributing weights.

## Hidden-area expansion note

No generative inpainting ONNX model is currently bundled in `models/manifest.json`. Behind-occlusion regions (`fillMask`) and extended part textures are built non-generatively in Rust (`src-tauri/src/analysis/templates.rs`, `src-tauri/src/core.rs`) via LD/SD canonical silhouette fitting and eroded local color fields, and complex occluded patterns or lineart remain estimated regions that users can override via manual mask/PNG editing.
