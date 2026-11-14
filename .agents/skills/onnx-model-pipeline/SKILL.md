---
name: onnx-model-pipeline
description: >-
  Use when adding, exporting, updating, or debugging ONNX vision models
  (face detector, landmarks, background matting, MobileSAM segmentation, or optional inpainting)
  in flat2rig, including models/manifest.json and Rust ort parity verification.
---

# ONNX Model Pipeline (Python Export -> Manifest -> Rust `ort` Parity)

`flat2rig`는 export·검증에만 Python(`tools/model-export/`, `tools/evaluation/`)을 사용하고, 실제 앱 실행은 Rust `ort`(`src-tauri/src/analysis/`)로 유지한다.

## Steps

1. **Confirm contract in `models/manifest.json` first**
   - Check `opsetVersion`, `maxIrVersion`, `role`, `input` (`name`, `shape`, `colorOrder`, `resize`, `normalize`), `outputs`, and `postprocessVersion` in [models/manifest.json](../../models/manifest.json).
   - Never hardcode normalization constants or input dimensions in docs without keeping `manifest.json` as the source of truth.

2. **Export and verify in Python (`tools/model-export/`)**
   - Use the pinned environment (`tools/model-export/.venv/Scripts/python.exe`) and [export_models.py](../../tools/model-export/export_models.py).
   - Verify CPU ONNX Runtime execution, finite outputs, opset/IR compatibility with `ort` 2 rc10, and numerical parity against PyTorch reference before touching Rust.
   - Update `models/manifest.json` with exact `sha256`, `bytes`, `upstreamRepo`/`Revision`, `license`, and `postprocessVersion`.

3. **Implement Rust inference (`src-tauri/src/analysis/`)**
   - Register the role in [models.rs](../../src-tauri/src/analysis/models.rs) (`ModelRole` + embedded `manifest.json` lookup).
   - Reuse [transforms.rs](../../src-tauri/src/analysis/transforms.rs) (`glam` affine transforms, bilinear sampling, letterbox/padding) so crop -> model -> work -> original coordinate roundtrips match Python.
   - Keep `inference` inside `spawn_blocking` / `AnalysisState` job lock with cancellation & revision checks (`src-tauri/src/lib.rs`).
   - Do not mark a model pipeline as ready/complete if weights are missing or parity is unverified; keep explicit `model-missing` / manual fallback states.

4. **Verify Rust vs Python parity and regression tests**
   - Run Rust unit tests (automatically skips missing weights if absent, verifies parity when models exist):
     `cargo test --manifest-path src-tauri/Cargo.toml --lib`
   - Run Clippy with warnings denied:
     `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings`
   - Build CLI and run end-to-end parity & overlay evaluation when models are present:
     `cargo build --manifest-path src-tauri/Cargo.toml --bin flat2rig-cli`
     `tools/model-export/.venv/Scripts/python.exe tools/evaluation/evaluate.py`
   - Inspect generated overlays and `report.json` under `artifacts/eval/` before updating `docs/implementation-plan.md` or claiming accuracy targets.
