# Local ONNX models

The app does not upload images or download models on startup. Run `powershell -ExecutionPolicy Bypass -File scripts/download-model.ps1` from the repository root to download the 4.7 MB U2NetP model, verify its upstream MD5, and print its SHA-256. Select the resulting `models/u2netp.onnx` in the desktop inspector.

Supported inference profile: RGB NCHW float32 `[1,3,320,320]`, ImageNet normalization, first output float32 `[1,1,320,320]`. U2Net and U2NetP use this profile. Arbitrary ONNX segmentation/landmark models are not interchangeable.

Source: [rembg U2NetP session](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/u2netp.py). The underlying [U-2-Net project](https://github.com/xuebinqin/U-2-Net) publishes an [Apache-2.0 license](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE). Model files are excluded from Git and are not redistributed with the app. Review the upstream distribution terms before redistribution.

This model removes backgrounds. Facial parts are proposed by editable geometry, not semantic AI parsing. No landmark or inpainting model is bundled.
