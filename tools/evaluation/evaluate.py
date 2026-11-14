"""End-to-end Rust ONNX vs Python ONNX parity, overlay rendering, and CPU benchmark harness.

Outputs:
- artifacts/eval/report.json
- artifacts/eval/overlays/*_overlay.png
- artifacts/eval/inputs/* (synthetic crop / letterbox / multi-face / negative checks)

Run with:
  tools/model-export/.venv/Scripts/python.exe tools/evaluation/evaluate.py
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "models" / "manifest.json"
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "artifacts" / "eval"
INPUTS_DIR = EVAL_DIR / "inputs"
OVERLAYS_DIR = EVAL_DIR / "overlays"
RUST_OUT_DIR = EVAL_DIR / "rust"

sys.path.insert(0, str(ROOT / "tools" / "model-export"))
from anime_face_detector._face import _preprocess as det_preprocess  # noqa: E402
from anime_face_detector._face import load_face_detector  # noqa: E402
from anime_face_detector._landmark import BBOX_PADDING, IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD  # noqa: E402
from anime_face_detector._vendor.mmpose import bbox_xywh2cs, bbox_xyxy2xywh, get_affine_transform  # noqa: E402
from anime_face_detector.detector import get_checkpoint_path  # noqa: E402
from export_models import decode_hrnet_heatmaps, decode_yolov3_pred_maps  # noqa: E402


def resolve_cli_exe() -> Path:
    candidates = [
        ROOT / "src-tauri" / "target" / "release" / "flat2rig-cli.exe",
        ROOT / "src-tauri" / "target" / "debug" / "flat2rig-cli.exe",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "flat2rig-cli.exe not found. Run `cargo build --manifest-path src-tauri/Cargo.toml --release --bin flat2rig-cli` first."
    )


def get_peak_rss_mb_windows() -> float | None:
    if sys.platform != "win32":
        return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel32.GetCurrentProcess()
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    kernel32.K32GetProcessMemoryInfo.restype = ctypes.c_int
    if kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
    return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def build_eval_cases() -> list[dict[str, Any]]:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []

    p1 = ROOT / "sutteang" / "2d_.png"
    p2 = ROOT / "sutteang" / "KakaoTalk_20261114_132504352.png"
    if not p1.is_file() or not p2.is_file():
        raise FileNotFoundError("Expected local sutteang/ test images for evaluation")

    cases.append({"name": "sutteang_2d", "path": p1, "split": "initial_local_repro", "expectFaces": 1})
    cases.append({"name": "sutteang_cutout", "path": p2, "split": "initial_local_repro", "expectFaces": 1})

    img1 = Image.open(p1).convert("RGBA")
    img2 = Image.open(p2).convert("RGBA")

    # 1) Non-square portrait crop (768x1024)
    crop_portrait = img1.crop((128, 0, 896, 1024))
    p_portrait = INPUTS_DIR / "nonsquare_portrait.png"
    crop_portrait.save(p_portrait)
    cases.append({"name": "nonsquare_portrait", "path": p_portrait, "split": "synthetic_transform_check", "expectFaces": 1})

    # 2) Wide letterboxed canvas (1280x720) with shifted character
    wide = Image.new("RGBA", (1280, 720), (236, 238, 242, 255))
    wide.paste(img1.resize((720, 720), Image.Resampling.BILINEAR), (440, 0))
    p_wide = INPUTS_DIR / "letterbox_wide.png"
    wide.save(p_wide)
    cases.append({"name": "letterbox_wide", "path": p_wide, "split": "synthetic_transform_check", "expectFaces": 1})

    # 3) Multi-face side-by-side synthetic canvas (1408x768)
    multi = Image.new("RGBA", (1408, 768), (242, 244, 248, 255))
    multi.paste(img1.crop((128, 32, 832, 800)).resize((640, 704), Image.Resampling.BILINEAR), (32, 32))
    multi.paste(img2.crop((128, 32, 832, 800)).resize((640, 704), Image.Resampling.BILINEAR), (736, 32))
    p_multi = INPUTS_DIR / "multi_face_pair.png"
    multi.save(p_multi)
    cases.append({"name": "multi_face_pair", "path": p_multi, "split": "synthetic_transform_check", "expectFaces": 2})

    # 4) Negative / non-face canvas (512x512 neutral gradient)
    grad = np.zeros((512, 512, 4), dtype=np.uint8)
    xs = np.linspace(180, 235, 512, dtype=np.uint8)
    grad[:, :, 0] = xs[None, :]
    grad[:, :, 1] = xs[:, None]
    grad[:, :, 2] = 225
    grad[:, :, 3] = 255
    p_neg = INPUTS_DIR / "negative_blank.png"
    Image.fromarray(grad, "RGBA").save(p_neg)
    cases.append({"name": "negative_blank", "path": p_neg, "split": "synthetic_negative_check", "expectFaces": 0})

    return cases


def scale_box_11(box: np.ndarray) -> np.ndarray:
    cx, cy = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
    w, h = (box[2] - box[0] + 1.0) * 1.1, (box[3] - box[1] + 1.0) * 1.1
    return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)


def python_onnx_reference(
    det_helper: Any,
    yolo_sess: ort.InferenceSession,
    hrnet_sess: ort.InferenceSession,
    sam_enc: ort.InferenceSession,
    sam_dec: ort.InferenceSession,
    img_path: Path,
) -> dict[str, Any]:
    # Load RGB matching Rust `image::open().into_rgba8()` -> RGB bytes (preserves underlying RGB even where alpha=0)
    rgb = np.array(Image.open(img_path).convert("RGBA"))[:, :, :3]
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    t0 = time.perf_counter()
    tensor, img_meta = det_preprocess(img_bgr, det_helper.IMG_SCALE, det_helper.MEAN, det_helper.STD)
    raw_maps = yolo_sess.run(None, {"image": tensor.numpy()})
    boxes_raw = decode_yolov3_pred_maps(raw_maps, img_meta, det_helper)
    valid = [b for b in boxes_raw if float(b[4]) >= 0.5]
    py_det_ms = (time.perf_counter() - t0) * 1000.0

    if not valid:
        return {
            "numFaces": 0,
            "topBox": None,
            "points28": None,
            "samFaceIoU": None,
            "samFaceMask": None,
            "pyDetectMs": round(py_det_ms, 2),
            "pyLandmarkMs": 0.0,
            "pySamEncoderMs": 0.0,
            "pySamDecoderMs": 0.0,
        }

    top_box = [float(v) for v in valid[0]]
    cb = scale_box_11(np.array(top_box[:4], dtype=np.float32))
    box_arr = np.stack([cb])
    xywh = bbox_xyxy2xywh(box_arr)[0]
    center, scale = bbox_xywh2cs(xywh, aspect_ratio=IMAGE_SIZE[0] / IMAGE_SIZE[1], padding=BBOX_PADDING)

    t1 = time.perf_counter()
    trans = get_affine_transform(center, scale, 0, IMAGE_SIZE)
    crop = cv2.warpAffine(rgb, trans, IMAGE_SIZE, flags=cv2.INTER_LINEAR)
    crop_t = torch.from_numpy(crop.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(1, 3, 1, 1)
    crop_t = (crop_t - mean) / std
    hm_orig = hrnet_sess.run(None, {"image": crop_t.numpy()})[0]
    hm_flip = hrnet_sess.run(None, {"image": crop_t.flip(3).numpy()})[0]
    kpts = decode_hrnet_heatmaps(hm_orig, hm_flip, np.array([center]), np.array([scale]))[0]
    py_lm_ms = (time.perf_counter() - t1) * 1000.0

    # MobileSAM face prompt check
    t2 = time.perf_counter()
    sam_scale = 1024.0 / max(w, h)
    sam_w, sam_h = int(w * sam_scale + 0.5), int(h * sam_scale + 0.5)
    sam_resized = cv2.resize(rgb, (sam_w, sam_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    sam_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    sam_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    padded = np.zeros((1024, 1024, 3), dtype=np.float32)
    padded[:sam_h, :sam_w] = (sam_resized - sam_mean) / sam_std
    sam_in = padded.transpose(2, 0, 1)[None, ...]
    emb = sam_enc.run(None, {"image": sam_in})[0]
    py_enc_ms = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    sx, sy = sam_w / w, sam_h / h
    coords_sam = np.array(
        [[[top_box[0] * sx, top_box[1] * sy], [top_box[2] * sx, top_box[3] * sy]]],
        dtype=np.float32,
    )
    labels_sam = np.array([[2, 3]], dtype=np.float32)
    masks, ious, _ = sam_dec.run(
        None,
        {
            "image_embeddings": emb,
            "point_coords": coords_sam,
            "point_labels": labels_sam,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros((1,), dtype=np.float32),
            "orig_im_size": np.array([h, w], dtype=np.float32),
        },
    )
    py_dec_ms = (time.perf_counter() - t3) * 1000.0
    best_mask = masks[0, int(np.argmax(ious[0]))] > 0.0

    return {
        "numFaces": len(valid),
        "topBox": top_box,
        "points28": [[float(x), float(y)] for x, y, _ in kpts],
        "samFaceIoU": float(np.max(ious[0])),
        "samFaceMask": best_mask,
        "pyDetectMs": round(py_det_ms, 2),
        "pyLandmarkMs": round(py_lm_ms, 2),
        "pySamEncoderMs": round(py_enc_ms, 2),
        "pySamDecoderMs": round(py_dec_ms, 2),
    }


def render_overlay(
    img_path: Path,
    rust_res: dict[str, Any],
    py_res: dict[str, Any],
    out_path: Path,
) -> None:
    base = Image.open(img_path).convert("RGBA")
    if py_res.get("samFaceMask") is not None:
        mask = py_res["samFaceMask"]
        tint = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        tint[mask] = [36, 86, 214, 52]
        base = Image.alpha_composite(base, Image.fromarray(tint, "RGBA"))

    draw = ImageDraw.Draw(base)
    cands = rust_res["detection"]["candidates"]
    for cand in cands:
        x1, y1, x2, y2 = cand["bbox"]
        cx1, cy1, cx2, cy2 = cand["cropBbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(36, 86, 214, 255), width=3)
        draw.rectangle([cx1, cy1, cx2, cy2], outline=(24, 121, 78, 220), width=1)
        draw.text((x1 + 4, max(2, y1 - 14)), f"face #{cand['id']} score={cand['score']:.3f}", fill=(36, 86, 214, 255))

    lm = rust_res.get("landmarks")
    if lm:
        for p in lm["points28"]:
            px, py = p["x"], p["y"]
            draw.ellipse([px - 2.5, py - 2.5, px + 2.5, py + 2.5], fill=(201, 55, 44, 255))
        rig = lm["rigLandmarks"]
        prov = lm["rigProvenance"]
        for key, label in [
            ("eyeR", f"eyeR ({prov['eyeR']})"),
            ("eyeL", f"eyeL ({prov['eyeL']})"),
            ("mouth", f"mouth ({prov['mouth']})"),
            ("chin", f"chin ({prov['chin']})"),
            ("crown", f"crown ({prov['crown']})"),
        ]:
            pt = rig[key]
            px, py = pt["x"], pt["y"]
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], outline=(22, 25, 32, 255), fill=(255, 255, 255, 230), width=2)
            draw.text((px + 7, py - 6), label, fill=(22, 25, 32, 255))

    if not cands:
        draw.text((16, 16), "NO FACE DETECTED (>= 0.50)", fill=(201, 55, 44, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(out_path)


def main() -> None:
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    RUST_OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cli_exe = resolve_cli_exe()

    det_helper = load_face_detector("yolov3", get_checkpoint_path("yolov3"), device="cpu")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    yolo_sess = ort.InferenceSession(str(MODELS_DIR / "anime_face_yolov3.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    hrnet_sess = ort.InferenceSession(str(MODELS_DIR / "anime_face_hrnetv2.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    sam_enc = ort.InferenceSession(str(MODELS_DIR / "mobilesam_encoder.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    sam_dec = ort.InferenceSession(str(MODELS_DIR / "mobilesam_decoder.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])

    cases = build_eval_cases()
    results: list[dict[str, Any]] = []
    rust_detect_times: list[float] = []
    rust_landmark_times: list[float] = []
    rust_sam_full_times: list[float] = []
    rust_cli_wall_times: list[float] = []
    py_sam_enc_times: list[float] = []
    py_sam_dec_times: list[float] = []

    for idx, case in enumerate(cases):
        name = case["name"]
        img_path = case["path"]
        rust_json_path = RUST_OUT_DIR / f"{name}.json"

        repeats = 3 if idx == 0 else 1
        rust_res: dict[str, Any] = {}
        for r in range(repeats):
            t_wall = time.perf_counter()
            subprocess.run(
                [str(cli_exe), "--analyze", "--full", str(img_path), str(rust_json_path)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            wall_ms = (time.perf_counter() - t_wall) * 1000.0
            rust_res = json.loads(rust_json_path.read_text(encoding="utf-8"))
            if r > 0 or repeats == 1:
                rust_cli_wall_times.append(wall_ms)
                rust_detect_times.append(float(rust_res["detectMs"]))
                if rust_res.get("landmarkMs", 0.0) > 0.0:
                    rust_landmark_times.append(float(rust_res["landmarkMs"]))
                if rust_res.get("samFullMs"):
                    rust_sam_full_times.append(float(rust_res["samFullMs"]))

        py_res = python_onnx_reference(det_helper, yolo_sess, hrnet_sess, sam_enc, sam_dec, img_path)
        if py_res["pySamEncoderMs"] > 0:
            py_sam_enc_times.append(py_res["pySamEncoderMs"])
            py_sam_dec_times.append(py_res["pySamDecoderMs"])

        rust_cands = rust_res["detection"]["candidates"]
        bbox_max_diff = None
        lm28_max_diff = None
        lm28_mean_diff = None
        score_diff = None

        if rust_cands and py_res["topBox"] is not None:
            rb = rust_cands[0]["bbox"]
            pb = py_res["topBox"][:4]
            bbox_max_diff = round(max(abs(rb[i] - pb[i]) for i in range(4)), 4)
            score_diff = round(abs(rust_cands[0]["score"] - py_res["topBox"][4]), 6)

        if rust_res.get("landmarks") and py_res["points28"] is not None:
            rpts = rust_res["landmarks"]["points28"]
            ppts = py_res["points28"]
            dists = [math.hypot(rpts[i]["x"] - ppts[i][0], rpts[i]["y"] - ppts[i][1]) for i in range(28)]
            lm28_max_diff = round(max(dists), 4)
            lm28_mean_diff = round(statistics.mean(dists), 4)

        overlay_path = OVERLAYS_DIR / f"{name}_overlay.png"
        render_overlay(img_path, rust_res, py_res, overlay_path)

        part_sum = rust_res.get("partSummary") or None
        results.append({
            "name": name,
            "split": case["split"],
            "expectFaces": case["expectFaces"],
            "rustNumFaces": len(rust_cands),
            "pythonNumFaces": py_res["numFaces"],
            "rustDetectMs": round(float(rust_res["detectMs"]), 2),
            "rustLandmarkMs": round(float(rust_res["landmarkMs"]), 2),
            "rustSamFullMs": round(float(rust_res["samFullMs"]), 2) if rust_res.get("samFullMs") else None,
            "bboxMaxDiffPx": bbox_max_diff,
            "scoreDiff": score_diff,
            "landmark28MaxDiffPx": lm28_max_diff,
            "landmark28MeanDiffPx": lm28_mean_diff,
            "samFacePredIoU": round(py_res["samFaceIoU"], 4) if py_res["samFaceIoU"] is not None else None,
            "rustPartOverlapPixels": part_sum.get("overlapPixels") if part_sum else None,
            "rustTotalTriangles": part_sum.get("totalTriangles") if part_sum else None,
            "rustMeshWarnings": part_sum.get("meshWarnings") if part_sum else None,
            "overlay": str(overlay_path.relative_to(ROOT)),
        })

    report = {
        "schemaVersion": 1,
        "datasetScopeNotice": "Evaluated on 2 local sample illustrations (initial_local_repro) plus 4 deterministic synthetic transform/multi-face/negative checks. No held-out 20-image external dataset is bundled.",
        "environment": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "cpuCores": os.cpu_count(),
            "cliBinary": str(cli_exe.relative_to(ROOT)),
            "pythonProcessPeakRssMb": get_peak_rss_mb_windows(),
        },
        "modelsManifest": [
            {"role": m["role"], "id": m["id"], "sha256": m["sha256"], "postprocessVersion": m["postprocessVersion"]}
            for m in manifest["models"]
        ],
        "timingMs": {
            "rustDetectP50": round(percentile(rust_detect_times, 50), 2),
            "rustDetectP95": round(percentile(rust_detect_times, 95), 2),
            "rustLandmarkP50": round(percentile(rust_landmark_times, 50), 2),
            "rustLandmarkP95": round(percentile(rust_landmark_times, 95), 2),
            "rustSamFullP50": round(percentile(rust_sam_full_times, 50), 2),
            "rustCliWallP50": round(percentile(rust_cli_wall_times, 50), 2),
            "pythonSamEncoderP50": round(percentile(py_sam_enc_times, 50), 2),
            "pythonSamDecoderP50": round(percentile(py_sam_dec_times, 50), 2),
        },
        "cases": results,
    }

    report_path = EVAL_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
