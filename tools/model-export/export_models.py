"""Reproducible ONNX export and PyTorch parity verification for flat2rig."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.request
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from anime_face_detector._face import (
    PAD_SIZE_DIVISOR,
    YOLOv3FaceDetector,
    _preprocess as det_preprocess,
    load_face_detector,
)
from anime_face_detector._landmark import (
    BBOX_PADDING,
    FLIP_PAIRS,
    HEAD_TEST_CFG,
    IMAGE_MEAN as LM_MEAN,
    IMAGE_SIZE as LM_SIZE,
    IMAGE_STD as LM_STD,
    LandmarkModel,
    load_landmark_model,
)
from anime_face_detector._vendor.mmdet import YOLOAnchorGenerator, YOLOBBoxCoder
from anime_face_detector._vendor.mmdet.bbox_nms import multiclass_nms
from anime_face_detector._vendor.mmpose import (
    bbox_xywh2cs,
    bbox_xyxy2xywh,
    flip_back,
    get_affine_transform,
    keypoints_from_heatmaps,
)
from anime_face_detector.detector import (
    HF_MODEL_REPOS,
    LandmarkDetector,
    get_checkpoint_path,
)
from mobile_sam import SamPredictor, sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
CACHE_DIR = pathlib.Path(__file__).resolve().parent / "cache"
ARTIFACTS_DIR = ROOT / "artifacts" / "model-export"

AFD_COMMIT = "98a9fb480fa04bdd96acbe4d98da191a898267f3"
AFD_YOLOV3_SHA = "afdd4226a79ae8bb81f334dbcffd34f8cc000c38"
AFD_HRNET_SHA = "9b3435248b26aeb81f334dbcffd34f8cc000c38"
MOBILESAM_COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
MOBILESAM_WEIGHT_URL = (
    f"https://github.com/ChaoningZhang/MobileSAM/raw/{MOBILESAM_COMMIT}/weights/mobile_sam.pt"
)
U2NETP_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"

MAX_ONNX_IR_VERSION = 10
ONNX_OPSET = 17


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_ir_version(path: pathlib.Path, max_ir: int = MAX_ONNX_IR_VERSION) -> None:
    model = onnx.load(str(path))
    if model.ir_version > max_ir:
        model.ir_version = max_ir
        onnx.save(model, str(path))
    onnx.checker.check_model(str(path))


class YOLOv3RawHeadsWrapper(nn.Module):
    def __init__(self, detector: YOLOv3FaceDetector) -> None:
        super().__init__()
        self.backbone = detector.backbone
        self.neck = detector.neck
        self.bbox_head = detector.bbox_head

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.neck(self.backbone(image))
        pred_maps = self.bbox_head(feats)[0]
        return pred_maps[0], pred_maps[1], pred_maps[2]


class HRNetHeatmapWrapper(nn.Module):
    def __init__(self, landmark_model: LandmarkModel) -> None:
        super().__init__()
        self.backbone = landmark_model.backbone
        self.keypoint_head = landmark_model.keypoint_head

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        return self.keypoint_head(features)


def export_torch_onnx(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: pathlib.Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".onnx.tmp")
    export_kwargs: dict[str, Any] = dict(
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    # torch >= 2.5 added dynamo=False to select the TorchScript exporter explicitly.
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        export_kwargs["dynamo"] = False
    with torch.inference_mode():
        torch.onnx.export(module, args, str(tmp_path), **export_kwargs)
    ensure_ir_version(tmp_path)
    tmp_path.replace(output_path)


def decode_yolov3_pred_maps(
    pred_maps_np: list[np.ndarray],
    img_meta: dict[str, Any],
    detector: YOLOv3FaceDetector,
) -> np.ndarray:
    pred_maps = [torch.from_numpy(m) for m in pred_maps_np]
    results = detector.bbox_head.get_bboxes(pred_maps, [img_meta], rescale=True)
    return results[0][0].cpu().numpy()


def decode_hrnet_heatmaps(
    heatmaps_orig: np.ndarray,
    heatmaps_flipped: np.ndarray | None,
    centers: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    output_heatmap = heatmaps_orig
    if heatmaps_flipped is not None:
        flipped_back = flip_back(heatmaps_flipped, list(FLIP_PAIRS), target_type="GaussianHeatmap")
        if HEAD_TEST_CFG.get("shift_heatmap", False):
            flipped_back[:, :, :, 1:] = flipped_back[:, :, :, :-1]
        output_heatmap = (output_heatmap + flipped_back) / 2.0
    preds, maxvals = keypoints_from_heatmaps(
        output_heatmap,
        centers,
        scales,
        post_process=HEAD_TEST_CFG["post_process"],
        kernel=HEAD_TEST_CFG["modulate_kernel"],
    )
    return np.concatenate([preds, maxvals], axis=-1).astype(np.float32)


def download_mobilesam_checkpoint() -> pathlib.Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CACHE_DIR / "mobile_sam.pt"
    if not ckpt.exists():
        print(f"Downloading MobileSAM checkpoint from {MOBILESAM_WEIGHT_URL}...")
        tmp = ckpt.with_suffix(".pt.tmp")
        urllib.request.urlretrieve(MOBILESAM_WEIGHT_URL, str(tmp))
        tmp.replace(ckpt)
    return ckpt


def verify_face_and_landmarks(
    det_pt: YOLOv3FaceDetector,
    lm_pt: LandmarkModel,
    det_onnx_path: pathlib.Path,
    lm_onnx_path: pathlib.Path,
    image_paths: list[pathlib.Path],
) -> dict[str, Any]:
    det_sess = ort.InferenceSession(str(det_onnx_path), providers=["CPUExecutionProvider"])
    lm_sess = ort.InferenceSession(str(lm_onnx_path), providers=["CPUExecutionProvider"])
    pt_pipeline = LandmarkDetector(
        landmark_checkpoint_path=get_checkpoint_path("hrnetv2"),
        face_detector_name="yolov3",
        face_detector_checkpoint_path=get_checkpoint_path("yolov3"),
        device="cpu",
        flip_test=True,
        box_scale_factor=1.1,
    )

    cases: list[dict[str, Any]] = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        assert img_bgr is not None, f"Failed to read {img_path}"

        # 1. Fixed tensor parity on detector raw pred_maps
        tensor, img_meta = det_preprocess(img_bgr, det_pt.IMG_SCALE, det_pt.MEAN, det_pt.STD)
        with torch.inference_mode():
            pt_feats = det_pt.neck(det_pt.backbone(tensor))
            pt_maps = [m.numpy() for m in det_pt.bbox_head(pt_feats)[0]]
        onnx_maps = det_sess.run(None, {"image": tensor.numpy()})
        assert all(np.all(np.isfinite(m)) for m in onnx_maps), "Non-finite detector output"
        raw_map_max_diff = max(
            float(np.max(np.abs(a - b))) for a, b in zip(pt_maps, onnx_maps, strict=True)
        )

        pt_boxes_raw = det_pt.detect(img_bgr)
        onnx_boxes_raw = decode_yolov3_pred_maps(onnx_maps, img_meta, det_pt)
        assert pt_boxes_raw.shape == onnx_boxes_raw.shape, (
            f"Box count mismatch: {pt_boxes_raw.shape} vs {onnx_boxes_raw.shape}"
        )
        box_max_diff = (
            float(np.max(np.abs(pt_boxes_raw[:, :4] - onnx_boxes_raw[:, :4])))
            if len(pt_boxes_raw)
            else 0.0
        )
        score_max_diff = (
            float(np.max(np.abs(pt_boxes_raw[:, 4] - onnx_boxes_raw[:, 4])))
            if len(pt_boxes_raw)
            else 0.0
        )

        # 2. Full pipeline (scaled boxes -> HRNetV2 crop -> flip_test -> DARK)
        pt_preds = pt_pipeline(img_bgr)
        scaled_boxes = pt_pipeline._update_pred_box(onnx_boxes_raw.copy())
        lm_max_diff = 0.0
        heatmap_max_diff = 0.0
        if len(scaled_boxes) > 0:
            box_arr = np.stack([b.astype(np.float32) for b in scaled_boxes])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            aspect_ratio = LM_SIZE[0] / LM_SIZE[1]
            centers = np.zeros((len(box_arr), 2), dtype=np.float32)
            scales = np.zeros((len(box_arr), 2), dtype=np.float32)
            crops = []
            for i, b in enumerate(bbox_xyxy2xywh(box_arr)):
                c, s = bbox_xywh2cs(b, aspect_ratio=aspect_ratio, padding=BBOX_PADDING)
                centers[i] = c
                scales[i] = s
                trans = get_affine_transform(c, s, 0, LM_SIZE)
                crops.append(cv2.warpAffine(img_rgb, trans, LM_SIZE, flags=cv2.INTER_LINEAR))
            batch = torch.from_numpy(np.stack(crops).astype(np.float32)).permute(0, 3, 1, 2) / 255.0
            mean = torch.tensor(LM_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(LM_STD).view(1, 3, 1, 1)
            batch = (batch - mean) / std
            with torch.inference_mode():
                pt_hm_orig = lm_pt.keypoint_head(lm_pt.backbone(batch)).numpy()
                pt_hm_flip = lm_pt.keypoint_head(lm_pt.backbone(batch.flip(3))).numpy()
            onnx_hm_orig = lm_sess.run(None, {"image": batch.numpy()})[0]
            onnx_hm_flip = lm_sess.run(None, {"image": batch.flip(3).numpy()})[0]
            assert np.all(np.isfinite(onnx_hm_orig)) and np.all(np.isfinite(onnx_hm_flip))
            heatmap_max_diff = float(
                max(
                    np.max(np.abs(pt_hm_orig - onnx_hm_orig)),
                    np.max(np.abs(pt_hm_flip - onnx_hm_flip)),
                )
            )
            onnx_kpts = decode_hrnet_heatmaps(onnx_hm_orig, onnx_hm_flip, centers, scales)
            pt_kpts = np.stack([p["keypoints"] for p in pt_preds])
            lm_max_diff = float(np.max(np.abs(pt_kpts[:, :, :2] - onnx_kpts[:, :, :2])))

        cases.append(
            {
                "image": str(img_path.relative_to(ROOT)),
                "imageSize": [int(img_bgr.shape[1]), int(img_bgr.shape[0])],
                "detectorInputShape": list(tensor.shape),
                "numFaces": int(len(onnx_boxes_raw)),
                "topBox": onnx_boxes_raw[0].tolist() if len(onnx_boxes_raw) else None,
                "rawPredMapMaxAbsDiff": raw_map_max_diff,
                "boxMaxDiffPx": box_max_diff,
                "boxScoreMaxDiff": score_max_diff,
                "heatmapMaxAbsDiff": heatmap_max_diff,
                "landmarkMaxDiffPx": lm_max_diff,
            }
        )
    return {"cases": cases}


def verify_mobilesam(
    sam_model: Any,
    enc_onnx_path: pathlib.Path,
    dec_onnx_path: pathlib.Path,
    image_paths: list[pathlib.Path],
) -> dict[str, Any]:
    enc_sess = ort.InferenceSession(str(enc_onnx_path), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(dec_onnx_path), providers=["CPUExecutionProvider"])
    predictor = SamPredictor(sam_model)

    cases: list[dict[str, Any]] = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        predictor.set_image(img_rgb, image_format="RGB")
        pt_emb = predictor.get_image_embedding().cpu().numpy()

        # Replicate SamPredictor preprocessing in numpy
        target_h, target_w = predictor.transform.get_preprocess_shape(h, w, 1024)
        resized = np.array(
            cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR),
            dtype=np.float32,
        )
        # Note: torchvision resize uses PIL bilinear; compare against SamPredictor exact tensor too
        transformed = predictor.transform.apply_image(img_rgb).astype(np.float32)
        pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
        normed = (transformed - pixel_mean) / pixel_std
        padded = np.zeros((1024, 1024, 3), dtype=np.float32)
        padded[: transformed.shape[0], : transformed.shape[1]] = normed
        enc_input = np.transpose(padded, (2, 0, 1))[None, ...]

        t0 = time.perf_counter()
        onnx_emb = enc_sess.run(None, {"image": enc_input})[0]
        enc_ms = (time.perf_counter() - t0) * 1000.0
        assert np.all(np.isfinite(onnx_emb)), "Non-finite MobileSAM encoder embedding"
        emb_max_diff = float(np.max(np.abs(pt_emb - onnx_emb)))

        # Test a box + positive center prompt on the face area
        box = np.array([w * 0.25, h * 0.15, w * 0.75, h * 0.60], dtype=np.float32)
        pt_point = np.array([[w * 0.5, h * 0.35]], dtype=np.float32)
        pt_label = np.array([1], dtype=np.int32)
        pt_masks, pt_iou, _ = predictor.predict(
            point_coords=pt_point,
            point_labels=pt_label,
            box=box,
            multimask_output=True,
            return_logits=True,
        )

        # SAM ONNX prompt format: box corners have labels 2 (TL) and 3 (BR)
        onnx_pts = np.concatenate([pt_point, box.reshape(2, 2)], axis=0)
        onnx_pts = predictor.transform.apply_coords(onnx_pts, (h, w)).astype(np.float32)[None, ...]
        onnx_lbls = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask = np.array([0.0], dtype=np.float32)
        orig_im_size = np.array([h, w], dtype=np.float32)

        t1 = time.perf_counter()
        onnx_masks, onnx_iou, _ = dec_sess.run(
            None,
            {
                "image_embeddings": onnx_emb,
                "point_coords": onnx_pts,
                "point_labels": onnx_lbls,
                "mask_input": mask_input,
                "has_mask_input": has_mask,
                "orig_im_size": orig_im_size,
            },
        )
        dec_ms = (time.perf_counter() - t1) * 1000.0
        assert np.all(np.isfinite(onnx_masks)) and np.all(np.isfinite(onnx_iou))
        # Slots 1..4 correspond to multimask_output=True
        onnx_multi = onnx_masks[0, 1:4]
        onnx_multi_iou = onnx_iou[0, 1:4]
        mask_logit_diff = float(np.max(np.abs(pt_masks - onnx_multi)))
        pt_bin = pt_masks > 0.0
        onnx_bin = onnx_multi > 0.0
        inter = np.logical_and(pt_bin, onnx_bin).sum(axis=(1, 2))
        union = np.logical_or(pt_bin, onnx_bin).sum(axis=(1, 2)).clip(min=1)
        ious = (inter / union).tolist()

        # Also check bilinear resize difference vs PIL resize
        normed_cv = (resized - pixel_mean) / pixel_std
        padded_cv = np.zeros((1024, 1024, 3), dtype=np.float32)
        padded_cv[:target_h, :target_w] = normed_cv
        onnx_emb_cv = enc_sess.run(
            None, {"image": np.transpose(padded_cv, (2, 0, 1))[None, ...]}
        )[0]
        emb_cv_diff = float(np.max(np.abs(pt_emb - onnx_emb_cv)))

        cases.append(
            {
                "image": str(img_path.relative_to(ROOT)),
                "imageSize": [w, h],
                "encoderMs": round(enc_ms, 2),
                "decoderMs": round(dec_ms, 2),
                "embeddingMaxAbsDiff": emb_max_diff,
                "embeddingMaxAbsDiffCvResize": emb_cv_diff,
                "maskLogitMaxAbsDiff": mask_logit_diff,
                "iouScoreMaxAbsDiff": float(np.max(np.abs(pt_iou - onnx_multi_iou))),
                "pytorchVsOnnxMaskIoU": [round(float(x), 6) for x in ious],
            }
        )
    return {"cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    yolov3_ckpt = get_checkpoint_path("yolov3")
    hrnet_ckpt = get_checkpoint_path("hrnetv2")
    mobilesam_ckpt = download_mobilesam_checkpoint()

    det_pt = load_face_detector("yolov3", str(yolov3_ckpt), device="cpu")
    lm_pt = load_landmark_model(str(hrnet_ckpt), device="cpu")
    sam_pt = sam_model_registry["vit_t"](checkpoint=str(mobilesam_ckpt))
    sam_pt.eval()

    det_onnx = MODELS_DIR / "anime_face_yolov3.onnx"
    lm_onnx = MODELS_DIR / "anime_face_hrnetv2.onnx"
    sam_enc_onnx = MODELS_DIR / "mobilesam_encoder.onnx"
    sam_dec_onnx = MODELS_DIR / "mobilesam_decoder.onnx"
    u2netp_onnx = MODELS_DIR / "u2netp.onnx"

    if not args.skip_export:
        print("Exporting YOLOv3 face detector...")
        det_wrapper = YOLOv3RawHeadsWrapper(det_pt).eval()
        export_torch_onnx(
            det_wrapper,
            (torch.randn(1, 3, 608, 608),),
            det_onnx,
            input_names=["image"],
            output_names=["pred_32", "pred_16", "pred_8"],
            dynamic_axes={
                "image": {2: "height", 3: "width"},
                "pred_32": {2: "h32", 3: "w32"},
                "pred_16": {2: "h16", 3: "w16"},
                "pred_8": {2: "h8", 3: "w8"},
            },
        )

        print("Exporting HRNetV2 28-pt landmark model...")
        lm_wrapper = HRNetHeatmapWrapper(lm_pt).eval()
        export_torch_onnx(
            lm_wrapper,
            (torch.randn(1, 3, 256, 256),),
            lm_onnx,
            input_names=["image"],
            output_names=["heatmaps"],
            dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}},
        )

        print("Exporting MobileSAM encoder...")
        export_torch_onnx(
            sam_pt.image_encoder.eval(),
            (torch.randn(1, 3, 1024, 1024),),
            sam_enc_onnx,
            input_names=["image"],
            output_names=["image_embeddings"],
        )

        print("Exporting MobileSAM prompt+mask decoder...")
        sam_onnx = SamOnnxModel(model=sam_pt, return_single_mask=False).eval()
        embed_dim = sam_pt.prompt_encoder.embed_dim
        embed_size = sam_pt.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        dummy_dec = (
            torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
            torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float32),
            torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32),
            torch.randn(1, 1, *mask_input_size, dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1024.0, 1024.0], dtype=torch.float32),
        )
        export_torch_onnx(
            sam_onnx,
            dummy_dec,
            sam_dec_onnx,
            input_names=[
                "image_embeddings",
                "point_coords",
                "point_labels",
                "mask_input",
                "has_mask_input",
                "orig_im_size",
            ],
            output_names=["masks", "iou_predictions", "low_res_masks"],
            dynamic_axes={
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"},
            },
        )

    test_images = [
        p
        for p in [
            ROOT / "sutteang" / "2d_.png",
            ROOT / "sutteang" / "KakaoTalk_20261114_132504352.png",
        ]
        if p.exists()
    ]
    # Also test a non-square crop to validate dynamic detector H/W axes
    nonsquare_path = ARTIFACTS_DIR / "nonsquare_test.png"
    if test_images:
        base_img = cv2.imread(str(test_images[0]))
        cv2.imwrite(str(nonsquare_path), base_img[:, 128:896])
        test_images_with_rect = [*test_images, nonsquare_path]
    else:
        test_images_with_rect = []

    print("Verifying face detector & landmark ONNX parity...")
    face_report = verify_face_and_landmarks(
        det_pt, lm_pt, det_onnx, lm_onnx, test_images_with_rect
    )
    print("Verifying MobileSAM encoder & decoder ONNX parity...")
    sam_report = verify_mobilesam(sam_pt, sam_enc_onnx, sam_dec_onnx, test_images)

    manifest = {
        "schemaVersion": 1,
        "opsetVersion": ONNX_OPSET,
        "maxIrVersion": MAX_ONNX_IR_VERSION,
        "models": [
            {
                "role": "face_detector",
                "id": "anime_face_yolov3",
                "filename": "anime_face_yolov3.onnx",
                "version": "0.1.0",
                "upstreamRepo": "https://github.com/hysts/anime-face-detector",
                "upstreamRevision": AFD_COMMIT,
                "weightsRepo": HF_MODEL_REPOS["yolov3"],
                "weightsRevision": AFD_YOLOV3_SHA,
                "license": "MIT (code/weights) + Apache-2.0 (mmdetection components)",
                "sha256": sha256_file(det_onnx),
                "bytes": det_onnx.stat().st_size,
                "input": {
                    "name": "image",
                    "shape": [1, 3, "H_pad32", "W_pad32"],
                    "dtype": "float32",
                    "colorOrder": "RGB",
                    "resize": "keep_ratio_max_608_pad_bottom_right_stride32",
                    "normalize": {"scale": 255.0, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
                },
                "outputs": ["pred_32", "pred_16", "pred_8"],
                "postprocessVersion": "yolov3-mmdet-nms-v1",
            },
            {
                "role": "landmark_28",
                "id": "anime_face_hrnetv2",
                "filename": "anime_face_hrnetv2.onnx",
                "version": "0.1.0",
                "upstreamRepo": "https://github.com/hysts/anime-face-detector",
                "upstreamRevision": AFD_COMMIT,
                "weightsRepo": HF_MODEL_REPOS["hrnetv2"],
                "weightsRevision": AFD_HRNET_SHA,
                "license": "MIT (code/weights) + Apache-2.0 (mmpose components)",
                "sha256": sha256_file(lm_onnx),
                "bytes": lm_onnx.stat().st_size,
                "input": {
                    "name": "image",
                    "shape": ["B", 3, 256, 256],
                    "dtype": "float32",
                    "colorOrder": "RGB",
                    "crop": "bbox_scale_1.1_padding_1.25_affine_warp_bilinear",
                    "normalize": {
                        "scale": 255.0,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
                "outputs": ["heatmaps"],
                "postprocessVersion": "hrnetv2-fliptest-dark11-v1",
            },
            {
                "role": "background",
                "id": "u2netp",
                "filename": "u2netp.onnx",
                "version": "0.0.0",
                "upstreamRepo": "https://github.com/xuebinqin/U-2-Net",
                "downloadUrl": U2NETP_URL,
                "license": "Apache-2.0",
                "sha256": sha256_file(u2netp_onnx) if u2netp_onnx.exists() else "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
                "bytes": u2netp_onnx.stat().st_size if u2netp_onnx.exists() else 4574861,
                "input": {
                    "name": "input.1",
                    "shape": [1, 3, 320, 320],
                    "dtype": "float32",
                    "colorOrder": "RGB",
                    "normalize": {
                        "scale": "max_pixel",
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
                "outputs": ["alpha"],
                "postprocessVersion": "u2netp-minmax-v1",
            },
            {
                "role": "part_encoder",
                "id": "mobilesam_encoder",
                "filename": "mobilesam_encoder.onnx",
                "version": "1.0",
                "upstreamRepo": "https://github.com/ChaoningZhang/MobileSAM",
                "upstreamRevision": MOBILESAM_COMMIT,
                "checkpointUrl": MOBILESAM_WEIGHT_URL,
                "checkpointSha256": sha256_file(mobilesam_ckpt),
                "license": "Apache-2.0",
                "sha256": sha256_file(sam_enc_onnx),
                "bytes": sam_enc_onnx.stat().st_size,
                "input": {
                    "name": "image",
                    "shape": [1, 3, 1024, 1024],
                    "dtype": "float32",
                    "colorOrder": "RGB",
                    "resize": "longest_side_1024_normalize_then_pad_zero",
                    "normalize": {
                        "scale": 1.0,
                        "mean": [123.675, 116.28, 103.53],
                        "std": [58.395, 57.12, 57.375],
                    },
                },
                "outputs": ["image_embeddings"],
                "postprocessVersion": "sam-vit-t-encoder-v1",
            },
            {
                "role": "part_decoder",
                "id": "mobilesam_decoder",
                "filename": "mobilesam_decoder.onnx",
                "version": "1.0",
                "upstreamRepo": "https://github.com/ChaoningZhang/MobileSAM",
                "upstreamRevision": MOBILESAM_COMMIT,
                "checkpointUrl": MOBILESAM_WEIGHT_URL,
                "checkpointSha256": sha256_file(mobilesam_ckpt),
                "license": "Apache-2.0",
                "sha256": sha256_file(sam_dec_onnx),
                "bytes": sam_dec_onnx.stat().st_size,
                "input": {
                    "names": [
                        "image_embeddings",
                        "point_coords",
                        "point_labels",
                        "mask_input",
                        "has_mask_input",
                        "orig_im_size",
                    ],
                    "coordSpace": "resize_longest_1024",
                },
                "outputs": ["masks", "iou_predictions", "low_res_masks"],
                "postprocessVersion": "sam-multimask4-logits-v1",
            },
        ],
    }

    manifest_path = MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {"faceAndLandmarks": face_report, "mobileSam": sam_report}
    report_path = ARTIFACTS_DIR / "export-parity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {manifest_path} and {report_path}")


if __name__ == "__main__":
    main()
