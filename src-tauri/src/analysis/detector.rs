use super::models::{ModelRegistry, ModelRole};
use super::transforms::{sample_bilinear_rgb, scale_face_box, YoloResizeSpec};
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

const YOLO_MAX_SIZE: usize = 608;
const YOLO_PAD_DIVISOR: usize = 32;
const CONF_THR: f64 = 0.005;
const SCORE_THR: f64 = 0.05;
const NMS_IOU_THR: f64 = 0.45;
const MAX_PER_IMG: usize = 100;
const BOX_SCALE_FACTOR: f64 = 1.1;
pub const DEFAULT_FACE_SCORE_THRESHOLD: f64 = 0.5;

const STRIDES: [f64; 3] = [32.0, 16.0, 8.0];
const CENTERS: [f64; 3] = [16.0, 8.0, 4.0];
const BASE_SIZES: [[(f64, f64); 3]; 3] = [
    [(116.0, 90.0), (156.0, 198.0), (373.0, 326.0)],
    [(30.0, 61.0), (62.0, 45.0), (59.0, 119.0)],
    [(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)],
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceCandidate {
    pub id: usize,
    pub bbox: [f64; 4],
    pub crop_bbox: [f64; 4],
    pub score: f64,
    pub area_fraction: f64,
    pub center: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceDetectionOutput {
    pub candidates: Vec<FaceCandidate>,
    pub best_raw_score: f64,
    pub model_sha256: String,
    pub postprocess_version: String,
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn bbox_iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let iw = (x2 - x1).max(0.0);
    let ih = (y2 - y1).max(0.0);
    let inter = iw * ih;
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let denom = area_a + area_b - inter;
    if denom <= 0.0 {
        0.0
    } else {
        inter / denom
    }
}

/// Preprocesses RGBA pixels into YOLOv3 NCHW float32 tensor `[1, 3, padded_h, padded_w]`.
pub fn prepare_yolo_input(rgba: &[u8], w: usize, h: usize) -> (YoloResizeSpec, Vec<f32>) {
    let spec = YoloResizeSpec::new(w, h, YOLO_MAX_SIZE, YOLO_PAD_DIVISOR);
    let plane = spec.padded_h * spec.padded_w;
    let mut tensor = vec![0f32; 3 * plane];
    let sx_ratio = w as f64 / spec.resized_w as f64;
    let sy_ratio = h as f64 / spec.resized_h as f64;

    for y in 0..spec.resized_h {
        let sy = (y as f64 + 0.5) * sy_ratio - 0.5;
        let row_off = y * spec.padded_w;
        for x in 0..spec.resized_w {
            let sx = (x as f64 + 0.5) * sx_ratio - 0.5;
            let [r, g, b] = sample_bilinear_rgb(rgba, w, h, sx, sy, true);
            let idx = row_off + x;
            tensor[idx] = r / 255.0;
            tensor[plane + idx] = g / 255.0;
            tensor[2 * plane + idx] = b / 255.0;
        }
    }
    (spec, tensor)
}

/// Decodes raw YOLOv3 head outputs into NMS-filtered face candidates above `score_threshold`.
pub fn decode_yolo_maps(
    maps: &[(&[i64], &[f32]); 3],
    spec: YoloResizeSpec,
    score_threshold: f64,
) -> Result<(Vec<FaceCandidate>, f64), String> {
    let mut raw_boxes: Vec<([f64; 4], f64)> = Vec::new();
    let mut best_raw_score = 0.0f64;

    for level in 0..3 {
        let (shape, data) = maps[level];
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 18 {
            return Err(format!("Unexpected YOLOv3 pred_{level} shape: {shape:?}"));
        }
        let fh = shape[2] as usize;
        let fw = shape[3] as usize;
        let hw = fh * fw;
        if data.len() != 18 * hw {
            return Err(format!("YOLOv3 pred_{level} buffer size mismatch"));
        }
        let stride = STRIDES[level];
        let center_off = CENTERS[level];

        for (a, &(base_w, base_h)) in BASE_SIZES[level].iter().enumerate() {
            let ch_base = a * 6;
            for gy in 0..fh {
                let row_base = gy * fw;
                for gx in 0..fw {
                    let cell = row_base + gx;
                    let obj_logit = data[(ch_base + 4) * hw + cell] as f64;
                    let obj = sigmoid(obj_logit);
                    if obj < CONF_THR {
                        continue;
                    }
                    let cls = sigmoid(data[(ch_base + 5) * hw + cell] as f64);
                    let score = obj * cls;
                    if !score.is_finite() {
                        return Err("YOLOv3 produced non-finite confidence score".into());
                    }
                    if score > best_raw_score {
                        best_raw_score = score;
                    }
                    if score < SCORE_THR {
                        continue;
                    }

                    let tx = sigmoid(data[ch_base * hw + cell] as f64);
                    let ty = sigmoid(data[(ch_base + 1) * hw + cell] as f64);
                    let tw = (data[(ch_base + 2) * hw + cell] as f64).clamp(-20.0, 20.0);
                    let th = (data[(ch_base + 3) * hw + cell] as f64).clamp(-20.0, 20.0);

                    let cx = gx as f64 * stride + center_off + (tx - 0.5) * stride;
                    let cy = gy as f64 * stride + center_off + (ty - 0.5) * stride;
                    let bw = base_w * tw.exp();
                    let bh = base_h * th.exp();

                    let x1 = ((cx - bw * 0.5) / spec.scale_x).clamp(0.0, spec.orig_w as f64);
                    let y1 = ((cy - bh * 0.5) / spec.scale_y).clamp(0.0, spec.orig_h as f64);
                    let x2 = ((cx + bw * 0.5) / spec.scale_x).clamp(0.0, spec.orig_w as f64);
                    let y2 = ((cy + bh * 0.5) / spec.scale_y).clamp(0.0, spec.orig_h as f64);
                    if x2 > x1 && y2 > y1 {
                        raw_boxes.push(([x1, y1, x2, y2], score));
                    }
                }
            }
        }
    }

    raw_boxes.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut kept: Vec<([f64; 4], f64)> = Vec::new();
    for (cand_box, cand_score) in raw_boxes {
        if kept
            .iter()
            .any(|(prev_box, _)| bbox_iou(prev_box, &cand_box) > NMS_IOU_THR)
        {
            continue;
        }
        kept.push((cand_box, cand_score));
        if kept.len() >= MAX_PER_IMG {
            break;
        }
    }

    let img_area = (spec.orig_w.max(1) * spec.orig_h.max(1)) as f64;
    let mut candidates: Vec<FaceCandidate> = kept
        .into_iter()
        .filter(|(_, s)| *s >= score_threshold)
        .enumerate()
        .map(|(id, (bbox, score))| {
            let w = (bbox[2] - bbox[0]).max(0.0);
            let h = (bbox[3] - bbox[1]).max(0.0);
            let crop_bbox = scale_face_box(bbox, BOX_SCALE_FACTOR);
            FaceCandidate {
                id,
                bbox,
                crop_bbox,
                score,
                area_fraction: (w * h) / img_area,
                center: [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5],
            }
        })
        .collect();

    // Sort primary candidate first: highest score, tie-breaking by larger face box.
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.area_fraction.total_cmp(&a.area_fraction))
    });
    for (i, c) in candidates.iter_mut().enumerate() {
        c.id = i;
    }

    Ok((candidates, best_raw_score))
}

/// Runs YOLOv3 face detection via `ModelRegistry` on RGBA image bytes.
pub fn detect_faces(
    registry: &mut ModelRegistry,
    rgba: &[u8],
    w: usize,
    h: usize,
    score_threshold: f64,
) -> Result<FaceDetectionOutput, String> {
    let (session_arc, sha256) = registry.get_session(ModelRole::FaceDetector, false)?;
    let (spec, tensor_data) = prepare_yolo_input(rgba, w, h);
    let tensor = Tensor::from_array(([1usize, 3, spec.padded_h, spec.padded_w], tensor_data))
        .map_err(|e| e.to_string())?;

    let mut session = session_arc
        .lock()
        .map_err(|_| "Face detector session lock poisoned".to_string())?;
    let outputs = session
        .run(ort::inputs!["image" => tensor])
        .map_err(|e| format!("Face detector inference failed: {e}"))?;
    if outputs.len() < 3 {
        return Err(format!("Expected 3 YOLOv3 outputs, got {}", outputs.len()));
    }

    let (s0, d0) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    let (s1, d1) = outputs[1].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    let (s2, d2) = outputs[2].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    let maps = [(s0.as_ref(), d0), (s1.as_ref(), d1), (s2.as_ref(), d2)];

    let (candidates, best_raw_score) = decode_yolo_maps(&maps, spec, score_threshold)?;
    Ok(FaceDetectionOutput {
        candidates,
        best_raw_score,
        model_sha256: sha256,
        postprocess_version: "yolov3-mmdet-nms-v1".into(),
    })
}
