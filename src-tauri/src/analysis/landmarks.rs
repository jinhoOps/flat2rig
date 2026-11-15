use super::models::{ModelRegistry, ModelRole};
use super::transforms::{bbox_to_center_scale, hrnet_affine_transform, sample_bilinear_rgb};
use crate::core::{Landmarks, Point};
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

const HRNET_INPUT_SIZE: usize = 256;
const HEATMAP_SIZE: usize = 64;
const NUM_KEYPOINTS: usize = 28;
const GAUSSIAN_KERNEL: usize = 11;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

const FLIP_PAIRS: [(usize, usize); 12] = [
    (0, 4),
    (1, 3),
    (5, 10),
    (6, 9),
    (7, 8),
    (11, 19),
    (12, 18),
    (13, 17),
    (14, 22),
    (15, 21),
    (16, 20),
    (24, 26),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LandmarkSource {
    Model,
    Estimated,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Landmark28Point {
    pub index: usize,
    pub x: f64,
    pub y: f64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigLandmarkProvenance {
    pub eye_l: LandmarkSource,
    pub eye_r: LandmarkSource,
    pub mouth: LandmarkSource,
    pub chin: LandmarkSource,
    pub crown: LandmarkSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LandmarksPredictionOutput {
    pub points_28: Vec<Landmark28Point>,
    pub rig_landmarks: Landmarks,
    pub rig_provenance: RigLandmarkProvenance,
    pub model_sha256: String,
    pub postprocess_version: String,
}

type HrnetBatchInputs = (Vec<f32>, [f64; 2], [f64; 2]);

fn prepare_hrnet_batch_inputs(
    rgba: &[u8],
    w: usize,
    h: usize,
    crop_bbox: [f64; 4],
) -> Result<HrnetBatchInputs, String> {
    let (center, scale) = bbox_to_center_scale(
        crop_bbox[0],
        crop_bbox[1],
        crop_bbox[2],
        crop_bbox[3],
        1.0,
        1.25,
    );
    let affine_inv = hrnet_affine_transform(
        center,
        scale,
        0.0,
        [HRNET_INPUT_SIZE as f64, HRNET_INPUT_SIZE as f64],
        [0.0, 0.0],
        true,
    )
    .ok_or("Singular HRNet affine crop transform")?;

    let plane = HRNET_INPUT_SIZE * HRNET_INPUT_SIZE;
    let img_stride = 3 * plane;
    let mut batch = vec![0f32; 2 * img_stride];

    for y in 0..HRNET_INPUT_SIZE {
        let row_off = y * HRNET_INPUT_SIZE;
        for x in 0..HRNET_INPUT_SIZE {
            let src = affine_inv.transform_point2(glam::DVec2::new(x as f64, y as f64));
            let [r, g, b] = sample_bilinear_rgb(rgba, w, h, src.x, src.y, false);
            let norm = [
                (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
                (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
                (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
            ];
            let idx = row_off + x;
            let flip_idx = row_off + (HRNET_INPUT_SIZE - 1 - x);
            for c in 0..3 {
                batch[c * plane + idx] = norm[c];
                batch[img_stride + c * plane + flip_idx] = norm[c];
            }
        }
    }

    Ok((batch, center, scale))
}

fn gaussian_kernel_1d(ksize: usize) -> Vec<f64> {
    let sigma = 0.3 * (((ksize as f64 - 1.0) * 0.5) - 1.0) + 0.8;
    let half = (ksize / 2) as isize;
    let mut weights: Vec<f64> = (-half..=half)
        .map(|i| {
            let d = i as f64 / sigma;
            (-0.5 * d * d).exp()
        })
        .collect();
    let sum: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum;
    }
    weights
}

#[inline]
fn reflect_101(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let mut i = idx;
    let max = len as isize - 1;
    while i < 0 || i > max {
        if i < 0 {
            i = -i;
        }
        if i > max {
            i = 2 * max - i;
        }
    }
    i as usize
}

/// Applies separable `cv2.GaussianBlur((k, k), 0)` (`BORDER_REFLECT_101`) and preserves `origin_max`.
fn dark_gaussian_blur(hm: &[f64], w: usize, h: usize, ksize: usize) -> Vec<f64> {
    let kernel = gaussian_kernel_1d(ksize);
    let half = (ksize / 2) as isize;
    let mut tmp = vec![0.0f64; w * h];
    let mut out = vec![0.0f64; w * h];

    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &kw) in kernel.iter().enumerate() {
                let sx = reflect_101(x as isize + (ki as isize - half), w);
                acc += hm[row + sx] * kw;
            }
            tmp[row + x] = acc;
        }
    }

    let mut origin_max = f64::NEG_INFINITY;
    for &v in hm {
        if v > origin_max {
            origin_max = v;
        }
    }

    let mut cur_max = f64::NEG_INFINITY;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &kw) in kernel.iter().enumerate() {
                let sy = reflect_101(y as isize + (ki as isize - half), h);
                acc += tmp[sy * w + x] * kw;
            }
            out[y * w + x] = acc;
            if acc > cur_max {
                cur_max = acc;
            }
        }
    }

    if origin_max.is_finite() && cur_max.is_finite() && cur_max > 1e-12 {
        let factor = origin_max / cur_max;
        for v in &mut out {
            *v *= factor;
        }
    }
    out
}

/// Decodes `flip_test + DARK(kernel=11)` heatmaps (`[2, 28, 64, 64]`) into 28 image points.
pub fn decode_hrnet_heatmaps(
    heatmaps: &[f32],
    center: [f64; 2],
    scale: [f64; 2],
) -> Result<Vec<Landmark28Point>, String> {
    let hw = HEATMAP_SIZE * HEATMAP_SIZE;
    let batch_stride = NUM_KEYPOINTS * hw;
    if heatmaps.len() != 2 * batch_stride {
        return Err(format!(
            "Expected HRNet heatmap length {}, got {}",
            2 * batch_stride,
            heatmaps.len()
        ));
    }

    let mut flipped_partner = vec![0usize; NUM_KEYPOINTS];
    for (i, slot) in flipped_partner.iter_mut().enumerate() {
        *slot = i;
    }
    for (a, b) in FLIP_PAIRS {
        flipped_partner[a] = b;
        flipped_partner[b] = a;
    }

    let affine_inv = hrnet_affine_transform(
        center,
        scale,
        0.0,
        [HEATMAP_SIZE as f64, HEATMAP_SIZE as f64],
        [0.0, 0.0],
        true,
    )
    .ok_or("Singular HRNet heatmap inverse transform")?;

    let mut points_28 = Vec::with_capacity(NUM_KEYPOINTS);

    for (k, &flip_k) in flipped_partner.iter().enumerate().take(NUM_KEYPOINTS) {
        let orig_off = k * hw;
        let flip_off = batch_stride + flip_k * hw;

        let mut merged = vec![0.0f64; hw];
        for y in 0..HEATMAP_SIZE {
            let row = y * HEATMAP_SIZE;
            for x in 0..HEATMAP_SIZE {
                // `flip_back` undoes horizontal flip, then `shift_heatmap` shifts columns right by 1.
                let src_x_after_shift = if x == 0 { 0 } else { x - 1 };
                let src_x = HEATMAP_SIZE - 1 - src_x_after_shift;
                let v_orig = heatmaps[orig_off + row + x] as f64;
                let v_flip = heatmaps[flip_off + row + src_x] as f64;
                let avg = (v_orig + v_flip) * 0.5;
                if !avg.is_finite() {
                    return Err("HRNet heatmap produced non-finite value".into());
                }
                merged[row + x] = avg;
            }
        }

        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (idx, &v) in merged.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = idx;
            }
        }
        let px = best_idx % HEATMAP_SIZE;
        let py = best_idx / HEATMAP_SIZE;

        let blurred = dark_gaussian_blur(&merged, HEATMAP_SIZE, HEATMAP_SIZE, GAUSSIAN_KERNEL);
        let log_hm: Vec<f64> = blurred.iter().map(|&v| v.max(1e-10).ln()).collect();

        let mut fx = px as f64;
        let mut fy = py as f64;
        if px > 1 && px < HEATMAP_SIZE - 2 && py > 1 && py < HEATMAP_SIZE - 2 {
            let at = |xx: usize, yy: usize| log_hm[yy * HEATMAP_SIZE + xx];
            let dx = 0.5 * (at(px + 1, py) - at(px - 1, py));
            let dy = 0.5 * (at(px, py + 1) - at(px, py - 1));
            let dxx = at(px + 1, py) - 2.0 * at(px, py) + at(px - 1, py);
            let dyy = at(px, py + 1) - 2.0 * at(px, py) + at(px, py - 1);
            let dxy = 0.25
                * (at(px + 1, py + 1)
                    - at(px + 1, py - 1)
                    - at(px - 1, py + 1)
                    + at(px - 1, py - 1));
            let det = dxx * dyy - dxy * dxy;
            if det.abs() > 1e-12 {
                let offset_x = -(dyy * dx - dxy * dy) / det;
                let offset_y = -(-dxy * dx + dxx * dy) / det;
                fx += offset_x;
                fy += offset_y;
            }
        }

        let mapped = affine_inv.transform_point2(glam::DVec2::new(fx, fy));
        points_28.push(Landmark28Point {
            index: k,
            x: mapped.x,
            y: mapped.y,
            score: best_val.max(0.0),
        });
    }

    Ok(points_28)
}

fn mean_points(points_28: &[Landmark28Point], indices: &[usize]) -> Point {
    let n = indices.len().max(1) as f64;
    let (sx, sy) = indices.iter().fold((0.0, 0.0), |acc, &idx| {
        let p = &points_28[idx];
        (acc.0 + p.x, acc.1 + p.y)
    });
    Point {
        x: sx / n,
        y: sy / n,
    }
}

/// Converts 28 predicted anime landmarks to 5 rig control points following character L/R:
/// - `eyeR` = viewer's left eye (`11..=16`)
/// - `eyeL` = viewer's right eye (`17..=22`)
/// - `mouth` = mouth center (`24..=27`)
/// - `chin` = chin (`2`)
/// - `crown` = estimated cranium top (explicitly flagged `LandmarkSource::Estimated`)
pub fn rig_landmarks_from_28(
    points_28: &[Landmark28Point],
    face_bbox: [f64; 4],
    img_w: usize,
    img_h: usize,
) -> Result<(Landmarks, RigLandmarkProvenance), String> {
    if points_28.len() != NUM_KEYPOINTS {
        return Err(format!("Expected 28 landmarks, got {}", points_28.len()));
    }
    let eye_r = mean_points(points_28, &[11, 12, 13, 14, 15, 16]);
    let eye_l = mean_points(points_28, &[17, 20, 18, 19, 21, 22]);
    let mouth = mean_points(points_28, &[24, 25, 26, 27]);
    let raw_chin = Point {
        x: points_28[2].x,
        y: points_28[2].y,
    };

    let eye_mid_x = 0.5 * (eye_r.x + eye_l.x);
    let eye_mid_y = 0.5 * (eye_r.y + eye_l.y);
    let dx = eye_l.x - eye_r.x;
    let dy = eye_l.y - eye_r.y;
    let eye_dist = dx.hypot(dy).max(12.0);
    let (ex_x, ex_y) = if dx >= 0.0 {
        (dx / eye_dist, dy / eye_dist)
    } else {
        (1.0, 0.0)
    };
    let (ey_x, ey_y) = (-ex_y, ex_x);

    // Guard against hand-on-chin / fluffy-hood occlusions pulling points_28[2] far out onto the cheek.
    let mouth_u = (mouth.x - eye_mid_x) * ex_x + (mouth.y - eye_mid_y) * ex_y;
    let mouth_v = (mouth.x - eye_mid_x) * ey_x + (mouth.y - eye_mid_y) * ey_y;
    let raw_chin_u = (raw_chin.x - eye_mid_x) * ex_x + (raw_chin.y - eye_mid_y) * ex_y;
    let raw_chin_v = (raw_chin.x - eye_mid_x) * ey_x + (raw_chin.y - eye_mid_y) * ey_y;
    let safe_chin_u = if (raw_chin_u - mouth_u).abs() > eye_dist * 0.32 {
        (mouth_u + (raw_chin_u - mouth_u).clamp(-eye_dist * 0.22, eye_dist * 0.22))
            .clamp(-eye_dist * 0.36, eye_dist * 0.36)
    } else {
        raw_chin_u
    };
    let min_chin_v = (mouth_v + eye_dist * 0.18).max(eye_dist * 0.42);
    let safe_chin_v = raw_chin_v.clamp(min_chin_v, eye_dist * 1.65);
    let chin = Point {
        x: (eye_mid_x + ex_x * safe_chin_u + ey_x * safe_chin_v)
            .clamp(0.0, (img_w.saturating_sub(1)) as f64),
        y: (eye_mid_y + ex_y * safe_chin_u + ey_y * safe_chin_v)
            .clamp(0.0, (img_h.saturating_sub(1)) as f64),
    };

    let brow_mid = mean_points(points_28, &[6, 9]);
    let up_x = eye_mid_x - chin.x;
    let up_y = eye_mid_y - chin.y;

    // Estimate crown using head axis above eyebrows and face bbox upper bound.
    let face_h = (face_bbox[3] - face_bbox[1]).max(1.0);
    let est_x = (brow_mid.x + up_x * 0.72).clamp(0.0, (img_w.saturating_sub(1)) as f64);
    let geom_y = eye_mid_y + up_y * 0.95;
    let box_top_y = face_bbox[1] - face_h * 0.12;
    let est_y = geom_y.min(box_top_y).clamp(0.0, (img_h.saturating_sub(1)) as f64);

    let rig = Landmarks {
        eye_l,
        eye_r,
        mouth,
        chin,
        crown: Point { x: est_x, y: est_y },
    };
    let provenance = RigLandmarkProvenance {
        eye_l: LandmarkSource::Model,
        eye_r: LandmarkSource::Model,
        mouth: LandmarkSource::Model,
        chin: LandmarkSource::Model,
        crown: LandmarkSource::Estimated,
    };
    Ok((rig, provenance))
}

/// Runs HRNetV2 28-pt landmark prediction for a single detected `crop_bbox`.
pub fn detect_landmarks(
    registry: &mut ModelRegistry,
    rgba: &[u8],
    w: usize,
    h: usize,
    raw_bbox: [f64; 4],
    crop_bbox: [f64; 4],
) -> Result<LandmarksPredictionOutput, String> {
    let (session_arc, sha256) = registry.get_session(ModelRole::Landmark28, false)?;
    let (batch, center, scale) = prepare_hrnet_batch_inputs(rgba, w, h, crop_bbox)?;
    let tensor = Tensor::from_array(([2usize, 3, HRNET_INPUT_SIZE, HRNET_INPUT_SIZE], batch))
        .map_err(|e| e.to_string())?;

    let mut session = session_arc
        .lock()
        .map_err(|_| "Landmark session lock poisoned".to_string())?;
    let outputs = session
        .run(ort::inputs!["image" => tensor])
        .map_err(|e| format!("Landmark inference failed: {e}"))?;
    let (shape, values) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    if shape.as_ref() != [2, NUM_KEYPOINTS as i64, HEATMAP_SIZE as i64, HEATMAP_SIZE as i64] {
        return Err(format!("Unexpected HRNet output shape: {shape:?}"));
    }

    let points_28 = decode_hrnet_heatmaps(values, center, scale)?;
    let (rig_landmarks, rig_provenance) = rig_landmarks_from_28(&points_28, raw_bbox, w, h)?;

    Ok(LandmarksPredictionOutput {
        points_28,
        rig_landmarks,
        rig_provenance,
        model_sha256: sha256,
        postprocess_version: "hrnetv2-fliptest-dark11-v1".into(),
    })
}
