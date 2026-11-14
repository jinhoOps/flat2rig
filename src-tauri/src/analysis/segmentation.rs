use super::models::{ModelRegistry, ModelRole};
use super::state::SamEmbeddingCache;
use super::transforms::{sample_bilinear_rgb, SamResizeSpec};
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

const SAM_INPUT_SIZE: usize = 1024;
const SAM_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const SAM_STD: [f32; 3] = [58.395, 57.12, 57.375];
pub const SAM_PREPROCESS_VERSION: &str = "sam-vit-t-encoder-v1";

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptInput {
    pub x: f64,
    pub y: f64,
    pub label: i32,
}

#[derive(Debug, Clone)]
pub struct SamDecodedMasks {
    pub masks: Vec<Vec<u8>>, // 4 candidate masks at target (w, h), 0 or 255
    pub iou_scores: Vec<f32>,
    pub best_idx: usize,
    pub decoder_sha256: String,
}

/// Normalizes RGB inside `resized_w x resized_h` BEFORE zero-padding to `1024 x 1024`,
/// matching `mobile_sam.Sam.preprocess` exactly.
pub fn prepare_sam_encoder_input(rgba: &[u8], w: usize, h: usize) -> (SamResizeSpec, Vec<f32>) {
    let spec = SamResizeSpec::new(w, h, SAM_INPUT_SIZE);
    let plane = SAM_INPUT_SIZE * SAM_INPUT_SIZE;
    let mut tensor = vec![0f32; 3 * plane];
    let sx_ratio = w as f64 / spec.resized_w as f64;
    let sy_ratio = h as f64 / spec.resized_h as f64;

    for y in 0..spec.resized_h {
        let sy = (y as f64 + 0.5) * sy_ratio - 0.5;
        let row_off = y * SAM_INPUT_SIZE;
        for x in 0..spec.resized_w {
            let sx = (x as f64 + 0.5) * sx_ratio - 0.5;
            let [r, g, b] = sample_bilinear_rgb(rgba, w, h, sx, sy, true);
            let idx = row_off + x;
            tensor[idx] = (r - SAM_MEAN[0]) / SAM_STD[0];
            tensor[plane + idx] = (g - SAM_MEAN[1]) / SAM_STD[1];
            tensor[2 * plane + idx] = (b - SAM_MEAN[2]) / SAM_STD[2];
        }
    }
    (spec, tensor)
}

/// Computes MobileSAM `[1, 256, 64, 64]` embeddings for an image buffer.
pub fn encode_sam_embeddings(
    registry: &mut ModelRegistry,
    rgba: &[u8],
    w: usize,
    h: usize,
    image_hash: String,
    crop_key: Option<String>,
) -> Result<SamEmbeddingCache, String> {
    let (session_arc, model_hash) = registry.get_session(ModelRole::PartEncoder, false)?;
    let (spec, input_data) = prepare_sam_encoder_input(rgba, w, h);
    let tensor = Tensor::from_array(([1usize, 3, SAM_INPUT_SIZE, SAM_INPUT_SIZE], input_data))
        .map_err(|e| e.to_string())?;

    let mut session = session_arc
        .lock()
        .map_err(|_| "MobileSAM encoder lock poisoned".to_string())?;
    let outputs = session
        .run(ort::inputs!["image" => tensor])
        .map_err(|e| format!("MobileSAM encoder failed: {e}"))?;
    let (shape, values) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    if shape.as_ref() != [1, 256, 64, 64] || values.iter().any(|v| !v.is_finite()) {
        return Err(format!("Invalid MobileSAM embedding shape {shape:?}"));
    }

    Ok(SamEmbeddingCache {
        image_hash,
        model_hash,
        preprocess_version: SAM_PREPROCESS_VERSION.into(),
        crop_key,
        orig_w: w,
        orig_h: h,
        resized_w: spec.resized_w,
        resized_h: spec.resized_h,
        embeddings: values.to_vec(),
    })
}

/// Runs MobileSAM prompt decoder on a cached embedding using point and/or box prompts.
/// `box_xyxy`: optional `[x1, y1, x2, y2]` in the embedding's original coordinate space.
pub fn decode_sam_prompts(
    registry: &mut ModelRegistry,
    cache: &SamEmbeddingCache,
    points: &[PromptInput],
    box_xyxy: Option<[f64; 4]>,
) -> Result<SamDecodedMasks, String> {
    if points.is_empty() && box_xyxy.is_none() {
        return Err("Provide at least one point or box prompt".into());
    }
    let (session_arc, decoder_sha256) = registry.get_session(ModelRole::PartDecoder, false)?;
    let sx = cache.resized_w as f32 / cache.orig_w.max(1) as f32;
    let sy = cache.resized_h as f32 / cache.orig_h.max(1) as f32;

    let mut coords: Vec<f32> = Vec::with_capacity((points.len() + 3) * 2);
    let mut labels: Vec<f32> = Vec::with_capacity(points.len() + 3);
    for p in points {
        coords.push((p.x as f32) * sx);
        coords.push((p.y as f32) * sy);
        labels.push(p.label as f32);
    }
    if let Some([x1, y1, x2, y2]) = box_xyxy {
        coords.push((x1 as f32) * sx);
        coords.push((y1 as f32) * sy);
        labels.push(2.0);
        coords.push((x2 as f32) * sx);
        coords.push((y2 as f32) * sy);
        labels.push(3.0);
    } else {
        // MobileSAM ONNX contract appends padding token `(0, 0)` with label `-1` when no box is supplied.
        coords.push(0.0);
        coords.push(0.0);
        labels.push(-1.0);
    }

    let num_pts = labels.len();
    let emb_tensor = Tensor::from_array(([1usize, 256, 64, 64], cache.embeddings.clone()))
        .map_err(|e| e.to_string())?;
    let pts_tensor = Tensor::from_array(([1usize, num_pts, 2], coords))
        .map_err(|e| e.to_string())?;
    let lbls_tensor =
        Tensor::from_array(([1usize, num_pts], labels)).map_err(|e| e.to_string())?;
    let mask_input =
        Tensor::from_array(([1usize, 1, 256, 256], vec![0f32; 256 * 256]))
            .map_err(|e| e.to_string())?;
    let has_mask =
        Tensor::from_array(([1usize], vec![0f32])).map_err(|e| e.to_string())?;
    let orig_im_size = Tensor::from_array((
        [2usize],
        vec![cache.orig_h as f32, cache.orig_w as f32],
    ))
    .map_err(|e| e.to_string())?;

    let mut session = session_arc
        .lock()
        .map_err(|_| "MobileSAM decoder lock poisoned".to_string())?;
    let outputs = session
        .run(ort::inputs![
            "image_embeddings" => emb_tensor,
            "point_coords" => pts_tensor,
            "point_labels" => lbls_tensor,
            "mask_input" => mask_input,
            "has_mask_input" => has_mask,
            "orig_im_size" => orig_im_size,
        ])
        .map_err(|e| format!("MobileSAM decoder failed: {e}"))?;

    let (m_shape, m_vals) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    let (iou_shape, iou_vals) = outputs[1]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;

    let expected_area = cache.orig_w * cache.orig_h;
    if m_shape.len() != 4
        || m_shape[0] != 1
        || m_shape[1] != 4
        || m_shape[2] as usize != cache.orig_h
        || m_shape[3] as usize != cache.orig_w
        || m_vals.len() != 4 * expected_area
        || iou_shape.as_ref() != [1, 4]
        || m_vals.iter().any(|v| !v.is_finite())
        || iou_vals.iter().any(|v| !v.is_finite())
    {
        return Err(format!("Unexpected MobileSAM decoder shapes {m_shape:?} / {iou_shape:?}"));
    }

    let mut masks = Vec::with_capacity(4);
    for slot in 0..4 {
        let slice = &m_vals[slot * expected_area..(slot + 1) * expected_area];
        let bin: Vec<u8> = slice.iter().map(|&v| if v > 0.0 { 255 } else { 0 }).collect();
        masks.push(bin);
    }

    // Slots 1..4 are multimask_output=True proposals; choose highest IoU among slots 1..4 (or slot 0 if multi-point).
    let search_range = if points.len() >= 2 && box_xyxy.is_none() {
        0..4
    } else {
        1..4
    };
    let best_idx = search_range
        .max_by(|&a, &b| iou_vals[a].total_cmp(&iou_vals[b]))
        .unwrap_or(1);

    Ok(SamDecodedMasks {
        masks,
        iou_scores: iou_vals.to_vec(),
        best_idx,
        decoder_sha256,
    })
}

/// Crops `[x0, y0, x1, y1]` (clamped integers inside `orig_w x orig_h`) from RGBA.
pub fn crop_rgba_region(
    rgba: &[u8],
    orig_w: usize,
    orig_h: usize,
    x0: usize,
    y0: usize,
    crop_w: usize,
    crop_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; crop_w * crop_h * 4];
    for y in 0..crop_h {
        let sy = (y0 + y).min(orig_h.saturating_sub(1));
        for x in 0..crop_w {
            let sx = (x0 + x).min(orig_w.saturating_sub(1));
            let s = (sy * orig_w + sx) * 4;
            let d = (y * crop_w + x) * 4;
            out[d..d + 4].copy_from_slice(&rgba[s..s + 4]);
        }
    }
    out
}

/// Blits a cropped mask (`crop_w x crop_h`) placed at `(x0, y0)` into a full `(orig_w x orig_h)` buffer.
pub fn uncrop_mask(
    crop_mask: &[u8],
    crop_w: usize,
    crop_h: usize,
    x0: usize,
    y0: usize,
    orig_w: usize,
    orig_h: usize,
) -> Vec<u8> {
    let mut full = vec![0u8; orig_w * orig_h];
    for y in 0..crop_h {
        let dy = y0 + y;
        if dy >= orig_h {
            break;
        }
        let copy_w = crop_w.min(orig_w.saturating_sub(x0));
        let src_off = y * crop_w;
        let dst_off = dy * orig_w + x0;
        full[dst_off..dst_off + copy_w].copy_from_slice(&crop_mask[src_off..src_off + copy_w]);
    }
    full
}
