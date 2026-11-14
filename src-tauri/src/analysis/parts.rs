use super::landmarks::Landmark28Point;
use super::models::ModelRole;
use super::segmentation::{
    crop_rgba_region, decode_sam_prompts, encode_sam_embeddings, uncrop_mask, PromptInput,
    SAM_PREPROCESS_VERSION,
};
use super::state::{hash_rgba_bytes, AnalysisState, SamEmbeddingCache};
use crate::core::{self, Landmarks, IDS};
use crate::inference;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartProposalMeta {
    pub id: String,
    pub provenance: String,
    pub confidence: f64,
    pub pixel_count: usize,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartsProposalOutput {
    pub part_ids: Vec<String>,
    pub masks: Vec<Vec<u8>>,
    pub metas: Vec<PartProposalMeta>,
    pub provider: String,
    pub model_hashes: HashMap<String, String>,
    pub postprocess_version: String,
    pub image_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptCandidateMask {
    pub index: usize,
    pub iou_score: f32,
    pub mask: Vec<u8>,
    pub pixel_count: usize,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinePromptOutput {
    pub part_id: String,
    pub best_index: usize,
    pub candidates: Vec<PromptCandidateMask>,
    pub used_face_crop: bool,
    pub decoder_sha256: String,
    pub encoder_sha256: String,
}

#[derive(Debug, Clone, Copy)]
pub struct CropRect {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
}

impl CropRect {
    pub fn key(self) -> String {
        format!("{}:{}:{}:{}", self.x0, self.y0, self.w, self.h)
    }

    pub fn contains_point(self, x: f64, y: f64) -> bool {
        x >= self.x0 as f64
            && y >= self.y0 as f64
            && x < (self.x0 + self.w) as f64
            && y < (self.y0 + self.h) as f64
    }
}

pub fn compute_face_crop_rect(landmarks: &Landmarks, w: usize, h: usize) -> CropRect {
    let eye_dist = ((landmarks.eye_l.x - landmarks.eye_r.x).powi(2)
        + (landmarks.eye_l.y - landmarks.eye_r.y).powi(2))
    .sqrt()
    .max(w as f64 * 0.08);
    let cx = 0.5 * (landmarks.eye_l.x + landmarks.eye_r.x);
    let cy = 0.45 * (landmarks.eye_l.y + landmarks.eye_r.y) + 0.55 * landmarks.mouth.y;
    let half = (eye_dist * 1.9).max(96.0);
    let x0 = (cx - half).floor().clamp(0.0, (w.saturating_sub(2)) as f64) as usize;
    let y0 = (cy - half).floor().clamp(0.0, (h.saturating_sub(2)) as f64) as usize;
    let x1 = (cx + half).ceil().clamp((x0 + 2) as f64, w as f64) as usize;
    let y1 = (cy + half).ceil().clamp((y0 + 2) as f64, h as f64) as usize;
    CropRect {
        x0,
        y0,
        w: (x1 - x0).max(2),
        h: (y1 - y0).max(2),
    }
}

pub fn get_or_encode_full_sam(
    state: &AnalysisState,
    rgba: &[u8],
    w: usize,
    h: usize,
    image_hash: &str,
) -> Result<SamEmbeddingCache, String> {
    let model_hash = {
        let mut reg = state
            .registry
            .lock()
            .map_err(|_| "Registry lock poisoned".to_string())?;
        let (_, hash) = reg.get_session(ModelRole::PartEncoder, false)?;
        hash
    };
    {
        let guard = state
            .sam_cache
            .lock()
            .map_err(|_| "SAM cache lock poisoned".to_string())?;
        if let Some(ref cached) = *guard {
            if cached.image_hash == image_hash
                && cached.model_hash == model_hash
                && cached.preprocess_version == SAM_PREPROCESS_VERSION
                && cached.orig_w == w
                && cached.orig_h == h
            {
                return Ok(cached.clone());
            }
        }
    }
    let mut reg = state
        .registry
        .lock()
        .map_err(|_| "Registry lock poisoned".to_string())?;
    let fresh = encode_sam_embeddings(&mut reg, rgba, w, h, image_hash.to_string(), None)?;
    if let Ok(mut guard) = state.sam_cache.lock() {
        *guard = Some(fresh.clone());
    }
    Ok(fresh)
}

fn get_or_encode_crop_sam(
    state: &AnalysisState,
    rgba: &[u8],
    w: usize,
    h: usize,
    image_hash: &str,
    rect: CropRect,
) -> Result<SamEmbeddingCache, String> {
    let crop_key = rect.key();
    let model_hash = {
        let mut reg = state
            .registry
            .lock()
            .map_err(|_| "Registry lock poisoned".to_string())?;
        let (_, hash) = reg.get_session(ModelRole::PartEncoder, false)?;
        hash
    };
    {
        let guard = state
            .sam_crop_cache
            .lock()
            .map_err(|_| "SAM crop cache lock poisoned".to_string())?;
        if let Some(ref cached) = *guard {
            if cached.image_hash == image_hash
                && cached.model_hash == model_hash
                && cached.preprocess_version == SAM_PREPROCESS_VERSION
                && cached.crop_key.as_deref() == Some(&crop_key)
            {
                return Ok(cached.clone());
            }
        }
    }
    let crop_rgba = crop_rgba_region(rgba, w, h, rect.x0, rect.y0, rect.w, rect.h);
    let mut reg = state
        .registry
        .lock()
        .map_err(|_| "Registry lock poisoned".to_string())?;
    let fresh = encode_sam_embeddings(
        &mut reg,
        &crop_rgba,
        rect.w,
        rect.h,
        image_hash.to_string(),
        Some(crop_key),
    )?;
    if let Ok(mut guard) = state.sam_crop_cache.lock() {
        *guard = Some(fresh.clone());
    }
    Ok(fresh)
}

fn box_from_points_28(pts: &[Landmark28Point], indices: &[usize], pad_ratio: f64, w: usize, h: usize) -> Option<[f64; 4]> {
    if pts.len() != 28 || indices.is_empty() {
        return None;
    }
    let mut x1 = f64::INFINITY;
    let mut y1 = f64::INFINITY;
    let mut x2 = f64::NEG_INFINITY;
    let mut y2 = f64::NEG_INFINITY;
    for &idx in indices {
        let p = &pts[idx];
        x1 = x1.min(p.x);
        y1 = y1.min(p.y);
        x2 = x2.max(p.x);
        y2 = y2.max(p.y);
    }
    let bw = (x2 - x1).max(6.0);
    let bh = (y2 - y1).max(6.0);
    let pad_x = bw * pad_ratio;
    let pad_y = bh * (pad_ratio + 0.15);
    Some([
        (x1 - pad_x).clamp(0.0, (w.saturating_sub(1)) as f64),
        (y1 - pad_y).clamp(0.0, (h.saturating_sub(1)) as f64),
        (x2 + pad_x).clamp(0.0, (w.saturating_sub(1)) as f64),
        (y2 + pad_y).clamp(0.0, (h.saturating_sub(1)) as f64),
    ])
}

fn mask_area_in_fg(mask: &[u8], fg: &[u8]) -> usize {
    mask.iter()
        .zip(fg)
        .filter(|(&m, &f)| m > 128 && f > 8)
        .count()
}

fn sample_patch_rgb(rgba: &[u8], w: usize, h: usize, cx: f64, cy: f64) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    let mut cnt = 0.0f64;
    let ix = cx.round() as isize;
    let iy = cy.round() as isize;
    for dy in -2..=2 {
        for dx in -2..=2 {
            let x = ix + dx;
            let y = iy + dy;
            if x >= 0 && y >= 0 && x < w as isize && y < h as isize {
                let i = ((y as usize) * w + (x as usize)) * 4;
                if rgba[i + 3] > 8 {
                    sum[0] += rgba[i] as f64;
                    sum[1] += rgba[i + 1] as f64;
                    sum[2] += rgba[i + 2] as f64;
                    cnt += 1.0;
                }
            }
        }
    }
    if cnt > 0.0 {
        [sum[0] / cnt, sum[1] / cnt, sum[2] / cnt]
    } else {
        [128.0, 128.0, 128.0]
    }
}

#[inline]
fn color_dist_sq(rgba: &[u8], idx: usize, rgb: [f64; 3]) -> f64 {
    let p = idx * 4;
    let dr = rgba[p] as f64 - rgb[0];
    let dg = rgba[p + 1] as f64 - rgb[1];
    let db = rgba[p + 2] as f64 - rgb[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

fn pick_best_plausible_slot(
    masks: &[Vec<u8>],
    iou_scores: &[f32],
    fg: &[u8],
    max_area_fraction: f64,
    min_area: usize,
) -> (Vec<u8>, f32) {
    let fg_total = fg.iter().filter(|&&v| v > 8).count().max(1) as f64;
    let start_slot = if masks.len() > 1 { 1 } else { 0 };
    let mut best_slot = start_slot;
    let mut best_score = -1.0f32;
    for slot in start_slot..masks.len() {
        let area = mask_area_in_fg(&masks[slot], fg);
        let frac = area as f64 / fg_total;
        if area >= min_area && frac <= max_area_fraction && iou_scores[slot] > best_score {
            best_score = iou_scores[slot];
            best_slot = slot;
        }
    }
    if best_score < 0.0 {
        let fallback_slot = iou_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        return (masks[fallback_slot].clone(), iou_scores[fallback_slot]);
    }
    (masks[best_slot].clone(), best_score)
}

fn connected_components_over_min(mask: &[u8], w: usize, h: usize, min_area: usize) -> usize {
    let mut visited = vec![false; w * h];
    let mut comps = 0usize;
    let mut queue = VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if visited[idx] || mask[idx] <= 8 {
                continue;
            }
            visited[idx] = true;
            queue.push_back((x, y));
            let mut area = 0usize;
            while let Some((cx, cy)) = queue.pop_front() {
                area += 1;
                let neighbors = [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx < w && ny < h {
                        let ni = ny * w + nx;
                        if !visited[ni] && mask[ni] > 8 {
                            visited[ni] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }
            if area >= min_area {
                comps += 1;
            }
        }
    }
    comps
}

fn mask_centroid_x(mask: &[u8], w: usize, h: usize) -> Option<f64> {
    let mut sum_x = 0.0f64;
    let mut sum_w = 0.0f64;
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let v = mask[row + x] as f64;
            if v > 8.0 {
                sum_x += x as f64 * v;
                sum_w += v;
            }
        }
    }
    if sum_w > 0.0 {
        Some(sum_x / sum_w)
    } else {
        None
    }
}

pub fn diagnose_part_mask(
    id: &str,
    mask: &[u8],
    rgba: &[u8],
    w: usize,
    h: usize,
    confidence: f64,
) -> (usize, Vec<String>) {
    let mut pixel_count = 0usize;
    let mut bg_leak = 0usize;
    for i in 0..(w * h) {
        if mask[i] > 8 {
            pixel_count += 1;
            if rgba[i * 4 + 3] < 8 {
                bg_leak += 1;
            }
        }
    }
    let mut diag = Vec::new();
    if pixel_count == 0 {
        diag.push("empty".into());
        return (0, diag);
    }
    let single_body_part = !id.starts_with("hair") && id != "body" && !id.starts_with("clothes");
    if single_body_part {
        let comps = connected_components_over_min(mask, w, h, 12);
        if comps > 3 {
            diag.push(format!("fragmented:{comps}"));
        }
    }
    if bg_leak > 16 {
        diag.push(format!("background_leak:{bg_leak}"));
    }
    if confidence > 0.0 && confidence < 0.55 {
        diag.push("low_confidence".into());
    }
    (pixel_count, diag)
}

fn separable_box_bool(src: &[bool], w: usize, h: usize, radius: usize, dilate: bool) -> Vec<bool> {
    if radius == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let n = w * h;
    let mut horiz = vec![false; n];
    let mut row_pref = vec![0usize; w + 1];
    for y in 0..h {
        let row = y * w;
        row_pref[0] = 0;
        for x in 0..w {
            row_pref[x + 1] = row_pref[x] + usize::from(src[row + x]);
        }
        for x in 0..w {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(w);
            let sum = row_pref[hi] - row_pref[lo];
            horiz[row + x] = if dilate { sum > 0 } else { sum == hi - lo };
        }
    }
    let mut out = vec![false; n];
    let mut col_pref = vec![0usize; h + 1];
    for x in 0..w {
        col_pref[0] = 0;
        for y in 0..h {
            col_pref[y + 1] = col_pref[y] + usize::from(horiz[y * w + x]);
        }
        for y in 0..h {
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(h);
            let sum = col_pref[hi] - col_pref[lo];
            out[y * w + x] = if dilate { sum > 0 } else { sum == hi - lo };
        }
    }
    out
}

/// Seals MobileSAM deconvolution checkerboard pinholes inside `fg` using O(N) morphological closing
/// while avoiding harsh blocky square artifacts.
fn clean_sam_mask_in_fg(mask: &[u8], fg: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    let bin: Vec<bool> = (0..n).map(|i| mask[i] > 128 && fg[i] > 8).collect();
    let radius = (w.max(h) / 360).clamp(1, 2);
    let dilated = separable_box_bool(&bin, w, h, radius, true);
    let closed = separable_box_bool(&dilated, w, h, radius, false);
    let mut cur: Vec<u8> = (0..n)
        .map(|i| if (bin[i] || closed[i]) && fg[i] > 8 { 255 } else { 0 })
        .collect();
    for _ in 0..2 {
        for y in 1..h.saturating_sub(1) {
            let row = y * w;
            for x in 1..w.saturating_sub(1) {
                let i = row + x;
                if fg[i] <= 8 || cur[i] > 0 {
                    continue;
                }
                let n4 = u8::from(cur[i - w] > 0)
                    + u8::from(cur[i + w] > 0)
                    + u8::from(cur[i - 1] > 0)
                    + u8::from(cur[i + 1] > 0);
                if n4 >= 3 {
                    cur[i] = 255;
                }
            }
        }
    }
    cur
}

fn build_hair_palette(
    rgba: &[u8],
    fg: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    skin_rgb: [f64; 3],
    eye_dist: f64,
) -> Vec<[f64; 3]> {
    let face_cx = 0.5 * (landmarks.eye_l.x + landmarks.eye_r.x);
    let crown_y = (landmarks.crown.y + eye_dist * 0.12).clamp(0.0, (h - 1) as f64);
    let candidates = [
        (landmarks.crown.x, crown_y),
        (face_cx - eye_dist * 0.55, landmarks.crown.y + eye_dist * 0.25),
        (face_cx + eye_dist * 0.55, landmarks.crown.y + eye_dist * 0.25),
        (landmarks.eye_r.x - eye_dist * 0.45, landmarks.eye_r.y - eye_dist * 0.12),
        (landmarks.eye_l.x + eye_dist * 0.45, landmarks.eye_l.y - eye_dist * 0.12),
        (face_cx - eye_dist * 0.90, 0.5 * (landmarks.eye_r.y + landmarks.chin.y)),
        (face_cx + eye_dist * 0.90, 0.5 * (landmarks.eye_l.y + landmarks.chin.y)),
        (face_cx - eye_dist * 0.45, landmarks.eye_r.y - eye_dist * 0.45),
        (face_cx + eye_dist * 0.45, landmarks.eye_l.y - eye_dist * 0.45),
    ];
    let mut all_samples = Vec::with_capacity(candidates.len());
    for (cx, cy) in candidates {
        if cx >= 0.0 && cy >= 0.0 && cx < w as f64 && cy < h as f64 {
            let idx = (cy as usize) * w + (cx as usize);
            if fg[idx] > 128 {
                let rgb = sample_patch_rgb(rgba, w, h, cx, cy);
                let d_skin = ((rgb[0] - skin_rgb[0]).powi(2)
                    + (rgb[1] - skin_rgb[1]).powi(2)
                    + (rgb[2] - skin_rgb[2]).powi(2))
                .sqrt();
                if d_skin >= 26.0 {
                    all_samples.push(rgb);
                }
            }
        }
    }
    // Filter out near-white glare highlights unless the character truly has white/silver hair,
    // and avoid washed-out crown rim highlights when saturated/deeper hair samples exist so
    // pale pastel garments don't match highlight colors as hair.
    let non_glare: Vec<[f64; 3]> = all_samples
        .iter()
        .copied()
        .filter(|&rgb| {
            let luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
            let span = rgb[0].max(rgb[1]).max(rgb[2]) - rgb[0].min(rgb[1]).min(rgb[2]);
            !(luma > 224.0 && span < 32.0)
        })
        .collect();
    let rich_hair: Vec<[f64; 3]> = non_glare
        .iter()
        .copied()
        .filter(|&rgb| {
            let luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
            let span = rgb[0].max(rgb[1]).max(rgb[2]) - rgb[0].min(rgb[1]).min(rgb[2]);
            !(luma > 180.0 && span < 68.0)
        })
        .collect();
    if rich_hair.len() >= 2 {
        return rich_hair;
    }
    if !non_glare.is_empty() {
        return non_glare;
    }
    if !all_samples.is_empty() {
        return all_samples;
    }
    vec![sample_patch_rgb(rgba, w, h, landmarks.crown.x, crown_y)]
}

#[inline]
fn palette_min_dist(rgba: &[u8], idx: usize, palette: &[[f64; 3]]) -> f64 {
    palette
        .iter()
        .map(|&rgb| color_dist_sq(rgba, idx, rgb))
        .fold(f64::INFINITY, f64::min)
}

#[inline]
fn chroma_min_dist(rgba: &[u8], idx: usize, palette: &[[f64; 3]]) -> f64 {
    let p = idx * 4;
    let r = rgba[p] as f64;
    let g = rgba[p + 1] as f64;
    let b = rgba[p + 2] as f64;
    let avg = (r + g + b) / 3.0;
    let cr = r - avg;
    let cg = g - avg;
    let cb = b - avg;
    palette
        .iter()
        .map(|&s| {
            let savg = (s[0] + s[1] + s[2]) / 3.0;
            let dr = cr - (s[0] - savg);
            let dg = cg - (s[1] - savg);
            let db = cb - (s[2] - savg);
            (dr * dr + dg * dg + db * db).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

#[inline]
fn opponent_dist(rgba: &[u8], idx: usize, rgb: [f64; 3]) -> f64 {
    let p = idx * 4;
    let r = rgba[p] as f64;
    let g = rgba[p + 1] as f64;
    let b = rgba[p + 2] as f64;
    let dl = ((r + g + b) - (rgb[0] + rgb[1] + rgb[2])) / 3.0 * 0.45;
    let drg = (r - g) - (rgb[0] - rgb[1]);
    let dby = (b - 0.5 * (r + g)) - (rgb[2] - 0.5 * (rgb[0] + rgb[1]));
    (dl * dl + 3.2 * drg * drg + 3.2 * dby * dby).sqrt()
}

#[inline]
fn opponent_min_dist(rgba: &[u8], idx: usize, palette: &[[f64; 3]]) -> f64 {
    palette
        .iter()
        .map(|&rgb| opponent_dist(rgba, idx, rgb))
        .fold(f64::INFINITY, f64::min)
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum HairRole {
    Back = 0,
    SideR = 1,
    SideL = 2,
    Front = 3,
}

impl HairRole {
    const ALL: [HairRole; 4] = [HairRole::Back, HairRole::SideR, HairRole::SideL, HairRole::Front];

    fn as_id(self) -> &'static str {
        match self {
            HairRole::Back => "hair_back",
            HairRole::SideR => "hair_side_R",
            HairRole::SideL => "hair_side_L",
            HairRole::Front => "hair_front",
        }
    }
}

struct HairSplitInput<'a> {
    w: usize,
    h: usize,
    rgba: &'a [u8],
    landmarks: &'a Landmarks,
    hair_front_mask: &'a [u8],
    hair_back_mask: &'a [u8],
    face_mask: &'a [u8],
    torso_mask: &'a [u8],
}

/// Computes local-darkness crease (`valley`) and Dijkstra transition `seam_cost` over `hair_alpha > 0`.
fn compute_hair_valley_and_cost(
    rgba: &[u8],
    hair_alpha: &[u8],
    w: usize,
    h: usize,
    eye_dist: f64,
) -> (Vec<f32>, Vec<u32>, f32) {
    let n = w * h;
    let mut luma = vec![0.0f32; n];
    let pw = w + 1;
    let mut pref_sum = vec![0.0f64; pw * (h + 1)];
    let mut pref_cnt = vec![0u32; pw * (h + 1)];

    for y in 0..h {
        let mut row_s = 0.0f64;
        let mut row_c = 0u32;
        let row = y * w;
        let prow = (y + 1) * pw;
        let prev_prow = y * pw;
        for x in 0..w {
            let i = row + x;
            if hair_alpha[i] > 0 {
                let p = i * 4;
                let yv = 0.299 * rgba[p] as f32 + 0.587 * rgba[p + 1] as f32 + 0.114 * rgba[p + 2] as f32;
                luma[i] = yv;
                row_s += yv as f64;
                row_c += 1;
            }
            pref_sum[prow + x + 1] = pref_sum[prev_prow + x + 1] + row_s;
            pref_cnt[prow + x + 1] = pref_cnt[prev_prow + x + 1] + row_c;
        }
    }

    let rad = ((eye_dist * 0.16).round() as usize).clamp(5, 18);
    let mut valley = vec![0.0f32; n];
    let mut seam_cost = vec![16u32; n];
    let mut hair_valleys = Vec::with_capacity(n / 4);

    for y in 0..h {
        let y0 = y.saturating_sub(rad);
        let y1 = (y + rad + 1).min(h);
        let row = y * w;
        for x in 0..w {
            let i = row + x;
            if hair_alpha[i] == 0 {
                continue;
            }
            let x0 = x.saturating_sub(rad);
            let x1 = (x + rad + 1).min(w);
            let sum = pref_sum[y1 * pw + x1] - pref_sum[y0 * pw + x1]
                - pref_sum[y1 * pw + x0]
                + pref_sum[y0 * pw + x0];
            let cnt = pref_cnt[y1 * pw + x1] + pref_cnt[y0 * pw + x0]
                - pref_cnt[y1 * pw + x0]
                - pref_cnt[y0 * pw + x1];
            let local_mean = if cnt > 0 { (sum / cnt as f64) as f32 } else { luma[i] };

            let xl = if x > 0 && hair_alpha[i - 1] > 0 { luma[i - 1] } else { luma[i] };
            let xr = if x + 1 < w && hair_alpha[i + 1] > 0 { luma[i + 1] } else { luma[i] };
            let yu = if y > 0 && hair_alpha[i - w] > 0 { luma[i - w] } else { luma[i] };
            let yd = if y + 1 < h && hair_alpha[i + w] > 0 { luma[i + w] } else { luma[i] };
            let gx = (xr - xl).abs() * 0.5;
            let gy = (yd - yu).abs() * 0.5;
            let grad = (gx * gx + gy * gy).sqrt();

            let v = (local_mean - luma[i]).max(0.0) + 0.35 * grad;
            valley[i] = v;
            hair_valleys.push(v);
            seam_cost[i] = ((1.0 + v * 2.5) * 16.0).round().clamp(16.0, 4096.0) as u32;
        }
    }

    let thresh = if hair_valleys.is_empty() {
        16.0
    } else {
        hair_valleys.sort_by(|a, b| a.total_cmp(b));
        hair_valleys[(hair_valleys.len() * 68 / 100).min(hair_valleys.len() - 1)].clamp(10.0, 42.0)
    };
    (valley, seam_cost, thresh)
}

/// Carves a thin continuity-tracked ridge along existing hair creases in an anatomical corridor
/// so bangs, temples, side locks, and rear hair detach cleanly without flat horizontal cuts.
#[allow(clippy::too_many_arguments)]
fn carve_ridge_corridor(
    core: &mut [bool],
    hair_alpha: &[u8],
    valley: &[f32],
    w: usize,
    h: usize,
    x_range: (f64, f64),
    y_range: (f64, f64),
    ridge_min: f32,
) {
    let x_lo = (x_range.0.floor() as isize).clamp(1, (w as isize).saturating_sub(2)) as usize;
    let x_hi = (x_range.1.ceil() as isize).clamp(x_lo as isize, (w as isize).saturating_sub(2)) as usize;
    let y_lo = (y_range.0.floor() as isize).clamp(0, (h as isize).saturating_sub(1)) as usize;
    let y_hi = (y_range.1.ceil() as isize).clamp(y_lo as isize, (h as isize).saturating_sub(1)) as usize;
    if x_lo >= x_hi || y_lo >= y_hi {
        return;
    }

    let mut prev_x: Option<usize> = None;
    for y in y_lo..=y_hi {
        let row = y * w;
        let sx0 = prev_x.map(|px| px.saturating_sub(3).max(x_lo)).unwrap_or(x_lo);
        let sx1 = prev_x.map(|px| (px + 3).min(x_hi)).unwrap_or(x_hi);
        let mut best_x = sx0;
        let mut best_v = -1.0f32;
        for x in sx0..=sx1 {
            let i = row + x;
            if hair_alpha[i] > 0 && valley[i] > best_v {
                best_v = valley[i];
                best_x = x;
            }
        }
        if best_v >= ridge_min {
            core[row + best_x] = false;
            if best_x + 1 < w && valley[row + best_x + 1] >= ridge_min * 0.6 {
                core[row + best_x + 1] = false;
            } else if best_x > 0 {
                core[row + best_x - 1] = false;
            }
            prev_x = Some(best_x);
        } else {
            prev_x = None;
        }
    }
}

/// Collects 4-connected components from `mask`, preventing diagonal ink bridges between hair locks.
fn find_connected_components_4(mask: &[bool], w: usize, h: usize) -> Vec<Vec<usize>> {
    let n = w * h;
    let mut visited = vec![false; n];
    let mut comps = Vec::new();
    let mut q = VecDeque::new();
    for (start, &flag) in mask.iter().enumerate() {
        if !flag || visited[start] {
            continue;
        }
        visited[start] = true;
        q.push_back(start);
        let mut pxs = Vec::new();
        while let Some(idx) = q.pop_front() {
            pxs.push(idx);
            let x = idx % w;
            let y = idx / w;
            if x > 0 {
                let ni = idx - 1;
                if mask[ni] && !visited[ni] {
                    visited[ni] = true;
                    q.push_back(ni);
                }
            }
            if x + 1 < w {
                let ni = idx + 1;
                if mask[ni] && !visited[ni] {
                    visited[ni] = true;
                    q.push_back(ni);
                }
            }
            if y > 0 {
                let ni = idx - w;
                if mask[ni] && !visited[ni] {
                    visited[ni] = true;
                    q.push_back(ni);
                }
            }
            if y + 1 < h {
                let ni = idx + w;
                if mask[ni] && !visited[ni] {
                    visited[ni] = true;
                    q.push_back(ni);
                }
            }
        }
        comps.push(pxs);
    }
    comps
}

struct SeedStats {
    pixels: Vec<usize>,
    role: HairRole,
    cx: f64,
    cy: f64,
}

fn build_torso_row_spans(torso_mask: &[u8], w: usize, h: usize) -> Vec<Option<(usize, usize)>> {
    let mut spans = vec![None; h];
    for (y, span) in spans.iter_mut().enumerate() {
        let row = y * w;
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for x in 0..w {
            if torso_mask[row + x] > 128 {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if lo <= hi {
            *span = Some((lo, hi));
        }
    }
    spans
}

fn classify_seed_role(
    pxs: &[usize],
    input: &HairSplitInput<'_>,
    torso_spans: &[Option<(usize, usize)>],
    face_cx: f64,
    eye_mid_y: f64,
    eye_dist: f64,
    hh: f64,
) -> (HairRole, f64, f64) {
    let w = input.w;
    let h = input.h;
    let lm = input.landmarks;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut min_x = w;
    let mut max_x = 0usize;
    let mut min_y = h;
    let mut max_y = 0usize;
    let mut touches_face = 0usize;
    let mut touches_torso = 0usize;
    let mut lower_px = 0usize;
    let mut enclosed_torso = 0usize;

    for &idx in pxs {
        let x = idx % w;
        let y = idx / w;
        sx += x as f64;
        sy += y as f64;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);

        let near = [
            x.checked_sub(1).map(|nx| y * w + nx),
            (x + 1 < w).then(|| y * w + x + 1),
            y.checked_sub(1).map(|ny| ny * w + x),
            (y + 1 < h).then(|| (y + 1) * w + x),
        ];
        if near.into_iter().flatten().any(|ni| input.face_mask[ni] > 128) {
            touches_face += 1;
        }
        if near.into_iter().flatten().any(|ni| input.torso_mask[ni] > 128) {
            touches_torso += 1;
        }
        if (y as f64) >= lm.chin.y + eye_dist * 0.1 {
            lower_px += 1;
            if let Some((t_lo, t_hi)) = torso_spans[y] {
                if x + 4 >= t_lo && x <= t_hi + 4 {
                    enclosed_torso += 1;
                }
            }
        }
    }

    let cnt = pxs.len().max(1) as f64;
    let cx = sx / cnt;
    let cy = sy / cnt;
    let abs_dx = (cx - face_cx).abs();
    let f_min_y = min_y as f64;
    let f_max_y = max_y as f64;
    let enclosed_ratio = if lower_px > 0 {
        enclosed_torso as f64 / lower_px as f64
    } else {
        0.0
    };

    // Upper forehead / ahoge / bangs: foreground hair arches above cheek/jawline that do not drape onto torso
    let is_front = (f_max_y <= lm.chin.y + eye_dist * 0.35
        && f_min_y <= eye_mid_y + eye_dist * 0.15
        && (abs_dx <= eye_dist * 1.10 || touches_face >= 4))
        || (f_min_y <= eye_mid_y - eye_dist * 0.05
            && cy <= lm.chin.y + eye_dist * 0.05
            && f_max_y <= lm.chin.y + hh * 0.65
            && (abs_dx <= eye_dist * 1.15 || touches_face >= 6)
            && enclosed_ratio < 0.25)
        || (f_max_y <= lm.crown.y + hh * 0.25 && abs_dx <= eye_dist * 1.15);

    let role = if is_front {
        HairRole::Front
    } else {
        let outer_halo = f_min_y < lm.crown.y + hh * 0.22
            && abs_dx > eye_dist * 1.48
            && enclosed_ratio < 0.22;
        let drapes_over_front = enclosed_ratio >= 0.34
            || touches_torso >= 4
            || (touches_face >= 2 && f_min_y >= eye_mid_y - eye_dist * 0.45);
        let side_zone = f_min_y <= lm.chin.y + eye_dist * 0.70
            && abs_dx >= eye_dist * 0.38
            && abs_dx <= eye_dist * 1.90
            && !outer_halo
            && (drapes_over_front || (abs_dx <= eye_dist * 1.48 && f_min_y >= eye_mid_y - eye_dist * 0.45));
        if side_zone {
            if cx < face_cx {
                HairRole::SideR
            } else {
                HairRole::SideL
            }
        } else {
            HairRole::Back
        }
    };
    (role, cx, cy)
}

/// Splits one wide monolithic hair role component along a thin vertical valley, if applicable.
fn try_split_wide_hair_component(
    pixels: &[usize],
    w: usize,
    h: usize,
    min_strand_area: usize,
    min_width: usize,
) -> Option<(Vec<usize>, Vec<usize>)> {
    if pixels.len() < min_strand_area * 2 {
        return None;
    }
    let mut min_x = w;
    let mut max_x = 0usize;
    let mut col_counts = vec![0usize; w];
    for &i in pixels {
        let x = i % w;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        col_counts[x] += 1;
    }
    let span = max_x.saturating_sub(min_x) + 1;
    if span < min_width.max(16) {
        return None;
    }
    let pad = (span / 5).max(4);
    let search_lo = min_x + pad;
    let search_hi = max_x.saturating_sub(pad);
    if search_lo >= search_hi {
        return None;
    }
    let mut non_zero_cols: Vec<usize> = (min_x..=max_x)
        .filter(|&x| col_counts[x] > 0)
        .map(|x| col_counts[x])
        .collect();
    non_zero_cols.sort_unstable();
    let median = non_zero_cols.get(non_zero_cols.len() / 2).copied().unwrap_or(1).max(1);

    let mid_x = (min_x + max_x) as f64 * 0.5;
    let mut best_x = search_lo;
    let mut best_score = usize::MAX;
    let mut best_dist = f64::MAX;
    for (x, &cnt) in col_counts.iter().enumerate().take(search_hi + 1).skip(search_lo) {
        let dist = (x as f64 - mid_x).abs();
        if cnt < best_score || (cnt == best_score && dist < best_dist) {
            best_score = cnt;
            best_x = x;
            best_dist = dist;
        }
    }
    if best_score as f64 > median as f64 * 0.78 && span < (min_width * 7 / 5) {
        return None;
    }

    let mut left = Vec::with_capacity(pixels.len() / 2);
    let mut right = Vec::with_capacity(pixels.len() / 2);
    for &i in pixels {
        if i % w <= best_x {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    if left.len() >= min_strand_area && right.len() >= min_strand_area && h > 0 {
        Some((left, right))
    } else {
        None
    }
}

/// Splits combined hair pixels into natural crease-aligned strands (`hair_back*`, `hair_side_R*`,
/// `hair_side_L*`, `hair_front*`) in Cubism Z-order with 100% alpha conservation and zero overlap.
fn split_hair_layers(input: &HairSplitInput<'_>) -> Vec<(String, Vec<u8>)> {
    let (w, h) = (input.w, input.h);
    let n = w * h;
    let lm = input.landmarks;
    let eye_dist = ((lm.eye_l.x - lm.eye_r.x).powi(2) + (lm.eye_l.y - lm.eye_r.y).powi(2))
        .sqrt()
        .max(24.0);
    let face_cx = 0.5 * (lm.eye_l.x + lm.eye_r.x);
    let eye_mid_y = 0.5 * (lm.eye_l.y + lm.eye_r.y);
    let hh = (lm.chin.y - lm.crown.y).max(eye_dist * 1.4);

    let hair_alpha: Vec<u8> = (0..n)
        .map(|i| input.hair_front_mask[i].max(input.hair_back_mask[i]))
        .collect();
    let total_hair_px = hair_alpha.iter().filter(|&&a| a > 0).count();
    if total_hair_px == 0 {
        return vec![
            ("hair_back".to_string(), vec![0u8; n]),
            ("hair_front".to_string(), vec![0u8; n]),
        ];
    }

    let hair_max_y = hair_alpha
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| (a > 0).then_some(i / w))
        .max()
        .unwrap_or(h - 1) as f64;

    let (valley, seam_cost, valley_thresh) =
        compute_hair_valley_and_cost(input.rgba, &hair_alpha, w, h, eye_dist);

    let mut core: Vec<bool> = (0..n)
        .map(|i| hair_alpha[i] > 0 && valley[i] < valley_thresh)
        .collect();

    // Cut major anatomical partings along actual dark creases before extracting 4-connected cores.
    carve_ridge_corridor(
        &mut core,
        &hair_alpha,
        &valley,
        w,
        h,
        (face_cx - eye_dist * 0.34, face_cx + eye_dist * 0.34),
        (lm.crown.y, eye_mid_y + eye_dist * 0.25),
        valley_thresh * 0.40,
    );
    for sign in [-1.0f64, 1.0] {
        let (tx0, tx1) = (face_cx + sign * eye_dist * 0.58, face_cx + sign * eye_dist * 1.15);
        carve_ridge_corridor(
            &mut core,
            &hair_alpha,
            &valley,
            w,
            h,
            (tx0.min(tx1), tx0.max(tx1)),
            (lm.crown.y + hh * 0.08, lm.chin.y + eye_dist * 0.15),
            valley_thresh * 0.35,
        );
        let (sx0, sx1) = (face_cx + sign * eye_dist * 1.15, face_cx + sign * eye_dist * 1.92);
        carve_ridge_corridor(
            &mut core,
            &hair_alpha,
            &valley,
            w,
            h,
            (sx0.min(sx1), sx0.max(sx1)),
            (eye_mid_y, hair_max_y),
            valley_thresh * 0.35,
        );
    }

    // Separate upper crown/temple arches from long lower shoulder/chest curls along cheek pinch rows.
    let pre_cores = find_connected_components_4(&core, w, h);
    let y_pinch_lo = ((eye_mid_y + eye_dist * 0.12).floor() as isize).max(0) as usize;
    let y_pinch_hi = ((lm.chin.y + eye_dist * 0.28).ceil() as isize).clamp(y_pinch_lo as isize, (h as isize).saturating_sub(1)) as usize;
    for comp in pre_cores {
        if comp.len() < 120 {
            continue;
        }
        let min_cy = comp.iter().map(|&i| i / w).min().unwrap_or(h);
        let max_cy = comp.iter().map(|&i| i / w).max().unwrap_or(0);
        if (min_cy as f64) <= lm.crown.y + hh * 0.35 && (max_cy as f64) >= lm.chin.y + hh * 0.60 {
            let mut row_px: HashMap<usize, Vec<usize>> = HashMap::new();
            for &i in &comp {
                let y = i / w;
                if y >= y_pinch_lo && y <= y_pinch_hi {
                    row_px.entry(y).or_default().push(i);
                }
            }
            if let Some((&best_y, pxs)) = row_px.iter().min_by(|(ay, apxs), (by, bpxs)| {
                let amax_v = apxs.iter().map(|&i| valley[i]).fold(0.0f32, f32::max);
                let bmax_v = bpxs.iter().map(|&i| valley[i]).fold(0.0f32, f32::max);
                let ascore = (apxs.len() as f64) / (1.0 + 0.06 * amax_v as f64);
                let bscore = (bpxs.len() as f64) / (1.0 + 0.06 * bmax_v as f64);
                ascore.total_cmp(&bscore).then_with(|| ay.cmp(by))
            }) {
                for &i in pxs {
                    core[i] = false;
                    if best_y + 1 < h {
                        core[i + w] = false;
                    }
                }
            }
        }
    }

    let min_seed_area = ((total_hair_px / 220).max(16)).min((total_hair_px / 24).max(6));
    let raw_cores = find_connected_components_4(&core, w, h);
    let torso_spans = build_torso_row_spans(input.torso_mask, w, h);

    let mut seeds: Vec<SeedStats> = Vec::new();
    let mut seed_of_hair_comp = vec![false; n];
    for c in raw_cores {
        if c.len() < min_seed_area {
            continue;
        }
        let (role, cx, cy) =
            classify_seed_role(&c, input, &torso_spans, face_cx, eye_mid_y, eye_dist, hh);
        for &idx in &c {
            seed_of_hair_comp[idx] = true;
        }
        seeds.push(SeedStats { pixels: c, role, cx, cy });
    }

    // Ensure any disconnected hair island without a core seed still receives its cleanest interior pixel as seed.
    let hair_bool: Vec<bool> = hair_alpha.iter().map(|&a| a > 0).collect();
    for island in find_connected_components_4(&hair_bool, w, h) {
        if island.iter().any(|&i| seed_of_hair_comp[i]) {
            continue;
        }
        if let Some(&best_px) = island.iter().min_by(|&&a, &&b| valley[a].total_cmp(&valley[b])) {
            let single = [best_px];
            let (role, cx, cy) = classify_seed_role(
                &single,
                input,
                &torso_spans,
                face_cx,
                eye_mid_y,
                eye_dist,
                hh,
            );
            seeds.push(SeedStats {
                pixels: vec![best_px],
                role,
                cx,
                cy,
            });
        }
    }

    if seeds.is_empty() {
        return vec![
            ("hair_back".to_string(), hair_alpha),
            ("hair_front".to_string(), vec![0u8; n]),
        ];
    }

    // Multi-source crease-weighted Dijkstra watershed over every hair_alpha > 0 pixel.
    let num_seeds = seeds.len();
    let mut dist = vec![u32::MAX; n];
    let mut owner = vec![u32::MAX; n];
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::with_capacity(total_hair_px / 4);

    for (sid, seed) in seeds.iter().enumerate() {
        for &idx in &seed.pixels {
            dist[idx] = 0;
            owner[idx] = sid as u32;
            heap.push(Reverse((0, idx)));
        }
    }

    while let Some(Reverse((d, idx))) = heap.pop() {
        if d > dist[idx] {
            continue;
        }
        let o = owner[idx];
        let x = idx % w;
        let y = idx / w;
        let neighbors = [
            (x > 0).then(|| idx - 1),
            (x + 1 < w).then(|| idx + 1),
            (y > 0).then(|| idx - w),
            (y + 1 < h).then(|| idx + w),
        ];
        for ni in neighbors.into_iter().flatten() {
            if hair_alpha[ni] == 0 {
                continue;
            }
            let nd = d.saturating_add(seam_cost[ni]);
            if nd < dist[ni] {
                dist[ni] = nd;
                owner[ni] = o;
                heap.push(Reverse((nd, ni)));
            }
        }
    }

    // Fallback for any diagonal single-pixel orphan inside hair_alpha.
    for (i, &ha) in hair_alpha.iter().enumerate() {
        if ha > 0 && owner[i] == u32::MAX {
            let fx = (i % w) as f64;
            let fy = (i / w) as f64;
            let best = seeds
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a.cx - fx).powi(2) + (a.cy - fy).powi(2);
                    let db = (b.cx - fx).powi(2) + (b.cy - fy).powi(2);
                    da.total_cmp(&db)
                })
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            owner[i] = best;
        }
    }

    // Collect pixels per watershed region, then merge regions within each role along weakest seams.
    let mut region_pxs: Vec<Vec<usize>> = vec![Vec::new(); num_seeds];
    for (i, &o) in owner.iter().enumerate() {
        if hair_alpha[i] > 0 && (o as usize) < num_seeds {
            region_pxs[o as usize].push(i);
        }
    }

    let min_strand_area = ((total_hair_px / 48).max(120)).min((total_hair_px / 8).max(24));
    let mut out = Vec::new();

    for role in HairRole::ALL {
        let mut active: Vec<usize> = (0..num_seeds)
            .filter(|&sid| seeds[sid].role == role && !region_pxs[sid].is_empty())
            .collect();
        if active.is_empty() {
            if role == HairRole::Back || role == HairRole::Front {
                out.push((role.as_id().to_string(), vec![0u8; n]));
            }
            continue;
        }

        // Compute current centroids from actual claimed pixels and adjacency seam weights inside `role`.
        let mut centroids: Vec<(f64, f64)> = vec![(0.0, 0.0); num_seeds];
        for &sid in &active {
            let pxs = &region_pxs[sid];
            let sx: f64 = pxs.iter().map(|&i| (i % w) as f64).sum();
            let sy: f64 = pxs.iter().map(|&i| (i / w) as f64).sum();
            centroids[sid] = (sx / pxs.len() as f64, sy / pxs.len() as f64);
        }

        while active.len() > 1 {
            let smallest_pos = active
                .iter()
                .enumerate()
                .min_by_key(|(_, &sid)| region_pxs[sid].len())
                .map(|(p, _)| p)
                .unwrap();
            let src_sid = active[smallest_pos];
            if region_pxs[src_sid].len() >= min_strand_area && active.len() <= 3 {
                break;
            }

            // Measure shared-border average seam cost to every adjacent region of the SAME role.
            let mut adj: HashMap<usize, (u64, u32)> = HashMap::new();
            for &i in &region_pxs[src_sid] {
                let x = i % w;
                let y = i / w;
                let nbs = [
                    (x > 0).then(|| i - 1),
                    (x + 1 < w).then(|| i + 1),
                    (y > 0).then(|| i - w),
                    (y + 1 < h).then(|| i + w),
                ];
                for ni in nbs.into_iter().flatten() {
                    if hair_alpha[ni] == 0 {
                        continue;
                    }
                    let nb_sid = owner[ni] as usize;
                    if nb_sid != src_sid && seeds[nb_sid].role == role {
                        let entry = adj.entry(nb_sid).or_insert((0, 0));
                        entry.0 += (seam_cost[i] + seam_cost[ni]) as u64 / 2;
                        entry.1 += 1;
                    }
                }
            }

            let dst_sid = if let Some((&best_nb, _)) = adj.iter().min_by(|(a_id, a_w), (b_id, b_w)| {
                let aa = a_w.0 as f64 / a_w.1 as f64;
                let bb = b_w.0 as f64 / b_w.1 as f64;
                aa.total_cmp(&bb).then_with(|| a_id.cmp(b_id))
            }) {
                best_nb
            } else {
                let (sx, sy) = centroids[src_sid];
                active
                    .iter()
                    .copied()
                    .filter(|&sid| sid != src_sid)
                    .min_by(|&a, &b| {
                        let da = (centroids[a].0 - sx).powi(2) + (centroids[a].1 - sy).powi(2);
                        let db = (centroids[b].0 - sx).powi(2) + (centroids[b].1 - sy).powi(2);
                        da.total_cmp(&db)
                    })
                    .unwrap()
            };

            let moving = std::mem::take(&mut region_pxs[src_sid]);
            for &i in &moving {
                owner[i] = dst_sid as u32;
            }
            let prev_len = region_pxs[dst_sid].len() as f64;
            let add_len = moving.len() as f64;
            let tot = (prev_len + add_len).max(1.0);
            centroids[dst_sid] = (
                (centroids[dst_sid].0 * prev_len + centroids[src_sid].0 * add_len) / tot,
                (centroids[dst_sid].1 * prev_len + centroids[src_sid].1 * add_len) / tot,
            );
            region_pxs[dst_sid].extend(moving);
            active.swap_remove(smallest_pos);
        }

        let mut strand_lists: Vec<Vec<usize>> =
            active.into_iter().map(|sid| std::mem::take(&mut region_pxs[sid])).collect();
        if strand_lists.len() == 1 && (role == HairRole::Front || role == HairRole::Back) {
            let min_w = (eye_dist * if role == HairRole::Front { 1.45 } else { 1.75 }) as usize;
            if let Some((left, right)) =
                try_split_wide_hair_component(&strand_lists[0], w, h, min_strand_area, min_w)
            {
                strand_lists = vec![left, right];
            }
        }
        strand_lists.sort_by_key(|v| std::cmp::Reverse(v.len()));
        for (idx, pxs) in strand_lists.into_iter().enumerate() {
            let id = if idx == 0 {
                role.as_id().to_string()
            } else {
                format!("{}_{}", role.as_id(), idx + 1)
            };
            let mut m = vec![0u8; n];
            for i in pxs {
                m[i] = hair_alpha[i];
            }
            out.push((id, m));
        }
    }

    out
}

struct ClothesExtra {
    mask: Vec<u8>,
    sam_hits: usize,
    total_hits: usize,
    confidence: f64,
}

struct TorsoSeeds {
    skin_pts: Vec<PromptInput>,
    clothes_pts: Vec<PromptInput>,
    clothes_palette: Vec<[f64; 3]>,
    skin_torso_count: usize,
    clothes_count: usize,
}

#[allow(clippy::too_many_arguments)]
fn find_torso_seeds(
    rgba: &[u8],
    fg: &[u8],
    face_mask: &[u8],
    hair_upper_mask: &[u8],
    neck_mask: &[u8],
    torso_macro_mask: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    skin_rgb: [f64; 3],
    hair_palette: &[[f64; 3]],
    eye_dist: f64,
) -> TorsoSeeds {
    let n = w * h;
    let face_cx = 0.5 * (landmarks.eye_l.x + landmarks.eye_r.x);
    let eye_mid_y = 0.5 * (landmarks.eye_l.y + landmarks.eye_r.y);
    let hh = (landmarks.chin.y - landmarks.crown.y).max(eye_dist * 1.4);
    let neck_skin_rgb = sample_patch_rgb(
        rgba,
        w,
        h,
        landmarks.chin.x,
        (landmarks.chin.y + eye_dist * 0.28).clamp(0.0, (h - 1) as f64),
    );
    let skin_palette = [skin_rgb, neck_skin_rgb];

    // Trace only continuous head-anchored core hair downward so sweater interiors aren't excluded.
    let mut prelim_hair = vec![false; n];
    let mut q = VecDeque::new();
    let head_hair_cut = landmarks.chin.y + eye_dist * 0.16;
    for y in (eye_mid_y.floor() as usize).min(h)..(head_hair_cut.ceil() as usize).min(h) {
        let row = y * w;
        for x in 0..w {
            let i = row + x;
            if fg[i] > 128 && face_mask[i] <= 128 {
                let hair_opp = opponent_min_dist(rgba, i, hair_palette);
                let hair_p = palette_min_dist(rgba, i, hair_palette);
                let skin_opp = opponent_min_dist(rgba, i, &skin_palette);
                if (hair_upper_mask[i] > 128 || hair_p < 36.0)
                    && hair_p < 42.0
                    && hair_opp < 36.0
                    && skin_opp > hair_opp + 12.0
                {
                    prelim_hair[i] = true;
                    q.push_back(i);
                }
            }
        }
    }
    while let Some(idx) = q.pop_front() {
        let x = idx % w;
        let y = idx / w;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if prelim_hair[ni]
                    || fg[ni] <= 128
                    || (ny as f64) < eye_mid_y
                    || (nx as f64 - face_cx).abs() > eye_dist * 2.5
                {
                    continue;
                }
                let hair_opp = opponent_min_dist(rgba, ni, hair_palette);
                let hair_p = palette_min_dist(rgba, ni, hair_palette);
                let skin_opp = opponent_min_dist(rgba, ni, &skin_palette);
                if hair_p < 34.0 && hair_opp < 28.0 && skin_opp > hair_opp + 12.0 {
                    prelim_hair[ni] = true;
                    q.push_back(ni);
                }
            }
        }
    }

    let y_start = ((landmarks.chin.y + eye_dist * 0.18).floor() as usize).min(h);
    let y_end = ((landmarks.chin.y + hh * 2.4).ceil() as usize).min(h);
    let x_start = ((face_cx - eye_dist * 2.2).floor() as isize).max(0) as usize;
    let x_end = ((face_cx + eye_dist * 2.2).ceil() as usize).min(w);

    // 1. Find the dominant garment interior color via a quantized lower-torso histogram so we
    //    sample broad knit/fabric interior instead of extreme dark hair shadow outliers.
    let cl_y0 = ((landmarks.chin.y + hh * 0.72).floor() as usize).min(h);
    let cl_y1 = ((landmarks.chin.y + hh * 2.0).ceil() as usize).min(h);
    let cl_x0 = ((face_cx - eye_dist * 1.6).floor() as isize).max(0) as usize;
    let cl_x1 = ((face_cx + eye_dist * 1.6).ceil() as usize).min(w);
    #[derive(Default, Clone, Copy)]
    struct ColorBin {
        count: usize,
        sum_r: f64,
        sum_g: f64,
        sum_b: f64,
    }
    let mut bins: HashMap<(u8, u8, u8), ColorBin> = HashMap::new();
    let mut macro_px_in_box = 0usize;
    for y in (cl_y0..cl_y1).step_by(2) {
        let row = y * w;
        for x in (cl_x0..cl_x1).step_by(2) {
            let i = row + x;
            if fg[i] <= 128 || face_mask[i] > 128 || neck_mask[i] > 128 || prelim_hair[i] {
                continue;
            }
            if torso_macro_mask[i] > 128 {
                macro_px_in_box += 1;
            }
        }
    }
    let require_macro = macro_px_in_box >= 200;
    for y in (cl_y0..cl_y1).step_by(2) {
        let row = y * w;
        for x in (cl_x0..cl_x1).step_by(2) {
            let i = row + x;
            if fg[i] <= 128
                || face_mask[i] > 128
                || neck_mask[i] > 128
                || prelim_hair[i]
                || (require_macro && torso_macro_mask[i] <= 128)
            {
                continue;
            }
            let skin_d = color_dist_sq(rgba, i, skin_rgb);
            let skin_opp = opponent_min_dist(rgba, i, &skin_palette);
            if skin_opp < 26.0 && skin_d < 46.0 {
                continue;
            }
            let p = i * 4;
            let r = rgba[p] as f64;
            let g = rgba[p + 1] as f64;
            let b = rgba[p + 2] as f64;
            let key = (rgba[p] >> 4, rgba[p + 1] >> 4, rgba[p + 2] >> 4);
            let entry = bins.entry(key).or_default();
            entry.count += 1;
            entry.sum_r += r;
            entry.sum_g += g;
            entry.sum_b += b;
        }
    }
    let mut ranked_bins: Vec<(usize, [f64; 3])> = bins
        .into_values()
        .filter(|b| b.count >= 24)
        .map(|b| (b.count, [b.sum_r / b.count as f64, b.sum_g / b.count as f64, b.sum_b / b.count as f64]))
        .collect();
    ranked_bins.sort_by_key(|b| std::cmp::Reverse(b.0));
    let modal_clothes_rgb = ranked_bins
        .iter()
        .find(|&&(_, mean)| {
            let hp = hair_palette
                .iter()
                .map(|&hr| ((mean[0] - hr[0]).powi(2) + (mean[1] - hr[1]).powi(2) + (mean[2] - hr[2]).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            hp >= 22.0
        })
        .or_else(|| ranked_bins.first())
        .map(|&(_, mean)| mean);

    let mut skin_buckets: [Option<(f64, PromptInput)>; 3] = [None, None, None];
    let mut clothes_buckets: [Option<(f64, PromptInput)>; 3] = [None, None, None];
    let mut skin_torso_count = 0usize;
    let mut clothes_count = 0usize;

    for y in y_start..y_end {
        let row = y * w;
        let fy = y as f64;
        for x in x_start..x_end {
            let i = row + x;
            if fg[i] <= 128
                || face_mask[i] > 128
                || prelim_hair[i]
                || neck_mask[i] > 128
            {
                continue;
            }
            let fx = x as f64;
            let skin_d = color_dist_sq(rgba, i, skin_rgb);
            let skin_opp = opponent_min_dist(rgba, i, &skin_palette);
            let hair_p = palette_min_dist(rgba, i, hair_palette);
            let hair_opp = opponent_min_dist(rgba, i, hair_palette);
            let dx_rel = (fx - face_cx) / eye_dist;
            let skin_y_limit = if dx_rel.abs() < 0.48 {
                landmarks.chin.y + hh * 0.60
            } else {
                landmarks.chin.y + hh * 0.65
            };
            let closer_to_modal_clothes = modal_clothes_rgb
                .is_some_and(|mc| color_dist_sq(rgba, i, mc) + 8.0 < skin_d);

            if skin_opp < 26.0
                && skin_d < 46.0
                && hair_opp > 24.0
                && !closer_to_modal_clothes
                && fy <= skin_y_limit
                && dx_rel.abs() <= 1.38
            {
                skin_torso_count += 1;
                if dx_rel.abs() >= 0.42 || fy >= landmarks.chin.y + eye_dist * 0.40 {
                    let side = if dx_rel < -0.50 {
                        0
                    } else if dx_rel <= 0.50 {
                        1
                    } else {
                        2
                    };
                    let score = skin_opp;
                    if skin_buckets[side].is_none_or(|(best_s, _)| score < best_s) {
                        skin_buckets[side] = Some((
                            score,
                            PromptInput { x: fx, y: fy, label: 1 },
                        ));
                    }
                }
            } else if skin_opp >= 22.0
                && fy >= landmarks.chin.y + hh * 0.55
                && (torso_macro_mask[i] > 128 || fy >= landmarks.chin.y + hh * 0.72)
                && (hair_p > 20.0 || hair_opp > 22.0)
            {
                clothes_count += 1;
                if fy >= landmarks.chin.y + hh * 0.74 {
                    let bucket = if dx_rel < -0.50 {
                        0
                    } else if dx_rel <= 0.50 {
                        1
                    } else {
                        2
                    };
                    let score = if let Some(mc) = modal_clothes_rgb {
                        let md = color_dist_sq(rgba, i, mc);
                        if md > 54.0 {
                            continue;
                        }
                        -md - dx_rel.abs() * 4.0
                    } else {
                        (hair_opp.min(120.0) + skin_opp.min(90.0)) - dx_rel.abs() * 6.0
                    };
                    if clothes_buckets[bucket].is_none_or(|(best_s, _)| score > best_s) {
                        clothes_buckets[bucket] = Some((
                            score,
                            PromptInput { x: fx, y: fy, label: 1 },
                        ));
                    }
                }
            }
        }
    }

    let skin_pts: Vec<PromptInput> = skin_buckets.into_iter().flatten().map(|(_, p)| p).collect();
    let clothes_pts: Vec<PromptInput> = clothes_buckets
        .into_iter()
        .flatten()
        .map(|(_, p)| p)
        .collect();
    let mut clothes_palette: Vec<[f64; 3]> = clothes_pts
        .iter()
        .map(|p| sample_patch_rgb(rgba, w, h, p.x, p.y))
        .collect();
    if let Some(mc) = modal_clothes_rgb {
        clothes_palette.push(mc);
    }

    TorsoSeeds {
        skin_pts,
        clothes_pts,
        clothes_palette,
        skin_torso_count,
        clothes_count,
    }
}

/// Traces lower cheek/shoulder/chest curls continuously connected from upper hair and selects up to
/// 3 extra positive MobileSAM points for chest hair rescue.
#[allow(clippy::too_many_arguments)]
fn propagate_chest_hair(
    rgba: &[u8],
    fg: &[u8],
    hair_upper: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    skin_rgb: [f64; 3],
    hair_palette: &[[f64; 3]],
    clothes_palette: &[[f64; 3]],
    eye_dist: f64,
) -> (Vec<bool>, Vec<PromptInput>) {
    let n = w * h;
    let face_cx = 0.5 * (landmarks.eye_l.x + landmarks.eye_r.x);
    let eye_mid_y = 0.5 * (landmarks.eye_l.y + landmarks.eye_r.y);

    let mut hair_geo = vec![false; n];
    let mut visited_steps = vec![u8::MAX; n];
    let mut q: VecDeque<(usize, u8)> = VecDeque::new();
    let chin_seed_limit = landmarks.chin.y + eye_dist * 0.14;
    let neck_seed_limit = landmarks.chin.y + eye_dist * 0.38;

    for y in (eye_mid_y.floor() as usize).min(h)..h {
        let row = y * w;
        let fy = y as f64;
        if fy > neck_seed_limit {
            break;
        }
        for x in 0..w {
            let i = row + x;
            if hair_upper[i] <= 128 || fg[i] <= 8 {
                continue;
            }
            let skin_d = color_dist_sq(rgba, i, skin_rgb);
            let hair_p = palette_min_dist(rgba, i, hair_palette);
            let hair_opp = opponent_min_dist(rgba, i, hair_palette);
            let clothes_ok = if clothes_palette.is_empty() {
                true
            } else {
                let cp = palette_min_dist(rgba, i, clothes_palette);
                let co = opponent_min_dist(rgba, i, clothes_palette);
                hair_opp + 8.0 < co || hair_p + 12.0 < cp
            };
            let is_upper_hair = fy <= chin_seed_limit && skin_d > hair_p + 6.0 && hair_p < 56.0;
            let is_strict_neck_hair = fy > chin_seed_limit
                && (x as f64 - face_cx).abs() <= eye_dist * 1.25
                && skin_d > 26.0
                && skin_d > hair_p + 12.0
                && hair_p < 38.0
                && hair_opp < 34.0
                && clothes_ok;
            // Never unconditionally lock hair_upper spills onto shoulders/sleeves as hair_geo;
            // trace downward along continuous hair pixels instead.
            if is_upper_hair || is_strict_neck_hair {
                hair_geo[i] = true;
                visited_steps[i] = 0;
                q.push_back((i, 0));
            }
        }
    }

    while let Some((idx, steps)) = q.pop_front() {
        let x = idx % w;
        let y = idx / w;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if fg[ni] <= 8
                    || (ny as f64) < eye_mid_y
                    || (nx as f64 - face_cx).abs() > eye_dist * 2.5
                {
                    continue;
                }
                let skin_d = color_dist_sq(rgba, ni, skin_rgb);
                let hair_p = palette_min_dist(rgba, ni, hair_palette);
                let hair_opp = opponent_min_dist(rgba, ni, hair_palette);
                let clothes_ok = if clothes_palette.is_empty() {
                    true
                } else {
                    let cp = palette_min_dist(rgba, ni, clothes_palette);
                    let co = opponent_min_dist(rgba, ni, clothes_palette);
                    hair_opp + 8.0 < co || hair_p + 12.0 < cp
                };

                let is_hair = skin_d > 26.0
                    && skin_d > hair_p + 10.0
                    && hair_p < 44.0
                    && hair_opp < 42.0
                    && clothes_ok;

                let p = ni * 4;
                let luma = 0.299 * rgba[p] as f64 + 0.587 * rgba[p + 1] as f64 + 0.114 * rgba[p + 2] as f64;
                if is_hair {
                    if visited_steps[ni] > 0 {
                        visited_steps[ni] = 0;
                        hair_geo[ni] = true;
                        q.push_back((ni, 0));
                    }
                } else if steps < 2 && luma < 85.0 && skin_d > 28.0 && visited_steps[ni] > steps + 1 {
                    visited_steps[ni] = steps + 1;
                    hair_geo[ni] = true;
                    q.push_back((ni, steps + 1));
                }
            }
        }
    }

    let mut lower_new: [Option<(f64, PromptInput)>; 3] = [None, None, None];
    let mut lower_count = 0usize;
    let chin_cut = landmarks.chin.y + eye_dist * 0.12;
    for (i, &in_geo) in hair_geo.iter().enumerate() {
        if !in_geo || hair_upper[i] > 128 {
            continue;
        }
        let x = i % w;
        let y = i / w;
        let fx = x as f64;
        let fy = y as f64;
        if fy < chin_cut {
            continue;
        }
        lower_count += 1;
        let hair_p = palette_min_dist(rgba, i, hair_palette);
        let bucket = if fx < face_cx - eye_dist * 0.25 {
            0
        } else if fx > face_cx + eye_dist * 0.25 {
            1
        } else {
            2
        };
        let score = (fy - chin_cut) * 0.25 - hair_p;
        if lower_new[bucket].is_none_or(|(best_s, _)| score > best_s) {
            lower_new[bucket] = Some((score, PromptInput { x: fx, y: fy, label: 1 }));
        }
    }

    let extra_pts = if lower_count >= 20 {
        lower_new.into_iter().flatten().map(|(_, p)| p).collect()
    } else {
        Vec::new()
    };
    (hair_geo, extra_pts)
}

type ForegroundAlphaOutput = (Vec<u8>, Option<Vec<u8>>, Option<String>);

fn resolve_foreground_alpha(
    state: &AnalysisState,
    rgba: &[u8],
    w: usize,
    h: usize,
    tolerance: f64,
    u2net_path: Option<&str>,
) -> Result<ForegroundAlphaOutput, String> {
    if tolerance <= 0.0 {
        return Ok((core::foreground(rgba, w, h, 0.0), None, None));
    }
    if let Some(p) = u2net_path {
        let (a, hash) = inference::background_alpha(std::path::Path::new(p), rgba, w, h)?;
        return Ok((a.clone(), Some(a), Some(hash)));
    }
    let bg_path = state
        .registry
        .lock()
        .map_err(|_| "Registry lock poisoned".to_string())?
        .resolve_path(ModelRole::Background);
    if let Some(p) = bg_path {
        if let Ok((a, hash)) = inference::background_alpha(&p, rgba, w, h) {
            return Ok((a.clone(), Some(a), Some(hash)));
        }
    }
    Ok((core::foreground(rgba, w, h, tolerance.clamp(0.0, 180.0)), None, None))
}

/// Runs MobileSAM hybrid segmentation + strict foreground alpha partition across 18 detail IDs
/// Proposes per-part non-overlapping masks using cached MobileSAM encoder+decoder
/// and U2Net/alpha foreground segmentation, collapsing to parent IDs when `detail_mode == "parent"`
/// and keeping 18-part layout unless `hair_split == true` via `propose_parts_with_sam_with_options`.
#[allow(clippy::too_many_arguments)]
pub fn propose_parts_with_sam(
    state: &AnalysisState,
    rgba: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    points_28: &[Landmark28Point],
    face_bbox: Option<[f64; 4]>,
    tolerance: f64,
    u2net_path: Option<&str>,
    detail_mode: &str,
) -> Result<PartsProposalOutput, String> {
    propose_parts_with_sam_with_options(
        state,
        rgba,
        w,
        h,
        landmarks,
        points_28,
        face_bbox,
        tolerance,
        u2net_path,
        detail_mode,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn propose_parts_with_sam_with_options(
    state: &AnalysisState,
    rgba: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    points_28: &[Landmark28Point],
    face_bbox: Option<[f64; 4]>,
    tolerance: f64,
    u2net_path: Option<&str>,
    detail_mode: &str,
    hair_split: bool,
) -> Result<PartsProposalOutput, String> {
    let image_hash = hash_rgba_bytes(rgba, w, h);

    // 1. Background removal: explicit u2net_path -> registered Background role -> border flood / alpha.
    let mut model_hashes = HashMap::new();
    let (fg, u2_alpha, bg_hash) =
        resolve_foreground_alpha(state, rgba, w, h, tolerance, u2net_path)?;
    if let Some(h_u2) = bg_hash {
        model_hashes.insert("background".into(), h_u2);
    }
    let geo_fallback = core::separate(rgba, w, h, landmarks, tolerance.clamp(0.0, 180.0), u2_alpha.as_deref());

    // 2. Encode full image + face crop via MobileSAM (reusing cache!).
    let full_cache = get_or_encode_full_sam(state, rgba, w, h, &image_hash)?;
    model_hashes.insert("part_encoder".into(), full_cache.model_hash.clone());
    let crop_rect = compute_face_crop_rect(landmarks, w, h);
    let crop_cache = get_or_encode_crop_sam(state, rgba, w, h, &image_hash, crop_rect)?;

    let eye_dist = ((landmarks.eye_l.x - landmarks.eye_r.x).powi(2)
        + (landmarks.eye_l.y - landmarks.eye_r.y).powi(2))
    .sqrt()
    .max(24.0);
    let face_cx = 0.5 * (landmarks.eye_l.x + landmarks.eye_r.x);
    let fbox = face_bbox.unwrap_or([
        (face_cx - eye_dist * 1.15).clamp(0.0, (w - 1) as f64),
        (landmarks.crown.y + (landmarks.eye_r.y - landmarks.crown.y) * 0.25)
            .clamp(0.0, (h - 1) as f64),
        (face_cx + eye_dist * 1.15).clamp(0.0, (w - 1) as f64),
        (landmarks.chin.y + 4.0).clamp(0.0, (h - 1) as f64),
    ]);

    let nose_pt = if points_28.len() == 28 {
        PromptInput {
            x: points_28[23].x,
            y: points_28[23].y,
            label: 1,
        }
    } else {
        PromptInput {
            x: face_cx,
            y: 0.5 * (landmarks.eye_l.y + landmarks.mouth.y),
            label: 1,
        }
    };

    let skin_rgb = sample_patch_rgb(rgba, w, h, nose_pt.x, nose_pt.y);
    let hair_seed_y = (landmarks.crown.y + eye_dist * 0.12).clamp(0.0, (h - 1) as f64);
    let hair_rgb = sample_patch_rgb(rgba, w, h, landmarks.crown.x, hair_seed_y);
    let hair_palette = build_hair_palette(rgba, &fg, w, h, landmarks, skin_rgb, eye_dist);

    let mut sam_masks: HashMap<&'static str, (Vec<u8>, f32)> = HashMap::new();
    let decoder_sha: String;
    let body_macro_mask: Vec<u8>;
    let torso_seeds: TorsoSeeds;
    let hair_geo: Vec<bool>;
    let mut has_exposed_torso_skin = false;

    {
        let mut reg = state
            .registry
            .lock()
            .map_err(|_| "Registry lock poisoned".to_string())?;

        // Macro face skin proposal
        let face_dec = decode_sam_prompts(
            &mut reg,
            &full_cache,
            &[
                nose_pt,
                PromptInput { x: landmarks.crown.x, y: landmarks.crown.y, label: 0 },
                PromptInput { x: landmarks.chin.x, y: (landmarks.chin.y + eye_dist * 0.35).min((h - 1) as f64), label: 0 },
            ],
            Some(fbox),
        )?;
        decoder_sha = face_dec.decoder_sha256.clone();
        let (face_raw, face_iou) =
            pick_best_plausible_slot(&face_dec.masks, &face_dec.iou_scores, &fg, 0.55, 64);
        let face_clean = clean_sam_mask_in_fg(&face_raw, &fg, w, h);

        let neck_y = (landmarks.chin.y + eye_dist * 0.28).clamp(0.0, (h - 1) as f64);
        let body_y = (landmarks.chin.y + eye_dist * 1.1).clamp(0.0, (h - 1) as f64);
        let neck_cx = 0.5 * (face_cx + landmarks.chin.x);

        // Macro torso proposal decoded first so we can discover bare shoulders & modal garment
        // seeds and pass them as explicit negatives into the initial upper hair proposal.
        let body_dec = decode_sam_prompts(
            &mut reg,
            &full_cache,
            &[
                PromptInput { x: landmarks.chin.x, y: body_y, label: 1 },
                PromptInput {
                    x: (landmarks.chin.x - eye_dist * 1.2).clamp(0.0, (w - 1) as f64),
                    y: body_y,
                    label: 1,
                },
                PromptInput {
                    x: (landmarks.chin.x + eye_dist * 1.2).clamp(0.0, (w - 1) as f64),
                    y: body_y,
                    label: 1,
                },
                PromptInput { x: nose_pt.x, y: nose_pt.y, label: 0 },
            ],
            None,
        )?;
        let (body_raw, mut body_iou) =
            pick_best_plausible_slot(&body_dec.masks, &body_dec.iou_scores, &fg, 0.85, 64);
        body_macro_mask = clean_sam_mask_in_fg(&body_raw, &fg, w, h);

        // Use a conservative central-throat exclusion mask so shoulders are discovered cleanly for `body` skin seeds
        let mut throat_guard = vec![0u8; w * h];
        let neck_cx_lo = ((face_cx - eye_dist * 0.38).floor() as isize).max(0) as usize;
        let neck_cx_hi = ((face_cx + eye_dist * 0.38).ceil() as isize).min((w as isize).saturating_sub(1)) as usize;
        let neck_cy_lo = (landmarks.chin.y.floor() as usize).min(h);
        let neck_cy_hi = ((landmarks.chin.y + eye_dist * 0.42).ceil() as usize).min(h);
        for y in neck_cy_lo..neck_cy_hi {
            let row = y * w;
            for x in neck_cx_lo..=neck_cx_hi {
                throat_guard[row + x] = 255;
            }
        }
        let empty_hair_hint = vec![0u8; w * h];
        torso_seeds = find_torso_seeds(
            rgba,
            &fg,
            &face_clean,
            &empty_hair_hint,
            &throat_guard,
            &body_macro_mask,
            w,
            h,
            landmarks,
            skin_rgb,
            &hair_palette,
            eye_dist,
        );

        // Multi-point initial hair proposal with torso skin + modal garment negatives
        let mut base_hair_positives = vec![PromptInput {
            x: landmarks.crown.x,
            y: hair_seed_y,
            label: 1,
        }];
        let extra_hair_candidates = [
            (face_cx - eye_dist * 0.55, landmarks.crown.y + eye_dist * 0.25),
            (face_cx + eye_dist * 0.55, landmarks.crown.y + eye_dist * 0.25),
            (landmarks.eye_r.x - eye_dist * 0.45, landmarks.eye_r.y - eye_dist * 0.12),
            (landmarks.eye_l.x + eye_dist * 0.45, landmarks.eye_l.y - eye_dist * 0.12),
        ];
        for (hx, hy) in extra_hair_candidates {
            if hx >= 0.0 && hy >= 0.0 && hx < w as f64 && hy < h as f64 {
                let idx = (hy as usize) * w + (hx as usize);
                if fg[idx] > 128
                    && color_dist_sq(rgba, idx, hair_rgb) < color_dist_sq(rgba, idx, skin_rgb) + 12.0
                {
                    base_hair_positives.push(PromptInput {
                        x: hx,
                        y: hy,
                        label: 1,
                    });
                }
            }
        }
        let mut hair_prompts = base_hair_positives.clone();
        hair_prompts.extend([
            PromptInput { x: nose_pt.x, y: nose_pt.y, label: 0 },
            PromptInput { x: landmarks.mouth.x, y: landmarks.mouth.y, label: 0 },
            PromptInput { x: landmarks.eye_r.x, y: landmarks.eye_r.y, label: 0 },
            PromptInput { x: landmarks.eye_l.x, y: landmarks.eye_l.y, label: 0 },
            PromptInput { x: landmarks.chin.x, y: landmarks.chin.y, label: 0 },
            PromptInput { x: landmarks.chin.x, y: neck_y, label: 0 },
        ]);
        for sp in &torso_seeds.skin_pts {
            hair_prompts.push(PromptInput { x: sp.x, y: sp.y, label: 0 });
        }
        for cp in &torso_seeds.clothes_pts {
            hair_prompts.push(PromptInput { x: cp.x, y: cp.y, label: 0 });
        }
        if torso_seeds.clothes_pts.is_empty() {
            let body_idx = (body_y as usize) * w + (landmarks.chin.x.clamp(0.0, (w - 1) as f64) as usize);
            if palette_min_dist(rgba, body_idx, &hair_palette) > 42.0 {
                hair_prompts.push(PromptInput {
                    x: landmarks.chin.x,
                    y: body_y,
                    label: 0,
                });
            }
        }
        let hair_dec = decode_sam_prompts(&mut reg, &full_cache, &hair_prompts, None)?;
        let (hair_raw, mut hair_iou) =
            pick_best_plausible_slot(&hair_dec.masks, &hair_dec.iou_scores, &fg, 0.72, 64);
        let hair_upper = clean_sam_mask_in_fg(&hair_raw, &fg, w, h);

        // Neck proposal with explicit negatives on bare shoulders, lateral shoulder guards & clothing
        let mut neck_prompts = vec![
            PromptInput { x: landmarks.chin.x, y: neck_y, label: 1 },
            PromptInput { x: nose_pt.x, y: nose_pt.y, label: 0 },
            PromptInput {
                x: landmarks.chin.x,
                y: (landmarks.chin.y + eye_dist * 0.9).clamp(0.0, (h - 1) as f64),
                label: 0,
            },
        ];
        for sx in [
            (neck_cx - eye_dist * 0.78).clamp(0.0, (w - 1) as f64),
            (neck_cx + eye_dist * 0.78).clamp(0.0, (w - 1) as f64),
        ] {
            let sy = (landmarks.chin.y + eye_dist * 0.52).clamp(0.0, (h - 1) as f64);
            let si = (sy as usize) * w + (sx as usize);
            if fg[si] > 128 && color_dist_sq(rgba, si, skin_rgb) < 48.0 {
                neck_prompts.push(PromptInput { x: sx, y: sy, label: 0 });
            }
        }
        for sp in &torso_seeds.skin_pts {
            neck_prompts.push(PromptInput { x: sp.x, y: sp.y, label: 0 });
        }
        for cp in &torso_seeds.clothes_pts {
            neck_prompts.push(PromptInput { x: cp.x, y: cp.y, label: 0 });
        }
        let neck_dec = decode_sam_prompts(&mut reg, &full_cache, &neck_prompts, None)?;
        let (neck_raw, neck_iou) =
            pick_best_plausible_slot(&neck_dec.masks, &neck_dec.iou_scores, &fg, 0.16, 24);
        let mut neck_clean = clean_sam_mask_in_fg(&neck_raw, &fg, w, h);
        // Keep only the main neck component closest to `(chin.x, neck_y)` so detached
        // shoulder sliver leaks don't pollute `neck`.
        let neck_bool: Vec<bool> = neck_clean.iter().map(|&v| v > 128).collect();
        let neck_comps = find_connected_components_4(&neck_bool, w, h);
        if neck_comps.len() > 1 {
            let best_comp = neck_comps.iter().min_by(|a, b| {
                let da = a
                    .iter()
                    .map(|&idx| {
                        let dx = (idx % w) as f64 - landmarks.chin.x;
                        let dy = (idx / w) as f64 - neck_y;
                        dx * dx + dy * dy
                    })
                    .fold(f64::INFINITY, f64::min);
                let db = b
                    .iter()
                    .map(|&idx| {
                        let dx = (idx % w) as f64 - landmarks.chin.x;
                        let dy = (idx / w) as f64 - neck_y;
                        dx * dx + dy * dy
                    })
                    .fold(f64::INFINITY, f64::min);
                da.total_cmp(&db)
            });
            if let Some(keep) = best_comp {
                let mut filtered = vec![0u8; w * h];
                for &idx in keep {
                    filtered[idx] = neck_clean[idx];
                }
                neck_clean = filtered;
            }
        }

        // Geodesic downward hair tracing + 2nd MobileSAM chest-hair pass when lower locks are found
        let (geo_traced, lower_hair_pts) = propagate_chest_hair(
            rgba,
            &fg,
            &hair_upper,
            w,
            h,
            landmarks,
            skin_rgb,
            &hair_palette,
            &torso_seeds.clothes_palette,
            eye_dist,
        );
        hair_geo = geo_traced;

        let mut hair_combined = hair_upper.clone();
        if !lower_hair_pts.is_empty() {
            let mut refined_prompts = base_hair_positives;
            refined_prompts.extend(lower_hair_pts.iter().copied());
            refined_prompts.extend([
                PromptInput { x: nose_pt.x, y: nose_pt.y, label: 0 },
                PromptInput { x: landmarks.mouth.x, y: landmarks.mouth.y, label: 0 },
                PromptInput { x: landmarks.eye_r.x, y: landmarks.eye_r.y, label: 0 },
                PromptInput { x: landmarks.eye_l.x, y: landmarks.eye_l.y, label: 0 },
                PromptInput { x: landmarks.chin.x, y: landmarks.chin.y, label: 0 },
                PromptInput { x: landmarks.chin.x, y: neck_y, label: 0 },
            ]);
            for sp in &torso_seeds.skin_pts {
                refined_prompts.push(PromptInput { x: sp.x, y: sp.y, label: 0 });
            }
            for cp in &torso_seeds.clothes_pts {
                refined_prompts.push(PromptInput { x: cp.x, y: cp.y, label: 0 });
            }
            let refined_dec = decode_sam_prompts(&mut reg, &full_cache, &refined_prompts, None)?;
            let (ref_raw, ref_iou) =
                pick_best_plausible_slot(&refined_dec.masks, &refined_dec.iou_scores, &fg, 0.72, 64);
            let ref_clean = clean_sam_mask_in_fg(&ref_raw, &fg, w, h);
            hair_iou = hair_iou.max(ref_iou);
            for (i, hc) in hair_combined.iter_mut().enumerate() {
                if fg[i] <= 8 {
                    continue;
                }
                if hair_geo[i] {
                    *hc = 255;
                } else if ref_clean[i] > 128 && *hc <= 128 {
                    let fy = (i / w) as f64;
                    if fy <= landmarks.chin.y + eye_dist * 0.15 {
                        *hc = 255;
                    } else {
                        let skin_d = color_dist_sq(rgba, i, skin_rgb);
                        let hair_p = palette_min_dist(rgba, i, &hair_palette);
                        let hair_c = chroma_min_dist(rgba, i, &hair_palette);
                        let clothes_beat = !torso_seeds.clothes_palette.is_empty() && {
                            let cp = palette_min_dist(rgba, i, &torso_seeds.clothes_palette);
                            let cc = chroma_min_dist(rgba, i, &torso_seeds.clothes_palette);
                            cp + 8.0 < hair_p && cc + 2.0 < hair_c
                        };
                        if skin_d > hair_p + 10.0 && !clothes_beat {
                            *hc = 255;
                        }
                    }
                }
            }
        } else {
            for (i, hc) in hair_combined.iter_mut().enumerate() {
                if hair_geo[i] && fg[i] > 8 {
                    *hc = 255;
                }
            }
        }

        sam_masks.insert("face", (face_clean, face_iou));
        sam_masks.insert("hair_front", (hair_combined, hair_iou));
        sam_masks.insert("neck", (neck_clean, neck_iou));

        let mut body_final_mask = body_macro_mask.clone();
        if hair_split && w.max(h) >= 200 {
            if torso_seeds.skin_torso_count >= 64 && !torso_seeds.skin_pts.is_empty() {
                let mut skin_neg = vec![
                    PromptInput { x: nose_pt.x, y: nose_pt.y, label: 0 },
                    PromptInput { x: landmarks.chin.x, y: landmarks.chin.y, label: 0 },
                    PromptInput { x: landmarks.chin.x, y: neck_y, label: 0 },
                    PromptInput { x: landmarks.crown.x, y: hair_seed_y, label: 0 },
                ];
                for cp in &torso_seeds.clothes_pts {
                    skin_neg.push(PromptInput { x: cp.x, y: cp.y, label: 0 });
                }
                for hp in &lower_hair_pts {
                    skin_neg.push(PromptInput { x: hp.x, y: hp.y, label: 0 });
                }
                let mut skin_prompts = torso_seeds.skin_pts.clone();
                skin_prompts.extend(skin_neg.iter().copied());
                let s_dec = decode_sam_prompts(&mut reg, &full_cache, &skin_prompts, None)?;
                let (s_raw, mut s_iou) =
                    pick_best_plausible_slot(&s_dec.masks, &s_dec.iou_scores, &fg, 0.35, 48);
                let mut merged_skin = clean_sam_mask_in_fg(&s_raw, &fg, w, h);

                // Decode uncovered shoulder/collarbone islands individually so hair strands dividing
                // shoulders don't drop one side to coarse geometry fallback.
                if torso_seeds.skin_pts.len() > 1 {
                    for &sp in &torso_seeds.skin_pts {
                        let px = (sp.x.round() as isize).clamp(0, w as isize - 1) as usize;
                        let py = (sp.y.round() as isize).clamp(0, h as isize - 1) as usize;
                        if merged_skin[py * w + px] > 128 {
                            continue;
                        }
                        let mut one_prompts = vec![sp];
                        one_prompts.extend(skin_neg.iter().copied());
                        let one_dec = decode_sam_prompts(&mut reg, &full_cache, &one_prompts, None)?;
                        let (one_raw, one_iou) =
                            pick_best_plausible_slot(&one_dec.masks, &one_dec.iou_scores, &fg, 0.25, 36);
                        let one_clean = clean_sam_mask_in_fg(&one_raw, &fg, w, h);
                        s_iou = s_iou.max(one_iou);
                        for (dst, &v) in merged_skin.iter_mut().zip(&one_clean) {
                            *dst = (*dst).max(v);
                        }
                    }
                }

                body_final_mask = merged_skin;
                body_iou = s_iou;
                has_exposed_torso_skin = true;
            }
            if torso_seeds.clothes_count >= 120 && !torso_seeds.clothes_pts.is_empty() {
                let mut cl_neg = vec![
                    PromptInput { x: nose_pt.x, y: nose_pt.y, label: 0 },
                    PromptInput { x: landmarks.chin.x, y: landmarks.chin.y, label: 0 },
                    PromptInput { x: landmarks.chin.x, y: neck_y, label: 0 },
                    PromptInput { x: landmarks.crown.x, y: hair_seed_y, label: 0 },
                ];
                for sp in &torso_seeds.skin_pts {
                    cl_neg.push(PromptInput { x: sp.x, y: sp.y, label: 0 });
                }
                for hp in &lower_hair_pts {
                    cl_neg.push(PromptInput { x: hp.x, y: hp.y, label: 0 });
                }
                let mut cl_prompts = torso_seeds.clothes_pts.clone();
                cl_prompts.extend(cl_neg.iter().copied());
                let cl_dec = decode_sam_prompts(&mut reg, &full_cache, &cl_prompts, None)?;
                let (cl_raw, mut cl_iou) =
                    pick_best_plausible_slot(&cl_dec.masks, &cl_dec.iou_scores, &fg, 0.85, 64);
                let mut merged_clothes = clean_sam_mask_in_fg(&cl_raw, &fg, w, h);

                // Rescue separate garment sleeves when front hair locks split the sweater visually.
                if torso_seeds.clothes_pts.len() > 1 {
                    for &cp in &torso_seeds.clothes_pts {
                        let px = (cp.x.round() as isize).clamp(0, w as isize - 1) as usize;
                        let py = (cp.y.round() as isize).clamp(0, h as isize - 1) as usize;
                        if merged_clothes[py * w + px] > 128 {
                            continue;
                        }
                        let mut one_prompts = vec![cp];
                        one_prompts.extend(cl_neg.iter().copied());
                        let one_dec = decode_sam_prompts(&mut reg, &full_cache, &one_prompts, None)?;
                        let (one_raw, one_iou) =
                            pick_best_plausible_slot(&one_dec.masks, &one_dec.iou_scores, &fg, 0.65, 64);
                        let one_clean = clean_sam_mask_in_fg(&one_raw, &fg, w, h);
                        cl_iou = cl_iou.max(one_iou);
                        for (dst, &v) in merged_clothes.iter_mut().zip(&one_clean) {
                            *dst = (*dst).max(v);
                        }
                    }
                }
                sam_masks.insert("clothes", (merged_clothes, cl_iou));
            }
        }
        sam_masks.insert("body", (body_final_mask, body_iou));

        // Small high-res facial features decoded on the ZOOMED face crop!
        let crop_to_prompt = |x: f64, y: f64, label: i32| PromptInput {
            x: (x - crop_rect.x0 as f64).clamp(0.0, (crop_rect.w - 1) as f64),
            y: (y - crop_rect.y0 as f64).clamp(0.0, (crop_rect.h - 1) as f64),
            label,
        };
        let to_crop_box = |b: [f64; 4]| [
            (b[0] - crop_rect.x0 as f64).clamp(0.0, (crop_rect.w - 1) as f64),
            (b[1] - crop_rect.y0 as f64).clamp(0.0, (crop_rect.h - 1) as f64),
            (b[2] - crop_rect.x0 as f64).clamp(0.0, (crop_rect.w - 1) as f64),
            (b[3] - crop_rect.y0 as f64).clamp(0.0, (crop_rect.h - 1) as f64),
        ];

        for (part_key, eye_pt, lm_indices) in [
            ("eyeR_all", landmarks.eye_r, &[11usize, 12, 13, 14, 15, 16][..]),
            ("eyeL_all", landmarks.eye_l, &[17usize, 18, 19, 20, 21, 22][..]),
        ] {
            let ebox = box_from_points_28(points_28, lm_indices, 0.22, w, h).unwrap_or([
                (eye_pt.x - eye_dist * 0.25).clamp(0.0, (w - 1) as f64),
                (eye_pt.y - eye_dist * 0.16).clamp(0.0, (h - 1) as f64),
                (eye_pt.x + eye_dist * 0.25).clamp(0.0, (w - 1) as f64),
                (eye_pt.y + eye_dist * 0.16).clamp(0.0, (h - 1) as f64),
            ]);
            let dec = decode_sam_prompts(
                &mut reg,
                &crop_cache,
                &[
                    crop_to_prompt(eye_pt.x, eye_pt.y, 1),
                    crop_to_prompt(nose_pt.x, nose_pt.y, 0),
                ],
                Some(to_crop_box(ebox)),
            )?;
            let best_i = (1..4)
                .max_by(|&a, &b| dec.iou_scores[a].total_cmp(&dec.iou_scores[b]))
                .unwrap_or(1);
            let full_eye = uncrop_mask(&dec.masks[best_i], crop_rect.w, crop_rect.h, crop_rect.x0, crop_rect.y0, w, h);
            sam_masks.insert(part_key, (full_eye, dec.iou_scores[best_i]));
        }

        for (part_key, lm_indices, default_pt) in [
            ("browR", &[5usize, 6, 7][..], landmarks.eye_r),
            ("browL", &[8usize, 9, 10][..], landmarks.eye_l),
        ] {
            let bbox_opt = box_from_points_28(points_28, lm_indices, 0.2, w, h);
            let center_pt = if points_28.len() == 28 {
                let mid = &points_28[lm_indices[1]];
                crop_to_prompt(mid.x, mid.y, 1)
            } else {
                crop_to_prompt(default_pt.x, (default_pt.y - eye_dist * 0.28).max(0.0), 1)
            };
            let dec = decode_sam_prompts(
                &mut reg,
                &crop_cache,
                &[center_pt, crop_to_prompt(default_pt.x, default_pt.y, 0)],
                bbox_opt.map(to_crop_box),
            )?;
            let best_i = (1..4)
                .max_by(|&a, &b| dec.iou_scores[a].total_cmp(&dec.iou_scores[b]))
                .unwrap_or(1);
            let full_brow = uncrop_mask(&dec.masks[best_i], crop_rect.w, crop_rect.h, crop_rect.x0, crop_rect.y0, w, h);
            sam_masks.insert(part_key, (full_brow, dec.iou_scores[best_i]));
        }

        let mbox = box_from_points_28(points_28, &[24usize, 25, 26, 27], 0.25, w, h).unwrap_or([
            (landmarks.mouth.x - eye_dist * 0.22).clamp(0.0, (w - 1) as f64),
            (landmarks.mouth.y - eye_dist * 0.1).clamp(0.0, (h - 1) as f64),
            (landmarks.mouth.x + eye_dist * 0.22).clamp(0.0, (w - 1) as f64),
            (landmarks.mouth.y + eye_dist * 0.1).clamp(0.0, (h - 1) as f64),
        ]);
        let mdec = decode_sam_prompts(
            &mut reg,
            &crop_cache,
            &[
                crop_to_prompt(landmarks.mouth.x, landmarks.mouth.y, 1),
                crop_to_prompt(landmarks.chin.x, landmarks.chin.y, 0),
            ],
            Some(to_crop_box(mbox)),
        )?;
        let m_best = (1..4)
            .max_by(|&a, &b| mdec.iou_scores[a].total_cmp(&mdec.iou_scores[b]))
            .unwrap_or(1);
        let full_mouth = uncrop_mask(&mdec.masks[m_best], crop_rect.w, crop_rect.h, crop_rect.x0, crop_rect.y0, w, h);
        sam_masks.insert("mouth_all", (full_mouth, mdec.iou_scores[m_best]));
    }

    model_hashes.insert("part_decoder".into(), decoder_sha);

    // 3. Strictly partition every foreground pixel (`fg[i] > 0`) into the 18 IDS cleanly.
    let n = w * h;
    let mut part_masks: Vec<Vec<u8>> = (0..IDS.len()).map(|_| vec![0u8; n]).collect();
    let id_index: HashMap<&str, usize> = IDS.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut sam_hits = vec![0usize; IDS.len()];
    let mut total_hits = vec![0usize; IDS.len()];

    let geo_owner = |i: usize| -> usize {
        geo_fallback
            .iter()
            .enumerate()
            .max_by_key(|(_, m)| m[i])
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    };

    let eye_rx = eye_dist * 0.24;
    let eye_ry = eye_dist * 0.15;
    let hh = (landmarks.chin.y - landmarks.crown.y).max(eye_dist * 1.4);
    let neck_skin_rgb = sample_patch_rgb(
        rgba,
        w,
        h,
        landmarks.chin.x,
        (landmarks.chin.y + eye_dist * 0.28).clamp(0.0, (h - 1) as f64),
    );
    let skin_palette = [skin_rgb, neck_skin_rgb];
    let clothes_candidate_active =
        hair_split && w.max(h) >= 200 && torso_seeds.clothes_count >= 120 && sam_masks.contains_key("clothes");
    let mut clothes_mask = vec![0u8; n];
    let mut clothes_sam_hits = 0usize;
    let mut clothes_total_hits = 0usize;

    for y in 0..h {
        let row = y * w;
        let fy = y as f64;
        for x in 0..w {
            let i = row + x;
            let a = fg[i];
            if a == 0 {
                continue;
            }
            let fx = x as f64;
            let mut chosen: Option<&str> = None;

            let in_face = sam_masks.get("face").is_some_and(|(m, _)| m[i] > 128);
            let in_hair = sam_masks.get("hair_front").is_some_and(|(m, _)| m[i] > 128);
            let in_neck = sam_masks.get("neck").is_some_and(|(m, _)| m[i] > 128);
            let in_body = sam_masks.get("body").is_some_and(|(m, _)| m[i] > 128);
            let in_clothes = sam_masks.get("clothes").is_some_and(|(m, _)| m[i] > 128);
            let skin_d = color_dist_sq(rgba, i, skin_rgb);
            let skin_opp_d = opponent_min_dist(rgba, i, &skin_palette);
            let hair_p_d = palette_min_dist(rgba, i, &hair_palette);
            let hair_opp_d = opponent_min_dist(rgba, i, &hair_palette);
            let clothes_opp_d = if torso_seeds.clothes_palette.is_empty() {
                f64::INFINITY
            } else {
                opponent_min_dist(rgba, i, &torso_seeds.clothes_palette)
            };

            let clothes_beats_hair = !hair_geo[i]
                && clothes_candidate_active
                && in_clothes
                && fy > landmarks.chin.y
                && clothes_opp_d + 4.0 < hair_opp_d;

            if sam_masks.get("eyeR_all").is_some_and(|(m, _)| m[i] > 128) {
                let dx = fx - landmarks.eye_r.x;
                let dy = fy - landmarks.eye_r.y;
                if dy < -eye_ry * 0.48 {
                    chosen = Some("eyeR_upper");
                } else if dy > eye_ry * 0.60 {
                    chosen = Some("eyeR_lower");
                } else if dx.abs() < eye_rx * 0.48 {
                    chosen = Some("eyeR_iris");
                } else {
                    chosen = Some("eyeR_white");
                }
            } else if sam_masks.get("eyeL_all").is_some_and(|(m, _)| m[i] > 128) {
                let dx = fx - landmarks.eye_l.x;
                let dy = fy - landmarks.eye_l.y;
                if dy < -eye_ry * 0.48 {
                    chosen = Some("eyeL_upper");
                } else if dy > eye_ry * 0.60 {
                    chosen = Some("eyeL_lower");
                } else if dx.abs() < eye_rx * 0.48 {
                    chosen = Some("eyeL_iris");
                } else {
                    chosen = Some("eyeL_white");
                }
            } else if sam_masks.get("browR").is_some_and(|(m, _)| m[i] > 128) {
                chosen = Some("browR");
            } else if sam_masks.get("browL").is_some_and(|(m, _)| m[i] > 128) {
                chosen = Some("browL");
            } else if sam_masks.get("mouth_all").is_some_and(|(m, _)| m[i] > 128) {
                let dy = fy - landmarks.mouth.y;
                if dy < -eye_dist * 0.018 {
                    chosen = Some("mouth_upper");
                } else if dy > eye_dist * 0.018 {
                    chosen = Some("mouth_lower");
                } else {
                    chosen = Some("mouth_inner");
                }
            } else if in_face && (!in_hair || skin_d <= hair_p_d + 12.0) {
                chosen = Some("face");
            } else if in_hair
                && !(fy > landmarks.chin.y + eye_dist * 0.18
                    && skin_opp_d < 28.0
                    && skin_opp_d + 10.0 < hair_opp_d)
                && !clothes_beats_hair
            {
                if fy < landmarks.chin.y + eye_dist * 0.15 {
                    chosen = Some("hair_front");
                } else {
                    chosen = Some("hair_back");
                }
            } else if in_neck && {
                let neck_cx = 0.5 * (face_cx + landmarks.chin.x);
                let nx = (fx - neck_cx) / (eye_dist * 0.58);
                let ny = (fy - (landmarks.chin.y + hh * 0.20)).max(0.0) / (hh * 0.42);
                let outside_neck_oval = nx * nx + ny * ny > 1.0;
                !(has_exposed_torso_skin && in_body && outside_neck_oval)
            } {
                chosen = Some("neck");
            } else if in_body || in_clothes || body_macro_mask[i] > 128 {
                if hair_geo[i]
                    || (!in_clothes && hair_opp_d + 4.0 < clothes_opp_d && hair_opp_d + 8.0 < skin_opp_d)
                    || (hair_p_d < 36.0
                        && hair_opp_d + 14.0 < clothes_opp_d
                        && hair_opp_d + 14.0 < skin_opp_d)
                {
                    chosen = Some(if fy < landmarks.chin.y + eye_dist * 0.15 {
                        "hair_front"
                    } else {
                        "hair_back"
                    });
                } else if clothes_candidate_active {
                    let is_skin_px = has_exposed_torso_skin
                        && ((in_body
                            && !in_clothes
                            && skin_opp_d < 32.0
                            && skin_opp_d <= clothes_opp_d + 4.0)
                            || (skin_opp_d < 28.0
                                && skin_opp_d + 8.0 < clothes_opp_d
                                && fy < landmarks.chin.y + hh * 0.90));
                    if is_skin_px {
                        chosen = Some("body");
                    } else if in_clothes || clothes_opp_d + 6.0 < hair_opp_d {
                        chosen = Some("clothes");
                    } else {
                        chosen = Some("hair_back");
                    }
                } else {
                    chosen = Some("body");
                }
            }

            let mut route_to_clothes = false;
            let mut target_idx = 0usize;
            if let Some(cid) = chosen {
                if cid == "clothes" {
                    route_to_clothes = true;
                    clothes_sam_hits += usize::from(in_clothes);
                } else {
                    let idx = id_index[cid];
                    sam_hits[idx] += 1;
                    target_idx = idx;
                }
            } else {
                let mut idx = geo_owner(i);
                // Prevent exposed collarbone/shoulder/jaw skin or garments missed by SAM from falling into hair_back,
                // and route hair vs garment continuously without a hard horizontal y-cut.
                if (idx == id_index["hair_back"] || idx == id_index["body"])
                    && fy > landmarks.eye_l.y - eye_dist * 0.15
                    && skin_opp_d + 8.0 < hair_opp_d
                    && skin_d < 52.0
                    && (!clothes_candidate_active || skin_opp_d <= clothes_opp_d + 4.0)
                {
                    if (fx - face_cx).abs() < eye_dist * 0.52
                        && fy >= landmarks.chin.y
                        && fy <= landmarks.chin.y + hh * 0.45
                    {
                        idx = id_index["neck"];
                    } else if fy > landmarks.chin.y + eye_dist * 0.02 {
                        idx = id_index["body"];
                    } else if (fx - face_cx).abs() < eye_dist * 1.15 && fy <= landmarks.chin.y {
                        idx = id_index["face"];
                    }
                } else if clothes_candidate_active
                    && (idx == id_index["hair_back"] || idx == id_index["body"])
                {
                    if hair_geo[i] || (hair_p_d < 44.0 && hair_opp_d + 4.0 < clothes_opp_d) {
                        idx = id_index["hair_back"];
                    } else if fy > landmarks.chin.y + hh * 0.40
                        && (fx - face_cx).abs() <= eye_dist * 1.65
                        && (in_clothes
                            || (clothes_opp_d < 34.0
                                && clothes_opp_d + 8.0 < hair_opp_d
                                && clothes_opp_d + 6.0 < skin_opp_d))
                    {
                        route_to_clothes = true;
                    } else if idx == id_index["body"] {
                        idx = id_index["hair_back"];
                    }
                }
                target_idx = idx;
            }

            if route_to_clothes {
                clothes_mask[i] = a;
                clothes_total_hits += 1;
            } else {
                part_masks[target_idx][i] = a;
                total_hits[target_idx] += 1;
            }
        }
    }

    let body_idx_slot = id_index["body"];
    let body_px_count = part_masks[body_idx_slot].iter().filter(|&&v| v > 8).count();
    let clothes_extra = if clothes_candidate_active && body_px_count >= 24 && clothes_total_hits >= 64 {
        Some(ClothesExtra {
            mask: clothes_mask,
            sam_hits: clothes_sam_hits,
            total_hits: clothes_total_hits,
            confidence: sam_masks
                .get("clothes")
                .map(|(_, c)| *c as f64)
                .unwrap_or(0.0),
        })
    } else {
        for (i, &v) in clothes_mask.iter().enumerate() {
            if v > 0 {
                part_masks[body_idx_slot][i] = part_masks[body_idx_slot][i].max(v);
                total_hits[body_idx_slot] += 1;
            }
        }
        None
    };

    // 4 & 5. Optional parent collapsing + multi-layer hair split + clothes layer + diagnostics.
    finalize_part_proposal(
        rgba,
        w,
        h,
        landmarks,
        part_masks,
        sam_hits,
        total_hits,
        &sam_masks,
        clothes_extra,
        detail_mode,
        hair_split,
        "mobilesam-vit-t+landmark-partition-v1",
        model_hashes,
        if hair_split {
            "sam-hybrid-hair-split-v3"
        } else {
            "sam-hybrid-partition-v1"
        },
        image_hash,
    )
}

#[allow(clippy::too_many_arguments)]
fn finalize_part_proposal(
    rgba: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    final_masks: Vec<Vec<u8>>,
    sam_hits: Vec<usize>,
    total_hits: Vec<usize>,
    sam_masks: &HashMap<&'static str, (Vec<u8>, f32)>,
    clothes_extra: Option<ClothesExtra>,
    detail_mode: &str,
    hair_split: bool,
    provider: &str,
    model_hashes: HashMap<String, String>,
    postprocess_version: &str,
    image_hash: String,
) -> Result<PartsProposalOutput, String> {
    let n = w * h;
    let id_index: HashMap<&str, usize> = IDS.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut final_ids: Vec<String> = IDS.iter().map(|s| s.to_string()).collect();
    let mut final_masks = final_masks;
    let mut final_confidences = vec![0.0f64; IDS.len()];
    let mut final_provenances: Vec<String> = Vec::with_capacity(IDS.len());

    let conf_lookup = |id: &str| -> f64 {
        let key = if id.starts_with("eyeR_") {
            "eyeR_all"
        } else if id.starts_with("eyeL_") {
            "eyeL_all"
        } else if id.starts_with("mouth_") {
            "mouth_all"
        } else if id.starts_with("hair_") {
            "hair_front"
        } else {
            id
        };
        sam_masks.get(key).map(|(_, s)| *s as f64).unwrap_or(0.0)
    };

    for (idx, &id) in IDS.iter().enumerate() {
        final_confidences[idx] = conf_lookup(id);
        let hits = sam_hits[idx];
        let tot = total_hits[idx].max(1);
        final_provenances.push(if hits >= 12 && (hits as f64 / tot as f64) >= 0.45 {
            "model".into()
        } else {
            "geometry".into()
        });
    }

    if detail_mode == "parent" {
        let mut collapsed_ids = Vec::new();
        let mut collapsed_masks = Vec::new();
        let mut collapsed_prov = Vec::new();
        let mut collapsed_conf = Vec::new();
        let groups: &[(&str, &[&str])] = &[
            ("eyeR_all", &["eyeR_white", "eyeR_iris", "eyeR_lower", "eyeR_upper"]),
            ("eyeL_all", &["eyeL_white", "eyeL_iris", "eyeL_lower", "eyeL_upper"]),
            ("mouth_all", &["mouth_inner", "mouth_lower", "mouth_upper"]),
        ];
        let mut skip = std::collections::HashSet::new();
        for &(pid, members) in groups {
            for &m in members {
                skip.insert(m);
            }
            let mut merged = vec![0u8; n];
            let mut hits_sum = 0usize;
            let mut tot_sum = 0usize;
            for &m in members {
                let idx = id_index[m];
                hits_sum += sam_hits[idx];
                tot_sum += total_hits[idx];
                for i in 0..n {
                    merged[i] = merged[i].saturating_add(final_masks[idx][i]);
                }
            }
            collapsed_ids.push(pid.to_string());
            collapsed_masks.push(merged);
            collapsed_conf.push(sam_masks.get(pid).map(|(_, s)| *s as f64).unwrap_or(0.0));
            collapsed_prov.push(if hits_sum >= 12 && (hits_sum as f64 / tot_sum.max(1) as f64) >= 0.45 {
                "model".into()
            } else {
                "geometry".into()
            });
        }
        for (idx, id) in final_ids.iter().enumerate() {
            if !skip.contains(id.as_str()) {
                collapsed_ids.push(id.clone());
                collapsed_masks.push(final_masks[idx].clone());
                collapsed_prov.push(final_provenances[idx].clone());
                collapsed_conf.push(final_confidences[idx]);
            }
        }
        final_ids = collapsed_ids;
        final_masks = collapsed_masks;
        final_provenances = collapsed_prov;
        final_confidences = collapsed_conf;
    }

    if hair_split && w.max(h) >= 200 {
        let empty = vec![0u8; n];
        let front_idx = final_ids.iter().position(|id| id == "hair_front");
        let back_idx = final_ids.iter().position(|id| id == "hair_back");
        let front_ref = front_idx.map(|i| final_masks[i].as_slice()).unwrap_or(&empty);
        let back_ref = back_idx.map(|i| final_masks[i].as_slice()).unwrap_or(&empty);
        let sam_hair_ref = sam_masks.get("hair_front").map(|(m, _)| m.as_slice());

        let mut face_union = vec![0u8; n];
        let mut torso_union = vec![0u8; n];
        for (idx, id) in final_ids.iter().enumerate() {
            if id == "face"
                || id.starts_with("eye")
                || id.starts_with("brow")
                || id.starts_with("mouth")
            {
                for (dst, &v) in face_union.iter_mut().zip(&final_masks[idx]) {
                    *dst = (*dst).max(v);
                }
            } else if id == "body" {
                for (dst, &v) in torso_union.iter_mut().zip(&final_masks[idx]) {
                    *dst = (*dst).max(v);
                }
            }
        }
        if let Some(ref cl) = clothes_extra {
            for (dst, &v) in torso_union.iter_mut().zip(&cl.mask) {
                *dst = (*dst).max(v);
            }
        }

        let hair_layers = split_hair_layers(&HairSplitInput {
            w,
            h,
            rgba,
            landmarks,
            hair_front_mask: front_ref,
            hair_back_mask: back_ref,
            face_mask: &face_union,
            torso_mask: &torso_union,
        });
        let hair_conf = conf_lookup("hair_front");

        let hair_prov = |mask: &[u8]| -> String {
            let Some(sam_m) = sam_hair_ref else {
                return "geometry".into();
            };
            let mut tot = 0usize;
            let mut hits = 0usize;
            for i in 0..n {
                if mask[i] > 0 {
                    tot += 1;
                    if sam_m[i] > 128 {
                        hits += 1;
                    }
                }
            }
            if hits >= 12 && (hits as f64 / tot.max(1) as f64) >= 0.45 {
                "model".into()
            } else {
                "geometry".into()
            }
        };

        let extra_slot = usize::from(clothes_extra.is_some());
        let mut reordered_ids = Vec::with_capacity(final_ids.len() + hair_layers.len() + extra_slot);
        let mut reordered_masks =
            Vec::with_capacity(final_masks.len() + hair_layers.len() + extra_slot);
        let mut reordered_prov =
            Vec::with_capacity(final_provenances.len() + hair_layers.len() + extra_slot);
        let mut reordered_conf =
            Vec::with_capacity(final_confidences.len() + hair_layers.len() + extra_slot);

        let (back_layers, front_side_layers): (Vec<_>, Vec<_>) = hair_layers
            .into_iter()
            .partition(|(id, _)| id.starts_with("hair_back"));

        for (hid, hmask) in back_layers {
            reordered_prov.push(hair_prov(&hmask));
            reordered_conf.push(hair_conf);
            reordered_ids.push(hid);
            reordered_masks.push(hmask);
        }
        let mut clothes_opt = clothes_extra;
        for idx in 0..final_ids.len() {
            if final_ids[idx] == "hair_back" || final_ids[idx] == "hair_front" {
                continue;
            }
            let is_body = final_ids[idx] == "body";
            reordered_ids.push(final_ids[idx].clone());
            reordered_masks.push(final_masks[idx].clone());
            reordered_prov.push(final_provenances[idx].clone());
            reordered_conf.push(final_confidences[idx]);
            if is_body {
                if let Some(cl) = clothes_opt.take() {
                    let cl_prov = if cl.sam_hits >= 12
                        && (cl.sam_hits as f64 / cl.total_hits.max(1) as f64) >= 0.45
                    {
                        "model".into()
                    } else {
                        "geometry".into()
                    };
                    reordered_ids.push("clothes".to_string());
                    reordered_masks.push(cl.mask);
                    reordered_prov.push(cl_prov);
                    reordered_conf.push(cl.confidence);
                }
            }
        }
        for (hid, hmask) in front_side_layers {
            reordered_prov.push(hair_prov(&hmask));
            reordered_conf.push(hair_conf);
            reordered_ids.push(hid);
            reordered_masks.push(hmask);
        }
        final_ids = reordered_ids;
        final_masks = reordered_masks;
        final_provenances = reordered_prov;
        final_confidences = reordered_conf;
    }

    let eye_r_cx = final_ids
        .iter()
        .position(|id| id == "eyeR_iris" || id == "eyeR_all" || id == "eyeR_white")
        .and_then(|i| mask_centroid_x(&final_masks[i], w, h));
    let eye_l_cx = final_ids
        .iter()
        .position(|id| id == "eyeL_iris" || id == "eyeL_all" || id == "eyeL_white")
        .and_then(|i| mask_centroid_x(&final_masks[i], w, h));
    let lr_swapped = matches!((eye_r_cx, eye_l_cx), (Some(rx), Some(lx)) if rx >= lx);

    let mut metas = Vec::with_capacity(final_ids.len());
    for (idx, id) in final_ids.iter().enumerate() {
        let (pixel_count, mut diag) = diagnose_part_mask(
            id,
            &final_masks[idx],
            rgba,
            w,
            h,
            final_confidences[idx],
        );
        if lr_swapped && (id.starts_with("eyeR") || id.starts_with("eyeL")) {
            diag.push("lr_swapped".into());
        }
        metas.push(PartProposalMeta {
            id: id.clone(),
            provenance: final_provenances[idx].clone(),
            confidence: final_confidences[idx],
            pixel_count,
            diagnostics: diag,
        });
    }

    Ok(PartsProposalOutput {
        part_ids: final_ids,
        masks: final_masks,
        metas,
        provider: provider.into(),
        model_hashes,
        postprocess_version: postprocess_version.into(),
        image_hash,
    })
}

/// Proposes parts using MobileSAM when encoder+decoder are present, falling back to U2-Net + geometry partition.
#[allow(clippy::too_many_arguments)]
pub fn propose_parts_auto(
    state: &AnalysisState,
    rgba: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    points_28: &[Landmark28Point],
    face_bbox: Option<[f64; 4]>,
    tolerance: f64,
    u2net_path: Option<&str>,
    detail_mode: &str,
) -> Result<PartsProposalOutput, String> {
    propose_parts_auto_with_options(
        state,
        rgba,
        w,
        h,
        landmarks,
        points_28,
        face_bbox,
        tolerance,
        u2net_path,
        detail_mode,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn propose_parts_auto_with_options(
    state: &AnalysisState,
    rgba: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    points_28: &[Landmark28Point],
    face_bbox: Option<[f64; 4]>,
    tolerance: f64,
    u2net_path: Option<&str>,
    detail_mode: &str,
    hair_split: bool,
) -> Result<PartsProposalOutput, String> {
    let has_sam = {
        let reg = state
            .registry
            .lock()
            .map_err(|_| "Registry lock poisoned".to_string())?;
        reg.resolve_path(ModelRole::PartEncoder).is_some()
            && reg.resolve_path(ModelRole::PartDecoder).is_some()
    };
    if has_sam {
        if let Ok(sam_out) = propose_parts_with_sam_with_options(
            state,
            rgba,
            w,
            h,
            landmarks,
            points_28,
            face_bbox,
            tolerance,
            u2net_path,
            detail_mode,
            hair_split,
        ) {
            return Ok(sam_out);
        }
    }

    let image_hash = hash_rgba_bytes(rgba, w, h);
    let (_fg, u2_alpha, bg_hash) =
        resolve_foreground_alpha(state, rgba, w, h, tolerance, u2net_path)?;
    let masks = core::separate(
        rgba,
        w,
        h,
        landmarks,
        tolerance.clamp(0.0, 180.0),
        u2_alpha.as_deref(),
    );
    let mut model_hashes = HashMap::new();
    let provider = if let Some(h_u2) = bg_hash {
        model_hashes.insert("background".into(), h_u2);
        "u2net-background+geometry-template-v1"
    } else {
        "geometry-template-v1"
    };
    let sam_hits = vec![0usize; IDS.len()];
    let total_hits: Vec<usize> = masks
        .iter()
        .map(|m| m.iter().filter(|&&v| v > 0).count())
        .collect();
    let empty_sam = HashMap::new();
    finalize_part_proposal(
        rgba,
        w,
        h,
        landmarks,
        masks,
        sam_hits,
        total_hits,
        &empty_sam,
        None,
        detail_mode,
        hair_split,
        provider,
        model_hashes,
        "geometry-template-v1",
        image_hash,
    )
}

/// Interactive single-part refinement using cached MobileSAM embeddings (**decoder-only**).
#[allow(clippy::too_many_arguments)]
pub fn refine_part_with_prompts(
    state: &AnalysisState,
    rgba: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    part_id: &str,
    prompts: &[PromptInput],
    tolerance: f64,
) -> Result<RefinePromptOutput, String> {
    if prompts.is_empty() {
        return Err("Click at least one point on the canvas".into());
    }
    let image_hash = hash_rgba_bytes(rgba, w, h);
    let (fg, _, _) = resolve_foreground_alpha(state, rgba, w, h, tolerance, None)?;

    let is_face_detail = part_id.starts_with("eye")
        || part_id.starts_with("brow")
        || part_id.starts_with("mouth");
    let rect = compute_face_crop_rect(landmarks, w, h);
    let all_in_crop = is_face_detail && prompts.iter().all(|p| rect.contains_point(p.x, p.y));

    let (cache, mapped_prompts): (SamEmbeddingCache, Vec<PromptInput>) = if all_in_crop {
        let c = get_or_encode_crop_sam(state, rgba, w, h, &image_hash, rect)?;
        let mp = prompts
            .iter()
            .map(|p| PromptInput {
                x: p.x - rect.x0 as f64,
                y: p.y - rect.y0 as f64,
                label: p.label,
            })
            .collect();
        (c, mp)
    } else {
        let c = get_or_encode_full_sam(state, rgba, w, h, &image_hash)?;
        (c, prompts.to_vec())
    };

    let dec = {
        let mut reg = state
            .registry
            .lock()
            .map_err(|_| "Registry lock poisoned".to_string())?;
        decode_sam_prompts(&mut reg, &cache, &mapped_prompts, None)?
    };

    let mut candidates = Vec::with_capacity(dec.masks.len());
    for (slot, m) in dec.masks.into_iter().enumerate() {
        let mut full_mask = if all_in_crop {
            uncrop_mask(&m, rect.w, rect.h, rect.x0, rect.y0, w, h)
        } else {
            m
        };
        for i in 0..(w * h) {
            full_mask[i] = if full_mask[i] > 128 { fg[i] } else { 0 };
        }
        let iou = dec.iou_scores[slot];
        let (pixel_count, diagnostics) =
            diagnose_part_mask(part_id, &full_mask, rgba, w, h, iou as f64);
        candidates.push(PromptCandidateMask {
            index: slot,
            iou_score: iou,
            mask: full_mask,
            pixel_count,
            diagnostics,
        });
    }

    Ok(RefinePromptOutput {
        part_id: part_id.to_string(),
        best_index: dec.best_idx,
        candidates,
        used_face_crop: all_in_crop,
        decoder_sha256: dec.decoder_sha256,
        encoder_sha256: cache.model_hash,
    })
}
