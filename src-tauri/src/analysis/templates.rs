//! Top-down parametric Live2D silhouette bank matching and hybrid texture projection.
//!
//! Combines illustration evidence (YOLOv3/HRNetV2/U2NetP/MobileSAM masks + clean eroded palettes)
//! with anatomical `HeadFrame` silhouette templates that pre-build motion-ready hidden geometry
//! (`fillMask` for complete round irises, full sclerae, closing eyelids, oral cavities, under-chin neck,
//! and behind-torso back hair) while keeping `masks` as a strict non-overlapping foreground partition.

use crate::analysis::landmarks::Landmark28Point;
use crate::core::{HeadFrame, Landmarks};
use base64::{engine::general_purpose::STANDARD, Engine};
use glam::DVec2;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, ImageEncoder};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Canonical layered template Z-order (bottom-to-top Live2D draw order: `neck` sits under `clothes` and `face`).
pub const TEMPLATE_LAYER_ORDER: [&str; 21] = [
    "hair_back",
    "body",
    "neck",
    "clothes",
    "face",
    "eyeR_white",
    "eyeR_iris",
    "eyeR_lower",
    "eyeR_upper",
    "eyeL_white",
    "eyeL_iris",
    "eyeL_lower",
    "eyeL_upper",
    "browR",
    "browL",
    "mouth_inner",
    "mouth_lower",
    "mouth_upper",
    "hair_side_R",
    "hair_side_L",
    "hair_front",
];

const L_HAIR_BACK: usize = 0;
const L_BODY: usize = 1;
const L_NECK: usize = 2;
const L_CLOTHES: usize = 3;
const L_FACE: usize = 4;
const L_EYE_R_BASE: usize = 5;
const L_EYE_L_BASE: usize = 9;
const L_MOUTH_INNER: usize = 15;
const L_MOUTH_LOWER: usize = 16;
const L_MOUTH_UPPER: usize = 17;
const L_HAIR_SIDE_R: usize = 18;
const L_HAIR_SIDE_L: usize = 19;
const L_HAIR_FRONT: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TemplateStyleSelection {
    pub proportion_mode: Option<String>,
    pub face_style: Option<String>,
    pub hair_front_style: Option<String>,
    pub hair_side_style: Option<String>,
    pub hair_back_style: Option<String>,
    pub eye_style: Option<String>,
    pub mouth_style: Option<String>,
    pub clothes_style: Option<String>,
}

/// Returns `true` when any style preset field is explicitly overridden away from `"auto"`.
pub fn has_explicit_style_override(overrides: &TemplateStyleSelection) -> bool {
    [
        overrides.proportion_mode.as_deref(),
        overrides.face_style.as_deref(),
        overrides.hair_front_style.as_deref(),
        overrides.hair_side_style.as_deref(),
        overrides.hair_back_style.as_deref(),
        overrides.eye_style.as_deref(),
        overrides.mouth_style.as_deref(),
        overrides.clothes_style.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|s| s != "auto")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FittedCharacterMetrics {
    pub face_rx_ratio: f64,
    pub jaw_taper: f64,
    pub eye_rx_ratio: f64,
    pub eye_ry_ratio: f64,
    pub iris_radius_ratio: f64,
    pub mouth_width_ratio: f64,
    pub mouth_open_ratio: f64,
    pub bang_coverage: f64,
    pub center_part_gap: f64,
    pub side_hair_reach: f64,
    pub back_hair_reach: f64,
    pub shoulder_width_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleMatchReport {
    pub resolved: TemplateStyleSelection,
    pub match_score: f64,
    pub metrics: FittedCharacterMetrics,
    pub palette_summary: HashMap<String, String>,
}

pub struct TemplateSynthesisOutput {
    pub part_ids: Vec<String>,
    pub masks: Vec<Vec<u8>>,
    pub fill_masks: Vec<Vec<u8>>,
    pub fill_kinds: Vec<String>,
    pub textures: Vec<Option<String>>,
    pub provenances: Vec<String>,
    pub confidences: Vec<f64>,
    pub style_match: StyleMatchReport,
}

struct PaletteInfo {
    skin_base: [u8; 3],
    skin_shadow: [u8; 3],
    hair_base: [u8; 3],
    hair_shadow: [u8; 3],
    sclera_white: [u8; 3],
    sclera_shadow: [u8; 3],
    lash_color: [u8; 3],
    brow_color: [u8; 3],
    mouth_line: [u8; 3],
    clothes_primary: [u8; 3],
}

#[derive(Clone, Copy)]
struct IrisFit {
    center_u: f64,
    center_v: f64,
    rx: f64,
    ry: f64,
    limbal: [u8; 3],
    mid: [u8; 3],
    pupil: [u8; 3],
    sample_count: usize,
}

struct AnalyzedEvidence {
    fg_alpha: Vec<u8>,
    ev_by_id: HashMap<String, usize>,
    metrics: FittedCharacterMetrics,
    resolved: TemplateStyleSelection,
    is_sd: bool,
    match_score: f64,
    palette: PaletteInfo,
    eye_r_center: (f64, f64),
    eye_l_center: (f64, f64),
    eye_r_rx: f64,
    eye_r_ry: f64,
    eye_l_rx: f64,
    eye_l_ry: f64,
    iris_r: IrisFit,
    iris_l: IrisFit,
    mouth_u: f64,
    mouth_v: f64,
    torso_cx_u: f64,
    clothes_top_v: f64,
    clothes_top_u_bins: Vec<f64>,
    fg_min_u_bins: Vec<f64>,
    fg_max_u_bins: Vec<f64>,
    fg_max_v: f64,
}

struct ColorField {
    gw: usize,
    gh: usize,
    step: usize,
    rgb: Vec<[f32; 3]>,
}

impl ColorField {
    fn sample(&self, x: usize, y: usize) -> [u8; 3] {
        let fx = (x as f32 / self.step.max(1) as f32).clamp(0.0, (self.gw.saturating_sub(1)) as f32);
        let fy = (y as f32 / self.step.max(1) as f32).clamp(0.0, (self.gh.saturating_sub(1)) as f32);
        let x0 = (fx.floor() as usize).min(self.gw.saturating_sub(1));
        let y0 = (fy.floor() as usize).min(self.gh.saturating_sub(1));
        let x1 = (x0 + 1).min(self.gw.saturating_sub(1));
        let y1 = (y0 + 1).min(self.gh.saturating_sub(1));
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let c00 = self.rgb[y0 * self.gw + x0];
        let c10 = self.rgb[y0 * self.gw + x1];
        let c01 = self.rgb[y1 * self.gw + x0];
        let c11 = self.rgb[y1 * self.gw + x1];

        let mut out = [0u8; 3];
        for ch in 0..3 {
            let top = c00[ch] * (1.0 - tx) + c10[ch] * tx;
            let bot = c01[ch] * (1.0 - tx) + c11[ch] * tx;
            out[ch] = (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8;
        }
        out
    }
}

#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let span = edge1 - edge0;
    let t = if span.abs() < 1e-6 {
        if (edge1 >= edge0 && x >= edge1) || (edge0 > edge1 && x <= edge1) {
            1.0
        } else {
            0.0
        }
    } else {
        ((x - edge0) / span).clamp(0.0, 1.0)
    };
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn luma(rgb: [u8; 3]) -> f64 {
    0.299 * rgb[0] as f64 + 0.587 * rgb[1] as f64 + 0.114 * rgb[2] as f64
}

#[inline]
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        (a[0] as f64 * (1.0 - t) + b[0] as f64 * t).round().clamp(0.0, 255.0) as u8,
        (a[1] as f64 * (1.0 - t) + b[1] as f64 * t).round().clamp(0.0, 255.0) as u8,
        (a[2] as f64 * (1.0 - t) + b[2] as f64 * t).round().clamp(0.0, 255.0) as u8,
    ]
}

#[inline]
fn scale_rgb(c: [u8; 3], k: f64) -> [u8; 3] {
    [
        (c[0] as f64 * k).round().clamp(0.0, 255.0) as u8,
        (c[1] as f64 * k).round().clamp(0.0, 255.0) as u8,
        (c[2] as f64 * k).round().clamp(0.0, 255.0) as u8,
    ]
}

#[inline]
fn hex_rgb(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

#[inline]
fn sdf_to_alpha(dist_px: f64) -> u8 {
    if dist_px <= -1.2 {
        255
    } else if dist_px >= 1.2 {
        0
    } else {
        (smoothstep(1.2, -1.2, dist_px) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8
    }
}

/// Erodes boolean mask inward so outer lineart pixels do not contaminate color samples.
fn erode_mask(src: &[bool], w: usize, h: usize, radius: usize) -> Vec<bool> {
    let n = w * h;
    let mut dist = vec![u16::MAX; n];
    let mut q = VecDeque::new();
    for i in 0..n {
        if !src[i] {
            continue;
        }
        let x = i % w;
        let y = i / w;
        let on_edge = x == 0
            || x + 1 == w
            || y == 0
            || y + 1 == h
            || !src[i - 1]
            || !src[i + 1]
            || !src[i - w]
            || !src[i + w];
        if on_edge {
            dist[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        let next_d = dist[i] + 1;
        if next_d as usize > radius {
            continue;
        }
        let x = i % w;
        let y = i / w;
        for nb in [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ]
        .into_iter()
        .flatten()
        {
            if src[nb] && dist[nb] == u16::MAX {
                dist[nb] = next_d;
                q.push_back(nb);
            }
        }
    }
    (0..n)
        .map(|i| src[i] && dist[i] as usize >= radius)
        .collect()
}

/// Builds a fast multi-grid harmonic color field from clean interior seed pixels.
fn build_color_field(
    rgba: &[u8],
    seeds: &[bool],
    w: usize,
    h: usize,
    fallback: [u8; 3],
) -> ColorField {
    let step = (w.max(h) / 128).clamp(2, 8);
    let gw = w.div_ceil(step).max(4);
    let gh = h.div_ceil(step).max(4);
    let gn = gw * gh;

    let mut sums = vec![[0f64; 3]; gn];
    let mut counts = vec![0usize; gn];

    for y in 0..h {
        let gy = (y / step).min(gh - 1);
        for x in 0..w {
            let i = y * w + x;
            if !seeds[i] {
                continue;
            }
            let gi = gy * gw + (x / step).min(gw - 1);
            let k = i * 4;
            sums[gi][0] += rgba[k] as f64;
            sums[gi][1] += rgba[k + 1] as f64;
            sums[gi][2] += rgba[k + 2] as f64;
            counts[gi] += 1;
        }
    }

    let fb = [fallback[0] as f32, fallback[1] as f32, fallback[2] as f32];
    let mut pinned = vec![false; gn];
    let mut rgb = vec![fb; gn];
    let mut q = VecDeque::new();

    for gi in 0..gn {
        if counts[gi] > 0 {
            let c = counts[gi] as f64;
            rgb[gi] = [
                (sums[gi][0] / c) as f32,
                (sums[gi][1] / c) as f32,
                (sums[gi][2] / c) as f32,
            ];
            pinned[gi] = true;
            q.push_back(gi);
        }
    }

    // BFS initialization so every cell receives its nearest seed first (zero zero-bias).
    let mut visited = pinned.clone();
    while let Some(gi) = q.pop_front() {
        let gx = gi % gw;
        let gy = gi / gw;
        for nb in [
            (gx > 0).then(|| gi - 1),
            (gx + 1 < gw).then(|| gi + 1),
            (gy > 0).then(|| gi - gw),
            (gy + 1 < gh).then(|| gi + gw),
        ]
        .into_iter()
        .flatten()
        {
            if !visited[nb] {
                visited[nb] = true;
                rgb[nb] = rgb[gi];
                q.push_back(nb);
            }
        }
    }

    // Harmonic Jacobi relaxation across non-pinned cells for smooth gradients across holes/occlusions.
    let mut next = rgb.clone();
    for _ in 0..18 {
        for gy in 0..gh {
            for gx in 0..gw {
                let gi = gy * gw + gx;
                if pinned[gi] {
                    continue;
                }
                let mut acc = [0f32; 3];
                let mut w_sum = 0f32;
                for nb in [
                    (gx > 0).then(|| gi - 1),
                    (gx + 1 < gw).then(|| gi + 1),
                    (gy > 0).then(|| gi - gw),
                    (gy + 1 < gh).then(|| gi + gw),
                ]
                .into_iter()
                .flatten()
                {
                    let wt = if pinned[nb] { 2.0 } else { 1.0 };
                    for ch in 0..3 {
                        acc[ch] += rgb[nb][ch] * wt;
                    }
                    w_sum += wt;
                }
                if w_sum > 0.0 {
                    for ch in 0..3 {
                        next[gi][ch] = acc[ch] / w_sum;
                    }
                }
            }
        }
        std::mem::swap(&mut rgb, &mut next);
    }

    ColorField { gw, gh, step, rgb }
}

fn sample_quantile_color(
    rgba: &[u8],
    mask: &[bool],
    quantile: f64,
    fallback: [u8; 3],
) -> [u8; 3] {
    let mut samples: Vec<(f64, [u8; 3])> = Vec::new();
    for (i, &keep) in mask.iter().enumerate() {
        if !keep {
            continue;
        }
        let k = i * 4;
        let c = [rgba[k], rgba[k + 1], rgba[k + 2]];
        samples.push((luma(c), c));
    }
    if samples.is_empty() {
        return fallback;
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let idx = ((samples.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    samples[idx.min(samples.len() - 1)].1
}

fn resolve_style_choice(override_val: &Option<String>, auto_val: &str, valid: &[&str]) -> String {
    if let Some(ref s) = override_val {
        let trimmed = s.trim();
        if !trimmed.is_empty() && trimmed != "auto" && valid.contains(&trimmed) {
            return trimmed.to_string();
        }
    }
    auto_val.to_string()
}

#[derive(Clone, Copy)]
struct IrisFitContext {
    eye_cu: f64,
    eye_cv: f64,
    eye_rx: f64,
    eye_ry: f64,
    is_sd: bool,
}

#[allow(clippy::too_many_arguments)]
fn fit_iris_from_evidence(
    rgba: &[u8],
    w: usize,
    h: usize,
    frame: HeadFrame,
    iris_ev: Option<&[u8]>,
    white_ev: Option<&[u8]>,
    extra_eye_ev: Option<&[bool]>,
    ctx: IrisFitContext,
) -> IrisFit {
    let (eye_cu, eye_cv, eye_rx, eye_ry, is_sd) =
        (ctx.eye_cu, ctx.eye_cv, ctx.eye_rx, ctx.eye_ry, ctx.is_sd);
    let mut sum_u = 0.0f64;
    let mut sum_v = 0.0f64;
    let mut count = 0usize;
    let mut bright_bg = [245u8, 244, 248];

    if let Some(wm) = white_ev {
        let mut w_bool = vec![false; w * h];
        for i in 0..(w * h) {
            if wm[i] > 128 {
                w_bool[i] = true;
            }
        }
        bright_bg = sample_quantile_color(rgba, &w_bool, 0.78, bright_bg);
    }
    let white_luma = luma(bright_bg);

    let mut samples: Vec<(f64, f64, [u8; 3])> = Vec::new();
    let collect_from = |mask: &[u8],
                        luma_cap: f64,
                        inner_ratio_u: f64,
                        inner_ratio_v: f64,
                        out: &mut Vec<(f64, f64, [u8; 3])>,
                        su: &mut f64,
                        sv: &mut f64,
                        cnt: &mut usize| {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if mask[i] <= 64 {
                    continue;
                }
                let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                if (u - eye_cu).abs() > eye_rx * inner_ratio_u
                    || (v - eye_cv).abs() > eye_ry * inner_ratio_v
                {
                    continue;
                }
                let k = i * 4;
                let rgb = [rgba[k], rgba[k + 1], rgba[k + 2]];
                if luma(rgb) > luma_cap && out.len() > 24 {
                    continue;
                }
                *su += u;
                *sv += v;
                *cnt += 1;
                out.push((u, v, rgb));
            }
        }
    };
    if let Some(im) = iris_ev {
        collect_from(
            im,
            white_luma * 0.96,
            1.15,
            1.25,
            &mut samples,
            &mut sum_u,
            &mut sum_v,
            &mut count,
        );
    }
    // Include dark eyeball pixels even when semantic splitting sliced tall SD irises into upper/lower/white buckets.
    if (is_sd || count < 36) && extra_eye_ev.is_some() {
        if let Some(em) = extra_eye_ev {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if !em[i] {
                        continue;
                    }
                    let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                    if (u - eye_cu).abs() > eye_rx * 0.84
                        || (v - eye_cv) < -eye_ry * 0.78
                        || (v - eye_cv) > eye_ry * 0.86
                    {
                        continue;
                    }
                    let k = i * 4;
                    let rgb = [rgba[k], rgba[k + 1], rgba[k + 2]];
                    if luma(rgb) > white_luma * 0.84 {
                        continue;
                    }
                    sum_u += u;
                    sum_v += v;
                    count += 1;
                    samples.push((u, v, rgb));
                }
            }
        }
    }
    if count < 12 {
        if let Some(wm) = white_ev {
            collect_from(
                wm,
                white_luma * 0.85,
                1.10,
                1.20,
                &mut samples,
                &mut sum_u,
                &mut sum_v,
                &mut count,
            );
        }
    }

    let (cu, cv) = if count >= 12 {
        let lift = if is_sd { 0.04 } else { 0.12 };
        (
            (sum_u / count as f64).clamp(
                eye_cu - eye_rx * 0.25,
                eye_cu + eye_rx * 0.25,
            ),
            (sum_v / count as f64 - eye_ry * lift).clamp(
                eye_cv - eye_ry * 0.24,
                eye_cv + eye_ry * 0.16,
            ),
        )
    } else {
        (eye_cu, eye_cv)
    };

    let rx = if count >= 24 {
        let mut du_vals: Vec<f64> = samples.iter().map(|(u, _, _)| (u - cu).abs()).collect();
        du_vals.sort_by(|a, b| a.total_cmp(b));
        let p90 = du_vals[((du_vals.len() - 1) * 90) / 100];
        if is_sd {
            p90.clamp(eye_rx * 0.56, eye_rx * 0.82)
        } else {
            p90.clamp(eye_rx * 0.36, eye_rx * 0.56)
        }
    } else if is_sd {
        eye_rx * 0.68
    } else {
        eye_rx * 0.46
    };
    let ry = if is_sd {
        (rx * 1.06).clamp(eye_ry * 0.78, (eye_rx * 0.86).max(eye_ry * 1.06))
    } else {
        (rx * 1.16).clamp(eye_ry * 0.95, (eye_rx * 0.72).max(eye_ry * 1.32))
    };

    // Filter dark mid-zone colors separately from ultra-dark top lash overlap.
    let mut mid_bool = vec![false; w * h];
    for &(u, v, rgb) in &samples {
        let dv = v - cv;
        let rn = ((u - cu) / rx.max(1.0)).hypot(dv / ry.max(1.0));
        if (0.22..=0.78).contains(&rn) && dv >= -ry * 0.35 && luma(rgb) > 28.0 {
            let wp = frame.origin + frame.ex * u + frame.ey * v;
            let px = (wp.x.round() as isize).clamp(0, w as isize - 1) as usize;
            let py = (wp.y.round() as isize).clamp(0, h as isize - 1) as usize;
            mid_bool[py * w + px] = true;
        }
    }
    let mid = sample_quantile_color(rgba, &mid_bool, 0.52, [88, 96, 152]);
    let limbal = scale_rgb(mid, 0.45);
    let pupil = scale_rgb(mid, 0.26);

    IrisFit {
        center_u: cu,
        center_v: cv,
        rx,
        ry,
        limbal,
        mid,
        pupil,
        sample_count: count,
    }
}

#[allow(clippy::too_many_arguments)]
fn analyze_evidence(
    rgba: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    points_28: &[Landmark28Point],
    evidence_ids: &[String],
    evidence_masks: &[Vec<u8>],
    overrides: &TemplateStyleSelection,
) -> AnalyzedEvidence {
    let n = w * h;
    let frame = HeadFrame::from_landmarks(landmarks, w, h);
    let eye_dist = frame.eye_dist.max(8.0);
    let hh = frame.hh.max(12.0);

    let mut ev_by_id = HashMap::new();
    for (idx, id) in evidence_ids.iter().enumerate() {
        ev_by_id.insert(id.clone(), idx);
    }

    let mut fg_alpha = vec![0u8; n];
    for m in evidence_masks {
        for i in 0..n {
            fg_alpha[i] = fg_alpha[i].max(m[i]);
        }
    }

    let get_ev = |id: &str| -> Option<&[u8]> {
        ev_by_id.get(id).map(|&idx| evidence_masks[idx].as_slice())
    };

    // Boolean helper masks for palette extraction.
    let mut face_skin_raw = vec![false; n];
    let mut neck_raw = vec![false; n];
    let mut hair_raw = vec![false; n];
    let mut clothes_raw = vec![false; n];
    let mut brow_raw = vec![false; n];
    let mut lash_raw = vec![false; n];
    let mut mouth_raw = vec![false; n];
    let mut sclera_raw = vec![false; n];

    let mut face_cheek_us = Vec::new();
    let mut face_jaw_us = Vec::new();
    let mut bangs_u_left = 0usize;
    let mut bangs_u_right = 0usize;
    let mut bangs_u_center = 0usize;
    let mut bangs_total = 0usize;
    let mut forehead_total = 0usize;
    let mut side_r_vs = Vec::new();
    let mut side_l_vs = Vec::new();
    let mut back_vs = Vec::new();
    let mut back_max_abs_u = eye_dist;
    let mut clothes_min_v = frame.chin_v + hh * 0.45;
    let mut clothes_max_abs_u = eye_dist * 1.4;
    let mut torso_u_sum = 0.0f64;
    let mut torso_u_cnt = 0usize;
    let mut fg_max_v = frame.chin_v + hh * 0.4;

    let mut eye_r_us = Vec::new();
    let mut eye_r_vs = Vec::new();
    let mut eye_l_us = Vec::new();
    let mut eye_l_vs = Vec::new();
    let mut eye_r_full_us = Vec::new();
    let mut eye_r_full_vs = Vec::new();
    let mut eye_l_full_us = Vec::new();
    let mut eye_l_full_vs = Vec::new();
    let mut eye_r_full_mask = vec![false; n];
    let mut eye_l_full_mask = vec![false; n];
    let mut mouth_us = Vec::new();
    let mut mouth_vs = Vec::new();
    let mut mouth_open_px = 0usize;
    let mut mouth_total_px = 0usize;

    let (eye_r_lm_u, eye_r_lm_v) = frame.to_local(DVec2::new(landmarks.eye_r.x, landmarks.eye_r.y));
    let (eye_l_lm_u, eye_l_lm_v) = frame.to_local(DVec2::new(landmarks.eye_l.x, landmarks.eye_l.y));
    let (mouth_lm_u, mouth_lm_v) = frame.to_local(DVec2::new(landmarks.mouth.x, landmarks.mouth.y));

    const V_BINS: usize = 64;
    const U_BINS: usize = 64;
    let v_top = frame.crown_v - hh * 0.12;
    let v_bot = (frame.chin_v + hh * 1.65).max(frame.crown_v + hh * 1.8);
    let mut fg_min_u_bins = vec![f64::INFINITY; V_BINS];
    let mut fg_max_u_bins = vec![f64::NEG_INFINITY; V_BINS];
    let mut clothes_v_by_bin: Vec<Vec<f64>> = vec![Vec::new(); U_BINS];
    let mut hair_over_torso_min_v = vec![f64::INFINITY; U_BINS];

    for (idx, id) in evidence_ids.iter().enumerate() {
        let mask = &evidence_masks[idx];
        let is_face = id == "face";
        let is_neck = id == "neck";
        let is_clothes = id.starts_with("clothes");
        let is_body = id == "body";
        let is_front = id.starts_with("hair_front");
        let is_side_r = id.starts_with("hair_side_R");
        let is_side_l = id.starts_with("hair_side_L");
        let is_back = id.starts_with("hair_back");
        let is_hair = is_front || is_side_r || is_side_l || is_back;
        let is_brow = id.starts_with("brow");
        let is_eye_r = id == "eyeR_white" || id == "eyeR_iris" || id == "eyeR_all";
        let is_eye_l = id == "eyeL_white" || id == "eyeL_iris" || id == "eyeL_all";
        let is_eye_r_any = is_eye_r || id == "eyeR_upper" || id == "eyeR_lower";
        let is_eye_l_any = is_eye_l || id == "eyeL_upper" || id == "eyeL_lower";
        let is_white = id.ends_with("_white");
        let is_lash = id.ends_with("_upper") || id.ends_with("_lower");
        let is_mouth = id.starts_with("mouth");
        let is_mouth_inner = id == "mouth_inner";

        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if mask[i] <= 64 {
                    continue;
                }
                let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                fg_max_v = fg_max_v.max(v);
                let vb = (((v - v_top) / (v_bot - v_top).max(1.0)) * (V_BINS as f64))
                    .floor()
                    .clamp(0.0, (V_BINS - 1) as f64) as usize;
                fg_min_u_bins[vb] = fg_min_u_bins[vb].min(u);
                fg_max_u_bins[vb] = fg_max_u_bins[vb].max(u);

                if is_face {
                    face_skin_raw[i] = true;
                    if v >= eye_r_lm_v.min(eye_l_lm_v) - eye_dist * 0.05 && v <= mouth_lm_v {
                        face_cheek_us.push(u.abs());
                    } else if v > mouth_lm_v && v <= frame.chin_v - eye_dist * 0.04 {
                        face_jaw_us.push(u.abs());
                    }
                    if v >= frame.crown_v + hh * 0.08 && v <= eye_r_lm_v.min(eye_l_lm_v) - hh * 0.08 && u.abs() < eye_dist * 0.95 {
                        forehead_total += 1;
                    }
                } else if is_neck {
                    neck_raw[i] = true;
                } else if is_clothes {
                    clothes_raw[i] = true;
                    clothes_min_v = clothes_min_v.min(v);
                    clothes_max_abs_u = clothes_max_abs_u.max(u.abs());
                    torso_u_sum += u;
                    torso_u_cnt += 1;
                    let ub = (((u + eye_dist * 3.0) / (eye_dist * 6.0).max(1.0)) * (U_BINS as f64))
                        .floor()
                        .clamp(0.0, (U_BINS - 1) as f64) as usize;
                    if v >= frame.chin_v + hh * 0.16 && u.abs() <= eye_dist * 2.4 {
                        clothes_v_by_bin[ub].push(v);
                    }
                } else if is_body {
                    torso_u_sum += u;
                    torso_u_cnt += 1;
                } else if is_hair {
                    hair_raw[i] = true;
                    if v >= frame.chin_v && u.abs() <= eye_dist * 2.4 {
                        let ub = (((u + eye_dist * 3.0) / (eye_dist * 6.0).max(1.0)) * (U_BINS as f64))
                            .floor()
                            .clamp(0.0, (U_BINS - 1) as f64) as usize;
                        hair_over_torso_min_v[ub] = hair_over_torso_min_v[ub].min(v);
                    }
                    if v >= frame.crown_v + hh * 0.05 && v <= eye_r_lm_v.min(eye_l_lm_v) - hh * 0.02 && u.abs() <= eye_dist * 0.95 {
                        forehead_total += 1;
                        bangs_total += 1;
                        if u.abs() <= eye_dist * 0.16 {
                            bangs_u_center += 1;
                        } else if u < 0.0 {
                            bangs_u_left += 1;
                        } else {
                            bangs_u_right += 1;
                        }
                    }
                    if is_side_r {
                        side_r_vs.push(v);
                    }
                    if is_side_l {
                        side_l_vs.push(v);
                    }
                    if is_back && v >= frame.crown_v + hh * 0.20 {
                        back_vs.push(v);
                        back_max_abs_u = back_max_abs_u.max(u.abs());
                    }
                } else if is_brow {
                    brow_raw[i] = true;
                } else if is_mouth {
                    mouth_raw[i] = true;
                    mouth_us.push(u);
                    mouth_vs.push(v);
                    mouth_total_px += 1;
                    if is_mouth_inner {
                        mouth_open_px += 1;
                    }
                }
                if is_eye_r {
                    eye_r_us.push(u);
                    eye_r_vs.push(v);
                }
                if is_eye_l {
                    eye_l_us.push(u);
                    eye_l_vs.push(v);
                }
                if is_eye_r_any
                    && (u - eye_r_lm_u).abs() <= eye_dist * 0.42
                    && (v - eye_r_lm_v).abs() <= eye_dist * 0.44
                {
                    eye_r_full_us.push(u);
                    eye_r_full_vs.push(v);
                    eye_r_full_mask[i] = true;
                }
                if is_eye_l_any
                    && (u - eye_l_lm_u).abs() <= eye_dist * 0.42
                    && (v - eye_l_lm_v).abs() <= eye_dist * 0.44
                {
                    eye_l_full_us.push(u);
                    eye_l_full_vs.push(v);
                    eye_l_full_mask[i] = true;
                }
                if is_white {
                    sclera_raw[i] = true;
                }
                if is_lash {
                    lash_raw[i] = true;
                }
            }
        }
    }

    // Fill empty vertical silhouette bins smoothly so bounds never have infinity gaps.
    for b in 0..V_BINS {
        if !fg_min_u_bins[b].is_finite() {
            fg_min_u_bins[b] = -eye_dist * 1.25;
            fg_max_u_bins[b] = eye_dist * 1.25;
        }
    }
    // Compute outlier-resistant 8th-percentile neckline v per u-column and fill short occluded gaps.
    let mut clothes_top_u_bins = vec![f64::INFINITY; U_BINS];
    for (b, vals) in clothes_v_by_bin.iter_mut().enumerate() {
        if vals.len() >= 6 {
            vals.sort_by(|a, c| a.total_cmp(c));
            clothes_top_u_bins[b] = vals[(vals.len() - 1) * 8 / 100];
        }
    }
    // Invalidate local clothes-top dips caused by hanging hair strands occluding upper shoulders/collar.
    let raw_top_snapshot = clothes_top_u_bins.clone();
    for b in 0..U_BINS {
        let cur = raw_top_snapshot[b];
        if !cur.is_finite() || hair_over_torso_min_v[b] > cur + 8.0 {
            continue;
        }
        let lo = b.saturating_sub(14);
        let hi = (b + 14).min(U_BINS - 1);
        let higher_neighbor = (lo..=hi).any(|nb| {
            let nv = raw_top_snapshot[nb];
            nv.is_finite() && hair_over_torso_min_v[nb] > nv + 8.0 && nv + 6.0 < cur
        });
        if higher_neighbor {
            clothes_top_u_bins[b] = f64::INFINITY;
        }
    }
    let valid_u_cols: Vec<(usize, f64)> = clothes_top_u_bins
        .iter()
        .enumerate()
        .filter_map(|(b, &val)| val.is_finite().then_some((b, val)))
        .collect();
    for pair in valid_u_cols.windows(2) {
        let (b0, v0) = pair[0];
        let (b1, v1) = pair[1];
        if b1 > b0 + 1 && b1 - b0 <= 18 {
            for (offset, dst) in clothes_top_u_bins[(b0 + 1)..b1].iter_mut().enumerate() {
                let t = (offset + 1) as f64 / (b1 - b0) as f64;
                *dst = v0 * (1.0 - t) + v1 * t;
            }
        }
    }

    let raw_cheek_rx = if face_cheek_us.len() >= 30 {
        face_cheek_us.sort_by(|a, b| a.total_cmp(b));
        face_cheek_us[((face_cheek_us.len() - 1) * 94) / 100]
    } else {
        eye_dist * 1.04
    };
    let raw_jaw_rx = if face_jaw_us.len() >= 20 {
        face_jaw_us.sort_by(|a, b| a.total_cmp(b));
        face_jaw_us[((face_jaw_us.len() - 1) * 90) / 100]
    } else {
        raw_cheek_rx * 0.72
    };

    // Estimate raw eye box across core + full eye evidence and 28-pt eye contours before LD/SD clamping.
    let measure_raw_eye = |core_us: &[f64],
                           core_vs: &[f64],
                           full_us: &[f64],
                           full_vs: &[f64],
                           p28_indices: &[usize],
                           fallback_u: f64,
                           fallback_v: f64|
     -> (f64, f64, f64, f64, f64) {
        let p28_ry = if points_28.len() >= 24 {
            let mut vs = Vec::new();
            for &idx in p28_indices {
                if let Some(p) = points_28.get(idx).filter(|p| p.score > 0.25) {
                    vs.push(frame.to_local(DVec2::new(p.x, p.y)).1);
                }
            }
            if vs.len() >= 4 {
                let min_v = vs.iter().copied().fold(f64::INFINITY, f64::min);
                let max_v = vs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                0.5 * (max_v - min_v) * 1.15
            } else {
                0.0
            }
        } else {
            0.0
        };
        let (use_us, use_vs) = if full_us.len() >= 24 {
            (full_us, full_vs)
        } else {
            (core_us, core_vs)
        };
        if use_us.len() < 16 {
            return (fallback_u, fallback_v, eye_dist * 0.28, (eye_dist * 0.145).max(p28_ry), p28_ry);
        }
        let mut su = use_us.to_vec();
        let mut sv = use_vs.to_vec();
        su.sort_by(|a, b| a.total_cmp(b));
        sv.sort_by(|a, b| a.total_cmp(b));
        let u0 = su[(su.len() - 1) * 5 / 100];
        let u1 = su[(su.len() - 1) * 95 / 100];
        let v0 = sv[(sv.len() - 1) * 7 / 100];
        let v1 = sv[(sv.len() - 1) * 93 / 100];
        let core_ry = if core_vs.len() >= 16 {
            let mut csv = core_vs.to_vec();
            csv.sort_by(|a, b| a.total_cmp(b));
            0.5 * (csv[(csv.len() - 1) * 94 / 100] - csv[(csv.len() - 1) * 6 / 100])
        } else {
            0.0
        };
        let cu = (0.5 * (u0 + u1)).clamp(fallback_u - eye_dist * 0.08, fallback_u + eye_dist * 0.08);
        let cv = (0.5 * (v0 + v1)).clamp(fallback_v - eye_dist * 0.06, fallback_v + eye_dist * 0.06);
        let raw_rx = 0.5 * (u1 - u0);
        let raw_ry = (0.5 * (v1 - v0) * 0.88).max(core_ry).max(p28_ry);
        (cu, cv, raw_rx, raw_ry, p28_ry)
    };
    let (raw_er_u, raw_er_v, raw_er_rx, raw_er_ry, _) = measure_raw_eye(
        &eye_r_us,
        &eye_r_vs,
        &eye_r_full_us,
        &eye_r_full_vs,
        &[11, 12, 13, 14, 15, 16],
        eye_r_lm_u,
        eye_r_lm_v,
    );
    let (raw_el_u, raw_el_v, raw_el_rx, raw_el_ry, _) = measure_raw_eye(
        &eye_l_us,
        &eye_l_vs,
        &eye_l_full_us,
        &eye_l_full_vs,
        &[17, 18, 19, 20, 21, 22],
        eye_l_lm_u,
        eye_l_lm_v,
    );

    // Classify archetype: regular proportions (`ld`) vs chibi (`sd`).
    let raw_rx_r = 0.5 * (raw_er_rx + raw_el_rx) / eye_dist;
    let raw_ry_r = 0.5 * (raw_er_ry + raw_el_ry) / eye_dist;
    let raw_eye_aspect = raw_ry_r / raw_rx_r.max(0.10);
    let mouth_drop_ratio = (mouth_lm_v / eye_dist).max(0.05);
    let chin_drop_ratio = (frame.chin_v / eye_dist).max(0.15);
    let neck_gap_ratio = (clothes_min_v - frame.chin_v) / eye_dist;
    let mut sd_score = 0.0f64;
    if mouth_drop_ratio < 0.42 {
        sd_score += 2.2;
    } else if mouth_drop_ratio < 0.49 {
        sd_score += 1.0;
    }
    if chin_drop_ratio < 0.72 {
        sd_score += 2.2;
    } else if chin_drop_ratio < 0.82 {
        sd_score += 1.0;
    }
    if raw_ry_r >= 0.22 {
        sd_score += 2.0;
    } else if raw_ry_r >= 0.19 {
        sd_score += 0.9;
    }
    if raw_rx_r >= 0.30 {
        sd_score += 1.2;
    }
    if raw_eye_aspect >= 0.68 {
        sd_score += 1.0;
    }
    if raw_cheek_rx / eye_dist >= 1.18 {
        sd_score += 0.8;
    }
    if neck_gap_ratio < 0.14 {
        sd_score += 0.8;
    } else if neck_gap_ratio >= 0.28 {
        sd_score -= 1.8;
    }
    let auto_proportion = if sd_score >= 3.4 { "sd" } else { "ld" };
    let proportion_mode = resolve_style_choice(
        &overrides.proportion_mode,
        auto_proportion,
        &["ld", "sd"],
    );
    let is_sd = proportion_mode == "sd";

    let cheek_rx = if is_sd {
        raw_cheek_rx.clamp(eye_dist * 1.12, eye_dist * 1.44)
    } else {
        raw_cheek_rx.clamp(eye_dist * 0.98, eye_dist * 1.28)
    };
    let jaw_rx = if is_sd {
        raw_jaw_rx.clamp(cheek_rx * 0.68, cheek_rx * 0.96)
    } else {
        raw_jaw_rx.clamp(cheek_rx * 0.45, cheek_rx * 0.94)
    };
    let jaw_taper = if is_sd {
        (jaw_rx / cheek_rx.max(1.0)).clamp(0.68, 0.96)
    } else {
        (jaw_rx / cheek_rx.max(1.0)).clamp(0.45, 0.92)
    };
    let face_rx_ratio = if is_sd {
        (cheek_rx / eye_dist).clamp(1.12, 1.44)
    } else {
        (cheek_rx / eye_dist).clamp(0.98, 1.28)
    };

    let clamp_eye_box = |cu: f64, cv: f64, rx_raw: f64, ry_raw: f64| -> (f64, f64, f64, f64) {
        if is_sd {
            let rx = rx_raw.clamp(eye_dist * 0.24, eye_dist * 0.40);
            let ry = ry_raw.max(rx * 0.72).clamp(eye_dist * 0.21, eye_dist * 0.44);
            (cu, cv, rx, ry)
        } else {
            let rx = rx_raw.clamp(eye_dist * 0.21, eye_dist * 0.36);
            let ry = ry_raw.clamp(eye_dist * 0.11, eye_dist * 0.22);
            (cu, cv, rx, ry)
        }
    };
    let (mut er_u, mut er_v, er_rx, er_ry) = clamp_eye_box(raw_er_u, raw_er_v, raw_er_rx, raw_er_ry);
    let (mut el_u, mut el_v, el_rx, el_ry) = clamp_eye_box(raw_el_u, raw_el_v, raw_el_rx, raw_el_ry);

    // Optional HRNetV2 28-pt eye alignment anchor (character right eye 11..=16, left eye 17..=22).
    if points_28.len() >= 24 {
        let p_ok = |idx: usize| points_28.get(idx).filter(|p| p.score > 0.25);
        let mean_pts_uv = |indices: &[usize]| -> Option<(f64, f64)> {
            let mut su = 0.0f64;
            let mut sv = 0.0f64;
            let mut cnt = 0usize;
            for &idx in indices {
                if let Some(p) = p_ok(idx) {
                    let (pu, pv) = frame.to_local(DVec2::new(p.x, p.y));
                    su += pu;
                    sv += pv;
                    cnt += 1;
                }
            }
            (cnt >= 3).then(|| (su / cnt as f64, sv / cnt as f64))
        };
        if let Some((pu, pv)) = mean_pts_uv(&[11, 12, 13, 14, 15, 16]) {
            er_u = 0.55 * er_u + 0.45 * pu;
            er_v = 0.55 * er_v + 0.45 * pv;
        }
        if let Some((pu, pv)) = mean_pts_uv(&[17, 18, 19, 20, 21, 22]) {
            el_u = 0.55 * el_u + 0.45 * pu;
            el_v = 0.55 * el_v + 0.45 * pv;
        }
    }

    let avg_eye_rx = 0.5 * (er_rx + el_rx);
    let avg_eye_ry = 0.5 * (er_ry + el_ry);
    let er_rx = 0.55 * er_rx + 0.45 * avg_eye_rx;
    let el_rx = 0.55 * el_rx + 0.45 * avg_eye_rx;
    let er_ry = 0.55 * er_ry + 0.45 * avg_eye_ry;
    let el_ry = 0.55 * el_ry + 0.45 * avg_eye_ry;

    let (mouth_w_ratio, mouth_h_ratio) = {
        let mut tight_u = Vec::new();
        let mut tight_v = Vec::new();
        for (&mu, &mv) in mouth_us.iter().zip(&mouth_vs) {
            if (mu - mouth_lm_u).abs() <= eye_dist * 0.22 && (mv - mouth_lm_v).abs() <= eye_dist * 0.14 {
                tight_u.push(mu);
                tight_v.push(mv);
            }
        }
        if tight_u.len() >= 10 {
            tight_u.sort_by(|a, b| a.total_cmp(b));
            tight_v.sort_by(|a, b| a.total_cmp(b));
            let w_span = tight_u[(tight_u.len() - 1) * 88 / 100] - tight_u[(tight_u.len() - 1) * 12 / 100];
            let h_span = tight_v[(tight_v.len() - 1) * 90 / 100] - tight_v[(tight_v.len() - 1) * 10 / 100];
            let min_w = if is_sd { 0.065 } else { 0.10 };
            (
                (0.5 * w_span / eye_dist).clamp(min_w, 0.24),
                (0.5 * h_span / eye_dist).clamp(0.02, 0.12),
            )
        } else {
            (if is_sd { 0.10 } else { 0.165 }, 0.04)
        }
    };
    let reach_quantile = |mut vs: Vec<f64>, fallback: f64| -> f64 {
        if vs.is_empty() {
            return fallback;
        }
        vs.sort_by(|a, b| a.total_cmp(b));
        let idx = ((vs.len() - 1) * 95) / 100;
        vs[idx].max(fallback)
    };
    let side_r_max_v = reach_quantile(side_r_vs, frame.chin_v);
    let side_l_max_v = reach_quantile(side_l_vs, frame.chin_v);
    let back_max_v = reach_quantile(back_vs, frame.chin_v);

    let is_thin_closed_mouth =
        mouth_h_ratio / mouth_w_ratio.max(0.03) < 0.36 || mouth_h_ratio < 0.035;
    let mouth_open_ratio = if is_thin_closed_mouth {
        0.0
    } else {
        (mouth_open_px as f64 / mouth_total_px.max(1) as f64).clamp(0.0, 1.0)
    };

    let bang_coverage = (bangs_total as f64 / forehead_total.max(1) as f64).clamp(0.0, 1.0);
    let bang_side_sum = (bangs_u_left + bangs_u_right).max(1) as f64;
    let center_part_gap = (1.0 - (bangs_u_center as f64 * 2.2 / bang_side_sum).min(1.0)).clamp(0.0, 1.0);
    let bang_lr_bias = (bangs_u_right as f64 - bangs_u_left as f64) / bang_side_sum;

    let side_reach = ((0.5 * (side_r_max_v + side_l_max_v) - frame.chin_v) / hh).clamp(0.0, 1.4);
    let back_reach = ((back_max_v - frame.chin_v) / hh).clamp(0.0, 1.8);
    let shoulder_width_ratio = (clothes_max_abs_u / eye_dist).clamp(1.15, 3.1);
    let torso_cx_u = if torso_u_cnt >= 32 {
        (torso_u_sum / torso_u_cnt as f64).clamp(-eye_dist * 0.45, eye_dist * 0.45)
    } else {
        0.0
    };

    // Match best preset style per category from illustration metrics.
    let auto_face = if is_sd {
        if jaw_taper >= 0.72 {
            "chibi_plump"
        } else {
            "soft_round"
        }
    } else if jaw_taper > 0.78 && face_rx_ratio > 1.08 {
        "chibi_plump"
    } else if jaw_taper < 0.62 {
        "sharp_v"
    } else if jaw_taper > 0.73 {
        "soft_round"
    } else {
        "oval_anime"
    };

    let auto_hair_front = if bang_coverage < 0.22 {
        "parted_curtain"
    } else if center_part_gap > 0.44 {
        "center_part"
    } else if bang_lr_bias > 0.28 {
        "side_sweep_left"
    } else if bang_lr_bias < -0.28 {
        "side_sweep_right"
    } else if bang_coverage > 0.62 {
        "blunt_bangs"
    } else {
        "wispy_air"
    };

    let auto_hair_side = if side_reach >= 0.48 {
        "chest_locks"
    } else if side_reach >= 0.20 {
        "chin_bob"
    } else if side_reach >= 0.05 {
        "short_tuck"
    } else {
        "minimal"
    };

    let auto_hair_back = if back_reach >= 0.85 {
        if back_max_abs_u > shoulder_width_ratio * eye_dist * 0.92 {
            "long_wave"
        } else {
            "long_straight"
        }
    } else if back_reach >= 0.30 {
        "shoulder_bob"
    } else {
        "short_crop"
    };

    let eye_aspect = avg_eye_ry / avg_eye_rx.max(1.0);
    let auto_eye = if is_sd {
        if eye_aspect >= 0.76 {
            "round_bright"
        } else {
            "tareme_large"
        }
    } else if eye_aspect >= 0.62 {
        "tareme_large"
    } else if eye_aspect >= 0.50 {
        "round_bright"
    } else if eye_aspect <= 0.38 {
        "almond_sharp"
    } else {
        "droopy_gentle"
    };

    let auto_mouth = if is_sd
        && mouth_open_ratio > 0.30
        && mouth_h_ratio / mouth_w_ratio.max(0.03) >= 0.48
        && mouth_h_ratio >= 0.045
    {
        "open_o"
    } else if mouth_open_ratio > 0.30 {
        "open_happy"
    } else if mouth_w_ratio >= 0.175 {
        "soft_smile"
    } else {
        "calm_line"
    };

    let neck_dip = (clothes_min_v - frame.chin_v) / hh;
    let auto_clothes = if neck_dip > 0.32 {
        "v_neck_cardigan"
    } else if shoulder_width_ratio > 1.85 {
        "crew_sweater"
    } else if neck_dip > 0.22 {
        "sailor_collar"
    } else {
        "hoodie_casual"
    };

    let resolved = TemplateStyleSelection {
        proportion_mode: Some(proportion_mode),
        face_style: Some(resolve_style_choice(
            &overrides.face_style,
            auto_face,
            &["oval_anime", "soft_round", "sharp_v", "chibi_plump"],
        )),
        hair_front_style: Some(resolve_style_choice(
            &overrides.hair_front_style,
            auto_hair_front,
            &[
                "center_part",
                "blunt_bangs",
                "side_sweep_right",
                "side_sweep_left",
                "wispy_air",
                "parted_curtain",
            ],
        )),
        hair_side_style: Some(resolve_style_choice(
            &overrides.hair_side_style,
            auto_hair_side,
            &["chest_locks", "chin_bob", "short_tuck", "twin_frame", "minimal"],
        )),
        hair_back_style: Some(resolve_style_choice(
            &overrides.hair_back_style,
            auto_hair_back,
            &[
                "long_straight",
                "long_wave",
                "shoulder_bob",
                "short_crop",
                "twin_tails",
            ],
        )),
        eye_style: Some(resolve_style_choice(
            &overrides.eye_style,
            auto_eye,
            &["round_bright", "almond_sharp", "droopy_gentle", "tareme_large"],
        )),
        mouth_style: Some(resolve_style_choice(
            &overrides.mouth_style,
            auto_mouth,
            &["soft_smile", "cat_w", "open_happy", "open_o", "calm_line"],
        )),
        clothes_style: Some(resolve_style_choice(
            &overrides.clothes_style,
            auto_clothes,
            &[
                "crew_sweater",
                "sailor_collar",
                "v_neck_cardigan",
                "hoodie_casual",
                "blazer_suit",
            ],
        )),
    };

    // Sample clean eroded palette colors (exclude hair overlaps so bangs never hijack `skin_shadow`).
    let mut face_skin_only: Vec<bool> = face_skin_raw
        .iter()
        .zip(hair_raw.iter())
        .map(|(&f, &hr)| f && !hr)
        .collect();
    if face_skin_only.iter().filter(|&&v| v).count() < 16 {
        face_skin_only.clone_from(&face_skin_raw);
    }
    let face_eroded = erode_mask(&face_skin_only, w, h, 3);
    let skin_base = sample_quantile_color(rgba, &face_eroded, 0.62, [246, 224, 212]);
    let raw_skin_shadow = sample_quantile_color(rgba, &face_eroded, 0.24, scale_rgb(skin_base, 0.87));
    let skin_shadow = if rgb_dist(raw_skin_shadow, skin_base) <= 72.0
        && luma(raw_skin_shadow) >= luma(skin_base) * 0.68
    {
        raw_skin_shadow
    } else {
        scale_rgb(skin_base, 0.87)
    };

    let hair_eroded = erode_mask(&hair_raw, w, h, 2);
    let hair_base = sample_quantile_color(rgba, &hair_eroded, 0.56, [72, 62, 86]);
    let hair_shadow = sample_quantile_color(rgba, &hair_eroded, 0.22, scale_rgb(hair_base, 0.72));

    let sclera_white = sample_quantile_color(rgba, &sclera_raw, 0.82, [248, 247, 251]);
    let sclera_shadow = lerp_rgb(sclera_white, [186, 188, 214], 0.38);
    let lash_color = sample_quantile_color(rgba, &lash_raw, 0.14, [42, 34, 44]);
    let brow_color = sample_quantile_color(rgba, &brow_raw, 0.24, scale_rgb(hair_base, 0.68));
    let mouth_line = sample_quantile_color(rgba, &mouth_raw, 0.22, [132, 74, 78]);

    let clothes_eroded = erode_mask(&clothes_raw, w, h, 3);
    let clothes_primary = sample_quantile_color(rgba, &clothes_eroded, 0.55, [196, 198, 208]);

    let palette = PaletteInfo {
        skin_base,
        skin_shadow,
        hair_base,
        hair_shadow,
        sclera_white,
        sclera_shadow,
        lash_color,
        brow_color,
        mouth_line,
        clothes_primary,
    };

    let mut iris_r = fit_iris_from_evidence(
        rgba,
        w,
        h,
        frame,
        get_ev("eyeR_iris"),
        get_ev("eyeR_white"),
        Some(&eye_r_full_mask),
        IrisFitContext {
            eye_cu: er_u,
            eye_cv: er_v,
            eye_rx: er_rx,
            eye_ry: er_ry,
            is_sd,
        },
    );
    let mut iris_l = fit_iris_from_evidence(
        rgba,
        w,
        h,
        frame,
        get_ev("eyeL_iris"),
        get_ev("eyeL_white"),
        Some(&eye_l_full_mask),
        IrisFitContext {
            eye_cu: el_u,
            eye_cv: el_v,
            eye_rx: el_rx,
            eye_ry: el_ry,
            is_sd,
        },
    );
    if iris_r.sample_count < 12 && iris_l.sample_count >= 16 {
        iris_r.rx = iris_l.rx;
        iris_r.ry = iris_l.ry;
        iris_r.limbal = iris_l.limbal;
        iris_r.mid = iris_l.mid;
        iris_r.pupil = iris_l.pupil;
    } else if iris_l.sample_count < 12 && iris_r.sample_count >= 16 {
        iris_l.rx = iris_r.rx;
        iris_l.ry = iris_r.ry;
        iris_l.limbal = iris_r.limbal;
        iris_l.mid = iris_r.mid;
        iris_l.pupil = iris_r.pupil;
    }

    let iris_r_ratio = 0.5 * (iris_r.rx + iris_l.rx) / eye_dist;
    let metrics = FittedCharacterMetrics {
        face_rx_ratio,
        jaw_taper,
        eye_rx_ratio: 0.5 * (er_rx + el_rx) / eye_dist,
        eye_ry_ratio: 0.5 * (er_ry + el_ry) / eye_dist,
        iris_radius_ratio: iris_r_ratio,
        mouth_width_ratio: mouth_w_ratio,
        mouth_open_ratio,
        bang_coverage,
        center_part_gap,
        side_hair_reach: side_reach,
        back_hair_reach: back_reach,
        shoulder_width_ratio,
    };

    let mut out = AnalyzedEvidence {
        fg_alpha,
        ev_by_id,
        metrics,
        resolved,
        is_sd,
        match_score: 0.0,
        palette,
        eye_r_center: (er_u, er_v),
        eye_l_center: (el_u, el_v),
        eye_r_rx: er_rx,
        eye_r_ry: er_ry,
        eye_l_rx: el_rx,
        eye_l_ry: el_ry,
        iris_r,
        iris_l,
        mouth_u: mouth_lm_u,
        mouth_v: mouth_lm_v,
        torso_cx_u,
        clothes_top_v: clothes_min_v.clamp(frame.chin_v + hh * 0.14, frame.chin_v + hh * 0.52),
        clothes_top_u_bins,
        fg_min_u_bins,
        fg_max_u_bins,
        fg_max_v,
    };

    // Measure actual boundary fitting agreement between fitted templates and observed evidence.
    let mut face_total = 0usize;
    let mut face_aligned = 0usize;
    let mut eye_total = 0usize;
    let mut eye_aligned = 0usize;
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            let i = y * w + x;
            let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
            if face_skin_raw[i] {
                face_total += 1;
                if face_sdf(u, v, frame, &out) <= eye_dist * 0.08 {
                    face_aligned += 1;
                }
            }
            if eye_r_full_mask[i] {
                eye_total += 1;
                let rn = ((u - er_u) / (er_rx * 1.18).max(1.0))
                    .hypot((v - er_v) / (er_ry * 1.24).max(1.0));
                if rn <= 1.0 {
                    eye_aligned += 1;
                }
            }
            if eye_l_full_mask[i] {
                eye_total += 1;
                let rn = ((u - el_u) / (el_rx * 1.18).max(1.0))
                    .hypot((v - el_v) / (el_ry * 1.24).max(1.0));
                if rn <= 1.0 {
                    eye_aligned += 1;
                }
            }
        }
    }
    let face_cov = (face_total.min(180) as f64) / 180.0;
    let face_fit = if face_total == 0 {
        0.0
    } else {
        (face_aligned as f64 / face_total as f64) * face_cov
    };
    let eye_cov = (eye_total.min(100) as f64) / 100.0;
    let eye_fit = if eye_total == 0 {
        0.0
    } else {
        (eye_aligned as f64 / eye_total as f64) * eye_cov
    };
    let hair_fit = if bangs_total >= 40 { 0.6 } else { (bangs_total as f64 / 40.0) * 0.6 }
        + if side_reach > 0.05 || back_reach > 0.10 {
            0.4
        } else {
            0.0
        };
    out.match_score = (0.45 * face_fit + 0.35 * eye_fit + 0.20 * hair_fit).clamp(0.0, 0.99);
    out
}

/// Evaluates observed+interpolated neckline `v` coordinate across torso horizontal `u`.
fn sample_clothes_top_v(ev: &AnalyzedEvidence, frame: HeadFrame, u: f64) -> Option<f64> {
    let eye_dist = frame.eye_dist.max(8.0);
    let len = ev.clothes_top_u_bins.len();
    if len == 0 {
        return None;
    }
    let b = (((u + eye_dist * 3.0) / (eye_dist * 6.0).max(1.0)) * (len as f64))
        .floor()
        .clamp(0.0, (len - 1) as f64) as usize;
    let val = ev.clothes_top_u_bins[b];
    val.is_finite().then_some(val)
}

/// Evaluates continuous foreground silhouette width bounds at `v` via smooth bin interpolation.
fn sample_fg_bounds(ev: &AnalyzedEvidence, frame: HeadFrame, v: f64) -> (f64, f64) {
    let v_top = frame.crown_v - frame.hh * 0.12;
    let v_bot = (frame.chin_v + frame.hh * 1.65).max(frame.crown_v + frame.hh * 1.8);
    let t = ((v - v_top) / (v_bot - v_top).max(1.0)).clamp(0.0, 1.0);
    let f_idx = t * (ev.fg_min_u_bins.len() - 1) as f64;
    let i0 = (f_idx.floor() as usize).min(ev.fg_min_u_bins.len() - 1);
    let i1 = (i0 + 1).min(ev.fg_min_u_bins.len() - 1);
    let frac = f_idx - i0 as f64;
    let min_u = ev.fg_min_u_bins[i0] * (1.0 - frac) + ev.fg_min_u_bins[i1] * frac;
    let max_u = ev.fg_max_u_bins[i0] * (1.0 - frac) + ev.fg_max_u_bins[i1] * frac;
    (min_u, max_u)
}

/// Computes the lower bang silhouette boundary `v_bang(u)` according to `hair_front_style`.
fn bang_tip_v(style: &str, u_norm: f64, frame: HeadFrame) -> f64 {
    let eye_top = frame.crown_v + frame.hh * 0.52;
    let brow_line = eye_top - frame.hh * 0.14;
    match style {
        "blunt_bangs" => {
            eye_top - frame.hh * 0.02 + u_norm.powi(2) * frame.hh * 0.04
        }
        "center_part" => {
            // Signature curved M-opening right above nose bridge exposing brow/forehead.
            let center_arch = (-u_norm * u_norm * 14.0).exp();
            let mid_lobe = (-(u_norm.abs() - 0.52).powi(2) * 10.0).exp();
            brow_line - frame.hh * 0.22 * center_arch + frame.hh * 0.14 * mid_lobe
        }
        "parted_curtain" => {
            let center_arch = (-u_norm * u_norm * 5.0).exp();
            brow_line - frame.hh * 0.30 * center_arch + u_norm.abs() * frame.hh * 0.18
        }
        "side_sweep_right" => {
            // Swept downward toward character right (negative u).
            brow_line + (0.4 - u_norm).clamp(-0.6, 1.0) * frame.hh * 0.12
        }
        "side_sweep_left" => {
            // Swept downward toward character left (positive u).
            brow_line + (0.4 + u_norm).clamp(-0.6, 1.0) * frame.hh * 0.12
        }
        _ => {
            // wispy_air: gentle tooth lobes across forehead.
            let wisp = (u_norm * 9.0).cos().abs();
            brow_line + frame.hh * (0.04 + 0.08 * wisp)
        }
    }
}

fn mouth_radii(ev: &AnalyzedEvidence, eye_dist: f64, mouth_style: &str) -> (f64, f64) {
    if mouth_style == "open_o" {
        let min_rx = if ev.is_sd { 0.065 } else { 0.11 };
        let m_rx = (ev.metrics.mouth_width_ratio * eye_dist).clamp(eye_dist * min_rx, eye_dist * 0.15);
        let m_ry = (m_rx * 0.88).clamp(eye_dist * 0.042, eye_dist * 0.11);
        (m_rx, m_ry)
    } else {
        let min_rx = if ev.is_sd { 0.095 } else { 0.13 };
        let m_rx = (ev.metrics.mouth_width_ratio * eye_dist).clamp(eye_dist * min_rx, eye_dist * 0.22);
        (m_rx, eye_dist * 0.042)
    }
}

/// Evaluates canonical SDF distance (`<= 0` inside) for a face silhouette style in `HeadFrame`.
fn face_sdf(u: f64, v: f64, frame: HeadFrame, ev: &AnalyzedEvidence) -> f64 {
    let eye_dist = frame.eye_dist;
    let hh = frame.hh;
    let style = ev.resolved.face_style.as_deref().unwrap_or("oval_anime");
    let (cheek_scale, chin_taper_mul, chin_round) = match style {
        "chibi_plump" => (1.10, 0.86, 0.24),
        "soft_round" => (1.04, 0.78, 0.16),
        "sharp_v" => (1.00, 0.56, 0.06),
        _ => (1.00, 0.68, 0.12),
    };
    let max_rx_mult = if ev.is_sd { 1.44 } else { 1.24 };
    let rx = (ev.metrics.face_rx_ratio * eye_dist * cheek_scale).clamp(eye_dist * 0.96, eye_dist * max_rx_mult);
    let top_v = frame.crown_v + hh * if ev.is_sd { 0.08 } else { 0.10 };
    let bot_v = if ev.is_sd {
        frame.chin_v.max(ev.mouth_v + eye_dist * 0.24) + hh * 0.03
    } else {
        frame.chin_v + hh * 0.012
    };
    let cy = 0.5 * (top_v + bot_v);
    let ry = 0.5 * (bot_v - top_v).max(eye_dist * 0.85);

    let y_norm = ((v - cy) / ry).clamp(-1.3, 1.3);
    // Narrow jaw smoothly below eye line while keeping upper cranium & chibi cheeks full.
    let lower_t = if ev.is_sd {
        smoothstep(0.08, 0.98, y_norm)
    } else {
        smoothstep(-0.05, 0.96, y_norm)
    };
    let half_w = rx * (1.0 - lower_t * (1.0 - chin_taper_mul + chin_round * 0.15));
    let dx = (u.abs() / half_w.max(1.0)).clamp(0.0, 2.0);
    let dy = y_norm;
    let oval = (dx * dx + dy * dy).sqrt() - 1.0;
    oval * rx.min(ry)
}

/// Evaluates smooth sclera SDF (`eye*_white`) shaped by `eyeStyle`.
fn eye_white_sdf(
    u: f64,
    v: f64,
    eye_box: (f64, f64, f64, f64),
    is_left_eye: bool,
    style: &str,
    is_sd: bool,
) -> f64 {
    let (cu, cv, rx, ry) = eye_box;
    let du = (u - cu) / rx.max(1.0);
    // Outward sign: +1 toward outer temple of each eye.
    let outer_sign = if is_left_eye { 1.0 } else { -1.0 };
    let skew_v = match style {
        "almond_sharp" => -du * outer_sign * 0.14 * ry,
        "droopy_gentle" | "tareme_large" => du * outer_sign * 0.12 * ry,
        _ => 0.0,
    };
    let v_scale = match style {
        "tareme_large" => if is_sd { 1.10 } else { 1.08 },
        "almond_sharp" => 0.82,
        _ => if is_sd { 1.04 } else { 0.96 },
    };
    let dv = (v - (cv + skew_v)) / (ry * v_scale).max(1.0);
    let dist_norm = (du * du + dv * dv).sqrt() - 1.0;
    dist_norm * rx.min(ry)
}

#[inline]
fn rgb_dist(a: [u8; 3], b: [u8; 3]) -> f64 {
    let dr = a[0] as f64 - b[0] as f64;
    let dg = a[1] as f64 - b[1] as f64;
    let db = a[2] as f64 - b[2] as f64;
    (dr * dr + dg * dg + db * db).sqrt()
}

#[inline]
fn pix_rgb(rgba: &[u8], i: usize) -> [u8; 3] {
    let k = i * 4;
    [rgba[k], rgba[k + 1], rgba[k + 2]]
}

#[inline]
fn rgb_chroma(c: [u8; 3]) -> f64 {
    let max_c = c[0].max(c[1]).max(c[2]) as f64;
    let min_c = c[0].min(c[1]).min(c[2]) as f64;
    max_c - min_c
}

#[inline]
fn is_warm_blush_or_skin(c: [u8; 3], skin_base: [u8; 3]) -> bool {
    let dist = rgb_dist(c, skin_base);
    if dist < 34.0 {
        return true;
    }
    // Warm rosy cheek blush stays warm-reddish (`r > b + 16`) with relatively high lightness.
    c[0] >= skin_base[0].saturating_sub(18)
        && c[0] as i32 - c[2] as i32 >= 16
        && luma(c) >= luma(skin_base) * 0.72
        && dist < 78.0
}

#[inline]
fn is_dark_lineart(c: [u8; 3]) -> bool {
    luma(c) < 76.0 && rgb_chroma(c) < 28.0
}

#[inline]
fn chroma_normalized_dist(a: [u8; 3], b: [u8; 3]) -> f64 {
    let va = [a[0] as f64 + 18.0, a[1] as f64 + 18.0, a[2] as f64 + 18.0];
    let vb = [b[0] as f64 + 18.0, b[1] as f64 + 18.0, b[2] as f64 + 18.0];
    let na = (va[0] * va[0] + va[1] * va[1] + va[2] * va[2]).sqrt().max(1.0);
    let nb = (vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2]).sqrt().max(1.0);
    let dr = va[0] / na - vb[0] / nb;
    let dg = va[1] / na - vb[1] / nb;
    let db = va[2] / na - vb[2] / nb;
    (dr * dr + dg * dg + db * db).sqrt() * 255.0
}

/// Collects up to `max_modes` well-separated modal RGB clusters for pixels passing `pred`.
fn collect_cluster_modes<F: Fn(usize) -> bool>(
    rgba: &[u8],
    n: usize,
    max_modes: usize,
    min_sep: f64,
    pred: F,
) -> Vec<[u8; 3]> {
    let mut bins: HashMap<(u8, u8, u8), (usize, [u64; 3])> = HashMap::new();
    for i in 0..n {
        if !pred(i) {
            continue;
        }
        let c = pix_rgb(rgba, i);
        let key = (c[0] >> 4, c[1] >> 4, c[2] >> 4);
        let entry = bins.entry(key).or_insert((0, [0, 0, 0]));
        entry.0 += 1;
        entry.1[0] += c[0] as u64;
        entry.1[1] += c[1] as u64;
        entry.1[2] += c[2] as u64;
    }
    let mut sorted: Vec<(usize, [u8; 3])> = bins
        .into_values()
        .filter(|(cnt, _)| *cnt > 0)
        .map(|(cnt, sum)| {
            (
                cnt,
                [
                    (sum[0] / cnt as u64) as u8,
                    (sum[1] / cnt as u64) as u8,
                    (sum[2] / cnt as u64) as u8,
                ],
            )
        })
        .collect();
    sorted.sort_by_key(|a| std::cmp::Reverse(a.0));
    let min_keep = sorted.first().map(|a| (a.0 / 28).max(8)).unwrap_or(1);
    let mut modes: Vec<[u8; 3]> = Vec::new();
    for (cnt, mean) in sorted {
        if cnt < min_keep && !modes.is_empty() {
            break;
        }
        if modes.iter().all(|&m| rgb_dist(m, mean) >= min_sep) {
            modes.push(mean);
            if modes.len() >= max_modes.max(1) {
                break;
            }
        }
    }
    modes
}

/// Estimates true modal chromatic color from pixels passing `pred` without outline noise.
fn estimate_modal_rgb<F: Fn(usize) -> bool>(rgba: &[u8], n: usize, fallback: [u8; 3], pred: F) -> [u8; 3] {
    collect_cluster_modes(rgba, n, 1, 0.0, pred)
        .into_iter()
        .next()
        .unwrap_or(fallback)
}

type ColorComponent = (Vec<usize>, [u8; 3], f64, f64, f64, f64);
type HeadAccTuple = (Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>);

/// Collects 8-connected components on `mask` clustered by local color distance (`max_color_dist`).
fn connected_color_components(
    rgba: &[u8],
    candidate: &[bool],
    w: usize,
    h: usize,
    max_color_dist: f64,
    min_area: usize,
) -> Vec<ColorComponent> {
    let n = w * h;
    let mut visited = vec![false; n];
    let mut comps = Vec::new();
    let mut queue = VecDeque::new();

    for start in 0..n {
        if !candidate[start] || visited[start] {
            continue;
        }
        let seed_c = pix_rgb(rgba, start);
        visited[start] = true;
        queue.push_back(start);
        let mut pixels = Vec::new();
        let mut sum_rgb = [0u64; 3];
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut min_x = w as f64;
        let mut max_x = 0.0f64;
        let mut min_y = h as f64;
        let mut max_y = 0.0f64;

        while let Some(i) = queue.pop_front() {
            let x = i % w;
            let y = i / w;
            let c = pix_rgb(rgba, i);
            pixels.push(i);
            sum_rgb[0] += c[0] as u64;
            sum_rgb[1] += c[1] as u64;
            sum_rgb[2] += c[2] as u64;
            sum_x += x as f64;
            sum_y += y as f64;
            min_x = min_x.min(x as f64);
            max_x = max_x.max(x as f64);
            min_y = min_y.min(y as f64);
            max_y = max_y.max(y as f64);

            let mean = [
                (sum_rgb[0] / pixels.len() as u64) as u8,
                (sum_rgb[1] / pixels.len() as u64) as u8,
                (sum_rgb[2] / pixels.len() as u64) as u8,
            ];
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
                    if visited[ni] || !candidate[ni] {
                        continue;
                    }
                    let nc = pix_rgb(rgba, ni);
                    if rgb_dist(nc, seed_c) <= max_color_dist || rgb_dist(nc, mean) <= max_color_dist {
                        visited[ni] = true;
                        queue.push_back(ni);
                    }
                }
            }
        }
        if pixels.len() >= min_area {
            let cnt = pixels.len() as u64;
            let mean = [
                (sum_rgb[0] / cnt) as u8,
                (sum_rgb[1] / cnt) as u8,
                (sum_rgb[2] / cnt) as u8,
            ];
            comps.push((
                pixels,
                mean,
                sum_x / cnt as f64,
                sum_y / cnt as f64,
                max_x - min_x + 1.0,
                max_y - min_y + 1.0,
            ));
        }
    }
    comps
}

/// Absorbs adjacent dark/anti-aliased outline pixels (<= `radius` px) bordering a wearable component mask.
#[allow(clippy::too_many_arguments)]
fn attach_adjacent_outline(
    mask: &mut [bool],
    rgba: &[u8],
    fg_alpha: &[u8],
    blocked: &[bool],
    w: usize,
    h: usize,
    radius: usize,
    max_outline_luma: f64,
) {
    let mut frontier: VecDeque<(usize, usize)> = VecDeque::new();
    for (i, &m) in mask.iter().enumerate() {
        if m {
            frontier.push_back((i, 0));
        }
    }
    while let Some((i, d)) = frontier.pop_front() {
        if d >= radius {
            continue;
        }
        let x = i % w;
        let y = i / w;
        for nb in [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ]
        .into_iter()
        .flatten()
        {
            if mask[nb] || blocked[nb] || fg_alpha[nb] == 0 {
                continue;
            }
            let c = pix_rgb(rgba, nb);
            if luma(c) < max_outline_luma && rgb_chroma(c) < 48.0 {
                mask[nb] = true;
                frontier.push_back((nb, d + 1));
            }
        }
    }
}

struct WearableExtractionCtx<'a> {
    rgba: &'a [u8],
    w: usize,
    h: usize,
    frame: HeadFrame,
    ev: &'a AnalyzedEvidence,
    ev_face: &'a [bool],
    ev_hair_all: &'a [bool],
    ev_eye_r_all: &'a [bool],
    ev_eye_l_all: &'a [bool],
    ev_mouth_all: &'a [bool],
    ev_brow_r: &'a [bool],
    ev_brow_l: &'a [bool],
    ev_neck: &'a [bool],
    face_rx: f64,
}

fn extract_glasses_mask(ctx: &WearableExtractionCtx<'_>, true_hair_rgb: [u8; 3]) -> Option<Vec<u8>> {
    let n = ctx.w * ctx.h;
    let eye_dist = ctx.frame.eye_dist.max(8.0);
    let (er_u, er_v) = ctx.ev.eye_r_center;
    let (el_u, el_v) = ctx.ev.eye_l_center;
    let er_rx = ctx.ev.eye_r_rx.max(6.0);
    let er_ry = ctx.ev.eye_r_ry.max(6.0);
    let el_rx = ctx.ev.eye_l_rx.max(6.0);
    let el_ry = ctx.ev.eye_l_ry.max(6.0);
    let eye_mid_v = 0.5 * (er_v + el_v);
    let skin = ctx.ev.palette.skin_base;

    let mut bridge_cnt = 0usize;
    let mut upper_gap_hair_cnt = 0usize;
    let mut r_rim_cnt = 0usize;
    let mut l_rim_cnt = 0usize;
    let mut r_min_u = f64::INFINITY;
    let mut r_max_u = f64::NEG_INFINITY;
    let mut l_min_u = f64::INFINITY;
    let mut l_max_u = f64::NEG_INFINITY;
    let mut r_radii = Vec::new();
    let mut l_radii = Vec::new();
    let mut rim_seed_rgb = Vec::new();

    for y in 0..ctx.h {
        for x in 0..ctx.w {
            let i = y * ctx.w + x;
            if ctx.ev.fg_alpha[i] <= 32 {
                continue;
            }
            let c = pix_rgb(ctx.rgba, i);
            let (u, v) = ctx.frame.to_local(DVec2::new(x as f64, y as f64));

            // Bridge band: narrow strip between left & right eye inner corners.
            if u.abs() <= eye_dist * 0.22 && (v - eye_mid_v).abs() <= eye_dist * 0.14 {
                if !is_warm_blush_or_skin(c, skin) {
                    let is_solid_hair = ctx.ev_hair_all[i] && rgb_dist(c, true_hair_rgb) <= 32.0;
                    if !is_solid_hair {
                        bridge_cnt += 1;
                        if rim_seed_rgb.len() < 400 && !is_dark_lineart(c) {
                            rim_seed_rgb.push(c);
                        }
                    }
                }
            } else if u.abs() <= eye_dist * 0.20
                && v < eye_mid_v - eye_dist * 0.16
                && v >= eye_mid_v - eye_dist * 0.34
                && !is_warm_blush_or_skin(c, skin)
                && rgb_dist(c, true_hair_rgb) <= 38.0
            {
                upper_gap_hair_cnt += 1;
            }

            // Bilateral lower-orbit rim arcs outside iris & sclera interior (checks `[0.76..=1.42]` for wide SD/LD sockets).
            let iris_r_rn = ((u - ctx.ev.iris_r.center_u) / ctx.ev.iris_r.rx.max(2.0))
                .hypot((v - ctx.ev.iris_r.center_v) / ctx.ev.iris_r.ry.max(2.0));
            let iris_l_rn = ((u - ctx.ev.iris_l.center_u) / ctx.ev.iris_l.rx.max(2.0))
                .hypot((v - ctx.ev.iris_l.center_v) / ctx.ev.iris_l.ry.max(2.0));
            for (is_left, cu, cv, rx, ry, irn) in [
                (false, er_u, er_v, er_rx, er_ry, iris_r_rn),
                (true, el_u, el_v, el_rx, el_ry, iris_l_rn),
            ] {
                let rn = ((u - cu) / rx).hypot((v - cv) / ry);
                if (0.76..=1.42).contains(&rn)
                    && irn >= 0.96
                    && v >= cv + ry * 0.22
                    && v <= cv + ry * 1.48
                {
                    if is_warm_blush_or_skin(c, skin) || rgb_dist(c, true_hair_rgb) <= 28.0 {
                        continue;
                    }
                    if rim_seed_rgb.len() < 400 && !is_dark_lineart(c) {
                        rim_seed_rgb.push(c);
                    }
                    if is_left {
                        l_rim_cnt += 1;
                        l_min_u = l_min_u.min(u);
                        l_max_u = l_max_u.max(u);
                        l_radii.push(rn);
                    } else {
                        r_rim_cnt += 1;
                        r_min_u = r_min_u.min(u);
                        r_max_u = r_max_u.max(u);
                        r_radii.push(rn);
                    }
                }
            }
        }
    }

    let min_rim_px = (eye_dist * 0.08).round().max(5.0) as usize;
    let min_bridge_px = (eye_dist * 0.05).round().max(4.0) as usize;
    let r_span = if r_max_u > r_min_u { r_max_u - r_min_u } else { 0.0 };
    let l_span = if l_max_u > l_min_u { l_max_u - l_min_u } else { 0.0 };
    let has_glasses = bridge_cnt >= min_bridge_px
        && r_rim_cnt >= min_rim_px
        && l_rim_cnt >= min_rim_px
        && r_span >= er_rx * 0.72
        && l_span >= el_rx * 0.72
        && upper_gap_hair_cnt <= bridge_cnt * 2 + 8;

    if !has_glasses {
        return None;
    }

    let median_rn = |mut v: Vec<f64>| -> f64 {
        if v.is_empty() {
            return 1.08;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2].clamp(0.92, 1.24)
    };
    let r_est = median_rn(r_radii);
    let l_est = median_rn(l_radii);

    let frame_color = if rim_seed_rgb.is_empty() {
        ctx.ev.palette.lash_color
    } else {
        let mut rs: Vec<u8> = rim_seed_rgb.iter().map(|c| c[0]).collect();
        let mut gs: Vec<u8> = rim_seed_rgb.iter().map(|c| c[1]).collect();
        let mut bs: Vec<u8> = rim_seed_rgb.iter().map(|c| c[2]).collect();
        rs.sort_unstable();
        gs.sort_unstable();
        bs.sort_unstable();
        let mid = rim_seed_rgb.len() / 2;
        [rs[mid], gs[mid], bs[mid]]
    };

    let sclera_luma = luma(ctx.ev.palette.sclera_white);
    let mut cand = vec![false; n];
    let mut seeds = vec![false; n];
    for y in 0..ctx.h {
        for x in 0..ctx.w {
            let i = y * ctx.w + x;
            let fg_a = ctx.ev.fg_alpha[i];
            if fg_a <= 32 || ctx.ev_brow_r[i] || ctx.ev_brow_l[i] {
                continue;
            }
            let c = pix_rgb(ctx.rgba, i);
            let (u, v) = ctx.frame.to_local(DVec2::new(x as f64, y as f64));
            let iris_r_rn = ((u - ctx.ev.iris_r.center_u) / ctx.ev.iris_r.rx.max(2.0))
                .hypot((v - ctx.ev.iris_r.center_v) / ctx.ev.iris_r.ry.max(2.0));
            let iris_l_rn = ((u - ctx.ev.iris_l.center_u) / ctx.ev.iris_l.rx.max(2.0))
                .hypot((v - ctx.ev.iris_l.center_v) / ctx.ev.iris_l.ry.max(2.0));
            if iris_r_rn < 0.92 || iris_l_rn < 0.92 {
                continue;
            }

            let is_skin = is_warm_blush_or_skin(c, skin);
            if is_skin {
                continue;
            }
            let in_bridge = u.abs() <= eye_dist * 0.25
                && (v - eye_mid_v).abs() <= eye_dist * 0.16
                && rgb_dist(c, true_hair_rgb) > 28.0;

            let matches_rim_stroke = rgb_dist(c, frame_color) <= 72.0
                || chroma_normalized_dist(c, frame_color) <= 22.0
                || (!ctx.ev_hair_all[i] && luma(c) < sclera_luma * 0.90)
                || is_dark_lineart(c);

            let check_rim = |cu: f64, cv: f64, rx: f64, ry: f64, est_r: f64, temple_cond: bool| -> (bool, bool) {
                let rn = ((u - cu) / rx).hypot((v - cv) / ry);
                let in_ring = ((est_r - 0.20).max(0.84)..=(est_r + 0.18).min(1.30)).contains(&rn);
                let in_lower = (0.78..=1.34).contains(&rn) && v >= cv + ry * 0.14;
                if !in_ring && !in_lower && !temple_cond {
                    return (false, false);
                }
                if !matches_rim_stroke {
                    return (false, false);
                }
                // Guard upper-interior bangs (`rn < est_r - 0.12 && v < cv`) so thick bangs don't leak into rim.
                if !temple_cond && v < cv && rn < est_r - 0.10 && ctx.ev_hair_all[i] && rgb_dist(c, true_hair_rgb) <= 44.0 {
                    return (false, false);
                }
                let is_seed = in_lower && rgb_dist(c, true_hair_rgb) > 28.0;
                (true, is_seed)
            };

            let r_temple = v >= er_v - er_ry * 0.30
                && v <= er_v + er_ry * 0.35
                && u <= er_u - er_rx * 0.85
                && u >= -ctx.face_rx * 1.18;
            let l_temple = v >= el_v - el_ry * 0.30
                && v <= el_v + el_ry * 0.35
                && u >= el_u + el_rx * 0.85
                && u <= ctx.face_rx * 1.18;

            let (r_ok, r_seed) = check_rim(er_u, er_v, er_rx, er_ry, r_est, r_temple);
            let (l_ok, l_seed) = check_rim(el_u, el_v, el_rx, el_ry, l_est, l_temple);

            if in_bridge || r_ok || l_ok {
                cand[i] = true;
                if in_bridge || r_seed || l_seed {
                    seeds[i] = true;
                }
            }
        }
    }

    // Propagate strictly from confirmed lower-rim & bridge seeds across 8-connected rim candidates
    // so full 360-degree round frames connect cleanly without jumping skin gaps onto eyebrows.
    let mut visited = vec![false; n];
    let mut q = VecDeque::new();
    for (i, &s) in seeds.iter().enumerate() {
        if s {
            visited[i] = true;
            q.push_back(i);
        }
    }
    let mut out = vec![0u8; n];
    let mut count = 0usize;
    while let Some(i) = q.pop_front() {
        out[i] = ctx.ev.fg_alpha[i];
        count += 1;
        let x = i % ctx.w;
        let y = i / ctx.w;
        for dy in -1isize..=1 {
            let ny = y as isize + dy;
            if ny < 0 || ny >= ctx.h as isize {
                continue;
            }
            for dx in -1isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as isize + dx;
                if nx < 0 || nx >= ctx.w as isize {
                    continue;
                }
                let ni = ny as usize * ctx.w + nx as usize;
                if cand[ni] && !visited[ni] {
                    visited[ni] = true;
                    q.push_back(ni);
                }
            }
        }
    }
    (count >= min_rim_px * 2).then_some(out)
}

fn extract_hat_and_head_acc(
    ctx: &WearableExtractionCtx<'_>,
    hair_palette: &[[u8; 3]],
    glasses_mask: Option<&[u8]>,
) -> HeadAccTuple {
    let n = ctx.w * ctx.h;
    let eye_dist = ctx.frame.eye_dist.max(8.0);
    let hh = ctx.frame.hh.max(12.0);
    let skin = ctx.ev.palette.skin_base;
    let skin_shadow = ctx.ev.palette.skin_shadow;
    let hair_shadow = ctx.ev.palette.hair_shadow;
    let eye_min_v = ctx.ev.eye_r_center.1.min(ctx.ev.eye_l_center.1);
    let eye_max_v = ctx.ev.eye_r_center.1.max(ctx.ev.eye_l_center.1);

    let matches_hair_family = |c: [u8; 3]| -> bool {
        let shadow_d = rgb_dist(c, hair_shadow);
        if shadow_d < 56.0 {
            return true;
        }
        hair_palette.iter().any(|&hc| {
            let d = rgb_dist(c, hc);
            let cd = chroma_normalized_dist(c, hc);
            d < 60.0 || (cd < 22.0 && (luma(c) - luma(hc)).abs() < 95.0)
        })
    };

    // 1. Mark head-region pixels that genuinely diverge from hair palette, skin, eyes, brows, mouth, glasses.
    let mut blocked = vec![false; n];
    let mut cand = vec![false; n];
    for y in 0..ctx.h {
        for x in 0..ctx.w {
            let i = y * ctx.w + x;
            if ctx.ev.fg_alpha[i] <= 32 {
                continue;
            }
            if glasses_mask.is_some_and(|g| g[i] > 0)
                || ctx.ev_eye_r_all[i]
                || ctx.ev_eye_l_all[i]
                || ctx.ev_brow_r[i]
                || ctx.ev_brow_l[i]
                || ctx.ev_mouth_all[i]
                || ctx.ev_neck[i]
            {
                blocked[i] = true;
                continue;
            }
            let (u, v) = ctx.frame.to_local(DVec2::new(x as f64, y as f64));
            if v > ctx.frame.chin_v + hh * 0.14 {
                continue;
            }
            let c = pix_rgb(ctx.rgba, i);
            let fd = face_sdf(u, v, ctx.frame, ctx.ev);
            let skin_like = rgb_dist(c, skin) <= 50.0 || rgb_dist(c, skin_shadow) <= 50.0;
            if (ctx.ev_face[i] && skin_like)
                || (skin_like && fd <= eye_dist * 0.14 && v >= ctx.frame.crown_v + hh * 0.22)
            {
                blocked[i] = true;
                continue;
            }
            if is_dark_lineart(c) || matches_hair_family(c) {
                continue;
            }
            // Exclude clothing pixels touching chest/collar center below chin unless out along cheeks/upper hood.
            if v >= ctx.frame.chin_v - hh * 0.04 && u.abs() < ctx.face_rx * 0.96 {
                continue;
            }
            cand[i] = true;
        }
    }

    let min_small_area = ((eye_dist * 0.05).powi(2) as usize).max(24);
    let hat_min_area = ((eye_dist * 0.36).powi(2) as usize).max(280);
    let comps = connected_color_components(ctx.rgba, &cand, ctx.w, ctx.h, 58.0, min_small_area);
    if comps.is_empty() {
        return (None, None, None);
    }

    let mut hat_bool = vec![false; n];
    let mut has_hat = false;
    let mut ribbon_bool = vec![false; n];
    let mut has_ribbon = false;
    let mut hairpin_bool = vec![false; n];
    let mut has_hairpin = false;
    let mut hat_primary = [240u8, 240, 245];

    // Sort components largest-first so the primary hat/hood is claimed before trims/bows.
    let mut ordered = comps;
    ordered.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
    let mut leftover = Vec::new();

    for (pxs, mean, cx, cy, cw, ch) in ordered {
        let (med_u, med_v) = ctx.frame.to_local(DVec2::new(cx, cy));
        let mut min_v = f64::INFINITY;
        let mut min_u = f64::INFINITY;
        let mut max_u = f64::NEG_INFINITY;
        let mut non_edge_cnt = 0usize;
        for &pi in &pxs {
            let x = pi % ctx.w;
            let y = pi / ctx.w;
            let (pu, pv) = ctx.frame.to_local(DVec2::new(x as f64, y as f64));
            min_v = min_v.min(pv);
            min_u = min_u.min(pu);
            max_u = max_u.max(pu);
            let touches_bg = x == 0
                || x + 1 == ctx.w
                || y == 0
                || y + 1 == ctx.h
                || ctx.ev.fg_alpha[pi - 1] == 0
                || ctx.ev.fg_alpha[pi + 1] == 0
                || ctx.ev.fg_alpha[pi - ctx.w] == 0
                || ctx.ev.fg_alpha[pi + ctx.w] == 0;
            if !touches_bg {
                non_edge_cnt += 1;
            }
        }
        let interior_frac = non_edge_cnt as f64 / pxs.len().max(1) as f64;
        if interior_frac < 0.28 || cw < 3.5 || ch < 3.5 {
            continue;
        }
        let area = pxs.len();
        let straddles_crown = min_u <= ctx.face_rx * 0.30
            && max_u >= -ctx.face_rx * 0.30
            && (max_u - min_u) >= ctx.face_rx * 0.85;
        let is_primary_hat = area >= hat_min_area
            && min_v <= eye_min_v + hh * 0.05
            && med_v <= eye_max_v + hh * 0.10
            && (if !has_hat {
                straddles_crown
            } else {
                rgb_dist(mean, hat_primary) <= 68.0
            });

        if is_primary_hat {
            if !has_hat {
                hat_primary = mean;
            }
            has_hat = true;
            for pi in pxs {
                hat_bool[pi] = true;
            }
        } else {
            leftover.push((pxs, mean, med_u, med_v, min_v, cw, ch, interior_frac));
        }
    }

    // Absorb smaller detached hat pieces split by ribbons/straps before assigning head ornaments.
    for (pxs, mean, med_u, med_v, min_v, cw, ch, interior_frac) in leftover {
        if has_hat
            && min_v <= eye_min_v
            && med_v <= eye_max_v + hh * 0.06
            && (rgb_dist(mean, hat_primary) <= 72.0
                || (chroma_normalized_dist(mean, hat_primary) <= 16.0
                    && (luma(mean) - luma(hat_primary)).abs() <= 45.0))
        {
            for pi in pxs {
                hat_bool[pi] = true;
            }
            continue;
        }
        if med_v > ctx.frame.chin_v + hh * 0.06 || interior_frac < 0.32 {
            continue;
        }
        let touches_hat = has_hat
            && pxs.iter().any(|&pi| {
                let x = pi % ctx.w;
                let y = pi / ctx.w;
                (x > 0 && hat_bool[pi - 1])
                    || (x + 1 < ctx.w && hat_bool[pi + 1])
                    || (y > 0 && hat_bool[pi - ctx.w])
                    || (y + 1 < ctx.h && hat_bool[pi + ctx.w])
            });

        if !touches_hat && (med_v > eye_max_v + hh * 0.02 || med_u.abs() > ctx.face_rx * 1.28) {
            continue;
        }

        // Guard against thin wispy hair highlight streaks and same-hue darker hair shadows masquerading as hairpins.
        let min_hair_rgb = hair_palette
            .iter()
            .map(|&hc| rgb_dist(mean, hc))
            .fold(f64::INFINITY, f64::min);
        let min_hair_chr = hair_palette
            .iter()
            .map(|&hc| chroma_normalized_dist(mean, hc))
            .fold(f64::INFINITY, f64::min);
        let hair_max_luma = hair_palette
            .iter()
            .map(|&hc| luma(hc))
            .fold(0.0f64, f64::max);
        let fill_density = pxs.len() as f64 / (cw * ch).max(1.0);
        let long_wisp = !touches_hat && cw.max(ch) >= eye_dist * 0.22 && fill_density < 0.30;
        let same_hue_hair_shade = !touches_hat
            && min_hair_chr < 24.0
            && (luma(mean) <= hair_max_luma + 14.0 || (interior_frac < 0.44 && fill_density < 0.36));
        if long_wisp || same_hue_hair_shade || (min_hair_rgb < 88.0 && min_hair_chr < 26.0 && !touches_hat) {
            continue;
        }

        // Guard against outer background-bleed slivers along silhouette hair edges (`2d` opaque room walls).
        if !touches_hat {
            let mut bg_edges = 0usize;
            for &pi in &pxs {
                let x = pi % ctx.w;
                let y = pi / ctx.w;
                if x == 0
                    || x + 1 == ctx.w
                    || y == 0
                    || y + 1 == ctx.h
                    || ctx.ev.fg_alpha[pi - 1] == 0
                    || ctx.ev.fg_alpha[pi + 1] == 0
                    || ctx.ev.fg_alpha[pi - ctx.w] == 0
                    || ctx.ev.fg_alpha[pi + ctx.w] == 0
                {
                    bg_edges += 1;
                }
            }
            if bg_edges * 100 > pxs.len() * 14 {
                continue;
            }
        }

        let min_ornament_area = ((eye_dist * 0.055).powi(2) as usize).max(64);
        if pxs.len() < min_ornament_area || cw < 6.0 || ch < 6.0 || cw.max(ch) < 8.0 {
            continue;
        }
        let big_ornament = pxs.len() >= ((eye_dist * 0.22).powi(2) as usize).max(320);
        if touches_hat || big_ornament {
            has_ribbon = true;
            for pi in pxs {
                ribbon_bool[pi] = true;
            }
        } else {
            has_hairpin = true;
            for pi in pxs {
                hairpin_bool[pi] = true;
            }
        }
    }

    // Absorb small enclosed graphic details (e.g. cat-hood eyes & mouth) completely surrounded inside `hat`.
    if has_hat {
        let mut enclosed_free = vec![false; n];
        for y in 0..ctx.h {
            for x in 0..ctx.w {
                let i = y * ctx.w + x;
                if ctx.ev.fg_alpha[i] > 32 && !hat_bool[i] && !ribbon_bool[i] && !blocked[i] {
                    let (_, v) = ctx.frame.to_local(DVec2::new(x as f64, y as f64));
                    if v <= eye_min_v - ctx.ev.eye_r_ry * 0.55 {
                        enclosed_free[i] = true;
                    }
                }
            }
        }
        // BFS from exterior through `enclosed_free` to separate outside hair from interior hood details.
        let mut reachable = vec![false; n];
        let mut q = VecDeque::new();
        for y in 0..ctx.h {
            for x in 0..ctx.w {
                let i = y * ctx.w + x;
                if !enclosed_free[i] {
                    continue;
                }
                let at_outside = x == 0
                    || x + 1 == ctx.w
                    || y == 0
                    || y + 1 == ctx.h
                    || (!hat_bool[i - 1] && !ribbon_bool[i - 1] && !enclosed_free[i - 1])
                    || (!hat_bool[i + 1] && !ribbon_bool[i + 1] && !enclosed_free[i + 1])
                    || (!hat_bool[i - ctx.w] && !ribbon_bool[i - ctx.w] && !enclosed_free[i - ctx.w])
                    || (!hat_bool[i + ctx.w] && !ribbon_bool[i + ctx.w] && !enclosed_free[i + ctx.w]);
                if at_outside {
                    reachable[i] = true;
                    q.push_back(i);
                }
            }
        }
        while let Some(i) = q.pop_front() {
            let x = i % ctx.w;
            let y = i / ctx.w;
            for nb in [
                (x > 0).then(|| i - 1),
                (x + 1 < ctx.w).then(|| i + 1),
                (y > 0).then(|| i - ctx.w),
                (y + 1 < ctx.h).then(|| i + ctx.w),
            ]
            .into_iter()
            .flatten()
            {
                if enclosed_free[nb] && !reachable[nb] {
                    reachable[nb] = true;
                    q.push_back(nb);
                }
            }
        }
        for i in 0..n {
            if enclosed_free[i] && !reachable[i] {
                hat_bool[i] = true;
            }
        }

        let hat_outline_r = ((eye_dist * 0.015).round() as usize).clamp(3, 5);
        attach_adjacent_outline(
            &mut hat_bool,
            ctx.rgba,
            &ctx.ev.fg_alpha,
            &blocked,
            ctx.w,
            ctx.h,
            hat_outline_r,
            158.0,
        );
    }

    if has_ribbon {
        let mut blk = blocked.clone();
        for (i, &h_px) in hat_bool.iter().enumerate() {
            blk[i] |= h_px;
        }
        attach_adjacent_outline(
            &mut ribbon_bool,
            ctx.rgba,
            &ctx.ev.fg_alpha,
            &blk,
            ctx.w,
            ctx.h,
            2,
            118.0,
        );
    }
    if has_hairpin {
        let mut blk = blocked;
        for i in 0..n {
            blk[i] |= hat_bool[i] || ribbon_bool[i];
        }
        attach_adjacent_outline(
            &mut hairpin_bool,
            ctx.rgba,
            &ctx.ev.fg_alpha,
            &blk,
            ctx.w,
            ctx.h,
            1,
            118.0,
        );
    }

    let to_mask = |flag: bool, b: Vec<bool>| -> Option<Vec<u8>> {
        if !flag {
            return None;
        }
        let mut out = vec![0u8; n];
        let mut cnt = 0usize;
        for (i, &on) in b.iter().enumerate() {
            if on && ctx.ev.fg_alpha[i] > 0 {
                out[i] = ctx.ev.fg_alpha[i];
                cnt += 1;
            }
        }
        (cnt > 0).then_some(out)
    };

    (
        to_mask(has_hat, hat_bool),
        to_mask(has_hairpin, hairpin_bool),
        to_mask(has_ribbon, ribbon_bool),
    )
}

fn extract_outfit_sublayers(
    ctx: &WearableExtractionCtx<'_>,
    clothes_vis: &mut [u8],
    face_vis: &mut [u8],
    neck_vis: &mut [u8],
) -> Vec<(String, Vec<u8>)> {
    let n = ctx.w * ctx.h;
    let eye_dist = ctx.frame.eye_dist.max(8.0);
    let hh = ctx.frame.hh.max(12.0);
    let skin = ctx.ev.palette.skin_base;
    let skin_shadow = ctx.ev.palette.skin_shadow;

    let mut result: Vec<(String, Vec<u8>)> = Vec::new();

    let clothes_mode = estimate_modal_rgb(ctx.rgba, n, ctx.ev.palette.clothes_primary, |i| {
        clothes_vis[i] > 0 && !is_dark_lineart(pix_rgb(ctx.rgba, i))
    });
    let warm_skin = skin[0] > skin[2].saturating_add(3);

    // 1. Extract `clothes_hands`: compact warm skin blobs near chin/cuffs cleanly separated from cool cloth highlights.
    let mut skin_cand = vec![false; n];
    for y in 0..ctx.h {
        for x in 0..ctx.w {
            let i = y * ctx.w + x;
            if ctx.ev.fg_alpha[i] <= 32 {
                continue;
            }
            if clothes_vis[i] == 0 && neck_vis[i] == 0 && face_vis[i] == 0 {
                continue;
            }
            let (u, v) = ctx.frame.to_local(DVec2::new(x as f64, y as f64));
            if v < ctx.frame.chin_v - hh * 0.04 || v > ctx.frame.chin_v + hh * 1.15 {
                continue;
            }
            let c = pix_rgb(ctx.rgba, i);
            let d_skin = rgb_dist(c, skin).min(rgb_dist(c, skin_shadow));
            let d_clothes = rgb_dist(c, clothes_mode);
            let same_clothes_hue_highlight = chroma_normalized_dist(c, clothes_mode) <= 13.0
                && d_clothes <= 68.0
                && luma(c) >= luma(clothes_mode) - 8.0;
            let warmth_ok = !warm_skin || (c[0] as i32 + 2 >= c[2] as i32);
            let skin_like = !same_clothes_hue_highlight
                && d_skin <= 42.0
                && warmth_ok
                && d_skin < d_clothes * 0.72;
            if !skin_like {
                continue;
            }
            let in_face_core = face_vis[i] > 0 && face_sdf(u, v, ctx.frame, ctx.ev) <= -1.5;
            if !in_face_core {
                skin_cand[i] = true;
            }
        }
    }

    let min_hand_area = ((eye_dist * 0.09).powi(2) as usize).max(48);
    let max_hand_area = (eye_dist * eye_dist * 0.18) as usize;
    let hand_comps = connected_color_components(ctx.rgba, &skin_cand, ctx.w, ctx.h, 52.0, min_hand_area);
    let mut hands_bool = vec![false; n];
    let mut hands_cnt = 0usize;
    for (pxs, mean, cx, cy, cw, ch) in hand_comps {
        if pxs.len() > max_hand_area || cw > eye_dist * 0.58 {
            continue;
        }
        let (cu, _) = ctx.frame.to_local(DVec2::new(cx, cy));
        let fill_density = pxs.len() as f64 / (cw * ch).max(1.0);
        let mut interior_cnt = 0usize;
        let mut touches_outer_edge = false;
        // Reject centered V-neck collarbone chest skin (`touches_upper_neck`, `neck` border, or spanning center chest).
        let mut touches_upper_neck = false;
        let mut touches_clothes_edge = 0usize;
        let mut min_u = f64::INFINITY;
        let mut max_u = f64::NEG_INFINITY;
        for &pi in &pxs {
            let x = pi % ctx.w;
            let y = pi / ctx.w;
            if x <= 2 || x + 3 >= ctx.w || y + 3 >= ctx.h {
                touches_outer_edge = true;
            }
            let (u, v) = ctx.frame.to_local(DVec2::new(x as f64, y as f64));
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            if v <= ctx.frame.chin_v + hh * 0.26 && u.abs() <= eye_dist * 0.22 {
                touches_upper_neck = true;
            }
            if x > 0
                && x + 1 < ctx.w
                && y > 0
                && y + 1 < ctx.h
                && skin_cand[pi - 1]
                && skin_cand[pi + 1]
                && skin_cand[pi - ctx.w]
                && skin_cand[pi + ctx.w]
            {
                interior_cnt += 1;
            }
            for nb in [
                (x > 0).then(|| pi - 1),
                (x + 1 < ctx.w).then(|| pi + 1),
                (y > 0).then(|| pi - ctx.w),
                (y + 1 < ctx.h).then(|| pi + ctx.w),
            ]
            .into_iter()
            .flatten()
            {
                if neck_vis[nb] > 0 && !skin_cand[nb] {
                    touches_upper_neck = true;
                }
                if clothes_vis[nb] > 0 && !skin_cand[nb] {
                    touches_clothes_edge += 1;
                }
            }
        }
        let interior_frac = interior_cnt as f64 / pxs.len().max(1) as f64;
        let cool_border_glare =
            (!warm_skin || (mean[0] as i32 <= mean[2] as i32 + 4)) && (touches_outer_edge || cu.abs() > eye_dist * 1.32);
        let centered_chest =
            min_u < -eye_dist * 0.16 && max_u > eye_dist * 0.16 && (pxs.len() as f64) > eye_dist * eye_dist * 0.05;
        if !touches_upper_neck
            && !centered_chest
            && !cool_border_glare
            && touches_clothes_edge >= 6
            && fill_density >= 0.36
            && interior_frac >= 0.32
        {
            for &pi in &pxs {
                hands_bool[pi] = true;
                hands_cnt += 1;
            }
        }
    }

    if hands_cnt > 0 {
        let mut blocked_h = vec![false; n];
        for i in 0..n {
            if face_vis[i] > 0 {
                let (u, v) = ctx.frame.to_local(DVec2::new((i % ctx.w) as f64, (i / ctx.w) as f64));
                if face_sdf(u, v, ctx.frame, ctx.ev) <= -1.5 {
                    blocked_h[i] = true;
                }
            }
        }
        attach_adjacent_outline(&mut hands_bool, ctx.rgba, &ctx.ev.fg_alpha, &blocked_h, ctx.w, ctx.h, 2, 118.0);
        let mut hands_mask = vec![0u8; n];
        for (i, &on) in hands_bool.iter().enumerate() {
            if on && ctx.ev.fg_alpha[i] > 0 {
                hands_mask[i] = ctx.ev.fg_alpha[i];
                clothes_vis[i] = 0;
                face_vis[i] = 0;
                neck_vis[i] = 0;
            }
        }
        result.push(("clothes_hands".to_string(), hands_mask));
    }

    // 2. Multi-tone garment clustering (`clothes`, `clothes_inner`, `clothes_outer`, `clothes_collar`, `clothes_acc`).
    let mut interior_px = Vec::new();
    let mut bins: HashMap<(u8, u8, u8), (usize, [u64; 3])> = HashMap::new();
    for (i, &ca) in clothes_vis.iter().enumerate() {
        if ca == 0 {
            continue;
        }
        let c = pix_rgb(ctx.rgba, i);
        if is_dark_lineart(c) || (luma(c) < 118.0 && rgb_chroma(c) < 32.0) {
            continue;
        }
        interior_px.push(i);
        let key = (c[0] >> 4, c[1] >> 4, c[2] >> 4);
        let e = bins.entry(key).or_insert((0, [0, 0, 0]));
        e.0 += 1;
        e.1[0] += c[0] as u64;
        e.1[1] += c[1] as u64;
        e.1[2] += c[2] as u64;
    }

    let total_int = interior_px.len();
    if total_int < 300 {
        return result;
    }

    let mut sorted_bins: Vec<_> = bins
        .into_values()
        .map(|(cnt, sum)| {
            (
                cnt,
                [
                    (sum[0] / cnt as u64) as u8,
                    (sum[1] / cnt as u64) as u8,
                    (sum[2] / cnt as u64) as u8,
                ],
            )
        })
        .collect();
    sorted_bins.sort_by_key(|a| std::cmp::Reverse(a.0));

    let rgb_sat = |c: [u8; 3]| -> f64 {
        let mx = c[0].max(c[1]).max(c[2]) as f64;
        let mn = c[0].min(c[1]).min(c[2]) as f64;
        (mx - mn) / mx.max(1.0)
    };
    let min_cluster_px = (total_int * 8 / 1000).max(90);
    let mut centers: Vec<[u8; 3]> = Vec::new();
    for (cnt, mean) in &sorted_bins {
        if *cnt < min_cluster_px / 3 {
            continue;
        }
        if centers.iter().all(|&ex| {
            let chr = chroma_normalized_dist(ex, *mean);
            let dist = rgb_dist(ex, *mean);
            if rgb_sat(ex) < 0.28 && rgb_sat(*mean) < 0.28 && chr < 28.0 {
                dist >= 88.0
            } else {
                chr >= 20.0 || (dist >= 72.0 && chr >= 12.0)
            }
        }) {
            centers.push(*mean);
            if centers.len() >= 4 {
                break;
            }
        }
    }
    if centers.len() < 2 {
        return result;
    }

    // Assign interior clothes pixels to nearest palette cluster.
    let mut labels = vec![u8::MAX; n];
    let mut cluster_counts = vec![0usize; centers.len()];
    for &i in &interior_px {
        let c = pix_rgb(ctx.rgba, i);
        let mut best_k = 0usize;
        let mut best_d = f64::INFINITY;
        for (k_idx, &ctr) in centers.iter().enumerate() {
            let d = rgb_dist(c, ctr);
            if d < best_d {
                best_d = d;
                best_k = k_idx;
            }
        }
        labels[i] = best_k as u8;
        cluster_counts[best_k] += 1;
    }

    // Keep only clusters meeting minimum significant area; remap tiny ones to nearest large cluster.
    let valid_ks: Vec<usize> = cluster_counts
        .iter()
        .enumerate()
        .filter(|(_, &cnt)| cnt >= min_cluster_px)
        .map(|(k_idx, _)| k_idx)
        .collect();
    if valid_ks.len() < 2 {
        return result;
    }
    for &i in &interior_px {
        let lk = labels[i] as usize;
        if cluster_counts[lk] < min_cluster_px {
            let c = pix_rgb(ctx.rgba, i);
            if let Some(&nk) = valid_ks.iter().min_by(|&&a, &&b| {
                rgb_dist(c, centers[a])
                    .partial_cmp(&rgb_dist(c, centers[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                labels[i] = nk as u8;
            }
        }
    }

    // Propagate dark lineart / shadow pixels inside clothes_vis to nearest cluster label (up to 4px BFS).
    let mut q = VecDeque::new();
    for (i, &lbl) in labels.iter().enumerate() {
        if lbl != u8::MAX {
            q.push_back((i, 0u8));
        }
    }
    while let Some((i, d)) = q.pop_front() {
        if d >= 5 {
            continue;
        }
        let x = i % ctx.w;
        let y = i / ctx.w;
        let lbl = labels[i];
        for nb in [
            (x > 0).then(|| i - 1),
            (x + 1 < ctx.w).then(|| i + 1),
            (y > 0).then(|| i - ctx.w),
            (y + 1 < ctx.h).then(|| i + ctx.w),
        ]
        .into_iter()
        .flatten()
        {
            if clothes_vis[nb] > 0 && labels[nb] == u8::MAX {
                labels[nb] = lbl;
                q.push_back((nb, d + 1));
            }
        }
    }

    // Recount assigned clusters & find largest primary cluster (remains `clothes`).
    let mut final_counts = vec![0usize; centers.len()];
    let mut u_sum = vec![0.0f64; centers.len()];
    let mut v_sum = vec![0.0f64; centers.len()];
    let mut u_min = vec![f64::INFINITY; centers.len()];
    let mut u_max = vec![f64::NEG_INFINITY; centers.len()];
    for (i, &ca) in clothes_vis.iter().enumerate() {
        if ca == 0 {
            continue;
        }
        let lbl = labels[i] as usize;
        if lbl < centers.len() {
            let (u, v) = ctx.frame.to_local(DVec2::new((i % ctx.w) as f64, (i / ctx.w) as f64));
            final_counts[lbl] += 1;
            u_sum[lbl] += u;
            v_sum[lbl] += v;
            u_min[lbl] = u_min[lbl].min(u);
            u_max[lbl] = u_max[lbl].max(u);
        }
    }
    let Some(primary_k) = valid_ks.iter().copied().max_by_key(|&k| final_counts[k]) else {
        return result;
    };
    let primary_v = v_sum[primary_k] / final_counts[primary_k].max(1) as f64;
    let primary_span = (u_max[primary_k] - u_min[primary_k]).max(1.0);
    let total_cl: usize = final_counts.iter().sum();

    let mut taken_names: Vec<String> = Vec::new();
    let mut extra_masks: HashMap<String, Vec<u8>> = HashMap::new();
    let mut cc_seen = vec![false; n];
    for &k in &valid_ks {
        if k == primary_k || final_counts[k] < min_cluster_px {
            continue;
        }
        // Reject scattered 1-3px fold/crease shading bands that lack cohesive garment components.
        let mut interior_cnt = 0usize;
        let mut cc_sizes: Vec<usize> = Vec::new();
        for (i, &ca) in clothes_vis.iter().enumerate() {
            if ca == 0 || labels[i] as usize != k {
                continue;
            }
            let x = i % ctx.w;
            let y = i / ctx.w;
            let all_match = x > 0
                && x + 1 < ctx.w
                && y > 0
                && y + 1 < ctx.h
                && labels[i - 1] as usize == k
                && labels[i + 1] as usize == k
                && labels[i - ctx.w] as usize == k
                && labels[i + ctx.w] as usize == k;
            if all_match {
                interior_cnt += 1;
            }
            if !cc_seen[i] {
                cc_seen[i] = true;
                let mut stack = vec![i];
                let mut sz = 0usize;
                while let Some(cur) = stack.pop() {
                    sz += 1;
                    let cx_i = cur % ctx.w;
                    let cy_i = cur / ctx.w;
                    for nb in [
                        (cx_i > 0).then(|| cur - 1),
                        (cx_i + 1 < ctx.w).then(|| cur + 1),
                        (cy_i > 0).then(|| cur - ctx.w),
                        (cy_i + 1 < ctx.h).then(|| cur + ctx.w),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        if !cc_seen[nb] && clothes_vis[nb] > 0 && labels[nb] as usize == k {
                            cc_seen[nb] = true;
                            stack.push(nb);
                        }
                    }
                }
                cc_sizes.push(sz);
            }
        }
        cc_sizes.sort_by_key(|a| std::cmp::Reverse(*a));
        let top2_cc: usize = cc_sizes.iter().take(2).sum();
        if interior_cnt * 100 < final_counts[k] * 56 || top2_cc * 100 < final_counts[k] * 72 {
            continue;
        }
        let frac = final_counts[k] as f64 / total_cl.max(1) as f64;
        let med_u = u_sum[k] / final_counts[k] as f64;
        let med_v = v_sum[k] / final_counts[k] as f64;
        let span = u_max[k] - u_min[k];

        let candidate_name = if med_v <= primary_v + hh * 0.05
            && med_v <= ctx.frame.chin_v + hh * 0.65
            && span >= eye_dist * 0.55
        {
            "clothes_collar"
        } else if frac <= 0.16 && med_u.abs() <= eye_dist * 0.45 && med_v <= ctx.frame.chin_v + hh * 1.05 {
            "clothes_acc"
        } else if span >= primary_span * 0.96 && frac >= 0.18 {
            "clothes_outer"
        } else {
            "clothes_inner"
        };

        let final_name = if !taken_names.iter().any(|s| s == candidate_name) {
            candidate_name.to_string()
        } else {
            let mut n_idx = 2usize;
            while taken_names.iter().any(|s| s == &format!("{candidate_name}_{n_idx}")) {
                n_idx += 1;
            }
            format!("{candidate_name}_{n_idx}")
        };
        taken_names.push(final_name.clone());
        let mask_entry = extra_masks.entry(final_name).or_insert_with(|| vec![0u8; n]);
        for (i, ca) in clothes_vis.iter_mut().enumerate() {
            if *ca > 0 && labels[i] as usize == k {
                mask_entry[i] = *ca;
                *ca = 0;
            }
        }
    }

    for (name, mask) in extra_masks {
        result.push((name, mask));
    }
    result
}

/// Generates layered parametric masks (`masks`, `fill_masks`, `textures`) in Z-order plus separated hats/clothes/accessories.
#[allow(clippy::too_many_arguments)]
pub fn synthesize_template_bank(
    rgba: &[u8],
    w: usize,
    h: usize,
    landmarks: &Landmarks,
    points_28: &[Landmark28Point],
    evidence_ids: &[String],
    evidence_masks: &[Vec<u8>],
    evidence_provenances: &[String],
    evidence_confidences: &[f64],
    overrides: &TemplateStyleSelection,
) -> TemplateSynthesisOutput {
    let n = w * h;
    let frame = HeadFrame::from_landmarks(landmarks, w, h);
    let eye_dist = frame.eye_dist.max(8.0);
    let hh = frame.hh.max(12.0);

    let ev = analyze_evidence(
        rgba,
        w,
        h,
        landmarks,
        points_28,
        evidence_ids,
        evidence_masks,
        overrides,
    );

    let eye_style = ev.resolved.eye_style.as_deref().unwrap_or("round_bright");
    let front_style = ev.resolved.hair_front_style.as_deref().unwrap_or("center_part");
    let side_style = ev.resolved.hair_side_style.as_deref().unwrap_or("chest_locks");
    let back_style = ev.resolved.hair_back_style.as_deref().unwrap_or("long_wave");
    let mouth_style = ev.resolved.mouth_style.as_deref().unwrap_or("soft_smile");
    let clothes_style = ev.resolved.clothes_style.as_deref().unwrap_or("crew_sweater");

    // 1. Collect aggregate evidence unions for fast lookup.
    let mut ev_hair_all = vec![false; n];
    let mut ev_hair_front = vec![false; n];
    let mut ev_hair_back = vec![false; n];
    let mut ev_clothes = vec![false; n];
    let mut ev_neck = vec![false; n];
    let mut ev_body = vec![false; n];
    let mut ev_face = vec![false; n];
    let mut ev_iris_r = vec![false; n];
    let mut ev_iris_l = vec![false; n];
    let mut ev_eye_r_all = vec![false; n];
    let mut ev_eye_l_all = vec![false; n];
    let mut ev_mouth_all = vec![false; n];
    let mut ev_brow_r = vec![false; n];
    let mut ev_brow_l = vec![false; n];
    let mut ev_upper_r = vec![false; n];
    let mut ev_upper_l = vec![false; n];
    let mut ev_lower_r = vec![false; n];
    let mut ev_lower_l = vec![false; n];

    for (idx, id) in evidence_ids.iter().enumerate() {
        let m = &evidence_masks[idx];
        for i in 0..n {
            if m[i] <= 32 {
                continue;
            }
            if id.starts_with("hair_") {
                ev_hair_all[i] = true;
                if id.starts_with("hair_front") {
                    ev_hair_front[i] = true;
                }
                if id.starts_with("hair_back") {
                    ev_hair_back[i] = true;
                }
            } else if id.starts_with("clothes") {
                ev_clothes[i] = true;
            } else if id == "neck" {
                ev_neck[i] = true;
            } else if id == "body" {
                ev_body[i] = true;
            } else if id == "face" {
                ev_face[i] = true;
            } else if id == "eyeR_iris" {
                ev_iris_r[i] = true;
                ev_eye_r_all[i] = true;
            } else if id == "eyeL_iris" {
                ev_iris_l[i] = true;
                ev_eye_l_all[i] = true;
            } else if id.starts_with("eyeR") {
                ev_eye_r_all[i] = true;
                if id == "eyeR_upper" {
                    ev_upper_r[i] = true;
                } else if id == "eyeR_lower" {
                    ev_lower_r[i] = true;
                }
            } else if id.starts_with("eyeL") {
                ev_eye_l_all[i] = true;
                if id == "eyeL_upper" {
                    ev_upper_l[i] = true;
                } else if id == "eyeL_lower" {
                    ev_lower_l[i] = true;
                }
            } else if id.starts_with("mouth") {
                ev_mouth_all[i] = true;
            } else if id == "browR" {
                ev_brow_r[i] = true;
            } else if id == "browL" {
                ev_brow_l[i] = true;
            }
        }
    }

    // 2. Build clean harmonic color fields from inward-eroded seeds so hidden layers have zero smear.
    let mut pure_face_seeds = erode_mask(&ev_face, w, h, 3);
    let skin_base_luma = luma(ev.palette.skin_base);
    for i in 0..n {
        if ev_hair_all[i]
            || ev_eye_r_all[i]
            || ev_eye_l_all[i]
            || ev_mouth_all[i]
            || ev_brow_r[i]
            || ev_brow_l[i]
        {
            pure_face_seeds[i] = false;
            continue;
        }
        let k = i * 4;
        let rgb = [rgba[k], rgba[k + 1], rgba[k + 2]];
        if luma(rgb) < skin_base_luma * 0.70 {
            pure_face_seeds[i] = false;
        }
    }
    let skin_field = build_color_field(rgba, &pure_face_seeds, w, h, ev.palette.skin_base);

    let hair_seed_src: Vec<bool> = (0..n)
        .map(|i| {
            if !ev_hair_all[i] || ev_clothes[i] || ev_face[i] {
                return false;
            }
            let k = i * 4;
            let rgb = [rgba[k], rgba[k + 1], rgba[k + 2]];
            luma(rgb) >= 44.0
        })
        .collect();
    let mut hair_eroded = erode_mask(&hair_seed_src, w, h, 2);
    if hair_eroded.iter().filter(|&&v| v).count() < 16 {
        hair_eroded = erode_mask(&ev_hair_all, w, h, 2);
    }
    let hair_field = build_color_field(rgba, &hair_eroded, w, h, ev.palette.hair_base);

    // 3. Evaluate template SDF masks (`tpl_masks[layer]`, 0..=255) across all 21 layers in Z-order.
    let num_layers = TEMPLATE_LAYER_ORDER.len();
    let mut tpl_masks = vec![vec![0u8; n]; num_layers];
    let mut visible_masks = vec![vec![0u8; n]; num_layers];

    let face_rx = (ev.metrics.face_rx_ratio * eye_dist).clamp(
        eye_dist * 0.96,
        eye_dist * if ev.is_sd { 1.44 } else { 1.24 },
    );
    let side_max_v = match side_style {
        "chest_locks" => frame.chin_v + hh * 0.72,
        "chin_bob" => frame.chin_v + hh * 0.28,
        "short_tuck" => frame.chin_v + hh * 0.08,
        "twin_frame" => frame.chin_v + hh * 0.85,
        _ => frame.chin_v - hh * 0.02,
    };
    let back_max_v = match back_style {
        "long_straight" | "long_wave" => ev.fg_max_v.max(frame.chin_v + hh * 1.15),
        "twin_tails" => ev.fg_max_v.max(frame.chin_v + hh * 1.05),
        "shoulder_bob" => frame.chin_v + hh * 0.55,
        _ => frame.chin_v + hh * 0.25,
    };

    let explicit_front_override = overrides
        .hair_front_style
        .as_deref()
        .is_some_and(|s| s != "auto");
    let explicit_face_override = overrides
        .face_style
        .as_deref()
        .is_some_and(|s| s != "auto");
    let explicit_eye_override = overrides
        .eye_style
        .as_deref()
        .is_some_and(|s| s != "auto");
    let explicit_mouth_override = overrides
        .mouth_style
        .as_deref()
        .is_some_and(|s| s != "auto");
    let skin_base_luma_thresh = luma(ev.palette.skin_base) * 0.65;
    let sclera_luma_thresh = luma(ev.palette.sclera_white) * 0.78;

    let clothes_v_neck_depth = match clothes_style {
        "v_neck_cardigan" => ev.clothes_top_v + hh * 0.18,
        "sailor_collar" => ev.clothes_top_v + hh * 0.14,
        "hoodie_casual" => ev.clothes_top_v - hh * 0.04,
        _ => ev.clothes_top_v,
    };

    // Map eye/mouth centers cleanly for local per-pixel checks.
    let (er_u, er_v) = ev.eye_r_center;
    let (el_u, el_v) = ev.eye_l_center;
    let er_rx = ev.eye_r_rx;
    let er_ry = ev.eye_r_ry;
    let el_rx = ev.eye_l_rx;
    let el_ry = ev.eye_l_ry;

    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
            let fg_a = ev.fg_alpha[i];
            let (fg_left_u, fg_right_u) = sample_fg_bounds(&ev, frame, v);

            // --- Layer 0: hair_back (strictly inside character silhouette: behind head/neck/torso only) ---
            let back_half_w = match back_style {
                "long_wave" => (face_rx * 1.48).max((fg_right_u - fg_left_u) * 0.50),
                "long_straight" => (face_rx * 1.32).max((fg_right_u - fg_left_u) * 0.48),
                "twin_tails" => face_rx * 1.68,
                "shoulder_bob" => face_rx * 1.30,
                _ => face_rx * 1.16,
            };
            let back_cx = ev.torso_cx_u * smoothstep(frame.chin_v, back_max_v, v);
            let back_span_v = (back_max_v - frame.chin_v).max(16.0);
            let back_oval_r = ((u - back_cx) / (back_half_w + 4.0).max(1.0))
                .hypot((v - frame.chin_v).max(0.0) / back_span_v);
            let skull_top_r = (u / (back_half_w + 4.0).max(1.0))
                .hypot(((frame.crown_v + hh * 0.42) - v).max(0.0) / (hh * 0.28).max(1.0));
            let deep_face_interior = face_sdf(u, v, frame, &ev) < -eye_dist * 0.36 && v < frame.chin_v;
            let in_back_hair_template = fg_a > 16
                && !ev_face[i]
                && !deep_face_interior
                && v >= frame.crown_v + hh * 0.14
                && skull_top_r <= 1.04
                && v <= back_max_v
                && (ev_hair_all[i]
                    || ev_neck[i]
                    || ev_body[i]
                    || ev_clothes[i]
                    || v >= frame.chin_v + hh * (if ev.is_sd { 0.06 } else { -0.10 }))
                && (u - back_cx).abs() <= back_half_w
                && (ev_hair_back[i] || back_oval_r <= 1.02);
            if ev_hair_back[i] || in_back_hair_template {
                let d_edge = (back_oval_r - 1.0) * back_half_w.min(back_span_v) * 0.55;
                let base_a = if ev_hair_back[i] {
                    fg_a
                } else {
                    sdf_to_alpha(d_edge).min(fg_a)
                };
                tpl_masks[0][i] = tpl_masks[0][i].max(base_a);
            }

            // --- Layer 1: body (continuous mannequin base behind neck & clothes) ---
            let shoulder_hw = (ev.metrics.shoulder_width_ratio * eye_dist).clamp(eye_dist * 1.18, eye_dist * 2.55);
            let body_top = frame.chin_v + hh * if ev.is_sd { 0.10 } else { 0.18 };
            let body_bot = ev.fg_max_v.max(frame.chin_v + hh * 1.05);
            if v >= body_top && v <= body_bot && (ev_body[i] || ev_clothes[i] || ev_neck[i]) {
                let slope_t = smoothstep(body_top, body_top + hh * 0.35, v);
                let w_at_v = eye_dist * 0.42 + slope_t * (shoulder_hw * 0.88 - eye_dist * 0.42);
                let bd = (u - ev.torso_cx_u).abs() - w_at_v;
                if bd <= 1.5 || ev_body[i] {
                    let ta = if ev_body[i] { fg_a } else { sdf_to_alpha(bd).min(fg_a) };
                    tpl_masks[L_BODY][i] = tpl_masks[L_BODY][i].max(ta);
                }
            }

            // --- Layer 2: neck (behind chin & collar so clothes cleanly sit on top of lower neck) ---
            let neck_rx = eye_dist * if ev.is_sd { 0.18 } else { 0.30 };
            let neck_top = frame.chin_v - hh * 0.10;
            let neck_bot = if ev.is_sd {
                (frame.chin_v + hh * 0.26).min(ev.clothes_top_v + hh * 0.12).max(frame.chin_v + hh * 0.10)
            } else {
                (frame.chin_v + hh * 0.48).max(ev.clothes_top_v + hh * 0.12)
            };
            if v >= neck_top
                && v <= neck_bot
                && (ev_neck[i] || ev_face[i] || ev_clothes[i] || ev_body[i] || ev_hair_all[i])
            {
                let flare = smoothstep(frame.chin_v + hh * 0.10, neck_bot, v)
                    * (eye_dist * if ev.is_sd { 0.06 } else { 0.14 });
                let nd = ((u - ev.torso_cx_u * 0.35).abs() - (neck_rx + flare))
                    .max(neck_top - v)
                    .max(v - neck_bot);
                let accept_ev_neck = ev_neck[i]
                    && !ev.is_sd
                    && !ev_clothes[i]
                    && u.abs() <= neck_rx + eye_dist * 0.14
                    && v <= neck_bot;
                if nd <= 1.2 || accept_ev_neck {
                    let ta = sdf_to_alpha(nd).min(fg_a);
                    tpl_masks[L_NECK][i] = tpl_masks[L_NECK][i].max(ta).max(if accept_ev_neck { fg_a } else { 0 });
                }
            }

            // --- Layer 3: clothes (preserve real neckline below chin + fill behind hair/arms) ---
            let _ = clothes_style;
            let _ = clothes_v_neck_depth;
            if ev_clothes[i] && v >= frame.chin_v - hh * 0.08 {
                tpl_masks[L_CLOTHES][i] = tpl_masks[L_CLOTHES][i].max(fg_a);
            } else if (ev_hair_all[i] || ev_neck[i]) && fg_a > 16 && v >= frame.chin_v && v <= body_bot {
                if let Some(local_top_v) = sample_clothes_top_v(&ev, frame, u) {
                    if v >= local_top_v + 1.0 {
                        let cl_d = local_top_v + 1.0 - v;
                        let ta = sdf_to_alpha(cl_d.max((u - ev.torso_cx_u).abs() - shoulder_hw * 0.96)).min(fg_a);
                        tpl_masks[L_CLOTHES][i] = tpl_masks[L_CLOTHES][i].max(ta);
                    }
                }
            }

            // --- Layer 4: face (solid hole-free skin sheet covering full eye sockets & mouth cavity) ---
            let fd = face_sdf(u, v, frame, &ev);
            let in_socket_interior = u.abs() <= er_u.abs().max(el_u.abs()) + er_rx.max(el_rx) * 1.45
                && v >= er_v.min(el_v) - er_ry.max(el_ry) * 1.9
                && v <= ev.mouth_v + hh * if ev.is_sd { 0.24 } else { 0.14 };
            if ev_face[i] && fd <= 4.0 {
                tpl_masks[L_FACE][i] = tpl_masks[L_FACE][i].max(fg_a);
            } else if fd <= 1.5 && (fg_a > 16 || in_socket_interior) {
                let cap = if fg_a > 16 { fg_a } else { 255 };
                tpl_masks[L_FACE][i] = tpl_masks[L_FACE][i].max(sdf_to_alpha(fd).min(cap));
            }

            // --- Layers 5..12: Eyes (R: 5..8, L: 9..12) ---
            let iris_max_rx_mult = if ev.is_sd { 0.82 } else { 0.56 };
            let px_luma = luma(pix_rgb(rgba, i));
            for (side_idx, (cu, cv, rx, ry, iris_fit, is_left, ev_iris_s, ev_up_s, ev_lo_s, ev_all_s)) in [
                (
                    L_EYE_R_BASE,
                    (er_u, er_v, er_rx, er_ry, &ev.iris_r, false, &ev_iris_r, &ev_upper_r, &ev_lower_r, &ev_eye_r_all),
                ),
                (
                    L_EYE_L_BASE,
                    (el_u, el_v, el_rx, el_ry, &ev.iris_l, true, &ev_iris_l, &ev_upper_l, &ev_lower_l, &ev_eye_l_all),
                ),
            ] {
                // Smooth sclera (`white`) inside upper/lower eyelid corners
                let sclera_rx = if ev.is_sd { rx * 0.99 } else { rx * 0.96 };
                let sclera_ry = if ev.is_sd { ry * 0.95 } else { ry * 0.88 };
                let wd = eye_white_sdf(u, v, (cu, cv, sclera_rx, sclera_ry), is_left, eye_style, ev.is_sd);
                if wd <= 1.2 || (!explicit_eye_override && ev_all_s[i] && !ev_iris_s[i] && !ev_up_s[i] && !ev_lo_s[i]) {
                    let ta = if ev_all_s[i] && !explicit_eye_override {
                        fg_a
                    } else {
                        sdf_to_alpha(wd)
                    };
                    tpl_masks[side_idx][i] = tpl_masks[side_idx][i].max(ta);
                }
                // Full round behind-lid iris (`iris`) tucked beneath upper eyelid crescent
                let clamped_ry = iris_fit.ry.min(ry * if ev.is_sd { 1.08 } else { 1.12 });
                let clamped_rx = iris_fit.rx.min(rx * iris_max_rx_mult);
                let clamped_cv = iris_fit.center_v.clamp(cv - ry * 0.16, cv + ry * 0.14);
                let id_dist = (((u - iris_fit.center_u) / clamped_rx.max(1.0))
                    .hypot((v - clamped_cv) / clamped_ry.max(1.0))
                    - 1.0)
                    * clamped_rx.min(clamped_ry);
                if id_dist <= 1.2 || (!explicit_eye_override && ev_iris_s[i] && !ev_up_s[i]) {
                    let ta = if ev_iris_s[i] && !explicit_eye_override {
                        fg_a
                    } else {
                        sdf_to_alpha(id_dist)
                    };
                    tpl_masks[side_idx + 1][i] = tpl_masks[side_idx + 1][i].max(ta);
                }
                let u_rel = (u - cu) / (rx * 1.06).max(1.0);
                let corner_taper = (1.0 - u_rel * u_rel).max(0.0).powf(0.75);
                // Motion-ready smooth crescent lower lid (`lower`)
                let lower_arch = cv + ry * (1.0 - u_rel * u_rel * 0.42) * if ev.is_sd { 0.86 } else { 0.72 };
                let lower_lash_h = if ev.is_sd { (rx * 0.07).clamp(1.5, 10.0) } else { (ry * 0.08).max(1.5) };
                if u_rel.abs() <= 1.04 {
                    let ld_top = (lower_arch - lower_lash_h * corner_taper) - v;
                    let ld_bot = v - (lower_arch + ry * 0.30 * corner_taper + 1.5);
                    let ld_side = (u_rel.abs() - 1.02) * rx;
                    let ld = ld_top.max(ld_bot).max(ld_side);
                    if ld <= 1.2 {
                        tpl_masks[side_idx + 2][i] = tpl_masks[side_idx + 2][i].max(sdf_to_alpha(ld));
                    }
                }
                if ev_lo_s[i]
                    && u_rel.abs() <= 1.08
                    && (v - lower_arch).abs() <= ry * 0.42
                {
                    tpl_masks[side_idx + 2][i] = tpl_masks[side_idx + 2][i].max(fg_a);
                }
                // Upper lid (`upper`): in default mode keep closing skin apron strictly above open iris/sclera!
                let outer_sign = if is_left { 1.0 } else { -1.0 };
                let wing = match eye_style {
                    "almond_sharp" => -u_rel * outer_sign * ry * 0.16,
                    "droopy_gentle" | "tareme_large" => u_rel * outer_sign * ry * 0.12,
                    _ => 0.0,
                };
                let arch_mult = if ev.is_sd { 0.86 } else { 0.80 };
                let arch_v = cv - ry * (1.0 - u_rel * u_rel * 0.56) * arch_mult + wing;
                let lash_drop = if ev.is_sd { (rx * 0.09).clamp(2.0, 12.0) } else { (ry * 0.12).max(1.8) };
                let px_rgb = pix_rgb(rgba, i);
                let px_chroma = px_rgb[0]
                    .max(px_rgb[1])
                    .max(px_rgb[2])
                    .saturating_sub(px_rgb[0].min(px_rgb[1]).min(px_rgb[2]));
                let is_open_eye_core_color = v >= arch_v - 1.5
                    && ((px_chroma >= 24 && px_luma >= 95.0) || px_luma >= sclera_luma_thresh);
                let is_exposed_eye_core = !explicit_eye_override
                    && ((ev_iris_s[i] && (!ev_up_s[i] || is_open_eye_core_color))
                        || (ev_all_s[i] && (!ev_up_s[i] && px_luma >= sclera_luma_thresh || is_open_eye_core_color)));
                if u_rel.abs() <= 1.08 && !is_exposed_eye_core {
                    let top_cover = (cv - eye_dist * 0.28).min(
                        arch_v - (ry * if ev.is_sd { 0.68 } else { 0.92 } * corner_taper + 2.5),
                    );
                    let bot_lash = if explicit_eye_override {
                        arch_v + lash_drop * (0.45 + 0.55 * corner_taper)
                    } else {
                        arch_v + 1.4 * corner_taper
                    };
                    let ud_bot = v - bot_lash;
                    let ud_top = top_cover - v;
                    let ud_side = (u_rel.abs() - 1.04) * rx;
                    let ud = ud_bot.max(ud_top).max(ud_side);
                    if ud <= 1.2 {
                        tpl_masks[side_idx + 3][i] = tpl_masks[side_idx + 3][i].max(sdf_to_alpha(ud));
                    }
                }
                if ev_up_s[i]
                    && !ev_hair_all[i]
                    && !is_open_eye_core_color
                    && u_rel.abs() <= 1.12
                    && (v - arch_v).abs() <= ry * 0.65
                {
                    tpl_masks[side_idx + 3][i] = tpl_masks[side_idx + 3][i].max(fg_a);
                }
            }

            // --- Layers 13..14: Brows (browR: 13, browL: 14) ---
            for (brow_slot, (cu, cv, rx, ry, ev_brow, sign)) in [
                (13usize, (er_u, er_v, er_rx, er_ry, &ev_brow_r, -1.0)),
                (14usize, (el_u, el_v, el_rx, el_ry, &ev_brow_l, 1.0)),
            ] {
                let u_rel = (u - cu) / (rx * 1.12).max(1.0);
                let brow_lift = if ev.is_sd { ry * 1.32 } else { ry * 1.85 };
                let brow_arch_v = cv - brow_lift + u_rel * sign * ry * 0.12 + u_rel * u_rel * ry * 0.24;
                let half_thick = if ev.is_sd {
                    (rx * 0.08).clamp(2.0, 9.0)
                } else {
                    ry * 0.22 * (1.0 - u_rel.abs() * 0.45)
                };
                if u_rel.abs() <= 1.0 {
                    let bd = (v - brow_arch_v).abs() - half_thick;
                    if bd <= 1.0 {
                        tpl_masks[brow_slot][i] = tpl_masks[brow_slot][i].max(sdf_to_alpha(bd));
                    }
                }
                if ev_brow[i]
                    && !ev_hair_all[i]
                    && u_rel.abs() <= 1.12
                    && (v - brow_arch_v).abs() <= (ry * 0.45).max(3.5)
                {
                    tpl_masks[brow_slot][i] = tpl_masks[brow_slot][i].max(fg_a);
                }
            }

            // --- Layers 15..17: Mouth (inner: 15, lower: 16, upper: 17) ---
            let (m_rx, m_inner_ry) = mouth_radii(&ev, eye_dist, mouth_style);
            let mdx = (u - ev.mouth_u) / m_rx.max(1.0);
            let mdy = (v - ev.mouth_v) / m_inner_ry.max(1.0);
            let m_dist = ((mdx * mdx + mdy * mdy).sqrt() - 1.0) * m_rx.min(m_inner_ry);
            if m_dist <= 1.2 {
                tpl_masks[15][i] = tpl_masks[15][i].max(sdf_to_alpha(m_dist));
            }
            if mdx.abs() <= 1.18 {
                let lip_curve = match mouth_style {
                    "soft_smile" => -mdx * mdx * eye_dist * 0.015,
                    "cat_w" => (mdx.abs() * std::f64::consts::PI).cos() * eye_dist * 0.012,
                    "open_happy" => mdx * mdx * eye_dist * 0.010,
                    _ => 0.0,
                };
                let lip_y = ev.mouth_v + lip_curve;
                let cushion_h = if mouth_style == "open_o" { m_inner_ry * 1.15 } else { eye_dist * 0.056 };
                let upper_overlap = if mouth_style == "open_o" { m_inner_ry * 0.25 } else { eye_dist * 0.008 };
                if mouth_style == "open_o" {
                    // For round `'o'` mouth, keep center visible inside `mouth_inner` unless outside oval arc!
                    if v >= lip_y && v <= lip_y + cushion_h && m_dist >= -1.0 {
                        let a_l = sdf_to_alpha((v - (lip_y + cushion_h)).max(lip_y - v));
                        tpl_masks[16][i] = tpl_masks[16][i].max(a_l);
                    }
                    if v >= lip_y - cushion_h && v <= lip_y + upper_overlap && m_dist >= -1.0 {
                        let a_u = sdf_to_alpha((v - (lip_y + upper_overlap)).max(lip_y - cushion_h - v));
                        tpl_masks[17][i] = tpl_masks[17][i].max(a_u);
                    }
                } else {
                    if v >= lip_y - eye_dist * 0.006 && v <= lip_y + cushion_h {
                        let a_l = sdf_to_alpha((v - (lip_y + cushion_h)).max(lip_y - eye_dist * 0.006 - v));
                        tpl_masks[16][i] = tpl_masks[16][i].max(a_l);
                    }
                    if v >= lip_y - cushion_h && v <= lip_y + upper_overlap {
                        let a_u = sdf_to_alpha((v - (lip_y + upper_overlap)).max(lip_y - cushion_h - v));
                        tpl_masks[17][i] = tpl_masks[17][i].max(a_u);
                    }
                }
            }

            // --- Layers 18..20: Foreground Hair & Head Accessories (hair_side_R: 18, hair_side_L: 19, hair_front: 20) ---
            let bang_v = bang_tip_v(front_style, u / face_rx.max(1.0), frame);
            let outer_hw = fg_right_u.max(-fg_left_u).max(face_rx * 1.15);
            let hood_cap_w = outer_hw.min(face_rx * 1.75);
            let temple_t = smoothstep(er_v.min(el_v) - hh * 0.18, er_v.min(el_v) + hh * 0.14, v);
            let front_half_w = hood_cap_w * (1.0 - temple_t) + (face_rx * 1.18) * temple_t;
            let temple_join_v = er_v.min(el_v)
                + hh * (0.20 * (1.0 - smoothstep(face_rx * 0.45, face_rx * 1.25, u.abs())));
            let is_head_accessory = ev_clothes[i] && v <= er_v.min(el_v) && !ev_face[i];
            let has_front_ev = ev_hair_all[i] || is_head_accessory;
            let in_scalp_dome = v <= er_v.min(el_v) - er_ry * 0.20 && u.abs() <= hood_cap_w;
            let in_front_zone = u.abs() <= front_half_w && (v <= bang_v + hh * 0.08 || in_scalp_dome);
            let is_template_front = if explicit_front_override {
                v >= frame.crown_v - hh * 0.04 && u.abs() <= face_rx * 0.96 && (v <= bang_v || (v <= er_v.min(el_v) - er_ry * 0.35 && has_front_ev))
            } else {
                has_front_ev && in_front_zone
            };
            let near_upper_lash = (ev_upper_r[i] || ev_upper_l[i]) && !ev_hair_front[i];
            if !near_upper_lash
                && (is_template_front
                    || ((ev_hair_front[i] || is_head_accessory)
                        && u.abs() <= front_half_w
                        && v <= temple_join_v.max(bang_v + hh * 0.08)))
            {
                let d_edge = if in_front_zone { v - bang_v - 4.0 } else { 0.0 };
                let a_f = if has_front_ev { fg_a } else { sdf_to_alpha(d_edge).min(fg_a) };
                tpl_masks[L_HAIR_FRONT][i] = tpl_masks[L_HAIR_FRONT][i].max(a_f);
            }

            // Organic temple-anchored side hair locks strictly from actual foreground hair,
            // with hidden `fillMask` extending upward behind `hair_front` cheek bangs so head/hair sway stays seamless.
            let side_overlap_top = er_v.min(el_v) - hh * 0.18;
            if ev_hair_all[i]
                && !ev_clothes[i]
                && !ev_face[i]
                && v >= side_overlap_top
                && v <= side_max_v
                && side_style != "minimal"
            {
                let overlap_t = smoothstep(side_overlap_top, er_v.min(el_v) + hh * 0.14, v);
                let inner_u = face_rx
                    * (0.42 + 0.06 * overlap_t + 0.14 * (1.0 - smoothstep(er_v.min(el_v), frame.chin_v, v)));
                for (side_slot, sign) in [(L_HAIR_SIDE_R, -1.0f64), (L_HAIR_SIDE_L, 1.0f64)] {
                    if u * sign <= 0.0 {
                        continue;
                    }
                    let au = u.abs();
                    if au >= inner_u && au <= face_rx * 1.55 {
                        tpl_masks[side_slot][i] = tpl_masks[side_slot][i].max(fg_a);
                    }
                }
            }
        }
    }

    // 4. Partition foreground pixels (`fg_alpha > 0`) strictly cleanly top-to-bottom (`owners <= 1`)!
    // Every visible pixel goes to a non-overlapping visible mask, while hidden/motion geometry goes to `fill_masks`.
    let mut owner_assigned = vec![false; n];
    for layer_idx in (0..num_layers).rev() {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let fg_a = ev.fg_alpha[i];
                if fg_a == 0 || owner_assigned[i] {
                    continue;
                }
                if layer_idx == L_NECK && ev.is_sd {
                    let (u, _) = frame.to_local(DVec2::new(x as f64, y as f64));
                    if u.abs() > eye_dist * 0.36 {
                        continue;
                    }
                }
                if !explicit_face_override && layer_idx == L_FACE {
                    let (_, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                    let non_skin_jaw_hair = (ev_hair_back[i] || ev_hair_all[i])
                        && !ev_face[i]
                        && v >= er_v.max(el_v)
                        && rgb_dist(pix_rgb(rgba, i), ev.palette.skin_base) > 36.0;
                    if ev_clothes[i] || (ev_neck[i] && !ev_face[i]) || non_skin_jaw_hair {
                        continue;
                    }
                }
                // Respect open-eye iris/sclera evidence in default mode so closing lids never squash rest eyes.
                if !explicit_eye_override {
                    let px_rgb = pix_rgb(rgba, i);
                    let px_luma = luma(px_rgb);
                    let px_chroma = px_rgb[0]
                        .max(px_rgb[1])
                        .max(px_rgb[2])
                        .saturating_sub(px_rgb[0].min(px_rgb[1]).min(px_rgb[2]));
                    let is_open_color = (px_chroma >= 24 && px_luma >= 95.0) || px_luma >= sclera_luma_thresh;
                    if layer_idx == L_EYE_R_BASE + 3 || layer_idx == L_EYE_R_BASE + 2 {
                        let (_, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                        let in_socket_opening = v >= er_v - er_ry * 0.55;
                        let is_open_r = (ev_iris_r[i] || (ev_eye_r_all[i] && is_open_color))
                            && (!ev_upper_r[i] || (in_socket_opening && is_open_color))
                            && !(ev_lower_r[i] && !ev_iris_r[i]);
                        if is_open_r {
                            continue;
                        }
                    } else if layer_idx == L_EYE_L_BASE + 3 || layer_idx == L_EYE_L_BASE + 2 {
                        let (_, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                        let in_socket_opening = v >= el_v - el_ry * 0.55;
                        let is_open_l = (ev_iris_l[i] || (ev_eye_l_all[i] && is_open_color))
                            && (!ev_upper_l[i] || (in_socket_opening && is_open_color))
                            && !(ev_lower_l[i] && !ev_iris_l[i]);
                        if is_open_l {
                            continue;
                        }
                    }
                }
                if !explicit_mouth_override
                    && (layer_idx == L_MOUTH_LOWER || layer_idx == L_MOUTH_UPPER)
                    && ev_mouth_all[i]
                    && (mouth_style == "open_happy" || mouth_style == "open_o")
                    && luma(pix_rgb(rgba, i)) < skin_base_luma_thresh
                {
                    continue;
                }
                if tpl_masks[layer_idx][i] >= 128 {
                    visible_masks[layer_idx][i] = fg_a;
                    owner_assigned[i] = true;
                }
            }
        }
    }

    // Keep closed mouth cavity (`mouth_inner`) hidden strictly inside upper/lower lip cushion bounds at rest.
    if !explicit_mouth_override {
        for i in 0..n {
            if visible_masks[L_MOUTH_INNER][i] == 0 {
                let lip_cap = tpl_masks[L_MOUTH_LOWER][i].max(tpl_masks[L_MOUTH_UPPER][i]);
                tpl_masks[L_MOUTH_INNER][i] = tpl_masks[L_MOUTH_INNER][i].min(lip_cap);
            }
        }
    }

    // Route any remaining unowned foreground pixels cleanly without ever leaking face/cheek pixels into back hair.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let fg_a = ev.fg_alpha[i];
            if fg_a == 0 || owner_assigned[i] {
                continue;
            }
            let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
            let target = if ev_clothes[i] && v >= frame.chin_v - hh * 0.08 {
                L_CLOTHES
            } else if ev_body[i] {
                L_BODY
            } else if ev_neck[i] && !ev_clothes[i] && v <= frame.chin_v + hh * 0.48 && (!ev.is_sd || u.abs() <= eye_dist * 0.36) {
                L_NECK
            } else if (ev_hair_front[i] || (ev_hair_all[i] && v <= er_v.max(el_v) + hh * 0.14 && !ev_hair_back[i]))
                && !(ev_upper_r[i] || ev_upper_l[i])
            {
                L_HAIR_FRONT
            } else if (ev_face[i] && face_sdf(u, v, frame, &ev) <= 4.0)
                || (face_sdf(u, v, frame, &ev) <= 3.0 && !ev_hair_all[i] && !ev_clothes[i] && !ev_neck[i])
            {
                L_FACE
            } else if ev_hair_all[i] && !ev_hair_back[i] && v <= side_max_v && u.abs() <= face_rx * 1.55 {
                if u < 0.0 { L_HAIR_SIDE_R } else { L_HAIR_SIDE_L }
            } else {
                L_HAIR_BACK
            };
            visible_masks[target][i] = fg_a;
            owner_assigned[i] = true;
        }
    }

    // 4b. Extract dynamic wearable layers (glasses, headwear/hoods, hairpins/ribbons, multi-part clothes/hands).
    let raw_hair_modes = collect_cluster_modes(rgba, n, 3, 34.0, |i| {
        if ev.fg_alpha[i] <= 32
            || !ev_hair_all[i]
            || ev_eye_r_all[i]
            || ev_eye_l_all[i]
            || ev_brow_r[i]
            || ev_brow_l[i]
            || ev_mouth_all[i]
        {
            return false;
        }
        let c = pix_rgb(rgba, i);
        if luma(c) < 36.0
            || rgb_dist(c, ev.palette.skin_base) < 42.0
            || (rgb_dist(ev.palette.skin_shadow, ev.palette.skin_base) <= 72.0
                && luma(c) >= luma(ev.palette.skin_base) * 0.65
                && rgb_dist(c, ev.palette.skin_shadow) < 38.0)
        {
            return false;
        }
        let (u, v) = frame.to_local(DVec2::new((i % w) as f64, (i / w) as f64));
        u.abs() <= face_rx * 1.05
            && v >= frame.crown_v + hh * 0.24
            && v <= er_v.max(el_v) + hh * 0.24
    });
    let base_hair_rgb = raw_hair_modes
        .first()
        .copied()
        .unwrap_or(ev.palette.hair_base);
    let mut hair_palette = vec![base_hair_rgb];
    for &hc in raw_hair_modes.iter().skip(1) {
        if rgb_dist(hc, base_hair_rgb) <= 68.0
            || (chroma_normalized_dist(hc, base_hair_rgb) <= 16.0
                && (luma(hc) - luma(base_hair_rgb)).abs() <= 55.0)
        {
            hair_palette.push(hc);
        }
    }
    let wearable_ctx = WearableExtractionCtx {
        rgba,
        w,
        h,
        frame,
        ev: &ev,
        ev_face: &ev_face,
        ev_hair_all: &ev_hair_all,
        ev_eye_r_all: &ev_eye_r_all,
        ev_eye_l_all: &ev_eye_l_all,
        ev_mouth_all: &ev_mouth_all,
        ev_brow_r: &ev_brow_r,
        ev_brow_l: &ev_brow_l,
        ev_neck: &ev_neck,
        face_rx,
    };

    let glasses_layer = extract_glasses_mask(&wearable_ctx, hair_palette[0]);
    if let Some(ref g_mask) = glasses_layer {
        for (i, &ga) in g_mask.iter().enumerate() {
            if ga > 0 {
                for base_mask in &mut visible_masks {
                    base_mask[i] = 0;
                }
            }
        }
    }

    let (mut hat_layer, hairpin_layer, mut ribbon_layer) =
        extract_hat_and_head_acc(&wearable_ctx, &hair_palette, glasses_layer.as_deref());
    let mut hat_fill_extra = vec![0u8; n];
    if let Some(ref mut hat_m) = hat_layer {
        // Pull any thin outer hat-rim fringe in upper head into `hat` so `hair_front` never keeps a silhouette halo.
        let mut fringe = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if hat_m[i] > 0 || ev.fg_alpha[i] == 0 {
                    continue;
                }
                if glasses_layer.as_ref().is_some_and(|g| g[i] > 0)
                    || hairpin_layer.as_ref().is_some_and(|hp| hp[i] > 0)
                    || ribbon_layer.as_ref().is_some_and(|r| r[i] > 0)
                {
                    continue;
                }
                let touches_hat = (x > 0 && hat_m[i - 1] > 0)
                    || (x + 1 < w && hat_m[i + 1] > 0)
                    || (y > 0 && hat_m[i - w] > 0)
                    || (y + 1 < h && hat_m[i + w] > 0);
                if touches_hat {
                    let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                    if (u.abs() > face_rx * 0.96 || v < frame.crown_v + hh * 0.12) && !ev_face[i] {
                        fringe.push(i);
                    }
                }
            }
        }
        for i in fringe {
            hat_m[i] = ev.fg_alpha[i];
        }
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if hat_m[i] > 0 {
                    for base_mask in &mut visible_masks {
                        base_mask[i] = 0;
                    }
                    let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                    if u.abs() > face_rx * 1.05 || v < frame.crown_v + hh * 0.04 {
                        tpl_masks[L_HAIR_FRONT][i] = 0;
                        tpl_masks[L_HAIR_BACK][i] = 0;
                    }
                }
            }
        }
        if let Some(ref rib_m) = ribbon_layer {
            for (i, &ra) in rib_m.iter().enumerate() {
                if ra > 0 {
                    hat_m[i] = 0;
                    let x = i % w;
                    let y = i / w;
                    let (_, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                    if v <= frame.crown_v + hh * 0.42 {
                        hat_fill_extra[i] = ev.fg_alpha[i];
                    }
                }
            }
        }
    }
    if let Some(ref hp_m) = hairpin_layer {
        for (i, &ha) in hp_m.iter().enumerate() {
            if ha > 0 {
                if let Some(ref mut hat_m) = hat_layer {
                    hat_m[i] = 0;
                }
                if let Some(ref mut rib_m) = ribbon_layer {
                    rib_m[i] = 0;
                }
            }
        }
    }
    for acc_m in [&hairpin_layer, &ribbon_layer].into_iter().flatten() {
        for (i, &aa) in acc_m.iter().enumerate() {
            if aa > 0 {
                for base_mask in &mut visible_masks {
                    base_mask[i] = 0;
                }
            }
        }
    }

    let [mut m_hair_back, m_body, mut m_neck, mut m_clothes, mut m_face, m_er_w, m_er_i, m_er_l, m_er_u, m_el_w, m_el_i, m_el_l, m_el_u, m_br, m_bl, m_mi, m_ml, m_mu, mut m_hsr, mut m_hsl, mut m_hf] = {
        let mut it = visible_masks.into_iter();
        [
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        ]
    };
    let outfit_subs = extract_outfit_sublayers(&wearable_ctx, &mut m_clothes, &mut m_face, &mut m_neck);
    // Build clothes color field after separating `clothes_inner`/collar/acc so occluded fill extends underlying fabric rather than ribbon colors.
    let clothes_seed_src: Vec<bool> = (0..n)
        .map(|i| m_clothes[i] > 128 && !is_dark_lineart(pix_rgb(rgba, i)))
        .collect();
    let mut clothes_eroded = erode_mask(&clothes_seed_src, w, h, 2);
    if clothes_eroded.iter().filter(|&&v| v).count() < 24 {
        clothes_eroded = erode_mask(&ev_clothes, w, h, 2);
    }
    let clothes_field = build_color_field(rgba, &clothes_eroded, w, h, ev.palette.clothes_primary);

    // Preserve any extra split hair strands (`hair_front_*`, `hair_side_R_*`, `hair_side_L_*`, `hair_back_*`) from upstream `hair_split`.
    let mut split_hair_extras: HashMap<usize, Vec<(String, Vec<u8>)>> = HashMap::new();
    for (ev_idx, ev_id) in evidence_ids.iter().enumerate() {
        let parent_slot = if ev_id.starts_with("hair_front_") && ev_id != "hair_front" {
            Some(L_HAIR_FRONT)
        } else if ev_id.starts_with("hair_side_R_") {
            Some(L_HAIR_SIDE_R)
        } else if ev_id.starts_with("hair_side_L_") {
            Some(L_HAIR_SIDE_L)
        } else if ev_id.starts_with("hair_back_") {
            Some(L_HAIR_BACK)
        } else {
            None
        };
        if let Some(slot) = parent_slot {
            let parent_vis = match slot {
                L_HAIR_FRONT => &mut m_hf,
                L_HAIR_SIDE_R => &mut m_hsr,
                L_HAIR_SIDE_L => &mut m_hsl,
                _ => &mut m_hair_back,
            };
            let ev_m = &evidence_masks[ev_idx];
            let mut sub_vis = vec![0u8; n];
            let mut any = false;
            for i in 0..n {
                if parent_vis[i] > 0 && ev_m[i] >= 128 {
                    sub_vis[i] = parent_vis[i];
                    parent_vis[i] = 0;
                    any = true;
                }
            }
            if any {
                split_hair_extras
                    .entry(slot)
                    .or_default()
                    .push((ev_id.clone(), sub_vis));
            }
        }
    }

    visible_masks = vec![
        m_hair_back, m_body, m_neck, m_clothes, m_face, m_er_w, m_er_i, m_er_l, m_er_u, m_el_w,
        m_el_i, m_el_l, m_el_u, m_br, m_bl, m_mi, m_ml, m_mu, m_hsr, m_hsl, m_hf,
    ];
    for (sub_id, sub_mask) in &outfit_subs {
        if sub_id != "clothes_hands" {
            for (i, &sa) in sub_mask.iter().enumerate() {
                if sa > 0 {
                    tpl_masks[L_CLOTHES][i] = tpl_masks[L_CLOTHES][i].max(ev.fg_alpha[i]);
                }
            }
        }
    }

    // Assemble final ordered layer list `(id, visible_mask, tpl_mask)` keeping base Z-order + spliced sublayers.
    let mut final_layer_ids: Vec<String> = Vec::with_capacity(num_layers + 12);
    let mut final_visible_masks: Vec<Vec<u8>> = Vec::with_capacity(num_layers + 12);
    let mut final_tpl_masks: Vec<Vec<u8>> = Vec::with_capacity(num_layers + 12);

    for (base_idx, id) in TEMPLATE_LAYER_ORDER.iter().enumerate() {
        final_layer_ids.push((*id).to_string());
        final_visible_masks.push(std::mem::take(&mut visible_masks[base_idx]));
        final_tpl_masks.push(std::mem::take(&mut tpl_masks[base_idx]));

        if let Some(extras) = split_hair_extras.remove(&base_idx) {
            for (sub_id, sub_vis) in extras {
                final_layer_ids.push(sub_id);
                final_tpl_masks.push(sub_vis.clone());
                final_visible_masks.push(sub_vis);
            }
        }

        if base_idx == L_CLOTHES {
            for (sub_id, sub_mask) in &outfit_subs {
                if sub_id != "clothes_hands" && sub_mask.iter().any(|&v| v > 0) {
                    final_layer_ids.push(sub_id.clone());
                    final_visible_masks.push(sub_mask.clone());
                    final_tpl_masks.push(sub_mask.clone());
                }
            }
        } else if base_idx == L_FACE {
            for (sub_id, sub_mask) in &outfit_subs {
                if sub_id == "clothes_hands" && sub_mask.iter().any(|&v| v > 0) {
                    final_layer_ids.push(sub_id.clone());
                    final_visible_masks.push(sub_mask.clone());
                    final_tpl_masks.push(sub_mask.clone());
                }
            }
        } else if base_idx == L_MOUTH_UPPER {
            if let Some(ref g_mask) = glasses_layer {
                if g_mask.iter().any(|&v| v > 0) {
                    final_layer_ids.push("acc_glasses".to_string());
                    final_visible_masks.push(g_mask.clone());
                    final_tpl_masks.push(g_mask.clone());
                }
            }
        } else if base_idx == L_HAIR_FRONT {
            if let Some(ref hat_m) = hat_layer {
                if hat_m.iter().any(|&v| v > 0) {
                    let mut hat_tpl = hat_m.clone();
                    for (i, &ex) in hat_fill_extra.iter().enumerate() {
                        hat_tpl[i] = hat_tpl[i].max(ex);
                    }
                    final_layer_ids.push("hat".to_string());
                    final_visible_masks.push(hat_m.clone());
                    final_tpl_masks.push(hat_tpl);
                }
            }
            if let Some(ref pin_m) = hairpin_layer {
                if pin_m.iter().any(|&v| v > 0) {
                    final_layer_ids.push("acc_hairpin".to_string());
                    final_visible_masks.push(pin_m.clone());
                    final_tpl_masks.push(pin_m.clone());
                }
            }
            if let Some(ref rib_m) = ribbon_layer {
                if rib_m.iter().any(|&v| v > 0) {
                    final_layer_ids.push("acc_ribbon".to_string());
                    final_visible_masks.push(rib_m.clone());
                    final_tpl_masks.push(rib_m.clone());
                }
            }
        }
    }

    let final_len = final_layer_ids.len();

    // Build strictly disjoint `fill_masks`: where template geometry exists outside visible `masks`.
    let mut fill_masks = vec![vec![0u8; n]; final_len];
    let mut fill_kinds = Vec::with_capacity(final_len);
    for layer_idx in 0..final_len {
        let mut any_fill = false;
        for i in 0..n {
            if final_visible_masks[layer_idx][i] == 0 && final_tpl_masks[layer_idx][i] > 0 {
                fill_masks[layer_idx][i] = final_tpl_masks[layer_idx][i];
                any_fill = true;
            }
        }
        fill_kinds.push(if any_fill {
            "template".to_string()
        } else {
            "none".to_string()
        });
    }

    // 5. Synthesize clean hybrid per-part RGBA PNG textures across all canonical + dynamic layers!
    let mut textures = Vec::with_capacity(final_len);
    for layer_idx in 0..final_len {
        let id = final_layer_ids[layer_idx].as_str();
        let eff_nonzero = final_visible_masks[layer_idx]
            .iter()
            .zip(&fill_masks[layer_idx])
            .any(|(&v, &f)| v > 0 || f > 0);
        if !eff_nonzero {
            textures.push(None);
            continue;
        }

        let mut tex = vec![0u8; n * 4];
        let m_vis = &final_visible_masks[layer_idx];
        let m_fill = &fill_masks[layer_idx];

        match id {
            "face" => {
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let ea = m_vis[i].max(m_fill[i]);
                        if ea == 0 {
                            continue;
                        }
                        let clean = skin_field.sample(x, y);
                        let is_hole_feature = ev_eye_r_all[i]
                            || ev_eye_l_all[i]
                            || ev_mouth_all[i]
                            || ev_brow_r[i]
                            || ev_brow_l[i];
                        let rgb = if m_vis[i] > 0
                            && (!explicit_face_override || (!is_hole_feature && ev_face[i]))
                        {
                            let k = i * 4;
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        } else {
                            let (_, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                            if v < er_v.min(el_v) - er_ry {
                                lerp_rgb(clean, ev.palette.skin_shadow, 0.25)
                            } else {
                                clean
                            }
                        };
                        let k = i * 4;
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "neck" => {
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let ea = m_vis[i].max(m_fill[i]);
                        if ea == 0 {
                            continue;
                        }
                        let rgb = if m_vis[i] > 0 {
                            let k = i * 4;
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        } else {
                            let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                            let under_chin = smoothstep(frame.chin_v - hh * 0.04, frame.chin_v + hh * 0.14, v)
                                * (1.0 - smoothstep(frame.chin_v + hh * 0.14, frame.chin_v + hh * 0.28, v));
                            let base = skin_field.sample(x, y);
                            lerp_rgb(base, ev.palette.skin_shadow, 0.48 * under_chin * (1.0 - (u.abs() / eye_dist).min(0.6)))
                        };
                        let k = i * 4;
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "body" => {
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let ea = m_vis[i].max(m_fill[i]);
                        if ea == 0 {
                            continue;
                        }
                        let rgb = if m_vis[i] > 0 {
                            let k = i * 4;
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        } else {
                            skin_field.sample(x, y)
                        };
                        let k = i * 4;
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "clothes" => {
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let ea = m_vis[i].max(m_fill[i]);
                        if ea == 0 {
                            continue;
                        }
                        let rgb = if m_vis[i] > 0 {
                            let k = i * 4;
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        } else {
                            clothes_field.sample(x, y)
                        };
                        let k = i * 4;
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            hair_id if hair_id.starts_with("hair_") => {
                let behind_neck_tint = hair_id.starts_with("hair_back");
                let is_front_overridden = explicit_front_override && hair_id.starts_with("hair_front");
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let ea = m_vis[i].max(m_fill[i]);
                        if ea == 0 {
                            continue;
                        }
                        let rgb = if m_vis[i] > 0 && !is_front_overridden {
                            let k = i * 4;
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        } else {
                            let base = hair_field.sample(x, y);
                            if behind_neck_tint && m_fill[i] > 0 {
                                let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                                let strand = ((u / (eye_dist * 0.06)).sin() * 0.5 + 0.5) * 0.08;
                                let neck_shade = if v > frame.chin_v && u.abs() < face_rx * 1.1 {
                                    0.32
                                } else {
                                    0.12
                                };
                                lerp_rgb(base, ev.palette.hair_shadow, (neck_shade + strand).min(0.72))
                            } else {
                                base
                            }
                        };
                        let k = i * 4;
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "eyeR_white" | "eyeL_white" => {
                let (_, cv, _, ry) = if id == "eyeR_white" {
                    (er_u, er_v, er_rx, er_ry)
                } else {
                    (el_u, el_v, el_rx, el_ry)
                };
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        if m_vis[i].max(m_fill[i]) == 0 {
                            continue;
                        }
                        let k = i * 4;
                        let rgb = if m_vis[i] > 0 && !explicit_eye_override {
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        } else {
                            let (_, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                            let top_t = smoothstep(cv + ry * 0.05, cv - ry * 0.85, v);
                            lerp_rgb(ev.palette.sclera_white, ev.palette.sclera_shadow, top_t * 0.72)
                        };
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "eyeR_iris" | "eyeL_iris" => {
                let (cv, ry, iris_fit, ev_iris, ev_up, ev_lo, ev_eye_all) = if id == "eyeR_iris" {
                    (er_v, er_ry, &ev.iris_r, &ev_iris_r, &ev_upper_r, &ev_lower_r, &ev_eye_r_all)
                } else {
                    (el_v, el_ry, &ev.iris_l, &ev_iris_l, &ev_upper_l, &ev_lower_l, &ev_eye_l_all)
                };
                let max_rx_cap = er_rx.max(el_rx) * if ev.is_sd { 0.82 } else { 0.58 };
                let clamped_ry = iris_fit.ry.min(ry * if ev.is_sd { 1.08 } else { 1.12 });
                let clamped_rx = iris_fit.rx.min(max_rx_cap);
                let clamped_cv = iris_fit.center_v.clamp(cv - ry * 0.16, cv + ry * 0.14);
                let white_luma = luma(ev.palette.sclera_white);
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        if m_vis[i].max(m_fill[i]) == 0 {
                            continue;
                        }
                        let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                        let k = i * 4;
                        let orig = [rgba[k], rgba[k + 1], rgba[k + 2]];
                        let du = u - iris_fit.center_u;
                        let dv = v - clamped_cv;
                        let rn = (du / clamped_rx.max(1.0)).hypot(dv / clamped_ry.max(1.0));
                        // Keep crisp original iris details & sparkles inside eye evidence (including tall SD eyes sliced into upper/lower).
                        let is_orig_iris_px = ev_iris[i] && (!ev_up[i] || ev.is_sd)
                            || (ev.is_sd
                                && (ev_eye_all[i] || ev_up[i] || ev_lo[i])
                                && !ev_hair_all[i]
                                && rn <= 0.94
                                && (luma(orig) < white_luma * 0.90 || rn < 0.78));
                        let rgb = if m_vis[i] > 0 && (!explicit_eye_override || is_orig_iris_px) {
                            orig
                        } else {
                            // Mirror clean symmetric visible iris pixels upward so catchlights/colors carry smoothly behind upper lid.
                            let mirror_u = iris_fit.center_u - du;
                            let mirror_v = if dv < -clamped_ry * 0.20 {
                                clamped_cv + dv.abs() * 0.55
                            } else {
                                v
                            };
                            let mp = frame.origin + frame.ex * mirror_u + frame.ey * mirror_v;
                            let mx = (mp.x.round() as isize).clamp(0, w as isize - 1) as usize;
                            let my = (mp.y.round() as isize).clamp(0, h as isize - 1) as usize;
                            let mi = my * w + mx;
                            let base_sample = if ev_iris[mi] && !ev_up[mi] {
                                let mk = mi * 4;
                                [rgba[mk], rgba[mk + 1], rgba[mk + 2]]
                            } else if ev_iris[i] {
                                orig
                            } else if rn < 0.28 {
                                iris_fit.pupil
                            } else if rn > 0.80 {
                                iris_fit.limbal
                            } else {
                                iris_fit.mid
                            };
                            let top_lid_cast = smoothstep(0.0, -clamped_ry * 0.88, dv);
                            let limbal_blend = smoothstep(0.72, 0.96, rn);
                            let c1 = lerp_rgb(base_sample, iris_fit.limbal, limbal_blend * 0.85);
                            lerp_rgb(c1, scale_rgb(c1, 0.65), top_lid_cast * 0.55)
                        };
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "eyeR_upper" | "eyeL_upper" => {
                let (cu, cv, rx, ry, ev_up, ev_eye_all, is_left) = if id == "eyeR_upper" {
                    (er_u, er_v, er_rx, er_ry, &ev_upper_r, &ev_eye_r_all, false)
                } else {
                    (el_u, el_v, el_rx, el_ry, &ev_upper_l, &ev_eye_l_all, true)
                };
                let outer_sign = if is_left { 1.0 } else { -1.0 };
                let dark_lash = [
                    ev.palette.lash_color[0].min(48),
                    ev.palette.lash_color[1].min(42),
                    ev.palette.lash_color[2].min(58),
                ];
                let skin_luma = luma(ev.palette.skin_base);
                let arch_mult = if ev.is_sd { 0.86 } else { 0.80 };
                let lash_half = if ev.is_sd {
                    (rx * 0.095).clamp(2.4, 12.0)
                } else {
                    (ry * 0.14).max(2.8)
                };
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        if m_vis[i].max(m_fill[i]) == 0 {
                            continue;
                        }
                        let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                        let u_rel = (u - cu) / (rx * 1.06).max(1.0);
                        let wing = match eye_style {
                            "almond_sharp" => -u_rel * outer_sign * ry * 0.16,
                            "droopy_gentle" | "tareme_large" => u_rel * outer_sign * ry * 0.12,
                            _ => 0.0,
                        };
                        let arch_v = cv - ry * (1.0 - u_rel * u_rel * 0.56) * arch_mult + wing;
                        let k = i * 4;
                        let orig = [rgba[k], rgba[k + 1], rgba[k + 2]];
                        let rgb = if m_vis[i] > 0 && !explicit_eye_override {
                            orig
                        } else if (v - arch_v).abs() <= lash_half || v >= arch_v {
                            let core_t = if v >= arch_v {
                                0.88
                            } else {
                                1.0 - ((arch_v - v) / lash_half).clamp(0.0, 1.0)
                            };
                            if !ev_hair_all[i] && ev_up[i] && luma(orig) < skin_luma * 0.65 {
                                lerp_rgb(orig, dark_lash, 0.55 * core_t)
                            } else {
                                let bg = if !ev_hair_all[i] && ev.fg_alpha[i] > 128 && !ev_eye_all[i] {
                                    orig
                                } else {
                                    skin_field.sample(x, y)
                                };
                                lerp_rgb(bg, dark_lash, (0.45 + 0.55 * core_t).clamp(0.0, 1.0))
                            }
                        } else if m_vis[i] > 0 && !ev_hair_all[i] && (ev_up[i] || (!ev_eye_all[i] && ev_face[i])) {
                            orig
                        } else {
                            let crease_d = ((arch_v - ry * 0.30) - v).abs();
                            let base_skin = if !ev_hair_all[i] && !ev_eye_all[i] && ev_face[i] {
                                orig
                            } else {
                                skin_field.sample(x, y)
                            };
                            if crease_d < 1.2 {
                                lerp_rgb(base_skin, ev.palette.skin_shadow, 0.45)
                            } else {
                                base_skin
                            }
                        };
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "eyeR_lower" | "eyeL_lower" => {
                let (cu, cv, rx, ry) = if id == "eyeR_lower" {
                    (er_u, er_v, er_rx, er_ry)
                } else {
                    (el_u, el_v, el_rx, el_ry)
                };
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        if m_vis[i].max(m_fill[i]) == 0 {
                            continue;
                        }
                        let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                        let u_rel = (u - cu) / (rx * 1.06).max(1.0);
                        let lower_arch = cv + ry * (1.0 - u_rel * u_rel * 0.42) * if ev.is_sd { 0.86 } else { 0.72 };
                        let k = i * 4;
                        let orig = [rgba[k], rgba[k + 1], rgba[k + 2]];
                        let rgb = if m_vis[i] > 0 && !explicit_eye_override {
                            orig
                        } else if (v - lower_arch).abs() <= (if ev.is_sd { rx * 0.03 } else { ry * 0.06 }).max(1.3) {
                            lerp_rgb(ev.palette.lash_color, ev.palette.skin_shadow, 0.45)
                        } else {
                            skin_field.sample(x, y)
                        };
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "browR" | "browL" => {
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        if m_vis[i].max(m_fill[i]) == 0 {
                            continue;
                        }
                        let k = i * 4;
                        let rgb = if m_vis[i] > 0 && !explicit_eye_override {
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        } else {
                            ev.palette.brow_color
                        };
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "mouth_inner" => {
                let (m_rx, m_ry) = mouth_radii(&ev, eye_dist, mouth_style);
                let cavity_dark = [118u8, 48, 62];
                let tongue_pink = [226u8, 128, 134];
                let teeth_white = [250u8, 244, 245];
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        if m_vis[i].max(m_fill[i]) == 0 {
                            continue;
                        }
                        let k = i * 4;
                        let rgb = if m_vis[i] > 0 && !explicit_mouth_override {
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        } else {
                            let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                            let mdx = ((u - ev.mouth_u) / m_rx.max(1.0)).clamp(-1.2, 1.2);
                            let mdy = ((v - ev.mouth_v) / m_ry.max(1.0)).clamp(-1.2, 1.2);
                            if mdy < -0.25 && mdx.abs() < 0.78 && mouth_style != "open_o" {
                                lerp_rgb(teeth_white, cavity_dark, (mdx * mdx * 0.4).min(0.45))
                            } else if mdy > 0.15 + mdx * mdx * 0.35 {
                                lerp_rgb(tongue_pink, cavity_dark, (mdx * mdx * 0.3).min(0.35))
                            } else {
                                cavity_dark
                            }
                        };
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            "mouth_upper" | "mouth_lower" => {
                // mouth_upper / mouth_lower: preserve original lip line + seamless skin cushion over closed mouth_inner
                let (m_rx, m_ry) = mouth_radii(&ev, eye_dist, mouth_style);
                let is_upper = id == "mouth_upper";
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        if m_vis[i].max(m_fill[i]) == 0 {
                            continue;
                        }
                        let (u, v) = frame.to_local(DVec2::new(x as f64, y as f64));
                        let mdx = (u - ev.mouth_u) / m_rx.max(1.0);
                        let lip_curve = match mouth_style {
                            "soft_smile" => -mdx * mdx * eye_dist * 0.015,
                            "cat_w" => (mdx.abs() * std::f64::consts::PI).cos() * eye_dist * 0.012,
                            "open_happy" => mdx * mdx * eye_dist * 0.010,
                            _ => 0.0,
                        };
                        let lip_y = ev.mouth_v + lip_curve;
                        let k = i * 4;
                        let oval_r = mdx.hypot((v - ev.mouth_v) / m_ry.max(1.0));
                        let rgb = if m_vis[i] > 0 && !explicit_mouth_override {
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        } else if mouth_style == "open_o" && (oval_r - 1.0).abs() <= 0.18 {
                            ev.palette.mouth_line
                        } else if (is_upper && (v - lip_y).abs() <= 1.2 && mdx.abs() <= 0.96)
                            || (!is_upper && (v - (lip_y + eye_dist * 0.016)).abs() <= 1.0 && mdx.abs() <= 0.55)
                        {
                            if is_upper {
                                ev.palette.mouth_line
                            } else {
                                lerp_rgb(skin_field.sample(x, y), ev.palette.skin_shadow, 0.36)
                            }
                        } else {
                            skin_field.sample(x, y)
                        };
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
            _ => {
                // Extracted wearable sub-layers (`hat`, `acc_*`, `clothes_*`): preserve original RGBA + fill occluded `hat` behind ribbons cleanly.
                let hat_field = if id == "hat" && m_fill.iter().any(|&v| v > 0) {
                    let hat_vis_b: Vec<bool> = m_vis.iter().map(|&v| v > 0).collect();
                    let hat_eroded = erode_mask(&hat_vis_b, w, h, 2);
                    let mut hat_seeds = vec![false; n];
                    let mut sum = [0u64; 3];
                    let mut cnt = 0u64;
                    let mut seed_cnt = 0usize;
                    for i in 0..n {
                        if !hat_vis_b[i] {
                            continue;
                        }
                        let c = pix_rgb(rgba, i);
                        sum[0] += c[0] as u64;
                        sum[1] += c[1] as u64;
                        sum[2] += c[2] as u64;
                        cnt += 1;
                        if hat_eroded[i] && !is_dark_lineart(c) {
                            hat_seeds[i] = true;
                            seed_cnt += 1;
                        }
                    }
                    if seed_cnt == 0 {
                        hat_seeds = hat_vis_b;
                    }
                    let fallback = [
                        sum[0].checked_div(cnt).unwrap_or(240) as u8,
                        sum[1].checked_div(cnt).unwrap_or(240) as u8,
                        sum[2].checked_div(cnt).unwrap_or(245) as u8,
                    ];
                    Some(build_color_field(rgba, &hat_seeds, w, h, fallback))
                } else {
                    None
                };
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        if m_vis[i].max(m_fill[i]) == 0 {
                            continue;
                        }
                        let k = i * 4;
                        let rgb = if let Some(ref hf) = hat_field {
                            if m_vis[i] == 0 {
                                hf.sample(x, y)
                            } else {
                                [rgba[k], rgba[k + 1], rgba[k + 2]]
                            }
                        } else {
                            [rgba[k], rgba[k + 1], rgba[k + 2]]
                        };
                        tex[k] = rgb[0];
                        tex[k + 1] = rgb[1];
                        tex[k + 2] = rgb[2];
                        tex[k + 3] = 255;
                    }
                }
            }
        }

        // Pad 2px halo with opaque nearest texel color so downstream mesh sampling never pulls black transparent zero RGB!
        pad_texture_rgb(&mut tex, m_vis, m_fill, w, h);

        let mut png_bytes = Vec::new();
        let encoder = PngEncoder::new_with_quality(
            &mut png_bytes,
            CompressionType::Fast,
            FilterType::Sub,
        );
        if encoder
            .write_image(&tex, w as u32, h as u32, ColorType::Rgba8.into())
            .is_ok()
        {
            textures.push(Some(format!(
                "data:image/png;base64,{}",
                STANDARD.encode(&png_bytes)
            )));
        } else {
            textures.push(None);
        }
    }

    // Build per-layer provenance & confidence aligned with evidence.
    let mut provenances = Vec::with_capacity(final_len);
    let mut confidences = Vec::with_capacity(final_len);
    for id in &final_layer_ids {
        if let Some(&ev_idx) = ev.ev_by_id.get(id.as_str()) {
            provenances.push(evidence_provenances[ev_idx].clone());
            confidences.push(evidence_confidences[ev_idx]);
        } else {
            provenances.push("geometry".to_string());
            confidences.push(ev.match_score);
        }
    }

    let mut palette_summary = HashMap::new();
    palette_summary.insert("skin".into(), hex_rgb(ev.palette.skin_base));
    palette_summary.insert("hair".into(), hex_rgb(ev.palette.hair_base));
    palette_summary.insert("irisR".into(), hex_rgb(ev.iris_r.mid));
    palette_summary.insert("irisL".into(), hex_rgb(ev.iris_l.mid));
    palette_summary.insert("clothes".into(), hex_rgb(ev.palette.clothes_primary));

    TemplateSynthesisOutput {
        part_ids: final_layer_ids,
        masks: final_visible_masks,
        fill_masks,
        fill_kinds,
        textures,
        provenances,
        confidences,
        style_match: StyleMatchReport {
            resolved: ev.resolved,
            match_score: ev.match_score,
            metrics: ev.metrics,
            palette_summary,
        },
    }
}

/// Extends valid texel RGB by 2px around effective mask edges to prevent bilinear dark-fringing.
fn pad_texture_rgb(tex: &mut [u8], m_vis: &[u8], m_fill: &[u8], w: usize, h: usize) {
    let n = w * h;
    let mut has = vec![false; n];
    let mut q = VecDeque::new();
    for i in 0..n {
        if m_vis[i] > 0 || m_fill[i] > 0 {
            has[i] = true;
            q.push_back(i);
        }
    }
    let mut depth = vec![0u8; n];
    while let Some(i) = q.pop_front() {
        if depth[i] >= 2 {
            continue;
        }
        let x = i % w;
        let y = i / w;
        let k_src = i * 4;
        for nb in [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ]
        .into_iter()
        .flatten()
        {
            if !has[nb] {
                has[nb] = true;
                depth[nb] = depth[i] + 1;
                let k_dst = nb * 4;
                tex[k_dst] = tex[k_src];
                tex[k_dst + 1] = tex[k_src + 1];
                tex[k_dst + 2] = tex[k_src + 2];
                tex[k_dst + 3] = 255;
                q.push_back(nb);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Point;

    #[test]
    fn template_bank_keeps_partition_disjoint_and_synthesizes_motion_layers() {
        let (w, h) = (256usize, 256usize);
        let n = w * h;
        let mut rgba = vec![0u8; n * 4];
        let lm = Landmarks {
            eye_r: Point { x: 96.0, y: 84.0 },
            eye_l: Point { x: 160.0, y: 84.0 },
            mouth: Point { x: 128.0, y: 118.0 },
            chin: Point { x: 128.0, y: 140.0 },
            crown: Point { x: 128.0, y: 24.0 },
        };
        let mut face_m = vec![0u8; n];
        let mut clothes_m = vec![0u8; n];
        let mut hair_m = vec![0u8; n];
        let mut iris_r_m = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let k = i * 4;
                if (x as f64 - 128.0).hypot((y as f64 - 88.0) * 0.9) < 58.0 {
                    rgba[k..k + 4].copy_from_slice(&[245, 220, 205, 255]);
                    if (x as f64 - 96.0).hypot(y as f64 - 84.0) < 10.0 && y >= 81 {
                        rgba[k..k + 4].copy_from_slice(&[80, 95, 170, 255]);
                        iris_r_m[i] = 255;
                    } else if y < 60 {
                        rgba[k..k + 4].copy_from_slice(&[70, 58, 88, 255]);
                        hair_m[i] = 255;
                    } else {
                        face_m[i] = 255;
                    }
                } else if y > 165 && (x as f64 - 128.0).abs() < 76.0 {
                    rgba[k..k + 4].copy_from_slice(&[210, 212, 224, 255]);
                    clothes_m[i] = 255;
                }
            }
        }
        let ids = vec![
            "hair_front".to_string(),
            "face".to_string(),
            "eyeR_iris".to_string(),
            "clothes".to_string(),
        ];
        let masks = vec![hair_m, face_m, iris_r_m, clothes_m];
        let provs = vec!["model".into(); 4];
        let confs = vec![0.9; 4];

        let out = synthesize_template_bank(
            &rgba,
            w,
            h,
            &lm,
            &[],
            &ids,
            &masks,
            &provs,
            &confs,
            &TemplateStyleSelection::default(),
        );
        assert_eq!(out.part_ids.len(), 21);
        for i in 0..n {
            let owners = out.masks.iter().filter(|m| m[i] > 0).count();
            assert!(owners <= 1, "visible overlap at {i}");
        }
        for (p_idx, id) in out.part_ids.iter().enumerate() {
            for i in 0..n {
                assert!(
                    !(out.masks[p_idx][i] > 0 && out.fill_masks[p_idx][i] > 0),
                    "{id} has overlapping mask and fill_mask at {i}"
                );
            }
        }
        let iris_idx = out.part_ids.iter().position(|s| s == "eyeR_iris").unwrap();
        let fill_px: usize = out.fill_masks[iris_idx].iter().filter(|&&v| v > 0).count();
        assert!(fill_px >= 20, "expected behind-lid iris fill pixels, got {fill_px}");
        let iris_l_idx = out.part_ids.iter().position(|s| s == "eyeL_iris").unwrap();
        let fill_l_px: usize = out.fill_masks[iris_l_idx].iter().filter(|&&v| v > 0).count();
        let eff_l_px: usize = out.masks[iris_l_idx]
            .iter()
            .zip(&out.fill_masks[iris_l_idx])
            .filter(|(&v, &f)| v > 0 || f > 0)
            .count();
        assert!(fill_l_px > 0 && eff_l_px >= 120, "expected full geometric iris_L, got eff={eff_l_px} fill={fill_l_px}");
        let mouth_inner_idx = out.part_ids.iter().position(|s| s == "mouth_inner").unwrap();
        assert!(
            out.fill_masks[mouth_inner_idx].iter().any(|&v| v > 0),
            "expected behind-lip mouth_inner cavity fill"
        );
        assert_eq!(out.style_match.resolved.proportion_mode.as_deref(), Some("ld"));

        // Verify SD chibi archetype supports tall circular eyes, compact neck, and disjoint partition.
        let sd_lm = Landmarks {
            eye_r: Point { x: 86.0, y: 120.0 },
            eye_l: Point { x: 170.0, y: 120.0 },
            mouth: Point { x: 128.0, y: 146.0 },
            chin: Point { x: 128.0, y: 164.0 },
            crown: Point { x: 128.0, y: 40.0 },
        };
        let mut sd_rgba = vec![0u8; n * 4];
        let mut sd_face = vec![0u8; n];
        let mut sd_iris_r = vec![0u8; n];
        let mut sd_iris_l = vec![0u8; n];
        let mut sd_clothes = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let k = i * 4;
                if (x as f64 - 128.0).hypot((y as f64 - 110.0) * 1.15) < 78.0 {
                    sd_rgba[k..k + 4].copy_from_slice(&[250, 232, 224, 255]);
                    if (x as f64 - 86.0).hypot(y as f64 - 120.0) < 22.0 {
                        sd_rgba[k..k + 4].copy_from_slice(&[72, 60, 124, 255]);
                        sd_iris_r[i] = 255;
                    } else if (x as f64 - 170.0).hypot(y as f64 - 120.0) < 22.0 {
                        sd_rgba[k..k + 4].copy_from_slice(&[72, 60, 124, 255]);
                        sd_iris_l[i] = 255;
                    } else {
                        sd_face[i] = 255;
                    }
                } else if y > 172 && (x as f64 - 128.0).abs() < 64.0 {
                    sd_rgba[k..k + 4].copy_from_slice(&[205, 194, 235, 255]);
                    sd_clothes[i] = 255;
                }
            }
        }
        let sd_ids = vec![
            "face".to_string(),
            "eyeR_iris".to_string(),
            "eyeL_iris".to_string(),
            "clothes".to_string(),
        ];
        let sd_masks = vec![sd_face, sd_iris_r, sd_iris_l, sd_clothes];
        let sd_out = synthesize_template_bank(
            &sd_rgba,
            w,
            h,
            &sd_lm,
            &[],
            &sd_ids,
            &sd_masks,
            &["model".into(), "model".into(), "model".into(), "model".into()],
            &[0.92; 4],
            &TemplateStyleSelection {
                proportion_mode: Some("auto".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(sd_out.style_match.resolved.proportion_mode.as_deref(), Some("sd"));
        assert!(
            sd_out.style_match.metrics.eye_ry_ratio >= 0.21,
            "SD archetype must allow tall circular eyes, got {}",
            sd_out.style_match.metrics.eye_ry_ratio
        );
        for i in 0..n {
            assert!(
                sd_out.masks.iter().filter(|m| m[i] > 0).count() <= 1,
                "SD visible overlap at {i}"
            );
        }

        // Verify wearable separation on the 256x256 `sd_lm` canvas: hat, glasses, hairpin/ribbon, multi-layer clothes.
        let mut wear_rgba = sd_rgba.clone();
        let mut wear_hair = vec![0u8; n];
        let mut wear_clothes = vec![0u8; n];
        let mut wear_face = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let k = i * 4;
                let dx = x as f64 - 128.0;
                // White cat-ear hat crown on top (`y in 16..68`)
                if (16..68).contains(&y) && dx.abs() <= 72.0 {
                    wear_rgba[k..k + 4].copy_from_slice(&[246, 240, 250, 255]);
                    wear_hair[i] = 255;
                } else if (68..126).contains(&y) && dx.abs() <= 62.0 && !(y >= 92 && dx.abs() <= 48.0) {
                    // Dark purple-brown hair bangs
                    wear_rgba[k..k + 4].copy_from_slice(&[82, 66, 86, 255]);
                    wear_hair[i] = 255;
                }
                // Contrast bow on white hat
                if (x as f64 - 178.0).hypot(y as f64 - 42.0) <= 15.0 {
                    wear_rgba[k..k + 4].copy_from_slice(&[195, 175, 232, 255]);
                    wear_hair[i] = 255;
                }
                // Contrast hairpin on bangs
                if (x as f64 - 180.0).hypot(y as f64 - 84.0) <= 6.5 {
                    wear_rgba[k..k + 4].copy_from_slice(&[195, 175, 232, 255]);
                    wear_hair[i] = 255;
                }
                // Bilateral round glasses rims + bridge across `sd_lm` eyes ((86, 120) and (170, 120), iris r=22)
                let dr = (x as f64 - 86.0).hypot(y as f64 - 120.0);
                let dl = (x as f64 - 170.0).hypot(y as f64 - 120.0);
                let on_bridge = dx.abs() <= 15.0 && (y as f64 - 119.0).abs() <= 2.5;
                if (25.0..=28.5).contains(&dr) || (25.0..=28.5).contains(&dl) || on_bridge {
                    wear_rgba[k..k + 4].copy_from_slice(&[140, 130, 185, 255]);
                    wear_face[i] = 255;
                } else if sd_masks[0][i] > 0 {
                    wear_face[i] = 255;
                }
                // Two-tone outfit + contrasting chest ribbon below chin (y >= 173)
                if (173..250).contains(&y) && dx.abs() <= 72.0 {
                    let is_ribbon = dx.abs() <= 12.0 && (182..210).contains(&y);
                    let is_collar = dx.abs() <= 40.0 && (173..192).contains(&y) && !is_ribbon;
                    let col = if is_ribbon {
                        [225, 85, 125, 255]
                    } else if is_collar {
                        [248, 246, 252, 255]
                    } else {
                        [185, 178, 232, 255]
                    };
                    wear_rgba[k..k + 4].copy_from_slice(&col);
                    wear_clothes[i] = 255;
                }
            }
        }
        let wear_ids = vec![
            "hair_front".to_string(),
            "face".to_string(),
            "eyeR_iris".to_string(),
            "eyeL_iris".to_string(),
            "clothes".to_string(),
        ];
        let wear_masks = vec![
            wear_hair,
            wear_face,
            sd_masks[1].clone(),
            sd_masks[2].clone(),
            wear_clothes,
        ];
        let wear_out = synthesize_template_bank(
            &wear_rgba,
            w,
            h,
            &sd_lm,
            &[],
            &wear_ids,
            &wear_masks,
            &["model".into(), "model".into(), "model".into(), "model".into(), "model".into()],
            &[0.92; 5],
            &TemplateStyleSelection {
                proportion_mode: Some("sd".to_string()),
                ..Default::default()
            },
        );
        assert!(wear_out.part_ids.iter().any(|id| id == "hat"), "missing hat: {:?}", wear_out.part_ids);
        assert!(wear_out.part_ids.iter().any(|id| id == "acc_glasses"), "missing glasses: {:?}", wear_out.part_ids);
        assert!(wear_out.part_ids.iter().any(|id| id == "acc_ribbon" || id == "acc_hairpin"), "missing hairpin/ribbon: {:?}", wear_out.part_ids);
        assert!(wear_out.part_ids.iter().any(|id| id.starts_with("clothes_")), "missing separated clothes layer: {:?}", wear_out.part_ids);
        for i in 0..n {
            assert!(
                wear_out.masks.iter().filter(|m| m[i] > 0).count() <= 1,
                "Wearable partition overlapped at {i}"
            );
        }
    }

    #[test]
    fn smoothstep_handles_forward_and_reverse_intervals() {
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-9);
        assert!((smoothstep(1.2, -1.2, 0.0) - 0.5).abs() < 1e-9);
        assert!((smoothstep(1.2, -1.2, 1.5) - 0.0).abs() < 1e-9);
        assert!((smoothstep(1.2, -1.2, -1.5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn template_layer_order_places_neck_below_clothes() {
        let neck_pos = TEMPLATE_LAYER_ORDER
            .iter()
            .position(|&id| id == "neck")
            .unwrap();
        let clothes_pos = TEMPLATE_LAYER_ORDER
            .iter()
            .position(|&id| id == "clothes")
            .unwrap();
        let face_pos = TEMPLATE_LAYER_ORDER
            .iter()
            .position(|&id| id == "face")
            .unwrap();
        assert!(neck_pos < clothes_pos);
        assert!(clothes_pos < face_pos);
    }

    #[test]
    fn template_bank_preserves_rest_pose_visible_pixels_and_extra_hair_strands() {
        assert!(!has_explicit_style_override(&TemplateStyleSelection::default()));
        assert!(!has_explicit_style_override(&TemplateStyleSelection {
            proportion_mode: Some("auto".into()),
            face_style: Some("auto".into()),
            ..Default::default()
        }));
        assert!(has_explicit_style_override(&TemplateStyleSelection {
            eye_style: Some("droopy_gentle".into()),
            ..Default::default()
        }));

        let (w, h) = (256usize, 256usize);
        let n = w * h;
        let mut rgba = vec![0u8; n * 4];
        let lm = Landmarks {
            eye_r: Point { x: 96.0, y: 100.0 },
            eye_l: Point { x: 160.0, y: 100.0 },
            mouth: Point { x: 128.0, y: 136.0 },
            chin: Point { x: 128.0, y: 158.0 },
            crown: Point { x: 128.0, y: 34.0 },
        };
        let mut face_m = vec![0u8; n];
        let mut clothes_m = vec![0u8; n];
        let mut hf_m = vec![0u8; n];
        let mut hf2_m = vec![0u8; n];
        let mut er_i_m = vec![0u8; n];
        let mut el_i_m = vec![0u8; n];
        let mut er_w_m = vec![0u8; n];
        let mut el_w_m = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let k = i * 4;
                let dx = x as f64 - 128.0;
                if (dx / 50.0).hypot((y as f64 - 102.0) / 58.0) <= 1.0 {
                    rgba[k..k + 4].copy_from_slice(&[251, 224, 210, 255]);
                    face_m[i] = 255;
                }
                if (156..230).contains(&y) && dx.abs() <= 68.0 {
                    // Distinct chest emblem right below neck (should remain on clothes, untouched by neck fill)
                    let is_emblem = dx.abs() <= 8.0 && (168..184).contains(&y);
                    let col = if is_emblem {
                        [222, 42, 68, 255]
                    } else {
                        [238, 240, 246, 255]
                    };
                    rgba[k..k + 4].copy_from_slice(&col);
                    clothes_m[i] = 255;
                }
                if (34..76).contains(&y) && dx.abs() <= 54.0 {
                    rgba[k..k + 4].copy_from_slice(&[88, 66, 98, 255]);
                    hf_m[i] = 255;
                }
                // Extra split front hair strand on viewer right
                if (54..108).contains(&y) && (158..178).contains(&x) {
                    rgba[k..k + 4].copy_from_slice(&[118, 92, 134, 255]);
                    hf2_m[i] = 255;
                    face_m[i] = 0;
                }
                // Crisp open sclera + distinct cyan iris pixels on both eyes
                let dr_w = ((x as f64 - 96.0) / 14.0).hypot((y as f64 - 100.0) / 9.0);
                let dl_w = ((x as f64 - 160.0) / 14.0).hypot((y as f64 - 100.0) / 9.0);
                let dr_i = (x as f64 - 96.0).hypot(y as f64 - 100.0);
                let dl_i = (x as f64 - 160.0).hypot(y as f64 - 100.0);
                if dr_w <= 1.0 {
                    rgba[k..k + 4].copy_from_slice(&[246, 249, 255, 255]);
                    er_w_m[i] = 255;
                    face_m[i] = 0;
                }
                if dl_w <= 1.0 {
                    rgba[k..k + 4].copy_from_slice(&[246, 249, 255, 255]);
                    el_w_m[i] = 255;
                    face_m[i] = 0;
                }
                if dr_i <= 6.0 {
                    rgba[k..k + 4].copy_from_slice(&[34, 152, 212, 255]);
                    er_i_m[i] = 255;
                    er_w_m[i] = 0;
                }
                if dl_i <= 6.0 {
                    rgba[k..k + 4].copy_from_slice(&[34, 152, 212, 255]);
                    el_i_m[i] = 255;
                    el_w_m[i] = 0;
                }
            }
        }
        let ids = vec![
            "hair_front".to_string(),
            "hair_front_2".to_string(),
            "face".to_string(),
            "eyeR_white".to_string(),
            "eyeL_white".to_string(),
            "eyeR_iris".to_string(),
            "eyeL_iris".to_string(),
            "clothes".to_string(),
        ];
        let masks = vec![hf_m, hf2_m, face_m, er_w_m, el_w_m, er_i_m, el_i_m, clothes_m];
        let out = synthesize_template_bank(
            &rgba,
            w,
            h,
            &lm,
            &[],
            &ids,
            &masks,
            &vec!["model".into(); ids.len()],
            &vec![0.9; ids.len()],
            &TemplateStyleSelection::default(),
        );
        assert!(
            out.part_ids.iter().any(|id| id == "hair_front_2"),
            "split hair strand dropped: {:?}",
            out.part_ids
        );
        // Composite visible masks bottom-to-top using synthesized PNG textures and verify 0px rest-pose distortion!
        let mut comp = vec![0u8; n * 4];
        for (idx, tex_opt) in out.textures.iter().enumerate() {
            let Some(data_url) = tex_opt else {
                continue;
            };
            let b64 = data_url.strip_prefix("data:image/png;base64,").unwrap();
            let raw = STANDARD.decode(b64).unwrap();
            let decoded = image::load_from_memory(&raw).unwrap().to_rgba8();
            let tex_raw = decoded.as_raw();
            let vis = &out.masks[idx];
            for (i, &va) in vis.iter().enumerate() {
                if va > 0 {
                    let k = i * 4;
                    comp[k] = tex_raw[k];
                    comp[k + 1] = tex_raw[k + 1];
                    comp[k + 2] = tex_raw[k + 2];
                    comp[k + 3] = va;
                }
            }
        }
        for i in 0..n {
            if rgba[i * 4 + 3] > 0 {
                let k = i * 4;
                assert_eq!(comp[k..k + 3], rgba[k..k + 3], "Rest-pose pixel altered at index {i}");
            }
        }
    }
}
