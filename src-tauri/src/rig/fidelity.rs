//! T14 — hidden-area fidelity harness (evaluation only; nothing here feeds a project).
//!
//! "Does the revealed hidden area look like the original drawing?" is measured **only inside an
//! exposure band** — pixels that are actually revealed — so shrinking a hidden area can never
//! improve a score: every metric block carries its sample count, exposed area, and exposure
//! frequency next to it, and unexposed hidden texels are reported, never removed.
//!
//! Two bands are supported:
//! - **Ground truth** (`ground_truth_fidelity`): hidden texels of a layer that the T13b exposure map
//!   (`ExposureMap`) reveals in at least one envelope pose, compared with the real PSD layer pixels
//!   (reference PSD, evaluation only). Weight = exposure frequency of the texel.
//! - **Self-supervised holdout** (`build_holdout_plan` / `holdout_fidelity`): for each seam the lower
//!   layer's visible pixels within the seam's measured maximum exposure depth (T13
//!   `SeamRequirement::max_exposure_depth_px`) of the upper layer are held out, re-synthesized by the
//!   *same* texture path as hidden texels (`templates::with_eval_holdout`), and compared with the
//!   original pixels. Weight = exposure frequency of the lower layer's hidden texels under the same
//!   upper layer at the mirrored depth (empty profile → no weighted value).
//!
//! Metrics (per band; `FidelityMetrics`): CIEDE2000 mean/p95, palette-cluster agreement, colour step
//! across the visible↔hidden texture seam (synth vs truth), line-art strokes crossing that seam that
//! break, and local-variance texture difference. Each has an exposure-frequency-weighted variant.

use crate::analysis::templates::EvalHoldout;
use crate::rig::exposure::ExposureMap;
use crate::rig::seams::{outside_distance, visible_owner, SeamRequirement};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};

pub const FIDELITY_VERSION: &str = "t14-v1";
/// Palette clusters (k-means in CIELAB over the layer's visible pixels).
pub const PALETTE_K: usize = 8;
/// Minimum visible context pixels for a palette (otherwise palette agreement is `None`).
pub const PALETTE_MIN_CONTEXT_PX: usize = 64;
/// Truth alpha needed for a texel to count as "truth exists".
pub const TRUTH_ALPHA_MIN: u8 = 128;
/// Line-art: L* at least this far below the layer's visible median L*.
pub const LINE_DROP_L: f32 = 25.0;
/// Stroke-crossing ring depth (px from the visible side).
pub const STROKE_RING_PX: u16 = 3;
/// A crossing stroke counts as broken when less than this fraction of it stays line art.
pub const STROKE_KEEP_MIN: f64 = 0.5;
/// Local variance window radius (5×5).
pub const TEXTURE_WINDOW_R: i64 = 2;

// ---------------------------------------------------------------- colour

fn srgb_linear(c: u8) -> f64 {
    let v = c as f64 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB (D65) → CIELAB (D65 white).
pub fn srgb_to_lab(c: [u8; 3]) -> [f64; 3] {
    let (r, g, b) = (srgb_linear(c[0]), srgb_linear(c[1]), srgb_linear(c[2]));
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    let f = |t: f64| {
        if t > 216.0 / 24389.0 {
            t.cbrt()
        } else {
            (24389.0 / 27.0 * t + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(x / 0.95047), f(y), f(z / 1.08883));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIEDE2000 colour difference (Sharma, Wu & Dalal 2005; kL = kC = kH = 1).
pub fn delta_e2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;
    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let cbar7 = ((c1 + c2) / 2.0).powi(7);
    let p25 = 25f64.powi(7);
    let g = 0.5 * (1.0 - (cbar7 / (cbar7 + p25)).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = a1p.hypot(b1);
    let c2p = a2p.hypot(b2);
    let hue = |b: f64, ap: f64| {
        if b == 0.0 && ap == 0.0 {
            0.0
        } else {
            let h = b.atan2(ap).to_degrees();
            if h < 0.0 {
                h + 360.0
            } else {
                h
            }
        }
    };
    let h1p = hue(b1, a1p);
    let h2p = hue(b2, a2p);
    let dlp = l2 - l1;
    let dcp = c2p - c1p;
    let cprod = c1p * c2p;
    let dhp = if cprod == 0.0 {
        0.0
    } else {
        let d = h2p - h1p;
        if d > 180.0 {
            d - 360.0
        } else if d < -180.0 {
            d + 360.0
        } else {
            d
        }
    };
    let dhp_big = 2.0 * cprod.sqrt() * (dhp.to_radians() / 2.0).sin();
    let lbarp = (l1 + l2) / 2.0;
    let cbarp = (c1p + c2p) / 2.0;
    let hbarp = if cprod == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };
    let t = 1.0 - 0.17 * (hbarp - 30.0).to_radians().cos()
        + 0.24 * (2.0 * hbarp).to_radians().cos()
        + 0.32 * (3.0 * hbarp + 6.0).to_radians().cos()
        - 0.20 * (4.0 * hbarp - 63.0).to_radians().cos();
    let dtheta = 30.0 * (-((hbarp - 275.0) / 25.0).powi(2)).exp();
    let cbarp7 = cbarp.powi(7);
    let rc = 2.0 * (cbarp7 / (cbarp7 + p25)).sqrt();
    let lm50 = (lbarp - 50.0).powi(2);
    let sl = 1.0 + 0.015 * lm50 / (20.0 + lm50).sqrt();
    let sc = 1.0 + 0.045 * cbarp;
    let sh = 1.0 + 0.015 * cbarp * t;
    let rt = -(2.0 * dtheta).to_radians().sin() * rc;
    let (ql, qc, qh) = (dlp / sl, dcp / sc, dhp_big / sh);
    (ql * ql + qc * qc + qh * qh + rt * qc * qh).max(0.0).sqrt()
}

fn lab32(c: [u8; 3]) -> [f32; 3] {
    let l = srgb_to_lab(c);
    [l[0] as f32, l[1] as f32, l[2] as f32]
}

fn de32(a: [f32; 3], b: [f32; 3]) -> f64 {
    delta_e2000(
        [a[0] as f64, a[1] as f64, a[2] as f64],
        [b[0] as f64, b[1] as f64, b[2] as f64],
    )
}

fn lab_d2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

// ---------------------------------------------------------------- metrics

/// Metrics of one exposure band. `None` fields mean "not measurable" (no samples / no weights /
/// no seam pairs), never zero.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FidelityMetrics {
    /// Band pixels measured (synthesized and truth both present).
    pub samples: usize,
    /// Mean / max exposure frequency over the samples (`None` without a weight source).
    pub mean_frequency: Option<f64>,
    pub max_frequency: Option<f64>,
    pub delta_e_mean: f64,
    pub delta_e_p95: f64,
    pub delta_e_mean_freq_weighted: Option<f64>,
    pub delta_e_p95_freq_weighted: Option<f64>,
    /// Fraction of samples whose synthesized colour falls in the same visible-palette cluster as
    /// the truth colour.
    pub palette_clusters: usize,
    /// `None` when the layer has no visible context left (no palette).
    pub palette_match: Option<f64>,
    pub palette_match_freq_weighted: Option<f64>,
    /// 4-adjacent (band, visible) pixel pairs on the visible↔hidden texture seam.
    pub seam_pairs: usize,
    /// Mean ΔE00 step across the seam with the synthesized / the true band colour.
    pub seam_step_synth_de: Option<f64>,
    pub seam_step_truth_de: Option<f64>,
    /// Mean |synth step − truth step| (ΔE00): extra (or missing) discontinuity at the seam.
    pub gradient_discontinuity_de: Option<f64>,
    pub gradient_discontinuity_freq_weighted: Option<f64>,
    /// True line-art strokes that cross from the visible side into the band (ring 3px).
    pub strokes_crossing: usize,
    /// …of which the synthesized band keeps < 50 % as line art.
    pub stroke_breaks: usize,
    /// Σ mean exposure frequency of broken strokes.
    pub stroke_breaks_freq_weighted: Option<f64>,
    /// Mean local L* standard deviation (5×5 window over the layer support).
    pub local_std_synth: f64,
    pub local_std_truth: f64,
    /// Mean |σ_synth − σ_truth|.
    pub texture_std_diff: f64,
    pub texture_std_diff_freq_weighted: Option<f64>,
}

/// Per-layer evaluation context: canvas colours (band = synthesized / true colour, visible = the
/// original pixel in both), visible context mask, band mask, optional frequency weights.
pub struct LayerBand<'a> {
    pub w: usize,
    pub h: usize,
    /// Visible (context) pixels of the layer, excluding the band.
    pub known: &'a [bool],
    /// All band pixels of the layer (measured).
    pub band: &'a [bool],
    pub synth_rgb: &'a [[u8; 3]],
    pub truth_rgb: &'a [[u8; 3]],
    /// Per-pixel weight (exposure frequency); NaN = no weight.
    pub freq: Option<&'a [f32]>,
}

/// Precomputed per-layer data shared by every group (seam) of the layer.
pub struct LayerCtx {
    w: usize,
    h: usize,
    lab_s: HashMap<usize, [f32; 3]>,
    lab_t: HashMap<usize, [f32; 3]>,
    lab_known: HashMap<usize, [f32; 3]>,
    palette: Vec<[f32; 3]>,
    line_thr: f32,
    ring_d: HashMap<usize, u16>,
    std_s: HashMap<usize, f32>,
    std_t: HashMap<usize, f32>,
    freq: HashMap<usize, f32>,
}

fn kmeans_palette(samples: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    if samples.is_empty() {
        return Vec::new();
    }
    let k = k.min(samples.len());
    let mean = samples.iter().fold([0f64; 3], |acc, s| {
        [
            acc[0] + s[0] as f64,
            acc[1] + s[1] as f64,
            acc[2] + s[2] as f64,
        ]
    });
    let nf = samples.len() as f64;
    let mean = [
        (mean[0] / nf) as f32,
        (mean[1] / nf) as f32,
        (mean[2] / nf) as f32,
    ];
    // Deterministic farthest-point initialisation from the sample nearest the mean.
    let first = (0..samples.len())
        .min_by(|&a, &b| lab_d2(samples[a], mean).total_cmp(&lab_d2(samples[b], mean)))
        .unwrap_or(0);
    let mut centers = vec![samples[first]];
    let mut dmin: Vec<f32> = samples.iter().map(|s| lab_d2(*s, centers[0])).collect();
    while centers.len() < k {
        let (far, &d) = dmin
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((0, &0.0));
        if d <= 1e-6 {
            break;
        }
        centers.push(samples[far]);
        for (dm, s) in dmin.iter_mut().zip(samples) {
            *dm = dm.min(lab_d2(*s, samples[far]));
        }
    }
    for _ in 0..12 {
        let mut acc = vec![[0f64; 4]; centers.len()];
        for s in samples {
            let c = nearest(&centers, *s);
            acc[c][0] += s[0] as f64;
            acc[c][1] += s[1] as f64;
            acc[c][2] += s[2] as f64;
            acc[c][3] += 1.0;
        }
        for (c, a) in centers.iter_mut().zip(&acc) {
            if a[3] > 0.0 {
                *c = [
                    (a[0] / a[3]) as f32,
                    (a[1] / a[3]) as f32,
                    (a[2] / a[3]) as f32,
                ];
            }
        }
    }
    centers
}

fn nearest(centers: &[[f32; 3]], s: [f32; 3]) -> usize {
    (0..centers.len())
        .min_by(|&a, &b| lab_d2(centers[a], s).total_cmp(&lab_d2(centers[b], s)))
        .unwrap_or(0)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let k = ((sorted.len() as f64 * q).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[k]
}

fn weighted_quantile(mut vw: Vec<(f64, f64)>, q: f64) -> Option<f64> {
    let total: f64 = vw.iter().map(|x| x.1).sum();
    if total <= 0.0 {
        return None;
    }
    vw.sort_by(|a, b| a.0.total_cmp(&b.0));
    let target = total * q;
    let mut cum = 0.0;
    for (v, wt) in &vw {
        cum += wt;
        if cum >= target {
            return Some(*v);
        }
    }
    vw.last().map(|x| x.0)
}

fn weighted_mean(vw: impl Iterator<Item = (f64, f64)>) -> Option<f64> {
    let (mut s, mut ws) = (0.0, 0.0);
    for (v, wt) in vw {
        s += v * wt;
        ws += wt;
    }
    (ws > 0.0).then(|| s / ws)
}

fn neighbors4(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % w, i / w);
    [
        (x > 0).then(|| i - 1),
        (x + 1 < w).then(|| i + 1),
        (y > 0).then(|| i - w),
        (y + 1 < h).then(|| i + w),
    ]
    .into_iter()
    .flatten()
}

impl LayerCtx {
    pub fn new(b: &LayerBand<'_>) -> Self {
        let (w, h) = (b.w, b.h);
        let n = w * h;
        let mut lab_s = HashMap::new();
        let mut lab_t = HashMap::new();
        let mut lab_known = HashMap::new();
        let mut freq = HashMap::new();
        let mut known_samples = Vec::new();
        let mut known_l = Vec::new();
        let known_count = b.known.iter().filter(|&&k| k).count();
        let stride = known_count.div_ceil(20_000).max(1);
        let mut kc = 0usize;
        for i in 0..n {
            if b.known[i] {
                let l = lab32(b.synth_rgb[i]);
                lab_known.insert(i, l);
                if kc.is_multiple_of(stride) {
                    known_samples.push(l);
                }
                known_l.push(l[0]);
                kc += 1;
            } else if b.band[i] {
                lab_s.insert(i, lab32(b.synth_rgb[i]));
                lab_t.insert(i, lab32(b.truth_rgb[i]));
                if let Some(f) = b.freq {
                    if f[i].is_finite() {
                        freq.insert(i, f[i]);
                    }
                }
            }
        }
        known_l.sort_by(|a, c| a.total_cmp(c));
        let median_l = known_l.get(known_l.len() / 2).copied().unwrap_or(50.0);
        // A palette needs visible context; a few leftover pixels would make agreement trivial.
        let palette = if kc >= PALETTE_MIN_CONTEXT_PX {
            kmeans_palette(&known_samples, PALETTE_K)
        } else {
            Vec::new()
        };
        // Ring distance (4-steps through the band) from the visible side.
        let mut ring_d = HashMap::new();
        let mut q = VecDeque::new();
        for &i in lab_s.keys() {
            if neighbors4(i, w, h).any(|j| b.known[j]) {
                ring_d.insert(i, 1u16);
                q.push_back(i);
            }
        }
        while let Some(i) = q.pop_front() {
            let d = ring_d[&i];
            if d >= STROKE_RING_PX {
                continue;
            }
            for j in neighbors4(i, w, h) {
                if b.band[j] && !b.known[j] && !ring_d.contains_key(&j) {
                    ring_d.insert(j, d + 1);
                    q.push_back(j);
                }
            }
        }
        // Local L* std over the support (visible ∪ band), synth-composed and truth-composed.
        let local_std = |use_truth: bool| -> HashMap<usize, f32> {
            let mut out = HashMap::with_capacity(lab_s.len());
            for &i in lab_s.keys() {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let (mut s, mut s2, mut c) = (0f64, 0f64, 0f64);
                for dy in -TEXTURE_WINDOW_R..=TEXTURE_WINDOW_R {
                    for dx in -TEXTURE_WINDOW_R..=TEXTURE_WINDOW_R {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        let l = if let Some(k) = lab_known.get(&j) {
                            k[0]
                        } else if let Some(v) = if use_truth {
                            lab_t.get(&j)
                        } else {
                            lab_s.get(&j)
                        } {
                            v[0]
                        } else {
                            continue;
                        } as f64;
                        s += l;
                        s2 += l * l;
                        c += 1.0;
                    }
                }
                let var = if c > 0.0 {
                    (s2 / c - (s / c).powi(2)).max(0.0)
                } else {
                    0.0
                };
                out.insert(i, var.sqrt() as f32);
            }
            out
        };
        let std_s = local_std(false);
        let std_t = local_std(true);
        LayerCtx {
            w,
            h,
            lab_s,
            lab_t,
            lab_known,
            palette,
            line_thr: median_l - LINE_DROP_L,
            ring_d,
            std_s,
            std_t,
            freq,
        }
    }

    fn is_line(&self, l: [f32; 3]) -> bool {
        l[0] < self.line_thr
    }

    /// Metrics over the subset `group` of this layer's band (`None` when nothing is measurable).
    pub fn metrics(&self, group: &[usize]) -> Option<FidelityMetrics> {
        let pix: Vec<usize> = group
            .iter()
            .copied()
            .filter(|i| self.lab_s.contains_key(i))
            .collect();
        if pix.is_empty() {
            return None;
        }
        let in_group: std::collections::HashSet<usize> = pix.iter().copied().collect();
        let wt = |i: usize| self.freq.get(&i).map(|&f| f as f64);
        let has_w = pix.iter().any(|&i| wt(i).is_some());
        let mut m = FidelityMetrics {
            samples: pix.len(),
            palette_clusters: self.palette.len(),
            ..Default::default()
        };
        if has_w {
            let fs: Vec<f64> = pix.iter().filter_map(|&i| wt(i)).collect();
            m.mean_frequency = Some(fs.iter().sum::<f64>() / fs.len() as f64);
            m.max_frequency = fs.iter().copied().reduce(f64::max);
        }
        // ΔE00
        let de: Vec<(usize, f64)> = pix
            .iter()
            .map(|&i| (i, de32(self.lab_s[&i], self.lab_t[&i])))
            .collect();
        let mut sorted: Vec<f64> = de.iter().map(|x| x.1).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        m.delta_e_mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        m.delta_e_p95 = quantile(&sorted, 0.95);
        if has_w {
            let vw: Vec<(f64, f64)> = de.iter().map(|&(i, d)| (d, wt(i).unwrap_or(0.0))).collect();
            m.delta_e_mean_freq_weighted = weighted_mean(vw.iter().copied());
            m.delta_e_p95_freq_weighted = weighted_quantile(vw, 0.95);
        }
        // Palette agreement
        if !self.palette.is_empty() {
            let hits: Vec<(usize, f64)> = pix
                .iter()
                .map(|&i| {
                    let same = nearest(&self.palette, self.lab_s[&i])
                        == nearest(&self.palette, self.lab_t[&i]);
                    (i, if same { 1.0 } else { 0.0 })
                })
                .collect();
            m.palette_match = Some(hits.iter().map(|x| x.1).sum::<f64>() / hits.len() as f64);
            if has_w {
                m.palette_match_freq_weighted =
                    weighted_mean(hits.iter().map(|&(i, v)| (v, wt(i).unwrap_or(0.0))));
            }
        }
        // Seam step (visible ↔ band)
        let mut steps: Vec<(usize, f64, f64)> = Vec::new();
        for &i in &pix {
            for j in neighbors4(i, self.w, self.h) {
                if let Some(&k) = self.lab_known.get(&j) {
                    steps.push((i, de32(self.lab_s[&i], k), de32(self.lab_t[&i], k)));
                }
            }
        }
        m.seam_pairs = steps.len();
        if !steps.is_empty() {
            let nf = steps.len() as f64;
            m.seam_step_synth_de = Some(steps.iter().map(|s| s.1).sum::<f64>() / nf);
            m.seam_step_truth_de = Some(steps.iter().map(|s| s.2).sum::<f64>() / nf);
            m.gradient_discontinuity_de =
                Some(steps.iter().map(|s| (s.1 - s.2).abs()).sum::<f64>() / nf);
            if has_w {
                m.gradient_discontinuity_freq_weighted = weighted_mean(
                    steps
                        .iter()
                        .map(|&(i, a, b)| ((a - b).abs(), wt(i).unwrap_or(0.0))),
                );
            }
        }
        // Line-art strokes crossing the seam (truth line components in the 3px ring).
        let ring: Vec<usize> = pix
            .iter()
            .copied()
            .filter(|i| self.ring_d.contains_key(i) && self.is_line(self.lab_t[i]))
            .collect();
        let ring_set: std::collections::HashSet<usize> = ring.iter().copied().collect();
        let mut seen = std::collections::HashSet::new();
        let mut broken_w = 0.0;
        for &s in &ring {
            if seen.contains(&s) {
                continue;
            }
            let mut comp = vec![s];
            seen.insert(s);
            let mut k = 0;
            while k < comp.len() {
                let i = comp[k];
                k += 1;
                let (x, y) = ((i % self.w) as i64, (i / self.w) as i64);
                for dy in -1..=1i64 {
                    for dx in -1..=1i64 {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= self.w as i64 || ny >= self.h as i64 {
                            continue;
                        }
                        let j = ny as usize * self.w + nx as usize;
                        if ring_set.contains(&j) && in_group.contains(&j) && seen.insert(j) {
                            comp.push(j);
                        }
                    }
                }
            }
            let crosses = comp.iter().any(|&i| {
                self.ring_d.get(&i) == Some(&1)
                    && neighbors4(i, self.w, self.h)
                        .any(|j| self.lab_known.get(&j).is_some_and(|&l| self.is_line(l)))
            });
            if !crosses {
                continue;
            }
            m.strokes_crossing += 1;
            let kept = comp
                .iter()
                .filter(|&&i| self.is_line(self.lab_s[&i]))
                .count();
            if (kept as f64) < STROKE_KEEP_MIN * comp.len() as f64 {
                m.stroke_breaks += 1;
                let fs: Vec<f64> = comp.iter().filter_map(|&i| wt(i)).collect();
                if !fs.is_empty() {
                    broken_w += fs.iter().sum::<f64>() / fs.len() as f64;
                }
            }
        }
        if has_w {
            m.stroke_breaks_freq_weighted = Some(broken_w);
        }
        // Texture: local std difference
        let ss: Vec<(usize, f64, f64)> = pix
            .iter()
            .map(|&i| (i, self.std_s[&i] as f64, self.std_t[&i] as f64))
            .collect();
        let nf = ss.len() as f64;
        m.local_std_synth = ss.iter().map(|s| s.1).sum::<f64>() / nf;
        m.local_std_truth = ss.iter().map(|s| s.2).sum::<f64>() / nf;
        m.texture_std_diff = ss.iter().map(|s| (s.1 - s.2).abs()).sum::<f64>() / nf;
        if has_w {
            m.texture_std_diff_freq_weighted = weighted_mean(
                ss.iter()
                    .map(|&(i, a, b)| ((a - b).abs(), wt(i).unwrap_or(0.0))),
            );
        }
        Some(m)
    }

    /// ΔE00 per band pixel (for heat maps).
    pub fn delta_e_at(&self, i: usize) -> Option<f64> {
        Some(de32(*self.lab_s.get(&i)?, *self.lab_t.get(&i)?))
    }
}

// ---------------------------------------------------------------- holdout plan

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HoldoutSeam {
    pub upper: String,
    pub lower: String,
    /// Band width = the seam's measured max exposure depth (T13 requirement).
    pub width_px: f64,
    /// Held-out pixels assigned to this seam (nearest upper layer).
    pub band_px: usize,
    /// `mirror-profile` (lower hidden texels under `upper`, by depth) or `none`.
    pub freq_weight_source: &'static str,
    pub skipped: Option<&'static str>,
}

pub struct HoldoutPlan {
    pub holdout: EvalHoldout,
    pub seams: Vec<HoldoutSeam>,
    /// Per canvas pixel: index into `seams` (−1 = not held out).
    pub seam_of: Vec<i32>,
    /// Mirror exposure-frequency weight per pixel (NaN = none).
    pub freq: Vec<f32>,
}

/// Nearest-source colour transform (multi-source 4-neighbour BFS) from the pixels of `src`.
fn nearest_source(src: &[bool], w: usize, h: usize) -> Vec<i64> {
    let n = w * h;
    let mut near = vec![-1i64; n];
    let mut q = VecDeque::new();
    for i in 0..n {
        if src[i] {
            near[i] = i as i64;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        for j in neighbors4(i, w, h) {
            if near[j] < 0 {
                near[j] = near[i];
                q.push_back(j);
            }
        }
    }
    near
}

/// Builds the self-supervised holdout: for every seam with a measured exposure depth ≥ 1px, the
/// lower layer's visible pixels within that depth (chamfer distance) of the upper layer's visible
/// mask. A pixel claimed by several seams goes to the nearest upper layer. Occluder colours are the
/// nearest upper-layer pixel's colour.
#[allow(clippy::too_many_arguments)]
pub fn build_holdout_plan(
    ids: &[String],
    visible: &[Vec<u8>],
    requirements: &[SeamRequirement],
    exposure: &ExposureMap,
    rgba: &[u8],
    w: usize,
    h: usize,
) -> HoldoutPlan {
    let n = w * h;
    let owner = visible_owner(visible, n);
    let mut dist_cache: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut lower_dist_cache: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut best = vec![f32::INFINITY; n];
    let mut seam_of = vec![-1i32; n];
    let mut seams = Vec::new();
    let mut profiles: Vec<Option<Vec<f32>>> = Vec::new();
    let mut seam_ul: Vec<(usize, usize)> = Vec::new();
    for r in requirements {
        seam_ul.push((r.upper, r.lower));
        let width = r.max_exposure_depth_px;
        let mut row = HoldoutSeam {
            upper: r.upper_id.clone(),
            lower: r.lower_id.clone(),
            width_px: width,
            band_px: 0,
            freq_weight_source: "none",
            skipped: None,
        };
        if width < 1.0 {
            row.skipped = Some("maxExposureDepthPx < 1");
            seams.push(row);
            profiles.push(None);
            continue;
        }
        let du = dist_cache
            .entry(r.upper)
            .or_insert_with(|| outside_distance(&visible[r.upper], w, h));
        let s_idx = seams.len() as i32;
        for i in 0..n {
            if visible[r.lower][i] > 0 && owner[i] == r.lower as i32 {
                let d = du[i];
                if d <= width as f32 && d < best[i] {
                    best[i] = d;
                    seam_of[i] = s_idx;
                }
            }
        }
        // Mirror profile: lower hidden texels under `upper` at rest, mean frequency by depth bin.
        let mut profile = None;
        if let (Some(hid), Some(cnt)) = (exposure.hidden.get(r.lower), exposure.counts.get(r.lower))
        {
            if !cnt.is_empty() && exposure.poses > 0 {
                let dl = lower_dist_cache
                    .entry(r.lower)
                    .or_insert_with(|| outside_distance(&visible[r.lower], w, h));
                let mut acc: Vec<(f64, f64)> = Vec::new();
                for i in 0..n {
                    if hid[i] && owner[i] == r.upper as i32 {
                        let b = dl[i].round().max(0.0) as usize;
                        if acc.len() <= b {
                            acc.resize(b + 1, (0.0, 0.0));
                        }
                        acc[b].0 += cnt[i] as f64 / exposure.poses as f64;
                        acc[b].1 += 1.0;
                    }
                }
                if acc.iter().any(|a| a.1 > 0.0) {
                    profile = Some(
                        acc.iter()
                            .map(|a| {
                                if a.1 > 0.0 {
                                    (a.0 / a.1) as f32
                                } else {
                                    f32::NAN
                                }
                            })
                            .collect(),
                    );
                    row.freq_weight_source = "mirror-profile";
                }
            }
        }
        seams.push(row);
        profiles.push(profile);
    }
    let mut layers: HashMap<String, Vec<bool>> = HashMap::new();
    let mut freq = vec![f32::NAN; n];
    let mut by_upper: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let s = seam_of[i];
        if s < 0 {
            continue;
        }
        let s = s as usize;
        let (upper, lower) = seam_ul[s];
        seams[s].band_px += 1;
        layers
            .entry(ids[lower].clone())
            .or_insert_with(|| vec![false; n])[i] = true;
        by_upper.entry(upper).or_default().push(i);
        if let Some(p) = &profiles[s] {
            let b = best[i].round().max(0.0) as usize;
            // Nearest populated bin within ±2; beyond the deepest bin nothing is exposed → 0.
            let mut f = f32::NAN;
            for off in 0..=2usize {
                for cand in [b.checked_sub(off), Some(b + off)].into_iter().flatten() {
                    if f.is_nan() {
                        if let Some(&v) = p.get(cand) {
                            if v.is_finite() {
                                f = v;
                            }
                        }
                    }
                }
            }
            if f.is_nan() && b >= p.len() {
                f = 0.0;
            }
            freq[i] = if f.is_nan() { 0.0 } else { f };
        }
    }
    let mut occluder_rgb = Vec::new();
    for (upper, pix) in by_upper {
        let src: Vec<bool> = (0..n).map(|i| owner[i] == upper as i32).collect();
        let near = nearest_source(&src, w, h);
        for i in pix {
            let j = near[i];
            if j >= 0 {
                let k = j as usize * 4;
                occluder_rgb.push((i, [rgba[k], rgba[k + 1], rgba[k + 2]]));
            }
        }
    }
    occluder_rgb.sort_by_key(|x| x.0);
    HoldoutPlan {
        holdout: EvalHoldout {
            layers,
            occluder_rgb,
        },
        seams,
        seam_of,
        freq,
    }
}

// ---------------------------------------------------------------- reports

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GroupReport {
    /// Seam upper layer (holdout) / occluding layer at rest (ground truth; `background` if none).
    pub occluder: String,
    pub exposed_px: usize,
    pub metrics: Option<FidelityMetrics>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PartReport {
    pub id: String,
    pub visible_px: usize,
    /// Ground truth: hidden texels / exposed (≥1 pose) / never exposed (T13b exposure map).
    /// Holdout: held-out band pixels in `exposed_px`; `hidden_px`/`unexposed_hidden_px` = None.
    pub hidden_px: Option<usize>,
    pub exposed_px: usize,
    pub unexposed_hidden_px: Option<usize>,
    /// Ground truth: exposed texels without a truth pixel of this part (truth alpha < 128) — the
    /// synthesized hidden area reaches where the PSD part has nothing (shape/coverage mismatch).
    pub no_truth_px: Option<usize>,
    /// …of which no mapped PSD layer at or below this part has a pixel either (truth shows background).
    pub no_truth_background_px: Option<usize>,
    /// Ground truth: the `no_truth_px` texels (minus background) compared with what the PSD shows
    /// there with the upper layers removed (truth of the parts at or below this one, composited).
    pub no_truth_visual_metrics: Option<FidelityMetrics>,
    pub held_out_fraction: Option<f64>,
    /// Holdout: set when ≥ 95 % of the layer's visible pixels are held out — no visible context
    /// remains, so the value is not a seam-band analog and is reported apart.
    pub whole_layer_held_out: Option<bool>,
    pub metrics: Option<FidelityMetrics>,
    pub groups: Vec<GroupReport>,
    pub image: Option<String>,
}

fn rgb_at(rgba: &[u8], i: usize) -> [u8; 3] {
    [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]]
}

/// ΔE heat colour: 0 → dark blue, 10 → yellow, ≥ 20 → red.
pub fn de_ramp(de: f64) -> [u8; 3] {
    let t = (de / 20.0).clamp(0.0, 1.0);
    let stops: [(f64, [f64; 3]); 3] = [
        (0.0, [24.0, 36.0, 92.0]),
        (0.5, [246.0, 212.0, 48.0]),
        (1.0, [214.0, 28.0, 36.0]),
    ];
    let (a, b) = if t <= 0.5 {
        (stops[0], stops[1])
    } else {
        (stops[1], stops[2])
    };
    let u = (t - a.0) / (b.0 - a.0);
    [0, 1, 2].map(|c| (a.1[c] + (b.1[c] - a.1[c]) * u).round() as u8)
}

/// Triptych `truth | synthesized | ΔE00` of one layer, cropped to the band (+24px), upscaled to at
/// least 240px tall. Non-layer pixels are a neutral checkerboard; the ΔE panel shows the visible
/// context in dimmed grey.
pub fn triptych(
    ctx: &LayerCtx,
    band_pixels: &[usize],
    synth: &[[u8; 3]],
    truth: &[[u8; 3]],
) -> Option<image::RgbImage> {
    if band_pixels.is_empty() {
        return None;
    }
    let (w, h) = (ctx.w, ctx.h);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for &i in band_pixels {
        let (x, y) = (i % w, i / w);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let pad = 24;
    let (x0, y0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
    let (x1, y1) = ((x1 + pad).min(w - 1), (y1 + pad).min(h - 1));
    let (cw, ch) = (x1 - x0 + 1, y1 - y0 + 1);
    let scale = (240usize.div_ceil(ch)).clamp(1, 8);
    let gap = 6;
    let pw = cw * scale;
    let mut img = image::RgbImage::from_pixel(
        (pw * 3 + gap * 2) as u32,
        (ch * scale) as u32,
        image::Rgb([255, 255, 255]),
    );
    for yy in 0..ch {
        for xx in 0..cw {
            let i = (y0 + yy) * w + (x0 + xx);
            let checker = if ((xx / 8) + (yy / 8)) % 2 == 0 {
                [205, 205, 205]
            } else {
                [235, 235, 235]
            };
            let (tc, sc, dc) = if ctx.lab_s.contains_key(&i) {
                let de = ctx.delta_e_at(i).unwrap_or(0.0);
                (truth[i], synth[i], de_ramp(de))
            } else if ctx.lab_known.contains_key(&i) {
                let l = ctx.lab_known[&i][0];
                let g = (l / 100.0 * 255.0 * 0.4 + 40.0).clamp(0.0, 255.0) as u8;
                (synth[i], synth[i], [g, g, g])
            } else {
                (checker, checker, checker)
            };
            for (panel, c) in [tc, sc, dc].into_iter().enumerate() {
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = panel * (pw + gap) + xx * scale + sx;
                        let py = yy * scale + sy;
                        img.put_pixel(px as u32, py as u32, image::Rgb(c));
                    }
                }
            }
        }
    }
    Some(img)
}

/// Self-supervised holdout report. `holdout_tex[p]` = the layer textures (straight RGBA, canvas
/// sized) synthesized under `templates::with_eval_holdout(plan.holdout)`; `rgba` = the original
/// input. `image_dir` receives one triptych per layer (`holdout_<id>.png`).
#[allow(clippy::too_many_arguments)]
pub fn holdout_fidelity(
    plan: &HoldoutPlan,
    ids: &[String],
    visible: &[Vec<u8>],
    holdout_tex: &[Option<Vec<u8>>],
    rgba: &[u8],
    w: usize,
    h: usize,
    image_dir: Option<&std::path::Path>,
) -> Vec<PartReport> {
    let n = w * h;
    let mut reports = Vec::new();
    for (p, id) in ids.iter().enumerate() {
        let Some(held) = plan.holdout.layers.get(id) else {
            continue;
        };
        let Some(tex) = holdout_tex.get(p).and_then(|t| t.as_ref()) else {
            continue;
        };
        let known: Vec<bool> = (0..n).map(|i| visible[p][i] > 0 && !held[i]).collect();
        let truth: Vec<[u8; 3]> = (0..n).map(|i| rgb_at(rgba, i)).collect();
        let synth: Vec<[u8; 3]> = (0..n)
            .map(|i| if held[i] { rgb_at(tex, i) } else { truth[i] })
            .collect();
        let band = LayerBand {
            w,
            h,
            known: &known,
            band: held,
            synth_rgb: &synth,
            truth_rgb: &truth,
            freq: Some(&plan.freq),
        };
        let ctx = LayerCtx::new(&band);
        let all: Vec<usize> = (0..n).filter(|&i| held[i]).collect();
        let mut groups = Vec::new();
        let mut by_seam: HashMap<i32, Vec<usize>> = HashMap::new();
        for &i in &all {
            by_seam.entry(plan.seam_of[i]).or_default().push(i);
        }
        let mut keys: Vec<i32> = by_seam.keys().copied().collect();
        keys.sort();
        for k in keys {
            let g = &by_seam[&k];
            groups.push(GroupReport {
                occluder: plan.seams[k as usize].upper.clone(),
                exposed_px: g.len(),
                metrics: ctx.metrics(g),
            });
        }
        let visible_px = visible[p].iter().filter(|&&a| a > 0).count();
        let image = image_dir.and_then(|dir| {
            let name = format!("holdout_{id}.png");
            triptych(&ctx, &all, &synth, &truth)
                .and_then(|im| im.save(dir.join(&name)).ok())
                .map(|_| name)
        });
        let fraction = all.len() as f64 / visible_px.max(1) as f64;
        reports.push(PartReport {
            id: id.clone(),
            visible_px,
            hidden_px: None,
            exposed_px: all.len(),
            unexposed_hidden_px: None,
            no_truth_px: None,
            no_truth_background_px: None,
            no_truth_visual_metrics: None,
            held_out_fraction: Some(fraction),
            whole_layer_held_out: Some(fraction >= 0.95),
            metrics: ctx.metrics(&all),
            groups,
            image,
        });
    }
    reports
}

/// Ground-truth report: exposed hidden texels (T13b exposure map, ≥1 envelope pose) of each layer
/// with a truth image, compared with the truth layer pixels. `tex[p]` = the product textures
/// (straight RGBA); `truth` maps layer id → straight RGBA truth (canvas sized, same pixel grid).
#[allow(clippy::too_many_arguments)]
pub fn ground_truth_fidelity(
    ids: &[String],
    visible: &[Vec<u8>],
    exposure: &ExposureMap,
    tex: &[Option<Vec<u8>>],
    truth: &HashMap<String, Vec<u8>>,
    rgba: &[u8],
    w: usize,
    h: usize,
    image_dir: Option<&std::path::Path>,
) -> Vec<PartReport> {
    let n = w * h;
    let owner = visible_owner(visible, n);
    let mut reports = Vec::new();
    for (p, id) in ids.iter().enumerate() {
        let Some(tr) = truth.get(id) else {
            continue;
        };
        let layer = exposure.layer(id);
        let counts = exposure.counts.get(p).filter(|c| !c.is_empty());
        let visible_px = visible[p].iter().filter(|&&a| a > 0).count();
        let mut report = PartReport {
            id: id.clone(),
            visible_px,
            hidden_px: layer.map(|l| l.hidden_px),
            exposed_px: layer.map_or(0, |l| l.exposed_px),
            unexposed_hidden_px: layer.map(|l| l.unexposed_hidden_px),
            no_truth_px: Some(0),
            no_truth_background_px: Some(0),
            no_truth_visual_metrics: None,
            held_out_fraction: None,
            whole_layer_held_out: None,
            metrics: None,
            groups: Vec::new(),
            image: None,
        };
        let (Some(counts), Some(tx)) = (counts, tex.get(p).and_then(|t| t.as_ref())) else {
            reports.push(report);
            continue;
        };
        let mut band = vec![false; n];
        let mut freq = vec![f32::NAN; n];
        let mut no_truth: Vec<usize> = Vec::new();
        for i in 0..n {
            if counts[i] > 0 && exposure.hidden[p][i] {
                freq[i] = counts[i] as f32 / exposure.poses.max(1) as f32;
                if tr[i * 4 + 3] >= TRUTH_ALPHA_MIN {
                    band[i] = true;
                } else {
                    no_truth.push(i);
                }
            }
        }
        report.no_truth_px = Some(no_truth.len());
        let known: Vec<bool> = (0..n).map(|i| visible[p][i] > 0).collect();
        let orig: Vec<[u8; 3]> = (0..n).map(|i| rgb_at(rgba, i)).collect();
        // What the PSD shows at texels this part does not cover: truth of the parts at or below it
        // (flat2rig order), composited bottom-to-top.
        if !no_truth.is_empty() {
            let below: Vec<&Vec<u8>> = ids[..=p].iter().filter_map(|q| truth.get(q)).collect();
            let mut vis_band = vec![false; n];
            let mut vis_rgb = orig.clone();
            let mut background = 0usize;
            for &i in &no_truth {
                let mut acc = [0f64; 4];
                for t in &below {
                    let a = t[i * 4 + 3] as f64 / 255.0;
                    for c in 0..3 {
                        acc[c] = t[i * 4 + c] as f64 * a + acc[c] * (1.0 - a);
                    }
                    acc[3] = a + acc[3] * (1.0 - a);
                }
                if acc[3] * 255.0 >= TRUTH_ALPHA_MIN as f64 {
                    vis_band[i] = true;
                    vis_rgb[i] =
                        [0, 1, 2].map(|c| (acc[c] / acc[3]).round().clamp(0.0, 255.0) as u8);
                } else {
                    background += 1;
                }
            }
            report.no_truth_background_px = Some(background);
            let synth_v: Vec<[u8; 3]> = (0..n)
                .map(|i| if vis_band[i] { rgb_at(tx, i) } else { orig[i] })
                .collect();
            let lbv = LayerBand {
                w,
                h,
                known: &known,
                band: &vis_band,
                synth_rgb: &synth_v,
                truth_rgb: &vis_rgb,
                freq: Some(&freq),
            };
            let ctx_v = LayerCtx::new(&lbv);
            let pix: Vec<usize> = no_truth.iter().copied().filter(|&i| vis_band[i]).collect();
            report.no_truth_visual_metrics = ctx_v.metrics(&pix);
        }
        let synth: Vec<[u8; 3]> = (0..n)
            .map(|i| if band[i] { rgb_at(tx, i) } else { orig[i] })
            .collect();
        let truth_rgb: Vec<[u8; 3]> = (0..n)
            .map(|i| if band[i] { rgb_at(tr, i) } else { orig[i] })
            .collect();
        for (i, f) in freq.iter_mut().enumerate() {
            if !band[i] {
                *f = f32::NAN;
            }
        }
        let lb = LayerBand {
            w,
            h,
            known: &known,
            band: &band,
            synth_rgb: &synth,
            truth_rgb: &truth_rgb,
            freq: Some(&freq),
        };
        let ctx = LayerCtx::new(&lb);
        let all: Vec<usize> = (0..n).filter(|&i| band[i]).collect();
        let mut by_owner: HashMap<i32, Vec<usize>> = HashMap::new();
        for &i in &all {
            by_owner.entry(owner[i]).or_default().push(i);
        }
        let mut keys: Vec<i32> = by_owner.keys().copied().collect();
        keys.sort();
        for k in keys {
            let g = &by_owner[&k];
            report.groups.push(GroupReport {
                occluder: if k >= 0 {
                    ids[k as usize].clone()
                } else {
                    "background".into()
                },
                exposed_px: g.len(),
                metrics: ctx.metrics(g),
            });
        }
        report.metrics = ctx.metrics(&all);
        report.image = image_dir.and_then(|dir| {
            let name = format!("truth_{id}.png");
            triptych(&ctx, &all, &synth, &truth_rgb)
                .and_then(|im| im.save(dir.join(&name)).ok())
                .map(|_| name)
        });
        reports.push(report);
    }
    reports
}

/// Short metric definitions written next to the numbers in `stats.json`.
pub fn metrics_spec() -> serde_json::Value {
    serde_json::json!({
        "version": FIDELITY_VERSION,
        "band": "groundTruth: hidden texels exposed in >=1 of the shared envelope poses (T13b ExposureMap) with truth alpha >= 128; holdout: lower-layer visible pixels within the seam's maxExposureDepthPx (T13 requirement) of the upper layer, re-synthesized by the hidden-texel path",
        "weights": "groundTruth: exposure frequency of the texel; holdout: mean exposure frequency of the lower layer's hidden texels under the same upper layer at the mirrored depth (none when that layer has no hidden texels there)",
        "deltaE": "CIEDE2000 (sRGB D65 -> CIELAB), mean and 95th percentile",
        "paletteMatch": "share of samples whose synthesized and true colours fall in the same k-means (k<=8, CIELAB) cluster of the layer's visible pixels (null with < 64 visible context px)",
        "gradientDiscontinuity": "mean |dE00(synth band px, visible neighbour) - dE00(true band px, visible neighbour)| over 4-adjacent band/visible pairs; seamStep* are the two means",
        "strokes": "true line-art components (L* < visible median - 25) in the 3px ring that continue a visible line pixel; broken when < 50% of the component is line art in the synthesized band",
        "texture": "mean |sigma_synth - sigma_truth| of L* in a 5x5 window over the layer support (visible + band)",
        "noGlobalScore": "no combined score or threshold is defined (T14 baseline only)"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sharma, Wu & Dalal (2005), Table 1 — selected pairs (L, a, b) → ΔE00 (4 decimals).
    const SHARMA: &[([f64; 3], [f64; 3], f64)] = &[
        ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
        ([50.0, 3.1571, -77.2803], [50.0, 0.0, -82.7485], 2.8615),
        ([50.0, 2.8361, -74.0200], [50.0, 0.0, -82.7485], 3.4412),
        ([50.0, -1.3802, -84.2814], [50.0, 0.0, -82.7485], 1.0000),
        ([50.0, 0.0, 0.0], [50.0, -1.0, 2.0], 2.3669),
        ([50.0, 2.4900, -0.0010], [50.0, -2.4900, 0.0009], 7.1792),
        ([50.0, 2.4900, -0.0010], [50.0, -2.4900, 0.0011], 7.2195),
        ([50.0, -0.0010, 2.4900], [50.0, 0.0009, -2.4900], 4.8045),
        ([50.0, -0.0010, 2.4900], [50.0, 0.0011, -2.4900], 4.7461),
        ([50.0, 2.5, 0.0], [50.0, 0.0, -2.5], 4.3065),
        ([50.0, 2.5, 0.0], [73.0, 25.0, -18.0], 27.1492),
        ([50.0, 2.5, 0.0], [61.0, -5.0, 29.0], 22.8977),
        ([50.0, 2.5, 0.0], [56.0, -27.0, -3.0], 31.9030),
        ([50.0, 2.5, 0.0], [58.0, 24.0, 15.0], 19.4535),
        ([50.0, 2.5, 0.0], [50.0, 3.1736, 0.5854], 1.0000),
        (
            [60.2574, -34.0099, 36.2677],
            [60.4626, -34.1751, 39.4387],
            1.2644,
        ),
        (
            [63.0109, -31.0961, -5.8663],
            [62.8187, -29.7946, -4.0864],
            1.2630,
        ),
        (
            [22.7233, 20.0904, -46.6940],
            [23.0331, 14.9730, -42.5619],
            2.0373,
        ),
        (
            [90.8027, -2.0831, 1.4410],
            [91.1528, -1.6435, 0.0447],
            1.4441,
        ),
        (
            [6.7747, -0.2908, -2.4247],
            [5.8714, -0.0985, -2.2286],
            0.6377,
        ),
        (
            [2.0776, 0.0795, -1.1350],
            [0.9033, -0.0636, -0.5514],
            0.9082,
        ),
    ];

    #[test]
    fn delta_e2000_matches_sharma_2005_reference_pairs_and_is_symmetric() {
        for (a, b, want) in SHARMA {
            let got = delta_e2000(*a, *b);
            assert!((got - want).abs() < 1e-4, "{a:?} {b:?}: {got} vs {want}");
            let rev = delta_e2000(*b, *a);
            assert!((rev - want).abs() < 1e-4, "symmetry {a:?} {b:?}: {rev}");
        }
        assert_eq!(delta_e2000([40.0, 10.0, -5.0], [40.0, 10.0, -5.0]), 0.0);
    }

    #[test]
    fn srgb_to_lab_white_black_and_primary() {
        let w = srgb_to_lab([255, 255, 255]);
        assert!(
            (w[0] - 100.0).abs() < 1e-3 && w[1].abs() < 1e-2 && w[2].abs() < 1e-2,
            "{w:?}"
        );
        let k = srgb_to_lab([0, 0, 0]);
        assert!(k[0].abs() < 1e-9, "{k:?}");
        // sRGB red ≈ (53.24, 80.09, 67.20).
        let r = srgb_to_lab([255, 0, 0]);
        assert!(
            (r[0] - 53.24).abs() < 0.02
                && (r[1] - 80.09).abs() < 0.05
                && (r[2] - 67.20).abs() < 0.05,
            "{r:?}"
        );
    }

    /// 32×16 layer: visible x<16 (flat skin with a vertical dark line at y∈[6,8]), band x∈[16,24).
    type Layer = (Vec<bool>, Vec<bool>, Vec<[u8; 3]>, Vec<[u8; 3]>);

    fn layer(synth_band: impl Fn(usize, usize) -> [u8; 3]) -> Layer {
        let (w, h) = (32usize, 16usize);
        let n = w * h;
        let mut known = vec![false; n];
        let mut band = vec![false; n];
        let mut truth = vec![[255u8, 255, 255]; n];
        let mut synth = vec![[255u8, 255, 255]; n];
        for y in 0..h {
            for x in 0..24 {
                let i = y * w + x;
                let line = (6..9).contains(&y);
                let c = if line { [40, 30, 40] } else { [240, 200, 180] };
                truth[i] = c;
                if x < 16 {
                    known[i] = true;
                    synth[i] = c;
                } else {
                    band[i] = true;
                    synth[i] = synth_band(x, y);
                }
            }
        }
        (known, band, synth, truth)
    }

    #[test]
    fn identical_band_scores_zero_and_keeps_strokes() {
        let (known, band, synth, truth) = layer(|_, y| {
            if (6..9).contains(&y) {
                [40, 30, 40]
            } else {
                [240, 200, 180]
            }
        });
        let freq = vec![0.5f32; 32 * 16];
        let lb = LayerBand {
            w: 32,
            h: 16,
            known: &known,
            band: &band,
            synth_rgb: &synth,
            truth_rgb: &truth,
            freq: Some(&freq),
        };
        let ctx = LayerCtx::new(&lb);
        let all: Vec<usize> = (0..32 * 16).filter(|&i| band[i]).collect();
        let m = ctx.metrics(&all).unwrap();
        assert_eq!(m.samples, 8 * 16);
        assert!(m.delta_e_mean < 1e-9 && m.delta_e_p95 < 1e-9);
        assert!((m.palette_match.unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(m.strokes_crossing, 1);
        assert_eq!(m.stroke_breaks, 0);
        assert!(m.gradient_discontinuity_de.unwrap() < 1e-9);
        assert!(m.texture_std_diff < 1e-6);
        assert!((m.mean_frequency.unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn flat_fill_breaks_the_stroke_and_raises_delta_e_but_not_by_shrinking_the_band() {
        // Harmonic-style flat fill: skin everywhere, the line is lost.
        let (known, band, synth, truth) = layer(|_, _| [240, 200, 180]);
        let lb = LayerBand {
            w: 32,
            h: 16,
            known: &known,
            band: &band,
            synth_rgb: &synth,
            truth_rgb: &truth,
            freq: None,
        };
        let ctx = LayerCtx::new(&lb);
        let all: Vec<usize> = (0..32 * 16).filter(|&i| band[i]).collect();
        let m = ctx.metrics(&all).unwrap();
        assert_eq!(m.strokes_crossing, 1);
        assert_eq!(m.stroke_breaks, 1);
        assert!(m.delta_e_p95 > 20.0, "{m:?}");
        assert!(m.texture_std_diff > 1.0, "{m:?}");
        assert!(m.gradient_discontinuity_de.unwrap() > 1.0, "{m:?}");
        assert!(m.mean_frequency.is_none() && m.delta_e_mean_freq_weighted.is_none());
        // Mean ΔE is per sample: dropping the line rows from the band would lower it, so the
        // report always carries `samples` next to the value (checked: 3 of 16 rows are line).
        let line_rows: Vec<usize> = all
            .iter()
            .copied()
            .filter(|&i| (6..9).contains(&(i / 32)))
            .collect();
        let ml = ctx.metrics(&line_rows).unwrap();
        assert!(ml.delta_e_mean > m.delta_e_mean);
        assert_eq!(ml.samples + 13 * 8, m.samples);
    }

    /// lower `neck` visible x<20 with hidden fill x∈[20,30) under upper `clothes` (visible x≥20).
    /// Seam depth 6px → the band is neck x∈[14,20); occluder colour = clothes; mirror weights come
    /// from the neck hidden texels' exposure counts at the same depth.
    #[test]
    fn holdout_plan_band_width_occluder_and_mirror_weights() {
        use crate::core::PartRole;
        let (w, h) = (40usize, 8usize);
        let n = w * h;
        let mut lower = vec![0u8; n];
        let mut upper = vec![0u8; n];
        let mut rgba = vec![0u8; n * 4];
        for i in 0..n {
            if i % w < 20 {
                lower[i] = 255;
                rgba[i * 4..i * 4 + 4].copy_from_slice(&[230, 200, 190, 255]);
            } else {
                upper[i] = 255;
                rgba[i * 4..i * 4 + 4].copy_from_slice(&[40, 50, 120, 255]);
            }
        }
        let ids = vec!["neck".to_string(), "clothes".to_string()];
        let req = SeamRequirement {
            upper: 1,
            lower: 0,
            upper_id: "clothes".into(),
            lower_id: "neck".into(),
            upper_role: PartRole::Clothes,
            lower_role: PartRole::Neck,
            boundary_points: h,
            sampled_points: h,
            poses: 4,
            max_relative_displacement_px: 6.0,
            max_exposure_depth_px: 6.0,
            required_underlap_px: 10,
            worst_pose: None,
            worst_point: None,
            unsupported_samples: 0,
            coupled_underlap_px: None,
            uncoupled_underlap_px: None,
            uncoupled_max_relative_displacement_px: None,
            adopted: None,
        };
        // Exposure: neck hidden x∈[20,30); exposed in 3 of 4 poses at depth ≤ 3, 1 pose deeper.
        let mut hidden = vec![false; n];
        let mut counts = vec![0u16; n];
        for i in 0..n {
            let x = i % w;
            if (20..30).contains(&x) {
                hidden[i] = true;
                counts[i] = if x < 23 { 3 } else { 1 };
            }
        }
        let exposure = ExposureMap {
            width: w,
            height: h,
            poses: 4,
            hidden: vec![hidden, vec![false; n]],
            counts: vec![counts, Vec::new()],
            ..Default::default()
        };
        let plan = build_holdout_plan(&ids, &[lower, upper], &[req], &exposure, &rgba, w, h);
        let band = &plan.holdout.layers["neck"];
        for (i, &held) in band.iter().enumerate() {
            let x = i % w;
            assert_eq!(held, (14..20).contains(&x), "x={x}");
        }
        assert_eq!(plan.seams[0].band_px, 6 * h);
        assert_eq!(plan.seams[0].freq_weight_source, "mirror-profile");
        assert_eq!(plan.holdout.occluder_rgb.len(), 6 * h);
        assert!(plan
            .holdout
            .occluder_rgb
            .iter()
            .all(|(_, c)| *c == [40, 50, 120]));
        // Band x=19 is 1px from clothes → mirror depth 1 → frequency 3/4; x=14 is 6px → 1/4.
        assert!((plan.freq[19] - 0.75).abs() < 1e-6, "{}", plan.freq[19]);
        assert!((plan.freq[14] - 0.25).abs() < 1e-6, "{}", plan.freq[14]);
        assert!(plan.freq[5].is_nan());
    }

    #[test]
    fn weighted_quantile_and_mean() {
        let v = vec![(1.0, 1.0), (2.0, 1.0), (10.0, 8.0)];
        assert_eq!(weighted_quantile(v.clone(), 0.5), Some(10.0));
        assert_eq!(weighted_quantile(v.clone(), 0.1), Some(1.0));
        assert!((weighted_mean(v.into_iter()).unwrap() - 8.3).abs() < 1e-12);
        assert_eq!(weighted_mean(std::iter::once((3.0, 0.0))), None);
    }
}
