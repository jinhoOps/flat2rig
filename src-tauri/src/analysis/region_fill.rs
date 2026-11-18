//! T9 — hidden-area colour synthesis by region type (Rust only).
//!
//! Goal: the hidden area (`fillMask`) of a layer must look like the original drawing when motion
//! reveals it. This module never changes a mask; it only decides *colours*:
//!
//! 1. **Clean inner seeds** (`analyze_layer`): visible pixels of the layer with alpha ≥
//!    `SEED_ALPHA_MIN`, eroded `SEED_ERODE_PX` from the visible boundary (background matte,
//!    neighbour line art and anti-aliased mixtures live in that rim), minus thin dark strokes
//!    (`line`: L* more than `LINE_DROP_L` below the layer median *and* removed by a
//!    `LINE_OPEN_PX` morphological opening — wide dark regions such as a navy collar stay seeds),
//!    minus caller exclusions (other-class evidence) and evaluation holdout pixels.
//! 2. **Region type** (`RegionKind`), from statistics of the seeds next to the hidden area (within
//!    `band_radius` px; whole layer when that is too few): `semantic` (caller role: iris, mouth
//!    cavity — structure that colour statistics cannot extend), `line` (stroke-dominated layer),
//!    `texture` (high-frequency variation), otherwise `flat_cel`. This is a *type*, separate from
//!    the existing `fillKind` (*source*: none / neighbor / external / template).
//! 3. **`flat_cel` palette field** (`RegionFiller`): k-means palette (≤ 4 clusters in CIELAB,
//!    merged below ΔE00 `PALETTE_MERGE_DE`, clusters under `PALETTE_MIN_SHARE` dropped). When the
//!    clusters are tight (`cel`: ≥ `CEL_CORE_FRAC` of seeds have ρ = d1/(d1+d2) < `CEL_RHO`, i.e. sit
//!    at a cluster centre rather than between two), the hidden colour is snapped: each cluster's
//!    indicator is interpolated harmonically and the argmax (a level set) picks the cluster, so a
//!    shadow boundary continues into the hidden area instead of being averaged into a gradient; the
//!    colour is that cluster's own local harmonic colour field (keeps in-cluster tone variation).
//!    Gradient-painted layers (clusters not tight) and `texture` layers use one local harmonic field
//!    of all seeds (no snap).
//! 4. **Rim decontamination** (`decontaminate_rim`): the outer `RIM_PX` visible pixels are compared
//!    with the nearest clean seed colour F and the colour on the other side of the boundary:
//!    - background side: a pixel on the F→matte mixing line (matte = opaque background colour, or
//!      white for transparent inputs) is a halo → F. Darker than F by `LINE_DROP_L` = the layer's
//!      own outline → kept.
//!    - upper-layer side (this layer is below): a line pixel (L* `RIM_LINE_DROP_L` below F) is the
//!      upper layer's outline → F, but only when the upper layer keeps a similarly dark pixel within
//!      `RIM_OWNER_SEARCH_PX` (otherwise the partition gave the whole line to this layer and erasing
//!      it would delete it at rest: counted as `keptUnownedLine`). Mixtures toward the upper colour
//!      → F.
//!    - lower-layer side (this layer is above): lightening / hue mixtures toward the lower colour
//!      → F; a pixel darker than F by more than `LOWER_MIX_MAX_DARKEN_L` is kept (it may be this
//!      layer's own seam outline over a darker lower layer).
//!    - F: for mixtures F is the clean seed within `RIM_F_WINDOW` px closest in colour to the rim pixel
//!      (its own cel region; the BFS-nearest seed when none is that close); line art is judged
//!      against and replaced by the spatially nearest clean seed (the region behind the line).
//!    - A rim pixel with no clean seed within `RIM_F_WINDOW` px is left as drawn (`noLocalSeed`).
//!    - The mixing test: with d = N − F (|d| ≥ `MIX_MIN_SEP`), t = (C − F)·d / |d|² ≥ `MIX_T_MIN`,
//!      t ≤ 1.15, perpendicular residual ≤ `MIX_PERP_BASE` + `MIX_PERP_REL`·|d| (RGB units), and
//!      ΔE00(C, F) ≥ `MIX_MIN_DE`.

use crate::rig::fidelity::{delta_e2000, srgb_to_lab};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const REGION_FILL_VERSION: &str = "t9-v1";
/// Seed: visible alpha at least this (anti-aliased edge pixels are excluded).
pub const SEED_ALPHA_MIN: u8 = 128;
/// Seed: eroded this many px (4-connected) from the visible boundary.
pub const SEED_ERODE_PX: usize = 2;
/// Line art: L* at least this far below the layer's visible median (same rule as T14 `fidelity`).
pub const LINE_DROP_L: f32 = 25.0;
/// Line art: dark pixels removed by an opening of this radius (strokes thinner than ~2r+1 px).
pub const LINE_OPEN_PX: usize = 2;
/// Fewer own seeds than this → caller falls back to its class field (`fieldSource = "class"`).
pub const MIN_LAYER_SEEDS: usize = 32;
/// Own palette / field only when the clean seeds are at least this fraction of the hidden texels:
/// a small visible sliver cannot predict a much larger hidden area (it would be extrapolated as
/// streaks); the caller keeps its class field then.
pub const OWN_FIELD_MIN_SEED_RATIO: f64 = 0.10;
/// Fewer band seeds than this → statistics use all seeds of the layer.
pub const MIN_BAND_SEEDS: usize = 64;
/// `line` when stroke pixels are at least this fraction of the visible support…
pub const CLASS_LINE_FRAC: f64 = 0.30;
/// …or when fewer than this fraction of the support survives seed erosion (stroke-thin layer).
pub const CLASS_THIN_SEED_FRAC: f64 = 0.15;
/// `texture` when the median 5×5 local L* standard deviation over band seeds is at least this.
pub const CLASS_TEXTURE_STD: f64 = 3.0;
pub const PALETTE_K_MAX: usize = 4;
pub const PALETTE_MERGE_DE: f64 = 6.0;
pub const PALETTE_MIN_SHARE: f64 = 0.03;
/// Cel test: ρ = d1/(d1+d2) below this = the seed sits at its cluster centre.
pub const CEL_RHO: f64 = 0.15;
/// Cel test: at least this fraction of band seeds must sit at a centre (a uniform ramp gives ≈0.3).
pub const CEL_CORE_FRAC: f64 = 0.75;
/// Rim depth (px, 4-connected) checked for contamination.
pub const RIM_PX: u16 = 2;
pub const RIM_LINE_DROP_L: f32 = 20.0;
pub const RIM_OWNER_SEARCH_PX: i64 = 3;
/// Lower-layer side: rim pixels darker than F by more than this (L*) are never replaced.
pub const LOWER_MIX_MAX_DARKEN_L: f32 = 3.0;
/// F candidates: clean seeds within this Chebyshev radius of the rim pixel.
pub const RIM_F_WINDOW: i64 = 4;
pub const RIM_OWNER_LINE_SLACK_L: f32 = 12.0;
pub const MIX_MIN_SEP: f32 = 24.0;
pub const MIX_T_MIN: f32 = 0.15;
pub const MIX_PERP_BASE: f32 = 12.0;
pub const MIX_PERP_REL: f32 = 0.20;
pub const MIX_MIN_DE: f64 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionKind {
    FlatCel,
    Line,
    Texture,
    Semantic,
}

impl RegionKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RegionKind::FlatCel => "flat_cel",
            RegionKind::Line => "line",
            RegionKind::Texture => "texture",
            RegionKind::Semantic => "semantic",
        }
    }
}

/// Rim decontamination counts (visible pixels whose texture colour was replaced by F).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RimReport {
    pub rim_px: usize,
    pub background_halo: usize,
    pub upper_line: usize,
    pub upper_mix: usize,
    pub lower_mix: usize,
    pub kept_unowned_line: usize,
    /// Rim pixels left as drawn because no clean seed lies within `RIM_F_WINDOW` px.
    pub no_local_seed: usize,
    /// Face only: eye / brow / mouth line residue on `face` replaced by the skin field (T9 item 2).
    pub feature_residue: usize,
    /// Upper layer only: lower-rim seam pixels (upper outline half, upper-side anti-alias mixture,
    /// face feature residue) whose original colour this layer received as overlay `fillMask`
    /// texels, so the rest pose keeps the full outline while the lower texture is cleaned.
    pub overlay_px: usize,
}

/// Per-layer region type and how its hidden colours were synthesized (stored as `Part.fillRegion`
/// / `Part.fillSynth`; the statistics are reported in CLI `stats.json`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RegionReport {
    pub version: String,
    pub kind: RegionKind,
    /// Palette snap + level-set extension used (flat/cel with tight clusters).
    pub snap: bool,
    /// Hidden-texel synthesis: `palette_levelset` | `local_field` | `class_field` | `procedural` |
    /// `none` (no hidden texels).
    pub synth: String,
    /// Seeds came from this layer (`layer`) or the caller's class field (`class`).
    pub field_source: String,
    pub palette: Vec<String>,
    pub support_px: usize,
    /// Hidden texels of the layer (fillMask, plus evaluation holdout pixels).
    pub hidden_px: usize,
    pub seeds: usize,
    pub band_seeds: usize,
    pub line_frac: f64,
    pub seed_frac: f64,
    pub local_std_median: f64,
    pub cel_core_frac: f64,
    pub residual_mean_de: f64,
    pub rim: RimReport,
}

/// Inclusive-exclusive pixel box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bbox {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Bbox {
    fn of(w: usize, h: usize, margin: usize, pred: impl Fn(usize) -> bool) -> Option<Bbox> {
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        for y in 0..h {
            for x in 0..w {
                if pred(y * w + x) {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        (x0 != usize::MAX).then(|| Bbox {
            x0: x0.saturating_sub(margin),
            y0: y0.saturating_sub(margin),
            x1: (x1 + 1 + margin).min(w),
            y1: (y1 + 1 + margin).min(h),
        })
    }
    fn bw(&self) -> usize {
        self.x1 - self.x0
    }
    fn bh(&self) -> usize {
        self.y1 - self.y0
    }
    fn local(&self, i: usize, w: usize) -> usize {
        (i / w - self.y0) * self.bw() + (i % w - self.x0)
    }
    fn global(&self, li: usize, w: usize) -> usize {
        (self.y0 + li / self.bw()) * w + self.x0 + li % self.bw()
    }
}

/// CIELAB of every canvas pixel (computed once per synthesis; LUT-accelerated).
pub struct ImageLab {
    pub lab: Vec<[f32; 3]>,
}

impl ImageLab {
    pub fn new(rgba: &[u8], n: usize) -> ImageLab {
        let mut lut = [0f64; 256];
        for (c, v) in lut.iter_mut().enumerate() {
            let s = c as f64 / 255.0;
            *v = if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
        }
        let f = |t: f64| {
            if t > 216.0 / 24389.0 {
                t.cbrt()
            } else {
                (24389.0 / 27.0 * t + 16.0) / 116.0
            }
        };
        let lab = (0..n)
            .map(|i| {
                let (r, g, b) = (lut[rgba[i * 4] as usize], lut[rgba[i * 4 + 1] as usize], lut[rgba[i * 4 + 2] as usize]);
                let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
                let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
                let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
                let (fx, fy, fz) = (f(x / 0.95047), f(y), f(z / 1.08883));
                [(116.0 * fy - 16.0) as f32, (500.0 * (fx - fy)) as f32, (200.0 * (fy - fz)) as f32]
            })
            .collect();
        ImageLab { lab }
    }
}

/// CIELAB L* of an sRGB colour.
pub fn srgb_l(c: [u8; 3]) -> f32 {
    lab_of(c)[0]
}

fn lab_of(c: [u8; 3]) -> [f32; 3] {
    let l = srgb_to_lab(c);
    [l[0] as f32, l[1] as f32, l[2] as f32]
}

fn de00(a: [f32; 3], b: [f32; 3]) -> f64 {
    delta_e2000(
        [a[0] as f64, a[1] as f64, a[2] as f64],
        [b[0] as f64, b[1] as f64, b[2] as f64],
    )
}

fn d2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn rgb_at(rgba: &[u8], i: usize) -> [u8; 3] {
    [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]]
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn neighbors4(li: usize, bw: usize, bh: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (li % bw, li / bw);
    [
        (x > 0).then(|| li - 1),
        (x + 1 < bw).then(|| li + 1),
        (y > 0).then(|| li - bw),
        (y + 1 < bh).then(|| li + bw),
    ]
    .into_iter()
    .flatten()
}

/// 4-connected distance (in a local grid) from `src` cells, explored up to `radius`.
fn local_distance(src: &[bool], bw: usize, bh: usize, radius: usize) -> Vec<u16> {
    let mut dist = vec![u16::MAX; src.len()];
    let mut q = VecDeque::new();
    for (i, &s) in src.iter().enumerate() {
        if s {
            dist[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        let nd = dist[i] + 1;
        if nd as usize > radius {
            continue;
        }
        for j in neighbors4(i, bw, bh) {
            if dist[j] == u16::MAX {
                dist[j] = nd;
                q.push_back(j);
            }
        }
    }
    dist
}

/// Pixels of `m` at least `r` px (4-connected) inside it; the local-box border counts as outside.
fn local_erode(m: &[bool], bw: usize, bh: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return m.to_vec();
    }
    let mut outside: Vec<bool> = m.iter().map(|&v| !v).collect();
    for x in 0..bw {
        outside[x] = true;
        outside[(bh - 1) * bw + x] = true;
    }
    for y in 0..bh {
        outside[y * bw] = true;
        outside[y * bw + bw - 1] = true;
    }
    let d = local_distance(&outside, bw, bh, r);
    // Cells within `r` of the outside (rings 1..=r) are removed.
    m.iter().zip(&d).map(|(&v, &dd)| v && dd == u16::MAX).collect()
}

fn local_dilate(m: &[bool], bw: usize, bh: usize, r: usize) -> Vec<bool> {
    local_distance(m, bw, bh, r).into_iter().map(|d| d != u16::MAX).collect()
}

/// Deterministic k-means (farthest-point initialisation from the sample nearest the mean).
fn kmeans(samples: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    if samples.is_empty() {
        return Vec::new();
    }
    let nf = samples.len() as f64;
    let mut m = [0f64; 3];
    for s in samples {
        for c in 0..3 {
            m[c] += s[c] as f64;
        }
    }
    let mean = [(m[0] / nf) as f32, (m[1] / nf) as f32, (m[2] / nf) as f32];
    let first = (0..samples.len())
        .min_by(|&a, &b| d2(samples[a], mean).total_cmp(&d2(samples[b], mean)))
        .unwrap_or(0);
    let mut centers = vec![samples[first]];
    let mut dmin: Vec<f32> = samples.iter().map(|s| d2(*s, centers[0])).collect();
    while centers.len() < k.min(samples.len()) {
        let (far, &d) = dmin
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((0, &0.0));
        if d <= 1.0 {
            break;
        }
        centers.push(samples[far]);
        for (dm, s) in dmin.iter_mut().zip(samples) {
            *dm = dm.min(d2(*s, samples[far]));
        }
    }
    for _ in 0..16 {
        let mut acc = vec![[0f64; 4]; centers.len()];
        for s in samples {
            let c = nearest(&centers, *s);
            for ch in 0..3 {
                acc[c][ch] += s[ch] as f64;
            }
            acc[c][3] += 1.0;
        }
        for (c, a) in centers.iter_mut().zip(&acc) {
            if a[3] > 0.0 {
                *c = [(a[0] / a[3]) as f32, (a[1] / a[3]) as f32, (a[2] / a[3]) as f32];
            }
        }
    }
    centers
}

fn nearest(centers: &[[f32; 3]], s: [f32; 3]) -> usize {
    (0..centers.len())
        .min_by(|&a, &b| d2(centers[a], s).total_cmp(&d2(centers[b], s)))
        .unwrap_or(0)
}

/// Palette of ≤ `PALETTE_K_MAX` clusters: k-means, merge centres closer than ΔE00
/// `PALETTE_MERGE_DE`, drop clusters below `PALETTE_MIN_SHARE` of the samples.
fn palette(samples: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut centers = kmeans(samples, PALETTE_K_MAX);
    loop {
        // Counts under the current centres.
        let mut cnt = vec![0usize; centers.len()];
        for s in samples {
            cnt[nearest(&centers, *s)] += 1;
        }
        let total = samples.len().max(1) as f64;
        // Closest pair below the merge distance.
        let mut best: Option<(usize, usize, f64)> = None;
        for a in 0..centers.len() {
            for b in a + 1..centers.len() {
                let de = de00(centers[a], centers[b]);
                if de < PALETTE_MERGE_DE && best.is_none_or(|x| de < x.2) {
                    best = Some((a, b, de));
                }
            }
        }
        if let Some((a, b, _)) = best {
            let (wa, wb) = (cnt[a].max(1) as f32, cnt[b].max(1) as f32);
            let mut m = [0f32; 3];
            for c in 0..3 {
                m[c] = (centers[a][c] * wa + centers[b][c] * wb) / (wa + wb);
            }
            centers[a] = m;
            centers.remove(b);
            continue;
        }
        if let Some(small) = (0..centers.len())
            .filter(|&c| (cnt[c] as f64) < PALETTE_MIN_SHARE * total && centers.len() > 1)
            .min_by_key(|&c| cnt[c])
        {
            centers.remove(small);
            continue;
        }
        // Re-centre once on the final assignment.
        let mut acc = vec![[0f64; 4]; centers.len()];
        for s in samples {
            let c = nearest(&centers, *s);
            for ch in 0..3 {
                acc[c][ch] += s[ch] as f64;
            }
            acc[c][3] += 1.0;
        }
        for (c, a) in centers.iter_mut().zip(&acc) {
            if a[3] > 0.0 {
                *c = [(a[0] / a[3]) as f32, (a[1] / a[3]) as f32, (a[2] / a[3]) as f32];
            }
        }
        return centers;
    }
}

fn lab_to_rgb(lab: [f32; 3]) -> [u8; 3] {
    let (l, a, b) = (lab[0] as f64, lab[1] as f64, lab[2] as f64);
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let inv = |t: f64| {
        if t.powi(3) > 216.0 / 24389.0 {
            t.powi(3)
        } else {
            (116.0 * t - 16.0) / (24389.0 / 27.0)
        }
    };
    let (x, y, z) = (inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883);
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let bb = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    let enc = |v: f64| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0).round().clamp(0.0, 255.0) as u8
    };
    [enc(r), enc(g), enc(bb)]
}

/// Multi-channel harmonic field on a cropped coarse grid (same scheme as `templates` colour
/// fields: seed cells pinned to their mean, BFS nearest-seed initialisation, 48 Gauss-Seidel sweeps).
#[derive(Debug, Clone)]
pub struct HGrid {
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    step: usize,
    ch: usize,
    v: Vec<f32>,
}

impl HGrid {
    fn build(bbox: Bbox, step: usize, ch: usize, w: usize, seeds: &[usize], vals: &[f32]) -> HGrid {
        let gw = bbox.bw().div_ceil(step).max(1);
        let gh = bbox.bh().div_ceil(step).max(1);
        let gn = gw * gh;
        let mut sums = vec![0f64; gn * ch];
        let mut cnt = vec![0usize; gn];
        for (s, &i) in seeds.iter().enumerate() {
            let (x, y) = (i % w, i / w);
            if x < bbox.x0 || x >= bbox.x1 || y < bbox.y0 || y >= bbox.y1 {
                continue;
            }
            let gi = ((y - bbox.y0) / step).min(gh - 1) * gw + ((x - bbox.x0) / step).min(gw - 1);
            for c in 0..ch {
                sums[gi * ch + c] += vals[s * ch + c] as f64;
            }
            cnt[gi] += 1;
        }
        let mut v = vec![0f32; gn * ch];
        let mut pinned = vec![false; gn];
        let mut q = VecDeque::new();
        for gi in 0..gn {
            if cnt[gi] > 0 {
                for c in 0..ch {
                    v[gi * ch + c] = (sums[gi * ch + c] / cnt[gi] as f64) as f32;
                }
                pinned[gi] = true;
                q.push_back(gi);
            }
        }
        let mut visited = pinned.clone();
        while let Some(gi) = q.pop_front() {
            for nb in neighbors4(gi, gw, gh) {
                if !visited[nb] {
                    visited[nb] = true;
                    for c in 0..ch {
                        v[nb * ch + c] = v[gi * ch + c];
                    }
                    q.push_back(nb);
                }
            }
        }
        let mut acc = vec![0f32; ch];
        for _ in 0..48 {
            for gi in 0..gn {
                if pinned[gi] {
                    continue;
                }
                acc.iter_mut().for_each(|a| *a = 0.0);
                let mut ws = 0f32;
                for nb in neighbors4(gi, gw, gh) {
                    let wt = if pinned[nb] { 1.5 } else { 1.0 };
                    for c in 0..ch {
                        acc[c] += v[nb * ch + c] * wt;
                    }
                    ws += wt;
                }
                if ws > 0.0 {
                    for c in 0..ch {
                        v[gi * ch + c] = acc[c] / ws;
                    }
                }
            }
        }
        HGrid { x0: bbox.x0, y0: bbox.y0, gw, gh, step, ch, v }
    }

    fn sample(&self, x: usize, y: usize, out: &mut [f32]) {
        let s = self.step as f32;
        let fx = ((x as f32 - self.x0 as f32 + 0.5) / s - 0.5).clamp(0.0, (self.gw - 1) as f32);
        let fy = ((y as f32 - self.y0 as f32 + 0.5) / s - 0.5).clamp(0.0, (self.gh - 1) as f32);
        let (gx0, gy0) = (fx.floor() as usize, fy.floor() as usize);
        let (gx1, gy1) = ((gx0 + 1).min(self.gw - 1), (gy0 + 1).min(self.gh - 1));
        let (tx, ty) = (fx - gx0 as f32, fy - gy0 as f32);
        let ch = self.ch;
        for (c, o) in out.iter_mut().enumerate().take(ch) {
            let v00 = self.v[(gy0 * self.gw + gx0) * ch + c];
            let v10 = self.v[(gy0 * self.gw + gx1) * ch + c];
            let v01 = self.v[(gy1 * self.gw + gx0) * ch + c];
            let v11 = self.v[(gy1 * self.gw + gx1) * ch + c];
            let top = v00 * (1.0 - tx) + v10 * tx;
            let bot = v01 * (1.0 - tx) + v11 * tx;
            *o = top * (1.0 - ty) + bot * ty;
        }
    }
}

/// Inputs of one layer for `analyze_layer`.
pub struct LayerInput<'a> {
    pub rgba: &'a [u8],
    pub lab: &'a ImageLab,
    pub w: usize,
    pub h: usize,
    /// Visible mask used for texturing (evaluation holdout pixels already removed).
    pub vis: &'a [u8],
    /// Hidden texels (fillMask, plus evaluation holdout pixels).
    pub hidden: &'a [u8],
    /// `true` = never a seed (other-class evidence, wearables, holdout pixels).
    pub exclude: Option<&'a [bool]>,
    /// Seeds within this distance of the hidden area drive the statistics.
    pub band_radius: usize,
    /// Role-based type set by the caller: `Semantic` for structured parts (iris, mouth cavity),
    /// `Line` for stroke layers (brows, eyelids). `None` = decided by the seed statistics.
    pub role_kind: Option<RegionKind>,
}

/// Seeds, palette, and region statistics of one layer.
pub struct LayerRegion {
    pub bbox: Option<Bbox>,
    /// Global indices of clean seeds.
    pub seeds: Vec<usize>,
    /// Palette cluster of each seed (`seeds` order).
    labels: Vec<u8>,
    centers: Vec<[f32; 3]>,
    pub report: RegionReport,
}

impl LayerRegion {
    pub fn kind(&self) -> RegionKind {
        self.report.kind
    }
    /// Enough own seeds for an own palette / field: at least `MIN_LAYER_SEEDS` and at least
    /// `OWN_FIELD_MIN_SEED_RATIO` of the hidden texels.
    pub fn has_own_seeds(&self) -> bool {
        self.seeds.len() >= MIN_LAYER_SEEDS
            && self.seeds.len() as f64 >= OWN_FIELD_MIN_SEED_RATIO * self.report.hidden_px as f64
    }
}

/// Classifies one layer from its clean inner seeds (see module docs for every threshold).
pub fn analyze_layer(inp: &LayerInput<'_>) -> LayerRegion {
    let (w, h) = (inp.w, inp.h);
    let margin = SEED_ERODE_PX + LINE_OPEN_PX + 3;
    let empty = |kind: RegionKind| LayerRegion {
        bbox: None,
        seeds: Vec::new(),
        labels: Vec::new(),
        centers: Vec::new(),
        report: RegionReport {
            version: REGION_FILL_VERSION.into(),
            kind,
            snap: false,
            synth: "none".into(),
            field_source: "none".into(),
            palette: Vec::new(),
            support_px: 0,
            hidden_px: inp.hidden.iter().filter(|&&v| v > 0).count(),
            seeds: 0,
            band_seeds: 0,
            line_frac: 0.0,
            seed_frac: 0.0,
            local_std_median: 0.0,
            cel_core_frac: 0.0,
            residual_mean_de: 0.0,
            rim: RimReport::default(),
        },
    };
    let Some(bbox) = Bbox::of(w, h, margin, |i| inp.vis[i] > 0 || inp.hidden[i] > 0) else {
        return empty(inp.role_kind.unwrap_or(RegionKind::FlatCel));
    };
    let (bw, bh) = (bbox.bw(), bbox.bh());
    let bn = bw * bh;
    let g = |li: usize| bbox.global(li, w);
    let support: Vec<bool> = (0..bn).map(|li| inp.vis[g(li)] >= SEED_ALPHA_MIN).collect();
    let support_px = support.iter().filter(|&&v| v).count();
    if support_px == 0 {
        let mut r = empty(inp.role_kind.unwrap_or(RegionKind::FlatCel));
        r.bbox = Some(bbox);
        return r;
    }
    // Line art: dark (relative to the layer median L*) and thin (removed by an opening).
    let mut ls: Vec<f32> = (0..bn).filter(|&li| support[li]).map(|li| inp.lab.lab[g(li)][0]).collect();
    ls.sort_by(|a, b| a.total_cmp(b));
    let med_l = ls[ls.len() / 2];
    let dark: Vec<bool> = (0..bn)
        .map(|li| support[li] && inp.lab.lab[g(li)][0] < med_l - LINE_DROP_L)
        .collect();
    let opened = local_dilate(&local_erode(&dark, bw, bh, LINE_OPEN_PX), bw, bh, LINE_OPEN_PX);
    let line: Vec<bool> = (0..bn).map(|li| dark[li] && !opened[li]).collect();
    let line_guard = local_dilate(&line, bw, bh, 1);
    // Seeds.
    let visible_any: Vec<bool> = (0..bn).map(|li| inp.vis[g(li)] > 0).collect();
    let allowed = |li: usize| {
        support[li] && !line_guard[li] && !inp.exclude.is_some_and(|e| e[g(li)])
    };
    // Thinness is judged at the full erosion even when the seed set below relaxes it.
    let full_eroded = local_erode(&visible_any, bw, bh, SEED_ERODE_PX);
    let thick_px = (0..bn).filter(|&li| full_eroded[li] && support[li]).count();
    let mut seeds_local: Vec<usize> = Vec::new();
    for er in (0..=SEED_ERODE_PX).rev() {
        let eroded = local_erode(&visible_any, bw, bh, er);
        seeds_local = (0..bn).filter(|&li| eroded[li] && allowed(li)).collect();
        if seeds_local.len() >= MIN_LAYER_SEEDS {
            break;
        }
    }
    // Band: seeds within `band_radius` of hidden texels.
    let hidden_local: Vec<bool> = (0..bn).map(|li| inp.hidden[g(li)] > 0).collect();
    let has_hidden = hidden_local.iter().any(|&v| v);
    let near_hidden = if has_hidden {
        Some(local_distance(&hidden_local, bw, bh, inp.band_radius.max(1)))
    } else {
        None
    };
    let in_band = |li: usize| near_hidden.as_ref().is_none_or(|d| d[li] != u16::MAX);
    let mut band: Vec<usize> = seeds_local.iter().copied().filter(|&li| in_band(li)).collect();
    if band.len() < MIN_BAND_SEEDS {
        band = seeds_local.clone();
    }
    let band_support: Vec<usize> = (0..bn)
        .filter(|&li| support[li] && (band.len() == seeds_local.len() || in_band(li)))
        .collect();
    let line_frac = if band_support.is_empty() {
        0.0
    } else {
        band_support.iter().filter(|&&li| line[li]).count() as f64 / band_support.len() as f64
    };
    let seed_frac = thick_px as f64 / support_px as f64;
    // Palette on band seeds (deterministic stride subsample ≤ 16384).
    let stride = band.len().div_ceil(16384).max(1);
    let samples: Vec<[f32; 3]> = band.iter().step_by(stride).map(|&li| inp.lab.lab[g(li)]).collect();
    let centers = palette(&samples);
    let (mut core, mut res_sum) = (0usize, 0f64);
    for s in &samples {
        let mut ds: Vec<f32> = centers.iter().map(|c| d2(*c, *s).sqrt()).collect();
        ds.sort_by(|a, b| a.total_cmp(b));
        let rho = if ds.len() >= 2 && ds[0] + ds[1] > 1e-6 {
            (ds[0] / (ds[0] + ds[1])) as f64
        } else {
            0.0
        };
        if rho < CEL_RHO {
            core += 1;
        }
        res_sum += de00(*s, centers[nearest(&centers, *s)]);
    }
    let cel_core_frac = core as f64 / samples.len().max(1) as f64;
    let residual_mean_de = res_sum / samples.len().max(1) as f64;
    // Median 5×5 local L* σ over band seeds (seed pixels only, ≥ 9 in the window).
    let mut is_seed = vec![false; bn];
    for &li in &seeds_local {
        is_seed[li] = true;
    }
    let mut sigmas: Vec<f64> = Vec::new();
    for &li in band.iter().step_by(stride) {
        let (x, y) = ((li % bw) as i64, (li / bw) as i64);
        let (mut s1, mut s2, mut c) = (0f64, 0f64, 0usize);
        for dy in -2..=2i64 {
            for dx in -2..=2i64 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= bw as i64 || ny >= bh as i64 {
                    continue;
                }
                let nl = ny as usize * bw + nx as usize;
                if !is_seed[nl] {
                    continue;
                }
                let l = inp.lab.lab[g(nl)][0] as f64;
                s1 += l;
                s2 += l * l;
                c += 1;
            }
        }
        if c >= 9 {
            let m = s1 / c as f64;
            sigmas.push((s2 / c as f64 - m * m).max(0.0).sqrt());
        }
    }
    sigmas.sort_by(|a, b| a.total_cmp(b));
    let local_std_median = sigmas.get(sigmas.len() / 2).copied().unwrap_or(0.0);
    let kind = if let Some(k) = inp.role_kind {
        k
    } else if line_frac >= CLASS_LINE_FRAC || seed_frac < CLASS_THIN_SEED_FRAC {
        RegionKind::Line
    } else if local_std_median >= CLASS_TEXTURE_STD {
        RegionKind::Texture
    } else {
        RegionKind::FlatCel
    };
    let snap = kind == RegionKind::FlatCel && centers.len() >= 2 && cel_core_frac >= CEL_CORE_FRAC;
    let seeds: Vec<usize> = seeds_local.iter().map(|&li| g(li)).collect();
    let labels: Vec<u8> = seeds
        .iter()
        .map(|&i| nearest(&centers, inp.lab.lab[i]) as u8)
        .collect();
    LayerRegion {
        bbox: Some(bbox),
        report: RegionReport {
            version: REGION_FILL_VERSION.into(),
            kind,
            snap,
            synth: "none".into(),
            field_source: "layer".into(),
            palette: centers.iter().map(|c| hex(lab_to_rgb(*c))).collect(),
            support_px,
            hidden_px: inp.hidden.iter().filter(|&&v| v > 0).count(),
            seeds: seeds.len(),
            band_seeds: band.len(),
            line_frac,
            seed_frac,
            local_std_median,
            cel_core_frac,
            residual_mean_de,
            rim: RimReport::default(),
        },
        seeds,
        labels,
        centers,
    }
}

/// Hidden-texel colour source built from a layer's own seeds.
pub struct RegionFiller {
    snap: bool,
    all: HGrid,
    indicator: Option<HGrid>,
    per_cluster: Vec<HGrid>,
}

impl RegionFiller {
    /// `None` without enough own seeds (caller keeps its class field).
    pub fn new(region: &LayerRegion, rgba: &[u8], w: usize, h: usize) -> Option<RegionFiller> {
        let bbox = region.bbox?;
        if !region.has_own_seeds() {
            return None;
        }
        let step = (w.max(h) / 256).clamp(2, 4);
        let rgb_vals = |idx: &[usize]| -> Vec<f32> {
            idx.iter()
                .flat_map(|&i| {
                    let c = rgb_at(rgba, i);
                    [c[0] as f32, c[1] as f32, c[2] as f32]
                })
                .collect()
        };
        let all = HGrid::build(bbox, step, 3, w, &region.seeds, &rgb_vals(&region.seeds));
        let k = region.centers.len();
        let (indicator, per_cluster) = if region.report.snap && k >= 2 {
            let ind_vals: Vec<f32> = region
                .labels
                .iter()
                .flat_map(|&l| (0..k).map(move |c| if c == l as usize { 1.0 } else { 0.0 }))
                .collect();
            let ind = HGrid::build(bbox, step, k, w, &region.seeds, &ind_vals);
            let per: Vec<HGrid> = (0..k)
                .map(|c| {
                    let idx: Vec<usize> = region
                        .seeds
                        .iter()
                        .zip(&region.labels)
                        .filter(|(_, &l)| l as usize == c)
                        .map(|(&i, _)| i)
                        .collect();
                    if idx.is_empty() {
                        all.clone()
                    } else {
                        HGrid::build(bbox, step, 3, w, &idx, &rgb_vals(&idx))
                    }
                })
                .collect();
            (Some(ind), per)
        } else {
            (None, Vec::new())
        };
        let _ = h;
        Some(RegionFiller { snap: indicator.is_some(), all, indicator, per_cluster })
    }

    pub fn snapped(&self) -> bool {
        self.snap
    }

    /// Hidden colour at (x, y): palette level set + cluster-local field when snapping, else the
    /// local harmonic field of all seeds.
    pub fn color_at(&self, x: usize, y: usize) -> [u8; 3] {
        let mut c = [0f32; 3];
        if let Some(ind) = &self.indicator {
            let mut v = [0f32; PALETTE_K_MAX];
            ind.sample(x, y, &mut v[..ind.ch]);
            let best = (0..ind.ch).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap_or(0);
            self.per_cluster[best].sample(x, y, &mut c);
        } else {
            self.all.sample(x, y, &mut c);
        }
        [
            c[0].round().clamp(0.0, 255.0) as u8,
            c[1].round().clamp(0.0, 255.0) as u8,
            c[2].round().clamp(0.0, 255.0) as u8,
        ]
    }
}

/// Opaque background colour when the input has one (≥ 1 % of pixels are outside the foreground
/// yet fully opaque); `None` for transparent inputs (white matte is assumed for halos).
pub fn opaque_background_rgb(rgba: &[u8], fg_alpha: &[u8]) -> Option<[u8; 3]> {
    let n = fg_alpha.len();
    let mut ch: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..n {
        if fg_alpha[i] == 0 && rgba[i * 4 + 3] == 255 {
            for c in 0..3 {
                ch[c].push(rgba[i * 4 + c]);
            }
        }
    }
    if ch[0].len() * 100 < n.max(1) {
        return None;
    }
    let mut out = [0u8; 3];
    for c in 0..3 {
        ch[c].sort_unstable();
        out[c] = ch[c][ch[c].len() / 2];
    }
    Some(out)
}

fn mixes_toward(c: [u8; 3], f: [u8; 3], nb: [u8; 3]) -> bool {
    let cf = [c[0] as f32 - f[0] as f32, c[1] as f32 - f[1] as f32, c[2] as f32 - f[2] as f32];
    let d = [nb[0] as f32 - f[0] as f32, nb[1] as f32 - f[1] as f32, nb[2] as f32 - f[2] as f32];
    let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if dd < MIX_MIN_SEP * MIX_MIN_SEP {
        return false;
    }
    let t = (cf[0] * d[0] + cf[1] * d[1] + cf[2] * d[2]) / dd;
    if !(MIX_T_MIN..=1.15).contains(&t) {
        return false;
    }
    let perp = ((cf[0] - t * d[0]).powi(2) + (cf[1] - t * d[1]).powi(2) + (cf[2] - t * d[2]).powi(2)).sqrt();
    perp <= MIX_PERP_BASE + MIX_PERP_REL * dd.sqrt()
}

/// Canvas-wide context shared by every layer's rim pass.
pub struct RimContext<'a> {
    pub rgba: &'a [u8],
    pub lab: &'a ImageLab,
    pub w: usize,
    pub h: usize,
    /// Rest-pose visible owner layer index of each pixel (-1 = background).
    pub owner: &'a [i32],
    /// Opaque background colour (`opaque_background_rgb`); `None` → white matte.
    pub background: Option<[u8; 3]>,
    /// Manual fill (no partition known): an unowned outside pixel with alpha > 0 is mixed against
    /// its own colour instead of the matte (its owner is unknown, so dark lines are kept).
    pub use_outside_color: bool,
}

/// Replaces contaminated outer-rim texels of layer `layer` (see module docs for the criteria).
/// Only texture colours change; `vis` is read, never written.
///
/// Upper-side replacements (the upper layer's outline half and upper-side mixtures) are seam
/// pixels the caller hands to the upper layer as overlay `fillMask` texels with their original
/// colour, so the rest pose is unchanged. `follow = None` decides them here and returns them as
/// `(pixel, upper layer)`; `follow = Some(plan)` applies a plan decided earlier on the product
/// partition (`plan[i]` = upper layer, `-1` = keep) so an evaluation holdout cannot change which
/// pixels move (and thus the overlay `fillMask`).
pub fn decontaminate_rim(
    tex: &mut [u8],
    vis: &[u8],
    layer: usize,
    region: &LayerRegion,
    ctx: &RimContext<'_>,
    follow: Option<&[i32]>,
) -> (RimReport, Vec<(usize, usize)>) {
    let mut rep = RimReport::default();
    let mut moved: Vec<(usize, usize)> = Vec::new();
    let (w, h) = (ctx.w, ctx.h);
    let Some(bbox) = region.bbox else {
        return (rep, moved);
    };
    if region.seeds.is_empty() {
        return (rep, moved);
    }
    let (bw, bh) = (bbox.bw(), bbox.bh());
    let bn = bw * bh;
    let g = |li: usize| bbox.global(li, w);
    let mut is_seed = vec![false; bn];
    for &i in &region.seeds {
        is_seed[bbox.local(i, w)] = true;
    }
    // Nearest clean seed colour through the visible support.
    let mut src = vec![u32::MAX; bn];
    let mut q = VecDeque::new();
    for &i in &region.seeds {
        let li = bbox.local(i, w);
        src[li] = i as u32;
        q.push_back(li);
    }
    while let Some(li) = q.pop_front() {
        for nl in neighbors4(li, bw, bh) {
            if src[nl] == u32::MAX && vis[g(nl)] > 0 {
                src[nl] = src[li];
                q.push_back(nl);
            }
        }
    }
    // Rim depth from the non-visible side.
    let outside: Vec<bool> = (0..bn).map(|li| vis[g(li)] == 0).collect();
    let depth = local_distance(&outside, bw, bh, RIM_PX as usize);
    let white = [255u8, 255, 255];
    for li in 0..bn {
        if outside[li] || depth[li] == u16::MAX || src[li] == u32::MAX {
            continue;
        }
        let i = g(li);
        rep.rim_px += 1;
        // Nearest non-visible pixel in the 5×5 neighbourhood (Chebyshev ring, then Manhattan).
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        let mut best: Option<(i64, i64, usize)> = None;
        for dy in -2..=2i64 {
            for dx in -2..=2i64 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if vis[j] > 0 {
                    continue;
                }
                let key = (dx.abs().max(dy.abs()), dx.abs() + dy.abs());
                if best.is_none_or(|b| key < (b.0, b.1)) {
                    best = Some((key.0, key.1, j));
                }
            }
        }
        let Some((_, _, j)) = best else {
            continue;
        };
        let oj = ctx.owner[j];
        if oj == layer as i32 {
            // Own evaluation-holdout band: not a seam.
            continue;
        }
        let k = i * 4;
        let c = [tex[k], tex[k + 1], tex[k + 2]];
        let lc = lab_of(c);
        // F: the clean seed within `RIM_F_WINDOW` px whose colour is closest to C (the clean
        // version of this pixel's own cel region); the BFS-nearest seed only when none is local.
        // (BFS-nearest alone can pick a neighbouring cel region, e.g. a navy collar for a white
        // shirt rim pixel, and would paint it dark.)
        let near_i = src[li] as usize;
        let mut fi = near_i;
        let mut best_d = f32::MAX;
        for dy in -RIM_F_WINDOW..=RIM_F_WINDOW {
            for dx in -RIM_F_WINDOW..=RIM_F_WINDOW {
                let (nx, ny) = (x + dx, y + dy);
                if nx < bbox.x0 as i64 || ny < bbox.y0 as i64 || nx >= bbox.x1 as i64 || ny >= bbox.y1 as i64 {
                    continue;
                }
                let kk = ny as usize * w + nx as usize;
                if !is_seed[bbox.local(kk, w)] {
                    continue;
                }
                let dd = d2(ctx.lab.lab[kk], lc);
                if dd < best_d {
                    best_d = dd;
                    fi = kk;
                }
            }
        }
        if best_d == f32::MAX {
            // No clean seed within the window: the clean colour here is unknown (e.g. a face
            // rim inside the feature guard zone). Leave the pixel as drawn.
            rep.no_local_seed += 1;
            continue;
        }
        let f = rgb_at(ctx.rgba, fi);
        let lf = ctx.lab.lab[fi];
        if de00(lc, lf) < MIX_MIN_DE {
            continue;
        }
        let dark = lc[0] < lf[0] - LINE_DROP_L;
        let replace = if oj < 0 {
            // Background silhouette: dark = own outline, keep; mixture toward the matte = halo.
            let matte = if ctx.use_outside_color && ctx.rgba[j * 4 + 3] > 0 {
                rgb_at(ctx.rgba, j)
            } else {
                ctx.background.unwrap_or(white)
            };
            let replace = !dark && mixes_toward(c, f, matte);
            if replace {
                rep.background_halo += 1;
            }
            replace
        } else if oj > layer as i32 {
            // Line art is judged against (and replaced by) the spatially nearest clean seed — the
            // region right behind the line — not the colour-closest one (that would be the
            // darkest local shade and leave a smudge).
            let lf_near = ctx.lab.lab[near_i];
            if lc[0] < lf_near[0] - RIM_LINE_DROP_L {
                let mut owned = false;
                'search: for dy in -RIM_OWNER_SEARCH_PX..=RIM_OWNER_SEARCH_PX {
                    for dx in -RIM_OWNER_SEARCH_PX..=RIM_OWNER_SEARCH_PX {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let kk = ny as usize * w + nx as usize;
                        if ctx.owner[kk] > layer as i32
                            && ctx.lab.lab[kk][0] <= lc[0] + RIM_OWNER_LINE_SLACK_L
                        {
                            owned = true;
                            break 'search;
                        }
                    }
                }
                let take = match follow {
                    Some(plan) => plan[i] >= 0,
                    None => owned,
                };
                if take {
                    rep.upper_line += 1;
                    moved.push((i, follow.map_or(oj, |plan| plan[i]) as usize));
                    let fn_ = rgb_at(ctx.rgba, near_i);
                    tex[k] = fn_[0];
                    tex[k + 1] = fn_[1];
                    tex[k + 2] = fn_[2];
                } else {
                    rep.kept_unowned_line += 1;
                }
                false
            } else {
                let take = match follow {
                    Some(plan) => plan[i] >= 0,
                    None => mixes_toward(c, f, rgb_at(ctx.rgba, j)),
                };
                if take {
                    rep.upper_mix += 1;
                    moved.push((i, follow.map_or(oj, |plan| plan[i]) as usize));
                }
                take
            }
        } else {
            // This layer owns the seam outline: a rim pixel darker than F may be that outline
            // (indistinguishable from a mixture toward a darker lower layer), so only lightening /
            // hue mixtures toward the lower colour are replaced.
            let replace = lc[0] >= lf[0] - LOWER_MIX_MAX_DARKEN_L && mixes_toward(c, f, rgb_at(ctx.rgba, j));
            if replace {
                rep.lower_mix += 1;
            }
            replace
        };
        if replace {
            tex[k] = f[0];
            tex[k + 1] = f[1];
            tex[k + 2] = f[2];
        }
    }
    (rep, moved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canvas(w: usize, h: usize, f: impl Fn(usize, usize) -> [u8; 3]) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let c = f(x, y);
                let k = (y * w + x) * 4;
                v[k..k + 3].copy_from_slice(&c);
                v[k + 3] = 255;
            }
        }
        v
    }

    fn analyze(rgba: &[u8], vis: &[u8], hidden: &[u8], w: usize, h: usize) -> LayerRegion {
        let lab = ImageLab::new(rgba, w * h);
        analyze_layer(&LayerInput {
            rgba,
            lab: &lab,
            w,
            h,
            vis,
            hidden,
            exclude: None,
            band_radius: 24,
            role_kind: None,
        })
    }

    const BASE: [u8; 3] = [236, 200, 180];
    const SHADOW: [u8; 3] = [200, 150, 140];

    /// Two-tone cel layer: shadow below a boundary; the right third is hidden.
    #[allow(clippy::type_complexity)]
    fn cel_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, usize, usize, impl Fn(usize, usize) -> [u8; 3]) {
        let (w, h) = (96usize, 64usize);
        let truth = move |_x: usize, y: usize| if y > 30 { SHADOW } else { BASE };
        let rgba = canvas(w, h, truth);
        let vis: Vec<u8> = (0..w * h).map(|i| if i % w < 64 { 255 } else { 0 }).collect();
        let hidden: Vec<u8> = (0..w * h).map(|i| if i % w >= 64 { 255 } else { 0 }).collect();
        (rgba, vis, hidden, w, h, truth)
    }

    #[test]
    fn cel_layer_snaps_and_extends_the_shadow_boundary_as_a_level_set() {
        let (rgba, vis, hidden, w, h, truth) = cel_fixture();
        let reg = analyze(&rgba, &vis, &hidden, w, h);
        assert_eq!(reg.kind(), RegionKind::FlatCel);
        assert!(reg.report.snap, "two tight clusters must snap: {:?}", reg.report);
        assert_eq!(reg.report.palette.len(), 2);
        let filler = RegionFiller::new(&reg, &rgba, w, h).unwrap();
        // Compare with a plain harmonic field (snap off) on the hidden texels.
        let mut plain = RegionFiller::new(&reg, &rgba, w, h).unwrap();
        plain.indicator = None;
        let lab = |c: [u8; 3]| lab_of(c);
        let (mut de_snap, mut de_plain, mut off_palette, mut cnt) = (0.0, 0.0, 0usize, 0usize);
        for y in 4..h - 4 {
            for x in 64..w - 4 {
                let t = truth(x, y);
                let s = filler.color_at(x, y);
                let p = plain.color_at(x, y);
                de_snap += de00(lab(s), lab(t));
                de_plain += de00(lab(p), lab(t));
                if de00(lab(s), lab(BASE)) > 2.0 && de00(lab(s), lab(SHADOW)) > 2.0 {
                    off_palette += 1;
                }
                cnt += 1;
            }
        }
        let (ms, mp) = (de_snap / cnt as f64, de_plain / cnt as f64);
        assert!(ms < mp, "snap {ms:.2} must beat plain harmonic {mp:.2}");
        assert!(ms < 0.5, "level-set extension should follow the boundary (mean ΔE {ms:.2})");
        assert_eq!(off_palette, 0, "snapped colours stay on the palette");
        let plain_off = (4..h - 4)
            .flat_map(|y| (64..w - 4).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let p = plain.color_at(x, y);
                de00(lab(p), lab(BASE)) > 2.0 && de00(lab(p), lab(SHADOW)) > 2.0
            })
            .count();
        assert!(plain_off > 0, "the plain harmonic field blurs the shadow boundary into a gradient");
    }

    #[test]
    fn gradient_layer_turns_snap_off() {
        let (w, h) = (96usize, 64usize);
        let rgba = canvas(w, h, |_, y| {
            let t = y as f64 / (h - 1) as f64;
            [
                (236.0 - 50.0 * t) as u8,
                (200.0 - 60.0 * t) as u8,
                (180.0 - 50.0 * t) as u8,
            ]
        });
        let vis: Vec<u8> = (0..w * h).map(|i| if i % w < 64 { 255 } else { 0 }).collect();
        let hidden: Vec<u8> = vis.iter().map(|&v| 255 - v).collect();
        let reg = analyze(&rgba, &vis, &hidden, w, h);
        assert_eq!(reg.kind(), RegionKind::FlatCel);
        assert!(!reg.report.snap, "a ramp is not cel: {:?}", reg.report);
        assert!(reg.report.cel_core_frac < CEL_CORE_FRAC);
    }

    #[test]
    fn noisy_layer_is_texture_and_stroke_layer_is_line() {
        let (w, h) = (96usize, 64usize);
        let mut s = 12345u32;
        let noise: Vec<u8> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) % 40) as u8
            })
            .collect();
        let rgba = canvas(w, h, |x, y| {
            let n = noise[y * w + x];
            [150 + n, 120 + n, 100 + n]
        });
        let vis: Vec<u8> = (0..w * h).map(|i| if i % w < 64 { 255 } else { 0 }).collect();
        let hidden: Vec<u8> = vis.iter().map(|&v| 255 - v).collect();
        assert_eq!(analyze(&rgba, &vis, &hidden, w, h).kind(), RegionKind::Texture);

        // 2px dark strokes every 5 rows: 40 % of the support is thin line art.
        let rgba = canvas(w, h, |_, y| if y % 5 < 2 { [40, 30, 30] } else { [230, 210, 200] });
        let reg = analyze(&rgba, &vis, &hidden, w, h);
        assert_eq!(reg.kind(), RegionKind::Line, "{:?}", reg.report);
        // A stroke-thin layer (2px strokes only) is `line` too.
        let thin: Vec<u8> = (0..w * h).map(|i| if i % w < 64 && (i / w) % 8 < 2 { 255 } else { 0 }).collect();
        let reg = analyze(&rgba, &thin, &hidden, w, h);
        assert_eq!(reg.kind(), RegionKind::Line, "{:?}", reg.report);
    }

    #[test]
    fn wide_dark_regions_are_seeds_not_line_art() {
        // Navy collar band (24px) on white: dark but wide → stays a seed, palette has both colours.
        let (w, h) = (96usize, 64usize);
        let rgba = canvas(w, h, |_, y| if (20..44).contains(&y) { [30, 40, 90] } else { [245, 245, 248] });
        let vis: Vec<u8> = (0..w * h).map(|i| if i % w < 64 { 255 } else { 0 }).collect();
        let hidden: Vec<u8> = vis.iter().map(|&v| 255 - v).collect();
        let reg = analyze(&rgba, &vis, &hidden, w, h);
        assert_eq!(reg.kind(), RegionKind::FlatCel);
        assert!(reg.report.line_frac < 0.01, "{:?}", reg.report);
        assert!(reg.report.snap);
        let filler = RegionFiller::new(&reg, &rgba, w, h).unwrap();
        assert_eq!(filler.color_at(80, 32), [30, 40, 90], "collar continues under the hidden area");
        assert_eq!(filler.color_at(80, 8), [245, 245, 248]);
    }

    #[test]
    fn rim_halo_and_upper_line_are_cleaned_but_own_outline_is_kept() {
        // Layer 0 (lower, skin) on the left half, layer 1 (upper, hair) on the right half,
        // background at the top rows. Contamination is painted into layer 0's rim.
        let (w, h) = (48usize, 32usize);
        let mut rgba = canvas(w, h, |x, _| if x < 24 { BASE } else { [90, 60, 50] });
        let mut owner = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                owner[i] = if y < 4 { -1 } else if x < 24 { 0 } else { 1 };
            }
        }
        let set = |rgba: &mut Vec<u8>, x: usize, y: usize, c: [u8; 3]| {
            let k = (y * w + x) * 4;
            rgba[k..k + 3].copy_from_slice(&c);
        };
        // White halo row next to the background (y = 4), skin's own dark outline at (10, 4).
        for x in 0..20 {
            set(&mut rgba, x, 4, [250, 240, 236]);
        }
        set(&mut rgba, 10, 4, [40, 25, 25]);
        // Upper layer's outline: dark at x = 23 (lower rim) and x = 24 (upper keeps it) for y ≥ 12;
        // for 6 ≤ y < 12 the line is only in the lower layer (unowned → kept).
        for y in 6..h {
            set(&mut rgba, 23, y, [35, 25, 25]);
            if y >= 12 {
                set(&mut rgba, 24, y, [35, 25, 25]);
            }
        }
        for i in 0..w * h {
            if owner[i] < 0 {
                rgba[i * 4..i * 4 + 4].copy_from_slice(&[0, 0, 0, 0]);
            }
        }
        let vis0: Vec<u8> = owner.iter().map(|&o| if o == 0 { 255 } else { 0 }).collect();
        let hidden0: Vec<u8> = owner.iter().map(|&o| if o == 1 { 255 } else { 0 }).collect();
        let lab = ImageLab::new(&rgba, w * h);
        let reg = analyze_layer(&LayerInput {
            rgba: &rgba,
            lab: &lab,
            w,
            h,
            vis: &vis0,
            hidden: &hidden0,
            exclude: None,
            band_radius: 24,
            role_kind: None,
        });
        let mut tex = rgba.clone();
        let ctx = RimContext { rgba: &rgba, lab: &lab, w, h, owner: &owner, background: None, use_outside_color: false };
        let (rep, moved) = decontaminate_rim(&mut tex, &vis0, 0, &reg, &ctx, None);
        // Every replaced upper-side pixel is handed to the upper layer (layer 1) for overlay.
        assert_eq!(moved.len(), rep.upper_line + rep.upper_mix);
        assert!(moved.iter().all(|&(_, u)| u == 1));
        // Following the same plan reproduces the same texture.
        let mut plan = vec![-1i32; w * h];
        for &(i, u) in &moved {
            plan[i] = u as i32;
        }
        let mut tex2 = rgba.clone();
        let (rep2, moved2) = decontaminate_rim(&mut tex2, &vis0, 0, &reg, &ctx, Some(&plan));
        assert_eq!(tex2, tex);
        assert_eq!(moved2, moved);
        assert_eq!(rep2, rep);
        let at = |x: usize, y: usize| [tex[(y * w + x) * 4], tex[(y * w + x) * 4 + 1], tex[(y * w + x) * 4 + 2]];
        assert_eq!(at(3, 4), BASE, "white halo at the silhouette → clean interior colour");
        assert_eq!(at(10, 4), [40, 25, 25], "own outline at the silhouette is kept");
        assert_eq!(at(23, 20), BASE, "upper layer's outline in the lower rim is removed");
        assert_eq!(at(23, 8), [35, 25, 25], "line the upper layer does not keep is not erased");
        assert!(rep.background_halo >= 15 && rep.upper_line >= 18 && rep.kept_unowned_line >= 3, "{rep:?}");
        // Masks are never touched: only `tex` changed.
        assert_eq!(vis0.iter().filter(|&&v| v > 0).count(), owner.iter().filter(|&&o| o == 0).count());
    }

    #[test]
    fn opaque_background_is_detected_only_for_opaque_inputs() {
        let (w, h) = (20usize, 20usize);
        let rgba = canvas(w, h, |_, _| [150, 130, 160]);
        let fg: Vec<u8> = (0..w * h).map(|i| if i % w < 10 { 255 } else { 0 }).collect();
        assert_eq!(opaque_background_rgb(&rgba, &fg), Some([150, 130, 160]));
        let mut t = rgba.clone();
        for i in 0..w * h {
            if fg[i] == 0 {
                t[i * 4 + 3] = 0;
            }
        }
        assert_eq!(opaque_background_rgb(&t, &fg), None);
    }
}
