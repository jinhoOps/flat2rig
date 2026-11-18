//! T15 — line-art / colour separation and structure extension into exposure bands.
//!
//! The T9 colour field (`region_fill`) fills hidden texels with line-free colour: a harmonic or
//! palette field cannot continue a face contour, a neck side line or a hair strand. This module
//! treats every part texture as two layers:
//!
//! - **colour layer** = the texture as synthesized by T9 (visible pixels = the artwork after the T9
//!   rim decontamination; hidden texels = the T9 palette / local / class field, whose seeds already
//!   exclude line art);
//! - **line layer** = dark thin strokes. On the visible side it is *detected* from the layer's texture;
//!   on the hidden side it is *synthesized* here from the same part's visible stroke statistics, then
//!   merged over the colour layer (`tex = α·line + (1 − α)·colour`).
//!
//! Visible pixels are never changed (the static artwork and the visible `mask` partition stay as
//! drawn), and `mask` / `fillMask` are never changed (hidden areas are not shrunk or grown here).
//!
//! **Separation rule** (thresholds shared with T9 seeds and the T14 stroke metric): an evidence pixel
//! (visible alpha ≥ `EVIDENCE_ALPHA_MIN`, not held out) is *strong* when its L* is at least
//! `region_fill::LINE_DROP_L` (25) below the part's visible median L*, *weak* when at least
//! `25 − LINE_WEAK_SLACK_L` (15) below; weak pixels 8-connected to strong ones join them
//! (hysteresis: an anti-aliased 1px stroke is not cut into fragments). The union is line art where it
//! disappears under a morphological opening with a `(2·LINE_OPEN_PX + 1)²` square (stroke thinner
//! than ≈5px; wide dark areas are cel shading = colour layer). Evidence is read from the T9 texture,
//! so seam residue that T9 handed to the upper layer is not taken as this layer's stroke.
//!
//! **Where new lines may be drawn**: only on hidden texels (`fillMask`, outside the visible mask,
//! not seam overlay texels) — or evaluation holdout pixels — whose Euclidean distance from the
//! part's visible mask is at most the T13 *maximum exposure depth* of the seam with the layer that
//! covers the texel at rest (`SeamRequirement::max_exposure_depth_px`, sampled from `deform` over the
//! shared pose envelope) plus `DEPTH_SLACK_PX`. Texels covered by an eye / brow / mouth layer are left
//! to T10 (`deform_eye`, the lash-line extension 5c and the brow / upper-lid stroke block 5b). This is
//! a limit on line *generation* only; no hidden texel is removed or recoloured beyond it.
//!
//! **Stroke extension** (items 2 and 4): an 8-connected line component that touches hidden texels
//! is a stroke candidate. Not ending strokes (no confidence decision, counted separately):
//! components with ≥ `SEAM_OUTLINE_FRAC` of their pixels within `SEAM_OUTLINE_PX` of an upper layer's
//! visible pixel (the upper outline / seam residue: extending it would double that outline),
//! components running along the hidden boundary (entry angle above `ENTRY_MAX_DEG`), and components
//! hugging an outlined silhouette (continued as a contour, item 3). The rest are fitted in a window of
//! `clamp(FIT_WINDOW_W · width, FIT_WINDOW_MIN_PX, FIT_WINDOW_MAX_PX)` px behind the contact:
//! principal axis, 1px slab centroids, least-squares quadratic centreline. Confidence gates (skip →
//! recorded as a `semantic` stroke flag): visible length `< max(STROKE_MIN_LEN_PX, STROKE_MIN_LEN_W ·
//! width)` (`short`; a hair stub instead takes the layer's stroke direction field when at least
//! `FIELD_MIN_STROKES` fitted strokes lie within 2σ with resultant ≥ `FIELD_MIN_COHERENCE`); RMS
//! slab-centroid residual `> max(FIT_RESID_MAX_PX, FIT_RESID_MAX_W · width)` (`fitResidual`); meeting
//! another stroke before `CROSS_MIN_EXT_PX` of extension (`crossing`). Otherwise the stroke is marched
//! along the fitted curve (turn bounded to `MAX_TURN_DEG`; hair layers blend in the same layer's
//! stroke direction field, Gaussian σ = `clamp(0.02·hh, 6, 24)` px) until it leaves the allowed texels,
//! reaches the exposure depth (tapered over the part's measured end taper), reconnects to visible
//! line art, meets another stroke (stops at the junction: it never crosses), or would run beside an
//! outline (double-line guard: within `2·width + 2` px of its own visible edge turning > 60° from the
//! seam normal, or within `2·width + 3` px of an upper layer's visible outline and within 60° of
//! parallel to it). Every drawn texel is then re-checked: texels within `2·width + 2` px of the own
//! visible edge and within 30° of parallel to it, or (strokes) within `2·width + 3` px of an upper
//! outline and within 60° of parallel / (contours) within `2·width + 2` px and 30°, are dropped; a
//! stroke left without texels counts as `seamParallel`. Silhouette-hugging components are always
//! outlines (contour class), never free strokes. Width = the stroke's own slab width at the contact, clamped to `[0.5, 1.25]×` the
//! part's median stroke width; colour = median of the darker half of the stroke's fit-window pixels.
//!
//! **Hidden contours** (item 3): where the hidden area extends the part silhouette, the boundary of
//! the effective mask (`mask ∪ fillMask`, the template SDF boundary the mesh is built from) inside
//! the allowed texels gets an outline when the part's visible silhouette is outlined (coverage
//! ≥ `OUTLINE_MIN_COVERAGE`). It is propagated along the silhouette ring only from the layer's own
//! outlined visible silhouette pixels (continuity at the junction), stops where the ring turns more
//! than `CONTOUR_MAX_TURN_DEG` from the carried outline tangent (an underlap edge is not the
//! silhouette), takes the local visible width / colour blended to the part median over `4 ×` the
//! median width, is anti-aliased from the exact Euclidean distance to the outside, and is never drawn
//! under or beside (parallel, within `2·width + 2` px) an upper layer's outline. An upper layer's
//! outline on a shared silhouette is never copied below it (that copy is the seam residue T9 removed).
//!
//! **Rest invariance**: the caller passes every layer; a modified hidden texel is accepted only if
//! the full rest composite (`mask ∪ fillMask` alpha, `core::blend_straight_pixel`, as the CLI
//! effective composite and `templates::merge_rest_invariant`) stays bit-identical; otherwise the
//! texel moves along the colour line toward the line colour as far as (1/64 steps) keeps the composite
//! identical (`clamped`: a fainter line of the same hue), or stays unchanged (`rejected`).

use crate::analysis::region_fill;
use crate::core::PartRole;
use crate::rig::fidelity::{delta_e2000, srgb_to_lab};
use crate::rig::seams::SeamRequirement;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::VecDeque;

pub const LINEART_VERSION: &str = "t15-v1";
/// Visible alpha for a pixel to count as line / colour evidence.
pub const EVIDENCE_ALPHA_MIN: u8 = 128;
/// Opening radius of the thin-stroke test (shared with T9 seeds).
pub const LINE_OPEN_PX: usize = region_fill::LINE_OPEN_PX;
/// Stroke fit window (px) = clamp(FIT_WINDOW_W · width, MIN, MAX).
pub const FIT_WINDOW_W: f64 = 6.0;
pub const FIT_WINDOW_MIN_PX: f64 = 8.0;
pub const FIT_WINDOW_MAX_PX: f64 = 24.0;
/// Minimum visible stroke length behind the contact.
pub const STROKE_MIN_LEN_PX: f64 = 4.0;
pub const STROKE_MIN_LEN_W: f64 = 1.5;
/// Maximum RMS residual of slab centroids around the quadratic centreline.
pub const FIT_RESID_MAX_PX: f64 = 0.75;
pub const FIT_RESID_MAX_W: f64 = 0.35;
/// A stroke whose axis makes more than this angle with the hidden-side direction runs along the
/// boundary (seam / silhouette outline), not into it.
pub const ENTRY_MAX_DEG: f64 = 75.0;
/// Fraction of a stroke within `width + 1.5` px of the silhouette for it to be a contour.
pub const CONTOUR_STROKE_FRAC: f64 = 0.5;
/// A component with at least this fraction of its pixels within `SEAM_OUTLINE_PX` of a visible pixel
/// of a layer above is the seam outline of that layer (or its residue), not an ending stroke.
pub const SEAM_OUTLINE_FRAC: f64 = 0.5;
pub const SEAM_OUTLINE_PX: f64 = 2.5;
/// An extension within `2·width + 2` px of the visible edge stops when |direction · seam normal| falls
/// below this (more than 60° from the normal).
pub const SEAM_PARALLEL_COS: f64 = 0.5;
/// A synthesized line within `2·width + 2` px of an upper layer's visible outline and within 30° of
/// parallel to it is not drawn (it would double that outline when the upper layer moves).
pub const UPPER_PARALLEL_COS: f64 = 0.866;
/// Strokes stop earlier: an upper outline within `2·width + 3` px (the posed double-line scan reach)
/// and within 60° of the stroke direction (also the per-texel rule for stroke texels).
pub const MARCH_UPPER_PARALLEL_COS: f64 = 0.5;
/// Maximum direction change of an extension (curvature is clamped, not a skip).
pub const MAX_TURN_DEG: f64 = 45.0;
/// Meeting another stroke before this much extension = `crossing` skip.
pub const CROSS_MIN_EXT_PX: f64 = 1.5;
/// Raster slack on the seam exposure depth (mesh interpolation / pixel rounding).
pub const DEPTH_SLACK_PX: f64 = 2.0;
/// March step along a stroke (px).
pub const MARCH_STEP_PX: f64 = 0.5;
/// Part silhouette counts as outlined when at least this fraction of visible silhouette pixels
/// has a line pixel within 1.5px.
pub const OUTLINE_MIN_COVERAGE: f64 = 0.5;
/// Hidden contour propagation stops where the silhouette ring turns more than this from the
/// carried outline tangent (corner of an underlap edge).
pub const CONTOUR_MAX_TURN_DEG: f64 = 35.0;
/// Exponential smoothing of the carried tangent per ring step.
pub const CONTOUR_TANGENT_SMOOTH: f64 = 0.2;
/// Hair stubs too short to fit use the layer's stroke direction field when at least
/// `FIELD_MIN_STROKES` fitted strokes lie within 2σ and their resultant length is ≥ `FIELD_MIN_COHERENCE`.
pub const FIELD_MIN_STROKES: usize = 2;
pub const FIELD_MIN_COHERENCE: f64 = 0.8;
/// Window for the local outline width / colour at a visible silhouette pixel.
pub const OUTLINE_LOCAL_R: i64 = 4;
/// Hair direction field: Gaussian σ = clamp(HAIR_FIELD_SIGMA_HH · hh, MIN, MAX) px.
pub const HAIR_FIELD_SIGMA_HH: f64 = 0.02;
pub const HAIR_FIELD_SIGMA_MIN: f64 = 6.0;
pub const HAIR_FIELD_SIGMA_MAX: f64 = 24.0;
/// Width clamp of an extended stroke relative to the part median stroke width.
pub const WIDTH_CLAMP_LO: f64 = 0.5;
pub const WIDTH_CLAMP_HI: f64 = 2.0;
/// Upper clamp of an extended *stroke* width (× part median stroke width): slab widths near branches
/// and junctions overestimate a thin stroke.
pub const STROKE_WIDTH_CLAMP_HI: f64 = 1.25;
/// Width at the tip of a taper.
pub const TAPER_TIP_W: f64 = 0.6;
/// Semantic stroke flags kept per layer in the report (the total is always counted).
pub const SEMANTIC_LIST_CAP: usize = 64;

thread_local! {
    static LINEART_ON: Cell<bool> = const { Cell::new(true) };
    static REST_GATE_ON: Cell<bool> = const { Cell::new(true) };
}

/// Measurement only (CLI `--lineart-no-rest-gate`): runs `f` without the rest-composite gate, to
/// quantify what bit-exact rest invariance costs in the exposure bands. The product path always gates.
pub fn without_rest_gate<R>(f: impl FnOnce() -> R) -> R {
    let prev = REST_GATE_ON.with(|c| c.replace(false));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            let p = self.0;
            REST_GATE_ON.with(|c| c.set(p));
        }
    }
    let _r = Restore(prev);
    f()
}

pub fn rest_gate_enabled() -> bool {
    REST_GATE_ON.with(|c| c.get())
}

/// Runs `f` with T15 line synthesis switched on/off on this thread (CLI `--no-lineart` measures
/// the colour-only result with the same binary).
pub fn with_lineart<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let prev = LINEART_ON.with(|c| c.replace(on));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            let p = self.0;
            LINEART_ON.with(|c| c.set(p));
        }
    }
    let _r = Restore(prev);
    f()
}

pub fn lineart_enabled() -> bool {
    LINEART_ON.with(|c| c.get())
}

/// A skipped stroke (low confidence → not extended; flagged as `semantic` for manual repair, T11).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SemanticFlag {
    pub x: f64,
    pub y: f64,
    pub reason: String,
    pub width_px: f64,
    /// Value that failed the gate (visible length px, residual px, or extension px at the crossing).
    pub value: f64,
    pub limit: f64,
}

/// Stroke candidates of one layer by outcome.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StrokeCounts {
    /// Line components touching the layer's hidden texels.
    pub touching_hidden: usize,
    /// Extended up to the exposure depth / the hidden edge / reconnection.
    pub extended: usize,
    /// Hair stubs shorter than the fit gate extended along the layer's stroke direction field.
    pub extended_by_field: usize,
    /// Extended up to another stroke (junction, never crossed).
    pub joined: usize,
    pub skipped_short: usize,
    pub skipped_fit_residual: usize,
    pub skipped_crossing: usize,
    /// Not candidates (no decision about confidence):
    /// every hidden neighbour is beyond the exposure depth (or the seam has no requirement).
    pub unexposed: usize,
    /// hidden neighbours are covered by an eye / brow / mouth layer (T10 domain).
    pub feature_seam: usize,
    /// runs along the hidden boundary (seam / silhouette outline).
    pub parallel: usize,
    /// lies along an upper layer's edge (≥ `SEAM_OUTLINE_FRAC` of its pixels within
    /// `SEAM_OUTLINE_PX` of an upper visible pixel): the upper's outline / seam residue.
    pub seam_outline: usize,
    /// its extension would run parallel to the seam right away (double-line risk).
    pub seam_parallel: usize,
    /// hugs the silhouette: an outline, continued only by the hidden contour (none when the part is
    /// not outlined).
    pub contour: usize,
    /// the first march step already left the allowed texels.
    pub no_room: usize,
}

impl StrokeCounts {
    pub fn skipped(&self) -> usize {
        self.skipped_short + self.skipped_fit_residual + self.skipped_crossing
    }
}

/// Rest-composite gate outcome (texels).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RestGate {
    pub accepted: usize,
    pub clamped: usize,
    pub rejected: usize,
    /// Holdout pixels (evaluation only; visible texels, not gated).
    pub held_out: usize,
}

/// One synthesized line texel (rest canvas index, coverage, stroke id, unit normal).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LineTexel {
    pub i: u32,
    pub alpha: f32,
    /// Stroke id within the layer; contours use `CONTOUR_ID_BASE + ring component`.
    pub stroke: u32,
    pub nx: f32,
    pub ny: f32,
    /// Accepted by the rest gate (fully or clamped).
    pub kept: bool,
}

pub const CONTOUR_ID_BASE: u32 = 1 << 24;

/// Per-layer T15 report (`RegionReport.lineart`, CLI `stats.json` `partCoverage[].fillRegion`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LineArtReport {
    pub version: String,
    /// Line threshold L* (visible median − 25).
    pub line_threshold_l: f64,
    pub visible_line_px: usize,
    pub line_components: usize,
    pub median_stroke_width_px: f64,
    pub taper_px: f64,
    pub strokes: StrokeCounts,
    pub extended_length_px: f64,
    pub stroke_texels: usize,
    /// Silhouette outline statistics of the visible part.
    pub outlined: bool,
    pub outline_coverage: f64,
    pub outline_width_px: f64,
    pub outline_color: String,
    /// Hidden silhouette ring pixels that received a contour (contour length, px).
    pub contour_length_px: usize,
    pub contour_texels: usize,
    /// Silhouette extension by the hidden area: hidden texels on the effective silhouette ring
    /// (`fillMask` outside the visible mask, within 1.5px of the outside of `mask ∪ fillMask`).
    pub hidden_ring_px: usize,
    /// …under an eye / brow / mouth layer (T10 domain).
    pub hidden_ring_feature_px: usize,
    /// …beyond the seam's exposure depth (never exposed: no new line by design).
    pub hidden_ring_beyond_depth_px: usize,
    /// …inside the exposure band (eligible for a contour).
    pub hidden_ring_exposed_px: usize,
    /// …of the eligible ones, also outside the covering upper layer (shared silhouette: at rest the
    /// upper layer's outline is drawn there; never copied below it).
    pub hidden_ring_shared_px: usize,
    /// Eligible ring pixels the contour reached but did not draw (under / beside an upper outline).
    pub contour_guard_skipped_px: usize,
    /// Synthesized texels removed by the texel-level seam-parallel guard (double-line risk).
    pub seam_parallel_guard_texels: usize,
    /// Junctions between visible and hidden contour and their continuity (width px / ΔE00).
    pub junctions: usize,
    pub junction_width_diff_mean: Option<f64>,
    pub junction_width_diff_max: Option<f64>,
    pub junction_de_mean: Option<f64>,
    pub junction_de_max: Option<f64>,
    pub rest_gate: RestGate,
    /// Low-confidence strokes not extended (first `SEMANTIC_LIST_CAP`; total in `semantic_total`).
    pub semantic: Vec<SemanticFlag>,
    pub semantic_total: usize,
    /// Kept line texels lying in hidden texels the 94-pose base synthesis already had (if T15 ran
    /// before `merge_rest_invariant`, these would take the base run's colours) / only the current
    /// synthesis has (the merge accepts them only where the rest composite is unchanged).
    pub texels_in_base_hidden: Option<usize>,
    pub texels_in_current_only_hidden: Option<usize>,
    #[serde(skip)]
    pub texels: Vec<LineTexel>,
}

/// Inputs of the scene-wide pass (all layers bottom-to-top).
pub struct LineArtScene<'a> {
    pub ids: &'a [String],
    pub roles: &'a [PartRole],
    pub masks: &'a [Vec<u8>],
    pub fills: &'a [Vec<u8>],
    /// The original artwork (never the holdout-substituted source).
    pub rgba: &'a [u8],
    pub w: usize,
    pub h: usize,
    /// Evaluation holdout pixels per layer (treated as hidden; their original pixels are not read).
    pub held: Vec<Option<&'a [bool]>>,
    pub seam_requirements: &'a [SeamRequirement],
    /// Head height (px) for scale-dependent constants.
    pub hh: f64,
}

/// T9 seam overlay texel of layer `p` at `i`: a `fillMask` texel above a lower layer's visible pixel
/// (no layer from `p` up is visible there). It carries the original seam colour (typically this
/// layer's outline half that the partition gave to the lower layer), so it is line / outline
/// evidence of `p` — and never a drawing target.
/// `owner` = rest visible owner (top-most layer with `mask > 0`, `-1` = none).
pub fn is_overlay_texel(fill_p: &[u8], owner: &[i32], p: usize, i: usize) -> bool {
    fill_p[i] > 0 && owner[i] >= 0 && (owner[i] as usize) < p
}

/// Rest visible owner of every pixel (top-most layer with `mask > 0`, `-1` = none).
pub fn rest_owner(masks: &[Vec<u8>], n: usize) -> Vec<i32> {
    let mut owner = vec![-1i32; n];
    for (p, m) in masks.iter().enumerate() {
        for (o, &a) in owner.iter_mut().zip(m) {
            if a > 0 {
                *o = p as i32;
            }
        }
    }
    owner
}

/// Does T15 synthesize lines for this role? Eyes / brows / mouth are T10 / T13b stroke domain.
pub fn role_takes_lines(role: PartRole) -> bool {
    !matches!(role, PartRole::Eye | PartRole::Brow | PartRole::Mouth)
}

fn is_hair(role: PartRole) -> bool {
    matches!(
        role,
        PartRole::HairFront | PartRole::HairSideL | PartRole::HairSideR | PartRole::HairBack
    )
}

/// Local box with a margin.
#[derive(Clone, Copy)]
struct Bx {
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    w: usize,
}

impl Bx {
    #[inline]
    fn g(&self, l: usize) -> usize {
        (self.y0 + l / self.bw) * self.w + self.x0 + l % self.bw
    }
    #[inline]
    fn at(&self, x: i64, y: i64) -> Option<usize> {
        if x < 0 || y < 0 || x >= self.bw as i64 || y >= self.bh as i64 {
            None
        } else {
            Some(y as usize * self.bw + x as usize)
        }
    }
    #[inline]
    fn xy(&self, l: usize) -> (i64, i64) {
        ((l % self.bw) as i64, (l / self.bw) as i64)
    }
}

/// Exact squared Euclidean distance transform (Felzenszwalb–Huttenlocher) to the `src` pixels.
/// Pixels without any source get ≥ `EDT_NONE` (1e9).
pub fn edt_sq(bw: usize, bh: usize, src: &[bool]) -> Vec<f32> {
    // f64 with a finite "infinity" keeps q² exact in the parabola intersections.
    const INF: f64 = 1e10;
    let dt1 = |f: &[f64], out: &mut [f64]| {
        let n = f.len();
        if n == 0 {
            return;
        }
        let mut v = vec![0usize; n];
        let mut z = vec![0f64; n + 1];
        let mut k = 0usize;
        z[0] = f64::NEG_INFINITY;
        z[1] = f64::INFINITY;
        for q in 1..n {
            let qf = q as f64;
            loop {
                let p = v[k] as f64;
                let s = ((f[q] + qf * qf) - (f[v[k]] + p * p)) / (2.0 * (qf - p));
                if s <= z[k] {
                    if k == 0 {
                        // Unreachable with z[0] = −∞; kept for robustness.
                        break;
                    }
                    k -= 1;
                    continue;
                }
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
        k = 0;
        for (q, o) in out.iter_mut().enumerate() {
            while z[k + 1] < q as f64 {
                k += 1;
            }
            let d = q as f64 - v[k] as f64;
            *o = d * d + f[v[k]];
        }
    };
    let mut f: Vec<f64> = src.iter().map(|&s| if s { 0.0 } else { INF }).collect();
    let mut col = vec![0f64; bh];
    let mut tmp = vec![0f64; bh.max(bw)];
    for x in 0..bw {
        for y in 0..bh {
            col[y] = f[y * bw + x];
        }
        dt1(&col, &mut tmp[..bh]);
        for y in 0..bh {
            f[y * bw + x] = tmp[y];
        }
    }
    for y in 0..bh {
        let row: Vec<f64> = f[y * bw..(y + 1) * bw].to_vec();
        dt1(&row, &mut f[y * bw..(y + 1) * bw]);
    }
    f.into_iter().map(|d| d as f32).collect()
}

/// Squared distances at or above this mean "no source pixel".
pub const EDT_NONE: f32 = 1e9;

/// Square-window erosion (`all` = true) or dilation of a boolean grid (separable).
fn morph(b: &[bool], bw: usize, bh: usize, r: usize, erode: bool) -> Vec<bool> {
    let pass = |src: &[bool], horiz: bool| -> Vec<bool> {
        let mut out = vec![false; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                let mut acc = erode;
                for d in -(r as i64)..=(r as i64) {
                    let (nx, ny) = if horiz {
                        (x as i64 + d, y as i64)
                    } else {
                        (x as i64, y as i64 + d)
                    };
                    let v = if nx < 0 || ny < 0 || nx >= bw as i64 || ny >= bh as i64 {
                        false
                    } else {
                        src[ny as usize * bw + nx as usize]
                    };
                    if erode && !v {
                        acc = false;
                        break;
                    }
                    if !erode && v {
                        acc = true;
                        break;
                    }
                }
                out[y * bw + x] = acc;
            }
        }
        out
    };
    let h1 = pass(b, true);
    pass(&h1, false)
}

/// Visible line-art pixels of a layer (canvas-sized), with hysteresis so an anti-aliased 1px
/// stroke is not cut into fragments where its pixels straddle the threshold: *strong* pixels have
/// L* at least `LINE_DROP_L` below the evidence median, *weak* pixels at least
/// `LINE_DROP_L − LINE_WEAK_SLACK_L`; weak pixels 8-connected to a strong one join it. The union is
/// line art where it disappears under an opening of radius `LINE_OPEN_PX` (thin). Returns
/// `(line, strong threshold L*)`; `line_strong` (the T14 metric rule) is `line ∧ L* < threshold`.
pub fn detect_line_pixels(
    lstar_at: &dyn Fn(usize) -> f32,
    evidence: &[bool],
    w: usize,
    h: usize,
) -> (Vec<bool>, f32) {
    let n = w * h;
    let mut ls: Vec<f32> = (0..n).filter(|&i| evidence[i]).map(lstar_at).collect();
    if ls.is_empty() {
        return (vec![false; n], 0.0);
    }
    ls.sort_by(|a, b| a.total_cmp(b));
    let thr = ls[ls.len() / 2] - region_fill::LINE_DROP_L;
    let Some(bx) = bbox(w, h, 2, |i| evidence[i]) else {
        return (vec![false; n], thr);
    };
    let nb = bx.bw * bx.bh;
    let strong: Vec<bool> = (0..nb)
        .map(|l| {
            let g = bx.g(l);
            evidence[g] && lstar_at(g) < thr
        })
        .collect();
    let weak: Vec<bool> = (0..nb)
        .map(|l| {
            let g = bx.g(l);
            evidence[g] && lstar_at(g) < thr + LINE_WEAK_SLACK_L
        })
        .collect();
    let mut dark = strong.clone();
    let mut q: VecDeque<usize> = (0..nb).filter(|&l| strong[l]).collect();
    while let Some(l) = q.pop_front() {
        let (x, y) = bx.xy(l);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if let Some(m) = bx.at(x + dx, y + dy) {
                    if weak[m] && !dark[m] {
                        dark[m] = true;
                        q.push_back(m);
                    }
                }
            }
        }
    }
    let opened = morph(
        &morph(&dark, bx.bw, bx.bh, LINE_OPEN_PX, true),
        bx.bw,
        bx.bh,
        LINE_OPEN_PX,
        false,
    );
    let mut out = vec![false; n];
    for l in 0..nb {
        if dark[l] && !opened[l] {
            out[bx.g(l)] = true;
        }
    }
    (out, thr)
}

/// Hysteresis slack of the weak line threshold (L*).
pub const LINE_WEAK_SLACK_L: f32 = 10.0;

fn bbox(w: usize, h: usize, margin: usize, pred: impl Fn(usize) -> bool) -> Option<Bx> {
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
    if x0 == usize::MAX {
        return None;
    }
    let x0 = x0.saturating_sub(margin);
    let y0 = y0.saturating_sub(margin);
    let x1 = (x1 + 1 + margin).min(w);
    let y1 = (y1 + 1 + margin).min(h);
    Some(Bx {
        x0,
        y0,
        bw: x1 - x0,
        bh: y1 - y0,
        w,
    })
}

fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

fn median_rgb(cs: &[[u8; 3]]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for c in 0..3 {
        let mut v: Vec<u8> = cs.iter().map(|p| p[c]).collect();
        v.sort_unstable();
        out[c] = v.get(v.len() / 2).copied().unwrap_or(0);
    }
    out
}

/// CIELAB L* of the RGB at canvas index `g` of an RGBA buffer (LUT linearisation).
pub fn lstar_px(rgba: &[u8], g: usize) -> f32 {
    static LIN: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    let lin = LIN.get_or_init(|| {
        let mut t = [0f32; 256];
        for (c, v) in t.iter_mut().enumerate() {
            let s = c as f64 / 255.0;
            *v = if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            } as f32;
        }
        t
    });
    let y = 0.2126729 * lin[rgba[g * 4] as usize]
        + 0.7151522 * lin[rgba[g * 4 + 1] as usize]
        + 0.072175 * lin[rgba[g * 4 + 2] as usize];
    if y > 216.0 / 24389.0 {
        116.0 * y.cbrt() - 16.0
    } else {
        24389.0 / 27.0 * y
    }
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn de00(a: [u8; 3], b: [u8; 3]) -> f64 {
    delta_e2000(srgb_to_lab(a), srgb_to_lab(b))
}

/// Fitted visible stroke near its hidden contact.
struct StrokeFit {
    c0: (f64, f64),
    u: (f64, f64),
    n: (f64, f64),
    /// s(t) = a + b·t + c·t² in the (u, n) frame around `c0`.
    a: f64,
    b: f64,
    c: f64,
    t_front: f64,
    width: f64,
    color: [u8; 3],
}

impl StrokeFit {
    fn point(&self, t: f64) -> (f64, f64) {
        let s = self.a + self.b * t + self.c * t * t;
        (
            self.c0.0 + self.u.0 * t + self.n.0 * s,
            self.c0.1 + self.u.1 * t + self.n.1 * s,
        )
    }
    fn tangent(&self, t: f64) -> (f64, f64) {
        let ds = self.b + 2.0 * self.c * t;
        let (x, y) = (self.u.0 + self.n.0 * ds, self.u.1 + self.n.1 * ds);
        let l = x.hypot(y).max(1e-9);
        (x / l, y / l)
    }
}

enum FitOutcome {
    Fit(StrokeFit),
    Parallel,
    Short { len: f64, limit: f64, width: f64 },
    Residual { resid: f64, limit: f64, width: f64 },
}

/// Least squares a + b t + c t² (weighted), falling back to lower degree.
fn quad_fit(pts: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let deg = if pts.len() >= 3 {
        2
    } else if pts.len() == 2 {
        1
    } else {
        0
    };
    let mut m = [[0f64; 3]; 3];
    let mut r = [0f64; 3];
    for &(t, s, wt) in pts {
        let p = [1.0, t, t * t];
        for i in 0..=deg {
            for j in 0..=deg {
                m[i][j] += wt * p[i] * p[j];
            }
            r[i] += wt * p[i] * s;
        }
    }
    // Gaussian elimination on the (deg+1)² system.
    let k = deg + 1;
    let mut a = m;
    let mut bvec = r;
    for col in 0..k {
        let piv = (col..k)
            .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
            .unwrap();
        if a[piv][col].abs() < 1e-12 {
            return (
                if pts.is_empty() {
                    0.0
                } else {
                    pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64
                },
                0.0,
                0.0,
            );
        }
        a.swap(col, piv);
        bvec.swap(col, piv);
        for row in 0..k {
            if row != col {
                let f = a[row][col] / a[col][col];
                let pivot = a[col];
                for (c2, v) in a[row].iter_mut().enumerate().take(k).skip(col) {
                    *v -= f * pivot[c2];
                }
                bvec[row] -= f * bvec[col];
            }
        }
    }
    let mut out = [0f64; 3];
    for i in 0..k {
        out[i] = bvec[i] / a[i][i];
    }
    (out[0], out[1], out[2])
}

#[allow(clippy::too_many_arguments)]
fn fit_stroke(
    comp: &[usize],
    contacts: &[(usize, (f64, f64))],
    bx: &Bx,
    tex: &[u8],
    median_w: f64,
) -> FitOutcome {
    let n_c = contacts.len() as f64;
    let c0 = contacts.iter().fold((0.0, 0.0), |acc, &(l, _)| {
        let (x, y) = bx.xy(l);
        (acc.0 + x as f64 + 0.5, acc.1 + y as f64 + 0.5)
    });
    let c0 = (c0.0 / n_c, c0.1 / n_c);
    let hd = contacts
        .iter()
        .fold((0.0, 0.0), |acc, &(_, d)| (acc.0 + d.0, acc.1 + d.1));
    let hl = hd.0.hypot(hd.1).max(1e-9);
    let hdir = (hd.0 / hl, hd.1 / hl);
    let mut radius = 16.0f64;
    let mut width = median_w.max(1.0);
    let mut local: Vec<(f64, f64, usize)> = Vec::new();
    let mut u = (1.0, 0.0);
    for _ in 0..2 {
        local = comp
            .iter()
            .filter_map(|&l| {
                let (x, y) = bx.xy(l);
                let p = (x as f64 + 0.5, y as f64 + 0.5);
                ((p.0 - c0.0).hypot(p.1 - c0.1) <= radius).then_some((p.0, p.1, l))
            })
            .collect();
        if local.len() < 2 {
            return FitOutcome::Short {
                len: 0.0,
                limit: STROKE_MIN_LEN_PX,
                width,
            };
        }
        let nl = local.len() as f64;
        let (mx, my) = local.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
        let (mx, my) = (mx / nl, my / nl);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for p in &local {
            let (dx, dy) = (p.0 - mx, p.1 - my);
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
        let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        u = (ang.cos(), ang.sin());
        // Extent along the axis → width estimate (pixels per 1px slab).
        let ts: Vec<f64> = local
            .iter()
            .map(|p| (p.0 - c0.0) * u.0 + (p.1 - c0.1) * u.1)
            .collect();
        let (tmin, tmax) = ts
            .iter()
            .fold((f64::MAX, f64::MIN), |a, &t| (a.0.min(t), a.1.max(t)));
        width = (nl / (tmax - tmin + 1.0).max(1.0)).max(1.0);
        radius = (FIT_WINDOW_W * width).clamp(FIT_WINDOW_MIN_PX, FIT_WINDOW_MAX_PX);
    }
    if u.0 * hdir.0 + u.1 * hdir.1 < 0.0 {
        u = (-u.0, -u.1);
    }
    let cosang = u.0 * hdir.0 + u.1 * hdir.1;
    if cosang < ENTRY_MAX_DEG.to_radians().cos() {
        return FitOutcome::Parallel;
    }
    let nrm = (-u.1, u.0);
    // 1px slabs along the axis.
    let mut slabs: std::collections::BTreeMap<i64, (f64, f64)> = std::collections::BTreeMap::new();
    let mut t_front = f64::MIN;
    let mut t_back = f64::MAX;
    for p in &local {
        let (dx, dy) = (p.0 - c0.0, p.1 - c0.1);
        let t = dx * u.0 + dy * u.1;
        let s = dx * nrm.0 + dy * nrm.1;
        t_front = t_front.max(t);
        t_back = t_back.min(t);
        let e = slabs.entry(t.floor() as i64).or_insert((0.0, 0.0));
        e.0 += s;
        e.1 += 1.0;
    }
    let vis_len = (t_front - t_back + 1.0).max(0.0);
    let mut counts: Vec<f64> = slabs.values().map(|v| v.1).collect();
    let slab_w = median(&mut counts).unwrap_or(1.0);
    // Width near the contact (front third of the window).
    let mut front_counts: Vec<f64> = slabs
        .iter()
        .filter(|(&k, _)| (k as f64) >= t_front - (vis_len / 3.0).max(3.0))
        .map(|(_, v)| v.1)
        .collect();
    let w_front = median(&mut front_counts).unwrap_or(slab_w);
    let len_limit = STROKE_MIN_LEN_PX.max(STROKE_MIN_LEN_W * slab_w);
    if vis_len < len_limit {
        return FitOutcome::Short {
            len: vis_len,
            limit: len_limit,
            width: slab_w,
        };
    }
    let pts: Vec<(f64, f64, f64)> = slabs
        .iter()
        .map(|(&k, v)| (k as f64 + 0.5, v.0 / v.1, v.1))
        .collect();
    let (a, b, c) = quad_fit(&pts);
    let resid = {
        let (mut s2, mut ws) = (0.0, 0.0);
        for &(t, s, wt) in &pts {
            let e = s - (a + b * t + c * t * t);
            s2 += wt * e * e;
            ws += wt;
        }
        (s2 / ws.max(1e-9)).sqrt()
    };
    let resid_limit = FIT_RESID_MAX_PX.max(FIT_RESID_MAX_W * slab_w);
    if resid > resid_limit {
        return FitOutcome::Residual {
            resid,
            limit: resid_limit,
            width: slab_w,
        };
    }
    // Colour: median of the darker half of the window pixels.
    let mut dark: Vec<(f32, [u8; 3])> = local
        .iter()
        .map(|p| {
            let g = bx.g(p.2);
            (
                lstar_px(tex, g),
                [tex[g * 4], tex[g * 4 + 1], tex[g * 4 + 2]],
            )
        })
        .collect();
    dark.sort_by(|x, y| x.0.total_cmp(&y.0));
    let half: Vec<[u8; 3]> = dark
        .iter()
        .take(dark.len().div_ceil(2))
        .map(|d| d.1)
        .collect();
    let width = if median_w > 0.0 {
        w_front.clamp(WIDTH_CLAMP_LO * median_w, STROKE_WIDTH_CLAMP_HI * median_w)
    } else {
        w_front
    };
    FitOutcome::Fit(StrokeFit {
        c0,
        u,
        n: nrm,
        a,
        b,
        c,
        t_front,
        width: width.max(1.0),
        color: median_rgb(&half),
    })
}

/// End taper length (px) of a part's free stroke ends: slabs from an end until the width reaches
/// 75 % of the component's median slab width (median over all ends of elongated components).
fn measure_taper(comps: &[Vec<usize>], touching: &[bool], bx: &Bx) -> Option<f64> {
    let mut tapers = Vec::new();
    for (k, comp) in comps.iter().enumerate() {
        if touching[k] || comp.len() < 12 {
            continue;
        }
        let nl = comp.len() as f64;
        let pts: Vec<(f64, f64)> = comp
            .iter()
            .map(|&l| {
                let (x, y) = bx.xy(l);
                (x as f64 + 0.5, y as f64 + 0.5)
            })
            .collect();
        let (mx, my) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
        let (mx, my) = (mx / nl, my / nl);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for p in &pts {
            let (dx, dy) = (p.0 - mx, p.1 - my);
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
        let tr = sxx + syy;
        let det = sxx * syy - sxy * sxy;
        let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
        let (l1, l2) = (tr / 2.0 + disc, (tr / 2.0 - disc).max(1e-9));
        if l1 / l2 < 16.0 {
            continue; // not elongated (blob / junction)
        }
        let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let u = (ang.cos(), ang.sin());
        let mut slabs: std::collections::BTreeMap<i64, f64> = std::collections::BTreeMap::new();
        for p in &pts {
            *slabs
                .entry(((p.0 - mx) * u.0 + (p.1 - my) * u.1).floor() as i64)
                .or_insert(0.0) += 1.0;
        }
        let counts: Vec<f64> = slabs.values().copied().collect();
        let mut c2 = counts.clone();
        let wm = median(&mut c2).unwrap_or(1.0);
        if wm < 1.5 || counts.len() < 6 {
            continue;
        }
        for dir in [false, true] {
            let it: Box<dyn Iterator<Item = &f64>> = if dir {
                Box::new(counts.iter().rev())
            } else {
                Box::new(counts.iter())
            };
            let mut k2 = 0usize;
            for &c in it {
                if c >= 0.75 * wm {
                    break;
                }
                k2 += 1;
            }
            if k2 < counts.len() / 2 {
                tapers.push(k2 as f64);
            }
        }
    }
    median(&mut tapers)
}

/// Synthesized line layer of one part (local box): coverage, colour, stroke id, normal.
struct LineLayer {
    a: Vec<f32>,
    c: Vec<[u8; 3]>,
    id: Vec<u32>,
    nrm: Vec<(f32, f32)>,
}

impl LineLayer {
    fn new(nb: usize) -> Self {
        Self {
            a: vec![0.0; nb],
            c: vec![[0; 3]; nb],
            id: vec![u32::MAX; nb],
            nrm: vec![(0.0, 0.0); nb],
        }
    }
    fn put(&mut self, l: usize, a: f32, c: [u8; 3], id: u32, nrm: (f64, f64)) {
        if a > self.a[l] {
            self.a[l] = a;
            self.c[l] = c;
            self.id[l] = id;
            self.nrm[l] = (nrm.0 as f32, nrm.1 as f32);
        }
    }
}

fn flag(rep: &mut LineArtReport, at: (f64, f64), reason: &str, width: f64, value: f64, limit: f64) {
    rep.semantic_total += 1;
    if rep.semantic.len() < SEMANTIC_LIST_CAP {
        rep.semantic.push(SemanticFlag {
            x: at.0,
            y: at.1,
            reason: reason.into(),
            width_px: width,
            value,
            limit,
        });
    }
}

/// Synthesized line texel before the gate: (canvas index, line rgb, coverage, stroke id, normal).
type Draw = (usize, [u8; 3], f32, u32, (f32, f32));
/// Short hair stub awaiting the direction field: (component, contact, width, length, limit).
type Stub = (usize, (f64, f64), f64, f64, f64);
/// Direction-field sample: (contact point, tangent, component).
type FieldSample = ((f64, f64), (f64, f64), usize);
/// Pending texel change: (layer, new rgb, old rgb, held out).
type PendingMod = (usize, [u8; 3], [u8; 3], bool);

struct LayerOut {
    report: LineArtReport,
    /// (canvas index, line rgb, alpha, stroke id, normal)
    draws: Vec<Draw>,
}

/// Synthesizes the line layer of one part (not yet merged / gated).
#[allow(clippy::too_many_arguments)]
fn layer_lines(
    scene: &LineArtScene<'_>,
    tex: &[u8],
    p: usize,
    owner: &[i32],
    depth_of: &dyn Fn(usize, usize) -> Option<f64>,
    upper_line: &[Option<u16>],
) -> Option<LayerOut> {
    let (w, h) = (scene.w, scene.h);
    let n = w * h;
    let vis = &scene.masks[p];
    let fill = &scene.fills[p];
    let held = scene.held[p];
    let is_held = |i: usize| held.is_some_and(|hm| hm[i] && vis[i] > 0);
    let cand_g = |i: usize| {
        (fill[i] > 0 && vis[i] == 0 && !(owner[i] >= 0 && (owner[i] as usize) < p)) || is_held(i)
    };
    let bx = bbox(w, h, 3, |i| vis[i] > 0 || fill[i] > 0)?;
    let nb = bx.bw * bx.bh;
    let cand: Vec<bool> = (0..nb).map(|l| cand_g(bx.g(l))).collect();
    if !cand.iter().any(|&c| c) {
        return None;
    }
    let evidence_g: Vec<bool> = (0..n)
        .map(|i| {
            (vis[i] >= EVIDENCE_ALPHA_MIN && !is_held(i))
                || is_overlay_texel(&scene.fills[p], owner, p, i)
        })
        .collect();
    let lt = |g: usize| lstar_px(tex, g);
    let (line_g, thr) = detect_line_pixels(&lt, &evidence_g, w, h);
    let ev: Vec<bool> = (0..nb).map(|l| evidence_g[bx.g(l)]).collect();
    let line: Vec<bool> = (0..nb).map(|l| line_g[bx.g(l)]).collect();
    let mut rep = LineArtReport {
        version: LINEART_VERSION.into(),
        line_threshold_l: thr as f64,
        visible_line_px: line.iter().filter(|&&b| b).count(),
        ..Default::default()
    };
    // Visible (non-held) support of this layer, for depth.
    let vis_l: Vec<bool> = (0..nb)
        .map(|l| {
            let g = bx.g(l);
            vis[g] > 0 && !is_held(g)
        })
        .collect();
    let dvis = edt_sq(bx.bw, bx.bh, &vis_l);
    // Nearest upper owner of every candidate (rest owner when it is above; else BFS through the
    // candidate area from upper-owned pixels — held pixels and uncovered fill).
    let mut up = vec![-1i32; nb];
    let mut q = VecDeque::new();
    for l in 0..nb {
        let o = owner[bx.g(l)];
        if o >= 0 && (o as usize) > p {
            up[l] = o;
            q.push_back(l);
        }
    }
    while let Some(l) = q.pop_front() {
        let (x, y) = bx.xy(l);
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            if let Some(m) = bx.at(x + dx, y + dy) {
                if up[m] < 0 && cand[m] {
                    up[m] = up[l];
                    q.push_back(m);
                }
            }
        }
    }
    // Allowed texels: candidate, outside feature seams, within the seam's exposure depth.
    let mut feature = vec![false; nb];
    let allowed: Vec<bool> = (0..nb)
        .map(|l| {
            if !cand[l] {
                return false;
            }
            let g = bx.g(l);
            let o = up[l];
            if o >= 0 && !role_takes_lines(scene.roles[o as usize]) {
                feature[l] = true;
                return false;
            }
            if is_held(g) {
                return true;
            }
            let limit = if o >= 0 {
                depth_of(o as usize, p)
            } else {
                depth_of(usize::MAX, p)
            };
            match limit {
                Some(d) => (dvis[l] as f64).sqrt() <= d + DEPTH_SLACK_PX,
                None => false,
            }
        })
        .collect();
    // Components of the visible line art.
    let mut comp_id = vec![u32::MAX; nb];
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for s in 0..nb {
        if !line[s] || comp_id[s] != u32::MAX {
            continue;
        }
        let id = comps.len() as u32;
        let mut c = vec![s];
        comp_id[s] = id;
        let mut k = 0;
        while k < c.len() {
            let (x, y) = bx.xy(c[k]);
            k += 1;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(m) = bx.at(x + dx, y + dy) {
                        if line[m] && comp_id[m] == u32::MAX {
                            comp_id[m] = id;
                            c.push(m);
                        }
                    }
                }
            }
        }
        comps.push(c);
    }
    rep.line_components = comps.len();
    // Median stroke width over components (area / skeleton-length proxy = slab median).
    let mut widths = Vec::new();
    for c in &comps {
        if c.len() >= 6 {
            let nl = c.len() as f64;
            let (mut minx, mut maxx, mut miny, mut maxy) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
            for &l in c {
                let (x, y) = bx.xy(l);
                minx = minx.min(x);
                maxx = maxx.max(x);
                miny = miny.min(y);
                maxy = maxy.max(y);
            }
            let len = ((maxx - minx + 1) as f64).hypot((maxy - miny + 1) as f64);
            widths.push((nl / len.max(1.0)).max(1.0));
        }
    }
    let median_w = median(&mut widths).unwrap_or(1.5);
    rep.median_stroke_width_px = median_w;

    // Effective silhouette (mask ∪ fill) and the inward distance from the outside.
    let eff: Vec<bool> = (0..nb)
        .map(|l| {
            let g = bx.g(l);
            vis[g] > 0 || fill[g] > 0
        })
        .collect();
    let outside: Vec<bool> = eff.iter().map(|&e| !e).collect();
    let din = edt_sq(bx.bw, bx.bh, &outside);
    let dist_in = |l: usize| (din[l] as f64).sqrt();
    // Pixels whose outside is only the canvas border do not count as silhouette.
    let at_canvas_edge = |l: usize| {
        let (x, y) = bx.xy(l);
        let gx = bx.x0 as i64 + x;
        let gy = bx.y0 as i64 + y;
        gx <= 0 || gy <= 0 || gx >= w as i64 - 1 || gy >= h as i64 - 1
    };
    let grad = |l: usize| -> (f64, f64) {
        let (x, y) = bx.xy(l);
        let d = |dx: i64, dy: i64| bx.at(x + dx, y + dy).map_or(0.0, dist_in);
        let (gx, gy) = (d(1, 0) - d(-1, 0), d(0, 1) - d(0, -1));
        let gl = gx.hypot(gy);
        if gl < 1e-9 {
            (0.0, 0.0)
        } else {
            (gx / gl, gy / gl)
        }
    };

    // ---- Item 3: visible silhouette outline statistics.
    let ring = |l: usize| eff[l] && dist_in(l) <= 1.5 && !at_canvas_edge(l);
    for l in 0..nb {
        if !(cand[l] && ring(l)) {
            continue;
        }
        rep.hidden_ring_px += 1;
        if feature[l] {
            rep.hidden_ring_feature_px += 1;
        } else if !allowed[l] {
            rep.hidden_ring_beyond_depth_px += 1;
        } else {
            rep.hidden_ring_exposed_px += 1;
            let q = up[l];
            let (x, y) = bx.xy(l);
            let shared = q >= 0
                && [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|&(dx, dy)| {
                        bx.at(x + dx, y + dy).is_some_and(|m| {
                            let g = bx.g(m);
                            !eff[m]
                                && scene.masks[q as usize][g] == 0
                                && scene.fills[q as usize][g] == 0
                        })
                    });
            if shared {
                rep.hidden_ring_shared_px += 1;
            }
        }
    }
    let mut sil_total = 0usize;
    let mut outline_w = vec![0f64; nb];
    let mut outline_c = vec![[0u8; 3]; nb];
    let mut outlined_px = vec![false; nb];
    let near_line = |l: usize| {
        let (x, y) = bx.xy(l);
        (-1..=1).any(|dy| (-1..=1).any(|dx| bx.at(x + dx, y + dy).is_some_and(|m| line[m])))
    };
    for l in 0..nb {
        if !(ev[l] && ring(l)) {
            continue;
        }
        sil_total += 1;
        if !near_line(l) {
            continue;
        }
        let (gx, gy) = grad(l);
        let (x, y) = bx.xy(l);
        let mut wv = 0usize;
        let mut cs = Vec::new();
        for k in 0..12 {
            let px = (x as f64 + 0.5 + gx * k as f64).floor() as i64;
            let py = (y as f64 + 0.5 + gy * k as f64).floor() as i64;
            match bx.at(px, py) {
                Some(m) if line[m] => {
                    wv += 1;
                    let g = bx.g(m);
                    cs.push([tex[g * 4], tex[g * 4 + 1], tex[g * 4 + 2]]);
                }
                Some(m) if k == 0 && !line[m] => continue,
                _ => break,
            }
        }
        if wv > 0 {
            outlined_px[l] = true;
            outline_w[l] = wv as f64;
            outline_c[l] = median_rgb(&cs);
        }
    }
    let n_out = outlined_px.iter().filter(|&&b| b).count();
    rep.outline_coverage = if sil_total > 0 {
        n_out as f64 / sil_total as f64
    } else {
        0.0
    };
    rep.outlined = sil_total >= 16 && rep.outline_coverage >= OUTLINE_MIN_COVERAGE;
    let mut ws: Vec<f64> = (0..nb)
        .filter(|&l| outlined_px[l])
        .map(|l| outline_w[l])
        .collect();
    let part_ow = median(&mut ws).unwrap_or(0.0);
    let all_oc: Vec<[u8; 3]> = (0..nb)
        .filter(|&l| outlined_px[l])
        .map(|l| outline_c[l])
        .collect();
    let part_oc = median_rgb(&all_oc);
    rep.outline_width_px = part_ow;
    rep.outline_color = if all_oc.is_empty() {
        String::new()
    } else {
        hex(part_oc)
    };

    let mut ll = LineLayer::new(nb);
    let to_global = |p: (f64, f64)| (p.0 + bx.x0 as f64, p.1 + bx.y0 as f64);

    // ---- Item 3: hidden contour along the effective silhouette ring.
    let upper_line_near = |l: usize| {
        let (x, y) = bx.xy(l);
        (-1..=1).any(|dy| {
            (-1..=1).any(|dx| {
                bx.at(x + dx, y + dy)
                    .is_some_and(|m| upper_line[bx.g(m)].is_some_and(|q| q as usize > p))
            })
        })
    };
    // Is there an upper layer's visible outline within `lim` px of local point `c` running within
    // 30° of direction `d`? (PCA of the upper line pixels in the window.) A T15 line there would
    // show beside that outline as a second, parallel line when the upper layer moves.
    let upper_parallel = |c: (f64, f64), d: (f64, f64), lim: f64, cos_min: f64| -> bool {
        let r = lim.ceil() as i64;
        let (cx, cy) = (c.0.floor() as i64, c.1.floor() as i64);
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                if ((dx * dx + dy * dy) as f64) > lim * lim {
                    continue;
                }
                if let Some(m) = bx.at(cx + dx, cy + dy) {
                    if upper_line[bx.g(m)].is_some_and(|q| q as usize > p) {
                        pts.push((dx as f64, dy as f64));
                    }
                }
            }
        }
        if pts.len() < 3 {
            return false;
        }
        let nl = pts.len() as f64;
        let (mx, my) = pts.iter().fold((0.0, 0.0), |a, q| (a.0 + q.0, a.1 + q.1));
        let (mx, my) = (mx / nl, my / nl);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for q in &pts {
            sxx += (q.0 - mx) * (q.0 - mx);
            sxy += (q.0 - mx) * (q.1 - my);
            syy += (q.1 - my) * (q.1 - my);
        }
        let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        (d.0 * ang.cos() + d.1 * ang.sin()).abs() >= cos_min
    };
    let mut junction_pairs: Vec<(usize, usize)> = Vec::new();
    let own_outline = rep.outlined && part_ow > 0.0;
    if own_outline {
        // Local width / colour at outlined visible ring pixels (window median).
        let local_stats = |l: usize| -> (f64, [u8; 3]) {
            let (x, y) = bx.xy(l);
            let mut wv = Vec::new();
            let mut cv = Vec::new();
            for dy in -OUTLINE_LOCAL_R..=OUTLINE_LOCAL_R {
                for dx in -OUTLINE_LOCAL_R..=OUTLINE_LOCAL_R {
                    if let Some(m) = bx.at(x + dx, y + dy) {
                        if outlined_px[m] {
                            wv.push(outline_w[m]);
                            cv.push(outline_c[m]);
                        }
                    }
                }
            }
            (
                median(&mut wv).unwrap_or(part_ow),
                if cv.is_empty() {
                    part_oc
                } else {
                    median_rgb(&cv)
                },
            )
        };
        // BFS along the hidden ring from outlined visible ring pixels. The silhouette tangent is
        // carried along the ring (exponential smoothing `CONTOUR_TANGENT_SMOOTH`); propagation stops
        // where the ring turns more than `CONTOUR_MAX_TURN_DEG` away from it — a corner where the
        // hidden area's underlap edge meets the silhouette is not a continuation of the outline.
        let grad2 = |l: usize| -> (f64, f64) {
            let (x, y) = bx.xy(l);
            let d = |dx: i64, dy: i64| bx.at(x + dx, y + dy).map_or(0.0, |m| dist_in(m).min(4.0));
            let (gx, gy) = (
                d(2, 0) - d(-2, 0) + d(1, 0) - d(-1, 0),
                d(0, 2) - d(0, -2) + d(0, 1) - d(0, -1),
            );
            let gl = gx.hypot(gy);
            if gl < 1e-9 {
                (0.0, 0.0)
            } else {
                (gx / gl, gy / gl)
            }
        };
        let tangent = |l: usize| {
            let g = grad2(l);
            (-g.1, g.0)
        };
        let max_turn_cos = CONTOUR_MAX_TURN_DEG.to_radians().cos();
        let mut src_w = vec![f64::NAN; nb];
        let mut src_c = vec![[0u8; 3]; nb];
        let mut path = vec![f64::INFINITY; nb];
        let mut ring_comp = vec![u32::MAX; nb];
        let mut carried = vec![(0f64, 0f64); nb];
        let mut q: VecDeque<usize> = VecDeque::new();
        let mut seeded = 0u32;
        for l in 0..nb {
            if !(allowed[l] && ring(l)) {
                continue;
            }
            let (x, y) = bx.xy(l);
            let mut best: Option<usize> = None;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    if let Some(m) = bx.at(x + dx, y + dy) {
                        if outlined_px[m] && best.is_none() {
                            best = Some(m);
                        }
                    }
                }
            }
            // Only the layer's own outlined silhouette seeds a hidden contour. An upper layer's
            // outline on a shared silhouette is not copied below it: that copy is the seam residue
            // T9 removed and reads as a second outline as soon as the upper layer moves.
            if let Some(m) = best {
                let tv = tangent(m);
                let th = tangent(l);
                if (tv.0 * th.0 + tv.1 * th.1).abs() < max_turn_cos {
                    continue;
                }
                let (wl, cl) = local_stats(m);
                src_w[l] = wl;
                src_c[l] = cl;
                path[l] = 0.0;
                ring_comp[l] = seeded;
                carried[l] = tv;
                seeded += 1;
                junction_pairs.push((m, l));
                q.push_back(l);
            }
        }
        while let Some(l) = q.pop_front() {
            let (x, y) = bx.xy(l);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    if let Some(m) = bx.at(x + dx, y + dy) {
                        let step = if dx != 0 && dy != 0 {
                            std::f64::consts::SQRT_2
                        } else {
                            1.0
                        };
                        if allowed[m] && ring(m) && path[l] + step < path[m] {
                            let tm = tangent(m);
                            let c = carried[l];
                            let dot = c.0 * tm.0 + c.1 * tm.1;
                            if dot.abs() < max_turn_cos {
                                continue;
                            }
                            let s = dot.signum();
                            let k = CONTOUR_TANGENT_SMOOTH;
                            let (nx, ny) = (
                                (1.0 - k) * c.0 + k * s * tm.0,
                                (1.0 - k) * c.1 + k * s * tm.1,
                            );
                            let nl = nx.hypot(ny).max(1e-9);
                            carried[m] = (nx / nl, ny / nl);
                            path[m] = path[l] + step;
                            src_w[m] = src_w[l];
                            src_c[m] = src_c[l];
                            ring_comp[m] = ring_comp[l];
                            q.push_back(m);
                        }
                    }
                }
            }
        }
        rep.junctions = junction_pairs.len();
        // Fill the band (distance-to-outside ≤ width + 1) from the nearest reached ring pixel.
        let reached: Vec<bool> = (0..nb).map(|l| path[l].is_finite()).collect();
        rep.contour_length_px = reached.iter().filter(|&&b| b).count();
        if rep.contour_length_px > 0 {
            let ref_w = part_ow;
            let max_w = (WIDTH_CLAMP_HI * ref_w).ceil() + 1.0;
            let mut owner_ring = vec![usize::MAX; nb];
            let mut steps = vec![u16::MAX; nb];
            let mut q2: VecDeque<usize> = VecDeque::new();
            for l in 0..nb {
                if reached[l] {
                    owner_ring[l] = l;
                    steps[l] = 0;
                    q2.push_back(l);
                }
            }
            while let Some(l) = q2.pop_front() {
                if steps[l] as f64 >= max_w {
                    continue;
                }
                let (x, y) = bx.xy(l);
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if let Some(m) = bx.at(x + dx, y + dy) {
                        if allowed[m] && steps[m] == u16::MAX {
                            steps[m] = steps[l] + 1;
                            owner_ring[m] = owner_ring[l];
                            q2.push_back(m);
                        }
                    }
                }
            }
            for (l, &r) in owner_ring.iter().enumerate() {
                if r == usize::MAX {
                    continue;
                }
                // Junction width / colour blended to the part median over 4× the median width.
                // Never under an upper layer's outline (a copy of it would double in motion).
                let gl = grad(l);
                let (lx, ly) = bx.xy(l);
                if upper_line_near(l)
                    || upper_parallel(
                        (lx as f64 + 0.5, ly as f64 + 0.5),
                        (-gl.1, gl.0),
                        2.0 * part_ow.max(1.0) + 2.0,
                        UPPER_PARALLEL_COS,
                    )
                {
                    if ring(l) {
                        rep.contour_guard_skipped_px += 1;
                    }
                    continue;
                }
                let (pw, pc) = (part_ow, part_oc);
                let t = (path[r] / (4.0 * pw).max(1.0)).min(1.0);
                let wl =
                    (src_w[r] * (1.0 - t) + pw * t).clamp(WIDTH_CLAMP_LO * pw, WIDTH_CLAMP_HI * pw);
                let c = src_c[r];
                let c = if t > 0.0 {
                    let mut o = [0u8; 3];
                    for ch in 0..3 {
                        o[ch] = (c[ch] as f64 * (1.0 - t) + pc[ch] as f64 * t).round() as u8;
                    }
                    o
                } else {
                    c
                };
                let a = (wl - dist_in(l) + 1.0).clamp(0.0, 1.0) as f32;
                if a > 0.0 {
                    let g = grad(l);
                    ll.put(l, a, c, CONTOUR_ID_BASE + ring_comp[r], g);
                }
            }
        }
    }

    // ---- Item 2 / 4: stroke extension.
    // Contacts: line pixels 4-adjacent to candidate texels, with the direction toward them.
    let mut contacts: Vec<Vec<(usize, (f64, f64))>> = vec![Vec::new(); comps.len()];
    let mut contact_allowed = vec![false; comps.len()];
    let mut contact_feature = vec![false; comps.len()];
    for (k, c) in comps.iter().enumerate() {
        for &l in c {
            let (x, y) = bx.xy(l);
            for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                if let Some(m) = bx.at(x + dx, y + dy) {
                    if cand[m] {
                        contacts[k].push((l, (dx as f64, dy as f64)));
                        contact_allowed[k] |= allowed[m];
                        contact_feature[k] |= feature[m];
                    }
                }
            }
        }
    }
    let touching: Vec<bool> = contacts.iter().map(|c| !c.is_empty()).collect();
    let taper = measure_taper(&comps, &touching, &bx)
        .unwrap_or(median_w)
        .clamp(1.0, (4.0 * median_w).max(1.0));
    rep.taper_px = taper;
    let sigma = (HAIR_FIELD_SIGMA_HH * scene.hh).clamp(HAIR_FIELD_SIGMA_MIN, HAIR_FIELD_SIGMA_MAX);
    let hair = is_hair(scene.roles[p]);
    // Distance to the visible pixels of every layer above (seam outline test).
    let upper_vis: Vec<bool> = (0..nb)
        .map(|l| {
            let o = owner[bx.g(l)];
            o >= 0 && (o as usize) > p
        })
        .collect();
    let d_upper = edt_sq(bx.bw, bx.bh, &upper_vis);
    // First pass: classify and fit.
    let mut fits: Vec<Option<StrokeFit>> = (0..comps.len()).map(|_| None).collect();
    // Drawn outcome per component (1 field, 2 joined, 3 extended; 0 = not drawn).
    let mut outcome = vec![0u8; comps.len()];
    let mut stubs: Vec<Stub> = Vec::new();
    for k in 0..comps.len() {
        if !touching[k] {
            continue;
        }
        rep.strokes.touching_hidden += 1;
        if !contact_allowed[k] {
            if contact_feature[k] {
                rep.strokes.feature_seam += 1;
            } else {
                rep.strokes.unexposed += 1;
            }
            continue;
        }
        let near_upper = comps[k]
            .iter()
            .filter(|&&l| (d_upper[l] as f64).sqrt() <= SEAM_OUTLINE_PX)
            .count();
        if near_upper as f64 >= SEAM_OUTLINE_FRAC * comps[k].len() as f64 {
            rep.strokes.seam_outline += 1;
            continue;
        }
        // Silhouette-hugging components are outlines, continued only as hidden contours (item 3) —
        // and not at all when the part's silhouette is not outlined — never as free strokes.
        {
            let band = if rep.outlined { part_ow } else { median_w }.max(1.0) + 1.5;
            let near = comps[k].iter().filter(|&&l| dist_in(l) <= band).count();
            if near as f64 >= CONTOUR_STROKE_FRAC * comps[k].len() as f64 {
                rep.strokes.contour += 1;
                continue;
            }
        }
        let at = {
            let (x, y) = bx.xy(contacts[k][0].0);
            to_global((x as f64 + 0.5, y as f64 + 0.5))
        };
        match fit_stroke(&comps[k], &contacts[k], &bx, tex, median_w) {
            FitOutcome::Parallel => rep.strokes.parallel += 1,
            FitOutcome::Short { len, limit, width } => {
                if hair {
                    stubs.push((k, at, width, len, limit));
                } else {
                    rep.strokes.skipped_short += 1;
                    flag(&mut rep, at, "short", width, len, limit);
                }
            }
            FitOutcome::Residual {
                resid,
                limit,
                width,
            } => {
                rep.strokes.skipped_fit_residual += 1;
                flag(&mut rep, at, "fitResidual", width, resid, limit);
            }
            FitOutcome::Fit(f) => fits[k] = Some(f),
        }
    }
    // Hair direction field samples: (contact point, tangent) of every fitted stroke.
    let field: Vec<FieldSample> = fits
        .iter()
        .enumerate()
        .filter_map(|(k, f)| {
            f.as_ref()
                .map(|f| (f.point(f.t_front), f.tangent(f.t_front), k))
        })
        .collect();
    // Hair stubs: direction from the fitted neighbours (stroke direction field) when coherent.
    let mut by_field = vec![false; comps.len()];
    for &(k, at, width, len, limit) in &stubs {
        let nc = contacts[k].len() as f64;
        let c0 = contacts[k].iter().fold((0.0, 0.0), |a, &(l, _)| {
            let (x, y) = bx.xy(l);
            (a.0 + x as f64 + 0.5, a.1 + y as f64 + 0.5)
        });
        let c0 = (c0.0 / nc, c0.1 / nc);
        let hd = contacts[k]
            .iter()
            .fold((0.0, 0.0), |a, &(_, d)| (a.0 + d.0, a.1 + d.1));
        let (mut fx, mut fy, mut ws, mut near) = (0.0, 0.0, 0.0, 0usize);
        for &(cp, tg, _) in &field {
            let d2 = (c0.0 - cp.0).powi(2) + (c0.1 - cp.1).powi(2);
            if d2 > 4.0 * sigma * sigma {
                continue;
            }
            near += 1;
            let g = (-d2 / (2.0 * sigma * sigma)).exp();
            let sgn = if tg.0 * hd.0 + tg.1 * hd.1 < 0.0 {
                -1.0
            } else {
                1.0
            };
            fx += g * sgn * tg.0;
            fy += g * sgn * tg.1;
            ws += g;
        }
        let res = fx.hypot(fy);
        let coherence = if ws > 0.0 { res / ws } else { 0.0 };
        if near < FIELD_MIN_STROKES
            || coherence < FIELD_MIN_COHERENCE
            || fx * hd.0 + fy * hd.1 <= 0.0
        {
            rep.strokes.skipped_short += 1;
            flag(&mut rep, at, "short", width, len, limit);
            continue;
        }
        let u = (fx / res, fy / res);
        let nrm = (-u.1, u.0);
        let t_front = comps[k]
            .iter()
            .map(|&l| {
                let (x, y) = bx.xy(l);
                (x as f64 + 0.5 - c0.0) * u.0 + (y as f64 + 0.5 - c0.1) * u.1
            })
            .fold(f64::MIN, f64::max);
        let mut px: Vec<(f32, [u8; 3])> = comps[k]
            .iter()
            .map(|&l| {
                let g = bx.g(l);
                (
                    lstar_px(tex, g),
                    [tex[g * 4], tex[g * 4 + 1], tex[g * 4 + 2]],
                )
            })
            .collect();
        px.sort_by(|a, b| a.0.total_cmp(&b.0));
        let half: Vec<[u8; 3]> = px.iter().take(px.len().div_ceil(2)).map(|p| p.1).collect();
        fits[k] = Some(StrokeFit {
            c0,
            u,
            n: nrm,
            a: 0.0,
            b: 0.0,
            c: 0.0,
            t_front,
            width: width
                .clamp(WIDTH_CLAMP_LO * median_w, STROKE_WIDTH_CLAMP_HI * median_w)
                .max(1.0),
            color: median_rgb(&half),
        });
        by_field[k] = true;
    }
    let max_depth = (0..nb)
        .filter(|&l| allowed[l])
        .map(|l| (dvis[l] as f64).sqrt())
        .fold(0.0f64, f64::max)
        + DEPTH_SLACK_PX;
    for k in 0..comps.len() {
        let Some(f) = fits[k].as_ref() else { continue };
        let max_len = (max_depth + f.width).max(2.0);
        // Bound the curvature so the extension turns at most MAX_TURN_DEG.
        let cmax = MAX_TURN_DEG.to_radians().tan() / (2.0 * max_len);
        let c_used = f.c.clamp(-cmax, cmax);
        let fit = StrokeFit {
            c: c_used,
            color: f.color,
            ..*f
        };
        let start = fit.point(fit.t_front);
        let mut pos = start;
        let mut dir = fit.tangent(fit.t_front);
        let mut t_own = fit.t_front;
        let mut arc = 0.0f64;
        let mut pts: Vec<((f64, f64), f64)> = Vec::new();
        let mut end_by_depth = false;
        let mut crossing_at: Option<f64> = None;
        let mut reconnected = false;
        let mut parallel_stop = false;
        while arc < max_len {
            let own = fit.tangent(t_own);
            let mut d = own;
            if hair {
                let (mut fx, mut fy, mut ws) = (0.0, 0.0, 0.0);
                for &(cp, tg, kk) in &field {
                    if kk == k {
                        continue;
                    }
                    let d2 = (pos.0 - cp.0).powi(2) + (pos.1 - cp.1).powi(2);
                    let g = (-d2 / (2.0 * sigma * sigma)).exp();
                    let sgn = if tg.0 * dir.0 + tg.1 * dir.1 < 0.0 {
                        -1.0
                    } else {
                        1.0
                    };
                    fx += g * sgn * tg.0;
                    fy += g * sgn * tg.1;
                    ws += g;
                }
                if ws > 0.05 {
                    d = (own.0 + fx / ws, own.1 + fy / ws);
                    let l = d.0.hypot(d.1).max(1e-9);
                    d = (d.0 / l, d.1 / l);
                }
            }
            dir = d;
            let next = (pos.0 + dir.0 * MARCH_STEP_PX, pos.1 + dir.1 * MARCH_STEP_PX);
            t_own += MARCH_STEP_PX * (dir.0 * fit.u.0 + dir.1 * fit.u.1);
            let Some(m) = bx.at(next.0.floor() as i64, next.1.floor() as i64) else {
                break;
            };
            let own_visible = line[m] && comp_id[m] == k as u32;
            if !allowed[m] && !own_visible {
                if cand[m] && !feature[m] {
                    end_by_depth = true;
                } else if line[m] {
                    reconnected = true;
                }
                break;
            }
            arc += MARCH_STEP_PX;
            pos = next;
            if own_visible {
                continue;
            }
            // Seam-parallel stop: within `2·width + 2` px of the visible edge, an extension turning
            // more than 60° away from the seam normal would run beside the upper layer's outline and
            // show as a second outline when that layer moves (double line).
            if arc > fit.width {
                let dv = (dvis[m] as f64).sqrt();
                if dv <= 2.0 * fit.width + 2.0 {
                    let (mx, my) = bx.xy(m);
                    let dd = |dx: i64, dy: i64| {
                        bx.at(mx + dx, my + dy)
                            .map_or(dv, |o| (dvis[o] as f64).sqrt().min(1e4))
                    };
                    let (gx, gy) = (dd(1, 0) - dd(-1, 0), dd(0, 1) - dd(0, -1));
                    let gl = gx.hypot(gy);
                    if gl > 1e-9 && ((dir.0 * gx + dir.1 * gy) / gl).abs() < SEAM_PARALLEL_COS {
                        parallel_stop = true;
                        break;
                    }
                }
            }
            if upper_parallel(
                pos,
                dir,
                2.0 * fit.width.max(median_w).max(rep.outline_width_px) + 3.0,
                MARCH_UPPER_PARALLEL_COS,
            ) {
                parallel_stop = true;
                break;
            }
            // Crossing: an earlier synthesized line (other stroke / contour) at or beside the
            // centreline, or another visible stroke beside it once past twice the stroke width.
            if arc > fit.width {
                let (mx, my) = bx.xy(m);
                let other = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|&(dx, dy)| {
                        bx.at(mx + dx, my + dy).is_some_and(|o| {
                            (ll.a[o] >= 0.5 && ll.id[o] != k as u32)
                                || (arc > 2.0 * fit.width && line[o] && comp_id[o] != k as u32)
                        })
                    });
                if other {
                    crossing_at = Some(arc);
                    break;
                }
            }
            pts.push((pos, arc));
        }
        if let Some(at) = crossing_at {
            if at < CROSS_MIN_EXT_PX || pts.is_empty() {
                rep.strokes.skipped_crossing += 1;
                flag(
                    &mut rep,
                    to_global(start),
                    "crossing",
                    fit.width,
                    at,
                    CROSS_MIN_EXT_PX,
                );
                continue;
            }
        }
        if pts.is_empty() {
            if parallel_stop {
                rep.strokes.seam_parallel += 1;
            } else {
                rep.strokes.no_room += 1;
            }
            continue;
        }
        outcome[k] = if by_field[k] {
            rep.strokes.extended_by_field += 1;
            1
        } else if crossing_at.is_some() || reconnected {
            rep.strokes.joined += 1;
            2
        } else {
            rep.strokes.extended += 1;
            3
        };
        let total = pts.last().map_or(0.0, |p| p.1);
        rep.extended_length_px += total;
        // Rasterise the stroke as a chain of capsules (local box coordinates).
        let mut chain = vec![(start, 0.0)];
        chain.extend(pts.iter().copied());
        let wd = |s: f64| -> f64 {
            if end_by_depth {
                let rem = total - s;
                (fit.width * (rem / taper).min(1.0)).max(TAPER_TIP_W)
            } else {
                fit.width
            }
        };
        let nrm = |a: (f64, f64), b: (f64, f64)| {
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let l = dx.hypot(dy).max(1e-9);
            (-dy / l, dx / l)
        };
        for win in chain.windows(2) {
            let ((a, sa), (b, sb)) = (win[0], win[1]);
            let r = 0.5 * wd(0.5 * (sa + sb));
            let lo_x = (a.0.min(b.0) - r - 1.0).floor() as i64;
            let hi_x = (a.0.max(b.0) + r + 1.0).ceil() as i64;
            let lo_y = (a.1.min(b.1) - r - 1.0).floor() as i64;
            let hi_y = (a.1.max(b.1) + r + 1.0).ceil() as i64;
            let nv = nrm(a, b);
            for yy in lo_y..=hi_y {
                for xx in lo_x..=hi_x {
                    let Some(m) = bx.at(xx, yy) else { continue };
                    if !allowed[m] {
                        continue;
                    }
                    let (px, py) = (xx as f64 + 0.5, yy as f64 + 0.5);
                    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
                    let len2 = abx * abx + aby * aby;
                    let t = if len2 > 1e-12 {
                        (((px - a.0) * abx + (py - a.1) * aby) / len2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let d = (px - a.0 - t * abx).hypot(py - a.1 - t * aby);
                    let al = (r + 0.5 - d).clamp(0.0, 1.0) as f32;
                    if al > 0.0 {
                        ll.put(m, al, fit.color, k as u32, nv);
                    }
                }
            }
        }
    }

    // Texel-level seam-parallel guard (the same rule `seam_parallel_texels` measures): a synthesized
    // line texel within `2·width + 2` px of this layer's visible edge whose line runs within 30° of
    // parallel to that edge would sit beside the covering layer's outline and double it when the
    // cover moves. Such texels are not drawn.
    let guard_lim = 2.0 * median_w.max(rep.outline_width_px).max(1.0) + 2.0;
    for l in 0..nb {
        if ll.a[l] <= 0.0 {
            continue;
        }
        let dv = (dvis[l] as f64).sqrt();
        if dv > guard_lim {
            continue;
        }
        let (x, y) = bx.xy(l);
        let dd = |dx: i64, dy: i64| {
            bx.at(x + dx, y + dy)
                .map_or(dv, |o| (dvis[o] as f64).sqrt().min(1e4))
        };
        let (gx, gy) = (dd(1, 0) - dd(-1, 0), dd(0, 1) - dd(0, -1));
        let gl = gx.hypot(gy);
        if gl < 1e-9 {
            continue;
        }
        let (nx, ny) = ll.nrm[l];
        if ((nx as f64 * gx + ny as f64 * gy) / gl).abs() >= UPPER_PARALLEL_COS {
            ll.a[l] = 0.0;
            rep.seam_parallel_guard_texels += 1;
        }
    }
    // Same rule against the rest outline of every layer above (per texel; the march tests only the
    // centreline).
    for l in 0..nb {
        if ll.a[l] <= 0.0 {
            continue;
        }
        let (x, y) = bx.xy(l);
        let (nx, ny) = ll.nrm[l];
        // Strokes use the march rule (45°, posed scan reach); contours 30°.
        let (lim, cmin) = if ll.id[l] >= CONTOUR_ID_BASE {
            (guard_lim, UPPER_PARALLEL_COS)
        } else {
            (guard_lim + 1.0, MARCH_UPPER_PARALLEL_COS)
        };
        if upper_parallel(
            (x as f64 + 0.5, y as f64 + 0.5),
            (-(ny as f64), nx as f64),
            lim,
            cmin,
        ) {
            ll.a[l] = 0.0;
            rep.seam_parallel_guard_texels += 1;
        }
    }

    // A stroke whose texels the guards removed entirely was not extended: count it as seam-parallel.
    let mut kept_per_stroke = vec![0usize; comps.len()];
    for l in 0..nb {
        if ll.a[l] > 0.0 && ll.id[l] < CONTOUR_ID_BASE {
            if let Some(c) = kept_per_stroke.get_mut(ll.id[l] as usize) {
                *c += 1;
            }
        }
    }
    for (k, &o) in outcome.iter().enumerate() {
        if o > 0 && kept_per_stroke[k] == 0 {
            match o {
                1 => rep.strokes.extended_by_field -= 1,
                2 => rep.strokes.joined -= 1,
                _ => rep.strokes.extended -= 1,
            }
            rep.strokes.seam_parallel += 1;
        }
    }

    // Junction continuity (contours): visible width / colour within OUTLINE_LOCAL_R of the junction
    // vs the synthesized contour texels within the same radius on the hidden side.
    if !junction_pairs.is_empty() {
        let mut dws = Vec::new();
        let mut des = Vec::new();
        for &(vm, hl) in junction_pairs
            .iter()
            .step_by((junction_pairs.len() / 256).max(1))
        {
            let (vx, vy) = bx.xy(vm);
            let (hx, hy) = bx.xy(hl);
            let mut vw = Vec::new();
            let mut vc = Vec::new();
            let mut hc = Vec::new();
            let mut hw = Vec::new();
            for dy in -OUTLINE_LOCAL_R..=OUTLINE_LOCAL_R {
                for dx in -OUTLINE_LOCAL_R..=OUTLINE_LOCAL_R {
                    if let Some(m) = bx.at(vx + dx, vy + dy) {
                        if outlined_px[m] {
                            vw.push(outline_w[m]);
                            vc.push(outline_c[m]);
                        }
                    }
                    if let Some(m) = bx.at(hx + dx, hy + dy) {
                        if ll.id[m] >= CONTOUR_ID_BASE && ll.a[m] >= 0.5 {
                            hc.push(ll.c[m]);
                            if ring(m) {
                                // Synthesized width at this ring pixel: covered steps inward.
                                let (gx, gy) = grad(m);
                                let (x, y) = bx.xy(m);
                                let mut k2 = 0.0;
                                for s in 0..12 {
                                    let px = (x as f64 + 0.5 + gx * s as f64).floor() as i64;
                                    let py = (y as f64 + 0.5 + gy * s as f64).floor() as i64;
                                    match bx.at(px, py) {
                                        Some(o) if ll.a[o] > 0.0 && ll.id[o] >= CONTOUR_ID_BASE => {
                                            k2 += ll.a[o] as f64
                                        }
                                        _ => break,
                                    }
                                }
                                hw.push(k2);
                            }
                        }
                    }
                }
            }
            if let (Some(a), Some(b)) = (median(&mut vw), median(&mut hw)) {
                dws.push((a - b).abs());
            }
            if !vc.is_empty() && !hc.is_empty() {
                des.push(de00(median_rgb(&vc), median_rgb(&hc)));
            }
        }
        let mean = |v: &[f64]| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);
        let max = |v: &[f64]| v.iter().copied().reduce(f64::max);
        rep.junction_width_diff_mean = mean(&dws);
        rep.junction_width_diff_max = max(&dws);
        rep.junction_de_mean = mean(&des);
        rep.junction_de_max = max(&des);
    }

    let mut draws = Vec::new();
    for (l, &ok) in allowed.iter().enumerate() {
        if ll.a[l] > 0.0 && ok {
            draws.push((bx.g(l), ll.c[l], ll.a[l], ll.id[l], ll.nrm[l]));
            if ll.id[l] >= CONTOUR_ID_BASE {
                rep.contour_texels += 1;
            } else {
                rep.stroke_texels += 1;
            }
        }
    }
    Some(LayerOut { report: rep, draws })
}

/// Composite (straight alpha, bottom-to-top) of every layer at pixel `i`, with optional colour
/// overrides — the rest composite of the CLI / `merge_rest_invariant`.
fn rest_px(
    scene: &LineArtScene<'_>,
    textures: &[Option<Vec<u8>>],
    i: usize,
    over: &[(usize, [u8; 3])],
) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (p, tex) in textures.iter().enumerate().take(scene.ids.len()) {
        let a = scene.masks[p][i].max(scene.fills[p][i]);
        let Some(t) = tex.as_ref() else { continue };
        if a == 0 {
            continue;
        }
        let c = over
            .iter()
            .find(|o| o.0 == p)
            .map_or([t[i * 4], t[i * 4 + 1], t[i * 4 + 2]], |o| o.1);
        crate::core::blend_straight_pixel(&mut out, [c[0], c[1], c[2], a]);
    }
    out
}

/// Scene-wide pass: synthesizes each eligible layer's line layer, merges it over the colour layer
/// and applies the rest-composite gate. `textures` are decoded straight RGBA canvases (modified in
/// place); returns one report per layer (`None` when the layer takes no lines or has no hidden
/// texels). `mask` / `fillMask` are never changed.
pub fn synthesize_lineart(
    scene: &LineArtScene<'_>,
    textures: &mut [Option<Vec<u8>>],
) -> Vec<Option<LineArtReport>> {
    let (w, h) = (scene.w, scene.h);
    let n = w * h;
    let layers = scene.ids.len();
    let mut reports: Vec<Option<LineArtReport>> = vec![None; layers];
    if !lineart_enabled() || layers == 0 {
        return reports;
    }

    let owner = rest_owner(scene.masks, n);
    let depth_of = |upper: usize, lower: usize| -> Option<f64> {
        let lid = &scene.ids[lower];
        if upper != usize::MAX {
            let uid = &scene.ids[upper];
            if let Some(d) = scene
                .seam_requirements
                .iter()
                .filter(|r| &r.upper_id == uid && &r.lower_id == lid)
                .map(|r| r.max_exposure_depth_px)
                .reduce(f64::max)
            {
                return Some(d);
            }
        }
        scene
            .seam_requirements
            .iter()
            .filter(|r| &r.lower_id == lid)
            .map(|r| r.max_exposure_depth_px)
            .reduce(f64::max)
    };
    // Pending modifications per pixel: (layer, new rgb, old rgb, held).
    let mut pending: std::collections::BTreeMap<usize, Vec<PendingMod>> = Default::default();
    // Visible line art of every layer (its texture), for the "never under an upper outline" guard.
    let mut line_owner: Vec<Option<u16>> = vec![None; n];
    for (p, tex) in textures.iter().enumerate().take(layers) {
        let Some(t) = tex.as_deref() else { continue };
        let ev: Vec<bool> = (0..n)
            .map(|i| {
                scene.masks[p][i] >= EVIDENCE_ALPHA_MIN
                    || is_overlay_texel(&scene.fills[p], &owner, p, i)
            })
            .collect();
        let (line, _) = detect_line_pixels(&|g| lstar_px(t, g), &ev, w, h);
        for (o, &b) in line_owner.iter_mut().zip(&line) {
            if b {
                *o = Some(p as u16);
            }
        }
    }
    let mut outs: Vec<Option<LayerOut>> = (0..layers).map(|_| None).collect();
    for p in 0..layers {
        if !role_takes_lines(scene.roles[p]) || textures[p].is_none() {
            continue;
        }
        let Some(out) = layer_lines(
            scene,
            textures[p].as_deref().unwrap_or(&[]),
            p,
            &owner,
            &depth_of,
            &line_owner,
        ) else {
            continue;
        };
        let tex = textures[p].as_ref().unwrap();
        for &(i, c, a, _, _) in &out.draws {
            let old = [tex[i * 4], tex[i * 4 + 1], tex[i * 4 + 2]];
            let mut new = [0u8; 3];
            for ch in 0..3 {
                new[ch] =
                    (a as f64 * c[ch] as f64 + (1.0 - a as f64) * old[ch] as f64).round() as u8;
            }
            if new != old {
                let held = scene.held[p].is_some_and(|hm| hm[i] && scene.masks[p][i] > 0);
                pending.entry(i).or_default().push((p, new, old, held));
            }
        }
        outs[p] = Some(out);
    }
    // Rest gate, bottom-to-top per pixel.
    let mut status: std::collections::HashMap<(usize, usize), u8> = Default::default(); // 1 kept full, 2 clamped, 0 rejected
    let mut gates = vec![RestGate::default(); layers];
    for (&i, mods) in &pending {
        let mut accepted: Vec<(usize, [u8; 3])> = Vec::new();
        let olds: Vec<(usize, [u8; 3])> = mods.iter().map(|m| (m.0, m.2)).collect();
        let reference = rest_px(scene, textures, i, &olds);
        let mut ms = mods.clone();
        ms.sort_by_key(|m| m.0);
        for (p, new, old, held) in ms {
            if held || !rest_gate_enabled() {
                accepted.push((p, new));
                status.insert((p, i), 1);
                if held {
                    gates[p].held_out += 1;
                } else {
                    gates[p].accepted += 1;
                }
                continue;
            }
            let try_over = |c: [u8; 3]| -> [u8; 4] {
                let mut o: Vec<(usize, [u8; 3])> = olds
                    .iter()
                    .filter(|x| x.0 != p && !accepted.iter().any(|a| a.0 == x.0))
                    .copied()
                    .collect();
                o.extend(accepted.iter().copied());
                o.push((p, c));
                rest_px(scene, textures, i, &o)
            };
            if try_over(new) == reference {
                accepted.push((p, new));
                status.insert((p, i), 1);
                gates[p].accepted += 1;
                continue;
            }
            // Along the colour line old → new (hue kept; a per-channel clamp would shift the hue):
            // the largest step k/64 whose colour leaves the composite bit-identical. Straight-alpha
            // blending is monotone in the texel colour, so the accepted steps form a prefix.
            let at = |k: i32| -> [u8; 3] {
                let t = k as f64 / 64.0;
                let mut c = [0u8; 3];
                for ch in 0..3 {
                    c[ch] = (old[ch] as f64 + t * (new[ch] as f64 - old[ch] as f64)).round() as u8;
                }
                c
            };
            let (mut lo, mut hi) = (0i32, 64i32);
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if try_over(at(mid)) == reference {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            let cur = at(lo);
            if cur != old && try_over(cur) == reference {
                accepted.push((p, cur));
                status.insert((p, i), 2);
                gates[p].clamped += 1;
            } else {
                status.insert((p, i), 0);
                gates[p].rejected += 1;
            }
        }
        for (p, c) in accepted {
            if let Some(t) = textures[p].as_mut() {
                t[i * 4..i * 4 + 3].copy_from_slice(&c);
            }
        }
    }
    for p in 0..layers {
        let Some(out) = outs[p].take() else { continue };
        let mut rep = out.report;
        rep.rest_gate = gates[p].clone();
        rep.texels = out
            .draws
            .iter()
            .map(|&(i, _, a, id, nrm)| LineTexel {
                i: i as u32,
                alpha: a,
                stroke: id,
                nx: nrm.0,
                ny: nrm.1,
                // Unchanged colour (line already matches) counts as kept.
                kept: status.get(&(p, i)).is_none_or(|&s| s > 0),
            })
            .collect();
        reports[p] = Some(rep);
    }
    reports
}

/// Seam-parallel synthesized line texels (double-line risk, AGENTS.md 2 ⑤): kept line texels
/// (strokes and contours, coverage ≥ 0.5) of layer `p` within `2·width + 2` px of its own visible
/// edge (the rest seam with the covering upper layer) whose direction is within 30° of that edge. At rest they are covered; when the upper layer moves
/// by less than that distance they would show next to the upper outline as a second line.
pub fn seam_parallel_texels(
    p: usize,
    rep: &LineArtReport,
    masks: &[Vec<u8>],
    w: usize,
    h: usize,
) -> usize {
    let n = w * h;
    // The rest seam is the edge of this layer's visible mask (hidden texels lie under the upper).
    let own: Vec<bool> = (0..n).map(|i| masks[p][i] > 0).collect();
    let d = edt_sq(w, h, &own);
    let dist = |x: i64, y: i64| -> f64 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            0.0
        } else {
            (d[y as usize * w + x as usize] as f64).sqrt()
        }
    };
    let lim = 2.0
        * rep
            .median_stroke_width_px
            .max(rep.outline_width_px)
            .max(1.0)
        + 2.0;
    rep.texels
        .iter()
        .filter(|t| t.kept && t.alpha >= 0.5)
        .filter(|t| {
            let (x, y) = ((t.i as usize % w) as i64, (t.i as usize / w) as i64);
            let dd = dist(x, y);
            if dd > lim || dd <= 0.0 {
                return false;
            }
            // Seam boundary direction ⟂ ∇dist; stroke direction ⟂ its normal.
            let (gx, gy) = (
                dist(x + 1, y) - dist(x - 1, y),
                dist(x, y + 1) - dist(x, y - 1),
            );
            let gl = gx.hypot(gy);
            if gl < 1e-9 {
                return false;
            }
            // Parallel when the stroke normal is within 30° of the distance gradient.
            let c = ((t.nx as f64 * gx + t.ny as f64 * gy) / gl).abs();
            c >= 30f64.to_radians().cos()
        })
        .count()
}

/// Posed double-line check (AGENTS.md 2 ⑤) over `poses`: for every output pixel whose top texel is
/// a kept T15 line texel (coverage ≥ 0.5) of layer `L`, scan both ways along the stroke normal up to
/// `2·width + 3` px: after leaving the stroke, if the scan crosses only `L`'s own non-line texels and
/// then meets line art of **another layer** (original visible line pixel or T15 line) running within
/// 30° of parallel to the stroke (PCA of that layer's posed line pixels within 3px), the pixel is
/// a cross-layer double line (a second, parallel line separated by `L`'s colour — the seam-residue
/// pattern). Scans that reach background or another layer's non-line texel stop (two edges of
/// different objects). Same-layer parallels (hair strands) are counted separately for reference.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DoubleLineReport {
    pub poses: usize,
    /// Σ over poses of exposed T15 line pixels (top texel, coverage ≥ 0.5).
    pub exposed_line_px: usize,
    pub cross_layer_double_px: usize,
    /// …of which within 3px of a baseline double (same pose).
    pub cross_layer_double_near_baseline_px: usize,
    /// The same pattern formed by the original (non-T15) visible line art of lower layers (HEAD has
    /// it too: the visible lines and outlines move exactly as at HEAD).
    pub baseline_cross_layer_double_px: usize,
    pub same_layer_parallel_px: usize,
    /// (pose, layer, x, y) of the first cross-layer hits (≤ 16).
    /// First cross-layer hits (≤ 16): pose, layer, posed x / y, rest texel x / y, the other layer and
    /// whether that other line is itself a T15 line.
    pub examples: Vec<DoubleExample>,
    pub per_layer: Vec<(String, usize, usize)>,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DoubleExample {
    pub pose: String,
    pub layer: String,
    pub x: usize,
    pub y: usize,
    pub rest_x: usize,
    pub rest_y: usize,
    pub other: String,
    pub other_is_t15: bool,
    /// A baseline (original-line) double lies within 3px in the same pose.
    pub near_baseline: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn posed_double_lines(
    ids: &[String],
    masks: &[Vec<u8>],
    fills: &[Vec<u8>],
    meshes: &[crate::core::Mesh],
    reports: &[Option<&LineArtReport>],
    textures: &[&[u8]],
    lm: &crate::core::Landmarks,
    w: usize,
    h: usize,
    poses: &[crate::rig::seams::EnvelopePose],
) -> DoubleLineReport {
    use std::collections::HashMap;
    let n = w * h;
    let layers = ids.len();
    let owner = rest_owner(masks, n);
    // Visible line art per layer (from its texture: T9-decontaminated rim, as the pass reads it).
    let orig_line: Vec<Vec<bool>> = (0..layers)
        .map(|p| {
            let ev: Vec<bool> = (0..n)
                .map(|i| {
                    masks[p][i] >= EVIDENCE_ALPHA_MIN || is_overlay_texel(&fills[p], &owner, p, i)
                })
                .collect();
            let tex = textures[p];
            if tex.len() < n * 4 {
                return vec![false; n];
            }
            detect_line_pixels(&|g| lstar_px(tex, g), &ev, w, h).0
        })
        .collect();
    let t15: Vec<HashMap<u32, (u32, f32, f32)>> = (0..layers)
        .map(|p| {
            reports
                .get(p)
                .and_then(|r| *r)
                .map(|r| {
                    r.texels
                        .iter()
                        .filter(|t| t.kept && t.alpha >= 0.5)
                        .map(|t| (t.i, (t.stroke, t.nx, t.ny)))
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();
    let width: Vec<f64> = (0..layers)
        .map(|p| {
            reports.get(p).and_then(|r| *r).map_or(1.0, |r| {
                r.median_stroke_width_px.max(r.outline_width_px).max(1.0)
            })
        })
        .collect();
    let eff: Vec<Vec<u8>> = (0..layers)
        .map(|p| crate::core::effective_part_mask(&masks[p], Some(&fills[p])))
        .collect();
    let clip = crate::rig::exposure::default_clip_targets(ids, fills);
    let deform_fn = |pt: crate::core::Point, id: &str, pose: &crate::core::Pose| {
        crate::core::deform(pt, id, lm, pose)
    };
    let mut rep = DoubleLineReport {
        poses: poses.len(),
        ..Default::default()
    };
    let mut per: Vec<(usize, usize)> = vec![(0, 0); layers];
    for pe in poses {
        let r =
            crate::rig::exposure::raster_pose(ids, &eff, meshes, &clip, w, h, &pe.pose, &deform_fn);
        // Line class of an output pixel: None = background; Some((layer, is_line, t15 stroke)).
        let class = |q: usize| -> Option<(usize, bool, Option<u32>)> {
            let t = r.top_layer[q];
            if t < 0 {
                return None;
            }
            let p = t as usize;
            let s = r.top_src[q];
            if let Some(&(k, _, _)) = t15[p].get(&s) {
                return Some((p, true, Some(k)));
            }
            Some((p, orig_line[p][s as usize], None))
        };
        // Is the posed line art of layer `lay` around (x, y) within 30° of parallel to `t`? (PCA of its
        // line pixels within 3px; fewer than 3 = no orientation = not parallel.)
        // Local orientation (unit tangent) of layer `lay`'s posed line art around (x, y): PCA of its
        // line pixels within 3px (fewer than 3 = none).
        let orient_at = |x: i64, y: i64, lay: usize| -> Option<(f64, f64)> {
            let mut pts: Vec<(f64, f64)> = Vec::new();
            for dy in -3..=3i64 {
                for dx in -3..=3i64 {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                        continue;
                    }
                    if let Some((l2, true, _)) = class(yy as usize * w + xx as usize) {
                        if l2 == lay {
                            pts.push((dx as f64, dy as f64));
                        }
                    }
                }
            }
            if pts.len() < 3 {
                return None;
            }
            let nl = pts.len() as f64;
            let (mx, my) = pts.iter().fold((0.0, 0.0), |a, q| (a.0 + q.0, a.1 + q.1));
            let (mx, my) = (mx / nl, my / nl);
            let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
            for q in &pts {
                sxx += (q.0 - mx) * (q.0 - mx);
                sxy += (q.0 - mx) * (q.1 - my);
                syy += (q.1 - my) * (q.1 - my);
            }
            let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
            Some((ang.cos(), ang.sin()))
        };
        // Scan both ways along `nrm` from output pixel q (line of layer p, stroke `k` or original):
        // returns (cross-layer double partner, same-layer parallel).
        let scan = |q: usize,
                    p: usize,
                    k: Option<u32>,
                    nrm: (f64, f64),
                    reach: i64|
         -> (Option<(usize, bool)>, bool) {
            let (qx, qy) = ((q % w) as f64 + 0.5, (q / w) as f64 + 0.5);
            let tangent = (-nrm.1, nrm.0);
            let mut same = false;
            for sign in [-1.0f64, 1.0] {
                let mut left_stroke = false;
                let mut gap = false;
                for d in 1..=reach {
                    let x = (qx + sign * nrm.0 * d as f64).floor() as i64;
                    let y = (qy + sign * nrm.1 * d as f64).floor() as i64;
                    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                        break;
                    }
                    let Some((pp, is_line, kk)) = class(y as usize * w + x as usize) else {
                        break;
                    };
                    if !left_stroke {
                        if is_line && pp == p && (kk == k || kk.is_none()) {
                            continue;
                        }
                        left_stroke = true;
                    }
                    if !is_line {
                        if pp != p {
                            break;
                        }
                        gap = true;
                        continue;
                    }
                    let parallel = orient_at(x, y, pp).is_some_and(|o| {
                        (o.0 * tangent.0 + o.1 * tangent.1).abs() >= UPPER_PARALLEL_COS
                    });
                    if gap && parallel {
                        if pp != p {
                            return (Some((pp, kk.is_some())), same);
                        } else if kk != k {
                            same = true;
                        }
                    }
                    break;
                }
            }
            (None, same)
        };
        // Baseline: the same pattern formed by the original (non-T15) line art of lower layers.
        let mut base_dbl: std::collections::HashSet<usize> = Default::default();
        for q in 0..n {
            let Some((p, true, None)) = class(q) else {
                continue;
            };
            let (x, y) = ((q % w) as i64, (q / w) as i64);
            let Some(t) = orient_at(x, y, p) else {
                continue;
            };
            let reach = (2.0 * width[p] + 3.0).ceil() as i64;
            if scan(q, p, None, (-t.1, t.0), reach).0.is_some() {
                base_dbl.insert(q);
            }
        }
        rep.baseline_cross_layer_double_px += base_dbl.len();
        for q in 0..n {
            let Some((p, true, Some(k))) = class(q) else {
                continue;
            };
            rep.exposed_line_px += 1;
            let s = r.top_src[q];
            let (_, nx, ny) = t15[p][&s];
            let reach = (2.0 * width[p] + 3.0).ceil() as i64;
            let (other, same) = scan(q, p, Some(k), (nx as f64, ny as f64), reach);
            if let Some((o, ot)) = other {
                rep.cross_layer_double_px += 1;
                per[p].0 += 1;
                let (x, y) = ((q % w) as i64, (q / w) as i64);
                let near_baseline = (-3..=3i64).any(|dy| {
                    (-3..=3i64).any(|dx| {
                        let (xx, yy) = (x + dx, y + dy);
                        xx >= 0
                            && yy >= 0
                            && xx < w as i64
                            && yy < h as i64
                            && base_dbl.contains(&(yy as usize * w + xx as usize))
                    })
                });
                if near_baseline {
                    rep.cross_layer_double_near_baseline_px += 1;
                }
                if rep.examples.len() < 16 {
                    rep.examples.push(DoubleExample {
                        pose: pe.name.clone(),
                        layer: ids[p].clone(),
                        x: q % w,
                        y: q / w,
                        rest_x: s as usize % w,
                        rest_y: s as usize / w,
                        other: ids[o].clone(),
                        other_is_t15: ot,
                        near_baseline,
                    });
                }
            } else if same {
                rep.same_layer_parallel_px += 1;
                per[p].1 += 1;
            }
        }
    }
    rep.per_layer = (0..layers)
        .filter(|&p| per[p] != (0, 0))
        .map(|p| (ids[p].clone(), per[p].0, per[p].1))
        .collect();
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rig::seams::SeamRequirement;

    fn req(upper: &str, lower: &str, depth: f64) -> SeamRequirement {
        SeamRequirement {
            upper: 0,
            lower: 0,
            upper_id: upper.into(),
            lower_id: lower.into(),
            upper_role: crate::core::infer_part_role(upper, None),
            lower_role: crate::core::infer_part_role(lower, None),
            boundary_points: 0,
            sampled_points: 0,
            poses: 0,
            max_relative_displacement_px: depth,
            max_exposure_depth_px: depth,
            required_underlap_px: (depth * 1.25 + 2.0).ceil() as usize,
            worst_pose: None,
            worst_point: None,
            unsupported_samples: 0,
            coupled_underlap_px: None,
            uncoupled_underlap_px: None,
            uncoupled_max_relative_displacement_px: None,
            adopted: None,
            legacy_eye_underlap_px: None,
        }
    }

    struct Fx {
        w: usize,
        h: usize,
        rgba: Vec<u8>,
        ids: Vec<String>,
        roles: Vec<PartRole>,
        masks: Vec<Vec<u8>>,
        fills: Vec<Vec<u8>>,
        tex: Vec<Option<Vec<u8>>>,
    }

    /// `clothes` (lower) with a vertical dark stroke at x∈[30,33) entering the area under `hair`
    /// (upper, y < 40); the clothes hidden area is a skin-free grey field under the hair.
    fn fixture(upper_alpha: u8, second_stroke: bool) -> Fx {
        let (w, h) = (80usize, 90usize);
        let n = w * h;
        let mut rgba = vec![0u8; n * 4];
        let mut lower = vec![0u8; n];
        let mut fill = vec![0u8; n];
        let mut upper = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let k = i * 4;
                if (10..70).contains(&x) && y >= 10 {
                    if y < 40 {
                        upper[i] = upper_alpha;
                        rgba[k..k + 4].copy_from_slice(&[40, 60, 160, upper_alpha]);
                        fill[i] = 255;
                    } else {
                        lower[i] = 255;
                        let line = (30..33).contains(&x)
                            || (second_stroke && (24..27).contains(&y) && false);
                        let c = if line { [30, 30, 40] } else { [200, 200, 205] };
                        rgba[k..k + 3].copy_from_slice(&c);
                        rgba[k + 3] = 255;
                    }
                }
            }
        }
        let mut tex_l = vec![0u8; n * 4];
        for i in 0..n {
            if lower[i] > 0 {
                tex_l[i * 4..i * 4 + 4].copy_from_slice(&[
                    rgba[i * 4],
                    rgba[i * 4 + 1],
                    rgba[i * 4 + 2],
                    255,
                ]);
            } else if fill[i] > 0 {
                tex_l[i * 4..i * 4 + 4].copy_from_slice(&[200, 200, 205, 255]);
            }
        }
        let mut tex_u = vec![0u8; n * 4];
        for i in 0..n {
            if upper[i] > 0 {
                tex_u[i * 4..i * 4 + 4].copy_from_slice(&[40, 60, 160, 255]);
            }
        }
        Fx {
            w,
            h,
            rgba,
            ids: vec!["clothes".into(), "hair_side_L".into()],
            roles: vec![PartRole::Clothes, PartRole::HairSideL],
            masks: vec![lower, upper],
            fills: vec![fill, vec![0u8; n]],
            tex: vec![Some(tex_l), Some(tex_u)],
        }
    }

    fn run(fx: &mut Fx, depth: f64) -> Vec<Option<LineArtReport>> {
        let reqs = vec![req("hair_side_L", "clothes", depth)];
        let scene = LineArtScene {
            ids: &fx.ids,
            roles: &fx.roles,
            masks: &fx.masks,
            fills: &fx.fills,
            rgba: &fx.rgba,
            w: fx.w,
            h: fx.h,
            held: vec![None, None],
            seam_requirements: &reqs,
            hh: 100.0,
        };
        synthesize_lineart(&scene, &mut fx.tex)
    }

    #[test]
    fn stroke_entering_the_hidden_area_is_extended_to_the_exposure_depth_with_its_width_and_colour()
    {
        let mut fx = fixture(255, false);
        let reps = run(&mut fx, 12.0);
        let rep = reps[0].as_ref().expect("clothes report");
        assert_eq!(rep.strokes.touching_hidden, 1, "{rep:?}");
        assert_eq!(rep.strokes.extended, 1, "{:?}", rep.strokes);
        assert_eq!(rep.strokes.skipped(), 0);
        let t = fx.tex[0].as_ref().unwrap();
        let w = fx.w;
        // Just under the seam (depth 3) the stroke continues at x 30..33, dark, 3px wide.
        for x in 29..34 {
            let i = 36 * w + x;
            let dark = t[i * 4] < 120;
            assert_eq!(
                dark,
                (30..33).contains(&x),
                "x={x} rgb={:?}",
                &t[i * 4..i * 4 + 3]
            );
        }
        // Beyond depth + slack (40 − 12 − 2 = 26 → y ≤ 24) nothing is drawn.
        for y in 10..24 {
            for x in 10..70 {
                let i = y * w + x;
                assert_eq!(
                    &t[i * 4..i * 4 + 3],
                    &[200, 200, 205],
                    "drew beyond depth at ({x},{y})"
                );
            }
        }
        // Visible pixels untouched.
        for i in 0..fx.w * fx.h {
            if fx.masks[0][i] > 0 {
                assert_eq!(&t[i * 4..i * 4 + 3], &fx.rgba[i * 4..i * 4 + 3]);
            }
        }
        assert_eq!(rep.rest_gate.rejected, 0);
        assert!(rep.rest_gate.accepted > 20);
    }

    #[test]
    fn hidden_texels_under_a_semi_transparent_upper_keep_the_rest_composite_bit_identical() {
        let mut fx = fixture(200, false);
        let before: Vec<[u8; 4]> = {
            let reqs: Vec<SeamRequirement> = Vec::new();
            let scene = LineArtScene {
                ids: &fx.ids,
                roles: &fx.roles,
                masks: &fx.masks,
                fills: &fx.fills,
                rgba: &fx.rgba,
                w: fx.w,
                h: fx.h,
                held: vec![None, None],
                seam_requirements: &reqs,
                hh: 100.0,
            };
            (0..fx.w * fx.h)
                .map(|i| rest_px(&scene, &fx.tex, i, &[]))
                .collect()
        };
        let reps = run(&mut fx, 12.0);
        let rep = reps[0].as_ref().unwrap();
        let reqs: Vec<SeamRequirement> = Vec::new();
        let scene = LineArtScene {
            ids: &fx.ids,
            roles: &fx.roles,
            masks: &fx.masks,
            fills: &fx.fills,
            rgba: &fx.rgba,
            w: fx.w,
            h: fx.h,
            held: vec![None, None],
            seam_requirements: &reqs,
            hh: 100.0,
        };
        for (i, b) in before.iter().enumerate() {
            assert_eq!(
                rest_px(&scene, &fx.tex, i, &[]),
                *b,
                "rest composite changed at {i}"
            );
        }
        // Under a 200/255 upper, the full line colour would change the composite; the gate keeps
        // what it can (clamped) or rejects.
        assert_eq!(rep.rest_gate.accepted, 0, "{:?}", rep.rest_gate);
        assert!(rep.rest_gate.clamped + rep.rest_gate.rejected > 0);
    }

    #[test]
    fn unexposed_seam_and_feature_seams_draw_nothing() {
        let mut fx = fixture(255, false);
        let reps = run(&mut fx, 0.0 - DEPTH_SLACK_PX - 1.0);
        let rep = reps[0].as_ref().unwrap();
        assert_eq!(rep.strokes.unexposed, 1, "{:?}", rep.strokes);
        assert_eq!(rep.stroke_texels, 0);
        let mut fx = fixture(255, false);
        fx.ids[1] = "browR".into();
        fx.roles[1] = PartRole::Brow;
        let reqs = vec![req("browR", "clothes", 12.0)];
        let scene = LineArtScene {
            ids: &fx.ids,
            roles: &fx.roles,
            masks: &fx.masks,
            fills: &fx.fills,
            rgba: &fx.rgba,
            w: fx.w,
            h: fx.h,
            held: vec![None, None],
            seam_requirements: &reqs,
            hh: 100.0,
        };
        let reps = synthesize_lineart(&scene, &mut fx.tex);
        let rep = reps[0].as_ref().unwrap();
        assert_eq!(rep.strokes.feature_seam, 1, "{:?}", rep.strokes);
        assert_eq!(rep.stroke_texels, 0);
        assert!(reps[1].is_none(), "brow layer takes no T15 lines");
    }

    #[test]
    fn silhouette_outline_continues_as_a_hidden_contour_with_the_same_width_and_colour() {
        // Face disc (visible lower half, hidden upper half under hair) with a 2px dark outline on
        // its visible silhouette; the hidden silhouette continues it.
        let (w, h) = (100usize, 100usize);
        let n = w * h;
        let mut rgba = vec![0u8; n * 4];
        let mut face = vec![0u8; n];
        let mut fill = vec![0u8; n];
        let mut hair = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let d = ((x as f64 + 0.5 - 50.0).powi(2) + (y as f64 + 0.5 - 50.0).powi(2)).sqrt();
                if y < 45 && (8..92).contains(&x) && y >= 4 {
                    hair[i] = 255;
                    rgba[i * 4..i * 4 + 4].copy_from_slice(&[90, 60, 40, 255]);
                    if d <= 30.0 {
                        fill[i] = 255;
                    }
                } else if d <= 30.0 {
                    face[i] = 255;
                    let c = if d > 28.0 {
                        [60, 40, 40]
                    } else {
                        [250, 225, 210]
                    };
                    rgba[i * 4..i * 4 + 3].copy_from_slice(&c);
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        let mut tex = vec![0u8; n * 4];
        for i in 0..n {
            if face[i] > 0 {
                tex[i * 4..i * 4 + 4].copy_from_slice(&[
                    rgba[i * 4],
                    rgba[i * 4 + 1],
                    rgba[i * 4 + 2],
                    255,
                ]);
            } else if fill[i] > 0 {
                tex[i * 4..i * 4 + 4].copy_from_slice(&[250, 225, 210, 255]);
            }
        }
        let ids = vec!["face".to_string(), "hair_side_R".to_string()];
        let roles = vec![PartRole::Face, PartRole::HairSideR];
        let masks = vec![face, hair];
        let fills = vec![fill, vec![0u8; n]];
        let reqs = vec![req("hair_side_R", "face", 10.0)];
        fn masks_hair_any(rgba: &[u8], i: usize) -> bool {
            rgba[i * 4..i * 4 + 4] == [90, 60, 40, 255]
        }
        let hair_tex: Vec<u8> = (0..n)
            .flat_map(|i| {
                if masks_hair_any(&rgba, i) {
                    [90u8, 60, 40, 255]
                } else {
                    [0u8; 4]
                }
            })
            .collect();
        let mut textures = vec![Some(tex), Some(hair_tex)];
        let scene = LineArtScene {
            ids: &ids,
            roles: &roles,
            masks: &masks,
            fills: &fills,
            rgba: &rgba,
            w,
            h,
            held: vec![None, None],
            seam_requirements: &reqs,
            hh: 100.0,
        };
        let reps = synthesize_lineart(&scene, &mut textures);
        let rep = reps[0].as_ref().unwrap();
        assert!(rep.outlined, "{rep:?}");
        assert!(rep.contour_length_px > 8, "{}", rep.contour_length_px);
        assert!(rep.junctions >= 2);
        assert!(rep.junction_width_diff_max.unwrap() <= 1.0, "{rep:?}");
        assert!(rep.junction_de_max.unwrap() < 3.0, "{rep:?}");
        let t = textures[0].as_ref().unwrap();
        // Hidden silhouette pixel near the visible junction at the disc's left edge, y = 42.
        let i = 42 * w + 21;
        assert!(fills[0][i] > 0 && masks[0][i] == 0);
        assert!(
            t[i * 4] < 120,
            "contour not drawn: {:?}",
            &t[i * 4..i * 4 + 3]
        );
        // Interior of the hidden disc stays skin; and nothing beyond the exposure depth (y < 33).
        let c = 40 * w + 50;
        assert_eq!(&t[c * 4..c * 4 + 3], &[250, 225, 210]);
        for y in 0..30 {
            for x in 0..w {
                let i = y * w + x;
                if fills[0][i] > 0 {
                    assert_eq!(
                        &t[i * 4..i * 4 + 3],
                        &[250, 225, 210],
                        "beyond depth at ({x},{y})"
                    );
                }
            }
        }
    }

    #[test]
    fn edt_matches_brute_force() {
        let (bw, bh) = (17usize, 11usize);
        let src: Vec<bool> = (0..bw * bh).map(|i| i % 37 == 5 || i == 100).collect();
        let d = edt_sq(bw, bh, &src);
        for y in 0..bh {
            for x in 0..bw {
                let mut best = f32::MAX;
                for (j, &s) in src.iter().enumerate() {
                    if s {
                        let (sx, sy) = ((j % bw) as f32, (j / bw) as f32);
                        best = best.min((sx - x as f32).powi(2) + (sy - y as f32).powi(2));
                    }
                }
                assert_eq!(d[y * bw + x], best, "({x},{y})");
            }
        }
    }
}
