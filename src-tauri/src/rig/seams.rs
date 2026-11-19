//! T13a — seam contracts (`SeamGraph`), the shared pose envelope, `deform`-sampled per-seam
//! underlaps, and the seam coverage check.
//!
//! Why this exists: T8 computed hidden-area (`fillMask`) underlaps by copying `deform`
//! coefficients into closed-form formulas (`core::part_overlap_margins`). Any `deform` change
//! silently desynchronized the margins. Here the required underlap of each seam is *measured*:
//! every visible boundary point between an upper and a lower layer is pushed through the real
//! `deform(upper)` and `deform(lower)` for every pose of the shared envelope, and the depth by
//! which the lower layer must continue under the upper layer is read off the masks. No `deform`
//! coefficient appears in this module.
//!
//! - Seam data is derived (never stored in the project file) and must be recomputed whenever the
//!   visible masks change, including user-approved edits.
//! - The pose envelope lives in `src/rig/pose-envelope.json`, embedded here with `include_str!`
//!   and imported by TS (`src/rig/poseEnvelope.ts`), so Rust and TS read the same table. Its
//!   ranges must equal `PARAM_DEFS` (no shrinking); the TS test enforces that.
//! - `core::part_overlap_margins` remains as the fallback when a seam is missing (landmarks only).
//! - `coupling` is filled by `annotate_seam_couplings` (T13b coupled seam deformation; the weights
//!   come from `core::seam_coupling_weights`, the same function `deform` uses).

use crate::core::{deform, effective_part_mask, HeadFrame, Landmarks, PartRole, Point, Pose};
use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

/// Shared deterministic pose envelope (Rust embeds the same file TS imports).
pub const POSE_ENVELOPE_JSON: &str = include_str!("../../../src/rig/pose-envelope.json");

/// Multiplicative safety factor on the measured exposure depth. Not a `deform` coefficient: it
/// absorbs mesh-triangle interpolation vs. point deformation, boundary midpoint rounding, and the
/// first-order lower-layer inverse used when sampling.
pub const SEAM_SAFETY_SCALE: f64 = 1.25;
/// Additive safety pad (px) so even motionless seams keep an anti-aliasing underlap.
pub const SEAM_SAFETY_PAD_PX: f64 = 2.0;
/// Boundary samples per seam used when computing requirements inside template synthesis.
pub const DEFAULT_MAX_SAMPLES_PER_SEAM: usize = 384;
/// Relative motion below this (px) cannot expose a pixel and is skipped.
const MIN_RELATIVE_MOTION_PX: f64 = 0.5;
/// Maximum number of violation records kept in a coverage report (counts are always complete).
pub const MAX_REPORTED_VIOLATIONS: usize = 64;

// ---------------------------------------------------------------------------------------------
// Pose envelope
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvelopePose {
    pub name: String,
    pub group: String,
    pub pose: Pose,
}

#[derive(Debug, Clone)]
pub struct PoseEnvelope {
    pub version: u32,
    /// Parameter ranges (`PARAM_DEFS` min/max) keyed by the camelCase `Pose` field name.
    pub ranges: BTreeMap<String, (f64, f64)>,
    pub poses: Vec<EnvelopePose>,
}

#[derive(Deserialize)]
struct RawEnvelope {
    version: u32,
    ranges: BTreeMap<String, [f64; 2]>,
    poses: Vec<RawEnvelopePose>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEnvelopePose {
    name: String,
    group: String,
    params: serde_json::Map<String, serde_json::Value>,
    /// T10 expression poses carry the eye shape input the app evaluates for that expression.
    #[serde(default)]
    shape: Option<crate::core::ShapeDeformInput>,
}

/// camelCase field names of `Pose` (the same keys TS `Parameters` uses).
pub fn pose_field_names() -> Vec<String> {
    match serde_json::to_value(Pose::default()) {
        Ok(serde_json::Value::Object(m)) => m.keys().cloned().collect(),
        _ => Vec::new(),
    }
}

/// Parses and validates an envelope document: every key must be a `Pose` field, every `Pose`
/// field must have a range, every value must lie inside its range, and names must be unique.
pub fn parse_pose_envelope(json: &str) -> Result<PoseEnvelope, String> {
    let raw: RawEnvelope =
        serde_json::from_str(json).map_err(|e| format!("pose envelope JSON: {e}"))?;
    let fields = pose_field_names();
    let known = |k: &str| fields.iter().any(|f| f == k);
    for k in raw.ranges.keys() {
        if !known(k) {
            return Err(format!("pose envelope: unknown range key `{k}`"));
        }
    }
    for f in &fields {
        if !raw.ranges.contains_key(f) {
            return Err(format!("pose envelope: range missing for Pose field `{f}`"));
        }
    }
    let mut names = HashSet::new();
    let mut poses = Vec::with_capacity(raw.poses.len());
    for p in raw.poses {
        if !names.insert(p.name.clone()) {
            return Err(format!("pose envelope: duplicate pose name `{}`", p.name));
        }
        for (k, v) in &p.params {
            if !known(k) {
                return Err(format!(
                    "pose envelope: pose `{}` has unknown key `{k}`",
                    p.name
                ));
            }
            let x = v
                .as_f64()
                .ok_or_else(|| format!("pose envelope: `{}`.{k} is not a number", p.name))?;
            let [lo, hi] = raw.ranges[k];
            if x < lo || x > hi {
                return Err(format!(
                    "pose envelope: `{}`.{k}={x} outside PARAM_DEFS range [{lo}, {hi}]",
                    p.name
                ));
            }
        }
        let mut pose: Pose = serde_json::from_value(serde_json::Value::Object(p.params))
            .map_err(|e| format!("pose envelope: `{}`: {e}", p.name))?;
        pose.shape = p.shape;
        poses.push(EnvelopePose {
            name: p.name,
            group: p.group,
            pose,
        });
    }
    Ok(PoseEnvelope {
        version: raw.version,
        ranges: raw
            .ranges
            .into_iter()
            .map(|(k, [a, b])| (k, (a, b)))
            .collect(),
        poses,
    })
}

thread_local! {
    static BASE_ENVELOPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with hidden-area synthesis restricted to the pre-T10 envelope (every pose except the
/// T10 `eyeSeal` group). Used only to build the rest-invariance baseline of template synthesis
/// (`templates::synthesize_template_bank_for_stage`); thread-local like `with_seam_coupling`.
pub fn with_base_envelope<R>(f: impl FnOnce() -> R) -> R {
    let prev = BASE_ENVELOPE.with(|c| c.replace(true));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            BASE_ENVELOPE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(prev);
    f()
}

/// Whether `with_base_envelope` is active on this thread.
pub fn base_envelope_enabled() -> bool {
    BASE_ENVELOPE.with(|c| c.get())
}

/// Envelope used by hidden-area synthesis: the shared envelope, or (inside `with_base_envelope`)
/// the pre-T10 subset without the `eyeSeal` group.
pub fn hidden_area_envelope() -> Vec<EnvelopePose> {
    let env = &pose_envelope().poses;
    if base_envelope_enabled() {
        env.iter().filter(|p| p.group != "eyeSeal").cloned().collect()
    } else {
        env.clone()
    }
}

/// The embedded shared envelope (parsed once).
pub fn pose_envelope() -> &'static PoseEnvelope {
    static ENV: OnceLock<PoseEnvelope> = OnceLock::new();
    ENV.get_or_init(|| {
        parse_pose_envelope(POSE_ENVELOPE_JSON)
            .expect("embedded src/rig/pose-envelope.json must be valid")
    })
}

// ---------------------------------------------------------------------------------------------
// Seam graph
// ---------------------------------------------------------------------------------------------

/// T13b coupled-deformation kinds (see `core::deform_with_shapes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SeamCouplingKind {
    /// Neck top follows the head field near the chin line (`face`/`neck`).
    NeckHead,
    /// Neck base and the collar column share one field (`clothes`|`body` / `neck`).
    Collar,
    /// Hair blends toward the head field inside its root band (hair / `face` or hair / hair).
    HairRoot,
    /// B2: hair below the chin line exchanges its head-follow field for the torso field while
    /// keeping its own sway displacement (`hair` / `body`|`clothes`|`hands`).
    HairBody,
}

/// T13b coupling attached to a seam. Weights are the mean `core::seam_coupling_weights` blend of
/// each side toward its partner field over the seam boundary (`0` = that side keeps its own field;
/// a hair seam lying entirely beyond the root band reports `0` on both sides).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeamCoupling {
    pub kind: SeamCouplingKind,
    pub upper_weight: f64,
    pub lower_weight: f64,
    /// Distance (px) over which the coupling falls from its seam-line value to 0.
    pub falloff_px: f64,
}

fn coupling_kind(upper: PartRole, lower: PartRole) -> Option<SeamCouplingKind> {
    use PartRole as R;
    let is_hair = |r: PartRole| matches!(r, R::HairFront | R::HairSideL | R::HairSideR | R::HairBack);
    match (upper, lower) {
        (R::Face, R::Neck) | (R::Neck, R::Face) => Some(SeamCouplingKind::NeckHead),
        (R::Clothes | R::Body, R::Neck) | (R::Neck, R::Clothes | R::Body) => {
            Some(SeamCouplingKind::Collar)
        }
        (a, b) if (is_hair(a) && (is_hair(b) || b == R::Face)) || (is_hair(b) && a == R::Face) => {
            Some(SeamCouplingKind::HairRoot)
        }
        (a, b)
            if (is_hair(a) && matches!(b, R::Body | R::Clothes | R::Hands))
                || (is_hair(b) && matches!(a, R::Body | R::Clothes | R::Hands)) =>
        {
            Some(SeamCouplingKind::HairBody)
        }
        _ => None,
    }
}

/// Blend weight of layer `id` toward its partner field for a coupling kind at `p`.
fn side_weight(kind: SeamCouplingKind, p: Point, id: &str, lm: &Landmarks) -> f64 {
    let w = crate::core::seam_coupling_weights(p, id, lm);
    match kind {
        // Only the neck side is blended (toward the head field); `face` keeps its own field.
        SeamCouplingKind::NeckHead => w.neck_head,
        // The neck side blends toward the body field; body/clothes blend toward the coupled neck.
        SeamCouplingKind::Collar => {
            if crate::core::infer_part_role(id, None) == PartRole::Neck {
                w.neck_body
            } else {
                w.collar
            }
        }
        SeamCouplingKind::HairRoot => w.hair_root,
        SeamCouplingKind::HairBody => w.hair_body,
    }
}

/// Fills `SeamContract.coupling` for every coupled seam (T13b + B2). Weights are read from
/// `core::seam_coupling_weights`, the same function `deform` uses, so they cannot drift.
pub fn annotate_seam_couplings(graph: &mut SeamGraph, lm: &Landmarks) {
    let hf = HeadFrame::from_landmarks(lm, 0, 0);
    let hh = hf.hh;
    for seam in &mut graph.seams {
        let Some(kind) = coupling_kind(seam.upper_role, seam.lower_role) else {
            seam.coupling = None;
            continue;
        };
        let pts = seam.boundary_samples(256);
        let mean = |id: &str| {
            if pts.is_empty() {
                0.0
            } else {
                pts.iter().map(|&p| side_weight(kind, p, id, lm)).sum::<f64>() / pts.len() as f64
            }
        };
        let falloff_px = match kind {
            SeamCouplingKind::NeckHead => hh * crate::core::NECK_HEAD_COUPLING_FALLOFF_HH,
            SeamCouplingKind::Collar => hh * crate::core::NECK_SPAN_HH,
            SeamCouplingKind::HairRoot => hh * crate::core::HAIR_ROOT_COUPLING_BAND_HH,
            SeamCouplingKind::HairBody => {
                if lm.torso_frame.as_deref().is_some_and(|f| !f.is_empty()) {
                    if pts.is_empty() {
                        hh * crate::core::HAIR_BODY_MIN_BAND_HH
                    } else {
                        pts.iter()
                            .filter_map(|&p| crate::core::hair_body_band(&hf, lm, p))
                            .map(|(y0, y1)| y1 - y0)
                            .sum::<f64>()
                            / pts.len() as f64
                    }
                } else {
                    0.0
                }
            }
        };
        seam.coupling = Some(SeamCoupling {
            kind,
            upper_weight: mean(&seam.upper_id),
            lower_weight: mean(&seam.lower_id),
            falloff_px,
        });
    }
}

/// One stacked layer pair with a shared visible boundary. `upper` draws above `lower`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeamContract {
    pub upper: usize,
    pub lower: usize,
    pub upper_id: String,
    pub lower_id: String,
    pub upper_role: PartRole,
    pub lower_role: PartRole,
    /// Visible boundary polylines in canvas pixels (midpoints between 4-adjacent pixels owned by
    /// `upper` and `lower`; pixel `(x, y)` is centred at integer coordinates like the template code).
    pub boundary: Vec<Vec<Point>>,
    pub coupling: Option<SeamCoupling>,
}

impl SeamContract {
    pub fn boundary_len(&self) -> usize {
        self.boundary.iter().map(Vec::len).sum()
    }

    /// Evenly strided boundary points (all points when `max_points` is 0 or large enough).
    pub fn boundary_samples(&self, max_points: usize) -> Vec<Point> {
        let total = self.boundary_len();
        let stride = if max_points == 0 || total <= max_points {
            1
        } else {
            total.div_ceil(max_points)
        };
        self.boundary
            .iter()
            .flatten()
            .step_by(stride.max(1))
            .copied()
            .collect()
    }
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeamGraph {
    pub seams: Vec<SeamContract>,
}

impl SeamGraph {
    pub fn find(&self, upper_id: &str, lower_id: &str) -> Option<&SeamContract> {
        self.seams
            .iter()
            .find(|s| s.upper_id == upper_id && s.lower_id == lower_id)
    }
}

/// Topmost visible owner per pixel (`-1` when no layer is visible). Layers are bottom-to-top.
pub fn visible_owner(visible_masks: &[Vec<u8>], n: usize) -> Vec<i32> {
    let mut owner = vec![-1i32; n];
    for (p, m) in visible_masks.iter().enumerate() {
        for (i, &a) in m.iter().enumerate().take(n) {
            if a > 0 {
                owner[i] = p as i32;
            }
        }
    }
    owner
}

/// Extracts every stacked pair with a shared visible boundary from bottom-to-top layers.
/// Roles are supplied by the caller (explicit role or `core::infer_part_role`), so variable hair
/// layers (`hair_front_2`, `hair_side_L_3`, ...) and accessories participate like any other layer.
pub fn extract_seam_graph(
    layer_ids: &[String],
    roles: &[PartRole],
    visible_masks: &[Vec<u8>],
    w: usize,
    h: usize,
) -> SeamGraph {
    let n = w * h;
    let owner = visible_owner(visible_masks, n);
    let mut pts: BTreeMap<(usize, usize), Vec<(i64, i64)>> = BTreeMap::new();
    let mut push = |a: i32, b: i32, key: (i64, i64)| {
        let (u, l) = if a > b { (a, b) } else { (b, a) };
        pts.entry((u as usize, l as usize)).or_default().push(key);
    };
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let a = owner[i];
            if a < 0 {
                continue;
            }
            if x + 1 < w {
                let b = owner[i + 1];
                if b >= 0 && b != a {
                    push(a, b, (2 * x as i64 + 1, 2 * y as i64));
                }
            }
            if y + 1 < h {
                let b = owner[i + w];
                if b >= 0 && b != a {
                    push(a, b, (2 * x as i64, 2 * y as i64 + 1));
                }
            }
        }
    }
    let seams = pts
        .into_iter()
        .map(|((u, l), keys)| SeamContract {
            upper: u,
            lower: l,
            upper_id: layer_ids[u].clone(),
            lower_id: layer_ids[l].clone(),
            upper_role: roles[u],
            lower_role: roles[l],
            boundary: chain_polylines(keys),
            coupling: None,
        })
        .collect();
    SeamGraph { seams }
}

/// Links half-pixel boundary midpoints (doubled integer coords) into ordered polylines.
fn chain_polylines(keys: Vec<(i64, i64)>) -> Vec<Vec<Point>> {
    let index: HashMap<(i64, i64), usize> = keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut offsets: Vec<(i64, i64)> = Vec::new();
    for dy in -2i64..=2 {
        for dx in -2i64..=2 {
            if (dx, dy) != (0, 0) {
                offsets.push((dx, dy));
            }
        }
    }
    offsets.sort_by_key(|&(dx, dy)| (dx * dx + dy * dy, dy, dx));
    let mut visited = vec![false; keys.len()];
    let walk = |start: usize, visited: &mut Vec<bool>| -> Vec<usize> {
        let mut out = Vec::new();
        let mut cur = start;
        loop {
            let (cx, cy) = keys[cur];
            let next = offsets.iter().find_map(|&(dx, dy)| {
                index
                    .get(&(cx + dx, cy + dy))
                    .copied()
                    .filter(|&j| !visited[j])
            });
            match next {
                Some(j) => {
                    visited[j] = true;
                    out.push(j);
                    cur = j;
                }
                None => break,
            }
        }
        out
    };
    let mut lines = Vec::new();
    for s in 0..keys.len() {
        if visited[s] {
            continue;
        }
        visited[s] = true;
        let fwd = walk(s, &mut visited);
        let bwd = walk(s, &mut visited);
        let to_pt = |j: usize| Point {
            x: keys[j].0 as f64 * 0.5,
            y: keys[j].1 as f64 * 0.5,
        };
        let mut line: Vec<Point> = bwd.iter().rev().map(|&j| to_pt(j)).collect();
        line.push(to_pt(s));
        line.extend(fwd.iter().map(|&j| to_pt(j)));
        lines.push(line);
    }
    lines
}

// ---------------------------------------------------------------------------------------------
// Numeric requirements
// ---------------------------------------------------------------------------------------------

#[inline]
fn dv(p: Point) -> DVec2 {
    DVec2::new(p.x, p.y)
}

#[inline]
pub(crate) fn pixel(p: DVec2, w: usize, h: usize) -> Option<usize> {
    let x = p.x.round();
    let y = p.y.round();
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
        None
    } else {
        Some(y as usize * w + x as usize)
    }
}

/// Chamfer (1, √2) distance from each pixel to the nearest `mask > 0` pixel (0 inside).
pub(crate) fn outside_distance(mask: &[u8], w: usize, h: usize) -> Vec<f32> {
    let inf = f32::MAX / 4.0;
    let d1 = 1.0f32;
    let d2 = std::f32::consts::SQRT_2;
    let mut d: Vec<f32> = mask
        .iter()
        .map(|&a| if a > 0 { 0.0 } else { inf })
        .collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut v = d[i];
            if x > 0 {
                v = v.min(d[i - 1] + d1);
            }
            if y > 0 {
                v = v.min(d[i - w] + d1);
                if x > 0 {
                    v = v.min(d[i - w - 1] + d2);
                }
                if x + 1 < w {
                    v = v.min(d[i - w + 1] + d2);
                }
            }
            d[i] = v;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            let mut v = d[i];
            if x + 1 < w {
                v = v.min(d[i + 1] + d1);
            }
            if y + 1 < h {
                v = v.min(d[i + w] + d1);
                if x + 1 < w {
                    v = v.min(d[i + w + 1] + d2);
                }
                if x > 0 {
                    v = v.min(d[i + w - 1] + d2);
                }
            }
            d[i] = v;
        }
    }
    d
}

/// Measured underlap requirement for one seam over the pose envelope.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeamRequirement {
    pub upper: usize,
    pub lower: usize,
    pub upper_id: String,
    pub lower_id: String,
    pub upper_role: PartRole,
    pub lower_role: PartRole,
    pub boundary_points: usize,
    pub sampled_points: usize,
    pub poses: usize,
    /// max |deform(upper)(b) − deform(lower)(b)| over sampled boundary points × poses.
    pub max_relative_displacement_px: f64,
    /// max distance (outside the lower visible mask) over every supported sample of the exposed
    /// band `b → x_l` (`x_l` = rest point of `lower` that lands on the moved upper boundary;
    /// see `walk_exposed_band`). A sample is supported when it is foreground and owned at rest
    /// by `lower` or a layer above it; the same band and predicate drive the coverage check.
    pub max_exposure_depth_px: f64,
    /// `ceil(max_exposure_depth_px · SEAM_SAFETY_SCALE + SEAM_SAFETY_PAD_PX)`.
    pub required_underlap_px: usize,
    pub worst_pose: Option<String>,
    pub worst_point: Option<Point>,
    /// Moving boundary-point × pose pairs whose whole exposed band has no supported sample
    /// (background or layers below `lower` only).
    pub unsupported_samples: usize,
    /// T13b hidden-area merge (`hidden_area_seam_requirements`): underlap measured with the coupled
    /// runtime `deform` and with the uncoupled field. `required_underlap_px` is their max so that
    /// coupling never shrinks a hidden area (PSD hand-off: Cubism deformers move independently).
    /// `None` for plain single-field measurements.
    pub coupled_underlap_px: Option<usize>,
    pub uncoupled_underlap_px: Option<usize>,
    pub uncoupled_max_relative_displacement_px: Option<f64>,
    /// Which field set `required_underlap_px`: `coupled`, `uncoupled`, `equal`, or (T10) `legacyEye`.
    pub adopted: Option<&'static str>,
    /// T10: underlap measured with the pre-T10 eye formulas (`core::with_legacy_eye_rig`) for seams
    /// that involve an eye layer. The hidden area never shrinks because of the sealed eye rig: an eye
    /// seam keeps the max of both, and a white/iris *lower* layer keeps the legacy value (its texels
    /// outside the aperture fold under the lid, so the folded field has no inverse to measure).
    pub legacy_eye_underlap_px: Option<usize>,
}

/// Requirements for every seam using the real `core::deform`.
#[allow(clippy::too_many_arguments)]
pub fn seam_requirements(
    graph: &SeamGraph,
    visible_masks: &[Vec<u8>],
    fg_alpha: &[u8],
    lm: &Landmarks,
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
    max_samples_per_seam: usize,
) -> Vec<SeamRequirement> {
    seam_requirements_with(
        graph,
        visible_masks,
        fg_alpha,
        w,
        h,
        poses,
        max_samples_per_seam,
        |p, id, pose| deform(p, id, lm, pose),
    )
}

/// Requirements for every seam with an injected deformation `deform_fn(point, layer_id, pose)`.
/// The function is the only source of motion; nothing about `deform` is restated here.
#[allow(clippy::too_many_arguments)]
pub fn seam_requirements_with<D>(
    graph: &SeamGraph,
    visible_masks: &[Vec<u8>],
    fg_alpha: &[u8],
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
    max_samples_per_seam: usize,
    deform_fn: D,
) -> Vec<SeamRequirement>
where
    D: Fn(Point, &str, &Pose) -> Point,
{
    let n = w * h;
    let owner = visible_owner(visible_masks, n);
    let mut dist_cache: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut out = Vec::with_capacity(graph.seams.len());
    for seam in &graph.seams {
        let dist = dist_cache
            .entry(seam.lower)
            .or_insert_with(|| outside_distance(&visible_masks[seam.lower], w, h));
        let samples = seam.boundary_samples(max_samples_per_seam);
        let mut max_rel = 0.0f64;
        let mut max_depth = 0.0f64;
        let mut worst: Option<(String, Point)> = None;
        let mut unsupported = 0usize;
        let lower = seam.lower as i32;
        for pe in poses {
            let f_lower =
                |x: DVec2| dv(deform_fn(Point::from(x), seam.lower_id.as_str(), &pe.pose));
            for &b in &samples {
                let bv = dv(b);
                let yu = dv(deform_fn(b, seam.upper_id.as_str(), &pe.pose));
                let rl = (yu - f_lower(bv)).length();
                if rl > max_rel {
                    max_rel = rl;
                }
                let walk = walk_exposed_band(
                    bv,
                    yu,
                    &f_lower,
                    fg_alpha,
                    &owner,
                    seam.lower,
                    w,
                    h,
                    |_, _, k| {
                        let depth = if owner[k] == lower {
                            0.0
                        } else {
                            dist[k] as f64
                        };
                        if depth > max_depth {
                            max_depth = depth;
                            worst = Some((pe.name.clone(), b));
                        }
                        true
                    },
                );
                if matches!(walk, BandWalk::Walked { supported: 0 }) {
                    unsupported += 1;
                }
            }
        }
        let required = (max_depth * SEAM_SAFETY_SCALE + SEAM_SAFETY_PAD_PX).ceil() as usize;
        out.push(SeamRequirement {
            upper: seam.upper,
            lower: seam.lower,
            upper_id: seam.upper_id.clone(),
            lower_id: seam.lower_id.clone(),
            upper_role: seam.upper_role,
            lower_role: seam.lower_role,
            boundary_points: seam.boundary_len(),
            sampled_points: samples.len(),
            poses: poses.len(),
            max_relative_displacement_px: max_rel,
            max_exposure_depth_px: max_depth,
            required_underlap_px: required,
            worst_pose: worst.as_ref().map(|(n, _)| n.clone()),
            worst_point: worst.map(|(_, p)| p),
            unsupported_samples: unsupported,
            coupled_underlap_px: None,
            uncoupled_underlap_px: None,
            uncoupled_max_relative_displacement_px: None,
            adopted: None,
            legacy_eye_underlap_px: None,
        });
    }
    out
}

/// Max relative displacement between two layers at the given points over the poses, plus the
/// pose that produced it. Used by the Rust/TS parity fixture.
pub fn max_relative_displacement<D>(
    points: &[Point],
    upper_id: &str,
    lower_id: &str,
    poses: &[EnvelopePose],
    deform_fn: D,
) -> (f64, Option<String>)
where
    D: Fn(Point, &str, &Pose) -> Point,
{
    let mut best = (0.0f64, None);
    for pe in poses {
        for &b in points {
            let r = dv(deform_fn(b, upper_id, &pe.pose)) - dv(deform_fn(b, lower_id, &pe.pose));
            let rl = r.length();
            if rl > best.0 + 1e-12 {
                best = (rl, Some(pe.name.clone()));
            }
        }
    }
    best
}

/// Requirements for hidden-area (underlap) generation: per seam, the max of the requirement with the
/// coupled runtime `deform` and with the uncoupled field (`core::with_seam_coupling(false, …)`).
/// Measurement fields (relative displacement, depth, worst pose) are the coupled runtime's; the
/// uncoupled values and the adopted side are recorded. When coupling is disabled on this thread the
/// two fields are identical and the plain uncoupled requirements are returned.
#[allow(clippy::too_many_arguments)]
pub fn hidden_area_seam_requirements(
    graph: &SeamGraph,
    visible_masks: &[Vec<u8>],
    fg_alpha: &[u8],
    lm: &Landmarks,
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
    max_samples_per_seam: usize,
) -> Vec<SeamRequirement> {
    let current = coupled_hidden_area_requirements(
        graph,
        visible_masks,
        fg_alpha,
        lm,
        w,
        h,
        poses,
        max_samples_per_seam,
    );
    if crate::core::legacy_eye_rig_enabled() {
        return current;
    }
    // T10: eye seams also measured with the pre-T10 eye formulas (never shrink a hidden area).
    let is_eye = |id: &str| crate::core::eye_layer_side(id).is_some();
    let eye_graph = SeamGraph {
        seams: graph
            .seams
            .iter()
            .filter(|s| is_eye(&s.upper_id) || is_eye(&s.lower_id))
            .cloned()
            .collect(),
    };
    if eye_graph.seams.is_empty() {
        return current;
    }
    let legacy = crate::core::with_legacy_eye_rig(true, || {
        coupled_hidden_area_requirements(
            &eye_graph,
            visible_masks,
            fg_alpha,
            lm,
            w,
            h,
            poses,
            max_samples_per_seam,
        )
    });
    current
        .into_iter()
        .map(|mut r| {
            if let Some(l) = legacy
                .iter()
                .find(|l| l.upper_id == r.upper_id && l.lower_id == r.lower_id)
            {
                let lr = l.required_underlap_px;
                r.legacy_eye_underlap_px = Some(lr);
                let interior_lower = crate::core::is_eye_interior_layer(&r.lower_id)
                    || r.lower_id.ends_with("_white")
                    || r.lower_id.ends_with("_iris");
                if interior_lower || lr > r.required_underlap_px {
                    if lr != r.required_underlap_px {
                        r.adopted = Some("legacyEye");
                    }
                    r.required_underlap_px = lr;
                }
            }
            r
        })
        .collect()
}

/// T13b coupled/uncoupled merge of `seam_requirements` (see `hidden_area_seam_requirements`).
#[allow(clippy::too_many_arguments)]
fn coupled_hidden_area_requirements(
    graph: &SeamGraph,
    visible_masks: &[Vec<u8>],
    fg_alpha: &[u8],
    lm: &Landmarks,
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
    max_samples_per_seam: usize,
) -> Vec<SeamRequirement> {
    let run = || seam_requirements(graph, visible_masks, fg_alpha, lm, w, h, poses, max_samples_per_seam);
    if !crate::core::seam_coupling_enabled() {
        return run();
    }
    let coupled = run();
    let uncoupled = crate::core::with_seam_coupling(false, run);
    coupled
        .into_iter()
        .zip(uncoupled)
        .map(|(mut c, u)| {
            let (cr, ur) = (c.required_underlap_px, u.required_underlap_px);
            c.coupled_underlap_px = Some(cr);
            c.uncoupled_underlap_px = Some(ur);
            c.uncoupled_max_relative_displacement_px = Some(u.max_relative_displacement_px);
            c.adopted = Some(match cr.cmp(&ur) {
                std::cmp::Ordering::Greater => "coupled",
                std::cmp::Ordering::Less => "uncoupled",
                std::cmp::Ordering::Equal => "equal",
            });
            c.required_underlap_px = cr.max(ur);
            c
        })
        .collect()
}

/// Seam requirements indexed for the hidden-area generator. Queries return `None` when no seam
/// matches, in which case the caller uses the `core::part_overlap_margins` fallback.
#[derive(Debug, Clone, Default)]
pub struct SeamMarginTable {
    pub requirements: Vec<SeamRequirement>,
}

impl SeamMarginTable {
    pub fn new(requirements: Vec<SeamRequirement>) -> Self {
        Self { requirements }
    }

    /// Largest required underlap over seams where `lower` is the lower layer and the upper layer
    /// passes `accept_upper(upper_index, upper_role)`.
    pub fn need<F: Fn(usize, PartRole) -> bool>(
        &self,
        lower: usize,
        accept_upper: F,
    ) -> Option<usize> {
        self.requirements
            .iter()
            .filter(|r| r.lower == lower && accept_upper(r.upper, r.upper_role))
            .map(|r| r.required_underlap_px)
            .max()
    }
}

/// Attribute probe: does this layer's `deform` respond to the global hair-sway parameter?
/// True for hair strands, hats, and hair-mounted accessories; read from `deform` itself rather
/// than from ID prefixes, so it tracks whatever `deform` actually does.
pub fn follows_hair_sway(id: &str, lm: &Landmarks) -> bool {
    let hf = HeadFrame::from_landmarks(lm, 0, 0);
    let probe = Point {
        x: lm.crown.x,
        y: lm.crown.y + hf.hh * 0.9,
    };
    let rest = Pose::default();
    let sway = Pose {
        hair: 1.0,
        ..Pose::default()
    };
    let a = deform(probe, id, lm, &sway);
    let b = deform(probe, id, lm, &rest);
    (a.x - b.x).abs() + (a.y - b.y).abs() > 1e-6
}

// ---------------------------------------------------------------------------------------------
// Coverage check
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CoverageViolation {
    pub upper_id: String,
    pub lower_id: String,
    pub pose: String,
    pub boundary_point: Point,
    /// Rest-space point in the band swept by the moving upper boundary that `lower` misses.
    pub rest_sample: Point,
    /// Where that point lands after posing `lower`.
    pub posed_sample: Point,
    /// Distance from the boundary point to `rest_sample`.
    pub depth_px: f64,
    /// True when no layer at all covers `posed_sample` after posing (background shows).
    pub hole: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeamCoverageSummary {
    pub upper_id: String,
    pub lower_id: String,
    pub checks: usize,
    pub moving_checks: usize,
    pub lower_misses: usize,
    pub holes: usize,
    pub max_miss_depth_px: f64,
    /// Holes per envelope pose group (`axis`, `headBodyOpposite`, `hairPhase`, ...).
    pub hole_pose_groups: BTreeMap<String, usize>,
    /// Pose with the most holes on this seam, and its hole count.
    pub worst_hole_pose: Option<String>,
    pub worst_hole_pose_holes: usize,
    /// Max band depth (px) of a hole sample on this seam.
    pub max_hole_depth_px: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeamCoverageReport {
    pub poses: usize,
    /// Boundary point × pose pairs checked.
    pub checks: usize,
    /// Pairs where the deformed lower effective mask (`mask ∪ fillMask`) fails to cover the band
    /// uncovered by the moving upper boundary (strict seam-contract violation).
    pub lower_misses: usize,
    /// Subset of `lower_misses` where no other layer covers the spot either (background exposed).
    pub holes: usize,
    pub seams: Vec<SeamCoverageSummary>,
    /// First `MAX_REPORTED_VIOLATIONS` violations (holes first, then other misses).
    pub violations: Vec<CoverageViolation>,
}

/// Outcome of walking one boundary point's exposed band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandWalk {
    /// Relative motion below `MIN_RELATIVE_MOTION_PX` (or a zero-length band): nothing exposed.
    Static,
    /// The band was walked; `supported` counts samples handed to the visitor.
    Walked { supported: usize },
}

/// Single definition of the band a moving upper boundary uncovers, shared by
/// `seam_requirements_with` and `check_seam_coverage_with` so the two cannot drift apart.
///
/// For boundary point `b` the upper boundary lands at `yu = deform(upper)(b)`. The lower layer's
/// rest point that lands there is `x_l = deform(lower)⁻¹(yu)` (fixed-point inverse, falling back
/// to the first-order `b + r` when it does not converge). Every rest-space sample on `b → x_l`
/// (1px steps from 0.5px, plus the end point) is exposed at that pose. Samples that are inside
/// the canvas, foreground, and owned at rest by `lower` or a layer above it are *supported*
/// (`lower` may legally fill them) and are passed to `visit(t, s, k)`; returning `false` stops.
#[allow(clippy::too_many_arguments)]
fn walk_exposed_band<FL, V>(
    b: DVec2,
    yu: DVec2,
    f_lower: &FL,
    fg_alpha: &[u8],
    owner: &[i32],
    lower: usize,
    w: usize,
    h: usize,
    mut visit: V,
) -> BandWalk
where
    FL: Fn(DVec2) -> DVec2,
    V: FnMut(f64, DVec2, usize) -> bool,
{
    let r = yu - f_lower(b);
    if r.length() < MIN_RELATIVE_MOTION_PX {
        return BandWalk::Static;
    }
    let (mut x_l, err) = invert(f_lower, yu, b + r);
    if err > 1.0 {
        x_l = b + r;
    }
    let span = (x_l - b).length();
    if span < 1e-9 {
        return BandWalk::Static;
    }
    let dir = (x_l - b) / span;
    let mut supported = 0usize;
    let mut t = 0.5f64.min(span);
    loop {
        let s = b + dir * t;
        if let Some(k) = pixel(s, w, h) {
            if fg_alpha[k] > 0 && owner[k] >= lower as i32 {
                supported += 1;
                if !visit(t, s, k) {
                    break;
                }
            }
        }
        if t >= span {
            break;
        }
        t = (t + 1.0).min(span);
    }
    BandWalk::Walked { supported }
}

/// Fixed-point inverse of a near-identity map: finds `x` with `f(x) ≈ target`.
fn invert<F: Fn(DVec2) -> DVec2>(f: &F, target: DVec2, guess: DVec2) -> (DVec2, f64) {
    let mut x = guess;
    let mut best = (x, f64::INFINITY);
    for _ in 0..10 {
        let res = target - f(x);
        let e = res.length();
        if e < best.1 {
            best = (x, e);
        }
        if e < 0.05 {
            break;
        }
        x += res;
    }
    best
}

/// Checks, for every seam boundary point × envelope pose, that the band uncovered by the moving
/// upper boundary is covered by the deformed lower effective mask (`mask ∪ fillMask`).
#[allow(clippy::too_many_arguments)]
pub fn check_seam_coverage(
    graph: &SeamGraph,
    layer_ids: &[String],
    visible_masks: &[Vec<u8>],
    fill_masks: &[Vec<u8>],
    fg_alpha: &[u8],
    lm: &Landmarks,
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
    stride: usize,
) -> SeamCoverageReport {
    check_seam_coverage_with_fold(
        graph,
        layer_ids,
        visible_masks,
        fill_masks,
        fg_alpha,
        w,
        h,
        poses,
        stride,
        |p, id, pose| deform(p, id, lm, pose),
        |p, id| crate::core::eye_interior_folded(p, id, lm),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn check_seam_coverage_with<D>(
    graph: &SeamGraph,
    layer_ids: &[String],
    visible_masks: &[Vec<u8>],
    fill_masks: &[Vec<u8>],
    fg_alpha: &[u8],
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
    stride: usize,
    deform_fn: D,
) -> SeamCoverageReport
where
    D: Fn(Point, &str, &Pose) -> Point,
{
    check_seam_coverage_with_fold(
        graph,
        layer_ids,
        visible_masks,
        fill_masks,
        fg_alpha,
        w,
        h,
        poses,
        stride,
        deform_fn,
        |_, _| false,
    )
}

/// `check_seam_coverage_with` plus a predicate `folded(rest_point, lower_id)` for rest samples the
/// lower layer's field folds away (T10 white/iris outside the aperture): those samples expose
/// nothing of the lower layer and are skipped, since the folded field has no inverse there.
#[allow(clippy::too_many_arguments)]
pub fn check_seam_coverage_with_fold<D, F>(
    graph: &SeamGraph,
    layer_ids: &[String],
    visible_masks: &[Vec<u8>],
    fill_masks: &[Vec<u8>],
    fg_alpha: &[u8],
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
    stride: usize,
    deform_fn: D,
    folded: F,
) -> SeamCoverageReport
where
    D: Fn(Point, &str, &Pose) -> Point,
    F: Fn(Point, &str) -> bool,
{
    let n = w * h;
    let owner = visible_owner(visible_masks, n);
    let eff: Vec<Vec<u8>> = visible_masks
        .iter()
        .enumerate()
        .map(|(p, vm)| effective_part_mask(vm, fill_masks.get(p).map(|f| f.as_slice())))
        .collect();
    let d = |p: DVec2, id: &str, pose: &Pose| dv(deform_fn(Point::from(p), id, pose));
    let covered_by_any = |posed: DVec2, pose: &Pose| -> bool {
        layer_ids.iter().enumerate().any(|(p, id)| {
            let f = |x: DVec2| d(x, id.as_str(), pose);
            let guess = posed - (f(posed) - posed);
            let (x, err) = invert(&f, posed, guess);
            err <= 1.0 && pixel(x, w, h).is_some_and(|k| eff[p][k] > 0)
        })
    };
    let mut report = SeamCoverageReport {
        poses: poses.len(),
        ..Default::default()
    };
    let mut holes_v: Vec<CoverageViolation> = Vec::new();
    let mut misses_v: Vec<CoverageViolation> = Vec::new();
    for seam in &graph.seams {
        let lower = seam.lower;
        let mut sum = SeamCoverageSummary {
            upper_id: seam.upper_id.clone(),
            lower_id: seam.lower_id.clone(),
            ..Default::default()
        };
        let pts: Vec<Point> = seam
            .boundary
            .iter()
            .flatten()
            .step_by(stride.max(1))
            .copied()
            .collect();
        for pe in poses {
            let f_lower = |x: DVec2| d(x, seam.lower_id.as_str(), &pe.pose);
            let holes_before = sum.holes;
            for &bp in &pts {
                sum.checks += 1;
                let b = dv(bp);
                let yu = d(b, seam.upper_id.as_str(), &pe.pose);
                let walk =
                    walk_exposed_band(b, yu, &f_lower, fg_alpha, &owner, lower, w, h, |t, s, k| {
                        if eff[lower][k] > 0 {
                            return true;
                        }
                        // T10: a white/iris rest sample outside the aperture folds onto the lid
                        // edge (zero posed area): nothing of the lower is exposed there and the
                        // folded field has no inverse. Background holes stay counted by the
                        // exposure raster.
                        if folded(Point::from(s), seam.lower_id.as_str()) {
                            return true;
                        }
                        let posed = f_lower(s);
                        let hole = !covered_by_any(posed, &pe.pose);
                        sum.lower_misses += 1;
                        sum.max_miss_depth_px = sum.max_miss_depth_px.max(t);
                        if hole {
                            sum.holes += 1;
                            sum.max_hole_depth_px = sum.max_hole_depth_px.max(t);
                            *sum.hole_pose_groups.entry(pe.group.clone()).or_default() += 1;
                        }
                        let v = CoverageViolation {
                            upper_id: seam.upper_id.clone(),
                            lower_id: seam.lower_id.clone(),
                            pose: pe.name.clone(),
                            boundary_point: bp,
                            rest_sample: Point::from(s),
                            posed_sample: Point::from(posed),
                            depth_px: t,
                            hole,
                        };
                        if hole {
                            if holes_v.len() < MAX_REPORTED_VIOLATIONS {
                                holes_v.push(v);
                            }
                        } else if misses_v.len() < MAX_REPORTED_VIOLATIONS {
                            misses_v.push(v);
                        }
                        false
                    });
                if walk != BandWalk::Static {
                    sum.moving_checks += 1;
                }
            }
            let pose_holes = sum.holes - holes_before;
            if pose_holes > sum.worst_hole_pose_holes {
                sum.worst_hole_pose_holes = pose_holes;
                sum.worst_hole_pose = Some(pe.name.clone());
            }
        }
        report.checks += sum.checks;
        report.lower_misses += sum.lower_misses;
        report.holes += sum.holes;
        report.seams.push(sum);
    }
    holes_v.extend(misses_v);
    holes_v.truncate(MAX_REPORTED_VIOLATIONS);
    report.violations = holes_v;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{deform_uncoupled, infer_part_role};

    fn roles(ids: &[String]) -> Vec<PartRole> {
        ids.iter().map(|id| infer_part_role(id, None)).collect()
    }

    fn lm_1024() -> Landmarks {
        Landmarks {
            eye_r: Point { x: 400.0, y: 360.0 },
            eye_l: Point { x: 624.0, y: 360.0 },
            mouth: Point { x: 512.0, y: 492.0 },
            chin: Point { x: 512.0, y: 560.0 },
            crown: Point { x: 512.0, y: 120.0 },
            eye_frames: None,
            torso_frame: None,
        }
    }

    #[test]
    fn embedded_pose_envelope_covers_full_param_ranges_and_combinations() {
        let env = pose_envelope();
        assert_eq!(env.version, 1);
        assert_eq!(
            env.poses.len(),
            145,
            "envelope size changed: update docs/tests"
        );
        assert_eq!(env.poses[0].name, "rest");
        let fields = pose_field_names();
        assert_eq!(env.ranges.len(), fields.len());
        // Every range extreme must be reached by an `axis` pose (never shrink PARAM_DEFS).
        for (key, &(lo, hi)) in &env.ranges {
            for target in [lo, hi] {
                let hit = env.poses.iter().any(|p| {
                    let v = serde_json::to_value(&p.pose).unwrap();
                    (v[key].as_f64().unwrap() - target).abs() < 1e-12
                });
                assert!(hit, "{key}={target} not in envelope");
            }
        }
        let groups: HashSet<&str> = env.poses.iter().map(|p| p.group.as_str()).collect();
        for g in [
            "axis",
            "headBodyOpposite",
            "headBodySame",
            "hairPhase",
            "headHairBody",
            "eyeSeal",
        ] {
            assert!(groups.contains(g), "missing group {g}");
        }
        // Head and body truly opposite on every axis in at least one pose.
        assert!(env
            .poses
            .iter()
            .any(|p| p.pose.angle_x * p.pose.body_angle_x < 0.0
                && p.pose.angle_y * p.pose.body_angle_y < 0.0
                && p.pose.angle_z * p.pose.body_angle_z < 0.0));
        // Hair phase difference: side and back swing against front.
        assert!(env
            .poses
            .iter()
            .any(|p| p.pose.hair * p.pose.hair_side < 0.0));
        assert!(env
            .poses
            .iter()
            .any(|p| p.pose.hair_side * p.pose.hair_back < 0.0));
        // Deterministic: a second parse yields the same table.
        let again = parse_pose_envelope(POSE_ENVELOPE_JSON).unwrap();
        let a: Vec<_> = env
            .poses
            .iter()
            .map(|p| serde_json::to_string(p).unwrap())
            .collect();
        let b: Vec<_> = again
            .poses
            .iter()
            .map(|p| serde_json::to_string(p).unwrap())
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn envelope_parser_rejects_unknown_keys_and_out_of_range_values() {
        let base: serde_json::Value = serde_json::from_str(POSE_ENVELOPE_JSON).unwrap();
        let mut typo = base.clone();
        typo["poses"][1]["params"] = serde_json::json!({ "angelX": 30 });
        assert!(parse_pose_envelope(&typo.to_string()).is_err());
        let mut over = base.clone();
        over["poses"][1]["params"] = serde_json::json!({ "angleX": 45 });
        assert!(parse_pose_envelope(&over.to_string()).is_err());
        let mut shrunk = base;
        shrunk["ranges"].as_object_mut().unwrap().remove("angleZ");
        assert!(parse_pose_envelope(&shrunk.to_string()).is_err());
    }

    /// Two-layer canvas: `lower` owns x < split, `upper` owns x >= split (both visible, fg everywhere
    /// in the band rows).
    fn split_canvas(w: usize, h: usize, split: usize) -> (Vec<String>, Vec<Vec<u8>>, Vec<u8>) {
        let n = w * h;
        let mut lower = vec![0u8; n];
        let mut upper = vec![0u8; n];
        let mut fg = vec![0u8; n];
        for y in 8..h - 8 {
            for x in 8..w - 8 {
                let i = y * w + x;
                fg[i] = 255;
                if x < split {
                    lower[i] = 255;
                } else {
                    upper[i] = 255;
                }
            }
        }
        (
            vec!["neck".into(), "clothes".into()],
            vec![lower, upper],
            fg,
        )
    }

    fn rest_and_moved() -> Vec<EnvelopePose> {
        vec![
            EnvelopePose {
                name: "rest".into(),
                group: "rest".into(),
                pose: Pose::default(),
            },
            EnvelopePose {
                name: "moved".into(),
                group: "test".into(),
                pose: Pose {
                    hair: 1.0,
                    ..Pose::default()
                },
            },
        ]
    }

    #[test]
    fn requirement_tracks_injected_deform_and_scales_with_its_coefficients() {
        let (w, h, split) = (96usize, 64usize, 40usize);
        let (ids, masks, fg) = split_canvas(w, h, split);
        let graph = extract_seam_graph(&ids, &roles(&ids), &masks, w, h);
        assert_eq!(graph.seams.len(), 1);
        let seam = &graph.seams[0];
        assert_eq!(
            (seam.upper_id.as_str(), seam.lower_id.as_str()),
            ("clothes", "neck")
        );
        assert_eq!(
            (seam.upper_role, seam.lower_role),
            (PartRole::Clothes, PartRole::Neck)
        );
        assert!(seam.coupling.is_none());
        let poses = rest_and_moved();
        // Upper layer slides right by `k · 10px` when `hair = 1`: exposure depth must be ≈ k·10.
        let run = |k: f64| {
            seam_requirements_with(&graph, &masks, &fg, w, h, &poses, 0, |p, id, pose| {
                if id == "clothes" {
                    Point {
                        x: p.x + k * 10.0 * pose.hair,
                        y: p.y,
                    }
                } else {
                    p
                }
            })
            .remove(0)
        };
        let r1 = run(1.0);
        let r2 = run(2.0);
        assert!((r1.max_relative_displacement_px - 10.0).abs() < 1e-9);
        assert!((r2.max_relative_displacement_px - 20.0).abs() < 1e-9);
        assert!((r1.max_exposure_depth_px - 10.0).abs() <= 1.0, "{r1:?}");
        assert!((r2.max_exposure_depth_px - 20.0).abs() <= 1.0, "{r2:?}");
        assert_eq!(
            r1.required_underlap_px,
            (r1.max_exposure_depth_px * SEAM_SAFETY_SCALE + SEAM_SAFETY_PAD_PX).ceil() as usize
        );
        assert!(r2.required_underlap_px > r1.required_underlap_px);
        assert_eq!(r1.worst_pose.as_deref(), Some("moved"));
        // Upper slides *into* the lower layer: nothing is exposed, only the pad remains.
        let into = seam_requirements_with(&graph, &masks, &fg, w, h, &poses, 0, |p, id, pose| {
            if id == "clothes" {
                Point {
                    x: p.x - 10.0 * pose.hair,
                    y: p.y,
                }
            } else {
                p
            }
        })
        .remove(0);
        assert_eq!(into.max_exposure_depth_px, 0.0);
        assert_eq!(
            into.required_underlap_px,
            SEAM_SAFETY_PAD_PX.ceil() as usize
        );
    }

    #[test]
    fn requirement_rescales_when_real_deform_coefficients_change() {
        // Scaling the real `core::deform` displacement field by k must scale the measured relative
        // displacement by exactly k: the margin is recomputed from `deform`, never hand-copied.
        let lm = lm_1024();
        let points: Vec<Point> = (0..9)
            .map(|i| Point {
                x: 470.0 + i as f64 * 10.0,
                y: 640.0,
            })
            .collect();
        let env = &pose_envelope().poses;
        let lmr = &lm;
        // Uncoupled field: inside the collar column the T13b coupled field makes clothes and neck
        // identical (relative displacement 0), which would make the scaling check vacuous.
        let base = |p: Point, id: &str, pose: &Pose| deform_uncoupled(p, id, lmr, pose);
        let (m1, pose1) = max_relative_displacement(&points, "clothes", "neck", env, base);
        let scaled = |k: f64| {
            move |p: Point, id: &str, pose: &Pose| {
                let q = deform_uncoupled(p, id, lmr, pose);
                Point {
                    x: p.x + k * (q.x - p.x),
                    y: p.y + k * (q.y - p.y),
                }
            }
        };
        let (m2, pose2) = max_relative_displacement(&points, "clothes", "neck", env, scaled(2.0));
        let (mh, _) = max_relative_displacement(&points, "clothes", "neck", env, scaled(0.5));
        assert!(m1 > 1.0);
        assert!((m2 - 2.0 * m1).abs() < 1e-6, "m1={m1} m2={m2}");
        assert!((mh - 0.5 * m1).abs() < 1e-6, "m1={m1} mh={mh}");
        assert_eq!(pose1, pose2);
    }

    #[test]
    fn thin_upper_strip_moving_past_its_width_still_requires_underlap_over_the_strip() {
        // lower `neck` x∈[8,40), thin upper `clothes` strip x∈[40,44), background beyond. The strip
        // slides right by 12px (3× its width): the end point `b + r` lands on background, but the
        // whole strip area it vacates must be supplied by `neck`.
        let (w, h, split, strip) = (96usize, 32usize, 40usize, 4usize);
        let n = w * h;
        let mut lower = vec![0u8; n];
        let mut upper = vec![0u8; n];
        let mut fg = vec![0u8; n];
        for y in 4..h - 4 {
            for x in 8..split + strip {
                let i = y * w + x;
                fg[i] = 255;
                if x < split {
                    lower[i] = 255;
                } else {
                    upper[i] = 255;
                }
            }
        }
        let ids: Vec<String> = vec!["neck".into(), "clothes".into()];
        let masks = vec![lower, upper];
        let graph = extract_seam_graph(&ids, &roles(&ids), &masks, w, h);
        let poses = rest_and_moved();
        let shift = |p: Point, id: &str, pose: &Pose| {
            if id == "clothes" {
                Point {
                    x: p.x + 12.0 * pose.hair,
                    y: p.y,
                }
            } else {
                p
            }
        };
        let req = seam_requirements_with(&graph, &masks, &fg, w, h, &poses, 0, shift).remove(0);
        assert!(
            req.max_exposure_depth_px >= strip as f64 - 0.5,
            "strip area skipped: {req:?}"
        );
        assert!(req.required_underlap_px >= strip, "{req:?}");
        assert_eq!(
            req.unsupported_samples, 0,
            "band had supported samples: {req:?}"
        );
        // The coverage check walks the same band: without fill it reports misses, with the
        // required underlap it reports none.
        let empty = vec![vec![0u8; n]; 2];
        let bad =
            check_seam_coverage_with(&graph, &ids, &masks, &empty, &fg, w, h, &poses, 1, shift);
        assert!(bad.lower_misses > 0);
        let mut fill = vec![vec![0u8; n]; 2];
        for y in 0..h {
            for x in split..(split + req.required_underlap_px).min(w) {
                let i = y * w + x;
                if fg[i] > 0 {
                    fill[0][i] = 255;
                }
            }
        }
        let ok = check_seam_coverage_with(&graph, &ids, &masks, &fill, &fg, w, h, &poses, 1, shift);
        assert_eq!(ok.lower_misses, 0, "{:?}", ok.violations.first());

        // A band that is entirely background (upper moves away over empty canvas only) is the
        // only case counted as unsupported.
        let mut fg_none = fg.clone();
        for y in 0..h {
            for x in split..w {
                fg_none[y * w + x] = 0;
            }
        }
        let none =
            seam_requirements_with(&graph, &masks, &fg_none, w, h, &poses, 0, shift).remove(0);
        assert!(none.unsupported_samples > 0);
        assert_eq!(none.max_exposure_depth_px, 0.0);
    }

    #[test]
    fn coverage_check_passes_with_required_fill_and_reports_locations_without_it() {
        let (w, h, split) = (96usize, 64usize, 40usize);
        let (ids, masks, fg) = split_canvas(w, h, split);
        let graph = extract_seam_graph(&ids, &roles(&ids), &masks, w, h);
        let poses = rest_and_moved();
        let shift = |p: Point, id: &str, pose: &Pose| {
            if id == "clothes" {
                Point {
                    x: p.x + 10.0 * pose.hair,
                    y: p.y,
                }
            } else {
                p
            }
        };
        let req = seam_requirements_with(&graph, &masks, &fg, w, h, &poses, 0, shift).remove(0);
        // Fill the lower layer under the upper one by the required underlap.
        let mut fill_ok = vec![vec![0u8; w * h]; 2];
        for y in 0..h {
            for x in split..(split + req.required_underlap_px).min(w) {
                let i = y * w + x;
                if fg[i] > 0 {
                    fill_ok[0][i] = 255;
                }
            }
        }
        let ok =
            check_seam_coverage_with(&graph, &ids, &masks, &fill_ok, &fg, w, h, &poses, 1, shift);
        assert!(ok.checks > 0);
        assert_eq!(ok.lower_misses, 0, "{:?}", ok.violations.first());
        assert_eq!(ok.holes, 0);

        let empty = vec![vec![0u8; w * h]; 2];
        let bad =
            check_seam_coverage_with(&graph, &ids, &masks, &empty, &fg, w, h, &poses, 1, shift);
        assert!(bad.lower_misses > 0);
        assert_eq!(
            bad.holes, bad.lower_misses,
            "nothing else is underneath: every miss is a hole"
        );
        let v = &bad.violations[0];
        assert_eq!(
            (v.upper_id.as_str(), v.lower_id.as_str(), v.pose.as_str()),
            ("clothes", "neck", "moved")
        );
        assert!(v.hole && v.rest_sample.x >= split as f64 - 0.5);
        assert_eq!(bad.seams[0].lower_misses, bad.lower_misses);

        // A third layer below the upper that covers the exposed band turns holes into misses only.
        let mut ids3 = ids.clone();
        ids3.insert(0, "hair_back_3".into());
        let mut masks3 = vec![vec![0u8; w * h]];
        masks3.extend(masks.iter().cloned());
        let mut fill3 = vec![fg.clone()];
        fill3.extend(empty.iter().cloned());
        let graph3 = extract_seam_graph(&ids3, &roles(&ids3), &masks3, w, h);
        let shift3 = |p: Point, id: &str, pose: &Pose| shift(p, id, pose);
        let r3 = check_seam_coverage_with(
            &graph3, &ids3, &masks3, &fill3, &fg, w, h, &poses, 1, shift3,
        );
        assert!(r3.lower_misses > 0);
        assert_eq!(r3.holes, 0);
    }

    #[test]
    fn seam_graph_uses_roles_for_variable_hair_and_accessories_with_continuous_polylines() {
        let (w, h) = (64usize, 64usize);
        let n = w * h;
        let ids: Vec<String> = [
            "hair_back_3",
            "face",
            "hair_side_L_2",
            "hair_front_2",
            "acc_ribbon",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut masks = vec![vec![0u8; n]; ids.len()];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let (xf, yf) = (x as f64, y as f64);
                let layer = if (xf - 32.0).hypot(yf - 10.0) < 6.0 {
                    4
                } else if yf < 20.0 && (8.0..56.0).contains(&xf) {
                    3
                } else if (44.0..56.0).contains(&xf) && (20.0..60.0).contains(&yf) {
                    2
                } else if (xf - 32.0).hypot(yf - 34.0) < 16.0 {
                    1
                } else if (4.0..60.0).contains(&xf) && (4.0..62.0).contains(&yf) {
                    0
                } else {
                    continue;
                };
                masks[layer][i] = 255;
            }
        }
        let graph = extract_seam_graph(&ids, &roles(&ids), &masks, w, h);
        let pairs: Vec<(PartRole, PartRole)> = graph
            .seams
            .iter()
            .map(|s| (s.upper_role, s.lower_role))
            .collect();
        for want in [
            (PartRole::Face, PartRole::HairBack),
            (PartRole::HairSideL, PartRole::Face),
            (PartRole::HairFront, PartRole::Face),
            (PartRole::Accessory, PartRole::HairFront),
        ] {
            assert!(pairs.contains(&want), "missing seam {want:?} in {pairs:?}");
        }
        let side = graph.find("hair_side_L_2", "face").unwrap();
        assert!(side.boundary_len() > 10);
        for line in &side.boundary {
            for pair in line.windows(2) {
                let d = (pair[0].x - pair[1].x).hypot(pair[0].y - pair[1].y);
                assert!(d <= 1.0 + 1e-9, "polyline jump {d} in {line:?}");
            }
        }
        // Every pair of layers sharing a visible edge yields exactly one seam with upper above lower.
        for s in &graph.seams {
            assert!(s.upper > s.lower);
        }
    }

    #[test]
    fn hair_sway_probe_reads_deform_behaviour() {
        let lm = lm_1024();
        assert!(follows_hair_sway("hat", &lm));
        assert!(follows_hair_sway("acc_ribbon_2", &lm));
        assert!(follows_hair_sway("hair_front_2", &lm));
        assert!(!follows_hair_sway("acc_glasses", &lm));
        assert!(!follows_hair_sway("face", &lm));
    }

    type Case = (&'static str, &'static str, Vec<(f64, f64)>);

    /// Seam point sets shared by the parity fixture and the T13b coupling test.
    fn parity_cases() -> Vec<Case> {
        vec![
            (
                "face",
                "neck",
                vec![(470.0, 560.5), (512.0, 562.5), (556.0, 548.5)],
            ),
            (
                "clothes",
                "neck",
                vec![(470.0, 639.5), (512.0, 639.5), (558.5, 650.0)],
            ),
            (
                "hair_front",
                "face",
                vec![(420.0, 280.5), (512.0, 325.5), (600.0, 290.5)],
            ),
            ("hair_side_R", "face", vec![(380.5, 360.0), (380.5, 520.0)]),
            (
                "hair_side_L_2",
                "clothes",
                vec![(650.5, 700.0), (700.0, 760.5)],
            ),
            ("body", "hair_back", vec![(300.5, 700.0), (724.5, 720.0)]),
            (
                "clothes",
                "hair_back_3",
                vec![(301.5, 760.0), (723.5, 780.0)],
            ),
            (
                "eyeL_upper",
                "eyeL_white",
                vec![(610.0, 344.5), (640.0, 348.5)],
            ),
            ("eyeR_lower", "eyeR_iris", vec![(400.0, 378.5)]),
            ("browL", "face", vec![(624.0, 300.5)]),
            (
                "mouth_upper",
                "mouth_inner",
                vec![(512.0, 489.5), (526.0, 491.5)],
            ),
            ("hat", "hair_front", vec![(512.0, 150.5)]),
            ("acc_ribbon", "hair_back", vec![(700.0, 300.5)]),
            ("clothes_hands", "clothes", vec![(560.0, 860.5)]),
            // T13b: hair root band (strand edges near the scalp), collar corner, body / neck.
            (
                "hair_front",
                "face",
                vec![(512.0, 120.5), (452.0, 168.5), (512.0, 182.5), (574.0, 200.5)],
            ),
            (
                "hair_side_L_2",
                "hair_back",
                vec![(650.5, 240.0), (690.5, 280.0)],
            ),
            ("body", "neck", vec![(486.0, 660.5), (538.0, 660.5)]),
            ("clothes", "neck", vec![(560.5, 645.0), (466.0, 639.5)]),
        ]
    }

    /// Points far from every coupled seam (name, layer, point). T13b must leave their motion as is.
    /// The hair layer's own crown is its root seam with the scalp (coupled), so the crown is checked
    /// on the head (`face`) layer.
    fn far_points() -> Vec<(&'static str, &'static str, (f64, f64))> {
        vec![
            ("crownFace", "face", (512.0, 120.0)),
            ("faceCentre", "face", (512.0, 400.0)),
            ("noseTip", "face", (512.0, 440.0)),
            ("bangTip", "hair_front", (510.0, 324.0)),
            ("bangTip2", "hair_front_2", (470.0, 368.0)),
            ("sideLockTipR", "hair_side_R", (330.0, 715.0)),
            ("sideLockTipL", "hair_side_L_2", (694.0, 755.0)),
            ("backHairTip", "hair_back", (512.0, 855.0)),
            ("torsoCentre", "clothes", (512.0, 800.0)),
            ("torsoCentreBody", "body", (512.0, 820.0)),
            ("shoulderR", "clothes", (330.0, 700.0)),
            ("shoulderL", "clothes", (694.0, 700.0)),
            ("irisR", "eyeR_iris", (400.0, 360.0)),
            ("irisL", "eyeL_iris", (624.0, 360.0)),
            ("browR", "browR", (400.0, 300.0)),
            ("mouth", "mouth_upper", (512.0, 488.0)),
        ]
    }

    fn max_displacement<D: Fn(Point, &str, &Pose) -> Point>(
        p: Point,
        id: &str,
        env: &[EnvelopePose],
        f: D,
    ) -> f64 {
        env.iter()
            .map(|pe| (dv(f(p, id, &pe.pose)) - dv(p)).length())
            .fold(0.0, f64::max)
    }

    /// T13b: coupling lowers the envelope max relative displacement of the coupled seams and leaves
    /// every point far from them exactly where the uncoupled field puts it, in every pose.
    #[test]
    fn t13b_coupling_reduces_seam_relative_displacement_and_keeps_far_points_unchanged() {
        let lm = lm_1024();
        let env = &pose_envelope().poses;
        let coupled = |p: Point, id: &str, pose: &Pose| deform(p, id, &lm, pose);
        let uncoupled = |p: Point, id: &str, pose: &Pose| deform_uncoupled(p, id, &lm, pose);
        println!("T13B_REL | upper | lower | points | uncoupled max rel px | coupled max rel px |");
        for (upper, lower, pts) in parity_cases() {
            let points: Vec<Point> = pts.iter().map(|&(x, y)| Point { x, y }).collect();
            let (mu, _) = max_relative_displacement(&points, upper, lower, env, uncoupled);
            let (mc, _) = max_relative_displacement(&points, upper, lower, env, coupled);
            println!("T13B_REL | {upper} | {lower} | {pts:?} | {mu:.2} | {mc:.2} |");
            let coupled_pair = matches!(
                (upper, lower),
                ("face", "neck") | ("clothes", "neck") | ("body", "neck")
            ) || (upper.starts_with("hair") && pts.iter().all(|p| p.1 < 300.0));
            if coupled_pair {
                assert!(
                    mc < mu - 1e-3,
                    "{upper}/{lower}: coupling must reduce relative displacement ({mu} -> {mc})"
                );
            }
        }
        // Far points: same displacement in every pose (not only the max) - no motion reduction.
        println!("T13B_FAR | point | layer | uncoupled max disp px | coupled max disp px |");
        for (name, id, (x, y)) in far_points() {
            let p = Point { x, y };
            for pe in env {
                let a = coupled(p, id, &pe.pose);
                let b = uncoupled(p, id, &pe.pose);
                assert!(
                    (a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9,
                    "{name} ({id}) moved differently in {}: {a:?} vs {b:?}",
                    pe.name
                );
            }
            let (du, dc) = (
                max_displacement(p, id, env, uncoupled),
                max_displacement(p, id, env, coupled),
            );
            println!("T13B_FAR | {name} | {id} | {du:.3} | {dc:.3} |");
            assert!(du > 0.0, "{name} must move in the envelope");
        }
        // The thread-local switch reproduces the uncoupled field through the public `deform`.
        let p = Point { x: 512.0, y: 562.5 };
        let pose = &env.iter().find(|e| e.pose.angle_z != 0.0).unwrap().pose;
        let off = crate::core::with_seam_coupling(false, || deform(p, "neck", &lm, pose));
        let raw = deform_uncoupled(p, "neck", &lm, pose);
        assert_eq!((off.x, off.y), (raw.x, raw.y));
        assert!(crate::core::seam_coupling_enabled());
    }

    #[test]
    fn coupling_annotation_reads_deform_weights_for_coupled_role_pairs_only() {
        let (w, h) = (1024usize, 1024usize);
        let n = w * h;
        let ids: Vec<String> = ["hair_back", "neck", "clothes", "face", "hair_front"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut masks = vec![vec![0u8; n]; ids.len()];
        for y in 0..h {
            for x in 0..w {
                let (xf, yf) = (x as f64, y as f64);
                let layer = if ((xf - 512.0) / 200.0).hypot((yf - 210.0) / 115.0) <= 1.0 {
                    4
                } else if ((xf - 512.0) / 180.0).hypot((yf - 340.0) / 220.0) <= 1.0 {
                    3
                } else if (xf - 512.0).abs() <= 210.0 && (640.0..=940.0).contains(&yf) {
                    2
                } else if (xf - 512.0).abs() <= 46.0 && (540.0..=660.0).contains(&yf) {
                    1
                } else if ((xf - 512.0) / 260.0).hypot((yf - 520.0) / 340.0) <= 1.0 {
                    0
                } else {
                    continue;
                };
                masks[layer][y * w + x] = 255;
            }
        }
        let lm = lm_1024();
        let mut graph = extract_seam_graph(&ids, &roles(&ids), &masks, w, h);
        assert!(graph.seams.iter().all(|s| s.coupling.is_none()));
        annotate_seam_couplings(&mut graph, &lm);
        let fn_ = graph.find("face", "neck").unwrap().coupling.unwrap();
        assert_eq!(fn_.kind, SeamCouplingKind::NeckHead);
        assert!(fn_.lower_weight > 0.95, "{fn_:?}");
        assert_eq!(fn_.upper_weight, 0.0);
        let cn = graph.find("clothes", "neck").unwrap().coupling.unwrap();
        assert_eq!(cn.kind, SeamCouplingKind::Collar);
        // Clothes blend toward the neck-base field only by the neck-base weight (≈0.4 at y=640).
        assert!(cn.upper_weight > 0.2 && cn.upper_weight < 0.9, "{cn:?}");
        assert!(cn.lower_weight > 0.2, "{cn:?}");
        let hf = graph.find("hair_front", "face").unwrap().coupling.unwrap();
        assert_eq!(hf.kind, SeamCouplingKind::HairRoot);
        let ch_no_frame = graph.find("clothes", "hair_back").unwrap().coupling.unwrap();
        assert_eq!(ch_no_frame.kind, SeamCouplingKind::HairBody);
        assert_eq!(ch_no_frame.upper_weight, 0.0);
        assert_eq!(ch_no_frame.lower_weight, 0.0);
        assert_eq!(ch_no_frame.falloff_px, 0.0);
        assert!(graph.find("face", "hair_back").unwrap().coupling.is_some());

        let lm_with_torso = lm.with_torso_frame(&ids, &masks, w, h);
        annotate_seam_couplings(&mut graph, &lm_with_torso);
        let ch = graph.find("clothes", "hair_back").unwrap().coupling.unwrap();
        assert_eq!(ch.kind, SeamCouplingKind::HairBody);
        assert_eq!(ch.upper_weight, 0.0);
        assert!(ch.lower_weight > 0.5, "{ch:?}");
        assert!(ch.falloff_px > 0.0, "{ch:?}");
    }

    /// Shared Rust/TS fixture: the envelope as each side interprets it, the max relative
    /// displacement of fixed seam points with the coupled (`deform`) and uncoupled
    /// (`deform_uncoupled`) fields, far reference point displacements, and coupling weights.
    /// Regenerate with `FLAT2RIG_BLESS=1 cargo test seam_envelope_parity`.
    #[test]
    fn seam_envelope_parity_fixture_matches_rust_deform() {
        let lm = lm_1024();
        let env = &pose_envelope().poses;
        let coupled = |p: Point, id: &str, pose: &Pose| deform(p, id, &lm, pose);
        let uncoupled = |p: Point, id: &str, pose: &Pose| deform_uncoupled(p, id, &lm, pose);
        let mut out_cases = Vec::new();
        for (upper, lower, pts) in &parity_cases() {
            let points: Vec<Point> = pts.iter().map(|&(x, y)| Point { x, y }).collect();
            let (m, pose) = max_relative_displacement(&points, upper, lower, env, coupled);
            let (mu, pose_u) = max_relative_displacement(&points, upper, lower, env, uncoupled);
            out_cases.push(serde_json::json!({
                "upper": upper,
                "lower": lower,
                "points": pts.iter().map(|&(x, y)| serde_json::json!([x, y])).collect::<Vec<_>>(),
                "maxRelativeDisplacementPx": m,
                "worstPose": pose,
                "uncoupledMaxRelativeDisplacementPx": mu,
                "uncoupledWorstPose": pose_u,
            }));
        }
        let far: Vec<serde_json::Value> = far_points()
            .iter()
            .map(|&(name, id, (x, y))| {
                let p = Point { x, y };
                serde_json::json!({
                    "name": name,
                    "id": id,
                    "point": [x, y],
                    "maxDisplacementPx": max_displacement(p, id, env, coupled),
                })
            })
            .collect();
        let weight_samples: Vec<(&str, (f64, f64))> = vec![
            ("neck", (512.0, 560.0)),
            ("neck", (512.0, 590.0)),
            ("neck", (512.0, 650.0)),
            ("clothes", (512.0, 650.0)),
            ("clothes", (590.0, 660.0)),
            ("body", (512.0, 760.0)),
            ("hair_front", (512.0, 170.0)),
            ("hair_side_L_2", (690.0, 260.0)),
            ("hair_back", (512.0, 200.0)),
            ("face", (512.0, 400.0)),
        ];
        let weights: Vec<serde_json::Value> = weight_samples
            .iter()
            .map(|&(id, (x, y))| {
                serde_json::json!({
                    "id": id,
                    "point": [x, y],
                    "weights": crate::core::seam_coupling_weights(Point { x, y }, id, &lm),
                })
            })
            .collect();
        let doc = serde_json::json!({
            "description": "T13a/T13b Rust/TS parity: envelope poses as parsed by Rust (Pose::default() merged); max relative seam displacement over the envelope with core::deform (coupled, T13b) and core::deform_uncoupled; far reference point max displacement (coupled); seam coupling weights. TS asserts the same values with engine.ts deform / deformUncoupled / seamCouplingWeights.",
            "landmarks": lm,
            "envelope": env.iter().map(|p| serde_json::json!({ "name": p.name, "pose": p.pose, "shape": p.pose.shape })).collect::<Vec<_>>(),
            "cases": out_cases,
            "farPoints": far,
            "couplingWeights": weights,
        });
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/seam-envelope-parity.json");
        if std::env::var("FLAT2RIG_BLESS").is_ok() {
            std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
        }
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("parity fixture present"))
                .unwrap();
        assert_eq!(
            fixture["envelope"], doc["envelope"],
            "envelope interpretation drifted"
        );
        let close = |a: &serde_json::Value, b: &serde_json::Value, what: &str| {
            let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!((x - y).abs() < 1e-9, "{what}: fixture {x} vs rust {y}");
        };
        let fc = fixture["cases"].as_array().unwrap();
        assert_eq!(fc.len(), out_cases.len());
        for (a, b) in fc.iter().zip(&out_cases) {
            assert_eq!(a["upper"], b["upper"]);
            assert_eq!(a["lower"], b["lower"]);
            let what = format!("{} / {}", a["upper"], a["lower"]);
            close(&a["maxRelativeDisplacementPx"], &b["maxRelativeDisplacementPx"], &what);
            close(
                &a["uncoupledMaxRelativeDisplacementPx"],
                &b["uncoupledMaxRelativeDisplacementPx"],
                &what,
            );
        }
        let ff = fixture["farPoints"].as_array().unwrap();
        assert_eq!(ff.len(), far.len());
        for (a, b) in ff.iter().zip(&far) {
            close(&a["maxDisplacementPx"], &b["maxDisplacementPx"], a["name"].as_str().unwrap());
        }
        let fw = fixture["couplingWeights"].as_array().unwrap();
        assert_eq!(fw.len(), weights.len());
        for (a, b) in fw.iter().zip(&weights) {
            for k in ["neckHead", "neckBody", "collar", "hairRoot"] {
                close(&a["weights"][k], &b["weights"][k], k);
            }
        }
    }

    /// B2 shared Rust/TS parity fixture: `derive_torso_frame` (including `clothes` + `hands`,
    /// tilted head, and scattered low-alpha noise pixels), `seam_coupling_weights` (`hair_body`),
    /// and coupled `deform` across extreme poses. Also checks bangs preservation and full sway
    /// amplitude preservation. Regenerate with `FLAT2RIG_BLESS=1 cargo test b2_hair_body_parity`.
    #[test]
    fn b2_hair_body_parity_fixture_matches_rust_deform() {
        let (w, h) = (256usize, 256usize);
        let n = w * h;
        // Tilted head (~-15°), matching Sutteang-style tilted geometry in miniature.
        let base_lm = Landmarks {
            eye_r: Point { x: 110.0, y: 98.0 },
            eye_l: Point { x: 162.0, y: 84.0 },
            mouth: Point { x: 144.0, y: 120.0 },
            chin: Point { x: 150.0, y: 142.0 },
            crown: Point { x: 120.0, y: 34.0 },
            eye_frames: None,
            torso_frame: None,
        };
        let ids: Vec<String> = ["clothes", "hands"].iter().map(|s| s.to_string()).collect();
        let mut clothes = vec![0u8; n];
        let mut hands = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                // Torso/sweater with shoulder slope + neck cutout.
                if (y >= 152 && (56..=208).contains(&x))
                    || (y >= 146 && (122..=156).contains(&x))
                {
                    clothes[y * w + x] = 255;
                }
                // Bare right shoulder labeled `hands` (like See-through Sutteang).
                if (145..=192).contains(&y) && (46..=92).contains(&x) {
                    hands[y * w + x] = 255;
                }
            }
        }
        // Scattered low-alpha noise (<128) above mouth must be ignored.
        clothes[40 * w + 100] = 64;
        hands[60 * w + 70] = 120;
        let effs = vec![clothes, hands];
        let lm = base_lm.with_torso_frame(&ids, &effs, w, h);
        let frame = lm.torso_frame.as_ref().expect("torso frame derived");
        assert!(frame.len() >= 10, "frame len {}", frame.len());

        let points: Vec<(&str, (f64, f64))> = vec![
            ("hair_front", (130.0, 68.0)),      // forehead bangs (above mouth -> hair_body == 0)
            ("hair_front", (136.0, 108.0)),     // bangs near nose (above mouth -> hair_body == 0)
            ("hair_front", (68.0, 162.0)),      // long lock over `hands` bare shoulder
            ("hair_front", (184.0, 185.0)),     // long lock over sweater
            ("hair_side_L", (174.0, 134.0)),    // cheek-side transition zone
            ("hair_side_L_2", (180.0, 175.0)),  // dynamic extra side hair on torso
            ("hair_side_R", (96.0, 165.0)),     // right side lock on torso
            ("hair_back", (138.0, 152.0)),      // behind neck column (neck_share > 0)
            ("hair_back", (195.0, 180.0)),      // outer back hair (neck_share == 0)
            ("hair_back_2", (72.0, 190.0)),     // dynamic extra back hair
        ];

        // Invariant 1: bangs above mouth have hair_body == 0 and identical position with/without frame.
        let env = &pose_envelope().poses;
        let pose_names = [
            "angleZ=30",
            "angleZ=-30",
            "opp3:head+-+/breath1",
            "same3:head--+/breath1",
            "hair=1",
            "hair=-1",
            "hairSide=1",
            "hairBack=-1",
        ];
        let selected: Vec<&EnvelopePose> = pose_names
            .iter()
            .map(|&name| env.iter().find(|p| p.name == name).expect(name))
            .collect();

        for &(id, (x, y)) in &points[..2] {
            let p = Point { x, y };
            let wts = crate::core::seam_coupling_weights(p, id, &lm);
            assert_eq!(wts.hair_body, 0.0, "bangs {id} ({x},{y}) must have hair_body == 0");
            for ep in &selected {
                let a = deform(p, id, &base_lm, &ep.pose);
                let b = deform(p, id, &lm, &ep.pose);
                assert!((a.x - b.x).abs() + (a.y - b.y).abs() < 1e-12);
            }
        }

        // Invariant 2: hair sway amplitude `deform(hair=1) - deform(hair=-1)` is preserved.
        let p_h_pos = &env.iter().find(|p| p.name == "hair=1").unwrap().pose;
        let p_h_neg = &env.iter().find(|p| p.name == "hair=-1").unwrap().pose;
        for &(id, (x, y)) in &points {
            let p = Point { x, y };
            let a1 = deform(p, id, &base_lm, p_h_pos);
            let a2 = deform(p, id, &base_lm, p_h_neg);
            let b1 = deform(p, id, &lm, p_h_pos);
            let b2 = deform(p, id, &lm, p_h_neg);
            let err = ((a1.x - a2.x) - (b1.x - b2.x)).hypot((a1.y - a2.y) - (b1.y - b2.y));
            assert!(err < 1e-9, "sway amplitude changed for {id} ({x},{y}): {err}");
        }

        let weights_json: Vec<serde_json::Value> = points
            .iter()
            .map(|&(id, (x, y))| {
                serde_json::json!({
                    "id": id,
                    "point": [x, y],
                    "weights": crate::core::seam_coupling_weights(Point { x, y }, id, &lm),
                })
            })
            .collect();
        let cases_json: Vec<serde_json::Value> = selected
            .iter()
            .map(|ep| {
                let out: Vec<[f64; 2]> = points
                    .iter()
                    .map(|&(id, (x, y))| {
                        let q = deform(Point { x, y }, id, &lm, &ep.pose);
                        [q.x, q.y]
                    })
                    .collect();
                serde_json::json!({ "pose": ep.name, "out": out })
            })
            .collect();
        let doc = serde_json::json!({
            "description": "B2 hair_body Rust/TS parity: tilted-head derive_torso_frame (clothes+hands+noise), seam_coupling_weights, and coupled deform.",
            "width": w,
            "height": h,
            "landmarks": base_lm,
            "torsoFrame": frame,
            "points": points.iter().map(|&(id, (x, y))| serde_json::json!({ "id": id, "point": [x, y] })).collect::<Vec<_>>(),
            "weights": weights_json,
            "cases": cases_json,
        });
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/hair-body-parity.json");
        if std::env::var("FLAT2RIG_BLESS").is_ok() {
            std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
        }
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("hair-body parity fixture present"))
                .unwrap();
        let close = |a: &serde_json::Value, b: &serde_json::Value, what: &str| {
            let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!((x - y).abs() < 1e-9, "{what}: fixture {x} vs rust {y}");
        };
        let ff = fixture["torsoFrame"].as_array().unwrap();
        assert_eq!(ff.len(), frame.len());
        for (i, (a, b)) in ff.iter().zip(frame).enumerate() {
            close(&a["u"], &serde_json::json!(b.u), &format!("torsoFrame[{i}].u"));
            close(&a["top"], &serde_json::json!(b.top), &format!("torsoFrame[{i}].top"));
        }
        let fw = fixture["weights"].as_array().unwrap();
        assert_eq!(fw.len(), weights_json.len());
        for (a, b) in fw.iter().zip(&weights_json) {
            for k in ["neckHead", "neckBody", "collar", "hairRoot", "hairBody"] {
                close(&a["weights"][k], &b["weights"][k], k);
            }
        }
        let fc = fixture["cases"].as_array().unwrap();
        assert_eq!(fc.len(), cases_json.len());
        for (a, b) in fc.iter().zip(&cases_json) {
            assert_eq!(a["pose"], b["pose"]);
            let ao = a["out"].as_array().unwrap();
            let bo = b["out"].as_array().unwrap();
            for (i, (pa, pb)) in ao.iter().zip(bo).enumerate() {
                close(&pa[0], &pb[0], &format!("{}[{i}].x", a["pose"]));
                close(&pa[1], &pb[1], &format!("{}[{i}].y", a["pose"]));
            }
        }
    }
}
