//! T13b — exposure map (`ExposureMap`).
//!
//! For every pose of the shared envelope (`seams::pose_envelope`), every layer's ArtMesh
//! (`core::art_mesh_for_part` over `mask ∪ fillMask`, the same mesh `build_rig` and the CLI use) is
//! posed with the real `core::deform` and rasterised bottom-to-top with barycentric interpolation of
//! the rest-space source point (the CLI/app triangle path: pixel centres at `+0.5`, source texel
//! `floor`). Render order and the app's clipping targets (`render.ts`: iris → white,
//! `face_detail`/mouth → `face` when `face` has a hidden area) are applied. The topmost opaque
//! (`alpha > 127`) layer of each output pixel is recorded together with its source texel.
//!
//! A *hidden* texel of a layer is `fillMask > 0` outside its visible `mask`. When it is the topmost
//! texel of some output pixel in a pose it is *exposed* in that pose. Per layer and texel we count
//! exposed poses (frequency = count / poses) and the exposure depth (chamfer distance from the
//! layer's visible mask). Unexposed hidden texels are reported but never removed: they stay for PSD
//! hand-off continuity; the map only says where hidden-area quality matters.
//!
//! Holes: an output pixel no layer covers is *enclosed* when its 4-connected uncovered component
//! does not reach the canvas border (background trapped inside the posed silhouette). Pixels already
//! enclosed at rest (natural loops, mesh coverage gaps) are excluded per pixel; the rest are holes.
//! Background that opens to the outside (a seam notch at the silhouette edge, or a hair lock swinging
//! clear of the torso) is not counted here: the seam coverage check (`seams::check_seam_coverage`,
//! `holes`) is the authoritative seam-opening test and is reported next to this raster count.

use crate::core::{
    art_mesh_for_part, deform, effective_part_mask, infer_part_role, Landmarks, Mesh, PartRole,
    Point, Pose,
};
use crate::rig::seams::{outside_distance, EnvelopePose};
use serde::Serialize;

/// Opacity threshold for a texel to occlude what is below it.
const OPAQUE: u8 = 127;
/// Mesh density used by `build_rig` / the CLI.
const MESH_DENSITY: usize = 14;

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerExposure {
    pub id: String,
    pub role: Option<PartRole>,
    /// Hidden texels (`fillMask > 0`, visible `mask == 0`).
    pub hidden_px: usize,
    /// Hidden texels exposed in at least one pose.
    pub exposed_px: usize,
    /// Hidden texels never exposed (kept, excluded from quality targets).
    pub unexposed_hidden_px: usize,
    /// Mean exposure frequency over exposed texels (0..1).
    pub mean_frequency: f64,
    pub max_frequency: f64,
    /// Max distance (px) of an exposed hidden texel from this layer's visible mask (`None` when
    /// the layer has no visible pixels, e.g. a fully synthesized `body`).
    pub max_depth_px: Option<f64>,
    /// 95th percentile of the exposure depth over exposed texels.
    pub p95_depth_px: Option<f64>,
    /// Hidden texels already exposed at rest (mesh coverage gaps of upper layers).
    pub rest_exposed_px: usize,
    /// T9 overlay texels: `fillMask` texels lying over a *lower* layer's visible pixel (seam
    /// outline / anti-alias pixels handed up from the lower rim). They are drawn at rest by design,
    /// so they are not hidden texels and are excluded from every hidden / exposure count.
    pub overlay_px: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HoleSummary {
    /// Enclosed background pixels at rest (excluded from holes).
    pub rest_enclosed_px: usize,
    /// Foreground pixels no layer covers at rest (mesh coverage gaps).
    pub rest_uncovered_fg_px: usize,
    /// Hole pixels summed over poses.
    pub hole_px_total: usize,
    /// Max hole pixels in a single pose.
    pub hole_px_max: usize,
    /// Poses with at least one hole pixel.
    pub hole_poses: usize,
    /// 8-connected hole components summed over poses.
    pub hole_components_total: usize,
    /// Max hole components in a single pose.
    pub hole_components_max: usize,
    pub worst_pose: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExposureMap {
    pub width: usize,
    pub height: usize,
    pub poses: usize,
    /// How layers were rendered.
    pub render: String,
    pub layers: Vec<LayerExposure>,
    pub hidden_px_total: usize,
    pub exposed_px_total: usize,
    pub max_depth_px: f64,
    pub holes: HoleSummary,
    /// Per layer, per rest texel: number of poses in which the hidden texel was exposed (empty when
    /// the layer has no hidden texels).
    #[serde(skip)]
    pub counts: Vec<Vec<u16>>,
    /// Per output pixel: number of poses in which it was a hole.
    #[serde(skip)]
    pub hole_counts: Vec<u16>,
    /// Per layer, per rest texel: hidden flag.
    #[serde(skip)]
    pub hidden: Vec<Vec<bool>>,
}

impl ExposureMap {
    /// Exposure frequency (0..1) of layer `p` at texel `i`.
    pub fn frequency(&self, p: usize, i: usize) -> f64 {
        if self.poses == 0 {
            return 0.0;
        }
        self.counts
            .get(p)
            .and_then(|c| c.get(i))
            .map_or(0.0, |&c| c as f64 / self.poses as f64)
    }

    pub fn layer(&self, id: &str) -> Option<&LayerExposure> {
        self.layers.iter().find(|l| l.id == id)
    }
}

/// Clipping targets matching `src/rig/render.ts`: `eye*_iris` → its `_white`, and when `face` has a
/// hidden area `face_detail` and `mouth_*` → `face`. Only targets drawn *below* the child are used.
///
/// T10 single rule (TS `defaultClippingTargets` in `render.ts`, same conditions for every render
/// path and pose): an iris layer (`eye{L,R}_iris`, numbered extras by suffix) clips to its side's
/// `eye{L,R}_white` when that white has a hidden area (`fillMask`) — the white must extend under the
/// iris, otherwise clipping would erase visible iris pixels at rest; `face_detail` and `mouth_*`
/// clip to `face` when `face` has a hidden area. The pose never changes the rule.
pub fn default_clip_targets(ids: &[String], fill: &[Vec<u8>]) -> Vec<Option<usize>> {
    let idx = |id: &str| ids.iter().position(|x| x == id);
    let has_fill = |t: usize| fill.get(t).is_some_and(|m| m.iter().any(|&a| a > 0));
    let face = idx("face");
    let face_has_fill = face.is_some_and(has_fill);
    let legacy = crate::core::legacy_eye_rig_enabled();
    ids.iter()
        .enumerate()
        .map(|(p, id)| {
            let target = if legacy {
                // Pre-T10 rule (measurement only): iris → white regardless of the white's fill.
                id.strip_suffix("_iris").and_then(|side| idx(&format!("{side}_white")))
            } else if crate::core::is_eye_iris_layer(id) {
                crate::core::eye_layer_side(id)
                    .and_then(|s| idx(&format!("eye{s}_white")))
                    .filter(|&t| has_fill(t))
            } else {
                None
            };
            let target = target.or_else(|| {
                (face_has_fill && (id == "face_detail" || id.starts_with("mouth")))
                    .then_some(face)
                    .flatten()
            });
            target.filter(|&t| t < p)
        })
        .collect()
}

/// Exposure map with the real `core::deform` (respects `core::with_seam_coupling`).
#[allow(clippy::too_many_arguments)]
pub fn compute_exposure_map(
    ids: &[String],
    visible: &[Vec<u8>],
    fill: &[Vec<u8>],
    lm: &Landmarks,
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
) -> ExposureMap {
    // T10: the eye rig needs the aperture lines of these masks (unless the caller derived them).
    let derived;
    let lm = if lm.eye_frames.is_some() {
        lm
    } else {
        derived = lm.with_eye_frames(ids, visible, w, h);
        &derived
    };
    let meshes: Vec<Mesh> = ids
        .iter()
        .enumerate()
        .map(|(p, id)| {
            let eff = effective_part_mask(&visible[p], fill.get(p).map(|f| f.as_slice()));
            art_mesh_for_part(&eff, w, h, MESH_DENSITY, id, lm)
        })
        .collect();
    compute_exposure_map_with(ids, visible, fill, &meshes, w, h, poses, |p, id, pose| {
        deform(p, id, lm, pose)
    })
}

pub(crate) struct PoseRaster {
    pub(crate) top_layer: Vec<i16>,
    pub(crate) top_src: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raster_pose<D>(
    ids: &[String],
    eff: &[Vec<u8>],
    meshes: &[Mesh],
    clip: &[Option<usize>],
    w: usize,
    h: usize,
    pose: &Pose,
    deform_fn: &D,
) -> PoseRaster
where
    D: Fn(Point, &str, &Pose) -> Point,
{
    let n = w * h;
    let mut top_layer = vec![-1i16; n];
    let mut top_src = vec![0u32; n];
    let is_target: Vec<bool> = (0..ids.len()).map(|p| clip.contains(&Some(p))).collect();
    let mut cov: Vec<Vec<bool>> = (0..ids.len())
        .map(|p| if is_target[p] { vec![false; n] } else { Vec::new() })
        .collect();
    for (p, id) in ids.iter().enumerate() {
        let mesh = &meshes[p];
        if mesh.triangles.is_empty() {
            continue;
        }
        let posed: Vec<Point> = mesh
            .vertices
            .iter()
            .map(|&v| deform_fn(v, id.as_str(), pose))
            .collect();
        let alpha = &eff[p];
        let clip_cov = clip[p].map(|t| std::mem::take(&mut cov[t]));
        for tri in mesh.triangles.as_chunks::<3>().0 {
            let (s0, s1, s2) = (
                mesh.vertices[tri[0]],
                mesh.vertices[tri[1]],
                mesh.vertices[tri[2]],
            );
            let (d0, d1, d2) = (posed[tri[0]], posed[tri[1]], posed[tri[2]]);
            let denom = (d1.y - d2.y) * (d0.x - d2.x) + (d2.x - d1.x) * (d0.y - d2.y);
            if denom.abs() < 1e-4 {
                continue;
            }
            let min_x = d0.x.min(d1.x).min(d2.x).floor().max(0.0) as i64;
            let max_x = d0.x.max(d1.x).max(d2.x).ceil().min((w - 1) as f64) as i64;
            let min_y = d0.y.min(d1.y).min(d2.y).floor().max(0.0) as i64;
            let max_y = d0.y.max(d1.y).max(d2.y).ceil().min((h - 1) as f64) as i64;
            for y in min_y..=max_y {
                let py = y as f64 + 0.5;
                for x in min_x..=max_x {
                    let px = x as f64 + 0.5;
                    let w0 = ((d1.y - d2.y) * (px - d2.x) + (d2.x - d1.x) * (py - d2.y)) / denom;
                    let w1 = ((d2.y - d0.y) * (px - d2.x) + (d0.x - d2.x) * (py - d2.y)) / denom;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < -1e-3 || w1 < -1e-3 || w2 < -1e-3 {
                        continue;
                    }
                    let sx = (w0 * s0.x + w1 * s1.x + w2 * s2.x).floor();
                    let sy = (w0 * s0.y + w1 * s1.y + w2 * s2.y).floor();
                    if sx < 0.0 || sy < 0.0 || sx >= w as f64 || sy >= h as f64 {
                        continue;
                    }
                    let si = sy as usize * w + sx as usize;
                    if alpha[si] <= OPAQUE {
                        continue;
                    }
                    let di = y as usize * w + x as usize;
                    if let Some(cc) = &clip_cov {
                        if !cc[di] {
                            continue;
                        }
                    }
                    top_layer[di] = p as i16;
                    top_src[di] = si as u32;
                    if is_target[p] {
                        cov[p][di] = true;
                    }
                }
            }
        }
        if let (Some(t), Some(cc)) = (clip[p], clip_cov) {
            cov[t] = cc;
        }
    }
    PoseRaster {
        top_layer,
        top_src,
    }
}

/// Uncovered pixels whose 4-connected uncovered component does not reach the canvas border.
fn enclosed_mask(covered: &[bool], w: usize, h: usize) -> Vec<bool> {
    let n = w * h;
    let mut outside = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let seed = |i: usize, outside: &mut Vec<bool>, stack: &mut Vec<usize>| {
        if !covered[i] && !outside[i] {
            outside[i] = true;
            stack.push(i);
        }
    };
    for x in 0..w {
        seed(x, &mut outside, &mut stack);
        seed((h - 1) * w + x, &mut outside, &mut stack);
    }
    for y in 0..h {
        seed(y * w, &mut outside, &mut stack);
        seed(y * w + w - 1, &mut outside, &mut stack);
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        if x > 0 {
            seed(i - 1, &mut outside, &mut stack);
        }
        if x + 1 < w {
            seed(i + 1, &mut outside, &mut stack);
        }
        if y > 0 {
            seed(i - w, &mut outside, &mut stack);
        }
        if y + 1 < h {
            seed(i + w, &mut outside, &mut stack);
        }
    }
    (0..n).map(|i| !covered[i] && !outside[i]).collect()
}

fn count_components(mask: &[bool], w: usize, h: usize) -> usize {
    let mut seen = vec![false; mask.len()];
    let mut comps = 0;
    let mut stack = Vec::new();
    for s in 0..mask.len() {
        if !mask[s] || seen[s] {
            continue;
        }
        comps += 1;
        seen[s] = true;
        stack.push(s);
        while let Some(i) = stack.pop() {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
    }
    comps
}

/// Exposure map with injected meshes and deformation (`deform_fn(point, layer_id, pose)`).
#[allow(clippy::too_many_arguments)]
pub fn compute_exposure_map_with<D>(
    ids: &[String],
    visible: &[Vec<u8>],
    fill: &[Vec<u8>],
    meshes: &[Mesh],
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
    deform_fn: D,
) -> ExposureMap
where
    D: Fn(Point, &str, &Pose) -> Point,
{
    let n = w * h;
    let layers = ids.len();
    let eff: Vec<Vec<u8>> = (0..layers)
        .map(|p| effective_part_mask(&visible[p], fill.get(p).map(|f| f.as_slice())))
        .collect();
    // Rest-pose visible owner (top-most layer with a visible pixel).
    let mut vis_owner = vec![-1i32; n];
    for (p, m) in visible.iter().enumerate() {
        for (o, &a) in vis_owner.iter_mut().zip(m) {
            if a > 0 {
                *o = p as i32;
            }
        }
    }
    let is_overlay = |p: usize, i: usize| vis_owner[i] >= 0 && (vis_owner[i] as usize) < p;
    let overlay_px: Vec<usize> = (0..layers)
        .map(|p| {
            let f = fill.get(p);
            (0..n)
                .filter(|&i| visible[p][i] == 0 && f.is_some_and(|f| f[i] > 0) && is_overlay(p, i))
                .count()
        })
        .collect();
    let hidden: Vec<Vec<bool>> = (0..layers)
        .map(|p| {
            let f = fill.get(p);
            (0..n)
                .map(|i| {
                    visible[p][i] == 0
                        && f.is_some_and(|f| f[i] > 0)
                        && eff[p][i] > OPAQUE
                        && !is_overlay(p, i)
                })
                .collect()
        })
        .collect();
    let has_hidden: Vec<bool> = hidden.iter().map(|m| m.iter().any(|&b| b)).collect();
    let clip = default_clip_targets(ids, fill);
    let mut counts: Vec<Vec<u16>> = (0..layers)
        .map(|p| if has_hidden[p] { vec![0u16; n] } else { Vec::new() })
        .collect();

    // Rest pass (not counted in frequency): baseline gaps, uncovered foreground, rest exposures.
    let rest = raster_pose(ids, &eff, meshes, &clip, w, h, &Pose::default(), &deform_fn);
    let rest_cov: Vec<bool> = rest.top_layer.iter().map(|&t| t >= 0).collect();
    let rest_enclosed = enclosed_mask(&rest_cov, w, h);
    let fg_any: Vec<bool> = (0..n).map(|i| eff.iter().any(|m| m[i] > OPAQUE)).collect();
    let mut rest_exposed = vec![0usize; layers];
    for i in 0..n {
        let t = rest.top_layer[i];
        if t >= 0 {
            let (p, s) = (t as usize, rest.top_src[i] as usize);
            if hidden[p][s] {
                rest_exposed[p] += 1;
            }
        }
    }
    let mut holes = HoleSummary {
        rest_enclosed_px: rest_enclosed.iter().filter(|&&g| g).count(),
        rest_uncovered_fg_px: (0..n).filter(|&i| fg_any[i] && !rest_cov[i]).count(),
        ..Default::default()
    };
    let mut hole_counts = vec![0u16; n];
    let mut hits: Vec<u64> = Vec::new();
    for pe in poses {
        let r = raster_pose(ids, &eff, meshes, &clip, w, h, &pe.pose, &deform_fn);
        hits.clear();
        let covered: Vec<bool> = r.top_layer.iter().map(|&t| t >= 0).collect();
        for (&t, &src) in r.top_layer.iter().zip(&r.top_src) {
            if t < 0 {
                continue;
            }
            let (p, s) = (t as usize, src as usize);
            if hidden[p][s] {
                hits.push(((p as u64) << 32) | s as u64);
            }
        }
        hits.sort_unstable();
        hits.dedup();
        for &k in &hits {
            let (p, s) = ((k >> 32) as usize, (k & 0xffff_ffff) as usize);
            counts[p][s] = counts[p][s].saturating_add(1);
        }
        let enclosed = enclosed_mask(&covered, w, h);
        let hole: Vec<bool> = (0..n).map(|i| enclosed[i] && !rest_enclosed[i]).collect();
        let hole_px = hole.iter().filter(|&&b| b).count();
        if hole_px > 0 {
            for (c, &b) in hole_counts.iter_mut().zip(&hole) {
                if b {
                    *c = c.saturating_add(1);
                }
            }
            let comps = count_components(&hole, w, h);
            holes.hole_poses += 1;
            holes.hole_px_total += hole_px;
            holes.hole_components_total += comps;
            holes.hole_components_max = holes.hole_components_max.max(comps);
            if hole_px > holes.hole_px_max {
                holes.hole_px_max = hole_px;
                holes.worst_pose = Some(pe.name.clone());
            }
        }
    }

    let pose_n = poses.len().max(1) as f64;
    let mut summaries = Vec::with_capacity(layers);
    let mut max_depth = 0.0f64;
    for p in 0..layers {
        let hidden_px = hidden[p].iter().filter(|&&b| b).count();
        let mut s = LayerExposure {
            id: ids[p].clone(),
            role: Some(infer_part_role(&ids[p], None)),
            hidden_px,
            rest_exposed_px: rest_exposed[p],
            overlay_px: overlay_px[p],
            ..Default::default()
        };
        let has_visible = visible[p].iter().any(|&a| a > 0);
        if has_hidden[p] {
            let dist = outside_distance(&visible[p], w, h);
            let mut depths: Vec<f32> = Vec::new();
            let mut sum = 0u64;
            let mut maxc = 0u16;
            for i in 0..n {
                let c = counts[p][i];
                if c > 0 {
                    depths.push(dist[i]);
                    sum += c as u64;
                    maxc = maxc.max(c);
                }
            }
            s.exposed_px = depths.len();
            s.unexposed_hidden_px = hidden_px - s.exposed_px;
            if !depths.is_empty() {
                s.mean_frequency = sum as f64 / (depths.len() as f64 * pose_n);
                s.max_frequency = maxc as f64 / pose_n;
                depths.sort_by(|a, b| a.total_cmp(b));
                if has_visible {
                    s.max_depth_px = depths.last().map(|&d| d as f64);
                    let k =
                        ((depths.len() as f64 * 0.95).ceil() as usize).clamp(1, depths.len()) - 1;
                    s.p95_depth_px = Some(depths[k] as f64);
                }
            }
        }
        max_depth = max_depth.max(s.max_depth_px.unwrap_or(0.0));
        summaries.push(s);
    }
    ExposureMap {
        width: w,
        height: h,
        poses: poses.len(),
        render: "artmesh-barycentric (core::art_mesh_for_part over mask∪fillMask, posed with \
                 core::deform, rasterised bottom-to-top with render.ts clipping; opaque = alpha>127)"
            .into(),
        hidden_px_total: summaries.iter().map(|s| s.hidden_px).sum(),
        exposed_px_total: summaries.iter().map(|s| s.exposed_px).sum(),
        max_depth_px: max_depth,
        layers: summaries,
        holes,
        counts,
        hole_counts,
        hidden,
    }
}

/// Colour ramp for exposure frequency (low = yellow, high = red).
fn ramp(f: f64) -> [u8; 3] {
    let f = f.clamp(0.0, 1.0);
    let stops: [(f64, [f64; 3]); 3] = [
        (0.0, [255.0, 226.0, 84.0]),
        (0.5, [244.0, 122.0, 32.0]),
        (1.0, [196.0, 18.0, 44.0]),
    ];
    let (a, b) = if f <= 0.5 { (stops[0], stops[1]) } else { (stops[1], stops[2]) };
    let t = (f - a.0) / (b.0 - a.0);
    [0, 1, 2].map(|c| (a.1[c] + (b.1[c] - a.1[c]) * t).round() as u8)
}

/// RGBA heatmap: exposed hidden texels by max frequency over layers (yellow → red, with a floor so
/// one-pose exposures stay visible), hidden-but-never-exposed texels faint blue, hole pixels magenta.
/// With `background` (straight RGBA of the artwork) the map is drawn over a dimmed grey copy;
/// without it the image is transparent elsewhere (app overlay).
pub fn exposure_heatmap_rgba(map: &ExposureMap, background: Option<&[u8]>) -> Vec<u8> {
    let n = map.width * map.height;
    let mut out = vec![0u8; n * 4];
    if let Some(bg) = background {
        for i in 0..n {
            let k = i * 4;
            if bg[k + 3] == 0 {
                continue;
            }
            let l = (0.299 * bg[k] as f64 + 0.587 * bg[k + 1] as f64 + 0.114 * bg[k + 2] as f64)
                * 0.35;
            let g = l.round().clamp(0.0, 255.0) as u8;
            out[k..k + 4].copy_from_slice(&[g, g, g, 255]);
        }
    }
    let pose_n = map.poses.max(1) as f64;
    for i in 0..n {
        let mut maxc = 0u16;
        let mut any_hidden = false;
        for (p, c) in map.counts.iter().enumerate() {
            if c.is_empty() {
                continue;
            }
            any_hidden |= map.hidden[p][i];
            maxc = maxc.max(c[i]);
        }
        let k = i * 4;
        let (rgb, a) = if map.hole_counts.get(i).copied().unwrap_or(0) > 0 {
            ([232, 36, 220], 170u8)
        } else if maxc > 0 {
            (ramp(maxc as f64 / pose_n), 230u8)
        } else if any_hidden {
            ([88, 140, 224], 56u8)
        } else {
            continue;
        };
        if background.is_some() {
            let af = a as f64 / 255.0;
            for c in 0..3 {
                out[k + c] = (rgb[c] as f64 * af + out[k + c] as f64 * (1.0 - af)).round() as u8;
            }
            out[k + 3] = 255;
        } else {
            out[k..k + 3].copy_from_slice(&rgb);
            out[k + 3] = a;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Mesh;

    /// Axis-aligned two-triangle mesh covering `[x0, x1] × [y0, y1]` (texel centres at +0.5).
    fn rect_mesh(x0: f64, y0: f64, x1: f64, y1: f64) -> Mesh {
        Mesh {
            vertices: vec![
                Point { x: x0, y: y0 },
                Point { x: x1, y: y0 },
                Point { x: x1, y: y1 },
                Point { x: x0, y: y1 },
            ],
            triangles: vec![0, 1, 2, 0, 2, 3],
            uv: Vec::new(),
        }
    }

    fn poses(shift: &[f64]) -> Vec<EnvelopePose> {
        shift
            .iter()
            .enumerate()
            .map(|(k, &s)| EnvelopePose {
                name: format!("p{k}"),
                group: "test".into(),
                pose: Pose {
                    hair: s,
                    ..Pose::default()
                },
            })
            .collect()
    }

    /// lower layer (`neck`) visible x<20 plus hidden fill x∈[20,30); upper (`clothes`) visible
    /// x∈[20,40). The upper slides right by `10 · hair` px.
    type TwoLayer = (Vec<String>, Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Mesh>);

    fn two_layer(fill_to: usize) -> TwoLayer {
        let (w, h) = (48usize, 16usize);
        let n = w * h;
        let mut lower = vec![0u8; n];
        let mut lower_fill = vec![0u8; n];
        let mut upper = vec![0u8; n];
        for y in 0..h {
            for x in 4..40 {
                let i = y * w + x;
                if x < 20 {
                    lower[i] = 255;
                } else {
                    upper[i] = 255;
                    if x < fill_to {
                        lower_fill[i] = 255;
                    }
                }
            }
        }
        let meshes = vec![
            rect_mesh(4.0, 0.0, fill_to.max(20) as f64, h as f64),
            rect_mesh(20.0, 0.0, 40.0, h as f64),
        ];
        (
            vec!["neck".into(), "clothes".into()],
            vec![lower, upper],
            vec![lower_fill, vec![0u8; n]],
            meshes,
        )
    }

    fn slide(p: Point, id: &str, pose: &Pose) -> Point {
        if id == "clothes" {
            Point {
                x: p.x + 10.0 * pose.hair,
                y: p.y,
            }
        } else {
            p
        }
    }

    #[test]
    fn exposure_counts_frequency_and_depth_of_hidden_texels_revealed_by_motion() {
        let (ids, vis, fill, meshes) = two_layer(30);
        let env = poses(&[0.0, 0.5, 1.0, 1.0]);
        let map = compute_exposure_map_with(&ids, &vis, &fill, &meshes, 48, 16, &env, slide);
        let neck = map.layer("neck").unwrap();
        assert_eq!(neck.hidden_px, 10 * 16);
        // Shift 5 exposes x∈[20,25) in 1 pose; shift 10 exposes x∈[20,30) in 2 poses.
        assert_eq!(neck.exposed_px, 10 * 16, "{neck:?}");
        assert!((neck.max_frequency - 3.0 / 4.0).abs() < 1e-9, "{neck:?}");
        let depth = neck.max_depth_px.unwrap();
        assert!((9.0..=10.0).contains(&depth), "{neck:?}");
        assert_eq!(neck.rest_exposed_px, 0);
        assert!((map.frequency(0, 22) - 0.75).abs() < 1e-9);
        assert!((map.frequency(0, 27) - 0.5).abs() < 1e-9);
        assert_eq!(map.holes.hole_px_total, 0, "{:?}", map.holes);
        // The clothes layer has no hidden texels.
        assert_eq!(map.layer("clothes").unwrap().hidden_px, 0);
    }

    #[test]
    fn overlay_fill_over_a_lower_visible_seam_is_not_a_hidden_texel() {
        // T9: the upper layer (`clothes`) receives the lower layer's seam column x = 19 as overlay
        // fill. It is drawn at rest above `neck`, so it is reported as overlay, never as a hidden
        // texel (no rest exposure, no frequency), and the lower layer's hidden texels are unchanged.
        let (ids, vis, mut fill, mut meshes) = two_layer(30);
        let (w, h) = (48usize, 16usize);
        for y in 0..h {
            fill[1][y * w + 19] = 255;
        }
        meshes[1] = rect_mesh(19.0, 0.0, 40.0, h as f64);
        let env = poses(&[0.0, 0.5, 1.0]);
        let map = compute_exposure_map_with(&ids, &vis, &fill, &meshes, w, h, &env, slide);
        let clothes = map.layer("clothes").unwrap();
        assert_eq!(clothes.overlay_px, h);
        assert_eq!(clothes.hidden_px, 0);
        assert_eq!(clothes.rest_exposed_px, 0);
        let neck = map.layer("neck").unwrap();
        assert_eq!(neck.hidden_px, 10 * 16);
        assert_eq!(neck.overlay_px, 0);
    }

    #[test]
    fn missing_fill_turns_exposure_into_holes_and_unexposed_fill_is_kept_and_reported() {
        let (mut ids, mut vis, mut fill, mut meshes) = two_layer(20);
        let env = poses(&[0.0, 1.0]);
        // Open to the canvas edge (top/bottom rows): background, not a hole.
        let open = compute_exposure_map_with(&ids, &vis, &fill, &meshes, 48, 16, &env, slide);
        assert_eq!(open.layer("neck").unwrap().hidden_px, 0);
        assert_eq!(open.holes.hole_px_total, 0, "{:?}", open.holes);
        // A static frame (bars on rows 0..2 and 14..16) encloses the seam: the opening is a hole.
        let n = 48 * 16;
        let mut frame = vec![0u8; n];
        for y in (0..2).chain(14..16) {
            for x in 0..48 {
                frame[y * 48 + x] = 255;
            }
        }
        ids.push("acc_frame".into());
        vis.push(frame);
        fill.push(vec![0u8; n]);
        let mut bars = rect_mesh(0.0, 0.0, 48.0, 2.0);
        let low = rect_mesh(0.0, 14.0, 48.0, 16.0);
        bars.vertices.extend(low.vertices);
        bars.triangles.extend(low.triangles.iter().map(|t| t + 4));
        meshes.push(bars);
        let map = compute_exposure_map_with(&ids, &vis, &fill, &meshes, 48, 16, &env, slide);
        assert_eq!(map.holes.hole_px_total, 10 * 12, "{:?}", map.holes);
        assert_eq!(map.holes.hole_poses, 1);
        assert_eq!(map.holes.hole_components_max, 1);
        assert_eq!(map.holes.worst_pose.as_deref(), Some("p1"));
        assert_eq!(map.holes.rest_enclosed_px, 0);

        // Fill deeper than any motion: the unexposed part is reported, not removed.
        let (ids, vis, fill, meshes) = two_layer(38);
        let map = compute_exposure_map_with(&ids, &vis, &fill, &meshes, 48, 16, &env, slide);
        let neck = map.layer("neck").unwrap();
        assert_eq!(neck.hidden_px, 18 * 16);
        assert_eq!(neck.exposed_px, 10 * 16);
        assert_eq!(neck.unexposed_hidden_px, 8 * 16);
        assert_eq!(map.holes.hole_px_total, 0);
        let rgba = exposure_heatmap_rgba(&map, None);
        assert_eq!(rgba[(5 * 48 + 22) * 4 + 3], 230, "exposed texel drawn");
        assert_eq!(rgba[(5 * 48 + 35) * 4 + 3], 56, "unexposed hidden texel drawn faint");
        assert_eq!(rgba[(5 * 48 + 10) * 4 + 3], 0, "visible texel untouched");
    }

    #[test]
    fn clipping_target_limits_child_and_targets_below_only() {
        let ids: Vec<String> = ["face", "eyeL_white", "eyeL_iris", "mouth_upper"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let n = 4;
        let mut fill = vec![vec![0u8; n]; 4];
        // T10: the iris clips to the white only when the white extends under it (hidden area).
        assert_eq!(default_clip_targets(&ids, &fill), vec![None, None, None, None]);
        fill[1][0] = 255;
        assert_eq!(
            default_clip_targets(&ids, &fill),
            vec![None, None, Some(1), None]
        );
        fill[0][0] = 255;
        assert_eq!(
            default_clip_targets(&ids, &fill),
            vec![None, None, Some(1), Some(0)]
        );
        // Numbered extra iris layers follow the suffix rule; a white above the iris is ignored.
        let ids: Vec<String> = ["eyeR_white", "eyeR_iris_2", "eyeL_iris", "eyeL_white"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let fill = vec![vec![255u8; n]; 4];
        assert_eq!(default_clip_targets(&ids, &fill), vec![None, Some(0), None, None]);
    }
}
