//! T10 eye seal checker: counts, per pose and per eye, (a) white/iris pixels shown outside the lid
//! silhouette, (b) skin/background gaps on the lid seams, and (c) breaks in the posed upper-lid
//! line, on the same ArtMesh raster (and clipping rule) as the exposure map (`exposure.rs`).
//!
//! The underlying raster is geometric (every layer drawn opaque), while screen-visible leak
//! (`leak_px`) gates white/iris samples by `raster::layer_opacity(id, pose) > 0.01` so closed eyes
//! where the app fades white/iris out (`eye <= 0.03` / `0.06`) report 0 screen leak and partial
//! blinks (`blink0.15`) are still checked; the opacity-ignored count is kept as
//! `geometric_leak_px` for diagnosis. Sampling runs on a head-local grid around each eye (1 px
//! steps along the head axes), so columns follow the head tilt.
//!
//! Definitions (side `s`; `I` = white/iris layers, `U` = upper lid, `Lo` = lower lid, skin = no layer
//! or a layer below every eye layer of the side, occluder = a non-eye layer above the eye stack):
//! - leak: at `eye = 0` every `I` sample; otherwise `I` samples above the posed lash line or below
//!   the posed lower line (the rest aperture lines moved by the upper / lower lid `deform`; 1px
//!   tolerance), and outside the aperture's span `I` samples showing a hidden (`fillMask`) texel.
//! - gap: skin samples directly under the lowest `U` sample of a column that end on an eye-layer
//!   sample (skin between the lid and the white / lower lid), plus skin samples between the lowest
//!   `I` sample and an `Lo` sample below it.
//! - lid breaks: runs of columns with no `U` (and no occluder in the lid's row window) strictly
//!   between the first and last `U` column.

use crate::core::{
    art_mesh_for_part, default_eye_frame, deform, effective_part_mask, eye_layer_side, eye_lines_at,
    is_eye_interior_layer, is_eye_iris_layer, HeadFrame, Landmarks, Mesh, Point, Pose,
};
use crate::rig::exposure::{default_clip_targets, raster_pose};
use crate::rig::seams::EnvelopePose;
use glam::DVec2;
use serde::Serialize;

pub const EYE_SEAL_VERSION: &str = "t10-v3";
const MESH_DENSITY: usize = 14;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EyeSealRow {
    pub pose: String,
    pub group: String,
    pub side: String,
    pub open: f64,
    /// Screen-visible white/iris leak (gated by `raster::layer_opacity(id, pose) > 0.01`).
    pub leak_px: usize,
    /// Opacity-ignored geometric white/iris leak (diagnostic).
    pub geometric_leak_px: usize,
    /// `leak_px` minus the rest-pose value for open eyes (rest leaks are segmented art inside
    /// the lash wing); equal to `leak_px` when this eye is closed (`eye = 0`).
    pub leak_excess_px: usize,
    pub gap_px: usize,
    pub lid_breaks: usize,
    /// Rest-relative increase of `gap_px` / `lid_breaks` (rest values are art, not defects).
    pub gap_excess_px: usize,
    pub lid_break_excess: usize,
    /// Iris visible-mask centroid displacement (px) in this pose.
    pub iris_shift_px: f64,
    /// Head-local downward travel (px) of the lash point above the eye centre column.
    pub lash_drop_px: f64,
    /// Image position of the first leak sample / first lid-line break (for reports).
    pub leak_at: Option<Point>,
    pub leak_layer: Option<String>,
    pub break_at: Option<Point>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EyeSealReport {
    pub version: &'static str,
    pub poses: usize,
    pub sides: Vec<String>,
    pub definition: &'static str,
    /// Rest-pose rows (baseline art: open eyes, centred gaze).
    pub rest: Vec<EyeSealRow>,
    pub leak_px_total: usize,
    pub geometric_leak_px_total: usize,
    pub leak_excess_px_total: usize,
    pub gap_excess_px_total: usize,
    pub lid_break_excess_total: usize,
    pub rows_with_defect: usize,
    pub max_iris_shift_px: f64,
    pub max_lash_drop_px: f64,
    pub rows: Vec<EyeSealRow>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Interior,
    Upper,
    Lower,
    OtherEye,
    Skin,
    Occluder,
}

struct SideLayers {
    side: char,
    interior: Vec<usize>,
    upper: Vec<usize>,
    lower: Vec<usize>,
    all: Vec<usize>,
    min_z: usize,
    iris_id: String,
    upper_id: String,
    lower_id: String,
    iris_centroid: Option<Point>,
}

fn side_layers(ids: &[String], visible: &[Vec<u8>], w: usize, side: char) -> Option<SideLayers> {
    let all: Vec<usize> = (0..ids.len())
        .filter(|&p| eye_layer_side(&ids[p]) == Some(side) && !ids[p].contains("_sub_"))
        .collect();
    if all.is_empty() {
        return None;
    }
    let interior: Vec<usize> = all.iter().copied().filter(|&p| is_eye_interior_layer(&ids[p])).collect();
    let upper: Vec<usize> = all.iter().copied().filter(|&p| ids[p].contains("_upper")).collect();
    let lower: Vec<usize> = all.iter().copied().filter(|&p| ids[p].contains("_lower")).collect();
    let iris = all.iter().copied().find(|&p| is_eye_iris_layer(&ids[p]));
    let iris_centroid = iris.and_then(|p| {
        let (mut sx, mut sy, mut n) = (0.0, 0.0, 0usize);
        for (i, &a) in visible[p].iter().enumerate() {
            if a >= 128 {
                sx += (i % w) as f64 + 0.5;
                sy += (i / w) as f64 + 0.5;
                n += 1;
            }
        }
        (n > 0).then(|| Point { x: sx / n as f64, y: sy / n as f64 })
    });
    Some(SideLayers {
        side,
        min_z: *all.iter().min().unwrap(),
        iris_id: iris.map_or(format!("eye{side}_iris"), |p| ids[p].clone()),
        upper_id: upper.first().map_or(format!("eye{side}_upper"), |&p| ids[p].clone()),
        lower_id: lower.first().map_or(format!("eye{side}_lower"), |&p| ids[p].clone()),
        interior,
        upper,
        lower,
        all,
        iris_centroid,
    })
}

struct Grid {
    u0: f64,
    v0: f64,
    cols: usize,
    rows: usize,
}

fn eye_grid(lm: &Landmarks, hf: &HeadFrame, side: char) -> Grid {
    let dist = hf.eye_dist;
    let eye = if side == 'L' { lm.eye_l } else { lm.eye_r };
    let (eu, ev) = hf.to_local(DVec2::new(eye.x, eye.y));
    let frame = lm
        .eye_frames
        .as_ref()
        .and_then(|f| if side == 'L' { f.l.clone() } else { f.r.clone() })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_eye_frame(eu, ev, dist));
    let (ua, ub) = (frame[0].u - 0.12 * dist, frame[frame.len() - 1].u + 0.12 * dist);
    let (va, vb) = (ev - 0.8 * dist, ev + 0.8 * dist);
    Grid {
        u0: ua.floor(),
        v0: va.floor(),
        cols: (ub - ua).ceil().max(1.0) as usize + 1,
        rows: (vb - va).ceil().max(1.0) as usize + 1,
    }
}

/// Seal metrics of one rasterised pose for one side.
#[allow(clippy::too_many_arguments)]
fn measure_side(
    top: &[i16],
    top_src: &[u32],
    layer_visible: &[bool],
    visible: &[Vec<u8>],
    w: usize,
    h: usize,
    hf: &HeadFrame,
    grid: &Grid,
    sl: &SideLayers,
    closed: bool,
    lines: &PosedLines,
) -> SideMeasure {
    let class_of = |t: i16| -> Class {
        if t < 0 {
            return Class::Skin;
        }
        let t = t as usize;
        if sl.interior.contains(&t) {
            Class::Interior
        } else if sl.upper.contains(&t) {
            Class::Upper
        } else if sl.lower.contains(&t) {
            Class::Lower
        } else if sl.all.contains(&t) {
            Class::OtherEye
        } else if t < sl.min_z {
            Class::Skin
        } else {
            Class::Occluder
        }
    };
    let mut cells = vec![Class::Skin; grid.cols * grid.rows];
    // Interior samples whose source texel is a hidden (`fillMask`) texel, not drawn art.
    let mut src_hidden = vec![false; grid.cols * grid.rows];
    // Whether the topmost layer at this cell is visible on screen (`layer_opacity > 0.01`).
    let mut screen_visible = vec![false; grid.cols * grid.rows];
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            let p = hf.to_world(grid.u0 + c as f64, grid.v0 + r as f64);
            let (x, y) = (p.x.floor(), p.y.floor());
            let cls = if x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
                Class::Skin
            } else {
                let di = y as usize * w + x as usize;
                let t = top[di];
                if t >= 0 {
                    src_hidden[r * grid.cols + c] = visible[t as usize][top_src[di] as usize] < 128;
                    screen_visible[r * grid.cols + c] = layer_visible[t as usize];
                }
                class_of(t)
            };
            cells[r * grid.cols + c] = cls;
        }
    }
    let at = |r: usize, c: usize| cells[r * grid.cols + c];
    let (mut leak, mut geometric_leak, mut gap) = (0usize, 0usize, 0usize);
    let mut leak_at: Option<(Point, usize)> = None;
    let world = |r: usize, c: usize| -> Point { hf.to_world(grid.u0 + c as f64, grid.v0 + r as f64).into() };
    let mut upper_rows: Option<(usize, usize)> = None;
    let mut col_has_upper = vec![false; grid.cols];
    for (c, has_upper) in col_has_upper.iter_mut().enumerate() {
        let col: Vec<Class> = (0..grid.rows).map(|r| at(r, c)).collect();
        let last_upper = col.iter().rposition(|&k| k == Class::Upper);
        let first_lower = col.iter().position(|&k| k == Class::Lower);
        if last_upper.is_some() {
            *has_upper = true;
            for (r, &k) in col.iter().enumerate() {
                if k == Class::Upper {
                    upper_rows = Some(upper_rows.map_or((r, r), |(a, b)| (a.min(r), b.max(r))));
                }
            }
        }
        for (r, &k) in col.iter().enumerate() {
            if k != Class::Interior {
                continue;
            }
            // Open eye: a white/iris sample leaks when it lies above the posed lash line or below
            // the posed lower line (the lids' actual edges), or, outside the aperture's span, when
            // it shows a hidden texel. Drawn art shown where it was drawn is never a leak.
            let _ = first_lower;
            let (gu, gv) = (grid.u0 + c as f64, grid.v0 + r as f64);
            let hidden_src = src_hidden[r * grid.cols + c];
            let outside = match lines.at(gu) {
                Some((lash, lower)) => gv < lash - 1.0 || gv > lower + 1.0,
                None => hidden_src,
            };
            if closed || outside {
                geometric_leak += 1;
                if screen_visible[r * grid.cols + c] {
                    leak += 1;
                    leak_at.get_or_insert_with(|| (world(r, c), c * grid.rows + r));
                }
            }
        }
        // Skin between the lid's lower edge and the next eye-layer sample.
        let skin_run = |start: usize, accept: &dyn Fn(Class) -> bool| -> usize {
            let mut n = 0;
            for &k in &col[start..] {
                if k == Class::Skin {
                    n += 1;
                } else {
                    return if accept(k) { n } else { 0 };
                }
            }
            0
        };
        if let Some(ru) = last_upper {
            gap += skin_run(ru + 1, &|k| {
                matches!(k, Class::Interior | Class::Lower | Class::OtherEye)
            });
        }
        if let Some(ri) = col.iter().rposition(|&k| k == Class::Interior) {
            gap += skin_run(ri + 1, &|k| k == Class::Lower);
        }
    }
    // Lid line breaks.
    let mut breaks = 0usize;
    let mut break_at: Option<Point> = None;
    if let (Some(first), Some(last)) = (
        col_has_upper.iter().position(|&b| b),
        col_has_upper.iter().rposition(|&b| b),
    ) {
        let (ra, rb) = upper_rows.unwrap_or((0, grid.rows - 1));
        let (ra, rb) = (ra.saturating_sub(2), (rb + 2).min(grid.rows - 1));
        let mut in_break = false;
        for (c, &has_upper) in col_has_upper.iter().enumerate().take(last + 1).skip(first) {
            let covered = has_upper || (ra..=rb).any(|r| at(r, c) == Class::Occluder);
            if !covered && !in_break {
                breaks += 1;
                break_at.get_or_insert_with(|| world((ra + rb) / 2, c));
            }
            in_break = !covered;
        }
    }
    SideMeasure {
        leak,
        geometric_leak,
        gap,
        breaks,
        leak_at,
        break_at,
    }
}

/// Posed aperture lines of one eye: the rest lash / lower samples moved by the upper / lower lid
/// fields (`deform` with the layer ids), as head-local `(u, v)` polylines sorted by `u`.
struct PosedLines {
    lash: Vec<(f64, f64)>,
    lower: Vec<(f64, f64)>,
}

impl PosedLines {
    fn interp(line: &[(f64, f64)], u: f64) -> Option<f64> {
        let (first, last) = (line.first()?, line.last()?);
        if u < first.0 || u > last.0 {
            return None;
        }
        for pair in line.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if u <= b.0 {
                let t = if b.0 > a.0 { (u - a.0) / (b.0 - a.0) } else { 0.0 };
                return Some(a.1 + (b.1 - a.1) * t);
            }
        }
        Some(last.1)
    }

    /// `(lash, lower)` at posed column `u`, or `None` outside the aperture's span.
    fn at(&self, u: f64) -> Option<(f64, f64)> {
        Some((Self::interp(&self.lash, u)?, Self::interp(&self.lower, u)?))
    }
}

struct SideMeasure {
    leak: usize,
    geometric_leak: usize,
    gap: usize,
    breaks: usize,
    leak_at: Option<(Point, usize)>,
    break_at: Option<Point>,
}

/// Runs the seal check over `poses` (plus the rest pose as baseline) with the real `deform`.
#[allow(clippy::too_many_arguments)]
pub fn check_eye_seal(
    ids: &[String],
    visible: &[Vec<u8>],
    fill: &[Vec<u8>],
    lm: &Landmarks,
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
) -> EyeSealReport {
    let derived;
    let lm = if lm.eye_frames.is_some() {
        lm
    } else {
        derived = lm.with_eye_frames(ids, visible, w, h);
        &derived
    };
    let eff: Vec<Vec<u8>> = (0..ids.len())
        .map(|p| effective_part_mask(&visible[p], fill.get(p).map(|f| f.as_slice())))
        .collect();
    let meshes: Vec<Mesh> = ids
        .iter()
        .enumerate()
        .map(|(p, id)| art_mesh_for_part(&eff[p], w, h, MESH_DENSITY, id, lm))
        .collect();
    check_eye_seal_with(ids, visible, fill, &meshes, lm, w, h, poses)
}

/// `check_eye_seal` with caller meshes (CLI reuses the motion-preview ArtMeshes).
#[allow(clippy::too_many_arguments)]
pub fn check_eye_seal_with(
    ids: &[String],
    visible: &[Vec<u8>],
    fill: &[Vec<u8>],
    meshes: &[Mesh],
    lm: &Landmarks,
    w: usize,
    h: usize,
    poses: &[EnvelopePose],
) -> EyeSealReport {
    let hf = HeadFrame::from_landmarks(lm, 0, 0);
    let eff: Vec<Vec<u8>> = (0..ids.len())
        .map(|p| effective_part_mask(&visible[p], fill.get(p).map(|f| f.as_slice())))
        .collect();
    let clip = default_clip_targets(ids, fill);
    let sides: Vec<SideLayers> = ['R', 'L']
        .iter()
        .filter_map(|&s| side_layers(ids, visible, w, s))
        .collect();
    let grids: Vec<Grid> = sides.iter().map(|s| eye_grid(lm, &hf, s.side)).collect();
    let deform_fn = |p: Point, id: &str, pose: &Pose| deform(p, id, lm, pose);
    let side_frame = |side: char| -> Vec<crate::core::EyeFrameSample> {
        let eye = if side == 'L' { lm.eye_l } else { lm.eye_r };
        let (eu, ev) = hf.to_local(DVec2::new(eye.x, eye.y));
        lm.eye_frames
            .as_ref()
            .and_then(|f| if side == 'L' { f.l.clone() } else { f.r.clone() })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default_eye_frame(eu, ev, hf.eye_dist))
    };
    let posed_lines = |sl: &SideLayers, pose: &Pose| -> PosedLines {
        let move_line = |pick: &dyn Fn(&crate::core::EyeFrameSample) -> f64, id: &str| {
            let mut pts: Vec<(f64, f64)> = side_frame(sl.side)
                .iter()
                .map(|s| {
                    let wp: Point = hf.to_world(s.u, pick(s)).into();
                    let q = deform(wp, id, lm, pose);
                    hf.to_local(DVec2::new(q.x, q.y))
                })
                .collect();
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            pts
        };
        PosedLines {
            lash: move_line(&|s| s.lash, &sl.upper_id),
            lower: move_line(&|s| s.lower, &sl.lower_id),
        }
    };
    let lash_point = |side: char| -> Point {
        let eye = if side == 'L' { lm.eye_l } else { lm.eye_r };
        let (eu, ev) = hf.to_local(DVec2::new(eye.x, eye.y));
        let frame = lm
            .eye_frames
            .as_ref()
            .and_then(|f| if side == 'L' { f.l.clone() } else { f.r.clone() })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default_eye_frame(eu, ev, hf.eye_dist));
        let (lash, _) = eye_lines_at(&frame, eu);
        hf.to_world(eu, lash).into()
    };
    let run = |name: &str, group: &str, pose: &Pose| -> Vec<EyeSealRow> {
        let raster = raster_pose(ids, &eff, meshes, &clip, w, h, pose, &deform_fn);
        let layer_visible: Vec<bool> = ids
            .iter()
            .map(|id| crate::rig::raster::layer_opacity(id, pose) > 0.01)
            .collect();
        sides
            .iter()
            .zip(&grids)
            .map(|(sl, grid)| {
                let open = if sl.side == 'L' { pose.eye_l } else { pose.eye_r };
                let lines = posed_lines(sl, pose);
                let m = measure_side(
                    &raster.top_layer,
                    &raster.top_src,
                    &layer_visible,
                    visible,
                    w,
                    h,
                    &hf,
                    grid,
                    sl,
                    open <= 1e-9,
                    &lines,
                );
                let iris_shift_px = sl.iris_centroid.map_or(0.0, |c| {
                    let d = deform(c, &sl.iris_id, lm, pose);
                    (d.x - c.x).hypot(d.y - c.y)
                });
                let lp = lash_point(sl.side);
                let (_, v0) = hf.to_local(DVec2::new(lp.x, lp.y));
                let moved = deform(lp, &sl.upper_id, lm, pose);
                let (_, v1) = hf.to_local(DVec2::new(moved.x, moved.y));
                EyeSealRow {
                    pose: name.to_string(),
                    group: group.to_string(),
                    side: sl.side.to_string(),
                    open,
                    leak_px: m.leak,
                    geometric_leak_px: m.geometric_leak,
                    leak_excess_px: m.leak,
                    gap_px: m.gap,
                    lid_breaks: m.breaks,
                    gap_excess_px: 0,
                    lid_break_excess: 0,
                    iris_shift_px,
                    lash_drop_px: v1 - v0,
                    leak_at: m.leak_at.map(|(p, _)| p),
                    leak_layer: m.leak_at.and_then(|(p, _)| {
                        let (x, y) = (p.x.floor(), p.y.floor());
                        (x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h)
                            .then(|| raster.top_layer[y as usize * w + x as usize])
                            .filter(|&t| t >= 0)
                            .map(|t| ids[t as usize].clone())
                    }),
                    break_at: m.break_at,
                }
            })
            .collect()
    };
    let rest = run("rest", "rest", &Pose::default());
    let mut rows: Vec<EyeSealRow> = Vec::new();
    for pe in poses {
        for mut row in run(&pe.name, &pe.group, &pe.pose) {
            if let Some(r) = rest.iter().find(|r| r.side == row.side) {
                row.gap_excess_px = row.gap_px.saturating_sub(r.gap_px);
                if row.open > 1e-9 {
                    row.leak_excess_px = row.leak_px.saturating_sub(r.leak_px);
                }
                row.lid_break_excess = row.lid_breaks.saturating_sub(r.lid_breaks);
            }
            rows.push(row);
        }
    }
    EyeSealReport {
        version: EYE_SEAL_VERSION,
        poses: poses.len(),
        sides: sides.iter().map(|s| s.side.to_string()).collect(),
        definition: "geometric ArtMesh raster with the shared clipping rule; head-local 1px grid \
                     per eye; leakPx = white/iris shown at eye=0, above the upper lid or below the \
                     lower lid, gated by raster::layer_opacity(id, pose) > 0.01 (screen-visible \
                     leak); geometricLeakPx = opacity-ignored geometric count for diagnosis; gap = \
                     skin/background between the lid and the white/lower lid; lid breaks = column \
                     runs without the upper lid or an occluder between its first and last column \
                     (excess = above the rest pose); diagnostic eye_seal/*.png renders are opaque \
                     (alpha=255, no alpha fades) and are not for pixel-quality judgment",
        leak_px_total: rows.iter().map(|r| r.leak_px).sum(),
        geometric_leak_px_total: rows.iter().map(|r| r.geometric_leak_px).sum(),
        leak_excess_px_total: rows.iter().map(|r| r.leak_excess_px).sum(),
        gap_excess_px_total: rows.iter().map(|r| r.gap_excess_px).sum(),
        lid_break_excess_total: rows.iter().map(|r| r.lid_break_excess).sum(),
        rows_with_defect: rows
            .iter()
            .filter(|r| r.leak_excess_px + r.gap_excess_px + r.lid_break_excess > 0)
            .count(),
        max_iris_shift_px: rows.iter().map(|r| r.iris_shift_px).fold(0.0, f64::max),
        max_lash_drop_px: rows.iter().map(|r| r.lash_drop_px).fold(0.0, f64::max),
        rest,
        rows,
    }
}

/// Envelope poses the seal check uses: the `eyeSeal` group (expressions × gaze extremes, partial
/// blinks) and the `eye` group (open-eye gaze corners, closed smile).
pub fn eye_seal_poses(env: &[EnvelopePose]) -> Vec<EnvelopePose> {
    env.iter()
        .filter(|p| p.group == "eyeSeal" || p.group == "eye")
        .cloned()
        .collect()
}

/// Hard (no anti-aliasing, opaque `alpha = 255` without app alpha fades) RGBA render of one pose
/// on the seal raster: each pixel takes the top layer's texel colour; also returns the posed
/// top-layer index per pixel (`-1` = none). For diagnostic `eye_seal/*.png` comparison only, not
/// for pixel-quality judgment.
#[allow(clippy::too_many_arguments)]
pub fn render_pose_rgba(
    ids: &[String],
    visible: &[Vec<u8>],
    fill: &[Vec<u8>],
    meshes: &[Mesh],
    textures: &[&[u8]],
    lm: &Landmarks,
    w: usize,
    h: usize,
    pose: &Pose,
) -> (Vec<u8>, Vec<i16>) {
    let eff: Vec<Vec<u8>> = (0..ids.len())
        .map(|p| effective_part_mask(&visible[p], fill.get(p).map(|f| f.as_slice())))
        .collect();
    let clip = default_clip_targets(ids, fill);
    let deform_fn = |p: Point, id: &str, pose: &Pose| deform(p, id, lm, pose);
    let raster = raster_pose(ids, &eff, meshes, &clip, w, h, pose, &deform_fn);
    let mut out = vec![0u8; w * h * 4];
    for i in 0..w * h {
        let t = raster.top_layer[i];
        if t < 0 {
            continue;
        }
        let s = raster.top_src[i] as usize;
        let tex = textures[t as usize];
        out[i * 4..i * 4 + 3].copy_from_slice(&tex[s * 4..s * 4 + 3]);
        out[i * 4 + 3] = 255;
    }
    (out, raster.top_layer)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::core::{derive_eye_frames, with_legacy_eye_rig, Landmarks};
    use crate::rig::seams::pose_envelope;

    pub(crate) fn lm_512() -> Landmarks {
        Landmarks::new(
            Point { x: 312.0, y: 200.0 },
            Point { x: 200.0, y: 200.0 },
            Point { x: 256.0, y: 270.0 },
            Point { x: 256.0, y: 320.0 },
            Point { x: 256.0, y: 60.0 },
        )
    }

    /// Analytic two-eye fixture (TS rebuilds the interior masks for the eye parity fixture).
    /// Returns ids bottom-to-top, visible masks (non-overlapping) and hidden areas.
    pub(crate) fn synthetic_eyes(
        w: usize,
        h: usize,
        lm: &Landmarks,
    ) -> (Vec<String>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let d = (lm.eye_l.x - lm.eye_r.x).hypot(lm.eye_l.y - lm.eye_r.y);
        let n = w * h;
        let ell = |x: f64, y: f64, cx: f64, cy: f64, rx: f64, ry: f64| {
            ((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2) <= 1.0
        };
        let mut ids = vec!["face".to_string()];
        let mut vis = vec![vec![0u8; n]];
        let mut fill = vec![vec![0u8; n]];
        for (side, eye) in [('R', lm.eye_r), ('L', lm.eye_l)] {
            let base = ids.len();
            for k in ["white", "iris", "lower", "upper"] {
                ids.push(format!("eye{side}_{k}"));
                vis.push(vec![0u8; n]);
                fill.push(vec![0u8; n]);
            }
            for y in 0..h {
                for x in 0..w {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let i = y * w + x;
                    let aperture = ell(px, py, eye.x, eye.y + 0.01 * d, 0.26 * d, 0.16 * d);
                    let socket = ell(px, py, eye.x, eye.y, 0.31 * d, 0.27 * d);
                    let iris = ell(px, py, eye.x, eye.y, 0.12 * d, 0.13 * d);
                    if aperture {
                        if iris {
                            vis[base + 1][i] = 255;
                        } else {
                            vis[base][i] = 255;
                        }
                    } else if socket && py < eye.y {
                        vis[base + 3][i] = 255;
                    } else if socket && py > eye.y + 0.06 * d {
                        vis[base + 2][i] = 255;
                    }
                    if socket && vis[base][i] == 0 {
                        fill[base][i] = 255;
                    }
                    if iris && vis[base + 1][i] == 0 {
                        fill[base + 1][i] = 255;
                    }
                }
            }
        }
        // Face: head ellipse minus the eye visible pixels; hidden area under every eye socket.
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let i = y * w + x;
                if ell(px, py, 256.0, 190.0, 170.0, 150.0) {
                    if vis[1..].iter().any(|m| m[i] > 0) {
                        fill[0][i] = 255;
                    } else {
                        vis[0][i] = 255;
                    }
                }
            }
        }
        (ids, vis, fill)
    }

    fn seal(legacy: bool) -> EyeSealReport {
        let (w, h) = (512usize, 400usize);
        let lm = lm_512();
        let (ids, vis, fill) = synthetic_eyes(w, h, &lm);
        let poses = eye_seal_poses(&pose_envelope().poses);
        with_legacy_eye_rig(legacy, || check_eye_seal(&ids, &vis, &fill, &lm, w, h, &poses))
    }

    #[test]
    fn eye_field_is_exact_identity_at_rest_inside_and_outside_the_aperture() {
        // T10: at the rest pose (eyes open, gaze centred, no expression) every eye layer — white
        // and iris texels outside the aperture included — stays exactly where it is drawn.
        let (w, h) = (512usize, 400usize);
        let lm0 = lm_512();
        let (ids, vis, _) = synthetic_eyes(w, h, &lm0);
        let lm = lm0.with_eye_frames(&ids, &vis, w, h);
        let rest = Pose::default();
        for id in ["eyeL_white", "eyeL_iris", "eyeL_upper", "eyeL_lower", "eyeR_white", "eyeR_iris_2", "eyeR_lash"] {
            for y in (100..300).step_by(3) {
                for x in (120..400).step_by(3) {
                    let p = Point { x: x as f64 + 0.25, y: y as f64 + 0.75 };
                    let q = deform(p, id, &lm, &rest);
                    assert_eq!((q.x, q.y), (p.x, p.y), "{id} at ({x},{y}) moved at rest");
                }
            }
        }
    }

    #[test]
    fn eye_frames_bound_every_visible_aperture_pixel() {
        let (w, h) = (512usize, 400usize);
        let lm = lm_512();
        let (ids, vis, _) = synthetic_eyes(w, h, &lm);
        let frames = derive_eye_frames(&lm, &ids, &vis, w, h);
        let hf = HeadFrame::from_landmarks(&lm, 0, 0);
        for (side, samples) in [('L', frames.l.as_ref().unwrap()), ('R', frames.r.as_ref().unwrap())] {
            assert!(samples.len() >= 12, "{side}: {}", samples.len());
            for (p, id) in ids.iter().enumerate() {
                if eye_layer_side(id) != Some(side) || !is_eye_interior_layer(id) {
                    continue;
                }
                for (i, &a) in vis[p].iter().enumerate() {
                    if a < 128 {
                        continue;
                    }
                    let c = DVec2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
                    let (u, v) = hf.to_local(c);
                    let (lash, lower) = eye_lines_at(samples, u);
                    assert!(v >= lash && v <= lower, "{id} px {i}: {v} outside [{lash}, {lower}]");
                }
            }
        }
    }

    #[test]
    fn sealed_rig_closes_without_leaks_or_gaps_and_keeps_full_motion() {
        let before = seal(true);
        let after = seal(false);
        assert_eq!(after.poses, before.poses);
        assert!(after.poses >= 51, "eyeSeal + eye groups: {}", after.poses);
        // Rest values (white hidden area left uncovered at the analytic sockets' corners) are the
        // baseline; open-eye rows are judged above it.
        println!(
            "T10_SEAL_FIXTURE legacy leak={} gap={} breaks={} defect_rows={} | T10 leak={} gap={} breaks={} defect_rows={}",
            before.leak_excess_px_total,
            before.gap_excess_px_total,
            before.lid_break_excess_total,
            before.rows_with_defect,
            after.leak_excess_px_total,
            after.gap_excess_px_total,
            after.lid_break_excess_total,
            after.rows_with_defect
        );
        for r in after
            .rows
            .iter()
            .filter(|r| r.leak_excess_px + r.gap_excess_px + r.lid_break_excess > 0)
        {
            println!(
                "T10_SEAL_FIXTURE_ROW {} {} open={} leak={} ({:?} @ {:?}) gap={} breaks={}",
                r.pose, r.side, r.open, r.leak_excess_px, r.leak_layer, r.leak_at, r.gap_excess_px, r.lid_break_excess
            );
        }
        assert!(
            before.geometric_leak_px_total > 1000,
            "legacy must leak geometrically: {}",
            before.geometric_leak_px_total
        );
        assert!(
            before.leak_excess_px_total > 100,
            "legacy must leak on screen in partial blinks: {}",
            before.leak_excess_px_total
        );
        // Not zero: 1–3 px slivers remain where the white reaches past the aperture end under the
        // lid wing (eye corners) and at partial blinks (mesh chords). Bounded per row and ≥ 99 %
        // below the pre-T10 rig; reported, not hidden.
        for r in &after.rows {
            assert!(r.leak_excess_px <= 4, "{r:?}");
            assert!(r.gap_excess_px <= 16, "{r:?}");
        }
        assert!(after.leak_excess_px_total * 100 <= before.leak_excess_px_total);
        assert!(after.gap_excess_px_total * 100 <= before.gap_excess_px_total);
        assert_eq!(after.lid_break_excess_total, 0);
        // Motion is never reduced: identical iris travel at the gaze extremes, and the lash travels at
        // least as far down as before whenever the eye closes without an expression shape.
        for (a, b) in after.rows.iter().zip(&before.rows) {
            assert_eq!((&a.pose, &a.side), (&b.pose, &b.side));
            // Pure gaze extremes (open eye, no expression): identical iris travel.
            if a.pose.starts_with("gaze:") {
                assert!((a.iris_shift_px - b.iris_shift_px).abs() < 1e-9, "{} {}", a.pose, a.side);
            }
            if a.pose.starts_with("seal:closed") || a.pose.starts_with("seal:blink") {
                assert!(
                    a.lash_drop_px >= b.lash_drop_px - 1e-9,
                    "{} {}: {} < {}",
                    a.pose,
                    a.side,
                    a.lash_drop_px,
                    b.lash_drop_px
                );
            }
        }
        assert!(after.max_iris_shift_px >= before.max_iris_shift_px - 1e-9);
    }

    #[test]
    fn closed_eye_screen_leak_is_zero_while_partial_blink_is_checked() {
        let env_poses = eye_seal_poses(&pose_envelope().poses);
        let closed_poses: Vec<_> = env_poses
            .iter()
            .filter(|p| p.pose.eye_l <= 1e-9 || p.pose.eye_r <= 1e-9)
            .collect();
        assert!(!closed_poses.is_empty());

        let blink015: Vec<_> = env_poses
            .iter()
            .filter(|p| p.name.contains("blink0.15"))
            .collect();
        assert!(!blink015.is_empty());
        for pe in &blink015 {
            assert!(crate::rig::raster::layer_opacity("eyeL_white", &pe.pose) > 0.01);
            assert!(crate::rig::raster::layer_opacity("eyeL_iris", &pe.pose) > 0.01);
            assert!(crate::rig::raster::layer_opacity("eyeR_white", &pe.pose) > 0.01);
            assert!(crate::rig::raster::layer_opacity("eyeR_iris", &pe.pose) > 0.01);
        }

        // Legacy rig leaks geometrically when closed, but screen-visible leak_px must be 0 at eye=0
        // and > 0 at blink0.15 (where layer_opacity > 0.01).
        let legacy = seal(true);
        let closed_legacy: Vec<_> = legacy.rows.iter().filter(|r| r.open <= 1e-9).collect();
        assert!(!closed_legacy.is_empty());
        assert!(closed_legacy.iter().any(|r| r.geometric_leak_px > 0));
        for r in &closed_legacy {
            assert_eq!(r.leak_px, 0, "closed eye must have 0 screen leak: {r:?}");
            assert_eq!(r.leak_excess_px, 0, "closed eye must have 0 screen excess: {r:?}");
        }
        let blink015_legacy: Vec<_> = legacy
            .rows
            .iter()
            .filter(|r| r.pose.contains("blink0.15"))
            .collect();
        assert!(!blink015_legacy.is_empty());
        assert!(blink015_legacy.iter().all(|r| r.leak_px > 0 && r.leak_px == r.geometric_leak_px));

        // Current T10 rig also reports 0 screen leak at eye=0.
        let current = seal(false);
        for r in current.rows.iter().filter(|r| r.open <= 1e-9) {
            assert_eq!(r.leak_px, 0, "closed eye must have 0 screen leak: {r:?}");
            assert_eq!(r.leak_excess_px, 0, "closed eye must have 0 screen excess: {r:?}");
        }
        assert!(current.leak_px_total <= current.geometric_leak_px_total);
    }

    /// Shared Rust/TS fixture for the T10 eye rig: analytic eye masks (TS rebuilds them), the
    /// derived aperture lines, and `deform` of eye-layer points over the `eyeSeal` poses.
    /// Regenerate with `FLAT2RIG_BLESS=1 cargo test eye_seal_parity`.
    #[test]
    fn eye_seal_parity_fixture_matches_rust_deform() {
        let (w, h) = (512usize, 400usize);
        let lm0 = lm_512();
        let (ids, vis, _) = synthetic_eyes(w, h, &lm0);
        let lm = lm0.with_eye_frames(&ids, &vis, w, h);
        let poses: Vec<_> = pose_envelope()
            .poses
            .iter()
            .filter(|p| p.group == "eyeSeal" || p.name == "rest" || p.name.starts_with("gaze:"))
            .cloned()
            .collect();
        let offsets = [
            (0.0, -0.3),
            (0.0, -0.2),
            (0.0, -0.1),
            (0.0, 0.0),
            (0.0, 0.1),
            (0.0, 0.2),
            (-0.2, -0.05),
            (0.2, 0.05),
            (0.3, -0.12),
            (-0.35, 0.0),
        ];
        let mut points = Vec::new();
        for (eye, side) in [(lm.eye_l, 'L'), (lm.eye_r, 'R')] {
            let d = 112.0;
            for (dx, dy) in offsets {
                for k in ["white", "iris", "upper", "lower", "lash_2"] {
                    points.push((format!("eye{side}_{k}"), eye.x + dx * d, eye.y + dy * d));
                }
            }
        }
        let cases: Vec<serde_json::Value> = poses
            .iter()
            .map(|pe| {
                let out: Vec<serde_json::Value> = points
                    .iter()
                    .map(|(id, x, y)| {
                        let q = deform(Point { x: *x, y: *y }, id, &lm, &pe.pose);
                        serde_json::json!([q.x, q.y])
                    })
                    .collect();
                serde_json::json!({ "pose": pe.name, "out": out })
            })
            .collect();
        let doc = serde_json::json!({
            "description": "T10 Rust/TS eye rig parity: analytic eye masks (rig::eye_seal::tests::synthetic_eyes, rebuilt in tests/pipeline.test.ts), core::derive_eye_frames samples, and core::deform of eye-layer points over the eyeSeal envelope poses (with their eye shape input). TS asserts deriveEyeFrames (1e-9) and engine.ts deform (1e-6).",
            "width": w,
            "height": h,
            "landmarks": lm0,
            "eyeFrames": lm.eye_frames,
            "points": points
                .iter()
                .map(|(id, x, y)| serde_json::json!({ "id": id, "point": [x, y] }))
                .collect::<Vec<_>>(),
            "cases": cases,
        });
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/eye-seal-parity.json");
        if std::env::var("FLAT2RIG_BLESS").is_ok() {
            std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
        }
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).expect("eye parity fixture present"),
        )
        .unwrap();
        assert_eq!(fixture["eyeFrames"], doc["eyeFrames"], "regenerate with FLAT2RIG_BLESS=1");
        let fa = fixture["cases"].as_array().unwrap();
        let da = doc["cases"].as_array().unwrap();
        assert_eq!(fa.len(), da.len());
        for (a, b) in fa.iter().zip(da) {
            assert_eq!(a["pose"], b["pose"]);
            for (p, q) in a["out"].as_array().unwrap().iter().zip(b["out"].as_array().unwrap()) {
                for k in 0..2 {
                    let (x, y) = (p[k].as_f64().unwrap(), q[k].as_f64().unwrap());
                    assert!((x - y).abs() < 1e-9, "{}: {x} vs {y}", a["pose"]);
                }
            }
        }
    }
}
