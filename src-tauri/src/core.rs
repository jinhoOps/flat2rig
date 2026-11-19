use base64::{engine::general_purpose::STANDARD, Engine};
use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

pub const IDS: [&str; 18] = [
    "hair_back",
    "body",
    "neck",
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
    "hair_front",
];
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}
impl Point {
    fn v(self) -> DVec2 {
        DVec2::new(self.x, self.y)
    }
}
impl From<DVec2> for Point {
    fn from(v: DVec2) -> Self {
        Self { x: v.x, y: v.y }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Landmarks {
    pub eye_l: Point,
    pub eye_r: Point,
    pub mouth: Point,
    pub chin: Point,
    pub crown: Point,
    /// T10 per-eye aperture lines derived from the visible eye masks (`derive_eye_frames`). Never
    /// serialized or persisted: every consumer that has the masks derives it again, and `deform`
    /// falls back to `default_eye_frame` when it is absent (TS `RigLandmarks.eyeFrames`).
    #[serde(skip)]
    pub eye_frames: Option<EyeFrames>,
    /// B2 torso attachment line (top of the body/clothes layers per head-local column) derived from
    /// the effective masks (`derive_torso_frame`). Never serialized: consumers with the masks
    /// derive it again (TS `RigLandmarks.torsoFrame`). `None` = no torso layer → hair is not
    /// coupled to the body field.
    #[serde(skip)]
    pub torso_frame: Option<Vec<TorsoFrameSample>>,
}
impl Landmarks {
    /// Five-point landmarks without derived eye frames.
    pub fn new(eye_l: Point, eye_r: Point, mouth: Point, chin: Point, crown: Point) -> Self {
        Self {
            eye_l,
            eye_r,
            mouth,
            chin,
            crown,
            eye_frames: None,
            torso_frame: None,
        }
    }
    /// Copy with eye frames derived from `ids`/`visible` (see `derive_eye_frames`).
    pub fn with_eye_frames(&self, ids: &[String], visible: &[Vec<u8>], w: usize, h: usize) -> Self {
        let mut out = self.clone();
        out.eye_frames = Some(derive_eye_frames(self, ids, visible, w, h));
        out
    }
    /// Copy with the torso attachment line derived from `effective` (`mask ∪ fillMask` per layer,
    /// see `derive_torso_frame`). Keeps any eye frames.
    pub fn with_torso_frame(&self, ids: &[String], effective: &[Vec<u8>], w: usize, h: usize) -> Self {
        let mut out = self.clone();
        out.torso_frame = derive_torso_frame(self, ids, effective, w, h);
        out
    }
    pub fn validate(&self, w: usize, h: usize) -> Result<(), String> {
        for p in [self.eye_l, self.eye_r, self.mouth, self.chin, self.crown] {
            if !p.x.is_finite()
                || !p.y.is_finite()
                || p.x < 0.
                || p.y < 0.
                || p.x >= w as f64
                || p.y >= h as f64
            {
                return Err("Landmark outside canvas".into());
            }
        }
        if self.eye_l.v().distance(self.eye_r.v()) < 4. || self.chin.y - self.crown.y < 8. {
            return Err("Invalid face landmarks".into());
        }
        Ok(())
    }
}
pub fn check_size(w: usize, h: usize) -> Result<usize, String> {
    if w == 0 || h == 0 || w > 2048 || h > 2048 {
        return Err("Image must be between 1 and 2048 pixels on each side".into());
    }
    Ok(w * h)
}
pub fn decode_bytes(encoded: &str, expected: usize) -> Result<Vec<u8>, String> {
    if encoded.len() > expected.div_ceil(3) * 4 + 4 {
        return Err("Payload exceeds expected size".into());
    }
    let bytes = STANDARD.decode(encoded).map_err(|e| e.to_string())?;
    if bytes.len() != expected {
        return Err("Payload dimensions do not match".into());
    }
    Ok(bytes)
}
pub fn foreground(data: &[u8], w: usize, h: usize, tolerance: f64) -> Vec<u8> {
    let n = w * h;
    let mut mask: Vec<u8> = data.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
    if tolerance <= 0. {
        return mask;
    }
    let samples: Vec<&[u8]> = [0, w - 1, (h - 1) * w, n - 1]
        .iter()
        .map(|i| &data[i * 4..i * 4 + 3])
        .collect();
    let mut seen = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    let visit = |i: usize,
                 seen: &mut Vec<bool>,
                 mask: &mut Vec<u8>,
                 queue: &mut std::collections::VecDeque<usize>| {
        if seen[i] {
            return;
        }
        seen[i] = true;
        if mask[i] < 4
            || samples.iter().any(|c| {
                (0..3)
                    .map(|j| (data[i * 4 + j] as f64 - c[j] as f64).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    < tolerance
            })
        {
            mask[i] = 0;
            queue.push_back(i);
        }
    };
    for x in 0..w {
        visit(x, &mut seen, &mut mask, &mut queue);
        visit((h - 1) * w + x, &mut seen, &mut mask, &mut queue);
    }
    for y in 0..h {
        visit(y * w, &mut seen, &mut mask, &mut queue);
        visit(y * w + w - 1, &mut seen, &mut mask, &mut queue);
    }
    while let Some(i) = queue.pop_front() {
        for j in neighbors(i, w, h) {
            visit(j, &mut seen, &mut mask, &mut queue);
        }
    }
    mask
}
fn neighbors(i: usize, w: usize, h: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(4);
    if !i.is_multiple_of(w) {
        out.push(i - 1);
    }
    if i % w + 1 < w {
        out.push(i + 1);
    }
    if i >= w {
        out.push(i - w);
    }
    if i + w < w * h {
        out.push(i + w);
    }
    out
}
#[inline]
fn ellipse_uv(u: f64, v: f64, cu: f64, cv: f64, rx: f64, ry: f64) -> bool {
    let dx = (u - cu) / rx.max(1.);
    let dy = (v - cv) / ry.max(1.);
    dx * dx + dy * dy <= 1.
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HeadFrame {
    pub origin: DVec2,
    pub ex: DVec2,
    pub ey: DVec2,
    pub eye_dist: f64,
    pub hh: f64,
    pub chin_v: f64,
    pub crown_v: f64,
}

impl HeadFrame {
    pub(crate) fn from_landmarks(lm: &Landmarks, w: usize, h: usize) -> Self {
        let eye_l = lm.eye_l.v();
        let eye_r = lm.eye_r.v();
        let origin = (eye_l + eye_r) * 0.5;
        let diff = eye_l - eye_r;
        let raw_len = diff.length();
        let eye_dist = raw_len.max(w as f64 * 0.05).max(1.0);
        // Character left (+ex) aligns with image +X unless landmarks are inverted.
        let ex = if raw_len > 1e-4 && diff.x >= 0.0 {
            diff / raw_len
        } else {
            DVec2::X
        };
        let ey = DVec2::new(-ex.y, ex.x);
        let crown_v = (lm.crown.v() - origin).dot(ey);
        let chin_v = (lm.chin.v() - origin).dot(ey);
        let hh = (chin_v - crown_v)
            .max(lm.chin.y - lm.crown.y)
            .max(h as f64 * 0.12)
            .max(1.0);
        Self {
            origin,
            ex,
            ey,
            eye_dist,
            hh,
            chin_v,
            crown_v,
        }
    }

    #[inline]
    pub(crate) fn to_local(self, p: DVec2) -> (f64, f64) {
        let d = p - self.origin;
        (d.dot(self.ex), d.dot(self.ey))
    }

    #[inline]
    pub(crate) fn to_world(self, u: f64, v: f64) -> DVec2 {
        self.origin + self.ex * u + self.ey * v
    }
}

pub fn separate(
    data: &[u8],
    w: usize,
    h: usize,
    lm: &Landmarks,
    tolerance: f64,
    alpha: Option<&[u8]>,
) -> Vec<Vec<u8>> {
    let fg = alpha
        .map(Vec::from)
        .unwrap_or_else(|| foreground(data, w, h, tolerance));
    let mut parts = vec![vec![0; w * h]; IDS.len()];
    let frame = HeadFrame::from_landmarks(lm, w, h);
    let dist = frame.eye_dist;
    let ew = dist * 0.32;
    let eh = dist * 0.16;
    let hh = frame.hh;
    let face_cv = frame.crown_v + hh * 0.55;
    let (eye_r_u, eye_r_v) = frame.to_local(lm.eye_r.v());
    let (eye_l_u, eye_l_v) = frame.to_local(lm.eye_l.v());
    let (mouth_u, mouth_v) = frame.to_local(lm.mouth.v());
    for i in 0..w * h {
        if fg[i] == 0 {
            continue;
        }
        let p = DVec2::new((i % w) as f64, (i / w) as f64);
        let (u, v) = frame.to_local(p);
        let mut idx = 0;
        if v > frame.chin_v + hh * 0.22 {
            idx = 1;
        } else if v > frame.chin_v && u.abs() < dist * 1.02 * 0.42 {
            idx = 2;
        }
        if ellipse_uv(u, v, 0.0, face_cv, dist * 1.02, hh * 0.49) {
            idx = 3;
        }
        if v < eye_l_v.min(eye_r_v) - hh * 0.22
            && ellipse_uv(u, v, 0.0, face_cv, dist * 1.02 * 1.1, hh * 0.49 * 1.1)
        {
            idx = 17;
        }
        for (side, (eu, ev)) in [(0, (eye_r_u, eye_r_v)), (1, (eye_l_u, eye_l_v))] {
            if ellipse_uv(u, v, eu, ev - eh * 2.1, ew * 1.02, eh * 0.3) {
                idx = 12 + side;
            }
            if ellipse_uv(u, v, eu, ev, ew, eh) {
                idx = 4 + side * 4;
                if ellipse_uv(u, v, eu, ev, ew * 0.46, eh * 0.95) {
                    idx += 1;
                }
                if v < ev - eh * 0.55 {
                    idx = 7 + side * 4;
                }
                if v > ev + eh * 0.65 {
                    idx = 6 + side * 4;
                }
            }
        }
        if ellipse_uv(u, v, mouth_u, mouth_v, dist * 0.18, dist * 0.065) {
            idx = if v < mouth_v - dist * 0.018 {
                16
            } else if v > mouth_v + dist * 0.018 {
                15
            } else {
                14
            };
        }
        parts[idx][i] = fg[i].min(data[i * 4 + 3]);
    }
    parts
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Mesh {
    pub vertices: Vec<Point>,
    pub triangles: Vec<usize>,
    pub uv: Vec<Point>,
}
fn mask_bounds(mask: &[u8], w: usize) -> Option<(usize, usize, usize, usize)> {
    if w == 0 || mask.is_empty() {
        return None;
    }
    let mut left = w;
    let mut top = usize::MAX;
    let mut right = 0;
    let mut bottom = 0;
    for (i, a) in mask.iter().enumerate() {
        if *a > 8 {
            left = left.min(i % w);
            right = right.max(i % w);
            top = top.min(i / w);
            bottom = bottom.max(i / w);
        }
    }
    (left < w).then_some((left, right, top, bottom))
}

pub fn mesh_for(mask: &[u8], w: usize, h: usize, density: usize) -> Mesh {
    mesh_for_with_anchors(mask, w, h, density, &[])
}

pub fn mesh_for_part(mask: &[u8], w: usize, h: usize, density: usize, lm: &Landmarks) -> Mesh {
    art_mesh_for_part(mask, w, h, density, "face", lm)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArtMeshRole {
    Eyelid,
    Iris,
    EyeAperture,
    LinearBrowOrLip,
    Hair,
    Face,
    StructuralBody,
    Generic,
}

fn classify_art_mesh_role(id: &str) -> ArtMeshRole {
    if id.is_empty() {
        ArtMeshRole::Generic
    } else if id.ends_with("_upper") && id.starts_with("eye")
        || id.ends_with("_lower") && id.starts_with("eye")
    {
        ArtMeshRole::Eyelid
    } else if id.starts_with("eye") && id.ends_with("_iris") {
        ArtMeshRole::Iris
    } else if (id.starts_with("eye") && id.ends_with("_white")) || id == "mouth_inner" {
        ArtMeshRole::EyeAperture
    } else if id.starts_with("brow") || id == "mouth_upper" || id == "mouth_lower" {
        ArtMeshRole::LinearBrowOrLip
    } else if id.starts_with("hair_") || id == "hair" {
        ArtMeshRole::Hair
    } else if id == "face" {
        ArtMeshRole::Face
    } else {
        ArtMeshRole::StructuralBody
    }
}

/// Traces ordered simple contour loops around `mask > 8` via pixel-crack edge following.
/// Returns `(points, is_outer)` where points lie slightly (`0.14px`) inside active pixels.
fn trace_contour_loops(
    mask: &[u8],
    w: usize,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Vec<(Vec<DVec2>, bool)> {
    let bw = right - left + 1;
    let bh = bottom - top + 1;
    let gw = bw + 1;
    let mut out_bits = vec![0u8; gw * (bh + 1)];
    let active = |lx: isize, ly: isize| -> bool {
        lx >= 0
            && ly >= 0
            && (lx as usize) < bw
            && (ly as usize) < bh
            && mask[(top + ly as usize) * w + (left + lx as usize)] > 8
    };

    // Directions: 0 = +x, 1 = -y, 2 = -x, 3 = +y (solid lies to the left of each directed edge).
    for py in 0..bh as isize {
        for px in 0..bw as isize {
            if !active(px, py) {
                continue;
            }
            let ux = px as usize;
            let uy = py as usize;
            if !active(px, py + 1) {
                out_bits[(uy + 1) * gw + ux] |= 1 << 0;
            }
            if !active(px + 1, py) {
                out_bits[(uy + 1) * gw + (ux + 1)] |= 1 << 1;
            }
            if !active(px, py - 1) {
                out_bits[uy * gw + (ux + 1)] |= 1 << 2;
            }
            if !active(px - 1, py) {
                out_bits[uy * gw + ux] |= 1 << 3;
            }
        }
    }

    const DX: [isize; 4] = [1, 0, -1, 0];
    const DY: [isize; 4] = [0, -1, 0, 1];
    let mut loops = Vec::new();
    for start_idx in 0..out_bits.len() {
        while out_bits[start_idx] != 0 {
            let start_dir = out_bits[start_idx].trailing_zeros() as usize;
            let mut cur_x = (start_idx % gw) as isize;
            let mut cur_y = (start_idx / gw) as isize;
            let mut dir = start_dir;
            let mut pts: Vec<DVec2> = Vec::new();
            let mut area2 = 0.0f64;
            let max_steps = gw * (bh + 1) * 4;
            for step in 0..max_steps {
                let idx = (cur_y as usize) * gw + (cur_x as usize);
                if (out_bits[idx] & (1 << dir)) == 0 {
                    break;
                }
                out_bits[idx] &= !(1 << dir);
                let cx = left as f64 + cur_x as f64;
                let cy = top as f64 + cur_y as f64;
                let mid = match dir {
                    0 => DVec2::new(cx + 0.5, cy - 0.14),
                    1 => DVec2::new(cx - 0.14, cy - 0.5),
                    2 => DVec2::new(cx - 0.5, cy + 0.14),
                    _ => DVec2::new(cx + 0.14, cy + 0.5),
                };
                if let Some(&prev) = pts.last() {
                    area2 += prev.x * mid.y - mid.x * prev.y;
                }
                pts.push(mid);
                cur_x += DX[dir];
                cur_y += DY[dir];
                let next_idx = (cur_y as usize) * gw + (cur_x as usize);
                if next_idx == start_idx && step > 0 && (out_bits[next_idx] & (1 << start_dir)) == 0 {
                    break;
                }
                let turn_order = [(dir + 1) & 3, dir, (dir + 3) & 3, (dir + 2) & 3];
                if let Some(&nd) = turn_order
                    .iter()
                    .find(|&&cand| (out_bits[next_idx] & (1 << cand)) != 0)
                {
                    dir = nd;
                } else {
                    break;
                }
            }
            if pts.len() >= 3 {
                let first = pts[0];
                let last = *pts.last().unwrap();
                area2 += last.x * first.y - first.x * last.y;
                loops.push((pts, area2 < 0.0));
            }
        }
    }
    loops
}

fn smooth_and_resample_loop(
    raw: &[DVec2],
    contour_step: f64,
    valid: &dyn Fn(DVec2) -> bool,
) -> Vec<(DVec2, DVec2)> {
    let m = raw.len();
    if m < 3 {
        return Vec::new();
    }
    let mut cur = raw.to_vec();
    if m >= 8 {
        for _ in 0..2 {
            let mut next = cur.clone();
            for i in 0..m {
                let prev = cur[(i + m - 1) % m];
                let p = cur[i];
                let nxt = cur[(i + 1) % m];
                let v1 = p - prev;
                let v2 = nxt - p;
                let l1 = v1.length();
                let l2 = v2.length();
                if l1 > 1e-6 && l2 > 1e-6 && v1.dot(v2) / (l1 * l2) < 0.35 {
                    continue;
                }
                let cand = prev * 0.25 + p * 0.5 + nxt * 0.25;
                if valid(cand) {
                    next[i] = cand;
                }
            }
            cur = next;
        }
    }

    let mut perim = 0.0;
    for i in 0..m {
        perim += cur[i].distance(cur[(i + 1) % m]);
    }
    if perim < 1.0 {
        return Vec::new();
    }
    let step = contour_step.min((perim / 6.0).max(0.8)).max(0.8);
    let mut sampled = Vec::new();
    let mut acc = 0.0;
    sampled.push(cur[0]);
    let mut prev_dir = (cur[1] - cur[0]).normalize_or_zero();
    for i in 0..m {
        let a = cur[i];
        let b = cur[(i + 1) % m];
        let seg = b - a;
        let seg_len = seg.length();
        if seg_len < 1e-6 {
            continue;
        }
        let dir = seg / seg_len;
        let last = *sampled.last().unwrap();
        if prev_dir.dot(dir) < 0.65 && last.distance(a) >= (step * 0.35).max(0.6) && valid(a) {
            sampled.push(a);
            acc = 0.0;
        }
        acc += seg_len;
        if acc >= step {
            let cand = if valid(b) { b } else { a };
            if sampled.last().unwrap().distance(cand) >= (step * 0.35).max(0.5) && valid(cand) {
                sampled.push(cand);
                acc = 0.0;
            }
        }
        prev_dir = dir;
    }
    if sampled.len() >= 2 && sampled[0].distance(*sampled.last().unwrap()) < step * 0.4 {
        sampled.pop();
    }
    let k = sampled.len();
    if k < 3 {
        return Vec::new();
    }
    (0..k)
        .map(|i| {
            let prev = sampled[(i + k - 1) % k];
            let nxt = sampled[(i + 1) % k];
            let tangent = nxt - prev;
            let in_normal = DVec2::new(tangent.y, -tangent.x).normalize_or_zero();
            (sampled[i], in_normal)
        })
        .collect()
}

fn chamfer_boundary_distance(
    mask: &[u8],
    w: usize,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Vec<u16> {
    let bw = right - left + 1;
    let bh = bottom - top + 1;
    let mut dist = vec![u16::MAX; bw * bh];
    let mut q = VecDeque::new();
    for ly in 0..bh {
        for lx in 0..bw {
            let idx = (top + ly) * w + (left + lx);
            if mask[idx] <= 8 {
                continue;
            }
            let x = left + lx;
            let y = top + ly;
            let touches_out = lx == 0
                || ly == 0
                || lx + 1 == bw
                || ly + 1 == bh
                || mask[y * w + (x - 1)] <= 8
                || mask[y * w + (x + 1)] <= 8
                || mask[(y - 1) * w + x] <= 8
                || mask[(y + 1) * w + x] <= 8;
            if touches_out {
                let d_idx = ly * bw + lx;
                dist[d_idx] = 10;
                q.push_back((lx, ly));
            }
        }
    }
    const OFF: [(isize, isize, u16); 8] = [
        (-1, 0, 10),
        (1, 0, 10),
        (0, -1, 10),
        (0, 1, 10),
        (-1, -1, 14),
        (1, -1, 14),
        (-1, 1, 14),
        (1, 1, 14),
    ];
    while let Some((cx, cy)) = q.pop_front() {
        let cur_d = dist[cy * bw + cx];
        for (dx, dy, cost) in OFF {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx < 0 || ny < 0 || (nx as usize) >= bw || (ny as usize) >= bh {
                continue;
            }
            let ux = nx as usize;
            let uy = ny as usize;
            if mask[(top + uy) * w + (left + ux)] <= 8 {
                continue;
            }
            let nd = cur_d.saturating_add(cost);
            let n_idx = uy * bw + ux;
            if nd < dist[n_idx] {
                dist[n_idx] = nd;
                q.push_back((ux, uy));
            }
        }
    }
    dist
}

/// Builds a contour-, margin- and role-aware Cubism ArtMesh for a Drawable mask.
pub fn art_mesh_for_part(
    mask: &[u8],
    w: usize,
    h: usize,
    density: usize,
    part_id: &str,
    lm: &Landmarks,
) -> Mesh {
    build_art_mesh(
        mask,
        w,
        h,
        density,
        classify_art_mesh_role(part_id),
        Some(lm),
        &[],
    )
}

/// Builds a contour- and landmark-aware Delaunay mesh covering active mask boundaries and interior.
pub fn mesh_for_with_anchors(
    mask: &[u8],
    w: usize,
    h: usize,
    density: usize,
    anchors: &[Point],
) -> Mesh {
    build_art_mesh(mask, w, h, density, ArtMeshRole::Generic, None, anchors)
}

/// B4: the ArtMesh outline sits this many pixels (Chebyshev) outside every texel with alpha > 0, so
/// a part's anti-aliased edge and faint strokes keep their bilinear falloff into transparent texels
/// instead of being cut by the mesh boundary (mirrored by `MESH_SUPPORT_RADIUS` in `src/engine.ts`).
pub const MESH_SUPPORT_RADIUS: usize = 2;

/// `src` dilated by `r` pixels (Chebyshev, separable max filter).
fn dilate_chebyshev(src: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let mut rows = vec![false; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let (lo, hi) = (x.saturating_sub(r), (x + r).min(w - 1));
            rows[y * w + x] = row[lo..=hi].iter().any(|&v| v);
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        let (lo, hi) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            out[y * w + x] = (lo..=hi).any(|yy| rows[yy * w + x]);
        }
    }
    out
}

/// B4: the mask an ArtMesh is built on — every texel with alpha > 0 (visible + hidden, including
/// anti-aliased edges) grown by [`MESH_SUPPORT_RADIUS`], as `0` / `255`.
pub fn mesh_support_mask(mask: &[u8], w: usize, h: usize) -> Vec<u8> {
    let base: Vec<bool> = mask.iter().map(|&a| a > 0).collect();
    dilate_chebyshev(&base, w, h, MESH_SUPPORT_RADIUS)
        .into_iter()
        .map(|b| if b { 255 } else { 0 })
        .collect()
}

/// B4: pixel centres an ArtMesh must cover at rest — alpha > 0 texels and their 8-neighbours (the
/// bilinear footprint of the edge texels).
fn mesh_coverage_target(mask: &[u8], w: usize, h: usize) -> Vec<bool> {
    let base: Vec<bool> = mask.iter().map(|&a| a > 0).collect();
    dilate_chebyshev(&base, w, h, 1)
}

/// Delaunay triangulation of `verts` with the ArtMesh sliver / hole filter: returns every Delaunay
/// triangle and whether the filter keeps it.
fn triangulate_filtered(
    verts: &[Point],
    valid: &dyn Fn(DVec2) -> bool,
    span: f64,
    ring_step: f64,
) -> (Vec<[usize; 3]>, Vec<bool>) {
    let points: Vec<_> = verts.iter().map(|p| delaunator::Point { x: p.x, y: p.y }).collect();
    let result = delaunator::triangulate(&points);
    let tris: Vec<[usize; 3]> = result.triangles.as_chunks::<3>().0.to_vec();
    let thin_part = verts.len() < 40;
    let max_edge = if thin_part {
        (span * 0.90).max(18.0)
    } else {
        (ring_step * 2.75).max(18.0)
    };
    let min_cov = if thin_part { 52 } else { 68 };
    let keep = tris
        .iter()
        .map(|ids| {
            let a = verts[ids[0]].v();
            let b = verts[ids[1]].v();
            let c = verts[ids[2]].v();
            let max_e = a.distance(b).max(a.distance(c)).max(b.distance(c));
            if max_e > max_edge {
                return false;
            }
            let min_aspect = if thin_part { 0.025 } else { 0.048 };
            let cross = ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs();
            if cross < min_aspect * max_e * max_e {
                return false;
            }
            let centroid = (a + b + c) / 3.0;
            let q1 = a * 0.5 + b * 0.25 + c * 0.25;
            let q2 = a * 0.25 + b * 0.5 + c * 0.25;
            let q3 = a * 0.25 + b * 0.25 + c * 0.5;
            if !valid(centroid) || !valid(q1) || !valid(q2) || !valid(q3) {
                return false;
            }
            let steps: usize = 4;
            let mut total = 0usize;
            let mut valid_samples = 0usize;
            for u in 0..=steps {
                for v in 0..=steps - u {
                    total += 1;
                    let p = a + (b - a) * (u as f64 / steps as f64) + (c - a) * (v as f64 / steps as f64);
                    if valid(p) {
                        valid_samples += 1;
                    }
                }
            }
            valid_samples * 100 >= total * min_cov
        })
        .collect();
    (tris, keep)
}

/// B4: coverage-target pixels (inside `bbox` = `(left, right, top, bottom)`) whose centre no kept
/// triangle covers under the renderer's top-left rule, as a full-canvas flag grid.
fn uncovered_target_pixels(
    verts: &[Point],
    tris: &[[usize; 3]],
    keep: &[bool],
    target: &[bool],
    bbox: (usize, usize, usize, usize),
    w: usize,
    h: usize,
) -> Vec<bool> {
    let (left, right, top, bottom) = bbox;
    let mut covered = vec![false; w * h];
    for (ids, &k) in tris.iter().zip(keep) {
        if k {
            let tri = [verts[ids[0]], verts[ids[1]], verts[ids[2]]];
            crate::rig::raster::rasterize_triangle(tri, w, h, |x, y, _| covered[y * w + x] = true);
        }
    }
    let mut missed = vec![false; w * h];
    for y in top..=bottom {
        for x in left..=right {
            let i = y * w + x;
            missed[i] = target[i] && !covered[i];
        }
    }
    missed
}

/// Builds an ArtMesh over the B4 support of `mask` (alpha > 0 grown by [`MESH_SUPPORT_RADIUS`]).
/// After the usual contour / role / Delaunay construction and sliver filtering, any Delaunay
/// triangle that owns (top-left rule, `rig::raster::rasterize_triangle`) an uncovered pixel centre of
/// the coverage target is kept, so the rest mesh covers every alpha > 0 texel.
fn build_art_mesh(
    mask: &[u8],
    w: usize,
    h: usize,
    density: usize,
    role: ArtMeshRole,
    lm: Option<&Landmarks>,
    anchors: &[Point],
) -> Mesh {
    let target = mesh_coverage_target(mask, w, h);
    let support = mesh_support_mask(mask, w, h);
    let mask: &[u8] = &support;
    let mut mesh = Mesh {
        vertices: Vec::new(),
        triangles: Vec::new(),
        uv: Vec::new(),
    };
    let Some((left, right, top, bottom)) = mask_bounds(mask, w) else {
        return mesh;
    };
    let bw = right - left + 1;
    let bh = bottom - top + 1;
    let span = bw.max(bh) as f64;
    let d_clamped = density.clamp(6, 32) as f64;
    let (ring_step, contour_step) = match role {
        ArtMeshRole::Eyelid
        | ArtMeshRole::Iris
        | ArtMeshRole::EyeAperture
        | ArtMeshRole::LinearBrowOrLip => {
            let r = (span / (d_clamped * 1.15)).clamp(2.2, 14.0);
            (r, (r * 0.60).clamp(1.8, 8.5))
        }
        ArtMeshRole::Hair | ArtMeshRole::Face => {
            let r = (span / d_clamped).clamp(3.0, 22.0);
            (r, (r * 0.52).clamp(2.4, 12.0))
        }
        ArtMeshRole::StructuralBody | ArtMeshRole::Generic => {
            let r = (span / d_clamped).clamp(3.0, 28.0);
            (r, (r * 0.48).clamp(2.8, 13.0))
        }
    };

    let valid = |p: DVec2| {
        let px = p.x + 1e-6;
        let py = p.y + 1e-6;
        px >= 0.
            && py >= 0.
            && (px as usize) < w
            && (py as usize) < h
            && mask[py as usize * w + px as usize] > 8
    };

    let mut vert_normals: Vec<Option<DVec2>> = Vec::new();
    let push_if_far = |verts: &mut Vec<Point>,
                       normals: &mut Vec<Option<DVec2>>,
                       cand: DVec2,
                       min_sq: f64,
                       in_n: Option<DVec2>| {
        if !valid(cand) {
            return;
        }
        for (i, v) in verts.iter().enumerate() {
            let dx = v.x - cand.x;
            let dy = v.y - cand.y;
            let dist_sq = dx * dx + dy * dy;
            // Preserve thin opposite-side silhouette walls (>=0.42px apart) while declustering same-side edges.
            let allowed_sq = match (in_n, normals[i]) {
                (Some(na), Some(nb)) if na.dot(nb) < 0.0 => min_sq.min(0.42 * 0.42),
                _ => min_sq,
            };
            if dist_sq < allowed_sq {
                return;
            }
        }
        verts.push(cand.into());
        normals.push(in_n);
    };

    // 1. Ordered closed contour rings (outer perimeter + internal holes) with corner preservation.
    let raw_loops = trace_contour_loops(mask, w, left, right, top, bottom);
    let contour_min_sq = (contour_step * 0.46).clamp(1.2, 8.0).powi(2);
    let inset_min_sq = (ring_step * 0.42).max(1.5).powi(2);
    let mut inset_candidates: Vec<(DVec2, DVec2)> = Vec::new();
    for (raw_pts, is_outer) in &raw_loops {
        let resampled = smooth_and_resample_loop(raw_pts, contour_step, &valid);
        for &(pt, in_n) in &resampled {
            push_if_far(
                &mut mesh.vertices,
                &mut vert_normals,
                pt,
                contour_min_sq,
                Some(in_n),
            );
            if *is_outer && in_n.length_squared() > 0.5 {
                inset_candidates.push((pt + in_n * (ring_step * 0.55), in_n));
            }
        }
    }

    // 2. Inset contour belt (ArtMesh smooth edge strip around outer contour).
    for &(pt, in_n) in &inset_candidates {
        push_if_far(
            &mut mesh.vertices,
            &mut vert_normals,
            pt,
            inset_min_sq,
            Some(in_n),
        );
    }

    // 3. Part-role structural deformation guide vertices (replacing uniform rectangular grids).
    let guide_min_sq = (ring_step * 0.40).max(1.4).powi(2);
    let dist_map = chamfer_boundary_distance(mask, w, left, right, top, bottom);
    match role {
        ArtMeshRole::Eyelid => {
            let cols = ((bw as f64 / (ring_step * 0.55)).round() as usize).clamp(7, 15);
            for c in 0..cols {
                let x = (left as f64 + (c as f64 / (cols - 1) as f64) * (bw.saturating_sub(1)) as f64)
                    .round() as usize;
                let mut y0 = None;
                let mut y1 = 0usize;
                for y in top..=bottom {
                    if mask[y * w + x] > 8 {
                        if y0.is_none() {
                            y0 = Some(y);
                        }
                        y1 = y;
                    }
                }
                if let Some(yt) = y0 {
                    let span_y = (y1 - yt) as f64;
                    if span_y >= 2.0 {
                        for frac in [0.22, 0.50, 0.78] {
                            push_if_far(
                                &mut mesh.vertices,
                                &mut vert_normals,
                                DVec2::new(x as f64 + 0.5, yt as f64 + 0.5 + span_y * frac),
                                guide_min_sq * 0.75,
                                None,
                            );
                        }
                    } else {
                        push_if_far(
                            &mut mesh.vertices,
                            &mut vert_normals,
                            DVec2::new(x as f64 + 0.5, (yt + y1) as f64 * 0.5 + 0.5),
                            guide_min_sq * 0.75,
                            None,
                        );
                    }
                }
            }
        }
        ArtMeshRole::Iris => {
            let mut sx = 0.0;
            let mut sy = 0.0;
            let mut cnt = 0.0;
            for y in top..=bottom {
                for x in left..=right {
                    if mask[y * w + x] > 8 {
                        sx += x as f64 + 0.5;
                        sy += y as f64 + 0.5;
                        cnt += 1.0;
                    }
                }
            }
            if cnt > 0.0 {
                let cx = sx / cnt;
                let cy = sy / cnt;
                let rx = (bw as f64 * 0.5).max(1.0);
                let ry = (bh as f64 * 0.5).max(1.0);
                push_if_far(
                    &mut mesh.vertices,
                    &mut vert_normals,
                    DVec2::new(cx, cy),
                    guide_min_sq * 0.5,
                    None,
                );
                let sectors = 12usize;
                for r_frac in [0.36, 0.68] {
                    for s in 0..sectors {
                        let ang = (s as f64 / sectors as f64) * std::f64::consts::TAU;
                        let p = DVec2::new(cx + rx * r_frac * ang.cos(), cy + ry * r_frac * ang.sin());
                        push_if_far(
                            &mut mesh.vertices,
                            &mut vert_normals,
                            p,
                            guide_min_sq * 0.65,
                            None,
                        );
                    }
                }
            }
        }
        ArtMeshRole::EyeAperture => {
            let cx = (left + right) as f64 * 0.5 + 0.5;
            let cy = (top + bottom) as f64 * 0.5 + 0.5;
            let rx = (bw as f64 * 0.5).max(1.0);
            let ry = (bh as f64 * 0.5).max(1.0);
            push_if_far(
                &mut mesh.vertices,
                &mut vert_normals,
                DVec2::new(cx, cy),
                guide_min_sq * 0.5,
                None,
            );
            for s in 0..10 {
                let ang = (s as f64 / 10.0) * std::f64::consts::TAU;
                push_if_far(
                    &mut mesh.vertices,
                    &mut vert_normals,
                    DVec2::new(cx + rx * 0.50 * ang.cos(), cy + ry * 0.50 * ang.sin()),
                    guide_min_sq * 0.70,
                    None,
                );
            }
        }
        ArtMeshRole::LinearBrowOrLip => {
            let cols = ((bw as f64 / (ring_step * 0.55)).round() as usize).clamp(7, 13);
            for c in 0..cols {
                let x = (left as f64 + (c as f64 / (cols - 1) as f64) * (bw.saturating_sub(1)) as f64)
                    .round() as usize;
                let mut sum_y = 0.0;
                let mut y_cnt = 0.0;
                let mut min_y = bottom;
                let mut max_y = top;
                for y in top..=bottom {
                    if mask[y * w + x] > 8 {
                        sum_y += y as f64 + 0.5;
                        y_cnt += 1.0;
                        min_y = min_y.min(y);
                        max_y = max_y.max(y);
                    }
                }
                if y_cnt > 0.0 {
                    let mid_y = sum_y / y_cnt;
                    push_if_far(
                        &mut mesh.vertices,
                        &mut vert_normals,
                        DVec2::new(x as f64 + 0.5, mid_y),
                        guide_min_sq * 0.6,
                        None,
                    );
                    if max_y > min_y + 3 {
                        for frac in [0.25, 0.75] {
                            push_if_far(
                                &mut mesh.vertices,
                                &mut vert_normals,
                                DVec2::new(
                                    x as f64 + 0.5,
                                    min_y as f64 + 0.5 + (max_y - min_y) as f64 * frac,
                                ),
                                guide_min_sq * 0.75,
                                None,
                            );
                        }
                    }
                }
            }
        }
        ArtMeshRole::Hair => {
            // Root-to-tip lock flow ribs: sparser at scalp root, denser along flexible lower strands.
            let mut y = top as f64 + ring_step * 0.5;
            let root_cutoff = top as f64 + bh as f64 * 0.36;
            while y <= bottom as f64 {
                let iy = (y.round() as usize).clamp(top, bottom);
                let row_step = if y < root_cutoff { ring_step * 1.12 } else { ring_step * 0.68 };
                let mut x = left;
                while x <= right {
                    if mask[iy * w + x] <= 8 {
                        x += 1;
                        continue;
                    }
                    let seg_start = x;
                    while x < right && mask[iy * w + x + 1] > 8 {
                        x += 1;
                    }
                    let seg_w = (x - seg_start + 1) as f64;
                    let fracs: &[f64] = if seg_w >= ring_step * 2.2 {
                        &[0.20, 0.40, 0.60, 0.80]
                    } else if seg_w >= ring_step * 1.2 {
                        &[0.25, 0.50, 0.75]
                    } else {
                        &[0.50]
                    };
                    for &f in fracs {
                        push_if_far(
                            &mut mesh.vertices,
                            &mut vert_normals,
                            DVec2::new(seg_start as f64 + 0.5 + (seg_w - 1.0) * f, iy as f64 + 0.5),
                            (row_step * 0.48).max(1.6).powi(2),
                            None,
                        );
                    }
                    x += 1;
                }
                y += row_step;
            }
        }
        ArtMeshRole::Face => {
            if let Some(lm) = lm {
                let mid_x = (lm.eye_l.x + lm.eye_r.x) * 0.5;
                let eye_d = (lm.eye_l.x - lm.eye_r.x).abs().max(8.0);
                // Midline symmetry spine (crown -> nose -> mouth -> chin).
                let steps = ((bh as f64 / (ring_step * 0.65)).round() as usize).clamp(5, 18);
                for s in 1..steps {
                    let fy = top as f64 + (s as f64 / steps as f64) * bh as f64;
                    let t = ((fy - lm.crown.y) / (lm.chin.y - lm.crown.y).max(1.0)).clamp(0.0, 1.0);
                    let fx = mid_x + (lm.chin.x - mid_x) * t;
                    push_if_far(
                        &mut mesh.vertices,
                        &mut vert_normals,
                        DVec2::new(fx, fy),
                        guide_min_sq * 0.75,
                        None,
                    );
                }
                for pt in [lm.chin, lm.mouth, lm.eye_l, lm.eye_r, lm.crown] {
                    push_if_far(
                        &mut mesh.vertices,
                        &mut vert_normals,
                        pt.v(),
                        guide_min_sq * 0.5,
                        None,
                    );
                }
                for center in [lm.eye_r.v(), lm.eye_l.v(), lm.mouth.v()] {
                    let r = (eye_d * 0.22).max(ring_step * 0.7);
                    for s in 0..6 {
                        let ang = (s as f64 / 6.0) * std::f64::consts::TAU;
                        push_if_far(
                            &mut mesh.vertices,
                            &mut vert_normals,
                            center + DVec2::new(r * ang.cos(), r * 0.75 * ang.sin()),
                            guide_min_sq * 0.75,
                            None,
                        );
                    }
                }
            }
        }
        ArtMeshRole::StructuralBody => {
            if let Some(lm) = lm {
                let mid_x = (lm.eye_l.x + lm.eye_r.x) * 0.5;
                let steps = ((bh as f64 / (ring_step * 0.75)).round() as usize).clamp(4, 16);
                for s in 1..steps {
                    let fy = top as f64 + (s as f64 / steps as f64) * bh as f64;
                    push_if_far(
                        &mut mesh.vertices,
                        &mut vert_normals,
                        DVec2::new(mid_x, fy),
                        guide_min_sq * 0.75,
                        None,
                    );
                }
            }
        }
        ArtMeshRole::Generic => {
            for a in anchors {
                push_if_far(
                    &mut mesh.vertices,
                    &mut vert_normals,
                    a.v(),
                    guide_min_sq * 0.5,
                    None,
                );
            }
        }
    }

    // 4. Organic distance-field interior rings (hexagonal contour-following bands, zero rigid XY checkerboard).
    let max_d = dist_map
        .iter()
        .copied()
        .filter(|&d| d != u16::MAX)
        .max()
        .unwrap_or(10) as f64
        / 10.0;
    let band_gap = (ring_step * 0.866).max(2.0);
    let mut band_d = (ring_step * 1.15).max(2.5);
    let mut band_idx = 0usize;
    while band_d <= max_d + 0.5 {
        let lo_d = ((band_d - 0.65).max(0.5) * 10.0) as u16;
        let hi_d = ((band_d + 0.65) * 10.0) as u16;
        let stagger = if band_idx % 2 == 1 { 0.5 * ring_step } else { 0.0 };
        let stride = (ring_step * 0.45).floor().max(1.0) as usize;
        for ly in (0..bh).step_by(stride) {
            for lx in (0..bw).step_by(stride) {
                let d_val = dist_map[ly * bw + lx];
                if d_val != u16::MAX && d_val >= lo_d && d_val <= hi_d {
                    let cand = DVec2::new(
                        (left + lx) as f64 + 0.5 + (stagger % 1.0),
                        (top + ly) as f64 + 0.5,
                    );
                    push_if_far(
                        &mut mesh.vertices,
                        &mut vert_normals,
                        cand,
                        guide_min_sq,
                        None,
                    );
                }
            }
        }
        band_d += band_gap;
        band_idx += 1;
    }

    // Guarantee broad interior coverage on large convex regions without dropping into checkerboard patterns.
    let fill_stride = (ring_step * 0.85).round().max(2.0) as usize;
    let mut row_i = 0usize;
    let mut y = top + fill_stride / 2;
    while y <= bottom {
        let x_offset = if row_i % 2 == 1 { fill_stride / 2 } else { 0 };
        let mut x = left + x_offset;
        while x <= right {
            if mask[y * w + x] > 8 {
                let cand = DVec2::new(x as f64 + 0.5, y as f64 + 0.5);
                push_if_far(
                    &mut mesh.vertices,
                    &mut vert_normals,
                    cand,
                    guide_min_sq,
                    None,
                );
            }
            x += fill_stride;
        }
        y += fill_stride;
        row_i += 1;
    }

    // 5. Delaunay triangulation + strict hole-safe sliver filtering + orphan vertex compaction.
    if mesh.vertices.len() >= 3 {
        let (mut tris, mut keep) = triangulate_filtered(&mesh.vertices, &valid, span, ring_step);
        // B4 refinement: where the smoothed / resampled contour cut a corner or a thin strand off
        // the coverage target, add the raw contour points (pixel-crack trace of the support, which
        // encloses every target pixel centre) near the missed pixels and triangulate again.
        let bbox = (left, right, top, bottom);
        for _ in 0..2 {
            let missed = uncovered_target_pixels(&mesh.vertices, &tris, &keep, &target, bbox, w, h);
            if missed.is_empty() {
                break;
            }
            let near = dilate_chebyshev(&missed, w, h, 3);
            let before = mesh.vertices.len();
            for (raw_pts, _) in &raw_loops {
                for &pt in raw_pts {
                    if near[pt.y as usize * w + pt.x as usize] {
                        push_if_far(&mut mesh.vertices, &mut vert_normals, pt, 0.6 * 0.6, None);
                    }
                }
            }
            if mesh.vertices.len() == before {
                break;
            }
            (tris, keep) = triangulate_filtered(&mesh.vertices, &valid, span, ring_step);
        }
        // Last resort: the Delaunay triangles partition their hull, so keep the triangle that owns
        // a still-uncovered target pixel centre (top-left rule, `rig::raster::rasterize_triangle`).
        let missed = uncovered_target_pixels(&mesh.vertices, &tris, &keep, &target, bbox, w, h);
        if missed.iter().any(|&m| m) {
            for (ti, ids) in tris.iter().enumerate() {
                if keep[ti] {
                    continue;
                }
                let tri = [mesh.vertices[ids[0]], mesh.vertices[ids[1]], mesh.vertices[ids[2]]];
                let mut owns = false;
                crate::rig::raster::rasterize_triangle(tri, w, h, |x, y, _| owns |= missed[y * w + x]);
                if owns {
                    keep[ti] = true;
                }
            }
        }
        for (ids, &k) in tris.iter().zip(&keep) {
            if k {
                mesh.triangles.extend_from_slice(ids);
            }
        }
    }

    // Compact away orphan vertices unused by surviving ArtMesh triangles.
    let mut remap = vec![usize::MAX; mesh.vertices.len()];
    let mut compact_verts = Vec::with_capacity(mesh.vertices.len());
    for tri in &mut mesh.triangles {
        let old = *tri;
        if remap[old] == usize::MAX {
            remap[old] = compact_verts.len();
            compact_verts.push(mesh.vertices[old]);
        }
        *tri = remap[old];
    }
    mesh.vertices = compact_verts;
    mesh.uv = mesh
        .vertices
        .iter()
        .map(|p| Point {
            x: p.x / w as f64,
            y: p.y / h as f64,
        })
        .collect();
    mesh
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Pose {
    pub angle_x: f64,
    pub angle_y: f64,
    pub angle_z: f64,
    pub eye_l: f64,
    pub eye_l_smile: f64,
    pub eye_r: f64,
    pub eye_r_smile: f64,
    pub eye_ball_x: f64,
    pub eye_ball_y: f64,
    pub brow_l_y: f64,
    pub brow_l_form: f64,
    pub brow_r_y: f64,
    pub brow_r_form: f64,
    pub mouth_form: f64,
    pub mouth: f64,
    pub body_angle_x: f64,
    pub body_angle_y: f64,
    pub body_angle_z: f64,
    pub breath: f64,
    pub hair: f64,
    pub hair_side: f64,
    pub hair_back: f64,
    /// Eye/mouth shape input carried by envelope expression poses (T10). Not a parameter: never
    /// serialized; `deform` passes it to `deform_with_shapes`.
    #[serde(skip)]
    pub shape: Option<ShapeDeformInput>,
}
impl Default for Pose {
    fn default() -> Self {
        Self {
            angle_x: 0.,
            angle_y: 0.,
            angle_z: 0.,
            eye_l: 1.,
            eye_l_smile: 0.,
            eye_r: 1.,
            eye_r_smile: 0.,
            eye_ball_x: 0.,
            eye_ball_y: 0.,
            brow_l_y: 0.,
            brow_l_form: 0.,
            brow_r_y: 0.,
            brow_r_form: 0.,
            mouth_form: 0.,
            mouth: 0.,
            body_angle_x: 0.,
            body_angle_y: 0.,
            body_angle_z: 0.,
            breath: 0.,
            hair: 0.,
            hair_side: 0.,
            hair_back: 0.,
            shape: None,
        }
    }
}
fn hair_sway_profile(id: &str) -> (f64, f64) {
    let strand_idx = id
        .rsplit('_')
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let strand_boost = 1.0 + (strand_idx.saturating_sub(1).min(4) as f64) * 0.10;
    if id.starts_with("hair_side") {
        (1.22 * strand_boost, 0.15)
    } else if id.starts_with("hair_back") {
        (0.85 * strand_boost, 0.10)
    } else {
        (1.0 * strand_boost, 0.0)
    }
}

fn apply_body_sway(p: DVec2, body_pivot: DVec2, hh: f64, pose: &Pose, scale: f64) -> DVec2 {
    if pose.body_angle_x == 0.
        && pose.body_angle_y == 0.
        && pose.body_angle_z == 0.
        && pose.breath == 0.
    {
        return p;
    }
    let lift_weight = ((body_pivot.y - p.y) / (hh * 1.35)).clamp(0.0, 1.35);
    let shifted = DVec2::new(
        p.x + pose.body_angle_x * hh * 0.0022 * lift_weight * scale,
        p.y + (pose.body_angle_y * hh * 0.0014 - pose.breath * hh * 0.015) * lift_weight * scale,
    );
    let rot = glam::DMat2::from_angle((pose.body_angle_z * 0.45 * scale).to_radians());
    body_pivot + rot * (shifted - body_pivot)
}

/// Resolution- and motion-proportional underlap margins (`fillMask`) derived from `HeadFrame`
/// and `deform_with_shapes` worst-case relative displacements across opposing head/body/hair poses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartOverlapMargins {
    pub neck_collar_x: usize,
    pub neck_collar_y: usize,
    pub neck_chin_y: usize,
    pub face_scalp_margin: usize,
    pub eye_gaze_x: usize,
    pub eye_gaze_y: usize,
    pub hair_side_front_seam: usize,
    pub hair_hat_root: usize,
    pub hair_back_underlap: usize,
    pub clothes_hair_underlap: usize,
}

/// Computes per-seam hidden-area (`fillMask`) underlap budgets proportional to `HeadFrame.hh`,
/// `HeadFrame.eye_dist`, and `deform_with_shapes` / `hair_sway_profile` / `apply_body_sway` maxima.
pub fn part_overlap_margins(lm: &Landmarks, w: usize, h: usize) -> PartOverlapMargins {
    let hf = HeadFrame::from_landmarks(lm, w, h);
    let hh = hf.hh.max(12.0);
    let eye_dist = hf.eye_dist.max(8.0);
    let head_dx_max = 30.0 * hh * 0.0025;
    let head_dy_max = 30.0 * hh * 0.0018;
    let body_dx_max = (10.0 * 0.0022 * 1.15 + (10.0 * 0.45_f64).to_radians()) * hh;
    let body_dy_max = (10.0 * 0.0014 + 0.015) * 1.15 * hh;
    let head_tilt_sin = (30.0_f64).to_radians().sin();
    let neck_rot_sin = (30.0 * 0.78 * 0.5_f64).to_radians().sin();

    let neck_collar_x = ((0.55 * head_dx_max + 0.35 * hh * neck_rot_sin + 0.55 * body_dx_max + 4.0)
        .max(eye_dist * 0.22))
    .ceil() as usize;
    let neck_collar_y = ((head_dy_max + 0.55 * body_dy_max + eye_dist * 0.27 * head_tilt_sin + 4.0)
        .max(eye_dist * 0.18))
    .ceil() as usize;
    let neck_chin_y = ((head_dy_max + eye_dist * 0.28 * head_tilt_sin + hh * 0.04 + 4.0)
        .max(eye_dist * 0.18))
    .ceil() as usize;
    let face_scalp_margin = ((0.08 * hh + 4.0).max(eye_dist * 0.14)).ceil() as usize;
    let eye_gaze_x = ((eye_dist * 0.065) + 3.0).ceil() as usize;
    let eye_gaze_y = ((eye_dist * 0.055) + 3.0).ceil() as usize;
    let hair_side_front_seam = ((0.082 * hh + 4.0).max(eye_dist * 0.15)).ceil() as usize;
    let hair_hat_root = ((0.068 * hh + 4.0).max(eye_dist * 0.13)).ceil() as usize;
    let hair_back_underlap = ((head_dx_max + hh * 0.04 * 0.85 * 1.5 * 1.8 + hh * 0.06 + 4.0)
        .max(eye_dist * 0.30))
    .ceil() as usize;
    let clothes_hair_underlap = ((1.5 * hh * 0.04 * 1.22 * 1.8 + 4.0).max(eye_dist * 0.22))
        .ceil() as usize;

    PartOverlapMargins {
        neck_collar_x,
        neck_collar_y,
        neck_chin_y,
        face_scalp_margin,
        eye_gaze_x,
        eye_gaze_y,
        hair_side_front_seam,
        hair_hat_root,
        hair_back_underlap,
        clothes_hair_underlap,
    }
}

/// Part role mirroring TS `PartRole` (`src/types.ts`). Serialized names match the TS strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartRole {
    #[serde(rename = "hair_front")]
    HairFront,
    #[serde(rename = "hair_side_L")]
    HairSideL,
    #[serde(rename = "hair_side_R")]
    HairSideR,
    #[serde(rename = "hair_back")]
    HairBack,
    #[serde(rename = "hat")]
    Hat,
    #[serde(rename = "accessory")]
    Accessory,
    #[serde(rename = "hands")]
    Hands,
    #[serde(rename = "body")]
    Body,
    #[serde(rename = "clothes")]
    Clothes,
    #[serde(rename = "neck")]
    Neck,
    #[serde(rename = "face")]
    Face,
    #[serde(rename = "eye")]
    Eye,
    #[serde(rename = "brow")]
    Brow,
    #[serde(rename = "mouth")]
    Mouth,
    #[serde(rename = "other")]
    Other,
}

/// Rust counterpart of TS `inferPartRole`: an explicit role always wins; otherwise the role is
/// resolved from the part ID. This is the single Rust ID→role fallback; keep it in lockstep with
/// `src/types.ts` (guarded by `tests/fixtures/fill-radius-parity.json`).
pub fn infer_part_role(id: &str, explicit_role: Option<PartRole>) -> PartRole {
    if let Some(role) = explicit_role {
        return role;
    }
    if id.starts_with("hair_side_L") {
        PartRole::HairSideL
    } else if id.starts_with("hair_side_R") {
        PartRole::HairSideR
    } else if id.starts_with("hair_side") {
        PartRole::HairSideL
    } else if id.starts_with("hair_back") {
        PartRole::HairBack
    } else if id.starts_with("hair") {
        PartRole::HairFront
    } else if id == "hat" || id.starts_with("hat_") {
        PartRole::Hat
    } else if id.starts_with("acc_") || id.starts_with("accessory") {
        PartRole::Accessory
    } else if id == "clothes_hands" || id == "hands" || id.starts_with("hands_") {
        PartRole::Hands
    } else if id == "body" {
        PartRole::Body
    } else if id == "clothes" || id.starts_with("clothes_") {
        PartRole::Clothes
    } else if id == "neck" {
        PartRole::Neck
    } else if id == "face" || id == "face_detail" {
        PartRole::Face
    } else if id.starts_with("eye") {
        PartRole::Eye
    } else if id.starts_with("brow") {
        PartRole::Brow
    } else if id.starts_with("mouth") {
        PartRole::Mouth
    } else {
        PartRole::Other
    }
}

/// Within the `eye` role, the white/iris sublayers are the ones occluded by lids/brows and need
/// gaze-travel underlap; lids/lashes are occluders and use the generic radius.
pub(crate) fn is_eye_occluded_sublayer(id: &str) -> bool {
    id.ends_with("_white") || id.ends_with("_iris")
}

/// Returns the motion-proportional manual/fallback occlusion fill radius for a part, dispatched by
/// its role (explicit or inferred via `infer_part_role`), identically to TS `recommendedFillRadius`.
pub fn recommended_fill_radius(
    part_id: &str,
    explicit_role: Option<PartRole>,
    lm: &Landmarks,
    w: usize,
    h: usize,
) -> usize {
    let m = part_overlap_margins(lm, w, h);
    let hf = HeadFrame::from_landmarks(lm, w, h);
    let generic = ((hf.eye_dist.max(8.0) * 0.08).ceil() as usize).max(6);
    match infer_part_role(part_id, explicit_role) {
        PartRole::HairBack => m.hair_back_underlap,
        PartRole::HairFront | PartRole::HairSideL | PartRole::HairSideR => {
            m.hair_side_front_seam.max(m.hair_hat_root)
        }
        PartRole::Neck => m.neck_collar_x.max(m.neck_collar_y).max(m.neck_chin_y),
        PartRole::Face => m.face_scalp_margin,
        PartRole::Body | PartRole::Clothes | PartRole::Hands => m.clothes_hair_underlap,
        PartRole::Eye if is_eye_occluded_sublayer(part_id) => m.eye_gaze_x.max(m.eye_gaze_y),
        _ => generic,
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegionDeformParams {
    pub width: f64,
    pub roundness_delta: f64,
    pub corner: f64,
    pub upper_curve_delta: f64,
    pub lower_curve_delta: f64,
    pub offset_x: f64,
    pub offset_y: f64,
}

impl RegionDeformParams {
    pub fn neutral() -> Self {
        Self {
            width: 1.0,
            roundness_delta: 0.0,
            corner: 0.0,
            upper_curve_delta: 0.0,
            lower_curve_delta: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShapeDeformInput {
    pub eye_l: Option<RegionDeformParams>,
    pub eye_r: Option<RegionDeformParams>,
    pub mouth: Option<RegionDeformParams>,
}

/// `deform` with the pose's own eye/mouth shape input (`Pose::shape`, set by envelope poses).
pub fn deform(point: Point, id: &str, lm: &Landmarks, pose: &Pose) -> Point {
    deform_with_shapes(point, id, lm, pose, pose.shape.as_ref())
}

// -------------------------------------------------------------------------------------------------
// T10 — sealed eye rig
// -------------------------------------------------------------------------------------------------
//
// Every non-substitute eye layer of one side (white, iris, upper lid, lower lid and any extra
// `eyeL_*` layer) is moved by ONE vertical field per column, like a single warp deformer:
//
//   v ≤ lash(u)            → upper-lid map  `m_lid`  (lid closes onto the closing line `t`)
//   v ≥ lower(u)           → lower-lid map  `m_lo`   (lower lid rises onto the same line)
//   lash(u) < v < lower(u) → linear between `m_lid(lash)` and `m_lo(lower)` (the aperture)
//
// `lash`/`lower` are the top/bottom of the rest-pose visible aperture (visible white ∪ iris pixels)
// per head-local column, derived from the masks (`derive_eye_frames`). Because all layers share the
// field, the white's top stays glued to the lid's lower edge (no skin gap) and anything hidden
// under a lid at rest stays under it; at `eye = 0` the aperture has zero height, so the white and
// the iris (clipped to the white) cannot show. Ranges, gaze offsets and expression curvature are
// unchanged; the lash travels the full distance to the closing line (more than the pre-T10 0.78).
// Mirrored 1:1 in `src/engine.ts` (`deformUncoupled`, `deriveEyeFrames`).

/// Column bins used to sample the rest aperture.
pub const EYE_FRAME_BINS: usize = 16;
/// Aperture pixels farther than this (eye distances) from the eye landmark are ignored.
pub const EYE_FRAME_REACH_ED: f64 = 0.9;
/// The upper-lid map is fully applied within this many eye distances above the lash line…
pub const EYE_LID_REACH_ED: f64 = 0.20;
/// …and fades to identity over this further span (brow gap stays put).
pub const EYE_LID_RAMP_ED: f64 = 0.12;
/// Lash band (eye distances above the lash line) that keeps its thickness when the lid closes.
pub const EYE_LASH_KEEP_ED: f64 = 0.03;
/// Mesh quality: an extreme-pose triangle counts as flipped only when its signed area falls below
/// `-ratio × rest area` (collapsed slivers on the closing eye line are not inversions).
pub const EXTREME_FLIP_COLLAPSE_RATIO: f64 = 0.05;
/// Eye motion (max of closure, |gaze|, eye smile, expression shape deltas) at which white/iris texels
/// outside the aperture are fully folded onto the lid edge; the fold ramps in linearly from zero, so
/// the field is the identity at the rest pose.
pub const EYE_FOLD_ONSET_CLOSE: f64 = 0.05;
/// Upward tuck (eye distances, × close²) of the collapsing white/iris into the closed upper-lid
/// band (see `deform_eye`).
pub const EYE_INTERIOR_TUCK_ED: f64 = 0.01;
/// Thickness ratio the lower-lid band keeps when fully closed.
pub const EYE_LOWER_CLOSED_SCALE: f64 = 0.4;

/// One aperture column sample in head-local coordinates (`HeadFrame::to_local`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EyeFrameSample {
    pub u: f64,
    /// Top of the visible aperture (upper lid's lower edge) at `u`.
    pub lash: f64,
    /// Bottom of the visible aperture (lower lid's upper edge) at `u`.
    pub lower: f64,
}

/// Aperture samples per eye (character L / R), sorted by `u`; `None` = not derivable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EyeFrames {
    #[serde(rename = "L")]
    pub l: Option<Vec<EyeFrameSample>>,
    #[serde(rename = "R")]
    pub r: Option<Vec<EyeFrameSample>>,
}

/// Character side of an eye layer by ID prefix (`eyeL*` / `eyeR*`).
pub fn eye_layer_side(id: &str) -> Option<char> {
    if id.starts_with("eyeL") {
        Some('L')
    } else if id.starts_with("eyeR") {
        Some('R')
    } else {
        None
    }
}

/// Eye interior layer (white or iris, including numbered extras such as `eyeL_iris_2`).
pub fn is_eye_interior_layer(id: &str) -> bool {
    let Some(rest) = id.strip_prefix("eyeL_").or_else(|| id.strip_prefix("eyeR_")) else {
        return false;
    };
    let base = match rest.split_once('_') {
        Some((b, n)) if !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()) => b,
        Some(_) => return false,
        None => rest,
    };
    base == "white" || base == "iris"
}

/// Iris layer of an eye (`eyeL_iris`, `eyeR_iris_2`, …).
pub fn is_eye_iris_layer(id: &str) -> bool {
    is_eye_interior_layer(id) && id.contains("_iris")
}

/// Rest aperture lines per eye from the visible white/iris masks (alpha ≥ 128).
pub fn derive_eye_frames(
    lm: &Landmarks,
    ids: &[String],
    visible: &[Vec<u8>],
    w: usize,
    h: usize,
) -> EyeFrames {
    let hf = HeadFrame::from_landmarks(lm, 0, 0);
    let dist = hf.eye_dist;
    let side_frame = |side: char, eye: Point| -> Option<Vec<EyeFrameSample>> {
        let (eu, ev) = hf.to_local(eye.v());
        let reach = dist * EYE_FRAME_REACH_ED;
        // Visible white ∪ iris near the eye; only the largest 8-connected component defines the
        // aperture (isolated specks segmented inside the lash wing would make the corner samples
        // falsely tall and drag the lid wing with the aperture).
        let mut sel = vec![false; w * h];
        for (p, id) in ids.iter().enumerate() {
            if eye_layer_side(id) != Some(side) || !is_eye_interior_layer(id) {
                continue;
            }
            let Some(m) = visible.get(p) else { continue };
            for y in 0..h {
                for x in 0..w {
                    if m[y * w + x] < 128 {
                        continue;
                    }
                    let (u, v) = hf.to_local(DVec2::new(x as f64 + 0.5, y as f64 + 0.5));
                    if (u - eu).abs() <= reach && (v - ev).abs() <= reach {
                        sel[y * w + x] = true;
                    }
                }
            }
        }
        let mut best: Vec<usize> = Vec::new();
        let mut seen = vec![false; w * h];
        for start in 0..w * h {
            if !sel[start] || seen[start] {
                continue;
            }
            let mut comp = vec![start];
            seen[start] = true;
            let mut k = 0;
            while k < comp.len() {
                let i = comp[k];
                k += 1;
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                for dy in -1..=1i64 {
                    for dx in -1..=1i64 {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        if sel[j] && !seen[j] {
                            seen[j] = true;
                            comp.push(j);
                        }
                    }
                }
            }
            if comp.len() > best.len() {
                best = comp;
            }
        }
        let pts: Vec<(f64, f64)> = best
            .iter()
            .map(|&i| hf.to_local(DVec2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5)))
            .collect();
        if pts.len() < 8 {
            return None;
        }
        let umin = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let umax = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let span = umax - umin;
        if span < 1.0 {
            return None;
        }
        let k = EYE_FRAME_BINS;
        let mut lash = vec![f64::INFINITY; k];
        let mut lower = vec![f64::NEG_INFINITY; k];
        for &(u, v) in &pts {
            let b = (((u - umin) / span * k as f64).floor() as usize).min(k - 1);
            lash[b] = lash[b].min(v - 0.5);
            lower[b] = lower[b].max(v + 0.5);
        }
        // Conservative lines: each sample takes the extreme over its neighbouring bins, so the
        // piecewise-linear line never cuts a visible aperture pixel (rest pose stays exact).
        let samples: Vec<EyeFrameSample> = (0..k)
            .filter(|&b| lash[b].is_finite())
            .map(|b| {
                let (a, c) = (b.saturating_sub(1), (b + 1).min(k - 1));
                let top = lash[a].min(lash[b]).min(lash[c]);
                let bottom = lower[a].max(lower[b]).max(lower[c]);
                EyeFrameSample {
                    u: umin + (b as f64 + 0.5) * span / k as f64,
                    lash: top,
                    lower: bottom.max(top + 1.0),
                }
            })
            .collect();
        Some(samples)
    };
    EyeFrames {
        l: side_frame('L', lm.eye_l),
        r: side_frame('R', lm.eye_r),
    }
}

/// Aperture used when no frame could be derived (landmarks only).
pub fn default_eye_frame(eye_u: f64, eye_v: f64, dist: f64) -> Vec<EyeFrameSample> {
    [-1.0, -0.5, 0.0, 0.5, 1.0]
        .iter()
        .map(|&t: &f64| {
            let k = 1.0 - 0.6 * t * t;
            EyeFrameSample {
                u: eye_u + t * 0.30 * dist,
                lash: eye_v - 0.20 * dist * k,
                lower: eye_v + 0.13 * dist * k,
            }
        })
        .collect()
}

/// `(lash, lower)` at head-local column `u` (piecewise linear, constant beyond the ends).
pub fn eye_lines_at(samples: &[EyeFrameSample], u: f64) -> (f64, f64) {
    let (first, last) = (samples[0], samples[samples.len() - 1]);
    if u <= first.u {
        return (first.lash, first.lower);
    }
    if u >= last.u {
        return (last.lash, last.lower);
    }
    for pair in samples.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if u <= b.u {
            let t = if b.u > a.u { (u - a.u) / (b.u - a.u) } else { 0.0 };
            return (a.lash + (b.lash - a.lash) * t, a.lower + (b.lower - a.lower) * t);
        }
    }
    (last.lash, last.lower)
}

/// T10: whether rest point `point` of eye layer `id` is a white/iris texel outside the rest
/// aperture (`lash < v < lower`), i.e. one the sealed rig folds onto a lid edge (never shown by
/// its own motion). Uses the derived frames when present, else `default_eye_frame`.
pub fn eye_interior_folded(point: Point, id: &str, lm: &Landmarks) -> bool {
    if !is_eye_interior_layer(id) || legacy_eye_rig_enabled() {
        return false;
    }
    let hf = HeadFrame::from_landmarks(lm, 0, 0);
    let is_left = id.starts_with("eyeL");
    let eye = if is_left { lm.eye_l } else { lm.eye_r };
    let (eu, ev) = hf.to_local(eye.v());
    let fallback;
    let samples: &[EyeFrameSample] = match lm
        .eye_frames
        .as_ref()
        .and_then(|f| if is_left { f.l.as_ref() } else { f.r.as_ref() })
        .filter(|s| !s.is_empty())
    {
        Some(s) => s,
        None => {
            fallback = default_eye_frame(eu, ev, hf.eye_dist);
            &fallback
        }
    };
    let (u, v) = hf.to_local(point.v());
    let (lash, lower) = eye_lines_at(samples, u);
    v <= lash || v >= lower
}

thread_local! {
    static LEGACY_EYE_RIG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether `deform` uses the pre-T10 per-layer eye formulas on this thread (measurement only).
pub fn legacy_eye_rig_enabled() -> bool {
    LEGACY_EYE_RIG.with(|c| c.get())
}

/// Runs `f` with the pre-T10 eye formulas and clipping rule (before/after measurements, CLI
/// `--legacy-eye`). Thread-local like `with_seam_coupling`.
pub fn with_legacy_eye_rig<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let prev = LEGACY_EYE_RIG.with(|c| c.replace(enabled));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            LEGACY_EYE_RIG.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(prev);
    f()
}

/// Static eye-shape vertical warp (`ShapeDeformInput`) at posed column `pu`. One function for
/// every eye layer (`legacy_masks` reproduces the pre-T10 per-layer masks for substitute layers).
#[allow(clippy::too_many_arguments)]
fn eye_shape_v(
    pv: f64,
    pu: f64,
    eye_u: f64,
    eye_v: f64,
    dist: f64,
    is_left: bool,
    shape: &RegionDeformParams,
    is_iris: bool,
    legacy: Option<(bool, bool, bool)>,
) -> f64 {
    let mut pv = pv;
    let nx = (pu - eye_u) / (dist * 0.24).max(1.0);
    let center_arch = (1.0 - nx * nx).max(0.0);
    let eye_sign = if is_left { 1.0 } else { -1.0 };
    let outer_signed = (nx * eye_sign).clamp(-0.4, 1.2);
    pv -= shape.corner * dist * 0.045 * outer_signed;
    // legacy = (is_upper, is_sub_lid, is_lower)
    let (skip_upper, full_upper, full_lower) = legacy.unwrap_or((false, false, false));
    if full_upper {
        pv -= shape.upper_curve_delta * dist * 0.055 * center_arch;
    } else if !skip_upper {
        let upper_mask = ((eye_v + dist * 0.06 - pv) / (dist * 0.14).max(1.0)).clamp(0.0, 1.0);
        pv -= shape.upper_curve_delta * dist * 0.055 * center_arch * upper_mask;
    }
    let lower_mask = if full_lower {
        1.0
    } else {
        ((pv - (eye_v - dist * 0.02)) / (dist * 0.14).max(1.0)).clamp(0.0, 1.0)
    };
    pv += shape.lower_curve_delta * dist * 0.045 * center_arch * lower_mask;
    if !is_iris {
        pv = eye_v + (pv - eye_v) * (1.0 + shape.roundness_delta * 0.32 * center_arch);
    }
    pv
}

/// Eye block of `deform` (head-local `(pu, pv)` in → out).
#[allow(clippy::too_many_arguments)]
fn deform_eye(
    pu: f64,
    pv: f64,
    id: &str,
    lm: &Landmarks,
    hf: &HeadFrame,
    pose: &Pose,
    shape_deform: Option<&ShapeDeformInput>,
) -> (f64, f64) {
    let dist = hf.eye_dist;
    let is_left = id.starts_with("eyeL");
    let (eye, open, smile) = if is_left {
        (lm.eye_l, pose.eye_l, pose.eye_l_smile)
    } else {
        (lm.eye_r, pose.eye_r, pose.eye_r_smile)
    };
    let (eye_u, eye_v) = hf.to_local(eye.v());
    let shape = shape_deform.and_then(|s| if is_left { s.eye_l } else { s.eye_r });
    let is_iris = id.ends_with("_iris") || is_eye_iris_layer(id);
    let is_sub_layer = id.contains("_sub_");
    if legacy_eye_rig_enabled() || is_sub_layer {
        return deform_eye_legacy(pu, pv, id, eye_u, eye_v, dist, is_left, open, smile, pose, shape);
    }
    let (mut pu, mut pv) = (pu, pv);
    let (off_x, off_y, width) = shape.map_or((0.0, 0.0, 1.0), |s| (s.offset_x, s.offset_y, s.width));
    if let Some(s) = shape.as_ref() {
        pu += off_x * dist * 0.14;
        pv += off_y * dist * 0.12;
        let ws = if is_iris { 1.0 + (width - 1.0) * 0.45 } else { width };
        pu = eye_u + (pu - eye_u) * ws;
        pv = eye_shape_v(pv, pu, eye_u, eye_v, dist, is_left, s, is_iris, None);
    }
    if is_iris {
        pu += pose.eye_ball_x * dist * 0.045;
        pv -= pose.eye_ball_y * dist * 0.025;
    }
    let close = (1.0 - open).clamp(0.0, 1.0);
    let widen = (open - 1.0).clamp(0.0, 0.3);
    let curve_delta = shape.map_or(0.0, |s| s.upper_curve_delta);
    let pos_curve = curve_delta.max(0.0);
    let neg_curve = (-curve_delta).max(0.0);
    let nx_c = (pu - eye_u) / (dist * 0.25).max(1.0);
    let arch = (1.0 - nx_c * nx_c).max(0.0);
    // Rest column of the aperture lines under this posed column (inverse of the white's width map).
    let mut u_rest = pu;
    if shape.is_some() && width.abs() > 1e-6 {
        u_rest = eye_u + (pu - eye_u) / width - off_x * dist * 0.14;
    }
    let frame = lm
        .eye_frames
        .as_ref()
        .and_then(|f| if is_left { f.l.as_ref() } else { f.r.as_ref() })
        .filter(|s| !s.is_empty());
    let fallback;
    let samples: &[EyeFrameSample] = match frame {
        Some(s) => s,
        None => {
            fallback = default_eye_frame(eye_u, eye_v, dist);
            &fallback
        }
    };
    let (lash_r, lower_r) = eye_lines_at(samples, u_rest);
    let interior = is_eye_interior_layer(id) || id.ends_with("_white") || is_iris;
    let warp = |v: f64| match shape.as_ref() {
        Some(s) => eye_shape_v(v + off_y * dist * 0.12, pu, eye_u, eye_v, dist, is_left, s, false, None),
        None => v,
    };
    let la = warp(lash_r);
    let ba = warp(lower_r).max(la + 0.5);
    let d_a = warp(eye_v) - eye_v;
    // Closing line: at the eye-centre column the pre-T10 full-closure seam (`eye_v + 0.022·dist`);
    // across the eye it keeps the same fraction `λ` of the aperture height, so it follows the
    // aperture shape and the corners (thin aperture) barely move, as in a real blink. On top: the
    // expression curvature terms and the sad (∪) drop the pre-T10 upper lid received, so no
    // expression's lash travel shrinks. Never above the lash line; it may sit below the lower line
    // (the lower lid then follows it down).
    let (lash_c, lower_c) = eye_lines_at(samples, eye_u);
    let lambda = ((eye_v + dist * 0.022 - lash_c) / (lower_c - lash_c).max(1.0)).clamp(0.0, 1.0);
    let s1 = (la + lambda * (ba - la)
        - (smile * 0.032 + pos_curve * 0.028) * dist * arch
        + neg_curve * (0.045 * arch - 0.018 * (1.0 - arch)) * dist
        + neg_curve * ((0.040 + 0.055) * arch - 0.025 * (1.0 - arch)) * dist)
        .max(la);
    let lift_open = smile * 0.045 * dist * arch;
    let target = s1 - smile * 0.018 * dist * arch;
    let sigma = (0.22 + (0.14 * smile + 0.12 * pos_curve - 0.03 * neg_curve).clamp(-0.03, 0.20) * arch)
        .clamp(0.20, 0.42);
    let reach = dist * EYE_LID_REACH_ED;
    let ramp = (dist * EYE_LID_RAMP_ED).max(1.0);
    let lower_smile = smile * 0.025 * dist * arch;
    // Closed-lid band: the lash stroke nearest the lid edge keeps its thickness (a σ-squashed thin
    // stroke aliases into a dotted line); only the eyelid skin above it is compressed by σ.
    let keep = dist * EYE_LASH_KEEP_ED;
    let closed_offset = |x: f64| if x >= -keep { x } else { -keep + (x + keep) * sigma };
    let m_lid = |v: f64| {
        let aw = ((v - (la - reach - ramp)) / ramp).clamp(0.0, 1.0);
        let mut r = v + aw * (close * (target + closed_offset(v - la) - v) - (1.0 - close) * lift_open);
        if widen > 0.0 {
            r -= aw * (eye_v + d_a - v).max(0.0) * 0.28 * widen;
        }
        r
    };
    let m_lo = |v: f64| {
        let aw = (((ba + reach + ramp) - v) / ramp).clamp(0.0, 1.0);
        v + aw * (close * (target + (v - ba) * EYE_LOWER_CLOSED_SCALE - v) - (1.0 - close) * lower_smile)
    };
    // White/iris texels outside the rest aperture are under a lid (or a static occluder) at rest;
    // they fold onto the lid edge so they can never be pulled into view (zero-height fold).
    // Tuck: as the eye closes, the collapsing white/iris and the lower lid glued to it rise
    // `EYE_INTERIOR_TUCK_ED · close²` into the closed upper-lid band, so near-degenerate mesh chords
    // along the curved closing line rasterize under the upper lid instead of on its edge. Zero at
    // rest; the upper lid itself is not shifted.
    let tuck = EYE_INTERIOR_TUCK_ED * dist * close * close;
    let lower_aw = |v: f64| (((ba + reach + ramp) - v) / ramp).clamp(0.0, 1.0);
    let interp = |v: f64| {
        let (lo, hi) = (m_lid(la), m_lo(ba));
        lo + (v - la) / (ba - la) * (hi - lo)
    };
    let shared = |v: f64| {
        if v <= la {
            m_lid(v)
        } else if v >= ba {
            m_lo(v)
        } else {
            interp(v)
        }
    };
    pv = if interior {
        // White/iris texels outside the rest aperture fold onto the lid edge as soon as the eye
        // moves (closure, gaze, eye smile, expression shape; full at `EYE_FOLD_ONSET_CLOSE`), so they
        // are never pulled into view; at the exact rest pose the field is the identity.
        let folded = if pv <= la {
            m_lid(la)
        } else if pv >= ba {
            m_lo(ba)
        } else {
            interp(pv)
        };
        let unfolded = shared(pv);
        let shape_mag = shape.map_or(0.0, |s| {
            (s.width - 1.0)
                .abs()
                .max(s.roundness_delta.abs())
                .max(s.corner.abs())
                .max(s.upper_curve_delta.abs())
                .max(s.lower_curve_delta.abs())
                .max(s.offset_x.abs())
                .max(s.offset_y.abs())
        });
        // Gaze moves only the iris; the white folds with lid motion alone.
        let gaze = if is_iris { pose.eye_ball_x.abs().max(pose.eye_ball_y.abs()) } else { 0.0 };
        let motion = close
            .max(gaze)
            .max(smile.abs())
            .max(widen)
            .max(shape_mag);
        let k = (motion / EYE_FOLD_ONSET_CLOSE).min(1.0);
        unfolded + (folded - unfolded) * k - tuck
    } else if id.contains("_upper") {
        // Upper-lid texels below the lash line (corner wings) translate with the lash instead of
        // being collapsed with the aperture, so the closed lid line stays unbroken at the corners.
        if pv <= la {
            m_lid(pv)
        } else {
            pv + (m_lid(la) - la)
        }
    } else if id.contains("_lower") {
        // Lower-lid texels above the lower line translate with it (mirror of the upper lid).
        if pv >= ba {
            m_lo(pv) - tuck * lower_aw(pv)
        } else {
            pv + (m_lo(ba) - ba) - tuck
        }
    } else {
        shared(pv)
    };
    (pu, pv)
}

/// Pre-T10 eye formulas (per-layer maps; kept for substitute layers and `with_legacy_eye_rig`).
#[allow(clippy::too_many_arguments)]
fn deform_eye_legacy(
    pu: f64,
    pv: f64,
    id: &str,
    eye_u: f64,
    eye_v: f64,
    dist: f64,
    is_left: bool,
    open: f64,
    smile: f64,
    pose: &Pose,
    shape: Option<RegionDeformParams>,
) -> (f64, f64) {
    let (mut pu, mut pv) = (pu, pv);
    let is_sub_layer = id.contains("_sub_");
    let is_iris = id.ends_with("_iris");
    let is_upper = id.ends_with("_upper");
    let is_lower = id.ends_with("_lower");
    if let Some(shape) = shape.as_ref() {
        pu += shape.offset_x * dist * 0.14;
        pv += shape.offset_y * dist * 0.12;
        let width_scale = if is_iris {
            1.0 + (shape.width - 1.0) * 0.45
        } else {
            shape.width
        };
        pu = eye_u + (pu - eye_u) * width_scale;
        pv = eye_shape_v(
            pv,
            pu,
            eye_u,
            eye_v,
            dist,
            is_left,
            shape,
            is_iris,
            Some((is_upper, id.ends_with("_sub_lid"), is_lower)),
        );
    }
    if is_sub_layer {
        return (pu, pv);
    }
    let close = (1.0 - open).clamp(0.0, 1.0);
    let widen = (open - 1.0).clamp(0.0, 0.3);
    let curve_delta = shape.map_or(0.0, |s| s.upper_curve_delta);
    let nx_c = (pu - eye_u) / (dist * 0.25).max(1.0);
    let arch = (1.0 - nx_c * nx_c).max(0.0);
    let pos_curve = curve_delta.max(0.0);
    let neg_curve = (-curve_delta).max(0.0);
    let seam_v = eye_v + dist * 0.022 * close - (smile * 0.032 + pos_curve * 0.028) * dist * close * arch
        + neg_curve * (0.045 * arch - 0.018 * (1.0 - arch)) * dist * close;
    if is_iris {
        pu += pose.eye_ball_x * dist * 0.045;
        pv -= pose.eye_ball_y * dist * 0.025;
        pv = seam_v + (pv - eye_v) * open.clamp(0.03, 1.3);
    } else if is_upper {
        let lid_anchor_v = eye_v - dist * 0.45;
        let anchor_w = ((pv - lid_anchor_v) / (dist * 0.12).max(1.0)).clamp(0.0, 1.0);
        let orig_pv = pv;
        if close > 0.0 && orig_pv < seam_v {
            let close_scale = (0.22
                + (0.14 * smile + 0.12 * pos_curve - 0.03 * neg_curve).clamp(-0.03, 0.20) * arch)
                .clamp(0.20, 0.42);
            let closed_v = seam_v + (orig_pv - seam_v) * close_scale;
            pv += (closed_v - orig_pv) * close * anchor_w;
        } else if widen > 0.0 && orig_pv < seam_v {
            pv -= (seam_v - orig_pv) * 0.28 * widen * anchor_w;
        }
        if smile != 0.0 || curve_delta != 0.0 {
            let vert_w =
                anchor_w * ((eye_v + dist * 0.16 - orig_pv) / (dist * 0.12).max(1.0)).clamp(0.0, 1.0);
            let lift_k = (1.0 - close) * 0.045 + close * 0.018;
            let smile_lift = smile * lift_k * dist * arch;
            let curve_lift = pos_curve * lift_k * dist * arch;
            let neg_drop =
                neg_curve * ((0.040 + 0.055 * close) * arch - 0.025 * close * (1.0 - arch)) * dist;
            pv -= (smile_lift + curve_lift - neg_drop) * vert_w;
        }
    } else if is_lower {
        let open_clamp = open.clamp(0.18, 1.3);
        pv = eye_v + (pv - eye_v) * open_clamp - smile * dist * 0.025 * arch;
    } else {
        let open_clamp = open.clamp(0.08, 1.3);
        pv = eye_v + (pv - eye_v) * open_clamp;
    }
    (pu, pv)
}

// -------------------------------------------------------------------------------------------------
// T13b — coupled seam deformation
// -------------------------------------------------------------------------------------------------
//
// Each layer keeps its own (uncoupled) displacement field `own(x)`. Near a seam the layer is blended
// toward its partner's field: `coupled(x) = own(x) + c(x) · (partner(x) − own(x))`, with `c = 1` on
// the seam line and smoothly falling to `0` at the falloff distance. Beyond the falloff the field is
// *exactly* the uncoupled one, so every point far from a seam (crown, face centre, hair tips, torso
// centre, irises) keeps its full motion. Parameter ranges and coefficients are untouched; only the
// *relative* displacement across the seam shrinks. Mirrored 1:1 in `src/engine.ts` (`deform`).
//
// - Neck top ↔ head: the neck follows the head (`face` field) fully above the chin line and fades
//   back to its own damped follow over `NECK_HEAD_COUPLING_FALLOFF_HH` head heights below it.
// - Neck base ↔ body: the neck blends toward the body field over its own head-follow span
//   (`NECK_SPAN_HH`, the same span its uncoupled weight uses), reaching the body field at the base.
// - Collar ↔ neck base: body/clothes inside the neck column (`COLLAR_COUPLING_*_ED` eye distances
//   around the head axis) blend toward the *neck-base* field (raw neck blended toward the body by the
//   neck-base weight) with weight `column · neck_body`. The collar never inherits the neck-top ↔
//   head term, so clothes just under the chin keep the body field and do not turn with the head;
//   the collar meets the neck only where the neck itself is attached to the body. Outside the
//   column (shoulders) and below the neck span (torso) clothes are unchanged.
// - Hair root ↔ scalp: hair layers blend toward the head field inside `HAIR_ROOT_COUPLING_BAND_HH`
//   head heights below their sway root (C¹ sway onset instead of a kink); beyond it sway is unchanged.

/// Neck head-follow span below the chin (head heights). Shared by the uncoupled neck weight and the
/// neck-base ↔ body coupling, so the two cannot drift apart.
pub const NECK_SPAN_HH: f64 = 0.42;
/// Neck-top ↔ head coupling falloff below the chin line (head heights).
pub const NECK_HEAD_COUPLING_FALLOFF_HH: f64 = 0.06;
/// Hair-root ↔ scalp coupling band below each strand's sway root (head heights).
pub const HAIR_ROOT_COUPLING_BAND_HH: f64 = 0.25;
/// Collar column: full coupling within this |u| (eye distances) of the head axis…
pub const COLLAR_COUPLING_INNER_ED: f64 = 0.35;
/// …fading to none at this |u|.
pub const COLLAR_COUPLING_OUTER_ED: f64 = 0.60;
/// B2 torso frame column width (eye distances).
pub const TORSO_FRAME_BIN_ED: f64 = 0.125;
/// B2 maximum slope |d(top)/dx| of the smoothed torso top line (prevents sharp shoulder cliffs).
pub const TORSO_FRAME_MAX_SLOPE: f64 = 0.50;
/// B2 minimum hair → body vertical transition band (head heights). Wide enough at `|dx| = 0` that
/// `angleY` + `breath` never fold; grows linearly with `|dx|` via `HAIR_BODY_BAND_X_RATIO`.
pub const HAIR_BODY_MIN_BAND_HH: f64 = 0.22;
/// B2 lateral scaling of the hair → body transition band (`band >= ratio · |x - cx|`). Guarantees
/// `(1.5 / band) · |x - cx| · sin(30°) ≤ 0.65 < 1` at any horizontal distance from the head axis.
pub const HAIR_BODY_BAND_X_RATIO: f64 = 1.15;
/// B2 outside the face (`|pu| ≥ outer_start_inner·ed` → `outer_start_outer·ed`), the floor for the
/// transition start rises smoothly from `mouth_v` toward `eye_mid_v` for wide shoulder-draping hair.
/// Inside the face (`|pu| ≤ inner·ed`), `y0 ≥ mouth_v` keeps bangs above the mouth 100% on the head.
pub const HAIR_BODY_OUTER_START_INNER_ED: f64 = 0.70;
pub const HAIR_BODY_OUTER_START_OUTER_ED: f64 = 1.10;
/// B2 `hair_back` directly behind the neck (`|dx| ≤ inner·ed`) blends toward `neck`
/// (`HAIR_BACK_NECK_MAX_SHARE`) so it stays behind `clothes > neck` gaps while still reducing
/// `clothes > hair_back` separation; fades smoothly to `clothes` by `outer·ed`.
pub const HAIR_BACK_NECK_INNER_ED: f64 = 0.45;
pub const HAIR_BACK_NECK_OUTER_ED: f64 = 0.65;
pub const HAIR_BACK_NECK_MAX_SHARE: f64 = 0.75;

thread_local! {
    static SEAM_COUPLING: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static HAIR_BODY_COUPLING: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Whether `deform` applies the B2 hair ↔ body coupling on this thread (default `true`; also
/// requires seam coupling and a torso frame).
pub fn hair_body_coupling_enabled() -> bool {
    HAIR_BODY_COUPLING.with(|c| c.get())
}

/// Runs `f` with the B2 hair ↔ body coupling enabled/disabled on the current thread only (the
/// T13b couplings are unaffected). Used for before/after measurements (`flat2rig-cli --no-hair-body`).
pub fn with_hair_body_coupling<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let prev = HAIR_BODY_COUPLING.with(|c| c.replace(enabled));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            HAIR_BODY_COUPLING.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(prev);
    f()
}

/// One torso attachment column sample in world canvas coordinates relative to `HeadFrame::origin.x`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TorsoFrameSample {
    /// Horizontal offset from `hf.origin.x` (`x - cx`, canvas pixels).
    pub u: f64,
    /// Smoothed top canvas Y of the torso (`body`/`clothes`/`hands`) layers in this column.
    pub top: f64,
}

/// B2: smoothed top line of the torso layers (roles `body` / `clothes` / `hands`, effective alpha
/// ≥ 128, canvas `y ≥ mouth.y`) per `TORSO_FRAME_BIN_ED`-wide column relative to `hf.origin.x`.
/// World-space `(x, y)` keeps tilted heads from skewing the shoulder line; Lipschitz slope clamping
/// and two binomial passes keep the transition C¹-smooth without shoulder cliffs.
pub fn derive_torso_frame(
    lm: &Landmarks,
    ids: &[String],
    effective: &[Vec<u8>],
    w: usize,
    h: usize,
) -> Option<Vec<TorsoFrameSample>> {
    let hf = HeadFrame::from_landmarks(lm, 0, 0);
    let bin = hf.eye_dist * TORSO_FRAME_BIN_ED;
    let cx = hf.origin.x;
    let min_y = lm.mouth.y;
    let mut tops: BTreeMap<i64, f64> = BTreeMap::new();
    for (p, id) in ids.iter().enumerate() {
        if !matches!(
            infer_part_role(id, None),
            PartRole::Body | PartRole::Clothes | PartRole::Hands
        ) {
            continue;
        }
        let Some(m) = effective.get(p) else { continue };
        for y in 0..h {
            let cy = y as f64 + 0.5;
            if cy < min_y {
                continue;
            }
            let row = y * w;
            for x in 0..w {
                if m[row + x] < 128 {
                    continue;
                }
                let dx = x as f64 + 0.5 - cx;
                let k = (dx / bin).floor() as i64;
                let top = y as f64;
                tops.entry(k)
                    .and_modify(|t| *t = t.min(top))
                    .or_insert(top);
            }
        }
    }
    let (&k0, _) = tops.iter().next()?;
    let (&k1, _) = tops.iter().next_back()?;
    let mut raw: Vec<f64> = (k0..=k1)
        .map(|k| {
            if let Some(&t) = tops.get(&k) {
                return t;
            }
            let (&ka, &ta) = tops.range(..k).next_back().expect("k0 present");
            let (&kb, &tb) = tops.range(k..).next().expect("k1 present");
            ta + (tb - ta) * (k - ka) as f64 / (kb - ka) as f64
        })
        .collect();
    let n = raw.len();
    // Conservative Lipschitz slope cap: no downward cliff steeper than `TORSO_FRAME_MAX_SLOPE`.
    let max_step = TORSO_FRAME_MAX_SLOPE * bin;
    for i in 1..n {
        raw[i] = raw[i].min(raw[i - 1] + max_step);
    }
    for i in (0..n.saturating_sub(1)).rev() {
        raw[i] = raw[i].min(raw[i + 1] + max_step);
    }
    // Two binomial smoothing passes `[0.25, 0.5, 0.25]` to round shoulder corners.
    for _ in 0..2 {
        let prev = raw.clone();
        for i in 0..n {
            let a = prev[i.saturating_sub(1)];
            let c = prev[(i + 1).min(n - 1)];
            raw[i] = 0.25 * a + 0.5 * prev[i] + 0.25 * c;
        }
    }
    Some(
        (0..n)
            .map(|i| TorsoFrameSample {
                u: ((k0 + i as i64) as f64 + 0.5) * bin,
                top: raw[i],
            })
            .collect(),
    )
}

/// Torso top at head-local column `u` (piecewise linear, constant beyond the ends).
pub fn torso_top_at(samples: &[TorsoFrameSample], u: f64) -> f64 {
    let (first, last) = (samples[0], samples[samples.len() - 1]);
    if u <= first.u {
        return first.top;
    }
    if u >= last.u {
        return last.top;
    }
    for pair in samples.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if u <= b.u {
            let t = if b.u > a.u { (u - a.u) / (b.u - a.u) } else { 0.0 };
            return a.top + (b.top - a.top) * t;
        }
    }
    last.top
}

/// Whether `deform` applies the T13b seam coupling on this thread (default `true`).
pub fn seam_coupling_enabled() -> bool {
    SEAM_COUPLING.with(|c| c.get())
}

/// Runs `f` with seam coupling enabled/disabled on the current thread only (used to measure the
/// uncoupled baseline end to end, e.g. `flat2rig-cli --exposure-baseline`). Thread-local so parallel
/// tests never observe each other's setting.
pub fn with_seam_coupling<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let prev = SEAM_COUPLING.with(|c| c.replace(enabled));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SEAM_COUPLING.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(prev);
    f()
}

#[inline]
fn smoothstep01(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Seam-coupling weights at a rest-space point for a layer (`0` = uncoupled).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeamCouplingWeights {
    /// Neck → head (`face`) field blend near the chin line.
    pub neck_head: f64,
    /// Neck → body field blend toward the neck base.
    pub neck_body: f64,
    /// Body/clothes → neck-base field blend (`column(|u|) · neck_body`; never the neck-top term).
    pub collar: f64,
    /// Hair → head field blend inside the root band.
    pub hair_root: f64,
    /// B2 hair → torso (coupled `clothes`) field blend below the chin line: `0` at the chin line,
    /// `1` at and below the torso top of the column (`torso_top_at`). Only the head-follow part of
    /// the hair field is exchanged; the hair's own sway displacement is kept in full.
    pub hair_body: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CouplingClass {
    Neck,
    Collar,
    Hair,
    None,
}

fn coupling_class(id: &str) -> CouplingClass {
    let role = infer_part_role(id, None);
    if role == PartRole::Neck {
        CouplingClass::Neck
    } else if matches!(
        role,
        PartRole::HairFront | PartRole::HairSideL | PartRole::HairSideR | PartRole::HairBack
    ) {
        CouplingClass::Hair
    } else if matches!(role, PartRole::Body | PartRole::Clothes) {
        CouplingClass::Collar
    } else {
        CouplingClass::None
    }
}

fn neck_coupling(hf: &HeadFrame, pv: f64) -> (f64, f64) {
    let hh = hf.hh;
    (
        1.0 - smoothstep01(hf.chin_v, hf.chin_v + hh * NECK_HEAD_COUPLING_FALLOFF_HH, pv),
        smoothstep01(hf.chin_v, hf.chin_v + hh * NECK_SPAN_HH, pv),
    )
}

/// Coupling weights the T13b `deform` uses for `id` at rest point `point` (independent of the pose).
pub fn seam_coupling_weights(point: Point, id: &str, lm: &Landmarks) -> SeamCouplingWeights {
    let hf = HeadFrame::from_landmarks(lm, 0, 0);
    let (pu, pv) = hf.to_local(point.v());
    let mut out = SeamCouplingWeights::default();
    match coupling_class(id) {
        CouplingClass::Neck => {
            (out.neck_head, out.neck_body) = neck_coupling(&hf, pv);
        }
        CouplingClass::Collar => {
            // Only the neck-base (body-side) component; the neck-top ↔ head weight stays 0.
            let (_, neck_body) = neck_coupling(&hf, pv);
            let column = 1.0
                - smoothstep01(
                    hf.eye_dist * COLLAR_COUPLING_INNER_ED,
                    hf.eye_dist * COLLAR_COUPLING_OUTER_ED,
                    pu.abs(),
                );
            out.neck_body = neck_body;
            out.collar = column * neck_body;
        }
        CouplingClass::Hair => {
            let (_, root_offset) = hair_sway_profile(id);
            let root_y = lm.crown.y + hf.hh * root_offset;
            let t = (point.y - root_y) / hf.hh;
            out.hair_root = 1.0 - smoothstep01(0.0, HAIR_ROOT_COUPLING_BAND_HH, t);
            out.hair_body = hair_body_weight(&hf, lm, point);
        }
        CouplingClass::None => {}
    }
    out
}

/// B2 hair → torso transition band `(y0, y1)` in canvas Y at `point`. Unified across all hair roles
/// so `hair_front`/`hair_side` and `hair_back` share the same field and never separate at hair-hair
/// seams. Anchors `y1` at `torso_top_at` by pulling `y0 = (top - min_band).max(y_floor)` upward
/// (clamped at `mouth_v` inside the face so bangs above the mouth stay 100% on the head, rising to
/// `eye_mid_v` outside the face), then `y1 = top.max(y0 + min_band)` guarantees `y1 - y0 >= min_band`.
pub(crate) fn hair_body_band(
    hf: &HeadFrame,
    lm: &Landmarks,
    point: Point,
) -> Option<(f64, f64)> {
    let frame = lm.torso_frame.as_deref().filter(|f| !f.is_empty())?;
    let (pu, _) = hf.to_local(point.v());
    let (_, mouth_v) = hf.to_local(lm.mouth.v());
    let (_, eye_l_v) = hf.to_local(lm.eye_l.v());
    let (_, eye_r_v) = hf.to_local(lm.eye_r.v());
    let eye_mid_v = (eye_l_v + eye_r_v) * 0.5;
    let outer_t = smoothstep01(
        hf.eye_dist * HAIR_BODY_OUTER_START_INNER_ED,
        hf.eye_dist * HAIR_BODY_OUTER_START_OUTER_ED,
        pu.abs(),
    );
    let v_floor = mouth_v + (eye_mid_v - mouth_v) * outer_t;
    let y_floor = hf.to_world(pu, v_floor).y;
    let dx = point.x - hf.origin.x;
    let min_band = (hf.hh * HAIR_BODY_MIN_BAND_HH).max(dx.abs() * HAIR_BODY_BAND_X_RATIO);
    let top = torso_top_at(frame, dx);
    let y0 = (top - min_band).max(y_floor);
    let y1 = top.max(y0 + min_band);
    Some((y0, y1))
}

/// B2 hair → torso blend at `point`: smooth C¹ S-curve `smoothstep(y0, y1, point.y)` with band
/// width `≥ max(0.22·hh, 1.15·|x - cx|)` so head rotation never folds or kinks hair.
fn hair_body_weight(hf: &HeadFrame, lm: &Landmarks, point: Point) -> f64 {
    if !hair_body_coupling_enabled() {
        return 0.0;
    }
    let Some((y0, y1)) = hair_body_band(hf, lm, point) else {
        return 0.0;
    };
    smoothstep01(y0, y1, point.y)
}

/// B2 neck-column share for `hair_back`: directly behind the neck (`|dx| ≤ 0.25·ed`), `hair_back`
/// follows `neck` instead of `clothes` so it does not pull away from `clothes > neck` gaps.
fn hair_back_neck_share(hf: &HeadFrame, role: PartRole, point: Point) -> f64 {
    if role != PartRole::HairBack {
        return 0.0;
    }
    let abs_dx = (point.x - hf.origin.x).abs();
    HAIR_BACK_NECK_MAX_SHARE
        * (1.0
            - smoothstep01(
                hf.eye_dist * HAIR_BACK_NECK_INNER_ED,
                hf.eye_dist * HAIR_BACK_NECK_OUTER_ED,
                abs_dx,
            ))
}

/// Coupled neck field at `point` given the raw (uncoupled) neck displacement there.
fn coupled_neck_field(
    point: Point,
    raw_neck: DVec2,
    wts: &SeamCouplingWeights,
    lm: &Landmarks,
    pose: &Pose,
) -> DVec2 {
    let mut n = raw_neck;
    if wts.neck_head > 0.0 {
        let head = deform_uncoupled(point, "face", lm, pose).v();
        n += (head - n) * wts.neck_head;
    }
    if wts.neck_body > 0.0 {
        let body = deform_uncoupled(point, "body", lm, pose).v();
        n += (body - n) * wts.neck_body;
    }
    n
}

/// T13b uncoupled per-layer field (the pre-T13b `deform`). Kept public for before/after
/// measurements and the Rust/TS parity fixture; rendering uses `deform`.
pub fn deform_uncoupled(point: Point, id: &str, lm: &Landmarks, pose: &Pose) -> Point {
    deform_uncoupled_with_shapes(point, id, lm, pose, pose.shape.as_ref())
}

pub fn deform_with_shapes(
    point: Point,
    id: &str,
    lm: &Landmarks,
    pose: &Pose,
    shape_deform: Option<&ShapeDeformInput>,
) -> Point {
    let raw = deform_uncoupled_with_shapes(point, id, lm, pose, shape_deform);
    if !seam_coupling_enabled() {
        return raw;
    }
    let class = coupling_class(id);
    if class == CouplingClass::None {
        return raw;
    }
    let wts = seam_coupling_weights(point, id, lm);
    match class {
        CouplingClass::Neck => coupled_neck_field(point, raw.v(), &wts, lm, pose).into(),
        CouplingClass::Collar => {
            if wts.collar <= 0.0 {
                return raw;
            }
            // Neck-base field: raw neck blended toward the body (no neck-top ↔ head term).
            let raw_neck = deform_uncoupled(point, "neck", lm, pose).v();
            let base = coupled_neck_field(point, raw_neck, &wts, lm, pose);
            (raw.v() + (base - raw.v()) * wts.collar).into()
        }
        CouplingClass::Hair => {
            if wts.hair_root <= 0.0 && wts.hair_body <= 0.0 {
                return raw;
            }
            // `head` = the hair field without its sway (hair follows the head exactly like `face`).
            let head = deform_uncoupled(point, "face", lm, pose).v();
            let mut p = raw.v() + (head - raw.v()) * wts.hair_root;
            if wts.hair_body > 0.0 {
                // B2: exchange the head-follow part for the coupled torso field; `raw − head` (the
                // sway) is untouched, so the sway amplitude is the same at every point.
                let mut torso = deform_with_shapes(point, "clothes", lm, pose, None).v();
                let hf = HeadFrame::from_landmarks(lm, 0, 0);
                let neck_share = hair_back_neck_share(&hf, infer_part_role(id, None), point);
                if neck_share > 0.0 {
                    let neck = deform_with_shapes(point, "neck", lm, pose, None).v();
                    torso += (neck - torso) * neck_share;
                }
                p += (torso - head) * wts.hair_body;
            }
            p.into()
        }
        CouplingClass::None => raw,
    }
}

fn deform_uncoupled_with_shapes(
    point: Point,
    id: &str,
    lm: &Landmarks,
    pose: &Pose,
    shape_deform: Option<&ShapeDeformInput>,
) -> Point {
    let hf = HeadFrame::from_landmarks(lm, 0, 0);
    let hh = hf.hh;
    let dist = hf.eye_dist;
    let cx = hf.origin.x;
    let body_pivot = DVec2::new(cx, lm.chin.y + hh * 1.15);
    if id == "body" || id.starts_with("clothes") || id == "hands" || id.starts_with("hands_") {
        return apply_body_sway(point.v(), body_pivot, hh, pose, 1.0).into();
    }
    let (mut pu, mut pv) = hf.to_local(point.v());
    let (mouth_u, mouth_v) = hf.to_local(lm.mouth.v());
    let (_, eye_l_v) = hf.to_local(lm.eye_l.v());
    let (_, eye_r_v) = hf.to_local(lm.eye_r.v());
    let eye_mid_v = (eye_l_v + eye_r_v) * 0.5;
    let is_sub_layer = id.contains("_sub_");
    if id.starts_with("eye") {
        (pu, pv) = deform_eye(pu, pv, id, lm, &hf, pose, shape_deform);
    }
    if id.starts_with("brow") {
        let (brow_y, brow_form) = if id.starts_with("browL") {
            (pose.brow_l_y, pose.brow_l_form)
        } else {
            (pose.brow_r_y, pose.brow_r_form)
        };
        pv -= brow_y * hh * 0.045;
        let inner_factor = 1.0 - (pu.abs() / (dist * 0.65)).clamp(0.0, 1.0);
        pv -= brow_form * hh * 0.028 * inner_factor;
    }
    if id.starts_with("mouth") {
        if let Some(m_shape) = shape_deform.and_then(|s| s.mouth) {
            pu += m_shape.offset_x * dist * 0.12;
            pv += m_shape.offset_y * hh * 0.04;
            pu = mouth_u + (pu - mouth_u) * m_shape.width;
            let mnx = (pu - mouth_u) / (dist * 0.16).max(1.0);
            let m_center = (1.0 - mnx * mnx).max(0.0);
            let m_corner_w = mnx.abs().clamp(0.0, 1.2);
            pv -= m_shape.corner * hh * 0.028 * m_corner_w * m_corner_w;
            pv = mouth_v + (pv - mouth_v) * (1.0 + m_shape.roundness_delta * 0.3 * m_center);
            let upper_w = if id == "mouth_upper" || id == "mouth_sub_lip" {
                1.0
            } else {
                ((mouth_v + hh * 0.02 - pv) / (hh * 0.04).max(1.0)).clamp(0.0, 1.0)
            };
            let lower_w = if id == "mouth_lower" || id == "mouth_sub_lip" {
                1.0
            } else {
                ((pv - (mouth_v - hh * 0.02)) / (hh * 0.04).max(1.0)).clamp(0.0, 1.0)
            };
            pv -= m_shape.upper_curve_delta * hh * 0.024 * m_center * upper_w;
            pv += m_shape.lower_curve_delta * hh * 0.026 * m_center * lower_w;
        }
        if !is_sub_layer {
            let mdx = pu - mouth_u;
            let corner_w = (mdx.abs() / (dist * 0.18).max(1.)).clamp(0., 1.2);
            pu += mdx * pose.mouth_form * 0.14;
            pv -= pose.mouth_form * hh * 0.016 * corner_w * corner_w;
            if id == "mouth_upper" {
                pv -= pose.mouth * hh * 0.012;
            }
            if id == "mouth_lower" {
                pv += pose.mouth * hh * 0.045;
            }
            if id == "mouth_inner" {
                pv = mouth_v + (pv - mouth_v) * (1. + pose.mouth * 2.);
            }
        }
    }
    let weight = if id == "neck" {
        ((hf.chin_v + hh * NECK_SPAN_HH - pv) / (hh * NECK_SPAN_HH)).clamp(0., 1.)
    } else {
        1.
    };
    pu += pose.angle_x * hh * 0.0025 * weight * (1. - (pv - eye_mid_v).abs() / (hh * 2.));
    pv += pose.angle_y * hh * 0.0018 * weight;
    let mut p = hf.to_world(pu, pv);
    if id.starts_with("hair") {
        let (amp, root_offset) = hair_sway_profile(id);
        let role_param = if id.starts_with("hair_side") {
            pose.hair_side
        } else if id.starts_with("hair_back") {
            pose.hair_back
        } else {
            0.0
        };
        let sway_param = (pose.hair + role_param).clamp(-1.5, 1.5);
        let root_y = lm.crown.y + hh * root_offset;
        p.x += sway_param * hh * 0.04 * amp * ((p.y - root_y) / hh).clamp(0., 1.8);
    } else if id.starts_with("hat") {
        // Crown stays locked to the skull; lower hood flaps breathe subtly with hair sway.
        let sway_param = pose.hair.clamp(-1.5, 1.5);
        let root_y = lm.crown.y + hh * 0.45;
        p.x += sway_param * hh * 0.04 * 0.18 * ((p.y - root_y) / hh).clamp(0., 1.8);
    } else if id == "acc_hairpin" || id == "acc_ribbon" || id.starts_with("acc_ribbon") {
        let sway_param = pose.hair.clamp(-1.5, 1.5);
        p.x += sway_param * hh * 0.04 * 0.45 * ((p.y - lm.crown.y) / hh).clamp(0., 1.8);
    }
    let pivot = DVec2::new(cx, lm.chin.y);
    let rot_weight = if id == "neck" { weight * 0.78 } else { weight };
    let rot = glam::DMat2::from_angle(pose.angle_z.to_radians() * rot_weight);
    let head_posed = pivot + rot * (p - pivot);
    let body_follow = if id == "neck" {
        0.45 + 0.55 * weight
    } else {
        1.0
    };
    apply_body_sway(head_posed, body_pivot, hh, pose, body_follow).into()
}
/// Combines a part's strict visible mask with any separated occlusion fill mask.
/// Straight-alpha "over" of one RGBA pixel onto `dst` (8-bit, rounded). The CLI effective
/// composite and the T10 rest-invariance check of template synthesis use this one function.
#[inline]
pub fn blend_straight_pixel(dst: &mut [u8], src: [u8; 4]) {
    let sa = src[3] as f32 / 255.0;
    if sa <= 0.0 {
        return;
    }
    let da = dst[3] as f32 / 255.0;
    let inv = 1.0 - sa;
    let out_a = sa + da * inv;
    if out_a <= 1e-6 {
        return;
    }
    let d_w = da * inv;
    for c in 0..3 {
        let blended = (src[c] as f32 * sa + dst[c] as f32 * d_w) / out_a;
        dst[c] = blended.round().clamp(0.0, 255.0) as u8;
    }
    dst[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

pub fn effective_part_mask(visible_mask: &[u8], fill_mask: Option<&[u8]>) -> Vec<u8> {
    visible_mask
        .iter()
        .enumerate()
        .map(|(i, v)| (*v).max(fill_mask.and_then(|f| f.get(i)).copied().unwrap_or(0)))
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcclusionFillOutput {
    pub rgba: Vec<u8>,
    pub fill_mask: Vec<u8>,
    pub combined_mask: Vec<u8>,
    pub filled_pixels: usize,
    pub fill_kind: String,
    /// T9: region type of the part and how the fill colours were made (Rust only; the browser
    /// fallback `expandFillSeparated` leaves this unset = unclassified diffusion draft).
    pub fill_region: Option<crate::analysis::region_fill::RegionReport>,
    pub warning: Option<String>,
}

/// Expands boundary neighbor pixels strictly into unoccupied space, keeping `visible_mask` separated from `fill_mask`.
/// Erodes 2px inward before sampling colors so outermost lineart/hair shadows do not smear into occluded fills.
pub fn fill_occlusion(
    data: &[u8],
    visible_mask: &[u8],
    w: usize,
    h: usize,
    radius: usize,
    part_id: Option<&str>,
) -> OcclusionFillOutput {
    let len = w.saturating_mul(h);
    let active_len = len.min(visible_mask.len());
    let mut out = data.to_vec();
    let mut fill_mask = vec![0u8; len];
    let mut combined_mask = visible_mask.to_vec();
    let mut filled_pixels = 0usize;

    // 1. Compute inward boundary depth (0 = outer rim touching transparent/unowned pixel)
    let mut depth = vec![u8::MAX; len];
    let mut erode_q = VecDeque::new();
    for i in 0..active_len {
        if visible_mask[i] > 0 && neighbors(i, w, h).iter().any(|j| visible_mask[*j] == 0) {
            depth[i] = 0;
            erode_q.push_back(i);
        }
    }
    while let Some(i) = erode_q.pop_front() {
        let next_d = depth[i].saturating_add(1);
        if next_d > 2 {
            continue;
        }
        for j in neighbors(i, w, h) {
            if j < active_len && visible_mask[j] > 0 && depth[j] == u8::MAX {
                depth[j] = next_d;
                erode_q.push_back(j);
            }
        }
    }

    // 2. Pick deepest available interior (>= 2 -> >= 1 -> >= 0) and propagate clean interior RGB to rim.
    let min_seed_depth = [2u8, 1u8, 0u8]
        .into_iter()
        .find(|&d| (0..active_len).any(|i| visible_mask[i] > 0 && depth[i] >= d))
        .unwrap_or(0);
    let mut sample_rgb = vec![[0u8; 3]; len];
    let mut seed_ready = vec![false; len];
    let mut seed_q = VecDeque::new();
    for i in 0..active_len {
        if visible_mask[i] > 0 && depth[i] >= min_seed_depth {
            sample_rgb[i] = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2]];
            seed_ready[i] = true;
            seed_q.push_back(i);
        }
    }
    while let Some(i) = seed_q.pop_front() {
        for j in neighbors(i, w, h) {
            if j < active_len && visible_mask[j] > 0 && !seed_ready[j] {
                sample_rgb[j] = sample_rgb[i];
                seed_ready[j] = true;
                seed_q.push_back(j);
            }
        }
    }

    // 3. BFS outward from visible rim into fill_mask using sample_rgb (leaving visible pixels untouched in `out`).
    let mut frontier = Vec::new();
    for i in 0..active_len {
        if visible_mask[i] > 0 && neighbors(i, w, h).iter().any(|j| visible_mask[*j] == 0) {
            frontier.push(i);
        }
    }
    for _ in 0..radius.min(256) {
        let mut next = Vec::new();
        for i in frontier {
            for j in neighbors(i, w, h) {
                if combined_mask[j] > 0 {
                    continue;
                }
                combined_mask[j] = 255;
                fill_mask[j] = 255;
                filled_pixels += 1;
                let mut sum_r = 0u32;
                let mut sum_g = 0u32;
                let mut sum_b = 0u32;
                let mut count = 0u32;
                for nb in neighbors(j, w, h) {
                    if nb != j
                        && (visible_mask[nb] > 0 || (fill_mask[nb] > 0 && out[nb * 4 + 3] > 0))
                    {
                        let rgb = if seed_ready[nb] {
                            sample_rgb[nb]
                        } else {
                            [out[nb * 4], out[nb * 4 + 1], out[nb * 4 + 2]]
                        };
                        sum_r += rgb[0] as u32;
                        sum_g += rgb[1] as u32;
                        sum_b += rgb[2] as u32;
                        count += 1;
                    }
                }
                let pixel = if let (Some(r), Some(g), Some(b)) = (
                    sum_r.checked_div(count),
                    sum_g.checked_div(count),
                    sum_b.checked_div(count),
                ) {
                    [r as u8, g as u8, b as u8, 255]
                } else if seed_ready[i] {
                    [sample_rgb[i][0], sample_rgb[i][1], sample_rgb[i][2], 255]
                } else {
                    [out[i * 4], out[i * 4 + 1], out[i * 4 + 2], 255]
                };
                sample_rgb[j] = [pixel[0], pixel[1], pixel[2]];
                seed_ready[j] = true;
                out[j * 4..j * 4 + 4].copy_from_slice(&pixel);
                next.push(j);
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    // T9: colours of the grown fill (shape unchanged) from the part's own clean inner seeds —
    // palette level set for tight cel palettes, local harmonic field otherwise — and outer-rim
    // decontamination of the visible texels. Structured / stroke parts keep the diffusion colours.
    let fill_region = if active_len == len && len > 0 {
        use crate::analysis::region_fill as rf;
        let role_kind = part_id.and_then(|id| {
            if id.ends_with("_iris") || id == "mouth_inner" {
                Some(rf::RegionKind::Semantic)
            } else if id.starts_with("brow") || id.ends_with("_upper") || id.ends_with("_lower") {
                Some(rf::RegionKind::Line)
            } else {
                None
            }
        });
        let lab = rf::ImageLab::new(data, len);
        let mut region = rf::analyze_layer(&rf::LayerInput {
            rgba: data,
            lab: &lab,
            w,
            h,
            vis: visible_mask,
            hidden: &fill_mask,
            exclude: None,
            band_radius: radius.clamp(12, 48),
            role_kind,
        });
        let field_ok = matches!(region.kind(), rf::RegionKind::FlatCel | rf::RegionKind::Texture);
        let rim_ok = region.kind() == rf::RegionKind::FlatCel;
        let filler = if field_ok && filled_pixels > 0 {
            rf::RegionFiller::new(&region, data, w, h)
        } else {
            None
        };
        region.report.synth = match &filler {
            _ if filled_pixels == 0 => "none".into(),
            Some(f) if f.snapped() => "palette_levelset".into(),
            Some(_) => "local_field".into(),
            None => "diffusion".into(),
        };
        if let Some(f) = &filler {
            for i in 0..len {
                if fill_mask[i] > 0 {
                    let c = f.color_at(i % w, i / w);
                    out[i * 4..i * 4 + 3].copy_from_slice(&c);
                }
            }
        }
        if rim_ok {
            let owner: Vec<i32> = visible_mask.iter().map(|&v| if v > 0 { 0 } else { -1 }).collect();
            let ctx = rf::RimContext {
                rgba: data,
                lab: &lab,
                w,
                h,
                owner: &owner,
                background: None,
                use_outside_color: true,
            };
            region.report.rim = rf::decontaminate_rim(&mut out, visible_mask, 0, &region, &ctx, None).0;
        }
        Some(region.report)
    } else {
        None
    };
    let hidden_feature_part = part_id
        .map(|id| id.starts_with("eye") || id.starts_with("mouth") || id.starts_with("hair"))
        .unwrap_or(false);
    let warning = if hidden_feature_part {
        Some(
            "경계 2px 침식 내삽(neighbor)으로 외곽선 번짐을 줄였으나 가려진 눈·입·머리카락 내부 디테일은 자동 복원되지 않습니다."
                .to_string(),
        )
    } else {
        None
    };
    OcclusionFillOutput {
        rgba: out,
        fill_mask,
        combined_mask,
        filled_pixels,
        fill_kind: "neighbor".to_string(),
        fill_region,
        warning,
    }
}

pub fn fill(data: &[u8], mask: &[u8], w: usize, h: usize, radius: usize) -> (Vec<u8>, Vec<u8>) {
    let out = fill_occlusion(data, mask, w, h, radius, None);
    (out.rgba, out.combined_mask)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshQualityReport {
    pub part_id: String,
    pub triangle_count: usize,
    pub hole_crossings: usize,
    pub long_triangles: usize,
    pub flipped_at_extremes: usize,
    pub extreme_poses_tested: usize,
    pub warnings: Vec<String>,
}

fn tri_cross(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Checks triangle hole crossings, excessive aspect elongation, and extreme-pose flips.
/// Hole crossings are judged against the B4 mesh support (`mesh_support_mask`), the region the
/// ArtMesh is built to cover; `mask` is the part's effective alpha.
pub fn check_mesh_quality(
    mesh: &Mesh,
    mask: &[u8],
    w: usize,
    h: usize,
    id: &str,
    lm: &Landmarks,
) -> MeshQualityReport {
    let tris = mesh.triangles.as_chunks::<3>().0;
    let triangle_count = tris.len();
    let has_pixels = mask.iter().any(|&a| a > 0);
    if triangle_count == 0 || mesh.vertices.is_empty() {
        return MeshQualityReport {
            part_id: id.to_string(),
            triangle_count: 0,
            hole_crossings: 0,
            long_triangles: 0,
            flipped_at_extremes: 0,
            extreme_poses_tested: 0,
            warnings: if has_pixels {
                vec![format!("{id}: 유효한 메쉬 삼각형이 생성되지 않았습니다.")]
            } else {
                vec![]
            },
        };
    }
    let support = mesh_support_mask(mask, w, h);
    let mask: &[u8] = &support;

    let valid_px = |p: DVec2| {
        let px = p.x + 1e-6;
        let py = p.y + 1e-6;
        px >= 0.
            && py >= 0.
            && (px as usize) < w
            && (py as usize) < h
            && mask[py as usize * w + px as usize] > 8
    };

    let mut max_edges = Vec::with_capacity(triangle_count);
    let mut rest_crosses = Vec::with_capacity(triangle_count);
    let mut hole_crossings = 0usize;

    for ids in tris {
        let a = mesh.vertices[ids[0]].v();
        let b = mesh.vertices[ids[1]].v();
        let c = mesh.vertices[ids[2]].v();
        let max_e = a.distance(b).max(a.distance(c)).max(b.distance(c));
        max_edges.push(max_e);
        rest_crosses.push(tri_cross(a, b, c));
        let centroid = (a + b + c) / 3.0;
        let q1 = a * 0.5 + b * 0.25 + c * 0.25;
        let q2 = a * 0.25 + b * 0.5 + c * 0.25;
        let q3 = a * 0.25 + b * 0.25 + c * 0.5;
        if !valid_px(centroid) || !valid_px(q1) || !valid_px(q2) || !valid_px(q3) {
            hole_crossings += 1;
        }
    }

    let mut sorted_edges = max_edges.clone();
    sorted_edges.sort_by(|x, y| x.total_cmp(y));
    let median_edge = sorted_edges[sorted_edges.len() / 2];
    let long_threshold = (median_edge * 3.2).max(48.0);
    let long_triangles = max_edges
        .iter()
        .filter(|&&e| e > long_threshold)
        .count();

    let extreme_poses = [
        Pose { angle_x: -30., angle_y: -30., angle_z: -30., ..Pose::default() },
        Pose { angle_x: 30., angle_y: 30., angle_z: 30., ..Pose::default() },
        Pose { angle_x: -30., angle_y: 30., angle_z: 15., body_angle_x: -10., body_angle_z: -10., ..Pose::default() },
        Pose { angle_x: 30., angle_y: -30., angle_z: -15., body_angle_x: 10., body_angle_z: 10., ..Pose::default() },
        Pose { eye_l: 0., eye_r: 0., eye_l_smile: 1., eye_r_smile: 1., eye_ball_x: -1., eye_ball_y: 1., ..Pose::default() },
        Pose { eye_l: 1.2, eye_r: 1.2, eye_ball_x: 1., eye_ball_y: -1., ..Pose::default() },
        Pose { mouth: 0., mouth_form: -1., brow_l_y: -1., brow_l_form: -1., brow_r_y: -1., brow_r_form: -1., ..Pose::default() },
        Pose { mouth: 1., mouth_form: 1., brow_l_y: 1., brow_l_form: 1., brow_r_y: 1., brow_r_form: 1., ..Pose::default() },
        Pose { hair: -1., hair_side: -1., hair_back: -1., ..Pose::default() },
        Pose { hair: 1., hair_side: 1., hair_back: 1., breath: 1., ..Pose::default() },
        Pose {
            angle_x: 30., angle_y: -30., angle_z: 30.,
            eye_l: 0., eye_r: 0., mouth: 1., mouth_form: 1.,
            body_angle_x: 10., body_angle_y: -10., body_angle_z: 10., breath: 1.,
            hair: -1., hair_side: -1., hair_back: -1.,
            ..Pose::default()
        },
        Pose {
            angle_x: -30., angle_y: 30., angle_z: -30.,
            eye_l: 0., eye_r: 0., mouth: 1., mouth_form: -1.,
            body_angle_x: -10., body_angle_y: 10., body_angle_z: -10., breath: 0.,
            hair: 1., hair_side: 1., hair_back: 1.,
            ..Pose::default()
        },
    ];

    let mut flipped_mask = vec![false; triangle_count];
    for pose in &extreme_poses {
        let posed_verts: Vec<DVec2> = mesh
            .vertices
            .iter()
            .map(|&v| deform(v, id, lm, pose).v())
            .collect();
        for (ti, ids) in tris.iter().enumerate() {
            if rest_crosses[ti].abs() < 1e-6 || flipped_mask[ti] {
                continue;
            }
            let posed_cross = tri_cross(
                posed_verts[ids[0]],
                posed_verts[ids[1]],
                posed_verts[ids[2]],
            );
            // T10: a triangle collapsing to a sliver (the closing eye aperture: its vertices land
            // on the curved closing line, so the sliver's orientation follows the curvature) is
            // not a visible inversion; only a signed area below -5 % of the rest area counts.
            let r = rest_crosses[ti];
            if posed_cross * r.signum() < -EXTREME_FLIP_COLLAPSE_RATIO * r.abs() {
                flipped_mask[ti] = true;
            }
        }
    }
    let flipped_at_extremes = flipped_mask.into_iter().filter(|&b| b).count();

    let mut warnings = Vec::new();
    if hole_crossings > 0 {
        warnings.push(format!("{id}: 빈 구멍(hole) 횡단 삼각형 {hole_crossings}개"));
    }
    if long_triangles > 0 {
        warnings.push(format!("{id}: 과도하게 긴 삼각형 {long_triangles}개"));
    }
    if flipped_at_extremes > 0 {
        warnings.push(format!(
            "{id}: 극단 포즈에서 뒤집히는 삼각형 {flipped_at_extremes}개"
        ));
    }

    MeshQualityReport {
        part_id: id.to_string(),
        triangle_count,
        hole_crossings,
        long_triangles,
        flipped_at_extremes,
        extreme_poses_tested: extreme_poses.len(),
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lm() -> Landmarks {
        Landmarks {
            eye_l: Point { x: 38., y: 24. },
            eye_r: Point { x: 22., y: 24. },
            mouth: Point { x: 30., y: 33. },
            chin: Point { x: 30., y: 40. },
            crown: Point { x: 30., y: 4. },
            eye_frames: None,
            torso_frame: None,
        }
    }
    #[test]
    fn segmentation_preserves_each_alpha_exactly_once() {
        let rgba = vec![255; 64 * 64 * 4];
        let parts = separate(&rgba, 64, 64, &lm(), 0., None);
        for i in 0..64 * 64 {
            assert_eq!(parts.iter().map(|p| p[i] as u32).sum::<u32>(), 255);
        }
    }
    #[test]
    fn border_flood_preserves_enclosed_same_color() {
        let mut rgba = vec![255; 9 * 9 * 4];
        for y in 2..7 {
            for x in 2..7 {
                let i = (y * 9 + x) * 4;
                rgba[i..i + 3].fill(0);
            }
        }
        rgba[(4 * 9 + 4) * 4..(4 * 9 + 4) * 4 + 3].fill(255);
        let mask = foreground(&rgba, 9, 9, 20.);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[4 * 9 + 4], 255);
    }
    #[test]
    fn mesh_does_not_cross_a_hole() {
        let mut mask = vec![255; 32 * 32];
        for y in 10..22 {
            for x in 10..22 {
                mask[y * 32 + x] = 0;
            }
        }
        let mesh = mesh_for(&mask, 32, 32, 14);
        assert!(!mesh.triangles.is_empty());
        // B4: the mesh extends `MESH_SUPPORT_RADIUS` px into the hole (transparent margin for the
        // anti-aliased edge) but never across the rest of it.
        let support = mesh_support_mask(&mask, 32, 32);
        for t in mesh.triangles.as_chunks::<3>().0 {
            let p =
                (mesh.vertices[t[0]].v() + mesh.vertices[t[1]].v() + mesh.vertices[t[2]].v()) / 3.;
            assert!(support[p.y as usize * 32 + p.x as usize] > 0);
        }
        let mut hole_drawn = 0;
        for t in mesh.triangles.as_chunks::<3>().0 {
            let tri = [mesh.vertices[t[0]], mesh.vertices[t[1]], mesh.vertices[t[2]]];
            crate::rig::raster::rasterize_triangle(tri, 32, 32, |x, y, _| {
                if (13..19).contains(&x) && (13..19).contains(&y) {
                    hole_drawn += 1;
                }
            });
        }
        assert_eq!(hole_drawn, 0, "mesh covers the hole centre");
    }
    #[test]
    fn contour_mesh_covers_boundary_and_passes_extreme_pose_quality() {
        let mut mask = vec![0u8; 64 * 64];
        for y in 8..56 {
            for x in 14..50 {
                mask[y * 64 + x] = 255;
            }
        }
        let landmarks = lm();
        let mesh = mesh_for_part(&mask, 64, 64, 14, &landmarks);
        let report = check_mesh_quality(&mesh, &mask, 64, 64, "face", &landmarks);
        assert!(report.triangle_count > 10);
        assert_eq!(report.hole_crossings, 0);
        assert_eq!(report.flipped_at_extremes, 0);
        assert!(mesh.vertices.iter().any(|v| (v.x - landmarks.chin.x).abs() < 4.0 && (v.y - landmarks.chin.y).abs() < 4.0));
    }
    #[test]
    fn rest_pose_and_rotation_keep_pivot() {
        let l = lm();
        let p = Point { x: 20., y: 30. };
        let q = deform(p, "face", &l, &Pose::default());
        assert!((p.x - q.x).abs() < 1e-9);
        assert!((p.y - q.y).abs() < 1e-9);
        let pivot = Point { x: 30., y: 40. };
        let q = deform(
            pivot,
            "face",
            &l,
            &Pose {
                angle_z: 15.,
                ..Pose::default()
            },
        );
        assert_eq!(q.x, pivot.x);
        assert_eq!(q.y, pivot.y);
    }
    #[test]
    fn invalid_payload_rejected() {
        assert!(check_size(100000, 1).is_err());
        assert!(decode_bytes("AAAA", 2).is_err());
    }
    #[test]
    fn fill_stays_inside_canvas_and_keeps_visible_separated() {
        let mut mask = vec![0; 9];
        mask[0] = 255;
        let (data, expanded) = fill(&[255; 36], &mask, 3, 3, 1);
        assert_eq!(expanded.iter().filter(|a| **a > 0).count(), 3);
        assert_eq!(data.len(), 36);

        let occ = fill_occlusion(&[255; 36], &mask, 3, 3, 1, Some("eye_r"));
        assert_eq!(occ.filled_pixels, 2);
        assert_eq!(occ.fill_kind, "neighbor");
        assert!(occ.warning.is_some());
        for (i, (&m, &fm)) in mask.iter().zip(occ.fill_mask.iter()).enumerate() {
            assert!(!(m > 0 && fm > 0));
            assert_eq!(occ.combined_mask[i], m.max(fm));
        }
        assert_eq!(effective_part_mask(&mask, Some(&occ.fill_mask)), occ.combined_mask);
    }
    #[test]
    fn eroded_fill_occlusion_avoids_outer_dark_outline() {
        let w = 11usize;
        let h = 11usize;
        let mut rgba = vec![0u8; w * h * 4];
        let mut mask = vec![0u8; w * h];
        for y in 2..9 {
            for x in 2..9 {
                let i = y * w + x;
                mask[i] = 255;
                let is_rim = x == 2 || x == 8 || y == 2 || y == 8;
                let rgb = if is_rim { [10, 10, 10] } else { [240, 200, 180] };
                rgba[i * 4..i * 4 + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            }
        }
        let occ = fill_occlusion(&rgba, &mask, w, h, 2, Some("face"));
        assert!(occ.filled_pixels > 0);
        // Check expanded ring pixel (1, 5) just outside the dark rim: must sample interior skin, not dark outline!
        let probe = 5 * w + 1;
        assert_eq!(occ.fill_mask[probe], 255);
        assert!(occ.rgba[probe * 4] >= 220, "expected interior skin RGB, got {:?}", &occ.rgba[probe * 4..probe * 4 + 4]);
        // Original visible pixels remain completely intact.
        let rim = 5 * w + 2;
        assert_eq!(occ.rgba[rim * 4], 10);
    }
    #[test]
    fn hair_sway_profile_scales_by_role_and_strand() {
        let l = lm();
        let pt = Point { x: 30., y: 30. };
        let pose = Pose { hair: 1.0, ..Pose::default() };
        let front = deform(pt, "hair_front", &l, &pose);
        let front2 = deform(pt, "hair_front_2", &l, &pose);
        let side = deform(pt, "hair_side_L", &l, &pose);
        let back = deform(pt, "hair_back", &l, &pose);
        assert!(front2.x > front.x);
        assert!(side.x > back.x);
    }
    #[test]
    fn part_overlap_margins_scale_with_resolution_and_cover_opposing_motion() {
        let scale_lm = |s: f64| Landmarks {
            eye_l: Point { x: 38.0 * s, y: 24.0 * s },
            eye_r: Point { x: 22.0 * s, y: 24.0 * s },
            mouth: Point { x: 30.0 * s, y: 33.0 * s },
            chin: Point { x: 30.0 * s, y: 40.0 * s },
            crown: Point { x: 30.0 * s, y: 4.0 * s },
            eye_frames: None,
            torso_frame: None,
        };
        let m_1024 = part_overlap_margins(&scale_lm(16.0), 1024, 1024);
        let m_2048 = part_overlap_margins(&scale_lm(32.0), 2048, 2048);
        assert!(m_1024.neck_collar_x > 14 && m_1024.hair_side_front_seam > 14);
        assert!(m_2048.neck_collar_x > m_1024.neck_collar_x);
        assert!(m_2048.hair_back_underlap > m_1024.hair_back_underlap);

        let lm_1024 = scale_lm(16.0);
        // Verify opposing head-body displacement at mid-collar stays within neck_collar_x/y.
        let collar_pt = Point { x: lm_1024.chin.x, y: lm_1024.chin.y + 120.0 };
        let opposing = Pose {
            angle_x: -30.0,
            angle_y: -30.0,
            angle_z: -20.0,
            body_angle_x: 10.0,
            body_angle_y: 10.0,
            body_angle_z: 10.0,
            breath: 1.0,
            ..Pose::default()
        };
        let neck_p = deform(collar_pt, "neck", &lm_1024, &opposing);
        let clothes_p = deform(collar_pt, "clothes", &lm_1024, &opposing);
        assert!(
            (neck_p.x - clothes_p.x).abs() <= m_1024.neck_collar_x as f64,
            "neck_p.x={} clothes_p.x={} margin={}",
            neck_p.x,
            clothes_p.x,
            m_1024.neck_collar_x
        );
        assert!(
            (neck_p.y - clothes_p.y).abs() <= m_1024.neck_collar_y as f64,
            "neck_p.y={} clothes_p.y={} margin={}",
            neck_p.y,
            clothes_p.y,
            m_1024.neck_collar_y
        );

        // Verify opposing hair_side vs hair_front sway at temple/cheek stays within hair_side_front_seam.
        let cheek_hair_pt = Point { x: lm_1024.eye_r.x - 100.0, y: lm_1024.chin.y };
        let sway_pose = Pose { hair: 1.0, hair_side: -1.0, ..Pose::default() };
        let front_p = deform(cheek_hair_pt, "hair_front", &lm_1024, &sway_pose);
        let side_p = deform(cheek_hair_pt, "hair_side_R", &lm_1024, &sway_pose);
        assert!(
            (front_p.x - side_p.x).abs() <= m_1024.hair_side_front_seam as f64,
            "front_p.x={} side_p.x={} seam={}",
            front_p.x,
            side_p.x,
            m_1024.hair_side_front_seam
        );
    }
    #[test]
    fn recommended_fill_radius_matches_shared_rust_ts_parity_fixture() {
        #[derive(Deserialize)]
        struct RoleCase {
            id: String,
            role: PartRole,
        }
        #[derive(Deserialize)]
        struct PartCase {
            id: String,
            role: Option<PartRole>,
            radius: usize,
        }
        #[derive(Deserialize)]
        struct FrameCase {
            label: String,
            width: usize,
            height: usize,
            landmarks: Landmarks,
            parts: Vec<PartCase>,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Fixture {
            role_inference: Vec<RoleCase>,
            frames: Vec<FrameCase>,
        }
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../tests/fixtures/fill-radius-parity.json"))
                .expect("parse fill-radius parity fixture");
        assert!(!fixture.role_inference.is_empty() && !fixture.frames.is_empty());
        for c in &fixture.role_inference {
            assert_eq!(infer_part_role(&c.id, None), c.role, "role of {}", c.id);
        }
        let mut checked = 0;
        for f in &fixture.frames {
            for p in &f.parts {
                let got = recommended_fill_radius(&p.id, p.role, &f.landmarks, f.width, f.height);
                assert_eq!(got, p.radius, "{} / {} ({:?})", f.label, p.id, p.role);
                checked += 1;
            }
        }
        // Variable multi-hair IDs must resolve through roles, not the fixed 18-ID table.
        let ids: Vec<&str> = fixture.frames[0].parts.iter().map(|p| p.id.as_str()).collect();
        for id in ["hair_front_2", "hair_side_L_2", "hair_side_R_3", "hair_back_3"] {
            assert!(ids.contains(&id), "fixture must cover {id}");
            assert!(!IDS.contains(&id));
        }
        assert!(checked > 0);
    }
    #[test]
    fn large_curved_torso_mesh_is_dense_and_hole_free() {
        // Large C-shaped off-shoulder silhouette (720x520) matching wide sleeve/collar geometry.
        let w = 720usize;
        let h = 520usize;
        let mut mask = vec![0u8; w * h];
        for y in 40..500 {
            for x in 40..680 {
                // Carve U-shaped neck cutout in upper-center
                let in_neck_bay = y < 220 && (x as f64 - 360.0).abs() < 85.0;
                // Curved shoulder outer contour
                let dx = (x as f64 - 360.0) / 315.0;
                let top_curve = 65.0 + 110.0 * dx * dx;
                if !in_neck_bay && (y as f64) >= top_curve {
                    mask[y * w + x] = 255;
                }
            }
        }
        let mesh = mesh_for(&mask, w, h, 14);
        let l = Landmarks {
            eye_l: Point { x: 400., y: 120. },
            eye_r: Point { x: 320., y: 120. },
            mouth: Point { x: 360., y: 165. },
            chin: Point { x: 360., y: 195. },
            crown: Point { x: 360., y: 40. },
            eye_frames: None,
            torso_frame: None,
        };
        let report = check_mesh_quality(&mesh, &mask, w, h, "clothes", &l);
        assert!(report.triangle_count >= 120, "expected >=120 tris, got {}", report.triangle_count);
        assert_eq!(report.hole_crossings, 0);
        assert_eq!(report.flipped_at_extremes, 0);
    }
    #[test]
    fn new_body_eye_brow_mouth_hair_params_deform_correctly() {
        let l = lm();
        let eye_pt = Point { x: 38., y: 24. };
        let gazed = deform(eye_pt, "eyeL_iris", &l, &Pose { eye_ball_x: 1., eye_ball_y: 1., ..Pose::default() });
        assert!(gazed.x > eye_pt.x && gazed.y < eye_pt.y);

        let brow_pt = Point { x: 38., y: 18. };
        let brow_up = deform(brow_pt, "browL", &l, &Pose { brow_l_y: 1., ..Pose::default() });
        assert!(brow_up.y < brow_pt.y);

        let shoulder = Point { x: 48., y: 52. };
        assert_eq!(deform(shoulder, "clothes", &l, &Pose::default()).x, shoulder.x);
        let breathed = deform(shoulder, "clothes", &l, &Pose { breath: 1., ..Pose::default() });
        assert!(breathed.y < shoulder.y);

        let hair_side = Point { x: 30., y: 30. };
        let moved_side = deform(hair_side, "hair_side_L", &l, &Pose { hair_side: 1., ..Pose::default() });
        let unmoved_front = deform(hair_side, "hair_front", &l, &Pose { hair_side: 1., ..Pose::default() });
        assert!(moved_side.x > hair_side.x);
        assert!((unmoved_front.x - hair_side.x).abs() < 1e-9);

        // Glasses stay anchored to head rotation without eye blink collapse; hairpins sway softly with hair.
        let glasses_pt = Point { x: 38., y: 24. };
        let glasses_blink = deform(glasses_pt, "acc_glasses", &l, &Pose { eye_l: 0., hair: 1., ..Pose::default() });
        assert!((glasses_blink.y - glasses_pt.y).abs() < 1e-9);
        assert!((glasses_blink.x - glasses_pt.x).abs() < 1e-9);
        let pin_pt = Point { x: 40., y: 26. };
        let pin_sway = deform(pin_pt, "acc_hairpin", &l, &Pose { hair: 1., ..Pose::default() });
        assert!(pin_sway.x > pin_pt.x);
        let collar_breathed = deform(shoulder, "clothes_collar", &l, &Pose { breath: 1., ..Pose::default() });
        assert!((collar_breathed.y - breathed.y).abs() < 1e-9);
    }
    #[test]
    fn closed_eyelid_keeps_smooth_brow_to_lid_curve_without_cheek_droop() {
        let l = lm();
        let closed = Pose {
            eye_l: 0.,
            ..Pose::default()
        };
        let top_corner = Point { x: 43., y: 20. };
        let mid_lid = Point { x: 38., y: 22.5 };
        let low_corner = Point { x: 43., y: 25. };
        let d_top = deform(top_corner, "eyeL_upper", &l, &closed);
        let d_mid = deform(mid_lid, "eyeL_upper", &l, &closed);
        let d_low = deform(low_corner, "eyeL_upper", &l, &closed);
        // T10: lid texels move down monotonically; below the lash line (default frame) they
        // translate with the lash, whose travel shrinks toward the thin aperture corners, so the
        // outer corner moves at most a tenth of an eye distance (no droop onto the cheek).
        let dist = 16.0;
        assert!(d_top.y >= top_corner.y);
        assert!(d_mid.y >= mid_lid.y && d_mid.y <= l.eye_l.y + dist * 0.2);
        assert!(d_low.y >= low_corner.y - 1e-6, "{d_low:?}");
        assert!(d_low.y - low_corner.y <= dist * 0.1, "outer corner droops onto cheek: {d_low:?}");
        assert!(d_mid.y >= d_top.y);
    }
    #[test]
    fn art_mesh_builds_role_aware_radial_iris_arch_eyelid_and_thin_ribbon_without_orphans() {
        let l = lm();
        let w = 64usize;
        let h = 64usize;
        // 1. Circular iris -> expects center pupil vertex & zero orphan vertices.
        let mut iris_mask = vec![0u8; w * h];
        for y in 16..33 {
            for x in 30..47 {
                let dx = x as f64 - 38.0;
                let dy = y as f64 - 24.0;
                if dx * dx + dy * dy <= 49.0 {
                    iris_mask[y * w + x] = 255;
                }
            }
        }
        let iris_mesh = art_mesh_for_part(&iris_mask, w, h, 14, "eyeL_iris", &l);
        assert!(!iris_mesh.triangles.is_empty());
        assert_eq!(iris_mesh.uv.len(), iris_mesh.vertices.len());
        let mut used = vec![false; iris_mesh.vertices.len()];
        for &idx in &iris_mesh.triangles {
            used[idx] = true;
        }
        assert!(used.into_iter().all(|b| b), "iris ArtMesh must contain zero orphan vertices");
        assert!(
            iris_mesh.vertices.iter().any(|v| (v.x - 38.5).abs() < 1.2 && (v.y - 24.5).abs() < 1.2),
            "iris ArtMesh should place pupil center vertex"
        );

        // 2. Thin curved hairline accessory / brow -> must never drop to 0 triangles.
        let mut thin_pin = vec![0u8; w * h];
        for x in 18..42 {
            let y = 14 + ((x as isize - 30).abs() / 6) as usize;
            thin_pin[y * w + x] = 255;
            thin_pin[(y + 1) * w + x] = 255;
        }
        let pin_mesh = art_mesh_for_part(&thin_pin, w, h, 14, "acc_hairpin", &l);
        let pin_report = check_mesh_quality(&pin_mesh, &thin_pin, w, h, "acc_hairpin", &l);
        assert!(pin_report.triangle_count >= 6, "thin hairpin collapsed to {} tris", pin_report.triangle_count);
        assert_eq!(pin_report.hole_crossings, 0);
        assert_eq!(pin_report.flipped_at_extremes, 0);

        // 3. Curved eyelid -> smooth contour + arch strips with 0 extreme-pose flips.
        let mut lid_mask = vec![0u8; w * h];
        for x in 28..49 {
            let dx = (x as f64 - 38.0) / 10.0;
            let yc = 20.0 + 3.0 * dx * dx;
            for y in 0..h {
                if (y as f64 - yc).abs() <= 2.2 {
                    lid_mask[y * w + x] = 255;
                }
            }
        }
        let lid_mesh = art_mesh_for_part(&lid_mask, w, h, 14, "eyeL_upper", &l);
        let lid_report = check_mesh_quality(&lid_mesh, &lid_mask, w, h, "eyeL_upper", &l);
        assert!(lid_report.triangle_count >= 12);
        assert_eq!(lid_report.hole_crossings, 0);
        assert_eq!(lid_report.flipped_at_extremes, 0);
    }

    /// B4: at rest an ArtMesh covers every texel with alpha > 0 (anti-aliased edges, faint strokes,
    /// isolated specks, thin strands, holes) plus its 8-neighbours, each pixel centre exactly once
    /// under the renderer's top-left rule — for every mesh role.
    #[test]
    fn art_mesh_covers_every_alpha_texel_once_for_every_role() {
        let (w, h) = (96usize, 96usize);
        let mut mask = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
                // Disk with a 3px anti-aliased falloff and a hole with its own AA ring.
                let r = ((fx - 34.0).powi(2) + (fy - 40.0).powi(2)).sqrt();
                let disk = ((22.0 - r) / 3.0).clamp(0.0, 1.0) * ((r - 6.0) / 2.0).clamp(0.0, 1.0);
                // A comb of 1–2px strands (hair tips) with faint alpha, 3px apart.
                let strand = if (62..86).contains(&x) && x % 4 == 0 && y > 10 + x / 4 && y < 80 {
                    if y > 70 { 0.04 } else { 0.6 }
                } else {
                    0.0
                };
                // A faint 1px diagonal stroke.
                let stroke = if (x as i64 - y as i64) == 20 && (8..40).contains(&y) { 0.02 } else { 0.0 };
                let a = disk.max(strand).max(stroke);
                mask[y * w + x] = (a * 255.0).round() as u8;
            }
        }
        // Isolated specks of alpha 1–3.
        for (i, &(x, y)) in [(5usize, 5usize), (90, 6), (8, 88), (70, 90)].iter().enumerate() {
            mask[y * w + x] = 1 + i as u8 % 3;
        }
        let l = Landmarks {
            eye_l: Point { x: 44., y: 36. },
            eye_r: Point { x: 24., y: 36. },
            mouth: Point { x: 34., y: 50. },
            chin: Point { x: 34., y: 60. },
            crown: Point { x: 34., y: 18. },
            eye_frames: None,
            torso_frame: None,
        };
        let target = mesh_coverage_target(&mask, w, h);
        let mut parity_meshes = Vec::new();
        for id in ["face", "hair_front", "eyeL_iris", "eyeL_white", "eyeL_upper", "browL", "clothes", ""] {
            let mesh = if id.is_empty() { mesh_for(&mask, w, h, 14) } else { art_mesh_for_part(&mask, w, h, 14, id, &l) };
            parity_meshes.push(serde_json::json!({
                "id": id,
                "vertices": mesh.vertices.iter().map(|v| [v.x, v.y]).collect::<Vec<_>>(),
                "triangles": mesh.triangles,
            }));
            let mut count = vec![0u32; w * h];
            for t in mesh.triangles.as_chunks::<3>().0 {
                let tri = [mesh.vertices[t[0]], mesh.vertices[t[1]], mesh.vertices[t[2]]];
                crate::rig::raster::rasterize_triangle(tri, w, h, |x, y, _| count[y * w + x] += 1);
            }
            assert!(count.iter().all(|&c| c <= 1), "{id}: overlapping triangles");
            let missed_alpha = (0..w * h).filter(|&i| mask[i] > 0 && count[i] == 0).count();
            let missed_target = (0..w * h).filter(|&i| target[i] && count[i] == 0).count();
            assert_eq!(missed_alpha, 0, "{id}: {missed_alpha} alpha > 0 texels outside the mesh");
            assert_eq!(missed_target, 0, "{id}: {missed_target} edge-footprint texels outside the mesh");
            // No last-resort triangle spanning a gap was needed.
            let q = check_mesh_quality(&mesh, &mask, w, h, if id.is_empty() { "generic" } else { id }, &l);
            assert_eq!(q.hole_crossings, 0, "{id}: hole-crossing triangles");
        }
        // Rust/TS parity fixture (`tests/pipeline.test.ts` rebuilds the same meshes with `engine.ts`).
        let doc = serde_json::json!({
            "description": "B4 Rust/TS ArtMesh parity: core::art_mesh_for_part / mesh_for over a mask with AA edges, a hole, faint strands, a faint stroke and specks (alpha > 0 pixels as [index, alpha]); TS meshForPart / meshFor must return the same vertices and triangles and cover every alpha > 0 texel.",
            "width": w,
            "height": h,
            "alpha": mask.iter().enumerate().filter(|(_, &a)| a > 0).map(|(i, &a)| [i as u32, a as u32]).collect::<Vec<_>>(),
            "landmarks": l,
            "meshes": parity_meshes,
        });
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/mesh-coverage-parity.json");
        if std::env::var("FLAT2RIG_BLESS").is_ok() {
            std::fs::write(&path, serde_json::to_string(&doc).unwrap() + "\n").unwrap();
        }
        let fixture = std::fs::read_to_string(&path).expect("mesh parity fixture present").replace("\r\n", "\n");
        assert_eq!(fixture, serde_json::to_string(&doc).unwrap() + "\n", "mesh parity fixture drifted (re-bless with FLAT2RIG_BLESS=1)");
    }

    #[test]
    fn deform_with_shapes_matches_rest_and_transforms_contours() {
        let l = lm();
        let eye_pt = Point { x: 40., y: 22. };
        let neutral = RegionDeformParams::neutral();
        let shape_neutral = ShapeDeformInput {
            eye_l: Some(neutral),
            eye_r: Some(neutral),
            mouth: Some(neutral),
        };
        let rest_4 = deform(eye_pt, "eyeL_upper", &l, &Pose::default());
        let rest_5 = deform_with_shapes(eye_pt, "eyeL_upper", &l, &Pose::default(), Some(&shape_neutral));
        assert!((rest_4.x - rest_5.x).abs() < 1e-6);
        assert!((rest_4.y - rest_5.y).abs() < 1e-6);

        let widened = ShapeDeformInput {
            eye_l: Some(RegionDeformParams {
                width: 1.25,
                upper_curve_delta: 0.4,
                corner: 0.3,
                ..neutral
            }),
            ..ShapeDeformInput::default()
        };
        let moved = deform_with_shapes(eye_pt, "eyeL_upper", &l, &Pose::default(), Some(&widened));
        assert!(moved.x > eye_pt.x);
        assert!(moved.y < eye_pt.y);

        let mouth_corner = Point { x: 36., y: 36. };
        let pursed = ShapeDeformInput {
            mouth: Some(RegionDeformParams {
                width: 0.6,
                ..neutral
            }),
            ..ShapeDeformInput::default()
        };
        let pursed_pt = deform_with_shapes(mouth_corner, "mouth_upper", &l, &Pose::default(), Some(&pursed));
        assert!(pursed_pt.x < mouth_corner.x);
    }

    #[test]
    fn tilted_head_frame_aligns_smile_arch_and_sad_u_curves() {
        // Simulate ~12 deg head tilt matching portrait illustrations
        let tilt_lm = Landmarks {
            eye_r: Point { x: 40., y: 36. },
            eye_l: Point { x: 80., y: 44. },
            mouth: Point { x: 56., y: 66. },
            chin: Point { x: 54., y: 84. },
            crown: Point { x: 68., y: 8. },
            eye_frames: None,
            torso_frame: None,
        };
        let hf = HeadFrame::from_landmarks(&tilt_lm, 128, 128);
        let (eye_lu, eye_lv) = hf.to_local(DVec2::new(tilt_lm.eye_l.x, tilt_lm.eye_l.y));
        // T10: probe the lash line itself (the lid's lower edge, default frame without masks); it
        // lands on the closing line, whose expression curvature is checked here.
        let hf0 = HeadFrame::from_landmarks(&tilt_lm, 0, 0);
        let frame = default_eye_frame(eye_lu, eye_lv, hf0.eye_dist);
        let (lash_c, _) = eye_lines_at(&frame, eye_lu);
        let (lash_k, _) = eye_lines_at(&frame, eye_lu + 10.0);
        let tip = hf.to_world(eye_lu, lash_c);
        let corner = hf.to_world(eye_lu + 10.0, lash_k);

        // Smiling expression arches real upper eyelid upward (^) along HeadFrame -v
        let smile_pose = Pose {
            eye_l: 0.0,
            eye_l_smile: 1.0,
            ..Default::default()
        };
        let smile_shape = ShapeDeformInput {
            eye_l: Some(RegionDeformParams {
                upper_curve_delta: 0.9,
                ..RegionDeformParams::neutral()
            }),
            ..ShapeDeformInput::default()
        };
        let tip_s = deform_with_shapes(
            tip.into(),
            "eyeL_upper",
            &tilt_lm,
            &smile_pose,
            Some(&smile_shape),
        );
        let corner_s = deform_with_shapes(
            corner.into(),
            "eyeL_upper",
            &tilt_lm,
            &smile_pose,
            Some(&smile_shape),
        );
        assert!(
            hf.to_local(DVec2::new(tip_s.x, tip_s.y)).1
                < hf.to_local(DVec2::new(corner_s.x, corner_s.y)).1
        );

        // Sad expression dips real upper eyelid downward into (u) where center v > corner v
        let sad_pose = Pose {
            eye_l: 0.0,
            ..Default::default()
        };
        let sad_shape = ShapeDeformInput {
            eye_l: Some(RegionDeformParams {
                upper_curve_delta: -0.95,
                ..RegionDeformParams::neutral()
            }),
            ..ShapeDeformInput::default()
        };
        let tip_sad = deform_with_shapes(
            tip.into(),
            "eyeL_upper",
            &tilt_lm,
            &sad_pose,
            Some(&sad_shape),
        );
        let corner_sad = deform_with_shapes(
            corner.into(),
            "eyeL_upper",
            &tilt_lm,
            &sad_pose,
            Some(&sad_shape),
        );
        assert!(
            hf.to_local(DVec2::new(tip_sad.x, tip_sad.y)).1
                > hf.to_local(DVec2::new(corner_sad.x, corner_sad.y)).1
        );
    }
}
