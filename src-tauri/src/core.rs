use base64::{engine::general_purpose::STANDARD, Engine};
use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

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
}
impl Landmarks {
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

fn build_art_mesh(
    mask: &[u8],
    w: usize,
    h: usize,
    density: usize,
    role: ArtMeshRole,
    lm: Option<&Landmarks>,
    anchors: &[Point],
) -> Mesh {
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
        let points: Vec<_> = mesh
            .vertices
            .iter()
            .map(|p| delaunator::Point { x: p.x, y: p.y })
            .collect();
        let result = delaunator::triangulate(&points);
        let thin_part = mesh.vertices.len() < 40;
        let max_edge = if thin_part {
            (span * 0.90).max(18.0)
        } else {
            (ring_step * 2.75).max(18.0)
        };
        let min_cov = if thin_part { 52 } else { 68 };
        for ids in result.triangles.as_chunks::<3>().0 {
            let a = mesh.vertices[ids[0]].v();
            let b = mesh.vertices[ids[1]].v();
            let c = mesh.vertices[ids[2]].v();
            let max_e = a.distance(b).max(a.distance(c)).max(b.distance(c));
            if max_e > max_edge {
                continue;
            }
            let min_aspect = if thin_part { 0.025 } else { 0.048 };
            let cross = ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs();
            if cross < min_aspect * max_e * max_e {
                continue;
            }
            let centroid = (a + b + c) / 3.0;
            let q1 = a * 0.5 + b * 0.25 + c * 0.25;
            let q2 = a * 0.25 + b * 0.5 + c * 0.25;
            let q3 = a * 0.25 + b * 0.25 + c * 0.5;
            if !valid(centroid) || !valid(q1) || !valid(q2) || !valid(q3) {
                continue;
            }
            let steps: usize = 4;
            let mut total = 0usize;
            let mut valid_samples = 0usize;
            for u in 0..=steps {
                for v in 0..=steps - u {
                    total += 1;
                    let p = a
                        + (b - a) * (u as f64 / steps as f64)
                        + (c - a) * (v as f64 / steps as f64);
                    if valid(p) {
                        valid_samples += 1;
                    }
                }
            }
            if valid_samples * 100 >= total * min_cov {
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

/// Returns the motion-proportional manual/fallback occlusion fill radius for a given part ID.
pub fn recommended_fill_radius(part_id: &str, lm: &Landmarks, w: usize, h: usize) -> usize {
    let m = part_overlap_margins(lm, w, h);
    let hf = HeadFrame::from_landmarks(lm, w, h);
    if part_id.starts_with("hair_back") {
        m.hair_back_underlap
    } else if part_id.starts_with("hair") {
        m.hair_side_front_seam.max(m.hair_hat_root)
    } else if part_id == "neck" {
        m.neck_collar_x.max(m.neck_collar_y).max(m.neck_chin_y)
    } else if part_id == "face" {
        m.face_scalp_margin
    } else if part_id == "body" || part_id.starts_with("clothes") || part_id.starts_with("hands") {
        m.clothes_hair_underlap
    } else if part_id.starts_with("eye") && (part_id.ends_with("_white") || part_id.ends_with("_iris")) {
        m.eye_gaze_x.max(m.eye_gaze_y)
    } else {
        ((hf.eye_dist.max(8.0) * 0.08).ceil() as usize).max(6)
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

pub fn deform(point: Point, id: &str, lm: &Landmarks, pose: &Pose) -> Point {
    deform_with_shapes(point, id, lm, pose, None)
}

pub fn deform_with_shapes(
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
        let is_left = id.starts_with("eyeL");
        let (eye, open, smile) = if is_left {
            (lm.eye_l, pose.eye_l, pose.eye_l_smile)
        } else {
            (lm.eye_r, pose.eye_r, pose.eye_r_smile)
        };
        let (eye_u, eye_v) = hf.to_local(eye.v());
        let shape = shape_deform.and_then(|s| if is_left { s.eye_l } else { s.eye_r });
        let is_iris = id.ends_with("_iris");
        let is_upper = id.ends_with("_upper");
        let is_lower = id.ends_with("_lower");
        if let Some(shape) = shape {
            pu += shape.offset_x * dist * 0.14;
            pv += shape.offset_y * dist * 0.12;
            let width_scale = if is_iris {
                1.0 + (shape.width - 1.0) * 0.45
            } else {
                shape.width
            };
            pu = eye_u + (pu - eye_u) * width_scale;
            let nx = (pu - eye_u) / (dist * 0.24).max(1.0);
            let center_arch = (1.0 - nx * nx).max(0.0);
            let eye_sign = if is_left { 1.0 } else { -1.0 };
            let outer_signed = (nx * eye_sign).clamp(-0.4, 1.2);
            pv -= shape.corner * dist * 0.045 * outer_signed;
            if !is_upper && !id.ends_with("_sub_lid") {
                let upper_mask =
                    ((eye_v + dist * 0.06 - pv) / (dist * 0.14).max(1.0)).clamp(0.0, 1.0);
                pv -= shape.upper_curve_delta * dist * 0.055 * center_arch * upper_mask;
            } else if id.ends_with("_sub_lid") {
                pv -= shape.upper_curve_delta * dist * 0.055 * center_arch;
            }
            let lower_mask = if is_lower {
                1.0
            } else {
                ((pv - (eye_v - dist * 0.02)) / (dist * 0.14).max(1.0)).clamp(0.0, 1.0)
            };
            pv += shape.lower_curve_delta * dist * 0.045 * center_arch * lower_mask;
            if !is_iris {
                pv = eye_v + (pv - eye_v) * (1.0 + shape.roundness_delta * 0.32 * center_arch);
            }
        }
        if !is_sub_layer {
            let close = (1.0 - open).clamp(0.0, 1.0);
            let widen = (open - 1.0).clamp(0.0, 0.3);
            let curve_delta = shape.map_or(0.0, |s| s.upper_curve_delta);
            let nx_c = (pu - eye_u) / (dist * 0.25).max(1.0);
            let arch = (1.0 - nx_c * nx_c).max(0.0);
            let pos_curve = curve_delta.max(0.0);
            let neg_curve = (-curve_delta).max(0.0);
            let seam_v = eye_v
                + dist * 0.022 * close
                - (smile * 0.032 + pos_curve * 0.028) * dist * close * arch
                + neg_curve * (0.045 * arch - 0.018 * (1.0 - arch)) * dist * close;
            if is_iris {
                pu += pose.eye_ball_x * dist * 0.045;
                pv -= pose.eye_ball_y * dist * 0.025;
                pv = seam_v + (pv - eye_v) * open.clamp(0.03, 1.3);
            } else if is_upper {
                // Bring real upper eyelid + lashes smoothly down to closing seam for both LD & SD
                // eyes while keeping the top eyebrow-gap apron anchored and bending smile (^)/sad (u).
                let lid_anchor_v = eye_v - dist * 0.45;
                let anchor_w = ((pv - lid_anchor_v) / (dist * 0.12).max(1.0)).clamp(0.0, 1.0);
                let orig_pv = pv;
                if close > 0.0 && orig_pv < seam_v {
                    let close_scale = (0.22
                        + (0.14 * smile + 0.12 * pos_curve - 0.03 * neg_curve).clamp(-0.03, 0.20)
                            * arch)
                        .clamp(0.20, 0.42);
                    let closed_v = seam_v + (orig_pv - seam_v) * close_scale;
                    pv += (closed_v - orig_pv) * close * anchor_w;
                } else if widen > 0.0 && orig_pv < seam_v {
                    pv -= (seam_v - orig_pv) * 0.28 * widen * anchor_w;
                }
                if smile != 0.0 || curve_delta != 0.0 {
                    let vert_w = anchor_w
                        * ((eye_v + dist * 0.16 - orig_pv) / (dist * 0.12).max(1.0)).clamp(0.0, 1.0);
                    let lift_k = (1.0 - close) * 0.045 + close * 0.018;
                    let smile_lift = smile * lift_k * dist * arch;
                    let curve_lift = pos_curve * lift_k * dist * arch;
                    let neg_drop = neg_curve
                        * ((0.040 + 0.055 * close) * arch - 0.025 * close * (1.0 - arch))
                        * dist;
                    pv -= (smile_lift + curve_lift - neg_drop) * vert_w;
                }
            } else if is_lower {
                let open_clamp = open.clamp(0.18, 1.3);
                pv = eye_v + (pv - eye_v) * open_clamp - smile * dist * 0.025 * arch;
            } else {
                // Keep sclera scaling smooth and uniform around `eye_v` so corner triangles never invert during extreme blink.
                let open_clamp = open.clamp(0.08, 1.3);
                pv = eye_v + (pv - eye_v) * open_clamp;
            }
        }
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
        ((hf.chin_v + hh * 0.42 - pv) / (hh * 0.42)).clamp(0., 1.)
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
    let has_pixels = mask.iter().any(|&a| a > 8);
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
            if posed_cross * rest_crosses[ti] <= 0.0 {
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
        for t in mesh.triangles.as_chunks::<3>().0 {
            let p =
                (mesh.vertices[t[0]].v() + mesh.vertices[t[1]].v() + mesh.vertices[t[2]].v()) / 3.;
            assert!(mask[p.y as usize * 32 + p.x as usize] > 0);
        }
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
        let close_y = l.eye_l.y + 16.0 * 0.04;
        assert!(d_top.y >= top_corner.y && d_top.y <= close_y + 1e-6);
        assert!(d_mid.y >= mid_lid.y && d_mid.y <= close_y + 1e-6);
        assert!((d_low.y - low_corner.y).abs() < 1e-6, "below close_y outer corner must not droop onto cheek: {d_low:?}");
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
        };
        let hf = HeadFrame::from_landmarks(&tilt_lm, 128, 128);
        let (eye_lu, eye_lv) = hf.to_local(DVec2::new(tilt_lm.eye_l.x, tilt_lm.eye_l.y));
        let tip = hf.to_world(eye_lu, eye_lv + 2.0);
        let corner = hf.to_world(eye_lu + 10.0, eye_lv + 2.0);

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
