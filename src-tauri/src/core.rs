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
fn ellipse(p: DVec2, c: DVec2, rx: f64, ry: f64) -> bool {
    ((p - c) / DVec2::new(rx.max(1.), ry.max(1.))).length_squared() <= 1.
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
    let dist = lm.eye_l.v().distance(lm.eye_r.v()).max(w as f64 * 0.05);
    let ew = dist * 0.32;
    let eh = dist * 0.16;
    let cx = (lm.eye_l.x + lm.eye_r.x) / 2.;
    let hh = (lm.chin.y - lm.crown.y).max(h as f64 * 0.12);
    let center = DVec2::new(cx, lm.crown.y + hh * 0.55);
    for i in 0..w * h {
        if fg[i] == 0 {
            continue;
        }
        let p = DVec2::new((i % w) as f64, (i / w) as f64);
        let mut idx = 0;
        if p.y > lm.chin.y + hh * 0.22 {
            idx = 1;
        } else if p.y > lm.chin.y && (p.x - cx).abs() < dist * 1.02 * 0.42 {
            idx = 2;
        }
        if ellipse(p, center, dist * 1.02, hh * 0.49) {
            idx = 3;
        }
        if p.y < lm.eye_l.y - hh * 0.22 && ellipse(p, center, dist * 1.02 * 1.1, hh * 0.49 * 1.1) {
            idx = 17;
        }
        for (side, eye) in [(0, lm.eye_r.v()), (1, lm.eye_l.v())] {
            if ellipse(p, eye - DVec2::new(0., eh * 2.1), ew * 1.02, eh * 0.3) {
                idx = 12 + side;
            }
            if ellipse(p, eye, ew, eh) {
                idx = 4 + side * 4;
                if ellipse(p, eye, ew * 0.46, eh * 0.95) {
                    idx += 1;
                }
                if p.y < eye.y - eh * 0.55 {
                    idx = 7 + side * 4;
                }
                if p.y > eye.y + eh * 0.65 {
                    idx = 6 + side * 4;
                }
            }
        }
        if ellipse(p, lm.mouth.v(), dist * 0.18, dist * 0.065) {
            idx = if p.y < lm.mouth.y - dist * 0.018 {
                16
            } else if p.y > lm.mouth.y + dist * 0.018 {
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
    mesh_for_with_anchors(
        mask,
        w,
        h,
        density,
        &[lm.eye_l, lm.eye_r, lm.mouth, lm.chin, lm.crown],
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
    let mut mesh = Mesh {
        vertices: Vec::new(),
        triangles: Vec::new(),
        uv: Vec::new(),
    };
    let Some((left, right, top, bottom)) = mask_bounds(mask, w) else {
        return mesh;
    };
    let span = (right - left + 1).max(bottom - top + 1) as f64;
    let step = (span / density.clamp(6, 32) as f64).max(3.);
    let contour_min_dist_sq = (step * 0.45).max(2.0).powi(2);
    let valid = |p: DVec2| {
        let px = p.x + 1e-6;
        let py = p.y + 1e-6;
        px >= 0.
            && py >= 0.
            && (px as usize) < w
            && (py as usize) < h
            && mask[py as usize * w + px as usize] > 8
    };

    let push_if_far = |verts: &mut Vec<Point>, cand: DVec2, min_sq: f64| {
        if !valid(cand) {
            return;
        }
        let too_close = verts.iter().any(|v| {
            let dx = v.x - cand.x;
            let dy = v.y - cand.y;
            dx * dx + dy * dy < min_sq
        });
        if !too_close {
            verts.push(cand.into());
        }
    };

    // 1. Sample contour boundary pixels first so silhouette shapes are never clipped.
    for y in top..=bottom {
        let row = y * w;
        for x in left..=right {
            if mask[row + x] <= 8 {
                continue;
            }
            let is_boundary = x == 0
                || y == 0
                || x + 1 >= w
                || y + 1 >= h
                || mask[row + x - 1] <= 8
                || mask[row + x + 1] <= 8
                || mask[(y - 1) * w + x] <= 8
                || mask[(y + 1) * w + x] <= 8;
            if is_boundary {
                push_if_far(
                    &mut mesh.vertices,
                    DVec2::new(x as f64, y as f64),
                    contour_min_dist_sq,
                );
            }
        }
    }

    // 2. Inject interior landmark anchors strictly inside active mask.
    for a in anchors {
        push_if_far(&mut mesh.vertices, a.v(), contour_min_dist_sq * 0.5);
    }

    // 3. Fill interior uniform grid points without duplicating contour vertices.
    let grid_min_dist_sq = (step * 0.35).max(1.5).powi(2);
    let mut y = top as f64;
    while y <= bottom as f64 {
        let mut x = left as f64;
        while x <= right as f64 {
            push_if_far(&mut mesh.vertices, DVec2::new(x, y), grid_min_dist_sq);
            x += step;
        }
        y += step;
    }

    if mesh.vertices.len() >= 3 {
        let points: Vec<_> = mesh
            .vertices
            .iter()
            .map(|p| delaunator::Point { x: p.x, y: p.y })
            .collect();
        let result = delaunator::triangulate(&points);
        for ids in result.triangles.as_chunks::<3>().0 {
            let a = mesh.vertices[ids[0]].v();
            let b = mesh.vertices[ids[1]].v();
            let c = mesh.vertices[ids[2]].v();
            let steps = a
                .distance(b)
                .max(a.distance(c))
                .max(b.distance(c))
                .ceil()
                .max(2.) as usize;
            let mut okay = true;
            'outer: for u in 0..=steps {
                for v in 0..=steps - u {
                    let p = a
                        + (b - a) * (u as f64 / steps as f64)
                        + (c - a) * (v as f64 / steps as f64);
                    if !valid(p) {
                        okay = false;
                        break 'outer;
                    }
                }
            }
            if okay {
                mesh.triangles.extend_from_slice(ids);
            }
        }
    }
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
#[serde(rename_all = "camelCase")]
pub struct Pose {
    pub angle_x: f64,
    pub angle_y: f64,
    pub angle_z: f64,
    pub eye_l: f64,
    pub eye_r: f64,
    pub mouth: f64,
    pub hair: f64,
}
impl Default for Pose {
    fn default() -> Self {
        Self {
            angle_x: 0.,
            angle_y: 0.,
            angle_z: 0.,
            eye_l: 1.,
            eye_r: 1.,
            mouth: 0.,
            hair: 0.,
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

pub fn deform(point: Point, id: &str, lm: &Landmarks, pose: &Pose) -> Point {
    if id == "body" || id.starts_with("clothes") {
        return point;
    }
    let mut p = point.v();
    let hh = (lm.chin.y - lm.crown.y).max(1.);
    let cx = (lm.eye_l.x + lm.eye_r.x) / 2.;
    if id.starts_with("eye") {
        let (eye, open) = if id.starts_with("eyeL") {
            (lm.eye_l, pose.eye_l)
        } else {
            (lm.eye_r, pose.eye_r)
        };
        p.y = eye.y + (p.y - eye.y) * open.max(0.04);
    }
    if id == "mouth_upper" {
        p.y -= pose.mouth * hh * 0.012;
    }
    if id == "mouth_lower" {
        p.y += pose.mouth * hh * 0.045;
    }
    if id == "mouth_inner" {
        p.y = lm.mouth.y + (p.y - lm.mouth.y) * (1. + pose.mouth * 2.);
    }
    let weight = if id == "neck" {
        ((lm.chin.y + hh * 0.25 - p.y) / (hh * 0.25)).clamp(0., 1.)
    } else {
        1.
    };
    p.x += pose.angle_x * hh * 0.0025 * weight * (1. - (p.y - lm.eye_l.y).abs() / (hh * 2.));
    p.y += pose.angle_y * hh * 0.0018 * weight;
    if id.starts_with("hair") {
        let (amp, root_offset) = hair_sway_profile(id);
        let root_y = lm.crown.y + hh * root_offset;
        p.x += pose.hair * hh * 0.04 * amp * ((p.y - root_y) / hh).clamp(0., 1.8);
    }
    let pivot = DVec2::new(cx, lm.chin.y);
    let rot = glam::DMat2::from_angle(pose.angle_z.to_radians() * weight);
    (pivot + rot * (p - pivot)).into()
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
    for _ in 0..radius.min(128) {
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
        Pose { angle_x: -30., angle_y: 30., angle_z: 15., ..Pose::default() },
        Pose { angle_x: 30., angle_y: -30., angle_z: -15., ..Pose::default() },
        Pose { eye_l: 0., eye_r: 0., ..Pose::default() },
        Pose { eye_l: 1.2, eye_r: 1.2, ..Pose::default() },
        Pose { mouth: 0., ..Pose::default() },
        Pose { mouth: 1., ..Pose::default() },
        Pose { hair: -1., ..Pose::default() },
        Pose { hair: 1., ..Pose::default() },
        Pose { angle_x: 30., angle_y: -30., angle_z: 30., eye_l: 0., eye_r: 0., mouth: 1., hair: -1. },
        Pose { angle_x: -30., angle_y: 30., angle_z: -30., eye_l: 0., eye_r: 0., mouth: 1., hair: 1. },
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
}
