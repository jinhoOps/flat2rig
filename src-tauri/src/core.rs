use base64::{engine::general_purpose::STANDARD, Engine};
use glam::DVec2;
use serde::{Deserialize, Serialize};

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
pub fn mesh_for(mask: &[u8], w: usize, h: usize, density: usize) -> Mesh {
    let mut left = w;
    let mut top = h;
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
    let mut mesh = Mesh {
        vertices: vec![],
        triangles: vec![],
        uv: vec![],
    };
    if left == w {
        return mesh;
    }
    let step =
        (((right - left + 1).max(bottom - top + 1)) as f64 / density.clamp(6, 32) as f64).max(3.);
    let valid = |p: DVec2| {
        p.x >= 0.
            && p.y >= 0.
            && p.x < w as f64
            && p.y < h as f64
            && mask[p.y as usize * w + p.x as usize] > 8
    };
    let mut y = top as f64;
    while y <= bottom as f64 {
        let mut x = left as f64;
        while x <= right as f64 {
            let p = DVec2::new(x, y);
            if valid(p) {
                mesh.vertices.push(p.into());
            } else {
                for delta in [
                    DVec2::new(step * 0.5, 0.),
                    DVec2::new(0., step * 0.5),
                    DVec2::splat(step * 0.5),
                ] {
                    if valid(p + delta) {
                        mesh.vertices.push((p + delta).into());
                    }
                }
            }
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
pub fn deform(point: Point, id: &str, lm: &Landmarks, pose: &Pose) -> Point {
    if id == "body" {
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
        p.x += pose.hair * hh * 0.04 * ((p.y - lm.crown.y) / hh).clamp(0., 1.8);
    }
    let pivot = DVec2::new(cx, lm.chin.y);
    let rot = glam::DMat2::from_angle(pose.angle_z.to_radians() * weight);
    (pivot + rot * (p - pivot)).into()
}
pub fn fill(data: &[u8], mask: &[u8], w: usize, h: usize, radius: usize) -> (Vec<u8>, Vec<u8>) {
    let mut out = data.to_vec();
    let mut expanded = mask.to_vec();
    let mut frontier = Vec::new();
    for i in 0..w * h {
        if mask[i] > 0 && neighbors(i, w, h).iter().any(|j| mask[*j] == 0) {
            frontier.push(i);
        }
    }
    for _ in 0..radius.min(64) {
        let mut next = Vec::new();
        for i in frontier {
            for j in neighbors(i, w, h) {
                if expanded[j] > 0 {
                    continue;
                }
                expanded[j] = 255;
                let pixel = [out[i * 4], out[i * 4 + 1], out[i * 4 + 2], 255];
                out[j * 4..j * 4 + 4].copy_from_slice(&pixel);
                next.push(j);
            }
        }
        frontier = next;
    }
    (out, expanded)
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
    fn fill_stays_inside_canvas() {
        let mut mask = vec![0; 9];
        mask[0] = 255;
        let (data, expanded) = fill(&[255; 36], &mask, 3, 3, 1);
        assert_eq!(expanded.iter().filter(|a| **a > 0).count(), 3);
        assert_eq!(data.len(), 36);
    }
}
