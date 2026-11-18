//! B4 shared CPU mesh renderer — the same rules as the app WebGL compositor
//! (`src/rig/glRenderer.ts`), so CLI motion previews, `poses/` renders and every evaluation image
//! show what the app shows.
//!
//! Rules (each mirrors one line of `glRenderer.ts`):
//! - **Texture**: straight RGBA8 is premultiplied on upload with 8-bit rounding
//!   (`UNPACK_PREMULTIPLY_ALPHA_WEBGL`), sampled **bilinearly** (`LINEAR`) at texel centres
//!   `i + 0.5`, **clamped to the edge** (`CLAMP_TO_EDGE`).
//! - **Geometry**: a mesh vertex carries its rest position as the texture coordinate (`a_src`) and
//!   its posed position as the canvas position (`a_dst`); attributes are interpolated linearly.
//!   A layer without posed points is drawn as the full-canvas quad (the app's rest path).
//! - **Coverage**: one sample at each pixel centre, vertices snapped to a 1/256 px grid and a
//!   **top-left fill rule** with exact integer edge functions, so a pixel centre on an edge shared by
//!   two triangles is drawn exactly once (no double compositing, no crack).
//! - **Blending**: premultiplied `ONE, ONE_MINUS_SRC_ALPHA` (`normal`) or `ONE, ONE` (`additive`),
//!   with the layer opacity (`u_alpha`) multiplied into the premultiplied colour.
//! - **Clipping**: a clip child is multiplied by the alpha of its target rendered alone at opacity 1
//!   (`u_mask`, the sclera FBO), sampled at the same pixel.
//!
//! The canvas accumulates in `f32` premultiplied RGBA; [`Canvas::to_straight_rgba8`] converts once at
//! the end (WebGL rounds to 8 bits after every draw, so the two differ by at most a level or two).

use crate::core::{self, Landmarks, Mesh, Point, Pose};

/// Sub-pixel grid of the coverage test (GPU rasterizers snap to 1/256 px or finer).
const SUBPIXEL: f64 = 256.0;

/// Premultiplied RGBA in `0..=1`.
pub type Premul = [f32; 4];

/// Layer blend modes the WebGL path draws (`multiply` falls back to Canvas2D in the app and is not
/// supported here — callers must not silently turn it into `normal`, AGENTS.md 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    Normal,
    Additive,
}

/// A premultiplied texture uploaded like `uploadTexture`.
#[derive(Clone, Debug)]
pub struct Texture {
    w: usize,
    h: usize,
    texels: Vec<Premul>,
}

impl Texture {
    /// Straight (non-premultiplied) RGBA8, row-major `w × h`. Premultiplies with 8-bit rounding.
    pub fn from_straight_rgba8(w: usize, h: usize, rgba: &[u8]) -> Self {
        assert_eq!(rgba.len(), w * h * 4, "texture size mismatch");
        let texels = rgba
            .as_chunks::<4>().0.iter()
            .map(|p| {
                let a = p[3] as u32;
                let pm = |c: u8| ((c as u32 * a + 127) / 255) as f32 / 255.0;
                [pm(p[0]), pm(p[1]), pm(p[2]), a as f32 / 255.0]
            })
            .collect();
        Self { w, h, texels }
    }

    pub fn width(&self) -> usize {
        self.w
    }

    pub fn height(&self) -> usize {
        self.h
    }

    /// Bilinear sample at texture position `(x, y)` in texel units (texel `i` spans `i..i+1`, its
    /// centre is `i + 0.5`), clamped to the edge.
    pub fn sample_bilinear(&self, x: f64, y: f64) -> Premul {
        if self.w == 0 || self.h == 0 {
            return [0.0; 4];
        }
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0f = fx.floor();
        let y0f = fy.floor();
        let tx = (fx - x0f) as f32;
        let ty = (fy - y0f) as f32;
        let clamp_x = |v: f64| v.clamp(0.0, (self.w - 1) as f64) as usize;
        let clamp_y = |v: f64| v.clamp(0.0, (self.h - 1) as f64) as usize;
        let (x0, x1) = (clamp_x(x0f), clamp_x(x0f + 1.0));
        let (y0, y1) = (clamp_y(y0f), clamp_y(y0f + 1.0));
        let t00 = self.texels[y0 * self.w + x0];
        let t10 = self.texels[y0 * self.w + x1];
        let t01 = self.texels[y1 * self.w + x0];
        let t11 = self.texels[y1 * self.w + x1];
        let mut out = [0.0f32; 4];
        for c in 0..4 {
            let top = t00[c] + (t10[c] - t00[c]) * tx;
            let bot = t01[c] + (t11[c] - t01[c]) * tx;
            out[c] = top + (bot - top) * ty;
        }
        out
    }
}

/// A premultiplied `f32` render target (the WebGL drawing buffer).
#[derive(Clone, Debug)]
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub px: Vec<Premul>,
}

impl Canvas {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, px: vec![[0.0; 4]; w * h] }
    }

    pub fn alpha(&self, i: usize) -> f32 {
        self.px[i][3]
    }

    /// Straight RGBA8 (what `getImageData` / a saved PNG holds).
    pub fn to_straight_rgba8(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.px.len() * 4];
        for (o, p) in out.as_chunks_mut::<4>().0.iter_mut().zip(&self.px) {
            let a = p[3].clamp(0.0, 1.0);
            let a8 = (a * 255.0).round();
            if a8 <= 0.0 {
                continue;
            }
            for c in 0..3 {
                o[c] = (p[c] / a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            o[3] = a8 as u8;
        }
        out
    }

    pub fn to_image(&self) -> image::RgbaImage {
        image::RgbaImage::from_raw(self.w as u32, self.h as u32, self.to_straight_rgba8())
            .expect("canvas buffer matches its size")
    }

    fn blend(&mut self, i: usize, src: Premul, blend: Blend) {
        let d = &mut self.px[i];
        match blend {
            Blend::Normal => {
                let inv = 1.0 - src[3];
                for c in 0..4 {
                    d[c] = src[c] + d[c] * inv;
                }
            }
            Blend::Additive => {
                for c in 0..4 {
                    d[c] = (src[c] + d[c]).min(1.0);
                }
            }
        }
    }
}

fn snap(p: Point) -> (i64, i64) {
    ((p.x * SUBPIXEL).round() as i64, (p.y * SUBPIXEL).round() as i64)
}

/// Edge function of `p` against the directed edge `a → b` (positive on the interior side once the
/// triangle is oriented to positive area).
fn edge(a: (i64, i64), b: (i64, i64), p: (i64, i64)) -> i64 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

/// Top-left rule for the directed edge `a → b` of a positive-area triangle (y down): a top edge is
/// horizontal with the interior below (`dx > 0`), a left edge goes up (`dy < 0`). For any edge
/// exactly one of `a → b` and `b → a` qualifies, so a shared edge is owned by one triangle.
fn is_top_left(a: (i64, i64), b: (i64, i64)) -> bool {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    (dy == 0 && dx > 0) || dy < 0
}

/// Calls `f(x, y, [b0, b1, b2])` for every pixel whose centre `(x + 0.5, y + 0.5)` the triangle
/// `dst` covers under the top-left rule, with the barycentric weights of that centre. Pixels outside
/// `w × h` are skipped. Degenerate triangles draw nothing.
pub fn rasterize_triangle(dst: [Point; 3], w: usize, h: usize, mut f: impl FnMut(usize, usize, [f64; 3])) {
    if w == 0 || h == 0 || dst.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return;
    }
    let mut v = [snap(dst[0]), snap(dst[1]), snap(dst[2])];
    let mut order = [0usize, 1, 2];
    let mut area = edge(v[0], v[1], v[2]);
    if area == 0 {
        return;
    }
    if area < 0 {
        v.swap(1, 2);
        order.swap(1, 2);
        area = -area;
    }
    let sub = SUBPIXEL as i64;
    let min_x = v.iter().map(|p| p.0).min().unwrap();
    let max_x = v.iter().map(|p| p.0).max().unwrap();
    let min_y = v.iter().map(|p| p.1).min().unwrap();
    let max_y = v.iter().map(|p| p.1).max().unwrap();
    // Pixel x covers centre (2x + 1)·sub/2; range of x whose centre may lie inside [min, max].
    let px_lo = |m: i64| ((m - sub / 2).div_euclid(sub)).max(0);
    let px_hi = |m: i64, lim: usize| ((m - sub / 2).div_euclid(sub) + 1).min(lim as i64 - 1);
    let (x0, x1) = (px_lo(min_x), px_hi(max_x, w));
    let (y0, y1) = (px_lo(min_y), px_hi(max_y, h));
    if x0 > x1 || y0 > y1 {
        return;
    }
    // Edge i is opposite vertex i: e0 = (v1 → v2), e1 = (v2 → v0), e2 = (v0 → v1).
    let edges = [(v[1], v[2]), (v[2], v[0]), (v[0], v[1])];
    let bias: [i64; 3] = std::array::from_fn(|k| if is_top_left(edges[k].0, edges[k].1) { 0 } else { -1 });
    let inv_area = 1.0 / area as f64;
    for y in y0..=y1 {
        let cy = y * sub + sub / 2;
        for x in x0..=x1 {
            let c = (x * sub + sub / 2, cy);
            let e = [edge(edges[0].0, edges[0].1, c), edge(edges[1].0, edges[1].1, c), edge(edges[2].0, edges[2].1, c)];
            // Inside: e > 0, or e == 0 on a top-left edge (integer exact: e + bias >= 0 ⇔ that).
            if e[0] + bias[0] < 0 || e[1] + bias[1] < 0 || e[2] + bias[2] < 0 {
                continue;
            }
            let mut b = [0.0f64; 3];
            for k in 0..3 {
                b[order[k]] = e[k] as f64 * inv_area;
            }
            f(x as usize, y as usize, b);
        }
    }
}

/// How one layer is drawn.
#[derive(Clone, Copy, Debug)]
pub struct DrawParams<'a> {
    /// Layer opacity (`u_alpha`).
    pub opacity: f32,
    pub blend: Blend,
    /// Clip mask (`u_mask`): the target layer rendered alone; its alpha multiplies the colour.
    pub clip: Option<&'a Canvas>,
}

impl Default for DrawParams<'_> {
    fn default() -> Self {
        Self { opacity: 1.0, blend: Blend::Normal, clip: None }
    }
}

/// Draws `triangles` (index triples) with texture coordinates `src` (rest positions, texel units)
/// at canvas positions `dst` (posed positions).
pub fn draw_mesh(canvas: &mut Canvas, tex: &Texture, src: &[Point], dst: &[Point], triangles: &[usize], params: DrawParams) {
    assert_eq!(src.len(), dst.len(), "src / dst vertex count mismatch");
    let (w, h) = (canvas.w, canvas.h);
    for tri in triangles.as_chunks::<3>().0.iter() {
        let s = [src[tri[0]], src[tri[1]], src[tri[2]]];
        let d = [dst[tri[0]], dst[tri[1]], dst[tri[2]]];
        rasterize_triangle(d, w, h, |x, y, b| {
            let sx = b[0] * s[0].x + b[1] * s[1].x + b[2] * s[2].x;
            let sy = b[0] * s[0].y + b[1] * s[1].y + b[2] * s[2].y;
            let i = y * w + x;
            let mut c = tex.sample_bilinear(sx, sy);
            let mut k = params.opacity;
            if let Some(clip) = params.clip {
                k *= clip.alpha(i);
            }
            if k != 1.0 {
                for v in &mut c {
                    *v *= k;
                }
            }
            if c[3] > 0.0 || c[0] > 0.0 || c[1] > 0.0 || c[2] > 0.0 {
                canvas.blend(i, c, params.blend);
            }
        });
    }
}

/// The full-canvas quad (`uploadGeometry` without posed points): texture `w × h` stretched over the
/// canvas. With equal sizes every pixel samples its own texel centre (exact copy).
pub fn draw_full(canvas: &mut Canvas, tex: &Texture, params: DrawParams) {
    let (cw, ch) = (canvas.w as f64, canvas.h as f64);
    let (tw, th) = (tex.w as f64, tex.h as f64);
    let p = |x: f64, y: f64| Point { x, y };
    let dst = [p(0.0, 0.0), p(cw, 0.0), p(cw, ch), p(0.0, ch)];
    let src = [p(0.0, 0.0), p(tw, 0.0), p(tw, th), p(0.0, th)];
    draw_mesh(canvas, tex, &src, &dst, &[0, 1, 2, 0, 2, 3], params);
}

/// One layer of a frame, bottom → top order in [`render_layers`].
#[derive(Clone, Copy, Debug)]
pub struct Layer<'a> {
    pub texture: &'a Texture,
    pub mesh: &'a Mesh,
    /// Posed vertex positions (same length as `mesh.vertices`); `None` draws the full-canvas quad.
    pub posed: Option<&'a [Point]>,
    pub opacity: f32,
    pub blend: Blend,
    /// Index of a layer below this one whose alpha clips it (`clippingTargets`).
    pub clip_to: Option<usize>,
}

/// Renders a frame like `GlMeshCompositor.render` with clipping active: layers bottom → top; a clip
/// target is additionally rendered alone (opacity 1) and its alpha masks every child drawn later.
pub fn render_layers(w: usize, h: usize, layers: &[Layer]) -> Canvas {
    let mut out = Canvas::new(w, h);
    let targets: std::collections::HashSet<usize> = layers.iter().filter_map(|l| l.clip_to).collect();
    let mut clip_canvases: std::collections::HashMap<usize, Canvas> = std::collections::HashMap::new();
    for (li, layer) in layers.iter().enumerate() {
        if layer.opacity <= 0.01 {
            continue;
        }
        let draw = |canvas: &mut Canvas, params: DrawParams| match layer.posed {
            Some(posed) if !layer.mesh.triangles.is_empty() => {
                draw_mesh(canvas, layer.texture, &layer.mesh.vertices, posed, &layer.mesh.triangles, params)
            }
            _ => draw_full(canvas, layer.texture, params),
        };
        let clip = layer.clip_to.and_then(|t| clip_canvases.get(&t));
        draw(&mut out, DrawParams { opacity: layer.opacity, blend: layer.blend, clip });
        if targets.contains(&li) {
            let mut alone = Canvas::new(w, h);
            draw(&mut alone, DrawParams::default());
            clip_canvases.insert(li, alone);
        }
    }
    out
}

/// Posed positions of a mesh's vertices (`core::deform`, the same call the WebGL path makes per
/// vertex in `computePosedPoints`).
pub fn posed_vertices(mesh: &Mesh, id: &str, lm: &Landmarks, pose: &Pose) -> Vec<Point> {
    mesh.vertices.iter().map(|&v| core::deform(v, id, lm, pose)).collect()
}

/// Renders one pose of a rig like the app's playback frame (`renderCharacterFrame` → WebGL):
/// layers bottom → top, each posed with `core::deform`; a layer whose vertices move by at most
/// `1e-3` px (sum of |dx| + |dy| per vertex, `computePosedPoints`) is drawn as the full-canvas quad;
/// per-frame opacity from [`layer_opacity`]; clipping by `clip` (`exposure::default_clip_targets`).
#[allow(clippy::too_many_arguments)]
pub fn render_pose(
    ids: &[String],
    meshes: &[Mesh],
    textures: &[Texture],
    clip: &[Option<usize>],
    lm: &Landmarks,
    pose: &Pose,
    w: usize,
    h: usize,
) -> Canvas {
    let posed: Vec<Option<Vec<Point>>> = ids
        .iter()
        .zip(meshes)
        .map(|(id, mesh)| {
            if mesh.triangles.is_empty() {
                return None;
            }
            let pts = posed_vertices(mesh, id, lm, pose);
            let max_delta = pts
                .iter()
                .zip(&mesh.vertices)
                .map(|(d, s)| (d.x - s.x).abs() + (d.y - s.y).abs())
                .fold(0.0f64, f64::max);
            (max_delta > 1e-3).then_some(pts)
        })
        .collect();
    let layers: Vec<Layer> = (0..ids.len())
        .map(|i| Layer {
            texture: &textures[i],
            mesh: &meshes[i],
            posed: posed[i].as_deref(),
            opacity: layer_opacity(&ids[i], pose),
            blend: Blend::Normal,
            clip_to: clip.get(i).copied().flatten(),
        })
        .collect();
    render_layers(w, h, &layers)
}

/// Layer opacity the app applies per frame (`evaluate.ts` `computeEyeAlphas` / `mouthInteriorAlpha`
/// with no legacy shape assets and full interior visibility): iris, white and lower lid fade out as
/// the eye closes, and the mouth interior fades in as the mouth opens.
pub fn layer_opacity(id: &str, pose: &Pose) -> f32 {
    // Exact ids, like `renderCharacterFrame` (numbered extra layers keep opacity 1 there too).
    let ramp = |v: f64, lo: f64, span: f64| if v <= lo { 0.0 } else { ((v - lo) / span).clamp(0.0, 1.0) };
    let v = match id {
        "mouth_inner" => ramp(pose.mouth, 0.03, 0.12),
        "eyeL_iris" => ramp(pose.eye_l, 0.06, 0.14),
        "eyeR_iris" => ramp(pose.eye_r, 0.06, 0.14),
        "eyeL_white" => ramp(pose.eye_l, 0.03, 0.11),
        "eyeR_white" => ramp(pose.eye_r, 0.03, 0.11),
        "eyeL_lower" => ramp(pose.eye_l, 0.15, 0.35),
        "eyeR_lower" => ramp(pose.eye_r, 0.15, 0.35),
        _ => 1.0,
    };
    v as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    fn solid(w: usize, h: usize, rgba: [u8; 4]) -> Texture {
        Texture::from_straight_rgba8(w, h, &rgba.repeat(w * h))
    }

    /// Two triangles sharing a diagonal, half-transparent solid colour: every covered pixel —
    /// including those whose centre lies exactly on the shared edge — gets the interior alpha once.
    #[test]
    fn shared_edge_pixels_are_composited_once() {
        let (w, h) = (16usize, 16usize);
        let tex = solid(w, h, [200, 40, 20, 128]);
        // Vertices on pixel centres, so the shared diagonal passes exactly through the centres
        // (x + 0.5, x + 0.5) — the worst case for double compositing.
        let verts = [p(1.5, 1.5), p(14.5, 1.5), p(14.5, 14.5), p(1.5, 14.5)];
        let tris = [0usize, 1, 2, 0, 2, 3];
        let mut c = Canvas::new(w, h);
        draw_mesh(&mut c, &tex, &verts, &verts, &tris, DrawParams::default());
        let out = c.to_straight_rgba8();
        let interior = out[(8 * w + 5) * 4 + 3];
        assert_eq!(interior, 128);
        let mut diag = 0;
        for y in 0..h {
            for x in 0..w {
                let a = out[(y * w + x) * 4 + 3];
                assert!(a == 0 || a == interior, "pixel ({x},{y}) alpha {a} ≠ {interior}");
                if x == y && (2..14).contains(&x) {
                    assert_eq!(a, interior, "diagonal pixel ({x},{y}) not drawn exactly once");
                    diag += 1;
                }
            }
        }
        assert_eq!(diag, 12);
        // Coverage of the 13 × 13 square of pixel centres 1.5..=14.5 under the top-left rule:
        // left and top edges included, right and bottom excluded → 13 × 13 pixels.
        let drawn = out.as_chunks::<4>().0.iter().filter(|p| p[3] > 0).count();
        assert_eq!(drawn, 13 * 13);
    }

    /// A fan of triangles around a centre vertex sitting on a pixel centre, arbitrary angles: the
    /// union is drawn exactly once per pixel (no gaps, no overlaps).
    #[test]
    fn triangle_fan_partitions_pixels() {
        let (w, h) = (40usize, 40usize);
        let centre = p(20.5, 20.5);
        let mut verts = vec![centre];
        let n = 11;
        for k in 0..n {
            let a = k as f64 / n as f64 * std::f64::consts::TAU + 0.3;
            verts.push(p(20.5 + 15.25 * a.cos(), 20.5 + 15.25 * a.sin()));
        }
        let mut tris = Vec::new();
        for k in 0..n {
            tris.extend_from_slice(&[0, 1 + k, 1 + (k + 1) % n]);
        }
        let mut count = vec![0u32; w * h];
        for t in tris.as_chunks::<3>().0.iter() {
            rasterize_triangle([verts[t[0]], verts[t[1]], verts[t[2]]], w, h, |x, y, _| count[y * w + x] += 1);
        }
        assert!(count.iter().all(|&c| c <= 1), "a pixel was covered twice");
        assert_eq!(count[20 * w + 20], 1, "the shared centre vertex pixel must be drawn once");
        // The polygon interior (well inside the inscribed circle) has no holes.
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f64 + 0.5 - 20.5).powi(2) + (y as f64 + 0.5 - 20.5).powi(2)).sqrt();
                if d < 13.0 {
                    assert_eq!(count[y * w + x], 1, "hole at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn bilinear_sampling_matches_hand_computation_and_clamps() {
        // 2 × 1 texture: opaque black | opaque white, plus a transparent texel row below.
        let rgba = [0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 0];
        let tex = Texture::from_straight_rgba8(2, 2, &rgba);
        // Texel centres return the texel exactly.
        assert_eq!(tex.sample_bilinear(0.5, 0.5), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(tex.sample_bilinear(1.5, 0.5), [1.0, 1.0, 1.0, 1.0]);
        // Halfway between the two centres: 50 % grey, opaque.
        let mid = tex.sample_bilinear(1.0, 0.5);
        assert!((mid[0] - 0.5).abs() < 1e-6 && (mid[3] - 1.0).abs() < 1e-6);
        // A quarter of the way: 25 %.
        let q = tex.sample_bilinear(0.75, 0.5);
        assert!((q[0] - 0.25).abs() < 1e-6);
        // Towards the transparent row: premultiplied, so colour fades with alpha (no dark fringe in
        // straight colour: straight = premul / alpha stays white on the right column).
        let down = tex.sample_bilinear(1.5, 1.0);
        assert!((down[3] - 0.5).abs() < 1e-6 && (down[0] / down[3] - 1.0).abs() < 1e-6);
        // Clamp to edge: beyond the border returns the border texel.
        assert_eq!(tex.sample_bilinear(-3.0, 0.5), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(tex.sample_bilinear(9.0, 0.0), [1.0, 1.0, 1.0, 1.0]);
    }

    /// Identity mesh over a texture reproduces it bit for bit, and so does the full-canvas quad.
    #[test]
    fn rest_mesh_and_full_quad_copy_the_texture_exactly() {
        let (w, h) = (9usize, 7usize);
        let mut rgba = vec![0u8; w * h * 4];
        for (i, px) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            px.copy_from_slice(&[(i * 37 % 256) as u8, (i * 91 % 256) as u8, (i * 13 % 256) as u8, (i * 59 % 256) as u8]);
        }
        let tex = Texture::from_straight_rgba8(w, h, &rgba);
        let verts = [p(0.0, 0.0), p(w as f64, 0.0), p(w as f64, h as f64), p(0.0, h as f64)];
        let mut a = Canvas::new(w, h);
        draw_mesh(&mut a, &tex, &verts, &verts, &[0, 1, 2, 0, 2, 3], DrawParams::default());
        let mut b = Canvas::new(w, h);
        draw_full(&mut b, &tex, DrawParams::default());
        let (oa, ob) = (a.to_straight_rgba8(), b.to_straight_rgba8());
        assert_eq!(oa, ob);
        for (o, s) in oa.as_chunks::<4>().0.iter().zip(rgba.as_chunks::<4>().0.iter()) {
            assert_eq!(o[3], s[3]);
            if s[3] >= 16 {
                // Premultiplied 8-bit upload loses a little colour precision at low alpha only.
                for c in 0..3 {
                    assert!((o[c] as i32 - s[c] as i32).abs() <= 8, "{o:?} vs {s:?}");
                }
            }
        }
    }

    #[test]
    fn normal_blend_is_premultiplied_over_and_clip_masks_by_target_alpha() {
        let (w, h) = (4usize, 4usize);
        let quad = Mesh {
            vertices: vec![p(0.0, 0.0), p(4.0, 0.0), p(4.0, 4.0), p(0.0, 4.0)],
            triangles: vec![0, 1, 2, 0, 2, 3],
            uv: vec![],
        };
        // Target covers the left half only.
        let mut half = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..2 {
                half[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let white = Texture::from_straight_rgba8(w, h, &half);
        let red = solid(w, h, [255, 0, 0, 128]);
        let posed = quad.vertices.clone();
        let layers = [
            Layer { texture: &white, mesh: &quad, posed: Some(&posed), opacity: 1.0, blend: Blend::Normal, clip_to: None },
            Layer { texture: &red, mesh: &quad, posed: Some(&posed), opacity: 1.0, blend: Blend::Normal, clip_to: Some(0) },
        ];
        let out = render_layers(w, h, &layers).to_straight_rgba8();
        // Left: half-red over white → (255, 127, 127, 255). Right: clipped away → transparent.
        let l = &out[(w + 1) * 4..(w + 1) * 4 + 4];
        assert_eq!(l[3], 255);
        assert_eq!(l[0], 255);
        assert!((l[1] as i32 - 127).abs() <= 1, "{l:?}");
        let r = &out[(w + 3) * 4..(w + 3) * 4 + 4];
        assert_eq!(r, &[0, 0, 0, 0]);
    }

    #[test]
    fn layer_opacity_follows_the_app_eye_and_mouth_ramps() {
        let open = Pose::default();
        assert_eq!(layer_opacity("eyeL_iris", &open), 1.0);
        assert_eq!(layer_opacity("face", &open), 1.0);
        assert_eq!(layer_opacity("mouth_inner", &open), 0.0);
        let closed = Pose { eye_l: 0.0, mouth: 1.0, ..Pose::default() };
        assert_eq!(layer_opacity("eyeL_iris", &closed), 0.0);
        assert_eq!(layer_opacity("eyeL_white", &closed), 0.0);
        assert_eq!(layer_opacity("eyeL_lower", &closed), 0.0);
        assert_eq!(layer_opacity("eyeL_upper", &closed), 1.0);
        assert_eq!(layer_opacity("eyeR_iris", &closed), 1.0);
        assert_eq!(layer_opacity("mouth_inner", &closed), 1.0);
        let half = Pose { eye_r: 0.13, ..Pose::default() };
        assert!((layer_opacity("eyeR_iris", &half) - 0.5).abs() < 1e-6);
    }
}
