import Delaunator from "delaunator";
import {
  PART_DEFS,
  type Landmarks,
  type Mesh,
  type Parameters,
  type Part,
  type Point,
} from "./types";

export const clamp = (v: number, a: number, b: number) =>
  Math.max(a, Math.min(b, v));

/** Cover every texel in preview; transparent pixels keep holes transparent.
 * Exported topology remains the boundary-validated mesh from meshFor/Rust.
 */
export function textureMesh(mask: Uint8Array, w: number, h: number): Mesh {
  const box = bounds(mask, w, h);
  if (!box) return { vertices: [], triangles: [], uv: [] };
  const steps = 8,
    vertices: Point[] = [],
    triangles: number[] = [];
  for (let y = 0; y <= steps; y++)
    for (let x = 0; x <= steps; x++)
      vertices.push({
        x: box.left + (box.width * x) / steps,
        y: box.top + (box.height * y) / steps,
      });
  for (let y = 0; y < steps; y++)
    for (let x = 0; x < steps; x++) {
      const a = y * (steps + 1) + x,
        b = a + 1,
        c = a + steps + 1,
        d = c + 1;
      triangles.push(a, b, c, b, d, c);
    }
  return {
    vertices,
    triangles,
    uv: vertices.map((p) => ({ x: p.x / w, y: p.y / h })),
  };
}
export function defaultLandmarks(w: number, h: number): Landmarks {
  return {
    eyeR: { x: w * 0.39, y: h * 0.31 },
    eyeL: { x: w * 0.61, y: h * 0.31 },
    mouth: { x: w * 0.5, y: h * 0.415 },
    chin: { x: w * 0.5, y: h * 0.48 },
    crown: { x: w * 0.5, y: h * 0.08 },
  };
}
export function faceBoxLandmarks(a: Point, b: Point): Landmarks {
  const x = Math.min(a.x, b.x),
    y = Math.min(a.y, b.y),
    w = Math.abs(a.x - b.x),
    h = Math.abs(a.y - b.y);
  return {
    eyeR: { x: x + w * 0.28, y: y + h * 0.52 },
    eyeL: { x: x + w * 0.72, y: y + h * 0.52 },
    mouth: { x: x + w * 0.5, y: y + h * 0.81 },
    chin: { x: x + w * 0.5, y: y + h },
    crown: { x: x + w * 0.5, y },
  };
}
/** Border-connected color flood, deliberately limited to simple backgrounds. */
export function foregroundMask(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  tolerance: number,
): Uint8Array {
  const n = w * h,
    mask = new Uint8Array(n),
    seen = new Uint8Array(n),
    queue = new Int32Array(n);
  for (let i = 0; i < n; i++) mask[i] = data[i * 4 + 3];
  if (tolerance <= 0) return mask;
  const samples = [0, w - 1, (h - 1) * w, n - 1].map((i) => [
    data[i * 4],
    data[i * 4 + 1],
    data[i * 4 + 2],
  ]);
  let start = 0,
    end = 0;
  const visit = (i: number) => {
    if (seen[i]) return;
    seen[i] = 1;
    const k = i * 4;
    if (
      mask[i] < 4 ||
      samples.some(
        (c) =>
          Math.hypot(data[k] - c[0], data[k + 1] - c[1], data[k + 2] - c[2]) <
          tolerance,
      )
    ) {
      queue[end++] = i;
      mask[i] = 0;
    }
  };
  for (let x = 0; x < w; x++) {
    visit(x);
    visit((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    visit(y * w);
    visit(y * w + w - 1);
  }
  while (start < end) {
    const i = queue[start++],
      x = i % w;
    if (x > 0) visit(i - 1);
    if (x + 1 < w) visit(i + 1);
    if (i >= w) visit(i - w);
    if (i < n - w) visit(i + w);
  }
  return mask;
}
const ellipse = (
  x: number,
  y: number,
  cx: number,
  cy: number,
  rx: number,
  ry: number,
) => ((x - cx) / Math.max(1, rx)) ** 2 + ((y - cy) / Math.max(1, ry)) ** 2 <= 1;
/** Geometric starting masks. Not a trained semantic segmentation model. */
export function separate(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  lm: Landmarks,
  tolerance = 32,
): Part[] {
  const parts = PART_DEFS.map(([id, name, group, color]) => ({
    id,
    name,
    group,
    color,
    visible: true,
    status: "draft" as const,
    mask: new Uint8Array(w * h),
  }));
  const indices = new Map<string, number>(parts.map((p, i) => [p.id, i]));
  const fg = foregroundMask(data, w, h, tolerance);
  const eyeDist = Math.max(
    w * 0.05,
    Math.hypot(lm.eyeL.x - lm.eyeR.x, lm.eyeL.y - lm.eyeR.y),
  );
  const eyeWidth = eyeDist * 0.32,
    eyeHeight = eyeDist * 0.16;
  const cx = (lm.eyeL.x + lm.eyeR.x) / 2,
    headHeight = Math.max(h * 0.12, lm.chin.y - lm.crown.y);
  const cy = lm.crown.y + headHeight * 0.55,
    rx = eyeDist * 1.02,
    ry = headHeight * 0.49;
  for (let i = 0; i < fg.length; i++) {
    if (!fg[i]) continue;
    const x = i % w,
      y = Math.floor(i / w);
    let id = "hair_back";
    if (y > lm.chin.y + headHeight * 0.22) id = "body";
    else if (y > lm.chin.y && Math.abs(x - cx) < rx * 0.42) id = "neck";
    if (ellipse(x, y, cx, cy, rx, ry)) id = "face";
    if (
      y < lm.eyeL.y - headHeight * 0.22 &&
      ellipse(x, y, cx, cy, rx * 1.1, ry * 1.1)
    )
      id = "hair_front";
    for (const side of ["R", "L"] as const) {
      const eye = lm[`eye${side}`];
      if (
        ellipse(
          x,
          y,
          eye.x,
          eye.y - eyeHeight * 2.1,
          eyeWidth * 1.02,
          eyeHeight * 0.3,
        )
      )
        id = `brow${side}`;
      if (ellipse(x, y, eye.x, eye.y, eyeWidth, eyeHeight)) {
        id = `eye${side}_white`;
        if (ellipse(x, y, eye.x, eye.y, eyeWidth * 0.46, eyeHeight * 0.95))
          id = `eye${side}_iris`;
        if (y < eye.y - eyeHeight * 0.55) id = `eye${side}_upper`;
        if (y > eye.y + eyeHeight * 0.65) id = `eye${side}_lower`;
      }
    }
    if (
      ellipse(x, y, lm.mouth.x, lm.mouth.y, eyeDist * 0.18, eyeDist * 0.065)
    ) {
      id =
        y < lm.mouth.y - eyeDist * 0.018
          ? "mouth_upper"
          : y > lm.mouth.y + eyeDist * 0.018
            ? "mouth_lower"
            : "mouth_inner";
    }
    parts[indices.get(id)!].mask[i] = fg[i];
  }
  return parts;
}
export function paint(
  mask: Uint8Array,
  w: number,
  h: number,
  a: Point,
  b: Point,
  radius: number,
  erase: boolean,
  sourceAlpha?: Uint8ClampedArray,
) {
  const length = Math.hypot(b.x - a.x, b.y - a.y),
    steps = Math.max(1, Math.ceil(length / Math.max(1, radius * 0.35)));
  for (let s = 0; s <= steps; s++) {
    const cx = a.x + ((b.x - a.x) * s) / steps,
      cy = a.y + ((b.y - a.y) * s) / steps;
    for (
      let y = Math.max(0, Math.floor(cy - radius));
      y <= Math.min(h - 1, Math.ceil(cy + radius));
      y++
    )
      for (
        let x = Math.max(0, Math.floor(cx - radius));
        x <= Math.min(w - 1, Math.ceil(cx + radius));
        x++
      ) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2) {
          const i = y * w + x;
          mask[i] = erase ? 0 : sourceAlpha ? sourceAlpha[i * 4 + 3] : 255;
        }
      }
  }
}
export function bounds(mask: Uint8Array, w: number, h: number) {
  let left = w,
    top = h,
    right = -1,
    bottom = -1;
  for (let i = 0; i < mask.length; i++)
    if (mask[i] > 8) {
      const x = i % w,
        y = Math.floor(i / w);
      left = Math.min(left, x);
      right = Math.max(right, x);
      top = Math.min(top, y);
      bottom = Math.max(bottom, y);
    }
  return right < left
    ? null
    : { left, top, width: right - left + 1, height: bottom - top + 1 };
}
export function maskedPixels(
  source: Uint8ClampedArray,
  mask: Uint8Array,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(source.length);
  for (let i = 0; i < mask.length; i++) {
    const k = i * 4;
    out[k] = source[k];
    out[k + 1] = source[k + 1];
    out[k + 2] = source[k + 2];
    out[k + 3] = Math.min(source[k + 3], mask[i]);
  }
  return out;
}
/** Validate triangle interiors against the mask, including all crossed edges. */
export function meshFor(
  mask: Uint8Array,
  w: number,
  h: number,
  density = 14,
): Mesh {
  const box = bounds(mask, w, h);
  if (!box) return { vertices: [], triangles: [], uv: [] };
  const spacing = Math.max(3, Math.max(box.width, box.height) / density),
    points: Point[] = [];
  const valid = (x: number, y: number) =>
    x >= 0 &&
    y >= 0 &&
    x < w &&
    y < h &&
    mask[Math.floor(y) * w + Math.floor(x)] > 8;
  for (let y = box.top; y < box.top + box.height; y += spacing)
    for (let x = box.left; x < box.left + box.width; x += spacing) {
      if (valid(x, y)) points.push({ x, y });
      else {
        // Boundary samples keep narrow contours from disappearing at low density.
        for (const [dx, dy] of [
          [spacing * 0.5, 0],
          [0, spacing * 0.5],
          [spacing * 0.5, spacing * 0.5],
        ])
          if (valid(x + dx, y + dy)) points.push({ x: x + dx, y: y + dy });
      }
    }
  if (points.length < 3)
    return {
      vertices: points,
      triangles: [],
      uv: points.map((p) => ({ x: p.x / w, y: p.y / h })),
    };
  const d = Delaunator.from(
      points,
      (p) => p.x,
      (p) => p.y,
    ),
    tris: number[] = [];
  for (let t = 0; t < d.triangles.length; t += 3) {
    const ids = Array.from(d.triangles.slice(t, t + 3)),
      [a, b, c] = ids.map((i) => points[i]);
    let okay = true;
    const steps = Math.max(
      2,
      Math.ceil(
        Math.max(
          Math.hypot(a.x - b.x, a.y - b.y),
          Math.hypot(a.x - c.x, a.y - c.y),
          Math.hypot(b.x - c.x, b.y - c.y),
        ),
      ),
    );
    for (let u = 0; u <= steps && okay; u++)
      for (let v = 0; v <= steps - u; v++) {
        const x = a.x + ((b.x - a.x) * u) / steps + ((c.x - a.x) * v) / steps,
          y = a.y + ((b.y - a.y) * u) / steps + ((c.y - a.y) * v) / steps;
        if (!valid(x, y)) {
          okay = false;
          break;
        }
      }
    if (okay) tris.push(...ids);
  }
  return {
    vertices: points,
    triangles: tris,
    uv: points.map((p) => ({ x: p.x / w, y: p.y / h })),
  };
}
export function deform(
  p: Point,
  part: Part,
  lm: Landmarks,
  params: Parameters,
): Point {
  let { x, y } = p;
  if (part.id === "body") return { x, y };
  const cx = (lm.eyeL.x + lm.eyeR.x) / 2,
    headH = Math.max(1, lm.chin.y - lm.crown.y);
  if (part.id.startsWith("eye")) {
    const side = part.id.startsWith("eyeL") ? "L" : "R",
      eye = lm[`eye${side}`],
      open = params[`eye${side}`];
    y = eye.y + (y - eye.y) * Math.max(0.04, open);
  }
  if (part.id.startsWith("mouth")) {
    if (part.id === "mouth_upper") y -= params.mouth * headH * 0.012;
    if (part.id === "mouth_lower") y += params.mouth * headH * 0.045;
    if (part.id === "mouth_inner")
      y = lm.mouth.y + (y - lm.mouth.y) * (1 + params.mouth * 2);
  }
  const weight =
    part.id === "neck"
      ? clamp((lm.chin.y + headH * 0.25 - y) / (headH * 0.25), 0, 1)
      : 1;
  x +=
    params.angleX *
    headH *
    0.0025 *
    weight *
    (1 - Math.abs(y - lm.eyeL.y) / (headH * 2));
  y += params.angleY * headH * 0.0018 * weight;
  if (part.id.startsWith("hair"))
    x += params.hair * headH * 0.04 * clamp((y - lm.crown.y) / headH, 0, 1.8);
  const angle = ((params.angleZ * Math.PI) / 180) * weight,
    dx = x - cx,
    dy = y - lm.chin.y;
  return {
    x: cx + dx * Math.cos(angle) - dy * Math.sin(angle),
    y: lm.chin.y + dx * Math.sin(angle) + dy * Math.cos(angle),
  };
}
/** Local color diffusion only; provides small hidden-area drafts, not generative inpainting. */
export function expandFill(
  source: Uint8ClampedArray,
  mask: Uint8Array,
  w: number,
  h: number,
  radius: number,
) {
  const out = source.slice(),
    expanded = mask.slice(),
    n = w * h;
  let frontier: number[] = [];
  for (let i = 0; i < n; i++)
    if (mask[i]) {
      const x = i % w;
      if (
        (x > 0 && !mask[i - 1]) ||
        (x + 1 < w && !mask[i + 1]) ||
        (i >= w && !mask[i - w]) ||
        (i < n - w && !mask[i + w])
      )
        frontier.push(i);
    }
  for (let r = 0; r < radius; r++) {
    const next: number[] = [];
    for (const i of frontier)
      for (const j of [
        i % w > 0 ? i - 1 : -1,
        i % w < w - 1 ? i + 1 : -1,
        i - w,
        i + w,
      ]) {
        if (j < 0 || j >= n || expanded[j]) continue;
        expanded[j] = 255;
        out[j * 4] = out[i * 4];
        out[j * 4 + 1] = out[i * 4 + 1];
        out[j * 4 + 2] = out[i * 4 + 2];
        out[j * 4 + 3] = 255;
        next.push(j);
      }
    frontier = next;
  }
  return { data: out, mask: expanded };
}
