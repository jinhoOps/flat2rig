import Delaunator from "delaunator";
import {
  PART_DEFS,
  inferPartRole,
  isHairPart,
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
export function effectiveMask(part: {
  mask: Uint8Array;
  fillMask?: Uint8Array;
}): Uint8Array {
  if (!part.fillMask || part.fillMask.length !== part.mask.length)
    return part.mask;
  const out = new Uint8Array(part.mask.length);
  for (let i = 0; i < out.length; i++)
    out[i] = Math.max(part.mask[i], part.fillMask[i]);
  return out;
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

/** Contour- and landmark-aware Delaunay mesh matching Rust `core::mesh_for_with_anchors`. */
export function meshFor(
  mask: Uint8Array,
  w: number,
  h: number,
  density = 14,
  anchors: Point[] = [],
): Mesh {
  const box = bounds(mask, w, h);
  if (!box) return { vertices: [], triangles: [], uv: [] };
  const span = Math.max(box.width, box.height),
    step = Math.max(3, span / clamp(density, 6, 32)),
    contourMinSq = Math.max(2, step * 0.45) ** 2,
    gridMinSq = Math.max(1.5, step * 0.35) ** 2,
    points: Point[] = [];
  const valid = (x: number, y: number) => {
    const px = x + 1e-6,
      py = y + 1e-6;
    return (
      px >= 0 &&
      py >= 0 &&
      Math.floor(px) < w &&
      Math.floor(py) < h &&
      mask[Math.floor(py) * w + Math.floor(px)] > 8
    );
  };
  const pushIfFar = (candX: number, candY: number, minSq: number) => {
    if (!valid(candX, candY)) return;
    for (const v of points) {
      const dx = v.x - candX,
        dy = v.y - candY;
      if (dx * dx + dy * dy < minSq) return;
    }
    points.push({ x: candX, y: candY });
  };
  const right = box.left + box.width - 1,
    bottom = box.top + box.height - 1;
  // 1. Sample contour boundary pixels first.
  for (let y = box.top; y <= bottom; y++) {
    const row = y * w;
    for (let x = box.left; x <= right; x++) {
      if (mask[row + x] <= 8) continue;
      const isBoundary =
        x === 0 ||
        y === 0 ||
        x + 1 >= w ||
        y + 1 >= h ||
        mask[row + x - 1] <= 8 ||
        mask[row + x + 1] <= 8 ||
        mask[(y - 1) * w + x] <= 8 ||
        mask[(y + 1) * w + x] <= 8;
      if (isBoundary) pushIfFar(x, y, contourMinSq);
    }
  }
  // 2. Inject interior landmark anchors strictly inside active mask.
  for (const a of anchors) pushIfFar(a.x, a.y, contourMinSq * 0.5);
  // 3. Fill interior uniform grid points.
  for (let y = box.top; y <= bottom; y += step)
    for (let x = box.left; x <= right; x += step)
      pushIfFar(x, y, gridMinSq);
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

export function meshForPart(
  mask: Uint8Array,
  w: number,
  h: number,
  density = 14,
  lm?: Landmarks,
): Mesh {
  return meshFor(
    mask,
    w,
    h,
    density,
    lm ? [lm.eyeL, lm.eyeR, lm.mouth, lm.chin, lm.crown] : [],
  );
}

export interface MeshQualityReport {
  partId: string;
  triangleCount: number;
  holeCrossings: number;
  longTriangles: number;
  flippedAtExtremes: number;
  extremePosesTested: number;
  warnings: string[];
}

function triCross(a: Point, b: Point, c: Point): number {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

export function checkMeshQuality(
  mesh: Mesh,
  mask: Uint8Array,
  w: number,
  h: number,
  part: Pick<Part, "id">,
  lm: Landmarks,
): MeshQualityReport {
  const triCount = Math.floor(mesh.triangles.length / 3);
  const hasPixels = mask.some((a) => a > 8);
  if (!triCount || !mesh.vertices.length) {
    return {
      partId: part.id,
      triangleCount: 0,
      holeCrossings: 0,
      longTriangles: 0,
      flippedAtExtremes: 0,
      extremePosesTested: 0,
      warnings: hasPixels
        ? [`${part.id}: 유효한 메쉬 삼각형이 생성되지 않았습니다.`]
        : [],
    };
  }
  const validPx = (x: number, y: number) => {
    const px = x + 1e-6,
      py = y + 1e-6;
    return (
      px >= 0 &&
      py >= 0 &&
      Math.floor(px) < w &&
      Math.floor(py) < h &&
      mask[Math.floor(py) * w + Math.floor(px)] > 8
    );
  };
  const maxEdges: number[] = [],
    restCrosses: number[] = [];
  let holeCrossings = 0;
  for (let t = 0; t < triCount; t++) {
    const a = mesh.vertices[mesh.triangles[t * 3]],
      b = mesh.vertices[mesh.triangles[t * 3 + 1]],
      c = mesh.vertices[mesh.triangles[t * 3 + 2]];
    maxEdges.push(
      Math.max(
        Math.hypot(a.x - b.x, a.y - b.y),
        Math.hypot(a.x - c.x, a.y - c.y),
        Math.hypot(b.x - c.x, b.y - c.y),
      ),
    );
    restCrosses.push(triCross(a, b, c));
    const cx = (a.x + b.x + c.x) / 3,
      cy = (a.y + b.y + c.y) / 3;
    const ok =
      validPx(cx, cy) &&
      validPx(a.x * 0.5 + b.x * 0.25 + c.x * 0.25, a.y * 0.5 + b.y * 0.25 + c.y * 0.25) &&
      validPx(a.x * 0.25 + b.x * 0.5 + c.x * 0.25, a.y * 0.25 + b.y * 0.5 + c.y * 0.25) &&
      validPx(a.x * 0.25 + b.x * 0.25 + c.x * 0.5, a.y * 0.25 + b.y * 0.25 + c.y * 0.5);
    if (!ok) holeCrossings++;
  }
  const sorted = [...maxEdges].sort((x, y) => x - y);
  const longThreshold = Math.max(48, sorted[Math.floor(sorted.length / 2)] * 3.2);
  const longTriangles = maxEdges.filter((e) => e > longThreshold).length;
  const zero: Parameters = {
    angleX: 0,
    angleY: 0,
    angleZ: 0,
    eyeL: 1,
    eyeR: 1,
    mouth: 0,
    hair: 0,
  };
  const extremePoses: Parameters[] = [
    { ...zero, angleX: -30, angleY: -30, angleZ: -30 },
    { ...zero, angleX: 30, angleY: 30, angleZ: 30 },
    { ...zero, angleX: -30, angleY: 30, angleZ: 15 },
    { ...zero, angleX: 30, angleY: -30, angleZ: -15 },
    { ...zero, eyeL: 0, eyeR: 0 },
    { ...zero, eyeL: 1.2, eyeR: 1.2 },
    { ...zero, mouth: 0 },
    { ...zero, mouth: 1 },
    { ...zero, hair: -1 },
    { ...zero, hair: 1 },
    { ...zero, angleX: 30, angleY: -30, angleZ: 30, eyeL: 0, eyeR: 0, mouth: 1, hair: -1 },
    { ...zero, angleX: -30, angleY: 30, angleZ: -30, eyeL: 0, eyeR: 0, mouth: 1, hair: 1 },
  ];
  const flipped = new Uint8Array(triCount);
  for (const pose of extremePoses) {
    const deformed = mesh.vertices.map((v) => deform(v, part, lm, pose));
    for (let t = 0; t < triCount; t++) {
      if (flipped[t] || Math.abs(restCrosses[t]) < 1e-6) continue;
      const pc = triCross(
        deformed[mesh.triangles[t * 3]],
        deformed[mesh.triangles[t * 3 + 1]],
        deformed[mesh.triangles[t * 3 + 2]],
      );
      if (pc * restCrosses[t] <= 0) flipped[t] = 1;
    }
  }
  const flippedAtExtremes = flipped.reduce((s, b) => s + b, 0);
  const warnings: string[] = [];
  if (holeCrossings > 0)
    warnings.push(`${part.id}: 빈 구멍(hole) 횡단 삼각형 ${holeCrossings}개`);
  if (longTriangles > 0)
    warnings.push(`${part.id}: 과도하게 긴 삼각형 ${longTriangles}개`);
  if (flippedAtExtremes > 0)
    warnings.push(`${part.id}: 극단 포즈에서 뒤집히는 삼각형 ${flippedAtExtremes}개`);
  return {
    partId: part.id,
    triangleCount: triCount,
    holeCrossings,
    longTriangles,
    flippedAtExtremes,
    extremePosesTested: extremePoses.length,
    warnings,
  };
}
function hairSwayProfile(part: Pick<Part, "id" | "role">): [number, number] {
  const role = inferPartRole(part.id, part.role);
  const match = /_(\d+)$/.exec(part.id);
  const strandIdx = Math.max(1, match ? Number(match[1]) : 1);
  const strandBoost = 1 + Math.min(4, Math.max(0, strandIdx - 1)) * 0.1;
  if (role === "hair_side_L" || role === "hair_side_R") {
    return [1.22 * strandBoost, 0.15];
  }
  if (role === "hair_back") {
    return [0.85 * strandBoost, 0.1];
  }
  return [1.0 * strandBoost, 0.0];
}

export function deform(
  p: Point,
  part: Pick<Part, "id" | "role">,
  lm: Landmarks,
  params: Parameters,
): Point {
  let { x, y } = p;
  if (
    part.id === "body" ||
    part.role === "clothes" ||
    part.id.startsWith("clothes")
  )
    return { x, y };
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
  if (isHairPart(part)) {
    const [amp, rootOffset] = hairSwayProfile(part);
    const rootY = lm.crown.y + headH * rootOffset;
    x +=
      params.hair *
      headH *
      0.04 *
      amp *
      clamp((y - rootY) / headH, 0, 1.8);
  }
  const angle = ((params.angleZ * Math.PI) / 180) * weight,
    dx = x - cx,
    dy = y - lm.chin.y;
  return {
    x: cx + dx * Math.cos(angle) - dy * Math.sin(angle),
    y: lm.chin.y + dx * Math.sin(angle) + dy * Math.cos(angle),
  };
}
/** Local neighbor pixel expansion keeping `visibleMask` separated from `fillMask`.
 * Erodes 2px inward before sampling colors so outermost lineart/hair shadows do not bleed outward. */
export function expandFillSeparated(
  source: Uint8ClampedArray,
  visibleMask: Uint8Array,
  w: number,
  h: number,
  radius: number,
  partId?: string,
) {
  const n = w * h,
    out = source.slice(),
    fillMask = new Uint8Array(n),
    combinedMask = visibleMask.slice();
  let filledPixels = 0;

  // 1. Inward boundary depth (0 = outer rim touching unowned pixel)
  const depth = new Uint8Array(n);
  depth.fill(255);
  const erodeQ: number[] = [];
  for (let i = 0; i < n; i++) {
    if (visibleMask[i] > 0) {
      const x = i % w;
      if (
        (x > 0 && visibleMask[i - 1] === 0) ||
        (x + 1 < w && visibleMask[i + 1] === 0) ||
        (i >= w && visibleMask[i - w] === 0) ||
        (i < n - w && visibleMask[i + w] === 0)
      ) {
        depth[i] = 0;
        erodeQ.push(i);
      }
    }
  }
  for (let head = 0; head < erodeQ.length; head++) {
    const i = erodeQ[head];
    const nextD = depth[i] + 1;
    if (nextD > 2) continue;
    const x = i % w;
    for (const j of [
      x > 0 ? i - 1 : -1,
      x + 1 < w ? i + 1 : -1,
      i >= w ? i - w : -1,
      i + w < n ? i + w : -1,
    ]) {
      if (j >= 0 && visibleMask[j] > 0 && depth[j] === 255) {
        depth[j] = nextD;
        erodeQ.push(j);
      }
    }
  }

  // 2. Pick deepest available interior (>=2 -> >=1 -> >=0) and propagate clean RGB to visible rim
  let minSeedDepth = 0;
  for (const candidate of [2, 1, 0]) {
    let found = false;
    for (let i = 0; i < n; i++) {
      if (visibleMask[i] > 0 && depth[i] >= candidate) {
        found = true;
        break;
      }
    }
    if (found) {
      minSeedDepth = candidate;
      break;
    }
  }
  const sampleRgb = new Uint8Array(n * 3);
  const seedReady = new Uint8Array(n);
  const seedQ: number[] = [];
  for (let i = 0; i < n; i++) {
    if (visibleMask[i] > 0 && depth[i] >= minSeedDepth) {
      sampleRgb[i * 3] = source[i * 4];
      sampleRgb[i * 3 + 1] = source[i * 4 + 1];
      sampleRgb[i * 3 + 2] = source[i * 4 + 2];
      seedReady[i] = 1;
      seedQ.push(i);
    }
  }
  for (let head = 0; head < seedQ.length; head++) {
    const i = seedQ[head];
    const x = i % w;
    for (const j of [
      x > 0 ? i - 1 : -1,
      x + 1 < w ? i + 1 : -1,
      i >= w ? i - w : -1,
      i + w < n ? i + w : -1,
    ]) {
      if (j >= 0 && visibleMask[j] > 0 && !seedReady[j]) {
        sampleRgb[j * 3] = sampleRgb[i * 3];
        sampleRgb[j * 3 + 1] = sampleRgb[i * 3 + 1];
        sampleRgb[j * 3 + 2] = sampleRgb[i * 3 + 2];
        seedReady[j] = 1;
        seedQ.push(j);
      }
    }
  }

  // 3. BFS outward from visible rim into fillMask using clean interior sampleRgb
  let frontier: number[] = [];
  for (let i = 0; i < n; i++)
    if (visibleMask[i]) {
      const x = i % w;
      if (
        (x > 0 && !visibleMask[i - 1]) ||
        (x + 1 < w && !visibleMask[i + 1]) ||
        (i >= w && !visibleMask[i - w]) ||
        (i < n - w && !visibleMask[i + w])
      )
        frontier.push(i);
    }
  const maxR = Math.min(128, Math.max(0, radius));
  for (let r = 0; r < maxR; r++) {
    const next: number[] = [];
    for (const i of frontier)
      for (const j of [
        i % w > 0 ? i - 1 : -1,
        i % w < w - 1 ? i + 1 : -1,
        i - w,
        i + w,
      ]) {
        if (j < 0 || j >= n || combinedMask[j]) continue;
        combinedMask[j] = 255;
        fillMask[j] = 255;
        filledPixels++;
        let sr = 0,
          sg = 0,
          sb = 0,
          cnt = 0;
        for (const nb of [
          j % w > 0 ? j - 1 : -1,
          j % w < w - 1 ? j + 1 : -1,
          j - w,
          j + w,
        ]) {
          if (nb < 0 || nb >= n || nb === j) continue;
          if (visibleMask[nb] > 0 || (fillMask[nb] > 0 && out[nb * 4 + 3] > 0)) {
            if (seedReady[nb]) {
              sr += sampleRgb[nb * 3];
              sg += sampleRgb[nb * 3 + 1];
              sb += sampleRgb[nb * 3 + 2];
            } else {
              sr += out[nb * 4];
              sg += out[nb * 4 + 1];
              sb += out[nb * 4 + 2];
            }
            cnt++;
          }
        }
        const pr = cnt
          ? (sr / cnt) | 0
          : seedReady[i]
            ? sampleRgb[i * 3]
            : out[i * 4];
        const pg = cnt
          ? (sg / cnt) | 0
          : seedReady[i]
            ? sampleRgb[i * 3 + 1]
            : out[i * 4 + 1];
        const pb = cnt
          ? (sb / cnt) | 0
          : seedReady[i]
            ? sampleRgb[i * 3 + 2]
            : out[i * 4 + 2];
        sampleRgb[j * 3] = pr;
        sampleRgb[j * 3 + 1] = pg;
        sampleRgb[j * 3 + 2] = pb;
        seedReady[j] = 1;
        out[j * 4] = pr;
        out[j * 4 + 1] = pg;
        out[j * 4 + 2] = pb;
        out[j * 4 + 3] = 255;
        next.push(j);
      }
    if (!next.length) break;
    frontier = next;
  }
  const hiddenFeature =
    !!partId &&
    (partId.startsWith("eye") ||
      partId.startsWith("mouth") ||
      partId.startsWith("hair"));
  return {
    data: out,
    fillMask,
    combinedMask,
    filledPixels,
    fillKind: "neighbor" as const,
    warning: hiddenFeature
      ? "경계 2px 침식 내삽(neighbor)으로 외곽선 번짐을 줄였으나 가려진 눈·입·머리카락 내부 디테일은 자동 복원되지 않습니다."
      : undefined,
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
  const res = expandFillSeparated(source, mask, w, h, radius);
  return { data: res.data, mask: res.combinedMask };
}
