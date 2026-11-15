import Delaunator from "delaunator";
import {
  PART_DEFS,
  REST,
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
interface HeadFrame {
  originX: number;
  originY: number;
  exX: number;
  exY: number;
  eyX: number;
  eyY: number;
  eyeDist: number;
  hh: number;
  chinV: number;
  crownV: number;
}

function buildHeadFrame(lm: Landmarks, w: number, h: number): HeadFrame {
  const originX = (lm.eyeL.x + lm.eyeR.x) / 2;
  const originY = (lm.eyeL.y + lm.eyeR.y) / 2;
  const dx = lm.eyeL.x - lm.eyeR.x;
  const dy = lm.eyeL.y - lm.eyeR.y;
  const rawLen = Math.hypot(dx, dy);
  const eyeDist = Math.max(w * 0.05, rawLen);
  const exX = rawLen > 1e-4 && dx >= 0 ? dx / rawLen : 1;
  const exY = rawLen > 1e-4 && dx >= 0 ? dy / rawLen : 0;
  const eyX = -exY;
  const eyY = exX;
  const crownV = (lm.crown.x - originX) * eyX + (lm.crown.y - originY) * eyY;
  const chinV = (lm.chin.x - originX) * eyX + (lm.chin.y - originY) * eyY;
  const hh = Math.max(h * 0.12, lm.chin.y - lm.crown.y, chinV - crownV);
  return {
    originX,
    originY,
    exX,
    exY,
    eyX,
    eyY,
    eyeDist,
    hh,
    chinV,
    crownV,
  };
}

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
  const frame = buildHeadFrame(lm, w, h);
  const toLocal = (px: number, py: number): [number, number] => {
    const dx = px - frame.originX;
    const dy = py - frame.originY;
    return [
      dx * frame.exX + dy * frame.exY,
      dx * frame.eyX + dy * frame.eyY,
    ];
  };
  const eyeDist = frame.eyeDist;
  const eyeWidth = eyeDist * 0.32,
    eyeHeight = eyeDist * 0.16;
  const headHeight = frame.hh;
  const faceCv = frame.crownV + headHeight * 0.55,
    rx = eyeDist * 1.02,
    ry = headHeight * 0.49;
  const [eyeRU, eyeRV] = toLocal(lm.eyeR.x, lm.eyeR.y);
  const [eyeLU, eyeLV] = toLocal(lm.eyeL.x, lm.eyeL.y);
  const [mouthU, mouthV] = toLocal(lm.mouth.x, lm.mouth.y);
  for (let i = 0; i < fg.length; i++) {
    if (!fg[i]) continue;
    const x = i % w,
      y = Math.floor(i / w);
    const [u, v] = toLocal(x, y);
    let id = "hair_back";
    if (v > frame.chinV + headHeight * 0.22) id = "body";
    else if (v > frame.chinV && Math.abs(u) < rx * 0.42) id = "neck";
    if (ellipse(u, v, 0, faceCv, rx, ry)) id = "face";
    if (
      v < Math.min(eyeLV, eyeRV) - headHeight * 0.22 &&
      ellipse(u, v, 0, faceCv, rx * 1.1, ry * 1.1)
    )
      id = "hair_front";
    for (const [side, eu, ev] of [
      ["R", eyeRU, eyeRV],
      ["L", eyeLU, eyeLV],
    ] as const) {
      if (
        ellipse(u, v, eu, ev - eyeHeight * 2.1, eyeWidth * 1.02, eyeHeight * 0.3)
      )
        id = `brow${side}`;
      if (ellipse(u, v, eu, ev, eyeWidth, eyeHeight)) {
        id = `eye${side}_white`;
        if (ellipse(u, v, eu, ev, eyeWidth * 0.46, eyeHeight * 0.95))
          id = `eye${side}_iris`;
        if (v < ev - eyeHeight * 0.55) id = `eye${side}_upper`;
        if (v > ev + eyeHeight * 0.65) id = `eye${side}_lower`;
      }
    }
    if (ellipse(u, v, mouthU, mouthV, eyeDist * 0.18, eyeDist * 0.065)) {
      id =
        v < mouthV - eyeDist * 0.018
          ? "mouth_upper"
          : v > mouthV + eyeDist * 0.018
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

/** Composites a part's RGBA buffer, recovering original pixels when visible masks are expanded past an older generated texture. */
export function compositePartTexture(
  part: Pick<Part, "mask" | "fillMask" | "fillKind">,
  sourcePixels: Uint8ClampedArray,
  texturePixels?: Uint8ClampedArray,
): Uint8ClampedArray {
  const mask = effectiveMask(part);
  if (!texturePixels) return maskedPixels(sourcePixels, mask);
  if (part.fillKind === "external") return maskedPixels(texturePixels, mask);

  const out = new Uint8ClampedArray(sourcePixels.length);
  for (let i = 0; i < mask.length; i++) {
    const eff = mask[i];
    if (eff === 0) continue;
    const k = i * 4;
    const texA = texturePixels[k + 3];
    const vis = part.mask[i];
    const useSource = vis > 0 && texA < vis && sourcePixels[k + 3] > texA;
    const rgb = useSource ? sourcePixels : texturePixels;
    out[k] = rgb[k];
    out[k + 1] = rgb[k + 1];
    out[k + 2] = rgb[k + 2];
    out[k + 3] = Math.min(rgb[k + 3], eff);
  }
  return out;
}

type ArtMeshRole =
  | "eyelid"
  | "iris"
  | "eyeAperture"
  | "linearBrowOrLip"
  | "hair"
  | "face"
  | "structuralBody"
  | "generic";

function classifyArtMeshRole(id: string): ArtMeshRole {
  if (!id) return "generic";
  if ((id.endsWith("_upper") || id.endsWith("_lower")) && id.startsWith("eye"))
    return "eyelid";
  if (id.startsWith("eye") && id.endsWith("_iris")) return "iris";
  if ((id.startsWith("eye") && id.endsWith("_white")) || id === "mouth_inner")
    return "eyeAperture";
  if (id.startsWith("brow") || id === "mouth_upper" || id === "mouth_lower")
    return "linearBrowOrLip";
  if (id.startsWith("hair_") || id === "hair") return "hair";
  if (id === "face") return "face";
  return "structuralBody";
}

function traceContourLoops(
  mask: Uint8Array,
  w: number,
  left: number,
  right: number,
  top: number,
  bottom: number,
): Array<{ pts: Point[]; isOuter: boolean }> {
  const bw = right - left + 1;
  const bh = bottom - top + 1;
  const gw = bw + 1;
  const outBits = new Uint8Array(gw * (bh + 1));
  const active = (lx: number, ly: number) =>
    lx >= 0 &&
    ly >= 0 &&
    lx < bw &&
    ly < bh &&
    mask[(top + ly) * w + (left + lx)] > 8;

  for (let py = 0; py < bh; py++) {
    for (let px = 0; px < bw; px++) {
      if (!active(px, py)) continue;
      if (!active(px, py + 1)) outBits[(py + 1) * gw + px] |= 1 << 0;
      if (!active(px + 1, py)) outBits[(py + 1) * gw + (px + 1)] |= 1 << 1;
      if (!active(px, py - 1)) outBits[py * gw + (px + 1)] |= 1 << 2;
      if (!active(px - 1, py)) outBits[py * gw + px] |= 1 << 3;
    }
  }

  const DX = [1, 0, -1, 0];
  const DY = [0, -1, 0, 1];
  const loops: Array<{ pts: Point[]; isOuter: boolean }> = [];
  for (let startIdx = 0; startIdx < outBits.length; startIdx++) {
    while (outBits[startIdx] !== 0) {
      let startDir = 0;
      while ((outBits[startIdx] & (1 << startDir)) === 0) startDir++;
      let curX = startIdx % gw;
      let curY = Math.floor(startIdx / gw);
      let dir = startDir;
      const pts: Point[] = [];
      let area2 = 0;
      const maxSteps = gw * (bh + 1) * 4;
      for (let step = 0; step < maxSteps; step++) {
        const idx = curY * gw + curX;
        if ((outBits[idx] & (1 << dir)) === 0) break;
        outBits[idx] &= ~(1 << dir);
        const cx = left + curX;
        const cy = top + curY;
        const mid: Point =
          dir === 0
            ? { x: cx + 0.5, y: cy - 0.14 }
            : dir === 1
              ? { x: cx - 0.14, y: cy - 0.5 }
              : dir === 2
                ? { x: cx - 0.5, y: cy + 0.14 }
                : { x: cx + 0.14, y: cy + 0.5 };
        if (pts.length > 0) {
          const prev = pts[pts.length - 1];
          area2 += prev.x * mid.y - mid.x * prev.y;
        }
        pts.push(mid);
        curX += DX[dir];
        curY += DY[dir];
        const nextIdx = curY * gw + curX;
        if (
          nextIdx === startIdx &&
          step > 0 &&
          (outBits[nextIdx] & (1 << startDir)) === 0
        ) {
          break;
        }
        const turnOrder = [(dir + 1) & 3, dir, (dir + 3) & 3, (dir + 2) & 3];
        const nd = turnOrder.find((cand) => (outBits[nextIdx] & (1 << cand)) !== 0);
        if (nd === undefined) break;
        dir = nd;
      }
      if (pts.length >= 3) {
        const first = pts[0];
        const last = pts[pts.length - 1];
        area2 += last.x * first.y - first.x * last.y;
        loops.push({ pts, isOuter: area2 < 0 });
      }
    }
  }
  return loops;
}

function smoothAndResampleLoop(
  raw: Point[],
  contourStep: number,
  valid: (x: number, y: number) => boolean,
): Array<{ pt: Point; inNormal: Point }> {
  const m = raw.length;
  if (m < 3) return [];
  let cur = raw.slice();
  if (m >= 8) {
    for (let pass = 0; pass < 2; pass++) {
      const next = cur.slice();
      for (let i = 0; i < m; i++) {
        const prev = cur[(i + m - 1) % m];
        const p = cur[i];
        const nxt = cur[(i + 1) % m];
        const v1x = p.x - prev.x;
        const v1y = p.y - prev.y;
        const v2x = nxt.x - p.x;
        const v2y = nxt.y - p.y;
        const l1 = Math.hypot(v1x, v1y);
        const l2 = Math.hypot(v2x, v2y);
        if (l1 > 1e-6 && l2 > 1e-6 && (v1x * v2x + v1y * v2y) / (l1 * l2) < 0.35)
          continue;
        const cx = prev.x * 0.25 + p.x * 0.5 + nxt.x * 0.25;
        const cy = prev.y * 0.25 + p.y * 0.5 + nxt.y * 0.25;
        if (valid(cx, cy)) next[i] = { x: cx, y: cy };
      }
      cur = next;
    }
  }

  let perim = 0;
  for (let i = 0; i < m; i++) {
    const a = cur[i];
    const b = cur[(i + 1) % m];
    perim += Math.hypot(b.x - a.x, b.y - a.y);
  }
  if (perim < 1) return [];
  const step = Math.max(0.8, Math.min(contourStep, Math.max(0.8, perim / 6)));
  const sampled: Point[] = [cur[0]];
  let acc = 0;
  let pdx = cur[1].x - cur[0].x;
  let pdy = cur[1].y - cur[0].y;
  const plen = Math.hypot(pdx, pdy);
  if (plen > 1e-6) {
    pdx /= plen;
    pdy /= plen;
  } else {
    pdx = 0;
    pdy = 0;
  }
  for (let i = 0; i < m; i++) {
    const a = cur[i];
    const b = cur[(i + 1) % m];
    const sx = b.x - a.x;
    const sy = b.y - a.y;
    const segLen = Math.hypot(sx, sy);
    if (segLen < 1e-6) continue;
    const dx = sx / segLen;
    const dy = sy / segLen;
    const last = sampled[sampled.length - 1];
    if (
      pdx * dx + pdy * dy < 0.65 &&
      Math.hypot(last.x - a.x, last.y - a.y) >= Math.max(0.6, step * 0.35) &&
      valid(a.x, a.y)
    ) {
      sampled.push(a);
      acc = 0;
    }
    acc += segLen;
    if (acc >= step) {
      const cand = valid(b.x, b.y) ? b : a;
      const tail = sampled[sampled.length - 1];
      if (
        Math.hypot(tail.x - cand.x, tail.y - cand.y) >=
          Math.max(0.5, step * 0.35) &&
        valid(cand.x, cand.y)
      ) {
        sampled.push(cand);
        acc = 0;
      }
    }
    pdx = dx;
    pdy = dy;
  }
  if (sampled.length >= 2) {
    const first = sampled[0];
    const last = sampled[sampled.length - 1];
    if (Math.hypot(first.x - last.x, first.y - last.y) < step * 0.4) {
      sampled.pop();
    }
  }
  const k = sampled.length;
  if (k < 3) return [];
  const out: Array<{ pt: Point; inNormal: Point }> = [];
  for (let i = 0; i < k; i++) {
    const prev = sampled[(i + k - 1) % k];
    const nxt = sampled[(i + 1) % k];
    const tx = nxt.x - prev.x;
    const ty = nxt.y - prev.y;
    const tl = Math.hypot(tx, ty);
    out.push({
      pt: sampled[i],
      inNormal: tl > 1e-6 ? { x: ty / tl, y: -tx / tl } : { x: 0, y: 0 },
    });
  }
  return out;
}

function chamferBoundaryDistance(
  mask: Uint8Array,
  w: number,
  left: number,
  right: number,
  top: number,
  bottom: number,
): Uint16Array {
  const bw = right - left + 1;
  const bh = bottom - top + 1;
  const dist = new Uint16Array(bw * bh);
  dist.fill(0xffff);
  const q = new Int32Array(bw * bh);
  let head = 0;
  let tail = 0;
  for (let ly = 0; ly < bh; ly++) {
    for (let lx = 0; lx < bw; lx++) {
      const idx = (top + ly) * w + (left + lx);
      if (mask[idx] <= 8) continue;
      const x = left + lx;
      const y = top + ly;
      const touchesOut =
        lx === 0 ||
        ly === 0 ||
        lx + 1 === bw ||
        ly + 1 === bh ||
        mask[y * w + (x - 1)] <= 8 ||
        mask[y * w + (x + 1)] <= 8 ||
        mask[(y - 1) * w + x] <= 8 ||
        mask[(y + 1) * w + x] <= 8;
      if (touchesOut) {
        const dIdx = ly * bw + lx;
        dist[dIdx] = 10;
        q[tail++] = dIdx;
      }
    }
  }
  const OFF = [
    [-1, 0, 10],
    [1, 0, 10],
    [0, -1, 10],
    [0, 1, 10],
    [-1, -1, 14],
    [1, -1, 14],
    [-1, 1, 14],
    [1, 1, 14],
  ] as const;
  while (head < tail) {
    const cur = q[head++];
    const cx = cur % bw;
    const cy = (cur - cx) / bw;
    const curD = dist[cur];
    for (const [dx, dy, cost] of OFF) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) continue;
      if (mask[(top + ny) * w + (left + nx)] <= 8) continue;
      const nd = curD + cost;
      const nIdx = ny * bw + nx;
      if (nd < dist[nIdx]) {
        dist[nIdx] = nd;
        q[tail++] = nIdx;
      }
    }
  }
  return dist;
}

function buildArtMesh(
  mask: Uint8Array,
  w: number,
  h: number,
  density: number,
  role: ArtMeshRole,
  lm?: Landmarks,
  anchors: Point[] = [],
): Mesh {
  const box = bounds(mask, w, h);
  if (!box) return { vertices: [], triangles: [], uv: [] };
  const left = box.left;
  const right = box.left + box.width - 1;
  const top = box.top;
  const bottom = box.top + box.height - 1;
  const bw = box.width;
  const bh = box.height;
  const span = Math.max(bw, bh);
  const dClamped = clamp(density, 6, 32);
  let ringStep: number;
  let contourStep: number;
  if (
    role === "eyelid" ||
    role === "iris" ||
    role === "eyeAperture" ||
    role === "linearBrowOrLip"
  ) {
    ringStep = clamp(span / (dClamped * 1.15), 2.2, 14);
    contourStep = clamp(ringStep * 0.6, 1.8, 8.5);
  } else if (role === "hair" || role === "face") {
    ringStep = clamp(span / dClamped, 3, 22);
    contourStep = clamp(ringStep * 0.52, 2.4, 12);
  } else {
    ringStep = clamp(span / dClamped, 3, 28);
    contourStep = clamp(ringStep * 0.48, 2.8, 13);
  }

  const valid = (x: number, y: number) => {
    const px = x + 1e-6;
    const py = y + 1e-6;
    return (
      px >= 0 &&
      py >= 0 &&
      Math.floor(px) < w &&
      Math.floor(py) < h &&
      mask[Math.floor(py) * w + Math.floor(px)] > 8
    );
  };

  const points: Point[] = [];
  const vertNormals: Array<Point | null> = [];
  const pushIfFar = (
    candX: number,
    candY: number,
    minSq: number,
    inN: Point | null,
  ) => {
    if (!valid(candX, candY)) return;
    for (let i = 0; i < points.length; i++) {
      const v = points[i];
      const dx = v.x - candX;
      const dy = v.y - candY;
      const distSq = dx * dx + dy * dy;
      const nb = vertNormals[i];
      const allowedSq =
        inN && nb && inN.x * nb.x + inN.y * nb.y < 0
          ? Math.min(minSq, 0.42 * 0.42)
          : minSq;
      if (distSq < allowedSq) return;
    }
    points.push({ x: candX, y: candY });
    vertNormals.push(inN);
  };

  const rawLoops = traceContourLoops(mask, w, left, right, top, bottom);
  const contourMinSq = clamp(contourStep * 0.46, 1.2, 8) ** 2;
  const insetMinSq = Math.max(1.5, ringStep * 0.42) ** 2;
  const insetCandidates: Array<{ pt: Point; inN: Point }> = [];

  for (const loop of rawLoops) {
    const resampled = smoothAndResampleLoop(loop.pts, contourStep, valid);
    for (const { pt, inNormal } of resampled) {
      pushIfFar(pt.x, pt.y, contourMinSq, inNormal);
      if (
        loop.isOuter &&
        inNormal.x * inNormal.x + inNormal.y * inNormal.y > 0.5
      ) {
        insetCandidates.push({
          pt: {
            x: pt.x + inNormal.x * ringStep * 0.55,
            y: pt.y + inNormal.y * ringStep * 0.55,
          },
          inN: inNormal,
        });
      }
    }
  }
  for (const { pt, inN } of insetCandidates) {
    pushIfFar(pt.x, pt.y, insetMinSq, inN);
  }

  const guideMinSq = Math.max(1.4, ringStep * 0.4) ** 2;
  const distMap = chamferBoundaryDistance(mask, w, left, right, top, bottom);

  if (role === "eyelid") {
    const cols = Math.round(clamp(bw / (ringStep * 0.55), 7, 15));
    for (let c = 0; c < cols; c++) {
      const x = Math.round(
        left + (c / (cols - 1)) * Math.max(0, bw - 1),
      );
      let yt = -1;
      let y1 = 0;
      for (let y = top; y <= bottom; y++) {
        if (mask[y * w + x] > 8) {
          if (yt < 0) yt = y;
          y1 = y;
        }
      }
      if (yt >= 0) {
        const spanY = y1 - yt;
        if (spanY >= 2) {
          for (const frac of [0.22, 0.5, 0.78]) {
            pushIfFar(x + 0.5, yt + 0.5 + spanY * frac, guideMinSq * 0.75, null);
          }
        } else {
          pushIfFar(x + 0.5, (yt + y1) * 0.5 + 0.5, guideMinSq * 0.75, null);
        }
      }
    }
  } else if (role === "iris") {
    let sx = 0,
      sy = 0,
      cnt = 0;
    for (let y = top; y <= bottom; y++) {
      for (let x = left; x <= right; x++) {
        if (mask[y * w + x] > 8) {
          sx += x + 0.5;
          sy += y + 0.5;
          cnt++;
        }
      }
    }
    if (cnt > 0) {
      const cx = sx / cnt;
      const cy = sy / cnt;
      const rx = Math.max(1, bw * 0.5);
      const ry = Math.max(1, bh * 0.5);
      pushIfFar(cx, cy, guideMinSq * 0.5, null);
      const sectors = 12;
      for (const rFrac of [0.36, 0.68]) {
        for (let s = 0; s < sectors; s++) {
          const ang = (s / sectors) * Math.PI * 2;
          pushIfFar(
            cx + rx * rFrac * Math.cos(ang),
            cy + ry * rFrac * Math.sin(ang),
            guideMinSq * 0.65,
            null,
          );
        }
      }
    }
  } else if (role === "eyeAperture") {
    const cx = (left + right) * 0.5 + 0.5;
    const cy = (top + bottom) * 0.5 + 0.5;
    const rx = Math.max(1, bw * 0.5);
    const ry = Math.max(1, bh * 0.5);
    pushIfFar(cx, cy, guideMinSq * 0.5, null);
    for (let s = 0; s < 10; s++) {
      const ang = (s / 10) * Math.PI * 2;
      pushIfFar(
        cx + rx * 0.5 * Math.cos(ang),
        cy + ry * 0.5 * Math.sin(ang),
        guideMinSq * 0.7,
        null,
      );
    }
  } else if (role === "linearBrowOrLip") {
    const cols = Math.round(clamp(bw / (ringStep * 0.55), 7, 13));
    for (let c = 0; c < cols; c++) {
      const x = Math.round(left + (c / (cols - 1)) * Math.max(0, bw - 1));
      let sumY = 0,
        yCnt = 0,
        minY = bottom,
        maxY = top;
      for (let y = top; y <= bottom; y++) {
        if (mask[y * w + x] > 8) {
          sumY += y + 0.5;
          yCnt++;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
      if (yCnt > 0) {
        pushIfFar(x + 0.5, sumY / yCnt, guideMinSq * 0.6, null);
        if (maxY > minY + 3) {
          for (const frac of [0.25, 0.75]) {
            pushIfFar(
              x + 0.5,
              minY + 0.5 + (maxY - minY) * frac,
              guideMinSq * 0.75,
              null,
            );
          }
        }
      }
    }
  } else if (role === "hair") {
    let y = top + ringStep * 0.5;
    const rootCutoff = top + bh * 0.36;
    while (y <= bottom) {
      const iy = clamp(Math.round(y), top, bottom);
      const rowStep = y < rootCutoff ? ringStep * 1.12 : ringStep * 0.68;
      let x = left;
      while (x <= right) {
        if (mask[iy * w + x] <= 8) {
          x++;
          continue;
        }
        const segStart = x;
        while (x + 1 <= right && mask[iy * w + x + 1] > 8) x++;
        const segW = x - segStart + 1;
        const fracs =
          segW >= ringStep * 2.2
            ? [0.2, 0.4, 0.6, 0.8]
            : segW >= ringStep * 1.2
              ? [0.25, 0.5, 0.75]
              : [0.5];
        const rowMinSq = Math.max(1.6, rowStep * 0.48) ** 2;
        for (const f of fracs) {
          pushIfFar(segStart + 0.5 + (segW - 1) * f, iy + 0.5, rowMinSq, null);
        }
        x++;
      }
      y += rowStep;
    }
  } else if (role === "face" && lm) {
    const midX = (lm.eyeL.x + lm.eyeR.x) * 0.5;
    const eyeD = Math.max(8, Math.abs(lm.eyeL.x - lm.eyeR.x));
    const steps = Math.round(clamp(bh / (ringStep * 0.65), 5, 18));
    for (let s = 1; s < steps; s++) {
      const fy = top + (s / steps) * bh;
      const t = clamp(
        (fy - lm.crown.y) / Math.max(1, lm.chin.y - lm.crown.y),
        0,
        1,
      );
      pushIfFar(
        midX + (lm.chin.x - midX) * t,
        fy,
        guideMinSq * 0.75,
        null,
      );
    }
    for (const pt of [lm.chin, lm.mouth, lm.eyeL, lm.eyeR, lm.crown]) {
      pushIfFar(pt.x, pt.y, guideMinSq * 0.5, null);
    }
    for (const c of [lm.eyeR, lm.eyeL, lm.mouth]) {
      const r = Math.max(eyeD * 0.22, ringStep * 0.7);
      for (let s = 0; s < 6; s++) {
        const ang = (s / 6) * Math.PI * 2;
        pushIfFar(
          c.x + r * Math.cos(ang),
          c.y + r * 0.75 * Math.sin(ang),
          guideMinSq * 0.75,
          null,
        );
      }
    }
  } else if (role === "structuralBody" && lm) {
    const midX = (lm.eyeL.x + lm.eyeR.x) * 0.5;
    const steps = Math.round(clamp(bh / (ringStep * 0.75), 4, 16));
    for (let s = 1; s < steps; s++) {
      pushIfFar(midX, top + (s / steps) * bh, guideMinSq * 0.75, null);
    }
  } else if (role === "generic") {
    for (const a of anchors) pushIfFar(a.x, a.y, guideMinSq * 0.5, null);
  }

  let maxD = 10;
  for (let i = 0; i < distMap.length; i++) {
    if (distMap[i] !== 0xffff && distMap[i] > maxD) maxD = distMap[i];
  }
  const maxDPx = maxD / 10;
  const bandGap = Math.max(2, ringStep * 0.866);
  let bandD = Math.max(2.5, ringStep * 1.15);
  let bandIdx = 0;
  const stride = Math.max(1, Math.floor(ringStep * 0.45));
  while (bandD <= maxDPx + 0.5) {
    const loD = Math.floor(Math.max(0.5, bandD - 0.65) * 10);
    const hiD = Math.floor((bandD + 0.65) * 10);
    const stagger = bandIdx % 2 === 1 ? (0.5 * ringStep) % 1 : 0;
    for (let ly = 0; ly < bh; ly += stride) {
      for (let lx = 0; lx < bw; lx += stride) {
        const dVal = distMap[ly * bw + lx];
        if (dVal !== 0xffff && dVal >= loD && dVal <= hiD) {
          pushIfFar(
            left + lx + 0.5 + stagger,
            top + ly + 0.5,
            guideMinSq,
            null,
          );
        }
      }
    }
    bandD += bandGap;
    bandIdx++;
  }

  const fillStride = Math.max(2, Math.round(ringStep * 0.85));
  let rowI = 0;
  for (let y = top + Math.floor(fillStride / 2); y <= bottom; y += fillStride) {
    const xOff = rowI % 2 === 1 ? Math.floor(fillStride / 2) : 0;
    for (let x = left + xOff; x <= right; x += fillStride) {
      if (mask[y * w + x] > 8) {
        pushIfFar(x + 0.5, y + 0.5, guideMinSq, null);
      }
    }
    rowI++;
  }

  if (points.length < 3) {
    return { vertices: [], triangles: [], uv: [] };
  }
  const d = Delaunator.from(
    points,
    (p) => p.x,
    (p) => p.y,
  );
  const tris: number[] = [];
  const thinPart = points.length < 40;
  const maxEdge = thinPart
    ? Math.max(18, span * 0.9)
    : Math.max(18, ringStep * 2.75);
  const minCov = thinPart ? 52 : 68;
  for (let t = 0; t < d.triangles.length; t += 3) {
    const ids = [d.triangles[t], d.triangles[t + 1], d.triangles[t + 2]];
    const a = points[ids[0]],
      b = points[ids[1]],
      c = points[ids[2]];
    const edgeLen = Math.max(
      Math.hypot(a.x - b.x, a.y - b.y),
      Math.hypot(a.x - c.x, a.y - c.y),
      Math.hypot(b.x - c.x, b.y - c.y),
    );
    if (edgeLen > maxEdge) continue;
    const minAspect = thinPart ? 0.025 : 0.048;
    const cross = Math.abs(
      (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x),
    );
    if (cross < minAspect * edgeLen * edgeLen) continue;
    const cx = (a.x + b.x + c.x) / 3,
      cy = (a.y + b.y + c.y) / 3;
    if (
      !valid(cx, cy) ||
      !valid(
        a.x * 0.5 + b.x * 0.25 + c.x * 0.25,
        a.y * 0.5 + b.y * 0.25 + c.y * 0.25,
      ) ||
      !valid(
        a.x * 0.25 + b.x * 0.5 + c.x * 0.25,
        a.y * 0.25 + b.y * 0.5 + c.y * 0.25,
      ) ||
      !valid(
        a.x * 0.25 + b.x * 0.25 + c.x * 0.5,
        a.y * 0.25 + b.y * 0.25 + c.y * 0.5,
      )
    )
      continue;
    const steps = 4;
    let total = 0,
      validSamples = 0;
    for (let u = 0; u <= steps; u++)
      for (let v = 0; v <= steps - u; v++) {
        total++;
        const x = a.x + ((b.x - a.x) * u) / steps + ((c.x - a.x) * v) / steps,
          y = a.y + ((b.y - a.y) * u) / steps + ((c.y - a.y) * v) / steps;
        if (valid(x, y)) validSamples++;
      }
    if (validSamples * 100 >= total * minCov) tris.push(...ids);
  }

  const remap = new Int32Array(points.length).fill(-1);
  const compactVerts: Point[] = [];
  for (let i = 0; i < tris.length; i++) {
    const old = tris[i];
    if (remap[old] === -1) {
      remap[old] = compactVerts.length;
      compactVerts.push(points[old]);
    }
    tris[i] = remap[old];
  }
  return {
    vertices: compactVerts,
    triangles: tris,
    uv: compactVerts.map((p) => ({ x: p.x / w, y: p.y / h })),
  };
}

/** Contour- and landmark-aware Delaunay ArtMesh matching Rust `core::mesh_for_with_anchors`. */
export function meshFor(
  mask: Uint8Array,
  w: number,
  h: number,
  density = 14,
  anchors: Point[] = [],
): Mesh {
  return buildArtMesh(mask, w, h, density, "generic", undefined, anchors);
}

export function meshForPart(
  mask: Uint8Array,
  w: number,
  h: number,
  density = 14,
  lm?: Landmarks,
  part?: Pick<Part, "id" | "role"> | string,
): Mesh {
  const partId = typeof part === "string" ? part : (part?.id ?? "face");
  return buildArtMesh(
    mask,
    w,
    h,
    density,
    classifyArtMeshRole(partId),
    lm,
    [],
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
  const zero: Parameters = { ...REST };
  const extremePoses: Parameters[] = [
    { ...zero, angleX: -30, angleY: -30, angleZ: -30 },
    { ...zero, angleX: 30, angleY: 30, angleZ: 30 },
    { ...zero, angleX: -30, angleY: 30, angleZ: 15, bodyAngleX: -10, bodyAngleZ: -10 },
    { ...zero, angleX: 30, angleY: -30, angleZ: -15, bodyAngleX: 10, bodyAngleZ: 10 },
    { ...zero, eyeL: 0, eyeR: 0, eyeLSmile: 1, eyeRSmile: 1, eyeBallX: -1, eyeBallY: 1 },
    { ...zero, eyeL: 1.2, eyeR: 1.2, eyeBallX: 1, eyeBallY: -1 },
    { ...zero, mouth: 0, mouthForm: -1, browLY: -1, browLForm: -1, browRY: -1, browRForm: -1 },
    { ...zero, mouth: 1, mouthForm: 1, browLY: 1, browLForm: 1, browRY: 1, browRForm: 1 },
    { ...zero, hair: -1, hairSide: -1, hairBack: -1 },
    { ...zero, hair: 1, hairSide: 1, hairBack: 1, breath: 1 },
    {
      ...zero,
      angleX: 30,
      angleY: -30,
      angleZ: 30,
      eyeL: 0,
      eyeR: 0,
      mouth: 1,
      mouthForm: 1,
      bodyAngleX: 10,
      bodyAngleY: -10,
      bodyAngleZ: 10,
      breath: 1,
      hair: -1,
      hairSide: -1,
      hairBack: -1,
    },
    {
      ...zero,
      angleX: -30,
      angleY: 30,
      angleZ: -30,
      eyeL: 0,
      eyeR: 0,
      mouth: 1,
      mouthForm: -1,
      bodyAngleX: -10,
      bodyAngleY: 10,
      bodyAngleZ: -10,
      breath: 0,
      hair: 1,
      hairSide: 1,
      hairBack: 1,
    },
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

function applyBodySway(
  x: number,
  y: number,
  bodyPivotX: number,
  bodyPivotY: number,
  headH: number,
  params: Parameters,
  scale: number,
): Point {
  if (
    params.bodyAngleX === 0 &&
    params.bodyAngleY === 0 &&
    params.bodyAngleZ === 0 &&
    params.breath === 0
  ) {
    return { x, y };
  }
  const liftWeight = clamp((bodyPivotY - y) / (headH * 1.35), 0, 1.35);
  const sx = x + params.bodyAngleX * headH * 0.0022 * liftWeight * scale;
  const sy =
    y +
    (params.bodyAngleY * headH * 0.0014 - params.breath * headH * 0.015) *
      liftWeight *
      scale;
  const angle = ((params.bodyAngleZ * 0.45 * scale) * Math.PI) / 180;
  const dx = sx - bodyPivotX;
  const dy = sy - bodyPivotY;
  return {
    x: bodyPivotX + dx * Math.cos(angle) - dy * Math.sin(angle),
    y: bodyPivotY + dx * Math.sin(angle) + dy * Math.cos(angle),
  };
}

export interface RegionDeformParams {
  width: number;
  roundnessDelta: number;
  corner: number;
  upperCurveDelta: number;
  lowerCurveDelta: number;
  offsetX: number;
  offsetY: number;
}

export interface ShapeDeformInput {
  eyeL?: RegionDeformParams;
  eyeR?: RegionDeformParams;
  mouth?: RegionDeformParams;
}

export function deform(
  p: Point,
  part: Pick<Part, "id" | "role">,
  lm: Landmarks,
  params: Parameters,
  shapeDeform?: ShapeDeformInput,
): Point {
  let { x, y } = p;
  const cx = (lm.eyeL.x + lm.eyeR.x) / 2,
    headH = Math.max(1, lm.chin.y - lm.crown.y),
    eyeDist = Math.max(1, Math.hypot(lm.eyeL.x - lm.eyeR.x, lm.eyeL.y - lm.eyeR.y)),
    bodyPivotX = cx,
    bodyPivotY = lm.chin.y + headH * 1.15;
  const role = inferPartRole(part.id, part.role);
  if (
    part.id === "body" ||
    role === "clothes" ||
    role === "hands" ||
    part.id.startsWith("clothes") ||
    part.id === "hands" ||
    part.id.startsWith("hands_")
  ) {
    return applyBodySway(x, y, bodyPivotX, bodyPivotY, headH, params, 1);
  }
  const isSubLayer = part.id.includes("_sub_");
  if (part.id.startsWith("eye")) {
    const isLeft = part.id.startsWith("eyeL"),
      side = isLeft ? "L" : "R",
      eye = lm[`eye${side}`],
      open = params[`eye${side}`],
      smile = params[`eye${side}Smile`];
    const shape = isLeft ? shapeDeform?.eyeL : shapeDeform?.eyeR;
    if (shape) {
      const isIris = part.id.endsWith("_iris");
      x += shape.offsetX * eyeDist * 0.14;
      y += shape.offsetY * eyeDist * 0.12;
      const widthScale = isIris ? 1 + (shape.width - 1) * 0.45 : shape.width;
      x = eye.x + (x - eye.x) * widthScale;
      const nx = (x - eye.x) / Math.max(1, eyeDist * 0.24);
      const centerArch = Math.max(0, 1 - nx * nx);
      const eyeSign = isLeft ? 1 : -1;
      const outerSigned = clamp(nx * eyeSign, -0.4, 1.2);
      y -= shape.corner * eyeDist * 0.045 * outerSigned;
      const upperMask = part.id.endsWith("_upper") || part.id.endsWith("_sub_lid")
        ? 1
        : clamp((eye.y + eyeDist * 0.06 - y) / Math.max(1, eyeDist * 0.14), 0, 1);
      const lowerMask = part.id.endsWith("_lower")
        ? 1
        : clamp((y - (eye.y - eyeDist * 0.02)) / Math.max(1, eyeDist * 0.14), 0, 1);
      y -= shape.upperCurveDelta * eyeDist * 0.055 * centerArch * upperMask;
      y += shape.lowerCurveDelta * eyeDist * 0.045 * centerArch * lowerMask;
      if (!isIris) {
        y = eye.y + (y - eye.y) * (1 + shape.roundnessDelta * 0.32 * centerArch);
      }
    }
    if (!isSubLayer) {
      if (part.id.endsWith("_iris")) {
        x += params.eyeBallX * eyeDist * 0.045;
        y -= params.eyeBallY * eyeDist * 0.025;
        y = eye.y + (y - eye.y) * clamp(open, 0.04, 1.3);
      } else if (part.id.endsWith("_upper")) {
        const close = clamp(1 - open, 0, 1);
        const widen = clamp(open - 1, 0, 0.3);
        const lidAnchorY = eye.y - eyeDist * 0.34;
        const closeY = eye.y + eyeDist * 0.04;
        const spanY = Math.max(1, closeY - lidAnchorY);
        const u = clamp((y - lidAnchorY) / spanY, 0, 1);
        if (close > 0 && u > 0 && u < 1) {
          const targetU = 1 - Math.pow(1 - u, 3.6);
          const closedY = lidAnchorY + spanY * targetU;
          y += (closedY - y) * close;
        } else if (widen > 0 && u > 0) {
          y -= spanY * 0.18 * u * widen;
        }
      } else {
        y = eye.y + (y - eye.y) * Math.max(0.04, open);
        if (part.id.endsWith("_lower")) {
          y -= smile * eyeDist * 0.032;
        }
      }
    }
  }
  if (part.id.startsWith("brow")) {
    const isL = part.id.startsWith("browL");
    const browY = isL ? params.browLY : params.browRY;
    const browForm = isL ? params.browLForm : params.browRForm;
    y -= browY * headH * 0.045;
    const innerFactor = 1 - clamp(Math.abs(x - cx) / (eyeDist * 0.65), 0, 1);
    y -= browForm * headH * 0.028 * innerFactor;
  }
  if (part.id.startsWith("mouth")) {
    const mShape = shapeDeform?.mouth;
    if (mShape) {
      x += mShape.offsetX * eyeDist * 0.12;
      y += mShape.offsetY * headH * 0.04;
      x = lm.mouth.x + (x - lm.mouth.x) * mShape.width;
      const mnx = (x - lm.mouth.x) / Math.max(1, eyeDist * 0.16);
      const mCenter = Math.max(0, 1 - mnx * mnx);
      const mCornerW = clamp(Math.abs(mnx), 0, 1.2);
      y -= mShape.corner * headH * 0.028 * mCornerW * mCornerW;
      y = lm.mouth.y + (y - lm.mouth.y) * (1 + mShape.roundnessDelta * 0.3 * mCenter);
      const upperW =
        part.id === "mouth_upper" || part.id === "mouth_sub_lip"
          ? 1
          : clamp((lm.mouth.y + headH * 0.02 - y) / Math.max(1, headH * 0.04), 0, 1);
      const lowerW =
        part.id === "mouth_lower" || part.id === "mouth_sub_lip"
          ? 1
          : clamp((y - (lm.mouth.y - headH * 0.02)) / Math.max(1, headH * 0.04), 0, 1);
      y -= mShape.upperCurveDelta * headH * 0.024 * mCenter * upperW;
      y += mShape.lowerCurveDelta * headH * 0.026 * mCenter * lowerW;
    }
    if (!isSubLayer) {
      const mdx = x - lm.mouth.x;
      const cornerW = clamp(Math.abs(mdx) / Math.max(1, eyeDist * 0.18), 0, 1.2);
      x += mdx * params.mouthForm * 0.14;
      y -= params.mouthForm * headH * 0.016 * cornerW * cornerW;
      if (part.id === "mouth_upper") y -= params.mouth * headH * 0.012;
      if (part.id === "mouth_lower") y += params.mouth * headH * 0.045;
      if (part.id === "mouth_inner")
        y = lm.mouth.y + (y - lm.mouth.y) * (1 + params.mouth * 2);
    }
  }
  const weight =
    part.id === "neck"
      ? clamp((lm.chin.y + headH * 0.42 - y) / (headH * 0.42), 0, 1)
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
    const roleParam =
      role === "hair_side_L" || role === "hair_side_R"
        ? params.hairSide
        : role === "hair_back"
          ? params.hairBack
          : 0;
    const swayParam = clamp(params.hair + roleParam, -1.5, 1.5);
    const rootY = lm.crown.y + headH * rootOffset;
    x +=
      swayParam *
      headH *
      0.04 *
      amp *
      clamp((y - rootY) / headH, 0, 1.8);
  } else if (role === "hat" || part.id.startsWith("hat")) {
    const swayParam = clamp(params.hair, -1.5, 1.5);
    const rootY = lm.crown.y + headH * 0.45;
    x +=
      swayParam *
      headH *
      0.04 *
      0.18 *
      clamp((y - rootY) / headH, 0, 1.8);
  } else if (
    part.id === "acc_hairpin" ||
    part.id === "acc_ribbon" ||
    part.id.startsWith("acc_ribbon")
  ) {
    const swayParam = clamp(params.hair, -1.5, 1.5);
    x +=
      swayParam *
      headH *
      0.04 *
      0.45 *
      clamp((y - lm.crown.y) / headH, 0, 1.8);
  }
  const rotWeight = part.id === "neck" ? weight * 0.78 : weight;
  const angle = ((params.angleZ * Math.PI) / 180) * rotWeight,
    dx = x - cx,
    dy = y - lm.chin.y;
  const headX = cx + dx * Math.cos(angle) - dy * Math.sin(angle);
  const headY = lm.chin.y + dx * Math.sin(angle) + dy * Math.cos(angle);
  const bodyFollow = part.id === "neck" ? 0.45 + 0.55 * weight : 1;
  return applyBodySway(
    headX,
    headY,
    bodyPivotX,
    bodyPivotY,
    headH,
    params,
    bodyFollow,
  );
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
