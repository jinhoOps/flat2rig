/**
 * B1: builds flat2rig parts from a parsed PSD and a layer → part mapping.
 *
 * Definition (same as `src-tauri/src/analysis/layer_input.rs`, guarded by
 * `tests/fixtures/layer-split-parity.json` on both sides):
 * - a part's texture = its layers composited bottom → top in PSD order (straight RGBA);
 * - visible `mask` = the part alpha not covered by parts above, `a_p · Π_{q above p}(1 − a_q)`
 *   (alpha-weighted, anti-aliased inner edges are shared, Σ masks = composite alpha);
 * - `fillMask` = the part alpha wherever it is (partly) covered, so `effectiveMask` = part alpha.
 *
 * Part order = PSD order: a part sits at the position of its topmost included layer. When a part's
 * layers are interleaved with another part's layers (Himikan `clothes` = 카라 뒤 … 몸통 … 카라 앞
 * around `neck`), the part-level stack cannot reproduce the PSD exactly; `interleaved` reports it.
 *
 * Inside a part: layer opacity (× group opacity) and layer masks apply; a clipped layer is clipped
 * to its base when the base is in the same part, otherwise it keeps its own alpha (the renderer clips
 * iris → white itself); `fold` layers apply their blend (multiply / screen / linear dodge, W3C) to
 * the part's layers below. Downscaling is PIL BOX resampling of the premultiplied part, the same
 * transform as `tools/evaluation/himikan_truth.py` (x' = x · W'/W); with the Himikan truth map at
 * 683×1024 every part matches `artifacts/t14/himikan-truth` within one alpha level.
 */
import {
  effectiveBlend,
  expandTarget,
  isFoldable,
  maskAt,
  sideOf,
  type LayerMapping,
  type ParsedPsd,
  type PsdLayerNode,
  type PsdLayerPixels,
} from "./psdLayers";

/** Same long-side rule as `readImage` (illustration input). */
export const WORK_MAX_SIDE = 2048;

export function workingSize(width: number, height: number, maxSide = WORK_MAX_SIDE) {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

export interface BuiltLayerPart {
  id: string;
  /** Source layer paths, bottom → top (a `{S}` split layer appears in both sides' lists). */
  layers: string[];
  /** Topmost source layer order (the part's position in the stack). */
  order: number;
  /** Straight RGBA crop at working resolution. */
  left: number;
  top: number;
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
}

export interface LayerPartsBuild {
  width: number;
  height: number;
  sx: number;
  sy: number;
  /** Bottom → top. */
  parts: BuiltLayerPart[];
  /** Full-canvas visible / hidden masks per part (same order). */
  masks: Uint8Array[];
  fills: Uint8Array[];
  /** Full-canvas straight composite of the parts (what the rig shows at rest). */
  composite: Uint8ClampedArray;
  maskSum: MaskSumCheck;
  /** Parts whose layers are interleaved with another part's layers in the PSD. */
  interleaved: { id: string; between: string[] }[];
  /** Layers whose clip base is in another part or excluded (kept own alpha). */
  clipKeptOwnAlpha: string[];
  /** `{S}` layers split into L/R: component counts per side. */
  splits: { path: string; L: number; R: number }[];
  midlineX: number;
}

export interface MaskSumCheck {
  foregroundPx: number;
  sharedVisiblePx: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
  pxOverOneLevel: number;
  matches: boolean;
}

// ---------------------------------------------------------------------------------------------
// Rust-parity primitives (layer_input.rs `split_visible_fill`, core.rs `blend_straight_pixel`)
// ---------------------------------------------------------------------------------------------

/** Visible (uncovered) and fill (covered) masks from per-part alpha, bottom → top. */
export function splitVisibleFill(alphas: readonly Uint8Array[], n: number) {
  const layers = alphas.length;
  const masks = alphas.map(() => new Uint8Array(n));
  const fills = alphas.map(() => new Uint8Array(n));
  for (let i = 0; i < n; i++) {
    let uncovered = 1.0;
    for (let p = layers - 1; p >= 0; p--) {
      const a8 = alphas[p][i];
      if (a8 === 0) continue;
      const a = a8 / 255.0;
      const vis = Math.min(roundHalfAway(a * uncovered * 255.0), a8);
      masks[p][i] = vis;
      if (vis < a8) fills[p][i] = a8;
      uncovered *= 1.0 - a;
    }
  }
  return { masks, fills };
}

/** Rust `f64::round` / `f32::round` (half away from zero). */
function roundHalfAway(v: number) {
  return v < 0 ? -Math.round(-v) : Math.round(v);
}

const f = Math.fround;

/** Rust `core::blend_straight_pixel` (f32 arithmetic) on `dst[k..k+4]`. */
export function blendStraightPixel(
  dst: Uint8ClampedArray | Uint8Array,
  k: number,
  r: number,
  g: number,
  b: number,
  a8: number,
) {
  const sa = f(a8 / 255);
  if (sa <= 0) return;
  const da = f(dst[k + 3] / 255);
  const inv = f(1 - sa);
  const outA = f(sa + f(da * inv));
  if (outA <= 1e-6) return;
  const dW = f(da * inv);
  const src = [r, g, b];
  for (let c = 0; c < 3; c++) {
    const blended = f(f(f(src[c] * sa) + f(dst[k + c] * dW)) / outA);
    dst[k + c] = Math.min(255, Math.max(0, roundHalfAway(blended)));
  }
  dst[k + 3] = Math.min(255, Math.max(0, roundHalfAway(f(outA * 255))));
}

/** Rust `layer_input::composite_layers` for full-canvas straight RGBA layers. */
export function compositeStraight(layers: readonly (Uint8Array | Uint8ClampedArray)[], n: number) {
  const out = new Uint8ClampedArray(n * 4);
  for (const l of layers) {
    for (let i = 0; i < n; i++) {
      const k = i * 4;
      if (l[k + 3] > 0) blendStraightPixel(out, k, l[k], l[k + 1], l[k + 2], l[k + 3]);
    }
  }
  return out;
}

/** Rust `layer_input::mask_sum_check`. */
export function maskSumCheck(masks: readonly Uint8Array[], compositeAlpha: (i: number) => number, n: number): MaskSumCheck {
  const c: MaskSumCheck = {
    foregroundPx: 0,
    sharedVisiblePx: 0,
    maxAbsDiff: 0,
    meanAbsDiff: 0,
    pxOverOneLevel: 0,
    matches: false,
  };
  let total = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0,
      contributors = 0;
    for (const m of masks) {
      if (m[i] > 0) {
        sum += m[i];
        contributors++;
      }
    }
    const alpha = compositeAlpha(i);
    if (alpha > 0) c.foregroundPx++;
    if (contributors > 1) c.sharedVisiblePx++;
    const d = Math.abs(sum - alpha);
    total += d;
    c.maxAbsDiff = Math.max(c.maxAbsDiff, d);
    if (d > 1) c.pxOverOneLevel++;
  }
  c.meanAbsDiff = n > 0 ? total / n : 0;
  c.matches = c.maxAbsDiff <= Math.max(1, masks.length);
  return c;
}

// ---------------------------------------------------------------------------------------------
// Part assembly
// ---------------------------------------------------------------------------------------------

interface Piece {
  node: PsdLayerNode;
  px: PsdLayerPixels;
  /** For `{S}` splits: per crop pixel, 1 = this piece's side. */
  side?: Uint8Array;
  blend: string;
  fold: boolean;
}

/** Connected components (8-neighbour) of a layer's alpha, each assigned to a character side by its
 * alpha-weighted centroid x against `midlineX`. Returns per-pixel side (1 = L, 0 = R) and counts. */
export function splitComponentsLR(px: PsdLayerPixels, midlineX: number) {
  const { width: w, height: h } = px;
  const label = new Int32Array(w * h).fill(-1);
  const comps: { sum: number; sx: number }[] = [];
  const stack: number[] = [];
  for (let s = 0; s < w * h; s++) {
    if (label[s] !== -1 || px.rgba[s * 4 + 3] === 0) continue;
    const id = comps.length;
    const comp = { sum: 0, sx: 0 };
    label[s] = id;
    stack.push(s);
    while (stack.length) {
      const i = stack.pop()!;
      const x = i % w,
        y = (i - x) / w;
      const a = px.rgba[i * 4 + 3];
      comp.sum += a;
      comp.sx += a * (px.left + x + 0.5);
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w || (dx === 0 && dy === 0)) continue;
          const j = yy * w + xx;
          if (label[j] === -1 && px.rgba[j * 4 + 3] > 0) {
            label[j] = id;
            stack.push(j);
          }
        }
      }
    }
    comps.push(comp);
  }
  const compSide = comps.map((c) => (sideOf(c.sx / Math.max(1, c.sum), midlineX) === "L" ? 1 : 0));
  const side = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (label[i] >= 0) side[i] = compSide[label[i]];
  return {
    side,
    L: compSide.filter((v) => v === 1).length,
    R: compSide.filter((v) => v === 0).length,
  };
}

function blendChannel(mode: string, cb: number, cs: number) {
  switch (mode) {
    case "multiply":
      return cs * cb;
    case "screen":
      return cb + cs - cb * cs;
    case "linear dodge":
      return Math.min(1, cb + cs);
    default:
      return cs;
  }
}

/**
 * PIL `Image.resize(..., Image.BOX)` coefficients (`precompute_coeffs`, support 0.5 × scale): an
 * output pixel averages the input pixels whose centres fall in its footprint, normalised over the
 * whole input row (so a crop must know the full input size). Restricted to the input span
 * [offset, offset + len) of a crop; pixels outside it are transparent.
 */
function axisWeights(offset: number, len: number, inSize: number, outSize: number) {
  const scale = inSize / outSize;
  const support = 0.5 * Math.max(1, scale);
  const ss = 1 / Math.max(1, scale);
  const first = Math.max(0, Math.floor((offset - support) / scale) - 1);
  const last = Math.min(outSize, Math.ceil((offset + len + support) / scale) + 1);
  const entries: { t: number; s: number; w: number }[] = [];
  let t0 = Infinity,
    t1 = -Infinity;
  for (let t = first; t < last; t++) {
    const center = (t + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
    let ww = 0;
    const ws: number[] = [];
    for (let x = xmin; x < xmax; x++) {
      const v = (x - center + 0.5) * ss;
      const w = v > -0.5 && v <= 0.5 ? 1 : 0;
      ws.push(w);
      ww += w;
    }
    if (ww === 0) continue;
    for (let j = 0; j < ws.length; j++) {
      const sIdx = xmin + j - offset;
      if (!ws[j] || sIdx < 0 || sIdx >= len) continue;
      entries.push({ t, s: sIdx, w: ws[j] / ww });
      t0 = Math.min(t0, t);
      t1 = Math.max(t1, t + 1);
    }
  }
  if (!entries.length) return { t0: 0, size: 0, entries };
  for (const e of entries) e.t -= t0;
  return { t0, size: t1 - t0, entries };
}

/** PIL BOX resample of a premultiplied float crop to the working grid (horizontal, then vertical). */
function resamplePremul(
  src: Float32Array,
  w: number,
  h: number,
  left: number,
  top: number,
  inW: number,
  inH: number,
  outW: number,
  outH: number,
) {
  const wx = axisWeights(left, w, inW, outW);
  const wy = axisWeights(top, h, inH, outH);
  const tw = wx.size,
    th = wy.size;
  const tmp = new Float32Array(tw * h * 4);
  for (let y = 0; y < h; y++) {
    for (const e of wx.entries) {
      const si = (y * w + e.s) * 4,
        ti = (y * tw + e.t) * 4;
      tmp[ti] += src[si] * e.w;
      tmp[ti + 1] += src[si + 1] * e.w;
      tmp[ti + 2] += src[si + 2] * e.w;
      tmp[ti + 3] += src[si + 3] * e.w;
    }
  }
  const out = new Float32Array(tw * th * 4);
  for (const e of wy.entries) {
    const srow = e.s * tw * 4,
      trow = e.t * tw * 4;
    for (let x = 0; x < tw * 4; x++) out[trow + x] += tmp[srow + x] * e.w;
  }
  return { data: out, left: wx.t0, top: wy.t0, width: tw, height: th };
}

/** `himikan_truth.py unpremultiply_u8`. */
function unpremultiplyU8(p: Float32Array) {
  const out = new Uint8ClampedArray(p.length);
  for (let k = 0; k < p.length; k += 4) {
    const a = p[k + 3];
    if (a > 1e-6) {
      for (let c = 0; c < 3; c++) out[k + c] = Math.floor(Math.min(1, Math.max(0, p[k + c] / a)) * 255 + 0.5);
    }
    out[k + 3] = Math.floor(Math.min(1, Math.max(0, a)) * 255 + 0.5);
  }
  return out;
}

function compositePart(pieces: Piece[], canvasW: number, canvasH: number) {
  let l = Infinity,
    t = Infinity,
    r = -Infinity,
    b = -Infinity;
  for (const p of pieces) {
    l = Math.min(l, p.px.left);
    t = Math.min(t, p.px.top);
    r = Math.max(r, p.px.left + p.px.width);
    b = Math.max(b, p.px.top + p.px.height);
  }
  l = Math.max(0, l);
  t = Math.max(0, t);
  r = Math.min(canvasW, r);
  b = Math.min(canvasH, b);
  const w = Math.max(0, r - l),
    h = Math.max(0, b - t);
  const acc = new Float32Array(w * h * 4);
  const inPart = new Set(pieces.map((p) => p.node.path));
  const neededBases = new Set(
    pieces.filter((p) => p.node.clipping && p.node.clipBase && inPart.has(p.node.clipBase)).map((p) => p.node.clipBase!),
  );
  const baseAlpha = new Map<string, Float32Array>();
  const keptOwn: string[] = [];
  for (const piece of pieces) {
    const { node, px } = piece;
    const clipTo =
      node.clipping && node.clipBase && inPart.has(node.clipBase) ? baseAlpha.get(node.clipBase) : undefined;
    if (node.clipping && !clipTo) keptOwn.push(node.path);
    const own = neededBases.has(node.path) ? new Float32Array(w * h) : null;
    const op = node.effectiveOpacity;
    const mode = piece.fold ? piece.blend : "normal";
    for (let y = 0; y < px.height; y++) {
      const cy = px.top + y - t;
      if (cy < 0 || cy >= h) continue;
      for (let x = 0; x < px.width; x++) {
        const cx = px.left + x - l;
        if (cx < 0 || cx >= w) continue;
        const si = y * px.width + x;
        let a = px.rgba[si * 4 + 3] / 255;
        if (a === 0) continue;
        if (px.mask) a *= maskAt(px, px.left + x, px.top + y) / 255;
        if (piece.side && !piece.side[si]) continue;
        const ci = cy * w + cx;
        if (own) own[ci] = a;
        if (clipTo) a *= clipTo[ci];
        const sa = a * op;
        if (sa <= 0) continue;
        const k = ci * 4;
        const da = acc[k + 3];
        for (let c = 0; c < 3; c++) {
          const cs = px.rgba[si * 4 + c] / 255;
          const dp = acc[k + c];
          if (mode === "normal") acc[k + c] = sa * cs + (1 - sa) * dp;
          else {
            const cb = da > 1e-6 ? dp / da : 0;
            acc[k + c] = (1 - da) * sa * cs + (1 - sa) * dp + sa * da * blendChannel(mode, cb, cs);
          }
        }
        acc[k + 3] = sa + da * (1 - sa);
      }
    }
    if (own) baseAlpha.set(node.path, own);
  }
  return { acc, left: l, top: t, width: w, height: h, keptOwn };
}

export interface BuildOptions {
  /** Working canvas size; default = `workingSize(psd)`. */
  width?: number;
  height?: number;
  /** Face midline at PSD resolution for `{S}` splits. */
  midlineX: number;
}

/** Groups included, assigned layers into parts (bottom → top by topmost layer). */
export function planParts(psd: ParsedPsd, mappings: readonly LayerMapping[]) {
  const byPath = new Map(psd.layers.map((n) => [n.path, n]));
  const parts = new Map<string, { id: string; paths: string[]; order: number; splitPaths: Set<string> }>();
  for (const m of mappings) {
    const node = byPath.get(m.path);
    if (!node || !m.include || !m.target || !psd.pixels.has(m.path)) continue;
    const split = m.target.includes("{S}");
    for (const id of expandTarget(m.target)) {
      const e = parts.get(id) ?? { id, paths: [], order: -1, splitPaths: new Set<string>() };
      e.paths.push(m.path);
      e.order = Math.max(e.order, node.order);
      if (split) e.splitPaths.add(m.path);
      parts.set(id, e);
    }
  }
  const list = [...parts.values()];
  for (const e of list) e.paths.sort((a, b) => byPath.get(a)!.order - byPath.get(b)!.order);
  list.sort((a, b) => a.order - b.order || a.id.localeCompare(b.id));
  return list;
}

/** Character side encoded in a part id (`eyeL_iris`, `browR`, `hair_side_L_2`). */
export function partSide(id: string): "L" | "R" | undefined {
  const m = /^(?:eye|brow)([LR])|^hair_side_([LR])/.exec(id);
  return m ? ((m[1] ?? m[2]) as "L" | "R") : undefined;
}

function sideless(id: string) {
  return id.replace(/^(eye|brow)[LR]/, "$1{S}").replace(/^hair_side_[LR]/, "hair_side_{S}");
}

/** Parts whose layer span contains another part's layer (part-level stack ≠ PSD stack). */
function findInterleaved(psd: ParsedPsd, plan: ReturnType<typeof planParts>) {
  const byPath = new Map(psd.layers.map((n) => [n.path, n]));
  const owner = new Map<number, string>();
  for (const p of plan) for (const path of p.paths) owner.set(byPath.get(path)!.order, p.id);
  const out: { id: string; between: string[] }[] = [];
  for (const p of plan) {
    const orders = p.paths.map((path) => byPath.get(path)!.order);
    const lo = Math.min(...orders),
      hi = Math.max(...orders);
    const between = new Set<string>();
    for (let o = lo + 1; o < hi; o++) {
      const id = owner.get(o);
      if (id && id !== p.id && sideless(id) !== sideless(p.id)) between.add(id);
    }
    if (between.size) out.push({ id: p.id, between: [...between] });
  }
  return out;
}

/** Builds part RGBA crops, visible / hidden masks and the composite. */
export function buildLayerParts(
  psd: ParsedPsd,
  mappings: readonly LayerMapping[],
  options: BuildOptions,
  onProgress?: (done: number, total: number, step: string) => void,
): LayerPartsBuild {
  const size = options.width && options.height
    ? { width: options.width, height: options.height }
    : workingSize(psd.width, psd.height);
  const W = size.width,
    H = size.height,
    n = W * H;
  const sx = W / psd.width,
    sy = H / psd.height;
  const byPath = new Map(psd.layers.map((x) => [x.path, x]));
  const mapByPath = new Map(mappings.map((m) => [m.path, m]));
  const plan = planParts(psd, mappings);
  const splitCache = new Map<string, ReturnType<typeof splitComponentsLR>>();
  const splits: LayerPartsBuild["splits"] = [];
  const parts: BuiltLayerPart[] = [];
  const keptOwn = new Set<string>();
  plan.forEach((entry, idx) => {
    onProgress?.(idx, plan.length, entry.id);
    const side = partSide(entry.id);
    const pieces: Piece[] = entry.paths.map((path) => {
      const node = byPath.get(path)!;
      const px = psd.pixels.get(path)!;
      const m = mapByPath.get(path)!;
      const blend = effectiveBlend(node);
      const piece: Piece = { node, px, blend, fold: blend !== "normal" && m.blend === "fold" && isFoldable(blend) };
      if (entry.splitPaths.has(path)) {
        let s = splitCache.get(path);
        if (!s) {
          s = splitComponentsLR(px, options.midlineX);
          splitCache.set(path, s);
          splits.push({ path, L: s.L, R: s.R });
        }
        const want = side === "L" ? 1 : 0;
        piece.side = s.side.map((v) => (v === want ? 1 : 0));
      }
      return piece;
    });
    const c = compositePart(pieces, psd.width, psd.height);
    for (const p of c.keptOwn) keptOwn.add(p);
    const scaled =
      sx === 1 && sy === 1
        ? { data: c.acc, left: c.left, top: c.top, width: c.width, height: c.height }
        : resamplePremul(c.acc, c.width, c.height, c.left, c.top, psd.width, psd.height, W, H);
    // Clamp the crop to the working canvas.
    const l = Math.max(0, scaled.left),
      t = Math.max(0, scaled.top);
    const r = Math.min(W, scaled.left + scaled.width),
      b = Math.min(H, scaled.top + scaled.height);
    const cw = Math.max(0, r - l),
      ch = Math.max(0, b - t);
    const u8 = unpremultiplyU8(scaled.data);
    const rgba = new Uint8ClampedArray(cw * ch * 4);
    for (let y = 0; y < ch; y++) {
      const srow = ((y + t - scaled.top) * scaled.width + (l - scaled.left)) * 4;
      rgba.set(u8.subarray(srow, srow + cw * 4), y * cw * 4);
    }
    parts.push({ id: entry.id, layers: entry.paths, order: entry.order, left: l, top: t, width: cw, height: ch, rgba });
  });
  onProgress?.(plan.length, plan.length, "가시/숨은 영역 분리");
  const alphas = parts.map((p) => {
    const a = new Uint8Array(n);
    for (let y = 0; y < p.height; y++)
      for (let x = 0; x < p.width; x++) a[(y + p.top) * W + x + p.left] = p.rgba[(y * p.width + x) * 4 + 3];
    return a;
  });
  const { masks, fills } = splitVisibleFill(alphas, n);
  const composite = new Uint8ClampedArray(n * 4);
  for (const p of parts) {
    for (let y = 0; y < p.height; y++) {
      for (let x = 0; x < p.width; x++) {
        const s = (y * p.width + x) * 4;
        if (p.rgba[s + 3] > 0)
          blendStraightPixel(composite, ((y + p.top) * W + x + p.left) * 4, p.rgba[s], p.rgba[s + 1], p.rgba[s + 2], p.rgba[s + 3]);
      }
    }
  }
  const maskSum = maskSumCheck(masks, (i) => composite[i * 4 + 3], n);
  return {
    width: W,
    height: H,
    sx,
    sy,
    parts,
    masks,
    fills,
    composite,
    maskSum,
    interleaved: findInterleaved(psd, plan),
    clipKeptOwnAlpha: [...keptOwn],
    splits,
    midlineX: options.midlineX,
  };
}

/** Full-canvas straight RGBA of one built part. */
export function partCanvasRgba(part: BuiltLayerPart, width: number, height: number) {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < part.height; y++) {
    const s = y * part.width * 4;
    out.set(part.rgba.subarray(s, s + part.width * 4), ((y + part.top) * width + part.left) * 4);
  }
  return out;
}
