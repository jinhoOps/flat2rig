import {
  compositePartTexture,
  deform,
  effectiveMask,
  eyeLayerSide,
  isEyeIrisLayer,
  meshForPart,
  rigLandmarks,
  type ShapeDeformInput,
} from "../engine";
import { canvas, loadImage, pixelsCanvas } from "../images";
import { base64 } from "../native";
import type {
  BlendMode,
  Mesh,
  Parameters,
  Part,
  Point,
  Project,
  ShapeSubstituteAsset,
  ShapeSubstituteLayer,
} from "../types";
import {
  evaluateRigFrame,
  type EvaluatedRigFrame,
  type EvaluateRigInput,
} from "./evaluate";
import { GlMeshCompositor, type GlLayerDrawSpec } from "./glRenderer";

export interface RenderScratch {
  scleraTargets: Map<string, HTMLCanvasElement>;
  iris: HTMLCanvasElement;
  meshCanvas?: HTMLCanvasElement;
  gl?: GlMeshCompositor | null;
}

export function createRenderScratch(
  w: number,
  h: number,
  enableGpu = false,
): RenderScratch {
  return {
    scleraTargets: new Map(),
    iris: canvas(w, h),
    gl: enableGpu ? GlMeshCompositor.create(w, h) : null,
  };
}

export function ensureRenderScratch(
  scratch: RenderScratch | undefined,
  w: number,
  h: number,
  enableGpu = false,
): RenderScratch {
  if (!scratch) {
    return createRenderScratch(w, h, enableGpu);
  }
  if (scratch.iris.width !== w || scratch.iris.height !== h) {
    scratch.iris.width = w;
    scratch.iris.height = h;
    scratch.scleraTargets.clear();
    scratch.gl?.resize(w, h);
  }
  if (enableGpu && scratch.gl === undefined) {
    scratch.gl = GlMeshCompositor.create(w, h);
  }
  return scratch;
}

export function drawTexturedTriangle(
  ctx: CanvasRenderingContext2D,
  texture: HTMLCanvasElement,
  src: Point[],
  dst: Point[],
) {
  const [a, b, c] = src;
  const [p, q, r] = dst;
  const det = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
  if (Math.abs(det) < 0.01) return;
  const aa = (p.x * (b.y - c.y) + q.x * (c.y - a.y) + r.x * (a.y - b.y)) / det;
  const bb = (p.y * (b.y - c.y) + q.y * (c.y - a.y) + r.y * (a.y - b.y)) / det;
  const cc = (p.x * (c.x - b.x) + q.x * (a.x - c.x) + r.x * (b.x - a.x)) / det;
  const dd = (p.y * (c.x - b.x) + q.y * (a.x - c.x) + r.y * (b.x - a.x)) / det;
  const ee =
    (p.x * (b.x * c.y - c.x * b.y) +
      q.x * (c.x * a.y - a.x * c.y) +
      r.x * (a.x * b.y - b.x * a.y)) /
    det;
  const ff =
    (p.y * (b.x * c.y - c.x * b.y) +
      q.y * (c.x * a.y - a.x * c.y) +
      r.y * (a.x * b.y - b.x * a.y)) /
    det;
  const sx0 = Math.max(0, Math.floor(Math.min(a.x, b.x, c.x)) - 3);
  const sy0 = Math.max(0, Math.floor(Math.min(a.y, b.y, c.y)) - 3);
  const sx1 = Math.min(texture.width, Math.ceil(Math.max(a.x, b.x, c.x)) + 3);
  const sy1 = Math.min(texture.height, Math.ceil(Math.max(a.y, b.y, c.y)) + 3);
  const sw = sx1 - sx0;
  const sh = sy1 - sy0;
  if (sw <= 0 || sh <= 0) return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  ctx.lineTo(q.x, q.y);
  ctx.lineTo(r.x, r.y);
  ctx.closePath();
  ctx.clip();
  ctx.transform(aa, bb, cc, dd, ee, ff);
  ctx.drawImage(texture, sx0, sy0, sw, sh, sx0, sy0, sw, sh);
  ctx.restore();
}

const SUBPIXEL = 256;

function isTopLeftEdge(ax: number, ay: number, bx: number, by: number): boolean {
  const dx = bx - ax;
  const dy = by - ay;
  return (dy === 0 && dx > 0) || dy < 0;
}

/**
 * Rasterises a deformed ArtMesh once into a tight bounding-box ImageData using the same
 * 1/256 px top-left fill rule and premultiplied bilinear sampling as `rig::raster`, avoiding both
 * interior-edge double compositing on semi-transparent layers and Canvas2D AA clip cracks.
 */
function drawMeshTopLeft(
  targetCtx: CanvasRenderingContext2D,
  texture: HTMLCanvasElement,
  mesh: Mesh,
  posed: Point[],
  scratch?: RenderScratch,
  texPixelsCache?: Map<HTMLCanvasElement, Uint8ClampedArray>,
): void {
  const tw = texture.width;
  const th = texture.height;
  if (tw <= 0 || th <= 0 || mesh.triangles.length === 0) return;

  let minXf = Infinity;
  let minYf = Infinity;
  let maxXf = -Infinity;
  let maxYf = -Infinity;
  for (let i = 0; i < posed.length; i++) {
    const pt = posed[i];
    if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y)) continue;
    if (pt.x < minXf) minXf = pt.x;
    if (pt.y < minYf) minYf = pt.y;
    if (pt.x > maxXf) maxXf = pt.x;
    if (pt.y > maxYf) maxYf = pt.y;
  }
  const ox = Math.max(0, Math.floor(minXf));
  const oy = Math.max(0, Math.floor(minYf));
  const ex = Math.min(tw, Math.ceil(maxXf));
  const ey = Math.min(th, Math.ceil(maxYf));
  const bw = ex - ox;
  const bh = ey - oy;
  if (bw <= 0 || bh <= 0) return;

  let src = texPixelsCache?.get(texture);
  if (!src) {
    const sCtx = texture.getContext("2d");
    if (!sCtx) return;
    src = sCtx.getImageData(0, 0, tw, th).data;
    texPixelsCache?.set(texture, src);
  }

  let offscreen = scratch?.meshCanvas;
  if (!offscreen) {
    offscreen = canvas(bw, bh);
    if (scratch) scratch.meshCanvas = offscreen;
  } else if (offscreen.width !== bw || offscreen.height !== bh) {
    offscreen.width = bw;
    offscreen.height = bh;
  }
  const oCtx = offscreen.getContext("2d");
  if (!oCtx) return;
  const imgData = oCtx.createImageData(bw, bh);
  const dst = imgData.data;

  const verts = mesh.vertices;
  const tris = mesh.triangles;
  const halfSub = SUBPIXEL / 2;

  for (let t = 0; t < tris.length; t += 3) {
    const i0 = tris[t];
    const i1 = tris[t + 1];
    const i2 = tris[t + 2];
    const p0 = posed[i0];
    const p1 = posed[i1];
    const p2 = posed[i2];
    if (
      !Number.isFinite(p0.x) ||
      !Number.isFinite(p0.y) ||
      !Number.isFinite(p1.x) ||
      !Number.isFinite(p1.y) ||
      !Number.isFinite(p2.x) ||
      !Number.isFinite(p2.y)
    ) {
      continue;
    }

    let ax = Math.round(p0.x * SUBPIXEL);
    let ay = Math.round(p0.y * SUBPIXEL);
    let bx = Math.round(p1.x * SUBPIXEL);
    let by = Math.round(p1.y * SUBPIXEL);
    let cx = Math.round(p2.x * SUBPIXEL);
    let cy = Math.round(p2.y * SUBPIXEL);

    let s0 = verts[i0];
    let s1 = verts[i1];
    let s2 = verts[i2];

    let area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (area === 0) continue;
    if (area < 0) {
      const tx = bx;
      const ty = by;
      bx = cx;
      by = cy;
      cx = tx;
      cy = ty;
      const ts = s1;
      s1 = s2;
      s2 = ts;
      area = -area;
    }

    const triMinX = Math.min(ax, bx, cx);
    const triMaxX = Math.max(ax, bx, cx);
    const triMinY = Math.min(ay, by, cy);
    const triMaxY = Math.max(ay, by, cy);

    const x0 = Math.max(ox, Math.floor((triMinX - halfSub) / SUBPIXEL));
    const x1 = Math.min(ex - 1, Math.floor((triMaxX - halfSub) / SUBPIXEL) + 1);
    const y0 = Math.max(oy, Math.floor((triMinY - halfSub) / SUBPIXEL));
    const y1 = Math.min(ey - 1, Math.floor((triMaxY - halfSub) / SUBPIXEL) + 1);
    if (x0 > x1 || y0 > y1) continue;

    const bias0 = isTopLeftEdge(bx, by, cx, cy) ? 0 : -1;
    const bias1 = isTopLeftEdge(cx, cy, ax, ay) ? 0 : -1;
    const bias2 = isTopLeftEdge(ax, ay, bx, by) ? 0 : -1;
    const invArea = 1 / area;

    for (let y = y0; y <= y1; y++) {
      const py = y * SUBPIXEL + halfSub;
      const rowOff = (y - oy) * bw;
      for (let x = x0; x <= x1; x++) {
        const px = x * SUBPIXEL + halfSub;
        const e0 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
        if (e0 + bias0 < 0) continue;
        const e1 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
        if (e1 + bias1 < 0) continue;
        const e2 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        if (e2 + bias2 < 0) continue;

        const w0 = e0 * invArea;
        const w1 = e1 * invArea;
        const w2 = 1 - w0 - w1;
        const sx = w0 * s0.x + w1 * s1.x + w2 * s2.x;
        const sy = w0 * s0.y + w1 * s1.y + w2 * s2.y;

        const u = Math.min(Math.max(sx - 0.5, 0), tw - 1);
        const v = Math.min(Math.max(sy - 0.5, 0), th - 1);
        const ix0 = Math.floor(u);
        const iy0 = Math.floor(v);
        const ix1 = Math.min(ix0 + 1, tw - 1);
        const iy1 = Math.min(iy0 + 1, th - 1);
        const fx = u - ix0;
        const fy = v - iy0;

        const k00 = (iy0 * tw + ix0) * 4;
        const k10 = (iy0 * tw + ix1) * 4;
        const k01 = (iy1 * tw + ix0) * 4;
        const k11 = (iy1 * tw + ix1) * 4;

        const w00 = (1 - fx) * (1 - fy);
        const w10 = fx * (1 - fy);
        const w01 = (1 - fx) * fy;
        const w11 = fx * fy;

        const a00 = src[k00 + 3] / 255;
        const a10 = src[k10 + 3] / 255;
        const a01 = src[k01 + 3] / 255;
        const a11 = src[k11 + 3] / 255;

        const a = w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11;
        const a8 = Math.round(a * 255);
        if (a8 <= 0) continue;

        const pr =
          w00 * src[k00] * a00 +
          w10 * src[k10] * a10 +
          w01 * src[k01] * a01 +
          w11 * src[k11] * a11;
        const pg =
          w00 * src[k00 + 1] * a00 +
          w10 * src[k10 + 1] * a10 +
          w01 * src[k01 + 1] * a01 +
          w11 * src[k11 + 1] * a11;
        const pb =
          w00 * src[k00 + 2] * a00 +
          w10 * src[k10 + 2] * a10 +
          w01 * src[k01 + 2] * a01 +
          w11 * src[k11 + 2] * a11;

        const di = (rowOff + (x - ox)) * 4;
        dst[di] = Math.min(255, Math.max(0, Math.round(pr / a)));
        dst[di + 1] = Math.min(255, Math.max(0, Math.round(pg / a)));
        dst[di + 2] = Math.min(255, Math.max(0, Math.round(pb / a)));
        dst[di + 3] = a8;
      }
    }
  }

  oCtx.putImageData(imgData, 0, 0);
  targetCtx.drawImage(offscreen, ox, oy);
}

export function blendModeToCanvasOp(
  blendMode?: BlendMode,
): GlobalCompositeOperation {
  if (blendMode === "multiply") return "multiply";
  if (blendMode === "additive") return "lighter";
  return "source-over";
}

function computePosedPoints(
  mesh: Mesh | undefined,
  part: Pick<Part, "id" | "role">,
  project: Project,
  params: Parameters,
  shapeDeform: ShapeDeformInput,
  moving: boolean,
): Point[] | null {
  if (!moving || !mesh || mesh.triangles.length === 0) return null;
  // T10: eye layers deform with the aperture lines of the project's visible eye masks.
  const lm = rigLandmarks(project);
  let maxDelta = 0;
  const posed = new Array<Point>(mesh.vertices.length);
  for (let i = 0; i < mesh.vertices.length; i++) {
    const sv = mesh.vertices[i];
    const dv = deform(sv, part, lm, params, shapeDeform);
    const d = Math.abs(dv.x - sv.x) + Math.abs(dv.y - sv.y);
    if (d > maxDelta) maxDelta = d;
    posed[i] = dv;
  }
  return maxDelta > 1e-3 ? posed : null;
}

function drawPartMeshOrImage(
  targetCtx: CanvasRenderingContext2D,
  texture: HTMLCanvasElement,
  mesh: Mesh | undefined,
  part: Pick<Part, "id" | "role">,
  project: Project,
  params: Parameters,
  shapeDeform: ShapeDeformInput,
  moving: boolean,
  precomputedPosed?: Point[] | null,
  scratch?: RenderScratch,
  texPixelsCache?: Map<HTMLCanvasElement, Uint8ClampedArray>,
): Point[] | null {
  const posed =
    precomputedPosed !== undefined
      ? precomputedPosed
      : computePosedPoints(mesh, part, project, params, shapeDeform, moving);
  if (posed && mesh && mesh.triangles.length > 0) {
    drawMeshTopLeft(targetCtx, texture, mesh, posed, scratch, texPixelsCache);
    return posed;
  }
  targetCtx.drawImage(texture, 0, 0);
  return null;
}

export function drawWireOverlay(
  ctx: CanvasRenderingContext2D,
  mesh: Mesh,
  part: Pick<Part, "id" | "role">,
  project: Project,
  pose: Parameters,
  shapeDeform: ShapeDeformInput,
  deformVertices: boolean,
  isSelected: boolean,
  scale: number,
  precomputedPosed?: Point[] | null,
) {
  if (!mesh.triangles.length) return;
  const posed =
    precomputedPosed ??
    (deformVertices
      ? mesh.vertices.map((pt) =>
          deform(pt, part, rigLandmarks(project), pose, shapeDeform),
        )
      : mesh.vertices);
  const edgeCounts = new Map<string, [number, number, number]>();
  const addEdge = (u: number, v: number) => {
    const lo = Math.min(u, v);
    const hi = Math.max(u, v);
    const key = `${lo}:${hi}`;
    const prev = edgeCounts.get(key);
    if (prev) prev[2]++;
    else edgeCounts.set(key, [lo, hi, 1]);
  };
  for (let t = 0; t < mesh.triangles.length; t += 3) {
    const a = mesh.triangles[t];
    const b = mesh.triangles[t + 1];
    const c = mesh.triangles[t + 2];
    addEdge(a, b);
    addEdge(b, c);
    addEdge(c, a);
  }
  ctx.save();
  ctx.beginPath();
  ctx.strokeStyle = isSelected ? "#49efcc" : "#9faecb";
  ctx.globalAlpha = isSelected ? 0.42 : 0.28;
  ctx.lineWidth = Math.max(0.5, 0.8 / scale);
  for (const [u, v, cnt] of edgeCounts.values()) {
    if (cnt <= 1) continue;
    ctx.moveTo(posed[u].x, posed[u].y);
    ctx.lineTo(posed[v].x, posed[v].y);
  }
  ctx.stroke();

  ctx.beginPath();
  ctx.strokeStyle = isSelected ? "#49efcc" : "#8fb0e0";
  ctx.globalAlpha = isSelected ? 0.95 : 0.72;
  ctx.lineWidth = Math.max(0.85, 1.35 / scale);
  for (const [u, v, cnt] of edgeCounts.values()) {
    if (cnt !== 1) continue;
    ctx.moveTo(posed[u].x, posed[u].y);
    ctx.lineTo(posed[v].x, posed[v].y);
  }
  ctx.stroke();
  ctx.restore();
}

export interface SubstituteLayerTexture {
  key: string;
  partId: string;
  blendMode?: BlendMode;
  canvas: HTMLCanvasElement;
  mesh: Mesh;
}

export function buildSubstituteLayerTexture(
  layer: ShapeSubstituteLayer,
  assetId: string,
  project: Project,
  decodedImage?: HTMLImageElement | HTMLCanvasElement,
): SubstituteLayerTexture {
  const w = project.width;
  const h = project.height;
  const c = canvas(w, h);
  const g = c.getContext("2d")!;
  const bw = layer.bounds.width;
  const bh = layer.bounds.height;

  if (decodedImage) {
    g.drawImage(
      decodedImage,
      0,
      0,
      bw,
      bh,
      layer.bounds.left,
      layer.bounds.top,
      bw,
      bh,
    );
    if (layer.mask) {
      const imgData = g.getImageData(layer.bounds.left, layer.bounds.top, bw, bh);
      for (let i = 0; i < bw * bh; i++) {
        imgData.data[i * 4 + 3] = Math.round(
          (imgData.data[i * 4 + 3] * layer.mask[i]) / 255,
        );
      }
      g.putImageData(imgData, layer.bounds.left, layer.bounds.top);
    }
  } else if (layer.mask) {
    const fullMask = new Uint8Array(w * h);
    for (let y = 0; y < bh; y++) {
      for (let x = 0; x < bw; x++) {
        fullMask[(layer.bounds.top + y) * w + (layer.bounds.left + x)] =
          layer.mask[y * bw + x];
      }
    }
    const imgData = g.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const a = fullMask[i];
      if (!a) continue;
      imgData.data[i * 4] = 44;
      imgData.data[i * 4 + 1] = 36;
      imgData.data[i * 4 + 2] = 48;
      imgData.data[i * 4 + 3] = a;
    }
    g.putImageData(imgData, 0, 0);
  }

  const fullAlphaMask = new Uint8Array(w * h);
  const fullData = g.getImageData(0, 0, w, h).data;
  for (let i = 0; i < w * h; i++) fullAlphaMask[i] = fullData[i * 4 + 3];
  const mesh = meshForPart(
    fullAlphaMask,
    w,
    h,
    14,
    project.landmarks,
    layer.partId,
  );

  return {
    key: `${assetId}/${layer.partId}`,
    partId: layer.partId,
    blendMode: layer.blendMode,
    canvas: c,
    mesh,
  };
}

/**
 * Decodes all `project.shapeAssets` PNG sprites into canvas textures and part-aware meshes.
 * Works identically in browser DOM and Node `@napi-rs/canvas` headless tests via data URLs.
 */
export async function buildProjectSubstituteTextures(
  project: Project,
): Promise<Map<string, SubstituteLayerTexture>> {
  const out = new Map<string, SubstituteLayerTexture>();
  if (!project.shapeAssets) return out;
  const jobs: Promise<readonly [string, SubstituteLayerTexture]>[] = [];
  for (const [assetId, asset] of Object.entries(project.shapeAssets)) {
    for (const layer of asset.layers) {
      jobs.push(
        (async () => {
          const decoded = layer.pngBytes
            ? await loadImage(`data:image/png;base64,${base64(layer.pngBytes)}`)
            : undefined;
          const tex = buildSubstituteLayerTexture(
            layer,
            assetId,
            project,
            decoded,
          );
          return [tex.key, tex] as const;
        })(),
      );
    }
  }
  const entries = await Promise.all(jobs);
  for (const [key, tex] of entries) {
    out.set(key, tex);
  }
  return out;
}

export interface RenderFrameOptions {
  ctx: CanvasRenderingContext2D;
  project: Project;
  evaluated: EvaluatedRigFrame;
  partTextures: Map<string, HTMLCanvasElement>;
  meshes: Map<string, Mesh>;
  substituteTextures?: Map<string, SubstituteLayerTexture>;
  sourceImage?: HTMLImageElement | HTMLCanvasElement;
  showOriginal?: boolean;
  soloPartId?: string;
  previewDeform?: boolean;
  wireframe?: boolean;
  selectedPartId?: string;
  scale?: number;
  scratch?: RenderScratch;
  enableGpu?: boolean;
}

interface ResolvedRenderLayer {
  spec: GlLayerDrawSpec;
  partObj: Pick<Part, "id" | "role" | "blendMode">;
  isBasePart: boolean;
}

const hasHiddenArea = (part: Pick<Part, "fillMask">) =>
  Boolean(part.fillMask?.some((v) => v > 0));

/**
 * T10 clipping rule (mirrors Rust `rig::exposure::default_clip_targets`): an iris layer
 * (`eye{L,R}_iris`, numbered extras by suffix) clips to its side's `eye{L,R}_white` when that white
 * has a hidden area (`fillMask`) and is drawn below it — otherwise clipping would erase visible iris
 * pixels; `face_detail` and `mouth_*` clip to `face` when `face` has a hidden area. Entries of the
 * rig profile's `clippingTargets` are honoured under the same condition (target below the child,
 * target with a hidden area). The pose never changes the result, so still previews, thumbnails,
 * snapshots and playback clip identically.
 */
export function resolveClippingTargets(
  orderedParts: ReadonlyArray<Pick<Part, "id" | "fillMask">>,
  profileTargets?: Record<string, string>,
): Record<string, string> {
  const index = new Map(orderedParts.map((part, i) => [part.id, i] as const));
  const usable = (childId: string, targetId: string) => {
    const c = index.get(childId);
    const t = index.get(targetId);
    return c !== undefined && t !== undefined && t < c && hasHiddenArea(orderedParts[t]);
  };
  const out: Record<string, string> = {};
  for (const part of orderedParts) {
    const side = eyeLayerSide(part.id);
    let target: string | undefined;
    if (side && isEyeIrisLayer(part.id)) target = `eye${side}_white`;
    else if (part.id === "face_detail" || part.id.startsWith("mouth")) target = "face";
    if (target && usable(part.id, target)) out[part.id] = target;
  }
  for (const [childId, targetId] of Object.entries(profileTargets ?? {})) {
    if (!(childId in out) && usable(childId, targetId)) out[childId] = targetId;
  }
  return out;
}

export function renderCharacterFrame(opts: RenderFrameOptions): RenderScratch {
  const {
    ctx,
    project: p,
    evaluated,
    partTextures,
    meshes,
    substituteTextures,
    sourceImage,
    showOriginal = false,
    soloPartId,
    previewDeform = true,
    wireframe = false,
    selectedPartId,
    scale = 1,
    enableGpu = false,
  } = opts;
  const scratch = ensureRenderScratch(opts.scratch, p.width, p.height, enableGpu);
  const { parameters: pose, shapeDeform, isRest } = evaluated;
  const moving = previewDeform && !isRest;

  ctx.save();
  ctx.clearRect(0, 0, p.width, p.height);

  if (showOriginal || !p.parts.length) {
    if (sourceImage) ctx.drawImage(sourceImage, 0, 0, p.width, p.height);
    ctx.restore();
    return scratch;
  }

  const orderedParts = [...p.parts]
    .map((part, i) => ({ part, i }))
    .sort(
      (a, b) => (a.part.compositeOrder ?? a.i) - (b.part.compositeOrder ?? b.i),
    );

  // Resolve substitute assets to an anchor slot among visible parts so
  // [skin, (inner), lid/lip] always composite in order under brows, glasses, and bangs
  const visibleOrdered = orderedParts.filter(
    ({ part }) => part.visible && part.status !== "missing" && (!soloPartId || part.id === soloPartId),
  );
  // T10: one pose-independent clipping rule for every render path (playback, still expression /
  // gaze previews, thumbnails, PNG/export snapshots), shared with Rust `exposure::default_clip_targets`.
  const clippingTargets = resolveClippingTargets(
    visibleOrdered.map(({ part }) => part),
    p.rigProfile.clippingTargets,
  );
  // Mouth substitute layers are drawn above `face` (not parts): same `mouth_*` → `face` rule.
  if (visibleOrdered.some(({ part }) => part.id === "face" && hasHiddenArea(part))) {
    for (const id of ["mouth_sub_skin", "mouth_sub_inner"]) clippingTargets[id] ??= "face";
  }
  const subsAtSlot = new Map<string, ShapeSubstituteAsset[]>();
  const subsAfterFace: ShapeSubstituteAsset[] = [];
  const subsUnanchored: ShapeSubstituteAsset[] = [];
  if (p.shapeAssets && previewDeform && !soloPartId) {
    const hasFace = visibleOrdered.some(({ part }) => part.id === "face");
    for (const region of ["eye_L", "eye_R", "mouth"] as const) {
      const aid = evaluated.activeShapes[region].assetId;
      const asset = aid ? p.shapeAssets[aid] : undefined;
      if (!asset) continue;
      const anchor = visibleOrdered.find(({ part }) =>
        asset.hidesPartIds.includes(part.id),
      );
      if (anchor) {
        const list = subsAtSlot.get(anchor.part.id) ?? [];
        list.push(asset);
        subsAtSlot.set(anchor.part.id, list);
      } else if (hasFace) {
        subsAfterFace.push(asset);
      } else {
        subsUnanchored.push(asset);
      }
    }
  }

  const drawnAssets = new Set<string>();
  const layerEntries: ResolvedRenderLayer[] = [];

  const pushLayer = (
    cacheKey: string,
    partObj: Pick<Part, "id" | "role" | "blendMode">,
    texCanvas: HTMLCanvasElement,
    mesh: Mesh | undefined,
    alpha: number,
    isBasePart: boolean,
    maskRef?: Uint8Array,
    fillRef?: Uint8Array,
  ) => {
    const posedPoints = computePosedPoints(
      mesh,
      partObj,
      p,
      pose,
      shapeDeform,
      moving,
    );
    layerEntries.push({
      spec: {
        cacheKey,
        id: partObj.id,
        blendMode: partObj.blendMode,
        texCanvas,
        maskRef,
        fillRef,
        mesh,
        posedPoints,
        alpha,
      },
      partObj,
      isBasePart,
    });
  };

  const collectSubstituteAsset = (asset: ShapeSubstituteAsset) => {
    if (drawnAssets.has(asset.assetId)) return;
    drawnAssets.add(asset.assetId);
    for (const layer of asset.layers) {
      const key = `${asset.assetId}/${layer.partId}`;
      const subTex = substituteTextures?.get(key);
      if (!subTex) continue;
      const layerAlpha =
        layer.partId === "mouth_sub_inner" ? evaluated.mouthInteriorAlpha : 1;
      pushLayer(
        `sub:${key}`,
        { id: subTex.partId, blendMode: subTex.blendMode },
        subTex.canvas,
        subTex.mesh,
        layerAlpha,
        false,
      );
    }
  };

  for (const { part } of visibleOrdered) {
    const slotSubs = subsAtSlot.get(part.id);
    if (slotSubs) {
      for (const asset of slotSubs) collectSubstituteAsset(asset);
    }

    const isHiddenBySub =
      previewDeform && !soloPartId && evaluated.hiddenBasePartIds.has(part.id);
    if (isHiddenBySub) {
      if (part.id === "face" && subsAfterFace.length > 0) {
        for (const asset of subsAfterFace) collectSubstituteAsset(asset);
      }
      continue;
    }

    const tex = partTextures.get(part.id);
    if (tex) {
      let partAlpha = 1;
      if (previewDeform && !soloPartId) {
        if (part.id === "eyeL_iris") partAlpha = evaluated.eyeIrisAlpha.L;
        else if (part.id === "eyeR_iris") partAlpha = evaluated.eyeIrisAlpha.R;
        else if (part.id === "eyeL_white") partAlpha = evaluated.eyeWhiteAlpha.L;
        else if (part.id === "eyeR_white") partAlpha = evaluated.eyeWhiteAlpha.R;
        else if (part.id === "eyeL_lower") partAlpha = evaluated.eyeLowerAlpha.L;
        else if (part.id === "eyeR_lower") partAlpha = evaluated.eyeLowerAlpha.R;
        else if (part.id === "mouth_inner") partAlpha = evaluated.mouthInteriorAlpha;
      }
      pushLayer(
        `part:${part.id}`,
        part,
        tex,
        meshes.get(part.id),
        partAlpha,
        true,
        part.mask,
        part.fillMask,
      );
    }

    if (part.id === "face" && subsAfterFace.length > 0) {
      for (const asset of subsAfterFace) collectSubstituteAsset(asset);
    }
  }

  for (const asset of subsUnanchored) {
    collectSubstituteAsset(asset);
  }

  const glRendered =
    enableGpu &&
    !!scratch.gl &&
    scratch.gl.render(
      layerEntries.map((entry) => entry.spec),
      p.width,
      p.height,
      clippingTargets,
      !soloPartId && Object.keys(clippingTargets).length > 0,
    );

  if (glRendered && scratch.gl) {
    ctx.drawImage(scratch.gl.canvas, 0, 0);
    if (wireframe) {
      for (const entry of layerEntries) {
        if (!entry.isBasePart || !entry.spec.mesh) continue;
        drawWireOverlay(
          ctx,
          entry.spec.mesh,
          entry.partObj,
          p,
          pose,
          shapeDeform,
          previewDeform,
          entry.partObj.id === selectedPartId,
          scale,
          entry.spec.posedPoints,
        );
      }
    }
    ctx.restore();
    return scratch;
  }

  const renderedSclera = new Map<string, HTMLCanvasElement>();
  const scleraSet = new Set(Object.values(clippingTargets));
  const texPixelsCache = new Map<HTMLCanvasElement, Uint8ClampedArray>();

  for (const entry of layerEntries) {
    const { spec, partObj, isBasePart } = entry;
    const extraAlpha = spec.alpha;
    if (extraAlpha > 0.01) {
      ctx.save();
      ctx.globalCompositeOperation = blendModeToCanvasOp(partObj.blendMode);
      ctx.globalAlpha = extraAlpha;
      if (
        partObj.id === "eyeR_white" ||
        partObj.id === "eyeL_white" ||
        scleraSet.has(partObj.id)
      ) {
        drawPartMeshOrImage(
          ctx,
          spec.texCanvas,
          spec.mesh,
          partObj,
          p,
          pose,
          shapeDeform,
          moving,
          spec.posedPoints,
          scratch,
          texPixelsCache,
        );
        if (!soloPartId) {
          let targetCanvas = scratch.scleraTargets.get(partObj.id);
          if (
            !targetCanvas ||
            targetCanvas.width !== p.width ||
            targetCanvas.height !== p.height
          ) {
            targetCanvas = canvas(p.width, p.height);
            scratch.scleraTargets.set(partObj.id, targetCanvas);
          }
          const wCtx = targetCanvas.getContext("2d")!;
          wCtx.clearRect(0, 0, p.width, p.height);
          wCtx.globalCompositeOperation = "source-over";
          drawPartMeshOrImage(
            wCtx,
            spec.texCanvas,
            spec.mesh,
            partObj,
            p,
            pose,
            shapeDeform,
            moving,
            spec.posedPoints,
            scratch,
            texPixelsCache,
          );
          renderedSclera.set(partObj.id, targetCanvas);
        }
      } else if (clippingTargets[partObj.id] && !soloPartId) {
        const sclera = renderedSclera.get(clippingTargets[partObj.id]);
        if (sclera) {
          const iCtx = scratch.iris.getContext("2d")!;
          iCtx.clearRect(0, 0, p.width, p.height);
          iCtx.globalCompositeOperation = "source-over";
          drawPartMeshOrImage(
            iCtx,
            spec.texCanvas,
            spec.mesh,
            partObj,
            p,
            pose,
            shapeDeform,
            moving,
            spec.posedPoints,
            scratch,
            texPixelsCache,
          );
          iCtx.globalCompositeOperation = "destination-in";
          iCtx.drawImage(sclera, 0, 0);
          iCtx.globalCompositeOperation = "source-over";
          ctx.drawImage(scratch.iris, 0, 0);
        } else {
          drawPartMeshOrImage(
            ctx,
            spec.texCanvas,
            spec.mesh,
            partObj,
            p,
            pose,
            shapeDeform,
            moving,
            spec.posedPoints,
            scratch,
            texPixelsCache,
          );
        }
      } else {
        drawPartMeshOrImage(
          ctx,
          spec.texCanvas,
          spec.mesh,
          partObj,
          p,
          pose,
          shapeDeform,
          moving,
          spec.posedPoints,
          scratch,
          texPixelsCache,
        );
      }
      ctx.restore();
    }

    if (wireframe && isBasePart && spec.mesh) {
      drawWireOverlay(
        ctx,
        spec.mesh,
        partObj,
        p,
        pose,
        shapeDeform,
        previewDeform,
        partObj.id === selectedPartId,
        scale,
        spec.posedPoints,
      );
    }
  }

  ctx.restore();
  return scratch;
}

export function buildAllPartTextures(
  project: Project,
  sourcePixels: Uint8ClampedArray,
  extraTextures?: Map<string, Uint8ClampedArray>,
): Map<string, HTMLCanvasElement> {
  const out = new Map<string, HTMLCanvasElement>();
  for (const part of project.parts) {
    const texPx = extraTextures?.get(part.id);
    out.set(
      part.id,
      pixelsCanvas(
        compositePartTexture(part, sourcePixels, texPx),
        project.width,
        project.height,
      ),
    );
  }
  return out;
}

export function buildAllPartMeshes(
  project: Project,
  existing?: Map<string, Mesh>,
): Map<string, Mesh> {
  const out = new Map<string, Mesh>();
  for (const part of project.parts) {
    out.set(
      part.id,
      existing?.get(part.id) ??
        meshForPart(
          effectiveMask(part),
          project.width,
          project.height,
          14,
          project.landmarks,
          part,
        ),
    );
  }
  return out;
}

let sharedSnapshotScratch: RenderScratch | undefined;

export function renderDeterministicSnapshot(
  input: EvaluateRigInput,
  partTextures: Map<string, HTMLCanvasElement>,
  meshes: Map<string, Mesh>,
  substituteTextures?: Map<string, SubstituteLayerTexture>,
  targetSize?: { width: number; height: number },
): { canvas: HTMLCanvasElement; evaluated: EvaluatedRigFrame } {
  const evaluated = evaluateRigFrame({
    ...input,
    deterministic: true,
    playing: false,
  });
  const full = canvas(input.project.width, input.project.height);
  const fullCtx = full.getContext("2d")!;
  sharedSnapshotScratch = renderCharacterFrame({
    ctx: fullCtx,
    project: input.project,
    evaluated,
    partTextures,
    meshes,
    substituteTextures,
    previewDeform: true,
    scratch: sharedSnapshotScratch,
    enableGpu: true,
  });
  if (
    targetSize &&
    (targetSize.width !== input.project.width ||
      targetSize.height !== input.project.height)
  ) {
    const scaled = canvas(targetSize.width, targetSize.height);
    const sg = scaled.getContext("2d")!;
    sg.drawImage(full, 0, 0, targetSize.width, targetSize.height);
    return { canvas: scaled, evaluated };
  }
  return { canvas: full, evaluated };
}
