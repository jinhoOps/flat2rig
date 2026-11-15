import {
  compositePartTexture,
  deform,
  effectiveMask,
  meshForPart,
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

  // Dilate only the destination clip polygon slightly outward (~0.72px from centroid)
  // while keeping the exact affine transform so adjacent interior edges overlap cleanly without AA cracks.
  const kCx = (p.x + q.x + r.x) / 3;
  const kCy = (p.y + q.y + r.y) / 3;
  const dilateClip = (pt: Point): Point => {
    const dx = pt.x - kCx;
    const dy = pt.y - kCy;
    const len = Math.hypot(dx, dy);
    if (len < 1e-4) return pt;
    const pad = 0.72;
    return { x: pt.x + (dx / len) * pad, y: pt.y + (dy / len) * pad };
  };
  const dp = dilateClip(p);
  const dq = dilateClip(q);
  const dr = dilateClip(r);

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(dp.x, dp.y);
  ctx.lineTo(dq.x, dq.y);
  ctx.lineTo(dr.x, dr.y);
  ctx.closePath();
  ctx.clip();
  ctx.transform(aa, bb, cc, dd, ee, ff);
  ctx.drawImage(texture, sx0, sy0, sw, sh, sx0, sy0, sw, sh);
  ctx.restore();
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
  let maxDelta = 0;
  const posed = new Array<Point>(mesh.vertices.length);
  for (let i = 0; i < mesh.vertices.length; i++) {
    const sv = mesh.vertices[i];
    const dv = deform(sv, part, project.landmarks, params, shapeDeform);
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
): Point[] | null {
  const posed =
    precomputedPosed !== undefined
      ? precomputedPosed
      : computePosedPoints(mesh, part, project, params, shapeDeform, moving);
  if (posed && mesh && mesh.triangles.length > 0) {
    for (let t = 0; t < mesh.triangles.length; t += 3) {
      const i0 = mesh.triangles[t];
      const i1 = mesh.triangles[t + 1];
      const i2 = mesh.triangles[t + 2];
      drawTexturedTriangle(
        targetCtx,
        texture,
        [mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]],
        [posed[i0], posed[i1], posed[i2]],
      );
    }
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
          deform(pt, part, project.landmarks, pose, shapeDeform),
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

  const baseClippingTargets = p.rigProfile.clippingTargets ?? {
    eyeL_iris: "eyeL_white",
    eyeR_iris: "eyeR_white",
    face_detail: "face",
  };

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
  const hasFaceFill = visibleOrdered.some(
    ({ part }) => part.id === "face" && Boolean(part.fillMask),
  );
  const scleraHasFill = (targetId: string) =>
    visibleOrdered.some(
      ({ part }) => part.id === targetId && Boolean(part.fillMask?.some((v) => v > 0)),
    );
  const clippingTargets: Record<string, string> = {};
  for (const [childId, targetId] of Object.entries(baseClippingTargets)) {
    if (
      targetId === "face"
        ? hasFaceFill
        : moving || scleraHasFill(targetId)
    ) {
      clippingTargets[childId] = targetId;
    }
  }
  if (hasFaceFill) {
    Object.assign(clippingTargets, {
      face_detail: "face",
      mouth_inner: "face",
      mouth_upper: "face",
      mouth_lower: "face",
      mouth_sub_skin: "face",
      mouth_sub_inner: "face",
    });
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
