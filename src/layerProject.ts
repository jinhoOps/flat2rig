/**
 * B1: layered PSD → flat2rig project.
 *
 * `openPsdSession` parses a PSD and proposes a mapping (`src/psdLayers.ts`); the mapping screen edits
 * `session.mappings`; `buildPsdProject` builds the parts (`src/layerParts.ts`) and wraps them in a
 * schema v3 project whose parts carry `provenance: "layer"` and whose `layerSource` records the PSD,
 * the mapping and the original layer paths. Missing default parts are listed, never synthesized.
 *
 * Textures are encoded here as straight-alpha PNG (fflate, no canvas) so the stored texture is the
 * built part bit for bit — a canvas round trip would premultiply and lose low-alpha colour.
 */
import { zlibSync } from "fflate";
import {
  createDefaultProjectRigState,
  describeDynamicPart,
  inferPartRole,
  PARENT_PART_DEFS,
  PART_DEFS,
  type Landmarks,
  type LayerSourceMapping,
  type LayerSourceRecord,
  type Part,
  type Project,
} from "./types";
import { defaultLandmarks } from "./engine";
import {
  effectiveBlend,
  missingDefaultParts,
  parsePsd,
  suggestMapping,
  PSD_LAYER_INPUT_VERSION,
  type LayerMapping,
  type MappingSuggestion,
  type ParsedPsd,
} from "./psdLayers";
import {
  buildLayerParts,
  partCanvasRgba,
  planParts,
  splitComponentsLR,
  workingSize,
  type LayerPartsBuild,
} from "./layerParts";
import { correctSeeThroughDrawOrder } from "./seethrough";

export interface PsdSession {
  fileName: string;
  sha256?: string;
  psd: ParsedPsd;
  suggestion: MappingSuggestion;
  /** Current (user-edited) mapping, one entry per pixel layer in PSD order. */
  mappings: LayerMapping[];
  /** Midline (PSD px) used for `{S}` splits; replaced by the landmark midline after analysis. */
  midlineX: number;
  /** Whether See-through depth-order correction is enabled (default true for See-through PSDs). */
  applySeeThroughOrderFix: boolean;
}

async function sha256Hex(bytes: Uint8Array): Promise<string | undefined> {
  try {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes as BufferSource);
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  } catch {
    return undefined;
  }
}

export async function openPsdSession(bytes: Uint8Array, fileName: string): Promise<PsdSession> {
  const psd = parsePsd(bytes, fileName);
  const suggestion = suggestMapping(psd);
  return {
    fileName,
    sha256: await sha256Hex(bytes),
    psd,
    suggestion,
    mappings: suggestion.mappings.map((m) => ({ ...m })),
    midlineX: suggestion.midlineX,
    applySeeThroughOrderFix: psd.seeThrough,
  };
}

// ---------------------------------------------------------------------------------------------
// PNG (straight RGBA, filter 0) — exact bytes, no canvas
// ---------------------------------------------------------------------------------------------

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(parts: Uint8Array[]) {
  let c = 0xffffffff;
  for (const p of parts) for (let i = 0; i < p.length; i++) c = CRC_TABLE[(c ^ p[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array) {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  const t = new Uint8Array([...type].map((ch) => ch.charCodeAt(0)));
  out.set(t, 4);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32([t, data]));
  return out;
}

export function encodePngRgba(rgba: Uint8Array | Uint8ClampedArray, w: number, h: number): Uint8Array {
  const stride = w * 4;
  const raw = new Uint8Array((stride + 1) * h);
  for (let y = 0; y < h; y++) raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w);
  dv.setUint32(4, h);
  ihdr.set([8, 6, 0, 0, 0], 8);
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlibSync(raw, { level: 4 })),
    chunk("IEND", new Uint8Array(0)),
  ];
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

function bytesToBase64(bytes: Uint8Array) {
  let text = "";
  for (let i = 0; i < bytes.length; i += 32768)
    text += String.fromCharCode(...bytes.subarray(i, i + 32768));
  return btoa(text);
}

export function pngDataUrl(rgba: Uint8Array | Uint8ClampedArray, w: number, h: number) {
  return `data:image/png;base64,${bytesToBase64(encodePngRgba(rgba, w, h))}`;
}

// ---------------------------------------------------------------------------------------------
// Parts and project
// ---------------------------------------------------------------------------------------------

const PART_META = new Map<string, readonly [string, string, string, string]>(
  [...PART_DEFS, ...PARENT_PART_DEFS].map((d) => [d[0], d]),
);

function partMeta(id: string) {
  const known = PART_META.get(id);
  if (known) return { name: known[1], group: known[2], color: known[3] };
  const d = describeDynamicPart(id);
  return { name: d.name, group: d.group, color: d.color };
}

/** Parts in stack order (bottom → top) with `provenance: "layer"`. */
export function partsFromLayerBuild(build: LayerPartsBuild): Part[] {
  return build.parts.map((bp, i) => {
    const meta = partMeta(bp.id);
    const fill = build.fills[i];
    let hasFill = false,
      visible = 0;
    for (let k = 0; k < fill.length; k++) {
      if (fill[k]) hasFill = true;
      if (build.masks[i][k]) visible++;
    }
    return {
      id: bp.id,
      name: meta.name,
      group: meta.group,
      color: meta.color,
      role: inferPartRole(bp.id),
      visible: true,
      status: "draft",
      mask: build.masks[i],
      ...(hasFill ? { fillMask: fill } : {}),
      texture: pngDataUrl(partCanvasRgba(bp, build.width, build.height), build.width, build.height),
      provenance: "layer",
      fillKind: hasFill ? "external" : "none",
      confidence: 1,
      diagnostics: visible ? [] : ["fully_covered"],
      compositeOrder: i,
    } satisfies Part;
  });
}

export function layerSourceRecord(session: PsdSession, build: LayerPartsBuild): LayerSourceRecord {
  const byPath = new Map(session.psd.layers.map((n) => [n.path, n]));
  const mappings: LayerSourceMapping[] = session.mappings.map((m) => {
    const node = byPath.get(m.path);
    const blendMode = node ? effectiveBlend(node) : "normal";
    return {
      path: m.path,
      target: m.target,
      include: m.include,
      ...(m.blend ? { blend: m.blend } : {}),
      source: m.source,
      ...(blendMode !== "normal" ? { blendMode } : {}),
      ...(m.note ? { note: m.note } : {}),
    };
  });
  const warnings = session.suggestion.warnings.map((w) => w.message);
  if (session.psd.seeThrough) {
    const rawOrder = planParts(session.psd, session.mappings).map((p) => p.id);
    const plan = correctSeeThroughDrawOrder(rawOrder);
    if (plan.changed) {
      warnings.push(
        session.applySeeThroughOrderFix
          ? `See-through 깊이 순서 보정 적용됨: ${plan.fixes.map((f) => f.partId).join(", ")}`
          : `See-through 원본 PSD 순서 유지됨(보정 미적용): ${plan.fixes.map((f) => f.partId).join(", ")}`,
      );
    }
  }
  return {
    version: PSD_LAYER_INPUT_VERSION,
    file: session.fileName,
    ...(session.sha256 ? { sha256: session.sha256 } : {}),
    width: session.psd.width,
    height: session.psd.height,
    seeThrough: session.psd.seeThrough,
    midlineX: build.midlineX,
    midlineSource: build.midlineX === session.suggestion.midlineX ? session.suggestion.midlineSource : "landmarks",
    mappings,
    parts: build.parts.map((p) => ({ id: p.id, layers: p.layers })),
    missingDefaultParts: missingDefaultParts(session.mappings),
    interleaved: build.interleaved,
    warnings,
  };
}

export function buildPsdParts(
  session: PsdSession,
  onProgress?: (done: number, total: number, step: string) => void,
) {
  const size = workingSize(session.psd.width, session.psd.height);
  let partOrder: readonly string[] | undefined;
  if (session.psd.seeThrough && session.applySeeThroughOrderFix) {
    const rawOrder = planParts(session.psd, session.mappings).map((p) => p.id);
    const plan = correctSeeThroughDrawOrder(rawOrder);
    if (plan.changed) partOrder = plan.correctedOrder;
  }
  return buildLayerParts(
    session.psd,
    session.mappings,
    { ...size, midlineX: session.midlineX, ...(partOrder ? { partOrder } : {}) },
    onProgress,
  );
}

/** A fresh project from built layer parts (landmarks are placeholders until face analysis runs). */
export function projectFromLayerBuild(session: PsdSession, build: LayerPartsBuild): Project {
  const { width: w, height: h } = build;
  const source = pngDataUrl(build.composite, w, h);
  return {
    schemaVersion: 3,
    name: session.psd.name,
    width: w,
    height: h,
    originalWidth: w,
    originalHeight: h,
    source,
    original: source,
    workToOriginalTransform: { matrix2: [1, 0, 0, 1], translation: [0, 0] },
    landmarks: defaultLandmarks(w, h),
    landmarkProvenance: {
      eyeL: "estimated",
      eyeR: "estimated",
      mouth: "estimated",
      chin: "estimated",
      crown: "estimated",
    },
    parts: partsFromLayerBuild(build),
    revision: 0,
    provider: "layers",
    postprocessVersions: { parts: PSD_LAYER_INPUT_VERSION },
    analysisStatus: "idle",
    detailMode: "detail",
    ...createDefaultProjectRigState("ld"),
    createdAt: new Date().toISOString(),
    layerSource: layerSourceRecord(session, build),
  };
}

/** Landmark midline in PSD pixels (between the eyes), or null when landmarks are placeholders. */
export function landmarkMidlinePsd(project: Pick<Project, "landmarkProvenance" | "landmarks">, sx: number) {
  const prov = project.landmarkProvenance;
  if (!prov || prov.eyeL === "estimated" || prov.eyeR === "estimated") return null;
  const lm: Landmarks = project.landmarks;
  return (lm.eyeL.x + lm.eyeR.x) / 2 / sx;
}

/** Whether moving the split midline to `midlineX` changes any `{S}` layer's L/R components. */
export function splitChangesWithMidline(session: PsdSession, build: LayerPartsBuild, midlineX: number) {
  for (const s of build.splits) {
    const px = session.psd.pixels.get(s.path);
    if (!px) continue;
    const a = splitComponentsLR(px, build.midlineX).side;
    const b = splitComponentsLR(px, midlineX).side;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return true;
  }
  return false;
}

/** Replaces a layer project's parts after a re-split, keeping everything else (landmarks etc.). */
export function withRebuiltLayerParts(project: Project, session: PsdSession, build: LayerPartsBuild): Project {
  return {
    ...project,
    parts: partsFromLayerBuild(build),
    layerSource: layerSourceRecord(session, build),
  };
}
