import { Channel, invoke, isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  describeDynamicPart,
  inferPartRole,
  PARENT_PART_DEFS,
  PARENT_TO_DETAIL_IDS,
  PART_DEFS,
  stageIncludesLayer,
  FILL_REGIONS,
  FILL_SYNTHS,
  type AnalysisStatus,
  type FaceCandidate,
  type FillKind,
  type FillRegion,
  type FillSynth,
  type Landmark28Point,
  type LandmarkProvenance,
  type Landmarks,
  type Mesh,
  type ModelStatusEntry,
  type Part,
  type PartProvenance,
  type PipelineStage,
  type Point,
  type PromptPoint,
  type StyleMatchReport,
  type TemplateStyleSelection,
} from "./types";
import {
  checkMeshQuality,
  deform,
  effectiveMask,
  expandFillSeparated,
  meshForPart,
  recommendedFillRadius,
  type MeshQualityReport,
} from "./engine";
import { REST, PARAM_DEFS } from "./types";
export type { MeshQualityReport } from "./engine";
export const desktop = isTauri();
/** T9: validated `{ fillRegion, fillSynth }` from a Rust `RegionReport` (unknown values dropped). */
export function parseFillRegion(
  raw: { kind?: string; synth?: string } | null | undefined,
): { fillRegion?: FillRegion; fillSynth?: FillSynth } {
  if (!raw) return {};
  const out: { fillRegion?: FillRegion; fillSynth?: FillSynth } = {};
  if (raw.kind && (FILL_REGIONS as readonly string[]).includes(raw.kind))
    out.fillRegion = raw.kind as FillRegion;
  if (raw.synth && (FILL_SYNTHS as readonly string[]).includes(raw.synth))
    out.fillSynth = raw.synth as FillSynth;
  return out;
}
export function base64(bytes: Uint8Array | Uint8ClampedArray) {
  let text = "";
  for (let i = 0; i < bytes.length; i += 32768)
    text += String.fromCharCode(...bytes.subarray(i, i + 32768));
  return btoa(text);
}
export function unbase64(value: string) {
  const text = atob(value);
  return Uint8Array.from(text, (c) => c.charCodeAt(0));
}
export async function chooseModel() {
  return open({
    title: "U2Net / U2NetP ONNX 모델 선택",
    multiple: false,
    filters: [{ name: "U2Net model", extensions: ["onnx"] }],
  });
}
export async function separateNative(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  landmarks: Landmarks,
  tolerance: number,
  modelPath: string | null,
  signal: AbortSignal,
): Promise<{ parts: Part[]; provider: string; modelHash?: string }> {
  if (desktop) {
    const result = await invoke<{
      masks: string[];
      provider: string;
      modelHash?: string;
    }>("generate_parts", {
      rgba: base64(pixels),
      width,
      height,
      landmarks,
      tolerance,
      modelPath,
    });
    signal.throwIfAborted();
    return {
      parts: PART_DEFS.map(([id, name, group, color], i) => ({
        id,
        name,
        group,
        color,
        visible: true,
        status: "draft",
        mask: unbase64(result.masks[i]),
      })),
      provider: result.provider,
      modelHash: result.modelHash,
    };
  }
  return new Promise((resolve, reject) => {
    const worker = new Worker(
      new URL("./pipeline.worker.ts", import.meta.url),
      { type: "module" },
    );
    const cleanup = () => {
      worker.terminate();
      signal.removeEventListener("abort", abort);
    };
    const abort = () => {
      cleanup();
      reject(new DOMException("Cancelled", "AbortError"));
    };
    if (signal.aborted) return abort();
    signal.addEventListener("abort", abort, { once: true });
    worker.onerror = (e) => {
      cleanup();
      reject(new Error(e.message));
    };
    worker.onmessage = (e) => {
      cleanup();
      if (e.data.error) reject(new Error(e.data.error));
      else
        resolve({
          parts: e.data.parts,
          provider: "geometry-template-v1 (browser fallback)",
        });
    };
    worker.postMessage({
      pixels: pixels.slice(),
      width,
      height,
      landmarks,
      tolerance,
      id: 1,
    });
  });
}
export interface RigPart {
  id: string;
  mesh: Mesh;
  samples: { parameter: string; value: number; vertices: Point[] }[];
  quality?: MeshQualityReport;
}
export interface PartProposalResult {
  parts: Part[];
  provider: string;
  modelHashes: Record<string, string>;
  postprocessVersion: string;
  imageHash: string;
  styleMatch?: StyleMatchReport;
}
export interface PromptCandidate {
  index: number;
  iouScore: number;
  mask: Uint8Array;
  pixelCount: number;
  diagnostics: string[];
}
export interface RefinePromptResult {
  jobId: number;
  partId: string;
  bestIndex: number;
  candidates: PromptCandidate[];
  usedFaceCrop: boolean;
  decoderSha256: string;
  encoderSha256: string;
}
export interface FillOcclusionResult {
  data: Uint8ClampedArray;
  fillMask: Uint8Array;
  combinedMask: Uint8Array;
  filledPixels: number;
  fillKind: FillKind;
  /** T9 (Rust only): region type; absent from the browser diffusion fallback. */
  fillRegion?: FillRegion;
  fillSynth?: FillSynth;
  warning?: string;
}

const PART_META_MAP = new Map<
  string,
  readonly [string, string, string, string]
>([...PART_DEFS, ...PARENT_PART_DEFS].map((entry) => [entry[0], entry]));

function collapseDetailPartsToParents(parts: Part[], n: number): Part[] {
  const byId = new Map(parts.map((p) => [p.id, p]));
  const hiddenIds = new Set(Object.values(PARENT_TO_DETAIL_IDS).flat());
  const out: Part[] = [];
  for (const [pid, name, group, color] of PARENT_PART_DEFS) {
    const merged = new Uint8Array(n);
    const members = PARENT_TO_DETAIL_IDS[pid] ?? [];
    for (const mid of members) {
      const m = byId.get(mid)?.mask;
      if (!m) continue;
      for (let i = 0; i < n; i++) merged[i] = Math.min(255, merged[i] + m[i]);
    }
    out.push({
      id: pid,
      name,
      group,
      color,
      visible: true,
      status: "draft",
      provenance: "geometry",
      compositeOrder: out.length,
      mask: merged,
    });
  }
  for (const p of parts) {
    if (!hiddenIds.has(p.id)) {
      out.push({ ...p, provenance: p.provenance ?? "geometry", compositeOrder: out.length });
    }
  }
  return out;
}

export async function proposeParts(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  landmarks: Landmarks,
  tolerance: number,
  options: {
    points28?: Landmark28Point[];
    faceBbox?: [number, number, number, number] | null;
    modelPath?: string | null;
    detailMode?: "detail" | "parent";
    hairSplit?: boolean;
    templateMode?: boolean;
    pipelineStage?: PipelineStage | "base_body";
    styleOverrides?: TemplateStyleSelection | null;
    jobId?: number | null;
  },
  signal: AbortSignal,
): Promise<PartProposalResult> {
  const detailMode = options.detailMode ?? "detail";
  const hairSplit = options.hairSplit ?? true;
  const templateMode = options.templateMode ?? detailMode === "detail";
  const pipelineStage = options.pipelineStage ?? "full";
  if (desktop) {
    const res = await invoke<{
      jobId: number;
      partIds: string[];
      masks: string[];
      fillMasks?: (string | null)[];
      fillKinds?: string[];
      fillRegions?: ({ kind: string; synth: string } | null)[];
      textures?: (string | null)[];
      metas: {
        id: string;
        provenance: PartProvenance;
        confidence: number;
        pixelCount: number;
        diagnostics: string[];
      }[];
      provider: string;
      modelHashes: Record<string, string>;
      postprocessVersion: string;
      imageHash: string;
      styleMatch?: StyleMatchReport | null;
    }>("propose_parts", {
      rgba: base64(pixels),
      width,
      height,
      landmarks,
      points28: options.points28 ?? null,
      faceBbox: options.faceBbox ?? null,
      tolerance,
      modelPath: options.modelPath ?? null,
      detailMode,
      hairSplit,
      templateMode,
      pipelineStage,
      styleOverrides: options.styleOverrides ?? null,
      jobId: options.jobId ?? null,
    });
    signal.throwIfAborted();
    const hasClothes = res.partIds.includes("clothes");
    const validFillKinds = new Set(["none", "neighbor", "external", "template"]);
    return {
      parts: res.partIds.map((id, i) => {
        const known = PART_META_MAP.get(id);
        const dynamic = known ? null : describeDynamicPart(id);
        const name =
          id === "body" && hasClothes
            ? "몸통 · 피부"
            : known
              ? known[1]
              : dynamic!.name;
        const group = known ? known[2] : dynamic!.group;
        const color = known ? known[3] : dynamic!.color;
        const meta = res.metas[i];
        const rawFill = res.fillMasks?.[i];
        const rawKind = res.fillKinds?.[i];
        const fillKind: FillKind =
          rawKind && validFillKinds.has(rawKind) ? (rawKind as FillKind) : "none";
        const region = parseFillRegion(res.fillRegions?.[i]);
        const tex = res.textures?.[i];
        return {
          id,
          name,
          group,
          color,
          role: inferPartRole(id),
          visible: true,
          status: "draft",
          mask: unbase64(res.masks[i]),
          ...(rawFill ? { fillMask: unbase64(rawFill) } : {}),
          fillKind,
          ...region,
          ...(tex ? { texture: tex } : {}),
          provenance: meta?.provenance ?? "geometry",
          confidence: meta?.confidence,
          diagnostics: meta?.diagnostics,
          compositeOrder: i,
        };
      }),
      provider: res.provider,
      modelHashes: res.modelHashes ?? {},
      postprocessVersion: res.postprocessVersion,
      imageHash: res.imageHash,
      styleMatch: res.styleMatch ?? undefined,
    };
  }
  const fallback = await separateNative(
    pixels,
    width,
    height,
    landmarks,
    tolerance,
    options.modelPath ?? null,
    signal,
  );
  const rawParts =
    detailMode === "parent"
      ? collapseDetailPartsToParents(fallback.parts, width * height)
      : fallback.parts.map((p, i) => ({
          ...p,
          provenance: "geometry" as const,
          compositeOrder: i,
        }));
  const fallbackParts = fillBrowserFallbackParts(
    rawParts,
    pixels,
    width,
    height,
    landmarks,
  );
  const parts =
    pipelineStage === "full"
      ? fallbackParts
      : fallbackParts.filter((p) => stageIncludesLayer(p.id, pipelineStage));
  return {
    parts,
    provider: fallback.provider,
    modelHashes: fallback.modelHash ? { background: fallback.modelHash } : {},
    postprocessVersion: "geometry-template-v1",
    imageHash: "",
  };
}

function encodeRgbaDataUrl(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): string | undefined {
  if (typeof document === "undefined") return undefined;
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const ctx = c.getContext("2d");
  if (!ctx) return undefined;
  ctx.putImageData(
    new ImageData(new Uint8ClampedArray(rgba), width, height),
    0,
    0,
  );
  return c.toDataURL("image/png");
}

function fillBrowserFallbackParts(
  parts: Part[],
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  landmarks: Landmarks,
): Part[] {
  if (typeof document === "undefined") return parts;
  const eyeDist = Math.max(
    12,
    Math.hypot(landmarks.eyeL.x - landmarks.eyeR.x, landmarks.eyeL.y - landmarks.eyeR.y),
  );
  const faceSupport = new Uint8Array(width * height);
  const supportIds = new Set([
    "face",
    "eyeL_white",
    "eyeL_iris",
    "eyeL_upper",
    "eyeL_lower",
    "eyeR_white",
    "eyeR_iris",
    "eyeR_upper",
    "eyeR_lower",
    "browL",
    "browR",
    "mouth_upper",
    "mouth_lower",
    "mouth_inner",
    "mouth_all",
  ]);
  for (const p of parts) {
    if (!supportIds.has(p.id)) continue;
    for (let i = 0; i < faceSupport.length; i++) {
      if (p.mask[i] > faceSupport[i]) faceSupport[i] = p.mask[i];
    }
  }
  const fillConfigs: Record<string, { radius: number; support?: Uint8Array }> = {
    face: {
      radius: recommendedFillRadius({ id: "face", role: "face" }, landmarks, width, height),
      support: faceSupport,
    },
    eyeL_white: {
      radius: recommendedFillRadius({ id: "eyeL_white", role: "eye" }, landmarks, width, height),
    },
    eyeR_white: {
      radius: recommendedFillRadius({ id: "eyeR_white", role: "eye" }, landmarks, width, height),
    },
    mouth_inner: { radius: Math.max(4, Math.round(eyeDist * 0.08)) },
  };
  return parts.map((part) => {
    const cfg = fillConfigs[part.id];
    if (!cfg) return part;
    const filled = expandFillSeparated(
      rgba,
      part.mask,
      width,
      height,
      cfg.radius,
      part.id,
    );
    if (cfg.support) {
      for (let i = 0; i < filled.fillMask.length; i++) {
        if (cfg.support[i] <= 16) {
          filled.fillMask[i] = 0;
          if (part.mask[i] === 0) filled.data[i * 4 + 3] = 0;
        }
      }
    }
    const texture = encodeRgbaDataUrl(filled.data, width, height);
    return {
      ...part,
      fillMask: filled.fillMask,
      fillKind: filled.fillKind,
      fillSynth: filled.fillSynth,
      ...(texture ? { texture } : {}),
      generated: true,
    };
  });
}

export async function refinePartPrompt(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  landmarks: Landmarks,
  partId: string,
  prompts: PromptPoint[],
  tolerance: number,
  jobId: number | null,
  signal: AbortSignal,
): Promise<RefinePromptResult> {
  if (!desktop) {
    throw new Error(
      "MobileSAM 클릭 보정은 데스크톱 앱 ONNX 런타임에서만 실행됩니다.",
    );
  }
  const res = await invoke<{
    jobId: number;
    partId: string;
    bestIndex: number;
    candidates: {
      index: number;
      iouScore: number;
      mask: string;
      pixelCount: number;
      diagnostics: string[];
    }[];
    usedFaceCrop: boolean;
    decoderSha256: string;
    encoderSha256: string;
  }>("refine_part_prompt", {
    rgba: base64(pixels),
    width,
    height,
    landmarks,
    partId,
    prompts,
    tolerance,
    jobId,
  });
  signal.throwIfAborted();
  return {
    ...res,
    candidates: res.candidates.map((c) => ({
      ...c,
      mask: unbase64(c.mask),
    })),
  };
}

export async function buildRig(
  parts: Part[],
  width: number,
  height: number,
  landmarks: Landmarks,
): Promise<RigPart[]> {
  if (desktop)
    return invoke("build_rig", {
      parts: parts.map((p) => ({
        id: p.id,
        mask: base64(p.mask),
        fillMask: p.fillMask ? base64(p.fillMask) : null,
      })),
      width,
      height,
      landmarks,
    });
  await new Promise((resolve) => setTimeout(resolve, 20));
  return parts.map((part) => {
    const eff = effectiveMask(part);
    const mesh = meshForPart(eff, width, height, 14, landmarks, part);
    return {
      id: part.id,
      mesh,
      samples: PARAM_DEFS.flatMap((def) =>
        [...new Set([def.previewMin, def.value, def.previewMax])].map(
          (value) => ({
            parameter: def.key,
            value,
            vertices: mesh.vertices.map((p) =>
              deform(p, part, landmarks, { ...REST, [def.key]: value }),
            ),
          }),
        ),
      ),
      quality: checkMeshQuality(mesh, eff, width, height, part, landmarks),
    };
  });
}

/** T13b exposure-map summary (mirrors Rust `rig::exposure::ExposureMap`, per-pixel data omitted). */
export interface ExposureLayerSummary {
  id: string;
  role: string | null;
  hiddenPx: number;
  exposedPx: number;
  unexposedHiddenPx: number;
  meanFrequency: number;
  maxFrequency: number;
  maxDepthPx: number | null;
  p95DepthPx: number | null;
  restExposedPx: number;
  /** T9: overlay texels (upper-layer fill over a lower layer's visible seam pixel), not hidden. */
  overlayPx?: number;
}

export interface ExposureOverlayResult {
  summary: {
    width: number;
    height: number;
    poses: number;
    render: string;
    layers: ExposureLayerSummary[];
    hiddenPxTotal: number;
    exposedPxTotal: number;
    maxDepthPx: number;
    holes: {
      restEnclosedPx: number;
      restUncoveredFgPx: number;
      holePxTotal: number;
      holePxMax: number;
      holePoses: number;
      holeComponentsTotal: number;
      holeComponentsMax: number;
      worstPose: string | null;
    };
  };
  seamHoles: number;
  seamLowerMisses: number;
  /** Transparent canvas-sized heatmap PNG data URL. */
  heatmap: string;
}

/**
 * T13b: exposure map over the shared pose envelope (Rust ArtMesh render), for the advanced-screen
 * heatmap overlay. Desktop only — the browser fallback has no exposure renderer and returns `null`.
 * `parts` must be bottom-to-top in composite order.
 */
export async function computeExposureMap(
  parts: Part[],
  width: number,
  height: number,
  landmarks: Landmarks,
): Promise<ExposureOverlayResult | null> {
  if (!desktop) return null;
  return invoke<ExposureOverlayResult>("compute_exposure_map", {
    parts: parts.map((p) => ({
      id: p.id,
      mask: base64(p.mask),
      fillMask: p.fillMask ? base64(p.fillMask) : null,
    })),
    width,
    height,
    landmarks,
  });
}

export async function fillOcclusionNative(
  data: Uint8ClampedArray,
  visibleMask: Uint8Array,
  width: number,
  height: number,
  radius: number,
  partId?: string,
): Promise<FillOcclusionResult> {
  if (!desktop) {
    return expandFillSeparated(data, visibleMask, width, height, radius, partId);
  }
  const result = await invoke<{
    rgba: string;
    fillMask: string;
    combinedMask: string;
    filledPixels: number;
    fillKind: FillKind;
    fillRegion?: { kind: string; synth: string } | null;
    warning?: string | null;
  }>("fill_part_occlusion", {
    rgba: base64(data),
    visibleMask: base64(visibleMask),
    width,
    height,
    radius,
    partId: partId ?? null,
  });
  return {
    data: new Uint8ClampedArray(unbase64(result.rgba)),
    fillMask: unbase64(result.fillMask),
    combinedMask: unbase64(result.combinedMask),
    filledPixels: result.filledPixels,
    fillKind: result.fillKind,
    ...parseFillRegion(result.fillRegion),
    warning: result.warning ?? undefined,
  };
}

export async function fillNative(
  data: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  radius: number,
) {
  const occ = await fillOcclusionNative(data, mask, width, height, radius);
  return {
    data: occ.data,
    mask: occ.combinedMask,
  };
}
export async function saveFile(
  name: string,
  bytes: Uint8Array,
  mime = "application/octet-stream",
): Promise<boolean> {
  if (desktop)
    return !!(await invoke<string | null>("save_artifact", {
      name,
      bytes: Array.from(bytes),
    }));
  const url = URL.createObjectURL(
    new Blob([new Uint8Array(bytes)], { type: mime }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
  return true;
}

export interface FaceAnalysisResponse {
  jobId: number;
  imageHash: string;
  status: AnalysisStatus;
  message: string;
  candidates: FaceCandidate[];
  selectedFaceId?: number | null;
  bestRawScore: number;
  points28: Landmark28Point[];
  rigLandmarks?: Landmarks | null;
  rigProvenance?: LandmarkProvenance | null;
  modelHashes: Record<string, string>;
  postprocessVersions: Record<string, string>;
}

export async function chooseOnnxModel(title = "ONNX 모델 선택") {
  if (!desktop) return null;
  return open({
    title,
    multiple: false,
    filters: [{ name: "ONNX model", extensions: ["onnx"] }],
  });
}

export async function getModelStatus(): Promise<ModelStatusEntry[]> {
  if (!desktop) return [];
  return invoke<ModelStatusEntry[]>("get_model_status");
}

export async function setModelPath(
  role: string,
  path: string | null,
): Promise<ModelStatusEntry[]> {
  if (!desktop) return [];
  return invoke<ModelStatusEntry[]>("set_model_path", { role, path });
}

export async function downloadModel(role: string): Promise<ModelStatusEntry[]> {
  if (!desktop) {
    throw new Error("모델 다운로드는 데스크톱 앱에서 실행할 수 있습니다.");
  }
  return invoke<ModelStatusEntry[]>("download_model", { role });
}

export async function cancelAnalysis(): Promise<void> {
  if (!desktop) return;
  await invoke("cancel_analysis");
}

export async function analyzeFace(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  faceIndex: number | null,
  jobId: number,
  signal: AbortSignal,
): Promise<FaceAnalysisResponse> {
  if (!desktop) {
    return {
      jobId,
      imageHash: "",
      status: "browser-fallback",
      message:
        "브라우저 프리뷰 모드입니다. 로컬 ONNX 자동 검출은 Tauri 데스크톱 앱에서 실행되며, 현재 화면에서는 수동 드래그 배치를 사용합니다.",
      candidates: [],
      selectedFaceId: null,
      bestRawScore: 0,
      points28: [],
      rigLandmarks: null,
      rigProvenance: null,
      modelHashes: {},
      postprocessVersions: {},
    };
  }
  const result = await invoke<FaceAnalysisResponse>("analyze_face", {
    rgba: base64(pixels),
    width,
    height,
    faceIndex,
    jobId,
  });
  signal.throwIfAborted();
  return result;
}

export interface UpdateCheckResult {
  status: "up-to-date" | "available" | "unsigned-release" | "unavailable";
  currentVersion: string;
  latestVersion?: string | null;
  date?: string | null;
  notes?: string | null;
  signed: boolean;
  endpointUsed: string;
  releaseUrl: string;
  message: string;
}

export type UpdateDownloadEvent =
  | { event: "Started"; data: { contentLength?: number | null } }
  | { event: "Progress"; data: { downloaded: number; contentLength?: number | null } }
  | { event: "Finished" };

export async function getAppVersion(): Promise<string> {
  if (!desktop) return "0.2.0-3";
  try {
    return await invoke<string>("get_app_version");
  } catch {
    return "0.2.0-3";
  }
}

export async function checkAppUpdate(
  customEndpoint?: string,
): Promise<UpdateCheckResult> {
  if (!desktop) {
    return {
      status: "unavailable",
      currentVersion: "0.2.0-3",
      latestVersion: null,
      signed: false,
      endpointUsed: customEndpoint || "",
      releaseUrl: "https://github.com/jinhoOps/flat2rig/releases",
      message: "자동 업데이트 확인은 데스크톱 앱(.msi) 환경에서 실행됩니다.",
    };
  }
  return invoke<UpdateCheckResult>("check_app_update", {
    customEndpoint: customEndpoint?.trim() || null,
  });
}

export async function installAppUpdate(
  onProgress: (ev: UpdateDownloadEvent) => void,
): Promise<void> {
  if (!desktop) {
    throw new Error("자동 업데이트 설치는 데스크톱 앱에서 실행됩니다.");
  }
  const channel = new Channel<UpdateDownloadEvent>();
  channel.onmessage = onProgress;
  await invoke("install_app_update", { onEvent: channel });
}

export async function relaunchApp(): Promise<void> {
  if (!desktop) return;
  await invoke("relaunch_app");
}

export async function detectLandmarksForFace(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  bbox: [number, number, number, number],
  cropBbox: [number, number, number, number],
  selectedFaceId: number,
  jobId: number,
  signal: AbortSignal,
): Promise<FaceAnalysisResponse> {
  if (!desktop) {
    throw new Error("ONNX 랜드마크 재추출은 데스크톱 앱에서 지원됩니다.");
  }
  const result = await invoke<FaceAnalysisResponse>(
    "detect_landmarks_for_face",
    {
      rgba: base64(pixels),
      width,
      height,
      bbox,
      cropBbox,
      selectedFaceId,
      jobId,
    },
  );
  signal.throwIfAborted();
  return result;
}
