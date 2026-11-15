import { zip, unzipSync, strToU8, strFromU8, type Zippable } from "fflate";
import { writePsdUint8Array, readPsd } from "ag-psd";
import {
  ALLOWED_BLEND_MODES,
  createDefaultProjectRigState,
  describeDynamicPart,
  HAIR_ROLE_META,
  isHairStageLayer,
  PART_DEFS,
  partPipelineStage,
  PARAM_DEFS,
  PARAM_GROUPS,
  REST,
  stageIncludesLayer,
  type ExpressionPreset,
  type HairRole,
  type Parameters,
  type PartRole,
  type PipelineStage,
  type PlaybackSettings,
  type Project,
  type Part,
  type RigProfile,
  type ShapePreset,
  type ShapeRegion,
  type ShapeSubstituteAsset,
  type ShapeSubstituteLayer,
} from "./types";
import { bounds, effectiveMask } from "./engine";
import { canvas, partCanvases, pngBytes } from "./images";
import { buildRig } from "./native";

const MAX_ZIP_FILE_BYTES = 160 * 1024 * 1024;
const MAX_UNCOMPRESSED_ZIP_BYTES = 450 * 1024 * 1024;
const MAX_SHAPE_PRESETS = 256;
const MAX_EXPRESSION_PRESETS = 128;
const MAX_SHAPE_ASSETS = 256;
const MAX_ASSET_LAYERS = 16;
const SAFE_ZIP_ID = /^[a-zA-Z0-9_-]{1,80}$/;
const SHAPE_REGIONS: readonly ShapeRegion[] = ["eye_L", "eye_R", "mouth"];
const PARAM_DEF_BOUNDS = new Map<keyof Parameters, { min: number; max: number }>(
  PARAM_DEFS.map((d) => [d.key, { min: d.min, max: d.max }]),
);

export const safeName = (name: string) =>
  name.replace(/[^\p{L}\p{N}_.-]+/gu, "_").slice(0, 90) || "character";
export function pack(files: Zippable): Promise<Uint8Array> {
  return new Promise((resolve, reject) =>
    zip(files, { level: 6 }, (error, result) =>
      error ? reject(error) : resolve(result),
    ),
  );
}
export async function saveProject(project: Project) {
  const files: Zippable = {};
  const { parts, shapeAssets, ...meta } = project;
  const serializedParts = parts.map(({ mask: _, fillMask: __, ...p }) => p);
  const serializedAssets: Record<string, ShapeSubstituteAsset> | undefined =
    shapeAssets
      ? Object.fromEntries(
          Object.entries(shapeAssets).map(([assetId, asset]) => {
            if (!SAFE_ZIP_ID.test(assetId) || asset.assetId !== assetId) {
              throw new Error(`잘못된 쉐이프 에셋 ID입니다: ${assetId}`);
            }
            const cleanLayers: ShapeSubstituteLayer[] = asset.layers.map(
              (layer) => {
                if (!SAFE_ZIP_ID.test(layer.partId)) {
                  throw new Error(`잘못된 쉐이프 파츠 ID입니다: ${layer.partId}`);
                }
                if (layer.mask) {
                  files[`shapes/${assetId}/${layer.partId}.bin`] = layer.mask;
                }
                if (layer.fillMask) {
                  files[`shapes/${assetId}/${layer.partId}.fill.bin`] =
                    layer.fillMask;
                }
                if (layer.pngBytes) {
                  files[`shapes/${assetId}/${layer.partId}.png`] =
                    layer.pngBytes;
                }
                const {
                  mask: _m,
                  fillMask: _fm,
                  pngBytes: _png,
                  ...restLayer
                } = layer;
                return {
                  ...restLayer,
                  hasMask: Boolean(layer.mask),
                  hasFillMask: Boolean(layer.fillMask),
                  hasTexturePng: Boolean(layer.pngBytes),
                };
              },
            );
            return [
              assetId,
              {
                ...asset,
                layers: cleanLayers,
              },
            ];
          }),
        )
      : undefined;
  files["project.json"] = strToU8(
    JSON.stringify({
      ...meta,
      schemaVersion: 3,
      parts: serializedParts,
      ...(serializedAssets ? { shapeAssets: serializedAssets } : {}),
    }),
  );
  for (const p of parts) {
    files[`masks/${p.id}.bin`] = p.mask;
    if (p.fillMask) {
      files[`masks/${p.id}.fill.bin`] = p.fillMask;
    }
  }
  return pack(files);
}
function validDataURL(v: unknown): v is string {
  return (
    typeof v === "string" &&
    /^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=]+$/.test(v)
  );
}
function inferLegacyProportionMode(p: Record<string, unknown>): "ld" | "sd" {
  const sm = p.styleMatch as
    | { resolved?: { proportionMode?: unknown } }
    | undefined;
  const sel = p.styleSelection as { proportionMode?: unknown } | undefined;
  if (sm?.resolved?.proportionMode === "sd" || sel?.proportionMode === "sd") {
    return "sd";
  }
  return "ld";
}
function migrateLegacyToV3(
  p: Record<string, unknown>,
  parts: Part[],
  fromVersion: 1 | 2,
): Project {
  const landmarks = p.landmarks as Project["landmarks"];
  const width = p.width as number;
  const height = p.height as number;
  const origW = (p.originalWidth as number) || width;
  const origH = (p.originalHeight as number) || height;
  const defaults = createDefaultProjectRigState(inferLegacyProportionMode(p));
  return {
    schemaVersion: 3,
    name: p.name as string,
    width,
    height,
    originalWidth: origW,
    originalHeight: origH,
    source: p.source as string,
    original: p.original as string,
    imageHash: typeof p.imageHash === "string" ? p.imageHash : undefined,
    workToOriginalTransform:
      (p.workToOriginalTransform as Project["workToOriginalTransform"]) ?? {
        matrix2: [origW / width, 0, 0, origH / height],
        translation: [0, 0],
      },
    landmarks,
    landmarkProvenance:
      fromVersion === 1
        ? {
            eyeL: "manual",
            eyeR: "manual",
            mouth: "manual",
            chin: "manual",
            crown: "manual",
          }
        : ((p.landmarkProvenance as Project["landmarkProvenance"]) ?? {
            eyeL: "estimated",
            eyeR: "estimated",
            mouth: "estimated",
            chin: "estimated",
            crown: "estimated",
          }),
    landmarks28: p.landmarks28 as Project["landmarks28"],
    faceCandidates: p.faceCandidates as Project["faceCandidates"],
    selectedFaceId:
      typeof p.selectedFaceId === "number" ? p.selectedFaceId : undefined,
    parts: parts.map((part, index) => ({
      ...part,
      blendMode: part.blendMode ?? "normal",
      provenance: part.provenance ?? "geometry",
      fillKind: part.fillKind ?? (part.generated ? "neighbor" : "none"),
      compositeOrder: part.compositeOrder ?? index,
    })),
    revision: 0,
    provider: (p.provider as string) || `migrated-v${fromVersion}`,
    modelHash: typeof p.modelHash === "string" ? p.modelHash : undefined,
    modelHashes: p.modelHashes as Project["modelHashes"],
    postprocessVersions:
      p.postprocessVersions as Project["postprocessVersions"],
    analysisStatus: (p.analysisStatus as Project["analysisStatus"]) ?? "idle",
    analysisNote:
      typeof p.analysisNote === "string" ? p.analysisNote : undefined,
    detailMode: p.detailMode as Project["detailMode"],
    rigStage: p.rigStage as Project["rigStage"],
    styleMatch: p.styleMatch as Project["styleMatch"],
    styleSelection: p.styleSelection as Project["styleSelection"],
    ...defaults,
    createdAt: (p.createdAt as string) || new Date().toISOString(),
  };
}

function validPngBytesForBounds(
  png: Uint8Array,
  expectedWidth: number,
  expectedHeight: number,
): boolean {
  if (png.length < 24) return false;
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < sig.length; i++) {
    if (png[i] !== sig[i]) return false;
  }
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  return (
    view.getUint32(16, false) === expectedWidth &&
    view.getUint32(20, false) === expectedHeight
  );
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function validateParamMap(
  raw: unknown,
  context: string,
): Partial<Parameters> {
  if (!isPlainObject(raw)) {
    throw new Error(`잘못된 ${context} 파라미터 정보입니다.`);
  }
  const out: Partial<Parameters> = {};
  for (const [k, v] of Object.entries(raw)) {
    const key = k as keyof Parameters;
    const range = PARAM_DEF_BOUNDS.get(key);
    if (!range || typeof v !== "number" || !Number.isFinite(v)) {
      throw new Error(`잘못된 ${context} 파라미터 키 또는 값(${k})입니다.`);
    }
    if (v < range.min - 1e-4 || v > range.max + 1e-4) {
      throw new Error(
        `${context} 파라미터(${k}=${v})가 허용 범위(${range.min}..${range.max})를 벗어났습니다.`,
      );
    }
    out[key] = v;
  }
  return out;
}

function validateShapeHandles(raw: unknown): ShapePreset["handles"] {
  if (!isPlainObject(raw)) {
    throw new Error("잘못된 쉐이프 핸들 정보입니다.");
  }
  const check = (
    name: keyof ShapePreset["handles"],
    min: number,
    max: number,
  ): number => {
    const v = raw[name];
    if (typeof v !== "number" || !Number.isFinite(v) || v < min || v > max) {
      throw new Error(`잘못된 쉐이프 핸들 값(${String(name)})입니다.`);
    }
    return v;
  };
  return {
    width: check("width", 0.25, 2.5),
    open: check("open", 0, 2),
    roundness: check("roundness", 0, 1),
    corner: check("corner", -1.5, 1.5),
    upperCurve: check("upperCurve", -1.5, 1.5),
    lowerCurve: check("lowerCurve", -1.5, 1.5),
    interiorVisibility: check("interiorVisibility", 0, 1),
    offsetX: check("offsetX", -1, 1),
    offsetY: check("offsetY", -1, 1),
  };
}

function hydrateAndValidateV3Fields(
  p: Record<string, unknown>,
  files: Record<string, Uint8Array>,
  width: number,
  height: number,
): Pick<
  Project,
  | "rigProfile"
  | "shapePresets"
  | "shapeAssets"
  | "expressions"
  | "activeExpressionId"
  | "playback"
> {
  let shapeAssets: Record<string, ShapeSubstituteAsset> | undefined;
  if (p.shapeAssets !== undefined) {
    if (!isPlainObject(p.shapeAssets)) {
      throw new Error("잘못된 쉐이프 대체 에셋 정보입니다.");
    }
    const entries = Object.entries(p.shapeAssets);
    if (entries.length > MAX_SHAPE_ASSETS) {
      throw new Error("쉐이프 대체 에셋 개수가 제한을 초과했습니다.");
    }
    shapeAssets = {};
    for (const [assetId, rawAsset] of entries) {
      if (
        !SAFE_ZIP_ID.test(assetId) ||
        !isPlainObject(rawAsset) ||
        rawAsset.assetId !== assetId ||
        !SHAPE_REGIONS.includes(rawAsset.region as ShapeRegion) ||
        typeof rawAsset.recipeVersion !== "number" ||
        !Number.isInteger(rawAsset.recipeVersion) ||
        rawAsset.recipeVersion < 1 ||
        !Array.isArray(rawAsset.hidesPartIds) ||
        rawAsset.hidesPartIds.length > 64 ||
        rawAsset.hidesPartIds.some(
          (id) => typeof id !== "string" || !SAFE_ZIP_ID.test(id),
        ) ||
        !Array.isArray(rawAsset.layers) ||
        rawAsset.layers.length < 1 ||
        rawAsset.layers.length > MAX_ASSET_LAYERS
      ) {
        throw new Error(`잘못된 쉐이프 대체 에셋(${assetId})입니다.`);
      }
      const seenLayerIds = new Set<string>();
      const hydratedLayers: ShapeSubstituteLayer[] = [];
      for (const rawLayer of rawAsset.layers) {
        if (
          !isPlainObject(rawLayer) ||
          typeof rawLayer.partId !== "string" ||
          !SAFE_ZIP_ID.test(rawLayer.partId) ||
          seenLayerIds.has(rawLayer.partId) ||
          (rawLayer.blendMode !== undefined &&
            !ALLOWED_BLEND_MODES.includes(
              rawLayer.blendMode as (typeof ALLOWED_BLEND_MODES)[number],
            ))
        ) {
          throw new Error(`잘못된 쉐이프 대체 레이어(${assetId})입니다.`);
        }
        seenLayerIds.add(rawLayer.partId);
        const b = rawLayer.bounds;
        if (
          !isPlainObject(b) ||
          !Number.isInteger(b.left) ||
          !Number.isInteger(b.top) ||
          !Number.isInteger(b.width) ||
          !Number.isInteger(b.height) ||
          (b.width as number) < 1 ||
          (b.height as number) < 1 ||
          (b.left as number) < 0 ||
          (b.top as number) < 0 ||
          (b.left as number) + (b.width as number) > width ||
          (b.top as number) + (b.height as number) > height
        ) {
          throw new Error(
            `쉐이프 대체 레이어 경계(bounds)가 캔버스 범위를 벗어났습니다: ${assetId}/${rawLayer.partId}`,
          );
        }
        const bw = b.width as number;
        const bh = b.height as number;
        const hasMask = rawLayer.hasMask !== false;
        const maskFile = files[`shapes/${assetId}/${rawLayer.partId}.bin`];
        if (hasMask) {
          if (!maskFile || maskFile.length !== bw * bh) {
            throw new Error(
              `쉐이프 대체 마스크 크기가 일치하지 않습니다: ${assetId}/${rawLayer.partId}`,
            );
          }
        } else if (maskFile && maskFile.length !== bw * bh) {
          throw new Error(
            `쉐이프 대체 마스크 크기가 일치하지 않습니다: ${assetId}/${rawLayer.partId}`,
          );
        }
        const fillFile =
          files[`shapes/${assetId}/${rawLayer.partId}.fill.bin`];
        if (
          (rawLayer.hasFillMask === true && !fillFile) ||
          (fillFile && fillFile.length !== bw * bh)
        ) {
          throw new Error(
            `쉐이프 대체 가림 보완 마스크가 손상되었습니다: ${assetId}/${rawLayer.partId}`,
          );
        }
        const pngFile = files[`shapes/${assetId}/${rawLayer.partId}.png`];
        if (rawLayer.hasTexturePng === true) {
          if (!pngFile || !validPngBytesForBounds(pngFile, bw, bh)) {
            throw new Error(
              `쉐이프 대체 PNG 이미지가 손상되었거나 크기가 일치하지 않습니다: ${assetId}/${rawLayer.partId}`,
            );
          }
        } else if (pngFile && !validPngBytesForBounds(pngFile, bw, bh)) {
          throw new Error(
            `쉐이프 대체 PNG 이미지가 손상되었거나 크기가 일치하지 않습니다: ${assetId}/${rawLayer.partId}`,
          );
        }
        if (!maskFile && !pngFile) {
          throw new Error(
            `쉐이프 대체 레이어에 마스크나 텍스처가 없습니다: ${assetId}/${rawLayer.partId}`,
          );
        }
        hydratedLayers.push({
          partId: rawLayer.partId,
          ...(rawLayer.blendMode !== undefined
            ? {
                blendMode:
                  rawLayer.blendMode as ShapeSubstituteLayer["blendMode"],
              }
            : {}),
          bounds: {
            left: b.left as number,
            top: b.top as number,
            width: bw,
            height: bh,
          },
          ...(Array.isArray(rawLayer.hidesPartIds)
            ? { hidesPartIds: rawLayer.hidesPartIds as string[] }
            : {}),
          ...(maskFile ? { mask: maskFile, hasMask: true } : { hasMask: false }),
          ...(fillFile ? { fillMask: fillFile, hasFillMask: true } : {}),
          ...(pngFile ? { pngBytes: pngFile, hasTexturePng: true } : {}),
        });
      }
      shapeAssets[assetId] = {
        assetId,
        region: rawAsset.region as ShapeRegion,
        recipeVersion: rawAsset.recipeVersion,
        hidesPartIds: rawAsset.hidesPartIds as string[],
        layers: hydratedLayers,
      };
    }
  }

  for (const zipPath of Object.keys(files)) {
    if (!zipPath.startsWith("shapes/")) continue;
    const parts = zipPath.split("/");
    const assetId = parts[1];
    if (!assetId || !shapeAssets || !shapeAssets[assetId]) {
      throw new Error(
        `참조되지 않은 쉐이프 에셋 파일이 포함되어 있습니다: ${zipPath}`,
      );
    }
  }

  if (
    !Array.isArray(p.shapePresets) ||
    p.shapePresets.length < 1 ||
    p.shapePresets.length > MAX_SHAPE_PRESETS
  ) {
    throw new Error("잘못된 쉐이프 프리셋 목록입니다.");
  }
  const shapeById = new Map<string, ShapePreset>();
  const shapePresets: ShapePreset[] = p.shapePresets.map((raw) => {
    if (
      !isPlainObject(raw) ||
      typeof raw.id !== "string" ||
      !SAFE_ZIP_ID.test(raw.id) ||
      shapeById.has(raw.id) ||
      !SHAPE_REGIONS.includes(raw.region as ShapeRegion) ||
      typeof raw.label !== "string" ||
      raw.label.trim().length < 1 ||
      raw.label.length > 120 ||
      typeof raw.kind !== "string" ||
      typeof raw.recipeVersion !== "number" ||
      !Number.isInteger(raw.recipeVersion) ||
      raw.recipeVersion < 1
    ) {
      throw new Error("잘못된 쉐이프 프리셋 형식입니다.");
    }
    if (raw.assetId !== undefined) {
      if (
        typeof raw.assetId !== "string" ||
        !shapeAssets ||
        !shapeAssets[raw.assetId] ||
        shapeAssets[raw.assetId].region !== raw.region
      ) {
        throw new Error(
          `쉐이프 프리셋(${raw.id})이 유효하지 않은 대체 에셋(${String(raw.assetId)})을 참조합니다.`,
        );
      }
    }
    const preset: ShapePreset = {
      id: raw.id,
      region: raw.region as ShapeRegion,
      label: raw.label,
      kind: raw.kind as ShapePreset["kind"],
      recipeVersion: raw.recipeVersion,
      ...(typeof raw.builtIn === "boolean" ? { builtIn: raw.builtIn } : {}),
      ...(typeof raw.mirrorLinked === "boolean"
        ? { mirrorLinked: raw.mirrorLinked }
        : {}),
      handles: validateShapeHandles(raw.handles),
      parameterOverrides: validateParamMap(
        raw.parameterOverrides,
        `쉐이프(${raw.id})`,
      ),
      ...(raw.assetId !== undefined ? { assetId: raw.assetId as string } : {}),
    };
    shapeById.set(preset.id, preset);
    return preset;
  });

  if (
    !Array.isArray(p.expressions) ||
    p.expressions.length < 1 ||
    p.expressions.length > MAX_EXPRESSION_PRESETS
  ) {
    throw new Error("잘못된 표정 프리셋 목록입니다.");
  }
  const expById = new Map<string, ExpressionPreset>();
  const expressions: ExpressionPreset[] = p.expressions.map((raw) => {
    if (
      !isPlainObject(raw) ||
      typeof raw.id !== "string" ||
      !SAFE_ZIP_ID.test(raw.id) ||
      expById.has(raw.id) ||
      typeof raw.name !== "string" ||
      raw.name.trim().length < 1 ||
      raw.name.length > 120 ||
      !isPlainObject(raw.shapeIds) ||
      typeof raw.strength !== "number" ||
      !Number.isFinite(raw.strength) ||
      raw.strength < 0 ||
      raw.strength > 1.5 ||
      typeof raw.transitionMs !== "number" ||
      !Number.isFinite(raw.transitionMs) ||
      raw.transitionMs < 0 ||
      raw.transitionMs > 3000
    ) {
      throw new Error("잘못된 표정 프리셋 형식입니다.");
    }
    const shapeIds: ExpressionPreset["shapeIds"] = {};
    for (const region of SHAPE_REGIONS) {
      const refId = raw.shapeIds[region];
      if (refId !== undefined) {
        if (
          typeof refId !== "string" ||
          shapeById.get(refId)?.region !== region
        ) {
          throw new Error(
            `표정(${raw.id})이 존재하지 않거나 부위가 다른 쉐이프(${String(refId)})를 참조합니다.`,
          );
        }
        shapeIds[region] = refId;
      }
    }
    const exp: ExpressionPreset = {
      id: raw.id,
      name: raw.name,
      ...(typeof raw.builtIn === "boolean" ? { builtIn: raw.builtIn } : {}),
      shapeIds,
      parameters: validateParamMap(raw.parameters, `표정(${raw.id})`),
      ...(isPlainObject(raw.handleTweaks)
        ? {
            handleTweaks:
              raw.handleTweaks as ExpressionPreset["handleTweaks"],
          }
        : {}),
      strength: raw.strength,
      transitionMs: raw.transitionMs,
      ...(typeof raw.shortcutKey === "string"
        ? { shortcutKey: raw.shortcutKey }
        : {}),
    };
    expById.set(exp.id, exp);
    return exp;
  });

  const rp = p.rigProfile;
  if (
    !isPlainObject(rp) ||
    (rp.proportionMode !== "ld" && rp.proportionMode !== "sd") ||
    !isPlainObject(rp.defaultShapeIds) ||
    !isPlainObject(rp.neutralParameters) ||
    !isPlainObject(rp.safeBounds) ||
    !isPlainObject(rp.motionScale)
  ) {
    throw new Error("잘못된 리깅 프로필 정보입니다.");
  }
  for (const region of SHAPE_REGIONS) {
    const defId = rp.defaultShapeIds[region];
    if (typeof defId !== "string" || shapeById.get(defId)?.region !== region) {
      throw new Error(
        `리깅 프로필 기본 쉐이프(${region}=${String(defId)}) 참조가 유효하지 않습니다.`,
      );
    }
  }
  const neutralRaw = validateParamMap(rp.neutralParameters, "기본 포즈");
  for (const k of Object.keys(REST) as (keyof Parameters)[]) {
    if (typeof neutralRaw[k] !== "number") {
      throw new Error(`리깅 기본 포즈에 필수 파라미터(${k})가 누락되었습니다.`);
    }
  }
  const validatedBounds: RigProfile["safeBounds"] = {};
  for (const [k, rawRange] of Object.entries(rp.safeBounds)) {
    const key = k as keyof Parameters;
    const defRange = PARAM_DEF_BOUNDS.get(key);
    if (
      !defRange ||
      !isPlainObject(rawRange) ||
      typeof rawRange.min !== "number" ||
      typeof rawRange.max !== "number" ||
      !Number.isFinite(rawRange.min) ||
      !Number.isFinite(rawRange.max) ||
      rawRange.min > rawRange.max ||
      rawRange.min < defRange.min - 1e-4 ||
      rawRange.max > defRange.max + 1e-4
    ) {
      throw new Error(`잘못된 리깅 안전 범위(${k})입니다.`);
    }
    validatedBounds[key] = { min: rawRange.min, max: rawRange.max };
  }
  const scaleKeys = ["head", "eyes", "mouth", "hair", "body"] as const;
  const motionScale = {} as RigProfile["motionScale"];
  for (const sk of scaleKeys) {
    const sv = rp.motionScale[sk];
    if (typeof sv !== "number" || !Number.isFinite(sv) || sv < 0 || sv > 3) {
      throw new Error(`잘못된 모션 강도 배율(${sk})입니다.`);
    }
    motionScale[sk] = sv;
  }
  const rigProfile: RigProfile = {
    proportionMode: rp.proportionMode,
    defaultShapeIds: {
      eye_L: rp.defaultShapeIds.eye_L as string,
      eye_R: rp.defaultShapeIds.eye_R as string,
      mouth: rp.defaultShapeIds.mouth as string,
    },
    neutralParameters: neutralRaw as Parameters,
    safeBounds: validatedBounds,
    motionScale,
    ...(isPlainObject(rp.clippingTargets)
      ? {
          clippingTargets: rp.clippingTargets as Record<string, string>,
        }
      : {}),
  };

  if (
    typeof p.activeExpressionId !== "string" ||
    !expById.has(p.activeExpressionId)
  ) {
    throw new Error("활성 표정 ID가 표정 목록에 없습니다.");
  }
  const pb = p.playback;
  if (
    !isPlainObject(pb) ||
    typeof pb.defaultExpressionId !== "string" ||
    !expById.has(pb.defaultExpressionId) ||
    typeof pb.autoBlink !== "boolean" ||
    typeof pb.autoBreath !== "boolean" ||
    typeof pb.autoIdle !== "boolean" ||
    typeof pb.blinkIntervalSec !== "number" ||
    !Number.isFinite(pb.blinkIntervalSec) ||
    pb.blinkIntervalSec < 0.5 ||
    pb.blinkIntervalSec > 30 ||
    !isPlainObject(pb.shortcuts)
  ) {
    throw new Error("잘못된 재생 설정입니다.");
  }
  const shortcuts: Record<string, string> = {};
  for (const [key, expId] of Object.entries(pb.shortcuts)) {
    if (typeof expId !== "string" || !expById.has(expId)) {
      throw new Error(`재생 단축키(${key})가 존재하지 않는 표정을 참조합니다.`);
    }
    shortcuts[key] = expId;
  }
  const playback: PlaybackSettings = {
    defaultExpressionId: pb.defaultExpressionId,
    autoBlink: pb.autoBlink,
    blinkIntervalSec: pb.blinkIntervalSec,
    autoBreath: pb.autoBreath,
    autoIdle: pb.autoIdle,
    ...(typeof pb.autoMouthDemo === "boolean"
      ? { autoMouthDemo: pb.autoMouthDemo }
      : {}),
    shortcuts,
  };

  return {
    rigProfile,
    shapePresets,
    ...(shapeAssets ? { shapeAssets } : {}),
    expressions,
    activeExpressionId: p.activeExpressionId,
    playback,
  };
}

export function loadProject(bytes: Uint8Array): Project {
  if (bytes.length > MAX_ZIP_FILE_BYTES)
    throw new Error("프로젝트 파일이 너무 큽니다.");
  let total = 0;
  const files = unzipSync(bytes, {
    filter: (f) => {
      total += f.originalSize;
      if (
        total > MAX_UNCOMPRESSED_ZIP_BYTES ||
        f.originalSize > 100 * 1024 * 1024
      )
        throw new Error("프로젝트 압축 해제 한도를 초과했습니다.");
      return true;
    },
  });
  let actualTotal = 0;
  for (const entry of Object.values(files)) {
    actualTotal += entry.byteLength;
    if (actualTotal > MAX_UNCOMPRESSED_ZIP_BYTES) {
      throw new Error("프로젝트 압축 해제 한도를 초과했습니다.");
    }
  }
  if (!files["project.json"]) throw new Error("flat2rig 프로젝트가 아닙니다.");
  const p = JSON.parse(strFromU8(files["project.json"]));
  if (
    (p.schemaVersion !== 1 && p.schemaVersion !== 2 && p.schemaVersion !== 3) ||
    !Number.isInteger(p.width) ||
    !Number.isInteger(p.height) ||
    p.width < 1 ||
    p.height < 1 ||
    p.width > 2048 ||
    p.height > 2048 ||
    !validDataURL(p.source) ||
    !validDataURL(p.original)
  )
    throw new Error("지원하지 않는 프로젝트 형식입니다.");
  if (
    typeof p.name !== "string" ||
    p.name.length > 200 ||
    !Array.isArray(p.parts) ||
    p.parts.length > 64
  )
    throw new Error("잘못된 프로젝트 정보입니다.");
  const ids = new Set<string>();
  const validProvenances = ["model", "prompted", "geometry", "manual"];
  const validFillKinds = ["none", "neighbor", "external", "template"];
  for (let idx = 0; idx < p.parts.length; idx++) {
    const part = p.parts[idx];
    if (
      typeof part.id !== "string" ||
      !/^[\w-]{1,80}$/.test(part.id) ||
      ids.has(part.id) ||
      typeof part.name !== "string" ||
      typeof part.group !== "string" ||
      !/^#[0-9a-f]{6}$/i.test(part.color) ||
      !["draft", "approved", "missing"].includes(part.status) ||
      typeof part.visible !== "boolean" ||
      (part.texture && !validDataURL(part.texture)) ||
      (part.provenance && !validProvenances.includes(part.provenance)) ||
      (part.fillKind && !validFillKinds.includes(part.fillKind)) ||
      (part.blendMode && !ALLOWED_BLEND_MODES.includes(part.blendMode))
    )
      throw new Error("잘못된 파츠 정보입니다.");
    ids.add(part.id);
    const mask = files[`masks/${part.id}.bin`];
    if (!mask || mask.length !== p.width * p.height)
      throw new Error("파츠 마스크 크기가 일치하지 않습니다.");
    part.mask = mask;
    const fillMask = files[`masks/${part.id}.fill.bin`];
    if (fillMask) {
      if (fillMask.length !== p.width * p.height)
        throw new Error("가림 보완 마스크 크기가 일치하지 않습니다.");
      part.fillMask = fillMask;
    }
    if (part.compositeOrder === undefined) {
      part.compositeOrder = idx;
    }
  }
  for (const id of ["eyeL", "eyeR", "mouth", "chin", "crown"]) {
    const pt = p.landmarks?.[id];
    if (
      !pt ||
      !Number.isFinite(pt.x) ||
      !Number.isFinite(pt.y) ||
      pt.x < 0 ||
      pt.y < 0 ||
      pt.x >= p.width ||
      pt.y >= p.height
    )
      throw new Error("잘못된 랜드마크입니다.");
  }
  if (p.schemaVersion === 1 || p.schemaVersion === 2) {
    return migrateLegacyToV3(p, p.parts as Part[], p.schemaVersion);
  }
  const v3Fields = hydrateAndValidateV3Fields(p, files, p.width, p.height);
  return {
    ...(p as Project),
    ...v3Fields,
    schemaVersion: 3,
    revision: 0,
  };
}
function quote(value: unknown) {
  return `"${String(value ?? "").replace(/"/g, '""')}"`;
}
export function referenceCsv() {
  const groupLabels = new Map<string, string>(
    PARAM_GROUPS.map((g) => [g.id, `${g.label} (${g.id})`]),
  );
  return (
    "\uFEFF" +
    [
      [
        "ID",
        "Name",
        "Group",
        "Minimum",
        "Default",
        "Maximum",
        "Preview minimum",
        "Preview maximum",
      ],
      ...PARAM_DEFS.map((p) => [
        p.id,
        p.name,
        groupLabels.get(p.groupId) ?? p.groupId,
        p.min,
        p.value,
        p.max,
        p.previewMin,
        p.previewMax,
      ]),
    ]
      .map((row) => row.map(quote).join(","))
      .join("\r\n")
  );
}
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [],
    field = "",
    quoted = false;
  text = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = !quoted;
    } else if (c === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((c === "\r" || c === "\n") && !quoted) {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      if (row.some(Boolean)) rows.push(row);
      row = [];
      field = "";
    } else field += c;
  }
  if (quoted) throw new Error("CSV 따옴표가 닫히지 않았습니다.");
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}
export interface CsvTemplate {
  headers: string[];
  sample: string[];
  columns: Record<string, number>;
}
export function readCsvTemplate(text: string): CsvTemplate {
  const rows = parseCsv(text);
  const aliases: Record<string, string[]> = {
    guid: ["guid"],
    type: ["type", "종류", "타입", "タイプ"],
    id: ["id"],
    name: ["name", "이름", "名前"],
    group: ["group", "그룹", "グループ"],
    min: ["minimum", "minimumvalue", "min", "최소값", "最小値"],
    value: ["default", "defaultvalue", "기본값", "デフォルト値"],
    max: ["maximum", "maximumvalue", "max", "최대값", "最大値"],
    repeat: ["repeat", "반복", "繰り返し"],
    description: ["description", "설명", "説明"],
  };
  const headerIndex = rows.findIndex(
    (row) =>
      row.some((c) => c.trim().toLowerCase() === "guid") &&
      row.some((c) => c.trim().toLowerCase() === "id"),
  );
  if (headerIndex < 0)
    throw new Error(
      "GUID/ID 헤더를 찾을 수 없습니다. Cubism 파라미터 일괄 설정 CSV를 사용하세요.",
    );
  const headers = rows[headerIndex],
    columns: Record<string, number> = {};
  for (const [key, names] of Object.entries(aliases)) {
    const i = headers.findIndex((h) =>
      names.includes(h.toLowerCase().replace(/[\s_-]/g, "")),
    );
    if (i >= 0) columns[key] = i;
  }
  if (
    ["guid", "type", "id", "name", "min", "value", "max"].some(
      (c) => columns[c] === undefined,
    )
  )
    throw new Error(
      "파라미터 CSV의 필수 열이 없습니다. 영어로 내보낸 CSV를 사용하세요.",
    );
  const sample = rows
    .slice(headerIndex + 1)
    .find((row) => /^Param/i.test(row[columns.id] || "") && row[columns.type]);
  if (!sample)
    throw new Error("Param으로 시작하는 ID의 파라미터 샘플 행이 필요합니다.");
  return { headers, sample, columns };
}
export function cubismCsv(template: CsvTemplate) {
  const { headers, sample, columns: c } = template;
  const groupLabels = new Map<string, string>(
    PARAM_GROUPS.map((g) => [g.id, g.label]),
  );
  const rows = PARAM_DEFS.map((p) => {
    const r = headers.map(() => "");
    const vals: Record<string, unknown> = {
      guid: "",
      type: sample[c.type],
      id: p.id,
      name: p.name,
      group: p.groupId,
      min: p.min,
      value: p.value,
      max: p.max,
      repeat: "false",
      description: `flat2rig draft (${groupLabels.get(p.groupId) ?? p.groupId}); keyforms must be created in Cubism`,
    };
    for (const [key, value] of Object.entries(vals))
      if (c[key] !== undefined) r[c[key]] = String(value);
    return r;
  });
  return (
    "\uFEFF" +
    [headers, ...rows].map((row) => row.map(quote).join(",")).join("\r\n")
  );
}
function orderedEntries(parts: Part[]) {
  return parts
    .map((part, index) => ({ part, index }))
    .sort(
      (a, b) =>
        (a.part.compositeOrder ?? a.index) - (b.part.compositeOrder ?? b.index),
    );
}

async function maskPng(mask: Uint8Array, w: number, h: number) {
  const m = canvas(w, h);
  const d = m.getContext("2d")!.createImageData(w, h);
  for (let j = 0; j < mask.length; j++) {
    d.data[j * 4] = d.data[j * 4 + 1] = d.data[j * 4 + 2] = mask[j];
    d.data[j * 4 + 3] = 255;
  }
  m.getContext("2d")!.putImageData(d, 0, 0);
  return pngBytes(m);
}

export async function exportPsd(
  project: Project,
  canvases?: HTMLCanvasElement[],
) {
  const images = canvases ?? (await partCanvases(project)),
    composite = canvas(project.width, project.height),
    ctx = composite.getContext("2d")!;
  // Visibility is a viewport aid. Export all non-missing approved/draft parts visibly in compositeOrder,
  // grouped by Cubism part hierarchy folders (`뒷머리`, `몸`, `얼굴`, `입`, `눈 R`, `눈 L`, `눈썹`, `앞옆머리`).
  type LeafLayer = {
    name: string;
    left: number;
    top: number;
    canvas: HTMLCanvasElement;
    blendMode: "normal";
    hidden: boolean;
  };
  const folders: Array<{ name: string; opened: boolean; children: LeafLayer[] }> = [];
  let leafCount = 0;
  for (const { part: p, index: i } of orderedEntries(project.parts)) {
    const eff = effectiveMask(p);
    if (
      p.status === "missing" ||
      !bounds(eff, project.width, project.height)
    )
      continue;
    const blendMode = p.blendMode ?? "normal";
    if (blendMode !== "normal") {
      throw new Error(
        `현재 PSD 내보내기는 일반(normal) 합성 모드만 지원합니다 (${p.id}: ${blendMode}). multiply/additive 모드는 실제 Cubism Editor 가져오기 검증 전까지 조용히 normal로 바꾸어 내보내지 않습니다.`,
      );
    }
    ctx.drawImage(images[i], 0, 0);
    leafCount += 1;
    const leaf: LeafLayer = {
      name: `${p.id} · ${p.name}`,
      left: 0,
      top: 0,
      canvas: images[i],
      blendMode: "normal",
      hidden: false,
    };
    const groupName = p.group || "기타";
    const tail = folders[folders.length - 1];
    if (tail && tail.name === groupName) {
      tail.children.push(leaf);
    } else {
      folders.push({ name: groupName, opened: true, children: [leaf] });
    }
  }
  if (!leafCount)
    throw new Error("내보낼 파츠가 없습니다. 먼저 분리 초안을 생성하세요.");
  const bytes = writePsdUint8Array(
    {
      width: project.width,
      height: project.height,
      canvas: composite,
      children: folders,
    },
    { trimImageData: true, noBackground: true, generateThumbnail: false },
  );
  const check = readPsd(bytes, {
    skipLayerImageData: true,
    skipCompositeImageData: true,
    skipThumbnail: true,
  });
  const countLeaves = (layers: typeof check.children): number =>
    (layers ?? []).reduce(
      (sum, layer) => sum + (layer.children ? countLeaves(layer.children) : 1),
      0,
    );
  if (
    check.width !== project.width ||
    check.children?.length !== folders.length ||
    countLeaves(check.children) !== leafCount
  )
    throw new Error("PSD 구조 검증 실패");
  return bytes;
}
export function handoff(project: Project, csv: boolean) {
  const unresolved = project.parts.filter((p) => p.status !== "approved");
  const modelHashesText =
    project.modelHashes && Object.keys(project.modelHashes).length
      ? Object.entries(project.modelHashes)
          .map(([k, v]) => `${k}=${v.slice(0, 12)}`)
          .join(", ")
      : project.modelHash || "사용하지 않음";
  return `# ${project.name} — Cubism 작업 인계\n\n- 캔버스: ${project.width} × ${project.height}px (원본 ${project.originalWidth} × ${project.originalHeight})\n- 생성 경로: ${project.provider}\n- 모델 SHA-256: ${modelHashesText}\n- 좌표: 원본 작업 캔버스 좌상단 (0,0), X 오른쪽, Y 아래. 좌우는 캐릭터 기준.\n- 미확인 파츠: ${unresolved.map((p) => p.id + " (" + p.status + ")").join(", ") || "없음"}\n\n## 가져오기\n1. character_import.psd를 Cubism Editor에서 열어 새 모델을 만듭니다. 레이어 순서와 알파를 확인합니다.\n2. ${csv ? "parameters.csv는 불러온 Cubism CSV 샘플의 형식으로 생성했습니다. 백업 후 파라미터 일괄 설정에서 UTF-8로 가져오고, 전체 삭제 옵션은 끕니다. 실제 버전 호환성은 확인이 필요합니다." : "parameters-reference.csv는 참고용 목록입니다. 공식 가져오기 파일이 아닙니다. Cubism에서 파라미터를 수동 생성하거나 flat2rig에 Editor의 CSV 샘플을 불러오세요."}\n3. guides/landmarks.png와 rig-draft.json의 파츠별 ArtMesh 토폴로지 초안을 참고하여 Cubism Editor에서 ArtMesh와 디포머를 정리합니다. 자체 JSON은 Editor로 직접 가져오지 않습니다.\n4. 목을 피벗으로 HeadRotation → HeadXY → Eyes/Mouth/Hair 디포머를 구성합니다. 눈 흰자를 클리핑 대상으로 활용하되 원화에 맞게 조정합니다.\n5. 눈 개폐 0/1, 입 개폐 0/1, 머리 X/Y -10/0/10, Z -15/0/15의 변형 키를 수동으로 작성합니다. 전체 -30~30 품질은 보장하지 않습니다.\n6. 작은 조합 입력부터 구멍, 눈꺼풀 경계, 입 안 노출을 확인하고 가림 영역을 보완합니다.\n7. 머리카락 흔들림 변형 키를 먼저 만든 뒤 Cubism에서 물리를 설정하고 .cmo3로 저장합니다.\n\n## 알려진 한계\n- 얼굴 검출(YOLOv3)·28점 랜드마크(HRNetV2)에서 crown 좌표는 기하 추정값입니다.\n- 파츠 분리는 한 레이어당 한 파츠 원칙과 움직임·가림 영역 기준의 MobileSAM + 랜드마크 하이브리드 분할 또는 기하 폴백이며, 수동 보정(브러시·클릭 프롬프트) 및 검토 상태(status)와 출처(provenance)를 분리해 기록합니다.\n- 가림 보완(template / neighbor)은 가시 마스크(\`mask\`)와 분리(\`fillMask\`) 저장되며, 실루엣 템플릿 투영 및 모션용 예비 형상(눈동자·눈꺼풀·입안)은 자동 생성되지 않는 원화 부위를 보강한 초안이므로 필요시 수동 보정하세요.\n- 레이어 합성 모드는 normal · multiply · additive 3종으로 제한되며, 현재 PSD 내보내기는 normal만 지원합니다. multiply/additive는 실제 Cubism Editor 가져오기 검증 전까지 지원 표시나 조용한 normal 변환을 하지 않습니다.\n- PSD는 이미지 레이어와 오프셋만 전달합니다. ArtMesh, 키폼, 클리핑, 물리는 자동 적용되지 않습니다.\n- 레이어 숨김/솔로는 편집기 표시 보조입니다. 내보내기는 미해당 파츠를 제외한 전체 파츠를 표시합니다.\n- 자동 PSD 재읽기는 통과했으나 실제 Cubism Editor에서의 가져오기는 별도 검증해야 합니다.\n\n## 파츠\n${orderedEntries(project.parts).map(({ part: p }, i) => `${i}. ${p.id} — ${p.name} [검토:${p.status}, 출처:${p.provenance ?? "geometry"}, 보완:${p.fillKind ?? "none"}, 합성:${p.blendMode ?? "normal"}]`).join("\n")}\n\n공식 참고: https://docs.live2d.com/en/cubism-editor-manual/psd-import/\n`;
}
export async function exportBundle(
  project: Project,
  template: CsvTemplate | null,
) {
  const images = await partCanvases(project),
    rig = await buildRig(
      project.parts,
      project.width,
      project.height,
      project.landmarks,
    );
  const ordered = orderedEntries(project.parts);
  const files: Zippable = {
    "character_import.psd": await exportPsd(project, images),
    "project.flat2rig": await saveProject(project),
    "handoff.md": strToU8(handoff(project, !!template)),
    [template ? "parameters.csv" : "parameters-reference.csv"]: strToU8(
      template ? cubismCsv(template) : referenceCsv(),
    ),
  };
  const guide = canvas(project.width, project.height),
    ctx = guide.getContext("2d")!;
  ordered.forEach(({ part, index }) => {
    if (part.status !== "missing") ctx.drawImage(images[index], 0, 0);
  });
  for (const [name, p] of Object.entries(project.landmarks)) {
    ctx.fillStyle = "#39d6b0";
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(4, project.width / 160), 0, Math.PI * 2);
    ctx.fill();
    ctx.font = `${Math.max(12, project.width / 65)}px sans-serif`;
    ctx.strokeStyle = "#132027";
    ctx.lineWidth = 3;
    ctx.strokeText(name, p.x + 10, p.y - 8);
    ctx.fillText(name, p.x + 10, p.y - 8);
  }
  files["guides/landmarks.png"] = await pngBytes(guide);
  for (let i = 0; i < project.parts.length; i++) {
    const p = project.parts[i];
    files[`parts/${p.id}.png`] = await pngBytes(images[i]);
    const visiblePng = await maskPng(p.mask, project.width, project.height);
    files[`masks/${p.id}.png`] = visiblePng;
    if (p.fillMask?.some((v) => v > 0)) {
      files[`masks/${p.id}.visible.png`] = visiblePng;
      files[`masks/${p.id}.fill.png`] = await maskPng(
        p.fillMask,
        project.width,
        project.height,
      );
    }
  }
  files["rig-draft.json"] = strToU8(
    JSON.stringify(
      {
        schemaVersion: 1,
        format: "flat2rig-rig-draft",
        directlyImportableInCubism: false,
        meshKind: "part-role-art-mesh-preview",
        supportedPsdBlendModes: ["normal"],
        coordinateSystem: "pixels-top-left-y-down",
        width: project.width,
        height: project.height,
        landmarks: project.landmarks,
        partBlendModes: Object.fromEntries(
          project.parts.map((p) => [p.id, p.blendMode ?? "normal"]),
        ),
        parameters: PARAM_DEFS,
        deformerSuggestions: [
          "Body_All (몸)",
          "Hair_Back (뒷머리)",
          "Face_All (얼굴)",
          "HeadRotation",
          "HeadXY",
          "Eyes (눈동자→눈흰자 클리핑)",
          "EyeballXY",
          "Brows",
          "Mouth",
          "HairFront / HairSide / HairBack",
        ],
        parts: rig,
      },
      null,
      2,
    ),
  );
  const provenanceSummary = project.parts.reduce(
    (acc, p) => {
      const k = p.provenance ?? "geometry";
      acc[k] = (acc[k] ?? 0) + 1;
      return acc;
    },
    {} as Record<string, number>,
  );
  const occlusionSummary = project.parts
    .filter(
      (p) =>
        p.fillMask?.some((v) => v > 0) ||
        (p.fillKind && p.fillKind !== "none"),
    )
    .map((p) => ({
      id: p.id,
      fillKind: p.fillKind ?? "neighbor",
      filledPixels: p.fillMask ? p.fillMask.filter((v) => v > 0).length : 0,
    }));
  const meshWarnings = rig.flatMap((p) => {
    const warns: string[] = [];
    if (p.mesh.triangles.length === 0) warns.push(`${p.id}: no triangles`);
    if (p.quality?.warnings?.length) warns.push(...p.quality.warnings);
    return warns;
  });
  files["validation.json"] = strToU8(
    JSON.stringify(
      {
        revision: project.revision,
        createdAt: new Date().toISOString(),
        psdStructureRoundTrip: "passed",
        cubismEditorImport: "not-tested",
        supportedPsdBlendModes: ["normal"],
        nonNormalBlendParts: project.parts
          .filter((p) => (p.blendMode ?? "normal") !== "normal")
          .map((p) => ({ id: p.id, blendMode: p.blendMode })),
        csv: template ? "sample-derived-not-editor-tested" : "reference-only",
        unapproved: project.parts
          .filter((p) => p.status === "draft")
          .map((p) => p.id),
        missing: project.parts
          .filter((p) => p.status === "missing")
          .map((p) => p.id),
        sourceProvider: project.provider,
        engine: project.provider.includes("browser")
          ? "typescript-browser-fallback"
          : "rust-ort-delaunator-glam",
        sourceModelHash: project.modelHash,
        modelHashes: project.modelHashes ?? {},
        postprocessVersions: project.postprocessVersions ?? {},
        provenanceSummary,
        occlusionSummary,
        meshWarnings,
      },
      null,
      2,
    ),
  );
  return pack(files);
}
export function newPart(w: number, h: number): Part {
  const [, , group, color] = PART_DEFS[3];
  return {
    id: `custom_${crypto.randomUUID().slice(0, 8)}`,
    name: "새 파츠",
    group,
    color,
    role: "other",
    visible: true,
    status: "draft",
    mask: new Uint8Array(w * h),
  };
}

export function newHairPart(
  role: HairRole,
  existingParts: Part[],
  w: number,
  h: number,
): Part {
  const existingIds = new Set(existingParts.map((p) => p.id));
  let idx = role === "hair_front" || role === "hair_back" ? 2 : 1;
  let id = idx === 1 ? role : `${role}_${idx}`;
  while (existingIds.has(id)) {
    idx++;
    id = `${role}_${idx}`;
  }
  const meta = HAIR_ROLE_META[role];
  const label = idx === 1 ? meta.name : `${meta.name} ${idx}`;
  return {
    id,
    name: label,
    group: meta.group,
    color: meta.color,
    role,
    visible: true,
    status: "draft",
    provenance: "manual",
    mask: new Uint8Array(w * h),
  };
}

export function newDynamicPart(
  baseId: HairRole | "hat" | "clothes_outer" | "acc_glasses" | "acc_hairpin",
  existingParts: Part[],
  w: number,
  h: number,
): Part {
  if (baseId in HAIR_ROLE_META) {
    return newHairPart(baseId as HairRole, existingParts, w, h);
  }
  const existingIds = new Set(existingParts.map((p) => p.id));
  let id: string = baseId;
  let idx = 2;
  while (existingIds.has(id)) {
    id = `${baseId}_${idx}`;
    idx++;
  }
  const meta = describeDynamicPart(id);
  return {
    id,
    name: meta.name,
    group: meta.group,
    color: meta.color,
    role: meta.role,
    visible: true,
    status: "draft",
    provenance: "manual",
    mask: new Uint8Array(w * h),
  };
}

const CANONICAL_LAYER_ORDER: readonly string[] = [
  "hair_back",
  "body",
  "neck",
  "clothes",
  "face",
  "clothes_hands",
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
  "acc_glasses",
  "hair_side_R",
  "hair_side_L",
  "hair_front",
  "hat",
  "acc_hairpin",
  "acc_ribbon",
];

const CANONICAL_INDEX = new Map(CANONICAL_LAYER_ORDER.map((id, i) => [id, i]));

function partLayerRank(part: Pick<Part, "id" | "role">): number {
  const exact = CANONICAL_INDEX.get(part.id);
  if (exact !== undefined) return exact;
  const role: PartRole | undefined = part.role;
  const prefixTable: readonly [string, PartRole | undefined, string, number][] = [
    ["hair_back_", "hair_back", "hair_back", 0.5],
    ["clothes_hands", "hands", "clothes_hands", 0],
    ["clothes_", "clothes", "clothes", 0.5],
    ["hair_side_R", "hair_side_R", "hair_side_R", 0.5],
    ["hair_side_L", "hair_side_L", "hair_side_L", 0.5],
    ["hair_front_", "hair_front", "hair_front", 0.5],
    ["hat_", "hat", "hat", 0.5],
    ["acc_glasses", undefined, "acc_glasses", 0],
    ["acc_hairpin", undefined, "acc_hairpin", 0],
    ["acc_ribbon", undefined, "acc_ribbon", 0],
  ];
  for (const [prefix, matchRole, anchor, offset] of prefixTable) {
    if (part.id.startsWith(prefix) || (matchRole && role === matchRole)) {
      return (CANONICAL_INDEX.get(anchor) ?? CANONICAL_LAYER_ORDER.length) + offset;
    }
  }
  if (role === "face") return CANONICAL_INDEX.get("face")!;
  if (role === "eye") return CANONICAL_INDEX.get("eyeL_upper")! + 0.25;
  if (role === "mouth") return CANONICAL_INDEX.get("mouth_upper")! + 0.25;
  return CANONICAL_LAYER_ORDER.length;
}

export function insertHairPartInOrder(parts: Part[], added: Part): Part[] {
  const list = [...parts];
  const rank = partLayerRank(added);
  const idx = list.findIndex((p) => partLayerRank(p) > rank);
  const insertAt = idx >= 0 ? idx : list.length;
  list.splice(insertAt, 0, added);
  return list.map((p, i) => ({ ...p, compositeOrder: i }));
}

export function isPreservedPart(part: Part): boolean {
  return (
    part.status === "approved" ||
    part.provenance === "manual" ||
    part.provenance === "prompted"
  );
}

export function mergePreservedParts(
  prevParts: Part[],
  nextParts: Part[],
): { parts: Part[]; preservedCount: number } {
  const prevById = new Map(prevParts.map((t) => [t.id, t]));
  const preservedIds = new Set<string>();
  let preservedCount = 0;

  let merged: Part[] = nextParts.map((nextPart) => {
    const prev = prevById.get(nextPart.id);
    if (prev && isPreservedPart(prev)) {
      preservedIds.add(prev.id);
      preservedCount++;
      return { ...prev, compositeOrder: nextPart.compositeOrder };
    }
    return nextPart;
  });

  // Keep dynamic extra parts (extra hair strands, hats, wearables) that user approved or edited manually.
  for (const prev of prevParts) {
    if (isPreservedPart(prev) && !preservedIds.has(prev.id)) {
      merged = insertHairPartInOrder(merged, prev);
      preservedIds.add(prev.id);
      preservedCount++;
    }
  }

  if (preservedIds.size > 0 && merged.length > 0) {
    const len = merged[0].mask.length;
    const ownerIdx = new Int32Array(len).fill(-1);
    const preservedList = merged.filter((part) => preservedIds.has(part.id));
    const pairOverlaps = new Map<string, number>();

    for (let pIdx = 0; pIdx < preservedList.length; pIdx++) {
      const part = preservedList[pIdx];
      for (let i = 0; i < len; i++) {
        if (part.mask[i] === 0) continue;
        const prevOwnerIdx = ownerIdx[i];
        if (prevOwnerIdx >= 0 && prevOwnerIdx !== pIdx) {
          const ownerId = preservedList[prevOwnerIdx].id;
          const key = `${ownerId}::${part.id}`;
          pairOverlaps.set(key, (pairOverlaps.get(key) ?? 0) + 1);
        } else {
          ownerIdx[i] = pIdx;
        }
      }
    }

    if (pairOverlaps.size > 0) {
      const summary = [...pairOverlaps.entries()]
        .map(([pair, px]) => `${pair.replace("::", "·")}(${px}px)`)
        .join(", ");
      throw new Error(
        `보존된 파츠끼리 이미 겹쳐 있어 병합을 진행할 수 없습니다: ${summary}. 브러시/지우개로 겹친 영역을 먼저 정리해 주세요.`,
      );
    }

    merged = merged.map((part, idx) => {
      if (preservedIds.has(part.id)) {
        return { ...part, compositeOrder: idx };
      }
      let touched = false;
      for (let i = 0; i < len; i++) {
        if (ownerIdx[i] >= 0 && part.mask[i] > 0) {
          touched = true;
          break;
        }
      }
      if (!touched) {
        return { ...part, compositeOrder: idx };
      }
      const nextMask = part.mask.slice();
      for (let i = 0; i < len; i++) {
        if (ownerIdx[i] >= 0) nextMask[i] = 0;
      }
      return { ...part, mask: nextMask, compositeOrder: idx };
    });
  } else {
    merged = merged.map((part, idx) => ({ ...part, compositeOrder: idx }));
  }

  return { parts: merged, preservedCount };
}

export { isHairStageLayer, partPipelineStage, stageIncludesLayer };

const STAGE_ORDER_RANK: Record<PipelineStage | "base_body", number> = {
  base_silhouette: 1,
  features_overlay: 2,
  clothes_overlay: 3,
  base_body: 3,
  hair_overlay: 4,
  full: 4,
};

const CANONICAL_STAGE_Z: Record<string, number> = {
  hair_back: 0,
  body: 10,
  neck: 20,
  clothes: 30,
  clothes_inner: 32,
  clothes_collar: 34,
  clothes_outer: 36,
  clothes_acc: 38,
  clothes_hands: 39,
  hands: 39,
  face: 40,
  face_detail: 42,
  eyeR_white: 50,
  eyeR_iris: 51,
  eyeR_lower: 52,
  eyeR_upper: 53,
  eyeR_all: 54,
  eyeL_white: 60,
  eyeL_iris: 61,
  eyeL_lower: 62,
  eyeL_upper: 63,
  eyeL_all: 64,
  browR: 70,
  browL: 71,
  mouth_inner: 80,
  mouth_lower: 81,
  mouth_upper: 82,
  mouth_all: 83,
  acc_glasses: 90,
  hair_side_R: 100,
  hair_side_L: 101,
  hair_front: 110,
  hat: 120,
  acc_hairpin: 130,
  acc_ribbon: 140,
};

function stageLayerZRank(id: string): number {
  if (id in CANONICAL_STAGE_Z) return CANONICAL_STAGE_Z[id];
  if (id.startsWith("hair_back")) return 1;
  if (id.startsWith("clothes_hands") || id.startsWith("hands_")) return 39;
  if (id.startsWith("clothes_")) return 35;
  if (id.startsWith("hair_side_R")) return 100;
  if (id.startsWith("hair_side_L") || id.startsWith("hair_side")) return 101;
  if (id.startsWith("hair_front") || id.startsWith("hair_")) return 111;
  if (id.startsWith("hat")) return 121;
  if (id.startsWith("acc_")) return 135;
  return 150;
}

function collectClaimedMask(parts: Part[], len: number): Uint8Array {
  const claimed = new Uint8Array(len);
  for (const part of parts) {
    const n = Math.min(len, part.mask.length);
    for (let i = 0; i < n; i++) {
      if (part.mask[i] > claimed[i]) claimed[i] = part.mask[i];
    }
  }
  return claimed;
}

function demoteOverlappingMask(part: Part, claimed: Uint8Array): Part {
  const n = Math.min(claimed.length, part.mask.length);
  let hasOverlap = false;
  for (let i = 0; i < n; i++) {
    if (claimed[i] > 0 && part.mask[i] > 0) {
      hasOverlap = true;
      break;
    }
  }
  if (!hasOverlap) return part;
  const nextMask = part.mask.slice();
  const nextFill = part.fillMask
    ? part.fillMask.slice()
    : new Uint8Array(part.mask.length);
  for (let i = 0; i < n; i++) {
    if (claimed[i] > 0 && nextMask[i] > 0) {
      if (nextMask[i] > nextFill[i]) nextFill[i] = nextMask[i];
      nextMask[i] = 0;
    }
  }
  return {
    ...part,
    mask: nextMask,
    fillMask: nextFill,
    fillKind:
      part.fillKind && part.fillKind !== "none"
        ? part.fillKind
        : ("template" as const),
  };
}

/**
 * Merges a generated pipeline stage cleanly onto existing layers across any of the 4 hybrid stacking stages:
 * - Replaces only the incoming stage's parts while keeping earlier/later stages and approved user edits.
 * - Shifts overlapped pixels on lower-stage bases from `mask` into `fillMask` so visible masks stay strictly disjoint.
 * - Rebuilds canonical Z-order (`hair_back* < body < neck < clothes* < face < face_detail < eyes/brows/mouth < glasses < hair_side/front/hat`).
 */
export function mergeStageParts(
  prevParts: Part[],
  incomingParts: Part[],
  targetStage: PipelineStage | "base_body",
  preserveApproved = true,
): { parts: Part[]; preservedCount: number } {
  if (targetStage === "full") {
    return preserveApproved
      ? mergePreservedParts(prevParts, incomingParts)
      : { parts: incomingParts, preservedCount: 0 };
  }

  const keptOther = prevParts.filter(
    (p) => !stageIncludesLayer(p.id, targetStage),
  );
  const incomingTarget = incomingParts.filter((p) =>
    stageIncludesLayer(p.id, targetStage),
  );
  const prevTargetById = new Map(
    prevParts
      .filter((p) => stageIncludesLayer(p.id, targetStage))
      .map((p) => [p.id, p]),
  );

  let preservedCount = 0;
  const mergedTarget: Part[] = incomingTarget.map((next) => {
    const prev = prevTargetById.get(next.id);
    if (preserveApproved && prev && isPreservedPart(prev)) {
      preservedCount += 1;
      return prev;
    }
    return next;
  });
  if (preserveApproved) {
    const mergedIds = new Set(mergedTarget.map((p) => p.id));
    for (const prev of prevTargetById.values()) {
      if (!mergedIds.has(prev.id) && isPreservedPart(prev)) {
        preservedCount += 1;
        mergedTarget.push(prev);
      }
    }
  }

  const incomingRank = STAGE_ORDER_RANK[targetStage];
  const earlierParts = keptOther.filter(
    (p) => STAGE_ORDER_RANK[partPipelineStage(p.id)] < incomingRank,
  );
  const laterParts = keptOther.filter(
    (p) => STAGE_ORDER_RANK[partPipelineStage(p.id)] > incomingRank,
  );

  const len =
    mergedTarget[0]?.mask.length ??
    keptOther[0]?.mask.length ??
    0;
  const laterClaimed = collectClaimedMask(laterParts, len);
  const adjustedIncoming = mergedTarget.map((p) =>
    demoteOverlappingMask(p, laterClaimed),
  );
  const combinedUpperClaimed = collectClaimedMask(
    [...adjustedIncoming, ...laterParts],
    len,
  );
  const adjustedEarlier = earlierParts.map((p) =>
    demoteOverlappingMask(p, combinedUpperClaimed),
  );

  const combined = [...adjustedEarlier, ...adjustedIncoming, ...laterParts]
    .map((part, idx) => ({ part, idx, rank: stageLayerZRank(part.id) }))
    .sort((a, b) => a.rank - b.rank || a.idx - b.idx)
    .map((entry, idx) => ({ ...entry.part, compositeOrder: idx }));

  return { parts: combined, preservedCount };
}

/**
 * Legacy helper preserving Stage-2 hair composition onto Stage-1 base layers.
 */
export function mergeStage2HairParts(
  baseParts: Part[],
  hairParts: Part[],
  preserveApproved = true,
): { parts: Part[]; preservedCount: number } {
  return mergeStageParts(baseParts, hairParts, "hair_overlay", preserveApproved);
}
