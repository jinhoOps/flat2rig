/**
 * B5: See-through (shitagaki-lab/see-through, SIGGRAPH 2026) PSD detection and V2 conversion rules.
 *
 * Ported from `tools/evaluation/v2_seethrough_to_layers.py` (see `artifacts/v2/WORKER_REPORT.md`):
 * - Detects See-through output PSDs by their `bodytags_v3` layer names (`isSeeThroughPsd`).
 * - Applies the alpha floor (`alpha <= 4/255` zeroed) to remove full-canvas low-alpha noise
 *   (e.g. Sutteang's `mouth` layer) before centroid/split/mesh computation.
 * - Estimates the character midline from the iris pair (else eye-white pair, else face centroid).
 * - Splits `eyelash` into character L/R and upper/lower (`eye{S}_upper` / `eye{S}_lower`) using
 *   each eye's white (else iris) vertical centroid.
 * - Proposes explicit, reversible depth-order fixes (`correctSeeThroughDrawOrder`):
 *   1. Lift face features (`eye*`, `brow*`, `mouth*`, `face_detail`, `acc_glasses`) that landed
 *      below `face` to just above `face`.
 *   2. Move `eye{S}_white` that landed above `eye{S}_iris` to just below `eye{S}_iris`.
 *
 * License note: See-through weights are OpenRAIL++ (not bundled in flat2rig; external tool only).
 */
import type { ParsedPsd, PsdLayerPixels } from "./psdLayers";

/** See-through `bodytags_v3` tags → flat2rig part ids (`{S}` = split into character L/R). */
export const SEE_THROUGH_TAGS: Record<string, string> = {
  "back hair": "hair_back",
  "front hair": "hair_front",
  face: "face",
  ears: "face",
  neck: "neck",
  topwear: "clothes",
  neckwear: "clothes_inner",
  eyewhite: "eye{S}_white",
  irides: "eye{S}_iris",
  eyelash: "eye{S}_upper",
  eyebrow: "brow{S}",
  mouth: "mouth_all",
  nose: "face_detail",
  headwear: "hat",
  eyewear: "acc_glasses",
  earwear: "acc_earwear",
  handwear: "hands",
  bottomwear: "clothes_bottom",
  legwear: "clothes_legwear",
  footwear: "clothes_footwear",
  tail: "other_tail",
  wings: "other_wings",
  objects: "other_objects",
};

/** Alpha <= 4/255 is zeroed on See-through layers (`ALPHA_FLOOR = 4.5 / 255.0` in V2). */
export const SEE_THROUGH_ALPHA_FLOOR = 4;

/** Component threshold matching `alpha > 0.1` in `v2_seethrough_to_layers.py`. */
export const SEE_THROUGH_COMPONENT_ALPHA = 25;

/** Face features + glasses that must sit above `face` in flat2rig's stack. */
export const SEE_THROUGH_FACE_FEATURES = new Set<string>([
  "browL",
  "browR",
  "eyeL_white",
  "eyeR_white",
  "eyeL_iris",
  "eyeR_iris",
  "eyeL_upper",
  "eyeR_upper",
  "eyeL_lower",
  "eyeR_lower",
  "mouth_all",
  "mouth_upper",
  "mouth_lower",
  "mouth_inner",
  "face_detail",
  "acc_glasses",
]);

export interface SeeThroughLimitation {
  id: string;
  title: string;
  detail: string;
}

/** Known V2 limitations shown in the mapping panel banner when a See-through PSD is opened. */
export const SEE_THROUGH_LIMITATIONS: readonly SeeThroughLimitation[] = [
  {
    id: "mouth-nose",
    title: "입·코 숨은 영역 없음",
    detail: "입(mouth)과 코(nose)는 원화 복사본 한 장이라 입안·윗/아랫입술이 분리되지 않아 입 벌리기 리깅이 제한됩니다.",
  },
  {
    id: "chibi-sd",
    title: "치비(SD)·큰 소품 취약",
    detail: "SD 비율이나 큰 모자·소품은 모델 생성 단계에서 누락되거나 얼굴 색이 틀어질 수 있습니다.",
  },
  {
    id: "seed-variance",
    title: "눈썹·경계 시드 편차",
    detail: "눈썹·아래 눈꺼풀·앞/뒷머리 경계는 실행 시드에 따라 편차가 크므로 2~3개 시드 중 선택하는 것이 좋습니다.",
  },
];

export const SEE_THROUGH_LICENSE_NOTE =
  "외부 도구(OpenRAIL++ 가중치, 앱 미번들) 출력입니다. 산출물 재배포 시 OpenRAIL++ 사용 제한을 확인하세요.";

export function normSeeThroughTag(name: string): string {
  return name.trim().toLowerCase().replace(/[_-]+/g, " ").replace(/\s+/g, " ");
}

/** True when all top-level layer/group names match See-through `bodytags_v3` tags. */
export function isSeeThroughPsd(topLevelNames: readonly string[]): boolean {
  if (topLevelNames.length === 0) return false;
  return topLevelNames.every((name) => normSeeThroughTag(name) in SEE_THROUGH_TAGS);
}

/** Zeroes alpha <= `floor` (default 4/255) on a straight-RGBA layer crop, returning a new crop if modified. */
export function applySeeThroughAlphaFloor(
  px: PsdLayerPixels,
  floor = SEE_THROUGH_ALPHA_FLOOR,
): PsdLayerPixels {
  let dirty = false;
  for (let i = 3; i < px.rgba.length; i += 4) {
    const a = px.rgba[i];
    if (a > 0 && a <= floor) {
      dirty = true;
      break;
    }
  }
  if (!dirty) return px;
  const rgba = new Uint8ClampedArray(px.rgba);
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i + 3] <= floor) {
      rgba[i] = 0;
      rgba[i + 1] = 0;
      rgba[i + 2] = 0;
      rgba[i + 3] = 0;
    }
  }
  return { ...px, rgba };
}

export interface AlphaComponent {
  /** Per-pixel mask inside `px` (1 = belongs to this component). */
  mask: Uint8Array;
  cx: number;
  cy: number;
  minX: number;
  maxX: number;
  count: number;
}

/** 8-neighbour connected components of `px` where alpha > `alphaThreshold` (default 25 ≈ 0.1). */
export function extractAlphaComponents(
  px: PsdLayerPixels,
  alphaThreshold = SEE_THROUGH_COMPONENT_ALPHA,
): AlphaComponent[] {
  const { width: w, height: h, rgba, left, top } = px;
  const visited = new Uint8Array(w * h);
  const out: AlphaComponent[] = [];
  const stack: number[] = [];

  for (let s = 0; s < w * h; s++) {
    if (visited[s] || rgba[s * 4 + 3] <= alphaThreshold) continue;
    visited[s] = 1;
    stack.push(s);
    const mask = new Uint8Array(w * h);
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    let minX = w;
    let maxX = -1;

    while (stack.length > 0) {
      const i = stack.pop()!;
      const x = i % w;
      const y = (i - x) / w;
      mask[i] = 1;
      sumX += left + x;
      sumY += top + y;
      count++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;

      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (!visited[j] && rgba[j * 4 + 3] > alphaThreshold) {
            visited[j] = 1;
            stack.push(j);
          }
        }
      }
    }

    out.push({
      mask,
      cx: sumX / count,
      cy: sumY / count,
      minX: left + minX,
      maxX: left + maxX,
      count,
    });
  }

  return out;
}

/**
 * Character midline for See-through PSDs (`v2_seethrough_to_layers.py`):
 * midpoint of the two largest iris (else eye-white) components when they look like a L/R pair
 * (`b.count >= 0.25 * a.count` and `|a.cx - b.cx| >= width(a)`), else face alpha centroid,
 * else canvas centre.
 */
export function estimateSeeThroughMidline(
  psd: ParsedPsd,
  classes: ReadonlyMap<string, string | null>,
): { x: number; source: "iris" | "eyewhite" | "face" | "canvas" } {
  const usable = (n: (typeof psd.layers)[number]) => !n.effectiveHidden && !n.guide && n.alphaSum > 0;

  for (const [targetClass, src] of [
    ["eye{S}_iris", "iris"],
    ["eye{S}_white", "eyewhite"],
  ] as const) {
    const layer = psd.layers.find((n) => usable(n) && classes.get(n.path) === targetClass);
    if (!layer) continue;
    const px = psd.pixels.get(layer.path);
    if (!px) continue;
    const comps = extractAlphaComponents(px).sort((a, b) => b.count - a.count);
    if (comps.length >= 2) {
      const [a, b] = comps;
      const wa = a.maxX - a.minX + 1;
      if (b.count >= 0.25 * a.count && Math.abs(a.cx - b.cx) >= wa) {
        return { x: (a.cx + b.cx) / 2, source: src };
      }
    }
  }

  const face = psd.layers.find(
    (n) => usable(n) && classes.get(n.path) === "face" && normSeeThroughTag(n.name) !== "ears",
  );
  if (face) return { x: face.centroidX, source: "face" };

  return { x: psd.width / 2, source: "canvas" };
}

export interface EyeCenterY {
  L?: number;
  R?: number;
}

/**
 * Vertical eye centre per character side (`L` / `R`) for upper/lower eyelash splitting.
 * Uses `eye{S}_white` centroid first, falling back to `eye{S}_iris` (See-through often omits
 * `eyewhite` on front-facing illustrations like Himikan).
 */
export function computeEyeCenterY(
  psd: ParsedPsd,
  classes: ReadonlyMap<string, string | null>,
  midlineX: number,
): EyeCenterY {
  const out: EyeCenterY = {};
  const usable = (n: (typeof psd.layers)[number]) => !n.effectiveHidden && !n.guide && n.alphaSum > 0;

  for (const targetClass of ["eye{S}_white", "eye{S}_iris", "eyeL_white", "eyeR_white", "eyeL_iris", "eyeR_iris"]) {
    for (const layer of psd.layers) {
      if (!usable(layer) || classes.get(layer.path) !== targetClass) continue;
      const px = psd.pixels.get(layer.path);
      if (!px) continue;
      const comps = extractAlphaComponents(px);
      const compMaskSide = new Uint8Array(px.width * px.height);
      const hasComp = new Uint8Array(px.width * px.height);
      for (const c of comps) {
        const s = c.cx > midlineX ? 1 : 0;
        for (let i = 0; i < c.mask.length; i++) {
          if (c.mask[i]) {
            hasComp[i] = 1;
            compMaskSide[i] = s;
          }
        }
      }
      for (const side of ["L", "R"] as const) {
        if (out[side] !== undefined) continue;
        if (targetClass.includes("L_") && side !== "L") continue;
        if (targetClass.includes("R_") && side !== "R") continue;
        const want = side === "L" ? 1 : 0;
        let sumA = 0;
        let sumY = 0;
        for (let y = 0; y < px.height; y++) {
          for (let x = 0; x < px.width; x++) {
            const idx = y * px.width + x;
            const a = px.rgba[idx * 4 + 3];
            if (a === 0) continue;
            const pixelSide = hasComp[idx] ? compMaskSide[idx] : px.left + x + 0.5 > midlineX ? 1 : 0;
            if (pixelSide !== want) continue;
            sumA += a;
            sumY += a * (px.top + y + 0.5);
          }
        }
        if (sumA > 0) out[side] = sumY / sumA;
      }
    }
  }
  return out;
}

export interface EyelashSplitResult {
  eyeL_upper: Uint8Array;
  eyeL_lower: Uint8Array;
  eyeR_upper: Uint8Array;
  eyeR_lower: Uint8Array;
  counts: {
    upperL: number;
    lowerL: number;
    upperR: number;
    lowerR: number;
  };
}

/**
 * Splits a combined See-through `eyelash` layer into character L/R and upper/lower
 * (`v2_seethrough_to_layers.py` lines 279–295).
 */
export function splitEyelashUpperLower(
  px: PsdLayerPixels,
  midlineX: number,
  eyeCenterY: EyeCenterY,
): EyelashSplitResult {
  const n = px.width * px.height;
  const eyeL_upper = new Uint8Array(n);
  const eyeL_lower = new Uint8Array(n);
  const eyeR_upper = new Uint8Array(n);
  const eyeR_lower = new Uint8Array(n);
  const counts = { upperL: 0, lowerL: 0, upperR: 0, lowerR: 0 };

  const comps = extractAlphaComponents(px);
  const covered = new Uint8Array(n);

  for (const c of comps) {
    const side = c.cx > midlineX ? "L" : "R";
    const cy0 = eyeCenterY[side];
    const isUpper = cy0 === undefined || c.cy <= cy0;
    const dst =
      side === "L" ? (isUpper ? eyeL_upper : eyeL_lower) : isUpper ? eyeR_upper : eyeR_lower;
    if (side === "L") {
      if (isUpper) counts.upperL++;
      else counts.lowerL++;
    } else {
      if (isUpper) counts.upperR++;
      else counts.lowerR++;
    }
    for (let i = 0; i < n; i++) {
      if (c.mask[i]) {
        dst[i] = 1;
        covered[i] = 1;
      }
    }
  }

  // Faint pixels below the component threshold follow x > midlineX and y <= cy0.
  for (let y = 0; y < px.height; y++) {
    for (let x = 0; x < px.width; x++) {
      const i = y * px.width + x;
      if (covered[i] || px.rgba[i * 4 + 3] === 0) continue;
      const side = px.left + x + 0.5 > midlineX ? "L" : "R";
      const cy0 = eyeCenterY[side];
      const isUpper = cy0 === undefined || px.top + y + 0.5 <= cy0;
      if (side === "L") (isUpper ? eyeL_upper : eyeL_lower)[i] = 1;
      else (isUpper ? eyeR_upper : eyeR_lower)[i] = 1;
    }
  }

  return { eyeL_upper, eyeL_lower, eyeR_upper, eyeR_lower, counts };
}

export interface SeeThroughOrderFix {
  kind: "lifted-above-face" | "eye-white-below-iris";
  partId: string;
  message: string;
}

export interface SeeThroughOrderPlan {
  /** Original bottom → top part order from the PSD. */
  rawOrder: string[];
  /** Corrected bottom → top part order. */
  correctedOrder: string[];
  /** Face features / glasses lifted from below `face` to just above `face`. */
  liftedAboveFace: string[];
  /** Eye whites moved from above their iris to just below their iris. */
  eyeWhiteMovedBelowIris: string[];
  /** True when `correctedOrder` differs from `rawOrder`. */
  changed: boolean;
  /** Human-readable list of applied fixes for UI badges and notes. */
  fixes: SeeThroughOrderFix[];
}

/**
 * Corrects See-through's pseudo-depth draw order defects (`v2_seethrough_to_layers.py` lines 322–345):
 * 1. Face features (`eye*`, `brow*`, `mouth*`, `face_detail`, `acc_glasses`) placed below `face`
 *    are lifted to just above `face` (preserving their relative order).
 * 2. `eye{S}_white` placed above `eye{S}_iris` is moved to just below `eye{S}_iris`.
 */
export function correctSeeThroughDrawOrder(rawOrder: readonly string[]): SeeThroughOrderPlan {
  const order = [...rawOrder];
  const liftedAboveFace: string[] = [];
  const eyeWhiteMovedBelowIris: string[] = [];
  const fixes: SeeThroughOrderFix[] = [];

  const faceIdx = order.indexOf("face");
  if (faceIdx !== -1) {
    const below = order.slice(0, faceIdx).filter((id) => SEE_THROUGH_FACE_FEATURES.has(id));
    if (below.length > 0) {
      const belowSet = new Set(below);
      const filtered = order.filter((id) => !belowSet.has(id));
      const newFaceIdx = filtered.indexOf("face");
      filtered.splice(newFaceIdx + 1, 0, ...below);
      order.splice(0, order.length, ...filtered);
      liftedAboveFace.push(...below);
      for (const id of below) {
        fixes.push({
          kind: "lifted-above-face",
          partId: id,
          message: `${id}: 얼굴(face) 아래에서 얼굴 바로 위로 올림`,
        });
      }
    }
  }

  for (const side of ["L", "R"] as const) {
    const whiteId = `eye${side}_white`;
    const irisId = `eye${side}_iris`;
    const wi = order.indexOf(whiteId);
    const ii = order.indexOf(irisId);
    if (wi !== -1 && ii !== -1 && wi > ii) {
      order.splice(wi, 1);
      const newIi = order.indexOf(irisId);
      order.splice(newIi, 0, whiteId);
      eyeWhiteMovedBelowIris.push(whiteId);
      fixes.push({
        kind: "eye-white-below-iris",
        partId: whiteId,
        message: `${whiteId}: 홍채(${irisId}) 위에서 홍채 바로 아래로 내림`,
      });
    }
  }

  const changed =
    order.length !== rawOrder.length || order.some((id, idx) => id !== rawOrder[idx]);

  return {
    rawOrder: [...rawOrder],
    correctedOrder: order,
    liftedAboveFace,
    eyeWhiteMovedBelowIris,
    changed,
    fixes,
  };
}

// ---------------------------------------------------------------------------------------------
// Checkpoint 2: External See-through tool settings & status types (not bundled in app)
// ---------------------------------------------------------------------------------------------

export const SEE_THROUGH_INSTALL_DIR_KEY = "flat2rig.seethrough.installDir";
export const SEE_THROUGH_SEED_KEY = "flat2rig.seethrough.seed";
export const SEE_THROUGH_DEFAULT_SEED = 42;

export const SEE_THROUGH_INSTALL_GUIDE = [
  "See-through(shitagaki-lab/see-through) 저장소를 로컬에 설치하고 Python 가상환경(venv)을 준비하세요.",
  "설치 폴더에 venv(Scripts/python.exe)와 repo/inference_psd_blockswap.py(또는 inference_psd.py)가 있어야 합니다.",
  "모델 가중치는 OpenRAIL++ 라이선스이므로 앱에 포함되지 않으며, 최초 실행 시 HF 캐시(hf-home)로 내려받습니다.",
] as const;

export interface SeeThroughInstallStatus {
  installed: boolean;
  installDir: string;
  pythonPath: string | null;
  scriptPath: string | null;
  hfHome: string | null;
  message: string;
  license: string;
  limitations: string[];
}

export interface SeeThroughDecomposeResult {
  psdPath: string;
  outputDir: string;
  seed: number;
  commandSummary: string;
}

export interface SeeThroughSettings {
  installDir: string;
  seed: number;
}

function defaultStorage(): Storage | undefined {
  return typeof window !== "undefined" ? window.localStorage : undefined;
}

export function loadSeeThroughSettings(storage: Storage | undefined = defaultStorage()): SeeThroughSettings {
  if (!storage) return { installDir: "", seed: SEE_THROUGH_DEFAULT_SEED };
  try {
    const installDir = (storage.getItem(SEE_THROUGH_INSTALL_DIR_KEY) ?? "").trim();
    const raw = storage.getItem(SEE_THROUGH_SEED_KEY);
    const rawSeed = raw !== null && raw.trim() !== "" ? Number(raw) : NaN;
    const seed =
      Number.isInteger(rawSeed) && rawSeed >= 0 && rawSeed <= 0xffffffff
        ? rawSeed
        : SEE_THROUGH_DEFAULT_SEED;
    return { installDir, seed };
  } catch {
    return { installDir: "", seed: SEE_THROUGH_DEFAULT_SEED };
  }
}

export function saveSeeThroughSettings(
  settings: SeeThroughSettings,
  storage: Storage | undefined = defaultStorage(),
): void {
  if (!storage) return;
  try {
    const dir = settings.installDir.trim();
    if (dir) storage.setItem(SEE_THROUGH_INSTALL_DIR_KEY, dir);
    else storage.removeItem(SEE_THROUGH_INSTALL_DIR_KEY);
    const seed =
      Number.isInteger(settings.seed) && settings.seed >= 0 && settings.seed <= 0xffffffff
        ? settings.seed
        : SEE_THROUGH_DEFAULT_SEED;
    storage.setItem(SEE_THROUGH_SEED_KEY, String(seed));
  } catch {
    // Ignore storage quota/security errors.
  }
}

export function clearSeeThroughSettings(storage: Storage | undefined = defaultStorage()): void {
  if (!storage) return;
  try {
    storage.removeItem(SEE_THROUGH_INSTALL_DIR_KEY);
    storage.removeItem(SEE_THROUGH_SEED_KEY);
  } catch {
    // Ignore storage errors.
  }
}
