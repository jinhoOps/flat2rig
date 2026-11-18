/**
 * B1: layered PSD as the app's primary input — parsing and layer → part mapping suggestions.
 *
 * Reads a PSD with `ag-psd` (works in the browser fallback and in the desktop webview), flattens the
 * group tree into a list of pixel layers in compositing order (bottom → top, the order ag-psd
 * returns `children` in) and proposes a flat2rig part for each layer from its name, its group names
 * and, for See-through output, its tag. Nothing here touches pixels beyond per-layer statistics;
 * building parts from a mapping is `layerParts.ts`.
 *
 * Rules that matter for the rig (AGENTS.md):
 * - L/R is character-relative (1절): character L = screen right (larger x). A layer's side comes from
 *   where its pixels are relative to the face midline, never from a `<`/`>`/左/右 token, because PSD
 *   authors use both screen- and character-relative names.
 * - Non-`normal` blend layers are never silently turned into `normal` (6절): they carry a warning and
 *   an explicit `blend` choice (`fold` = apply the blend inside its part, `normal`, or excluded).
 * - Hidden layers and guide layers are excluded by default but stay in the list.
 * - Anything the rules are not sure about stays unassigned (`target: null`) for the user to pick.
 */
import { readPsd, type Layer, type Psd } from "ag-psd";
import { PART_DEFS } from "./types";

export const PSD_LAYER_INPUT_VERSION = "psd-layers-v1";
/** Largest PSD the app reads (Rust `PASTED_PSD_MAX_BYTES` for pasted paths). */
export const MAX_PSD_BYTES = 300 * 1024 * 1024;

/** Blend modes the part builder can apply inside a part (W3C separable formulas). */
export const FOLDABLE_BLEND_MODES = ["multiply", "linear dodge", "screen"] as const;
export type FoldableBlendMode = (typeof FOLDABLE_BLEND_MODES)[number];

export interface PsdBounds {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** One straight-alpha RGBA crop of a layer at PSD resolution, mask and opacity NOT applied. */
export interface PsdLayerPixels {
  left: number;
  top: number;
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
  /** Layer (user) mask, 0..255 grey, cropped to `maskBounds`; outside it `maskDefault` applies. */
  mask?: Uint8ClampedArray;
  maskBounds?: PsdBounds;
  maskDefault?: number;
}

export interface PsdLayerNode {
  /** Unique slash path of group names and the layer name, e.g. `/모델/눈</눈흰자<` (a `#2` suffix
   * disambiguates same-named siblings). */
  path: string;
  name: string;
  parentPath: string | null;
  depth: number;
  isGroup: boolean;
  /** Index among pixel layers, bottom → top (−1 for groups). */
  order: number;
  hidden: boolean;
  /** Hidden itself or through a hidden ancestor group. */
  effectiveHidden: boolean;
  /** Own opacity × fill opacity, 0..1. */
  opacity: number;
  /** Own opacity times every ancestor group's opacity. */
  effectiveOpacity: number;
  blendMode: string;
  /** The first non-pass-through, non-normal ancestor group blend mode (applied to the layer's
   * handling because the part builder composites layers individually). */
  groupBlendMode?: string;
  clipping: boolean;
  /** Clipping base: nearest non-clipped pixel layer below in the same group. */
  clipBase: string | null;
  bounds: PsdBounds;
  hasMask: boolean;
  /** Name looks like a guide / sketch / reference layer (own name or a group name). */
  guide: boolean;
  /** Pixel statistics (pixel layers only): alpha-weighted centroid and mass at PSD resolution. */
  alphaSum: number;
  centroidX: number;
  centroidY: number;
}

export interface ParsedPsd {
  name: string;
  width: number;
  height: number;
  /** Groups and layers in tree pre-order (parents before children, siblings bottom → top). */
  nodes: PsdLayerNode[];
  /** Pixel layers bottom → top (subset of `nodes`). */
  layers: PsdLayerNode[];
  pixels: Map<string, PsdLayerPixels>;
  /** Looks like See-through output (every top-level layer name is a See-through tag). */
  seeThrough: boolean;
  bitsPerChannel: number;
}

const PATH_SEP = "/";

function toU8(data: ArrayLike<number> & { length: number }, bits: number): Uint8ClampedArray {
  if (data instanceof Uint8ClampedArray) return data;
  const out = new Uint8ClampedArray(data.length);
  const scale = bits === 16 ? 1 / 257 : bits === 32 ? 255 : 1;
  for (let i = 0; i < data.length; i++) out[i] = Math.round(data[i] * scale);
  return out;
}

function layerBounds(l: Layer): PsdBounds {
  return { left: l.left ?? 0, top: l.top ?? 0, right: l.right ?? 0, bottom: l.bottom ?? 0 };
}

/** Parses PSD bytes. Throws on unreadable files or PSDs without pixel layers. */
export function parsePsd(bytes: Uint8Array, name = "layers.psd"): ParsedPsd {
  if (bytes.length > MAX_PSD_BYTES) throw new Error("PSD 최대 크기는 300MB입니다.");
  let psd: Psd;
  try {
    psd = readPsd(bytes, {
      useImageData: true,
      skipCompositeImageData: true,
      skipThumbnail: true,
      skipLinkedFilesData: true,
    });
  } catch (e) {
    throw new Error(
      `PSD를 읽을 수 없습니다: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
  const bits = psd.bitsPerChannel ?? 8;
  const nodes: PsdLayerNode[] = [];
  const pixels = new Map<string, PsdLayerPixels>();
  const layers: PsdLayerNode[] = [];
  const undecoded: string[] = [];
  const walk = (
    children: Layer[] | undefined,
    parent: PsdLayerNode | null,
    ancestorHidden: boolean,
    ancestorOpacity: number,
    ancestorGuide: boolean,
    ancestorBlend: string | undefined,
  ) => {
    const used = new Map<string, number>();
    let lastBase: string | null = null;
    for (const l of children ?? []) {
      const rawName = (l.name ?? "").trim() || "(이름 없음)";
      const n = (used.get(rawName) ?? 0) + 1;
      used.set(rawName, n);
      const unique = n === 1 ? rawName : `${rawName}#${n}`;
      const path = `${parent ? parent.path : ""}${PATH_SEP}${unique.replace(/\//g, "∕")}`;
      const isGroup = Array.isArray(l.children);
      const hidden = Boolean(l.hidden);
      const opacity = (l.opacity ?? 1) * (l.fillOpacity ?? 1);
      const blendMode = l.blendMode ?? (isGroup ? "pass through" : "normal");
      const guide = ancestorGuide || isGuideName(rawName);
      const node: PsdLayerNode = {
        path,
        name: rawName,
        parentPath: parent?.path ?? null,
        depth: parent ? parent.depth + 1 : 0,
        isGroup,
        order: -1,
        hidden,
        effectiveHidden: ancestorHidden || hidden,
        opacity,
        effectiveOpacity: ancestorOpacity * opacity,
        blendMode,
        ...(ancestorBlend ? { groupBlendMode: ancestorBlend } : {}),
        clipping: !isGroup && Boolean(l.clipping),
        clipBase: null,
        bounds: layerBounds(l),
        hasMask: Boolean(l.mask?.imageData) && !l.mask?.disabled,
        guide,
        alphaSum: 0,
        centroidX: 0,
        centroidY: 0,
      };
      nodes.push(node);
      if (isGroup) {
        const groupBlend =
          blendMode !== "pass through" && blendMode !== "normal" ? blendMode : undefined;
        walk(
          l.children,
          node,
          node.effectiveHidden,
          node.effectiveOpacity,
          guide,
          ancestorBlend ?? groupBlend,
        );
        lastBase = null;
        continue;
      }
      if (node.clipping) node.clipBase = lastBase;
      else lastBase = path;
      node.order = layers.length;
      layers.push(node);
      const img = l.imageData;
      if (img && img.width > 0 && img.height > 0) {
        const px: PsdLayerPixels = {
          left: node.bounds.left,
          top: node.bounds.top,
          width: img.width,
          height: img.height,
          rgba: toU8(img.data as ArrayLike<number> & { length: number }, bits),
        };
        const m = l.mask;
        if (m?.imageData && !m.disabled) {
          px.mask = extractMaskChannel(toU8(m.imageData.data as ArrayLike<number> & { length: number }, bits));
          px.maskBounds = {
            left: m.left ?? 0,
            top: m.top ?? 0,
            right: m.right ?? 0,
            bottom: m.bottom ?? 0,
          };
          px.maskDefault = m.defaultColor ?? 0;
        } else if (m && !m.disabled && !m.imageData) {
          // A mask record without pixels is all `defaultColor`.
          px.mask = new Uint8ClampedArray(0);
          px.maskBounds = { left: 0, top: 0, right: 0, bottom: 0 };
          px.maskDefault = m.defaultColor ?? 0;
        }
        pixels.set(path, px);
        const stats = alphaStats(px);
        node.alphaSum = stats.sum;
        node.centroidX = stats.cx;
        node.centroidY = stats.cy;
      } else if (
        node.bounds.right > node.bounds.left &&
        node.bounds.bottom > node.bounds.top &&
        !l.text &&
        !l.placedLayer
      ) {
        undecoded.push(path);
      }
    }
  };
  walk(psd.children, null, false, 1, false, undefined);
  if (!layers.length) throw new Error("PSD에 이미지 레이어가 없습니다.");
  if (undecoded.length)
    throw new Error(
      `PSD 레이어 ${undecoded.length}개의 픽셀을 읽지 못했습니다(예: ${undecoded[0]}). 지원하지 않는 압축·색상 모드일 수 있습니다.`,
    );
  const top = (psd.children ?? []).map((l) => normTag(l.name ?? ""));
  const seeThrough = top.length > 0 && top.every((t) => t in SEE_THROUGH_TAGS);
  return {
    name: name.replace(/\.[^.]+$/, ""),
    width: psd.width,
    height: psd.height,
    nodes,
    layers,
    pixels,
    seeThrough,
    bitsPerChannel: bits,
  };
}

function extractMaskChannel(rgba: Uint8ClampedArray): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgba.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4];
  return out;
}

/** Layer mask value (0..255) at PSD pixel (x, y). */
export function maskAt(px: PsdLayerPixels, x: number, y: number): number {
  if (!px.mask || !px.maskBounds) return 255;
  const b = px.maskBounds;
  const mw = b.right - b.left;
  if (x < b.left || y < b.top || x >= b.right || y >= b.bottom || mw <= 0)
    return px.maskDefault ?? 0;
  return px.mask[(y - b.top) * mw + (x - b.left)];
}

function alphaStats(px: PsdLayerPixels) {
  let sum = 0,
    sx = 0,
    sy = 0;
  for (let y = 0; y < px.height; y++) {
    for (let x = 0; x < px.width; x++) {
      let a = px.rgba[(y * px.width + x) * 4 + 3];
      if (!a) continue;
      if (px.mask) a = (a * maskAt(px, px.left + x, px.top + y)) / 255;
      sum += a;
      sx += a * (px.left + x + 0.5);
      sy += a * (px.top + y + 0.5);
    }
  }
  return { sum: sum / 255, cx: sum ? sx / sum : 0, cy: sum ? sy / sum : 0 };
}

// ---------------------------------------------------------------------------------------------
// Mapping
// ---------------------------------------------------------------------------------------------

/** How a mapped non-`normal` layer is composited. `fold` applies its blend mode against the part's
 * layers below it (only for `FOLDABLE_BLEND_MODES`); `normal` composites it as `normal` (stated in
 * the UI and the report, never silent). */
export type LayerBlendHandling = "fold" | "normal";

export type MappingSource = "name" | "group" | "seethrough" | "manual" | "none";

export interface LayerMapping {
  path: string;
  /** Part id, a `{S}` template (`eye{S}_iris`) meaning "split this layer into character L/R by
   * connected components around the face midline", or null (unassigned). */
  target: string | null;
  include: boolean;
  /** Set only for layers whose blend mode (own or group) is not `normal`. */
  blend?: LayerBlendHandling;
  source: MappingSource;
  /** Why the layer is excluded / unassigned / what rule matched (UI and report). */
  note?: string;
}

export interface MappingWarning {
  path: string;
  kind: "blend" | "clip-base-hidden" | "draw-order" | "split";
  message: string;
}

export interface MappingSuggestion {
  mappings: LayerMapping[];
  /** Face midline x at PSD resolution used for L/R (null = canvas centre fallback). */
  midlineX: number;
  midlineSource: "face" | "iris" | "composite" | "canvas";
  warnings: MappingWarning[];
}

const DEFAULT_IDS: readonly string[] = PART_DEFS.map((d) => d[0]);

/** See-through tags → flat2rig ids (`tools/evaluation/v2_seethrough_to_layers.py` TAG_MAP and its
 * role-bearing extra ids). `{S}` = split into character L/R. */
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

function normTag(name: string) {
  return name.trim().toLowerCase().replace(/[_-]+/g, " ").replace(/\s+/g, " ");
}

const GUIDE_RE =
  /가이드|guide|ガイド|下書き|下描き|ラフ|밑그림|스케치|sketch|参考|참고|레퍼런스|reference|^ref\b|メモ|메모|memo|아타리|アタリ/i;

export function isGuideName(name: string): boolean {
  return GUIDE_RE.test(name);
}

type Rule = {
  re: RegExp;
  /** Part id, `{S}` template, or null for "recognised but ambiguous" (stays unassigned). */
  target: string | null;
  note?: string;
};

/**
 * Name rules, first match wins. Order matters: eyelash before eyebrow (속눈썹 ⊃ 눈썹, まつげ),
 * lower lid before upper, glasses before eye, earrings / necklaces before ear / neck.
 * Korean single-syllable words (코, 목, 귀, 입, 손) only match as a whole word.
 */
const WORD = (w: string) => `(?:^|[^가-힣])${w}(?:$|[^가-힣])`;
const NAME_RULES: readonly Rule[] = [
  // Ambiguous on purpose (Himikan truth leaves these out; the user decides).
  { re: /쌍커풀|쌍꺼풀|二重|double ?eyelid|crease/i, target: null, note: "쌍꺼풀 선: 얼굴/눈꺼풀 중 선택 필요" },
  { re: /가리개|cover(?!ed)|隠し/i, target: null, note: "가리개 레이어: 용도 확인 필요" },
  // Accessories that contain body-part words.
  { re: /안경|眼鏡|メガネ|めがね|glasses|eyewear|goggles?|고글/i, target: "acc_glasses" },
  { re: /귀걸이|피어싱|イヤリング|ピアス|earring|earwear|piercing/i, target: "acc_earwear" },
  { re: /목걸이|ネックレス|necklace|choker|초커/i, target: "acc_necklace" },
  { re: /모자|帽子|ぼうし|베레|beret|headwear|\bhat\b|\bcap\b|후드|フード|\bhood/i, target: "hat" },
  { re: /머리핀|헤어핀|ヘアピン|hairpin|hair ?clip|髪飾り|머리장식|머리띠|カチューシャ|headband/i, target: "acc_hairpin" },
  // Eyes.
  { re: /흰자|白目|eye ?white|eyewhite|sclera|눈알/i, target: "eye{S}_white" },
  { re: /눈동자|홍채|동공|瞳|虹彩|黒目|ひとみ|iris|irides|pupil/i, target: "eye{S}_iris" },
  { re: /눈언더|눈 ?아래|아랫 ?눈꺼풀|아래 ?속눈썹|下まつ毛|下まつげ|下睫毛|下まぶた|lower ?(?:eye)?(?:lash|lid)|eye ?under/i, target: "eye{S}_lower" },
  { re: /속눈썹|눈꺼풀|눈꼬리|아이라인|まつ毛|まつげ|睫毛|まぶた|瞼|上まぶた|アイライン|eyelash|eyelid|\blash|upper ?lid|eye ?line/i, target: "eye{S}_upper" },
  { re: /눈썹|眉|まゆ|brow/i, target: "brow{S}" },
  // Mouth.
  { re: /윗입술|上唇|うわくちびる|upper ?lip/i, target: "mouth_upper" },
  { re: /아랫입술|下唇|したくちびる|lower ?lip/i, target: "mouth_lower" },
  { re: /입안|입 ?속|口内|口の中|口腔|혀|舌|tongue|윗니|아랫니|이빨|치아|歯|teeth|tooth|inner ?mouth|mouth ?inner/i, target: "mouth_inner" },
  { re: new RegExp(`${WORD("입")}|^口$|^くち$|^mouth$`, "i"), target: "mouth_all", note: "입 전체 한 장" },
  // Face.
  { re: new RegExp(`${WORD("코")}|鼻|はな|\\bnose`, "i"), target: "face_detail" },
  { re: /홍조|볼터치|頬|ほお|チーク|blush|cheek/i, target: "face" },
  { re: new RegExp(`${WORD("귀")}|耳|みみ|\\bears?\\b`, "i"), target: "face" },
  { re: /얼굴|顔|かお|\bface\b|윤곽|輪郭/i, target: "face" },
  // Hair.
  { re: /앞머리|前髪|まえがみ|front ?hair|bangs?\b|fringe|더듬이|アホ毛|あほ毛|ahoge|아호게/i, target: "hair_front" },
  { re: /옆머리|横髪|よこがみ|サイド|もみあげ|side ?hair|sidelocks?|触覚/i, target: "hair_side_{S}" },
  { re: /뒷머리|後ろ髪|後髪|うしろがみ|back ?hair|ポニー|포니|ponytail|트윈테일|ツインテ|twin ?tail|おさげ|땋은/i, target: "hair_back" },
  // Neck / body / clothes.
  { re: /리본|ribbon|リボン|넥타이|ネクタイ|necktie|neckwear|스카프|スカーフ|scarf|neckerchief|타이$/i, target: "clothes_inner" },
  { re: new RegExp(`${WORD("목")}|首|くび|\\bneck\\b`, "i"), target: "neck" },
  { re: /피부|맨살|素肌|\bskin\b|bare ?body/i, target: "body" },
  { re: new RegExp(`${WORD("손")}|${WORD("팔")}|手袋|腕|\\bhands?\\b|\\barms?\\b|handwear|glove|장갑`, "i"), target: "hands" },
  { re: /치마|스커트|スカート|skirt|바지|ズボン|pants|trousers|bottomwear/i, target: "clothes_bottom" },
  { re: /옷|의상|몸통|상의|셔츠|服|衣装|制服|シャツ|上着|胴|카라|칼라|collar|襟|えり|topwear|clothes|clothing|shirt|jacket|coat|uniform|dress|소매|袖|sleeve|torso|body/i, target: "clothes" },
];

/** Generic names that only make sense with their group (highlight, shadow, line, colour, base). */
const GENERIC_RE =
  /^(?:하이라이트|ハイライト|highlight|hl|그림자|影|かげ|shadow|선화?|線画?|line ?art|lines?|색|채색|塗り|color|colour|베이스|base|기본|광택|reflection)(?:$|[\s_\-\d<>#.()])/i;
const EYE_CONTEXT_HIGHLIGHT_RE = /하이라이트|ハイライト|highlight|\bhl\b|반사|光/i;
const EYE_CONTEXT_RE = /(?:^|[^가-힣])눈(?!썹)|目|瞳|\beyes?\b/i;

function stripSide(name: string) {
  return name.replace(/[<>]/g, " ").trim();
}

function matchRule(name: string): Rule | null {
  const n = stripSide(name);
  for (const r of NAME_RULES) if (r.re.test(n)) return r;
  return null;
}

function hairNumber(id: string, groupIndex: number) {
  return groupIndex <= 1 ? id : `${id}_${groupIndex}`;
}

/** Character side of a point at PSD x (AGENTS.md 1절: L = screen right). */
export function sideOf(x: number, midlineX: number): "L" | "R" {
  return x > midlineX ? "L" : "R";
}

/** Fraction of a layer's alpha mass on each side of the midline (PSD resolution). */
export function sideMass(psd: ParsedPsd, path: string, midlineX: number) {
  const px = psd.pixels.get(path);
  let left = 0,
    right = 0;
  if (!px) return { L: 0, R: 0 };
  for (let y = 0; y < px.height; y++) {
    for (let x = 0; x < px.width; x++) {
      const a = px.rgba[(y * px.width + x) * 4 + 3];
      if (!a) continue;
      if (px.left + x + 0.5 > midlineX) left += a;
      else right += a;
    }
  }
  const t = left + right || 1;
  return { L: left / t, R: right / t };
}

/** Below this share of alpha mass on the minor side, a layer belongs to one side. */
export const SPLIT_MINOR_SHARE = 0.15;

function resolveSide(
  psd: ParsedPsd,
  node: PsdLayerNode,
  template: string,
  midlineX: number,
): { target: string; split: boolean } {
  if (!template.includes("{S}")) return { target: template, split: false };
  const m = sideMass(psd, node.path, midlineX);
  if (Math.min(m.L, m.R) > SPLIT_MINOR_SHARE) return { target: template, split: true };
  return { target: template.replace("{S}", m.L >= m.R ? "L" : "R"), split: false };
}

function estimateMidline(
  psd: ParsedPsd,
  classes: Map<string, string | null>,
): { x: number; source: MappingSuggestion["midlineSource"] } {
  const usable = (n: PsdLayerNode) => !n.effectiveHidden && !n.guide && n.alphaSum > 0;
  const faces = psd.layers.filter(
    (n) => usable(n) && classes.get(n.path) === "face" && !/귀|耳|ear/i.test(n.name),
  );
  if (faces.length) {
    const best = faces.reduce((a, b) => (b.alphaSum > a.alphaSum ? b : a));
    return { x: best.centroidX, source: "face" };
  }
  const irises = psd.layers.filter(
    (n) => usable(n) && (classes.get(n.path) ?? "").includes("_iris"),
  );
  if (irises.length) {
    let s = 0,
      sx = 0;
    for (const n of irises) {
      s += n.alphaSum;
      sx += n.alphaSum * n.centroidX;
    }
    if (s > 0) return { x: sx / s, source: "iris" };
  }
  const all = psd.layers.filter(usable);
  let s = 0,
    sx = 0;
  for (const n of all) {
    s += n.alphaSum;
    sx += n.alphaSum * n.centroidX;
  }
  if (s > 0) return { x: sx / s, source: "composite" };
  return { x: psd.width / 2, source: "canvas" };
}

/** Effective blend mode of a pixel layer for handling (own, else the group's non-normal one). */
export function effectiveBlend(node: PsdLayerNode): string {
  if (node.blendMode !== "normal" && node.blendMode !== "pass through") return node.blendMode;
  return node.groupBlendMode ?? "normal";
}

export function isFoldable(mode: string): mode is FoldableBlendMode {
  return (FOLDABLE_BLEND_MODES as readonly string[]).includes(mode);
}

/**
 * Proposes a mapping for every pixel layer. Suggestions come from (in order) the See-through tag
 * table, the layer's own name, then its nearest group whose name matches a rule (for generic
 * layer names such as "선", "影", "highlight"). Hair layers of the same class in different groups
 * get numbered ids (`hair_back`, `hair_back_2`, …) so they can move independently; within one
 * group they merge into one part.
 */
export function suggestMapping(psd: ParsedPsd): MappingSuggestion {
  const byPath = new Map(psd.nodes.map((n) => [n.path, n]));
  const classes = new Map<string, string | null>();
  const sources = new Map<string, MappingSource>();
  const notes = new Map<string, string>();
  const groupOf = new Map<string, string | null>();
  for (const node of psd.layers) {
    let target: string | null = null;
    let source: MappingSource = "none";
    let note: string | undefined;
    let matchedGroup: string | null = null;
    const tag = normTag(node.name);
    if (psd.seeThrough && tag in SEE_THROUGH_TAGS) {
      target = SEE_THROUGH_TAGS[tag];
      source = "seethrough";
    } else {
      const own = matchRule(node.name);
      const generic = GENERIC_RE.test(stripSide(node.name));
      if (own && !generic) {
        target = own.target;
        note = own.note;
        source = own.target ? "name" : "none";
        matchedGroup = node.parentPath;
      } else {
        // Eye highlight: "눈 하이라이트", or "highlight" inside an eye group.
        const ancestors: PsdLayerNode[] = [];
        for (let p = node.parentPath; p; p = byPath.get(p)?.parentPath ?? null) {
          const g = byPath.get(p);
          if (g) ancestors.push(g);
        }
        const eyeCtx =
          EYE_CONTEXT_RE.test(node.name) || ancestors.some((g) => EYE_CONTEXT_RE.test(g.name));
        if (EYE_CONTEXT_HIGHLIGHT_RE.test(node.name) && eyeCtx) {
          target = "eye{S}_iris";
          source = "name";
          note = "눈 하이라이트 → 홍채";
        } else {
          for (const g of ancestors) {
            const r = matchRule(g.name);
            if (r) {
              target = r.target;
              note = r.note;
              source = r.target ? "group" : "none";
              matchedGroup = g.path;
              break;
            }
          }
          if (!target && own) {
            target = own.target;
            note = own.note;
            source = own.target ? "name" : "none";
          }
        }
      }
    }
    classes.set(node.path, target);
    sources.set(node.path, source);
    if (note) notes.set(node.path, note);
    groupOf.set(node.path, matchedGroup ?? node.parentPath);
  }
  const mid = estimateMidline(psd, classes);
  // Hair numbering: same hair class in a different group gets the next number.
  const hairGroups = new Map<string, string[]>();
  const mappings: LayerMapping[] = [];
  const warnings: MappingWarning[] = [];
  for (const node of psd.layers) {
    let target = classes.get(node.path) ?? null;
    const source = sources.get(node.path) ?? "none";
    let note = notes.get(node.path);
    let include = true;
    // A guide layer is not artwork: no part suggestion even when its group name matches a rule.
    if (node.guide) target = null;
    if (node.alphaSum <= 0) {
      include = false;
      note = "빈 레이어";
    } else if (node.effectiveHidden) {
      include = false;
      note = node.guide ? "숨김 · 가이드 레이어" : "숨김 레이어";
    } else if (node.guide) {
      include = false;
      note = "가이드 레이어";
    }
    if (target) {
      const r = resolveSide(psd, node, target, mid.x);
      target = r.target;
      if (r.split)
        warnings.push({
          path: node.path,
          kind: "split",
          message: `${node.name}: 좌우가 한 레이어 → 연결 요소로 L/R 분리 제안`,
        });
    }
    if (target && /^hair_(front|back|side_[LR])$/.test(target) && source !== "seethrough") {
      const key = target;
      const g = groupOf.get(node.path) ?? "";
      const list = hairGroups.get(key) ?? [];
      if (include && !list.includes(g)) list.push(g);
      hairGroups.set(key, list);
      const idx = list.indexOf(g) + 1;
      if (idx > 0) target = hairNumber(target, idx);
    }
    const m: LayerMapping = { path: node.path, target, include, source };
    if (note) m.note = note;
    const blend = effectiveBlend(node);
    if (blend !== "normal") {
      m.blend = isFoldable(blend) ? "fold" : "normal";
      if (!isFoldable(blend)) {
        m.include = false;
        m.note = `${blend} 블렌드: 파츠 안에서 재현 불가 → 기본 제외`;
      }
      if (include)
        warnings.push({
          path: node.path,
          kind: "blend",
          message: isFoldable(blend)
            ? `${node.name}: ${blend} 블렌드 — 파츠 안에서 아래 레이어에 ${blend}로 접어 넣습니다(리그 출력은 normal 한 장).`
            : `${node.name}: ${blend} 블렌드 — 그대로 재현할 수 없어 기본 제외. 포함하면 normal로 합성됩니다.`,
        });
    }
    if (node.clipping && node.clipBase) {
      const base = byPath.get(node.clipBase);
      if (base?.effectiveHidden && !node.effectiveHidden) {
        m.include = false;
        m.note = "클리핑 기준 레이어가 숨김";
        warnings.push({
          path: node.path,
          kind: "clip-base-hidden",
          message: `${node.name}: 클리핑 기준(${base.name})이 숨김이라 제외`,
        });
      }
    }
    if (!m.target && m.include) m.note = m.note ?? "미지정: 파츠를 골라 주세요";
    mappings.push(m);
  }
  warnings.push(...drawOrderWarnings(psd, mappings));
  return { mappings, midlineX: mid.x, midlineSource: mid.source, warnings };
}

/** Face features drawn below the face, or an eye white above its iris (See-through depth order
 * defects, V2 report). The PSD order is kept; the user is told. */
function drawOrderWarnings(psd: ParsedPsd, mappings: LayerMapping[]): MappingWarning[] {
  const out: MappingWarning[] = [];
  const pos = new Map<string, number>();
  psd.layers.forEach((n, i) => {
    const m = mappings[i];
    if (!m.include || !m.target) return;
    for (const id of expandTarget(m.target)) pos.set(id, Math.max(pos.get(id) ?? -1, n.order));
  });
  const face = pos.get("face");
  if (face !== undefined) {
    for (const [id, o] of pos) {
      if (o < face && /^(eye|brow|mouth|acc_glasses|face_detail)/.test(id))
        out.push({
          path: id,
          kind: "draw-order",
          message: `${id}가 얼굴보다 아래에 있어 가려질 수 있습니다(PSD 순서 유지).`,
        });
    }
  }
  for (const s of ["L", "R"]) {
    const w = pos.get(`eye${s}_white`),
      i = pos.get(`eye${s}_iris`);
    if (w !== undefined && i !== undefined && w > i)
      out.push({
        path: `eye${s}_white`,
        kind: "draw-order",
        message: `eye${s}_white가 홍채 위에 있습니다(PSD 순서 유지).`,
      });
  }
  return out;
}

/** Concrete part ids a mapping target produces (`{S}` → both sides). */
export function expandTarget(target: string): string[] {
  return target.includes("{S}") ? [target.replace("{S}", "L"), target.replace("{S}", "R")] : [target];
}

/** Part ids offered in the mapping UI: the 18 defaults, then extras. */
export function mappingTargetOptions(extra: readonly string[] = []): string[] {
  const base = [
    ...DEFAULT_IDS,
    "hair_side_L",
    "hair_side_R",
    "face_detail",
    "clothes_inner",
    "hat",
    "acc_glasses",
    "hands",
    "mouth_all",
  ];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const id of [...base, ...extra]) {
    if (!seen.has(id)) {
      seen.add(id);
      out.push(id);
    }
  }
  return out;
}

/** Default part ids with no included layer (shown as "없음", never auto-synthesized). */
export function missingDefaultParts(mappings: readonly LayerMapping[]): string[] {
  const have = new Set<string>();
  for (const m of mappings) if (m.include && m.target) for (const id of expandTarget(m.target)) have.add(id);
  if (have.has("mouth_all")) for (const id of ["mouth_inner", "mouth_lower", "mouth_upper"]) have.add(id);
  return DEFAULT_IDS.filter((id) => !have.has(id));
}
