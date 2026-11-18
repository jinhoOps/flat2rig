export type Point = { x: number; y: number };
export type LandmarkId = "eyeL" | "eyeR" | "mouth" | "chin" | "crown";
export type Landmarks = Record<LandmarkId, Point>;
export type LandmarkSource = "model" | "estimated" | "manual";
export type LandmarkProvenance = Record<LandmarkId, LandmarkSource>;
export type PartStatus = "draft" | "approved" | "missing";
export type PartProvenance = "model" | "prompted" | "geometry" | "manual";
export type FillKind = "none" | "neighbor" | "external" | "template";
/** T9: region type of the part's exposure band, from clean inner-seed statistics (Rust
 * `analysis::region_fill`). Separate from `fillKind`, which is the fill *source*. */
export type FillRegion = "flat_cel" | "line" | "texture" | "semantic";
export const FILL_REGIONS: readonly FillRegion[] = ["flat_cel", "line", "texture", "semantic"];
/** T9: how the hidden colours were made. `diffusion` = neighbour-averaging draft (browser fallback
 * `expandFillSeparated` or a Rust part without own seeds) — lower quality, type unclassified. */
export type FillSynth =
  | "palette_levelset"
  | "local_field"
  | "class_field"
  | "procedural"
  | "diffusion"
  | "none";
export const FILL_SYNTHS: readonly FillSynth[] = [
  "palette_levelset",
  "local_field",
  "class_field",
  "procedural",
  "diffusion",
  "none",
];
export type BlendMode = "normal" | "multiply" | "additive";
export const ALLOWED_BLEND_MODES: readonly BlendMode[] = [
  "normal",
  "multiply",
  "additive",
] as const;

export interface TemplateStyleSelection {
  proportionMode?: string;
  faceStyle?: string;
  hairFrontStyle?: string;
  hairSideStyle?: string;
  hairBackStyle?: string;
  eyeStyle?: string;
  mouthStyle?: string;
  clothesStyle?: string;
}

export interface FittedCharacterMetrics {
  faceRxRatio: number;
  jawTaper: number;
  eyeRxRatio: number;
  eyeRyRatio: number;
  irisRadiusRatio: number;
  mouthWidthRatio: number;
  mouthOpenRatio: number;
  bangCoverage: number;
  centerPartGap: number;
  sideHairReach: number;
  backHairReach: number;
  shoulderWidthRatio: number;
}

export interface StyleMatchReport {
  resolved: TemplateStyleSelection;
  matchScore: number;
  metrics: FittedCharacterMetrics;
  paletteSummary: Record<string, string>;
}

export interface StyleCategoryDef {
  key: keyof TemplateStyleSelection;
  label: string;
  options: readonly { id: string; label: string }[];
}

export const TEMPLATE_STYLE_CATALOG: readonly StyleCategoryDef[] = [
  {
    key: "proportionMode",
    label: "체형 실루엣",
    options: [
      { id: "auto", label: "자동 판별 (LD/SD)" },
      { id: "ld", label: "일반 등신 (LD)" },
      { id: "sd", label: "SD · 치비 (큰 눈)" },
    ],
  },
  {
    key: "faceStyle",
    label: "얼굴형",
    options: [
      { id: "auto", label: "자동 매칭" },
      { id: "oval_anime", label: "표준 계란형" },
      { id: "soft_round", label: "부드러운 둥근형" },
      { id: "sharp_v", label: "날렵 V라인" },
      { id: "chibi_plump", label: "통통 볼살형" },
    ],
  },
  {
    key: "hairFrontStyle",
    label: "앞머리",
    options: [
      { id: "auto", label: "자동 매칭" },
      { id: "center_part", label: "가르마 M형" },
      { id: "parted_curtain", label: "넓은 여신가르마" },
      { id: "blunt_bangs", label: "일자 풀뱅" },
      { id: "side_sweep_right", label: "우측 사선넘김" },
      { id: "side_sweep_left", label: "좌측 사선넘김" },
      { id: "wispy_air", label: "시스루 웨이브" },
    ],
  },
  {
    key: "hairSideStyle",
    label: "옆머리",
    options: [
      { id: "auto", label: "자동 매칭" },
      { id: "chest_locks", label: "가슴 길이" },
      { id: "chin_bob", label: "턱선 보브" },
      { id: "short_tuck", label: "귀뒤 짧은형" },
      { id: "twin_frame", label: "트윈 롱프레임" },
      { id: "minimal", label: "없음/최소" },
    ],
  },
  {
    key: "hairBackStyle",
    label: "뒷머리",
    options: [
      { id: "auto", label: "자동 매칭" },
      { id: "long_wave", label: "롱 웨이브" },
      { id: "long_straight", label: "롱 스트레이트" },
      { id: "shoulder_bob", label: "어깨 단발" },
      { id: "short_crop", label: "숏컷" },
      { id: "twin_tails", label: "트윈테일 볼륨" },
    ],
  },
  {
    key: "eyeStyle",
    label: "눈매",
    options: [
      { id: "auto", label: "자동 매칭" },
      { id: "round_bright", label: "초롱 둥근눈" },
      { id: "tareme_large", label: "큰 처진눈(타레메)" },
      { id: "almond_sharp", label: "날렵 캣아이" },
      { id: "droopy_gentle", label: "온순 반달눈" },
    ],
  },
  {
    key: "mouthStyle",
    label: "입모양",
    options: [
      { id: "auto", label: "자동 매칭" },
      { id: "soft_smile", label: "부드러운 미소" },
      { id: "cat_w", label: "고양이 입(w)" },
      { id: "open_happy", label: "밝은 웃음" },
      { id: "open_o", label: "작은 'o' 입 (치비)" },
      { id: "calm_line", label: "차분 일자입" },
    ],
  },
  {
    key: "clothesStyle",
    label: "의상 넥라인",
    options: [
      { id: "auto", label: "자동 매칭" },
      { id: "crew_sweater", label: "라운드 니트/맨투맨" },
      { id: "sailor_collar", label: "세일러 카라" },
      { id: "v_neck_cardigan", label: "V넥 가디건" },
      { id: "hoodie_casual", label: "하이넥/후디" },
      { id: "blazer_suit", label: "블레이저/셔츠" },
    ],
  },
];
export type HairRole = "hair_front" | "hair_side_L" | "hair_side_R" | "hair_back";
export type WearableRole = "hat" | "accessory" | "hands";
export type PartRole =
  | HairRole
  | WearableRole
  | "body"
  | "clothes"
  | "neck"
  | "face"
  | "eye"
  | "brow"
  | "mouth"
  | "other";
export type AnalysisStatus =
  | "idle"
  | "detected"
  | "no-face"
  | "model-missing"
  | "browser-fallback"
  | "error";

export interface PromptPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

export interface Landmark28Point {
  index: number;
  x: number;
  y: number;
  score: number;
}

export interface FaceCandidate {
  id: number;
  bbox: [number, number, number, number];
  cropBbox: [number, number, number, number];
  score: number;
  areaFraction: number;
  center: [number, number];
}

export interface AffineTransform2D {
  matrix2: [number, number, number, number];
  translation: [number, number];
}

export interface ModelStatusEntry {
  role: string;
  id: string;
  filename: string;
  version: string;
  license: string;
  expectedSha256: string;
  expectedBytes: number;
  downloadUrl?: string | null;
  path?: string | null;
  actualSha256?: string | null;
  installed: boolean;
  hashMatches: boolean;
  loaded: boolean;
  error?: string | null;
}

export interface Part {
  id: string;
  name: string;
  group: string;
  color: string;
  visible: boolean;
  status: PartStatus;
  role?: PartRole;
  /** Layer blend mode strictly restricted to normal, multiply, or additive. */
  blendMode?: BlendMode;
  /** Strictly non-overlapping visible mask extracted from the original artwork. */
  mask: Uint8Array;
  /** Reconstructed behind-occlusion pixels added independently of `mask`. */
  fillMask?: Uint8Array;
  texture?: string;
  generated?: boolean;
  provenance?: PartProvenance;
  fillKind?: FillKind;
  /** T9 region type (absent = unclassified, e.g. browser fallback or projects saved before T9). */
  fillRegion?: FillRegion;
  /** T9 hidden-colour synthesis method (absent = unknown / saved before T9). */
  fillSynth?: FillSynth;
  compositeOrder?: number;
  prompts?: PromptPoint[];
  confidence?: number;
  diagnostics?: string[];
}

export type RigProportionMode = "ld" | "sd";

export interface RegionHandleRange {
  min: number;
  max: number;
}

export type ShapeRegion = "eye_L" | "eye_R" | "mouth";

export type EyeShapeKind = "original" | "half_closed" | "closed" | "smiling" | "custom";
export type MouthShapeKind = "original" | "closed" | "a" | "i" | "u" | "e" | "o" | "custom";

export interface ShapeHandleValues {
  /** Normalized horizontal width multiplier relative to base region width (0.5..1.5, default 1). */
  width: number;
  /** Vertical openness / height ratio (0..1.4). */
  open: number;
  /** Roundness vs sharpness of lid/lip contour (0..1). */
  roundness: number;
  /** Corner lift or droop (-1..1). */
  corner: number;
  /** Upper lid / lip arc bend (-1..1). */
  upperCurve: number;
  /** Lower lid / lip arc bend (-1..1). */
  lowerCurve: number;
  /** Mouth interior (teeth/tongue) visibility or eye iris/highlight visibility weight (0..1). */
  interiorVisibility: number;
  /** Horizontal offset normalized to region size (-0.5..0.5). */
  offsetX: number;
  /** Vertical offset normalized to region size (-0.5..0.5). */
  offsetY: number;
}

export interface ShapeSubstituteLayer {
  /** Target part ID being substituted or added for this shape (e.g. eyeL_upper, mouth_inner). */
  partId: string;
  /** Layer blend mode strictly restricted to normal, multiply, or additive. */
  blendMode?: BlendMode;
  /** Bounding box in working canvas pixels. */
  bounds: {
    left: number;
    top: number;
    width: number;
    height: number;
  };
  /** Base part IDs whose rest pixels are hidden when this substitute asset is active. */
  hidesPartIds?: string[];
  /** Cropped mask aligned with `bounds` (`bounds.width * bounds.height`), persisted as shapes/<assetId>/<partId>.bin. */
  mask?: Uint8Array;
  /** Optional fill mask aligned with `bounds`, persisted as shapes/<assetId>/<partId>.fill.bin. */
  fillMask?: Uint8Array;
  /** Optional RGBA PNG bytes persisted as shapes/<assetId>/<partId>.png. */
  pngBytes?: Uint8Array;
  /** Serialization hint set in project.json when binary mask exists in ZIP. */
  hasMask?: boolean;
  /** Serialization hint set in project.json when fill mask exists in ZIP. */
  hasFillMask?: boolean;
  /** Serialization hint set in project.json when PNG sprite exists in ZIP. */
  hasTexturePng?: boolean;
}

export interface ShapeSubstituteAsset {
  assetId: string;
  region: ShapeRegion;
  recipeVersion: number;
  hidesPartIds: string[];
  layers: ShapeSubstituteLayer[];
}

export interface ShapePreset {
  /** Stable preset identifier distinct from display `label`. */
  id: string;
  region: ShapeRegion;
  label: string;
  kind: EyeShapeKind | MouthShapeKind;
  recipeVersion: number;
  builtIn?: boolean;
  mirrorLinked?: boolean;
  handles: ShapeHandleValues;
  parameterOverrides: Partial<Parameters>;
  /** Optional reference into `Project.shapeAssets`. */
  assetId?: string;
}

export interface ExpressionPreset {
  /** Stable unique identifier distinct from display `name`. */
  id: string;
  name: string;
  builtIn?: boolean;
  shapeIds: {
    eye_L?: string;
    eye_R?: string;
    mouth?: string;
  };
  parameters: Partial<Parameters>;
  handleTweaks?: Partial<Record<ShapeRegion, Partial<ShapeHandleValues>>>;
  /** Blending multiplier (0..1.5, default 1). */
  strength: number;
  /** Interpolation duration in milliseconds (0..3000, default 180). */
  transitionMs: number;
  shortcutKey?: string;
}

export type PipelineStage =
  | "base_silhouette"
  | "features_overlay"
  | "clothes_overlay"
  | "hair_overlay"
  | "full";

export interface RigProfile {
  proportionMode: RigProportionMode;
  defaultShapeIds: {
    eye_L: string;
    eye_R: string;
    mouth: string;
  };
  neutralParameters: Parameters;
  safeBounds: Partial<Record<keyof Parameters, RegionHandleRange>>;
  motionScale: {
    head: number;
    eyes: number;
    mouth: number;
    hair: number;
    body: number;
  };
  clippingTargets?: Record<string, string>;
}

export interface PlaybackSettings {
  defaultExpressionId: string;
  autoBlink: boolean;
  blinkIntervalSec: number;
  autoBreath: boolean;
  autoIdle: boolean;
  autoMouthDemo?: boolean;
  shortcuts: Record<string, string>;
}

export interface Project {
  schemaVersion: 3;
  name: string;
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
  source: string;
  original: string;
  imageHash?: string;
  workToOriginalTransform?: AffineTransform2D;
  landmarks: Landmarks;
  landmarkProvenance?: LandmarkProvenance;
  landmarks28?: Landmark28Point[];
  faceCandidates?: FaceCandidate[];
  selectedFaceId?: number;
  parts: Part[];
  revision: number;
  provider: string;
  modelHash?: string;
  modelHashes?: Record<string, string>;
  postprocessVersions?: Record<string, string>;
  analysisStatus?: AnalysisStatus;
  analysisNote?: string;
  detailMode?: "detail" | "parent";
  rigStage?: PipelineStage | "base_body";
  styleMatch?: StyleMatchReport;
  styleSelection?: TemplateStyleSelection;
  rigProfile: RigProfile;
  shapePresets: ShapePreset[];
  shapeAssets?: Record<string, ShapeSubstituteAsset>;
  expressions: ExpressionPreset[];
  activeExpressionId: string;
  playback: PlaybackSettings;
  createdAt: string;
}
export interface Mesh {
  vertices: Point[];
  triangles: number[];
  uv: Point[];
}
export interface Parameters {
  angleX: number;
  angleY: number;
  angleZ: number;
  eyeL: number;
  eyeLSmile: number;
  eyeR: number;
  eyeRSmile: number;
  eyeBallX: number;
  eyeBallY: number;
  browLY: number;
  browLForm: number;
  browRY: number;
  browRForm: number;
  mouthForm: number;
  mouth: number;
  bodyAngleX: number;
  bodyAngleY: number;
  bodyAngleZ: number;
  breath: number;
  hair: number;
  hairSide: number;
  hairBack: number;
}
export const REST: Parameters = {
  angleX: 0,
  angleY: 0,
  angleZ: 0,
  eyeL: 1,
  eyeLSmile: 0,
  eyeR: 1,
  eyeRSmile: 0,
  eyeBallX: 0,
  eyeBallY: 0,
  browLY: 0,
  browLForm: 0,
  browRY: 0,
  browRForm: 0,
  mouthForm: 0,
  mouth: 0,
  bodyAngleX: 0,
  bodyAngleY: 0,
  bodyAngleZ: 0,
  breath: 0,
  hair: 0,
  hairSide: 0,
  hairBack: 0,
};
export type Tool = "select" | "add" | "erase" | "prompt" | "landmark" | "face";
export const PARENT_PART_DEFS = [
  ["eyeR_all", "눈 R 전체", "눈 R", "#74c7c0"],
  ["eyeL_all", "눈 L 전체", "눈 L", "#86b7f5"],
  ["mouth_all", "입 전체", "입", "#e5a0b6"],
] as const;
export const PARENT_TO_DETAIL_IDS: Record<string, readonly string[]> = {
  eyeR_all: ["eyeR_white", "eyeR_iris", "eyeR_lower", "eyeR_upper"],
  eyeL_all: ["eyeL_white", "eyeL_iris", "eyeL_lower", "eyeL_upper"],
  mouth_all: ["mouth_inner", "mouth_lower", "mouth_upper"],
};
export const PART_DEFS = [
  ["hair_back", "뒷머리", "뒷머리", "#8c83ff"],
  ["body", "몸통", "몸", "#6ba5ff"],
  ["neck", "목", "몸", "#f3b48c"],
  ["face", "얼굴", "얼굴", "#ecaa9e"],
  ["eyeR_white", "눈흰자 R", "눈 R", "#89d4cc"],
  ["eyeR_iris", "눈동자 R", "눈 R", "#5ab9bf"],
  ["eyeR_lower", "눈언더 R", "눈 R", "#6bb3ac"],
  ["eyeR_upper", "눈꺼풀 R", "눈 R", "#a1d8d1"],
  ["eyeL_white", "눈흰자 L", "눈 L", "#9bc4ff"],
  ["eyeL_iris", "눈동자 L", "눈 L", "#76a9ed"],
  ["eyeL_lower", "눈언더 L", "눈 L", "#7fbae5"],
  ["eyeL_upper", "눈꺼풀 L", "눈 L", "#b3d5ff"],
  ["browR", "눈썹 R", "눈썹", "#d5b783"],
  ["browL", "눈썹 L", "눈썹", "#d5b783"],
  ["mouth_inner", "입안", "입", "#d686a5"],
  ["mouth_lower", "아랫입술", "입", "#e5a0b6"],
  ["mouth_upper", "윗입술", "입", "#efb1c7"],
  ["hair_front", "앞머리", "앞옆머리", "#b3a0f5"],
] as const;
export const PARAM_GROUPS = [
  { id: "ParamGroup", label: "XYZ" },
  { id: "ParamGroup2", label: "눈" },
  { id: "ParamGroup3", label: "눈썹" },
  { id: "ParamGroup4", label: "입" },
  { id: "ParamGroup5", label: "몸" },
  { id: "ParamGroup6", label: "물리" },
] as const;
export const PARAM_DEFS = [
  {
    key: "angleX",
    id: "ParamAngleX",
    name: "고개 좌우 (각도 X)",
    groupId: "ParamGroup",
    group: "XYZ",
    description: "+로 화면의 오른쪽을 향하기",
    min: -30,
    max: 30,
    value: 0,
    previewMin: -15,
    previewMax: 15,
  },
  {
    key: "angleY",
    id: "ParamAngleY",
    name: "고개 상하 (각도 Y)",
    groupId: "ParamGroup",
    group: "XYZ",
    description: "+로 화면 상단을 향하기",
    min: -30,
    max: 30,
    value: 0,
    previewMin: -15,
    previewMax: 15,
  },
  {
    key: "angleZ",
    id: "ParamAngleZ",
    name: "고개 기울기 (각도 Z)",
    groupId: "ParamGroup",
    group: "XYZ",
    description: "+로 화면의 오른쪽으로 기울이기",
    min: -30,
    max: 30,
    value: 0,
    previewMin: -15,
    previewMax: 15,
  },
  {
    key: "eyeL",
    id: "ParamEyeLOpen",
    name: "왼쪽 눈 깜빡임",
    groupId: "ParamGroup2",
    group: "눈",
    description: "+로 눈을 뜨기",
    min: 0,
    max: 1,
    value: 1,
    previewMin: 0,
    previewMax: 1,
  },
  {
    key: "eyeLSmile",
    id: "ParamEyeLSmile",
    name: "왼쪽 눈 미소",
    groupId: "ParamGroup2",
    group: "눈",
    description: "+로 웃는 눈",
    min: 0,
    max: 1,
    value: 0,
    previewMin: 0,
    previewMax: 1,
  },
  {
    key: "eyeR",
    id: "ParamEyeROpen",
    name: "오른쪽 눈 깜빡임",
    groupId: "ParamGroup2",
    group: "눈",
    description: "+로 눈을 뜨기",
    min: 0,
    max: 1,
    value: 1,
    previewMin: 0,
    previewMax: 1,
  },
  {
    key: "eyeRSmile",
    id: "ParamEyeRSmile",
    name: "오른쪽 눈 미소",
    groupId: "ParamGroup2",
    group: "눈",
    description: "+로 웃는 눈",
    min: 0,
    max: 1,
    value: 0,
    previewMin: 0,
    previewMax: 1,
  },
  {
    key: "eyeBallX",
    id: "ParamEyeBallX",
    name: "눈 X",
    groupId: "ParamGroup2",
    group: "눈",
    description: "+로 오른쪽 보기",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
  {
    key: "eyeBallY",
    id: "ParamEyeBallY",
    name: "눈 Y",
    groupId: "ParamGroup2",
    group: "눈",
    description: "+로 위를 보기",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
  {
    key: "browLY",
    id: "ParamBrowLY",
    name: "왼쪽 눈썹 상하",
    groupId: "ParamGroup3",
    group: "눈썹",
    description: "+로 눈썹을 올리기",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
  {
    key: "browLForm",
    id: "ParamBrowLForm",
    name: "왼쪽 눈썹 변형",
    groupId: "ParamGroup3",
    group: "눈썹",
    description: "-로 눈썹을 화나게",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
  {
    key: "browRY",
    id: "ParamBrowRY",
    name: "오른쪽 눈썹 상하",
    groupId: "ParamGroup3",
    group: "눈썹",
    description: "+로 눈썹을 올리기",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
  {
    key: "browRForm",
    id: "ParamBrowRForm",
    name: "오른쪽 눈썹 변형",
    groupId: "ParamGroup3",
    group: "눈썹",
    description: "-로 눈썹을 화나게",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
  {
    key: "mouthForm",
    id: "ParamMouthForm",
    name: "입 변형",
    groupId: "ParamGroup4",
    group: "입",
    description: "+로 웃는 입, -로 화가난 입",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
  {
    key: "mouth",
    id: "ParamMouthOpenY",
    name: "입 여닫기",
    groupId: "ParamGroup4",
    group: "입",
    description: "+로 입을 열기",
    min: 0,
    max: 1,
    value: 0,
    previewMin: 0,
    previewMax: 1,
  },
  {
    key: "bodyAngleX",
    id: "ParamBodyAngleX",
    name: "몸의 회전 X",
    groupId: "ParamGroup5",
    group: "몸",
    description: "+로 화면의 오른쪽을 향하기",
    min: -10,
    max: 10,
    value: 0,
    previewMin: -10,
    previewMax: 10,
  },
  {
    key: "bodyAngleY",
    id: "ParamBodyAngleY",
    name: "몸의 회전 Y",
    groupId: "ParamGroup5",
    group: "몸",
    description: "+로 화면 위로 움직임",
    min: -10,
    max: 10,
    value: 0,
    previewMin: -10,
    previewMax: 10,
  },
  {
    key: "bodyAngleZ",
    id: "ParamBodyAngleZ",
    name: "몸의 회전 Z",
    groupId: "ParamGroup5",
    group: "몸",
    description: "+로 화면의 오른쪽으로 기울이기",
    min: -10,
    max: 10,
    value: 0,
    previewMin: -10,
    previewMax: 10,
  },
  {
    key: "breath",
    id: "ParamBreath",
    name: "호흡",
    groupId: "ParamGroup5",
    group: "몸",
    description: "+로 숨 들이마시기",
    min: 0,
    max: 1,
    value: 0,
    previewMin: 0,
    previewMax: 1,
  },
  {
    key: "hair",
    id: "ParamHairFront",
    name: "앞머리 흔들림",
    groupId: "ParamGroup6",
    group: "물리",
    description: "앞머리 좌우 물리 흔들림",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
  {
    key: "hairSide",
    id: "ParamHairSide",
    name: "옆머리 흔들림",
    groupId: "ParamGroup6",
    group: "물리",
    description: "옆머리 좌우 물리 흔들림",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
  {
    key: "hairBack",
    id: "ParamHairBack",
    name: "뒷머리 흔들림",
    groupId: "ParamGroup6",
    group: "물리",
    description: "뒷머리 좌우 물리 흔들림",
    min: -1,
    max: 1,
    value: 0,
    previewMin: -1,
    previewMax: 1,
  },
] as const;

export const HAIR_ROLE_META: Record<
  HairRole,
  { name: string; group: string; color: string; defaultOrder: number }
> = {
  hair_back: {
    name: "뒷머리",
    group: "뒷머리",
    color: "#334155",
    defaultOrder: 0,
  },
  hair_side_R: {
    name: "옆머리 R",
    group: "앞옆머리",
    color: "#0284c7",
    defaultOrder: 15,
  },
  hair_side_L: {
    name: "옆머리 L",
    group: "앞옆머리",
    color: "#0ea5e9",
    defaultOrder: 16,
  },
  hair_front: {
    name: "앞머리",
    group: "앞옆머리",
    color: "#1e40af",
    defaultOrder: 17,
  },
};

export function inferPartRole(id: string, explicitRole?: PartRole): PartRole {
  if (explicitRole) return explicitRole;
  if (id.startsWith("hair_side_L")) return "hair_side_L";
  if (id.startsWith("hair_side_R")) return "hair_side_R";
  if (id.startsWith("hair_side")) return "hair_side_L";
  if (id.startsWith("hair_back")) return "hair_back";
  if (id.startsWith("hair")) return "hair_front";
  if (id === "hat" || id.startsWith("hat_")) return "hat";
  if (id.startsWith("acc_") || id.startsWith("accessory")) return "accessory";
  if (id === "clothes_hands" || id === "hands" || id.startsWith("hands_")) return "hands";
  if (id === "body") return "body";
  if (id === "clothes" || id.startsWith("clothes_")) return "clothes";
  if (id === "neck") return "neck";
  if (id === "face" || id === "face_detail") return "face";
  if (id.startsWith("eye")) return "eye";
  if (id.startsWith("brow")) return "brow";
  if (id.startsWith("mouth")) return "mouth";
  return "other";
}

export function isHairPart(partOrId: string | Pick<Part, "id" | "role">): boolean {
  const id = typeof partOrId === "string" ? partOrId : partOrId.id;
  const role = typeof partOrId === "string" ? undefined : partOrId.role;
  const inferred = inferPartRole(id, role);
  return (
    inferred === "hair_front" ||
    inferred === "hair_side_L" ||
    inferred === "hair_side_R" ||
    inferred === "hair_back" ||
    id.startsWith("hair")
  );
}

export function isHairStageLayer(id: string): boolean {
  return (
    id.startsWith("hair_") ||
    id === "hat" ||
    id === "acc_hairpin" ||
    id === "acc_ribbon"
  );
}

export function partPipelineStage(
  id: string,
): Exclude<PipelineStage, "full"> {
  if (isHairStageLayer(id)) return "hair_overlay";
  if (
    id === "clothes" ||
    id.startsWith("clothes_") ||
    id === "hands" ||
    id.startsWith("hands_") ||
    id === "acc_glasses"
  ) {
    return "clothes_overlay";
  }
  if (
    id === "body" ||
    id === "neck" ||
    id === "face" ||
    id === "eyeR_white" ||
    id === "eyeL_white" ||
    id === "mouth_inner"
  ) {
    return "base_silhouette";
  }
  return "features_overlay";
}

export function stageIncludesLayer(
  id: string,
  stage: PipelineStage | "base_body",
): boolean {
  if (stage === "full") return true;
  if (stage === "base_body") return !isHairStageLayer(id);
  return partPipelineStage(id) === stage;
}

const WEARABLE_ID_META: Record<
  string,
  { name: string; group: string; color: string; role: PartRole }
> = {
  hat: { name: "모자 · 후드", group: "모자 · 소품", color: "#0ea5e9", role: "hat" },
  acc_glasses: { name: "안경", group: "안경 · 소품", color: "#38bdf8", role: "accessory" },
  acc_hairpin: { name: "머리핀 · 클립", group: "모자 · 소품", color: "#c084fc", role: "accessory" },
  acc_ribbon: { name: "리본 · 장식", group: "모자 · 소품", color: "#f472b6", role: "accessory" },
  clothes_outer: { name: "겉옷 · 아우터", group: "몸", color: "#4f46e5", role: "clothes" },
  clothes_inner: { name: "이너 의상", group: "몸", color: "#6366f1", role: "clothes" },
  clothes_collar: { name: "카라 · 넥라인", group: "몸", color: "#818cf8", role: "clothes" },
  clothes_acc: { name: "의상 장식 · 타이", group: "몸", color: "#ec4899", role: "clothes" },
  clothes_hands: { name: "손 · 소매끝", group: "손", color: "#f59e0b", role: "hands" },
};

export function describeDynamicPart(id: string): {
  name: string;
  group: string;
  color: string;
  role: PartRole;
} {
  if (id === "face_detail") {
    return {
      name: "얼굴 디테일 (볼터치·코)",
      group: "얼굴",
      color: "#fb7185",
      role: "face",
    };
  }
  if (id in WEARABLE_ID_META) {
    return WEARABLE_ID_META[id];
  }
  const role = inferPartRole(id);
  if (role in HAIR_ROLE_META) {
    const hairRole = role as HairRole;
    const meta = HAIR_ROLE_META[hairRole];
    const match = /_(\d+)$/.exec(id);
    const suffix = match ? ` ${match[1]}` : "";
    return {
      name: `${meta.name}${suffix}`,
      group: meta.group,
      color: meta.color,
      role: hairRole,
    };
  }
  if (role === "hat") {
    const match = /_(\d+)$/.exec(id);
    const suffix = match ? ` ${match[1]}` : "";
    return {
      name: `모자 · 후드${suffix}`,
      group: "모자 · 소품",
      color: "#0ea5e9",
      role: "hat",
    };
  }
  if (role === "accessory") {
    const match = /_(\d+)$/.exec(id);
    const suffix = match ? ` ${match[1]}` : "";
    return {
      name: `악세사리${suffix}`,
      group: "모자 · 소품",
      color: "#c084fc",
      role: "accessory",
    };
  }
  if (role === "hands") {
    const match = /_(\d+)$/.exec(id);
    const suffix = match ? ` ${match[1]}` : "";
    return {
      name: `손 · 소매끝${suffix}`,
      group: "손",
      color: "#f59e0b",
      role: "hands",
    };
  }
  if (role === "clothes") {
    const match = /_(\d+)$/.exec(id);
    const suffix = match ? ` ${match[1]}` : "";
    return {
      name: `의상${suffix}`,
      group: "몸",
      color: "#5872d8",
      role: "clothes",
    };
  }
  return {
    name: id,
    group: "기타",
    color: "#64748b",
    role,
  };
}

export const DEFAULT_EYE_HANDLES: ShapeHandleValues = {
  width: 1,
  open: 1,
  roundness: 0.55,
  corner: 0,
  upperCurve: 0.35,
  lowerCurve: 0.15,
  interiorVisibility: 1,
  offsetX: 0,
  offsetY: 0,
};

export const DEFAULT_MOUTH_HANDLES: ShapeHandleValues = {
  width: 1,
  open: 0,
  roundness: 0.25,
  corner: 0,
  upperCurve: 0.05,
  lowerCurve: 0,
  interiorVisibility: 1,
  offsetX: 0,
  offsetY: 0,
};

export const DEFAULT_RIG_SAFE_BOUNDS: Record<keyof Parameters, RegionHandleRange> = {
  angleX: { min: -25, max: 25 },
  angleY: { min: -25, max: 25 },
  angleZ: { min: -25, max: 25 },
  eyeL: { min: 0, max: 1 },
  eyeLSmile: { min: 0, max: 1 },
  eyeR: { min: 0, max: 1 },
  eyeRSmile: { min: 0, max: 1 },
  eyeBallX: { min: -1, max: 1 },
  eyeBallY: { min: -1, max: 1 },
  browLY: { min: -1, max: 1 },
  browLForm: { min: -1, max: 1 },
  browRY: { min: -1, max: 1 },
  browRForm: { min: -1, max: 1 },
  mouthForm: { min: -1, max: 1 },
  mouth: { min: 0, max: 1 },
  bodyAngleX: { min: -10, max: 10 },
  bodyAngleY: { min: -10, max: 10 },
  bodyAngleZ: { min: -10, max: 10 },
  breath: { min: 0, max: 1 },
  hair: { min: -1, max: 1 },
  hairSide: { min: -1, max: 1 },
  hairBack: { min: -1, max: 1 },
};

function makeEyeShapePresets(
  region: "eye_L" | "eye_R",
  proportionMode: RigProportionMode,
): ShapePreset[] {
  const isLeft = region === "eye_L";
  const openKey = isLeft ? "eyeL" : "eyeR";
  const smileKey = isLeft ? "eyeLSmile" : "eyeRSmile";
  const baseRound = proportionMode === "sd" ? 0.72 : 0.55;
  return [
    {
      id: `${region}_original`,
      region,
      label: "원본",
      kind: "original",
      recipeVersion: 1,
      builtIn: true,
      mirrorLinked: true,
      handles: { ...DEFAULT_EYE_HANDLES, roundness: baseRound },
      parameterOverrides: {},
    },
    {
      id: `${region}_half_closed`,
      region,
      label: "반감김",
      kind: "half_closed",
      recipeVersion: 1,
      builtIn: true,
      mirrorLinked: true,
      handles: {
        ...DEFAULT_EYE_HANDLES,
        open: 0.45,
        roundness: baseRound * 0.85,
        upperCurve: 0.1,
        lowerCurve: 0.1,
        interiorVisibility: 0.85,
      },
      parameterOverrides: { [openKey]: 0.45, [smileKey]: 0 },
    },
    {
      id: `${region}_closed`,
      region,
      label: "감김",
      kind: "closed",
      recipeVersion: 1,
      builtIn: true,
      mirrorLinked: true,
      handles: {
        ...DEFAULT_EYE_HANDLES,
        open: 0,
        roundness: 0.25,
        upperCurve: 0,
        lowerCurve: 0,
        interiorVisibility: 0,
      },
      parameterOverrides: { [openKey]: 0, [smileKey]: 0 },
    },
    {
      id: `${region}_smiling`,
      region,
      label: "웃는 눈",
      kind: "smiling",
      recipeVersion: 1,
      builtIn: true,
      mirrorLinked: true,
      handles: {
        ...DEFAULT_EYE_HANDLES,
        open: 0,
        roundness: 0.65,
        corner: 0.28,
        upperCurve: 0.9,
        lowerCurve: 0.35,
        interiorVisibility: 0,
      },
      parameterOverrides: { [openKey]: 0, [smileKey]: 1 },
    },
  ];
}

export function createDefaultShapePresets(
  proportionMode: RigProportionMode = "ld",
): ShapePreset[] {
  const narrowScale = proportionMode === "sd" ? 0.92 : 1;
  return [
    ...makeEyeShapePresets("eye_L", proportionMode),
    ...makeEyeShapePresets("eye_R", proportionMode),
    {
      id: "mouth_original",
      region: "mouth",
      label: "원본",
      kind: "original",
      recipeVersion: 1,
      builtIn: true,
      handles: { ...DEFAULT_MOUTH_HANDLES },
      parameterOverrides: {},
    },
    {
      id: "mouth_closed",
      region: "mouth",
      label: "다문 입",
      kind: "closed",
      recipeVersion: 1,
      builtIn: true,
      handles: {
        ...DEFAULT_MOUTH_HANDLES,
        open: 0,
        roundness: 0.15,
        upperCurve: 0.08,
        interiorVisibility: 0,
      },
      parameterOverrides: { mouth: 0, mouthForm: 0.1 },
    },
    {
      id: "mouth_a",
      region: "mouth",
      label: "아",
      kind: "a",
      recipeVersion: 1,
      builtIn: true,
      handles: {
        ...DEFAULT_MOUTH_HANDLES,
        width: 0.92 * narrowScale,
        open: 0.78,
        roundness: 0.72,
        upperCurve: 0.15,
        lowerCurve: -0.55,
        interiorVisibility: 0.9,
      },
      parameterOverrides: { mouth: 0.78, mouthForm: 0.25 },
    },
    {
      id: "mouth_i",
      region: "mouth",
      label: "이",
      kind: "i",
      recipeVersion: 1,
      builtIn: true,
      handles: {
        ...DEFAULT_MOUTH_HANDLES,
        width: 1.22,
        open: 0.34,
        roundness: 0.28,
        corner: 0.45,
        upperCurve: 0.2,
        lowerCurve: -0.15,
        interiorVisibility: 0.95,
      },
      parameterOverrides: { mouth: 0.34, mouthForm: 0.68 },
    },
    {
      id: "mouth_u",
      region: "mouth",
      label: "우",
      kind: "u",
      recipeVersion: 1,
      builtIn: true,
      handles: {
        ...DEFAULT_MOUTH_HANDLES,
        width: proportionMode === "sd" ? 0.56 : 0.62,
        open: 0.46,
        roundness: 0.95,
        corner: -0.15,
        upperCurve: 0.35,
        lowerCurve: -0.35,
        interiorVisibility: 0.55,
      },
      parameterOverrides: { mouth: 0.46, mouthForm: -0.25 },
    },
    {
      id: "mouth_e",
      region: "mouth",
      label: "에",
      kind: "e",
      recipeVersion: 1,
      builtIn: true,
      handles: {
        ...DEFAULT_MOUTH_HANDLES,
        width: 1.12,
        open: 0.52,
        roundness: 0.42,
        corner: 0.2,
        upperCurve: 0.15,
        lowerCurve: -0.3,
        interiorVisibility: 0.85,
      },
      parameterOverrides: { mouth: 0.52, mouthForm: 0.32 },
    },
    {
      id: "mouth_o",
      region: "mouth",
      label: "오",
      kind: "o",
      recipeVersion: 1,
      builtIn: true,
      handles: {
        ...DEFAULT_MOUTH_HANDLES,
        width: proportionMode === "sd" ? 0.68 : 0.74,
        open: 0.86,
        roundness: 0.96,
        upperCurve: 0.4,
        lowerCurve: -0.65,
        interiorVisibility: 0.88,
      },
      parameterOverrides: { mouth: 0.86, mouthForm: -0.1 },
    },
  ];
}

export function createDefaultExpressionPresets(): ExpressionPreset[] {
  return [
    {
      id: "neutral",
      name: "기본",
      builtIn: true,
      shapeIds: {
        eye_L: "eye_L_original",
        eye_R: "eye_R_original",
        mouth: "mouth_original",
      },
      parameters: {},
      strength: 1,
      transitionMs: 160,
      shortcutKey: "1",
    },
    {
      id: "smile",
      name: "미소",
      builtIn: true,
      shapeIds: {
        eye_L: "eye_L_smiling",
        eye_R: "eye_R_smiling",
        mouth: "mouth_closed",
      },
      parameters: {
        eyeL: 0,
        eyeR: 0,
        eyeLSmile: 0.9,
        eyeRSmile: 0.9,
        browLY: 0.15,
        browRY: 0.15,
        mouthForm: 0.55,
        mouth: 0.08,
      },
      strength: 1,
      transitionMs: 180,
      shortcutKey: "2",
    },
    {
      id: "happy",
      name: "기쁨",
      builtIn: true,
      shapeIds: {
        eye_L: "eye_L_smiling",
        eye_R: "eye_R_smiling",
        mouth: "mouth_a",
      },
      parameters: {
        eyeL: 0,
        eyeR: 0,
        eyeLSmile: 1,
        eyeRSmile: 1,
        browLY: 0.3,
        browRY: 0.3,
        mouthForm: 0.75,
        mouth: 0.45,
      },
      strength: 1,
      transitionMs: 180,
      shortcutKey: "3",
    },
    {
      id: "angry",
      name: "화남",
      builtIn: true,
      shapeIds: {
        eye_L: "eye_L_half_closed",
        eye_R: "eye_R_half_closed",
        mouth: "mouth_e",
      },
      parameters: {
        eyeL: 0.85,
        eyeR: 0.85,
        browLY: -0.45,
        browRY: -0.45,
        browLForm: -0.85,
        browRForm: -0.85,
        mouthForm: -0.65,
        mouth: 0.28,
      },
      strength: 1,
      transitionMs: 160,
      shortcutKey: "4",
    },
    {
      id: "sad",
      name: "슬픔",
      builtIn: true,
      shapeIds: {
        eye_L: "eye_L_closed",
        eye_R: "eye_R_closed",
        mouth: "mouth_closed",
      },
      parameters: {
        eyeL: 0,
        eyeR: 0,
        eyeLSmile: 0,
        eyeRSmile: 0,
        browLY: 0.35,
        browRY: 0.35,
        browLForm: -0.65,
        browRForm: -0.65,
        mouthForm: -0.65,
        mouth: 0.02,
        angleY: -6,
      },
      handleTweaks: {
        eye_L: { upperCurve: -0.95 },
        eye_R: { upperCurve: -0.95 },
      },
      strength: 1,
      transitionMs: 220,
      shortcutKey: "5",
    },
    {
      id: "surprised",
      name: "놀람",
      builtIn: true,
      shapeIds: {
        eye_L: "eye_L_original",
        eye_R: "eye_R_original",
        mouth: "mouth_o",
      },
      parameters: {
        eyeL: 1,
        eyeR: 1,
        browLY: 0.65,
        browRY: 0.65,
        browLForm: 0.25,
        browRForm: 0.25,
        mouth: 0.75,
        mouthForm: -0.05,
      },
      strength: 1,
      transitionMs: 140,
      shortcutKey: "6",
    },
    {
      id: "wink",
      name: "왼윙크 (L)",
      builtIn: true,
      shapeIds: {
        eye_L: "eye_L_smiling",
        eye_R: "eye_R_original",
        mouth: "mouth_i",
      },
      parameters: {
        eyeL: 0,
        eyeLSmile: 1,
        eyeR: 1,
        eyeRSmile: 0.15,
        mouthForm: 0.4,
        mouth: 0.15,
        angleZ: 4,
      },
      strength: 1,
      transitionMs: 150,
      shortcutKey: "7",
    },
    {
      id: "wink_r",
      name: "오른윙크 (R)",
      builtIn: true,
      shapeIds: {
        eye_L: "eye_L_original",
        eye_R: "eye_R_smiling",
        mouth: "mouth_i",
      },
      parameters: {
        eyeL: 1,
        eyeLSmile: 0.15,
        eyeR: 0,
        eyeRSmile: 1,
        mouthForm: 0.4,
        mouth: 0.15,
        angleZ: -4,
      },
      strength: 1,
      transitionMs: 150,
      shortcutKey: "8",
    },
  ];
}

export function createDefaultRigProfile(
  proportionMode: RigProportionMode = "ld",
): RigProfile {
  return {
    proportionMode,
    defaultShapeIds: {
      eye_L: "eye_L_original",
      eye_R: "eye_R_original",
      mouth: "mouth_original",
    },
    neutralParameters: { ...REST },
    safeBounds: Object.fromEntries(
      Object.entries(DEFAULT_RIG_SAFE_BOUNDS).map(([k, r]) => [k, { ...r }]),
    ) as Record<keyof Parameters, RegionHandleRange>,
    motionScale: {
      head: 1,
      eyes: 1,
      mouth: 1,
      hair: 1,
      body: 1,
    },
    clippingTargets: {
      eyeL_iris: "eyeL_white",
      eyeR_iris: "eyeR_white",
      face_detail: "face",
    },
  };
}

export function createDefaultPlaybackSettings(
  expressions: ExpressionPreset[] = createDefaultExpressionPresets(),
): PlaybackSettings {
  const shortcuts: Record<string, string> = {};
  for (const exp of expressions) {
    if (exp.shortcutKey) shortcuts[exp.shortcutKey] = exp.id;
  }
  return {
    defaultExpressionId: expressions[0]?.id ?? "neutral",
    autoBlink: true,
    blinkIntervalSec: 3.6,
    autoBreath: true,
    autoIdle: true,
    autoMouthDemo: false,
    shortcuts,
  };
}

export function createDefaultProjectRigState(
  proportionMode: RigProportionMode = "ld",
) {
  const expressions = createDefaultExpressionPresets();
  return {
    rigProfile: createDefaultRigProfile(proportionMode),
    shapePresets: createDefaultShapePresets(proportionMode),
    expressions,
    activeExpressionId: expressions[0].id,
    playback: createDefaultPlaybackSettings(expressions),
  };
}

export function clampParameterValue(
  key: keyof Parameters,
  value: number,
  bounds?: Partial<Record<keyof Parameters, RegionHandleRange>>,
): number {
  const fallback = DEFAULT_RIG_SAFE_BOUNDS[key];
  const custom = bounds?.[key];
  const lo = Math.max(fallback.min, custom?.min ?? fallback.min);
  const hi = Math.min(fallback.max, custom?.max ?? fallback.max);
  if (!Number.isFinite(value)) return REST[key];
  return Math.max(Math.min(lo, hi), Math.min(Math.max(lo, hi), value));
}

export function clampParameters(
  params: Parameters,
  bounds?: Partial<Record<keyof Parameters, RegionHandleRange>>,
): Parameters {
  const next = { ...params };
  for (const key of Object.keys(REST) as (keyof Parameters)[]) {
    next[key] = clampParameterValue(key, params[key], bounds);
  }
  return next;
}
