export type Point = { x: number; y: number };
export type LandmarkId = "eyeL" | "eyeR" | "mouth" | "chin" | "crown";
export type Landmarks = Record<LandmarkId, Point>;
export type LandmarkSource = "model" | "estimated" | "manual";
export type LandmarkProvenance = Record<LandmarkId, LandmarkSource>;
export type PartStatus = "draft" | "approved" | "missing";
export type PartProvenance = "model" | "prompted" | "geometry" | "manual";
export type FillKind = "none" | "neighbor" | "external" | "template";
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
  compositeOrder?: number;
  prompts?: PromptPoint[];
  confidence?: number;
  diagnostics?: string[];
}

export interface Project {
  schemaVersion: 2;
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
  styleMatch?: StyleMatchReport;
  styleSelection?: TemplateStyleSelection;
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
  if (id === "face") return "face";
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
