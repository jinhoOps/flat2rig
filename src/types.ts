export type Point = { x: number; y: number };
export type LandmarkId = "eyeL" | "eyeR" | "mouth" | "chin" | "crown";
export type Landmarks = Record<LandmarkId, Point>;
export type LandmarkSource = "model" | "estimated" | "manual";
export type LandmarkProvenance = Record<LandmarkId, LandmarkSource>;
export type PartStatus = "draft" | "approved" | "missing";
export type PartProvenance = "model" | "prompted" | "geometry" | "manual";
export type FillKind = "none" | "neighbor" | "external";
export type HairRole = "hair_front" | "hair_side_L" | "hair_side_R" | "hair_back";
export type PartRole =
  | HairRole
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
  eyeR: number;
  mouth: number;
  hair: number;
}
export const REST: Parameters = {
  angleX: 0,
  angleY: 0,
  angleZ: 0,
  eyeL: 1,
  eyeR: 1,
  mouth: 0,
  hair: 0,
};
export type Tool = "select" | "add" | "erase" | "prompt" | "landmark" | "face";
export const PARENT_PART_DEFS = [
  ["eyeR_all", "오른쪽 눈 전체", "Eyes R", "#74c7c0"],
  ["eyeL_all", "왼쪽 눈 전체", "Eyes L", "#86b7f5"],
  ["mouth_all", "입 전체", "Mouth", "#e5a0b6"],
] as const;
export const PARENT_TO_DETAIL_IDS: Record<string, readonly string[]> = {
  eyeR_all: ["eyeR_white", "eyeR_iris", "eyeR_lower", "eyeR_upper"],
  eyeL_all: ["eyeL_white", "eyeL_iris", "eyeL_lower", "eyeL_upper"],
  mouth_all: ["mouth_inner", "mouth_lower", "mouth_upper"],
};
export const PART_DEFS = [
  ["hair_back", "뒤머리", "Hair", "#8c83ff"],
  ["body", "몸통 · 의상", "Body", "#6ba5ff"],
  ["neck", "목", "Body", "#f3b48c"],
  ["face", "얼굴 베이스", "Face", "#ecaa9e"],
  ["eyeR_white", "오른쪽 흰자", "Eyes R", "#89d4cc"],
  ["eyeR_iris", "오른쪽 홍채 · 동공", "Eyes R", "#5ab9bf"],
  ["eyeR_lower", "오른쪽 아래 눈꺼풀", "Eyes R", "#6bb3ac"],
  ["eyeR_upper", "오른쪽 위 눈꺼풀", "Eyes R", "#a1d8d1"],
  ["eyeL_white", "왼쪽 흰자", "Eyes L", "#9bc4ff"],
  ["eyeL_iris", "왼쪽 홍채 · 동공", "Eyes L", "#76a9ed"],
  ["eyeL_lower", "왼쪽 아래 눈꺼풀", "Eyes L", "#7fbae5"],
  ["eyeL_upper", "왼쪽 위 눈꺼풀", "Eyes L", "#b3d5ff"],
  ["browR", "오른쪽 눈썹", "Brows", "#d5b783"],
  ["browL", "왼쪽 눈썹", "Brows", "#d5b783"],
  ["mouth_inner", "입 안", "Mouth", "#d686a5"],
  ["mouth_lower", "아랫입술", "Mouth", "#e5a0b6"],
  ["mouth_upper", "윗입술", "Mouth", "#efb1c7"],
  ["hair_front", "앞머리", "Hair", "#b3a0f5"],
] as const;
export const PARAM_DEFS = [
  {
    key: "angleX",
    id: "ParamAngleX",
    name: "고개 좌우",
    min: -30,
    max: 30,
    value: 0,
    previewMin: -10,
    previewMax: 10,
  },
  {
    key: "angleY",
    id: "ParamAngleY",
    name: "고개 상하",
    min: -30,
    max: 30,
    value: 0,
    previewMin: -10,
    previewMax: 10,
  },
  {
    key: "angleZ",
    id: "ParamAngleZ",
    name: "고개 기울기",
    min: -30,
    max: 30,
    value: 0,
    previewMin: -15,
    previewMax: 15,
  },
  {
    key: "eyeL",
    id: "ParamEyeLOpen",
    name: "왼쪽 눈 개폐",
    min: 0,
    max: 1,
    value: 1,
    previewMin: 0,
    previewMax: 1,
  },
  {
    key: "eyeR",
    id: "ParamEyeROpen",
    name: "오른쪽 눈 개폐",
    min: 0,
    max: 1,
    value: 1,
    previewMin: 0,
    previewMax: 1,
  },
  {
    key: "mouth",
    id: "ParamMouthOpenY",
    name: "입 개폐",
    min: 0,
    max: 1,
    value: 0,
    previewMin: 0,
    previewMax: 1,
  },
  {
    key: "hair",
    id: "ParamHairFront",
    name: "머리카락 흔들림",
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
    group: "머리카락",
    color: "#334155",
    defaultOrder: 0,
  },
  hair_side_R: {
    name: "오른쪽 옆머리",
    group: "머리카락",
    color: "#0284c7",
    defaultOrder: 15,
  },
  hair_side_L: {
    name: "왼쪽 옆머리",
    group: "머리카락",
    color: "#0ea5e9",
    defaultOrder: 16,
  },
  hair_front: {
    name: "앞머리",
    group: "머리카락",
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

export function describeDynamicPart(id: string): {
  name: string;
  group: string;
  color: string;
  role: PartRole;
} {
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
  if (role === "clothes") {
    const match = /_(\d+)$/.exec(id);
    const suffix = match ? ` ${match[1]}` : "";
    return {
      name: `의상${suffix}`,
      group: "Body",
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
