export type Point = { x: number; y: number };
export type LandmarkId = "eyeL" | "eyeR" | "mouth" | "chin" | "crown";
export type Landmarks = Record<LandmarkId, Point>;
export type PartStatus = "draft" | "approved" | "missing";
export interface Part {
  id: string;
  name: string;
  group: string;
  color: string;
  visible: boolean;
  status: PartStatus;
  mask: Uint8Array;
  texture?: string;
  generated?: boolean;
}
export interface Project {
  schemaVersion: 1;
  name: string;
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
  source: string;
  original: string;
  landmarks: Landmarks;
  parts: Part[];
  revision: number;
  provider: string;
  modelHash?: string;
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
export type Tool = "select" | "add" | "erase" | "landmark" | "face";
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
