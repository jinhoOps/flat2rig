import { canvas } from "../images";
import type { Mesh, Project, ShapeRegion } from "../types";
import type { EvaluateRigInput } from "./evaluate";
import {
  renderDeterministicSnapshot,
  type SubstituteLayerTexture,
} from "./render";

export type ThumbnailCrop = "face" | ShapeRegion;

export interface ThumbnailSpec {
  id: string;
  crop: ThumbnailCrop;
  input: EvaluateRigInput;
}

export function getThumbnailCropRect(
  project: Project,
  crop: ThumbnailCrop,
): [number, number, number, number] {
  const { landmarks: lm, width: w, height: h } = project;
  const eyeDist = Math.max(
    16,
    Math.hypot(lm.eyeL.x - lm.eyeR.x, lm.eyeL.y - lm.eyeR.y),
  );
  const clampBox = (cx: number, cy: number, half: number): [number, number, number, number] => {
    const size = Math.max(24, Math.min(Math.min(w, h), Math.round(half * 2)));
    const sx = Math.max(0, Math.min(w - size, Math.round(cx - size * 0.5)));
    const sy = Math.max(0, Math.min(h - size, Math.round(cy - size * 0.5)));
    return [sx, sy, size, size];
  };
  if (crop === "eye_L") return clampBox(lm.eyeL.x, lm.eyeL.y, eyeDist * 0.62);
  if (crop === "eye_R") return clampBox(lm.eyeR.x, lm.eyeR.y, eyeDist * 0.62);
  if (crop === "mouth") return clampBox(lm.mouth.x, lm.mouth.y, eyeDist * 0.66);
  const midX = (lm.eyeL.x + lm.eyeR.x) * 0.5;
  const midY = (lm.crown.y + lm.chin.y) * 0.5;
  const faceHalf = Math.max(eyeDist * 1.45, (lm.chin.y - lm.crown.y) * 0.72);
  return clampBox(midX, midY, faceHalf);
}

export function renderRigThumbnailDataUrl(
  spec: ThumbnailSpec,
  partTextures: Map<string, HTMLCanvasElement>,
  meshes: Map<string, Mesh>,
  substituteTextures?: Map<string, SubstituteLayerTexture>,
  thumbSize = 96,
): string {
  const snap = renderDeterministicSnapshot(
    spec.input,
    partTextures,
    meshes,
    substituteTextures,
  );
  const [sx, sy, sw, sh] = getThumbnailCropRect(spec.input.project, spec.crop);
  const out = canvas(thumbSize, thumbSize);
  const g = out.getContext("2d")!;
  g.drawImage(snap.canvas, sx, sy, sw, sh, 0, 0, thumbSize, thumbSize);
  return out.toDataURL("image/png");
}
