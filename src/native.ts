import { invoke, isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  PART_DEFS,
  type Landmarks,
  type Mesh,
  type Part,
  type Point,
} from "./types";
import { meshFor, expandFill, deform } from "./engine";
import { REST, PARAM_DEFS } from "./types";
export const desktop = isTauri();
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
}
export async function buildRig(
  parts: Part[],
  width: number,
  height: number,
  landmarks: Landmarks,
): Promise<RigPart[]> {
  if (desktop)
    return invoke("build_rig", {
      parts: parts.map((p) => ({ id: p.id, mask: base64(p.mask) })),
      width,
      height,
      landmarks,
    });
  await new Promise((resolve) => setTimeout(resolve, 20));
  return parts.map((part) => {
    const mesh = meshFor(part.mask, width, height);
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
    };
  });
}
export async function fillNative(
  data: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  radius: number,
) {
  if (!desktop) return expandFill(data, mask, width, height, radius);
  const result = await invoke<{ rgba: string; mask: string }>("fill_part", {
    rgba: base64(data),
    mask: base64(mask),
    width,
    height,
    radius,
  });
  return {
    data: new Uint8ClampedArray(unbase64(result.rgba)),
    mask: unbase64(result.mask),
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
