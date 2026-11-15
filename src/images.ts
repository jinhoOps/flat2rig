import type { Project } from "./types";
import { compositePartTexture, defaultLandmarks } from "./engine";
export function canvas(w: number, h: number) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}
export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () =>
      reject(
        new Error("이미지를 읽을 수 없습니다. PNG/JPEG 파일을 확인하세요."),
      );
    img.src = url;
  });
}
export function pixels(img: CanvasImageSource, w: number, h: number) {
  const c = canvas(w, h);
  c.getContext("2d")!.drawImage(img, 0, 0, w, h);
  return c.getContext("2d")!.getImageData(0, 0, w, h).data;
}
export function pixelsCanvas(data: Uint8ClampedArray, w: number, h: number) {
  const c = canvas(w, h);
  c.getContext("2d")!.putImageData(
    new ImageData(new Uint8ClampedArray(data), w, h),
    0,
    0,
  );
  return c;
}
export async function readImage(file: File): Promise<Project> {
  if (file.size > 60 * 1024 * 1024)
    throw new Error("입력 이미지의 최대 크기는 60MB입니다.");
  if (
    !/image\/(png|jpeg)/.test(file.type) &&
    !/\.(png|jpe?g)$/i.test(file.name)
  )
    throw new Error("PNG 또는 JPEG 이미지를 선택하세요.");
  const original = await new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
  const img = await loadImage(original);
  if (img.width * img.height > 40_000_000)
    throw new Error("입력은 4천만 픽셀 이하로 준비하세요.");
  const scale = Math.min(1, 2048 / Math.max(img.width, img.height)),
    w = Math.max(1, Math.round(img.width * scale)),
    h = Math.max(1, Math.round(img.height * scale));
  const c = canvas(w, h);
  c.getContext("2d")!.drawImage(img, 0, 0, w, h);
  return {
    schemaVersion: 2,
    name: file.name.replace(/\.[^.]+$/, ""),
    width: w,
    height: h,
    originalWidth: img.width,
    originalHeight: img.height,
    source: c.toDataURL("image/png"),
    original,
    workToOriginalTransform: {
      matrix2: [img.width / w, 0, 0, img.height / h],
      translation: [0, 0],
    },
    landmarks: defaultLandmarks(w, h),
    landmarkProvenance: {
      eyeL: "estimated",
      eyeR: "estimated",
      mouth: "estimated",
      chin: "estimated",
      crown: "estimated",
    },
    parts: [],
    revision: 0,
    provider: "manual",
    analysisStatus: "idle",
    createdAt: new Date().toISOString(),
  };
}
export async function partCanvases(project: Project) {
  const original = await loadImage(project.source),
    source = pixels(original, project.width, project.height);
  return Promise.all(
    project.parts.map(async (p) => {
      const tex = p.texture
        ? pixels(await loadImage(p.texture), project.width, project.height)
        : undefined;
      return pixelsCanvas(
        compositePartTexture(p, source, tex),
        project.width,
        project.height,
      );
    }),
  );
}
export function pngBytes(c: HTMLCanvasElement): Promise<Uint8Array> {
  return new Promise((resolve, reject) =>
    c.toBlob(
      async (b) =>
        b
          ? resolve(new Uint8Array(await b.arrayBuffer()))
          : reject(new Error("PNG encoding failed")),
      "image/png",
    ),
  );
}
