import { useEffect, useRef } from "react";

/** Thumbnail box (CSS px). The canvas keeps the document aspect inside it. */
export const THUMB_BOX = 30;

export function thumbSize(width: number, height: number, box = THUMB_BOX) {
  const s = box / Math.max(1, width, height);
  return {
    w: Math.max(1, Math.round(width * s)),
    h: Math.max(1, Math.round(height * s)),
  };
}

const maskCache = new WeakMap<Uint8Array, { key: string; data: ImageData }>();

/**
 * Photoshop-style mask thumbnail: white = inside, black = outside. Each thumbnail pixel takes the
 * max over a 3×3 sub-grid of its cell so thin parts (eyelids, brows) stay visible. Cached per mask
 * array, so unchanged parts cost nothing on re-render.
 */
function maskImage(mask: Uint8Array, width: number, height: number, tw: number, th: number) {
  const key = `${width}x${height}:${tw}x${th}`;
  const hit = maskCache.get(mask);
  if (hit && hit.key === key) return hit.data;
  const data = new ImageData(tw, th);
  const px = data.data;
  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      let v = 0;
      for (let sy = 0; sy < 3 && v < 255; sy++) {
        const y = Math.min(height - 1, Math.floor(((ty + (sy + 0.5) / 3) * height) / th));
        for (let sx = 0; sx < 3; sx++) {
          const x = Math.min(width - 1, Math.floor(((tx + (sx + 0.5) / 3) * width) / tw));
          const m = mask[y * width + x] ?? 0;
          if (m > v) v = m;
        }
      }
      const o = (ty * tw + tx) * 4;
      px[o] = px[o + 1] = px[o + 2] = v;
      px[o + 3] = 255;
    }
  }
  maskCache.set(mask, { key, data });
  return data;
}

export interface LayerThumbProps {
  width: number;
  height: number;
  /** Part texture (drawn over a checkerboard). */
  image?: CanvasImageSource | null;
  /** Visible `mask` or hidden `fillMask`. */
  mask?: Uint8Array | null;
  /** Changes whenever the drawn content may have changed (e.g. the part object). */
  version?: unknown;
}

export default function LayerThumb({ width, height, image, mask, version }: LayerThumbProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { w, h } = thumbSize(width, height);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    ctx.clearRect(0, 0, w, h);
    if (mask && mask.length >= width * height) {
      ctx.putImageData(maskImage(mask, width, height, w, h), 0, 0);
    } else if (image) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      try {
        ctx.drawImage(image, 0, 0, w, h);
      } catch {
        /* image not decoded yet; the next version change redraws */
      }
    }
  }, [image, mask, width, height, w, h, version]);
  return <canvas ref={ref} width={w} height={h} style={{ width: w, height: h }} />;
}
