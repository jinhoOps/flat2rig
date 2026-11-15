import { zlibSync } from "fflate";
import { buildHeadFrame, clamp, toHeadLocal, toHeadWorld } from "../engine";
import { canvas } from "../images";
import type {
  Part,
  Project,
  ShapeRegion,
  ShapeSubstituteAsset,
  ShapeSubstituteLayer,
} from "../types";

const PNG_SIGNATURE = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);

let crcTable: Uint32Array | null = null;
function getCrcTable(): Uint32Array {
  if (crcTable) return crcTable;
  const tbl = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    tbl[n] = c >>> 0;
  }
  crcTable = tbl;
  return tbl;
}

function crc32(bytes: Uint8Array): number {
  const tbl = getCrcTable();
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = tbl[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function makePngChunk(type: string, data: Uint8Array): Uint8Array {
  const chunk = new Uint8Array(12 + data.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, data.length, false);
  for (let i = 0; i < 4; i++) {
    chunk[4 + i] = type.charCodeAt(i);
  }
  chunk.set(data, 8);
  view.setUint32(8 + data.length, crc32(chunk.subarray(4, 8 + data.length)), false);
  return chunk;
}

/**
 * Synchronously encodes straight RGBA pixel buffer into a valid PNG byte stream.
 * Works identically in browser DOM and Node `@napi-rs/canvas` headless tests.
 */
export function encodeRgbaToPngBytes(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): Uint8Array {
  const stride = width * 4 + 1;
  const raw = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    raw[y * stride] = 0; // filter None
    const srcOff = y * width * 4;
    raw.set(rgba.subarray(srcOff, srcOff + width * 4), y * stride + 1);
  }
  const ihdr = new Uint8Array(13);
  const ihdrView = new DataView(ihdr.buffer);
  ihdrView.setUint32(0, width, false);
  ihdrView.setUint32(4, height, false);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const compressed = zlibSync(raw, { level: 6 });
  const ihdrChunk = makePngChunk("IHDR", ihdr);
  const idatChunk = makePngChunk("IDAT", compressed);
  const iendChunk = makePngChunk("IEND", new Uint8Array(0));

  const out = new Uint8Array(
    PNG_SIGNATURE.length +
      ihdrChunk.length +
      idatChunk.length +
      iendChunk.length,
  );
  let pos = 0;
  out.set(PNG_SIGNATURE, pos);
  pos += PNG_SIGNATURE.length;
  out.set(ihdrChunk, pos);
  pos += ihdrChunk.length;
  out.set(idatChunk, pos);
  pos += idatChunk.length;
  out.set(iendChunk, pos);
  return out;
}

function medianColor(samples: [number, number, number][], fallback: [number, number, number]): [number, number, number] {
  if (!samples.length) return fallback;
  const rs = samples.map((c) => c[0]).sort((a, b) => a - b);
  const gs = samples.map((c) => c[1]).sort((a, b) => a - b);
  const bs = samples.map((c) => c[2]).sort((a, b) => a - b);
  const mid = Math.floor(samples.length / 2);
  return [rs[mid], gs[mid], bs[mid]];
}

function rgbCss(c: [number, number, number], alpha = 1): string {
  return alpha >= 1
    ? `rgb(${c[0]}, ${c[1]}, ${c[2]})`
    : `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha.toFixed(3)})`;
}

function unionPartsMask(
  parts: Part[],
  targetIds: string[],
  w: number,
  h: number,
): { mask: Uint8Array; bounds: { left: number; top: number; right: number; bottom: number } | null } {
  const out = new Uint8Array(w * h);
  let left = w;
  let top = h;
  let right = -1;
  let bottom = -1;
  for (const p of parts) {
    if (!targetIds.includes(p.id) || p.status === "missing") continue;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const a = p.mask[y * w + x];
        if (a > 0) {
          out[y * w + x] = Math.max(out[y * w + x], a);
          if (x < left) left = x;
          if (x > right) right = x;
          if (y < top) top = y;
          if (y > bottom) bottom = y;
        }
      }
    }
  }
  return {
    mask: out,
    bounds: right >= left && bottom >= top ? { left, top, right, bottom } : null,
  };
}

function clampRect(
  left: number,
  top: number,
  width: number,
  height: number,
  maxW: number,
  maxH: number,
): { left: number; top: number; width: number; height: number } {
  const l = clamp(Math.floor(left), 0, Math.max(0, maxW - 1));
  const t = clamp(Math.floor(top), 0, Math.max(0, maxH - 1));
  const r = clamp(Math.ceil(left + width), l + 1, maxW);
  const b = clamp(Math.ceil(top + height), t + 1, maxH);
  return { left: l, top: t, width: r - l, height: b - t };
}

function canvasToLayer(
  partId: string,
  c: HTMLCanvasElement,
  bounds: { left: number; top: number; width: number; height: number },
  hidesPartIds?: string[],
): ShapeSubstituteLayer {
  const g = c.getContext("2d")!;
  const imgData = g.getImageData(0, 0, bounds.width, bounds.height).data;
  const mask = new Uint8Array(bounds.width * bounds.height);
  for (let i = 0; i < mask.length; i++) {
    mask[i] = imgData[i * 4 + 3];
  }
  const pngBytes = encodeRgbaToPngBytes(imgData, bounds.width, bounds.height);
  return {
    partId,
    blendMode: "normal",
    bounds,
    ...(hidesPartIds ? { hidesPartIds } : {}),
    mask,
    pngBytes,
    hasMask: true,
    hasTexturePng: true,
  };
}

function sampleRegionColors(
  project: Project,
  sourcePixels: Uint8ClampedArray,
  centerX: number,
  centerY: number,
  radius: number,
  inkPartIds: string[],
  excludePartIds: string[],
): {
  skin: [number, number, number];
  ink: [number, number, number];
  rose: [number, number, number];
} {
  const { width: w, height: h, parts } = project;
  const facePart = parts.find((p) => p.id === "face");
  const excludeSet = new Uint8Array(w * h);
  for (const p of parts) {
    if (
      excludePartIds.includes(p.id) ||
      p.id.startsWith("acc_") ||
      p.id.startsWith("hair")
    ) {
      for (let i = 0; i < w * h; i++) {
        if (p.mask[i] > 32) excludeSet[i] = 1;
      }
    }
  }

  const skinSamples: [number, number, number][] = [];
  const x0 = clamp(Math.floor(centerX - radius * 1.4), 0, w - 1);
  const x1 = clamp(Math.ceil(centerX + radius * 1.4), 0, w - 1);
  const y0 = clamp(Math.floor(centerY - radius * 1.2), 0, h - 1);
  const y1 = clamp(Math.ceil(centerY + radius * 1.2), 0, h - 1);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const idx = y * w + x;
      if (excludeSet[idx]) continue;
      if (facePart && facePart.mask[idx] < 180) continue;
      const a = sourcePixels[idx * 4 + 3];
      if (a < 200) continue;
      const r = sourcePixels[idx * 4];
      const g = sourcePixels[idx * 4 + 1];
      const b = sourcePixels[idx * 4 + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      const sat = Math.max(r, g, b) - Math.min(r, g, b);
      // Exclude dark lineart ink and flat sclera white
      if (luma < 125 || (luma > 246 && sat < 14)) continue;
      skinSamples.push([r, g, b]);
    }
  }

  const inkCandidates: { rgb: [number, number, number]; luma: number }[] = [];
  for (const p of parts) {
    if (!inkPartIds.includes(p.id)) continue;
    for (let i = 0; i < w * h; i++) {
      if (p.mask[i] < 120 || sourcePixels[i * 4 + 3] < 160) continue;
      const r = sourcePixels[i * 4];
      const g = sourcePixels[i * 4 + 1];
      const b = sourcePixels[i * 4 + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      inkCandidates.push({ rgb: [r, g, b], luma });
    }
  }
  const skin = medianColor(skinSamples, [248, 224, 212]);
  const skinLuma = skin[0] * 0.299 + skin[1] * 0.587 + skin[2] * 0.114;
  inkCandidates.sort((a, b) => a.luma - b.luma);
  const distinctInks = inkCandidates.filter((c) => c.luma < skinLuma - 28);
  const inkPool = distinctInks.length >= 3 ? distinctInks : inkCandidates;
  const darkFrac = distinctInks.length >= 3 ? 0.14 : 0.05;
  const darkCount = Math.max(1, Math.min(inkPool.length, Math.ceil(inkPool.length * darkFrac)));
  const darkInks = inkPool.slice(0, darkCount).map((c) => c.rgb);

  // Sample cheek/lip rose strictly from bright skin/blush pixels on face (never dark eye/lash ink)
  const roseSamples: [number, number, number][] = [];
  for (const px of skinSamples) {
    const luma = 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2];
    if (px[0] > px[1] + 10 && luma > 175) roseSamples.push(px);
  }
  const fallbackRose: [number, number, number] = [
    Math.min(255, Math.round(skin[0] * 0.96 + 10)),
    Math.max(110, Math.round(skin[1] * 0.72)),
    Math.max(115, Math.round(skin[2] * 0.74)),
  ];

  return {
    skin,
    ink: medianColor(darkInks, [56, 42, 54]),
    rose: medianColor(roseSamples, fallbackRose),
  };
}

const SECTORS = 16;

function buildPerimeterSkinPalette(
  project: Project,
  sourcePixels: Uint8ClampedArray,
  centerX: number,
  centerY: number,
  rx: number,
  ry: number,
  excludePartIds: string[],
  fallbackSkin: [number, number, number],
): [number, number, number][] {
  const { width: w, height: h, parts } = project;
  const facePart = parts.find((p) => p.id === "face");
  const excludeSet = new Uint8Array(w * h);
  for (const p of parts) {
    if (
      excludePartIds.includes(p.id) ||
      p.id.startsWith("acc_") ||
      p.id.startsWith("hair")
    ) {
      for (let i = 0; i < w * h; i++) {
        if (p.mask[i] > 32) excludeSet[i] = 1;
      }
    }
  }
  const fbLuma =
    0.299 * fallbackSkin[0] + 0.587 * fallbackSkin[1] + 0.114 * fallbackSkin[2];
  const buckets: [number, number, number][][] = Array.from(
    { length: SECTORS },
    () => [],
  );
  const x0 = clamp(Math.floor(centerX - rx * 1.45), 0, w - 1);
  const x1 = clamp(Math.ceil(centerX + rx * 1.45), 0, w - 1);
  const y0 = clamp(Math.floor(centerY - ry * 1.45), 0, h - 1);
  const y1 = clamp(Math.ceil(centerY + ry * 1.45), 0, h - 1);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const idx = y * w + x;
      if (excludeSet[idx]) continue;
      if (facePart && facePart.mask[idx] < 160) continue;
      const a = sourcePixels[idx * 4 + 3];
      if (a < 200) continue;
      const nx = (x - centerX) / Math.max(4, rx);
      const ny = (y - centerY) / Math.max(4, ry);
      const dist = Math.hypot(nx, ny);
      if (dist < 0.9 || dist > 1.45) continue;
      const r = sourcePixels[idx * 4];
      const g = sourcePixels[idx * 4 + 1];
      const b = sourcePixels[idx * 4 + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      const sat = Math.max(r, g, b) - Math.min(r, g, b);
      if (
        luma < Math.max(130, fbLuma - 30) ||
        b > r + 4 ||
        (luma > 248 && sat < 12)
      ) {
        continue;
      }
      const angle = Math.atan2(ny, nx);
      const sec = Math.floor(((angle + Math.PI) / (2 * Math.PI)) * SECTORS) % SECTORS;
      buckets[sec].push([r, g, b]);
    }
  }
  return buckets.map((b) => medianColor(b, fallbackSkin));
}

function sampleSectorSkin(
  palette: [number, number, number][],
  centerSkin: [number, number, number],
  nx: number,
  ny: number,
): [number, number, number] {
  const dist = clamp(Math.hypot(nx, ny), 0, 1);
  const angle = Math.atan2(ny, nx);
  const f = (((angle + Math.PI) / (2 * Math.PI)) * SECTORS) % SECTORS;
  const i0 = Math.floor(f) % SECTORS;
  const i1 = (i0 + 1) % SECTORS;
  const frac = f - Math.floor(f);
  const s0 = palette[i0];
  const s1 = palette[i1];
  const edgeR = s0[0] + (s1[0] - s0[0]) * frac;
  const edgeG = s0[1] + (s1[1] - s0[1]) * frac;
  const edgeB = s0[2] + (s1[2] - s0[2]) * frac;
  const t = clamp(dist * 0.9, 0, 1);
  return [
    Math.round(centerSkin[0] * (1 - t) + edgeR * t),
    Math.round(centerSkin[1] * (1 - t) + edgeG * t),
    Math.round(centerSkin[2] * (1 - t) + edgeB * t),
  ];
}

function buildMouthSubstituteAsset(
  project: Project,
  sourcePixels: Uint8ClampedArray,
  kind: "closed" | "a" | "i" | "u" | "e" | "o",
): ShapeSubstituteAsset {
  const isSd = project.rigProfile.proportionMode === "sd";
  const frame = buildHeadFrame(project.landmarks, project.width, project.height);
  const tiltRad = Math.atan2(frame.exY, frame.exX);
  const mouth = project.landmarks.mouth;
  const eyeDist = frame.eyeDist;
  const headH = frame.hh;
  const hidesPartIds = ["mouth_upper", "mouth_lower", "mouth_inner", "mouth_all"];

  const colors = sampleRegionColors(
    project,
    sourcePixels,
    mouth.x,
    mouth.y,
    Math.max(10, eyeDist * 0.28),
    hidesPartIds,
    [...hidesPartIds, "face_shadow"],
  );

  const union = unionPartsMask(
    project.parts,
    hidesPartIds,
    project.width,
    project.height,
  );
  const [mouthU, mouthV] = toHeadLocal(frame, mouth.x, mouth.y);
  const defaultRu = Math.max(5.5, eyeDist * (isSd ? 0.11 : 0.135));
  const defaultRv = Math.max(3.5, headH * (isSd ? 0.036 : 0.042));
  const pad = clamp(Math.round(headH * 0.022), 3, 7);

  // Sample perioral skin ring in tilted HeadFrame coordinates before scanning mouth ink
  const facePart = project.parts.find((p) => p.id === "face");
  let ringR = 0,
    ringG = 0,
    ringB = 0,
    ringCnt = 0;
  const scanWin = Math.ceil(eyeDist * 0.34);
  const ringX0 = clamp(Math.floor(mouth.x - scanWin), 0, project.width - 1);
  const ringX1 = clamp(Math.ceil(mouth.x + scanWin), 0, project.width - 1);
  const ringY0 = clamp(Math.floor(mouth.y - scanWin), 0, project.height - 1);
  const ringY1 = clamp(Math.ceil(mouth.y + scanWin), 0, project.height - 1);
  for (let gy = ringY0; gy <= ringY1; gy++) {
    for (let gx = ringX0; gx <= ringX1; gx++) {
      const idx = gy * project.width + gx;
      if (union.mask[idx] > 80 || sourcePixels[idx * 4 + 3] < 180) continue;
      if (facePart && facePart.mask[idx] <= 128) continue;
      const [lu, lv] = toHeadLocal(frame, gx + 0.5, gy + 0.5);
      const d = Math.hypot(lu - mouthU, lv - mouthV);
      if (d < eyeDist * 0.19 || d > eyeDist * 0.33) continue;
      const k = idx * 4;
      ringR += sourcePixels[k];
      ringG += sourcePixels[k + 1];
      ringB += sourcePixels[k + 2];
      ringCnt++;
    }
  }
  const refSkin: [number, number, number] =
    ringCnt >= 8 ? [ringR / ringCnt, ringG / ringCnt, ringB / ringCnt] : colors.skin;
  const refSkinLuma = 0.299 * refSkin[0] + 0.587 * refSkin[1] + 0.114 * refSkin[2];

  // Scan strictly within tight perioral HeadFrame box for real painted mouth/lip pixels
  let lipMinU = Infinity,
    lipMaxU = -Infinity,
    lipMinV = Infinity,
    lipMaxV = -Infinity;
  let lipCount = 0;
  const innerLimitU = eyeDist * 0.175;
  const innerLimitV = headH * 0.052;
  const sx0 = clamp(Math.floor(mouth.x - eyeDist * 0.24), 0, project.width - 1);
  const sx1 = clamp(Math.ceil(mouth.x + eyeDist * 0.24), 0, project.width - 1);
  const sy0 = clamp(Math.floor(mouth.y - eyeDist * 0.24), 0, project.height - 1);
  const sy1 = clamp(Math.ceil(mouth.y + eyeDist * 0.24), 0, project.height - 1);
  for (let gy = sy0; gy <= sy1; gy++) {
    for (let gx = sx0; gx <= sx1; gx++) {
      const idx = gy * project.width + gx;
      const inFaceOrMouth =
        union.mask[idx] > 80 || (facePart ? facePart.mask[idx] > 160 : true);
      if (!inFaceOrMouth || sourcePixels[idx * 4 + 3] < 180) continue;
      const [lu, lv] = toHeadLocal(frame, gx + 0.5, gy + 0.5);
      if (Math.abs(lu - mouthU) > innerLimitU || Math.abs(lv - mouthV) > innerLimitV) {
        continue;
      }
      const k = idx * 4;
      const r = sourcePixels[k];
      const g = sourcePixels[k + 1];
      const b = sourcePixels[k + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      const colorDist = Math.hypot(r - refSkin[0], g - refSkin[1], b - refSkin[2]);
      if (luma < refSkinLuma - 28 || colorDist > 34 || union.mask[idx] > 160) {
        if (lu < lipMinU) lipMinU = lu;
        if (lu > lipMaxU) lipMaxU = lu;
        if (lv < lipMinV) lipMinV = lv;
        if (lv > lipMaxV) lipMaxV = lv;
        lipCount++;
      }
    }
  }
  const hasLipBlob = lipCount >= 6 && lipMaxU >= lipMinU && lipMaxV >= lipMinV;
  const blobMidU = hasLipBlob ? (lipMinU + lipMaxU) * 0.5 : mouthU;
  const blobMidV = hasLipBlob ? (lipMinV + lipMaxV) * 0.5 : mouthV;
  const drawU =
    hasLipBlob && Math.abs(blobMidU - mouthU) <= eyeDist * 0.12 ? blobMidU : mouthU;
  const drawV =
    hasLipBlob && Math.abs(blobMidV - mouthV) <= headH * 0.035 ? blobMidV : mouthV;
  const coverU = (mouthU + drawU) * 0.5;
  const coverV = (mouthV + drawV) * 0.5;

  const skinRu = hasLipBlob
    ? clamp((lipMaxU - lipMinU) * 0.5 + 3.5, defaultRu * 0.85, eyeDist * 0.18)
    : defaultRu;
  const skinRv = hasLipBlob
    ? clamp((lipMaxV - lipMinV) * 0.5 + 3.5, defaultRv * 0.85, headH * 0.058)
    : defaultRv;

  // Also cap union mask contribution strictly around cover box so oversized draft parts don't erase blush
  const [drawX, drawY] = toHeadWorld(frame, drawU, drawV);
  const [coverX, coverY] = toHeadWorld(frame, coverU, coverV);
  const coverSpan = Math.hypot(skinRu, skinRv) + pad;
  const rxOpen = Math.max(4, eyeDist * (isSd ? 0.125 : 0.155));
  const ryOpen = Math.max(4, headH * (isSd ? 0.048 : 0.056));
  const drawSpan = Math.max(coverSpan, Math.hypot(rxOpen, ryOpen) + pad);

  let boxLeft = coverX - drawSpan;
  let boxTop = coverY - drawSpan;
  let boxRight = coverX + drawSpan;
  let boxBot = coverY + drawSpan;
  if (union.bounds) {
    const maxReach = eyeDist * 0.21;
    if (union.bounds.right - union.bounds.left <= maxReach * 2) {
      boxLeft = Math.min(boxLeft, union.bounds.left - pad);
      boxTop = Math.min(boxTop, union.bounds.top - pad);
      boxRight = Math.max(boxRight, union.bounds.right + 1 + pad);
      boxBot = Math.max(boxBot, union.bounds.bottom + 1 + pad);
    }
  }
  const bounds = clampRect(
    boxLeft,
    boxTop,
    boxRight - boxLeft,
    boxBot - boxTop,
    project.width,
    project.height,
  );
  const cx = drawX - bounds.left;
  const cy = drawY - bounds.top;

  // Layer 1: Compact HeadFrame-tilted feathered skin cover hiding ONLY original painted mouth
  const perimeter = buildPerimeterSkinPalette(
    project,
    sourcePixels,
    coverX,
    coverY,
    skinRu,
    skinRv,
    [...hidesPartIds, "face_shadow"],
    refSkin,
  );
  const skinCanvas = canvas(bounds.width, bounds.height);
  const sg = skinCanvas.getContext("2d")!;
  const skinImg = sg.createImageData(bounds.width, bounds.height);
  for (let y = 0; y < bounds.height; y++) {
    for (let x = 0; x < bounds.width; x++) {
      const gx = bounds.left + x;
      const gy = bounds.top + y;
      const [pu, pv] = toHeadLocal(frame, gx + 0.5, gy + 0.5);
      const dx = (pu - coverU) / Math.max(3, skinRu);
      const dy = (pv - coverV) / Math.max(3, skinRv);
      const r2 = dx * dx + dy * dy;
      const inTightUnion =
        r2 < 1.35 ? union.mask[gy * project.width + gx] : 0;
      const softA =
        r2 >= 1 ? 0 : Math.round(clamp((1 - r2) / 0.42, 0, 1) * 255);
      const a = Math.max(inTightUnion, softA);
      if (!a) continue;
      const rgb = sampleSectorSkin(perimeter, refSkin, dx, dy);
      const di = (y * bounds.width + x) * 4;
      skinImg.data[di] = rgb[0];
      skinImg.data[di + 1] = rgb[1];
      skinImg.data[di + 2] = rgb[2];
      skinImg.data[di + 3] = a;
    }
  }
  sg.putImageData(skinImg, 0, 0);

  const lipCanvas = canvas(bounds.width, bounds.height);
  const lg = lipCanvas.getContext("2d")!;
  const strokeW = clamp(headH * 0.0085, 1.15, 3.4);

  if (kind === "closed") {
    const detHalfW = hasLipBlob ? (lipMaxU - lipMinU) * 0.52 : eyeDist * 0.1;
    const span = clamp(detHalfW, eyeDist * 0.072, eyeDist * (isSd ? 0.105 : 0.13));
    lg.save();
    lg.translate(cx, cy);
    lg.rotate(tiltRad);
    lg.strokeStyle = rgbCss(colors.ink);
    lg.lineWidth = strokeW;
    lg.lineCap = "round";
    lg.beginPath();
    lg.moveTo(-span, -strokeW * 0.28);
    lg.quadraticCurveTo(0, strokeW * 0.95, span, -strokeW * 0.28);
    lg.stroke();
    lg.restore();
    return {
      assetId: `mouth_closed_v1`,
      region: "mouth",
      recipeVersion: 2,
      hidesPartIds,
      layers: [
        canvasToLayer("mouth_sub_skin", skinCanvas, bounds, hidesPartIds),
        canvasToLayer("mouth_sub_lip", lipCanvas, bounds),
      ],
    };
  }

  // Layer 2: Compact HeadFrame-aligned illustration mouth cavity + soft teeth + dome tongue
  const innerCanvas = canvas(bounds.width, bounds.height);
  const ig = innerCanvas.getContext("2d")!;
  const sizeMul = hasLipBlob
    ? clamp((lipMaxU - lipMinU) / Math.max(8, eyeDist * 0.18), 0.78, 1.12)
    : 1;
  const rxMap = {
    a: eyeDist * (isSd ? 0.105 : 0.13) * sizeMul,
    i: eyeDist * (isSd ? 0.13 : 0.16) * sizeMul,
    u: eyeDist * (isSd ? 0.052 : 0.068) * sizeMul,
    e: eyeDist * (isSd ? 0.118 : 0.145) * sizeMul,
    o: eyeDist * (isSd ? 0.074 : 0.092) * sizeMul,
  };
  const ryTopMap = {
    a: headH * 0.017,
    i: headH * 0.015,
    u: headH * 0.019,
    e: headH * 0.015,
    o: headH * 0.03,
  };
  const ryBotMap = {
    a: headH * (isSd ? 0.044 : 0.05),
    i: headH * (isSd ? 0.024 : 0.028),
    u: headH * (isSd ? 0.028 : 0.032),
    e: headH * (isSd ? 0.034 : 0.039),
    o: headH * (isSd ? 0.046 : 0.052),
  };
  const rx = Math.max(3, rxMap[kind]);
  const topY = -Math.max(1.2, ryTopMap[kind]);
  const botY = Math.max(2.0, ryBotMap[kind]);

  const traceMouthCavity = (g: CanvasRenderingContext2D) => {
    g.beginPath();
    if (kind === "u" || kind === "o") {
      const midY = (topY + botY) * 0.5;
      const rY = (botY - topY) * 0.52;
      g.ellipse(0, midY, rx, rY, 0, 0, Math.PI * 2);
    } else if (kind === "i") {
      g.moveTo(-rx, topY + 0.4);
      g.quadraticCurveTo(0, topY + 1.8, rx, topY + 0.4);
      g.quadraticCurveTo(0, botY + 1.2, -rx, topY + 0.4);
    } else {
      const cornerY = topY + (botY - topY) * 0.18;
      g.moveTo(-rx, cornerY);
      g.quadraticCurveTo(0, topY - (botY - topY) * 0.14, rx, cornerY);
      g.quadraticCurveTo(0, botY + (botY - topY) * 0.22, -rx, cornerY);
    }
    g.closePath();
  };

  ig.save();
  ig.translate(cx, cy);
  ig.rotate(tiltRad);
  traceMouthCavity(ig);
  const grad = ig.createLinearGradient(0, topY, 0, botY);
  grad.addColorStop(0, "#4f1c28");
  grad.addColorStop(1, "#7b2d3e");
  ig.fillStyle = grad;
  ig.fill();
  ig.clip();

  // Tongue dome inside lower cavity
  ig.fillStyle = rgbCss(
    [
      Math.round((colors.rose[0] + 235) * 0.5),
      Math.round((colors.rose[1] + 132) * 0.5),
      Math.round((colors.rose[2] + 144) * 0.5),
    ],
  );
  ig.beginPath();
  ig.ellipse(0, botY + (botY - topY) * 0.08, rx * 0.74, (botY - topY) * 0.5, 0, Math.PI, Math.PI * 2);
  ig.fill();

  // Subtle curved upper teeth highlight
  if (kind === "i" || kind === "a" || kind === "e") {
    ig.fillStyle = "rgba(255, 248, 246, 0.92)";
    const teethH = (botY - topY) * (kind === "i" ? 0.46 : 0.24);
    ig.beginPath();
    ig.ellipse(0, topY, rx * 0.88, teethH, 0, 0, Math.PI);
    ig.fill();
  }
  ig.restore();

  // Layer 3: Rotated anime lip ink outline + subtle lower lip shadow hint
  lg.save();
  lg.translate(cx, cy);
  lg.rotate(tiltRad);
  lg.strokeStyle = rgbCss(colors.ink);
  lg.lineWidth = strokeW;
  lg.lineJoin = "round";
  lg.lineCap = "round";
  traceMouthCavity(lg);
  lg.stroke();
  lg.strokeStyle = rgbCss(colors.ink, 0.28);
  lg.lineWidth = Math.max(0.8, strokeW * 0.55);
  lg.beginPath();
  lg.moveTo(-rx * 0.35, botY + strokeW * 1.4);
  lg.quadraticCurveTo(0, botY + strokeW * 1.9, rx * 0.35, botY + strokeW * 1.4);
  lg.stroke();
  lg.restore();

  return {
    assetId: `mouth_${kind}_v1`,
    region: "mouth",
    recipeVersion: 2,
    hidesPartIds,
    layers: [
      canvasToLayer("mouth_sub_skin", skinCanvas, bounds, hidesPartIds),
      canvasToLayer("mouth_sub_inner", innerCanvas, bounds),
      canvasToLayer("mouth_sub_lip", lipCanvas, bounds),
    ],
  };
}

/**
 * Generates or refreshes HeadFrame-aligned mouth substitute assets while keeping eye expressions
 * driven natively by real upper-eyelid mesh curves (`^` smile and downward `u` sad).
 */
export function synthesizeShapeAssetsForProject(
  project: Project,
  sourcePixels: Uint8ClampedArray,
  forceRegenerate = false,
): Project {
  const existingAssets = { ...(project.shapeAssets ?? {}) };
  const presetAssetAssignments = new Map<string, { asset: ShapeSubstituteAsset; region: ShapeRegion }>();

  // Remove stale synthetic eye sticker links/assets on regeneration so real upper lashes drive blink/smile/sad.
  let clearedEyeLinks = false;
  if (forceRegenerate) {
    for (const side of ["L", "R"] as const) {
      for (const kind of ["closed", "smiling"] as const) {
        const sid = `eye_${side}_${kind}_v1`;
        if (existingAssets[sid]) {
          delete existingAssets[sid];
          clearedEyeLinks = true;
        }
      }
    }
  }

  const mouthKinds: ("closed" | "a" | "i" | "u" | "e" | "o")[] = [
    "closed",
    "a",
    "i",
    "u",
    "e",
    "o",
  ];
  for (const kind of mouthKinds) {
    const presetId = `mouth_${kind}`;
    const preset = project.shapePresets.find((s) => s.id === presetId);
    if (!preset) continue;
    if (!forceRegenerate && preset.assetId && existingAssets[preset.assetId]) continue;
    const asset = buildMouthSubstituteAsset(project, sourcePixels, kind);
    existingAssets[asset.assetId] = asset;
    presetAssetAssignments.set(presetId, { asset, region: "mouth" });
  }

  const hasSyntheticEyeLink = project.shapePresets.some(
    (p) =>
      (p.region === "eye_L" || p.region === "eye_R") &&
      !!p.assetId &&
      /^eye_[LR]_(closed|smiling)_v1$/.test(p.assetId) &&
      (forceRegenerate || !existingAssets[p.assetId]),
  );

  if (presetAssetAssignments.size === 0 && !clearedEyeLinks && !hasSyntheticEyeLink) {
    return project;
  }

  return {
    ...project,
    shapeAssets: existingAssets,
    shapePresets: project.shapePresets.map((preset) => {
      const assigned = presetAssetAssignments.get(preset.id);
      if (assigned) return { ...preset, assetId: assigned.asset.assetId };
      if (
        (preset.region === "eye_L" || preset.region === "eye_R") &&
        preset.assetId &&
        /^eye_[LR]_(closed|smiling)_v1$/.test(preset.assetId) &&
        (forceRegenerate || !existingAssets[preset.assetId])
      ) {
        const next = { ...preset };
        delete next.assetId;
        return next;
      }
      return preset;
    }),
  };
}
