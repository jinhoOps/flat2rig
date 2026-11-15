import { zlibSync } from "fflate";
import { clamp } from "../engine";
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

function buildEyeSubstituteAsset(
  project: Project,
  sourcePixels: Uint8ClampedArray,
  region: "eye_L" | "eye_R",
  kind: "closed" | "smiling",
): ShapeSubstituteAsset {
  const isLeft = region === "eye_L";
  const prefix = isLeft ? "eyeL" : "eyeR";
  const isSd = project.rigProfile.proportionMode === "sd";
  const eye = isLeft ? project.landmarks.eyeL : project.landmarks.eyeR;
  const eyeDist = Math.max(
    12,
    Math.hypot(
      project.landmarks.eyeL.x - project.landmarks.eyeR.x,
      project.landmarks.eyeL.y - project.landmarks.eyeR.y,
    ),
  );
  const hidesPartIds = [
    `${prefix}_white`,
    `${prefix}_iris`,
    `${prefix}_lower`,
    `${prefix}_upper`,
    `${prefix}_all`,
  ];

  const nonSkinPartIds = [
    ...hidesPartIds,
    "browL",
    "browR",
    ...project.parts
      .filter(
        (p) =>
          p.id.startsWith("hair") ||
          p.id.startsWith("acc_") ||
          p.id === "hat",
      )
      .map((p) => p.id),
  ];

  // Exclude eye parts, brows, hair, and headwear/glasses during skin sampling
  const colors = sampleRegionColors(
    project,
    sourcePixels,
    eye.x,
    eye.y,
    eyeDist * 0.45,
    [`${prefix}_upper`, `${prefix}_all`],
    nonSkinPartIds,
  );

  const union = unionPartsMask(
    project.parts,
    hidesPartIds,
    project.width,
    project.height,
  );
  const defaultRx = eyeDist * (isSd ? 0.26 : 0.22);
  const defaultRy = eyeDist * (isSd ? 0.22 : 0.14);
  const pad = Math.max(4, Math.round(eyeDist * 0.08));
  const boxLeft = union.bounds
    ? Math.min(union.bounds.left, eye.x - defaultRx) - pad
    : eye.x - defaultRx - pad;
  const boxTop = union.bounds
    ? Math.min(union.bounds.top, eye.y - defaultRy) - pad
    : eye.y - defaultRy - pad;
  const boxRight = union.bounds
    ? Math.max(union.bounds.right + 1, eye.x + defaultRx) + pad
    : eye.x + defaultRx + pad;
  const boxBot = union.bounds
    ? Math.max(union.bounds.bottom + 1, eye.y + defaultRy) + pad
    : eye.y + defaultRy + pad;
  const bounds = clampRect(
    boxLeft,
    boxTop,
    boxRight - boxLeft,
    boxBot - boxTop,
    project.width,
    project.height,
  );

  const localX = eye.x - bounds.left;
  const localY = eye.y - bounds.top;

  // Layer 1: Smooth perimeter-blended skin cover hiding open eye aperture cleanly
  const unionRx = union.bounds
    ? Math.max(
        localX - (union.bounds.left - bounds.left),
        union.bounds.right + 1 - bounds.left - localX,
      )
    : defaultRx;
  const unionRy = union.bounds
    ? Math.max(
        localY - (union.bounds.top - bounds.top),
        union.bounds.bottom + 1 - bounds.top - localY,
      )
    : defaultRy;
  const rxCover = Math.max(6, defaultRx * 1.12, unionRx * 1.08);
  const ryCover = Math.max(5, defaultRy * 1.12, unionRy * 1.08);
  const perimeter = buildPerimeterSkinPalette(
    project,
    sourcePixels,
    eye.x,
    eye.y,
    rxCover,
    ryCover,
    nonSkinPartIds,
    colors.skin,
  );

  const skinCanvas = canvas(bounds.width, bounds.height);
  const sg = skinCanvas.getContext("2d")!;
  const skinImg = sg.createImageData(bounds.width, bounds.height);
  for (let y = 0; y < bounds.height; y++) {
    for (let x = 0; x < bounds.width; x++) {
      const gx = bounds.left + x;
      const gy = bounds.top + y;
      const partAlpha = union.mask[gy * project.width + gx];
      const dx = (x + 0.5 - localX) / rxCover;
      const dy = (y + 0.5 - localY) / ryCover;
      const r2 = dx * dx + dy * dy;
      const ellipseAlpha =
        r2 >= 1 ? 0 : Math.round(clamp((1 - r2) / 0.35, 0, 1) * 255);
      const a = Math.max(partAlpha, ellipseAlpha);
      if (!a) continue;
      const baseRgb = sampleSectorSkin(perimeter, colors.skin, dx, dy);
      const di = (y * bounds.width + x) * 4;
      // Subtle soft tint only on upper closed lid crease or warm lower cheek
      const tintMix =
        kind === "smiling" && dy > 0.25
          ? 0.08
          : kind === "closed" && dy < -0.1
            ? 0.03
            : 0;
      const tintRgb: [number, number, number] =
        kind === "smiling"
          ? colors.rose
          : [
              Math.max(0, colors.skin[0] - 8),
              Math.max(0, colors.skin[1] - 10),
              Math.max(0, colors.skin[2] - 8),
            ];
      skinImg.data[di] = Math.round(baseRgb[0] * (1 - tintMix) + tintRgb[0] * tintMix);
      skinImg.data[di + 1] = Math.round(baseRgb[1] * (1 - tintMix) + tintRgb[1] * tintMix);
      skinImg.data[di + 2] = Math.round(baseRgb[2] * (1 - tintMix) + tintRgb[2] * tintMix);
      skinImg.data[di + 3] = a;
    }
  }
  sg.putImageData(skinImg, 0, 0);

  // Layer 2: Crisp closed or smiling eyelash arc (sampled character ink)
  const lidCanvas = canvas(bounds.width, bounds.height);
  const lg = lidCanvas.getContext("2d")!;
  const span = Math.max(4, eyeDist * (isSd ? 0.21 : 0.17));
  const strokeW = clamp(eyeDist * (isSd ? 0.042 : 0.03), 1.6, 8);
  const outerDir = isLeft ? 1 : -1;

  lg.lineCap = "round";
  lg.lineJoin = "round";
  lg.strokeStyle = rgbCss(colors.ink);
  lg.lineWidth = strokeW;

  if (kind === "closed") {
    const midY = localY + eyeDist * (isSd ? 0.045 : 0.025);
    const outerTipX = localX + outerDir * span;
    const innerTipX = localX - outerDir * span * 0.88;
    const tipY = midY - eyeDist * (isSd ? 0.025 : 0.015);
    const dipY = midY + eyeDist * (isSd ? 0.045 : 0.022);
    lg.beginPath();
    lg.moveTo(innerTipX, tipY);
    lg.quadraticCurveTo(localX, dipY, outerTipX, tipY - eyeDist * 0.008);
    lg.stroke();

    // Subtle tapered outer lash wing
    lg.lineWidth = Math.max(1.2, strokeW * 0.55);
    lg.beginPath();
    lg.moveTo(outerTipX - outerDir * strokeW * 0.4, tipY);
    lg.lineTo(outerTipX + outerDir * span * 0.14, tipY + eyeDist * 0.012);
    lg.stroke();
  } else {
    const baseY = localY + eyeDist * (isSd ? 0.02 : 0.015);
    const peakY = localY - eyeDist * (isSd ? 0.13 : 0.08);
    const innerTipX = localX - outerDir * span * 0.88;
    const outerTipX = localX + outerDir * span;
    lg.beginPath();
    lg.moveTo(innerTipX, baseY);
    lg.quadraticCurveTo(localX, peakY, outerTipX, baseY - eyeDist * 0.006);
    lg.stroke();

    // Soft happy double-lid / lower cheek smile crease hint
    lg.strokeStyle = rgbCss(colors.ink, 0.35);
    lg.lineWidth = Math.max(1.0, strokeW * 0.4);
    lg.beginPath();
    lg.moveTo(localX - outerDir * span * 0.45, peakY - eyeDist * 0.032);
    lg.quadraticCurveTo(
      localX,
      peakY - eyeDist * 0.058,
      localX + outerDir * span * 0.45,
      peakY - eyeDist * 0.035,
    );
    lg.stroke();
  }

  return {
    assetId: `${region}_${kind}_v1`,
    region,
    recipeVersion: 1,
    hidesPartIds,
    layers: [
      canvasToLayer(`${prefix}_sub_skin`, skinCanvas, bounds, hidesPartIds),
      canvasToLayer(`${prefix}_sub_lid`, lidCanvas, bounds),
    ],
  };
}

function buildMouthSubstituteAsset(
  project: Project,
  sourcePixels: Uint8ClampedArray,
  kind: "closed" | "a" | "i" | "u" | "e" | "o",
): ShapeSubstituteAsset {
  const isSd = project.rigProfile.proportionMode === "sd";
  const mouth = project.landmarks.mouth;
  const eyeDist = Math.max(
    12,
    Math.hypot(
      project.landmarks.eyeL.x - project.landmarks.eyeR.x,
      project.landmarks.eyeL.y - project.landmarks.eyeR.y,
    ),
  );
  const headH = Math.max(12, project.landmarks.chin.y - project.landmarks.crown.y);
  const hidesPartIds = ["mouth_upper", "mouth_lower", "mouth_inner", "mouth_all"];

  const colors = sampleRegionColors(
    project,
    sourcePixels,
    mouth.x,
    mouth.y,
    Math.max(12, eyeDist * 0.32),
    hidesPartIds,
    [...hidesPartIds, "face_shadow"],
  );

  const union = unionPartsMask(
    project.parts,
    hidesPartIds,
    project.width,
    project.height,
  );
  const defaultRx = Math.max(8, eyeDist * (isSd ? 0.25 : 0.25));
  const defaultRy = Math.max(6, headH * (isSd ? 0.12 : 0.1));
  const pad = Math.max(4, Math.round(headH * 0.035));

  // Also scan nearby face pixels for drawn lip/mouth ink in case landmark mouth.x is slightly offset on SD faces
  const facePart = project.parts.find((p) => p.id === "face");
  let lipMinX = project.width;
  let lipMaxX = -1;
  let lipMinY = project.height;
  let lipMaxY = -1;
  const scanX0 = clamp(Math.floor(mouth.x - eyeDist * 0.34), 0, project.width - 1);
  const scanX1 = clamp(Math.ceil(mouth.x + eyeDist * 0.34), 0, project.width - 1);
  const scanY0 = clamp(Math.floor(mouth.y - headH * 0.08), 0, project.height - 1);
  const scanY1 = clamp(Math.ceil(mouth.y + headH * 0.08), 0, project.height - 1);
  const skinLuma =
    0.299 * colors.skin[0] + 0.587 * colors.skin[1] + 0.114 * colors.skin[2];
  for (let gy = scanY0; gy <= scanY1; gy++) {
    for (let gx = scanX0; gx <= scanX1; gx++) {
      const idx = gy * project.width + gx;
      const inMouthOrFace =
        union.mask[idx] > 80 || (facePart ? facePart.mask[idx] > 180 : true);
      if (!inMouthOrFace || sourcePixels[idx * 4 + 3] < 180) continue;
      const r = sourcePixels[idx * 4];
      const g = sourcePixels[idx * 4 + 1];
      const b = sourcePixels[idx * 4 + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      const colorDist = Math.hypot(r - colors.skin[0], g - colors.skin[1], b - colors.skin[2]);
      if (luma < skinLuma - 38 || colorDist > 42) {
        if (gx < lipMinX) lipMinX = gx;
        if (gx > lipMaxX) lipMaxX = gx;
        if (gy < lipMinY) lipMinY = gy;
        if (gy > lipMaxY) lipMaxY = gy;
      }
    }
  }
  const hasLipBlob = lipMaxX >= lipMinX && lipMaxY >= lipMinY;
  const blobCenterX = hasLipBlob ? (lipMinX + lipMaxX) * 0.5 : mouth.x;
  const blobCenterY = hasLipBlob ? (lipMinY + lipMaxY) * 0.5 : mouth.y;
  const coverCenterX = hasLipBlob ? (mouth.x + blobCenterX) * 0.5 : mouth.x;
  const coverCenterY = hasLipBlob ? (mouth.y + blobCenterY) * 0.5 : mouth.y;

  let boxLeft = mouth.x - defaultRx - pad;
  let boxTop = mouth.y - defaultRy - pad;
  let boxRight = mouth.x + defaultRx + pad;
  let boxBot = mouth.y + defaultRy + pad;
  if (union.bounds) {
    boxLeft = Math.min(boxLeft, union.bounds.left - pad);
    boxTop = Math.min(boxTop, union.bounds.top - pad);
    boxRight = Math.max(boxRight, union.bounds.right + 1 + pad);
    boxBot = Math.max(boxBot, union.bounds.bottom + 1 + pad);
  }
  if (hasLipBlob) {
    boxLeft = Math.min(boxLeft, lipMinX - pad);
    boxTop = Math.min(boxTop, lipMinY - pad);
    boxRight = Math.max(boxRight, lipMaxX + 1 + pad);
    boxBot = Math.max(boxBot, lipMaxY + 1 + pad);
  }

  const bounds = clampRect(
    boxLeft,
    boxTop,
    boxRight - boxLeft,
    boxBot - boxTop,
    project.width,
    project.height,
  );
  // Center drawn substitute lip near detected lip center if close to landmark, else landmark
  const drawShiftX =
    hasLipBlob && Math.abs(blobCenterX - mouth.x) <= eyeDist * 0.25
      ? blobCenterX
      : mouth.x;
  const drawShiftY =
    hasLipBlob && Math.abs(blobCenterY - mouth.y) <= headH * 0.05
      ? blobCenterY
      : mouth.y;
  const cx = drawShiftX - bounds.left;
  const cy = drawShiftY - bounds.top;
  const skinCx = coverCenterX - bounds.left;
  const skinCy = coverCenterY - bounds.top;
  const skinRx = Math.max(
    defaultRx * 1.05,
    (boxRight - boxLeft) * 0.5 - pad * 0.4,
  );
  const skinRy = Math.max(
    defaultRy * 1.05,
    (boxBot - boxTop) * 0.5 - pad * 0.4,
  );

  // Layer 1: Perimeter-blended skin cover hiding original mouth contour cleanly
  const perimeter = buildPerimeterSkinPalette(
    project,
    sourcePixels,
    coverCenterX,
    coverCenterY,
    skinRx,
    skinRy,
    [...hidesPartIds, "face_shadow"],
    colors.skin,
  );
  const skinCanvas = canvas(bounds.width, bounds.height);
  const sg = skinCanvas.getContext("2d")!;
  const skinImg = sg.createImageData(bounds.width, bounds.height);
  for (let y = 0; y < bounds.height; y++) {
    for (let x = 0; x < bounds.width; x++) {
      const gx = bounds.left + x;
      const gy = bounds.top + y;
      const origA = union.mask[gy * project.width + gx];
      const dx = (x + 0.5 - skinCx) / Math.max(4, skinRx);
      const dy = (y + 0.5 - skinCy) / Math.max(4, skinRy);
      const r2 = dx * dx + dy * dy;
      const softA =
        r2 >= 1 ? 0 : Math.round(clamp((1 - r2) / 0.35, 0, 1) * 255);
      const a = Math.max(origA, softA);
      if (!a) continue;
      const rgb = sampleSectorSkin(perimeter, colors.skin, dx, dy);
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
  const strokeW = clamp(headH * 0.011, 1.3, 4.2);

  if (kind === "closed") {
    const span = Math.max(4, eyeDist * (isSd ? 0.1 : 0.14));
    lg.strokeStyle = rgbCss(colors.ink);
    lg.lineWidth = strokeW;
    lg.lineCap = "round";
    lg.beginPath();
    lg.moveTo(cx - span, cy - strokeW * 0.35);
    lg.quadraticCurveTo(cx, cy + strokeW * 1.1, cx + span, cy - strokeW * 0.35);
    lg.stroke();
    return {
      assetId: `mouth_closed_v1`,
      region: "mouth",
      recipeVersion: 1,
      hidesPartIds,
      layers: [
        canvasToLayer("mouth_sub_skin", skinCanvas, bounds, hidesPartIds),
        canvasToLayer("mouth_sub_lip", lipCanvas, bounds),
      ],
    };
  }

  // Layer 2: Vowel mouth interior cavity + teeth + tongue
  const innerCanvas = canvas(bounds.width, bounds.height);
  const ig = innerCanvas.getContext("2d")!;

  const rxMap = {
    a: eyeDist * (isSd ? 0.14 : 0.17),
    i: eyeDist * (isSd ? 0.17 : 0.21),
    u: eyeDist * (isSd ? 0.068 : 0.088),
    e: eyeDist * (isSd ? 0.155 : 0.19),
    o: eyeDist * (isSd ? 0.095 : 0.115),
  };
  const ryTopMap = {
    a: headH * 0.025,
    i: headH * 0.022,
    u: headH * 0.028,
    e: headH * 0.022,
    o: headH * 0.045,
  };
  const ryBotMap = {
    a: headH * (isSd ? 0.068 : 0.072),
    i: headH * (isSd ? 0.032 : 0.036),
    u: headH * (isSd ? 0.038 : 0.042),
    e: headH * (isSd ? 0.048 : 0.052),
    o: headH * (isSd ? 0.065 : 0.072),
  };
  const rx = Math.max(3, rxMap[kind]);
  const topY = cy - Math.max(1.5, ryTopMap[kind]);
  const botY = cy + Math.max(2.5, ryBotMap[kind]);

  const traceMouthCavity = (g: CanvasRenderingContext2D) => {
    g.beginPath();
    if (kind === "u" || kind === "o") {
      const midY = (topY + botY) * 0.5;
      const rY = (botY - topY) * 0.52;
      g.ellipse(cx, midY, rx, rY, 0, 0, Math.PI * 2);
    } else if (kind === "i") {
      g.moveTo(cx - rx, topY + 0.5);
      g.quadraticCurveTo(cx, topY + 2.2, cx + rx, topY + 0.5);
      g.quadraticCurveTo(cx, botY + 1.5, cx - rx, topY + 0.5);
    } else {
      const cornerY = topY + (botY - topY) * 0.18;
      g.moveTo(cx - rx, cornerY);
      g.quadraticCurveTo(cx, topY - (botY - topY) * 0.16, cx + rx, cornerY);
      g.quadraticCurveTo(cx, botY + (botY - topY) * 0.24, cx - rx, cornerY);
    }
    g.closePath();
  };

  // Paint interior cavity
  ig.save();
  traceMouthCavity(ig);
  ig.fillStyle = "#6e2938";
  ig.fill();
  ig.clip();

  // Tongue dome inside lower mouth
  ig.fillStyle = rgbCss(
    [
      Math.round((colors.rose[0] + 235) * 0.5),
      Math.round((colors.rose[1] + 132) * 0.5),
      Math.round((colors.rose[2] + 144) * 0.5),
    ],
  );
  ig.beginPath();
  ig.ellipse(
    cx,
    botY + (botY - topY) * 0.08,
    rx * 0.78,
    (botY - topY) * 0.52,
    0,
    Math.PI,
    Math.PI * 2,
  );
  ig.fill();

  // Upper teeth bar for open articulated vowels (A / I / E / O)
  if (kind === "i" || kind === "a" || kind === "e") {
    ig.fillStyle = "#fff8f7";
    const teethH = (botY - topY) * (kind === "i" ? 0.52 : 0.28);
    ig.fillRect(cx - rx * 0.88, topY - 1, rx * 1.76, teethH);
  }
  ig.restore();

  // Layer 3: Lip ink border hugging the cavity
  lg.strokeStyle = rgbCss(colors.ink);
  lg.lineWidth = strokeW;
  lg.lineJoin = "round";
  lg.lineCap = "round";
  traceMouthCavity(lg);
  lg.stroke();

  return {
    assetId: `mouth_${kind}_v1`,
    region: "mouth",
    recipeVersion: 1,
    hidesPartIds,
    layers: [
      canvasToLayer("mouth_sub_skin", skinCanvas, bounds, hidesPartIds),
      canvasToLayer("mouth_sub_inner", innerCanvas, bounds),
      canvasToLayer("mouth_sub_lip", lipCanvas, bounds),
    ],
  };
}

/**
 * Generates or refreshes per-character LD/SD shape substitute assets (`closed`/`smiling` eyes and
 * `closed`/`a`/`i`/`u`/`e`/`o` mouths) and links them onto built-in `ShapePreset` entries.
 */
export function synthesizeShapeAssetsForProject(
  project: Project,
  sourcePixels: Uint8ClampedArray,
  forceRegenerate = false,
): Project {
  const existingAssets = { ...(project.shapeAssets ?? {}) };
  const presetAssetAssignments = new Map<string, { asset: ShapeSubstituteAsset; region: ShapeRegion }>();

  const eyeKinds: ("closed" | "smiling")[] = ["closed", "smiling"];
  for (const region of ["eye_L", "eye_R"] as const) {
    for (const kind of eyeKinds) {
      const presetId = `${region}_${kind}`;
      const preset = project.shapePresets.find((s) => s.id === presetId);
      if (!preset) continue;
      if (!forceRegenerate && preset.assetId && existingAssets[preset.assetId]) continue;
      const asset = buildEyeSubstituteAsset(project, sourcePixels, region, kind);
      existingAssets[asset.assetId] = asset;
      presetAssetAssignments.set(presetId, { asset, region });
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

  if (presetAssetAssignments.size === 0) return project;

  return {
    ...project,
    shapeAssets: existingAssets,
    shapePresets: project.shapePresets.map((preset) => {
      const assigned = presetAssetAssignments.get(preset.id);
      return assigned ? { ...preset, assetId: assigned.asset.assetId } : preset;
    }),
  };
}
