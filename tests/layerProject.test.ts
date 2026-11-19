import { beforeAll, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createCanvas, ImageData } from "@napi-rs/canvas";
import { initializeCanvas, writePsdUint8Array, type Layer } from "ag-psd";
import { strFromU8, strToU8, unzipSync, unzlibSync, zipSync } from "fflate";
import {
  buildPsdParts,
  encodePngRgba,
  landmarkMidlinePsd,
  openPsdSession,
  projectFromLayerBuild,
  splitChangesWithMidline,
  withRebuiltLayerParts,
} from "../src/layerProject";
import { partCanvasRgba } from "../src/layerParts";
import { loadProject, saveProject } from "../src/io";
import { effectiveMask } from "../src/engine";
import PsdMappingPanel from "../src/workspace/PsdMappingPanel";
import LayersPanel from "../src/workspace/LayersPanel";
import { layerRows } from "../src/workspace/layers";

beforeAll(() => {
  vi.stubGlobal("document", {
    createElement: (tag: string) => {
      if (tag !== "canvas") throw new Error("Unexpected element");
      return createCanvas(1, 1);
    },
  });
  vi.stubGlobal("ImageData", ImageData);
  initializeCanvas(
    (w, h) => createCanvas(w, h) as unknown as HTMLCanvasElement,
    (w, h) => new ImageData(w, h) as unknown as globalThis.ImageData,
  );
});

type Px = [number, number, number, number];
function rect(name: string, x0: number, y0: number, x1: number, y1: number, color: Px, extra: Partial<Layer> = {}): Layer {
  const w = x1 - x0,
    h = y1 - y0;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) data.set(color, i * 4);
  return {
    name,
    left: x0,
    top: y0,
    right: x1,
    bottom: y1,
    imageData: { width: w, height: h, data } as unknown as globalThis.ImageData,
    ...extra,
  };
}
const group = (name: string, children: Layer[], extra: Partial<Layer> = {}): Layer => ({ name, children, ...extra });

/** A small character: back hair, neck, face (+multiply blush), two eyes, brows, mouth, bangs,
 * a hidden guide group, an unassignable crease and one combined See-through-style iris layer. */
function characterPsd() {
  const half = (x0: number, x1: number, y0: number, y1: number, c: Px) => [rect("", x0, y0, x1, y1, c)][0];
  void half;
  return writePsdUint8Array(
    {
      width: 120,
      height: 90,
      children: [
        group("뒷머리", [rect("뒷머리", 20, 5, 100, 85, [70, 50, 110, 255])]),
        group("몸", [rect("목", 52, 55, 68, 80, [235, 200, 180, 255]), rect("옷", 30, 72, 90, 90, [40, 60, 120, 255])]),
        group("얼굴", [
          rect("얼굴", 35, 15, 85, 62, [240, 210, 190, 255]),
          rect("홍조", 40, 45, 80, 50, [255, 128, 128, 255], { blendMode: "multiply", clipping: true }),
        ]),
        group("눈>", [rect("눈흰자>", 66, 32, 78, 38, [255, 255, 255, 255]), rect("속눈썹>", 64, 29, 80, 32, [30, 20, 20, 255])]),
        group("눈<", [rect("눈흰자<", 42, 32, 54, 38, [255, 255, 255, 255]), rect("속눈썹<", 40, 29, 56, 32, [30, 20, 20, 255])]),
        // One layer holding both irises (See-through style): split by connected components.
        {
          name: "눈동자",
          left: 44,
          top: 31,
          right: 76,
          bottom: 39,
          imageData: (() => {
            const w = 32,
              h = 8,
              data = new Uint8ClampedArray(w * h * 4);
            for (let y = 0; y < h; y++)
              for (let x = 0; x < w; x++) if (x < 8 || x >= 24) data.set([40, 60, 160, 255], (y * w + x) * 4);
            return { width: w, height: h, data } as unknown as globalThis.ImageData;
          })(),
        },
        rect("쌍커풀", 64, 27, 80, 28, [90, 50, 50, 255]),
        group("눈썹", [rect("눈썹>", 66, 22, 78, 25, [80, 50, 40, 255]), rect("눈썹<", 42, 22, 54, 25, [80, 50, 40, 255])]),
        group("입", [rect("입안", 55, 50, 65, 54, [120, 30, 40, 255]), rect("윗입술", 54, 49, 66, 50, [60, 30, 30, 255])]),
        group("입 가이드", [rect("A", 54, 48, 66, 56, [0, 0, 0, 255])], { hidden: true }),
        group("앞머리", [rect("앞머리", 35, 5, 85, 24, [90, 60, 140, 255])]),
      ],
    },
    { noBackground: true },
  );
}

/** Minimal PNG decoder for the encoder's own output (8-bit RGBA, filter 0). */
function decodeOwnPng(bytes: Uint8Array) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let o = 8,
    w = 0,
    h = 0;
  const idat: Uint8Array[] = [];
  while (o < bytes.length) {
    const len = dv.getUint32(o);
    const type = strFromU8(bytes.subarray(o + 4, o + 8));
    const data = bytes.subarray(o + 8, o + 8 + len);
    if (type === "IHDR") {
      w = dv.getUint32(o + 8);
      h = dv.getUint32(o + 12);
    } else if (type === "IDAT") idat.push(data);
    o += 12 + len;
  }
  const z = new Uint8Array(idat.reduce((n, d) => n + d.length, 0));
  let k = 0;
  for (const d of idat) {
    z.set(d, k);
    k += d.length;
  }
  const raw = unzlibSync(z);
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    expect(raw[y * (w * 4 + 1)]).toBe(0);
    out.set(raw.subarray(y * (w * 4 + 1) + 1, (y + 1) * (w * 4 + 1)), y * w * 4);
  }
  return { w, h, rgba: out };
}
const dataUrlBytes = (url: string) => Uint8Array.from(atob(url.split(",")[1]), (c) => c.charCodeAt(0));

async function built() {
  const session = await openPsdSession(characterPsd(), "character.psd");
  const build = buildPsdParts(session);
  return { session, build, project: projectFromLayerBuild(session, build) };
}

describe("B1 layered PSD → project", () => {
  it("builds layer parts with provenance layer, exact textures and the PSD record", async () => {
    const { session, build, project } = await built();
    expect(session.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(project.provider).toBe("layers");
    expect([project.width, project.height]).toEqual([120, 90]);
    const ids = project.parts.map((p) => p.id);
    expect(ids).toEqual(build.parts.map((p) => p.id));
    expect(ids).toEqual(
      expect.arrayContaining(["hair_back", "neck", "clothes", "face", "eyeL_white", "eyeR_white", "eyeL_iris", "eyeR_iris", "eyeL_upper", "eyeR_upper", "browL", "browR", "mouth_inner", "mouth_upper", "hair_front"]),
    );
    for (const [i, p] of project.parts.entries()) {
      expect(p.provenance).toBe("layer");
      expect(p.compositeOrder).toBe(i);
      // Texture PNG is the built part bit for bit (no canvas premultiply round trip).
      const png = decodeOwnPng(dataUrlBytes(p.texture!));
      expect([png.w, png.h]).toEqual([120, 90]);
      expect(Buffer.from(png.rgba).equals(Buffer.from(partCanvasRgba(build.parts[i], 120, 90)))).toBe(true);
      // effective mask = part alpha
      const eff = effectiveMask(p);
      let diff = 0;
      for (let k = 0; k < 120 * 90; k++) if (eff[k] !== png.rgba[k * 4 + 3]) diff++;
      expect(diff).toBe(0);
      expect(p.fillKind).toBe(p.fillMask ? "external" : "none");
    }
    const face = project.parts.find((p) => p.id === "face")!;
    expect(face.fillKind).toBe("external"); // covered by eyes, brows, mouth, bangs
    const ls = project.layerSource!;
    expect(ls.version).toBe("psd-layers-v1");
    expect(ls.file).toBe("character.psd");
    expect([ls.width, ls.height]).toEqual([120, 90]);
    expect(ls.missingDefaultParts).toEqual(["body", "eyeR_lower", "eyeL_lower", "mouth_lower"]);
    expect(ls.parts.find((p) => p.id === "face")!.layers).toEqual(["/얼굴/얼굴", "/얼굴/홍조"]);
    expect(ls.parts.find((p) => p.id === "eyeL_iris")!.layers).toEqual(["/눈동자"]);
    const blush = ls.mappings.find((m) => m.path === "/얼굴/홍조")!;
    expect(blush).toMatchObject({ target: "face", include: true, blend: "fold", blendMode: "multiply" });
    expect(ls.mappings.find((m) => m.path === "/쌍커풀")!.target).toBeNull();
    expect(ls.mappings.find((m) => m.path === "/입 가이드/A")!.include).toBe(false);
    // The composite becomes the project image used for face analysis.
    const src = decodeOwnPng(dataUrlBytes(project.source));
    expect(Buffer.from(src.rgba).equals(Buffer.from(build.composite))).toBe(true);
  });

  it("round-trips through save → reopen with masks, fills, provenance and the layer record", async () => {
    const { project } = await built();
    const bytes = await saveProject(project);
    const reopened = loadProject(bytes);
    expect(reopened.layerSource).toEqual(project.layerSource);
    expect(reopened.parts.map((p) => p.id)).toEqual(project.parts.map((p) => p.id));
    for (const [i, p] of reopened.parts.entries()) {
      const o = project.parts[i];
      expect(p.provenance).toBe("layer");
      expect(p.fillKind).toBe(o.fillKind);
      expect(p.texture).toBe(o.texture);
      expect([...p.mask]).toEqual([...o.mask]);
      expect(p.fillMask ? [...p.fillMask] : null).toEqual(o.fillMask ? [...o.fillMask] : null);
    }
    // A second save of the reopened project is byte-identical in project.json.
    const again = unzipSync(await saveProject(reopened));
    const first = unzipSync(bytes);
    expect(strFromU8(again["project.json"])).toBe(strFromU8(first["project.json"]));
  });

  it("rejects a malformed layer record instead of loading it half-validated", async () => {
    const { project } = await built();
    const files = unzipSync(await saveProject(project));
    const meta = JSON.parse(strFromU8(files["project.json"]));
    for (const broken of [
      { ...meta.layerSource, version: "psd-layers-v9" },
      { ...meta.layerSource, mappings: [{ path: "/a", target: 3, include: true, source: "name" }] },
      { ...meta.layerSource, mappings: [{ path: "/a", target: null, include: true, source: "guess" }] },
      { ...meta.layerSource, sha256: "xyz" },
    ]) {
      const bad = { ...files, "project.json": strToU8(JSON.stringify({ ...meta, layerSource: broken })) };
      expect(() => loadProject(zipSync(bad))).toThrow(/레이어 PSD/);
    }
    // Projects without the field still load (field is optional in schema v3).
    const { layerSource: _, ...plain } = meta;
    const ok = loadProject(zipSync({ ...files, "project.json": strToU8(JSON.stringify(plain)) }));
    expect(ok.layerSource).toBeUndefined();
  });

  it("re-splits a combined L/R layer around the landmark midline only when it changes components", async () => {
    const { session, build, project } = await built();
    expect(build.splits.map((s) => s.path)).toEqual(["/눈동자"]);
    // Placeholder landmarks: no landmark midline.
    expect(landmarkMidlinePsd(project, build.sx)).toBeNull();
    const located = {
      ...project,
      landmarks: { ...project.landmarks, eyeL: { x: 72, y: 35 }, eyeR: { x: 48, y: 35 } },
      landmarkProvenance: { ...project.landmarkProvenance!, eyeL: "model" as const, eyeR: "model" as const },
    };
    const mid = landmarkMidlinePsd(located, build.sx)!;
    expect(mid).toBe(60);
    expect(splitChangesWithMidline(session, build, mid)).toBe(false);
    // A midline right of both irises sends both components to R.
    expect(splitChangesWithMidline(session, build, 100)).toBe(true);
    const moved = { ...session, midlineX: 100 };
    const rebuilt = buildPsdParts(moved);
    const next = withRebuiltLayerParts(located, moved, rebuilt);
    expect(next.landmarks).toEqual(located.landmarks);
    expect(next.layerSource!.midlineSource).toBe("landmarks");
    const irisL = rebuilt.parts.find((p) => p.id === "eyeL_iris")!;
    expect(irisL.rgba.some((v, k) => k % 4 === 3 && v > 0)).toBe(false);
  });

  it("encodes PNGs that a canvas decoder reads back", async () => {
    const rgba = new Uint8Array([255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 9, 9, 9, 255]);
    const png = encodePngRgba(rgba, 2, 2);
    const { loadImage } = await import("@napi-rs/canvas");
    const img = await loadImage(Buffer.from(png));
    expect([img.width, img.height]).toEqual([2, 2]);
  });
});

describe("B1 mapping screen and layer panel", () => {
  it("renders the PSD tree top-first with targets, blend choice, and missing parts as 없음", async () => {
    const { session } = await built();
    const html = renderToStaticMarkup(
      createElement(PsdMappingPanel, {
        session,
        busy: false,
        onChange: () => {},
        onGenerate: () => {},
        onCancel: () => {},
      }),
    );
    // Top-first: 앞머리 group comes before 뒷머리.
    expect(html.indexOf('data-path="/앞머리"')).toBeLessThan(html.indexOf('data-path="/뒷머리"'));
    expect(html).toContain("파츠 생성 (");
    expect(html).toContain('data-action="generate-layer-parts"');
    expect(html).toMatch(/data-part-id="body"[^>]*>.*?없음/s);
    expect(html).toContain("파츠 안에서 multiply 적용");
    expect(html).toContain("미지정 1");
    expect(html).toContain("가이드");
  });

  it("lists default parts with no layer as 없음 rows in the layer panel", async () => {
    const { project } = await built();
    const html = renderToStaticMarkup(
      createElement(LayersPanel, {
        rows: layerRows(project.parts),
        width: project.width,
        height: project.height,
        selected: "face",
        editTarget: "visible",
        busy: false,
        hasProject: true,
        partTextures: new Map(),
        missingParts: project.layerSource!.missingDefaultParts,
        approved: 0,
        total: project.parts.length,
        onSelectRow: () => {},
        onSelectThumb: () => {},
        onToggleVisible: () => {},
        onToggleGroup: () => {},
        onOpenImage: () => {},
        onOpenProject: () => {},
        onAddPart: () => {},
        onAddDynamic: () => {},
      }),
    );
    expect(html).toContain('data-part-id="body" data-absent="true"');
    expect(html).toContain("레이어 없음");
    expect(html).toContain("레이어 PSD"); // provenance label of layer parts
  });
});
