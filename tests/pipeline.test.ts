import { beforeAll, describe, expect, it, vi } from "vitest";
import { createCanvas, Image, ImageData } from "@napi-rs/canvas";
import { initializeCanvas, readPsd } from "ag-psd";
import { strFromU8, unzipSync, zipSync, strToU8 } from "fflate";
import {
  defaultLandmarks,
  foregroundMask,
  paint,
  separate,
  meshFor,
} from "../src/engine";
import {
  cubismCsv,
  exportBundle,
  exportPsd,
  loadProject,
  parseCsv,
  readCsvTemplate,
  saveProject,
} from "../src/io";
import type { Project } from "../src/types";

beforeAll(() => {
  vi.stubGlobal("document", {
    createElement: (tag: string) => {
      if (tag !== "canvas") throw new Error("Unexpected element");
      return createCanvas(1, 1);
    },
  });
  vi.stubGlobal("Image", Image);
  vi.stubGlobal("ImageData", ImageData);
  initializeCanvas(
    (w, h) => createCanvas(w, h) as unknown as HTMLCanvasElement,
    (w, h) => new ImageData(w, h) as unknown as globalThis.ImageData,
  );
});
function fixture(): Project {
  const width = 64,
    height = 64,
    c = createCanvas(width, height),
    ctx = c.getContext("2d");
  ctx.fillStyle = "#b8a4e5";
  ctx.fillRect(8, 4, 48, 58);
  ctx.fillStyle = "#f6d6c1";
  ctx.fillRect(19, 12, 26, 26);
  const data = new Uint8ClampedArray(
      ctx.getImageData(0, 0, width, height).data,
    ),
    landmarks = defaultLandmarks(width, height);
  return {
    schemaVersion: 1,
    name: "테스트 캐릭터",
    width,
    height,
    originalWidth: width,
    originalHeight: height,
    source: c.toDataURL("image/png"),
    original: c.toDataURL("image/png"),
    landmarks,
    parts: separate(data, width, height, landmarks, 0),
    revision: 3,
    provider: "geometry-template-v1 (browser fallback)",
    createdAt: "2026-11-15T00:00:00Z",
  };
}
describe("image partition and editing invariants", () => {
  it("does not duplicate or drop source alpha when splitting", () => {
    const w = 32,
      h = 32,
      data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      data[i * 4] = 140;
      data[i * 4 + 3] = i % 256;
    }
    const parts = separate(data, w, h, defaultLandmarks(w, h), 0);
    for (let i = 0; i < w * h; i++)
      expect(parts.reduce((n, p) => n + p.mask[i], 0)).toBe(data[i * 4 + 3]);
  });
  it("removes only border-connected background", () => {
    const w = 9,
      h = 9,
      data = new Uint8ClampedArray(w * h * 4).fill(255);
    for (let y = 2; y < 7; y++)
      for (let x = 2; x < 7; x++)
        data.fill(0, (y * w + x) * 4, (y * w + x) * 4 + 3);
    data.fill(255, (4 * w + 4) * 4, (4 * w + 4) * 4 + 3);
    const mask = foregroundMask(data, w, h, 20);
    expect(mask[0]).toBe(0);
    expect(mask[4 * w + 4]).toBe(255);
  });
  it("paints continuous strokes and never wraps the canvas edge", () => {
    const mask = new Uint8Array(40 * 20);
    paint(mask, 40, 20, { x: 0, y: 0 }, { x: 30, y: 0 }, 2, false);
    for (let x = 0; x <= 30; x++) expect(mask[x]).toBe(255);
    expect(mask[39]).toBe(0);
    paint(mask, 40, 20, { x: 10, y: 0 }, { x: 20, y: 0 }, 2, true);
    expect(mask[15]).toBe(0);
    expect(mask[2]).toBe(255);
  });
  it("does not triangulate across an interior hole", () => {
    const mask = new Uint8Array(32 * 32).fill(255);
    for (let y = 10; y < 22; y++)
      for (let x = 10; x < 22; x++) mask[y * 32 + x] = 0;
    const m = meshFor(mask, 32, 32);
    expect(m.triangles.length).toBeGreaterThan(0);
    for (let i = 0; i < m.triangles.length; i += 3) {
      const pts = m.triangles.slice(i, i + 3).map((j) => m.vertices[j]);
      const x = Math.floor(pts.reduce((n, p) => n + p.x, 0) / 3),
        y = Math.floor(pts.reduce((n, p) => n + p.y, 0) / 3);
      expect(mask[y * 32 + x]).toBe(255);
    }
  });
});
describe("persistent project and export contract", () => {
  it("round-trips every byte of masks, metadata, and generated texture", async () => {
    const p = fixture();
    p.parts[0].generated = true;
    p.parts[0].texture = p.source;
    p.parts[1].visible = false;
    p.parts[2].status = "missing";
    const restored = loadProject(await saveProject(p));
    expect(restored.name).toBe(p.name);
    expect(restored.source).toBe(p.source);
    expect(restored.landmarks).toEqual(p.landmarks);
    expect(
      restored.parts.map((p) => ({ ...p, mask: Array.from(p.mask) })),
    ).toEqual(p.parts.map((p) => ({ ...p, mask: Array.from(p.mask) })));
  });
  it("rejects corrupted dimensions and missing masks", async () => {
    const files = unzipSync(await saveProject(fixture()));
    delete files["masks/face.bin"];
    expect(() => loadProject(zipSync(files))).toThrow("마스크");
    const p = fixture();
    p.width = 999999;
    files["project.json"] = strToU8(JSON.stringify(p));
    expect(() => loadProject(zipSync(files))).toThrow("형식");
  });
  it("writes a real layered RGB PSD with matching offsets and alpha", async () => {
    const p = fixture();
    p.parts[0].visible = false;
    const bytes = await exportPsd(p);
    expect(String.fromCharCode(...bytes.slice(0, 4))).toBe("8BPS");
    const psd = readPsd(bytes, { useImageData: true, skipThumbnail: true });
    expect(psd.width).toBe(64);
    expect(psd.height).toBe(64);
    expect(psd.bitsPerChannel).toBe(8);
    expect(psd.children!.length).toBeGreaterThan(5);
    expect(
      psd.children!.some((layer) => layer.name?.startsWith("hair_back")),
    ).toBe(true);
    for (const layer of psd.children!) {
      expect(layer.hidden).toBe(false);
      const id = layer.name!.split(" · ")[0],
        part = p.parts.find((p) => p.id === id)!;
      const image = layer.imageData!;
      for (let y = 0; y < image.height; y++)
        for (let x = 0; x < image.width; x++) {
          const alpha = image.data[(y * image.width + x) * 4 + 3];
          expect(alpha).toBe(
            part.mask[(y + layer.top!) * p.width + x + layer.left!],
          );
        }
    }
  });
  it("packages re-openable project, real PSD, rig and truthful handoff", async () => {
    const p = fixture(),
      bytes = await exportBundle(p, null),
      files = unzipSync(bytes);
    expect(files["character_import.psd"]).toBeDefined();
    expect(loadProject(files["project.flat2rig"]).parts.length).toBe(18);
    const rig = JSON.parse(strFromU8(files["rig-draft.json"]));
    expect(rig.directlyImportableInCubism).toBe(false);
    expect(rig.parts.length).toBe(18);
    expect(files["parameters-reference.csv"]).toBeDefined();
    expect(files["parameters.csv"]).toBeUndefined();
    expect(
      JSON.parse(strFromU8(files["validation.json"])).cubismEditorImport,
    ).toBe("not-tested");
    expect(files["guides/landmarks.png"]).toBeDefined();
  });
});
describe("sample-derived parameter CSV", () => {
  it("preserves template headers/type, clears GUID, and quotes correctly", () => {
    const template = readCsvTemplate(
      '\uFEFFGUID,Type,ID,Name,Group,Minimum value,Default value,Maximum value,Repeat,Description\r\n123,Parameter,ParamAngleX,"Angle, X",,-30,0,30,false,original',
    );
    const rows = parseCsv(cubismCsv(template));
    expect(rows[0]).toEqual(template.headers);
    expect(rows[1][0]).toBe("");
    expect(rows[1][1]).toBe("Parameter");
    expect(rows[1][2]).toBe("ParamAngleX");
    expect(rows[1][5]).toBe("-30");
    expect(rows.length).toBe(8);
  });
  it("rejects arbitrary CSV rather than claiming Cubism compatibility", () => {
    expect(() => readCsvTemplate("ID,Name\nParamAngleX,Head")).toThrow();
    expect(() => parseCsv('a,"not closed')).toThrow();
  });
});
