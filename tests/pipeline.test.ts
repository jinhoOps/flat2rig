import { beforeAll, describe, expect, it, vi } from "vitest";
import { createCanvas, Image, ImageData } from "@napi-rs/canvas";
import { initializeCanvas, readPsd } from "ag-psd";
import { strFromU8, unzipSync, zipSync, strToU8 } from "fflate";
import {
  checkMeshQuality,
  deform,
  defaultLandmarks,
  effectiveMask,
  expandFillSeparated,
  foregroundMask,
  meshFor,
  meshForPart,
  paint,
  separate,
} from "../src/engine";
import {
  cubismCsv,
  exportBundle,
  exportPsd,
  insertHairPartInOrder,
  loadProject,
  newHairPart,
  parseCsv,
  readCsvTemplate,
  saveProject,
} from "../src/io";
import { REST, type Project } from "../src/types";

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
    schemaVersion: 2,
    name: "테스트 캐릭터",
    width,
    height,
    originalWidth: width,
    originalHeight: height,
    source: c.toDataURL("image/png"),
    original: c.toDataURL("image/png"),
    landmarks,
    landmarkProvenance: {
      eyeL: "model",
      eyeR: "model",
      mouth: "model",
      chin: "model",
      crown: "estimated",
    },
    parts: separate(data, width, height, landmarks, 0).map((part, index) => ({
      ...part,
      provenance: "geometry",
      fillKind: "none",
      compositeOrder: index,
    })),
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
  it("builds contour-aware part meshes that pass extreme-pose quality checks", () => {
    const mask = new Uint8Array(64 * 64);
    for (let y = 10; y < 52; y++)
      for (let x = 14; x < 50; x++) mask[y * 64 + x] = 255;
    const lm = defaultLandmarks(64, 64);
    const mesh = meshForPart(mask, 64, 64, 14, lm);
    const report = checkMeshQuality(mesh, mask, 64, 64, { id: "face" }, lm);
    expect(report.triangleCount).toBeGreaterThan(10);
    expect(report.holeCrossings).toBe(0);
    expect(report.flippedAtExtremes).toBe(0);
    expect(report.extremePosesTested).toBe(12);
  });
  it("keeps body and clothes stationary during head and hair deformation", () => {
    const lm = defaultLandmarks(128, 128);
    const pose = {
      ...REST,
      angleX: 30,
      angleY: -25,
      angleZ: 20,
      hair: 1,
    };
    const pt = { x: 64, y: 96 };
    expect(deform(pt, { id: "body" }, lm, pose)).toEqual(pt);
    expect(deform(pt, { id: "clothes", role: "clothes" }, lm, pose)).toEqual(pt);
    expect(deform(pt, { id: "hair_side_R", role: "hair_side_R" }, lm, pose)).not.toEqual(pt);
  });
  it("keeps visible mask separate from neighbor fillMask and emits hidden feature warning", () => {
    const visible = new Uint8Array(16 * 16);
    for (let y = 5; y < 11; y++)
      for (let x = 5; x < 11; x++) visible[y * 16 + x] = 255;
    const rgba = new Uint8ClampedArray(16 * 16 * 4).fill(255);
    const occ = expandFillSeparated(rgba, visible, 16, 16, 2, "eyeR_white");
    expect(occ.fillKind).toBe("neighbor");
    expect(occ.filledPixels).toBeGreaterThan(0);
    expect(occ.warning).toBeTruthy();
    const eff = effectiveMask({ mask: visible, fillMask: occ.fillMask });
    for (let i = 0; i < visible.length; i++) {
      expect(visible[i] > 0 && occ.fillMask[i] > 0).toBe(false);
      expect(eff[i]).toBe(occ.combinedMask[i]);
    }
  });
  it("erodes dark outermost outline before filling occluded pixels outward", () => {
    const w = 11,
      h = 11;
    const visible = new Uint8Array(w * h);
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let y = 2; y < 9; y++)
      for (let x = 2; x < 9; x++) {
        const i = y * w + x;
        visible[i] = 255;
        const rim = x === 2 || x === 8 || y === 2 || y === 8;
        rgba[i * 4] = rim ? 10 : 240;
        rgba[i * 4 + 1] = rim ? 10 : 200;
        rgba[i * 4 + 2] = rim ? 10 : 180;
        rgba[i * 4 + 3] = 255;
      }
    const occ = expandFillSeparated(rgba, visible, w, h, 2, "face");
    const probe = 5 * w + 1;
    expect(occ.fillMask[probe]).toBe(255);
    expect(occ.data[probe * 4]).toBeGreaterThanOrEqual(220);
    expect(occ.data[(5 * w + 2) * 4]).toBe(10);
  });
  it("scales hair deform sway by role and strand suffix matching Rust parity", () => {
    const lm = defaultLandmarks(64, 64);
    const pt = { x: 32, y: 40 };
    const pose = { ...REST, hair: 1 };
    const front = deform(pt, { id: "hair_front" }, lm, pose);
    const front2 = deform(pt, { id: "hair_front_2" }, lm, pose);
    const side = deform(pt, { id: "hair_side_L" }, lm, pose);
    const back = deform(pt, { id: "hair_back" }, lm, pose);
    expect(front2.x).toBeGreaterThan(front.x);
    expect(side.x).toBeGreaterThan(back.x);
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
    expect(restored.schemaVersion).toBe(2);
    expect(restored.name).toBe(p.name);
    expect(restored.source).toBe(p.source);
    expect(restored.landmarks).toEqual(p.landmarks);
    expect(
      restored.parts.map((p) => ({ ...p, mask: Array.from(p.mask) })),
    ).toEqual(p.parts.map((p) => ({ ...p, mask: Array.from(p.mask) })));
  });
  it("migrates legacy schemaVersion 1 projects to v2 preserving landmarks", async () => {
    const v2 = fixture();
    const {
      landmarkProvenance: _lp,
      schemaVersion: _sv,
      ...legacyBase
    } = v2;
    const legacyFiles = unzipSync(await saveProject(v2));
    const legacyMeta = {
      ...legacyBase,
      schemaVersion: 1,
      parts: v2.parts.map(({ mask: _, provenance: _p, fillKind: _f, compositeOrder: _c, ...rest }) => rest),
    };
    legacyFiles["project.json"] = strToU8(JSON.stringify(legacyMeta));
    const migrated = loadProject(zipSync(legacyFiles));
    expect(migrated.schemaVersion).toBe(2);
    expect(migrated.landmarks).toEqual(v2.landmarks);
    expect(migrated.landmarkProvenance?.eyeL).toBe("manual");
    expect(migrated.parts.length).toBe(18);
    expect(migrated.parts[0].provenance).toBe("geometry");
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
  it("round-trips fillMask and compositeOrder through project, PSD, and validation bundle", async () => {
    const p = fixture();
    const face = p.parts.find((t) => t.id === "face")!;
    const rgba = new Uint8ClampedArray(p.width * p.height * 4).fill(240);
    const occ = expandFillSeparated(rgba, face.mask, p.width, p.height, 2, face.id);
    face.fillMask = occ.fillMask;
    face.fillKind = occ.fillKind;
    face.provenance = "model";
    p.modelHashes = { part_decoder: "abc123def456" };

    const restored = loadProject(await saveProject(p));
    const rFace = restored.parts.find((t) => t.id === "face")!;
    expect(rFace.fillMask).toBeDefined();
    expect(rFace.fillMask!.length).toBe(p.width * p.height);
    expect(rFace.fillKind).toBe("neighbor");

    const psdBytes = await exportPsd(p);
    const psd = readPsd(psdBytes, { useImageData: true, skipThumbnail: true });
    const psdFace = psd.children!.find((c) => c.name?.startsWith("face ·"))!;
    expect(psdFace).toBeDefined();

    const bundle = await exportBundle(p, null);
    const files = unzipSync(bundle);
    expect(files["masks/face.visible.png"]).toBeDefined();
    expect(files["masks/face.fill.png"]).toBeDefined();
    const validation = JSON.parse(strFromU8(files["validation.json"]));
    expect(validation.cubismEditorImport).toBe("not-tested");
    expect(validation.modelHashes.part_decoder).toBe("abc123def456");
    expect(validation.occlusionSummary.some((o: { id: string }) => o.id === "face")).toBe(true);
    expect(validation.provenanceSummary.model).toBeGreaterThanOrEqual(1);
  });
  it("supports dynamic multi-layer hair parts in Z-order and exports them into layered PSD", async () => {
    const p = fixture();
    const front2 = newHairPart("hair_front", p.parts, p.width, p.height);
    const sideL = newHairPart("hair_side_L", p.parts, p.width, p.height);
    for (let y = 10; y < 22; y++)
      for (let x = 36; x < 48; x++) {
        front2.mask[y * p.width + x] = 255;
        sideL.mask[(y + 12) * p.width + x] = 255;
      }
    let parts = insertHairPartInOrder(p.parts, sideL);
    parts = insertHairPartInOrder(parts, front2);
    const sideIdx = parts.findIndex((t) => t.id === sideL.id);
    const frontIdx = parts.findIndex((t) => t.id === front2.id);
    const faceIdx = parts.findIndex((t) => t.id === "face");
    expect(sideIdx).toBeGreaterThan(faceIdx);
    expect(frontIdx).toBeGreaterThan(sideIdx);
    const psd = readPsd(await exportPsd({ ...p, parts }), {
      useImageData: true,
      skipThumbnail: true,
    });
    const names = psd.children!.map((c) => c.name!);
    expect(names.some((n) => n.startsWith(`${front2.id} · `))).toBe(true);
    expect(names.some((n) => n.startsWith(`${sideL.id} · `))).toBe(true);
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
