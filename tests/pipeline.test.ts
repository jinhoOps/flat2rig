import { beforeAll, describe, expect, it, vi } from "vitest";
import { createCanvas, Image, ImageData } from "@napi-rs/canvas";
import { initializeCanvas, readPsd } from "ag-psd";
import { strFromU8, unzipSync, zipSync, strToU8 } from "fflate";
import {
  checkMeshQuality,
  compositePartTexture,
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
  mergePreservedParts,
  newDynamicPart,
  newHairPart,
  parseCsv,
  readCsvTemplate,
  saveProject,
} from "../src/io";
import {
  createDefaultProjectRigState,
  DEFAULT_MOUTH_HANDLES,
  PARAM_DEFS,
  REST,
  type Project,
} from "../src/types";
import {
  createCustomExpression,
  createUndoHistory,
  deleteExpressionPreset,
  saveNeutralParameters,
  selectExpressionPreset,
  updateShapePresetHandles,
} from "../src/rig/state";
import { createPhysicsState, evaluateRigFrame } from "../src/rig/evaluate";
import {
  buildAllPartMeshes,
  buildAllPartTextures,
  buildProjectSubstituteTextures,
  renderCharacterFrame,
  renderDeterministicSnapshot,
} from "../src/rig/render";
import { synthesizeShapeAssetsForProject } from "../src/rig/shapes";

function flattenPsdLayers<T extends { children?: T[] }>(layers?: T[]): T[] {
  const out: T[] = [];
  for (const layer of layers ?? []) {
    if (layer.children?.length) out.push(...flattenPsdLayers(layer.children));
    else out.push(layer);
  }
  return out;
}

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
    schemaVersion: 3,
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
    ...createDefaultProjectRigState("ld"),
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
  it("triangulates large curved torso masks cleanly without center collapse", () => {
    const w = 160,
      h = 200,
      mask = new Uint8Array(w * h);
    for (let y = 24; y < 184; y++) {
      const t = (y - 24) / 160;
      const cx = 80 + Math.sin(t * Math.PI) * 18;
      const halfW = 24 + t * 22;
      for (let x = 0; x < w; x++) {
        if (Math.abs(x - cx) <= halfW) mask[y * w + x] = 255;
      }
    }
    const mesh = meshFor(mask, w, h, 8);
    expect(mesh.triangles.length / 3).toBeGreaterThan(40);
    for (let i = 0; i < mesh.triangles.length; i += 3) {
      const a = mesh.vertices[mesh.triangles[i]],
        b = mesh.vertices[mesh.triangles[i + 1]],
        c = mesh.vertices[mesh.triangles[i + 2]];
      const cx = Math.floor((a.x + b.x + c.x) / 3),
        cy = Math.floor((a.y + b.y + c.y) / 3);
      expect(mask[cy * w + cx]).toBe(255);
    }
  });
  it("deforms expanded body, eye, brow, mouth, and hair parameters accurately", () => {
    const lm = defaultLandmarks(256, 256);
    const irisMoved = deform({ x: 150, y: 90 }, { id: "eyeL_iris" }, lm, {
      ...REST,
      eyeBallX: 1,
      eyeBallY: -1,
    });
    expect(irisMoved.x).toBeGreaterThan(150);
    expect(irisMoved.y).toBeGreaterThan(90);

    const browRaised = deform({ x: 150, y: 74 }, { id: "browL" }, lm, {
      ...REST,
      browLY: 1,
    });
    expect(browRaised.y).toBeLessThan(74);

    const mouthSmiled = deform({ x: 146, y: 120 }, { id: "mouth_upper" }, lm, {
      ...REST,
      mouthForm: 1,
    });
    expect(mouthSmiled.y).toBeLessThan(120);

    const shoulderPt = { x: 128, y: 200 };
    const breathed = deform(shoulderPt, { id: "clothes", role: "clothes" }, lm, {
      ...REST,
      breath: 1,
    });
    expect(breathed.y).toBeLessThan(shoulderPt.y);

    const bodyTurnedFace = deform({ x: 128, y: 96 }, { id: "face" }, lm, {
      ...REST,
      bodyAngleX: 10,
    });
    expect(bodyTurnedFace.x).toBeGreaterThan(128);

    const sideHair = deform({ x: 170, y: 130 }, { id: "hair_side_L", role: "hair_side_L" }, lm, {
      ...REST,
      hairSide: 1,
    });
    expect(sideHair.x).toBeGreaterThan(170);

    const backHair = deform({ x: 80, y: 150 }, { id: "hair_back", role: "hair_back" }, lm, {
      ...REST,
      hairBack: -1,
    });
    expect(backHair.x).toBeLessThan(80);
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
  it("round-trips every byte of masks, metadata, template fillKind, and generated texture", async () => {
    const p = fixture();
    p.styleSelection = { hairFrontStyle: "center_part", eyeStyle: "round_bright" };
    p.styleMatch = {
      resolved: { hairFrontStyle: "center_part", eyeStyle: "round_bright" },
      matchScore: 0.91,
      metrics: {
        faceRxRatio: 1.0,
        jawTaper: 0.68,
        eyeRxRatio: 0.28,
        eyeRyRatio: 0.14,
        irisRadiusRatio: 0.13,
        mouthWidthRatio: 0.16,
        mouthOpenRatio: 0.05,
        bangCoverage: 0.5,
        centerPartGap: 0.6,
        sideHairReach: 0.5,
        backHairReach: 0.9,
        shoulderWidthRatio: 1.7,
      },
      paletteSummary: { skin: "#f4dfd2" },
    };
    p.parts[0].generated = true;
    p.parts[0].fillKind = "template";
    p.parts[0].fillMask = new Uint8Array(p.parts[0].mask.length).fill(128);
    p.parts[0].texture = p.source;
    p.parts[1].visible = false;
    p.parts[2].status = "missing";
    const restored = loadProject(await saveProject(p));
    expect(restored.schemaVersion).toBe(3);
    expect(restored.name).toBe(p.name);
    expect(restored.source).toBe(p.source);
    expect(restored.landmarks).toEqual(p.landmarks);
    expect(restored.styleSelection).toEqual(p.styleSelection);
    expect(restored.styleMatch).toEqual(p.styleMatch);
    expect(restored.rigProfile).toEqual(p.rigProfile);
    expect(restored.shapePresets).toEqual(p.shapePresets);
    expect(restored.expressions).toEqual(p.expressions);
    expect(restored.activeExpressionId).toBe(p.activeExpressionId);
    expect(restored.playback).toEqual(p.playback);
    expect(restored.parts[0].fillKind).toBe("template");
    expect(
      restored.parts.map((part) => ({
        ...part,
        mask: Array.from(part.mask),
        fillMask: part.fillMask ? Array.from(part.fillMask) : undefined,
      })),
    ).toEqual(
      p.parts.map((part) => ({
        ...part,
        mask: Array.from(part.mask),
        fillMask: part.fillMask ? Array.from(part.fillMask) : undefined,
      })),
    );
  });
  it("migrates legacy schemaVersion 1 and 2 projects to v3 with defaults and landmarks", async () => {
    const v3 = fixture();
    const {
      landmarkProvenance: _lp,
      schemaVersion: _sv,
      rigProfile: _rp,
      shapePresets: _sp,
      expressions: _ex,
      activeExpressionId: _ae,
      playback: _pb,
      ...legacyBase
    } = v3;
    const legacyFiles = unzipSync(await saveProject(v3));
    const v1Meta = {
      ...legacyBase,
      schemaVersion: 1,
      parts: v3.parts.map(({ mask: _, provenance: _p, fillKind: _f, compositeOrder: _c, ...rest }) => rest),
    };
    legacyFiles["project.json"] = strToU8(JSON.stringify(v1Meta));
    const migratedV1 = loadProject(zipSync(legacyFiles));
    expect(migratedV1.schemaVersion).toBe(3);
    expect(migratedV1.landmarks).toEqual(v3.landmarks);
    expect(migratedV1.landmarkProvenance?.eyeL).toBe("manual");
    expect(migratedV1.parts.length).toBe(18);
    expect(migratedV1.parts[0].provenance).toBe("geometry");
    expect(migratedV1.rigProfile.defaultShapeIds.eye_L).toBe("eye_L_original");
    expect(migratedV1.expressions.length).toBeGreaterThanOrEqual(8);
    expect(migratedV1.playback.defaultExpressionId).toBe("neutral");

    const v2Meta = {
      ...legacyBase,
      landmarkProvenance: v3.landmarkProvenance,
      styleSelection: { proportionMode: "sd" as const },
      schemaVersion: 2,
      parts: v3.parts.map(({ mask: _, ...rest }) => rest),
    };
    legacyFiles["project.json"] = strToU8(JSON.stringify(v2Meta));
    const migratedV2 = loadProject(zipSync(legacyFiles));
    expect(migratedV2.schemaVersion).toBe(3);
    expect(migratedV2.rigProfile.proportionMode).toBe("sd");
    expect(migratedV2.shapePresets.length).toBeGreaterThanOrEqual(15);
  });
  it("round-trips and strictly validates ZIP shape substitute assets and v3 guards", async () => {
    const p = fixture();
    const bw = 16;
    const bh = 12;
    const spriteCanvas = createCanvas(bw, bh);
    const sg = spriteCanvas.getContext("2d");
    sg.fillStyle = "#4b2c3b";
    sg.fillRect(2, 2, 12, 8);
    const pngBuf = new Uint8Array(spriteCanvas.toBuffer("image/png"));
    const subMask = new Uint8Array(bw * bh).fill(255);
    p.shapeAssets = {
      asset_smile_l: {
        assetId: "asset_smile_l",
        region: "eye_L",
        recipeVersion: 1,
        hidesPartIds: ["eyeL_upper", "eyeL_white", "eyeL_iris"],
        layers: [
          {
            partId: "eyeL_closed_arc",
            blendMode: "normal",
            bounds: { left: 32, top: 14, width: bw, height: bh },
            mask: subMask,
            pngBytes: pngBuf,
          },
        ],
      },
    };
    const smilePreset = p.shapePresets.find((s) => s.id === "eye_L_smiling")!;
    smilePreset.assetId = "asset_smile_l";

    const packed = await saveProject(p);
    const zipEntries = unzipSync(packed);
    expect(zipEntries["shapes/asset_smile_l/eyeL_closed_arc.bin"]).toBeDefined();
    expect(zipEntries["shapes/asset_smile_l/eyeL_closed_arc.png"]).toBeDefined();

    const restored = loadProject(packed);
    expect(restored.shapeAssets?.asset_smile_l.layers[0].mask?.length).toBe(bw * bh);
    expect(restored.shapeAssets?.asset_smile_l.layers[0].pngBytes?.slice(0, 4)).toEqual(
      new Uint8Array([0x89, 0x50, 0x4e, 0x47]),
    );

    // Reject missing shape asset ref
    const brokenRef = unzipSync(packed);
    const metaBrokenRef = JSON.parse(strFromU8(brokenRef["project.json"]));
    metaBrokenRef.shapePresets[0].assetId = "missing_asset_id";
    brokenRef["project.json"] = strToU8(JSON.stringify(metaBrokenRef));
    expect(() => loadProject(zipSync(brokenRef))).toThrow("대체 에셋");

    // Reject out-of-bounds substitute bounds
    const badBounds = unzipSync(packed);
    const metaBadBounds = JSON.parse(strFromU8(badBounds["project.json"]));
    metaBadBounds.shapeAssets.asset_smile_l.layers[0].bounds = {
      left: 56,
      top: 14,
      width: 16,
      height: 12,
    };
    badBounds["project.json"] = strToU8(JSON.stringify(metaBadBounds));
    expect(() => loadProject(zipSync(badBounds))).toThrow("경계");

    // Reject wrong substitute mask size
    const badMask = unzipSync(packed);
    badMask["shapes/asset_smile_l/eyeL_closed_arc.bin"] = new Uint8Array(10);
    expect(() => loadProject(zipSync(badMask))).toThrow("마스크 크기");

    // Reject future schemaVersion
    const futureSchema = unzipSync(packed);
    const metaFuture = JSON.parse(strFromU8(futureSchema["project.json"]));
    metaFuture.schemaVersion = 99;
    futureSchema["project.json"] = strToU8(JSON.stringify(metaFuture));
    expect(() => loadProject(zipSync(futureSchema))).toThrow("형식");

    // Reject expression referencing missing shape ID
    const badExpRef = unzipSync(packed);
    const metaBadExp = JSON.parse(strFromU8(badExpRef["project.json"]));
    metaBadExp.expressions[0].shapeIds.mouth = "nonexistent_mouth";
    badExpRef["project.json"] = strToU8(JSON.stringify(metaBadExp));
    expect(() => loadProject(zipSync(badExpRef))).toThrow("쉐이프");
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
    expect(psd.children!.some((g) => g.name === "앞옆머리" && g.children?.length)).toBe(true);
    expect(psd.children!.some((g) => g.name === "얼굴" && g.children?.length)).toBe(true);
    expect(psd.children!.some((g) => g.name === "몸" && g.children?.length)).toBe(true);
    const leaves = flattenPsdLayers(psd.children);
    expect(leaves.length).toBeGreaterThan(5);
    expect(leaves.some((layer) => layer.name?.startsWith("hair_back"))).toBe(true);
    for (const layer of leaves) {
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
    const psdFace = flattenPsdLayers(psd.children).find((c) =>
      c.name?.startsWith("face ·"),
    )!;
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
    const names = flattenPsdLayers(psd.children).map((c) => c.name!);
    expect(names.some((n) => n.startsWith(`${front2.id} · `))).toBe(true);
    expect(names.some((n) => n.startsWith(`${sideL.id} · `))).toBe(true);
  });
  it("supports separated hat, glasses, accessory, and outerwear layers in Z-order, deform, and PSD", async () => {
    const p = fixture();
    const hat = newDynamicPart("hat", p.parts, p.width, p.height);
    const glasses = newDynamicPart("acc_glasses", p.parts, p.width, p.height);
    const hairpin = newDynamicPart("acc_hairpin", p.parts, p.width, p.height);
    const outer = newDynamicPart("clothes_outer", p.parts, p.width, p.height);
    for (let y = 10; y < 22; y++)
      for (let x = 28; x < 44; x++) {
        hat.mask[y * p.width + x] = 255;
        glasses.mask[(y + 10) * p.width + x] = 255;
        hairpin.mask[(y + 4) * p.width + x] = 255;
        outer.mask[(y + 34) * p.width + x] = 255;
      }
    let parts = insertHairPartInOrder(p.parts, outer);
    parts = insertHairPartInOrder(parts, glasses);
    parts = insertHairPartInOrder(parts, hat);
    parts = insertHairPartInOrder(parts, hairpin);
    const glassesIdx = parts.findIndex((t) => t.id === glasses.id);
    const frontIdx = parts.findIndex((t) => t.id === "hair_front");
    const hatIdx = parts.findIndex((t) => t.id === hat.id);
    const outerIdx = parts.findIndex((t) => t.id === outer.id);
    const faceIdx = parts.findIndex((t) => t.id === "face");
    expect(outerIdx).toBeLessThan(faceIdx);
    expect(glassesIdx).toBeLessThan(frontIdx);
    expect(hatIdx).toBeGreaterThan(frontIdx);

    const glassBlink = deform(
      { x: 38, y: 24 },
      { id: "acc_glasses", role: "accessory" },
      p.landmarks,
      { ...REST, eyeL: 0, hair: 1 },
    );
    expect(glassBlink.x).toBeCloseTo(38, 6);
    expect(glassBlink.y).toBeCloseTo(24, 6);
    const pinSway = deform(
      { x: 40, y: 26 },
      { id: "acc_hairpin", role: "accessory" },
      p.landmarks,
      { ...REST, hair: 1 },
    );
    expect(pinSway.x).toBeGreaterThan(40);

    const psd = readPsd(await exportPsd({ ...p, parts }), {
      useImageData: true,
      skipThumbnail: true,
    });
    const names = flattenPsdLayers(psd.children).map((c) => c.name!);
    expect(names.some((n) => n.startsWith("hat · "))).toBe(true);
    expect(names.some((n) => n.startsWith("acc_glasses · "))).toBe(true);
    expect(names.some((n) => n.startsWith("clothes_outer · "))).toBe(true);
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
    expect(rows[1][4]).toBe("ParamGroup");
    expect(rows[1][5]).toBe("-30");
    expect(rows.length).toBe(PARAM_DEFS.length + 1);
  });
  it("rejects arbitrary CSV rather than claiming Cubism compatibility", () => {
    expect(() => readCsvTemplate("ID,Name\nParamAngleX,Head")).toThrow();
    expect(() => parseCsv('a,"not closed')).toThrow();
  });
});

describe("preserved part merging on regeneration", () => {
  it("keeps approved, manual, and extra dynamic hair parts without mask overlap", () => {
    const p = fixture();
    const face = p.parts.find((t) => t.id === "face")!;
    const hairFront = p.parts.find((t) => t.id === "hair_front")!;
    face.status = "approved";
    const manualStrand = newHairPart("hair_front", p.parts, p.width, p.height);
    manualStrand.mask[10 * p.width + 20] = 255;
    manualStrand.mask[10 * p.width + 21] = 255;
    hairFront.mask[10 * p.width + 20] = 0;
    hairFront.mask[10 * p.width + 21] = 0;
    // Also mark an existing standard part as manually edited (`provenance: "manual"`, status still draft)
    const manualFaceMaskVal = face.mask[32 * p.width + 32];
    const nextParts = fixture().parts.map((t) => ({ ...t, mask: t.mask.slice() }));
    const nextBody = nextParts.find((t) => t.id === "body")!;
    // Simulate regenerated draft trying to claim the same pixels as manualStrand & hairFront
    nextBody.mask[10 * p.width + 20] = 255;
    hairFront.provenance = "manual";
    hairFront.mask[12 * p.width + 22] = 255;
    const nextFront = nextParts.find((t) => t.id === "hair_front")!;
    nextFront.mask.fill(0);
    nextBody.mask[12 * p.width + 22] = 255;

    const prevWithExtra = insertHairPartInOrder(p.parts, manualStrand);
    const { parts: merged, preservedCount } = mergePreservedParts(prevWithExtra, nextParts);
    expect(preservedCount).toBe(3);
    const keptExtra = merged.find((t) => t.id === manualStrand.id);
    expect(keptExtra).toBeDefined();
    expect(merged.find((t) => t.id === "face")!.mask[32 * p.width + 32]).toBe(manualFaceMaskVal);
    expect(merged.find((t) => t.id === "hair_front")!.mask[12 * p.width + 22]).toBe(255);
    // Clipped from non-preserved body so overlap never occurs
    const mergedBody = merged.find((t) => t.id === "body")!;
    expect(mergedBody.mask[10 * p.width + 20]).toBe(0);
    expect(mergedBody.mask[12 * p.width + 22]).toBe(0);
  });

  it("preserves Click Prompt (`prompted`) edits and clears overlapping fill masks cleanly in Z-order", () => {
    const p = fixture();
    const hat = newDynamicPart("hat", p.parts, p.width, p.height);
    const baseWithHat = insertHairPartInOrder(p.parts, hat);
    const neck = baseWithHat.find((t) => t.id === "neck")!;
    neck.provenance = "prompted";
    neck.status = "draft";
    neck.mask[15 * p.width + 32] = 255;

    const extraFront = newHairPart("hair_front", baseWithHat, p.width, p.height);
    const withExtraFront = insertHairPartInOrder(baseWithHat, extraFront);
    const frontIdx = withExtraFront.findIndex((t) => t.id === "hair_front");
    const extraIdx = withExtraFront.findIndex((t) => t.id === extraFront.id);
    const hatIdx = withExtraFront.findIndex((t) => t.id === "hat");
    expect(extraIdx).toBe(frontIdx + 1);
    expect(hatIdx).toBeGreaterThan(extraIdx);

    const regen = fixture().parts.map((t) => ({ ...t, mask: t.mask.slice() }));
    const regenBody = regen.find((t) => t.id === "body")!;
    regenBody.mask[15 * p.width + 32] = 255;
    const { parts: merged, preservedCount } = mergePreservedParts(withExtraFront, regen);
    expect(preservedCount).toBe(3);
    expect(merged.some((t) => t.id === hat.id)).toBe(true);
    expect(merged.find((t) => t.id === "neck")!.mask[15 * p.width + 32]).toBe(255);
    expect(merged.find((t) => t.id === "body")!.mask[15 * p.width + 32]).toBe(0);
    expect(merged.some((t) => t.id === extraFront.id)).toBe(true);
    expect(merged.findIndex((t) => t.id === "neck")).toBe(
      merged.findIndex((t) => t.id === "body") + 1,
    );
  });

  it("composites expanded manual/prompt masks using sourcePixels without dropping occlusion fill", () => {
    const source = new Uint8ClampedArray([
      200, 100, 50, 255, // pixel 0: newly brushed visible pixel outside stale texture
      40, 80, 120, 255, // pixel 1: hidden occlusion fill pixel
    ]);
    const staleTexture = new Uint8ClampedArray([
      0, 0, 0, 0, // stale texture lacked newly painted visible pixel
      90, 95, 110, 255, // generated fill pixel
    ]);
    const out = compositePartTexture(
      {
        mask: new Uint8Array([255, 0]),
        fillMask: new Uint8Array([0, 200]),
        fillKind: "template",
      },
      source,
      staleTexture,
    );
    expect(Array.from(out.slice(0, 4))).toEqual([200, 100, 50, 255]);
    expect(Array.from(out.slice(4, 8))).toEqual([90, 95, 110, 200]);

    const extOut = compositePartTexture(
      {
        mask: new Uint8Array([255, 0]),
        fillMask: new Uint8Array([0, 200]),
        fillKind: "external",
      },
      source,
      staleTexture,
    );
    expect(Array.from(extOut.slice(0, 4))).toEqual([0, 0, 0, 0]);
  });

  it("keeps closed eye_upper lids monotonic without outer corner droop onto cheek", () => {
    const lm = defaultLandmarks(64, 64);
    const dist = Math.hypot(lm.eyeL.x - lm.eyeR.x, lm.eyeL.y - lm.eyeR.y);
    const anchor = { x: lm.eyeL.x, y: lm.eyeL.y - dist * 0.34 };
    const lashPeak = { x: lm.eyeL.x, y: lm.eyeL.y - dist * 0.16 };
    const cornerTip = { x: lm.eyeL.x + dist * 0.22, y: lm.eyeL.y - dist * 0.02 };

    // Rest pose: zero displacement
    expect(deform(lashPeak, { id: "eyeL_upper" }, lm, REST).y).toBeCloseTo(lashPeak.y, 6);

    const closedPose = { ...REST, eyeL: 0 };
    const dAnchor = deform(anchor, { id: "eyeL_upper" }, lm, closedPose);
    const dPeak = deform(lashPeak, { id: "eyeL_upper" }, lm, closedPose);
    const dCorner = deform(cornerTip, { id: "eyeL_upper" }, lm, closedPose);
    expect(dAnchor.y).toBeCloseTo(anchor.y, 6);
    expect(dPeak.y).toBeGreaterThan(lashPeak.y + dist * 0.10);
    expect(dCorner.y).toBeGreaterThanOrEqual(dPeak.y);
    expect(dCorner.y).toBeLessThanOrEqual(lm.eyeL.y + dist * 0.04 + 1e-6);
  });

  it("builds role-aware Cubism ArtMesh topology without orphan vertices or collapsing thin parts", () => {
    const w = 64;
    const h = 64;
    const lm = defaultLandmarks(w, h);
    const irisMask = new Uint8Array(w * h);
    const cx = Math.round(lm.eyeL.x);
    const cy = Math.round(lm.eyeL.y);
    for (let y = cy - 8; y <= cy + 8; y++) {
      for (let x = cx - 8; x <= cx + 8; x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= 49) irisMask[y * w + x] = 255;
      }
    }
    const irisMesh = meshForPart(irisMask, w, h, 14, lm, "eyeL_iris");
    expect(irisMesh.triangles.length).toBeGreaterThan(18);
    expect(irisMesh.uv.length).toBe(irisMesh.vertices.length);
    const used = new Array(irisMesh.vertices.length).fill(false);
    for (const idx of irisMesh.triangles) used[idx] = true;
    expect(used.every(Boolean)).toBe(true);
    expect(
      irisMesh.vertices.some(
        (v) => Math.abs(v.x - (cx + 0.5)) < 1.2 && Math.abs(v.y - (cy + 0.5)) < 1.2,
      ),
    ).toBe(true);

    const thinPin = new Uint8Array(w * h);
    for (let x = 18; x < 42; x++) {
      const y = 14 + Math.floor(Math.abs(x - 30) / 6);
      thinPin[y * w + x] = 255;
      thinPin[(y + 1) * w + x] = 255;
    }
    const pinMesh = meshForPart(thinPin, w, h, 14, lm, "acc_hairpin");
    const report = checkMeshQuality(pinMesh, thinPin, w, h, { id: "acc_hairpin" }, lm);
    expect(report.triangleCount).toBeGreaterThanOrEqual(6);
    expect(report.holeCrossings).toBe(0);
    expect(report.flippedAtExtremes).toBe(0);
  });

  it("persists restricted blendModes and refuses silent coercion to normal during PSD export", async () => {
    const project = fixture();
    const face = project.parts.find((p) => p.id === "face")!;
    face.blendMode = "multiply";

    const reloaded = loadProject(await saveProject(project));
    expect(reloaded.parts.find((p) => p.id === "face")?.blendMode).toBe("multiply");

    await expect(exportPsd(reloaded)).rejects.toThrow(/normal/);

    const badZip = unzipSync(await saveProject(project));
    const json = JSON.parse(strFromU8(badZip["project.json"]));
    json.parts[0].blendMode = "overlay";
    badZip["project.json"] = strToU8(JSON.stringify(json));
    expect(() => loadProject(zipSync(badZip))).toThrow(/잘못된 파츠 정보/);
  });

  it("refuses silent success when already-preserved parts overlap each other during merge", () => {
    const p = fixture();
    const a = p.parts.find((t) => t.id === "face")!;
    const b = p.parts.find((t) => t.id === "hair_front")!;
    a.status = "approved";
    b.provenance = "manual";
    const overlapIdx = 12 * p.width + 28;
    a.mask[overlapIdx] = 255;
    b.mask[overlapIdx] = 200;

    expect(() => mergePreservedParts(p.parts, p.parts)).toThrow(/겹쳐 있어/);
  });

  it("batches shape/expression drags into single undo/redo steps and protects built-in defaults", () => {
    const p0 = fixture();
    const history = createUndoHistory(p0);
    history.beginDrag();
    const pMid = updateShapePresetHandles(p0, "eye_L_half_closed", { open: 0.38 }, true);
    history.previewDrag(pMid);
    const pFinal = updateShapePresetHandles(pMid, "eye_L_half_closed", { open: 0.3 }, true);
    history.previewDrag(pFinal);
    const afterCommit = history.commitDrag();

    expect(afterCommit.past.length).toBe(1);
    expect(afterCommit.current.shapePresets.find((s) => s.id === "eye_L_half_closed")?.handles.open).toBeCloseTo(0.3, 6);
    expect(afterCommit.current.shapePresets.find((s) => s.id === "eye_R_half_closed")?.handles.open).toBeCloseTo(0.3, 6);

    const undone = history.undo();
    expect(undone.current.shapePresets.find((s) => s.id === "eye_L_half_closed")?.handles.open).toBeCloseTo(0.45, 6);
    expect(undone.future.length).toBe(1);
    const redone = history.redo();
    expect(redone.current.shapePresets.find((s) => s.id === "eye_L_half_closed")?.handles.open).toBeCloseTo(0.3, 6);

    const { project: withCustom, expression: custom } = createCustomExpression(
      redone.current,
      "커스텀 윙크",
      "wink",
    );
    expect(withCustom.activeExpressionId).toBe(custom.id);
    // Neutral built-in cannot be deleted
    expect(deleteExpressionPreset(withCustom, "neutral").expressions.length).toBe(
      withCustom.expressions.length,
    );
    const afterDelete = deleteExpressionPreset(withCustom, custom.id);
    expect(afterDelete.activeExpressionId).toBe("neutral");
    expect(afterDelete.expressions.length).toBe(p0.expressions.length);

    const savedDefaults = saveNeutralParameters(afterDelete, { ...REST, angleX: 6 });
    expect(savedDefaults.rigProfile.neutralParameters.angleX).toBe(6);
  });

  it("rejects mismatched PNG dimensions and excessive preset counts in v3 ZIPs", async () => {
    const p = fixture();
    const bw = 12;
    const bh = 10;
    const wrongPng = new Uint8Array(createCanvas(8, 8).toBuffer("image/png"));
    p.shapeAssets = {
      sub_a: {
        assetId: "sub_a",
        region: "mouth",
        recipeVersion: 1,
        hidesPartIds: ["mouth_inner"],
        layers: [
          {
            partId: "mouth_sub",
            bounds: { left: 20, top: 20, width: bw, height: bh },
            mask: new Uint8Array(bw * bh).fill(255),
            pngBytes: wrongPng,
          },
        ],
      },
    };
    const packed = await saveProject(p);
    expect(() => loadProject(packed)).toThrow(/PNG/);

    const clean = fixture();
    const zipEntries = unzipSync(await saveProject(clean));
    const json = JSON.parse(strFromU8(zipEntries["project.json"]));
    json.expressions = Array.from({ length: 200 }, (_, i) => ({
      ...json.expressions[0],
      id: `exp_${i}`,
      name: `표정 ${i}`,
    }));
    zipEntries["project.json"] = strToU8(JSON.stringify(json));
    expect(() => loadProject(zipSync(zipEntries))).toThrow(/표정 프리셋/);
  });

  it("evaluates rig frames in fixed stage order without auto mouth opening or blink override on wink", () => {
    const p = fixture();
    const winkProj = selectExpressionPreset(p, "wink");

    // 1. Default playback does NOT automatically open mouth unless autoMouthDemo === true
    let phys = createPhysicsState();
    for (let t = 0; t <= 4.5; t += 0.35) {
      const frame = evaluateRigFrame({
        project: p,
        timeSec: t,
        playing: true,
        physicsState: phys,
      });
      phys = frame.nextPhysicsState;
      expect(frame.parameters.mouth).toBe(0);
    }

    // 2. Auto-blink hits peak around phase = 0.09s (since phase = (t + period*0.5) % period):
    // period = 3.6 -> blink center at t = 1.89s -> eyeR blinks, while winking eyeL stays closed!
    const blinkFrame = evaluateRigFrame({
      project: winkProj,
      timeSec: 1.89,
      playing: true,
    });
    expect(blinkFrame.eyeClosedOrSmiling.L).toBe(true);
    expect(blinkFrame.eyeClosedOrSmiling.R).toBe(false);
    expect(blinkFrame.parameters.eyeL).toBe(0);
    expect(blinkFrame.parameters.eyeR).toBeLessThan(0.2);

    // Character-relative Right Wink (`wink_r`) keeps right eye closed & left eye blinking naturally
    const winkRProj = selectExpressionPreset(p, "wink_r");
    const blinkRFrame = evaluateRigFrame({
      project: winkRProj,
      timeSec: 1.89,
      playing: true,
    });
    expect(blinkRFrame.eyeClosedOrSmiling.R).toBe(true);
    expect(blinkRFrame.eyeClosedOrSmiling.L).toBe(false);
    expect(blinkRFrame.parameters.eyeR).toBe(0);
    expect(blinkRFrame.parameters.eyeL).toBeLessThan(0.2);

    // All 8 built-in expressions exist
    expect(p.expressions.filter((e) => e.builtIn).map((e) => e.id)).toEqual([
      "neutral",
      "smile",
      "happy",
      "angry",
      "sad",
      "surprised",
      "wink",
      "wink_r",
    ]);

    // 3. Pause / deterministic / backward seek resets physics state to 0
    const movingFrame = evaluateRigFrame({
      project: p,
      timeSec: 1.0,
      playing: true,
      physicsState: createPhysicsState(),
    });
    const pausedFrame = evaluateRigFrame({
      project: p,
      timeSec: 1.0,
      playing: false,
      physicsState: movingFrame.nextPhysicsState,
    });
    expect(pausedFrame.nextPhysicsState.position).toBe(0);
    expect(pausedFrame.nextPhysicsState.velocity).toBe(0);

    // 4. Priority order: expression -> user offset -> safety clamp
    const clampedFrame = evaluateRigFrame({
      project: selectExpressionPreset(p, "happy"),
      userOffsets: { angleX: 90 },
      deterministic: true,
    });
    expect(clampedFrame.parameters.angleX).toBe(
      p.rigProfile.safeBounds.angleX!.max,
    );
    expect(clampedFrame.parameters.mouthForm).toBeGreaterThan(0.5);
  });

  it("renders deterministic snapshots with identical pixels across calls and non-empty transparent bounds", async () => {
    const p = fixture();
    const img = await new Promise<InstanceType<typeof Image>>((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = reject;
      im.src = p.source;
    });
    const tmp = createCanvas(p.width, p.height);
    tmp.getContext("2d").drawImage(img, 0, 0);
    const px = new Uint8ClampedArray(tmp.getContext("2d").getImageData(0, 0, p.width, p.height).data);

    const textures = buildAllPartTextures(p, px);
    const meshes = buildAllPartMeshes(p);

    const snapNeutral = renderDeterministicSnapshot(
      { project: selectExpressionPreset(p, "neutral") },
      textures,
      meshes,
    );
    const dataNeutral = (snapNeutral.canvas as unknown as ReturnType<typeof createCanvas>)
      .getContext("2d")
      .getImageData(0, 0, p.width, p.height).data;
    const snapA = renderDeterministicSnapshot(
      { project: selectExpressionPreset(p, "smile") },
      textures,
      meshes,
    );
    const snapB = renderDeterministicSnapshot(
      { project: selectExpressionPreset(p, "smile") },
      textures,
      meshes,
    );
    const dataA = (snapA.canvas as unknown as ReturnType<typeof createCanvas>)
      .getContext("2d")
      .getImageData(0, 0, p.width, p.height).data;
    const dataB = (snapB.canvas as unknown as ReturnType<typeof createCanvas>)
      .getContext("2d")
      .getImageData(0, 0, p.width, p.height).data;
    expect(dataA).toEqual(dataB);
    // Corners must stay transparent (alpha === 0)
    expect(dataA[3]).toBe(0);
    // Center character face must be well populated and body interior fully opaque
    expect(dataA[(22 * p.width + 32) * 4 + 3]).toBeGreaterThan(128);
    expect(dataA[(48 * p.width + 32) * 4 + 3]).toBe(255);
    // Smile expression produces measurable pixel difference from neutral
    let changedPx = 0;
    for (let i = 0; i < dataA.length; i += 4) {
      if (
        Math.abs(dataA[i] - dataNeutral[i]) +
          Math.abs(dataA[i + 1] - dataNeutral[i + 1]) +
          Math.abs(dataA[i + 2] - dataNeutral[i + 2]) +
          Math.abs(dataA[i + 3] - dataNeutral[i + 3]) >
        12
      ) {
        changedPx++;
      }
    }
    expect(changedPx).toBeGreaterThan(15);
  });

  it("synthesizes distinct T3 eye and T4 mouth shape assets, preserves across ZIP round-trip, and handles blink/interior alpha", async () => {
    const width = 128;
    const height = 128;
    const landmarks = defaultLandmarks(width, height);
    const c = createCanvas(width, height);
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#b8a4e5";
    ctx.fillRect(16, 6, 96, 118);
    ctx.fillStyle = "#f9dcd0";
    ctx.fillRect(32, 18, 64, 54);
    // Draw distinct irises & mouth right at landmarks so T3/T4 covers and vowel shapes differ clearly
    for (const pt of [landmarks.eyeR, landmarks.eyeL]) {
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.ellipse(pt.x, pt.y, 7, 5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#2b1d2e";
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#c95b6b";
    ctx.fillRect(landmarks.mouth.x - 6, landmarks.mouth.y - 2, 12, 4);

    const data = new Uint8ClampedArray(ctx.getImageData(0, 0, width, height).data);
    const base: Project = {
      schemaVersion: 3,
      name: "모양 테스트",
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
      revision: 1,
      provider: "geometry-template-v1",
      ...createDefaultProjectRigState("ld"),
      createdAt: "2026-11-15T00:00:00Z",
    };

    const synth = synthesizeShapeAssetsForProject(base, data, true);
    const expectedAssetIds = [
      "eye_L_closed_v1",
      "eye_L_smiling_v1",
      "eye_R_closed_v1",
      "eye_R_smiling_v1",
      "mouth_closed_v1",
      "mouth_a_v1",
      "mouth_i_v1",
      "mouth_u_v1",
      "mouth_e_v1",
      "mouth_o_v1",
    ];
    for (const id of expectedAssetIds) {
      const asset = synth.shapeAssets?.[id];
      expect(asset).toBeDefined();
      expect(asset!.layers.length).toBeGreaterThanOrEqual(2);
      for (const layer of asset!.layers) {
        expect(layer.pngBytes?.length ?? 0).toBeGreaterThan(20);
      }
    }

    // ZIP save/load round-trip preserves shapeAssets and preset assetId links
    const reloaded = await loadProject(await saveProject(synth));
    for (const id of expectedAssetIds) {
      expect(reloaded.shapeAssets?.[id]?.layers.length).toBe(
        synth.shapeAssets?.[id]?.layers.length,
      );
      expect(reloaded.shapeAssets?.[id]?.layers[0]?.pngBytes).toEqual(
        synth.shapeAssets?.[id]?.layers[0]?.pngBytes,
      );
    }

    const partTextures = buildAllPartTextures(reloaded, data);
    const meshes = buildAllPartMeshes(reloaded);
    const subTextures = await buildProjectSubstituteTextures(reloaded);

    const diffCount = (a: Uint8ClampedArray, b: Uint8ClampedArray) => {
      let changed = 0;
      for (let i = 0; i < a.length; i += 4) {
        if (
          Math.abs(a[i] - b[i]) +
            Math.abs(a[i + 1] - b[i + 1]) +
            Math.abs(a[i + 2] - b[i + 2]) +
            Math.abs(a[i + 3] - b[i + 3]) >
          14
        ) {
          changed++;
        }
      }
      return changed;
    };

    const renderEye = (eyeId: string) => {
      const snap = renderDeterministicSnapshot(
        {
          project: selectExpressionPreset(reloaded, "neutral"),
          overrideShapeIds: { eye_L: eyeId, eye_R: eyeId.replace("eye_L", "eye_R") },
        },
        partTextures,
        meshes,
        subTextures,
      );
      return (snap.canvas as unknown as ReturnType<typeof createCanvas>)
        .getContext("2d")
        .getImageData(0, 0, width, height).data;
    };
    const eyeImgs = ["eye_L_original", "eye_L_half_closed", "eye_L_closed", "eye_L_smiling"].map(
      renderEye,
    );
    for (let i = 0; i < eyeImgs.length; i++) {
      for (let j = i + 1; j < eyeImgs.length; j++) {
        expect(diffCount(eyeImgs[i], eyeImgs[j])).toBeGreaterThan(20);
      }
    }

    const renderMouth = (mouthId: string) => {
      const snap = renderDeterministicSnapshot(
        {
          project: selectExpressionPreset(reloaded, "neutral"),
          overrideShapeIds: { mouth: mouthId },
        },
        partTextures,
        meshes,
        subTextures,
      );
      return (snap.canvas as unknown as ReturnType<typeof createCanvas>)
        .getContext("2d")
        .getImageData(0, 0, width, height).data;
    };
    const mouthIds = [
      "mouth_original",
      "mouth_closed",
      "mouth_a",
      "mouth_i",
      "mouth_u",
      "mouth_e",
      "mouth_o",
    ];
    const mouthImgs = mouthIds.map(renderMouth);
    for (let i = 0; i < mouthImgs.length; i++) {
      for (let j = i + 1; j < mouthImgs.length; j++) {
        expect(diffCount(mouthImgs[i], mouthImgs[j])).toBeGreaterThan(15);
      }
    }

    // Auto-blink swaps eye_L to closed asset when pulse drops eyeL <= 0.22
    const autoBlinkProj: Project = {
      ...reloaded,
      playback: { ...reloaded.playback, autoBlink: true, blinkIntervalSec: 3.6, autoBreath: false, autoIdle: false },
    };
    const blinkEval = evaluateRigFrame({
      project: autoBlinkProj,
      timeSec: 1.89,
      playing: true,
    });
    expect(blinkEval.parameters.eyeL).toBeLessThanOrEqual(0.22);
    expect(blinkEval.activeShapes.eye_L?.assetId).toBe("eye_L_closed_v1");

    // DEFAULT_MOUTH_HANDLES reveals mouthInteriorAlpha only when mouth opens (> 0.03), 0 at rest / mouth_closed
    expect(DEFAULT_MOUTH_HANDLES.interiorVisibility).toBe(1);
    const restMouthEval = evaluateRigFrame({ project: reloaded, deterministic: true });
    expect(restMouthEval.mouthInteriorAlpha).toBe(0);
    const openOriginalMouthEval = evaluateRigFrame({
      project: reloaded,
      userOffsets: { mouth: 1 },
      deterministic: true,
    });
    expect(openOriginalMouthEval.mouthInteriorAlpha).toBeGreaterThan(0.5);
    const closedPresetEval = evaluateRigFrame({
      project: reloaded,
      overrideShapeIds: { mouth: "mouth_closed" },
      userOffsets: { mouth: 1 },
      deterministic: true,
    });
    expect(closedPresetEval.mouthInteriorAlpha).toBe(0);

    // enableGpu gracefully falls back to Canvas2D in headless/Node environments without WebGL
    const fallbackCanvas = createCanvas(width, height);
    const fallbackScratch = renderCharacterFrame({
      ctx: fallbackCanvas.getContext("2d") as unknown as CanvasRenderingContext2D,
      project: reloaded,
      evaluated: evaluateRigFrame({ project: selectExpressionPreset(reloaded, "smile"), deterministic: true }),
      partTextures,
      meshes,
      substituteTextures: subTextures,
      previewDeform: true,
      enableGpu: true,
    });
    expect(fallbackScratch.gl).toBeNull();
    const cpuSmile = renderDeterministicSnapshot(
      { project: selectExpressionPreset(reloaded, "smile") },
      partTextures,
      meshes,
      subTextures,
    );
    const a = fallbackCanvas.getContext("2d").getImageData(0, 0, width, height).data;
    const b = (cpuSmile.canvas as unknown as ReturnType<typeof createCanvas>)
      .getContext("2d")
      .getImageData(0, 0, width, height).data;
    expect(diffCount(a, b)).toBe(0);
  });
});
