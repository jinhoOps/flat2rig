import { beforeAll, describe, expect, it, vi } from "vitest";
import { createCanvas, Image, ImageData } from "@napi-rs/canvas";
import { initializeCanvas, readPsd } from "ag-psd";
import { strFromU8, unzipSync, zipSync, strToU8 } from "fflate";
import {
  checkMeshQuality,
  cleanSubtractedSiblingMask,
  compositePartTexture,
  deform,
  deformUncoupled,
  defaultLandmarks,
  effectiveMask,
  expandFillSeparated,
  foregroundMask,
  meshFor,
  meshForPart,
  paint,
  maxSeamRelativeDisplacement,
  partOverlapMargins,
  recommendedFillRadius,
  seamCouplingWeights,
  separate,
} from "../src/engine";
import {
  cubismCsv,
  exportBundle,
  exportPsd,
  insertHairPartInOrder,
  loadProject,
  isHairStageLayer,
  mergePreservedParts,
  mergeStage2HairParts,
  mergeStageParts,
  newDynamicPart,
  newHairPart,
  parseCsv,
  partPipelineStage,
  readCsvTemplate,
  saveProject,
  stageIncludesLayer,
} from "../src/io";
import {
  createDefaultProjectRigState,
  DEFAULT_MOUTH_HANDLES,
  describeDynamicPart,
  inferPartRole,
  PARAM_DEFS,
  REST,
  type Landmarks,
  type PartRole,
  type Project,
} from "../src/types";
import fillRadiusParity from "./fixtures/fill-radius-parity.json";
import seamEnvelopeParity from "./fixtures/seam-envelope-parity.json";
import { POSE_ENVELOPE, POSE_ENVELOPE_RANGES } from "../src/rig/poseEnvelope";
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
  it("T13b couples the collar to the neck base only (never the head) while torso centre and shoulders stay put", () => {
    const lm: Landmarks = {
      eyeR: { x: 400, y: 360 },
      eyeL: { x: 624, y: 360 },
      mouth: { x: 512, y: 492 },
      chin: { x: 512, y: 560 },
      crown: { x: 512, y: 120 },
    };
    const pose = { ...REST, angleX: 30, angleY: -25, angleZ: 20, bodyAngleZ: -10, hair: 1 };
    // Collar inside the neck column: clothes and neck move closer (relative displacement shrinks).
    const collar = { x: 512, y: 645 };
    const c = deform(collar, { id: "clothes", role: "clothes" }, lm, pose);
    const n = deform(collar, { id: "neck" }, lm, pose);
    const cRaw = deformUncoupled(collar, { id: "clothes", role: "clothes" }, lm, pose);
    const nRaw = deformUncoupled(collar, { id: "neck" }, lm, pose);
    const relRaw = Math.hypot(cRaw.x - nRaw.x, cRaw.y - nRaw.y);
    expect(relRaw).toBeGreaterThan(5);
    expect(Math.hypot(c.x - n.x, c.y - n.y)).toBeLessThan(relRaw * 0.6);
    // Clothes right under the chin never inherit the neck-top ↔ head term: they stay (almost) on
    // the body field instead of turning with the head.
    const underChin = { x: 512, y: 566 };
    const w = seamCouplingWeights(underChin, { id: "clothes", role: "clothes" }, lm);
    expect(w.neckHead).toBe(0);
    expect(w.collar).toBeLessThan(0.01);
    const cu = deform(underChin, { id: "clothes", role: "clothes" }, lm, pose);
    const cuRaw = deformUncoupled(underChin, { id: "clothes", role: "clothes" }, lm, pose);
    expect(Math.hypot(cu.x - cuRaw.x, cu.y - cuRaw.y)).toBeLessThan(0.5);
    // Neck classification is role based: an explicit-role neck layer couples like `neck`.
    const custom = seamCouplingWeights({ x: 512, y: 560 }, { id: "neck_inner", role: "neck" }, lm);
    expect(custom.neckHead).toBe(1);
    // Jaw line: the neck top follows the head (face) field.
    const jaw = { x: 512, y: 560 };
    const f = deform(jaw, { id: "face" }, lm, pose);
    const nj = deform(jaw, { id: "neck" }, lm, pose);
    expect(Math.hypot(f.x - nj.x, f.y - nj.y)).toBeLessThan(1e-6);
    // Far from the seams nothing changes: torso centre, shoulders, face centre, hair tips.
    for (const [id, p] of [
      ["clothes", { x: 512, y: 800 }],
      ["clothes", { x: 330, y: 700 }],
      ["body", { x: 694, y: 720 }],
      ["face", { x: 512, y: 400 }],
      ["hair_side_R", { x: 330, y: 715 }],
      ["hair_front", { x: 510, y: 324 }],
    ] as const) {
      expect(deform(p, { id }, lm, pose), id).toEqual(deformUncoupled(p, { id }, lm, pose));
    }
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
    const anchor = { x: lm.eyeL.x, y: lm.eyeL.y - dist * 0.45 };
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

  it("synthesizes HeadFrame mouth shape assets, drives blink/smile/sad via real upper eyelid mesh, and round-trips ZIP", async () => {
    const width = 128;
    const height = 128;
    const landmarks = defaultLandmarks(width, height);
    const c = createCanvas(width, height);
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#b8a4e5";
    ctx.fillRect(16, 6, 96, 118);
    ctx.fillStyle = "#f9dcd0";
    ctx.fillRect(32, 18, 64, 54);
    // Draw distinct irises, real dark upper lash arcs, & mouth right at landmarks
    for (const pt of [landmarks.eyeR, landmarks.eyeL]) {
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.ellipse(pt.x, pt.y, 7, 5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#2b1d2e";
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#1f1524";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(pt.x - 8, pt.y - 4);
      ctx.quadraticCurveTo(pt.x, pt.y - 7, pt.x + 8, pt.y - 4);
      ctx.stroke();
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
    // Eye closed/smiling presets stay driven by real upper eyelid meshes (no synthetic sticker replacement)
    expect(synth.shapeAssets?.["eye_L_closed_v1"]).toBeUndefined();
    expect(synth.shapeAssets?.["eye_L_smiling_v1"]).toBeUndefined();
    expect(synth.shapePresets.find((s) => s.id === "eye_L_closed")?.assetId).toBeUndefined();
    expect(synth.shapePresets.find((s) => s.id === "eye_L_smiling")?.assetId).toBeUndefined();

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

    // Auto-blink animates real upper lid + fades sclera/iris/lower alpha cleanly when eyeL <= 0.22
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
    expect(blinkEval.hiddenBasePartIds.has("eyeL_upper")).toBe(false);
    expect(blinkEval.eyeLowerAlpha.L).toBeLessThan(0.25);

    // Sad expression uses closed eye + negative upperCurveDelta (u curve), Smile uses positive ^ arch
    const sadEval = evaluateRigFrame({
      project: selectExpressionPreset(reloaded, "sad"),
      deterministic: true,
    });
    expect(sadEval.parameters.eyeL).toBe(0);
    expect(sadEval.eyeWhiteAlpha.L).toBe(0);
    expect(sadEval.eyeIrisAlpha.L).toBe(0);
    expect(sadEval.shapeDeform.eyeL?.upperCurveDelta ?? 0).toBeLessThan(-0.8);

    const smileEval = evaluateRigFrame({
      project: selectExpressionPreset(reloaded, "smile"),
      deterministic: true,
    });
    expect(smileEval.parameters.eyeL).toBe(0);
    expect(smileEval.parameters.eyeLSmile).toBeGreaterThan(0.8);
    expect(smileEval.hiddenBasePartIds.has("eyeL_upper")).toBe(false);

    // Tilted HeadFrame deform in TS keeps ^ smile above u sad on rotated portrait faces
    const tiltLm: Landmarks = {
      eyeR: { x: 40, y: 36 },
      eyeL: { x: 80, y: 44 },
      mouth: { x: 56, y: 66 },
      chin: { x: 54, y: 84 },
      crown: { x: 68, y: 8 },
    };
    const tipPt = { x: landmarks.eyeL.x, y: landmarks.eyeL.y };
    const smileTip = deform(
      tipPt,
      { id: "eyeL_upper" },
      tiltLm,
      smileEval.parameters,
      smileEval.shapeDeform,
    );
    const sadTip = deform(
      tipPt,
      { id: "eyeL_upper" },
      tiltLm,
      sadEval.parameters,
      sadEval.shapeDeform,
    );
    expect(smileTip.y).toBeLessThan(sadTip.y);

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

  it("cleanSubtractedSiblingMask strips 1px AA halos and tiny seam rings without deleting distant curls", () => {
    const w = 64;
    const h = 64;
    const sibling = new Uint8Array(w * h);
    const claimed = new Uint8Array(w * h);

    // Main hair_front island on left (18x20 = 360px)
    for (let y = 4; y < 24; y++) {
      for (let x = 4; x < 22; x++) sibling[y * w + x] = 255;
    }
    // Claimed hair_side_L overlaps x=[16..34), y=[6..24) with soft AA perimeter at x=15
    for (let y = 6; y < 24; y++) {
      for (let x = 16; x < 34; x++) claimed[y * w + x] = 255;
      sibling[y * w + 15] = 100; // AA fringe next to claimed boundary
    }
    // Thin detached 2x10 (20px) sliver left behind at right side touching claimed border (x=34..35)
    for (let y = 10; y < 20; y++) {
      sibling[y * w + 34] = 240;
      sibling[y * w + 35] = 240;
    }
    // Distant genuine floating curl on bottom right (4x8 = 32px, does NOT touch claimed)
    for (let y = 46; y < 54; y++) {
      for (let x = 50; x < 54; x++) sibling[y * w + x] = 255;
    }

    const res = cleanSubtractedSiblingMask(sibling, claimed, w, h);
    expect(res.changed).toBe(true);
    // Overlap & 1px AA border cleared
    expect(res.mask[12 * w + 16]).toBe(0);
    expect(res.mask[12 * w + 15]).toBe(0);
    // Thin detached seam ring sliver stripped
    expect(res.mask[14 * w + 34]).toBe(0);
    // Main hair_front island preserved
    expect(res.mask[12 * w + 8]).toBe(255);
    // Distant untouched strand preserved
    expect(res.mask[49 * w + 51]).toBe(255);
  });

  it("mergeStage2HairParts keeps Stage-1 bald base hole-free in fillMask and splices hair cleanly", () => {
    expect(isHairStageLayer("hair_front")).toBe(true);
    expect(isHairStageLayer("hair_side_L_2")).toBe(true);
    expect(isHairStageLayer("hat")).toBe(true);
    expect(isHairStageLayer("acc_hairpin")).toBe(true);
    expect(isHairStageLayer("acc_ribbon")).toBe(true);
    expect(isHairStageLayer("face")).toBe(false);
    expect(isHairStageLayer("clothes_hands")).toBe(false);
    expect(isHairStageLayer("acc_glasses")).toBe(false);

    const n = 16;
    const faceMask = new Uint8Array(n);
    const faceFill = new Uint8Array(n);
    faceMask[0] = 255;
    faceMask[1] = 255;
    faceMask[2] = 255;
    faceFill[3] = 255;

    const clothesMask = new Uint8Array(n);
    clothesMask[4] = 255;
    clothesMask[5] = 255;

    const baseParts = [
      {
        id: "clothes",
        name: "의상",
        group: "몸" as const,
        color: "#888",
        role: "clothes" as const,
        visible: true,
        status: "draft" as const,
        mask: clothesMask,
        fillKind: "none" as const,
      },
      {
        id: "face",
        name: "얼굴 피부",
        group: "머리" as const,
        color: "#fca",
        role: "face" as const,
        visible: true,
        status: "approved" as const,
        mask: faceMask,
        fillMask: faceFill,
        fillKind: "template" as const,
      },
    ];

    const backMask = new Uint8Array(n);
    backMask[5] = 255;
    backMask[6] = 255;
    const frontMask = new Uint8Array(n);
    frontMask[1] = 255;
    frontMask[3] = 255;

    const hairParts = [
      {
        id: "hair_back",
        name: "뒷머리",
        group: "머리" as const,
        color: "#557",
        role: "hair_back" as const,
        visible: true,
        status: "draft" as const,
        mask: backMask,
        fillKind: "none" as const,
      },
      {
        id: "hair_front",
        name: "앞머리",
        group: "머리" as const,
        color: "#779",
        role: "hair_front" as const,
        visible: true,
        status: "draft" as const,
        mask: frontMask,
        fillKind: "none" as const,
      },
    ];

    const { parts: merged } = mergeStage2HairParts(baseParts, hairParts);
    expect(merged.map((p) => p.id)).toEqual([
      "hair_back",
      "clothes",
      "face",
      "hair_front",
    ]);

    const mergedFace = merged.find((p) => p.id === "face")!;
    expect(mergedFace.status).toBe("approved");
    // Pixel 1 overlapped by hair_front shifted out of face.mask into face.fillMask
    expect(mergedFace.mask[1]).toBe(0);
    expect(mergedFace.fillMask![1]).toBe(255);
    expect(mergedFace.mask[2]).toBe(255);
    expect(mergedFace.fillMask![3]).toBe(255);

    const mergedClothes = merged.find((p) => p.id === "clothes")!;
    // Pixel 5 overlapped by hair_back shifted out of clothes.mask into clothes.fillMask
    expect(mergedClothes.mask[5]).toBe(0);
    expect(mergedClothes.fillMask![5]).toBe(255);

    // Verify visible masks across all merged parts remain strictly pairwise disjoint
    for (let px = 0; px < n; px++) {
      const active = merged.filter((p) => p.mask[px] > 0);
      expect(active.length).toBeLessThanOrEqual(1);
    }
  });

  it("4-stage mergeStageParts composes silhouette -> features -> clothes -> hair in canonical order with clipping", () => {
    expect(partPipelineStage("body")).toBe("base_silhouette");
    expect(partPipelineStage("neck")).toBe("base_silhouette");
    expect(partPipelineStage("face")).toBe("base_silhouette");
    expect(partPipelineStage("eyeR_white")).toBe("base_silhouette");
    expect(partPipelineStage("eyeL_white")).toBe("base_silhouette");
    expect(partPipelineStage("mouth_inner")).toBe("base_silhouette");

    expect(partPipelineStage("face_detail")).toBe("features_overlay");
    expect(partPipelineStage("eyeR_iris")).toBe("features_overlay");
    expect(partPipelineStage("eyeL_upper")).toBe("features_overlay");
    expect(partPipelineStage("browL")).toBe("features_overlay");
    expect(partPipelineStage("mouth_upper")).toBe("features_overlay");

    expect(partPipelineStage("clothes")).toBe("clothes_overlay");
    expect(partPipelineStage("clothes_inner")).toBe("clothes_overlay");
    expect(partPipelineStage("clothes_hands")).toBe("clothes_overlay");
    expect(partPipelineStage("acc_glasses")).toBe("clothes_overlay");

    expect(partPipelineStage("hair_back")).toBe("hair_overlay");
    expect(partPipelineStage("hair_side_L_2")).toBe("hair_overlay");
    expect(partPipelineStage("hair_front")).toBe("hair_overlay");
    expect(partPipelineStage("hat")).toBe("hair_overlay");

    expect(stageIncludesLayer("face_detail", "features_overlay")).toBe(true);
    expect(stageIncludesLayer("face_detail", "base_silhouette")).toBe(false);
    expect(stageIncludesLayer("face_detail", "base_body")).toBe(true);
    expect(stageIncludesLayer("hair_front", "base_body")).toBe(false);

    expect(inferPartRole("face_detail")).toBe("face");
    expect(describeDynamicPart("face_detail")).toMatchObject({
      name: "얼굴 디테일 (볼터치·코)",
      group: "얼굴",
      role: "face",
    });

    const n = 20;
    const mkMask = (...indices: number[]) => {
      const m = new Uint8Array(n);
      for (const idx of indices) m[idx] = 255;
      return m;
    };
    const mkPart = (
      id: string,
      maskIndices: number[],
      status: "draft" | "approved" = "draft",
      fillIndices: number[] = [],
    ) => ({
      id,
      name: id,
      group: "얼굴",
      color: "#aaa",
      role: inferPartRole(id),
      visible: true,
      status,
      mask: mkMask(...maskIndices),
      ...(fillIndices.length > 0
        ? { fillMask: mkMask(...fillIndices), fillKind: "template" as const }
        : { fillKind: "none" as const }),
    });

    const s1 = [
      mkPart("body", [0, 1]),
      mkPart("neck", [2, 3]),
      mkPart("face", [4, 5, 6, 7]),
      mkPart("eyeL_white", [8, 9]),
    ];
    const s2 = [
      mkPart("face_detail", [5]),
      mkPart("eyeL_iris", [9], "approved"),
      mkPart("eyeL_upper", [6, 9]),
    ];
    // Overlaps in s2 intentionally yield mask->fillMask cleanly per lower layer
    const stage12 = mergeStageParts(s1, [
      mkPart("face_detail", [5]),
      mkPart("eyeL_iris", [9], "approved"),
      mkPart("eyeL_upper", [6]),
    ], "features_overlay");

    // Re-running Stage 2 preserves approved eyeL_iris
    const regenStage2 = mergeStageParts(
      stage12.parts,
      [
        mkPart("face_detail", [5]),
        mkPart("eyeL_iris", [8], "draft"),
        mkPart("eyeL_upper", [6]),
      ],
      "features_overlay",
      true,
    );
    expect(regenStage2.preservedCount).toBe(1);
    expect(
      regenStage2.parts.find((p) => p.id === "eyeL_iris")!.mask[9],
    ).toBe(255);

    const stage123 = mergeStageParts(
      regenStage2.parts,
      [
        mkPart("clothes", [1, 2, 10]),
        mkPart("clothes_hands", [11]),
        mkPart("acc_glasses", [5, 12]),
      ],
      "clothes_overlay",
    );
    const stage1234 = mergeStageParts(
      stage123.parts,
      [
        mkPart("hair_back", [0, 13]),
        mkPart("hair_front", [4, 14]),
      ],
      "hair_overlay",
    );

    expect(stage1234.parts.map((p) => p.id)).toEqual([
      "hair_back",
      "body",
      "neck",
      "clothes",
      "clothes_hands",
      "face",
      "face_detail",
      "eyeL_white",
      "eyeL_iris",
      "eyeL_upper",
      "acc_glasses",
      "hair_front",
    ]);

    // Base face stays hole-free in mask|fillMask when covered by face_detail, upper lid, glasses, hair
    const finalFace = stage1234.parts.find((p) => p.id === "face")!;
    expect(finalFace.mask[4]).toBe(0);
    expect(finalFace.fillMask![4]).toBe(255);
    expect(finalFace.mask[5]).toBe(0);
    expect(finalFace.fillMask![5]).toBe(255);
    expect(finalFace.mask[7]).toBe(255);

    // Every visible pixel across all 4 stages stays strictly pairwise disjoint
    for (let px = 0; px < n; px++) {
      const active = stage1234.parts.filter((p) => p.mask[px] > 0);
      expect(active.length).toBeLessThanOrEqual(1);
    }
    void s2;
  });

  it("clips face_detail strictly inside face boundary at rest and during motion unless soloed", async () => {
    const p = fixture();
    const n = p.width * p.height;
    const faceBase = p.parts.find((pt) => pt.id === "face")!;
    const faceFill = new Uint8Array(n);
    const fdMask = new Uint8Array(n);
    // Keep face inside (20..44, 14..32), fill all face area, and put an intentionally overflowing blush at (10, 22)
    const clippedFaceMask = new Uint8Array(n);
    for (let y = 14; y <= 32; y++) {
      for (let x = 20; x <= 44; x++) {
        clippedFaceMask[y * p.width + x] = 255;
        faceFill[y * p.width + x] = 255;
      }
    }
    // Outside-face stray blush blob at x=8..11, y=20..23
    for (let y = 20; y <= 23; y++) {
      for (let x = 8; x <= 11; x++) {
        fdMask[y * p.width + x] = 255;
      }
    }
    // Inside-face cheek blush blob at x=24..27, y=20..23
    for (let y = 20; y <= 23; y++) {
      for (let x = 24; x <= 27; x++) {
        fdMask[y * p.width + x] = 255;
        clippedFaceMask[y * p.width + x] = 0;
      }
    }

    const srcCanvas = createCanvas(p.width, p.height);
    const sCtx = srcCanvas.getContext("2d");
    sCtx.fillStyle = "rgb(245, 220, 208)";
    sCtx.fillRect(20, 14, 25, 19);
    // Distinct hot pink outside face (x=8..11) + inside face (x=24..27)
    sCtx.fillStyle = "rgb(255, 60, 140)";
    sCtx.fillRect(8, 20, 4, 4);
    sCtx.fillRect(24, 20, 4, 4);
    const px = new Uint8ClampedArray(sCtx.getImageData(0, 0, p.width, p.height).data);

    const rigState = createDefaultProjectRigState("ld");
    const clippedProject: Project = {
      ...p,
      ...rigState,
      parts: [
        {
          ...faceBase,
          compositeOrder: 0,
          mask: clippedFaceMask,
          fillMask: faceFill,
          fillKind: "template",
        },
        {
          id: "face_detail",
          compositeOrder: 1,
          name: "얼굴 디테일 (볼터치·코)",
          group: "얼굴",
          color: "#fb7185",
          role: "face",
          visible: true,
          status: "draft",
          mask: fdMask,
          fillKind: "none",
        },
      ],
    };

    const textures = buildAllPartTextures(clippedProject, px);
    const meshes = buildAllPartMeshes(clippedProject);
    const restSnap = renderDeterministicSnapshot(
      { project: clippedProject },
      textures,
      meshes,
    );
    const restData = (restSnap.canvas as unknown as ReturnType<typeof createCanvas>)
      .getContext("2d")
      .getImageData(0, 0, p.width, p.height).data;
    // Pixel (10, 22) outside face boundary MUST be clipped away even at rest
    expect(restData[(22 * p.width + 10) * 4 + 3]).toBe(0);
    // Pixel (26, 22) inside face boundary MUST remain visible
    expect(restData[(22 * p.width + 26) * 4]).toBeGreaterThan(200);
    expect(restData[(22 * p.width + 26) * 4 + 3]).toBe(255);

    // Moving pose also clips face_detail cleanly to face
    const movingSnap = renderDeterministicSnapshot(
      { project: clippedProject, userOffsets: { angleX: 12 } },
      textures,
      meshes,
    );
    const movingData = (movingSnap.canvas as unknown as ReturnType<typeof createCanvas>)
      .getContext("2d")
      .getImageData(0, 0, p.width, p.height).data;
    expect(movingData[(22 * p.width + 8) * 4 + 3]).toBe(0);

    // Solo inspection of face_detail renders unclipped blob at (10, 22)
    const soloEval = evaluateRigFrame({
      project: clippedProject,
      deterministic: true,
      playing: false,
    });
    const soloCanvas = createCanvas(p.width, p.height);
    renderCharacterFrame({
      ctx: soloCanvas.getContext("2d") as unknown as CanvasRenderingContext2D,
      project: clippedProject,
      evaluated: soloEval,
      partTextures: textures,
      meshes,
      soloPartId: "face_detail",
      enableGpu: false,
    });
    const soloData = soloCanvas
      .getContext("2d")
      .getImageData(0, 0, p.width, p.height).data;
    expect(soloData[(22 * p.width + 10) * 4 + 3]).toBeGreaterThan(200);
  });

  it("scales partOverlapMargins and recommendedFillRadius proportionally across resolutions and covers opposing motion", () => {
    const scaleLm = (s: number): Landmarks => ({
      eyeL: { x: 38 * s, y: 24 * s },
      eyeR: { x: 22 * s, y: 24 * s },
      mouth: { x: 30 * s, y: 33 * s },
      chin: { x: 30 * s, y: 40 * s },
      crown: { x: 30 * s, y: 4 * s },
    });
    const m512 = partOverlapMargins(scaleLm(8), 512, 512);
    const m1024 = partOverlapMargins(scaleLm(16), 1024, 1024);
    const m2048 = partOverlapMargins(scaleLm(32), 2048, 2048);

    expect(m1024.neckCollarX).toBeGreaterThan(m512.neckCollarX);
    expect(m2048.neckCollarX).toBeGreaterThan(m1024.neckCollarX);
    expect(m1024.hairSideFrontSeam).toBeGreaterThan(14);
    expect(m2048.hairBackUnderlap).toBeGreaterThan(m1024.hairBackUnderlap);

    expect(recommendedFillRadius({ id: "neck" }, scaleLm(16), 1024, 1024)).toBeGreaterThan(14);
    expect(recommendedFillRadius({ id: "hair_back_2" }, scaleLm(32), 2048, 2048)).toBeGreaterThan(128);

    const lm1024 = scaleLm(16);
    const collarPt = { x: lm1024.chin.x, y: lm1024.chin.y + 120 };
    const opposing = {
      ...REST,
      ParamAngleX: -30,
      ParamAngleY: -30,
      ParamAngleZ: -20,
      ParamBodyAngleX: 10,
      ParamBodyAngleY: 10,
      ParamBodyAngleZ: 10,
      ParamBreath: 1,
    };
    const neckP = deform(collarPt, { id: "neck" }, lm1024, opposing);
    const clothesP = deform(collarPt, { id: "clothes" }, lm1024, opposing);
    expect(Math.abs(neckP.x - clothesP.x)).toBeLessThanOrEqual(m1024.neckCollarX);
    expect(Math.abs(neckP.y - clothesP.y)).toBeLessThanOrEqual(m1024.neckCollarY);

    const cheekHairPt = { x: lm1024.eyeR.x - 100, y: lm1024.chin.y };
    const swayPose = { ...REST, ParamHairFront: 1, ParamHairSide: -1 };
    const frontP = deform(cheekHairPt, { id: "hair_front" }, lm1024, swayPose);
    const sideP = deform(cheekHairPt, { id: "hair_side_R" }, lm1024, swayPose);
    expect(Math.abs(frontP.x - sideP.x)).toBeLessThanOrEqual(m1024.hairSideFrontSeam);
  });

  it("dispatches recommendedFillRadius by role and matches the shared Rust/TS parity fixture", () => {
    const fixture = fillRadiusParity as {
      roleInference: { id: string; role: PartRole }[];
      frames: {
        label: string;
        width: number;
        height: number;
        landmarks: Landmarks;
        parts: { id: string; role?: PartRole; radius: number }[];
      }[];
    };
    expect(fixture.roleInference.length).toBeGreaterThan(0);
    expect(fixture.frames.length).toBeGreaterThan(0);
    for (const c of fixture.roleInference) {
      expect(inferPartRole(c.id), c.id).toBe(c.role);
    }
    for (const f of fixture.frames) {
      for (const p of f.parts) {
        expect(
          recommendedFillRadius({ id: p.id, role: p.role }, f.landmarks, f.width, f.height),
          `${f.label} / ${p.id} (${p.role ?? "inferred"})`,
        ).toBe(p.radius);
      }
    }
    // Variable multi-hair IDs must resolve through roles, not the fixed 18-ID table.
    const ids = fixture.frames[0].parts.map((p) => p.id);
    for (const id of ["hair_front_2", "hair_side_L_2", "hair_side_R_3", "hair_back_3"]) {
      expect(ids).toContain(id);
    }
  });

  it("T13a pose envelope keeps every PARAM_DEFS extreme and head/body/hair phase combinations", () => {
    // Ranges are PARAM_DEFS exactly (never shrunk) and cover every parameter.
    expect(Object.keys(POSE_ENVELOPE_RANGES).sort()).toEqual(Object.keys(REST).sort());
    for (const def of PARAM_DEFS) {
      expect(POSE_ENVELOPE_RANGES[def.key as keyof typeof REST], def.key).toEqual([def.min, def.max]);
      for (const extreme of [def.min, def.max]) {
        expect(
          POSE_ENVELOPE.some((p) => p.params[def.key as keyof typeof REST] === extreme),
          `${def.key}=${extreme}`,
        ).toBe(true);
      }
    }
    expect(PARAM_DEFS.length).toBe(Object.keys(REST).length);
    const names = new Set(POSE_ENVELOPE.map((p) => p.name));
    expect(names.size).toBe(POSE_ENVELOPE.length);
    expect(POSE_ENVELOPE.length).toBe(94);
    for (const p of POSE_ENVELOPE) {
      for (const def of PARAM_DEFS) {
        const v = p.params[def.key as keyof typeof REST];
        expect(v, `${p.name}.${def.key}`).toBeGreaterThanOrEqual(def.min);
        expect(v, `${p.name}.${def.key}`).toBeLessThanOrEqual(def.max);
      }
    }
    expect(
      POSE_ENVELOPE.some(
        (p) =>
          p.params.angleX * p.params.bodyAngleX < 0 &&
          p.params.angleY * p.params.bodyAngleY < 0 &&
          p.params.angleZ * p.params.bodyAngleZ < 0,
      ),
    ).toBe(true);
    expect(POSE_ENVELOPE.some((p) => p.params.hair * p.params.hairSide < 0)).toBe(true);
    expect(POSE_ENVELOPE.some((p) => p.params.hairSide * p.params.hairBack < 0)).toBe(true);
  });

  it("T13a envelope and seam relative displacement match Rust (shared parity fixture)", () => {
    const fixture = seamEnvelopeParity as unknown as {
      landmarks: Landmarks;
      envelope: { name: string; pose: Record<string, number> }[];
      cases: {
        upper: string;
        lower: string;
        points: [number, number][];
        maxRelativeDisplacementPx: number;
        worstPose: string | null;
        uncoupledMaxRelativeDisplacementPx: number;
      }[];
      farPoints: { name: string; id: string; point: [number, number]; maxDisplacementPx: number }[];
      couplingWeights: {
        id: string;
        point: [number, number];
        weights: { neckHead: number; neckBody: number; collar: number; hairRoot: number };
      }[];
    };
    // Rust parses the same JSON over Pose::default(); TS merges over REST. Both must agree.
    expect(fixture.envelope.map((e) => e.name)).toEqual(POSE_ENVELOPE.map((p) => p.name));
    fixture.envelope.forEach((e, i) => {
      expect(POSE_ENVELOPE[i].params, e.name).toEqual(e.pose);
    });
    expect(fixture.cases.length).toBeGreaterThan(10);
    for (const c of fixture.cases) {
      const got = maxSeamRelativeDisplacement(
        c.points.map(([x, y]) => ({ x, y })),
        { id: c.upper },
        { id: c.lower },
        fixture.landmarks,
      );
      expect(Math.abs(got.maxPx - c.maxRelativeDisplacementPx), `${c.upper}/${c.lower}`).toBeLessThan(1e-6);
      const raw = maxSeamRelativeDisplacement(
        c.points.map(([x, y]) => ({ x, y })),
        { id: c.upper },
        { id: c.lower },
        fixture.landmarks,
        POSE_ENVELOPE,
        deformUncoupled,
      );
      expect(
        Math.abs(raw.maxPx - c.uncoupledMaxRelativeDisplacementPx),
        `uncoupled ${c.upper}/${c.lower}`,
      ).toBeLessThan(1e-6);
    }
    // T13b far reference points: coupled max displacement equals Rust, and equals the uncoupled
    // field pose by pose (coupling never reduces motion away from seams).
    expect(fixture.farPoints.length).toBeGreaterThan(10);
    for (const f of fixture.farPoints) {
      const p = { x: f.point[0], y: f.point[1] };
      let maxD = 0;
      for (const pose of POSE_ENVELOPE) {
        const a = deform(p, { id: f.id }, fixture.landmarks, pose.params);
        const b = deformUncoupled(p, { id: f.id }, fixture.landmarks, pose.params);
        expect(Math.abs(a.x - b.x) + Math.abs(a.y - b.y), `${f.name} ${pose.name}`).toBeLessThan(1e-9);
        maxD = Math.max(maxD, Math.hypot(a.x - p.x, a.y - p.y));
      }
      expect(Math.abs(maxD - f.maxDisplacementPx), f.name).toBeLessThan(1e-6);
    }
    for (const c of fixture.couplingWeights) {
      const got = seamCouplingWeights({ x: c.point[0], y: c.point[1] }, { id: c.id }, fixture.landmarks);
      for (const k of ["neckHead", "neckBody", "collar", "hairRoot"] as const) {
        expect(Math.abs(got[k] - c.weights[k]), `${c.id} ${k}`).toBeLessThan(1e-9);
      }
    }
    // Coupled seams: relative displacement shrinks (face/neck, collar).
    const fn = fixture.cases.find((c) => c.upper === "face" && c.lower === "neck")!;
    expect(fn.maxRelativeDisplacementPx).toBeLessThan(fn.uncoupledMaxRelativeDisplacementPx);
    // Scaling the deform displacement field scales the measured requirement (no copied constants).
    const c0 = fixture.cases.find((c) => c.upper === "clothes" && c.lower === "neck")!;
    const pts = c0.points.map(([x, y]) => ({ x, y }));
    const base = maxSeamRelativeDisplacement(
      pts,
      { id: "clothes" },
      { id: "neck" },
      fixture.landmarks,
      POSE_ENVELOPE,
      deformUncoupled,
    );
    expect(base.maxPx).toBeGreaterThan(1);
    const scaled = maxSeamRelativeDisplacement(
      pts,
      { id: "clothes" },
      { id: "neck" },
      fixture.landmarks,
      POSE_ENVELOPE,
      (p, part, lm, params) => {
        const q = deformUncoupled(p, part, lm, params);
        return { x: p.x + 2 * (q.x - p.x), y: p.y + 2 * (q.y - p.y) };
      },
    );
    expect(Math.abs(scaled.maxPx - 2 * base.maxPx)).toBeLessThan(1e-6);
  });
});
