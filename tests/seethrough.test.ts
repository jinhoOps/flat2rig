import fs from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { createCanvas, ImageData } from "@napi-rs/canvas";
import { initializeCanvas, writePsdUint8Array, type Layer } from "ag-psd";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { parsePsd, suggestMapping } from "../src/psdLayers";
import { buildPsdParts, openPsdSession, runSeeThroughDecomposeFlow } from "../src/layerProject";
import {
  applySeeThroughAlphaFloor,
  clearSeeThroughSettings,
  correctSeeThroughDrawOrder,
  isSeeThroughPsd,
  loadSeeThroughSettings,
  saveSeeThroughSettings,
  SEE_THROUGH_DEFAULT_SEED,
} from "../src/seethrough";
import PsdMappingPanel from "../src/workspace/PsdMappingPanel";
import SeeThroughModal from "../src/workspace/SeeThroughModal";

beforeAll(() => {
  initializeCanvas(
    (w, h) => createCanvas(w, h) as unknown as HTMLCanvasElement,
    (w, h) => new ImageData(w, h) as unknown as globalThis.ImageData,
  );
});

type Px = [number, number, number, number];

function rect(
  name: string,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: Px = [200, 150, 120, 255],
  extra: Partial<Layer> = {},
): Layer {
  const w = x1 - x0;
  const h = y1 - y0;
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

function pixelsLayer(
  name: string,
  x0: number,
  y0: number,
  w: number,
  h: number,
  fn: (x: number, y: number) => Px,
): Layer {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) data.set(fn(x, y), (y * w + x) * 4);
  }
  return {
    name,
    left: x0,
    top: y0,
    right: x0 + w,
    bottom: y0 + h,
    imageData: { width: w, height: h, data } as unknown as globalThis.ImageData,
  };
}

function psdBytes(width: number, height: number, children: Layer[]) {
  return writePsdUint8Array({ width, height, children }, { noBackground: true });
}

/**
 * Synthetic Sutteang-like See-through PSD (200x120, character centre at x=100):
 * - `eyebrow` placed BELOW `face` (depth-order defect)
 * - `eyewhite` (L only, x=115..135) placed ABOVE `irides` (causes white-circle eye defect)
 * - `eyelash` has upper lashes on both sides and a lower lash on character L (x=115..135, y=54..56)
 * - `mouth` has faint alpha<=4 noise across a wide area
 */
function sutteangLikeSeeThroughPsd() {
  return psdBytes(200, 120, [
    rect("back hair", 30, 5, 170, 115, [80, 60, 120, 255]),
    rect("neck", 90, 70, 110, 100, [240, 200, 180, 255]),
    rect("topwear", 55, 85, 145, 120, [50, 70, 110, 255]),
    // Eyebrow below face (See-through pseudo-depth defect)
    pixelsLayer("eyebrow", 60, 28, 80, 6, (x) =>
      x < 20 || x >= 60 ? [70, 45, 40, 255] : [0, 0, 0, 0],
    ),
    // Off-centre face to verify iris-pair midline takes precedence over face centroid
    rect("face", 50, 15, 140, 80, [245, 215, 195, 255]),
    pixelsLayer("mouth", 0, 0, 200, 120, (x, y) => {
      if (x >= 94 && x < 106 && y >= 65 && y < 70) return [190, 90, 90, 255];
      if (x < 30 && y < 30) return [120, 120, 120, 3]; // noise <= 4/255
      return [0, 0, 0, 0];
    }),
    // Irides: R at x=70..80, L at x=120..130 (midpoint x=100)
    pixelsLayer("irides", 70, 42, 60, 10, (x) =>
      x < 10 || x >= 50 ? [40, 70, 170, 255] : [0, 0, 0, 0],
    ),
    // Eyewhite: L only (x=115..135), stacked ABOVE irides!
    rect("eyewhite", 115, 43, 135, 51, [255, 255, 255, 255]),
    // Eyelash: upper at y=39..41 (both sides), lower at y=54..56 (L side only)
    pixelsLayer("eyelash", 65, 39, 70, 18, (x, y) => {
      const upper = y < 3 && (x < 20 || x >= 50);
      const lowerL = y >= 14 && x >= 50;
      return upper || lowerL ? [25, 20, 25, 255] : [0, 0, 0, 0];
    }),
    rect("front hair", 60, 5, 140, 32, [90, 70, 130, 255]),
  ]);
}

describe("B5 See-through — detection, alpha floor, L/R & lash split, depth order", () => {
  it("detects See-through PSD tags and ignores regular artist layer names", () => {
    expect(
      isSeeThroughPsd(["back hair", "front hair", "face", "irides", "eyewhite", "eyelash", "mouth"]),
    ).toBe(true);
    expect(isSeeThroughPsd(["front_hair", "Topwear", "Handwear"])).toBe(true);
    expect(isSeeThroughPsd([])).toBe(false);
    expect(isSeeThroughPsd(["뒷머리", "얼굴", "앞머리"])).toBe(false);
    expect(isSeeThroughPsd(["back hair", "얼굴"])).toBe(false);
  });

  it("zeroes alpha <= 4/255 noise while keeping visible pixels intact", () => {
    const px = {
      left: 0,
      top: 0,
      width: 3,
      height: 1,
      rgba: new Uint8ClampedArray([10, 20, 30, 4, 40, 50, 60, 5, 70, 80, 90, 255]),
    };
    const cleaned = applySeeThroughAlphaFloor(px);
    expect([...cleaned.rgba]).toEqual([0, 0, 0, 0, 40, 50, 60, 5, 70, 80, 90, 255]);
  });

  it("corrects Sutteang eye white above iris and face features below face, preserving relative order", () => {
    const plan = correctSeeThroughDrawOrder([
      "hair_back",
      "browR",
      "browL",
      "acc_glasses",
      "face",
      "eyeL_iris",
      "eyeR_iris",
      "eyeL_white",
      "hair_front",
    ]);
    expect(plan.changed).toBe(true);
    expect(plan.liftedAboveFace).toEqual(["browR", "browL", "acc_glasses"]);
    expect(plan.eyeWhiteMovedBelowIris).toEqual(["eyeL_white"]);
    expect(plan.correctedOrder).toEqual([
      "hair_back",
      "face",
      "browR",
      "browL",
      "acc_glasses",
      "eyeL_white",
      "eyeL_iris",
      "eyeR_iris",
      "hair_front",
    ]);
  });

  it("opens a See-through PSD, splits L/R + upper/lower lash, and applies reversible depth-order fix", async () => {
    const bytes = sutteangLikeSeeThroughPsd();
    const session = await openPsdSession(bytes, "sutteang-like.psd");
    expect(session.psd.seeThrough).toBe(true);
    expect(session.applySeeThroughOrderFix).toBe(true);
    // Midline comes from iris pair (99.5), not off-centre face centroid (95).
    expect(session.suggestion.midlineSource).toBe("iris");
    expect(session.midlineX).toBeCloseTo(99.5, 1);

    // Alpha floor stripped the <=4/255 noise from `mouth`.
    const mouthNode = session.psd.layers.find((n) => n.name === "mouth")!;
    expect(mouthNode.centroidX).toBeCloseTo(100, 1);

    // Depth-order fix is enabled by default: eyeL_white is below eyeL_iris, brow* above face.
    const fixedBuild = buildPsdParts(session);
    const fixedIds = fixedBuild.parts.map((p) => p.id);
    expect(fixedIds.indexOf("browL")).toBeGreaterThan(fixedIds.indexOf("face"));
    expect(fixedIds.indexOf("browR")).toBeGreaterThan(fixedIds.indexOf("face"));
    expect(fixedIds.indexOf("eyeL_white")).toBeLessThan(fixedIds.indexOf("eyeL_iris"));
    // Eyelash split produced upper (L/R) and lower (L only, since R has no lower pixels).
    expect(fixedIds).toContain("eyeL_upper");
    expect(fixedIds).toContain("eyeR_upper");
    expect(fixedIds).toContain("eyeL_lower");
    expect(fixedIds).not.toContain("eyeR_lower");

    // Reverting to raw PSD order puts eyeL_white back above eyeL_iris and brow* below face.
    const rawBuild = buildPsdParts({ ...session, applySeeThroughOrderFix: false });
    const rawIds = rawBuild.parts.map((p) => p.id);
    expect(rawIds.indexOf("browL")).toBeLessThan(rawIds.indexOf("face"));
    expect(rawIds.indexOf("eyeL_white")).toBeGreaterThan(rawIds.indexOf("eyeL_iris"));
  });

  it("does not trigger See-through detection or reorder on regular artist PSDs", async () => {
    const bytes = psdBytes(200, 100, [
      rect("뒷머리", 40, 5, 160, 95),
      rect("눈썹>", 110, 30, 130, 33),
      rect("얼굴", 60, 15, 140, 70),
      rect("앞머리", 70, 5, 130, 30),
    ]);
    const session = await openPsdSession(bytes, "himikan-artist.psd");
    expect(session.psd.seeThrough).toBe(false);
    expect(session.applySeeThroughOrderFix).toBe(false);
    expect(session.suggestion.seeThroughOrder).toBeUndefined();
    const build = buildPsdParts(session);
    // Artist's original PSD order is preserved untouched.
    expect(build.parts.map((p) => p.id)).toEqual(["hair_back", "browL", "face", "hair_front"]);
  });

  it("renders See-through banner, toggle, badges, and limitations in PsdMappingPanel", async () => {
    const session = await openPsdSession(sutteangLikeSeeThroughPsd(), "sutteang-like.psd");
    const htmlFixed = renderToStaticMarkup(
      createElement(PsdMappingPanel, {
        session,
        busy: false,
        onChange: () => {},
        onToggleSeeThroughOrder: () => {},
        onGenerate: () => {},
        onCancel: () => {},
      }),
    );
    expect(htmlFixed).toContain("See-through PSD 감지됨 — 깊이 순서 보정 적용됨 (원본 순서로 되돌리기)");
    expect(htmlFixed).toContain("원본 순서로 되돌리기");
    expect(htmlFixed).toContain("순서 보정");
    expect(htmlFixed).toContain("입·코 숨은 영역 없음");
    expect(htmlFixed).toContain("치비(SD)");
    expect(htmlFixed).toContain("OpenRAIL++");

    const htmlRaw = renderToStaticMarkup(
      createElement(PsdMappingPanel, {
        session: { ...session, applySeeThroughOrderFix: false },
        busy: false,
        onChange: () => {},
        onToggleSeeThroughOrder: () => {},
        onGenerate: () => {},
        onCancel: () => {},
      }),
    );
    expect(htmlRaw).toContain("See-through PSD 감지됨 — 원본 PSD 순서 유지 중");
    expect(htmlRaw).toContain("깊이 순서 보정 적용");
    expect(htmlRaw).toContain("순서 역전");
  });
});

const SUTTEANG_PSD = "C:/Users/kjh/orca/flat2rig/artifacts/seethrough/out/sutteang/sutteang.psd";
const HIMIKAN_PSD = "C:/Users/kjh/orca/flat2rig/artifacts/seethrough/out/himikan_s42/himikan.psd";
const hasRealSeeThroughPsds = fs.existsSync(SUTTEANG_PSD) && fs.existsSync(HIMIKAN_PSD);

describe.skipIf(!hasRealSeeThroughPsds)("B5 See-through — real V2 PSD integration", () => {
  it("detects Sutteang eyewhite/iris inversion and Himikan below-face features", () => {
    const sut = parsePsd(new Uint8Array(fs.readFileSync(SUTTEANG_PSD)), "sutteang.psd");
    expect(sut.seeThrough).toBe(true);
    const sutSug = suggestMapping(sut);
    expect(sutSug.seeThroughOrder?.changed).toBe(true);
    expect(sutSug.seeThroughOrder?.eyeWhiteMovedBelowIris).toContain("eyeL_white");

    const him = parsePsd(new Uint8Array(fs.readFileSync(HIMIKAN_PSD)), "himikan.psd");
    expect(him.seeThrough).toBe(true);
    const himSug = suggestMapping(him);
    expect(himSug.seeThroughOrder?.changed).toBe(true);
    expect(himSug.seeThroughOrder?.liftedAboveFace.length).toBeGreaterThan(0);
  });
});

describe("B5 See-through Checkpoint 2 — settings, unconfigured notice, mocked decompose flow", () => {
  function memoryStorage(): Storage {
    const map = new Map<string, string>();
    return {
      get length() {
        return map.size;
      },
      clear: () => map.clear(),
      getItem: (k) => map.get(k) ?? null,
      key: (i) => [...map.keys()][i] ?? null,
      removeItem: (k) => {
        map.delete(k);
      },
      setItem: (k, v) => {
        map.set(k, v);
      },
    };
  }

  it("persists, validates, and clears See-through installDir and seed settings", () => {
    const storage = memoryStorage();
    expect(loadSeeThroughSettings(storage)).toEqual({
      installDir: "",
      seed: SEE_THROUGH_DEFAULT_SEED,
    });
    saveSeeThroughSettings({ installDir: "  C:\\tools\\seethrough  ", seed: 7 }, storage);
    expect(loadSeeThroughSettings(storage)).toEqual({
      installDir: "C:\\tools\\seethrough",
      seed: 7,
    });
    // Invalid seed falls back to default.
    saveSeeThroughSettings({ installDir: "C:\\tools\\seethrough", seed: -5 }, storage);
    expect(loadSeeThroughSettings(storage).seed).toBe(SEE_THROUGH_DEFAULT_SEED);
    clearSeeThroughSettings(storage);
    expect(loadSeeThroughSettings(storage)).toEqual({
      installDir: "",
      seed: SEE_THROUGH_DEFAULT_SEED,
    });
  });

  it("renders unconfigured install guide, OpenRAIL++ notice, and chibi/SD warning in SeeThroughModal", () => {
    const html = renderToStaticMarkup(
      createElement(SeeThroughModal, {
        desktop: true,
        busy: false,
        installDir: "",
        seed: 42,
        status: null,
        error: null,
        onChangeInstallDir: () => {},
        onChangeSeed: () => {},
        onBrowseDir: () => {},
        onVerify: () => {},
        onClear: () => {},
        onRunDecompose: () => {},
        onClose: () => {},
      }),
    );
    expect(html).toContain("외부 설치 안내");
    expect(html).toContain("OpenRAIL++");
    expect(html).toContain("치비(SD)");
    expect(html).toContain('data-action="seethrough-decompose"');
    expect(html).toContain("disabled");
  });

  it("runs mocked decompose → reads generated PSD → opens B1 PsdMappingPanel with depth fix", async () => {
    const psd = sutteangLikeSeeThroughPsd();
    const { result, session } = await runSeeThroughDecomposeFlow(
      {
        decompose: async (installDir, imagePath, opts) => ({
          psdPath: `${installDir}\\out\\sutteang.psd`,
          outputDir: `${installDir}\\out`,
          seed: opts?.seed ?? 42,
          commandSummary: `python inference_psd_blockswap.py --srcp ${imagePath}`,
        }),
        readPsdFile: async () => ({ name: "sutteang.psd", bytes: psd }),
      },
      "C:\\tools\\seethrough",
      "C:\\art\\sutteang.png",
      42,
      1,
    );

    expect(result.psdPath).toContain("sutteang.psd");
    expect(session.psd.seeThrough).toBe(true);
    expect(session.applySeeThroughOrderFix).toBe(true);

    const panelHtml = renderToStaticMarkup(
      createElement(PsdMappingPanel, {
        session,
        busy: false,
        onChange: () => {},
        onToggleSeeThroughOrder: () => {},
        onGenerate: () => {},
        onCancel: () => {},
      }),
    );
    expect(panelHtml).toContain("See-through PSD 감지됨 — 깊이 순서 보정 적용됨");
    expect(panelHtml).toContain("eyeL_white");
  });
});
