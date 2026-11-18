import { beforeAll, describe, expect, it } from "vitest";
import { createCanvas, ImageData } from "@napi-rs/canvas";
import { initializeCanvas, writePsdUint8Array, type Layer, type BlendMode } from "ag-psd";
import {
  missingDefaultParts,
  parsePsd,
  suggestMapping,
  type LayerMapping,
} from "../src/psdLayers";
import {
  buildLayerParts,
  compositeStraight,
  partCanvasRgba,
  splitVisibleFill,
  workingSize,
} from "../src/layerParts";
import { describeDynamicPart, inferPartRole } from "../src/types";
import layerSplitParity from "./fixtures/layer-split-parity.json";

beforeAll(() => {
  initializeCanvas(
    (w, h) => createCanvas(w, h) as unknown as HTMLCanvasElement,
    (w, h) => new ImageData(w, h) as unknown as globalThis.ImageData,
  );
});

type Px = [number, number, number, number];
/** A layer filled with `color` on [x0, x1) × [y0, y1) (PSD coordinates). */
function rect(
  name: string,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: Px = [200, 150, 120, 255],
  extra: Partial<Layer> = {},
): Layer {
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
/** A layer from a per-pixel function over its own box. */
function pixelsLayer(
  name: string,
  x0: number,
  y0: number,
  w: number,
  h: number,
  fn: (x: number, y: number) => Px,
  extra: Partial<Layer> = {},
): Layer {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(fn(x, y), (y * w + x) * 4);
  return {
    name,
    left: x0,
    top: y0,
    right: x0 + w,
    bottom: y0 + h,
    imageData: { width: w, height: h, data } as unknown as globalThis.ImageData,
    ...extra,
  };
}
const group = (name: string, children: Layer[], extra: Partial<Layer> = {}): Layer => ({
  name,
  children,
  opened: true,
  ...extra,
});
function psdBytes(width: number, height: number, children: Layer[]) {
  return writePsdUint8Array({ width, height, children }, { noBackground: true });
}
const byName = (maps: LayerMapping[], path: string) => {
  const m = maps.find((x) => x.path === path);
  if (!m) throw new Error(`no mapping for ${path}: ${maps.map((x) => x.path).join(", ")}`);
  return m;
};
const target = (maps: LayerMapping[], path: string) => {
  const m = byName(maps, path);
  return m.include ? m.target : `excluded:${m.target}`;
};

/** Himikan-like Korean PSD on a 200×100 canvas, face centred at x = 100. Screen-right `>` layers
 * are the character's left eye etc. */
function koreanPsd() {
  const R = (x: number) => 200 - x; // mirror to screen-left
  return psdBytes(200, 100, [
    group("뒷머리", [rect("뒷머리", 40, 5, 160, 95)]),
    group("몸", [rect("카라 뒤", 80, 70, 120, 75), rect("목", 90, 60, 110, 80), rect("몸통", 60, 75, 140, 100), rect("리본", 95, 80, 105, 90)]),
    group("얼굴", [
      rect("귀<", 52, 40, 60, 55),
      rect("귀>", 140, 40, 148, 55),
      rect("얼굴", 60, 15, 140, 70, [240, 210, 190, 255]),
      rect("홍조", 70, 50, 130, 55, [255, 128, 128, 255], { blendMode: "multiply", clipping: true }),
    ]),
    group("눈>", [
      rect("눈흰자>", 110, 40, 130, 48, [255, 255, 255, 255]),
      rect("눈동자>", 115, 39, 125, 49, [40, 60, 160, 255], { clipping: true }),
      rect("눈 하이라이트>", 118, 41, 120, 43, [255, 255, 255, 255], { clipping: true }),
      rect("쌍커풀>", 110, 35, 130, 36, [90, 50, 50, 255]),
      rect("눈언더>", 112, 48, 128, 49, [90, 50, 50, 255]),
      rect("속눈썹>", 108, 37, 132, 40, [30, 20, 20, 255]),
    ]),
    group("눈<", [
      rect("눈흰자<", R(130), 40, R(110), 48, [255, 255, 255, 255]),
      rect("눈동자<", R(125), 39, R(115), 49, [40, 60, 160, 255], { clipping: true }),
      rect("눈언더<", R(128), 48, R(112), 49, [90, 50, 50, 255]),
      rect("눈꺼풀<", R(132), 37, R(108), 40, [30, 20, 20, 255]),
    ]),
    group("눈썹", [rect("눈썹>", 110, 30, 130, 33, [80, 50, 40, 255]), rect("눈썹<", R(130), 30, R(110), 33, [80, 50, 40, 255])]),
    group("입", [
      rect("입안", 95, 58, 105, 62, [120, 30, 40, 255]),
      rect("입술옆 가리개<", 92, 58, 95, 62, [240, 210, 190, 255]),
      rect("아랫입술 색", 95, 61, 105, 63, [200, 100, 100, 255]),
      rect("윗입술 선", 94, 57, 106, 58, [60, 30, 30, 255]),
    ]),
    group("입 가이드", [rect("A", 95, 56, 105, 64, [0, 0, 0, 255], { hidden: true })]),
    group("코", [rect("코", 99, 52, 101, 54, [200, 150, 130, 255])]),
    group("앞옆머리", [
      rect("앞머리 중앙", 70, 5, 130, 30, [90, 60, 140, 255]),
      rect("옆머리<", 30, 10, 55, 90, [90, 60, 140, 255]),
      rect("옆머리>", 145, 10, 170, 90, [90, 60, 140, 255]),
      rect("더듬이", 98, 0, 102, 5, [90, 60, 140, 255]),
      rect("하이라이트", 150, 10, 160, 12, [200, 180, 255, 255]),
    ]),
    rect("메모", 0, 0, 10, 10, [255, 0, 0, 255]),
    rect("광원 효과", 0, 90, 10, 100, [255, 255, 0, 255], { blendMode: "overlay" }),
  ]);
}

describe("B1 layered PSD — parsing", () => {
  it("reads the group tree, hidden flags, offsets, opacity, blend and clipping bases", () => {
    const psd = parsePsd(koreanPsd(), "himikan-like.psd");
    expect(psd.name).toBe("himikan-like");
    expect([psd.width, psd.height]).toEqual([200, 100]);
    const iris = psd.layers.find((n) => n.path === "/눈>/눈동자>")!;
    expect(iris.clipping).toBe(true);
    expect(iris.clipBase).toBe("/눈>/눈흰자>");
    // The highlight clips to the same base (nearest non-clipped layer below).
    expect(psd.layers.find((n) => n.path === "/눈>/눈 하이라이트>")!.clipBase).toBe("/눈>/눈흰자>");
    const blush = psd.layers.find((n) => n.path === "/얼굴/홍조")!;
    expect(blush.blendMode).toBe("multiply");
    expect(blush.clipBase).toBe("/얼굴/얼굴");
    const guide = psd.layers.find((n) => n.path === "/입 가이드/A")!;
    expect(guide.effectiveHidden).toBe(true);
    expect(guide.guide).toBe(true);
    const ear = psd.layers.find((n) => n.path === "/얼굴/귀<")!;
    expect(ear.bounds).toEqual({ left: 52, top: 40, right: 60, bottom: 55 });
    expect(ear.centroidX).toBeCloseTo(56, 5);
    // Pixel layers come bottom → top.
    expect(psd.layers[0].path).toBe("/뒷머리/뒷머리");
    expect(psd.layers.at(-1)!.path).toBe("/광원 효과");
  });

  it("fails loudly instead of returning empty layers", () => {
    expect(() => parsePsd(new Uint8Array([1, 2, 3, 4]))).toThrow(/PSD/);
  });
});

describe("B1 layered PSD — mapping suggestions", () => {
  it("maps Korean Live2D names with character-relative L/R from geometry", () => {
    const psd = parsePsd(koreanPsd());
    const s = suggestMapping(psd);
    const m = s.mappings;
    expect(s.midlineSource).toBe("face");
    expect(s.midlineX).toBeCloseTo(100, 5);
    expect(target(m, "/뒷머리/뒷머리")).toBe("hair_back");
    expect(target(m, "/몸/카라 뒤")).toBe("clothes");
    expect(target(m, "/몸/목")).toBe("neck");
    expect(target(m, "/몸/몸통")).toBe("clothes");
    expect(target(m, "/몸/리본")).toBe("clothes_inner");
    expect(target(m, "/얼굴/귀<")).toBe("face");
    expect(target(m, "/얼굴/얼굴")).toBe("face");
    expect(target(m, "/얼굴/홍조")).toBe("face");
    // Screen-right `>` eye = character left (AGENTS.md 1절).
    expect(target(m, "/눈>/눈흰자>")).toBe("eyeL_white");
    expect(target(m, "/눈>/눈동자>")).toBe("eyeL_iris");
    expect(target(m, "/눈>/눈 하이라이트>")).toBe("eyeL_iris");
    expect(target(m, "/눈>/눈언더>")).toBe("eyeL_lower");
    expect(target(m, "/눈>/속눈썹>")).toBe("eyeL_upper"); // 속눈썹 ⊃ 눈썹: not a brow
    expect(target(m, "/눈</눈흰자<")).toBe("eyeR_white");
    expect(target(m, "/눈</눈꺼풀<")).toBe("eyeR_upper");
    expect(target(m, "/눈썹/눈썹>")).toBe("browL");
    expect(target(m, "/눈썹/눈썹<")).toBe("browR");
    expect(target(m, "/입/입안")).toBe("mouth_inner");
    expect(target(m, "/입/아랫입술 색")).toBe("mouth_lower");
    expect(target(m, "/입/윗입술 선")).toBe("mouth_upper");
    expect(target(m, "/코/코")).toBe("face_detail");
    expect(target(m, "/앞옆머리/앞머리 중앙")).toBe("hair_front");
    expect(target(m, "/앞옆머리/더듬이")).toBe("hair_front");
    expect(target(m, "/앞옆머리/옆머리<")).toBe("hair_side_R");
    expect(target(m, "/앞옆머리/옆머리>")).toBe("hair_side_L");
    // Generic name takes its group's rule.
    expect(byName(m, "/앞옆머리/하이라이트").source).toBe("group");
    expect(target(m, "/앞옆머리/하이라이트")).toBe("hair_side_L");
    // Ambiguous ones stay unassigned for the user.
    expect(byName(m, "/눈>/쌍커풀>").target).toBeNull();
    expect(byName(m, "/입/입술옆 가리개<").target).toBeNull();
    expect(byName(m, "/눈>/쌍커풀>").include).toBe(true);
  });

  it("excludes hidden and guide layers by default but keeps them listed", () => {
    const s = suggestMapping(parsePsd(koreanPsd()));
    const guide = byName(s.mappings, "/입 가이드/A");
    expect(guide.include).toBe(false);
    expect(guide.target).toBeNull();
    expect(guide.note).toMatch(/숨김/);
    const memo = byName(s.mappings, "/메모");
    expect(memo.include).toBe(false);
    expect(memo.note).toMatch(/가이드/);
  });

  it("never turns a non-normal blend into normal silently", () => {
    const s = suggestMapping(parsePsd(koreanPsd()));
    const blush = byName(s.mappings, "/얼굴/홍조");
    expect(blush.blend).toBe("fold");
    expect(blush.include).toBe(true);
    const overlay = byName(s.mappings, "/광원 효과");
    expect(overlay.blend).toBe("normal");
    expect(overlay.include).toBe(false);
    expect(overlay.note).toMatch(/overlay/);
    const msgs = s.warnings.filter((w) => w.kind === "blend").map((w) => w.message);
    expect(msgs.some((t) => t.includes("홍조") && t.includes("multiply"))).toBe(true);
    // The overlay layer is excluded by default; its warning is in its note, not a silent drop.
    expect(overlay.note).toMatch(/기본 제외/);
  });

  it("maps Japanese and English names; side from pixels even when the name says otherwise", () => {
    const bytes = psdBytes(200, 100, [
      rect("後ろ髪", 40, 5, 160, 95),
      rect("首", 90, 60, 110, 80),
      rect("顔", 60, 15, 140, 70),
      // Named 右 (right) but drawn on the screen right = character left.
      rect("白目_右", 110, 40, 130, 48, [255, 255, 255, 255]),
      rect("瞳 右", 115, 39, 125, 49),
      rect("下まつげ右", 112, 48, 128, 49),
      rect("まつげ右", 108, 37, 132, 40),
      rect("眉 右", 110, 30, 130, 33),
      rect("口", 95, 57, 105, 63),
      rect("前髪", 70, 5, 130, 30),
      rect("Eye White L", 70, 40, 90, 48),
      rect("Iris L", 75, 39, 85, 49),
      rect("Eyelash L", 68, 37, 92, 40),
      rect("Brow L", 70, 30, 90, 33),
      rect("Side Hair", 30, 10, 55, 90),
      rect("Front Hair", 70, 0, 130, 5),
    ]);
    const m = suggestMapping(parsePsd(bytes)).mappings;
    expect(target(m, "/後ろ髪")).toBe("hair_back");
    expect(target(m, "/首")).toBe("neck");
    expect(target(m, "/顔")).toBe("face");
    expect(target(m, "/白目_右")).toBe("eyeL_white");
    expect(target(m, "/瞳 右")).toBe("eyeL_iris");
    expect(target(m, "/下まつげ右")).toBe("eyeL_lower");
    expect(target(m, "/まつげ右")).toBe("eyeL_upper");
    expect(target(m, "/眉 右")).toBe("browL");
    expect(target(m, "/口")).toBe("mouth_all");
    expect(target(m, "/前髪")).toBe("hair_front");
    // Named L but drawn on the screen left = character right.
    expect(target(m, "/Eye White L")).toBe("eyeR_white");
    expect(target(m, "/Iris L")).toBe("eyeR_iris");
    expect(target(m, "/Eyelash L")).toBe("eyeR_upper");
    expect(target(m, "/Brow L")).toBe("browR");
    expect(target(m, "/Side Hair")).toBe("hair_side_R");
    expect(target(m, "/Front Hair")).toBe("hair_front");
  });

  it("numbers hair parts of the same role in different groups (variable hair ids)", () => {
    const bytes = psdBytes(200, 100, [
      rect("얼굴", 60, 15, 140, 70),
      group("트윈테일 묶음 1", [rect("뒷머리", 20, 10, 60, 95)]),
      group("트윈테일 묶음 2", [rect("뒷머리", 140, 10, 180, 95), rect("뒷머리 그림자", 150, 10, 170, 95)]),
      group("앞", [rect("앞머리", 70, 5, 130, 30)]),
      group("앞2", [rect("앞머리", 60, 0, 140, 8)]),
    ]);
    const psd = parsePsd(bytes);
    const m = suggestMapping(psd).mappings;
    expect(target(m, "/트윈테일 묶음 1/뒷머리")).toBe("hair_back");
    expect(target(m, "/트윈테일 묶음 2/뒷머리")).toBe("hair_back_2");
    expect(target(m, "/트윈테일 묶음 2/뒷머리 그림자")).toBe("hair_back_2");
    expect(target(m, "/앞/앞머리")).toBe("hair_front");
    expect(target(m, "/앞2/앞머리")).toBe("hair_front_2");
    const built = buildLayerParts(psd, m, { midlineX: 100 });
    expect(built.parts.map((p) => p.id)).toEqual(["face", "hair_back", "hair_back_2", "hair_front", "hair_front_2"]);
    expect(inferPartRole("hair_back_2")).toBe("hair_back");
    expect(describeDynamicPart("hair_front_2").name).toMatch(/2$/);
  });

  it("recognises See-through output and proposes L/R splits for combined layers", () => {
    const blob = (x0: number) => (x: number, y: number): Px =>
      (x >= x0 && x < x0 + 10) || (x >= x0 + 40 && x < x0 + 50) ? [40, 60, 160, y < 100 ? 255 : 0] : [0, 0, 0, 0];
    const bytes = psdBytes(200, 100, [
      rect("back hair", 40, 5, 160, 95),
      rect("face", 60, 15, 140, 70),
      pixelsLayer("irides", 70, 40, 60, 8, blob(5)),
      rect("front hair", 70, 5, 130, 30),
      rect("headwear", 60, 0, 140, 6),
    ]);
    const psd = parsePsd(bytes);
    expect(psd.seeThrough).toBe(true);
    const s = suggestMapping(psd);
    expect(target(s.mappings, "/irides")).toBe("eye{S}_iris");
    expect(byName(s.mappings, "/irides").source).toBe("seethrough");
    expect(target(s.mappings, "/headwear")).toBe("hat");
    expect(s.warnings.some((w) => w.kind === "split")).toBe(true);
    const built = buildLayerParts(psd, s.mappings, { midlineX: s.midlineX });
    const ids = built.parts.map((p) => p.id);
    expect(ids).toContain("eyeL_iris");
    expect(ids).toContain("eyeR_iris");
    const L = built.parts.find((p) => p.id === "eyeL_iris")!;
    const R = built.parts.find((p) => p.id === "eyeR_iris")!;
    // Character L = screen right: the blob at x 115..125, R = x 75..85.
    const xs = (p: typeof L) => {
      const full = partCanvasRgba(p, built.width, built.height);
      const out = new Set<number>();
      for (let i = 0; i < built.width * built.height; i++) if (full[i * 4 + 3]) out.add(i % built.width);
      return [Math.min(...out), Math.max(...out)];
    };
    expect(xs(L)).toEqual([115, 124]);
    expect(xs(R)).toEqual([75, 84]);
    expect(built.splits[0]).toMatchObject({ path: "/irides", L: 1, R: 1 });
  });

  it("lists default parts with no layer as missing (never synthesized)", () => {
    const s = suggestMapping(parsePsd(koreanPsd()));
    expect(missingDefaultParts(s.mappings)).toEqual(["body"]);
  });
});

describe("B1 layered PSD — building parts", () => {
  it("places offset layers, multiplies group and layer opacity, and keeps the canvas size", () => {
    const bytes = psdBytes(10, 10, [
      group("g", [rect("얼굴", 5, 3, 7, 5, [255, 0, 0, 255], { opacity: 0.5 })], { opacity: 0.5 }),
    ]);
    const psd = parsePsd(bytes);
    const s = suggestMapping(psd);
    const built = buildLayerParts(psd, s.mappings, { midlineX: 5 });
    expect([built.width, built.height]).toEqual([10, 10]);
    const face = partCanvasRgba(built.parts[0], 10, 10);
    const a = (x: number, y: number) => face[(y * 10 + x) * 4 + 3];
    expect(a(5, 3)).toBe(64); // 255 × 0.5 × 0.5
    expect(a(6, 4)).toBe(64);
    expect(a(4, 3)).toBe(0);
    expect(a(7, 3)).toBe(0);
    expect(face[(3 * 10 + 5) * 4]).toBe(255);
  });

  it("merges several layers into one part in PSD order and stacks parts by their topmost layer", () => {
    const bytes = psdBytes(4, 1, [
      rect("카라 뒤", 0, 0, 4, 1, [10, 10, 10, 255]),
      rect("목", 1, 0, 3, 1, [200, 150, 120, 255]),
      rect("카라 앞", 0, 0, 1, 1, [250, 250, 250, 255]),
    ]);
    const psd = parsePsd(bytes);
    const s = suggestMapping(psd);
    const built = buildLayerParts(psd, s.mappings, { midlineX: 2 });
    expect(built.parts.map((p) => p.id)).toEqual(["neck", "clothes"]);
    const clothes = built.parts[1];
    expect(clothes.layers).toEqual(["/카라 뒤", "/카라 앞"]);
    // Top layer of the part wins at x = 0.
    expect([...clothes.rgba.subarray(0, 4)]).toEqual([250, 250, 250, 255]);
    expect(built.interleaved).toEqual([{ id: "clothes", between: ["neck"] }]);
  });

  it("clips to a base in the same part and keeps own alpha when the base is in another part", () => {
    const bytes = psdBytes(4, 1, [
      rect("얼굴", 0, 0, 2, 1, [240, 210, 190, 255]),
      rect("홍조", 0, 0, 4, 1, [240, 100, 100, 255], { clipping: true }),
      rect("눈흰자", 0, 0, 1, 1, [255, 255, 255, 255]),
      rect("눈동자", 0, 0, 3, 1, [40, 60, 160, 255], { clipping: true }),
    ]);
    const psd = parsePsd(bytes);
    const mappings: LayerMapping[] = psd.layers.map((n) => ({
      path: n.path,
      target: { "/얼굴": "face", "/홍조": "face", "/눈흰자": "eyeL_white", "/눈동자": "eyeL_iris" }[n.path]!,
      include: true,
      source: "manual",
    }));
    const built = buildLayerParts(psd, mappings, { midlineX: 0 });
    const face = built.parts.find((p) => p.id === "face")!;
    const iris = built.parts.find((p) => p.id === "eyeL_iris")!;
    expect(face.width).toBe(4);
    expect([...partCanvasRgba(face, 4, 1)].filter((_, k) => k % 4 === 3)).toEqual([255, 255, 0, 0]);
    expect(iris.width).toBe(3); // base is another part: own alpha kept
    expect(built.clipKeptOwnAlpha).toEqual(["/눈동자"]);
  });

  it("folds a multiply layer into its part and composites `normal` only when the user picks it", () => {
    const bytes = psdBytes(2, 1, [
      rect("얼굴", 0, 0, 2, 1, [200, 200, 200, 255]),
      rect("홍조", 0, 0, 1, 1, [128, 128, 128, 255], { blendMode: "multiply" as BlendMode }),
      rect("광원", 1, 0, 2, 1, [255, 255, 255, 128], { blendMode: "overlay" as BlendMode }),
    ]);
    const psd = parsePsd(bytes);
    const s = suggestMapping(psd);
    expect(byName(s.mappings, "/광원").include).toBe(false);
    s.mappings = s.mappings.map((m) => (m.path === "/광원" ? { ...m, target: "face" } : m));
    let face = buildLayerParts(psd, s.mappings, { midlineX: 1 }).parts[0];
    expect(face.rgba[0]).toBe(Math.round((200 * 128) / 255)); // W3C multiply over opaque
    expect(face.rgba[4]).toBe(200); // overlay layer excluded
    s.mappings = s.mappings.map((m) => (m.path === "/광원" ? { ...m, include: true, blend: "normal" } : m));
    face = buildLayerParts(psd, s.mappings, { midlineX: 1 }).parts[0];
    expect(face.rgba[4]).toBe(Math.round(200 + (255 - 200) * (128 / 255))); // composited as normal
  });

  it("splits visible / hidden alpha exactly like Rust layer_input (shared fixture)", () => {
    for (const c of layerSplitParity.cases) {
      const n = c.width * c.height;
      const alphas = c.layers.map((l) => Uint8Array.from(l.rgba.filter((_, k) => k % 4 === 3)));
      const { masks, fills } = splitVisibleFill(alphas, n);
      masks.forEach((m, p) => expect([...m], `${c.name} mask ${p}`).toEqual(c.expected.masks[p]));
      fills.forEach((f, p) => expect([...f], `${c.name} fill ${p}`).toEqual(c.expected.fills[p]));
      const comp = compositeStraight(c.layers.map((l) => Uint8Array.from(l.rgba)), n);
      expect([...comp], `${c.name} composite`).toEqual(c.expected.composite);
    }
  });

  it("gives the part stack visible masks that sum to the composite alpha and effective = part alpha", () => {
    const psd = parsePsd(koreanPsd());
    const s = suggestMapping(psd);
    const b = buildLayerParts(psd, s.mappings, { midlineX: s.midlineX });
    expect(b.maskSum.matches).toBe(true);
    expect(b.maskSum.pxOverOneLevel).toBe(0);
    let mismatches = 0;
    b.parts.forEach((p, i) => {
      const full = partCanvasRgba(p, b.width, b.height);
      for (let k = 0; k < b.width * b.height; k++)
        if (Math.max(b.masks[i][k], b.fills[i][k]) !== full[k * 4 + 3]) mismatches++;
    });
    expect(mismatches).toBe(0);
  });

  it("downscales with the long-side rule using box averaging of premultiplied colour", () => {
    expect(workingSize(2000, 3000)).toEqual({ width: 1365, height: 2048 });
    expect(workingSize(683, 1024)).toEqual({ width: 683, height: 1024 });
    // 4×2 PSD → 2×1: left pixel averages an opaque red and a transparent pixel → alpha 128, red.
    const bytes = psdBytes(4, 2, [
      pixelsLayer("얼굴", 0, 0, 4, 2, (x) => (x === 0 ? [255, 0, 0, 255] : x === 1 ? [0, 0, 0, 0] : [0, 0, 255, 255])),
    ]);
    const psd = parsePsd(bytes);
    const b = buildLayerParts(psd, suggestMapping(psd).mappings, { midlineX: 2, width: 2, height: 1 });
    const full = partCanvasRgba(b.parts[0], 2, 1);
    expect([...full]).toEqual([255, 0, 0, 128, 0, 0, 255, 255]);
  });
});
