import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { Part } from "../src/types";
import {
  layerRows,
  partHasFill,
  selectLayerRow,
  selectLayerThumb,
  type LayerSelection,
} from "../src/workspace/layers";
import { combineProgress, progressPercent } from "../src/workspace/progress";
import { resolveShortcut, stepBrushRadius, type ShortcutContext } from "../src/workspace/shortcuts";
import { CAT_FRAMES, CAT_H, CAT_PALETTE, CAT_W, frameRuns } from "../src/workspace/catSprite";
import CatProgress from "../src/workspace/CatProgress";
import { blendDisplay } from "../src/workspace/LayersPanel";

const W = 4,
  H = 4;
function part(id: string, order: number | undefined, extra: Partial<Part> = {}): Part {
  return {
    id,
    name: id,
    group: "g",
    color: "#888888",
    visible: true,
    status: "draft",
    mask: new Uint8Array(W * H),
    compositeOrder: order,
    ...extra,
  };
}

describe("T11a layer panel rows", () => {
  it("lists every part top-first in composite order, including variable hair layers", () => {
    const fill = new Uint8Array(W * H);
    fill[3] = 255;
    const parts = [
      part("hair_back", 0),
      part("hair_back_2", 1),
      part("body", 2),
      part("face", 3, { fillMask: fill }),
      part("hair_side_L_3", 5),
      part("hair_front_2", 6, { role: "hair_front" }),
      part("hair_front", 4),
    ];
    const rows = layerRows(parts);
    expect(rows.map((r) => r.part.id)).toEqual([
      "hair_front_2",
      "hair_side_L_3",
      "hair_front",
      "face",
      "body",
      "hair_back_2",
      "hair_back",
    ]);
    // Stack position counts from the bottom; roles come from the role/id, not a fixed 18-part list.
    expect(rows[0].order).toBe(6);
    expect(rows.at(-1)!.order).toBe(0);
    const byId = Object.fromEntries(rows.map((r) => [r.part.id, r]));
    expect(byId.hair_side_L_3.role).toBe("hair_side_L");
    expect(byId.hair_back_2.role).toBe("hair_back");
    expect(byId.hair_front_2.roleLabel).toBe("앞머리");
    expect(byId.face.hasFill).toBe(true);
    expect(byId.body.hasFill).toBe(false);
  });

  it("falls back to array order when compositeOrder is missing (as the renderer does)", () => {
    const rows = layerRows([part("a", undefined), part("b", undefined), part("c", undefined)]);
    expect(rows.map((r) => r.part.id)).toEqual(["c", "b", "a"]);
  });

  it("detects hidden-area pixels and ignores an all-zero fillMask", () => {
    expect(partHasFill({ fillMask: undefined })).toBe(false);
    expect(partHasFill({ fillMask: new Uint8Array(8) })).toBe(false);
    const m = new Uint8Array(8);
    m[7] = 1;
    expect(partHasFill({ fillMask: m })).toBe(true);
  });

  it("shows only normal as a supported blend mode", () => {
    expect(blendDisplay(undefined)).toEqual({ label: "일반 (Normal)", supported: true });
    expect(blendDisplay("multiply").supported).toBe(false);
    expect(blendDisplay("additive").label).toContain("PSD 미지원");
  });
});

describe("T11a edit target (visible / hidden)", () => {
  const fill = new Uint8Array(W * H);
  fill[0] = 200;
  const withFill = { part: part("face", 0, { fillMask: fill }), hasFill: true };
  const noFill = { part: part("mouth", 1), hasFill: false };
  const start: LayerSelection = { selected: "body", editTarget: "visible" };

  it("hidden thumbnail selects the row and targets the hidden area", () => {
    expect(selectLayerThumb(start, withFill, "fill")).toEqual({ selected: "face", editTarget: "hidden" });
  });
  it("texture / visible-mask thumbnails target the visible area", () => {
    const hidden: LayerSelection = { selected: "face", editTarget: "hidden" };
    expect(selectLayerThumb(hidden, withFill, "mask").editTarget).toBe("visible");
    expect(selectLayerThumb(hidden, withFill, "texture").editTarget).toBe("visible");
  });
  it("a part without hidden pixels cannot become a hidden target", () => {
    expect(selectLayerThumb(start, noFill, "fill")).toEqual({ selected: "mouth", editTarget: "visible" });
    const hidden: LayerSelection = { selected: "face", editTarget: "hidden" };
    expect(selectLayerRow(hidden, noFill).editTarget).toBe("visible");
    expect(selectLayerRow(hidden, withFill).editTarget).toBe("hidden");
  });
});

describe("T11a task progress", () => {
  it("is indeterminate with no steps, or a single front-end step and no Rust events", () => {
    expect(combineProgress(null, null).fraction).toBeNull();
    expect(combineProgress({ done: 0, total: 1, step: "x" }, null)).toEqual({ fraction: null, step: "x" });
  });
  it("uses Rust stage events alone, or nested inside front-end steps", () => {
    expect(combineProgress(null, { done: 1, total: 3, step: "얼굴 검출" })).toEqual({
      fraction: 1 / 3,
      step: "얼굴 검출",
    });
    // step 2 of 2 on the front end, Rust half done inside it -> 0.5 + 0.5 * 0.5
    const c = combineProgress({ done: 1, total: 2, step: "얼굴 검출" }, { done: 1, total: 2, step: "28점" });
    expect(c.fraction).toBeCloseTo(0.75);
    expect(c.step).toBe("얼굴 검출 · 28점");
    expect(combineProgress({ done: 2, total: 4 }, null).fraction).toBe(0.5);
    expect(progressPercent(0.756)).toBe(76);
    expect(progressPercent(null)).toBeNull();
    expect(combineProgress({ done: 9, total: 2 }, { done: 5, total: 1 }).fraction).toBe(1);
  });
});

describe("T11a pixel cat", () => {
  it("is an original 16×12 sprite with four run frames of known colours", () => {
    expect(CAT_FRAMES).toHaveLength(4);
    for (const frame of CAT_FRAMES) {
      expect(frame).toHaveLength(CAT_H);
      for (const row of frame) {
        expect(row).toHaveLength(CAT_W);
        for (const ch of row) expect(ch === "." || ch in CAT_PALETTE).toBe(true);
      }
    }
    // frames differ (legs move)
    expect(new Set(CAT_FRAMES.map((f) => f.join("|"))).size).toBe(4);
    const runs = frameRuns(CAT_FRAMES[0]);
    const pixels = runs.reduce((n, r) => n + r.w, 0);
    const painted = CAT_FRAMES[0].join("").replace(/\./g, "").length;
    expect(pixels).toBe(painted);
  });

  it("renders determinate with aria-valuenow and a filled bar", () => {
    const html = renderToStaticMarkup(
      <CatProgress label="파츠 생성 중" fraction={0.42} step="2단계" reducedMotion={false} />,
    );
    expect(html).toContain('role="progressbar"');
    expect(html).toContain('aria-valuenow="42"');
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain("cat-fill");
    expect(html).toContain('data-mode="determinate"');
    expect(html).toContain("42%");
    // four frames animate
    expect((html.match(/<rect /g) ?? []).length).toBeGreaterThan(
      frameRuns(CAT_FRAMES[0]).length,
    );
  });

  it("renders indeterminate without aria-valuenow", () => {
    const html = renderToStaticMarkup(
      <CatProgress label="PSD 생성 중" fraction={null} reducedMotion={false} />,
    );
    expect(html).toContain('role="progressbar"');
    expect(html).not.toContain("aria-valuenow");
    expect(html).not.toContain("cat-fill");
    expect(html).toContain('data-mode="indeterminate"');
    expect(html).toContain('aria-valuetext="진행 중"');
  });

  it("reduced motion shows a still cat (one frame) plus percent or step text", () => {
    const det = renderToStaticMarkup(
      <CatProgress label="x" fraction={0.5} step="결과 정리" reducedMotion />,
    );
    expect(det).toContain('data-still="true"');
    expect(det).toContain("is-still");
    expect(det).toContain("50%");
    expect(det).toContain("결과 정리");
    expect((det.match(/<rect /g) ?? []).length).toBe(frameRuns(CAT_FRAMES[0]).length);
    const ind = renderToStaticMarkup(<CatProgress label="x" fraction={null} reducedMotion />);
    expect(ind).toContain("진행 중");
  });

  it("the status-bar copy is decorative (not announced twice)", () => {
    const html = renderToStaticMarkup(
      <CatProgress label="x" fraction={0.1} compact decorative reducedMotion={false} />,
    );
    expect(html).not.toContain("progressbar");
    expect(html).toContain('aria-hidden="true"');
  });
});

describe("T11a shortcuts", () => {
  class FakeElement {
    tagName = "DIV";
    isContentEditable = false;
    getAttribute() {
      return null;
    }
  }
  class FakeInput extends FakeElement {
    tagName = "INPUT";
    constructor(public type: string) {
      super();
    }
  }
  class FakeTextArea extends FakeElement {
    tagName = "TEXTAREA";
  }
  class FakeSelect extends FakeElement {
    tagName = "SELECT";
  }
  afterEach(() => vi.unstubAllGlobals());
  const stubDom = () => {
    vi.stubGlobal("HTMLElement", FakeElement);
    vi.stubGlobal("HTMLInputElement", FakeInput);
    vi.stubGlobal("HTMLTextAreaElement", FakeTextArea);
    vi.stubGlobal("HTMLSelectElement", FakeSelect);
  };
  const ctx: ShortcutContext = {
    busy: false,
    hasProject: true,
    hasParts: true,
    editTools: true,
    rigStage: false,
    modal: false,
  };

  it("maps tool, navigation, brush and history keys", () => {
    stubDom();
    const div = new FakeElement();
    const k = (key: string, mods: Record<string, boolean> = {}) =>
      resolveShortcut({ key, target: div as unknown as EventTarget, ...mods }, ctx);
    expect(k("v")).toEqual({ type: "tool", tool: "select" });
    expect(k("B")).toEqual({ type: "tool", tool: "add" });
    expect(k("e")).toEqual({ type: "tool", tool: "erase" });
    expect(k("h")).toEqual({ type: "nav", nav: "hand" });
    expect(k("z")).toEqual({ type: "nav", nav: "zoom" });
    expect(k("[")).toEqual({ type: "brush", delta: -1 });
    expect(k("]")).toEqual({ type: "brush", delta: 1 });
    expect(k("z", { ctrlKey: true })).toEqual({ type: "undo" });
    expect(k("Z", { ctrlKey: true, shiftKey: true })).toEqual({ type: "redo" });
    expect(k("y", { metaKey: true })).toEqual({ type: "redo" });
    expect(k("s", { ctrlKey: true })).toEqual({ type: "save" });
    expect(k("Escape")).toEqual({ type: "escape" });
  });

  it("ignores every shortcut while typing in a text field or select", () => {
    stubDom();
    for (const target of [new FakeInput("text"), new FakeInput("search"), new FakeTextArea(), new FakeSelect()]) {
      for (const key of ["v", "b", "h", "z", "[", "1", " "]) {
        expect(resolveShortcut({ key, target: target as unknown as EventTarget }, ctx)).toBeNull();
      }
      expect(
        resolveShortcut({ key: "z", ctrlKey: true, target: target as unknown as EventTarget }, ctx),
      ).toBeNull();
    }
    const editable = Object.assign(new FakeElement(), { isContentEditable: true });
    expect(resolveShortcut({ key: "b", target: editable as unknown as EventTarget }, ctx)).toBeNull();
    // Range sliders and checkboxes are not text editing: shortcuts still work there.
    expect(
      resolveShortcut({ key: "b", target: new FakeInput("range") as unknown as EventTarget }, ctx),
    ).toEqual({ type: "tool", tool: "add" });
  });

  it("respects busy, stage and modal state", () => {
    stubDom();
    const t = new FakeElement() as unknown as EventTarget;
    expect(resolveShortcut({ key: "v", target: t }, { ...ctx, busy: true })).toBeNull();
    expect(resolveShortcut({ key: "z", ctrlKey: true, target: t }, { ...ctx, busy: true })).toBeNull();
    // rig stages: edit tools off, navigation and expressions on
    const rig = { ...ctx, editTools: false, rigStage: true };
    expect(resolveShortcut({ key: "b", target: t }, rig)).toBeNull();
    expect(resolveShortcut({ key: "h", target: t }, rig)).toEqual({ type: "nav", nav: "hand" });
    expect(resolveShortcut({ key: "3", target: t }, rig)).toEqual({ type: "expression", key: "3" });
    expect(resolveShortcut({ key: " ", code: "Space", target: t }, rig)).toEqual({ type: "togglePlay" });
    expect(resolveShortcut({ key: "v", target: t }, { ...ctx, modal: true })).toBeNull();
    expect(resolveShortcut({ key: "v", target: t }, { ...ctx, hasProject: false })).toBeNull();
  });

  it("steps the brush within the slider range", () => {
    expect(stepBrushRadius(18, 1)).toBe(20);
    expect(stepBrushRadius(5, -1)).toBe(4);
    expect(stepBrushRadius(2, -1)).toBe(2);
    expect(stepBrushRadius(98, 1)).toBe(100);
    expect(stepBrushRadius(60, -1)).toBe(55);
  });
});
