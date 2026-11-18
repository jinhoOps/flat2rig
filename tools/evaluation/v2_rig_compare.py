"""V2: side-by-side motion renders of flat2rig-cli runs (automatic path vs See-through layers, and
optionally the human-drawn truth layers) plus the CLI's own motion/seam/eye metrics.

Each column is a flat2rig-cli output directory (`motion_*.png`, `fill_only.png`, `stats.json`) on
the same canvas. For every render it writes
  <out>/<name>_<render>.png       full frame, columns side by side
  <out>/<name>_<render>_head.png  head crop (from the first column's landmarks), 2x nearest
on a dark teal background so transparent gaps (holes) show, and <out>/<name>_metrics.md/json.

  artifacts/seethrough/venv/Scripts/python.exe -I tools/evaluation/v2_rig_compare.py --name himikan \
      --col "auto (T15)=artifacts/t15/after/himikan" --col "See-through=artifacts/v2/rig/himikan_s42" \
      [--col "human layers (V1)=artifacts/v1/himikan-layers"] --out artifacts/v2/rig/compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RENDERS = ["motion_turn", "motion_blink", "motion_mouth", "motion_preview", "fill_only"]
BG = np.array([30, 90, 110], np.float32)
H_FULL = 860


def on_bg(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    im = np.asarray(Image.open(path).convert("RGBA"), np.float32)
    a = im[..., 3:4] / 255
    return Image.fromarray((im[..., :3] * a + BG * (1 - a)).astype(np.uint8))


def head_box(stats: dict, w: int, h: int):
    lm = stats.get("landmarks") or {}
    try:
        crown, chin = lm["crown"], lm["chin"]
        el, er = lm["eyeL"], lm["eyeR"]
    except KeyError:
        return 0, 0, w, h
    hh = max(chin["y"] - crown["y"], 8.0)
    cx = (el["x"] + er["x"]) / 2
    x0, x1 = cx - 1.05 * hh, cx + 1.05 * hh
    y0, y1 = crown["y"] - 0.55 * hh, chin["y"] + 0.75 * hh
    c = lambda v, m: int(max(0, min(m, round(v))))  # noqa: E731
    return c(x0, w), c(y0, h), c(x1, w), c(y1, h)


def row(images, labels, title, path: Path):
    gap, top = 8, 44
    W = sum(i.width for i in images) + gap * (len(images) - 1)
    H = max(i.height for i in images) + top
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    dr.text((4, 4), title, fill=(0, 0, 0))
    x = 0
    for im, lb in zip(images, labels):
        sheet.paste(im, (x, top))
        dr.text((x + 4, 24), lb, fill=(0, 0, 0))
        x += im.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def metrics(stats: dict) -> dict:
    sc, ex, es = stats.get("seamCoverage") or {}, stats.get("exposure") or {}, stats.get("eyeSeal") or {}
    holes = ex.get("holes") or {}
    li = stats.get("layerInput") or {}
    return {
        "landmarkSource": stats.get("landmarkSource"),
        "seamHoles": sc.get("holes"),
        "seamLowerMisses": sc.get("lowerMisses"),
        "seamChecks": sc.get("checks"),
        "exposureHolePxTotal": holes.get("holePxTotal"),
        "exposureHolePxMax": holes.get("holePxMax"),
        "exposureHolePoses": holes.get("holePoses"),
        "exposureWorstPose": holes.get("worstPose"),
        "exposedPxTotal": ex.get("exposedPxTotal"),
        "eyeLeakPxTotal": es.get("leakPxTotal"),
        "eyeLeakExcessPx": es.get("leakExcessPxTotal"),
        "eyeGapExcessPx": es.get("gapExcessPxTotal"),
        "eyeRowsWithDefect": es.get("rowsWithDefect"),
        "missingDefaultParts": li.get("missingDefaultParts"),
        "maskSumMaxDiff": (li.get("maskSum") or {}).get("maxAbsDiff"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--col", action="append", required=True, help='"label=dir"')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cols = [(c.split("=", 1)[0], Path(c.split("=", 1)[1])) for c in args.col]
    out = Path(args.out)
    stats = [json.loads((d / "stats.json").read_text(encoding="utf-8")) for _, d in cols]
    w, h = stats[0]["width"], stats[0]["height"]
    box = head_box(stats[0], w, h)
    for r in RENDERS:
        ims = [on_bg(d / f"{r}.png") for _, d in cols]
        if any(i is None for i in ims):
            print(f"skip {r}: missing in {[l for (l, _), i in zip(cols, ims) if i is None]}", file=sys.stderr)
            continue
        s = H_FULL / h
        full = [i.resize((round(w * s), H_FULL), Image.LANCZOS) for i in ims]
        row(full, [l for l, _ in cols], f"{args.name} · {r}.png (flat2rig-cli motion render; background = dark teal)", out / f"{args.name}_{r}.png")
        bw, bh = box[2] - box[0], box[3] - box[1]
        k = max(1, min(3, int(700 // max(bw, bh))))
        heads = [i.crop(box).resize((bw * k, bh * k), Image.NEAREST) for i in ims]
        row(heads, [l for l, _ in cols], f"{args.name} · {r}.png head crop {box} x{k}", out / f"{args.name}_{r}_head.png")
    res = {l: {"dir": str(d).replace("\\", "/"), **metrics(s)} for (l, d), s in zip(cols, stats)}
    (out / f"{args.name}_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = list(next(iter(res.values())).keys())[1:]
    lines = ["| metric | " + " | ".join(res) + " |", "| --- |" + " ---: |" * len(res)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(res[c][k]) for c in res) + " |")
    (out / f"{args.name}_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
