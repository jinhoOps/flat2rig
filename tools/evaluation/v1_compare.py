"""V1: side-by-side pose renders, finished layers vs automatic path (visual judgement aid only).

Both directories come from `flat2rig-cli --pose-renders` (`poses/<pose>.png`). Renders are put
on a saturated green so background holes (transparent texels inside the figure) stand out.

<out>/<pose>.png        layers | auto, full figure
<out>/<pose>_zoom.png   ×3 nearest crops (face, eyes, shoulder L/R, hair tips) layers over auto
<out>/holes.png         per pose: transparent texels inside the posed figure's filled hull

  artifacts/seethrough/venv/Scripts/python.exe -I tools/evaluation/v1_compare.py \
      --layers artifacts/v1/himikan-layers --auto artifacts/v1/himikan-auto \
      --out artifacts/v1/compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BG = (60, 220, 60)


def on_bg(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", rgba.size, BG + (255,))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 22), (255, 255, 255))
    out.paste(img, (0, 22))
    ImageDraw.Draw(out).text((4, 4), text, fill=(0, 0, 0))
    return out


def hstack(imgs: list[Image.Image], gap: int = 6) -> Image.Image:
    h = max(i.height for i in imgs)
    out = Image.new("RGB", (sum(i.width for i in imgs) + gap * (len(imgs) - 1), h), (255, 255, 255))
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def vstack(imgs: list[Image.Image], gap: int = 6) -> Image.Image:
    w = max(i.width for i in imgs)
    out = Image.new("RGB", (w, sum(i.height for i in imgs) + gap * (len(imgs) - 1)), (255, 255, 255))
    y = 0
    for i in imgs:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


def interior_holes(alpha: np.ndarray) -> np.ndarray:
    """Transparent texels enclosed by the figure (not reachable from the canvas border)."""
    from collections import deque

    h, w = alpha.shape
    empty = alpha < 8
    seen = np.zeros_like(empty)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and empty[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return empty & ~seen


def crops(lm: dict) -> dict[str, tuple[int, int, int, int]]:
    el, er, mo, ch = lm["eyeL"], lm["eyeR"], lm["mouth"], lm["chin"]
    d = el["x"] - er["x"]
    cx = (el["x"] + er["x"]) / 2

    def box(x, y, rw, rh):
        return (int(x - rw), int(y - rh), int(x + rw), int(y + rh))

    return {
        "eyeR": box(er["x"], er["y"], d * 0.45, d * 0.32),
        "eyeL": box(el["x"], el["y"], d * 0.45, d * 0.32),
        "mouth": box(mo["x"], mo["y"], d * 0.4, d * 0.3),
        "chin_neck": box(ch["x"], ch["y"] + d * 0.35, d * 0.8, d * 0.55),
        "shoulder_R": box(cx - d * 1.35, ch["y"] + d * 1.0, d * 0.8, d * 0.8),
        "shoulder_L": box(cx + d * 1.35, ch["y"] + d * 1.0, d * 0.8, d * 0.8),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=Path, required=True)
    ap.add_argument("--auto", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--zoom", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sl = json.loads((a.layers / "stats.json").read_text(encoding="utf-8"))
    sa = json.loads((a.auto / "stats.json").read_text(encoding="utf-8"))
    boxes = crops(sl["landmarks"])
    rows = []
    hole_rows = []
    for r in sl["poseRenders"]:
        f = r["file"]
        L = Image.open(a.layers / f).convert("RGBA")
        A = Image.open(a.auto / f).convert("RGBA")
        name = Path(f).stem
        half = lambda im: im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        hstack([label(half(on_bg(L)), f"layers  {r['pose']}"), label(half(on_bg(A)), f"auto  {r['pose']}")]).save(
            a.out / f"{name}.png"
        )
        tiles_l, tiles_a = [], []
        for cname, (x0, y0, x1, y1) in boxes.items():
            for src, dst in ((L, tiles_l), (A, tiles_a)):
                c = on_bg(src.crop((x0, y0, x1, y1)))
                c = c.resize((c.width * a.zoom, c.height * a.zoom), Image.NEAREST)
                dst.append(label(c, cname))
        vstack([label(hstack(tiles_l), "layers"), label(hstack(tiles_a), "auto")]).save(a.out / f"{name}_zoom.png")
        hl = interior_holes(np.asarray(L)[:, :, 3])
        ha = interior_holes(np.asarray(A)[:, :, 3])
        hole_rows.append({"pose": r["pose"], "layersInteriorHolePx": int(hl.sum()), "autoInteriorHolePx": int(ha.sum())})
        rows.append((r["pose"], L, A, hl, ha))
    # Hole overview: posed figure dimmed, interior holes in magenta, for the poses that have any.
    tiles = []
    for pose, L, A, hl, ha in rows:
        if hl.sum() == 0 and ha.sum() == 0:
            continue
        pair = []
        for im, hm, tag in ((L, hl, "layers"), (A, ha, "auto")):
            rgb = np.asarray(on_bg(im)).astype(np.float32) * 0.45 + 140
            rgb[hm] = (255, 0, 255)
            # Thicken for visibility at half size.
            out = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
            pair.append(label(out.resize((out.width // 2, out.height // 2), Image.NEAREST), f"{tag} {pose}: {int(hm.sum())}px"))
        tiles.append(hstack(pair))
    if tiles:
        grid = [hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
        vstack(grid).save(a.out / "holes.png")
    summary = {
        "poses": hole_rows,
        "seamCoverage": {
            "layers": {k: sl["seamCoverage"][k] for k in ("poses", "checks", "lowerMisses", "holes")},
            "auto": {k: sa["seamCoverage"][k] for k in ("poses", "checks", "lowerMisses", "holes")},
        },
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
