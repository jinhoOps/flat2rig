"""V2: contact sheets for converted See-through layers (visual judgement aid, evaluation only).

<sheet>.png         input | recomposite | every part/extra layer (cropped to its bbox, on checker)
<sheet>_hidden.png  every layer with its hidden area (pixels covered by layers drawn above it, in
                    See-through draw order) in full colour and the visible area dimmed to 25 %.

  artifacts/seethrough/venv/Scripts/python.exe -I tools/evaluation/v2_sheets.py \
      --layers artifacts/v2/layers/himikan --src artifacts/seethrough/inputs/himikan.png \
      --out artifacts/v2/sheets/himikan.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TILE = 320
LABEL = 30


def checker(h, w, s=8):
    yy, xx = np.indices((h, w))
    c = np.where(((yy // s) + (xx // s)) % 2 == 0, 235, 205).astype(np.float32)
    return np.stack([c, c, c], -1)


def on_bg(rgb: np.ndarray, a: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = a[..., None]
    return (rgb.astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)


def bbox(mask, margin=10):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return (max(xs.min() - margin, 0), max(ys.min() - margin, 0),
            min(xs.max() + margin + 1, mask.shape[1]), min(ys.max() + margin + 1, mask.shape[0]))


def fit(img: np.ndarray) -> Image.Image:
    im = Image.fromarray(img)
    s = min(TILE / im.width, TILE / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))),
                   Image.NEAREST if s >= 1 else Image.LANCZOS)
    tile = Image.new("RGB", (TILE, TILE), (255, 255, 255))
    tile.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
    return tile


def grid(tiles, labels, path: Path, title: str):
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (TILE + 6), 40 + rows * (TILE + LABEL + 6)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    dr.text((6, 10), title, fill=(0, 0, 0))
    for i, (t, lb) in enumerate(zip(tiles, labels)):
        x, y = (i % cols) * (TILE + 6), 40 + (i // cols) * (TILE + LABEL + 6)
        sheet.paste(t, (x, y + LABEL))
        for j, line in enumerate(lb.split("\n")[:2]):
            dr.text((x + 3, y + 2 + j * 13), line, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d, out = Path(args.layers), Path(args.out)
    man = json.loads((d / "truth-manifest.json").read_text(encoding="utf-8"))
    files = {**man["parts"], **man.get("extraLayers", {})}
    order = [p for p in man["drawOrder"] if p in files]
    W, H = man["width"], man["height"]
    imgs = {p: np.asarray(Image.open(d / files[p]).convert("RGBA")) for p in order}

    src = Image.open(args.src).convert("RGBA").resize((W, H), Image.LANCZOS)
    src = np.asarray(src)
    rec = np.asarray(Image.open(d / "_recomposite.png").convert("RGBA"))
    tiles = [fit(on_bg(src[..., :3], src[..., 3] / 255, checker(H, W))),
             fit(on_bg(rec[..., :3], rec[..., 3] / 255, checker(H, W)))]
    labels = ["input", "recomposite (all layers)"]
    htiles, hlabels = list(tiles), list(labels)
    for i, pid in enumerate(order):
        im = imgs[pid]
        a = im[..., 3].astype(np.float32) / 255
        x0, y0, x1, y1 = bbox(a >= 0.5)
        crop = lambda v: v[y0:y1, x0:x1]  # noqa: E731
        tiles.append(fit(on_bg(crop(im[..., :3]), crop(a), checker(y1 - y0, x1 - x0))))
        labels.append(f"{i}: {pid}\n{int((a >= 0.5).sum())} px  bbox {x0},{y0}-{x1},{y1}")
        occ = np.zeros((H, W), bool)
        for q in order[i + 1:]:
            occ |= imgs[q][..., 3] >= 128
        hid = (a >= 0.5) & occ
        aa = np.where(hid, a, a * 0.25)
        bg = np.full((H, W, 3), 60, np.float32)
        htiles.append(fit(on_bg(crop(im[..., :3]), crop(aa), crop(bg))))
        frac = hid.sum() / max(1, (a >= 0.5).sum())
        hlabels.append(f"{i}: {pid}\nhidden {int(hid.sum())} px ({frac:.0%})")
    name = man.get("name", d.name)
    grid(tiles, labels, out, f"{name}: See-through layers -> flat2rig parts (draw order bottom->top)")
    grid(htiles, hlabels, out.with_name(out.stem + "_hidden.png"),
         f"{name}: hidden area = pixels covered by layers above (full colour); visible part dimmed to 25%")
    print(f"wrote {out} and {out.with_name(out.stem + '_hidden.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
