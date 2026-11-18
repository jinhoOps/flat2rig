"""T15 face / neck edge comparison images (report helper only).

For every T15 extreme-pose render (`lineart/pose_*.png`: side-hair sway ±1, head rotation ±30,
head/body opposite turns, hair sign combinations) crops the face's left / right edge and the neck
from BEFORE (`--no-lineart`, same binary) and AFTER (T15) and writes one sheet per pose:
rows = windows, columns = before | after (nearest ×SCALE).

  tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/t15_compare.py BEFORE_DIR AFTER_DIR OUT_DIR [SCALE]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    before, after, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    scale = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((after / "stats.json").read_text(encoding="utf-8"))
    lm = s["landmarks"]
    ex, ey = (lm["eyeR"]["x"] + lm["eyeL"]["x"]) / 2, (lm["eyeR"]["y"] + lm["eyeL"]["y"]) / 2
    ed = ((lm["eyeL"]["x"] - lm["eyeR"]["x"]) ** 2 + (lm["eyeL"]["y"] - lm["eyeR"]["y"]) ** 2) ** 0.5
    cx, cy = lm["chin"]["x"], lm["chin"]["y"]
    mid_y = (ey + cy) / 2
    r = int(0.5 * ed)
    windows = {
        "face_R_edge": (lm["eyeR"]["x"] - 0.75 * ed, mid_y),
        "face_L_edge": (lm["eyeL"]["x"] + 0.75 * ed, mid_y),
        "neck": (cx, cy + 0.45 * ed),
    }
    sheets = []
    for pose in sorted((after / "lineart").glob("pose_*.png")):
        b = before / "lineart" / pose.name
        if not b.exists():
            continue
        ib = Image.open(b).convert("RGBA")
        ia = Image.open(pose).convert("RGBA")
        tiles = []
        for name, (x, y) in windows.items():
            box = (int(x) - r, int(y) - r, int(x) + r, int(y) + r)
            row = []
            for im in (ib, ia):
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                bg.alpha_composite(im)
                row.append(bg.crop(box).resize((2 * r * scale, 2 * r * scale), Image.NEAREST))
            tiles.append((name, row))
        tw = 2 * r * scale
        sheet = Image.new("RGB", (2 * tw + 12, len(tiles) * (tw + 18) + 18), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        d.text((4, 2), f"{pose.stem}: before (--no-lineart) | after (T15)", fill=(0, 0, 0))
        for k, (name, row) in enumerate(tiles):
            y0 = 18 + k * (tw + 18)
            d.text((4, y0), name, fill=(0, 0, 0))
            sheet.paste(row[0].convert("RGB"), (0, y0 + 14))
            sheet.paste(row[1].convert("RGB"), (tw + 12, y0 + 14))
        path = out / f"{pose.stem}.png"
        sheet.save(path)
        sheets.append(path.as_posix())
    print("\n".join(sheets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
