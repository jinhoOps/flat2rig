"""V1: texels of a part's effective texture that its ArtMesh does not cover at rest (evaluation only).

`draw_deformed_part` (flat2rig-cli) and the app renderer draw a part only inside its mesh
triangles, so an uncovered texel with alpha > 2 is lost in every pose, including rest.

  artifacts/seethrough/venv/Scripts/python.exe -I tools/evaluation/v1_mesh_coverage.py RUN_DIR [OUT.png]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    run = Path(sys.argv[1])
    rig = json.loads((run / "rig.json").read_text(encoding="utf-8"))
    total = 0
    lost_all = None
    rows = []
    for entry in rig:
        pid, mesh = entry["id"], entry["mesh"]
        tex = np.asarray(Image.open(run / "parts" / f"{pid}.png").convert("RGBA"))
        h, w = tex.shape[:2]
        if lost_all is None:
            lost_all = np.zeros((h, w), bool)
        cover = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(cover)
        v = mesh["vertices"]
        t = mesh["triangles"]
        for k in range(0, len(t), 3):
            pts = [(v[t[k + j]]["x"] - 0.5, v[t[k + j]]["y"] - 0.5) for j in range(3)]
            dr.polygon(pts, fill=255)
        cov = np.asarray(cover) > 0
        a = tex[:, :, 3]
        lost = (a > 2) & ~cov
        lost_strong = (a >= 128) & ~cov
        total += int(lost.sum())
        lost_all |= lost
        rows.append((pid, int((a > 2).sum()), int(lost.sum()), int(lost_strong.sum()), len(t) // 3, len(v)))
    print(f"{'part':<14}{'texels':>8}{'lost>2':>8}{'lost>=128':>10}{'tris':>6}{'verts':>6}")
    for r in rows:
        print(f"{r[0]:<14}{r[1]:>8}{r[2]:>8}{r[3]:>10}{r[4]:>6}{r[5]:>6}")
    print("total lost texels", total)
    if len(sys.argv) > 2:
        comp = run / "effective_foreground.png"
        base = np.asarray(Image.open(comp).convert("RGB")).astype(np.float32) * 0.4 + 150
        base[lost_all] = (255, 0, 255)
        Image.fromarray(base.clip(0, 255).astype(np.uint8)).save(sys.argv[2])


if __name__ == "__main__":
    main()
