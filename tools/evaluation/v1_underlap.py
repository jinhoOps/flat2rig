"""V1: how far a lower part's drawn alpha reaches under its upper neighbour, per seam (evaluation only).

For a seam (upper U over lower L) the CLI reports `requiredUnderlapPx` (T13: sampled from
`deform` over the pose envelope). This measures what the artwork actually provides: among the
rest pixels whose visible owner is U (`owner.png`), grouped by chamfer distance d to L's visible
region, the fraction where L's effective alpha (`parts/<L>.png`) is > 0. `reach50` / `reach90`
are the largest d up to which the coverage stays ≥ 50 % / 90 %.

  artifacts/seethrough/venv/Scripts/python.exe -I tools/evaluation/v1_underlap.py \
      artifacts/v1/himikan-layers artifacts/v1/himikan-auto --out artifacts/v1/compare/underlap.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PAIRS = [
    ("clothes", "hair_back"),
    ("clothes", "neck"),
    ("hair_side_R", "hair_back"),
    ("hair_side_L", "hair_back"),
    ("hair_side_R", "clothes"),
    ("hair_side_L", "clothes"),
    ("hair_side_R", "clothes_inner"),
    ("hair_side_L", "clothes_inner"),
    ("face", "neck"),
    ("face", "hair_back"),
    ("hair_front", "face"),
    ("hair_front", "hair_back"),
]


def profile(run: Path, upper: str, lower: str, req: float | None) -> dict | None:
    stats = json.loads((run / "stats.json").read_text(encoding="utf-8"))
    ids = [m["id"] for m in stats["partsMeta"]]
    if upper not in ids or lower not in ids:
        return None
    owner = np.asarray(Image.open(run / "owner.png")).astype(np.int64) - 1
    u, l = ids.index(upper), ids.index(lower)
    eff_l = np.asarray(Image.open(run / "parts" / f"{lower}.png").convert("RGBA"))[:, :, 3] > 0
    l_vis = owner == l
    if not l_vis.any():
        return None
    dist = ndimage.distance_transform_edt(~l_vis)
    region = owner == u
    rows = []
    reach50 = reach90 = 0
    alive50 = alive90 = True
    for d in range(1, 121):
        sel = region & (dist > d - 0.5) & (dist <= d + 0.5)
        n = int(sel.sum())
        if n == 0:
            break
        cov = float(eff_l[sel].mean())
        rows.append((d, n, round(cov, 3)))
        if alive50 and cov >= 0.5:
            reach50 = d
        else:
            alive50 = False
        if alive90 and cov >= 0.9:
            reach90 = d
        else:
            alive90 = False
    return {
        "upper": upper,
        "lower": lower,
        "requiredUnderlapPx": req,
        "reach90Px": reach90,
        "reach50Px": reach50,
        "coverageByDistance": {str(d): c for d, _, c in rows if d in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 100)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {}
    for run in a.runs:
        stats = json.loads((run / "stats.json").read_text(encoding="utf-8"))
        req = {(s["upper"], s["lower"]): s for s in stats["seams"]}
        rows = []
        for up, lo in PAIRS:
            s = req.get((up, lo))
            p = profile(run, up, lo, s["requiredUnderlapPx"] if s else None)
            if p is None:
                continue
            if s:
                p.update(
                    maxRelativeDisplacementPx=s["maxRelativeDisplacementPx"],
                    worstPose=s["worstPose"],
                    lowerMisses=s["lowerMisses"],
                    holes=s["holes"],
                    maxHoleDepthPx=s["maxHoleDepthPx"],
                )
            rows.append(p)
        out[run.name] = rows
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, rows in out.items():
        print("==", name)
        for r in rows:
            print(
                f"{r['upper']:>12} > {r['lower']:<13} req {r['requiredUnderlapPx']!s:>6}  "
                f"reach90 {r['reach90Px']:>3}  reach50 {r['reach50Px']:>3}  "
                f"holes {r.get('holes', '-')!s:>4}  disp {r.get('maxRelativeDisplacementPx', 0) or 0:.1f}  "
                f"cov {r['coverageByDistance']}"
            )


if __name__ == "__main__":
    main()
