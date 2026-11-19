"""B4 evaluation only: rest render vs an independent float 'normal' over-composite of the layer PNGs
(bottom -> top in the run's order), and vs the run's composite.png.

  artifacts/seethrough/venv/Scripts/python.exe -I tools/evaluation/b4_rest_check.py LAYER_DIR RUN_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t9_report import de2000, srgb_to_lab  # noqa: E402


def over_white(rgba):
    a = rgba[..., 3:4] / 255.0
    return rgba[..., :3] * a + 255.0 * (1 - a)


def stats(x, y):
    ax, ay = x[..., 3], y[..., 3]
    m = (ax > 0) | (ay > 0)
    da = np.abs(ax - ay)
    de = de2000(srgb_to_lab(np.round(over_white(x)).astype(np.uint8)), srgb_to_lab(np.round(over_white(y)).astype(np.uint8)))[m]
    return {"alphaMax": float(da.max()), "alphaGt2": int((da > 2).sum()), "dE00Mean": round(float(de.mean()), 4),
            "dE00Max": round(float(de.max()), 3), "dE00Gt1": int((de > 1).sum()), "dE00Gt2": int((de > 2).sum()), "dE00Gt5": int((de > 5).sum())}


def main():
    layer_dir, run = Path(sys.argv[1]), Path(sys.argv[2])
    st = json.loads((run / "stats.json").read_text(encoding="utf-8"))
    order = [r["id"] for r in json.loads((run / "rig.json").read_text(encoding="utf-8"))]
    acc = None
    for pid in order:
        t = np.asarray(Image.open(run / "parts" / f"{pid}.png").convert("RGBA")).astype(np.float64)
        a = t[..., 3:4] / 255.0
        premul = np.concatenate([t[..., :3] * a, a * 255.0], -1)
        acc = premul if acc is None else premul + acc * (1 - a)
    A = acc[..., 3:4] / 255.0
    straight = np.concatenate([np.where(A > 0, acc[..., :3] / np.maximum(A, 1e-9), 0), acc[..., 3:4]], -1)
    ref = np.round(straight)
    rest = np.asarray(Image.open(run / "poses" / "rest.png").convert("RGBA")).astype(np.float64)
    comp = np.asarray(Image.open(run / "composite.png").convert("RGBA")).astype(np.float64)
    print(json.dumps({"run": str(run), "restVsFloatOver": stats(rest, ref), "compositePngVsFloatOver": stats(comp, ref)}, indent=1))


if __name__ == "__main__":
    main()
