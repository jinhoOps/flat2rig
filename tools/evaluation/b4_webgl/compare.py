"""B4 evaluation only: per-pose difference between two render folders (e.g. CLI `rig::raster` vs app
WebGL from tools/evaluation/b4_webgl/run.mjs). Pairs files by name.

  artifacts/seethrough/venv/Scripts/python.exe -I tools/evaluation/b4_webgl/compare.py \
      A_DIR B_DIR OUT_JSON [--diff-dir DIR]

Per pose: alpha |diff| (max, mean, count > 2, > 8) and CIEDE2000 over white on pixels with alpha > 0
in either (mean, p99, max, count > 1 / > 2 / > 5), plus the share of compared pixels that are
bit-identical in RGBA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t9_report import de2000, srgb_to_lab  # noqa: E402


def over_white(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    return rgba[..., :3].astype(np.float64) * a + 255.0 * (1 - a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--diff-dir", type=Path)
    ap.add_argument("--b-name", default=None, help="map A file name to B file name: 'prefix' to prepend")
    args = ap.parse_args()
    rows = []
    for fa in sorted(args.a.glob("*.png")):
        fb = args.b / fa.name
        if not fb.exists():
            continue
        A = np.asarray(Image.open(fa).convert("RGBA"))
        B = np.asarray(Image.open(fb).convert("RGBA"))
        aa, ab = A[..., 3].astype(int), B[..., 3].astype(int)
        m = (aa > 0) | (ab > 0)
        da = np.abs(aa - ab)
        de = de2000(srgb_to_lab(np.round(over_white(A)).astype(np.uint8)), srgb_to_lab(np.round(over_white(B)).astype(np.uint8)))[m]
        same = np.all(A == B, axis=-1)[m]
        rows.append({
            "pose": fa.stem,
            "pixels": int(m.sum()),
            "identicalShare": round(float(same.mean()), 4),
            "alphaMax": int(da.max()),
            "alphaMean": round(float(da[m].mean()), 4),
            "alphaGt2": int((da > 2).sum()),
            "alphaGt8": int((da > 8).sum()),
            "dE00Mean": round(float(de.mean()), 4),
            "dE00P99": round(float(np.percentile(de, 99)), 3),
            "dE00Max": round(float(de.max()), 3),
            "dE00Gt1": int((de > 1).sum()),
            "dE00Gt2": int((de > 2).sum()),
            "dE00Gt5": int((de > 5).sum()),
        })
        if args.diff_dir:
            args.diff_dir.mkdir(parents=True, exist_ok=True)
            d = np.maximum(np.abs(over_white(A) - over_white(B)).max(-1), da)
            Image.fromarray(np.clip(d * 8, 0, 255).astype(np.uint8)).save(args.diff_dir / fa.name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
