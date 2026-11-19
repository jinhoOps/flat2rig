"""B4 report helper (evaluation only): before / after the shared renderer and the full-coverage
ArtMesh, for artifacts/b4/WORKER_REPORT.md.

  artifacts/seethrough/venv/Scripts/python.exe -I tools/evaluation/b4_report.py \
      --before artifacts/b4/himikan-before --after artifacts/b4/himikan-after \
      --eyeleak artifacts/b4/eyeleak \
      --default-before artifacts/b4/default-before --default-after artifacts/b4/default-after \
      --out artifacts/b4/compare

Writes into --out:
  <pose>.png, <pose>_zoom.png   29 pose renders, before | after (green = transparent), and x3
                                nearest crops (eyes, mouth, chin/neck, shoulders) before over after
  rest_vs_composite.png         rest render vs the straight layer composite, |diff| x8, per run
  eyeleak_<pose>.png            closed-eye crops: old/new mesh x app opacity / opacity 1, leak
                                pixels (eye white / iris visible at a closed eye) in magenta
  summary.json                  every number in the report
Numbers:
  - rest vs composite: alpha |diff| and CIEDE2000 over white, on pixels with alpha > 0 in either;
  - dark dots: 1 px pixels (over white) darker than all 8 neighbours by > 20 L* inside the figure,
    a proxy for double-composited shared edges;
  - eye leak: pixels on a closed eye's side whose colour changes (max channel > 2 levels over
    white) when the eye white / iris layers are dropped, i.e. eye interior visible at a closed eye;
  - mesh coverage: tools/evaluation/v1_mesh_coverage.py totals;
  - default run: SHA-256 per file before / after, stats.json keys that differ (time fields ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t9_report import de2000, srgb_to_lab  # noqa: E402
from v1_compare import crops, hstack, label, on_bg, vstack  # noqa: E402

PY = sys.executable


def over_white(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    return rgba[..., :3].astype(np.float64) * a + 255.0 * (1 - a)


def rest_vs_composite(run: Path) -> dict:
    rest = np.asarray(Image.open(run / "poses" / "rest.png").convert("RGBA"))
    comp = np.asarray(Image.open(run / "composite.png").convert("RGBA"))
    ar, ac = rest[..., 3].astype(int), comp[..., 3].astype(int)
    m = (ar > 0) | (ac > 0)
    da = np.abs(ar - ac)
    de = de2000(srgb_to_lab(np.round(over_white(rest)).astype(np.uint8)), srgb_to_lab(np.round(over_white(comp)).astype(np.uint8)))
    dm = de[m]
    return {
        "pixels": int(m.sum()),
        "alphaAbsDiffMax": int(da.max()),
        "alphaDiffGt2": int((da > 2).sum()),
        "alphaDiffGt0": int((da > 0).sum()),
        "restAlphaZeroWhereCompositeGt0": int(((ar == 0) & (ac > 0)).sum()),
        "dE00Mean": round(float(dm.mean()), 4),
        "dE00P99": round(float(np.percentile(dm, 99)), 3),
        "dE00Max": round(float(dm.max()), 3),
        "dE00Gt1": int((dm > 1).sum()),
        "dE00Gt2": int((dm > 2).sum()),
        "dE00Gt5": int((dm > 5).sum()),
        "_diff": np.maximum(np.abs(over_white(rest) - over_white(comp)).max(-1), da),
    }


def dark_dots(rgba: np.ndarray) -> int:
    lab = srgb_to_lab(np.round(over_white(rgba)).astype(np.uint8))[..., 0]
    a = rgba[..., 3]
    h, w = lab.shape
    c = lab[1:-1, 1:-1]
    darker = np.ones_like(c, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            darker &= lab[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx] - c > 20
    inside = a[1:-1, 1:-1] > 0
    return int((darker & inside).sum())


def pose_compare(before: Path, after: Path, out: Path, zoom: int) -> list[dict]:
    sb = json.loads((before / "stats.json").read_text(encoding="utf-8"))
    boxes = crops(sb["landmarks"])
    rows = []
    for r in sb["poseRenders"]:
        f = r["file"]
        B = Image.open(before / f).convert("RGBA")
        A = Image.open(after / f).convert("RGBA")
        name = Path(f).stem
        half = lambda im: im.resize((im.width // 2, im.height // 2), Image.LANCZOS)  # noqa: E731
        hstack([label(half(on_bg(B)), f"before (V1 CLI)  {r['pose']}"), label(half(on_bg(A)), f"after (B4)  {r['pose']}")]).save(
            out / f"{name}.png"
        )
        tb, ta = [], []
        for cname, (x0, y0, x1, y1) in boxes.items():
            for src, dst in ((B, tb), (A, ta)):
                c = on_bg(src.crop((x0, y0, x1, y1)))
                dst.append(label(c.resize((c.width * zoom, c.height * zoom), Image.NEAREST), cname))
        vstack([label(hstack(tb), "before (V1 CLI renderer, V1 mesh)"), label(hstack(ta), "after (B4 renderer, B4 mesh)")]).save(
            out / f"{name}_zoom.png"
        )
        nb, na = np.asarray(B), np.asarray(A)
        rows.append({
            "pose": r["pose"],
            "darkDotsBefore": dark_dots(nb),
            "darkDotsAfter": dark_dots(na),
            "opaquePxBefore": int((nb[..., 3] > 0).sum()),
            "opaquePxAfter": int((na[..., 3] > 0).sum()),
        })
    return rows


def eye_leak(root: Path, out: Path) -> dict:
    res: dict = {}
    summ = json.loads((root / "new-app" / "summary.json").read_text(encoding="utf-8"))
    lm = summ["landmarks"]
    mid = (lm["eyeL"]["x"] + lm["eyeR"]["x"]) / 2
    d = lm["eyeL"]["x"] - lm["eyeR"]["x"]
    worst: dict = {}
    for mesh in ("old", "new"):
        for op in ("app", "one"):
            key = f"{mesh}-{op}"
            per = []
            total = 0
            for p in summ["poses"]:
                closed = [s for s, v in (("L", p["eyeL"]), ("R", p["eyeR"])) if v <= 0.0]
                if not closed:
                    continue
                full = np.asarray(Image.open(root / key / p["file"]).convert("RGBA"))
                drop = np.asarray(Image.open(root / f"{key}-drop" / p["file"]).convert("RGBA"))
                diff = np.abs(over_white(full) - over_white(drop)).max(-1) > 2
                xs = np.arange(full.shape[1])[None, :]
                side = np.zeros_like(diff)
                for s in closed:
                    side |= (xs > mid) if s == "L" else (xs <= mid)
                leak = diff & side
                n = int(leak.sum())
                total += n
                per.append({"pose": p["pose"], "closed": closed, "leakPx": n})
                if n > worst.get(p["pose"], (-1,))[0]:
                    worst[p["pose"]] = (n, p["file"])
            res[key] = {"leakPxTotal": total, "posesWithLeak": sum(1 for r in per if r["leakPx"] > 0), "closedPoses": len(per), "perPose": per}
    # Crops of the poses with the largest geometric leak: four variants side by side, leak in magenta.
    picks = sorted(worst.items(), key=lambda kv: -kv[1][0])[:3]
    for pose, (_, f) in picks:
        tiles = []
        for key in ("old-app", "new-app", "old-one", "new-one"):
            full = np.asarray(Image.open(root / key / f).convert("RGBA"))
            drop = np.asarray(Image.open(root / f"{key}-drop" / f).convert("RGBA"))
            diff = np.abs(over_white(full) - over_white(drop)).max(-1) > 2
            rgb = over_white(full)
            rgb[diff] = (255, 0, 255)
            img = Image.fromarray(np.round(rgb).astype(np.uint8))
            for eye in ("eyeR", "eyeL"):
                e = lm[eye]
                box = (int(e["x"] - d * 0.45), int(e["y"] - d * 0.32), int(e["x"] + d * 0.45), int(e["y"] + d * 0.32))
                c = img.crop(box)
                tiles.append(label(c.resize((c.width * 4, c.height * 4), Image.NEAREST), f"{key} {eye}"))
        rows = [hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
        vstack([label(hstack(rows[:2]), f"{pose}: mesh old | new, app opacity (top) / opacity 1 (bottom); magenta = eye interior visible")] + [hstack(rows[2:])]).save(
            out / f"eyeleak_{pose.replace(':', '_').replace('/', '_').replace('+', 'p').replace('-', 'm')}.png"
        )
    return res


def mesh_coverage(run: Path) -> int:
    o = subprocess.run([PY, "-I", str(Path(__file__).resolve().parent / "v1_mesh_coverage.py"), str(run)], capture_output=True, text=True, check=True).stdout
    return int(o.strip().splitlines()[-1].split()[-1])


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strip_ms(v):
    if isinstance(v, dict):
        return {k: strip_ms(x) for k, x in v.items() if not k.endswith("Ms")}
    if isinstance(v, list):
        return [strip_ms(x) for x in v]
    return v


def default_diff(before: Path, after: Path) -> dict:
    fb = {p.relative_to(before).as_posix() for p in before.rglob("*") if p.is_file()}
    fa = {p.relative_to(after).as_posix() for p in after.rglob("*") if p.is_file()}
    same, diff = [], []
    for f in sorted(fb & fa):
        (same if sha(before / f) == sha(after / f) else diff).append(f)
    sb = strip_ms(json.loads((before / "stats.json").read_text(encoding="utf-8")))
    sa = strip_ms(json.loads((after / "stats.json").read_text(encoding="utf-8")))
    keys = sorted(k for k in set(sb) | set(sa) if sb.get(k) != sa.get(k))
    rigb = json.loads((before / "rig.json").read_text(encoding="utf-8"))
    riga = json.loads((after / "rig.json").read_text(encoding="utf-8"))
    tris = {e["id"]: [len(e["mesh"]["triangles"]) // 3, None] for e in rigb}
    for e in riga:
        tris.setdefault(e["id"], [None, None])[1] = len(e["mesh"]["triangles"]) // 3
    return {
        "onlyBefore": sorted(fb - fa),
        "onlyAfter": sorted(fa - fb),
        "identical": same,
        "different": diff,
        "statsKeysDifferent": keys,
        "trianglesBeforeAfter": tris,
        "meshQualityHoleCrossings": [
            sum(m.get("holeCrossings", 0) for m in sb.get("meshQuality", [])),
            sum(m.get("holeCrossings", 0) for m in sa.get("meshQuality", [])),
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--eyeleak", type=Path)
    ap.add_argument("--default-before", type=Path)
    ap.add_argument("--default-after", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--zoom", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    summary["poses"] = pose_compare(a.before, a.after, a.out, a.zoom)
    rb, ra = rest_vs_composite(a.before), rest_vs_composite(a.after)
    tiles = []
    for tag, r in (("before", rb), ("after", ra)):
        img = Image.fromarray(np.clip(r.pop("_diff") * 8, 0, 255).astype(np.uint8))
        tiles.append(label(img.resize((img.width // 2, img.height // 2), Image.LANCZOS), f"{tag}: |rest - composite| x8"))
    hstack(tiles).save(a.out / "rest_vs_composite.png")
    summary["restVsComposite"] = {"before": rb, "after": ra}
    summary["meshUncoveredTexels"] = {"before": mesh_coverage(a.before), "after": mesh_coverage(a.after)}
    if a.eyeleak:
        summary["eyeLeak"] = eye_leak(a.eyeleak, a.out)
    if a.default_before and a.default_after:
        summary["defaultRun"] = default_diff(a.default_before, a.default_after)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    brief = {k: v for k, v in summary.items() if k not in ("poses",)}
    if "eyeLeak" in brief:
        brief["eyeLeak"] = {k: {kk: vv for kk, vv in v.items() if kk != "perPose"} for k, v in brief["eyeLeak"].items()}
    print(json.dumps(brief, indent=1, ensure_ascii=False)[:6000])
    print("darkDots total before/after:", sum(r["darkDotsBefore"] for r in summary["poses"]), sum(r["darkDotsAfter"] for r in summary["poses"]))


if __name__ == "__main__":
    main()
