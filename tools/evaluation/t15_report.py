"""T15 before/after report (evaluation / report helper only; Python never runs in the app).

Compares two flat2rig-cli output folders of the same artwork (BEFORE = HEAD or `--no-lineart`,
AFTER = T15) and prints Markdown tables:

- T14 harness per part and per seam group (exposure band): stroke breaks / crossing strokes, gradient
  discontinuity, ΔE00 mean / p95, exposed px (samples) — PSD ground truth and self-supervised holdout;
- T15 extension statistics per part (`stats.json` `lineart.parts[].report`);
- fillMask area per part (must not shrink), visible partition identity (`owner.png`, `foreground.png`),
  rest composites changed px (`foreground.png` visible-only, `effective_foreground.png` mask ∪ fillMask);
- T10 eye seal totals, exposure totals, run time.

  tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/t15_report.py NAME BEFORE_DIR AFTER_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load(d: Path) -> dict:
    return json.loads((d / "stats.json").read_text(encoding="utf-8"))


def f2(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}"


def img(d: Path, name: str) -> np.ndarray:
    return np.asarray(Image.open(d / name))


def diff_px(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        return -1
    d = a != b
    return int(d.reshape(d.shape[0], d.shape[1], -1).any(axis=2).sum()) if d.ndim == 3 else int(d.sum())


def metrics_rows(sa: dict, sb: dict, key: str) -> list[str]:
    pa = {p["id"]: p for p in sa.get("fidelity", {}).get(key, {}).get("parts", [])}
    pb = {p["id"]: p for p in sb.get("fidelity", {}).get(key, {}).get("parts", [])}
    rows = []
    for pid in pa:
        a, b = pa[pid], pb.get(pid)
        if b is None:
            continue
        ma, mb = a.get("metrics") or {}, b.get("metrics") or {}
        if not ma:
            continue

        def cell(m, k, nd=2):
            return f2(m.get(k), nd)

        rows.append(
            f"| {pid} | all | {a['exposedPx']} ({ma.get('samples', 0)}) | "
            f"{ma.get('strokeBreaks', 0)}/{ma.get('strokesCrossing', 0)} → {mb.get('strokeBreaks', 0)}/{mb.get('strokesCrossing', 0)} | "
            f"{cell(ma, 'gradientDiscontinuityDe')} → {cell(mb, 'gradientDiscontinuityDe')} | "
            f"{cell(ma, 'deltaEMean')} / {cell(ma, 'deltaEP95')} → {cell(mb, 'deltaEMean')} / {cell(mb, 'deltaEP95')} |"
        )
        ga = {g["occluder"]: g for g in a.get("groups", [])}
        gb = {g["occluder"]: g for g in b.get("groups", [])}
        for occ, g in ga.items():
            h = gb.get(occ)
            m1, m2 = g.get("metrics") or {}, (h or {}).get("metrics") or {}
            if not m1 or m1.get("samples", 0) < 50:
                continue
            changed = any(
                m1.get(k) != m2.get(k)
                for k in ("strokeBreaks", "gradientDiscontinuityDe", "deltaEMean", "deltaEP95")
            )
            if not changed and m1.get("strokesCrossing", 0) == 0:
                continue
            rows.append(
                f"| {pid} | ← {occ} | {g['exposedPx']} ({m1.get('samples', 0)}) | "
                f"{m1.get('strokeBreaks', 0)}/{m1.get('strokesCrossing', 0)} → {m2.get('strokeBreaks', 0)}/{m2.get('strokesCrossing', 0)} | "
                f"{cell(m1, 'gradientDiscontinuityDe')} → {cell(m2, 'gradientDiscontinuityDe')} | "
                f"{cell(m1, 'deltaEMean')} / {cell(m1, 'deltaEP95')} → {cell(m2, 'deltaEMean')} / {cell(m2, 'deltaEP95')} |"
            )
    return rows


def totals(s: dict, key: str) -> tuple[int, int]:
    c = b = 0
    for p in s.get("fidelity", {}).get(key, {}).get("parts", []):
        m = p.get("metrics") or {}
        c += m.get("strokesCrossing", 0)
        b += m.get("strokeBreaks", 0)
    return b, c


def main() -> int:
    name, da, db = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    sa, sb = load(da), load(db)
    out = [f"## {name}", ""]
    for key, title in (("groundTruth", "PSD ground truth"), ("holdout", "self-supervised holdout")):
        if key not in sa.get("fidelity", {}):
            continue
        ba, ca = totals(sa, key)
        bb, cb = totals(sb, key)
        out += [
            f"### T14 {title}: stroke breaks {ba}/{ca} → {bb}/{cb}",
            "",
            "| part | seam (occluder) | exposed px (samples) | stroke breaks / crossing | gradient discontinuity ΔE00 | ΔE00 mean / p95 |",
            "| --- | --- | ---: | --- | --- | --- |",
            *metrics_rows(sa, sb, key),
            "",
        ]
    la = sb.get("lineart", {})
    out += [
        "### T15 extension statistics (after)",
        "",
        f"totals: {json.dumps(la.get('strokes'))}; skipped {la.get('strokesSkipped')}; contour length {la.get('contourLengthPx')} px; "
        f"stroke texels {la.get('strokeTexels')}, contour texels {la.get('contourTexels')}; rest gate {json.dumps(la.get('restGate'))}; "
        f"seam-parallel texels {la.get('seamParallelTexels')}; double lines {json.dumps({k: la.get('doubleLines', {}).get(k) for k in ('poses', 'exposedLinePx', 'crossLayerDoublePx', 'sameLayerParallelPx')})}",
        "",
        "| part | line px | w | touching | extended (+field / joined) | skipped short / fit / crossing | not candidates: seamOutline / parallel / seamParallel / contour / feature / unexposed / noRoom | ext. length px | contour px | outline cov. / w / colour | junction Δw / ΔE | gate acc / clamp / rej | merge: base / current-only |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for p in la.get("parts", []):
        r = p["report"]
        s = r["strokes"]
        g = r["restGate"]
        out.append(
            f"| {p['id']} | {r['visibleLinePx']} | {r['medianStrokeWidthPx']:.1f} | {s['touchingHidden']} | "
            f"{s['extended']} (+{s['extendedByField']} / {s['joined']}) | {s['skippedShort']} / {s['skippedFitResidual']} / {s['skippedCrossing']} | "
            f"{s['seamOutline']} / {s['parallel']} / {s['seamParallel']} / {s['contour']} / {s['featureSeam']} / {s['unexposed']} / {s['noRoom']} | "
            f"{r['extendedLengthPx']:.0f} | {r['contourLengthPx']} | {r['outlineCoverage']:.2f} / {r['outlineWidthPx']:.1f} / {r['outlineColor'] or '–'} | "
            f"{f2(r.get('junctionWidthDiffMean'))} / {f2(r.get('junctionDeMean'))} | {g['accepted']} / {g['clamped']} / {g['rejected']} | "
            f"{r.get('texelsInBaseHidden')} / {r.get('texelsInCurrentOnlyHidden')} |"
        )
    out += [
        "",
        "### Hidden silhouette (fillMask extends the part silhouette) → contour",
        "",
        "| part | outlined (coverage ≥ 0.5) | hidden ring px | under eye/brow/mouth | beyond exposure depth | in exposure band | of which shared with upper outline | contour reached | guard-skipped (under/beside upper outline) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in la.get("parts", []):
        r = p["report"]
        if not r.get("hiddenRingPx"):
            continue
        out.append(
            f"| {p['id']} | {r['outlined']} ({r['outlineCoverage']:.2f}) | {r['hiddenRingPx']} | {r['hiddenRingFeaturePx']} | "
            f"{r['hiddenRingBeyondDepthPx']} | {r['hiddenRingExposedPx']} | {r['hiddenRingSharedPx']} | {r['contourLengthPx']} | {r['contourGuardSkippedPx']} |"
        )
    # Semantic flags (skipped strokes)
    flags = [(p["id"], f) for p in la.get("parts", []) for f in p["report"].get("semantic", [])]
    if flags:
        out += ["", "skipped (semantic) strokes: " + "; ".join(
            f"{pid} ({f['x']:.0f},{f['y']:.0f}) {f['reason']} {f['value']:.2f} > / < {f['limit']:.2f}" for pid, f in flags[:24])]
    # Partition / fill / rest composites
    ca = {p["id"]: p for p in sa["partCoverage"]}
    cb = {p["id"]: p for p in sb["partCoverage"]}
    shrunk = [pid for pid in ca if cb.get(pid, {}).get("fillPixels", 0) < ca[pid]["fillPixels"]]
    fill_changed = [f"{pid} {ca[pid]['fillPixels']}→{cb[pid]['fillPixels']}" for pid in ca if pid in cb and cb[pid]["fillPixels"] != ca[pid]["fillPixels"]]
    vis_same = all(cb.get(pid, {}).get("visiblePixels") == ca[pid]["visiblePixels"] for pid in ca)
    owner_px = diff_px(img(da, "owner.png"), img(db, "owner.png"))
    fg_px = diff_px(img(da, "foreground.png"), img(db, "foreground.png"))
    eff_px = diff_px(img(da, "effective_foreground.png"), img(db, "effective_foreground.png"))
    fill_only_px = diff_px(img(da, "fill_only.png"), img(db, "fill_only.png"))
    out += [
        "",
        "### Invariants",
        "",
        "- fillMask px per part: " + ", ".join(pid + " " + str(ca[pid]["fillPixels"]) for pid in ca),
        f"- fillMask changed: {fill_changed or 'none'}; shrunk: {shrunk or 'none'}; visible px per part identical: {vis_same}; owner.png changed px: {owner_px}",
        f"- rest composite changed px: visible-only foreground.png {fg_px}, effective_foreground.png (mask ∪ fillMask) {eff_px}; fill_only.png (hidden texel colours) {fill_only_px}",
    ]
    ea, eb = sa["eyeSeal"], sb["eyeSeal"]
    out.append(
        f"- T10 eye seal (excess px): leak {ea['leakExcessPxTotal']} → {eb['leakExcessPxTotal']}, gap {ea['gapExcessPxTotal']} → {eb['gapExcessPxTotal']}, "
        f"lid breaks {ea['lidBreakExcessTotal']} → {eb['lidBreakExcessTotal']}, defect rows {ea['rowsWithDefect']} → {eb['rowsWithDefect']}"
    )
    xa, xb = sa["exposure"], sb["exposure"]
    out.append(
        f"- exposure: hidden {xa['hiddenPxTotal']} → {xb['hiddenPxTotal']}, exposed {xa['exposedPxTotal']} → {xb['exposedPxTotal']}, holes {xa['holes']['holePxTotal']} → {xb['holes']['holePxTotal']}; "
        f"seam coverage holes {sa['seamCoverage']['holes']} → {sb['seamCoverage']['holes']}"
    )
    out.append(f"- elapsedMs (stats, incl. evaluation) {sa['elapsedMs']} → {sb['elapsedMs']}")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
