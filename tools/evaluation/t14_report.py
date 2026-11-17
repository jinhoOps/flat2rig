"""T14 report helper (evaluation/report only): prints the `fidelity` block of flat2rig-cli
stats.json files as Markdown tables for docs/validation.md.

  tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/t14_report.py \
      artifacts/t14/himikan/stats.json [more stats.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def f(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def row(name, occl, m, extra):
    if m is None:
        return f"| {name} | {occl} | {extra} | 0 | – | – | – | – | – | – | – | – | – |"
    return (
        f"| {name} | {occl} | {extra} | {m['samples']:,} | {f(m['meanFrequency'])} "
        f"| {f(m['deltaEMean'])} / {f(m['deltaEP95'])} "
        f"| {f(m['deltaEMeanFreqWeighted'])} / {f(m['deltaEP95FreqWeighted'])} "
        f"| {f(m['paletteMatch'])} ({f(m['paletteMatchFreqWeighted'])}) "
        f"| {f(m['seamStepSynthDe'])} / {f(m['seamStepTruthDe'])} → {f(m['gradientDiscontinuityDe'])} "
        f"| {m['strokeBreaks']}/{m['strokesCrossing']} "
        f"| {f(m['localStdSynth'])} / {f(m['localStdTruth'])} → {f(m['textureStdDiff'])} "
        f"| {f(m['textureStdDiffFreqWeighted'])} |"
    )


HEADER = (
    "| part | occluder (seam upper) | area | samples | mean freq | ΔE00 mean / p95 | freq-weighted ΔE00 mean / p95 "
    "| palette match (weighted) | seam step synth / truth → \\|Δ\\| | stroke breaks / crossing "
    "| local σL* synth / truth → \\|Δ\\| | σ \\|Δ\\| weighted |\n"
    "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- | ---: |"
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        fid = s.get("fidelity")
        print(f"\n### {p}\n")
        if not fid:
            print("(no fidelity block)")
            continue
        gt = fid.get("groundTruth")
        if gt:
            print("Ground truth (PSD layers):\n")
            print(HEADER)
            for part in gt["parts"]:
                extra = (f"hidden {part['hiddenPx']:,} · exposed {part['exposedPx']:,} · noTruth {part['noTruthPx']:,}"
                         f" (bg {part['noTruthBackgroundPx']:,})")
                print(row(part["id"], "all", part["metrics"], extra))
                if part.get("noTruthVisualMetrics"):
                    print(row(part["id"], "noTruth→PSD below", part["noTruthVisualMetrics"], "texels outside the PSD part"))
                for g in part["groups"]:
                    if g["exposedPx"] >= 50:
                        print(row(part["id"], g["occluder"], g["metrics"], f"{g['exposedPx']:,}"))
            print(f"\nparts without truth: {', '.join(gt['partsWithoutTruth'])}")
        ho = fid.get("holdout")
        if ho:
            print(
                f"\nHoldout: masksIdentical={ho['masksIdentical']} fillMasksIdentical={ho['fillMasksIdentical']} "
                f"heldOutPx={ho['heldOutPx']:,}\n"
            )
            print(HEADER)
            for part in ho["parts"]:
                extra = f"held {part['exposedPx']:,} ({part['heldOutFraction']:.0%} of visible {part['visiblePx']:,})"
                if part.get("wholeLayerHeldOut"):
                    extra += " **whole layer**"
                print(row(part["id"], "all", part["metrics"], extra))
                for g in part["groups"]:
                    if g["exposedPx"] >= 50:
                        print(row(part["id"], g["occluder"], g["metrics"], f"{g['exposedPx']:,}"))
            skipped = [f"{x['upper']}/{x['lower']}" for x in ho["seams"] if x["skipped"]]
            print(f"\nskipped seams (depth < 1px): {', '.join(skipped) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
