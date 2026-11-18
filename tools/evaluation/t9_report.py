"""T9 report helper (evaluation/report only): compares two flat2rig-cli output folders (before /
after a synthesizer change) for docs/validation.md.

  tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/t9_report.py \
      INPUT.png BEFORE_DIR AFTER_DIR [--crops OUT_DIR x,y,w,h:name ...]

Prints Markdown tables:
  1. partition invariance: per-part visible mask (foreground alpha + top-most owner) and hidden
     area (effective alpha of parts/<id>.png minus visible) before vs after — hidden area must not
     shrink, visible mask must be identical;
  2. rest pose vs the input, CIEDE2000 mean / p95 / share > 5 / > 10, for the visible-only
     composite (foreground.png) and for the rest render (effective_foreground.png: every part's
     `mask ∪ fillMask` drawn bottom to top, what the app shows at rest), over the whole foreground
     (alpha >= 128), the seam/silhouette rim (<= 2 px, 4-connected, from a pixel with another owner
     or background), the seam outline pixels (rim pixels next to another part whose original L* is
     at least 25 below the owning part's median L*), and the interior;
  3. region type per part (stats.json partCoverage[].fillRegion, after run);
  4. T14 fidelity before -> after per part (ground truth and holdout), with the region type.
Optional crops write side-by-side PNGs (input | before | after) of foreground.png,
motion_turn.png and motion_blink.png. `--auto-crops OUT_DIR` derives the boxes from the run's
landmarks (chin, neck top, collar, both eyes / eyelids, mouth) and the `clothes_hands` part.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def de2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cb7 = ((c1 + c2) / 2) ** 7
    g = 0.5 * (1 - np.sqrt(cb7 / (cb7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.where((b1 == 0) & (a1p == 0), 0, np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = np.where((b2 == 0) & (a2p == 0), 0, np.degrees(np.arctan2(b2, a2p)) % 360)
    dlp, dcp = l2 - l1, c2p - c1p
    cprod = c1p * c2p
    d = h2p - h1p
    dhp = np.where(cprod == 0, 0, np.where(d > 180, d - 360, np.where(d < -180, d + 360, d)))
    dhb = 2 * np.sqrt(cprod) * np.sin(np.radians(dhp) / 2)
    lbp, cbp = (l1 + l2) / 2, (c1p + c2p) / 2
    hs = h1p + h2p
    hbp = np.where(cprod == 0, hs, np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    t = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    rc = 2 * np.sqrt(cbp**7 / (cbp**7 + 25.0**7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / np.sqrt(20 + (lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    rt = -np.sin(np.radians(2 * dth)) * rc
    return np.sqrt(np.maximum((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhb / sh) ** 2 + rt * (dcp / sc) * (dhb / sh), 0))


def load(d: Path):
    stats = json.loads((d / "stats.json").read_text(encoding="utf-8"))
    ids = [p["id"] for p in stats["partCoverage"]]
    fg = np.array(Image.open(d / "foreground.png").convert("RGBA"))
    eff = {i: np.array(Image.open(d / "parts" / f"{i}.png").convert("RGBA"))[..., 3] for i in ids}
    if (d / "owner.png").exists():
        # T9+ CLI: exact rest-pose visible owner (overlay fill texels sit above lower layers).
        owner = np.array(Image.open(d / "owner.png")).astype(np.int32) - 1
    else:
        # Pre-T9 runs: every fill texel lies under a higher visible layer, so the top-most
        # effective alpha is the visible owner.
        owner = np.full(fg.shape[:2], -1, np.int32)
        for k, i in enumerate(ids):
            owner[eff[i] > 0] = k
    eff_comp = np.array(Image.open(d / "effective_foreground.png").convert("RGBA"))
    return stats, ids, fg, eff, owner, eff_comp


def rim_mask(owner: np.ndarray, r: int = 2) -> np.ndarray:
    h, w = owner.shape
    diff = np.zeros_like(owner, bool)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) == 0 or abs(dx) + abs(dy) > r:
                continue
            sh = np.full_like(owner, -2)
            ys, yd = (slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0)))
            xs, xd = (slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0)))
            sh[yd, xd] = owner[ys, xs]
            diff |= sh != owner
    return diff & (owner >= 0)


def f2(v, nd=2):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    crops_dir = None
    crops = []
    auto = False
    if "--auto-crops" in args:
        k = args.index("--auto-crops")
        crops_dir = Path(args[k + 1])
        auto = True
        args = args[:k]
    if "--crops" in args:
        k = args.index("--crops")
        crops_dir = Path(args[k + 1])
        crops = args[k + 2:]
        args = args[:k]
    inp, before, after = Path(args[0]), Path(args[1]), Path(args[2])
    src = np.array(Image.open(inp).convert("RGBA"))
    sb, ids_b, fg_b, eff_b, own_b, ec_b = load(before)
    sa, ids_a, fg_a, eff_a, own_a, ec_a = load(after)
    assert ids_a == ids_b, "part lists differ"
    print(f"\n### {after.name}\n")
    print("| part | visible px (identical?) | fillMask px before → after (of which overlay) | region (after) | snap | synth | seeds / band | line / thin / σL* / cel core | palette | rim: halo / upper line / upper mix / lower mix / kept / residue |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    vis_same_all = np.array_equal(fg_a[..., 3], fg_b[..., 3]) and np.array_equal(own_a, own_b)
    cov_a = {p["id"]: p for p in sa["partCoverage"]}
    cov_b = {p["id"]: p for p in sb["partCoverage"]}
    for k, i in enumerate(ids_a):
        vis_b = (own_b == k) & (fg_b[..., 3] > 0)
        vis_a = (own_a == k) & (fg_a[..., 3] > 0)
        same = np.array_equal(vis_a, vis_b) and np.array_equal(fg_a[..., 3][vis_a], fg_b[..., 3][vis_b])
        hid_b, hid_a = cov_b[i]["fillPixels"], cov_a[i]["fillPixels"]
        r = cov_a[i].get("fillRegion") or {}
        rim = r.get("rim", {})
        ov = rim.get("overlayPx", 0)
        # Effective alpha may only grow (overlay texels added); never lose a pre-T9 texel.
        lost = int(((eff_b[i] > 0) & (eff_a[i] == 0)).sum())
        print(
            f"| {i} | {int(vis_a.sum()):,} ({'yes' if same else '**NO**'}) | {hid_b:,} → {hid_a:,} ({ov:,})"
            f"{' **effective lost ' + str(lost) + '**' if lost else ''}"
            f"{' **shrunk**' if hid_a - ov < hid_b else ''} | {r.get('kind', '–')} | {r.get('snap', '–')} | {r.get('synth', '–')} "
            f"| {r.get('seeds', 0):,} / {r.get('bandSeeds', 0):,} | {f2(r.get('lineFrac'))} / {f2(r.get('seedFrac'))} / {f2(r.get('localStdMedian'))} / {f2(r.get('celCoreFrac'))} "
            f"| {' '.join(r.get('palette', []))} | {rim.get('backgroundHalo', 0)} / {rim.get('upperLine', 0)} / {rim.get('upperMix', 0)} / {rim.get('lowerMix', 0)} / {rim.get('keptUnownedLine', 0)} / {rim.get('featureResidue', 0)} |"
        )
    print(f"\nvisible partition identical (foreground alpha + owner map): **{vis_same_all}**\n")
    # Rest composite vs input.
    lab_src = srgb_to_lab(src[..., :3])
    fgm = (src[..., 3] >= 128) & (fg_b[..., 3] >= 128)
    rim = rim_mask(own_a) & fgm
    # Seam outline pixels: rim pixels next to another *part* (not background), line-dark against
    # the owning part's median L*.
    seam_rim = np.zeros_like(rim)
    h_, w_ = own_a.shape
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dx) + abs(dy) == 0 or abs(dx) + abs(dy) > 2:
                continue
            sh = np.full_like(own_a, -1)
            ys, yd = (slice(max(dy, 0), h_ + min(dy, 0)), slice(max(-dy, 0), h_ + min(-dy, 0)))
            xs, xd = (slice(max(dx, 0), w_ + min(dx, 0)), slice(max(-dx, 0), w_ + min(-dx, 0)))
            sh[yd, xd] = own_a[ys, xs]
            seam_rim |= (sh >= 0) & (sh != own_a)
    seam_rim &= rim
    med = np.zeros(len(ids_a))
    for k in range(len(ids_a)):
        v = lab_src[..., 0][own_a == k]
        med[k] = np.median(v) if v.size else 0
    own_med = np.where(own_a >= 0, med[np.clip(own_a, 0, None)], 0)
    seam_line = seam_rim & (lab_src[..., 0] < own_med - 25)
    print("| rest image vs input | region | px | ΔE00 mean | p95 | share > 5 | share > 10 |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name, img in (
        ("before visible-only", fg_b),
        ("after visible-only", fg_a),
        ("before rest render", ec_b),
        ("after rest render", ec_a),
    ):
        de = de2000(srgb_to_lab(img[..., :3]), lab_src)
        for rn, m in (
            ("foreground", fgm),
            ("rim ≤2px", rim),
            ("seam outline", seam_line & fgm),
            ("interior", fgm & ~rim),
        ):
            v = de[m]
            print(f"| {name} | {rn} | {v.size:,} | {v.mean():.3f} | {np.percentile(v, 95):.3f} | {(v > 5).mean():.4f} | {(v > 10).mean():.4f} |")
    # Fidelity.
    def fid(s):
        out = {}
        f = s.get("fidelity") or {}
        for key in ("groundTruth", "holdout"):
            for p in (f.get(key) or {}).get("parts", []):
                out[(key, p["id"])] = p
        return out

    fb, fa = fid(sb), fid(sa)
    print("\n| set | part | region | samples | ΔE00 mean / p95 before → after | weighted mean before → after | palette match before → after | grad. discont. before → after | σL* synth/truth before → after | stroke breaks before → after |")
    print("| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |")
    for key in sorted(fa):
        pa, pb = fa[key], fb.get(key)
        ma, mb = pa.get("metrics"), (pb or {}).get("metrics")
        if not ma or not mb:
            continue
        r = (cov_a.get(key[1], {}).get("fillRegion") or {})
        tag = r.get("kind", "–") + ("+snap" if r.get("snap") else "")
        whole = " (whole layer)" if pa.get("wholeLayerHeldOut") else ""
        print(
            f"| {'truth' if key[0] == 'groundTruth' else 'holdout'} | {key[1]}{whole} | {tag} | {ma['samples']:,} "
            f"| {f2(mb['deltaEMean'])} / {f2(mb['deltaEP95'])} → {f2(ma['deltaEMean'])} / {f2(ma['deltaEP95'])} "
            f"| {f2(mb['deltaEMeanFreqWeighted'])} → {f2(ma['deltaEMeanFreqWeighted'])} "
            f"| {f2(mb['paletteMatch'])} → {f2(ma['paletteMatch'])} "
            f"| {f2(mb['gradientDiscontinuityDe'])} → {f2(ma['gradientDiscontinuityDe'])} "
            f"| {f2(mb['localStdSynth'])}/{f2(mb['localStdTruth'])} → {f2(ma['localStdSynth'])}/{f2(ma['localStdTruth'])} "
            f"| {mb['strokeBreaks']}/{mb['strokesCrossing']} → {ma['strokeBreaks']}/{ma['strokesCrossing']} |"
        )
    if auto:
        lm = sa["landmarks"]
        ex, ey = lm["eyeL"]["x"] - lm["eyeR"]["x"], lm["eyeL"]["y"] - lm["eyeR"]["y"]
        ed = max(16.0, (ex * ex + ey * ey) ** 0.5)
        hh = max(24.0, ((lm["chin"]["x"] - lm["crown"]["x"]) ** 2 + (lm["chin"]["y"] - lm["crown"]["y"]) ** 2) ** 0.5)
        H, W = src.shape[:2]

        def box(cx, cy, bw, bh, name):
            bw, bh = int(min(bw, W)), int(min(bh, H))
            x = int(min(max(cx - bw / 2, 0), W - bw))
            y = int(min(max(cy - bh / 2, 0), H - bh))
            crops.append(f"{x},{y},{bw},{bh}:{name}")

        cx, cy = lm["chin"]["x"], lm["chin"]["y"]
        box(cx, cy, ed * 1.2, ed * 0.6, "chin")
        box(cx, cy + 0.12 * hh, ed * 1.0, ed * 0.5, "neck_top")
        box(cx, cy + 0.32 * hh, ed * 1.8, ed * 0.8, "collar")
        box(lm["eyeR"]["x"], lm["eyeR"]["y"], ed * 0.8, ed * 0.5, "eyeR_lids")
        box(lm["eyeL"]["x"], lm["eyeL"]["y"], ed * 0.8, ed * 0.5, "eyeL_lids")
        box(lm["mouth"]["x"], lm["mouth"]["y"], ed * 0.6, ed * 0.3, "mouth")
        if "clothes_hands" in ids_a:
            ys, xs = np.nonzero(eff_a["clothes_hands"] > 0)
            if len(xs):
                box(xs.mean(), ys.mean(), max(48, int(np.ptp(xs)) + 24), max(48, int(np.ptp(ys)) + 24), "hands")
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
        for spec in crops:
            box, name = spec.split(":")
            x, y, w, h = map(int, box.split(","))
            for img_name in ("effective_foreground.png", "motion_turn.png", "motion_blink.png"):
                tiles = [src] + [np.array(Image.open(d / img_name).convert("RGBA")) for d in (before, after)]
                row = []
                for t in tiles:
                    c = t[y:y + h, x:x + w].astype(np.float64)
                    a = c[..., 3:4] / 255.0
                    bg = np.full_like(c[..., :3], 128.0)
                    row.append((c[..., :3] * a + bg * (1 - a)).astype(np.uint8))
                im = Image.fromarray(np.concatenate(row, 1)).resize((w * 3 * 3, h * 3), Image.NEAREST)
                im.save(crops_dir / f"{after.name}_{name}_{img_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
