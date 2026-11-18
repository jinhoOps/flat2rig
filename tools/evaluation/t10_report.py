"""T10 report (evaluation only): eye seal, fillMask areas, rest composite ΔE, T13 seams/coverage and
T14 eye-part metrics between CLI output directories.

Usage:
  python -I tools/evaluation/t10_report.py NAME HEAD_DIR LEGACY_DIR AFTER_DIR

- HEAD_DIR:   pre-T10 binary output (fillMask / T13 / T14 / rest composite "before").
- LEGACY_DIR: T10 binary with `--legacy-eye` (pre-T10 eye formulas + clipping rule) — the eye seal
              "before" (the seal checker does not exist in the pre-T10 binary).
- AFTER_DIR:  T10 binary output.
Prints Markdown tables.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict

import numpy as np
from PIL import Image

name, head_dir, legacy_dir, after_dir = sys.argv[1:5]


def stats(d):
    return json.load(open(d + "/stats.json", encoding="utf-8"))


def srgb_to_lab(rgb):
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def de76_over_alpha(a_path, b_path):
    a = np.asarray(Image.open(a_path).convert("RGBA")).astype(float)
    b = np.asarray(Image.open(b_path).convert("RGBA")).astype(float)
    alpha_diff = int((np.abs(a[..., 3] - b[..., 3]) > 0).sum())
    m = (a[..., 3] > 0) | (b[..., 3] > 0)
    pa = a[..., :3] * (a[..., 3:] / 255.0) + 255 * (1 - a[..., 3:] / 255.0)
    pb = b[..., :3] * (b[..., 3:] / 255.0) + 255 * (1 - b[..., 3:] / 255.0)
    d = np.linalg.norm(srgb_to_lab(pa) - srgb_to_lab(pb), axis=-1)[m]
    return float(d.mean()) if d.size else 0.0, float(d.max()) if d.size else 0.0, int((d > 0.5).sum()), alpha_diff


H, L, A = stats(head_dir), stats(legacy_dir), stats(after_dir)
print(f"\n## {name}\n")

# Eye seal
def expr_of(pose):
    return pose.split("/")[0][5:] if pose.startswith("seal:") else pose


def seal_agg(st):
    agg = defaultdict(lambda: {"leak": 0, "leakEx": 0, "gapEx": 0, "brkEx": 0, "iris": 0.0, "lash": 0.0, "rows": 0, "defect": 0})
    for r in st["eyeSeal"]["rows"]:
        a = agg[expr_of(r["pose"])]
        a["leak"] += r["leakPx"]
        a["leakEx"] += r["leakExcessPx"]
        a["gapEx"] += r["gapExcessPx"]
        a["brkEx"] += r["lidBreakExcess"]
        a["iris"] = max(a["iris"], r["irisShiftPx"])
        a["lash"] = max(a["lash"], r["lashDropPx"])
        a["rows"] += 1
        a["defect"] += int(r["leakExcessPx"] + r["gapExcessPx"] + r["lidBreakExcess"] > 0)
    return agg


la, aa = seal_agg(L), seal_agg(A)
print("### 눈 밀폐 (표정 × 시선, 행 = 포즈 × 눈) — legacy → T10\n")
print("| 포즈 묶음 | 행 | 누출 px (휴지 초과) | 이음새 틈 px (휴지 초과) | 윗눈꺼풀 선 끊김 (휴지 초과) | 결함 행 | 홍채 중심 최대 이동 px | 윗눈꺼풀 최대 하강 px |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for k in la:
    b, a = la[k], aa[k]
    print(f"| {k} | {a['rows']} | {b['leakEx']} → {a['leakEx']} | {b['gapEx']} → {a['gapEx']} | {b['brkEx']} → {a['brkEx']} | {b['defect']} → {a['defect']} | {b['iris']:.1f} → {a['iris']:.1f} | {b['lash']:.1f} → {a['lash']:.1f} |")
for st, tag in ((L, "legacy"), (A, "T10")):
    es = st["eyeSeal"]
    print(f"\n- {tag}: 누출 합 {es['leakExcessPxTotal']} (원시 {es['leakPxTotal']}), 틈 초과 {es['gapExcessPxTotal']}, 끊김 초과 {es['lidBreakExcessTotal']}, 결함 행 {es['rowsWithDefect']}/{len(es['rows'])}; 휴지 기준 " + ", ".join(f"{r['side']}: 누출 {r['leakPx']} 틈 {r['gapPx']} 끊김 {r['lidBreaks']}" for r in es["rest"]))
print("\n잔여 (T10, 휴지 초과 > 0 인 첫 행):\n")
seen = set()
for r in A["eyeSeal"]["rows"]:
    key = (expr_of(r["pose"]), r["side"])
    if (r["leakExcessPx"] or r["gapExcessPx"] or r["lidBreakExcess"]) and key not in seen:
        seen.add(key)
        at = r.get("leakAt") or r.get("breakAt") or {}
        print(f"- `{r['pose']}` {r['side']}: 누출 {r['leakExcessPx']} ({r.get('leakLayer')}), 틈 {r['gapExcessPx']}, 끊김 {r['lidBreakExcess']} @ ({at.get('x', 0):.0f}, {at.get('y', 0):.0f})")

# fillMask areas
print("\n### 파츠별 fillMask 면적 (HEAD → T10)\n")
hc = {c["id"]: c for c in H["partCoverage"]}
ac = {c["id"]: c for c in A["partCoverage"]}
rows = []
dec = []
for pid, c in ac.items():
    b = hc.get(pid, {}).get("fillPixels")
    rows.append(f"{pid} {b} → {c['fillPixels']}")
    if b is not None and c["fillPixels"] < b:
        dec.append(pid)
    if hc.get(pid, {}).get("visiblePixels") != c["visiblePixels"]:
        dec.append(pid + "(visible changed)")
print(" · ".join(rows))
print(f"\n감소/가시 변경 파츠: {dec or '없음'}")

# Rest composite
print("\n### 정지 기본 포즈 합성 (HEAD → T10)\n")
for f in ("foreground.png", "effective_foreground.png"):
    mean, mx, n, ad = de76_over_alpha(f"{head_dir}/{f}", f"{after_dir}/{f}")
    print(f"- `{f}`: ΔE76 평균 {mean:.4f}, 최대 {mx:.2f}, ΔE>0.5 픽셀 {n}, 알파 다른 픽셀 {ad}")
for f in ("eye_seal/rest.png",):
    try:
        mean, mx, n, ad = de76_over_alpha(f"{legacy_dir}/{f}", f"{after_dir}/{f}")
        print(f"- 메쉬 래스터 휴지 포즈 `{f}` (legacy → T10): ΔE76 평균 {mean:.4f}, 최대 {mx:.2f}, ΔE>0.5 픽셀 {n}, 알파 다른 픽셀 {ad}")
    except FileNotFoundError:
        pass

# T13
print("\n### T13 이음새 커버리지 · 눈 관련 이음새 요구량 (HEAD → T10)\n")
hs, as_ = H["seamCoverage"], A["seamCoverage"]
print(f"- 포즈 {hs['poses']} → {as_['poses']}, 검사 {hs['checks']} → {as_['checks']}, `holes` {hs['holes']} → {as_['holes']}, `lowerMisses` {hs['lowerMisses']} → {as_['lowerMisses']}")
hseam = {(s["upper"], s["lower"]): s for s in H["seams"]}
print("\n| 상위 / 하위 | 요구 언더랩 px | 최대 상대 변위 px | lowerMisses | holes |\n| --- | --- | --- | --- | --- |")
for s in A["seams"]:
    if not (s["upper"].startswith("eye") or s["lower"].startswith("eye")):
        continue
    b = hseam.get((s["upper"], s["lower"]))
    if not b:
        print(f"| {s['upper']} / {s['lower']} | new {s['requiredUnderlapPx']} | {s['maxRelativeDisplacementPx']:.1f} | {s['lowerMisses']} | {s['holes']} |")
        continue
    print(f"| {s['upper']} / {s['lower']} | {b['requiredUnderlapPx']} → {s['requiredUnderlapPx']} | {b['maxRelativeDisplacementPx']:.1f} → {s['maxRelativeDisplacementPx']:.1f} | {b['lowerMisses']} → {s['lowerMisses']} | {b['holes']} → {s['holes']} |")
he = {l["id"]: l for l in H["exposure"]["layers"]}
print("\n노출 (HEAD → T10, 눈 파츠·face): " + " · ".join(
    f"{l['id']} 숨은 {he.get(l['id'], {}).get('hiddenPx')}→{l['hiddenPx']} 노출 {he.get(l['id'], {}).get('exposedPx')}→{l['exposedPx']}"
    for l in A["exposure"]["layers"] if l["id"].startswith("eye") or l["id"] == "face"))
print(f"\n노출 구멍 합 {H['exposure']['holes']['holePxTotal']} → {A['exposure']['holes']['holePxTotal']}")

# T14
print("\n### T14 눈 파츠 지표 (HEAD → T10)\n")
for kind in ("groundTruth", "holdout"):
    hf_ = H.get("fidelity", {}).get(kind)
    af_ = A.get("fidelity", {}).get(kind)
    if not hf_ or not af_:
        continue
    hp = {p["id"]: p for p in hf_["parts"]}
    print(f"\n{kind}:\n\n| 파츠 | 노출 px | ΔE00 평균 / p95 | 팔레트 일치 | 끊김 |\n| --- | --- | --- | --- | --- |")
    for p in af_["parts"]:
        if not (p["id"].startswith("eye")):
            continue
        b = hp.get(p["id"])
        m, bm = p.get("metrics") or {}, (b or {}).get("metrics") or {}

        def f(x):
            return "–" if x is None else f"{x:.2f}"

        print(f"| {p['id']} | {(b or {}).get('exposedPx')} → {p.get('exposedPx')} | {f(bm.get('deltaEMean'))} / {f(bm.get('deltaEP95'))} → {f(m.get('deltaEMean'))} / {f(m.get('deltaEP95'))} | {f(bm.get('paletteMatch'))} → {f(m.get('paletteMatch'))} | {bm.get('strokeBreaks')}/{bm.get('strokesCrossing')} → {m.get('strokeBreaks')}/{m.get('strokesCrossing')} |")
