"""V2: convert a See-through layered PSD into flat2rig part layers (evaluation only).

See-through (https://github.com/shitagaki-lab/see-through) writes one PSD per input image on a
square canvas (`--resolution`, default 1280): the input is centre-padded to a square of side
max(H, W) and resized (common/utils/cv.py `center_square_pad_resize`). Layers are named by their
body tag (bodytags_v3: `front hair`, `back hair`, `face`, `irides`, `eyewhite`, `eyelash`,
`eyebrow`, `mouth`, `neck`, `topwear`, ...) and stacked far-to-near (psd-tools order = bottom to top).

This script
1. undoes the square pad + resize so every layer lands on the source image grid (or a requested
   target grid with the same area mapping as `himikan_truth.py`: PIL BOX on premultiplied RGBA),
2. maps tags to flat2rig part ids (TAG_MAP below; anything without a flat2rig id is kept as an
   `extra_<tag>` layer and listed in the manifest, never dropped),
3. splits combined left/right layers (eyewhite, irides, eyelash, eyebrow) into character-relative
   L/R by connected components (AGENTS.md 1: character L = screen right = larger x) around a
   midline from the iris (else eye-white) component pair, else the face centroid; eyelash
   components below the eye centre become `eye*_lower`,
4. zeroes alpha <= 4/255 (See-through noise, e.g. a full-canvas mouth layer) and fixes two draw
   order defects for the rig order only (face features below the face, eye white above the iris);
   See-through's raw order stays in the manifest,
5. writes straight-alpha full-canvas RGBA PNGs plus `truth-manifest.json` in the same shape as
   `artifacts/t14/himikan-truth/` (width/height/parts/transform), with extra fields for draw order,
   source layers and the mapping table, and `layers.json` for `flat2rig-cli --from-layers`
   (extra layers renamed to role-bearing ids such as `hat`, `acc_glasses`, `hands`, explicit order).

Run with the See-through venv (it has numpy, pillow, psd-tools, scipy):
  artifacts/seethrough/venv/Scripts/python.exe -I tools/evaluation/v2_seethrough_to_layers.py \
      --psd artifacts/seethrough/out/himikan/himikan.psd --src artifacts/seethrough/inputs/himikan.png \
      --out artifacts/v2/layers/himikan [--target-size 683x1024] [--name himikan]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

warnings.filterwarnings("ignore")
from psd_tools import PSDImage  # noqa: E402

# See-through tag -> (flat2rig id or None, how). `how`:
#   "direct"  one layer -> one part
#   "lr"      split into character L/R by connected components -> f"{id}" with {S} = L/R
#   "lash"    eyelash: per side, components above the eye-white centre -> eye{S}_upper,
#             below -> eye{S}_lower
#   "merge"   folded into another part (composited in PSD order)
#   None      no flat2rig part; written as extra_<tag>
TAG_MAP: dict[str, tuple[str | None, str, str]] = {
    "back hair": ("hair_back", "direct", "뒷머리"),
    "front hair": ("hair_front", "direct", "앞머리 + 옆머리 (See-through는 옆머리를 따로 내지 않음)"),
    "face": ("face", "direct", "얼굴 피부"),
    "ears": ("face", "merge", "귀 -> face에 합침 (Himikan 정답 face도 귀 포함)"),
    "neck": ("neck", "direct", "목"),
    "topwear": ("clothes", "direct", "상의 (Himikan 정답 clothes = 몸통+카라+리본 묶음)"),
    "neckwear": ("clothes_inner", "direct", "넥웨어(리본·넥타이) -> clothes_inner (Himikan 정답 = 빨간 스카프)"),
    "eyewhite": ("eye{S}_white", "lr", "눈 흰자, 연결 요소로 L/R"),
    "irides": ("eye{S}_iris", "lr", "홍채, 연결 요소로 L/R"),
    "eyelash": ("eye{S}_upper", "lash", "속눈썹/눈꺼풀 선 -> upper, 흰자 중심보다 아래 요소 -> lower"),
    "eyebrow": ("brow{S}", "lr", "눈썹, 연결 요소로 L/R"),
    "mouth": ("mouth_all", "direct", "입 전체 1장 (윗입술/아랫입술/입안 분리 없음; RGB는 원화 복사)"),
    "nose": ("face_detail", "direct", "코 (RGB는 원화 복사)"),
    "headwear": (None, "extra", "모자·머리 장식: flat2rig 파츠 없음"),
    "eyewear": (None, "extra", "안경: flat2rig 파츠 없음"),
    "earwear": (None, "extra", "귀걸이: flat2rig 파츠 없음"),
    "handwear": (None, "extra", "손·장갑·팔: flat2rig 파츠 없음 (body/clothes_hands 후보)"),
    "bottomwear": (None, "extra", "하의: flat2rig 파츠 없음 (clothes 후보)"),
    "legwear": (None, "extra", "다리·스타킹: flat2rig 파츠 없음"),
    "footwear": (None, "extra", "신발: flat2rig 파츠 없음"),
    "tail": (None, "extra", "꼬리: flat2rig 파츠 없음"),
    "wings": (None, "extra", "날개: flat2rig 파츠 없음"),
    "objects": (None, "extra", "소품: flat2rig 파츠 없음"),
}
ALPHA_ON = 0.5 / 255.0
ALPHA_FLOOR = 4.5 / 255.0  # alpha <= 4/255 is zeroed (recorded as alphaFloor in the manifest)
FACE_FEATURES = {
    "browL", "browR", "eyeL_white", "eyeR_white", "eyeL_iris", "eyeR_iris", "eyeL_upper", "eyeR_upper",
    "eyeL_lower", "eyeR_lower", "mouth_all", "face_detail",
}
FACE_FEATURE_EXTRAS = {"extra_eyewear"}
# Rig input (`layers.json`, read by `flat2rig-cli --from-layers` before truth-manifest.json): extra
# layers get ids whose prefix gives them a flat2rig role (core::infer_part_role) instead of being
# dropped from the render. The PNG files keep their extra_<tag> names.
RIG_ID = {
    "extra_headwear": "hat",
    "extra_eyewear": "acc_glasses",
    "extra_earwear": "acc_earwear",
    "extra_handwear": "hands",
    "extra_bottomwear": "clothes_bottom",
    "extra_legwear": "clothes_legwear",
    "extra_footwear": "clothes_footwear",
    "extra_tail": "other_tail",
    "extra_wings": "other_wings",
    "extra_objects": "other_objects",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def layer_canvas(layer, w: int, h: int) -> np.ndarray:
    """Straight RGBA float32 (h, w, 4) in [0, 1] on the PSD canvas."""
    canvas = np.zeros((h, w, 4), np.float32)
    im = layer.topil()
    if im is None:
        return canvas
    arr = np.asarray(im.convert("RGBA"), np.float32) / 255.0
    l, t = layer.left, layer.top
    x0, y0 = max(l, 0), max(t, 0)
    x1, y1 = min(l + arr.shape[1], w), min(t + arr.shape[0], h)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = arr[y0 - t : y1 - t, x0 - l : x1 - l]
    return canvas


def premul(s: np.ndarray) -> np.ndarray:
    return np.concatenate([s[..., :3] * s[..., 3:4], s[..., 3:4]], axis=-1)


def over_p(dst: np.ndarray, src_p: np.ndarray) -> np.ndarray:
    return src_p + dst * (1.0 - src_p[..., 3:4])


def unpremultiply_u8(p: np.ndarray) -> np.ndarray:
    a = p[..., 3:4]
    rgb = np.divide(p[..., :3], a, out=np.zeros_like(p[..., :3]), where=a > 1e-6)
    return (np.clip(np.concatenate([rgb, a], axis=-1), 0, 1) * 255.0 + 0.5).astype(np.uint8)


def to_grid(p: np.ndarray, box: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray:
    """Resample premultiplied canvas region `box` (x0, y0, x1, y1 in canvas px edges) to `size`.

    Downscale uses BOX (area), upscale uses BILINEAR; both on premultiplied channels."""
    sx = size[0] / (box[2] - box[0])
    filt = Image.BOX if sx <= 1.0 else Image.BILINEAR
    chans = [
        np.asarray(Image.fromarray(np.ascontiguousarray(p[..., c]), "F").resize(size, filt, box=box))
        for c in range(4)
    ]
    return np.clip(np.stack(chans, axis=-1), 0.0, 1.0)


def components(alpha: np.ndarray):
    lab, n = ndimage.label(alpha > 0.1, structure=np.ones((3, 3), int))
    out = []
    for k in range(1, n + 1):
        m = lab == k
        ys, xs = np.nonzero(m)
        out.append({"mask": m, "cx": float(xs.mean()), "cy": float(ys.mean()), "n": int(m.sum())})
    return out


def split_lr(p: np.ndarray, split_x: float):
    """Split premultiplied layer into (L, R) character sides; L = screen right (x > split_x)."""
    a = p[..., 3]
    L = np.zeros(a.shape, bool)
    for c in components(a):
        if c["cx"] > split_x:
            L |= c["mask"]
    # Faint pixels below the component threshold follow their x.
    faint = (a > 0) & (a <= 0.1)
    xs = np.broadcast_to(np.arange(a.shape[1])[None, :], a.shape)
    L |= faint & (xs > split_x)
    return p * L[..., None], p * (~L)[..., None]


def centroid_x(alpha: np.ndarray) -> float | None:
    s = float(alpha.sum())
    if s <= 0:
        return None
    return float((np.arange(alpha.shape[1])[None, :] * alpha).sum() / s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd", required=True, help="See-through output PSD")
    ap.add_argument("--src", required=True, help="image given to See-through (for size/pad inverse)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-size", default=None, help="WxH output grid (default: source size)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    psd_path, src_path, out = Path(args.psd), Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    psd = PSDImage.open(psd_path)
    CW, CH = psd.size
    with Image.open(src_path) as im:
        W, H = im.size
    S = max(W, H)
    px1, py1 = (S - W) // 2, (S - H) // 2
    if CW != CH:
        print(f"warning: non-square See-through canvas {CW}x{CH}", file=sys.stderr)
    r = CW / S
    box = (px1 * r, py1 * r, (px1 + W) * r, (py1 + H) * r)
    if args.target_size:
        tw, th = (int(v) for v in args.target_size.lower().split("x"))
    else:
        tw, th = W, H

    # Load layers bottom -> top on the target grid (premultiplied).
    layers = []
    for idx, layer in enumerate(psd):
        if layer.is_group():
            continue
        p = to_grid(premul(layer_canvas(layer, CW, CH)), box, (tw, th))
        # Alpha floor: See-through leaves near-zero alpha over large areas (e.g. a full-canvas
        # mouth layer whose RGB is copied from the input). It is invisible but turns into meshes.
        p = p * (p[..., 3:4] > ALPHA_FLOOR)
        layers.append({"idx": idx, "tag": layer.name, "p": p, "visible": layer.is_visible()})

    by_tag: dict[str, list[dict]] = {}
    for L in layers:
        by_tag.setdefault(L["tag"], []).append(L)

    # Character midline for L/R: the two largest iris (then eye-white) components when they look
    # like a pair (comparable size, side by side), else the face centroid, else the canvas centre.
    split_x, split_src = tw / 2.0, "canvas centre"
    for tag in ("irides", "eyewhite"):
        if tag not in by_tag:
            continue
        comps = sorted(components(by_tag[tag][0]["p"][..., 3]), key=lambda c: -c["n"])
        if len(comps) >= 2:
            a, b = comps[0], comps[1]
            wa = np.ptp(np.nonzero(a["mask"])[1]) + 1
            if b["n"] >= 0.25 * a["n"] and abs(a["cx"] - b["cx"]) >= wa:
                split_x, split_src = (a["cx"] + b["cx"]) / 2.0, f"midpoint of the two largest {tag} components"
                break
    if split_src == "canvas centre" and "face" in by_tag:
        cx = centroid_x(by_tag["face"][0]["p"][..., 3])
        if cx is not None:
            split_x, split_src = cx, "face alpha centroid"

    # Eye centre y per side for the upper/lower lash split: eye-white centroid, else (See-through
    # sometimes emits no eyewhite layer) the iris centroid.
    white_cy: dict[str, float] = {}
    lash_ref = {}
    for tag in ("eyewhite", "irides"):
        if tag not in by_tag:
            continue
        wl, wr = split_lr(by_tag[tag][0]["p"], split_x)
        for S_, q in (("L", wl), ("R", wr)):
            a = q[..., 3]
            if S_ not in white_cy and a.sum() > 0:
                white_cy[S_] = float((np.arange(th)[:, None] * a).sum() / a.sum())
                lash_ref[S_] = tag

    # parts: id -> list of (order key, premultiplied canvas, source tag)
    parts: dict[str, list] = {}
    mapping_rows = []
    unmapped = []

    def add(pid: str, order: float, p: np.ndarray, tag: str):
        if float(p[..., 3].max()) <= ALPHA_ON:
            return
        parts.setdefault(pid, []).append((order, p, tag))

    for L in layers:
        tag, p, order = L["tag"], L["p"], float(L["idx"])
        pid, how, note = TAG_MAP.get(tag, (None, "extra", "알 수 없는 태그: flat2rig 파츠 없음"))
        if how in ("direct", "merge"):
            add(pid, order, p, tag)
            mapping_rows.append({"tag": tag, "parts": [pid], "how": how, "note": note})
        elif how == "lr":
            pl, pr = split_lr(p, split_x)
            add(pid.format(S="L"), order, pl, tag)
            add(pid.format(S="R"), order, pr, tag)
            mapping_rows.append({"tag": tag, "parts": [pid.format(S="L"), pid.format(S="R")], "how": how, "note": note})
        elif how == "lash":
            pl, pr = split_lr(p, split_x)
            got = []
            for S_, q in (("L", pl), ("R", pr)):
                up = np.zeros(q.shape[:2], bool)
                cy0 = white_cy.get(S_)
                for c in components(q[..., 3]):
                    if cy0 is None or c["cy"] <= cy0:
                        up |= c["mask"]
                faint = (q[..., 3] > 0) & (q[..., 3] <= 0.1)
                if cy0 is not None:
                    up |= faint & (np.arange(th)[:, None] <= cy0)
                else:
                    up |= faint
                add(f"eye{S_}_upper", order, q * up[..., None], tag)
                add(f"eye{S_}_lower", order, q * (~up)[..., None], tag)
                got += [f"eye{S_}_upper", f"eye{S_}_lower"]
            mapping_rows.append({"tag": tag, "parts": got, "how": how, "note": note})
        else:
            xid = "extra_" + tag.replace(" ", "_")
            add(xid, order, p, tag)
            mapping_rows.append({"tag": tag, "parts": [xid], "how": "extra", "note": note})
            unmapped.append({"tag": tag, "file": f"{xid}.png", "reason": note})

    # Composite each part in PSD order and write.
    draw = []
    manifest_parts, extra_parts, reports = {}, {}, {}
    full = np.zeros((th, tw, 4), np.float32)
    for pid, items in parts.items():
        items.sort(key=lambda t: t[0])
        acc = np.zeros((th, tw, 4), np.float32)
        for _, p, _ in items:
            acc = over_p(acc, p)
        img = unpremultiply_u8(acc)
        Image.fromarray(img, "RGBA").save(out / f"{pid}.png")
        (extra_parts if pid.startswith("extra_") else manifest_parts)[pid] = f"{pid}.png"
        draw.append((min(t[0] for t in items), pid))
        reports[pid] = {
            "sourceLayers": sorted({t[2] for t in items}),
            "opaquePx": int((img[..., 3] >= 128).sum()),
        }
    draw.sort()
    raw_order = [pid for _, pid in draw]
    # See-through sorts layers by median pseudo-depth; eyebrows / eyewear on the forehead often get
    # a larger depth than the face and land *below* the opaque face layer, so they vanish from the
    # PSD's own composite. flat2rig always draws face features above the face (PART_DEFS order), so
    # lift any face feature that sits below the face to just above it (relative order kept).
    order = list(raw_order)
    lifted = []
    if "face" in order:
        fi = order.index("face")
        below = [p for p in order[:fi] if p in FACE_FEATURES or p in FACE_FEATURE_EXTRAS]
        if below:
            order = [p for p in order if p not in below]
            fi = order.index("face")
            order[fi + 1 : fi + 1] = below
            lifted = below
    # Within an eye flat2rig draws white < iris (< lids). See-through can stack the eye white above
    # the iris, which paints the white over the pupil; move such a white just below its iris.
    eye_fixed = []
    for S_ in ("L", "R"):
        w_id, i_id = f"eye{S_}_white", f"eye{S_}_iris"
        if w_id in order and i_id in order and order.index(w_id) > order.index(i_id):
            order.remove(w_id)
            order.insert(order.index(i_id), w_id)
            eye_fixed.append(w_id)
    parts_img = {pid: np.asarray(Image.open(out / f"{pid}.png"), np.float32) / 255.0 for pid in order}
    for pid in order:
        full = over_p(full, premul(parts_img[pid]))
    Image.fromarray(unpremultiply_u8(full), "RGBA").save(out / "_recomposite.png")
    raw_full = np.zeros((th, tw, 4), np.float32)
    for L in layers:
        raw_full = over_p(raw_full, L["p"])
    Image.fromarray(unpremultiply_u8(raw_full), "RGBA").save(out / "_recomposite_seethrough_order.png")

    manifest = {
        "width": tw,
        "height": th,
        "source": "See-through (shitagaki-lab/see-through) layered PSD, converted by tools/evaluation/v2_seethrough_to_layers.py",
        "name": args.name or out.name,
        "psd": str(psd_path).replace("\\", "/"),
        "psdSha256": sha256(psd_path),
        "srcImage": str(src_path).replace("\\", "/"),
        "srcSha256": sha256(src_path),
        "srcSize": [W, H],
        "seeThroughCanvas": [CW, CH],
        "transform": {
            "padToSquare": {"side": S, "padLeft": px1, "padTop": py1},
            "canvasBox": list(box),
            "filter": "PIL BOX (down) / BILINEAR (up) on premultiplied float RGBA",
            "note": "canvas px edge x_c = (x_src + padLeft) * canvas/side; target grid scales the source grid uniformly",
        },
        "lrConvention": "character-relative (AGENTS.md 1): L = screen right (x > splitX)",
        "splitX": split_x,
        "splitXSource": split_src,
        "lashSplitY": {S_: {"y": v, "source": lash_ref[S_]} for S_, v in white_cy.items()},
        "parts": manifest_parts,
        "extraLayers": extra_parts,
        "drawOrder": order,
        "seeThroughDrawOrder": raw_order,
        "liftedAboveFace": lifted,
        "eyeWhiteMovedBelowIris": eye_fixed,
        "alphaFloor": "alpha <= 4/255 set to 0 before conversion",
        "tagMapping": mapping_rows,
        "unmappedLayers": unmapped,
        "notes": [
            "Parts are straight-alpha full-canvas RGBA including See-through's inpainted hidden areas.",
            "seeThroughDrawOrder is bottom to top by the first source layer (See-through depth order); split L/R parts share their source layer's slot.",
            "drawOrder = seeThroughDrawOrder with face features found below the face (liftedAboveFace) moved to just above it and eye whites above their iris (eyeWhiteMovedBelowIris) moved just below it; _recomposite.png uses drawOrder, _recomposite_seethrough_order.png the raw PSD order.",
            "mouth/nose RGB is copied from the input by See-through (further_extr), so they carry no hidden-area colour.",
        ],
        "partReports": reports,
    }
    (out / "truth-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rig_of = lambda pid: RIG_ID.get(pid, pid)  # noqa: E731
    files_all = {**manifest_parts, **extra_parts}
    rig = {
        "width": tw,
        "height": th,
        "parts": {rig_of(pid): files_all[pid] for pid in order},
        "order": [rig_of(pid) for pid in order],
        "source": "See-through via tools/evaluation/v2_seethrough_to_layers.py (rig input; truth-manifest.json holds the full record)",
        "rigIds": {pid: rig_of(pid) for pid in order if rig_of(pid) != pid},
        "orderNote": "See-through depth order with face features lifted above the face (truth-manifest.json liftedAboveFace)",
    }
    (out / "layers.json").write_text(json.dumps(rig, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(manifest_parts)} parts, {len(extra_parts)} extra layers -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
