//! Finished part layers as input (V1, groundwork for P1 "layered PSD input").
//!
//! A layer directory holds one straight-alpha RGBA PNG per flat2rig part (`<partId>.png`, hidden
//! area already drawn) plus a manifest. Instead of segmenting a flat image and synthesizing hidden
//! texels, every part takes its layer as-is:
//!
//! - `texture` = the layer RGBA, unchanged;
//! - visible `mask` = the layer alpha not covered by the layers above it in compositing order,
//!   `a_p · Π_{q above p} (1 − a_q)` (so the per-part sum is the composite alpha up to rounding);
//! - `fillMask` = the layer alpha wherever the part is (partly) covered, so
//!   `effective_part_mask(mask, fillMask)` is exactly the layer alpha.
//!
//! The composite of the layers (straight-alpha `normal` over, bottom → top) is the image every
//! analysis that needs a flat picture (face detection, landmarks, eye frames) runs on.
//!
//! Manifest (`layers.json`, else `truth-manifest.json` written by
//! `tools/evaluation/himikan_truth.py`):
//!
//! ```json
//! {
//!   "width": 683, "height": 1024,
//!   "parts": { "face": "face.png", "hair_front": "hair_front.png" },
//!   "order": ["face", "hair_front"]
//! }
//! ```
//!
//! `order` (optional) lists every part id bottom → top; without it the flat2rig default order
//! (`default_layer_order`) is used. Unknown keys (`excluded`, `unmappedParts`, …) are kept
//! verbatim in the report so missing artwork is recorded, not hidden.

use crate::analysis::parts::{PartProposalMeta, PartsProposalOutput};
use crate::analysis::state::hash_rgba_bytes;
use crate::core::{blend_straight_pixel, IDS};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, ImageEncoder};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Manifest file names, in lookup order.
pub const MANIFEST_FILES: [&str; 2] = ["layers.json", "truth-manifest.json"];
pub const LAYER_INPUT_VERSION: &str = "layers-v1";
/// `fillKind` of a part whose hidden texels come from the supplied layer.
pub const LAYER_FILL_KIND: &str = "external";

/// A loaded layer directory: part ids bottom → top with their straight-alpha RGBA.
#[derive(Debug, Clone)]
pub struct LayerSet {
    pub dir: PathBuf,
    pub manifest_file: String,
    pub width: usize,
    pub height: usize,
    pub ids: Vec<String>,
    pub files: Vec<String>,
    pub rgba: Vec<Vec<u8>>,
    /// `true` when the order came from the manifest, `false` for the flat2rig default order.
    pub order_from_manifest: bool,
    /// Manifest keys other than `width` / `height` / `parts` / `order`, kept for the report.
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Visible-mask sum check and per-part split, written to `stats.json` (`layerInput`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerStackReport {
    pub version: &'static str,
    pub dir: String,
    pub manifest: String,
    pub order: Vec<String>,
    pub order_source: &'static str,
    /// Default flat2rig parts (`core::IDS`) with no layer in the directory.
    pub missing_default_parts: Vec<String>,
    pub mask_sum: MaskSumCheck,
    pub parts: Vec<LayerPartReport>,
    pub manifest_extra: BTreeMap<String, serde_json::Value>,
}

/// `Σ_p mask_p` against the composite alpha, per pixel, in 8-bit alpha levels.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskSumCheck {
    pub foreground_px: usize,
    pub max_abs_diff: u32,
    pub mean_abs_diff: f64,
    /// Pixels whose difference exceeds one alpha level (rounding of several layers is allowed).
    pub px_over_one_level: usize,
    /// Pixels with a visible contribution from more than one part (anti-aliased internal edges,
    /// semi-transparent layers); the rest owner is the topmost of them.
    pub shared_visible_px: usize,
    pub matches: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerPartReport {
    pub id: String,
    pub file: String,
    pub layer_px: usize,
    pub visible_px: usize,
    pub fill_px: usize,
}

fn valid_part_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn safe_relative_file(file: &str) -> bool {
    let p = Path::new(file);
    !file.is_empty()
        && p.is_relative()
        && p.components().all(|c| matches!(c, std::path::Component::Normal(_)))
}

/// Default compositing rank (bottom = 0). Mirrors the order the automatic proposal emits:
/// back hair, body, neck, clothes, face, eyes (white < iris < lower < upper), brows, mouth,
/// other parts, side hair, front hair. Ties keep the id order.
fn default_rank(id: &str) -> usize {
    const FIXED: [&str; 22] = [
        "body",
        "neck",
        "clothes",
        "clothes_inner",
        "face",
        "face_detail",
        "clothes_hands",
        "eyeR_white",
        "eyeR_iris",
        "eyeR_lower",
        "eyeR_upper",
        "eyeL_white",
        "eyeL_iris",
        "eyeL_lower",
        "eyeL_upper",
        "browR",
        "browL",
        "mouth_inner",
        "mouth_lower",
        "mouth_upper",
        // Placeholders keep the bands below readable.
        "",
        "",
    ];
    if id.starts_with("hair_back") {
        return 0;
    }
    if id.starts_with("hair_side") {
        return 200;
    }
    if id.starts_with("hair") {
        return 300;
    }
    match FIXED.iter().position(|&f| !f.is_empty() && f == id) {
        Some(i) => 10 + i,
        None => 100,
    }
}

/// flat2rig default bottom → top order of `ids` (stable for equal ranks).
pub fn default_layer_order(ids: &[String]) -> Vec<String> {
    let mut out = ids.to_vec();
    out.sort_by_key(|id| default_rank(id));
    out
}

/// Loads `dir` (see the module docs). Every layer must share the manifest canvas.
pub fn load_layer_dir(dir: &Path) -> Result<LayerSet, String> {
    let manifest_file = MANIFEST_FILES
        .iter()
        .find(|f| dir.join(f).is_file())
        .ok_or_else(|| {
            format!(
                "{}: no layer manifest ({})",
                dir.display(),
                MANIFEST_FILES.join(" or ")
            )
        })?;
    let raw = std::fs::read(dir.join(manifest_file))
        .map_err(|e| format!("{manifest_file}: {e}"))?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| format!("{manifest_file}: {e}"))?;
    let obj = manifest
        .as_object()
        .ok_or_else(|| format!("{manifest_file}: expected a JSON object"))?;
    let dim = |k: &str| -> Result<usize, String> {
        obj.get(k)
            .and_then(|v| v.as_u64())
            .filter(|&v| v > 0)
            .map(|v| v as usize)
            .ok_or_else(|| format!("{manifest_file}: missing positive integer '{k}'"))
    };
    let (width, height) = (dim("width")?, dim("height")?);
    crate::core::check_size(width, height)?;
    let parts = obj
        .get("parts")
        .and_then(|v| v.as_object())
        .filter(|m| !m.is_empty())
        .ok_or_else(|| format!("{manifest_file}: 'parts' must be a non-empty object"))?;
    let mut file_of: HashMap<String, String> = HashMap::new();
    for (id, file) in parts {
        if !valid_part_id(id) {
            return Err(format!("{manifest_file}: invalid part id '{id}'"));
        }
        let file = file
            .as_str()
            .filter(|f| safe_relative_file(f))
            .ok_or_else(|| format!("{manifest_file}: part '{id}' needs a relative file name"))?;
        file_of.insert(id.clone(), file.to_string());
    }
    let ids_sorted: Vec<String> = parts.keys().cloned().collect();
    let (ids, order_from_manifest) = match obj.get("order") {
        Some(order) => {
            let order: Vec<String> = order
                .as_array()
                .ok_or_else(|| format!("{manifest_file}: 'order' must be an array of part ids"))?
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or_else(|| format!("{manifest_file}: 'order' must be an array of part ids"))?;
            let mut seen = std::collections::HashSet::new();
            for id in &order {
                if !file_of.contains_key(id) {
                    return Err(format!("{manifest_file}: 'order' lists '{id}' which has no file in 'parts'"));
                }
                if !seen.insert(id.as_str()) {
                    return Err(format!("{manifest_file}: 'order' lists '{id}' twice"));
                }
            }
            if let Some(missing) = ids_sorted.iter().find(|id| !seen.contains(id.as_str())) {
                return Err(format!("{manifest_file}: 'order' does not list part '{missing}'"));
            }
            (order, true)
        }
        None => (default_layer_order(&ids_sorted), false),
    };
    let mut files = Vec::with_capacity(ids.len());
    let mut rgba = Vec::with_capacity(ids.len());
    for id in &ids {
        let file = &file_of[id];
        let img = image::open(dir.join(file))
            .map_err(|e| format!("{}: {e}", dir.join(file).display()))?
            .into_rgba8();
        if img.dimensions() != (width as u32, height as u32) {
            return Err(format!(
                "{file} is {}x{}, manifest canvas is {width}x{height}",
                img.width(),
                img.height()
            ));
        }
        files.push(file.clone());
        rgba.push(img.into_raw());
    }
    let extra = obj
        .iter()
        .filter(|(k, _)| !matches!(k.as_str(), "width" | "height" | "parts" | "order"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Ok(LayerSet {
        dir: dir.to_path_buf(),
        manifest_file: manifest_file.to_string(),
        width,
        height,
        ids,
        files,
        rgba,
        order_from_manifest,
        extra,
    })
}

/// Straight-alpha `normal` composite of the layers, bottom → top.
pub fn composite_layers(set: &LayerSet) -> Vec<u8> {
    let n = set.width * set.height;
    let mut out = vec![0u8; n * 4];
    for layer in &set.rgba {
        for (d, s) in out.as_chunks_mut::<4>().0.iter_mut().zip(layer.as_chunks::<4>().0) {
            if s[3] > 0 {
                blend_straight_pixel(d, *s);
            }
        }
    }
    out
}

/// Visible masks (uncovered alpha) and fill masks (layer alpha where covered) per layer.
pub fn split_visible_fill(set: &LayerSet) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let n = set.width * set.height;
    let layers = set.rgba.len();
    let mut masks = vec![vec![0u8; n]; layers];
    let mut fills = vec![vec![0u8; n]; layers];
    for i in 0..n {
        let mut uncovered = 1.0f64;
        for p in (0..layers).rev() {
            let a8 = set.rgba[p][i * 4 + 3];
            if a8 == 0 {
                continue;
            }
            let a = a8 as f64 / 255.0;
            let vis = (a * uncovered * 255.0).round().min(a8 as f64) as u8;
            masks[p][i] = vis;
            if vis < a8 {
                fills[p][i] = a8;
            }
            uncovered *= 1.0 - a;
        }
    }
    (masks, fills)
}

fn mask_sum_check(masks: &[Vec<u8>], composite: &[u8]) -> MaskSumCheck {
    let n = composite.len() / 4;
    let mut c = MaskSumCheck::default();
    let mut total = 0u64;
    for i in 0..n {
        let mut sum = 0u32;
        let mut contributors = 0usize;
        for m in masks {
            if m[i] > 0 {
                sum += m[i] as u32;
                contributors += 1;
            }
        }
        let alpha = composite[i * 4 + 3] as u32;
        if alpha > 0 {
            c.foreground_px += 1;
        }
        if contributors > 1 {
            c.shared_visible_px += 1;
        }
        let d = sum.abs_diff(alpha);
        total += d as u64;
        c.max_abs_diff = c.max_abs_diff.max(d);
        if d > 1 {
            c.px_over_one_level += 1;
        }
    }
    c.mean_abs_diff = if n > 0 { total as f64 / n as f64 } else { 0.0 };
    // Per-layer rounding can stack one level per contributing layer; allow that, nothing more.
    c.matches = c.max_abs_diff <= masks.len().max(1) as u32;
    c
}

fn encode_png_data_url(rgba: &[u8], w: usize, h: usize) -> Result<String, String> {
    let mut png = Vec::new();
    PngEncoder::new_with_quality(&mut png, CompressionType::Fast, FilterType::Sub)
        .write_image(rgba, w as u32, h as u32, ColorType::Rgba8.into())
        .map_err(|e| e.to_string())?;
    Ok(format!("data:image/png;base64,{}", STANDARD.encode(&png)))
}

/// Part proposal built from finished layers (no segmentation, no hidden-texel synthesis).
/// `composite` is `composite_layers(set)`.
pub fn proposal_from_layers(
    set: &LayerSet,
    composite: &[u8],
) -> Result<(PartsProposalOutput, LayerStackReport), String> {
    let (w, h) = (set.width, set.height);
    let (masks, fills) = split_visible_fill(set);
    let mask_sum = mask_sum_check(&masks, composite);
    let mut textures = Vec::with_capacity(set.ids.len());
    let mut fill_masks = Vec::with_capacity(set.ids.len());
    let mut fill_kinds = Vec::with_capacity(set.ids.len());
    let mut metas = Vec::with_capacity(set.ids.len());
    let mut parts = Vec::with_capacity(set.ids.len());
    for (p, id) in set.ids.iter().enumerate() {
        let layer_px = set.rgba[p].as_chunks::<4>().0.iter().filter(|px| px[3] > 0).count();
        let visible_px = masks[p].iter().filter(|&&a| a > 0).count();
        let fill_px = fills[p].iter().filter(|&&a| a > 0).count();
        textures.push(Some(encode_png_data_url(&set.rgba[p], w, h)?));
        if fill_px > 0 {
            fill_masks.push(Some(fills[p].clone()));
            fill_kinds.push(LAYER_FILL_KIND.to_string());
        } else {
            fill_masks.push(None);
            fill_kinds.push("none".to_string());
        }
        let mut diagnostics = Vec::new();
        if visible_px == 0 {
            diagnostics.push("fully_covered".to_string());
        }
        metas.push(PartProposalMeta {
            id: id.clone(),
            provenance: "layer".into(),
            confidence: 1.0,
            pixel_count: visible_px,
            diagnostics,
        });
        parts.push(LayerPartReport {
            id: id.clone(),
            file: set.files[p].clone(),
            layer_px,
            visible_px,
            fill_px,
        });
    }
    let missing_default_parts = IDS
        .iter()
        .filter(|d| !set.ids.iter().any(|id| id == *d))
        .map(|d| d.to_string())
        .collect();
    let report = LayerStackReport {
        version: LAYER_INPUT_VERSION,
        dir: set.dir.to_string_lossy().into_owned(),
        manifest: set.manifest_file.clone(),
        order: set.ids.clone(),
        order_source: if set.order_from_manifest { "manifest" } else { "default" },
        missing_default_parts,
        mask_sum,
        parts,
        manifest_extra: set.extra.clone(),
    };
    let proposal = PartsProposalOutput {
        part_ids: set.ids.clone(),
        masks,
        fill_masks,
        fill_kinds,
        fill_regions: vec![None; set.ids.len()],
        textures,
        metas,
        provider: "layers".into(),
        model_hashes: HashMap::new(),
        postprocess_version: LAYER_INPUT_VERSION.into(),
        image_hash: hash_rgba_bytes(composite, w, h),
        style_match: None,
    };
    Ok((proposal, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::effective_part_mask;

    fn set_of(w: usize, h: usize, layers: Vec<(&str, Vec<u8>)>) -> LayerSet {
        LayerSet {
            dir: PathBuf::from("mem"),
            manifest_file: "layers.json".into(),
            width: w,
            height: h,
            ids: layers.iter().map(|(id, _)| id.to_string()).collect(),
            files: layers.iter().map(|(id, _)| format!("{id}.png")).collect(),
            rgba: layers.into_iter().map(|(_, px)| px).collect(),
            order_from_manifest: true,
            extra: BTreeMap::new(),
        }
    }

    fn solid(n: usize, rgb: [u8; 3], alpha: &[u8]) -> Vec<u8> {
        (0..n)
            .flat_map(|i| [rgb[0], rgb[1], rgb[2], alpha[i]])
            .collect()
    }

    #[test]
    fn visible_is_uncovered_alpha_and_effective_is_layer_alpha() {
        // 4 px: face everywhere opaque; hair opaque on px 1, half on px 2, absent on 0 and 3.
        let n = 4;
        let face = solid(n, [240, 200, 180], &[255, 255, 255, 0]);
        let hair = solid(n, [60, 40, 30], &[0, 255, 128, 0]);
        let set = set_of(n, 1, vec![("face", face), ("hair_front", hair)]);
        let comp = composite_layers(&set);
        let (prop, rep) = proposal_from_layers(&set, &comp).unwrap();
        let (face_m, hair_m) = (&prop.masks[0], &prop.masks[1]);
        assert_eq!(face_m, &vec![255, 0, 127, 0]);
        assert_eq!(hair_m, &vec![0, 255, 128, 0]);
        // Face is hidden under hair on px 1 and partly on px 2: fill = its own alpha there.
        assert_eq!(prop.fill_masks[0].as_deref(), Some(&[0u8, 255, 255, 0][..]));
        assert!(prop.fill_masks[1].is_none());
        assert_eq!(prop.fill_kinds, vec!["external", "none"]);
        for p in 0..2 {
            let eff = effective_part_mask(&prop.masks[p], prop.fill_masks[p].as_deref());
            let alpha: Vec<u8> = set.rgba[p].chunks(4).map(|c| c[3]).collect();
            assert_eq!(eff, alpha, "effective mask must equal the layer alpha");
        }
        assert!(rep.mask_sum.matches, "{:?}", rep.mask_sum);
        assert!(rep.mask_sum.max_abs_diff <= 1);
        assert_eq!(rep.mask_sum.shared_visible_px, 1);
        assert_eq!(rep.mask_sum.foreground_px, 3);
    }

    #[test]
    fn textures_are_the_layer_pixels_unchanged() {
        let n = 3;
        let layer = vec![10, 20, 30, 255, 1, 2, 3, 7, 0, 0, 0, 0];
        let set = set_of(n, 1, vec![("face", layer.clone())]);
        let comp = composite_layers(&set);
        let (prop, _) = proposal_from_layers(&set, &comp).unwrap();
        let b64 = prop.textures[0].as_deref().unwrap().strip_prefix("data:image/png;base64,").unwrap();
        let img = image::load_from_memory(&STANDARD.decode(b64).unwrap()).unwrap().into_rgba8();
        assert_eq!(img.into_raw(), layer);
    }

    #[test]
    fn default_order_follows_flat2rig_stack() {
        let ids: Vec<String> = [
            "hair_front", "mouth_upper", "face", "hair_side_L", "eyeR_iris", "hair_back", "eyeR_white",
            "neck", "clothes", "hair_side_R", "browL", "acc_ribbon",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(
            default_layer_order(&sorted),
            [
                "hair_back", "neck", "clothes", "face", "eyeR_white", "eyeR_iris", "browL", "mouth_upper",
                "acc_ribbon", "hair_side_L", "hair_side_R", "hair_front",
            ]
        );
    }

    #[test]
    fn load_layer_dir_reads_manifest_order_and_rejects_bad_input() {
        let dir = tempfile::tempdir().unwrap();
        let px = |a: u8| image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, a]));
        px(255).save(dir.path().join("face.png")).unwrap();
        px(200).save(dir.path().join("hair.png")).unwrap();
        let write = |v: serde_json::Value| {
            std::fs::write(dir.path().join("layers.json"), serde_json::to_vec(&v).unwrap()).unwrap()
        };
        write(serde_json::json!({
            "width": 2, "height": 2,
            "parts": {"hair_front": "hair.png", "face": "face.png"},
            "excluded": ["nose"]
        }));
        let set = load_layer_dir(dir.path()).unwrap();
        assert_eq!(set.ids, ["face", "hair_front"]);
        assert!(!set.order_from_manifest);
        assert_eq!(set.extra.get("excluded"), Some(&serde_json::json!(["nose"])));

        write(serde_json::json!({
            "width": 2, "height": 2,
            "parts": {"hair_front": "hair.png", "face": "face.png"},
            "order": ["hair_front", "face"]
        }));
        let set = load_layer_dir(dir.path()).unwrap();
        assert_eq!(set.ids, ["hair_front", "face"]);
        assert!(set.order_from_manifest);

        for bad in [
            serde_json::json!({"width": 2, "height": 2, "parts": {"face": "face.png"}, "order": ["face", "hair"]}),
            serde_json::json!({"width": 2, "height": 2, "parts": {"face": "face.png", "hair_front": "hair.png"}, "order": ["face"]}),
            serde_json::json!({"width": 2, "height": 2, "parts": {"face": "../face.png"}}),
            serde_json::json!({"width": 2, "height": 2, "parts": {"fa ce": "face.png"}}),
            serde_json::json!({"width": 3, "height": 2, "parts": {"face": "face.png"}}),
        ] {
            write(bad.clone());
            assert!(load_layer_dir(dir.path()).is_err(), "accepted {bad}");
        }
    }

    /// B1: the app builds layer parts in TypeScript (`src/layerParts.ts`); both sides assert the
    /// same visible / fill split and composite against `tests/fixtures/layer-split-parity.json`.
    #[test]
    fn split_and_composite_match_the_shared_ts_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/layer-split-parity.json");
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert!(cases.len() >= 3);
        let bytes = |v: &serde_json::Value| -> Vec<u8> {
            v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect()
        };
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let w = case["width"].as_u64().unwrap() as usize;
            let h = case["height"].as_u64().unwrap() as usize;
            let layers: Vec<(String, Vec<u8>)> = case["layers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| (l["id"].as_str().unwrap().to_string(), bytes(&l["rgba"])))
                .collect();
            let set = set_of(w, h, layers.iter().map(|(id, px)| (id.as_str(), px.clone())).collect());
            let (masks, fills) = split_visible_fill(&set);
            let composite = composite_layers(&set);
            let exp = &case["expected"];
            for (p, (m, f)) in masks.iter().zip(&fills).enumerate() {
                assert_eq!(m, &bytes(&exp["masks"][p]), "{name}: mask of layer {p}");
                assert_eq!(f, &bytes(&exp["fills"][p]), "{name}: fill of layer {p}");
            }
            assert_eq!(composite, bytes(&exp["composite"]), "{name}: composite");
            let check = mask_sum_check(&masks, &composite);
            assert_eq!(check.max_abs_diff as u64, exp["maskSum"]["maxAbsDiff"].as_u64().unwrap(), "{name}");
            assert_eq!(check.shared_visible_px as u64, exp["maskSum"]["sharedVisiblePx"].as_u64().unwrap(), "{name}");
            assert_eq!(check.foreground_px as u64, exp["maskSum"]["foregroundPx"].as_u64().unwrap(), "{name}");
            assert!(check.matches, "{name}: visible masks must sum to the composite alpha");
        }
    }
}
