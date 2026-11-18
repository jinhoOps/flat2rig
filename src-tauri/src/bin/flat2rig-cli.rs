use base64::{engine::general_purpose::STANDARD, Engine};
use flat2rig_lib::analysis::{
    detector::{detect_faces, DEFAULT_FACE_SCORE_THRESHOLD},
    landmarks::detect_landmarks,
    layer_input,
    lineart,
    models::{ModelRegistry, ModelRole},
    parts::{propose_parts_auto_with_template_stage, propose_parts_with_sam_with_options},
    state::AnalysisState,
    templates::{with_eval_holdout, TemplateStyleSelection},
};
use flat2rig_lib::core;
use flat2rig_lib::rig::{exposure, eye_seal, fidelity, seams};
use serde_json::json;
use std::path::PathBuf;

fn decode_part_rgba(tex_url: Option<&str>, fallback: &image::RgbaImage) -> image::RgbaImage {
    if let Some(url) = tex_url {
        if let Some(b64) = url.strip_prefix("data:image/png;base64,") {
            if let Ok(bytes) = STANDARD.decode(b64) {
                if let Ok(img) = image::load_from_memory(&bytes) {
                    return img.into_rgba8();
                }
            }
        }
    }
    fallback.clone()
}

fn alpha_blend_layer(dst: &mut image::RgbaImage, src: &image::RgbaImage) {
    for (dp, sp) in dst.pixels_mut().zip(src.pixels()) {
        core::blend_straight_pixel(&mut dp.0, sp.0);
    }
}

fn draw_deformed_part(
    dst: &mut image::RgbaImage,
    src_tex: &image::RgbaImage,
    mesh: &core::Mesh,
    id: &str,
    lm: &core::Landmarks,
    pose: &core::Pose,
) {
    let (w, h) = (dst.width() as i32, dst.height() as i32);
    let src_raw = src_tex.as_raw();
    let stride = dst.width() as usize;
    let dst_raw = dst.as_mut();

    for tri in mesh.triangles.as_chunks::<3>().0 {
        let s0 = mesh.vertices[tri[0]];
        let s1 = mesh.vertices[tri[1]];
        let s2 = mesh.vertices[tri[2]];
        let d0 = core::deform(s0, id, lm, pose);
        let d1 = core::deform(s1, id, lm, pose);
        let d2 = core::deform(s2, id, lm, pose);

        let denom = (d1.y - d2.y) * (d0.x - d2.x) + (d2.x - d1.x) * (d0.y - d2.y);
        if denom.abs() < 1e-4 {
            continue;
        }
        let min_x = d0.x.min(d1.x).min(d2.x).floor().max(0.0) as i32;
        let max_x = d0.x.max(d1.x).max(d2.x).ceil().min((w - 1) as f64) as i32;
        let min_y = d0.y.min(d1.y).min(d2.y).floor().max(0.0) as i32;
        let max_y = d0.y.max(d1.y).max(d2.y).ceil().min((h - 1) as f64) as i32;

        for y in min_y..=max_y {
            let py = y as f64 + 0.5;
            for x in min_x..=max_x {
                let px = x as f64 + 0.5;
                let w0 = ((d1.y - d2.y) * (px - d2.x) + (d2.x - d1.x) * (py - d2.y)) / denom;
                let w1 = ((d2.y - d0.y) * (px - d2.x) + (d0.x - d2.x) * (py - d2.y)) / denom;
                let w2 = 1.0 - w0 - w1;
                if w0 < -1e-3 || w1 < -1e-3 || w2 < -1e-3 {
                    continue;
                }
                let sx = (w0 * s0.x + w1 * s1.x + w2 * s2.x).floor() as i32;
                let sy = (w0 * s0.y + w1 * s1.y + w2 * s2.y).floor() as i32;
                if sx < 0 || sy < 0 || sx >= w || sy >= h {
                    continue;
                }
                let si = (sy as usize * stride + sx as usize) * 4;
                if src_raw[si + 3] <= 2 {
                    continue;
                }
                let di = (y as usize * stride + x as usize) * 4;
                core::blend_straight_pixel(
                    &mut dst_raw[di..di + 4],
                    [src_raw[si], src_raw[si + 1], src_raw[si + 2], src_raw[si + 3]],
                );
            }
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run_analyze(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut full_parts = false;
    let mut positional: Vec<&str> = Vec::new();
    for a in args {
        if a == "--full" {
            full_parts = true;
        } else {
            positional.push(a.as_str());
        }
    }
    if positional.is_empty() {
        return Err("Usage: flat2rig-cli --analyze [--full] INPUT.png [OUTPUT.json]".into());
    }
    let img_path = positional[0];
    let img = image::open(img_path)?.into_rgba8();
    let (w, h) = img.dimensions();
    core::check_size(w as usize, h as usize)?;

    let mut reg = ModelRegistry::new(Vec::new())?;
    let t_det = std::time::Instant::now();
    let det = detect_faces(
        &mut reg,
        img.as_raw(),
        w as usize,
        h as usize,
        DEFAULT_FACE_SCORE_THRESHOLD,
    )?;
    let detect_ms = t_det.elapsed().as_secs_f64() * 1000.0;

    let t_lm = std::time::Instant::now();
    let landmarks = if let Some(top) = det.candidates.first() {
        Some(detect_landmarks(
            &mut reg,
            img.as_raw(),
            w as usize,
            h as usize,
            top.bbox,
            top.crop_bbox,
        )?)
    } else {
        None
    };
    let landmark_ms = if landmarks.is_some() {
        t_lm.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };

    let mut sam_full_ms: Option<f64> = None;
    let mut part_summary = serde_json::Value::Null;
    if full_parts {
        if let (Some(top), Some(ref lm)) = (det.candidates.first(), &landmarks) {
            let has_sam = reg.resolve_path(ModelRole::PartEncoder).is_some()
                && reg.resolve_path(ModelRole::PartDecoder).is_some();
            if has_sam {
                drop(reg);
                let state = AnalysisState::new(None)?;
                let t_sam = std::time::Instant::now();
                let proposed = propose_parts_with_sam_with_options(
                    &state,
                    img.as_raw(),
                    w as usize,
                    h as usize,
                    &lm.rig_landmarks,
                    &lm.points_28,
                    Some(top.bbox),
                    32.0,
                    None,
                    "detail",
                    true,
                )?;
                sam_full_ms = Some(t_sam.elapsed().as_secs_f64() * 1000.0);

                let n = w as usize * h as usize;
                let mut overlap_pixels = 0usize;
                let mut covered_pixels = 0usize;
                for i in 0..n {
                    let owners = proposed.masks.iter().filter(|m| m[i] > 0).count();
                    if owners > 1 {
                        overlap_pixels += 1;
                    }
                    if owners > 0 {
                        covered_pixels += 1;
                    }
                }
                let mut total_tris = 0usize;
                let mut mesh_warnings = Vec::new();
                for (pid, mask) in proposed.part_ids.iter().zip(&proposed.masks) {
                    let mesh = core::art_mesh_for_part(
                        mask,
                        w as usize,
                        h as usize,
                        14,
                        pid,
                        &lm.rig_landmarks,
                    );
                    total_tris += mesh.triangles.len() / 3;
                    let report = core::check_mesh_quality(
                        &mesh,
                        mask,
                        w as usize,
                        h as usize,
                        pid,
                        &lm.rig_landmarks,
                    );
                    mesh_warnings.extend(report.warnings);
                }
                part_summary = json!({
                    "provider": proposed.provider,
                    "partCount": proposed.part_ids.len(),
                    "coveredPixels": covered_pixels,
                    "overlapPixels": overlap_pixels,
                    "totalTriangles": total_tris,
                    "meshWarnings": mesh_warnings,
                    "metas": proposed.metas,
                });
            }
        }
    }

    let payload = json!({
        "image": img_path,
        "width": w,
        "height": h,
        "detectMs": detect_ms,
        "landmarkMs": landmark_ms,
        "samFullMs": sam_full_ms,
        "detection": det,
        "landmarks": landmarks,
        "partSummary": part_summary,
    });
    let pretty = serde_json::to_string_pretty(&payload)?;
    if let Some(out_path) = positional.get(1) {
        if let Some(parent) = std::path::Path::new(out_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(out_path, pretty.as_bytes())?;
    }
    println!("{pretty}");
    Ok(())
}

fn parse_proportion_override(raw: &str) -> Result<String, Box<dyn std::error::Error>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" | "ld" | "sd" => Ok(raw.trim().to_ascii_lowercase()),
        other => Err(format!("Unsupported --proportion '{other}' (expected auto, ld, or sd)").into()),
    }
}

fn parse_stage_override(raw: &str) -> Result<String, Box<dyn std::error::Error>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "full" | "base_silhouette" | "features_overlay" | "clothes_overlay" | "hair_overlay"
        | "base_body" => Ok(raw.trim().to_ascii_lowercase()),
        other => Err(format!(
            "Unsupported --stage '{other}' (expected full, base_silhouette, features_overlay, clothes_overlay, hair_overlay, or base_body)"
        )
        .into()),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|s| s.as_str()) == Some("--analyze") {
        return run_analyze(&args[1..]);
    }
    let mut proportion: Option<String> = None;
    let mut stage: Option<String> = None;
    let mut coupled = true;
    let mut legacy_eye = false;
    let mut line_art = true;
    let mut rest_gate = true;
    let mut eval = EvalOptions::default();
    let mut from_layers: Option<PathBuf> = None;
    let mut pose_renders = false;
    let mut positional: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if let Some(val) = arg.strip_prefix("--proportion=") {
            proportion = Some(parse_proportion_override(val)?);
        } else if arg == "--proportion" {
            i += 1;
            let val = args
                .get(i)
                .ok_or("Missing value for --proportion (expected auto, ld, or sd)")?;
            proportion = Some(parse_proportion_override(val)?);
        } else if let Some(val) = arg.strip_prefix("--stage=") {
            stage = Some(parse_stage_override(val)?);
        } else if arg == "--stage" {
            i += 1;
            let val = args.get(i).ok_or(
                "Missing value for --stage (expected full, base_silhouette, features_overlay, clothes_overlay, hair_overlay, or base_body)",
            )?;
            stage = Some(parse_stage_override(val)?);
        } else if arg == "--uncoupled" {
            // T13b baseline: run the whole pipeline (hidden-area synthesis, seam requirements,
            // exposure map, motion previews) with the pre-T13b uncoupled `deform`.
            coupled = false;
        } else if arg == "--legacy-eye" {
            // T10 baseline (measurement only): the whole pipeline with the pre-T10 per-layer eye
            // formulas and clipping rule.
            legacy_eye = true;
        } else if arg == "--lineart-no-rest-gate" {
            // T15 measurement only: synthesized lines without the bit-exact rest composite gate.
            rest_gate = false;
        } else if arg == "--pose-renders" {
            // Envelope extremes rendered with the motion-preview rasterizer (`poses/`).
            pose_renders = true;
        } else if arg == "--no-lineart" {
            // T15 baseline (measurement only): hidden texels keep the T9 colour layer, no
            // synthesized line layer.
            line_art = false;
        } else if arg == "--eval-holdout" {
            // T14 (evaluation only): seam holdout re-synthesized by the hidden-texel path.
            eval.holdout = true;
        } else if let Some(val) = arg.strip_prefix("--eval-truth=") {
            eval.truth_dir = Some(PathBuf::from(val));
        } else if arg == "--eval-truth" {
            i += 1;
            let val = args
                .get(i)
                .ok_or("Missing value for --eval-truth (directory with truth-manifest.json)")?;
            eval.truth_dir = Some(PathBuf::from(val));
        } else if let Some(val) = arg.strip_prefix("--from-layers=") {
            from_layers = Some(PathBuf::from(val));
        } else if arg == "--from-layers" {
            i += 1;
            let val = args
                .get(i)
                .ok_or("Missing value for --from-layers (directory with <partId>.png + layers.json)")?;
            from_layers = Some(PathBuf::from(val));
        } else if arg.starts_with("--") {
            return Err(format!("Unknown flag: {arg}").into());
        } else {
            positional.push(arg);
        }
        i += 1;
    }
    const USAGE: &str = "Usage: flat2rig-cli [--proportion auto|ld|sd] [--uncoupled] [--legacy-eye] [--no-lineart] [--pose-renders] [--eval-holdout] [--eval-truth DIR] [--stage full|base_silhouette|features_overlay|clothes_overlay|hair_overlay|base_body] INPUT.png OUTPUT_DIR [MODEL.onnx] [LANDMARKS.json] | --from-layers LAYER_DIR [--uncoupled] [--legacy-eye] [--pose-renders] OUTPUT_DIR [LANDMARKS.json] | --analyze [--full] INPUT.png [OUTPUT.json]";
    let input = if let Some(dir) = from_layers {
        // Finished part layers: segmentation and hidden-area synthesis do not run, so the flags
        // that only steer them (or re-run them for evaluation) do not apply.
        if proportion.is_some() || stage.is_some() || eval.any() {
            return Err(
                "--from-layers cannot be combined with --proportion, --stage, --eval-holdout or --eval-truth".into(),
            );
        }
        if positional.is_empty() || positional.len() > 2 {
            return Err(USAGE.into());
        }
        PipelineInput {
            source: PartSource::Layers(dir),
            output: positional[0],
            landmarks: positional.get(1).copied(),
            pose_renders,
        }
    } else {
        if positional.len() < 2 {
            return Err(USAGE.into());
        }
        PipelineInput {
            source: PartSource::Image {
                path: positional[0],
                model: positional.get(2).copied(),
            },
            output: positional[1],
            landmarks: positional.get(3).copied(),
            pose_renders,
        }
    };
    core::with_legacy_eye_rig(legacy_eye, || {
        core::with_seam_coupling(coupled, || {
            lineart::with_lineart(line_art, || {
                let go = || run_pipeline(&input, proportion, stage, coupled, legacy_eye, &eval);
                if rest_gate { go() } else { lineart::without_rest_gate(go) }
            })
        })
    })
}

/// Where the parts come from.
enum PartSource<'a> {
    /// One flat image: automatic segmentation + hidden-area synthesis (the default run).
    Image { path: &'a str, model: Option<&'a str> },
    /// A directory of finished part layers (`analysis::layer_input`).
    Layers(PathBuf),
}

struct PipelineInput<'a> {
    source: PartSource<'a>,
    output: &'a str,
    landmarks: Option<&'a str>,
    /// Write `poses/` (`POSE_RENDERS`); off by default so the default run's outputs are unchanged.
    pose_renders: bool,
}

/// Envelope poses rendered by `--pose-renders`: the range extremes of head / body / hair / eyes /
/// mouth, head-body opposite and same-direction turns, and hair phase extremes. Never narrowed to
/// hide a defect.
const POSE_RENDERS: &[&str] = &[
    "rest",
    "angleX=-30",
    "angleX=30",
    "angleY=-30",
    "angleY=30",
    "angleZ=-30",
    "angleZ=30",
    "opp:angleX+/bodyX-",
    "opp:angleX-/bodyX+",
    "opp:angleY+/bodyY-",
    "opp:angleY-/bodyY+",
    "opp:angleZ+/bodyZ-",
    "opp:angleZ-/bodyZ+",
    "same3:head+--/breath1",
    "same3:head--+/breath1",
    "hair=-1",
    "hair=1",
    "hairSide=-1",
    "hairSide=1",
    "hairBack=-1",
    "hairBack=1",
    "hairSign:+++",
    "hairSign:---",
    "headHairBody:head+-/hair-",
    "headHairBody:head--/hair+",
    "seal:closed/gaze:00",
    "eyesClosedSmile",
    "mouth=1",
    "mouthOpen/form+",
];

/// `--pose-renders`: each `POSE_RENDERS` envelope pose drawn like the motion previews
/// (`draw_deformed_part`, every layer bottom → top) into `poses/<pose>.png`.
#[allow(clippy::too_many_arguments)]
fn render_pose_set(
    output: &std::path::Path,
    ids: &[String],
    meshes: &[core::Mesh],
    textures: &[Vec<u8>],
    landmarks: &core::Landmarks,
    w: u32,
    h: u32,
    env: &[seams::EnvelopePose],
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let dir = output.join("poses");
    std::fs::create_dir_all(&dir)?;
    let mut rows = Vec::new();
    for &name in POSE_RENDERS {
        let pe = env
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| format!("pose envelope has no '{name}'"))?;
        let mut buf = image::RgbaImage::new(w, h);
        for ((id, mesh), tex) in ids.iter().zip(meshes).zip(textures) {
            let tex = image::RgbaImage::from_raw(w, h, tex.clone()).ok_or("pose texture size mismatch")?;
            draw_deformed_part(&mut buf, &tex, mesh, id, landmarks, &pe.pose);
        }
        let file = format!(
            "{}.png",
            name.replace([':', '/', '='], "_").replace('+', "p").replace('-', "m")
        );
        buf.save(dir.join(&file))?;
        rows.push(json!({ "pose": name, "group": pe.group, "file": format!("poses/{file}") }));
    }
    Ok(serde_json::Value::Array(rows))
}

/// T14 evaluation flags. Both off = the product run (outputs unchanged, no `fidelity` key).
#[derive(Default)]
struct EvalOptions {
    holdout: bool,
    truth_dir: Option<PathBuf>,
}

impl EvalOptions {
    fn any(&self) -> bool {
        self.holdout || self.truth_dir.is_some()
    }
}

fn decode_texture_raw(tex_url: Option<&str>) -> Option<Vec<u8>> {
    let b64 = tex_url?.strip_prefix("data:image/png;base64,")?;
    let bytes = STANDARD.decode(b64).ok()?;
    Some(image::load_from_memory(&bytes).ok()?.into_rgba8().into_raw())
}

type TruthSet = (serde_json::Value, std::collections::HashMap<String, Vec<u8>>);

/// Loads `DIR/truth-manifest.json` (`{"width","height","parts":{id: file}}`) written by
/// `tools/evaluation/himikan_truth.py`; every truth image must share the input pixel grid.
fn load_truth(dir: &std::path::Path, w: u32, h: u32) -> Result<TruthSet, Box<dyn std::error::Error>> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("truth-manifest.json"))?)?;
    let (mw, mh) = (
        manifest["width"].as_u64().unwrap_or(0),
        manifest["height"].as_u64().unwrap_or(0),
    );
    if (mw, mh) != (w as u64, h as u64) {
        return Err(format!("truth canvas {mw}x{mh} does not match input {w}x{h}").into());
    }
    let mut out = std::collections::HashMap::new();
    if let Some(parts) = manifest["parts"].as_object() {
        for (id, file) in parts {
            let file = file.as_str().ok_or("truth part file must be a string")?;
            let img = image::open(dir.join(file))?.into_rgba8();
            if img.dimensions() != (w, h) {
                return Err(format!("truth image {file} is not {w}x{h}").into());
            }
            out.insert(id.clone(), img.into_raw());
        }
    }
    Ok((manifest, out))
}

/// T14: fidelity report (`stats.json` `fidelity`) and comparison images in `OUTPUT/fidelity/`.
/// The holdout pass re-runs the same proposal under `with_eval_holdout`; its textures are only
/// compared, never written as parts.
fn run_fidelity<F>(
    eval: &EvalOptions,
    output: &std::path::Path,
    proposed: &flat2rig_lib::analysis::parts::PartsProposalOutput,
    propose: &F,
    requirements: &[seams::SeamRequirement],
    exposure: &exposure::ExposureMap,
    img: &image::RgbaImage,
) -> Result<serde_json::Value, Box<dyn std::error::Error>>
where
    F: Fn() -> Result<flat2rig_lib::analysis::parts::PartsProposalOutput, String>,
{
    let (w, h) = img.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let fid_dir = output.join("fidelity");
    std::fs::create_dir_all(&fid_dir)?;
    let mut fid = serde_json::Map::new();
    fid.insert("spec".into(), fidelity::metrics_spec());
    if eval.holdout {
        let t_h = std::time::Instant::now();
        let plan = fidelity::build_holdout_plan(
            &proposed.part_ids,
            &proposed.masks,
            requirements,
            exposure,
            img.as_raw(),
            wu,
            hu,
        );
        let held = with_eval_holdout(plan.holdout.clone(), propose)?;
        let masks_identical = held.part_ids == proposed.part_ids && held.masks == proposed.masks;
        let fills_identical = held.fill_masks == proposed.fill_masks;
        let held_tex: Vec<Option<Vec<u8>>> = (0..held.part_ids.len())
            .map(|i| decode_texture_raw(held.textures.get(i).and_then(|o| o.as_deref())))
            .collect();
        // Only comparable when the partition is unchanged (holdout touches textures only).
        let parts = if masks_identical {
            fidelity::holdout_fidelity(
                &plan,
                &proposed.part_ids,
                &proposed.masks,
                &held_tex,
                img.as_raw(),
                wu,
                hu,
                Some(&fid_dir),
            )
        } else {
            Vec::new()
        };
        fid.insert(
            "holdout".into(),
            json!({
                "masksIdentical": masks_identical,
                "fillMasksIdentical": fills_identical,
                "heldOutPx": plan.seam_of.iter().filter(|&&s| s >= 0).count(),
                "seams": plan.seams,
                "parts": parts,
                "elapsedMs": t_h.elapsed().as_millis(),
            }),
        );
    }
    if let Some(dir) = &eval.truth_dir {
        let (manifest, truth) = load_truth(dir, w, h)?;
        let product_tex: Vec<Option<Vec<u8>>> = (0..proposed.part_ids.len())
            .map(|i| decode_texture_raw(proposed.textures.get(i).and_then(|o| o.as_deref())))
            .collect();
        let parts = fidelity::ground_truth_fidelity(
            &proposed.part_ids,
            &proposed.masks,
            exposure,
            &product_tex,
            &truth,
            img.as_raw(),
            wu,
            hu,
            Some(&fid_dir),
        );
        let without: Vec<&String> = proposed
            .part_ids
            .iter()
            .filter(|id| !truth.contains_key(*id))
            .collect();
        fid.insert(
            "groundTruth".into(),
            json!({
                "truthDir": dir.to_string_lossy(),
                "layerMap": manifest.get("layerMap"),
                "layerMapVersion": manifest.get("layerMapVersion"),
                "excludedLayers": manifest.get("excluded"),
                "partsWithoutTruth": without,
                "parts": parts,
            }),
        );
    }
    fid.insert("imageDir".into(), json!("fidelity"));
    Ok(serde_json::Value::Object(fid))
}

/// Envelope poses rendered for the T15 face / neck edge comparison (side-hair sway and head
/// rotation extremes, head/body opposite turns).
const LINEART_RENDER_POSES: &[&str] = &[
    "hairSide=-1",
    "hairSide=1",
    "angleX=-30",
    "angleX=30",
    "angleZ=-30",
    "angleZ=30",
    "opp:angleX+/bodyX-",
    "opp:angleX-/bodyX+",
    "hairSign:+++",
    "hairSign:---",
];

/// T15 outputs: per-part line-art reports (+ seam-parallel texels), the posed double-line check,
/// hard renders of the extreme poses (`lineart/pose_*.png`) and crops of skipped strokes
/// (`lineart/skip_*.png`: original · texture, ×4, contact point marked).
#[allow(clippy::too_many_arguments)]
fn lineart_outputs(
    output: &std::path::Path,
    proposed: &flat2rig_lib::analysis::parts::PartsProposalOutput,
    meshes: &[core::Mesh],
    layer_textures: &[Vec<u8>],
    landmarks: &core::Landmarks,
    img: &image::RgbaImage,
    env: &[seams::EnvelopePose],
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let t0 = std::time::Instant::now();
    let (w, h) = img.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let n = wu * hu;
    let dir = output.join("lineart");
    std::fs::create_dir_all(&dir)?;
    let ids = &proposed.part_ids;
    let reps: Vec<Option<&lineart::LineArtReport>> = (0..ids.len())
        .map(|p| {
            proposed
                .fill_regions
                .get(p)
                .and_then(|r| r.as_ref())
                .and_then(|r| r.lineart.as_ref())
        })
        .collect();
    let fills: Vec<Vec<u8>> = (0..ids.len())
        .map(|p| {
            proposed
                .fill_masks
                .get(p)
                .and_then(|o| o.clone())
                .unwrap_or_else(|| vec![0u8; n])
        })
        .collect();
    let mut parts = Vec::new();
    let mut totals = lineart::StrokeCounts::default();
    let (mut contour_len, mut stroke_tex, mut contour_tex, mut seam_par_total) = (0usize, 0usize, 0usize, 0usize);
    let mut gate = lineart::RestGate::default();
    for (p, id) in ids.iter().enumerate() {
        let Some(r) = reps[p] else { continue };
        let seam_par = lineart::seam_parallel_texels(p, r, &proposed.masks, wu, hu);
        seam_par_total += seam_par;
        let s = &r.strokes;
        totals.touching_hidden += s.touching_hidden;
        totals.extended += s.extended;
        totals.extended_by_field += s.extended_by_field;
        totals.joined += s.joined;
        totals.skipped_short += s.skipped_short;
        totals.skipped_fit_residual += s.skipped_fit_residual;
        totals.skipped_crossing += s.skipped_crossing;
        totals.unexposed += s.unexposed;
        totals.feature_seam += s.feature_seam;
        totals.parallel += s.parallel;
        totals.contour += s.contour;
        totals.seam_parallel += s.seam_parallel;
        totals.seam_outline += s.seam_outline;
        totals.no_room += s.no_room;
        contour_len += r.contour_length_px;
        stroke_tex += r.stroke_texels;
        contour_tex += r.contour_texels;
        gate.accepted += r.rest_gate.accepted;
        gate.clamped += r.rest_gate.clamped;
        gate.rejected += r.rest_gate.rejected;
        gate.held_out += r.rest_gate.held_out;
        parts.push(json!({ "id": id, "report": r, "seamParallelTexels": seam_par }));
    }
    let check_poses: Vec<seams::EnvelopePose> = env
        .iter()
        .filter(|p| {
            LINEART_RENDER_POSES.contains(&p.name.as_str())
                || p.name.starts_with("hairSign:")
                || p.name.starts_with("opp:")
        })
        .cloned()
        .collect();
    let tex_refs: Vec<&[u8]> = layer_textures.iter().map(|t| t.as_slice()).collect();
    let doubles = lineart::posed_double_lines(
        ids,
        &proposed.masks,
        &fills,
        meshes,
        &reps,
        &tex_refs,
        landmarks,
        wu,
        hu,
        &check_poses,
    );
    // Line layer diagnostic: dimmed artwork, visible line art (blue), kept T15 strokes (red) and
    // contours (green), gate-rejected T15 texels (yellow).
    {
        let mut ov: Vec<u8> = img
            .as_raw()
            .chunks(4)
            .flat_map(|c| {
                let a = c[3] as f64 / 255.0;
                let mix = |v: u8| (255.0 - (255.0 - v as f64 * 0.35 - 255.0 * 0.65) * a) as u8;
                [mix(c[0]), mix(c[1]), mix(c[2]), 255]
            })
            .collect();
        let owner = lineart::rest_owner(&proposed.masks, n);
        for (p, r) in reps.iter().enumerate() {
            if r.is_none() {
                continue;
            }
            let ev: Vec<bool> = (0..n)
                .map(|i| proposed.masks[p][i] >= lineart::EVIDENCE_ALPHA_MIN || lineart::is_overlay_texel(&fills[p], &owner, p, i))
                .collect();
            let tex = layer_textures[p].as_slice();
            let (line, _) = lineart::detect_line_pixels(&|g| lineart::lstar_px(tex, g), &ev, wu, hu);
            for (i, &b) in line.iter().enumerate() {
                if b {
                    ov[i * 4..i * 4 + 3].copy_from_slice(&[40, 90, 255]);
                }
            }
        }
        for r in reps.iter().flatten() {
            for t in &r.texels {
                if t.alpha < 0.5 {
                    continue;
                }
                let c = if !t.kept {
                    [255, 220, 0]
                } else if t.stroke >= lineart::CONTOUR_ID_BASE {
                    [0, 200, 60]
                } else {
                    [255, 30, 30]
                };
                let i = t.i as usize;
                ov[i * 4..i * 4 + 3].copy_from_slice(&c);
            }
        }
        image::RgbaImage::from_raw(w, h, ov)
            .ok_or("line layer size mismatch")?
            .save(dir.join("line_layer.png"))?;
    }
    let textures: Vec<&[u8]> = layer_textures.iter().map(|t| t.as_slice()).collect();
    let mut renders = Vec::new();
    for pe in env.iter().filter(|p| LINEART_RENDER_POSES.contains(&p.name.as_str())) {
        let (rgba, _) = eye_seal::render_pose_rgba(ids, &proposed.masks, &fills, meshes, &textures, landmarks, wu, hu, &pe.pose);
        let file = format!(
            "pose_{}.png",
            pe.name.replace([':', '/', '='], "_").replace('+', "p").replace('-', "m")
        );
        image::RgbaImage::from_raw(w, h, rgba)
            .ok_or("lineart render size mismatch")?
            .save(dir.join(&file))?;
        renders.push(json!({ "pose": pe.name, "file": format!("lineart/{file}") }));
    }
    // Skipped-stroke crops (≤ 2 per reason per part, ≤ 24 total).
    let mut crops = Vec::new();
    for (p, id) in ids.iter().enumerate() {
        let Some(r) = reps[p] else { continue };
        let mut per_reason: std::collections::HashMap<&str, usize> = Default::default();
        for f in &r.semantic {
            let c = per_reason.entry(f.reason.as_str()).or_insert(0);
            if *c >= 2 || crops.len() >= 24 {
                continue;
            }
            *c += 1;
            const R: i64 = 20;
            const S: u32 = 4;
            let side = (2 * R) as u32;
            let mut im = image::RgbaImage::new(side * S * 2 + 4, side * S);
            let tex = &layer_textures[p];
            for dy in 0..2 * R {
                for dx in 0..2 * R {
                    let x = f.x as i64 - R + dx;
                    let y = f.y as i64 - R + dy;
                    let inside = x >= 0 && y >= 0 && x < w as i64 && y < h as i64;
                    let k = if inside { (y as usize * wu + x as usize) * 4 } else { 0 };
                    let orig = if inside { [img.as_raw()[k], img.as_raw()[k + 1], img.as_raw()[k + 2], 255] } else { [0, 0, 0, 255] };
                    let t = if inside && tex[k + 3] > 0 {
                        [tex[k], tex[k + 1], tex[k + 2], 255]
                    } else if (dx / 4 + dy / 4) % 2 == 0 {
                        [200, 200, 200, 255]
                    } else {
                        [150, 150, 150, 255]
                    };
                    let mark = dx == R && dy == R;
                    for sy in 0..S {
                        for sx in 0..S {
                            let (px, py) = (dx as u32 * S + sx, dy as u32 * S + sy);
                            let red = image::Rgba([255, 0, 0, 255]);
                            im.put_pixel(px, py, if mark { red } else { image::Rgba(orig) });
                            im.put_pixel(px + side * S + 4, py, if mark { red } else { image::Rgba(t) });
                        }
                    }
                }
            }
            let file = format!("skip_{}_{}_{}.png", id, f.reason, c);
            im.save(dir.join(&file))?;
            crops.push(json!({ "id": id, "reason": f.reason, "x": f.x, "y": f.y, "file": format!("lineart/{file}") }));
        }
    }
    Ok(json!({
        "version": lineart::LINEART_VERSION,
        "enabled": lineart::lineart_enabled(),
        "strokes": totals,
        "strokesSkipped": totals.skipped(),
        "contourLengthPx": contour_len,
        "strokeTexels": stroke_tex,
        "contourTexels": contour_tex,
        "restGate": gate,
        "seamParallelTexels": seam_par_total,
        "doubleLines": doubles,
        "renders": renders,
        "skippedCrops": crops,
        "parts": parts,
        "elapsedMs": t0.elapsed().as_millis(),
    }))
}

fn run_pipeline(
    input: &PipelineInput,
    proportion: Option<String>,
    stage: Option<String>,
    coupled: bool,
    legacy_eye: bool,
    eval: &EvalOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    // Layer input: the analysis image (detection, landmarks) is the composite of the layers.
    let (img, layers) = match &input.source {
        PartSource::Image { path, .. } => (image::open(path)?.into_rgba8(), None),
        PartSource::Layers(dir) => {
            let set = layer_input::load_layer_dir(dir)?;
            let comp = layer_input::composite_layers(&set);
            let img = image::RgbaImage::from_raw(set.width as u32, set.height as u32, comp)
                .ok_or("layer composite size mismatch")?;
            (img, Some(set))
        }
    };
    let (w, h) = img.dimensions();
    core::check_size(w as usize, h as usize)?;

    let state = AnalysisState::new(None)?;
    let mut points_28 = Vec::new();
    let mut face_bbox: Option<[f64; 4]> = None;
    let mut detection_source = "cli-json";

    let landmarks = if let Some(path) = input.landmarks {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        let mut auto_lm = None;
        {
            let mut reg = state
                .registry
                .lock()
                .map_err(|_| "Model registry lock poisoned")?;
            if reg.resolve_path(ModelRole::FaceDetector).is_some()
                && reg.resolve_path(ModelRole::Landmark28).is_some()
            {
                if let Ok(det) = detect_faces(
                    &mut reg,
                    img.as_raw(),
                    w as usize,
                    h as usize,
                    DEFAULT_FACE_SCORE_THRESHOLD,
                ) {
                    if let Some(top) = det.candidates.first() {
                        if let Ok(lm_out) = detect_landmarks(
                            &mut reg,
                            img.as_raw(),
                            w as usize,
                            h as usize,
                            top.bbox,
                            top.crop_bbox,
                        ) {
                            points_28 = lm_out.points_28;
                            face_bbox = Some(top.bbox);
                            auto_lm = Some(lm_out.rig_landmarks);
                            detection_source = "onnx-auto";
                        }
                    }
                }
            }
        }
        if let Some(lm) = auto_lm {
            lm
        } else {
            detection_source = "default-ratio";
            core::Landmarks {
                eye_l: core::Point {
                    x: w as f64 * 0.61,
                    y: h as f64 * 0.31,
                },
                eye_r: core::Point {
                    x: w as f64 * 0.39,
                    y: h as f64 * 0.31,
                },
                mouth: core::Point {
                    x: w as f64 * 0.5,
                    y: h as f64 * 0.415,
                },
                chin: core::Point {
                    x: w as f64 * 0.5,
                    y: h as f64 * 0.48,
                },
                crown: core::Point {
                    x: w as f64 * 0.5,
                    y: h as f64 * 0.08,
                },
                eye_frames: None,
            }
        }
    };
    landmarks.validate(w as usize, h as usize)?;
    let output = PathBuf::from(input.output);
    std::fs::create_dir_all(output.join("parts"))?;

    let style_overrides = proportion.as_ref().map(|mode| TemplateStyleSelection {
        proportion_mode: Some(mode.clone()),
        face_style: None,
        hair_front_style: None,
        hair_side_style: None,
        hair_back_style: None,
        eye_style: None,
        mouth_style: None,
        clothes_style: None,
    });
    let propose = || match &input.source {
        PartSource::Image { model, .. } => propose_parts_auto_with_template_stage(
            &state,
            img.as_raw(),
            w as usize,
            h as usize,
            &landmarks,
            &points_28,
            face_bbox,
            32.0,
            *model,
            "detail",
            true,
            true,
            style_overrides.as_ref(),
            stage.as_deref(),
        ),
        PartSource::Layers(_) => Err("layer input has no re-runnable proposal".to_string()),
    };
    let (proposed, layer_report) = match &layers {
        None => (propose()?, None),
        Some(set) => {
            let (p, r) = layer_input::proposal_from_layers(set, img.as_raw())?;
            img.save(output.join("composite.png"))?;
            (p, Some(r))
        }
    };
    // T10: eye layers deform with the aperture lines of the proposed visible masks (every motion
    // preview, seam, coverage and exposure computation below uses these landmarks).
    let landmarks = landmarks.with_eye_frames(&proposed.part_ids, &proposed.masks, w as usize, h as usize);

    let mut meshes = Vec::new();
    let mut mesh_quality = Vec::new();
    let mut part_coverage = Vec::new();
    let mut composite = image::RgbaImage::new(w, h);
    let mut effective_composite = image::RgbaImage::new(w, h);
    let mut fill_only_composite = image::RgbaImage::new(w, h);
    let motion_poses = [
        (
            "motion_preview.png",
            core::Pose {
                angle_x: 10.0,
                eye_l: 0.0,
                eye_r: 1.0,
                eye_ball_x: 0.75,
                mouth: 0.90,
                hair: 0.65,
                hair_side: 0.75,
                hair_back: 0.5,
                ..core::Pose::default()
            },
        ),
        (
            "motion_blink.png",
            core::Pose {
                eye_l: 0.0,
                eye_r: 0.0,
                ..core::Pose::default()
            },
        ),
        (
            "motion_mouth.png",
            core::Pose {
                mouth: 0.75,
                ..core::Pose::default()
            },
        ),
        (
            "motion_turn.png",
            core::Pose {
                angle_x: 10.0,
                angle_y: -8.0,
                angle_z: 6.0,
                eye_ball_x: 0.5,
                eye_ball_y: -0.3,
                hair: 0.55,
                hair_side: 0.65,
                hair_back: 0.4,
                ..core::Pose::default()
            },
        ),
    ];
    let mut motion_buffers: Vec<image::RgbaImage> =
        (0..motion_poses.len()).map(|_| image::RgbaImage::new(w, h)).collect();
    let mut part_meshes: Vec<core::Mesh> = Vec::with_capacity(proposed.part_ids.len());
    let mut layer_textures: Vec<Vec<u8>> = Vec::with_capacity(proposed.part_ids.len());

    for idx in 0..proposed.part_ids.len() {
        let id = &proposed.part_ids[idx];
        let mask = &proposed.masks[idx];
        let fill = proposed.fill_masks.get(idx).and_then(|o| o.as_deref());
        let fill_kind = proposed.fill_kinds.get(idx).map(|s| s.as_str());
        let fill_region = proposed.fill_regions.get(idx).and_then(|r| r.as_ref());
        let effective = core::effective_part_mask(mask, fill);
        let tex_src = decode_part_rgba(
            proposed.textures.get(idx).and_then(|o| o.as_deref()),
            &img,
        );

        let mut vis_part = tex_src.clone();
        let mut eff_part = tex_src.clone();
        let mut fill_part = tex_src;
        let mut visible_px = 0usize;
        let mut fill_px = 0usize;
        for (i, pixel) in vis_part.pixels_mut().enumerate() {
            pixel[3] = mask[i];
            if mask[i] > 0 {
                visible_px += 1;
                if layers.is_none() {
                    composite.as_mut()[i * 4..i * 4 + 4].copy_from_slice(&pixel.0);
                }
            }
        }
        if layers.is_some() {
            // Layer visible masks are uncovered alpha (parts can share an anti-aliased edge
            // pixel), so they stack with `normal` over instead of one owner per pixel.
            alpha_blend_layer(&mut composite, &vis_part);
        }
        for (i, pixel) in eff_part.pixels_mut().enumerate() {
            pixel[3] = effective[i];
        }
        for (i, pixel) in fill_part.pixels_mut().enumerate() {
            let f = fill.map_or(0, |fm| fm[i]);
            pixel[3] = f;
            if f > 0 {
                fill_px += 1;
            }
        }
        alpha_blend_layer(&mut effective_composite, &eff_part);
        alpha_blend_layer(&mut fill_only_composite, &fill_part);

        eff_part.save(output.join("parts").join(format!("{id}.png")))?;
        layer_textures.push(eff_part.as_raw().clone());

        let mesh = core::art_mesh_for_part(&effective, w as usize, h as usize, 14, id, &landmarks);
        let quality = core::check_mesh_quality(&mesh, &effective, w as usize, h as usize, id, &landmarks);
        for (buf, (_, pose)) in motion_buffers.iter_mut().zip(motion_poses.iter()) {
            draw_deformed_part(buf, &eff_part, &mesh, id, &landmarks, pose);
        }
        meshes.push(json!({"id": id, "mesh": mesh}));
        part_meshes.push(mesh);
        mesh_quality.push(quality);
        part_coverage.push(json!({
            "id": id,
            "visiblePixels": visible_px,
            "fillPixels": fill_px,
            "fillKind": fill_kind,
            "fillRegion": fill_region,
        }));
    }
    composite.save(output.join("foreground.png"))?;
    effective_composite.save(output.join("effective_foreground.png"))?;
    // Rest-pose visible owner (layer index + 1, 0 = none; 16-bit) — the partition, independent of
    // overlay `fillMask` texels that sit above a lower layer (T9).
    {
        let mut owner_img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(w, h);
        for (li, m) in proposed.masks.iter().enumerate() {
            for (i, &a) in m.iter().enumerate() {
                if a > 0 {
                    owner_img.as_mut()[i] = li as u16 + 1;
                }
            }
        }
        owner_img.save(output.join("owner.png"))?;
    }
    fill_only_composite.save(output.join("fill_only.png"))?;
    for (buf, (filename, _)) in motion_buffers.iter().zip(motion_poses.iter()) {
        buf.save(output.join(filename))?;
    }
    // T13b: seam contracts with coupling, numeric requirements, coverage, and the exposure map over
    // the shared pose envelope, using the same ArtMeshes as the motion previews.
    let (wu, hu) = (w as usize, h as usize);
    let n = wu * hu;
    let fill_full: Vec<Vec<u8>> = (0..proposed.part_ids.len())
        .map(|i| {
            proposed
                .fill_masks
                .get(i)
                .and_then(|o| o.clone())
                .unwrap_or_else(|| vec![0u8; n])
        })
        .collect();
    let fg_alpha: Vec<u8> = (0..n)
        .map(|i| proposed.masks.iter().map(|m| m[i]).max().unwrap_or(0))
        .collect();
    let roles: Vec<core::PartRole> = proposed
        .part_ids
        .iter()
        .map(|id| core::infer_part_role(id, None))
        .collect();
    let env = &seams::pose_envelope().poses;
    let mut graph = seams::extract_seam_graph(&proposed.part_ids, &roles, &proposed.masks, wu, hu);
    seams::annotate_seam_couplings(&mut graph, &landmarks);
    let requirements = seams::hidden_area_seam_requirements(
        &graph,
        &proposed.masks,
        &fg_alpha,
        &landmarks,
        wu,
        hu,
        env,
        seams::DEFAULT_MAX_SAMPLES_PER_SEAM,
    );
    let coverage = seams::check_seam_coverage(
        &graph,
        &proposed.part_ids,
        &proposed.masks,
        &fill_full,
        &fg_alpha,
        &landmarks,
        wu,
        hu,
        env,
        2,
    );
    let exposure = exposure::compute_exposure_map_with(
        &proposed.part_ids,
        &proposed.masks,
        &fill_full,
        &part_meshes,
        wu,
        hu,
        env,
        |p, id, pose| core::deform(p, id, &landmarks, pose),
    );
    let heat = exposure::exposure_heatmap_rgba(&exposure, Some(img.as_raw()));
    image::RgbaImage::from_raw(w, h, heat)
        .ok_or("exposure heatmap size mismatch")?
        .save(output.join("exposure.png"))?;
    // T10: eye seal (leaks / lid seam gaps / lid-line breaks) over the expression × gaze poses,
    // plus hard renders of the seal poses for before/after comparison (`eye_seal/*.png`).
    let seal_poses = eye_seal::eye_seal_poses(env);
    let eye_seal_report = eye_seal::check_eye_seal_with(
        &proposed.part_ids,
        &proposed.masks,
        &fill_full,
        &part_meshes,
        &landmarks,
        wu,
        hu,
        &seal_poses,
    );
    {
        std::fs::create_dir_all(output.join("eye_seal"))?;
        let textures: Vec<&[u8]> = layer_textures.iter().map(|t| t.as_slice()).collect();
        let rest = seams::EnvelopePose {
            name: "rest".into(),
            group: "rest".into(),
            pose: core::Pose::default(),
        };
        for pe in std::iter::once(&rest).chain(seal_poses.iter().filter(|p| {
            p.name.ends_with("/gaze:00") || p.name == "seal:sad/gaze:++" || p.name == "seal:closed/gaze:--"
        })) {
            let (rgba, top) = eye_seal::render_pose_rgba(
                &proposed.part_ids,
                &proposed.masks,
                &fill_full,
                &part_meshes,
                &textures,
                &landmarks,
                wu,
                hu,
                &pe.pose,
            );
            let file = pe.name.replace([':', '/'], "_").replace("+", "p").replace("-", "m");
            image::RgbaImage::from_raw(w, h, rgba)
                .ok_or("eye seal render size mismatch")?
                .save(output.join("eye_seal").join(format!("{file}.png")))?;
            // Posed top-layer owner (layer index + 1, 0 = none; 16-bit) for defect diagnosis.
            let owner: Vec<u16> = top.iter().map(|&t| (t + 1) as u16).collect();
            image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(w, h, owner)
                .ok_or("eye seal owner size mismatch")?
                .save(output.join("eye_seal").join(format!("{file}_owner.png")))?;
        }
    }
    let seam_rows: Vec<serde_json::Value> = requirements
        .iter()
        .map(|r| {
            let cov = coverage
                .seams
                .iter()
                .find(|s| s.upper_id == r.upper_id && s.lower_id == r.lower_id);
            json!({
                "upper": r.upper_id,
                "lower": r.lower_id,
                "coupling": graph.find(&r.upper_id, &r.lower_id).and_then(|s| s.coupling),
                "boundaryPoints": r.boundary_points,
                "maxRelativeDisplacementPx": r.max_relative_displacement_px,
                "maxExposureDepthPx": r.max_exposure_depth_px,
                "requiredUnderlapPx": r.required_underlap_px,
                "coupledUnderlapPx": r.coupled_underlap_px,
                "uncoupledUnderlapPx": r.uncoupled_underlap_px,
                "uncoupledMaxRelativeDisplacementPx": r.uncoupled_max_relative_displacement_px,
                "adoptedUnderlap": r.adopted,
                "legacyEyeUnderlapPx": r.legacy_eye_underlap_px,
                "worstPose": r.worst_pose,
                "lowerMisses": cov.map_or(0, |c| c.lower_misses),
                "holes": cov.map_or(0, |c| c.holes),
                "maxMissDepthPx": cov.map(|c| c.max_miss_depth_px),
                "maxHoleDepthPx": cov.map(|c| c.max_hole_depth_px),
                "holePoseGroups": cov.map(|c| &c.hole_pose_groups),
                "worstHolePose": cov.and_then(|c| c.worst_hole_pose.clone()),
                "worstHolePoseHoles": cov.map_or(0, |c| c.worst_hole_pose_holes),
            })
        })
        .collect();
    let model_hash = proposed
        .model_hashes
        .get("part_decoder")
        .or_else(|| proposed.model_hashes.get("background"))
        .cloned();
    let mut stats = json!({
        "width": w,
        "height": h,
        "parts": proposed.masks.len(),
        "proportionOverride": proportion,
        "landmarks": landmarks,
        "landmarkSource": detection_source,
        "provider": proposed.provider,
        "postprocessVersion": proposed.postprocess_version,
        "styleMatch": proposed.style_match,
        "modelHash": model_hash,
        "modelHashes": proposed.model_hashes,
        "partsMeta": proposed.metas,
        "partCoverage": part_coverage,
        "meshQuality": mesh_quality,
        "seamCoupling": coupled,
        "exposure": exposure,
        "exposureImage": "exposure.png",
        "exposureLegend": "rest-space hidden texels exposed in the pose envelope: yellow (rare) -> red (every pose); faint blue = hidden but never exposed (kept for PSD continuity); magenta = posed-space hole (uncovered gap not present at rest); grey = dimmed artwork",
        "seamCoverage": {
            "poses": coverage.poses,
            "checks": coverage.checks,
            "lowerMisses": coverage.lower_misses,
            "holes": coverage.holes,
            "violations": coverage.violations.iter().take(16).collect::<Vec<_>>(),
        },
        "seams": seam_rows,
        "eyeSeal": eye_seal_report,
        "eyeRig": { "legacy": legacy_eye, "eyeFrames": landmarks.eye_frames },
        "elapsedMs": start.elapsed().as_millis(),
        "engine": "Rust / ort / delaunator / glam"
    });
    if input.pose_renders {
        stats["poseRenders"] = render_pose_set(
            &output,
            &proposed.part_ids,
            &part_meshes,
            &layer_textures,
            &landmarks,
            w,
            h,
            env,
        )?;
    }
    stats["lineart"] = lineart_outputs(
        &output,
        &proposed,
        &part_meshes,
        &layer_textures,
        &landmarks,
        &img,
        env,
    )?;
    if let Some(report) = &layer_report {
        stats["layerInput"] = serde_json::to_value(report)?;
        stats["analysisImage"] = json!("composite.png");
    }
    stats["elapsedMs"] = json!(start.elapsed().as_millis());
    if eval.any() {
        // T14 (evaluation only): nothing below changes the product outputs computed above.
        stats["fidelity"] = run_fidelity(
            eval,
            &output,
            &proposed,
            &propose,
            &requirements,
            &exposure,
            &img,
        )?;
    }
    std::fs::write(output.join("rig.json"), serde_json::to_vec_pretty(&meshes)?)?;
    std::fs::write(
        output.join("stats.json"),
        serde_json::to_vec_pretty(&stats)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}
