use flat2rig_lib::analysis::{
    detector::{detect_faces, DEFAULT_FACE_SCORE_THRESHOLD},
    landmarks::detect_landmarks,
    models::{ModelRegistry, ModelRole},
    parts::{propose_parts_auto_with_options, propose_parts_with_sam_with_options},
    state::AnalysisState,
};
use flat2rig_lib::core;
use serde_json::json;
use std::path::PathBuf;

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
                    let mesh =
                        core::mesh_for_part(mask, w as usize, h as usize, 14, &lm.rig_landmarks);
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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|s| s.as_str()) == Some("--analyze") {
        return run_analyze(&args[1..]);
    }
    if args.len() < 2 {
        return Err(
            "Usage: flat2rig-cli INPUT.png OUTPUT_DIR [MODEL.onnx] [LANDMARKS.json] | --analyze [--full] INPUT.png [OUTPUT.json]".into(),
        );
    }
    let start = std::time::Instant::now();
    let img = image::open(&args[0])?.into_rgba8();
    let (w, h) = img.dimensions();
    core::check_size(w as usize, h as usize)?;

    let state = AnalysisState::new(None)?;
    let mut points_28 = Vec::new();
    let mut face_bbox: Option<[f64; 4]> = None;
    let mut detection_source = "cli-json";

    let landmarks = if let Some(path) = args.get(3) {
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
            }
        }
    };
    landmarks.validate(w as usize, h as usize)?;
    let output = PathBuf::from(&args[1]);
    std::fs::create_dir_all(output.join("parts"))?;

    let proposed = propose_parts_auto_with_options(
        &state,
        img.as_raw(),
        w as usize,
        h as usize,
        &landmarks,
        &points_28,
        face_bbox,
        32.0,
        args.get(2).map(|s| s.as_str()),
        "detail",
        true,
    )?;

    let mut meshes = Vec::new();
    let mut mesh_quality = Vec::new();
    let mut composite = image::RgbaImage::new(w, h);
    for (id, mask) in proposed.part_ids.iter().zip(&proposed.masks) {
        let mut part = img.clone();
        for (i, pixel) in part.pixels_mut().enumerate() {
            pixel[3] = mask[i];
            if mask[i] > 0 {
                composite.as_mut()[i * 4..i * 4 + 4].copy_from_slice(&pixel.0);
            }
        }
        part.save(output.join("parts").join(format!("{id}.png")))?;
        let mesh = core::mesh_for_part(mask, w as usize, h as usize, 14, &landmarks);
        let quality = core::check_mesh_quality(&mesh, mask, w as usize, h as usize, id, &landmarks);
        meshes.push(json!({"id": id, "mesh": mesh}));
        mesh_quality.push(quality);
    }
    composite.save(output.join("foreground.png"))?;
    let model_hash = proposed
        .model_hashes
        .get("part_decoder")
        .or_else(|| proposed.model_hashes.get("background"))
        .cloned();
    let stats = json!({
        "width": w,
        "height": h,
        "parts": proposed.masks.len(),
        "landmarks": landmarks,
        "landmarkSource": detection_source,
        "provider": proposed.provider,
        "modelHash": model_hash,
        "modelHashes": proposed.model_hashes,
        "partsMeta": proposed.metas,
        "meshQuality": mesh_quality,
        "elapsedMs": start.elapsed().as_millis(),
        "engine": "Rust / ort / delaunator / glam"
    });
    std::fs::write(output.join("rig.json"), serde_json::to_vec_pretty(&meshes)?)?;
    std::fs::write(
        output.join("stats.json"),
        serde_json::to_vec_pretty(&stats)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}
