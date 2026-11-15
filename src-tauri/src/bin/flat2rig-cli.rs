use base64::{engine::general_purpose::STANDARD, Engine};
use flat2rig_lib::analysis::{
    detector::{detect_faces, DEFAULT_FACE_SCORE_THRESHOLD},
    landmarks::detect_landmarks,
    models::{ModelRegistry, ModelRole},
    parts::{propose_parts_auto_with_template, propose_parts_with_sam_with_options},
    state::AnalysisState,
    templates::TemplateStyleSelection,
};
use flat2rig_lib::core;
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

#[inline]
fn blend_straight_pixel(dst: &mut [u8], src: [u8; 4]) {
    let sa = src[3] as f32 / 255.0;
    if sa <= 0.0 {
        return;
    }
    let da = dst[3] as f32 / 255.0;
    let inv = 1.0 - sa;
    let out_a = sa + da * inv;
    if out_a <= 1e-6 {
        return;
    }
    let d_w = da * inv;
    for c in 0..3 {
        let blended = (src[c] as f32 * sa + dst[c] as f32 * d_w) / out_a;
        dst[c] = blended.round().clamp(0.0, 255.0) as u8;
    }
    dst[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

fn alpha_blend_layer(dst: &mut image::RgbaImage, src: &image::RgbaImage) {
    for (dp, sp) in dst.pixels_mut().zip(src.pixels()) {
        blend_straight_pixel(&mut dp.0, sp.0);
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
                blend_straight_pixel(
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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|s| s.as_str()) == Some("--analyze") {
        return run_analyze(&args[1..]);
    }
    let mut proportion: Option<String> = None;
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
        } else if arg.starts_with("--") {
            return Err(format!("Unknown flag: {arg}").into());
        } else {
            positional.push(arg);
        }
        i += 1;
    }
    if positional.len() < 2 {
        return Err(
            "Usage: flat2rig-cli [--proportion auto|ld|sd] INPUT.png OUTPUT_DIR [MODEL.onnx] [LANDMARKS.json] | --analyze [--full] INPUT.png [OUTPUT.json]".into(),
        );
    }
    let start = std::time::Instant::now();
    let img = image::open(positional[0])?.into_rgba8();
    let (w, h) = img.dimensions();
    core::check_size(w as usize, h as usize)?;

    let state = AnalysisState::new(None)?;
    let mut points_28 = Vec::new();
    let mut face_bbox: Option<[f64; 4]> = None;
    let mut detection_source = "cli-json";

    let landmarks = if let Some(path) = positional.get(3) {
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
    let output = PathBuf::from(positional[1]);
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
    let proposed = propose_parts_auto_with_template(
        &state,
        img.as_raw(),
        w as usize,
        h as usize,
        &landmarks,
        &points_28,
        face_bbox,
        32.0,
        positional.get(2).copied(),
        "detail",
        true,
        true,
        style_overrides.as_ref(),
    )?;

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

    for idx in 0..proposed.part_ids.len() {
        let id = &proposed.part_ids[idx];
        let mask = &proposed.masks[idx];
        let fill = proposed.fill_masks.get(idx).and_then(|o| o.as_deref());
        let fill_kind = proposed.fill_kinds.get(idx).map(|s| s.as_str());
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
                composite.as_mut()[i * 4..i * 4 + 4].copy_from_slice(&pixel.0);
            }
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

        let mesh = core::art_mesh_for_part(&effective, w as usize, h as usize, 14, id, &landmarks);
        let quality = core::check_mesh_quality(&mesh, &effective, w as usize, h as usize, id, &landmarks);
        for (buf, (_, pose)) in motion_buffers.iter_mut().zip(motion_poses.iter()) {
            draw_deformed_part(buf, &eff_part, &mesh, id, &landmarks, pose);
        }
        meshes.push(json!({"id": id, "mesh": mesh}));
        mesh_quality.push(quality);
        part_coverage.push(json!({
            "id": id,
            "visiblePixels": visible_px,
            "fillPixels": fill_px,
            "fillKind": fill_kind,
        }));
    }
    composite.save(output.join("foreground.png"))?;
    effective_composite.save(output.join("effective_foreground.png"))?;
    fill_only_composite.save(output.join("fill_only.png"))?;
    for (buf, (filename, _)) in motion_buffers.iter().zip(motion_poses.iter()) {
        buf.save(output.join(filename))?;
    }
    let model_hash = proposed
        .model_hashes
        .get("part_decoder")
        .or_else(|| proposed.model_hashes.get("background"))
        .cloned();
    let stats = json!({
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
