pub mod analysis;
pub mod core;
pub mod inference;
pub mod rig;
pub mod updater;
use analysis::{
    detector::{detect_faces, FaceCandidate, DEFAULT_FACE_SCORE_THRESHOLD},
    landmarks::{detect_landmarks, Landmark28Point, RigLandmarkProvenance},
    models::{atomic_write_bytes, download_and_install_model, ModelRole, ModelStatusEntry},
    parts::{
        propose_parts_auto, propose_parts_auto_with_template_stage, refine_part_with_prompts,
        PartProposalMeta,
    },
    segmentation::PromptInput,
    state::{hash_rgba_bytes, AnalysisState},
    templates::{StyleMatchReport, TemplateStyleSelection},
};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use image::ImageEncoder;
use std::sync::atomic::Ordering;
use tauri::Manager;
use tauri_plugin_dialog::DialogExt;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceAnalysisResponse {
    job_id: u64,
    image_hash: String,
    status: String,
    message: String,
    candidates: Vec<FaceCandidate>,
    selected_face_id: Option<usize>,
    best_raw_score: f64,
    points_28: Vec<Landmark28Point>,
    rig_landmarks: Option<core::Landmarks>,
    rig_provenance: Option<RigLandmarkProvenance>,
    model_hashes: HashMap<String, String>,
    postprocess_versions: HashMap<String, String>,
}

#[tauri::command]
async fn get_model_status(app: tauri::AppHandle) -> Result<Vec<ModelStatusEntry>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AnalysisState>();
        let mut reg = state
            .registry
            .lock()
            .map_err(|_| "Model registry lock poisoned".to_string())?;
        Ok(reg.status_all())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn set_model_path(
    app: tauri::AppHandle,
    role: String,
    path: Option<String>,
) -> Result<Vec<ModelStatusEntry>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let parsed_role =
            ModelRole::parse_id(&role).ok_or_else(|| format!("Unknown model role: {role}"))?;
        let state = app.state::<AnalysisState>();
        let status = {
            let mut reg = state
                .registry
                .lock()
                .map_err(|_| "Model registry lock poisoned".to_string())?;
            if let Some(ref p) = path {
                let pb = PathBuf::from(p);
                if pb.extension().and_then(|v| v.to_str()) != Some("onnx") {
                    return Err("Select an .onnx model file".into());
                }
                let _ = reg.file_sha256(&pb)?;
                reg.set_override(parsed_role, Some(pb));
            } else {
                reg.set_override(parsed_role, None);
            }
            reg.status_all()
        };
        state.persist_config()?;
        Ok(status)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn cancel_analysis(app: tauri::AppHandle) {
    let state = app.state::<AnalysisState>();
    state.cancel_requested.store(true, Ordering::SeqCst);
}

#[tauri::command]
async fn download_model(
    app: tauri::AppHandle,
    role: String,
) -> Result<Vec<ModelStatusEntry>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let parsed_role =
            ModelRole::parse_id(&role).ok_or_else(|| format!("Unknown model role: {role}"))?;
        let state = app.state::<AnalysisState>();
        let dest_dir = state
            .models_dir
            .clone()
            .unwrap_or_else(|| std::env::current_dir().map(|d| d.join("models")).unwrap_or_else(|_| PathBuf::from("models")));
        let installed_path = download_and_install_model(&role, &dest_dir)?;
        let status = {
            let mut reg = state
                .registry
                .lock()
                .map_err(|_| "Model registry lock poisoned".to_string())?;
            reg.set_override(parsed_role, Some(installed_path));
            reg.status_all()
        };
        state.persist_config()?;
        Ok(status)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn analyze_face(
    app: tauri::AppHandle,
    rgba: String,
    width: usize,
    height: usize,
    face_index: Option<usize>,
    job_id: Option<u64>,
) -> Result<FaceAnalysisResponse, String> {
    let state = app.state::<AnalysisState>();
    let my_job = state.begin_job(job_id);
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AnalysisState>();
        let _guard = state
            .job_lock
            .lock()
            .map_err(|_| "Analysis job lock poisoned".to_string())?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }

        let n = core::check_size(width, height)?;
        let data = core::decode_bytes(&rgba, n * 4)?;
        let image_hash = hash_rgba_bytes(&data, width, height);

        let mut reg = state
            .registry
            .lock()
            .map_err(|_| "Model registry lock poisoned".to_string())?;
        let has_det = reg.resolve_path(ModelRole::FaceDetector).is_some();
        let has_lm = reg.resolve_path(ModelRole::Landmark28).is_some();
        if !has_det || !has_lm {
            return Ok(FaceAnalysisResponse {
                job_id: my_job,
                image_hash,
                status: "model-missing".into(),
                message: "얼굴 검출 또는 랜드마크 ONNX 모델이 연결되지 않았습니다. 모델 설정에서 파일을 연결하거나 수동 배치하세요.".into(),
                candidates: Vec::new(),
                selected_face_id: None,
                best_raw_score: 0.0,
                points_28: Vec::new(),
                rig_landmarks: None,
                rig_provenance: None,
                model_hashes: HashMap::new(),
                postprocess_versions: HashMap::new(),
            });
        }

        let det = detect_faces(&mut reg, &data, width, height, DEFAULT_FACE_SCORE_THRESHOLD)?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }

        if det.candidates.is_empty() {
            let mut hashes = HashMap::new();
            hashes.insert("face_detector".into(), det.model_sha256);
            return Ok(FaceAnalysisResponse {
                job_id: my_job,
                image_hash,
                status: "no-face".into(),
                message: format!(
                    "신뢰도 임계값(0.50) 이상의 얼굴을 찾지 못했습니다 (최고 후보 {:.2}). 수동으로 얼굴 영역과 눈·입 좌표를 배치하세요.",
                    det.best_raw_score
                ),
                candidates: Vec::new(),
                selected_face_id: None,
                best_raw_score: det.best_raw_score,
                points_28: Vec::new(),
                rig_landmarks: None,
                rig_provenance: None,
                model_hashes: hashes,
                postprocess_versions: HashMap::from([(
                    "face_detector".into(),
                    det.postprocess_version,
                )]),
            });
        }

        let chosen = face_index.unwrap_or(0).min(det.candidates.len() - 1);
        let top = &det.candidates[chosen];
        let lm = detect_landmarks(&mut reg, &data, width, height, top.bbox, top.crop_bbox)?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }

        let model_hashes = HashMap::from([
            ("face_detector".into(), det.model_sha256),
            ("landmark_28".into(), lm.model_sha256),
        ]);
        let postprocess_versions = HashMap::from([
            ("face_detector".into(), det.postprocess_version),
            ("landmark_28".into(), lm.postprocess_version),
        ]);
        let message = if det.candidates.len() > 1 {
            format!(
                "얼굴 후보 {}명 검출 (#{} 선택, 신뢰도 {:.0}%). crown 좌표는 기하 추정입니다.",
                det.candidates.len(),
                chosen + 1,
                top.score * 100.0
            )
        } else {
            format!(
                "얼굴·28점 랜드마크 추출 완료 (신뢰도 {:.0}%). crown 좌표는 기하 추정입니다.",
                top.score * 100.0
            )
        };

        Ok(FaceAnalysisResponse {
            job_id: my_job,
            image_hash,
            status: "detected".into(),
            message,
            candidates: det.candidates,
            selected_face_id: Some(chosen),
            best_raw_score: det.best_raw_score,
            points_28: lm.points_28,
            rig_landmarks: Some(lm.rig_landmarks),
            rig_provenance: Some(lm.rig_provenance),
            model_hashes,
            postprocess_versions,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn detect_landmarks_for_face(
    app: tauri::AppHandle,
    rgba: String,
    width: usize,
    height: usize,
    bbox: [f64; 4],
    crop_bbox: [f64; 4],
    selected_face_id: usize,
    job_id: Option<u64>,
) -> Result<FaceAnalysisResponse, String> {
    let state = app.state::<AnalysisState>();
    let my_job = state.begin_job(job_id);
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AnalysisState>();
        let _guard = state
            .job_lock
            .lock()
            .map_err(|_| "Analysis job lock poisoned".to_string())?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }

        let n = core::check_size(width, height)?;
        let data = core::decode_bytes(&rgba, n * 4)?;
        let image_hash = hash_rgba_bytes(&data, width, height);
        let mut reg = state
            .registry
            .lock()
            .map_err(|_| "Model registry lock poisoned".to_string())?;
        let lm = detect_landmarks(&mut reg, &data, width, height, bbox, crop_bbox)?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }

        Ok(FaceAnalysisResponse {
            job_id: my_job,
            image_hash,
            status: "detected".into(),
            message: format!("후보 #{} 기준 28점 랜드마크를 갱신했습니다.", selected_face_id + 1),
            candidates: Vec::new(),
            selected_face_id: Some(selected_face_id),
            best_raw_score: 0.0,
            points_28: lm.points_28,
            rig_landmarks: Some(lm.rig_landmarks),
            rig_provenance: Some(lm.rig_provenance),
            model_hashes: HashMap::from([("landmark_28".into(), lm.model_sha256)]),
            postprocess_versions: HashMap::from([("landmark_28".into(), lm.postprocess_version)]),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Separation {
    masks: Vec<String>,
    provider: String,
    model_hash: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProposePartsResponse {
    job_id: u64,
    part_ids: Vec<String>,
    masks: Vec<String>,
    fill_masks: Vec<Option<String>>,
    fill_kinds: Vec<String>,
    fill_regions: Vec<Option<analysis::region_fill::RegionReport>>,
    textures: Vec<Option<String>>,
    metas: Vec<PartProposalMeta>,
    provider: String,
    model_hashes: HashMap<String, String>,
    postprocess_version: String,
    image_hash: String,
    style_match: Option<StyleMatchReport>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RefineCandidateResponse {
    index: usize,
    iou_score: f32,
    mask: String,
    pixel_count: usize,
    diagnostics: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RefinePromptResponse {
    job_id: u64,
    part_id: String,
    best_index: usize,
    candidates: Vec<RefineCandidateResponse>,
    used_face_crop: bool,
    decoder_sha256: String,
    encoder_sha256: String,
}

#[tauri::command]
async fn generate_parts(
    app: tauri::AppHandle,
    rgba: String,
    width: usize,
    height: usize,
    landmarks: core::Landmarks,
    tolerance: f64,
    model_path: Option<String>,
) -> Result<Separation, String> {
    let state = app.state::<AnalysisState>();
    let my_job = state.begin_job(None);
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AnalysisState>();
        let _guard = state
            .job_lock
            .lock()
            .map_err(|_| "Analysis job lock poisoned".to_string())?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }
        let n = core::check_size(width, height)?;
        landmarks.validate(width, height)?;
        let data = core::decode_bytes(&rgba, n * 4)?;
        let proposed = propose_parts_auto(
            &state,
            &data,
            width,
            height,
            &landmarks,
            &[],
            None,
            tolerance.clamp(0., 180.),
            model_path.as_deref(),
            "detail",
        )?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }
        let model_hash = proposed
            .model_hashes
            .get("part_decoder")
            .or_else(|| proposed.model_hashes.get("background"))
            .cloned();
        Ok(Separation {
            masks: proposed.masks.iter().map(|m| STANDARD.encode(m)).collect(),
            provider: proposed.provider,
            model_hash,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn propose_parts(
    app: tauri::AppHandle,
    rgba: String,
    width: usize,
    height: usize,
    landmarks: core::Landmarks,
    points_28: Option<Vec<Landmark28Point>>,
    face_bbox: Option<[f64; 4]>,
    tolerance: f64,
    model_path: Option<String>,
    detail_mode: Option<String>,
    hair_split: Option<bool>,
    template_mode: Option<bool>,
    pipeline_stage: Option<String>,
    style_overrides: Option<TemplateStyleSelection>,
    job_id: Option<u64>,
) -> Result<ProposePartsResponse, String> {
    let state = app.state::<AnalysisState>();
    let my_job = state.begin_job(job_id);
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AnalysisState>();
        let _guard = state
            .job_lock
            .lock()
            .map_err(|_| "Analysis job lock poisoned".to_string())?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }
        let n = core::check_size(width, height)?;
        landmarks.validate(width, height)?;
        let data = core::decode_bytes(&rgba, n * 4)?;
        let pts28 = points_28.unwrap_or_default();
        let mode = detail_mode.unwrap_or_else(|| "detail".into());
        let hs = hair_split.unwrap_or(true);
        let use_tpl = template_mode.unwrap_or(hs && mode == "detail");
        let out = propose_parts_auto_with_template_stage(
            &state,
            &data,
            width,
            height,
            &landmarks,
            &pts28,
            face_bbox,
            tolerance.clamp(0., 180.),
            model_path.as_deref(),
            &mode,
            hs,
            use_tpl,
            style_overrides.as_ref(),
            pipeline_stage.as_deref(),
        )?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }
        Ok(ProposePartsResponse {
            job_id: my_job,
            part_ids: out.part_ids,
            masks: out.masks.iter().map(|m| STANDARD.encode(m)).collect(),
            fill_masks: out
                .fill_masks
                .iter()
                .map(|fm| fm.as_ref().map(|m| STANDARD.encode(m)))
                .collect(),
            fill_kinds: out.fill_kinds,
            fill_regions: out.fill_regions,
            textures: out.textures,
            metas: out.metas,
            provider: out.provider,
            model_hashes: out.model_hashes,
            postprocess_version: out.postprocess_version,
            image_hash: out.image_hash,
            style_match: out.style_match,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn refine_part_prompt(
    app: tauri::AppHandle,
    rgba: String,
    width: usize,
    height: usize,
    landmarks: core::Landmarks,
    part_id: String,
    prompts: Vec<PromptInput>,
    tolerance: f64,
    job_id: Option<u64>,
) -> Result<RefinePromptResponse, String> {
    let state = app.state::<AnalysisState>();
    let my_job = state.begin_job(job_id);
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AnalysisState>();
        let _guard = state
            .job_lock
            .lock()
            .map_err(|_| "Analysis job lock poisoned".to_string())?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }
        let n = core::check_size(width, height)?;
        landmarks.validate(width, height)?;
        let data = core::decode_bytes(&rgba, n * 4)?;
        let refined = refine_part_with_prompts(
            &state,
            &data,
            width,
            height,
            &landmarks,
            &part_id,
            &prompts,
            tolerance.clamp(0., 180.),
        )?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }
        Ok(RefinePromptResponse {
            job_id: my_job,
            part_id: refined.part_id,
            best_index: refined.best_index,
            candidates: refined
                .candidates
                .into_iter()
                .map(|c| RefineCandidateResponse {
                    index: c.index,
                    iou_score: c.iou_score,
                    mask: STANDARD.encode(&c.mask),
                    pixel_count: c.pixel_count,
                    diagnostics: c.diagnostics,
                })
                .collect(),
            used_face_crop: refined.used_face_crop,
            decoder_sha256: refined.decoder_sha256,
            encoder_sha256: refined.encoder_sha256,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeshInput {
    id: String,
    mask: String,
    #[serde(default, alias = "fill_mask")]
    fill_mask: Option<String>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RigPart {
    id: String,
    mesh: core::Mesh,
    samples: Vec<Sample>,
    quality: core::MeshQualityReport,
}
#[derive(Serialize)]
struct Sample {
    parameter: String,
    value: f64,
    vertices: Vec<core::Point>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExposureOverlay {
    /// Summary (per-layer exposed area, frequency, depth; holes) of `rig::exposure::ExposureMap`.
    summary: rig::exposure::ExposureMap,
    /// Seam coverage summary over the same envelope (`holes` / `lowerMisses`).
    seam_holes: usize,
    seam_lower_misses: usize,
    /// Transparent RGBA heatmap as a PNG data URL (canvas-sized).
    heatmap: String,
}

/// T13b: exposure map over the shared pose envelope for the advanced-screen heatmap overlay.
/// `parts` must be bottom-to-top in composite order (the same order the renderer draws).
#[tauri::command]
async fn compute_exposure_map(
    parts: Vec<MeshInput>,
    width: usize,
    height: usize,
    landmarks: core::Landmarks,
) -> Result<ExposureOverlay, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let n = core::check_size(width, height)?;
        landmarks.validate(width, height)?;
        if parts.len() > 64 {
            return Err("Too many parts".into());
        }
        let mut ids = Vec::with_capacity(parts.len());
        let mut visible = Vec::with_capacity(parts.len());
        let mut fills = Vec::with_capacity(parts.len());
        for part in parts {
            visible.push(core::decode_bytes(&part.mask, n)?);
            fills.push(match part.fill_mask {
                Some(ref f) => core::decode_bytes(f, n)?,
                None => vec![0u8; n],
            });
            ids.push(part.id);
        }
        let landmarks = landmarks.with_eye_frames(&ids, &visible, width, height);
        let env = &rig::seams::pose_envelope().poses;
        let summary = rig::exposure::compute_exposure_map(
            &ids, &visible, &fills, &landmarks, width, height, env,
        );
        let roles: Vec<core::PartRole> =
            ids.iter().map(|id| core::infer_part_role(id, None)).collect();
        let fg: Vec<u8> = (0..n)
            .map(|i| visible.iter().map(|m| m[i]).max().unwrap_or(0))
            .collect();
        let graph = rig::seams::extract_seam_graph(&ids, &roles, &visible, width, height);
        let coverage = rig::seams::check_seam_coverage(
            &graph, &ids, &visible, &fills, &fg, &landmarks, width, height, env, 2,
        );
        let rgba = rig::exposure::exposure_heatmap_rgba(&summary, None);
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(
                &rgba,
                width as u32,
                height as u32,
                image::ExtendedColorType::Rgba8,
            )
            .map_err(|e| format!("exposure heatmap PNG: {e}"))?;
        Ok(ExposureOverlay {
            summary,
            seam_holes: coverage.holes,
            seam_lower_misses: coverage.lower_misses,
            heatmap: format!("data:image/png;base64,{}", STANDARD.encode(png)),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn build_rig(
    parts: Vec<MeshInput>,
    width: usize,
    height: usize,
    landmarks: core::Landmarks,
) -> Result<Vec<RigPart>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let n = core::check_size(width, height)?;
        landmarks.validate(width, height)?;
        if parts.len() > 64 {
            return Err("Too many parts".into());
        }
        let mut result = Vec::new();
        let mut decoded = Vec::with_capacity(parts.len());
        for part in parts {
            let visible = core::decode_bytes(&part.mask, n)?;
            let fill = match part.fill_mask {
                Some(ref f) => Some(core::decode_bytes(f, n)?),
                None => None,
            };
            decoded.push((part, visible, fill));
        }
        // T10: eye layers deform with the aperture lines of all visible eye masks.
        let ids: Vec<String> = decoded.iter().map(|(p, _, _)| p.id.clone()).collect();
        let vis: Vec<Vec<u8>> = decoded.iter().map(|(_, v, _)| v.clone()).collect();
        let landmarks = landmarks.with_eye_frames(&ids, &vis, width, height);
        for (part, visible, fill) in decoded {
            let effective = core::effective_part_mask(&visible, fill.as_deref());
            let mesh = core::art_mesh_for_part(&effective, width, height, 14, &part.id, &landmarks);
            let quality =
                core::check_mesh_quality(&mesh, &effective, width, height, &part.id, &landmarks);
            let mut samples = Vec::new();
            for (parameter, values) in [
                ("angleX", vec![-10., 0., 10.]),
                ("angleY", vec![-10., 0., 10.]),
                ("angleZ", vec![-15., 0., 15.]),
                ("eyeL", vec![0., 1.]),
                ("eyeLSmile", vec![0., 1.]),
                ("eyeR", vec![0., 1.]),
                ("eyeRSmile", vec![0., 1.]),
                ("eyeBallX", vec![-1., 0., 1.]),
                ("eyeBallY", vec![-1., 0., 1.]),
                ("browLY", vec![-1., 0., 1.]),
                ("browLForm", vec![-1., 0., 1.]),
                ("browRY", vec![-1., 0., 1.]),
                ("browRForm", vec![-1., 0., 1.]),
                ("mouthForm", vec![-1., 0., 1.]),
                ("mouth", vec![0., 1.]),
                ("bodyAngleX", vec![-10., 0., 10.]),
                ("bodyAngleY", vec![-10., 0., 10.]),
                ("bodyAngleZ", vec![-10., 0., 10.]),
                ("breath", vec![0., 1.]),
                ("hair", vec![-1., 0., 1.]),
                ("hairSide", vec![-1., 0., 1.]),
                ("hairBack", vec![-1., 0., 1.]),
            ] {
                for value in values {
                    let mut pose = core::Pose::default();
                    match parameter {
                        "angleX" => pose.angle_x = value,
                        "angleY" => pose.angle_y = value,
                        "angleZ" => pose.angle_z = value,
                        "eyeL" => pose.eye_l = value,
                        "eyeLSmile" => pose.eye_l_smile = value,
                        "eyeR" => pose.eye_r = value,
                        "eyeRSmile" => pose.eye_r_smile = value,
                        "eyeBallX" => pose.eye_ball_x = value,
                        "eyeBallY" => pose.eye_ball_y = value,
                        "browLY" => pose.brow_l_y = value,
                        "browLForm" => pose.brow_l_form = value,
                        "browRY" => pose.brow_r_y = value,
                        "browRForm" => pose.brow_r_form = value,
                        "mouthForm" => pose.mouth_form = value,
                        "mouth" => pose.mouth = value,
                        "bodyAngleX" => pose.body_angle_x = value,
                        "bodyAngleY" => pose.body_angle_y = value,
                        "bodyAngleZ" => pose.body_angle_z = value,
                        "breath" => pose.breath = value,
                        "hair" => pose.hair = value,
                        "hairSide" => pose.hair_side = value,
                        _ => pose.hair_back = value,
                    };
                    samples.push(Sample {
                        parameter: parameter.into(),
                        value,
                        vertices: mesh
                            .vertices
                            .iter()
                            .map(|p| core::deform(*p, &part.id, &landmarks, &pose))
                            .collect(),
                    });
                }
            }
            result.push(RigPart {
                id: part.id,
                mesh,
                samples,
                quality,
            });
        }
        Ok(result)
    })
    .await
    .map_err(|e| e.to_string())?
}
#[derive(Serialize)]
struct FillResult {
    rgba: String,
    mask: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FillOcclusionResult {
    rgba: String,
    fill_mask: String,
    combined_mask: String,
    filled_pixels: usize,
    fill_kind: String,
    fill_region: Option<analysis::region_fill::RegionReport>,
    warning: Option<String>,
}

#[tauri::command]
async fn fill_part(
    rgba: String,
    mask: String,
    width: usize,
    height: usize,
    radius: usize,
) -> Result<FillResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let n = core::check_size(width, height)?;
        let data = core::decode_bytes(&rgba, n * 4)?;
        let mask = core::decode_bytes(&mask, n)?;
        let (data, mask) = core::fill(&data, &mask, width, height, radius);
        Ok(FillResult {
            rgba: STANDARD.encode(data),
            mask: STANDARD.encode(mask),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn fill_part_occlusion(
    rgba: String,
    visible_mask: String,
    width: usize,
    height: usize,
    radius: usize,
    part_id: Option<String>,
) -> Result<FillOcclusionResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let n = core::check_size(width, height)?;
        let data = core::decode_bytes(&rgba, n * 4)?;
        let visible = core::decode_bytes(&visible_mask, n)?;
        let out = core::fill_occlusion(&data, &visible, width, height, radius, part_id.as_deref());
        Ok(FillOcclusionResult {
            rgba: STANDARD.encode(out.rgba),
            fill_mask: STANDARD.encode(out.fill_mask),
            combined_mask: STANDARD.encode(out.combined_mask),
            filled_pixels: out.filled_pixels,
            fill_kind: out.fill_kind,
            fill_region: out.fill_region,
            warning: out.warning,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn save_artifact(
    app: tauri::AppHandle,
    name: String,
    bytes: Vec<u8>,
) -> Result<Option<String>, String> {
    if bytes.len() > 512 * 1024 * 1024
        || name.is_empty()
        || name.contains(['/', '\\'])
        || name.contains("..")
    {
        return Err("Invalid artifact".into());
    }
    if std::env::var_os("FLAT2RIG_HEADLESS_SMOKE").is_some() {
        if let Some(dir) = std::env::var_os("FLAT2RIG_HEADLESS_SAVE_DIR") {
            let file_name = std::path::Path::new(&name)
                .file_name()
                .ok_or_else(|| "Invalid artifact".to_string())?;
            let dir_path = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir_path).map_err(|e| e.to_string())?;
            let path = dir_path.join(file_name);
            atomic_write_bytes(&path, &bytes)?;
            return Ok(Some(path.to_string_lossy().into_owned()));
        }
    }
    tauri::async_runtime::spawn_blocking(move || {
        let Some(selected) = app
            .dialog()
            .file()
            .set_file_name(&name)
            .blocking_save_file()
        else {
            return Ok(None);
        };
        let path = selected.into_path().map_err(|e| e.to_string())?;
        atomic_write_bytes(&path, &bytes)?;
        Ok(Some(path.to_string_lossy().into_owned()))
    })
    .await
    .map_err(|e| e.to_string())?
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PastedPathFile {
    name: String,
    bytes: String,
}

const PASTED_IMAGE_MAX_BYTES: u64 = 60 * 1024 * 1024;
const PASTED_PROJECT_MAX_BYTES: u64 = 512 * 1024 * 1024;

/// Validates a path pasted from the clipboard: an existing regular file whose extension is one the
/// app opens (PNG / JPEG illustration or `.flat2rig` project) within the input size limits.
/// Returns the file name and the size limit; never reads other file types.
fn validate_pasted_path(path: &std::path::Path) -> Result<(String, u64), String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let limit = match ext.as_str() {
        "png" | "jpg" | "jpeg" => PASTED_IMAGE_MAX_BYTES,
        "flat2rig" => PASTED_PROJECT_MAX_BYTES,
        _ => return Err("PNG, JPG 이미지 또는 .flat2rig 파일 경로만 붙여넣을 수 있습니다.".into()),
    };
    if !path.is_absolute() {
        return Err("붙여넣은 경로는 전체 경로여야 합니다.".into());
    }
    let meta = std::fs::metadata(path).map_err(|_| "붙여넣은 경로의 파일을 찾을 수 없습니다.".to_string())?;
    if !meta.is_file() {
        return Err("붙여넣은 경로가 파일이 아닙니다.".into());
    }
    if meta.len() > limit {
        return Err(format!(
            "파일이 너무 큽니다(최대 {}MB).",
            limit / (1024 * 1024)
        ));
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| "파일 이름을 읽을 수 없습니다.".to_string())?
        .to_string();
    Ok((name, limit))
}

/// Reads a PNG / JPEG / `.flat2rig` file whose path the user pasted (Ctrl+V of a path or
/// "Copy as path"). The webview cannot read arbitrary paths itself.
#[tauri::command]
async fn read_pasted_path(path: String) -> Result<PastedPathFile, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let p = PathBuf::from(path.trim());
        let (name, limit) = validate_pasted_path(&p)?;
        let bytes = std::fs::read(&p).map_err(|e| e.to_string())?;
        if bytes.len() as u64 > limit {
            return Err("파일이 너무 큽니다.".into());
        }
        Ok(PastedPathFile {
            name,
            bytes: STANDARD.encode(bytes),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let app_data_dir = app.path().app_local_data_dir().ok();
            let analysis_state =
                AnalysisState::new(app_data_dir).map_err(std::io::Error::other)?;
            app.manage(analysis_state);
            app.manage(updater::PendingUpdate::default());
            if std::env::var_os("FLAT2RIG_HEADLESS_SMOKE").is_some() {
                if let Some(window) = app.get_webview_window("main") {
                    window.hide()?;
                }
            }
            Ok(())
        })
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            generate_parts,
            propose_parts,
            refine_part_prompt,
            build_rig,
            compute_exposure_map,
            fill_part,
            fill_part_occlusion,
            save_artifact,
            read_pasted_path,
            get_model_status,
            set_model_path,
            download_model,
            cancel_analysis,
            analyze_face,
            detect_landmarks_for_face,
            updater::get_app_version,
            updater::check_app_update,
            updater::install_app_update,
            updater::relaunch_app
        ])
        .run(tauri::generate_context!())
        .expect("failed to run flat2rig");
}

#[cfg(test)]
mod pasted_path_tests {
    use super::validate_pasted_path;

    #[test]
    fn pasted_path_accepts_only_existing_images_and_projects() {
        let dir = std::env::temp_dir().join(format!("flat2rig-paste-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("art.PNG");
        std::fs::write(&png, b"x").unwrap();
        let txt = dir.join("notes.txt");
        std::fs::write(&txt, b"x").unwrap();
        assert_eq!(validate_pasted_path(&png).unwrap().0, "art.PNG");
        assert!(validate_pasted_path(&txt).is_err());
        assert!(validate_pasted_path(&dir.join("missing.png")).is_err());
        assert!(validate_pasted_path(std::path::Path::new("relative.png")).is_err());
        let folder = dir.join("folder.png");
        std::fs::create_dir_all(&folder).unwrap();
        assert!(validate_pasted_path(&folder).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
