pub mod analysis;
pub mod core;
pub mod inference;
use analysis::{
    detector::{detect_faces, FaceCandidate, DEFAULT_FACE_SCORE_THRESHOLD},
    landmarks::{detect_landmarks, Landmark28Point, RigLandmarkProvenance},
    models::{atomic_write_bytes, download_and_install_model, ModelRole, ModelStatusEntry},
    parts::{propose_parts_auto, propose_parts_auto_with_options, refine_part_with_prompts, PartProposalMeta},
    segmentation::PromptInput,
    state::{hash_rgba_bytes, AnalysisState},
};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
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
    metas: Vec<PartProposalMeta>,
    provider: String,
    model_hashes: HashMap<String, String>,
    postprocess_version: String,
    image_hash: String,
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
        let out = propose_parts_auto_with_options(
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
            hair_split.unwrap_or(true),
        )?;
        if state.is_stale_or_cancelled(my_job) {
            return Err("CANCELLED".into());
        }
        Ok(ProposePartsResponse {
            job_id: my_job,
            part_ids: out.part_ids,
            masks: out.masks.iter().map(|m| STANDARD.encode(m)).collect(),
            metas: out.metas,
            provider: out.provider,
            model_hashes: out.model_hashes,
            postprocess_version: out.postprocess_version,
            image_hash: out.image_hash,
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
        for part in parts {
            let visible = core::decode_bytes(&part.mask, n)?;
            let fill = match part.fill_mask {
                Some(ref f) => Some(core::decode_bytes(f, n)?),
                None => None,
            };
            let effective = core::effective_part_mask(&visible, fill.as_deref());
            let mesh = core::mesh_for_part(&effective, width, height, 14, &landmarks);
            let quality =
                core::check_mesh_quality(&mesh, &effective, width, height, &part.id, &landmarks);
            let mut samples = Vec::new();
            for (parameter, values) in [
                ("angleX", vec![-10., 0., 10.]),
                ("angleY", vec![-10., 0., 10.]),
                ("angleZ", vec![-15., 0., 15.]),
                ("eyeL", vec![0., 1.]),
                ("eyeR", vec![0., 1.]),
                ("mouth", vec![0., 1.]),
                ("hair", vec![-1., 0., 1.]),
            ] {
                for value in values {
                    let mut pose = core::Pose::default();
                    match parameter {
                        "angleX" => pose.angle_x = value,
                        "angleY" => pose.angle_y = value,
                        "angleZ" => pose.angle_z = value,
                        "eyeL" => pose.eye_l = value,
                        "eyeR" => pose.eye_r = value,
                        "mouth" => pose.mouth = value,
                        _ => pose.hair = value,
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
    if bytes.len() > 512 * 1024 * 1024 || name.contains(['/', '\\']) {
        return Err("Invalid artifact".into());
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
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let app_data_dir = app.path().app_local_data_dir().ok();
            let analysis_state =
                AnalysisState::new(app_data_dir).map_err(std::io::Error::other)?;
            app.manage(analysis_state);
            if std::env::var_os("FLAT2RIG_HEADLESS_SMOKE").is_some() {
                if let Some(window) = app.get_webview_window("main") {
                    window.hide()?;
                }
            }
            Ok(())
        })
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            generate_parts,
            propose_parts,
            refine_part_prompt,
            build_rig,
            fill_part,
            fill_part_occlusion,
            save_artifact,
            get_model_status,
            set_model_path,
            download_model,
            cancel_analysis,
            analyze_face,
            detect_landmarks_for_face
        ])
        .run(tauri::generate_context!())
        .expect("failed to run flat2rig");
}
