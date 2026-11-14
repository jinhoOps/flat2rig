pub mod core;
pub mod inference;
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use tauri::Manager;
use tauri_plugin_dialog::DialogExt;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Separation {
    masks: Vec<String>,
    provider: String,
    model_hash: Option<String>,
}
#[tauri::command]
async fn generate_parts(
    rgba: String,
    width: usize,
    height: usize,
    landmarks: core::Landmarks,
    tolerance: f64,
    model_path: Option<String>,
) -> Result<Separation, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let n = core::check_size(width, height)?;
        landmarks.validate(width, height)?;
        let data = core::decode_bytes(&rgba, n * 4)?;
        let (alpha, hash) = if let Some(path) = model_path {
            let (a, h) =
                inference::background_alpha(std::path::Path::new(&path), &data, width, height)?;
            (Some(a), Some(h))
        } else {
            (None, None)
        };
        let masks = core::separate(
            &data,
            width,
            height,
            &landmarks,
            tolerance.clamp(0., 180.),
            alpha.as_deref(),
        )
        .iter()
        .map(|m| STANDARD.encode(m))
        .collect();
        Ok(Separation {
            masks,
            provider: if hash.is_some() {
                "u2net-background+geometry-template-v1".into()
            } else {
                "geometry-template-v1".into()
            },
            model_hash: hash,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}
#[derive(Deserialize)]
struct MeshInput {
    id: String,
    mask: String,
}
#[derive(Serialize)]
struct RigPart {
    id: String,
    mesh: core::Mesh,
    samples: Vec<Sample>,
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
            let mask = core::decode_bytes(&part.mask, n)?;
            let mesh = core::mesh_for(&mask, width, height, 14);
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
        let parent = path.parent().ok_or("Invalid destination")?;
        let filename = path
            .file_name()
            .ok_or("Invalid filename")?
            .to_string_lossy();
        let temp = parent.join(format!(".{filename}.flat2rig-{}.tmp", std::process::id()));
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| e.to_string())?;
        let write_result = file.write_all(&bytes).and_then(|_| file.sync_all());
        drop(file);
        if let Err(e) = write_result {
            let _ = std::fs::remove_file(&temp);
            return Err(e.to_string());
        }
        // Windows rename replaces an existing file only through the platform API.
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            #[link(name = "kernel32")]
            extern "system" {
                fn MoveFileExW(from: *const u16, to: *const u16, flags: u32) -> i32;
            }
            let from: Vec<u16> = temp.as_os_str().encode_wide().chain(Some(0)).collect();
            let to: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
            if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), 0x1 | 0x8) } == 0 {
                let err = std::io::Error::last_os_error();
                let _ = std::fs::remove_file(&temp);
                return Err(err.to_string());
            }
        }
        #[cfg(not(windows))]
        std::fs::rename(&temp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&temp);
            e.to_string()
        })?;
        Ok(Some(path.to_string_lossy().into_owned()))
    })
    .await
    .map_err(|e| e.to_string())?
}
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
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
            build_rig,
            fill_part,
            save_artifact
        ])
        .run(tauri::generate_context!())
        .expect("failed to run flat2rig");
}
