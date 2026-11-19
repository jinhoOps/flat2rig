use crate::{emit_progress, AnalysisState};
use serde::Serialize;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use tauri::Manager;

pub const SEE_THROUGH_LICENSE_NOTE: &str =
    "OpenRAIL++ (모델 가중치) / Apache-2.0 (코드) — 앱에 번들하지 않으며 사용자가 설치한 외부 도구만 실행합니다.";

pub const SEE_THROUGH_LIMITATIONS: &[&str] = &[
    "입·코는 원화 복사본이라 숨은 영역이 없고 입 벌리기 리깅이 제한됩니다.",
    "치비(SD) 비율 및 큰 소품(모자 등)은 누락되거나 얼굴 색이 틀어질 수 있습니다.",
    "눈썹·아래 눈꺼풀·앞/뒷머리 경계는 시드에 따라 편차가 크므로 2~3개 시드 비교를 권장합니다.",
];

const SCRIPT_CANDIDATES: &[&str] = &[
    "inference_psd_blockswap.py",
    "inference_psd.py",
    "inference_psd_quantized.py",
];

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughInstallStatus {
    pub installed: bool,
    pub install_dir: String,
    pub python_path: Option<String>,
    pub script_path: Option<String>,
    pub hf_home: Option<String>,
    pub message: String,
    pub license: String,
    pub limitations: Vec<String>,
}

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughDecomposeResult {
    pub psd_path: String,
    pub output_dir: String,
    pub seed: u64,
    pub command_summary: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeeThroughCommandSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    pub hf_home: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub expected_psd_path: PathBuf,
    pub seed: u64,
}

fn validate_safe_absolute_path(raw: &str, label: &str) -> Result<PathBuf, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!("{label} 경로가 비어 있습니다."));
    }
    let path = PathBuf::from(trimmed);
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("{label} 경로에 상위 디렉터리 이동(..)을 포함할 수 없습니다."));
    }
    if !path.is_absolute() {
        return Err(format!("{label} 경로는 절대 경로여야 합니다."));
    }
    Ok(path)
}

fn check_readable_dir(path: &Path, label: &str) -> Result<(), String> {
    let meta = std::fs::metadata(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => format!("{label} 폴더를 찾을 수 없습니다: {}", path.display()),
        std::io::ErrorKind::PermissionDenied => {
            format!("{label} 폴더에 접근할 권한이 없습니다: {}", path.display())
        }
        _ => format!("{label} 폴더를 확인할 수 없습니다({}): {e}", path.display()),
    })?;
    if !meta.is_dir() {
        return Err(format!("{label} 경로가 폴더가 아닙니다: {}", path.display()));
    }
    std::fs::read_dir(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::PermissionDenied => {
            format!("{label} 폴더를 읽을 권한이 없습니다: {}", path.display())
        }
        _ => format!("{label} 폴더를 읽을 수 없습니다({}): {e}", path.display()),
    })?;
    Ok(())
}

fn find_python_executable(install_dir: &Path, repo_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        install_dir.join("venv").join("Scripts").join("python.exe"),
        install_dir.join("venv").join("bin").join("python"),
        install_dir.join(".venv").join("Scripts").join("python.exe"),
        install_dir.join(".venv").join("bin").join("python"),
        repo_dir.join("venv").join("Scripts").join("python.exe"),
        repo_dir.join("venv").join("bin").join("python"),
        repo_dir.join(".venv").join("Scripts").join("python.exe"),
        repo_dir.join(".venv").join("bin").join("python"),
        install_dir.join("python").join("python.exe"),
        install_dir.join("python").join("bin").join("python3"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

fn find_inference_script(install_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let repo_sub = install_dir.join("repo");
    for base in [&repo_sub, install_dir] {
        if !base.is_dir() {
            continue;
        }
        for name in SCRIPT_CANDIDATES {
            let p = base.join(name);
            if p.is_file() {
                return Some((base.to_path_buf(), p));
            }
        }
    }
    None
}

pub fn check_install_dir(install_dir: &str) -> Result<SeeThroughInstallStatus, String> {
    let root = validate_safe_absolute_path(install_dir, "See-through 설치")?;
    check_readable_dir(&root, "See-through 설치")?;

    let script_hit = find_inference_script(&root);
    let repo_dir = script_hit
        .as_ref()
        .map(|(dir, _)| dir.clone())
        .unwrap_or_else(|| {
            let sub = root.join("repo");
            if sub.is_dir() {
                sub
            } else {
                root.clone()
            }
        });
    let python_hit = find_python_executable(&root, &repo_dir);
    let hf_home = {
        let p = root.join("hf-home");
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    };

    let installed = python_hit.is_some() && script_hit.is_some();
    let message = match (&python_hit, &script_hit) {
        (Some(_), Some((_, script))) => format!(
            "설치 확인됨 ({})",
            script
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("inference_psd")
        ),
        (None, Some(_)) => {
            "Python 가상환경 실행 파일(venv/Scripts/python.exe 또는 venv/bin/python)을 찾을 수 없습니다.".to_string()
        }
        (Some(_), None) => {
            "추론 스크립트(inference_psd_blockswap.py 또는 inference_psd.py)를 찾을 수 없습니다.".to_string()
        }
        (None, None) => {
            "Python 가상환경과 See-through 추론 스크립트를 모두 찾을 수 없습니다.".to_string()
        }
    };

    Ok(SeeThroughInstallStatus {
        installed,
        install_dir: root.to_string_lossy().into_owned(),
        python_path: python_hit.map(|p| p.to_string_lossy().into_owned()),
        script_path: script_hit.map(|(_, p)| p.to_string_lossy().into_owned()),
        hf_home: hf_home.map(|p| p.to_string_lossy().into_owned()),
        message,
        license: SEE_THROUGH_LICENSE_NOTE.to_string(),
        limitations: SEE_THROUGH_LIMITATIONS
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
    })
}

pub fn build_seethrough_command_spec(
    status: &SeeThroughInstallStatus,
    image_path: &str,
    output_dir: Option<&str>,
    seed: Option<u64>,
) -> Result<SeeThroughCommandSpec, String> {
    if !status.installed {
        return Err(format!("See-through가 설치되지 않았습니다: {}", status.message));
    }
    let python_path = PathBuf::from(
        status
            .python_path
            .as_deref()
            .ok_or_else(|| "Python 실행 파일 경로가 없습니다.".to_string())?,
    );
    let script_path = PathBuf::from(
        status
            .script_path
            .as_deref()
            .ok_or_else(|| "See-through 추론 스크립트 경로가 없습니다.".to_string())?,
    );

    let img = validate_safe_absolute_path(image_path, "입력 이미지")?;
    let meta = std::fs::metadata(&img).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => format!("입력 이미지 파일을 찾을 수 없습니다: {}", img.display()),
        std::io::ErrorKind::PermissionDenied => {
            format!("입력 이미지 파일에 접근할 권한이 없습니다: {}", img.display())
        }
        _ => format!("입력 이미지 파일을 확인할 수 없습니다({}): {e}", img.display()),
    })?;
    if !meta.is_file() {
        return Err(format!("입력 이미지 경로가 파일이 아닙니다: {}", img.display()));
    }
    let ext = img
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    if !matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp") {
        return Err("PNG, JPG, WebP 이미지만 See-through로 분해할 수 있습니다.".into());
    }

    let seed = seed.unwrap_or(42);
    let stem = img
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("illustration");
    let out_dir = match output_dir {
        Some(dir) if !dir.trim().is_empty() => validate_safe_absolute_path(dir, "출력")?,
        _ => std::env::temp_dir()
            .join("flat2rig-seethrough")
            .join(format!("{stem}_s{seed}_{}", std::process::id())),
    };

    let cwd = script_path
        .parent()
        .ok_or_else(|| "추론 스크립트 상위 폴더를 확인할 수 없습니다.".to_string())?
        .to_path_buf();

    let args = vec![
        script_path.to_string_lossy().into_owned(),
        "--srcp".to_string(),
        img.to_string_lossy().into_owned(),
        "--save_to_psd".to_string(),
        "--seed".to_string(),
        seed.to_string(),
        "--save_dir".to_string(),
        out_dir.to_string_lossy().into_owned(),
    ];

    let expected_psd_path = out_dir.join(format!("{stem}.psd"));

    Ok(SeeThroughCommandSpec {
        program: python_path,
        args,
        cwd,
        hf_home: status.hf_home.as_ref().map(PathBuf::from),
        output_dir: out_dir,
        expected_psd_path,
        seed,
    })
}

fn resolve_output_psd(spec: &SeeThroughCommandSpec) -> Result<PathBuf, String> {
    if spec.expected_psd_path.is_file() {
        return Ok(spec.expected_psd_path.clone());
    }
    if let Ok(entries) = std::fs::read_dir(&spec.output_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("psd"))
            {
                return Ok(p);
            }
        }
    }
    Err(format!(
        "See-through 실행은 완료되었지만 출력 PSD를 찾을 수 없습니다: {}",
        spec.expected_psd_path.display()
    ))
}

#[tauri::command]
pub async fn seethrough_check_install(install_dir: String) -> Result<SeeThroughInstallStatus, String> {
    tauri::async_runtime::spawn_blocking(move || check_install_dir(&install_dir))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn seethrough_decompose(
    app: tauri::AppHandle,
    install_dir: String,
    image_path: String,
    output_dir: Option<String>,
    seed: Option<u64>,
    job_id: Option<u64>,
) -> Result<SeeThroughDecomposeResult, String> {
    let state = app.state::<AnalysisState>();
    let my_job = state.begin_job(job_id);
    tauri::async_runtime::spawn_blocking(move || {
        emit_progress(&app, my_job, "See-through 설치 및 입력 확인", 0, 3);
        let status = check_install_dir(&install_dir)?;
        let spec = build_seethrough_command_spec(&status, &image_path, output_dir.as_deref(), seed)?;
        std::fs::create_dir_all(&spec.output_dir)
            .map_err(|e| format!("출력 폴더를 생성할 수 없습니다({}): {e}", spec.output_dir.display()))?;

        emit_progress(&app, my_job, "See-through 외부 실행 중", 1, 3);
        let mut cmd = Command::new(&spec.program);
        cmd.args(&spec.args)
            .current_dir(&spec.cwd)
            .env("PYTHONUTF8", "1");
        if let Some(ref hf_home) = spec.hf_home {
            cmd.env("HF_HOME", hf_home);
        }

        let output = cmd.output().map_err(|e| {
            format!(
                "See-through 외부 프로세스를 실행할 수 없습니다({}): {e}",
                spec.program.display()
            )
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let tail = if !stderr.trim().is_empty() {
                stderr.trim()
            } else {
                stdout.trim()
            };
            let snippet: String = tail.chars().rev().take(400).collect::<String>().chars().rev().collect();
            return Err(format!(
                "See-through 실행 실패(종료 코드 {:?}): {}",
                output.status.code(),
                if snippet.is_empty() { "오류 출력 없음" } else { &snippet }
            ));
        }

        emit_progress(&app, my_job, "생성된 PSD 확인", 2, 3);
        let psd_path = resolve_output_psd(&spec)?;
        emit_progress(&app, my_job, "완료", 3, 3);

        Ok(SeeThroughDecomposeResult {
            psd_path: psd_path.to_string_lossy().into_owned(),
            output_dir: spec.output_dir.to_string_lossy().into_owned(),
            seed: spec.seed,
            command_summary: format!("{} {}", spec.program.display(), spec.args.join(" ")),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::{build_seethrough_command_spec, check_install_dir, SEE_THROUGH_LICENSE_NOTE};

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "flat2rig-seethrough-test-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn rejects_empty_relative_traversal_and_missing_install_dirs() {
        assert!(check_install_dir("").is_err());
        assert!(check_install_dir("relative/seethrough").is_err());
        assert!(check_install_dir("C:/foo/../bar").is_err());
        let missing = std::env::temp_dir().join("flat2rig-nonexistent-seethrough-dir");
        let _ = std::fs::remove_dir_all(&missing);
        assert!(check_install_dir(&missing.to_string_lossy()).is_err());
    }

    #[test]
    fn detects_v2_layout_and_builds_expected_cli_args() {
        let root = temp_dir("v2");
        let py_dir = root.join("venv").join("Scripts");
        let repo_dir = root.join("repo");
        let hf_dir = root.join("hf-home");
        std::fs::create_dir_all(&py_dir).unwrap();
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::create_dir_all(&hf_dir).unwrap();
        let py_exe = py_dir.join("python.exe");
        let script = repo_dir.join("inference_psd_blockswap.py");
        std::fs::write(&py_exe, b"").unwrap();
        std::fs::write(&script, b"").unwrap();

        let img = root.join("himikan.png");
        std::fs::write(&img, b"png").unwrap();

        let status = check_install_dir(&root.to_string_lossy()).unwrap();
        assert!(status.installed);
        assert_eq!(status.license, SEE_THROUGH_LICENSE_NOTE);
        assert!(!status.limitations.is_empty());
        assert_eq!(status.python_path.as_deref(), Some(py_exe.to_string_lossy().as_ref()));
        assert_eq!(status.script_path.as_deref(), Some(script.to_string_lossy().as_ref()));
        assert_eq!(status.hf_home.as_deref(), Some(hf_dir.to_string_lossy().as_ref()));

        let out_dir = root.join("out");
        let spec = build_seethrough_command_spec(
            &status,
            &img.to_string_lossy(),
            Some(&out_dir.to_string_lossy()),
            Some(7),
        )
        .unwrap();
        assert_eq!(spec.program, py_exe);
        assert_eq!(spec.cwd, repo_dir);
        assert_eq!(spec.seed, 7);
        assert_eq!(spec.expected_psd_path, out_dir.join("himikan.psd"));
        assert_eq!(
            spec.args,
            vec![
                script.to_string_lossy().into_owned(),
                "--srcp".to_string(),
                img.to_string_lossy().into_owned(),
                "--save_to_psd".to_string(),
                "--seed".to_string(),
                "7".to_string(),
                "--save_dir".to_string(),
                out_dir.to_string_lossy().into_owned(),
            ]
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn reports_uninstalled_when_python_or_script_is_missing() {
        let root = temp_dir("partial");
        let status = check_install_dir(&root.to_string_lossy()).unwrap();
        assert!(!status.installed);
        assert!(status.python_path.is_none());
        assert!(status.script_path.is_none());

        let img = root.join("art.png");
        std::fs::write(&img, b"png").unwrap();
        assert!(build_seethrough_command_spec(&status, &img.to_string_lossy(), None, None).is_err());

        let _ = std::fs::remove_dir_all(&root);
    }
}
