//! Desktop application auto-update checking, signature verification, and installer staging.
//!
//! Uses `tauri-plugin-updater` with the trusted minisign public key in `tauri.conf.json`, plus
//! an optional read-only GitHub Releases fallback notice (never installs unsigned binaries).

use semver::Version;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::{ipc::Channel, AppHandle, State};
use tauri_plugin_updater::{Update, UpdaterExt};

pub const DEFAULT_UPDATE_ENDPOINT: &str =
    "https://github.com/jinhoOps/flat2rig/releases/latest/download/latest.json";
pub const GITHUB_RELEASES_API: &str =
    "https://api.github.com/repos/jinhoOps/flat2rig/releases/latest";
pub const GITHUB_RELEASES_PAGE: &str = "https://github.com/jinhoOps/flat2rig/releases";

pub struct PendingUpdate(pub Mutex<Option<Update>>);

impl Default for PendingUpdate {
    fn default() -> Self {
        Self(Mutex::new(None))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateCheckResult {
    pub status: String,
    pub current_version: String,
    pub latest_version: Option<String>,
    pub date: Option<String>,
    pub notes: Option<String>,
    pub signed: bool,
    pub endpoint_used: String,
    pub release_url: String,
    pub message: String,
}

#[derive(Clone, Serialize)]
#[serde(tag = "event", content = "data")]
pub enum UpdateDownloadEvent {
    #[serde(rename_all = "camelCase")]
    Started { content_length: Option<u64> },
    #[serde(rename_all = "camelCase")]
    Progress {
        downloaded: u64,
        content_length: Option<u64>,
    },
    Finished,
}

/// Parses semantic versions tolerating a leading `v`/`V` prefix.
pub fn parse_semver_tag(raw: &str) -> Option<Version> {
    let trimmed = raw.trim();
    let no_v = trimmed.strip_prefix('v').or_else(|| trimmed.strip_prefix('V')).unwrap_or(trimmed);
    Version::parse(no_v).ok()
}

/// Returns true iff `candidate` is strictly newer than `current` under SemVer rules.
pub fn is_newer_version(current: &str, candidate: &str) -> bool {
    match (parse_semver_tag(current), parse_semver_tag(candidate)) {
        (Some(cur), Some(cand)) => cand > cur,
        _ => false,
    }
}

/// Validates user-supplied custom update manifest URLs (`https://` or local loopback dev server).
pub fn normalize_endpoint_url(custom_endpoint: Option<&str>) -> Result<String, String> {
    let Some(raw) = custom_endpoint.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(DEFAULT_UPDATE_ENDPOINT.to_string());
    };
    let parsed = tauri::Url::parse(raw).map_err(|e| format!("업데이트 URL 형식 오류: {e}"))?;
    let host = parsed.host_str().unwrap_or_default();
    let is_loopback = matches!(host, "127.0.0.1" | "localhost" | "::1");
    if parsed.scheme() == "https" || (parsed.scheme() == "http" && is_loopback) {
        Ok(parsed.to_string())
    } else {
        Err("보안상 https:// 업데이트 엔드포인트만 허용됩니다.".into())
    }
}

#[derive(Debug, Deserialize)]
struct GithubLatestRelease {
    tag_name: Option<String>,
    html_url: Option<String>,
    published_at: Option<String>,
    body: Option<String>,
}

fn fetch_github_release_fallback() -> Result<GithubLatestRelease, String> {
    let mut resp = ureq::get(GITHUB_RELEASES_API)
        .header("User-Agent", "flat2rig-updater")
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| format!("{e}"))?;
    let body = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("{e}"))?;
    serde_json::from_str::<GithubLatestRelease>(&body).map_err(|e| format!("{e}"))
}

#[tauri::command]
pub fn get_app_version(app: AppHandle) -> String {
    app.package_info().version.to_string()
}

#[tauri::command]
pub async fn check_app_update(
    app: AppHandle,
    pending: State<'_, PendingUpdate>,
    custom_endpoint: Option<String>,
) -> Result<UpdateCheckResult, String> {
    let current_version = app.package_info().version.to_string();
    let endpoint = normalize_endpoint_url(custom_endpoint.as_deref())?;
    let is_default_repo = endpoint == DEFAULT_UPDATE_ENDPOINT;

    let parsed_url = tauri::Url::parse(&endpoint).map_err(|e| format!("URL 파싱 실패: {e}"))?;
    let updater_res = app
        .updater_builder()
        .endpoints(vec![parsed_url])
        .map_err(|e| format!("업데이터 설정 실패: {e}"))?
        .build()
        .map_err(|e| format!("업데이터 초기화 실패: {e}"))?
        .check()
        .await;

    match updater_res {
        Ok(Some(update)) => {
            let latest = update.version.clone();
            let date = update.date.map(|d| d.to_string());
            let notes = update.body.clone();
            *pending
                .0
                .lock()
                .map_err(|_| "Updater lock poisoned".to_string())? = Some(update);
            Ok(UpdateCheckResult {
                status: "available".into(),
                current_version,
                latest_version: Some(latest.clone()),
                date,
                notes,
                signed: true,
                endpoint_used: endpoint,
                release_url: GITHUB_RELEASES_PAGE.into(),
                message: format!("서명 검증이 가능한 새 버전 v{latest}을(를) 찾았습니다."),
            })
        }
        Ok(None) => {
            if let Ok(mut guard) = pending.0.lock() {
                *guard = None;
            }
            Ok(UpdateCheckResult {
                status: "up-to-date".into(),
                current_version: current_version.clone(),
                latest_version: Some(current_version.clone()),
                date: None,
                notes: None,
                signed: true,
                endpoint_used: endpoint,
                release_url: GITHUB_RELEASES_PAGE.into(),
                message: format!("현재 최신 버전(v{current_version})을 사용 중입니다."),
            })
        }
        Err(updater_err) => {
            if let Ok(mut guard) = pending.0.lock() {
                *guard = None;
            }
            if is_default_repo {
                let gh = tauri::async_runtime::spawn_blocking(fetch_github_release_fallback)
                    .await
                    .ok()
                    .and_then(|r| r.ok());
                if let Some(rel) = gh {
                    if let Some(tag) = rel.tag_name {
                        let release_url = rel
                            .html_url
                            .unwrap_or_else(|| GITHUB_RELEASES_PAGE.to_string());
                        if is_newer_version(&current_version, &tag) {
                            let clean_latest = parse_semver_tag(&tag)
                                .map(|v| v.to_string())
                                .unwrap_or(tag);
                            return Ok(UpdateCheckResult {
                                status: "unsigned-release".into(),
                                current_version,
                                latest_version: Some(clean_latest.clone()),
                                date: rel.published_at,
                                notes: rel.body,
                                signed: false,
                                endpoint_used: endpoint,
                                release_url,
                                message: format!(
                                    "릴리스 v{clean_latest}이(가) 있지만 서명 매니페스트(latest.json)가 없어 자동 설치를 건너뛰었습니다. 릴리스 페이지에서 직접 확인해 주세요."
                                ),
                            });
                        }
                        return Ok(UpdateCheckResult {
                            status: "up-to-date".into(),
                            current_version: current_version.clone(),
                            latest_version: Some(current_version.clone()),
                            date: rel.published_at,
                            notes: None,
                            signed: false,
                            endpoint_used: endpoint,
                            release_url,
                            message: format!("현재 최신 버전(v{current_version})입니다."),
                        });
                    }
                }
            }

            Ok(UpdateCheckResult {
                status: "unavailable".into(),
                current_version,
                latest_version: None,
                date: None,
                notes: None,
                signed: false,
                endpoint_used: endpoint,
                release_url: GITHUB_RELEASES_PAGE.into(),
                message: format!(
                    "업데이트 피드에 연결하지 못했습니다(오프라인 또는 릴리스 미배포): {updater_err}"
                ),
            })
        }
    }
}

#[tauri::command]
pub async fn install_app_update(
    pending: State<'_, PendingUpdate>,
    on_event: Channel<UpdateDownloadEvent>,
) -> Result<(), String> {
    let Some(update) = pending
        .0
        .lock()
        .map_err(|_| "Updater lock poisoned".to_string())?
        .take()
    else {
        return Err("설치할 업데이트가 없습니다. 먼저 업데이트 확인을 실행하세요.".into());
    };

    let mut started = false;
    let mut downloaded: u64 = 0;
    update
        .download_and_install(
            |chunk_length, content_length| {
                if !started {
                    let _ = on_event.send(UpdateDownloadEvent::Started { content_length });
                    started = true;
                }
                downloaded += chunk_length as u64;
                let _ = on_event.send(UpdateDownloadEvent::Progress {
                    downloaded,
                    content_length,
                });
            },
            || {
                let _ = on_event.send(UpdateDownloadEvent::Finished);
            },
        )
        .await
        .map_err(|e| format!("업데이트 다운로드 또는 설치 실패: {e}"))?;

    Ok(())
}

#[tauri::command]
pub fn relaunch_app(app: AppHandle) {
    app.restart();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_comparison_handles_v_prefix_and_prerelease() {
        assert!(is_newer_version("0.1.0", "0.1.1"));
        assert!(is_newer_version("0.1.0", "0.2.0-1"));
        assert!(is_newer_version("v0.2.0-1", "0.2.0"));
        assert!(is_newer_version("0.1.0", "v0.2.0"));
        assert!(is_newer_version("v0.2.0-beta.1", "0.2.0"));
        assert!(!is_newer_version("0.2.0", "0.1.9"));
        assert!(!is_newer_version("0.1.0", "0.1.0"));
        assert!(!is_newer_version("0.1.0", "not-a-version"));
    }

    #[test]
    fn endpoint_normalization_enforces_https_or_loopback() {
        assert_eq!(
            normalize_endpoint_url(None).unwrap(),
            DEFAULT_UPDATE_ENDPOINT
        );
        assert_eq!(
            normalize_endpoint_url(Some("  ")).unwrap(),
            DEFAULT_UPDATE_ENDPOINT
        );
        assert!(normalize_endpoint_url(Some("https://releases.example.com/latest.json")).is_ok());
        assert!(normalize_endpoint_url(Some("http://127.0.0.1:8080/latest.json")).is_ok());
        assert!(normalize_endpoint_url(Some("http://evil.example.com/latest.json")).is_err());
    }
}
