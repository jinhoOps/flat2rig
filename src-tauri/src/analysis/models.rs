use ort::session::Session;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

pub const EMBEDDED_MANIFEST_JSON: &str = include_str!("../../../models/manifest.json");
const MAX_MODEL_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    FaceDetector,
    Landmark28,
    Background,
    PartEncoder,
    PartDecoder,
}

impl ModelRole {
    pub const ALL: [ModelRole; 5] = [
        ModelRole::FaceDetector,
        ModelRole::Landmark28,
        ModelRole::Background,
        ModelRole::PartEncoder,
        ModelRole::PartDecoder,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ModelRole::FaceDetector => "face_detector",
            ModelRole::Landmark28 => "landmark_28",
            ModelRole::Background => "background",
            ModelRole::PartEncoder => "part_encoder",
            ModelRole::PartDecoder => "part_decoder",
        }
    }

    pub fn parse_id(s: &str) -> Option<Self> {
        match s {
            "face_detector" => Some(ModelRole::FaceDetector),
            "landmark_28" => Some(ModelRole::Landmark28),
            "background" => Some(ModelRole::Background),
            "part_encoder" => Some(ModelRole::PartEncoder),
            "part_decoder" => Some(ModelRole::PartDecoder),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestModel {
    pub role: String,
    pub id: String,
    pub filename: String,
    pub version: String,
    pub license: String,
    pub sha256: String,
    pub bytes: u64,
    #[serde(default)]
    pub upstream_repo: Option<String>,
    #[serde(default)]
    pub upstream_revision: Option<String>,
    #[serde(default)]
    pub download_url: Option<String>,
    #[serde(default)]
    pub checkpoint_url: Option<String>,
    pub postprocess_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifest {
    pub schema_version: u32,
    pub opset_version: u32,
    pub max_ir_version: u32,
    pub models: Vec<ManifestModel>,
}

impl ModelManifest {
    pub fn embedded() -> Result<Self, String> {
        serde_json::from_str(EMBEDDED_MANIFEST_JSON)
            .map_err(|e| format!("Invalid embedded models/manifest.json: {e}"))
    }

    pub fn entry(&self, role: ModelRole) -> Option<&ManifestModel> {
        self.models.iter().find(|m| m.role == role.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatusEntry {
    pub role: String,
    pub id: String,
    pub filename: String,
    pub version: String,
    pub license: String,
    pub expected_sha256: String,
    pub expected_bytes: u64,
    pub download_url: Option<String>,
    pub path: Option<String>,
    pub actual_sha256: Option<String>,
    pub installed: bool,
    pub hash_matches: bool,
    pub loaded: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
struct HashCacheEntry {
    len: u64,
    modified_ns: u128,
    sha256: String,
}

#[derive(Debug)]
pub struct CachedSession {
    pub session: Arc<Mutex<Session>>,
    pub sha256: String,
    pub path: PathBuf,
}

/// Discovers, validates, and caches ONNX models declared in `models/manifest.json`.
pub struct ModelRegistry {
    manifest: ModelManifest,
    search_dirs: Vec<PathBuf>,
    overrides: HashMap<ModelRole, PathBuf>,
    hash_cache: HashMap<PathBuf, HashCacheEntry>,
    sessions: HashMap<ModelRole, CachedSession>,
}

impl ModelRegistry {
    pub fn new(extra_search_dirs: Vec<PathBuf>) -> Result<Self, String> {
        let manifest = ModelManifest::embedded()?;
        let mut search_dirs = extra_search_dirs;

        if let Ok(cwd) = std::env::current_dir() {
            search_dirs.push(cwd.join("models"));
            if let Some(parent) = cwd.parent() {
                search_dirs.push(parent.join("models"));
            }
        }
        // Fallback compile-time repo root `models/` for cargo test / dev runs.
        let compile_models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models");
        search_dirs.push(compile_models);

        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                search_dirs.push(dir.join("models"));
                search_dirs.push(dir.join("resources").join("models"));
            }
        }

        let mut deduped = Vec::new();
        for dir in search_dirs {
            if !deduped.contains(&dir) {
                deduped.push(dir);
            }
        }

        Ok(Self {
            manifest,
            search_dirs: deduped,
            overrides: HashMap::new(),
            hash_cache: HashMap::new(),
            sessions: HashMap::new(),
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn set_override(&mut self, role: ModelRole, path: Option<PathBuf>) {
        self.sessions.remove(&role);
        match path {
            Some(p) => {
                self.overrides.insert(role, p);
            }
            None => {
                self.overrides.remove(&role);
            }
        }
    }

    pub fn load_overrides_map(&mut self, map: &HashMap<String, String>) {
        for (k, v) in map {
            if let Some(role) = ModelRole::parse_id(k) {
                let p = PathBuf::from(v);
                if p.exists() {
                    self.overrides.insert(role, p);
                }
            }
        }
    }

    pub fn export_overrides_map(&self) -> HashMap<String, String> {
        self.overrides
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_string_lossy().to_string()))
            .collect()
    }

    pub fn resolve_path(&self, role: ModelRole) -> Option<PathBuf> {
        if let Some(p) = self.overrides.get(&role) {
            if p.is_file() {
                return Some(p.clone());
            }
        }
        let entry = self.manifest.entry(role)?;
        for dir in &self.search_dirs {
            let candidate = dir.join(&entry.filename);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        None
    }

    pub fn file_sha256(&mut self, path: &Path) -> Result<String, String> {
        let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.len() > MAX_MODEL_BYTES {
            return Err(format!("Model {} exceeds 1 GiB limit", path.display()));
        }
        let modified_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        if let Some(cached) = self.hash_cache.get(path) {
            if cached.len == meta.len() && cached.modified_ns == modified_ns {
                return Ok(cached.sha256.clone());
            }
        }
        let sha = compute_file_sha256(path)?;
        self.hash_cache.insert(
            path.to_path_buf(),
            HashCacheEntry {
                len: meta.len(),
                modified_ns,
                sha256: sha.clone(),
            },
        );
        Ok(sha)
    }

    pub fn status_all(&mut self) -> Vec<ModelStatusEntry> {
        let mut out = Vec::with_capacity(ModelRole::ALL.len());
        for role in ModelRole::ALL {
            let Some(entry) = self.manifest.entry(role).cloned() else {
                continue;
            };
            let override_path = self.overrides.get(&role).cloned();
            let resolved = self.resolve_path(role);
            let loaded = self.sessions.contains_key(&role);
            let (path_str, actual_sha256, installed, hash_matches, error) = match resolved {
                Some(p) => match self.file_sha256(&p) {
                    Ok(sha) => {
                        let matches = sha.eq_ignore_ascii_case(&entry.sha256);
                        let err = if !matches {
                            Some(format!(
                                "SHA-256 mismatch for {}: expected {}, got {}",
                                entry.filename, entry.sha256, sha
                            ))
                        } else {
                            None
                        };
                        (
                            Some(p.to_string_lossy().to_string()),
                            Some(sha),
                            true,
                            matches,
                            err,
                        )
                    }
                    Err(e) => (
                        Some(p.to_string_lossy().to_string()),
                        None,
                        false,
                        false,
                        Some(e),
                    ),
                },
                None => {
                    let missing_msg = match override_path {
                        Some(op) => Some(format!("Configured model path not found: {}", op.display())),
                        None => Some(format!(
                            "Model {} not installed (run `tools/model-export` or link local .onnx)",
                            entry.filename
                        )),
                    };
                    (None, None, false, false, missing_msg)
                }
            };
            out.push(ModelStatusEntry {
                role: entry.role,
                id: entry.id,
                filename: entry.filename,
                version: entry.version,
                license: entry.license,
                expected_sha256: entry.sha256,
                expected_bytes: entry.bytes,
                download_url: entry.download_url,
                path: path_str,
                actual_sha256,
                installed,
                hash_matches,
                loaded,
                error,
            });
        }
        out
    }

    /// Loads (or returns cached) `ort::Session` for `role`, validating extension & SHA-256.
    pub fn get_session(
        &mut self,
        role: ModelRole,
        require_manifest_hash: bool,
    ) -> Result<(Arc<Mutex<Session>>, String), String> {
        let entry = self
            .manifest
            .entry(role)
            .cloned()
            .ok_or_else(|| format!("Unknown model role {}", role.as_str()))?;
        let path = self.resolve_path(role).ok_or_else(|| {
            format!(
                "Missing model for role `{}` ({}). Link it in Settings or export models first.",
                role.as_str(),
                entry.filename
            )
        })?;
        if path.extension().and_then(|v| v.to_str()) != Some("onnx") {
            return Err(format!("Expected .onnx file, got {}", path.display()));
        }
        let sha256 = self.file_sha256(&path)?;
        if require_manifest_hash && !sha256.eq_ignore_ascii_case(&entry.sha256) {
            return Err(format!(
                "Model {} failed SHA-256 check (expected {}, got {})",
                path.display(),
                entry.sha256,
                sha256
            ));
        }

        if let Some(cached) = self.sessions.get(&role) {
            if cached.path == path && cached.sha256 == sha256 {
                return Ok((Arc::clone(&cached.session), sha256));
            }
        }

        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_intra_threads(4)
            .map_err(|e| e.to_string())?
            .commit_from_file(&path)
            .map_err(|e| format!("Failed to load ONNX model {}: {e}", path.display()))?;

        let arc = Arc::new(Mutex::new(session));
        self.sessions.insert(
            role,
            CachedSession {
                session: Arc::clone(&arc),
                sha256: sha256.clone(),
                path,
            },
        );
        Ok((arc, sha256))
    }
}

pub fn compute_file_sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn atomic_replace(temp: &Path, dest: &Path) -> Result<(), String> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        #[link(name = "kernel32")]
        extern "system" {
            fn MoveFileExW(from: *const u16, to: *const u16, flags: u32) -> i32;
        }
        let from: Vec<u16> = temp.as_os_str().encode_wide().chain(Some(0)).collect();
        let to: Vec<u16> = dest.as_os_str().encode_wide().chain(Some(0)).collect();
        if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), 0x1 | 0x8) } == 0 {
            let err = std::io::Error::last_os_error();
            let _ = std::fs::remove_file(temp);
            return Err(err.to_string());
        }
    }
    #[cfg(not(windows))]
    std::fs::rename(temp, dest).map_err(|e| {
        let _ = std::fs::remove_file(temp);
        e.to_string()
    })?;
    Ok(())
}

pub fn atomic_write_bytes(dest: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = dest.parent().ok_or("Invalid destination path")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let filename = dest
        .file_name()
        .ok_or("Invalid destination filename")?
        .to_string_lossy();
    let temp = parent.join(format!(".{filename}.flat2rig-{}.tmp", std::process::id()));
    let _ = std::fs::remove_file(&temp);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|e| e.to_string())?;
    let write_res = file.write_all(bytes).and_then(|_| file.sync_all());
    drop(file);
    if let Err(e) = write_res {
        let _ = std::fs::remove_file(&temp);
        return Err(e.to_string());
    }
    atomic_replace(&temp, dest)
}

/// Downloads a direct `.onnx` URL declared in `models/manifest.json` and verifies SHA-256 before atomic install.
pub fn download_and_install_model(role_str: &str, dest_dir: &Path) -> Result<PathBuf, String> {
    let role = ModelRole::parse_id(role_str)
        .ok_or_else(|| format!("Unknown model role `{role_str}`"))?;
    let manifest = ModelManifest::embedded()?;
    let entry = manifest
        .entry(role)
        .ok_or_else(|| format!("Model role `{role_str}` missing from manifest.json"))?;
    let Some(ref url) = entry.download_url else {
        return Err(format!(
            "Model `{}` ({}) has no direct ONNX downloadUrl. Run `tools/model-export/export_models.py` or link a local .onnx file.",
            entry.id, entry.filename
        ));
    };
    if !url.starts_with("https://") {
        return Err(format!("Refusing non-HTTPS model URL: {url}"));
    }
    std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    let dest = dest_dir.join(&entry.filename);
    let temp = dest_dir.join(format!(
        ".{}.download-{}.tmp",
        entry.filename,
        std::process::id()
    ));
    let _ = std::fs::remove_file(&temp);

    let result: Result<String, String> = (|| {
        let mut resp = ureq::get(url)
            .call()
            .map_err(|e| format!("Failed downloading {}: {e}", entry.filename))?;
        let mut reader = resp.body_mut().as_reader();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| e.to_string())?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 64 * 1024];
        let mut total = 0u64;
        loop {
            let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            total += n as u64;
            if total > MAX_MODEL_BYTES {
                return Err(format!("Downloaded file for {} exceeded size limit", entry.filename));
            }
            hasher.update(&buf[..n]);
            file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        }
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        Ok(format!("{:x}", hasher.finalize()))
    })();

    let actual_sha = match result {
        Ok(sha) => sha,
        Err(e) => {
            let _ = std::fs::remove_file(&temp);
            return Err(e);
        }
    };
    if !actual_sha.eq_ignore_ascii_case(&entry.sha256) {
        let _ = std::fs::remove_file(&temp);
        return Err(format!(
            "Downloaded {} failed SHA-256 check (expected {}, got {})",
            entry.filename, entry.sha256, actual_sha
        ));
    }
    atomic_replace(&temp, &dest)?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parses_all_five_roles() {
        let manifest = ModelManifest::embedded().unwrap();
        for role in ModelRole::ALL {
            assert!(manifest.entry(role).is_some(), "missing {}", role.as_str());
        }
    }

    #[test]
    fn load_exported_onnx_models_when_present() {
        let mut reg = ModelRegistry::new(Vec::new()).unwrap();
        // Verify each exported model loads into ort cleanly if present in `models/`.
        for role in [
            ModelRole::FaceDetector,
            ModelRole::Landmark28,
            ModelRole::PartEncoder,
            ModelRole::PartDecoder,
        ] {
            if reg.resolve_path(role).is_some() {
                let (_sess, sha) = reg.get_session(role, true).unwrap();
                assert_eq!(sha.len(), 64);
            }
        }
    }
}
