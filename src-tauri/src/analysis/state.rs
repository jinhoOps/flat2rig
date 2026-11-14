use super::models::{atomic_write_bytes, ModelRegistry};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct SamEmbeddingCache {
    pub image_hash: String,
    pub model_hash: String,
    pub preprocess_version: String,
    pub crop_key: Option<String>,
    pub orig_w: usize,
    pub orig_h: usize,
    pub resized_w: usize,
    pub resized_h: usize,
    pub embeddings: Vec<f32>,
}

/// Shared Tauri state for ONNX models, job cancellation, serialization, and embedding cache.
pub struct AnalysisState {
    pub registry: Mutex<ModelRegistry>,
    pub job_lock: Mutex<()>,
    pub active_job_id: AtomicU64,
    pub cancel_requested: AtomicBool,
    pub config_path: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
    pub sam_cache: Mutex<Option<SamEmbeddingCache>>,
    pub sam_crop_cache: Mutex<Option<SamEmbeddingCache>>,
}

impl AnalysisState {
    pub fn new(app_data_dir: Option<PathBuf>) -> Result<Self, String> {
        let mut search_dirs = Vec::new();
        let models_dir = app_data_dir.as_ref().map(|d| d.join("models"));
        if let Some(ref md) = models_dir {
            search_dirs.push(md.clone());
        }
        let mut registry = ModelRegistry::new(search_dirs)?;

        let config_path = app_data_dir.map(|d| d.join("models-config.json"));
        if let Some(ref cfg) = config_path {
            if let Ok(bytes) = std::fs::read(cfg) {
                if let Ok(overrides) = serde_json::from_slice::<HashMap<String, String>>(&bytes) {
                    registry.load_overrides_map(&overrides);
                }
            }
        }

        Ok(Self {
            registry: Mutex::new(registry),
            job_lock: Mutex::new(()),
            active_job_id: AtomicU64::new(0),
            cancel_requested: AtomicBool::new(false),
            config_path,
            models_dir,
            sam_cache: Mutex::new(None),
            sam_crop_cache: Mutex::new(None),
        })
    }

    pub fn persist_config(&self) -> Result<(), String> {
        let Some(ref cfg_path) = self.config_path else {
            return Ok(());
        };
        if let Some(parent) = cfg_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let overrides = self
            .registry
            .lock()
            .map_err(|_| "Model registry lock poisoned".to_string())?
            .export_overrides_map();
        let bytes = serde_json::to_vec_pretty(&overrides).map_err(|e| e.to_string())?;
        atomic_write_bytes(cfg_path, &bytes)
    }

    pub fn begin_job(&self, requested_id: Option<u64>) -> u64 {
        self.cancel_requested.store(false, Ordering::SeqCst);
        let id = requested_id.unwrap_or_else(|| {
            self.active_job_id.fetch_add(1, Ordering::SeqCst) + 1
        });
        self.active_job_id.store(id, Ordering::SeqCst);
        id
    }

    pub fn is_stale_or_cancelled(&self, job_id: u64) -> bool {
        self.cancel_requested.load(Ordering::SeqCst)
            || self.active_job_id.load(Ordering::SeqCst) != job_id
    }
}

pub fn hash_rgba_bytes(rgba: &[u8], width: usize, height: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update((width as u64).to_le_bytes());
    hasher.update((height as u64).to_le_bytes());
    hasher.update(rgba);
    format!("{:x}", hasher.finalize())
}
