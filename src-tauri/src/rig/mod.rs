//! Rig-level derived data shared by hidden-area generation, evaluation, and verification.
//!
//! - [`seams`]: T13a seam contracts (`SeamGraph`), the shared pose envelope, numerically
//!   sampled per-seam underlaps, and the seam coverage check.
//! - [`exposure`]: T13b exposure map — hidden texels revealed over the pose envelope (ArtMesh render).
//! - [`fidelity`]: T14 hidden-area fidelity harness (evaluation only) — ΔE2000 / palette / seam
//!   step / stroke / texture metrics inside exposure bands (PSD ground truth + seam holdout).
//! - [`eye_seal`]: T10 eye seal checker — white/iris leaks, lid seam gaps and lid-line breaks over the
//!   expression × gaze poses (same raster and clipping rule as the exposure map).
pub mod exposure;
pub mod eye_seal;
pub mod fidelity;
pub mod seams;
