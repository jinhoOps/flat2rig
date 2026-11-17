//! Rig-level derived data shared by hidden-area generation, evaluation, and verification.
//!
//! - [`seams`]: T13a seam contracts (`SeamGraph`), the shared pose envelope, numerically
//!   sampled per-seam underlaps, and the seam coverage check.
//! - [`exposure`]: T13b exposure map — hidden texels revealed over the pose envelope (ArtMesh render).
//! - [`fidelity`]: T14 hidden-area fidelity harness (evaluation only) — ΔE2000 / palette / seam
//!   step / stroke / texture metrics inside exposure bands (PSD ground truth + seam holdout).
pub mod exposure;
pub mod fidelity;
pub mod seams;
