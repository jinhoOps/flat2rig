//! Rig-level derived data shared by hidden-area generation, evaluation, and verification.
//!
//! - [`seams`]: T13a seam contracts (`SeamGraph`), the shared pose envelope, numerically
//!   sampled per-seam underlaps, and the seam coverage check.
//! - [`exposure`]: T13b exposure map — hidden texels revealed over the pose envelope (ArtMesh render).
pub mod exposure;
pub mod seams;
