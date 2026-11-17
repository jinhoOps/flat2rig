//! Rig-level derived data shared by hidden-area generation, evaluation, and verification.
//!
//! - [`seams`]: T13a seam contracts (`SeamGraph`), the shared pose envelope, numerically
//!   sampled per-seam underlaps, and the seam coverage check.
pub mod seams;
