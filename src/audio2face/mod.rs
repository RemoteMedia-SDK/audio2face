//! Audio2Face math + inference + data-loader kernels.
//!
//! Ported verbatim from `remotemedia-core::nodes::lip_sync::audio2face`
//! into the standalone `audio2face-plugin` cdylib. No behavioural
//! changes — this is a mechanical extraction so the host crate can
//! drop its `ort`/`zip`/`ndarray` dependencies for Audio2Face.

pub mod animator_skin_config;
pub mod blendshape_data;
pub mod bvls_solver;
pub mod identity;
pub mod inference;
pub mod npy;
pub mod npz;
pub mod pgd_solver;
pub mod response_curves;
pub mod solver_math;
pub mod solver_trait;

pub use animator_skin_config::{AnimatorSkinConfig, AnimatorSkinConfigError};
pub use blendshape_data::{BlendshapeConfig, BlendshapeData, BlendshapeDataError};
pub use bvls_solver::BvlsBlendshapeSolver;
pub use identity::{Audio2FaceIdentity, BundlePaths};
pub use inference::{Audio2FaceInference, Audio2FaceOutput};
pub use pgd_solver::PgdBlendshapeSolver;
pub use solver_math::{
    apply_regularization, compute_bounding_box_diagonal, compute_dt_d, compute_transpose,
    L1_MULTIPLIER, L2_MULTIPLIER, TEMPORAL_MULTIPLIER,
};
pub use solver_trait::BlendshapeSolver;
