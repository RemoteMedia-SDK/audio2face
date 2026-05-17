//! `BlendshapeConfig` + `BlendshapeData` — Rust port of
//! [`external/.../Audio2Face/BlendshapeData.cs`](../../../../../../external/handcrafted-persona-engine/src/PersonaEngine/PersonaEngine.Lib/TTS/Synthesis/LipSync/Audio2Face/BlendshapeData.cs).
//!
//! Parses the per-identity `bs_skin_config_<Identity>.json` regularization
//! params + active-pose mask, and assembles the dense `[V*3, K]` delta
//! matrix the `PgdBlendshapeSolver` / `BvlsBlendshapeSolver` consume,
//! by reading the per-blendshape NPYs out of `bs_skin_<Identity>.npz`
//! and the `model_data_<Identity>.npz` extras (neutral skin, eye/lip
//! pose deltas, saccade).
//!
//! The 52 ARKit blendshape names match the canonical order in
//! [`crate::arkit::ARKIT_BLENDSHAPE_NAMES`] — the C# version embeds
//! the same array verbatim, but we reuse the constant we already ship.

use super::npz::{NpzArchive, NpzError};
use crate::arkit::ARKIT_BLENDSHAPE_NAMES;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Errors surfaced by the blendshape data loader.
#[derive(Debug, thiserror::Error)]
pub enum BlendshapeDataError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json parse error in {context}: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("npz error: {0}")]
    Npz(#[from] NpzError),

    #[error("invalid bs_skin_config: {0}")]
    Config(String),

    #[error("invalid blendshape data: {0}")]
    Data(String),
}

/// Mirror of the JSON written by `bs_skin_config_<Identity>.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ConfigEnvelope {
    blendshape_params: BlendshapeParams,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct BlendshapeParams {
    num_poses: usize,
    template_b_b_size: f32,
    strength_l1regularization: f32,
    strength_l2regularization: f32,
    strength_temporal_smoothing: f32,
    strength_symmetry: f32,
    bs_solve_active_poses: Vec<i32>,
    /// Multipliers and offsets are applied after solve to produce the
    /// final ARKit blendshape values (length 52).
    bs_weight_multipliers: Vec<f32>,
    bs_weight_offsets: Vec<f32>,
}

/// Parsed blendshape solver config — drives the PGD/BVLS regularization
/// strengths and selects which of the 52 ARKit blendshapes are active
/// for a given identity.
#[derive(Debug, Clone)]
pub struct BlendshapeConfig {
    /// Total ARKit blendshape count (always 52 in this bundle).
    pub num_poses: usize,
    /// Reference face bounding-box diagonal — used to normalize
    /// regularization across face sizes.
    pub template_bb_size: f32,
    pub strength_l1: f32,
    pub strength_l2: f32,
    pub strength_temporal: f32,
    pub strength_symmetry: f32,
    /// Indices `i ∈ [0, 52)` where `bs_solve_active_poses[i] == 1`.
    /// These are the K blendshapes the solver runs against; their
    /// names map to [`ARKIT_BLENDSHAPE_NAMES[i]`].
    pub active_indices: Vec<usize>,
    /// Length-52 multiplier applied per ARKit blendshape after solve.
    pub multipliers: Vec<f32>,
    /// Length-52 offset applied per ARKit blendshape after solve.
    pub offsets: Vec<f32>,
}

impl BlendshapeConfig {
    /// Parse from a JSON string. Mirrors `BlendshapeConfig.FromJson` in C#.
    pub fn from_json(json: &str) -> Result<Self, BlendshapeDataError> {
        let env: ConfigEnvelope =
            serde_json::from_str(json).map_err(|e| BlendshapeDataError::Json {
                context: "bs_skin_config".to_string(),
                source: e,
            })?;
        let bp = env.blendshape_params;
        if bp.bs_solve_active_poses.len() != bp.num_poses {
            return Err(BlendshapeDataError::Config(format!(
                "bs_solve_active_poses length ({}) != num_poses ({})",
                bp.bs_solve_active_poses.len(),
                bp.num_poses
            )));
        }
        if bp.bs_weight_multipliers.len() != bp.num_poses
            || bp.bs_weight_offsets.len() != bp.num_poses
        {
            return Err(BlendshapeDataError::Config(format!(
                "weight multiplier/offset length must equal num_poses ({})",
                bp.num_poses
            )));
        }
        let active_indices = bp
            .bs_solve_active_poses
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v == 1 { Some(i) } else { None })
            .collect();
        Ok(Self {
            num_poses: bp.num_poses,
            template_bb_size: bp.template_b_b_size,
            strength_l1: bp.strength_l1regularization,
            strength_l2: bp.strength_l2regularization,
            strength_temporal: bp.strength_temporal_smoothing,
            strength_symmetry: bp.strength_symmetry,
            active_indices,
            multipliers: bp.bs_weight_multipliers,
            offsets: bp.bs_weight_offsets,
        })
    }

    /// Read + parse from a JSON file on disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, BlendshapeDataError> {
        let json = std::fs::read_to_string(path.as_ref())?;
        Self::from_json(&json)
    }

    /// Active count K — the dimension the solver works in.
    pub fn active_count(&self) -> usize {
        self.active_indices.len()
    }
}

/// Loaded blendshape basis — masked neutral pose + dense `[V*3, K]`
/// delta matrix that the PGD/BVLS solvers consume.
#[derive(Debug)]
pub struct BlendshapeData {
    /// Masked neutral vertex positions, flat `[M*3]`.
    pub neutral_flat: Vec<f32>,
    /// Delta matrix `[V*3, K]` row-major. `V*3` rows (masked vertex
    /// components), `K` columns (active blendshapes per
    /// [`BlendshapeConfig::active_indices`]).
    pub delta_matrix: Vec<f32>,
    /// Number of masked vertex components — `M * 3`.
    pub masked_position_count: usize,
    /// Number of active blendshapes (K).
    pub active_count: usize,
    /// Vertex indices from the frontal mask — length M.
    pub frontal_mask: Vec<i32>,
    /// Full neutral skin vertex positions `[V*3]` from `model_data` NPZ.
    pub neutral_skin_flat: Vec<f32>,
    /// Eye-close pose delta `[24002*3]` from `model_data` NPZ.
    pub eye_close_pose_delta_flat: Vec<f32>,
    /// Lip-open pose delta `[24002*3]` from `model_data` NPZ.
    pub lip_open_pose_delta_flat: Vec<f32>,
    /// Optional saccade rotation matrix `[N, 2]` flattened to
    /// length `2N`, in row-major order. Not all model_data NPZs
    /// include this (older bundles).
    pub saccade_rot_flat: Option<Vec<f32>>,
    /// Saccade row count (N) when `saccade_rot_flat` is `Some`.
    pub saccade_rot_rows: Option<usize>,
}

impl BlendshapeData {
    /// Load from per-identity NPZ paths plus the parsed config.
    /// Mirrors `BlendshapeData.Load` in C# but takes already-resolved
    /// paths from [`super::identity::BundlePaths`].
    pub fn load(
        bs_skin_npz: impl AsRef<Path>,
        model_data_npz: impl AsRef<Path>,
        config: &BlendshapeConfig,
    ) -> Result<Self, BlendshapeDataError> {
        let mut skin = NpzArchive::open(bs_skin_npz.as_ref())?;

        // 1. Neutral pose + frontal mask.
        let neutral = skin.read_f32("neutral.npy")?;
        let frontal_mask = skin.read_i32("frontalMask.npy")?;

        let masked_vertex_count = frontal_mask.data.len();
        let masked_position_count = masked_vertex_count * 3;
        let active_count = config.active_count();

        // 2. Build masked neutral by gather.
        let mut neutral_flat = vec![0.0f32; masked_position_count];
        for m in 0..masked_vertex_count {
            let vi = frontal_mask.data[m] as usize;
            if vi * 3 + 2 >= neutral.data.len() {
                return Err(BlendshapeDataError::Data(format!(
                    "frontal_mask index {vi} out of bounds for neutral.npy of \
                     length {}",
                    neutral.data.len()
                )));
            }
            neutral_flat[m * 3] = neutral.data[vi * 3];
            neutral_flat[m * 3 + 1] = neutral.data[vi * 3 + 1];
            neutral_flat[m * 3 + 2] = neutral.data[vi * 3 + 2];
        }

        // 3. Delta matrix [V*3, K]. For each active blendshape index k,
        //    read the corresponding `<arkit_name>.npy`, gather to the
        //    masked vertex set, and write column k row-major.
        let mut delta_matrix = vec![0.0f32; masked_position_count * active_count];
        for (k, &pose_index) in config.active_indices.iter().enumerate() {
            if pose_index >= ARKIT_BLENDSHAPE_NAMES.len() {
                return Err(BlendshapeDataError::Data(format!(
                    "active pose index {pose_index} out of range for ARKit-52"
                )));
            }
            let name = ARKIT_BLENDSHAPE_NAMES[pose_index];
            let entry = format!("{name}.npy");
            let blend = skin.read_f32(&entry)?;
            for m in 0..masked_vertex_count {
                let vi = frontal_mask.data[m] as usize;
                if vi * 3 + 2 >= blend.data.len() {
                    return Err(BlendshapeDataError::Data(format!(
                        "frontal_mask index {vi} out of bounds for {entry} of \
                         length {}",
                        blend.data.len()
                    )));
                }
                let row_x = m * 3;
                let row_y = m * 3 + 1;
                let row_z = m * 3 + 2;
                delta_matrix[row_x * active_count + k] = blend.data[vi * 3];
                delta_matrix[row_y * active_count + k] = blend.data[vi * 3 + 1];
                delta_matrix[row_z * active_count + k] = blend.data[vi * 3 + 2];
            }
        }

        // 4. Model-data NPZ: neutral skin + eye/lip pose deltas + optional saccade.
        let mut model_data = NpzArchive::open(model_data_npz.as_ref())?;
        let neutral_skin_flat = model_data.read_f32("neutral_skin.npy")?.data;
        let eye_close_pose_delta_flat = model_data.read_f32("eye_close_pose_delta.npy")?.data;
        let lip_open_pose_delta_flat = model_data.read_f32("lip_open_pose_delta.npy")?.data;

        let (saccade_rot_flat, saccade_rot_rows) = if model_data.has_entry("saccade_rot_matrix.npy")
        {
            let sac = model_data.read_f32("saccade_rot_matrix.npy")?;
            let rows = sac.data.len() / 2;
            (Some(sac.data), Some(rows))
        } else {
            (None, None)
        };

        Ok(Self {
            neutral_flat,
            delta_matrix,
            masked_position_count,
            active_count,
            frontal_mask: frontal_mask.data,
            neutral_skin_flat,
            eye_close_pose_delta_flat,
            lip_open_pose_delta_flat,
            saccade_rot_flat,
            saccade_rot_rows,
        })
    }
}
