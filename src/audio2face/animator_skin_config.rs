//! `AnimatorSkinConfig` — Rust port of `AnimatorSkinConfig.cs`.
//!
//! Loaded from the bundle's `model_config_<Identity>.json`. Carries
//! the per-identity strength/offset values that compose the model's
//! raw skin output into the final masked-vertex delta consumed by the
//! solver. Without this, the C# delta formula
//! `NeutralSkin[v] + composed[v] - NeutralFlat[m]` collapses to
//! `-NeutralFlat[m]` (rough magnitude ≈ 100s in mesh units), which
//! saturates the PGD/BVLS solver against its `[0,1]` box every frame.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum AnimatorSkinConfigError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Deserialize)]
struct Envelope {
    config: RawConfig,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    skin_strength: f32,
    eyelid_open_offset: f32,
    lip_open_offset: f32,
    #[serde(default = "default_blink")]
    blink_strength: f32,
    #[serde(default)]
    upper_face_smoothing: f32,
    #[serde(default)]
    lower_face_smoothing: f32,
    #[serde(default)]
    face_mask_level: f32,
    #[serde(default)]
    face_mask_softness: f32,
}

fn default_blink() -> f32 {
    1.0
}

/// Per-identity animator config. The fields the solver uses today are
/// `skin_strength`, `eyelid_open_offset`, and `lip_open_offset` — the
/// rest are kept for parity with the C# struct and future renderer
/// work (vertex EMA, face mask).
#[derive(Debug, Clone, Copy)]
pub struct AnimatorSkinConfig {
    pub skin_strength: f32,
    pub eyelid_open_offset: f32,
    pub lip_open_offset: f32,
    pub blink_strength: f32,
    pub upper_face_smoothing: f32,
    pub lower_face_smoothing: f32,
    pub face_mask_level: f32,
    pub face_mask_softness: f32,
}

impl AnimatorSkinConfig {
    pub fn from_json(json: &str) -> Result<Self, AnimatorSkinConfigError> {
        let env: Envelope = serde_json::from_str(json)?;
        let c = env.config;
        Ok(Self {
            skin_strength: c.skin_strength,
            eyelid_open_offset: c.eyelid_open_offset,
            lip_open_offset: c.lip_open_offset,
            blink_strength: c.blink_strength,
            upper_face_smoothing: c.upper_face_smoothing,
            lower_face_smoothing: c.lower_face_smoothing,
            face_mask_level: c.face_mask_level,
            face_mask_softness: c.face_mask_softness,
        })
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, AnimatorSkinConfigError> {
        let json = std::fs::read_to_string(path.as_ref())?;
        Self::from_json(&json)
    }
}

impl Default for AnimatorSkinConfig {
    fn default() -> Self {
        Self {
            skin_strength: 1.0,
            eyelid_open_offset: 0.0,
            lip_open_offset: 0.0,
            blink_strength: 1.0,
            upper_face_smoothing: 0.0,
            lower_face_smoothing: 0.0,
            face_mask_level: 0.0,
            face_mask_softness: 0.0,
        }
    }
}
