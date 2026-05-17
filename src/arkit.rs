//! ARKit-52 blendshape constants, `BlendshapeFrame` envelope, and
//! `ArkitSmoother` — inlined from
//! `remotemedia-core::nodes::lip_sync::{blendshape, arkit_smoother}`.
//!
//! These types are kept private to this plugin so the host crate
//! doesn't have to re-export them publicly for plugins to consume.

use crate::Error;
use serde_json::{json, Value};

/// Number of ARKit blendshapes (canonical face animation set).
pub const ARKIT_52: usize = 52;

/// ARKit blendshape names in canonical order. Index `i` of the
/// `arkit_52` array corresponds to `ARKIT_BLENDSHAPE_NAMES[i]`.
///
/// Mirrors Apple's `ARFaceAnchor.BlendShapeLocation` enumeration.
pub const ARKIT_BLENDSHAPE_NAMES: [&str; ARKIT_52] = [
    // Eyes (8)
    "eyeBlinkLeft",
    "eyeLookDownLeft",
    "eyeLookInLeft",
    "eyeLookOutLeft",
    "eyeLookUpLeft",
    "eyeSquintLeft",
    "eyeWideLeft",
    "eyeBlinkRight",
    // (continuing — index 8..16)
    "eyeLookDownRight",
    "eyeLookInRight",
    "eyeLookOutRight",
    "eyeLookUpRight",
    "eyeSquintRight",
    "eyeWideRight",
    // Jaw / mouth area (4)
    "jawForward",
    "jawLeft",
    // index 16..27
    "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    // index 28..39
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    // index 40..47
    "mouthUpperUpRight",
    // Brows (4)
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    // Cheeks (3)
    "cheekPuff",
    "cheekSquintLeft",
    // index 48..51
    "cheekSquintRight",
    // Nose (2)
    "noseSneerLeft",
    "noseSneerRight",
    // Tongue (1)
    "tongueOut",
];

const _ASSERT_ARKIT_NAMES_LEN: () = assert!(ARKIT_BLENDSHAPE_NAMES.len() == ARKIT_52);

/// One timed blendshape keyframe — the unit a `LipSyncNode` emits
/// per output tick. Renderer treats consecutive keyframes as a
/// sampleable timeline keyed by `pts_ms` (audio playback time).
#[derive(Debug, Clone, PartialEq)]
pub struct BlendshapeFrame {
    /// 52 ARKit blendshape activations, indexed per
    /// [`ARKIT_BLENDSHAPE_NAMES`]. Values are not strictly bounded —
    /// the persona-engine's Audio2Face model emits raw predictions in
    /// roughly `[-1.0, 2.0]` and the PGD/BVLS solver clips to `[0, 1]`
    /// for animation. We don't enforce bounds at the envelope level
    /// because future solvers / phoneme impls may use other ranges.
    pub arkit_52: [f32; ARKIT_52],
    /// Presentation timestamp (ms) — matches the audio frame the
    /// keyframe was derived from, NOT wall time. Renderer samples the
    /// keyframe ring against the `audio.out.clock` tap.
    pub pts_ms: u64,
    /// Conversational turn id, forwarded if upstream metadata had one.
    /// Lets the renderer group blendshapes by turn for diagnostics or
    /// barge handling, without the lip-sync node tracking turns.
    pub turn_id: Option<u64>,
}

impl BlendshapeFrame {
    /// Build a frame; the array is borrowed in by value.
    pub fn new(arkit_52: [f32; ARKIT_52], pts_ms: u64, turn_id: Option<u64>) -> Self {
        Self {
            arkit_52,
            pts_ms,
            turn_id,
        }
    }

    /// All-zero blendshapes — the neutral pose.
    pub fn neutral(pts_ms: u64) -> Self {
        Self::new([0.0; ARKIT_52], pts_ms, None)
    }

    /// Encode the frame as the canonical `RuntimeData::Json` payload.
    pub fn to_json(&self) -> Value {
        let mut v = json!({
            "kind": "blendshapes",
            "arkit_52": self.arkit_52.as_slice(),
            "pts_ms": self.pts_ms,
        });
        if let Some(turn) = self.turn_id {
            v["turn_id"] = json!(turn);
        }
        v
    }

    /// Inverse of [`Self::to_json`]. Tolerant: missing `turn_id` is
    /// fine; arrays of the wrong length are an error.
    pub fn from_json(v: &Value) -> Result<Self, Error> {
        let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        if kind != "blendshapes" {
            return Err(Error::InvalidData(format!(
                "BlendshapeFrame::from_json: expected kind='blendshapes', got {:?}",
                kind
            )));
        }
        let arr = v
            .get("arkit_52")
            .and_then(|a| a.as_array())
            .ok_or_else(|| Error::InvalidData("missing arkit_52 array".into()))?;
        if arr.len() != ARKIT_52 {
            return Err(Error::InvalidData(format!(
                "arkit_52 must have {} entries, got {}",
                ARKIT_52,
                arr.len()
            )));
        }
        let mut arkit_52 = [0.0f32; ARKIT_52];
        for (i, item) in arr.iter().enumerate() {
            arkit_52[i] = item
                .as_f64()
                .ok_or_else(|| Error::InvalidData(format!("arkit_52[{}] is not a number", i)))?
                as f32;
        }
        let pts_ms = v
            .get("pts_ms")
            .and_then(|p| p.as_u64())
            .ok_or_else(|| Error::InvalidData("missing or non-u64 pts_ms".into()))?;
        let turn_id = v.get("turn_id").and_then(|t| t.as_u64());
        Ok(Self::new(arkit_52, pts_ms, turn_id))
    }
}

/// Uniform EMA smoother over the 52-element ARKit blendshape vector.
///
/// `out[i] = alpha · prev[i] + (1 - alpha) · in[i]` for each `i ∈ [0, 52)`.
/// - `alpha = 0` → no smoothing (passthrough).
/// - `alpha = 1` → infinite smoothing (output = prev forever; degenerate).
/// - typical `alpha ∈ [0.1, 0.4]` matches persona-engine's per-axis tunings.
///
/// First frame passes through unchanged (no `prev` to mix with).
pub struct ArkitSmoother {
    alpha: f32,
    prev: [f32; ARKIT_52],
    has_prev: bool,
}

impl ArkitSmoother {
    /// Build a smoother. `alpha` is clamped to `[0, 1]`.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            prev: [0.0; ARKIT_52],
            has_prev: false,
        }
    }

    /// Apply EMA to one frame. First call passes through unchanged.
    pub fn smooth(&mut self, input: &[f32; ARKIT_52]) -> [f32; ARKIT_52] {
        if !self.has_prev {
            self.has_prev = true;
            self.prev = *input;
            return *input;
        }
        let mut out = [0.0f32; ARKIT_52];
        let a = self.alpha;
        let inv = 1.0 - a;
        for i in 0..ARKIT_52 {
            out[i] = a * self.prev[i] + inv * input[i];
        }
        self.prev = out;
        out
    }

    /// Reset state — the next `smooth()` call will pass through.
    /// Used by `barge_in` so the smoother doesn't pull the new turn
    /// toward the prior turn's frozen pose.
    pub fn reset(&mut self) {
        self.prev = [0.0; ARKIT_52];
        self.has_prev = false;
    }

    /// Snapshot current state for save/restore.
    pub fn save(&self) -> ([f32; ARKIT_52], bool) {
        (self.prev, self.has_prev)
    }

    /// Restore from a snapshot.
    pub fn restore(&mut self, state: ([f32; ARKIT_52], bool)) {
        self.prev = state.0;
        self.has_prev = state.1;
    }
}
