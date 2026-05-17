//! `Audio2FaceInference` — Rust port of
//! [`external/.../Audio2Face/Audio2FaceInference.cs`](../../../../../../external/handcrafted-persona-engine/src/PersonaEngine/PersonaEngine.Lib/TTS/Synthesis/LipSync/Audio2Face/Audio2FaceInference.cs).
//!
//! Wraps the Audio2Face-3D v3.0 ONNX network via `ort`. Each call
//! produces 30 center frames of skin vertex deltas + eye rotations
//! from a 1-second 16 kHz audio window, with the recurrent GRU state
//! carried across calls.
//!
//! ## I/O contract (lifted from `network_info.json` + the C# constants)
//!
//! Input tensors (binding names match the ONNX graph):
//! - `window`: `[1, 16000]` f32 — 1 sec audio at 16 kHz, zero-padded if short.
//! - `identity`: `[1, 3]` f32 — one-hot for Claire/James/Mark.
//! - `emotion`: `[1, 30, 10]` f32 — per-center-frame emotion vector;
//!   we leave at zero for now (neutral expression).
//! - `input_latents`: `[2, 2, 1, 256]` f32 — GRU recurrent state.
//! - `noise`: `[1, 3, 60, 88831]` f32 — Box-Muller Gaussian, fixed
//!   seed 0 for reproducibility.
//!
//! Output tensors:
//! - `prediction`: `[1, 60, 88831]` f32 — full skin+tongue+jaw+eyes
//!   per total-frame (15 left pad + 30 center + 15 right pad).
//! - `output_latents`: `[2, 2, 1, 256]` f32 — new GRU state.
//!
//! We extract from the prediction's 30 center frames:
//! - **skin**: bytes `[0 .. 72006)` per frame (24002 vertex × 3 components)
//! - **eyes**: bytes `[88827 .. 88831)` per frame (4 floats:
//!   `right_x`, `right_y`, `left_x`, `left_y`)

use crate::{Error, Result};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProviderDispatch,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

/// Audio buffer length: 1 second at 16 kHz.
pub const AUDIO_BUFFER_LEN: usize = 16_000;
/// Emotion vector dimensions (amazement … sadness, ten total).
pub const NUM_EMOTIONS: usize = 10;
/// Number of center frames extracted from the prediction.
pub const NUM_CENTER_FRAMES: usize = 30;
/// Number of identity slots (one-hot dim).
pub const NUM_IDENTITIES: usize = 3;
/// Number of diffusion denoising steps.
pub const NUM_DIFFUSION_STEPS: usize = 2;
/// Number of GRU layers.
pub const NUM_GRU_LAYERS: usize = 2;
/// Hidden dim of each GRU layer.
pub const GRU_LATENT_DIM: usize = 256;
/// Total frames per inference window (15 left pad + 30 center + 15 right pad).
pub const TOTAL_FRAMES: usize = 60;
/// Total output dim per frame (skin + tongue + jaw + eyes).
pub const TOTAL_OUTPUT_DIM: usize = 88_831;
/// Skin vertex data size per frame (24002 vertices × 3 components).
pub const SKIN_SIZE: usize = 72_006;
/// Eye rotation data: `[right_x, right_y, left_x, left_y]` per frame.
pub const EYES_SIZE: usize = 4;
/// Offset where eye rotation starts (after skin + tongue + jaw).
pub const EYES_OFFSET: usize = 72_006 + 16_806 + 15;
/// Frames to skip at the start of the prediction (left padding).
pub const LEFT_PAD_FRAMES: usize = 15;
/// Box-Muller seed — fixed so two consecutive calls with identical
/// inputs produce identical outputs (matches C#'s `NoiseSeed = 0`).
pub const NOISE_SEED: u64 = 0;

/// Total noise tensor element count: `[1, 3, 60, 88831]`.
const NOISE_TENSOR_LEN: usize = 1 * (NUM_DIFFUSION_STEPS + 1) * TOTAL_FRAMES * TOTAL_OUTPUT_DIM;

/// One inference output — 30 frames of skin + eye data.
#[derive(Debug, Clone)]
pub struct Audio2FaceOutput {
    /// `[NUM_CENTER_FRAMES * SKIN_SIZE]` flat — skin vertex deltas
    /// per center frame, frame-major.
    pub skin_flat: Vec<f32>,
    /// `[NUM_CENTER_FRAMES * EYES_SIZE]` flat — eye rotation angles
    /// per center frame.
    pub eye_flat: Vec<f32>,
    /// Number of center frames returned. Always [`NUM_CENTER_FRAMES`]
    /// in normal operation; included so future variable-frame impls
    /// can stay backward-compatible.
    pub frame_count: usize,
}

/// Audio2Face inference wrapper. Holds the ONNX session, the
/// pre-generated noise tensor, and the recurrent GRU hidden state
/// across calls.
pub struct Audio2FaceInference {
    session: Session,
    /// Recurrent GRU hidden state `[NUM_DIFFUSION_STEPS, NUM_GRU_LAYERS,
    /// 1, GRU_LATENT_DIM]`. Updated each `infer()` from the model's
    /// `output_latents` so successive calls maintain temporal context.
    gru_state: Vec<f32>,
    /// Pre-generated Box-Muller noise — large (~64 MB) and constant
    /// across calls. Cached to avoid regenerating per inference; the
    /// model's "diffusion" denoising uses the same noise field.
    noise: Vec<f32>,
}

impl Audio2FaceInference {
    /// Build an inference object from `network.onnx` on disk.
    ///
    /// When `use_gpu` is true the EP fallback list is
    /// `[CUDA, CoreML, CPU]` — ort tries each in order and silently
    /// drops the ones the host can't satisfy (CUDA on a Mac,
    /// CoreML on Linux, CUDA on a CUDA-less Linux box). CPU is the
    /// terminal fallback so loading never fails for lack of an EP,
    /// only for an actually-broken model.
    ///
    /// On WSL2 + RTX 40-series with CUDA 12.x present, CUDA EP cuts
    /// per-window inference from ~700–900 ms to ~80–150 ms, which
    /// is what makes the per-window blendshape burst feel ≈
    /// continuous to a viewer (one batch every ~250 ms with stride
    /// shrinkage, instead of every 1000 ms).
    pub fn load(model_path: impl AsRef<Path>, use_gpu: bool) -> Result<Self> {
        let providers: Vec<ExecutionProviderDispatch> = if use_gpu {
            tracing::info!(
                target: "audio2face",
                "Audio2Face: requesting GPU execution providers (CUDA → CoreML → CPU fallback)"
            );
            vec![
                CUDAExecutionProvider::default().build(),
                CoreMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ]
        } else {
            tracing::info!(
                target: "audio2face",
                "Audio2Face: CPU execution provider only (use_gpu=false)"
            );
            vec![CPUExecutionProvider::default().build()]
        };

        let session = Session::builder()
            .map_err(|e| Error::Execution(format!("ort builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Execution(format!("ort optimization level: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| Error::Execution(format!("ort intra threads: {e}")))?
            .with_execution_providers(providers)
            .map_err(|e| Error::Execution(format!("ort execution providers: {e}")))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| {
                Error::Execution(format!(
                    "ort failed to load Audio2Face network from {}: {e}",
                    model_path.as_ref().display()
                ))
            })?;

        // GRU hidden state — zero on first call, model warms it up.
        let gru_state = vec![0.0f32; NUM_DIFFUSION_STEPS * NUM_GRU_LAYERS * GRU_LATENT_DIM];

        // Pre-generate the Box-Muller noise tensor.
        let noise = generate_gaussian_noise(NOISE_TENSOR_LEN, NOISE_SEED);

        Ok(Self {
            session,
            gru_state,
            noise,
        })
    }

    /// Run one inference pass.
    ///
    /// `audio_16khz` is f32 mono PCM at 16 kHz. If it's shorter than
    /// [`AUDIO_BUFFER_LEN`] it's zero-padded; if it's longer, only the
    /// first `AUDIO_BUFFER_LEN` samples are used. (Spec §3.4 invariant
    /// is "1-second window of audio at the model's required rate".)
    ///
    /// `identity_index` selects the one-hot identity slot. Use
    /// [`super::Audio2FaceIdentity::one_hot_index`] to get a valid value.
    pub fn infer(
        &mut self,
        audio_16khz: &[f32],
        identity_index: usize,
    ) -> Result<Audio2FaceOutput> {
        if identity_index >= NUM_IDENTITIES {
            return Err(Error::InvalidData(format!(
                "identity index {identity_index} must be < {NUM_IDENTITIES}"
            )));
        }

        // 1. Build window: zero-padded copy of input audio.
        let mut window = vec![0.0f32; AUDIO_BUFFER_LEN];
        let copy_len = audio_16khz.len().min(AUDIO_BUFFER_LEN);
        window[..copy_len].copy_from_slice(&audio_16khz[..copy_len]);

        // 2. Identity one-hot.
        let mut identity = vec![0.0f32; NUM_IDENTITIES];
        identity[identity_index] = 1.0;

        // 3. Emotion (zero = neutral; future: thread emotion in via `infer_with_emotion`).
        let emotion = vec![0.0f32; 1 * NUM_CENTER_FRAMES * NUM_EMOTIONS];

        // 4. GRU input — clone the carried state.
        let input_latents = self.gru_state.clone();

        // 5. Build ort tensors.
        let window_tensor = Tensor::from_array(([1, AUDIO_BUFFER_LEN], window))
            .map_err(|e| Error::Execution(format!("ort window tensor: {e}")))?;
        let identity_tensor = Tensor::from_array(([1, NUM_IDENTITIES], identity))
            .map_err(|e| Error::Execution(format!("ort identity tensor: {e}")))?;
        let emotion_tensor = Tensor::from_array(([1, NUM_CENTER_FRAMES, NUM_EMOTIONS], emotion))
            .map_err(|e| Error::Execution(format!("ort emotion tensor: {e}")))?;
        let latents_tensor = Tensor::from_array((
            [NUM_DIFFUSION_STEPS, NUM_GRU_LAYERS, 1, GRU_LATENT_DIM],
            input_latents,
        ))
        .map_err(|e| Error::Execution(format!("ort latents tensor: {e}")))?;
        // Noise is the heavy one — clone the cached Vec; ort owns the
        // memory for the call duration. ~64 MB clone per call is the
        // perf knob to tune later if it shows up in profiles.
        let noise_tensor = Tensor::from_array((
            [1, NUM_DIFFUSION_STEPS + 1, TOTAL_FRAMES, TOTAL_OUTPUT_DIM],
            self.noise.clone(),
        ))
        .map_err(|e| Error::Execution(format!("ort noise tensor: {e}")))?;

        // 6. Run.
        let outputs = self
            .session
            .run(ort::inputs![
                "window" => window_tensor,
                "identity" => identity_tensor,
                "emotion" => emotion_tensor,
                "input_latents" => latents_tensor,
                "noise" => noise_tensor,
            ])
            .map_err(|e| Error::Execution(format!("Audio2Face session.run: {e}")))?;

        // 7. Extract prediction → skin + eye slices for each center frame.
        let (_pred_shape, pred) = outputs["prediction"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Execution(format!("Audio2Face extract prediction: {e}")))?;

        let mut skin_flat = vec![0.0f32; NUM_CENTER_FRAMES * SKIN_SIZE];
        let mut eye_flat = vec![0.0f32; NUM_CENTER_FRAMES * EYES_SIZE];
        for frame in 0..NUM_CENTER_FRAMES {
            let src_offset = (LEFT_PAD_FRAMES + frame) * TOTAL_OUTPUT_DIM;
            let skin_src = &pred[src_offset..src_offset + SKIN_SIZE];
            skin_flat[frame * SKIN_SIZE..(frame + 1) * SKIN_SIZE].copy_from_slice(skin_src);

            let eye_src_offset = src_offset + EYES_OFFSET;
            let eye_src = &pred[eye_src_offset..eye_src_offset + EYES_SIZE];
            eye_flat[frame * EYES_SIZE..(frame + 1) * EYES_SIZE].copy_from_slice(eye_src);
        }

        // 8. Update GRU state from output_latents (same shape).
        let (_, new_latents) = outputs["output_latents"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Execution(format!("Audio2Face extract latents: {e}")))?;
        self.gru_state.copy_from_slice(new_latents);

        Ok(Audio2FaceOutput {
            skin_flat,
            eye_flat,
            frame_count: NUM_CENTER_FRAMES,
        })
    }

    /// Reset the GRU state to zeros — used by `barge_in` to clear any
    /// recurrent context from the previous turn.
    pub fn reset_state(&mut self) {
        self.gru_state.fill(0.0);
    }

    /// Snapshot the GRU state (for save/restore).
    pub fn save_gru_state(&self) -> Vec<f32> {
        self.gru_state.clone()
    }

    /// Overwrite the GRU state from a previously saved copy.
    pub fn restore_gru_state(&mut self, saved: &[f32]) -> Result<()> {
        if saved.len() != self.gru_state.len() {
            return Err(Error::InvalidData(format!(
                "GRU state size mismatch: expected {}, got {}",
                self.gru_state.len(),
                saved.len()
            )));
        }
        self.gru_state.copy_from_slice(saved);
        Ok(())
    }
}

/// Generate Box-Muller distributed Gaussian noise. Mirrors C#'s
/// `Random.NextDouble`-based implementation but uses a small
/// deterministic LCG so the seed-0 sequence is reproducible across
/// platforms (C#'s `Random` isn't cross-platform-deterministic).
///
/// Note: the C# reference uses `System.Random` which has its own
/// platform-specific behavior; we match the *shape* of Box-Muller
/// (pairs of `(magnitude * cos(angle), magnitude * sin(angle))`) but
/// not the exact bit-for-bit values. The model is robust to
/// alternate Gaussian samples — it's a diffusion denoising step,
/// not a parity check.
fn generate_gaussian_noise(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut out = vec![0.0f32; count];
    let mut i = 0;
    while i + 1 < count {
        // Box-Muller: u1 ∈ (0, 1] (avoid 0 to keep log finite); u2 ∈ [0, 1).
        let u1 = 1.0 - rng.next_f64();
        let u2 = rng.next_f64();
        let magnitude = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        out[i] = (magnitude * angle.cos()) as f32;
        out[i + 1] = (magnitude * angle.sin()) as f32;
        i += 2;
    }
    if i < count {
        let u1 = 1.0 - rng.next_f64();
        let u2 = rng.next_f64();
        out[i] = ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
    }
    out
}

/// Tiny deterministic PRNG. SplitMix64 is good enough for noise
/// generation (it's not cryptographic); pulled in here rather than
/// adding the `rand` crate as a new dep for a single use site.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a uniform u64.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Returns a uniform `[0, 1)` f64. Uses the top 53 bits of a u64
    /// (an f64 mantissa) — same trick `numpy` and `rand` use.
    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 high bits
        bits as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}
