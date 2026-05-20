//! `Audio2FaceLipSyncNode` as a standalone Path 3 loadable plugin.
//!
//! Coordinates the NVIDIA Audio2Face bundle into a streaming
//! `LipSyncNode`-shaped node: consumes `RuntimeData::Audio` at 16 kHz
//! in 1-second windows, runs ONNX inference (`audio2face/inference`),
//! solves the masked-vertex delta to 39-D blendshape weights with
//! PGD or BVLS, expands to 52-D ARKit, optionally smooths via uniform
//! EMA, and emits one `RuntimeData::Json {kind: "blendshapes", ...}`
//! per ~33 ms output frame.
//!
//! Originally lived in `remotemedia-core` under the
//! `avatar-audio2face` feature; extracted here so the host crate
//! doesn't drag in `ort` + `zip` + the NPZ/NPY readers just for this
//! single coordinator.
//!
//! ## Node types exported
//!
//!   Audio2FaceLipSyncNode — Audio → Audio passthrough +
//!                            Json{kind:"blendshapes", arkit_52, pts_ms, turn_id?}

mod arkit;
mod audio2face;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;
use remotemedia_plugin_sdk::types::{AudioSamples, Error, RuntimeData};
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};

use crate::arkit::{ArkitSmoother, BlendshapeFrame, ARKIT_52};
use crate::audio2face::inference::{
    Audio2FaceInference, AUDIO_BUFFER_LEN, NUM_CENTER_FRAMES, SKIN_SIZE,
};
use crate::audio2face::solver_trait::BlendshapeSolver;
use crate::audio2face::{
    AnimatorSkinConfig, Audio2FaceIdentity, BlendshapeConfig, BlendshapeData, BundlePaths,
    BvlsBlendshapeSolver, PgdBlendshapeSolver,
};

/// Local `Result` alias so the ported source can keep using the
/// bare-name `Result<T>` style from the original module.
type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Inlined session-control constants/helpers
// ---------------------------------------------------------------------------
//
// Mirrors `crate::transport::session_control::{aux_port_of,
// BARGE_IN_PORT, AUX_PORT_ENVELOPE_KEY}` from the host. Replicated
// here so this plugin doesn't link `remotemedia-core`.

/// Envelope field name for aux-port publishes.
const AUX_PORT_ENVELOPE_KEY: &str = "__aux_port__";

/// Reserved aux port name for "the user has barged in / cancel the
/// in-flight call".
const BARGE_IN_PORT: &str = "barge_in";

/// If `data` is an aux-port envelope, return the port name. Otherwise
/// return `None`.
fn aux_port_of(data: &RuntimeData) -> Option<&str> {
    match data {
        RuntimeData::Json(v) => v.get(AUX_PORT_ENVELOPE_KEY).and_then(|p| p.as_str()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Solver to use for the masked-delta → 39-D weight step. PGD is the
/// persona-engine default (faster, slightly less accurate); BVLS is the
/// reference (slower, scipy-equivalent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SolverChoice {
    /// Projected gradient descent with LU-warm-started initial guess.
    Pgd,
    /// Bounded-variable least squares (active-set + Cholesky).
    Bvls,
}

impl Default for SolverChoice {
    fn default() -> Self {
        SolverChoice::Pgd
    }
}

/// Configuration for [`Audio2FaceLipSyncNode`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Audio2FaceLipSyncConfig {
    /// Path to the unpacked persona-engine Audio2Face bundle (the
    /// directory containing `network.onnx`, `bs_skin_<Identity>.npz`,
    /// etc.). See `scripts/install-audio2face.sh`.
    pub bundle_path: PathBuf,
    /// Identity slot (Claire / James / Mark).
    pub identity: Audio2FaceIdentity,
    /// Solver to use.
    pub solver: SolverChoice,
    /// Whether to use GPU execution providers (CUDA → CoreML → CPU
    /// fallback when true; CPU-only when false).
    pub use_gpu: bool,
    /// Uniform EMA alpha applied to the 52-D ARKit vector before
    /// emission. `0.0` = no smoothing (passthrough).
    pub smoothing_alpha: f32,
    /// When `true`, sleep between emitted blendshape envelopes so they
    /// reach downstream consumers at content-time pacing
    /// (`pts_ms` rate). Default `true` for offline-batch renders;
    /// the live WebRTC pipeline can set `false` because the upstream
    /// audio chunks already arrive paced by playback.
    #[serde(default = "default_pace_realtime")]
    pub pace_realtime: bool,
}

fn default_pace_realtime() -> bool {
    true
}

impl Default for Audio2FaceLipSyncConfig {
    fn default() -> Self {
        Self {
            bundle_path: PathBuf::new(),
            identity: Audio2FaceIdentity::Claire,
            solver: SolverChoice::Pgd,
            use_gpu: false,
            smoothing_alpha: 0.0,
            pace_realtime: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// Inference + solve coordinator implementing the streaming
/// lip-sync contract.
pub struct Audio2FaceLipSyncNode {
    config: Audio2FaceLipSyncConfig,
    inference: Arc<Mutex<Audio2FaceInference>>,
    solver: Arc<Mutex<Box<dyn BlendshapeSolver + Send>>>,
    smoother: Arc<Mutex<ArkitSmoother>>,
    bs_config: Arc<BlendshapeConfig>,
    animator_config: Arc<AnimatorSkinConfig>,
    data: Arc<BlendshapeData>,
    /// Audio accumulator — appended on every Audio chunk; drained in
    /// [`AUDIO_BUFFER_LEN`]-sample windows.
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    /// Cumulative ms of audio that has been *fully consumed* by an
    /// inference call (i.e. multiples of 1000). The pts_ms of frame `f`
    /// emitted from a window is `cum_window_ms + f * 1000 / 30`.
    cum_window_ms: Arc<AtomicU64>,
    /// Real-time pacing anchor. `(wallclock_when_first_emitted, first_pts_ms)`.
    pace_anchor: Arc<Mutex<Option<(std::time::Instant, u64)>>>,
}

impl Audio2FaceLipSyncNode {
    /// Load the bundle from disk and assemble the inference + solver
    /// stack. Heavy: reads ~700 MiB of model weights + ~150 MiB of NPZ
    /// data + builds K-D solver matrices.
    pub fn load(config: Audio2FaceLipSyncConfig) -> Result<Self> {
        let paths = BundlePaths::new(&config.bundle_path, config.identity);

        let bs_config = BlendshapeConfig::from_path(paths.bs_skin_config())
            .map_err(|e| Error::Execution(format!("blendshape config: {e}")))?;
        let animator_config = AnimatorSkinConfig::from_path(paths.model_config())
            .map_err(|e| Error::Execution(format!("animator skin config: {e}")))?;
        let data = BlendshapeData::load(paths.bs_skin_npz(), paths.model_data_npz(), &bs_config)
            .map_err(|e| Error::Execution(format!("blendshape data: {e}")))?;
        let inference = Audio2FaceInference::load(paths.network_onnx(), config.use_gpu)?;

        let solver: Box<dyn BlendshapeSolver + Send> = match config.solver {
            SolverChoice::Pgd => Box::new(PgdBlendshapeSolver::new(
                &data.delta_matrix,
                data.masked_position_count,
                data.active_count,
                &data.neutral_flat,
                bs_config.template_bb_size,
                bs_config.strength_l2,
                bs_config.strength_l1,
                bs_config.strength_temporal,
            )),
            SolverChoice::Bvls => Box::new(BvlsBlendshapeSolver::new(
                &data.delta_matrix,
                data.masked_position_count,
                data.active_count,
                &data.neutral_flat,
                bs_config.template_bb_size,
                bs_config.strength_l2,
                bs_config.strength_l1,
                bs_config.strength_temporal,
            )),
        };

        let smoother = ArkitSmoother::new(config.smoothing_alpha);

        Ok(Self {
            config,
            inference: Arc::new(Mutex::new(inference)),
            solver: Arc::new(Mutex::new(solver)),
            smoother: Arc::new(Mutex::new(smoother)),
            bs_config: Arc::new(bs_config),
            animator_config: Arc::new(animator_config),
            data: Arc::new(data),
            audio_buffer: Arc::new(Mutex::new(Vec::with_capacity(AUDIO_BUFFER_LEN * 2))),
            cum_window_ms: Arc::new(AtomicU64::new(0)),
            pace_anchor: Arc::new(Mutex::new(None)),
        })
    }

    /// Drop all in-flight state: GRU, solver temporal pull, smoother,
    /// and the audio accumulator. Idempotent. Used by `barge_in`.
    pub fn barge(&self) {
        self.inference.lock().reset_state();
        self.solver.lock().reset_temporal();
        self.smoother.lock().reset();
        self.audio_buffer.lock().clear();
        self.cum_window_ms.store(0, Ordering::Release);
        *self.pace_anchor.lock() = None;
    }
}

/// Run inference + solve for one window and return the emitted
/// `RuntimeData::Json` envelopes. Free-standing (not a method) so it
/// can be moved into `tokio::task::spawn_blocking` from
/// [`Audio2FaceLipSyncNode::process_streaming`] without dragging
/// `&self`.
fn run_window_blocking(
    window: &[f32],
    window_start_ms: u64,
    identity: Audio2FaceIdentity,
    inference: &Mutex<Audio2FaceInference>,
    solver: &Mutex<Box<dyn BlendshapeSolver + Send>>,
    smoother: &Mutex<ArkitSmoother>,
    bs_config: &BlendshapeConfig,
    animator_config: &AnimatorSkinConfig,
    data: &BlendshapeData,
) -> Result<Vec<RuntimeData>> {
    let identity_idx = identity.one_hot_index();
    let out = {
        let mut infer = inference.lock();
        infer.infer(window, identity_idx)?
    };
    debug_assert_eq!(out.frame_count, NUM_CENTER_FRAMES);

    let mut solver_guard = solver.lock();
    let mut smoother_guard = smoother.lock();

    // Step is f64 to avoid drift when summed over many seconds.
    let frame_ms_step = 1000.0_f64 / NUM_CENTER_FRAMES as f64;

    let mut outputs = Vec::with_capacity(NUM_CENTER_FRAMES);
    let mut first_pts: Option<u64> = None;
    let mut last_pts: u64 = 0;
    for f in 0..NUM_CENTER_FRAMES {
        let skin_frame = &out.skin_flat[f * SKIN_SIZE..(f + 1) * SKIN_SIZE];
        let arkit = skin_frame_to_arkit(
            skin_frame,
            solver_guard.as_mut(),
            &mut smoother_guard,
            bs_config,
            animator_config,
            data,
        );
        let pts_ms = window_start_ms + (f as f64 * frame_ms_step) as u64;
        if first_pts.is_none() {
            first_pts = Some(pts_ms);
        }
        last_pts = pts_ms;
        let frame = BlendshapeFrame::new(arkit, pts_ms, None);
        outputs.push(RuntimeData::Json(frame.to_json()));
    }
    tracing::debug!(
        target: "lipsync.bs",
        window_start_ms,
        first_pts_ms = first_pts.unwrap_or(0),
        last_pts_ms = last_pts,
        "audio2face emitted {} blendshapes",
        outputs.len(),
    );
    Ok(outputs)
}

/// Convert one skin frame (full 24002-vertex × 3 deltas) into a
/// 52-D ARKit blendshape vector. Mirrors persona-engine's
/// `Audio2FaceLipSyncProcessor.cs:ProcessSkinFrame`:
///
/// ```text
///   composed[v] = skin_strength * skin_flat[v]
///               + eye_close_pose_delta[v] * (-eyelid_open_offset)
///               + lip_open_pose_delta[v] * lip_open_offset
///   delta[m]    = neutral_skin_flat[v] + composed[v] - neutral_flat[m]
/// ```
///
/// where `v = frontal_mask[m]`. `neutral_skin_flat` is the V*3
/// model-frame neutral from `model_data_<Identity>.npz`;
/// `neutral_flat` is the M*3 masked neutral from
/// `bs_skin_<Identity>.npz`. Their difference at matched indices is
/// ~0, so the practical signal is `composed[v]` — small, audio-driven.
fn skin_frame_to_arkit(
    skin_frame: &[f32],
    solver: &mut (dyn BlendshapeSolver + Send),
    smoother: &mut ArkitSmoother,
    bs_config: &BlendshapeConfig,
    animator_config: &AnimatorSkinConfig,
    data: &BlendshapeData,
) -> [f32; ARKIT_52] {
    let mask = &data.frontal_mask;
    let bs_neutral = &data.neutral_flat;
    let model_neutral = &data.neutral_skin_flat;
    let eye_close = &data.eye_close_pose_delta_flat;
    let lip_open = &data.lip_open_pose_delta_flat;
    let masked_count = data.masked_position_count;
    let skin_strength = animator_config.skin_strength;
    let eyelid_open_offset = animator_config.eyelid_open_offset;
    let lip_open_offset = animator_config.lip_open_offset;

    let mut masked_delta = vec![0.0f32; masked_count];
    for (m, &vi_i32) in mask.iter().enumerate() {
        let vi = vi_i32 as usize;
        let v_base = vi * 3;
        let m_base = m * 3;
        if v_base + 2 >= skin_frame.len()
            || v_base + 2 >= model_neutral.len()
            || v_base + 2 >= eye_close.len()
            || v_base + 2 >= lip_open.len()
            || m_base + 2 >= bs_neutral.len()
        {
            continue;
        }
        for c in 0..3 {
            let composed = skin_strength * skin_frame[v_base + c]
                + eye_close[v_base + c] * (-eyelid_open_offset)
                + lip_open[v_base + c] * lip_open_offset;
            masked_delta[m_base + c] =
                model_neutral[v_base + c] + composed - bs_neutral[m_base + c];
        }
    }

    let weights = solver.solve(&masked_delta);

    let mut arkit = [0.0f32; ARKIT_52];
    for (k, &pose_index) in bs_config.active_indices.iter().enumerate() {
        if pose_index < ARKIT_52 {
            arkit[pose_index] = weights[k];
        }
    }
    for i in 0..ARKIT_52 {
        arkit[i] = arkit[i] * bs_config.multipliers[i] + bs_config.offsets[i];
    }

    smoother.smooth(&arkit)
}

#[async_trait]
impl AsyncStreamingNode for Audio2FaceLipSyncNode {
    fn node_type(&self) -> &str {
        "Audio2FaceLipSyncNode"
    }

    async fn process(&self, _data: RuntimeData) -> Result<RuntimeData> {
        Err(Error::Execution(
            "Audio2FaceLipSyncNode requires streaming mode — use process_streaming()".into(),
        ))
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        _session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize>
    where
        F: FnMut(RuntimeData) -> Result<()> + Send,
    {
        // Barge envelope: {kind: "barge_in"}. Clears state and emits nothing.
        if let RuntimeData::Json(v) = &data {
            if v.get("kind").and_then(|k| k.as_str()) == Some("barge_in") {
                self.barge();
                return Ok(0);
            }
        }

        let (
            samples,
            sample_rate,
            chunk_pts_us,
            channels,
            stream_id,
            timestamp_us,
            arrival_ts_us,
            metadata,
        ) = match data {
            RuntimeData::Audio {
                samples,
                sample_rate,
                channels,
                stream_id,
                timestamp_us,
                arrival_ts_us,
                metadata,
            } => {
                // Upstream content-time pts (e.g. set by KokoroTTSNode
                // on every emitted chunk via `metadata.annotations`).
                let pts_us = metadata
                    .as_ref()
                    .and_then(|m| m.get("pts_us"))
                    .and_then(|v| v.as_u64());
                (
                    samples,
                    sample_rate,
                    pts_us,
                    channels,
                    stream_id,
                    timestamp_us,
                    arrival_ts_us,
                    metadata,
                )
            }
            other => {
                // Pass non-Audio through untouched.
                callback(other)?;
                return Ok(1);
            }
        };

        // PASSTHROUGH: emit the input Audio so downstream consumers
        // (RenderReadyGateNode → CcRenderNode) see the same audio
        // stream the lip-sync inference is running on.
        let pt_pts_ms = chunk_pts_us.map(|us| (us / 1000) as i64).unwrap_or(-1);
        tracing::info!(
            target: "timing",
            stage = "audio2face_recv",
            pts_ms = pt_pts_ms,
            samples = samples.len() as u64,
            sample_rate = sample_rate as u64,
        );
        tracing::info!(
            target: "timing",
            stage = "audio2face_emit_audio",
            pts_ms = pt_pts_ms,
            samples = samples.len() as u64,
        );
        callback(RuntimeData::Audio {
            samples: samples.clone(),
            sample_rate,
            channels,
            stream_id,
            timestamp_us,
            arrival_ts_us,
            metadata,
        })?;

        if sample_rate != 16_000 {
            return Err(Error::InvalidData(format!(
                "Audio2FaceLipSyncNode requires 16 kHz audio (capability \
                 resolver should insert a resampler upstream); got {sample_rate}"
            )));
        }

        // Window stride. The model takes a fixed `AUDIO_BUFFER_LEN`
        // (1s @ 16 kHz) inference window. Advancing the buffer by less
        // than that per fire would halve the time-to-first-emit for
        // the tail of every chunk, but only if inference runs faster
        // than real-time on the host. Keep stride at a full window
        // so audio2face runs at ≤1× the audio rate.
        const STRIDE: usize = AUDIO_BUFFER_LEN;
        const STRIDE_MS: u64 = 1000;
        let windows: Vec<Vec<f32>> = {
            let mut buf = self.audio_buffer.lock();

            // If upstream stamped a content-time pts on this chunk
            // (KokoroTTSNode does), anchor `cum_window_ms` to it.
            if let Some(pts_us) = chunk_pts_us {
                let chunk_pts_ms = pts_us / 1000;
                let buf_offset_ms = (buf.len() as u64) * 1000 / sample_rate as u64;
                let anchor_ms = chunk_pts_ms.saturating_sub(buf_offset_ms);
                self.cum_window_ms.store(anchor_ms, Ordering::Release);
            }

            buf.extend_from_slice(samples.as_slice());
            let mut ws = Vec::new();
            // Fire while a full window is available. Each fire copies
            // the leading `AUDIO_BUFFER_LEN` samples (the inference
            // window), then drains only `STRIDE` samples — the trailing
            // `AUDIO_BUFFER_LEN - STRIDE` stay as overlap context for
            // the next fire.
            while buf.len() >= AUDIO_BUFFER_LEN {
                let window: Vec<f32> = buf[..AUDIO_BUFFER_LEN].to_vec();
                buf.drain(..STRIDE);
                ws.push(window);
            }
            ws
        };

        let mut emitted = 1; // +1 for the audio passthrough above
        for window in windows {
            let window_start_ms = self.cum_window_ms.fetch_add(STRIDE_MS, Ordering::AcqRel);
            // Move ONNX inference + the PGD/BVLS solver onto tokio's
            // blocking thread pool so the worker stays free to drain
            // sibling tasks.
            let inference = Arc::clone(&self.inference);
            let solver = Arc::clone(&self.solver);
            let smoother = Arc::clone(&self.smoother);
            let bs_config = Arc::clone(&self.bs_config);
            let animator_config = Arc::clone(&self.animator_config);
            let data = Arc::clone(&self.data);
            let identity = self.config.identity;
            let outputs = tokio::task::spawn_blocking(move || {
                run_window_blocking(
                    &window,
                    window_start_ms,
                    identity,
                    &inference,
                    &solver,
                    &smoother,
                    &bs_config,
                    &animator_config,
                    &data,
                )
            })
            .await
            .map_err(|e| {
                Error::Execution(format!(
                    "Audio2FaceLipSyncNode inference task panicked: {e}"
                ))
            })??;
            for out in outputs {
                // Pace per-envelope by pts_ms so downstream latest-wins
                // consumers (e.g. CcRenderNode's pose watch channel) see
                // every blendshape frame instead of just the LAST one
                // bursted in microseconds.
                if self.config.pace_realtime {
                    if let RuntimeData::Json(v) = &out {
                        if let Some(pts_ms) = v.get("pts_ms").and_then(|p| p.as_u64()) {
                            let now = std::time::Instant::now();
                            // Scope the (non-Send) MutexGuard so it cannot
                            // possibly straddle the await below — required
                            // since `process_streaming` returns a `Send`
                            // future.
                            let (anchor_wall, anchor_pts) = {
                                let mut anchor = self.pace_anchor.lock();
                                match *anchor {
                                    Some(a) => a,
                                    None => {
                                        tracing::info!(
                                            target: "lipsync.bs",
                                            "pace_realtime: anchored at pts_ms={}", pts_ms
                                        );
                                        *anchor = Some((now, pts_ms));
                                        (now, pts_ms)
                                    }
                                }
                            };
                            let target = anchor_wall
                                + std::time::Duration::from_millis(
                                    pts_ms.saturating_sub(anchor_pts),
                                );
                            if target > now {
                                let dur = target - now;
                                if pts_ms % 200 == 0 {
                                    tracing::info!(
                                        target: "lipsync.bs",
                                        "pace_realtime: sleep {}ms before emit pts_ms={}",
                                        dur.as_millis(), pts_ms
                                    );
                                }
                                tokio::time::sleep(dur).await;
                            } else {
                                // Late branch: re-anchor at (now, pts_ms)
                                // so subsequent outputs in the same batch
                                // pace at realtime from this point forward.
                                {
                                    let mut anchor = self.pace_anchor.lock();
                                    *anchor = Some((now, pts_ms));
                                }
                                if pts_ms % 200 == 0 {
                                    tracing::info!(
                                        target: "lipsync.bs",
                                        "pace_realtime: re-anchored — emit pts_ms={} was late by {}ms",
                                        pts_ms, (now - target).as_millis()
                                    );
                                }
                            }
                        }
                    }
                }
                if let RuntimeData::Json(v) = &out {
                    let pts_ms = v
                        .get("pts_ms")
                        .and_then(|p| p.as_u64())
                        .map(|u| u as i64)
                        .unwrap_or(-1);
                    let max_w = v
                        .get("arkit_52")
                        .and_then(|a| a.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|n| n.as_f64())
                                .fold(0.0_f64, f64::max)
                        })
                        .unwrap_or(0.0);
                    tracing::info!(
                        target: "timing",
                        stage = "audio2face_emit_bs",
                        pts_ms = pts_ms,
                        max_w = max_w,
                    );
                }
                callback(out)?;
                emitted += 1;
            }
        }
        Ok(emitted)
    }

    /// Runtime-dispatched control message handler. The session router
    /// forwards `<node>.in.barge_in` aux-port envelopes here; on
    /// receipt we drop GRU + solver-temporal + smoother + audio buffer
    /// + pts clock.
    async fn process_control_message(
        &self,
        message: RuntimeData,
        _session_id: Option<String>,
    ) -> Result<bool> {
        if matches!(aux_port_of(&message), Some(BARGE_IN_PORT)) {
            self.barge();
            return Ok(true);
        }
        Ok(false)
    }
}

// `AudioSamples::as_slice` is a nicety we lean on above. Make sure it
// stays referenced by the compiler so feature-gated builds catch any
// upstream rename early.
#[allow(dead_code)]
fn _audio_samples_as_slice_witness(s: &AudioSamples) -> &[f32] {
    s.as_slice()
}

// ---------------------------------------------------------------------------
// Factory + plugin registration
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Audio2FaceLipSyncNodeFactory;

impl FfiNodeFactory for Audio2FaceLipSyncNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("Audio2FaceLipSyncNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        // Accepts both snake_case and camelCase manifest keys via
        // plugin-sdk's lenient deserializer. Surface parse errors
        // explicitly — silently falling back to defaults would leave
        // bundlePath empty and surface only as ENOENT at load time.
        let cfg: Audio2FaceLipSyncConfig =
            match remotemedia_plugin_sdk::params::deserialize_params(params.as_str()) {
                Ok(c) => c,
                Err(e) => {
                    return remotemedia_plugin_sdk::abi_stable::std_types::RErr(RString::from(
                        format!("Audio2FaceLipSyncNode params parse failed: {e}"),
                    ));
                }
            };
        let node = match Audio2FaceLipSyncNode::load(cfg) {
            Ok(n) => n,
            Err(e) => {
                return remotemedia_plugin_sdk::abi_stable::std_types::RErr(RString::from(
                    format!("Audio2FaceLipSyncNode load failed: {e}"),
                ));
            }
        };
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(node),
            TD_Opaque,
        ))
    }
}

// Emits the abi_stable root-module symbol for dlopen. Gated behind
// the `plugin-export` cargo feature so the rlib can be linked
// alongside other plugins without duplicate-symbol collisions.
#[cfg(feature = "plugin-export")]
remotemedia_plugin_sdk::plugin_export!(Audio2FaceLipSyncNodeFactory);
