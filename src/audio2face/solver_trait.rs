//! `BlendshapeSolver` trait — Rust port of `IBlendshapeSolver.cs`.

/// Strategy interface for solving blendshape weights from masked
/// vertex deltas.
///
/// Implementations minimize `‖A·x − b‖²` subject to `0 ≤ x ≤ 1`
/// with temporal regularization applied across calls (so frame *n+1*
/// stays close to frame *n*).
///
/// The C# version returns `float[]`; we return `&[f32]` (a slice
/// borrow into a per-instance `result` buffer, mirroring the C#
/// pre-allocated `_result` field).
pub trait BlendshapeSolver {
    /// Solve for one frame's blendshape weights from a masked vertex
    /// delta. Length of `delta` must be the masked-position count
    /// the solver was constructed with.
    fn solve(&mut self, delta: &[f32]) -> &[f32];

    /// Reset temporal smoothing state so the next `solve()` call has
    /// no temporal pull from prior frames. Used by `barge_in`.
    fn reset_temporal(&mut self);

    /// Snapshot the previous-frame weights for later restore.
    fn save_temporal(&self) -> Vec<f64>;

    /// Restore previous-frame weights from a snapshot. Length must
    /// equal `active_count`.
    fn restore_temporal(&mut self, saved: &[f64]);

    /// Number of active blendshapes (K) — the length of returned
    /// solve outputs.
    fn active_count(&self) -> usize;
}
