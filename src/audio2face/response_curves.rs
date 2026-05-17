//! Response curves — Rust port of `ResponseCurves.cs`.
//!
//! The persona-engine uses these to reshape the linear blendshape
//! activations the solvers emit into more visually-pleasing curves
//! before the renderer's ARKit→VBridger mapper sees them. Two shapes:
//!
//! - [`ease_in`] — steep at start, flat at end. Used for `JawOpen` /
//!   `MouthOpen` so a small audio response opens the mouth quickly.
//! - [`center_weighted`] — flat at extremes, steep through center.
//!   Used for `MouthPressLipOpen` and `EyeBallY` where the
//!   informative range is the middle of the activation envelope.

/// Hermite spline `f(t) = t · (2 − t)`. Steep tangent at `t=0`, flat
/// at `t=1`. Output range `[0, 1]`. Input is clamped to `[0, 1]`.
pub fn ease_in(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * (2.0 - t)
}

/// Three-key center-weighted Hermite curve. Keys at `(lo, lo, m=0)`,
/// `(0, 0, m=2)`, `(hi, hi, m=0)`. Steep through zero, flat at the
/// extremes. Input clamped to `[lo, hi]`.
///
/// `lo` is expected ≤ 0 and `hi` ≥ 0; degenerate spans (`|span|<ε`)
/// pass `t` through unchanged for the matching segment.
pub fn center_weighted(t: f32, lo: f32, hi: f32) -> f32 {
    let t = t.clamp(lo, hi);
    if t <= 0.0 {
        // Lower segment: lo..0
        let span = -lo;
        if span < 1e-6 {
            return t;
        }
        let s = (t - lo) / span;
        hermite_segment(s, lo, 0.0, 0.0, 2.0 * span)
    } else {
        // Upper segment: 0..hi
        let span = hi;
        if span < 1e-6 {
            return t;
        }
        let s = t / span;
        hermite_segment(s, 0.0, hi, 2.0 * span, 0.0)
    }
}

/// Cubic Hermite interpolation between (p0, m0) and (p1, m1).
/// `s ∈ [0, 1]`. Standard basis polynomials.
fn hermite_segment(s: f32, p0: f32, p1: f32, m0: f32, m1: f32) -> f32 {
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}
