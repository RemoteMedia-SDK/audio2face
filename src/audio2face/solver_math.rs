//! Shared math utilities for the BVLS / PGD blendshape solvers.
//!
//! 1:1 port of `external/handcrafted-persona-engine/.../Audio2Face/SolverMath.cs`.
//! Centralizes:
//! - bounding-box-diagonal computation (vertex-cloud span used to
//!   normalize regularization across face sizes)
//! - row-major D^T and D^T·D products (the QP normal-equation prep)
//! - regularization application (L2 on diagonal, L1 on full matrix,
//!   temporal on diagonal)
//!
//! Numeric semantics mirror the reference — `f64` accumulation
//! throughout, regularization multipliers as named constants.

/// Audio2Face SDK default L2 (diagonal) regularization multiplier.
pub const L2_MULTIPLIER: f64 = 10.0;
/// Audio2Face SDK default L1 (full-matrix) regularization multiplier.
pub const L1_MULTIPLIER: f64 = 0.25;
/// Audio2Face SDK default temporal (diagonal) regularization multiplier.
pub const TEMPORAL_MULTIPLIER: f64 = 100.0;

/// Compute the Euclidean length of the bounding-box diagonal of a
/// flattened `[N*3]` vertex buffer (XYZ-interleaved). Returns `0.0`
/// for empty input.
///
/// Used to scale regularization strengths so they're consistent across
/// faces of different absolute sizes — the SDK's "scale" factor is
/// `(bbSize / templateBBSize)²`.
pub fn compute_bounding_box_diagonal(neutral_flat3: &[f32]) -> f64 {
    let n_verts = neutral_flat3.len() / 3;
    if n_verts == 0 {
        return 0.0;
    }
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut min_z = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    let mut max_z = f64::MIN;
    for i in 0..n_verts {
        let x = neutral_flat3[i * 3] as f64;
        let y = neutral_flat3[i * 3 + 1] as f64;
        let z = neutral_flat3[i * 3 + 2] as f64;
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
        if z < min_z {
            min_z = z;
        }
        if z > max_z {
            max_z = z;
        }
    }
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let dz = max_z - min_z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute `D^T` — given `D[V x K]` row-major, write `Dᵀ[K x V]` row-major
/// into `dt`.
///
/// `dt` must be at least `k * v` elements. C#'s `Span<T>` becomes
/// a `&mut [f64]` slice; we trust the caller to size correctly,
/// matching the C# API.
pub fn compute_transpose(d: &[f32], v: usize, k: usize, dt: &mut [f64]) {
    debug_assert!(d.len() >= v * k, "D too small");
    debug_assert!(dt.len() >= k * v, "Dᵀ too small");
    for vi in 0..v {
        for ki in 0..k {
            dt[ki * v + vi] = d[vi * k + ki] as f64;
        }
    }
}

/// Compute `D^T·D` — given `Dᵀ[K x V]` row-major, write
/// `(D^T·D)[K x K]` row-major into `dtd`.
///
/// Pure inner-product loop. f64 accumulation throughout.
pub fn compute_dt_d(dt: &[f64], k: usize, v: usize, dtd: &mut [f64]) {
    debug_assert!(dt.len() >= k * v);
    debug_assert!(dtd.len() >= k * k);
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0f64;
            for p in 0..v {
                sum += dt[i * v + p] * dt[j * v + p];
            }
            dtd[i * k + j] = sum;
        }
    }
}

/// Apply L2 (diagonal), L1 (full-matrix), and temporal (diagonal)
/// regularization to a `K×K` row-major matrix in place.
///
/// Returns the temporal scale (`TEMPORAL_MULTIPLIER · scale ·
/// strengthTemporal`) so the caller can reuse it when assembling the
/// `b` vector.
pub fn apply_regularization(
    a: &mut [f64],
    k: usize,
    scale: f64,
    strength_l2: f32,
    strength_l1: f32,
    strength_temporal: f32,
) -> f64 {
    debug_assert!(a.len() >= k * k);
    let l2_weight = L2_MULTIPLIER * scale * strength_l2 as f64;
    let l1_weight = L1_MULTIPLIER * scale * strength_l1 as f64;
    let temporal_scale = TEMPORAL_MULTIPLIER * scale * strength_temporal as f64;

    for i in 0..k {
        for j in 0..k {
            a[i * k + j] += l1_weight;
            if i == j {
                a[i * k + j] += l2_weight + temporal_scale;
            }
        }
    }
    temporal_scale
}
