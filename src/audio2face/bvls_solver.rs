//! `BvlsBlendshapeSolver` — Rust port of `BvlsBlendshapeSolver.cs`.
//!
//! Active-set Bounded-Variable Least Squares — matches scipy's
//! `lsq_linear(method="bvls")` in spirit. Higher per-frame cost than
//! PGD, but more accurate on tightly-coupled blendshapes (where
//! several poses can produce overlapping vertex deltas).
//!
//! Algorithm sketch:
//! 1. Start with all variables at their bound (`x = 0`, all "fixed").
//! 2. Repeat:
//!    a. Compute gradient `g = A·x - b`.
//!    b. Find the most-violating fixed variable (negative g for
//!       lower-bound, positive g for upper-bound). If none, optimal.
//!    c. Free that variable.
//!    d. Solve the unconstrained sub-problem on the free set via
//!       Cholesky. If the sub-solution lies inside `[0, 1]`,
//!       commit. Otherwise step a fraction and re-fix the worst
//!       offender at its bound; repeat until in-bounds.

use super::solver_math::{
    apply_regularization, compute_bounding_box_diagonal, compute_dt_d, compute_transpose,
};
use super::solver_trait::BlendshapeSolver;

const TOLERANCE: f64 = 1e-10;
const MAX_OUTER_ITERATIONS: usize = 200;

pub struct BvlsBlendshapeSolver {
    /// Active blendshape count (K).
    active_count: usize,
    /// Masked vertex-component count (V).
    masked_position_count: usize,
    /// `Dᵀ[K×V]` row-major.
    dt: Vec<f64>,
    /// `A[K×K] = DᵀD + L2 + L1 + temporal` row-major.
    a: Vec<f64>,
    temporal_scale: f64,
    prev_weights: Vec<f64>,

    // Pre-allocated working buffers.
    b: Vec<f64>,
    x: Vec<f64>,
    free: Vec<bool>,
    g: Vec<f64>,
    free_idx: Vec<usize>,
    /// Sub-system matrix `A_free[free_count × free_count]`.
    aff_max: Vec<f64>,
    /// Sub-system right-hand side `b_free[free_count]`.
    rhs_max: Vec<f64>,
    /// Cholesky scratch (L, y, x).
    chol_l: Vec<f64>,
    chol_y: Vec<f64>,
    chol_x: Vec<f64>,
    /// Output buffer (returned from solve()).
    result: Vec<f32>,
}

impl BvlsBlendshapeSolver {
    pub fn new(
        delta_matrix: &[f32],
        masked_position_count: usize,
        active_count: usize,
        masked_neutral_flat: &[f32],
        template_bb_size: f32,
        strength_l2: f32,
        strength_l1: f32,
        strength_temporal: f32,
    ) -> Self {
        assert_eq!(
            delta_matrix.len(),
            masked_position_count * active_count,
            "delta_matrix must be V*K elements"
        );

        let k = active_count;
        let v = masked_position_count;

        let mut dt = vec![0.0f64; k * v];
        compute_transpose(delta_matrix, v, k, &mut dt);

        let mut dtd = vec![0.0f64; k * k];
        compute_dt_d(&dt, k, v, &mut dtd);

        // Bounding-box scale. C# uses ternary: `bb>0 && tmpl>0 ? (bb/tmpl)² : 0`.
        let bb_size = compute_bounding_box_diagonal(masked_neutral_flat);
        let scale = if bb_size > 0.0 && template_bb_size > 0.0 {
            let r = bb_size / template_bb_size as f64;
            r * r
        } else {
            0.0
        };

        let mut a = dtd;
        let temporal_scale = apply_regularization(
            &mut a,
            k,
            scale,
            strength_l2,
            strength_l1,
            strength_temporal,
        );

        Self {
            active_count: k,
            masked_position_count: v,
            dt,
            a,
            temporal_scale,
            prev_weights: vec![0.0f64; k],
            b: vec![0.0f64; k],
            x: vec![0.0f64; k],
            free: vec![false; k],
            g: vec![0.0f64; k],
            free_idx: vec![0usize; k],
            aff_max: vec![0.0f64; k * k],
            rhs_max: vec![0.0f64; k],
            chol_l: vec![0.0f64; k * k],
            chol_y: vec![0.0f64; k],
            chol_x: vec![0.0f64; k],
            result: vec![0.0f32; k],
        }
    }
}

impl BlendshapeSolver for BvlsBlendshapeSolver {
    fn solve(&mut self, delta: &[f32]) -> &[f32] {
        assert_eq!(
            delta.len(),
            self.masked_position_count,
            "delta length must match V"
        );
        let k = self.active_count;
        let v = self.masked_position_count;

        // b = Dᵀ·delta + temporal · prev_weights
        for i in 0..k {
            let mut sum = 0.0f64;
            for j in 0..v {
                sum += self.dt[i * v + j] * delta[j] as f64;
            }
            self.b[i] = sum + self.temporal_scale * self.prev_weights[i];
        }

        // Initialize all fixed at 0.
        for x in &mut self.x[..k] {
            *x = 0.0;
        }
        for f in &mut self.free[..k] {
            *f = false;
        }

        for _outer in 0..MAX_OUTER_ITERATIONS {
            // Gradient g = A·x - b.
            for i in 0..k {
                let mut sum = 0.0f64;
                for j in 0..k {
                    sum += self.a[i * k + j] * self.x[j];
                }
                self.g[i] = sum - self.b[i];
            }

            // Pick the most-violating fixed variable. For a variable
            // currently at its lower bound (x ≈ 0), violation is `-g`
            // (gradient pushes it up); at upper bound (x ≈ 1) it's
            // `g` (gradient pushes it down).
            let mut best_idx: i64 = -1;
            let mut best_violation = TOLERANCE;
            for i in 0..k {
                if self.free[i] {
                    continue;
                }
                let violation = if self.x[i] <= 0.0 {
                    -self.g[i]
                } else {
                    self.g[i]
                };
                if violation > best_violation {
                    best_violation = violation;
                    best_idx = i as i64;
                }
            }
            if best_idx < 0 {
                break; // KKT satisfied.
            }
            let best_idx = best_idx as usize;
            self.free[best_idx] = true;

            // Inner loop: solve the unconstrained sub-system on the
            // current free set; if any free variable lands outside
            // [0, 1], step partway and re-fix the worst offender.
            loop {
                let mut free_count = 0;
                for i in 0..k {
                    if self.free[i] {
                        self.free_idx[free_count] = i;
                        free_count += 1;
                    }
                }
                if free_count == 0 {
                    break;
                }

                // Build A_free and b_free, accounting for fixed-variable
                // contributions on the rhs.
                for fi in 0..free_count {
                    let ii = self.free_idx[fi];
                    self.rhs_max[fi] = self.b[ii];
                    for fj in 0..free_count {
                        self.aff_max[fi * free_count + fj] = self.a[ii * k + self.free_idx[fj]];
                    }
                    for j in 0..k {
                        if !self.free[j] {
                            self.rhs_max[fi] -= self.a[ii * k + j] * self.x[j];
                        }
                    }
                }

                if !cholesky_solve(
                    &self.aff_max,
                    &self.rhs_max,
                    free_count,
                    &mut self.chol_l,
                    &mut self.chol_y,
                    &mut self.chol_x,
                ) {
                    // Indefinite sub-system — undo the latest free
                    // and break, mirroring C#.
                    self.free[best_idx] = false;
                    break;
                }

                // Test in-bounds-ness; if any free var fell outside
                // [0, 1], step the smallest fraction that hits a
                // bound and re-fix that variable.
                let mut all_in_bounds = true;
                let mut min_alpha = 1.0f64;
                let mut worst_free_idx: i64 = -1;
                for fi in 0..free_count {
                    let cx = self.chol_x[fi];
                    if cx < 0.0 || cx > 1.0 {
                        all_in_bounds = false;
                        let xi = self.x[self.free_idx[fi]];
                        let alpha = if cx < 0.0 {
                            xi / (xi - cx)
                        } else {
                            (1.0 - xi) / (cx - xi)
                        };
                        if alpha < min_alpha {
                            min_alpha = alpha;
                            worst_free_idx = fi as i64;
                        }
                    }
                }

                if all_in_bounds {
                    for fi in 0..free_count {
                        self.x[self.free_idx[fi]] = self.chol_x[fi];
                    }
                    break;
                }

                // Step partway toward the unconstrained solution.
                for fi in 0..free_count {
                    let idx = self.free_idx[fi];
                    self.x[idx] += min_alpha * (self.chol_x[fi] - self.x[idx]);
                }
                let worst_global_idx = self.free_idx[worst_free_idx as usize];
                self.x[worst_global_idx] = if self.chol_x[worst_free_idx as usize] < 0.0 {
                    0.0
                } else {
                    1.0
                };
                self.free[worst_global_idx] = false;
            }
        }

        for i in 0..k {
            self.prev_weights[i] = self.x[i];
            self.result[i] = self.x[i] as f32;
        }
        &self.result
    }

    fn reset_temporal(&mut self) {
        for w in &mut self.prev_weights {
            *w = 0.0;
        }
    }

    fn save_temporal(&self) -> Vec<f64> {
        self.prev_weights.clone()
    }

    fn restore_temporal(&mut self, saved: &[f64]) {
        assert_eq!(saved.len(), self.prev_weights.len());
        self.prev_weights.copy_from_slice(saved);
    }

    fn active_count(&self) -> usize {
        self.active_count
    }
}

/// Cholesky-solve `a · x = b` where `a` is symmetric positive-definite,
/// using caller-provided scratch buffers `chol_l` (size ≥ n²),
/// `chol_y` (size ≥ n), `chol_x` (size ≥ n). Result is written into
/// `chol_x[0..n]`.
///
/// Returns `false` if the matrix isn't SPD (a diagonal goes ≤ 0
/// during decomposition), in which case BVLS rolls the latest
/// `free` flag back.
fn cholesky_solve(
    a: &[f64],
    b: &[f64],
    n: usize,
    chol_l: &mut [f64],
    chol_y: &mut [f64],
    chol_x: &mut [f64],
) -> bool {
    debug_assert!(a.len() >= n * n);
    debug_assert!(b.len() >= n);
    debug_assert!(chol_l.len() >= n * n);
    debug_assert!(chol_y.len() >= n);
    debug_assert!(chol_x.len() >= n);

    // Copy a into chol_l — we'll factor in place.
    chol_l[..n * n].copy_from_slice(&a[..n * n]);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f64;
            for p in 0..j {
                sum += chol_l[i * n + p] * chol_l[j * n + p];
            }
            if i == j {
                let diag = chol_l[i * n + i] - sum;
                if diag <= 0.0 {
                    return false;
                }
                chol_l[i * n + j] = diag.sqrt();
            } else {
                chol_l[i * n + j] = (chol_l[i * n + j] - sum) / chol_l[j * n + j];
            }
        }
        // Zero the upper triangle (we only use the lower).
        for j in (i + 1)..n {
            chol_l[i * n + j] = 0.0;
        }
    }

    // Forward: L · y = b.
    for i in 0..n {
        let mut sum = 0.0f64;
        for j in 0..i {
            sum += chol_l[i * n + j] * chol_y[j];
        }
        chol_y[i] = (b[i] - sum) / chol_l[i * n + i];
    }
    // Back: Lᵀ · x = y.
    for i in (0..n).rev() {
        let mut sum = 0.0f64;
        for j in (i + 1)..n {
            sum += chol_l[j * n + i] * chol_x[j];
        }
        chol_x[i] = (chol_y[i] - sum) / chol_l[i * n + i];
    }
    true
}
