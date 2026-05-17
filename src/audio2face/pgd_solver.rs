//! `PgdBlendshapeSolver` — Rust port of `PgdBlendshapeSolver.cs`.
//!
//! Hybrid solver: precomputed `M = inv(A_noTemporal) · D^T` provides a
//! warm-started initial guess, then projected gradient descent with
//! the *full* `A` (including temporal) refines under box constraints
//! `0 ≤ x ≤ 1`. Cheaper per frame than full BVLS — the trade is
//! lower accuracy on tightly-coupled blendshapes.
//!
//! Step size is the inverse of an estimate of `A`'s largest
//! eigenvalue (50-iter power iteration), so a single descent step
//! provably doesn't overshoot.

use super::solver_math::{compute_bounding_box_diagonal, compute_dt_d, compute_transpose};
use super::solver_trait::BlendshapeSolver;

const MAX_ITERATIONS: usize = 10;
const CONVERGENCE_THRESHOLD: f64 = 1e-6;
const POWER_ITER_STEPS: usize = 50;

pub struct PgdBlendshapeSolver {
    /// Number of active blendshapes (K).
    active_count: usize,
    /// Number of masked vertex components (V).
    masked_position_count: usize,
    /// `Dᵀ[K×V]` row-major.
    dt: Vec<f64>,
    /// Full system matrix `A = DᵀD + L2 + L1 + temporal` `[K×K]`.
    a: Vec<f64>,
    /// `M = inv(A_noTemporal) · Dᵀ` `[K×V]` row-major. Warm-start
    /// for the PGD inner loop.
    m: Vec<f64>,
    /// Step size (1 / λ_max(A)).
    step_size: f64,
    /// Temporal scale value used when assembling `b`.
    temporal_scale: f64,
    /// Previous-frame weights, length `K`.
    prev_weights: Vec<f64>,
    // Pre-allocated working buffers (avoid per-frame allocation).
    dt_delta: Vec<f64>,
    x_buf: Vec<f64>,
    b_buf: Vec<f64>,
    result_buf: Vec<f32>,
    pb: Vec<f64>,
}

impl PgdBlendshapeSolver {
    /// Build a solver. `delta_matrix` is `[V×K]` row-major (masked
    /// vertex components × active blendshapes). See
    /// `BlendshapeData.cs:DeltaMatrix` for the layout.
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

        // Bounding-box scale — exactly mirrors the C# branching logic.
        let mut scale = 1.0f64;
        let n_verts = masked_position_count / 3;
        if n_verts > 0 && template_bb_size > 0.0 {
            let bb_size = compute_bounding_box_diagonal(masked_neutral_flat);
            if bb_size > 0.0 {
                let r = bb_size / template_bb_size as f64;
                scale = r * r;
            }
        }

        // Build Dᵀ[K×V] from D[V×K].
        let mut dt = vec![0.0f64; k * v];
        compute_transpose(delta_matrix, v, k, &mut dt);

        // DᵀD[K×K].
        let mut dtd = vec![0.0f64; k * k];
        compute_dt_d(&dt, k, v, &mut dtd);

        // A_noTemporal = DᵀD + L2 + L1 (no temporal term yet).
        // We need this both (a) for the LU solve that builds M, and
        // (b) as the base for the full A = A_noTemporal + temporal_diag.
        let mut a_no_temp = dtd.clone();
        let l2_weight = super::solver_math::L2_MULTIPLIER * scale * strength_l2 as f64;
        let l1_weight = super::solver_math::L1_MULTIPLIER * scale * strength_l1 as f64;
        for i in 0..k {
            for j in 0..k {
                a_no_temp[i * k + j] += l1_weight;
                if i == j {
                    a_no_temp[i * k + i] += l2_weight;
                }
            }
        }

        // Full A = A_noTemporal + temporal_diag.
        let temporal_scale =
            super::solver_math::TEMPORAL_MULTIPLIER * scale * strength_temporal as f64;
        let mut a = a_no_temp.clone();
        for i in 0..k {
            a[i * k + i] += temporal_scale;
        }

        // Step size from power-iteration eigenvalue estimate.
        let max_eig = estimate_max_eigenvalue(&a, k, POWER_ITER_STEPS);
        let step_size = 1.0 / max_eig;

        // Compute M = inv(A_noTemporal) · Dᵀ via LU.
        // A_noTemp is K×K (small), so LU is trivial.
        let mut lu = a_no_temp.clone();
        let mut pivot = vec![0usize; k];
        lu_decompose(&mut lu, k, &mut pivot);

        let mut m = vec![0.0f64; k * v];
        let mut col = vec![0.0f64; k];
        let mut sol_col = vec![0.0f64; k];
        let mut pb = vec![0.0f64; k];
        for vi in 0..v {
            for ki in 0..k {
                col[ki] = dt[ki * v + vi];
            }
            lu_solve(&lu, k, &pivot, &col, &mut sol_col, &mut pb);
            for ki in 0..k {
                m[ki * v + vi] = sol_col[ki];
            }
        }

        Self {
            active_count: k,
            masked_position_count: v,
            dt,
            a,
            m,
            step_size,
            temporal_scale,
            prev_weights: vec![0.0f64; k],
            dt_delta: vec![0.0f64; k],
            x_buf: vec![0.0f64; k],
            b_buf: vec![0.0f64; k],
            result_buf: vec![0.0f32; k],
            pb,
        }
    }
}

impl BlendshapeSolver for PgdBlendshapeSolver {
    fn solve(&mut self, delta: &[f32]) -> &[f32] {
        assert_eq!(
            delta.len(),
            self.masked_position_count,
            "delta length must match V"
        );
        let k = self.active_count;
        let v = self.masked_position_count;

        // 1. dt_delta = Dᵀ · delta
        for i in 0..k {
            let mut sum = 0.0f64;
            for j in 0..v {
                sum += self.dt[i * v + j] * delta[j] as f64;
            }
            self.dt_delta[i] = sum;
        }

        // 2. Initial guess from precomputed M, clipped to [0, 1].
        for i in 0..k {
            let mut sum = 0.0f64;
            for j in 0..v {
                sum += self.m[i * v + j] * delta[j] as f64;
            }
            self.x_buf[i] = sum.clamp(0.0, 1.0);
        }

        // 3. b = Dᵀ·delta + temporal · prev (so the optimum drifts
        //    toward last frame's weights).
        for i in 0..k {
            self.b_buf[i] = self.dt_delta[i] + self.temporal_scale * self.prev_weights[i];
        }

        // 4. PGD inner loop. Each iter:
        //    - g = A·x - b  (gradient of ½‖Ax-b‖² wrt x; symmetric A)
        //    - x ← clip(x - step·g, 0, 1)
        //    - early-exit when no slot moves more than CONVERGENCE.
        for _iter in 0..MAX_ITERATIONS {
            let mut max_change = 0.0f64;
            for i in 0..k {
                let mut g = -self.b_buf[i];
                for j in 0..k {
                    g += self.a[i * k + j] * self.x_buf[j];
                }
                let old_x = self.x_buf[i];
                self.x_buf[i] = (old_x - self.step_size * g).clamp(0.0, 1.0);
                let change = (self.x_buf[i] - old_x).abs();
                if change > max_change {
                    max_change = change;
                }
            }
            if max_change < CONVERGENCE_THRESHOLD {
                break;
            }
        }

        // 5. Persist prev_weights for next call's temporal pull.
        for i in 0..k {
            self.prev_weights[i] = self.x_buf[i];
            self.result_buf[i] = self.x_buf[i] as f32;
        }
        &self.result_buf
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

/// Power-iteration estimate of the largest eigenvalue of a symmetric
/// `K×K` matrix. 50 iterations is more than enough — the SDK uses
/// the same count and the matrix is small (K is the active blendshape
/// count, typically < 60).
fn estimate_max_eigenvalue(a: &[f64], k: usize, iterations: usize) -> f64 {
    let mut v = vec![0.0f64; k];
    v[0] = 1.0;
    let mut av = vec![0.0f64; k];

    for _ in 0..iterations {
        for i in 0..k {
            let mut sum = 0.0f64;
            for j in 0..k {
                sum += a[i * k + j] * v[j];
            }
            av[i] = sum;
        }
        let mut norm = 0.0f64;
        for &x in &av {
            norm += x * x;
        }
        norm = norm.sqrt();
        if norm < 1e-15 {
            break;
        }
        for i in 0..k {
            v[i] = av[i] / norm;
        }
    }

    let mut lambda = 0.0f64;
    for i in 0..k {
        let mut avi = 0.0f64;
        for j in 0..k {
            avi += a[i * k + j] * v[j];
        }
        lambda += v[i] * avi;
    }
    lambda.max(1e-10)
}

/// LU decomposition with partial pivoting. Overwrites `a` in-place.
/// `pivot[i]` records the source row that ended up at row `i`.
fn lu_decompose(a: &mut [f64], n: usize, pivot: &mut [usize]) {
    for i in 0..n {
        pivot[i] = i;
    }
    for col in 0..n {
        let mut max_val = a[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_row != col {
            pivot.swap(col, max_row);
            for j in 0..n {
                let (a_ij, a_kj) = (a[col * n + j], a[max_row * n + j]);
                a[col * n + j] = a_kj;
                a[max_row * n + j] = a_ij;
            }
        }
        let diag_val = a[col * n + col];
        if diag_val.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..n {
            a[row * n + col] /= diag_val;
            for j in (col + 1)..n {
                let l = a[row * n + col];
                let u = a[col * n + j];
                a[row * n + j] -= l * u;
            }
        }
    }
}

/// Solve `LU · x = b` via forward + back substitution. `pb` is a
/// caller-provided scratch buffer of length `n`.
fn lu_solve(lu: &[f64], n: usize, pivot: &[usize], b: &[f64], x: &mut [f64], pb: &mut [f64]) {
    // Permute b.
    for i in 0..n {
        pb[i] = b[pivot[i]];
    }
    // Forward substitution: L·y = pb (L unit-lower).
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= lu[i * n + j] * pb[j];
        }
        pb[i] = sum;
    }
    // Back substitution: U·x = y.
    for i in (0..n).rev() {
        let mut sum = pb[i];
        for j in (i + 1)..n {
            sum -= lu[i * n + j] * x[j];
        }
        x[i] = sum / lu[i * n + i];
    }
}
