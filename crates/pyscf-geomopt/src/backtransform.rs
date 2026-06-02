//! Internal→Cartesian back-transformation (geomeTRIC `internal.py`
//! `newCartesian` analog).
//!
//! **Algorithm ported from geomeTRIC** (BSD 3-clause Non-AI License — see the
//! crate-level record in `lib.rs`). Given a target internal-coordinate
//! displacement `dq`, find the Cartesian displacement `dx` that produces it.
//! Because `q(x)` is nonlinear, this is an iterative fixed-point:
//!
//! ```text
//!   x_0 = x_init
//!   loop:
//!     dx   = G⁻(x_k) · B(x_k) · (q_target − q(x_k))      # the Bᵀ-projected step
//!     x_{k+1} = x_k + dx
//!     if |q(x_{k+1}) − q_target| < tol: done
//! ```
//!
//! In geomeTRIC the step uses `Bᵀ G⁻ dq` (the right pseudo-inverse of B);
//! here `dx = Bᵀ · (G⁻ · dq_residual)` so the Cartesian step is the
//! minimum-norm solution of `B dx = dq_residual`. The B-matrix and `G⁻` are
//! rebuilt each iteration (geomeTRIC's `iterative` flag default).
//!
//! Per Pitfall 1/2 every reduction materialises then routes through
//! `pyscf_algebra::oracle_sum`/`oracle_dot` — no bare `+=`.

use crate::bmatrix;
use crate::error::GeomError;
use crate::internals::{self, Primitive};
use pyscf_algebra::oracle_dot;

/// Max back-transform iterations before declaring divergence (geomeTRIC uses
/// `microiter=20`).
pub const MAX_BT_ITERS: usize = 50;

/// Convergence tolerance on the internal-coordinate residual `|dq|` (RMS),
/// matching geomeTRIC's `1e-10` Cartesian back-transform tolerance.
pub const BT_TOL: f64 = 1e-10;

/// Subtract two internal-coordinate vectors, wrapping angular components
/// (Angle/Dihedral) into `(−π, π]` so a torsion crossing ±π doesn't produce a
/// spurious 2π residual.
fn internal_delta(prims: &[Primitive], q_target: &[f64], q_now: &[f64]) -> Vec<f64> {
    prims
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mut d = q_target[i] - q_now[i];
            if matches!(p, Primitive::Angle(..) | Primitive::Dihedral(..)) {
                // wrap into (−π, π]
                while d > std::f64::consts::PI {
                    d -= 2.0 * std::f64::consts::PI;
                }
                while d <= -std::f64::consts::PI {
                    d += 2.0 * std::f64::consts::PI;
                }
            }
            d
        })
        .collect()
}

/// RMS of a slice via the oracle-ordered reduction.
fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let sq: Vec<f64> = v.iter().map(|x| x * x).collect();
    (oracle_sum_helper(&sq) / v.len() as f64).sqrt()
}

fn oracle_sum_helper(xs: &[f64]) -> f64 {
    pyscf_algebra::oracle_sum(xs)
}

/// Back-transform an internal-coordinate displacement `dq` (added to the
/// internals at `coords`) into a new Cartesian geometry.
///
/// Returns the new `(natm, 3)` Cartesian geometry.
///
/// # Errors
/// [`GeomError::BacktransformDiverged`] if the fixed-point fails to converge
/// within [`MAX_BT_ITERS`] (a pathological / ill-conditioned B-matrix,
/// T-07-13).
pub fn to_cartesian(
    prims: &[Primitive],
    coords: &[[f64; 3]],
    dq: &[f64],
) -> Result<Vec<[f64; 3]>, GeomError> {
    let natm = coords.len();
    let ncart = 3 * natm;
    let nint = prims.len();

    // Target internals = current internals + requested displacement.
    let q0 = internals::values(prims, coords);
    let q_target: Vec<f64> = (0..nint).map(|i| q0[i] + dq[i]).collect();

    let mut x: Vec<[f64; 3]> = coords.to_vec();

    for iter in 0..MAX_BT_ITERS {
        let q_now = internals::values(prims, &x);
        let residual = internal_delta(prims, &q_target, &q_now);
        if rms(&residual) < BT_TOL {
            return Ok(x);
        }

        // dx = Bᵀ · (G⁻ · residual)   (the minimum-norm Cartesian step).
        let (b, _g, ginv) = bmatrix::build(prims, &x).map_err(GeomError::Scanner)?;

        // w = G⁻ · residual   (nint)
        let mut w = vec![0.0_f64; nint];
        for i in 0..nint {
            let row = &ginv[i * nint..(i + 1) * nint];
            w[i] = oracle_dot(row, &residual);
        }

        // dx = Bᵀ · w   (ncart): column j of B is the j-th Cartesian comp.
        let mut dx = vec![0.0_f64; ncart];
        for j in 0..ncart {
            // Gather column j of B (row-major (nint, ncart)).
            let col: Vec<f64> = (0..nint).map(|i| b[i * ncart + j]).collect();
            dx[j] = oracle_dot(&col, &w);
        }

        // Apply.
        for a in 0..natm {
            x[a][0] += dx[3 * a];
            x[a][1] += dx[3 * a + 1];
            x[a][2] += dx[3 * a + 2];
        }

        // Final guard: on the last iteration, accept if residual already
        // small enough after this update.
        if iter == MAX_BT_ITERS - 1 {
            let q_final = internals::values(prims, &x);
            let r_final = internal_delta(prims, &q_target, &q_final);
            if rms(&r_final) < BT_TOL * 1e3 {
                return Ok(x);
            }
        }
    }

    Err(GeomError::BacktransformDiverged {
        iters: MAX_BT_ITERS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::Primitive;
    use approx::assert_relative_eq;

    #[test]
    fn roundtrip_stretch_h2_bond() {
        // H2: one Distance internal. Displace the bond by +0.1 Bohr and verify
        // the back-transformed geometry has the new bond length.
        let coords = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
        let prims = [Primitive::Distance(0, 1)];
        let dq = [0.1_f64];
        let new = to_cartesian(&prims, &coords, &dq).expect("backtransform");
        let new_len = Primitive::Distance(0, 1).value(&new);
        assert_relative_eq!(new_len, 1.5, epsilon = 1e-8);
    }

    #[test]
    fn zero_displacement_is_identity() {
        let coords = [[0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0]];
        let prims = crate::internals::generate(&coords, &[8, 1, 1]);
        let dq = vec![0.0_f64; prims.len()];
        let new = to_cartesian(&prims, &coords, &dq).expect("backtransform");
        for a in 0..3 {
            for c in 0..3 {
                assert_relative_eq!(new[a][c], coords[a][c], epsilon = 1e-9);
            }
        }
    }
}
