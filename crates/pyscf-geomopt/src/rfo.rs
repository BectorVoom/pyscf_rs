//! RFO step + trust-radius + neg-eigenvalue tracking + BFGS update.
//!
//! Bodies land in Task 2 of plan 07-04; this Task-1 skeleton ships the
//! [`BfgsHessian`] type + the [`rfo_step`] signature so `lib.rs` compiles.
//! Task 2 fills the augmented-Hessian eigen-step (via
//! `pyscf_algebra::eigh_gen`), the trust-radius quality-factor update, the
//! negative-eigenvalue shift `v0`, the BFGS update, and the `rfo`/`conv`
//! unit tests.

use crate::converge::ConvParams;

/// A BFGS approximate Hessian in the internal-coordinate basis
/// (`(nint, nint)`, row-major).
#[derive(Debug, Clone)]
pub struct BfgsHessian {
    /// The Hessian matrix, row-major `(nint, nint)`.
    pub h: Vec<f64>,
    /// Dimension.
    pub nint: usize,
    /// Number of BFGS updates applied (capped at `MAX_BFGS_UPDATES`).
    pub n_updates: usize,
}

/// Max BFGS updates before the history is reset (geomeTRIC `max_updates=100`).
pub const MAX_BFGS_UPDATES: usize = 100;

impl BfgsHessian {
    /// An identity initial Hessian (`nint × nint`).
    pub fn identity(nint: usize) -> Self {
        let mut h = vec![0.0_f64; nint * nint];
        for i in 0..nint {
            h[i * nint + i] = 1.0;
        }
        Self {
            h,
            nint,
            n_updates: 0,
        }
    }

    /// BFGS update (Task-1 skeleton: no-op; Task 2 fills the rank-2 update).
    pub fn bfgs_update(&mut self, _dq: &[f64], _dg: &[f64]) {
        // Task-1 skeleton stub.
    }
}

/// Compute the RFO step in internals (Task-1 skeleton: returns a zero step;
/// Task 2 fills the augmented-Hessian eigen-step + trust-radius + neg-eig
/// shift). Returns `(dq, predicted_energy_change, new_trust)`.
#[allow(clippy::type_complexity)]
pub fn rfo_step(
    hessian: &BfgsHessian,
    _g_int: &[f64],
    trust: f64,
    _params: &ConvParams,
    _actual_de: Option<f64>,
    _prev: Option<&(f64, Vec<f64>, Vec<f64>)>,
) -> (Vec<f64>, f64, f64) {
    // Task-1 skeleton stub: zero step, unchanged trust.
    (vec![0.0_f64; hessian.nint], 0.0, trust)
}
