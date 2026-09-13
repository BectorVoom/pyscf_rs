//! Finite-difference electron-phonon matrix (`pbc/eph/eph_fd.py`, 181 l).
//!
//! Upstream displaces each Cartesian coordinate by `±disp/2` (`gen_cells` with
//! `disp/2.0`), rebuilds the mean field in every displaced cell (`run_mfs`),
//! and central-differences the Fock matrices (`get_vmat:
//! (V⁺−V⁻)/disp`) and the nuclear Hessian (`run_hess`). This module owns the
//! difference itself — the one line every caller shares — plus the floor that
//! comes with it.
//!
//! ## Cancellation floor (18-01 Task-3 discipline)
//!
//! Step size: `disp = 1e-4` Bohr (`kernel(mf, disp=1e-4)`; cells at `±disp/2`).
//! A central difference `(E⁺−E⁻)/disp` cancels ~4 digits at this step
//! (`eps_machine·|E|/disp ≈ 2.2e-16·10/1e-4 ≈ 2e-11` per element for
//! Hartree-scale quantities), while truncation is `O(disp²) ≈ 1e-8` relative
//! to the curvature scale. The implied element floor is therefore **~1e-8 in
//! absolute Fock-derivative units on Hartree-scale systems** — four orders
//! looser than X2C's 1e-8-on-the-energy gate character, and it must never
//! inherit X2C's number (19-18). The truncation side is asserted in tests
//! (quadratic convergence under step halving); the cancellation side is
//! recorded here, not asserted (it is hardware arithmetic, not code).

use crate::error::PbcEphError;

/// Upstream default displacement (`eph_fd.kernel(mf, disp=1e-4)`), Bohr.
pub const EPH_FD_DEFAULT_DISP: f64 = 1e-4;

/// Implied absolute element floor of the central difference at the default
/// step on Hartree-scale systems (see module docs). Recorded, not asserted.
pub const EPH_FD_IMPLIED_FLOOR: f64 = 1e-8;

/// Central-difference derivative `(q_plus − q_minus)/disp`, element-wise.
///
/// `q_plus`/`q_minus` are the quantity (Fock matrix, Hessian, MO energy) at
/// `+disp/2`/`−disp/2` along one Cartesian mode. Lengths must agree; `disp`
/// must be positive and finite — a zero step divides by zero, a negative step
/// flips the sign, both silently. Refused here.
pub fn central_difference(q_plus: &[f64], q_minus: &[f64], disp: f64) -> Result<Vec<f64>, PbcEphError> {
    if q_plus.len() != q_minus.len() {
        return Err(PbcEphError::ShapeMismatch {
            expected: q_plus.len(),
            got: q_minus.len(),
        });
    }
    if !(disp > 0.0) || !disp.is_finite() {
        return Err(PbcEphError::ShapeMismatch { expected: 1, got: 0 });
    }
    Ok(q_plus
        .iter()
        .zip(q_minus.iter())
        .map(|(a, b)| (a - b) / disp)
        .collect())
}

/// Electron-phonon coupling matrix from finite nuclear displacements.
///
/// Ports the `get_vmat` contraction shape: `e_plus`/`e_minus` hold the MO
/// energies at `+disp`/`-disp` along one mode; returns the central-difference
/// derivative `dE/dR` at the default step.
pub fn eph_fd_coupling(e_plus: &[f64], e_minus: &[f64], disp: f64) -> Result<Vec<f64>, PbcEphError> {
    central_difference(e_plus, e_minus, disp)
}
