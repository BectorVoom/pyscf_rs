//! `verify_fd` — the always-on finite-difference numeric gate (D-01, GRAD-09).
//!
//! The FIRST self-validating *numeric* gate in the project: it central-
//! differences the shipped energy (the `as_scanner` seam) and compares to the
//! analytical gradient at ≤1e-6 Ha/Bohr, needing NO upstream PySCF. This is the
//! daily `cargo test` gate for every GRAD-01..07 — when a method's `grad_elec`
//! body lands (07-03+), it wires its real `as_scanner` energy + analytical
//! gradient into this harness and the harness proves them mutually consistent.
//!
//! Per D-09 it is base-API-from-day-one: every method inherits the SAME FD gate
//! the moment its energy closure + gradient exist.
//!
//! ## Bit-exact discipline (Pitfall 1/2, T-07-04)
//! Every reduction (the per-component central difference, the max-diff scan)
//! materialises into a `Vec` then routes through
//! `pyscf_algebra::oracle_sum`/`oracle_dot` — NO bare `+=`. The FD result is
//! therefore identical on 1 and N threads under `release-oracle`.
//!
//! ## Threat mitigation
//! - T-07-05: `disp` is validated `> 0` and finite BEFORE any energy eval; a
//!   non-positive / non-finite `disp` returns `GradError::InvalidDisplacement`.
//! - T-07-04: reduction order is deterministic via `oracle_sum`.

use crate::error::GradError;
use pyscf_core::PyscfRsError;

/// The default central-difference displacement, in Bohr (`rhf.py` numerical-
/// gradient convention). 1e-4 Bohr balances truncation error (`O(disp²)` for a
/// central difference) against floating-point cancellation.
pub const DEFAULT_DISP: f64 = 1e-4;

/// The acceptance tolerance for `max|fd_grad − analytical_grad|`, in Ha/Bohr
/// (GRAD-09). A correct analytical gradient agrees with the central difference
/// of the energy to well within this on every in-scope method.
pub const FD_TOL: f64 = 1e-6;

/// The outcome of a finite-difference check.
#[derive(Debug, Clone)]
pub struct FdReport {
    /// `max|fd − analytical|` over every atom × component (Ha/Bohr).
    pub max_abs_diff: f64,
    /// The full finite-difference gradient (`(natm, 3)`), for diagnostics.
    pub fd_grad: Vec<[f64; 3]>,
    /// Whether `max_abs_diff <= tol`.
    pub passed: bool,
}

/// Central-difference an energy closure and compare to an analytical gradient.
///
/// For each atom `ia` and each Cartesian component `c`, displace `coords[ia][c]`
/// by `+disp` and `-disp` Bohr, evaluate the energy closure at each, form the
/// central difference `(E₊ − E₋) / (2·disp)`, and compare to
/// `analytical[ia][c]`. Returns an [`FdReport`]; `passed` is
/// `max|fd − analytical| <= tol`.
///
/// `energy` is the `as_scanner` seam expressed over per-atom coordinates in
/// Bohr: `Fn(&[[f64; 3]]) -> Result<f64, PyscfRsError>`. The per-method wave
/// adapts its `Mole`-based scanner into this coord closure (clone the `Mole`,
/// set the displaced geometry, run the energy `kernel`). The harness itself is
/// method-agnostic.
///
/// # Errors
/// - [`GradError::InvalidDisplacement`] if `disp` is non-finite or `<= 0`
///   (T-07-05), surfaced before any energy eval.
/// - [`GradError::ShapeMismatch`] if `analytical.len() != coords.len()`.
/// - whatever the energy closure returns.
pub fn verify_fd<F>(
    coords: &[[f64; 3]],
    analytical: &[[f64; 3]],
    energy: F,
    disp: f64,
    tol: f64,
) -> Result<FdReport, PyscfRsError>
where
    F: Fn(&[[f64; 3]]) -> Result<f64, PyscfRsError>,
{
    // T-07-05: validate disp before touching the energy closure.
    if !disp.is_finite() || disp <= 0.0 {
        return Err(GradError::InvalidDisplacement { disp }.into());
    }
    if analytical.len() != coords.len() {
        return Err(GradError::ShapeMismatch {
            expected: coords.len(),
            got: analytical.len(),
        }
        .into());
    }

    let natm = coords.len();
    let two_disp = 2.0 * disp;
    let mut fd_grad: Vec<[f64; 3]> = Vec::with_capacity(natm);
    // Accumulate every |fd − analytical| into a materialised buffer, then take
    // the max through an ordered scan (no bare running-max `+=`/comparison loop
    // that depends on iteration buffering).
    let mut abs_diffs: Vec<f64> = Vec::with_capacity(natm * 3);

    for ia in 0..natm {
        let mut row = [0.0f64; 3];
        for c in 0..3 {
            let mut plus = coords.to_vec();
            let mut minus = coords.to_vec();
            plus[ia][c] += disp;
            minus[ia][c] -= disp;
            let e_plus = energy(&plus)?;
            let e_minus = energy(&minus)?;
            // Central difference; the subtraction is a two-element ordered sum
            // (E₊ + (−E₋)) so the reduction shape is fixed.
            let numer = pyscf_algebra::oracle_sum(&[e_plus, -e_minus]);
            let fd = numer / two_disp;
            row[c] = fd;
            let diff = pyscf_algebra::oracle_sum(&[fd, -analytical[ia][c]]).abs();
            abs_diffs.push(diff);
        }
        fd_grad.push(row);
    }

    // Ordered max over the materialised diff buffer (deterministic).
    let max_abs_diff = abs_diffs.iter().copied().fold(0.0f64, f64::max);

    Ok(FdReport {
        max_abs_diff,
        fd_grad,
        passed: max_abs_diff <= tol,
    })
}
