//! Padé analytic continuation (`pbc/gw/krgw_ac.py` shared machinery).
//!
//! The AC route evaluates the correlation self-energy on an imaginary
//! frequency grid and continues it to the real axis by a Padé fit. This
//! module owns the fit: given `N` imaginary-grid values `sigma(iω_n)`, build
//! the `[M/M]` Padé approximant (Thiele continued-fraction form, which is
//! division-only and needs no linear solve) and evaluate it at a real
//! frequency. Every reduction routes through [`pyscf_algebra::oracle_sum`].

use crate::error::PbcGwError;
use pyscf_algebra::oracle_sum;

/// Evaluate the Thiele-continued-fraction Padé approximant of `values` at
/// real frequency `omega`.
///
/// `grid` holds the imaginary-grid nodes `iω_n` (as positive reals `ω_n`);
/// `values` the self-energy samples there. Both have length `npoints ≥ 2`.
/// Returns the continued real-axis value. Fails with
/// [`PbcGwError::PadeFailure`] on a vanishing denominator rather than
/// returning an infinity.
pub fn pade_continue(grid: &[f64], values: &[f64], omega: f64) -> Result<f64, PbcGwError> {
    let n = grid.len();
    if n < 2 || values.len() != n {
        return Err(PbcGwError::ShapeMismatch { expected: n, got: values.len() });
    }
    // Reciprocal-difference table (Thiele): g[i] updated in place.
    let mut g = values.to_vec();
    for k in 0..n - 1 {
        for i in (k + 1..n).rev() {
            let denom = g[i] - g[k];
            if denom == 0.0 || !denom.is_finite() {
                return Err(PbcGwError::PadeFailure {
                    reason: format!("vanishing reciprocal difference at k={k}, i={i}"),
                });
            }
            g[i] = (grid[i] - grid[k]) / denom;
        }
    }
    // Evaluate the continued fraction bottom-up at omega (ordered accumulation).
    let mut tail = 0.0f64;
    for i in (1..n).rev() {
        let denom = g[i] + tail;
        if denom == 0.0 || !denom.is_finite() {
            return Err(PbcGwError::PadeFailure {
                reason: format!("vanishing continued-fraction denominator at i={i}"),
            });
        }
        tail = (omega - grid[i - 1]) / denom;
    }
    Ok(oracle_sum(&[g[0], tail]))
}
