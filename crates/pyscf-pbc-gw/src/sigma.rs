//! Shared screened-interaction / self-energy machinery (`pbc/gw`).
//!
//! The AC and CD routes share the polarizability build and the diagonal
//! correlation self-energy on the imaginary axis; the contour (CD) vs the
//! Padé fit (AC) is the difference, not the whole method (19-11). This module
//! owns the shared half: Gauss-Legendre imaginary-frequency nodes and the
//! diagonal RPA polarizability contraction.

use crate::error::PbcGwError;
use pyscf_algebra::oracle_sum;

/// Gauss-Legendre nodes mapped to the semi-infinite imaginary axis.
///
/// Returns `nomega` positive frequencies `ω_n` via the `x → ω0·(1+x)/(1-x)`
/// map of the `[-1,1]` Gauss-Legendre rule, with `weights` for the same map.
/// Pure host arithmetic; no cubecl (host-loop pattern, D-PBC-29 clause 2).
pub fn imag_grid(nomega: usize, omega0: f64) -> Result<(Vec<f64>, Vec<f64>), PbcGwError> {
    if nomega == 0 {
        return Err(PbcGwError::ShapeMismatch { expected: 1, got: 0 });
    }
    let (xs, ws) = gauss_legendre(nomega);
    let mut omegas = Vec::with_capacity(nomega);
    let mut weights = Vec::with_capacity(nomega);
    for (x, w) in xs.iter().zip(ws.iter()) {
        omegas.push(omega0 * (1.0 + x) / (1.0 - x));
        weights.push(w * 2.0 * omega0 / ((1.0 - x) * (1.0 - x)));
    }
    Ok((omegas, weights))
}

/// Diagonal RPA polarizability `χ0_{ia,ia}(iω)` summed over one occupied–virtual
/// pair set, evaluated at every grid node.
///
/// `e_ia` holds the `nocc·nvir` positive particle-hole gaps; `f_ia` the
/// corresponding oscillator-weighted ERI-diagonal products. Returns the
/// length-`nomega` contraction `Σ_ia f_ia · 2·e_ia/(ω² + e_ia²)` per node,
/// each accumulated through [`oracle_sum`].
pub fn polarizability_diag(
    e_ia: &[f64],
    f_ia: &[f64],
    omegas: &[f64],
) -> Result<Vec<f64>, PbcGwError> {
    if e_ia.len() != f_ia.len() {
        return Err(PbcGwError::ShapeMismatch { expected: e_ia.len(), got: f_ia.len() });
    }
    let mut out = Vec::with_capacity(omegas.len());
    for &w in omegas {
        let mut terms = Vec::with_capacity(e_ia.len());
        for (e, f) in e_ia.iter().zip(f_ia.iter()) {
            terms.push(f * 2.0 * e / (w * w + e * e));
        }
        out.push(oracle_sum(&terms));
    }
    Ok(out)
}

/// Gauss-Legendre rule on `[-1, 1]` by Newton iteration on `P_n` (host scalar
/// code; `n ≤ 256`). Returned ASCENDING (matching `numpy.leggauss` order —
/// the grid is pinned against it element-wise, so order matters).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut xs = vec![0.0f64; n];
    let mut ws = vec![0.0f64; n];
    for i in 1..=n {
        // Initial guess (Abramowitz–Stegun 25.4.30 cos approximation).
        let mut x = (std::f64::consts::PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
        for _ in 0..20 {
            let (p, dp) = legendre_p(n, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let (_, dp) = legendre_p(n, x);
        xs[i - 1] = x;
        ws[i - 1] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    // Newton loop above yields DESCENDING nodes (cos guess starts near +1);
    // reverse to ascending (numpy.leggauss order).
    xs.reverse();
    ws.reverse();
    (xs, ws)
}

/// `P_n(x)` and `P'_n(x)` by three-term recurrence.
fn legendre_p(n: usize, x: f64) -> (f64, f64) {
    let mut p0 = 1.0f64;
    let mut p1 = x;
    for k in 1..n {
        let p2 = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / (k + 1) as f64;
        p0 = p1;
        p1 = p2;
    }
    if n == 0 {
        return (1.0, 0.0);
    }
    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}
