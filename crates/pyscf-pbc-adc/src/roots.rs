//! Dense non-symmetric root solver shared by the IP/EA manifolds.
//!
//! Upstream solves the ADC secular problem with `davidson_nosym1` because the
//! matrix is genuinely NON-SYMMETRIC (0.048 on upstream's own IP matrix at
//! the Gate-D fixture — measured, not assumed). The dense path therefore
//! needs a non-symmetric eigensolver: host-only faer `Eigen::new_from_real`
//! (same ALG-06 justification as 17-02).
//!
//! Scope boundary, stated loudly: only the REAL-spectrum path is implemented.
//! The fixture problems are essentially real (`max|im(H)| ≤ 2.2e-15`) with
//! real spectra (imaginary parts ≤ 1e-21), matching upstream's own `.real`
//! reporting. A genuinely complex matrix (general-k without time-reversal
//! pairing) is REFUSED here — silently dropping imaginary parts would be the
//! plausible-wrong-number this phase guards against. The complex-nosym dense
//! solve is the documented extension, sharing the sigma code unchanged.

use faer::{Mat, linalg::solvers::Eigen};

use crate::error::PbcAdcError;

/// Boundary: matrices with `max|im| > IMAG_BOUND` are refused.
pub const IMAG_BOUND: f64 = 1e-10;

/// Boundary: roots with `|imag| > ROOT_IMAG_BOUND` are refused (upstream
/// reports `.real`; a silently-complex converged root cannot pass here).
pub const ROOT_IMAG_BOUND: f64 = 1e-6;

/// Dense roots of a real non-symmetric matrix: lowest `nroots` eigenvalues
/// sorted by real part, with their (right) eigenvector columns.
///
/// `hre` is row-major `dim × dim`; `him` (same shape) must satisfy the
/// [`IMAG_BOUND`]. Returns `(energies, eigvec_columns)` with columns in the
/// same sorted order (F-order-compatible: column r holds the vector for
/// `energies[r]`, as `(re, im)` pairs).
pub fn nosym_roots(
    hre: &[f64],
    him: &[f64],
    dim: usize,
    nroots: usize,
) -> Result<(Vec<f64>, Vec<Vec<(f64, f64)>>), PbcAdcError> {
    if hre.len() != dim * dim || him.len() != dim * dim || dim == 0 {
        return Err(PbcAdcError::ShapeMismatch { expected: dim * dim, got: hre.len().min(him.len()) });
    }
    if nroots > dim {
        return Err(PbcAdcError::ShapeMismatch { expected: dim, got: nroots });
    }
    let max_im: f64 = him.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if max_im > IMAG_BOUND {
        return Err(PbcAdcError::NotYetImplemented {
            module: "adc complex non-symmetric dense solve (general-k without time-reversal pairing)",
        });
    }
    let mat = Mat::<f64>::from_fn(dim, dim, |i, j| hre[i * dim + j]);
    let evd = Eigen::new_from_real(mat.as_ref()).map_err(|e| {
        PbcAdcError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!("adc nosym eigendecomposition failed: {e:?}")),
        ))
    })?;
    let s = evd.S();
    let u = evd.U();
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        s[a].re
            .partial_cmp(&s[b].re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(s[a].im.abs().partial_cmp(&s[b].im.abs()).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut energies = Vec::with_capacity(nroots);
    let mut columns = Vec::with_capacity(nroots);
    for r in 0..nroots {
        let j = order[r];
        if s[j].im.abs() > ROOT_IMAG_BOUND {
            return Err(PbcAdcError::ShapeMismatch { expected: 0, got: (s[j].im.abs() * 1e12) as usize });
        }
        energies.push(s[j].re);
        let mut col = Vec::with_capacity(dim);
        for i in 0..dim {
            col.push((u[(i, j)].re, u[(i, j)].im));
        }
        columns.push(col);
    }
    Ok((energies, columns))
}
