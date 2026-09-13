//! Dense full-space response eigensolver (`pbc/tdscf` shared).
//!
//! Upstream solves the response eigenproblem iteratively
//! (`lib/linalg_helper.py::davidson1`). On the Phase-19 gate fixtures the
//! response dimension is tiny (`nocc·nvir ≤ O(10²)`), where full
//! diagonalization IS the converged Davidson limit — the subspace already
//! spans the whole space, so no iteration can change the number. This module
//! therefore diagonalizes the TDA matrix densely through
//! [`pyscf_algebra::eigh_gen`] (identity metric) and truncates to `nroots`.
//! The blocked-Davidson iteration is the scaling path for production cells,
//! not a gate requirement.
//!
//! Full TDHF (`tda = false`, the non-symmetric `[A B; -B -A]` problem) is
//! refused with [`PbcTdscfError::NotYetImplemented`]: silently returning TDA
//! roots for a TDHF request is exactly the plausible-wrong-number shape this
//! phase guards against.

use crate::error::PbcTdscfError;
use crate::types::{TdaConfig, TdaResult};
use pyscf_algebra::{eigh_gen, oracle_sum};

/// Solve the TDA eigenproblem `A·X = X·diag(w)` densely and return the lowest
/// `nroots` roots.
///
/// `a` is the row-major symmetric TDA matrix of dimension `dim × dim`.
/// Roots are ascending by construction (`eigh_gen` returns nondecreasing
/// eigenvalues); `oscillator` entries are zero (transition dipoles need the
/// MO-basis dipole integrals the method driver supplies via
/// [`tda_with_dipoles`]).
pub fn tda_dense(a: &[f64], dim: usize, cfg: &TdaConfig) -> Result<TdaResult, PbcTdscfError> {
    tda_with_dipoles(a, dim, cfg, None)
}

/// [`tda_dense`] plus length-gauge oscillator strengths from transition
/// dipoles.
///
/// `dipoles` holds the flattened `(x, y, z)` transition-dipole vector per
/// response basis function, row-major `3 × dim`. The strength of root `r` with
/// eigenvector `x` is `f_r = (2/3)·w_r·|d·x|²`, contracted through
/// [`oracle_sum`] (ordered reduction, D-PBC-17 thread-count invariance).
pub fn tda_with_dipoles(
    a: &[f64],
    dim: usize,
    cfg: &TdaConfig,
    dipoles: Option<&[f64]>,
) -> Result<TdaResult, PbcTdscfError> {
    if !cfg.tda {
        return Err(PbcTdscfError::NotYetImplemented { module: "full-TDHF non-symmetric eigensolver" });
    }
    if a.len() != dim * dim {
        return Err(PbcTdscfError::ShapeMismatch {
            expected: dim * dim,
            got: a.len(),
        });
    }
    if dim == 0 {
        return Err(PbcTdscfError::ShapeMismatch { expected: 1, got: 0 });
    }
    if cfg.nroots > dim {
        return Err(PbcTdscfError::TooManyRoots { nroots: cfg.nroots, dim });
    }
    if let Some(d) = dipoles {
        if d.len() != 3 * dim {
            return Err(PbcTdscfError::ShapeMismatch {
                expected: 3 * dim,
                got: d.len(),
            });
        }
    }

    // Symmetrize defensively: (A + Aᵀ)/2 through ordered means. Upstream builds
    // A symmetric by construction; an asymmetric input here is a caller bug,
    // and halving the antisymmetric part can only hide it — so assert instead.
    let mut asym = 0.0f64;
    for i in 0..dim {
        for j in 0..dim {
            asym = asym.max((a[i * dim + j] - a[j * dim + i]).abs());
        }
    }
    if asym > 1e-10 {
        return Err(PbcTdscfError::ShapeMismatch {
            expected: 0,
            got: (asym * 1e12) as usize,
        });
    }

    let ident: Vec<f64> = {
        let mut s = vec![0.0f64; dim * dim];
        for i in 0..dim {
            s[i * dim + i] = 1.0;
        }
        s
    };
    let (energies_all, evecs_f) = eigh_gen(a, &ident, dim)?;
    let energies: Vec<f64> = energies_all[..cfg.nroots].to_vec();

    // evecs are F-order (column-major): column r at evecs_f[r*dim .. (r+1)*dim].
    let mut oscillator = vec![0.0f64; cfg.nroots];
    if let Some(d) = dipoles {
        for r in 0..cfg.nroots {
            let col = &evecs_f[r * dim..(r + 1) * dim];
            // |d·x|² = Σ_c (Σ_p d[c,p]·x[p])², each inner sum ordered.
            let mut norm2 = 0.0f64;
            for c in 0..3 {
                let mut comp_terms = Vec::with_capacity(dim);
                for p in 0..dim {
                    comp_terms.push(d[c * dim + p] * col[p]);
                }
                let comp = oracle_sum(&comp_terms);
                norm2 += comp * comp;
            }
            oscillator[r] = (2.0 / 3.0) * energies[r] * norm2;
        }
    }

    Ok(TdaResult {
        energies,
        oscillator,
        kshift: cfg.kshift,
        converged: true,
    })
}
