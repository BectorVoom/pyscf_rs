//! Generalized eigh + SCF-13 sign canonicalization. Body filled in plan 03-11.
//!
//! Source: `pyscf/scf/hf.py:1349-1357` — `def eig(self, h, s):` returns
//! `scipy.linalg.eigh(h, s)` followed by inline sign canonicalization
//! (the inline canonicalize was extracted into pyscf-core::canonicalize_signs
//! in plan 03-01 — SCF-13 anchor, Pitfall 4 + 12).
//!
//! Routes to `pyscf_algebra::eigh_gen` (plan 03-11 slice-based wrapper
//! over faer 0.24 self-adjoint EVD with Löwdin orthogonalization). Then
//! calls `pyscf_core::canonicalize_signs` on the F-order MO coefficient
//! buffer to make eigenvector signs vendor-stable.

use pyscf_core::{Density, MOCoefficients, PyscfRsError, canonicalize_signs};

use crate::error::ScfError;

/// Default `eig(F, S) → MO` — solves `F·C = S·C·diag(ε)` and applies the
/// SCF-13 sign canonicalization. Returns `MOCoefficients` with F-order
/// (column-major) data, eigenvalues sorted nondecreasing, and zeroed
/// occupations (caller fills via `default_get_occ`).
pub fn default_eig(fock: &Density, s1e: &Density) -> Result<MOCoefficients, PyscfRsError> {
    let nao = fock.nao;
    if s1e.nao != nao {
        return Err(ScfError::Core(pyscf_core::CoreError::DimensionMismatch {
            expected: nao,
            actual: s1e.nao,
        })
        .into());
    }
    let (eigvals, mut eigvecs) =
        pyscf_algebra::eigh_gen(&fock.data, &s1e.data, nao).map_err(ScfError::Algebra)?;

    // SCF-13 anchor (Pitfall 4 + Pitfall 12) — vendor-stable eigenvector
    // signs. Idempotent; the canonicalize_post_eigh.rs integration test
    // asserts the second pass is a no-op.
    canonicalize_signs(&mut eigvecs, nao, nao);

    Ok(MOCoefficients {
        nao,
        nmo: nao,
        data: eigvecs,
        energies: eigvals,
        occupations: vec![0.0; nao],
    })
}
