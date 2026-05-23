//! In-core AO→MO 4-index transform entry points (`general` / `full`).
//!
//! Mirrors upstream `pyscf/ao2mo/incore.py` — specifically the
//! `eri_ao.size == nao**4` branch of `general` (incore.py:125-128):
//! `einsum('pqrs,pi,qj,rk,sl->ijkl', eri, C0.conj(), C1, C2.conj(), C3)`.
//!
//! pyscf-rs is real-only for v1, so the upstream `.conj()` on `C0`/`C2` is a
//! NO-OP here (real coefficients). The actual contraction is delegated to
//! [`crate::transform::quarter_transform`], the memory-bounded
//! quarter-transform sequence written as host loops through `oracle_sum`
//! (D-03; `pyscf_algebra::gemm` is `NotYetImplemented{phase:2}`).

use crate::error::Ao2moError;
use crate::transform;
use pyscf_core::MOCoefficients;

/// General AO→MO transform: transform the AO-basis 2-electron integrals
/// `eri_ao` (F-order `[nao, nao, nao, nao]`) into the MO basis using up to
/// four DISTINCT MO coefficient blocks `mo_coeffs` (one per integral index
/// `p, q, r, s`).
///
/// Each caller passes the relevant occupied/virtual column subset as an
/// [`MOCoefficients`] whose `data`/`nmo` already represent the slice: `data`
/// is column-major `[nao, nmo]` (element `C[ao, mo]` at `ao + mo*nao`), so the
/// `i`-th MO column is `data[i*nao .. (i+1)*nao]`.
///
/// Returns the transformed integrals as a flat F-order
/// `[n0, n1, n2, n3]` buffer (`ni = mo_coeffs[i].nmo`), where element
/// `(i, j, k, l)` lives at `i + j*n0 + k*n0*n1 + l*n0*n1*n2`.
///
/// Mirrors upstream `ao2mo.general` (the `eri_ao.size == nao**4` branch,
/// `pyscf/ao2mo/incore.py:125-128`). Conjugation is a no-op for the
/// real-valued case (pyscf-rs is real-only for v1).
///
/// # Errors
/// Returns [`Ao2moError::ShapeMismatch`] when `eri_ao.len() != nao^4` or any
/// `mo_coeffs[i].nao != nao` (i.e. a coefficient block does not span the AO
/// space) — never indexes out of bounds or panics (T-05-02-SHAPE).
pub fn general(
    eri_ao: &[f64],
    nao: usize,
    mo_coeffs: [&MOCoefficients; 4],
) -> Result<Vec<f64>, Ao2moError> {
    // T-05-02-SHAPE: every coefficient block must span the AO dimension.
    for c in mo_coeffs.iter() {
        if c.nao != nao {
            return Err(Ao2moError::ShapeMismatch {
                expected: nao,
                got: c.nao,
            });
        }
        // Defensive: data must hold exactly nao*nmo column-major elements.
        if c.data.len() != c.nao * c.nmo {
            return Err(Ao2moError::ShapeMismatch {
                expected: c.nao * c.nmo,
                got: c.data.len(),
            });
        }
    }

    let [c_p, c_q, c_r, c_s] = mo_coeffs;
    transform::quarter_transform(
        eri_ao, nao, &c_p.data, c_p.nmo, &c_q.data, c_q.nmo, &c_r.data, c_r.nmo, &c_s.data, c_s.nmo,
    )
}

/// Full AO→MO transform: the symmetric special case of [`general`] where the
/// SAME MO coefficient block is used for all four indices — i.e.
/// `general(eri_ao, nao, [mo_coeff; 4])`.
///
/// Mirrors upstream `ao2mo.incore.full` (the all-same-coefficient case).
/// Returns the full MO ERI tensor as a flat F-order `[nmo, nmo, nmo, nmo]`
/// buffer.
///
/// # Errors
/// Returns [`Ao2moError::ShapeMismatch`] when `mo_coeff.nao != nao` or
/// `eri_ao.len() != nao^4` (validated by [`general`]).
pub fn full(eri_ao: &[f64], nao: usize, mo_coeff: &MOCoefficients) -> Result<Vec<f64>, Ao2moError> {
    if mo_coeff.nao != nao {
        return Err(Ao2moError::ShapeMismatch {
            expected: nao,
            got: mo_coeff.nao,
        });
    }
    general(eri_ao, nao, [mo_coeff; 4])
}
