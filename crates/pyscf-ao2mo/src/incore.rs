//! In-core AO→MO 4-index transform entry points (`general` / `full`).
//!
//! Mirrors upstream `pyscf/ao2mo/incore.py`. Body filled in plan 05-02.
//! For THIS plan these are placeholder signatures returning
//! `Err(Ao2moError::NotYetImplemented)` so `lib.rs` (and downstream
//! consumers / test scaffolds) compile against the real symbol surface.

use crate::error::Ao2moError;

/// General AO→MO transform: transform the AO-basis 2-electron integrals
/// `eri_ao` into the MO basis using up to four distinct MO coefficient
/// blocks `mo_coeffs`. Returns the transformed integrals as a flat
/// (F-order) buffer. Body lands in plan 05-02.
pub fn general(eri_ao: &[f64], mo_coeffs: &[&[f64]], nao: usize) -> Result<Vec<f64>, Ao2moError> {
    let _ = (eri_ao, mo_coeffs, nao);
    Err(Ao2moError::NotYetImplemented)
}

/// Full AO→MO transform: the symmetric special case of `general` where the
/// same MO coefficient block is used for all four indices. Body lands in
/// plan 05-02.
pub fn full(eri_ao: &[f64], mo_coeff: &[f64], nao: usize) -> Result<Vec<f64>, Ao2moError> {
    let _ = (eri_ao, mo_coeff, nao);
    Err(Ao2moError::NotYetImplemented)
}
