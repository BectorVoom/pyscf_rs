//! MP2-08: the five helper functions CCSD imports verbatim from
//! `pyscf.mp.mp2` (`cc/ccsd.py:35`):
//!
//! ```python
//! from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
//! ```
//!
//! Upstream `pyscf/mp/mp2.py` defines these on the `mp` instance
//! (`get_nocc(mp)` etc.). In Rust we expose them as free functions over the
//! underlying data (the MO occupation vector + frozen spec) so both MP2 and
//! CCSD can call them without a shared instance type. The Python `_mo_without_core`
//! maps to the Rust `mo_without_core` (Rust has no leading-underscore privacy
//! convention; the doc records the mapping).
//!
//! This plan (05-01) ships the SIGNATURES so the MP2-08 contract test compiles
//! and references the exact CCSD import symbol set. Bodies land in plan 05-03.

use crate::error::Mp2Error;

/// Number of occupied orbitals (counting doubly-occupied as 1 each), minus the
/// frozen-core count. Upstream: `pyscf/mp/mp2.py:get_nocc` (line 351).
///
/// Body lands in plan 05-03.
pub fn get_nocc(mo_occ: &[f64], n_frozen: usize) -> Result<usize, Mp2Error> {
    let _ = (mo_occ, n_frozen);
    Err(Mp2Error::NotYetImplemented { plan: 3 })
}

/// Number of active MOs (total MOs minus frozen). Upstream:
/// `pyscf/mp/mp2.py:get_nmo` (line 371).
///
/// Body lands in plan 05-03.
pub fn get_nmo(mo_occ: &[f64], n_frozen: usize) -> Result<usize, Mp2Error> {
    let _ = (mo_occ, n_frozen);
    Err(Mp2Error::NotYetImplemented { plan: 3 })
}

/// Boolean mask over MOs marking which are active (true) vs frozen (false).
/// Upstream: `pyscf/mp/mp2.py:get_frozen_mask` (line 383).
///
/// Body lands in plan 05-03.
pub fn get_frozen_mask(mo_occ: &[f64], n_frozen: usize) -> Result<Vec<bool>, Mp2Error> {
    let _ = (mo_occ, n_frozen);
    Err(Mp2Error::NotYetImplemented { plan: 3 })
}

/// Hartree–Fock reference energy used as the MP2 baseline. Upstream:
/// `pyscf/mp/mp2.py:get_e_hf` (line 402).
///
/// Body lands in plan 05-03.
pub fn get_e_hf(mo_energy: &[f64], mo_occ: &[f64]) -> Result<f64, Mp2Error> {
    let _ = (mo_energy, mo_occ);
    Err(Mp2Error::NotYetImplemented { plan: 3 })
}

/// MO coefficient block with the frozen-core columns removed. Maps to the
/// Python `_mo_without_core(mp, mo)`. Upstream: `pyscf/mp/mp2.py:_mo_without_core`
/// (line 732).
///
/// Body lands in plan 05-03.
#[doc(alias = "_mo_without_core")]
pub fn mo_without_core(mo_coeff: &[f64], mask: &[bool]) -> Result<Vec<f64>, Mp2Error> {
    let _ = (mo_coeff, mask);
    Err(Mp2Error::NotYetImplemented { plan: 3 })
}
