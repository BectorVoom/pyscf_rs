//! Closed-shell `update_amps` — one CCSD amplitude-equation update
//! (port of `pyscf/cc/ccsd.py:104` `update_amps` + `:362-490`
//! `_add_vvvv`/`_contract_vvvv_t2`).
//!
//! **STUB (Wave 1, 06-03).** Ships `default_update_amps`'s signature (the
//! `NoCcsdOverrides::update_amps` delegate); the body returns
//! `CcsdError::NotYetImplemented { wave: 1 }`.
//!
//! Reduction discipline for the eventual body: every einsum (incl. the heavy
//! `vvvv` `'ijcd,acdb->ijab'` contraction — the `Wabef ≈ nv⁴` arena tenant)
//! materializes products into a `Vec` then reduces via `oracle_sum`/`oracle_dot`
//! (NO `+=`, NO `gemm`). The `vvvv`/`Wabef` buffer is `pool.reserve`d ONCE
//! before the iteration loop, NOT per cycle (Pitfall 20).
#![allow(dead_code)]

use crate::eris::ChemistsEris;
use crate::error::CcsdError;
use pyscf_core::{Amplitudes, PyscfRsError};

/// One `(t1, t2) → (t1new, t2new)` CCSD amplitude update.
///
/// **STUB (Wave 1, 06-03).**
pub fn default_update_amps(
    _t1: &Amplitudes,
    _t2: &Amplitudes,
    _eris: &ChemistsEris,
) -> Result<(Amplitudes, Amplitudes), PyscfRsError> {
    Err(CcsdError::NotYetImplemented { wave: 1 }.into())
}
