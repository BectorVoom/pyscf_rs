//! UHF analytical gradient (GRAD-02). Body lands in Wave 4 (07-04+).
//!
//! Structural analog: `crates/pyscf-mp2/src/ump2.rs` (the spin-resolved α/β
//! shape, file-for-file from its restricted sibling) + `crate::rhf` once
//! written. Upstream port ref: `pyscf/grad/uhf.py`.
use crate::error::GradError;
use pyscf_core::PyscfRsError;

/// UHF electronic gradient. `NotYetImplemented { wave: 4 }` until 07-04.
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(GradError::NotYetImplemented { wave: 4 }.into())
}
