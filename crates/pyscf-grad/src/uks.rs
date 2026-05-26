//! UKS analytical gradient (GRAD-04). Body lands in Wave 5 (07-05+).
//!
//! Structural analog: `crate::rks` (the sibling within this crate). Upstream
//! port ref: `pyscf/grad/uks.py`. The spin-resolved KS-gradient; same grid-
//! weight-derivative seam as RKS, two spin channels.
use crate::error::GradError;
use pyscf_core::PyscfRsError;

/// UKS electronic gradient. `NotYetImplemented { wave: 5 }` until 07-05.
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(GradError::NotYetImplemented { wave: 5 }.into())
}
