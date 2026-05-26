//! MP2 analytical gradient (GRAD-05). Body lands in Wave 6 (07-06+).
//!
//! Structural analog: `crates/pyscf-mp2/src/rdm.rs` (the relaxed-density
//! assembly shape). Upstream port ref: `pyscf/grad/mp2.py:268-280` (the Z-vector
//! response). The single CPHF solver ([`crate::cphf`]) is invoked with the
//! MP2-specific `max_cycle=30` override (Pitfall 5) and the MP2 `fvind`.
use crate::error::GradError;
use pyscf_core::PyscfRsError;

/// MP2 electronic gradient. `NotYetImplemented { wave: 6 }` until 07-06.
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(GradError::NotYetImplemented { wave: 6 }.into())
}
