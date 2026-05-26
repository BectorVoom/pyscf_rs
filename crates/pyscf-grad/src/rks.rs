//! RKS analytical gradient (GRAD-03). Body lands in Wave 5 (07-05+).
//!
//! Structural analog: `crate::rhf` + the Phase-4 `pyscf-dft` numint /
//! `pyscf-grids` Becke weights. Upstream port ref: `pyscf/grad/rks.py`. Adds
//! the XC-potential derivative + the optional `grid_response=True` Becke-
//! weight-derivative term; `grid_response` defaults OFF (upstream default) and
//! is NOT a CPHF consumer (D-04 — it is a grid-weight term, not a response
//! solve). libxc stays feature-gated OFF.
use crate::error::GradError;
use pyscf_core::PyscfRsError;

/// RKS electronic gradient. `NotYetImplemented { wave: 5 }` until 07-05.
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(GradError::NotYetImplemented { wave: 5 }.into())
}
