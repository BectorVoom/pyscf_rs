//! ECP gradient term (GRAD-07). Body lands in Wave 8 (07-08+).
//!
//! Structural analog: the ECP term inside `crate::rhf` once written + the cintx
//! dispatch in `crates/pyscf-gto/src/ecp_engine_cintx.rs`. Upstream port ref:
//! `pyscf/grad/rhf.py:109-143` (the `get_hcore` + `hcore_deriv` ECP branches).
//! `ECPscalar_ipnuc` is READY in cintx; `ECPscalar_iprinv` is a cintx
//! workstream item (07-RESEARCH D-02).
use crate::error::GradError;
use pyscf_core::PyscfRsError;

/// ECP gradient contribution. `NotYetImplemented { wave: 8 }` until 07-08.
pub fn default_grad_ecp() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(GradError::NotYetImplemented { wave: 8 }.into())
}
