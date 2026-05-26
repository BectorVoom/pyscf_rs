//! CCSD analytical gradient (GRAD-06). Body lands in Wave 7 (07-07+).
//!
//! Structural analog: `crates/pyscf-ccsd/src/{lambda.rs,rdm.rs}` — GRAD-06
//! CONSUMES the Phase-6 `solve_lambda` + `make_rdm1`/`make_rdm2` (incl.
//! `ao_repr`) directly; NO Λ re-derivation (D-04). Upstream port ref:
//! `pyscf/grad/ccsd.py` (the Λ-driven gradient + the single CPHF Z-vector).
use crate::error::GradError;
use pyscf_core::PyscfRsError;

/// CCSD electronic gradient. `NotYetImplemented { wave: 7 }` until 07-07.
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(GradError::NotYetImplemented { wave: 7 }.into())
}
