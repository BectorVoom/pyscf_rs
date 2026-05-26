//! RHF analytical gradient (GRAD-01). Body lands in Wave 3 (07-03).
//!
//! Structural analog: `crates/pyscf-mp2/src/mp2.rs` (the `default_*` free-fn +
//! `Result<_, Error>` kernel shape). Upstream port ref: `pyscf/grad/rhf.py:38-189`
//! (the Hellmann-Feynman + 2e-Pulay + overlap-Pulay assembly). The
//! `grad_elec`/`make_rdm1e` bodies + the `int1e_ipovlp`/`int2e_ip1` wiring wait
//! on the Phase-7 cintx gradient-integral workstream (07-RESEARCH D-02); the
//! always-on FD gate ([`crate::verify_fd`]) validates them once they land.
use crate::error::GradError;
use pyscf_core::PyscfRsError;

/// RHF electronic gradient. `NotYetImplemented { wave: 3 }` until 07-03.
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(GradError::NotYetImplemented { wave: 3 }.into())
}
