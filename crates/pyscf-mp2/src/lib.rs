//! pyscf-mp2: Møller–Plesset 2nd-order (RMP2 / UMP2 / DF-MP2).
//!
//! Phase 5 deliverable. Ports upstream `pyscf/mp/{mp2,ump2,dfmp2,dfmp2_native}.py`
//! plus the `ao2mo` integral transform (consumed via the sibling `pyscf-ao2mo`
//! crate, D-01). This crate stays strictly pyo3-free + cubecl-free per the
//! algebra+pyo3 wall (D-07/D-08): the PyO3 bridge + subclass-override dispatch
//! lives exclusively in `pyscf-py`, and all reductions go through
//! `oracle_sum`/`oracle_dot` for bit-exactness under `release-oracle`.
//!
//! This plan (05-01) ships the module + test scaffolds so the implementation
//! waves (05-02..05-07) have concrete, compiling targets. Each module file is a
//! stub whose body lands in a later plan; `helpers.rs` already exports the five
//! MP2-08 function signatures (the exact CCSD import contract) and `error.rs`
//! carries the real `Mp2Error` enum + `From<Mp2Error> for PyscfRsError` bridge.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod dfmp2;
pub mod dfmp2_native;
pub mod error;
pub mod frozen;
pub mod helpers;
pub mod hooks;
pub mod mp2;
pub mod rdm;
pub mod ump2;

pub use dfmp2::{DFRMP2, DFUMP2, df_ao2mo, dfrmp2_kernel, dfump2_kernel};
pub use error::Mp2Error;
pub use frozen::Frozen;
pub use helpers::{get_e_hf, get_frozen_mask, get_nmo, get_nocc, mo_without_core};
pub use hooks::{ChemistsEris, Mp2OverrideHooks, NoMp2Overrides};
pub use mp2::{Mp2Reference, Mp2Result, default_ao2mo, default_energy, rmp2_kernel, scs_energy};
pub use rdm::{gamma1_intermediates, make_rdm1, make_rdm2};
pub use ump2::{UmpAmplitudes, UmpReference, UmpResult, ump2_kernel};
