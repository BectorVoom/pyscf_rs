//! pyscf-ccsd: Coupled-cluster singles-doubles (RCCSD / UCCSD / DF-CCSD).
//!
//! Phase 6 deliverable. Ports upstream `pyscf/cc/*.py` file-for-file (sibling-
//! crate fidelity, mirroring the Phase-5 `pyscf-mp2` discipline): the module
//! split, the error/hooks/reference/kernel shapes, and the host-loop reduction
//! discipline all copy `pyscf-mp2`. This crate stays strictly pyo3-free (D-09 —
//! the PyO3 bridge lives exclusively in `pyscf-py`) and cubecl-free + hdf5-owner-
//! free (the algebra+pyo3 wall; the HDF5 spill goes through the re-exported
//! `pyscf_chkfile::hdf5` alias). All reductions go through
//! `oracle_sum`/`oracle_dot` for bit-exactness under `release-oracle`; `gemm`
//! is `NotYetImplemented{phase:2}`, so every contraction is a host loop.
//!
//! This plan (06-01) ships the 17-module skeleton + the four load-bearing
//! contract types (`CcsdError`, `ChemistsEris`, `CcsdOverrideHooks` +
//! `NoCcsdOverrides`, `CcsdReference`) every later wave consumes. Each non-
//! contract module is a stub whose body returns `CcsdError::NotYetImplemented
//! { wave }` (`wave` = the Phase-6 wave that fills it); the bodies land in
//! 06-02..06-11.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod ccsd;
pub mod diagnostics;
pub mod diis_amps;
pub mod dfccsd;
pub mod direct;
pub mod eris;
pub mod error;
pub mod hooks;
pub mod lambda;
pub mod rdm;
pub mod reference;
pub mod rintermediates;
pub mod uccsd;
pub mod uintermediates;
pub mod ulambda;
pub mod update_amps;
pub mod urdm;

pub use ccsd::{
    CONV_TOL, CONV_TOL_NORMT, CcsdResult, DIIS_SPACE, DIIS_START_CYCLE, MAX_CYCLE, ccsd_kernel,
    default_ao2mo, default_energy, init_amps,
};
pub use eris::ChemistsEris;
pub use error::CcsdError;
pub use hooks::{CcsdOverrideHooks, NoCcsdOverrides};
pub use reference::{CcsdReference, UccsdReference};
pub use rintermediates::{
    Loo, Lvv, cc_Foo, cc_Fov, cc_Fvv, cc_Woooo, cc_Wvoov, cc_Wvovo, cc_Wvvvv, cc_Wvvvv_into,
    make_tau,
};
pub use uccsd::{UccsdAmplitudes, UccsdResult, uccsd_kernel};
pub use uintermediates::SpinOrbitalEris;
pub use update_amps::{default_update_amps, default_update_amps_with_wvvvv};
