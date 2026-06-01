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
pub mod dfccsd;
pub mod diagnostics;
pub mod diis_amps;
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
    ccsd_kernel_diis, ccsd_kernel_direct, ccsd_kernel_direct_diis, default_ao2mo, default_energy,
    init_amps,
};
pub use dfccsd::{DFRCCSD, DFUCCSD, block_sizing, df_ao2mo, dfrccsd_kernel};
pub use diagnostics::{get_d1_diagnostic, get_d2_diagnostic, get_t1_diagnostic};
pub use diis_amps::{AmplitudeSubspace, amplitudes_to_vector, packed_len, vector_to_amplitudes};
pub use direct::{contract_vvvv_t2_aodirect, contract_vvvv_t2_from_eris};
pub use eris::ChemistsEris;
pub use error::CcsdError;
pub use hooks::{CcsdOverrideHooks, NoCcsdOverrides};
pub use lambda::{LambdaAmplitudes, solve_lambda, update_lambda};
pub use rdm::{Gamma1, gamma1_intermediates, make_rdm1, make_rdm2};
pub use reference::{CcsdReference, UccsdReference};
pub use rintermediates::{
    Loo, Lvv, cc_Foo, cc_Fov, cc_Fvv, cc_Woooo, cc_Wvoov, cc_Wvovo, cc_Wvvvv, cc_Wvvvv_into,
    make_tau,
};
pub use uccsd::{UccsdAmplitudes, UccsdResult, uccsd_kernel};
pub use uintermediates::SpinOrbitalEris;
pub use ulambda::{ULambdaAmplitudes, solve_ulambda, update_ulambda};
pub use update_amps::{
    default_update_amps, default_update_amps_direct, default_update_amps_with_wvvvv,
};
