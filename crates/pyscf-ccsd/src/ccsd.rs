//! (R)CCSD kernel loop, MP2-seeded `init_amps`, energy, and the `default_*`
//! free functions the `NoCcsdOverrides` hook delegates to.
//!
//! **STUB (Wave 1).** This plan (06-01) ships the signatures every later wave
//! and the `hooks`/`lib` re-exports compile against; the bodies return
//! `CcsdError::NotYetImplemented { wave: 1 }`. The algorithm lands in 06-03
//! (port of `pyscf/cc/ccsd.py:44-101` `kernel`, the MP2-seed `init_amps`
//! `ccsd.py:1048-1077`, and the closed-shell `energy`).
//!
//! Reduction discipline for the eventual body: EVERY contraction materializes
//! products into a `Vec` then reduces via `pyscf_algebra::oracle_sum`/
//! `oracle_dot` (NO `+=`, NO `gemm` — `gemm` is `NotYetImplemented{phase:2}`).
//! Convergence defaults (verified from `ccsd.py:920-928`): `max_cycle=50`,
//! `conv_tol=1e-7`, `conv_tol_normt=1e-5`, `diis_space=6`, `diis_start_cycle=0`.
#![allow(dead_code)]

use crate::eris::ChemistsEris;
use crate::error::CcsdError;
use crate::reference::CcsdReference;
use pyscf_core::{Amplitudes, Energy, PyscfRsError};
use pyscf_mp2::Frozen;
use pyscf_runtime::WorkspacePool;

/// Result of the (R)CCSD kernel.
///
/// Mirrors `pyscf_mp2::Mp2Result` with the CCSD additions (converged flag,
/// iteration count, and the converged `t1`/`t2` amplitudes the λ-equations and
/// RDMs consume in Wave 3).
#[derive(Debug, Clone)]
pub struct CcsdResult {
    /// CCSD correlation energy.
    pub e_corr: f64,
    /// Whether the amplitude iteration converged.
    pub converged: bool,
    /// Number of iterations performed.
    pub niter: usize,
    /// Converged amplitudes (`t1`/`t2`).
    pub amplitudes: Amplitudes,
}

/// (R)CCSD driver — port of `pyscf/cc/ccsd.py:44-101` `kernel`.
///
/// **STUB (Wave 1, 06-03).** Generic over the override seam (mirrors the
/// `pyscf_mp2::rmp2_kernel` shape) and takes the `WorkspacePool` arena for the
/// D-01 memory pre-flight (`pool.try_reserve(estimate_bytes)?`) before the
/// `vvvv`/`Wabef` allocation.
pub fn ccsd_kernel<H: crate::hooks::CcsdOverrideHooks>(
    _refr: &CcsdReference,
    _frozen: &Frozen,
    _hooks: &H,
    _pool: &WorkspacePool,
) -> Result<CcsdResult, PyscfRsError> {
    Err(CcsdError::NotYetImplemented { wave: 1 }.into())
}

/// In-core `ao2mo` default — builds the full `ChemistsEris` block set from
/// `int2e` (the `NoCcsdOverrides::ao2mo` delegate).
///
/// **STUB (Wave 1, 06-03).** Will mirror `pyscf_mp2::default_ao2mo` but produce
/// all of `oooo`/`ovoo`/`oovv`/`ovov`/`ovvo`/`ovvv`/`vvvv` (not just `ovov`).
pub fn default_ao2mo(
    _refr: &CcsdReference,
    _frozen: &Frozen,
) -> Result<ChemistsEris, PyscfRsError> {
    Err(CcsdError::NotYetImplemented { wave: 1 }.into())
}

/// Closed-shell CCSD correlation energy from amplitudes + ERIs — the
/// `NoCcsdOverrides::energy` / `CcsdOverrideHooks::energy` default delegate.
///
/// **STUB (Wave 1, 06-03).** Will port `ccsd.py` `energy(t1, t2, eris)` with
/// every reduction routed through `oracle_sum`/`oracle_dot`.
pub fn default_energy(
    _t1: &Amplitudes,
    _t2: &Amplitudes,
    _eris: &ChemistsEris,
) -> Result<Energy, PyscfRsError> {
    Err(CcsdError::NotYetImplemented { wave: 1 }.into())
}
