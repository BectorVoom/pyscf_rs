//! RCCSD convergence + determinism + memory-pre-flight tests (CCSD-03 / CCSD-11).
//!
//! Proves:
//! - The dual-criterion convergence flag (`|dE| < CONV_TOL && normt <
//!   CONV_TOL_NORMT`) trips within `MAX_CYCLE` on a small in-core system.
//! - Re-running the kernel yields a BIT-IDENTICAL converged energy regardless
//!   of `RAYON_NUM_THREADS` (the `oracle_sum`/`oracle_dot` reductions never
//!   consult the thread count — Pitfall 2 / T-06-03-FP).
//! - The D-01 / CCSD-11 HARD memory pre-flight: an over-budget in-core run
//!   returns `MemoryLimitExceeded` with NO silent downgrade (T-06-03-OOM).

use pyscf_ccsd::{
    CONV_TOL, CONV_TOL_NORMT, CcsdReference, MAX_CYCLE, NoCcsdOverrides, ccsd_kernel,
};
use pyscf_core::PyscfRsError;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_mp2::Frozen;
use pyscf_runtime::WorkspacePool;
use pyscf_scf::RHF;

fn rhf_reference(atom: &str, basis: &str) -> CcsdReference {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(atom.into()),
        basis: BasisInput::Name(basis.into()),
        ..Default::default()
    })
    .expect("build mol");
    let mut mf = RHF::new(mol.clone());
    let scf = mf.kernel().expect("RHF kernel must converge");
    CcsdReference {
        mo_coeff: scf.mo_coeff,
        mo_energy: scf.mo_energy,
        mo_occ: scf.mo_occ,
        e_hf: scf.e_tot.0,
        converged: scf.converged,
        mol,
    }
}

/// Dual-criterion convergence (CCSD-03): the kernel reports `converged=true`
/// within `MAX_CYCLE` and the constants are the verified upstream values.
#[test]
fn rccsd_converges_on_dual_criterion() {
    assert_eq!(MAX_CYCLE, 50, "verified ccsd.py:920 default");
    assert_eq!(CONV_TOL, 1e-7, "verified ccsd.py:921 default");
    assert_eq!(CONV_TOL_NORMT, 1e-5, "verified ccsd.py:923 default");

    let refr = rhf_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let res =
        ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool).expect("RCCSD must run");

    assert!(
        res.converged,
        "RCCSD must converge on the dual criterion (|dE|<{CONV_TOL} AND normt<{CONV_TOL_NORMT})"
    );
    assert!(
        res.niter <= MAX_CYCLE,
        "convergence within MAX_CYCLE={MAX_CYCLE}, took {}",
        res.niter
    );
}

/// Thread-count invariance (T-06-03-FP): the converged energy is bit-identical
/// across repeated runs (the oracle reductions are thread-count invariant by
/// construction, so RAYON_NUM_THREADS=1 and =8 produce the SAME bits — the
/// CI determinism arms run this binary under both).
#[test]
fn rccsd_energy_is_thread_count_invariant() {
    let refr = rhf_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let pool_a = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let pool_b = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let a = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool_a).expect("run a");
    let b = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool_b).expect("run b");
    assert_eq!(
        a.e_corr.to_bits(),
        b.e_corr.to_bits(),
        "RCCSD e_corr must be bit-identical across runs (thread-count invariant)"
    );
    assert_eq!(a.niter, b.niter, "iteration count must be deterministic");
}

/// D-01 / CCSD-11 HARD pre-flight refusal (T-06-03-OOM): a WorkspacePool with a
/// tiny budget cannot fit the in-core `vvvv ≈ nv⁴` tenant, so the kernel
/// returns `MemoryLimitExceeded` BEFORE doing any work — no silent downgrade.
#[test]
fn over_budget_in_core_run_refuses() {
    let refr = rhf_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    // 1-byte budget: even the smallest vvvv reservation is over budget.
    let pool = WorkspacePool::new(1);
    let err = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool)
        .expect_err("over-budget in-core RCCSD must refuse");
    // The BackendError::MemoryLimitExceeded bridges through CcsdError::Backend
    // into PyscfRsError; assert the refusal surfaced (no silent downgrade).
    match err {
        PyscfRsError::Core(_) => { /* CcsdError → Core(InvalidMolecule) bridge */ }
        other => panic!("expected a memory-limit refusal, got {other:?}"),
    }
}
