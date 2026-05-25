//! AO-direct CCSD integration test (CCSD-07, `mycc.direct=True`).
//!
//! The always-on CCSD-07 arm: run the in-core [`ccsd_kernel`] and the AO-direct
//! [`ccsd_kernel_direct`] on the SAME small-system RHF reference and prove
//!   (1) AO-direct converges to the SAME `e_corr` as in-core (≤ 1e-9) — the
//!       equivalence proof (AO-direct is a different contraction ORDER of the
//!       same math: `einsum('ijcd,acbd->ijab', tau, vvvv)` on-the-fly vs the
//!       in-core full-`nv^4` `cc_Wvvvv` contraction), and
//!   (2) the AO-direct path's pre-flight reservation is LOWER than the in-core
//!       path's — a `nv^3` MO `vvvv`-step peak budget that the in-core run
//!       (which reserves the full `nv^4`) HARD-refuses but the AO-direct run
//!       accepts (the memory-frugality proof, D-01 / T-06-08-OOM).
//!
//! Live upstream-PySCF `direct=True` byte-identity is the 06-08-closeout
//! `workflow_dispatch` arm (the sandbox has no PySCF). Mirrors the
//! `rccsd_numeric_smoke.rs` real-RHF-reference pattern.

use pyscf_ccsd::{CcsdReference, NoCcsdOverrides, ccsd_kernel, ccsd_kernel_direct};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_mp2::Frozen;
use pyscf_runtime::WorkspacePool;
use pyscf_scf::RHF;

/// Build a CcsdReference by running a REAL in-tree RHF on `atom`/`basis`.
fn rhf_reference(atom: &str, basis: &str) -> CcsdReference {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(atom.into()),
        basis: BasisInput::Name(basis.into()),
        ..Default::default()
    })
    .expect("build mol");

    let mut mf = RHF::new(mol.clone());
    let scf = mf.kernel().expect("RHF kernel must converge");
    assert!(scf.converged, "RHF reference must converge");

    CcsdReference {
        mo_coeff: scf.mo_coeff,
        mo_energy: scf.mo_energy,
        mo_occ: scf.mo_occ,
        e_hf: scf.e_tot.0,
        converged: scf.converged,
        mol,
    }
}

/// CCSD-07 equivalence: AO-direct `e_corr` == in-core `e_corr` on LiH/STO-3G
/// (nocc=2, nvir=4 — `nv^4 = 256`, big enough that the AO-direct `nv^3` peak is
/// strictly smaller than the in-core `nv^4`). Both run with full DIIS.
#[test]
fn aodirect_matches_incore_e_corr_lih_sto3g() {
    let refr = rhf_reference("Li 0 0 0; H 0 0 1.6", "sto-3g");

    // Generous budget so BOTH paths run to convergence.
    let pool_in = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let pool_dir = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

    let incore = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool_in)
        .expect("in-core RCCSD must run end-to-end");
    let direct = ccsd_kernel_direct(&refr, &Frozen::None, &pool_dir)
        .expect("AO-direct RCCSD must run end-to-end");

    eprintln!(
        "LiH/STO-3G: in-core e_corr = {:.12} (conv={}, niter={})  \
         AO-direct e_corr = {:.12} (conv={}, niter={})",
        incore.e_corr,
        incore.converged,
        incore.niter,
        direct.e_corr,
        direct.converged,
        direct.niter
    );

    assert!(incore.converged, "in-core RCCSD must converge");
    assert!(direct.converged, "AO-direct RCCSD must converge");
    assert!(incore.e_corr.is_finite() && direct.e_corr.is_finite());

    // The equivalence proof: same minimum via a different contraction order.
    assert!(
        (incore.e_corr - direct.e_corr).abs() < 1e-9,
        "AO-direct e_corr {} differs from in-core e_corr {} by > 1e-9 \
         (the same math, different contraction order — must agree)",
        direct.e_corr,
        incore.e_corr
    );
}

/// CCSD-07 memory-frugality: a pool budget sized BETWEEN the AO-direct `nv^3`
/// peak and the in-core `nv^4` peak. The in-core run HARD-refuses (its pre-flight
/// `try_reserve(nv^4·8)` exceeds the budget → `MemoryLimitExceeded`, D-01), but
/// the AO-direct run (whose pre-flight is only `nv^3·8`) accepts and converges —
/// proving the AO-direct path's reservation is strictly lower (it never holds
/// the full `nv^4` MO `vvvv`).
#[test]
fn aodirect_preflight_reservation_is_lower_than_incore() {
    let refr = rhf_reference("Li 0 0 0; H 0 0 1.6", "sto-3g");

    // LiH/STO-3G: nocc=2, nvir=4. nv^4·8 = 2048 bytes; nv^3·8 = 512 bytes.
    // Pick a budget in between: in-core refuses (needs 2048), direct accepts
    // (needs 512). Both buffers are the only pool reservations the kernels make.
    let nvir = 4usize;
    let in_core_bytes = nvir.pow(4) * 8; // 2048
    let direct_bytes = nvir.pow(3) * 8; // 512
    assert!(
        direct_bytes < in_core_bytes,
        "the AO-direct nv^3 peak must be strictly below the in-core nv^4 peak"
    );
    let budget = (in_core_bytes + direct_bytes) / 2; // 1280: direct ok, in-core refused

    let pool_in = WorkspacePool::new(budget);
    let pool_dir = WorkspacePool::new(budget);

    // In-core HARD-refuses the over-budget nv^4 reservation (D-01, no downgrade).
    let incore = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool_in);
    assert!(
        incore.is_err(),
        "in-core RCCSD must HARD-refuse the over-budget nv^4 reservation at budget={budget}"
    );

    // AO-direct accepts the same budget (its peak is only nv^3) and converges to
    // the real correlation energy.
    let direct = ccsd_kernel_direct(&refr, &Frozen::None, &pool_dir)
        .expect("AO-direct RCCSD must accept the lower-budget pool and converge");
    assert!(
        direct.converged && direct.e_corr.is_finite() && direct.e_corr < 0.0,
        "AO-direct must converge to a real (negative) closed-shell e_corr at the lower budget"
    );
}
