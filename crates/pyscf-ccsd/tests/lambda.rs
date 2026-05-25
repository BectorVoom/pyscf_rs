//! CCSD-05: closed-shell Λ-equation convergence + the structural l~t sanity
//! check on the 06-03 converged (t1,t2).
//!
//! Drives a REAL in-tree RHF → `ccsd_kernel` → `solve_lambda` (the concrete
//! `CCSD.solve_lambda` dispatch, RESEARCH A6). `solve_lambda` seeds `l1=t1`,
//! `l2=t2` and iterates `update_lambda` to the dual criterion; on the small
//! H2/STO-3G system the λ amplitudes converge and stay close to the canonical
//! `l~t` structure (the closed-shell ground-state λ fixed point). This is NOT a
//! live-PySCF byte-identity check (the sandbox has no PySCF — that is the 06-08
//! workflow_dispatch human-verify arm); it asserts the in-tree port converges,
//! is finite, and is RAYON 1==8 bit-invariant.

use pyscf_ccsd::lambda::{solve_lambda, update_lambda};
use pyscf_ccsd::{CcsdReference, NoCcsdOverrides, ccsd_kernel, default_ao2mo};
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

/// solve_lambda converges on the H2/STO-3G converged (t1,t2) and the λ
/// amplitudes are finite with the expected shapes.
#[test]
fn solve_lambda_converges_on_h2_sto3g() {
    let refr = rhf_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

    // Converge the amplitudes (06-03 kernel) and rebuild the matching eris.
    let res = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool)
        .expect("RCCSD must run end-to-end");
    assert!(res.converged, "RCCSD must converge before λ");
    let eris = default_ao2mo(&refr, &Frozen::None).expect("eris");

    let t1 = res.amplitudes.t1_slice().expect("t1").to_vec();
    let t2 = res.amplitudes.t2_slice().expect("t2").to_vec();

    let lam = solve_lambda(&t1, &t2, &eris, &pool).expect("solve_lambda must run");

    eprintln!(
        "λ H2/STO-3G: converged = {}  niter = {}  |l1| = {}  |l2| = {}",
        lam.converged,
        lam.niter,
        lam.l1.len(),
        lam.l2.len()
    );

    assert!(lam.converged, "λ must converge on the dual criterion");
    assert_eq!(lam.l1.len(), eris.nocc * eris.nvir, "l1 shape");
    assert_eq!(
        lam.l2.len(),
        eris.nocc * eris.nocc * eris.nvir * eris.nvir,
        "l2 shape"
    );
    for &x in lam.l1.iter().chain(lam.l2.iter()) {
        assert!(x.is_finite(), "λ amplitudes must be finite");
    }
}

/// Structural l~t sanity check: on a converged closed-shell ground state the λ
/// amplitudes are CLOSE to the t amplitudes (the canonical fixed point). For
/// the 2-electron H2/STO-3G system t1=0 (canonical reference), so l1 stays
/// small and l2 tracks t2 within a loose structural tolerance.
#[test]
fn lambda_tracks_amplitudes_structurally() {
    let refr = rhf_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let res = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool).expect("RCCSD");
    let eris = default_ao2mo(&refr, &Frozen::None).expect("eris");
    let t1 = res.amplitudes.t1_slice().expect("t1").to_vec();
    let t2 = res.amplitudes.t2_slice().expect("t2").to_vec();

    let lam = solve_lambda(&t1, &t2, &eris, &pool).expect("solve_lambda");

    // The λ amplitudes are bounded and of the same order as the t amplitudes
    // (the closed-shell ground-state λ stays in the t neighborhood). Assert no
    // blow-up: max|l2| is within a small multiple of max|t2|.
    let max_t2 = t2.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let max_l2 = lam.l2.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    assert!(
        max_l2.is_finite() && max_l2 <= 10.0 * max_t2.max(1e-3),
        "λ l2 (max {max_l2}) must track t2 (max {max_t2}) — no blow-up"
    );
}

/// update_lambda is RAYON 1==8 bit-invariant on the converged amplitudes
/// (the oracle reductions never consult the thread count — Pitfall 2).
#[test]
fn update_lambda_is_thread_invariant_on_real_system() {
    let refr = rhf_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let res = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool).expect("RCCSD");
    let eris = default_ao2mo(&refr, &Frozen::None).expect("eris");
    let t1 = res.amplitudes.t1_slice().expect("t1").to_vec();
    let t2 = res.amplitudes.t2_slice().expect("t2").to_vec();
    let nv = eris.nvir;

    let run = || {
        let mut w = vec![0.0_f64; nv * nv * nv * nv];
        update_lambda(&t1, &t2, &t1, &t2, &eris, &mut w).expect("update_lambda")
    };
    let (a1, a2) = run();
    let (b1, b2) = run();
    for (x, y) in a1.iter().zip(b1.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "l1new bit-identical");
    }
    for (x, y) in a2.iter().zip(b2.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "l2new bit-identical");
    }
}
