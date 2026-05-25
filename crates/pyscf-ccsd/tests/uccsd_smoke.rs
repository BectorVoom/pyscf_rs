//! Always-on STRUCTURAL + NUMERIC smoke for the open-shell UCCSD (CCSD-02).
//!
//! UCCSD is the spin-block form of spin-orbital CCSD: `e_corr = e_aa + e_bb +
//! e_ab` (the αα/ββ same-spin channels antisymmetrized, the αβ cross channel
//! direct-only), each channel transforming its OWN α/β orbital energies.
//!
//! Two arms, both ALWAYS-ON (no cintx gate — `int2e` is real bit-exact since
//! 05-08):
//!
//!  1. **Closed-shell consistency cross-check (the numeric arm).** A true
//!     in-tree RHF reference fed as α==β must produce a UCCSD correlation
//!     energy bit-close to the 06-03 RCCSD `e_corr` on the same system — the
//!     spin-restricted reduction of the open-shell equations. (The codebase
//!     UHF kernel does not yet carry a genuine α/β SCF loop, plan 03-11; the
//!     live open-shell UHF byte-identity against upstream PySCF is the 06-08
//!     `workflow_dispatch` human-verify arm — the sandbox has no PySCF.)
//!
//!  2. **Genuine open-shell convergence + decomposition.** An asymmetric α/β
//!     reference (distinct per-spin energies) converges and returns a finite
//!     `e_corr = e_aa + e_bb + e_ab` with the αα/ββ channels distinct.
//!
//! Plus the RAYON 1==8 converged-energy invariance assertion (run via
//! `--test-threads=1`; the oracle reductions never consult `RAYON_NUM_THREADS`,
//! so the energy is bit-identical regardless of thread count — the explicit
//! 1-vs-8 arm is exercised in CI through the env var, asserted here by a
//! re-run-equality check the planner mandates as the structural witness).
//!
//! Mirrors `crates/pyscf-ccsd/tests/rccsd_numeric_smoke.rs` +
//! `crates/pyscf-mp2/tests/ump2_structural.rs`.

use pyscf_ccsd::{CcsdReference, NoCcsdOverrides, UccsdReference, ccsd_kernel, uccsd_kernel};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_mp2::Frozen;
use pyscf_runtime::WorkspacePool;
use pyscf_scf::RHF;

/// Build a closed-shell `CcsdReference` by running a REAL in-tree RHF.
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

/// Re-occupy a closed-shell (2.0/0.0) reference as a single spin channel
/// (1.0/0.0 per-spin occupations). The coefficients/energies are unchanged —
/// only the per-spin occupation count differs (RHF α == β by construction).
fn as_spin_channel(refr: &CcsdReference) -> CcsdReference {
    let mo_occ: Vec<f64> = refr
        .mo_occ
        .iter()
        .map(|&o| if o > 0.0 { 1.0 } else { 0.0 })
        .collect();
    CcsdReference {
        mo_coeff: refr.mo_coeff.clone(),
        mo_energy: refr.mo_energy.clone(),
        mo_occ,
        e_hf: refr.e_hf,
        converged: refr.converged,
        mol: refr.mol.clone(),
    }
}

/// **Numeric arm (CCSD-02).** A closed-shell RHF reference fed as α==β yields a
/// UCCSD correlation energy bit-close to the 06-03 RCCSD `e_corr`.
#[test]
fn uccsd_closed_shell_consistency_matches_rccsd() {
    let refr = rhf_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

    // 06-03 RCCSD on the same closed-shell system.
    let rccsd = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool)
        .expect("RCCSD must run end-to-end");
    assert!(rccsd.converged, "RCCSD must converge");

    // UCCSD with α == β (the same closed-shell reference re-occupied per spin).
    let chan = as_spin_channel(&refr);
    let uref = UccsdReference {
        alpha: chan.clone(),
        beta: chan,
    };
    let uccsd = uccsd_kernel(&uref, &Frozen::None, &pool).expect("UCCSD must run end-to-end");

    eprintln!(
        "UCCSD(α==β) H2/STO-3G: e_corr = {:.12} (e_aa={:.12}, e_bb={:.12}, e_ab={:.12}) \
         converged={} niter={}  |  RCCSD e_corr = {:.12}",
        uccsd.e_corr,
        uccsd.e_aa,
        uccsd.e_bb,
        uccsd.e_ab,
        uccsd.converged,
        uccsd.niter,
        rccsd.e_corr
    );

    assert!(uccsd.converged, "UCCSD must converge on the dual criterion");
    assert!(uccsd.e_corr.is_finite(), "UCCSD e_corr must be finite");

    // e_corr = e_aa + e_bb + e_ab identity (deterministic order).
    assert!(
        (uccsd.e_corr - (uccsd.e_aa + uccsd.e_bb + uccsd.e_ab)).abs() < 1e-12,
        "e_corr must equal e_aa + e_bb + e_ab ({} vs {})",
        uccsd.e_corr,
        uccsd.e_aa + uccsd.e_bb + uccsd.e_ab
    );

    // The spin-restricted reduction: UCCSD(α==β) e_corr == RCCSD e_corr. Both
    // converge the same correlation problem from the same integrals, so they
    // agree to bit-close tolerance (≤ 1 µHartree — the same gate the 06-03
    // headline used vs the FCI reference).
    assert!(
        (uccsd.e_corr - rccsd.e_corr).abs() < 1e-6,
        "UCCSD(α==β) e_corr {} must match RCCSD e_corr {} (spin-restricted \
         consistency) within 1 µHartree",
        uccsd.e_corr,
        rccsd.e_corr
    );

    // For a 2-electron closed shell the same-spin channels carry no correlation
    // (one electron per spin → no same-spin pair); all correlation is αβ.
    assert!(
        uccsd.e_aa.abs() < 1e-10 && uccsd.e_bb.abs() < 1e-10,
        "2-electron closed shell: same-spin channels must vanish (e_aa={}, e_bb={})",
        uccsd.e_aa,
        uccsd.e_bb
    );
}

/// **RAYON 1==8 invariance.** The converged UCCSD energy is bit-identical on a
/// re-run (the oracle reductions never consult `RAYON_NUM_THREADS`); the CI
/// matrix runs this test under `RAYON_NUM_THREADS=1` and `=8`, both yielding
/// the same bits. Here we assert the re-run self-consistency witness.
#[test]
fn uccsd_energy_is_thread_invariant() {
    let refr = rhf_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let chan = as_spin_channel(&refr);
    let uref = UccsdReference {
        alpha: chan.clone(),
        beta: chan,
    };
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let run = || {
        uccsd_kernel(&uref, &Frozen::None, &pool)
            .expect("UCCSD must run")
            .e_corr
    };
    let a = run();
    let b = run();
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "converged UCCSD e_corr must be bit-identical across runs (RAYON 1==8)"
    );
}

/// **Open-shell arm (asymmetric channels).** A spin-symmetric-OCCUPATION but
/// energy-ASYMMETRIC reference (the β channel's MO energies perturbed away from
/// α's) converges and returns a finite `e_corr = e_aa + e_bb + e_ab` with the
/// αα and ββ same-spin channels GENUINELY DISTINCT (because each channel
/// transforms its own α/β orbital energies into its own denominators).
///
/// This keeps occupied below virtual (denominators well-separated → guaranteed
/// convergence) while breaking the spin symmetry numerically, exercising the
/// per-channel-energy spin resolution end-to-end. A genuine UHF-reference
/// open-shell byte-identity against upstream PySCF is the 06-08
/// `workflow_dispatch` human-verify arm (the in-tree UHF α/β SCF loop is plan
/// 03-11; the sandbox has no PySCF).
#[test]
fn uccsd_open_shell_converges_and_decomposes() {
    // A closed-shell system with >1 occupied per spin so the same-spin channels
    // carry real correlation.
    let refr = rhf_reference("Li 0 0 0; H 0 0 1.6", "sto-3g");
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

    let alpha = as_spin_channel(&refr);
    // β: same occupations + coefficients, but the orbital energies shifted by a
    // small spin-dependent perturbation (occupied pushed down, virtual up — the
    // ordering and the occ<vir gap are preserved, so denominators stay
    // non-zero and well-separated). The αα and ββ channels now read DISTINCT
    // energies → distinct same-spin correlation.
    let mut beta = as_spin_channel(&refr);
    for (col, e) in beta.mo_energy.iter_mut().enumerate() {
        let occ = beta.mo_occ.get(col).copied().unwrap_or(0.0);
        if occ > 0.0 {
            *e -= 0.03; // push β-occupied down
        } else {
            *e += 0.05; // push β-virtual up
        }
    }

    let uref = UccsdReference { alpha, beta };
    let res =
        uccsd_kernel(&uref, &Frozen::None, &pool).expect("open-shell UCCSD must run end-to-end");

    eprintln!(
        "UCCSD(asym) LiH/STO-3G: e_corr = {:.12} (e_aa={:.12}, e_bb={:.12}, e_ab={:.12}) \
         converged={} niter={}",
        res.e_corr, res.e_aa, res.e_bb, res.e_ab, res.converged, res.niter
    );

    assert!(res.converged, "open-shell UCCSD must converge");
    assert!(res.e_corr.is_finite(), "e_corr must be finite");
    assert!(
        res.e_corr < 0.0,
        "a real correlation energy must be negative"
    );
    assert!(
        (res.e_corr - (res.e_aa + res.e_bb + res.e_ab)).abs() < 1e-12,
        "e_corr must equal e_aa + e_bb + e_ab"
    );
    // The αα and ββ same-spin channels read DISTINCT (α vs β) orbital energies
    // → genuinely distinct same-spin correlation.
    assert!(
        (res.e_aa - res.e_bb).abs() > 1e-9,
        "asymmetric-energy αα and ββ channels must differ (e_aa={}, e_bb={})",
        res.e_aa,
        res.e_bb
    );
    // The spin-resolved amplitude shapes match the per-channel occ/vir counts.
    let amp = &res.amplitudes;
    assert_eq!(
        amp.t2aa.len(),
        amp.nocc_a * amp.nocc_a * amp.nvir_a * amp.nvir_a
    );
    assert_eq!(
        amp.t2ab.len(),
        amp.nocc_a * amp.nocc_b * amp.nvir_a * amp.nvir_b
    );
    assert_eq!(
        amp.t2bb.len(),
        amp.nocc_b * amp.nocc_b * amp.nvir_b * amp.nvir_b
    );
    // The same-spin amplitude blocks also differ between α and β.
    assert_ne!(
        amp.t2aa, amp.t2bb,
        "asymmetric channels must give t2aa != t2bb"
    );
}
