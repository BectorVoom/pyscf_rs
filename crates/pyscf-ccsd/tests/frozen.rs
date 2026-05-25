//! CCSD-10: the frozen-core contract — CCSD reuses the five MP2-08 helpers +
//! the `Frozen` enum VERBATIM (`cc/ccsd.py:35`), so CCSD's frozen behavior IS
//! the MP2 helper behavior (no new frozen logic — D-Discretion).
//!
//! This is a CONTRACT/validation test over already-shipped helpers (the MP2-08
//! helpers were contract-tested vs upstream in Phase 5's `ccsd_import_contract`
//! arm). It proves, end-to-end on a REAL in-tree RHF → CCSD run, that:
//!
//!   1. `Frozen::Count(n)` / `Frozen::List(..)` / `Frozen::Auto` produce the
//!      SAME active-space `nocc`/`nvir`/frozen-mask through CCSD's
//!      `default_ao2mo` as the MP2 helpers `get_nocc`/`get_nmo`/
//!      `get_frozen_mask`/`mo_without_core` compute on the same reference.
//!   2. A frozen-core CCSD `e_corr` differs from the all-electron `e_corr` in
//!      the expected direction (freezing core orbitals recovers LESS
//!      correlation → `e_corr` rises toward zero).
//!
//! NOTE on `Frozen::Auto`: through the data-only helper surface (and therefore
//! through `default_ao2mo`, which calls `pyscf_mp2::get_frozen_mask`), `Auto`
//! resolves with NO element info → freezes 0 orbitals (documented in
//! `pyscf-mp2/src/helpers.rs`). So the CCSD `Auto` active space equals the
//! `None` active space — and that identity IS the contract this test pins (the
//! MP2 numeric kernel resolves chemcore with real charges separately; CCSD's
//! helper-routed `Auto` is element-blind by the same verbatim-reuse contract).
//!
//! NOT a live-PySCF byte-identity check (the sandbox has no PySCF — that is the
//! 06-08 workflow_dispatch arm). Mirrors the MP2 frozen fixtures.

use pyscf_ccsd::{CcsdReference, NoCcsdOverrides, ccsd_kernel, default_ao2mo};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_mp2::{Frozen, get_frozen_mask, get_nmo, get_nocc, mo_without_core};
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

/// The active occupied / active virtual counts the MP2 helpers compute on a
/// reference for a given frozen spec. `nvir = nmo_active - nocc`.
fn helper_active_space(refr: &CcsdReference, frozen: &Frozen) -> (usize, usize) {
    let nocc = get_nocc(&refr.mo_occ, frozen).expect("get_nocc");
    let nmo = get_nmo(&refr.mo_occ, frozen).expect("get_nmo");
    (nocc, nmo - nocc)
}

/// CONTRACT (CCSD-10), `Frozen::Count`: CCSD `default_ao2mo`'s active space
/// equals the MP2 helper active space; freezing one orbital drops the helper
/// `nocc` by exactly one (matching `default_ao2mo`).
#[test]
fn ccsd_frozen_count_matches_mp2_helper_active_space() {
    let refr = rhf_reference("Li 0 0 0; H 0 0 1.6", "sto-3g"); // LiH: nocc=2, nmo=6

    for frozen in [Frozen::None, Frozen::Count(1)] {
        let (h_nocc, h_nvir) = helper_active_space(&refr, &frozen);
        let eris = default_ao2mo(&refr, &frozen).expect("default_ao2mo");
        assert_eq!(
            (eris.nocc, eris.nvir),
            (h_nocc, h_nvir),
            "CCSD default_ao2mo active space must equal the MP2 helper active \
             space for {frozen:?} (CCSD-10: no new frozen logic)"
        );
    }

    // Freezing exactly one inner orbital drops nocc by 1 (and nmo by 1).
    let (none_nocc, none_nvir) = helper_active_space(&refr, &Frozen::None);
    let (c1_nocc, c1_nvir) = helper_active_space(&refr, &Frozen::Count(1));
    assert_eq!(
        c1_nocc,
        none_nocc - 1,
        "Count(1) freezes one occupied orbital"
    );
    assert_eq!(
        c1_nvir, none_nvir,
        "Count(1) leaves the virtual count unchanged"
    );
}

/// CONTRACT (CCSD-10), `Frozen::List`: a list spec freezes exactly the named
/// orbitals; CCSD's active space equals the helper's, and `mo_without_core`
/// drops exactly those columns.
#[test]
fn ccsd_frozen_list_matches_mp2_helper_and_mo_without_core() {
    let refr = rhf_reference("Li 0 0 0; H 0 0 1.6", "sto-3g");

    // Freeze the inner-most occupied orbital via a list (index 0).
    let frozen = Frozen::List(vec![0]);
    let (h_nocc, h_nvir) = helper_active_space(&refr, &frozen);
    let eris = default_ao2mo(&refr, &frozen).expect("default_ao2mo");
    assert_eq!(
        (eris.nocc, eris.nvir),
        (h_nocc, h_nvir),
        "CCSD List active space must equal the MP2 helper active space"
    );

    // mo_without_core drops the frozen column: nmo_active = full_nmo - 1.
    let active_mo = mo_without_core(&refr.mo_coeff, &frozen).expect("mo_without_core");
    let mask = get_frozen_mask(&refr.mo_occ, &frozen).expect("mask");
    let n_active = mask.iter().filter(|&&a| a).count();
    assert_eq!(
        active_mo.nmo, n_active,
        "mo_without_core keeps exactly the active (unmasked) columns"
    );
    assert_eq!(
        active_mo.nmo,
        eris.nocc + eris.nvir,
        "mo_without_core's active column count equals the CCSD active MO count"
    );
    // The List([0]) mask freezes exactly orbital 0.
    assert!(!mask[0], "orbital 0 is frozen");
    assert!(
        mask[1..].iter().all(|&a| a),
        "all other orbitals stay active"
    );
}

/// CONTRACT (CCSD-10), `Frozen::Auto`: through the verbatim-reused helper
/// surface, `Auto` is element-blind (count 0), so the CCSD `Auto` active space
/// equals the `None` active space — exactly the MP2 helper behavior.
#[test]
fn ccsd_frozen_auto_matches_mp2_helper_behavior() {
    let refr = rhf_reference("Li 0 0 0; H 0 0 1.6", "sto-3g");

    let (auto_nocc, auto_nvir) = helper_active_space(&refr, &Frozen::Auto);
    let (none_nocc, none_nvir) = helper_active_space(&refr, &Frozen::None);
    assert_eq!(
        (auto_nocc, auto_nvir),
        (none_nocc, none_nvir),
        "helper-routed Frozen::Auto is element-blind (freezes 0) — same as None"
    );

    // The CCSD default_ao2mo Auto active space matches the helper (and None).
    let eris_auto = default_ao2mo(&refr, &Frozen::Auto).expect("ao2mo auto");
    assert_eq!(
        (eris_auto.nocc, eris_auto.nvir),
        (auto_nocc, auto_nvir),
        "CCSD default_ao2mo(Auto) active space == MP2 helper(Auto) active space"
    );

    // The Auto mask is all-active (no element info at the helper surface).
    let mask = get_frozen_mask(&refr.mo_occ, &Frozen::Auto).expect("auto mask");
    assert!(
        mask.iter().all(|&a| a),
        "helper Frozen::Auto mask is all-active (count 0, no chemcore at helper surface)"
    );
}

/// CONTRACT (CCSD-10), e_corr direction: a frozen-core CCSD run recovers LESS
/// correlation than the all-electron run, so its `e_corr` is greater (closer to
/// zero). Both must converge; the frozen run reuses the helper-resolved active
/// space (no new frozen logic).
#[test]
fn frozen_core_ccsd_ecorr_rises_toward_zero() {
    let refr = rhf_reference("Li 0 0 0; H 0 0 1.6", "sto-3g");
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

    let all = ccsd_kernel(&refr, &Frozen::None, &NoCcsdOverrides, &pool)
        .expect("all-electron CCSD must run");
    let frozen = ccsd_kernel(&refr, &Frozen::Count(1), &NoCcsdOverrides, &pool)
        .expect("frozen-core CCSD must run");

    assert!(all.converged, "all-electron CCSD must converge");
    assert!(frozen.converged, "frozen-core CCSD must converge");
    assert!(all.e_corr.is_finite() && frozen.e_corr.is_finite());

    // Both are negative correlation energies; freezing recovers LESS correlation.
    assert!(all.e_corr < 0.0, "all-electron e_corr must be negative");
    assert!(frozen.e_corr < 0.0, "frozen-core e_corr must be negative");
    assert!(
        frozen.e_corr > all.e_corr,
        "frozen-core e_corr ({}) must be greater (less correlation) than the \
         all-electron e_corr ({}) — freezing raises e_corr toward zero",
        frozen.e_corr,
        all.e_corr
    );
}

/// CONTRACT (CCSD-10): a frozen-core CCSD run's active space matches the helper
/// active space AND the resulting eris blocks are sized to the active space
/// (vvvv = nvir⁴ etc.) — proving the kernel routes its whole transform through
/// the helper-resolved subset, not a separate frozen path.
#[test]
fn frozen_core_ccsd_eris_sized_to_active_space() {
    let refr = rhf_reference("Li 0 0 0; H 0 0 1.6", "sto-3g");
    let frozen = Frozen::Count(1);

    let (h_nocc, h_nvir) = helper_active_space(&refr, &frozen);
    let eris = default_ao2mo(&refr, &frozen).expect("ao2mo");

    assert_eq!(eris.nocc, h_nocc);
    assert_eq!(eris.nvir, h_nvir);
    // ovov = nocc²·nvir², vvvv = nvir⁴ — sized to the ACTIVE space.
    assert_eq!(eris.ovov.len(), h_nocc * h_nvir * h_nocc * h_nvir);
    assert_eq!(eris.vvvv.len(), h_nvir * h_nvir * h_nvir * h_nvir);
    assert_eq!(eris.oooo.len(), h_nocc * h_nocc * h_nocc * h_nocc);
    // Active MO energies are nocc + nvir long.
    assert_eq!(eris.mo_energy.len(), h_nocc + h_nvir);
}
