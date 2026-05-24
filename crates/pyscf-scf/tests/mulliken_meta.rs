//! Plan 03-15 — `mulliken_meta` (meta-Löwdin population analysis, SCF-09)
//! conservation-invariant integration test.
//!
//! `mulliken_meta` re-expresses the converged RHF density in the meta-Löwdin
//! orthogonalized AO basis built by `crate::orth::orth_ao`, then runs the
//! `mulliken_pop` AO→atom aggregation with the overlap = identity. Because
//! `orth_ao` produces a complete S-orthonormalization (`C_orthᵀ·S·C_orth ≈ I`,
//! the `_nao_sub` invariant), the meta-Löwdin transform is electron-count
//! preserving, so the populations obey the physical conservation invariants:
//!
//!   - `Σ atom_charges ≈ molecule total charge` (0 for neutral H2 / H2O);
//!   - `Σ ao_populations ≈ nelectron`.
//!
//! These are the in-tree gate for SCF-09. Upstream byte-identity of the
//! meta-Löwdin charges (vs PySCF's full NAO core/valence/Rydberg partition)
//! is a human-verify item (the sandbox has no maturin/upstream-pyscf),
//! consistent with SCF-07's Rust-satisfied + upstream-byte-identity treatment.
//!
//! RHF is converged via the `1e` (hcore) init guess (the minao default also
//! converges, but `1e` keeps the fixture independent of the ANO data caveat
//! documented in plan 03-13). Test molecules model the
//! `no_overrides_drives_kernel.rs` / `init_guess_minao.rs` style.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_scf::RHF;

/// Build an RHF on `atom` / `basis`, converge it via the 1e guess, and return
/// the populated RHF (mo_coeff set). Panics (test-only) if the SCF fails.
fn converged_rhf(atom: &str, basis: &str) -> RHF {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(atom.into()),
        basis: BasisInput::Name(basis.into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build molecule");
    let mut rhf = RHF::new(mol);
    // 1e (hcore) init guess — independent of the minao ANO-data caveat.
    rhf.init_guess = "1e".to_string();
    let result = rhf.kernel().expect("RHF kernel");
    assert!(result.converged, "RHF must converge for {atom}/{basis}");
    rhf
}

/// H2/STO-3G: neutral (Σ chg ≈ 0), Σ ao_pop ≈ 2 electrons, and by symmetry
/// the two H charges are equal.
#[test]
fn mulliken_meta_h2_conservation_and_symmetry() {
    let rhf = converged_rhf("H 0 0 0; H 0 0 1.4", "sto-3g");
    let nelec = rhf.mol.nelectron as f64;
    let natm = rhf.mol.natm;

    let res = pyscf_scf::mulliken_meta(&rhf).expect("mulliken_meta H2");

    assert_eq!(res.atom_charges.len(), natm, "one charge per atom");
    assert!(
        res.atom_charges.iter().all(|c| c.is_finite()),
        "all atom charges finite"
    );
    assert!(
        res.ao_populations.iter().all(|p| p.is_finite()),
        "all AO populations finite"
    );

    // Conservation: Σ ao_pop ≈ nelec (the electron-count invariant).
    let sum_pop = pyscf_algebra::oracle_sum(&res.ao_populations);
    assert!(
        (sum_pop - nelec).abs() < 1e-8,
        "Σ ao_pop = {sum_pop} must ≈ nelec = {nelec} (electron-count invariant)"
    );

    // Conservation: Σ chg ≈ molecule total charge (0, neutral H2).
    let sum_chg = pyscf_algebra::oracle_sum(&res.atom_charges);
    assert!(
        sum_chg.abs() < 1e-8,
        "Σ atom_charges = {sum_chg} must ≈ 0 for neutral H2"
    );

    // Symmetry: the two H charges are equal (H2 is homonuclear).
    assert!(
        (res.atom_charges[0] - res.atom_charges[1]).abs() < 1e-7,
        "symmetric H2: chg[0] = {} ≈ chg[1] = {}",
        res.atom_charges[0],
        res.atom_charges[1]
    );

    eprintln!(
        "H2 mulliken_meta: Σ ao_pop = {sum_pop} (nelec {nelec}), Σ chg = {sum_chg}, \
         per-atom chg = {:?}",
        res.atom_charges
    );
}

/// H2O/STO-3G: neutral (Σ chg ≈ 0), Σ ao_pop ≈ 10 electrons — the heavy-atom
/// case (O carries an s/p block partition the per-(atom,l) meta-Löwdin
/// orthogonalizes within).
#[test]
fn mulliken_meta_h2o_conservation() {
    let rhf = converged_rhf("O 0 0 0; H 0 0 1.8; H 0 1.76 -0.49", "sto-3g");
    let nelec = rhf.mol.nelectron as f64;
    let natm = rhf.mol.natm;

    let res = pyscf_scf::mulliken_meta(&rhf).expect("mulliken_meta H2O");

    assert_eq!(res.atom_charges.len(), natm, "one charge per atom");
    assert!(
        res.atom_charges.iter().all(|c| c.is_finite()),
        "all atom charges finite"
    );
    assert!(
        res.ao_populations.iter().all(|p| p.is_finite()),
        "all AO populations finite"
    );

    // Conservation: Σ ao_pop ≈ nelec = 10.
    let sum_pop = pyscf_algebra::oracle_sum(&res.ao_populations);
    assert!(
        (sum_pop - nelec).abs() < 1e-8,
        "Σ ao_pop = {sum_pop} must ≈ nelec = {nelec} (H2O electron-count invariant)"
    );

    // Conservation: Σ chg ≈ 0 (neutral H2O).
    let sum_chg = pyscf_algebra::oracle_sum(&res.atom_charges);
    assert!(
        sum_chg.abs() < 1e-8,
        "Σ atom_charges = {sum_chg} must ≈ 0 for neutral H2O"
    );

    eprintln!(
        "H2O mulliken_meta: Σ ao_pop = {sum_pop} (nelec {nelec}), Σ chg = {sum_chg}, \
         per-atom chg = {:?}",
        res.atom_charges
    );
}
