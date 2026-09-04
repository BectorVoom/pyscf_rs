//! Plan 03-14 — `atom` + `huckel` init guesses (SCF-05, the last 2 of 5 modes).
//!
//! Task 2 (this file, created here): `init_guess_by_atom` produces a symmetric
//! row-major Density with `Tr(D·S) ≈ nelec` on H2/STO-3G (the atomic dm is built
//! from normalized atomic orbitals, so the trace is close to nelec — the minao
//! non-normalization caveat does NOT apply).
//!
//! Task 3 extends this with the `huckel` Tr(D·S) sanity check AND the closing
//! gate: RHF seeded with `init_guess='atom'` and `init_guess='huckel'` each
//! converge to the SAME total energy as the `1e` guess on H2/STO-3G
//! (mode-independence at convergence — a guess that produces a plausible-but-
//! wrong matrix fails this).

use pyscf_core::{Density, Unit};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_scf::{
    InitGuessMode, KernelConfig, NoOverrides, default_get_init_guess, default_get_ovlp, kernel,
};

fn h2_sto3g() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build H2/STO-3G")
}

/// `Tr(D·S) = Σ_{μν} D[μ,ν]·S[ν,μ]` (both row-major nao×nao).
fn trace_ds(dm: &Density, s: &Density) -> f64 {
    let nao = dm.nao;
    let mut diag = Vec::with_capacity(nao);
    for mu in 0..nao {
        let row: Vec<f64> = (0..nao)
            .map(|nu| dm.data[mu * nao + nu] * s.data[nu * nao + mu])
            .collect();
        diag.push(pyscf_algebra::oracle_sum(&row));
    }
    pyscf_algebra::oracle_sum(&diag)
}

fn assert_symmetric(dm: &Density) {
    let nao = dm.nao;
    for mu in 0..nao {
        for nu in 0..nao {
            let a = dm.data[mu * nao + nu];
            let b = dm.data[nu * nao + mu];
            assert!(
                (a - b).abs() < 1e-12,
                "density not symmetric at ({mu},{nu}): {a} vs {b}"
            );
        }
    }
}

#[test]
fn atom_guess_h2_trace_ds_is_nelec() {
    let mol = h2_sto3g();
    let dm = default_get_init_guess(&mol, &InitGuessMode::Atom).expect("atom init guess");
    assert_eq!(dm.nao, mol.nao_nr, "atom guess returns nao×nao Density");
    assert_eq!(dm.nao, 2, "H2/STO-3G has 2 AOs");
    assert_symmetric(&dm);

    let s = default_get_ovlp(&mol).expect("overlap");
    let tr = trace_ds(&dm, &s);
    println!("atom guess H2/STO-3G Tr(D·S) = {tr}");
    // The atomic dm is built from normalized spherically-averaged atomic RHF
    // orbitals: Tr(D·S) should be ≈ nelec = 2 (tolerance documented: the only
    // departure is the molecular vs atomic overlap of the two H atomic dm
    // blocks, which is small at the equilibrium bond length).
    assert!(
        (tr - 2.0).abs() < 0.15,
        "atom guess Tr(D·S) should be ≈ 2.0 (H2 electron count), got {tr}"
    );
}

#[test]
fn huckel_guess_h2_trace_ds_is_nelec() {
    let mol = h2_sto3g();
    let dm = default_get_init_guess(&mol, &InitGuessMode::Huckel).expect("huckel init guess");
    assert_eq!(dm.nao, mol.nao_nr, "huckel guess returns nao×nao Density");
    assert_eq!(dm.nao, 2, "H2/STO-3G has 2 AOs");
    assert_symmetric(&dm);

    let s = default_get_ovlp(&mol).expect("overlap");
    let tr = trace_ds(&dm, &s);
    println!("huckel guess H2/STO-3G Tr(D·S) = {tr}");
    // The Hückel guess solves eig(orb_H, orb_S) in the occupied-orbital basis
    // then Aufbau-fills; the resulting density holds exactly nelec electrons.
    assert!(
        (tr - 2.0).abs() < 1e-9,
        "huckel guess Tr(D·S) should be ≈ 2.0 (H2 electron count), got {tr}"
    );
}

/// THE closing gate (T-03-14-CORRECT): RHF seeded with `atom` and `huckel`
/// each converge to the SAME total energy as the `1e` guess on H2/STO-3G
/// (mode-independence at convergence — proves the guesses seed a CORRECT SCF,
/// not just produce a plausible-looking matrix).
#[test]
fn atom_and_huckel_seed_rhf_converging_to_1e_energy() {
    let mol = h2_sto3g();

    let run = |mode: InitGuessMode| {
        let cfg = KernelConfig {
            init_guess: mode,
            ..Default::default()
        };
        kernel(&mol, &NoOverrides, cfg)
    };

    let one_e = run(InitGuessMode::OneElectron).expect("1e converges");
    let atom = run(InitGuessMode::Atom).expect("atom converges");
    let huckel = run(InitGuessMode::Huckel).expect("huckel converges");

    assert!(one_e.converged, "1e-seeded RHF must converge");
    assert!(atom.converged, "atom-seeded RHF must converge");
    assert!(huckel.converged, "huckel-seeded RHF must converge");

    println!(
        "H2/STO-3G e_tot: 1e={} atom={} huckel={}",
        one_e.e_tot.0, atom.e_tot.0, huckel.e_tot.0
    );

    // All three land near the known H2/STO-3G RHF energy (≈ -1.117 Hartree).
    for (name, e) in [
        ("1e", one_e.e_tot.0),
        ("atom", atom.e_tot.0),
        ("huckel", huckel.e_tot.0),
    ] {
        assert!(
            e > -2.0 && e < -1.0,
            "{name}-seeded H2/STO-3G e_tot should be ≈ -1.117, got {e}"
        );
    }

    // Mode-independence at convergence: atom & huckel match the 1e energy.
    assert!(
        (atom.e_tot.0 - one_e.e_tot.0).abs() < 1e-6,
        "atom ({}) must converge to the 1e energy ({})",
        atom.e_tot.0,
        one_e.e_tot.0
    );
    assert!(
        (huckel.e_tot.0 - one_e.e_tot.0).abs() < 1e-6,
        "huckel ({}) must converge to the 1e energy ({})",
        huckel.e_tot.0,
        one_e.e_tot.0
    );
}

// ── Cartesian-basis branch (NYI removed, quick 260602-b62) ──────────────────
//
// Running the `atom`/`huckel` guesses on a `cart=true` molecule previously
// returned `NotYetImplemented`. The cart branch now builds the guess on a
// spherical sibling and projects `D_cart = C·D_sph·Cᵀ` (C = cart2sph_coeff).
// H2O/cc-pVDZ carries a d shell on O (the first place cart≠sph: ncart_d=6 >
// nsph_d=5 → nao_cart=25 > nao_sph=24), so this genuinely exercises the d-block
// projection. The upstream-faithful correctness invariant is
// `Tr(D_cart·S_cart) = nelec`.

fn h2o_ccpvdz_cart() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467".into()),
        basis: BasisInput::Name("cc-pvdz".into()),
        unit: Unit::Ang,
        cart: true,
        ..Default::default()
    })
    .expect("build H2O/cc-pVDZ cartesian")
}

#[test]
fn atom_guess_cart_h2o_trace_ds_is_nelec() {
    let mol = h2o_ccpvdz_cart();
    assert!(mol.cart, "molecule must be cartesian for this test");
    assert!(
        mol.nao_nr >= 25,
        "cc-pVDZ cart H2O should have a d shell (nao_cart=25 > nao_sph=24), got {}",
        mol.nao_nr
    );
    let dm = default_get_init_guess(&mol, &InitGuessMode::Atom).expect("cart atom init guess");
    assert_eq!(dm.nao, mol.nao_nr, "density is in the cartesian AO basis");
    assert_symmetric(&dm);
    let s = default_get_ovlp(&mol).expect("cart overlap");
    let tr = trace_ds(&dm, &s);
    assert!(
        (tr - mol.nelectron as f64).abs() < 1e-9,
        "cart atom guess Tr(D·S) = {tr} must equal nelec = {}",
        mol.nelectron
    );
}

#[test]
fn huckel_guess_cart_h2o_trace_ds_is_nelec() {
    let mol = h2o_ccpvdz_cart();
    let dm = default_get_init_guess(&mol, &InitGuessMode::Huckel).expect("cart huckel init guess");
    assert_eq!(dm.nao, mol.nao_nr);
    assert_symmetric(&dm);
    let s = default_get_ovlp(&mol).expect("cart overlap");
    let tr = trace_ds(&dm, &s);
    assert!(
        (tr - mol.nelectron as f64).abs() < 1e-9,
        "cart huckel guess Tr(D·S) = {tr} must equal nelec = {}",
        mol.nelectron
    );
}
