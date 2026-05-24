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
use pyscf_scf::{InitGuessMode, default_get_init_guess, default_get_ovlp};

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
