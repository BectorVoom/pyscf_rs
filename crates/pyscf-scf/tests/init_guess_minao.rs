//! minao init guess (plan 03-13) — byte-match the upstream
//! `scf.hf.init_guess_by_minao` H2 docstring density + sanity (trace, symmetry).
//!
//! Upstream docstring (`pyscf/scf/hf.py:init_guess_by_minao`):
//! ```text
//! >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')   # default basis sto-3g, Angstrom
//! >>> scf.hf.init_guess_by_minao(mol)
//! array([[ 0.94758917,  0.09227308],
//!        [ 0.09227308,  0.94758917]])
//! ```

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor};
use pyscf_scf::InitGuessMode;
use pyscf_scf::init_guess::default_get_init_guess;

#[test]
fn minao_h2_byte_matches_upstream_docstring() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.1".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .expect("build H2/STO-3G @ 1.1 Angstrom");

    let dm = default_get_init_guess(&mol, &InitGuessMode::Minao).expect("minao init guess");
    assert_eq!(dm.nao, 2);

    // Upstream reference density (row-major [2,2]).
    let expect = [0.94758917_f64, 0.09227308, 0.09227308, 0.94758917];
    for (i, (&got, &want)) in dm.data.iter().zip(expect.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "minao dm[{i}] = {got} but upstream = {want} (Δ={:.3e})",
            (got - want).abs()
        );
    }
    eprintln!("minao H2 dm = {:?}", dm.data);
}

#[test]
fn minao_dm_symmetric_and_traces_to_nelec() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.1".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .expect("H2");
    let dm = default_get_init_guess(&mol, &InitGuessMode::Minao).expect("minao");
    let nao = dm.nao;

    // Symmetric.
    for mu in 0..nao {
        for nu in 0..nao {
            assert!(
                (dm.data[mu * nao + nu] - dm.data[nu * nao + mu]).abs() < 1e-12,
                "minao dm must be symmetric"
            );
        }
    }

    // Tr(dm · S) == nelec (2 for H2). S is the working overlap (F-order, symmetric).
    let s = intor(&mol, "int1e_ovlp_sph").expect("ovlp");
    let mut tr = 0.0;
    for mu in 0..nao {
        for nu in 0..nao {
            // (dm·S) diagonal sum = Σ_{μν} dm[μ,ν] S[ν,μ]; S symmetric.
            tr += dm.data[mu * nao + nu] * s.values[nu + mu * nao];
        }
    }
    // minao is an unnormalized guess (upstream's `dm *= nelec/(dm*s).sum()` is
    // commented out), so Tr(dm·S) ≈ nelec but not exact — a loose sanity bound.
    assert!(
        (tr - mol.nelectron as f64).abs() < 0.1,
        "Tr(dm·S) = {tr} should be ≈ nelec = {} (minao is unnormalized)",
        mol.nelectron
    );
}
