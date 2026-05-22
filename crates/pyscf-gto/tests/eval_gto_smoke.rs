//! GTO-07 smoke fixtures (plan 02-06 Task 2).
//!
//! Verifies the user-facing `pyscf_gto::eval_gto` wrapper:
//!   - dispatches `GTOval` / `GTOval_sph` / `GTOval_cart` to the
//!     pyscf-kernels host CPU implementation (Phase 4 DFT extends with
//!     l ≥ 1 cart2sph + the deriv1/deriv2/ip/ig variants);
//!   - returns clean `NotYetImplemented{phase:4}` errors for the deriv
//!     variants and `phase:7` for the ip/ig variants;
//!   - the alias `GTOval` routes by `mol.cart` (false → sph, true → cart);
//!   - unbuilt Mole / unknown variant produce typed errors;
//!   - the algebra-wall surface holds — `pyscf-gto::eval_gto` imports
//!     only `pyscf-kernels` + `pyscf-algebra`, never `cubecl-*` (the
//!     `! grep cubecl crates/pyscf-gto/src/eval_gto.rs` check in the
//!     plan's verify section is the static guarantee; this file
//!     exercises the runtime guarantee).
//!
//! Numerical reference: STO-3G H 1s at the nucleus is the contracted
//! radial sum after per-prim gto_norm + per-contraction
//! `_nomalize_contracted_ao`. With STO-3G α = (3.42525, 0.62391,
//! 0.16886) and unnormalised c = (0.15433, 0.53533, 0.44464), the
//! analytical psi(0) is approximately 0.6325. Matches upstream
//! `pyscf.dft.numint.eval_ao(mol, [[0,0,0]])`.

use pyscf_core::Unit;
use pyscf_gto::{eval_gto, AtomInput, BasisInput, MoleBuildArgs, M};

fn h_at_origin_sto3g() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn h_1s_at_nucleus_returns_contracted_radial_sum() {
    let mol = h_at_origin_sto3g();
    let coords = vec![[0.0_f64, 0.0, 0.0]];
    let out = eval_gto(&mol, "GTOval_sph", &coords).unwrap();
    assert_eq!(out.shape, vec![1, 1]);
    assert_eq!(out.values.len(), 1);
    // For STO-3G H at r=0: psi(0) = Σ_p c_norm[p] * exp(0) = Σ_p c_norm[p].
    // After per-prim gto_norm + per-contraction _nomalize_contracted_ao,
    // the analytical psi(0) for H 1s STO-3G is ≈ 0.6325. Allow a wider
    // sanity envelope here so the test is robust against minor
    // normalisation conventions; tighter byte-identity oracle gating is
    // 02-09's job.
    assert!(
        out.values[0] > 0.5 && out.values[0] < 0.8,
        "psi(0) = {} (expected ≈ 0.63 for H 1s STO-3G)",
        out.values[0]
    );
}

#[test]
fn h_1s_at_far_distance_decays() {
    let mol = h_at_origin_sto3g();
    let coords = vec![[0.0_f64, 0.0, 5.0]];
    let out = eval_gto(&mol, "GTOval_sph", &coords).unwrap();
    let psi_5 = out.values[0];
    // Ratio psi(5)/psi(0) is on the order of exp(-α_min * 25). For
    // STO-3G α_min ≈ 0.16886, so psi(5)/psi(0) ≈ exp(-4.22) ≈ 0.0146.
    // Assert the absolute value is well below 0.1 (a generous envelope).
    assert!(
        psi_5.abs() < 0.1,
        "psi(5 Bohr) = {} (expected ≪ 1)",
        psi_5
    );
}

#[test]
fn output_shape_matches_ngrids_times_nao() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .unwrap();
    let ngrids = 10;
    let coords: Vec<[f64; 3]> = (0..ngrids).map(|i| [0.0, 0.0, i as f64 * 0.1]).collect();
    let out = eval_gto(&mol, "GTOval_sph", &coords).unwrap();
    assert_eq!(out.shape, vec![ngrids, 2]);
    assert_eq!(out.values.len(), ngrids * 2);
}

#[test]
fn deriv1_sph_returns_four_component_result() {
    // Phase 4 plan 04-03 landed GTOval_sph_deriv1; it no longer returns
    // NotYetImplemented. Element-wise parity is gated by
    // tests/eval_gto_deriv1_oracle.rs — here we just confirm the shape
    // and that component 0 equals the plain GTOval_sph value.
    let mol = h_at_origin_sto3g();
    let coords = [[0.0, 0.0, 0.0_f64], [0.0, 0.0, 0.5]];
    let d1 = eval_gto(&mol, "GTOval_sph_deriv1", &coords).expect("deriv1 ok");
    assert_eq!(d1.shape, vec![4, 2, mol.nao_nr]);
    assert_eq!(d1.values.len(), 4 * 2 * mol.nao_nr);

    let val = eval_gto(&mol, "GTOval_sph", &coords).expect("value ok");
    for idx in 0..(2 * mol.nao_nr) {
        assert_eq!(
            d1.values[idx].to_bits(),
            val.values[idx].to_bits(),
            "deriv1 component 0 must equal GTOval_sph at idx {idx}"
        );
    }
}

#[test]
fn deriv1_cart_returns_not_yet_implemented_phase_4() {
    // Cartesian deriv1 stays deferred (needs cart ao_loc).
    let mol = h_at_origin_sto3g();
    let r = eval_gto(&mol, "GTOval_cart_deriv1", &[[0.0, 0.0, 0.0]]);
    match r {
        Err(pyscf_core::PyscfRsError::NotYetImplemented { phase: 4, what }) => {
            assert!(what.contains("deriv1"), "what = {}", what);
        }
        other => panic!("expected NotYetImplemented{{phase: 4}}, got {:?}", other),
    }
}

#[test]
fn deriv2_returns_not_yet_implemented_phase_4() {
    let mol = h_at_origin_sto3g();
    let r = eval_gto(&mol, "GTOval_sph_deriv2", &[[0.0, 0.0, 0.0]]);
    match r {
        Err(pyscf_core::PyscfRsError::NotYetImplemented { phase: 4, what }) => {
            assert!(what.contains("deriv2"), "what = {}", what);
        }
        other => panic!("expected NotYetImplemented{{phase: 4}}, got {:?}", other),
    }
}

#[test]
fn ip_returns_not_yet_implemented_phase_7() {
    let mol = h_at_origin_sto3g();
    let r = eval_gto(&mol, "GTOval_ip", &[[0.0, 0.0, 0.0]]);
    assert!(matches!(
        r,
        Err(pyscf_core::PyscfRsError::NotYetImplemented { phase: 7, .. })
    ));
}

#[test]
fn ig_returns_not_yet_implemented_phase_7() {
    let mol = h_at_origin_sto3g();
    let r = eval_gto(&mol, "GTOval_ig", &[[0.0, 0.0, 0.0]]);
    assert!(matches!(
        r,
        Err(pyscf_core::PyscfRsError::NotYetImplemented { phase: 7, .. })
    ));
}

#[test]
fn alias_gtoval_routes_to_sph_by_default() {
    // mol.cart = false default → GTOval routes to GTOval_sph.
    let mol = h_at_origin_sto3g();
    let out_alias = eval_gto(&mol, "GTOval", &[[0.0, 0.0, 0.0]]).unwrap();
    let out_sph = eval_gto(&mol, "GTOval_sph", &[[0.0, 0.0, 0.0]]).unwrap();
    assert_eq!(out_alias.values, out_sph.values);
}

#[test]
fn cart_variant_works_for_s_shells() {
    let mol = h_at_origin_sto3g();
    let out_cart = eval_gto(&mol, "GTOval_cart", &[[0.0, 0.0, 0.0]]).unwrap();
    let out_sph = eval_gto(&mol, "GTOval_sph", &[[0.0, 0.0, 0.0]]).unwrap();
    // For s shells, sph and cart yield identical AOs (Y_00 = 1, no
    // harmonic transform).
    approx::assert_abs_diff_eq!(out_cart.values[0], out_sph.values[0], epsilon = 1e-12);
}

#[test]
fn unbuilt_mol_errors() {
    let mol = pyscf_core::Mole::default();
    let r = eval_gto(&mol, "GTOval_sph", &[[0.0, 0.0, 0.0]]);
    assert!(matches!(r, Err(pyscf_core::PyscfRsError::Core(_))));
}

#[test]
fn unknown_variant_errors() {
    let mol = h_at_origin_sto3g();
    let r = eval_gto(&mol, "GTOval_quantum_voodoo", &[[0.0, 0.0, 0.0]]);
    assert!(matches!(r, Err(pyscf_core::PyscfRsError::Core(_))));
}
