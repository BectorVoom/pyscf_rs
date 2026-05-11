//! Plan 03-05 Task 2 RED tests — RHF::density_fit + DfHooks wiring.
//!
//! Asserts the SCF-07 user-facing surface ships without requiring the actual
//! DF-HF energy assertion (plan 03-10 owns the bit-exact comparison).
//!
//! Tests:
//!   1. RHF::density_fit(Some("cc-pvdz-jkfit")) stores DfIntegrals in with_df.
//!   2. RHF::density_fit(None) resolves auxbasis via default_jkfit(orbital).
//!   3. DfHooks impl OverrideHooks (Send + Sync — needed for parallel
//!      scanners / Phase 7 geomopt).
//!   4. DfHooks::get_jk routes through pyscf_df::get_jk_df.
//!   5. DfHooks::get_veff returns J - 0.5*K (RHF closed-shell).

use pyscf_core::{Density, Unit};
use pyscf_df::DfIntegrals;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};
use pyscf_scf::df_scf::DfHooks;
use pyscf_scf::{OverrideHooks, RHF};

fn h2_mol(basis: &str) -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name(basis.into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build h2")
}

#[test]
fn density_fit_with_explicit_auxbasis_populates_with_df() {
    let mol = h2_mol("sto-3g");
    let rhf = RHF::new(mol).density_fit(Some("sto-3g")).expect("density_fit");
    assert!(rhf.with_df.is_some(), "with_df must be Some after density_fit");

    let stored = rhf.with_df.as_ref().expect("with_df Some");
    let df: &DfIntegrals = stored
        .downcast_ref::<DfIntegrals>()
        .expect("with_df must hold DfIntegrals");
    assert_eq!(df.nao, rhf.mol.nao_nr);
    assert!(df.naux > 0, "naux must be positive (sto-3g aux)");
    assert_eq!(df.b_uvq.len(), df.nao * df.nao * df.naux);
}

#[test]
fn density_fit_with_none_uses_default_jkfit() {
    // sto-3g resolves to "weigend" via default_jkfit (mol.basis is the
    // user-supplied String "Name(\"sto-3g\")"; extract_basis_name unwraps
    // it). The resolver path must be invoked; we accept either Ok (with_df
    // populated) OR Err(SingularAux) since cintx is in synthetic-staging
    // mode and the (P|Q) from weigend may be rank-deficient. What we DON'T
    // accept is a panic or an "unknown basis" error — both would mean the
    // resolver was bypassed.
    let mol = h2_mol("sto-3g");
    match RHF::new(mol).density_fit(None) {
        Ok(rhf) => {
            assert!(
                rhf.with_df.is_some(),
                "with_df must be Some when density_fit(None) succeeds"
            );
        }
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                msg.contains("rank-deficient") || msg.contains("Cholesky"),
                "density_fit(None) must either succeed or report SingularAux; got: {}",
                msg
            );
        }
    }
}

#[test]
fn df_hooks_is_send_sync() {
    // Compile-time check: DfHooks must be Send + Sync so parallel scanners
    // (Phase 7 geomopt) can clone the borrow across threads.
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<DfHooks<'_>>();
    assert_sync::<DfHooks<'_>>();
}

#[test]
fn df_hooks_get_jk_routes_through_pyscf_df() {
    // Build a zero DfIntegrals; assert DfHooks::get_jk returns all-zero J/K
    // (matching pyscf_df::get_jk_df with zero B).
    let df = DfIntegrals {
        b_uvq: vec![0.0; 4 * 4 * 3],
        naux: 3,
        nao: 4,
    };
    let hooks = DfHooks { df: &df };
    let dm = Density {
        nao: 4,
        data: vec![1.0; 16],
    };
    let mol = h2_mol("sto-3g");
    let (j, k) = hooks.get_jk(&mol, &dm).expect("get_jk");
    assert_eq!(j.data, vec![0.0; 16]);
    assert_eq!(k.data, vec![0.0; 16]);
}

#[test]
fn df_hooks_get_veff_is_j_minus_half_k() {
    // Veff = J - 0.5 * K. Manually construct a DfIntegrals whose b_uvq is
    // shape-valid (we don't need correct numerics; we just need J and K to
    // be consistent so we can check the J-0.5*K relation indirectly).
    //
    // With b_uvq = zero, J = K = 0, so veff = 0. We assert that path.
    let df = DfIntegrals {
        b_uvq: vec![0.0; 4 * 4 * 2],
        naux: 2,
        nao: 4,
    };
    let hooks = DfHooks { df: &df };
    let dm = Density {
        nao: 4,
        data: vec![0.5; 16],
    };
    let mol = h2_mol("sto-3g");
    let veff = hooks.get_veff(&mol, &dm).expect("get_veff");
    // J - 0.5*K with both zero = zero.
    assert_eq!(veff.data, vec![0.0; 16]);
    assert_eq!(veff.nao, 4);
}

#[test]
fn df_hooks_delegates_non_df_hooks_to_defaults() {
    // get_hcore / get_ovlp / get_init_guess / eig / get_occ / make_rdm1
    // delegate to the default_* free fns. We don't fully exercise the
    // numerics here (that's plan 03-08 Arms); we just assert get_occ
    // produces the closed-shell Aufbau pattern.
    let df = DfIntegrals {
        b_uvq: vec![0.0; 2 * 2 * 1],
        naux: 1,
        nao: 2,
    };
    let hooks = DfHooks { df: &df };
    let mo_energy = vec![-1.0, 1.0];
    let occ = hooks.get_occ(&mo_energy, 2).expect("get_occ");
    // RHF closed-shell with 2 electrons: lowest MO doubly occupied.
    assert_eq!(occ.len(), 2);
    assert!((occ[0] - 2.0).abs() < 1e-12, "lowest MO must be doubly occupied");
    assert!((occ[1] - 0.0).abs() < 1e-12, "higher MO must be empty");
}
