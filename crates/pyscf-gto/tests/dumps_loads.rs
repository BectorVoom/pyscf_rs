//! GTO-09 — `dumps()` / `loads()` JSON round-trip.
//!
//! Per CONTEXT D-09 + RESEARCH "Don't Hand-Roll": the contract is
//! **semantic round-trip** (read your own JSON, get the same internal
//! arrays), NOT byte-identical to upstream PySCF JSON. Phase 8 ORACLE-08
//! covers cross-language chkfile interop.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, dumps, loads};

fn h2o_args() -> MoleBuildArgs {
    MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 0.7 0.6; H 0 -0.7 0.6".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    }
}

#[test]
fn h2o_dumps_loads_round_trip_preserves_arrays() {
    let mol_a = M(h2o_args()).unwrap();
    let json = dumps(&mol_a).unwrap();
    let mol_b = loads(&json).unwrap();
    assert_eq!(mol_a._atm, mol_b._atm, "_atm byte-equal across round-trip");
    assert_eq!(mol_a._bas, mol_b._bas, "_bas byte-equal");
    assert_eq!(mol_a._env, mol_b._env, "_env byte-equal");
    assert_eq!(mol_a.ao_loc_nr, mol_b.ao_loc_nr);
    assert_eq!(mol_a.nao_nr, mol_b.nao_nr);
    assert_eq!(mol_a.nelectron, mol_b.nelectron);
    assert_eq!(mol_a.charge, mol_b.charge);
    assert_eq!(mol_a.spin, mol_b.spin);
    assert_eq!(mol_a.cart, mol_b.cart);
    assert_eq!(mol_a.unit, mol_b.unit);
}

#[test]
fn cart_charge_spin_round_trip() {
    let args = MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        charge: 1,
        spin: 0,
        cart: true,
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    };
    let mol_a = M(args).unwrap();
    let json = dumps(&mol_a).unwrap();
    let mol_b = loads(&json).unwrap();
    assert!(mol_b.cart);
    assert_eq!(mol_b.charge, 1);
    assert_eq!(mol_b._atm, mol_a._atm);
    assert_eq!(mol_b._bas, mol_a._bas);
    assert_eq!(mol_b._env, mol_a._env);
}

#[test]
fn malformed_json_returns_error_not_panic() {
    let r = loads("not json at all");
    assert!(r.is_err());
    match r {
        Err(pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(msg))) => {
            assert!(
                msg.contains("loads") || msg.contains("serde_json"),
                "expected loads/serde_json marker in error message, got: {msg}"
            );
        }
        other => panic!("expected InvalidMolecule, got {:?}", other),
    }
}
