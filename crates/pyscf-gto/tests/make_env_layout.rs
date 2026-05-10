//! GTO-04 verification: `make_env` produces correctly-shaped `_atm` / `_bas`
//! / `_env` / `ao_loc_nr` / `nao_nr` arrays for representative molecules.
//!
//! Covers Pitfalls 2 (slot-dim invariants), 4 (per-symbol first-occurrence
//! grouping), 5 (KAPPA_OF == 0 for sph/cart), and the `_built = true` flag.

use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF,
    NUC_MOD_OF, POINT_NUC, PTR_COORD, PTR_ENV_START,
};
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};

fn h2_sto3g_mol() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("H2/STO-3G build must succeed")
}

#[test]
fn h2_sto3g_atm_dimensions() {
    let mol = h2_sto3g_mol();
    assert_eq!(mol._atm.len(), 2 * ATM_SLOTS, "_atm = natm * ATM_SLOTS");
    assert_eq!(
        mol._atm.len() % ATM_SLOTS,
        0,
        "Pitfall 2: _atm dimension must be a multiple of ATM_SLOTS"
    );
    assert_eq!(mol.natm, 2);
}

#[test]
fn h2_sto3g_bas_dimensions() {
    let mol = h2_sto3g_mol();
    // STO-3G H = 1 ShellSpec; 2 atoms cloned = 2 _bas rows.
    assert_eq!(mol._bas.len(), 2 * BAS_SLOTS);
    assert_eq!(
        mol._bas.len() % BAS_SLOTS,
        0,
        "Pitfall 2: _bas dimension must be a multiple of BAS_SLOTS"
    );
    assert_eq!(mol.nbas, 2);
}

#[test]
fn h2_sto3g_env_reserved_zeros() {
    let mol = h2_sto3g_mol();
    assert!(mol._env.len() >= PTR_ENV_START);
    for i in 0..PTR_ENV_START {
        assert_eq!(
            mol._env[i], 0.0,
            "_env[{}] must be 0.0 (libcint reserved [0, PTR_ENV_START))",
            i
        );
    }
}

#[test]
fn h2_sto3g_atm_charges_and_ptr_coords() {
    let mol = h2_sto3g_mol();
    assert_eq!(mol._atm[CHARGE_OF], 1, "first H charge");
    assert_eq!(mol._atm[ATM_SLOTS + CHARGE_OF], 1, "second H charge");
    assert_eq!(mol._atm[PTR_COORD], PTR_ENV_START as i32);
    // First atom claims env[20..23] xyz + env[23] zeta = 4 doubles; second
    // atom starts at env[20+4] = 24.
    assert_eq!(mol._atm[ATM_SLOTS + PTR_COORD], (PTR_ENV_START + 4) as i32);
    assert_eq!(
        mol._atm[NUC_MOD_OF],
        POINT_NUC,
        "v1: NUC_MOD_OF=POINT_NUC for every atom"
    );
}

#[test]
fn h2_sto3g_bas_atom_of_patched() {
    let mol = h2_sto3g_mol();
    assert_eq!(mol._bas[ATOM_OF], 0);
    assert_eq!(mol._bas[BAS_SLOTS + ATOM_OF], 1);
    // Both shells share angular momentum (l=0 for s) and contraction shape.
    assert_eq!(mol._bas[ANG_OF], 0);
    assert_eq!(mol._bas[BAS_SLOTS + ANG_OF], 0);
    assert_eq!(mol._bas[NPRIM_OF], 3);
    assert_eq!(mol._bas[NCTR_OF], 1);
}

#[test]
fn h2_sto3g_bas_kappa_zero_for_sph() {
    let mol = h2_sto3g_mol();
    for i in 0..mol.nbas {
        assert_eq!(
            mol._bas[i * BAS_SLOTS + KAPPA_OF],
            0,
            "Pitfall 5: KAPPA_OF must be 0 for sph/cart in v1 (shell {i})"
        );
    }
}

#[test]
fn h2_sto3g_nao_and_ao_loc() {
    let mol = h2_sto3g_mol();
    assert_eq!(mol.nao_nr, 2, "H2/STO-3G has 2 AOs (1s on each H)");
    assert_eq!(mol.ao_loc_nr, vec![0, 1, 2]);
}

#[test]
fn h2_sto3g_built_flag() {
    let mol = h2_sto3g_mol();
    assert!(
        mol._built,
        "M(args) must set _built=true after make_env + projection"
    );
}

#[test]
fn h2o_sto3g_attribute_floor_populated() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 0.7 0.6; H 0 -0.7 0.6".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("H2O/STO-3G build must succeed");
    // STO-3G H2O: O has 1 S shell + 1 SP shell (parser splits SP into l=0 + l=1
    // ShellSpec entries) → 3 ShellSpec for O. H = 1 ShellSpec. Cloned per atom:
    // 3 (O) + 1 (H1) + 1 (H2) = 5 shells.
    assert_eq!(mol._atm.len(), 3 * ATM_SLOTS);
    assert_eq!(mol.nbas, 5);
    assert_eq!(mol._bas.len(), 5 * BAS_SLOTS);
    // ao_loc_nr cumsum: 0, 1 (O 1s), 2 (O 2s), 5 (O 2p×3), 6 (H1 1s), 7 (H2 1s).
    assert_eq!(mol.ao_loc_nr, vec![0, 1, 2, 5, 6, 7]);
    assert_eq!(mol.nao_nr, 7);
    assert!(mol._built);
}

#[test]
fn first_occurrence_per_symbol_grouping() {
    // H, O, H, O ordering — H appears first; H per-symbol block in _env
    // must come before O block (Pitfall 4).
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; O 0 0 1; H 0 0 2; O 0 0 3".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("H/O/H/O build must succeed");
    assert_eq!(mol.natm, 4);
    // After PTR_ENV_START reserved + 4 atoms × (3 xyz + 1 zeta) = 16 doubles,
    // the next entries are the H per-symbol exponents (3 floats), then H coeffs
    // (3 floats), then the O block.
    let h_sym_block_start = PTR_ENV_START + 4 * 4;
    // H STO-3G first exponent ≈ 3.42525091 (canonical value from sto-3g.dat).
    let h_first_exp = mol._env[h_sym_block_start];
    assert!(
        (h_first_exp - 3.42525091).abs() < 1e-6,
        "expected first env entry after xyz block to be H STO-3G first exponent 3.42525091, got {h_first_exp}"
    );
}

#[test]
fn slot_dimension_invariants_hold() {
    // Pitfall 2 across multiple molecules.
    for mol in [
        h2_sto3g_mol(),
        M(MoleBuildArgs {
            atom: AtomInput::String("O 0 0 0; H 0 0.7 0.6; H 0 -0.7 0.6".into()),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        })
        .unwrap(),
    ] {
        assert_eq!(mol._atm.len() % ATM_SLOTS, 0);
        assert_eq!(mol._bas.len() % BAS_SLOTS, 0);
        // ao_loc_nr has nbas+1 entries.
        assert_eq!(mol.ao_loc_nr.len(), mol.nbas + 1);
        // nao_nr equals last ao_loc_nr entry.
        assert_eq!(mol.nao_nr as i32, *mol.ao_loc_nr.last().unwrap());
    }
}
