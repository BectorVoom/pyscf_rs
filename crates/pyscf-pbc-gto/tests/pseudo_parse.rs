//! Plan 10-01 — the GTH pseudopotential data model (D-PBC-11).
//!
//! Reference numbers read straight out of `pyscf/pbc/gto/pseudo/gth-pade.dat`
//! (the `C GTH-PADE-q4` and `Si GTH-PADE-q4` blocks) and cross-checked against
//! upstream's parsed `cell._pseudo` dict.

use pyscf_pbc_gto::pseudo::{PseudoData, normalise_symbol, resolve_pseudo};
use pyscf_pbc_gto::test_systems;

#[test]
fn gth_pade_carbon_matches_the_datafile() {
    let atoms = vec![("C".to_string(), [0.0, 0.0, 0.0])];
    let data: PseudoData = resolve_pseudo("gth-pade", &atoms).expect("resolve gth-pade for C");

    let pp = data.get("C").expect("C must be in gth-pade");

    // `2 2` — two s and two p valence electrons; Zion = 4.
    assert_eq!(pp.nelec, vec![2, 2]);
    assert_eq!(data.zion("C"), Some(4));

    // `0.34883045 2 -8.51377110 1.22843203`
    assert_eq!(pp.rloc, 0.34883045);
    assert_eq!(pp.local_coeffs, vec![-8.51377110, 1.22843203]);

    // `2` projector channels: l=0 with one projector, l=1 with none.
    assert_eq!(pp.projectors.len(), 2);
    assert_eq!(pp.projectors[0].r, 0.30455321);
    assert_eq!(pp.projectors[0].nproj, 1);
    assert_eq!(pp.projectors[0].h, vec![9.52284179]);
    assert_eq!(pp.projectors[1].r, 0.23267730);
    assert_eq!(pp.projectors[1].nproj, 0);
    assert!(pp.projectors[1].h.is_empty());
}

/// Si has a 2x2 `h` block whose upper triangle spans a continuation line —
/// the mirrored lower triangle is what `_contract_ppnl` consumes.
#[test]
fn gth_pade_silicon_h_matrix_is_symmetric_and_mirrored() {
    let atoms = vec![("Si".to_string(), [0.0, 0.0, 0.0])];
    let data = resolve_pseudo("gth-pade", &atoms).expect("resolve gth-pade for Si");
    let pp = data.get("Si").expect("Si must be in gth-pade");

    assert_eq!(pp.nelec, vec![2, 2]);
    assert_eq!(pp.rloc, 0.44000000);
    assert_eq!(pp.local_coeffs, vec![-7.33610297]);

    let l0 = &pp.projectors[0];
    assert_eq!(l0.nproj, 2);
    // file: 5.90692831 -1.26189397 / 3.25819622
    assert_eq!(l0.h, vec![5.90692831, -1.26189397, -1.26189397, 3.25819622]);
    assert_eq!(pp.projectors[1].h, vec![2.72701346]);
}

/// The single line that changes every downstream energy: `_atm[CHARGE_OF]`
/// becomes `Zion`, so `atom_charges()` and `tot_electrons()` are PP-adjusted.
#[test]
fn cell_build_applies_valence_charges() {
    let cell = test_systems::diamond();

    assert!(cell.pseudo.is_some(), "diamond is a gth-pade cell");
    assert_eq!(
        cell.atom_charges(),
        vec![4, 4],
        "C valence charge is 4, not 6"
    );
    assert_eq!(cell.tot_electrons(1), 8);
    assert_eq!(cell.tot_electrons(8), 64);
    assert_eq!(cell.mol.nelectron, 8);

    // LiF: Li-q3 and F-q7 in gth-pade.
    let lif = test_systems::lif();
    assert_eq!(lif.atom_charges(), vec![3, 7]);
    assert_eq!(lif.tot_electrons(1), 10);
}

/// A cell WITHOUT `pseudo` keeps the all-electron charges.
#[test]
fn all_electron_cell_is_untouched() {
    let mut cell = test_systems::diamond();
    // Rebuild the same geometry with no pseudopotential.
    cell.pseudo = None;
    let ae = {
        use pyscf_core::Unit;
        use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
        use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};
        let h = 3.5668 / 2.0;
        Cell::build(CellBuildArgs {
            mole: MoleBuildArgs {
                atom: AtomInput::Tuples(vec![
                    ("C".into(), [0.0, 0.0, 0.0]),
                    ("C".into(), [0.8917, 0.8917, 0.8917]),
                ]),
                basis: BasisInput::Name("gth-szv".into()),
                unit: Unit::Ang,
                ..Default::default()
            },
            a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
            pseudo: None,
            ..Default::default()
        })
        .expect("all-electron diamond builds")
    };
    assert!(ae.pseudo.is_none());
    assert_eq!(ae.atom_charges(), vec![6, 6]);
    assert_eq!(ae.tot_electrons(1), 12);
}

#[test]
fn symbol_normalisation_follows_the_basis_convention() {
    assert_eq!(normalise_symbol("C1"), "C");
    assert_eq!(normalise_symbol("si"), "SI");
    assert_eq!(normalise_symbol("Fe2+"), "FE");
}

/// `loads(dumps(cell))` must not silently drop the pseudopotential.
#[test]
fn pseudo_survives_a_dumps_loads_round_trip() {
    let cell = test_systems::diamond();
    let json = pyscf_pbc_gto::dumps(&cell).expect("dumps");
    let back = pyscf_pbc_gto::loads(&json).expect("loads");
    assert_eq!(back.pseudo, cell.pseudo);
    assert_eq!(back.atom_charges(), vec![4, 4]);
    assert_eq!(back.tot_electrons(1), 8);
}
