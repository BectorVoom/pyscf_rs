#![allow(dead_code)]

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

pub fn diamond_anchor() -> Cell {
    let h = 3.370137329;
    let q = 1.685068664391;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("diamond cell")
}

pub fn helium_631g() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("6-31g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([
            [0.0, 2.834589, 2.834589],
            [2.834589, 0.0, 2.834589],
            [2.834589, 2.834589, 0.0],
        ]),
        mesh: Some([9, 9, 9]),
        ..Default::default()
    })
    .expect("He/6-31g cell")
}

/// The fixture committed in `pyscf/pbc/mp/kmp2_stagger.py:357-374`: an H2 dimer
/// in a 6 Bohr cube, `gth-szv`/`gth-pade`, `ke_cutoff = 100`. It is the only
/// staggered-mesh system upstream ships reference energies for.
pub fn h2_dimer_stagger() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("H".into(), [3.00, 3.00, 2.10]),
                ("H".into(), [3.00, 3.00, 3.90]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]),
        ke_cutoff: Some(100.0),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("H2 dimer stagger cell")
}
