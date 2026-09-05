#![allow(dead_code)]
//! Shared fixtures for the Phase-16 `pyscf-pbc-cc` tests.
//!
//! `diamond` is `PBC-MASTER-PLAN §9.2`'s reference cell. The mesh is PINNED at
//! `[15,15,15]` for every test, matching the pin 16-01's measurements ran under
//! (`measurements/README.md`, fixture pin): at `cell.precision = 1e-8` the
//! default mesh is `[47,47,47]`, where one `KRHF` at `[1,1,2]` alone costs
//! 79 s. Every gate number these tests use was measured at the same pin, so
//! the two sides are comparable.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// `§9.2` `diamond` — C2 fcc `a = 3.5668 A`, `gth-szv` / `gth-pade`, mesh pinned.
pub fn diamond(mesh: [usize; 3]) -> Cell {
    let a0 = 3.5668;
    let q = a0 / 4.0;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                ("C".into(), [q, q, q]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([
            [0.0, a0 / 2.0, a0 / 2.0],
            [a0 / 2.0, 0.0, a0 / 2.0],
            [a0 / 2.0, a0 / 2.0, 0.0],
        ]),
        pseudo: Some("gth-pade".into()),
        mesh: Some(mesh),
        ..Default::default()
    })
    .expect("diamond cell")
}
