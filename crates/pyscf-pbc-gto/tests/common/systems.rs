//! The five shared PBC reference systems (PBC-MASTER-PLAN §9.2) as seen from
//! this crate's integration tests.
//!
//! The DEFINITIONS live once, in `pyscf_pbc_gto::test_systems` behind the
//! `test-systems` feature, so downstream PBC crates get the identical cells
//! through their own `[dev-dependencies]` entry
//! (`features = ["test-systems"]`) instead of copying them. §9.2 is explicit:
//! **do not redefine them per crate.** These wrappers exist so a test can write
//! `systems::diamond()` and so this file is the single place a future rename
//! would have to touch.

#![allow(dead_code)]
// Not every integration test uses every re-export (plan 09-04 added a second
// consumer that needs the cells but not the reference table).
#![allow(unused_imports)]

use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::test_systems;

pub use pyscf_pbc_gto::test_systems::REFERENCES;

/// C2 diamond, fcc `a = 3.5668 A`, `gth-szv` / `gth-pade`.
pub fn diamond() -> Cell {
    test_systems::diamond()
}

/// Si2, fcc `a = 5.4306 A`, `gth-szv` / `gth-pade`.
pub fn si() -> Cell {
    test_systems::si()
}

/// LiF rocksalt, fcc `a = 4.03 A`, `gth-szv` / `gth-pade`.
pub fn lif() -> Cell {
    test_systems::lif()
}

/// He on an fcc lattice, `a = 3.0 A`, `gth-szv` / `gth-pade`.
pub fn he_fcc() -> Cell {
    test_systems::he_fcc()
}

/// Graphene, hexagonal `a = 2.46 A` with 20 A vacuum, `dimension = 2`.
pub fn graphene() -> Cell {
    test_systems::graphene()
}

/// All five, paired with their names.
pub fn all() -> Vec<(&'static str, Cell)> {
    test_systems::all()
}
