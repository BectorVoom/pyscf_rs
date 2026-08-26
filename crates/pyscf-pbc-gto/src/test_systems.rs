//! The five shared PBC reference systems — PBC-MASTER-PLAN §9.2.
//!
//! **Use these everywhere. Do NOT redefine them per crate.** Every later PBC
//! crate reaches them with
//! `pyscf-pbc-gto = { path = "../pyscf-pbc-gto", features = ["test-systems"] }`
//! in its `[dev-dependencies]`.
//!
//! | name | cell | basis / PP | why |
//! |---|---|---|---|
//! | [`diamond`] | C2, fcc `a = 3.5668 A` | `gth-szv` / `gth-pade` | smallest realistic 3D insulator; 8 AOs |
//! | [`si`] | Si2, fcc `a = 5.4306 A` | `gth-szv` / `gth-pade` | narrow gap, occupation edge cases |
//! | [`lif`] | LiF, rocksalt `a = 4.03 A` | `gth-szv` / `gth-pade` | strongly ionic; large Ewald term |
//! | [`he_fcc`] | He, fcc `a = 3.0 A` | `gth-szv` | tiny; the all-electron `get_nuc` path |
//! | [`graphene`] | C2, hexagonal, 20 A vacuum | `gth-szv` / `gth-pade` | the `dimension = 2` path (Phase 12) |
//!
//! # Pseudopotentials
//!
//! Since plan 10-01 (D-PBC-11) `pseudo` is PARSED, not merely recorded: these
//! systems carry real `gth-pade` data, `atom_charges()` returns the valence
//! charge (diamond: `[4, 4]`, not `[6, 6]`) and `tot_electrons` follows. What
//! does NOT change is the geometry — lattices, volumes, reciprocal vectors,
//! G-vectors, k-meshes — nor `nao_nr`, which comes from the basis alone.
//!
//! # Reference values
//!
//! [`REFERENCES`] carries the tier-2 hard-coded numbers (D-PBC-19), generated
//! once from live PySCF 2.12.1 and committed. See `tests/cell_build.rs`.

use crate::cell::Cell;
use crate::types::{ALattice, CellBuildArgs, LowDimFtType};
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};

/// Face-centred-cubic PRIMITIVE lattice for conventional cube edge `a0`:
/// `a1 = (0, a0/2, a0/2)`, `a2 = (a0/2, 0, a0/2)`, `a3 = (a0/2, a0/2, 0)`.
/// Volume is `a0^3 / 4`.
fn fcc(a0: f64) -> ALattice {
    let h = a0 / 2.0;
    ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
}

/// Build a cell from Cartesian Angstrom coordinates and a `gth-szv` basis.
fn gth_szv_cell(
    a: ALattice,
    atoms: Vec<(String, [f64; 3])>,
    pseudo: Option<&str>,
    dimension: u8,
) -> Cell {
    let mole = MoleBuildArgs {
        atom: AtomInput::Tuples(atoms),
        basis: BasisInput::Name("gth-szv".into()),
        unit: Unit::Ang,
        ..Default::default()
    };
    let args = CellBuildArgs {
        mole,
        a,
        dimension,
        low_dim_ft_type: LowDimFtType::None,
        pseudo: pseudo.map(str::to_string),
        ..Default::default()
    };
    Cell::build(args).expect("reference system must build")
}

/// C2 diamond, fcc `a = 3.5668 A`, `gth-szv` / `gth-pade`.
///
/// The second carbon sits at scaled `(0.25, 0.25, 0.25)`, which for the fcc
/// primitive lattice above is Cartesian `(a0/4, a0/4, a0/4)`.
pub fn diamond() -> Cell {
    let a0 = 3.5668;
    let q = a0 / 4.0;
    gth_szv_cell(
        fcc(a0),
        vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])],
        Some("gth-pade"),
        3,
    )
}

/// Si2, fcc `a = 5.4306 A`, `gth-szv` / `gth-pade`. Same structure as
/// [`diamond`]; the narrow gap exercises occupation edge cases.
pub fn si() -> Cell {
    let a0 = 5.4306;
    let q = a0 / 4.0;
    gth_szv_cell(
        fcc(a0),
        vec![("Si".into(), [0.0, 0.0, 0.0]), ("Si".into(), [q, q, q])],
        Some("gth-pade"),
        3,
    )
}

/// LiF rocksalt, fcc `a = 4.03 A`, `gth-szv` / `gth-pade`.
/// F sits at scaled `(0.5, 0.5, 0.5)` = Cartesian `(a0/2, a0/2, a0/2)`.
pub fn lif() -> Cell {
    let a0 = 4.03;
    let h = a0 / 2.0;
    gth_szv_cell(
        fcc(a0),
        vec![("Li".into(), [0.0, 0.0, 0.0]), ("F".into(), [h, h, h])],
        Some("gth-pade"),
        3,
    )
}

/// A single He on an fcc lattice with `a = 3.0 A`, `gth-szv` / `gth-pade`.
/// The smallest system in the set — one atom, one AO.
pub fn he_fcc() -> Cell {
    gth_szv_cell(
        fcc(3.0),
        vec![("He".into(), [0.0, 0.0, 0.0])],
        Some("gth-pade"),
        3,
    )
}

/// Graphene: C2 on a hexagonal lattice with `a = 2.46 A` and 20 A of vacuum
/// along `z`. `dimension = 2` — the Phase 12 low-dimensional path.
///
/// Lattice: `a1 = (a, 0, 0)`, `a2 = (-a/2, a*sqrt(3)/2, 0)`, `a3 = (0, 0, 20)`.
/// The second carbon is at `(0, a/sqrt(3), 0)`.
pub fn graphene() -> Cell {
    let a0 = 2.46_f64;
    let a = ALattice::Matrix([
        [a0, 0.0, 0.0],
        [-a0 / 2.0, a0 * 3.0_f64.sqrt() / 2.0, 0.0],
        [0.0, 0.0, 20.0],
    ]);
    gth_szv_cell(
        a,
        vec![
            ("C".into(), [0.0, 0.0, 0.0]),
            ("C".into(), [0.0, a0 / 3.0_f64.sqrt(), 0.0]),
        ],
        Some("gth-pade"),
        2,
    )
}

/// Every reference system, paired with its name — for sweep tests.
pub fn all() -> Vec<(&'static str, Cell)> {
    vec![
        ("diamond", diamond()),
        ("si", si()),
        ("lif", lif()),
        ("he_fcc", he_fcc()),
        ("graphene", graphene()),
    ]
}

/// Tier-2 hard-coded reference values (D-PBC-19 / §9.1 tier 2).
///
/// Generated ONCE from live PySCF 2.12.1 with the identical cell definitions
/// (`.venv/bin/python`, `pyscf.pbc.gto.Cell` with `unit='Angstrom'`,
/// `basis='gth-szv'`, `pseudo='gth-pade'`) and committed. `vol` is in Bohr^3.
///
/// `nelectron_pp` is upstream's PSEUDOPOTENTIAL (valence) count. Since plan
/// 10-01 it IS what [`Cell::tot_electrons`] returns for these systems.
pub struct Reference {
    /// System name, matching the constructor function.
    pub name: &'static str,
    /// `cell.vol` in Bohr^3.
    pub vol: f64,
    /// `cell.natm`.
    pub natm: usize,
    /// `cell.nao_nr()`.
    pub nao_nr: usize,
    /// `cell.nelectron` WITH `gth-pade` — what `tot_electrons(1)` returns since
    /// plan 10-01.
    pub nelectron_pp: usize,
    /// `cell.lattice_vectors()` in Bohr, one row per lattice vector.
    pub a_bohr: [[f64; 3]; 3],
}

/// See [`Reference`].
pub const REFERENCES: [Reference; 5] = [
    Reference {
        name: "diamond",
        vol: 76.55488063251218,
        natm: 2,
        nao_nr: 8,
        nelectron_pp: 8,
        a_bohr: [
            [0.0, 3.3701375705493315, 3.3701375705493315],
            [3.3701375705493315, 0.0, 3.3701375705493315],
            [3.3701375705493315, 3.3701375705493315, 0.0],
        ],
    },
    Reference {
        name: "si",
        vol: 270.1967093603764,
        natm: 2,
        nao_nr: 8,
        nelectron_pp: 8,
        a_bohr: [
            [0.0, 5.131173346031512, 5.131173346031512],
            [5.131173346031512, 0.0, 5.131173346031512],
            [5.131173346031512, 5.131173346031512, 0.0],
        ],
    },
    Reference {
        name: "lif",
        vol: 110.42101837541341,
        natm: 2,
        nao_nr: 6,
        nelectron_pp: 10,
        a_bohr: [
            [0.0, 3.8077981409986, 3.8077981409986],
            [3.8077981409986, 0.0, 3.8077981409986],
            [3.8077981409986, 3.8077981409986, 0.0],
        ],
    },
    Reference {
        name: "he_fcc",
        vol: 45.551257834162435,
        natm: 1,
        nao_nr: 1,
        nelectron_pp: 2,
        a_bohr: [
            [0.0, 2.8345891868475928, 2.8345891868475928],
            [2.8345891868475928, 0.0, 2.8345891868475928],
            [2.8345891868475928, 2.8345891868475928, 0.0],
        ],
    },
    Reference {
        name: "graphene",
        vol: 707.3387370358154,
        natm: 2,
        nao_nr: 8,
        nelectron_pp: 8,
        a_bohr: [
            [4.648726266430052, 0.0, 0.0],
            [-2.324363133215026, 4.025915041968412, 0.0],
            [0.0, 0.0, 37.79452249130124],
        ],
    },
];
