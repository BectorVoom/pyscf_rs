//! Molecule type. Phase 1 declares the shape; Phase 2 (GTO-01..11)
//! implements geometry/basis/ECP loading and the ≥30 attribute floor
//! (GTO-08).

use crate::error::PyscfRsError;

/// Molecular structure — atoms, basis set, charge, spin, electrons.
/// Phase 1 stub: only the bare-minimum shape so downstream traits can
/// reference `&Mole`. Phase 2 fills geometry parsing, basis loading,
/// `_atm`/`_bas`/`_env` arrays, and the full ≥30 attribute surface
/// (GTO-08 in REQUIREMENTS.md).
#[derive(Debug, Default, Clone)]
pub struct Mole {
    /// Atom positions in Bohr. Phase 2 parses from upstream PySCF
    /// atom-input forms (string, list-of-tuples, file path, etc. —
    /// GTO-01).
    pub atom_coords: Vec<[f64; 3]>,
    /// Atomic numbers, one per atom. Phase 2 fills.
    pub atom_charges: Vec<u8>,
    /// Total molecular charge (electrons subtracted). Phase 2 wires.
    pub charge: i32,
    /// Spin multiplicity − 1 (so singlet = 0). Phase 2 wires.
    pub spin: i32,
    /// Number of electrons. Phase 2 wires (computed from charges -
    /// charge).
    pub nelectron: usize,
}

impl Mole {
    /// Phase 2 (GTO-04) implements `mol.build()` to populate `_atm`,
    /// `_bas`, `_env`. Phase 1 returns a not-yet-implemented error.
    pub fn build(&mut self) -> Result<&mut Self, PyscfRsError> {
        Err(PyscfRsError::NotYetImplemented {
            phase: 2,
            what: "Mole::build (GTO-04)",
        })
    }
}
