//! GTO-10: `set_geom_` — in-place geometry mutation with granular cache
//! invalidation per RESEARCH Pattern 5.
//!
//! Source: `pyscf/gto/mole.py` `set_geom_` + RESEARCH.md Pattern 5
//! ("Architecture Patterns" — granular invalidation).
//!
//! Contract:
//!   - `_atom`'s xyz coords AND `_env[PTR_COORD..PTR_COORD+3]` slots are
//!     mutated in place.
//!   - Atom count + per-position symbol must match the existing geometry —
//!     basis structure (`_bas`, `_basis`, `ao_loc_nr`, `nao_nr`) MUST
//!     remain valid, so we forbid any change that would invalidate them.
//!   - `basis_set` `Arc` identity is preserved (zero-copy contract from
//!     GTO-11).
//!   - `mol.unit` is honoured: a position string parsed with `Unit::Ang`
//!     is converted to Bohr before the mutation lands.
//!
//! v1 limitation: `Arc<BasisSet>.atoms[*].coord` is a snapshot at original
//! construction time. After `set_geom_`, the libcint-facing source of
//! truth is `_env`; downstream consumers that need a coord-fresh
//! `BasisSet` must rebuild via `pyscf_gto::M(MoleBuildArgs{ ... })`
//! instead. This is intentional — see threat model T-02-08-05.

use crate::format_atom;
use crate::types::AtomInput;
use pyscf_core::raw_layout::{ATM_SLOTS, PTR_COORD};
use pyscf_core::{CoreError, Mole, PyscfRsError};

/// Mutate `mol`'s geometry in place from a new atom-string in the same
/// format `format_atom` accepts.
///
/// Returns `&mut Mole` so calls chain naturally.
///
/// Errors:
///   - Atom count mismatch ⇒ `CoreError::InvalidMolecule`.
///   - Per-position symbol mismatch ⇒ `CoreError::InvalidMolecule`.
///   - Underlying `format_atom` parse errors propagate as-is.
pub fn set_geom_<'m>(
    mol: &'m mut Mole,
    new_atom: &str,
) -> Result<&'m mut Mole, PyscfRsError> {
    // Parse the new atoms using the SAME unit (and identity origin/axes);
    // the contract is that `set_geom_` honours `mol.unit` so an Ang user
    // can keep typing Ang strings.
    let parsed = format_atom::format_atom(
        &AtomInput::String(new_atom.to_string()),
        mol.unit,
        [0.0_f64; 3],
        [
            [1.0_f64, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )?;

    // Validate atom count.
    if parsed.len() != mol._atom.len() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "set_geom_ atom count mismatch: was {}, got {}",
            mol._atom.len(),
            parsed.len()
        ))));
    }

    // Validate symbols (basis is keyed by symbol; must NOT change).
    for (i, (old, new)) in mol._atom.iter().zip(parsed.iter()).enumerate() {
        if old.0 != new.0 {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "set_geom_ symbol mismatch at atom {}: was '{}', got '{}'",
                i, old.0, new.0
            ))));
        }
    }

    // === Granular invalidation (Pattern 5) ===
    // 1. Update _atom (the typed coords).
    mol._atom = parsed;

    // 2. Update _env coordinate slots — for each atom, _atm[PTR_COORD]
    //    tells us the env offset. We mutate env[ptr..ptr+3] in place.
    //    _bas / _basis / ao_loc_nr / nao_nr / basis_set Arc are PRESERVED
    //    (zero invalidation).
    for (i, (_sym, coord)) in mol._atom.iter().enumerate() {
        let row_start = i * ATM_SLOTS;
        let ptr_coord = mol._atm[row_start + PTR_COORD] as usize;
        mol._env[ptr_coord] = coord[0];
        mol._env[ptr_coord + 1] = coord[1];
        mol._env[ptr_coord + 2] = coord[2];
    }
    // _atm rows: CHARGE_OF, NUC_MOD_OF, PTR_ZETA all unchanged (geometry-
    // independent). basis_set Arc: NOT replaced — see module doc-comment
    // for the v1 limitation note.
    Ok(mol)
}
