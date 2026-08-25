//! GTH pseudopotentials (D-PBC-11).
//!
//! Plan 09-03 lands only the PLACEHOLDER type so [`crate::Cell`] can declare its
//! `pseudo` field. The parser (`pyscf/pbc/gto/pseudo/parse_cp2k.py`) and the
//! local/non-local matrix elements arrive in plan 10-01; until then
//! [`crate::Cell::pseudo`] is always `None` and the pseudopotential NAME the
//! user supplied is preserved verbatim in [`crate::Cell::pseudo_name`] so it
//! survives a `dumps`/`loads` round-trip.

use serde::{Deserialize, Serialize};

/// Parsed GTH pseudopotential data for one cell.
///
/// Deliberately empty in plan 09-03 — it exists so `Cell`'s field type is final
/// and later plans only have to fill the body, never change the signature.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PseudoData {}
