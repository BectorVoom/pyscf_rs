//! GTO-05 (loading half): `format_ecp` + `make_ecp_env`.
//!
//! Source: `pyscf/gto/mole.py:1107-1160` (`make_ecp_env`) +
//! `pyscf/gto/basis/__init__.py:730-779` (`load_ecp` / `parse`).
//!
//! D-06 sequencing: loading ships in Phase 2; evaluation closes via
//! gap-closure plan 02-10 once cintx ECP merges.
//!
//! Plan 02-07 Task 1 (this commit) ships the module skeleton + an empty
//! `format_ecp` so the dispatcher (`crate::ecp_engine`) and the
//! `lib.rs` re-exports compile. Plan 02-07 Task 2 (TDD) fills the body
//! and adds `make_ecp_env`.

use crate::types::EcpInput;
use pyscf_core::{ParsedAtom, ParsedEcp, PyscfRsError};
use std::collections::HashMap;

/// Resolve `EcpInput` → per-element-symbol `ParsedEcp` map.
///
/// Mirrors the shape of `format_basis::format_basis` (Phase 2 02-03).
/// Per upstream `pyscf/gto/mole.py:1107-1160` semantics, symbols with
/// no ECP entry are simply absent from the returned map (NOT an error).
///
/// Plan 02-07 Task 1 ships an empty implementation that always returns
/// an empty map (matching `EcpInput::None`). Plan 02-07 Task 2 (TDD)
/// fills the dispatch.
pub fn format_ecp(
    _input: &EcpInput,
    _atoms: &[ParsedAtom],
) -> Result<HashMap<String, ParsedEcp>, PyscfRsError> {
    Ok(HashMap::new())
}

/// `make_ecp_env` — projection of ECP data to flat arrays. Mutates
/// `_atm` (CHARGE_OF subtraction per Pitfall 2) and `_env`, and returns
/// the freshly built `_ecpbas` row vector.
///
/// Source: `pyscf/gto/mole.py:1107-1160`.
///
/// Plan 02-07 Task 1 returns an empty `_ecpbas`. Plan 02-07 Task 2
/// (TDD) wires the projection.
pub fn make_ecp_env(
    _atoms: &[ParsedAtom],
    _ecp: &HashMap<String, ParsedEcp>,
    _atm: &mut [i32],
    _env: &mut Vec<f64>,
) -> Vec<i32> {
    let _ = (_atm, _env);
    Vec::new()
}
