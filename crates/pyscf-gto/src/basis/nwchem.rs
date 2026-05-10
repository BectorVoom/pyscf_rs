//! NWChem / Gaussian-94 basis-set parser.
//!
//! Source: `pyscf/gto/basis/parse_nwchem.py` (Apache-2.0).
//!
//! Task 1 ships a stub returning a structured error so the module skeleton
//! compiles. Task 2 (TDD) fills the body with the upstream port.

use pyscf_core::{BasisLoadError, ParsedBasis};

/// Parse NWChem / Gaussian-94 text for one element symbol.
/// Task 2 fills the body; this stub keeps Task 1 compiling.
pub fn parse_nwchem(
    _text: &str,
    _symbol: &str,
    source: &str,
) -> Result<ParsedBasis, BasisLoadError> {
    Err(BasisLoadError::Parse {
        file: source.into(),
        line: 0,
        reason: "parse_nwchem not yet implemented (Task 2 of plan 02-03)".into(),
    })
}
