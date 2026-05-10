//! NWChem ECP-block parser (loading half of GTO-05).
//!
//! Source: `pyscf/gto/basis/parse_nwchem_ecp.py` (Apache-2.0).
//!
//! Task 1 ships a stub; Task 2 fills the body.

use pyscf_core::{EcpLoadError, ParsedEcp};

/// Parse a NWChem ECP block for one element symbol.
/// Task 2 fills the body; this stub keeps Task 1 compiling.
pub fn parse_nwchem_ecp(
    _text: &str,
    _symbol: &str,
    source: &str,
) -> Result<ParsedEcp, EcpLoadError> {
    Err(EcpLoadError::Parse {
        file: source.into(),
        line: 0,
        reason: "parse_nwchem_ecp not yet implemented (Task 2 of plan 02-03)".into(),
    })
}
