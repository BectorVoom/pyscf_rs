//! CP2K pseudopotential parser (Phase-2 stub).
//!
//! Source: `pyscf/gto/basis/parse_cp2k_pp.py` (Apache-2.0).

use pyscf_core::{EcpLoadError, ParsedEcp};

/// Stub — Phase 2 ships no CP2K pseudopotential support; future-proof the surface.
pub fn parse_cp2k_pp(
    _text: &str,
    _symbol: &str,
    source: &str,
) -> Result<ParsedEcp, EcpLoadError> {
    Err(EcpLoadError::Parse {
        file: source.into(),
        line: 0,
        reason: "CP2K pseudopotential parsing not yet implemented in Phase 2".into(),
    })
}
