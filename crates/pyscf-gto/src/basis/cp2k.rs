//! CP2K / GTH basis-set parser (Phase-2 stub).
//!
//! Source: `pyscf/gto/basis/parse_cp2k.py` (Apache-2.0).
//!
//! Phase 2's PR-CI corpus (H2O/cc-pVDZ + benzene/6-31G* + water-trimer/STO-3G)
//! does NOT need CP2K-format basis sets. The stub preserves the dispatch
//! surface so the loader's GTH-prefix detection routes consistently; Phase 4
//! (DFT) may revisit if a user surfaces the need.

use pyscf_core::{BasisLoadError, ParsedBasis};

/// Stub: returns a structured "not yet implemented" error. The dispatch
/// surface remains so future GTH-format basis files can be wired without
/// touching call sites.
pub fn parse_cp2k(_text: &str, _symbol: &str, source: &str) -> Result<ParsedBasis, BasisLoadError> {
    Err(BasisLoadError::Parse {
        file: source.into(),
        line: 0,
        reason: "CP2K basis parsing not yet implemented in Phase 2 (Phase 4 DFT may need this; \
                 pyscf-rs CI corpus uses NWChem-format basis only)"
            .into(),
    })
}
