//! GTH pseudopotentials (D-PBC-11) — data model, local part, non-local part.
//!
//! Plan 10-01 replaces the plan-09-03 placeholder with the real container.
//! The PARSER already exists (`pyscf_gto::basis::cp2k_pp`, which produces
//! [`pyscf_core::GthPseudo`]); this module owns the per-CELL view of it, the
//! `Zion` bookkeeping that `Cell::build` applies to `_atm[CHARGE_OF]`, and
//! (plans 10-05 / 10-06) the matrix elements themselves.
//!
//! # Upstream correspondence
//!
//! | upstream | here |
//! |---|---|
//! | `cell._pseudo` (`dict symbol -> [nelec, rloc, nexp, cexp, nproj_types, [...]]`) | [`PseudoData::per_symbol`] |
//! | `mole.py:2591` `self._atm[ia,0] = sum(_pseudo[symb][0])` | [`PseudoData::zion`] applied in `Cell::build` |
//! | `pyscf/pbc/gto/pseudo/pp.py`      | [`vloc`] |
//! | `pyscf/pbc/gto/pseudo/pp_int.py`  | [`vloc`] + [`vnl`] |

use pyscf_core::GthPseudo;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// Plan 10-05 — the local part (G-space form factors + the auxiliary expansion).
pub mod vloc;
pub mod vloc_part2;
// Plan 10-06 — the non-local part.
pub mod vnl;

pub use vloc_part2::{
    EPS_PPL, PART2_INTORS, PRESCREEN_EPS, get_pp_loc_part2, get_pp_loc_part2_gamma,
};

pub use vloc::{
    HALF_SPH_NORM, VlocAux, fake_cell_vloc, get_alphas, get_alphas_gth, get_coulg, get_gth_vlocg,
    get_gth_vlocg_part1, get_vlocg,
};

pub use vnl::{
    FakeCellVnl, HlBlock, MAX_NPROJ, PLI_FAC, VNL_INTORS, fake_cell_vnl, get_pp_nl, int_vnl,
};

/// Parsed GTH pseudopotential data for one cell, keyed by ELEMENT SYMBOL
/// (upper-cased alphabetic prefix, matching `Mole::_basis`' convention).
///
/// A `BTreeMap` rather than a `HashMap`: iteration order is part of the
/// bit-reproducibility contract (FOUND-06 / D-PBC-17) whenever a caller folds
/// over the whole table, and a `HashMap`'s order is seed-dependent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PseudoData {
    /// One entry per element that actually carries a pseudopotential. An atom
    /// whose symbol is absent is treated as ALL-ELECTRON (upstream allows a
    /// mixed cell; `mole.py:2588` only rewrites `_atm[CHARGE_OF]` for symbols
    /// present in `_pseudo`).
    pub per_symbol: BTreeMap<String, GthPseudo>,
}

impl PseudoData {
    /// `true` when no element carries a pseudopotential.
    pub fn is_empty(&self) -> bool {
        self.per_symbol.is_empty()
    }

    /// Look a symbol up the way upstream does — by the alphabetic prefix of the
    /// atom label, upper-cased (`"C1"` and `"c"` both hit the `"C"` entry).
    pub fn get(&self, symbol: &str) -> Option<&GthPseudo> {
        self.per_symbol.get(&normalise_symbol(symbol))
    }

    /// The effective core charge `Zion = Σ_l nelec[l]` for `symbol`, or `None`
    /// when that element is all-electron. Ports `mole.py:2591`.
    pub fn zion(&self, symbol: &str) -> Option<i32> {
        self.get(symbol)
            .map(|pp| pp.nelec.iter().map(|n| *n as i32).sum())
    }
}

/// The upper-cased alphabetic prefix of an atom label — the key convention
/// shared with `Mole::_basis` and [`PseudoData::per_symbol`].
pub fn normalise_symbol(symbol: &str) -> String {
    symbol
        .chars()
        .take_while(|c| c.is_alphabetic())
        .collect::<String>()
        .to_ascii_uppercase()
}

/// Resolve a pseudopotential NAME (e.g. `"gth-pade"`) for every distinct
/// element in `atoms`. Ports `mole.py:2578-2579`
/// (`_parse_default_basis` + `format_pseudo`) for the string-valued input.
///
/// Elements the file does not carry are SKIPPED rather than rejected: upstream
/// `_parse_default_basis` only assigns the named potential to atoms that have
/// one, and warns (`pp_int.py:141`) if none of the cell's elements matched.
///
/// # Errors
/// [`pyscf_core::PyscfRsError`] when the pseudopotential file itself cannot be
/// located or is malformed (as opposed to simply not covering an element).
pub fn resolve_pseudo(
    name: &str,
    atoms: &[pyscf_core::ParsedAtom],
) -> Result<PseudoData, pyscf_core::PyscfRsError> {
    let mut per_symbol: BTreeMap<String, GthPseudo> = BTreeMap::new();
    for (label, _) in atoms {
        let sym = normalise_symbol(label);
        if sym.is_empty() || per_symbol.contains_key(&sym) {
            continue;
        }
        match pyscf_gto::basis::load_pseudo(name, &sym) {
            Ok(pp) => {
                per_symbol.insert(sym, pp);
            }
            // "this element is not in the file" is not an error — see the doc
            // comment. Any other failure (missing file, malformed block) is.
            Err(pyscf_core::EcpLoadError::Parse { ref reason, .. })
                if reason.contains("not found") =>
            {
                tracing::debug!(
                    "pseudopotential '{name}' has no entry for element '{sym}'; \
                     treating it as all-electron"
                );
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(PseudoData { per_symbol })
}
