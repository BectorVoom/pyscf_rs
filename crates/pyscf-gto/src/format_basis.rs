//! GTO-02: `format_basis` — collapse the 11 upstream basis-input forms to
//! the 5-arm `BasisInput` match dispatch.
//!
//! Source: `pyscf/gto/mole.py:418-466` `format_basis` +
//! `pyscf/gto/basis/__init__.py:621-728` `load`.
//!
//! Per RESEARCH Pitfall 4: output keys are unique element symbols in
//! first-occurrence order from `atoms`. Ghost-atom suffixes (`H1`, `H2`)
//! collapse to their alphabetic prefix (`H`), so a water-trimer gives a
//! 2-key map (`H`, `O`) regardless of how many H atoms appear.

use crate::basis;
use crate::types::BasisInput;
use pyscf_core::{BasisLoadError, ParsedAtom, ParsedBasis, PyscfRsError};
use std::collections::HashMap;

/// Resolve the user-supplied `BasisInput` to a per-element-symbol map.
///
/// Output keys are UPPERCASE element symbols (matches `load_basis` convention).
/// Per-symbol order: first-occurrence in `atoms` (Pitfall 4).
///
/// The 11 upstream forms collapse as follows:
///
/// | Upstream form                        | `BasisInput` arm                |
/// |--------------------------------------|---------------------------------|
/// | `'cc-pvdz'`                          | `Name`                          |
/// | `'cc-pVDZ'` / `'CC-PVDZ'`            | `Name` (canonicalise normalises)|
/// | `gto.basis.parse(text)`              | `Parsed`                        |
/// | `{'O': 'cc-pvdz', 'H': 'sto-3g'}`    | `PerElement`                    |
/// | `{'default': 'cc-pvdz', 'O': '...'}` | `PerElement` (lookup falls back)|
/// | `{'O': gto.basis.parse(...)}`        | `PerElement` → `Parsed`         |
/// | `{'O': gto.basis.load('cc-pvdz','O')}` | `PerElement` → `Parsed`       |
/// | `'X' (single string)`                | `Name`                          |
/// | NWChem text (raw)                    | `NwchemText`                    |
/// | CP2K-text (`'GTH...'`)               | `Cp2kText`                      |
/// | `gto.basis.load(file, sym)`          | `Parsed`                        |
pub fn format_basis(
    input: &BasisInput,
    atoms: &[ParsedAtom],
) -> Result<HashMap<String, ParsedBasis>, PyscfRsError> {
    // Compute the unique-symbol list in first-occurrence order. Ghost atoms
    // (`H1`, `H2`) collapse to their alphabetic prefix.
    let mut seen: Vec<String> = Vec::new();
    for (s, _) in atoms {
        let alpha: String = s.chars().take_while(|c| c.is_alphabetic()).collect();
        let upper = alpha.to_ascii_uppercase();
        if upper.is_empty() {
            continue;
        }
        if !seen.contains(&upper) {
            seen.push(upper);
        }
    }

    let mut out: HashMap<String, ParsedBasis> = HashMap::new();
    for symbol in &seen {
        let parsed = resolve_for_symbol(input, symbol)?;
        out.insert(symbol.clone(), parsed);
    }
    Ok(out)
}

/// Recursive resolver — `PerElement` arms point at sub-`BasisInput`s.
fn resolve_for_symbol(input: &BasisInput, symbol: &str) -> Result<ParsedBasis, PyscfRsError> {
    match input {
        BasisInput::Name(name) => basis::load_basis(name, symbol).map_err(PyscfRsError::from),
        BasisInput::PerElement(map) => {
            // Lookup priority: exact UPPERCASE → lowercase → "default" key.
            let sub = map
                .get(symbol)
                .or_else(|| map.get(&symbol.to_ascii_lowercase()))
                .or_else(|| {
                    // Try canonicalised forms (e.g., user passed "h" or "H").
                    let mut found = None;
                    for (k, v) in map {
                        if k.eq_ignore_ascii_case(symbol) {
                            found = Some(v);
                            break;
                        }
                    }
                    found
                })
                .or_else(|| map.get("default"))
                .or_else(|| map.get("DEFAULT"))
                .ok_or_else(|| {
                    PyscfRsError::from(BasisLoadError::UnknownName {
                        name: format!("per-element basis missing for symbol '{}'", symbol),
                    })
                })?;
            resolve_for_symbol(sub, symbol)
        }
        BasisInput::NwchemText(text) => basis::parse(text, symbol).map_err(PyscfRsError::from),
        BasisInput::Cp2kText(text) => {
            basis::cp2k::parse_cp2k(text, symbol, "<cp2k-text>").map_err(PyscfRsError::from)
        }
        BasisInput::Parsed(parsed) => Ok(parsed.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_occurrence_order_collapses_repeats() {
        // H, O, H, O — should yield two keys (H + O), in encounter order.
        let atoms = vec![
            ("H".to_string(), [0.0, 0.0, 0.0]),
            ("O".to_string(), [0.0, 0.0, 1.0]),
            ("H".to_string(), [0.0, 0.0, 2.0]),
            ("O".to_string(), [0.0, 0.0, 3.0]),
        ];
        let parsed = format_basis(&BasisInput::Name("sto-3g".into()), &atoms).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(parsed.contains_key("H"));
        assert!(parsed.contains_key("O"));
    }

    #[test]
    fn ghost_atom_suffix_collapses_to_alpha_prefix() {
        // H1 / H2 → both contribute to the H entry.
        let atoms = vec![
            ("H1".to_string(), [0.0, 0.0, 0.0]),
            ("H2".to_string(), [0.0, 0.0, 1.4]),
        ];
        let parsed = format_basis(&BasisInput::Name("sto-3g".into()), &atoms).unwrap();
        assert_eq!(parsed.len(), 1);
        assert!(parsed.contains_key("H"));
    }
}
