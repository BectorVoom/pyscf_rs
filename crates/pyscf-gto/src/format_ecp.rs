//! GTO-05 (loading half): `format_ecp` + `make_ecp_env`.
//!
//! Source: `pyscf/gto/mole.py:1107-1160` (`make_ecp_env`) +
//! `pyscf/gto/basis/__init__.py:730-779` (`load_ecp` / `parse`).
//!
//! D-06 sequencing: loading ships in Phase 2; ECP evaluation closes
//! via gap-closure plan 02-10 once cintx ECP merges. Plan 02-07 is
//! the loading half.

use crate::basis;
use crate::types::EcpInput;
use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF, PTR_COEFF,
    PTR_EXP,
};
use pyscf_core::{BasisLoadError, ParsedAtom, ParsedEcp, PyscfRsError};
use std::collections::HashMap;

/// Resolve `EcpInput` → per-element-symbol `ParsedEcp` map.
///
/// Mirrors the shape of `format_basis::format_basis` (Phase 2 02-03).
/// Per upstream `pyscf/gto/mole.py:1107-1160` semantics, symbols with
/// no ECP entry are simply absent from the returned map (NOT an error).
///
/// Returns:
///   * `Ok(HashMap)` — keys are element symbols UPPER-cased; values
///     are the parsed ECP records.
///   * `Err(...)` — propagated from the underlying parser / file IO.
pub fn format_ecp(
    input: &EcpInput,
    atoms: &[ParsedAtom],
) -> Result<HashMap<String, ParsedEcp>, PyscfRsError> {
    if matches!(input, EcpInput::None) {
        return Ok(HashMap::new());
    }

    // First-occurrence-symbol pattern (matches `format_basis` Pitfall 4
    // alignment). Strip non-alpha trailing characters (e.g. ghost-atom
    // `Cu1` → `CU`).
    let mut seen: Vec<String> = Vec::new();
    for (raw_symbol, _) in atoms {
        let alpha: String = raw_symbol.chars().take_while(|c| c.is_alphabetic()).collect();
        let upper = alpha.to_ascii_uppercase();
        if !upper.is_empty() && !seen.contains(&upper) {
            seen.push(upper);
        }
    }

    let mut out: HashMap<String, ParsedEcp> = HashMap::new();
    for symbol in &seen {
        if let Some(parsed) = resolve_for_symbol(input, symbol)? {
            out.insert(symbol.clone(), parsed);
        }
    }
    Ok(out)
}

/// Inner dispatch: one element-symbol at a time.
fn resolve_for_symbol(
    input: &EcpInput,
    symbol: &str,
) -> Result<Option<ParsedEcp>, PyscfRsError> {
    match input {
        EcpInput::None => Ok(None),
        EcpInput::Name(name) => {
            // ECP files in `pyscf/gto/basis/` (e.g. `lanl2dz.dat`) carry
            // BOTH basis blocks AND ECP blocks. Resolve via the same
            // ALIAS table as basis loading; read the file; pass to
            // `parse_nwchem_ecp`.
            let canonical = basis::canonicalise_basis_name(name);
            let filename = match basis::alias::lookup(&canonical) {
                Some(f) => f,
                None => {
                    return Err(PyscfRsError::from(BasisLoadError::UnknownName {
                        name: format!("ECP basis '{}' not in ALIAS", canonical),
                    }));
                }
            };
            let dir = basis::path::basis_dir().map_err(PyscfRsError::from)?;
            let full = dir.join(filename);
            let text = std::fs::read_to_string(&full).map_err(|e| {
                PyscfRsError::from(BasisLoadError::Io {
                    path: full.display().to_string(),
                    source: e,
                })
            })?;
            // No ECP section in the file → no ECP for this symbol.
            // (This is normal for non-ECP basis files like sto-3g.dat.)
            if !text.contains("ECP") {
                return Ok(None);
            }
            // Parse the ECP block; absence of THIS symbol's block is
            // not an error (e.g. asking for H ECP from lanl2dz.dat).
            match basis::nwchem_ecp::parse_nwchem_ecp(
                &text,
                symbol,
                &full.display().to_string(),
            ) {
                Ok(p) => Ok(Some(p)),
                Err(pyscf_core::EcpLoadError::UnknownName(_)) => Ok(None),
                Err(other) => Err(PyscfRsError::from(other)),
            }
        }
        EcpInput::PerElement(map) => {
            let sub = map
                .get(symbol)
                .or_else(|| map.get(&symbol.to_ascii_lowercase()))
                .or_else(|| {
                    let titlecase = title_case(symbol);
                    map.get(&titlecase).or_else(|| map.get("default"))
                });
            match sub {
                Some(s) => resolve_for_symbol(s, symbol),
                None => Ok(None), // no ECP for this symbol; not an error
            }
        }
        EcpInput::NwchemEcpText(text) => {
            match basis::nwchem_ecp::parse_nwchem_ecp(
                text,
                symbol,
                "<inline-ecp-text>",
            ) {
                Ok(p) if p.channels.is_empty() => Ok(None),
                Ok(p) => Ok(Some(p)),
                Err(pyscf_core::EcpLoadError::UnknownName(_)) => Ok(None),
                Err(other) => Err(PyscfRsError::from(other)),
            }
        }
        EcpInput::Parsed(p) => {
            if p.channels.is_empty() {
                Ok(None)
            } else {
                Ok(Some(p.clone()))
            }
        }
    }
}

/// `Cu` → `Cu`. Lower-cased input → title-cased.
fn title_case(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => {
            let mut out = String::new();
            out.extend(first.to_uppercase());
            out.push_str(&chars.as_str().to_ascii_lowercase());
            out
        }
        None => String::new(),
    }
}

/// `make_ecp_env` — port of `pyscf/gto/mole.py:1107-1160`.
///
/// Mutates `_atm` (CHARGE_OF subtraction per Pitfall 2) and `_env`
/// (appends per-channel exponents + coefficients), and returns the
/// freshly built `_ecpbas` row vector.
///
/// `_ecpbas` row layout (`BAS_SLOTS = 8` i32 entries each):
///
/// | Slot index   | Meaning                            |
/// |--------------|------------------------------------|
/// | `ATOM_OF=0`  | atom index this channel sits on    |
/// | `ANG_OF=1`   | angular momentum `l` (`-1` for UL) |
/// | `NPRIM_OF=2` | number of primitives in this row   |
/// | `NCTR_OF=3`  | radial-power order `r_order`       |
/// | `KAPPA_OF=4` | spin-orbit flag (always `0` v1)    |
/// | `PTR_EXP=5`  | env offset of exponents block      |
/// | `PTR_COEFF=6`| env offset of coefficients block   |
/// | _slot 7_     | reserved (always `0`)              |
///
/// Pitfall 2: CHARGE_OF is decremented EXACTLY ONCE per ECP atom in
/// this routine. Caller MUST NOT subtract again from the same atom.
pub fn make_ecp_env(
    atoms: &[ParsedAtom],
    ecp: &HashMap<String, ParsedEcp>,
    _atm: &mut [i32],
    _env: &mut Vec<f64>,
) -> Vec<i32> {
    let mut _ecpbas: Vec<i32> = Vec::new();

    if ecp.is_empty() {
        return _ecpbas;
    }

    for (atom_id, (sym_with_suffix, _xyz)) in atoms.iter().enumerate() {
        let alpha: String = sym_with_suffix
            .chars()
            .take_while(|c| c.is_alphabetic())
            .collect();
        let upper = alpha.to_ascii_uppercase();

        let parsed = match ecp.get(&upper) {
            Some(p) => p,
            None => continue,
        };

        // Pitfall 2: subtract n_core once.
        let row_start = atom_id * ATM_SLOTS;
        debug_assert!(
            row_start + CHARGE_OF < _atm.len(),
            "_atm too small for atom_id={atom_id}",
        );
        _atm[row_start + CHARGE_OF] -= parsed.n_core as i32;
        // Note: NUC_MOD_OF stays at POINT_NUC=1 for v1 (cintx-compat
        // does not export NUC_ECP=4 — the upstream PySCF marker for ECP
        // atoms in the nuclear-model slot — so the Phase 2 stub leaves
        // NUC_MOD_OF unchanged and signals ECP via `_ecpbas` + the
        // `int1e_ecp*` intor name routing in `intor.rs`).

        // Group primitives by `n_power` (== rorder + 2 in upstream's
        // convention, but we just key on the raw n_power value). One
        // _ecpbas row per (channel, distinct n_power) so the cintx-side
        // Type-1/Type-2 evaluator (gap-closure plan 02-10) gets clean
        // per-r_order blocks.
        for ch in &parsed.channels {
            // Distinct n_power values, preserving first-encounter
            // order.
            let mut seen_orders: Vec<i32> = Vec::new();
            for &np in &ch.n_powers {
                if !seen_orders.contains(&np) {
                    seen_orders.push(np);
                }
            }
            for &order in &seen_orders {
                // Collect primitives matching this n_power.
                let mut exps_buf: Vec<f64> = Vec::new();
                let mut coefs_buf: Vec<f64> = Vec::new();
                for ((np, exp), coef) in ch
                    .n_powers
                    .iter()
                    .zip(ch.exponents.iter())
                    .zip(ch.coeffs.iter())
                {
                    if *np == order {
                        exps_buf.push(*exp);
                        coefs_buf.push(*coef);
                    }
                }
                if exps_buf.is_empty() {
                    continue;
                }

                let ptr_exp = _env.len();
                _env.extend_from_slice(&exps_buf);
                let ptr_coeff = _env.len();
                _env.extend_from_slice(&coefs_buf);

                let mut row = [0_i32; BAS_SLOTS];
                row[ATOM_OF] = atom_id as i32;
                row[ANG_OF] = ch.l as i32;
                row[NPRIM_OF] = exps_buf.len() as i32;
                // Upstream stores `rorder` here (the radial-power
                // index from `for rorder, bi in enumerate(lb[1])`); we
                // store the raw n_power value which is what the cintx
                // Type-1/2 evaluator will read. Phase 2 D-06 promises
                // gap-closure plan 02-10 may rewrite the row layout
                // when cintx ECP lands.
                row[NCTR_OF] = order;
                row[KAPPA_OF] = 0;
                row[PTR_EXP] = ptr_exp as i32;
                row[PTR_COEFF] = ptr_coeff as i32;
                _ecpbas.extend_from_slice(&row);
            }
        }
    }

    debug_assert_eq!(
        _ecpbas.len() % BAS_SLOTS,
        0,
        "Pitfall 2: _ecpbas rows must be multiples of BAS_SLOTS",
    );
    _ecpbas
}
