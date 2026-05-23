//! NWChem / Gaussian-94 basis-set parser.
//!
//! Source: `pyscf/gto/basis/parse_nwchem.py` (Apache-2.0).
//!
//! Algorithm — line-stream parser; `BASIS` / `END` headers optional;
//! blank lines + `#` comments tolerated. A header line opens a shell block
//! by combining an element symbol with one angular-momentum letter token:
//!
//! ```text
//! H    S            # one shell, l=0
//! Li   SP           # SHARED-EXPONENT Pople form: TWO shells (l=0 + l=1)
//! C    D            # one shell, l=2
//! ```
//!
//! Important: upstream uses `SP` as a SINGLE token (not `S P`). Per
//! `pyscf/gto/basis/parse_nwchem.py:_parse`, only the second whitespace-
//! separated token of the header line is the angular-momentum key:
//! `keys = dat.split(); key = keys[1].upper()`. The special string
//! `"SP"` triggers the shared-exponent path which produces an `s`-shell
//! AND a `p`-shell sharing the same exponent vector.
//!
//! This parser ports that behaviour exactly so it agrees with the
//! `pyscf/gto/basis/sto-3g.dat`, `6-31G.dat`, etc. files shipped under
//! `pyscf/gto/basis/`.

use pyscf_core::{BasisLoadError, ParsedBasis, ShellSpec};

/// Parse NWChem / Gaussian-94 text for one element symbol.
/// Only shells matching `symbol` (case-insensitive) are kept. Shells for
/// other elements are skipped silently.
pub fn parse_nwchem(text: &str, symbol: &str, source: &str) -> Result<ParsedBasis, BasisLoadError> {
    let symbol_upper = symbol.to_ascii_uppercase();

    // Per-l accumulator: each entry is one ShellSpec for that angular momentum.
    // Final ParsedBasis preserves the order the shells were encountered in the
    // input text (matching upstream `basis_sorted = [b for bs in ... for b in bs]`).
    let mut shells: Vec<ShellSpec> = Vec::new();

    let mut current = CurrentShell::None;
    let mut active_for_symbol = false; // whether the current header opened a block we keep

    let lines: Vec<&str> = text.lines().collect();
    for (i, raw_line) in lines.iter().enumerate() {
        // Strip "# comment" suffix.
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        // Skip all-uppercase BASIS / END / SET headers. A bare `ECP`
        // line marks the end of basis-set data in mixed files like
        // `lanl2dz.dat` (which carries both basis blocks AND ECP blocks
        // in the same .dat). Plan 02-07 (GTO-05 loading half) needs the
        // basis parser to stop at the ECP boundary; the ECP parser
        // (`nwchem_ecp::parse_nwchem_ecp`) handles the ECP section.
        let upper = line.to_ascii_uppercase();
        if upper.starts_with("BASIS") || upper.starts_with("END") {
            continue;
        }
        if upper == "ECP" {
            // Flush any pending basis shell, then stop parsing.
            break;
        }

        // Tokenise.
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        // Decide if the line is a header (alphabetic first char) or a primitive
        // row (numeric first char).
        let first_alpha = tokens[0].chars().next().is_some_and(|c| c.is_alphabetic());

        if first_alpha {
            // Flush any pending shell before starting a new one.
            flush_current(&mut current, &mut shells);

            // Parse the header. Per upstream:
            //   keys = dat.split()
            //   key  = keys[1].upper()  if len(keys) > 1 else keys[0].upper()
            //   if key == 'SP': shared-exponent path
            //   elif key in MAPSPDF (S/P/D/F/G/H/I/K): single-l path
            // The element symbol is keys[0] when len(keys) > 1; otherwise the
            // line is just an element token with no angular-momentum key (rare).
            let elem_token = tokens[0];
            let elem_alpha: String = elem_token
                .chars()
                .take_while(|c| c.is_alphabetic())
                .collect();
            let elem = elem_alpha.to_ascii_uppercase();
            let key = if tokens.len() >= 2 {
                tokens[1].to_ascii_uppercase()
            } else {
                // Just an element-only line; nothing to open. Set inactive and
                // continue — primitives following such a header are an error,
                // but upstream's _parse raises BasisNotFoundError too.
                active_for_symbol = false;
                continue;
            };

            // Match-element-symbol check. Upstream `_search_basis_block` filters
            // by symbol BEFORE calling `_parse`; we fold the filter inline.
            active_for_symbol = elem == symbol_upper;
            if !active_for_symbol {
                current = CurrentShell::None;
                continue;
            }

            // Open the shell.
            if key == "SP" {
                current = CurrentShell::SharedSP {
                    exps: Vec::new(),
                    s_coeffs: Vec::new(),
                    p_coeffs: Vec::new(),
                };
            } else if let Some(l) = ang_mom_from_letter(&key) {
                current = CurrentShell::Single {
                    l,
                    exps: Vec::new(),
                    coeffs: Vec::new(),
                };
            } else {
                return Err(BasisLoadError::Parse {
                    file: source.into(),
                    line: i + 1,
                    reason: format!(
                        "unknown angular-momentum key '{}' in header '{}'",
                        key, line
                    ),
                });
            }
        } else {
            // Primitive row. Parse numbers (NWChem allows Fortran-style 'D'
            // exponent: '1.23D-04' → '1.23e-04'). Upstream does
            // `dat.replace('D','e').split()`.
            if !active_for_symbol {
                continue; // skip primitives for other elements' blocks
            }
            let normalised: String = line.replace(['D', 'd'], "e");
            let cols: Result<Vec<f64>, _> = normalised
                .split_whitespace()
                .map(|s| s.parse::<f64>())
                .collect();
            let cols = cols.map_err(|_| BasisLoadError::Parse {
                file: source.into(),
                line: i + 1,
                reason: format!("non-numeric primitive row: {}", line),
            })?;

            match &mut current {
                CurrentShell::None => {
                    return Err(BasisLoadError::Parse {
                        file: source.into(),
                        line: i + 1,
                        reason: format!("primitive row before any header: {}", line),
                    });
                }
                CurrentShell::Single { exps, coeffs, .. } => {
                    if cols.len() < 2 {
                        return Err(BasisLoadError::Parse {
                            file: source.into(),
                            line: i + 1,
                            reason: format!(
                                "single-shell primitive row needs ≥ 2 columns, got {}: {}",
                                cols.len(),
                                line
                            ),
                        });
                    }
                    exps.push(cols[0]);
                    coeffs.push(cols[1]);
                }
                CurrentShell::SharedSP {
                    exps,
                    s_coeffs,
                    p_coeffs,
                } => {
                    if cols.len() < 3 {
                        return Err(BasisLoadError::Parse {
                            file: source.into(),
                            line: i + 1,
                            reason: format!(
                                "SP-shared primitive row needs ≥ 3 columns (exp + s_coeff + \
                                 p_coeff), got {}: {}",
                                cols.len(),
                                line
                            ),
                        });
                    }
                    exps.push(cols[0]);
                    s_coeffs.push(cols[1]);
                    p_coeffs.push(cols[2]);
                }
            }
        }
    }

    // Flush trailing shell.
    flush_current(&mut current, &mut shells);

    Ok(ParsedBasis { shells })
}

/// Helper — push the in-flight shell onto `shells`, leaving `current = None`.
fn flush_current(current: &mut CurrentShell, shells: &mut Vec<ShellSpec>) {
    let taken = std::mem::replace(current, CurrentShell::None);
    match taken {
        CurrentShell::None => {}
        CurrentShell::Single { l, exps, coeffs } => {
            if !exps.is_empty() {
                shells.push(ShellSpec {
                    l,
                    exponents: exps,
                    coeffs: vec![coeffs],
                });
            }
        }
        CurrentShell::SharedSP {
            exps,
            s_coeffs,
            p_coeffs,
        } => {
            if !exps.is_empty() {
                // Upstream emits the s-shell before the p-shell.
                shells.push(ShellSpec {
                    l: 0,
                    exponents: exps.clone(),
                    coeffs: vec![s_coeffs],
                });
                shells.push(ShellSpec {
                    l: 1,
                    exponents: exps,
                    coeffs: vec![p_coeffs],
                });
            }
        }
    }
}

/// Local enum used by `parse_nwchem`'s state machine. Re-declared at module
/// scope for `flush_current` consumption.
enum CurrentShell {
    None,
    Single {
        l: u8,
        exps: Vec<f64>,
        coeffs: Vec<f64>,
    },
    SharedSP {
        exps: Vec<f64>,
        s_coeffs: Vec<f64>,
        p_coeffs: Vec<f64>,
    },
}

/// Strip "# comment" suffix from a line.
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Map an angular-momentum letter to `l`. SPDFGHIK supported (matches upstream
/// `MAPSPDF` and pyscf/data/elements.py constants).
fn ang_mom_from_letter(letter: &str) -> Option<u8> {
    match letter.to_ascii_uppercase().as_str() {
        "S" => Some(0),
        "P" => Some(1),
        "D" => Some(2),
        "F" => Some(3),
        "G" => Some(4),
        "H" => Some(5),
        "I" => Some(6),
        "K" => Some(7),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1 — H STO-3G primitives match upstream-quoted values.
    #[test]
    fn h_sto3g_primitives() {
        let text = "BASIS \"ao basis\" PRINT\n\
                    #BASIS SET: (3s) -> [1s]\n\
                    H    S\n\
                          3.42525091             0.15432897\n\
                          0.62391373             0.53532814\n\
                          0.16885540             0.44463454\n\
                    END\n";
        let p = parse_nwchem(text, "H", "test").unwrap();
        assert_eq!(p.shells.len(), 1);
        assert_eq!(p.shells[0].l, 0);
        assert_eq!(p.shells[0].exponents.len(), 3);
        approx::assert_abs_diff_eq!(p.shells[0].exponents[0], 3.42525091, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[0].exponents[1], 0.62391373, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[0].exponents[2], 0.16885540, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[0].coeffs[0][0], 0.15432897, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[0].coeffs[0][2], 0.44463454, epsilon = 1e-9);
    }

    /// Test 3 — SP shared-exponent (Pople-canonical SINGLE-token form).
    /// Upstream produces TWO shells (l=0 and l=1) sharing one exponent vector.
    #[test]
    fn li_sp_shared_exponent_creates_two_shells() {
        let text = "Li    S\n\
                    16.1195750  0.15432897\n\
                     2.9362007  0.53532814\n\
                     0.7946505  0.44463454\n\
                    Li    SP\n\
                    0.6362897  -0.09996723   0.15591627\n\
                    0.1478601   0.39951283   0.60768372\n\
                    0.0480887   0.70011547   0.39195739\n\
                    END\n";
        let p = parse_nwchem(text, "Li", "test").unwrap();
        // Three shells: 1 plain S + (S, P) shared-exponent pair.
        assert_eq!(p.shells.len(), 3);
        assert_eq!(p.shells[0].l, 0); // plain S
        assert_eq!(p.shells[1].l, 0); // SP -> S
        assert_eq!(p.shells[2].l, 1); // SP -> P
        // The two SP-derived shells share exponents.
        assert_eq!(p.shells[1].exponents, p.shells[2].exponents);
        assert_eq!(p.shells[1].exponents.len(), 3);
        approx::assert_abs_diff_eq!(p.shells[1].exponents[0], 0.6362897, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[1].coeffs[0][0], -0.09996723, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[2].coeffs[0][0], 0.15591627, epsilon = 1e-9);
    }

    /// Test 4 — comments and blank lines tolerated.
    #[test]
    fn handles_comments_and_blank_lines() {
        let text = "# header comment\n\
                    BASIS\n\
                    \n\
                    H S  # inline comment after header\n\
                         1.0  1.0\n\
                    \n\
                    END\n";
        let p = parse_nwchem(text, "H", "test").unwrap();
        assert_eq!(p.shells.len(), 1);
        assert_eq!(p.shells[0].l, 0);
        approx::assert_abs_diff_eq!(p.shells[0].exponents[0], 1.0, epsilon = 1e-12);
    }

    /// Other elements' blocks must be skipped (so `parse_nwchem(text, "H", ...)`
    /// against a multi-element file returns only H's shells).
    #[test]
    fn skips_unwanted_element_blocks() {
        let text = "C    S\n\
                    100.0  1.0\n\
                    H    S\n\
                    1.0    1.0\n\
                    O    S\n\
                    50.0   1.0\n\
                    END\n";
        let p = parse_nwchem(text, "H", "test").unwrap();
        assert_eq!(p.shells.len(), 1);
        approx::assert_abs_diff_eq!(p.shells[0].exponents[0], 1.0, epsilon = 1e-12);
    }

    /// `D` Fortran-exponent normalisation: `1.5D-2` → `1.5e-2`.
    #[test]
    fn fortran_d_exponent_parses() {
        let text = "H    S\n\
                    1.5D-2   1.0D0\n\
                    END\n";
        let p = parse_nwchem(text, "H", "test").unwrap();
        approx::assert_abs_diff_eq!(p.shells[0].exponents[0], 0.015, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(p.shells[0].coeffs[0][0], 1.0, epsilon = 1e-12);
    }

    /// Multi-shell element produces multiple ShellSpec entries in order.
    #[test]
    fn multi_shell_element_order_preserved() {
        let text = "C    S\n\
                    100.0  1.0\n\
                    C    S\n\
                    50.0   1.0\n\
                    C    P\n\
                    20.0   1.0\n\
                    END\n";
        let p = parse_nwchem(text, "C", "test").unwrap();
        assert_eq!(p.shells.len(), 3);
        assert_eq!(p.shells[0].l, 0);
        assert_eq!(p.shells[1].l, 0);
        assert_eq!(p.shells[2].l, 1);
    }

    /// Unknown angular-momentum letter → structured Parse error.
    #[test]
    fn unknown_letter_errors() {
        let text = "H    Z\n\
                    1.0  1.0\n\
                    END\n";
        let err = parse_nwchem(text, "H", "test").unwrap_err();
        match err {
            BasisLoadError::Parse { reason, .. } => {
                assert!(reason.contains("unknown angular-momentum"), "{}", reason);
            }
            other => panic!("expected Parse error, got {:?}", other),
        }
    }
}
