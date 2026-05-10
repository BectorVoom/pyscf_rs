//! NWChem ECP-block parser (loading half of GTO-05).
//!
//! Source: `pyscf/gto/basis/parse_nwchem_ecp.py` (Apache-2.0).
//!
//! Format:
//! ```text
//!   ECP
//!   <Symbol> nelec <n_core>
//!   <Symbol> ul              # local channel (l = -1 in upstream convention)
//!   <n_power> <exponent> <coeff>
//!   ...
//!   <Symbol> <l-letter>      # projector channel (l = 0..6)
//!   <n_power> <exponent> <coeff>
//!   ...
//!   END
//! ```
//!
//! Phase 2 ships a flat per-channel representation: each `EcpShell` carries
//! its own `(n_powers, exponents, coeffs)` parallel vectors. Plan 02-07
//! refines this when the ECP-engine trait + `cint1e_ecp` integration land.

use pyscf_core::{EcpLoadError, EcpShell, ParsedEcp};

/// Parse a NWChem-format ECP block for one element symbol.
///
/// Returns `ParsedEcp { n_core, channels }` with `channels[i].l = -1` marking
/// the local (UL) channel and `channels[i].l in 0..=6` marking projector
/// channels (S/P/D/F/G/H/I).
pub fn parse_nwchem_ecp(
    text: &str,
    symbol: &str,
    source: &str,
) -> Result<ParsedEcp, EcpLoadError> {
    let symbol_upper = symbol.to_ascii_uppercase();
    let mut n_core: u32 = 0;
    let mut channels: Vec<EcpShell> = Vec::new();
    let mut current: Option<EcpShell> = None;
    let mut active_for_symbol = false;
    let mut found_symbol = false;
    // Mixed .dat files (e.g. `lanl2dz.dat`) carry both basis blocks
    // AND ECP blocks separated by an `END` line and an `ECP` header.
    // The ECP parser must IGNORE everything before the bare `ECP`
    // line (otherwise basis primitives get misread as ECP rows).
    //
    // If the source text contains an `ECP` marker, gate parsing on it;
    // otherwise treat the entire input as the ECP block (matches the
    // `EcpInput::NwchemEcpText` inline-fixture path).
    let mut in_ecp_section = !text
        .lines()
        .any(|l| l.split('#').next().unwrap_or("").trim().eq_ignore_ascii_case("ECP"));

    for (i, raw_line) in text.lines().enumerate() {
        let line = match raw_line.find('#') {
            Some(j) => &raw_line[..j],
            None => raw_line,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let upper = line.to_ascii_uppercase();
        if upper == "ECP" {
            in_ecp_section = true;
            // Reset any half-built state — the basis section is over.
            current = None;
            active_for_symbol = false;
            continue;
        }
        if !in_ecp_section {
            continue;
        }
        if upper == "END" {
            continue;
        }

        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }

        // Header / element line. First token alphabetic.
        let first_alpha = toks[0]
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic());

        if first_alpha {
            // Element-symbol prefix MUST match `symbol_upper`. Other elements'
            // blocks are skipped.
            let elem = toks[0].to_ascii_uppercase();
            // Strip ghost-atom suffix from element token (Cu1 → Cu).
            let elem_alpha: String = elem.chars().take_while(|c| c.is_alphabetic()).collect();

            if elem_alpha != symbol_upper {
                // Flush + deactivate.
                if let Some(c) = current.take() {
                    channels.push(c);
                }
                active_for_symbol = false;
                continue;
            }
            found_symbol = true;
            active_for_symbol = true;

            if toks.len() < 2 {
                continue; // bare symbol line — nothing to open
            }
            let key = toks[1].to_ascii_uppercase();

            // "Symbol nelec N" — sets core count.
            if key == "NELEC" {
                if toks.len() < 3 {
                    return Err(EcpLoadError::Parse {
                        file: source.into(),
                        line: i + 1,
                        reason: format!("`nelec` line missing N: {}", line),
                    });
                }
                n_core = toks[2].parse().map_err(|_| EcpLoadError::Parse {
                    file: source.into(),
                    line: i + 1,
                    reason: format!("non-numeric n_core: {}", line),
                })?;
                continue;
            }

            // Otherwise: opens a new channel. Flush any prior one.
            if let Some(c) = current.take() {
                channels.push(c);
            }
            let l: i8 = match key.as_str() {
                "UL" => -1,
                "S" => 0,
                "P" => 1,
                "D" => 2,
                "F" => 3,
                "G" => 4,
                "H" => 5,
                "I" => 6,
                other => {
                    return Err(EcpLoadError::Parse {
                        file: source.into(),
                        line: i + 1,
                        reason: format!("unknown ECP channel '{}'", other),
                    });
                }
            };
            current = Some(EcpShell {
                l,
                n_powers: vec![],
                exponents: vec![],
                coeffs: vec![],
            });
        } else {
            // Primitive row: "<n_power> <exponent> <coeff>".
            if !active_for_symbol {
                continue;
            }
            if toks.len() < 3 {
                return Err(EcpLoadError::Parse {
                    file: source.into(),
                    line: i + 1,
                    reason: format!(
                        "ECP primitive row needs ≥ 3 cols (n_power, exp, coef), got {}: {}",
                        toks.len(),
                        line
                    ),
                });
            }
            let np: i32 = toks[0].parse().map_err(|_| EcpLoadError::Parse {
                file: source.into(),
                line: i + 1,
                reason: format!("bad n_power: {}", line),
            })?;
            let exp: f64 = toks[1].parse().map_err(|_| EcpLoadError::Parse {
                file: source.into(),
                line: i + 1,
                reason: format!("bad exp: {}", line),
            })?;
            let coef: f64 = toks[2].parse().map_err(|_| EcpLoadError::Parse {
                file: source.into(),
                line: i + 1,
                reason: format!("bad coef: {}", line),
            })?;
            match &mut current {
                Some(c) => {
                    c.n_powers.push(np);
                    c.exponents.push(exp);
                    c.coeffs.push(coef);
                }
                None => {
                    return Err(EcpLoadError::Parse {
                        file: source.into(),
                        line: i + 1,
                        reason: "primitive row before any channel header".into(),
                    });
                }
            }
        }
    }

    if let Some(c) = current.take() {
        channels.push(c);
    }

    if !found_symbol {
        return Err(EcpLoadError::UnknownName(format!(
            "ECP block for symbol '{}' not found in input",
            symbol
        )));
    }

    Ok(ParsedEcp { n_core, channels })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h_ul_minimal() {
        let text = "ECP\n\
                    H nelec 0\n\
                    H ul\n\
                    2     1.0     0.0\n\
                    END\n";
        let p = parse_nwchem_ecp(text, "H", "test").unwrap();
        assert_eq!(p.n_core, 0);
        assert_eq!(p.channels.len(), 1);
        assert_eq!(p.channels[0].l, -1); // UL
        assert_eq!(p.channels[0].n_powers, vec![2]);
        approx::assert_abs_diff_eq!(p.channels[0].exponents[0], 1.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(p.channels[0].coeffs[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn cu_lanl2dz_n_core_18() {
        // Synthetic — mirrors the structure of upstream lanl2dz.dat for Cu.
        // We only check the framework: n_core extraction + multi-channel
        // accumulation. Real Cu LANL2DZ has 18 core electrons.
        let text = "ECP\n\
                    Cu nelec 18\n\
                    Cu ul\n\
                      2  10.0  1.0\n\
                    Cu s\n\
                      2  20.0  -3.0\n\
                    Cu p\n\
                      2  30.0  -2.0\n\
                    Cu d\n\
                      2  40.0  -1.0\n\
                    END\n";
        let p = parse_nwchem_ecp(text, "Cu", "test").unwrap();
        assert_eq!(p.n_core, 18);
        // 4 channels: UL + S + P + D.
        assert_eq!(p.channels.len(), 4);
        assert_eq!(p.channels[0].l, -1);
        assert_eq!(p.channels[1].l, 0);
        assert_eq!(p.channels[2].l, 1);
        assert_eq!(p.channels[3].l, 2);
    }

    #[test]
    fn skips_other_elements() {
        let text = "ECP\n\
                    Na nelec 2\n\
                    Na ul\n\
                      2  5.0  1.0\n\
                    Cl nelec 10\n\
                    Cl ul\n\
                      2  3.0  1.0\n\
                    END\n";
        // Asking for Cl should give n_core=10 and 1 channel (UL), not Na's data.
        let p = parse_nwchem_ecp(text, "Cl", "test").unwrap();
        assert_eq!(p.n_core, 10);
        assert_eq!(p.channels.len(), 1);
        approx::assert_abs_diff_eq!(p.channels[0].exponents[0], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn missing_symbol_errors() {
        let text = "ECP\n\
                    Cu nelec 18\n\
                    Cu ul\n\
                      2  10.0  1.0\n\
                    END\n";
        let err = parse_nwchem_ecp(text, "H", "test").unwrap_err();
        match err {
            EcpLoadError::UnknownName(s) => {
                assert!(s.contains("H"), "{}", s);
            }
            other => panic!("expected UnknownName, got {:?}", other),
        }
    }

    #[test]
    fn unknown_channel_letter_errors() {
        let text = "ECP\n\
                    H nelec 0\n\
                    H z\n\
                    2  1.0  0.0\n\
                    END\n";
        let err = parse_nwchem_ecp(text, "H", "test").unwrap_err();
        match err {
            EcpLoadError::Parse { reason, .. } => {
                assert!(reason.contains("unknown ECP channel"), "{}", reason);
            }
            other => panic!("expected Parse, got {:?}", other),
        }
    }
}
