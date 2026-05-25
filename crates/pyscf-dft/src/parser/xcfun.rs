//! xcfun-side XC-string parser — the **alternate** resolver.
//!
//! Direct port of `pyscf/dft/xcfun.py:parse_xc` (lines 416-569), with the
//! xcfun `XC_CODES` (xcfun.py:94-228) and `XC_ALIAS` (xcfun.py:240-291) tables
//! transcribed as inline `const` slices (NO codegen / NO build.rs). Helper
//! ports: `format_xc_code` + `remove_dup` (`pyscf/dft/xc/utils.py`),
//! `_NAME_WITH_DASH` (xcfun.py:571-577).
//!
//! License: Apache-2.0 (matching upstream PySCF).
//! Requirement: DFT-02 (parser parity), alternate-resolver half (D-01).
//!
//! The integer IDs below are xcfun functional numbers (0..77) and map 1:1 to
//! `xcfun_rs`'s `FunctionalId` discriminants (verified against
//! `xcfun-core/src/functional_id.rs`: SLATERX=0, VWN5C=3, PBEC=4, PBEX=5,
//! BECKEX=6, LYPC=16, P86C=56). The `XcBackend::Xcfun` dispatch maps each id
//! back to its xcfun name for `Functional::set`.
//!
//! Two upstream divergences from the libxc parser (preserved here):
//!   1. xcfun's `parse_token` tries `key + suffix` (X/C/XC) when the bare key
//!      misses (xcfun.py:534-535) — e.g. `PBE` in the X part → `PBEX`.
//!   2. After the loop, if no omega was assigned, LR_HF (`hyb[1]`) is zeroed
//!      (xcfun.py:567-568).
//!
//! Security (T-04-05a/b): pure string→spec mapping; depth-bounded expansion.

use crate::error::DftError;
use crate::parser::{Component, MAX_EXPANSION_DEPTH, XcSpec, remove_dup};

#[derive(Clone, Copy)]
enum Resolved {
    Id(u32),
    Expand(&'static str),
}

/// xcfun `XC_CODES` (xcfun.py:94-228). UPPERCASE keys; only the
/// parity-relevant subset is transcribed.
const XC_CODES: &[(&str, Resolved)] = &[
    // Primitive xcfun functional IDs.
    ("SLATERX", Resolved::Id(0)),
    ("SLATER", Resolved::Id(0)),
    ("LDA", Resolved::Id(0)),
    ("VWN3C", Resolved::Id(2)),
    ("VWN3", Resolved::Id(2)),
    ("VWN5C", Resolved::Id(3)),
    ("VWN", Resolved::Id(3)),
    ("VWN5", Resolved::Id(3)),
    ("PBEC", Resolved::Id(4)),
    ("PBEX", Resolved::Id(5)),
    ("BECKEX", Resolved::Id(6)),
    ("B88", Resolved::Id(6)),
    ("LYPC", Resolved::Id(16)),
    ("LYP", Resolved::Id(16)),
    ("OPTX", Resolved::Id(17)),
    ("REVPBEX", Resolved::Id(19)),
    ("RPBEX", Resolved::Id(20)),
    ("PW91X", Resolved::Id(26)),
    ("PW92C", Resolved::Id(28)),
    ("P86C", Resolved::Id(56)),
    ("P86", Resolved::Id(56)),
    ("PW91C", Resolved::Id(77)),
    // Compound expansions (xcfun.py:181-227).
    ("SVWN", Resolved::Expand("SLATERX + VWN5")),
    ("BLYP", Resolved::Expand("B88 + LYP")),
    ("BP86", Resolved::Expand("B88 + P86")),
    ("PBE0", Resolved::Expand(".25*HF + .75*PBEX + PBEC")),
    ("PBE1PBE", Resolved::Expand("PBE0")),
    ("B3LYP", Resolved::Expand("B3LYPG")),
    (
        "B3LYPG",
        Resolved::Expand(".2*HF + .08*SLATER + .72*B88 + .81*LYP + .19*VWN3C"),
    ),
    (
        "B3LYP5",
        Resolved::Expand(".2*HF + .08*SLATER + .72*B88 + .81*LYP + .19*VWN5C"),
    ),
];

/// xcfun `XC_ALIAS` (xcfun.py:240-291). UPPERCASE keys.
const XC_ALIAS: &[(&str, &str)] = &[
    ("BLYP", "B88,LYP"),
    ("BP86", "B88,P86"),
    ("PBE", "PBE,PBE"),
    ("REVPBE", "REVPBE,PBE"),
    ("OLYP", "OPTX,LYP"),
    ("RPBE", "RPBE,PBE"),
    ("BPBE", "B88,PBE"),
    ("HFLYP", "HF,LYP"),
    ("SVWN", "SLATER,VWN"),
];

/// xcfun `_NAME_WITH_DASH` (xcfun.py:571-577).
const NAME_WITH_DASH: &[(&str, &str)] = &[
    ("SR-HF", "SR_HF"),
    ("LR-HF", "LR_HF"),
    ("M06-L", "M06L"),
    ("E-", "E_"),
];

struct ParseState {
    hyb: [f64; 3],
    fn_facs: Vec<Component>,
}

/// `format_xc_code` port (`pyscf/dft/xc/utils.py:51`) — shared logic with the
/// libxc parser.
fn format_xc_code(description: &str) -> String {
    let cleaned: String = description
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_uppercase();

    if !cleaned.contains("RSH") {
        return cleaned;
    }
    let frags: Vec<&str> = cleaned.split("RSH").collect();
    let mut out = String::from(frags[0]);
    for frag in &frags[1..] {
        out.push_str("RSH");
        if let Some(close) = frag.find(')') {
            let rsh_key = &frag[..close];
            let rest = &frag[close + 1..];
            let inner = &rsh_key[1.min(rsh_key.len())..];
            if inner.contains(',') {
                let parts: Vec<&str> = inner.split(',').collect();
                if parts.len() == 3 {
                    out.push('(');
                    out.push_str(parts[1]);
                    out.push(';');
                    out.push_str(parts[2]);
                    out.push(';');
                    out.push_str(parts[0]);
                    out.push(')');
                    out.push_str(rest);
                    continue;
                }
            }
            out.push_str(rsh_key);
            out.push(')');
            out.push_str(rest);
        } else {
            out.push_str(frag);
        }
    }
    out
}

fn assign_omega(
    state: &mut ParseState,
    xc: &str,
    omega: f64,
    hyb_or_sr: f64,
    lr: f64,
) -> Result<(), DftError> {
    if state.hyb[2] == omega || omega == 0.0 {
        state.hyb[0] += hyb_or_sr;
        state.hyb[1] += lr;
    } else if state.hyb[2] == 0.0 {
        state.hyb[0] += hyb_or_sr;
        state.hyb[1] += lr;
        state.hyb[2] = omega;
    } else {
        return Err(DftError::ConflictingOmega {
            xc: xc.to_string(),
            existing: state.hyb[2],
            requested: omega,
        });
    }
    Ok(())
}

fn xc_codes_get(key: &str) -> Option<Resolved> {
    XC_CODES.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn xc_alias_get(key: &str) -> Option<&'static str> {
    XC_ALIAS.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn parse_factor(xc: &str, token: &str, raw: &str) -> Result<f64, DftError> {
    raw.replace("E_", "E-")
        .parse::<f64>()
        .map_err(|e| DftError::MalformedToken {
            xc: xc.to_string(),
            token: token.to_string(),
            reason: format!("invalid factor '{raw}': {e}"),
        })
}

fn parse_paren_omega(xc: &str, token: &str, key: &str) -> Result<Option<f64>, DftError> {
    let Some(open) = key.find('(') else {
        return Ok(None);
    };
    let Some(close) = key[open..].find(')') else {
        return Err(DftError::MalformedToken {
            xc: xc.to_string(),
            token: token.to_string(),
            reason: "missing ')' in SR_HF/LR_HF term".into(),
        });
    };
    let inner = &key[open + 1..open + close];
    inner
        .parse::<f64>()
        .map(Some)
        .map_err(|e| DftError::MalformedToken {
            xc: xc.to_string(),
            token: token.to_string(),
            reason: format!("invalid omega '{inner}': {e}"),
        })
}

/// `parse_token` port (xcfun.py:492-547). `suffix` is the X/C/XC fallback
/// suffix tried after a bare-key miss (xcfun.py:534).
#[allow(clippy::too_many_lines)]
fn parse_token(
    state: &mut ParseState,
    xc: &str,
    token: &str,
    suffix: &str,
    search_xc_alias: bool,
    depth: usize,
) -> Result<(), DftError> {
    if token.is_empty() {
        return Ok(());
    }
    if depth > MAX_EXPANSION_DEPTH {
        return Err(DftError::ExpansionDepthExceeded {
            xc: xc.to_string(),
            limit: MAX_EXPANSION_DEPTH,
        });
    }

    let (sign, token) = if let Some(stripped) = token.strip_prefix('-') {
        (-1.0_f64, stripped)
    } else {
        (1.0_f64, token)
    };

    let (fac, key): (f64, &str) = if let Some(star) = token.find('*') {
        let (a, b) = (&token[..star], &token[star + 1..]);
        let (fac_str, key) = if a.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
            (b, a)
        } else {
            (a, b)
        };
        (sign * parse_factor(xc, token, fac_str)?, key)
    } else {
        (sign, token)
    };

    // RSH(alpha;beta;omega) (xcfun.py:507-511).
    if key.len() >= 3 && &key[..3] == "RSH" {
        if key.len() < 5 || !key.ends_with(')') {
            return Err(DftError::MalformedToken {
                xc: xc.to_string(),
                token: token.to_string(),
                reason: "malformed RSH(...) term".into(),
            });
        }
        let inner = &key[4..key.len() - 1];
        let nums: Result<Vec<f64>, _> = inner
            .split(';')
            .map(|x| {
                x.parse::<f64>().map_err(|e| DftError::MalformedToken {
                    xc: xc.to_string(),
                    token: token.to_string(),
                    reason: format!("invalid RSH parameter '{x}': {e}"),
                })
            })
            .collect();
        let nums = nums?;
        if nums.len() != 3 {
            return Err(DftError::MalformedToken {
                xc: xc.to_string(),
                token: token.to_string(),
                reason: "RSH expects exactly three (alpha;beta;omega) parameters".into(),
            });
        }
        let (alpha, beta, omega) = (nums[0], nums[1], nums[2]);
        return assign_omega(state, xc, omega, fac * (alpha + beta), fac * alpha);
    }

    // HF (xcfun.py:512-514).
    if key == "HF" {
        state.hyb[0] += fac;
        state.hyb[1] += fac;
        return Ok(());
    }

    // SR_HF / SRHF (xcfun.py:515-520).
    if key.contains("SR_HF") || key.contains("SRHF") {
        if let Some(omega) = parse_paren_omega(xc, token, key)? {
            return assign_omega(state, xc, omega, fac, 0.0);
        }
        state.hyb[0] += fac;
        return Ok(());
    }
    // LR_HF (xcfun.py:521-526).
    if key.contains("LR_HF") {
        if let Some(omega) = parse_paren_omega(xc, token, key)? {
            return assign_omega(state, xc, omega, 0.0, fac);
        }
        state.hyb[1] += fac;
        return Ok(());
    }

    // Raw integer ID (xcfun.py:527-528).
    if !key.is_empty() && key.chars().all(|c| c.is_ascii_digit()) {
        let id: u32 = key.parse().map_err(|e| DftError::MalformedToken {
            xc: xc.to_string(),
            token: token.to_string(),
            reason: format!("invalid integer functional ID '{key}': {e}"),
        })?;
        state.fn_facs.push((id, fac));
        return Ok(());
    }

    // Name resolution (xcfun.py:529-547): XC_ALIAS → XC_CODES[key] →
    // XC_CODES[key+suffix] → KeyError.
    let resolved = if search_xc_alias {
        if let Some(s) = xc_alias_get(key) {
            Some(Resolved::Expand(s))
        } else {
            xc_codes_get(key)
        }
    } else {
        xc_codes_get(key)
    };
    let resolved = match resolved {
        Some(r) => Some(r),
        None => xc_codes_get(&format!("{key}{suffix}")),
    };

    let resolved = resolved.ok_or_else(|| DftError::UnknownFunctional {
        xc: xc.to_string(),
        token: token.to_string(),
    })?;

    match resolved {
        Resolved::Id(id) => {
            state.fn_facs.push((id, fac));
            Ok(())
        }
        Resolved::Expand(s) => {
            let sub = parse_xc_inner(s, depth + 1)?;
            let (sub_hyb, sub_alpha, sub_omega) = sub.hyb;
            if sub_hyb != 0.0 || sub_alpha != 0.0 {
                assign_omega(state, xc, sub_omega, sub_hyb * fac, sub_alpha * fac)?;
            }
            for (xid, c) in sub.components {
                state.fn_facs.push((xid, c * fac));
            }
            Ok(())
        }
    }
}

fn split_tokens(code: &str) -> Vec<String> {
    code.replace('-', "+-")
        .replace(";+", ";")
        .split('+')
        .map(str::to_string)
        .collect()
}

fn parse_xc_inner(description: &str, depth: usize) -> Result<XcSpec, DftError> {
    if depth > MAX_EXPANSION_DEPTH {
        return Err(DftError::ExpansionDepthExceeded {
            xc: description.to_string(),
            limit: MAX_EXPANSION_DEPTH,
        });
    }

    let mut description = format_xc_code(description);
    if description.contains('-') {
        for (k, v) in NAME_WITH_DASH {
            if description.contains(k) {
                description = description.replace(k, v);
            }
        }
    }

    let mut state = ParseState {
        hyb: [0.0; 3],
        fn_facs: Vec::new(),
    };

    let comma_count = description.matches(',').count();
    if comma_count > 1 {
        return Err(DftError::TooManyCommas {
            xc: description.clone(),
        });
    }

    if comma_count == 1 {
        let mut split = description.splitn(2, ',');
        let x_code = split.next().unwrap_or("");
        let c_code = split.next().unwrap_or("");
        for token in split_tokens(x_code) {
            parse_token(&mut state, &description, &token, "X", false, depth)?;
        }
        for token in split_tokens(c_code) {
            parse_token(&mut state, &description, &token, "C", false, depth)?;
        }
    } else {
        for token in split_tokens(&description) {
            parse_token(&mut state, &description, &token, "XC", true, depth)?;
        }
    }

    // xcfun.py:567-568 — if no omega assigned, LR_HF is 0 (normal Coulomb).
    if state.hyb[2] == 0.0 {
        state.hyb[1] = 0.0;
    }

    Ok(XcSpec {
        hyb: (state.hyb[0], state.hyb[1], state.hyb[2]),
        components: remove_dup(&state.fn_facs),
    })
}

/// Parse an XC description into the xcfun spec — the **alternate** resolver.
///
/// Returns `((hyb, alpha, omega), [(xcfun_id, fac), ...])` exactly as upstream
/// `pyscf/dft/xcfun.py:parse_xc`.
///
/// # Errors
/// Returns [`DftError`] (never panics, never recurses unbounded) on malformed
/// input.
pub fn parse_xc(description: &str) -> Result<XcSpec, DftError> {
    parse_xc_inner(description, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(spec: &XcSpec) -> Vec<(u32, f64)> {
        spec.components.clone()
    }

    #[test]
    fn comma_form_pbe() {
        // suffix fallback: PBE in X part -> PBEX(5); PBE in C part -> PBEC(4).
        let spec = parse_xc("pbe,pbe").unwrap();
        assert_eq!(ids(&spec), vec![(5, 1.0), (4, 1.0)]);
    }

    #[test]
    fn shorthand_alias_blyp() {
        // BLYP -> XC_ALIAS "B88,LYP" -> BECKEX(6), LYPC(16).
        let spec = parse_xc("blyp").unwrap();
        assert_eq!(ids(&spec), vec![(6, 1.0), (16, 1.0)]);
    }

    #[test]
    fn b3lyp_expands_to_components() {
        // xcfun b3lyp expands to a weighted primitive set (NOT a single id).
        let spec = parse_xc("b3lyp").unwrap();
        // .2*HF -> hyb 0.2; components: SLATER(0).08, B88(6).72, LYP(16).81, VWN3C(2).19
        assert!((spec.hyb.0 - 0.2).abs() < 1e-12);
        let comps = ids(&spec);
        assert!(comps.contains(&(0, 0.08)));
        assert!(comps.contains(&(6, 0.72)));
        assert!(comps.contains(&(16, 0.81)));
        assert!(comps.contains(&(2, 0.19)));
    }

    #[test]
    fn explicit_weights_with_hf() {
        let spec = parse_xc(".5*HF + .5*B88,LYP").unwrap();
        assert_eq!(spec.hyb, (0.5, 0.0, 0.0)); // LR_HF zeroed (no omega)
        assert_eq!(ids(&spec), vec![(6, 0.5), (16, 1.0)]);
    }

    #[test]
    fn malformed_unknown_is_err_not_panic() {
        assert!(matches!(
            parse_xc("not_a_real_functional"),
            Err(DftError::UnknownFunctional { .. })
        ));
    }

    #[test]
    fn malformed_too_many_commas_is_err() {
        assert!(matches!(
            parse_xc("b88,lyp,extra"),
            Err(DftError::TooManyCommas { .. })
        ));
    }
}
