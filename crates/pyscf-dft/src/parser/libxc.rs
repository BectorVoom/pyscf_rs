//! libxc-side XC-string parser — the **default** resolver (D-01).
//!
//! Direct port of `pyscf/dft/libxc.py:parse_xc` (lines 491-718), with the
//! `XC_CODES` (libxc.py:154-202) and `XC_ALIAS` (libxc.py:217-292) tables
//! transcribed as inline `const` slices (NO codegen / NO build.rs — the
//! `pyscf-df::auxbasis::DEFAULT_AUXBASIS` convention). Helper ports:
//! `format_xc_code` + `remove_dup` (`pyscf/dft/xc/utils.py`),
//! `_NAME_WITH_DASH` (libxc.py:684-716).
//!
//! License: Apache-2.0 (matching upstream PySCF).
//! Requirement: DFT-02 (parser parity) + D-01 (libxc-default routing).
//!
//! The libxc functional IDs below are the canonical libxc 7.0.0 numbers — the
//! exact values produced by `libxc_rs::lookup_by_name` (verified against
//! `libxc_rs/src/registry/by_name.rs`: B88=106, LYP=131, GGA_X_PBE=101,
//! GGA_C_PBE=130, B3LYP=402, PBEH=406, CAM_B3LYP=433, LDA_X=1, VWN5=7,
//! VWN_RPA=8). The default build never compiles `libxc_rs`; these are inline
//! constants so the parser stands alone, and `lookup_by_name` is consulted
//! ONLY for names absent from the inline table, behind `#[cfg(feature =
//! "libxc")]`.
//!
//! Security (T-04-05a/b): pure string→spec mapping; never evaluates code; the
//! compound-expansion recursion is depth-bounded.

use crate::error::DftError;
use crate::parser::{Component, MAX_EXPANSION_DEPTH, XcSpec, remove_dup};

/// An `XC_CODES` / `XC_ALIAS` entry either resolves directly to a libxc
/// integer ID, or expands to another XC string (compound, recursively parsed).
#[derive(Clone, Copy)]
enum Resolved {
    /// A direct libxc functional ID (e.g. `B88` → 106).
    Id(u32),
    /// A string that is itself an XC description, recursively parsed and scaled
    /// (e.g. `B3LYP5` → `.2*HF + .08*SLATER + .72*B88, .81*LYP + .19*VWN`).
    Expand(&'static str),
}

/// `XC_CODES` (libxc.py:154-202) — name → libxc ID *or* compound expansion.
/// Only the entries exercised by the parser/parity surface are transcribed;
/// any name absent here falls through to the gated `libxc_rs::lookup_by_name`
/// (under `--features libxc`) or yields a clean `UnknownFunctional` error.
/// Names are stored UPPERCASE (parser upper-cases the input first).
const XC_CODES: &[(&str, Resolved)] = &[
    // Primitive libxc IDs (libxc 7.0.0 canonical numbers).
    ("LDA", Resolved::Id(1)),
    ("SLATER", Resolved::Id(1)),
    ("LDA_X", Resolved::Id(1)),
    ("VWN", Resolved::Id(7)),
    ("VWN5", Resolved::Id(7)),
    ("LDA_C_VWN", Resolved::Id(7)),
    ("VWN3", Resolved::Id(8)),
    ("VWNRPA", Resolved::Id(8)),
    ("LDA_C_VWN_RPA", Resolved::Id(8)),
    ("B88", Resolved::Id(106)),
    ("GGA_X_B88", Resolved::Id(106)),
    ("LYP", Resolved::Id(131)),
    ("GGA_C_LYP", Resolved::Id(131)),
    ("P86", Resolved::Id(132)),
    ("GGA_C_P86", Resolved::Id(132)),
    // NOTE: there is intentionally NO bare `PBE` primitive — upstream resolves
    // it via the part-aware family search (`GGA_X_PBE` in the X part,
    // `GGA_C_PBE` in the C part) and via `XC_ALIAS["PBE"] = "PBE,PBE"` in the
    // compound (no-comma) branch.
    ("GGA_X_PBE", Resolved::Id(101)),
    ("PBE_R", Resolved::Id(102)),
    ("GGA_X_PBE_R", Resolved::Id(102)),
    ("GGA_C_PBE", Resolved::Id(130)),
    ("OPTX", Resolved::Id(110)),
    ("GGA_X_OPTX", Resolved::Id(110)),
    ("PBE0", Resolved::Id(406)),
    ("PBE1PBE", Resolved::Id(406)),
    ("HYB_GGA_XC_PBEH", Resolved::Id(406)),
    ("B3LYP", Resolved::Id(402)),
    ("B3LYPG", Resolved::Id(402)),
    ("HYB_GGA_XC_B3LYP", Resolved::Id(402)),
    ("CAMB3LYP", Resolved::Expand("HYB_GGA_XC_CAM_B3LYP")),
    ("CAM_B3LYP", Resolved::Id(433)),
    ("HYB_GGA_XC_CAM_B3LYP", Resolved::Id(433)),
    ("SCAN", Resolved::Id(263)), // MGGA_X_SCAN primitive
    ("MGGA_X_SCAN", Resolved::Id(263)),
    ("MGGA_C_SCAN", Resolved::Id(267)),
    // Compound expansions (libxc.py:173,195,198 ...).
    (
        "B3LYP5",
        Resolved::Expand(".2*HF + .08*SLATER + .72*B88, .81*LYP + .19*VWN"),
    ),
    (
        "B5050LYP",
        Resolved::Expand(".5*HF + .08*SLATER + .42*B88, .81*LYP + .19*VWN"),
    ),
    ("PBE50", Resolved::Expand(".5*HF + .5*PBE, PBE")),
];

/// `XC_ALIAS` (libxc.py:217-292) — conventional compound names that MUST be
/// treated as a shortcut (X,C) string rather than a primitive (otherwise they
/// would recurse). Only consulted for the no-comma compound branch
/// (`search_xc_alias=true`). Names stored UPPERCASE.
const XC_ALIAS: &[(&str, &str)] = &[
    ("BLYP", "B88,LYP"),
    ("BP86", "B88,P86"),
    ("PW91", "PW91,PW91"),
    ("PBE", "PBE,PBE"),
    ("REVPBE", "PBE_R,PBE"),
    ("OLYP", "OPTX,LYP"),
    ("OPBE", "OPTX,PBE"),
    ("RPBE", "RPBE,PBE"),
    ("BPBE", "B88,PBE"),
    ("HFLYP", "HF,LYP"),
    ("SVWN", "SLATER,VWN"),
    ("SCAN", "SCAN,SCAN"),
];

/// `_NAME_WITH_DASH` (libxc.py:684-716) — dashed names whose dash is part of
/// the *name*, not a subtraction operator. Applied before the `'-' → '+-'`
/// operator split. Only the parity-relevant subset is transcribed.
const NAME_WITH_DASH: &[(&str, &str)] = &[
    ("SR-HF", "SR_HF"),
    ("LR-HF", "LR_HF"),
    ("M06-L", "M06_L"),
    ("CAM-B3LYP", "CAM_B3LYP"),
    ("E-", "E_"),
];

/// Which functional part a token belongs to — selects the family-prefix set
/// for the fuzzy `possible_*_for` fallback (libxc.py:643-662).
#[derive(Clone, Copy)]
enum Ftype {
    /// X part of a comma form, or the standalone X functional ("X or K").
    XorK,
    /// C part of a comma form.
    C,
    /// Compound XC (no-comma branch).
    CompoundXc,
}

impl Ftype {
    /// Candidate family-prefixed names to try, in upstream-equivalent order.
    /// The bare `key` is tried first by the caller; this returns the prefixed
    /// variants. `_X_`/`X` candidates precede others to mirror upstream's
    /// "Prefer X functional" tie-break (libxc.py:614-616).
    fn prefixes(self) -> &'static [&'static str] {
        match self {
            // possible_x_for ∪ possible_k_for
            Ftype::XorK => &[
                "LDA_X_",
                "GGA_X_",
                "MGGA_X_",
                "HYB_GGA_X_",
                "HYB_MGGA_X_",
                "LDA_K_",
                "GGA_K_",
            ],
            // possible_c_for
            Ftype::C => &["LDA_C_", "GGA_C_", "MGGA_C_"],
            // possible_xc_for
            Ftype::CompoundXc => &[
                "LDA_XC_",
                "GGA_XC_",
                "MGGA_XC_",
                "HYB_LDA_XC_",
                "HYB_GGA_XC_",
                "HYB_MGGA_XC_",
            ],
        }
    }
}

/// Mutable parse state threaded through the token loop (mirrors the Python
/// closure captures `hyb` + `fn_facs`).
struct ParseState {
    /// `hyb = [hybrid, alpha, omega]` (libxc.py:539).
    hyb: [f64; 3],
    /// Accumulated `(id, fac)` components.
    fn_facs: Vec<Component>,
}

/// `format_xc_code` port (`pyscf/dft/xc/utils.py:51`): strip whitespace,
/// upper-case, and rewrite `RSH(omega,alpha,beta)` → internal
/// `RSH(alpha;beta;omega)`.
fn format_xc_code(description: &str) -> String {
    let cleaned: String = description
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_uppercase();

    if !cleaned.contains("RSH") {
        return cleaned;
    }

    // Split on "RSH"; first fragment is the prefix, each subsequent fragment
    // starts at "(...)rest". (libxc.py utils.py:55-62)
    let frags: Vec<&str> = cleaned.split("RSH").collect();
    let mut out = String::from(frags[0]);
    for frag in &frags[1..] {
        out.push_str("RSH");
        if let Some(close) = frag.find(')') {
            let rsh_key = &frag[..close]; // "(omega,alpha,beta"
            let rest = &frag[close + 1..];
            // rsh_key[1:] drops the leading '('.
            let inner = &rsh_key[1.min(rsh_key.len())..];
            if inner.contains(',') {
                let parts: Vec<&str> = inner.split(',').collect();
                if parts.len() == 3 {
                    // (omega, alpha, beta) -> (alpha; beta; omega)
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
            // No comma form — already in internal notation; keep verbatim.
            out.push_str(rsh_key);
            out.push(')');
            out.push_str(rest);
        } else {
            out.push_str(frag);
        }
    }
    out
}

/// `assign_omega` port (libxc.py:553-562): accumulate SR/LR HF mixing while
/// enforcing a single consistent omega across all RSH terms.
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

/// Look up a name in `XC_CODES` (UPPERCASE input expected).
fn xc_codes_get(key: &str) -> Option<Resolved> {
    XC_CODES.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// Look up a name in `XC_ALIAS` (UPPERCASE input expected).
fn xc_alias_get(key: &str) -> Option<&'static str> {
    XC_ALIAS.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// Parse a float, applying the upstream `E_` → `E-` exponent fixup
/// (libxc.py:576). Returns a clean `MalformedToken` on failure rather than
/// panicking (T-04-05a).
fn parse_factor(xc: &str, token: &str, raw: &str) -> Result<f64, DftError> {
    raw.replace("E_", "E-")
        .parse::<f64>()
        .map_err(|e| DftError::MalformedToken {
            xc: xc.to_string(),
            token: token.to_string(),
            reason: format!("invalid factor '{raw}': {e}"),
        })
}

/// `parse_token` port (libxc.py:565-642). `search_xc_alias` enables the
/// `XC_ALIAS` shortcut lookup (only in the no-comma compound branch).
#[allow(clippy::too_many_lines)]
fn parse_token(
    state: &mut ParseState,
    xc: &str,
    token: &str,
    ftype: Ftype,
    search_xc_alias: bool,
    depth: usize,
) -> Result<(), DftError> {
    if token.is_empty() {
        return Ok(());
    }
    // Bounded-depth guard against adversarial / cyclic compound expansion
    // (T-04-05b — DoS via unbounded recursion).
    if depth > MAX_EXPANSION_DEPTH {
        return Err(DftError::ExpansionDepthExceeded {
            xc: xc.to_string(),
            limit: MAX_EXPANSION_DEPTH,
        });
    }

    // Sign prefix (libxc.py:567-571).
    let (sign, token) = if let Some(stripped) = token.strip_prefix('-') {
        (-1.0_f64, stripped)
    } else {
        (1.0_f64, token)
    };

    // Factor: `0.5*B88` or `B88*0.5` (libxc.py:572-578).
    let (fac, key): (f64, &str) = if let Some(star) = token.find('*') {
        let (a, b) = (&token[..star], &token[star + 1..]);
        // If the first operand starts with a letter, it's the *key*, not the factor.
        let (fac_str, key) = if a.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
            (b, a)
        } else {
            (a, b)
        };
        (sign * parse_factor(xc, token, fac_str)?, key)
    } else {
        (sign, token)
    };

    // --- RSH(alpha;beta;omega) (libxc.py:580-584) ---
    if key.len() >= 3 && &key[..3] == "RSH" {
        // key[4:-1] drops "RSH(" and ")".
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

    // --- HF (libxc.py:585-587) ---
    if key == "HF" {
        state.hyb[0] += fac;
        state.hyb[1] += fac;
        return Ok(());
    }

    // --- SR_HF / LR_HF (libxc.py:588-599) ---
    if key.contains("SR_HF") {
        if let Some(omega) = parse_paren_omega(xc, token, key)? {
            return assign_omega(state, xc, omega, fac, 0.0);
        }
        state.hyb[0] += fac;
        return Ok(());
    }
    if key.contains("LR_HF") {
        if let Some(omega) = parse_paren_omega(xc, token, key)? {
            return assign_omega(state, xc, omega, 0.0, fac);
        }
        state.hyb[1] += fac;
        return Ok(());
    }

    // --- raw integer ID (libxc.py:600-601) ---
    if !key.is_empty() && key.chars().all(|c| c.is_ascii_digit()) {
        let id: u32 = key.parse().map_err(|e| DftError::MalformedToken {
            xc: xc.to_string(),
            token: token.to_string(),
            reason: format!("invalid integer functional ID '{key}': {e}"),
        })?;
        state.fn_facs.push((id, fac));
        return Ok(());
    }

    // --- name resolution (libxc.py:602-642) ---
    // 1. XC_ALIAS (compound branch only), 2. bare key in XC_CODES,
    // 3. part-aware family-prefix fuzzy search (possible_*_for ∩ XC_KEYS).
    let mut resolved = if search_xc_alias {
        if let Some(s) = xc_alias_get(key) {
            Some(Resolved::Expand(s))
        } else {
            xc_codes_get(key)
        }
    } else {
        xc_codes_get(key)
    };
    if resolved.is_none() {
        for prefix in ftype.prefixes() {
            if let Some(r) = xc_codes_get(&format!("{prefix}{key}")) {
                resolved = Some(r);
                break;
            }
        }
    }

    let resolved = match resolved {
        Some(r) => r,
        None => {
            // Names absent from the inline table fall through to the gated
            // libxc_rs registry (CI-only, --features libxc). Default builds
            // never reference libxc_rs and return a clean error.
            #[cfg(feature = "libxc")]
            {
                match libxc_rs::lookup_by_name(key) {
                    Ok(id) => {
                        state.fn_facs.push((u32::from(id.raw()), fac));
                        return Ok(());
                    }
                    Err(_) => {
                        return Err(DftError::UnknownFunctional {
                            xc: xc.to_string(),
                            token: token.to_string(),
                        });
                    }
                }
            }
            #[cfg(not(feature = "libxc"))]
            {
                return Err(DftError::UnknownFunctional {
                    xc: xc.to_string(),
                    token: token.to_string(),
                });
            }
        }
    };

    match resolved {
        Resolved::Id(id) => {
            state.fn_facs.push((id, fac));
            Ok(())
        }
        Resolved::Expand(s) => {
            // Recursively scale the composed functional (libxc.py:633-638).
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

/// Extract `omega` from a `SR_HF(0.1)` / `LR_HF(0.1)` term. `Ok(None)` when no
/// parenthesis is present (libxc.py:589-599).
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

/// Split a code fragment into `+`/`-`-separated tokens, preserving the upstream
/// `'-' → '+-'` and `';+' → ';'` rewrites (libxc.py:673-678).
fn split_tokens(code: &str) -> Vec<String> {
    code.replace('-', "+-")
        .replace(";+", ";")
        .split('+')
        .map(str::to_string)
        .collect()
}

/// Internal recursive entry — carries the expansion depth.
fn parse_xc_inner(description: &str, depth: usize) -> Result<XcSpec, DftError> {
    if depth > MAX_EXPANSION_DEPTH {
        return Err(DftError::ExpansionDepthExceeded {
            xc: description.to_string(),
            limit: MAX_EXPANSION_DEPTH,
        });
    }

    let mut description = format_xc_code(description);

    // _NAME_WITH_DASH rewrites (libxc.py:666-669) — protect dashes that are
    // part of a functional name from the operator split.
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
            parse_token(&mut state, &description, &token, Ftype::XorK, false, depth)?;
        }
        for token in split_tokens(c_code) {
            parse_token(&mut state, &description, &token, Ftype::C, false, depth)?;
        }
    } else {
        for token in split_tokens(&description) {
            parse_token(
                &mut state,
                &description,
                &token,
                Ftype::CompoundXc,
                true,
                depth,
            )?;
        }
    }

    Ok(XcSpec {
        hyb: (state.hyb[0], state.hyb[1], state.hyb[2]),
        components: remove_dup(&state.fn_facs),
    })
}

/// Parse an XC description into the libxc spec — the **default** resolver.
///
/// Returns `((hyb, alpha, omega), [(libxc_id, fac), ...])` exactly as upstream
/// `pyscf/dft/libxc.py:parse_xc` (DFT-02, D-01).
///
/// # Errors
/// Returns [`DftError`] (never panics, never recurses unbounded) on malformed
/// input: unknown functional, bad factor/RSH/omega, multiple commas,
/// conflicting omega, or expansion-depth overflow.
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
    fn single_compound_b3lyp() {
        let spec = parse_xc("b3lyp").unwrap();
        // Compound libxc id 402, factor 1; no separate HF mixing (libxc folds
        // the hybrid coefficient into the functional itself).
        assert_eq!(ids(&spec), vec![(402, 1.0)]);
        assert_eq!(spec.hyb, (0.0, 0.0, 0.0));
    }

    #[test]
    fn comma_form_pbe() {
        let spec = parse_xc("pbe,pbe").unwrap();
        assert_eq!(ids(&spec), vec![(101, 1.0), (130, 1.0)]);
    }

    #[test]
    fn shorthand_alias_blyp() {
        // BLYP is in XC_ALIAS -> "B88,LYP".
        let spec = parse_xc("blyp").unwrap();
        assert_eq!(ids(&spec), vec![(106, 1.0), (131, 1.0)]);
    }

    #[test]
    fn explicit_weights_with_hf() {
        let spec = parse_xc(".5*HF + .5*B88,LYP").unwrap();
        assert_eq!(spec.hyb, (0.5, 0.5, 0.0));
        assert_eq!(ids(&spec), vec![(106, 0.5), (131, 1.0)]);
    }

    #[test]
    fn unit_scaled_compound() {
        // 0.5*b3lyp5 scales the expansion as a unit.
        let spec = parse_xc("0.5*b3lyp5").unwrap();
        // .2*HF*.5 = .1 HF/LR; components halved.
        assert!((spec.hyb.0 - 0.1).abs() < 1e-12);
        assert!((spec.hyb.1 - 0.1).abs() < 1e-12);
    }

    #[test]
    fn malformed_unknown_is_err_not_panic() {
        assert!(matches!(
            parse_xc("definitely_not_a_functional"),
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

    #[test]
    fn malformed_bad_factor_is_err() {
        assert!(matches!(
            parse_xc("abc*b88,lyp"),
            Err(DftError::MalformedToken { .. }) | Err(DftError::UnknownFunctional { .. })
        ));
    }
}
