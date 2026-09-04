//! CP2K / GTH pseudopotential parser.
//!
//! Source: `pyscf/gto/basis/parse_cp2k_pp.py` (`_parse`, Apache-2.0).
//!
//! # Format
//!
//! A GTH pseudopotential file is a sequence of `#PSEUDOPOTENTIAL`-delimited
//! per-element blocks. Each block is:
//!
//! ```text
//! #PSEUDOPOTENTIAL
//! Na GTH-PADE-q1 GTH-LDA-q1        # element + potential-name aliases
//!     1                            # n_elec per l (s p d …); sum = Zion
//!      0.88550938  1  -1.23886713  # rloc  nexp  C_1 … C_nexp   (local part)
//!     2                            # nproj_types — non-local projector channels
//!      0.66110390  2   1.84727135  -0.22540903   # l=0: r  nproj  h-row-0
//!                                   0.58200362    #        h-row-1 (continuation)
//!      0.85711928  1   0.47113258  # l=1: r  nproj  h(0,0)
//! ```
//!
//! The local line carries `rloc`, the count `nexp`, then exactly `nexp`
//! polynomial coefficients. Each non-local channel `l` carries its Gaussian
//! radius `r_l`, the projector count `nproj`, and the UPPER TRIANGLE of the
//! `nproj × nproj` coupling matrix `h` in row-major order: the first row
//! (`nproj` values) sits on the `r/nproj` line, and rows `1 … nproj-1`
//! (`nproj-i` values each) follow on continuation lines. The matrix is
//! symmetric — the lower triangle is mirrored on parse.
//!
//! # Why not `ParsedEcp`
//!
//! GTH pseudopotentials are the PBC-pseudopotential construct (`Cell.pseudo`),
//! structurally distinct from the NWChem `r^n e^{-ζr²}` molecular ECPs that
//! [`pyscf_core::ParsedEcp`] models. They parse into the dedicated
//! [`pyscf_core::GthPseudo`] type. This is a faithful PARSER only — wiring a
//! GTH-pseudo integral path (`Cell.pseudo`) is out of scope for the molecular
//! stack.

use pyscf_core::{EcpLoadError, GthProjector, GthPseudo};

/// Parse CP2K / GTH pseudopotential text for one element `symbol`, taking that
/// element's DEFAULT potential.
///
/// See [`parse_cp2k_pp_with_suffix`] for what "default" means — a `.dat` file
/// carries several blocks per element (`Li GTH-PADE-q1 …`, `Li GTH-PADE-q3 …
/// GTH-PADE GTH-LDA`) and picking the first one silently gives the wrong
/// valence charge.
pub fn parse_cp2k_pp(text: &str, symbol: &str, source: &str) -> Result<GthPseudo, EcpLoadError> {
    parse_cp2k_pp_with_suffix(text, symbol, None, source)
}

/// Parse CP2K / GTH pseudopotential text for one element `symbol`, selecting
/// the block by upstream's rule (`parse_cp2k_pp._search_gthpp_block`).
///
/// A `.dat` file holds one `#PSEUDOPOTENTIAL` block per (element, valence
/// configuration), each opened by a NAME line listing the potential's aliases:
///
/// ```text
/// Li GTH-PADE-q1 GTH-LDA-q1                        <- explicit q1 only
/// Li GTH-PADE-q3 GTH-LDA-q3 GTH-PADE GTH-LDA       <- ALSO the default
/// ```
///
/// * `suffix = None` (the default potential): take the first block whose LAST
///   alias token does NOT end in `-q<digits>`, i.e. the block that also answers
///   to the bare potential name. For `Li`/`gth-pade` that is the **q3** block —
///   taking the first matching block instead yields `Zion = 1` and a cell with
///   the wrong electron count.
/// * `suffix = Some("q4")`: take the first block any of whose alias tokens ends
///   in `-q4`.
///
/// Element matching is case-insensitive on the alphabetic prefix of token 0.
/// Returns a structured [`EcpLoadError::Parse`] when no block matches or the
/// data is malformed (never panics).
pub fn parse_cp2k_pp_with_suffix(
    text: &str,
    symbol: &str,
    suffix: Option<&str>,
    source: &str,
) -> Result<GthPseudo, EcpLoadError> {
    let symbol_upper = symbol.to_ascii_uppercase();

    // Pre-filter: strip `#`-comments (incl. the `#PSEUDOPOTENTIAL` delimiters
    // and the file's `#`-banner), trim, drop blanks and END lines, retaining
    // 1-based source line numbers for diagnostics.
    let mut filtered: Vec<(usize, String)> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let no_comment = match raw.find('#') {
            Some(p) => &raw[..p],
            None => raw,
        };
        let line = no_comment.trim();
        if line.is_empty() || line.to_ascii_uppercase().starts_with("END") {
            continue;
        }
        filtered.push((i + 1, line.to_string()));
    }

    // Segment into element blocks. A NAME line opens a block: its first
    // character is alphabetic, whereas every count / local / projector line
    // begins with a digit, sign, or dot. Take the FIRST block matching `symbol`
    // (the leading-alpha prefix of token 0, uppercased).
    let seg = find_element_block(&filtered, &symbol_upper, suffix).ok_or_else(|| {
        EcpLoadError::Parse {
            file: source.into(),
            line: 0,
            reason: match suffix {
                None => {
                    format!("pseudopotential for element '{symbol}' not found in CP2K/GTH text")
                }
                Some(q) => format!(
                    "pseudopotential for element '{symbol}' with suffix '{q}' \
                     not found in CP2K/GTH text"
                ),
            },
        }
    })?;

    parse_block(seg, source)
}

/// `true` when `tok` (an alias like `GTH-PADE-q4`) carries a `-q<digits>` tail.
/// Ports the `qsuffix.startswith('q') and qsuffix[1:].isdigit()` test at
/// `parse_cp2k_pp.py:152-153`.
fn has_q_suffix(tok: &str) -> bool {
    let tail = tok.rsplit('-').next().unwrap_or("");
    let Some(rest) = tail.strip_prefix(['q', 'Q']) else {
        return false;
    };
    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
}

/// Select an element block per upstream `_search_gthpp_block`
/// (`parse_cp2k_pp.py:145-158`).
fn find_element_block<'a>(
    filtered: &'a [(usize, String)],
    symbol_upper: &str,
    suffix: Option<&str>,
) -> Option<&'a [(usize, String)]> {
    let starts_with_alpha = |s: &str| s.chars().next().is_some_and(|c| c.is_ascii_alphabetic());

    // Blocks for this element, in file order, so the fallback below can reach
    // them after the primary rule finds nothing.
    let mut candidates: Vec<&'a [(usize, String)]> = Vec::new();

    let mut idx = 0;
    while idx < filtered.len() {
        if !starts_with_alpha(&filtered[idx].1) {
            idx += 1;
            continue;
        }
        let first_tok = filtered[idx].1.split_whitespace().next().unwrap_or("");
        let elem: String = first_tok
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect::<String>()
            .to_ascii_uppercase();

        let start = idx;
        idx += 1;
        while idx < filtered.len() && !starts_with_alpha(&filtered[idx].1) {
            idx += 1;
        }
        if elem == symbol_upper {
            candidates.push(&filtered[start..idx]);
        }
    }

    let name_line = |seg: &&'a [(usize, String)]| seg[0].1.clone();

    match suffix {
        // Default potential: the block whose LAST alias is not `-q<n>`.
        None => {
            if let Some(seg) = candidates.iter().find(|seg| {
                name_line(seg)
                    .split_whitespace()
                    .next_back()
                    .is_some_and(|last| !has_q_suffix(last))
            }) {
                return Some(seg);
            }
        }
        // Explicit `-q<n>`: any alias on the NAME line may carry it.
        Some(q) => {
            if let Some(seg) = candidates.iter().find(|seg| {
                name_line(seg).split_whitespace().any(|tok| {
                    tok.rsplit('-')
                        .next()
                        .is_some_and(|tail| tail.eq_ignore_ascii_case(q))
                })
            }) {
                return Some(seg);
            }
            return None;
        }
    }

    // Fallback: a file whose only block for this element IS q-suffixed (upstream
    // returns None here and the caller raises; several one-block-per-element
    // files in the tree — e.g. gth-hf-rev.dat — would then be unusable, so the
    // single unambiguous candidate is accepted).
    if candidates.len() == 1 {
        return Some(candidates[0]);
    }
    None
}

/// Port of `parse_cp2k_pp._parse` for a single element block.
fn parse_block(seg: &[(usize, String)], source: &str) -> Result<GthPseudo, EcpLoadError> {
    let mut it = seg.iter();

    // NAME line (skipped).
    it.next()
        .ok_or_else(|| pp_err(source, 0, "empty pseudopotential segment".into()))?;

    // n_elec per angular momentum.
    let (ne_ln, ne_line) = it
        .next()
        .ok_or_else(|| pp_err(source, 0, "missing n_elec line".into()))?;
    let nelec: Vec<u32> = ne_line
        .split_whitespace()
        .map(str::parse::<u32>)
        .collect::<Result<_, _>>()
        .map_err(|_| {
            pp_err(
                source,
                *ne_ln,
                format!("non-integer n_elec line: '{ne_line}'"),
            )
        })?;

    // Local part: rloc  nexp  C_1 … C_nexp.
    let (loc_ln, loc_line) = it
        .next()
        .ok_or_else(|| pp_err(source, 0, "missing local-part line".into()))?;
    let loc_toks: Vec<&str> = loc_line.split_whitespace().collect();
    if loc_toks.len() < 2 {
        return Err(pp_err(
            source,
            *loc_ln,
            format!("local-part line needs >= 2 fields (rloc nexp …): '{loc_line}'"),
        ));
    }
    let rloc = parse_f64(loc_toks[0], source, *loc_ln, "rloc")?;
    let nexp = parse_usize(loc_toks[1], source, *loc_ln, "nexp")?;
    let local_coeffs: Vec<f64> = loc_toks[2..]
        .iter()
        .map(|t| parse_f64(t, source, *loc_ln, "local coefficient"))
        .collect::<Result<_, _>>()?;
    if local_coeffs.len() != nexp {
        return Err(pp_err(
            source,
            *loc_ln,
            format!(
                "local part declares nexp={nexp} but carries {} coefficient(s): '{loc_line}'",
                local_coeffs.len()
            ),
        ));
    }

    // Non-local projector channels.
    let (np_ln, np_line) = it
        .next()
        .ok_or_else(|| pp_err(source, 0, "missing nproj_types line".into()))?;
    let nproj_types = parse_usize(
        np_line.split_whitespace().next().unwrap_or(""),
        source,
        *np_ln,
        "nproj_types",
    )?;

    let mut projectors = Vec::with_capacity(nproj_types);
    for _ in 0..nproj_types {
        let (r_ln, r_line) = it.next().ok_or_else(|| {
            pp_err(
                source,
                *np_ln,
                "projector channels truncated (pseudo data incomplete)".into(),
            )
        })?;
        let toks: Vec<&str> = r_line.split_whitespace().collect();
        if toks.len() < 2 {
            return Err(pp_err(
                source,
                *r_ln,
                format!("projector line needs >= 2 fields (r nproj …): '{r_line}'"),
            ));
        }
        let r = parse_f64(toks[0], source, *r_ln, "projector radius")?;
        let nproj = parse_usize(toks[1], source, *r_ln, "nproj")?;

        // Upper triangle, row-major: first row (`nproj` values) on this line,
        // then rows 1..nproj-1 on continuation lines.
        let mut tri: Vec<f64> = toks[2..]
            .iter()
            .map(|t| parse_f64(t, source, *r_ln, "h-matrix entry"))
            .collect::<Result<_, _>>()?;
        for _ in 1..nproj {
            let (cont_ln, cont_line) = it.next().ok_or_else(|| {
                pp_err(
                    source,
                    *r_ln,
                    "h-matrix continuation row missing (pseudo data incomplete)".into(),
                )
            })?;
            for t in cont_line.split_whitespace() {
                tri.push(parse_f64(t, source, *cont_ln, "h-matrix entry")?);
            }
        }
        let expected = nproj * (nproj + 1) / 2;
        if tri.len() != expected {
            return Err(pp_err(
                source,
                *r_ln,
                format!(
                    "projector h-matrix upper triangle expected {expected} value(s) for \
                     nproj={nproj}, got {}",
                    tri.len()
                ),
            ));
        }

        // Expand the row-major upper triangle into a full symmetric matrix.
        let mut h = vec![0.0_f64; nproj * nproj];
        let mut k = 0;
        for i in 0..nproj {
            for j in i..nproj {
                let v = tri[k];
                k += 1;
                h[i * nproj + j] = v;
                h[j * nproj + i] = v;
            }
        }
        projectors.push(GthProjector { r, nproj, h });
    }

    Ok(GthPseudo {
        nelec,
        rloc,
        local_coeffs,
        projectors,
    })
}

fn parse_f64(tok: &str, source: &str, line: usize, what: &str) -> Result<f64, EcpLoadError> {
    tok.parse::<f64>()
        .map_err(|_| pp_err(source, line, format!("non-numeric {what}: '{tok}'")))
}

fn parse_usize(tok: &str, source: &str, line: usize, what: &str) -> Result<usize, EcpLoadError> {
    tok.parse::<usize>()
        .map_err(|_| pp_err(source, line, format!("non-integer {what}: '{tok}'")))
}

fn pp_err(source: &str, line: usize, reason: String) -> EcpLoadError {
    EcpLoadError::Parse {
        file: source.into(),
        line,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H GTH-PADE-q1 — local part only, no non-local projectors. Verbatim from
    /// `pyscf/pbc/gto/pseudo/gth-pade.dat`.
    #[test]
    fn h_pade_local_only() {
        let text = "#PSEUDOPOTENTIAL\n\
                    H GTH-PADE-q1 GTH-LDA-q1 GTH-PADE GTH-LDA\n\
                    \x20   1\n\
                    \x20    0.20000000    2    -4.18023680     0.72507482\n\
                    \x20   0\n";
        let p = parse_cp2k_pp(text, "H", "test").unwrap();
        assert_eq!(p.nelec, vec![1]);
        approx::assert_abs_diff_eq!(p.rloc, 0.20000000, epsilon = 1e-9);
        assert_eq!(p.local_coeffs.len(), 2);
        approx::assert_abs_diff_eq!(p.local_coeffs[0], -4.18023680, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.local_coeffs[1], 0.72507482, epsilon = 1e-9);
        assert!(p.projectors.is_empty());
    }

    /// Na GTH-PADE-q1 — two non-local channels; the l=0 channel has a 2×2
    /// upper-triangular h-matrix split across a continuation line, exercising
    /// the symmetrisation. Verbatim from `gth-pade.dat`.
    #[test]
    fn na_pade_2x2_hmatrix_symmetrised() {
        let text = "#PSEUDOPOTENTIAL\n\
                    Na GTH-PADE-q1 GTH-LDA-q1\n\
                    \x20   1\n\
                    \x20    0.88550938    1    -1.23886713\n\
                    \x20   2\n\
                    \x20    0.66110390    2     1.84727135    -0.22540903\n\
                    \x20                                       0.58200362\n\
                    \x20    0.85711928    1     0.47113258\n";
        let p = parse_cp2k_pp(text, "Na", "test").unwrap();
        assert_eq!(p.nelec, vec![1]);
        approx::assert_abs_diff_eq!(p.rloc, 0.88550938, epsilon = 1e-9);
        assert_eq!(p.local_coeffs, vec![-1.23886713]);
        assert_eq!(p.projectors.len(), 2);

        // l=0 channel: 2×2 symmetric h.
        let p0 = &p.projectors[0];
        approx::assert_abs_diff_eq!(p0.r, 0.66110390, epsilon = 1e-9);
        assert_eq!(p0.nproj, 2);
        assert_eq!(p0.h.len(), 4);
        approx::assert_abs_diff_eq!(p0.h[0], 1.84727135, epsilon = 1e-9); // h(0,0)
        approx::assert_abs_diff_eq!(p0.h[1], -0.22540903, epsilon = 1e-9); // h(0,1)
        approx::assert_abs_diff_eq!(p0.h[2], -0.22540903, epsilon = 1e-9); // h(1,0) mirrored
        approx::assert_abs_diff_eq!(p0.h[3], 0.58200362, epsilon = 1e-9); // h(1,1)

        // l=1 channel: scalar projector.
        let p1 = &p.projectors[1];
        assert_eq!(p1.nproj, 1);
        approx::assert_abs_diff_eq!(p1.r, 0.85711928, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p1.h[0], 0.47113258, epsilon = 1e-9);
    }

    /// Multi-l n_elec is parsed in full (Si GTH-PADE-q4: s=2, p=2). Verbatim.
    #[test]
    fn si_pade_multi_l_nelec() {
        let text = "#PSEUDOPOTENTIAL\n\
                    Si GTH-PADE-q4 GTH-LDA-q4 GTH-PADE GTH-LDA\n\
                    \x20   2    2\n\
                    \x20    0.44000000    1    -7.33610297\n\
                    \x20   2\n\
                    \x20    0.42273813    2     5.90692831    -1.26189397\n\
                    \x20                                       3.25819622\n\
                    \x20    0.48427842    1     2.72701346\n";
        let p = parse_cp2k_pp(text, "Si", "test").unwrap();
        assert_eq!(p.nelec, vec![2, 2]);
        assert_eq!(p.projectors.len(), 2);
        assert_eq!(p.projectors[0].nproj, 2);
    }

    /// Other elements' blocks are skipped; the requested element is isolated.
    #[test]
    fn skips_other_element_blocks() {
        let text = "#PSEUDOPOTENTIAL\n\
                    H GTH-PADE-q1\n\
                    \x20   1\n\
                    \x20    0.20000000    2    -4.18023680     0.72507482\n\
                    \x20   0\n\
                    #PSEUDOPOTENTIAL\n\
                    He GTH-PADE-q2\n\
                    \x20   2\n\
                    \x20    0.20000000    2    -9.11202340     1.69836797\n\
                    \x20   0\n";
        let p = parse_cp2k_pp(text, "He", "test").unwrap();
        assert_eq!(p.nelec, vec![2]);
        approx::assert_abs_diff_eq!(p.local_coeffs[0], -9.11202340, epsilon = 1e-9);
    }

    /// Absent element → structured Parse error (never a panic).
    #[test]
    fn missing_element_errors() {
        let text = "#PSEUDOPOTENTIAL\n\
                    H GTH-PADE-q1\n\
                    \x20   1\n\
                    \x20    0.20000000    2    -4.18023680     0.72507482\n\
                    \x20   0\n";
        let err = parse_cp2k_pp(text, "U", "test").unwrap_err();
        match err {
            EcpLoadError::Parse { reason, .. } => assert!(reason.contains("not found"), "{reason}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    /// A truncated h-matrix continuation → completeness Parse error.
    #[test]
    fn truncated_hmatrix_errors() {
        let text = "Na GTH-PADE-q1\n\
                    \x20   1\n\
                    \x20    0.88550938    1    -1.23886713\n\
                    \x20   1\n\
                    \x20    0.66110390    2     1.84727135    -0.22540903\n"; // missing h(1,1) row
        let err = parse_cp2k_pp(text, "Na", "test").unwrap_err();
        match err {
            EcpLoadError::Parse { .. } => {}
            other => panic!("expected Parse error, got {other:?}"),
        }
    }
}
