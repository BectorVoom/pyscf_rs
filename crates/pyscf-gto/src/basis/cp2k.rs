//! CP2K / GTH basis-set parser.
//!
//! Source: `pyscf/gto/basis/parse_cp2k.py` (`_parse`) + `parse_nwchem.remove_zero`
//! (Apache-2.0).
//!
//! # Format
//!
//! A CP2K / GTH basis file is a sequence of per-element blocks. Each block
//! opens with a NAME line (element symbol + one or more basis aliases),
//! followed by an integer `nsets`, then `nsets` *sets*. Each set is a
//! COMPONENT line `n lmin lmax nexp nshell(lmin) … nshell(lmax)` followed by
//! `nexp` primitive rows. `#`-prefixed lines (`#BASIS SET`, `#`) are comments.
//!
//! ```text
//! #BASIS SET
//! Li DZVP-GTH
//!   2
//!   1  0  1  5  3  2          # n=1 lmin=0 lmax=1 nexp=5 nshell=[3,2]
//!         7.2610457926  -0.2798628497  0.0501840556  0.0  0.0  0.0
//!         …                                                          (5 rows)
//!   3  2  2  1  1             # n=3 lmin=2 lmax=2 nexp=1 nshell=[1]
//!         0.1239000000   1.0000000000
//! ```
//!
//! Each primitive row carries `exp` followed by `sum(nshell)` contraction
//! coefficients; the coefficients are split across the angular momenta
//! `lmin..=lmax` in order (`nshell[i]` columns per `l`). Angular momenta inside
//! one set therefore SHARE the set's exponent vector — the multi-`l` analogue
//! of NWChem's `SP` shared-exponent form.
//!
//! # Faithfulness to upstream
//!
//! This ports `_parse` exactly: per-set component decoding, the
//! `len(row) == sum(nshell)+1` completeness check, the final stable sort by
//! `l` over `0..MAXL` (shells with `l >= MAXL` are dropped), and the
//! `remove_zero` pass — which drops every primitive (exponent) whose
//! contraction coefficients are all zero, then drops a shell left with no
//! primitives. `remove_zero` is what collapses the zero-padded shared-exponent
//! columns (e.g. Li DZVP's `l=1` shell from 5 shared exponents down to 2).

use pyscf_core::{BasisLoadError, ParsedBasis, ShellSpec};

/// Upstream `parse_cp2k.MAXL`. The final l-sort iterates `0..MAXL`, so a shell
/// with `l >= MAXL` is silently dropped (mirrored here).
const MAXL: u8 = 8;

/// Parse CP2K / GTH-format text for one element `symbol`.
///
/// Only the FIRST block whose NAME-line element matches `symbol`
/// (case-insensitive) is kept; other elements' blocks are skipped. Returns a
/// structured [`BasisLoadError::Parse`] when the element is absent or the data
/// is malformed (never panics — T-02-11).
pub fn parse_cp2k(text: &str, symbol: &str, source: &str) -> Result<ParsedBasis, BasisLoadError> {
    let symbol_upper = symbol.to_ascii_uppercase();

    // Pre-filter: strip `#`-comments, trim, drop blanks and END markers,
    // retaining 1-based source line numbers for diagnostics.
    let mut filtered: Vec<(usize, String)> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let no_comment = match raw.find('#') {
            Some(p) => &raw[..p],
            None => raw,
        };
        let line = no_comment.trim();
        if line.is_empty() {
            continue;
        }
        // `search_seg` filters lines containing END (mixed-file boundary guard).
        if line.to_ascii_uppercase().starts_with("END") {
            continue;
        }
        filtered.push((i + 1, line.to_string()));
    }

    // Segment the stream into element blocks. A NAME line begins a block: its
    // first character is alphabetic, whereas every count / component /
    // primitive line begins with a digit, sign, or dot. Mirrors
    // `_search_basis_block`, which keys off the first whitespace token being
    // the element symbol. Take the FIRST block matching `symbol`.
    let seg =
        find_element_block(&filtered, &symbol_upper).ok_or_else(|| BasisLoadError::Parse {
            file: source.into(),
            line: 0,
            reason: format!("element '{symbol}' not found in CP2K/GTH basis text"),
        })?;

    // Decode the segment (`_parse`), then sort-by-l + remove_zero.
    let raw_shells = parse_segment(seg, source)?;
    let shells = finalize(raw_shells);

    if shells.is_empty() {
        return Err(BasisLoadError::Parse {
            file: source.into(),
            line: 0,
            reason: format!("no basis shells parsed for element '{symbol}'"),
        });
    }
    Ok(ParsedBasis { shells })
}

/// Return the first element block (`&[(line_no, line)]`) whose NAME-line
/// element (leading-alphabetic prefix of the first token, uppercased) equals
/// `symbol_upper`.
fn find_element_block<'a>(
    filtered: &'a [(usize, String)],
    symbol_upper: &str,
) -> Option<&'a [(usize, String)]> {
    let starts_with_alpha = |s: &str| s.chars().next().is_some_and(|c| c.is_ascii_alphabetic());

    let mut idx = 0;
    while idx < filtered.len() {
        if !starts_with_alpha(&filtered[idx].1) {
            // Stray non-header line before any block opens — skip.
            idx += 1;
            continue;
        }
        // NAME line opens a block. Element = leading-alpha prefix of token 0
        // (consistent with the nwchem sibling parser's `elem_alpha`).
        let first_tok = filtered[idx].1.split_whitespace().next().unwrap_or("");
        let elem: String = first_tok
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect::<String>()
            .to_ascii_uppercase();

        // Block body: NAME line + following non-NAME lines.
        let start = idx;
        idx += 1;
        while idx < filtered.len() && !starts_with_alpha(&filtered[idx].1) {
            idx += 1;
        }
        if elem == symbol_upper {
            return Some(&filtered[start..idx]);
        }
    }
    None
}

/// Port of `parse_cp2k._parse` for a single element block. Produces one
/// [`ShellSpec`] per angular momentum per set (still in source order; the
/// l-sort + remove_zero happen in [`finalize`]).
fn parse_segment(seg: &[(usize, String)], source: &str) -> Result<Vec<ShellSpec>, BasisLoadError> {
    let mut it = seg.iter();

    // NAME line (skipped).
    it.next()
        .ok_or_else(|| parse_err(source, 0, "empty CP2K basis segment".into()))?;

    // nsets.
    let (nsets_ln, nsets_line) = it
        .next()
        .ok_or_else(|| parse_err(source, 0, "missing nsets line".into()))?;
    let nsets: usize = nsets_line
        .split_whitespace()
        .next()
        .unwrap_or("")
        .parse()
        .map_err(|_| {
            parse_err(
                source,
                *nsets_ln,
                format!("expected integer set-count, got '{nsets_line}'"),
            )
        })?;

    let mut shells: Vec<ShellSpec> = Vec::new();
    for _ in 0..nsets {
        let (comp_ln, comp_line) = it.next().ok_or_else(|| {
            parse_err(
                source,
                0,
                "missing component line (basis data incomplete)".into(),
            )
        })?;
        let comp: Vec<i64> = comp_line
            .split_whitespace()
            .map(str::parse::<i64>)
            .collect::<Result<_, _>>()
            .map_err(|_| {
                parse_err(
                    source,
                    *comp_ln,
                    format!("non-integer component line: '{comp_line}'"),
                )
            })?;
        if comp.len() < 5 {
            return Err(parse_err(
                source,
                *comp_ln,
                format!(
                    "component line needs >= 5 integers (n lmin lmax nexp nshell…), got {}: '{comp_line}'",
                    comp.len()
                ),
            ));
        }
        let (lmin, lmax, nexp) = (comp[1], comp[2], comp[3]);
        if lmin < 0 || lmax < lmin || nexp < 0 {
            return Err(parse_err(
                source,
                *comp_ln,
                format!("invalid lmin/lmax/nexp in component line: '{comp_line}'"),
            ));
        }
        let nls = (lmax - lmin + 1) as usize;
        let ncontr = &comp[4..];
        if ncontr.len() != nls {
            return Err(parse_err(
                source,
                *comp_ln,
                format!(
                    "component line lists {} contraction count(s) but lmin..=lmax spans {} \
                     angular momenta: '{comp_line}'",
                    ncontr.len(),
                    nls
                ),
            ));
        }
        if ncontr.iter().any(|&c| c < 0) {
            return Err(parse_err(
                source,
                *comp_ln,
                format!("negative contraction count in component line: '{comp_line}'"),
            ));
        }
        let sum_ncontr: i64 = ncontr.iter().sum();

        // Per-l accumulators. `coeffs_per_l[li][ctr][prim]` matches the
        // `ShellSpec.coeffs[ctr][prim]` column-major layout.
        let mut exps_per_l: Vec<Vec<f64>> = vec![Vec::with_capacity(nexp as usize); nls];
        let mut coeffs_per_l: Vec<Vec<Vec<f64>>> = ncontr
            .iter()
            .map(|&c| vec![Vec::with_capacity(nexp as usize); c as usize])
            .collect();

        for _ in 0..nexp {
            let (row_ln, row) = it.next().ok_or_else(|| {
                parse_err(
                    source,
                    *comp_ln,
                    "primitive rows truncated (basis data incomplete)".into(),
                )
            })?;
            let nums: Vec<f64> = row
                .split_whitespace()
                .map(str::parse::<f64>)
                .collect::<Result<_, _>>()
                .map_err(|_| {
                    parse_err(
                        source,
                        *row_ln,
                        format!("non-numeric primitive row: '{row}'"),
                    )
                })?;
            // Upstream's `len(bfun) == sum(ncontractions) + 1` completeness check.
            if nums.len() as i64 != sum_ncontr + 1 {
                return Err(parse_err(
                    source,
                    *row_ln,
                    format!(
                        "primitive row has {} number(s), expected {} (exp + sum of \
                         contractions): '{row}'",
                        nums.len(),
                        sum_ncontr + 1
                    ),
                ));
            }
            let exp = nums[0];
            let mut off = 1usize;
            for li in 0..nls {
                exps_per_l[li].push(exp);
                let nc = ncontr[li] as usize;
                for c in 0..nc {
                    coeffs_per_l[li][c].push(nums[off + c]);
                }
                off += nc;
            }
        }

        for li in 0..nls {
            shells.push(ShellSpec {
                l: (lmin + li as i64) as u8,
                exponents: std::mem::take(&mut exps_per_l[li]),
                coeffs: std::mem::take(&mut coeffs_per_l[li]),
            });
        }
    }

    Ok(shells)
}

/// Stable sort by `l` over `0..MAXL` (dropping `l >= MAXL`), then `remove_zero`.
fn finalize(mut shells: Vec<ShellSpec>) -> Vec<ShellSpec> {
    shells.retain(|s| s.l < MAXL);
    // `sort_by_key` is stable, preserving source order within each `l` — the
    // exact semantics of upstream's `for l in range(MAXL): [b for b in basis
    // if b[0] == l]`.
    shells.sort_by_key(|s| s.l);

    // remove_zero: drop every primitive whose contraction coefficients are all
    // zero (`any(c != 0 ...)`); drop a shell left with no primitives.
    let mut out = Vec::with_capacity(shells.len());
    for s in shells {
        let nprim = s.exponents.len();
        let keep: Vec<usize> = (0..nprim)
            .filter(|&j| {
                s.coeffs
                    .iter()
                    .any(|col| col.get(j).is_some_and(|&c| c != 0.0))
            })
            .collect();
        if keep.is_empty() {
            continue;
        }
        let exponents = keep.iter().map(|&j| s.exponents[j]).collect();
        let coeffs = s
            .coeffs
            .iter()
            .map(|col| keep.iter().map(|&j| col[j]).collect())
            .collect();
        out.push(ShellSpec {
            l: s.l,
            exponents,
            coeffs,
        });
    }
    out
}

fn parse_err(source: &str, line: usize, reason: String) -> BasisLoadError {
    BasisLoadError::Parse {
        file: source.into(),
        line,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H SZV-GTH — single set, single l=0 shell, one contraction, 4 exponents.
    /// Values quoted verbatim from `pyscf/pbc/gto/basis/gth-szv.dat`.
    #[test]
    fn h_szv_single_shell() {
        let text = "#BASIS SET\n\
                    H SZV-GTH\n\
                    \x20 1\n\
                    \x20 1  0  0  4  1\n\
                    \x20      8.3744350009  -0.0283380461\n\
                    \x20      1.8058681460  -0.1333810052\n\
                    \x20      0.4852528328  -0.3995676063\n\
                    \x20      0.1658236932  -0.5531027541\n\
                    #\n";
        let p = parse_cp2k(text, "H", "test").unwrap();
        assert_eq!(p.shells.len(), 1);
        assert_eq!(p.shells[0].l, 0);
        assert_eq!(p.shells[0].exponents.len(), 4);
        assert_eq!(p.shells[0].coeffs.len(), 1); // one contraction column
        approx::assert_abs_diff_eq!(p.shells[0].exponents[0], 8.3744350009, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[0].exponents[3], 0.1658236932, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[0].coeffs[0][0], -0.0283380461, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[0].coeffs[0][3], -0.5531027541, epsilon = 1e-9);
    }

    /// H DZVP-GTH — two sets (l=0 with 2 contractions sharing 4 exps; l=1 with
    /// 1 exp). The l=0 second column `[0,0,0,1]` is NOT all-zero per-row, so
    /// remove_zero keeps all 4 exponents. Verbatim from `gth-dzvp.dat`.
    #[test]
    fn h_dzvp_two_sets_l_sorted() {
        let text = "#BASIS SET\n\
                    H DZVP-GTH\n\
                    \x20 2\n\
                    \x20 1  0  0  4  2\n\
                    \x20      8.3744350009  -0.0283380461   0.0000000000\n\
                    \x20      1.8058681460  -0.1333810052   0.0000000000\n\
                    \x20      0.4852528328  -0.3995676063   0.0000000000\n\
                    \x20      0.1658236932  -0.5531027541   1.0000000000\n\
                    \x20 2  1  1  1  1\n\
                    \x20      0.7270000000   1.0000000000\n\
                    #\n";
        let p = parse_cp2k(text, "H", "test").unwrap();
        assert_eq!(p.shells.len(), 2);
        // l-sorted: s shell first, then p.
        assert_eq!(p.shells[0].l, 0);
        assert_eq!(p.shells[0].exponents.len(), 4);
        assert_eq!(p.shells[0].coeffs.len(), 2);
        approx::assert_abs_diff_eq!(p.shells[0].coeffs[1][3], 1.0, epsilon = 1e-12);
        assert_eq!(p.shells[1].l, 1);
        assert_eq!(p.shells[1].exponents.len(), 1);
        approx::assert_abs_diff_eq!(p.shells[1].exponents[0], 0.7270000000, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[1].coeffs[0][0], 1.0, epsilon = 1e-12);
    }

    /// Li DZVP-GTH — the remove_zero validator. Set 1 (`1 0 1 5 3 2`) shares 5
    /// exponents across an l=0 shell (3 contractions) and an l=1 shell (2
    /// contractions). For the l=1 shell, exponents 7.26 / 2.11 / 0.0294 carry
    /// all-zero p-coefficients, so remove_zero collapses it from 5 exps to 2.
    /// Set 2 (`3 2 2 1 1`) adds a single l=2 shell. Verbatim from `gth-dzvp.dat`.
    #[test]
    fn li_dzvp_remove_zero_prunes_shared_exponents() {
        let text = "#BASIS SET\n\
                    Li DZVP-GTH\n\
                    \x20 2\n\
                    \x20 1  0  1  5  3  2\n\
                    \x20      7.2610457926  -0.2798628497   0.0501840556   0.0000000000   0.0000000000   0.0000000000\n\
                    \x20      2.1056583087  -0.4477420790   0.1084202571   0.0000000000   0.0000000000   0.0000000000\n\
                    \x20      0.6439906571  -0.3912929157   0.1448363201   0.0000000000   1.0000000000   0.0000000000\n\
                    \x20      0.0797152017  -0.0282408543  -0.4643021689   0.0000000000   0.0000000000   1.0000000000\n\
                    \x20      0.0294029590   0.0106542324  -0.6220213304   1.0000000000   0.0000000000   0.0000000000\n\
                    \x20 3  2  2  1  1\n\
                    \x20      0.1239000000   1.0000000000\n\
                    #\n";
        let p = parse_cp2k(text, "Li", "test").unwrap();
        assert_eq!(p.shells.len(), 3, "expect l=0, l=1, l=2 shells");

        // l=0: all 5 exponents survive (each contributes to a contraction).
        assert_eq!(p.shells[0].l, 0);
        assert_eq!(p.shells[0].exponents.len(), 5);
        assert_eq!(p.shells[0].coeffs.len(), 3);

        // l=1: remove_zero prunes to the two exponents with nonzero p-coeffs.
        assert_eq!(p.shells[1].l, 1);
        assert_eq!(p.shells[1].coeffs.len(), 2);
        assert_eq!(
            p.shells[1].exponents.len(),
            2,
            "remove_zero must drop the 3 all-zero p-coefficient exponents"
        );
        approx::assert_abs_diff_eq!(p.shells[1].exponents[0], 0.6439906571, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(p.shells[1].exponents[1], 0.0797152017, epsilon = 1e-9);
        // Surviving p-coefficients form the identity-like contraction block.
        approx::assert_abs_diff_eq!(p.shells[1].coeffs[0][0], 1.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(p.shells[1].coeffs[0][1], 0.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(p.shells[1].coeffs[1][0], 0.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(p.shells[1].coeffs[1][1], 1.0, epsilon = 1e-12);

        // l=2: single primitive, single contraction.
        assert_eq!(p.shells[2].l, 2);
        assert_eq!(p.shells[2].exponents.len(), 1);
        approx::assert_abs_diff_eq!(p.shells[2].exponents[0], 0.1239000000, epsilon = 1e-9);
    }

    /// Other elements' blocks are skipped; the requested element is isolated.
    #[test]
    fn skips_other_element_blocks() {
        let text = "#BASIS SET\n\
                    H SZV-GTH\n\
                    \x20 1\n\
                    \x20 1  0  0  1  1\n\
                    \x20      8.0  1.0\n\
                    #\n\
                    #BASIS SET\n\
                    He SZV-GTH\n\
                    \x20 1\n\
                    \x20 1  0  0  2  1\n\
                    \x20      13.1305278312  -0.0500802904\n\
                    \x20       4.1977275150  -0.1474339352\n\
                    #\n";
        let p = parse_cp2k(text, "He", "test").unwrap();
        assert_eq!(p.shells.len(), 1);
        assert_eq!(p.shells[0].exponents.len(), 2);
        approx::assert_abs_diff_eq!(p.shells[0].exponents[0], 13.1305278312, epsilon = 1e-9);
    }

    /// Absent element → structured Parse error (never a panic).
    #[test]
    fn missing_element_errors() {
        let text = "#BASIS SET\n\
                    H SZV-GTH\n\
                    \x20 1\n\
                    \x20 1  0  0  1  1\n\
                    \x20      8.0  1.0\n";
        let err = parse_cp2k(text, "Xe", "test").unwrap_err();
        match err {
            BasisLoadError::Parse { reason, .. } => {
                assert!(reason.contains("not found"), "{reason}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    /// A primitive row with the wrong coefficient count → completeness Parse error.
    #[test]
    fn incomplete_primitive_row_errors() {
        let text = "H SZV-GTH\n\
                    \x20 1\n\
                    \x20 1  0  0  2  1\n\
                    \x20      8.0  1.0\n\
                    \x20      1.0\n"; // missing the contraction coefficient
        let err = parse_cp2k(text, "H", "test").unwrap_err();
        match err {
            BasisLoadError::Parse { reason, .. } => {
                assert!(reason.contains("expected"), "{reason}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }
}
