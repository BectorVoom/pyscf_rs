//! MOLOPT GTH basis loading for the transition metals the plain `gth-*` tables
//! do not reach — specifically **Y** and **Ta**.
//!
//! # Why this file exists
//!
//! `gth-szv.dat` / `gth-dzvp.dat` carry no block for Y or Ta, so no periodic
//! fixture built from them can contain either element. Upstream's answer is the
//! MOLOPT family: `pyscf/gto/basis/__init__.py:438-446` maps six MOLOPT names
//! into the same `GTH_ALIAS` table as the plain sets, pointing at standalone
//! one-block-per-element `.dat` files under `pyscf/pbc/gto/basis/`. Those six
//! entries were absent from this port's `alias::build_gth_alias`, which made
//! `Cell.basis = "gth-szv-molopt-sr"` an `UnknownName` error.
//!
//! The numbers asserted below are quoted VERBATIM from
//! `pyscf/pbc/gto/basis/gth-szv-molopt-sr.dat` (Y at line 354, Ta at line 543)
//! and `pyscf/pbc/gto/pseudo/gth-pade.dat` (Y at 605, Ta at 1140), so the test
//! fails if the alias ever resolves to a different file or a different block.
//!
//! # The q-channel trap this pins down
//!
//! Both elements have MORE THAN ONE block in the pseudopotential file — Y has
//! `q11` and `q3`, Ta has `q13` and `q5` — and the basis is only valid against
//! the matching valence count. The bare `GTH-PADE` alias is attached to the
//! `q11` / `q13` blocks, and `SZV-MOLOPT-SR-GTH` is the `q11` / `q13` basis, so
//! the pair is consistent; `nelec.iter().sum()` is asserted to prove the
//! loader did not silently pick up the small-core block.

use pyscf_gto::basis::{load_basis, load_pseudo};

/// Angular momenta present, and the `(n_contractions, n_primitives)` shape of
/// each shell's contraction matrix.
fn shape(parsed: &pyscf_core::ParsedBasis) -> Vec<(u8, usize, usize)> {
    parsed
        .shells
        .iter()
        .map(|s| (s.l, s.coeffs.len(), s.exponents.len()))
        .collect()
}

/// `gth-szv-molopt-sr` must resolve, and must hand back the Y `q11` block.
///
/// The file's component line is `2 0 2 7 2 1 1` — one set, seven shared
/// exponents, `l = 0..=2` with 2 / 1 / 1 contractions. After the `remove_zero`
/// pass that leaves three shells (s with two contractions, p, d) = 10 AOs.
#[test]
fn yttrium_loads_from_gth_szv_molopt_sr() {
    let b = load_basis("gth-szv-molopt-sr", "Y").expect("Y must load from gth-szv-molopt-sr.dat");

    assert_eq!(
        shape(&b),
        vec![(0, 2, 7), (1, 1, 7), (2, 1, 7)],
        "Y SZV-MOLOPT-SR-GTH is one 7-primitive set spanning l = 0..=2 with \
         2/1/1 contractions"
    );

    // First and last exponents, verbatim from gth-szv-molopt-sr.dat:354.
    let e = &b.shells[0].exponents;
    assert_eq!(e[0], 7.169829557085, "Y leading exponent");
    assert_eq!(e[6], 0.041587097849, "Y trailing exponent");
    // First s contraction column, first row.
    assert_eq!(b.shells[0].coeffs[0][0], 0.064756738089, "Y s(1) c(1)");
    // The d column is the LAST of the seven coefficient columns.
    assert_eq!(b.shells[2].coeffs[0][0], -0.000498514326, "Y d c(1)");
}

/// Same for Ta (`q13`). Distinct from Y only in the numbers, but a second
/// element proves the block selector is not returning the file's first match.
#[test]
fn tantalum_loads_from_gth_szv_molopt_sr() {
    let b = load_basis("gth-szv-molopt-sr", "Ta").expect("Ta must load from gth-szv-molopt-sr.dat");

    // Component line `2 0 2 6 2 1 1` — SIX shared exponents, not Y's seven.
    // Asserting the count separately from Y is what proves the block selector
    // walked to the Ta block instead of reusing an earlier element's.
    assert_eq!(
        shape(&b),
        vec![(0, 2, 6), (1, 1, 6), (2, 1, 6)],
        "Ta SZV-MOLOPT-SR-GTH is a 6-primitive set with 2/1/1 contractions"
    );

    // Verbatim from gth-szv-molopt-sr.dat:543.
    let e = &b.shells[0].exponents;
    assert_eq!(e[0], 2.946055777701, "Ta leading exponent");
    assert_eq!(e[5], 0.065633603795, "Ta trailing exponent");
    assert_eq!(b.shells[0].coeffs[0][0], 0.481243902001, "Ta s(1) c(1)");
    assert_eq!(b.shells[2].coeffs[0][0], -0.093115240785, "Ta d c(1)");
}

/// Y and Ta must NOT be reachable from the plain `gth-szv` table — that is the
/// whole reason the MOLOPT entries had to be added. If upstream ever grows
/// those blocks, this test fails and the fixture can be simplified.
#[test]
fn plain_gth_szv_still_has_no_y_or_ta() {
    for symbol in ["Y", "Ta"] {
        let err = pyscf_gto::basis::load_basis_local("gth-szv", symbol)
            .expect_err("gth-szv.dat carries no transition-metal block");
        // The CP2K parser reports an absent element as a `Parse` failure from
        // inside the block scan, not as the loader-level `ElementAbsent` (which
        // only fires for a file that parsed to zero shells). Either way the
        // element is unreachable, which is the property under test.
        assert!(
            matches!(
                err,
                pyscf_core::BasisLoadError::Parse { .. }
                    | pyscf_core::BasisLoadError::ElementAbsent { .. }
            ),
            "{symbol}: expected a miss from gth-szv, got {err:?}"
        );
    }
}

/// The `gth-pade` pseudopotential must resolve to the LARGE-core-charge block
/// that the MOLOPT basis was optimised against: Y `q11`, Ta `q13`.
#[test]
fn gth_pade_gives_y_q11_and_ta_q13() {
    let y = load_pseudo("gth-pade", "Y").expect("Y GTH-PADE");
    assert_eq!(
        y.nelec.iter().sum::<u32>(),
        11,
        "bare GTH-PADE for Y must be the q11 block, not q3 — the q3 block is \
         also in the file and would silently halve the valence charge"
    );

    let ta = load_pseudo("gth-pade", "Ta").expect("Ta GTH-PADE");
    assert_eq!(
        ta.nelec.iter().sum::<u32>(),
        13,
        "bare GTH-PADE for Ta must be the q13 block, not q5"
    );
}

/// All six newly added MOLOPT aliases must resolve to a file that exists and
/// parses. Carbon is in every one of them, so it is the probe element.
#[test]
fn every_molopt_alias_resolves() {
    for name in [
        "gth-szv-molopt",
        "gth-dzvp-molopt",
        "gth-tzvp-molopt",
        "gth-tzv2p-molopt",
        "gth-szv-molopt-sr",
        "gth-dzvp-molopt-sr",
    ] {
        let b = load_basis(name, "C").unwrap_or_else(|e| panic!("{name} / C must load: {e:?}"));
        assert!(!b.shells.is_empty(), "{name} / C parsed to zero shells");
    }
}
