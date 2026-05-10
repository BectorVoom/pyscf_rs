//! GTO-03 acceptance — the lazy loader resolves real builtin .dat files
//! through the ALIAS table + `PYSCF_BASIS_PATH` walk-up resolver.
//!
//! Full sweep across all ALIAS entries is gated behind `--include-ignored`
//! (per VALIDATION.md); the default test set covers the PR-CI corpus.

use pyscf_gto::basis;

#[test]
fn sto3g_resolves_for_h() {
    let parsed = basis::load_basis("sto-3g", "H").unwrap();
    assert_eq!(parsed.shells.len(), 1);
    assert_eq!(parsed.shells[0].l, 0);
    assert_eq!(parsed.shells[0].exponents.len(), 3);
}

#[test]
fn ccpvdz_resolves_for_o() {
    let parsed = basis::load_basis("cc-pvdz", "O").unwrap();
    assert!(
        parsed.shells.len() >= 4,
        "cc-pVDZ O has ≥ 4 shells, got {}",
        parsed.shells.len()
    );
}

#[test]
fn def2svp_resolves_for_c() {
    let parsed = basis::load_basis("def2-svp", "C").unwrap();
    assert!(
        parsed.shells.len() >= 3,
        "def2-SVP C has ≥ 3 shells, got {}",
        parsed.shells.len()
    );
}

#[test]
fn six_thirty_one_g_resolves_for_c() {
    // 6-31G — Pople basis in pople-basis/ subdirectory; case-sensitive
    // file name is `6-31G.dat` (uppercase G).
    let parsed = basis::load_basis("6-31g", "C").unwrap();
    assert!(
        parsed.shells.len() >= 3,
        "6-31G C has ≥ 3 shells, got {}",
        parsed.shells.len()
    );
}

#[test]
fn case_insensitive_name_resolution() {
    // canonicalise_basis_name should make "STO-3G", "sto3g", "STO3g" equivalent.
    let a = basis::load_basis("STO-3G", "H").unwrap();
    let b = basis::load_basis("sto3g", "H").unwrap();
    let c = basis::load_basis("Sto-3G", "H").unwrap();
    assert_eq!(a.shells.len(), b.shells.len());
    assert_eq!(a.shells.len(), c.shells.len());
    approx::assert_abs_diff_eq!(
        a.shells[0].exponents[0],
        b.shells[0].exponents[0],
        epsilon = 1e-12
    );
}

/// Exercise a representative subset of the ALIAS table to detect drift.
/// Full sweep behind `--include-ignored`.
#[test]
fn representative_alias_subset_resolves() {
    // Only choose (basis, sym) pairs where the file is shipped in the upstream
    // distribution AND the ALIAS table maps to a file that actually exists.
    let cases: &[(&str, &str)] = &[
        ("sto-3g", "H"),
        ("sto-3g", "C"),
        ("6-31g", "C"),
        ("cc-pvdz", "N"),
        ("cc-pvtz", "O"),
        ("def2-svp", "O"),
        ("def2-tzvp", "F"),
    ];
    for (name, sym) in cases {
        let r = basis::load_basis(name, sym);
        assert!(
            r.is_ok(),
            "load_basis({:?}, {:?}) failed: {:?}",
            name,
            sym,
            r.err()
        );
    }
}

#[test]
fn unknown_basis_name_returns_unknownname_error() {
    let r = basis::load_basis("totally-fake-basis", "H");
    match r.unwrap_err() {
        pyscf_core::BasisLoadError::UnknownName { name } => {
            assert!(name.contains("totallyfakebasis"), "{}", name);
        }
        other => panic!("expected UnknownName, got {:?}", other),
    }
}

#[test]
#[ignore = "full ALIAS sweep — runs in nightly per VALIDATION.md"]
fn full_alias_sweep_alias_count_is_at_least_30() {
    // Phase 8 ORACLE-06 is the canonical full-coverage sweep; this is the
    // Phase 2 smoke version.
    let count = basis::alias::alias_count();
    assert!(count >= 30, "ALIAS has {} entries", count);
}
