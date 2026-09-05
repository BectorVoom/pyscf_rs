//! Basis Set Exchange support that needs no network: the on-disk cache, the
//! `bse_meta.json` name/ECP table, and the offline guard.
//!
//! These run in the DEFAULT build (no `bse` feature). That is deliberate — a
//! cache hit is served before the downloader is ever consulted, so a machine
//! with a pre-seeded cache can resolve BSE basis sets from a build that has no
//! HTTP client compiled in at all.

use pyscf_gto::basis::{self, bse};
use std::path::PathBuf;

/// The def2-SVP oxygen block exactly as the Basis Set Exchange serves it at
/// `/api/basis/def2-svp/format/nwchem/?elements=O`. Verified byte-identical to
/// the vendored `pyscf/gto/basis/def2-svp.dat` oxygen block, which is what
/// `cache_hit_matches_the_local_basis_file` below asserts.
const DEF2_SVP_OXYGEN: &str = r#"#----------------------------------------------------------------------
# Basis Set Exchange
#----------------------------------------------------------------------
BASIS "ao basis" SPHERICAL PRINT
O    S
   2266.1767785             -0.53431809926E-02
    340.87010191            -0.39890039230E-01
     77.363135167           -0.17853911985
     21.479644940           -0.46427684959
      6.6589433124          -0.44309745172
O    S
      0.80975975668          1.0000000
O    S
      0.25530772234          1.0000000
O    P
     17.721504317            0.43394573193E-01
      3.8635505440           0.23094120765
      1.0480920883           0.51375311064
O    P
      0.27641544411          1.0000000
O    D
      1.2000000              1.0000000
END
"#;

/// A scratch cache directory unique to this test binary.
fn scratch_cache() -> PathBuf {
    std::env::temp_dir().join(format!("pyscf-rs-bse-offline-{}", std::process::id()))
}

/// Everything that mutates the environment lives in ONE test: integration
/// tests share a process, and `set_var` is process-global.
#[test]
fn offline_cache_serves_basis_without_network() {
    let dir = scratch_cache();
    let _ = std::fs::remove_dir_all(&dir);

    // SAFETY: single-threaded section of a single-test binary; no other thread
    // is reading the environment concurrently.
    unsafe {
        std::env::set_var("PYSCF_BSE_CACHE_DIR", &dir);
        std::env::set_var("PYSCF_BSE_OFFLINE", "1");
    }

    // --- the name table -------------------------------------------------
    // `bse_meta.json` maps the pyscf alias to the database's own spelling.
    assert_eq!(
        bse::official_name("def2-svp"),
        "def2-SVP",
        "bse_meta.json must resolve the pyscf alias to the official BSE name"
    );
    assert_eq!(
        bse::official_name("cc-pVDZ"),
        "cc-pVDZ",
        "resolution must be case- and dash-insensitive on the way in"
    );
    // A name with no BSE counterpart passes through untouched, so the database
    // gets a chance to normalise it itself.
    assert_eq!(bse::official_name("not-a-real-basis"), "not-a-real-basis");

    // def2-SVP uses an ECP from Rb (37) up, Eu (63) included. This is the table
    // upstream consults in `bse_predefined_ecp` (`pyscf/gto/mole.py:4317`).
    let ecp_z = bse::ecp_elements("def2-svp").expect("def2-svp is in bse_meta.json");
    assert!(
        ecp_z.contains(&63),
        "def2-SVP must be recorded as carrying an ECP for Eu (Z=63); got {ecp_z:?}"
    );
    assert!(
        !ecp_z.contains(&8),
        "def2-SVP is all-electron for oxygen; it must not be listed as ECP-bearing"
    );

    // --- a cache miss under PYSCF_BSE_OFFLINE fails, and says why ---------
    let miss = bse::fetch_basis("def2-svp", "O")
        .expect_err("an empty cache with PYSCF_BSE_OFFLINE set must not succeed");
    let msg = miss.to_string();
    assert!(
        msg.contains("PYSCF_BSE_OFFLINE"),
        "the offline miss must name the variable holding the download back: {msg}"
    );

    // --- seed the cache, then the same call succeeds ----------------------
    let path = bse::cache_path("def2-SVP", "O").expect("PYSCF_BSE_CACHE_DIR gives a cache path");
    assert!(
        path.starts_with(&dir),
        "the cache path must sit under PYSCF_BSE_CACHE_DIR: {}",
        path.display()
    );
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, DEF2_SVP_OXYGEN).unwrap();

    let from_cache = bse::fetch_basis("def2-svp", "O")
        .expect("a seeded cache entry must be served even with the network refused");

    // --- and it agrees with the vendored file, shell for shell ------------
    let from_disk = basis::load_basis("def2-svp", "O").expect("def2-svp is a local basis set");
    assert_eq!(
        from_cache.shells.len(),
        from_disk.shells.len(),
        "BSE and the vendored def2-svp.dat must give oxygen the same shell count"
    );
    for (i, (a, b)) in from_cache
        .shells
        .iter()
        .zip(from_disk.shells.iter())
        .enumerate()
    {
        assert_eq!(a.l, b.l, "shell {i}: angular momentum differs");
        assert_eq!(
            a.exponents, b.exponents,
            "shell {i}: exponents differ between BSE and def2-svp.dat"
        );
        assert_eq!(
            a.coeffs, b.coeffs,
            "shell {i}: contraction coefficients differ between BSE and def2-svp.dat"
        );
    }

    unsafe {
        std::env::remove_var("PYSCF_BSE_CACHE_DIR");
        std::env::remove_var("PYSCF_BSE_OFFLINE");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The two misses that route to BSE must be distinguishable, and neither may
/// come back as a silently empty basis. `def2-svp` is a known name with no Eu
/// block; `ano-r2` is not a local name at all. Needs no network — the local
/// resolution happens before any fetch is considered.
#[test]
fn the_two_local_misses_are_reported_distinctly() {
    let absent =
        basis::load_basis_local("def2-svp", "Eu").expect_err("def2-svp.dat carries no Eu block");
    match &absent {
        pyscf_core::BasisLoadError::ElementAbsent { name, symbol, .. } => {
            assert_eq!(name, "def2svp");
            assert_eq!(symbol, "EU");
        }
        other => panic!("a known basis missing an element must be ElementAbsent, got {other:?}"),
    }

    let unknown = basis::load_basis_local("ano-r2", "Eu")
        .expect_err("ano-r2 is not in any local ALIAS table");
    assert!(
        matches!(unknown, pyscf_core::BasisLoadError::UnknownName { .. }),
        "an unaliased name must be UnknownName, got {unknown:?}"
    );

    // And the element the file DOES cover still resolves locally.
    assert!(
        !basis::load_basis_local("def2-svp", "O")
            .expect("oxygen is in def2-svp.dat")
            .shells
            .is_empty()
    );
}

/// The ECP fallback must be gated by `bse_meta.json`, not attempted blindly.
/// An all-electron basis/element pair has to resolve to "no ECP" from local
/// data alone — otherwise every light element of every calculation would open
/// a network connection just to be told there is no ECP.
#[test]
fn an_all_electron_pair_resolves_locally_with_no_fetch() {
    // Belt and braces: even if something tried to fetch, this would refuse it.
    unsafe {
        std::env::set_var("PYSCF_BSE_OFFLINE", "1");
    }

    // sto-3g has no ECP section at all.
    assert!(
        basis::load_ecp("sto-3g", "H")
            .expect("an all-electron basis must resolve without the network")
            .is_none()
    );
    // def2-svp IS ECP-bearing, but not for oxygen — the metadata knows that, so
    // this must also stay local.
    assert!(
        basis::load_ecp("def2-svp", "O")
            .expect("def2-svp is all-electron for oxygen")
            .is_none()
    );
    // ...and lanl2dz genuinely carries one for Na, straight off disk.
    let na = basis::load_ecp("lanl2dz", "Na")
        .expect("lanl2dz.dat carries a Na ECP")
        .expect("Na is ECP-bearing under lanl2dz");
    assert_eq!(na.n_core, 10, "LANL2DZ removes a 10-electron core from Na");

    unsafe {
        std::env::remove_var("PYSCF_BSE_OFFLINE");
    }
}

/// The periodic table must reach past krypton now, and agree with itself in
/// both directions. Eu is the case that motivated widening it: before, every
/// element above Z=36 was rejected outright by `atom_symbol`.
#[test]
fn the_element_table_covers_the_whole_periodic_table() {
    assert_eq!(
        pyscf_core::ELEMENTS.len(),
        119,
        "118 elements plus the ghost"
    );
    assert_eq!(pyscf_core::ELEMENTS[63], "Eu");
    assert_eq!(pyscf_core::ELEMENTS[118], "Og");

    for (z, sym) in pyscf_core::ELEMENTS.iter().enumerate().skip(1) {
        assert_eq!(
            pyscf_core::charge_for_symbol(sym),
            Some(z as i32),
            "symbol {sym} must map back to Z={z}"
        );
    }

    // Case-insensitive, suffix-tolerant, ghost-aware — the behaviours
    // `atom_symbol` and the ECP gate rely on.
    assert_eq!(pyscf_core::charge_for_symbol("EU"), Some(63));
    assert_eq!(pyscf_core::charge_for_symbol("Cu1"), Some(29));
    assert_eq!(pyscf_core::charge_for_symbol("GHOST"), Some(0));
    assert_eq!(pyscf_core::charge_for_symbol("Zz"), None);
}

/// Without the `bse` feature the loader must still fail cleanly, surfacing the
/// local miss rather than a network error.
#[test]
#[cfg(not(feature = "bse"))]
fn a_local_miss_without_the_feature_keeps_the_local_error() {
    assert!(
        !bse::is_available(),
        "a default build must report BSE as unavailable"
    );
    let err =
        basis::load_basis("ano-r2", "Eu").expect_err("ano-r2 is not in any local ALIAS table");
    assert!(
        matches!(err, pyscf_core::BasisLoadError::UnknownName { .. }),
        "expected UnknownName without the bse feature, got {err:?}"
    );

    let absent = basis::load_basis("def2-svp", "Eu")
        .expect_err("def2-svp has no Eu and this build cannot download it");
    assert!(
        matches!(absent, pyscf_core::BasisLoadError::ElementAbsent { .. }),
        "expected ElementAbsent without the bse feature, got {absent:?}"
    );
}
