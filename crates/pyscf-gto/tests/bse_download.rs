//! Live Basis Set Exchange downloads.
//!
//! Every test here is `#[ignore]`d: it needs the `bse` feature AND network
//! egress, neither of which a default `cargo test` run can assume. Run with
//!
//! ```text
//! cargo test -p pyscf-gto --features bse --test bse_download -- --ignored
//! ```

#![cfg(feature = "bse")]

use pyscf_gto::basis::{self, bse};
use std::path::PathBuf;

/// Point the cache at scratch space so a live run never writes into the
/// developer's real `~/.cache`, and never reads a stale entry instead of
/// exercising the network.
fn use_scratch_cache() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pyscf-rs-bse-live-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // SAFETY: these tests run under `--test-threads=1` semantics in practice —
    // each sets the same value, so a race would be benign anyway.
    unsafe {
        std::env::set_var("PYSCF_BSE_CACHE_DIR", &dir);
        std::env::remove_var("PYSCF_BSE_OFFLINE");
    }
    dir
}

/// The oracle: a basis set that exists BOTH locally and at BSE must come back
/// identical. def2-SVP oxygen is bit-for-bit the same in
/// `pyscf/gto/basis/def2-svp.dat` and in the database's NWChem serialisation,
/// so any drift in the fetch/parse path shows up here as an exact mismatch.
#[test]
#[ignore = "requires network egress and --features bse"]
fn downloaded_def2_svp_oxygen_matches_the_vendored_file() {
    let dir = use_scratch_cache();

    let downloaded = bse::fetch_basis("def2-svp", "O").expect("def2-SVP O must download");
    let vendored = basis::load_basis("def2-svp", "O").expect("def2-svp.dat is vendored");

    assert_eq!(
        downloaded.shells.len(),
        vendored.shells.len(),
        "shell count differs between the download and def2-svp.dat"
    );
    for (i, (a, b)) in downloaded
        .shells
        .iter()
        .zip(vendored.shells.iter())
        .enumerate()
    {
        assert_eq!(a.l, b.l, "shell {i}: angular momentum");
        assert_eq!(a.exponents, b.exponents, "shell {i}: exponents");
        assert_eq!(a.coeffs, b.coeffs, "shell {i}: coefficients");
    }

    // The response must have landed in the cache, so the second call is free.
    let cached = bse::cache_path("def2-SVP", "O").expect("a cache path exists");
    assert!(
        cached.is_file(),
        "the fetched document should have been cached at {}",
        cached.display()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Europium: the element that motivated this feature. It is absent from every
/// vendored def2 file, so the download is the only way to get it — and it
/// arrives with the 28-electron-core ECP the basis requires.
#[test]
#[ignore = "requires network egress and --features bse"]
fn europium_arrives_with_its_ecp() {
    let dir = use_scratch_cache();

    let basis_eu = bse::fetch_basis("def2-svp", "Eu").expect("def2-SVP Eu must download");
    assert!(
        basis_eu.shells.len() > 10,
        "def2-SVP Eu should be a real valence basis, got {} shells",
        basis_eu.shells.len()
    );
    assert!(
        basis_eu.shells.iter().any(|s| s.l == 3),
        "a lanthanide valence basis must carry f functions"
    );

    let ecp = bse::fetch_ecp("def2-svp", "Eu")
        .expect("the ECP fetch must succeed")
        .expect("def2-SVP is ECP-bearing for Eu");
    assert_eq!(
        ecp.n_core, 28,
        "def2-SVP uses the 28-electron-core lanthanide ECP for Eu"
    );
    assert!(
        ecp.channels.iter().any(|c| c.l == -1),
        "the ECP must define a local (UL) channel"
    );

    // The vendored file genuinely lacks Eu — this is what the download buys.
    assert!(
        matches!(
            basis::load_basis_local("def2-svp", "Eu"),
            Err(pyscf_core::BasisLoadError::ElementAbsent { .. })
        ),
        "def2-svp.dat is not expected to carry Eu; if it now does, this test's premise is stale"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The behaviour this feature exists for: `def2-svp` IS a local basis set, but
/// it has no Eu block, so `load_basis` must carry that miss through to BSE and
/// come back with real shells.
#[test]
#[ignore = "requires network egress and --features bse"]
fn def2_svp_eu_falls_through_to_bse() {
    let dir = use_scratch_cache();

    assert!(
        basis::alias::lookup(&basis::canonicalise_basis_name("def2-svp")).is_some(),
        "def2-svp must be a LOCAL basis name, or this test proves nothing"
    );

    let eu = basis::load_basis("def2-svp", "Eu")
        .expect("a known basis missing an element must fall through to BSE");
    assert!(
        eu.shells.iter().any(|s| s.l == 3),
        "the downloaded def2-SVP Eu basis must carry f functions"
    );

    // The local path still reports the miss: the fallback lives in
    // `load_basis`, not in the file reader.
    assert!(
        basis::load_basis_local("def2-svp", "Eu").is_err(),
        "load_basis_local must never reach the network"
    );

    // Oxygen still comes off disk — the fallback must not shadow local data.
    let o = basis::load_basis("def2-svp", "O").expect("oxygen is in def2-svp.dat");
    let o_local = basis::load_basis_local("def2-svp", "O").expect("oxygen is local");
    assert_eq!(
        o.shells.len(),
        o_local.shells.len(),
        "an element the local file covers must still be served locally"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The ECP path mirrors the basis path: `def2-svp` has no Eu ECP on disk, but
/// `bse_meta.json` says the basis defines one, so `load_ecp` must fetch it.
/// Getting this wrong is not a missing-feature bug — a valence-only Eu basis
/// with no ECP describes an atom with 28 electrons too many.
#[test]
#[ignore = "requires network egress and --features bse"]
fn def2_svp_eu_ecp_falls_through_to_bse() {
    let dir = use_scratch_cache();

    // Local data alone yields nothing for Eu...
    assert!(
        basis::load_ecp_local("def2-svp", "Eu")
            .expect("the local lookup itself must not fail")
            .is_none(),
        "def2-svp.dat carries no Eu ECP"
    );

    // ...but the full path supplies it.
    let ecp = basis::load_ecp("def2-svp", "Eu")
        .expect("the ECP fetch must succeed")
        .expect("bse_meta.json says def2-SVP is ECP-bearing for Eu");
    assert_eq!(
        ecp.n_core, 28,
        "def2-SVP removes a 28-electron core from Eu"
    );
    assert!(
        ecp.channels.iter().any(|c| c.l == -1),
        "the ECP must define a local (UL) channel"
    );

    // And the basis it pairs with is the downloaded valence-only one, so the
    // two agree about how many electrons the atom has.
    let basis_eu = basis::load_basis("def2-svp", "Eu").expect("the Eu basis must download");
    assert!(basis_eu.shells.iter().any(|s| s.l == 3));

    // An all-electron pair still answers "no ECP" rather than erroring.
    assert!(
        basis::load_ecp("def2-svp", "O")
            .expect("oxygen must resolve")
            .is_none()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A name absent from the local tables AND from BSE must produce a BSE error
/// naming the failure, not a panic and not a silent empty basis.
#[test]
#[ignore = "requires network egress and --features bse"]
fn a_nonexistent_basis_reports_a_bse_failure() {
    let dir = use_scratch_cache();

    let err = basis::load_basis("definitely-not-a-basis-set-xyzzy", "H")
        .expect_err("a bogus name must not resolve");
    assert!(
        matches!(err, pyscf_core::BasisLoadError::Bse { .. }),
        "expected a Bse error once the feature is on, got {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The end-to-end path: a basis name that is NOT in any local ALIAS table
/// resolves through `load_basis`, which is where upstream puts its own BSE
/// fallback (`pyscf/gto/basis/__init__.py:699-714`).
#[test]
#[ignore = "requires network egress and --features bse"]
fn load_basis_falls_back_to_bse_for_an_unaliased_name() {
    let dir = use_scratch_cache();

    assert!(
        basis::alias::lookup(&basis::canonicalise_basis_name("ano-r2")).is_none(),
        "ano-r2 must not be in the local ALIAS table, or this test proves nothing"
    );
    let parsed = basis::load_basis("ano-r2", "Eu").expect("ano-r2 Eu must come from BSE");
    assert!(
        !parsed.shells.is_empty(),
        "the fallback must yield a non-empty basis"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
