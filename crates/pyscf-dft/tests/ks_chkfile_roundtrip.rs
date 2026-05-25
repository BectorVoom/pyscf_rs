//! DFT-07 chkfile (D-06): `KsResult` chkfile persistence round-trips through
//! the Phase 3 `pyscf-chkfile` primitives (with `xc`/`grids` metadata) and is
//! h5py-readable on the upstream `/scf` schema.
//!
//! Upstream reference: `pyscf/dft/rks.py` (KS result attrs) +
//! `pyscf/scf/chkfile.py:25-42` (the SCF base schema KS extends).
//! Owning plan: 04-08 (this plan).
//! Verify commands:
//!   - always-on: `cargo test -p pyscf-dft ks_chkfile_roundtrip`
//!   - h5py seal (CI): `cargo test --features python -p pyscf-dft ks_chkfile_roundtrip -- --ignored`
//!
//! ### Two-layer test (the established 04-04/04-05/04-06 convention)
//!
//! 1. **Rust↔Rust layer (always-on).** `KsResult::dump → load` is identical
//!    (e_tot, mo_energy, mo_occ, mo_coeff F-order, AND the xc/grids metadata),
//!    and the on-disk schema carries the upstream `/scf` keys PLUS the
//!    `xc`/`grids_*` metadata. This proves the schema + F-order convention
//!    round-trips inside pyscf-rs (the Rust-only seal, like the 03-06
//!    `chkfile_dump_load` test).
//!
//! 2. **h5py layer (CI-only, `#[cfg(feature = "python")]`).** The
//!    `oracle_check!("ks_chkfile_roundtrip", H2O_CC_PVDZ, 1e-12)` arm (the
//!    ORACLE-08 harness extended to the KS result type, 04-08) seals
//!    cross-language compatibility: a PySCF DFT chkfile is pyscf-rs-readable
//!    on the shared `/scf` block, AND a pyscf-rs-written `KsResult` chkfile is
//!    h5py-readable on the upstream `/scf` schema + the `xc`/`grids` extension.
//!    Runs only under `--features python` (libpython + importable pyscf+h5py).

use pyscf_chkfile::{Checkpointable, primitives};
use pyscf_core::{Energy, MOCoefficients};
use pyscf_dft::{GridsMeta, KsResult, dump_ks_to_file, load_ks_from_file};
use pyscf_scf::ScfResult;

fn sample_ks_result() -> KsResult {
    let nao = 4;
    let nmo = 4;
    // F-order flat data: data[i + j*nao] for the (i, j) entry (Pitfall 8).
    let mut data = vec![0.0_f64; nao * nmo];
    for j in 0..nmo {
        for i in 0..nao {
            data[i + j * nao] = ((i * 3 + j + 1) as f64) * 0.1;
        }
    }
    let scf = ScfResult {
        e_tot: Energy(-76.319_876_f64),
        mo_coeff: MOCoefficients {
            nao,
            nmo,
            data: data.clone(),
            energies: vec![-18.74, -0.92, -0.47, -0.39],
            occupations: vec![2.0, 2.0, 2.0, 2.0],
        },
        mo_energy: vec![-18.74, -0.92, -0.47, -0.39],
        mo_occ: vec![2.0, 2.0, 2.0, 2.0],
        converged: true,
        cycles: 9,
    };
    KsResult::new(
        scf,
        "b3lyp",
        GridsMeta {
            level: 3,
            scheme: "nwchem".to_string(),
        },
    )
}

// ────────────────────── Rust↔Rust layer (always-on) ───────────────────────

/// `KsResult` dump → load is identical on e_tot, mo_*, AND the xc/grids
/// metadata (the DFT-07 chkfile behavior assertion).
#[test]
fn ks_result_round_trips() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    let original = sample_ks_result();
    dump_ks_to_file(path, r#"{"atom":"O 0 0 0","basis":"cc-pvdz"}"#, &original).expect("dump");
    let loaded = load_ks_from_file(path).expect("load");

    // e_tot
    assert!((loaded.scf.e_tot.0 - original.scf.e_tot.0).abs() < 1e-12);
    // xc + grids metadata (the DFT extension).
    assert_eq!(loaded.xc, original.xc);
    assert_eq!(loaded.grids, original.grids);
    // mo_energy / mo_occ
    for (a, b) in loaded.scf.mo_energy.iter().zip(&original.scf.mo_energy) {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in loaded.scf.mo_occ.iter().zip(&original.scf.mo_occ) {
        assert!((a - b).abs() < 1e-12);
    }
    // mo_coeff: F-order data must match element-wise (Pitfall 8 round-trip).
    assert_eq!(loaded.scf.mo_coeff.nao, original.scf.mo_coeff.nao);
    assert_eq!(loaded.scf.mo_coeff.nmo, original.scf.mo_coeff.nmo);
    for (k, (a, b)) in loaded
        .scf
        .mo_coeff
        .data
        .iter()
        .zip(&original.scf.mo_coeff.data)
        .enumerate()
    {
        assert!((a - b).abs() < 1e-12, "mo_coeff[{k}] mismatch: {a} vs {b}");
    }
}

/// The on-disk schema carries the upstream `/scf` keys PLUS the DFT
/// `xc`/`grids_*` metadata (source/schema assertion).
#[test]
fn schema_has_upstream_scf_keys_plus_dft_metadata() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    dump_ks_to_file(path, "{}", &sample_ks_result()).expect("dump");
    let f = primitives::open_for_read(path).expect("reopen");
    assert!(f.link_exists("mol"), "/mol missing");
    let scf = f.group("scf").expect("scf group");
    // Upstream SCF schema (pyscf/scf/chkfile.py:25-42).
    assert!(scf.link_exists("e_tot"), "/scf/e_tot missing");
    assert!(scf.link_exists("mo_energy"), "/scf/mo_energy missing");
    assert!(scf.link_exists("mo_occ"), "/scf/mo_occ missing");
    assert!(scf.link_exists("mo_coeff"), "/scf/mo_coeff missing");
    // DFT metadata extension (DFT-07 chkfile).
    assert!(scf.link_exists("xc"), "/scf/xc missing");
    assert!(scf.link_exists("grids_level"), "/scf/grids_level missing");
    assert!(scf.link_exists("grids_scheme"), "/scf/grids_scheme missing");
}

/// `Checkpointable::dump`/`load` exercised directly under a custom group name
/// (the trait surface, D-06).
#[test]
fn checkpointable_trait_dump_load_directly() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    let original = sample_ks_result();
    {
        let f = primitives::open_for_write(path).expect("create");
        primitives::write_mol(&f, "{}").expect("mol");
        let g = f.create_group("scf").expect("group");
        original.dump(&g).expect("dump");
        f.flush().expect("flush");
    }
    {
        let f = primitives::open_for_read(path).expect("read");
        let g = f.group("scf").expect("group");
        let loaded = KsResult::load(&g).expect("load");
        assert!((loaded.scf.e_tot.0 - original.scf.e_tot.0).abs() < 1e-12);
        assert_eq!(loaded.xc, original.xc);
        assert_eq!(loaded.grids, original.grids);
    }
}

/// T-04-08: `KsResult::load` returns a `ChkfileError` (never panics) on a
/// chkfile missing the DFT metadata (a malformed/partial checkpoint).
#[test]
fn load_missing_metadata_errors_not_panics() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();
    // Write only the SCF block (no xc/grids) — i.e. a vanilla SCF chkfile.
    {
        let f = primitives::open_for_write(path).expect("create");
        primitives::write_mol(&f, "{}").expect("mol");
        let g = f.create_group("scf").expect("group");
        sample_ks_result().scf.dump(&g).expect("scf dump");
        f.flush().expect("flush");
    }
    // Loading as a KsResult must fail cleanly (missing /scf/xc), not panic.
    assert!(
        load_ks_from_file(path).is_err(),
        "load of metadata-less chkfile must error, not panic"
    );
}

/// pyscf-dft adds NO `hdf5-metno` dependency of its own (D-05 sole-owner):
/// the chkfile module uses the `pyscf_chkfile::primitives` + the re-exported
/// `pyscf_chkfile::hdf5` alias. Source assertion on Cargo.toml + chkfile.rs.
#[test]
fn no_own_hdf5_metno_dep_d05_sole_owner() {
    // Scan for an actual dependency DECLARATION (a line that starts a
    // `hdf5-metno`/`hdf5_metno` dep), not a mere mention in a comment. A dep
    // line looks like `hdf5-metno = ...` or `hdf5-metno.workspace = ...` at the
    // start of a (trimmed) line.
    let cargo = include_str!("../Cargo.toml");
    let declares_hdf5 = cargo.lines().any(|line| {
        let t = line.trim_start();
        t.starts_with("hdf5-metno") || t.starts_with("hdf5_metno")
    });
    assert!(
        !declares_hdf5,
        "pyscf-dft must NOT declare an hdf5-metno dep (D-05 — pyscf-chkfile sole owner)"
    );
    let src = include_str!("../src/chkfile.rs");
    assert!(
        src.contains("pyscf_chkfile::primitives"),
        "chkfile.rs must use pyscf_chkfile::primitives (D-06 reuse)"
    );
}

// ───────────────────────── h5py layer (CI-only) ───────────────────────────

/// ORACLE-08 (extended to KS): the pyscf-rs `KsResult` chkfile is h5py-readable
/// on the upstream `/scf` schema + the `xc`/`grids` metadata, and a PySCF DFT
/// chkfile is pyscf-rs-readable. Gated on `python` + `#[ignore]` (libpython +
/// importable pyscf + h5py).
#[cfg(feature = "python")]
#[test]
#[ignore = "ORACLE-08 KS chkfile h5py seal — run on CI with --features python + libpython + upstream pyscf + h5py"]
fn ks_chkfile_roundtrip() {
    use pyscf_oracle::{H2O_CC_PVDZ, oracle_check};
    oracle_check!("ks_chkfile_roundtrip", H2O_CC_PVDZ, 1e-12);
}
