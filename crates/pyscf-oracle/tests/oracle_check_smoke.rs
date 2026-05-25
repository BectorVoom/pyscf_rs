//! Smoke: every oracle_check! arm produces a sane comparison.
//! Per checker iteration 1 BLOCKER 1: all 8 arms exercised here.
//!
//! Each live-arm test (1–8) is `#[ignore]`'d by default because in-process
//! `Python::attach` (pyo3 `auto-initialize`) requires a Python shared-library
//! install — not always available in the dev environment (see 03-02-SUMMARY.md
//! "Issues Encountered"). CI runners with `python3.10+` + libpython-dev +
//! upstream `pyscf` installed run with `cargo test --features python
//! -- --include-ignored`.
//!
//! The unknown-method panic test (bottom) runs in EVERY build mode because
//! the dispatch table rejects unknown names before any Python work.

use pyscf_oracle::oracle_check;

// ── Always-on: dispatch-layer behaviour (no Python required) ──────────────

/// Unknown method dispatch — must always panic with `"unknown method"`.
#[test]
#[should_panic(expected = "unknown method")]
fn unknown_method_panics() {
    oracle_check!("nonexistent_method_xyz", pyscf_oracle::H2O_CC_PVDZ, 1.0);
}

// ── Python-feature-gated: the 8 live oracle arms ──────────────────────────

#[cfg(feature = "python")]
mod live_arms {
    use pyscf_oracle::{H2O_CC_PVDZ, H2O_TRIPLET_CCPVDZ, oracle_check};

    #[test]
    #[ignore = "requires libpython shared-lib at link time + upstream pyscf importable"]
    fn arm_1_scf_rhf_energy() {
        oracle_check!("scf_rhf_energy", H2O_CC_PVDZ, 1e-6);
    }

    #[test]
    #[ignore = "requires libpython shared-lib at link time"]
    fn arm_2_scf_uhf_energy() {
        oracle_check!("scf_uhf_energy", H2O_TRIPLET_CCPVDZ, 1e-6);
    }

    #[test]
    #[ignore = "requires libpython shared-lib at link time"]
    fn arm_3_scf_diis_iter_count() {
        oracle_check!("scf_diis_iter_count", H2O_CC_PVDZ, 1.0);
    }

    #[test]
    #[ignore = "requires libpython shared-lib at link time + '1e' init-guess body"]
    fn arm_4_scf_init_guess_1e() {
        oracle_check!("scf_init_guess", "h2o_ccpvdz_1e", 1e-12);
    }

    #[test]
    #[ignore = "requires libpython shared-lib at link time"]
    fn arm_5_df_hf_energy() {
        oracle_check!("df_hf_energy", H2O_CC_PVDZ, 1e-6);
    }

    #[test]
    #[ignore = "requires libpython shared-lib at link time — empirical h5py↔hdf5-metno seal"]
    fn arm_6_chkfile_roundtrip() {
        oracle_check!("chkfile_roundtrip", H2O_CC_PVDZ, 1e-12);
    }

    #[test]
    #[ignore = "requires libpython shared-lib at link time"]
    fn arm_7_mulliken_pop() {
        oracle_check!("mulliken_pop", H2O_CC_PVDZ, 1e-8);
    }

    #[test]
    #[ignore = "requires libpython shared-lib at link time"]
    fn arm_8_dip_moment() {
        oracle_check!("dip_moment", H2O_CC_PVDZ, 1e-8);
    }

    /// Tolerance-enforcement: a 1e-30 tolerance on RHF energy is far below
    /// any feasible numerical match; the `Diff` error path MUST panic at
    /// the macro site (`oracle_check failed`).
    #[test]
    #[ignore = "requires libpython shared-lib + intentional tolerance failure"]
    #[should_panic(expected = "oracle_check failed")]
    fn tolerance_enforcement_panics_on_diff() {
        oracle_check!("scf_rhf_energy", H2O_CC_PVDZ, 1e-30);
    }
}
