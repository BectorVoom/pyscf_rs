//! ORACLE-08 — chkfile round-trip both directions on H2O/cc-pVDZ.
//! This IS the h5py↔hdf5-metno empirical seal (STATE.md Blockers/Concerns line 90).
//!
//! `#[ignore]` is retained because `pyo3 auto-initialize` requires a Python
//! shared-library install (libpython.so) and CPython distribution with
//! upstream `pyscf` installed; not always present in the dev environment
//! (see 03-02-SUMMARY.md). On CI runners with `python3.10+` + libpython-dev
//! + `pip install pyscf`, drop `--ignored` to run.
#![cfg(feature = "python")]

use pyscf_oracle::{oracle_check, H2O_CC_PVDZ};

#[test]
#[ignore = "requires libpython shared-lib + upstream pyscf importable"]
fn chkfile_roundtrip_h2o_ccpvdz() {
    oracle_check!("chkfile_roundtrip", H2O_CC_PVDZ, 1e-12);
}
