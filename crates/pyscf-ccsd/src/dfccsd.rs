//! DF-CCSD (`DFRCCSD`/`DFUCCSD`) — reuses the in-core `ccsd_kernel` and swaps
//! the `ao2mo`/`_add_vvvv` ERI source for the DF `vvL` B-tensor (port of
//! `pyscf/cc/dfccsd.py`; CCSD-08).
//!
//! **STUB (Wave 4).** Reserves the module. The DF B-tensor comes from
//! `pyscf_df::{cholesky_eri, default_ri}`. The `vvL` spill uses the
//! re-exported `pyscf_chkfile::hdf5` alias (NO new `hdf5-metno` dep, D-07); the
//! spill handle deletes its temp file on `Drop` (RAII, mirrors `H5TmpFile`).
#![allow(dead_code)]

// DFRCCSD / DFUCCSD + vvL block sizing + HDF5 spill land in Wave 4.
