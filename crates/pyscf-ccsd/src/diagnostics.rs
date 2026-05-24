//! CCSD wavefunction diagnostics — `get_t1_diagnostic` (Frobenius norm of t1),
//! `get_d1_diagnostic`/`get_d2_diagnostic` (port of `pyscf/cc/ccsd.py:748-776`).
//!
//! **STUB (Wave 3).** Reserves the module. The D1/D2 diagnostics need
//! `pyscf_algebra::eigh` of `t1·t1ᵀ`; reductions route through `oracle_*`.
#![allow(dead_code)]

// get_t1_diagnostic / get_d1_diagnostic / get_d2_diagnostic land in Wave 3.
