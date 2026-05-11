//! pyscf-df: density-fitting (DF) integrals.
//!
//! Source: D-10 — Phase 3 introduces this crate. Mirrors upstream `pyscf/df/`
//! (df.py / df_jk.py / incore.py / addons.py). Owns 3-center integrals via
//! `mol.intor('int3c2e_sph')`, 2-center integrals via `mol.intor('int2c2e_sph')`,
//! Cholesky of (P|Q) via pyscf-algebra::cholesky (host-faer per ALG-05), and
//! B-integral assembly. Public surface: `DfIntegrals { b_uvq, naux, nao }`
//! consumed uniformly by SCF / DFT / MP2 / CCSD.
//!
//! Phase 3 ships in-memory only; HDF5 spill deferred to Phase 6 (D-11).
//! Phase 3 status: empty skeleton (Plan 03-01). Plan 03-05 fills.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]
