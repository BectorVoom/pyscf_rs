//! Closed-shell CCSD intermediates — `cc_Foo`/`cc_Fvv`/`cc_Fov`/`cc_Woooo`/
//! `cc_Wvvvv`/`cc_Wvoov`/`cc_Wovvo`/`make_tau` (port of
//! `pyscf/cc/rintermediates.py:30-326`).
//!
//! **STUB (Wave 1, 06-03).** The concrete intermediate functions land with
//! `update_amps` (06-03). This file reserves the module so `lib.rs` declares
//! the full 17-file upstream-mirroring skeleton from day one.
//!
//! Reduction discipline for the eventual bodies: every contraction materializes
//! products into a `Vec` then reduces via `oracle_sum`/`oracle_dot` (NO `+=`,
//! NO `gemm`).
#![allow(dead_code)]

// Intermediate free functions (cc_Foo / cc_Fvv / cc_Fov / cc_Woooo /
// cc_Wvvvv / cc_Wvoov / cc_Wovvo / make_tau) land in Wave 1 (06-03) alongside
// `update_amps`.
