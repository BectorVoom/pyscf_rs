//! UCCSD α/β/αβ spin-channel kernel (port of `pyscf/cc/uccsd.py`).
//!
//! **STUB (Wave 2).** Reserves the module for the open-shell driver. The
//! spin-resolved amplitude container mirrors `pyscf_mp2::UmpAmplitudes
//! { t2aa, t2ab, t2bb }` and the reference reuses the `UmpReference
//! { alpha, beta }` two-channel shape; `e_corr = e_aa + e_bb + e_ab`. Same
//! materialize-then-`oracle_*` reduction discipline per channel.
#![allow(dead_code)]

// UCCSD driver + UccsdAmplitudes/UccsdReference land in Wave 2.
