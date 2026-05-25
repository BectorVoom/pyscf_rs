//! Open-shell Λ-equations (port of `pyscf/cc/uccsd_lambda.py`).
//!
//! The α/β/αβ spin-resolved counterpart of [`crate::lambda`]. The closed-shell
//! λ surface (`solve_lambda`/`update_lambda`) ships and is numerically validated
//! THIS plan (06-06, CCSD-05); the open-shell mirror reuses the SAME discipline
//! per spin channel — every contraction a host-loop materialize-then-`oracle_sum`
//! (NO gemm, NO bare `+=`), the `wvvvv ≈ nv⁴` per-channel arena tenant reserved
//! once, the `l~t` canonical seed, the dual-criterion convergence.
//!
//! **DEFERRED open-shell mirror (documented).** The spin-resolved UCCSD λ
//! equations are wired when the open-shell response surface is exercised
//! (Phase-7 open-shell gradients / GRAD-06). The closed-shell `lambda.rs` is the
//! validated reference path; this module is intentionally reserved (NOT silent
//! wrong numeric code) until an open-shell λ consumer + test lands. See the
//! 06-06-SUMMARY `Known Stubs` section.
#![allow(dead_code)]

// Open-shell solve_lambda / update_lambda mirror crate::lambda per spin channel;
// wired with the Phase-7 open-shell response consumer.
