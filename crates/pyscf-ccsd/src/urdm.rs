//! Open-shell CCSD reduced density matrices (port of `pyscf/cc/uccsd_rdm.py`).
//!
//! The α/β/αβ spin-resolved counterpart of [`crate::rdm`]. The closed-shell RDM
//! surface (`make_rdm1`/`make_rdm2` + `gamma1_intermediates` + the `ao_repr=true`
//! nmo⁴→nao⁴ AO back-transform, D-03) ships and is numerically validated THIS
//! plan (06-06, CCSD-06); the open-shell mirror reuses the SAME discipline per
//! spin channel — host-loop `oracle_sum` reductions (NO gemm, NO bare `+=`), the
//! `ao_repr` AO back-transform as the heaviest per-channel arena tenant.
//!
//! **DEFERRED open-shell mirror (documented).** The spin-resolved UCCSD RDMs are
//! wired when the open-shell response surface is exercised (Phase-7 open-shell
//! gradients). The closed-shell `rdm.rs` is the validated reference path; this
//! module is intentionally reserved (NOT silent wrong numeric code) until an
//! open-shell RDM consumer + test lands. See the 06-06-SUMMARY `Known Stubs`.
#![allow(dead_code)]

// Open-shell make_rdm1 / make_rdm2 mirror crate::rdm per spin channel; wired
// with the Phase-7 open-shell response consumer.
