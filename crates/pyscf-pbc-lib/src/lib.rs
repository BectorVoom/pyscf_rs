//! pyscf-pbc-lib: kpts_helper, KPoints, ktensor, linalg_helper, arnoldi
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;

// Phase 9 plan 09-06 — `round_to_fbz` + `lib.cleanse`, needed by
// `pyscf_pbc_tools::lattice::round_to_cell0`.
// Phase 9 plan 09-07 — `is_zero`, `member`, `intersection`, `unique`,
// `get_kconserv`, `get_kconserv3`.
pub mod kpts_helper;

pub use kpts_helper::{
    KCONSERV_TOL, KIdx, KPT_DIFF_TOL, Kconserv, Kconserv3, UniqueKpts, gamma_point, get_kconserv,
    get_kconserv3, intersection, is_gamma_point, is_zero, member, round_to_fbz, unique,
};
// Plan 14-02 Task 2 — the wrap-around unique and the conjugation pairing that
// `gdf_builder::gen_uniq_kpts_groups` groups its 2-centre metrics by.
pub use kpts_helper::{
    ConjPair, KkGroup, group_by_conj_pairs, kk_adapted_iter, unique_with_wrap_around,
};
