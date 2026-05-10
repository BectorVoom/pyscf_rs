//! pyscf-kernels — cubecl numerical kernels.
//!
//! Phase 2 (D-04) ships the `eval_gto` AO-on-grid kernel here (plan 02-06).
//! Phase 4 adds DFT XC evaluation and grid loops. The algebra wall (Phase 1
//! ALG-06 + xtask/check_dependency_wall.rs) permits this crate to import
//! `cubecl-*`; method crates (pyscf-gto, pyscf-scf, pyscf-dft, …) must NOT.
//!
//! Wave 0 (this commit, plan 02-01): smoke-test that cubecl-cpu can launch a
//! `#[cube(launch_unchecked)]` kernel from inside this crate (proves the
//! kernel-launch shape works before plan 02-06's eval_gto investment). The
//! actual eval_gto kernel module lands in plan 02-06.

// `unsafe` is required for cubecl `launch_unchecked`. Allow it crate-wide
// rather than the `#![forbid(unsafe_code)]` Phase 1 stub had.
