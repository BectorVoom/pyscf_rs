//! pyscf-kernels — cubecl numerical kernels.
//!
//! Phase 2 (D-04) ships the `eval_gto` AO-on-grid kernel here (plan 02-06).
//! Phase 4 adds DFT XC evaluation and grid loops. The algebra wall (Phase 1
//! ALG-06 + xtask/check_dependency_wall.rs) permits this crate to import
//! `cubecl-*`; method crates (pyscf-gto, pyscf-scf, pyscf-dft, …) must NOT.
//!
//! Wave 0 (plan 02-01) shipped the cubecl-cpu launch smoke test
//! (tests/wave0_cubecl_smoke.rs) — proof that `#[cube(launch_unchecked)]`
//! compiles + launches from inside this crate before the eval_gto port.
//! Plan 02-06 (this commit) adds the actual `eval_gto_sph` kernel + the
//! `AlgebraClient`-typed public dispatch entry point that pyscf-gto's
//! algebra-wall-friendly wrapper imports.

// `unsafe` is required for cubecl `launch_unchecked`. Allow it crate-wide
// rather than the `#![forbid(unsafe_code)]` Phase 1 stub had.

pub mod eval_gto;

pub use eval_gto::{eval_gto_sph, EvalGtoBuffers};
