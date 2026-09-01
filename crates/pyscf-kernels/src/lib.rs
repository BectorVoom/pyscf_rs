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
// Plan 17-11 — v1 multigrid's real-space collocation kernel (Task 2).
pub mod multigrid_collocate;
// Plan 17-12 — v2 multigrid's pair-fused real-space collocation (Task 2).
pub mod multigrid_gspace;
pub mod multigrid_pair;
// Phase 9 plan 09-02 (v2.0 PBC milestone) — periodic device kernels. Seeded
// with K-04 `zhadamard`; see PBC-MASTER-PLAN §6 for the full inventory.
pub mod pbc;
// quick-260522-b06: precision seam. Re-exports DeviceScalar (the in-wall
// cubecl::Float bound) and documents the f64 kernel default + f32-future seam.
pub mod scalar;

pub use eval_gto::{
    EvalGtoBuffers, cart2sph_l_matrix, cart_powers, common_fac_sp, eval_gto_cart_deriv1,
    eval_gto_sph, eval_gto_sph_deriv1,
};
pub use multigrid_collocate::{PshellGridTable, collocate};
pub use multigrid_gspace::{get_gga_vrho_gs, gradient_gs};
pub use multigrid_pair::{PairSlotTable, collocate_pairs};
pub use pbc::ewald::{EWALD_G0_SENTINEL, ewald_gs_terms, ewald_rlij};
pub use pbc::gv::gv;
pub use pbc::struct_factor::struct_factor;
pub use pbc::zhadamard::zhadamard;
pub use scalar::{DeviceScalar, KernelScalar};
