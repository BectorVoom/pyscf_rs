//! `pyscf/pbc/dft/multigrid/multigrid.py` — v1 multigrid (plan 17-11).
//!
//! Host task-list logic and the `MultiGridNumInt` driver live here;
//! the real-space collocation kernel lives in
//! `pyscf_kernels::multigrid_collocate` (D-PBC-25 corollary / ALG-06 — this
//! crate must not name `cubecl` directly).

pub mod colloc;
pub mod numint;
pub mod tasks;

// Plan 17-12 — v2 multigrid (`MultiGridNumInt2`): pair-fused task list +
// collocation (Tasks 1/2), G-space helpers (Task 3), pseudopotential (Task
// 4) and the assembled driver (Task 5).
pub mod pair;
pub mod pp;
pub mod utils;

pub use numint::MultiGridNumInt;
pub use pair::MultiGridNumInt2;
