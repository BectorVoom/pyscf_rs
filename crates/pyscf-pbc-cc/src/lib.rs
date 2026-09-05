//! pyscf-pbc-cc: KCCSD RHF/UHF/GHF, (T), EOM
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
// Plan 16-02 Task 3 — the k-indexed complex block container every later
// Phase-16 tensor is expressed in, over `pyscf-runtime`'s `ZWorkspacePool`.
pub mod ktensor;
// Plan 16-04 onward — the shaped planar-complex host array and the
// deterministic `einsum` every k-point CC contraction is written in.
pub mod zarr;

pub use error::*;
pub use ktensor::{KRank, KTensor, Tier};
pub use zarr::{ZArr, einsum, einsum_scaled};
