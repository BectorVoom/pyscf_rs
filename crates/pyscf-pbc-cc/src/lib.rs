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
// Plan 16-05 Task 1 — the seven k-point MO integral blocks, their three
// storage tiers, and the exxdiv/Madelung treatment they must ship with.
pub mod keris;
// Plan 16-04 — the k-point restricted CC intermediates 16-05's `update_amps`
// contracts against.
pub mod kintermediates_rhf;
// Plan 16-05 — KRCCSD itself: update_amps, energy, init_amps, the DIIS kernel.
pub mod kccsd_rhf;
// Plan 16-08 Task 1 — the loop-explicit (T) reference, ported BEFORE either
// fast path because it is the only oracle-free gate the blocked path has.
// Plan 16-07 — KGCCSD, the spin-orbital k-point coupled cluster.
pub mod kccsd;
pub mod kccsd_t_rhf_slow;
pub mod kintermediates;
// Plan 16-08 Task 3 — the SPIN-ORBITAL (T), on 16-07's KGCCSD amplitudes.
pub mod kccsd_t;
// Plan 16-08 Task 2 — the BLOCKED (T), replacing kccsd_t_rhf.py:236's C kernel.
/// EOM-CCSD over spin orbitals at k-points (16-09).
pub mod eom_kccsd_ghf;
/// EOM-CCSD over spin-adapted k-point orbitals (16-10).
pub mod eom_kccsd_rhf;
pub mod kccsd_t_rhf;
/// Unrestricted k-point CCSD (16-06 Tasks 3-5).
pub mod kccsd_uhf;
/// The unrestricted k-point CC intermediates (16-06 Task 2).
pub mod kintermediates_uhf;
/// The KUCCSD one-particle density matrix (16-12).
pub mod kuccsd_rdm;
/// The unrestricted k-point ERIs (16-06 Task 1).
pub mod kueris;

pub use eom_kccsd_ghf::{EomImds, ip_vector_size};
pub use error::*;
pub use kccsd::{GBlk, KgEris, Kgccsd};
pub use kccsd_rhf::{KrccsdOpts, KrccsdResult, LARGE_DENOM};
pub use kccsd_uhf::{Kuccsd, KuccsdResult};
pub use keris::{Blk, ErisMethod, KEris, KErisOpts, adjust_occ};
pub use kintermediates_rhf as imdk;
pub use ktensor::{KBlocks, KRank, KTensor, Tier};
pub use kuccsd_rdm::{Gamma1, gamma1_intermediates, make_rdm1_from_gamma1};
pub use kueris::{KuEris, UBlk, UFock, UKind, UPass};
pub use zarr::{ZArr, einsum, einsum_scaled};
