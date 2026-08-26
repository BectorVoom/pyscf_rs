//! pyscf-pbc-df: FFTDF, AFTDF, GDF, MDF, RSDF, ft_ao, jk builders
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;

// Phase 11 — FFTDF (plans 11-05 … 11-08).
pub mod df_jk;
pub mod fft_jk;
pub mod fftdf;
pub mod traits;
pub mod zlinalg;

pub use df_jk::{KMats, all_gamma, ewald_exxdiv_for_g0, format_kpts_band};
pub use fft_jk::{get_j_kpts, get_k_kpts};
pub use fftdf::{AoKpts, Fftdf, get_hcore, get_nuc, get_pp};
pub use traits::{JkOpts, JkResult, PeriodicDf};
