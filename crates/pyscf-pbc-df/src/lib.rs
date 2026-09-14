//! pyscf-pbc-df: FFTDF, AFTDF, GDF, MDF, RSDF, ft_ao, jk builders
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;

// Phase 11 — FFTDF (plans 11-05 … 11-08).
pub mod df_jk;
pub mod fft_jk;
pub mod fftdf;

// Phase 18 — the budgeted k-point AO table (18-04 Task 2, shared with 18-05)
// and the FFTDF gradient JK (18-04 Tasks 1/3/5).
pub mod ao_cache;
pub mod fft_jk_grad;

// Phase 14 — the auxiliary cell and the 3-centre double lattice sum (plan 14-01),
// and the compensated-charge GDF builder (plan 14-02).
pub mod gdf;
pub mod gdf_builder;
pub mod incore;
pub mod pp_int;

// Phase 14 — MO integrals from `cderi` and the out-of-core 3-index drivers
// (plan 14-05), and mixed density fitting (plan 14-06).
pub mod df_ao2mo;
pub mod mdf;
pub mod outcore;
// Phase 14 — the range-separation machinery (plan 14-07, sub-task 7a) and the
// user-facing RSDF class (plan 14-08). The `_RSGDFBuilder` and everything that
// needs a SHORT-RANGE integral are BLOCKED; see the module docs.
pub mod density_fit;
pub mod rsdf;
pub mod rsdf_builder;

// Phase 13 — analytic FT of AO pairs and AFTDF (plans 13-01 … 13-05).
pub mod aft_jk;
pub mod aftdf;
pub mod ft_ao;
pub mod pbc_ao2mo;
pub mod traits;
pub mod zlinalg;

pub use aftdf::Aftdf;
pub use density_fit::{DfKind, DfOpts, density_fit};
pub use df_ao2mo::{Eri, Eri7d, MoCoeff, MoKpts, PairDims};
pub use df_jk::{KMats, all_gamma, ewald_exxdiv_for_g0, format_kpts_band};
pub use fft_jk::{get_j_kpts, get_k_kpts, get_k_kpts_opts};
pub use fftdf::{AoKpts, CoulgEntry, Fftdf, get_hcore, get_nuc, get_pp};
pub use gdf::{CderiFile, Gdf, SrBlock};
pub use gdf_builder::{ETA_MIN, EtaChoice, FusedCell, auxbar, fuse_auxcell, guess_eta};
pub use incore::{Aosym, AuxCell, aux_e2, fill_2c2e, make_auxcell, make_modrho_basis};
pub use mdf::Mdf;
pub use outcore::{Aux3cFile, Blocking, Orientation, balance_segs};
pub use pbc_ao2mo::CoulGCache;
pub use pp_int::get_pp_loc_part2_kpts;
pub use rsdf::{Rsdf, get_aux_chg};
pub use rsdf_builder::{OMEGA_MIN, RsGdfBuilder};
pub use traits::{JkOpts, JkResult, PeriodicDf};
pub use ao_cache::{
    AoEvalCount, BUFFER_MULTIPLICITY, DERIV1_COMPONENTS, Deriv1Table, KGradFootprint,
    deriv1_blksize, deriv1_blksize_from_avail, deriv1_table_bytes, eval_deriv1_chunk,
    k_e1_footprint, kdiff_index, resident_k_count, resident_mb,
};
pub use fft_jk_grad::{
    GradJkResult, GradMats, KGradStats, TaggedMo, TaggedMoBlock, get_j_e1_kpts, get_k_e1_kpts,
};
