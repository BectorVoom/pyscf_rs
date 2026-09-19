//! pyscf-pbc-grad: periodic gradients + stress tensor
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub mod gamma_rhf;
pub mod gamma_uhf;
pub mod gradients;
pub mod krhf;
pub mod krks;
pub mod kuhf;
pub mod kuks;
pub mod scanner;
pub mod stress;
pub mod tagged_dm;
pub mod verify_fd;
pub use error::*;
pub use gradients::*;
pub use krhf::{
    HcoreFusedStats, HcoreTables, KrhfGradients, assemble_hcore_deriv, fused_local_contraction,
    get_hcore, hcore_deriv_matrices, make_rdm1e_kpts, precompute_hcore, vloc_g_atom,
};
pub use kuhf::{KuhfGradients, contract_vhf_atom_spin, sum_sets};
pub use krks::KrksGradients;
pub use kuks::KuksGradients;
pub use scanner::{KrhfScannerConfig, ScfGradScanner, as_scanner, fingerprint};
pub use tagged_dm::TaggedDm;
pub use verify_fd::*;
pub mod contract;
