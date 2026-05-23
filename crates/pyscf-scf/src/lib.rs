//! pyscf-scf: SCF kernels (RHF/UHF/GHF + DIIS + DF-HF + chkfile).
//!
//! Checker iteration 1 WARNING 3 split: plan 03-03 shipped the trait
//! scaffolding + 30-attribute structs + InitGuessMode declarations.
//! Plan 03-11 (Wave 3, this commit) fills the kernel cycle loop, Fock
//! build, eig + sign canonicalization, init_guess bodies, analyze/
//! mulliken/dip, convert helpers, and as_scanner — and re-exports the
//! `default_*` free fns + convert/analyze/scanner surface that 03-03
//! deliberately omitted because the bodies didn't yet exist.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod analyze;
pub mod convert;
pub mod eig;
pub mod energy;
pub mod error;
pub mod fock;
pub mod ghf;
pub mod hooks;
pub mod init_guess;
pub mod kernel;
pub mod kernel_impl;
pub mod occ;
pub mod rdm;
pub mod rhf;
pub mod scanner;
pub mod uhf;
// Plan 03-06 — SCF chkfile schema (impl Checkpointable for ScfResult +
// dump_scf_to_file/load_scf_from_file helpers).
pub mod chkfile;
// Plan 03-04 — DIIS adapter (FockSubspace + diis_step).
pub mod diis_adapter;
// Plan 03-05 — DF-HF entry (RHF::density_fit + DfHooks).
pub mod df_scf;

pub use error::ScfError;
pub use ghf::GHF;
pub use hooks::{NoOverrides, OverrideHooks};
pub use init_guess::parse_init_guess_mode;
pub use kernel::{InitGuessMode, KernelConfig, ScfResult, kernel};
pub use rhf::RHF;
pub use uhf::UHF;

// Plan 03-11 re-exports — the `default_*` free fns + convert/analyze/
// scanner surface 03-03 deliberately omitted (bodies didn't exist yet).
pub use analyze::{MullikenResult, analyze, dip_moment, mulliken_meta, mulliken_pop};
pub use convert::{KsConversion, to_ghf, to_rhf, to_rks, to_rks_stub, to_uhf, to_uks, to_uks_stub};
pub use eig::default_eig;
pub use energy::{default_energy_elec, default_energy_tot};
pub use fock::{
    default_get_fock, default_get_hcore, default_get_jk, default_get_ovlp, default_get_veff,
};
pub use init_guess::default_get_init_guess;
pub use occ::default_get_occ;
pub use rdm::default_make_rdm1;
pub use scanner::as_scanner;

// Plan 03-06 re-exports — top-level chkfile helpers (Checkpointable impl
// lives in `chkfile::` and is available transparently on ScfResult).
pub use chkfile::{dump_scf_to_file, load_scf_from_file};

// Plan 03-04 re-exports — DIIS adapter surface (FockSubspace + diis_step).
pub use diis_adapter::{FockSubspace, diis_step};
// Plan 03-05 re-exports — DF-HF entry (DfHooks). RHF::density_fit lives on the
// RHF impl block in df_scf.rs.
pub use df_scf::DfHooks;
