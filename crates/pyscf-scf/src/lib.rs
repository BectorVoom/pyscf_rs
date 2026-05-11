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

pub mod hooks;
pub mod kernel;
pub mod kernel_impl;
pub mod init_guess;
pub mod fock;
pub mod eig;
pub mod occ;
pub mod rdm;
pub mod energy;
pub mod analyze;
pub mod convert;
pub mod scanner;
pub mod rhf;
pub mod uhf;
pub mod ghf;
pub mod error;
// Plan 03-04 — SCF DIIS adapter (FockSubspace impls DiisStorable +
// diis_step helper wired into kernel_impl::scf_loop).
pub mod diis_adapter;

pub use error::ScfError;
pub use ghf::GHF;
pub use hooks::{NoOverrides, OverrideHooks};
pub use init_guess::parse_init_guess_mode;
pub use kernel::{kernel, InitGuessMode, KernelConfig, ScfResult};
pub use rhf::RHF;
pub use uhf::UHF;

// Plan 03-11 re-exports — the `default_*` free fns + convert/analyze/
// scanner surface 03-03 deliberately omitted (bodies didn't exist yet).
pub use analyze::{analyze, dip_moment, mulliken_meta, mulliken_pop, MullikenResult};
pub use convert::{to_ghf, to_rhf, to_rks_stub, to_uhf, to_uks_stub};
pub use eig::default_eig;
pub use energy::{default_energy_elec, default_energy_tot};
pub use fock::{default_get_fock, default_get_hcore, default_get_jk, default_get_ovlp, default_get_veff};
pub use init_guess::default_get_init_guess;
pub use occ::default_get_occ;
pub use rdm::default_make_rdm1;
pub use scanner::as_scanner;

// Plan 03-04 re-exports.
pub use diis_adapter::{diis_step, FockSubspace};
