//! pyscf-scf: SCF kernels (RHF/UHF/GHF + DIIS + DF-HF + chkfile).
//!
//! Checker iteration 1 WARNING 3 split: this plan (03-03) owns the trait
//! scaffolding + 30-attribute structs + InitGuessMode declarations. Plan
//! 03-11 (Wave 3) fills the kernel cycle loop, Fock build, eig + sign
//! canonicalization, init_guess bodies, analyze/mulliken/dip, convert
//! helpers, and as_scanner.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod hooks;
pub mod kernel;
pub mod kernel_impl; // plan 03-11 fills the body
pub mod init_guess;
pub mod fock; // plan 03-11 fills the body
pub mod eig; // plan 03-11 fills the body
pub mod occ; // plan 03-11 fills the body
pub mod rdm; // plan 03-11 fills the body
pub mod energy; // plan 03-11 fills the body
pub mod analyze; // plan 03-11 fills the body
pub mod convert; // plan 03-11 fills the body
pub mod scanner; // plan 03-11 fills the body
pub mod rhf;
pub mod uhf;
pub mod ghf;
pub mod error;

pub use error::ScfError;
pub use ghf::GHF;
pub use hooks::{NoOverrides, OverrideHooks};
pub use init_guess::parse_init_guess_mode;
pub use kernel::{kernel, InitGuessMode, KernelConfig, ScfResult};
pub use rhf::RHF;
pub use uhf::UHF;
// The default_* free fns + convert/analyze/scanner re-exports move to plan 03-11.
