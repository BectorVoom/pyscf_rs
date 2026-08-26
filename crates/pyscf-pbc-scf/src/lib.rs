//! pyscf-pbc-scf: SCF (gamma) + KSCF/KRHF/KUHF/KROHF/KGHF, smearing, addons, stability, newton
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;

// Phase 11 — the periodic SCF driver and its Hartree-Fock methods
// (plans 11-09 … 11-11).
pub mod addons;
pub mod chkfile;
pub mod gamma;
pub mod init_guess;
pub mod kdiis;
pub mod khooks;
pub mod kocc;
pub mod krdm;
pub mod kghf;
pub mod krohf;
pub mod krhf;
pub mod kuhf;
pub mod kscf;
pub mod smearing;
pub mod types;

pub use chkfile::{KScfCheckpoint, dump_kscf_to_file, load_kscf_from_file};
pub use gamma::{ghf, rhf, rhf_at, rohf, uhf};
pub use khooks::KOverrideHooks;
pub use krhf::Krhf;
pub use kghf::Kghf;
pub use krohf::Krohf;
pub use kuhf::Kuhf;
pub use kscf::kernel;
pub use smearing::{Smearing, SmearingMethod};
pub use types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};
