//! pyscf-pbc-scf: SCF (gamma) + KSCF/KRHF/KUHF/KROHF/KGHF, smearing, addons, stability, newton
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub use error::*;

// Phase 11 — the periodic SCF driver and its Hartree-Fock methods
// (plans 11-09 … 11-11).
pub mod addons;
pub mod chkfile;
pub mod cphf;
pub mod gamma;
pub mod init_guess;
pub mod kdiis;
pub mod kghf;
pub mod khf_ksymm;
pub mod khooks;
pub mod kocc;
pub mod krdm;
pub mod krhf;
pub mod krohf;
pub mod kscf;
pub mod kuhf;
pub mod newton_ah;
pub mod response;
pub mod stability;
// Phase 14 — range-separated J/K with no density fitting (plan 14-08 Task 4).
// BLOCKED on the cintx short-range-integral gap; see the module docs.
pub mod rsjk;
pub mod smearing;
pub mod types;

pub use chkfile::{KScfCheckpoint, dump_kscf_to_file, load_kscf_from_file};
pub use cphf::{KCPHF_DEFAULT_MAX_CYCLE, KCPHF_DEFAULT_TOL, KFvind, KcphfInput, dense_kvind, run_kcphf};
pub use gamma::{ghf, rhf, rhf_at, rohf, uhf};
pub use kghf::Kghf;
pub use khf_ksymm::{JkRoute, KsymAdaptedKrhf};
pub use khooks::KOverrideHooks;
pub use krhf::Krhf;
pub use krohf::Krohf;
pub use kscf::kernel;
pub use kuhf::Kuhf;
pub use newton_ah::{NewtonConfig, NewtonModel, PbscfNewtonError, ah_step, kernel_newton};
pub use stability::{ExternalStability, InternalStability, PbscfStabilityError, STABILITY_THRESHOLD, rhf_external, rhf_internal, rotate_mo_real};
pub use response::{PbcKohnShamBase, ResponseError, ResponseJkBackend, RksGenResponse, resolve_jk_route};
pub use rsjk::RangeSeparatedJkBuilder;
pub use smearing::{Smearing, SmearingMethod};
pub use types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};
