//! pyscf-pbc-tdscf: periodic TDA/TDHF/TDDFT
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod davidson;
pub mod error;
pub mod krhf;
pub mod krks;
pub mod kuhf;
pub mod kuks;
pub mod rhf;
pub mod rks;
pub mod types;
pub mod uhf;
pub mod uks;
pub use davidson::*;
pub use error::*;
pub use krhf::*;
pub use krks::*;
pub use kuhf::*;
pub use kuks::*;
pub use rhf::*;
pub use rks::*;
pub use types::*;
pub use uhf::*;
pub use uks::*;

/// Compile-time tie to 19-03's response seam (§1.6): the KS drivers consume
/// `gen_response`, which exists only on the concrete RKS
/// (`RksGenResponse`), never on the PBC base. Referencing it here keeps the
/// seam in this crate's build graph rather than in prose.
pub fn require_rks_response() -> bool {
    pyscf_pbc_scf::response::RksGenResponse::HAS_GEN_RESPONSE
}
