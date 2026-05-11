//! pyscf-core: universal types and method traits for pyscf-rs.
//!
//! Per FOUND-02, this crate has zero compute dependencies — only
//! `thiserror`, `serde`, `tracing`. Method crates (gto/scf/dft/mp2/ccsd/
//! grad/geomopt) depend on pyscf-core for shared types (Mole, Density,
//! Energy) and method-dispatch traits (Method/Scf/KohnSham/PostScf/
//! Gradient/IntegralEngine).
//!
//! Phase 1 status: skeleton — type shells and trait signatures only.
//! Phases 2-7 fill the bodies (Phase 2: Mole geometry+basis; Phase 3:
//! Scf trait impl; Phase 4: KohnSham; Phase 5: PostScf for MP2; Phase 6:
//! PostScf for CCSD; Phase 7: Gradient).
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod amplitudes;
pub mod basis_set;
// Phase 3 plan 03-01 (SCF-13, Pitfall 4 + Pitfall 12 anchor) — pure
// eigenvector sign canonicalization extracted from upstream
// `pyscf/scf/hf.py:1349-1357`. Zero algebra dep per FOUND-02.
pub mod canonicalize;
pub mod density;
pub mod energy;
pub mod error;
pub mod mo;
pub mod mole;
pub mod traits;

pub use amplitudes::Amplitudes;
pub use basis_set::{raw_layout, Atom, BasisMeta, BasisSet, CintxNuclearModel, Shell};
pub use canonicalize::canonicalize_signs;
pub use density::Density;
pub use energy::Energy;
pub use error::{BasisLoadError, CoreError, EcpLoadError, PyscfRsError};
pub use mo::MOCoefficients;
pub use mole::{
    EcpShell, Mole, NuclearModel, ParsedAtom, ParsedBasis, ParsedEcp, ShellSpec, Unit,
};
pub use traits::{EcpEngine, Gradient, IntegralEngine, KohnSham, Method, PostScf, Scf};
