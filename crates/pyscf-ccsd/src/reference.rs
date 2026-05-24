//! `CcsdReference` — a converged-SCF snapshot consumed by the CCSD kernel.
//!
//! Field-for-field copy of `Mp2Reference` (`crates/pyscf-mp2/src/mp2.rs:29-43`),
//! renamed `Mp2`→`Ccsd`. pyo3-free: takes plain arrays. The `mol` is
//! snapshotted from `mf.mol` by the pyscf-py bridge (D-09) so `default_ao2mo`
//! / `CcsdOverrideHooks::ao2mo` can reach the molecule without an extra
//! `&Mole` parameter.
//!
//! UCCSD (Wave 2) reuses the `UmpReference { alpha, beta }` two-channel shape
//! (`crates/pyscf-mp2/src/ump2.rs`) — two `CcsdReference`s sharing one `mol`.

use pyscf_core::{MOCoefficients, Mole};

/// Snapshot of a converged SCF reference, consumed by the CCSD kernel (D-09).
#[derive(Debug, Clone)]
pub struct CcsdReference {
    /// MO coefficients, F-order, sign-canonicalized (SCF-13).
    pub mo_coeff: MOCoefficients,
    /// MO energies (one per MO).
    pub mo_energy: Vec<f64>,
    /// MO occupations (2.0 / 0.0 for restricted).
    pub mo_occ: Vec<f64>,
    /// Converged SCF total energy (the CCSD baseline; `get_e_hf`).
    pub e_hf: f64,
    /// Whether the reference SCF converged.
    pub converged: bool,
    /// Molecule snapshot — needed by `default_ao2mo` for the intor call.
    pub mol: Mole,
}
