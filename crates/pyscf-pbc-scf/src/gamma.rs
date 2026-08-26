//! Gamma-point (single-k) SCF — `pbc/scf/{hf,uhf,rohf,ghf}.py`.
//!
//! # Why these are constructors, not separate drivers
//!
//! Upstream keeps a separate single-`kpt` class hierarchy (`pbc.scf.RHF` is
//! `hf.SCF`, not `khf.KSCF`) because its `kpt` attribute is a `(3,)` array
//! rather than a `(nkpts, 3)` one and NumPy shape bookkeeping differs
//! throughout. That distinction does not survive translation: in this port a
//! k-resolved quantity is a `Vec` of length `nkpts`, and `nkpts == 1` needs no
//! special case. Every gamma-point method here is therefore its k-point
//! counterpart at a one-element k-list — the same code path upstream's own
//! `_check_kpts` collapses to, and provably the same numbers (see
//! `tests/kscf.rs`'s `gamma_matches_single_kpt_krhf`).
//!
//! A non-gamma single k-point (`mf.kpt = ...`) works too: pass it to
//! [`rhf_at`].

use pyscf_core::PyscfRsError;
use pyscf_pbc_gto::Cell;

use crate::{Kghf, Krhf, Krohf, Kuhf};

/// `pbc.scf.RHF(cell)` — restricted HF at the gamma point.
///
/// # Errors
/// Propagates the `FFTDF` construction.
pub fn rhf(cell: Cell) -> Result<Krhf, PyscfRsError> {
    Krhf::new(cell, &[[0.0; 3]])
}

/// `pbc.scf.RHF(cell, kpt=...)` — restricted HF at ONE arbitrary k-point.
///
/// # Errors
/// As [`rhf`].
pub fn rhf_at(cell: Cell, kpt: [f64; 3]) -> Result<Krhf, PyscfRsError> {
    Krhf::new(cell, &[kpt])
}

/// `pbc.scf.UHF(cell)` — unrestricted HF at the gamma point.
///
/// # Errors
/// As [`rhf`].
pub fn uhf(cell: Cell) -> Result<Kuhf, PyscfRsError> {
    Kuhf::new(cell, &[[0.0; 3]])
}

/// `pbc.scf.ROHF(cell)` — restricted open-shell HF at the gamma point.
///
/// # Errors
/// As [`rhf`].
pub fn rohf(cell: Cell) -> Result<Krohf, PyscfRsError> {
    Krohf::new(cell, &[[0.0; 3]])
}

/// `pbc.scf.GHF(cell)` — generalised HF at the gamma point.
///
/// # Errors
/// As [`rhf`].
pub fn ghf(cell: Cell) -> Result<Kghf, PyscfRsError> {
    Kghf::new(cell, &[[0.0; 3]])
}
