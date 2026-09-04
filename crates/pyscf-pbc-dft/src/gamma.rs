//! Gamma-point (single-k) Kohn-Sham — plan 12-05.
//!
//! Ports `pyscf/pbc/dft/rks.py:400-447` (`RKS`), `uks.py`, `roks.py` and
//! `gks.py`. Upstream keeps these as separate classes because its molecular and
//! periodic class hierarchies differ; numerically a single-k-point `KRKS` IS
//! `RKS`, and upstream's own `rks.RKS.__init__` is `pbchf.RHF.__init__(cell,
//! kpt)` + `KohnShamDFT.__init__(xc)` — the same object with `kpts = [kpt]`.
//!
//! This module is therefore constructors, not a second driver: exactly the
//! shape `pyscf_pbc_scf::gamma` already uses for the Hartree-Fock four.

use pyscf_pbc_gto::Cell;

use crate::error::PbcDftError;
use crate::kgks::Kgks;
use crate::krks::Krks;
use crate::kroks::Kroks;
use crate::kuks::Kuks;

/// `pbc.dft.RKS(cell, xc=...)` — restricted KS at gamma.
///
/// # Errors
/// Propagates the `FFTDF` and grid construction.
pub fn rks(cell: Cell, xc: &str) -> Result<Krks, PbcDftError> {
    Krks::new(cell, &[[0.0; 3]], xc)
}

/// `pbc.dft.RKS(cell, kpt=..., xc=...)` — restricted KS at ONE arbitrary
/// k-point.
///
/// # Errors
/// As [`rks`].
pub fn rks_at(cell: Cell, kpt: [f64; 3], xc: &str) -> Result<Krks, PbcDftError> {
    Krks::new(cell, &[kpt], xc)
}

/// `pbc.dft.UKS(cell, xc=...)` — unrestricted KS at gamma.
///
/// # Errors
/// As [`rks`].
pub fn uks(cell: Cell, xc: &str) -> Result<Kuks, PbcDftError> {
    Kuks::new(cell, &[[0.0; 3]], xc)
}

/// `pbc.dft.UKS(cell, kpt=..., xc=...)`.
///
/// # Errors
/// As [`rks`].
pub fn uks_at(cell: Cell, kpt: [f64; 3], xc: &str) -> Result<Kuks, PbcDftError> {
    Kuks::new(cell, &[kpt], xc)
}

/// `pbc.dft.ROKS(cell, xc=...)` — restricted open-shell KS at gamma.
///
/// # Errors
/// As [`rks`].
pub fn roks(cell: Cell, xc: &str) -> Result<Kroks, PbcDftError> {
    Kroks::new(cell, &[[0.0; 3]], xc)
}

/// `pbc.dft.GKS(cell, xc=...)` — generalized (2-component) KS at gamma.
///
/// # Errors
/// As [`rks`].
pub fn gks(cell: Cell, xc: &str) -> Result<Kgks, PbcDftError> {
    Kgks::new(cell, &[[0.0; 3]], xc)
}
