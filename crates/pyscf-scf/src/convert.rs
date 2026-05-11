//! Cross-module dispatch helpers (SCF-11) — to_rhf / to_uhf / to_ghf
//! and the to_uks / to_rks stubs that defer to Phase 4 DFT.
//!
//! Source: `pyscf/scf/hf.py:2272-2300` — `def to_rhf`, `def to_uhf`,
//! `def to_ghf` via `addons.convert_to_*`. The Phase 3 ports copy
//! scalar SCF state across the target struct's matching fields. MO
//! coefficients are NOT copied (the conversion would need an alpha/beta
//! reconstruction that lives in plan 03-04+).

use crate::{GHF, RHF, UHF};
use pyscf_core::PyscfRsError;

/// Build an RHF instance from a UHF. Copies the convergence + DIIS +
/// damping scalar settings; MO coefficients are NOT copied (the
/// alpha/beta → restricted projection is not part of plan 03-11's
/// surface — plan 03-04+ may add a real projection if needed).
pub fn to_rhf(uhf: &UHF) -> Result<RHF, PyscfRsError> {
    let mut rhf = RHF::new(uhf.mol.clone());
    rhf.conv_tol = uhf.conv_tol;
    rhf.conv_tol_grad = uhf.conv_tol_grad;
    rhf.max_cycle = uhf.max_cycle;
    rhf.diis = uhf.diis;
    rhf.diis_space = uhf.diis_space;
    rhf.diis_start_cycle = uhf.diis_start_cycle;
    rhf.diis_damp = uhf.diis_damp;
    rhf.level_shift = uhf.level_shift;
    rhf.damp = uhf.damp;
    rhf.direct_scf = uhf.direct_scf;
    rhf.direct_scf_tol = uhf.direct_scf_tol;
    rhf.init_guess = uhf.init_guess.clone();
    rhf.verbose = uhf.verbose;
    rhf.max_memory = uhf.max_memory;
    Ok(rhf)
}

/// Build a UHF instance from an RHF. Copies the convergence + DIIS +
/// damping scalar settings; MO coefficients are NOT copied (the
/// restricted → alpha/beta promotion is not part of plan 03-11's
/// surface — plan 03-04+ may add a real promotion).
pub fn to_uhf(rhf: &RHF) -> Result<UHF, PyscfRsError> {
    let mut uhf = UHF::new(rhf.mol.clone());
    uhf.conv_tol = rhf.conv_tol;
    uhf.conv_tol_grad = rhf.conv_tol_grad;
    uhf.max_cycle = rhf.max_cycle;
    uhf.diis = rhf.diis;
    uhf.diis_space = rhf.diis_space;
    uhf.diis_start_cycle = rhf.diis_start_cycle;
    uhf.diis_damp = rhf.diis_damp;
    uhf.level_shift = rhf.level_shift;
    uhf.damp = rhf.damp;
    uhf.direct_scf = rhf.direct_scf;
    uhf.direct_scf_tol = rhf.direct_scf_tol;
    uhf.init_guess = rhf.init_guess.clone();
    uhf.verbose = rhf.verbose;
    uhf.max_memory = rhf.max_memory;
    Ok(uhf)
}

/// Build a GHF instance from an RHF. Copies the scalar settings (no
/// 2c-spinor MO reconstruction in plan 03-11 — that's a Phase 4 follow-up).
pub fn to_ghf(rhf: &RHF) -> Result<GHF, PyscfRsError> {
    let mut ghf = GHF::new(rhf.mol.clone());
    ghf.conv_tol = rhf.conv_tol;
    ghf.conv_tol_grad = rhf.conv_tol_grad;
    ghf.max_cycle = rhf.max_cycle;
    ghf.diis = rhf.diis;
    ghf.diis_space = rhf.diis_space;
    ghf.diis_start_cycle = rhf.diis_start_cycle;
    ghf.diis_damp = rhf.diis_damp;
    ghf.level_shift = rhf.level_shift;
    ghf.damp = rhf.damp;
    ghf.direct_scf = rhf.direct_scf;
    ghf.direct_scf_tol = rhf.direct_scf_tol;
    ghf.init_guess = rhf.init_guess.clone();
    ghf.verbose = rhf.verbose;
    ghf.max_memory = rhf.max_memory;
    Ok(ghf)
}

/// `to_uks` — UHF → UKS conversion. Phase 4 DFT territory; returns
/// `NotYetImplemented{phase:4}` until the DFT crate lands.
pub fn to_uks_stub(_rhf: &RHF) -> Result<(), PyscfRsError> {
    Err(PyscfRsError::NotYetImplemented {
        phase: 4,
        what: "to_uks (UKS target lands in Phase 4 DFT)",
    })
}

/// `to_rks` — RHF → RKS conversion. Phase 4 DFT territory.
pub fn to_rks_stub(_rhf: &RHF) -> Result<(), PyscfRsError> {
    Err(PyscfRsError::NotYetImplemented {
        phase: 4,
        what: "to_rks (RKS target lands in Phase 4 DFT)",
    })
}
