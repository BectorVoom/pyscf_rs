//! Cross-module dispatch helpers (SCF-11) — to_rhf / to_uhf / to_ghf
//! and the to_uks / to_rks stubs that defer to Phase 4 DFT.
//!
//! Source: `pyscf/scf/hf.py:2272-2300` — `def to_rhf`, `def to_uhf`,
//! `def to_ghf` via `addons.convert_to_*`. The Phase 3 ports copy
//! scalar SCF state across the target struct's matching fields. MO
//! coefficients are NOT copied (the conversion would need an alpha/beta
//! reconstruction that lives in plan 03-04+).

use crate::{GHF, RHF, UHF};
use pyscf_core::{Mole, PyscfRsError};

/// The carried-over state for an SCF → KS (Kohn-Sham) conversion (RHF→RKS /
/// UHF→UKS). `pyscf-scf` cannot construct a `pyscf_dft::RKS`/`UKS` directly
/// (that would invert the dependency edge — pyscf-dft depends ON pyscf-scf,
/// so a back-dep is a cycle). Instead `to_uks`/`to_rks` return this descriptor
/// of the mol + scalar SCF settings; the PyO3 boundary (pyscf-py, which DOES
/// depend on both crates) assembles the real `pyscf_dft::UKS`/`RKS` target
/// from it (plan 04-09 wires the `PyRHF::to_rks` / `PyUHF::to_rks` surface).
///
/// This replaces the Phase 3 `NotYetImplemented{phase:4}` stubs now that the
/// DFT crate has landed (DFT-01, plan 04-06).
#[derive(Debug, Clone)]
pub struct KsConversion {
    /// The molecule (carried verbatim).
    pub mol: Mole,
    /// Convergence + DIIS + damping scalar settings copied from the source
    /// SCF object (MO coefficients are NOT copied — same convention as the
    /// SCF↔SCF `to_rhf`/`to_uhf`/`to_ghf` ports above).
    pub conv_tol: f64,
    pub conv_tol_grad: Option<f64>,
    pub max_cycle: u32,
    pub diis: bool,
    pub diis_space: u32,
    pub diis_start_cycle: u32,
    pub diis_damp: f64,
    pub level_shift: f64,
    pub damp: f64,
    pub direct_scf: bool,
    pub direct_scf_tol: f64,
    pub init_guess: String,
    pub verbose: u8,
    pub max_memory: f64,
}

impl KsConversion {
    /// Build the conversion descriptor from an `RHF` (RHF → RKS path).
    fn from_rhf(rhf: &RHF) -> Self {
        Self {
            mol: rhf.mol.clone(),
            conv_tol: rhf.conv_tol,
            conv_tol_grad: rhf.conv_tol_grad,
            max_cycle: rhf.max_cycle,
            diis: rhf.diis,
            diis_space: rhf.diis_space,
            diis_start_cycle: rhf.diis_start_cycle,
            diis_damp: rhf.diis_damp,
            level_shift: rhf.level_shift,
            damp: rhf.damp,
            direct_scf: rhf.direct_scf,
            direct_scf_tol: rhf.direct_scf_tol,
            init_guess: rhf.init_guess.clone(),
            verbose: rhf.verbose,
            max_memory: rhf.max_memory,
        }
    }
}

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

/// `to_uks` — RHF → UKS conversion (DFT-11). Phase 4 DFT has landed
/// (plan 04-06), so this no longer returns `NotYetImplemented{phase:4}`: it
/// produces the [`KsConversion`] descriptor (mol + scalar SCF settings) that
/// the PyO3 boundary (pyscf-py) assembles into a real `pyscf_dft::UKS`. The
/// SCF crate cannot name `pyscf_dft` (dependency-cycle wall), so the descriptor
/// is the seam.
pub fn to_uks(rhf: &RHF) -> Result<KsConversion, PyscfRsError> {
    Ok(KsConversion::from_rhf(rhf))
}

/// `to_rks` — RHF → RKS conversion (DFT-11). See [`to_uks`]; produces the
/// [`KsConversion`] descriptor the PyO3 boundary turns into a `pyscf_dft::RKS`.
pub fn to_rks(rhf: &RHF) -> Result<KsConversion, PyscfRsError> {
    Ok(KsConversion::from_rhf(rhf))
}

/// Phase 3 compatibility aliases for the old stub names. Now that the DFT
/// crate has landed these forward to the real [`to_uks`]/[`to_rks`] (no longer
/// `NotYetImplemented{phase:4}`) — kept so existing call sites referencing the
/// `_stub` names compile during the 04-09 transition.
pub fn to_uks_stub(rhf: &RHF) -> Result<KsConversion, PyscfRsError> {
    to_uks(rhf)
}

/// See [`to_uks_stub`].
pub fn to_rks_stub(rhf: &RHF) -> Result<KsConversion, PyscfRsError> {
    to_rks(rhf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyscf_core::Unit;
    use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};

    fn h2o() -> Mole {
        M(MoleBuildArgs {
            atom: AtomInput::String("O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24".into()),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Ang,
            ..Default::default()
        })
        .expect("build H2O")
    }

    /// DFT-11: to_rks/to_uks no longer return NotYetImplemented{phase:4};
    /// they carry the scalar SCF settings into a KsConversion descriptor.
    #[test]
    fn to_rks_to_uks_carry_settings_not_nyi() {
        let mut rhf = RHF::new(h2o());
        rhf.conv_tol = 1e-11;
        rhf.max_cycle = 77;
        rhf.diis_space = 12;
        rhf.init_guess = "1e".to_string();

        let rks_conv = to_rks(&rhf).expect("to_rks must not be NYI");
        assert_eq!(rks_conv.conv_tol, 1e-11);
        assert_eq!(rks_conv.max_cycle, 77);
        assert_eq!(rks_conv.diis_space, 12);
        assert_eq!(rks_conv.init_guess, "1e");

        let uks_conv = to_uks(&rhf).expect("to_uks must not be NYI");
        assert_eq!(uks_conv.conv_tol, 1e-11);
        assert_eq!(uks_conv.max_cycle, 77);

        // The compat alias names forward to the real impls.
        assert!(to_rks_stub(&rhf).is_ok());
        assert!(to_uks_stub(&rhf).is_ok());
    }
}
