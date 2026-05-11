//! DF-HF entry point — SCF-07 user-facing surface.
//!
//! Source: `pyscf/scf/hf.py:2165-2172` `density_fit` method (the
//! upstream impl on `SCF` base class). Plan 03-05 Task 2 ships this as:
//!
//!   1. `RHF::density_fit(auxbasis)` — pre-computes B integrals via
//!      `pyscf_df::cholesky_eri` and stores in `self.with_df` as a
//!      `Box<dyn Any + Send + Sync>` per the plan 03-03 30-attribute floor
//!      decision (D-01 trait-callback bridge).
//!   2. `DfHooks { df: &DfIntegrals }` — `OverrideHooks` impl that routes
//!      `get_jk` (and therefore `get_veff` = J - 0.5*K) through
//!      `pyscf_df::get_jk_df`, while delegating every other hook
//!      (`get_hcore`, `eig`, ...) to the default_* free fns.
//!
//! The wave-3/4 SCF cycle loop in `kernel_impl::scf_loop` calls
//! `hooks.get_jk` / `hooks.get_veff` once per cycle; with `DfHooks` in
//! place, those calls bypass `int2e_sph` entirely — closing the Phase 3
//! convergence gap (the `h2_no_overrides_converges` test remains
//! `#[ignore]`'d until plan 03-10 runs the full DF-HF energy assertion).

use pyscf_core::{CoreError, Density, Energy, MOCoefficients, Mole, PyscfRsError};
use pyscf_df::{cholesky_eri, default_jkfit, DfIntegrals};

use crate::{InitGuessMode, OverrideHooks, RHF};

impl RHF {
    /// Configure this RHF instance to use density-fitting.
    ///
    /// If `auxbasis` is `None`, falls back to `pyscf_df::default_jkfit(orbital_basis)`
    /// where `orbital_basis` is derived from `self.mol.basis` (the raw user-supplied
    /// basis name — already a `String` on `Mole`). For `BasisInput::PerElement` /
    /// `BasisInput::NwchemText` cases, the stringified Debug form won't be in
    /// `DEFAULT_AUXBASIS`, so we fall through to `"weigend"` (universal fallback).
    ///
    /// On success, `self.with_df` is `Some(Box<DfIntegrals>)`. The downstream
    /// `RHF::kernel` (plan 03-03) calls `kernel(&mol, &NoOverrides, cfg)`; to
    /// actually route through DfHooks, the PyO3 binding (plan 03-07) calls
    /// `kernel(&mol, &DfHooks { df: &df }, cfg)` instead. For pure-Rust users:
    /// downcast `self.with_df` and pass `DfHooks` directly.
    ///
    /// SCF-07 contract.
    pub fn density_fit(mut self, auxbasis: Option<&str>) -> Result<Self, PyscfRsError> {
        let basis_name_owned = extract_basis_name(&self.mol.basis);
        let aux = auxbasis.unwrap_or_else(|| default_jkfit(&basis_name_owned));
        tracing::info!(
            target: "pyscf_scf::df",
            orbital_basis = %basis_name_owned,
            auxbasis = aux,
            "RHF::density_fit pre-computing B integrals"
        );
        let df = cholesky_eri(&self.mol, aux)?;
        self.with_df = Some(Box::new(df));
        Ok(self)
    }
}

/// Extract a canonical orbital-basis name from `mol.basis` (the raw user-supplied
/// `String`). Phase 2's `pyscf_gto::build_from` stores `format!("{:?}", args.basis)`,
/// so for `BasisInput::Name("cc-pvdz")` we see `Name("cc-pvdz")`. We strip the
/// enum-variant wrapper and the surrounding quotes. For non-`Name` variants we
/// return `mol.basis` verbatim — DEFAULT_AUXBASIS lookup will miss and fall
/// through to `weigend` (universal fallback), which is the documented behaviour.
fn extract_basis_name(basis_field: &str) -> String {
    // Pattern: `Name("cc-pvdz")` → `cc-pvdz`.
    if let Some(rest) = basis_field.strip_prefix("Name(\"") {
        if let Some(name) = rest.strip_suffix("\")") {
            return name.to_string();
        }
    }
    basis_field.to_string()
}

/// `OverrideHooks` impl that routes `get_jk` (and `get_veff`) through
/// pre-computed DF integrals, while delegating every other hook to the
/// `default_*` free fns. The `'a` lifetime is the DfIntegrals borrow.
///
/// Sized + Send + Sync so parallel geometry-optimization drivers (Phase 7)
/// can share a single `DfIntegrals` across thread boundaries.
pub struct DfHooks<'a> {
    pub df: &'a DfIntegrals,
}

impl<'a> OverrideHooks for DfHooks<'a> {
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        crate::fock::default_get_hcore(mol)
    }

    fn get_ovlp(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        crate::fock::default_get_ovlp(mol)
    }

    fn get_init_guess(&self, mol: &Mole, mode: &InitGuessMode) -> Result<Density, PyscfRsError> {
        crate::init_guess::default_get_init_guess(mol, mode)
    }

    fn get_jk(&self, _mol: &Mole, dm: &Density) -> Result<(Density, Density), PyscfRsError> {
        pyscf_df::get_jk_df(dm, self.df)
    }

    fn get_veff(&self, _mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
        let (j, k) = pyscf_df::get_jk_df(dm, self.df)?;
        if j.nao != k.nao || j.data.len() != k.data.len() {
            return Err(PyscfRsError::Core(CoreError::DimensionMismatch {
                expected: j.data.len(),
                actual: k.data.len(),
            }));
        }
        // veff = J - 0.5 * K. Plan 03-04 SUMMARY note: pyscf_algebra::axpy
        // is Tensor-API and NotYetImplemented{phase:2}; inline the loop
        // (matches plan 03-11 pattern for the same reason).
        let n = j.data.len();
        let mut veff = Vec::with_capacity(n);
        for i in 0..n {
            veff.push(j.data[i] - 0.5 * k.data[i]);
        }
        Ok(Density {
            nao: j.nao,
            data: veff,
        })
    }

    fn get_fock(
        &self,
        h1e: &Density,
        s1e: &Density,
        vhf: &Density,
        dm: &Density,
        cycle: i32,
        diis_state: Option<&Density>,
    ) -> Result<Density, PyscfRsError> {
        crate::fock::default_get_fock(h1e, s1e, vhf, dm, cycle, diis_state)
    }

    fn eig(&self, fock: &Density, s1e: &Density) -> Result<MOCoefficients, PyscfRsError> {
        crate::eig::default_eig(fock, s1e)
    }

    fn get_occ(&self, mo_energy: &[f64], nelec: usize) -> Result<Vec<f64>, PyscfRsError> {
        crate::occ::default_get_occ(mo_energy, nelec)
    }

    fn make_rdm1(&self, mo: &MOCoefficients) -> Result<Density, PyscfRsError> {
        crate::rdm::default_make_rdm1(mo)
    }

    fn energy_elec(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<(Energy, Energy), PyscfRsError> {
        crate::energy::default_energy_elec(dm, h1e, vhf)
    }

    fn energy_tot(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<Energy, PyscfRsError> {
        crate::energy::default_energy_tot(dm, h1e, vhf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_basis_name_unwraps_name_variant() {
        assert_eq!(extract_basis_name("Name(\"cc-pvdz\")"), "cc-pvdz");
        assert_eq!(extract_basis_name("Name(\"sto-3g\")"), "sto-3g");
    }

    #[test]
    fn extract_basis_name_passes_through_other_forms() {
        assert_eq!(extract_basis_name("PerElement({})"), "PerElement({})");
        assert_eq!(extract_basis_name(""), "");
    }
}
