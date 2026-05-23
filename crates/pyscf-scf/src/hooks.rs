//! 10 overrideable hooks — D-01 trait-callback bridge.
//!
//! NOTE: The trait actually exposes 11 methods (`get_hcore`, `get_ovlp`,
//! `get_init_guess`, `get_jk`, `get_veff`, `get_fock`, `eig`, `get_occ`,
//! `make_rdm1`, `energy_elec`, `energy_tot`) — `energy_elec` and
//! `energy_tot` are siblings derived from the same upstream pyscf-method
//! seam (`Method.energy_elec` + `Method.energy_tot`). The "10" in the
//! plan title is the SCF-08 logical-override count; the trait expansion
//! to 11 surface methods is a fidelity choice (upstream `Method` exposes
//! both).
use pyscf_core::{Density, Energy, MOCoefficients, Mole, PyscfRsError};

pub trait OverrideHooks {
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError>;
    fn get_ovlp(&self, mol: &Mole) -> Result<Density, PyscfRsError>;
    fn get_init_guess(
        &self,
        mol: &Mole,
        mode: &crate::InitGuessMode,
    ) -> Result<Density, PyscfRsError>;
    fn get_jk(&self, mol: &Mole, dm: &Density) -> Result<(Density, Density), PyscfRsError>;
    fn get_veff(&self, mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError>;
    fn get_fock(
        &self,
        h1e: &Density,
        s1e: &Density,
        vhf: &Density,
        dm: &Density,
        cycle: i32,
        diis_state: Option<&Density>,
    ) -> Result<Density, PyscfRsError>;
    fn eig(&self, fock: &Density, s1e: &Density) -> Result<MOCoefficients, PyscfRsError>;
    fn get_occ(&self, mo_energy: &[f64], nelec: usize) -> Result<Vec<f64>, PyscfRsError>;
    fn make_rdm1(&self, mo: &MOCoefficients) -> Result<Density, PyscfRsError>;
    fn energy_elec(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<(Energy, Energy), PyscfRsError>;
    fn energy_tot(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<Energy, PyscfRsError>;
}

pub struct NoOverrides;

impl OverrideHooks for NoOverrides {
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        crate::fock::default_get_hcore(mol)
    }
    fn get_ovlp(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        crate::fock::default_get_ovlp(mol)
    }
    fn get_init_guess(
        &self,
        mol: &Mole,
        mode: &crate::InitGuessMode,
    ) -> Result<Density, PyscfRsError> {
        crate::init_guess::default_get_init_guess(mol, mode)
    }
    fn get_jk(&self, mol: &Mole, dm: &Density) -> Result<(Density, Density), PyscfRsError> {
        crate::fock::default_get_jk(mol, dm)
    }
    fn get_veff(&self, mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
        crate::fock::default_get_veff(mol, dm)
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
