//! UHF — Unrestricted Hartree-Fock. Source: pyscf/scf/uhf.py:754.
//!
//! 30-attribute floor (SCF-14) — same _keys set as RHF except
//! `mo_coeff` / `mo_energy` / `mo_occ` carry an (alpha, beta) pair
//! rather than a single matrix. `nelec` is meaningful (alpha, beta).
//!
//! `Debug` is implemented manually (see rhf.rs explanation) because of
//! the `Box<dyn Any>` / `Box<dyn Fn>` opaque slots.

use crate::{InitGuessMode, KernelConfig, NoOverrides, ScfResult, kernel};
use pyscf_core::{MOCoefficients, Mole, PyscfRsError};

pub struct UHF {
    pub mol: Mole,
    /// (alpha, beta) MO coefficient pair.
    pub mo_coeff: Option<(MOCoefficients, MOCoefficients)>,
    /// (alpha, beta) MO energies.
    pub mo_energy: Option<(Vec<f64>, Vec<f64>)>,
    /// (alpha, beta) occupations.
    pub mo_occ: Option<(Vec<f64>, Vec<f64>)>,
    pub e_tot: f64,
    pub e_elec: f64,
    pub converged: bool,
    pub cycles: u32,
    pub verbose: u8,
    pub chkfile: Option<std::path::PathBuf>,
    pub max_memory: f64,
    pub direct_scf: bool,
    pub direct_scf_tol: f64,
    pub init_guess: String,
    pub level_shift: f64,
    pub damp: f64,
    pub diis: bool,
    pub diis_space: u32,
    pub diis_start_cycle: u32,
    pub diis_damp: f64,
    pub diis_file: Option<std::path::PathBuf>,
    pub max_cycle: u32,
    pub conv_tol: f64,
    pub conv_tol_grad: Option<f64>,
    pub with_df: Option<Box<dyn std::any::Any + Send + Sync>>,
    pub disp: Option<String>,
    pub do_disp: bool,
    pub irrep_nelec: std::collections::HashMap<String, u32>,
    /// UHF carries the explicit (n_alpha, n_beta) split.
    pub nelec: Option<(u32, u32)>,
    #[allow(clippy::type_complexity)]
    pub callback: Option<Box<dyn Fn(&ScfResult) + Send + Sync>>,
    pub scf_summary: std::collections::HashMap<String, f64>,
    pub opt: Option<Box<dyn std::any::Any + Send + Sync>>,
}

impl std::fmt::Debug for UHF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UHF")
            .field("mol", &self.mol)
            .field("mo_coeff", &self.mo_coeff.as_ref().map(|_| "<(MO,MO)>"))
            .field("mo_energy", &self.mo_energy)
            .field("mo_occ", &self.mo_occ)
            .field("e_tot", &self.e_tot)
            .field("converged", &self.converged)
            .field("cycles", &self.cycles)
            .field("max_cycle", &self.max_cycle)
            .field("conv_tol", &self.conv_tol)
            .field("nelec", &self.nelec)
            .finish_non_exhaustive()
    }
}

impl UHF {
    pub fn new(mol: Mole) -> Self {
        Self {
            mol,
            mo_coeff: None,
            mo_energy: None,
            mo_occ: None,
            e_tot: 0.0,
            e_elec: 0.0,
            converged: false,
            cycles: 0,
            verbose: 3,
            chkfile: None,
            max_memory: 4000.0,
            direct_scf: true,
            direct_scf_tol: 1e-13,
            init_guess: "minao".to_string(),
            level_shift: 0.0,
            damp: 0.0,
            diis: true,
            diis_space: 8,
            diis_start_cycle: 1,
            diis_damp: 0.0,
            diis_file: None,
            max_cycle: 50,
            conv_tol: 1e-9,
            conv_tol_grad: None,
            with_df: None,
            disp: None,
            do_disp: false,
            irrep_nelec: Default::default(),
            nelec: None,
            callback: None,
            scf_summary: Default::default(),
            opt: None,
        }
    }

    /// Drives SCF via the generic kernel with `NoOverrides`.
    /// Body PANICS at runtime until plan 03-11 ships the cycle loop.
    pub fn kernel(&mut self) -> Result<ScfResult, PyscfRsError> {
        let cfg = self.to_kernel_config();
        let result = kernel(&self.mol, &NoOverrides, cfg)?;
        // alpha/beta packing into the (MO, MO) slots happens in plan 03-11
        // once UHF::kernel has the alpha/beta-aware SCF loop. For now we
        // just record the converged scalar state.
        self.e_tot = result.e_tot.0;
        self.converged = result.converged;
        self.cycles = result.cycles;
        Ok(result)
    }

    fn to_kernel_config(&self) -> KernelConfig {
        KernelConfig {
            conv_tol: self.conv_tol,
            conv_tol_grad: self.conv_tol_grad,
            max_cycle: self.max_cycle,
            diis: self.diis,
            diis_space: self.diis_space,
            diis_start_cycle: self.diis_start_cycle,
            diis_damp: self.diis_damp,
            level_shift: self.level_shift,
            damp: self.damp,
            direct_scf: self.direct_scf,
            direct_scf_tol: self.direct_scf_tol,
            init_guess: match self.init_guess.as_str() {
                "minao" => InitGuessMode::Minao,
                "atom" => InitGuessMode::Atom,
                "1e" => InitGuessMode::OneElectron,
                "huckel" => InitGuessMode::Huckel,
                "chkfile" => {
                    InitGuessMode::Chkfile(self.chkfile.clone().unwrap_or_else(|| "scf.chk".into()))
                }
                _ => InitGuessMode::Minao,
            },
        }
    }
}
