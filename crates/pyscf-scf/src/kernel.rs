//! Generic SCF kernel signature — D-02 pub trait + pub generic kernel.
//! Body lives in plan 03-11's kernel_impl.rs.
use crate::OverrideHooks;
use pyscf_core::{Density, Energy, MOCoefficients, Mole, PyscfRsError};

#[derive(Debug, Clone)]
pub struct KernelConfig {
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
    pub init_guess: InitGuessMode,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            conv_tol: 1e-9,
            conv_tol_grad: None,
            max_cycle: 50,
            diis: true,
            diis_space: 8,
            diis_start_cycle: 1,
            diis_damp: 0.0,
            level_shift: 0.0,
            damp: 0.0,
            direct_scf: true,
            direct_scf_tol: 1e-13,
            init_guess: InitGuessMode::Minao,
        }
    }
}

#[derive(Debug, Clone)]
pub enum InitGuessMode {
    Minao,
    Atom,
    OneElectron,
    Huckel,
    Chkfile(std::path::PathBuf),
    UserDM(Density),
}

#[derive(Debug, Clone)]
pub struct ScfResult {
    pub e_tot: Energy,
    pub mo_coeff: MOCoefficients,
    pub mo_energy: Vec<f64>,
    pub mo_occ: Vec<f64>,
    pub converged: bool,
    pub cycles: u32,
}

/// Generic kernel signature — works for Rust-only and Python-driven SCF identically.
/// Body delegates to `kernel_impl::scf_loop` (which plan 03-11 implements).
pub fn kernel<H: OverrideHooks>(
    mol: &Mole,
    hooks: &H,
    cfg: KernelConfig,
) -> Result<ScfResult, PyscfRsError> {
    crate::kernel_impl::scf_loop(mol, hooks, cfg)
}
