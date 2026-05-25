//! `as_scanner(rhf)` — SCF-12.
//!
//! Source: `pyscf/scf/hf.py:1538-1602` — `def as_scanner(mf)` and
//! `SCF_Scanner.__call__(self, mol)`. The scanner is a closure that
//! re-builds an RHF on a new Mole (keeping the original convergence +
//! DIIS settings), runs `kernel`, and returns just the total energy.
//! This is the standard pyscf entry point for geometry-optimization
//! drivers (`geomopt.optimize(mf)`).

use crate::RHF;
use pyscf_core::{Energy, Mole, PyscfRsError};

/// Returns a closure that, given a new Mole, builds an RHF with the
/// same scalar settings as `rhf`, runs SCF, and returns the converged
/// total energy. Captures `Send + Sync` so it can be moved across
/// threads (geometry-optimization drivers may run scanner evaluations
/// in parallel).
#[allow(clippy::type_complexity)]
pub fn as_scanner(rhf: &RHF) -> Box<dyn Fn(&Mole) -> Result<Energy, PyscfRsError> + Send + Sync> {
    // Capture scalar SCF settings by value.
    let conv_tol = rhf.conv_tol;
    let conv_tol_grad = rhf.conv_tol_grad;
    let max_cycle = rhf.max_cycle;
    let init_guess = rhf.init_guess.clone();
    let diis = rhf.diis;
    let diis_space = rhf.diis_space;
    let diis_start_cycle = rhf.diis_start_cycle;
    let diis_damp = rhf.diis_damp;
    let level_shift = rhf.level_shift;
    let damp = rhf.damp;
    let direct_scf = rhf.direct_scf;
    let direct_scf_tol = rhf.direct_scf_tol;
    let verbose = rhf.verbose;
    let max_memory = rhf.max_memory;
    Box::new(move |new_mol: &Mole| -> Result<Energy, PyscfRsError> {
        let mut new_rhf = RHF::new(new_mol.clone());
        new_rhf.conv_tol = conv_tol;
        new_rhf.conv_tol_grad = conv_tol_grad;
        new_rhf.max_cycle = max_cycle;
        new_rhf.init_guess = init_guess.clone();
        new_rhf.diis = diis;
        new_rhf.diis_space = diis_space;
        new_rhf.diis_start_cycle = diis_start_cycle;
        new_rhf.diis_damp = diis_damp;
        new_rhf.level_shift = level_shift;
        new_rhf.damp = damp;
        new_rhf.direct_scf = direct_scf;
        new_rhf.direct_scf_tol = direct_scf_tol;
        new_rhf.verbose = verbose;
        new_rhf.max_memory = max_memory;
        new_rhf.kernel()?;
        Ok(Energy(new_rhf.e_tot))
    })
}
