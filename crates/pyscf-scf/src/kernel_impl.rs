//! SCF cycle loop. Body filled in plan 03-11.
//!
//! Source: `pyscf/scf/hf.py:48-244` — `def kernel(mf, conv_tol, ...)`.
//! Verbatim port of the cycle loop:
//!
//!   1. S, h_core, init guess D₀
//!   2. V_HF₀ = get_veff(D₀)
//!   3. E₀ = energy_elec(D₀, h_core, V_HF₀) + E_nuc
//!   4. for cycle in 0..max_cycle:
//!        F = get_fock(h_core, S, V_HF, D, cycle)
//!        (eigvals, C) = eig(F, S)              ← canonicalize_signs inside
//!        occ = get_occ(eigvals, nelec)
//!        D = make_rdm1(C, occ)
//!        V_HF = get_veff(D)
//!        E_new = energy_elec(D, h_core, V_HF) + E_nuc
//!        if |E_new - E_old| < conv_tol: converged
//!        E_old = E_new
//!
//! Plan 03-04 — CDIIS extrapolation now lives inline below: after the
//! `hooks.get_fock(...)` call we invoke `diis_adapter::diis_step` to
//! extrapolate the Fock matrix when `cycle >= cfg.diis_start_cycle`.
//! No DF-HF entry point — plan 03-05 ships DfHooks impl of OverrideHooks
//! that routes `get_jk` through pyscf-df.

use crate::diis_adapter::{diis_step, FockSubspace};
use crate::error::ScfError;
use crate::{InitGuessMode, KernelConfig, OverrideHooks, ScfResult};
use pyscf_core::{Density, Energy, Mole, PyscfRsError};
use pyscf_diis::Diis;

pub(crate) fn scf_loop<H: OverrideHooks>(
    mol: &Mole,
    hooks: &H,
    cfg: KernelConfig,
) -> Result<ScfResult, PyscfRsError> {
    tracing::info!(
        target: "pyscf_scf::kernel",
        nao = mol.nao_nr,
        nelectron = mol.nelectron,
        max_cycle = cfg.max_cycle,
        conv_tol = cfg.conv_tol,
        diis = cfg.diis,
        diis_space = cfg.diis_space,
        diis_start_cycle = cfg.diis_start_cycle,
        "SCF kernel entry"
    );

    // Step 1: 1-electron integrals + initial density.
    let s1e = hooks.get_ovlp(mol)?;
    let h1e = hooks.get_hcore(mol)?;
    let mut dm = match &cfg.init_guess {
        InitGuessMode::UserDM(d) => d.clone(),
        other => hooks.get_init_guess(mol, other)?,
    };

    // Step 2: Initial V_HF + electronic energy.
    let mut vhf = hooks.get_veff(mol, &dm)?;
    let (e_elec_init, _) = hooks.energy_elec(&dm, &h1e, &vhf)?;
    let e_nuc = mol.enuc();
    let mut e_tot = Energy(e_elec_init.0 + e_nuc);

    // Convergence-gradient threshold (currently informational; the DIIS
    // error vector — built inside diis_step — provides the natural
    // gradient-norm proxy. Plan 03-08 oracle Arm 6 will assert against
    // the error-vector norm.)
    let _conv_tol_grad = cfg.conv_tol_grad.unwrap_or_else(|| cfg.conv_tol.sqrt());

    // Plan 03-04 — instantiate the CDIIS Pulay stack when DIIS is enabled.
    // `diis_space` defaults to 8 and `diis_start_cycle` defaults to 1 per
    // upstream `pyscf/scf/hf.py:1701-1704` (SCF-04 contract).
    let nao = mol.nao_nr;
    let mut diis: Option<Diis<FockSubspace>> = if cfg.diis {
        Some(Diis::<FockSubspace>::new(cfg.diis_space as usize))
    } else {
        None
    };

    // Step 4: SCF cycle loop.
    let mut converged = false;
    let mut cycles = 0u32;
    let mut last_diff = f64::INFINITY;
    let mut mo_final: Option<pyscf_core::MOCoefficients> = None;

    for cycle in 0..cfg.max_cycle {
        cycles = cycle + 1;

        // Build base Fock via hook (no DIIS yet — get_fock is overrideable).
        let fock_base = hooks.get_fock(&h1e, &s1e, &vhf, &dm, cycle as i32, None)?;

        // Plan 03-04 — apply CDIIS extrapolation when enabled. When
        // `cycle < diis_start_cycle` or `cfg.diis == false`, diis_step
        // is skipped or returns the base Fock unchanged. The
        // OverrideHooks::get_fock signature is unchanged from plan 03-03,
        // so existing override impls keep working.
        let fock = if let Some(diis_stack) = diis.as_mut() {
            let extrap = diis_step(
                diis_stack,
                &s1e.data,
                &dm.data,
                &fock_base.data,
                nao,
                cycle as usize,
                cfg.diis_start_cycle as usize,
            )
            .map_err(|e| PyscfRsError::ConvergenceFailure {
                iterations: cycle,
                reason: format!("DIIS extrapolation failed: {}", e),
            })?;
            Density {
                nao,
                data: extrap,
            }
        } else {
            fock_base
        };

        // Diagonalize F·C = S·C·diag(ε); SCF-13 canonicalize_signs inside default_eig.
        let mut mo = hooks.eig(&fock, &s1e)?;

        // Aufbau-fill occupations.
        let occ = hooks.get_occ(&mo.energies, mol.nelectron)?;
        mo.occupations = occ;

        // RDM1 update.
        dm = hooks.make_rdm1(&mo)?;

        // V_HF update.
        vhf = hooks.get_veff(mol, &dm)?;

        // Energy update.
        let (e_elec, _) = hooks.energy_elec(&dm, &h1e, &vhf)?;
        let e_new = Energy(e_elec.0 + e_nuc);
        last_diff = (e_new.0 - e_tot.0).abs();

        tracing::debug!(
            target: "pyscf_scf::kernel",
            cycle = cycle,
            e_tot = e_new.0,
            delta = last_diff,
            "cycle update"
        );
        e_tot = e_new;
        mo_final = Some(mo);

        if last_diff < cfg.conv_tol {
            converged = true;
            tracing::info!(
                target: "pyscf_scf::kernel",
                cycles = cycles,
                e_tot = e_tot.0,
                "SCF converged"
            );
            break;
        }
    }

    let mo = mo_final.ok_or_else(|| {
        // cycles == 0 means max_cycle was 0; no MO ever computed.
        ScfError::ConvergenceFailure {
            cycles: 0,
            last_diff: f64::INFINITY,
        }
    })?;

    if !converged {
        return Err(ScfError::ConvergenceFailure { cycles, last_diff }.into());
    }

    Ok(ScfResult {
        e_tot,
        mo_energy: mo.energies.clone(),
        mo_occ: mo.occupations.clone(),
        mo_coeff: mo,
        converged,
        cycles,
    })
}
