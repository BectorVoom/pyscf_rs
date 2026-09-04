//! The k-point SCF driver — plan 11-09, port of `khf.py:133-160` (`get_fock`)
//! and `scf/hf.py:kernel` with the periodic specifics folded in.
//!
//! Everything method-specific lives behind [`KOverrideHooks`]; this file is the
//! loop, and it is the ONLY loop — `KRHF`/`KUHF`/`KROHF`/`KGHF` and (Phase 12)
//! `KRKS`/`KUKS` are implementations of the trait, never copies of the cycle.

use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_diis::Diis;

use crate::kdiis::{KFockSubspace, diis_step};
use crate::khooks::KOverrideHooks;
use crate::kocc::norm;
use crate::types::{KDms, KMats, KScfConfig, KScfResult};

/// `F = H + V_HF`, per `(set, k)` — `khf.py:137`.
pub fn bare_fock(h1e: &KMats, vhf: &KDms) -> KDms {
    vhf.iter()
        .map(|set| {
            set.iter()
                .enumerate()
                .map(|(k, v)| {
                    let mut f = v.clone();
                    for i in 0..f.len() {
                        f.re[i] += h1e[k].re[i];
                        f.im[i] += h1e[k].im[i];
                    }
                    f
                })
                .collect()
        })
        .collect()
}

/// `mol_hf.damping(f, f_prev, factor)` — `scf/hf.py`, applied per `(set, k)`
/// before DIIS takes over (`khf.py:150-152`).
fn damp(fock: &mut KDms, prev: &KDms, factor: f64) {
    for (s, set) in fock.iter_mut().enumerate() {
        for (k, f) in set.iter_mut().enumerate() {
            for i in 0..f.len() {
                f.re[i] = f.re[i] * (1.0 - factor) + prev[s][k].re[i] * factor;
                f.im[i] = f.im[i] * (1.0 - factor) + prev[s][k].im[i] * factor;
            }
        }
    }
}

/// `mol_hf.level_shift(s, dm, f, factor)` — `khf.py:155-157`.
///
/// `F' = F + (S - S D S / 2) * factor` for a restricted density (where
/// `D S D = 2 D`); the general form upstream uses is
/// `F + (S - S D S * 0.5) * shift`.
fn level_shift(fock: &mut KDms, s1e: &KMats, dms: &KDms, factor: f64, nao: usize) {
    for (s, set) in fock.iter_mut().enumerate() {
        for (k, f) in set.iter_mut().enumerate() {
            let sd = mm(&s1e[k], &dms[s][k], nao);
            let sds = mm(&sd, &s1e[k], nao);
            for i in 0..f.len() {
                f.re[i] += factor * (s1e[k].re[i] - 0.5 * sds.re[i]);
                f.im[i] += factor * (s1e[k].im[i] - 0.5 * sds.im[i]);
            }
        }
    }
}

fn mm(a: &CTensor, b: &CTensor, n: usize) -> CTensor {
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sr = 0.0_f64;
            let mut si = 0.0_f64;
            for t in 0..n {
                let (ar, ai) = (a.re[i * n + t], a.im[i * n + t]);
                let (br, bi) = (b.re[t * n + j], b.im[t * n + j]);
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }
            re[i * n + j] = sr;
            im[i * n + j] = si;
        }
    }
    CTensor::from_planes(re, im)
}

/// The periodic SCF cycle.
///
/// ```text
/// s1e   = get_ovlp()
/// h1e   = get_hcore()
/// dm    = get_init_guess()
/// e_nuc = cell.ewald()
/// loop:
///     vhf   = get_veff(dm)
///     fock  = h1e + vhf         (+ damping, DIIS, level shift)
///     eps,C = eig(fock, s1e)    per (set, k)
///     occ   = get_occ(eps)      ONE Fermi level over the whole BZ
///     dm    = make_rdm1(C, occ)
///     e_tot = energy_elec(dm, h1e, vhf) + e_nuc
///     converged when |dE| < conv_tol AND |g| < conv_tol_grad
/// ```
///
/// The convergence test uses the UNEXTRAPOLATED Fock matrix for the gradient,
/// as `scf/hf.py`'s kernel does — a DIIS-extrapolated Fock is not the Fock of
/// the current density and its gradient would converge to the wrong thing.
///
/// # Errors
/// Propagates every hook, `cell.ewald()`, and the DIIS solve.
pub fn kernel<H: KOverrideHooks>(hooks: &H, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
    let nao = hooks.nao();
    let nkpts = hooks.kpts().len();
    let nset = hooks.nset();
    if nkpts == 0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "periodic SCF: no k-points".into(),
        )));
    }

    let s1e = hooks.get_ovlp()?;
    let h1e = hooks.get_hcore()?;
    let mut dm = hooks.get_init_guess(&cfg.init_guess, &s1e)?;
    let e_nuc = hooks.energy_nuc()?;

    let mut vhf = hooks.get_veff(&dm)?;
    let (mut e_elec, mut e_coul) = hooks.energy_elec(&dm, &h1e, &vhf)?;
    let mut e_tot = e_elec + e_nuc;
    if cfg.verbose {
        tracing::info!(cycle = -1, e_tot, "init E");
    }

    let mut diis = if cfg.diis {
        Some(Diis::<KFockSubspace>::new(cfg.diis_space))
    } else {
        None
    };
    let grad_tol = cfg.grad_tol();

    let mut mo_energy: Vec<Vec<f64>> = Vec::new();
    let mut mo_coeff: Vec<CTensor> = Vec::new();
    let mut mo_occ: Vec<Vec<f64>> = Vec::new();
    let mut fermi: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut cycles = 0_u32;
    let mut fock_last: Option<KDms> = None;

    for cycle in 0..cfg.max_cycle {
        cycles = cycle + 1;
        let last_e = e_tot;

        // khf.py:137-158 — the Fock build with its three modifiers.
        let mut fock = hooks.get_fock(&h1e, &vhf, &dm)?;
        if (cycle as i64) < cfg.diis_start_cycle as i64 - 1
            && cfg.damp.abs() > 1e-4
            && let Some(prev) = fock_last.as_ref()
        {
            damp(&mut fock, prev, cfg.damp);
        }
        if let Some(d) = diis.as_mut()
            && cycle >= cfg.diis_start_cycle
        {
            fock = diis_step(d, &s1e, &hooks.diis_dms(&dm), &fock, nao).map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "periodic SCF: DIIS failed at cycle {cycle}: {e}"
                )))
            })?;
        }
        if cfg.level_shift.abs() > 1e-4 {
            level_shift(&mut fock, &s1e, &hooks.diis_dms(&dm), cfg.level_shift, nao);
        }
        fock_last = Some(fock.clone());

        let (e, c) = hooks.eig(&fock, &s1e)?;
        let (occ, f_levels) = hooks.get_occ(&e)?;
        dm = hooks.make_rdm1(&c, &occ)?;
        vhf = hooks.get_veff(&dm)?;
        let (ee, ec) = hooks.energy_elec(&dm, &h1e, &vhf)?;
        e_elec = ee;
        e_coul = ec;
        e_tot = e_elec + e_nuc;

        // The gradient is measured on the BARE Fock of the NEW density: a
        // DIIS-extrapolated Fock is not the Fock of the current density, and
        // its gradient would converge to the wrong stationary point.
        let norm_gorb = norm(&hooks.get_grad(&c, &occ, &h1e, &vhf));
        let de = e_tot - last_e;
        if cfg.verbose {
            tracing::info!(cycle, e_tot, de, norm_gorb, "periodic SCF cycle");
        }

        mo_energy = e;
        mo_coeff = c;
        mo_occ = occ;
        fermi = f_levels;

        if de.abs() < cfg.conv_tol && norm_gorb < grad_tol {
            converged = true;
            break;
        }
    }

    // `max_cycle = 0` is upstream's "just build the Fock and diagonalise once"
    // mode (`scf/hf.py`'s kernel does the same after its loop). Without this the
    // result would carry empty orbital vectors.
    if mo_coeff.is_empty() {
        let fock = hooks.get_fock(&h1e, &vhf, &dm)?;
        let (e, c) = hooks.eig(&fock, &s1e)?;
        let (occ, f_levels) = hooks.get_occ(&e)?;
        mo_energy = e;
        mo_coeff = c;
        mo_occ = occ;
        fermi = f_levels;
    }

    // `free_energy()` reports the ENTROPY TERM `-sigma * S`; upstream's
    // `e_free = e_tot - sigma*S` and `e_zero = e_tot - sigma*S/2`
    // (`scf/smearing.py:258-260`).
    let entropy_term = hooks.free_energy();
    Ok(KScfResult {
        e_tot,
        e_elec,
        e_coul,
        e_nuc,
        mo_energy,
        mo_coeff,
        mo_occ,
        dm,
        converged,
        cycles,
        nset,
        nkpts,
        fermi,
        e_free: entropy_term.map(|t| e_tot + t),
        e_zero: entropy_term.map(|t| e_tot + t * 0.5),
    })
}
