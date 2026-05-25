//! (R)CCSD kernel loop, MP2-seeded `init_amps`, `default_energy`, and the
//! `default_*` free functions the `NoCcsdOverrides` hook delegates to.
//!
//! Port of `pyscf/cc/ccsd.py:44-101` (`kernel`), `:1050-1077` (`init_amps`),
//! `rccsd.py:146-162` (`energy`), with the verified CCSDBase class convergence
//! defaults (`ccsd.py:920-928`). Every contraction / reduction routes through
//! `pyscf_algebra::oracle_sum`/`oracle_dot` (NO `+=`, NO `gemm` — `gemm` is
//! `NotYetImplemented{phase:2}`).
//!
//! **Arena (CCSD-11 / D-01):** `ccsd_kernel` runs a HARD `pool.try_reserve`
//! pre-flight on the dominant `vvvv`/`Wabef ≈ nv⁴` buffer BEFORE building eris —
//! an over-budget in-core job returns `MemoryLimitExceeded` with no silent
//! downgrade. The `Wvvvv` scratch is `pool.reserve`d ONCE before the iteration
//! loop and reused every cycle (Pitfall 20), then `release`d after.
#![allow(clippy::needless_range_loop)]

use crate::diis_amps::AmplitudeSubspace;
use crate::eris::ChemistsEris;
use crate::error::CcsdError;
use crate::reference::CcsdReference;
use crate::update_amps::default_update_amps_with_wvvvv;
use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::{Amplitudes, Energy, MOCoefficients, PyscfRsError};
use pyscf_diis::Diis;
use pyscf_mp2::{Frozen, get_nocc};
use pyscf_runtime::WorkspacePool;

// ---- Verified upstream CCSDBase class convergence defaults (ccsd.py:920-928).
// The class attribute wins over the kernel-signature default at runtime; the
// RESEARCH correction (A1) is conv_tol_normt = 1e-5 (NOT the CONTEXT 1e-6 nor
// the kernel-signature 1e-6).

/// Max amplitude iterations (`ccsd.py:920`).
pub const MAX_CYCLE: usize = 50;
/// Energy convergence threshold (`ccsd.py:921`).
pub const CONV_TOL: f64 = 1e-7;
/// Amplitude-norm convergence threshold (`ccsd.py:923`).
pub const CONV_TOL_NORMT: f64 = 1e-5;
/// Amplitude-DIIS subspace size (`ccsd.py:926`; wired in 06-05).
pub const DIIS_SPACE: usize = 6;
/// Amplitude-DIIS start cycle (`ccsd.py:928`; wired in 06-05).
pub const DIIS_START_CYCLE: usize = 0;

/// Result of the (R)CCSD kernel.
///
/// Mirrors `pyscf_mp2::Mp2Result` with the CCSD additions (converged flag,
/// iteration count, MP2 seed energy, and the converged `t1`/`t2` amplitudes the
/// λ-equations and RDMs consume in Wave 3).
#[derive(Debug, Clone)]
pub struct CcsdResult {
    /// CCSD correlation energy.
    pub e_corr: f64,
    /// Whether the amplitude iteration converged.
    pub converged: bool,
    /// Number of iterations performed.
    pub niter: usize,
    /// MP2 seed energy reported by `init_amps` (`ccsd.py:1072`).
    pub emp2: f64,
    /// Converged amplitudes (`t1`/`t2`).
    pub amplitudes: Amplitudes,
}

#[inline]
fn t1_idx(nv: usize, i: usize, a: usize) -> usize {
    i * nv + a
}

#[inline]
fn t2_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

#[inline]
fn ovov_idx(no: usize, nv: usize, i: usize, a: usize, j: usize, b: usize) -> usize {
    ((i * nv + a) * no + j) * nv + b
}

#[inline]
fn fock_idx(nmo: usize, p: usize, q: usize) -> usize {
    p * nmo + q
}

/// Bytes the dominant `vvvv`/`Wabef ≈ nv⁴` arena tenant needs (the in-core
/// pre-flight estimate). Conservative: the single largest in-core buffer
/// (`nv^4` f64).
fn estimate_vvvv_bytes(nvir: usize) -> usize {
    nvir.saturating_mul(nvir)
        .saturating_mul(nvir)
        .saturating_mul(nvir)
        .saturating_mul(8)
}

/// Bytes the AO-direct path's dominant MO `vvvv`-step tenant needs (CCSD-07).
/// The AO-direct path NEVER materializes the `nv^4` MO `vvvv`: it tiles the
/// AO→MO transform over the leading virtual index, holding at most ONE
/// `[1,nv,nv,nv]` slice (`nv^3` f64). This pre-flight estimate is correspondingly
/// LOWER than [`estimate_vvvv_bytes`] (the memory-frugality whole point — D-01 /
/// T-06-08-OOM). The AO source `int2e` is held separately (the v1 path-b cost);
/// the GATED reservation is the MO `vvvv`-step peak.
fn estimate_direct_vvvv_bytes(nvir: usize) -> usize {
    nvir.saturating_mul(nvir)
        .saturating_mul(nvir)
        .saturating_mul(8)
}

/// (R)CCSD driver — port of `pyscf/cc/ccsd.py:44-101` `kernel`.
///
/// Thin wrapper over [`ccsd_kernel_diis`] with `diis = true` (the upstream
/// `mycc.diis = True` default, `ccsd.py:924`). Existing callers keep the
/// 4-arg signature; the no-DIIS path is reachable via `ccsd_kernel_diis`.
pub fn ccsd_kernel<H: crate::hooks::CcsdOverrideHooks>(
    refr: &CcsdReference,
    frozen: &Frozen,
    hooks: &H,
    pool: &WorkspacePool,
) -> Result<CcsdResult, PyscfRsError> {
    ccsd_kernel_diis(refr, frozen, hooks, pool, true)
}

/// AO-direct (R)CCSD driver — port of `pyscf/cc/ccsd.py` with `mycc.direct=True`
/// (CCSD-07). Thin wrapper over [`ccsd_kernel_direct_diis`] with `diis = true`.
///
/// The AO-direct path routes the heavy `nv^4` vvvv contraction through
/// [`crate::direct`] (the AO `int2e` is contracted on-the-fly, tiled per
/// leading-virtual index) instead of holding the full `vvvv` MO tensor — so the
/// pre-flight arena reservation is the LOWER `nv^3` peak, not `nv^4`. It
/// converges to the SAME `e_corr` as the in-core [`ccsd_kernel`] (a different
/// contraction order of the same math; CCSD-07's equivalence contract).
pub fn ccsd_kernel_direct(
    refr: &CcsdReference,
    frozen: &Frozen,
    pool: &WorkspacePool,
) -> Result<CcsdResult, PyscfRsError> {
    ccsd_kernel_direct_diis(refr, frozen, pool, true)
}

/// (R)CCSD driver with an explicit DIIS toggle — port of
/// `pyscf/cc/ccsd.py:44-101` `kernel` + the amplitude-DIIS slot (`:1206`).
///
/// Generic over the override seam (mirrors `pyscf_mp2::rmp2_kernel`). Sequence:
/// 1. PRE-FLIGHT `pool.try_reserve(estimate_vvvv_bytes(nvir))?` — an over-budget
///    in-core run returns `MemoryLimitExceeded` (D-01, no downgrade).
/// 2. Build eris via `hooks.ao2mo`.
/// 3. `init_amps`: `t1=0`, `t2=(ia|jb)/Dijab`, compute `emp2`.
/// 4. `pool.reserve` the `Wvvvv` buffer ONCE here (Pitfall 20).
/// 5. iterate `update_amps` → `normt` → `run_diis` → `energy`; converged when
///    `|dE| < CONV_TOL && normt < CONV_TOL_NORMT` within `MAX_CYCLE`.
/// 6. `pool.release` the buffer.
///
/// **Amplitude-DIIS (CCSD-04 / D-06):** when `diis == true`, the new amplitudes
/// `(t1new, t2new)` and the amplitude residual `(t1new − t1, t2new − t2)` are
/// fed to a `Diis::<AmplitudeSubspace>::new(DIIS_SPACE)` ring buffer
/// (`DIIS_SPACE = 6`, NOT SCF's 8) and extrapolated whenever
/// `istep >= DIIS_START_CYCLE` (`= 0`, port `ccsd.py:1206`). The extrapolated
/// vector is unpacked back to `(t1, t2)`. `AmplitudeSubspace::dot` routes
/// through `oracle_dot` (Pitfall 9). When `diis == false` the iterate is
/// `t1,t2 = t1new,t2new` directly (the un-accelerated path, for the iter-count
/// comparison). DIIS changes the path, not the minimum: both converge to the
/// same `e_corr`.
pub fn ccsd_kernel_diis<H: crate::hooks::CcsdOverrideHooks>(
    refr: &CcsdReference,
    frozen: &Frozen,
    hooks: &H,
    pool: &WorkspacePool,
    diis: bool,
) -> Result<CcsdResult, PyscfRsError> {
    let nocc = get_nocc(&refr.mo_occ, frozen)?;

    // We need nvir for the pre-flight estimate. Resolve it from the active mask
    // BEFORE building eris (the eris build is the expensive step we gate).
    let nmo_active = pyscf_mp2::get_nmo(&refr.mo_occ, frozen)?;
    let nvir = nmo_active.saturating_sub(nocc);

    // (1) HARD pre-flight refusal (D-01 / CCSD-11 / T-06-03-OOM): refuse an
    // over-budget in-core vvvv reservation BEFORE building eris. No downgrade.
    pool.try_reserve(estimate_vvvv_bytes(nvir))
        .map_err(CcsdError::from)?;

    // (2) Build the full ChemistsEris block set through the override seam.
    let eris = hooks.ao2mo(refr, frozen)?;
    if eris.nocc != nocc || eris.nvir != nvir {
        return Err(CcsdError::ShapeMismatch {
            expected: nocc * nvir,
            got: eris.nocc * eris.nvir,
        }
        .into());
    }

    // (3) MP2 seed: t1=0, t2=(ia|jb)/Dijab; report emp2.
    let (emp2, t1, t2) = init_amps(&eris)?;

    // Initial energy (with the seed amplitudes).
    let mut eccsd = default_energy(&amps_t1(&eris, &t1), &amps_t2(&eris, &t2), &eris)?.0;

    // (4) Reserve the Wvvvv arena tenant ONCE before the loop (CCSD-11). Pure
    // in-core (allow_spill=false): the DF/AO-direct spill path is a later wave.
    let wvvvv_id = pool
        .reserve(&[nvir, nvir, nvir, nvir], false)
        .map_err(CcsdError::from)?;

    let mut t1_cur = t1;
    let mut t2_cur = t2;
    let mut converged = false;
    let mut niter = 0usize;

    // Amplitude-DIIS stack (CCSD-04 / D-06). ONE Diis, second storable —
    // reuses the entire Phase-3 `pyscf-diis` machinery (NO new DIIS body),
    // `diis_space = 6` (the verified `ccsd.py:926` constant, NOT SCF's 8).
    // Only constructed when `diis == true`.
    let mut diis_stack: Option<Diis<AmplitudeSubspace>> = if diis {
        Some(Diis::<AmplitudeSubspace>::new(DIIS_SPACE))
    } else {
        None
    };

    // (5) Amplitude iteration.
    for istep in 0..MAX_CYCLE {
        niter = istep + 1;

        // update_amps writes the cc_Wvvvv tenant into the reserved pool buffer
        // (read-modify via with_mut_slice — the buffer is the working store).
        let (t1new, t2new) = pool
            .with_mut_slice(&wvvvv_id, |wvvvv| {
                default_update_amps_with_wvvvv(&t1_cur, &t2_cur, &eris, wvvvv)
            })
            .map_err(CcsdError::from)??;

        // normt = ||vector(new) - vector(old)|| (the flat amplitude packing).
        let normt = amplitude_delta_norm(nocc, nvir, &t1_cur, &t2_cur, &t1new, &t2new);

        // run_diis (port `ccsd.py:1206`): extrapolate the amplitudes through
        // the DIIS ring buffer when enabled and `istep >= DIIS_START_CYCLE`.
        // `DIIS_START_CYCLE` is the verified `ccsd.py:928` configurable
        // start-cycle constant (currently 0); keeping the `>=` guard preserves
        // its configurable semantics (raise the const → DIIS starts later). With
        // the const pinned at 0 the comparison is trivially true, which clippy's
        // `absurd_extreme_comparisons` flags — allowed here BECAUSE the
        // start-cycle is intentionally a tunable constant, not a literal `0`.
        #[allow(clippy::absurd_extreme_comparisons)]
        match diis_stack.as_mut() {
            Some(stack) if istep >= DIIS_START_CYCLE => {
                // Current iterate = packed (t1new, t2new); error vector =
                // packed residual (t1new − t1, t2new − t2) — the same flat
                // amplitudes_to_vector layout (Pitfall 9 path).
                let current = AmplitudeSubspace::from_amplitudes(&t1new, &t2new, nocc, nvir);
                let prev = AmplitudeSubspace::from_amplitudes(&t1_cur, &t2_cur, nocc, nvir);
                let err: Vec<f64> = current.as_flat_residual(&prev);
                let extrap = stack.extrapolate(current, err).map_err(CcsdError::from)?;
                let (t1e, t2e) = extrap.to_amplitudes();
                t1_cur = t1e;
                t2_cur = t2e;
            }
            _ => {
                // No DIIS (or below start cycle): accept the new amplitudes.
                t1_cur = t1new;
                t2_cur = t2new;
            }
        }

        let eold = eccsd;
        eccsd = default_energy(&amps_t1(&eris, &t1_cur), &amps_t2(&eris, &t2_cur), &eris)?.0;

        if (eccsd - eold).abs() < CONV_TOL && normt < CONV_TOL_NORMT {
            converged = true;
            break;
        }
    }

    // (6) Release the arena tenant back to the free-list.
    pool.release(wvvvv_id);

    let amplitudes = Amplitudes::from_vec(nocc, nvir, t1_cur, t2_cur);
    Ok(CcsdResult {
        e_corr: eccsd,
        converged,
        niter,
        emp2,
        amplitudes,
    })
}

/// AO-direct (R)CCSD driver with an explicit DIIS toggle (CCSD-07,
/// `mycc.direct=True`). Mirrors [`ccsd_kernel_diis`] but routes the heavy `nv^4`
/// vvvv contraction through [`crate::update_amps::default_update_amps_direct`]
/// (AO `int2e` tiled per leading-virtual index) instead of the in-core
/// `cc_Wvvvv`. Sequence:
/// 1. PRE-FLIGHT `pool.try_reserve(estimate_direct_vvvv_bytes(nvir))?` — the
///    LOWER `nv^3` MO `vvvv`-step peak (vs the in-core `nv^4`), the
///    memory-frugality proof (D-01 / T-06-08-OOM). Over-budget still HARD-refuses.
/// 2. Build the AO `int2e` + the virtual MO coeff block `cv` ONCE (the AO source
///    the per-cycle AO-direct contraction tiles over).
/// 3. Build eris through the in-tree default ao2mo; `init_amps` MP2 seed.
/// 4. iterate `default_update_amps_direct` → `normt` → `run_diis` → `energy`.
///
/// Bound to the in-tree default ao2mo (NOT the hook seam): the AO-direct path
/// needs the raw AO `int2e` + MO coeffs, which only the default in-tree path
/// exposes. Converges to the SAME `e_corr` as [`ccsd_kernel_diis`].
pub fn ccsd_kernel_direct_diis(
    refr: &CcsdReference,
    frozen: &Frozen,
    pool: &WorkspacePool,
    diis: bool,
) -> Result<CcsdResult, PyscfRsError> {
    let nocc = get_nocc(&refr.mo_occ, frozen)?;
    let nmo_active = pyscf_mp2::get_nmo(&refr.mo_occ, frozen)?;
    let nvir = nmo_active.saturating_sub(nocc);

    // (1) HARD pre-flight refusal on the LOWER AO-direct peak (nv^3, not nv^4).
    pool.try_reserve(estimate_direct_vvvv_bytes(nvir))
        .map_err(CcsdError::from)?;

    // (2) The AO source: full int2e + the active virtual MO coeff block. Built
    // ONCE; the per-cycle AO-direct vvvv contraction tiles over it.
    let mask: Vec<bool> = pyscf_mp2::get_frozen_mask(&refr.mo_occ, frozen)?;
    let cv = mo_subset(refr, &mask, false)?;
    let nao = refr.mol.nao_nr;
    let eri_ao = pyscf_gto::intor(&refr.mol, "int2e")?;
    let eri_ao_vals = eri_ao.values;

    // (3) Build eris through the in-tree default ao2mo + MP2 seed.
    let eris = default_ao2mo(refr, frozen)?;
    if eris.nocc != nocc || eris.nvir != nvir {
        return Err(CcsdError::ShapeMismatch {
            expected: nocc * nvir,
            got: eris.nocc * eris.nvir,
        }
        .into());
    }
    let (emp2, t1, t2) = init_amps(&eris)?;
    let mut eccsd = default_energy(&amps_t1(&eris, &t1), &amps_t2(&eris, &t2), &eris)?.0;

    let mut t1_cur = t1;
    let mut t2_cur = t2;
    let mut converged = false;
    let mut niter = 0usize;

    let mut diis_stack: Option<Diis<AmplitudeSubspace>> = if diis {
        Some(Diis::<AmplitudeSubspace>::new(DIIS_SPACE))
    } else {
        None
    };

    // (4) Amplitude iteration through the AO-direct vvvv step.
    for istep in 0..MAX_CYCLE {
        niter = istep + 1;

        let (t1new, t2new) = crate::update_amps::default_update_amps_direct(
            &t1_cur,
            &t2_cur,
            &eris,
            &eri_ao_vals,
            nao,
            &cv,
        )
        .map_err(PyscfRsError::from)?;

        let normt = amplitude_delta_norm(nocc, nvir, &t1_cur, &t2_cur, &t1new, &t2new);

        match diis_stack.as_mut() {
            Some(stack) if diis => {
                let current = AmplitudeSubspace::from_amplitudes(&t1new, &t2new, nocc, nvir);
                let prev = AmplitudeSubspace::from_amplitudes(&t1_cur, &t2_cur, nocc, nvir);
                let err: Vec<f64> = current.as_flat_residual(&prev);
                let extrap = stack.extrapolate(current, err).map_err(CcsdError::from)?;
                let (t1e, t2e) = extrap.to_amplitudes();
                t1_cur = t1e;
                t2_cur = t2e;
            }
            _ => {
                t1_cur = t1new;
                t2_cur = t2new;
            }
        }

        let eold = eccsd;
        eccsd = default_energy(&amps_t1(&eris, &t1_cur), &amps_t2(&eris, &t2_cur), &eris)?.0;

        if (eccsd - eold).abs() < CONV_TOL && normt < CONV_TOL_NORMT {
            converged = true;
            break;
        }
    }

    let amplitudes = Amplitudes::from_vec(nocc, nvir, t1_cur, t2_cur);
    Ok(CcsdResult {
        e_corr: eccsd,
        converged,
        niter,
        emp2,
        amplitudes,
    })
}

/// Wrap a flat t1 buffer in an owned `Amplitudes` (energy/update_amps want the
/// `Amplitudes` handle for the hook signature).
fn amps_t1(eris: &ChemistsEris, t1: &[f64]) -> Amplitudes {
    Amplitudes::from_vec(eris.nocc, eris.nvir, t1.to_vec(), Vec::new())
}

fn amps_t2(eris: &ChemistsEris, t2: &[f64]) -> Amplitudes {
    Amplitudes::from_vec(eris.nocc, eris.nvir, Vec::new(), t2.to_vec())
}

/// `||amplitudes_to_vector(new) - vector(old)||` (Frobenius norm), the
/// dual-criterion `normt` (`ccsd.py:74-76`). For `normt` only the flat norm of
/// the (t1,t2) difference matters — a local pack (t1 ‖ t2) is sufficient; 06-04
/// makes the lower-triangular pack the DIIS storable.
fn amplitude_delta_norm(
    nocc: usize,
    nvir: usize,
    t1_old: &[f64],
    t2_old: &[f64],
    t1_new: &[f64],
    t2_new: &[f64],
) -> f64 {
    let _ = (nocc, nvir);
    let mut sq: Vec<f64> = Vec::with_capacity(t1_old.len() + t2_old.len());
    for (n, o) in t1_new.iter().zip(t1_old.iter()) {
        let d = n - o;
        sq.push(d * d);
    }
    for (n, o) in t2_new.iter().zip(t2_old.iter()) {
        let d = n - o;
        sq.push(d * d);
    }
    oracle_sum(&sq).sqrt()
}

/// MP2-seed `init_amps` (port `ccsd.py:1050-1077`): `t1 = fock[:no,no:]/eia`,
/// `t2[i,j,a,b] = (ia|jb)/Dijab`, `emp2 = 2*einsum('ijab,iajb',t2,ovov) -
/// einsum('jiab,iajb',t2,ovov)`. Returns `(emp2, t1, t2)` (flat C-order).
///
/// The `t2=(ia|jb)/Dijab` block matches the `pyscf_mp2::rmp2_kernel` seed and
/// reuses the same `ovov` block from `default_ao2mo`.
pub fn init_amps(eris: &ChemistsEris) -> Result<(f64, Vec<f64>, Vec<f64>), CcsdError> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let nmo = no + nv;
    let expected_ovov = no * nv * no * nv;
    if eris.ovov.len() != expected_ovov {
        return Err(CcsdError::ShapeMismatch {
            expected: expected_ovov,
            got: eris.ovov.len(),
        });
    }
    if eris.fock.len() != nmo * nmo || eris.mo_energy.len() != nmo {
        return Err(CcsdError::ShapeMismatch {
            expected: nmo * nmo,
            got: eris.fock.len(),
        });
    }

    let mo_e_o: Vec<f64> = (0..no).map(|i| eris.mo_energy[i]).collect();
    let mo_e_v: Vec<f64> = (0..nv).map(|a| eris.mo_energy[no + a]).collect();
    let eia = |i: usize, a: usize| mo_e_o[i] - mo_e_v[a];

    // t1 = fock[:no, no:] / eia  (zero for a canonical reference — fock_ov = 0).
    let mut t1 = vec![0.0_f64; no * nv];
    for i in 0..no {
        for a in 0..nv {
            t1[t1_idx(nv, i, a)] = eris.fock[fock_idx(nmo, i, no + a)] / eia(i, a);
        }
    }

    // t2[i,j,a,b] = ovov[i,a,j,b] / (eia[i,a] + eia[j,b]).
    let mut t2 = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let denom = eia(i, a) + eia(j, b);
                    t2[t2_idx(no, nv, i, j, a, b)] =
                        eris.ovov[ovov_idx(no, nv, i, a, j, b)] / denom;
                }
            }
        }
    }

    // emp2 = 2*einsum('ijab,iajb',t2,ovov) - einsum('jiab,iajb',t2,ovov):
    // materialize the per-element products then reduce with oracle_sum.
    let mut terms: Vec<f64> = Vec::with_capacity(no * no * nv * nv * 2);
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let g = eris.ovov[ovov_idx(no, nv, i, a, j, b)]; // (ia|jb)
                    // 2*t2[i,j,a,b]*g
                    terms.push(2.0 * t2[t2_idx(no, nv, i, j, a, b)] * g);
                    // -t2[j,i,a,b]*g  (the 'jiab,iajb' exchange)
                    terms.push(-t2[t2_idx(no, nv, j, i, a, b)] * g);
                }
            }
        }
    }
    let emp2 = oracle_sum(&terms);

    Ok((emp2, t1, t2))
}

/// Closed-shell RCCSD correlation energy (port `rccsd.py:146-162`):
/// ```text
/// tau = t2 + einsum('ia,jb->ijab', t1, t1)
/// e   = 2*einsum('ia,ia', fock_ov, t1)
///       + 2*einsum('ijab,iajb', tau, ovov)
///       -   einsum('ijab,ibja', tau, ovov)
/// ```
/// Every reduction routes through `oracle_dot`/`oracle_sum`.
pub fn default_energy(
    t1: &Amplitudes,
    t2: &Amplitudes,
    eris: &ChemistsEris,
) -> Result<Energy, PyscfRsError> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let nmo = no + nv;

    let t1s = t1.t1_slice().ok_or(CcsdError::ShapeMismatch {
        expected: no * nv,
        got: 0,
    })?;
    let t2s = t2.t2_slice().ok_or(CcsdError::ShapeMismatch {
        expected: no * no * nv * nv,
        got: 0,
    })?;
    if t1s.len() != no * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * nv,
            got: t1s.len(),
        }
        .into());
    }
    if t2s.len() != no * no * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * no * nv * nv,
            got: t2s.len(),
        }
        .into());
    }

    let t1e = |i: usize, a: usize| t1s[t1_idx(nv, i, a)];
    let tau = |i: usize, j: usize, a: usize, b: usize| {
        t2s[t2_idx(no, nv, i, j, a, b)] + t1e(i, a) * t1e(j, b)
    };

    let mut terms: Vec<f64> = Vec::new();
    // 2*einsum('ia,ia', fock_ov, t1).
    for i in 0..no {
        for a in 0..nv {
            terms.push(2.0 * eris.fock[fock_idx(nmo, i, no + a)] * t1e(i, a));
        }
    }
    // 2*einsum('ijab,iajb', tau, ovov) - einsum('ijab,ibja', tau, ovov).
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let tv = tau(i, j, a, b);
                    // 'iajb' = ovov[i,a,j,b]; 'ibja' = ovov[i,b,j,a]
                    terms.push(2.0 * tv * eris.ovov[ovov_idx(no, nv, i, a, j, b)]);
                    terms.push(-tv * eris.ovov[ovov_idx(no, nv, i, b, j, a)]);
                }
            }
        }
    }
    let _ = oracle_dot; // (oracle_dot available; this energy uses oracle_sum)
    Ok(Energy(oracle_sum(&terms)))
}

// ---------------------------------------------------------------------------
// default_ao2mo: build the FULL ChemistsEris block set from int2e.
// ---------------------------------------------------------------------------

/// Per-atom atomic numbers `Z` for the reference's molecule (for `Frozen::Auto`).
fn reference_elements(refr: &CcsdReference) -> Vec<u32> {
    refr.mol
        .atom_charges()
        .into_iter()
        .map(|z| z.max(0) as u32)
        .collect()
}

/// Active occupied (`want_occupied=true`) / virtual (`false`) MO column subset,
/// frozen-aware. Mirrors `pyscf_mp2::mp2::mo_subset`.
fn mo_subset(
    refr: &CcsdReference,
    mask: &[bool],
    want_occupied: bool,
) -> Result<MOCoefficients, CcsdError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    let mut data = Vec::new();
    let mut energies = Vec::new();
    let mut occupations = Vec::new();
    for col in 0..nmo {
        let active = mask.get(col).copied().unwrap_or(false);
        if !active {
            continue;
        }
        let occ = refr.mo_occ.get(col).copied().unwrap_or(0.0);
        if (occ > 0.0) != want_occupied {
            continue;
        }
        let start = col * nao;
        let end = start + nao;
        if end > refr.mo_coeff.data.len() {
            return Err(CcsdError::ShapeMismatch {
                expected: refr.mo_coeff.data.len(),
                got: end,
            });
        }
        data.extend_from_slice(&refr.mo_coeff.data[start..end]);
        if let Some(e) = refr.mo_energy.get(col) {
            energies.push(*e);
        }
        occupations.push(occ);
    }
    let n_selected = data.len().checked_div(nao).unwrap_or(0);
    Ok(MOCoefficients {
        nao,
        nmo: n_selected,
        data,
        energies,
        occupations,
    })
}

/// Convert an `ao2mo::general` F-order `[n0,n1,n2,n3]` result (element
/// `(i,j,k,l)` at `i + j*n0 + k*n0*n1 + l*n0*n1*n2`) into a flat C-order
/// `[n0,n1,n2,n3]` buffer (element `(i,j,k,l)` at `((i*n1 + j)*n2 + k)*n3 + l`).
fn fto_c_order(f: &[f64], n0: usize, n1: usize, n2: usize, n3: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n0 * n1 * n2 * n3];
    for i in 0..n0 {
        for j in 0..n1 {
            for k in 0..n2 {
                for l in 0..n3 {
                    let fidx = i + j * n0 + k * n0 * n1 + l * n0 * n1 * n2;
                    let cidx = ((i * n1 + j) * n2 + k) * n3 + l;
                    c[cidx] = f[fidx];
                }
            }
        }
    }
    c
}

/// In-core `ao2mo` default — builds the full `ChemistsEris` block set from
/// `int2e` (the `NoCcsdOverrides::ao2mo` delegate). Mirrors
/// `pyscf_mp2::default_ao2mo` but produces ALL CCSD blocks
/// (`oooo`/`ovoo`/`oovv`/`ovov`/`ovvo`/`ovvv`/`vvvv`) plus the diagonal Fock +
/// active MO energies (canonical reference: `fock = diag(mo_energy)`).
///
/// `int2e` is a real arity-4 tensor since 05-08 (cintx#11 closed), so this path
/// runs end-to-end. Errors `?`-propagate; never panics, never substitutes zeros.
pub fn default_ao2mo(refr: &CcsdReference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
    // The frozen mask resolves None/Count/List directly. (Frozen::Auto element
    // resolution via mol charges is exercised by the MP2 path; the CCSD smoke
    // uses Frozen::None.) reference_elements is available for the Auto path.
    let _elements = reference_elements(refr);
    let mask: Vec<bool> = pyscf_mp2::get_frozen_mask(&refr.mo_occ, frozen)?;

    let co = mo_subset(refr, &mask, true)?;
    let cv = mo_subset(refr, &mask, false)?;
    let nocc = co.nmo;
    let nvir = cv.nmo;
    let nao = refr.mol.nao_nr;

    let eri = pyscf_gto::intor(&refr.mol, "int2e")?;
    let g = |sets: [&MOCoefficients; 4]| -> Result<Vec<f64>, PyscfRsError> {
        pyscf_ao2mo::general(&eri.values, nao, sets).map_err(|e| CcsdError::from(e).into())
    };

    // Build each block as (o/v, o/v | o/v, o/v) then F→C reorder.
    // ovov = (o,v|o,v) → C-order (i,a,j,b)
    let ovov = fto_c_order(&g([&co, &cv, &co, &cv])?, nocc, nvir, nocc, nvir);
    // oooo = (o,o|o,o)
    let oooo = fto_c_order(&g([&co, &co, &co, &co])?, nocc, nocc, nocc, nocc);
    // ovoo = (o,v|o,o)
    let ovoo = fto_c_order(&g([&co, &cv, &co, &co])?, nocc, nvir, nocc, nocc);
    // oovv = (o,o|v,v)
    let oovv = fto_c_order(&g([&co, &co, &cv, &cv])?, nocc, nocc, nvir, nvir);
    // ovvo = (o,v|v,o)
    let ovvo = fto_c_order(&g([&co, &cv, &cv, &co])?, nocc, nvir, nvir, nocc);
    // ovvv = (o,v|v,v)
    let ovvv = fto_c_order(&g([&co, &cv, &cv, &cv])?, nocc, nvir, nvir, nvir);
    // vvvv = (v,v|v,v)
    let vvvv = fto_c_order(&g([&cv, &cv, &cv, &cv])?, nvir, nvir, nvir, nvir);

    // Active MO energies (occ first, then vir) in column order.
    let mut mo_energy = Vec::with_capacity(nocc + nvir);
    for (col, &active) in mask.iter().enumerate() {
        if !active {
            continue;
        }
        let occ = refr.mo_occ.get(col).copied().unwrap_or(0.0);
        if occ > 0.0 {
            mo_energy.push(refr.mo_energy.get(col).copied().unwrap_or(0.0));
        }
    }
    for (col, &active) in mask.iter().enumerate() {
        if !active {
            continue;
        }
        let occ = refr.mo_occ.get(col).copied().unwrap_or(0.0);
        if occ <= 0.0 {
            mo_energy.push(refr.mo_energy.get(col).copied().unwrap_or(0.0));
        }
    }

    // Canonical reference: the MO Fock is diagonal (= mo_energy). The
    // off-diagonal Brueckner/non-canonical blocks are zero, so fock_ov = 0 and
    // init_amps' t1 seed is 0 (matching upstream for a converged RHF reference).
    let nmo = nocc + nvir;
    let mut fock = vec![0.0_f64; nmo * nmo];
    for p in 0..nmo {
        fock[p * nmo + p] = mo_energy[p];
    }

    Ok(ChemistsEris {
        oooo,
        ovoo,
        oovv,
        ovov,
        ovvo,
        ovvv,
        vvvv,
        fock,
        mo_energy,
        nocc,
        nvir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// estimate_vvvv_bytes is nv^4 * 8.
    #[test]
    fn vvvv_estimate_is_nv4_f64() {
        assert_eq!(estimate_vvvv_bytes(3), 81 * 8);
        assert_eq!(estimate_vvvv_bytes(10), 10_000 * 8);
    }

    /// The verified upstream defaults are exposed as module constants.
    #[test]
    fn convergence_defaults_match_upstream() {
        assert_eq!(MAX_CYCLE, 50);
        assert_eq!(CONV_TOL, 1e-7);
        assert_eq!(CONV_TOL_NORMT, 1e-5);
        assert_eq!(DIIS_SPACE, 6);
        assert_eq!(DIIS_START_CYCLE, 0);
    }
}
