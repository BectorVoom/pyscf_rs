//! `KRCCSD` — restricted k-point coupled cluster (plan 16-05;
//! `pyscf/pbc/cc/kccsd_rhf.py`).
//!
//! The phase's centre of gravity: 17-09 is blocked on it by name
//! (`PBC-MASTER-PLAN §8.9`), 16-08/09/10 all read its amplitudes, and the phase
//! gate is stated on its `e_corr`.
//!
//! # `LARGE_DENOM` is arithmetic, not a guard (`16-CONTEXT §3.3`)
//!
//! `_get_epq` (`kccsd_rhf.py:263-297`) fills the denominator entries of PADDED
//! orbitals with `LARGE_DENOM = 1e14` (`pyscf/lib/parameters.py:55`) rather
//! than skipping them, so a padded orbital contributes `~1e-28` to the
//! amplitude. **Skipping them is a different program** and no gate in this
//! phase would catch it: the amplitudes would differ only where the padding is,
//! and the padding is exactly where `nocc` varies between k-points.
//!
//! # The contraction primitive
//!
//! Every contraction here is an [`einsum`] subscript string transcribed from
//! the upstream line above it, so it is UNCONJUGATED by construction — see
//! `crate::zarr`'s module doc. The four places that DO conjugate are explicit
//! [`ZArr::conj`] calls, each at the upstream `.conj()` it came from:
//! `t1new = fov.conj()` (`:87`), `t2new = oovv.conj()` (`:133`), and the two
//! `transpose(3,2,1,0).conj()` / `transpose(3,2,1,0).conj()` terms at `:170`
//! and `:175`.
//!
//! # Determinism (`16-CONTEXT §3.7`)
//!
//! The `nkpts³ · nocc² · nvir²` accumulation is the D-PBC-17 shape and it goes
//! through `oracle_zsum` FROM THE FIRST VERSION — `0bcff45`'s D-PBC-17 fix had
//! to retrofit `ztrace_ab`/`trace_ab`/`trace_dm_v`, which is the more expensive
//! order. Bit-identity is gated on `t1`, `t2` AND `e_corr`, not on `e_corr`
//! alone: a non-deterministic `t2` that converges to the same energy passes an
//! energy-only gate and then fails EOM.

use std::sync::Arc;

use pyscf_algebra::oracle_sum;
use pyscf_diis::{Diis, DiisStorable};
use pyscf_pbc_lib::Kconserv;
use pyscf_pbc_mp::{PaddedMos, PaddingIdx, PaddingKind, padding_k_idx};
use pyscf_runtime::ZWorkspacePool;

use crate::error::PbcCcError;
use crate::keris::{Blk, KEris};
use crate::kintermediates_rhf as imdk;
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// `pyscf/lib/parameters.py:55` — the padded-orbital denominator. **Arithmetic,
/// not a guard.**
pub const LARGE_DENOM: f64 = 1e14;

/// Knobs of the amplitude iteration, with upstream's defaults
/// (`kccsd_rhf.py:498-547`, `pyscf/cc/ccsd.py:920-930`).
#[derive(Debug, Clone, Copy)]
pub struct KrccsdOpts {
    /// `conv_tol` — the energy-change threshold. 16-01 measured the plateau at
    /// **1e-9** (`measurements/README §3`); tightening to 1e-11 costs 46× the
    /// wall clock and moves `e_corr` by under 1e-9.
    pub conv_tol: f64,
    /// `conv_tol_normt` — the amplitude-change threshold. A SEPARATE knob with
    /// a separate default.
    pub conv_tol_normt: f64,
    /// `max_cycle`.
    pub max_cycle: usize,
    /// `level_shift`, added to the virtual orbital energies (`:65`).
    pub level_shift: f64,
    /// `diis` on/off.
    pub diis: bool,
    /// `diis_space` (`ccsd.py:926` — 6, not SCF's 8).
    pub diis_space: usize,
    /// `diis_start_cycle`.
    pub diis_start_cycle: usize,
    /// `iterative_damping` (`ccsd.py:78-88`). `1.0` disables it.
    pub iterative_damping: f64,
    /// `max_memory`, MEGABYTES.
    pub max_memory: f64,
}

impl Default for KrccsdOpts {
    fn default() -> Self {
        Self {
            conv_tol: 1e-9,
            conv_tol_normt: 1e-7,
            max_cycle: 50,
            level_shift: 0.0,
            diis: true,
            diis_space: 6,
            diis_start_cycle: 0,
            iterative_damping: 1.0,
            max_memory: 4000.0,
        }
    }
}

/// What [`kernel`] returns.
#[derive(Debug, Clone)]
pub struct KrccsdResult {
    pub e_corr: f64,
    pub emp2: f64,
    pub converged: bool,
    pub cycles: usize,
    /// `[nkpts, nocc, nvir]`.
    pub t1: ZArr,
    /// `[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]`.
    pub t2: ZArr,
}

/// The split occupied/virtual padding index sets — Phase 15's
/// `padding_k_idx(kind="split")`. **This port does NOT reimplement them**
/// (`16-CONTEXT §1.1`).
fn split_padding(padded: &PaddedMos) -> Result<(Vec<Vec<usize>>, Vec<Vec<usize>>), PbcCcError> {
    match padding_k_idx(&padded.nmo_per_kpt, &padded.nocc_per_kpt, PaddingKind::Split) {
        Ok(PaddingIdx::Split { occupied, virtuals }) => Ok((occupied, virtuals)),
        Ok(_) => Err(PbcCcError::Shape("padding_k_idx returned a joint set".into())),
        Err(e) => Err(PbcCcError::Shape(format!("padding_k_idx: {e}"))),
    }
}

/// `_get_epq` (`kccsd_rhf.py:263-297`) for the `(occ, vir)` case:
/// `e[kp, :nocc] - e[kq, nocc:]`, with PADDED entries set to
/// [`LARGE_DENOM`] rather than skipped.
fn get_eia(
    mo_e_o: &[Vec<f64>],
    mo_e_v: &[Vec<f64>],
    kp: usize,
    kq: usize,
    nonzero_o: &[Vec<usize>],
    nonzero_v: &[Vec<usize>],
) -> Vec<f64> {
    let (nocc, nvir) = (mo_e_o[kp].len(), mo_e_v[kq].len());
    let mut epq = vec![LARGE_DENOM; nocc * nvir];
    for &p in &nonzero_o[kp] {
        if p >= nocc {
            continue;
        }
        for &q in &nonzero_v[kq] {
            if q >= nvir {
                continue;
            }
            epq[p * nvir + q] = mo_e_o[kp][p] - mo_e_v[kq][q];
        }
    }
    epq
}

/// `energy(cc, t1, t2, eris)` — `kccsd_rhf.py:390-413`.
///
/// # Errors
/// Propagates the ERI access and the shape checks.
pub fn energy(t1: &ZArr, t2: &ZArr, eris: &KEris, kconserv: &Kconserv) -> Result<f64, PbcCcError> {
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    // Accumulated through `oracle_sum` over a fixed-length buffer per plane, so
    // the total is thread-count invariant (§9.3, D-PBC-17).
    let mut terms_re: Vec<f64> = Vec::new();
    let mut terms_im: Vec<f64> = Vec::new();

    for ki in 0..nkpts {
        let f = eris.fov(ki)?;
        let t = t1.slice_leading(&[ki])?;
        let e = einsum_scaled("ia,ia->", &[&f, &t], 2.0)?;
        let (re, im) = e.at(&[])?;
        terms_re.push(re);
        terms_im.push(im);
    }

    // `:398-403` — tau = t2 + einsum('ia,jb->ijab', t1[ki], t1[kj]) at ka == ki.
    let mut tau = t2.clone();
    for ki in 0..nkpts {
        let ka = ki;
        for kj in 0..nkpts {
            let mut blk = tau.slice_leading(&[ki, kj, ka])?;
            let a = t1.slice_leading(&[ki])?;
            let b = t1.slice_leading(&[kj])?;
            blk.add_assign(&einsum("ia,jb->ijab", &[&a, &b])?)?;
            tau.set_leading(&[ki, kj, ka], &blk)?;
        }
    }
    let _ = (nocc, nvir);

    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let tt = tau.slice_leading(&[ki, kj, ka])?;
                let a = eris.blk(Blk::Oovv, ki, kj, ka)?;
                let (re, im) = einsum_scaled("ijab,ijab->", &[&tt, &a], 2.0)?.at(&[])?;
                terms_re.push(re);
                terms_im.push(im);
                let b = eris.blk(Blk::Oovv, ki, kj, kb)?;
                let (re, im) = einsum_scaled("ijab,ijba->", &[&tt, &b], -1.0)?.at(&[])?;
                terms_re.push(re);
                terms_im.push(im);
            }
        }
    }
    let re = oracle_sum(&terms_re) / nkpts as f64;
    let im = oracle_sum(&terms_im) / nkpts as f64;
    if im.abs() > 1e-4 {
        tracing::warn!(
            imaginary = im,
            "non-zero imaginary part in the KRCCSD energy (kccsd_rhf.py:411)"
        );
    }
    Ok(re)
}

/// `init_amps(eris)` — `kccsd_rhf.py:548-597`. Returns `(emp2, t1, t2)`.
///
/// # Errors
/// Propagates the ERI access and the shape checks.
pub fn init_amps(
    eris: &KEris,
    padded: &PaddedMos,
    kconserv: &Kconserv,
) -> Result<(f64, ZArr, ZArr), PbcCcError> {
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let t1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    let mut t2 = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();
    let (nz_o, nz_v) = split_padding(padded)?;

    let mut emp2_terms: Vec<f64> = Vec::new();
    let mut touched = vec![false; nkpts * nkpts * nkpts];
    let at = |a: usize, b: usize, c: usize| (a * nkpts + b) * nkpts + c;

    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                if touched[at(ki, kj, ka)] {
                    continue;
                }
                let kb = kconserv.get(ki, ka, kj) as usize;
                let eia = get_eia(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v);
                let ejb = get_eia(&mo_e_o, &mo_e_v, kj, kb, &nz_o, &nz_v);

                let eris_ijab = eris.blk(Blk::Oovv, ki, kj, ka)?;
                let eris_ijba = eris.blk(Blk::Oovv, ki, kj, kb)?;

                let t_ka = divide_by_eijab(&eris_ijab.conj(), &eia, &ejb, nocc, nvir, false)?;
                let mut woovv = eris_ijab.clone();
                woovv.scale(2.0);
                woovv.sub_assign(&eris_ijba.transpose(&[0, 1, 3, 2])?)?;
                let (re, _) = einsum("ijab,ijab->", &[&t_ka, &woovv])?.at(&[])?;
                emp2_terms.push(re);
                t2.set_leading(&[ki, kj, ka], &t_ka)?;
                touched[at(ki, kj, ka)] = true;

                if ka != kb {
                    let t_kb =
                        divide_by_eijab(&eris_ijba.conj(), &eia, &ejb, nocc, nvir, true)?;
                    let mut woovv = eris_ijba.clone();
                    woovv.scale(2.0);
                    woovv.sub_assign(&eris_ijab.transpose(&[0, 1, 3, 2])?)?;
                    let (re, _) = einsum("ijab,ijab->", &[&t_kb, &woovv])?.at(&[])?;
                    emp2_terms.push(re);
                    t2.set_leading(&[ki, kj, kb], &t_kb)?;
                    touched[at(ki, kj, kb)] = true;
                }
            }
        }
    }
    Ok((oracle_sum(&emp2_terms) / nkpts as f64, t1, t2))
}

/// `x / eijab` where `eijab[i,j,a,b] = eia[i,a] + ejb[j,b]`, or its
/// `transpose(0,1,3,2)` when `swap` — `kccsd_rhf.py:583`/`:589`.
fn divide_by_eijab(
    x: &ZArr,
    eia: &[f64],
    ejb: &[f64],
    nocc: usize,
    nvir: usize,
    swap: bool,
) -> Result<ZArr, PbcCcError> {
    let mut out = x.clone();
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let d = if swap {
                        eia[i * nvir + b] + ejb[j * nvir + a]
                    } else {
                        eia[i * nvir + a] + ejb[j * nvir + b]
                    };
                    let f = ((i * nocc + j) * nvir + a) * nvir + b;
                    out.data_mut().re[f] /= d;
                    out.data_mut().im[f] /= d;
                }
            }
        }
    }
    Ok(out)
}

/// `update_amps(cc, t1, t2, eris)` — `kccsd_rhf.py:64-228`.
///
/// # Errors
/// Propagates every ERI access, intermediate build and shape check.
#[allow(clippy::too_many_arguments)]
pub fn update_amps(
    pool: &Arc<ZWorkspacePool>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KEris,
    padded: &PaddedMos,
    kconserv: &Kconserv,
    opts: &KrccsdOpts,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let budget = (opts.max_memory * 1e6 * 0.9).max(0.0) as usize;
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris
        .mo_energy
        .iter()
        .map(|e| e[nocc..].iter().map(|x| x + opts.level_shift).collect())
        .collect();
    let (nz_o, nz_v) = split_padding(padded)?;

    let mut foo = imdk::cc_foo(t1, t2, eris, kconserv)?;
    let mut fvv = imdk::cc_fvv(t1, t2, eris, kconserv)?;
    let fov = imdk::cc_fov(t1, t2, eris)?;
    let mut loo = imdk::loo(t1, t2, eris, kconserv)?;
    let mut lvv = imdk::lvv(t1, t2, eris, kconserv)?;

    // `:79-83` — move the energy terms to the other side.
    for k in 0..nkpts {
        shift_diagonal(&mut foo, k, &mo_e_o[k])?;
        shift_diagonal(&mut loo, k, &mo_e_o[k])?;
        shift_diagonal(&mut fvv, k, &mo_e_v[k])?;
        shift_diagonal(&mut lvv, k, &mo_e_v[k])?;
    }

    // ---------------------------------------------------------------- T1
    // `:87` t1new = fov.conj() — the FIRST of the four explicit conjugations.
    let mut t1new = ZArr::zeros(&[nkpts, nocc, nvir]);
    for k in 0..nkpts {
        t1new.set_leading(&[k], &eris.fov(k)?.conj())?;
    }

    for ka in 0..nkpts {
        let ki = ka;
        let mut acc = t1new.slice_leading(&[ka])?;
        let f = eris.fov(ki)?;
        let t1a = t1.slice_leading(&[ka])?;
        let t1i = t1.slice_leading(&[ki])?;
        acc.add_assign(&einsum_scaled("kc,ka,ic->ia", &[&f, &t1a, &t1i], -2.0)?)?;
        acc.add_assign(&einsum("ac,ic->ia", &[&fvv.slice_leading(&[ka])?, &t1i])?)?;
        acc.sub_assign(&einsum("ki,ka->ia", &[&foo.slice_leading(&[ki])?, &t1a])?)?;

        // `:96-100` — the tau term.
        let mut tau_term = ZArr::zeros(&[nkpts, nocc, nocc, nvir, nvir]);
        for kk in 0..nkpts {
            let mut b = t2.slice_leading(&[kk, ki, kk])?;
            b.scale(2.0);
            b.sub_assign(&t2.slice_leading(&[ki, kk, kk])?.transpose(&[1, 0, 2, 3])?)?;
            tau_term.set_leading(&[kk], &b)?;
        }
        let mut ta = tau_term.slice_leading(&[ka])?;
        ta.add_assign(&einsum("ic,ka->kica", &[&t1i, &t1a])?)?;
        tau_term.set_leading(&[ka], &ta)?;

        for kk in 0..nkpts {
            let kc = kk;
            acc.add_assign(&einsum(
                "kc,kica->ia",
                &[&fov.slice_leading(&[kc])?, &tau_term.slice_leading(&[kk])?],
            )?)?;
            let v = eris.blk(Blk::Voov, ka, kk, ki)?;
            acc.add_assign(&einsum_scaled(
                "akic,kc->ia",
                &[&v, &t1.slice_leading(&[kc])?],
                2.0,
            )?)?;
            let o = eris.blk(Blk::Ovov, kk, ka, ki)?;
            acc.sub_assign(&einsum("kaic,kc->ia", &[&o, &t1.slice_leading(&[kc])?])?)?;

            for kc in 0..nkpts {
                let kd = kconserv.get(ka, kc, kk) as usize;
                let mut svovv = eris.blk(Blk::Vovv, ka, kk, kc)?;
                svovv.scale(2.0);
                svovv.sub_assign(&eris.blk(Blk::Vovv, ka, kk, kd)?.transpose(&[0, 1, 3, 2])?)?;
                let mut tau1 = t2.slice_leading(&[ki, kk, kc])?;
                if ki == kc && kk == kd {
                    tau1.add_assign(&einsum(
                        "ic,kd->ikcd",
                        &[&t1i, &t1.slice_leading(&[kk])?],
                    )?)?;
                }
                acc.add_assign(&einsum("akcd,ikcd->ia", &[&svovv, &tau1])?)?;

                let kl = kconserv.get(ki, kk, kc) as usize;
                let mut sooov = eris.blk(Blk::Ooov, kk, kl, ki)?;
                sooov.scale(2.0);
                sooov.sub_assign(&eris.blk(Blk::Ooov, kl, kk, ki)?.transpose(&[1, 0, 2, 3])?)?;
                let mut tau1 = t2.slice_leading(&[kk, kl, ka])?;
                if kk == ka && kl == kc {
                    tau1.add_assign(&einsum(
                        "ka,lc->klac",
                        &[&t1a, &t1.slice_leading(&[kc])?],
                    )?)?;
                }
                acc.sub_assign(&einsum("klic,klac->ia", &[&sooov, &tau1])?)?;
            }
        }
        t1new.set_leading(&[ka], &acc)?;
    }

    // ---------------------------------------------------------------- T2
    // `:131-133` t2new = oovv.conj() — the SECOND explicit conjugation.
    let mut t2new = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                t2new.set_leading(
                    &[ki, kj, ka],
                    &eris.blk(Blk::Oovv, ki, kj, ka)?.conj(),
                )?;
            }
        }
    }

    // oooo ladder
    {
        let woooo = imdk::cc_woooo(pool, t1, t2, eris, kconserv, budget)?;
        for ki in 0..nkpts {
            for kj in 0..nkpts {
                for ka in 0..nkpts {
                    let kb = kconserv.get(ki, ka, kj) as usize;
                    let mut tmp = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
                    for kl in 0..nkpts {
                        let kk = kconserv.get(kj, kl, ki) as usize;
                        let mut tau = t2.slice_leading(&[kk, kl, ka])?;
                        if kl == kb && kk == ka {
                            tau.add_assign(&einsum(
                                "ic,jd->ijcd",
                                &[&t1.slice_leading(&[ka])?, &t1.slice_leading(&[kb])?],
                            )?)?;
                        }
                        tmp.add_assign(&einsum_scaled(
                            "klij,klab->ijab",
                            &[&woooo.get([kk, kl, ki])?, &tau],
                            0.5,
                        )?)?;
                    }
                    accumulate_pair(&mut t2new, ki, kj, ka, kb, &tmp)?;
                }
            }
        }
        woooo.release();
    }

    // vvvv ladder — `add_vvvv_` (`kccsd_rhf.py:416-481`).
    {
        let wvvvv = imdk::cc_wvvvv(pool, t1, t2, eris, kconserv, budget)?;
        for ka in 0..nkpts {
            for kb in 0..nkpts {
                for kc in 0..nkpts {
                    let kd = kconserv.get(ka, kc, kb) as usize;
                    let w = wvvvv.get([ka, kb, kc])?;
                    for ki in 0..nkpts {
                        let kj = kconserv.get(ka, ki, kb) as usize;
                        let mut tau = t2.slice_leading(&[ki, kj, kc])?;
                        if ki == kc && kj == kd {
                            tau.add_assign(&einsum(
                                "ic,jd->ijcd",
                                &[&t1.slice_leading(&[ki])?, &t1.slice_leading(&[kj])?],
                            )?)?;
                        }
                        let mut blk = t2new.slice_leading(&[ki, kj, ka])?;
                        blk.add_assign(&einsum("abcd,ijcd->ijab", &[&w, &tau])?)?;
                        t2new.set_leading(&[ki, kj, ka], &blk)?;
                    }
                }
            }
        }
        wvvvv.release();
    }

    // the L / singles-dressed terms — `:164-180`
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let t2ija = t2.slice_leading(&[ki, kj, ka])?;
                let mut tmp = einsum("ac,ijcb->ijab", &[&lvv.slice_leading(&[ka])?, &t2ija])?;
                let mut nloo = loo.slice_leading(&[ki])?;
                nloo.scale(-1.0);
                tmp.add_assign(&einsum("ki,kjab->ijab", &[&nloo, &t2ija])?)?;

                // `:170` — the THIRD explicit conjugation.
                let kc = kconserv.get(ka, ki, kb) as usize;
                let mut tmp2 = eris
                    .blk(Blk::Vovv, kc, ki, kb)?
                    .transpose(&[3, 2, 1, 0])?
                    .conj();
                tmp2.sub_assign(&einsum(
                    "kbic,ka->abic",
                    &[&eris.blk(Blk::Ovov, ka, kb, ki)?, &t1.slice_leading(&[ka])?],
                )?)?;
                tmp.add_assign(&einsum(
                    "abic,jc->ijab",
                    &[&tmp2, &t1.slice_leading(&[kj])?],
                )?)?;

                // `:175` — the FOURTH explicit conjugation.
                let kk = kconserv.get(ki, ka, kj) as usize;
                let mut tmp2 = eris
                    .blk(Blk::Ooov, kj, ki, kk)?
                    .transpose(&[3, 2, 1, 0])?
                    .conj();
                tmp2.add_assign(&einsum(
                    "akic,jc->akij",
                    &[&eris.blk(Blk::Voov, ka, kk, ki)?, &t1.slice_leading(&[kj])?],
                )?)?;
                tmp.sub_assign(&einsum(
                    "akij,kb->ijab",
                    &[&tmp2, &t1.slice_leading(&[kb])?],
                )?)?;

                accumulate_pair(&mut t2new, ki, kj, ka, kb, &tmp)?;
            }
        }
    }

    // the voov / vovo ring terms — `:182-206`
    {
        let wvoov = imdk::cc_wvoov(pool, t1, t2, eris, kconserv, budget)?;
        let wvovo = imdk::cc_wvovo(pool, t1, t2, eris, kconserv, budget)?;
        for ki in 0..nkpts {
            for kj in 0..nkpts {
                for ka in 0..nkpts {
                    let kb = kconserv.get(ki, ka, kj) as usize;
                    let mut tmp = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
                    for kk in 0..nkpts {
                        let kc = kconserv.get(ka, ki, kk) as usize;
                        let mut tv = wvoov.get([ka, kk, ki])?;
                        tv.scale(2.0);
                        tv.sub_assign(&wvovo.get([ka, kk, kc])?.transpose(&[0, 1, 3, 2])?)?;
                        tmp.add_assign(&einsum(
                            "akic,kjcb->ijab",
                            &[&tv, &t2.slice_leading(&[kk, kj, kc])?],
                        )?)?;

                        tmp.sub_assign(&einsum(
                            "akic,kjbc->ijab",
                            &[
                                &wvoov.get([ka, kk, ki])?,
                                &t2.slice_leading(&[kk, kj, kb])?,
                            ],
                        )?)?;

                        let kc2 = kconserv.get(kk, ka, kj) as usize;
                        tmp.sub_assign(&einsum(
                            "bkci,kjac->ijab",
                            &[
                                &wvovo.get([kb, kk, kc2])?,
                                &t2.slice_leading(&[kk, kj, ka])?,
                            ],
                        )?)?;
                    }
                    accumulate_pair(&mut t2new, ki, kj, ka, kb, &tmp)?;
                }
            }
        }
        wvoov.release();
        wvovo.release();
    }

    // ------------------------------------------------- the LARGE_DENOM divide
    for ki in 0..nkpts {
        let ka = ki;
        let eia = get_eia(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v);
        let mut blk = t1new.slice_leading(&[ki])?;
        for i in 0..nocc {
            for a in 0..nvir {
                let f = i * nvir + a;
                blk.data_mut().re[f] /= eia[f];
                blk.data_mut().im[f] /= eia[f];
            }
        }
        t1new.set_leading(&[ki], &blk)?;
    }
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let eia = get_eia(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v);
                let ejb = get_eia(&mo_e_o, &mo_e_v, kj, kb, &nz_o, &nz_v);
                let blk = t2new.slice_leading(&[ki, kj, ka])?;
                let out = divide_by_eijab(&blk, &eia, &ejb, nocc, nvir, false)?;
                t2new.set_leading(&[ki, kj, ka], &out)?;
            }
        }
    }

    Ok((t1new, t2new))
}

/// `t2new[ki,kj,ka] += tmp;  t2new[kj,ki,kb] += tmp.transpose(1,0,3,2)` —
/// the pair-accumulation upstream repeats at `:150-151`, `:178-179`, `:205-206`.
fn accumulate_pair(
    t2new: &mut ZArr,
    ki: usize,
    kj: usize,
    ka: usize,
    kb: usize,
    tmp: &ZArr,
) -> Result<(), PbcCcError> {
    let mut a = t2new.slice_leading(&[ki, kj, ka])?;
    a.add_assign(tmp)?;
    t2new.set_leading(&[ki, kj, ka], &a)?;
    let mut b = t2new.slice_leading(&[kj, ki, kb])?;
    b.add_assign(&tmp.transpose(&[1, 0, 3, 2])?)?;
    t2new.set_leading(&[kj, ki, kb], &b)?;
    Ok(())
}

/// `X[k][diag_indices] -= mo_e` — `kccsd_rhf.py:80-83`.
fn shift_diagonal(x: &mut ZArr, k: usize, e: &[f64]) -> Result<(), PbcCcError> {
    let mut blk = x.slice_leading(&[k])?;
    let n = e.len();
    for i in 0..n {
        blk.data_mut().re[i * n + i] -= e[i];
    }
    x.set_leading(&[k], &blk)
}

/// The DIIS iterate: `t1` and `t2` flattened plane-by-plane.
///
/// CDIIS forms a REAL linear combination of stored iterates, and a real
/// combination of complex vectors is exactly the same operation applied to each
/// plane — so packing `[re…, im…]` loses nothing and lets the whole Phase-3
/// `pyscf-diis` machinery be reused with no new DIIS body, exactly as
/// `pyscf-ccsd`'s `AmplitudeSubspace` does for the molecular case.
#[derive(Debug, Clone)]
struct KAmplitudeSubspace {
    flat: Vec<f64>,
}

impl KAmplitudeSubspace {
    fn from_amplitudes(t1: &ZArr, t2: &ZArr) -> Self {
        let mut flat = Vec::with_capacity(2 * (t1.len() + t2.len()));
        flat.extend_from_slice(&t1.data().re);
        flat.extend_from_slice(&t1.data().im);
        flat.extend_from_slice(&t2.data().re);
        flat.extend_from_slice(&t2.data().im);
        Self { flat }
    }

    fn to_amplitudes(&self, t1: &ZArr, t2: &ZArr) -> (ZArr, ZArr) {
        let n1 = t1.len();
        let n2 = t2.len();
        let mut a = t1.clone();
        let mut b = t2.clone();
        a.data_mut().re.copy_from_slice(&self.flat[..n1]);
        a.data_mut().im.copy_from_slice(&self.flat[n1..2 * n1]);
        b.data_mut()
            .re
            .copy_from_slice(&self.flat[2 * n1..2 * n1 + n2]);
        b.data_mut()
            .im
            .copy_from_slice(&self.flat[2 * n1 + n2..2 * n1 + 2 * n2]);
        (a, b)
    }

    fn residual(&self, prev: &Self) -> Vec<f64> {
        self.flat
            .iter()
            .zip(prev.flat.iter())
            .map(|(a, b)| a - b)
            .collect()
    }
}

impl DiisStorable for KAmplitudeSubspace {
    fn as_flat(&self) -> &[f64] {
        &self.flat
    }
    fn from_flat(&mut self, slice: &[f64]) {
        self.flat.copy_from_slice(slice);
    }
    fn dot(&self, other: &Self) -> f64 {
        pyscf_algebra::oracle_dot(&self.flat, &other.flat)
    }
    fn len(&self) -> usize {
        self.flat.len()
    }
}

/// `pyscf.cc.ccsd.kernel` driven with the k-point `update_amps` / `energy`
/// (`pyscf/cc/ccsd.py:44-101`, reached through `kccsd_rhf.py:56`).
///
/// # Errors
/// Propagates every intermediate build, and the DIIS solve.
pub fn kernel(
    pool: &Arc<ZWorkspacePool>,
    eris: &KEris,
    padded: &PaddedMos,
    kconserv: &Kconserv,
    opts: &KrccsdOpts,
) -> Result<KrccsdResult, PbcCcError> {
    let (emp2, mut t1, mut t2) = init_amps(eris, padded, kconserv)?;
    let mut eccsd = energy(&t1, &t2, eris, kconserv)?;
    let mut converged = false;
    let mut cycles = 0_usize;

    let mut diis: Option<Diis<KAmplitudeSubspace>> = if opts.diis {
        Some(Diis::new(opts.diis_space))
    } else {
        None
    };

    for istep in 0..opts.max_cycle {
        cycles = istep + 1;
        let (mut t1new, mut t2new) =
            update_amps(pool, &t1, &t2, eris, padded, kconserv, opts)?;

        // `ccsd.py:74-76` normt = ||vector(new) - vector(old)||, through
        // `oracle_dot` so it is thread-count invariant.
        let cur = KAmplitudeSubspace::from_amplitudes(&t1new, &t2new);
        let prev = KAmplitudeSubspace::from_amplitudes(&t1, &t2);
        let res = cur.residual(&prev);
        let normt = pyscf_algebra::oracle_dot(&res, &res).sqrt();

        // `ccsd.py:78-88` iterative_damping.
        if opts.iterative_damping < 1.0 {
            let a = opts.iterative_damping;
            t1new.scale(a);
            t1new.zip_assign(&t1, 1.0 - a)?;
            t2new.scale(a);
            t2new.zip_assign(&t2, 1.0 - a)?;
        }

        t1 = t1new;
        t2 = t2new;

        if let Some(stack) = diis.as_mut()
            && istep >= opts.diis_start_cycle
        {
            let cur = KAmplitudeSubspace::from_amplitudes(&t1, &t2);
            let err = cur.residual(&prev);
            let extrap = stack
                .extrapolate(cur, err)
                .map_err(|e| PbcCcError::Algebra(format!("amplitude DIIS: {e}")))?;
            let (a, b) = extrap.to_amplitudes(&t1, &t2);
            t1 = a;
            t2 = b;
        }

        let eold = eccsd;
        eccsd = energy(&t1, &t2, eris, kconserv)?;
        if (eccsd - eold).abs() < opts.conv_tol && normt < opts.conv_tol_normt {
            converged = true;
            break;
        }
    }

    Ok(KrccsdResult {
        e_corr: eccsd,
        emp2,
        converged,
        cycles,
        t1,
        t2,
    })
}

/// `KRCCSD` — the object `kccsd_rhf.py:498` declares, tying a converged k-point
/// mean field to the amplitude iteration.
///
/// `khelper` is constructed WITHOUT the symmetry map (`kccsd_rhf.py:512` passes
/// `init_symm_map=False`); [`KEris::new`] builds it lazily, so a `Krccsd`
/// constructed and never run does not pay `O(nkpts³)`.
#[derive(Debug)]
pub struct Krccsd<'a> {
    pub with_df: &'a dyn pyscf_pbc_df::PeriodicDf,
    pub khelper: pyscf_pbc_lib::KptsHelper,
    pub padded: PaddedMos,
    pub frozen: pyscf_pbc_mp::FrozenK,
    pub opts: KrccsdOpts,
    pub eris_opts: crate::keris::KErisOpts,
    dm: Vec<pyscf_algebra::CTensor>,
    e_hf: f64,
    converged: bool,
}

impl<'a> Krccsd<'a> {
    /// Build from a converged restricted k-point SCF.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if the SCF is not a single restricted channel over
    /// this builder's k-points, or if the padding surface refuses.
    pub fn new(
        scf: &'a pyscf_pbc_scf::KScfResult,
        with_df: &'a dyn pyscf_pbc_df::PeriodicDf,
    ) -> Result<Self, PbcCcError> {
        if scf.nset != 1 || scf.nkpts != with_df.kpts().len() {
            return Err(PbcCcError::Shape(
                "KRCCSD needs one restricted SCF channel over with_df.kpts()".into(),
            ));
        }
        let cell = with_df.cell();
        let nao = cell.mol.nao_nr;
        let mf = pyscf_pbc_mp::spin_block(scf, 0)
            .map_err(|e| PbcCcError::Shape(format!("spin_block: {e}")))?;
        let raw: Result<Vec<pyscf_pbc_df::MoCoeff>, _> = mf
            .mo_coeff
            .iter()
            .zip(mf.mo_occ)
            .map(|(c, occ)| pyscf_pbc_mp::mo_coeff_from_kscf(c, nao, occ.len()))
            .collect();
        let raw = raw.map_err(|e| PbcCcError::Shape(format!("mo_coeff_from_kscf: {e}")))?;
        let frozen = pyscf_pbc_mp::FrozenK::default();
        let padded = pyscf_pbc_mp::add_padding(&raw, mf.mo_energy, mf.mo_occ, &frozen)
            .map_err(|e| PbcCcError::Shape(format!("add_padding: {e}")))?;
        // The density matrix is built from the UNPADDED, unfrozen orbitals —
        // `kccsd_rhf.py:741` uses `cc.mo_coeff` / `cc.mo_occ`, i.e. the mean
        // field's own, not the padded ones.
        let dm = pyscf_pbc_scf::krdm::make_rdm1(mf.mo_coeff, mf.mo_occ, nao);
        Ok(Self {
            with_df,
            khelper: pyscf_pbc_lib::KptsHelper::without_symm_map(&cell.a, with_df.kpts()),
            padded,
            frozen,
            opts: KrccsdOpts::default(),
            eris_opts: crate::keris::KErisOpts::default(),
            dm,
            e_hf: scf.e_tot,
            converged: scf.converged,
        })
    }

    /// `cc.ao2mo()` — build the seven blocks.
    ///
    /// # Errors
    /// Propagates the density-fitting builder and the arena.
    pub fn ao2mo(&mut self) -> Result<KEris, PbcCcError> {
        // `eris_opts.max_memory` is the ERI build's OWN budget and is the
        // authority here. It is initialised from `opts.max_memory` in
        // [`Krccsd::new`] so the two agree by default; overwriting it on every
        // call would make the storage tier unreachable from a caller, which is
        // exactly what 16-05 test 4 has to set.
        let opts = self.eris_opts;
        KEris::new(
            self.with_df.cell(),
            self.with_df,
            &mut self.khelper,
            &self.padded,
            &self.dm,
            opts,
        )
    }

    /// Run the amplitude iteration.
    ///
    /// # Errors
    /// [`PbcCcError::NotConverged`] if the reference SCF did not converge;
    /// otherwise propagates the kernel.
    pub fn kernel(&mut self) -> Result<KrccsdResult, PbcCcError> {
        if !self.converged {
            return Err(PbcCcError::NotConverged {
                what: "the reference k-point SCF",
                detail: "KRCCSD refuses an unconverged mean field".into(),
            });
        }
        let eris = self.ao2mo()?;
        self.kernel_with(&eris)
    }

    /// Run the amplitude iteration on an already-built [`KEris`].
    ///
    /// # Errors
    /// Propagates the kernel.
    pub fn kernel_with(&self, eris: &KEris) -> Result<KrccsdResult, PbcCcError> {
        let pool = Arc::new(ZWorkspacePool::new(
            (self.opts.max_memory * 1e6).max(0.0) as usize,
        ));
        kernel(&pool, eris, &self.padded, &self.khelper.kconserv, &self.opts)
    }

    /// The mean-field total energy the correlation energy adds to.
    pub fn e_hf(&self) -> f64 {
        self.e_hf
    }
}
