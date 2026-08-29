//! J and K with discrete Fourier transformation — plans 11-06 / 11-07.
//!
//! Statement-for-statement port of `pyscf/pbc/df/fft_jk.py:33-112`
//! (`get_j_kpts`) and `:181-309` (`get_k_kpts`). PBC-MASTER-PLAN's warning about
//! `get_k_kpts` — "the most bug-prone routine in the whole milestone; port it
//! statement-by-statement, do not restructure the loops" — is why the upstream
//! variable names (`ao1T`, `ao2T`, `ao_dms`, `rho1`, `vR_dm`, `expmikr`) appear
//! verbatim as identifiers below.
//!
//! # Layout
//!
//! Every AO block is `(nao, ngrids)` ROW-MAJOR, which is exactly upstream's
//! `ao.T` and exactly what `eval_ao_kpts` produces. Every `nao x nao` matrix is
//! ROW-MAJOR (see `zlinalg::forder_to_c` for the boundary with Phase 10).

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{get_coulg, get_gv, is_zero, CoulGArgs, ExxDiv};
use pyscf_pbc_tools::{fft, ifft};

use crate::df_jk::{all_gamma, ewald_exxdiv_for_g0, format_kpts_band, KMats};
use crate::error::PbcDfError;
use crate::fftdf::Fftdf;
use crate::zlinalg::zscale_real;

/// `get_j_kpts(mydf, dm_kpts, hermi, kpts, kpts_band)` — `fft_jk.py:33-112`.
///
/// Returns `vj[iset][kband]`, `nao x nao` row-major.
///
/// # Errors
/// Propagates the AO evaluation, `get_coulG` and the FFT.
pub fn get_j_kpts(
    df: &Fftdf,
    dms: &[KMats],
    hermi: i32,
    kpts: &[[f64; 3]],
    kpts_band: Option<&[[f64; 3]]>,
    omega: Option<f64>,
) -> Result<Vec<KMats>, PbcDfError> {
    let cell = &df.cell;
    let mesh = df.mesh;
    let nset = dms.len();
    let nkpts = kpts.len();
    let nao = cell.mol.nao_nr;

    // fft_jk.py:60-61. `omega` reaches `get_coulG` the way upstream's
    // `range_coulomb` delivers it through `cell.omega` (`pbc/df/fft.py`).
    let gv = get_gv(cell, Some(mesh))?;
    let coulg = get_coulg(
        cell,
        CoulGArgs {
            mesh: Some(mesh),
            gv: Some(&gv),
            omega,
            ..CoulGArgs::new()
        },
    )?;
    let ngrids = coulg.len();

    let ao = df.ao_kpts(kpts)?;
    // fft_jk.py:63 — the density is REAL when the DMs are Hermitian or the
    // sampling is gamma-only. Upstream keeps two separate branches; the
    // arithmetic is identical up to the `.real` truncation, which is what the
    // `real_rho` flag selects here.
    let real_rho = hermi == 1 || all_gamma(kpts);

    let mut vr: Vec<CTensor> = Vec::with_capacity(nset);
    for dmset in dms.iter().take(nset) {
        let mut rho = CTensor::zeros(ngrids);
        for k in 0..nkpts {
            accumulate_rho(&mut rho, ao.at(k), &dmset[k], nao, ngrids);
        }
        // fft_jk.py:84 — rhoR *= 1./nkpts
        zscale_real(&mut rho, 1.0 / nkpts as f64);
        if real_rho {
            for v in rho.im.iter_mut() {
                *v = 0.0;
            }
        }
        // fft_jk.py:86-89 / :106-109
        let mut rhog = fft(&rho, mesh)?;
        for g in 0..ngrids {
            rhog.re[g] *= coulg[g];
            rhog.im[g] *= coulg[g];
        }
        let mut v = ifft(&rhog, mesh)?;
        if real_rho {
            for t in v.im.iter_mut() {
                *t = 0.0;
            }
        }
        vr.push(v);
    }

    // fft_jk.py:91-94 — weight = vol/ngrids; vR *= weight.
    let weight = df.weight();
    for v in vr.iter_mut() {
        zscale_real(v, weight);
    }

    let band = format_kpts_band(kpts_band, kpts);
    let ao_band = if kpts_band.is_none() {
        ao
    } else {
        df.ao_kpts(band)?
    };

    // fft_jk.py:100-110
    let mut out: Vec<KMats> = Vec::with_capacity(nset);
    for v in vr.iter() {
        let mut per_k = Vec::with_capacity(band.len());
        for k in 0..band.len() {
            per_k.push(contract_ao_v_ao(ao_band.at(k), v, nao, ngrids));
        }
        out.push(per_k);
    }
    Ok(out)
}

/// `rho[g] += sum_{mu,nu} conj(ao[mu,g]) dm[nu,mu] ao[nu,g]` — upstream's
/// `ao_dm = ao . dm; rho += einsum('xi,xi->x', ao_dm, ao.conj())`
/// (`fft_jk.py:81-83`), transposed into the `(nao, ngrids)` layout.
fn accumulate_rho(rho: &mut CTensor, ao: &CTensor, dm: &CTensor, nao: usize, ngrids: usize) {
    // c0[mu, g] = sum_nu dm[nu, mu] ao[nu, g]
    let mut c0_re = vec![0.0_f64; nao * ngrids];
    let mut c0_im = vec![0.0_f64; nao * ngrids];
    for nu in 0..nao {
        let ab = nu * ngrids;
        for mu in 0..nao {
            let (dr, di) = (dm.re[nu * nao + mu], dm.im[nu * nao + mu]);
            if dr == 0.0 && di == 0.0 {
                continue;
            }
            let cb = mu * ngrids;
            for g in 0..ngrids {
                let (ar, ai) = (ao.re[ab + g], ao.im[ab + g]);
                c0_re[cb + g] += dr * ar - di * ai;
                c0_im[cb + g] += dr * ai + di * ar;
            }
        }
    }
    for mu in 0..nao {
        let b = mu * ngrids;
        for g in 0..ngrids {
            let (br, bi) = (ao.re[b + g], -ao.im[b + g]);
            let (cr, ci) = (c0_re[b + g], c0_im[b + g]);
            rho.re[g] += br * cr - bi * ci;
            rho.im[g] += br * ci + bi * cr;
        }
    }
}

/// `v[p, q] = sum_g conj(ao[p, g]) vR[g] ao[q, g]` — upstream's
/// `aow = ao * vR; v += lib.dot(ao.conj().T, aow)` (`fft_jk.py:107-109`).
fn contract_ao_v_ao(ao: &CTensor, vr: &CTensor, nao: usize, ngrids: usize) -> CTensor {
    // aow[q, g] = ao[q, g] * vR[g]
    let mut aow_re = vec![0.0_f64; nao * ngrids];
    let mut aow_im = vec![0.0_f64; nao * ngrids];
    for q in 0..nao {
        let b = q * ngrids;
        for g in 0..ngrids {
            let (ar, ai) = (ao.re[b + g], ao.im[b + g]);
            let (vrr, vri) = (vr.re[g], vr.im[g]);
            aow_re[b + g] = ar * vrr - ai * vri;
            aow_im[b + g] = ar * vri + ai * vrr;
        }
    }
    let mut re = vec![0.0_f64; nao * nao];
    let mut im = vec![0.0_f64; nao * nao];
    for p in 0..nao {
        let pb = p * ngrids;
        for q in 0..nao {
            let qb = q * ngrids;
            let mut sr = 0.0_f64;
            let mut si = 0.0_f64;
            for g in 0..ngrids {
                let (pr, pi) = (ao.re[pb + g], -ao.im[pb + g]);
                let (qr, qi) = (aow_re[qb + g], aow_im[qb + g]);
                sr += pr * qr - pi * qi;
                si += pr * qi + pi * qr;
            }
            re[p * nao + q] = sr;
            im[p * nao + q] = si;
        }
    }
    CTensor::from_planes(re, im)
}

/// `get_k_kpts(mydf, dm_kpts, hermi, kpts, kpts_band, exxdiv)` —
/// `fft_jk.py:181-309`.
///
/// Returns `vk[iset][kband]`, `nao x nao` row-major.
///
/// # Errors
/// Propagates the AO evaluation, `get_coulG`, the FFT and — for
/// `exxdiv = Ewald` — `madelung`.
pub fn get_k_kpts(
    df: &Fftdf,
    dms: &[KMats],
    _hermi: i32,
    kpts: &[[f64; 3]],
    kpts_band: Option<&[[f64; 3]]>,
    exxdiv: Option<ExxDiv>,
    omega: Option<f64>,
) -> Result<Vec<KMats>, PbcDfError> {
    let cell = &df.cell;
    let mesh = df.mesh;
    let nset = dms.len();
    let nkpts = kpts.len();
    let nao = cell.mol.nao_nr;
    let coords = &df.grids.coords;
    let ngrids = coords.len();

    // fft_jk.py:223
    let weight = 1.0 / nkpts as f64 * df.weight();

    let band = format_kpts_band(kpts_band, kpts);
    let nband = band.len();

    // fft_jk.py:228-231 — the result is REAL only when everything is gamma.
    let real_out = all_gamma(band) && all_gamma(kpts);

    let ao2_kpts = df.ao_kpts(kpts)?;
    let ao1_kpts = if kpts_band.is_none() {
        ao2_kpts.clone()
    } else {
        df.ao_kpts(band)?
    };

    let gv = get_gv(cell, Some(mesh))?;

    // fft_jk.py:246-248 — the AO block size, memory-derived. It bounds the
    // `(blksize, nao, ngrids)` scratch and changes nothing numerically.
    let budget = (df.max_memory * 1e6 / 16.0 / 4.0 / ngrids as f64 / nao as f64) as usize;
    let blksize = nao.min(budget.max(1));

    let mut vk_kpts: Vec<KMats> =
        vec![vec![CTensor::zeros(nao * nao); nband]; nset];

    // fft_jk.py:256 — for k2, ao2T in enumerate(ao2_kpts)
    for k2 in 0..nkpts {
        let ao2t = ao2_kpts.at(k2);
        // fft_jk.py:263-266 — ao_dms[i] = dms[i,k2] . conj(ao2T)
        let ao_dms: Vec<CTensor> = dms
            .iter()
            .map(|d| dm_times_conj_ao(&d[k2], ao2t, nao, ngrids))
            .collect();

        // fft_jk.py:268 — for k1, ao1T in enumerate(ao1_kpts)
        for k1 in 0..nband {
            let ao1t = ao1_kpts.at(k1);
            let kpt1 = band[k1];
            let kpt2 = kpts[k2];
            let dk = [kpt2[0] - kpt1[0], kpt2[1] - kpt1[1], kpt2[2] - kpt1[2]];

            // fft_jk.py:271-277 — an Ewald exxdiv is applied at the END (see
            // upstream's comment: it bypasses the FFT discretization error), so
            // the kernel inside the loop is the plain one.
            let inner_exxdiv = match exxdiv {
                Some(ExxDiv::Ewald) | None => None,
                other => other,
            };
            let coulg = get_coulg(
                cell,
                CoulGArgs {
                    k: dk,
                    exxdiv: inner_exxdiv,
                    kpts: Some(kpts),
                    mesh: Some(mesh),
                    gv: Some(&gv),
                    wrap_around: true,
                    omega,
                },
            )?;

            // fft_jk.py:278-281
            let expmikr = if is_zero(&dk) {
                None
            } else {
                let mut re = vec![0.0_f64; ngrids];
                let mut im = vec![0.0_f64; ngrids];
                for (g, r) in coords.iter().enumerate() {
                    let ph = -(r[0] * dk[0] + r[1] * dk[1] + r[2] * dk[2]);
                    re[g] = ph.cos();
                    im[g] = ph.sin();
                }
                Some(CTensor::from_planes(re, im))
            };

            // vR_dm[i][p, g], allocated once per (k2, k1) as upstream does.
            let mut vr_dm: Vec<CTensor> =
                vec![CTensor::zeros(nao * ngrids); nset];

            // fft_jk.py:283-296 — the AO block loop.
            let mut p0 = 0usize;
            while p0 < nao {
                let p1 = (p0 + blksize).min(nao);
                let nblk = p1 - p0;

                // rho1[i, j, g] = conj(ao1T[p0+i, g]) * expmikr[g] * ao2T[j, g]
                let rho1 = build_rho1(ao1t, ao2t, expmikr.as_ref(), p0, p1, nao, ngrids);

                // fft_jk.py:286-291 — vG = fft(rho1) * coulG; vR = ifft(vG).
                let mut vg = fft(&rho1, mesh)?;
                for row in 0..nblk * nao {
                    let b = row * ngrids;
                    for g in 0..ngrids {
                        let (xr, xi) = (vg.re[b + g], vg.im[b + g]);
                        vg.re[b + g] = xr * coulg[g];
                        vg.im[b + g] = xi * coulg[g];
                    }
                }
                let mut vr = ifft(&vg, mesh)?;
                // fft_jk.py:292-293 — `if vR_dm.dtype == np.double: vR = vR.real`
                if real_out {
                    for t in vr.im.iter_mut() {
                        *t = 0.0;
                    }
                }

                // fft_jk.py:294-295 — einsum('ijg,jg->ig', vR, ao_dms[i])
                for i in 0..nset {
                    contract_vr_aodm(
                        &mut vr_dm[i], &vr, &ao_dms[i], p0, nblk, nao, ngrids,
                    );
                }
                p0 = p1;
            }

            // fft_jk.py:297 — vR_dm *= expmikr.conj()
            if let Some(ph) = expmikr.as_ref() {
                for v in vr_dm.iter_mut() {
                    for p in 0..nao {
                        let b = p * ngrids;
                        for g in 0..ngrids {
                            let (xr, xi) = (v.re[b + g], v.im[b + g]);
                            let (pr, pi) = (ph.re[g], -ph.im[g]);
                            v.re[b + g] = xr * pr - xi * pi;
                            v.im[b + g] = xr * pi + xi * pr;
                        }
                    }
                }
            }

            // fft_jk.py:299-300 — vk_kpts[i,k1] += weight * dot(vR_dm[i], ao1T.T)
            for i in 0..nset {
                accumulate_vk(
                    &mut vk_kpts[i][k1], &vr_dm[i], ao1t, weight, nao, ngrids,
                );
            }
        }
    }

    if real_out {
        for set in vk_kpts.iter_mut() {
            for m in set.iter_mut() {
                for v in m.im.iter_mut() {
                    *v = 0.0;
                }
            }
        }
    }

    // fft_jk.py:303-307
    if exxdiv == Some(ExxDiv::Ewald) && cell.dimension != 0 {
        ewald_exxdiv_for_g0(cell, kpts, dms, &mut vk_kpts, kpts_band)?;
    }
    Ok(vk_kpts)
}

/// `ao_dms[j, g] = sum_l dm[j, l] conj(ao2T[l, g])` — upstream's
/// `lib.dot(dms[i,k2], ao2T.conj())` (`fft_jk.py:264`).
fn dm_times_conj_ao(dm: &CTensor, ao2t: &CTensor, nao: usize, ngrids: usize) -> CTensor {
    let mut re = vec![0.0_f64; nao * ngrids];
    let mut im = vec![0.0_f64; nao * ngrids];
    for j in 0..nao {
        let ob = j * ngrids;
        for l in 0..nao {
            let (dr, di) = (dm.re[j * nao + l], dm.im[j * nao + l]);
            if dr == 0.0 && di == 0.0 {
                continue;
            }
            let ab = l * ngrids;
            for g in 0..ngrids {
                let (ar, ai) = (ao2t.re[ab + g], -ao2t.im[ab + g]);
                re[ob + g] += dr * ar - di * ai;
                im[ob + g] += dr * ai + di * ar;
            }
        }
    }
    CTensor::from_planes(re, im)
}

/// `rho1[(i, j), g] = conj(ao1T[p0+i, g]) * expmikr[g] * ao2T[j, g]` —
/// `fft_jk.py:284`, flattened to `(nblk*nao, ngrids)` for the batched FFT.
fn build_rho1(
    ao1t: &CTensor,
    ao2t: &CTensor,
    expmikr: Option<&CTensor>,
    p0: usize,
    p1: usize,
    nao: usize,
    ngrids: usize,
) -> CTensor {
    let nblk = p1 - p0;
    let mut re = vec![0.0_f64; nblk * nao * ngrids];
    let mut im = vec![0.0_f64; nblk * nao * ngrids];
    // `ao1T[p].conj() * expmikr` is hoisted out of the `j` loop exactly as
    // upstream's `ao1T[p0:p1].conj()*expmikr` is.
    let mut br = vec![0.0_f64; ngrids];
    let mut bi = vec![0.0_f64; ngrids];
    for (i, p) in (p0..p1).enumerate() {
        let ab = p * ngrids;
        match expmikr {
            None => {
                for g in 0..ngrids {
                    br[g] = ao1t.re[ab + g];
                    bi[g] = -ao1t.im[ab + g];
                }
            }
            Some(ph) => {
                for g in 0..ngrids {
                    let (ar, ai) = (ao1t.re[ab + g], -ao1t.im[ab + g]);
                    let (pr, pi) = (ph.re[g], ph.im[g]);
                    br[g] = ar * pr - ai * pi;
                    bi[g] = ar * pi + ai * pr;
                }
            }
        }
        for j in 0..nao {
            let ob = (i * nao + j) * ngrids;
            let jb = j * ngrids;
            for g in 0..ngrids {
                let (cr, ci) = (ao2t.re[jb + g], ao2t.im[jb + g]);
                re[ob + g] = br[g] * cr - bi[g] * ci;
                im[ob + g] = br[g] * ci + bi[g] * cr;
            }
        }
    }
    CTensor::from_planes(re, im)
}

/// `vR_dm[p0+i, g] = sum_j vR[(i, j), g] ao_dms[j, g]` — `fft_jk.py:295`.
fn contract_vr_aodm(
    vr_dm: &mut CTensor,
    vr: &CTensor,
    ao_dm: &CTensor,
    p0: usize,
    nblk: usize,
    nao: usize,
    ngrids: usize,
) {
    for i in 0..nblk {
        let ob = (p0 + i) * ngrids;
        for g in 0..ngrids {
            vr_dm.re[ob + g] = 0.0;
            vr_dm.im[ob + g] = 0.0;
        }
        for j in 0..nao {
            let vb = (i * nao + j) * ngrids;
            let ab = j * ngrids;
            for g in 0..ngrids {
                let (xr, xi) = (vr.re[vb + g], vr.im[vb + g]);
                let (yr, yi) = (ao_dm.re[ab + g], ao_dm.im[ab + g]);
                vr_dm.re[ob + g] += xr * yr - xi * yi;
                vr_dm.im[ob + g] += xr * yi + xi * yr;
            }
        }
    }
}

/// `vk[p, q] += weight * sum_g vR_dm[p, g] ao1T[q, g]` — `fft_jk.py:300`.
/// Note `ao1T` is NOT conjugated here: the conjugation already happened in
/// [`build_rho1`].
fn accumulate_vk(
    vk: &mut CTensor,
    vr_dm: &CTensor,
    ao1t: &CTensor,
    weight: f64,
    nao: usize,
    ngrids: usize,
) {
    for p in 0..nao {
        let pb = p * ngrids;
        for q in 0..nao {
            let qb = q * ngrids;
            let mut sr = 0.0_f64;
            let mut si = 0.0_f64;
            for g in 0..ngrids {
                let (xr, xi) = (vr_dm.re[pb + g], vr_dm.im[pb + g]);
                let (yr, yi) = (ao1t.re[qb + g], ao1t.im[qb + g]);
                sr += xr * yr - xi * yi;
                si += xr * yi + xi * yr;
            }
            vk.re[p * nao + q] += weight * sr;
            vk.im[p * nao + q] += weight * si;
        }
    }
}
