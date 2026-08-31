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
//!
//! # Parallelism (W-02b, `.planning/pbc/KRKS-OPTIMISATION-PLAN.md`)
//!
//! Every contraction here is parallelised over a **disjoint output partition** —
//! one rayon worker owns one output row (or one `nao`-wide row of `vk`) and no
//! two workers ever touch the same cell. The summation order *within* each
//! output cell is untouched, so the result is bit-identical for any
//! `RAYON_NUM_THREADS`, exactly as `10_grid_stride_occupancy.md` §6 argues for
//! disjoint grid-stride writes and as `pyscf-pbc-tools`'s `transform_axis`
//! already does for the transform batch. Gate B (D-PBC-17) therefore holds by
//! construction, not by luck: `crates/pyscf-pbc-df/tests/fft_jk_threads.rs`
//! asserts it on the real `get_j_kpts`/`get_k_kpts` output.
//!
//! Two rules for anything added here:
//!
//! * **Never parallelise a reduction axis.** `accumulate_rho`'s sum over `nu`
//!   and `contract_vr_aodm`'s sum over `j` stay serial and in ascending index
//!   order; only the axis that indexes *distinct outputs* is split.
//! * **Never fold the `+= 0.0` short-circuits away while parallelising.** The
//!   `if dr == 0.0 && di == 0.0 { continue; }` skips below are load-bearing for
//!   bit-parity in one case — `-0.0 + 0.0 == +0.0` — so removing them is a
//!   numerical change, not a cleanup.

use rayon::prelude::*;

use pyscf_algebra::{CTensor, oracle_dot};
use pyscf_pbc_gto::{CoulGArgs, ExxDiv, get_coulg, get_gv};
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
    //
    // W-02b: `mu` indexes DISJOINT output rows of `c0`, so it is the axis that
    // is split across workers; the reduction axis `nu` stays serial and
    // ascending, which is what makes this bit-identical to the pre-W-02b
    // `nu`-outer nest (the same terms reach each `c0[mu, g]` in the same order).
    let mut c0_re = vec![0.0_f64; nao * ngrids];
    let mut c0_im = vec![0.0_f64; nao * ngrids];
    c0_re
        .par_chunks_mut(ngrids)
        .zip(c0_im.par_chunks_mut(ngrids))
        .enumerate()
        .for_each(|(mu, (crow, cirow))| {
            for nu in 0..nao {
                let (dr, di) = (dm.re[nu * nao + mu], dm.im[nu * nao + mu]);
                if dr == 0.0 && di == 0.0 {
                    continue;
                }
                let ab = nu * ngrids;
                for g in 0..ngrids {
                    let (ar, ai) = (ao.re[ab + g], ao.im[ab + g]);
                    crow[g] += dr * ar - di * ai;
                    cirow[g] += dr * ai + di * ar;
                }
            }
        });
    // `rho[g]` is the output cell here and `mu` is the reduction axis, so the
    // split is over `g` (disjoint chunks) with `mu` serial and ascending inside
    // each chunk — again the pre-W-02b order, term for term.
    rho.re
        .par_chunks_mut(RHO_CHUNK)
        .zip(rho.im.par_chunks_mut(RHO_CHUNK))
        .enumerate()
        .for_each(|(c, (rre, rim))| {
            let g0 = c * RHO_CHUNK;
            for mu in 0..nao {
                let b = mu * ngrids;
                for t in 0..rre.len() {
                    let g = g0 + t;
                    let (br, bi) = (ao.re[b + g], -ao.im[b + g]);
                    let (cr, ci) = (c0_re[b + g], c0_im[b + g]);
                    rre[t] += br * cr - bi * ci;
                    rim[t] += br * ci + bi * cr;
                }
            }
        });
}

/// Grid points one worker owns in [`accumulate_rho`]'s second stage.
///
/// The split there is over the GRID, not over an AO index, so it needs an
/// explicit chunk: one grid point per worker would be pure dispatch overhead.
/// Large enough that a chunk is real work, small enough that a `mesh = 21` grid
/// (9261 points) still spreads over every core.
const RHO_CHUNK: usize = 512;

/// `v[p, q] = sum_g conj(ao[p, g]) vR[g] ao[q, g]` — upstream's
/// `aow = ao * vR; v += lib.dot(ao.conj().T, aow)` (`fft_jk.py:107-109`).
fn contract_ao_v_ao(ao: &CTensor, vr: &CTensor, nao: usize, ngrids: usize) -> CTensor {
    // aow[q, g] = ao[q, g] * vR[g]
    let mut aow_re = vec![0.0_f64; nao * ngrids];
    let mut aow_im = vec![0.0_f64; nao * ngrids];
    // W-02b: element-wise, so every output cell is its own partition.
    aow_re
        .par_chunks_mut(ngrids)
        .zip(aow_im.par_chunks_mut(ngrids))
        .enumerate()
        .for_each(|(q, (wre, wim))| {
            let b = q * ngrids;
            for g in 0..ngrids {
                let (ar, ai) = (ao.re[b + g], ao.im[b + g]);
                let (vrr, vri) = (vr.re[g], vr.im[g]);
                wre[g] = ar * vrr - ai * vri;
                wim[g] = ar * vri + ai * vrr;
            }
        });
    // D-PBC-17 (W-05): `Σ_g conj(ao[p,g]) aow[q,g]` is the zdotc contraction —
    // route it through the FOUND-06 pairwise-tree `oracle_dot`, not a naive
    // sequential `+=` over `ngrids` terms. `re = dot(ar,qr) + dot(ai,qi)`,
    // `im = dot(ar,qi) - dot(ai,qr)` is the zdotc identity from
    // `pyscf_algebra::oracle_zdot`, inlined here because `oracle_zdot` takes
    // owned `CTensor` planes and this contracts row *slices* of one.
    //
    // W-02b: one worker per output ROW `p`. `oracle_dot`'s pairwise tree shape
    // depends only on the input length, never on which thread runs it, so this
    // is bit-identical to the serial nest for any thread count.
    let mut re = vec![0.0_f64; nao * nao];
    let mut im = vec![0.0_f64; nao * nao];
    re.par_chunks_mut(nao)
        .zip(im.par_chunks_mut(nao))
        .enumerate()
        .for_each(|(p, (rrow, irow))| {
            let pb = p * ngrids;
            let ar = &ao.re[pb..pb + ngrids];
            let ai = &ao.im[pb..pb + ngrids];
            for q in 0..nao {
                let qb = q * ngrids;
                let qr = &aow_re[qb..qb + ngrids];
                let qi = &aow_im[qb..qb + ngrids];
                let rr = oracle_dot(ar, qr);
                let ii = oracle_dot(ai, qi);
                let ri = oracle_dot(ar, qi);
                let ir = oracle_dot(ai, qr);
                rrow[q] = rr + ii;
                irow[q] = ri - ir;
            }
        });
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
    let ngrids = df.grids.coords.len();

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
            // W-01: `coulG(dk)` and `expmikr(dk)` (fft_jk.py:262-281) depend
            // only on `dk`, `omega` and `inner_exxdiv` — never on the density
            // matrix or on which `(k1,k2)` pair produced this `dk` — so they
            // are hoisted into a cache on `Fftdf` that survives the whole SCF
            // instead of being rebuilt on every one of the `Nk^2` pairs.
            let (coulg, expmikr) = {
                let entry = df.coulg_and_expmikr(dk, omega, inner_exxdiv, kpts, &gv)?;
                (entry.0.clone(), entry.1.clone())
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
                // W-02b: element-wise over `(row, g)`; one worker per row.
                vg.re
                    .par_chunks_mut(ngrids)
                    .zip(vg.im.par_chunks_mut(ngrids))
                    .for_each(|(gre, gim)| {
                        for g in 0..ngrids {
                            gre[g] *= coulg[g];
                            gim[g] *= coulg[g];
                        }
                    });
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
                    // W-02b: element-wise over `(p, g)`; one worker per row `p`.
                    v.re.par_chunks_mut(ngrids)
                        .zip(v.im.par_chunks_mut(ngrids))
                        .for_each(|(vre, vim)| {
                            for g in 0..ngrids {
                                let (xr, xi) = (vre[g], vim[g]);
                                let (pr, pi) = (ph.re[g], -ph.im[g]);
                                vre[g] = xr * pr - xi * pi;
                                vim[g] = xr * pi + xi * pr;
                            }
                        });
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
    // W-02b: `j` indexes disjoint output rows; the reduction over `l` stays
    // serial and ascending inside each of them.
    let mut re = vec![0.0_f64; nao * ngrids];
    let mut im = vec![0.0_f64; nao * ngrids];
    re.par_chunks_mut(ngrids)
        .zip(im.par_chunks_mut(ngrids))
        .enumerate()
        .for_each(|(j, (orow, oirow))| {
            for l in 0..nao {
                let (dr, di) = (dm.re[j * nao + l], dm.im[j * nao + l]);
                if dr == 0.0 && di == 0.0 {
                    continue;
                }
                let ab = l * ngrids;
                for g in 0..ngrids {
                    let (ar, ai) = (ao2t.re[ab + g], -ao2t.im[ab + g]);
                    orow[g] += dr * ar - di * ai;
                    oirow[g] += dr * ai + di * ar;
                }
            }
        });
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
    // W-02b: one worker per `i` — a contiguous `nao * ngrids` slab of the output
    // that no other worker touches. This is pure element-wise arithmetic (no
    // reduction anywhere), so the split cannot change a single rounding. The
    // `br`/`bi` scratch becomes per-worker, which is also what makes the closure
    // `Send`: it used to be one buffer reused across `i`.
    let block = nao * ngrids;
    re.par_chunks_mut(block)
        .zip(im.par_chunks_mut(block))
        .enumerate()
        .for_each(|(i, (orow, oirow))| {
            let p = p0 + i;
            let ab = p * ngrids;
            // `ao1T[p].conj() * expmikr` is hoisted out of the `j` loop exactly
            // as upstream's `ao1T[p0:p1].conj()*expmikr` is.
            let mut br = vec![0.0_f64; ngrids];
            let mut bi = vec![0.0_f64; ngrids];
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
                let ob = j * ngrids;
                let jb = j * ngrids;
                for g in 0..ngrids {
                    let (cr, ci) = (ao2t.re[jb + g], ao2t.im[jb + g]);
                    orow[ob + g] = br[g] * cr - bi[g] * ci;
                    oirow[ob + g] = br[g] * ci + bi[g] * cr;
                }
            }
        });
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
    // W-02b: `i` indexes disjoint output rows `p0+i` of `vR_dm`; the reduction
    // over `j` stays serial and ascending inside each row.
    let lo = p0 * ngrids;
    let hi = (p0 + nblk) * ngrids;
    vr_dm.re[lo..hi]
        .par_chunks_mut(ngrids)
        .zip(vr_dm.im[lo..hi].par_chunks_mut(ngrids))
        .enumerate()
        .for_each(|(i, (orow, oirow))| {
            orow.fill(0.0);
            oirow.fill(0.0);
            for j in 0..nao {
                let vb = (i * nao + j) * ngrids;
                let ab = j * ngrids;
                for g in 0..ngrids {
                    let (xr, xi) = (vr.re[vb + g], vr.im[vb + g]);
                    let (yr, yi) = (ao_dm.re[ab + g], ao_dm.im[ab + g]);
                    orow[g] += xr * yr - xi * yi;
                    oirow[g] += xr * yi + xi * yr;
                }
            }
        });
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
    // D-PBC-17 (W-05): `Σ_g vR_dm[p,g] ao1T[q,g]` is a PLAIN (unconjugated)
    // complex dot — `re = dot(xr,yr) - dot(xi,yi)`, `im = dot(xr,yi) +
    // dot(xi,yr)` — routed through the same FOUND-06 pairwise-tree
    // `oracle_dot` as `contract_ao_v_ao`'s zdotc, so the error bound is
    // `O(log2(ngrids)*eps)` instead of `O(ngrids*eps)`. Note the sign pattern
    // differs from `oracle_zdot`'s zdotc identity because there is no
    // conjugation here (`accumulate_vk`'s own doc comment: `ao1T` is NOT
    // conjugated).
    //
    // W-02b: one worker per output ROW `p` of `vk`. Every `oracle_dot` is over
    // the whole grid and its pairwise tree depends only on `ngrids`, so nothing
    // about the result depends on the thread count.
    vk.re
        .par_chunks_mut(nao)
        .zip(vk.im.par_chunks_mut(nao))
        .enumerate()
        .for_each(|(p, (vrow, virow))| {
            let pb = p * ngrids;
            let xr = &vr_dm.re[pb..pb + ngrids];
            let xi = &vr_dm.im[pb..pb + ngrids];
            for q in 0..nao {
                let qb = q * ngrids;
                let yr = &ao1t.re[qb..qb + ngrids];
                let yi = &ao1t.im[qb..qb + ngrids];
                let rr = oracle_dot(xr, yr);
                let ii = oracle_dot(xi, yi);
                let ri = oracle_dot(xr, yi);
                let ir = oracle_dot(xi, yr);
                vrow[q] += weight * (rr - ii);
                virow[q] += weight * (ri + ir);
            }
        });
}
