//! Density-fitted ADC (`pbc/adc/dfadc.py`, 62 l).
//!
//! Ports the DF three-index machinery over 19-14's base: `get_ovvv_df` /
//! `get_vvvv_df` chunk builders plus the `transform_integrals_df` block
//! assembly (`einsum('Lpq,Lrs->pqrs')` over the sliced `Lpq_mo` tensors,
//! `/nkpts`). The manifolds (19-15/19-16) consume the resulting
//! [`crate::kadc_ao2mo::KadcEris`] unchanged — DF is a different ERI source,
//! not a different method.
//!
//! The DF route and the conventional route are DIFFERENT approximations
//! (measured on the Gate-D fixtures: DF EA vs incore EA differ at 1.5e-2
//! while DF IP vs incore IP agree to 1e-15 — the ovvv treatment differs,
//! the ovoo one barely does). Each is gated against its OWN upstream number;
//! no assertion here compares a DF number to a conventional one (19-17).

use pyscf_algebra::{CTensor, oracle_sum};

use crate::ea::EaIntermediates;
use crate::error::PbcAdcError;
use crate::kadc_ao2mo::KadcEris;
use crate::types::{AdcConfig, AdcRoots};

/// DF `ovvv` chunk (`dfadc.get_ovvv_df`).
///
/// `lov` is `(naux,nocc,nvir)` for one (ki,kj) pair, `lvv` is
/// `(naux,nvir,nvir)`; returns the UNDIVIDED `(chunk,nvir,nvir,nvir)`
/// contraction over occupied rows `[p:p+chnk_size]` (`chnk_size >= nocc`
/// takes all rows). Complex-linear host product (no conjugation — exactly as
/// written).
///
/// The `/nkpts` normalization lives at the CALLER (`get_ovvv_df(...).reshape(
/// ...)/nkpts` at every upstream call site) — this function does NOT divide,
/// and every caller must. Forgetting it doubles every EA root on a 2-k mesh
/// (caught by Gate D at 4.8e-2 during development).
pub fn df_ovvv_chunk(
    lov: &CTensor,
    lvv: &CTensor,
    naux: usize,
    nocc: usize,
    nvir: usize,
    p: usize,
    chnk_size: usize,
) -> Result<CTensor, PbcAdcError> {
    if lov.re.len() != naux * nocc * nvir || lvv.re.len() != naux * nvir * nvir {
        return Err(PbcAdcError::ShapeMismatch { expected: naux * nocc * nvir, got: lov.re.len() });
    }
    let rows = chnk_size.min(nocc.saturating_sub(p));
    // Lov_temp: rows [p:p+rows] of transpose(1,2,0) = (occ,vir,aux) → (rows·nvir, naux).
    // lov flat (naux,nocc,nvir): lov[x][i][a] at (x*nocc+i)*nvir+a.
    // transposed T[i][a][x] = lov[x][i][a], row (i,a) at ((i*nvir+a)*naux + x).
    let mut out = CTensor::zeros(rows * nvir * nvir * nvir);
    for ri in 0..rows {
        let i = p + ri;
        for a in 0..nvir {
            for b in 0..nvir {
                for c in 0..nvir {
                    // Σ_x T[(i,a)][x] · Lvv[x][b][c].
                    let mut acc = (0.0f64, 0.0f64);
                    for x in 0..naux {
                        let t_re = lov.re[(x * nocc + i) * nvir + a];
                        let t_im = lov.im[(x * nocc + i) * nvir + a];
                        let v_re = lvv.re[(x * nvir + b) * nvir + c];
                        let v_im = lvv.im[(x * nvir + b) * nvir + c];
                        acc.0 += t_re * v_re - t_im * v_im;
                        acc.1 += t_re * v_im + t_im * v_re;
                    }
                    let o = ((ri * nvir + a) * nvir + b) * nvir + c;
                    out.re[o] = acc.0;
                    out.im[o] = acc.1;
                }
            }
        }
    }
    Ok(out)
}

/// DF `vvvv` chunk (`dfadc.get_vvvv_df`), including the trailing
/// `transpose(0,2,1,3)`.
///
/// `vv1`/`vv2` are `(naux,nvir,nvir)`; chunks over `vv1`'s first virtual
/// index (`[p:p+chnk_size]`, `chnk_size >= nvir` takes all).
pub fn df_vvvv_chunk(
    vv1: &CTensor,
    vv2: &CTensor,
    naux: usize,
    nvir: usize,
    p: usize,
    chnk_size: usize,
) -> Result<CTensor, PbcAdcError> {
    if vv1.re.len() != naux * nvir * nvir || vv2.re.len() != naux * nvir * nvir {
        return Err(PbcAdcError::ShapeMismatch {
            expected: naux * nvir * nvir,
            got: vv1.re.len(),
        });
    }
    let rows = chnk_size.min(nvir.saturating_sub(p));
    // vv1_temp: rows [p:p+rows] of transpose(1,2,0) = (vir,vir,aux).
    // vv1[x][a][b] at (x*nvir+a)*nvir+b; T[a][b][x] = vv1[x][a][b].
    // out0[a][b][c] = Σ_x T[a][b][x]·vv2[x][c→(b? )] — port literally:
    // vvvv = vv1_temp·vv2 reshaped (-1,nvir,nvir,nvir), then transpose(0,2,1,3).
    let mut out0 = CTensor::zeros(rows * nvir * nvir * nvir);
    for ra in 0..rows {
        let a = p + ra;
        for b in 0..nvir {
            for c in 0..nvir {
                for d in 0..nvir {
                    let mut acc = (0.0f64, 0.0f64);
                    for x in 0..naux {
                        let t_re = vv1.re[(x * nvir + a) * nvir + b];
                        let t_im = vv1.im[(x * nvir + a) * nvir + b];
                        // vv2 reshaped (naux, nvir*nvir): vv2[x][c][d].
                        let v_re = vv2.re[(x * nvir + c) * nvir + d];
                        let v_im = vv2.im[(x * nvir + c) * nvir + d];
                        acc.0 += t_re * v_re - t_im * v_im;
                        acc.1 += t_re * v_im + t_im * v_re;
                    }
                    let o = ((ra * nvir + b) * nvir + c) * nvir + d;
                    out0.re[o] = acc.0;
                    out0.im[o] = acc.1;
                }
            }
        }
    }
    // transpose(0,2,1,3): out[a][c][b][d] = out0[a][b][c][d].
    let mut out = CTensor::zeros(rows * nvir * nvir * nvir);
    for ra in 0..rows {
        for b in 0..nvir {
            for c in 0..nvir {
                for d in 0..nvir {
                    let src = ((ra * nvir + b) * nvir + c) * nvir + d;
                    let dst = ((ra * nvir + c) * nvir + b) * nvir + d;
                    out.re[dst] = out0.re[src];
                    out.im[dst] = out0.im[src];
                }
            }
        }
    }
    Ok(out)
}

/// Assemble the five DF ERI blocks from `Lpq_mo` three-index tensors
/// (`transform_integrals_df`: `einsum('Lpq,Lrs->pqrs', …)/nkpts`).
///
/// `lpq[(ki·nk+kj)]` holds `(naux,nmo,nmo)`; sliced per call site into
/// `Loo` (`[:,:nocc,:nocc]`), `Lov` (`[:,:nocc,nocc:]`), `Lvo`
/// (`[:,nocc:,:nocc]`), `Lvv` (`[:,nocc:,nocc:]`). `ovvv` stays unset (None
/// upstream) — the chunk path serves EA.
pub fn build_df_blocks(
    lpq: &[CTensor],
    kconserv: &[usize],
    nkpts: usize,
    nocc: usize,
    nmo: usize,
    naux: usize,
) -> Result<KadcEris, PbcAdcError> {
    let nvir = nmo - nocc;
    if lpq.len() != nkpts * nkpts || kconserv.len() != nkpts * nkpts * nkpts {
        return Err(PbcAdcError::ShapeMismatch { expected: nkpts * nkpts, got: lpq.len() });
    }
    for b in lpq {
        if b.re.len() != naux * nmo * nmo {
            return Err(PbcAdcError::ShapeMismatch { expected: naux * nmo * nmo, got: b.re.len() });
        }
    }
    // Slices: Loo[ki,kj][x][i][j], Lov[ki,kj][x][i][a], Lvo, Lvv.
    let sl = |b: &CTensor, x: usize, p: usize, q: usize| -> (f64, f64) {
        let o = (x * nmo + p) * nmo + q;
        (b.re[o], b.im[o])
    };
    let nk3 = nkpts * nkpts * nkpts;
    let inv = 1.0 / nkpts as f64;
    let mut oooo = CTensor::zeros(nk3 * nocc * nocc * nocc * nocc);
    let mut oovv = CTensor::zeros(nk3 * nocc * nocc * nvir * nvir);
    let mut ovoo = CTensor::zeros(nk3 * nocc * nvir * nocc * nocc);
    let mut ovov = CTensor::zeros(nk3 * nocc * nvir * nocc * nvir);
    let mut ovvo = CTensor::zeros(nk3 * nocc * nvir * nvir * nocc);
    for kp in 0..nkpts {
        for kq in 0..nkpts {
            for kr in 0..nkpts {
                let ks = kconserv[(kp * nkpts + kq) * nkpts + kr];
                if ks >= nkpts {
                    return Err(PbcAdcError::ShapeMismatch { expected: nkpts, got: ks });
                }
                let left = &lpq[(kp * nkpts + kq)];
                let right = &lpq[(kr * nkpts + ks)];
                let base = ((kp * nkpts + kq) * nkpts + kr) as usize;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for k in 0..nocc {
                            for l in 0..nocc {
                                let mut acc = (0.0f64, 0.0f64);
                                for x in 0..naux {
                                    let (lr, li) = sl(left, x, i, j);
                                    let (rr, ri) = sl(right, x, k, l);
                                    acc.0 += lr * rr - li * ri;
                                    acc.1 += lr * ri + li * rr;
                                }
                                let o = (base * nocc * nocc * nocc + i * nocc * nocc + j * nocc + k) * nocc + l;
                                oooo.re[o] = acc.0 * inv;
                                oooo.im[o] = acc.1 * inv;
                            }
                        }
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let mut acc = (0.0f64, 0.0f64);
                                for x in 0..naux {
                                    let (lr, li) = sl(left, x, i, j);
                                    let (rr, ri) = sl(right, x, nocc + a, nocc + b);
                                    acc.0 += lr * rr - li * ri;
                                    acc.1 += lr * ri + li * rr;
                                }
                                let o = (base * nocc * nocc * nvir + i * nocc * nvir + j * nvir + a) * nvir + b;
                                oovv.re[o] = acc.0 * inv;
                                oovv.im[o] = acc.1 * inv;
                            }
                            for jj in 0..nocc {
                                for kk in 0..nocc {
                                    // ovoo[i,a,j,k] = Σ_x Lov[left][x,i,a]·Loo[right][x,j,k].
                                    let mut acc = (0.0f64, 0.0f64);
                                    for x in 0..naux {
                                        let (lr, li) = sl(left, x, i, nocc + a);
                                        let (rr, ri) = sl(right, x, jj, kk);
                                        acc.0 += lr * rr - li * ri;
                                        acc.1 += lr * ri + li * rr;
                                    }
                                    let o = (base * nocc * nvir * nocc + i * nvir * nocc + a * nocc + jj) * nocc + kk;
                                    ovoo.re[o] = acc.0 * inv;
                                    ovoo.im[o] = acc.1 * inv;
                                }
                                // ovov[i,a,j,b] = Σ_x Lov[left][x,i,a]·Lov[right][x,j,b].
                                for b in 0..nvir {
                                    let mut acc = (0.0f64, 0.0f64);
                                    for x in 0..naux {
                                        let (lr, li) = sl(left, x, i, nocc + a);
                                        let (rr, ri) = sl(right, x, jj, nocc + b);
                                        acc.0 += lr * rr - li * ri;
                                        acc.1 += lr * ri + li * rr;
                                    }
                                    let o = (base * nocc * nvir * nocc + i * nvir * nocc + a * nocc + jj) * nvir + b;
                                    ovov.re[o] = acc.0 * inv;
                                    ovov.im[o] = acc.1 * inv;
                                }
                            }
                            // ovvo[i,a,b,j] = Σ_x Lov[left][x,i,a]·Lvo[right][x,b,j].
                            for b in 0..nvir {
                                for jj in 0..nocc {
                                    let mut acc = (0.0f64, 0.0f64);
                                    for x in 0..naux {
                                        let (lr, li) = sl(left, x, i, nocc + a);
                                        let (rr, ri) = sl(right, x, nocc + b, jj);
                                        acc.0 += lr * rr - li * ri;
                                        acc.1 += lr * ri + li * rr;
                                    }
                                    let o = (base * nocc * nvir * nvir + i * nvir * nvir + a * nvir + b) * nocc + jj;
                                    ovvo.re[o] = acc.0 * inv;
                                    ovvo.im[o] = acc.1 * inv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ovvv unset in DF (None upstream) — NaN marker with the right shape so
    // shape contracts hold but any incore-EA read propagates NaN LOUDLY
    // instead of a silent zero. The DF-chunk path (sigma_ea_df) never reads
    // eris.ovvv. A test pins the NaN so the marker cannot silently vanish.
    let mut ovvv = CTensor::zeros(nk3 * nocc * nvir * nvir * nvir);
    ovvv.re.fill(f64::NAN);
    ovvv.im.fill(f64::NAN);
    KadcEris::from_blocks(nkpts, nocc, nvir, oooo, oovv, ovoo, ovov, ovvv, ovvo)
}

/// DF-chunk EA sigma at fixed `kshift` (the `eris.ovvv is None` branch).
///
/// Ports the DF branch literally — INCLUDING its two-fetch shape (direct
/// `2·icab` over `(Lov[ki,kc], Lvv[kshift,kb])` plus exchange `−ibac` over
/// `(Lov[ki,kb], Lvv[kshift,kc])`), which the incore branch lacks. `lov`/`lvv`
/// are the per-pair `(naux,nocc,nvir)` / `(naux,nvir,nvir)` tables in
/// `[ki][kj]` order; `naux` their shared auxiliary dimension. `s2` diagonal
/// and `s1 = M_ab·r1` are identical to the incore route.
#[allow(clippy::too_many_arguments)]
pub fn sigma_ea_df(
    r1: &[(f64, f64)],
    r2: &[(f64, f64)],
    m: &EaIntermediates,
    eris: &KadcEris,
    lov: &[CTensor],
    lvv: &[CTensor],
    naux: usize,
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    kshift: usize,
) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>), PbcAdcError> {
    let (nk, no, nv) = (eris.nkpts, eris.nocc, eris.nvir);
    let nd = nk * nk * no * nv * nv;
    if r1.len() != nv || r2.len() != nd || kshift >= nk {
        return Err(PbcAdcError::ShapeMismatch { expected: nv, got: r1.len() });
    }
    if lov.len() != nk * nk || lvv.len() != nk * nk {
        return Err(PbcAdcError::ShapeMismatch { expected: nk * nk, got: lov.len() });
    }
    let r2_at = |ki: usize, kb: usize, i: usize, b: usize, c: usize| -> (f64, f64) {
        r2[((ki * nk + kb) * no + i) * nv * nv + b * nv + c]
    };
    let mk = &m.m_ab[kshift];
    let mut s1 = vec![(0.0f64, 0.0f64); nv];
    for a in 0..nv {
        let mut acc = (0.0f64, 0.0f64);
        for b in 0..nv {
            let (mr, mi) = (mk.re[a * nv + b], mk.im[a * nv + b]);
            acc.0 += mr * r1[b].0 - mi * r1[b].1;
            acc.1 += mr * r1[b].1 + mi * r1[b].0;
        }
        s1[a] = acc;
    }
    let mut s2 = vec![(0.0f64, 0.0f64); nd];
    for kb in 0..nk {
        for kc in 0..nk {
            let ki = kconserv[(kb * nk + kshift) * nk + kc];
            if ki >= nk {
                return Err(PbcAdcError::ShapeMismatch { expected: nk, got: ki });
            }
            // Full-occ chunks (fixture scale; chunking over occ rows is the
            // scaling path for production cells, same equation). The /nkpts
            // is applied HERE (upstream divides the reshaped chunk at every
            // call site — forgetting it doubles EA roots on a 2-k mesh).
            let inv_nk = 1.0 / nk as f64;
            let ch_direct_raw =
                df_ovvv_chunk(&lov[(ki * nk + kc)], &lvv[(kshift * nk + kb)], naux, no, nv, 0, no)?;
            let ch_exch_raw =
                df_ovvv_chunk(&lov[(ki * nk + kb)], &lvv[(kshift * nk + kc)], naux, no, nv, 0, no)?;
            let mut ch_direct = CTensor::zeros(ch_direct_raw.re.len());
            let mut ch_exch = CTensor::zeros(ch_exch_raw.re.len());
            for (d, s) in ch_direct.re.iter_mut().zip(ch_direct_raw.re.iter()) {
                *d = s * inv_nk;
            }
            for (d, s) in ch_direct.im.iter_mut().zip(ch_direct_raw.im.iter()) {
                *d = s * inv_nk;
            }
            for (d, s) in ch_exch.re.iter_mut().zip(ch_exch_raw.re.iter()) {
                *d = s * inv_nk;
            }
            for (d, s) in ch_exch.im.iter_mut().zip(ch_exch_raw.im.iter()) {
                *d = s * inv_nk;
            }
            // ch axes [i,c,a,b] over all occ rows.
            let ch_at = |ch: &CTensor, i: usize, c: usize, a: usize, b: usize| -> (f64, f64) {
                let o = ((i * nv + c) * nv + a) * nv + b;
                (ch.re[o], ch.im[o])
            };
            for a in 0..nv {
                // s1[a] += 2·Σ conj(ch_direct[i,c,a,b])·r2 − Σ conj(ch_exch[i,b? ...])·r2.
                // Exchange term: 'ibac,ibc->a' over (Lov[ki,kb], Lvv[kshift,kc]):
                // ch_exch axes [i,b,a,c]: element (i,b,a,c).
                let mut acc = (0.0f64, 0.0f64);
                let mut accx = (0.0f64, 0.0f64);
                for i in 0..no {
                    for b in 0..nv {
                        for c in 0..nv {
                            let (vr, vi) = ch_at(&ch_direct, i, c, a, b);
                            let (rr, ri) = r2_at(ki, kb, i, b, c);
                            acc.0 += vr * rr + vi * ri;
                            acc.1 += vr * ri - vi * rr;
                            // 'ibac': ch_exch[i,b,a,c].
                            let (xr, xi) = ch_at(&ch_exch, i, b, a, c);
                            accx.0 += xr * rr + xi * ri;
                            accx.1 += xr * ri - xi * rr;
                        }
                    }
                }
                s1[a].0 += 2.0 * acc.0 - accx.0;
                s1[a].1 += 2.0 * acc.1 - accx.1;
            }
            for i in 0..no {
                for b in 0..nv {
                    for c in 0..nv {
                        // s2[ki,kb][i,b,c] += Σ_a ch_direct[i,c,a,b]·r1[a].
                        let mut acc = (0.0f64, 0.0f64);
                        for a in 0..nv {
                            let (vr, vi) = ch_at(&ch_direct, i, c, a, b);
                            acc.0 += vr * r1[a].0 - vi * r1[a].1;
                            acc.1 += vr * r1[a].1 + vi * r1[a].0;
                        }
                        let d = -e_occ_k[ki][i] + e_vir_k[kb][b] + e_vir_k[kc][c];
                        let (rr, ri) = r2_at(ki, kb, i, b, c);
                        let o = ((ki * nk + kb) * no + i) * nv * nv + b * nv + c;
                        s2[o].0 += acc.0 + d * rr;
                        s2[o].1 += acc.1 + d * ri;
                    }
                }
            }
        }
    }
    Ok((s1, s2))
}

/// DF-chunk EA roots at `kshift` (dense Davidson over [`sigma_ea_df`]).
///
/// Same solve contract as [`crate::ea::kernel_ea`]: materialize, assert the
/// real-spectrum boundary via [`crate::roots::nosym_roots`], return the
/// lowest `cfg.nroots` sorted with singles weights.
#[allow(clippy::too_many_arguments)]
pub fn kernel_ea_df(
    m: &EaIntermediates,
    eris: &KadcEris,
    lov: &[CTensor],
    lvv: &[CTensor],
    naux: usize,
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    kshift: usize,
    cfg: &AdcConfig,
) -> Result<AdcRoots, PbcAdcError> {
    let (nk, no, nv) = (eris.nkpts, eris.nocc, eris.nvir);
    let nd = nk * nk * no * nv * nv;
    let dim = nv + nd;
    if cfg.nroots > dim || dim == 0 {
        return Err(PbcAdcError::ShapeMismatch { expected: dim, got: cfg.nroots });
    }
    let mut hre = vec![0.0f64; dim * dim];
    let mut him = vec![0.0f64; dim * dim];
    for c in 0..dim {
        let mut r1 = vec![(0.0f64, 0.0f64); nv];
        let mut r2 = vec![(0.0f64, 0.0f64); nd];
        if c < nv {
            r1[c] = (1.0, 0.0);
        } else {
            r2[c - nv] = (1.0, 0.0);
        }
        let (s1, s2) = sigma_ea_df(&r1, &r2, m, eris, lov, lvv, naux, e_occ_k, e_vir_k, kconserv, kshift)?;
        for i in 0..nv {
            hre[i * dim + c] = s1[i].0;
            him[i * dim + c] = s1[i].1;
        }
        for i in 0..nd {
            hre[(nv + i) * dim + c] = s2[i].0;
            him[(nv + i) * dim + c] = s2[i].1;
        }
    }
    let (energies_all, columns) = crate::roots::nosym_roots(&hre, &him, dim, cfg.nroots)?;
    let mut roots = AdcRoots { energies: Vec::with_capacity(cfg.nroots), spec_factors: Vec::with_capacity(cfg.nroots), converged: true };
    for r in 0..cfg.nroots {
        roots.energies.push(energies_all[r]);
        let mut w_terms = Vec::with_capacity(2 * nv);
        for i in 0..nv {
            w_terms.push(columns[r][i].0 * columns[r][i].0);
            w_terms.push(columns[r][i].1 * columns[r][i].1);
        }
        roots.spec_factors.push(oracle_sum(&w_terms));
    }
    Ok(roots)
}
