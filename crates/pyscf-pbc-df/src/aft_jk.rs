//! `aft_jk` — the AFTDF Coulomb and exchange builds (`pyscf/pbc/df/aft_jk.py`,
//! plan 13-05).
//!
//! # The one place AFTDF is not FFTDF — read this before touching `get_k_kpts`
//!
//! `aft_jk.py:285` applies the Ewald exchange correction only
//! `if exxdiv == 'ewald' and cell.low_dim_ft_type == 'inf_vacuum'`. For an
//! ordinary 3-D cell that is FALSE, and the correction instead arrives through
//! `weighted_coulG(q, exxdiv, mesh)` → `get_coulG(..., exx='ewald')`, which adds
//! `Nk·vol·madelung` at `G+q = 0` (`tools/pbc.py:480-484`). FFTDF in 2.12.1 does
//! the opposite: no `exx` in `coulG`, and `_ewald_exxdiv_for_G0` applied
//! analytically afterwards from the EXACT overlap (`df_jk.py:1480`).
//!
//! **The two agree exactly to the extent that `ft_aopair[G=0] == S`.** Measured
//! on diamond at mesh 31, that difference is ~96% of `max|vk_AFT − vk_FFT|`
//! (6.2e-10 of 6.487e-10; `exxdiv=None` drops it 25× to 2.653e-11) — see
//! `.planning/phases/13-ft-ao-aftdf/measurements/`. It is a documented
//! divergence (risk R-15), NOT a bug to "fix", and it is the same change PySCF
//! 2.14 later imposed on `fft_jk`.
//!
//! # Deviation: ordered pairs instead of `kk_adapted_iter`
//!
//! Upstream groups `(ki, kj)` by unique `q = kj − ki`, then exploits
//! time-reversal symmetry and a `swap_2e` second pass that fills `vk[kj]` from
//! the conjugate group. This port iterates every ORDERED pair once and performs
//! only upstream's "case 1", which fills `vk[ki]` directly. That is the same sum
//! — every ordered pair is visited exactly once either way — with no symmetry
//! bookkeeping to get wrong. It costs `nkpts²` transforms where upstream does
//! roughly half; the cross-builder test against FFTDF is what guards it, and the
//! saving is recorded as a carry-over rather than taken on trust.

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::ExxDiv;

use crate::aftdf::Aftdf;
use crate::df_jk::KMats;
use crate::error::PbcDfError;
use crate::traits::{JkOpts, JkResult};

/// `get_j_kpts(mydf, dm_kpts, hermi, kpts)` — `aft_jk.py:41-94`.
///
/// # Errors
/// Propagates `weighted_coulG` and `ft_loop`.
pub fn get_j_kpts(
    df: &Aftdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    omega: Option<f64>,
) -> Result<Vec<KMats>, PbcDfError> {
    let nao = df.cell.mol.nao_nr;
    let nkpts = kpts.len();
    let nset = dms.len();
    let mesh = df.mesh;
    let n2 = nao * nao;

    // `kpt_allow = 0`, `exx = False` — J never carries the divergence.
    let coulg = df.weighted_coulg([0.0; 3], None, mesh, omega)?;
    let weight = 1.0 / nkpts as f64;

    let mut vj: Vec<KMats> = (0..nset)
        .map(|_| {
            (0..nkpts)
                .map(|_| CTensor {
                    re: vec![0.0; n2],
                    im: vec![0.0; n2],
                })
                .collect()
        })
        .collect();

    df.ft_loop(mesh, [0.0; 3], kpts, |b| {
        let nblk = b.p1 - b.p0;
        for (n, dm) in dms.iter().enumerate() {
            // rho[g] = Σ_k Σ_ij conj(dm[k][i,j]) · Gpq[k][g,i,j], conjugated —
            // `_update_vj_`'s four real einsums, written out.
            let mut rho_re = vec![0.0f64; nblk];
            let mut rho_im = vec![0.0f64; nblk];
            for ((br, bi), d) in b.re.iter().zip(b.im.iter()).zip(dm.iter()) {
                for gi in 0..nblk {
                    let base = gi * n2;
                    let (mut ar, mut ai) = (0.0f64, 0.0f64);
                    for p in 0..n2 {
                        ar += d.re[p] * br[base + p] + d.im[p] * bi[base + p];
                        ai += d.re[p] * bi[base + p] - d.im[p] * br[base + p];
                    }
                    rho_re[gi] += ar;
                    rho_im[gi] -= ai;
                }
            }
            for gi in 0..nblk {
                let c = coulg[b.p0 + gi] * weight;
                rho_re[gi] *= c;
                rho_im[gi] *= c;
            }
            for ((br, bi), out) in b.re.iter().zip(b.im.iter()).zip(vj[n].iter_mut()) {
                for gi in 0..nblk {
                    let (vr, vi) = (rho_re[gi], rho_im[gi]);
                    let base = gi * n2;
                    for p in 0..n2 {
                        out.re[p] += vr * br[base + p] - vi * bi[base + p];
                        out.im[p] += vr * bi[base + p] + vi * br[base + p];
                    }
                }
            }
        }
        Ok(())
    })?;

    // Upstream drops the imaginary part for an all-gamma k-set with a real dm.
    if kpts.iter().all(|k| k[0].abs() + k[1].abs() + k[2].abs() < 1e-9) {
        for m in vj.iter_mut().flatten() {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    Ok(vj)
}

/// `get_k_kpts(mydf, dm_kpts, hermi, kpts, exxdiv)` — `aft_jk.py:135-293`,
/// "case 1" over every ordered `(ki, kj)` pair (see the module docs).
///
/// # Errors
/// Propagates `weighted_coulG` and `ft_loop`.
pub fn get_k_kpts(
    df: &Aftdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    exxdiv: Option<ExxDiv>,
    omega: Option<f64>,
) -> Result<Vec<KMats>, PbcDfError> {
    let nao = df.cell.mol.nao_nr;
    let nkpts = kpts.len();
    let nset = dms.len();
    let mesh = df.mesh;
    let n2 = nao * nao;
    let weight = 1.0 / nkpts as f64;

    let mut vk: Vec<KMats> = (0..nset)
        .map(|_| {
            (0..nkpts)
                .map(|_| CTensor {
                    re: vec![0.0; n2],
                    im: vec![0.0; n2],
                })
                .collect()
        })
        .collect();

    // The record table depends on the KET k-point only, never on `q`, so build
    // `nkpts` of them rather than one per `(ki, kj)` pair — an 8× saving on a
    // 2×2×2 mesh, and the difference between an SCF that finishes and one that
    // does not.
    let kernels = df.ft_kernels(kpts)?;
    let gv = df.gv(mesh)?;
    let blocks = df.g_blocks(mesh, 1)?;

    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let q = [
                kpts[kj][0] - kpts[ki][0],
                kpts[kj][1] - kpts[ki][1],
                kpts[kj][2] - kpts[ki][2],
            ];
            // THE divergence: `exxdiv` goes into `coulG`, not into a separate
            // `_ewald_exxdiv_for_G0` pass.
            let wcoulg = df.weighted_coulg(q, exxdiv, mesh, omega)?;
            for &(p0, p1) in &blocks {
                let o = kernels[kj].eval(&df.cell, &gv[p0..p1], q)?;
                let b = crate::aftdf::FtBlock {
                    re: vec![o.re],
                    im: vec![o.im],
                    p0,
                    p1,
                    nao,
                };
                let nblk = b.p1 - b.p0;
                let (br, bi) = (&b.re[0], &b.im[0]);
                for (n, dm) in dms.iter().enumerate() {
                    let d = &dm[kj];
                    let out = &mut vk[n][ki];
                    for gi in 0..nblk {
                        let w = wcoulg[b.p0 + gi] * weight;
                        let base = gi * n2;
                        // t[i,k] = Σ_q G[i,q]·dm[q,k]
                        let mut tr = vec![0.0f64; n2];
                        let mut ti = vec![0.0f64; n2];
                        for i in 0..nao {
                            for qq in 0..nao {
                                let (gr, gim) =
                                    (br[base + i * nao + qq], bi[base + i * nao + qq]);
                                if gr == 0.0 && gim == 0.0 {
                                    continue;
                                }
                                for kk in 0..nao {
                                    let (dr, di) =
                                        (d.re[qq * nao + kk], d.im[qq * nao + kk]);
                                    tr[i * nao + kk] += gr * dr - gim * di;
                                    ti[i * nao + kk] += gr * di + gim * dr;
                                }
                            }
                        }
                        // vk[i,l] += w · Σ_k t[i,k]·conj(G[l,k])
                        for i in 0..nao {
                            for kk in 0..nao {
                                let (ar, ai) = (tr[i * nao + kk] * w, ti[i * nao + kk] * w);
                                if ar == 0.0 && ai == 0.0 {
                                    continue;
                                }
                                for l in 0..nao {
                                    let (gr, gim) =
                                        (br[base + l * nao + kk], bi[base + l * nao + kk]);
                                    out.re[i * nao + l] += ar * gr + ai * gim;
                                    out.im[i * nao + l] += ai * gr - ar * gim;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if kpts.iter().all(|k| k[0].abs() + k[1].abs() + k[2].abs() < 1e-9) {
        for m in vk.iter_mut().flatten() {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    Ok(vk)
}

/// `get_j_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band)` — `aft_jk.py:96-131`.
///
/// Splits into the SAME two passes [`get_j_kpts`] fuses into one `ft_loop`
/// (pass 1 accumulates `rho[G]` against `dm` over `kpts`; pass 2 evaluates
/// `vj` at `kpts_band`), because fusing them assumed bra = ket = `kpts`.
/// Plan 17-10 Task 4 — closes what was `NotYetImplemented { phase: 14 }`.
///
/// # Errors
/// Propagates `weighted_coulG` and `ft_loop`.
pub fn get_j_kpts_band(
    df: &Aftdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    kpts_band: &[[f64; 3]],
    omega: Option<f64>,
) -> Result<Vec<KMats>, PbcDfError> {
    let nao = df.cell.mol.nao_nr;
    let nset = dms.len();
    let mesh = df.mesh;
    let n2 = nao * nao;

    let coulg = df.weighted_coulg([0.0; 3], None, mesh, omega)?;
    let ngrids = coulg.len();
    let weight = 1.0 / kpts.len() as f64;

    // Pass 1 — `rho[G]` over the SAMPLING k-points (unchanged from
    // `get_j_kpts`'s single fused loop, just written to a buffer instead of
    // being consumed in the same block).
    let mut rho_re = vec![0.0f64; nset * ngrids];
    let mut rho_im = vec![0.0f64; nset * ngrids];
    df.ft_loop(mesh, [0.0; 3], kpts, |b| {
        let nblk = b.p1 - b.p0;
        for (n, dm) in dms.iter().enumerate() {
            for ((br, bi), d) in b.re.iter().zip(b.im.iter()).zip(dm.iter()) {
                for gi in 0..nblk {
                    let base = gi * n2;
                    let (mut ar, mut ai) = (0.0f64, 0.0f64);
                    for p in 0..n2 {
                        ar += d.re[p] * br[base + p] + d.im[p] * bi[base + p];
                        ai += d.re[p] * bi[base + p] - d.im[p] * br[base + p];
                    }
                    rho_re[n * ngrids + b.p0 + gi] += ar;
                    rho_im[n * ngrids + b.p0 + gi] -= ai;
                }
            }
        }
        Ok(())
    })?;
    for n in 0..nset {
        for gi in 0..ngrids {
            let c = coulg[gi] * weight;
            rho_re[n * ngrids + gi] *= c;
            rho_im[n * ngrids + gi] *= c;
        }
    }

    // Pass 2 — `vj[kband] = SUM_G rho[G] · Gpq[kband]`, at `kpts_band`.
    let nband = kpts_band.len();
    let mut vj: Vec<KMats> = (0..nset)
        .map(|_| (0..nband).map(|_| CTensor::zeros(n2)).collect())
        .collect();
    df.ft_loop(mesh, [0.0; 3], kpts_band, |b| {
        let nblk = b.p1 - b.p0;
        for n in 0..nset {
            for (k, out) in vj[n].iter_mut().enumerate() {
                let (br, bi) = (&b.re[k], &b.im[k]);
                for gi in 0..nblk {
                    let (vr, vi) = (rho_re[n * ngrids + b.p0 + gi], rho_im[n * ngrids + b.p0 + gi]);
                    let base = gi * n2;
                    for p in 0..n2 {
                        out.re[p] += vr * br[base + p] - vi * bi[base + p];
                        out.im[p] += vr * bi[base + p] + vi * br[base + p];
                    }
                }
            }
        }
        Ok(())
    })?;

    if kpts_band.iter().all(|k| k[0].abs() + k[1].abs() + k[2].abs() < 1e-9) {
        for m in vj.iter_mut().flatten() {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    Ok(vj)
}

/// `get_k_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band, exxdiv)` —
/// `aft_jk.py:295-364`'s underlying physics, written against THIS port's own
/// simpler ordered-pair loop (see the module docs — this port already deviates
/// from upstream's `kk_adapted_iter`/`ExtendedMole`/MO-factorised route in
/// [`get_k_kpts`], so the band variant generalises THAT structure rather than
/// upstream's, with bra ranging over `kpts_band` and ket over `kpts` — the
/// dm index set). Plan 17-10 Task 4 — closes what was
/// `NotYetImplemented { phase: 14 }`.
///
/// # Errors
/// Propagates `weighted_coulG` and `ft_loop`.
pub fn get_k_kpts_band(
    df: &Aftdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    kpts_band: &[[f64; 3]],
    exxdiv: Option<ExxDiv>,
    omega: Option<f64>,
) -> Result<Vec<KMats>, PbcDfError> {
    let nao = df.cell.mol.nao_nr;
    let nband = kpts_band.len();
    let nset = dms.len();
    let mesh = df.mesh;
    let n2 = nao * nao;
    let weight = 1.0 / kpts.len() as f64;

    let mut vk: Vec<KMats> = (0..nset)
        .map(|_| (0..nband).map(|_| CTensor::zeros(n2)).collect())
        .collect();

    // The record table depends on the KET k-point only — build it over
    // `kpts` (the dm/ket set), same reasoning as `get_k_kpts`.
    let kernels = df.ft_kernels(kpts)?;
    let gv = df.gv(mesh)?;
    let blocks = df.g_blocks(mesh, 1)?;

    for (bi, kpt_i) in kpts_band.iter().enumerate() {
        for (kj, kpt_j) in kpts.iter().enumerate() {
            let q = [
                kpt_j[0] - kpt_i[0],
                kpt_j[1] - kpt_i[1],
                kpt_j[2] - kpt_i[2],
            ];
            let wcoulg = df.weighted_coulg(q, exxdiv, mesh, omega)?;
            for &(p0, p1) in &blocks {
                let o = kernels[kj].eval(&df.cell, &gv[p0..p1], q)?;
                let b = crate::aftdf::FtBlock {
                    re: vec![o.re],
                    im: vec![o.im],
                    p0,
                    p1,
                    nao,
                };
                let nblk = b.p1 - b.p0;
                let (br, bim) = (&b.re[0], &b.im[0]);
                for (n, dm) in dms.iter().enumerate() {
                    let d = &dm[kj];
                    let out = &mut vk[n][bi];
                    for gi in 0..nblk {
                        let w = wcoulg[b.p0 + gi] * weight;
                        let base = gi * n2;
                        let mut tr = vec![0.0f64; n2];
                        let mut ti = vec![0.0f64; n2];
                        for i in 0..nao {
                            for qq in 0..nao {
                                let (gr, gim) =
                                    (br[base + i * nao + qq], bim[base + i * nao + qq]);
                                if gr == 0.0 && gim == 0.0 {
                                    continue;
                                }
                                for kk in 0..nao {
                                    let (dr, di) =
                                        (d.re[qq * nao + kk], d.im[qq * nao + kk]);
                                    tr[i * nao + kk] += gr * dr - gim * di;
                                    ti[i * nao + kk] += gr * di + gim * dr;
                                }
                            }
                        }
                        for i in 0..nao {
                            for kk in 0..nao {
                                let (ar, ai) = (tr[i * nao + kk] * w, ti[i * nao + kk] * w);
                                if ar == 0.0 && ai == 0.0 {
                                    continue;
                                }
                                for l in 0..nao {
                                    let (gr, gim) =
                                        (br[base + l * nao + kk], bim[base + l * nao + kk]);
                                    out.re[i * nao + l] += ar * gr + ai * gim;
                                    out.im[i * nao + l] += ai * gr - ar * gim;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if kpts_band.iter().all(|k| k[0].abs() + k[1].abs() + k[2].abs() < 1e-9) {
        for m in vk.iter_mut().flatten() {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    Ok(vk)
}

/// `AFTDF.get_jk` — the [`crate::traits::PeriodicDf`] entry point.
///
/// # Errors
/// Propagates the J and K builds, or their band-k-point counterparts.
pub fn get_jk(
    df: &Aftdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    opts: JkOpts<'_>,
) -> Result<JkResult, PbcDfError> {
    if let Some(kpts_band) = opts.kpts_band {
        if !crate::df_jk::band_is_kpts(Some(kpts_band), kpts) {
            let vj = if opts.with_j {
                Some(get_j_kpts_band(df, dms, kpts, kpts_band, opts.omega)?)
            } else {
                None
            };
            let vk = if opts.with_k {
                Some(get_k_kpts_band(df, dms, kpts, kpts_band, opts.exxdiv, opts.omega)?)
            } else {
                None
            };
            return Ok(JkResult { vj, vk });
        }
    }
    let vj = if opts.with_j {
        Some(get_j_kpts(df, dms, kpts, opts.omega)?)
    } else {
        None
    };
    let vk = if opts.with_k {
        Some(get_k_kpts(df, dms, kpts, opts.exxdiv, opts.omega)?)
    } else {
        None
    };
    Ok(JkResult { vj, vk })
}
