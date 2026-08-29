//! `df_jk` — J and K from `cderi` instead of from a grid
//! (`pyscf/pbc/df/df_jk.py:83-171`, `:281-685`). Plan 14-04.
//!
//! # The point of the phase, in two contractions
//!
//! FFTDF and AFTDF build `vj`/`vk` by sweeping a plane-wave grid every SCF
//! iteration. GDF sweeps the fitted 3-index tensor instead — the same
//! contraction shape with the auxiliary index `L` in place of `G`, and `L` runs
//! over `naux = 108` where `G` runs over `47³ = 103 823`. Upstream's own
//! wall-clock on diamond `gth-szv` 2×2×2 is 6.4 s for a GDF-driven `KRHF`
//! against 30.0 s for FFTDF and 450.6 s for AFTDF.
//!
//! ```text
//! J:  rho[L]      = SUM_k SUM_{mu nu} cderi[k][L, mu nu] · dm[k][nu, mu]
//!     vj[k][mu nu] = (1/nkpts) SUM_L rho[L] · cderi[k][L, mu nu]
//!
//! K:  t[i, k]      = SUM_q cderi[ki,kj][L, i q] · dm[kj][q, k]
//!     vk[ki][i, l] = (1/nkpts) SUM_kj SUM_L SUM_k t[i, k] · conj(cderi[ki,kj][L, l k])
//! ```
//!
//! The K shape is deliberately the SAME as `aft_jk::get_k_kpts`'s, with `cderi`
//! replacing the weighted AO-pair Fourier transform — so the two can be read
//! side by side and a divergence in one shows up as a divergence from the other.
//!
//! # `exxdiv` is applied AFTER the contraction, not inside it
//!
//! This is where GDF and AFTDF genuinely differ. AFTDF folds the `G + k = 0`
//! correction into `coulG` (`aft.py`'s `exx` flag); GDF's integrals are analytic,
//! so upstream applies `_ewald_exxdiv_for_G0` once to the assembled `vk`
//! (`df_jk.py:676-679`) and says so in a comment. Phase 13 measured that this
//! term is ~96 % of the MATRIX difference between builders while barely moving
//! the ENERGY (risk R-15), so matrices and energies need different tolerances.
//!
//! # Only the density-matrix branch
//!
//! Upstream also has an MO-factorised `get_k_kpts` (`force_dm_kbuild = False`)
//! that exploits `dm = C_occ C_occ†` to drop an `nao` factor. It produces the
//! same numbers; it is a performance variant and is Phase 17. Selecting it
//! returns `NotYetImplemented`.

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{Cell, ExxDiv};

use crate::df_jk::{KMats, ewald_exxdiv_for_g0};
use crate::error::PbcDfError;
use crate::gdf::Gdf;
use crate::incore::Aosym;

/// `get_j_kpts(mydf, dm_kpts, hermi, kpts, kpts_band)` — `df_jk.py:83-171`.
///
/// # Errors
/// [`PbcDfError::Core`] when the builder has not run, or a k-pair is missing.
pub fn get_j_kpts(
    df: &Gdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
) -> Result<Vec<KMats>, PbcDfError> {
    let nao = df.cell.mol.nao_nr;
    let nkpts = kpts.len();
    let nset = dms.len();
    let n2 = nao * nao;
    let naux = df.get_naoaux()?;
    let weight = 1.0 / nkpts as f64;

    // Pass 1 — `rho[L] = SUM_k SUM_{mu nu} cderi[L, mu nu] · dm[nu, mu]`.
    //
    // NOTE the TRANSPOSE on the density: upstream contracts against
    // `dms.transpose(0,1,3,2)` (`df_jk.py:110-111`). It is invisible for a
    // Hermitian real density and wrong for anything else.
    let mut rho: Vec<CTensor> = (0..nset).map(|_| CTensor::zeros(naux)).collect();
    for (k, _kpt) in kpts.iter().enumerate() {
        for blk in df.sr_loop(k, k, false)? {
            let sgn = blk.sign as f64;
            for (n, dm) in dms.iter().enumerate() {
                let d = &dm[k];
                for l in 0..blk.naux {
                    let (mut ar, mut ai) = (0.0_f64, 0.0_f64);
                    for mu in 0..nao {
                        for nu in 0..nao {
                            let c = l * n2 + mu * nao + nu;
                            let (cr, ci) = (blk.re[c], blk.im[c]);
                            // dm[nu, mu] — the transpose.
                            let (dr, di) = (d.re[nu * nao + mu], d.im[nu * nao + mu]);
                            ar += cr * dr - ci * di;
                            ai += cr * di + ci * dr;
                        }
                    }
                    rho[n].re[l] += sgn * ar;
                    rho[n].im[l] += sgn * ai;
                }
            }
        }
    }
    for r in &mut rho {
        r.re.iter_mut().for_each(|v| *v *= weight);
        r.im.iter_mut().for_each(|v| *v *= weight);
    }

    // Pass 2 — `vj[k][mu nu] = SUM_L rho[L] · cderi[k][L, mu nu]`. No conjugate:
    // upstream's `dot(rhoR, LpqR) - dot(rhoI, LpqI)` is a plain complex product.
    let mut vj: Vec<KMats> = (0..nset)
        .map(|_| (0..nkpts).map(|_| CTensor::zeros(n2)).collect())
        .collect();
    for (k, _kpt) in kpts.iter().enumerate() {
        for blk in df.sr_loop(k, k, false)? {
            for (n, out) in vj.iter_mut().enumerate() {
                let o = &mut out[k];
                for l in 0..blk.naux {
                    let (rr, ri) = (rho[n].re[l], rho[n].im[l]);
                    if rr == 0.0 && ri == 0.0 {
                        continue;
                    }
                    let base = l * n2;
                    for p in 0..n2 {
                        let (cr, ci) = (blk.re[base + p], blk.im[base + p]);
                        o.re[p] += rr * cr - ri * ci;
                        o.im[p] += rr * ci + ri * cr;
                    }
                }
            }
        }
    }

    if crate::df_jk::all_gamma(kpts) {
        for m in vj.iter_mut().flatten() {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    Ok(vj)
}

/// `get_k_kpts(mydf, dm_kpts, hermi, kpts, kpts_band, exxdiv)` —
/// `df_jk.py:281-685`, the density-matrix branch.
///
/// # Errors
/// [`PbcDfError::Core`] when the builder has not run with `j_only = false`, and
/// for an `exxdiv` GDF does not support (upstream raises for anything but
/// `ewald` or `None`).
pub fn get_k_kpts(
    df: &Gdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    exxdiv: Option<ExxDiv>,
) -> Result<Vec<KMats>, PbcDfError> {
    let cell = &df.cell;
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    let nset = dms.len();
    let n2 = nao * nao;
    let weight = 1.0 / nkpts as f64;

    let mut vk: Vec<KMats> = (0..nset)
        .map(|_| (0..nkpts).map(|_| CTensor::zeros(n2)).collect())
        .collect();

    let mut tr = vec![0.0_f64; n2];
    let mut ti = vec![0.0_f64; n2];
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            // The block is loaded ONCE and reused for every density channel —
            // Phase 13 found the mirror-image defect in its own `get_k_kpts`
            // (one `FtKernel` per `(ki, kj)` where the table depended on the
            // ket alone), and the structure here makes the same mistake
            // impossible.
            for blk in df.sr_loop(ki, kj, false)? {
                let sgn = blk.sign as f64;
                for (n, dm) in dms.iter().enumerate() {
                    let d = &dm[kj];
                    let out = &mut vk[n][ki];
                    for l in 0..blk.naux {
                        let base = l * n2;
                        // t[i,k] = SUM_q C[L, i q] · dm[q, k]
                        tr.iter_mut().for_each(|v| *v = 0.0);
                        ti.iter_mut().for_each(|v| *v = 0.0);
                        for i in 0..nao {
                            for q in 0..nao {
                                let c = base + i * nao + q;
                                let (cr, ci) = (blk.re[c], blk.im[c]);
                                if cr == 0.0 && ci == 0.0 {
                                    continue;
                                }
                                for kk in 0..nao {
                                    let (dr, di) = (d.re[q * nao + kk], d.im[q * nao + kk]);
                                    tr[i * nao + kk] += cr * dr - ci * di;
                                    ti[i * nao + kk] += cr * di + ci * dr;
                                }
                            }
                        }
                        // vk[i,l] += w · SUM_k t[i,k] · conj(C[L, l k])
                        let w = weight * sgn;
                        for i in 0..nao {
                            for kk in 0..nao {
                                let (ar, ai) = (tr[i * nao + kk] * w, ti[i * nao + kk] * w);
                                if ar == 0.0 && ai == 0.0 {
                                    continue;
                                }
                                for l2 in 0..nao {
                                    let c = base + l2 * nao + kk;
                                    let (cr, ci) = (blk.re[c], blk.im[c]);
                                    out.re[i * nao + l2] += ar * cr + ai * ci;
                                    out.im[i * nao + l2] += ai * cr - ar * ci;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if crate::df_jk::all_gamma(kpts) {
        for m in vk.iter_mut().flatten() {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    // `df_jk.py:676-679` — GDF's integrals are analytic, so the finite-size
    // exchange correction is applied to the ASSEMBLED matrix rather than folded
    // into `coulG` the way AFTDF does it.
    match exxdiv {
        None => {}
        Some(ExxDiv::Ewald) => {
            if cell.dimension != 0 {
                ewald_exxdiv_for_g0(cell, kpts, dms, &mut vk, None)?;
            }
        }
        Some(other) => {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "GDF does not support exxdiv {other:?}; it must be Ewald or None \
                     (df_jk.py:288-292 raises the same way)"
                )),
            )));
        }
    }
    Ok(vk)
}

/// `GDF.get_jk` — the [`crate::traits::PeriodicDf`] entry point.
///
/// # Errors
/// Propagates [`get_j_kpts`] / [`get_k_kpts`], and refuses `kpts_band`
/// (upstream rebuilds `_cderi` to cover band k-points; that is Phase 17).
pub fn get_jk(
    df: &Gdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    opts: crate::traits::JkOpts<'_>,
) -> Result<crate::traits::JkResult, PbcDfError> {
    if opts.kpts_band.is_some() && !crate::df_jk::band_is_kpts(opts.kpts_band, kpts) {
        return Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 17,
                what: "GDF.get_jk with band k-points — upstream REBUILDS _cderi to \
                       cover them (df_jk.py:86-92)",
            },
        ));
    }
    if opts.omega.is_some() {
        return Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 14,
                what: "GDF.get_jk(omega) — the range-separated kernel needs \
                       GDF.range_coulomb (df.py:515-553), plan 14-07",
            },
        ));
    }
    Ok(crate::traits::JkResult {
        vj: if opts.with_j {
            Some(get_j_kpts(df, dms, kpts)?)
        } else {
            None
        },
        vk: if opts.with_k {
            Some(get_k_kpts(df, dms, kpts, opts.exxdiv)?)
        } else {
            None
        },
    })
}

/// The `aosym` a J-only build needs. J touches only the diagonal `(k, k)`
/// pairs, so `j_only = true` is enough; K needs every pair.
pub fn required_aosym() -> Aosym {
    Aosym::S2
}

/// The cell a `Gdf` was built on — used by the trait impl's error paths.
pub fn cell_of(df: &Gdf) -> &Cell {
    &df.cell
}
