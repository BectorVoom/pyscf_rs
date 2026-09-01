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

/// `dm[q,k] = SUM_o C[q,o] · occ[o] · conj(C[k,o])` — reconstructs the full
/// density [`ewald_exxdiv_for_g0`] needs from the MO factors
/// [`get_k_kpts_mo`] otherwise never assembles it into.
fn dm_from_mo(c: &CTensor, occ: &[f64], nao: usize, nocc: usize) -> CTensor {
    let mut dm = CTensor::zeros(nao * nao);
    for q in 0..nao {
        for k in 0..nao {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for o in 0..nocc {
                let (cqr, cqi) = (c.re[q * nocc + o], c.im[q * nocc + o]);
                let (ckr, cki) = (c.re[k * nocc + o], c.im[k * nocc + o]);
                // C[q,o] * conj(C[k,o])
                let (pr, pi) = (cqr * ckr + cqi * cki, cqi * ckr - cqr * cki);
                re += occ[o] * pr;
                im += occ[o] * pi;
            }
            dm.re[q * nao + k] = re;
            dm.im[q * nao + k] = im;
        }
    }
    dm
}

/// `get_k_kpts` — the MO-factorised route (`df_jk.py:281-685`,
/// `force_dm_kbuild = False`). Exploits `dm[kj] = C_occ[kj] · diag(occ[kj]) ·
/// C_occ[kj]†` to avoid ever assembling the full `nao × nao` density: the
/// `t[i,k] = SUM_q C_L[i,q] · dm[q,k]` / `vk += t · conj(C_L)` pair
/// [`get_k_kpts`] runs at `O(nao³)` per `(ki, kj, L)` collapses to a SINGLE
/// intermediate
///
/// ```text
/// U[i,o]   = SUM_q C_L[i,q] · C_occ[kj][q,o]              (O(nao²·nocc))
/// vk[i,l2] += w · SUM_o occ[o] · U[i,o] · conj(U[l2,o])    (O(nao²·nocc))
/// ```
///
/// at `O(nao²·nocc)` per `(ki, kj, L)` — upstream's claimed "drop an `nao`
/// factor" (module docs), realised whenever `nocc < nao`. Produces the SAME
/// numbers as the density-matrix branch for `dm := C_occ·diag(occ)·C_occ†`
/// (gated at 1e-13, `tests/gdf_mo_k.rs`, not against upstream — "two routes
/// to the same number inside one process is a stronger test than either
/// against a third implementation", 17-10-PLAN.md Task 4).
///
/// This port adds **no automatic dispatch** on a `force_dm_kbuild` flag or an
/// `mo_coeff`-tagged density array the way upstream's `df_jk.get_k_kpts`
/// sniffs `getattr(dm_kpts, 'mo_coeff', None)`: [`crate::traits::PeriodicDf`]
/// carries only `dms: &[KMats]`, and plumbing MO coefficients through that
/// trait (and every `pyscf-pbc-scf` caller) is out of scope here — see
/// `.planning/phases/17-ksymm-multigrid/17-10-SUMMARY.md`'s "Follow-up
/// completion" section for the measured wall-clock verdict on whether that
/// wiring would even be worth it on this port's own (small, `nao ≈ 2·nocc`)
/// test fixtures.
///
/// `mo_coeff[n][k]` is `nao × nocc_k`, row-major (`p * nocc_k + o`);
/// `mo_occ[n][k]` is that k-point's `nocc_k` occupation weights. Both are the
/// ALREADY-TRUNCATED occupied columns — upstream's `_format_mo` does the same
/// truncation before calling its own low-level kernel.
///
/// # Errors
/// As [`get_k_kpts`].
pub fn get_k_kpts_mo(
    df: &Gdf,
    mo_coeff: &[Vec<CTensor>],
    mo_occ: &[Vec<Vec<f64>>],
    kpts: &[[f64; 3]],
    exxdiv: Option<ExxDiv>,
) -> Result<Vec<KMats>, PbcDfError> {
    let cell = &df.cell;
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    let nset = mo_coeff.len();
    let n2 = nao * nao;
    let weight = 1.0 / nkpts as f64;

    let mut vk: Vec<KMats> = (0..nset)
        .map(|_| (0..nkpts).map(|_| CTensor::zeros(n2)).collect())
        .collect();

    let mut ur = vec![0.0_f64; nao * nao];
    let mut ui = vec![0.0_f64; nao * nao];
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for blk in df.sr_loop(ki, kj, false)? {
                let sgn = blk.sign as f64;
                for n in 0..nset {
                    let c = &mo_coeff[n][kj];
                    let occ = &mo_occ[n][kj];
                    let nocc = occ.len();
                    if nocc == 0 {
                        continue;
                    }
                    let out = &mut vk[n][ki];
                    for l in 0..blk.naux {
                        let base = l * n2;
                        // U[i,o] = SUM_q C_L[i,q] · C_occ[q,o]
                        ur[..nao * nocc].iter_mut().for_each(|v| *v = 0.0);
                        ui[..nao * nocc].iter_mut().for_each(|v| *v = 0.0);
                        for i in 0..nao {
                            for q in 0..nao {
                                let cc = base + i * nao + q;
                                let (cr, ci) = (blk.re[cc], blk.im[cc]);
                                if cr == 0.0 && ci == 0.0 {
                                    continue;
                                }
                                for o in 0..nocc {
                                    let (mr, mi) = (c.re[q * nocc + o], c.im[q * nocc + o]);
                                    ur[i * nocc + o] += cr * mr - ci * mi;
                                    ui[i * nocc + o] += cr * mi + ci * mr;
                                }
                            }
                        }
                        // vk[i,l2] += w · SUM_o occ[o] · U[i,o] · conj(U[l2,o])
                        let w = weight * sgn;
                        for i in 0..nao {
                            for l2 in 0..nao {
                                let (mut ar, mut ai) = (0.0_f64, 0.0_f64);
                                for o in 0..nocc {
                                    let (a_r, a_i) = (ur[i * nocc + o], ui[i * nocc + o]);
                                    let (b_r, b_i) = (ur[l2 * nocc + o], ui[l2 * nocc + o]);
                                    // U[i,o] · conj(U[l2,o])
                                    ar += occ[o] * (a_r * b_r + a_i * b_i);
                                    ai += occ[o] * (a_i * b_r - a_r * b_i);
                                }
                                out.re[i * nao + l2] += w * ar;
                                out.im[i * nao + l2] += w * ai;
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

    match exxdiv {
        None => {}
        Some(ExxDiv::Ewald) => {
            if cell.dimension != 0 {
                let dms: Vec<KMats> = (0..nset)
                    .map(|n| {
                        (0..nkpts)
                            .map(|k| {
                                let c = &mo_coeff[n][k];
                                let occ = &mo_occ[n][k];
                                dm_from_mo(c, occ, nao, occ.len())
                            })
                            .collect()
                    })
                    .collect();
                ewald_exxdiv_for_g0(cell, kpts, &dms, &mut vk, None)?;
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

/// `kpts ∪ kpts_band`, deduplicated — `df_jk.py:86-94`, `df.py:299-313`.
/// Just the k-point SET; callers decide how to rebuild a `Cderi` over it (an
/// ordinary GDF tensor for [`build_band_gdf`]/[`get_jk`]; MDF's
/// `Scheme::Mixed` tensor for `mdf_jk`'s own composition, which reuses this
/// but NOT `build_band_gdf` — see `mdf::Mdf::band_gdf`'s doc for why a
/// generic rebuild is wrong there).
pub(crate) fn kpts_union(kpts: &[[f64; 3]], kpts_band: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut all = kpts.to_vec();
    all.extend_from_slice(kpts_band);
    pyscf_pbc_lib::kpts_helper::unique(&all).kpts
}

/// Build a `Gdf` over `kpts ∪ kpts_band` — `df_jk.py:86-94`, `df.py:299-313`.
/// Mirrors upstream's rebuild-on-band-request: same cell/config, but over the
/// UNION k-point list, so [`Gdf::sr_loop`] can be indexed positionally into
/// it for both the sampling AND the band k-points. Plan 17-10 Task 4.
///
/// Returns the rebuilt `Gdf` plus the union list itself (needed by callers to
/// translate a k-point VALUE into its position in the rebuilt tensor, since
/// [`Gdf::sr_loop`]/[`crate::gdf_builder::j3c::Cderi`] are purely positional —
/// see `.planning/phases/17-ksymm-multigrid/17-10-SUMMARY.md`'s "Follow-up
/// completion" section for why no structural change to `Cderi` was needed).
pub(crate) fn build_band_gdf(df: &Gdf, kpts: &[[f64; 3]], kpts_band: &[[f64; 3]]) -> (Gdf, Vec<[f64; 3]>) {
    let union = kpts_union(kpts, kpts_band);

    let mut g = Gdf::new(df.cell.clone(), &union);
    g.auxbasis = df.auxbasis.clone();
    g.exp_to_discard = df.exp_to_discard;
    g.aosym = df.aosym;
    g.prefer_ccdf = df.prefer_ccdf;
    g.j2c_eig_always = df.j2c_eig_always;
    g.exclude_dd_block = df.exclude_dd_block;
    g.rs_rcut = df.rs_rcut;
    g.rs_mesh = df.rs_mesh;
    // `df.py:312` — `j_only = self._j_only or len(kpts_union) == 1`. K always
    // calls this with distinct bra/ket, so `j_only` would silently drop the
    // off-diagonal pairs it needs; only honour the caller's `j_only` when the
    // union collapses to one point (gamma-only).
    g.j_only = df.j_only && union.len() == 1;
    (g, union)
}

/// The positional index of `kpt` inside `haystack`, `KPT_DIFF_TOL`-tolerant —
/// the value-based lookup [`crate::gdf_builder::j3c::Cderi`]'s positional
/// storage needs once it is built over a union k-set (upstream's `member`).
pub(crate) fn kpt_index(kpt: &[f64; 3], haystack: &[[f64; 3]]) -> Result<usize, PbcDfError> {
    pyscf_pbc_lib::kpts_helper::member(kpt, haystack)
        .first()
        .copied()
        .ok_or_else(|| {
            PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "GDF band k-point rebuild: {kpt:?} missing from its own union \
                     k-set (internal bug — build_band_gdf's union must contain every \
                     input k-point)"
                )),
            ))
        })
}

/// `get_j_kpts` at `kpts_band` — `df_jk.py:83-171`'s pass-1/pass-2 split, with
/// pass 2 evaluated at `kpts_band` (bra) instead of `kpts` (which stays the
/// pass-1 ket/dm index set), over `band_df` — a `Gdf` ALREADY rebuilt on
/// `union = kpts ∪ kpts_band` by the caller (standalone GDF:
/// [`build_band_gdf`]; MDF: `mdf::Mdf::band_gdf`, which keeps `Scheme::Mixed`
/// — the two callers need DIFFERENT rebuild recipes, so this function takes
/// the already-built result rather than building it itself).
///
/// # Errors
/// As [`get_j_kpts`], plus [`kpt_index`] (should not trigger — see its doc).
pub fn get_j_kpts_band(
    band_df: &Gdf,
    union: &[[f64; 3]],
    dms: &[KMats],
    kpts: &[[f64; 3]],
    kpts_band: &[[f64; 3]],
) -> Result<Vec<KMats>, PbcDfError> {
    let nao = band_df.cell.mol.nao_nr;
    let nband = kpts_band.len();
    let nset = dms.len();
    let n2 = nao * nao;
    let weight = 1.0 / kpts.len() as f64;

    let naux = band_df.get_naoaux()?;

    // Pass 1 — `rho[L] = SUM_k SUM_{mu nu} cderi[L, mu nu] · dm[nu, mu]`, over
    // the SAMPLING k-points (unchanged from `get_j_kpts`, just index-mapped
    // through the union rebuild).
    let mut rho: Vec<CTensor> = (0..nset).map(|_| CTensor::zeros(naux)).collect();
    for (k, kpt) in kpts.iter().enumerate() {
        let uk = kpt_index(kpt, union)?;
        for blk in band_df.sr_loop(uk, uk, false)? {
            let sgn = blk.sign as f64;
            for (n, dm) in dms.iter().enumerate() {
                let d = &dm[k];
                for l in 0..blk.naux {
                    let (mut ar, mut ai) = (0.0_f64, 0.0_f64);
                    for mu in 0..nao {
                        for nu in 0..nao {
                            let c = l * n2 + mu * nao + nu;
                            let (cr, ci) = (blk.re[c], blk.im[c]);
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

    // Pass 2 — `vj[kband][mu nu] = SUM_L rho[L] · cderi[kband, mu nu]`.
    let mut vj: Vec<KMats> = (0..nset)
        .map(|_| (0..nband).map(|_| CTensor::zeros(n2)).collect())
        .collect();
    for (bi, kpt) in kpts_band.iter().enumerate() {
        let ub = kpt_index(kpt, union)?;
        for blk in band_df.sr_loop(ub, ub, false)? {
            for (n, out) in vj.iter_mut().enumerate() {
                let o = &mut out[bi];
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

    if crate::df_jk::all_gamma(kpts_band) {
        for m in vj.iter_mut().flatten() {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    Ok(vj)
}

/// `get_k_kpts` at `kpts_band` — `df_jk.py:281-685`'s density-matrix branch,
/// bra ranging over `kpts_band` and ket over `kpts` (the dm index set), over
/// `band_df` — a `Gdf` ALREADY rebuilt on `union = kpts ∪ kpts_band` by the
/// caller. See [`get_j_kpts_band`]'s doc for why the rebuild is the caller's
/// job, not this function's.
///
/// # Errors
/// As [`get_k_kpts`], plus [`kpt_index`] (should not trigger — see its doc).
pub fn get_k_kpts_band(
    band_df: &Gdf,
    union: &[[f64; 3]],
    dms: &[KMats],
    kpts: &[[f64; 3]],
    kpts_band: &[[f64; 3]],
    exxdiv: Option<ExxDiv>,
) -> Result<Vec<KMats>, PbcDfError> {
    let cell = &band_df.cell;
    let nao = cell.mol.nao_nr;
    let nband = kpts_band.len();
    let nset = dms.len();
    let n2 = nao * nao;
    let weight = 1.0 / kpts.len() as f64;

    let mut vk: Vec<KMats> = (0..nset)
        .map(|_| (0..nband).map(|_| CTensor::zeros(n2)).collect())
        .collect();

    let mut tr = vec![0.0_f64; n2];
    let mut ti = vec![0.0_f64; n2];
    for (bi, kpt_i) in kpts_band.iter().enumerate() {
        let ui = kpt_index(kpt_i, union)?;
        for (kj, kpt_j) in kpts.iter().enumerate() {
            let uj = kpt_index(kpt_j, union)?;
            for blk in band_df.sr_loop(ui, uj, false)? {
                let sgn = blk.sign as f64;
                for (n, dm) in dms.iter().enumerate() {
                    let d = &dm[kj];
                    let out = &mut vk[n][bi];
                    for l in 0..blk.naux {
                        let base = l * n2;
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

    if crate::df_jk::all_gamma(kpts_band) {
        for m in vk.iter_mut().flatten() {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    match exxdiv {
        None => {}
        Some(ExxDiv::Ewald) => {
            if cell.dimension != 0 {
                ewald_exxdiv_for_g0(cell, kpts, dms, &mut vk, Some(kpts_band))?;
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
/// Propagates [`get_j_kpts`] / [`get_k_kpts`] (or their band-k-point
/// counterparts, which rebuild `_cderi` over the union k-set — plan 17-10
/// Task 4, closing what was `NotYetImplemented { phase: 17 }`).
pub fn get_jk(
    df: &Gdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    opts: crate::traits::JkOpts<'_>,
) -> Result<crate::traits::JkResult, PbcDfError> {
    if opts.omega.is_some() {
        return Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 14,
                what: "GDF.get_jk(omega) — the range-separated kernel needs \
                       GDF.range_coulomb (df.py:515-553), plan 14-07",
            },
        ));
    }
    if opts.kpts_band.is_some() && !crate::df_jk::band_is_kpts(opts.kpts_band, kpts) {
        let kpts_band = opts.kpts_band.expect("checked Some above");
        let (band_df, union) = build_band_gdf(df, kpts, kpts_band);
        return Ok(crate::traits::JkResult {
            vj: if opts.with_j {
                Some(get_j_kpts_band(&band_df, &union, dms, kpts, kpts_band)?)
            } else {
                None
            },
            vk: if opts.with_k {
                Some(get_k_kpts_band(&band_df, &union, dms, kpts, kpts_band, opts.exxdiv)?)
            } else {
                None
            },
        });
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
