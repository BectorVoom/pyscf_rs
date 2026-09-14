//! FFTDF gradient JK — `get_j_e1_kpts` + `get_k_e1_kpts` (plan 18-04).
//!
//! Statement-for-statement port of `pyscf/pbc/df/fft_jk.py:113-...`
//! (`get_j_e1_kpts`) and `:310-...` (`get_k_e1_kpts`). These are the
//! two-electron half of every k-point gradient, and the only density-fitting
//! route in PySCF 2.12.1 that has one: `get_jk_e1`/`get_j_e1`/`get_k_e1` exist
//! solely on `FFTDF` (`pyscf/pbc/df/fft.py:324-340`), and `GDF`/`MDF`/`RSDF`/
//! `AFTDF` are not subclasses — hence the named refusal on
//! [`crate::traits::PeriodicDf`], never a fallback to another route.
//!
//! # What is shared with the energy path, and what is not
//!
//! * Shared: the FFT/`coulG` pipeline (`int2e_ip1` is never called here
//!   either) and the W-01 `coulG`/`expmikr` cache on [`Fftdf`], which is keyed
//!   by the k-difference index (D-PBC-31 clause 9) — one cache serves both
//!   entry points, so `nkpts²` builds collapse to `nkpts`.
//! * Not shared: the k-pair symmetry. In `get_k_e1_kpts` the bra carries the
//!   **derivative** AO (`ao1T[1:,p0:p1]`, `fft_jk.py:391`) and the ket the
//!   value AO, so the `(j,i)` swap moves the derivative onto the ket and the
//!   energy-path conjugate identity does not close (D-PBC-30 clause 4b,
//!   `18-CONTEXT §3.2`). The gradient entry points take no k-pair flag, and
//!   [`Fftdf::get_jk_e1`] refuses one rather than ignoring it.
//!
//! # Layout and parallelism
//!
//! Same conventions as `fft_jk.rs`: every AO block is `(nao, ngrids)`
//! row-major (each deriv-1 component block reads that way — see
//! [`crate::ao_cache::Deriv1Table`]), every `nao x nao` matrix is row-major,
//! and every contraction is parallelised over a **disjoint output partition**
//! with the reduction axis serial and ascending, so results are bit-identical
//! for any `RAYON_NUM_THREADS`. Reductions route through `oracle_sum` (over a
//! materialised partial buffer) or `oracle_dot` (the pairwise-tree dots),
//! never through a naive sequential `+=` over grid terms — the D-PBC-17
//! discipline.
//!
//! No CubeCL kernels are added here (ALG-06): derivative AOs come from the
//! existing `eval_ao_kpts` evaluators in `pyscf-kernels`, and the rest is
//! host-side arithmetic plus the FFT. The CubeCL manual (`INDEX.md` +
//! `Cubecl_generics.md`, `Float` generics) was consulted before writing; there
//! is no device kernel in this file for it to apply to.

use rayon::prelude::*;

use pyscf_algebra::{CTensor, oracle_dot, oracle_sum};
use pyscf_pbc_gto::{CoulGArgs, ExxDiv, get_coulg, get_gv};
use pyscf_pbc_tools::{fft, ifft};

use crate::ao_cache::{self, AoEvalCount};
use crate::df_jk::{KMats, all_gamma, format_kpts_band};
use crate::error::PbcDfError;
use crate::fft_jk::dm_times_conj_ao;
use crate::fftdf::{AoKpts, Fftdf};
use crate::traits::JkOpts;
use crate::zlinalg::zscale_real;

/// `[x][iset][kband]`, `x = 0..3` the Cartesian component, each entry an
/// `nao x nao` row-major matrix. Upstream's `(3, nset, nband, nao, nao)`
/// with the single-set squeeze left to the caller (this port's `dms` already
/// carry the set axis explicitly, as on the energy path).
pub type GradMats = Vec<Vec<KMats>>;

/// The `(vj, vk)` pair a gradient J/K build returns. Either half is `None`
/// when the caller asked for the other only.
#[derive(Debug, Clone, Default)]
pub struct GradJkResult {
    /// `vj[x][iset][kband]`.
    pub vj: Option<GradMats>,
    /// `vk[x][iset][kband]`.
    pub vk: Option<GradMats>,
}

/// Upstream's `mo_coeff`/`mo_occ` tag on the density
/// (`fft_jk.py:322, :357-359`), as owned data: one block per sampling
/// k-point holding the **already-truncated, `sqrt(occ)`-scaled** occupied
/// columns, `nao × nocc` row-major — i.e. `mo_coeff[k][:,occ>0] *
/// sqrt(occ[occ>0])`. Use [`TaggedMo::from_coeff_occ`] to build it from raw
/// coefficients and occupations.
///
/// When present (and `nset == 1`, exactly like upstream), `get_k_e1_kpts`
/// collapses the ket index from `nao` to `nocc`: the inner `rho1` falls from
/// `3·nao·nao·ngrids` to `3·nao·nocc·ngrids` (6.5× at diamond `gth-dzvp`,
/// `18-REVIEW §3.1`) against a resident `nkpts·nocc·ngrids` ket table (§8.2).
#[derive(Debug, Clone, Default)]
pub struct TaggedMo {
    /// One block per sampling k-point.
    pub blocks: Vec<TaggedMoBlock>,
}

/// One k-point's scaled occupied block: `nao × nocc` row-major.
#[derive(Debug, Clone)]
pub struct TaggedMoBlock {
    /// `C[:,occ>0] * sqrt(occ[occ>0])`.
    pub scaled: CTensor,
    /// Occupied count at this k-point.
    pub nocc: usize,
}

impl TaggedMo {
    /// `fft_jk.py:357-358`: keep the `occ > 0` columns of each `coeff[k]`
    /// (`nao × nfull` row-major) and fold in `sqrt(occ)`.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] when the coefficient/occupation lists do not line
    /// up with each other or with `nao`.
    pub fn from_coeff_occ(
        coeff: &[CTensor],
        occ: &[Vec<f64>],
        nao: usize,
    ) -> Result<Self, PbcDfError> {
        let bad = |what: String| {
            PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!("TaggedMo: {what}")),
            ))
        };
        if coeff.len() != occ.len() {
            return Err(bad(format!(
                "coeff has {} k-points but occ has {}",
                coeff.len(),
                occ.len()
            )));
        }
        let mut blocks = Vec::with_capacity(coeff.len());
        for (k, (c, o)) in coeff.iter().zip(occ.iter()).enumerate() {
            if c.len() % nao != 0 {
                return Err(bad(format!(
                    "coeff[{k}] has {} entries, not a multiple of nao = {nao}",
                    c.len()
                )));
            }
            let nfull = c.len() / nao;
            if o.len() != nfull {
                return Err(bad(format!(
                    "occ[{k}] has {} entries but coeff[{k}] has {nfull} columns",
                    o.len()
                )));
            }
            let keep: Vec<usize> = (0..nfull).filter(|&j| o[j] > 0.0).collect();
            let nocc = keep.len();
            let mut scaled_re = vec![0.0_f64; nao * nocc];
            let mut scaled_im = vec![0.0_f64; nao * nocc];
            for (o2, &j) in keep.iter().enumerate() {
                let s = o[j].sqrt();
                for p in 0..nao {
                    scaled_re[p * nocc + o2] = c.re[p * nfull + j] * s;
                    scaled_im[p * nocc + o2] = c.im[p * nfull + j] * s;
                }
            }
            blocks.push(TaggedMoBlock {
                scaled: CTensor::from_planes(scaled_re, scaled_im),
                nocc,
            });
        }
        Ok(Self { blocks })
    }
}

/// What a gradient K/J run actually did: the budgeted residency `m`, the
/// re-derived `blksize` (`blksize = nao` on the J path, which contracts the
/// full AO block and has no `p0:p1` loop), and the realised deriv-1 build
/// counts from [`crate::ao_cache::AoEvalCount`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KGradStats {
    /// Resident deriv-1 k-point tables.
    pub m: usize,
    /// Inner AO block size.
    pub blksize: usize,
    /// `eval_ao_kpts` calls issued for deriv-1 chunks.
    pub chunk_builds: usize,
    /// Total k-point tables built.
    pub k_tables: usize,
}

/// `(re, im)` of deriv component `c` (`0` = value, `1..3` = x/y/z), AO `p`,
/// grid `g` in a deriv-1 `kao` buffer (`(c·nao + p)·ngrids + g` — see
/// [`crate::ao_cache::Deriv1Table`]).
fn d1_at(kao: &CTensor, c: usize, p: usize, g: usize, ngrids: usize, nao: usize) -> (f64, f64) {
    let b = (c * nao + p) * ngrids + g;
    (kao.re[b], kao.im[b])
}

/// Grid points one worker owns in [`build_rho_j`].
const RHO_CHUNK: usize = 512;

/// `rho[g] = Σ_k Σ_{mu,nu} conj(ao_k[mu,g]) dm[nu,mu] ao_k[nu,g]` — the
/// `make_rho`/`aoR_loop` accumulation of `get_j_e1_kpts` (`fft_jk.py:124-146`)
/// over cached full-grid value tables, for ONE density set (the caller loops
/// sets, exactly like upstream's `for i in range(nset)`).
///
/// The per-grid-point reduction over `mu` runs through [`oracle_sum`] over a
/// materialised partial buffer (D-PBC-17): `c0[mu,g]` is built exactly like
/// the energy path's first stage (mu-parallel, `nu` serial ascending with the
/// `+= 0.0` short-circuit kept), then each grid chunk fills
/// `partials[t·nao+mu]` in fixed `(mu, t)` order and reduces each row with
/// `oracle_sum`. Grid chunks are disjoint across workers and the `k` loop is
/// serial, so the result is bit-identical for any thread count.
fn build_rho_j(ao: &AoKpts, dm: &KMats, nao: usize, ngrids: usize) -> CTensor {
    let nkpts = ao.aot.len();
    let mut rho = CTensor::zeros(ngrids);
    let mut c0re = vec![0.0_f64; nao * ngrids];
    let mut c0im = vec![0.0_f64; nao * ngrids];
    for k in 0..nkpts {
        let a = &ao.aot[k];
        let dmk = &dm[k];
        c0re
            .par_chunks_mut(ngrids)
            .zip(c0im.par_chunks_mut(ngrids))
            .enumerate()
            .for_each(|(mu, (crow, cirow))| {
                crow.fill(0.0);
                cirow.fill(0.0);
                for nu in 0..nao {
                    let (dr, di) = (dmk.re[nu * nao + mu], dmk.im[nu * nao + mu]);
                    if dr == 0.0 && di == 0.0 {
                        continue;
                    }
                    let ab = nu * ngrids;
                    for g in 0..ngrids {
                        let (ar, ai) = (a.re[ab + g], a.im[ab + g]);
                        crow[g] += dr * ar - di * ai;
                        cirow[g] += dr * ai + di * ar;
                    }
                }
            });
        rho.re
            .par_chunks_mut(RHO_CHUNK)
            .zip(rho.im.par_chunks_mut(RHO_CHUNK))
            .enumerate()
            .for_each(|(c, (rre, rim))| {
                let g0 = c * RHO_CHUNK;
                let glen = rre.len();
                let mut pr = vec![0.0_f64; glen * nao];
                let mut pi = vec![0.0_f64; glen * nao];
                for mu in 0..nao {
                    let cb = mu * ngrids;
                    let ab = mu * ngrids;
                    for t in 0..glen {
                        let g = g0 + t;
                        let (br, bi) = (a.re[ab + g], -a.im[ab + g]);
                        let (cr, ci) = (c0re[cb + g], c0im[cb + g]);
                        pr[t * nao + mu] = br * cr - bi * ci;
                        pi[t * nao + mu] = br * ci + bi * cr;
                    }
                }
                for t in 0..glen {
                    rre[t] += oracle_sum(&pr[t * nao..(t + 1) * nao]);
                    rim[t] += oracle_sum(&pi[t * nao..(t + 1) * nao]);
                }
            });
    }
    rho
}

/// `get_j_e1_kpts(mydf, dm_kpts, kpts, kpts_band)` — `fft_jk.py:113-...`.
///
/// Returns `vj[x][iset][kband]`, `nao x nao` row-major. The density Coulomb
/// potential `vR` is built exactly like the energy path (density on the grid
/// → FFT → `×coulG` → iFFT → quadrature weight); the derivative enters only
/// in the final contraction, where the bra is the deriv-1 AO and the ket the
/// value AO, with upstream's minus sign (`vj[...] -= ...`).
///
/// Upstream has no `omega` here (`tools.get_coulG(cell, mesh=mesh)` is the
/// Gamma kernel) and no `hermi` (the evaluator is always the Hermitian one,
/// truncated to real when the sampling is gamma-only) — so neither appears in
/// this signature, and the doc states that rather than leaving it implicit.
/// `stats` reports the budgeted residency when given.
///
/// # Errors
/// Propagates the AO evaluations, `get_coulG` and the FFT.
pub fn get_j_e1_kpts(
    df: &Fftdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    kpts_band: Option<&[[f64; 3]]>,
    stats: Option<&mut KGradStats>,
) -> Result<GradMats, PbcDfError> {
    let cell = &df.cell;
    let mesh = df.mesh;
    let nset = dms.len();
    let nao = cell.mol.nao_nr;
    let ngrids = df.ngrids();
    let band = format_kpts_band(kpts_band, kpts);
    let nband = band.len();

    // fft_jk.py:131 — the Gamma Coulomb kernel, no range separation.
    let gv = get_gv(cell, Some(mesh))?;
    let coulg = get_coulg(
        cell,
        CoulGArgs {
            mesh: Some(mesh),
            gv: Some(&gv),
            ..CoulGArgs::new()
        },
    )?;

    // fft_jk.py:133-157 — one vR per set. `rhoR *= 1/nkpts`, then
    // FFT → ×coulG → iFFT → ×weight, real iff the sampling is gamma-only.
    let real_rho = all_gamma(kpts);
    let weight = df.weight();
    let mut vr: Vec<CTensor> = Vec::with_capacity(nset);
    let ao = df.ao_kpts(kpts)?;
    for dmset in dms.iter().take(nset) {
        let mut rho = build_rho_j(&ao, dmset, nao, ngrids);
        zscale_real(&mut rho, 1.0 / kpts.len() as f64);
        if real_rho {
            for v in rho.im.iter_mut() {
                *v = 0.0;
            }
        }
        let mut rhog = fft(&rho, mesh)?;
        for g in 0..ngrids {
            rhog.re[g] *= coulg[g];
            rhog.im[g] *= coulg[g];
        }
        let mut v = ifft(&rhog, mesh)?;
        zscale_real(&mut v, weight);
        if real_rho {
            for t in v.im.iter_mut() {
                *t = 0.0;
            }
        }
        vr.push(v);
    }

    // The band value tables (deriv = 0, cached) and the band deriv-1 tables
    // tiled in chunks of m (the budgeted residency — `ao_cache`).
    let m = ao_cache::resident_k_count(df.max_memory, ngrids, nao, nband);
    let counter = AoEvalCount::default();
    let ao_val = df.ao_kpts(band)?;

    // fft_jk.py:165-177 — `aow = ao[0]*vR`; `vj -= einsum('axi,xj->aij',
    // ao[1:].conj(), aow)`. The zdotc identity is the energy path's
    // `contract_ao_v_ao` one, routed through the same pairwise-tree
    // `oracle_dot`; the only differences are the deriv bra and the minus.
    let mut out: GradMats = (0..3)
        .map(|_| {
            (0..nset)
                .map(|_| (0..nband).map(|_| CTensor::zeros(nao * nao)).collect())
                .collect()
        })
        .collect();
    for chunk in (0..nband).collect::<Vec<_>>().chunks(m) {
        let tabs = ao_cache::eval_deriv1_chunk(&df.cell, &df.grids.coords, band, chunk, Some(&counter))?;
        for tab in tabs.iter() {
            let k = tab.kidx;
            let avt = ao_val.at(k);
            for i in 0..nset {
                let mut aow_re = vec![0.0_f64; nao * ngrids];
                let mut aow_im = vec![0.0_f64; nao * ngrids];
                aow_re
                    .par_chunks_mut(ngrids)
                    .zip(aow_im.par_chunks_mut(ngrids))
                    .enumerate()
                    .for_each(|(q, (wre, wim))| {
                        let b = q * ngrids;
                        for g in 0..ngrids {
                            let (ar, ai) = (avt.re[b + g], avt.im[b + g]);
                            let (vrr, vri) = (vr[i].re[g], vr[i].im[g]);
                            wre[g] = ar * vrr - ai * vri;
                            wim[g] = ar * vri + ai * vrr;
                        }
                    });
                for x in 0..3 {
                    let c = x + 1;
                    // One worker per deriv-bra row `p` (disjoint output
                    // rows); `q` stays serial and ascending inside each row,
                    // and the `conj(dao)` planes are materialised once per
                    // row in fixed `g` order — deterministic by construction.
                    let mx = &mut out[x][i][k];
                    mx.re
                        .par_chunks_mut(nao)
                        .zip(mx.im.par_chunks_mut(nao))
                        .enumerate()
                        .for_each(|(p, (rrow, irow))| {
                            let mut dr = vec![0.0_f64; ngrids];
                            let mut di = vec![0.0_f64; ngrids];
                            for g in 0..ngrids {
                                let (r, im) = d1_at(&tab.kao, c, p, g, ngrids, nao);
                                dr[g] = r;
                                di[g] = -im;
                            }
                            for q in 0..nao {
                                let qb = q * ngrids;
                                let qr = &aow_re[qb..qb + ngrids];
                                let qi = &aow_im[qb..qb + ngrids];
                                let rr = oracle_dot(&dr, qr);
                                let ii = oracle_dot(&di, qi);
                                let ri = oracle_dot(&dr, qi);
                                let ir = oracle_dot(&di, qr);
                                rrow[q] -= rr + ii;
                                irow[q] -= ri - ir;
                            }
                        });
                }
            }
        }
    }
    if all_gamma(band) {
        for set in out.iter_mut() {
            for mats in set.iter_mut() {
                for t in mats.iter_mut() {
                    for v in t.im.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
        }
    }

    if let Some(s) = stats {
        *s = KGradStats {
            m,
            blksize: nao,
            chunk_builds: counter.builds(),
            k_tables: counter.kpoints(),
        };
    }
    Ok(out)
}

/// `ao2 = Cᵀ·ao` with the `sqrt(occ)`-scaled coefficients — `fft_jk.py:359`
/// (`np.dot(mo_coeff[k].T, ao)`), i.e. `ket[o,g] = Σ_p scaled[p,o]·ao[p,g]`.
///
/// One worker per occupied row `o` (disjoint); `p` serial ascending — the same
/// row-partition discipline as `dm_times_conj_ao`, hence deterministic across
/// thread counts.
fn mo_ket_table(scaled: &CTensor, nocc: usize, ao: &CTensor, nao: usize, ngrids: usize) -> CTensor {
    let mut re = vec![0.0_f64; nocc * ngrids];
    let mut im = vec![0.0_f64; nocc * ngrids];
    re.par_chunks_mut(ngrids)
        .zip(im.par_chunks_mut(ngrids))
        .enumerate()
        .for_each(|(o, (orow, oirow))| {
            for p in 0..nao {
                let (cr, ci) = (scaled.re[p * nocc + o], scaled.im[p * nocc + o]);
                if cr == 0.0 && ci == 0.0 {
                    continue;
                }
                let ab = p * ngrids;
                for g in 0..ngrids {
                    let (ar, ai) = (ao.re[ab + g], ao.im[ab + g]);
                    orow[g] += cr * ar - ci * ai;
                    oirow[g] += cr * ai + ci * ar;
                }
            }
        });
    CTensor::from_planes(re, im)
}

/// `rho1[(a,i,j),g] = conj(dao[a,p0+i,g]) · expmikr[g] · ao2[j,g]` —
/// `fft_jk.py:391`, flattened to `(3·nblk·naoj, ngrids)` for the batched FFT.
/// `dao` component `a` lives at `Deriv1Table` component `a + 1` (component `0`
/// is the value table); `ao2` is `(naoj, ngrids)` with `naoj = nao` untagged
/// and `nocc` tagged.
///
/// Pure element-wise arithmetic, one worker per `(a, i)` slab — no reduction
/// anywhere, so the split cannot change a single rounding.
fn build_rho1_e1(
    dao: &CTensor,
    ao2: &CTensor,
    expmikr: Option<&CTensor>,
    p0: usize,
    p1: usize,
    nao: usize,
    naoj: usize,
    ngrids: usize,
) -> CTensor {
    let nblk = p1 - p0;
    let mut re = vec![0.0_f64; 3 * nblk * naoj * ngrids];
    let mut im = vec![0.0_f64; 3 * nblk * naoj * ngrids];
    let block = nblk * naoj * ngrids;
    re.par_chunks_mut(block)
        .zip(im.par_chunks_mut(block))
        .enumerate()
        .for_each(|(a, (orow, oirow))| {
            let c = a + 1;
            for i in 0..nblk {
                let p = p0 + i;
                let mut br = vec![0.0_f64; ngrids];
                let mut bi = vec![0.0_f64; ngrids];
                match expmikr {
                    None => {
                        for g in 0..ngrids {
                            let (r, im) = d1_at(dao, c, p, g, ngrids, nao);
                            br[g] = r;
                            bi[g] = -im;
                        }
                    }
                    Some(ph) => {
                        for g in 0..ngrids {
                            let (r, im) = d1_at(dao, c, p, g, ngrids, nao);
                            let (ar, ai) = (r, -im);
                            let (pr, pi) = (ph.re[g], ph.im[g]);
                            br[g] = ar * pr - ai * pi;
                            bi[g] = ar * pi + ai * pr;
                        }
                    }
                }
                for j in 0..naoj {
                    let ob = (i * naoj + j) * ngrids;
                    let jb = j * ngrids;
                    for g in 0..ngrids {
                        let (cr, ci) = (ao2.re[jb + g], ao2.im[jb + g]);
                        orow[ob + g] = br[g] * cr - bi[g] * ci;
                        oirow[ob + g] = br[g] * ci + bi[g] * cr;
                    }
                }
            }
        });
    CTensor::from_planes(re, im)
}

/// `vR_dm[p0+i,g] = Σ_j vR[(a,i,j),g]·ao_dm[j,g]` for one component `a` — the
/// `fft_jk.py:399` contraction (`np.einsum('aijg,jg->aig', vR, ao_dms[i],
/// out=vR_dm[:,i,p0:p1])`) with the component axis indexed by the caller.
///
/// `i` indexes disjoint output rows; `j` is serial ascending — the mirror of
/// the energy path's `contract_vr_aodm`, including its hoisted-`vR_dm`
/// contract (every element is overwritten on every pair, so no re-zeroing).
fn contract_vr_aodm_e1(
    vr_dm: &mut CTensor,
    vr: &CTensor,
    a: usize,
    ao_dm: &CTensor,
    p0: usize,
    nblk: usize,
    naoj: usize,
    ngrids: usize,
) {
    let abase = a * (vr.len() / 3);
    let lo = p0 * ngrids;
    let hi = (p0 + nblk) * ngrids;
    vr_dm.re[lo..hi]
        .par_chunks_mut(ngrids)
        .zip(vr_dm.im[lo..hi].par_chunks_mut(ngrids))
        .enumerate()
        .for_each(|(i, (orow, oirow))| {
            orow.fill(0.0);
            oirow.fill(0.0);
            for j in 0..naoj {
                let vb = abase + (i * naoj + j) * ngrids;
                let jb = j * ngrids;
                for g in 0..ngrids {
                    let (xr, xi) = (vr.re[vb + g], vr.im[vb + g]);
                    let (yr, yi) = (ao_dm.re[jb + g], ao_dm.im[jb + g]);
                    orow[g] += xr * yr - xi * yi;
                    oirow[g] += xr * yi + xi * yr;
                }
            }
        });
}

/// `vk[p,q] -= weight·Σ_g vR_dm[p,g]·ao1v[q,g]` — `fft_jk.py:407`
/// (`vk_kpts[:,i,k1] -= weight * einsum('aig,jg->aij', vR_dm[:,i], ao1T[0])`).
/// `ao1v` is the VALUE table: no conjugation here, exactly like the energy
/// path's `accumulate_vk` — the plain (unconjugated) dot through the same
/// pairwise-tree `oracle_dot`, one worker per output row `p`.
fn accumulate_vk_e1(
    vk: &mut CTensor,
    vr_dm: &CTensor,
    ao1v: &CTensor,
    weight: f64,
    nao: usize,
    ngrids: usize,
) {
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
                let yr = &ao1v.re[qb..qb + ngrids];
                let yi = &ao1v.im[qb..qb + ngrids];
                let rr = oracle_dot(xr, yr);
                let ii = oracle_dot(xi, yi);
                let ri = oracle_dot(xr, yi);
                let ir = oracle_dot(xi, yr);
                vrow[q] -= weight * (rr - ii);
                virow[q] -= weight * (ri + ir);
            }
        });
}

/// `get_k_e1_kpts(mydf, dm_kpts, kpts, kpts_band, exxdiv)` — `fft_jk.py:310-...`.
/// with the MO factorisation (`:357-359`, D-PBC-30 clause 4a).
///
/// Returns `vk[x][iset][kband]`, `nao x nao` row-major. Double loop — `:369`
/// over `k2`, `:381` over `k1` — with the inner `k1` loop tiled in chunks of
/// the budgeted residency `m` (D-PBC-31 clause 8): peak `(m+1)·4·ngrids·nao`
/// complex, `nkpts·ceil(nband/m)` deriv-1 chunk builds, reported in `stats`.
///
/// * `mo` carries the `mo_coeff`/`mo_occ` tag and is honoured **only when
///   `nset == 1`**, exactly like upstream (`:357`). Otherwise the untagged
///   route runs.
/// * `exxdiv` passes **through** to `get_coulG` inside the pair loop
///   (`fft_jk.py:384`), including `'ewald'`. Unlike the energy path there is
///   no post-loop `_ewald_exxdiv_for_G0` — the tail of `get_k_e1_kpts` is just
///   the format and return — so none is applied here either.
/// * `coulG` comes from the shared W-01 cache keyed by the k-difference
///   index (clause 9): `nkpts² → nkpts` builds, each `O(ngrids)` plus the
///   `exxdiv` correction riding on it.
/// * The `vR_dm *= expmikr.conj()` multiply (`:402`) is skipped on the
///   diagonal, where `expmikr` is the scalar `1.` (`:386-387`) — clause 11.
///   The diagonal is exactly where `is_zero(kpt1-kpt2)` holds, which is also
///   where the shared cache stores `None`, so the skip keys off the same
///   `Option`.
/// * There is deliberately no k-pair symmetry flag (clause 4b — see the
///   module docs). Use [`Fftdf::get_jk_e1`] for the guarded entry point.
///
/// # Errors
/// Propagates the AO evaluations, `get_coulG`, the FFT and a mistagged `mo`.
#[allow(clippy::too_many_arguments)]
pub fn get_k_e1_kpts(
    df: &Fftdf,
    dms: &[KMats],
    kpts: &[[f64; 3]],
    kpts_band: Option<&[[f64; 3]]>,
    exxdiv: Option<ExxDiv>,
    omega: Option<f64>,
    mo: Option<&TaggedMo>,
    stats: Option<&mut KGradStats>,
) -> Result<GradMats, PbcDfError> {
    let cell = &df.cell;
    let mesh = df.mesh;
    let nset = dms.len();
    let nkpts = kpts.len();
    let nao = cell.mol.nao_nr;
    let ngrids = df.grids.coords.len();

    if let Some(tag) = mo {
        if tag.blocks.len() != nkpts {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "get_k_e1_kpts: mo tag has {} k-points for {nkpts} sampling k-points",
                    tag.blocks.len()
                )),
            )));
        }
    }
    // fft_jk.py:357 — the tag collapses the ket only for a single set.
    let tagged = mo.is_some() && nset == 1;

    let band = format_kpts_band(kpts_band, kpts);
    let nband = band.len();
    // fft_jk.py:343.
    let weight = 1.0 / nkpts as f64 * df.weight();
    // Same reality rule as the energy path: gamma-only sampling and band.
    let real_out = all_gamma(band) && all_gamma(kpts);

    // The budgeted residency (clause 8) and the re-derived blksize
    // (clause 12). fft_jk.py:362 computes `max_memory = mydf.max_memory -
    // mem_now` and :363 subtracts `mem_now` AGAIN — the defect
    // (18-CONTEXT §3.7). Only one subtraction survives here:
    let m = ao_cache::resident_k_count(df.max_memory, ngrids, nao, nband);
    let mem_now_mb = ao_cache::resident_mb(m, ngrids, nao);
    let blksize = ao_cache::deriv1_blksize_from_avail(df.max_memory - mem_now_mb, ngrids, nao);
    let counter = AoEvalCount::default();

    // Value tables: ket at the sampling k-points (cached), bra at the band
    // (cached; the same table when there is no band).
    let ao2_kpts = df.ao_kpts(kpts)?;
    let ao1_kpts = if kpts_band.is_none() {
        ao2_kpts.clone()
    } else {
        df.ao_kpts(band)?
    };

    // fft_jk.py:357-359 — the MO ket tables, one per k2, beside the deriv-1
    // table (which stays alive as the bra side). Untagged, `ao2_kpts` entries
    // are used directly (upstream's views into the deriv-1 buffer, `:347-349`).
    let ket_kpts: Vec<CTensor> = if tagged {
        let tag = mo.expect("tagged mo checked above");
        tag.blocks
            .iter()
            .enumerate()
            .map(|(k, b)| mo_ket_table(&b.scaled, b.nocc, ao2_kpts.at(k), nao, ngrids))
            .collect()
    } else {
        Vec::new()
    };

    let gv = get_gv(cell, Some(mesh))?;

    let mut vk: GradMats = (0..3)
        .map(|_| {
            (0..nset)
                .map(|_| (0..nband).map(|_| CTensor::zeros(nao * nao)).collect())
                .collect()
        })
        .collect();
    // U-06, same reasoning as the energy path: hoisted, never re-zeroed —
    // `contract_vr_aodm_e1` fills every `p0:p1` row and the block loop covers
    // `0..nao` in full on every pair.
    let mut vr_dm: Vec<CTensor> = vec![CTensor::zeros(nao * ngrids); 3 * nset];

    // fft_jk.py:369 — for k2, ao2T in enumerate(ao2_kpts).
    for k2 in 0..nkpts {
        let (ao2t, naoj) = if tagged {
            (&ket_kpts[k2], mo.expect("tagged mo checked above").blocks[k2].nocc)
        } else {
            (ao2_kpts.at(k2), nao)
        };
        if naoj == 0 {
            continue;
        }
        // fft_jk.py:373-376 — untagged, `dms[i,k2] . conj(ao2T)`; tagged
        // (single set), bare `ao2T.conj()`.
        let ao_dms: Vec<CTensor> = if tagged {
            vec![ao2t.conj()]
        } else {
            dms.iter().map(|d| dm_times_conj_ao(&d[k2], ao2t, nao, ngrids)).collect()
        };

        // fft_jk.py:381 — the inner k1 loop, tiled in chunks of m. Each
        // chunk's deriv-1 tables are built on entry and dropped at the end
        // of the chunk: one code path from resident (`m = nband`) to
        // streaming (`m = 1`), counted on `counter`.
        for chunk in (0..nband).collect::<Vec<_>>().chunks(m) {
            let tabs =
                ao_cache::eval_deriv1_chunk(&df.cell, &df.grids.coords, band, chunk, Some(&counter))?;
            for tab in tabs.iter() {
                let k1 = tab.kidx;
                let kpt1 = band[k1];
                let kpt2 = kpts[k2];
                let dk = [kpt2[0] - kpt1[0], kpt2[1] - kpt1[1], kpt2[2] - kpt1[2]];

                let entry = df.coulg_and_expmikr(dk, omega, exxdiv, kpts, &gv)?;
                let (coulg, expmikr) = (entry.coulg.as_slice(), entry.expmikr.as_deref());

                // fft_jk.py:389-401 — the AO block loop over the bra index.
                let mut p0 = 0usize;
                while p0 < nao {
                    let p1 = (p0 + blksize).min(nao);

                    let rho1 =
                        build_rho1_e1(&tab.kao, ao2t, expmikr, p0, p1, nao, naoj, ngrids);
                    let mut vg = fft(&rho1, mesh)?;
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
                    if real_out {
                        for t in vr.im.iter_mut() {
                            *t = 0.0;
                        }
                    }

                    for a in 0..3 {
                        for i in 0..nset {
                            contract_vr_aodm_e1(
                                &mut vr_dm[a * nset + i],
                                &vr,
                                a,
                                &ao_dms[i],
                                p0,
                                p1 - p0,
                                naoj,
                                ngrids,
                            );
                        }
                    }
                    p0 = p1;
                }

                // fft_jk.py:402 — skipped on the diagonal, where expmikr is
                // the scalar 1. (clause 11).
                if let Some(ph) = expmikr {
                    for v in vr_dm.iter_mut() {
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

                // fft_jk.py:407.
                let ao1v = ao1_kpts.at(k1);
                for i in 0..nset {
                    for a in 0..3 {
                        accumulate_vk_e1(
                            &mut vk[a][i][k1],
                            &vr_dm[a * nset + i],
                            ao1v,
                            weight,
                            nao,
                            ngrids,
                        );
                    }
                }
            }
        }
    }

    if let Some(s) = stats {
        *s = KGradStats {
            m,
            blksize,
            chunk_builds: counter.builds(),
            k_tables: counter.kpoints(),
        };
    }
    Ok(vk)
}

/// The named refusal every non-FFTDF route serves (plan 18-04 Task 4):
/// PySCF 2.12.1 has no periodic analytic gradient for GDF/MDF/RSDF/AFTDF, and
/// the fix is to run the gradient on FFTDF, not to substitute a route. A
/// fallback here would produce a gradient that is plausible, wrong, and
/// consistent run to run — so this is an error, never a number.
pub fn grad_route_refusal(route: &str, what: &str) -> PbcDfError {
    PbcDfError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(format!(
            "{route} has no periodic analytic gradient in PySCF 2.12.1: {what} \
             exists only on FFTDF (pyscf/pbc/df/fft.py:324-340). Run the gradient \
             on FFTDF; do not substitute a route."
        )),
    ))
}

impl Fftdf {
    /// `FFTDF.get_jk_e1(dm, kpts, kpts_band, exxdiv)` — `fft.py:324-328`.
    ///
    /// The gradient route takes no k-pair symmetry flag (clause 4b — the
    /// derivative sits on the bra, so the energy-path conjugate identity does
    /// not close). A `JkOpts` that carries one is **refused** rather than
    /// ignored: a wrongly enabled flag here moves the last bits of an
    /// otherwise-correct gradient, the hardest class of error to attribute. A
    /// comment asking implementers not to enable it is not a mechanism.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] when `opts` carries the k-pair flag; otherwise as
    /// [`get_j_e1_kpts`] / [`get_k_e1_kpts`].
    pub fn get_jk_e1(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        opts: JkOpts<'_>,
        mo: Option<&TaggedMo>,
    ) -> Result<GradJkResult, PbcDfError> {
        if opts.kk_symmetry {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(
                    "FFTDF gradient JK refuses kk_symmetry: the energy-path (k1,k2) \
                     <-> (k2,k1) conjugate identity needs the bra and ket to carry \
                     the same AO table, but get_k_e1_kpts carries the derivative AO \
                     on the bra (18-CONTEXT §3.2) — the (j,i) swap moves the \
                     derivative onto the ket and the identity does not close."
                        .into(),
                ),
            )));
        }
        let vj = if opts.with_j {
            Some(get_j_e1_kpts(self, dms, kpts, opts.kpts_band, None)?)
        } else {
            None
        };
        let vk = if opts.with_k {
            Some(get_k_e1_kpts(
                self,
                dms,
                kpts,
                opts.kpts_band,
                opts.exxdiv,
                opts.omega,
                mo,
                None,
            )?)
        } else {
            None
        };
        Ok(GradJkResult { vj, vk })
    }

    /// `FFTDF.get_j_e1(dm, kpts, kpts_band)` — `fft.py:330-333`.
    ///
    /// # Errors
    /// As [`get_j_e1_kpts`].
    pub fn get_j_e1(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<GradMats, PbcDfError> {
        get_j_e1_kpts(self, dms, kpts, kpts_band, None)
    }

    /// `FFTDF.get_k_e1(dm, kpts, kpts_band, exxdiv)` — `fft.py:335-340`.
    ///
    /// # Errors
    /// As [`get_k_e1_kpts`].
    pub fn get_k_e1(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
        exxdiv: Option<ExxDiv>,
        omega: Option<f64>,
        mo: Option<&TaggedMo>,
    ) -> Result<GradMats, PbcDfError> {
        get_k_e1_kpts(self, dms, kpts, kpts_band, exxdiv, omega, mo, None)
    }
}
