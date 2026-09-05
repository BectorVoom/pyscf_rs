//! Periodic AO→MO integral transforms — `aft_ao2mo.py` and `fft_ao2mo.py`
//! (plan 13-06).
//!
//! # The contraction, once, for both builders
//!
//! ```text
//! (pq|rs) = Σ_G  coulG(G+q) · ρ_pq(G) · conj(ρ'_rs(G))
//! ```
//!
//! with `q = k_j − k_i`, `ρ_pq` the AO-pair Fourier transform at
//! `(q, kpt = k_j)`, and — the part that is easy to get wrong —
//! `ρ'_rs` the transform at `(q, kpt = −k_l)`, **the same `q`, negated ket
//! k-point**, conjugated on contraction. `aft_ao2mo.py:104-118` derives it: for
//! real AO functions
//!
//! ```text
//! ρ_rs(−G+k_rs) = conj( Σ_{Ts} e^{−i k_s·Ts} ∫ e^{−i(G+k_pq)·r} r(r) s(r−Ts) dr )
//!               = conj( pw_loop(−k_s, G+k_pq) )
//! ```
//!
//! Both builders use this identical contraction; they differ only in how `ρ` is
//! produced — analytically (AFTDF) or as the FFT of the real-space AO product
//! (FFTDF). That is what makes the cross-builder test a measurement of
//! `ft_aopair` rather than of the transform.
//!
//! # Scope — the 13-06 carry-over is CLOSED here (plan 14-05)
//!
//! Plan 13-06 shipped `get_eri` and `get_ao_pairs_g` and deliberately withheld
//! `general`, `get_mo_pairs_G` and `ao2mo_7d`, on the stated grounds that
//! `ao2mo_7d`'s index order is a contract with Phase 15's KMP2 and should not
//! be guessed. Plan 14-05 settles that order against upstream for all three
//! builders at once — **see [`crate::df_ao2mo`]'s module docs, which state it
//! once** — and the remaining three functions ship here.
//!
//! ## One deviation, stated
//!
//! Upstream's `aft_ao2mo.general` and `fft_ao2mo.ao2mo_7d` fold the MO
//! transform INSIDE the `pw_loop` / FFT sweep so the `nao^4` AO block is never
//! materialised. This port transforms the ASSEMBLED AO block that
//! [`aft_get_eri`] / [`fft_get_eri`] already return. The numbers are the same
//! contraction in a different association order (agreement is asserted against
//! upstream at 1e-11 in `tests/df_ao2mo.rs`); what is given up is memory, and
//! the builder that actually cares about memory is GDF, whose `general` in
//! [`crate::df_ao2mo`] IS factorised through `cderi`.

use pyscf_algebra::{CTensor, oracle_zdotu};
use pyscf_pbc_gto::Cell;

use crate::aftdf::Aftdf;
use crate::error::PbcDfError;
use crate::fftdf::Fftdf;
use crate::ft_ao::{FtKernel, FtScreen, RcutChoice};

/// Caller-owned reciprocal-grid data at one momentum transfer. KMP2 reuses
/// one of these per distinct `q` instead of rebuilding it for every triple.
#[derive(Debug, Clone)]
pub struct CoulGCache {
    pub q: [f64; 3],
    pub mesh: [usize; 3],
    pub gv: Vec<[f64; 3]>,
    pub coulg: Vec<f64>,
    pub weighted_coulg: Vec<f64>,
}

impl CoulGCache {
    pub fn build(cell: &Cell, mesh: [usize; 3], q: [f64; 3]) -> Result<Self, PbcDfError> {
        let weights = pyscf_pbc_gto::gv::get_gv_weights(cell, Some(mesh))?;
        let gv = pyscf_pbc_gto::gv::get_gv(cell, Some(mesh))?;
        let coulg = pyscf_pbc_gto::get_coulg(
            cell,
            pyscf_pbc_gto::CoulGArgs {
                k: q,
                mesh: Some(mesh),
                gv: Some(&gv),
                ..Default::default()
            },
        )?;
        let mut weighted_coulg = coulg.clone();
        for (g, v) in weighted_coulg.iter_mut().enumerate() {
            *v *= weights.weight(g);
        }
        Ok(Self {
            q,
            mesh,
            gv,
            coulg,
            weighted_coulg,
        })
    }
}

/// `_iskconserv(cell, kptijkl)` — `aft_ao2mo.py` via `kpts_helper`.
///
/// `(k_i − k_j + k_k − k_l) · a / 2π` must be integral on every lattice vector.
pub fn is_kconserv(cell: &Cell, k: &[[f64; 3]; 4]) -> bool {
    let d = [
        k[0][0] - k[1][0] + k[2][0] - k[3][0],
        k[0][1] - k[1][1] + k[2][1] - k[3][1],
        k[0][2] - k[1][2] + k[2][2] - k[3][2],
    ];
    let a = cell.a;
    (0..3).all(|i| {
        let x = (d[0] * a[i][0] + d[1] * a[i][1] + d[2] * a[i][2]) / (2.0 * std::f64::consts::PI);
        (x - x.round()).abs() < 1e-9
    })
}

/// `get_ao_pairs_G` for AFTDF — the analytic AO-pair transform, dense
/// `(ngrids, nao, nao)` planar.
///
/// # Errors
/// Propagates the kernel build and launch.
pub fn aft_ao_pairs_g(
    df: &Aftdf,
    kpt_ket: [f64; 3],
    q: [f64; 3],
    gv: &[[f64; 3]],
) -> Result<(Vec<f64>, Vec<f64>), PbcDfError> {
    let radius = df.rcut.resolve_for(&df.cell)?;
    let ls = pyscf_pbc_gto::lattice::get_lattice_ls(&df.cell, Some(radius), None, true)?;
    let screen = if df.rcut == RcutChoice::Upstream {
        FtScreen::Upstream
    } else {
        FtScreen::None
    };
    let k = FtKernel::build(&df.cell, kpt_ket, &ls, screen)?;
    let o = k.eval(&df.cell, gv, q)?;
    Ok((o.re, o.im))
}

/// `get_ao_pairs_G` for FFTDF — the FFT of the real-space AO product.
///
/// `ρ_pq(G+q) = FFT[ conj(φ_p^{k_i}(r)) · φ_q^{k_j}(r) · e^{−iq·r} ] · (Ω/N)`,
/// which is the discrete approximation to the same integral `ft_aopair`
/// evaluates analytically — so the two feed the SAME contraction and their
/// difference is FFTDF's aliasing.
///
/// # Errors
/// Propagates the AO evaluation and the FFT.
pub fn fft_ao_pairs_g(
    df: &Fftdf,
    kpts2: [[f64; 3]; 2],
    q: [f64; 3],
) -> Result<(Vec<f64>, Vec<f64>), PbcDfError> {
    let cell = &df.cell;
    let mesh = df.mesh;
    let nao = cell.mol.nao_nr;
    let coords = pyscf_pbc_gto::gv::get_uniform_grids(cell, Some(mesh), false)?;
    let ngrids = coords.len();
    let ao = df.ao_kpts(&kpts2)?;
    let scale = cell.vol() / ngrids as f64;

    // e^{−iq·r} on the grid, so the FFT lands on `G + q` rather than `G`.
    let (mut qr, mut qi) = (vec![1.0f64; ngrids], vec![0.0f64; ngrids]);
    if q.iter().any(|c| c.abs() > 1e-12) {
        for (r, c) in coords.iter().enumerate() {
            let th = -(q[0] * c[0] + q[1] * c[1] + q[2] * c[2]);
            let (s, co) = th.sin_cos();
            qr[r] = co;
            qi[r] = s;
        }
    }

    let mut out_re = vec![0.0f64; ngrids * nao * nao];
    let mut out_im = vec![0.0f64; ngrids * nao * nao];
    let (ai, aj) = (ao.at(0), ao.at(1));
    let mut buf = CTensor {
        re: vec![0.0; ngrids],
        im: vec![0.0; ngrids],
    };
    for p in 0..nao {
        for qq in 0..nao {
            for r in 0..ngrids {
                // AO blocks are F-order `(ngrids, nao)`.
                let (pr, pim) = (ai.re[r + p * ngrids], -ai.im[r + p * ngrids]); // conj
                let (kr, ki) = (aj.re[r + qq * ngrids], aj.im[r + qq * ngrids]);
                let (mr, mi) = (pr * kr - pim * ki, pr * ki + pim * kr);
                buf.re[r] = (mr * qr[r] - mi * qi[r]) * scale;
                buf.im[r] = (mr * qi[r] + mi * qr[r]) * scale;
            }
            let f = pyscf_pbc_tools::fft::fft(&buf, mesh)?;
            for g in 0..ngrids {
                out_re[g * nao * nao + p * nao + qq] = f.re[g];
                out_im[g * nao * nao + p * nao + qq] = f.im[g];
            }
        }
    }
    Ok((out_re, out_im))
}

/// `(pq|rs) = Σ_G coulG(G+q)·ρ_pq(G)·conj(ρ'_rs(G))`, returned as the
/// `nao² × nao²` matrix in row-major `[(pq), (rs)]` order.
fn contract_eri(
    pq: &(Vec<f64>, Vec<f64>),
    rs: &(Vec<f64>, Vec<f64>),
    coulg: &[f64],
    nao: usize,
) -> CTensor {
    let n2 = nao * nao;
    let mut out = CTensor {
        re: vec![0.0; n2 * n2],
        im: vec![0.0; n2 * n2],
    };
    for (g, &w) in coulg.iter().enumerate() {
        if w == 0.0 {
            continue;
        }
        let base = g * n2;
        for a in 0..n2 {
            let (ar, ai) = (pq.0[base + a] * w, pq.1[base + a] * w);
            if ar == 0.0 && ai == 0.0 {
                continue;
            }
            for b in 0..n2 {
                let (br, bi) = (rs.0[base + b], rs.1[base + b]);
                // a · conj(b)
                out.re[a * n2 + b] += ar * br + ai * bi;
                out.im[a * n2 + b] += ai * br - ar * bi;
            }
        }
    }
    out
}

/// `AFTDF.get_eri(kptijkl)` — `aft_ao2mo.py:37-124`.
///
/// Returns the `nao² × nao²` matrix. Momentum non-conservation returns zeros,
/// as upstream does (with a warning there, a `tracing::warn` here).
///
/// # Errors
/// Propagates `weighted_coulG` and the AO-pair transforms.
pub fn aft_get_eri(df: &Aftdf, kptijkl: [[f64; 3]; 4]) -> Result<CTensor, PbcDfError> {
    let cell = &df.cell;
    let nao = cell.mol.nao_nr;
    let n2 = nao * nao;
    if !is_kconserv(cell, &kptijkl) {
        tracing::warn!("aft_ao2mo: momentum conservation not found in the given k-points");
        return Ok(CTensor {
            re: vec![0.0; n2 * n2],
            im: vec![0.0; n2 * n2],
        });
    }
    let q = [
        kptijkl[1][0] - kptijkl[0][0],
        kptijkl[1][1] - kptijkl[0][1],
        kptijkl[1][2] - kptijkl[0][2],
    ];
    let mesh = df.mesh;
    let gv = df.gv(mesh)?;
    let coulg = df.weighted_coulg(q, None, mesh, None)?;
    let pq = aft_ao_pairs_g(df, kptijkl[1], q, &gv)?;
    // NOTE the negated ket k-point, same `q` — `aft_ao2mo.py:104-118`.
    let neg_l = [-kptijkl[3][0], -kptijkl[3][1], -kptijkl[3][2]];
    let rs = aft_ao_pairs_g(df, neg_l, q, &gv)?;
    Ok(contract_eri(&pq, &rs, &coulg, nao))
}

pub fn aft_get_eri_with_cache(
    df: &Aftdf,
    kptijkl: [[f64; 3]; 4],
    cache: &CoulGCache,
) -> Result<CTensor, PbcDfError> {
    let nao = df.cell.mol.nao_nr;
    if !is_kconserv(&df.cell, &kptijkl) {
        return Ok(CTensor::zeros(nao.pow(4)));
    }
    let q = [
        kptijkl[1][0] - kptijkl[0][0],
        kptijkl[1][1] - kptijkl[0][1],
        kptijkl[1][2] - kptijkl[0][2],
    ];
    if cache.mesh != df.mesh || cache.q.iter().zip(q).any(|(a, b)| (*a - b).abs() > 1e-12) {
        return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(
                "CoulGCache does not match AFT AO2MO q/mesh".into(),
            ),
        )));
    }
    let pq = aft_ao_pairs_g(df, kptijkl[1], q, &cache.gv)?;
    let neg_l = [-kptijkl[3][0], -kptijkl[3][1], -kptijkl[3][2]];
    let rs = aft_ao_pairs_g(df, neg_l, q, &cache.gv)?;
    Ok(contract_eri(&pq, &rs, &cache.weighted_coulg, nao))
}

/// `FFTDF.get_eri(kptijkl)` — `fft_ao2mo.py`, through the SAME contraction as
/// [`aft_get_eri`].
///
/// Plan 13-06 pulls this into Phase 13 (Phase 11 skipped it) so `aft_get_eri`
/// has an independent same-phase cross-check rather than only an oracle.
///
/// # Errors
/// Propagates the AO evaluation, the FFT and `get_coulG`.
pub fn fft_get_eri(df: &Fftdf, kptijkl: [[f64; 3]; 4]) -> Result<CTensor, PbcDfError> {
    let cell = &df.cell;
    let nao = cell.mol.nao_nr;
    let n2 = nao * nao;
    if !is_kconserv(cell, &kptijkl) {
        tracing::warn!("fft_ao2mo: momentum conservation not found");
        return Ok(CTensor {
            re: vec![0.0; n2 * n2],
            im: vec![0.0; n2 * n2],
        });
    }
    let q = [
        kptijkl[1][0] - kptijkl[0][0],
        kptijkl[1][1] - kptijkl[0][1],
        kptijkl[1][2] - kptijkl[0][2],
    ];
    let cache = CoulGCache::build(cell, df.mesh, q)?;
    fft_get_eri_with_cache(df, kptijkl, &cache)
}

pub fn fft_get_eri_with_cache(
    df: &Fftdf,
    kptijkl: [[f64; 3]; 4],
    cache: &CoulGCache,
) -> Result<CTensor, PbcDfError> {
    let cell = &df.cell;
    let nao = cell.mol.nao_nr;
    let n2 = nao * nao;
    if !is_kconserv(cell, &kptijkl) {
        return Ok(CTensor::zeros(n2 * n2));
    }
    let q = [
        kptijkl[1][0] - kptijkl[0][0],
        kptijkl[1][1] - kptijkl[0][1],
        kptijkl[1][2] - kptijkl[0][2],
    ];
    if cache.mesh != df.mesh || cache.q.iter().zip(q).any(|(a, b)| (*a - b).abs() > 1e-12) {
        return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(
                "CoulGCache does not match FFT AO2MO q/mesh".into(),
            ),
        )));
    }
    // Use the real-space `_contract_plain` convention even for AO output.
    // At a Brillouin-zone boundary, reconstructing the ket through
    // `(-k_k,-k_l,q)` requires an additional reciprocal-vector permutation of
    // the FFT bins; treating the bins as identical gives the wrong non-gamma
    // block. Identity MOs preserve the public AO layout while sharing the
    // upstream phase spelling with `fft_ao2mo.general`.
    let identity = MoCoeff::identity(nao);
    Ok(fft_general_mo_first(
        df,
        [&identity, &identity, &identity, &identity],
        kptijkl,
        Some(cache),
    )?
    .data)
}

// ---------------------------------------------------------------------------
// The 13-06 carry-over: general, get_mo_pairs_G and ao2mo_7d (plan 14-05)
// ---------------------------------------------------------------------------

use crate::df_ao2mo::{Eri, Eri7d, MoCoeff, MoKpts, PairDims, warn_pbc2d_eri};

/// `eri_mo[ij, kl] = Σ_pqrs conj(Ci[p,i]) Cj[q,j] · ao[pq, rs] · conj(Ck[r,k]) Cl[s,l]`
///
/// The AO block is the `nao² × nao²` `s1` matrix [`aft_get_eri`] /
/// [`fft_get_eri`] return; the conjugation convention is `_ao2mo.r_e2`'s — the
/// BRA of each pair, see [`crate::df_ao2mo`].
///
/// Done as four half-transforms so the cost is `nao^4 · nmo` rather than
/// `nao^4 · nmo^4`.
///
/// # Panics
/// When any coefficient block's `nao` disagrees with `nao`.
pub fn transform_ao_eri(ao: &CTensor, nao: usize, mos: [&MoCoeff; 4]) -> CTensor {
    for m in mos {
        assert_eq!(m.nao, nao, "transform_ao_eri: MO block has the wrong nao");
    }
    let n2 = nao * nao;
    let (ni, nj, nk, nl) = (mos[0].nmo, mos[1].nmo, mos[2].nmo, mos[3].nmo);

    // Column half-transform first — it shrinks the big axis soonest.
    // t1[pq, k s] = Σ_r conj(Ck[r,k]) ao[pq, r s]
    let mut t1 = CTensor::zeros(n2 * nk * nao);
    for row in 0..n2 {
        for r in 0..nao {
            for kk in 0..nk {
                let (cr, ci) = (mos[2].c.re[r * nk + kk], -mos[2].c.im[r * nk + kk]);
                if cr == 0.0 && ci == 0.0 {
                    continue;
                }
                for s in 0..nao {
                    let (ar, ai) = (ao.re[row * n2 + r * nao + s], ao.im[row * n2 + r * nao + s]);
                    let o = row * nk * nao + kk * nao + s;
                    t1.re[o] += cr * ar - ci * ai;
                    t1.im[o] += cr * ai + ci * ar;
                }
            }
        }
    }
    // t2[pq, k l] = Σ_s t1[pq, k s] Cl[s,l]
    let mut t2 = CTensor::zeros(n2 * nk * nl);
    for row in 0..n2 {
        for kk in 0..nk {
            for s in 0..nao {
                let o = row * nk * nao + kk * nao + s;
                let (tr, ti) = (t1.re[o], t1.im[o]);
                if tr == 0.0 && ti == 0.0 {
                    continue;
                }
                for l in 0..nl {
                    let (cr, ci) = (mos[3].c.re[s * nl + l], mos[3].c.im[s * nl + l]);
                    let d = row * nk * nl + kk * nl + l;
                    t2.re[d] += tr * cr - ti * ci;
                    t2.im[d] += tr * ci + ti * cr;
                }
            }
        }
    }
    // t3[i q, kl] = Σ_p conj(Ci[p,i]) t2[p q, kl]
    let ncol = nk * nl;
    let mut t3 = CTensor::zeros(ni * nao * ncol);
    for p in 0..nao {
        for i in 0..ni {
            let (cr, ci) = (mos[0].c.re[p * ni + i], -mos[0].c.im[p * ni + i]);
            if cr == 0.0 && ci == 0.0 {
                continue;
            }
            for q in 0..nao {
                for c in 0..ncol {
                    let s = (p * nao + q) * ncol + c;
                    let (tr, ti) = (t2.re[s], t2.im[s]);
                    let d = (i * nao + q) * ncol + c;
                    t3.re[d] += cr * tr - ci * ti;
                    t3.im[d] += cr * ti + ci * tr;
                }
            }
        }
    }
    // out[i j, kl] = Σ_q t3[i q, kl] Cj[q,j]
    let mut out = CTensor::zeros(ni * nj * ncol);
    for i in 0..ni {
        for q in 0..nao {
            for j in 0..nj {
                let (cr, ci) = (mos[1].c.re[q * nj + j], mos[1].c.im[q * nj + j]);
                if cr == 0.0 && ci == 0.0 {
                    continue;
                }
                for c in 0..ncol {
                    let s = (i * nao + q) * ncol + c;
                    let (tr, ti) = (t3.re[s], t3.im[s]);
                    let d = (i * nj + j) * ncol + c;
                    out.re[d] += tr * cr - ti * ci;
                    out.im[d] += tr * ci + ti * cr;
                }
            }
        }
    }
    out
}

/// `get_mo_pairs_G` — `fft_ao2mo.py:284-343` / `aft_ao2mo.py`.
///
/// `mo_pairs[G, i*nmoj + j] = Σ_pq conj(Ci[p,i]) · ao_pairs[G, p, q] · Cj[q,j]`,
/// with `ao_pairs` one of [`aft_ao_pairs_g`] / [`fft_ao_pairs_g`] — the same
/// `(ngrids, nao, nao)` planar layout those return.
///
/// # Panics
/// When `ao_pairs` is not a whole number of `nao²` blocks.
pub fn get_mo_pairs_g(
    ao_pairs: &(Vec<f64>, Vec<f64>),
    nao: usize,
    a: &MoCoeff,
    b: &MoCoeff,
) -> CTensor {
    let n2 = nao * nao;
    assert_eq!(ao_pairs.0.len() % n2, 0, "get_mo_pairs_g: ragged AO pairs");
    let ngrids = ao_pairs.0.len() / n2;
    let (ni, nj) = (a.nmo, b.nmo);
    let mut out = CTensor::zeros(ngrids * ni * nj);
    let mut t = CTensor::zeros(ni * nao);
    for g in 0..ngrids {
        t.re.iter_mut().for_each(|v| *v = 0.0);
        t.im.iter_mut().for_each(|v| *v = 0.0);
        let base = g * n2;
        for p in 0..nao {
            for i in 0..ni {
                let (cr, ci) = (a.c.re[p * ni + i], -a.c.im[p * ni + i]);
                if cr == 0.0 && ci == 0.0 {
                    continue;
                }
                for q in 0..nao {
                    let (lr, li) = (
                        ao_pairs.0[base + p * nao + q],
                        ao_pairs.1[base + p * nao + q],
                    );
                    t.re[i * nao + q] += cr * lr - ci * li;
                    t.im[i * nao + q] += cr * li + ci * lr;
                }
            }
        }
        let o = g * ni * nj;
        for i in 0..ni {
            for q in 0..nao {
                let (tr, ti) = (t.re[i * nao + q], t.im[i * nao + q]);
                if tr == 0.0 && ti == 0.0 {
                    continue;
                }
                for j in 0..nj {
                    let (cr, ci) = (b.c.re[q * nj + j], b.c.im[q * nj + j]);
                    out.re[o + i * nj + j] += tr * cr - ti * ci;
                    out.im[o + i * nj + j] += tr * ci + ti * cr;
                }
            }
        }
    }
    out
}

/// `aft_ao2mo.general(mydf, mo_coeffs, kpts)` — `aft_ao2mo.py:125-217`.
///
/// # Errors
/// Propagates [`aft_get_eri`].
pub fn aft_general(
    df: &Aftdf,
    mos: [&MoCoeff; 4],
    kptijkl: [[f64; 3]; 4],
) -> Result<Eri, PbcDfError> {
    warn_pbc2d_eri(&df.cell);
    let nao = df.cell.mol.nao_nr;
    let ao = aft_get_eri(df, kptijkl)?;
    Ok(mo_eri(&ao, nao, mos))
}

fn transform_pairs_g(
    pair: &(Vec<f64>, Vec<f64>),
    ng: usize,
    a: &MoCoeff,
    b: &MoCoeff,
) -> (Vec<f64>, Vec<f64>) {
    let out = get_mo_pairs_g(pair, a.nao, a, b);
    debug_assert_eq!(out.re.len(), ng * a.nmo * b.nmo);
    (out.re, out.im)
}

/// Transform analytical AO-pair Fourier slices into MO-pair slices before
/// forming the four-index block, avoiding a materialised `nao^2 x nao^2` ERI.
pub fn aft_general_mo_first(
    df: &Aftdf,
    mos: [&MoCoeff; 4],
    kptijkl: [[f64; 3]; 4],
    cache: Option<&CoulGCache>,
) -> Result<Eri, PbcDfError> {
    warn_pbc2d_eri(&df.cell);
    let dims = mos.map(|m| m.nmo);
    if !is_kconserv(&df.cell, &kptijkl) {
        return Ok(Eri {
            data: CTensor::zeros(dims.iter().product()),
            row: PairDims::plain(dims[0], dims[1]),
            col: PairDims::plain(dims[2], dims[3]),
        });
    }
    let q = [
        kptijkl[1][0] - kptijkl[0][0],
        kptijkl[1][1] - kptijkl[0][1],
        kptijkl[1][2] - kptijkl[0][2],
    ];
    let owned_gv;
    let owned_w;
    let (gv, w) = match cache {
        Some(c) => {
            if c.mesh != df.mesh || c.q.iter().zip(q).any(|(a, b)| (*a - b).abs() > 1e-12) {
                return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                    pyscf_core::CoreError::InvalidMolecule(
                        "CoulGCache does not match AFT MO-first q/mesh".into(),
                    ),
                )));
            }
            (&c.gv[..], &c.weighted_coulg[..])
        }
        None => {
            owned_gv = df.gv(df.mesh)?;
            owned_w = df.weighted_coulg(q, None, df.mesh, None)?;
            (&owned_gv[..], &owned_w[..])
        }
    };
    let pq = transform_pairs_g(
        &aft_ao_pairs_g(df, kptijkl[1], q, gv)?,
        gv.len(),
        mos[0],
        mos[1],
    );
    let neg_l = [-kptijkl[3][0], -kptijkl[3][1], -kptijkl[3][2]];
    // `aft_ao2mo.py:215-216` conjugates the ket AO pair BEFORE the MO
    // transform (`rskR - rskI*1j`, then `r_e2`), and `r_e2` itself conjugates
    // the bra coefficient. Conjugating the TRANSFORMED pair instead gives
    // `conj(conj(c_k) c_l A) = c_k conj(c_l) conj(A)` — the wrong one of the
    // two coefficients — which is invisible for real coefficients and wrong by
    // O(1) for complex ones.
    let rs_ao = aft_ao_pairs_g(df, neg_l, q, gv)?;
    let rs_ao = (rs_ao.0, rs_ao.1.iter().map(|v| -v).collect::<Vec<f64>>());
    let rs = transform_pairs_g(&rs_ao, gv.len(), mos[2], mos[3]);
    let (nrow, ncol) = (dims[0] * dims[1], dims[2] * dims[3]);
    let mut data = CTensor::zeros(nrow * ncol);
    for a in 0..nrow {
        for b in 0..ncol {
            let mut x = CTensor::zeros(gv.len());
            let mut y = CTensor::zeros(gv.len());
            for (g, &wg) in w.iter().enumerate().take(gv.len()) {
                x.re[g] = pq.0[g * nrow + a] * wg;
                x.im[g] = pq.1[g * nrow + a] * wg;
                y.re[g] = rs.0[g * ncol + b];
                y.im[g] = rs.1[g * ncol + b];
            }
            let (re, im) = oracle_zdotu(&x, &y);
            data.re[a * ncol + b] = re;
            data.im[a * ncol + b] = im;
        }
    }
    Ok(Eri {
        data,
        row: PairDims::plain(dims[0], dims[1]),
        col: PairDims::plain(dims[2], dims[3]),
    })
}

/// `fft_ao2mo.general(mydf, mo_coeffs, kpts)` — `fft_ao2mo.py:101-155`.
///
/// # Errors
/// Propagates [`fft_get_eri`].
pub fn fft_general(
    df: &Fftdf,
    mos: [&MoCoeff; 4],
    kptijkl: [[f64; 3]; 4],
) -> Result<Eri, PbcDfError> {
    warn_pbc2d_eri(&df.cell);
    let nao = df.cell.mol.nao_nr;
    let ao = fft_get_eri(df, kptijkl)?;
    Ok(mo_eri(&ao, nao, mos))
}

fn mo_on_grid(mo: &MoCoeff, ao: &CTensor, ngrids: usize) -> CTensor {
    let mut out = CTensor::zeros(mo.nmo * ngrids);
    for m in 0..mo.nmo {
        for r in 0..ngrids {
            for p in 0..mo.nao {
                let ci = p * mo.nmo + m;
                let ai = p * ngrids + r;
                let (cr, cim) = (mo.c.re[ci], mo.c.im[ci]);
                let (ar, aim) = (ao.re[ai], ao.im[ai]);
                let z = m * ngrids + r;
                out.re[z] += cr * ar - cim * aim;
                out.im[z] += cr * aim + cim * ar;
            }
        }
    }
    out
}

/// The actual upstream FFT AO2MO route: AO→MO on the real-space grid before
/// pair formation. One bra pair is live at a time, bounding scratch memory.
pub fn fft_general_mo_first(
    df: &Fftdf,
    mos: [&MoCoeff; 4],
    kptijkl: [[f64; 3]; 4],
    cache: Option<&CoulGCache>,
) -> Result<Eri, PbcDfError> {
    warn_pbc2d_eri(&df.cell);
    let dims = mos.map(|m| m.nmo);
    if !is_kconserv(&df.cell, &kptijkl) {
        return Ok(Eri {
            data: CTensor::zeros(dims.iter().product()),
            row: PairDims::plain(dims[0], dims[1]),
            col: PairDims::plain(dims[2], dims[3]),
        });
    }
    let q = [
        kptijkl[1][0] - kptijkl[0][0],
        kptijkl[1][1] - kptijkl[0][1],
        kptijkl[1][2] - kptijkl[0][2],
    ];
    let owned;
    let cache = match cache {
        Some(v) => v,
        None => {
            owned = CoulGCache::build(&df.cell, df.mesh, q)?;
            &owned
        }
    };
    if cache.mesh != df.mesh || cache.q.iter().zip(q).any(|(a, b)| (*a - b).abs() > 1e-12) {
        return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(
                "CoulGCache does not match FFT MO-first q/mesh".into(),
            ),
        )));
    }
    let ao = df.ao_kpts(&kptijkl)?;
    let ng = ao.ngrids;
    let mg = [
        mo_on_grid(mos[0], ao.at(0), ng),
        mo_on_grid(mos[1], ao.at(1), ng),
        mo_on_grid(mos[2], ao.at(2), ng),
        mo_on_grid(mos[3], ao.at(3), ng),
    ];
    let mut phase = CTensor::zeros(ng);
    for (r, c) in df.grids.coords.iter().enumerate() {
        let th = -(q[0] * c[0] + q[1] * c[1] + q[2] * c[2]);
        let (s, co) = th.sin_cos();
        phase.re[r] = co;
        phase.im[r] = s;
    }
    let mut data = CTensor::zeros(dims.iter().product());
    let wcoul_scale = df.cell.vol() / ng as f64;
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            let mut pair = CTensor::zeros(ng);
            for r in 0..ng {
                let x = i * ng + r;
                let y = j * ng + r;
                let (xr, xi) = (mg[0].re[x], -mg[0].im[x]);
                let (yr, yi) = (mg[1].re[y], mg[1].im[y]);
                let (pr, pi) = (xr * yr - xi * yi, xr * yi + xi * yr);
                pair.re[r] = pr * phase.re[r] - pi * phase.im[r];
                pair.im[r] = pr * phase.im[r] + pi * phase.re[r];
            }
            let mut v = pyscf_pbc_tools::fft::fft(&pair, df.mesh)?;
            for g in 0..ng {
                v.re[g] *= cache.coulg[g] * wcoul_scale;
                v.im[g] *= cache.coulg[g] * wcoul_scale;
            }
            v = pyscf_pbc_tools::fft::ifft(&v, df.mesh)?;
            for r in 0..ng {
                let (vr, vi) = (v.re[r], v.im[r]);
                v.re[r] = vr * phase.re[r] + vi * phase.im[r];
                v.im[r] = vi * phase.re[r] - vr * phase.im[r];
            }
            for k in 0..dims[2] {
                for l in 0..dims[3] {
                    let mut ket = CTensor::zeros(ng);
                    for r in 0..ng {
                        let x = k * ng + r;
                        let y = l * ng + r;
                        let (xr, xi) = (mg[2].re[x], -mg[2].im[x]);
                        let (yr, yi) = (mg[3].re[y], mg[3].im[y]);
                        ket.re[r] = xr * yr - xi * yi;
                        ket.im[r] = xr * yi + xi * yr;
                    }
                    let (re, im) = oracle_zdotu(&v, &ket);
                    let z = ((i * dims[1] + j) * dims[2] + k) * dims[3] + l;
                    data.re[z] = re;
                    data.im[z] = im;
                }
            }
        }
    }
    // DEVIATION: fft_ao2mo.py:126-143's gamma/all-real compact shortcut is
    // intentionally omitted; KMP2 requests non-identical (o,v,o,v) blocks.
    Ok(Eri {
        data,
        row: PairDims::plain(dims[0], dims[1]),
        col: PairDims::plain(dims[2], dims[3]),
    })
}

fn mo_eri(ao: &CTensor, nao: usize, mos: [&MoCoeff; 4]) -> Eri {
    Eri {
        data: transform_ao_eri(ao, nao, mos),
        row: PairDims::plain(mos[0].nmo, mos[1].nmo),
        col: PairDims::plain(mos[2].nmo, mos[3].nmo),
    }
}

/// `aft_ao2mo.ao2mo_7d` — `aft_ao2mo.py:294-…`. The index contract is
/// [`crate::df_ao2mo`]'s and is shared by all three builders.
///
/// # Errors
/// Propagates [`aft_get_eri`]; errors on a ragged or wrongly shaped
/// coefficient list.
pub fn aft_ao2mo_7d(df: &Aftdf, mos: MoKpts<'_>, factor: f64) -> Result<Eri7d, PbcDfError> {
    ao2mo_7d_over(&df.cell, df.kpts.clone(), mos, factor, &mut |k| {
        aft_get_eri(df, k)
    })
}

/// `fft_ao2mo.ao2mo_7d` — `fft_ao2mo.py:344-428`.
///
/// # Errors
/// As [`aft_ao2mo_7d`].
pub fn fft_ao2mo_7d(df: &Fftdf, mos: MoKpts<'_>, factor: f64) -> Result<Eri7d, PbcDfError> {
    ao2mo_7d_over(&df.cell, df.kpts.clone(), mos, factor, &mut |k| {
        fft_get_eri(df, k)
    })
}

/// The shared `ao2mo_7d` driver: for every `(ki, kj, kk)` take
/// `kl = kconserv[ki, kj, kk]`, evaluate the AO block at that quadruple, and
/// transform it. Only the AO-block evaluation differs between the two
/// plane-wave builders.
fn ao2mo_7d_over(
    cell: &Cell,
    kpts: Vec<[f64; 3]>,
    mos: MoKpts<'_>,
    factor: f64,
    eri: &mut dyn FnMut([[f64; 3]; 4]) -> Result<CTensor, PbcDfError>,
) -> Result<Eri7d, PbcDfError> {
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    let bad = |m: String| {
        PbcDfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(m),
        ))
    };
    for (n, l) in mos.iter().enumerate() {
        if l.len() != nkpts {
            return Err(bad(format!(
                "ao2mo_7d: mo_coeff_kpts[{n}] has {} blocks for {nkpts} k-points",
                l.len()
            )));
        }
        if l.iter().any(|m| m.nao != nao || m.nmo != l[0].nmo) {
            return Err(bad(format!(
                "ao2mo_7d: mo_coeff_kpts[{n}] is ragged or has the wrong nao"
            )));
        }
    }
    let nmo = [mos[0][0].nmo, mos[1][0].nmo, mos[2][0].nmo, mos[3][0].nmo];
    let mut out = Eri7d {
        nkpts,
        nmo,
        data: CTensor::zeros(nkpts * nkpts * nkpts * nmo.iter().product::<usize>()),
    };
    let gamma = kpts.iter().all(|k| k.iter().all(|v| v.abs() < 1e-9));
    let real_out = gamma && mos.iter().all(|l| l.iter().all(MoCoeff::is_real));
    let kconserv = pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &kpts);

    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for kk in 0..nkpts {
                let kl = kconserv.get(ki, kj, kk) as usize;
                let ao = eri([kpts[ki], kpts[kj], kpts[kk], kpts[kl]])?;
                let m = transform_ao_eri(
                    &ao,
                    nao,
                    [&mos[0][ki], &mos[1][kj], &mos[2][kk], &mos[3][kl]],
                );
                let o = out.block_offset(ki, kj, kk);
                for p in 0..m.re.len() {
                    out.data.re[o + p] = m.re[p] * factor;
                    out.data.im[o + p] = if real_out { 0.0 } else { m.im[p] * factor };
                }
            }
        }
    }
    Ok(out)
}
