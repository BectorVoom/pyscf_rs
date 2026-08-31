//! `_get_jk` — the hybrid / range-separated J/K dispatch shared by every
//! periodic KS driver. Port of `pyscf/pbc/dft/krks.py:215-249`.
//!
//! ```text
//! (omega, alpha, hyb) = rsh_and_hybrid_coeff(xc)
//! pure          : vj = J,                       vk = 0
//! omega == 0    : vj = J,  vk = hyb·K
//! alpha == 0    : vj = J,  vk = hyb·K_SR(-omega)          (SR-only exchange)
//! hyb   == 0    : vj = J,  vk = alpha·K_LR(+omega)        (LR-only exchange)
//! otherwise     : vj = J,  vk = hyb·K + (alpha-hyb)·K_LR(+omega)
//! ```
//!
//! The exchange matrix is scaled HERE; the caller subtracts `0.5·vk` (RKS/KGKS)
//! or `vk` (KUKS, whose channels are not spin-degenerate).

use pyscf_algebra::CTensor;
use pyscf_pbc_df::{Fftdf, JkOpts, PeriodicDf};
use pyscf_pbc_gto::ExxDiv;
use pyscf_pbc_scf::types::{KDms, KMats};

use crate::error::PbcDftError;
use crate::xc::{err, is_hybrid_xc, rsh_and_hybrid_coeff};

/// What [`get_jk`] returns. `vk` is `None` for a pure functional.
#[derive(Debug, Clone)]
pub struct KsJk {
    /// `vj[iset][kband]`, `nao x nao` row-major. `None` when `with_j` was off.
    pub vj: Option<Vec<KMats>>,
    /// The SCALED exchange matrix, `None` for a pure functional.
    pub vk: Option<Vec<KMats>>,
}

/// `_get_jk(mf, cell, dm, hermi, kpts, kpts_band, with_j)` — `krks.py:215-249`.
///
/// # Errors
/// Propagates the density-fitting build and the XC-string parse.
#[allow(clippy::too_many_arguments)]
pub fn get_jk(
    df: &dyn PeriodicDf,
    xc_code: &str,
    dms: &KDms,
    hermi: i32,
    kpts: &[[f64; 3]],
    kpts_band: Option<&[[f64; 3]]>,
    exxdiv: Option<ExxDiv>,
    with_j: bool,
) -> Result<KsJk, PbcDftError> {
    let (omega, alpha, hyb) = rsh_and_hybrid_coeff(xc_code)?;
    let hybrid = is_hybrid_xc(xc_code)?;

    let build = |wj: bool, wk: bool, w: Option<f64>| -> Result<_, PbcDftError> {
        df.get_jk(
            dms,
            kpts,
            JkOpts {
                hermi,
                kpts_band,
                with_j: wj,
                with_k: wk,
                exxdiv,
                omega: w,
                // W-08: opt-in and OFF by default; `PYSCF_PBC_KK_SYMMETRY`
                // turns it on for a re-baselined run. It changes the last bits
                // of `vk`, so it must never become the silent default.
                kk_symmetry: JkOpts::kk_symmetry_default(),
            },
        )
        .map_err(|e| err(format!("periodic KS: density fitting failed: {e}")))
    };

    if !hybrid {
        // krks.py:222-228 — a pure functional never builds K, and `hermi == 2`
        // (an anti-Hermitian response density) has no Coulomb contribution.
        let vj = if hermi == 2 || !with_j {
            None
        } else {
            build(true, false, None)?.vj
        };
        return Ok(KsJk { vj, vk: None });
    }

    let (vj, mut vk) = if omega == 0.0 {
        // krks.py:230-232
        let r = build(with_j, true, None)?;
        (r.vj, r.vk.ok_or_else(|| err("periodic KS: no vk returned"))?)
    } else if alpha == 0.0 {
        // krks.py:233-236 — LR = 0, only short-range exchange.
        let k = build(false, true, Some(-omega))?
            .vk
            .ok_or_else(|| err("periodic KS: no vk returned"))?;
        let j = if with_j { build(true, false, None)?.vj } else { None };
        (j, k)
    } else if hyb == 0.0 {
        // krks.py:237-240 — SR = 0, only long-range exchange.
        let k = build(false, true, Some(omega))?
            .vk
            .ok_or_else(|| err("periodic KS: no vk returned"))?;
        let j = if with_j { build(true, false, None)?.vj } else { None };
        (j, k)
    } else {
        // krks.py:241-247 — both, with different ratios.
        let r = build(with_j, true, None)?;
        let mut k = r.vk.ok_or_else(|| err("periodic KS: no vk returned"))?;
        let klr = build(false, true, Some(omega))?
            .vk
            .ok_or_else(|| err("periodic KS: no vk returned"))?;
        scale(&mut k, hyb);
        let mut klr = klr;
        scale(&mut klr, alpha - hyb);
        add(&mut k, &klr);
        return Ok(KsJk {
            vj: r.vj,
            vk: Some(k),
        });
    };

    // The single-coefficient branches share one scaling.
    let coeff = if omega == 0.0 || alpha == 0.0 {
        hyb
    } else {
        alpha
    };
    scale(&mut vk, coeff);
    Ok(KsJk { vj, vk: Some(vk) })
}

fn scale(v: &mut [KMats], s: f64) {
    for set in v.iter_mut() {
        for m in set.iter_mut() {
            for i in 0..m.len() {
                m.re[i] *= s;
                m.im[i] *= s;
            }
        }
    }
}

fn add(a: &mut [KMats], b: &[KMats]) {
    for (sa, sb) in a.iter_mut().zip(b) {
        for (ma, mb) in sa.iter_mut().zip(sb) {
            for i in 0..ma.len() {
                ma.re[i] += mb.re[i];
                ma.im[i] += mb.im[i];
            }
        }
    }
}

/// `einsum('Kij,Kji->', dm, v)` summed over k and channel, times `1/nkpts` —
/// the shape every periodic KS energy term takes (`krks.py:96`).
pub fn trace_dm_v(dms: &KDms, v: &[KMats], nao: usize) -> (f64, f64) {
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    for (s, set) in dms.iter().enumerate() {
        for (k, d) in set.iter().enumerate() {
            let (r, i) = trace_ab(d, &v[s][k], nao);
            re += r;
            im += i;
        }
    }
    (re, im)
}

/// `Tr(A B)` over one row-major `n x n` pair.
pub fn trace_ab(a: &CTensor, b: &CTensor, n: usize) -> (f64, f64) {
    let mut sr = 0.0_f64;
    let mut si = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = (a.re[i * n + j], a.im[i * n + j]);
            let (br, bi) = (b.re[j * n + i], b.im[j * n + i]);
            sr += ar * br - ai * bi;
            si += ar * bi + ai * br;
        }
    }
    (sr, si)
}

/// `a += b` over a `(nset, nkpts)` stack.
pub fn add_assign(a: &mut [KMats], b: &[KMats]) {
    add(a, b);
}

/// `a -= s * b` over a `(nset, nkpts)` stack.
pub fn sub_scaled(a: &mut [KMats], s: f64, b: &[KMats]) {
    for (sa, sb) in a.iter_mut().zip(b) {
        for (ma, mb) in sa.iter_mut().zip(sb) {
            for i in 0..ma.len() {
                ma.re[i] -= s * mb.re[i];
                ma.im[i] -= s * mb.im[i];
            }
        }
    }
}
