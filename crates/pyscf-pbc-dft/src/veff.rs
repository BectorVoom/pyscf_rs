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

use pyscf_algebra::{CTensor, oracle_sum};
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
        (
            r.vj,
            r.vk.ok_or_else(|| err("periodic KS: no vk returned"))?,
        )
    } else if alpha == 0.0 {
        // krks.py:233-236 — LR = 0, only short-range exchange.
        let k = build(false, true, Some(-omega))?
            .vk
            .ok_or_else(|| err("periodic KS: no vk returned"))?;
        let j = if with_j {
            build(true, false, None)?.vj
        } else {
            None
        };
        (j, k)
    } else if hyb == 0.0 {
        // krks.py:237-240 — SR = 0, only long-range exchange.
        let k = build(false, true, Some(omega))?
            .vk
            .ok_or_else(|| err("periodic KS: no vk returned"))?;
        let j = if with_j {
            build(true, false, None)?.vj
        } else {
            None
        };
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
///
/// # D-PBC-17 — the reduction is ORDERED
///
/// `ecoul` (`krks.rs:193`, `kuks.rs:248`) and the exchange half of `exc`
/// (`krks.rs:203`, `kuks.rs:258`) are both this function's `.0`, so it is on
/// the energy path and its reduction must be thread- and layout-invariant. The
/// per-`(channel, k)` partials are collected and reduced with the FOUND-06
/// pairwise tree instead of a running `+=`, and each partial is itself an
/// ordered [`trace_ab`]. The nesting is deliberate: two ordered reductions
/// compose into an ordered reduction, whereas an ordered inner sum folded by a
/// naive outer loop is only as good as the outer loop.
pub fn trace_dm_v(dms: &KDms, v: &[KMats], nao: usize) -> (f64, f64) {
    let n: usize = dms.iter().map(Vec::len).sum();
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for (s, set) in dms.iter().enumerate() {
        for (k, d) in set.iter().enumerate() {
            let (r, i) = trace_ab(d, &v[s][k], nao);
            re.push(r);
            im.push(i);
        }
    }
    (oracle_sum(&re), oracle_sum(&im))
}

/// [`trace_dm_v`] against ONE shared matrix stack instead of one per channel —
/// `einsum('nKij,Kji->', dm, v)` where `v` has no `n` axis.
///
/// # U-06 — why this exists
///
/// `kuks.py:87` computes `ecoul` from the SPIN-SUMMED Coulomb matrix, which is
/// shared by both channels. Expressing that through [`trace_dm_v`] required
/// materialising `vec![jtot.clone(), jtot.clone()]` — two full `nkpts x nao^2`
/// complex k-stacks cloned and dropped on every single `get_veff`, purely to
/// satisfy a `v[s][k]` index shape. This takes the shared stack directly.
///
/// **Bit-exact against the two-clone form**: the partials are pushed in the
/// same `(channel, k)` order, each is the same ordered [`trace_ab`] against
/// numerically identical operands, and the same [`oracle_sum`] reduces them.
pub fn trace_dm_v_shared(dms: &KDms, v: &KMats, nao: usize) -> (f64, f64) {
    let n: usize = dms.iter().map(Vec::len).sum();
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for set in dms.iter() {
        for (k, d) in set.iter().enumerate() {
            let (r, i) = trace_ab(d, &v[k], nao);
            re.push(r);
            im.push(i);
        }
    }
    (oracle_sum(&re), oracle_sum(&im))
}

/// `Re einsum('K,nKij,nKji->', weights, dm, v)` — [`trace_dm_v`] with a
/// PER-K-POINT WEIGHT instead of a uniform `1/nkpts`.
///
/// # Why this exists — plan item P-02
///
/// Under k-point symmetry the energy contractions weight each irreducible
/// k-point by its star multiplicity (`weights_ibz`), not by `1/nkpts`
/// (17-CONTEXT §3.5). `krks_ksymm.rs` expressed that with two hand-rolled
/// nests — `weighted_trace` and `weighted_trace_uks` — whose `nao^2` product
/// sum AND whose `nkpts_ibz` outer fold were both naive running sums, on the
/// quantities that become `ecoul`, the hybrid `exc` correction and
/// `energy_elec`'s `e1` for every k-symmetric KS driver. That is precisely
/// the shape D-PBC-17 forbids and U-03 closed for the non-symmetric drivers;
/// the symmetric twins were written afterwards and never got it.
///
/// # D-PBC-17 — the reduction is ORDERED, twice
///
/// Each `(set, k)` partial is `weights[k] * trace_ab(..).0`, itself an
/// ordered reduction over the `n^2` products in a fixed index order; the
/// partials are then reduced with the FOUND-06 pairwise tree rather than a
/// running `+=`. Two ordered reductions compose into an ordered reduction.
///
/// # Bit-parity against the loops this replaced
///
/// EXACT wherever `nao^2 <= PAIRWISE_CHUNK` (128, i.e. `nao <= 11`) and
/// `nset * nkpts_ibz <= PAIRWISE_CHUNK`, because `oracle_sum`'s base case is
/// a strict left-to-right fold from `0.0` — the same arithmetic the nests
/// did, term for term, in the same order. Every reference cell in this
/// repository has `nao = 8` and `nkpts_ibz <= 10`. Beyond that the tree
/// engages and the error bound improves from `O(n^2 · eps)` to
/// `O(log2(n^2) · eps)`.
///
/// Only the REAL part is returned: every consumer takes `.real` (upstream's
/// `krks_ksymm.py:76`, `:81`), and the imaginary residue is checked
/// separately by `coulomb_imag`.
pub fn weighted_trace_dm_v(dms: &KDms, v: &[KMats], weights: &[f64], nao: usize) -> f64 {
    let n: usize = dms.iter().map(Vec::len).sum();
    let mut parts = Vec::with_capacity(n);
    for (s, set) in dms.iter().enumerate() {
        for (k, d) in set.iter().enumerate() {
            parts.push(weights[k] * trace_ab(d, &v[s][k], nao).0);
        }
    }
    oracle_sum(&parts)
}

/// [`weighted_trace_dm_v`] against ONE shared matrix stack instead of one per
/// channel — `Re einsum('K,nKij,Kji->', weights, dm, v)` where `v` has no `n`
/// axis.
///
/// # Why this exists — plan item P-02
///
/// The same reason [`trace_dm_v_shared`] exists for the non-symmetric
/// drivers, and the same two call sites, one file over:
/// `KsymAdaptedKuks::get_veff_tagged` traced the spin-summed Coulomb matrix
/// by materialising `vec![jtot.clone(), jtot.clone()]`, and its
/// `energy_elec` traced the one-electron matrix by materialising
/// `vec![h1e.to_vec(), h1e.to_vec()]` — two full `nkpts_ibz x nao^2` complex
/// stacks cloned and dropped on every `get_veff` and every `energy_elec`,
/// purely to satisfy a `v[s][k]` index shape. U-06 deleted exactly these two
/// clones from `kuks.rs` and did not reach the k-symmetric twin.
///
/// **Bit-exact against the two-clone form**: the partials are pushed in the
/// same `(channel, k)` order, each is the same ordered [`trace_ab`] against
/// numerically identical operands, and the same [`oracle_sum`] reduces them.
pub fn weighted_trace_dm_v_shared(dms: &KDms, v: &KMats, weights: &[f64], nao: usize) -> f64 {
    let n: usize = dms.iter().map(Vec::len).sum();
    let mut parts = Vec::with_capacity(n);
    for set in dms.iter() {
        for (k, d) in set.iter().enumerate() {
            parts.push(weights[k] * trace_ab(d, &v[k], nao).0);
        }
    }
    oracle_sum(&parts)
}

/// `Tr(A B)` over one row-major `n x n` pair.
///
/// # D-PBC-17 — the reduction is ORDERED
///
/// The `n^2` products are materialised in a FIXED index order (`i`-major,
/// `j`-minor — the order the pre-existing loop accumulated in) and each plane
/// goes through [`oracle_sum`], so the recursion-tree shape depends only on
/// `n^2` and the fixed `PAIRWISE_CHUNK`.
///
/// * For `n^2 <= PAIRWISE_CHUNK` (`nao <= 11`, which covers the `nao = 8`
///   reference cells) `oracle_sum`'s base case is a strict left-to-right fold
///   from `0.0` — **bit-identical** to the loop this replaced.
/// * For `nao >= 12` the tree engages and the error bound improves from
///   `O(n^2 * eps)` to `O(log2(n^2) * eps)` — at `nao = 26` (`gth-dzvp`, 676
///   terms) a worst case of ~1.5e-13 relative becomes ~2e-15, against a KRKS
///   gate of 1e-11.
///
/// `oracle_zdot` is NOT the primitive here: `b` is read TRANSPOSED and neither
/// operand is conjugated, so this is not the `zdotc` contraction it implements.
/// This is the same routine, and the same reasoning, as
/// `pyscf_pbc_df::zlinalg::ztrace_ab`; the two are separate only because the
/// ALG-06 crate split puts them on opposite sides of it.
pub fn trace_ab(a: &CTensor, b: &CTensor, n: usize) -> (f64, f64) {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n * n);
    let mut tr = vec![0.0_f64; n * n];
    let mut ti = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = (a.re[i * n + j], a.im[i * n + j]);
            let (br, bi) = (b.re[j * n + i], b.im[j * n + i]);
            tr[i * n + j] = ar * br - ai * bi;
            ti[i * n + j] = ar * bi + ai * br;
        }
    }
    (oracle_sum(&tr), oracle_sum(&ti))
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
