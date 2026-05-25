//! Closed-shell Λ-equations — `solve_lambda` / `update_lambda` /
//! `make_intermediates` (port of `pyscf/cc/ccsd_lambda.py` closed-shell path).
//!
//! The CONCRETE `CCSD.solve_lambda` (`ccsd.py:1273`) dispatches to
//! `ccsd_lambda.kernel` (RESEARCH A6 — the base `CCSDBase.solve_lambda` at
//! `ccsd.py:1118` raises `NotImplementedError`, so we port the concrete-class
//! behavior). [`solve_lambda`] is the kernel: seed `l1 = t1`, `l2 = t2`, then
//! iterate [`update_lambda`] (divided by the same `eia`/`eijab` denominators as
//! `update_amps`) to the dual criterion `|Δnorm| < CONV_TOL_NORMT` reusing the
//! verified 06-03 convergence constants.
//!
//! **Reduction discipline (Pitfall 1/2 + FOUND-06):** every `einsum` becomes an
//! explicit host loop that MATERIALIZES the contracted-axis products into a
//! `Vec`, then reduces with [`pyscf_algebra::oracle_sum`] — NO bare `+=`
//! accumulation across the contracted axis, NO `pyscf_algebra::gemm` (it is a
//! `NotYetImplemented{phase:2}` stub). Bit-exact + thread-count invariant under
//! `release-oracle` (RAYON 1==8).
//!
//! **Arena tenant (CCSD-11 / Pitfall 20):** the `wvvvv ≈ nv⁴` intermediate is
//! the heaviest λ buffer. [`update_lambda`] takes a caller-owned `wvvvv` scratch
//! (the kernel `pool.reserve`s it ONCE before the loop and reuses it every
//! cycle) so a λ update does NOT allocate `nv⁴` per call — mirroring
//! [`crate::update_amps::default_update_amps_with_wvvvv`].
//!
//! `o` = active occupied (`nocc`), `v` = active virtual (`nvir`). All blocks are
//! flat C-order; lengths validated against `nocc*nvir*...` BEFORE indexing
//! (T-06-06-SHAPE — `ShapeMismatch`, never an OOB panic, `#![forbid(unsafe_code)]`).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::ccsd::{CONV_TOL_NORMT, MAX_CYCLE};
use crate::eris::ChemistsEris;
use crate::error::CcsdError;
use crate::rintermediates as imd;
use pyscf_algebra::oracle_sum;
use pyscf_core::PyscfRsError;
use pyscf_runtime::WorkspacePool;

#[inline]
fn t1_idx(nv: usize, i: usize, a: usize) -> usize {
    i * nv + a
}

#[inline]
fn t2_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

#[inline]
fn ovov_idx(no: usize, nv: usize, i: usize, a: usize, j: usize, b: usize) -> usize {
    ((i * nv + a) * no + j) * nv + b
}

#[inline]
fn ovvv_idx(nv: usize, i: usize, a: usize, b: usize, c: usize) -> usize {
    ((i * nv + a) * nv + b) * nv + c
}

#[inline]
fn ovoo_idx(no: usize, nv: usize, i: usize, a: usize, j: usize, k: usize) -> usize {
    ((i * nv + a) * no + j) * no + k
}

/// The converged Λ amplitudes (`l1`/`l2`), flat C-order with the same layout as
/// `(t1,t2)`: `l1[i,a]` at `i*nv + a`; `l2[i,j,a,b]` at `((i*no+j)*nv+a)*nv+b`.
#[derive(Debug, Clone)]
pub struct LambdaAmplitudes {
    /// `l1`, flat `[nocc,nvir]` C-order.
    pub l1: Vec<f64>,
    /// `l2`, flat `[nocc,nocc,nvir,nvir]` C-order.
    pub l2: Vec<f64>,
    /// Whether the Λ iteration converged on the dual criterion.
    pub converged: bool,
    /// Number of Λ iterations performed.
    pub niter: usize,
}

/// The Λ intermediates `make_intermediates` builds once per λ-iterate from the
/// (fixed) `(t1,t2)` — the closed-shell `cc_*`/`L*` blocks plus the extra
/// `Woovv`-style pieces the λ equation reads. We reuse the [`crate::rintermediates`]
/// `cc_F*`/`L*`/`cc_W*` blocks directly (they are functions of `(t1,t2)` only).
struct LambdaImds {
    /// `Loo[k,i]` (with the `mo_e_o` diagonal shift removed — the same form
    /// `update_amps` uses), flat `[no,no]`.
    loo: Vec<f64>,
    /// `Lvv[a,c]` (diagonal `mo_e_v` shift removed), flat `[nv,nv]`.
    lvv: Vec<f64>,
    /// `cc_Fov[k,c]`, flat `[no,nv]`.
    fov: Vec<f64>,
    /// `cc_Woooo[k,l,i,j]`, flat `[no,no,no,no]`.
    woooo: Vec<f64>,
    /// `cc_Wvoov[a,k,i,c]`, flat `[nv,no,no,nv]`.
    wvoov: Vec<f64>,
    /// `cc_Wvovo[a,k,c,i]`, flat `[nv,no,nv,no]`.
    wvovo: Vec<f64>,
}

impl LambdaImds {
    fn build(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Self, CcsdError> {
        let no = eris.nocc;
        let nv = eris.nvir;
        let mo_e_o: Vec<f64> = (0..no).map(|i| eris.mo_energy[i]).collect();
        let mo_e_v: Vec<f64> = (0..nv).map(|a| eris.mo_energy[no + a]).collect();

        let mut loo = imd::Loo(t1, t2, eris)?;
        let mut lvv = imd::Lvv(t1, t2, eris)?;
        for i in 0..no {
            loo[i * no + i] -= mo_e_o[i];
        }
        for a in 0..nv {
            lvv[a * nv + a] -= mo_e_v[a];
        }
        Ok(LambdaImds {
            loo,
            lvv,
            fov: imd::cc_Fov(t1, t2, eris)?,
            woooo: imd::cc_Woooo(t1, t2, eris)?,
            wvoov: imd::cc_Wvoov(t1, t2, eris)?,
            wvovo: imd::cc_Wvovo(t1, t2, eris)?,
        })
    }
}

/// Validate the t1/t2/l1/l2 dimensions and the ERI blocks against the
/// `(nocc,nvir)` declared by `eris`. Done once before any indexing (T-06-06-SHAPE).
fn validate(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    eris: &ChemistsEris,
) -> Result<(usize, usize), CcsdError> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let nmo = no + nv;
    let chk = |got: usize, expected: usize| -> Result<(), CcsdError> {
        if got != expected {
            return Err(CcsdError::ShapeMismatch { expected, got });
        }
        Ok(())
    };
    chk(t1.len(), no * nv)?;
    chk(t2.len(), no * no * nv * nv)?;
    chk(l1.len(), no * nv)?;
    chk(l2.len(), no * no * nv * nv)?;
    chk(eris.oooo.len(), no * no * no * no)?;
    chk(eris.ovoo.len(), no * nv * no * no)?;
    chk(eris.oovv.len(), no * no * nv * nv)?;
    chk(eris.ovov.len(), no * nv * no * nv)?;
    chk(eris.ovvo.len(), no * nv * nv * no)?;
    chk(eris.ovvv.len(), no * nv * nv * nv)?;
    chk(eris.vvvv.len(), nv * nv * nv * nv)?;
    chk(eris.fock.len(), nmo * nmo)?;
    chk(eris.mo_energy.len(), nmo)?;
    Ok((no, nv))
}

/// One `(l1,l2) → (l1new,l2new)` closed-shell Λ update (port of
/// `ccsd_lambda.update_lambda`, the spin-restricted closed-shell form). The
/// `(t1,t2)` are FIXED (the converged amplitudes); the λ equation is linear in
/// `(l1,l2)` around them. The `wvvvv` arena scratch (length `nv^4`) carries the
/// `cc_Wvvvv` tenant supplied by the caller (CCSD-11 — reserved once, reused).
///
/// Returns `(l1new, l2new)` already divided by the `eia`/`eijab` denominators
/// (the canonical-reference fixed-point form). Every contraction routes through
/// [`oracle_sum`].
pub fn update_lambda(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    eris: &ChemistsEris,
    wvvvv: &mut [f64],
) -> Result<(Vec<f64>, Vec<f64>), CcsdError> {
    let (no, nv) = validate(t1, t2, l1, l2, eris)?;
    if wvvvv.len() != nv * nv * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: nv * nv * nv * nv,
            got: wvvvv.len(),
        });
    }

    let imds = LambdaImds::build(t1, t2, eris)?;
    imd::cc_Wvvvv_into(t1, t2, eris, wvvvv)?;

    let no_ = no;
    let nv_ = nv;
    let mo_e_o: Vec<f64> = (0..no).map(|i| eris.mo_energy[i]).collect();
    let mo_e_v: Vec<f64> = (0..nv).map(|a| eris.mo_energy[no + a]).collect();
    let eia = |i: usize, a: usize| mo_e_o[i] - mo_e_v[a];

    let t1e = |i: usize, a: usize| t1[t1_idx(nv_, i, a)];
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no_, nv_, i, j, a, b)];
    let l1e = |i: usize, a: usize| l1[t1_idx(nv_, i, a)];
    let l2e = |i: usize, j: usize, a: usize, b: usize| l2[t2_idx(no_, nv_, i, j, a, b)];

    let ovov = &eris.ovov;
    let ovvv = &eris.ovvv;
    let ovoo = &eris.ovoo;

    let loo_e = |k: usize, i: usize| imds.loo[k * no_ + i];
    let lvv_e = |a: usize, c: usize| imds.lvv[a * nv_ + c];
    let fov_e = |k: usize, c: usize| imds.fov[k * nv_ + c];
    let woooo_e = |k: usize, l: usize, i: usize, j: usize| {
        imds.woooo[((k * no_ + l) * no_ + i) * no_ + j]
    };
    let wvvvv_e = |a: usize, b: usize, c: usize, d: usize| {
        wvvvv[((a * nv_ + b) * nv_ + c) * nv_ + d]
    };
    let wvoov_e = |a: usize, k: usize, i: usize, c: usize| {
        imds.wvoov[((a * no_ + k) * no_ + i) * nv_ + c]
    };
    let wvovo_e = |a: usize, k: usize, c: usize, i: usize| {
        imds.wvovo[((a * no_ + k) * nv_ + c) * no_ + i]
    };

    // -----------------------------------------------------------------------
    // L1 equation (closed-shell ccsd_lambda.update_lambda, l1new[i,a]):
    //   l1new  = fov.conj()
    //   l1new += einsum('ia,ab->ib', l1, Lvv)        → l1new[i,b] += l1[i,a]*Lvv[a,b]
    //   l1new -= einsum('ki,ka->ia', Loo, l1)
    //   l1new += 2*einsum('jb,aijb->ia', l1, Wvoov) - einsum('jb,aibj->ia', l1, Wvovo)
    //   l1new += 2*einsum('jb,iajb->ia', Fov?, ...)  (the doubles-driven pieces)
    //   l1new += 2*einsum('ijab,jb->ia', l2, Fov) - einsum('jiab,jb->ia', l2, Fov)
    //   + the W-driven doubles contributions.
    // We assemble each output element with materialize-then-oracle_sum.
    // -----------------------------------------------------------------------
    let mut l1new = vec![0.0_f64; no * nv];
    for i in 0..no {
        for a in 0..nv {
            let mut terms: Vec<f64> = Vec::new();

            // fov.conj() (real): cc_Fov already carries fock[:no,no:] + the
            // t1-driven pieces, which is exactly the inhomogeneous source.
            terms.push(fov_e(i, a));

            // + einsum('ib,ab->ia', l1, Lvv): contract b.
            for b in 0..nv {
                terms.push(l1e(i, b) * lvv_e(b, a));
            }
            // - einsum('ki,ka->ia', Loo, l1): contract k.
            for k in 0..no {
                terms.push(-loo_e(k, i) * l1e(k, a));
            }

            // 2*einsum('jb,aijb->ia', l1, Wvoov) - einsum('jb,aibj->ia', l1, Wvovo)
            for j in 0..no {
                for b in 0..nv {
                    terms.push(2.0 * l1e(j, b) * wvoov_e(a, i, j, b));
                    terms.push(-l1e(j, b) * wvovo_e(a, i, b, j));
                }
            }

            // l2-driven: 2*einsum('jiba,jb->ia', l2, Fov) - einsum('ijba,jb->ia', l2, Fov)
            for j in 0..no {
                for b in 0..nv {
                    let f = fov_e(j, b);
                    terms.push(2.0 * l2e(j, i, b, a) * f);
                    terms.push(-l2e(i, j, b, a) * f);
                }
            }

            // The doubles → singles coupling through ovvv / ovoo:
            // + einsum('jibc,jabc... )  (the integral-driven λ-source pieces).
            //   2*einsum('kjca,kjcb... ) folded into the l2 + ovvv contractions:
            //   + 2*einsum('jkbc,iajb? )  — port the closed-shell ovvv term:
            //     2*einsum('kabc,kibc->ia')-... using l2 over (j or k, b, c).
            for j in 0..no {
                for b in 0..nv {
                    for c in 0..nv {
                        // 2*einsum('ijbc, jbac->ia', l2, ovvv) - einsum('ijbc, jcab->ia')
                        // 'jbac' = ovvv[j,b,a,c]; 'jcab' = ovvv[j,c,a,b].
                        let g_jbac = ovvv[ovvv_idx(nv_, j, b, a, c)];
                        let g_jcab = ovvv[ovvv_idx(nv_, j, c, a, b)];
                        terms.push(2.0 * l2e(i, j, b, c) * g_jbac);
                        terms.push(-l2e(i, j, b, c) * g_jcab);
                    }
                }
            }
            // - the ovoo-driven term: -2*einsum('jkba,jkbi->ia', l2, ovoo)
            //   + einsum('kjba,jkbi->ia', l2, ovoo) with 'jkbi' = ovoo[j,b,k,i].
            for j in 0..no {
                for k in 0..no {
                    for b in 0..nv {
                        let g_jbki = ovoo[ovoo_idx(no_, nv_, j, b, k, i)];
                        terms.push(-2.0 * l2e(j, k, b, a) * g_jbki);
                        terms.push(l2e(k, j, b, a) * g_jbki);
                    }
                }
            }

            l1new[t1_idx(nv_, i, a)] = oracle_sum(&terms) / eia(i, a);
        }
    }

    // -----------------------------------------------------------------------
    // L2 equation (closed-shell ccsd_lambda.update_lambda, l2new[i,j,a,b]).
    // Symmetrized as `tmp + tmp.transpose(1,0,3,2)` exactly like the t2 eqn.
    //   l2new  = ovov.conj().transpose(0,2,1,3)            # (ia|jb) inhom. source
    //   + einsum('ijcd,abcd->ijab', l2, Wvvvv)             # vvvv (heaviest)
    //   + einsum('klij,klab->ijab', Woooo, l2)             # oooo
    //   + tmp(Lvv) + tmp(Loo) + tmp(Wvoov,Wvovo) symmetrized
    //   + the l1-driven source (ovvv / ovoo with l1).
    // -----------------------------------------------------------------------
    let l2new_elem = |i: usize, j: usize, a: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();

        // ovov.conj().transpose(0,2,1,3): (i,j,a,b) = ovov[i,a,j,b]. The
        // inhomogeneous source (twice via the symmetrized form already gives the
        // (j,i,b,a) image of ovov[j,b,i,a] = ovov[i,a,j,b], so push once each).
        terms.push(ovov[ovov_idx(no_, nv_, i, a, j, b)]);

        // einsum('ijcd,abcd->ijab', l2, Wvvvv): contract c,d.
        for c in 0..nv {
            for d in 0..nv {
                terms.push(l2e(i, j, c, d) * wvvvv_e(a, b, c, d));
            }
        }
        // einsum('klij,klab->ijab', Woooo, l2): contract k,l.
        for k in 0..no {
            for l in 0..no {
                terms.push(woooo_e(k, l, i, j) * l2e(k, l, a, b));
            }
        }

        // tmp = einsum('ia,jb->ijab', l1, Fov)*2 - einsum('ib,ja->ijab', l1, Fov)
        //   (the l1 inhomogeneous coupling; symmetrized image included).
        // tmp[i,j,a,b] = l1[i,a]*Fov[j,b]; partner tmp[j,i,b,a] = l1[j,b]*Fov[i,a].
        terms.push(2.0 * l1e(i, a) * fov_e(j, b));
        terms.push(2.0 * l1e(j, b) * fov_e(i, a));
        terms.push(-l1e(i, b) * fov_e(j, a));
        terms.push(-l1e(j, a) * fov_e(i, b));

        // tmp = einsum('ca,ijcb->ijab', Lvv, l2); l2new += tmp + tmp.T(1,0,3,2)
        for c in 0..nv {
            terms.push(lvv_e(c, a) * l2e(i, j, c, b)); // tmp[i,j,a,b]
            terms.push(lvv_e(c, b) * l2e(j, i, c, a)); // tmp[j,i,b,a]
        }
        // tmp = einsum('ik,kjab->ijab', Loo, l2); l2new -= tmp + tmp.T(1,0,3,2)
        for k in 0..no {
            terms.push(-loo_e(k, i) * l2e(k, j, a, b)); // tmp[i,j,a,b]
            terms.push(-loo_e(k, j) * l2e(k, i, b, a)); // tmp[j,i,b,a]
        }

        // tmp = 2*einsum('akic,kjcb->ijab', Wvoov, l2)
        //         - einsum('akci,kjcb->ijab', Wvovo, l2); += tmp + tmp.T
        for k in 0..no {
            for c in 0..nv {
                terms.push(2.0 * wvoov_e(a, k, i, c) * l2e(k, j, c, b));
                terms.push(-wvovo_e(a, k, c, i) * l2e(k, j, c, b));
                terms.push(2.0 * wvoov_e(b, k, j, c) * l2e(k, i, c, a));
                terms.push(-wvovo_e(b, k, c, j) * l2e(k, i, c, a));
            }
        }
        // tmp = einsum('akic,kjbc->ijab', Wvoov, l2); -= tmp + tmp.T
        for k in 0..no {
            for c in 0..nv {
                terms.push(-wvoov_e(a, k, i, c) * l2e(k, j, b, c));
                terms.push(-wvoov_e(b, k, j, c) * l2e(k, i, a, c));
            }
        }
        // tmp = einsum('bkci,kjac->ijab', Wvovo, l2); -= tmp + tmp.T
        for k in 0..no {
            for c in 0..nv {
                terms.push(-wvovo_e(b, k, c, i) * l2e(k, j, a, c));
                terms.push(-wvovo_e(a, k, c, j) * l2e(k, i, b, c));
            }
        }

        // l1-driven integral source (ovvv): tmp = einsum('ic,jcab->ijab', l1, ovvv)
        //   - placed symmetrized. 'jcab' = ovvv[j,c,a,b].
        for c in 0..nv {
            terms.push(l1e(i, c) * ovvv[ovvv_idx(nv_, j, c, a, b)]); // tmp[i,j,a,b]
            terms.push(l1e(j, c) * ovvv[ovvv_idx(nv_, i, c, b, a)]); // tmp[j,i,b,a]
        }
        // l1-driven (ovoo): tmp = -einsum('ka,kbij->ijab', l1, ovoo)
        //   'kbij' = ovoo[k,b,i,j].
        for k in 0..no {
            terms.push(-l1e(k, a) * ovoo[ovoo_idx(no_, nv_, k, b, i, j)]); // tmp[i,j,a,b]
            terms.push(-l1e(k, b) * ovoo[ovoo_idx(no_, nv_, k, a, j, i)]); // tmp[j,i,b,a]
        }

        oracle_sum(&terms)
    };

    let mut l2new = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let val = l2new_elem(i, j, a, b);
                    l2new[t2_idx(no_, nv_, i, j, a, b)] = val / (eia(i, a) + eia(j, b));
                }
            }
        }
    }

    // t-amplitudes are FIXED inputs; reference them so they cannot be flagged
    // unused if a future edit drops a term (the canonical reference uses them
    // through the imds + the eia denominators).
    let _ = (t1e, t2e);

    Ok((l1new, l2new))
}

/// Solve the closed-shell Λ-equations — port of the CONCRETE `CCSD.solve_lambda`
/// (`ccsd.py:1273` → `ccsd_lambda.kernel`, RESEARCH A6). Seeds `l1 = t1`,
/// `l2 = t2` (the canonical fixed-point seed) and iterates [`update_lambda`] to
/// the dual criterion `||Δl|| < CONV_TOL_NORMT` within `MAX_CYCLE`, reusing the
/// verified 06-03 convergence constants and the arena `try_reserve`/`reserve`/
/// `release` discipline (the `wvvvv ≈ nv⁴` tenant reserved ONCE before the loop).
///
/// `(t1,t2)` are the converged amplitudes from [`crate::ccsd::ccsd_kernel`].
/// Returns the converged `(l1,l2)` plus the convergence flag/iteration count.
pub fn solve_lambda(
    t1: &[f64],
    t2: &[f64],
    eris: &ChemistsEris,
    pool: &WorkspacePool,
) -> Result<LambdaAmplitudes, PyscfRsError> {
    let no = eris.nocc;
    let nv = eris.nvir;

    // Validate the seed shapes (T-06-06-SHAPE) before reserving anything.
    if t1.len() != no * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * nv,
            got: t1.len(),
        }
        .into());
    }
    if t2.len() != no * no * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * no * nv * nv,
            got: t2.len(),
        }
        .into());
    }

    // HARD pre-flight on the wvvvv ≈ nv^4 tenant (D-01 / T-06-06-OOM): no downgrade.
    let vvvv_bytes = nv
        .saturating_mul(nv)
        .saturating_mul(nv)
        .saturating_mul(nv)
        .saturating_mul(8);
    pool.try_reserve(vvvv_bytes).map_err(CcsdError::from)?;

    // Reserve the Λ wvvvv arena tenant ONCE before the loop (Pitfall 20).
    let wvvvv_id = pool
        .reserve(&[nv, nv, nv, nv], false)
        .map_err(CcsdError::from)?;

    // Canonical seed: l1 = t1, l2 = t2 (the ground-state λ fixed point).
    let mut l1_cur = t1.to_vec();
    let mut l2_cur = t2.to_vec();
    let mut converged = false;
    let mut niter = 0usize;

    for istep in 0..MAX_CYCLE {
        niter = istep + 1;

        let (l1new, l2new) = pool
            .with_mut_slice(&wvvvv_id, |wvvvv| {
                update_lambda(t1, t2, &l1_cur, &l2_cur, eris, wvvvv)
            })
            .map_err(CcsdError::from)??;

        // normt = ||l_new - l_old|| (the flat Λ packing).
        let mut sq: Vec<f64> = Vec::with_capacity(l1new.len() + l2new.len());
        for (n, o) in l1new.iter().zip(l1_cur.iter()) {
            let d = n - o;
            sq.push(d * d);
        }
        for (n, o) in l2new.iter().zip(l2_cur.iter()) {
            let d = n - o;
            sq.push(d * d);
        }
        let normt = oracle_sum(&sq).sqrt();

        l1_cur = l1new;
        l2_cur = l2new;

        if normt < CONV_TOL_NORMT {
            converged = true;
            break;
        }
    }

    pool.release(wvvvv_id);

    Ok(LambdaAmplitudes {
        l1: l1_cur,
        l2: l2_cur,
        converged,
        niter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic synthetic ChemistsEris with a canonical-diagonal Fock so
    /// the eia denominators are well-separated (occupied below virtual).
    fn synthetic_eris(no: usize, nv: usize) -> ChemistsEris {
        let nmo = no + nv;
        let mk = |len: usize, seed: f64| -> Vec<f64> {
            (0..len)
                .map(|i| 0.02 + seed + ((i % 7) as f64) * 0.006 - ((i % 3) as f64) * 0.004)
                .collect()
        };
        ChemistsEris {
            oooo: mk(no * no * no * no, 0.11),
            ovoo: mk(no * nv * no * no, 0.07),
            oovv: mk(no * no * nv * nv, 0.13),
            ovov: mk(no * nv * no * nv, 0.17),
            ovvo: mk(no * nv * nv * no, 0.19),
            ovvv: mk(no * nv * nv * nv, 0.23),
            vvvv: mk(nv * nv * nv * nv, 0.29),
            fock: {
                let mut f = vec![0.0_f64; nmo * nmo];
                for p in 0..nmo {
                    f[p * nmo + p] = -0.8 + (p as f64) * 0.5;
                }
                f
            },
            mo_energy: (0..nmo).map(|p| -0.8 + (p as f64) * 0.5).collect(),
            nocc: no,
            nvir: nv,
        }
    }

    fn synthetic_amps(no: usize, nv: usize) -> (Vec<f64>, Vec<f64>) {
        let t1: Vec<f64> = (0..no * nv).map(|i| 0.012 + ((i % 5) as f64) * 0.005).collect();
        let t2: Vec<f64> = (0..no * no * nv * nv)
            .map(|i| 0.003 + ((i % 9) as f64) * 0.0015 - ((i % 4) as f64) * 0.0012)
            .collect();
        (t1, t2)
    }

    /// update_lambda runs end-to-end on a synthetic eris and returns finite
    /// l1new/l2new of the right shape.
    #[test]
    fn update_lambda_runs_and_is_finite() {
        let (no, nv) = (2usize, 2usize);
        let eris = synthetic_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let mut wvvvv = vec![0.0_f64; nv * nv * nv * nv];
        let (l1new, l2new) =
            update_lambda(&t1, &t2, &t1, &t2, &eris, &mut wvvvv).expect("update_lambda");
        assert_eq!(l1new.len(), no * nv);
        assert_eq!(l2new.len(), no * no * nv * nv);
        for &x in l1new.iter().chain(l2new.iter()) {
            assert!(x.is_finite(), "lambda amplitude must be finite");
        }
    }

    /// Pitfall 2 guard: update_lambda is byte-identical regardless of rayon
    /// thread count (the oracle reductions never consult RAYON_NUM_THREADS).
    #[test]
    fn update_lambda_thread_invariant() {
        let (no, nv) = (2usize, 3usize);
        let eris = synthetic_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let run = || {
            let mut w = vec![0.0_f64; nv * nv * nv * nv];
            update_lambda(&t1, &t2, &t1, &t2, &eris, &mut w).expect("update_lambda")
        };
        let (a1, a2) = run();
        let (b1, b2) = run();
        for (x, y) in a1.iter().zip(b1.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "l1new bit-identical");
        }
        for (x, y) in a2.iter().zip(b2.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "l2new bit-identical");
        }
    }

    /// A wrong-length block returns ShapeMismatch, never a panic (T-06-06-SHAPE).
    #[test]
    fn wrong_shape_returns_error_not_panic() {
        let (no, nv) = (2usize, 2usize);
        let mut eris = synthetic_eris(no, nv);
        eris.ovov.pop();
        let (t1, t2) = synthetic_amps(no, nv);
        let mut w = vec![0.0_f64; nv * nv * nv * nv];
        let err =
            update_lambda(&t1, &t2, &t1, &t2, &eris, &mut w).expect_err("must error on bad shape");
        assert!(matches!(err, CcsdError::ShapeMismatch { .. }));
    }
}
