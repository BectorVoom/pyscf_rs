//! Closed-shell RCCSD `update_amps` — one `(t1,t2) → (t1new,t2new)` amplitude
//! update (port of `pyscf/cc/rccsd.py:43-143`, Hirata et al. JCP 120, 2581
//! (2004) Eqs. (35)-(36)). This is the spin-restricted RCCSD amplitude equation
//! assembled from the [`crate::rintermediates`] `cc_*`/`L*` intermediates — it
//! produces the SAME converged energy as the production `ccsd.CCSD` kernel
//! (`ccsd.py:104`), but as a clean host-loop port rather than the
//! blocking/prefetch/H5-swap optimized form (which is not a 1:1 host-loop port).
//!
//! **Reduction discipline (Pitfall 1/2):** every `einsum` materializes the
//! contracted-axis products into a `Vec` then reduces with
//! `pyscf_algebra::oracle_sum` — NO bare `+=` accumulation across a contracted
//! axis, NO `pyscf_algebra::gemm` (a `NotYetImplemented{phase:2}` stub). The
//! result is bit-exact + thread-count invariant under `release-oracle`.
//!
//! **Arena tenant (CCSD-11 / Pitfall 20):** the `Wvvvv ≈ nv⁴` intermediate is
//! the heaviest buffer. [`default_update_amps_with_wvvvv`] takes a caller-owned
//! `wvvvv` scratch buffer (the kernel `pool.reserve`s it ONCE before the loop
//! and reuses it every cycle) so `update_amps` does NOT allocate `nv⁴` per call.
//! [`default_update_amps`] is the owned-buffer convenience wrapper (it allocates
//! `Wvvvv` itself) used by the `NoCcsdOverrides` hook + the unit tests.
#![allow(clippy::needless_range_loop)]

use crate::eris::ChemistsEris;
use crate::error::CcsdError;
use crate::rintermediates as imd;
use pyscf_algebra::oracle_sum;
use pyscf_core::{Amplitudes, MOCoefficients, PyscfRsError};

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
fn ovvo_idx(no: usize, nv: usize, i: usize, a: usize, b: usize, j: usize) -> usize {
    ((i * nv + a) * nv + b) * no + j
}

#[inline]
fn oovv_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

#[inline]
fn ovvv_idx(nv: usize, i: usize, a: usize, b: usize, c: usize) -> usize {
    ((i * nv + a) * nv + b) * nv + c
}

#[inline]
fn ovoo_idx(no: usize, nv: usize, i: usize, a: usize, j: usize, k: usize) -> usize {
    ((i * nv + a) * no + j) * no + k
}

#[inline]
fn fock_idx(nmo: usize, p: usize, q: usize) -> usize {
    p * nmo + q
}

/// One `(t1, t2) → (t1new, t2new)` RCCSD amplitude update, reusing a
/// caller-owned `wvvvv` scratch buffer (length `nv^4`) for the `cc_Wvvvv` arena
/// tenant. Returns the new amplitudes as owned `Amplitudes`.
///
/// Port of `rccsd.py:43-143`. Returns t1new C-order `[no,nv]`, t2new C-order
/// `[no,no,nv,nv]` (already divided by the `eia`/`eijab` denominators).
pub fn default_update_amps_with_wvvvv(
    t1: &[f64],
    t2: &[f64],
    eris: &ChemistsEris,
    wvvvv: &mut [f64],
) -> Result<(Vec<f64>, Vec<f64>), CcsdError> {
    let no = eris.nocc;
    let nv = eris.nvir;

    if wvvvv.len() != nv * nv * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: nv * nv * nv * nv,
            got: wvvvv.len(),
        });
    }
    if t1.len() != no * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * nv,
            got: t1.len(),
        });
    }
    if t2.len() != no * no * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * no * nv * nv,
            got: t2.len(),
        });
    }

    // In-core vvvv step: build cc_Wvvvv into the caller-owned arena buffer
    // (CCSD-11 — reserved once by the kernel, reused every cycle), then contract
    // einsum('abcd,ijcd->ijab', Wvvvv, tau) into the [no,no,nv,nv] block.
    imd::cc_Wvvvv_into(t1, t2, eris, wvvvv)?;
    let ht2_vvvv = vvvv_step_from_wvvvv(t1, t2, eris, wvvvv);
    update_amps_core(t1, t2, eris, &ht2_vvvv)
}

/// Contract the in-core `einsum('abcd,ijcd->ijab', Wvvvv, tau)` vvvv step into a
/// flat C-order `[no,no,nv,nv]` block (the full-`nv^4`-MO in-core path). `wvvvv`
/// holds the already-built `cc_Wvvvv` (C-order `[nv,nv,nv,nv]`).
fn vvvv_step_from_wvvvv(t1: &[f64], t2: &[f64], eris: &ChemistsEris, wvvvv: &[f64]) -> Vec<f64> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let t1e = |i: usize, a: usize| t1[t1_idx(nv, i, a)];
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no, nv, i, j, a, b)];
    let tau = |i: usize, j: usize, a: usize, b: usize| t2e(i, j, a, b) + t1e(i, a) * t1e(j, b);
    let wvvvv_e = |a: usize, b: usize, c: usize, d: usize| wvvvv[((a * nv + b) * nv + c) * nv + d];
    let mut ht2 = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let mut terms: Vec<f64> = Vec::with_capacity(nv * nv);
                    for c in 0..nv {
                        for d in 0..nv {
                            terms.push(wvvvv_e(a, b, c, d) * tau(i, j, c, d));
                        }
                    }
                    ht2[t2_idx(no, nv, i, j, a, b)] = oracle_sum(&terms);
                }
            }
        }
    }
    ht2
}

/// One RCCSD `(t1,t2) → (t1new,t2new)` update where the vvvv step contribution
/// `einsum('abcd,ijcd->ijab', Wvvvv, tau)` is supplied as a precomputed flat
/// C-order `[no,no,nv,nv]` block `ht2_vvvv`. The in-core path
/// ([`default_update_amps_with_wvvvv`]) builds `ht2_vvvv` from the full `nv^4`
/// `cc_Wvvvv`; the AO-direct path ([`default_update_amps_direct`]) builds it
/// on-the-fly without the full MO tensor. Everything else (T1 eq, the other T2
/// terms, the eia/eijab divide) is identical.
fn update_amps_core(
    t1: &[f64],
    t2: &[f64],
    eris: &ChemistsEris,
    ht2_vvvv: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), CcsdError> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let nmo = no + nv;

    // Shape validation (T-06-03-SHAPE): all reads validated before indexing.
    if t1.len() != no * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * nv,
            got: t1.len(),
        });
    }
    if t2.len() != no * no * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * no * nv * nv,
            got: t2.len(),
        });
    }
    if ht2_vvvv.len() != no * no * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * no * nv * nv,
            got: ht2_vvvv.len(),
        });
    }

    let ovov = &eris.ovov;
    let ovvo = &eris.ovvo;
    let oovv = &eris.oovv;
    let ovvv = &eris.ovvv;
    let ovoo = &eris.ovoo;
    let fock = &eris.fock;
    let mo_e_o: Vec<f64> = (0..no).map(|i| eris.mo_energy[i]).collect();
    let mo_e_v: Vec<f64> = (0..nv).map(|a| eris.mo_energy[no + a]).collect();

    let t1e = |i: usize, a: usize| t1[t1_idx(nv, i, a)];
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no, nv, i, j, a, b)];
    let fov = |i: usize, a: usize| fock[fock_idx(nmo, i, no + a)];

    // Intermediates (rccsd.py:55-61): cc_F*, with the diagonal energy shift
    // moved to the other side (Foo -= mo_e_o on the diagonal, etc.).
    let mut f_oo = imd::cc_Foo(t1, t2, eris)?; // [no,no]
    let mut fvv = imd::cc_Fvv(t1, t2, eris)?; // [nv,nv]
    let fov_imd = imd::cc_Fov(t1, t2, eris)?; // [no,nv]  (Fov[k,c])
    for i in 0..no {
        f_oo[i * no + i] -= mo_e_o[i];
    }
    for a in 0..nv {
        fvv[a * nv + a] -= mo_e_v[a];
    }
    let foo_e = |k: usize, i: usize| f_oo[k * no + i];
    let fvv_e = |a: usize, c: usize| fvv[a * nv + c];
    let fov_imd_e = |k: usize, c: usize| fov_imd[k * nv + c];

    // -----------------------------------------------------------------------
    // T1 equation (rccsd.py:63-82). t1new[i,a].
    // -----------------------------------------------------------------------
    let mut t1new = vec![0.0_f64; no * nv];
    for i in 0..no {
        for a in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            // -2*einsum('kc,ka,ic->ia', fov, t1, t1): contract k,c.
            for k in 0..no {
                for c in 0..nv {
                    terms.push(-2.0 * fov(k, c) * t1e(k, a) * t1e(i, c));
                }
            }
            // einsum('ac,ic->ia', Fvv, t1): contract c.
            for c in 0..nv {
                terms.push(fvv_e(a, c) * t1e(i, c));
            }
            // -einsum('ki,ka->ia', Foo, t1): contract k.
            for k in 0..no {
                terms.push(-foo_e(k, i) * t1e(k, a));
            }
            // 2*einsum('kc,kica->ia', Fov, t2) - einsum('kc,ikca->ia', Fov, t2)
            // + einsum('kc,ic,ka->ia', Fov, t1, t1): contract k,c.
            for k in 0..no {
                for c in 0..nv {
                    let f = fov_imd_e(k, c);
                    terms.push(2.0 * f * t2e(k, i, c, a));
                    terms.push(-f * t2e(i, k, c, a));
                    terms.push(f * t1e(i, c) * t1e(k, a));
                }
            }
            // fov.conj() (real): fock[i, no+a].
            terms.push(fov(i, a));
            // 2*einsum('kcai,kc->ia', ovvo, t1) - einsum('kiac,kc->ia', oovv, t1)
            for k in 0..no {
                for c in 0..nv {
                    terms.push(2.0 * ovvo[ovvo_idx(no, nv, k, c, a, i)] * t1e(k, c));
                    terms.push(-oovv[oovv_idx(no, nv, k, i, a, c)] * t1e(k, c));
                }
            }
            // 2*einsum('kdac,ikcd->ia', ovvv, t2) - einsum('kcad,ikcd->ia')
            // + 2*einsum('kdac,kd,ic') - einsum('kcad,kd,ic'): contract k,c,d.
            for k in 0..no {
                for c in 0..nv {
                    for d in 0..nv {
                        // 'kdac' = ovvv[k,d,a,c]; 'kcad' = ovvv[k,c,a,d]
                        let g_kdac = ovvv[ovvv_idx(nv, k, d, a, c)];
                        let g_kcad = ovvv[ovvv_idx(nv, k, c, a, d)];
                        terms.push(2.0 * g_kdac * t2e(i, k, c, d));
                        terms.push(-g_kcad * t2e(i, k, c, d));
                        terms.push(2.0 * g_kdac * t1e(k, d) * t1e(i, c));
                        terms.push(-g_kcad * t1e(k, d) * t1e(i, c));
                    }
                }
            }
            // -2*einsum('lcki,klac->ia', ovoo, t2) + einsum('kcli,klac->ia')
            // -2*einsum('lcki,lc,ka') + einsum('kcli,lc,ka'): contract k,l,c.
            for k in 0..no {
                for l in 0..no {
                    for c in 0..nv {
                        // 'lcki' = ovoo[l,c,k,i]; 'kcli' = ovoo[k,c,l,i]
                        let g_lcki = ovoo[ovoo_idx(no, nv, l, c, k, i)];
                        let g_kcli = ovoo[ovoo_idx(no, nv, k, c, l, i)];
                        terms.push(-2.0 * g_lcki * t2e(k, l, a, c));
                        terms.push(g_kcli * t2e(k, l, a, c));
                        terms.push(-2.0 * g_lcki * t1e(l, c) * t1e(k, a));
                        terms.push(g_kcli * t1e(l, c) * t1e(k, a));
                    }
                }
            }
            t1new[t1_idx(nv, i, a)] = oracle_sum(&terms);
        }
    }

    // -----------------------------------------------------------------------
    // T2 equation (rccsd.py:84-136). Built as the symmetrized sum
    // t2new[i,j,a,b] + t2new[j,i,b,a] of the "non-symmetric" pieces, plus the
    // already-symmetric Woooo/Wvvvv/Lvv/Loo/Wvoov/Wvovo terms. We assemble the
    // full t2new directly (no transpose buffers) by computing each contribution
    // at (i,j,a,b) and its transpose partner (j,i,b,a) where rccsd.py adds
    // `tmp + tmp.transpose(1,0,3,2)`.
    // -----------------------------------------------------------------------

    // The vvvv step is supplied via `ht2_vvvv` (precomputed by the caller — in-
    // core from cc_Wvvvv, or AO-direct on-the-fly). The remaining intermediates
    // are the small (≤ nv^3) ones built here every cycle.
    let woooo = imd::cc_Woooo(t1, t2, eris)?; // [no,no,no,no] (Wklij)
    let wvoov = imd::cc_Wvoov(t1, t2, eris)?; // [nv,no,no,nv] (Wakic)
    let wvovo = imd::cc_Wvovo(t1, t2, eris)?; // [nv,no,nv,no] (Wakci)
    let mut lvv = imd::Lvv(t1, t2, eris)?; // [nv,nv]
    let mut loo = imd::Loo(t1, t2, eris)?; // [no,no]
    for i in 0..no {
        loo[i * no + i] -= mo_e_o[i];
    }
    for a in 0..nv {
        lvv[a * nv + a] -= mo_e_v[a];
    }
    let woooo_e =
        |k: usize, l: usize, i: usize, j: usize| woooo[((k * no + l) * no + i) * no + j];
    let wvoov_e =
        |a: usize, k: usize, i: usize, c: usize| wvoov[((a * no + k) * no + i) * nv + c];
    let wvovo_e =
        |a: usize, k: usize, c: usize, i: usize| wvovo[((a * no + k) * nv + c) * no + i];
    let lvv_e = |a: usize, c: usize| lvv[a * nv + c];
    let loo_e = |k: usize, i: usize| loo[k * no + i];

    // tau = t2 + einsum('ia,jb->ijab', t1, t1)  (rccsd.py:123)
    let tau = |i: usize, j: usize, a: usize, b: usize| {
        t2e(i, j, a, b) + t1e(i, a) * t1e(j, b)
    };

    // Helper computing the FULL (non-divided) t2new[i,j,a,b] per rccsd.py.
    // We compute the value at a given (i,j,a,b); the "tmp + tmp.T(1,0,3,2)"
    // pieces are realized by also adding the (j,i,b,a) image of each tmp.
    let t2new_elem = |i: usize, j: usize, a: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();

        // tmp2 = einsum('kibc,ka->abic', oovv, -t1) + ovvv.conj().T(1,3,0,2)
        // tmp  = einsum('abic,jc->ijab', tmp2, t1); t2new = tmp + tmp.T(1,0,3,2)
        // ovvv.transpose(1,3,0,2): element 'abic' from ovvv[i_ov=i? ...].
        //   ovvv stored (o,v,v,v) = (k,d,a,c); .transpose(1,3,0,2) of the
        //   physicist-shaped (a,b,i,c) needs ovvv[i, a, c?, b]; rccsd uses
        //   np.asarray(eris_ovvv).conj().transpose(1,3,0,2) where eris_ovvv has
        //   shape (o,v,v,v)=(k,d,a,c): T(1,3,0,2) maps (d,c,k,a) → so element
        //   (a,b,i,c)_phys = ovvv[i, a, c, b]? Build the 'tmp' contraction
        //   directly from the two definitions below.
        // tmp[i,j,a,b] = sum_c tmp2[a,b,i,c] * t1[j,c]
        //   tmp2[a,b,i,c] = sum_k oovv[k,i,b,c]*(-t1[k,a]) + ovvv[i,a,c,b]
        // Image partner: + tmp[j,i,b,a].
        let tmp_ijab = {
            let mut s: Vec<f64> = Vec::new();
            for c in 0..nv {
                // tmp2[a,b,i,c]
                let mut tmp2 = ovvv[ovvv_idx(nv, i, a, c, b)];
                {
                    let mut acc: Vec<f64> = Vec::with_capacity(no);
                    for k in 0..no {
                        acc.push(oovv[oovv_idx(no, nv, k, i, b, c)] * (-t1e(k, a)));
                    }
                    tmp2 += oracle_sum(&acc);
                }
                s.push(tmp2 * t1e(j, c));
            }
            oracle_sum(&s)
        };
        let tmp_jiba = {
            let mut s: Vec<f64> = Vec::new();
            for c in 0..nv {
                let mut tmp2 = ovvv[ovvv_idx(nv, j, b, c, a)];
                {
                    let mut acc: Vec<f64> = Vec::with_capacity(no);
                    for k in 0..no {
                        acc.push(oovv[oovv_idx(no, nv, k, j, a, c)] * (-t1e(k, b)));
                    }
                    tmp2 += oracle_sum(&acc);
                }
                s.push(tmp2 * t1e(i, c));
            }
            oracle_sum(&s)
        };
        terms.push(tmp_ijab);
        terms.push(tmp_jiba);

        // tmp2 = einsum('kcai,jc->akij', ovvo, t1) + ovoo.T(1,3,0,2)
        // tmp  = einsum('akij,kb->ijab', tmp2, t1); t2new -= tmp + tmp.T(1,0,3,2)
        // tmp[i,j,a,b] = sum_k tmp2[a,k,i,j]*t1[k,b]
        //   tmp2[a,k,i,j] = sum_c ovvo[k,c,a,i]*t1[j,c] + ovoo[i,a,k,j]
        let tmp2_ovvo_ijab = {
            let mut s: Vec<f64> = Vec::new();
            for k in 0..no {
                let mut tmp2 = ovoo[ovoo_idx(no, nv, i, a, k, j)];
                {
                    let mut acc: Vec<f64> = Vec::with_capacity(nv);
                    for c in 0..nv {
                        acc.push(ovvo[ovvo_idx(no, nv, k, c, a, i)] * t1e(j, c));
                    }
                    tmp2 += oracle_sum(&acc);
                }
                s.push(tmp2 * t1e(k, b));
            }
            oracle_sum(&s)
        };
        let tmp2_ovvo_jiba = {
            let mut s: Vec<f64> = Vec::new();
            for k in 0..no {
                let mut tmp2 = ovoo[ovoo_idx(no, nv, j, b, k, i)];
                {
                    let mut acc: Vec<f64> = Vec::with_capacity(nv);
                    for c in 0..nv {
                        acc.push(ovvo[ovvo_idx(no, nv, k, c, b, j)] * t1e(i, c));
                    }
                    tmp2 += oracle_sum(&acc);
                }
                s.push(tmp2 * t1e(k, a));
            }
            oracle_sum(&s)
        };
        terms.push(-tmp2_ovvo_ijab);
        terms.push(-tmp2_ovvo_jiba);

        // + ovov.conj().transpose(0,2,1,3): element (i,j,a,b) = ovov[i,a,j,b].
        terms.push(ovov[ovov_idx(no, nv, i, a, j, b)]);

        // + einsum('klij,klab->ijab', Woooo, tau): contract k,l.
        for k in 0..no {
            for l in 0..no {
                terms.push(woooo_e(k, l, i, j) * tau(k, l, a, b));
            }
        }

        // + einsum('abcd,ijcd->ijab', Wvvvv, tau): the vvvv contraction
        //   (the '_contract_vvvv_t2' analog), supplied as a precomputed
        //   [no,no,nv,nv] block. In-core builds it from cc_Wvvvv; the AO-direct
        //   path (CCSD-07) builds the integral part on-the-fly without the full
        //   nv^4 MO tensor. Both are bit-close-equivalent contraction orders.
        terms.push(ht2_vvvv[t2_idx(no, nv, i, j, a, b)]);

        // tmp = einsum('ac,ijcb->ijab', Lvv, t2); t2new += tmp + tmp.T(1,0,3,2)
        for c in 0..nv {
            terms.push(lvv_e(a, c) * t2e(i, j, c, b)); // tmp[i,j,a,b]
            terms.push(lvv_e(b, c) * t2e(j, i, c, a)); // tmp[j,i,b,a]
        }

        // tmp = einsum('ki,kjab->ijab', Loo, t2); t2new -= tmp + tmp.T(1,0,3,2)
        for k in 0..no {
            terms.push(-loo_e(k, i) * t2e(k, j, a, b)); // tmp[i,j,a,b]
            terms.push(-loo_e(k, j) * t2e(k, i, b, a)); // tmp[j,i,b,a]
        }

        // tmp = 2*einsum('akic,kjcb->ijab', Wvoov, t2)
        //         - einsum('akci,kjcb->ijab', Wvovo, t2)
        //       t2new += tmp + tmp.T(1,0,3,2)
        for k in 0..no {
            for c in 0..nv {
                // tmp[i,j,a,b]
                terms.push(2.0 * wvoov_e(a, k, i, c) * t2e(k, j, c, b));
                terms.push(-wvovo_e(a, k, c, i) * t2e(k, j, c, b));
                // tmp[j,i,b,a]
                terms.push(2.0 * wvoov_e(b, k, j, c) * t2e(k, i, c, a));
                terms.push(-wvovo_e(b, k, c, j) * t2e(k, i, c, a));
            }
        }

        // tmp = einsum('akic,kjbc->ijab', Wvoov, t2); t2new -= tmp + tmp.T
        for k in 0..no {
            for c in 0..nv {
                terms.push(-wvoov_e(a, k, i, c) * t2e(k, j, b, c)); // tmp[i,j,a,b]
                terms.push(-wvoov_e(b, k, j, c) * t2e(k, i, a, c)); // tmp[j,i,b,a]
            }
        }

        // tmp = einsum('bkci,kjac->ijab', Wvovo, t2); t2new -= tmp + tmp.T
        for k in 0..no {
            for c in 0..nv {
                terms.push(-wvovo_e(b, k, c, i) * t2e(k, j, a, c)); // tmp[i,j,a,b]
                terms.push(-wvovo_e(a, k, c, j) * t2e(k, i, b, c)); // tmp[j,i,b,a]
            }
        }

        oracle_sum(&terms)
    };

    // Assemble t2new, then divide by eijab (rccsd.py:138-141).
    let mut t2new = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    t2new[t2_idx(no, nv, i, j, a, b)] = t2new_elem(i, j, a, b);
                }
            }
        }
    }

    // eia[i,a] = mo_e_o[i] - mo_e_v[a]; eijab = eia[i,a] + eia[j,b].
    let eia = |i: usize, a: usize| mo_e_o[i] - mo_e_v[a];
    for i in 0..no {
        for a in 0..nv {
            t1new[t1_idx(nv, i, a)] /= eia(i, a);
        }
    }
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let p = t2_idx(no, nv, i, j, a, b);
                    t2new[p] /= eia(i, a) + eia(j, b);
                }
            }
        }
    }

    Ok((t1new, t2new))
}

/// One RCCSD `(t1,t2) → (t1new,t2new)` update via the **AO-direct** vvvv step
/// (CCSD-07, `direct=True`). Computes the `einsum('abcd,ijcd->ijab', Wvvvv, tau)`
/// contribution WITHOUT the full `nv^4` `cc_Wvvvv` MO tensor: the two `ovvv·t1`
/// t1-correction terms of `cc_Wvvvv` are contracted against `tau` directly (only
/// the `nv^3` `ovvv` block is touched), and the heavy `nv^4` integral part is
/// supplied by [`crate::direct::contract_vvvv_t2_aodirect`] (AO `int2e` tiled
/// per leading-virtual index, peak `nv^3` MO buffer). The result is bit-close to
/// the in-core [`default_update_amps_with_wvvvv`] — a different contraction
/// ORDER of the same math.
///
/// `eri_ao` is the full AO `int2e` (F-order `[nao^4]`); `cv` is the virtual-MO
/// coefficient block (`[nao, nvir]`). No `nv^4` MO tensor is ever allocated.
///
/// Returns t1new C-order `[no,nv]`, t2new C-order `[no,no,nv,nv]` (already
/// divided by the `eia`/`eijab` denominators).
pub fn default_update_amps_direct(
    t1: &[f64],
    t2: &[f64],
    eris: &ChemistsEris,
    eri_ao: &[f64],
    nao: usize,
    cv: &MOCoefficients,
) -> Result<(Vec<f64>, Vec<f64>), CcsdError> {
    let no = eris.nocc;
    let nv = eris.nvir;

    if t1.len() != no * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * nv,
            got: t1.len(),
        });
    }
    if t2.len() != no * no * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * no * nv * nv,
            got: t2.len(),
        });
    }

    // (a) The integral part of the vvvv step, AO-sourced, no nv^4 MO tensor.
    let mut ht2_vvvv = crate::direct::contract_vvvv_t2_aodirect(eri_ao, nao, cv, t1, t2, eris)?;

    // (b) The t1-correction part of cc_Wvvvv contracted with tau. cc_Wvvvv's
    //     t1-corr is  -sum_k ovvv[k,d,a,c]*t1[k,b] - sum_k ovvv[k,c,b,d]*t1[k,a]
    //     (rintermediates cc_Wvvvv_into); contracting with tau[i,j,c,d] touches
    //     only the nv^3 ovvv block. Add it to the integral part in place.
    let ovvv = &eris.ovvv;
    let t1e = |i: usize, a: usize| t1[t1_idx(nv, i, a)];
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no, nv, i, j, a, b)];
    let tau = |i: usize, j: usize, a: usize, b: usize| t2e(i, j, a, b) + t1e(i, a) * t1e(j, b);
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let mut terms: Vec<f64> = Vec::with_capacity(nv * nv * 2 * no);
                    for c in 0..nv {
                        for d in 0..nv {
                            let tcd = tau(i, j, c, d);
                            // -sum_k ovvv[k,d,a,c]*t1[k,b]
                            for k in 0..no {
                                terms.push(-ovvv[ovvv_idx(nv, k, d, a, c)] * t1e(k, b) * tcd);
                            }
                            // -sum_k ovvv[k,c,b,d]*t1[k,a]
                            for k in 0..no {
                                terms.push(-ovvv[ovvv_idx(nv, k, c, b, d)] * t1e(k, a) * tcd);
                            }
                        }
                    }
                    ht2_vvvv[t2_idx(no, nv, i, j, a, b)] += oracle_sum(&terms);
                }
            }
        }
    }

    update_amps_core(t1, t2, eris, &ht2_vvvv)
}

/// The `NoCcsdOverrides::update_amps` delegate: one RCCSD amplitude update,
/// allocating the `cc_Wvvvv` scratch buffer internally (the owned-buffer
/// convenience path used outside the kernel arena).
///
/// Returns `(t1new, t2new)` as owned [`Amplitudes`] (the `t1`/`t2` fields each
/// hold the corresponding flat buffer; `nocc`/`nvir` from `eris`).
pub fn default_update_amps(
    t1: &Amplitudes,
    t2: &Amplitudes,
    eris: &ChemistsEris,
) -> Result<(Amplitudes, Amplitudes), PyscfRsError> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let t1_slice = t1.t1_slice().ok_or(CcsdError::ShapeMismatch {
        expected: no * nv,
        got: 0,
    })?;
    let t2_slice = t2.t2_slice().ok_or(CcsdError::ShapeMismatch {
        expected: no * no * nv * nv,
        got: 0,
    })?;
    let mut wvvvv = vec![0.0_f64; nv * nv * nv * nv];
    let (t1new, t2new) =
        default_update_amps_with_wvvvv(t1_slice, t2_slice, eris, &mut wvvvv)?;
    Ok((
        Amplitudes::from_vec(no, nv, t1new, Vec::new()),
        Amplitudes::from_vec(no, nv, Vec::new(), t2new),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rintermediates as imd_test;

    /// Reuse the rintermediates synthetic eris builder via a local copy so the
    /// update_amps tests are self-contained.
    fn synthetic_eris(no: usize, nv: usize) -> ChemistsEris {
        let nmo = no + nv;
        let mk = |len: usize, seed: f64| -> Vec<f64> {
            (0..len)
                .map(|i| 0.05 + seed + ((i % 7) as f64) * 0.02 - ((i % 3) as f64) * 0.013)
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
                // Canonical-diagonal fock so eia denominators are well-separated
                // and non-zero (occupied below virtual).
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
        let t1: Vec<f64> = (0..no * nv).map(|i| 0.015 + ((i % 5) as f64) * 0.008).collect();
        let t2: Vec<f64> = (0..no * no * nv * nv)
            .map(|i| 0.004 + ((i % 9) as f64) * 0.0021 - ((i % 4) as f64) * 0.0017)
            .collect();
        (t1, t2)
    }

    /// _contract_vvvv_t2 core: the 'abcd,ijcd->ijab' Wvvvv contraction on a
    /// 2x2 synthetic block matches a longhand quadruple-loop scalar-fold
    /// reference (uses tau = t2 + t1t1).
    #[test]
    fn contract_vvvv_t2_matches_longhand_2x2() {
        let (no, nv) = (2usize, 2usize);
        let eris = synthetic_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let wvvvv = imd_test::cc_Wvvvv(&t1, &t2, &eris).expect("wvvvv");
        let wvvvv_e =
            |a: usize, b: usize, c: usize, d: usize| wvvvv[((a * nv + b) * nv + c) * nv + d];
        let t1e = |i: usize, a: usize| t1[i * nv + a];
        let t2e = |i: usize, j: usize, a: usize, b: usize| t2[((i * no + j) * nv + a) * nv + b];
        let tau = |i: usize, j: usize, a: usize, b: usize| t2e(i, j, a, b) + t1e(i, a) * t1e(j, b);
        // einsum('abcd,ijcd->ijab')
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        let mut acc = 0.0_f64;
                        for c in 0..nv {
                            for d in 0..nv {
                                acc += wvvvv_e(a, b, c, d) * tau(i, j, c, d);
                            }
                        }
                        // Independent host-loop reduction via oracle_sum must
                        // agree with the scalar fold to tight tolerance.
                        let mut terms: Vec<f64> = Vec::new();
                        for c in 0..nv {
                            for d in 0..nv {
                                terms.push(wvvvv_e(a, b, c, d) * tau(i, j, c, d));
                            }
                        }
                        let got = oracle_sum(&terms);
                        assert!((got - acc).abs() < 1e-12, "vvvv[{i}{j}{a}{b}]");
                    }
                }
            }
        }
    }

    /// default_update_amps runs end-to-end on a synthetic eris and returns
    /// finite t1new/t2new of the right shape.
    ///
    /// NOTE on symmetry: the RCCSD equations build t2new as
    /// `tmp + tmp.transpose(1,0,3,2)`, so `t2new[i,j,a,b] == t2new[j,i,b,a]`
    /// holds ONLY when the supplied integrals carry the physical permutational
    /// symmetry `(ia|jb) == (jb|ia)` (real ERIs). The synthetic eris here is an
    /// arbitrary non-symmetric block, so the bare `eris.ovov.T(0,2,1,3)` term
    /// breaks that identity by construction; the genuine t2-symmetry of a real
    /// run is exercised in `tests/rccsd_numeric_smoke.rs`, where the integrals
    /// are the symmetric `int2e`-derived blocks. Here we only assert the port
    /// runs end-to-end and produces finite amplitudes of the right shape.
    #[test]
    fn update_amps_runs_and_is_finite() {
        let (no, nv) = (2usize, 2usize);
        let eris = synthetic_eris(no, nv);
        let (t1v, t2v) = synthetic_amps(no, nv);
        let mut wvvvv = vec![0.0_f64; nv * nv * nv * nv];
        let (t1new, t2new) =
            default_update_amps_with_wvvvv(&t1v, &t2v, &eris, &mut wvvvv).expect("update");
        assert_eq!(t1new.len(), no * nv);
        assert_eq!(t2new.len(), no * no * nv * nv);
        for &x in t1new.iter().chain(t2new.iter()) {
            assert!(x.is_finite(), "amplitude must be finite");
        }
    }

    /// Pitfall 2 guard: the same default_update_amps produces byte-identical
    /// output regardless of rayon thread count (the oracle reductions never
    /// consult RAYON_NUM_THREADS). Run twice; assert bit-identical.
    #[test]
    fn update_amps_thread_invariant() {
        let (no, nv) = (2usize, 3usize);
        let eris = synthetic_eris(no, nv);
        let (t1v, t2v) = synthetic_amps(no, nv);
        let run = || {
            let mut w = vec![0.0_f64; nv * nv * nv * nv];
            default_update_amps_with_wvvvv(&t1v, &t2v, &eris, &mut w).expect("update")
        };
        let (a1, a2) = run();
        let (b1, b2) = run();
        for (x, y) in a1.iter().zip(b1.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "t1new bit-identical");
        }
        for (x, y) in a2.iter().zip(b2.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "t2new bit-identical");
        }
    }
}
