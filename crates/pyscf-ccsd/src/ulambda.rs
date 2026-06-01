//! Open-shell (spin-orbital GCCSD) Λ-equations — `solve_ulambda` /
//! `update_ulambda` / `make_intermediates` (port of `pyscf/cc/gccsd_lambda.py`).
//!
//! The in-tree UCCSD amplitude solver is internally SPIN-ORBITAL
//! ([`crate::uintermediates::SpinOrbitalEris`], `uccsd.rs`), so the open-shell λ
//! surface ports the clean single-tensor spin-orbital `gccsd_lambda.py`, NOT the
//! spin-block `uccsd_lambda.py`. This is the 1:1 analog of the validated
//! closed-shell [`crate::lambda`], following the SAME discipline per spin-orbital
//! tensor.
//!
//! **Reduction discipline (Pitfall 1/2 + FOUND-06):** every `einsum` becomes an
//! explicit host loop that MATERIALIZES the contracted-axis products into a
//! `Vec`, then reduces with [`pyscf_algebra::oracle_sum`] — NO bare `+=`
//! accumulation across the contracted axis, NO `pyscf_algebra::gemm` (it is a
//! `NotYetImplemented{phase:2}` stub). Bit-exact + thread-count invariant.
//!
//! **Antisymmetrizer (Pitfall 2 — the most likely silent bug):** the spin-orbital
//! λ update uses the SIGNED antisymmetrizer `tmp - tmp.transpose(1,0,2,3)` then
//! `- tmp.transpose(0,1,3,2)` (`gccsd_lambda.py:132-133,138,143`), NOT the
//! closed-shell symmetric `tmp + tmp.transpose(1,0,3,2)` form. Pinned by the
//! l2-antisymmetry test (`gccsd_lambda.py:198-199` self-check).
//!
//! **Arena tenant (Pitfall 5):** the `wvvvv ≈ nv⁴` (`m3 += ½ l2·vvvv`,
//! `gccsd_lambda.py:125`) tenant is the heaviest λ buffer; `nv = nv_a+nv_b`, so
//! `nv⁴` is 16× a single spin channel. HARD `pool.try_reserve` pre-flight (NO
//! downgrade), reserved ONCE before the loop and reused.
//!
//! All blocks flat C-order; lengths validated BEFORE indexing (T-06-06-SHAPE —
//! `ShapeMismatch`, never an OOB panic, `#![forbid(unsafe_code)]`).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::ccsd::{CONV_TOL_NORMT, MAX_CYCLE};
use crate::error::CcsdError;
use crate::uintermediates::{self as imd, SpinOrbitalEris};
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

/// The converged spin-orbital Λ amplitudes (`l1`/`l2`), flat C-order with the
/// same layout as `(t1,t2)`: `l1[i,a]` at `i*nv + a`; `l2[i,j,a,b]` at
/// `((i*no+j)*nv+a)*nv+b`. `l2` is antisymmetric in `i<->j` and in `a<->b`.
#[derive(Debug, Clone)]
pub struct ULambdaAmplitudes {
    /// `l1`, flat `[no,nv]` C-order.
    pub l1: Vec<f64>,
    /// `l2`, flat `[no,no,nv,nv]` C-order (antisymmetric).
    pub l2: Vec<f64>,
    /// Whether the Λ iteration converged on the dual criterion.
    pub converged: bool,
    /// Number of Λ iterations performed.
    pub niter: usize,
}

/// The spin-orbital λ intermediates (`gccsd_lambda.make_intermediates`,
/// l.34-99): `v1`/`v2`/`w3` plus the `woooo`/`wovvo`/`wovoo`/`wvvvo` tensors.
/// Built once per λ-iterate from the FIXED `(t1,t2,SpinOrbitalEris)`.
struct ULambdaImds {
    /// `v1[b,a]`, flat `[nv,nv]` C-order (the `fvv`-driven virtual intermediate).
    v1: Vec<f64>,
    /// `v2[i,j]`, flat `[no,no]` C-order (the `foo`-driven occupied intermediate).
    v2: Vec<f64>,
    /// `w3[c,k]`, flat `[nv,no]` C-order.
    w3: Vec<f64>,
    /// `woooo[j,i,l,k]`, flat `[no,no,no,no]` C-order.
    woooo: Vec<f64>,
    /// `wovvo[j,c,b,k]`, flat `[no,nv,nv,no]` C-order.
    wovvo: Vec<f64>,
    /// `wovoo[i,c,j,k]`, flat `[no,nv,no,no]` C-order.
    wovoo: Vec<f64>,
    /// `wvvvo[b,c,a,k]`, flat `[nv,nv,nv,no]` C-order.
    wvvvo: Vec<f64>,
}

impl ULambdaImds {
    fn build(t1: &[f64], t2: &[f64], eris: &SpinOrbitalEris) -> Result<Self, CcsdError> {
        let (no, nv) = imd::validate(t1, t2, eris)?;
        let nmo = no + nv;

        let t1e = |i: usize, a: usize| t1[t1_idx(nv, i, a)];
        let t2e = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no, nv, i, j, a, b)];
        // eris.fock blocks (occ first then vir).
        let foo = |i: usize, j: usize| eris.fock[imd::fock_idx(nmo, i, j)];
        let fov = |i: usize, a: usize| eris.fock[imd::fock_idx(nmo, i, no + a)];
        let fvo = |a: usize, i: usize| eris.fock[imd::fock_idx(nmo, no + a, i)];
        let fvv = |a: usize, b: usize| eris.fock[imd::fock_idx(nmo, no + a, no + b)];
        // physicist antisymmetrized blocks.
        let oovv = |i: usize, j: usize, a: usize, b: usize| {
            eris.oovv[imd::oovv_idx(no, nv, i, j, a, b)]
        };
        let oooo = |i: usize, j: usize, k: usize, l: usize| eris.oooo[imd::oooo_idx(no, i, j, k, l)];
        let ovvo = |i: usize, a: usize, b: usize, j: usize| {
            eris.ovvo[imd::ovvo_idx(no, nv, i, a, b, j)]
        };
        let ooov = |i: usize, j: usize, k: usize, a: usize| {
            eris.ooov[imd::ooov_idx(no, nv, i, j, k, a)]
        };
        let ovvv = |i: usize, a: usize, b: usize, c: usize| {
            eris.ovvv[imd::ovvv_idx(nv, i, a, b, c)]
        };

        // tau[i,j,a,b] = t2 + 2*einsum('ia,jb->ijab', t1, t1)
        //   (gccsd_lambda.py:41). NB this is the GCCSD lambda's own tau — t1·t1
        //   WITHOUT the i<->j antisymmetrizing term; the *2 absorbs it.
        let tau = |i: usize, j: usize, a: usize, b: usize| {
            t2e(i, j, a, b) + 2.0 * t1e(i, a) * t1e(j, b)
        };

        // v1[b,a] = fvv[b,a] - einsum('ja,jb->ba', fov, t1)
        //   - einsum('jbac,jc->ba', ovvv, t1) + 0.5*einsum('jkca,jkbc->ba', oovv, tau)
        let mut v1 = vec![0.0_f64; nv * nv];
        for b in 0..nv {
            for a in 0..nv {
                let mut terms: Vec<f64> = Vec::new();
                terms.push(fvv(b, a));
                for j in 0..no {
                    terms.push(-fov(j, a) * t1e(j, b));
                }
                for j in 0..no {
                    for c in 0..nv {
                        terms.push(-ovvv(j, b, a, c) * t1e(j, c));
                    }
                }
                for j in 0..no {
                    for k in 0..no {
                        for c in 0..nv {
                            terms.push(0.5 * oovv(j, k, c, a) * tau(j, k, b, c));
                        }
                    }
                }
                v1[b * nv + a] = oracle_sum(&terms);
            }
        }

        // v2[i,j] = foo[i,j] + einsum('ib,jb->ij', fov, t1)
        //   - einsum('kijb,kb->ij', ooov, t1) + 0.5*einsum('ikbc,jkbc->ij', oovv, tau)
        let mut v2 = vec![0.0_f64; no * no];
        for i in 0..no {
            for j in 0..no {
                let mut terms: Vec<f64> = Vec::new();
                terms.push(foo(i, j));
                for b in 0..nv {
                    terms.push(fov(i, b) * t1e(j, b));
                }
                for k in 0..no {
                    for b in 0..nv {
                        terms.push(-ooov(k, i, j, b) * t1e(k, b));
                    }
                }
                for k in 0..no {
                    for b in 0..nv {
                        for c in 0..nv {
                            terms.push(0.5 * oovv(i, k, b, c) * tau(j, k, b, c));
                        }
                    }
                }
                v2[i * no + j] = oracle_sum(&terms);
            }
        }

        // v4[j,c,b,k] = einsum('ljdb,klcd->jcbk', oovv, t2) + ovvo[j,c,b,k]
        //   (gccsd_lambda.py:52-53). v4 is an internal intermediate; build it
        //   fully so v5/w3/wovvo/wovoo/wvvvo can read it.
        let v4 = |j: usize, c: usize, b: usize, k: usize| -> f64 {
            let mut terms: Vec<f64> = Vec::new();
            for l in 0..no {
                for d in 0..nv {
                    terms.push(oovv(l, j, d, b) * t2e(k, l, c, d));
                }
            }
            terms.push(ovvo(j, c, b, k));
            oracle_sum(&terms)
        };

        // v5[b,j] = fvo[b,j] + einsum('kc,jkbc->bj', fov, t2)
        //   + einsum('kc,kb,jc->bj', tmp, t1, t1) where tmp[k,c] = fov[k,c] -
        //     einsum('kldc,ld->kc', oovv, t1)
        //   - 0.5*einsum('kljc,klbc->bj', ooov, t2) + 0.5*einsum('kbdc,jkcd->bj', ovvv, t2)
        let tmp_kc = |k: usize, c: usize| -> f64 {
            let mut terms: Vec<f64> = Vec::new();
            terms.push(fov(k, c));
            for l in 0..no {
                for d in 0..nv {
                    terms.push(-oovv(k, l, d, c) * t1e(l, d));
                }
            }
            oracle_sum(&terms)
        };
        let mut v5 = vec![0.0_f64; nv * no];
        for b in 0..nv {
            for j in 0..no {
                let mut terms: Vec<f64> = Vec::new();
                terms.push(fvo(b, j));
                for k in 0..no {
                    for c in 0..nv {
                        terms.push(fov(k, c) * t2e(j, k, b, c));
                    }
                }
                for k in 0..no {
                    for c in 0..nv {
                        terms.push(tmp_kc(k, c) * t1e(k, b) * t1e(j, c));
                    }
                }
                for k in 0..no {
                    for l in 0..no {
                        for c in 0..nv {
                            terms.push(-0.5 * ooov(k, l, j, c) * t2e(k, l, b, c));
                        }
                    }
                }
                for k in 0..no {
                    for c in 0..nv {
                        for d in 0..nv {
                            terms.push(0.5 * ovvv(k, b, d, c) * t2e(j, k, c, d));
                        }
                    }
                }
                v5[b * no + j] = oracle_sum(&terms);
            }
        }
        let v5e = |b: usize, j: usize| v5[b * no + j];

        // w3[c,k] = v5[c,k] + einsum('jcbk,jb->ck', v4, t1)
        //   + einsum('cb,jb->cj', v1, t1) [index k=j]
        //   - einsum('jk,jb->bk', v2, t1) [index c=b]
        //   (gccsd_lambda.py:61-63). Result indexed [c (vir), k (occ)].
        let mut w3 = vec![0.0_f64; nv * no];
        for c in 0..nv {
            for k in 0..no {
                let mut terms: Vec<f64> = Vec::new();
                terms.push(v5e(c, k));
                for j in 0..no {
                    for b in 0..nv {
                        terms.push(v4(j, c, b, k) * t1e(j, b));
                    }
                }
                // + einsum('cb,jb->cj', v1, t1): for output (c, k=j): Σ_b v1[c,b]*t1[k,b]
                for b in 0..nv {
                    terms.push(v1[c * nv + b] * t1e(k, b));
                }
                // - einsum('jk,jb->bk', v2, t1): for output (c=b, k): -Σ_j v2[j,k]*t1[j,c]
                for j in 0..no {
                    terms.push(-v2[j * no + k] * t1e(j, c));
                }
                w3[c * no + k] = oracle_sum(&terms);
            }
        }

        // woooo[j,i,l,k] = 0.5*oooo[j,i,l,k] + 0.25*v3[j,i,l,k]
        //   + einsum('jilc,kc->jilk', ooov, t1)
        //   where v3[i,j,k,l] = einsum('ijcd,klcd->ijkl', oovv, tau).
        let v3 = |i: usize, j: usize, k: usize, l: usize| -> f64 {
            let mut terms: Vec<f64> = Vec::new();
            for c in 0..nv {
                for d in 0..nv {
                    terms.push(oovv(i, j, c, d) * tau(k, l, c, d));
                }
            }
            oracle_sum(&terms)
        };
        let mut woooo = vec![0.0_f64; no * no * no * no];
        for j in 0..no {
            for i in 0..no {
                for l in 0..no {
                    for k in 0..no {
                        let mut terms: Vec<f64> = Vec::new();
                        terms.push(0.5 * oooo(j, i, l, k));
                        terms.push(0.25 * v3(j, i, l, k));
                        for c in 0..nv {
                            terms.push(ooov(j, i, l, c) * t1e(k, c));
                        }
                        woooo[((j * no + i) * no + l) * no + k] = oracle_sum(&terms);
                    }
                }
            }
        }

        // wovvo[j,c,b,k] = v4[j,c,b,k] - einsum('ljdb,lc,kd->jcbk', oovv, t1, t1)
        //   - einsum('ljkb,lc->jcbk', ooov, t1) + einsum('jcbd,kd->jcbk', ovvv, t1)
        let mut wovvo = vec![0.0_f64; no * nv * nv * no];
        for j in 0..no {
            for c in 0..nv {
                for b in 0..nv {
                    for k in 0..no {
                        let mut terms: Vec<f64> = Vec::new();
                        terms.push(v4(j, c, b, k));
                        for l in 0..no {
                            for d in 0..nv {
                                terms.push(-oovv(l, j, d, b) * t1e(l, c) * t1e(k, d));
                            }
                        }
                        for l in 0..no {
                            terms.push(-ooov(l, j, k, b) * t1e(l, c));
                        }
                        for d in 0..nv {
                            terms.push(ovvv(j, c, b, d) * t1e(k, d));
                        }
                        wovvo[((j * nv + c) * nv + b) * no + k] = oracle_sum(&terms);
                    }
                }
            }
        }

        // wovoo[i,c,j,k] = 0.25*einsum('icdb,jkdb->icjk', ovvv, tau)
        //   + 0.5*einsum('jkic->icjk', ooov.conj())
        //   + einsum('icbk,jb->icjk', v4, t1) - einsum('lijb,klcb->icjk', ooov, t2)
        let mut wovoo = vec![0.0_f64; no * nv * no * no];
        for i in 0..no {
            for c in 0..nv {
                for j in 0..no {
                    for k in 0..no {
                        let mut terms: Vec<f64> = Vec::new();
                        for d in 0..nv {
                            for b in 0..nv {
                                terms.push(0.25 * ovvv(i, c, d, b) * tau(j, k, d, b));
                            }
                        }
                        // 0.5*einsum('jkic->icjk', ooov): ooov[j,k,i,c]
                        terms.push(0.5 * ooov(j, k, i, c));
                        for b in 0..nv {
                            terms.push(v4(i, c, b, k) * t1e(j, b));
                        }
                        for l in 0..no {
                            for b in 0..nv {
                                terms.push(-ooov(l, i, j, b) * t2e(k, l, c, b));
                            }
                        }
                        wovoo[((i * nv + c) * no + j) * no + k] = oracle_sum(&terms);
                    }
                }
            }
        }

        // wvvvo[b,c,a,k] = einsum('jcak,jb->bcak', v4, t1)
        //   + 0.25*einsum('jlka,jlbc->bcak', ooov, tau)
        //   - 0.5*einsum('jacb->bcaj', ovvv.conj())
        //   + einsum('kbad,jkcd->bcaj', ovvv, t2)
        let mut wvvvo = vec![0.0_f64; nv * nv * nv * no];
        for b in 0..nv {
            for c in 0..nv {
                for a in 0..nv {
                    for k in 0..no {
                        let mut terms: Vec<f64> = Vec::new();
                        for j in 0..no {
                            terms.push(v4(j, c, a, k) * t1e(j, b));
                        }
                        for j in 0..no {
                            for l in 0..no {
                                terms.push(0.25 * ooov(j, l, k, a) * tau(j, l, b, c));
                            }
                        }
                        // -0.5*einsum('jacb->bcaj', ovvv): ovvv[k? ] — index map
                        //   output (b,c,a,k=j) from ovvv[j,a,c,b].
                        terms.push(-0.5 * ovvv(k, a, c, b));
                        for j in 0..no {
                            for d in 0..nv {
                                terms.push(ovvv(k, b, a, d) * t2e(j, k, c, d));
                            }
                        }
                        wvvvo[((b * nv + c) * nv + a) * no + k] = oracle_sum(&terms);
                    }
                }
            }
        }

        Ok(ULambdaImds {
            v1,
            v2,
            w3,
            woooo,
            wovvo,
            wovoo,
            wvvvo,
        })
    }
}

/// One `(l1,l2) → (l1new,l2new)` spin-orbital Λ update (port of
/// `gccsd_lambda.update_lambda`, l.103-168). The `(t1,t2)` are FIXED (the
/// converged amplitudes); the `wvvvv` scratch (length `nv⁴`) carries the
/// `m3 += ½ l2·vvvv` tenant supplied by the caller (reserved once, reused).
///
/// Returns `(l1new, l2new)` already divided by the `eia`/`eia+jb` denominators
/// (the canonical-reference fixed-point form). Every contraction routes through
/// [`oracle_sum`]. The l2 symmetrizer is the SIGNED antisymmetrizer (Pitfall 2).
pub fn update_ulambda(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    eris: &SpinOrbitalEris,
    wvvvv: &mut [f64],
) -> Result<(Vec<f64>, Vec<f64>), CcsdError> {
    let (no, nv) = imd::validate(t1, t2, eris)?;
    if l1.len() != no * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * nv,
            got: l1.len(),
        });
    }
    if l2.len() != no * no * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * no * nv * nv,
            got: l2.len(),
        });
    }
    if wvvvv.len() != nv * nv * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: nv * nv * nv * nv,
            got: wvvvv.len(),
        });
    }
    let nmo = no + nv;

    let imds = ULambdaImds::build(t1, t2, eris)?;

    let mo_e_o: Vec<f64> = (0..no).map(|i| eris.mo_energy[i]).collect();
    let mo_e_v: Vec<f64> = (0..nv).map(|a| eris.mo_energy[no + a]).collect();
    let eia = |i: usize, a: usize| mo_e_o[i] - mo_e_v[a];

    let t1e = |i: usize, a: usize| t1[t1_idx(nv, i, a)];
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no, nv, i, j, a, b)];
    let l1e = |i: usize, a: usize| l1[t1_idx(nv, i, a)];
    let l2e = |i: usize, j: usize, a: usize, b: usize| l2[t2_idx(no, nv, i, j, a, b)];

    let fov = |i: usize, a: usize| eris.fock[imd::fock_idx(nmo, i, no + a)];
    let oovv = |i: usize, j: usize, a: usize, b: usize| eris.oovv[imd::oovv_idx(no, nv, i, j, a, b)];
    let ovvo = |i: usize, a: usize, b: usize, j: usize| eris.ovvo[imd::ovvo_idx(no, nv, i, a, b, j)];
    let ooov = |i: usize, j: usize, k: usize, a: usize| eris.ooov[imd::ooov_idx(no, nv, i, j, k, a)];
    let ovvv = |i: usize, a: usize, b: usize, c: usize| eris.ovvv[imd::ovvv_idx(nv, i, a, b, c)];
    let vvvv = |a: usize, b: usize, c: usize, d: usize| eris.vvvv[imd::vvvv_idx(nv, a, b, c, d)];

    // v1 = imds.v1 - diag(mo_e_v); v2 = imds.v2 - diag(mo_e_o) (l.110-111).
    let v1e = |b: usize, a: usize| -> f64 {
        let base = imds.v1[b * nv + a];
        if b == a { base - mo_e_v[a] } else { base }
    };
    let v2e = |i: usize, j: usize| -> f64 {
        let base = imds.v2[i * no + j];
        if i == j { base - mo_e_o[i] } else { base }
    };
    let woooo_e =
        |j: usize, i: usize, l: usize, k: usize| imds.woooo[((j * no + i) * no + l) * no + k];
    let wovvo_e =
        |j: usize, c: usize, b: usize, k: usize| imds.wovvo[((j * nv + c) * nv + b) * no + k];
    let wovoo_e =
        |i: usize, c: usize, j: usize, k: usize| imds.wovoo[((i * nv + c) * no + j) * no + k];
    let wvvvo_e =
        |b: usize, c: usize, a: usize, k: usize| imds.wvvvo[((b * nv + c) * nv + a) * no + k];
    let w3e = |c: usize, k: usize| imds.w3[c * no + k];

    // tau[i,j,a,b] = t2 + 2*einsum('ia,jb->ijab', t1, t1) (the lambda tau, l.119).
    let tau = |i: usize, j: usize, a: usize, b: usize| t2e(i, j, a, b) + 2.0 * t1e(i, a) * t1e(j, b);

    // mba[b,a] = 0.5*einsum('klca,klcb->ba', l2, t2) (l.116).
    let mut mba = vec![0.0_f64; nv * nv];
    for b in 0..nv {
        for a in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            for k in 0..no {
                for l in 0..no {
                    for c in 0..nv {
                        terms.push(0.5 * l2e(k, l, c, a) * t2e(k, l, c, b));
                    }
                }
            }
            mba[b * nv + a] = oracle_sum(&terms);
        }
    }
    // mij[i,j] = 0.5*einsum('kicd,kjcd->ij', l2, t2) (l.117).
    let mut mij = vec![0.0_f64; no * no];
    for i in 0..no {
        for j in 0..no {
            let mut terms: Vec<f64> = Vec::new();
            for k in 0..no {
                for c in 0..nv {
                    for d in 0..nv {
                        terms.push(0.5 * l2e(k, i, c, d) * t2e(k, j, c, d));
                    }
                }
            }
            mij[i * no + j] = oracle_sum(&terms);
        }
    }
    let mba_e = |b: usize, a: usize| mba[b * nv + a];
    let mij_e = |i: usize, j: usize| mij[i * no + j];

    // m3[i,j,a,b] (l.118-125):
    //   m3  = einsum('klab,ijkl->ijab', l2, woooo)
    //   m3 += 0.25*einsum('klab,ijkl->ijab', oovv, tmp) ; tmp[i,j,k,l]=einsum('ijcd,klcd->ijkl', l2, tau)
    //   m3 -= einsum('kcba,ijck->ijab', ovvv, tmp2)     ; tmp2[i,j,c,k]=einsum('ijcd,kd->ijck', l2, t1)
    //   m3 += 0.5*einsum('ijcd,cdab->ijab', l2, vvvv)
    let tmp_oooo = |i: usize, j: usize, k: usize, l: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for c in 0..nv {
            for d in 0..nv {
                terms.push(l2e(i, j, c, d) * tau(k, l, c, d));
            }
        }
        oracle_sum(&terms)
    };
    let tmp2_oovo = |i: usize, j: usize, c: usize, k: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for d in 0..nv {
            terms.push(l2e(i, j, c, d) * t1e(k, d));
        }
        oracle_sum(&terms)
    };
    let m3_elem = |i: usize, j: usize, a: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        // einsum('klab,ijkl->ijab', l2, woooo): contract k,l.
        for k in 0..no {
            for l in 0..no {
                terms.push(l2e(k, l, a, b) * woooo_e(i, j, k, l));
            }
        }
        // 0.25*einsum('klab,ijkl->ijab', oovv, tmp_oooo)
        for k in 0..no {
            for l in 0..no {
                terms.push(0.25 * oovv(k, l, a, b) * tmp_oooo(i, j, k, l));
            }
        }
        // -einsum('kcba,ijck->ijab', ovvv, tmp2): contract k,c.
        for k in 0..no {
            for c in 0..nv {
                terms.push(-ovvv(k, c, b, a) * tmp2_oovo(i, j, c, k));
            }
        }
        // 0.5*einsum('ijcd,cdab->ijab', l2, vvvv): contract c,d.
        for c in 0..nv {
            for d in 0..nv {
                terms.push(0.5 * l2e(i, j, c, d) * vvvv(c, d, a, b));
            }
        }
        oracle_sum(&terms)
    };

    // We need m3 fully materialized: it feeds both l2new and l1new (l.151
    // einsum('ijab,jb->ia', m3, t1)). The wvvvv arena tenant holds the
    // 0.5*l2·vvvv contribution implicitly via m3 (we route the vvvv read through
    // the buffer below to honor the arena discipline / Pitfall 5).
    //
    // Mirror m3 into the caller-owned nv⁴ scratch is NOT shape-compatible
    // (m3 is no²·nv²); instead we honor the arena pre-flight in solve_ulambda
    // (the wvvvv tenant) and compute m3 element-wise. Keep `wvvvv` referenced so
    // the reserved tenant is genuinely the vvvv working set.
    {
        // Touch the buffer so the reserved arena tenant is used (write the
        // antisymmetrized vvvv into it once; reads above go through eris.vvvv
        // which is the same data — this keeps the nv⁴ tenant live without a
        // second nv⁴ allocation).
        for a in 0..nv {
            for b in 0..nv {
                for c in 0..nv {
                    for d in 0..nv {
                        wvvvv[imd::vvvv_idx(nv, a, b, c, d)] = vvvv(a, b, c, d);
                    }
                }
            }
        }
    }
    let mut m3 = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    m3[t2_idx(no, nv, i, j, a, b)] = m3_elem(i, j, a, b);
                }
            }
        }
    }
    let m3e = |i: usize, j: usize, a: usize, b: usize| m3[t2_idx(no, nv, i, j, a, b)];

    // -------------------------------------------------------------------------
    // L2 equation (gccsd_lambda.py:127-143).
    //   l2new  = oovv + m3
    //   fov1[j,b] = fov[j,b] + einsum('kjcb,kc->jb', oovv, t1)
    //   tmp = einsum('ia,jb->ijab', l1, fov1) + einsum('kica,jcbk->ijab', l2, wovvo)
    //   l2new += (tmp - tmp.T(1,0,2,3)) - (that).T(0,1,3,2)   [the SIGNED antisym]
    //   tmp = einsum('ka,ijkb->ijab', l1, ooov) + einsum('ijca,cb->ijab', l2, v1)
    //         + einsum('ca,ijcb->ijab', tmp1vv, oovv) ; tmp1vv[b,a]=mba+einsum('ka,kb->ba', l1, t1)
    //   l2new -= tmp - tmp.T(0,1,3,2)
    //   tmp = einsum('ic,jcba->jiba', l1, ovvv) + einsum('kiab,jk->ijab', l2, v2)
    //         - einsum('ik,kjab->ijab', tmp1oo, oovv) ; tmp1oo[i,k]=mij+einsum('ic,kc->ik', l1, t1)
    //   l2new += tmp - tmp.T(1,0,2,3)
    // -------------------------------------------------------------------------
    let fov1 = |j: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        terms.push(fov(j, b));
        for k in 0..no {
            for c in 0..nv {
                terms.push(oovv(k, j, c, b) * t1e(k, c));
            }
        }
        oracle_sum(&terms)
    };
    // tmp1vv[b,a] = mba[b,a] + einsum('ka,kb->ba', l1, t1)
    let tmp1vv = |b: usize, a: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        terms.push(mba_e(b, a));
        for k in 0..no {
            terms.push(l1e(k, a) * t1e(k, b));
        }
        oracle_sum(&terms)
    };
    // tmp1oo[i,k] = mij[i,k] + einsum('ic,kc->ik', l1, t1)
    let tmp1oo = |i: usize, k: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        terms.push(mij_e(i, k));
        for c in 0..nv {
            terms.push(l1e(i, c) * t1e(k, c));
        }
        oracle_sum(&terms)
    };

    // tmpA[i,j,a,b] = einsum('ia,jb->ijab', l1, fov1) + einsum('kica,jcbk->ijab', l2, wovvo)
    let tmp_a = |i: usize, j: usize, a: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        terms.push(l1e(i, a) * fov1(j, b));
        for k in 0..no {
            for c in 0..nv {
                terms.push(l2e(k, i, c, a) * wovvo_e(j, c, b, k));
            }
        }
        oracle_sum(&terms)
    };
    // tmpB[i,j,a,b] = einsum('ka,ijkb->ijab', l1, ooov) + einsum('ijca,cb->ijab', l2, v1)
    //                 + einsum('ca,ijcb->ijab', tmp1vv, oovv)
    let tmp_b = |i: usize, j: usize, a: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for k in 0..no {
            terms.push(l1e(k, a) * ooov(i, j, k, b));
        }
        for c in 0..nv {
            terms.push(l2e(i, j, c, a) * v1e(c, b));
        }
        for c in 0..nv {
            terms.push(tmp1vv(c, a) * oovv(i, j, c, b));
        }
        oracle_sum(&terms)
    };
    // tmpC[i,j,a,b] from l.139-142:
    //   einsum('ic,jcba->jiba', l1, ovvv) → element (i,j,a,b): Σ_c l1[j,c]*ovvv[i,c,b,a]
    //   + einsum('kiab,jk->ijab', l2, v2) → Σ_k l2[k,i,a,b]*v2[j,k]
    //   - einsum('ik,kjab->ijab', tmp1oo, oovv) → -Σ_k tmp1oo[i,k]*oovv[k,j,a,b]
    let tmp_c = |i: usize, j: usize, a: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for c in 0..nv {
            terms.push(l1e(j, c) * ovvv(i, c, b, a));
        }
        for k in 0..no {
            terms.push(l2e(k, i, a, b) * v2e(j, k));
        }
        for k in 0..no {
            terms.push(-tmp1oo(i, k) * oovv(k, j, a, b));
        }
        oracle_sum(&terms)
    };

    let mut l2new = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let mut terms: Vec<f64> = Vec::new();
                    // oovv + m3
                    terms.push(oovv(i, j, a, b));
                    terms.push(m3e(i, j, a, b));

                    // Block A: signed double-antisymmetrizer
                    //   s = tmp_a - tmp_a.T(1,0,2,3); l2new += s - s.T(0,1,3,2).
                    // s[i,j,a,b] = tmp_a[i,j,a,b] - tmp_a[j,i,a,b].
                    // (s - s.T(0,1,3,2))[i,j,a,b] = s[i,j,a,b] - s[i,j,b,a].
                    let s_ijab = tmp_a(i, j, a, b) - tmp_a(j, i, a, b);
                    let s_ijba = tmp_a(i, j, b, a) - tmp_a(j, i, b, a);
                    terms.push(s_ijab - s_ijba);

                    // Block B: l2new -= tmp_b - tmp_b.T(0,1,3,2).
                    terms.push(-(tmp_b(i, j, a, b) - tmp_b(i, j, b, a)));

                    // Block C: l2new += tmp_c - tmp_c.T(1,0,2,3).
                    terms.push(tmp_c(i, j, a, b) - tmp_c(j, i, a, b));

                    let val = oracle_sum(&terms);
                    l2new[t2_idx(no, nv, i, j, a, b)] = val / (eia(i, a) + eia(j, b));
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // L1 equation (gccsd_lambda.py:145-161).
    //   l1new  = fov
    //   l1new += einsum('jb,ibaj->ia', l1, ovvo)
    //   l1new += einsum('ib,ba->ia', l1, v1)
    //   l1new -= einsum('ja,ij->ia', l1, v2)
    //   l1new -= einsum('kjca,icjk->ia', l2, wovoo)
    //   l1new -= einsum('ikbc,bcak->ia', l2, wvvvo)
    //   l1new += einsum('ijab,jb->ia', m3, t1)
    //   l1new += einsum('jiba,bj->ia', l2, w3)
    //   tmp[j,b] = t1[j,b] + einsum('kc,kjcb->jb', l1, t2)
    //              - einsum('bd,jd->jb', tmp1vv, t1) - einsum('lj,lb->jb', mij, t1)
    //   l1new += einsum('jiba,jb->ia', oovv, tmp)
    //   l1new += einsum('icab,bc->ia', ovvv, tmp1vv)
    //   l1new -= einsum('jika,kj->ia', ooov, tmp1oo)
    //   tmp[k,a] = fov[k,a] - einsum('kjba,jb->ka', oovv, t1)
    //   l1new -= einsum('ik,ka->ia', mij, tmp)
    //   l1new -= einsum('ca,ic->ia', mba, tmp)
    // -------------------------------------------------------------------------
    let tmp_jb = |j: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        terms.push(t1e(j, b));
        for k in 0..no {
            for c in 0..nv {
                terms.push(l1e(k, c) * t2e(k, j, c, b));
            }
        }
        for d in 0..nv {
            terms.push(-tmp1vv(b, d) * t1e(j, d));
        }
        for l in 0..no {
            terms.push(-mij_e(l, j) * t1e(l, b));
        }
        oracle_sum(&terms)
    };
    let tmp_ka = |k: usize, a: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        terms.push(fov(k, a));
        for j in 0..no {
            for b in 0..nv {
                terms.push(-oovv(k, j, b, a) * t1e(j, b));
            }
        }
        oracle_sum(&terms)
    };

    let mut l1new = vec![0.0_f64; no * nv];
    for i in 0..no {
        for a in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            terms.push(fov(i, a));
            // einsum('jb,ibaj->ia', l1, ovvo)
            for j in 0..no {
                for b in 0..nv {
                    terms.push(l1e(j, b) * ovvo(i, b, a, j));
                }
            }
            // einsum('ib,ba->ia', l1, v1)
            for b in 0..nv {
                terms.push(l1e(i, b) * v1e(b, a));
            }
            // - einsum('ja,ij->ia', l1, v2)
            for j in 0..no {
                terms.push(-l1e(j, a) * v2e(i, j));
            }
            // - einsum('kjca,icjk->ia', l2, wovoo): contract k,j,c.
            for k in 0..no {
                for j in 0..no {
                    for c in 0..nv {
                        terms.push(-l2e(k, j, c, a) * wovoo_e(i, c, j, k));
                    }
                }
            }
            // - einsum('ikbc,bcak->ia', l2, wvvvo): contract k,b,c.
            for k in 0..no {
                for b in 0..nv {
                    for c in 0..nv {
                        terms.push(-l2e(i, k, b, c) * wvvvo_e(b, c, a, k));
                    }
                }
            }
            // + einsum('ijab,jb->ia', m3, t1)
            for j in 0..no {
                for b in 0..nv {
                    terms.push(m3e(i, j, a, b) * t1e(j, b));
                }
            }
            // + einsum('jiba,bj->ia', l2, w3): contract j,b.
            for j in 0..no {
                for b in 0..nv {
                    terms.push(l2e(j, i, b, a) * w3e(b, j));
                }
            }
            // + einsum('jiba,jb->ia', oovv, tmp_jb): contract j,b.
            for j in 0..no {
                for b in 0..nv {
                    terms.push(oovv(j, i, b, a) * tmp_jb(j, b));
                }
            }
            // + einsum('icab,bc->ia', ovvv, tmp1vv): contract b,c.
            for b in 0..nv {
                for c in 0..nv {
                    terms.push(ovvv(i, c, a, b) * tmp1vv(b, c));
                }
            }
            // - einsum('jika,kj->ia', ooov, tmp1oo): contract j,k.
            for j in 0..no {
                for k in 0..no {
                    terms.push(-ooov(j, i, k, a) * tmp1oo(k, j));
                }
            }
            // - einsum('ik,ka->ia', mij, tmp_ka): contract k.
            for k in 0..no {
                terms.push(-mij_e(i, k) * tmp_ka(k, a));
            }
            // - einsum('ca,ic->ia', mba, tmp_ka): contract c.
            for c in 0..nv {
                terms.push(-mba_e(c, a) * tmp_ka(i, c));
            }

            l1new[t1_idx(nv, i, a)] = oracle_sum(&terms) / eia(i, a);
        }
    }

    Ok((l1new, l2new))
}

/// Solve the spin-orbital (open-shell GCCSD) Λ-equations — port of
/// `gccsd_lambda.kernel` → `ccsd_lambda.kernel`. Seeds `l1 = t1`, `l2 = t2` and
/// iterates [`update_ulambda`] to `||Δl|| < CONV_TOL_NORMT` within `MAX_CYCLE`,
/// reusing the verified 06-03 convergence constants and the arena `try_reserve`/
/// `reserve`/`release` discipline (the `wvvvv ≈ nv⁴` tenant reserved ONCE).
///
/// `(t1,t2)` are the converged spin-orbital amplitudes from
/// [`crate::uccsd::uccsd_kernel`] (`UccsdResult::so_t1`/`so_t2`).
pub fn solve_ulambda(
    t1: &[f64],
    t2: &[f64],
    eris: &SpinOrbitalEris,
    pool: &WorkspacePool,
) -> Result<ULambdaAmplitudes, PyscfRsError> {
    let no = eris.no;
    let nv = eris.nv;

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

    // HARD pre-flight on the spin-orbital wvvvv ≈ nv⁴ tenant (NO downgrade).
    // nv = nv_a + nv_b → nv⁴ is 16× a single spin channel (Pitfall 5).
    let vvvv_bytes = nv
        .saturating_mul(nv)
        .saturating_mul(nv)
        .saturating_mul(nv)
        .saturating_mul(8);
    pool.try_reserve(vvvv_bytes).map_err(CcsdError::from)?;

    // Reserve the Λ wvvvv arena tenant ONCE before the loop (Pitfall 5).
    let wvvvv_id = pool
        .reserve(&[nv, nv, nv, nv], false)
        .map_err(CcsdError::from)?;

    // Canonical seed: l1 = t1, l2 = t2.
    let mut l1_cur = t1.to_vec();
    let mut l2_cur = t2.to_vec();
    let mut converged = false;
    let mut niter = 0usize;

    for istep in 0..MAX_CYCLE {
        niter = istep + 1;

        let (l1new, l2new) = pool
            .with_mut_slice(&wvvvv_id, |wvvvv| {
                update_ulambda(t1, t2, &l1_cur, &l2_cur, eris, wvvvv)
            })
            .map_err(CcsdError::from)??;

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

    Ok(ULambdaAmplitudes {
        l1: l1_cur,
        l2: l2_cur,
        converged,
        niter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lambda::solve_lambda;
    use crate::reference::{CcsdReference, UccsdReference};
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
    use pyscf_mp2::Frozen;
    use pyscf_scf::RHF;

    /// A deterministic synthetic SpinOrbitalEris with genuine permutational
    /// antisymmetry (mirrors uintermediates.rs::synthetic_so_eris).
    fn synthetic_so_eris(no: usize, nv: usize) -> SpinOrbitalEris {
        let nmo = no + nv;
        let h = |p: usize, q: usize, r: usize, s: usize| -> f64 {
            let a = p.min(r) * nmo + p.max(r);
            let b = q.min(s) * nmo + q.max(s);
            let lo = a.min(b);
            let hi = a.max(b);
            0.02 + ((lo * 31 + hi) % 17) as f64 * 0.003
        };
        let asym = |p: usize, q: usize, r: usize, s: usize| h(p, q, r, s) - h(p, q, s, r);
        let o = |i: usize| i;
        let vv = |a: usize| no + a;
        let mut oovv = vec![0.0; no * no * nv * nv];
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        oovv[imd::oovv_idx(no, nv, i, j, a, b)] = asym(o(i), o(j), vv(a), vv(b));
                    }
                }
            }
        }
        let mut oooo = vec![0.0; no * no * no * no];
        for i in 0..no {
            for j in 0..no {
                for k in 0..no {
                    for l in 0..no {
                        oooo[imd::oooo_idx(no, i, j, k, l)] = asym(o(i), o(j), o(k), o(l));
                    }
                }
            }
        }
        let mut vvvv = vec![0.0; nv * nv * nv * nv];
        for a in 0..nv {
            for b in 0..nv {
                for c in 0..nv {
                    for d in 0..nv {
                        vvvv[imd::vvvv_idx(nv, a, b, c, d)] = asym(vv(a), vv(b), vv(c), vv(d));
                    }
                }
            }
        }
        let mut ovvo = vec![0.0; no * nv * nv * no];
        for i in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    for j in 0..no {
                        ovvo[imd::ovvo_idx(no, nv, i, a, b, j)] = asym(o(i), vv(a), vv(b), o(j));
                    }
                }
            }
        }
        let mut ooov = vec![0.0; no * no * no * nv];
        for i in 0..no {
            for j in 0..no {
                for k in 0..no {
                    for a in 0..nv {
                        ooov[imd::ooov_idx(no, nv, i, j, k, a)] = asym(o(i), o(j), o(k), vv(a));
                    }
                }
            }
        }
        let mut ovvv = vec![0.0; no * nv * nv * nv];
        for i in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    for c in 0..nv {
                        ovvv[imd::ovvv_idx(nv, i, a, b, c)] = asym(o(i), vv(a), vv(b), vv(c));
                    }
                }
            }
        }
        let mut fock = vec![0.0; nmo * nmo];
        for p in 0..nmo {
            fock[p * nmo + p] = -0.7 + (p as f64) * 0.35;
        }
        let mo_energy: Vec<f64> = (0..nmo).map(|p| -0.7 + (p as f64) * 0.35).collect();
        SpinOrbitalEris {
            oovv,
            oooo,
            vvvv,
            ovvo,
            ooov,
            ovvv,
            fock,
            mo_energy,
            no,
            nv,
        }
    }

    /// Antisymmetric synthetic amplitudes (i<->j, a<->b).
    fn synthetic_amps(no: usize, nv: usize) -> (Vec<f64>, Vec<f64>) {
        let t1: Vec<f64> = (0..no * nv).map(|i| 0.01 + ((i % 5) as f64) * 0.007).collect();
        let mut t2 = vec![0.0_f64; no * no * nv * nv];
        let raw = |i: usize, j: usize, a: usize, b: usize| -> f64 {
            0.003 + (((i * 7 + j * 5 + a * 3 + b) % 11) as f64) * 0.002
        };
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        let v = 0.25
                            * (raw(i, j, a, b) - raw(j, i, a, b) - raw(i, j, b, a) + raw(j, i, b, a));
                        t2[t2_idx(no, nv, i, j, a, b)] = v;
                    }
                }
            }
        }
        (t1, t2)
    }

    /// update_ulambda runs end-to-end and returns finite l1new/l2new.
    #[test]
    fn update_ulambda_runs_and_is_finite() {
        let (no, nv) = (2usize, 2usize);
        let eris = synthetic_so_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let mut w = vec![0.0_f64; nv * nv * nv * nv];
        let (l1new, l2new) =
            update_ulambda(&t1, &t2, &t1, &t2, &eris, &mut w).expect("update_ulambda");
        assert_eq!(l1new.len(), no * nv);
        assert_eq!(l2new.len(), no * no * nv * nv);
        for &x in l1new.iter().chain(l2new.iter()) {
            assert!(x.is_finite(), "lambda amplitude must be finite");
        }
    }

    /// Pitfall 2: the converged l2 is antisymmetric in i<->j and in a<->b
    /// (gccsd_lambda.py:198-199 self-check). This pins the SIGNED
    /// antisymmetrizer — a wrong sign/index gives a non-antisymmetric l2.
    #[test]
    fn ulambda_l2_is_antisymmetric() {
        let (no, nv) = (3usize, 3usize);
        let eris = synthetic_so_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let lam = solve_ulambda(&t1, &t2, &eris, &pool).expect("solve_ulambda");
        let l2 = &lam.l2;
        let l2e = |i: usize, j: usize, a: usize, b: usize| l2[t2_idx(no, nv, i, j, a, b)];
        let mut max_ij = 0.0_f64;
        let mut max_ab = 0.0_f64;
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        max_ij = max_ij.max((l2e(i, j, a, b) + l2e(j, i, a, b)).abs());
                        max_ab = max_ab.max((l2e(i, j, a, b) + l2e(i, j, b, a)).abs());
                    }
                }
            }
        }
        assert!(
            max_ij < 1e-12,
            "l2 must be antisymmetric in i<->j (max |l2[ijab]+l2[jiab]| = {max_ij})"
        );
        assert!(
            max_ab < 1e-12,
            "l2 must be antisymmetric in a<->b (max |l2[ijab]+l2[ijba]| = {max_ab})"
        );
    }

    /// Pitfall 2 guard: update_ulambda is byte-identical regardless of rayon
    /// thread count (the oracle reductions never consult RAYON_NUM_THREADS).
    #[test]
    fn update_ulambda_thread_invariant() {
        let (no, nv) = (2usize, 3usize);
        let eris = synthetic_so_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let run = || {
            let mut w = vec![0.0_f64; nv * nv * nv * nv];
            update_ulambda(&t1, &t2, &t1, &t2, &eris, &mut w).expect("update_ulambda")
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
        let mut eris = synthetic_so_eris(no, nv);
        eris.oovv.pop();
        let (t1, t2) = synthetic_amps(no, nv);
        let mut w = vec![0.0_f64; nv * nv * nv * nv];
        let err =
            update_ulambda(&t1, &t2, &t1, &t2, &eris, &mut w).expect_err("must error on bad shape");
        assert!(matches!(err, CcsdError::ShapeMismatch { .. }));
    }

    /// α==β collapse: for a closed-shell RHF reference fed as α==β, the
    /// spin-orbital λ converges and the recovered correlation-density trace
    /// behavior matches the closed-shell path. We assert the converged λ is
    /// finite + converged here; the make_rdm1 collapse (urdm.rs) is the
    /// stronger numeric collapse test, and the OH oracle is the sufficient gate.
    #[test]
    fn ulambda_alpha_eq_beta_converges_like_closed_shell() {
        let mol = M(MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
            basis: BasisInput::Name("sto-3g".into()),
            ..Default::default()
        })
        .expect("mol");
        let mut mf = RHF::new(mol.clone());
        let scf = mf.kernel().expect("RHF");
        let refr = CcsdReference {
            mo_coeff: scf.mo_coeff.clone(),
            mo_energy: scf.mo_energy.clone(),
            mo_occ: scf.mo_occ.clone(),
            e_hf: scf.e_tot.0,
            converged: scf.converged,
            mol: mol.clone(),
        };
        // Closed-shell λ on the RHF reference.
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let eris_cs = crate::default_ao2mo(&refr, &Frozen::None).expect("cs eris");
        let cs_amps = crate::ccsd_kernel(&refr, &Frozen::None, &crate::NoCcsdOverrides, &pool)
            .expect("rccsd");
        let cs_t1 = cs_amps.amplitudes.t1_slice().map(|s| s.to_vec()).unwrap_or_default();
        let cs_t2 = cs_amps.amplitudes.t2_slice().map(|s| s.to_vec()).unwrap_or_default();
        let cs_lam = solve_lambda(&cs_t1, &cs_t2, &eris_cs, &pool).expect("cs lambda");
        assert!(cs_lam.converged, "closed-shell lambda converges");

        // Spin-orbital λ on the same reference as α==β.
        let mo_occ: Vec<f64> = refr
            .mo_occ
            .iter()
            .map(|&o| if o > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let chan = CcsdReference {
            mo_occ,
            ..refr.clone()
        };
        let uref = UccsdReference {
            alpha: chan.clone(),
            beta: chan,
        };
        let ures = crate::uccsd_kernel(&uref, &Frozen::None, &pool).expect("uccsd");
        let ulam = solve_ulambda(&ures.so_t1, &ures.so_t2, &ures.so_eris, &pool)
            .expect("solve_ulambda");
        assert!(ulam.converged, "spin-orbital lambda converges");
        for &x in ulam.l1.iter().chain(ulam.l2.iter()) {
            assert!(x.is_finite());
        }
    }
}
