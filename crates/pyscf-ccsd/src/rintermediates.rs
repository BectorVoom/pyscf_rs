//! Closed-shell RCCSD intermediates — `make_tau`/`cc_Foo`/`cc_Fvv`/`cc_Fov`/
//! `Loo`/`Lvv`/`cc_Woooo`/`cc_Wvvvv`/`cc_Wvoov`/`cc_Wvovo` (port of
//! `pyscf/cc/rintermediates.py:30-188`, Hirata et al. JCP 120, 2581 (2004)).
//!
//! **Reduction discipline (Pitfall 1/2 + FOUND-06):** every `einsum` becomes an
//! explicit host loop that MATERIALIZES the elementwise products of the
//! contracted axis into a `Vec`, then reduces with `pyscf_algebra::oracle_sum`
//! (a fixed-chunk pairwise tree — thread-count invariant). NO bare `+=`
//! accumulation across the contracted axis, NO `pyscf_algebra::gemm` (it is a
//! `NotYetImplemented{phase:2}` stub). Independent per-output-element terms are
//! collected into a `Vec<f64>` and reduced with a single `oracle_sum` so the
//! result is bit-exact under `release-oracle`.
//!
//! **Flat-index discipline (Pitfall 3):** every block is a flat C-order
//! (row-major, last index fastest) `Vec<f64>` over the active occ/vir ranges.
//! The exact offset formula is doc-commented at every block boundary. Lengths
//! are validated against the expected `nocc*nvir*...` product BEFORE indexing —
//! a logic error returns `CcsdError::ShapeMismatch`, never an OOB panic
//! (`#![forbid(unsafe_code)]` → no UB).
//!
//! `o` = active occupied (length `nocc`), `v` = active virtual (length `nvir`).
//! `fock`/`mo_energy` come from [`crate::eris::ChemistsEris`].
//!
//! The intermediate function names (`cc_Foo`, `cc_Fvv`, `cc_Woooo`, `Loo`,
//! `Lvv`, …) are deliberately the verbatim upstream `rintermediates.py` names
//! (sibling-crate fidelity), so `#![allow(non_snake_case)]` is intentional here.
#![allow(clippy::needless_range_loop)]
#![allow(non_snake_case)]

use crate::eris::ChemistsEris;
use crate::error::CcsdError;
use pyscf_algebra::oracle_sum;

// ---------------------------------------------------------------------------
// Flat-index helpers for the ChemistsEris blocks (all C-order). Each returns
// the flat offset for the named block; see crate::eris for the layout docs.
// ---------------------------------------------------------------------------

/// `oooo` element `(i,j,k,l)` at `((i*no + j)*no + k)*no + l`.
#[inline]
fn oooo_idx(no: usize, i: usize, j: usize, k: usize, l: usize) -> usize {
    ((i * no + j) * no + k) * no + l
}

/// `ovoo` element `(i,a,j,k)` at `((i*nv + a)*no + j)*no + k`.
#[inline]
fn ovoo_idx(no: usize, nv: usize, i: usize, a: usize, j: usize, k: usize) -> usize {
    ((i * nv + a) * no + j) * no + k
}

/// `oovv` element `(i,j,a,b)` at `((i*no + j)*nv + a)*nv + b`.
#[inline]
fn oovv_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

/// `ovov` element `(i,a,j,b)` at `((i*nv + a)*no + j)*nv + b`.
#[inline]
fn ovov_idx(no: usize, nv: usize, i: usize, a: usize, j: usize, b: usize) -> usize {
    ((i * nv + a) * no + j) * nv + b
}

/// `ovvo` element `(i,a,b,j)` at `((i*nv + a)*nv + b)*no + j`.
#[inline]
fn ovvo_idx(no: usize, nv: usize, i: usize, a: usize, b: usize, j: usize) -> usize {
    ((i * nv + a) * nv + b) * no + j
}

/// `ovvv` element `(i,a,b,c)` at `((i*nv + a)*nv + b)*nv + c`.
#[inline]
fn ovvv_idx(nv: usize, i: usize, a: usize, b: usize, c: usize) -> usize {
    ((i * nv + a) * nv + b) * nv + c
}

/// `vvvv` element `(a,b,c,d)` at `((a*nv + b)*nv + c)*nv + d`.
#[inline]
fn vvvv_idx(nv: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    ((a * nv + b) * nv + c) * nv + d
}

/// `fock` element `(p,q)` at `p*(no+nv) + q` over ALL active MOs.
#[inline]
fn fock_idx(nmo: usize, p: usize, q: usize) -> usize {
    p * nmo + q
}

/// t1 `(i,a)` at `i*nv + a`; t2 `(i,j,a,b)` at `((i*no + j)*nv + a)*nv + b`.
#[inline]
fn t1_idx(nv: usize, i: usize, a: usize) -> usize {
    i * nv + a
}

#[inline]
fn t2_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

/// Validate a block length, returning `ShapeMismatch` (never panics).
#[inline]
fn check_len(name: &str, got: usize, expected: usize) -> Result<(), CcsdError> {
    if got != expected {
        // Encode the offending block name in the error string via the
        // expected/got fields (the enum carries only usizes); the name aids
        // debugging through the test harness messages.
        let _ = name;
        return Err(CcsdError::ShapeMismatch { expected, got });
    }
    Ok(())
}

/// Validate the t1/t2 dimensions + the ERI blocks this module reads against the
/// `(nocc,nvir)` declared by `eris`. Done once at every intermediate entry so a
/// wrong-shaped hook return (T-06-03-SHAPE) is caught before any indexing.
fn validate(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<(usize, usize), CcsdError> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let nmo = no + nv;
    check_len("t1", t1.len(), no * nv)?;
    check_len("t2", t2.len(), no * no * nv * nv)?;
    check_len("oooo", eris.oooo.len(), no * no * no * no)?;
    check_len("ovoo", eris.ovoo.len(), no * nv * no * no)?;
    check_len("oovv", eris.oovv.len(), no * no * nv * nv)?;
    check_len("ovov", eris.ovov.len(), no * nv * no * nv)?;
    check_len("ovvo", eris.ovvo.len(), no * nv * nv * no)?;
    check_len("ovvv", eris.ovvv.len(), no * nv * nv * nv)?;
    check_len("vvvv", eris.vvvv.len(), nv * nv * nv * nv)?;
    check_len("fock", eris.fock.len(), nmo * nmo)?;
    check_len("mo_energy", eris.mo_energy.len(), nmo)?;
    Ok((no, nv))
}

// ---------------------------------------------------------------------------
// make_tau (port rintermediates is implicit; the rccsd.py tau is
// `t2 + einsum('ia,jb->ijab', t1, t1)`).
// ---------------------------------------------------------------------------

/// `tau[i,j,a,b] = t2[i,j,a,b] + t1[i,a]*t1[j,b]` (no contracted axis — pure
/// elementwise, so no reduction is needed). C-order `[no,no,nv,nv]`.
pub fn make_tau(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let mut tau = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let p = t2_idx(no, nv, i, j, a, b);
                    tau[p] = t2[p] + t1[t1_idx(nv, i, a)] * t1[t1_idx(nv, j, b)];
                }
            }
        }
    }
    Ok(tau)
}

// ---------------------------------------------------------------------------
// cc_Foo / cc_Fvv / cc_Fov  (rintermediates.py:30-59)
// ---------------------------------------------------------------------------

/// `cc_Foo` (rintermediates.py:30) → `Fki[k,i]`, C-order `[no,no]`:
/// ```text
/// Fki  = 2*einsum('kcld,ilcd->ki', ovov, t2)
/// Fki -=   einsum('kdlc,ilcd->ki', ovov, t2)
/// Fki += 2*einsum('kcld,ic,ld->ki', ovov, t1, t1)
/// Fki -=   einsum('kdlc,ic,ld->ki', ovov, t1, t1)
/// Fki += fock[:no,:no]
/// ```
pub fn cc_Foo(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let nmo = no + nv;
    let ovov = &eris.ovov;
    let mut fki = vec![0.0_f64; no * no];
    for k in 0..no {
        for i in 0..no {
            // Contract over (l,c,d) — materialize each term's product then
            // reduce. terms collected per (k,i) output element.
            let mut terms: Vec<f64> = Vec::with_capacity(no * nv * nv * 4 + 1);
            for l in 0..no {
                for c in 0..nv {
                    for d in 0..nv {
                        // ovov[k,c,l,d] and ovov[k,d,l,c]
                        let g_kcld = ovov[ovov_idx(no, nv, k, c, l, d)];
                        let g_kdlc = ovov[ovov_idx(no, nv, k, d, l, c)];
                        let t2_ilcd = t2[t2_idx(no, nv, i, l, c, d)];
                        let t1t1 = t1[t1_idx(nv, i, c)] * t1[t1_idx(nv, l, d)];
                        terms.push(2.0 * g_kcld * t2_ilcd);
                        terms.push(-g_kdlc * t2_ilcd);
                        terms.push(2.0 * g_kcld * t1t1);
                        terms.push(-g_kdlc * t1t1);
                    }
                }
            }
            terms.push(eris.fock[fock_idx(nmo, k, i)]);
            fki[k * no + i] = oracle_sum(&terms);
        }
    }
    Ok(fki)
}

/// `cc_Fvv` (rintermediates.py:41) → `Fac[a,c]`, C-order `[nv,nv]`:
/// ```text
/// Fac  =-2*einsum('kcld,klad->ac', ovov, t2)
/// Fac +=   einsum('kdlc,klad->ac', ovov, t2)
/// Fac -= 2*einsum('kcld,ka,ld->ac', ovov, t1, t1)
/// Fac +=   einsum('kdlc,ka,ld->ac', ovov, t1, t1)
/// Fac += fock[no:,no:]
/// ```
pub fn cc_Fvv(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let nmo = no + nv;
    let ovov = &eris.ovov;
    let mut fac = vec![0.0_f64; nv * nv];
    for a in 0..nv {
        for c in 0..nv {
            let mut terms: Vec<f64> = Vec::with_capacity(no * no * nv * 4 + 1);
            for k in 0..no {
                for l in 0..no {
                    for d in 0..nv {
                        // 'kcld' = ovov[k,c,l,d]; 'kdlc' = ovov[k,d,l,c]
                        let g_kcld = ovov[ovov_idx(no, nv, k, c, l, d)];
                        let g_kdlc = ovov[ovov_idx(no, nv, k, d, l, c)];
                        let t2_klad = t2[t2_idx(no, nv, k, l, a, d)];
                        let t1t1 = t1[t1_idx(nv, k, a)] * t1[t1_idx(nv, l, d)];
                        terms.push(-2.0 * g_kcld * t2_klad);
                        terms.push(g_kdlc * t2_klad);
                        terms.push(-2.0 * g_kcld * t1t1);
                        terms.push(g_kdlc * t1t1);
                    }
                }
            }
            terms.push(eris.fock[fock_idx(nmo, no + a, no + c)]);
            fac[a * nv + c] = oracle_sum(&terms);
        }
    }
    Ok(fac)
}

/// `cc_Fov` (rintermediates.py:52) → `Fkc[k,c]`, C-order `[no,nv]`:
/// ```text
/// Fkc  = 2*einsum('kcld,ld->kc', ovov, t1)
/// Fkc -=   einsum('kdlc,ld->kc', ovov, t1)
/// Fkc += fock[:no,no:]
/// ```
pub fn cc_Fov(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let nmo = no + nv;
    let ovov = &eris.ovov;
    let mut fkc = vec![0.0_f64; no * nv];
    for k in 0..no {
        for c in 0..nv {
            let mut terms: Vec<f64> = Vec::with_capacity(no * nv * 2 + 1);
            for l in 0..no {
                for d in 0..nv {
                    let g_kcld = ovov[ovov_idx(no, nv, k, c, l, d)];
                    let g_kdlc = ovov[ovov_idx(no, nv, k, d, l, c)];
                    let t1_ld = t1[t1_idx(nv, l, d)];
                    terms.push(2.0 * g_kcld * t1_ld);
                    terms.push(-g_kdlc * t1_ld);
                }
            }
            terms.push(eris.fock[fock_idx(nmo, k, no + c)]);
            fkc[k * nv + c] = oracle_sum(&terms);
        }
    }
    Ok(fkc)
}

// ---------------------------------------------------------------------------
// Loo / Lvv  (rintermediates.py:63-79)
// ---------------------------------------------------------------------------

/// `Loo` (rintermediates.py:63) → `Lki[k,i]`, C-order `[no,no]`:
/// ```text
/// Lki  = cc_Foo + einsum('kc,ic->ki', fov, t1)
/// Lki += 2*einsum('lcki,lc->ki', ovoo, t1)
/// Lki -=   einsum('kcli,lc->ki', ovoo, t1)
/// ```
pub fn Loo(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let nmo = no + nv;
    let f_oo = cc_Foo(t1, t2, eris)?;
    let ovoo = &eris.ovoo;
    let mut lki = vec![0.0_f64; no * no];
    for k in 0..no {
        for i in 0..no {
            let mut terms: Vec<f64> = Vec::with_capacity(nv + no * nv * 2 + 1);
            terms.push(f_oo[k * no + i]);
            // einsum('kc,ic->ki', fov, t1): contract over c.
            for c in 0..nv {
                let fov_kc = eris.fock[fock_idx(nmo, k, no + c)];
                terms.push(fov_kc * t1[t1_idx(nv, i, c)]);
            }
            // ovoo 'lcki' = ovoo[l,c,k,i]; 'kcli' = ovoo[k,c,l,i]
            for l in 0..no {
                for c in 0..nv {
                    let t1_lc = t1[t1_idx(nv, l, c)];
                    terms.push(2.0 * ovoo[ovoo_idx(no, nv, l, c, k, i)] * t1_lc);
                    terms.push(-ovoo[ovoo_idx(no, nv, k, c, l, i)] * t1_lc);
                }
            }
            lki[k * no + i] = oracle_sum(&terms);
        }
    }
    Ok(lki)
}

/// `Lvv` (rintermediates.py:72) → `Lac[a,c]`, C-order `[nv,nv]`:
/// ```text
/// Lac  = cc_Fvv - einsum('kc,ka->ac', fov, t1)
/// Lac += 2*einsum('kdac,kd->ac', ovvv, t1)
/// Lac -=   einsum('kcad,kd->ac', ovvv, t1)
/// ```
/// `ovvv` here is `eris.get_ovvv()` = the `(o,v,v,v)` block `(k,d,a,c)`.
pub fn Lvv(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let nmo = no + nv;
    let fvv = cc_Fvv(t1, t2, eris)?;
    let ovvv = &eris.ovvv;
    let mut lac = vec![0.0_f64; nv * nv];
    for a in 0..nv {
        for c in 0..nv {
            let mut terms: Vec<f64> = Vec::with_capacity(no + no * nv * 2 + 1);
            terms.push(fvv[a * nv + c]);
            // - einsum('kc,ka->ac', fov, t1): contract over k.
            for k in 0..no {
                let fov_kc = eris.fock[fock_idx(nmo, k, no + c)];
                terms.push(-fov_kc * t1[t1_idx(nv, k, a)]);
            }
            // 'kdac' = ovvv[k,d,a,c]; 'kcad' = ovvv[k,c,a,d]; contract k,d.
            for k in 0..no {
                for d in 0..nv {
                    let t1_kd = t1[t1_idx(nv, k, d)];
                    terms.push(2.0 * ovvv[ovvv_idx(nv, k, d, a, c)] * t1_kd);
                    terms.push(-ovvv[ovvv_idx(nv, k, c, a, d)] * t1_kd);
                }
            }
            lac[a * nv + c] = oracle_sum(&terms);
        }
    }
    Ok(lac)
}

// ---------------------------------------------------------------------------
// cc_Woooo / cc_Wvvvv / cc_Wvoov / cc_Wvovo  (rintermediates.py:83-123)
// ---------------------------------------------------------------------------

/// `cc_Woooo` (rintermediates.py:83) → `Wklij`, C-order `[no,no,no,no]`:
/// ```text
/// Wklij  = einsum('lcki,jc->klij', ovoo, t1)
/// Wklij += einsum('kclj,ic->klij', ovoo, t1)
/// Wklij += einsum('kcld,ijcd->klij', ovov, t2)
/// Wklij += einsum('kcld,ic,jd->klij', ovov, t1, t1)
/// Wklij += oooo.transpose(0,2,1,3)   # oooo[k,i,l,j] → element (k,l,i,j)
/// ```
pub fn cc_Woooo(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let ovoo = &eris.ovoo;
    let ovov = &eris.ovov;
    let mut w = vec![0.0_f64; no * no * no * no];
    for k in 0..no {
        for l in 0..no {
            for i in 0..no {
                for j in 0..no {
                    let mut terms: Vec<f64> = Vec::with_capacity(nv * nv + 2 * nv + 1);
                    // einsum('lcki,jc->klij'): contract c.
                    for c in 0..nv {
                        terms.push(ovoo[ovoo_idx(no, nv, l, c, k, i)] * t1[t1_idx(nv, j, c)]);
                    }
                    // einsum('kclj,ic->klij'): contract c.
                    for c in 0..nv {
                        terms.push(ovoo[ovoo_idx(no, nv, k, c, l, j)] * t1[t1_idx(nv, i, c)]);
                    }
                    // einsum('kcld,ijcd->klij') + einsum('kcld,ic,jd->klij').
                    for c in 0..nv {
                        for d in 0..nv {
                            let g = ovov[ovov_idx(no, nv, k, c, l, d)];
                            terms.push(g * t2[t2_idx(no, nv, i, j, c, d)]);
                            terms.push(g * t1[t1_idx(nv, i, c)] * t1[t1_idx(nv, j, d)]);
                        }
                    }
                    // oooo.transpose(0,2,1,3): W(k,l,i,j) += oooo[k,i,l,j].
                    terms.push(eris.oooo[oooo_idx(no, k, i, l, j)]);
                    w[oooo_idx(no, k, l, i, j)] = oracle_sum(&terms);
                }
            }
        }
    }
    Ok(w)
}

/// `cc_Wvvvv` (rintermediates.py:93) → `Wabcd`, C-order `[nv,nv,nv,nv]` — the
/// `Wabef ≈ nv⁴` arena tenant. Written INTO `out` (a pool-reserved buffer
/// supplied by the kernel, NOT allocated here — Pitfall 20 / CCSD-11):
/// ```text
/// Wabcd  = einsum('kdac,kb->abcd', ovvv, -t1)
/// Wabcd -= einsum('kcbd,ka->abcd', ovvv,  t1)
/// Wabcd += vvvv.transpose(0,2,1,3)   # vvvv[a,c,b,d] → element (a,b,c,d)
/// ```
/// `out.len()` must equal `nv^4`.
pub fn cc_Wvvvv_into(
    t1: &[f64],
    t2: &[f64],
    eris: &ChemistsEris,
    out: &mut [f64],
) -> Result<(), CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    check_len("Wvvvv-out", out.len(), nv * nv * nv * nv)?;
    let ovvv = &eris.ovvv;
    let vvvv = &eris.vvvv;
    for a in 0..nv {
        for b in 0..nv {
            for c in 0..nv {
                for d in 0..nv {
                    let mut terms: Vec<f64> = Vec::with_capacity(2 * no + 1);
                    // einsum('kdac,kb->abcd', ovvv, -t1): contract k.
                    for k in 0..no {
                        terms.push(-ovvv[ovvv_idx(nv, k, d, a, c)] * t1[t1_idx(nv, k, b)]);
                    }
                    // einsum('kcbd,ka->abcd', ovvv, t1) subtracted: contract k.
                    for k in 0..no {
                        terms.push(-ovvv[ovvv_idx(nv, k, c, b, d)] * t1[t1_idx(nv, k, a)]);
                    }
                    // vvvv.transpose(0,2,1,3): W(a,b,c,d) += vvvv[a,c,b,d].
                    terms.push(vvvv[vvvv_idx(nv, a, c, b, d)]);
                    out[vvvv_idx(nv, a, b, c, d)] = oracle_sum(&terms);
                }
            }
        }
    }
    Ok(())
}

/// Owned-buffer convenience wrapper for tests; the kernel uses
/// [`cc_Wvvvv_into`] with a pool-reserved buffer.
pub fn cc_Wvvvv(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let nv = eris.nvir;
    let mut out = vec![0.0_f64; nv * nv * nv * nv];
    cc_Wvvvv_into(t1, t2, eris, &mut out)?;
    Ok(out)
}

/// `cc_Wvoov` (rintermediates.py:101) → `Wakic`, C-order `[nv,no,no,nv]`:
/// ```text
/// Wakic  = einsum('kcad,id->akic', ovvv, t1)
/// Wakic -= einsum('kcli,la->akic', ovoo, t1)
/// Wakic += ovvo.transpose(2,0,3,1)   # ovvo[k,c,a,i] → element (a,k,i,c)
/// Wakic -= 0.5*einsum('ldkc,ilda->akic', ovov, t2)
/// Wakic -= 0.5*einsum('lckd,ilad->akic', ovov, t2)
/// Wakic -=     einsum('ldkc,id,la->akic', ovov, t1, t1)
/// Wakic +=     einsum('ldkc,ilad->akic', ovov, t2)
/// ```
pub fn cc_Wvoov(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let ovvv = &eris.ovvv;
    let ovoo = &eris.ovoo;
    let ovvo = &eris.ovvo;
    let ovov = &eris.ovov;
    let mut w = vec![0.0_f64; nv * no * no * nv];
    for a in 0..nv {
        for k in 0..no {
            for i in 0..no {
                for c in 0..nv {
                    let mut terms: Vec<f64> = Vec::with_capacity(nv + no + 1 + no * nv * 4);
                    // einsum('kcad,id->akic', ovvv, t1): contract d.
                    for d in 0..nv {
                        terms.push(ovvv[ovvv_idx(nv, k, c, a, d)] * t1[t1_idx(nv, i, d)]);
                    }
                    // - einsum('kcli,la->akic', ovoo, t1): contract l.
                    for l in 0..no {
                        terms.push(-ovoo[ovoo_idx(no, nv, k, c, l, i)] * t1[t1_idx(nv, l, a)]);
                    }
                    // ovvo.transpose(2,0,3,1): W(a,k,i,c) += ovvo[k,c,a,i].
                    terms.push(ovvo[ovvo_idx(no, nv, k, c, a, i)]);
                    // ovov contractions over (l,d).
                    for l in 0..no {
                        for d in 0..nv {
                            // 'ldkc' = ovov[l,d,k,c]; 'lckd' = ovov[l,c,k,d]
                            let g_ldkc = ovov[ovov_idx(no, nv, l, d, k, c)];
                            let g_lckd = ovov[ovov_idx(no, nv, l, c, k, d)];
                            // -0.5*einsum('ldkc,ilda'): t2[i,l,d,a]
                            terms.push(-0.5 * g_ldkc * t2[t2_idx(no, nv, i, l, d, a)]);
                            // -0.5*einsum('lckd,ilad'): t2[i,l,a,d]
                            terms.push(-0.5 * g_lckd * t2[t2_idx(no, nv, i, l, a, d)]);
                            // -einsum('ldkc,id,la'): t1[i,d]*t1[l,a]
                            terms.push(-g_ldkc * t1[t1_idx(nv, i, d)] * t1[t1_idx(nv, l, a)]);
                            // +einsum('ldkc,ilad'): t2[i,l,a,d]
                            terms.push(g_ldkc * t2[t2_idx(no, nv, i, l, a, d)]);
                        }
                    }
                    // C-order Wakic index (a,k,i,c): ((a*no + k)*no + i)*nv + c
                    w[((a * no + k) * no + i) * nv + c] = oracle_sum(&terms);
                }
            }
        }
    }
    Ok(w)
}

/// `cc_Wvovo` (rintermediates.py:114) → `Wakci`, C-order `[nv,no,nv,no]`:
/// ```text
/// Wakci  = einsum('kdac,id->akci', ovvv, t1)
/// Wakci -= einsum('lcki,la->akci', ovoo, t1)
/// Wakci += oovv.transpose(2,0,3,1)   # oovv[k,i,a,c] → element (a,k,c,i)
/// Wakci -= 0.5*einsum('lckd,ilda->akci', ovov, t2)
/// Wakci -=     einsum('lckd,id,la->akci', ovov, t1, t1)
/// ```
pub fn cc_Wvovo(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let ovvv = &eris.ovvv;
    let ovoo = &eris.ovoo;
    let oovv = &eris.oovv;
    let ovov = &eris.ovov;
    let mut w = vec![0.0_f64; nv * no * nv * no];
    for a in 0..nv {
        for k in 0..no {
            for c in 0..nv {
                for i in 0..no {
                    let mut terms: Vec<f64> = Vec::with_capacity(nv + no + 1 + no * nv * 2);
                    // einsum('kdac,id->akci', ovvv, t1): contract d.
                    for d in 0..nv {
                        terms.push(ovvv[ovvv_idx(nv, k, d, a, c)] * t1[t1_idx(nv, i, d)]);
                    }
                    // - einsum('lcki,la->akci', ovoo, t1): contract l.
                    for l in 0..no {
                        terms.push(-ovoo[ovoo_idx(no, nv, l, c, k, i)] * t1[t1_idx(nv, l, a)]);
                    }
                    // oovv.transpose(2,0,3,1): W(a,k,c,i) += oovv[k,i,a,c].
                    terms.push(oovv[oovv_idx(no, nv, k, i, a, c)]);
                    // ovov contractions over (l,d).
                    for l in 0..no {
                        for d in 0..nv {
                            // 'lckd' = ovov[l,c,k,d]
                            let g_lckd = ovov[ovov_idx(no, nv, l, c, k, d)];
                            // -0.5*einsum('lckd,ilda'): t2[i,l,d,a]
                            terms.push(-0.5 * g_lckd * t2[t2_idx(no, nv, i, l, d, a)]);
                            // -einsum('lckd,id,la'): t1[i,d]*t1[l,a]
                            terms.push(-g_lckd * t1[t1_idx(nv, i, d)] * t1[t1_idx(nv, l, a)]);
                        }
                    }
                    // C-order Wakci index (a,k,c,i): ((a*no + k)*nv + c)*no + i
                    w[((a * no + k) * nv + c) * no + i] = oracle_sum(&terms);
                }
            }
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic synthetic ChemistsEris of size (nocc,nvir) with
    /// arbitrary-but-fixed values keyed off the flat index, so the longhand
    /// references are reproducible.
    fn synthetic_eris(no: usize, nv: usize) -> ChemistsEris {
        let nmo = no + nv;
        let mk = |len: usize, seed: f64| -> Vec<f64> {
            (0..len)
                .map(|i| 0.1 + seed + ((i % 7) as f64) * 0.05 - ((i % 3) as f64) * 0.03)
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
            // Diagonal-ish fock (canonical-ish): diagonal mo_energy + small off-diag.
            fock: {
                let mut f = vec![0.0_f64; nmo * nmo];
                for p in 0..nmo {
                    for q in 0..nmo {
                        f[p * nmo + q] = if p == q {
                            -0.6 + (p as f64) * 0.4
                        } else {
                            0.01 * (((p + q) % 5) as f64)
                        };
                    }
                }
                f
            },
            mo_energy: (0..nmo).map(|p| -0.6 + (p as f64) * 0.4).collect(),
            nocc: no,
            nvir: nv,
        }
    }

    fn synthetic_amps(no: usize, nv: usize) -> (Vec<f64>, Vec<f64>) {
        let t1: Vec<f64> = (0..no * nv).map(|i| 0.02 + ((i % 5) as f64) * 0.011).collect();
        let t2: Vec<f64> = (0..no * no * nv * nv)
            .map(|i| 0.005 + ((i % 9) as f64) * 0.003 - ((i % 4) as f64) * 0.002)
            .collect();
        (t1, t2)
    }

    /// make_tau: longhand element-for-element reference.
    #[test]
    fn make_tau_matches_longhand() {
        for (no, nv) in [(1usize, 1usize), (2, 2), (2, 3)] {
            let eris = synthetic_eris(no, nv);
            let (t1, t2) = synthetic_amps(no, nv);
            let tau = make_tau(&t1, &t2, &eris).expect("tau");
            for i in 0..no {
                for j in 0..no {
                    for a in 0..nv {
                        for b in 0..nv {
                            let p = ((i * no + j) * nv + a) * nv + b;
                            let want = t2[p] + t1[i * nv + a] * t1[j * nv + b];
                            assert!((tau[p] - want).abs() < 1e-14, "tau[{i}{j}{a}{b}]");
                        }
                    }
                }
            }
        }
    }

    /// cc_Foo / cc_Fvv / cc_Fov: independent quadruple-loop scalar-fold
    /// references for nocc=1,nvir=1 and nocc=2,nvir=2 (Hirata einsums).
    #[test]
    fn cc_f_intermediates_match_longhand() {
        for (no, nv) in [(1usize, 1usize), (2, 2)] {
            let eris = synthetic_eris(no, nv);
            let (t1, t2) = synthetic_amps(no, nv);
            let nmo = no + nv;
            let fock = &eris.fock;
            let ovov = &eris.ovov;
            let oo = |i: usize, a: usize, j: usize, b: usize| {
                ovov[((i * nv + a) * no + j) * nv + b]
            };
            let t2e = |i: usize, j: usize, a: usize, b: usize| {
                t2[((i * no + j) * nv + a) * nv + b]
            };
            let t1e = |i: usize, a: usize| t1[i * nv + a];

            // --- cc_Foo reference ---
            let foo = cc_Foo(&t1, &t2, &eris).expect("foo");
            for k in 0..no {
                for i in 0..no {
                    let mut acc = 0.0_f64;
                    for l in 0..no {
                        for c in 0..nv {
                            for d in 0..nv {
                                let g_kcld = oo(k, c, l, d);
                                let g_kdlc = oo(k, d, l, c);
                                acc += 2.0 * g_kcld * t2e(i, l, c, d);
                                acc -= g_kdlc * t2e(i, l, c, d);
                                acc += 2.0 * g_kcld * t1e(i, c) * t1e(l, d);
                                acc -= g_kdlc * t1e(i, c) * t1e(l, d);
                            }
                        }
                    }
                    acc += fock[k * nmo + i];
                    assert!((foo[k * no + i] - acc).abs() < 1e-12, "Foo[{k}{i}]");
                }
            }

            // --- cc_Fvv reference ---
            let fvv = cc_Fvv(&t1, &t2, &eris).expect("fvv");
            for a in 0..nv {
                for c in 0..nv {
                    let mut acc = 0.0_f64;
                    for k in 0..no {
                        for l in 0..no {
                            for d in 0..nv {
                                let g_kcld = oo(k, c, l, d);
                                let g_kdlc = oo(k, d, l, c);
                                acc -= 2.0 * g_kcld * t2e(k, l, a, d);
                                acc += g_kdlc * t2e(k, l, a, d);
                                acc -= 2.0 * g_kcld * t1e(k, a) * t1e(l, d);
                                acc += g_kdlc * t1e(k, a) * t1e(l, d);
                            }
                        }
                    }
                    acc += fock[(no + a) * nmo + (no + c)];
                    assert!((fvv[a * nv + c] - acc).abs() < 1e-12, "Fvv[{a}{c}]");
                }
            }

            // --- cc_Fov reference ---
            let fov = cc_Fov(&t1, &t2, &eris).expect("fov");
            for k in 0..no {
                for c in 0..nv {
                    let mut acc = 0.0_f64;
                    for l in 0..no {
                        for d in 0..nv {
                            acc += 2.0 * oo(k, c, l, d) * t1e(l, d);
                            acc -= oo(k, d, l, c) * t1e(l, d);
                        }
                    }
                    acc += fock[k * nmo + (no + c)];
                    assert!((fov[k * nv + c] - acc).abs() < 1e-12, "Fov[{k}{c}]");
                }
            }
        }
    }

    /// The heaviest einsum: cc_Wvvvv 'kdac,kb->abcd' / 'kcbd,ka->abcd' +
    /// vvvv.transpose(0,2,1,3). Longhand quadruple/scalar-fold reference on a
    /// 2x2 synthetic block — proves the host-loop port + flat-index arithmetic.
    #[test]
    fn cc_wvvvv_matches_longhand_2x2() {
        let (no, nv) = (2usize, 2usize);
        let eris = synthetic_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let w = cc_Wvvvv(&t1, &t2, &eris).expect("wvvvv");
        let ovvv = &eris.ovvv;
        let vvvv = &eris.vvvv;
        let ov = |k: usize, d: usize, a: usize, c: usize| {
            ovvv[((k * nv + d) * nv + a) * nv + c]
        };
        let vv = |a: usize, b: usize, c: usize, d: usize| {
            vvvv[((a * nv + b) * nv + c) * nv + d]
        };
        for a in 0..nv {
            for b in 0..nv {
                for c in 0..nv {
                    for d in 0..nv {
                        let mut acc = 0.0_f64;
                        for k in 0..no {
                            acc += ov(k, d, a, c) * (-t1[k * nv + b]);
                            acc -= ov(k, c, b, d) * t1[k * nv + a];
                        }
                        acc += vv(a, c, b, d); // transpose(0,2,1,3)
                        let got = w[((a * nv + b) * nv + c) * nv + d];
                        assert!((got - acc).abs() < 1e-12, "Wvvvv[{a}{b}{c}{d}]");
                    }
                }
            }
        }
    }

    /// cc_Wvvvv_into writes into a caller-supplied buffer (the arena path) and
    /// produces the same values as the owned wrapper.
    #[test]
    fn cc_wvvvv_into_matches_owned() {
        let (no, nv) = (2usize, 3usize);
        let eris = synthetic_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let owned = cc_Wvvvv(&t1, &t2, &eris).expect("owned");
        let mut buf = vec![0.0_f64; nv * nv * nv * nv];
        cc_Wvvvv_into(&t1, &t2, &eris, &mut buf).expect("into");
        assert_eq!(owned, buf);
    }

    /// Shape validation: a wrong-length block returns ShapeMismatch, never a
    /// panic (T-06-03-SHAPE).
    #[test]
    fn wrong_shape_returns_error_not_panic() {
        let (no, nv) = (2usize, 2usize);
        let mut eris = synthetic_eris(no, nv);
        eris.ovov.pop(); // corrupt a block length
        let (t1, t2) = synthetic_amps(no, nv);
        let err = cc_Foo(&t1, &t2, &eris).expect_err("must error on bad shape");
        assert!(matches!(err, CcsdError::ShapeMismatch { .. }));
    }
}
