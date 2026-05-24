//! Spin-orbital UCCSD intermediates — the open-shell counterpart of
//! [`crate::rintermediates`] (port of the spin-block intermediates in
//! `pyscf/cc/uintermediates.py`; Gauss & Stanton, J. Chem. Phys. 103, 3561
//! (1995) Table III and Stanton, Gauss, Watts & Bartlett, J. Chem. Phys. 94,
//! 4334 (1991) Eqs. (3)-(13)).
//!
//! UCCSD is the spin-block form of spin-orbital CCSD: the αα/ββ same-spin
//! channels are antisymmetrized, the αβ cross channel is direct-only, and each
//! channel carries its OWN α/β orbital energies (`uintermediates.py:47-50`).
//! The cleanest 1:1 host-loop port of that algebra is to assemble the combined
//! spin-orbital antisymmetrized integrals `<pq||rs> = (pr|qs) − (ps|qr)` once
//! (the αα/ββ/αβ blocks naturally fall out of the spin-orbital index ranges:
//! a spin-orbital integral is non-zero only when the spins of `p,r` and of
//! `q,s` separately match), then run the compact spin-orbital CCSD
//! intermediates. The spin-resolved aa/bb/ab amplitudes are recovered by their
//! spin labels — and a spin-symmetric (α==β) reference reproduces the closed-
//! shell RCCSD energy exactly (the cross-check the smoke asserts).
//!
//! **Reduction discipline (Pitfall 1/2 + FOUND-06):** every contracted-axis
//! sum MATERIALIZES the per-output-element products into a `Vec` and reduces
//! with `pyscf_algebra::oracle_sum` — NO bare `+=` accumulation across a
//! contracted axis, NO `pyscf_algebra::gemm` (a `NotYetImplemented{phase:2}`
//! stub). Bit-exact + thread-count invariant under `release-oracle`.
//!
//! **Flat-index discipline (Pitfall 3):** every tensor is a flat C-order
//! (row-major, last index fastest) `Vec<f64>`. The offset formula is
//! doc-commented at every block boundary. Block lengths are validated against
//! `no/nv` BEFORE indexing — a logic error returns `CcsdError::ShapeMismatch`,
//! never an OOB panic (`#![forbid(unsafe_code)]`).
//!
//! `o` = spin-orbital occupied (length `no = nocc_a + nocc_b`), `v` =
//! spin-orbital virtual (length `nv = nvir_a + nvir_b`).
#![allow(clippy::needless_range_loop)]

use crate::error::CcsdError;
use pyscf_algebra::oracle_sum;

/// Spin-orbital antisymmetrized ERIs + Fock for UCCSD, in the physicist's
/// `<pq||rs>` convention (`= <pq|rs> − <pq|sr>`), block-resolved over the
/// occupied (`o`, length `no`) and virtual (`v`, length `nv`) spin-orbital
/// ranges. Every block is a flat C-order `Vec<f64>`.
///
/// The blocks consumed by the spin-orbital CCSD equations:
/// - `oovv`: `<ij||ab>`, `[no,no,nv,nv]`, `((i*no + j)*nv + a)*nv + b`.
/// - `oooo`: `<ij||kl>`, `[no,no,no,no]`, `((i*no + j)*no + k)*no + l`.
/// - `vvvv`: `<ab||cd>`, `[nv,nv,nv,nv]`, `((a*nv + b)*nv + c)*nv + d`.
/// - `ovvo`: `<ia||bj>`, `[no,nv,nv,no]`, `((i*nv + a)*nv + b)*no + j`.
/// - `ooov`: `<ij||ka>`, `[no,no,no,nv]`, `((i*no + j)*no + k)*nv + a`.
/// - `ovvv`: `<ia||bc>`, `[no,nv,nv,nv]`, `((i*nv + a)*nv + b)*nv + c`.
/// - `fock`: spin-orbital Fock, `[nmo,nmo]` over `nmo = no + nv` (occ first),
///   diagonal for a canonical reference (`= mo_energy`).
/// - `mo_energy`: `[nmo]` (occ first, then vir).
#[derive(Debug, Clone)]
pub struct SpinOrbitalEris {
    /// `<ij||ab>`, flat `[no,no,nv,nv]` C-order.
    pub oovv: Vec<f64>,
    /// `<ij||kl>`, flat `[no,no,no,no]` C-order.
    pub oooo: Vec<f64>,
    /// `<ab||cd>`, flat `[nv,nv,nv,nv]` C-order (the `nv⁴` arena tenant).
    pub vvvv: Vec<f64>,
    /// `<ia||bj>`, flat `[no,nv,nv,no]` C-order.
    pub ovvo: Vec<f64>,
    /// `<ij||ka>`, flat `[no,no,no,nv]` C-order.
    pub ooov: Vec<f64>,
    /// `<ia||bc>`, flat `[no,nv,nv,nv]` C-order.
    pub ovvv: Vec<f64>,
    /// Spin-orbital Fock matrix, flat `[nmo,nmo]` (`nmo = no + nv`).
    pub fock: Vec<f64>,
    /// Spin-orbital MO energies, `[nmo]` (occ first, then vir).
    pub mo_energy: Vec<f64>,
    /// Number of occupied spin-orbitals (`nocc_a + nocc_b`).
    pub no: usize,
    /// Number of virtual spin-orbitals (`nvir_a + nvir_b`).
    pub nv: usize,
}

// ---------------------------------------------------------------------------
// Flat-index helpers (all C-order).
// ---------------------------------------------------------------------------

/// `oovv` element `<ij||ab>` at `((i*no + j)*nv + a)*nv + b`.
#[inline]
pub(crate) fn oovv_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

/// `oooo` element `<ij||kl>` at `((i*no + j)*no + k)*no + l`.
#[inline]
pub(crate) fn oooo_idx(no: usize, i: usize, j: usize, k: usize, l: usize) -> usize {
    ((i * no + j) * no + k) * no + l
}

/// `vvvv` element `<ab||cd>` at `((a*nv + b)*nv + c)*nv + d`.
#[inline]
pub(crate) fn vvvv_idx(nv: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    ((a * nv + b) * nv + c) * nv + d
}

/// `ovvo` element `<ia||bj>` at `((i*nv + a)*nv + b)*no + j`.
#[inline]
pub(crate) fn ovvo_idx(no: usize, nv: usize, i: usize, a: usize, b: usize, j: usize) -> usize {
    ((i * nv + a) * nv + b) * no + j
}

/// `ooov` element `<ij||ka>` at `((i*no + j)*no + k)*nv + a`.
#[inline]
pub(crate) fn ooov_idx(no: usize, nv: usize, i: usize, j: usize, k: usize, a: usize) -> usize {
    ((i * no + j) * no + k) * nv + a
}

/// `ovvv` element `<ia||bc>` at `((i*nv + a)*nv + b)*nv + c`.
#[inline]
pub(crate) fn ovvv_idx(nv: usize, i: usize, a: usize, b: usize, c: usize) -> usize {
    ((i * nv + a) * nv + b) * nv + c
}

/// t1 `(i,a)` at `i*nv + a`.
#[inline]
pub(crate) fn t1_idx(nv: usize, i: usize, a: usize) -> usize {
    i * nv + a
}

/// t2 `(i,j,a,b)` at `((i*no + j)*nv + a)*nv + b`.
#[inline]
pub(crate) fn t2_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

/// `fock` element `(p,q)` at `p*nmo + q`.
#[inline]
pub(crate) fn fock_idx(nmo: usize, p: usize, q: usize) -> usize {
    p * nmo + q
}

#[inline]
fn check_len(got: usize, expected: usize) -> Result<(), CcsdError> {
    if got != expected {
        return Err(CcsdError::ShapeMismatch { expected, got });
    }
    Ok(())
}

/// Validate the t1/t2 dimensions and the spin-orbital ERI blocks against the
/// `(no,nv)` declared by `eris`. Run once at every intermediate entry so a
/// wrong-shaped block (T-06-04-SHAPE) is caught before any indexing.
pub fn validate(
    t1: &[f64],
    t2: &[f64],
    eris: &SpinOrbitalEris,
) -> Result<(usize, usize), CcsdError> {
    let no = eris.no;
    let nv = eris.nv;
    let nmo = no + nv;
    check_len(t1.len(), no * nv)?;
    check_len(t2.len(), no * no * nv * nv)?;
    check_len(eris.oovv.len(), no * no * nv * nv)?;
    check_len(eris.oooo.len(), no * no * no * no)?;
    check_len(eris.vvvv.len(), nv * nv * nv * nv)?;
    check_len(eris.ovvo.len(), no * nv * nv * no)?;
    check_len(eris.ooov.len(), no * no * no * nv)?;
    check_len(eris.ovvv.len(), no * nv * nv * nv)?;
    check_len(eris.fock.len(), nmo * nmo)?;
    check_len(eris.mo_energy.len(), nmo)?;
    Ok((no, nv))
}

// ---------------------------------------------------------------------------
// make_tau (the effective doubles, `uintermediates.py:33-39`, spin-orbital
// form): tau[i,j,a,b] = t2[i,j,a,b] + t1[i,a]*t1[j,b] − t1[i,b]*t1[j,a].
// Pure elementwise (no contracted axis) — no reduction needed.
// ---------------------------------------------------------------------------

/// `tau[i,j,a,b] = t2 + t1[i,a]·t1[j,b] − t1[i,b]·t1[j,a]` (spin-orbital
/// `make_tau`). C-order `[no,no,nv,nv]`.
pub fn make_tau(t1: &[f64], t2: &[f64], eris: &SpinOrbitalEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let mut tau = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let p = t2_idx(no, nv, i, j, a, b);
                    tau[p] = t2[p]
                        + t1[t1_idx(nv, i, a)] * t1[t1_idx(nv, j, b)]
                        - t1[t1_idx(nv, i, b)] * t1[t1_idx(nv, j, a)];
                }
            }
        }
    }
    Ok(tau)
}

/// `tau_tilde[i,j,a,b] = t2 + 0.5·(t1[i,a]·t1[j,b] − t1[i,b]·t1[j,a])`
/// (Stanton 1991 Eq. (9) `\tilde\tau`). C-order `[no,no,nv,nv]`.
pub fn make_tau_tilde(
    t1: &[f64],
    t2: &[f64],
    eris: &SpinOrbitalEris,
) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let mut tau = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let p = t2_idx(no, nv, i, j, a, b);
                    tau[p] = t2[p]
                        + 0.5
                            * (t1[t1_idx(nv, i, a)] * t1[t1_idx(nv, j, b)]
                                - t1[t1_idx(nv, i, b)] * t1[t1_idx(nv, j, a)]);
                }
            }
        }
    }
    Ok(tau)
}

// ---------------------------------------------------------------------------
// One-body intermediates (Stanton 1991 Eqs. (3)-(5)). Each spin channel uses
// its OWN orbital energies through the spin-orbital Fock (diagonal off the
// canonical reference) — the αα and ββ diagonal blocks of `fock` carry the α
// and β energies respectively (uintermediates.py:47-50).
// ---------------------------------------------------------------------------

/// `f_vv` (Stanton 1991 Eq. (3)) → `Fae`, C-order `[nv,nv]`:
/// ```text
/// Fae = (1−δ_ae)·f_vv[a,e]
///       − 0.5·Σ_m f_ov[m,e]·t1[m,a]
///       + Σ_mf t1[m,f]·<ma||fe>
///       − 0.5·Σ_mnf taut[m,n,a,f]·<mn||ef>
/// ```
pub fn f_vv(t1: &[f64], t2: &[f64], eris: &SpinOrbitalEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let nmo = no + nv;
    let taut = make_tau_tilde(t1, t2, eris)?;
    let mut fae = vec![0.0_f64; nv * nv];
    for a in 0..nv {
        for e in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            // (1−δ_ae)·f_vv[a,e]  (the canonical off-diagonal Fock; 0 here).
            if a != e {
                terms.push(eris.fock[fock_idx(nmo, no + a, no + e)]);
            }
            // −0.5·Σ_m f_ov[m,e]·t1[m,a]
            for m in 0..no {
                terms.push(-0.5 * eris.fock[fock_idx(nmo, m, no + e)] * t1[t1_idx(nv, m, a)]);
            }
            // + Σ_m,f t1[m,f]·<ma||fe>  =  ovvv[m,a,f,e]
            for m in 0..no {
                for f in 0..nv {
                    terms.push(t1[t1_idx(nv, m, f)] * eris.ovvv[ovvv_idx(nv, m, a, f, e)]);
                }
            }
            // −0.5·Σ_m,n,f taut[m,n,a,f]·<mn||ef>  =  oovv[m,n,e,f]
            for m in 0..no {
                for n in 0..no {
                    for f in 0..nv {
                        terms.push(
                            -0.5 * taut[t2_idx(no, nv, m, n, a, f)]
                                * eris.oovv[oovv_idx(no, nv, m, n, e, f)],
                        );
                    }
                }
            }
            fae[a * nv + e] = oracle_sum(&terms);
        }
    }
    Ok(fae)
}

/// `f_oo` (Stanton 1991 Eq. (4)) → `Fmi`, C-order `[no,no]`:
/// ```text
/// Fmi = (1−δ_mi)·f_oo[m,i]
///       + 0.5·Σ_e t1[i,e]·f_ov[m,e]
///       + Σ_ne t1[n,e]·<mn||ie>
///       + 0.5·Σ_nef taut[i,n,e,f]·<mn||ef>
/// ```
pub fn f_oo(t1: &[f64], t2: &[f64], eris: &SpinOrbitalEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let nmo = no + nv;
    let taut = make_tau_tilde(t1, t2, eris)?;
    let mut fmi = vec![0.0_f64; no * no];
    for m in 0..no {
        for i in 0..no {
            let mut terms: Vec<f64> = Vec::new();
            // (1−δ_mi)·f_oo[m,i]
            if m != i {
                terms.push(eris.fock[fock_idx(nmo, m, i)]);
            }
            // +0.5·Σ_e t1[i,e]·f_ov[m,e]
            for e in 0..nv {
                terms.push(0.5 * t1[t1_idx(nv, i, e)] * eris.fock[fock_idx(nmo, m, no + e)]);
            }
            // + Σ_n,e t1[n,e]·<mn||ie>  =  ooov[m,n,i,e]
            for n in 0..no {
                for e in 0..nv {
                    terms.push(t1[t1_idx(nv, n, e)] * eris.ooov[ooov_idx(no, nv, m, n, i, e)]);
                }
            }
            // +0.5·Σ_n,e,f taut[i,n,e,f]·<mn||ef>  =  oovv[m,n,e,f]
            for n in 0..no {
                for e in 0..nv {
                    for f in 0..nv {
                        terms.push(
                            0.5 * taut[t2_idx(no, nv, i, n, e, f)]
                                * eris.oovv[oovv_idx(no, nv, m, n, e, f)],
                        );
                    }
                }
            }
            fmi[m * no + i] = oracle_sum(&terms);
        }
    }
    Ok(fmi)
}

/// `f_ov` (Stanton 1991 Eq. (5)) → `Fme`, C-order `[no,nv]`:
/// ```text
/// Fme = f_ov[m,e] + Σ_nf t1[n,f]·<mn||ef>
/// ```
pub fn f_ov(t1: &[f64], t2: &[f64], eris: &SpinOrbitalEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let nmo = no + nv;
    let mut fme = vec![0.0_f64; no * nv];
    for m in 0..no {
        for e in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            terms.push(eris.fock[fock_idx(nmo, m, no + e)]);
            for n in 0..no {
                for f in 0..nv {
                    terms.push(t1[t1_idx(nv, n, f)] * eris.oovv[oovv_idx(no, nv, m, n, e, f)]);
                }
            }
            fme[m * nv + e] = oracle_sum(&terms);
        }
    }
    Ok(fme)
}

// ---------------------------------------------------------------------------
// Two-body intermediates (Stanton 1991 Eqs. (6)-(8)).
// ---------------------------------------------------------------------------

/// `w_oooo` (Stanton 1991 Eq. (6)) → `Wmnij`, C-order `[no,no,no,no]`:
/// ```text
/// Wmnij = <mn||ij>
///       + Σ_e ( t1[j,e]·<mn||ie> − t1[i,e]·<mn||je> )
///       + 0.25·Σ_ef tau[i,j,e,f]·<mn||ef>
/// ```
pub fn w_oooo(t1: &[f64], t2: &[f64], eris: &SpinOrbitalEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let tau = make_tau(t1, t2, eris)?;
    let mut w = vec![0.0_f64; no * no * no * no];
    for m in 0..no {
        for n in 0..no {
            for i in 0..no {
                for j in 0..no {
                    let mut terms: Vec<f64> = Vec::new();
                    // <mn||ij> = oooo[m,n,i,j]
                    terms.push(eris.oooo[oooo_idx(no, m, n, i, j)]);
                    // Σ_e P(ij) t1[j,e]·<mn||ie>:
                    //   + t1[j,e]·ooov[m,n,i,e] − t1[i,e]·ooov[m,n,j,e]
                    for e in 0..nv {
                        terms.push(
                            t1[t1_idx(nv, j, e)] * eris.ooov[ooov_idx(no, nv, m, n, i, e)],
                        );
                        terms.push(
                            -t1[t1_idx(nv, i, e)] * eris.ooov[ooov_idx(no, nv, m, n, j, e)],
                        );
                    }
                    // 0.25·Σ_ef tau[i,j,e,f]·<mn||ef> = oovv[m,n,e,f]
                    for e in 0..nv {
                        for f in 0..nv {
                            terms.push(
                                0.25 * tau[t2_idx(no, nv, i, j, e, f)]
                                    * eris.oovv[oovv_idx(no, nv, m, n, e, f)],
                            );
                        }
                    }
                    w[oooo_idx(no, m, n, i, j)] = oracle_sum(&terms);
                }
            }
        }
    }
    Ok(w)
}

/// `w_vvvv` (Stanton 1991 Eq. (7)) → `Wabef`, C-order `[nv,nv,nv,nv]` — the
/// `nv⁴` arena tenant, written INTO a caller-supplied buffer (Pitfall 20 /
/// CCSD-11; `out.len()` must equal `nv⁴`):
/// ```text
/// Wabef = <ab||ef>
///       − Σ_m ( t1[m,b]·<am||ef> − t1[m,a]·<bm||ef> )
///       + 0.25·Σ_mn tau[m,n,a,b]·<mn||ef>
/// ```
/// `<am||ef>` is recovered from `ovvv` via the antisymmetry `<am||ef> =
/// −<ma||ef> = −ovvv[m,a,e,f]`.
pub fn w_vvvv_into(
    t1: &[f64],
    t2: &[f64],
    eris: &SpinOrbitalEris,
    out: &mut [f64],
) -> Result<(), CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    check_len(out.len(), nv * nv * nv * nv)?;
    let tau = make_tau(t1, t2, eris)?;
    for a in 0..nv {
        for b in 0..nv {
            for e in 0..nv {
                for f in 0..nv {
                    let mut terms: Vec<f64> = Vec::new();
                    // <ab||ef> = vvvv[a,b,e,f]
                    terms.push(eris.vvvv[vvvv_idx(nv, a, b, e, f)]);
                    // −Σ_m P(ab) t1[m,b]·<am||ef>:
                    //   <am||ef> = −ovvv[m,a,e,f];  <bm||ef> = −ovvv[m,b,e,f]
                    for m in 0..no {
                        terms.push(t1[t1_idx(nv, m, b)] * eris.ovvv[ovvv_idx(nv, m, a, e, f)]);
                        terms.push(-t1[t1_idx(nv, m, a)] * eris.ovvv[ovvv_idx(nv, m, b, e, f)]);
                    }
                    // 0.25·Σ_mn tau[m,n,a,b]·<mn||ef> = oovv[m,n,e,f]
                    for m in 0..no {
                        for n in 0..no {
                            terms.push(
                                0.25 * tau[t2_idx(no, nv, m, n, a, b)]
                                    * eris.oovv[oovv_idx(no, nv, m, n, e, f)],
                            );
                        }
                    }
                    out[vvvv_idx(nv, a, b, e, f)] = oracle_sum(&terms);
                }
            }
        }
    }
    Ok(())
}

/// Owned-buffer convenience wrapper for [`w_vvvv_into`] (tests + non-arena
/// callers); the kernel uses the `_into` form with a pool-reserved buffer.
pub fn w_vvvv(t1: &[f64], t2: &[f64], eris: &SpinOrbitalEris) -> Result<Vec<f64>, CcsdError> {
    let nv = eris.nv;
    let mut out = vec![0.0_f64; nv * nv * nv * nv];
    w_vvvv_into(t1, t2, eris, &mut out)?;
    Ok(out)
}

/// `w_ovvo` (Stanton 1991 Eq. (8)) → `Wmbej`, C-order `[no,nv,nv,no]`:
/// ```text
/// Wmbej = <mb||ej>
///       + Σ_f t1[j,f]·<mb||ef>
///       − Σ_n t1[n,b]·<mn||ej>
///       − Σ_nf ( 0.5·t2[j,n,f,b] + t1[j,f]·t1[n,b] )·<mn||ef>
/// ```
/// `<mb||ej> = ovvo[m,b,e,j]`; `<mb||ef> = ovvv[m,b,e,f]`; `<mn||ej>` is
/// recovered from `ooov` via `<mn||ej> = −<mn||je> = −ooov[m,n,j,e]`.
pub fn w_ovvo(t1: &[f64], t2: &[f64], eris: &SpinOrbitalEris) -> Result<Vec<f64>, CcsdError> {
    let (no, nv) = validate(t1, t2, eris)?;
    let mut w = vec![0.0_f64; no * nv * nv * no];
    for m in 0..no {
        for b in 0..nv {
            for e in 0..nv {
                for j in 0..no {
                    let mut terms: Vec<f64> = Vec::new();
                    // <mb||ej> = ovvo[m,b,e,j]
                    terms.push(eris.ovvo[ovvo_idx(no, nv, m, b, e, j)]);
                    // + Σ_f t1[j,f]·<mb||ef> = ovvv[m,b,e,f]
                    for f in 0..nv {
                        terms.push(
                            t1[t1_idx(nv, j, f)] * eris.ovvv[ovvv_idx(nv, m, b, e, f)],
                        );
                    }
                    // − Σ_n t1[n,b]·<mn||ej> ;  <mn||ej> = −ooov[m,n,j,e]
                    for n in 0..no {
                        terms.push(
                            t1[t1_idx(nv, n, b)] * eris.ooov[ooov_idx(no, nv, m, n, j, e)],
                        );
                    }
                    // − Σ_n,f ( 0.5·t2[j,n,f,b] + t1[j,f]·t1[n,b] )·<mn||ef>
                    //   <mn||ef> = oovv[m,n,e,f]
                    for n in 0..no {
                        for f in 0..nv {
                            let g = eris.oovv[oovv_idx(no, nv, m, n, e, f)];
                            terms.push(-0.5 * t2[t2_idx(no, nv, j, n, f, b)] * g);
                            terms.push(
                                -t1[t1_idx(nv, j, f)] * t1[t1_idx(nv, n, b)] * g,
                            );
                        }
                    }
                    w[ovvo_idx(no, nv, m, b, e, j)] = oracle_sum(&terms);
                }
            }
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic synthetic SpinOrbitalEris of size (no,nv) with
    /// values keyed off the flat index, antisymmetrized in the physical
    /// permutational symmetries so the intermediate longhand references are
    /// well-defined. We synthesize a base `<pq|rs>`-like tensor `g(p,q,r,s)`
    /// over all spin-orbitals and antisymmetrize `<pq||rs> = g - g_swap_rs`.
    fn synthetic_so_eris(no: usize, nv: usize) -> SpinOrbitalEris {
        let nmo = no + nv;
        // A base tensor h(p,q,r,s) symmetric under (p,r)<->(q,s) bra-ket swap so
        // the antisymmetrized <pq||rs> = h(p,r,q,s) - h(p,s,q,r) carries the
        // genuine permutational antisymmetry. Index over ALL nmo orbitals.
        let h = |p: usize, q: usize, r: usize, s: usize| -> f64 {
            let a = p.min(r) * nmo + p.max(r);
            let b = q.min(s) * nmo + q.max(s);
            let lo = a.min(b);
            let hi = a.max(b);
            0.02 + ((lo * 31 + hi) % 17) as f64 * 0.003
        };
        // antisymmetrized <pq||rs> over global orbital indices.
        let asym = |p: usize, q: usize, r: usize, s: usize| h(p, q, r, s) - h(p, q, s, r);
        let o = |i: usize| i; // occ spin-orbitals 0..no
        let vv = |a: usize| no + a; // vir spin-orbitals no..nmo

        let mut oovv = vec![0.0; no * no * nv * nv];
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        oovv[oovv_idx(no, nv, i, j, a, b)] = asym(o(i), o(j), vv(a), vv(b));
                    }
                }
            }
        }
        let mut oooo = vec![0.0; no * no * no * no];
        for i in 0..no {
            for j in 0..no {
                for k in 0..no {
                    for l in 0..no {
                        oooo[oooo_idx(no, i, j, k, l)] = asym(o(i), o(j), o(k), o(l));
                    }
                }
            }
        }
        let mut vvvv = vec![0.0; nv * nv * nv * nv];
        for a in 0..nv {
            for b in 0..nv {
                for c in 0..nv {
                    for d in 0..nv {
                        vvvv[vvvv_idx(nv, a, b, c, d)] = asym(vv(a), vv(b), vv(c), vv(d));
                    }
                }
            }
        }
        let mut ovvo = vec![0.0; no * nv * nv * no];
        for i in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    for j in 0..no {
                        ovvo[ovvo_idx(no, nv, i, a, b, j)] = asym(o(i), vv(a), vv(b), o(j));
                    }
                }
            }
        }
        let mut ooov = vec![0.0; no * no * no * nv];
        for i in 0..no {
            for j in 0..no {
                for k in 0..no {
                    for a in 0..nv {
                        ooov[ooov_idx(no, nv, i, j, k, a)] = asym(o(i), o(j), o(k), vv(a));
                    }
                }
            }
        }
        let mut ovvv = vec![0.0; no * nv * nv * nv];
        for i in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    for c in 0..nv {
                        ovvv[ovvv_idx(nv, i, a, b, c)] = asym(o(i), vv(a), vv(b), vv(c));
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

    fn synthetic_amps(no: usize, nv: usize) -> (Vec<f64>, Vec<f64>) {
        let t1: Vec<f64> = (0..no * nv).map(|i| 0.01 + ((i % 5) as f64) * 0.007).collect();
        // t2 must be antisymmetric in i<->j and a<->b (spin-orbital amplitudes).
        let mut t2 = vec![0.0_f64; no * no * nv * nv];
        let raw = |i: usize, j: usize, a: usize, b: usize| -> f64 {
            0.003 + (((i * 7 + j * 5 + a * 3 + b) % 11) as f64) * 0.002
        };
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        // Antisymmetrize: 0.25*(r[ijab]-r[jiab]-r[ijba]+r[jiba]).
                        let v = 0.25
                            * (raw(i, j, a, b) - raw(j, i, a, b) - raw(i, j, b, a)
                                + raw(j, i, b, a));
                        t2[t2_idx(no, nv, i, j, a, b)] = v;
                    }
                }
            }
        }
        (t1, t2)
    }

    /// make_tau: longhand element-for-element reference (spin-orbital form).
    #[test]
    fn make_tau_matches_longhand() {
        for (no, nv) in [(2usize, 2usize), (2, 3)] {
            let eris = synthetic_so_eris(no, nv);
            let (t1, t2) = synthetic_amps(no, nv);
            let tau = make_tau(&t1, &t2, &eris).expect("tau");
            for i in 0..no {
                for j in 0..no {
                    for a in 0..nv {
                        for b in 0..nv {
                            let p = t2_idx(no, nv, i, j, a, b);
                            let want = t2[p]
                                + t1[i * nv + a] * t1[j * nv + b]
                                - t1[i * nv + b] * t1[j * nv + a];
                            assert!((tau[p] - want).abs() < 1e-14, "tau[{i}{j}{a}{b}]");
                        }
                    }
                }
            }
        }
    }

    /// f_vv / f_oo / f_ov: independent scalar-fold longhand references on a
    /// 2-occ/2-vir spin-orbital block (Stanton Eqs. (3)-(5)).
    #[test]
    fn one_body_intermediates_match_longhand() {
        let (no, nv) = (2usize, 2usize);
        let eris = synthetic_so_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let nmo = no + nv;
        let taut = make_tau_tilde(&t1, &t2, &eris).expect("taut");
        let fk = |p: usize, q: usize| eris.fock[p * nmo + q];
        let t1e = |i: usize, a: usize| t1[i * nv + a];

        // f_vv
        let fae = f_vv(&t1, &t2, &eris).expect("fvv");
        for a in 0..nv {
            for e in 0..nv {
                let mut acc = 0.0;
                if a != e {
                    acc += fk(no + a, no + e);
                }
                for m in 0..no {
                    acc -= 0.5 * fk(m, no + e) * t1e(m, a);
                }
                for m in 0..no {
                    for f in 0..nv {
                        acc += t1e(m, f) * eris.ovvv[ovvv_idx(nv, m, a, f, e)];
                    }
                }
                for m in 0..no {
                    for n in 0..no {
                        for f in 0..nv {
                            acc -= 0.5
                                * taut[t2_idx(no, nv, m, n, a, f)]
                                * eris.oovv[oovv_idx(no, nv, m, n, e, f)];
                        }
                    }
                }
                assert!((fae[a * nv + e] - acc).abs() < 1e-12, "Fvv[{a}{e}]");
            }
        }

        // f_oo
        let fmi = f_oo(&t1, &t2, &eris).expect("foo");
        for m in 0..no {
            for i in 0..no {
                let mut acc = 0.0;
                if m != i {
                    acc += fk(m, i);
                }
                for e in 0..nv {
                    acc += 0.5 * t1e(i, e) * fk(m, no + e);
                }
                for n in 0..no {
                    for e in 0..nv {
                        acc += t1e(n, e) * eris.ooov[ooov_idx(no, nv, m, n, i, e)];
                    }
                }
                for n in 0..no {
                    for e in 0..nv {
                        for f in 0..nv {
                            acc += 0.5
                                * taut[t2_idx(no, nv, i, n, e, f)]
                                * eris.oovv[oovv_idx(no, nv, m, n, e, f)];
                        }
                    }
                }
                assert!((fmi[m * no + i] - acc).abs() < 1e-12, "Foo[{m}{i}]");
            }
        }

        // f_ov
        let fme = f_ov(&t1, &t2, &eris).expect("fov");
        for m in 0..no {
            for e in 0..nv {
                let mut acc = fk(m, no + e);
                for n in 0..no {
                    for f in 0..nv {
                        acc += t1e(n, f) * eris.oovv[oovv_idx(no, nv, m, n, e, f)];
                    }
                }
                assert!((fme[m * nv + e] - acc).abs() < 1e-12, "Fov[{m}{e}]");
            }
        }
    }

    /// The heaviest intermediate: w_vvvv (Stanton Eq. (7)) longhand on a 2x2
    /// spin-orbital block, and w_vvvv_into matches the owned wrapper.
    #[test]
    fn w_vvvv_matches_longhand_and_into() {
        let (no, nv) = (2usize, 2usize);
        let eris = synthetic_so_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let tau = make_tau(&t1, &t2, &eris).expect("tau");
        let w = w_vvvv(&t1, &t2, &eris).expect("wvvvv");
        let t1e = |m: usize, a: usize| t1[m * nv + a];
        for a in 0..nv {
            for b in 0..nv {
                for e in 0..nv {
                    for f in 0..nv {
                        let mut acc = eris.vvvv[vvvv_idx(nv, a, b, e, f)];
                        for m in 0..no {
                            acc += t1e(m, b) * eris.ovvv[ovvv_idx(nv, m, a, e, f)];
                            acc -= t1e(m, a) * eris.ovvv[ovvv_idx(nv, m, b, e, f)];
                        }
                        for m in 0..no {
                            for n in 0..no {
                                acc += 0.25
                                    * tau[t2_idx(no, nv, m, n, a, b)]
                                    * eris.oovv[oovv_idx(no, nv, m, n, e, f)];
                            }
                        }
                        assert!(
                            (w[vvvv_idx(nv, a, b, e, f)] - acc).abs() < 1e-12,
                            "Wvvvv[{a}{b}{e}{f}]"
                        );
                    }
                }
            }
        }
        // _into path matches the owned wrapper.
        let mut buf = vec![0.0; nv * nv * nv * nv];
        w_vvvv_into(&t1, &t2, &eris, &mut buf).expect("into");
        assert_eq!(w, buf);
    }

    /// w_oooo + w_ovvo run on a 2x3 block and produce finite tensors of the
    /// right shape (the longhand of the full chain is exercised by the
    /// converged-energy smoke).
    #[test]
    fn two_body_intermediates_finite() {
        let (no, nv) = (2usize, 3usize);
        let eris = synthetic_so_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let woooo = w_oooo(&t1, &t2, &eris).expect("woooo");
        let wovvo = w_ovvo(&t1, &t2, &eris).expect("wovvo");
        assert_eq!(woooo.len(), no * no * no * no);
        assert_eq!(wovvo.len(), no * nv * nv * no);
        for &x in woooo.iter().chain(wovvo.iter()) {
            assert!(x.is_finite());
        }
    }

    /// Shape validation: a wrong-length block returns ShapeMismatch, not a
    /// panic (T-06-04-SHAPE).
    #[test]
    fn wrong_shape_returns_error_not_panic() {
        let (no, nv) = (2usize, 2usize);
        let mut eris = synthetic_so_eris(no, nv);
        eris.oovv.pop();
        let (t1, t2) = synthetic_amps(no, nv);
        let err = f_oo(&t1, &t2, &eris).expect_err("must error on bad shape");
        assert!(matches!(err, CcsdError::ShapeMismatch { .. }));
    }
}
