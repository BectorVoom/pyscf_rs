//! k-point ADC(2) electron affinities (`pbc/adc/kadc_rhf_ea.py`, 1,324 l).
//!
//! Structurally 19-15 with the particle-attached manifold, over the same
//! 19-14 base — **no fork of the base** (19-16). Only ADC(2) executes;
//! `(2)-x`/`(3)` refuse here, as in [`crate::ip`].
//!
//! Vector layout at fixed `kshift`: singles `r1[a]` (`nvir`), then doubles
//! `r2[ki,kb][i,b,c]` (`nk²·nocc·nvir²`, row-major `[ki][kb][i][b][c]`).
//! The secular matrix is Hermitian (the `s1 ← r2` couplings conjugate the
//! same `ovvv` blocks the `s2 ← r1` couplings use plain), assembled as
//! [`CTensor`] and solved with [`pyscf_algebra::zeigh_gen`] after a
//! Hermiticity assert — the same sigma-bug tripwire as IP.
//!
//! Incore branch only (`eris.ovvv` present): the DF-`ovvv` chunk path is
//! 19-17's `dfadc` territory, not this manifold's.

use pyscf_algebra::{CTensor, oracle_sum};

use crate::amplitudes::KadcAmplitudes;
use crate::error::PbcAdcError;
use crate::kadc_ao2mo::KadcEris;
use crate::types::{AdcConfig, AdcRoots};

/// ADC(2) 1p effective-Hamiltonian blocks (`get_imds`, ADC(2) part), one
/// `nvir × nvir` [`CTensor`] per k-point.
#[derive(Debug, Clone)]
pub struct EaIntermediates {
    /// `m_ab[k]` — the 1p block at k-point k.
    pub m_ab: Vec<CTensor>,
}

/// Build `M_ab` (ADC(2) part).
///
/// `M_ab[ka] = diag(e_vir[ka])` plus the four `t2_1`–`ovov` families over
/// `kd = kconserv[kl,ka,km]` (two plain, two conjugated), mirroring the loop
/// nest exactly. `kb = ka` throughout (the loop header binds it).
pub fn build_m_ab(
    amps: &KadcAmplitudes,
    eris: &KadcEris,
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
) -> Result<EaIntermediates, PbcAdcError> {
    let (nk, no, nv) = (eris.nkpts, eris.nocc, eris.nvir);
    if e_vir_k.len() != nk || kconserv.len() != nk * nk * nk {
        return Err(PbcAdcError::ShapeMismatch { expected: nk, got: e_vir_k.len() });
    }
    let t2_at = |ki: usize, kj: usize, ka: usize, i: usize, j: usize, a: usize, b: usize| -> (f64, f64) {
        let o = (((ki * nk + kj) * nk + ka) * no + i) * no * nv * nv + (j * nv + a) * nv + b;
        (amps.t2_1.re[o], amps.t2_1.im[o])
    };
    let ovov_at = |ki: usize, kj: usize, ka: usize, i: usize, a: usize, j: usize, b: usize| -> (f64, f64) {
        let t = no * nv * no * nv;
        let o = ((ki * nk + kj) * nk + ka) * t + ((i * nv + a) * no + j) * nv + b;
        (eris.ovov.re[o], eris.ovov.im[o])
    };
    let mut m_ab = Vec::with_capacity(nk);
    for ka in 0..nk {
        let kb = ka;
        let mut m = CTensor::zeros(nv * nv);
        for a in 0..nv {
            m.re[a * nv + a] = e_vir_k[ka][a];
        }
        for kl in 0..nk {
            for km in 0..nk {
                let kd = kconserv[(kl * nk + ka) * nk + km];
                if kd >= nk {
                    return Err(PbcAdcError::ShapeMismatch { expected: nk, got: kd });
                }
                for a in 0..nv {
                    for b in 0..nv {
                        let mut acc = (0.0f64, 0.0f64);
                        for m_ in 0..no {
                            for l in 0..no {
                                for d in 0..nv {
                                    // F1: +0.25 t2[km,kl,ka][m,l,a,d]·V[kl,kb,km][l,b,m,d]
                                    let (tr, ti) = t2_at(km, kl, ka, m_, l, a, d);
                                    let (vr, vi) = ovov_at(kl, kb, km, l, b, m_, d);
                                    acc.0 += 0.25 * (tr * vr - ti * vi);
                                    acc.1 += 0.25 * (tr * vi + ti * vr);
                                    // F1x: −0.25 t2[km,kl,ka][m,l,a,d]·V[kl,kd,km][l,d,m,b]
                                    let (wr, wi) = ovov_at(kl, kd, km, l, d, m_, b);
                                    acc.0 -= 0.25 * (tr * wr - ti * wi);
                                    acc.1 -= 0.25 * (tr * wi + ti * wr);
                                    // F2: −0.25 t2[kl,km,ka][l,m,a,d]·V[kl,kb,km][l,b,m,d]
                                    let (ur, ui) = t2_at(kl, km, ka, l, m_, a, d);
                                    acc.0 -= 0.25 * (ur * vr - ui * vi);
                                    acc.1 -= 0.25 * (ur * vi + ui * vr);
                                    // F2b: −0.5 t2[kl,km,ka][l,m,a,d]·V[kl,kb,km][l,b,m,d]
                                    acc.0 -= 0.5 * (ur * vr - ui * vi);
                                    acc.1 -= 0.5 * (ur * vi + ui * vr);
                                    // F2x: +0.25 t2[kl,km,ka][l,m,a,d]·V[kl,kd,km][l,d,m,b]
                                    acc.0 += 0.25 * (ur * wr - ui * wi);
                                    acc.1 += 0.25 * (ur * wi + ui * wr);
                                }
                            }
                        }
                        // Conjugated families (conj(t)·conj(V): real part as
                        // the plain product, imaginary part negated).
                        for m_ in 0..no {
                            for l in 0..no {
                                for d in 0..nv {
                                    // F3: −0.25 conj(t2[kl,km,kb][l,m,b,d])·conj(V[kl,ka,km][l,a,m,d])
                                    let (tr, ti) = t2_at(kl, km, kb, l, m_, b, d);
                                    let (vr, vi) = ovov_at(kl, ka, km, l, a, m_, d);
                                    acc.0 -= 0.25 * (tr * vr - ti * vi);
                                    acc.1 += 0.25 * (tr * vi + ti * vr);
                                    // F3b: −0.5, same pattern.
                                    acc.0 -= 0.5 * (tr * vr - ti * vi);
                                    acc.1 += 0.5 * (tr * vi + ti * vr);
                                    // F3x: +0.25 conj(t2)·conj(V[kl,kd,km][l,d,m,a]).
                                    let (wr, wi) = ovov_at(kl, kd, km, l, d, m_, a);
                                    acc.0 += 0.25 * (tr * wr - ti * wi);
                                    acc.1 -= 0.25 * (tr * wi + ti * wr);
                                    // F4: +0.25 conj(t2[km,kl,kb][m,l,b,d])·conj(V[kl,ka,km][l,a,m,d])
                                    let (ur, ui) = t2_at(km, kl, kb, m_, l, b, d);
                                    acc.0 += 0.25 * (ur * vr - ui * vi);
                                    acc.1 -= 0.25 * (ur * vi + ui * vr);
                                    // F4x: −0.25 conj(t2)·conj(V[kl,kd,km][l,d,m,a]).
                                    acc.0 -= 0.25 * (ur * wr - ui * wi);
                                    acc.1 += 0.25 * (ur * wi + ui * wr);
                                }
                            }
                        }
                        m.re[a * nv + b] += acc.0;
                        m.im[a * nv + b] += acc.1;
                    }
                }
            }
        }
        m_ab.push(m);
    }
    Ok(EaIntermediates { m_ab })
}

/// ADC(2) EA sigma-vector product at fixed `kshift` (incore `ovvv` branch).
///
/// `s1[a] = M_ab[kshift]·r1 + 2·Σ conj(ovvv[ki,kc,kshift])·r2`,
/// `s2[ki,kb] += ovvv[ki,kc,kshift]·r1 + diag·r2` with
/// `ki = kconserv[kb,kshift,kc]`,
/// `diag = −e_occ[ki][i] + e_vir[kb][b] + e_vir[kc][c]`.
/// Ports the incore (`eris.ovvv` present) branch literally — including its
/// single-`ovvv`-fetch shape (the DF chunk branch is 19-17's).
#[allow(clippy::too_many_arguments)]
pub fn sigma_ea(
    r1: &[(f64, f64)],
    r2: &[(f64, f64)],
    m: &EaIntermediates,
    eris: &KadcEris,
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    kshift: usize,
) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>), PbcAdcError> {
    let (nk, no, nv) = (eris.nkpts, eris.nocc, eris.nvir);
    let nd = nk * nk * no * nv * nv;
    if r1.len() != nv || r2.len() != nd || kshift >= nk {
        return Err(PbcAdcError::ShapeMismatch { expected: nv, got: r1.len() });
    }
    let ovvv_at = |ki: usize, kj: usize, ka: usize, i: usize, a: usize, b: usize, c: usize| -> (f64, f64) {
        let t = no * nv * nv * nv;
        let o = ((ki * nk + kj) * nk + ka) * t + ((i * nv + a) * nv + b) * nv + c;
        (eris.ovvv.re[o], eris.ovvv.im[o])
    };
    let r2_at = |ki: usize, kb: usize, i: usize, b: usize, c: usize| -> (f64, f64) {
        r2[((ki * nk + kb) * no + i) * nv * nv + b * nv + c]
    };
    let mk = &m.m_ab[kshift];
    let mut s1 = vec![(0.0f64, 0.0f64); nv];
    for a in 0..nv {
        let mut acc = (0.0f64, 0.0f64);
        for b in 0..nv {
            let (mr, mi) = (mk.re[a * nv + b], mk.im[a * nv + b]);
            acc.0 += mr * r1[b].0 - mi * r1[b].1;
            acc.1 += mr * r1[b].1 + mi * r1[b].0;
        }
        s1[a] = acc;
    }
    let mut s2 = vec![(0.0f64, 0.0f64); nd];
    for kb in 0..nk {
        for kc in 0..nk {
            let ki = kconserv[(kb * nk + kshift) * nk + kc];
            if ki >= nk {
                return Err(PbcAdcError::ShapeMismatch { expected: nk, got: ki });
            }
            for a in 0..nv {
                // s1[a] += 2·Σ_{i,b,c} conj(ovvv[ki,kc,kshift][i,c,a,b])·r2[ki,kb][i,b,c].
                let mut acc = (0.0f64, 0.0f64);
                for i in 0..no {
                    for b in 0..nv {
                        for c in 0..nv {
                            let (vr, vi) = ovvv_at(ki, kc, kshift, i, c, a, b);
                            let (rr, ri) = r2_at(ki, kb, i, b, c);
                            acc.0 += vr * rr + vi * ri;
                            acc.1 += vr * ri - vi * rr;
                        }
                    }
                }
                s1[a].0 += 2.0 * acc.0;
                s1[a].1 += 2.0 * acc.1;
            }
            for i in 0..no {
                for b in 0..nv {
                    for c in 0..nv {
                        // s2[ki,kb][i,b,c] += Σ_a ovvv[ki,kc,kshift][i,c,a,b]·r1[a].
                        let mut acc = (0.0f64, 0.0f64);
                        for a in 0..nv {
                            let (vr, vi) = ovvv_at(ki, kc, kshift, i, c, a, b);
                            acc.0 += vr * r1[a].0 - vi * r1[a].1;
                            acc.1 += vr * r1[a].1 + vi * r1[a].0;
                        }
                        let d = -e_occ_k[ki][i] + e_vir_k[kb][b] + e_vir_k[kc][c];
                        let (rr, ri) = r2_at(ki, kb, i, b, c);
                        let o = ((ki * nk + kb) * no + i) * nv * nv + b * nv + c;
                        s2[o].0 += acc.0 + d * rr;
                        s2[o].1 += acc.1 + d * ri;
                    }
                }
            }
        }
    }
    Ok((s1, s2))
}

/// ADC(2) EA roots at `kshift` (dense Davidson — exact on fixture sizes).
///
/// Materializes the secular matrix through [`sigma_ea`] (non-symmetric, as
/// IP — same [`crate::roots::nosym_roots`] solve), returns the lowest
/// `cfg.nroots` roots sorted with singles-weight spectroscopic factors.
pub fn kernel_ea(
    m: &EaIntermediates,
    eris: &KadcEris,
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    kshift: usize,
    cfg: &AdcConfig,
) -> Result<AdcRoots, PbcAdcError> {
    let (nk, no, nv) = (eris.nkpts, eris.nocc, eris.nvir);
    let nd = nk * nk * no * nv * nv;
    let dim = nv + nd;
    if cfg.nroots > dim || dim == 0 {
        return Err(PbcAdcError::ShapeMismatch { expected: dim, got: cfg.nroots });
    }
    let mut hre = vec![0.0f64; dim * dim];
    let mut him = vec![0.0f64; dim * dim];
    for c in 0..dim {
        let mut r1 = vec![(0.0f64, 0.0f64); nv];
        let mut r2 = vec![(0.0f64, 0.0f64); nd];
        if c < nv {
            r1[c] = (1.0, 0.0);
        } else {
            r2[c - nv] = (1.0, 0.0);
        }
        let (s1, s2) = sigma_ea(&r1, &r2, m, eris, e_occ_k, e_vir_k, kconserv, kshift)?;
        for i in 0..nv {
            hre[i * dim + c] = s1[i].0;
            him[i * dim + c] = s1[i].1;
        }
        for i in 0..nd {
            hre[(nv + i) * dim + c] = s2[i].0;
            him[(nv + i) * dim + c] = s2[i].1;
        }
    }
    let (energies_all, columns) = crate::roots::nosym_roots(&hre, &him, dim, cfg.nroots)?;
    let mut roots = AdcRoots { energies: Vec::with_capacity(cfg.nroots), spec_factors: Vec::with_capacity(cfg.nroots), converged: true };
    for r in 0..cfg.nroots {
        roots.energies.push(energies_all[r]);
        let mut w_terms = Vec::with_capacity(2 * nv);
        for i in 0..nv {
            w_terms.push(columns[r][i].0 * columns[r][i].0);
            w_terms.push(columns[r][i].1 * columns[r][i].1);
        }
        roots.spec_factors.push(oracle_sum(&w_terms));
    }
    Ok(roots)
}
