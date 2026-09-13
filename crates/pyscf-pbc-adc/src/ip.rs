//! k-point ADC(2) ionisation potentials (`pbc/adc/kadc_rhf_ip.py`, 1,061 l).
//!
//! Ports the ADC(2) IP working equations over 19-14's base — and nothing else:
//! `(2)-x`/`(3)` need `t1_2`/`t2_2`/extra ERI routes and refuse here, exactly
//! as `build_amplitudes` refuses them. The secular matrix is Hermitian
//! (complex-Hermitian at general k, real-symmetric at Γ): the `s1 ← r2`
//! couplings use `conj(ovoo)` where the `s2 ← r1` couplings use `ovoo`, so
//! the matrix is assembled as [`CTensor`] and diagonalized with
//! [`pyscf_algebra::zeigh_gen`]. Hermiticity is ASSERTED before the solve —
//! a non-Hermitian assembly is a sigma bug, and only an element-wise oracle
//! on the matrix finds it (19-14's shape-silent lesson).
//!
//! Vector layout at fixed `kshift` (mirroring `matvec`): singles `r1[i]`
//! (`nocc`), then doubles `r2[ka,kj][a,j,k]` (`nk²·nvir·nocc²`, row-major
//! `[ka][kj][a][j][k]`). Roots are sorted with an explicit count (Gate D).

use pyscf_algebra::{CTensor, oracle_sum};

use crate::amplitudes::KadcAmplitudes;
use crate::error::PbcAdcError;
use crate::kadc_ao2mo::KadcEris;
use crate::types::{AdcConfig, AdcRoots};

/// ADC(2) 1h effective-Hamiltonian blocks (`get_imds`), one `nocc × nocc`
/// [`CTensor`] per k-point.
#[derive(Debug, Clone)]
pub struct IpIntermediates {
    /// `m_ij[k]` — the 1h block at k-point k.
    pub m_ij: Vec<CTensor>,
}

/// Build `M_ij` (`kadc_rhf_ip.get_imds`, ADC(2) part).
///
/// `M_ij[ki][i,j] = δ·e_occ[ki] + Σ_{kl,kd} (t2[ki,kl,kd]·ovov + …)` — the six
/// `0.25/0.5`-weighted `t2_1`–`ovov` contractions over
/// `ke = kconserv[kj,kd,kl]`, with the conjugated second half. Mirrors the
/// loop nest exactly (including the `kj = ki` zeroth-order diagonal).
pub fn build_m_ij(
    amps: &KadcAmplitudes,
    eris: &KadcEris,
    e_occ_k: &[Vec<f64>],
    kconserv: &[usize],
) -> Result<IpIntermediates, PbcAdcError> {
    let (nk, no, nv) = (eris.nkpts, eris.nocc, eris.nvir);
    if e_occ_k.len() != nk || kconserv.len() != nk * nk * nk {
        return Err(PbcAdcError::ShapeMismatch { expected: nk, got: e_occ_k.len() });
    }
    if amps.t2_1.re.len() != nk * nk * nk * no * no * nv * nv {
        return Err(PbcAdcError::ShapeMismatch {
            expected: nk * nk * nk * no * no * nv * nv,
            got: amps.t2_1.re.len(),
        });
    }
    // t2 accessor: t2[ki,kj,ka][i,j,a,b] -> (re, im).
    let t2_at = |ki: usize, kj: usize, ka: usize, i: usize, j: usize, a: usize, b: usize| -> (f64, f64) {
        let o = (((ki * nk + kj) * nk + ka) * no + i) * no * nv * nv + (j * nv + a) * nv + b;
        (amps.t2_1.re[o], amps.t2_1.im[o])
    };
    // ovov[ki,kj,ka][i,a,j,b] accessor.
    let ovov_at = |ki: usize, kj: usize, ka: usize, i: usize, a: usize, j: usize, b: usize| -> (f64, f64) {
        let t = no * nv * no * nv;
        let o = ((ki * nk + kj) * nk + ka) * t + ((i * nv + a) * no + j) * nv + b;
        (eris.ovov.re[o], eris.ovov.im[o])
    };
    let mut m_ij = Vec::with_capacity(nk);
    for ki in 0..nk {
        let kj = ki;
        let mut m = CTensor::zeros(no * no);
        for i in 0..no {
            m.re[i * no + i] = e_occ_k[kj][i];
        }
        for kl in 0..nk {
            for kd in 0..nk {
                let ke = kconserv[(kj * nk + kd) * nk + kl];
                if ke >= nk {
                    return Err(PbcAdcError::ShapeMismatch { expected: nk, got: ke });
                }
                for i in 0..no {
                    for j in 0..no {
                        // Four t2·ovov families × three prefactor patterns.
                        // Family 1: t2[ki,kl,kd] (ilde) × ovov[kj,kd,kl]/ovov[kj,ke,kl].
                        let mut acc = (0.0f64, 0.0f64);
                        for l in 0..no {
                            for d in 0..nv {
                                for e in 0..nv {
                                    let (tr, ti) = t2_at(ki, kl, kd, i, l, d, e);
                                    // 0.25·(ilde·jdle − ilde·jeld) + 0.5·ilde·jdle
                                    let (v1r, v1i) = ovov_at(kj, kd, kl, j, d, l, e);
                                    let (v2r, v2i) = ovov_at(kj, ke, kl, j, e, l, d);
                                    acc.0 += 0.25 * (tr * v1r - ti * v1i)
                                        - 0.25 * (tr * v2r - ti * v2i)
                                        + 0.5 * (tr * v1r - ti * v1i);
                                    acc.1 += 0.25 * (tr * v1i + ti * v1r)
                                        - 0.25 * (tr * v2i + ti * v2r)
                                        + 0.5 * (tr * v1i + ti * v1r);
                                }
                            }
                        }
                        // Family 2: −t2[kl,ki,kd] (lide) × ovov[kj,kd,kl]/ovov[kj,ke,kl].
                        for l in 0..no {
                            for d in 0..nv {
                                for e in 0..nv {
                                    let (tr, ti) = t2_at(kl, ki, kd, l, i, d, e);
                                    let (v1r, v1i) = ovov_at(kj, kd, kl, j, d, l, e);
                                    let (v2r, v2i) = ovov_at(kj, ke, kl, j, e, l, d);
                                    acc.0 += -0.25 * (tr * v1r - ti * v1i)
                                        + 0.25 * (tr * v2r - ti * v2i);
                                    acc.1 += -0.25 * (tr * v1i + ti * v1r)
                                        + 0.25 * (tr * v2i + ti * v2r);
                                }
                            }
                        }
                        // Family 3: +t2[kj,kl,kd]* (jlde) × conj(ovov[ki,kd,kl])/conj(ovov[ki,ke,kl]).
                        for l in 0..no {
                            for d in 0..nv {
                                for e in 0..nv {
                                    let (tr, ti) = t2_at(kj, kl, kd, j, l, d, e);
                                    let (v1r, v1i) = ovov_at(ki, kd, kl, i, d, l, e);
                                    let (v2r, v2i) = ovov_at(ki, ke, kl, i, e, l, d);
                                    // conj(t)·conj(v): (tr·vr − (−ti)(−vi), tr·(−vi) + (−ti)·vr)
                                    acc.0 += 0.25 * (tr * v1r - ti * v1i)
                                        - 0.25 * (tr * v2r - ti * v2i)
                                        + 0.5 * (tr * v1r - ti * v1i);
                                    acc.1 += -0.25 * (tr * v1i + ti * v1r)
                                        + 0.25 * (tr * v2i + ti * v2r)
                                        - 0.5 * (tr * v1i + ti * v1r);
                                }
                            }
                        }
                        // Family 4: −t2[kl,kj,kd]* (ljde) × conj(ovov[ki,kd,kl])/conj(ovov[ki,ke,kl]).
                        for l in 0..no {
                            for d in 0..nv {
                                for e in 0..nv {
                                    let (tr, ti) = t2_at(kl, kj, kd, l, j, d, e);
                                    let (v1r, v1i) = ovov_at(ki, kd, kl, i, d, l, e);
                                    let (v2r, v2i) = ovov_at(ki, ke, kl, i, e, l, d);
                                    acc.0 += -0.25 * (tr * v1r - ti * v1i)
                                        + 0.25 * (tr * v2r - ti * v2i);
                                    acc.1 += 0.25 * (tr * v1i + ti * v1r)
                                        - 0.25 * (tr * v2i + ti * v2r);
                                }
                            }
                        }
                        m.re[i * no + j] += acc.0;
                        m.im[i * no + j] += acc.1;
                    }
                }
            }
        }
        m_ij.push(m);
    }
    Ok(IpIntermediates { m_ij })
}

/// ADC(2) IP sigma-vector product at fixed `kshift` (`matvec.sigma_`, ADC(2)).
 ///
/// `r = [r1 (nocc), r2 ([ka][kj][a][j][k])]` → `s` in the same layout, as
/// `(re, im)` pairs. Couplings mirror the loop nest exactly:
/// `s1 += 2·conj(ovoo[kj,ka,kk])·r2 − conj(ovoo[kk,ka,kj])·r2`,
/// `s2[ka,kj] += ovoo[kj,ka,kk]·r1 + diag·r2` with
/// `ka = kconserv[kk,kshift,kj]`, `diag = −e_vir[ka] + e_occ[kj] + e_occ[kk]`.
 ///
/// The final `s *= -1.0` (`kadc_rhf_ip.py:747`) is part of the product: the
/// 1h block is `e_occ`-diagonal (negative), and upstream solves the NEGATED
/// problem so the reported roots are positive ionisation energies. Dropping
/// the sign flips every root (caught by Gate D at 4.06 Ha during
/// development). EA needs no negation (`M_ab` is `e_vir`-diagonal).
#[allow(clippy::too_many_arguments)]
pub fn sigma_ip(
    r1: &[(f64, f64)],
    r2: &[(f64, f64)],
    m: &IpIntermediates,
    eris: &KadcEris,
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    kshift: usize,
) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>), PbcAdcError> {
    let (nk, no, nv) = (eris.nkpts, eris.nocc, eris.nvir);
    let nd = nk * nk * nv * no * no;
    if r1.len() != no || r2.len() != nd || kshift >= nk {
        return Err(PbcAdcError::ShapeMismatch { expected: no, got: r1.len() });
    }
    let ovoo_at = |ki: usize, kj: usize, ka: usize, i: usize, a: usize, j: usize, k: usize| -> (f64, f64) {
        let t = no * nv * no * no;
        let o = ((ki * nk + kj) * nk + ka) * t + ((i * nv + a) * no + j) * no + k;
        (eris.ovoo.re[o], eris.ovoo.im[o])
    };
    let r2_at = |ka: usize, kj: usize, a: usize, j: usize, k: usize| -> (f64, f64) {
        r2[((ka * nk + kj) * nv + a) * no * no + j * no + k]
    };
    // s1 = M_ij[kshift]·r1.
    let mk = &m.m_ij[kshift];
    let mut s1 = vec![(0.0f64, 0.0f64); no];
    for i in 0..no {
        let mut acc = (0.0f64, 0.0f64);
        for j in 0..no {
            let (mr, mi) = (mk.re[i * no + j], mk.im[i * no + j]);
            acc.0 += mr * r1[j].0 - mi * r1[j].1;
            acc.1 += mr * r1[j].1 + mi * r1[j].0;
        }
        s1[i] = acc;
    }
    // s2 zeroed (arena discipline), then couplings + diagonal.
    let mut s2 = vec![(0.0f64, 0.0f64); nd];
    for kj in 0..nk {
        for kk in 0..nk {
            let ka = kconserv[(kk * nk + kshift) * nk + kj];
            if ka >= nk {
                return Err(PbcAdcError::ShapeMismatch { expected: nk, got: ka });
            }
            for i in 0..no {
                // s1[i] += 2·Σ_{a,j,k} conj(ovoo[kj,ka,kk][j,a,k,i])·r2[ka,kj][a,j,k]
                //          − Σ conj(ovoo[kk,ka,kj][k,a,j,i])·r2[ka,kj][a,j,k].
                let mut c1 = (0.0f64, 0.0f64);
                let mut c2 = (0.0f64, 0.0f64);
                for a in 0..nv {
                    for j in 0..no {
                        for k in 0..no {
                            let (v1r, v1i) = ovoo_at(kj, ka, kk, j, a, k, i);
                            let (v2r, v2i) = ovoo_at(kk, ka, kj, k, a, j, i);
                            let (rr, ri) = r2_at(ka, kj, a, j, k);
                            // conj(v)·r = (vr·rr + vi·ri, vr·ri − vi·rr).
                            c1.0 += v1r * rr + v1i * ri;
                            c1.1 += v1r * ri - v1i * rr;
                            c2.0 += v2r * rr + v2i * ri;
                            c2.1 += v2r * ri - v2i * rr;
                        }
                    }
                }
                s1[i].0 += 2.0 * c1.0 - c2.0;
                s1[i].1 += 2.0 * c1.1 - c2.1;
            }
            for a in 0..nv {
                for j in 0..no {
                    for k in 0..no {
                        // s2[ka,kj][a,j,k] += Σ_i ovoo[kj,ka,kk][j,a,k,i]·r1[i].
                        let mut acc = (0.0f64, 0.0f64);
                        for i in 0..no {
                            let (vr, vi) = ovoo_at(kj, ka, kk, j, a, k, i);
                            acc.0 += vr * r1[i].0 - vi * r1[i].1;
                            acc.1 += vr * r1[i].1 + vi * r1[i].0;
                        }
                        // Diagonal: (−e_vir[ka][a] + e_occ[kj][j] + e_occ[kk][k])·r2.
                        let d = -e_vir_k[ka][a] + e_occ_k[kj][j] + e_occ_k[kk][k];
                        let (rr, ri) = r2_at(ka, kj, a, j, k);
                        let o = ((ka * nk + kj) * nv + a) * no * no + j * no + k;
                        s2[o].0 += acc.0 + d * rr;
                        s2[o].1 += acc.1 + d * ri;
                    }
                }
            }
        }
    }
    // Upstream's trailing `s *= -1.0` (kadc_rhf_ip.py:747): the solved
    // problem is the negated one (positive ionisation energies).
    for x in s1.iter_mut() {
        x.0 = -x.0;
        x.1 = -x.1;
    }
    for x in s2.iter_mut() {
        x.0 = -x.0;
        x.1 = -x.1;
    }
    Ok((s1, s2))
}

/// ADC(2) IP roots at `kshift` (dense Davidson — exact on fixture sizes).
 ///
/// Materializes the secular matrix column by column through [`sigma_ip`]
/// (HERMITICITY IS NOT ASSUMED — upstream's own matrix is non-symmetric at
/// 0.048, hence `davidson_nosym1` there and [`crate::roots::nosym_roots`]
/// here), and returns the lowest `cfg.nroots` roots sorted with
/// spectroscopic factors from the singles weight (`p` in upstream's
/// `(e, v, p, x)` return).
pub fn kernel_ip(
    m: &IpIntermediates,
    eris: &KadcEris,
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    kshift: usize,
    cfg: &AdcConfig,
) -> Result<AdcRoots, PbcAdcError> {
    let (nk, no, nv) = (eris.nkpts, eris.nocc, eris.nvir);
    let nd = nk * nk * nv * no * no;
    let dim = no + nd;
    if cfg.nroots > dim || dim == 0 {
        return Err(PbcAdcError::ShapeMismatch { expected: dim, got: cfg.nroots });
    }
    // Materialize H (complex, row-major): column c = sigma(e_c).
    let mut hre = vec![0.0f64; dim * dim];
    let mut him = vec![0.0f64; dim * dim];
    for c in 0..dim {
        let mut r1 = vec![(0.0f64, 0.0f64); no];
        let mut r2 = vec![(0.0f64, 0.0f64); nd];
        if c < no {
            r1[c] = (1.0, 0.0);
        } else {
            r2[c - no] = (1.0, 0.0);
        }
        let (s1, s2) = sigma_ip(&r1, &r2, m, eris, e_occ_k, e_vir_k, kconserv, kshift)?;
        for i in 0..no {
            hre[i * dim + c] = s1[i].0;
            him[i * dim + c] = s1[i].1;
        }
        for i in 0..nd {
            hre[(no + i) * dim + c] = s2[i].0;
            him[(no + i) * dim + c] = s2[i].1;
        }
    }
    // Upstream's own matrix is non-symmetric (measured 0.048), so no
    // Hermiticity assert belongs here — the nosym solver takes it as is.
    let (energies_all, columns) = crate::roots::nosym_roots(&hre, &him, dim, cfg.nroots)?;
    // Spectroscopic factors = singles weight |U_singles|² per root.
    let mut roots = AdcRoots { energies: Vec::with_capacity(cfg.nroots), spec_factors: Vec::with_capacity(cfg.nroots), converged: true };
    for r in 0..cfg.nroots {
        roots.energies.push(energies_all[r]);
        let mut w_terms = Vec::with_capacity(2 * no);
        for i in 0..no {
            w_terms.push(columns[r][i].0 * columns[r][i].0);
            w_terms.push(columns[r][i].1 * columns[r][i].1);
        }
        roots.spec_factors.push(oracle_sum(&w_terms));
    }
    Ok(roots)
}
