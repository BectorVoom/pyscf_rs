//! k-point ADC amplitudes (`pbc/adc/kadc_rhf_amplitudes.py`, 346 l).
//!
//! Ports the first-order doubles `t2_1[ki,kj,ka] =
//! ovov[ki,ka,kj]*.conj().transpose(0,2,1,3) / eijab` and the ADC(2) energy
//! (`compute_energy`: `2·direct − exchange`, real part, `/nkpts`). Only
//! `adc(2)` is supported: `adc(2)-x`/`adc(3)` need `t1_2`/`t2_2` and land with
//! the IP/EA manifolds that consume them — requesting them here returns
//! [`PbcAdcError::NotYetImplemented`], mirroring upstream's
//! `NotImplementedError` on unknown methods.
//!
//! Complex (non-Γ) arithmetic is first-class: conjugation negates `im`, and
//! the energy takes the real part of explicitly accumulated complex products
//! (re/im reduced separately through `oracle_sum`, D-PBC-29 clause 2).

use pyscf_algebra::{CTensor, oracle_sum};

use crate::error::PbcAdcError;
use crate::kadc_ao2mo::KadcEris;

/// First-order ADC doubles (populated by [`t2_first_order`]).
#[derive(Debug, Clone)]
pub struct KadcAmplitudes {
    /// Number of k-points.
    pub nkpts: usize,
    /// Occupied / virtual counts.
    pub nocc: usize,
    /// Virtual count.
    pub nvir: usize,
    /// `[ki][kj][ka][i,j,a,b]` first-order doubles.
    pub t2_1: CTensor,
    /// Placeholder for second-order singles (ADC(2)-x/ADC(3) only).
    pub t1_2: Option<CTensor>,
}

impl Default for KadcAmplitudes {
    fn default() -> Self {
        Self {
            nkpts: 0,
            nocc: 0,
            nvir: 0,
            t2_1: CTensor::zeros(0),
            t1_2: None,
        }
    }
}

/// Build the first-order doubles over the ERI blocks.
///
/// `e_occ_k[k]`/`e_vir_k[k]` are the per-k occupied/virtual energies;
/// `kconserv` is flat `nkpts³` (`[p][q][r]` at `(p·nk+q)·nk+r`).
/// `kb = kconserv[ki,ka,kj]`,
/// `eijab = (e_occ[ki][i] − e_vir[ka][a]) + (e_occ[kj][j] − e_vir[kb][b])` —
/// the `_get_epq(..., fac=[1.0,-1.0])` sign convention (negative gaps; the
/// correlation energy comes out negative as it must). A positive-gap
/// denominator flips every t2 sign and the energy with it — same shape,
/// caught by the live-block comparison in `tests/ip.rs` during development.
pub fn t2_first_order(
    eris: &KadcEris,
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
) -> Result<KadcAmplitudes, PbcAdcError> {
    let (nk, no, nv) = (eris.nkpts, eris.nocc, eris.nvir);
    if e_occ_k.len() != nk || e_vir_k.len() != nk || kconserv.len() != nk * nk * nk {
        return Err(PbcAdcError::ShapeMismatch { expected: nk, got: e_occ_k.len() });
    }
    let mut t2 = CTensor::zeros(nk * nk * nk * no * no * nv * nv);
    t2.re.fill(0.0);
    t2.im.fill(0.0);
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let kb = kconserv[(ki * nk + ka) * nk + kj];
                if kb >= nk {
                    return Err(PbcAdcError::ShapeMismatch { expected: nk, got: kb });
                }
                // ovov[ki,ka,kj] slice, [i,a,j,b].
                let t = no * nv * no * nv;
                let o = ((ki * nk + ka) * nk + kj) * t;
                for i in 0..no {
                    for a in 0..nv {
                        let eia = e_occ_k[ki][i] - e_vir_k[ka][a];
                        for j in 0..no {
                            for b in 0..nv {
                                let ejb = e_occ_k[kj][j] - e_vir_k[kb][b];
                                let denom = eia + ejb;
                                if denom == 0.0 || !denom.is_finite() {
                                    return Err(PbcAdcError::ShapeMismatch {
                                        expected: 1,
                                        got: 0,
                                    });
                                }
                                // conj + transpose(0,2,1,3): t[i,j,a,b] =
                                // conj(ovov[i,a,j,b]) / eijab.
                                let src = o + ((i * nv + a) * no + j) * nv + b;
                                let dst = (((ki * nk + kj) * nk + ka) * no + i) * no * nv * nv
                                    + (j * nv + a) * nv
                                    + b;
                                t2.re[dst] = eris.ovov.re[src] / denom;
                                t2.im[dst] = -eris.ovov.im[src] / denom;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(KadcAmplitudes { nkpts: nk, nocc: no, nvir: nv, t2_1: t2, t1_2: None })
}

/// ADC(2) correlation energy (`compute_amplitudes.py::compute_energy`).
///
/// `emp2 = Σ_{ki,kj,ka} [2·Σ_{ijab} t[ki,kj,ka][i,j,a,b]·ovov[ki,ka,kj][i,a,j,b]
/// − Σ_{ijab} t[ki,kj,ka][i,j,a,b]·ovov[kj,ka,ki][j,a,i,b]]`, real part,
/// divided by `nkpts`. The k-index orders differ between the direct and
/// exchange terms — that is upstream's equation, and the `kadc_base` test
/// pins both orders element-wise (the shape-silent trap).
pub fn adc2_energy(
    t2: &KadcAmplitudes,
    eris: &KadcEris,
    kconserv: &[usize],
) -> Result<f64, PbcAdcError> {
    let (nk, no, nv) = (t2.nkpts, t2.nocc, t2.nvir);
    if eris.nkpts != nk || eris.nocc != no || eris.nvir != nv {
        return Err(PbcAdcError::ShapeMismatch { expected: nk, got: eris.nkpts });
    }
    let _ = kconserv;
    let mut re_terms: Vec<f64> = Vec::new();
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let tt = no * no * nv * nv;
                let ot = no * nv * no * nv;
                let to = (((ki * nk + kj) * nk + ka) * tt) as usize;
                let od = ((ki * nk + ka) * nk + kj) * ot;
                let ox = ((kj * nk + ka) * nk + ki) * ot;
                for i in 0..no {
                    for j in 0..no {
                        for a in 0..nv {
                            for b in 0..nv {
                                let t_idx = to + ((i * no + j) * nv + a) * nv + b;
                                let (tr, ti) = (t2.t2_1.re[t_idx], t2.t2_1.im[t_idx]);
                                // Direct: ovov[ki,ka,kj][i,a,j,b].
                                let d_idx = od + ((i * nv + a) * no + j) * nv + b;
                                let (dr, di) = (eris.ovov.re[d_idx], eris.ovov.im[d_idx]);
                                // Exchange: ovov[kj,ka,ki][j,a,i,b].
                                let x_idx = ox + ((j * nv + a) * no + i) * nv + b;
                                let (xr, xi) = (eris.ovov.re[x_idx], eris.ovov.im[x_idx]);
                                // 2·Re(t·d) − Re(t·x), complex products expanded.
                                re_terms.push(2.0 * (tr * dr - ti * di) - (tr * xr - ti * xi));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(oracle_sum(&re_terms) / nk as f64)
}

/// Build amplitudes (the `compute_amplitudes_energy` entry for ADC(2)).
pub fn build_amplitudes(
    eris: &KadcEris,
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    method: &str,
) -> Result<(f64, KadcAmplitudes), PbcAdcError> {
    if method != "adc(2)" {
        return Err(PbcAdcError::NotYetImplemented { module: "adc t1_2/t2_2 (adc(2)-x/adc(3))" });
    }
    let amps = t2_first_order(eris, e_occ_k, e_vir_k, kconserv)?;
    let e = adc2_energy(&amps, eris, kconserv)?;
    Ok((e, amps))
}
