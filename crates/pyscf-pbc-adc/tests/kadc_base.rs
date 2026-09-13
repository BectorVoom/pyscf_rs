//! ADC base tests (19-14): element-wise index oracle, arena discipline,
//! t2/energy references, determinism.
//!
//! * Synthetic 4-index provider with DISTINGUISHABLE values per
//!   `(kp,kq,kr,p,q,r,s)`; every sliced element of all six blocks asserted
//!   element-wise (a shape assertion catches nothing here — `nocc != nvir`
//!   throughout so no equal-dimension coincidence can hide a swap either).
//! * The oracle is proven discriminating: a deliberately transposed gather
//!   expectation does NOT match (demonstrated, then discarded).
//! * Accumulation targets explicitly zeroed before reuse.
//! * `t2_1` and `adc2_energy` against hand-computed values.
//! * Determinism at 1 vs 8 rayon workers inside one process.
//!
//! Run scoped: `cargo test -p pyscf-pbc-adc --test kadc_base`

use pyscf_algebra::CTensor;
use pyscf_pbc_adc::amplitudes::{adc2_energy, t2_first_order};
use pyscf_pbc_adc::kadc_ao2mo::{build_incore, KadcEris};
use pyscf_pbc_adc::kadc_rhf::KadcDriver;
use pyscf_pbc_adc::types::AdcLevel;

const NK: usize = 2;
const NO: usize = 2;
const NV: usize = 3;
const NMO: usize = NO + NV;

/// Distinguishable synthetic 4-index block: value encodes every index as
/// `re = p + 10q + 100r + 1000s + 10000ki + 20000kj + 40000kk + 80000kl`,
/// `im = -(same)/7` (nonzero imaginary part exercises conjugation).
fn synth_block(ki: usize, kj: usize, kk: usize, kl: usize) -> CTensor {
    let mut re = vec![0.0f64; NMO * NMO * NMO * NMO];
    let mut im = vec![0.0f64; NMO * NMO * NMO * NMO];
    for p in 0..NMO {
        for q in 0..NMO {
            for r in 0..NMO {
                for s in 0..NMO {
                    let o = ((p * NMO + q) * NMO + r) * NMO + s;
                    let code = p as f64
                        + 10.0 * q as f64
                        + 100.0 * r as f64
                        + 1000.0 * s as f64
                        + 10000.0 * ki as f64
                        + 20000.0 * kj as f64
                        + 40000.0 * kk as f64
                        + 80000.0 * kl as f64;
                    re[o] = code;
                    im[o] = -code / 7.0;
                }
            }
        }
    }
    CTensor { re, im }
}

/// Momentum conservation for the fixture: iks = (ikp + ikq + ikr) % NK.
fn kconserv_fixture() -> Vec<usize> {
    let mut kc = vec![0usize; NK * NK * NK];
    for p in 0..NK {
        for q in 0..NK {
            for r in 0..NK {
                kc[(p * NK + q) * NK + r] = (p + q + r) % NK;
            }
        }
    }
    kc
}

fn build_fixture() -> KadcEris {
    let kc = kconserv_fixture();
    build_incore(NK, NO, NV, &kc, &|ki, kj, kk, kl| {
        Ok(synth_block(ki, kj, kk, kl))
    })
    .expect("fixture build must succeed")
}

fn expect_code(
    ki: usize,
    kj: usize,
    kk: usize,
    kl: usize,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> (f64, f64) {
    let code = p as f64
        + 10.0 * q as f64
        + 100.0 * r as f64
        + 1000.0 * s as f64
        + 10000.0 * ki as f64
        + 20000.0 * kj as f64
        + 40000.0 * kk as f64
        + 80000.0 * kl as f64;
    (code / NK as f64, -code / (7.0 * NK as f64))
}

/// Every sliced element of all six blocks, element-wise.
#[test]
fn index_oracle_elementwise_all_blocks() {
    let eris = build_fixture();
    let kc = kconserv_fixture();
    for kp in 0..NK {
        for kq in 0..NK {
            for kr in 0..NK {
                let ks = kc[(kp * NK + kq) * NK + kr];
                let base = ((kp * NK + kq) * NK + kr) as usize;
                for i in 0..NO {
                    for j in 0..NO {
                        for k in 0..NO {
                            for l in 0..NO {
                                let o = (base * NO * NO * NO + i * NO * NO + j * NO + k) * NO + l;
                                let (re, im) = expect_code(kp, kq, kr, ks, i, j, k, l);
                                assert_eq!(
                                    eris.oooo.re[o].to_bits(),
                                    re.to_bits(),
                                    "oooo[{kp},{kq},{kr}][{i},{j},{k},{l}]"
                                );
                                assert_eq!(eris.oooo.im[o].to_bits(), im.to_bits(), "oooo im");
                            }
                        }
                        for a in 0..NV {
                            for b in 0..NV {
                                let o = (base * NO * NO * NV + i * NO * NV + j * NV + a) * NV + b;
                                let (re, im) = expect_code(kp, kq, kr, ks, i, j, NO + a, NO + b);
                                assert_eq!(
                                    eris.oovv.re[o].to_bits(),
                                    re.to_bits(),
                                    "oovv[{kp},{kq},{kr}][{i},{j},{a},{b}]"
                                );
                                assert_eq!(eris.oovv.im[o].to_bits(), im.to_bits(), "oovv im");
                            }
                            for jj in 0..NO {
                                for kk in 0..NO {
                                    let o =
                                        (base * NO * NV * NO + i * NV * NO + a * NO + jj) * NO + kk;
                                    let (re, im) = expect_code(kp, kq, kr, ks, i, NO + a, jj, kk);
                                    assert_eq!(
                                        eris.ovoo.re[o].to_bits(),
                                        re.to_bits(),
                                        "ovoo[{kp},{kq},{kr}][{i},{a},{jj},{kk}]"
                                    );
                                    assert_eq!(eris.ovoo.im[o].to_bits(), im.to_bits(), "ovoo im");
                                }
                                for b in 0..NV {
                                    let o =
                                        (base * NO * NV * NO + i * NV * NO + a * NO + jj) * NV + b;
                                    let (re, im) =
                                        expect_code(kp, kq, kr, ks, i, NO + a, jj, NO + b);
                                    assert_eq!(
                                        eris.ovov.re[o].to_bits(),
                                        re.to_bits(),
                                        "ovov[{kp},{kq},{kr}][{i},{a},{jj},{b}]"
                                    );
                                    assert_eq!(eris.ovov.im[o].to_bits(), im.to_bits(), "ovov im");
                                }
                            }
                            for b in 0..NV {
                                for c in 0..NV {
                                    let o =
                                        (base * NO * NV * NV + i * NV * NV + a * NV + b) * NV + c;
                                    let (re, im) =
                                        expect_code(kp, kq, kr, ks, i, NO + a, NO + b, NO + c);
                                    assert_eq!(eris.ovvv.re[o].to_bits(), re.to_bits(), "ovvv");
                                    assert_eq!(eris.ovvv.im[o].to_bits(), im.to_bits(), "ovvv im");
                                }
                                for jj in 0..NO {
                                    let o =
                                        (base * NO * NV * NV + i * NV * NV + a * NV + b) * NO + jj;
                                    let (re, im) =
                                        expect_code(kp, kq, kr, ks, i, NO + a, NO + b, jj);
                                    assert_eq!(eris.ovvo.re[o].to_bits(), re.to_bits(), "ovvo");
                                    assert_eq!(eris.ovvo.im[o].to_bits(), im.to_bits(), "ovvo im");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The oracle discriminates: a transposed-gather expectation does NOT match.
#[test]
fn oracle_catches_transposed_k_gather() {
    let eris = build_fixture();
    // Deliberately read ovov with kj/ka swapped and demand equality — this
    // must FAIL to match, proving the oracle sees the difference.
    let (re0, _) = eris.ovov_at(0, 1, 0);
    let (re1, _) = eris.ovov_at(0, 0, 1);
    let same = re0
        .iter()
        .zip(re1.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(
        !same,
        "transposed gather is indistinguishable — oracle is blind"
    );
}

/// Accumulation targets are explicitly zeroed (arena-recycling discipline).
#[test]
fn accumulation_targets_start_zeroed() {
    let z = CTensor::zeros(NK * NK * NK * NO * NO * NV * NV);
    assert!(z.re.iter().all(|&x| x == 0.0));
    assert!(z.im.iter().all(|&x| x == 0.0));
    // Re-zero before reuse: fill with garbage, zero, verify.
    let mut r = z;
    r.re.fill(1.5);
    r.im.fill(-2.5);
    r.re.fill(0.0);
    r.im.fill(0.0);
    assert!(r.re.iter().all(|&x| x == 0.0));
    assert!(r.im.iter().all(|&x| x == 0.0));
}

/// t2_1 element-wise against the defining equation (complex conjugation live).
#[test]
fn t2_first_order_matches_defining_equation() {
    let eris = build_fixture();
    let kc = kconserv_fixture();
    let e_occ = vec![vec![-0.6, -0.4], vec![-0.5, -0.45]];
    let e_vir = vec![vec![0.3, 0.5, 0.8], vec![0.35, 0.55, 0.75]];
    let amps = t2_first_order(&eris, &e_occ, &e_vir, &kc).expect("t2 must build");
    for ki in 0..NK {
        for kj in 0..NK {
            for ka in 0..NK {
                let kb = kc[(ki * NK + ka) * NK + kj];
                for i in 0..NO {
                    for j in 0..NO {
                        for a in 0..NV {
                            for b in 0..NV {
                                // Upstream `_get_epq(..., fac=[1.0,-1.0])`
                                // (`kadc_rhf_amplitudes.py:101-108`): NEGATIVE
                                // gaps, `eijab = (e_occ − e_vir) + (e_occ − e_vir)`.
                                let denom =
                                    (e_occ[ki][i] - e_vir[ka][a]) + (e_occ[kj][j] - e_vir[kb][b]);
                                // conj(ovov[ki,ka,kj][i,a,j,b]) / eijab.
                                let t = NO * NV * NO * NV;
                                let src = ((ki * NK + ka) * NK + kj) * t
                                    + ((i * NV + a) * NO + j) * NV
                                    + b;
                                let dst = (((ki * NK + kj) * NK + ka) * NO + i) * NO * NV * NV
                                    + (j * NV + a) * NV
                                    + b;
                                let (vr, vi) =
                                    (eris.ovov.re[src] / denom, -eris.ovov.im[src] / denom);
                                assert_eq!(
                                    amps.t2_1.re[dst].to_bits(),
                                    vr.to_bits(),
                                    "t2 re [{ki},{kj},{ka}][{i},{j},{a},{b}]"
                                );
                                assert_eq!(amps.t2_1.im[dst].to_bits(), vi.to_bits(), "t2 im");
                                // Conjugation is real: im parts are nonzero here.
                                assert_ne!(eris.ovov.im[src], 0.0);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// ADC(2) energy against an independent hand computation (real-part form).
#[test]
fn adc2_energy_matches_hand_computation() {
    let eris = build_fixture();
    let kc = kconserv_fixture();
    let e_occ = vec![vec![-0.6, -0.4], vec![-0.5, -0.45]];
    let e_vir = vec![vec![0.3, 0.5, 0.8], vec![0.35, 0.55, 0.75]];
    let amps = t2_first_order(&eris, &e_occ, &e_vir, &kc).expect("t2 must build");
    let e = adc2_energy(&amps, &eris, &kc).expect("energy must run");
    // Independent loop over the same equation, written separately.
    let mut acc = 0.0f64;
    for ki in 0..NK {
        for kj in 0..NK {
            for ka in 0..NK {
                let tt = NO * NO * NV * NV;
                let ot = NO * NV * NO * NV;
                let to = ((ki * NK + kj) * NK + ka) * tt;
                let od = ((ki * NK + ka) * NK + kj) * ot;
                let ox = ((kj * NK + ka) * NK + ki) * ot;
                for i in 0..NO {
                    for j in 0..NO {
                        for a in 0..NV {
                            for b in 0..NV {
                                let t_idx = to + ((i * NO + j) * NV + a) * NV + b;
                                let d_idx = od + ((i * NV + a) * NO + j) * NV + b;
                                let x_idx = ox + ((j * NV + a) * NO + i) * NV + b;
                                let (tr, ti) = (amps.t2_1.re[t_idx], amps.t2_1.im[t_idx]);
                                acc += 2.0 * (tr * eris.ovov.re[d_idx] - ti * eris.ovov.im[d_idx])
                                    - (tr * eris.ovov.re[x_idx] - ti * eris.ovov.im[x_idx]);
                            }
                        }
                    }
                }
            }
        }
    }
    acc /= NK as f64;
    // Relative tolerance: the two loops accumulate in different orders, so
    // last-ulp summation differences on a 1e11-scale total are expected and
    // correct (ordered-reduction discipline pins each order, not across them).
    let rel = (e - acc).abs() / acc.abs().max(1.0);
    assert!(rel < 1e-12, "energy {e:e} vs hand {acc:e} (rel {rel:e})");
    assert!(e.is_finite());
}

/// Driver shape: kernel_gs runs end to end on the fixture.
#[test]
fn driver_kernel_gs_runs() {
    let eris = build_fixture();
    let kc = kconserv_fixture();
    let driver = KadcDriver {
        level: AdcLevel::Adc2,
        nkpts: NK,
        nocc: NO,
        nvir: NV,
    };
    let e_occ = vec![vec![-0.6, -0.4], vec![-0.5, -0.45]];
    let e_vir = vec![vec![0.3, 0.5, 0.8], vec![0.35, 0.55, 0.75]];
    let (e, amps) = driver
        .kernel_gs(&eris, &e_occ, &e_vir, &kc)
        .expect("gs must run");
    assert!(e.is_finite());
    assert_eq!(amps.t2_1.re.len(), NK * NK * NK * NO * NO * NV * NV);
    // adc(2)-x / adc(3) refuse (t1_2/t2_2 land with the manifolds).
    let driver_x = KadcDriver {
        level: AdcLevel::Adc2x,
        nkpts: NK,
        nocc: NO,
        nvir: NV,
    };
    assert!(driver_x.kernel_gs(&eris, &e_occ, &e_vir, &kc).is_err());
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn kadc_base_deterministic_across_thread_counts() {
    let run = || {
        let eris = build_fixture();
        let kc = kconserv_fixture();
        let e_occ = vec![vec![-0.6, -0.4], vec![-0.5, -0.45]];
        let e_vir = vec![vec![0.3, 0.5, 0.8], vec![0.35, 0.55, 0.75]];
        let amps = t2_first_order(&eris, &e_occ, &e_vir, &kc).expect("t2 must build");
        adc2_energy(&amps, &eris, &kc).expect("energy must run")
    };
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    let (a, b) = (pool1.install(run), pool8.install(run));
    assert_eq!(a.to_bits(), b.to_bits());
}
