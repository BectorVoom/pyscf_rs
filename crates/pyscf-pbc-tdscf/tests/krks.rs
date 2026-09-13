//! Kohn-Sham TDA/TDDFT tests (19-09): thin subclasses, hybrid kernel, Gate A.
//!
//! * XC table (`hyb_fraction`), RSH refusal, pure-under-hybrid-name refusal,
//!   and the compiled-in `gen_response` seam (`require_rks_response`).
//! * Gate A2 (gamma RKS/UKS, pure PBE): HF part at `hyb = 0` plus the
//!   fixture-derived fxc matrix → sorted/counted roots at 1e-8 Ha
//!   (upstream `test_rks.py`/`test_uks.py` assert 8dp).
//! * Hybrid non-vacuity + end-to-end: exact hyb-linearity of the HF part
//!   (`build(0.5) − build(0.25) == build(0.25) − build(0)`, catching any
//!   misplaced exchange factor) and PBE0 roots vs upstream at 1e-8.
//! * Gate A (KRKS diamond, PBE): per-shift roots at upstream's own 5dp (eV)
//!   (`test_krks.py` asserts 5dp — per-method inheritance, 19-01 Task 4).
//! * KUKS: single-k consistency vs the UKS driver + synthetic complex oracle.
//!
//! The fxc GRID contraction (Becke/numint, Phases 4/12) feeding the fxc
//! MATRIX is the documented next step; the assembly, scaling, solve and
//! routing are all live here.
//!
//! Run scoped: `cargo test -p pyscf-pbc-tdscf --test krks`

use pyscf_pbc_tdscf::krks::kernel_krks_tda;
use pyscf_pbc_tdscf::kuks::kernel_kuks_tda;
use pyscf_pbc_tdscf::rks::{check_hybrid_kernel, hyb_fraction, is_hybrid, kernel_rks_tda};
use pyscf_pbc_tdscf::types::TdaConfig;
use pyscf_pbc_tdscf::uhf::UhfDims;
use pyscf_pbc_tdscf::uks::kernel_uks_tda;
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture must exist")).unwrap()
}

fn flat(v: &Value) -> Vec<f64> {
    fn fl(v: &Value, o: &mut Vec<f64>) {
        match v {
            Value::Number(x) => o.push(x.as_f64().unwrap()),
            Value::Array(items) => {
                for it in items {
                    fl(it, o);
                }
            }
            _ => panic!("unexpected JSON"),
        }
    }
    let mut o = Vec::new();
    fl(v, &mut o);
    o
}

fn flat_cx(v: &Value) -> (Vec<f64>, Vec<f64>) {
    (flat(&v["re"]), flat(&v["im"]))
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn cfg(singlet: bool, nroots: usize) -> TdaConfig {
    TdaConfig { nroots, conv_tol: 1e-9, max_cycle: 50, singlet, tda: true, kshift: 0 }
}

/// XC table + refusals + seam.
#[test]
fn xc_table_and_seam() {
    assert_eq!(hyb_fraction("pbe").unwrap().to_bits(), 0.0f64.to_bits());
    assert_eq!(hyb_fraction("lda").unwrap().to_bits(), 0.0f64.to_bits());
    assert!((hyb_fraction("pbe0").unwrap() - 0.25).abs() < 1e-15);
    assert!((hyb_fraction("b3lyp").unwrap() - 0.20).abs() < 1e-15);
    assert!((hyb_fraction("hf").unwrap() - 1.0).abs() < 1e-15);
    assert!(hyb_fraction("cam-b3lyp").is_err());
    assert!(hyb_fraction("wb97x").is_err());
    assert!(is_hybrid("pbe0"));
    assert!(!is_hybrid("pbe"));
    assert!(check_hybrid_kernel("pbe0", 0.25).is_ok());
    assert!(check_hybrid_kernel("pbe0", 0.0).is_err());
    assert!(check_hybrid_kernel("pbe", 0.0).is_ok());
    // §1.6 seam compiled in: concrete RKS provides gen_response.
    assert!(pyscf_pbc_tdscf::require_rks_response());
}

/// Hybrid scaling is exactly linear in hyb (non-vacuous kernel switch).
#[test]
fn hybrid_scaling_exactly_linear() {
    use pyscf_pbc_tdscf::rhf::build_ab;
    let v = fixture("ks_h2.json");
    let r = &v["rks"];
    let (nocc, nmo) = (r["nocc"].as_u64().unwrap() as usize, r["nmo"].as_u64().unwrap() as usize);
    let (eri, e_ia) = (flat(&r["eri"]), flat(&r["e_ia"]));
    let (a0, _) = build_ab(&eri, &e_ia, nocc, nmo, true, 0.0).unwrap();
    let (a25, _) = build_ab(&eri, &e_ia, nocc, nmo, true, 0.25).unwrap();
    let (a50, _) = build_ab(&eri, &e_ia, nocc, nmo, true, 0.5).unwrap();
    // build(0.25) − build(0) == build(0.5) − build(0.25) element-wise:
    // the exchange enters EXACTLY once, scaled by hyb.
    for i in 0..a0.len() {
        let d1 = a25[i] - a0[i];
        let d2 = a50[i] - a25[i];
        assert!((d1 - d2).abs() < 1e-12, "hyb non-linearity at {i}: {d1:e} vs {d2:e}");
    }
    // ...and it is nonzero (a vacuous switch would give all zeros).
    let norm: f64 = a25.iter().zip(a0.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(norm > 1e-6, "hybrid kernel identical to pure?!");
}

/// Gate A2: RKS/PBE roots at 1e-8 Ha.
#[test]
fn gate_a_rks_pbe() {
    let v = fixture("ks_h2.json");
    let r = &v["rks"];
    let (nocc, nmo) = (r["nocc"].as_u64().unwrap() as usize, r["nmo"].as_u64().unwrap() as usize);
    let (eri, e_ia) = (flat(&r["eri"]), flat(&r["e_ia"]));
    // fxc = upstream a − HF(hyb=0) (same inputs → tight).
    let (a_hf, _) = pyscf_pbc_tdscf::rhf::build_ab(&eri, &e_ia, nocc, nmo, true, 0.0).unwrap();
    let a_up = flat(&r["a"]["re"]);
    let dim = nocc * (nmo - nocc);
    assert_eq!(a_up.len(), dim * dim);
    let fxc_a: Vec<f64> = a_up.iter().zip(a_hf.iter()).map(|(u, h)| u - h).collect();
    let fxc_b = vec![0.0f64; dim * dim];
    let out = kernel_rks_tda(&eri, &e_ia, &fxc_a, &fxc_b, nocc, nmo, "pbe", &cfg(true, 3))
        .expect("RKS must run");
    assert_eq!(out.energies.len(), 3);
    assert!(out.energies.windows(2).all(|w| w[0] <= w[1]));
    let eref = flat(&r["e"]);
    let d = max_diff(&out.energies, &eref[..3]);
    assert!(d < 1e-8, "Gate A2 RKS deviates {d:e} Ha");
}

/// Hybrid end-to-end: PBE0 roots at 1e-8 Ha.
#[test]
fn gate_a_rks_pbe0() {
    let v = fixture("ks_h2_hybrid_k.json");
    let r = &v["rks_pbe0"];
    let (nocc, nmo) = (r["nocc"].as_u64().unwrap() as usize, r["nmo"].as_u64().unwrap() as usize);
    let (eri, e_ia) = (flat(&r["eri"]), flat(&r["e_ia"]));
    let (a_hf, _) = pyscf_pbc_tdscf::rhf::build_ab(&eri, &e_ia, nocc, nmo, true, 0.25).unwrap();
    let a_up = flat(&r["a"]["re"]);
    let dim = nocc * (nmo - nocc);
    let fxc_a: Vec<f64> = a_up.iter().zip(a_hf.iter()).map(|(u, h)| u - h).collect();
    let fxc_b = vec![0.0f64; dim * dim];
    let out = kernel_rks_tda(&eri, &e_ia, &fxc_a, &fxc_b, nocc, nmo, "pbe0", &cfg(true, 3))
        .expect("RKS/PBE0 must run");
    let eref = flat(&r["e"]);
    let d = max_diff(&out.energies, &eref[..3]);
    assert!(d < 1e-8, "Gate A2 PBE0 deviates {d:e} Ha");
}

/// Gate A2: UKS/PBE triplet roots at 1e-8 Ha.
#[test]
fn gate_a_uks_pbe() {
    use pyscf_pbc_tdscf::uhf::{UhfDims, build_uab};
    let v = fixture("ks_h2.json");
    let r = &v["uks"];
    let d = UhfDims {
        nocc_a: r["noa"].as_u64().unwrap() as usize,
        nvir_a: r["nva"].as_u64().unwrap() as usize,
        nocc_b: r["nob"].as_u64().unwrap() as usize,
        nvir_b: r["nvb"].as_u64().unwrap() as usize,
    };
    let (nmo_a, nmo_b) = (r["nmo"].as_u64().unwrap() as usize, r["nmo"].as_u64().unwrap() as usize);
    let (da, db) = (d.da(), d.db());
    let args = (
        flat(&r["eri_aa"]), flat(&r["eri_ab"]), flat(&r["eri_bb"]),
        flat(&r["e_ia_a"]), flat(&r["e_ia_b"]),
    );
    // Spin-resolved fxc from upstream blocks: fxc_aa = A_aa − HF_aa(hyb=0), etc.
    // Fixture blocks are {re,im} dicts (real data — im asserted zero).
    let (a_hf, _) = build_uab(&args.0, &args.1, &args.2, &args.3, &args.4, d, nmo_a, nmo_b, 0.0).unwrap();
    let au = [flat(&r["a"][0]["re"]), flat(&r["a"][1]["re"]), flat(&r["a"][2]["re"])];
    for k in 0..3 {
        assert!(flat(&r["a"][k]["im"]).iter().all(|x| *x == 0.0));
    }
    let dim = da + db;
    let mut fxc_full = vec![0.0f64; dim * dim];
    for i in 0..da {
        for j in 0..da {
            fxc_full[i * dim + j] = au[0][i * da + j] - a_hf[i * dim + j];
        }
        for j in 0..db {
            fxc_full[i * dim + da + j] = au[1][i * db + j] - a_hf[i * dim + da + j];
        }
    }
    for i in 0..db {
        for j in 0..db {
            fxc_full[(da + i) * dim + da + j] = au[2][i * db + j] - a_hf[(da + i) * dim + da + j];
        }
    }
    // Split into the kernel's aa/ab/bb inputs.
    let mut fxc_aa = vec![0.0f64; da * da];
    let mut fxc_ab = vec![0.0f64; da * db];
    let mut fxc_bb = vec![0.0f64; db * db];
    for i in 0..da {
        for j in 0..da {
            fxc_aa[i * da + j] = fxc_full[i * dim + j];
        }
        for j in 0..db {
            fxc_ab[i * db + j] = fxc_full[i * dim + da + j];
        }
    }
    for i in 0..db {
        for j in 0..db {
            fxc_bb[i * db + j] = fxc_full[(da + i) * dim + da + j];
        }
    }
    let out = kernel_uks_tda(
        &args.0, &args.1, &args.2, &args.3, &args.4,
        &fxc_aa, &fxc_ab, &fxc_bb, d, nmo_a, nmo_b, "pbe", &cfg(true, 3),
    )
    .expect("UKS must run");
    let eref = flat(&r["e"]);
    let dev = max_diff(&out.energies, &eref[..3]);
    assert!(dev < 1e-8, "Gate A2 UKS deviates {dev:e} Ha");
}

/// Gate A: KRKS/PBE diamond shift-0 at upstream's 5dp (eV).
#[test]
fn gate_a_krks_pbe() {
    let v = fixture("ks_h2_hybrid_k.json");
    let r = &v["krks"];
    let (nk, nocc, nmo) = (
        r["nkpts"].as_u64().unwrap() as usize,
        r["nocc"].as_u64().unwrap() as usize,
        r["nmo"].as_u64().unwrap() as usize,
    );
    let nmo_nocc = nmo * nocc * nmo * nmo;
    let (all_re, all_im) = (flat(&r["eri7"]["re"]), flat(&r["eri7"]["im"]));
    let mut eri7re = Vec::new();
    let mut eri7im = Vec::new();
    for t in 0..nk * nk * nk {
        eri7re.push(all_re[t * nmo_nocc..(t + 1) * nmo_nocc].to_vec());
        eri7im.push(all_im[t * nmo_nocc..(t + 1) * nmo_nocc].to_vec());
    }
    let kc: Vec<usize> = flat(&r["kconserv"]).iter().map(|x| *x as usize).collect();
    let e_occ = vec![flat(&r["e_occ"][0]), flat(&r["e_occ"][1])];
    let e_vir = vec![flat(&r["e_vir"][0]), flat(&r["e_vir"][1])];
    // fxc = upstream a − HF(hyb=0) at shift 0.
    let (a_hf, _) = pyscf_pbc_tdscf::krhf::build_kab(
        &eri7re, &eri7im, &e_occ, &e_vir, &kc, nk, nocc, nmo, true, 0.0, 1.0 / nk as f64,
    )
    .unwrap();
    let (ar, ai) = flat_cx(&r["a"]);
    let nkd = nk * nocc * (nmo - nocc);
    assert_eq!(ar.len(), nkd * nkd);
    let fxc_re: Vec<f64> = ar.iter().zip(a_hf.re.iter()).map(|(u, h)| u - h).collect();
    let fxc_im: Vec<f64> = ai.iter().zip(a_hf.im.iter()).map(|(u, h)| u - h).collect();
    let mut c = cfg(true, 2);
    c.kshift = 0;
    let out = kernel_krks_tda(
        &eri7re, &eri7im, &fxc_re, &fxc_im, &e_occ, &e_vir, &kc, nk, nocc, nmo, "pbe", 0, &c,
    )
    .expect("KRKS must run");
    assert_eq!(out.kshift, 0);
    let eref = flat(&r["e"]);
    let dev_ev = max_diff(
        &out.energies.iter().map(|x| x * 27.211386018).collect::<Vec<_>>(),
        &eref[..2].iter().map(|x| x * 27.211386018).collect::<Vec<_>>(),
    );
    assert!(dev_ev < 5e-5, "Gate A KRKS deviates {dev_ev:e} eV");
}

/// KUKS single-k consistency vs the UKS driver (real data, shift stamped).
#[test]
fn kuks_single_k_matches_uks() {
    use pyscf_pbc_tdscf::uhf::UhfDims;
    let v = fixture("ks_h2.json");
    let r = &v["uks"];
    let d = UhfDims {
        nocc_a: r["noa"].as_u64().unwrap() as usize,
        nvir_a: r["nva"].as_u64().unwrap() as usize,
        nocc_b: r["nob"].as_u64().unwrap() as usize,
        nvir_b: r["nvb"].as_u64().unwrap() as usize,
    };
    let (nmo_a, nmo_b) = (r["nmo"].as_u64().unwrap() as usize, r["nmo"].as_u64().unwrap() as usize);
    let (da, db) = (d.da(), d.db());
    // Transpose gamma [i][p][q][r] → 7d [p][i][q][r] (19-08 lesson).
    fn to_7d(g: Vec<f64>, no: usize, nmo: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; g.len()];
        for i in 0..no {
            for p in 0..nmo {
                for q in 0..nmo {
                    for rr in 0..nmo {
                        out[((p * no + i) * nmo + q) * nmo + rr] =
                            g[((i * nmo + p) * nmo + q) * nmo + rr];
                    }
                }
            }
        }
        out
    }
    fn to_7d_ab(g: Vec<f64>, noa: usize, nmo_a: usize, nmo_b: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; g.len()];
        for i in 0..noa {
            for p in 0..nmo_a {
                for q in 0..nmo_b {
                    for rr in 0..nmo_b {
                        out[((p * noa + i) * nmo_b + q) * nmo_b + rr] =
                            g[((i * nmo_a + p) * nmo_b + q) * nmo_b + rr];
                    }
                }
            }
        }
        out
    }
    let wrap = |x: Vec<f64>| vec![x];
    let zaa = vec![0.0f64; nmo_a * d.nocc_a * nmo_a * nmo_a];
    let zab = vec![0.0f64; nmo_a * d.nocc_a * nmo_b * nmo_b];
    let zbb = vec![0.0f64; nmo_b * d.nocc_b * nmo_b * nmo_b];
    let e_oa = vec![flat(&r["e_occ_a"])];
    let e_va = vec![flat(&r["e_vir_a"])];
    let e_ob = vec![flat(&r["e_occ_b"])];
    let e_vb = vec![flat(&r["e_vir_b"])];
    // Full fxc from the UKS assembly path (same derivation, single-k layout).
    let args = (flat(&r["eri_aa"]), flat(&r["eri_ab"]), flat(&r["eri_bb"]), flat(&r["e_ia_a"]), flat(&r["e_ia_b"]));
    let (a_hf, _) = pyscf_pbc_tdscf::uhf::build_uab(&args.0, &args.1, &args.2, &args.3, &args.4, d, nmo_a, nmo_b, 0.0).unwrap();
    let au = [flat(&r["a"][0]["re"]), flat(&r["a"][1]["re"]), flat(&r["a"][2]["re"])];
    let dim = da + db;
    let mut fxc_full = vec![0.0f64; dim * dim];
    for i in 0..da {
        for j in 0..da {
            fxc_full[i * dim + j] = au[0][i * da + j] - a_hf[i * dim + j];
        }
        for j in 0..db {
            fxc_full[i * dim + da + j] = au[1][i * db + j] - a_hf[i * dim + da + j];
        }
    }
    for i in 0..db {
        for j in 0..db {
            fxc_full[(da + i) * dim + da + j] = au[2][i * db + j] - a_hf[(da + i) * dim + da + j];
        }
        for j in 0..da {
            fxc_full[(da + i) * dim + j] = au[1][j * db + i] - a_hf[(da + i) * dim + j];
        }
    }
    let fxc_im = vec![0.0f64; dim * dim];
    let out = kernel_kuks_tda(
        &wrap(to_7d(args.0.clone(), d.nocc_a, nmo_a)), &wrap(zaa.clone()),
        &wrap(to_7d_ab(args.1.clone(), d.nocc_a, nmo_a, nmo_b)), &wrap(zab.clone()),
        &wrap(to_7d(args.2.clone(), d.nocc_b, nmo_b)), &wrap(zbb.clone()),
        &fxc_full, &fxc_im,
        &e_oa, &e_va, &e_ob, &e_vb, &[0], 1, d, nmo_a, nmo_b, "pbe", 0, &cfg(true, 3),
    )
    .expect("KUKS single-k must run");
    assert_eq!(out.kshift, 0);
    let eref = flat(&r["e"]);
    let dev = max_diff(&out.energies, &eref[..3]);
    assert!(dev < 1e-8, "KUKS single-k deviates {dev:e} Ha");
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn ks_deterministic_across_thread_counts() {
    let run = || {
        let v = fixture("ks_h2.json");
        let r = &v["rks"];
        let (nocc, nmo) = (r["nocc"].as_u64().unwrap() as usize, r["nmo"].as_u64().unwrap() as usize);
        let (eri, e_ia) = (flat(&r["eri"]), flat(&r["e_ia"]));
        let (a_hf, _) = pyscf_pbc_tdscf::rhf::build_ab(&eri, &e_ia, nocc, nmo, true, 0.0).unwrap();
        let a_up = flat(&r["a"]["re"]);
        let dim = nocc * (nmo - nocc);
        let fxc_a: Vec<f64> = a_up.iter().zip(a_hf.iter()).map(|(u, h)| u - h).collect();
        kernel_rks_tda(&eri, &e_ia, &fxc_a, &vec![0.0; dim * dim], nocc, nmo, "pbe", &cfg(true, 2))
            .expect("RKS must run")
            .energies
    };
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let (a, b) = (pool1.install(run), pool8.install(run));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}
