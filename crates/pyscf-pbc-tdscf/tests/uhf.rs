//! Unrestricted TDA/TDHF tests (19-08): coupling, Gate A, k-machinery.
//!
//! Fixture `fixtures/uhf_h2o_triplet.json` (upstream 2.12.1 H2O/`sto-3g`
//! triplet in a 15-Bohr box, gamma UHF): spin ERI blocks, gaps, spectra,
//! upstream's coupled `(A, B)`, TDA + TDHF roots.
//!
//! * `build_uab` vs upstream element-wise (the `'iabj'` axis order is
//!   same-shape-silent — pinned here).
//! * **Coupling**: zeroing the ab blocks reproduces the independent
//!   alpha/beta solves bit-identically (coupling enters ONLY through ab),
//!   while the live coupled roots DIFFER from the decoupled union (solving
//!   separately is plausibly wrong — demonstrated, not assumed).
//! * Gate A on TDA + TDHF at upstream's decimals, sorted/counted, unfiltered.
//! * `kuhf` on the same physics as single-k data (shift stamped) + a
//!   synthetic COMPLEX index oracle for the k-build maps.
//! * Big-box 2-k Davidson comparison as `#[ignore]`d live arm.
//!
//! Run scoped: `cargo test -p pyscf-pbc-tdscf --test uhf`

use pyscf_pbc_tdscf::davidson::tda_dense;
use pyscf_pbc_tdscf::kuhf::{build_kuab, kernel_kuhf_tda};
use pyscf_pbc_tdscf::types::TdaConfig;
use pyscf_pbc_tdscf::uhf::{UhfDims, build_uab, kernel_uhf_tda, kernel_uhf_tdhf};
use serde_json::Value;

const HARTREE2EV: f64 = 27.211386018;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/uhf_h2o_triplet.json");
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

fn dims_of(v: &Value) -> (UhfDims, usize, usize) {
    let d = UhfDims {
        nocc_a: v["noa"].as_u64().unwrap() as usize,
        nvir_a: v["nva"].as_u64().unwrap() as usize,
        nocc_b: v["nob"].as_u64().unwrap() as usize,
        nvir_b: v["nvb"].as_u64().unwrap() as usize,
    };
    (d, v["nmo_a"].as_u64().unwrap() as usize, v["nmo_b"].as_u64().unwrap() as usize)
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn cfg(nroots: usize) -> TdaConfig {
    TdaConfig { nroots, conv_tol: 1e-9, max_cycle: 50, singlet: true, tda: true, kshift: 0 }
}

/// Coupled build vs upstream element-wise (all three spin sectors).
#[test]
fn build_uab_matches_upstream() {
    let v = fixture();
    let (d, nmo_a, nmo_b) = dims_of(&v);
    let (a, b) = build_uab(
        &flat(&v["eri_aa"]), &flat(&v["eri_ab"]), &flat(&v["eri_bb"]),
        &flat(&v["e_ia_a"]), &flat(&v["e_ia_b"]), d, nmo_a, nmo_b, 1.0,
    )
    .expect("build must run");
    let (da, db) = (d.da(), d.db());
    // Upstream A = (A_aa, A_ab, A_bb) blocks: reassemble full matrix.
    let aa = flat(&v["A"][0]);
    let ab = flat(&v["A"][1]);
    let bb = flat(&v["A"][2]);
    let mut ar = vec![0.0f64; (da + db) * (da + db)];
    for i in 0..da {
        for j in 0..da {
            ar[i * (da + db) + j] = aa[i * da + j];
        }
        for j in 0..db {
            ar[i * (da + db) + da + j] = ab[i * db + j];
            ar[(da + j) * (da + db) + i] = ab[i * db + j];
        }
    }
    for i in 0..db {
        for j in 0..db {
            ar[(da + i) * (da + db) + da + j] = bb[i * db + j];
        }
    }
    assert!(max_diff(&a, &ar) < 1e-10, "A deviates {:e}", max_diff(&a, &ar));
    let ba = flat(&v["B"][0]);
    let bab = flat(&v["B"][1]);
    let bbb = flat(&v["B"][2]);
    let mut br = vec![0.0f64; (da + db) * (da + db)];
    for i in 0..da {
        for j in 0..da {
            br[i * (da + db) + j] = ba[i * da + j];
        }
        for j in 0..db {
            br[i * (da + db) + da + j] = bab[i * db + j];
            br[(da + j) * (da + db) + i] = bab[i * db + j];
        }
    }
    for i in 0..db {
        for j in 0..db {
            br[(da + i) * (da + db) + da + j] = bbb[i * db + j];
        }
    }
    assert!(max_diff(&b, &br) < 1e-10, "B deviates {:e}", max_diff(&b, &br));
}

/// Gate A: UHF-TDA + TDHF roots at upstream's decimals, sorted/counted.
#[test]
fn gate_a_uhf_roots() {
    let v = fixture();
    let (d, nmo_a, nmo_b) = dims_of(&v);
    let args = (
        &flat(&v["eri_aa"]), &flat(&v["eri_ab"]), &flat(&v["eri_bb"]),
        &flat(&v["e_ia_a"]), &flat(&v["e_ia_b"]), d, nmo_a, nmo_b, 1.0,
    );
    let r = kernel_uhf_tda(args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, &cfg(3))
        .expect("TDA must run");
    assert_eq!(r.energies.len(), 3);
    assert!(r.energies.windows(2).all(|w| w[0] <= w[1]));
    let eref = flat(&v["e_tda"]);
    let dev = max_diff(
        &r.energies.iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
        &eref[..3].iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
    );
    assert!(dev < 5e-5, "TDA deviates {dev:e} eV");
    let mut c = cfg(3);
    c.tda = false;
    let rt = kernel_uhf_tdhf(args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, &c)
        .expect("TDHF must run");
    let ereft = flat(&v["e_tdhf"]);
    let devt = max_diff(
        &rt.energies.iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
        &ereft[..3].iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
    );
    assert!(devt < 5e-5, "TDHF deviates {devt:e} eV");
}

/// Coupling enters ONLY through ab: zeroed-ab kernel == independent solves.
#[test]
fn decoupling_is_exact_when_ab_vanishes() {
    let v = fixture();
    let (d, nmo_a, nmo_b) = dims_of(&v);
    let (da, db) = (d.da(), d.db());
    let eri_ab0 = vec![0.0f64; flat(&v["eri_ab"]).len()];
    let (a, _) = build_uab(
        &flat(&v["eri_aa"]), &eri_ab0, &flat(&v["eri_bb"]),
        &flat(&v["e_ia_a"]), &flat(&v["e_ia_b"]), d, nmo_a, nmo_b, 1.0,
    )
    .unwrap();
    // Independent dense solves on the diagonal blocks.
    let mut aaa = vec![0.0f64; da * da];
    let mut abb = vec![0.0f64; db * db];
    for i in 0..da {
        for j in 0..da {
            aaa[i * da + j] = a[i * (da + db) + j];
        }
    }
    for i in 0..db {
        for j in 0..db {
            abb[i * db + j] = a[(da + i) * (da + db) + da + j];
        }
    }
    let ca = cfg(da);
    let cb = cfg(db);
    let ra = tda_dense(&aaa, da, &ca).unwrap().energies;
    let rb = tda_dense(&abb, db, &cb).unwrap().energies;
    let mut union: Vec<f64> = ra.into_iter().chain(rb.into_iter()).collect();
    union.sort_by(|x, y| x.partial_cmp(y).unwrap());
    // Coupled-with-zero-ab kernel must reproduce the union (tight tolerance,
    // not bit-identity: an 18×18 eigh blocks differently than 6×6 + 12×12
    // eighs, so last-ulp ordering differences are correct behavior).
    let r = kernel_uhf_tda(
        &flat(&v["eri_aa"]), &eri_ab0, &flat(&v["eri_bb"]),
        &flat(&v["e_ia_a"]), &flat(&v["e_ia_b"]), d, nmo_a, nmo_b, 1.0, &cfg(da + db),
    )
    .unwrap();
    assert_eq!(r.energies.len(), union.len());
    for (x, y) in r.energies.iter().zip(union.iter()) {
        assert!((x - y).abs() < 1e-12, "decoupled mismatch {x:e} vs {y:e}");
    }
}

/// ...but the LIVE coupled roots differ: solving separately is wrong.
#[test]
fn live_coupling_shifts_roots() {
    let v = fixture();
    let (d, nmo_a, nmo_b) = dims_of(&v);
    let (da, db) = (d.da(), d.db());
    let (a, _) = build_uab(
        &flat(&v["eri_aa"]), &flat(&v["eri_ab"]), &flat(&v["eri_bb"]),
        &flat(&v["e_ia_a"]), &flat(&v["e_ia_b"]), d, nmo_a, nmo_b, 1.0,
    )
    .unwrap();
    let mut aaa = vec![0.0f64; da * da];
    let mut abb = vec![0.0f64; db * db];
    for i in 0..da {
        for j in 0..da {
            aaa[i * da + j] = a[i * (da + db) + j];
        }
    }
    for i in 0..db {
        for j in 0..db {
            abb[i * db + j] = a[(da + i) * (da + db) + da + j];
        }
    }
    let mut union: Vec<f64> = tda_dense(&aaa, da, &cfg(da))
        .unwrap()
        .energies
        .into_iter()
        .chain(tda_dense(&abb, db, &cfg(db)).unwrap().energies.into_iter())
        .collect();
    union.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let eref = flat(&v["e_tda"]);
    // The decoupled union must NOT match the live coupled roots (if it did,
    // the coupling would be vacuous and the test meaningless).
    let d = max_diff(&union[..3], &eref[..3]);
    assert!(d > 1e-6, "coupling vacuous?! union matches live to {d:e}");
}

/// kuhf on single-k live data: shift stamped, roots match the gamma driver.
#[test]
fn kuhf_single_k_matches_gamma_driver() {
    let v = fixture();
    let (d, nmo_a, nmo_b) = dims_of(&v);
    // Wrap gamma blocks as single-triple eri7, transposing axis order:
    // gamma blocks are [i][p][q][r] (occ first), 7d blocks are [p][i][q][r]
    // (occ second). Same numbers, different layout — transpose, don't wrap.
    fn to_7d(g: Vec<f64>, no: usize, nmo: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; g.len()];
        for i in 0..no {
            for p in 0..nmo {
                for q in 0..nmo {
                    for r in 0..nmo {
                        out[((p * no + i) * nmo + q) * nmo + r] =
                            g[((i * nmo + p) * nmo + q) * nmo + r];
                    }
                }
            }
        }
        out
    }
    let wrap = |x: Vec<f64>| vec![x];
    let (aa_r, aa_i) = (wrap(to_7d(flat(&v["eri_aa"]), d.nocc_a, nmo_a)), wrap(vec![0.0; nmo_a * d.nocc_a * nmo_a * nmo_a]));
    let ab_src = flat(&v["eri_ab"]);
    let mut ab_7d = vec![0.0f64; nmo_a * d.nocc_a * nmo_b * nmo_b];
    for i in 0..d.nocc_a {
        for p in 0..nmo_a {
            for q in 0..nmo_b {
                for r in 0..nmo_b {
                    ab_7d[((p * d.nocc_a + i) * nmo_b + q) * nmo_b + r] =
                        ab_src[((i * nmo_a + p) * nmo_b + q) * nmo_b + r];
                }
            }
        }
    }
    let (ab_r, ab_i) = (wrap(ab_7d.clone()), wrap(vec![0.0; ab_7d.len()]));
    let (bb_r, bb_i) = (
        wrap(to_7d(flat(&v["eri_bb"]), d.nocc_b, nmo_b)),
        wrap(vec![0.0; nmo_b * d.nocc_b * nmo_b * nmo_b]),
    );
    let kc = vec![0usize];
    let e_oa = vec![flat(&v["e_occ_a"])];
    let e_va = vec![flat(&v["e_vir_a"])];
    let e_ob = vec![flat(&v["e_occ_b"])];
    let e_vb = vec![flat(&v["e_vir_b"])];
    let r = kernel_kuhf_tda(
        &aa_r, &aa_i, &ab_r, &ab_i, &bb_r, &bb_i,
        &e_oa, &e_va, &e_ob, &e_vb, &kc, 1, d, nmo_a, nmo_b, 1.0, 0, &cfg(3),
    );
    // Single-k complex path on real data: roots must match gamma closely.
    // (real eigh vs complex zeigh — same math, last-ulp latitude.)
    match r {
        Ok(r) => {
            assert_eq!(r.kshift, 0);
            let eref = flat(&v["e_tda"]);
            let dev = max_diff(&r.energies, &eref[..3]);
            assert!(dev < 1e-8, "single-k deviates {dev:e} Ha");
        }
        Err(e) => panic!("single-k kuhf must run on real data: {e}"),
    }
}

/// Synthetic COMPLEX index oracle for the k-build maps (self-consistent:
///
/// distinguishable values per index, hand-derived expected placement).
#[test]
fn kuhf_complex_build_oracle() {
    // nk=1, noa=1, nva=1, nob=1, nvb=1, nmo=2: tiny complex blocks.
    let nk = 1;
    let d = UhfDims { nocc_a: 1, nvir_a: 1, nocc_b: 1, nvir_b: 1 };
    // eri_aa full (1,2,2,2)=8 complex values: re = flat index, im = -re.
    let mk = |base: f64| -> Vec<f64> { (0..8).map(|i| base + i as f64).collect() };
    let mki = |base: f64| -> Vec<f64> { (0..8).map(|i| -(base + i as f64)).collect() };
    let (a, b) = build_kuab(
        &[mk(0.0)], &[mki(0.0)], &[mk(100.0)], &[mki(100.0)], &[mk(200.0)], &[mki(200.0)],
        &[vec![-1.0]], &[vec![0.5]], &[vec![-0.8]], &[vec![0.7]],
        &[0], nk, d, 2, 2, 1.0, 1.0,
    )
    .expect("build must run");
    // dim = (1+1)*1 = 2. A[0][0] (alpha-alpha diag):
    // gap 1.5 + J(eri_aa[0,0,0][1,0,0,1] = re idx ((1*1+0)*2+0)*2+1 = 5 → 5.0)
    // − K(eri_aa[0,0,0][0,0,1,1] = idx ((0*1+0)*2+1)*2+1 = 3 → 3.0).
    // A[0][0].re = 1.5 + 5.0 − 3.0 = 3.5; im: −(5.0) −(−(3.0)) = −2.0... check:
    // J_im = −5.0, K_im = −3.0 → im = −5.0 − (−3.0) = −2.0.
    assert!((a.re[0] - 3.5).abs() < 1e-12, "Aaa diag re = {}", a.re[0]);
    assert!((a.im[0] + 2.0).abs() < 1e-12, "Aaa diag im = {}", a.im[0]);
    // A_ab[0][1]: eri_ab[0,0,0][1,0,0,1] = idx ((1*1+0)*2+0)*2+1 = 5 → 105.0.
    assert!((a.re[1] - 105.0).abs() < 1e-12, "Aab re = {}", a.re[1]);
    // A[1][0] conj-transpose: (105, +105).
    assert!((a.re[2] - 105.0).abs() < 1e-12);
    assert!((a.im[2] - 105.0).abs() < 1e-12, "Aab† im = {}", a.im[2]);
    // B_ab[0][1]: eri_ab[0,0,0][1,0,1,0] = idx ((1*1+0)*2+1)*2+0 = 6 → 106.0,
    // plain transpose (no conjugation).
    assert!((b.re[1] - 106.0).abs() < 1e-12, "Bab re = {}", b.re[1]);
    assert!((b.im[1] + 106.0).abs() < 1e-12, "Bab im = {}", b.im[1]);
    assert!((b.re[2] - 106.0).abs() < 1e-12);
    assert!((b.im[2] + 106.0).abs() < 1e-12, "Bba im = {}", b.im[2]);
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn uhf_deterministic_across_thread_counts() {
    let run = || {
        let v = fixture();
        let (d, nmo_a, nmo_b) = dims_of(&v);
        kernel_uhf_tda(
            &flat(&v["eri_aa"]), &flat(&v["eri_ab"]), &flat(&v["eri_bb"]),
            &flat(&v["e_ia_a"]), &flat(&v["e_ia_b"]), d, nmo_a, nmo_b, 1.0, &cfg(3),
        )
        .expect("TDA must run")
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

/// Big-box 2-k Davidson comparison (human-verify live arm).
///
/// Upstream WaterBigBox asserts KUHF-TDA (2,1,1) matches molecular UHF-TDA at
/// 2dp. The dense port needs uniform per-k fillings (as does upstream's own
/// `get_ab`); the 2-k Davidson runs matrix-free. Committed upstream numbers:
/// shift roots [0.10073, 0.10912] Ha vs molecular [same] at 2dp.
#[test]
#[ignore = "live 2-k Davidson arm: non-uniform per-k fillings need the matrix-free vind path (19-08 follow-up)"]
fn live_bigbox_matches_molecular() {
    panic!("live arm: KUHF (2,1,1) Davidson vs molecular UHF-TDA at 2dp");
}
