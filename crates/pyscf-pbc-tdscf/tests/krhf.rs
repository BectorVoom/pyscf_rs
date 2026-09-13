//! k-point TDA/TDHF tests (19-07): Gate A per shift, never across.
//!
//! Fixture `fixtures/ktda_diamond_211.json` (upstream 2.12.1 diamond
//! RS-DF-KRHF, `(2,1,1)` mesh): `ao2mo_7d` blocks, exxdiv-none spectra,
//! per-shift `kconserv`, upstream's own `(a, b)` and roots for shifts 0,1.
//! The shifts are NOT symmetry-equivalent (10.957 vs 11.042 eV) — a global
//! cross-shift sort is observably wrong here, and the tests pin per-shift
//! reporting (including a demonstration that pooling mixes them).
//!
//! TDHF: the fixture shifts are genuinely complex (`B ≠ B†` at 5e-4), so the
//! refusal is asserted; the real-valued path runs on a synthetic diagonal
//! problem with hand-computed roots.
//!
//! Run scoped: `cargo test -p pyscf-pbc-tdscf --test krhf`

use pyscf_pbc_tdscf::krhf::{build_kab, kernel_krhf_tda, kernel_krhf_tdhf};
use pyscf_pbc_tdscf::types::TdaConfig;
use serde_json::Value;

const HARTREE2EV: f64 = 27.211386018;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ktda_diamond_211.json");
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

struct Fix {
    nk: usize,
    nocc: usize,
    nmo: usize,
    eri7re: Vec<Vec<f64>>,
    eri7im: Vec<Vec<f64>>,
    e_occ: Vec<Vec<f64>>,
    e_vir: Vec<Vec<f64>>,
}

fn load() -> (Fix, Value) {
    let v = fixture();
    let (nk, nocc, nmo) = (
        v["nkpts"].as_u64().unwrap() as usize,
        v["nocc"].as_u64().unwrap() as usize,
        v["nmo"].as_u64().unwrap() as usize,
    );
    let nmo_nocc = nmo * (v["nocc"].as_u64().unwrap() as usize) * nmo * nmo;
    let (all_re, all_im) = (flat(&v["eri7re"]), flat(&v["eri7im"]));
    assert_eq!(all_re.len(), nk * nk * nk * nmo_nocc);
    let mut eri7re = Vec::new();
    let mut eri7im = Vec::new();
    for t in 0..nk * nk * nk {
        eri7re.push(all_re[t * nmo_nocc..(t + 1) * nmo_nocc].to_vec());
        eri7im.push(all_im[t * nmo_nocc..(t + 1) * nmo_nocc].to_vec());
    }
    let e_occ = vec![flat(&v["e_occ"][0]), flat(&v["e_occ"][1])];
    let e_vir = vec![flat(&v["e_vir"][0]), flat(&v["e_vir"][1])];
    (Fix { nk, nocc, nmo, eri7re, eri7im, e_occ, e_vir }, v)
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn cfg(singlet: bool, nroots: usize) -> TdaConfig {
    TdaConfig { nroots, conv_tol: 1e-9, max_cycle: 50, singlet, tda: true, kshift: 0 }
}

/// k-shift A/B build vs upstream's get_ab, element-wise, both shifts.
#[test]
fn build_kab_matches_upstream_per_shift() {
    let (f, v) = load();
    for ks in [0, 1] {
        let kc: Vec<usize> = flat(&v["shifts"][&ks.to_string()]["kconserv"])
            .iter()
            .map(|x| *x as usize)
            .collect();
        let (a, b) = build_kab(
            &f.eri7re, &f.eri7im, &f.e_occ, &f.e_vir, &kc, f.nk, f.nocc, f.nmo, true, 1.0,
            1.0 / f.nk as f64,
        )
        .expect("build must run");
        let sh = &v["shifts"][&ks.to_string()];
        let (ar, ai) = (flat(&sh["a"]["re"]), flat(&sh["a"]["im"]));
        let (br, bi) = (flat(&sh["b"]["re"]), flat(&sh["b"]["im"]));
        assert!(max_diff(&a.re, &ar) < 1e-10, "shift {ks} A.re {:e}", max_diff(&a.re, &ar));
        assert!(max_diff(&a.im, &ai) < 1e-10, "shift {ks} A.im {:e}", max_diff(&a.im, &ai));
        assert!(max_diff(&b.re, &br) < 1e-10, "shift {ks} B.re {:e}", max_diff(&b.re, &br));
        assert!(max_diff(&b.im, &bi) < 1e-10, "shift {ks} B.im {:e}", max_diff(&b.im, &bi));
    }
}

/// Gate A1 headline: KRHF-TDA lowest excitation per shift at upstream's 4dp.
#[test]
fn gate_a_krhf_tda_per_shift() {
    let (f, v) = load();
    for ks in [0, 1] {
        let kc: Vec<usize> = flat(&v["shifts"][&ks.to_string()]["kconserv"])
            .iter()
            .map(|x| *x as usize)
            .collect();
        let r = kernel_krhf_tda(
            &f.eri7re, &f.eri7im, &f.e_occ, &f.e_vir, &kc, f.nk, f.nocc, f.nmo, 1.0, ks,
            &cfg(true, 1),
        )
        .expect("TDA must run");
        assert_eq!(r.kshift, ks, "roots must carry their shift");
        assert!(r.converged);
        let eref = flat(&v["shifts"][&ks.to_string()]["e_tda_sing"]);
        let d_ev = (r.energies[0] - eref[0]).abs() * HARTREE2EV;
        assert!(d_ev < 5e-5, "shift {ks} deviates {d_ev:e} eV");
    }
}

/// Triplet at shift 0 (validates the triplet build end-to-end at k).
#[test]
fn gate_a_krhf_tda_triplet() {
    let (f, v) = load();
    let kc: Vec<usize> = flat(&v["shifts"]["0"]["kconserv"]).iter().map(|x| *x as usize).collect();
    let r = kernel_krhf_tda(
        &f.eri7re, &f.eri7im, &f.e_occ, &f.e_vir, &kc, f.nk, f.nocc, f.nmo, 1.0, 0,
        &cfg(false, 2),
    )
    .expect("TDA must run");
    assert_eq!(r.kshift, 0);
    assert!(r.energies[0] <= r.energies[1]);
    let eref = flat(&v["shifts"]["0"]["e_tda_trip"]);
    let d = max_diff(
        &r.energies.iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
        &eref[..2].iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
    );
    assert!(d < 5e-5, "triplet deviates {d:e} eV");
}

/// Cross-shift pooling is observably wrong here: the shifts differ by 0.085
/// eV (10.957 vs 11.042), far above the 5e-5 gate — a global sort across
/// shifts cannot pass the per-shift gate.
#[test]
fn cross_shift_sorting_is_caught() {
    let (f, v) = load();
    let mut lows = Vec::new();
    for ks in [0, 1] {
        let kc: Vec<usize> = flat(&v["shifts"][&ks.to_string()]["kconserv"])
            .iter()
            .map(|x| *x as usize)
            .collect();
        let r = kernel_krhf_tda(
            &f.eri7re, &f.eri7im, &f.e_occ, &f.e_vir, &kc, f.nk, f.nocc, f.nmo, 1.0, ks,
            &cfg(true, 1),
        )
        .unwrap();
        lows.push(r.energies[0]);
    }
    let gap_ev = (lows[0] - lows[1]).abs() * HARTREE2EV;
    assert!(gap_ev > 1e-4, "shifts unexpectedly equivalent ({gap_ev:e} eV)");
    // A pooled-and-sorted report would attribute shift 1's root to shift 0's
    // gate (or vice versa) and miss by the full gap:
    assert!(gap_ev > 5e-5, "pooling would pass — fixture too symmetric");
}

/// Complex TDHF refuses loudly (non-Hermitian B at 5e-4, measured).
#[test]
fn complex_tdhf_refuses() {
    let (f, v) = load();
    let kc: Vec<usize> = flat(&v["shifts"]["0"]["kconserv"]).iter().map(|x| *x as usize).collect();
    let mut c = cfg(true, 1);
    c.tda = false;
    let r = kernel_krhf_tdhf(
        &f.eri7re, &f.eri7im, &f.e_occ, &f.e_vir, &kc, f.nk, f.nocc, f.nmo, 1.0, 0, &c,
    );
    assert!(r.is_err(), "complex TDHF must refuse, not truncate");
}

/// Real-valued TDHF path on a synthetic diagonal problem (hand roots).
#[test]
fn real_tdhf_path_on_diagonal_problem() {
    // nk=1, nocc=1, nvir=2: A=diag(4,9), B=0 → w=(4,9).
    let nk = 1;
    let nmo = 3;
    let nocc = 1;
    // eri7 single block (nmo,nocc,nmo,nmo) zeros.
    let z = vec![0.0f64; nmo * nocc * nmo * nmo];
    let (eri7re, eri7im) = (vec![z.clone()], vec![z]);
    // Gaps via spectra: e_vir[0]=[4,9], e_occ[0]=[0].
    let e_occ = vec![vec![0.0]];
    let e_vir = vec![vec![4.0, 9.0]];
    let kc = vec![0usize];
    let mut c = cfg(true, 2);
    c.tda = false;
    let r = kernel_krhf_tdhf(&eri7re, &eri7im, &e_occ, &e_vir, &kc, nk, nocc, nmo, 1.0, 0, &c)
        .expect("real TDHF must run");
    assert!((r.energies[0] - 4.0).abs() < 1e-9);
    assert!((r.energies[1] - 9.0).abs() < 1e-9);
    // And TDA agrees on the diagonal problem.
    let rt = kernel_krhf_tda(&eri7re, &eri7im, &e_occ, &e_vir, &kc, nk, nocc, nmo, 1.0, 0, &cfg(true, 2))
        .expect("TDA must run");
    assert!((rt.energies[0] - 4.0).abs() < 1e-9);
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn ktda_deterministic_across_thread_counts() {
    let run = || {
        let (f, v) = load();
        let kc: Vec<usize> = flat(&v["shifts"]["0"]["kconserv"]).iter().map(|x| *x as usize).collect();
        kernel_krhf_tda(&f.eri7re, &f.eri7im, &f.e_occ, &f.e_vir, &kc, f.nk, f.nocc, f.nmo, 1.0, 0, &cfg(true, 2))
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
