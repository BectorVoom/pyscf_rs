//! Gamma-point TDA/TDHF tests (19-06): Gate A, sorted and counted.
//!
//! Fixture `fixtures/tda_diamond_gamma.json` replicates upstream
//! `test_rhf.py::Diamond` (RS-DF, `gth-hf-rev`, minimal basis) from live
//! upstream 2.12.1: the half-transformed `eri`, `e_ia`, upstream's own
//! `(a, b)` for singlet, and TDA-singlet/triplet + TDHF-singlet roots.
//!
//! * `build_ab` vs upstream's `get_ab` element-wise (same inputs → tight).
//! * Roots SORTED with explicit `nroots` vs upstream at upstream's own 4dp
//!   (eV); triplet roots validate the triplet build end-to-end (upstream's
//!   `get_ab` only builds singlet matrices).
//! * TDA and TDHF get different numbers (different approximations).
//! * No assertion indexes a root positionally across methods.
//!
//! Run scoped: `cargo test -p pyscf-pbc-tdscf --test rhf`

use pyscf_pbc_tdscf::rhf::{build_ab, kernel_rhf_tda, kernel_rhf_tdhf};
use pyscf_pbc_tdscf::types::TdaConfig;
use serde_json::Value;

const HARTREE2EV: f64 = 27.211386018;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tda_diamond_gamma.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture must exist")).unwrap()
}

fn get(v: &Value, k: &str) -> Vec<f64> {
    let mut out = Vec::new();
    flatten(&v[k], &mut out);
    out
}

fn flatten(v: &Value, out: &mut Vec<f64>) {
    match v {
        Value::Number(x) => out.push(x.as_f64().unwrap()),
        Value::Array(items) => {
            for it in items {
                flatten(it, out);
            }
        }
        _ => panic!("unexpected JSON value in fixture"),
    }
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn cfg(singlet: bool, nroots: usize) -> TdaConfig {
    TdaConfig { nroots, conv_tol: 1e-9, max_cycle: 50, singlet, tda: true, kshift: 0 }
}

/// A/B build vs upstream's get_ab (singlet), element-wise.
#[test]
fn build_ab_matches_upstream_get_ab() {
    let v = fixture();
    let (nocc, nmo) = (v["nocc"].as_u64().unwrap() as usize, v["nmo"].as_u64().unwrap() as usize);
    let (a, b) = build_ab(&get(&v, "eri"), &get(&v, "e_ia"), nocc, nmo, true, 1.0)
        .expect("build must run");
    let (ar, br) = (get(&v["tda_sing"], "a"), get(&v["tda_sing"], "b"));
    assert_eq!(a.len(), ar.len());
    assert!(max_diff(&a, &ar) < 1e-10, "A deviates {:e}", max_diff(&a, &ar));
    assert!(max_diff(&b, &br) < 1e-10, "B deviates {:e}", max_diff(&b, &br));
}

/// Gate A1: TDA singlet roots, sorted, counted, at upstream's 4dp (eV).
#[test]
fn gate_a_tda_singlet() {
    let v = fixture();
    let (nocc, nmo) = (v["nocc"].as_u64().unwrap() as usize, v["nmo"].as_u64().unwrap() as usize);
    let r = kernel_rhf_tda(&get(&v, "eri"), &get(&v, "e_ia"), nocc, nmo, 1.0, &cfg(true, 2))
        .expect("TDA must run");
    assert_eq!(r.energies.len(), 2);
    assert!(r.energies[0] <= r.energies[1], "roots must be sorted");
    assert_eq!(r.kshift, 0);
    assert!(r.converged);
    let eref = get(&v["tda_sing"], "e");
    let d_ev = max_diff(
        &r.energies.iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
        &eref[..2].iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
    );
    assert!(d_ev < 5e-5, "TDA singlet deviates {d_ev:e} eV");
}

/// Triplet roots validate the triplet build end-to-end.
#[test]
fn gate_a_tda_triplet() {
    let v = fixture();
    let (nocc, nmo) = (v["nocc"].as_u64().unwrap() as usize, v["nmo"].as_u64().unwrap() as usize);
    let r = kernel_rhf_tda(&get(&v, "eri"), &get(&v, "e_ia"), nocc, nmo, 1.0, &cfg(false, 2))
        .expect("TDA must run");
    assert_eq!(r.energies.len(), 2);
    assert!(r.energies[0] <= r.energies[1]);
    let eref = get(&v["tda_trip"], "e");
    let d_ev = max_diff(
        &r.energies.iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
        &eref[..2].iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
    );
    assert!(d_ev < 5e-5, "TDA triplet deviates {d_ev:e} eV");
}

/// Gate A1: TDHF singlet roots via the Casida reduction.
#[test]
fn gate_a_tdhf_singlet() {
    let v = fixture();
    let (nocc, nmo) = (v["nocc"].as_u64().unwrap() as usize, v["nmo"].as_u64().unwrap() as usize);
    let mut c = cfg(true, 2);
    c.tda = false;
    let r = kernel_rhf_tdhf(&get(&v, "eri"), &get(&v, "e_ia"), nocc, nmo, 1.0, &c)
        .expect("TDHF must run");
    assert_eq!(r.energies.len(), 2);
    assert!(r.energies[0] <= r.energies[1]);
    let eref = get(&v["tdhf_sing"], "e");
    let d_ev = max_diff(
        &r.energies.iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
        &eref[..2].iter().map(|x| x * HARTREE2EV).collect::<Vec<_>>(),
    );
    assert!(d_ev < 5e-5, "TDHF singlet deviates {d_ev:e} eV");
}

/// TDA and TDHF are different approximations with different numbers.
#[test]
fn tda_and_tdhf_differ() {
    let v = fixture();
    let (nocc, nmo) = (v["nocc"].as_u64().unwrap() as usize, v["nmo"].as_u64().unwrap() as usize);
    let a = kernel_rhf_tda(&get(&v, "eri"), &get(&v, "e_ia"), nocc, nmo, 1.0, &cfg(true, 2)).unwrap();
    let mut c = cfg(true, 2);
    c.tda = false;
    let b = kernel_rhf_tdhf(&get(&v, "eri"), &get(&v, "e_ia"), nocc, nmo, 1.0, &c).unwrap();
    assert!((a.energies[0] - b.energies[0]).abs() > 1e-6);
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn tda_deterministic_across_thread_counts() {
    let run = || {
        let v = fixture();
        let (nocc, nmo) = (v["nocc"].as_u64().unwrap() as usize, v["nmo"].as_u64().unwrap() as usize);
        kernel_rhf_tda(&get(&v, "eri"), &get(&v, "e_ia"), nocc, nmo, 1.0, &cfg(true, 3))
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
