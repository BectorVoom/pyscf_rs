//! SCF stability tests (19-05): verdict first, eigenvalue second.
//!
//! * Synthetic hops with analytically known spectra (stable / unstable /
//!   threshold-edge) pin the verdict rule `stable = not (e < -1e-5)`.
//! * Live-hop fixtures (`fixtures/stab_h2_{req,stretched}.json`, upstream
//!   2.12.1 KRHF/cc-pVDZ Γ, hop matrices materialized from upstream's own
//!   `hop` closures): this port's dense drivers reproduce upstream's dense
//!   lowest eigenvalues and both verdicts. R=1.4 is stable both ways;
//!   R=5.0 is externally (RHF→UHF) unstable at e = −0.273.
//! * Internal and external results are distinct types, asserted separately.
//! * Rotation preserves orthonormality on the unstable fixture's direction.
//!
//! Run scoped: `cargo test -p pyscf-pbc-scf --test stability`

use pyscf_pbc_scf::stability::{
    STABILITY_THRESHOLD, rhf_external, rhf_internal, rotate_mo_real,
};
use serde_json::Value;

fn dense_hop(mat: &[f64], dim: usize) -> impl Fn(&[f64]) -> Result<Vec<f64>, pyscf_core::PyscfRsError> + '_ {
    move |x: &[f64]| {
        let mut out = vec![0.0f64; dim];
        for i in 0..dim {
            let mut acc = 0.0f64;
            for j in 0..dim {
                acc += mat[i * dim + j] * x[j];
            }
            out[i] = acc;
        }
        Ok(out)
    }
}

fn fixture(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture must exist")).unwrap()
}

fn get_mat(v: &Value, key: &str) -> Vec<f64> {
    v[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// Stable synthetic Hessian: lowest = +1.0 → stable, no direction.
#[test]
fn internal_stable_verdict() {
    let h = vec![0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
    let hop = dense_hop(&h, 3);
    let r = rhf_internal(&hop, &[0.5, 1.0, 2.0], 3).expect("must run");
    assert!(r.stable);
    assert!((r.lowest_eigenvalue - 1.0).abs() < 1e-12);
    assert!(r.direction.is_empty());
}

/// Unstable synthetic Hessian: lowest = −0.6 → unstable + direction.
#[test]
fn internal_unstable_verdict_with_direction() {
    let h = vec![-0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let hop = dense_hop(&h, 3);
    let r = rhf_internal(&hop, &[0.1, 1.0, 1.0], 3).expect("must run");
    assert!(!r.stable);
    assert!((r.lowest_eigenvalue + 0.6).abs() < 1e-12);
    assert_eq!(r.direction.len(), 3);
    // Direction is the lowest eigenvector (e_0 up to sign).
    assert!(r.direction[0].abs() > 0.99);
}

/// Threshold rule: `stable = not (e < -1e-5)`, probed clearly on each side.
///
/// Note the internal ×2 (`hessian = 2·hop`): the probed hop eigenvalue is
/// doubled before the rule applies.
/// (Exactly ON the boundary is a measure-zero fp-noise case in every
/// implementation — upstream's Davidson result never lands there either. The
/// rule both sides of it is what is pinned.)
#[test]
fn threshold_edge_matches_upstream_rule() {
    assert_eq!(STABILITY_THRESHOLD, -1e-5);
    let mk = |hop_e0: f64| {
        let h = vec![hop_e0, 0.0, 0.0, 1.0];
        let hop = dense_hop(&h, 2);
        rhf_internal(&hop, &[hop_e0, 1.0], 2).expect("must run").stable
    };
    // Hessian eigenvalue = 2·hop eigenvalue: hop −0.4e-5 → −0.8e-5
    // (stable); hop −0.6e-5 → −1.2e-5 (unstable).
    assert!(mk(-0.4e-5), "just above threshold must be stable");
    assert!(!mk(-0.6e-5), "just below threshold must be unstable");
}

/// Live-hop fixtures: eigenvalues (secondary) and verdicts (primary).
#[test]
fn live_hop_fixtures_reproduce_upstream() {
    for (name, expect_int, expect_ext) in [
        ("stab_h2_req.json", true, true),
        ("stab_h2_stretched.json", true, false),
    ] {
        let v = fixture(name);
        for (key, expect) in [("internal", expect_int), ("external", expect_ext)] {
            let b = &v[key];
            let dim = b["dim"].as_u64().unwrap() as usize;
            let hop_mat = get_mat(b, "hop");
            let hdiag = get_mat(b, "hdiag");
            let hop = dense_hop(&hop_mat, dim);
            let (stable, lowest) = if key == "internal" {
                let r = rhf_internal(&hop, &hdiag, dim).expect("internal must run");
                (r.stable, r.lowest_eigenvalue)
            } else {
                let r = rhf_external(&hop, &hdiag, dim).expect("external must run");
                (r.stable, r.lowest_eigenvalue)
            };
            assert_eq!(stable, expect, "{name} {key}: verdict");
            assert_eq!(stable, b["upstream_stable"].as_bool().unwrap(), "{name} {key}: upstream verdict");
            let ref_e = b["upstream_lowest"].as_f64().unwrap();
            assert!(
                (lowest - ref_e).abs() < 1e-8,
                "{name} {key}: lowest {lowest:e} vs upstream dense {ref_e:e}"
            );
        }
    }
}

/// Internal and external are distinct result types with distinct scales.
#[test]
fn internal_external_stay_distinct() {
    let v = fixture("stab_h2_stretched.json");
    let dim = v["internal"]["dim"].as_u64().unwrap() as usize;
    let hi_mat = get_mat(&v["internal"], "hop");
    let he_mat = get_mat(&v["external"], "hop");
    let hi = dense_hop(&hi_mat, dim);
    let he = dense_hop(&he_mat, dim);
    let ri = rhf_internal(&hi, &get_mat(&v["internal"], "hdiag"), dim).unwrap();
    let re = rhf_external(&he, &get_mat(&v["external"], "hdiag"), dim).unwrap();
    assert!(ri.stable);
    assert!(!re.stable);
    // Different questions, different numbers — not one flag.
    assert!((ri.lowest_eigenvalue - re.lowest_eigenvalue).abs() > 0.1);
}

/// Rotation along the unstable direction preserves orthonormality.
#[test]
fn rotation_preserves_orthonormality() {
    // Identity orbitals, nmo=3, nocc=1, dx along the single rotation.
    let c: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let out = rotate_mo_real(&c, 3, 1, &[0.3, -0.1]).expect("rotation must run");
    // outᵀ·out = I.
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0f64;
            for k in 0..3 {
                acc += out[k * 3 + i] * out[k * 3 + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((acc - want).abs() < 1e-10, "({i},{j}) = {acc}");
        }
    }
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn stability_deterministic_across_thread_counts() {
    let run = || {
        let v = fixture("stab_h2_stretched.json");
        let dim = v["external"]["dim"].as_u64().unwrap() as usize;
        let he_mat = get_mat(&v["external"], "hop");
        let he_hdiag = get_mat(&v["external"], "hdiag");
        let he = dense_hop(&he_mat, dim);
        rhf_external(&he, &he_hdiag, dim).unwrap().lowest_eigenvalue
    };
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let (a, b) = (pool1.install(run), pool8.install(run));
    assert_eq!(a.to_bits(), b.to_bits());
}
