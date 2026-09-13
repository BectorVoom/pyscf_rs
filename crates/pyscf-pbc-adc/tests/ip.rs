//! ADC(2) IP tests (19-15): sigma oracle, Gate D.
//!
//! Fixture `fixtures/kadc_he2.json` (upstream 2.12.1 He2/`gth-dzv` DF-KRADC,
//! `[1,1,2]` mesh, kpt 0): the six ERI blocks, `t2[0]`, `M_ij`, `M_ab`, and
//! IP/EA roots. The pipeline runs MY code on upstream's blocks at every
//! stage — amplitudes → intermediates → sigma-roots — so each stage is gated
//! independently before Gate D.
//!
//! * `t2_1` vs upstream `t2[0]` element-wise (complex, conjugated).
//! * `M_ij` vs upstream `get_imds` element-wise.
//! * **Gate D**: sorted, counted roots at upstream's 4dp.
//!
//! Run scoped: `cargo test -p pyscf-pbc-adc --test ip`

use pyscf_algebra::CTensor;
use pyscf_pbc_adc::amplitudes::t2_first_order;
use pyscf_pbc_adc::ip::{build_m_ij, kernel_ip};
use pyscf_pbc_adc::kadc_ao2mo::KadcEris;
use pyscf_pbc_adc::types::AdcConfig;
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/kadc_he2.json");
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

fn eris_of(v: &Value) -> KadcEris {
    let (nk, no, nv) = (
        v["nkpts"].as_u64().unwrap() as usize,
        v["nocc"].as_u64().unwrap() as usize,
        v["nvir"].as_u64().unwrap() as usize,
    );
    let ct = |name: &str| CTensor {
        re: flat(&v["blocks"][name]["re"]),
        im: flat(&v["blocks"][name]["im"]),
    };
    KadcEris::from_blocks(
        nk, no, nv,
        ct("oooo"), ct("oovv"), ct("ovoo"), ct("ovov"), ct("ovvv"), ct("ovvo"),
    )
    .expect("fixture blocks must fit")
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Amplitudes on live blocks vs upstream t2[0].
#[test]
fn t2_matches_upstream_on_live_blocks() {
    let v = fixture();
    let eris = eris_of(&v);
    let kc: Vec<usize> = flat(&v["kconserv"]).iter().map(|x| *x as usize).collect();
    let e_occ: Vec<Vec<f64>> = vec![flat(&v["e_occ"][0]), flat(&v["e_occ"][1])];
    let e_vir: Vec<Vec<f64>> = vec![flat(&v["e_vir"][0]), flat(&v["e_vir"][1])];
    let amps = t2_first_order(&eris, &e_occ, &e_vir, &kc).expect("t2 must build");
    let (tr, ti) = (flat(&v["t2"]["re"]), flat(&v["t2"]["im"]));
    assert_eq!(amps.t2_1.re.len(), tr.len());
    assert!(max_diff(&amps.t2_1.re, &tr) < 1e-10, "t2.re deviates {:e}", max_diff(&amps.t2_1.re, &tr));
    assert!(max_diff(&amps.t2_1.im, &ti) < 1e-10, "t2.im deviates {:e}", max_diff(&amps.t2_1.im, &ti));
}

/// M_ij vs upstream get_imds.
#[test]
fn m_ij_matches_upstream() {
    let v = fixture();
    let eris = eris_of(&v);
    let kc: Vec<usize> = flat(&v["kconserv"]).iter().map(|x| *x as usize).collect();
    let e_occ: Vec<Vec<f64>> = vec![flat(&v["e_occ"][0]), flat(&v["e_occ"][1])];
    let e_vir: Vec<Vec<f64>> = vec![flat(&v["e_vir"][0]), flat(&v["e_vir"][1])];
    let amps = t2_first_order(&eris, &e_occ, &e_vir, &kc).expect("t2 must build");
    let m = build_m_ij(&amps, &eris, &e_occ, &kc).expect("M_ij must build");
    let nk = v["nkpts"].as_u64().unwrap() as usize;
    let no = v["nocc"].as_u64().unwrap() as usize;
    for k in 0..nk {
        // Upstream M_ij is (nkpts,nocc,nocc) C-order.
        let base = k * no * no;
        let rr = flat(&v["M_ij"]["re"])[base..base + no * no].to_vec();
        let ri = flat(&v["M_ij"]["im"])[base..base + no * no].to_vec();
        assert!(max_diff(&m.m_ij[k].re, &rr) < 1e-9, "M_ij[{k}].re deviates {:e}", max_diff(&m.m_ij[k].re, &rr));
        assert!(max_diff(&m.m_ij[k].im, &ri) < 1e-9, "M_ij[{k}].im deviates {:e}", max_diff(&m.m_ij[k].im, &ri));
    }
}

/// Gate D: IP roots sorted + counted at upstream's 4dp.
#[test]
fn gate_d_ip_roots() {
    let v = fixture();
    let eris = eris_of(&v);
    let kc: Vec<usize> = flat(&v["kconserv"]).iter().map(|x| *x as usize).collect();
    let e_occ: Vec<Vec<f64>> = vec![flat(&v["e_occ"][0]), flat(&v["e_occ"][1])];
    let e_vir: Vec<Vec<f64>> = vec![flat(&v["e_vir"][0]), flat(&v["e_vir"][1])];
    let amps = t2_first_order(&eris, &e_occ, &e_vir, &kc).expect("t2 must build");
    let m = build_m_ij(&amps, &eris, &e_occ, &kc).expect("M_ij must build");
    let cfg = AdcConfig::default();
    let roots = kernel_ip(&m, &eris, &e_occ, &e_vir, &kc, 0, &cfg).expect("IP must solve");
    assert_eq!(roots.energies.len(), 3);
    assert!(roots.energies.windows(2).all(|w| w[0] <= w[1]), "roots must be sorted");
    assert!(roots.converged);
    let eref = flat(&v["ip_roots"]);
    let d = max_diff(&roots.energies, &eref[..3]);
    assert!(d < 5e-5, "Gate D IP deviates {d:e}");
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn ip_deterministic_across_thread_counts() {
    let run = || {
        let v = fixture();
        let eris = eris_of(&v);
        let kc: Vec<usize> = flat(&v["kconserv"]).iter().map(|x| *x as usize).collect();
        let e_occ: Vec<Vec<f64>> = vec![flat(&v["e_occ"][0]), flat(&v["e_occ"][1])];
        let e_vir: Vec<Vec<f64>> = vec![flat(&v["e_vir"][0]), flat(&v["e_vir"][1])];
        let amps = t2_first_order(&eris, &e_occ, &e_vir, &kc).expect("t2 must build");
        let m = build_m_ij(&amps, &eris, &e_occ, &kc).expect("M_ij must build");
        kernel_ip(&m, &eris, &e_occ, &e_vir, &kc, 0, &AdcConfig::default())
            .expect("IP must solve")
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
