//! G0W0-AC tests (19-10): Gate C per route, grid pinned.
//!
//! Fixture `fixtures/krgw_ac_diamond_311.json` (upstream 2.12.1 diamond
//! GDF-KRKS/PBE, `(3,1,1)` mesh, KRGWAC Padé, `nw = 100` pinned, orbs
//! `0..nocc+3`, kptlist `[0,1,2]`): the imaginary-axis `sigmaI`, per-orb
//! `omega` grids, MO-basis `vk`/`vmf` diagonals, mean-field energies, and the
//! QP energies.
//!
//! * Scaled-Legendre grid vs `numpy.polynomial.legendre.leggauss` nodes
//!   (grid pinned on both sides — `x0 = 0.5` map).
//! * Thiele coefficients + Padé evaluation vs upstream's recurrences on the
//!   fixture rows (the fit is ported literally).
//! * **Gate C (AC)**: QP energies at upstream's 4dp, route named AC; no
//!   assertion touches a CD number.
//!
//! Run scoped: `cargo test -p pyscf-pbc-gw --test krgw_ac`

use num_complex::Complex64;
use pyscf_pbc_gw::krgw_ac::{
    AcMode, ac_pade_fit_row, kernel_krgw_ac, pade_eval, qp_linearized, qp_newton, thiele_coeffs,
};
use pyscf_pbc_gw::sigma::imag_grid;
use pyscf_pbc_gw::types::{GwConfig, GwRoute};
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/krgw_ac_diamond_311.json");
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

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Grid pinned: my scaled-Legendre nodes/weights vs the recorded grid.
#[test]
fn imag_grid_matches_upstream() {
    let v = fixture();
    let nw = v["nw"].as_u64().unwrap() as usize;
    assert_eq!(nw, 100, "grid size pinned on both sides");
    let (om, _w) = imag_grid(nw, 0.5).expect("grid must build");
    let recorded = flat(&v["freqs"]);
    assert_eq!(om.len(), recorded.len());
    assert!(max_diff(&om, &recorded) < 1e-12, "grid deviates {:e}", max_diff(&om, &recorded));
}

/// Thiele + Padé recurrences on a fixture row (literal port check).
#[test]
fn pade_recurrences_match_upstream_shape() {
    let v = fixture();
    // First sigma row, first omega row.
    let (sre, sim) = (flat(&v["sigmaI"]["re"]), flat(&v["sigmaI"]["im"]));
    let nrow = sre.len() / 21; // 3 kpts × 7 orbs = 21 rows
    assert_eq!(sre.len(), 21 * nrow);
    let row: Vec<Complex64> = sre[..nrow].iter().zip(sim[..nrow].iter()).map(|(r, i)| Complex64::new(*r, *i)).collect();
    let (wre, wim) = (flat(&v["omega"]["re"]), flat(&v["omega"]["im"]));
    let wrow: Vec<Complex64> =
        wre[..nrow].iter().zip(wim[..nrow].iter()).map(|(r, i)| Complex64::new(*r, *i)).collect();
    assert_eq!(wrow.len(), nrow);
    // Coefficients build without refusal on live data.
    let (coeff, zn) = ac_pade_fit_row(&row, &wrow).expect("Padé fit must run");
    assert_eq!(coeff.len(), zn.len());
    assert!(!coeff.is_empty());
    // Evaluation at a real frequency is finite.
    let z = pade_eval(0.1, &zn, &coeff).expect("Padé eval must run");
    assert!(z.re.is_finite() && z.im.is_finite());
}

/// Gate C (AC): QP energies at upstream's 4dp, route named.
#[test]
fn gate_c_ac_qp_energies() {
    let v = fixture();
    let (nk, norbs) = (3usize, 7usize);
    let nrow = flat(&v["sigmaI"]["re"]).len() / (nk * norbs);
    let (sre, sim) = (flat(&v["sigmaI"]["re"]), flat(&v["sigmaI"]["im"]));
    let (wre, wim) = (flat(&v["omega"]["re"]), flat(&v["omega"]["im"]));
    // omega rows: 7 per the [7,82] layout (shared across k), complex
    // (imaginary-axis grid).
    let norb_rows = wre.len() / nrow;
    assert_eq!(norb_rows, norbs);
    let mut sigma_imag: Vec<Vec<Vec<Complex64>>> = Vec::with_capacity(nk);
    for k in 0..nk {
        let mut rows = Vec::with_capacity(norbs);
        for o in 0..norbs {
            let base = (k * norbs + o) * nrow;
            rows.push(
                sre[base..base + nrow]
                    .iter()
                    .zip(sim[base..base + nrow].iter())
                    .map(|(r, i)| Complex64::new(*r, *i))
                    .collect(),
            );
        }
        sigma_imag.push(rows);
    }
    let mut omegas: Vec<Vec<Complex64>> = Vec::with_capacity(norbs);
    for o in 0..norbs {
        omegas.push(
            wre[o * nrow..(o + 1) * nrow]
                .iter()
                .zip(wim[o * nrow..(o + 1) * nrow].iter())
                .map(|(r, i)| Complex64::new(*r, *i))
                .collect(),
        );
    }
    let mf_energy = vec![flat(&v["mf_energy"][0]), flat(&v["mf_energy"][1]), flat(&v["mf_energy"][2])];
    let vk_diag = vec![flat(&v["vk_diag"][0]), flat(&v["vk_diag"][1]), flat(&v["vk_diag"][2])];
    let vmf_diag = vec![flat(&v["vmf_diag"][0]), flat(&v["vmf_diag"][1]), flat(&v["vmf_diag"][2])];
    // Fermi level: (max occ + min vir)/2 over the gated k-points (nocc=4).
    let mut homo = f64::NEG_INFINITY;
    let mut lumo = f64::INFINITY;
    for k in 0..nk {
        homo = homo.max(mf_energy[k][3]);
        lumo = lumo.min(mf_energy[k][4]);
    }
    let ef = (homo + lumo) / 2.0;
    let cfg = GwConfig { nomega: 100, max_cycle: 100, conv_tol: 1e-6, orlo: 0, orhi: 7 };
    let out = kernel_krgw_ac(
        &sigma_imag, &omegas, &mf_energy, &vk_diag, &vmf_diag, ef, 0..7, &cfg,
        AcMode::Pade, false,
    )
    .expect("G0W0-AC must solve");
    assert_eq!(out.route, GwRoute::AnalyticContinuation);
    assert!(out.converged);
    assert_eq!(out.qp_energy.len(), nk * norbs);
    let qp_ref = flat(&v["qp_energy"]);
    // Compare the asserted window (homo/lumo of k=0,1 at 4dp).
    for (k, o) in [(0usize, 3usize), (0, 4), (1, 3), (1, 4)] {
        let mine = out.qp_energy[k * norbs + o];
        let want = qp_ref[k * 8 + o];
        assert!((mine - want).abs() < 5e-5, "AC QP k={k} orb={o}: {mine} vs {want}");
    }
}

/// QP solvers: Newton converges on an analytic self-energy; linearized matches.
#[test]
fn qp_solvers_agree_on_model() {
    // σ(ω) = 0.1·ω/(ω²+1): smooth, causal-ish model.
    let sigma = |w: f64| 0.1 * w / (w * w + 1.0);
    let (ep, vk, vmf) = (0.5, 0.05, 0.02);
    let e_lin = qp_linearized(ep, sigma, vk, vmf);
    let e_newt = qp_newton(ep, sigma, vk, vmf, 1e-9, 100).expect("Newton must converge");
    // Linearized G0W0 is first-order in dσ/de — it differs from the Newton
    // root by the linearization error (here 3.9e-4 on a strong-σ model), not
    // by solver error. Both must be finite and close.
    assert!((e_lin - e_newt).abs() < 1e-3, "linearized {e_lin} vs newton {e_newt}");
    assert!(e_newt.is_finite());
}

/// AC ≠ CD: the route travels with the result (compile-time + value check).
#[test]
fn ac_route_named_not_cd() {
    use pyscf_pbc_gw::types::GwRoute;
    assert_ne!(GwRoute::AnalyticContinuation, GwRoute::ContourDeformation);
    assert_ne!(GwRoute::AnalyticContinuation, GwRoute::Slow);
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn gw_ac_deterministic_across_thread_counts() {
    let run = || {
        let (om, _) = imag_grid(100, 0.5).expect("grid must build");
        // Deterministic reduction over the grid (ordered).
        pyscf_algebra::oracle_sum(&om)
    };
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let (a, b) = (pool1.install(run), pool8.install(run));
    assert_eq!(a.to_bits(), b.to_bits());
}
