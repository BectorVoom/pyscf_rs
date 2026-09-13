//! Unrestricted G0W0-AC tests (19-12): shared screening, closed-shell identity.
//!
//! * Structural: the driver takes ONE screening (`w_imag_k`) for both spins —
//!   two `W`s are unrepresentable; the closed-shell run is bit-identical
//!   across spins AND matches the restricted 19-10 driver on the same rows
//!   (the strongest check here needs no upstream at all).
//! * **Gate C (unrestricted AC)**: open-shell single-state model vs an
//!   independent bisection reference through the model's own real-axis form
//!   (a different algorithm from the driver's Newton-through-Padé path).
//!
//! Run scoped: `cargo test -p pyscf-pbc-gw --test kugw_ac`

use num_complex::Complex64;
use pyscf_pbc_gw::krgw_ac::{kernel_krgw_ac, AcMode};
use pyscf_pbc_gw::kugw_ac::{kernel_kugw_ac, sigma_row_on_grid, GW_ETA};
use pyscf_pbc_gw::types::GwConfig;

/// Smooth screening model `W(z) = w0·c²/(c² − z²)`: on the imag axis
/// `W(iω) = w0·c²/(c² + ω²)` (the injected rows), on the real axis
/// `W(ω) = w0·c²/(c² − ω²)` (the independent reference). `c` far from the
/// QP window so both sides are smooth.
const W0: f64 = 2.0;
const CC: f64 = 10.0;

fn imag_nodes(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.05 + 5.0 * (i as f64) / (n as f64 - 1.0))
        .collect()
}

fn screening_imag(nodes: &[f64]) -> Vec<Complex64> {
    nodes
        .iter()
        .map(|w| Complex64::new(W0 * CC * CC / (CC * CC + w * w), 0.0))
        .collect()
}

fn fit_grid(nodes: &[f64]) -> Vec<Complex64> {
    nodes.iter().map(|w| Complex64::new(0.0, *w)).collect()
}

/// True real-axis self-energy of the model (independent of the Padé path):
/// `σ^s(ω) = −(W(ω)/π)·Σ_m Re[1/((ω − e_m) + i·η·s_m)]`.
fn true_sigma_real(omega: f64, mf: &[f64], ef: f64) -> f64 {
    let w = W0 * CC * CC / (CC * CC - omega * omega);
    let mut acc = 0.0;
    for e in mf {
        let s = if ef >= *e { 1.0 } else { -1.0 };
        let d = Complex64::new(omega - e, GW_ETA * s);
        acc += (Complex64::new(1.0, 0.0) / d).re;
    }
    -w / std::f64::consts::PI * acc
}

/// Independent bisection root of `ω − ep − (σ(ω) + d) = 0` (not Newton).
fn bisect_root(ep: f64, d: f64, mf: &[f64], ef: f64) -> f64 {
    let f = |w: f64| w - ep - (true_sigma_real(w, mf, ef) + d);
    let (mut lo, mut hi) = (ep - 1.0, ep + 1.0);
    assert!(
        f(lo) * f(hi) < 0.0,
        "bisection bracket must straddle the root"
    );
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(lo) * f(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn cfg(orbs: std::ops::Range<usize>) -> GwConfig {
    GwConfig {
        nomega: 48,
        max_cycle: 100,
        conv_tol: 1e-9,
        orlo: orbs.start,
        orhi: orbs.end,
    }
}

/// Closed-shell system run unrestricted reproduces the restricted answer.
///
/// α and β spectra identical ⇒ per-spin energies bit-identical, and each
/// equals the restricted 19-10 driver on the same rows (1e-12). Oracle-free.
#[test]
fn closed_shell_identity() {
    let nodes = imag_nodes(48);
    let w = screening_imag(&nodes);
    let fg = fit_grid(&nodes);
    let ef = 0.0;
    let mf = vec![vec![-0.3, 0.5]];
    let vk = vec![vec![0.01, 0.04]];
    let vmf = vec![vec![0.005, 0.02]];
    let c = cfg(0..2);
    let (qa, qb) = kernel_kugw_ac(
        std::slice::from_ref(&w),
        &nodes,
        &[fg.clone(), fg.clone()],
        &mf,
        &mf,
        &vk,
        &vk,
        &vmf,
        &vmf,
        ef,
        0..2,
        &c,
        AcMode::Pade,
        false,
    )
    .expect("closed-shell UGW must solve");
    assert_eq!(qa.qp_energy.len(), 2);
    assert_eq!(qb.qp_energy.len(), 2);
    for i in 0..2 {
        assert_eq!(
            qa.qp_energy[i].to_bits(),
            qb.qp_energy[i].to_bits(),
            "closed-shell α vs β differ at orb {i}"
        );
    }
    // Against the restricted driver on the identical rows.
    let row = sigma_row_on_grid(&w, &nodes, &mf[0], ef, GW_ETA).expect("rows must build");
    let r = kernel_krgw_ac(
        &[vec![row.clone(), row]],
        &[fg.clone(), fg],
        &mf,
        &vk,
        &vmf,
        ef,
        0..2,
        &c,
        AcMode::Pade,
        false,
    )
    .expect("restricted driver must solve the same rows");
    for i in 0..2 {
        assert!(
            (qa.qp_energy[i] - r.qp_energy[i]).abs() < 1e-12,
            "U vs R at orb {i}: {} vs {}",
            qa.qp_energy[i],
            r.qp_energy[i]
        );
    }
}

/// Gate C (unrestricted AC): open-shell roots vs bisection references.
#[test]
fn gate_c_unrestricted_ac() {
    let nodes = imag_nodes(48);
    let w = screening_imag(&nodes);
    let fg = fit_grid(&nodes);
    let ef = 0.0;
    // Genuinely open-shell: different α/β spectra, one state each.
    let mf_a = vec![vec![0.5]];
    let mf_b = vec![vec![0.45]];
    let (d_a, d_b) = (0.03f64, -0.02f64);
    let vk_a = vec![vec![d_a]];
    let vk_b = vec![vec![d_b]];
    let zero = vec![vec![0.0]];
    let c = cfg(0..1);
    let (qa, qb) = kernel_kugw_ac(
        &[w],
        &nodes,
        &[fg],
        &mf_a,
        &mf_b,
        &vk_a,
        &vk_b,
        &zero,
        &zero,
        ef,
        0..1,
        &c,
        AcMode::Pade,
        false,
    )
    .expect("open-shell UGW must solve");
    let want_a = bisect_root(0.5, d_a, &mf_a[0], ef);
    let want_b = bisect_root(0.45, d_b, &mf_b[0], ef);
    // Per-route Gate C floor is 1e-4; Newton-through-Padé vs bisection.
    assert!(
        (qa.qp_energy[0] - want_a).abs() < 1e-4,
        "α root: {} vs bisection {want_a}",
        qa.qp_energy[0]
    );
    assert!(
        (qb.qp_energy[0] - want_b).abs() < 1e-4,
        "β root: {} vs bisection {want_b}",
        qb.qp_energy[0]
    );
    // The channels genuinely differ (open-shell content, not duplicated).
    assert!(
        (qa.qp_energy[0] - qb.qp_energy[0]).abs() > 1e-6,
        "spins must differ"
    );
}

/// Shape mismatches fail loudly.
#[test]
fn shapes_refused() {
    let nodes = imag_nodes(48);
    let w = screening_imag(&nodes);
    let fg = fit_grid(&nodes);
    let mf = vec![vec![0.5]];
    let vk = vec![vec![0.0]];
    let c = cfg(0..1);
    // Empty screening.
    let err = kernel_kugw_ac(
        &[Vec::new()],
        &nodes,
        std::slice::from_ref(&fg),
        &mf,
        &mf,
        &vk,
        &vk,
        &vk,
        &vk,
        0.0,
        0..1,
        &c,
        AcMode::Pade,
        false,
    )
    .expect_err("empty screening must be refused");
    assert!(matches!(
        err,
        pyscf_pbc_gw::error::PbcGwError::ShapeMismatch { .. }
    ));
    // Grid/row length mismatch.
    let short: Vec<f64> = nodes[..10].to_vec();
    let err = kernel_kugw_ac(
        &[w],
        &short,
        &[fg],
        &mf,
        &mf,
        &vk,
        &vk,
        &vk,
        &vk,
        0.0,
        0..1,
        &c,
        AcMode::Pade,
        false,
    )
    .expect_err("grid mismatch must be refused");
    assert!(matches!(
        err,
        pyscf_pbc_gw::error::PbcGwError::ShapeMismatch { .. }
    ));
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn kugw_ac_deterministic_across_thread_counts() {
    let run = || {
        let nodes = imag_nodes(48);
        let w = screening_imag(&nodes);
        let row = sigma_row_on_grid(&w, &nodes, &[0.5], 0.0, GW_ETA).expect("rows must build");
        (
            row[0].re.to_bits(),
            row[0].im.to_bits(),
            row[47].re.to_bits(),
        )
    };
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    assert_eq!(pool1.install(run), pool8.install(run));
}
