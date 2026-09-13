//! Slow-reference GW tests (19-13): supercell equivalence + slow-checks-fast.
//!
//! * `slow_sigma_real` against a hand-computed single pole (wiring).
//! * Supercell equivalence, oracle-free: the replicated pole set reproduces
//!   the k-mean self-energy (1e-12 — accumulation order differs, so NOT
//!   bit-identity); at `nk = 1` all three drivers are term-for-term identical
//!   and ARE bit-identical; the gamma `gw_slow` alias matches the k-point
//!   driver at `nk = 1` bit-for-bit (upstream's aliasing, `:21-24`, honored).
//! * The slow route checks the fast one: `kgw_slow` vs 19-10's `krgw_ac` on a
//!   small cell inside Gate C.
//!
//! Run scoped: `cargo test -p pyscf-pbc-gw --test kgw_slow`

use num_complex::Complex64;
use pyscf_pbc_gw::gw_slow::kernel_gw_slow;
use pyscf_pbc_gw::kgw_slow::{kernel_kgw_slow, slow_sigma_real, LehmannPole, SLOW_ETA};
use pyscf_pbc_gw::kgw_slow_supercell::{
    kernel_kgw_slow_supercell, mean_primitive_sigma, replicate_supercell,
};
use pyscf_pbc_gw::krgw_ac::{kernel_krgw_ac, AcMode};
use pyscf_pbc_gw::types::GwConfig;

fn cfg(n: usize) -> GwConfig {
    GwConfig {
        nomega: n,
        max_cycle: 100,
        conv_tol: 1e-9,
        orlo: 0,
        orhi: 1,
    }
}

/// Single-pole hand check (wiring: sign side, `η²`, ordered sum).
#[test]
fn lehmann_matches_direct() {
    let poles = vec![LehmannPole {
        energy: 0.5,
        weight: 0.1,
    }];
    let (omega, eta) = (0.7f64, 1e-3f64);
    let got = slow_sigma_real(&poles, omega, eta);
    let d = omega - 0.5;
    let want = 0.1 * d / (d * d + eta * eta);
    assert!(
        (got - want).abs() < 1e-15,
        "lehmann deviates {:e}",
        (got - want).abs()
    );
    // Empty pole set is exactly zero (no silent NaN).
    assert_eq!(slow_sigma_real(&[], omega, eta), 0.0);
}

/// Supercell Γ on the replicated set == k-mean of the primitive set.
///
/// Same crystal, two samplings — no upstream involved. 1e-12 (order differs),
/// plus the `nk = 1` term-for-term bit-identity across all three drivers.
#[test]
fn supercell_equivalence_oracle_free() {
    let poles_k = vec![
        vec![
            LehmannPole {
                energy: -0.3,
                weight: 0.08,
            },
            LehmannPole {
                energy: 0.9,
                weight: 0.05,
            },
        ],
        vec![
            LehmannPole {
                energy: -0.25,
                weight: 0.07,
            },
            LehmannPole {
                energy: 0.95,
                weight: 0.06,
            },
        ],
    ];
    let sup = replicate_supercell(&poles_k);
    assert_eq!(sup.len(), 4);
    // Weights scaled by 1/nk.
    assert!((sup[0].weight - 0.04).abs() < 1e-15);
    for omega in [0.5, -0.1, 1.2, 3.0] {
        let a = slow_sigma_real(&sup, omega, SLOW_ETA);
        let b = mean_primitive_sigma(&poles_k, omega, SLOW_ETA);
        assert!(
            (a - b).abs() < 1e-12,
            "supercell identity at ω={omega}: {a} vs {b}"
        );
    }
    // nk = 1: all three drivers term-for-term identical ⇒ bit-identical roots.
    let mf = vec![0.5f64];
    let vk = vec![0.03f64];
    let vmf = vec![0.0f64];
    let one_k = vec![poles_k[0].clone()];
    let c = cfg(48);
    let slow_k = kernel_kgw_slow(
        std::slice::from_ref(&mf),
        std::slice::from_ref(&one_k),
        std::slice::from_ref(&vk),
        std::slice::from_ref(&vmf),
        0..1,
        &c,
        SLOW_ETA,
    )
    .expect("k-slow must solve");
    let slow_g = kernel_gw_slow(&mf, &one_k, &vk, &vmf, 0..1, &c).expect("gamma slow must solve");
    let slow_s = kernel_kgw_slow_supercell(&mf, &one_k, &vk, &vmf, 0..1, &c)
        .expect("supercell slow must solve");
    assert_eq!(slow_k.qp_energy[0].to_bits(), slow_g.qp_energy[0].to_bits());
    assert_eq!(slow_k.qp_energy[0].to_bits(), slow_s.qp_energy[0].to_bits());
    // nk = 2 supercell root vs k-mesh roots: same crystal, roots near each
    // other (mean-field pole sets differ per k, so roots differ — the identity
    // lives at the σ level above, recorded here as finite + converged).
    let mf2 = vec![vec![0.5], vec![0.52]];
    let vk2 = vec![vec![0.03], vec![0.03]];
    let vm2 = vec![vec![0.0], vec![0.0]];
    let out = kernel_kgw_slow(
        &mf2,
        &[vec![poles_k[0].clone()], vec![poles_k[1].clone()]],
        &vk2,
        &vm2,
        0..1,
        &c,
        SLOW_ETA,
    )
    .expect("k-mesh slow must solve");
    assert_eq!(out.qp_energy.len(), 2);
    assert!(out.qp_energy.iter().all(|e| e.is_finite()));
}

/// The slow route checks the fast one: `kgw_slow` vs `krgw_ac`.
///
/// Same two-pole content: slow evaluates the Lehmann form directly, fast
/// continues the imag-axis rows by Padé. Agreement inside Gate C (1e-4).
#[test]
fn slow_checks_fast_on_small_cell() {
    let ef = 0.0;
    let poles = vec![
        LehmannPole {
            energy: -0.3,
            weight: 0.08,
        },
        LehmannPole {
            energy: 0.9,
            weight: 0.05,
        },
    ];
    let (ep, d) = (0.5f64, 0.03f64);
    let c = cfg(48);
    // Slow: direct real-axis root.
    let slow = kernel_kgw_slow(
        &[vec![ep]],
        &[vec![poles.clone()]],
        &[vec![d]],
        &[vec![0.0]],
        0..1,
        &c,
        SLOW_ETA,
    )
    .expect("slow must solve");
    // Fast: imag-axis rows of the SAME poles, Padé-continued.
    let n = 48;
    let nodes: Vec<f64> = (0..n)
        .map(|i| 0.05 + 5.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let s_of = |z: Complex64| {
        let mut re = Vec::with_capacity(poles.len());
        let mut im = Vec::with_capacity(poles.len());
        for pole in &poles {
            let s = if ef >= pole.energy { 1.0 } else { -1.0 };
            let g = Complex64::new(pole.weight, 0.0)
                / (z - Complex64::new(pole.energy, 0.0) + Complex64::new(0.0, SLOW_ETA * s));
            re.push(g.re);
            im.push(g.im);
        }
        Complex64::new(
            pyscf_algebra::oracle_sum(&re),
            pyscf_algebra::oracle_sum(&im),
        )
    };
    let srow: Vec<Complex64> = nodes
        .iter()
        .map(|w| s_of(Complex64::new(0.0, *w)))
        .collect();
    let zrow: Vec<Complex64> = nodes.iter().map(|w| Complex64::new(0.0, *w)).collect();
    let fast = kernel_krgw_ac(
        &[vec![srow]],
        &[zrow],
        &[vec![ep]],
        &[vec![d]],
        &[vec![0.0]],
        ef,
        0..1,
        &c,
        AcMode::Pade,
        false,
    )
    .expect("fast must solve the same content");
    assert!(
        (slow.qp_energy[0] - fast.qp_energy[0]).abs() < 1e-4,
        "slow {} vs fast {}",
        slow.qp_energy[0],
        fast.qp_energy[0]
    );
    // Routes stay named: Slow ≠ AC.
    assert_ne!(slow.route, fast.route);
}

/// Determinism at 1 vs 8 rayon workers inside one process.
#[test]
fn gw_slow_deterministic_across_thread_counts() {
    let run = || {
        let poles = vec![
            LehmannPole {
                energy: -0.3,
                weight: 0.08,
            },
            LehmannPole {
                energy: 0.9,
                weight: 0.05,
            },
        ];
        slow_sigma_real(&poles, 0.5, SLOW_ETA).to_bits()
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
