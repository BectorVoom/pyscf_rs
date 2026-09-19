//! 20-18 Task 4 — the `oracle-determinism` pattern (`ci.yml`, `RAYON_NUM_THREADS=1|8`)
//! extended to one PERIODIC driver.
//!
//! The molecular job asserts that `pyscf-algebra`'s reductions are bit-identical
//! across thread counts. D-PBC-17 (`PBC-MASTER-PLAN.md`) requires the same of every
//! periodic reduction that reaches an energy or a density matrix, and
//! `fft_jk_threads.rs` (pyscf-pbc-df) gates the J/K half of it. This file gates the
//! whole driver on top: a converged `KRHF` over FFTDF, run under explicit
//! `rayon::ThreadPool`s of 1 and 8 workers INSIDE ONE PROCESS (the `cphf_k.rs` /
//! `fft_jk_threads.rs` pattern), plus once on the global pool, whose size the CI
//! step varies through `RAYON_NUM_THREADS` and compares across the two processes
//! from the printed bits.
//!
//! This is a same-implementation A/B, so `to_bits()` equality is the correct
//! standard (`20-pbc-python-bindings/measurements/README.md` §3) — never a
//! tolerance.
//!
//! Fixture: He-fcc `sto-3g`, all-electron, Bohr, 2×2×2, mesh `[15,15,15]` —
//! exactly `krhf_bands_oracle.rs`'s fixture and convergence settings.

mod common;

use common::he_all_electron;
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_scf::{KScfConfig, KScfResult, Krhf};

const MESH: [usize; 3] = [15, 15, 15];
const NK: [usize; 3] = [2, 2, 2];

fn tight() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        ..KScfConfig::default()
    }
}

fn run_krhf() -> KScfResult {
    let cell = he_all_electron();
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, MESH).expect("FFTDF");
    let mf = Krhf::from_df(Box::new(df));
    let scf = mf.kernel(&tight()).expect("KRHF");
    assert!(scf.converged, "the fixture must converge");
    scf
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

fn assert_bit_identical(a: &KScfResult, b: &KScfResult, who: &str) {
    assert_eq!(
        a.e_tot.to_bits(),
        b.e_tot.to_bits(),
        "{who}: e_tot moved ({:.17e} vs {:.17e})",
        a.e_tot,
        b.e_tot
    );
    assert_eq!(
        a.e_elec.to_bits(),
        b.e_elec.to_bits(),
        "{who}: e_elec moved"
    );
    assert_eq!(a.cycles, b.cycles, "{who}: cycle count moved");
    assert_eq!(a.mo_energy.len(), b.mo_energy.len(), "{who}: block count");
    for (k, (ea, eb)) in a.mo_energy.iter().zip(&b.mo_energy).enumerate() {
        assert_eq!(ea.len(), eb.len(), "{who}: nmo at k={k}");
        for (i, (x, y)) in ea.iter().zip(eb).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{who}: mo_energy[{k}][{i}] moved");
        }
    }
}

#[test]
fn krhf_e_tot_is_bit_identical_at_1_and_8_rayon_threads() {
    let clock = std::time::Instant::now();
    let t1 = pool(1).install(run_krhf);
    let wall1 = clock.elapsed();
    let clock = std::time::Instant::now();
    let t8 = pool(8).install(run_krhf);
    let wall8 = clock.elapsed();
    // Evidence (not asserted: timings are noisy) that rayon is on the path at
    // all — a driver with no parallel reduction would pass the bit check vacuously.
    eprintln!("KRHF_DETERMINISM wall pool(1) = {wall1:.2?}, pool(8) = {wall8:.2?}");
    let global = run_krhf();

    // Non-vacuity: the 8-worker pool really has 8 workers, and the fixture is a
    // real k-point SCF (8 blocks, more than one cycle).
    assert_eq!(pool(8).install(rayon::current_num_threads), 8);
    assert_eq!(t1.mo_energy.len(), 8, "2x2x2 must give 8 k blocks");
    assert!(t1.cycles > 1, "a one-cycle SCF would not exercise the loop");

    assert_bit_identical(&t1, &t8, "pool(1) vs pool(8)");
    assert_bit_identical(&t1, &global, "pool(1) vs global pool");

    // The CI step compares this line between RAYON_NUM_THREADS=1 and =8 runs.
    eprintln!(
        "KRHF_DETERMINISM e_tot_bits=0x{:016x} e_tot={:.17e} cycles={} global_threads={}",
        global.e_tot.to_bits(),
        global.e_tot,
        global.cycles,
        rayon::current_num_threads()
    );
}
