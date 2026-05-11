//! Plan 03-11 Task 1 RED — unit tests for the kernel-internals defaults.
//!
//! Each default_* function gets a deterministic 2-AO toy fixture. We
//! avoid running the full SCF loop here (that needs `int2e_sph` which
//! Phase 2 deferred to plan 02-09); the H2 end-to-end smoke is in
//! `no_overrides_drives_kernel.rs` (#[ignore]'d until plan 03-10).

use pyscf_core::{Density, Energy, MOCoefficients};
use pyscf_scf::{
    energy::{default_energy_elec, default_energy_tot},
    occ::default_get_occ,
    rdm::default_make_rdm1,
};

#[test]
fn occ_aufbau_fill_closed_shell() {
    // 4 MOs, 2 electrons → only the lowest MO is doubly occupied.
    let mo_energy = vec![-1.0, -0.5, 0.5, 1.0];
    let occ = default_get_occ(&mo_energy, 2).expect("occ");
    assert_eq!(occ, vec![2.0, 0.0, 0.0, 0.0]);
}

#[test]
fn occ_aufbau_fill_four_electrons() {
    // 4 MOs, 4 electrons → lowest two doubly occupied.
    let mo_energy = vec![-1.0, -0.5, 0.5, 1.0];
    let occ = default_get_occ(&mo_energy, 4).expect("occ");
    assert_eq!(occ, vec![2.0, 2.0, 0.0, 0.0]);
}

#[test]
fn occ_aufbau_rejects_odd_nelec() {
    let mo_energy = vec![-1.0, 0.0];
    let r = default_get_occ(&mo_energy, 1);
    assert!(r.is_err(), "RHF Aufbau must reject odd nelec");
}

#[test]
fn make_rdm1_diagonal_mos() {
    // Trivial MO: C = I (2×2), occ = (2, 0). D = C · diag(occ) · C^T = diag(2, 0).
    // F-order: C[mu + i*nao]. C = identity ⇒ C[0]=1, C[1]=0, C[2]=0, C[3]=1.
    let mo = MOCoefficients {
        nao: 2,
        nmo: 2,
        data: vec![1.0, 0.0, 0.0, 1.0], // column-major identity
        energies: vec![-1.0, 1.0],
        occupations: vec![2.0, 0.0],
    };
    let dm = default_make_rdm1(&mo).expect("rdm");
    assert_eq!(dm.nao, 2);
    // D[0,0] = Σ_i occ[i] * C[0,i] * C[0,i] = 2 * 1 * 1 + 0 = 2.
    // D[0,1] = 2 * 1 * 0 + 0 = 0.
    // D[1,1] = 2 * 0 * 0 + 0 * 1 * 1 = 0.
    assert!((dm.data[0] - 2.0).abs() < 1e-12, "D[0,0] = 2 (one doubly-occupied MO at AO 0)");
    assert!((dm.data[1]).abs() < 1e-12, "D[0,1] = 0");
    assert!((dm.data[2]).abs() < 1e-12, "D[1,0] = 0");
    assert!((dm.data[3]).abs() < 1e-12, "D[1,1] = 0 (second MO empty)");
}

#[test]
fn energy_elec_simple() {
    // D = diag(1, 1), h1e = diag(-1, -1), vhf = diag(0.5, 0.5).
    // E_1e = sum D*h1e = -1 - 1 = -2
    // E_coul = 0.5 * sum D*vhf = 0.5 * (0.5 + 0.5) = 0.5
    // E_elec = E_1e + E_coul = -1.5
    let dm = Density { nao: 2, data: vec![1.0, 0.0, 0.0, 1.0] };
    let h1e = Density { nao: 2, data: vec![-1.0, 0.0, 0.0, -1.0] };
    let vhf = Density { nao: 2, data: vec![0.5, 0.0, 0.0, 0.5] };
    let (e_elec, e_coul) = default_energy_elec(&dm, &h1e, &vhf).expect("e_elec");
    assert!((e_elec.0 - (-1.5)).abs() < 1e-12, "e_elec = -1.5, got {}", e_elec.0);
    assert!((e_coul.0 - 0.5).abs() < 1e-12, "e_coul = 0.5, got {}", e_coul.0);
}

#[test]
fn energy_tot_equals_elec_plus_nuc_caller_adds() {
    // The hook returns e_elec (caller adds nuc-repulsion in the kernel loop).
    let dm = Density { nao: 1, data: vec![1.0] };
    let h1e = Density { nao: 1, data: vec![-2.0] };
    let vhf = Density { nao: 1, data: vec![1.0] };
    let e_tot: Energy = default_energy_tot(&dm, &h1e, &vhf).expect("e_tot");
    // e_1e = -2, e_coul = 0.5 * 1 = 0.5, sum = -1.5
    assert!((e_tot.0 - (-1.5)).abs() < 1e-12, "energy_tot returns e_elec, got {}", e_tot.0);
}

#[test]
fn make_rdm1_uses_oracle_sum_path() {
    // Smoke: the path is deterministic across re-runs (oracle_sum invariance).
    let mo = MOCoefficients {
        nao: 3,
        nmo: 3,
        data: vec![
            // F-order: col 0, col 1, col 2
            0.6, 0.8, 0.0,
            -0.8, 0.6, 0.0,
            0.0, 0.0, 1.0,
        ],
        energies: vec![-1.0, 0.5, 0.8],
        occupations: vec![2.0, 0.0, 0.0],
    };
    let dm_a = default_make_rdm1(&mo).expect("rdm 1");
    let dm_b = default_make_rdm1(&mo).expect("rdm 2");
    assert_eq!(dm_a.data, dm_b.data, "deterministic re-run (oracle_sum invariance)");
    // sum of diagonal == trace = sum of occupations = 2.0 (closed shell, 2 electrons)
    let trace = dm_a.data[0] + dm_a.data[4] + dm_a.data[8];
    assert!((trace - 2.0).abs() < 1e-12, "trace(D) = nelec = 2, got {}", trace);
}
