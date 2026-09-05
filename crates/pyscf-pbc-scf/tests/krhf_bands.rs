//! `Krhf::get_bands` under the same two invariants `kuhf_bands.rs` applies to
//! the unrestricted driver.
//!
//! This file exists for two reasons. It is real coverage — `Krhf::get_bands`
//! shipped without any test of its own — and it is the CONTROL for the
//! unrestricted one: both files run the same cell shape, the same mesh and the
//! same convergence settings, so if an invariant fails in `kuhf_bands.rs` but
//! holds here, the fault is in the unrestricted assembly rather than in the
//! shared band machinery (`get_hcore` at band k-points, the `kpts_band` J/K
//! path, `eig_channel`).
//!
//! Closed-shell He replaces the spin-1 Li of the unrestricted file; everything
//! else — 6-Bohr cube, all-electron `sto-3g`, 3x1x1 mesh — is held fixed.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::Krhf;
use pyscf_pbc_scf::types::KScfConfig;

fn he_atom() -> Cell {
    let a = 6.0;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
        ..Default::default()
    })
    .expect("the He fixture must build")
}

/// Identical to `kuhf_bands.rs`'s `tight`, deliberately.
fn tight(cell: &Cell) -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-7),
        max_cycle: 200,
        ..KScfConfig::for_cell(cell)
    }
}

#[test]
fn bands_on_the_scf_mesh_reproduce_the_scf_eigenvalues() {
    let cell = he_atom();
    let kpts = make_kpts_default(&cell, [3, 1, 1]).expect("3x1x1 mesh");
    let cfg = tight(&cell);
    let mf = Krhf::new(cell, &kpts).expect("Krhf must build");
    let scf = mf.kernel(&cfg).expect("KRHF must converge");
    assert!(scf.converged, "the fixture must converge");

    let (e_band, _) = mf
        .get_bands(&kpts, &scf.dm)
        .expect("get_bands on the SCF mesh must succeed");

    assert_eq!(e_band.len(), kpts.len(), "one block per k-point, one channel");
    let mut worst = 0.0_f64;
    for (got, want) in e_band.iter().zip(scf.mo_energy.iter()) {
        for (g, w) in got.iter().zip(want.iter()) {
            worst = worst.max((g - w).abs());
        }
    }
    eprintln!("KRHF worst |E_band - E_scf| on the mesh = {worst:.3e}");
    assert!(
        worst < 1e-8,
        "bands on the SCF mesh must reproduce the SCF eigenvalues; worst |d| = {worst:.3e}"
    );
}

#[test]
fn off_mesh_bands_respect_time_reversal() {
    let cell = he_atom();
    let kpts = make_kpts_default(&cell, [3, 1, 1]).expect("3x1x1 mesh");
    let cfg = tight(&cell);
    let mf = Krhf::new(cell.clone(), &kpts).expect("Krhf must build");
    let scf = mf.kernel(&cfg).expect("KRHF must converge");

    let k = cell
        .get_abs_kpts(&[[0.3, 0.15, 0.0]])
        .expect("scaled -> absolute k");
    let minus_k = cell
        .get_abs_kpts(&[[-0.3, -0.15, 0.0]])
        .expect("scaled -> absolute k");

    let (e_plus, _) = mf.get_bands(&k, &scf.dm).expect("bands at +k");
    let (e_minus, _) = mf.get_bands(&minus_k, &scf.dm).expect("bands at -k");

    let mut worst = 0.0_f64;
    for (p, m) in e_plus.iter().zip(e_minus.iter()) {
        for (a, b) in p.iter().zip(m.iter()) {
            worst = worst.max((a - b).abs());
        }
    }
    eprintln!("KRHF worst |E(k) - E(-k)| off mesh = {worst:.3e}");
    assert!(
        worst < 1e-8,
        "time reversal requires E(k) == E(-k); worst |d| = {worst:.3e}"
    );
}
