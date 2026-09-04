//! M-01: the selectable KS numerical-integration seam and its SCF gate.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_dft::error::PbcDftError;
use pyscf_pbc_dft::gen_grid::{BeckeGrids, PeriodicGrids};
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::kuks::Kuks;
use pyscf_pbc_dft::numint::KsNumInt;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_scf::{KScfConfig, KScfResult};

const GAMMA: [[f64; 3]; 1] = [[0.0, 0.0, 0.0]];
const MESH: [usize; 3] = [11, 11, 11];

fn cfg() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-7),
        max_cycle: 100,
        ..KScfConfig::default()
    }
}

fn silicon() -> Cell {
    let mut cell = common::silicon();
    cell.mesh = MESH;
    cell
}

fn lithium() -> Cell {
    let mut cell = common::li_atom_spin1();
    cell.mesh = MESH;
    cell
}

fn run_krks(cell: Cell, ni: KsNumInt) -> KScfResult {
    let df = Fftdf::with_mesh(cell, &GAMMA, MESH).expect("FFTDF");
    let mut mf = Krks::from_df(Box::new(df), "lda,vwn").expect("KRKS");
    mf.ni = ni;
    mf.kernel(&cfg()).expect("KRKS kernel")
}

fn run_kuks(cell: Cell, ni: KsNumInt) -> KScfResult {
    let df = Fftdf::with_mesh(cell, &GAMMA, MESH).expect("FFTDF");
    let mut mf = Kuks::from_df(Box::new(df), "lda,vwn").expect("KUKS");
    mf.ni = ni;
    mf.kernel(&cfg()).expect("KUKS kernel")
}

#[test]
fn grid_arm_remains_the_default_and_refusals_are_typed() {
    let cell = silicon();
    let mf = Krks::new(cell.clone(), &GAMMA, "lda,vwn").expect("KRKS");
    assert!(matches!(mf.ni, KsNumInt::Grid(_)));

    let nao = cell.mol.nao_nr;
    let dm = vec![vec![CTensor::from_planes(
        vec![0.0; nao * nao],
        vec![0.0; nao * nao],
    )]];
    let grids = PeriodicGrids::uniform(&cell, Some(MESH)).expect("uniform grid");

    let non_gamma = KsNumInt::multigrid()
        .nr_rks(&cell, &grids, "lda,vwn", &dm, 1, &[[0.1, 0.0, 0.0]], None)
        .expect_err("multi-k multigrid must be refused");
    assert!(matches!(
        non_gamma,
        PbcDftError::MultiGridRequiresGamma { nkpts: 1 }
    ));

    let hybrid = KsNumInt::multigrid()
        .nr_rks(&cell, &grids, "pbe0", &dm, 1, &GAMMA, None)
        .expect_err("hybrid multigrid must be refused");
    assert!(matches!(hybrid, PbcDftError::MultiGridHybridUnsupported(_)));

    let band = KsNumInt::multigrid()
        .nr_rks(&cell, &grids, "lda,vwn", &dm, 1, &GAMMA, Some(&GAMMA))
        .expect_err("band multigrid must be refused");
    assert!(matches!(band, PbcDftError::MultiGridBandUnsupported));

    let becke = PeriodicGrids::Becke(BeckeGrids::new());
    let nonuniform = KsNumInt::multigrid()
        .nr_rks(&cell, &becke, "lda,vwn", &dm, 1, &GAMMA, None)
        .expect_err("non-uniform multigrid must be refused");
    assert!(matches!(
        nonuniform,
        PbcDftError::MultiGridRequiresUniformGrid
    ));
}

#[test]
fn krks_multigrid_arms_converge_at_their_quadrature_floors() {
    let reference = run_krks(silicon(), KsNumInt::grid(&GAMMA));
    assert!(reference.converged, "grid KRKS did not converge");

    for (name, ni, tol) in [
        ("v1", KsNumInt::multigrid(), 1e-4),
        ("v2", KsNumInt::multigrid2(), 2e-3),
    ] {
        let got = run_krks(silicon(), ni);
        assert!(got.converged, "{name} KRKS did not converge");
        let de = (got.e_tot - reference.e_tot).abs();
        println!("KRKS {name}: |E_mg-E_grid| = {de:.3e}");
        assert!(de < tol, "KRKS {name} energy floor {de:.3e} >= {tol:.3e}");
    }
}

#[test]
fn kuks_multigrid_arms_converge_on_a_genuine_open_shell() {
    let reference = run_kuks(lithium(), KsNumInt::grid(&GAMMA));
    assert!(reference.converged, "grid KUKS did not converge");

    for (name, ni, tol) in [
        ("v1", KsNumInt::multigrid(), 1e-4),
        ("v2", KsNumInt::multigrid2(), 2e-3),
    ] {
        let got = run_kuks(lithium(), ni);
        assert!(got.converged, "{name} KUKS did not converge");
        let de = (got.e_tot - reference.e_tot).abs();
        let spin_delta = got.dm[0][0]
            .re
            .iter()
            .zip(&got.dm[1][0].re)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("KUKS {name}: |E_mg-E_grid| = {de:.3e}, max |dm_a-dm_b| = {spin_delta:.3e}");
        assert!(de < tol, "KUKS {name} energy floor {de:.3e} >= {tol:.3e}");
        assert!(
            spin_delta > 1.0,
            "KUKS {name} fixture is not genuinely open-shell: {spin_delta:.3e}"
        );
    }
}
