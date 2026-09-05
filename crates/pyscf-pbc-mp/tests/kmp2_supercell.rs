mod common;

use pyscf_pbc_df::Fftdf;
use pyscf_pbc_gto::super_cell;
use pyscf_pbc_mp::Kmp2;
use pyscf_pbc_scf::{KScfConfig, Krhf};

fn correlation(cell: pyscf_pbc_gto::Cell, kmesh: [usize; 3], mesh: [usize; 3]) -> f64 {
    let kpts = cell.make_kpts(kmesh).expect("k-points");
    let df = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    let mut mf = Krhf::from_df(Box::new(df));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-11;
    let result = mf.kernel(&cfg).expect("SCF");
    assert!(result.converged);
    let mut mp = Kmp2::new(&result, mf.with_df.as_ref()).expect("KMP2");
    mp.with_t2 = false;
    mp.kernel().expect("KMP2 kernel").e_corr
}

#[test]
fn primitive_kmesh_matches_gamma_supercell_per_cell() {
    let cell = common::helium_631g();
    let primitive = correlation(cell.clone(), [1, 1, 2], [9, 9, 9]);
    let supercell = super_cell(&cell, [1, 1, 2], false).expect("supercell");
    let gamma = correlation(supercell, [1, 1, 1], [9, 9, 18]);
    assert!(
        (primitive - gamma / 2.0).abs() < 2e-8,
        "primitive={primitive}, supercell/2={}",
        gamma / 2.0
    );
}
