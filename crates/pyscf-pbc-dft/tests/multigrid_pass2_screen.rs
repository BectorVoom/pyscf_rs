//! M-04 step 3: opt-in periodic-radius screening stays below the v1 gate floor.

use pyscf_pbc_dft::multigrid::MultiGridNumInt;
use pyscf_pbc_gto::test_systems::si_precision;

fn max_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn radius_screen_is_opt_in_and_below_the_v1_floor() {
    let mut cell = si_precision(1e-10);
    cell.mesh = [11, 11, 11];
    let nao = cell.mol.nao_nr;
    let mut dm = vec![0.0; nao * nao];
    for i in 0..nao {
        dm[i * nao + i] = 0.5 + i as f64 * 0.03;
        for j in 0..i {
            let x = ((i * 17 + j * 11) as f64).sin() * 0.02;
            dm[i * nao + j] = x;
            dm[j * nao + i] = x;
        }
    }

    unsafe { std::env::remove_var("PYSCF_PBC_MULTIGRID_PASS2_SCREEN") };
    let plain = MultiGridNumInt::new()
        .nr_rks(&cell, "lda,vwn", &dm)
        .expect("unscreened");
    unsafe { std::env::set_var("PYSCF_PBC_MULTIGRID_PASS2_SCREEN", "on") };
    let screened = MultiGridNumInt::new()
        .nr_rks(&cell, "lda,vwn", &dm)
        .expect("screened");
    unsafe { std::env::remove_var("PYSCF_PBC_MULTIGRID_PASS2_SCREEN") };

    let residual = max_delta(&plain.veff, &screened.veff);
    println!("M-04 pass2 screen max |dV| = {residual:.3e}");
    assert!(residual < 2.1e-11, "screen residual {residual:.3e}");
}
