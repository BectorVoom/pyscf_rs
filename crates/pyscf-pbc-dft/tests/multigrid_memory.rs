//! M-09: retaining all level values and bounded-memory re-collocation agree exactly.

use pyscf_pbc_dft::multigrid::MultiGridNumInt;
use pyscf_pbc_gto::test_systems::si_precision;

#[test]
fn forced_low_memory_matches_retained_levels_bit_for_bit() {
    let mut cell = si_precision(1e-10);
    cell.mesh = [11, 11, 11];
    let nao = cell.mol.nao_nr;
    let mut dm = vec![0.0; nao * nao];
    for i in 0..nao {
        dm[i * nao + i] = 0.7 + i as f64 * 0.01;
    }

    unsafe { std::env::set_var("PYSCF_MAX_MEMORY", "4000") };
    let retained = MultiGridNumInt::new()
        .nr_rks(&cell, "lda,vwn", &dm)
        .expect("retained levels");
    unsafe { std::env::set_var("PYSCF_MAX_MEMORY", "0") };
    let streamed = MultiGridNumInt::new()
        .nr_rks(&cell, "lda,vwn", &dm)
        .expect("streamed levels");
    unsafe { std::env::remove_var("PYSCF_MAX_MEMORY") };

    assert_eq!(retained.nelec.to_bits(), streamed.nelec.to_bits());
    assert_eq!(retained.exc.to_bits(), streamed.exc.to_bits());
    assert_eq!(retained.ecoul.to_bits(), streamed.ecoul.to_bits());
    for (i, (a, b)) in retained.veff.iter().zip(&streamed.veff).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "veff[{i}]");
    }
}
