mod common;

use pyscf_pbc_df::Fftdf;
use pyscf_pbc_mp::Kmp2;
use pyscf_pbc_scf::{KScfConfig, Krhf};

#[test]
fn correlation_energy_is_bit_identical_across_rayon_widths() {
    let cell = common::helium_631g();
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let mut mf = Krhf::from_df(Box::new(
        Fftdf::with_mesh(cell.clone(), &kpts, [9, 9, 9]).unwrap(),
    ));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-11;
    let result = mf.kernel(&cfg).expect("SCF");
    let run = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut mp = Kmp2::new(&result, mf.with_df.as_ref()).unwrap();
            mp.with_t2 = false;
            mp.kernel().unwrap().e_corr
        })
    };
    assert_eq!(run(1).to_bits(), run(8).to_bits());
}
