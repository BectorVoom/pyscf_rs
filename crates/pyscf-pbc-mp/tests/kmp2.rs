mod common;

use pyscf_pbc_df::{Fftdf, Gdf};
use pyscf_pbc_mp::{Kmp2, RdmKind};
use pyscf_pbc_scf::{KScfConfig, Krhf};

#[test]
fn diamond_anchor_and_without_t2() {
    let cell = common::diamond_anchor();
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let mut mf = Krhf::new(cell, &kpts).expect("krhf");
    // The committed upstream anchor explicitly constructs KRHF(exxdiv=None).
    mf.exxdiv = None;
    let result = mf.run().expect("SCF");
    assert!(result.converged, "SCF reference did not converge");
    let mp = Kmp2::new(&result, mf.with_df.as_ref()).expect("KMP2");
    let with = mp.kernel().expect("KMP2 kernel");
    assert!(
        (with.e_corr - -0.204721432828996).abs() < 2e-6,
        "e_corr={}",
        with.e_corr
    );
    assert_eq!(with.e_corr_ss + with.e_corr_os, with.e_corr);
    let t2 = with.t2.as_ref().expect("T2 requested");
    let dm1 = mp.make_rdm1(t2, RdmKind::Padded).expect("RDM1");
    for d in &dm1 {
        let nmo = (d.re.len() as f64).sqrt() as usize;
        let trace: f64 = (0..nmo).map(|p| d.re[p * nmo + p]).sum();
        assert!((trace - 8.0).abs() < 2e-10, "RDM1 trace={trace}");
        for p in 0..nmo {
            for q in 0..nmo {
                let pq = p * nmo + q;
                let qp = q * nmo + p;
                assert!((d.re[pq] - d.re[qp]).abs() < 2e-12);
                assert!((d.im[pq] + d.im[qp]).abs() < 2e-12);
            }
        }
    }
    let mut no_t2 = Kmp2::new(&result, mf.with_df.as_ref()).unwrap();
    no_t2.with_t2 = false;
    let without = no_t2.kernel().unwrap();
    assert_eq!(with.e_corr, without.e_corr);
    assert!(without.t2.is_none());
}

#[test]
fn fft_matches_upstream_and_gdf_integral_routes_agree() {
    let cell = common::helium_631g();
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-11;
    for (df, expected, is_gdf) in [
        (
            Box::new(Fftdf::with_mesh(cell.clone(), &kpts, [9, 9, 9]).unwrap())
                as Box<dyn pyscf_pbc_df::PeriodicDf>,
            -0.033241446759957924,
            false,
        ),
        (
            Box::new(Gdf::new(cell.clone(), &kpts)) as Box<dyn pyscf_pbc_df::PeriodicDf>,
            -0.015572369890603862,
            true,
        ),
    ] {
        let mut mf = Krhf::from_df(df);
        mf.exxdiv = None;
        let result = mf.kernel(&cfg).expect("SCF");
        let mut mp = Kmp2::new(&result, mf.with_df.as_ref()).expect("KMP2");
        mp.with_t2 = false;
        let got = mp.kernel().expect("KMP2").e_corr;
        if is_gdf {
            mp.with_df_ints = false;
            let direct = mp.kernel().expect("GDF AO2MO route").e_corr;
            // A REGRESSION PIN, not an accuracy gate. This port's GDF mean
            // field on this cell is 1.461e-1 Ha from upstream's
            // (`15-VERIFICATION.md §1a`), so `expected` is what THIS port
            // currently produces, pinned so a change is visible. The gate
            // against upstream is the FFTDF branch below; the GDF row is
            // NOT MET and belongs to Phase 14.
            assert!(
                (direct - expected).abs() < 2e-12,
                "GDF regression: AO2MO={direct}, expected={expected}"
            );
            assert!((got - direct).abs() < 2e-15, "Lov={got}, AO2MO={direct}");
        } else {
            assert!(
                (got - expected).abs() < 2e-6,
                "got {got}, expected {expected}"
            );
        }
    }
}
