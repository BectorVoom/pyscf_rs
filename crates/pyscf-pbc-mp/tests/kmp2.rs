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

    // **`Tr(gamma_k)` is NOT `nelec` at each k-point, and upstream does not
    // satisfy that.** The occupied-block correction `-2 t2 t2` and the
    // virtual-block one `+2 t2 t2` have equal and opposite traces only after
    // the k-sum; at a single k-point the MP2 correction moves charge between
    // k-points. Measured on THIS anchor
    // (`measurements/kmp2_gdf_and_rdm1.out` §2, PySCF 2.12.1):
    //
    // | k | upstream `Tr(gamma_k)` |
    // |---|---|
    // | 0 | 8.028298787714228 |
    // | 1 | 7.971701212285773 |
    // | mean | 8.0 exactly |
    //
    // An earlier version of this test asserted `Tr(gamma_k) == 8` per k-point
    // to 2e-10, which upstream misses by 2.8e-2 on the very first k-point.
    const UPSTREAM_TRACES: [f64; 2] = [8.028_298_787_714_228, 7.971_701_212_285_773];
    assert_eq!(dm1.len(), UPSTREAM_TRACES.len());
    let mut traces = Vec::with_capacity(dm1.len());
    for (k, d) in dm1.iter().enumerate() {
        let nmo = (d.re.len() as f64).sqrt() as usize;
        let trace: f64 = (0..nmo).map(|p| d.re[p * nmo + p]).sum();
        let dev = (trace - UPSTREAM_TRACES[k]).abs();
        eprintln!(
            "RDM1 Tr(gamma_{k}) = {trace:.15}, upstream {:.15}, |d| = {dev:e}",
            UPSTREAM_TRACES[k]
        );
        assert!(
            dev < 1e-7,
            "RDM1 Tr(gamma_{k}) = {trace}, upstream {}, |d| = {dev:e}",
            UPSTREAM_TRACES[k]
        );
        traces.push(trace);
        for p in 0..nmo {
            for q in 0..nmo {
                let pq = p * nmo + q;
                let qp = q * nmo + p;
                assert!((d.re[pq] - d.re[qp]).abs() < 2e-12);
                assert!((d.im[pq] + d.im[qp]).abs() < 2e-12);
            }
        }
    }
    // The identity that DOES hold, and the one worth gating: the k-AVERAGE is
    // the electron count. Upstream lands on it to 0.0e0.
    let mean = traces.iter().sum::<f64>() / traces.len() as f64;
    assert!(
        (mean - 8.0).abs() < 2e-10,
        "the k-averaged RDM1 trace must be nelec: {mean}"
    );
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
            -0.016_989_369_077_568_164,
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
            // **An ORACLE gate since 2026-09-05, where it used to be a pin
            // against this port's own output.** It was written that way for a
            // reason the comment stated: the GDF mean field under it was
            // 1.461e-1 Ha from upstream's, so there was nothing to gate a
            // correlation energy against. `14-VERIFICATION.md §11` fixed the
            // two defects behind that (an `s2` k-pair packing error in
            // `sr_loop` and a nuclear-attraction mesh), the mean field now
            // lands at 3.017e-9, and `measurements/kmp2_gdf_and_rdm1.out` §1
            // records what upstream actually produces here. The old constant,
            // -0.015572369890603862, was 1.417e-3 from it.
            //
            // Upstream's own two routes agree to 4e-18 on this cell, so the
            // route-agreement assertion below is upstream's identity, not a
            // tolerance.
            let dev = (direct - expected).abs();
            eprintln!("KMP2/GDF He/6-31g: AO2MO={direct:.17}, upstream={expected:.17}, |d|={dev:e}");
            assert!(
                dev < 2e-6,
                "KMP2 on GDF: AO2MO={direct}, upstream={expected}, |d|={dev:e}"
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
