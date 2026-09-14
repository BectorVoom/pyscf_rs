//! Plan 18-04 Task 4 (+ Task 5's refusal half) — the named refusal for every
//! non-FFTDF route, and the gradient route's refusal of the k-pair flag.
//!
//! `get_jk_e1`/`get_j_e1`/`get_k_e1` exist only on FFTDF
//! (`pyscf/pbc/df/fft.py:324-340`); upstream raises `AttributeError` anywhere
//! else. This port serves a loud error that **names the route and says why**,
//! and the test asserts it is an error, not a number — a fallback here would
//! produce a gradient that is plausible, wrong, and consistent run to run.
//!
//! Always on (no oracle needed).

mod common;

use common::diamond;
use pyscf_algebra::CTensor;
use pyscf_pbc_df::{Aftdf, Fftdf, Gdf, JkOpts, Mdf, PeriodicDf, Rsdf};
use pyscf_pbc_gto::make_kpts_default;

fn gamma_dm(nao: usize) -> Vec<Vec<CTensor>> {
    let mut dm = CTensor::zeros(nao * nao);
    for i in 0..nao {
        dm.re[i * nao + i] = 0.5;
    }
    vec![vec![dm; 1]]
}

/// GDF, MDF, RSDF and AFTDF each refuse `get_jk_e1`/`get_j_e1`/`get_k_e1`
/// with an error naming the route.
#[test]
fn non_fftdf_routes_refuse_gradient_jk_by_name() {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let kpts = vec![[0.0_f64; 3]];
    let dms = gamma_dm(nao);

    let gdf = Gdf::new(cell.clone(), &kpts);
    let mdf = Mdf::new(cell.clone(), &kpts);
    let rsdf = Rsdf::new(cell.clone(), &kpts);
    let aftdf = Aftdf::new(cell.clone(), &kpts).expect("AFTDF");

    for (route, df) in [
        ("GDF", &gdf as &dyn PeriodicDf),
        ("MDF", &mdf as &dyn PeriodicDf),
        ("RSGDF", &rsdf as &dyn PeriodicDf),
        ("AFTDF", &aftdf as &dyn PeriodicDf),
    ] {
        assert_eq!(df.name(), route);
        let opts = JkOpts {
            hermi: 1,
            kpts_band: None,
            with_j: true,
            with_k: true,
            exxdiv: None,
            omega: None,
            kk_symmetry: false,
        };
        for (what, err) in [
            (
                "get_jk_e1",
                df.get_jk_e1(&dms, &kpts, opts, None).map(|_| ()),
            ),
            ("get_j_e1", df.get_j_e1(&dms, &kpts, None).map(|_| ())),
            (
                "get_k_e1",
                df.get_k_e1(&dms, &kpts, None, None, None, None).map(|_| ()),
            ),
        ] {
            let msg = err.expect_err(&format!("{route}::{what} must refuse, not compute"));
            let text = msg.to_string();
            assert!(
                text.contains(route),
                "{route}::{what} refusal does not name the route: {text}"
            );
            assert!(
                text.contains("FFTDF"),
                "{route}::{what} refusal does not point at FFTDF: {text}"
            );
        }
    }
}

/// The gradient route refuses the k-pair flag (clause 4b) — and runs fine
/// without it.
#[test]
fn gradient_route_refuses_kk_symmetry() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [1, 1, 2]).expect("kpts");
    let nao = cell.mol.nao_nr;
    let mut dm = CTensor::zeros(nao * nao);
    for i in 0..nao {
        dm.re[i * nao + i] = 0.5;
    }
    let dms = vec![vec![dm; kpts.len()]];
    let df = Fftdf::with_mesh(cell, &kpts, [11, 11, 11]).expect("FFTDF");

    let refused = df
        .get_jk_e1(
            &dms,
            &kpts,
            JkOpts {
                kk_symmetry: true,
                ..JkOpts::hermitian()
            },
            None,
        )
        .map(|_| ());
    let msg = refused.expect_err("gradient route must refuse kk_symmetry");
    assert!(
        msg.to_string().contains("kk_symmetry"),
        "refusal must name the flag: {msg}"
    );

    // Without the flag the same entry point computes both halves.
    let r = df
        .get_jk_e1(&dms, &kpts, JkOpts::hermitian(), None)
        .expect("gradient JK");
    let (vj, vk) = (r.vj.expect("vj"), r.vk.expect("vk"));
    assert_eq!((vj.len(), vk.len()), (3, 3));
    assert_eq!((vj[0].len(), vk[0].len()), (1, 1));
    assert_eq!((vj[0][0].len(), vk[0][0].len()), (kpts.len(), kpts.len()));
}
