mod common;

use std::collections::HashSet;

use pyscf_pbc_mp::{PbcMpError, staggered_submesh};

#[test]
fn odd_mesh_is_refused() {
    let cell = common::diamond_anchor();
    let kpts = cell.make_kpts([1, 1, 2]).expect("k-points");
    let err = staggered_submesh(&cell, &kpts, [1, 1, 2]).expect_err("odd mesh");
    assert!(matches!(
        err,
        PbcMpError::OddStaggerMesh { mesh: [1, 1, 2] }
    ));
}

#[test]
fn submesh_maps_are_bijective_and_staggered() {
    let cell = common::diamond_anchor();
    let kpts = cell.make_kpts([2, 2, 2]).expect("k-points");
    let stagger = staggered_submesh(&cell, &kpts, [2, 2, 2]).expect("staggered mesh");
    assert_eq!(stagger.kpts_idx_occ.len(), 1);
    assert_eq!(stagger.kpts_idx_vir.len(), 1);
    assert_eq!(
        stagger
            .kpts_idx_occ
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len(),
        stagger.kpts_idx_occ.len()
    );
    assert_eq!(
        stagger
            .kpts_idx_vir
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len(),
        stagger.kpts_idx_vir.len()
    );
    assert!(stagger.kpts_idx_occ.iter().all(|&i| i < kpts.len()));
    assert!(stagger.kpts_idx_vir.iter().all(|&i| i < kpts.len()));

    let occ = cell.get_scaled_kpts(&stagger.kpts_occ);
    let vir = cell.get_scaled_kpts(&stagger.kpts_vir);
    for (ko, kv) in occ.iter().zip(&vir) {
        for axis in 0..3 {
            let d = ko[axis] - kv[axis] - 0.5;
            assert!((d - d.round()).abs() < 1e-12, "axis {axis}: {d}");
        }
    }
}

/// Upstream's own staggered-mesh reference system, measured live on PySCF
/// 2.12.1 and committed in
/// `.planning/phases/15-periodic-ao2mo-kmp2/measurements/stagger.out`.
///
/// The constants embedded in `kmp2_stagger.py:385/390/395` are the 1e-5-gated
/// values from the paper; the live 2.12.1 numbers below sit 2.8e-7…3.5e-7 from
/// them, which is why this test gates against the measured values, not the
/// source-tree ones.
#[test]
fn h2_dimer_matches_upstream_staggered_energies() {
    use pyscf_pbc_mp::Kmp2;
    use pyscf_pbc_scf::{KScfConfig, Krhf};

    let cell = common::h2_dimer_stagger();
    assert_eq!(cell.mesh, [29, 29, 29], "mesh drifted from the oracle run");
    let kpts = cell.make_kpts([2, 2, 2]).expect("kpts");
    let mf = Krhf::new(cell, &kpts).expect("krhf");
    let mut cfg = KScfConfig::for_cell(mf.cell());
    cfg.conv_tol = 1e-11;
    let result = mf.kernel(&cfg).expect("SCF");
    assert!(result.converged);
    assert!(
        (result.e_tot - -1.1004620466064836).abs() < 2e-6,
        "e_tot={}",
        result.e_tot
    );

    let mp = Kmp2::new(&result, mf.with_df.as_ref()).expect("KMP2");
    let e_std = mp.kernel().expect("KMP2 kernel").e_corr;
    println!(
        "[h2 standard KMP2] rust={e_std:.17} upstream=-0.014390203713094872 residual={:.3e}",
        (e_std - -0.014390203713094872).abs()
    );
    assert!(
        (e_std - -0.014390203713094872).abs() < 2e-6,
        "standard KMP2 e_corr={e_std}"
    );

    let mp = Kmp2::new(&result, mf.with_df.as_ref()).expect("KMP2");
    let stagger = pyscf_pbc_mp::Kmp2Stagger::new(mp, [2, 2, 2]).expect("stagger");
    assert_eq!(stagger.mesh.kpts_idx_occ, vec![7]);
    assert_eq!(stagger.mesh.kpts_idx_vir, vec![0]);
    let e_sub = stagger.kernel().expect("stagger kernel");
    println!(
        "[h2 stagger submesh] rust={e_sub:.17} upstream=-0.016089900380356827 residual={:.3e}",
        (e_sub - -0.016089900380356827).abs()
    );
    assert!(
        (e_sub - -0.016089900380356827).abs() < 2e-6,
        "submesh e_corr={e_sub}"
    );

    let full = pyscf_pbc_mp::Kmp2Stagger::new_full_mesh(&mf, &result).expect("full mesh stagger");
    let e_full = full.kernel().expect("full mesh kernel");
    println!(
        "[h2 stagger full mesh] rust={e_full:.17} upstream=-0.014028716824109303 residual={:.3e}",
        (e_full - -0.014028716824109303).abs()
    );
    assert!(
        (e_full - -0.014028716824109303).abs() < 2e-6,
        "full-mesh e_corr={e_full}"
    );
}
