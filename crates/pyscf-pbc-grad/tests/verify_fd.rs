use pyscf_algebra::oracle_sum;
use pyscf_pbc_grad::{finite_diff_cells, verify_fd};

#[test]
fn rejects_nonfinite_inputs_and_unrepresentable_steps() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let good = vec![[0.0; 3]; cell.natm];
    for tol in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(
            verify_fd(
                &cell,
                &good,
                |_| panic!("invalid input reached energy"),
                1e-4,
                tol
            )
            .is_err()
        );
    }
    assert!(verify_fd(&cell, &good, |_| Ok(f64::NAN), 1e-4, 1e-6).is_err());
    let bad = vec![[f64::NAN; 3]; cell.natm];
    assert!(
        verify_fd(
            &cell,
            &bad,
            |_| panic!("invalid input reached energy"),
            1e-4,
            1e-6
        )
        .is_err()
    );
    assert!(verify_fd(&cell, &good, |_| Ok(0.0), f64::MIN_POSITIVE, 1e-6).is_err());
}

#[test]
fn displaced_cells_refresh_the_typed_basis() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let original = cell.mol.basis_set.as_ref().unwrap();
    let pair = finite_diff_cells(&cell, &[[0.0; 3]], 0, 1, 1e-4).unwrap();
    for displaced in [&pair.plus, &pair.minus] {
        assert!(!std::sync::Arc::ptr_eq(
            original,
            displaced.mol.basis_set.as_ref().unwrap()
        ));
        assert_eq!(displaced.atom_charges(), cell.atom_charges());
        let before = cell.atom_coord(1);
        let after = displaced.atom_coord(1);
        assert_ne!(before[0], after[0]);
        assert_eq!(before[1], after[1]);
        assert_eq!(before[2], after[2]);
    }
}

#[test]
fn cell_fd_matches_an_analytic_quadratic() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let analytical: Vec<[f64; 3]> = cell
        .atom_coords()
        .iter()
        .map(|r| [2.0 * r[0], 2.0 * r[1], 2.0 * r[2]])
        .collect();
    let report = verify_fd(
        &cell,
        &analytical,
        |c| {
            Ok(oracle_sum(
                &c.atom_coords()
                    .iter()
                    .flat_map(|r| [r[0] * r[0], r[1] * r[1], r[2] * r[2]])
                    .collect::<Vec<_>>(),
            ))
        },
        1e-5,
        1e-9,
    )
    .expect("analytic finite difference");
    assert!(report.passed, "max residual = {:.3e}", report.max_abs_diff);
}

#[test]
fn strain_cells_preserve_fractional_kpoints_and_pin_mesh() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let kpts = cell
        .get_abs_kpts(&[[0.25, 0.125, 0.0]])
        .expect("non-singular reference cell");
    let pair = finite_diff_cells(&cell, &kpts, 0, 0, 1e-4).expect("strain cells");

    assert_eq!(pair.minus.mesh, cell.mesh);
    assert_eq!(pair.plus.mesh, cell.mesh);
    assert!(pair.minus.vol() < cell.vol() && cell.vol() < pair.plus.vol());
    assert_ne!(pair.kpts_plus[0], kpts[0], "Cartesian k-point must move");
    let expected = cell.get_scaled_kpts(&kpts);
    for (displaced, kpts_displaced) in [
        (&pair.minus, &pair.kpts_minus),
        (&pair.plus, &pair.kpts_plus),
    ] {
        let actual = displaced.get_scaled_kpts(kpts_displaced);
        for (got, want) in actual.iter().zip(&expected) {
            for axis in 0..3 {
                assert!((got[axis] - want[axis]).abs() < 1e-13);
            }
        }
    }
}
