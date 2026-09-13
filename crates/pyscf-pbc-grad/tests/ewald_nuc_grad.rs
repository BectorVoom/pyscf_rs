use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_grad::verify_fd;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, ewald, ewald_nuc_grad};

fn asymmetric_cell(pme: bool) -> Cell {
    let mut cell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("C .1 .2 .3; C 1.5 1.7 1.9".into()),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.1, 3.4, 3.2], [3.3, 0.2, 3.5], [3.6, 3.1, 0.15]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .unwrap();
    cell.use_particle_mesh_ewald = pme;
    cell
}

/// PySCF 2.12.1 ewald_methods.ewald_nuc_grad on asymmetric_cell, in Bohr.
/// Both branches have nonzero components, so an overall sign error is visible.
#[test]
fn asymmetric_ewald_gradients_match_upstream() {
    for (pme, expected) in [
        (
            false,
            [
                0.21604500283899777,
                0.08097077176416595,
                0.06500839334458094,
            ],
        ),
        (
            true,
            [
                0.21604487827663563,
                0.08096995737474637,
                0.06500920873711857,
            ],
        ),
    ] {
        let cell = asymmetric_cell(pme);
        let actual = ewald_nuc_grad(&cell, None, None).unwrap();
        let mut worst = 0.0_f64;
        for c in 0..3 {
            worst = worst.max((actual[0][c] - expected[c]).abs());
            worst = worst.max((actual[1][c] + expected[c]).abs());
        }
        println!("pme={pme}: gradient={actual:?}, oracle residual={worst:e}");
        assert!(worst < 1e-9);
    }
}

#[test]
fn ewald_coordinate_finite_difference_without_scf() {
    for pme in [false, true] {
        let cell = asymmetric_cell(pme);
        let analytic = ewald_nuc_grad(&cell, None, None).unwrap();
        let report = verify_fd(&cell, &analytic, |c| ewald(c, None, None), 1e-4, 1e-6).unwrap();
        println!("pme={pme}: FD residual={:e}", report.max_abs_diff);
        assert!(report.passed);
    }
}
