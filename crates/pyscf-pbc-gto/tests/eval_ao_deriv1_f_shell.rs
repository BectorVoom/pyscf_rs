//! `GTOval_sph_deriv1` on f shells: the three gradient components must match
//! a central finite difference of the AO values, for every AO of a cell whose
//! basis has l = 3 functions (Ta `gth-dzvp-molopt-sr`), at gamma and off gamma.
//! GGA's rho gradient and V_xc matrix are built from these components; a wrong
//! l = 3 gradient leaves S, T and V_nl (cintx) intact.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, eval_ao_kpts};

fn tao_cell(basis: &str) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("Ta".into(), [0.0, 0.0, 0.0]),
                ("O".into(), [1.95, 0.0, 0.0]),
            ]),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([[3.9, 0.0, 0.0], [0.0, 3.9, 0.0], [0.0, 0.0, 3.9]]),
        pseudo: Some("gth-pbe".into()),
        ke_cutoff: Some(60.0),
        ..Default::default()
    })
    .expect("TaO cell must build")
}

/// Worst |d/dx_c ao - FD| over AOs and points, with the AO index.
fn worst_gradient_error(basis: &str) -> (f64, usize, usize, f64) {
    let cell = tao_cell(basis);
    let nao = cell.mol.nao_nr;
    let kpts = [[0.0; 3], [0.13, -0.21, 0.05]];
    // Points around both atoms (Bohr), off every symmetry plane.
    let pts: Vec<[f64; 3]> = (0..40)
        .map(|i| {
            let t = i as f64;
            [0.31 + 0.137 * t, -0.52 + 0.071 * (t * 1.7).sin(), 0.23 + 0.093 * (t * 0.9).cos()]
        })
        .collect();
    let h = 1e-4;
    let d1 = eval_ao_kpts(&cell, "GTOval_sph_deriv1", &pts, &kpts).expect("deriv1");
    let ng = pts.len();
    let (mut worst, mut wao, mut wc, mut scale) = (0.0_f64, 0, 0, 0.0_f64);
    for c in 0..3 {
        let shift = |s: f64| -> Vec<[f64; 3]> {
            pts.iter().map(|p| { let mut q = *p; q[c] += s; q }).collect()
        };
        let plus = eval_ao_kpts(&cell, "GTOval_sph", &shift(h), &kpts).expect("ao+");
        let minus = eval_ao_kpts(&cell, "GTOval_sph", &shift(-h), &kpts).expect("ao-");
        for k in 0..kpts.len() {
            let (dk, pk, mk) = (&d1.kaos[k], &plus.kaos[k], &minus.kaos[k]);
            // deriv1 layout: component-major, each component (ngrids, nao) F-order.
            let off = (1 + c) * ng * nao;
            for mu in 0..nao {
                for g in 0..ng {
                    let i = mu * ng + g;
                    let an_re = dk.re[off + i];
                    let an_im = dk.im.get(off + i).copied().unwrap_or(0.0);
                    let fd_re = (pk.re[i] - mk.re[i]) / (2.0 * h);
                    let fd_im = (pk.im.get(i).copied().unwrap_or(0.0) - mk.im.get(i).copied().unwrap_or(0.0)) / (2.0 * h);
                    let e = (an_re - fd_re).abs().max((an_im - fd_im).abs());
                    scale = scale.max(fd_re.abs());
                    if e > worst {
                        (worst, wao, wc) = (e, mu, c);
                    }
                }
            }
        }
    }
    (worst, wao, wc, scale)
}

#[test]
fn f_shell_ao_gradients_match_finite_differences() {
    for basis in ["gth-szv-molopt-sr", "gth-dzvp-molopt-sr"] {
        let (worst, mu, c, scale) = worst_gradient_error(basis);
        eprintln!("{basis}: max |dAO - FD| = {worst:.3e} at ao {mu}, component {c}; max |dAO| {scale:.3}");
        assert!(worst < 1e-6 * scale.max(1.0), "{basis}: AO gradient off by {worst:e} at ao {mu}, comp {c}");
    }
}
