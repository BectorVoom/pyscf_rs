//! The periodic AO evaluator on f shells: `Σ_g conj(ao_i) ao_j w` on a fine
//! uniform grid must reproduce the analytic lattice-summed overlap
//! (`pbc_intor("int1e_ovlp")`) for a cell whose basis has l = 3 functions
//! (Ta `gth-dzvp-molopt-sr`: s, p, d, f; O: s, p, d), at gamma and off gamma.
//! A wrong l = 3 angular part leaves the cintx integrals (S, T, V_nl) intact
//! while corrupting every grid quantity (rho, J, V_xc, V_loc).

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, PbcIntorOpts, UniformGrids, eval_ao_kpts, pbc_intor};

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
        ke_cutoff: Some(400.0),
        ..Default::default()
    })
    .expect("TaO cell must build")
}

fn grid_overlap_error(basis: &str) -> (f64, usize, usize, f64) {
    let cell = tao_cell(basis);
    let nao = cell.mol.nao_nr;
    let kpts = [[0.0; 3], [0.13, -0.21, 0.05]];
    let grids = UniformGrids::build(&cell, None).expect("grid");
    let w = grids.weight();
    let ng = grids.coords.len();
    let ao = eval_ao_kpts(&cell, "GTOval_sph", &grids.coords, &kpts).expect("ao");
    let s = pbc_intor(&cell, "int1e_ovlp", &kpts, PbcIntorOpts::default()).expect("ovlp");
    let (mut worst, mut wi, mut wj, mut scale) = (0.0_f64, 0, 0, 0.0_f64);
    for k in 0..kpts.len() {
        let a = &ao.kaos[k];
        let sk = &s.kmats[k];
        for i in 0..nao {
            for j in 0..nao {
                let (mut re, mut im) = (0.0, 0.0);
                for g in 0..ng {
                    let (ar, ai) = (a.re[i * ng + g], a.im.get(i * ng + g).copied().unwrap_or(0.0));
                    let (br, bi) = (a.re[j * ng + g], a.im.get(j * ng + g).copied().unwrap_or(0.0));
                    re += (ar * br + ai * bi) * w;
                    im += (ar * bi - ai * br) * w;
                }
                let (er, ei) = (sk.re[i + j * nao], sk.im.get(i + j * nao).copied().unwrap_or(0.0));
                let d = (re - er).abs().max((im - ei).abs());
                scale = scale.max(er.abs());
                if d > worst {
                    (worst, wi, wj) = (d, i, j);
                }
            }
        }
    }
    (worst, wi, wj, scale)
}

#[test]
fn f_shell_grid_overlap_matches_analytic() {
    for basis in ["gth-szv-molopt-sr", "gth-dzvp-molopt-sr"] {
        let (worst, i, j, scale) = grid_overlap_error(basis);
        eprintln!("{basis}: max |S_grid - S| = {worst:.3e} at ({i}, {j}), max |S| {scale:.3}");
        assert!(worst < 1e-5 * scale, "{basis}: grid AO overlap off by {worst:e} at ({i}, {j})");
    }
}
