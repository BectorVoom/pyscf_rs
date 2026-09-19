//! `eval_ao_kpts_upstream` for `l = 0..=4` against the production
//! `eval_ao_kpts` (oracle-free, `5e-12` — see the tolerance note on `check`).
//!
//! The production path uses different screens and accumulation order, so this
//! catches wrong formulas (Cartesian powers, `c2s` rows, component order) but
//! not bits — bits are gated by `pyscf-pbc-df`'s `get_nuc_bitexact.rs`.
//! Synthetic single-shell cells isolate each `l`; one mixed cell checks the
//! `ao0` tiling across shells.

use pyscf_core::{ParsedBasis, ShellSpec, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, eval_ao_kpts, eval_ao_kpts_upstream};

fn shell_cell(l: u8) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Parsed(ParsedBasis {
                shells: vec![ShellSpec {
                    l,
                    exponents: vec![0.8, 2.0],
                    coeffs: vec![vec![1.0, 0.5]],
                }],
            }),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([
            [0.0, 2.834589, 2.834589],
            [2.834589, 0.0, 2.834589],
            [2.834589, 2.834589, 0.0],
        ]),
        ..Default::default()
    })
    .unwrap_or_else(|e| panic!("l={l} cell builds: {e:?}"))
}

fn mixed_cell() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Parsed(ParsedBasis {
                shells: vec![
                    ShellSpec {
                        l: 0,
                        exponents: vec![1.0],
                        coeffs: vec![vec![1.0]],
                    },
                    ShellSpec {
                        l: 2,
                        exponents: vec![0.9],
                        coeffs: vec![vec![1.0]],
                    },
                    ShellSpec {
                        l: 1,
                        exponents: vec![1.2, 0.4],
                        coeffs: vec![vec![1.0, 0.3]],
                    },
                    ShellSpec {
                        l: 4,
                        exponents: vec![1.1],
                        coeffs: vec![vec![1.0]],
                    },
                    ShellSpec {
                        l: 3,
                        exponents: vec![0.7],
                        coeffs: vec![vec![1.0]],
                    },
                ],
            }),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([
            [0.0, 2.834589, 2.834589],
            [2.834589, 0.0, 2.834589],
            [2.834589, 2.834589, 0.0],
        ]),
        ..Default::default()
    })
    .expect("mixed spdfg cell builds")
}

fn probe_coords(cell: &Cell) -> Vec<[f64; 3]> {
    let a = cell.lattice_vectors();
    [[0.10, 0.20, 0.30], [0.50, 0.50, 0.50], [0.97, 0.03, 0.45]]
        .iter()
        .map(|f| {
            [
                f[0] * a[0][0] + f[1] * a[1][0] + f[2] * a[2][0],
                f[0] * a[0][1] + f[1] * a[1][1] + f[2] * a[2][1],
                f[0] * a[0][2] + f[1] * a[1][2] + f[2] * a[2][2],
            ]
        })
        .collect()
}

fn max_dev(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn check(cell: &Cell, what: &str) {
    let kpts = [[0.0, 0.0, 0.0], [0.13, -0.07, 0.21]];
    let coords = probe_coords(cell);
    let up = eval_ao_kpts_upstream(cell, &coords, &kpts)
        .unwrap_or_else(|e| panic!("{what}: upstream eval errors: {e:?}"))
        .unwrap_or_else(|| panic!("{what}: l<=4 must be ported"));
    let prod = eval_ao_kpts(cell, "GTOval_sph", &coords, &kpts)
        .unwrap_or_else(|e| panic!("{what}: production eval errors: {e:?}"));
    assert_eq!(up.nao, prod.nao, "{what}: nao");
    assert_eq!(up.ngrids, prod.ngrids, "{what}: ngrids");
    // Tolerance note: the plan suggests `1e-12`, but the PRE-EXISTING `l = 0`
    // path already deviates `2.1e-12` on the diffuse synthetic cell (different
    // screens and image accumulation order — the documented `~1e-12`
    // agreement). `5e-12` keeps 10 orders of magnitude between this noise and
    // a wrong formula (a swapped `c2s` row or wrong Cartesian power moves
    // values by ~1e-1; measured new-code deviations are all <= 1.4e-12).
    for (k, (u, p)) in up.kaos.iter().zip(prod.kaos.iter()).enumerate() {
        let dr = max_dev(&u.re, &p.re);
        let di = max_dev(&u.im, &p.im);
        assert!(
            dr <= 5e-12 && di <= 5e-12,
            "{what} k={k}: re {dr:e} / im {di:e} exceed 5e-12"
        );
    }
}

#[test]
fn upstream_order_matches_production_for_spdfg_shells() {
    for l in 0..=4 {
        check(&shell_cell(l), &format!("single shell l={l}"));
    }
    check(&mixed_cell(), "mixed spdfg cell");
}
