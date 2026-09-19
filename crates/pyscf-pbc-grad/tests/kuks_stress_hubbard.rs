//! `test_hubbard_U` for KUKS (`pyscf/pbc/grad/test/test_kuks_stress.py:194-219`):
//! the analytic `_hubbard_U_deriv1` (`kuks_stress.py:263-307`, ported as
//! [`hubbard_u_deriv1_uks`]) against a central difference of the KUKSpU
//! `E_U` (`kukspu.py:96`) over strained cells at fixed fractional k-points.
//!
//! 20-13-FIX: `hubbard_u_deriv1_uks` was already the unrestricted derivative
//! (`* 2` / `* 4`, `kuks_stress.py:305-307`), but the `E_U` it differentiates
//! came from `kspu::add_vhubbard`, which applied the RESTRICTED expression to
//! each spin channel (20-13 D3), and no test tied the two together — its doc
//! named a `tests/kuks_stress.rs` that did not exist. This file is that gate.
//! The same density-independent fixture as `krks_stress.rs::
//! hubbard_u_deriv1_matches_eu_fd_at_1e8`, with a spin-POLARISED density so
//! the two spins' `Tr P_s^2` differ.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{ParsedBasis, ShellSpec, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::kspu::{HubbardU, USite, add_vhubbard};
use pyscf_pbc_grad::stress::{finite_diff_cells, hubbard_u_deriv1_uks};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// `np.random.seed(5); eye(3)*3 + rand(3,3) - .5` (`krks_stress.rs::SEED5_A3`).
const SEED5_A3: [[f64; 3]; 3] = [
    [2.7219931710897396, 0.3707323061773764, -0.2932808446605736],
    [0.41861090793792155, 2.9884111887948293, 0.11174386290264571],
    [0.26590785648031556, 0.018417987872943242, 2.796800501576222],
];

/// `krks_stress.rs::hubbard_c2_cell` — 18 AOs over 10 MINAO local orbitals,
/// so the Löwdin metric is full-rank.
fn hubbard_c2_cell() -> Cell {
    let shell = |l: u8, e: f64| ShellSpec {
        l,
        exponents: vec![e],
        coeffs: vec![vec![1.0]],
    };
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("C 1 1 1; C 2 1.5 2.4".into()),
            basis: BasisInput::Parsed(ParsedBasis {
                shells: vec![shell(0, 1.3), shell(1, 0.8), shell(2, 0.6)],
            }),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(SEED5_A3),
        mesh: None,
        precision: 1e-10,
        pseudo: None,
        ..Default::default()
    })
    .expect("Hubbard C2 cell builds")
}

fn hubbard_cfg() -> HubbardU {
    HubbardU {
        sites: vec![USite::Shell {
            element: "C".into(),
            l: 1,
            contraction: Some(0),
        }],
        u_val: vec![5.0],
        ..HubbardU::default()
    }
}

#[test]
fn hubbard_u_deriv1_uks_matches_kukspu_eu_fd_at_1e8() {
    let cell = hubbard_c2_cell();
    let kpts = cell.make_kpts([2, 1, 1]).expect("k-mesh");
    let nkpts = kpts.len();
    let nao = cell.mol.nao_nr;
    let cfg = hubbard_cfg();
    // A Hermitian, spin-POLARISED, fractionally occupied density. Upstream
    // feeds a 1-cycle KUKS density (`:210`); the derivative algebra is exact
    // for ANY fixed Hermitian dm, so a synthetic one keeps an SCF (the
    // KRKS twin runs one) out of the gate.
    let hermitian = |occ: f64, seed: usize| -> Vec<CTensor> {
        (0..nkpts)
            .map(|k| {
                let mut m = CTensor::zeros(nao * nao);
                for i in 0..nao {
                    for j in 0..i {
                        let re = (((k + seed) * 31 + i * 7 + j * 13) as f64).sin() * 0.04;
                        let im = (((k + seed + 1) * 17 + i * 11 + j * 5) as f64).cos() * 0.02;
                        m.re[i * nao + j] = re;
                        m.re[j * nao + i] = re;
                        m.im[i * nao + j] = im;
                        m.im[j * nao + i] = -im;
                    }
                    m.re[i * nao + i] = occ + 0.05 * ((k + 3 * i + seed) as f64).cos();
                }
                m
            })
            .collect()
    };
    let (dm_a, dm_b) = (hermitian(0.45, 1), hermitian(0.25, 2));
    let dm_spin: Vec<CTensor> = dm_a.iter().chain(dm_b.iter()).cloned().collect();

    let sigma = hubbard_u_deriv1_uks(&cell, &dm_spin, &kpts, &cfg).expect("hubbard deriv1 uks");
    // The FULL strain separation. Upstream's `_finite_diff_cells(disp=1e-4)`
    // is a HALF step (2e-4 apart) on ITS cell; on this compact seed-5 fixture
    // the central difference's O(h^2) truncation dominates at that step.
    // Measured with a 0.6/0.4 split of the converged KRKS density, |analytic -
    // FD| on (2,2): 3.28e-7 / 8.19e-8 / 2.04e-8 / 5.04e-9 / 1.22e-9 at full
    // separations 8e-4 / 4e-4 / 2e-4 / 1e-4 / 5e-5 — a clean factor 4 per
    // halving, i.e. truncation, not a derivative defect (the KRKS twin,
    // `krks_stress.rs`, reads 1.84e-8 at 2e-4 and 4.55e-9 at 1e-4 on the same
    // cell). The step is therefore 1e-4; the 1e-8 gate is unchanged.
    const U_FULL: f64 = 1e-4;
    for (x, y) in [(1usize, 0usize), (2, 2)] {
        let pair = finite_diff_cells(&cell, &kpts, x, y, U_FULL).expect("strain pair");
        let eu = |c: &Cell, k: &[[f64; 3]]| {
            let mut v = vec![vec![CTensor::zeros(nao * nao); nkpts]; 2];
            add_vhubbard(&mut v, c, k, &vec![dm_a.clone(), dm_b.clone()], &cfg)
                .expect("KUKSpU E_U oracle")
        };
        let e1 = eu(&pair.plus, &pair.kpts_plus);
        let e2 = eu(&pair.minus, &pair.kpts_minus);
        let fd = oracle_sum(&[e1, -e2]) / U_FULL;
        let d = (sigma[x][y] - fd).abs();
        eprintln!(
            "hubbard-U UKS ({x},{y}): analytic {:.12e}  E_U FD {fd:.12e}  |d| = {d:.3e}",
            sigma[x][y]
        );
        assert!(
            d < 1e-8,
            "hubbard-U UKS ({x},{y}): analytic vs KUKSpU E_U FD = {d:.3e} (Gate = 1e-8)"
        );
    }
}
