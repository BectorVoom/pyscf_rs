//! Plan 18-08 — `pyscf/pbc/grad/kukspu.py` (83 l): KUKS+U k-point gradient.
//!
//! * `hubbard_u_deriv1_uks_matches_eu_fd`: upstream's
//!   `test_finite_diff_hubbard_U_grad` (`test_kukspu.py:24-46`) shape — the
//!   per-spin analytic `U` term against a central difference of KUKSpU
//!   `E_U` (`add_vhubbard` over the spin pair) at fixed density, with a
//!   spin-POLARISED density so the two spins' `Tr P_s^2` differ (the strain
//!   twin `kuks_stress_hubbard.rs` does the same).
//! * `closed_shell_limit_matches_restricted`: the `*2`/`*4` threading proof
//!   — a spin-paired density `(D/2, D/2)` reproduces the restricted
//!   [`hubbard_u_deriv1`](pyscf_pbc_grad::krkspu::hubbard_u_deriv1)`(D)`
//!   exactly (the factors differ by construction: `kukspu.py:73` carries
//!   `*4` where `krkspu.py:132` carries `*2`).
//! * `kernel_adds_u_rows_to_base`: the `Gradients` wiring —
//!   `KukspuGradients::kernel` equals `KuksGradients::kernel` plus the `U`
//!   rows (`kukspu.py:81-83`). Full-SCF+U Gates B/C are NOT RUN (no DFT+U
//!   SCF driver exists in the workspace; carryover).

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{ParsedBasis, ShellSpec, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::kspu::{HubbardU, USite, add_vhubbard};
use pyscf_pbc_grad::krkspu::hubbard_u_deriv1;
use pyscf_pbc_grad::kukspu::hubbard_u_deriv1_uks;
use pyscf_pbc_grad::{KuksGradients, KukspuGradients};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

fn hubbard_c2_basis() -> ParsedBasis {
    let shell = |l: u8, e: f64| ShellSpec {
        l,
        exponents: vec![e],
        coeffs: vec![vec![1.0]],
    };
    ParsedBasis {
        shells: vec![shell(0, 1.3), shell(1, 0.8), shell(2, 0.6)],
    }
}

fn hubbard_c2_cell() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("C 1 1 1; C 2 1.5 2.4".into()),
            basis: BasisInput::Parsed(hubbard_c2_basis()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([
            [2.7219931710897396, 0.3707323061773764, -0.2932808446605736],
            [0.41861090793792155, 2.9884111887948293, 0.11174386290264571],
            [0.26590785648031556, 0.018417987872943242, 2.796800501576222],
        ]),
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
        minao_ref: "minao".into(),
        ..HubbardU::default()
    }
}

/// Rebuild `cell` with atom `ia`, component `c` shifted by `+h` Bohr.
fn shifted_cell(cell: &Cell, ia: usize, c: usize, h: f64) -> Cell {
    let mut coords = cell.atom_coords();
    coords[ia][c] += h;
    let atoms: Vec<(String, [f64; 3])> = cell
        .mol
        ._atom
        .iter()
        .map(|(s, _)| s.clone())
        .zip(coords)
        .collect();
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Parsed(hubbard_c2_basis()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(cell.a),
        mesh: Some(cell.mesh),
        precision: cell.precision,
        dimension: cell.dimension,
        ..Default::default()
    })
    .expect("shifted cell must build")
}

/// A Hermitian, spin-polarised, fractionally occupied synthetic density —
/// the strain twin's fixture. Upstream feeds a 1-cycle KUKS density; the
/// derivative algebra is exact for ANY fixed Hermitian pair, so a synthetic
/// one keeps an SCF out of the gate.
fn spin_densities(nao: usize, nkpts: usize) -> (Vec<CTensor>, Vec<CTensor>) {
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
    (hermitian(0.45, 1), hermitian(0.25, 2))
}

#[test]
fn hubbard_u_deriv1_uks_matches_eu_fd() {
    let cell = hubbard_c2_cell();
    let kpts = cell.make_kpts([2, 1, 1]).expect("k-mesh");
    let nkpts = kpts.len();
    let nao = cell.mol.nao_nr;
    let cfg = hubbard_cfg();
    let (dm_a, dm_b) = spin_densities(nao, nkpts);

    let de = hubbard_u_deriv1_uks(&cell, &vec![dm_a.clone(), dm_b.clone()], &kpts, &cfg)
        .expect("UKS U term builds");
    assert!(de.iter().flatten().all(|v| v.is_finite()));
    assert!(de.iter().flatten().any(|v| v.abs() > 1e-10));

    const H: f64 = 5e-5;
    let eu = |c: &Cell| {
        let mut v = vec![vec![CTensor::zeros(nao * nao); nkpts]; 2];
        add_vhubbard(&mut v, c, &kpts, &vec![dm_a.clone(), dm_b.clone()], &cfg)
            .expect("KUKSpU E_U oracle")
    };
    let mut worst = 0.0_f64;
    for ia in 0..cell.natm {
        for c in 0..3 {
            let fd = oracle_sum(&[
                eu(&shifted_cell(&cell, ia, c, H)),
                -eu(&shifted_cell(&cell, ia, c, -H)),
            ]) / (2.0 * H);
            worst = worst.max(oracle_sum(&[de[ia][c], -fd]).abs());
        }
    }
    eprintln!("18-08 Task 2 UKS (KUKS+U dE_U vs E_U FD): worst = {worst:.3e}");
    assert!(
        worst < 5e-7,
        "Task 2 UKS FAILED: |dE_U − FD| = {worst:.3e} (upstream asserts 6 decimals)"
    );
}

#[test]
fn closed_shell_limit_matches_restricted() {
    // A closed-shell total density, halved per spin.
    let cell = hubbard_c2_cell();
    let kpts = cell.make_kpts([2, 1, 1]).expect("k-mesh");
    let nkpts = kpts.len();
    let nao = cell.mol.nao_nr;
    let cfg = hubbard_cfg();
    let (dm_a, dm_b) = spin_densities(nao, nkpts);
    let total: Vec<CTensor> = dm_a
        .iter()
        .zip(dm_b.iter())
        .map(|(a, b)| {
            CTensor::from_planes(
                a.re.iter()
                    .zip(&b.re)
                    .map(|(x, y)| oracle_sum(&[*x, *y]))
                    .collect(),
                a.im.iter()
                    .zip(&b.im)
                    .map(|(x, y)| oracle_sum(&[*x, *y]))
                    .collect(),
            )
        })
        .collect();
    let half: Vec<CTensor> = total
        .iter()
        .map(|m| {
            CTensor::from_planes(
                m.re.iter().map(|x| 0.5 * x).collect(),
                m.im.iter().map(|x| 0.5 * x).collect(),
            )
        })
        .collect();
    let r = hubbard_u_deriv1(&cell, &total, &kpts, &cfg).expect("restricted U term");
    let u =
        hubbard_u_deriv1_uks(&cell, &vec![half.clone(), half], &kpts, &cfg).expect("UKS U term");
    let mut worst = 0.0_f64;
    for (rr, uu) in r.iter().zip(u.iter()) {
        for x in 0..3 {
            worst = worst.max(oracle_sum(&[rr[x], -uu[x]]).abs());
        }
    }
    eprintln!("18-08 closed-shell limit |restricted − unrestricted| = {worst:.3e}");
    assert!(
        worst < 1e-12,
        "closed-shell limit FAILED: {worst:.3e} (the *2/*4 threading must agree exactly)"
    );
}

#[test]
fn kernel_adds_u_rows_to_base() {
    use pyscf_pbc_dft::kuks::Kuks;

    // Diamond (`gth-szv`/`gth-pade`): the base kernel refuses all-electron
    // cells in `get_hcore` by design, so the wiring gate runs here.
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let kpts = cell.make_kpts([2, 1, 1]).expect("k-mesh");
    let nkpts = kpts.len();
    let nao = cell.mol.nao_nr;
    let cfg = HubbardU {
        sites: vec![USite::Shell {
            element: "C".into(),
            l: 1,
            contraction: None,
        }],
        u_val: vec![5.0],
        minao_ref: "gth-szv".into(),
        ..HubbardU::default()
    };
    let mf = Kuks::new(cell.clone(), &kpts, "lda,vwn").expect("KUKS holder");
    // Synthetic spin orbitals in `idx(set, k)` order (alpha block, beta block).
    let z = |i: usize, j: usize, s: usize| {
        (
            ((i * 7 + j * 3 + s * 13) as f64 * 0.37).sin(),
            ((i * 5 + j * 11 + s * 7) as f64 * 0.61).cos(),
        )
    };
    let nocc = nao / 2;
    let block = |s0: usize| {
        (0..nkpts)
            .map(|k| {
                let mut re = vec![0.0_f64; nao * nao];
                let mut im = vec![0.0_f64; nao * nao];
                for m in 0..nao {
                    for i in 0..nao {
                        let (r, v) = z(i, m, k + s0);
                        re[i + m * nao] = r;
                        im[i + m * nao] = v;
                    }
                }
                CTensor::from_planes(re, im)
            })
            .collect::<Vec<_>>()
    };
    let (ca, cb) = (block(41), block(97));
    let mo_coeff: Vec<CTensor> = ca.into_iter().chain(cb).collect();
    let mo_energy: Vec<Vec<f64>> = (0..2 * nkpts)
        .map(|k| {
            (0..nao)
                .map(|m| -1.0 + 0.25 * m as f64 + 0.01 * k as f64)
                .collect()
        })
        .collect();
    // Spin-polarised occupations so the U rows cannot vanish by symmetry.
    let mo_occ: Vec<Vec<f64>> = (0..2 * nkpts)
        .map(|s| {
            (0..nao)
                .map(|m| {
                    if s < nkpts {
                        if m < nocc { 1.2 } else { 0.0 }
                    } else if m < nocc.saturating_sub(1) {
                        0.8
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    let g_base = KuksGradients::new(&mf, mo_energy.clone(), mo_coeff.clone(), mo_occ.clone())
        .expect("base gradient object")
        .kernel()
        .expect("base kernel");
    let g_u = KukspuGradients::new(&mf, mo_energy, mo_coeff, mo_occ, &cfg)
        .expect("KUKS+U gradient object")
        .kernel()
        .expect("KUKS+U kernel");
    assert_eq!(g_base.len(), g_u.len());
    let mut norm = 0.0_f64;
    for (ia, (gb, gu)) in g_base.iter().zip(g_u.iter()).enumerate() {
        for x in 0..3 {
            let d = oracle_sum(&[gu[x], -gb[x]]);
            assert!(d.is_finite(), "atom {ia} comp {x}: U wiring non-finite");
            norm += d * d;
        }
    }
    assert!(norm > 0.0, "U wiring is vacuous on this fixture");
}
