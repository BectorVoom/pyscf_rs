//! Plan 18-08 — `pyscf/pbc/grad/krkspu.py` (142 l): KRKS+U k-point gradient.
//!
//! * Task 1's component test (`local_orbitals_match_fd_at_6dp`): upstream's
//!   own `test_finite_diff_local_orbitals` (`test_krkspu.py:24-43`) —
//!   `make_coeff(1)` against a central difference of `_make_minao_lo` at
//!   6 decimals. No SCF is involved.
//! * Task 2 (`hubbard_u_deriv1_matches_eu_fd`): upstream's
//!   `test_finite_diff_hubbard_U_grad` (`:45-67`) shape — the analytic `U`
//!   term against a central difference of `add_vhubbard`'s `E_U` at fixed
//!   density. Upstream feeds a 1-cycle KRKS density; the derivative algebra
//!   is exact for ANY fixed Hermitian density, so the gate uses the
//!   converged KRKS density (the strain twin
//!   `krks_stress.rs::hubbard_u_deriv1_matches_eu_fd_at_1e8` does the same).
//! * Task 3 (`kernel_adds_u_rows_to_base`): the `Gradients` wiring —
//!   `KrkspuGradients::kernel` equals `KrksGradients::kernel` plus the `U`
//!   rows, which is where upstream's `extra_force` sits (`krkspu.py:140-142`).
//!   Full-SCF+U Gates B/C are NOT RUN here: `KrkspU` is a `veff` wrapper,
//!   and no DFT+U SCF driver exists in the workspace yet (carryover).
//!
//! # Geometry is specified in BOHR
//!
//! Upstream's fixture is in Ångström. `Unit::Ang` is CODATA-2014 while
//! upstream is CODATA-2010 (differing in the 8th digit of every lattice
//! vector), so the test converts with upstream's own `BOHR_ANG` explicitly
//! instead of relying on the unit flag.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{ParsedBasis, ShellSpec, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::kspu::{HubbardU, USite, add_vhubbard};
use pyscf_pbc_grad::krkspu::{first_order_local_orbitals, hubbard_u_deriv1, make_coeff};
use pyscf_pbc_grad::{KrksGradients, KrkspuGradients};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// Upstream's Å→Bohr factor (`pyscf.data.nist.BOHR`), explicit so the
/// fixture does not inherit the CODATA-2014/2010 unit-flag drift.
const BOHR_ANG: f64 = 0.52917721067;

/// Upstream `test_krkspu.py:setUpModule`'s uncontracted basis,
/// `[[0,[1.3,1]],[1,[0.8,1]]]`, on both atoms.
fn co_basis() -> ParsedBasis {
    ParsedBasis {
        shells: vec![
            ShellSpec {
                l: 0,
                exponents: vec![1.3],
                coeffs: vec![vec![1.0]],
            },
            ShellSpec {
                l: 1,
                exponents: vec![0.8],
                coeffs: vec![vec![1.0]],
            },
        ],
    }
}

/// Upstream `test_krkspu.py`'s C/O cell, in Bohr: atoms
/// `C 0 0 0; O 0.5 0.8 1.1` (Å), lattice rows `1.7834`-based (Å),
/// `pseudo = 'gth-pbe'`. `o_y_ang` overrides the oxygen y in Å for the
/// finite-difference cells.
fn upstream_co_cell(o_y_ang: f64) -> Cell {
    let a = 1.7834 / BOHR_ANG;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                (
                    "O".into(),
                    [0.5 / BOHR_ANG, o_y_ang / BOHR_ANG, 1.1 / BOHR_ANG],
                ),
            ]),
            basis: BasisInput::Parsed(co_basis()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, a, a], [a, 0.0, a], [a, a, 0.0]]),
        pseudo: Some("gth-pbe".into()),
        ..Default::default()
    })
    .expect("upstream test_krkspu C/O cell must build")
}

fn kpts_311(cell: &Cell) -> Vec<[f64; 3]> {
    let kpts = cell.make_kpts([3, 1, 1]).expect("3x1x1 k-mesh");
    assert_eq!(kpts.len(), 3, "upstream test_krkspu uses 3 k-points");
    kpts
}

// ---------------------------------------------------------------------------
// Task 1: first-order local orbitals vs finite difference, 6 decimals.
// ---------------------------------------------------------------------------

#[test]
fn local_orbitals_match_fd_at_6dp() {
    use pyscf_pbc_dft::kspu::make_minao_lo;
    use pyscf_pbc_dft::kspu::reference_cell;

    let cell = upstream_co_cell(0.8);
    let kpts = kpts_311(&cell);
    let minao = "gth-szv";
    let flo = first_order_local_orbitals(&cell, minao, &kpts).expect("C0 builds");
    assert_eq!(flo.nkpts(), 3);
    assert!(
        flo.c0
            .iter()
            .flat_map(|c| c.re.iter().chain(&c.im))
            .all(|v| v.is_finite()),
        "C0 must be finite"
    );
    let pcell = reference_cell(&cell, minao).expect("reference cell");
    let tables = pyscf_pbc_grad::krkspu::ip_tables(&cell, &pcell, &kpts).expect("ip tables");
    let slices = pyscf_gto::aoslice_by_atom(&cell.mol).expect("aoslices");
    let mslices = pyscf_gto::aoslice_by_atom(&pcell.mol).expect("MINAO aoslices");
    let (_, _, p0, p1) = slices[1];
    let (_, _, q0, q1) = mslices[1];
    let nao = cell.mol.nao_nr;
    let nlo = flo.nlo;

    // Upstream displaces O by ±0.001 Å in y; the reference is
    // `(C0p − C0m)/2e-3*BOHR` (`test_krkspu.py:41-42`). In Bohr the half
    // step is `0.001/BOHR_ANG` and the divisor is twice that.
    let h = 0.001 / BOHR_ANG;
    let mut worst = 0.0_f64;
    for k in 0..3 {
        let c1 = make_coeff(&flo, k, p0, p1, q0, q1, &tables).expect("C1 builds")[1].clone();
        let cp = upstream_co_cell(0.801);
        let cm = upstream_co_cell(0.799);
        let kp = kpts_311(&cp);
        let km = kpts_311(&cm);
        let c0p = make_minao_lo(&cp, &reference_cell(&cp, minao).expect("pcell+"), &kp)
            .expect("C0+ builds")[k]
            .clone();
        let c0m = make_minao_lo(&cm, &reference_cell(&cm, minao).expect("pcell-"), &km)
            .expect("C0- builds")[k]
            .clone();
        assert_eq!(c0p.re.len(), nao * nlo);
        for i in 0..nao {
            for j in 0..nlo {
                // `make_minao_lo` is COLUMN-MAJOR (`C[i,j]` at `i+j*nao`);
                // `make_coeff` is row-major.
                let fd = oracle_sum(&[c0p.re[i + j * nao], -c0m.re[i + j * nao]]) / (2.0 * h);
                let fdi = oracle_sum(&[c0p.im[i + j * nao], -c0m.im[i + j * nao]]) / (2.0 * h);
                let d = (oracle_sum(&[c1.re[i * nlo + j], -fd]).abs())
                    .max(oracle_sum(&[c1.im[i * nlo + j], -fdi]).abs());
                worst = worst.max(d);
            }
        }
    }
    eprintln!("18-08 Task 1 (KRKS+U C1 vs FD): worst = {worst:.3e}");
    assert!(
        worst < 5e-7,
        "Task 1 FAILED: |C1 − FD| = {worst:.3e} (upstream asserts 6 decimals)"
    );
}

// ---------------------------------------------------------------------------
// Task 2: the U term vs E_U finite difference.
// ---------------------------------------------------------------------------

/// All-electron C2 cell with a provably full-rank Löwdin metric (the strain
/// twin's `hubbard_c2_cell`: 18 AOs over 10 MINAO local orbitals), so the
/// gate measures the derivative algebra, not a metric rank.
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
/// A fresh build refreshes the typed basis exactly as `verify_fd`'s
/// coordinate path does.
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

#[test]
fn hubbard_u_deriv1_matches_eu_fd() {
    use pyscf_pbc_dft::krks::Krks;

    let cell = hubbard_c2_cell();
    let kpts = cell.make_kpts([2, 1, 1]).expect("k-mesh");
    let nkpts = kpts.len();
    let nao = cell.mol.nao_nr;
    let cfg = hubbard_cfg();
    let mf = Krks::new(cell.clone(), &kpts, "lda,vwn").expect("KRKS builds");
    let res = mf.run().expect("KRKS converges");
    assert!(
        res.converged,
        "KRKS must converge — the density is the gate input"
    );
    let dm = res.dm[0].clone();

    let de = hubbard_u_deriv1(&cell, &dm, &kpts, &cfg).expect("U term builds");
    assert!(
        de.iter().flatten().all(|v| v.is_finite()),
        "U term must be finite"
    );
    assert!(
        de.iter().flatten().any(|v| v.abs() > 1e-10),
        "U term is vacuous on this fixture"
    );

    // Central difference of `add_vhubbard`'s E_U at FIXED dm — the energy
    // path is an independent code path, not the same algebra re-run.
    const H: f64 = 5e-5;
    let eu = |c: &Cell| {
        let mut v = vec![vec![CTensor::zeros(nao * nao); nkpts]];
        add_vhubbard(&mut v, c, &kpts, &vec![dm.clone()], &cfg).expect("E_U oracle")
    };
    let mut worst = 0.0_f64;
    for ia in 0..cell.natm {
        for c in 0..3 {
            let e1 = eu(&shifted_cell(&cell, ia, c, H));
            let e2 = eu(&shifted_cell(&cell, ia, c, -H));
            let fd = oracle_sum(&[e1, -e2]) / (2.0 * H);
            let d = oracle_sum(&[de[ia][c], -fd]).abs();
            worst = worst.max(d);
        }
    }
    eprintln!("18-08 Task 2 (KRKS+U dE_U vs E_U FD): worst = {worst:.3e}");
    assert!(
        worst < 5e-7,
        "Task 2 FAILED: |dE_U − FD| = {worst:.3e} (upstream asserts 6 decimals)"
    );
}

// ---------------------------------------------------------------------------
// Task 3: driver wiring — kernel = base kernel + U rows.
// ---------------------------------------------------------------------------

/// Seeded column-major orbitals + ascending energies + RHF occupations.
fn synthetic_orbitals(nao: usize, nkpts: usize) -> (Vec<CTensor>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let z = |i: usize, j: usize, s: usize| {
        (
            ((i * 7 + j * 3 + s * 13) as f64 * 0.37).sin(),
            ((i * 5 + j * 11 + s * 7) as f64 * 0.61).cos(),
        )
    };
    let nocc = nao / 2;
    let coeff = (0..nkpts)
        .map(|k| {
            let mut re = vec![0.0_f64; nao * nao];
            let mut im = vec![0.0_f64; nao * nao];
            for m in 0..nao {
                for i in 0..nao {
                    let (r, v) = z(i, m, k + 41);
                    re[i + m * nao] = r;
                    im[i + m * nao] = v;
                }
            }
            CTensor::from_planes(re, im)
        })
        .collect();
    let energy = (0..nkpts)
        .map(|k| {
            (0..nao)
                .map(|m| -1.0 + 0.25 * m as f64 + 0.01 * k as f64)
                .collect()
        })
        .collect();
    let occ = (0..nkpts)
        .map(|_| (0..nao).map(|m| if m < nocc { 2.0 } else { 0.0 }).collect())
        .collect();
    (coeff, energy, occ)
}

#[test]
fn kernel_adds_u_rows_to_base() {
    use pyscf_pbc_dft::krks::Krks;

    // Diamond (`gth-szv`/`gth-pade`): the base kernel refuses all-electron
    // cells in `get_hcore` by design (upstream raises too), so the wiring
    // gate runs on a pseudopotential cell. The Hubbard site is every carbon
    // valence `p` (`USite::Shell`, like upstream's `'C 2p'` label); the
    // reference is `gth-szv` because MINAO cannot span a GTH valence space
    // (`kspu.rs` docs).
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
    let mf = Krks::new(cell.clone(), &kpts, "lda,vwn").expect("KRKS holder");
    let (coeff, energy, occ) = synthetic_orbitals(nao, nkpts);

    let g_base = KrksGradients::new(&mf, energy.clone(), coeff.clone(), occ.clone())
        .expect("base gradient object")
        .kernel()
        .expect("base kernel");
    // The U term on the same synthetic density, computed directly: the
    // wrapper must reproduce exactly these rows on top of the base.
    let dm_direct = pyscf_pbc_scf::krdm::make_rdm1(&coeff, &occ, nao);
    let u_direct = hubbard_u_deriv1(&cell, &dm_direct, &kpts, &cfg).expect("U term");
    let g_u = KrkspuGradients::new(&mf, energy, coeff, occ, &cfg)
        .expect("KRKS+U gradient object")
        .kernel()
        .expect("KRKS+U kernel");
    assert_eq!(g_base.len(), g_u.len());
    assert_eq!(g_u.len(), u_direct.len());
    let mut norm = 0.0_f64;
    for (ia, ((gb, gu), ud)) in g_base
        .iter()
        .zip(g_u.iter())
        .zip(u_direct.iter())
        .enumerate()
    {
        for x in 0..3 {
            // Wrapper adds exactly the direct U row.
            let wire = oracle_sum(&[gu[x], -gb[x], -ud[x]]).abs();
            assert!(
                wire < 1e-12,
                "atom {ia} comp {x}: wrapper/base mismatch = {wire:.3e} (must be exact)"
            );
            norm += ud[x] * ud[x];
        }
    }
    // The wiring is non-vacuous: the U rows are nonzero on this fixture.
    assert!(norm > 0.0, "U wiring is vacuous on this fixture");
}
