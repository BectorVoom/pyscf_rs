//! Plan 18-13 Task 1 + Task 5 gates — `uks_stress` (spin index, gamma).
//!
//! Gate A at the tier each upstream test carries (`test_uks_stress.py`):
//! LDA `get_vxc` at 5e-9 (`:55`), `with_j` at 1e-8 (`:124`); GGA/MGGA are
//! named refusals port-side (missing `deriv2` kernel / `vtau` — the 18-12
//! §12 boundary), pinning the refusal rather than the unreachable number.
//! Gate D (LDA end to end) at 1e-6 Ha/Bohr³ with `vol` named (`:143`).
//!
//! The finite-difference oracles live HERE, never in `src/stress/uks.rs`.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::Unit;
use pyscf_core::{ParsedBasis, ShellSpec};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::fftdf::Fftdf;
use pyscf_pbc_df::traits::{JkOpts, PeriodicDf};
use pyscf_pbc_dft::gamma::uks as gamma_uks;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_grad::gamma_uhf::gamma_make_rdm1e_uhf;
use pyscf_pbc_grad::stress::{
    VxcStrainOpts, finite_diff_cells, strain_block_footprint, strain_block_footprint_kpts,
    strain_block_size_nset, uks_get_vxc, uks_stress_kernel,
};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// Upstream's half step for the integral strain differences; [`finite_diff_cells`]
/// takes the FULL separation, hence `2 * HALF`.
const HALF_DISP: f64 = 1e-5;
const FULL_DISP: f64 = 2e-5;

const GAMMA: [[f64; 3]; 1] = [[0.0; 3]];

/// `np.random.seed(5); eye(3)*5 + rand(3,3) - .5` — the component-test lattice.
const SEED5_A5: [[f64; 3]; 3] = [
    [4.721993171089739, 0.3707323061773764, -0.2932808446605736],
    [0.41861090793792155, 4.988411188794829, 0.11174386290264571],
    [0.26590785648031556, 0.018417987872943242, 4.796800501576222],
];

/// `np.random.seed(5); eye(3)*3 + rand(3,3) - .5` — Gate-D LDA lattice.
const SEED5_A3: [[f64; 3]; 3] = [
    [2.7219931710897396, 0.3707323061773764, -0.2932808446605736],
    [0.41861090793792155, 2.9884111887948293, 0.11174386290264571],
    [0.26590785648031556, 0.018417987872943242, 2.796800501576222],
];

fn he_sp_basis() -> BasisInput {
    BasisInput::Parsed(ParsedBasis {
        shells: vec![
            ShellSpec {
                l: 0,
                exponents: vec![0.5],
                coeffs: vec![vec![1.0]],
            },
            ShellSpec {
                l: 1,
                exponents: vec![0.8],
                coeffs: vec![vec![1.0]],
            },
        ],
    })
}

fn seed5_he_cell() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("He 1 1 1; He 2 1.5 2.4".into()),
            basis: he_sp_basis(),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(SEED5_A5),
        mesh: None,
        precision: 1e-10,
        pseudo: None,
        ..Default::default()
    })
    .expect("seed-5 He cell builds")
}

fn rng_unit(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Symmetric test density `dm = A·Aᵀ`, spin-major flat `(2, nao, nao)`.
fn sym_dm_spin(nao: usize, seed: u64) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * nao * nao);
    let mut s = seed;
    for _ in 0..2 {
        let a: Vec<f64> = (0..nao * nao).map(|_| rng_unit(&mut s) - 0.5).collect();
        for i in 0..nao {
            for j in 0..nao {
                let mut t = 0.0;
                for k in 0..nao {
                    t += a[i * nao + k] * a[j * nao + k];
                }
                out.push(t);
            }
        }
    }
    out
}

/// `ni.nr_uks(cell, UniformGrids(cell), xc, dm)[1]` — the Gate-A XC oracle
/// (`test_uks_stress.py:53-54` etc.). The grid follows the (possibly
/// strained) cell, exactly as upstream's fresh `UniformGrids(cell1)` does.
fn nr_uks_exc(cell: &Cell, xc: &str, dm: &[f64]) -> f64 {
    let ni = KNumInt::new(&[]);
    let grids = PeriodicGrids::uniform(cell, None).expect("uniform grids");
    let nao = cell.mol.nao_nr;
    let dm_a = CTensor::from_planes(dm[..nao * nao].to_vec(), vec![0.0; nao * nao]);
    let dm_b = CTensor::from_planes(dm[nao * nao..].to_vec(), vec![0.0; nao * nao]);
    ni.nr_uks(cell, &grids, xc, &[vec![vec![dm_a]], vec![vec![dm_b]]], 1, None)
        .expect("nr_uks oracle")
        .excsum[0]
}

/// `FFTDF(cell1).get_jk(dm_tot, with_k=False)[0]`, row-major.
fn fftdf_j(cell: &Cell, dm_tot: &[f64]) -> Vec<f64> {
    let df = Fftdf::new(cell.clone(), &[]).expect("fftdf builds");
    let nao = cell.mol.nao_nr;
    let dm_ct = CTensor::from_planes(dm_tot.to_vec(), vec![0.0; nao * nao]);
    let out = df
        .get_jk(
            &[vec![dm_ct]],
            &GAMMA,
            JkOpts {
                hermi: 1,
                kpts_band: None,
                with_j: true,
                with_k: false,
                exxdiv: None,
                omega: None,
                kk_symmetry: false,
            },
        )
        .expect("fftdf get_jk");
    out.vj.expect("vj")[0][0].re.clone()
}

/// `np.einsum('sij,ji->', dm, dv)` — upstream's spin-summed trace order
/// (`:122`), via `oracle_sum`.
fn trace_spin_dot(dm: &[f64], dv: &[f64], nao: usize) -> f64 {
    let mut terms = Vec::with_capacity(2 * nao * nao);
    for s in 0..2 {
        for i in 0..nao {
            for j in 0..nao {
                terms.push(dm[s * nao * nao + i * nao + j] * dv[j * nao + i]);
            }
        }
    }
    oracle_sum(&terms)
}

const FIVE: [(usize, usize); 5] = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)];

// ---------------------------------------------------------------------------
// Gate A — get_vxc LDA (upstream :36-55, bound 5e-9 at :55).
// ---------------------------------------------------------------------------

/// Gate A — `test_get_vxc_lda` (`test_uks_stress.py:36-55`, bound 5e-9 at
/// `:55`). `uks_get_vxc` without `with_j`/`with_nuc` is the pure-XC strain.
#[test]
fn gate_a_uks_get_vxc_lda_matches_nr_uks_fd() {
    let cell = seed5_he_cell();
    let nao = cell.mol.nao_nr;
    let dm = sym_dm_spin(nao, 0x51ab);
    let dat = uks_get_vxc(&cell, &dm, "lda,", VxcStrainOpts::default()).expect("uks get_vxc lda");
    for (x, y) in FIVE {
        let pair = finite_diff_cells(&cell, &[], x, y, FULL_DISP).expect("strain pair");
        let e1 = nr_uks_exc(&pair.plus, "lda,", &dm);
        let e2 = nr_uks_exc(&pair.minus, "lda,", &dm);
        let d = (dat[x][y] - oracle_sum(&[e1, -e2]) / FULL_DISP).abs();
        eprintln!("uks get_vxc lda ({x},{y}): {d:.3e}");
        assert!(
            d < 5e-9,
            "uks get_vxc lda ({x},{y}): analytic vs FD = {d:.3e} (Gate A = 5e-9)"
        );
    }
}

// ---------------------------------------------------------------------------
// get_vxc GGA/MGGA — NAMED REFUSALS (not gates).
// ---------------------------------------------------------------------------

/// `test_get_vxc_gga` (`:57-77`) has no port: GGA refuses by name (missing
/// `deriv2` AO kernel), never by falling back to LDA.
#[test]
fn uks_get_vxc_gga_refuses_without_deriv2() {
    let cell = seed5_he_cell();
    let dm = sym_dm_spin(cell.mol.nao_nr, 0x51ab);
    let err = uks_get_vxc(&cell, &dm, "pbe", VxcStrainOpts::default()).expect_err("GGA must refuse");
    assert!(
        err.to_string().contains("deriv2"),
        "GGA refusal must name the missing deriv2 kernel, got: {err}"
    );
}

/// `test_get_vxc_mgga` (`:79-98`) has no port: MGGA refuses by name (missing
/// `tau`/`vtau` supply plus the same `deriv2` kernel).
#[test]
fn uks_get_vxc_mgga_refuses_without_tau() {
    let cell = seed5_he_cell();
    let dm = sym_dm_spin(cell.mol.nao_nr, 0x51ab);
    let err =
        uks_get_vxc(&cell, &dm, "scan", VxcStrainOpts::default()).expect_err("MGGA must refuse");
    assert!(
        err.to_string().contains("tau"),
        "MGGA refusal must name the missing tau supply, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Gate A — with_j (upstream :100-124, bound 1e-8 at :124).
// ---------------------------------------------------------------------------

/// Gate A — `test_get_j` (`test_uks_stress.py:100-124`, bound 1e-8 at
/// `:124`): `with_j` strain vs FD of `0.5·Tr(dm·vj) + exc` over the
/// spin-summed density.
#[test]
fn gate_a_uks_get_j_matches_fd() {
    let cell = seed5_he_cell();
    let nao = cell.mol.nao_nr;
    let dm = sym_dm_spin(nao, 0x51ab);
    let mut dm_tot = vec![0.0; nao * nao];
    for p in 0..nao * nao {
        dm_tot[p] = oracle_sum(&[dm[p], dm[nao * nao + p]]);
    }
    let dat = uks_get_vxc(
        &cell,
        &dm,
        "lda,",
        VxcStrainOpts {
            with_j: true,
            ..Default::default()
        },
    )
    .expect("uks get_vxc with_j");
    for (x, y) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)] {
        let pair = finite_diff_cells(&cell, &[], x, y, FULL_DISP).expect("strain pair");
        let vj1 = fftdf_j(&pair.plus, &dm_tot);
        let vj2 = fftdf_j(&pair.minus, &dm_tot);
        let mut dv = vec![0.0; nao * nao];
        for (d, (a, b)) in dv.iter_mut().zip(vj1.iter().zip(vj2.iter())) {
            *d = oracle_sum(&[*a, -*b]);
        }
        let de = oracle_sum(&[
            0.5 * trace_spin_dot(&dm, &dv, nao),
            oracle_sum(&[
                nr_uks_exc(&pair.plus, "lda,", &dm),
                -nr_uks_exc(&pair.minus, "lda,", &dm),
            ]),
        ]);
        let d = (dat[x][y] - de / FULL_DISP).abs();
        eprintln!("uks get_j ({x},{y}): {d:.3e}");
        assert!(
            d < 1e-8,
            "uks get_j ({x},{y}): analytic vs FD = {d:.3e} (Gate A = 1e-8)"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 1 — the block budget counts both AO arrays and is parameterised on
// nset (spin), not hard-coded to 1.
// ---------------------------------------------------------------------------

#[test]
fn uks_block_budget_counts_both_arrays_and_nset() {
    // GGA-scale row from 18-REVIEW §3.1: the footprint formula the gate pins.
    let (nao, blk, nkpts) = (26usize, 8000usize, 8usize);
    let fp = strain_block_footprint_kpts(blk, nao, 10, 36, 16, nkpts);
    assert_eq!(fp, nkpts as u128 * 46 * blk as u128 * nao as u128 * 16);
    assert_eq!(
        strain_block_footprint_kpts(blk, nao, 10, 36, 16, 1),
        strain_block_footprint(blk, nao, 10, 36, 16)
    );
    // At a fixed tight budget the spin-doubled scratch shrinks the block:
    // nset = 2 must never exceed nset = 1, and a tiny budget clamps to 1.
    let tight = 0.05;
    let b1 = strain_block_size_nset(100_000, nao, 4, 9, 8, tight, 1, 1);
    let b2 = strain_block_size_nset(100_000, nao, 4, 9, 8, tight, 1, 2);
    assert!(b2 <= b1, "nset = 2 block {b2} must not exceed nset = 1 block {b1}");
    assert_eq!(strain_block_size_nset(100_000, nao, 4, 9, 8, 1e-9, 1, 2), 1);
    assert_eq!(strain_block_size_nset(0, nao, 4, 9, 8, 4000.0, 1, 2), 0);
}

// ---------------------------------------------------------------------------
// Clause-7 determinism: bit-identical at RAYON_NUM_THREADS = 1 and 8.
// ---------------------------------------------------------------------------

#[test]
fn uks_accumulator_is_bit_identical_across_rayon_threads() {
    let cell = seed5_he_cell();
    let nao = cell.mol.nao_nr;
    let dm = sym_dm_spin(nao, 0x51ab);
    let ngrids = cell.uniform_grids(None).expect("grids").coords.len();
    let mb = ((ngrids / 4).max(1) * nao * 13 * 8) as f64 / 1e6;
    let opts = VxcStrainOpts {
        max_memory_mb: Some(mb),
        ..Default::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| uks_get_vxc(&cell, &dm, "lda,", opts).expect("uks get_vxc"))
    };
    let one = run(1);
    let eight = run(8);
    for x in 0..3 {
        for y in 0..3 {
            assert_eq!(
                one[x][y].to_bits(),
                eight[x][y].to_bits(),
                "uks block accumulator ({x},{y}) differs across thread counts"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Gate D — end to end (upstream :126-143, 1e-6 Ha/Bohr³ at :143).
// ---------------------------------------------------------------------------

/// Upstream's Gate-D LDA fixture (`:127-132`): H2 on the seed-5 `a = 3`
/// lattice, `svwn`, basis `[[0, [1.5, 1]], [1, [.8, 1]]]`.
fn gate_d_h_cell() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("H 1 1 1; H 2 1.5 2.4".into()),
            basis: BasisInput::Parsed(ParsedBasis {
                shells: vec![
                    ShellSpec {
                        l: 0,
                        exponents: vec![1.5],
                        coeffs: vec![vec![1.0]],
                    },
                    ShellSpec {
                        l: 1,
                        exponents: vec![0.8],
                        coeffs: vec![vec![1.0]],
                    },
                ],
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
    .expect("Gate-D H cell builds")
}

fn uks_energy(cell: &Cell, xc: &str) -> (f64, Vec<f64>, Vec<f64>) {
    let mf = gamma_uks(cell.clone(), xc).expect("uks builds");
    let res = mf.run().expect("uks converges");
    assert!(res.converged, "Gate-D SCF must converge");
    assert_eq!(res.nset, 2, "UKS must carry two spin channels");
    let nao = cell.mol.nao_nr;
    let ka = res.idx(0, 0);
    let kb = res.idx(1, 0);
    let mut dm_spin = vec![0.0; 2 * nao * nao];
    dm_spin[..nao * nao].copy_from_slice(&res.dm[0][0].re);
    dm_spin[nao * nao..].copy_from_slice(&res.dm[1][0].re);
    let [dme_a, dme_b] = gamma_make_rdm1e_uhf(
        [&res.mo_coeff[ka].re[..], &res.mo_coeff[kb].re[..]],
        [&res.mo_energy[ka][..], &res.mo_energy[kb][..]],
        [&res.mo_occ[ka][..], &res.mo_occ[kb][..]],
        nao,
    )
    .expect("dme0");
    let mut dme0 = vec![0.0; nao * nao];
    for p in 0..nao * nao {
        dme0[p] = oracle_sum(&[dme_a[p], dme_b[p]]);
    }
    (res.e_tot, dm_spin, dme0)
}

/// Gate D — `test_lda_vs_finite_difference`
/// (`test_uks_stress.py:126-143`, bound 1e-6 Ha/Bohr³ at `:143`).
#[test]
fn gate_d_uks_lda_stress_matches_scf_fd_over_vol() {
    let cell = gate_d_h_cell();
    let xc = "lda,vwn";
    let (_, dm_spin, dme0) = uks_energy(&cell, xc);
    let dat = uks_stress_kernel(&cell, &dm_spin, &dme0, xc).expect("uks stress kernel");
    let vol = cell.vol();
    for (x, y) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)] {
        let pair = finite_diff_cells(&cell, &[], x, y, 2e-3).expect("strain pair");
        let e1 = gamma_uks(pair.plus, xc)
            .expect("uks plus")
            .run()
            .expect("plus converges")
            .e_tot;
        let e2 = gamma_uks(pair.minus, xc)
            .expect("uks minus")
            .run()
            .expect("minus converges")
            .e_tot;
        let d = (dat[x][y] - oracle_sum(&[e1, -e2]) / 2e-3 / vol).abs();
        eprintln!("uks Gate D lda ({x},{y}): {d:.3e} Ha/Bohr^3");
        assert!(
            d < 1e-6,
            "uks Gate D lda ({x},{y}): kernel vs SCF FD = {d:.3e} (Gate D = 1e-6 Ha/Bohr^3)"
        );
    }
}

/// Gate D at GGA/MGGA has no port: `kernel` refuses through `uks_get_vxc`
/// (missing `deriv2` kernel / `tau`). Upstream: `test_gga_vs_finite_difference_high_cost`
/// (`:145-163`) and `test_mgga_vs_finite_difference_high_cost` (`:165-182`).
#[test]
fn gate_d_uks_gga_mgga_refuse() {
    let cell = gate_d_h_cell();
    let nao = cell.mol.nao_nr;
    let dm = sym_dm_spin(nao, 0x51ab);
    let dme = vec![0.0; nao * nao];
    for xc in ["pbe", "scan"] {
        let err =
            uks_stress_kernel(&cell, &dm, &dme, xc).expect_err("non-LDA kernel must refuse");
        assert!(
            err.to_string().contains("deriv2") || err.to_string().contains("tau"),
            "{xc} kernel refusal must name deriv2/tau, got: {err}"
        );
    }
}

/// `kernel` refuses hybrid functionals, mirroring upstream's
/// `NotImplementedError` (`uks_stress.py:208-209`).
#[test]
fn uks_stress_kernel_refuses_hybrid() {
    let cell = seed5_he_cell();
    let nao = cell.mol.nao_nr;
    let dm = sym_dm_spin(nao, 0x51ab);
    let dme = vec![0.0; nao * nao];
    let err = uks_stress_kernel(&cell, &dm, &dme, "pbe0").expect_err("hybrid must refuse");
    assert!(
        err.to_string().contains("hybrid"),
        "hybrid refusal must name hybrid DFT, got: {err}"
    );
}
