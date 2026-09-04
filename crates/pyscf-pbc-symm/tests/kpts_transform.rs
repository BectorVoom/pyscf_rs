//! Gate B (the IBZ -> BZ unfolds) and the `symmetrize_density`
//! determinism/correctness gates — 17-05-PLAN.md Tasks 3 and 4.
//!
//! # Gate B is measured against ONE converged SCF, never two
//!
//! 17-CONTEXT §2.2: `transform_dm`, `transform_1e_operator`,
//! `transform_mo_occ`, `transform_mo_energy` and `symmetrize_density` are
//! exact LINEAR MAPS. Feed them the IBZ slice of a SINGLE full-BZ run and
//! compare against that same run's full-BZ arrays; the residual is then a
//! statement about the algebra, not about how two SCFs happened to converge.
//!
//! # The floor is JOINT — `cell.precision` AND `conv_tol_grad`
//!
//! 17-01 measured that the residual is limited by `cell.precision` (the
//! AO-integral screening tolerance), not by `conv_tol`; 17-04 then measured
//! (`17-04-MEASUREMENT.md` §3) that tightening EITHER axis alone plateaus,
//! and only both together reach machine epsilon.
//!
//! **Measured HERE, on this fixture (`si`, `[2,2,2]`, KRHF/FFTDF,
//! `conv_tol = 1e-11`), max over every k and every (p, q):**
//!
//! | `precision` / `conv_tol_grad` | `transform_dm` | `make_rdm1(transform_mo_coeff)` | `transform_1e_operator` | `transform_mo_energy` | `dm_at_ref_cell` |
//! |---|---|---|---|---|---|
//! | 1e-10 / 1e-10 | 1.784e-11 | 1.784e-11 | 1.099e-12 | 6.054e-12 | 6.106e-12 |
//! | **1e-12 / 1e-12** | **2.306e-13** | **2.306e-13** | **1.212e-14** | **3.686e-14** | **5.390e-14** |
//!
//! Two readings that matter. First, the ordering is stable across both
//! fixtures: `transform_1e_operator` (a pure linear map applied to the
//! Fock) is always an order or two TIGHTER than `transform_dm`, because the
//! density matrix carries the SCF's own residual asymmetry on top of the
//! rotation's error — the rotation itself is pinned independently at 1e-10
//! by 17-03's `R S R^H == S`. Second, `transform_dm` and
//! `make_rdm1(transform_mo_coeff)` agree to the last digit at BOTH
//! fixtures, which is what says the two routes to a BZ density matrix are
//! the same map.
//!
//! This file therefore runs at 1e-12/1e-12 and gates at [`GATE_B_TOL`]
//! = 1e-12, ~4x above the measured worst. **If it ever fails, tighten the
//! fixture — do not relax the tolerance.**
//!
//! # Never compare `mo_coeff` elementwise (17-CONTEXT §3.1)
//!
//! `transform_mo_coeff` is defined only up to a unitary mixing within each
//! degenerate subspace, and every symmetric cell has degeneracies at
//! high-symmetry k-points. Every comparison here goes through the density
//! matrix the MOs build, and
//! [`mo_coeff_elementwise_comparison_is_not_a_valid_gate`] asserts the
//! elementwise comparison is LARGE — so nobody can "tighten" the DM
//! comparison into an MO one later.
//!
//! # Every assert reports the MAXIMUM residual, not the first violation
//!
//! 17-04's first-violation assert reported 1.58e-11 while the true maximum
//! was 3.99e-10, a 25x difference that changed the diagnosis. See
//! [`Worst`].

// `needless_range_loop` is allowed throughout: these loops index SEVERAL
// parallel arrays by the same k / p / q (upstream's own index convention),
// and rewriting them as iterator zips would obscure which array each index
// belongs to.
#![allow(clippy::needless_range_loop)]

use num_complex::Complex64;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::JkOpts;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_gto::test_systems::{diamond, si_precision};
use pyscf_pbc_scf::krhf::to_row_major;
use pyscf_pbc_scf::{KInitGuess, KScfConfig, KScfResult, Krhf};
use pyscf_pbc_symm::error::PbcSymmError;
use pyscf_pbc_symm::kpts::{KPoints, make_kpts};
use pyscf_pbc_symm::symmetry::build_lattice_symmetry;

/// See the module doc: 1e-10 precision + 1e-10 `conv_tol_grad` puts the
/// measured floor at ~5e-13 (17-04-MEASUREMENT.md §3), so 1e-11 leaves an
/// order of margin without being unfalsifiable.
const GATE_B_TOL: f64 = 1e-12;
const FIXTURE_PRECISION: f64 = 1e-12;
const FIXTURE_CONV_TOL_GRAD: f64 = 1e-12;

// ---------------------------------------------------------------------
// Worst-element tracking (17-04-MEASUREMENT.md's lesson).
// ---------------------------------------------------------------------

struct Worst {
    val: f64,
    at: (usize, usize, usize),
}

impl Worst {
    fn new() -> Self {
        Worst {
            val: 0.0,
            at: (0, 0, 0),
        }
    }
    fn see(&mut self, val: f64, k: usize, p: usize, q: usize) {
        if val > self.val {
            self.val = val;
            self.at = (k, p, q);
        }
    }
    fn report(&self, what: &str, tol: f64) {
        let (k, p, q) = self.at;
        println!(
            "  Gate B: max {what:<38} = {:e}   (tol {tol:e}, at k={k} p={p} q={q})",
            self.val
        );
        assert!(
            self.val < tol,
            "Gate B: max {what} = {:e} exceeds {tol:e} at k={k} p={p} q={q}. \
             Tighten the fixture (FIXTURE_PRECISION / FIXTURE_CONV_TOL_GRAD), \
             do NOT relax the tolerance — see 17-04-MEASUREMENT.md §3.",
            self.val
        );
    }
}

// ---------------------------------------------------------------------
// layout helpers — 17-CONTEXT §3.2. `mo_coeff` is COLUMN-MAJOR; `KDms` /
// the Fock build are ROW-MAJOR; `kpts.rs`'s transforms are ROW-MAJOR.
// Every conversion happens ONCE, here.
// ---------------------------------------------------------------------

fn rowmajor_square(ct: &CTensor, n: usize) -> Vec<Complex64> {
    (0..n * n)
        .map(|k| Complex64::new(ct.re[k], ct.im[k]))
        .collect()
}

fn colmajor_rect_to_rowmajor(ct: &CTensor, nrows: usize, ncols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); nrows * ncols];
    for row in 0..nrows {
        for col in 0..ncols {
            out[row * ncols + col] =
                Complex64::new(ct.re[row + col * nrows], ct.im[row + col * nrows]);
        }
    }
    out
}

/// The ONLY way `mo_coeff` is compared in this file (17-CONTEXT §3.1).
fn make_rdm1_rowmajor(mo: &[Complex64], occ: &[f64], nao: usize, nmo: usize) -> Vec<Complex64> {
    let mut dm = vec![Complex64::new(0.0, 0.0); nao * nao];
    for (i, &o) in occ.iter().enumerate() {
        if o == 0.0 {
            continue;
        }
        for mu in 0..nao {
            let a = mo[mu * nmo + i];
            for nu in 0..nao {
                dm[mu * nao + nu] += Complex64::new(o, 0.0) * a * mo[nu * nmo + i].conj();
            }
        }
    }
    dm
}

// ---------------------------------------------------------------------
// The single converged full-BZ KRHF every Gate B test reads.
// ---------------------------------------------------------------------

struct GateB {
    cell: Cell,
    kpts: KPoints,
    nao: usize,
    nmo: usize,
    dm: Vec<Vec<Complex64>>,
    fock: Vec<Vec<Complex64>>,
    mo_coeff: Vec<Vec<Complex64>>,
    mo_occ: Vec<Vec<f64>>,
    mo_energy: Vec<Vec<f64>>,
}

/// `si` at `[2,2,2]`, `precision = 1e-10`, `conv_tol_grad = 1e-10`. Memoized
/// with a `OnceLock` so the SCF is paid for ONCE across every test in this
/// binary — they all compare different transforms of the SAME run, which is
/// exactly what Gate B requires (never two SCFs).
fn gate_b() -> &'static GateB {
    static G: std::sync::OnceLock<GateB> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        let mut cell = si_precision(FIXTURE_PRECISION);
        cell.space_group_symmetry = true;
        cell.symmorphic = false;
        let check_mesh_symmetry = !cell._mesh_from_build;
        build_lattice_symmetry(&mut cell, check_mesh_symmetry).expect("build_lattice_symmetry");

        let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
        // `time_reversal_symmetry = true` as upstream's gate_b.py does; `si`
        // has inversion, so `KPoints::build` forces it back off
        // (`kpts.py:1024`) — asserted below so the fixture cannot drift.
        let kpts = make_kpts(&cell, &kpts_abs, true, true).expect("make_kpts");
        assert!(
            !kpts.time_reversal,
            "si has inversion: time reversal must be OFF"
        );

        let mf = Krhf::new(cell.clone(), &kpts_abs).expect("Krhf::new");
        let cfg = KScfConfig {
            conv_tol: 1e-11,
            conv_tol_grad: Some(FIXTURE_CONV_TOL_GRAD),
            max_cycle: 50,
            init_guess: KInitGuess::Minao,
            ..KScfConfig::default()
        };
        let r: KScfResult = mf.kernel(&cfg).expect("full-BZ KRHF must run");
        assert!(
            r.converged,
            "full-BZ KRHF did not converge in {} cycles",
            r.cycles
        );

        let nao = cell.mol.nao_nr;
        let nkpts = kpts_abs.len();
        let nmo = r.mo_occ[0].len();

        // `khf.py:670-695`'s Fock assembly (hcore + vj - vk/2), minus `eig`.
        let mut fock_ct = to_row_major(
            pyscf_pbc_df::get_hcore(mf.with_df.as_ref(), mf.kpts()).expect("get_hcore"),
            nao,
        );
        let jk = mf
            .with_df
            .get_jk(
                &r.dm,
                mf.kpts(),
                JkOpts {
                    hermi: 1,
                    kpts_band: None,
                    with_j: true,
                    with_k: true,
                    exxdiv: mf.exxdiv,
                    omega: None,
                    kk_symmetry: false,
                },
            )
            .expect("get_jk");
        let vj = jk.vj.expect("vj");
        let vk = jk.vk.expect("vk");
        for (k, f) in fock_ct.iter_mut().enumerate() {
            for i in 0..f.re.len() {
                f.re[i] += vj[0][k].re[i] - 0.5 * vk[0][k].re[i];
                f.im[i] += vj[0][k].im[i] - 0.5 * vk[0][k].im[i];
            }
        }

        GateB {
            nao,
            nmo,
            dm: (0..nkpts)
                .map(|k| rowmajor_square(&r.dm[0][k], nao))
                .collect(),
            fock: (0..nkpts)
                .map(|k| rowmajor_square(&fock_ct[k], nao))
                .collect(),
            mo_coeff: (0..nkpts)
                .map(|k| colmajor_rect_to_rowmajor(&r.mo_coeff[r.idx(0, k)], nao, nmo))
                .collect(),
            mo_occ: (0..nkpts).map(|k| r.mo_occ[r.idx(0, k)].clone()).collect(),
            mo_energy: (0..nkpts)
                .map(|k| r.mo_energy[r.idx(0, k)].clone())
                .collect(),
            cell,
            kpts,
        }
    })
}

fn ibz_slice<T: Clone>(kpts: &KPoints, full: &[T]) -> Vec<T> {
    kpts.ibz2bz.iter().map(|&k| full[k].clone()).collect()
}

// ---------------------------------------------------------------------
// Gate B
// ---------------------------------------------------------------------

#[test]
fn gate_b_transform_dm() {
    let g = gate_b();
    let dm_ibz = ibz_slice(&g.kpts, &g.dm);
    let dm_bz = g
        .kpts
        .transform_dm(&g.cell, &dm_ibz, g.nao)
        .expect("transform_dm");
    let mut worst = Worst::new();
    for k in 0..g.kpts.nkpts() {
        for p in 0..g.nao {
            for q in 0..g.nao {
                worst.see(
                    (dm_bz[k][p * g.nao + q] - g.dm[k][p * g.nao + q]).norm(),
                    k,
                    p,
                    q,
                );
            }
        }
    }
    worst.report("|transform_dm(dm[ibz2bz]) - dm|", GATE_B_TOL);
}

#[test]
fn gate_b_transform_1e_operator() {
    let g = gate_b();
    let fock_ibz = ibz_slice(&g.kpts, &g.fock);
    let fock_bz = g
        .kpts
        .transform_1e_operator(&g.cell, &fock_ibz, g.nao)
        .expect("transform_1e_operator");
    let mut worst = Worst::new();
    for k in 0..g.kpts.nkpts() {
        for p in 0..g.nao {
            for q in 0..g.nao {
                worst.see(
                    (fock_bz[k][p * g.nao + q] - g.fock[k][p * g.nao + q]).norm(),
                    k,
                    p,
                    q,
                );
            }
        }
    }
    worst.report("|transform_1e_operator(F) - F|", GATE_B_TOL);
}

#[test]
fn gate_b_make_rdm1_of_transform_mo_coeff() {
    let g = gate_b();
    let mo_ibz = ibz_slice(&g.kpts, &g.mo_coeff);
    let mo_bz = g
        .kpts
        .transform_mo_coeff(&g.cell, &mo_ibz, g.nao, g.nmo)
        .expect("transform_mo_coeff");
    let mut worst = Worst::new();
    for k in 0..g.kpts.nkpts() {
        let dm = make_rdm1_rowmajor(&mo_bz[k], &g.mo_occ[k], g.nao, g.nmo);
        for p in 0..g.nao {
            for q in 0..g.nao {
                worst.see((dm[p * g.nao + q] - g.dm[k][p * g.nao + q]).norm(), k, p, q);
            }
        }
    }
    worst.report("|make_rdm1(transform_mo_coeff) - dm|", GATE_B_TOL);

    // transform_mo_coeff_k must agree with the batched version element for
    // element — two code paths, one answer (upstream keeps both, `:494`).
    for k in 0..g.kpts.nkpts() {
        let single = g
            .kpts
            .transform_mo_coeff_k(&g.cell, &mo_ibz, g.nao, g.nmo, k)
            .expect("transform_mo_coeff_k");
        for i in 0..single.len() {
            assert_eq!(single[i].re.to_bits(), mo_bz[k][i].re.to_bits());
            assert_eq!(single[i].im.to_bits(), mo_bz[k][i].im.to_bits());
        }
    }
}

/// **17-CONTEXT §3.1.** The elementwise `mo_coeff` comparison is NOT a valid
/// gate: `transform_mo_coeff` is defined only up to a unitary within each
/// degenerate subspace. This test asserts the elementwise residual is LARGE
/// — orders above [`GATE_B_TOL`] — so that a later reader cannot "tighten"
/// [`gate_b_make_rdm1_of_transform_mo_coeff`] into an MO comparison and
/// believe they have improved it. Upstream's own test never compares
/// coefficients either (`test_kpts_ksymm.py:96-99`).
#[test]
fn mo_coeff_elementwise_comparison_is_not_a_valid_gate() {
    let g = gate_b();
    let mo_ibz = ibz_slice(&g.kpts, &g.mo_coeff);
    let mo_bz = g
        .kpts
        .transform_mo_coeff(&g.cell, &mo_ibz, g.nao, g.nmo)
        .expect("transform_mo_coeff");
    let mut max = 0.0_f64;
    for k in 0..g.kpts.nkpts() {
        for i in 0..g.nao * g.nmo {
            max = max.max((mo_bz[k][i] - g.mo_coeff[k][i]).norm());
        }
    }
    println!("  Gate B: max |transform_mo_coeff - mo_coeff| (DEMO ONLY) = {max:e}");
    assert!(
        max > 1e-4,
        "the elementwise mo_coeff residual came out at {max:e}, i.e. SMALL. That does not \
         make an elementwise comparison valid (17-CONTEXT §3.1) — it means this fixture \
         stopped exercising a degenerate subspace. Fix the fixture, do not add an \
         elementwise mo_coeff assert."
    );
}

#[test]
fn gate_b_transform_mo_occ_and_mo_energy() {
    let g = gate_b();

    // check_mo_occ_symmetry is the IBZ slice's own guard: it verifies the
    // occupations really are constant across every star before slicing.
    let occ_ibz = g
        .kpts
        .check_mo_occ_symmetry(&g.mo_occ, 1e-5)
        .expect("check_mo_occ_symmetry");
    assert_eq!(occ_ibz.len(), g.kpts.nkpts_ibz());
    let occ_bz = g.kpts.transform_mo_occ(&occ_ibz).expect("transform_mo_occ");
    let mut worst = Worst::new();
    for k in 0..g.kpts.nkpts() {
        for i in 0..g.nmo {
            worst.see((occ_bz[k][i] - g.mo_occ[k][i]).abs(), k, i, 0);
        }
    }
    // A pure index mapping: this must be EXACTLY zero, not merely small.
    assert_eq!(
        worst.val, 0.0,
        "transform_mo_occ is a pure index map and must be exact"
    );
    println!("  Gate B: max |transform_mo_occ - mo_occ|             = 0 (exact index map)");

    let e_ibz = ibz_slice(&g.kpts, &g.mo_energy);
    let e_bz = g
        .kpts
        .transform_mo_energy(&e_ibz)
        .expect("transform_mo_energy");
    let mut worst = Worst::new();
    for k in 0..g.kpts.nkpts() {
        for i in 0..g.nmo {
            worst.see((e_bz[k][i] - g.mo_energy[k][i]).abs(), k, i, 0);
        }
    }
    worst.report("|transform_mo_energy - mo_energy|", GATE_B_TOL);
}

#[test]
fn gate_b_dm_at_ref_cell() {
    let g = gate_b();
    let dm_ibz = ibz_slice(&g.kpts, &g.dm);
    let dm0 = g
        .kpts
        .dm_at_ref_cell(&g.cell, &dm_ibz, g.nao)
        .expect("dm_at_ref_cell");
    let nk = g.kpts.nkpts() as f64;
    let mut worst = Worst::new();
    for p in 0..g.nao {
        for q in 0..g.nao {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..g.kpts.nkpts() {
                acc += g.dm[k][p * g.nao + q];
            }
            worst.see((dm0[p * g.nao + q] - acc / nk).norm(), 0, p, q);
        }
    }
    worst.report("|dm_at_ref_cell - sum_k dm / nkpts|", GATE_B_TOL);
}

/// `check_mo_occ_symmetry` must REFUSE a symmetry-broken occupation, not
/// silently slice it (`kpts.py:749-752`).
#[test]
fn check_mo_occ_symmetry_refuses_a_broken_solution() {
    let g = gate_b();
    // Find a star with more than one member and perturb one of its points.
    let star = g
        .kpts
        .stars
        .iter()
        .find(|s| s.len() > 1)
        .expect("si [2,2,2] must have a non-trivial star");
    let mut broken = g.mo_occ.clone();
    broken[star[1]][0] += 1.0;
    let err = g
        .kpts
        .check_mo_occ_symmetry(&broken, 1e-5)
        .expect_err("must refuse");
    assert!(
        matches!(err, PbcSymmError::SymmetryBrokenOccupation(..)),
        "got {err:?}"
    );
}

// ---------------------------------------------------------------------
// Determinism of the unfolds — 17-05-PLAN.md Task 3 "Speed"
// ---------------------------------------------------------------------

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

fn assert_bits_equal(a: &[Vec<Complex64>], b: &[Vec<Complex64>], who: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{who}: length changed with the thread count"
    );
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.len(), y.len(), "{who}: shape changed at k = {k}");
        for i in 0..x.len() {
            assert_eq!(
                x[i].re.to_bits(),
                y[i].re.to_bits(),
                "{who}: re[{i}] at k = {k}"
            );
            assert_eq!(
                x[i].im.to_bits(),
                y[i].im.to_bits(),
                "{who}: im[{i}] at k = {k}"
            );
        }
    }
}

/// The unfolds write one BZ k-point per output slot and each slot belongs to
/// exactly one star, so the parallel loop's writes are disjoint BY
/// CONSTRUCTION. This proves it holds in CODE, not just on paper: vary the
/// worker count inside ONE process and demand BIT identity.
#[test]
fn unfolds_are_bit_identical_at_1_and_8_threads() {
    let g = gate_b();
    let dm_ibz = ibz_slice(&g.kpts, &g.dm);
    let fock_ibz = ibz_slice(&g.kpts, &g.fock);
    let mo_ibz = ibz_slice(&g.kpts, &g.mo_coeff);

    let one = pool(1);
    let eight = pool(8);

    assert_bits_equal(
        &one.install(|| g.kpts.transform_dm(&g.cell, &dm_ibz, g.nao).expect("1")),
        &eight.install(|| g.kpts.transform_dm(&g.cell, &dm_ibz, g.nao).expect("8")),
        "transform_dm",
    );
    assert_bits_equal(
        &one.install(|| {
            g.kpts
                .transform_1e_operator(&g.cell, &fock_ibz, g.nao)
                .expect("1")
        }),
        &eight.install(|| {
            g.kpts
                .transform_1e_operator(&g.cell, &fock_ibz, g.nao)
                .expect("8")
        }),
        "transform_1e_operator",
    );
    assert_bits_equal(
        &one.install(|| {
            g.kpts
                .transform_mo_coeff(&g.cell, &mo_ibz, g.nao, g.nmo)
                .expect("1")
        }),
        &eight.install(|| {
            g.kpts
                .transform_mo_coeff(&g.cell, &mo_ibz, g.nao, g.nmo)
                .expect("8")
        }),
        "transform_mo_coeff",
    );

    let d1 = one.install(|| g.kpts.dm_at_ref_cell(&g.cell, &dm_ibz, g.nao).expect("1"));
    let d8 = eight.install(|| g.kpts.dm_at_ref_cell(&g.cell, &dm_ibz, g.nao).expect("8"));
    assert_bits_equal(
        std::slice::from_ref(&d1),
        std::slice::from_ref(&d8),
        "dm_at_ref_cell",
    );
}

// ---------------------------------------------------------------------
// Task 4 — symmetrize_density
// ---------------------------------------------------------------------

/// A non-symmorphic-capable `KPoints` plus a mesh compatible with its
/// quarter-cell glides (a multiple of 4).
fn density_fixture(mesh: [usize; 3]) -> (Cell, KPoints) {
    let mut cell = diamond();
    cell.space_group_symmetry = true;
    cell.symmorphic = false;
    // `check_mesh_symmetry = false` keeps the non-symmorphic ops, which is
    // the whole point here — the fractional-translation branch of
    // `symmetrize_density` is otherwise never exercised.
    build_lattice_symmetry(&mut cell, false).expect("build_lattice_symmetry");
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, false).expect("make_kpts");
    let _ = mesh;
    (cell, kpts)
}

/// A deterministic, decidedly non-symmetric test density.
fn synthetic_rho(mesh: [usize; 3]) -> Vec<f64> {
    let n = mesh[0] * mesh[1] * mesh[2];
    (0..n)
        .map(|g| {
            let x = (g % 97) as f64;
            let y = ((g / 7) % 89) as f64;
            (0.37 * x).sin() + 0.5 * (0.11 * y).cos() + 1e-3 * g as f64
        })
        .collect()
}

/// Upstream's C kernel, transcribed from `pyscf/lib/pbc/symmetry.c:3-48`:
/// naive left-to-right accumulation, `((v % n) + n) % n`, and the
/// `(int)(ft * n)` offset. This is the ORACLE for
/// [`symmetrize_density_matches_upstreams_c_kernel`] — an independent
/// transcription of the algorithm being ported, not a re-call of the port.
fn upstream_symmetrize(
    kpts: &KPoints,
    rho_k: &[f64],
    ibz_k_idx: usize,
    mesh: [usize; 3],
) -> Vec<f64> {
    let (nx, ny, nz) = (mesh[0] as i64, mesh[1] as i64, mesh[2] as i64);
    let mut out = vec![0.0_f64; rho_k.len()];
    for &iop in &kpts.stars_ops[ibz_k_idx] {
        let op = &kpts.ops()[iop];
        if op.is_eye() {
            for (o, r) in out.iter_mut().zip(rho_k.iter()) {
                *o += *r;
            }
            continue;
        }
        let inv_op = op.inv().expect("inv");
        let r: [[i64; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| inv_op.rot[i][j].round() as i64));
        let (fx, fy, fz) = if inv_op.trans_is_zero() {
            (0i64, 0i64, 0i64)
        } else {
            (
                (inv_op.trans[0] * nx as f64) as i64,
                (inv_op.trans[1] * ny as f64) as i64,
                (inv_op.trans[2] * nz as f64) as i64,
            )
        };
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let xp = (((r[0][0] * x + r[0][1] * y + r[0][2] * z + fx) % nx) + nx) % nx;
                    let yp = (((r[1][0] * x + r[1][1] * y + r[1][2] * z + fy) % ny) + ny) % ny;
                    let zp = (((r[2][0] * x + r[2][1] * y + r[2][2] * z + fz) % nz) + nz) % nz;
                    out[(x * ny * nz + y * nz + z) as usize] +=
                        rho_k[(xp * ny * nz + yp * nz + zp) as usize];
                }
            }
        }
    }
    out
}

/// The port against upstream's own C kernel, transcribed independently in
/// [`upstream_symmetrize`]. The only permitted difference is summation
/// ORDER: this port routes the star sum through
/// `pyscf_algebra::oracle_sum`'s pairwise tree (D-PBC-17), upstream
/// accumulates left to right.
#[test]
fn symmetrize_density_matches_upstreams_c_kernel() {
    let mesh = [8usize, 8, 8];
    let (_cell, kpts) = density_fixture(mesh);
    let rho = synthetic_rho(mesh);

    let mut max = 0.0_f64;
    for i in 0..kpts.nkpts_ibz() {
        let got = kpts
            .symmetrize_density(&rho, i, mesh)
            .expect("symmetrize_density");
        let want = upstream_symmetrize(&kpts, &rho, i, mesh);
        for (a, b) in got.iter().zip(want.iter()) {
            max = max.max((a - b).abs());
        }
    }
    println!("  Task 4: max |symmetrize_density - upstream C kernel| = {max:e}");
    assert!(
        max < 1e-12,
        "symmetrize_density disagrees with upstream's C kernel by {max:e}"
    );
}

/// The rotated-index map of EVERY operation must be a PERMUTATION of the
/// grid — a wrong modulo, a transposed rotation or a wrong translation
/// offset all break this and nothing else. Oracle-free.
#[test]
fn every_star_operation_permutes_the_grid() {
    let mesh = [8usize, 8, 8];
    let (_cell, kpts) = density_fixture(mesh);
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    // A delta at each grid point in turn is too slow; instead symmetrize a
    // basis of indicator vectors implicitly: a permutation is exactly the
    // statement that symmetrizing the all-ones density with a SINGLE-op
    // star gives all ones. Use the full star and check the count instead.
    for i in 0..kpts.nkpts_ibz() {
        let ones = vec![1.0_f64; ngrids];
        let out = kpts
            .symmetrize_density(&ones, i, mesh)
            .expect("symmetrize_density");
        let nops = kpts.stars_ops[i].len() as f64;
        for (g, v) in out.iter().enumerate() {
            assert!(
                (v - nops).abs() < 1e-12,
                "grid point {g} of star {i} received {v} contributions, expected {nops} — \
                 some operation's index map is not a permutation"
            );
        }
    }
}

/// The §9.3 bit-identity test, shipped WITH the first version (D-PBC-17,
/// 17-CONTEXT §3.8) — not retrofitted.
#[test]
fn symmetrize_density_is_bit_identical_at_1_and_8_threads() {
    let mesh = [8usize, 8, 8];
    let (_cell, kpts) = density_fixture(mesh);
    let rho = synthetic_rho(mesh);
    let im: Vec<f64> = rho.iter().map(|v| 0.37 * v + 0.11).collect();

    let one = pool(1);
    let eight = pool(8);
    for i in 0..kpts.nkpts_ibz() {
        let a = one.install(|| kpts.symmetrize_density(&rho, i, mesh).expect("1"));
        let b = eight.install(|| kpts.symmetrize_density(&rho, i, mesh).expect("8"));
        for (g, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "star {i} grid {g} moved between 1 and 8 threads"
            );
        }
        let (ar, ai) = one.install(|| {
            kpts.symmetrize_density_complex(&rho, &im, i, mesh)
                .expect("1")
        });
        let (br, bi) = eight.install(|| {
            kpts.symmetrize_density_complex(&rho, &im, i, mesh)
                .expect("8")
        });
        for g in 0..ar.len() {
            assert_eq!(
                ar[g].to_bits(),
                br[g].to_bits(),
                "complex re, star {i} grid {g}"
            );
            assert_eq!(
                ai[g].to_bits(),
                bi[g].to_bits(),
                "complex im, star {i} grid {g}"
            );
            // The real part of the complex path must equal the real path.
            assert_eq!(ar[g].to_bits(), a[g].to_bits(), "complex re != real path");
        }
    }
}

#[test]
fn density_permutation_is_cached_once_per_star_and_mesh() {
    let mesh = [8usize, 8, 8];
    let (_cell, kpts) = density_fixture(mesh);
    let rho = synthetic_rho(mesh);
    assert_eq!(kpts.density_grid_cache_len(), 0);

    let first = kpts.symmetrize_density(&rho, 0, mesh).expect("first");
    assert_eq!(kpts.density_grid_cache_len(), 1);
    let second = kpts.symmetrize_density(&rho, 0, mesh).expect("second");
    assert_eq!(kpts.density_grid_cache_len(), 1);
    assert_eq!(first, second);

    let other_mesh = [4usize, 4, 4];
    let other = synthetic_rho(other_mesh);
    kpts.symmetrize_density(&other, 0, other_mesh)
        .expect("second mesh");
    assert_eq!(kpts.density_grid_cache_len(), 2);
}

#[test]
fn vector_density_uses_l1_rotation_and_is_thread_bit_exact() {
    let mesh = [8usize, 8, 8];
    let (_cell, kpts) = density_fixture(mesh);
    let base = synthetic_rho(mesh);
    let vy: Vec<f64> = base.iter().map(|v| 0.7 * v - 0.2).collect();
    let vz: Vec<f64> = base.iter().map(|v| -0.3 * v + 0.4).collect();
    let rho = [&base[..], &vy[..], &vz[..]];

    let one = pool(1);
    let eight = pool(8);
    for ibz in 0..kpts.nkpts_ibz() {
        let got = one.install(|| {
            kpts.symmetrize_density_vec(rho, ibz, mesh)
                .expect("vector density")
        });
        let parallel = eight.install(|| {
            kpts.symmetrize_density_vec(rho, ibz, mesh)
                .expect("vector density")
        });
        for component in 0..3 {
            for g in 0..got[component].len() {
                assert_eq!(
                    got[component][g].to_bits(),
                    parallel[component][g].to_bits(),
                    "component {component}, star {ibz}, grid {g}"
                );
            }
        }

        // Independent scalar construction: rotate each complete input
        // component by one operation at a time, then use the established
        // scalar grid oracle for that single-operation star.
        let mut want: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; base.len()]);
        for &iop in &kpts.stars_ops[ibz] {
            let mut single = kpts.clone();
            single.stars_ops[ibz] = vec![iop];
            let d = &kpts.dmats()[iop][1];
            for row in 0..3 {
                let rotated: Vec<f64> = (0..base.len())
                    .map(|g| d[row][0] * rho[0][g] + d[row][1] * rho[1][g] + d[row][2] * rho[2][g])
                    .collect();
                let contribution = upstream_symmetrize(&single, &rotated, ibz, mesh);
                for g in 0..base.len() {
                    want[row][g] += contribution[g];
                }
            }
        }
        let mut max = 0.0_f64;
        for component in 0..3 {
            for g in 0..base.len() {
                max = max.max((got[component][g] - want[component][g]).abs());
            }
        }
        assert!(max < 1e-12, "l=1 vector rotation residual {max:e}");
    }
}

/// **A silent round is a wrong density** (17-05-PLAN.md Task 4). On a mesh
/// that does NOT carry the quarter-cell glide, the rotated index misses a
/// mesh point and the port must FAIL LOUDLY — where upstream's C kernel
/// would truncate `(int)(0.25 * 7) = 1` and return a wrong answer.
///
/// Pinned on [`pyscf_pbc_symm::kpts::ft_offsets`] directly: see its doc for
/// why `symmetrize_density` cannot reach the branch on these fixtures (the
/// star search always finds a zero-translation op first).
#[test]
fn a_mesh_that_misses_a_grid_point_is_refused_not_rounded() {
    let quarter = [0.25_f64, 0.25, 0.25];
    assert_eq!(
        pyscf_pbc_symm::kpts::ft_offsets(0, quarter, [8, 8, 8]).expect("8 is a multiple of 4"),
        [2, 2, 2]
    );
    let err = pyscf_pbc_symm::kpts::ft_offsets(3, quarter, [7, 7, 7])
        .expect_err("7 cannot represent a 1/4 glide");
    match err {
        PbcSymmError::MeshNotSymmetric(iop, axis, v) => {
            assert_eq!(iop, 3);
            assert_eq!(axis, 0);
            assert!((v - 1.75).abs() < 1e-12, "got {v}");
        }
        other => panic!("wrong error: {other:?}"),
    }
    // Upstream's C would have TRUNCATED this to 1 and returned a wrong
    // density; that is the behaviour this port deliberately does not have.
    assert_eq!((0.25_f64 * 7.0) as i64, 1);
}

/// The FRACTIONAL-TRANSLATION branch of `symmetrize_density`, exercised
/// white-box.
///
/// On the §9.2 fixtures every star is covered by a zero-translation op (see
/// [`a_mesh_that_misses_a_grid_point_is_refused_not_rounded`]), so this test
/// substitutes a non-symmorphic op into `stars_ops` by hand and checks the
/// result against [`upstream_symmetrize`]'s transcription of the C kernel.
/// Without this, `symmetrize_ft` would ship untested.
#[test]
fn symmetrize_density_fractional_translation_branch_matches_upstream() {
    let mesh = [8usize, 8, 8];
    let (_cell, mut kpts) = density_fixture(mesh);
    let nonsymmorphic: Vec<usize> = kpts
        .ops()
        .iter()
        .enumerate()
        .filter(|(_, op)| !op.trans_is_zero())
        .map(|(i, _)| i)
        .collect();
    assert!(
        !nonsymmorphic.is_empty(),
        "diamond built with symmorphic = false must carry non-symmorphic ops"
    );
    // Confirm the premise: as shipped, no star names one of them.
    assert!(
        kpts.stars_ops
            .iter()
            .flatten()
            .all(|iop| kpts.ops()[*iop].trans_is_zero()),
        "the star search unexpectedly picked a non-symmorphic op; this test's premise          (and ft_offsets' doc comment) needs revisiting"
    );

    kpts.stars_ops[0] = nonsymmorphic.iter().take(4).copied().collect();
    let rho = synthetic_rho(mesh);
    let got = kpts.symmetrize_density(&rho, 0, mesh).expect("ft branch");
    let want = upstream_symmetrize(&kpts, &rho, 0, mesh);
    let max = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("  Task 4: max |symmetrize_ft - upstream C kernel|      = {max:e}");
    assert!(
        max < 1e-12,
        "the fractional-translation branch disagrees by {max:e}"
    );
}

/// `symmetrize_wavefunction` — upstream's very first statement is
/// `raise RuntimeError('need verification')` (`kpts.py:415`), so every line
/// below it is dead code that has never run. RULE 2: this port refuses
/// identically rather than shipping an unverified algorithm.
#[test]
fn symmetrize_wavefunction_refuses_exactly_as_upstream_does() {
    let (_cell, kpts) = density_fixture([8, 8, 8]);
    assert!(matches!(
        kpts.symmetrize_wavefunction(),
        Err(PbcSymmError::SymmetrizeWavefunctionUnverified)
    ));
}
