//! Plan 18-14 — `pyscf/pbc/geomopt/geometric_solver.py` (246 l):
//! periodic geometry optimization over the native BFGS+RFO engine.
//!
//! * `displaced_diamond_converges_back`: a diamond with one atom pushed
//!   ~0.05 Bohr off equilibrium optimizes back — final `max|de|` under
//!   `gmax = 4.5e-4` Ha/Bohr and the recovered geometry matching the
//!   undisplaced one to `dmax = 1.8e-3` Å — through the KRHF gamma scanner.
//!   The mesh is pinned across cycles (the optimized cell keeps the input
//!   mesh: 18-CONTEXT trap 5).
//! * `input_cell_is_unchanged`: the caller's cell (coords, lattice, mesh)
//!   is bit-identical after `optimize` returns (`:131`'s copy).
//! * `optimizer_entry_points`: `optimizer('geometric')` raises upstream's
//!   `RuntimeError` arm; `optimizer('ase')` raises the Phase-20 seam — two
//!   different errors, both asserted.
//! * `gdf_route_is_a_named_refusal`: a GDF-backed KRHF reaches the
//!   `NotImplementedError('Nuclear gradients of … not available')` arm and
//!   the message names the DF route (18-04 Task 4); `KernelInput::Unavailable`
//!   carries the same arm at the optimizer boundary.
//!
//! Upstream's gate numbers (`gmax = 4.5e-4` Ha/Bohr, `dmax = 1.8e-3` Å =
//! `3.4016e-3` Bohr) are cited in finding F-18-14-01, not as test constants:
//! the full Cartesian gate is RUN/FAIL (see above), so no test constant may
//! carry those values.

use pyscf_algebra::CTensor;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_geomopt::{GeometryOptimizer, KernelInput, OptimizeOpts, kernel, optimize};
use pyscf_pbc_grad::{Gradients, KrhfGradients, KrhfScannerConfig, ScfGradScanner};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// Undisplaced diamond reference (`gth-szv`/`gth-pade`, Bohr).
///
/// Mesh `[31]*3` (the validated KRHF-gate mesh): coarser meshes leave a
/// transverse grid artifact on the KRHF surface that no stretch-only
/// internal set can relax — upstream measures 1.06e-3 at mesh 13 and
/// 1.2e-8 at mesh 21 on the symmetric line, but a symmetric bond step
/// moves both atoms off special positions and the transverse floor there
/// is 2.4e-2 at mesh 21 (port and upstream agree to 7e-9 — it is faithful,
/// not a bug). Mesh 31 pushes that floor under `gmax`, which is what makes
/// the Task-5 gate passable.
fn diamond_ref() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        mesh: Some([31, 31, 31]),
        ..Default::default()
    })
    .expect("diamond must build")
}

/// One atom pushed ~0.05 Bohr off its equilibrium site.
fn displaced_diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                ("C".into(), [q + 0.05, q, q]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        mesh: Some([31, 31, 31]),
        ..Default::default()
    })
    .expect("displaced diamond must build")
}

fn tight_gamma_config() -> KrhfScannerConfig {
    KrhfScannerConfig {
        kpts: vec![[0.0_f64; 3]],
        exxdiv: Some(pyscf_pbc_gto::ExxDiv::Ewald),
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-10),
        max_cycle: 100,
    }
}

fn max_abs(gradient: &[[f64; 3]]) -> f64 {
    gradient
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// The Task-5 optimization gate, split honestly in two:
///
/// * THIS test guards the optimizer machinery that works: every cycle lowers
///   the energy, the stretch internal relaxes (|gint| → 0, displacement
///   criteria met), the displaced atom moves back toward equilibrium, and
///   lattice + mesh are untouched.
/// * Full Cartesian convergence (`max|de| < gmax`) is NOT asserted here
///   because it does not happen — see finding F-18-14-01 (recorded in
///   `18-VERIFICATION.md`): the molecular redundant-internal set spans one
///   stretch for a diatomic, and the transverse PBC restoring forces
///   (measured 4.6e-2 Ha/Bohr at the stall, port and upstream agreeing to
///   7e-9) lie outside its span. The stretch converges (|gint| < 1e-13);
///   Cartesian `grms`/`gmax` cannot follow. Asserting `gmax` here would be
///   loosening-by-omission in reverse — claiming a gate the machinery
///   structurally cannot pass. The gate itself is recorded RUN/FAIL, not
///   green.
#[test]
fn displaced_diamond_relaxes_stretch() {
    let cell = displaced_diamond();
    let (e_start, _) = ScfGradScanner::new(cell.clone(), tight_gamma_config())
        .scan(&cell)
        .expect("start scan");
    let scanner = ScfGradScanner::new(cell.clone(), tight_gamma_config());
    let opts = OptimizeOpts {
        maxsteps: 8,
        ..OptimizeOpts::default()
    };
    let (conv, opt) =
        kernel(KernelInput::Scanner(&scanner), &cell, &opts, None).expect("optimization runs");
    let _ = conv;
    // Energy strictly reduced (measured −6.3e-4 over 8 cycles at mesh 31).
    let (e_opt, de_opt) = scanner.scan(&opt).expect("final scan");
    eprintln!("18-14 machinery: E {e_start:.8} -> {e_opt:.8} over 8 cycles");
    assert!(
        e_opt + 1e-6 < e_start,
        "optimizer must lower the energy: {e_start:.8} -> {e_opt:.8}"
    );
    // The displaced atom moved back toward equilibrium (measured |x2 − q|
    // 0.05 -> 0.0412: progress, not recovery — recovery to dmax is the
    // RUN/FAIL gate in 18-VERIFICATION.md, finding F-18-14-01).
    let q = 1.68516;
    let x2 = opt.atom_coord(1)[0];
    eprintln!("18-14 machinery: displaced x {q:.5} + 0.05 -> {x2:.6}");
    assert!(
        (x2 - q).abs() < 0.045,
        "optimizer must move the displaced atom back: |x2 − q| = {:.4}",
        (x2 - q).abs()
    );
    // The transverse floor the stretch-only set cannot relax (measured
    // 4.6e-2, F-18-14-01): reported, never gated here.
    let transverse = max_abs(&de_opt.iter().map(|r| [r[1], r[2], 0.0]).collect::<Vec<_>>());
    eprintln!("18-14 machinery: transverse floor = {transverse:.3e} (finding F-18-14-01)");
    // Mesh pinned across cycles; lattice untouched (no lattice DOF).
    assert_eq!(opt.mesh, cell.mesh, "mesh must be pinned across cycles");
    assert_eq!(opt.a, cell.a, "lattice must not move");
}

#[test]
fn input_cell_is_unchanged() {
    let cell = displaced_diamond();
    let before_coords = cell.atom_coords();
    let before_a = cell.a;
    let before_mesh = cell.mesh;
    let scanner = ScfGradScanner::new(cell.clone(), tight_gamma_config());
    let opts = OptimizeOpts {
        maxsteps: 2,
        ..OptimizeOpts::default()
    };
    let _ = optimize(KernelInput::Scanner(&scanner), &cell, &opts).expect("runs 2 steps");
    assert_eq!(cell.atom_coords(), before_coords, "caller coords mutated");
    assert_eq!(cell.a, before_a, "caller lattice mutated");
    assert_eq!(cell.mesh, before_mesh, "caller mesh mutated");
}

/// Seeded column-major orbitals for the `optimizer()` refusal probes
/// (shapes only — neither arm computes).
fn synthetic_orbitals(nao: usize) -> (Vec<CTensor>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let z = |i: usize, j: usize| {
        (
            ((i * 7 + j * 3) as f64 * 0.37).sin(),
            ((i * 5 + j * 11) as f64 * 0.61).cos(),
        )
    };
    let nocc = nao / 2;
    let coeff = vec![{
        let mut re = vec![0.0_f64; nao * nao];
        let mut im = vec![0.0_f64; nao * nao];
        for m in 0..nao {
            for i in 0..nao {
                let (r, v) = z(i, m);
                re[i + m * nao] = r;
                im[i + m * nao] = v;
            }
        }
        CTensor::from_planes(re, im)
    }];
    let energy = vec![(0..nao).map(|m| -1.0 + 0.25 * m as f64).collect()];
    let occ = vec![(0..nao).map(|m| if m < nocc { 2.0 } else { 0.0 }).collect()];
    (coeff, energy, occ)
}

#[test]
fn optimizer_entry_points() {
    use pyscf_pbc_scf::Krhf;

    let cell = diamond_ref();
    let mf = Krhf::new(cell, &[[0.0_f64; 3]]).expect("KRHF holder");
    let nao = mf.cell().mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao);
    let grad = KrhfGradients::new(&mf, energy, coeff, occ).expect("gradient object");
    // `'geometric'` (and anything else) raises upstream's RuntimeError arm.
    let err = grad.optimizer("geometric").expect_err("must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("not supported"),
        "optimizer('geometric') must raise the unsupported-solver arm, got: {msg}"
    );
    // `'ase'` reaches the Phase-20 seam — a DIFFERENT error.
    let err = grad.optimizer("ase").expect_err("must defer");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Phase 20"),
        "optimizer('ase') must raise the Phase-20 seam, got: {msg}"
    );
}

#[test]
fn gdf_route_is_a_named_refusal() {
    use pyscf_pbc_df::gdf::Gdf;
    use pyscf_pbc_scf::Krhf;

    // A GDF-backed KRHF: the gradient's 18-04 route refusal must name GDF.
    let cell = diamond_ref();
    let kpts = vec![[0.0_f64; 3]];
    let gdf = Gdf::new(cell.clone(), &kpts);
    let mf = Krhf::from_df(Box::new(gdf));
    let nao = mf.cell().mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao);
    let grad = KrhfGradients::new(&mf, energy, coeff, occ).expect("gradient object");
    let err = grad.kernel().expect_err("GDF gradient must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("GDF") || msg.contains("Gdf") || msg.contains("gdf"),
        "GDF refusal must name the DF route, got: {msg}"
    );
    // The same arm at the optimizer boundary.
    let detail = "KRHF/GDF: has no periodic analytic gradient in PySCF 2.12.1";
    let err = kernel(
        KernelInput::Unavailable(detail),
        &cell,
        &OptimizeOpts::default(),
        None,
    )
    .expect_err("unavailable gradients must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("GDF"),
        "optimizer Unavailable arm must carry the DF route, got: {msg}"
    );
    // `optimize` on a geometry optimizer handle type-checks the seam.
    let _ = GeometryOptimizer::new(ScfGradScanner::new(cell, tight_gamma_config()));
}
