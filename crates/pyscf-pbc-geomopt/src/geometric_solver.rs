//! Periodic geometry optimization — `pyscf/pbc/geomopt/geometric_solver.py`
//! (246 l) + `__init__.py` (23 l).
//!
//! The atom-coordinate optimizer over `pyscf-geomopt`'s native BFGS+RFO
//! engine. Upstream's `PySCFEngine.calc_new` (`:65-90`) is the engine
//! adapter — displace, run the scanner, return `{energy, gradient}` — and
//! everything above it (internal coordinates, the RFO step, convergence
//! testing) is coordinate-space work that does not know the system is
//! periodic. This module ports the adapter and the driver; the engine is
//! **reused** (`pyscf_geomopt::{internals, bmatrix, rfo, backtransform,
//! converge}`), never rewritten — `grep "fn rfo\|struct Rfo" here` finds
//! nothing, by construction.
//!
//! # What upstream has and what it does not (18-CONTEXT §1.7)
//!
//! * **No lattice degrees of freedom.** `grep -rn lattice
//!   pyscf/pbc/geomopt/*.py` hits once, inside the `__main__` example.
//!   The optimizer moves atoms (`:80-81`,
//!   `cell.set_geom_(coords, unit='Bohr')` then `g_scanner(cell)`) and
//!   nothing else: no `cell.a` update, no strain coordinate, no
//!   variable-cell relaxation anywhere in `pyscf/pbc`. Lattice optimization
//!   is a post-v2.0 feature whose *precondition* (an analytic `dE/dε`) plans
//!   18-12/18-13 deliver — it is not a port, and this module varies no
//!   lattice vector (a `grep "lattice\|strain\|cell.a"` over this file's
//!   non-doc lines finds nothing that does).
//! * **Two entry points, one of which cannot reach this module.**
//!   `pbc.geomopt.optimize(method)` (`__init__.py:18-23`) is the working
//!   path, ported as [`optimize`]. `mf.nuc_grad_method().optimizer(solver)`
//!   accepts ONLY `'ase'` and raises
//!   `RuntimeError('Optimization solver … not supported')` for anything
//!   else — including `'geometric'` — ported as the `Gradients::optimizer`
//!   refusal 18-02 declared (tested in `tests/optimize.rs`, both arms with
//!   distinct error types). The `'ase'` branch itself is Phase 20's
//!   `tools/pyscf_ase`, a `NotYetImplemented { phase: 20 }` seam there.
//!
//! # Upstream correspondence (`geometric_solver.py` line → here)
//!
//! | upstream | here |
//! |---|```
//! | `PySCFEngine.calc_new` `:65-90` (displace, scan, `{energy, gradient}`) | the scan call inside [`run_loop`] |
//! | `kernel` `:91-137` input shapes + `NotImplementedError` `:77-80` | [`kernel`] over [`KernelInput`] |
//! | `include_ghost = False` restricts `atmlst` `:123-124` | [`OptimizeOpts::include_ghost`] → frozen zero-charge atoms |
//! | `engine.cell = g_scanner.cell.copy()` `:131` | [`run_loop`] clones the input cell first; the caller's cell is never mutated |
//! | symmetry shift/rotate hazard `:132-137` | the mesh is pinned: [`with_coords`](pyscf_pbc_grad::with_coords) retains `cell.mesh` (18-CONTEXT trap 5) |
//! | `kernel` returns `(conv, cell)`, `optimize` returns `kernel(...)[1]` `:145-152` | [`kernel`] / [`optimize`] |
//! | convergence defaults `:161-167` | [`ConvParams::gau`](pyscf_geomopt::ConvParams) — confirmed identical, not duplicated |
//! | `GeometryOptimizer` `:156-176` | [`GeometryOptimizer`] |
//!
//! # Reductions (D-PBC-17)
//!
//! Every accumulation materialises then routes through `oracle_sum` /
//! `oracle_dot` — no bare `+=`.

use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::PyscfRsError;
use pyscf_geomopt::converge::{ConvParams, MAX_ALLOWED_MAXSTEPS, check_converged};
use pyscf_pbc_grad::{Gradient, ScfGradScanner};
use pyscf_pbc_gto::Cell;

use crate::error::PbcGeomoptError;

/// Options for [`kernel`] / [`optimize`] — upstream `kernel`'s keyword
/// surface (`geometric_solver.py:91-96`) minus the geomeTRIC-only layers
/// (`constraints` is a refusal, `callback`/`logIni`/`params` plumbing does
/// not exist outside geomeTRIC).
#[derive(Debug, Clone)]
pub struct OptimizeOpts {
    /// The 5 GAU convergence thresholds + trust radii. Default is
    /// [`ConvParams::gau`] — upstream's `:161-167` set, confirmed identical
    /// rather than duplicated.
    pub conv_params: ConvParams,
    /// Max optimizer steps (`maxsteps = 100`, upstream `:96`; capped at
    /// `MAX_ALLOWED_MAXSTEPS`, the molecular T-07-10 rule).
    pub maxsteps: usize,
    /// `True` (upstream default) optimizes every atom. `False` freezes
    /// zero-charge (ghost) atoms: their gradient rows are zeroed and their
    /// coordinates pinned after every step (`:123-124` restricts `atmlst`
    /// to `atom_charges() != 0`).
    pub include_ghost: bool,
    /// Mirror `assert_convergence` (upstream default `True`): a cycle whose
    /// SCF did not converge raises instead of stepping along a meaningless
    /// gradient (`calc_new`'s `RuntimeError('Nuclear gradients of %s not
    /// converged')`).
    pub assert_convergence: bool,
}

impl Default for OptimizeOpts {
    fn default() -> Self {
        Self {
            conv_params: ConvParams::gau(),
            maxsteps: 100,
            include_ghost: true,
            assert_convergence: true,
        }
    }
}

/// What [`kernel`] accepts — upstream's three input shapes
/// (`geometric_solver.py:113-122`).
///
/// * A [`ScfGradScanner`] is upstream's `GradScanner` arm (`:113-114`) —
//   18-05's seam, already bound to its cell and k-points.
/// * Anything else must arrive with its unavailability named: a GDF-backed
///   (or otherwise gradient-less) mean field reaches the `else` arm
///   (`NotImplementedError('Nuclear gradients of %s not available')`,
///   `:121-122`), and the detail names the DF route, not just the method
///   (18-04 Task 4 rides through here).
#[derive(Debug, Clone, Copy)]
pub enum KernelInput<'a> {
    /// A ready scanner (upstream's `isinstance(method, lib.GradScanner)`).
    Scanner(&'a ScfGradScanner),
    /// A method with no usable nuclear gradient — carries the
    /// `NotImplementedError` detail verbatim (e.g. `"KRHF/GDF: …"`).
    Unavailable(&'a str),
}

/// `kernel(method, …)` — `geometric_solver.py:91-137`.
///
/// Returns `(conv, cell)`: the convergence flag and the optimized cell.
/// The input `cell` is never mutated — the loop works on a clone
/// (`engine.cell = g_scanner.cell.copy()`, `:131`).
///
/// # Errors
/// * [`PbcGeomoptError::GradientsUnavailable`] — a [`KernelInput::Unavailable`]
///   arm, or any scanner failure whose message already names its cause.
/// * [`PbcGeomoptError::ConstraintsUnsupported`] — `constraints.is_some()`.
/// * [`PbcGeomoptError::ScfNotConverged`] — `assert_convergence` and a
///   non-converged cycle.
/// * [`PbcGeomoptError::InvalidMaxSteps`] — a zero or over-cap budget.
pub fn kernel(
    input: KernelInput<'_>,
    cell: &Cell,
    opts: &OptimizeOpts,
    constraints: Option<&str>,
) -> Result<(bool, Cell), PbcGeomoptError> {
    if constraints.is_some() {
        return Err(PbcGeomoptError::ConstraintsUnsupported);
    }
    if opts.maxsteps == 0 || opts.maxsteps > MAX_ALLOWED_MAXSTEPS {
        return Err(PbcGeomoptError::InvalidMaxSteps { got: opts.maxsteps });
    }
    let scanner = match input {
        KernelInput::Scanner(s) => s,
        KernelInput::Unavailable(detail) => {
            return Err(PbcGeomoptError::GradientsUnavailable {
                detail: detail.to_string(),
            });
        }
    };
    run_loop(scanner, cell, opts)
}

/// `optimize(method, …)` — `geometric_solver.py:138-152`.
///
/// `kernel(...)[1]`: the optimized cell. (Upstream spells it `[1]` at
/// `:152`; the tuple order is `(conv, cell)` at `:136`.)
///
/// # Errors
/// As [`kernel`].
pub fn optimize(
    input: KernelInput<'_>,
    cell: &Cell,
    opts: &OptimizeOpts,
) -> Result<Cell, PbcGeomoptError> {
    Ok(kernel(input, cell, opts, None)?.1)
}

/// `GeometryOptimizer(method)` — `geometric_solver.py:156-176`.
///
/// Owns the scanner handle (clones share its state, so resume-style reuse
/// sees the same guess chain), the convergence params and the step budget.
/// `kernel()` returns `(conv, cell)`; [`optimize`](Self::optimize) returns
/// the cell. Note upstream's pointed remark at `:168`: after `.kernel()`,
/// `method.mol` (here: the scanner's recorded cell) WILL have changed — but
/// the *caller's* cell never does.
#[derive(Debug, Clone)]
pub struct GeometryOptimizer {
    scanner: ScfGradScanner,
    /// Convergence thresholds + trust radii (upstream `.params`, `:161`).
    pub params: ConvParams,
    /// Step budget (upstream `.max_cycle = 100`, `:160`).
    pub max_cycle: usize,
    /// Ghost-atom policy (upstream `kernel`'s `include_ghost`, `:93`).
    pub include_ghost: bool,
    /// SCF-convergence assertion (upstream default `True`, `:92`).
    pub assert_convergence: bool,
}

impl GeometryOptimizer {
    /// Wrap a scanner with upstream's defaults (`max_cycle = 100`,
    /// GAU params, ghosts included, convergence asserted).
    pub fn new(scanner: ScfGradScanner) -> Self {
        Self {
            scanner,
            params: ConvParams::gau(),
            max_cycle: 100,
            include_ghost: true,
            assert_convergence: true,
        }
    }

    /// Run the optimization: `(converged, cell)`.
    ///
    /// # Errors
    /// As [`kernel`].
    pub fn kernel(&self, cell: &Cell) -> Result<(bool, Cell), PbcGeomoptError> {
        kernel(
            KernelInput::Scanner(&self.scanner),
            cell,
            &OptimizeOpts {
                conv_params: self.params,
                maxsteps: self.max_cycle,
                include_ghost: self.include_ghost,
                assert_convergence: self.assert_convergence,
            },
            None,
        )
    }

    /// Run the optimization: the optimized cell.
    ///
    /// # Errors
    /// As [`kernel`].
    pub fn optimize(&self, cell: &Cell) -> Result<Cell, PbcGeomoptError> {
        Ok(self.kernel(cell)?.1)
    }
}

/// Project a Cartesian gradient `(natm, 3)` to internal coordinates:
/// `g_int = G⁻ B g_cart`, through `oracle_dot` (the molecular
/// `gradient_to_internal`, same discipline).
fn gradient_to_internal(b: &[f64], ginv: &[f64], de: &Gradient, nint: usize) -> Vec<f64> {
    let natm = de.len();
    let ncart = 3 * natm;
    let mut g_cart = vec![0.0_f64; ncart];
    for a in 0..natm {
        g_cart[3 * a] = de[a][0];
        g_cart[3 * a + 1] = de[a][1];
        g_cart[3 * a + 2] = de[a][2];
    }
    let mut bg = vec![0.0_f64; nint];
    for i in 0..nint {
        bg[i] = oracle_dot(&b[i * ncart..(i + 1) * ncart], &g_cart);
    }
    let mut g_int = vec![0.0_f64; nint];
    for i in 0..nint {
        g_int[i] = oracle_dot(&ginv[i * nint..(i + 1) * nint], &bg);
    }
    g_int
}

fn rms_flat(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let sq: Vec<f64> = v.iter().map(|x| x * x).collect();
    (oracle_sum(&sq) / v.len() as f64).sqrt()
}

fn max_abs_flat(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(message.into()))
}

/// The geomeTRIC outer loop over a cell — `PySCFEngine` + `run_optimizer`
/// with the native engine inlined (it is the same loop the molecular
/// `run_loop` runs; the cell-typed scanner and [`with_coords`](pyscf_pbc_grad::with_coords)
/// are the only periodic parts).
///
/// * `work` starts as a clone of `cell` and is the only cell ever mutated
///   (`:131`); the mesh never changes because `with_coords` preserves it.
/// * Frozen atoms (`include_ghost = false` + zero charge) have their
///   gradient rows zeroed before the internal projection and their
///   coordinates re-pinned after the back-transform.
/// * Energy/gradient/displacement reductions all route through
///   `oracle_sum` / `oracle_dot`.
fn run_loop(
    scanner: &ScfGradScanner,
    cell: &Cell,
    opts: &OptimizeOpts,
) -> Result<(bool, Cell), PbcGeomoptError> {
    use pyscf_geomopt::{backtransform, bmatrix, internals, rfo};

    let natm = cell.natm;
    if natm == 0 {
        return Err(PbcGeomoptError::Core(invalid(
            "periodic geometry optimization needs at least one atom",
        )));
    }
    let charges = cell.atom_charges();
    if charges.len() != natm {
        return Err(PbcGeomoptError::Core(invalid(
            "periodic geometry optimization: charge count disagrees with natm",
        )));
    }
    // Ghost freeze mask (`:123-124`).
    let frozen: Vec<bool> = charges
        .iter()
        .map(|q| !opts.include_ghost && *q == 0)
        .collect();

    // The working cell — the caller's cell is never touched.
    let mut work = cell.clone();
    let mut coords = cell.atom_coords();
    let prims = internals::generate(&coords, &charges);
    let nint = prims.len();
    let mut hessian = rfo::BfgsHessian::identity(nint);
    let mut trust = opts.conv_params.trust;
    let mut prev: Option<(f64, Vec<f64>, Vec<f64>)> = None;
    let mut e_tot = 0.0_f64;
    let mut cycle = 0usize;

    for _step in 0..opts.maxsteps {
        cycle += 1;
        let (b, _g, ginv) = bmatrix::build(&prims, &coords).map_err(PbcGeomoptError::Core)?;

        // `calc_new`: run the scanner on the current geometry.
        let (energy, mut de) = scanner.scan(&work).map_err(PbcGeomoptError::Core)?;
        if opts.assert_convergence && scanner.converged() == Some(false) {
            return Err(PbcGeomoptError::ScfNotConverged { cycle });
        }
        if !energy.is_finite() || de.iter().flatten().any(|v| !v.is_finite()) {
            return Err(PbcGeomoptError::Core(invalid(format!(
                "periodic geometry optimization cycle {cycle}: non-finite scanner energy/gradient"
            ))));
        }
        e_tot = energy;
        // Freeze ghosts: their rows do not drive the step.
        for (ia, row) in de.iter_mut().enumerate() {
            if frozen[ia] {
                *row = [0.0; 3];
            }
        }

        let g_int = gradient_to_internal(&b, &ginv, &de, nint);
        let q = internals::values(&prims, &coords);
        if let Some((_, q_prev, g_prev)) = &prev {
            let dq: Vec<f64> = (0..nint).map(|i| q[i] - q_prev[i]).collect();
            let dg: Vec<f64> = (0..nint).map(|i| g_int[i] - g_prev[i]).collect();
            hessian.bfgs_update(&dq, &dg);
        }

        let g_cart_flat: Vec<f64> = de.iter().flat_map(|r| [r[0], r[1], r[2]]).collect();
        let (grad_rms, grad_max) = (rms_flat(&g_cart_flat), max_abs_flat(&g_cart_flat));

        let de_energy = prev.as_ref().map(|(ep, _, _)| e_tot - ep);
        let (dq, _predicted, new_trust) = rfo::rfo_step(
            &hessian,
            &g_int,
            trust,
            &opts.conv_params,
            de_energy,
            prev.as_ref(),
        );
        trust = new_trust;

        // Back-transform, then re-pin frozen atoms (the displacement gate
        // compares full Cartesian steps, so ghosts must not drift).
        let mut new_coords = backtransform::to_cartesian(&prims, &coords, &dq).map_err(|e| {
            PbcGeomoptError::Core(invalid(format!(
                "periodic geometry optimization cycle {cycle}: back-transform failed: {e}"
            )))
        })?;
        for (ia, frozen_ia) in frozen.iter().enumerate() {
            if *frozen_ia {
                new_coords[ia] = coords[ia];
            }
        }

        let mut disp_flat = vec![0.0_f64; 3 * natm];
        for a in 0..natm {
            disp_flat[3 * a] = new_coords[a][0] - coords[a][0];
            disp_flat[3 * a + 1] = new_coords[a][1] - coords[a][1];
            disp_flat[3 * a + 2] = new_coords[a][2] - coords[a][2];
        }
        let report = check_converged(
            &opts.conv_params,
            de_energy.map(f64::abs).unwrap_or(f64::INFINITY),
            grad_rms,
            grad_max,
            rms_flat(&disp_flat),
            max_abs_flat(&disp_flat),
        );
        if std::env::var("PBCGEOMOPT_DEBUG").is_ok() {
            let gnorm: f64 = g_int.iter().map(|x| x * x).sum::<f64>().sqrt();
            let dqnorm: f64 = dq.iter().map(|x| x * x).sum::<f64>().sqrt();
            let yz: f64 = de
                .iter()
                .flat_map(|r| [r[1].abs(), r[2].abs()])
                .fold(0.0_f64, f64::max);
            eprintln!(
                "cycle {cycle} E={e_tot:.8} dE={:.2e}({}) grms={:.2e}({}) gmax={:.2e}({}) drms={:.2e}({}) dmax={:.2e}({}) trust={trust:.4} x2={:.6} |gint|={gnorm:.3e} |dq|={dqnorm:.3e} nint={nint} maxyz={yz:.3e} de={de:.4?}",
                de_energy.unwrap_or(f64::INFINITY),
                report.energy_ok,
                grad_rms,
                report.grms_ok,
                grad_max,
                report.gmax_ok,
                rms_flat(&disp_flat),
                report.drms_ok,
                max_abs_flat(&disp_flat),
                report.dmax_ok,
                coords[1][0],
            );
        }
        if report.converged {
            return Ok((true, work));
        }

        // `cell.set_geom_(coords, unit='Bohr')` (`:80`): coordinates only —
        // the lattice, mesh and basis settings ride along untouched.
        work = pyscf_pbc_grad::with_coords(&work, &new_coords).map_err(PbcGeomoptError::Core)?;
        prev = Some((e_tot, q, g_int));
        coords = new_coords;
    }
    Ok((false, work))
}
