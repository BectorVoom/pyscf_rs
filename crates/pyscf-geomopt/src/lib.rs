//! pyscf-geomopt: native BFGS+RFO redundant-internal-coordinate geometry
//! optimizer.
//!
//! Phase 7 deliverable (GEOMOPT-04/06/07, D-08). This is the phase's single
//! biggest novelty: there is **no in-tree optimizer source** — upstream
//! `pyscf/geomopt` only shims external `geometric`/`pyberny`. The redundant-
//! internal engine, Wilson B-matrix, RFO/trust-radius step, neg-eigenvalue
//! handling, internal↔Cartesian back-transform, and the 5 GAU convergence
//! defaults are ported from **geomeTRIC**'s published algorithm (D-06).
//!
//! ## geomeTRIC license (Task 1 gating record, D-06 / RESEARCH A3)
//!
//! geomeTRIC (`github.com/leeping/geomeTRIC`) is distributed under the
//! **"BSD 3-clause (aka BSD 2.0) Non-AI License"**, Copyright 2016-2024
//! Regents of the University of California and the Authors (Lee-Ping Wang et
//! al.). Clauses 1-3 are the standard permissive BSD-3-Clause terms (source +
//! binary redistribution with modification permitted, with attribution and
//! the no-endorsement clause); clause 4 is an additional restriction
//! forbidding inclusion of the source/binary in machine-learning *training*
//! datasets. The redistribution terms are BSD-3-Clause-compatible, so porting
//! the **algorithm** (re-derived in Rust below — not a verbatim source copy)
//! is permitted. This deviates from RESEARCH A3's expectation of plain
//! BSD-3-Clause: the actual license carries the extra Non-AI clause 4. The
//! algorithm structure is re-derived here; geomeTRIC retains its copyright.
//!
//! ## Architecture
//!
//! The optimizer drives the RHF [`pyscf_grad::GradScanner`] seam (07-02/07-03):
//! a `Mole → (e_tot, de)` evaluator. The outer loop (per geomeTRIC `optimize`):
//!   1. build redundant internals (`internals`) from the current geometry,
//!   2. build the Wilson B-matrix + `G⁻` pseudo-inverse (`bmatrix`),
//!   3. evaluate `(e_tot, de)` via the `GradScanner`,
//!   4. project the Cartesian gradient to internals (`Bᵀ`),
//!   5. take an RFO step + trust-radius update + neg-eig shift (`rfo`),
//!   6. back-transform the internal step to Cartesians (`backtransform`),
//!   7. `set_geom_` the new geometry and check the 5-criterion GAU convergence
//!      (`converge`).
//!
//! All eigh/solve route through `pyscf_algebra` (the algebra wall). Every
//! reduction materialises then `oracle_sum`/`oracle_dot` — no bare `+=`. This
//! crate is strictly pyo3-free (D-08 — the bridge lands in `pyscf-py`, 07-09),
//! cubecl-free (denylist), and hdf5-owner-free (checkpoint reuses
//! `pyscf_chkfile::hdf5`, body lands 07-06).
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod backtransform;
pub mod bmatrix;
pub mod converge;
pub mod error;
pub mod internals;
pub mod rfo;

pub use backtransform::to_cartesian;
pub use bmatrix::{build as build_bmatrix, g_inverse, g_matrix, wilson_b};
pub use converge::{ConvParams, ConvReport, check_converged};
pub use error::GeomError;
pub use internals::{Primitive, generate as generate_internals};
pub use rfo::{BfgsHessian, rfo_step};

use converge::{DEFAULT_MAXSTEPS, MAX_ALLOWED_MAXSTEPS};
use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::Mole;
use pyscf_grad::GradScanner;

/// The result of an `optimize` run: the converged geometry + the convergence
/// flag + the step count + the final energy.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// The optimized Cartesian geometry `(natm, 3)` in Bohr.
    pub coords: Vec<[f64; 3]>,
    /// `true` if all 5 GAU criteria were satisfied; `false` if `maxsteps` hit.
    pub converged: bool,
    /// Number of optimizer steps taken.
    pub nsteps: usize,
    /// The final total energy (Hartree).
    pub e_tot: f64,
}

/// The native BFGS+RFO redundant-internal geometry optimizer (the geomeTRIC
/// `OptParams`/`Optimizer` analog).
///
/// Holds the convergence parameters + the optimizer-level defaults
/// (`maxsteps=100`, initial `trust=0.1`, `tmax=0.3` — all locked, GEOMOPT-04).
#[derive(Debug, Clone)]
pub struct GeometryOptimizer {
    /// The 5 GAU convergence thresholds (locked defaults).
    pub conv_params: ConvParams,
    /// Max optimizer steps (capped at the `optimize` entry, T-07-10).
    pub maxsteps: usize,
    /// `constraints` are not yet supported (D-07); `true` raises a clear error.
    pub has_constraints: bool,
}

impl Default for GeometryOptimizer {
    fn default() -> Self {
        Self {
            conv_params: ConvParams::gau(),
            maxsteps: DEFAULT_MAXSTEPS,
            has_constraints: false,
        }
    }
}

impl GeometryOptimizer {
    /// New optimizer with the locked GAU defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate/cap `maxsteps` (T-07-10: never an unbounded loop).
    fn validate_maxsteps(&self) -> Result<usize, GeomError> {
        if self.maxsteps == 0 || self.maxsteps > MAX_ALLOWED_MAXSTEPS {
            return Err(GeomError::InvalidMaxSteps { got: self.maxsteps });
        }
        Ok(self.maxsteps)
    }
}

/// Project a Cartesian gradient `(natm, 3)` to internal coordinates:
/// `g_int = G⁻ B g_cart`. Every reduction is oracle-ordered.
fn gradient_to_internal(b: &[f64], ginv: &[f64], de: &[[f64; 3]], nint: usize) -> Vec<f64> {
    let natm = de.len();
    let ncart = 3 * natm;
    // Flatten the Cartesian gradient.
    let mut g_cart = vec![0.0_f64; ncart];
    for a in 0..natm {
        g_cart[3 * a] = de[a][0];
        g_cart[3 * a + 1] = de[a][1];
        g_cart[3 * a + 2] = de[a][2];
    }
    // Bg = B · g_cart   (nint)
    let mut bg = vec![0.0_f64; nint];
    for i in 0..nint {
        let row = &b[i * ncart..(i + 1) * ncart];
        bg[i] = oracle_dot(row, &g_cart);
    }
    // g_int = G⁻ · Bg   (nint)
    let mut g_int = vec![0.0_f64; nint];
    for i in 0..nint {
        let row = &ginv[i * nint..(i + 1) * nint];
        g_int[i] = oracle_dot(row, &bg);
    }
    g_int
}

/// Build an `atom`-string from a `(natm, 3)` Bohr geometry for `set_geom_`,
/// reusing the original symbols. Coordinates are emitted in Bohr; the caller
/// must build the optimizer `Mole` with `unit = Bohr`.
fn coords_to_atom_string(symbols: &[String], coords: &[[f64; 3]]) -> String {
    symbols
        .iter()
        .zip(coords.iter())
        .map(|(s, c)| format!("{} {} {} {}", s, c[0], c[1], c[2]))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Drive the `GradScanner` from the starting geometry to a stationary point
/// (the geomeTRIC outer loop). Returns the optimized geometry + convergence
/// flag.
///
/// `mol` supplies the starting geometry, the nuclear charges (for the bonding
/// graph), and the per-atom symbols (for `set_geom_`). The `mol.unit` MUST be
/// `Bohr` so the emitted geometry strings round-trip (the engine operates
/// internally in Bohr — Pitfall 6).
///
/// # Errors
/// - [`GeomError::ConstraintsUnsupported`] if `has_constraints` (T-07-11).
/// - [`GeomError::InvalidMaxSteps`] if `maxsteps` is out of range (T-07-10).
/// - [`GeomError::Scanner`] if a `GradScanner` evaluation fails.
/// - [`GeomError::BacktransformDiverged`] on a pathological B-matrix (T-07-13).
pub fn optimize(
    opt: &GeometryOptimizer,
    scanner: &GradScanner,
    mol: &Mole,
) -> Result<OptimizeResult, GeomError> {
    if opt.has_constraints {
        return Err(GeomError::ConstraintsUnsupported);
    }
    let maxsteps = opt.validate_maxsteps()?;

    let charges = mol.atom_charges();
    let symbols: Vec<String> = mol._atom.iter().map(|(s, _)| s.clone()).collect();
    let mut coords = mol.atom_coords();
    let natm = coords.len();

    // Build the redundant-internal set ONCE from the starting connectivity
    // (geomeTRIC rebuilds only on a topology change; for a single
    // minimization the bonding graph is stable).
    let prims = internals::generate(&coords, &charges);
    let nint = prims.len();

    let mut hessian = rfo::BfgsHessian::identity(nint);
    let mut trust = opt.conv_params.trust;
    let mut prev: Option<(f64, Vec<f64>, Vec<f64>)> = None; // (E, q, g_int)

    // Working Mole we mutate via set_geom_.
    let mut work = mol.clone();
    let mut e_tot = 0.0_f64;

    for step in 0..maxsteps {
        // 1-2. internals + B-matrix + G⁻ at the current geometry.
        let (b, _g, ginv) = bmatrix::build(&prims, &coords)?;

        // 3. (e_tot, de) via the GradScanner.
        let (energy, de) = scanner.eval(&work, None).map_err(GeomError::Scanner)?;
        e_tot = energy.0;

        // 4. project Cartesian gradient to internals.
        let g_int = gradient_to_internal(&b, &ginv, &de, nint);
        let q = internals::values(&prims, &coords);

        // BFGS Hessian update from the previous step (skip on step 0).
        if let Some((_e_prev, q_prev, g_prev)) = &prev {
            let dq: Vec<f64> = (0..nint).map(|i| q[i] - q_prev[i]).collect();
            let dg: Vec<f64> = (0..nint).map(|i| g_int[i] - g_prev[i]).collect();
            hessian.bfgs_update(&dq, &dg);
        }

        // Convergence check needs the Cartesian gradient + the (eventual)
        // displacement; we check gradient criteria first, then displacement
        // after the step is computed.
        let g_cart_flat = flatten_grad(&de);
        let grad_rms = rms_flat(&g_cart_flat);
        let grad_max = max_abs_flat(&g_cart_flat);

        // 5. RFO step (internal displacement) + trust-radius update.
        let de_energy = prev.as_ref().map(|(ep, _, _)| e_tot - ep);
        let (dq, predicted_de, new_trust) =
            rfo::rfo_step(&hessian, &g_int, trust, &opt.conv_params, de_energy, prev.as_ref());
        trust = new_trust;

        // 6. back-transform to Cartesians.
        let new_coords = backtransform::to_cartesian(&prims, &coords, &dq)?;

        // displacement criteria (Cartesian, Bohr).
        let mut disp_flat = vec![0.0_f64; 3 * natm];
        for a in 0..natm {
            disp_flat[3 * a] = new_coords[a][0] - coords[a][0];
            disp_flat[3 * a + 1] = new_coords[a][1] - coords[a][1];
            disp_flat[3 * a + 2] = new_coords[a][2] - coords[a][2];
        }
        let disp_rms = rms_flat(&disp_flat);
        let disp_max = max_abs_flat(&disp_flat);
        let e_change = de_energy.map(f64::abs).unwrap_or(f64::INFINITY);

        let report = converge::check_converged(
            &opt.conv_params,
            e_change,
            grad_rms,
            grad_max,
            disp_rms,
            disp_max,
        );

        if report.converged {
            return Ok(OptimizeResult {
                coords,
                converged: true,
                nsteps: step + 1,
                e_tot,
            });
        }

        // 7. commit the new geometry into the working Mole.
        let atom_str = coords_to_atom_string(&symbols, &new_coords);
        pyscf_gto_set_geom(&mut work, &atom_str)?;

        prev = Some((e_tot, q, g_int));
        coords = new_coords;
        let _ = predicted_de; // retained for trust bookkeeping inside rfo_step
    }

    // maxsteps exhausted without convergence — return the last geometry with
    // converged=false (the caller decides whether to error).
    Ok(OptimizeResult {
        coords,
        converged: false,
        nsteps: maxsteps,
        e_tot,
    })
}

/// Apply `set_geom_` to the working Mole (Bohr geometry string).
fn pyscf_gto_set_geom(work: &mut Mole, atom_str: &str) -> Result<(), GeomError> {
    pyscf_gto::set_geom_(work, atom_str).map_err(GeomError::Scanner)?;
    Ok(())
}

fn flatten_grad(de: &[[f64; 3]]) -> Vec<f64> {
    let mut v = Vec::with_capacity(3 * de.len());
    for row in de {
        v.push(row[0]);
        v.push(row[1]);
        v.push(row[2]);
    }
    v
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
