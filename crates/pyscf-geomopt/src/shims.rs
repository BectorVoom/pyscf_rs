//! GEOMOPT-02/03 + D-07: `geometric_solver` / `berny_solver` shim entry points.
//!
//! Upstream `pyscf/geomopt/geometric_solver.py` and `berny_solver.py` are thin
//! wrappers that dispatch a `method` (a `GradScanner` / `GradientsBase` /
//! anything with `nuc_grad_method()`) into an external optimizer
//! (`geometric` / `pyberny`). Per **D-06** there is exactly ONE native engine
//! in this project (`crate::optimize`, the geomeTRIC-algorithm port from
//! 07-04); the `berny_solver` shim is a thin ALIAS over that SAME engine — there
//! is no second optimizer (T-07-20 / RESEARCH anti-pattern).
//!
//! Per **D-07** the shims mirror the upstream call signatures + return shapes:
//!
//! ```text
//!   kernel(method, constraints=None, callback=None, maxsteps=100, **kwargs)
//!       -> (conv, mol)              # geometric_solver.py:96-167
//!   optimize(method, ...) -> mol    # == kernel(...)[1]  (geometric_solver.py:169-192)
//! ```
//!
//! The Rust shim entry points (this module) are **pyo3-free** — they take the
//! native `GradScanner` + the starting `Mole` + a [`ShimParams`] bundle and
//! return `(conv, Mole)` / `Mole`. The Python `pyscf.geomopt.*` entry points
//! (with the `method` dispatch + the no-runtime-dep proof, GEOMOPT-01) land in
//! the 07-09 PyO3 bridge, which wires the `method` → `GradScanner` adaptation
//! and calls these functions.
//!
//! ## Threat mitigations
//!
//! - **T-07-17 (constraints silently ignored):** a non-`None` `constraints`
//!   raises a clear [`GeomError::ConstraintsUnsupported`] (GEOMOPT-EXT-01
//!   deferred) — never a silent no-op that would mis-optimize a constrained
//!   input.
//! - **T-07-18 (unbounded maxsteps):** `maxsteps` defaults to 100 and is
//!   validated at the shim boundary via the native engine's `validate_maxsteps`
//!   (the DoS guard, T-07-10).
//! - **T-07-20 (a second berny optimizer drifting):** both shims delegate to
//!   the ONE [`crate::optimize`] engine; [`engine_name`](geometric_solver::engine_name)
//!   reports the single shared [`crate::NATIVE_ENGINE_NAME`] for both.

use crate::error::GeomError;
use crate::{ConvParams, GeometryOptimizer, optimize};
use pyscf_core::Mole;
use pyscf_grad::GradScanner;

/// A `Send + Sync` per-step callback (the upstream `callback` kwarg seam):
/// invoked after each accepted optimizer step with the step index + the current
/// geometry `(natm, 3)` (Bohr) + the current total energy. Pyo3-free — the
/// 07-09 bridge adapts a Python callable to this shape.
pub type ShimCallback = Box<dyn Fn(usize, &[[f64; 3]], f64) + Send + Sync>;

/// The upstream-mirroring shim kwargs (D-07): `conv_params`, `callback`,
/// `maxsteps`, `constraints`.
///
/// Mirrors `kernel(method, constraints=None, callback=None, maxsteps=100,
/// **kwargs)` — the `conv_params` dict maps to the optional [`ConvParams`]
/// override (defaulting to the locked GAU preset, GEOMOPT-04).
pub struct ShimParams {
    /// The convergence-threshold override (the upstream `conv_params` dict).
    /// `None` → the locked GAU preset (GEOMOPT-04).
    pub conv_params: Option<ConvParams>,
    /// The per-step callback (the upstream `callback` kwarg). `None` → no-op.
    pub callback: Option<ShimCallback>,
    /// Max optimizer steps (the upstream `maxsteps`, default 100). Validated at
    /// the shim boundary (T-07-18 / T-07-10).
    pub maxsteps: usize,
    /// The `constraints` kwarg (the upstream `constraints`). `Some(_)` raises a
    /// clear [`GeomError::ConstraintsUnsupported`] (GEOMOPT-EXT-01 deferred,
    /// T-07-17) — NEVER a silent no-op.
    pub constraints: Option<String>,
}

impl Clone for ShimParams {
    /// `ShimParams` is `Clone` for ergonomics, but the boxed `callback` closure
    /// cannot be cloned — a clone drops the callback (it is `None` afterwards).
    /// Tests that compare two shim runs supply fresh params per call.
    fn clone(&self) -> Self {
        Self {
            conv_params: self.conv_params,
            callback: None,
            maxsteps: self.maxsteps,
            constraints: self.constraints.clone(),
        }
    }
}

impl ShimParams {
    /// New shim params with the upstream defaults (`maxsteps=100`,
    /// `constraints=None`, the locked GAU `conv_params`).
    pub fn new() -> Self {
        Self {
            conv_params: None,
            callback: None,
            maxsteps: crate::converge::DEFAULT_MAXSTEPS,
            constraints: None,
        }
    }
}

// `ShimParams::default()` must mirror `ShimParams::new()` (maxsteps=100), NOT
// the derived all-zero default (maxsteps=0). We hand-implement Default.
impl Default for ShimParams {
    fn default() -> Self {
        Self::new()
    }
}

/// The SHARED shim core: build a [`GeometryOptimizer`] from the (D-07) kwargs,
/// reject `constraints` with a clear error (T-07-17), validate `maxsteps`
/// (T-07-18), then drive the ONE native [`optimize`] engine (D-06). Returns
/// `(conv, optimized_mol)`.
///
/// Both `geometric_solver::kernel` and `berny_solver::kernel` call THIS — the
/// single-engine invariant (T-07-20). The per-step callback is invoked once on
/// completion with the final geometry (the native engine does not yet expose a
/// per-step hook; the callback contract is preserved for the 07-09 bridge).
fn run_shim(
    scanner: &GradScanner,
    mol: &Mole,
    params: ShimParams,
) -> Result<(bool, Mole), GeomError> {
    // T-07-17: constraints are accepted-but-unsupported — a CLEAR error, never
    // a silent no-op. (GEOMOPT-EXT-01 deferred.)
    if params.constraints.is_some() {
        return Err(GeomError::ConstraintsUnsupported);
    }

    let opt = GeometryOptimizer {
        conv_params: params.conv_params.unwrap_or_else(ConvParams::gau),
        maxsteps: params.maxsteps,
        has_constraints: false,
    };

    // Drive the ONE native engine (D-06). `optimize` validates `maxsteps`
    // (T-07-18 / T-07-10) and returns the optimized geometry + convergence flag.
    let result = optimize(&opt, scanner, mol)?;

    // Reconstruct the optimized Mole at the converged geometry (the upstream
    // `engine.mol` — the optimized molecule the shim returns).
    let optimized = mol_at_geometry(mol, &result.coords)?;

    // Honour the callback contract (invoked once with the final state; the
    // 07-09 bridge re-routes this to per-step once the engine exposes a hook).
    if let Some(cb) = params.callback.as_ref() {
        cb(result.nsteps, &result.coords, result.e_tot);
    }

    Ok((result.converged, optimized))
}

/// Build a clone of `template` at the given Cartesian geometry (Bohr) via the
/// cache-safe `set_geom_` (GTO-10) — the optimized `Mole` the shim returns.
fn mol_at_geometry(template: &Mole, coords: &[[f64; 3]]) -> Result<Mole, GeomError> {
    let mut m = template.clone();
    let atom_str = template
        ._atom
        .iter()
        .zip(coords.iter())
        .map(|((s, _), c)| format!("{} {} {} {}", s, c[0], c[1], c[2]))
        .collect::<Vec<_>>()
        .join("; ");
    pyscf_gto::set_geom_(&mut m, &atom_str).map_err(GeomError::Scanner)?;
    Ok(m)
}

/// The `geometric_solver` shim (the geomeTRIC entry point, mirroring
/// `pyscf/geomopt/geometric_solver.py:96-192`).
pub mod geometric_solver {
    use super::*;

    /// Mirror `geometric_solver.kernel(method, constraints=None, callback=None,
    /// maxsteps=100, **kwargs) -> (conv, mol)`.
    ///
    /// Delegates to the ONE native engine ([`crate::optimize`], D-06).
    ///
    /// # Errors
    /// - [`GeomError::ConstraintsUnsupported`] if `params.constraints` is set
    ///   (T-07-17).
    /// - [`GeomError::InvalidMaxSteps`] if `params.maxsteps` is out of range
    ///   (T-07-18).
    /// - any error the native engine raises during the optimization.
    pub fn kernel(
        scanner: &GradScanner,
        mol: &Mole,
        params: ShimParams,
    ) -> Result<(bool, Mole), GeomError> {
        run_shim(scanner, mol, params)
    }

    /// Mirror `geometric_solver.optimize(method, ...) -> mol` (`== kernel(...)[1]`).
    ///
    /// # Errors
    /// Same as [`kernel`].
    pub fn optimize(
        scanner: &GradScanner,
        mol: &Mole,
        params: ShimParams,
    ) -> Result<Mole, GeomError> {
        kernel(scanner, mol, params).map(|(_conv, mol)| mol)
    }

    /// The name of the single native engine this shim delegates to (D-06 /
    /// T-07-20 single-engine marker).
    pub fn engine_name() -> &'static str {
        crate::NATIVE_ENGINE_NAME
    }
}

/// The `berny_solver` shim (the pyberny entry point, mirroring
/// `pyscf/geomopt/berny_solver.py`). Per **D-06** this is a thin ALIAS over the
/// SAME native engine as [`geometric_solver`] — there is NO second optimizer
/// implementation (T-07-20). Every entry point delegates straight to its
/// `geometric_solver` twin (and thus to the one [`crate::optimize`] engine).
pub mod berny_solver {
    use super::*;

    /// Mirror `berny_solver.kernel(method, ...) -> (conv, mol)` — a thin alias
    /// over [`geometric_solver::kernel`] (the ONE native engine, D-06).
    ///
    /// # Errors
    /// Same as [`geometric_solver::kernel`].
    pub fn kernel(
        scanner: &GradScanner,
        mol: &Mole,
        params: ShimParams,
    ) -> Result<(bool, Mole), GeomError> {
        // No second optimizer (T-07-20): delegate to the geometric_solver twin.
        geometric_solver::kernel(scanner, mol, params)
    }

    /// Mirror `berny_solver.optimize(method, ...) -> mol` — a thin alias over
    /// [`geometric_solver::optimize`] (the ONE native engine, D-06).
    ///
    /// # Errors
    /// Same as [`geometric_solver::optimize`].
    pub fn optimize(
        scanner: &GradScanner,
        mol: &Mole,
        params: ShimParams,
    ) -> Result<Mole, GeomError> {
        geometric_solver::optimize(scanner, mol, params)
    }

    /// The name of the single native engine this shim delegates to — IDENTICAL
    /// to [`geometric_solver::engine_name`] (the single-engine marker, T-07-20).
    pub fn engine_name() -> &'static str {
        crate::NATIVE_ENGINE_NAME
    }
}
