//! Periodic second-order SCF (`pbc/scf/newton_ah.py`, 303 l).
//!
//! Ports the augmented-Hessian driver shape over 19-03's response seam:
//! `gen_g_hop_rhf`-style `(g, hop, h_diag)` triples feed a CIAH loop whose
//! micro-cycle solves the augmented-Hessian (AH) eigenproblem and whose macro
//! cycle rotates the orbitals. This module never rebuilds the seam — the
//! Hessian-vector product's `vind` part arrives as the same per-k
//! matrix-free closure 19-03 defines ([`crate::cphf::KFvind`]).
//!
//! The gate is the ENERGY, never the path (19-04): second-order SCF reaches
//! the same minimum by a different route, so iteration counts and
//! intermediate densities legitimately differ from first-order SCF and from
//! upstream. `tests/newton_ah.rs` converges the same analytic 2-level model
//! with both drivers and asserts the energies agree.
//!
//! Scale note: the AH subproblem is solved densely (exact CIAH on test-size
//! spaces — the subspace already spans the whole space). Production cells
//! replace the dense AH with Davidson micro-cycles over the same `hop`; the
//! driver loop is unchanged.

use pyscf_algebra::{eigh_gen, oracle_sum};
use pyscf_core::PyscfRsError;

/// Newton driver configuration.
#[derive(Debug, Clone)]
pub struct NewtonConfig {
    /// Gradient-norm convergence tolerance.
    pub conv_tol_grad: f64,
    /// Maximum macro (orbital-update) cycles.
    pub max_macro: usize,
    /// Maximum AH micro-cycles per macro step (dense path: informational —
    /// the full-space AH is solved exactly, so micros always "converge" in
    /// one shot; the field exists so the production Davidson path inherits
    /// the same budget type).
    pub max_micro: usize,
    /// Step-size cap on the orbital rotation (trust region).
    pub max_step: f64,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            conv_tol_grad: 1e-7,
            max_macro: 50,
            max_micro: 20,
            max_step: 0.5,
        }
    }
}

/// Gradient + Hessian-vector product + diagonal (`gen_g_hop_rhf` triple).
///
/// `g` is the flattened orbital gradient (vir-major per k, k-stacked);
/// `h_diag` its diagonal preconditioner; `hop` applies the orbital Hessian
/// (the `2·Fvv·x − 2·x·Foo + 2·C_virᵀ·vind(dm1[x])·C_occ` action of
/// `newton_ah.py:52-70`).
pub struct GHop<'a> {
    /// Flattened orbital gradient.
    pub g: Vec<f64>,
    /// Flattened Hessian diagonal (preconditioner).
    pub h_diag: Vec<f64>,
    /// Hessian-vector product.
    pub hop: Box<dyn Fn(&[f64]) -> Result<Vec<f64>, PyscfRsError> + 'a>,
}

/// One dense augmented-Hessian step: solve `[[0, gᵀ],[g, H]]·[1;dx] = e·[1;dx]`
/// for the lowest eigenpair and return the scaled rotation `dx`.
///
/// `hop`/`dim` define `H` (built column by column — test sizes only);
/// `level_shift` regularizes the AH (the CIAH keyframe shift). The step is
/// capped at `max_step` (trust region): over-long steps are rescaled, never
/// rejected silently — the rescale factor is returned alongside.
pub fn ah_step(
    g: &[f64],
    hop: &dyn Fn(&[f64]) -> Result<Vec<f64>, PyscfRsError>,
    dim: usize,
    level_shift: f64,
    max_step: f64,
) -> Result<(Vec<f64>, f64), PbscfNewtonError> {
    if g.len() != dim || dim == 0 {
        return Err(PbscfNewtonError::Shape { expected: dim, got: g.len() });
    }
    // Materialize H column by column (dense-AH path, test sizes).
    let mut h = vec![0.0f64; dim * dim];
    for j in 0..dim {
        let mut ej = vec![0.0f64; dim];
        ej[j] = 1.0;
        let col = hop(&ej).map_err(PbscfNewtonError::Hop)?;
        for i in 0..dim {
            h[i * dim + j] = col[i];
        }
    }
    // Symmetrize through the ordered mean (the analytic Hessian is symmetric;
    // an asymmetric hop is a caller bug, caught by the symmetry assert below).
    let mut asym = 0.0f64;
    for i in 0..dim {
        for j in 0..dim {
            asym = asym.max((h[i * dim + j] - h[j * dim + i]).abs());
        }
    }
    if asym > 1e-8 {
        return Err(PbscfNewtonError::AsymmetricHop { asym });
    }
    // Augmented Hessian [[shift, gᵀ],[g, H+shift·I]]... upstream shifts the
    // whole AH by the keyframe level shift; the step direction is unaffected
    // by a uniform shift, only the eigenvalue is.
    let n1 = dim + 1;
    let mut ah = vec![0.0f64; n1 * n1];
    ah[0] = level_shift;
    for i in 0..dim {
        ah[(i + 1) * n1] = g[i];
        ah[i + 1] = g[i];
        for j in 0..dim {
            ah[(i + 1) * n1 + j + 1] = 0.5 * (h[i * dim + j] + h[j * dim + i]);
        }
        ah[(i + 1) * n1 + i + 1] += level_shift;
    }
    let ident: Vec<f64> = {
        let mut s = vec![0.0f64; n1 * n1];
        for i in 0..n1 {
            s[i * n1 + i] = 1.0;
        }
        s
    };
    let (_e, vecs_f) = eigh_gen(&ah, &ident, n1).map_err(PbscfNewtonError::Algebra)?;
    // Lowest eigenvector, F-order column 0; de-homogenize by its 0th component.
    let v0 = vecs_f[0];
    if v0 == 0.0 || !v0.is_finite() {
        return Err(PbscfNewtonError::VanishingHomogeneous);
    }
    let mut dx: Vec<f64> = (0..dim).map(|i| vecs_f[i + 1] / v0).collect();
    // Trust region.
    let norm = oracle_sum(&dx.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt();
    let mut scale = 1.0;
    if norm > max_step {
        scale = max_step / norm;
        for x in dx.iter_mut() {
            *x *= scale;
        }
    }
    Ok((dx, scale))
}

/// Newton errors.
#[derive(Debug)]
pub enum PbscfNewtonError {
    /// Shape mismatch.
    Shape { expected: usize, got: usize },
    /// The hop closure failed.
    Hop(PyscfRsError),
    /// Hop matrix asymmetric beyond tolerance (caller bug).
    AsymmetricHop { asym: f64 },
    /// AH lowest eigenvector has vanishing homogeneous component.
    VanishingHomogeneous,
    /// Algebra failure.
    Algebra(pyscf_algebra::AlgebraError),
    /// Did not converge in `max_macro` macro cycles.
    NotConverged { cycles: usize, grad_norm: f64 },
}

impl std::fmt::Display for PbscfNewtonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PbscfNewtonError::Shape { expected, got } => {
                write!(f, "newton_ah: shape mismatch (expected {expected}, got {got})")
            }
            PbscfNewtonError::Hop(e) => write!(f, "newton_ah: hop failed: {e}"),
            PbscfNewtonError::AsymmetricHop { asym } => {
                write!(f, "newton_ah: hop asymmetric (max {asym:e})")
            }
            PbscfNewtonError::VanishingHomogeneous => {
                write!(f, "newton_ah: AH homogeneous component vanishes")
            }
            PbscfNewtonError::Algebra(e) => write!(f, "newton_ah algebra: {e}"),
            PbscfNewtonError::NotConverged { cycles, grad_norm } => {
                write!(f, "newton_ah: no convergence in {cycles} cycles (|g|={grad_norm:e})")
            }
        }
    }
}

impl From<PbscfNewtonError> for PyscfRsError {
    fn from(e: PbscfNewtonError) -> Self {
        PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(e.to_string()))
    }
}

/// Abstract orbital-optimization model the Newton driver converges.
///
/// Production use wires this to the k-point mean field (`g`/`hop` from
/// `gen_g_hop`-style Fock builds + the 19-03 response seam); tests wire it to
/// the analytic 2-level model. The driver only sees gradients and
/// Hessian-vector products — never integrals.
pub trait NewtonModel {
    /// Orbital-rotation dimension.
    fn dim(&self) -> usize;
    /// Current orbital gradient (flattened).
    fn gradient(&self) -> Vec<f64>;
    /// Hessian-vector product at the current orbitals.
    fn hop(&self, x: &[f64]) -> Result<Vec<f64>, PyscfRsError>;
    /// Apply an orbital rotation.
    fn apply_step(&mut self, dx: &[f64]);
    /// Current total energy.
    fn energy(&self) -> f64;
}

/// Second-order SCF driver: macro cycles of dense-AH steps to `conv_tol_grad`.
///
/// Returns the converged energy and the macro-cycle count. The count is
/// REPORTED, never gated (19-04: paths legitimately differ; only the energy
/// is asserted).
pub fn kernel_newton(
    model: &mut dyn NewtonModel,
    cfg: &NewtonConfig,
) -> Result<(f64, usize), PbscfNewtonError> {
    let dim = model.dim();
    for cycle in 0..cfg.max_macro {
        let g = model.gradient();
        if g.len() != dim {
            return Err(PbscfNewtonError::Shape { expected: dim, got: g.len() });
        }
        let grad_norm = oracle_sum(&g.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt();
        if grad_norm < cfg.conv_tol_grad {
            return Ok((model.energy(), cycle));
        }
        let (dx, _scale) = ah_step(&g, &|x| model.hop(x), dim, 0.0, cfg.max_step)?;
        model.apply_step(&dx);
    }
    let grad_norm = oracle_sum(
        &model.gradient().iter().map(|x| x * x).collect::<Vec<_>>(),
    )
    .sqrt();
    Err(PbscfNewtonError::NotConverged { cycles: cfg.max_macro, grad_norm })
}
