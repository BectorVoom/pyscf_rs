use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcGeomoptError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),

    /// Mirrors `geometric_solver.py:kernel`'s `else` branch:
    /// `NotImplementedError('Nuclear gradients of %s not available')`. A
    /// GDF/MDF/RSDF/AFTDF-backed mean field reaches this arm through 18-04's
    /// named route refusal — the message names the DF route, never just the
    /// method.
    #[error("periodic nuclear gradients of {detail} are not available")]
    GradientsUnavailable { detail: String },

    /// `constraints` belong to geomeTRIC's input layer, which this port does
    /// not model — the native engine runs unconstrained (as does the
    /// molecular `has_constraints` refusal).
    #[error("periodic geometry optimization does not support constraints")]
    ConstraintsUnsupported,

    /// Mirrors `PySCFEngine.calc_new`'s `assert_convergence` arm:
    /// `RuntimeError('Nuclear gradients of %s not converged')`. The SCF
    /// underlying a cycle's gradient did not converge, so the step direction
    /// is meaningless.
    #[error("periodic SCF underlying the gradient did not converge at optimization cycle {cycle}")]
    ScfNotConverged { cycle: usize },

    /// The step budget was exhausted without meeting the convergence
    /// criteria — the molecular `OptimizeResult::converged = false` shape,
    /// surfaced here because `kernel` reports `(conv, cell)` separately.
    #[error("periodic geometry optimization did not converge in {maxsteps} steps")]
    NotConverged { maxsteps: usize },

    /// A non-positive or non-finite step budget.
    #[error("invalid maxsteps: {got}")]
    InvalidMaxSteps { got: usize },
}
