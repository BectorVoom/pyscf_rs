use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcMpError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
    #[error(transparent)]
    Df(#[from] pyscf_pbc_df::PbcDfError),
    #[error(
        "fractional occupation numbers encountered at k-point {kpt}; disable Krhf/Kuhf smearing before MP2"
    )]
    FractionalOccupation { kpt: usize },
    #[error("frozen orbital specification is invalid at k-point {kpt}: {reason}")]
    InvalidFrozen { kpt: usize, reason: String },
    #[error("frozen list has {got} k-points but the calculation has {expected}")]
    FrozenKpointCount { expected: usize, got: usize },
    #[error("frozen='auto'/'window' at k-points is not yet implemented (Phase 15)")]
    UnsupportedFrozen,
    #[error("shape mismatch: {what}")]
    Shape { what: String },
    #[error(
        "insufficient memory: KMP2 requires {required_mb:.1} MB but max_memory is {available_mb:.1} MB"
    )]
    Memory { required_mb: f64, available_mb: f64 },
    #[error("the SCF reference must be converged before KMP2")]
    UnconvergedReference,
    #[error("KUMP2 energy is not implemented upstream (pyscf/pbc/mp/kump2.py:38, :384, :402)")]
    Kump2NotImplemented,
    #[error("staggered KMP2 requires an even Monkhorst-Pack mesh, got {mesh:?}")]
    OddStaggerMesh { mesh: [usize; 3] },
}
