use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcDftError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),

    #[error(
        "multigrid numerical integration requires exactly one gamma point, got {nkpts} k-points"
    )]
    MultiGridRequiresGamma { nkpts: usize },

    #[error("multigrid numerical integration requires the uniform FFT grid")]
    MultiGridRequiresUniformGrid,

    #[error("multigrid numerical integration does not support hybrid functional '{0}'")]
    MultiGridHybridUnsupported(String),

    #[error("multigrid numerical integration does not support a separate band k-point grid")]
    MultiGridBandUnsupported,
}
