use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcAo2moError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
    #[error(transparent)]
    Df(#[from] pyscf_pbc_df::PbcDfError),
    #[error("shape mismatch: {0}")]
    Shape(String),
}
