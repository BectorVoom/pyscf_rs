use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcDfError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
