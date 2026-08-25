use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcScfError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
