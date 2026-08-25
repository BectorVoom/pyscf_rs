use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcCiError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
