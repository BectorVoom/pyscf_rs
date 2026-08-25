use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcGtoError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
