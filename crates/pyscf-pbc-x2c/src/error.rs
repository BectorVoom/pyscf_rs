use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcX2cError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
