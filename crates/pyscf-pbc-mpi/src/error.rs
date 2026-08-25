use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcMpiError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
