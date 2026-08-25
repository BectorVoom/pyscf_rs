use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcEphError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
