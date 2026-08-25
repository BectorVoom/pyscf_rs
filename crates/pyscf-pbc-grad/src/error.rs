use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcGradError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
