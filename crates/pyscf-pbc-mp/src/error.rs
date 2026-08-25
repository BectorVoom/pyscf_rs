use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcMpError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
