use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcLibError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
