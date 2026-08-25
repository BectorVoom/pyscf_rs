use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcCcError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
