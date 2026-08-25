use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcGwError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
