use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcSymmError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
