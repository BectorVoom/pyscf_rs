use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcAdcError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
