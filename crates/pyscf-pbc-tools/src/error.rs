use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcToolsError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
