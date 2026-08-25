use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcAo2moError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
