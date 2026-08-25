use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcDftError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
