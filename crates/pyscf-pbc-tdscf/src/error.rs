use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcTdscfError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
