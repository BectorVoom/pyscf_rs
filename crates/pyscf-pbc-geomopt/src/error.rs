use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcGeomoptError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
}
