use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcDfError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
    /// The FFT engine — `pyscf-pbc-tools` carries its own error type, and the
    /// periodic J/K builders are its heaviest caller.
    #[error(transparent)]
    Tools(#[from] pyscf_pbc_tools::PbcToolsError),
}
