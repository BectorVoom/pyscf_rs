use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcDfError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),
    /// The FFT engine — `pyscf-pbc-tools` carries its own error type, and the
    /// periodic J/K builders are its heaviest caller.
    #[error(transparent)]
    Tools(#[from] pyscf_pbc_tools::PbcToolsError),
    /// A device-backend failure — selection or a kernel launch. Plan 13-01's
    /// `ft_aopair` is the first `pyscf-pbc-df` caller to reach the device
    /// directly rather than through `pyscf-pbc-gto`.
    #[error("device backend: {0}")]
    Backend(String),
}
