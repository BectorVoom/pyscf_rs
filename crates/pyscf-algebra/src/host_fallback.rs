//! Host-fallback linear algebra (ALG-05). All eigh/cholesky/qr/svd
//! route to faer 0.24 on host. On a GPU AlgebraClient, the
//! implementation copies down → faer → uploads back (Vec<f64>
//! round-trip per RESEARCH §9 + Pitfall 3 faer-ext incompat).
//!
//! Phase 1: signatures locked, bodies are NotYetImplemented stubs.
//! Phase 3 (SCF Fock-matrix diagonalization) is the first call site
//! and provides the actual wiring.

use crate::{AlgebraClient, AlgebraError, Tensor};

/// Self-adjoint eigendecomposition. Returns (eigenvalues, eigenvectors).
/// Phase 3 wires `faer::Mat::self_adjoint_eigen(Side::Lower)` per RESEARCH §9.
pub fn eigh(_client: &AlgebraClient, _matrix: &Tensor) -> Result<(Vec<f64>, Tensor), AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 3,
        what: "eigh — Phase 3 wires faer::Mat::self_adjoint_eigen with Vec<f64> round-trip",
    })
}

/// Cholesky LLT (positive-definite only). Phase 3 wires `faer::Mat::llt`.
pub fn cholesky(_client: &AlgebraClient, _matrix: &Tensor) -> Result<Tensor, AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 3,
        what: "cholesky — Phase 3 wires faer::Mat::llt",
    })
}

/// QR (no pivot). Phase 6 (CCSD intermediate canonicalization) wires.
pub fn qr(_client: &AlgebraClient, _matrix: &Tensor) -> Result<(Tensor, Tensor), AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 6,
        what: "qr — Phase 6 wires faer::Mat::qr",
    })
}

/// SVD (full). Phase 7 (gradient null-space projection) wires.
pub fn svd(
    _client: &AlgebraClient,
    _matrix: &Tensor,
) -> Result<(Tensor, Vec<f64>, Tensor), AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 7,
        what: "svd — Phase 7 wires faer::Mat::svd",
    })
}
