//! GEMM dispatch — `cubecl_matmul::launch::<R, T>(&Strategy::Auto, ...)`
//! per docs/manual/Cubecl/cubecl_matmul_gemm_example.md (RESEARCH §8).
//! Phase 1 ships the public signature; Phase 2 (GTO integral driver
//! consuming GEMM) implements the body.

use crate::{AlgebraClient, AlgebraError, Tensor};

/// Dense matrix multiply: `out = lhs @ rhs`. Shapes must satisfy
/// `lhs.shape == [M, K]`, `rhs.shape == [K, N]`, `out.shape == [M, N]`,
/// all dtypes equal.
pub fn gemm(
    _client: &AlgebraClient,
    _lhs: &Tensor,
    _rhs: &Tensor,
    _out: &mut Tensor,
) -> Result<(), AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 2,
        what: "gemm dispatch (cubecl_matmul::launch wiring lands with first GTO call site)",
    })
}
