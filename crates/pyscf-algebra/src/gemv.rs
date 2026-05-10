//! GEMV — rank-1 case of GEMM. Phase 1 stub.
use crate::{AlgebraClient, AlgebraError, Tensor};

pub fn gemv(
    _client: &AlgebraClient,
    _a: &Tensor,
    _x: &Tensor,
    _y: &mut Tensor,
) -> Result<(), AlgebraError> {
    Err(AlgebraError::NotYetImplemented { phase: 2, what: "gemv (delegates to gemm)" })
}
