//! Transpose — element-wise via #[cube]. Phase 1 stub.
use crate::{AlgebraClient, AlgebraError, Tensor};

pub fn transpose(
    _client: &AlgebraClient,
    _x: &Tensor,
    _out: &mut Tensor,
) -> Result<(), AlgebraError> {
    Err(AlgebraError::NotYetImplemented { phase: 2, what: "transpose #[cube] kernel" })
}
