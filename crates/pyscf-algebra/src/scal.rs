//! SCAL — element-wise `x *= alpha` via #[cube]. Phase 1 stub.
use crate::{AlgebraClient, AlgebraError, Tensor};

pub fn scal(
    _client: &AlgebraClient,
    _alpha: f64,
    _x: &mut Tensor,
) -> Result<(), AlgebraError> {
    Err(AlgebraError::NotYetImplemented { phase: 2, what: "scal #[cube] kernel" })
}
