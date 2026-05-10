//! DOT — special-case reduction `sum(x * y)`. Phase 1 stub.
use crate::{AlgebraClient, AlgebraError, Tensor};

pub fn dot(
    _client: &AlgebraClient,
    _x: &Tensor,
    _y: &Tensor,
) -> Result<f64, AlgebraError> {
    Err(AlgebraError::NotYetImplemented { phase: 2, what: "dot reduction" })
}
