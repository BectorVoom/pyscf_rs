//! AXPY — element-wise `y += alpha * x` via #[cube] launch_unchecked
//! per docs/manual/Cubecl/Cubecl_multi_ compute.md.
use crate::{AlgebraClient, AlgebraError, Tensor};

pub fn axpy(
    _client: &AlgebraClient,
    _alpha: f64,
    _x: &Tensor,
    _y: &mut Tensor,
) -> Result<(), AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 2,
        what: "axpy #[cube] kernel",
    })
}
