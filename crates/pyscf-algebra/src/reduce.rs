//! Reduce-sum — `cubecl_reduce::reduce::<R, Sum>(...)` per
//! docs/manual/Cubecl/cubecl_reduce_sum.md. Phase 1 stub.
use crate::{AlgebraClient, AlgebraError, Tensor};

pub fn reduce_sum(
    _client: &AlgebraClient,
    _x: &Tensor,
    _axis: usize,
    _out: &mut Tensor,
) -> Result<(), AlgebraError> {
    Err(AlgebraError::NotYetImplemented {
        phase: 2,
        what: "reduce_sum cubecl-reduce dispatch",
    })
}
