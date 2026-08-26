//! Device-side precision bridge (quick-260522-b06).
//!
//! `pyscf-algebra` is one of only three crates the algebra-wall (ALG-06)
//! permits to name `cubecl`. This module is therefore the place that:
//!
//!   1. concretely reconciles the host-side `Scalar::KIND` (a cubecl-free
//!      `pyscf_core::ScalarKind`) with the runtime `pyscf_runtime::DType` tag —
//!      it lives here because algebra is the lowest crate that depends on BOTH
//!      `pyscf-core` and `pyscf-runtime`; and
//!   2. adds the device-element bound `cubecl::Float` on top of the host-only
//!      `Scalar`, via the sealed `DeviceScalar` extension trait.
//!
//! The `cubecl::Float` bound appears ONLY here and (re-exported) in
//! `pyscf-kernels` — never in `pyscf-core` or any method crate.

use pyscf_core::{Scalar, ScalarKind};
use pyscf_runtime::DType;

/// Concrete `T -> DType` reconciliation: map the compile-time `Scalar::KIND`
/// to the runtime `DType` tag a `Tensor` carries. Crate-private — the runtime
/// tag is an algebra/runtime implementation detail, not part of the method-crate
/// surface.
pub(crate) fn dtype_of<T: Scalar>() -> DType {
    match T::KIND {
        ScalarKind::F32 => DType::F32,
        ScalarKind::F64 => DType::F64,
        // `ScalarKind` is `#[non_exhaustive]` (reserves f16/bf16 etc.), so a
        // wildcard is required cross-crate. Until a new precision lands here
        // AND in `DType`, fall back to the chemistry-safe F64 default.
        _ => DType::F64,
    }
}

/// Device-capable scalar: a host `Scalar` that is ALSO a cubecl device float.
///
/// This is the only place (besides the `pyscf-kernels` re-export) the
/// `cubecl::prelude::Float` bound is allowed to appear. Future f32 device
/// kernels bound their element type on `DeviceScalar` rather than on bare
/// `Scalar`, keeping the cubecl dependency inside the wall.
///
/// Sealed via the `pyscf_core::Scalar` supertrait (itself sealed), so only
/// `f32` and `f64` can implement it.
///
/// quick-260529-i2x: also bounds `bytemuck::Pod` so the device launchers in
/// this crate (e.g. `gemm_dense`) can move `&[F]` host data through
/// `cubecl::client::ComputeClient::create_from_slice` and read it back via
/// `bytemuck::cast_slice` without re-stating the byte-copy bound at every call
/// site. f32/f64 are Pod.
///
/// quick-260826-spd: and `cubecl::prelude::CubeElement`, which is what lets an
/// `F` be passed to a kernel as a bare SCALAR launch argument. Without it the
/// element-wise engines had to stage `alpha` in a one-element device buffer —
/// an allocation plus a host-to-device copy on every launch of an op that is
/// otherwise purely memory-bound. It also admits `F` as the element of a
/// `Vector<F, N>`, which is how those kernels now issue SIMD loads.
pub trait DeviceScalar:
    Scalar + cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod
{
}

impl DeviceScalar for f32 {}
impl DeviceScalar for f64 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_of_maps_kinds() {
        assert_eq!(dtype_of::<f32>(), DType::F32);
        assert_eq!(dtype_of::<f64>(), DType::F64);
    }
}
