//! Host-only precision generic (`Scalar`) for pyscf-rs (quick-260522-b06).
//!
//! `Scalar` is the cubecl-free leaf of the precision-generic stack. It bounds
//! only host-side float traits so it can live in `pyscf-core`, the crate every
//! other active crate depends on, WITHOUT dragging `cubecl` across the
//! algebra-wall (D-04 / ALG-06). The device-side `cubecl::Float` bound is added
//! separately inside `pyscf-algebra` / `pyscf-kernels` via `DeviceScalar`.
//!
//! T -> DType reconciliation: `pyscf-core` does NOT depend on `pyscf-runtime`
//! (it is a lower leaf), so it cannot name the runtime `DType` enum directly.
//! Instead, each `Scalar` carries a `KIND: ScalarKind` const; `pyscf-algebra`
//! (which depends on both core and runtime) maps `ScalarKind -> DType`.
//!
//! `Scalar` is sealed: only `f32` and `f64` implement it, and no downstream
//! crate can add impls.

use crate::scalar::sealed::Sealed;

/// Compile-time precision tag, mirrored by the runtime `pyscf_runtime::DType`
/// enum. Lives in `pyscf-core` (which has no runtime dep) so that `Scalar` can
/// expose a `T -> precision` mapping without naming `DType`. `pyscf-algebra`
/// bridges `ScalarKind -> DType`.
///
/// `#[non_exhaustive]` reserves room for future precisions (e.g. f16/bf16)
/// without a breaking change.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    /// Single precision (maps to `DType::F32`).
    F32,
    /// Double precision (maps to `DType::F64`).
    F64,
}

/// Host-only floating-point scalar usable as a tensor element type.
///
/// Supertrait bounds are intentionally host-side ONLY — there is NO
/// `cubecl::Float` bound here (that would violate the algebra-wall by pulling
/// cubecl into method crates). The device bound is added downstream by
/// `pyscf_algebra::DeviceScalar`.
///
/// Sealed: implemented for exactly `f32` and `f64`.
pub trait Scalar:
    num_traits::Float + Copy + Send + Sync + bytemuck::Pod + 'static + Sealed
{
    /// Compile-time precision tag for this scalar type. `pyscf-algebra` maps
    /// this to the runtime `DType` (the T -> DType reconciliation).
    const KIND: ScalarKind;

    /// Stable, lowercase precision name (`"f32"` / `"f64"`).
    const NAME: &'static str;
}

impl Scalar for f64 {
    const KIND: ScalarKind = ScalarKind::F64;
    const NAME: &'static str = "f64";
}

impl Scalar for f32 {
    const KIND: ScalarKind = ScalarKind::F32;
    const NAME: &'static str = "f32";
}

/// Sealing module: prevents downstream crates from implementing `Scalar`.
mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_kind_is_double() {
        assert_eq!(<f64 as Scalar>::KIND, ScalarKind::F64);
    }

    #[test]
    fn f32_kind_is_single() {
        assert_eq!(<f32 as Scalar>::KIND, ScalarKind::F32);
    }

    #[test]
    fn scalar_names() {
        assert_eq!(<f64 as Scalar>::NAME, "f64");
        assert_eq!(<f32 as Scalar>::NAME, "f32");
    }

    #[test]
    fn host_float_ops_usable_through_bound() {
        // Proves the `num_traits::Float` bound is callable generically — this
        // is the host arithmetic path the f32 smoke test (Task 6) exercises.
        fn sum2<T: Scalar>(a: T, b: T) -> T {
            a + b
        }
        assert_eq!(sum2(1.5_f32, 2.5_f32), 4.0_f32);
        assert_eq!(sum2(1.5_f64, 2.5_f64), 4.0_f64);
    }
}
