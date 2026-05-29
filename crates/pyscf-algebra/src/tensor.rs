//! D-05: opaque Tensor / BufferId surface.
//!
//! quick-260522-b06: `Tensor` is now generic over its host element type
//! `T: Scalar`, with `f64` as the DEFAULT type parameter. Bare `Tensor` still
//! means `Tensor<f64>`, so every existing f64 call site (in algebra and in the
//! method crates) compiles unchanged and the normal build only monomorphizes
//! f64. f32 is exercised through one narrow smoke test (tests/f32_smoke.rs).
//!
//! The runtime `DType` tag is RETAINED (per the reconciliation constraint) but
//! is now DERIVED from `T` via `dtype_of::<T>()` rather than passed in, so the
//! compile-time element type and the runtime tag can never disagree.

use core::marker::PhantomData;

use pyscf_core::Scalar;
use pyscf_runtime::DType;

use crate::scalar::dtype_of;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(pub(crate) u64);

impl BufferId {
    /// Wrap a registry-assigned id. Used by the Phase-2 device-buffer registry
    /// (`crate::device_buffer`) when it `upload`s host data and hands back a
    /// `Tensor` referencing the stored buffer.
    pub(crate) fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// The sentinel id a [`Tensor::placeholder`] carries — never handed out by
    /// the registry, so a placeholder tensor always misses the buffer lookup
    /// and surfaces a clean `UnallocatedBuffer` error rather than aliasing real
    /// device data.
    pub(crate) const SENTINEL: u64 = u64::MAX;

    /// The raw registry key. Crate-private — the integer id is an
    /// algebra-internal detail, not part of the opaque method-crate surface.
    pub(crate) fn raw(&self) -> u64 {
        self.0
    }
}

/// Opaque device buffer handle, generic over its host element type.
///
/// `T` defaults to `f64` so bare `Tensor` is `Tensor<f64>`. The `dtype` field
/// is the runtime precision tag, always derived from `T` via the constructors
/// (it cannot be set independently). The `PhantomData<fn() -> T>` field carries
/// `T` at compile time while keeping `Tensor` `Send + Sync` regardless of `T`
/// (the `fn() -> T` form is variance-correct and imposes no auto-trait drag).
#[derive(Clone, Debug)]
pub struct Tensor<T: Scalar = f64> {
    pub id: BufferId,
    pub shape: Vec<usize>,
    pub dtype: DType,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar> Tensor<T> {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn elem_size(&self) -> usize {
        match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
    pub fn nbytes(&self) -> usize {
        self.numel().saturating_mul(self.elem_size())
    }

    /// Phase-1 placeholder constructor for integration tests. The id is
    /// the [`BufferId::SENTINEL`], never handed out by the device-buffer
    /// registry, so a placeholder is not backed by any uploaded data — the
    /// Phase-2 `Tensor` ops surface `UnallocatedBuffer` for it. Build a
    /// device-backed tensor with [`crate::device_buffer::upload`] instead.
    ///
    /// quick-260522-b06: the `dtype` is now derived from `T` via
    /// `dtype_of::<T>()`, so the old two-arg `placeholder(shape, dtype)` form
    /// is gone — callers pick precision through the type parameter
    /// (`Tensor::<f32>::placeholder(..)`), defaulting to f64.
    #[doc(hidden)]
    pub fn placeholder(shape: Vec<usize>) -> Self {
        Self {
            id: BufferId(BufferId::SENTINEL),
            shape,
            dtype: dtype_of::<T>(),
            _marker: PhantomData,
        }
    }

    /// Construct a tensor that references a real registry buffer. Crate-private:
    /// the only callers are the device-buffer registry's `upload` paths, which
    /// pair the returned `id` with the bytes they just stored. The `dtype` is
    /// derived from `T` (never set independently), matching `placeholder`.
    pub(crate) fn from_buffer(id: BufferId, shape: Vec<usize>) -> Self {
        Self {
            id,
            shape,
            dtype: dtype_of::<T>(),
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_tensor_reports_single_precision() {
        let t = Tensor::<f32>::placeholder(vec![2, 2]);
        assert_eq!(t.elem_size(), 4);
        assert_eq!(t.dtype, DType::F32);
    }

    #[test]
    fn f64_tensor_reports_double_precision() {
        let t = Tensor::<f64>::placeholder(vec![2, 2]);
        assert_eq!(t.elem_size(), 8);
        assert_eq!(t.dtype, DType::F64);
    }

    #[test]
    fn default_type_param_resolves_to_f64() {
        // Bare `Tensor` (no explicit type arg) resolves to `Tensor<f64>` via
        // the default type parameter — the property that keeps every existing
        // f64 call site compiling unchanged. The annotation here stands in for
        // the `&Tensor` parameter types the algebra primitives already use.
        let t: Tensor = Tensor::placeholder(vec![1]);
        assert_eq!(t.dtype, DType::F64);
    }
}
