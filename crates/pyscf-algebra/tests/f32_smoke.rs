//! The single, deliberate f32 instantiation point (quick-260522-b06).
//!
//! This is the ONLY place the precision generic is forced through the surface
//! at f32. Per the avoid-monomorphization-blowup constraint, the rest of the
//! workspace stays on the f64 default so the normal build only monomorphizes
//! f64; f32 is exercised here and nowhere else.
//!
//! TRADE-OFF (explicit): f32 is a speed/GPU path only. It CANNOT satisfy
//! bit-exact PySCF parity, so all oracle/regression tests intentionally stay on
//! the f64 default and are NOT retargeted to f32. This smoke test proves only
//! that the generic Tensor/Scalar path instantiates and reports the correct
//! precision metadata at f32.

use pyscf_algebra::{DType, Tensor};

#[test]
fn f32_tensor_instantiates_and_reports_single_precision() {
    let t = Tensor::<f32>::placeholder(vec![4, 4]);
    assert_eq!(t.dtype, DType::F32);
    assert_eq!(t.elem_size(), 4);
    assert_eq!(t.numel(), 16);
    assert_eq!(t.nbytes(), 64); // 16 elems * 4 bytes
}

#[test]
fn f64_tensor_reports_double_precision() {
    let t = Tensor::<f64>::placeholder(vec![4, 4]);
    assert_eq!(t.dtype, DType::F64);
    assert_eq!(t.elem_size(), 8);
    assert_eq!(t.nbytes(), 128); // 16 elems * 8 bytes
}

#[test]
fn bare_tensor_resolves_to_f64_default() {
    // No turbofish: bare `Tensor` resolves to the f64 default type parameter.
    let t: Tensor = Tensor::placeholder(vec![2]);
    assert_eq!(t.dtype, DType::F64);
}

#[test]
fn host_scalar_arithmetic_works_at_f32_and_f64() {
    // Proves the host-only `num_traits::Float` bound on `Scalar` is usable
    // generically at both precisions — the f32 host arithmetic path.
    fn sum2<T: pyscf_core::Scalar>(a: T, b: T) -> T {
        a + b
    }
    assert_eq!(sum2(1.5_f32, 2.5_f32), 4.0_f32);
    assert_eq!(sum2(1.5_f64, 2.5_f64), 4.0_f64);
}

#[test]
fn f32_scalar_kind_maps_to_f32_dtype() {
    // The ScalarKind -> DType reconciliation reaches DType::F32 from f32 via the
    // same path Tensor uses (Tensor::<f32> derives its dtype from the kind).
    let t = Tensor::<f32>::placeholder(vec![1]);
    assert_eq!(t.dtype, DType::F32);
    assert_eq!(
        <f32 as pyscf_core::Scalar>::KIND,
        pyscf_core::ScalarKind::F32
    );
}
