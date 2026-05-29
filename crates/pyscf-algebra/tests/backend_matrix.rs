//! ALG-07: CPU-baseline backend-matrix smoke. GPU rows are Phase 8.

use pyscf_algebra::{AlgebraError, Tensor, axpy, gemm, reduce_sum, select_backend};

fn setup_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}

#[test]
fn select_default_returns_cpu() {
    setup_tracing();
    let saved = std::env::var("PYSCF_BACKEND").ok();
    unsafe {
        std::env::remove_var("PYSCF_BACKEND");
    }
    let sel = select_backend().expect("CPU fallback must succeed");
    assert_eq!(sel.kind, pyscf_runtime::BackendKind::Cpu);
    if let Some(v) = saved {
        unsafe {
            std::env::set_var("PYSCF_BACKEND", v);
        }
    }
}

#[test]
fn primitive_signatures_callable_returning_notyetimplemented() {
    setup_tracing();
    let saved = std::env::var("PYSCF_BACKEND").ok();
    unsafe {
        std::env::remove_var("PYSCF_BACKEND");
    }
    let sel = select_backend().expect("CPU selection");
    // quick-260522-b06: placeholder now derives DType from the type param
    // (default f64), so the old two-arg form is gone.
    let lhs = Tensor::placeholder(vec![4, 4]);
    let rhs = Tensor::placeholder(vec![4, 4]);
    let mut out = Tensor::placeholder(vec![4, 4]);
    // gemm is STILL a Phase-1 stub — the contract is NotYetImplemented until the
    // cubecl-matmul Tensor launch is wired (quick-260529-mtx wired the
    // element-wise ops, not gemm).
    let r = gemm(&sel.client, &lhs, &rhs, &mut out);
    assert!(matches!(r, Err(AlgebraError::NotYetImplemented { .. })));

    // axpy is now wired through the device-buffer registry (quick-260529-mtx).
    // A `placeholder` carries the sentinel BufferId and was never uploaded, so
    // the Tensor path must reject it with UnallocatedBuffer (NOT silently
    // succeed and NOT NotYetImplemented). The real upload→op→download round-trip
    // lives in tests/tensor_registry.rs.
    let r = axpy(&sel.client, 1.0, &lhs, &mut out);
    assert!(matches!(r, Err(AlgebraError::UnallocatedBuffer { .. })));

    // reduce_sum is wired only for full reduction to a SCALAR `out`; a [4,4]
    // `out` (per-axis) is not yet implemented — that guard fires before any
    // registry lookup, so a placeholder out still yields NotYetImplemented here.
    let r = reduce_sum(&sel.client, &lhs, 0, &mut out);
    assert!(matches!(r, Err(AlgebraError::NotYetImplemented { .. })));

    if let Some(v) = saved {
        unsafe {
            std::env::set_var("PYSCF_BACKEND", v);
        }
    }
}
