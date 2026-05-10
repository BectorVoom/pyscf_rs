//! ALG-07: CPU-baseline backend-matrix smoke. GPU rows are Phase 8.

use pyscf_algebra::{axpy, gemm, reduce_sum, select_backend, AlgebraError, Tensor};
use pyscf_runtime::DType;

fn setup_tracing() { let _ = tracing_subscriber::fmt::try_init(); }

#[test]
fn select_default_returns_cpu() {
    setup_tracing();
    let saved = std::env::var("PYSCF_BACKEND").ok();
    unsafe { std::env::remove_var("PYSCF_BACKEND"); }
    let sel = select_backend().expect("CPU fallback must succeed");
    assert_eq!(sel.kind, pyscf_runtime::BackendKind::Cpu);
    if let Some(v) = saved { unsafe { std::env::set_var("PYSCF_BACKEND", v); } }
}

#[test]
fn primitive_signatures_callable_returning_notyetimplemented() {
    setup_tracing();
    let saved = std::env::var("PYSCF_BACKEND").ok();
    unsafe { std::env::remove_var("PYSCF_BACKEND"); }
    let sel = select_backend().expect("CPU selection");
    let lhs = Tensor::placeholder(vec![4, 4], DType::F64);
    let rhs = Tensor::placeholder(vec![4, 4], DType::F64);
    let mut out = Tensor::placeholder(vec![4, 4], DType::F64);
    // Phase 1 primitives return NotYetImplemented — that's the contract.
    // Phase 2 wires the actual cubecl-matmul launch.
    let r = gemm(&sel.client, &lhs, &rhs, &mut out);
    assert!(matches!(r, Err(AlgebraError::NotYetImplemented { .. })));

    let r = axpy(&sel.client, 1.0, &lhs, &mut out);
    assert!(matches!(r, Err(AlgebraError::NotYetImplemented { .. })));

    let r = reduce_sum(&sel.client, &lhs, 0, &mut out);
    assert!(matches!(r, Err(AlgebraError::NotYetImplemented { .. })));

    if let Some(v) = saved { unsafe { std::env::set_var("PYSCF_BACKEND", v); } }
}
