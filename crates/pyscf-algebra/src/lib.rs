//! pyscf-algebra: backend-agnostic linear-algebra surface for pyscf-rs.
//!
//! Phase 1 surface (D-06): gemm, gemv, axpy, scal, dot, reduce_sum,
//! transpose, oracle_sum, oracle_dot, oracle_einsum, eigh, cholesky,
//! qr, svd. Eigh family routes to faer 0.24 on host (ALG-05).
//!
//! ALG-06 dep-wall: this crate (alongside pyscf-runtime) is the ONLY
//! workspace crate permitted to declare cubecl-* dependencies. Method
//! crates consume Tensor only and never name a cubecl::* type
//! (D-04, D-05).
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]    // FOUND-07

pub mod client;
pub mod tensor;
pub mod error;
pub mod select;
// quick-260522-b06: device-side precision bridge. The cubecl::Float bound
// (DeviceScalar) and the ScalarKind -> DType reconciliation (dtype_of) live
// here — inside the wall — never in pyscf-core or a method crate.
pub mod scalar;

pub mod gemm;
pub mod gemv;
pub mod axpy;
pub mod scal;
pub mod transpose;
pub mod dot;
pub mod reduce;
pub mod oracle;
pub mod host_fallback;
// Phase 3 plan 03-01 — Pulay DIIS B-matrix LU solve (RESEARCH Open Question 1).
pub mod solve_linear;
// Phase 3 plan 03-11 — slice-based generalized self-adjoint eigh for SCF.
// Bridges the algebra-wall (D-04) so pyscf-scf can call a flat-slice API
// without naming Tensor/AlgebraClient. Mirrors solve_linear's wrapper
// shape from plan 03-01.
pub mod eigh_gen;

pub use client::AlgebraClient;
pub use error::AlgebraError;
pub use pyscf_runtime::DType;
pub use scalar::DeviceScalar;
pub use select::{select_backend, BackendSelection};
pub use tensor::{BufferId, Tensor};

pub use axpy::axpy;
pub use dot::dot;
pub use eigh_gen::eigh_gen;
pub use gemm::gemm;
pub use gemv::gemv;
pub use host_fallback::{cholesky, eigh, qr, svd};
pub use oracle::{oracle_dot, oracle_einsum, oracle_sum};
pub use reduce::reduce_sum;
pub use scal::scal;
pub use solve_linear::solve_linear;
pub use transpose::transpose;
