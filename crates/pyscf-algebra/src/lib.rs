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
#![warn(clippy::unwrap_used)] // FOUND-07

pub mod client;
pub mod error;
pub mod select;
pub mod tensor;
// quick-260522-b06: device-side precision bridge. The cubecl::Float bound
// (DeviceScalar) and the ScalarKind -> DType reconciliation (dtype_of) live
// here — inside the wall — never in pyscf-core or a method crate.
pub mod scalar;

pub mod axpy;
pub mod dot;
pub mod gemm;
pub mod gemv;
pub mod host_fallback;
pub mod oracle;
pub mod reduce;
pub mod scal;
pub mod transpose;
// Phase 3 plan 03-01 — Pulay DIIS B-matrix LU solve (RESEARCH Open Question 1).
pub mod solve_linear;
// Phase 3 plan 03-11 — slice-based generalized self-adjoint eigh for SCF.
// Bridges the algebra-wall (D-04) so pyscf-scf can call a flat-slice API
// without naming Tensor/AlgebraClient. Mirrors solve_linear's wrapper
// shape from plan 03-01.
pub mod eigh_gen;
// Phase 5 plan 05-09 — rank-revealing DF/RI 2-center metric fit (eigh route)
// for ill-conditioned (P|Q) auxiliary metrics that a plain Cholesky rejects.
pub mod df_metric;

pub use client::AlgebraClient;
pub use error::AlgebraError;
pub use pyscf_runtime::DType;
pub use scalar::DeviceScalar;
pub use select::{BackendSelection, select_backend};
pub use tensor::{BufferId, Tensor};

pub use axpy::axpy;
pub use df_metric::{DF_METRIC_LINEAR_DEP, df_metric_fit};
pub use dot::{dot, dot_dense};
pub use eigh_gen::eigh_gen;
pub use gemm::{gemm, gemm_dense};
pub use gemv::gemv;
pub use host_fallback::{cholesky, eigh, qr, svd};
pub use oracle::{oracle_dot, oracle_einsum, oracle_sum};
pub use reduce::{reduce_sum, reduce_sum_dense};
pub use scal::{scal, scal_dense};
pub use solve_linear::solve_linear;
pub use transpose::transpose;
