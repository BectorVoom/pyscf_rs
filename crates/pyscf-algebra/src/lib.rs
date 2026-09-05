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

// quick-260530-l29: the SINGLE cfg-gated AlgebraClient backend fanout. Declared
// with `#[macro_use]` BEFORE the engine modules so `dispatch_backend!` is in
// scope for axpy/dot/gemm/… without a `use`.
//
// quick-260530-ljv: the macro now ALSO carries `#[macro_export]`, hoisting it to
// the crate root (`pyscf_algebra::dispatch_backend`) so pyscf-kernels (a
// downstream wall-allowlisted crate) can fan a cube launch out over every
// backend. `#[macro_use]` is RETAINED alongside `#[macro_export]` (they coexist)
// so the 16 in-crate call sites keep resolving it unqualified — minimal diff. Do
// NOT add it to the `pub use` list below: `#[macro_export]` already publishes it
// at the crate root, and a `pub use self::dispatch::dispatch_backend;` would
// double-export and warn.
#[macro_use]
mod dispatch;

pub mod client;
pub mod error;
pub mod select;
pub mod tensor;
// quick-260522-b06: device-side precision bridge. The cubecl::Float bound
// (DeviceScalar) and the ScalarKind -> DType reconciliation (dtype_of) live
// here — inside the wall — never in pyscf-core or a method crate.
pub mod scalar;
// quick-260529-mtx (Phase-2 of the algebra crate): the device-buffer registry
// backing the opaque `Tensor`/`BufferId` surface. Turns the `placeholder`
// sentinel into real, uploadable buffers so the `Tensor`-based element-wise ops
// (axpy/scal/dot/reduce_sum) run end-to-end instead of returning
// `NotYetImplemented`. Stays inside the wall: cubecl never appears in its API.
pub mod device_buffer;

// quick-260826-spd: device-aware launch geometry + vectorization width, shared by
// every engine below.
//
// `pub` for the ALG-06 allowlist ONLY (pyscf-kernels, pyscf-runtime): its
// signatures name cubecl `Runtime`/`CubeCount`/`CubeDim`, so a method crate
// calling it would breach the wall — and cannot, since a method crate has no
// cubecl dependency to resolve those types with. It is published rather than
// duplicated because the CPU-runtime dispatch heuristics it encodes have to
// agree across every kernel in the workspace, not just the ones in this crate.
pub mod launch;

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
// ---------------------------------------------------------------------------
// v2.0 PBC milestone (Phase 9, plan 09-02) — the complex-algebra contract of
// PBC-MASTER-PLAN §5. Complex quantities are carried as the PLANAR `CTensor`
// (D-PBC-02 / RULE 8) so every complex operation reduces to the existing REAL
// cubecl primitives above and no new numeric type crosses the ALG-06 wall.
// ---------------------------------------------------------------------------
pub mod complex;
pub mod zblas;
pub mod zeigh;
pub mod zgemm;
pub mod zoracle;
// Phase 5 plan 05-09 — rank-revealing DF/RI 2-center metric fit (eigh route)
// for ill-conditioned (P|Q) auxiliary metrics that a plain Cholesky rejects.
pub mod df_metric;

pub use client::AlgebraClient;
pub use error::AlgebraError;
pub use pyscf_runtime::DType;
pub use scalar::DeviceScalar;
pub use select::{BackendSelection, select_backend};
pub use tensor::{BufferId, Tensor};

pub use axpy::{axpy, axpy_dense};
pub use complex::CTensor;
pub use device_buffer::{download, release, upload};
pub use df_metric::{DF_METRIC_LINEAR_DEP, df_metric_fit};
pub use dot::{dot, dot_dense};
pub use eigh_gen::eigh_gen;
pub use gemm::{gemm, gemm_dense};
pub use gemv::{gemv, gemv_dense};
pub use host_fallback::{cholesky, eigh, qr, svd};
pub use oracle::{oracle_dot, oracle_einsum, oracle_sum};
pub use reduce::{reduce_sum, reduce_sum_axis_dense, reduce_sum_dense, reduced_shape};
pub use scal::{scal, scal_dense};
pub use solve_linear::solve_linear;
pub use transpose::{transpose, transpose_dense};
pub use zblas::{
    zaxpy_dense, zdotc_dense, zdotu_dense, zhadamard_dense, zreduce_sum_dense, zscal_dense,
    ztranspose_dense,
};
pub use zeigh::{
    zcholesky, zcholesky_crout, zcholesky_faer, zeigh_gen, zeigh_gen_embedding, zeigh_gen_faer,
    zsolve_linear, zsolve_linear_embedding, zsolve_linear_faer,
};
pub use zgemm::{zgemm_dense, zgemm_h_dense};
pub use zoracle::{oracle_zdot, oracle_zdot_re, oracle_zdotu, oracle_zdotu_re, oracle_zsum};
