---
phase: quick-260529-mtx
plan: 01
subsystem: pyscf-algebra (BLAS-1 element-wise kernels)
tags: [cubecl, axpy, blas-1, rocm, generic-float, oracle-test]
requires:
  - crates/pyscf-algebra/src/scalar.rs (DeviceScalar)
  - crates/pyscf-algebra/src/error.rs (AlgebraError::{DimensionMismatch, NotYetImplemented})
  - crates/pyscf-algebra/src/client.rs (AlgebraClient)
provides:
  - axpy_dense (host-slice device path for y[i] += alpha*x[i])
  - axpy_kernel (#[cube(launch)] generic-float kernel)
affects:
  - crates/pyscf-algebra/src/lib.rs (re-export surface)
tech-stack:
  added: []
  patterns:
    - cubecl 0.10 generic-float element-wise kernel (alpha as single-element Array<F>)
    - runtime-generic launcher behind AlgebraClient dispatch (ALG-06 wall)
    - randomized oracle differential test (CPU always + ROCm under #[cfg])
key-files:
  created:
    - crates/pyscf-algebra/tests/axpy_oracle.rs
  modified:
    - crates/pyscf-algebra/src/axpy.rs
    - crates/pyscf-algebra/src/lib.rs
decisions:
  - "alpha rides as a single-element Array<F> (DeviceScalar lacks CubeElement/ScalarArgSettings bounds for a bare scalar launch arg) — mirrors scal/dot/reduce siblings"
  - "y_handle cloned for read-back (y is the output buffer); x_handle consumed by from_raw_parts (read-only, not read back)"
  - "length-mismatch check runs FIRST (before empty no-op and before any launch) → DimensionMismatch, y untouched"
metrics:
  duration: ~6 min
  completed: 2026-05-29
---

# Phase quick-260529-mtx Plan 01: AXPY cubecl kernel Summary

Refactored the Phase-1 `NotYetImplemented` axpy stub into a real cubecl generic-float
`y[i] += alpha*x[i]` kernel plus a backend-dispatched host-slice launcher (`axpy_dense`),
mirroring the just-completed `scal` workstream exactly; added a randomized oracle differential
test that passes on CPU and on real gfx1152 ROCm hardware.

## What was built

**Task 1 — `crates/pyscf-algebra/src/axpy.rs` + `lib.rs` (commit `1d4bb52`)**
- `axpy_kernel<F: Float>` — `#[cube(launch)]` one-thread-per-element kernel: `x: &Array<F>`
  (read-only), `y: &mut Array<F>` (in place), `alpha: &Array<F>` (single-element), `n: usize`.
  Bounds-guarded `if tid < n` against the launch tail.
- `launch_axpy<R: Runtime, F: DeviceScalar>` — runtime-generic launcher; uploads x/y/alpha,
  launches, reads the cloned `y_handle` back. `Runtime` bound stays inside the wall.
- `axpy_dense<F: DeviceScalar>(client, alpha, x, y)` — public host-slice entry. Order of checks:
  length-mismatch → `DimensionMismatch { op: "axpy", .. }`; empty → `Ok(())` no-op; else dispatch
  `R` off `AlgebraClient` (Cpu always; Cuda/Wgpu/Rocm under `#[cfg]`), then `y.copy_from_slice`.
- `axpy()` over `Tensor` — retained as a documented Phase-2 `NotYetImplemented` stub.
- `lib.rs`: `pub use axpy::axpy;` → `pub use axpy::{axpy, axpy_dense};`.

**Task 2 — `crates/pyscf-algebra/tests/axpy_oracle.rs` (commit `4ec6700`)**
- `axpy_kernel_matches_oracle_on_cpu` (always) — host ground truth `y0[i] + alpha*x[i]`; device
  result within `1e-12` over LENS `[1,2,13,31,64,97,128,255,256,257,512,1000]` ×
  ALPHAS `[1.0,0.0,-1.0,2.5,-3.75,0.001,1234.5]`. Empty-input no-op asserted.
- `axpy_dense_length_mismatch_is_error` (CPU-only) — x len 4, y len 5 → `Err`.
- `axpy_kernel_matches_oracle_on_rocm` (`#[cfg(feature = "rocm")]`) — same differential check on
  `AmdDevice::default()` (gfx1152).
- Mirrors `scal_oracle.rs`: `Lcg` (Knuth/MMIX), `random_vector`, direct-client construction (no
  `select_backend` → no `PYSCF_BACKEND` race), fresh distinct seeds (CPU `0xA1FABEEF`, ROCm `0xD00DF00D`).

## Verification

- `cargo build -p pyscf-algebra` — Finished (6.95s). `WIRED` confirmed (axpy_dense + lib.rs re-export).
- `cargo test -p pyscf-algebra --test axpy_oracle` — `test result: ok. 2 passed` (CPU + length-mismatch).
- `cargo build -p pyscf-algebra --features rocm --tests` — Finished (11.28s), axpy_oracle compiles.
- GPU hardware present (gfx1152): `cargo test -p pyscf-algebra --features rocm --test axpy_oracle` —
  `test result: ok. 3 passed` including `axpy_kernel_matches_oracle_on_rocm`.
- No `cubecl::Runtime` in any public signature (axpy_dense/axpy take only `AlgebraClient` + slices/Tensor).
- All cargo output saved under `log/quick-260529-mtx-*.log` per project convention.

## Deviations from Plan

None — plan executed exactly as written. The optional length-mismatch test (suggested in the plan)
was included as `axpy_dense_length_mismatch_is_error` (CPU-only).

## Known Stubs

- `axpy()` over `Tensor` remains `NotYetImplemented { phase: 2 }` — intentional, per plan
  (Tensor has no device allocator until Phase 2). The working device path is `axpy_dense`.

## Self-Check: PASSED

- FOUND: crates/pyscf-algebra/src/axpy.rs
- FOUND: crates/pyscf-algebra/tests/axpy_oracle.rs
- FOUND: commit 1d4bb52 (Task 1)
- FOUND: commit 4ec6700 (Task 2)
