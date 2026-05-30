---
phase: quick-260529-iji
plan: 01
subsystem: pyscf-algebra
tags: [cubecl, rocm, reduction, dot, algebra-wall]
requires: [oracle_dot, DeviceScalar, AlgebraClient, gemm.rs-precedent]
provides: [dot_dense, dot_kernel, launch_dot]
affects: [crates/pyscf-algebra/src/dot.rs, crates/pyscf-algebra/src/lib.rs, crates/pyscf-algebra/tests/dot_oracle.rs]
tech-stack:
  added: []
  patterns: [cube-launch-reduction, one-thread-per-element-products, host-f64-sum, backend-dispatch-off-AlgebraClient]
key-files:
  created:
    - crates/pyscf-algebra/tests/dot_oracle.rs
  modified:
    - crates/pyscf-algebra/src/dot.rs
    - crates/pyscf-algebra/src/lib.rs
decisions:
  - "dot device path computes per-element products in a #[cube(launch)] kernel and sums on the host in F (naive, no device atomics / cubecl-reduce) — mirrors gemm.rs philosophy"
  - "out_handle buffer sized via core::mem::size_of_val(x) to satisfy clippy::manual_slice_size_calculation (gemm.rs's m*n*size_of form did not trip it; the dot form does)"
metrics:
  duration: ~6 min
  completed: 2026-05-29
---

# Phase quick-260529-iji Plan 01: Refactor pyscf-algebra dot.rs to cubecl Summary

Refactored `pyscf-algebra/src/dot.rs` from a `NotYetImplemented` stub into a real generic-float cubecl `#[cube(launch)]` products kernel plus a backend-dispatched host launcher `dot_dense`, with a randomized oracle differential test passing on both CPU and ROCm (gfx1152). Mirrors the just-landed gemm.rs precedent exactly.

## What Was Built

- `dot_kernel<F: Float>` — `#[cube(launch)]` one-thread-per-element products kernel: `out[tid] = x[tid] * y[tid]` with a `tid < n` bounds guard. `n` is a bare `usize` scalar arg.
- `launch_dot<R: Runtime, F: DeviceScalar>` — uploads `x`/`y`, launches the kernel, reads back the products, and sums them on the host in `F` (plain `acc += p`, not an FMA). The `Runtime` generic `R` is confined to this fn.
- `dot_dense<F: DeviceScalar>(client, x, y) -> Result<F, AlgebraError>` — public device path; rejects mismatched lengths with `ShapeMismatch` before any upload, then dispatches `launch_dot::<Runtime, F>` off `AlgebraClient` (Cpu / Cuda / Wgpu / Rocm), so `Runtime` never appears in a public signature (ALG-06 wall intact).
- `dot()` over `Tensor` preserved as a Phase-2 `NotYetImplemented` stub (now with the descriptive `what` message pointing to `dot_dense`).
- `lib.rs` re-exports `dot_dense` alongside `dot`.
- `tests/dot_oracle.rs` — deterministic-LCG randomized differential test against `oracle_dot`; CPU test always runs, ROCm test under `#[cfg(feature = "rocm")]` asserts the `Rocm` variant (no silent fallback). Lengths cover degenerate 1, primes (13, 31, 97), and BLOCK=256-boundary straddlers (255, 256, 257, 512, 1000).

## Verification

- `cargo build -p pyscf-algebra` — succeeds (CPU default), exit 0. `log/dot-iji-t1-build.log`.
- `cargo clippy -p pyscf-algebra` — clean on dot.rs (no source warnings; only the pre-existing workspace `fma4`/cintx-patch noise). `log/dot-iji-t1-clippy.log`.
- `cargo test -p pyscf-algebra --test dot_oracle dot_kernel_matches_oracle_on_cpu` — pass. `log/dot-iji-t2-cpu.log`:

```
running 1 test
test dot_kernel_matches_oracle_on_cpu ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.26s
```

- `cargo test -p pyscf-algebra --features rocm --test dot_oracle` — pass on gfx1152 (cubecl-hip built, ~24s, no libxc). `log/dot-iji-t2-rocm.log`:

```
     Running tests/dot_oracle.rs (target/debug/deps/dot_oracle-4bc1d4ca8d843547)

running 2 tests
test dot_kernel_matches_oracle_on_cpu ... ok
test dot_kernel_matches_oracle_on_rocm ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.68s
```

- `dot()` over Tensor still returns `NotYetImplemented { phase: 2, .. }` (grep-verified).
- No command pulled libxc_rs into the dep graph — all scoped to `-p pyscf-algebra`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Lint] clippy::manual_slice_size_calculation on the out_handle allocation**
- **Found during:** Task 1 (clippy gate).
- **Issue:** The plan specified `client.empty(x.len() * core::mem::size_of::<F>())`, copying gemm.rs's `m * n * size_of` shape. Because `x.len()` is a direct slice `.len()`, clippy fired `manual_slice_size_calculation` (gemm.rs's `m * n` multiplier is not a bare `.len()`, so it did not trip there).
- **Fix:** Replaced with the equivalent `core::mem::size_of_val(x)` (identical byte count for a contiguous slice).
- **Files modified:** `crates/pyscf-algebra/src/dot.rs`
- **Commit:** `7ab843b`

## Threat Surface Scan

No new security-relevant surface beyond the plan's threat model. The host→device and device→host length-integrity boundaries (T-iji-01, T-iji-03) are handled exactly as in gemm.rs (lengths derived from `x.len()`/`size_of_val`, out_handle cloned before consumption). The DoS guard (T-iji-02) is the `ShapeMismatch` check in `dot_dense`. No new dependencies (T-iji-SC).

## Known Stubs

`dot()` over the opaque `Tensor` surface intentionally remains a Phase-2 `NotYetImplemented` stub — the device allocator that would back a `Tensor` read-back lands in Phase 2. This is the preserved Phase-2 contract, not an accidental stub; the working device path (`dot_dense`) is fully implemented and tested.

## Self-Check: PASSED

- Files: `crates/pyscf-algebra/src/dot.rs`, `crates/pyscf-algebra/src/lib.rs`, `crates/pyscf-algebra/tests/dot_oracle.rs` — all FOUND.
- Commits: `7ab843b` (feat), `6ee0aff` (test) — both FOUND.
