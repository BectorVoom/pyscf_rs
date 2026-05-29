---
quick_id: 260529-i2x
slug: refactor-gemm-rs-to-cubecl-generic-float
date: 2026-05-29
status: complete
commit: b720570
---

# Quick Task 260529-i2x — Summary

## Outcome

Refactored `crates/pyscf-algebra/src/gemm.rs` from its Phase-1
`NotYetImplemented` stub into a real **cubecl `#[cube(launch)]` GEMM kernel
generic over the device float** (`F: Float`), and validated it with a
**randomized oracle differential test that runs on the ROCm backend**
(`cubecl_hip::HipRuntime`, device **gfx1152** / Radeon 860M).

**Both tests pass — CPU and real ROCm hardware:**
```
running 2 tests
test gemm_kernel_matches_oracle_on_cpu  ... ok
test gemm_kernel_matches_oracle_on_rocm ... ok
test result: ok. 2 passed; 0 failed
```

## What changed

- **`gemm.rs`**
  - `#[cube(launch)] fn gemm_kernel<F: Float>(lhs, rhs, out, m, k, n)` — naive
    one-thread-per-output-element kernel; `ABSOLUTE_POS`/dims are `usize`
    (cubecl 0.10 idiom); bounds-guarded tail; runtime K-loop.
  - `fn launch_gemm<R: Runtime, F: DeviceScalar>(...)` — private runtime-generic
    launcher (upload → launch `groups×256` → read back). Keeps the cubecl
    `Runtime` generic internal.
  - `pub fn gemm_dense<F: DeviceScalar>(client: &AlgebraClient, lhs, rhs, m, k, n)
    -> Result<Vec<F>>` — shape-validates, then dispatches off the resolved
    `AlgebraClient` arm (Cpu always; Cuda/Wgpu/Rocm cfg-gated), honoring the
    ALG-06 algebra wall (no cubecl type in the public signature).
  - `gemm()` (Tensor API) intentionally **stays a stub** — `Tensor` carries a
    sentinel `BufferId` (device allocator is Phase 2); `backend_matrix.rs` pins
    that contract. Doc updated to point at `gemm_dense`.
- **`scalar.rs`** — `DeviceScalar` gains `+ bytemuck::Pod` so launchers move
  `&[F]` host data through `Bytes::from_elems` / `cast_slice` without restating
  the bound. f32/f64 already satisfy it.
- **`lib.rs`** — re-export `gemm_dense`.
- **`tests/gemm_oracle.rs`** (new) — LCG-seeded random matrices over 7 shapes
  (square / tall / wide / prime / 1×1×1); device result vs `oracle_einsum
  ("ij,jk->ik")` ground truth; `max_abs_diff < 1e-9`. CPU test always runs; ROCm
  test (`#[cfg(feature = "rocm")]`) constructs the HIP client directly and
  asserts it is the ROCm backend (not a fallback).

## Verification

- `cargo clippy -p pyscf-algebra --all-targets` — clean.
- `cargo test -p pyscf-algebra` (cpu, default) — all suites green, no
  regressions (backend_matrix still sees `gemm()` = NotYetImplemented).
- `cargo test -p pyscf-algebra --features rocm --test gemm_oracle` — **green on
  gfx1152**. Logs under `log/gemm_oracle_rocm_*.log`.
- `libxc` stays 0 in the rocm dep graph (verified) — no 6h compile triggered.

## Notes / residual risk

- **Not verified:** f32 precision path through `gemm_dense` (oracle is f64-only);
  large shapes needing tiling (naive kernel is O(M·N·K) per thread, adequate for
  the correctness gate, not tuned for throughput).
- **Follow-up landmine:** wiring the opaque `Tensor`-based `gemm()` is blocked on
  the Phase-2 device allocator (`Tensor.id` is currently a sentinel).
- Executed inline (no worktree isolation) on the dirty `fix/ci-local-gates`
  branch, staging only the four source paths — per the known GSD
  worktree-leak caveat and to supervise the ROCm compile/run on live hardware.
