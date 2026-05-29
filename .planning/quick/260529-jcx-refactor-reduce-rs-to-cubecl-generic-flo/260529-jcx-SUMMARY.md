---
phase: quick-260529-jcx
plan: 01
subsystem: pyscf-algebra
tags: [cubecl, rocm, reduction, reduce-sum, algebra-wall]
requires: [oracle_sum, DeviceScalar, AlgebraClient, dot.rs-precedent]
provides: [reduce_sum_dense, reduce_kernel, launch_reduce_sum]
affects:
  - crates/pyscf-algebra/src/reduce.rs
  - crates/pyscf-algebra/src/lib.rs
  - crates/pyscf-algebra/tests/reduce_oracle.rs
tech-stack:
  added: []
  patterns: [cube-launch-reduction, per-thread-partial-sum, host-f64-sum, backend-dispatch-off-AlgebraClient]
key-files:
  created:
    - crates/pyscf-algebra/tests/reduce_oracle.rs
  modified:
    - crates/pyscf-algebra/src/reduce.rs
    - crates/pyscf-algebra/src/lib.rs
decisions:
  - "reduce device path uses a PARTIAL-SUM kernel (one thread sequentially sums a CHUNK=256 contiguous slice into out[tid] via a `while` loop in F), then the host sums the `groups` partials in F — the while-loop inside #[cube] compiled cleanly on cubecl 0.10.0, so the dot.rs identity-pass fallback was NOT needed"
  - "empty input short-circuits to F::from_int(0) in launch_reduce_sum so a zero-length grid is never launched"
metrics:
  duration: ~5 min
  completed: 2026-05-29
---

# Phase quick-260529-jcx Plan 01: Refactor reduce.rs to cubecl generic-float reduction Summary

Refactored `pyscf-algebra/src/reduce.rs` from a `NotYetImplemented` stub into a real generic-float cubecl `#[cube(launch)]` partial-sum reduction kernel plus a backend-dispatched host launcher `reduce_sum_dense`, with a randomized oracle differential test passing on both CPU and ROCm (gfx1152). Mirrors the just-landed dot.rs precedent (quick-260529-iji), with the partial-sum kernel shape the plan preferred — no fallback required.

## What Was Built

- `reduce_kernel<F: Float>` — `#[cube(launch)]` partial-sum kernel. Thread `tid = ABSOLUTE_POS` sequentially accumulates input indices `[tid*chunk, (tid+1)*chunk)` (clamped to `n`) into one partial in `F` via a `while` loop, writing it to `out[tid]`. Tail threads (`start >= n`) write the identity `0`. `n` and `chunk` are bare `usize` scalar args.
- `launch_reduce_sum<R: Runtime, F: DeviceScalar>` — short-circuits empty input to `F::from_int(0)`; otherwise uploads `x`, allocates the `partials = n.div_ceil(CHUNK)` output buffer, launches the kernel, reads the partials back, and sums them on the host in `F` (plain `acc += p`, not an FMA). The `Runtime` generic `R` is confined to this fn.
- `reduce_sum_dense<F: DeviceScalar>(client, x) -> Result<F, AlgebraError>` — public device path; no shape validation needed (single input). Dispatches `launch_reduce_sum::<Runtime, F>` off `AlgebraClient` (Cpu / Cuda / Wgpu / Rocm), so `Runtime` never appears in a public signature (ALG-06 wall intact). Matches `oracle_sum`'s axis-free full-sum signature shape.
- `reduce_sum()` over `Tensor` preserved as a Phase-2 `NotYetImplemented { phase: 2, .. }` stub (now with a descriptive `what` message pointing to `reduce_sum_dense`).
- `lib.rs` re-exports `reduce_sum_dense` alongside `reduce_sum`.
- `tests/reduce_oracle.rs` — deterministic-LCG randomized differential test against `oracle_sum`; CPU test always runs, ROCm test under `#[cfg(feature = "rocm")]` asserts the `Rocm` variant (no silent fallback). Lengths cover empty (0 → identity), degenerate 1, primes (13, 31, 97), and CHUNK/BLOCK=256-boundary straddlers (255, 256, 257, 512, 1000).

## Kernel Shape Landed

The PREFERRED partial-sum kernel — not the fallback. A dynamic-bound `while i < end { acc += x[i]; i += 1; }` loop inside `#[cube]` compiled without issue on the pinned cubecl 0.10.0, so the dot.rs identity-pass (device elementwise, host sums everything) fallback was not needed. This pushes the bulk of the summation work onto the device (`groups` partials, each summing up to CHUNK=256 elements), leaving only the `partials`-length final reduction on the host.

## Verification

- `cargo build -p pyscf-algebra` — succeeds (CPU default), exit 0. `log/reduce-jcx-t1-build.log`.
- `cargo clippy -p pyscf-algebra` — clean on reduce.rs (no source warnings; only the pre-existing workspace `fma4`/cintx-patch noise). `log/reduce-jcx-t1-clippy.log`.
- `cargo test -p pyscf-algebra --test reduce_oracle reduce_kernel_matches_oracle_on_cpu` — PASS. `log/reduce-jcx-t2-cpu.log`:

```
running 1 test
test reduce_kernel_matches_oracle_on_cpu ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.25s
```

- `cargo test -p pyscf-algebra --features rocm --test reduce_oracle` — PASS on gfx1152 (cubecl-hip built, ~7s incremental, no libxc). `log/reduce-jcx-t3-rocm.log`:

```
     Running tests/reduce_oracle.rs (target/debug/deps/reduce_oracle-7d1e737659ea4dbc)

running 2 tests
test reduce_kernel_matches_oracle_on_cpu ... ok
test reduce_kernel_matches_oracle_on_rocm ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.62s
```

- `reduce_sum()` over Tensor still returns `NotYetImplemented { phase: 2, .. }` (source-verified).
- No command pulled libxc_rs into the dep graph — all scoped to `-p pyscf-algebra`.

## Deviations from Plan

None — plan executed exactly as written. The partial-sum kernel (the plan's preferred shape) compiled and matched the oracle on both backends on the first attempt, so the documented dot.rs fallback was not invoked.

## Threat Surface Scan

No new security-relevant surface beyond the dot.rs precedent. The host→device and device→host length-integrity boundaries are handled exactly as in dot.rs (lengths derived from `x.len()` / `n.div_ceil(CHUNK)`, `out_handle` cloned before consumption, tail threads bounds-guarded against out-of-range writes). The empty-input short-circuit prevents a zero-length grid launch. No new dependencies.

## Known Stubs

`reduce_sum()` over the opaque `Tensor` surface intentionally remains a Phase-2 `NotYetImplemented` stub — the device allocator that would back a `Tensor` read-back lands in Phase 2. This is the preserved Phase-2 contract, not an accidental stub; the working device path (`reduce_sum_dense`) is fully implemented and tested.

## Self-Check: PASSED

- Files: `crates/pyscf-algebra/src/reduce.rs`, `crates/pyscf-algebra/src/lib.rs`, `crates/pyscf-algebra/tests/reduce_oracle.rs` — all FOUND.
- Commits: `6497560` (feat), `be22fe8` (test) — both FOUND.
