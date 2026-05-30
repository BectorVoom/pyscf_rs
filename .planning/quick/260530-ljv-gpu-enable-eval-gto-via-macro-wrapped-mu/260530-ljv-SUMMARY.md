---
phase: quick-260530-ljv
plan: 01
subsystem: gpu-enable / eval_gto
tags: [cubecl, eval_gto, s-shell, macro-export, differential-oracle, rocm]
requires:
  - pyscf_algebra::dispatch_backend (newly exported)
  - eval_gto_sph_cpu (unchanged host oracle target)
provides:
  - pyscf_algebra::dispatch_backend! exported #[macro_export] (cross-crate)
  - eval_gto_sph s-shell device path (#[cube(launch_unchecked)])
  - differential-oracle gate (cpu always-on + rocm gfx1152)
affects:
  - crates/pyscf-algebra (macro now crate-root-exported)
  - crates/pyscf-kernels (first real GPU compute path for eval_gto)
tech-stack:
  added: []
  patterns:
    - "downstream crates fan a cube launch over backends via the exported dispatch_backend!"
    - "f64-restricted #[cube] kernel for the f64-only chemistry path"
    - "per-backend Cargo feature forwards to pyscf-algebra so AlgebraClient::<Arm> exists"
key-files:
  created:
    - crates/pyscf-kernels/tests/eval_gto_oracle.rs
  modified:
    - crates/pyscf-algebra/src/dispatch.rs
    - crates/pyscf-algebra/src/lib.rs
    - crates/pyscf-kernels/src/eval_gto.rs
    - crates/pyscf-kernels/Cargo.toml
decisions:
  - "device arrays: coords/env as Array<f64>, bas/atm/ao_loc as Array<i32> (i32-array encoding for the integer libcint slots)"
  - "#[macro_use] RETAINED alongside #[macro_export] (minimal diff, 16 in-crate sites untouched)"
  - "kernel f64-restricted (NOT generic F: Float) — chemistry path is f64-only, sidesteps generic-Float .exp() risk"
metrics:
  duration: ~25 min
  completed: 2026-05-30
---

# Phase quick-260530-ljv Plan 01: GPU-enable eval_gto s-shell via macro-wrapped multi-backend cube kernel — Summary

One-liner: the l=0 (s-shell) eval_gto radial slice now runs a real
`#[cube(launch_unchecked)]` f64 kernel on the resolved backend — fanned out via
the newly `#[macro_export]`ed `dispatch_backend!` — and is byte/ULP-validated
against the host longhand on both CpuRuntime (diff = 0) and real AMD gfx1152
(diff = 1.11e-16).

## What shipped

- **Task 1** (`e095d2d`) — `#[macro_export]` on `dispatch_backend!` in
  pyscf-algebra, hoisting it to the crate root (`pyscf_algebra::dispatch_backend`)
  so pyscf-kernels can call it. `#[macro_use]` retained alongside; all 16
  in-crate call sites still build and `cargo test -p pyscf-algebra --lib` stays
  green (16 passed).
- **Task 2** (`7c97f54`) — `eval_gto_sph_kernel` (f64 `#[cube(launch_unchecked)]`,
  one thread per `(g, ao_idx)`, ordered primitive accumulation, F-order write)
  + `launch_eval_gto_s<R>` host launcher + routing in `eval_gto_sph`: pure-s-shell
  bases (all `ANG_OF==0`, non-empty, `ngrids*nao>0`) route to the device kernel
  via `dispatch_backend!`; ANY `l>=1` shell / empty grid / empty basis falls back
  to the UNCHANGED `eval_gto_sph_cpu`. All existing pyscf-kernels tests pass
  byte-for-byte (lib 2, eval_gto_lge1 4, wave0 smoke 1).
- **Task 3** (`12ea384`) — `tests/eval_gto_oracle.rs` (318 lines): randomized
  pure-s-shell fixtures vs an inline host oracle byte-matching
  `eval_gto_sph_cpu` lines 603-614. Always-on CpuRuntime arm + `#[cfg(rocm)]`
  gfx1152 arm. Plus a Rule-3 Cargo.toml feature-forward fix (see Deviations).

## Required output statements (per plan `<output>`)

1. **Device-array encoding chosen:** i32 arrays for the integer libcint slots.
   `coords` and `env` upload as `&Array<f64>`; `bas`, `atm`, `ao_loc` upload as
   `&Array<i32>` (cubecl 0.10 indexes i32 arrays fine). The libcint slot
   constants (`ATM_SLOTS`/`BAS_SLOTS`/`ATOM_OF`/`ANG_OF`/`NPRIM_OF`/`NCTR_OF`/
   `PTR_EXP`/`PTR_COEFF`/`PTR_COORD`) and `y00` ride in as bare scalar args.
2. **`#[macro_use]` retained or replaced:** RETAINED alongside `#[macro_export]`
   (they coexist). No per-module `use crate::dispatch_backend;` was needed —
   minimal diff, all 16 in-crate sites untouched.
3. **Generic-F vs f64-restricted kernel:** f64-RESTRICTED. The chemistry path is
   f64-only, and restricting sidesteps the generic-`Float` `.exp()` expansion
   risk flagged in the eval_gto.rs header. `y00` is a bare `f64` scalar arg.
4. **Observed max_abs_diff on the CpuRuntime oracle arm:** `0e0` (exactly zero —
   bit-identical: the CpuRuntime `exp()` matches std `f64::exp` term-for-term in
   the single-thread ordered accumulation). Well within TOL=1e-9.
5. **Was the rocm gfx1152 arm RUN or only compiled:** RUN. Executed
   `cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle
   eval_gto_s_matches_oracle_on_rocm` on real AMD gfx1152 hardware — PASSED with
   worst `max_abs_diff = 1.1102230246251565e-16` (~1 ULP, well within
   TOL=1e-9). This confirms the AMD device `exp()` differs from std `f64::exp`
   by <1 ULP/term, exactly as ORACLE-07 anticipates (documented tolerance, not
   bit-identical).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking config] Per-backend Cargo features must forward to pyscf-algebra**
- **Found during:** Task 3 (compile-checking the rocm arm).
- **Issue:** `pyscf-kernels`'s `rocm` feature enabled `dep:cubecl-hip` locally
  but did NOT forward `rocm` to the `pyscf-algebra` dependency (pulled with
  `default-features = false`). The exported `dispatch_backend!` — now invoked
  from `eval_gto.rs` for the first time — expands every cfg-enabled
  `AlgebraClient` arm, so `--features rocm` produced
  `error[E0599]: variant ... Rocm not found in AlgebraClient` (the variant is
  `#[cfg(feature="rocm")]`-gated inside pyscf-algebra, which was not enabled).
  Latent before this plan because pyscf-kernels never invoked the macro.
- **Fix:** `cpu/cuda/wgpu/rocm` features now forward to
  `pyscf-algebra/{same}` so the matching enum variant exists per backend.
- **Files modified:** `crates/pyscf-kernels/Cargo.toml`
- **Commit:** `12ea384`

(Tasks 1 and 2 executed exactly as written.)

## Self-Check: PASSED

- crates/pyscf-algebra/src/dispatch.rs — FOUND (`#[macro_export]` present)
- crates/pyscf-algebra/src/lib.rs — FOUND
- crates/pyscf-kernels/src/eval_gto.rs — FOUND (`#[cube(launch_unchecked)]` + `dispatch_backend!`)
- crates/pyscf-kernels/tests/eval_gto_oracle.rs — FOUND (318 lines, rocm arm present)
- crates/pyscf-kernels/Cargo.toml — FOUND (feature forwards)
- commits e095d2d, 7c97f54, 12ea384 — all FOUND in git log

Verify status: pyscf-algebra build + lib test GREEN; pyscf-kernels build GREEN
(no libxc); full pyscf-kernels default test suite GREEN; eval_gto_oracle CPU arm
GREEN (diff=0); rocm gfx1152 arm GREEN (diff=1.11e-16). All clippy clean.

## DEFERRED remainder (named, NOT executed — user's continue/stop decision)

1. **l>=1 cart→sph kernel** — port the cartesian-monomial + libcint c2s
   transform (`cart_powers`, `c2s_coeff`, `common_fac_sp`) into a cube kernel.
   Blocked-by: the "calling a normal Rust fn from inside #[cube]" pitfall — the
   c2s tables must become device arrays or `#[comptime]` data. Currently routes
   to the unchanged `eval_gto_sph_cpu` (covered by tests/eval_gto_lge1.rs).
2. **deriv1 / deriv2 stencils** — `eval_gto_sph_deriv1` (analytic gradient) and
   future deriv2 as cube kernels (still host-only + the `BackendKind` warn-guard).
3. **pyscf-dft numint cubecl backend** — wire the DFT grid-loop (numint) to call
   the device eval_gto path instead of the host loop.
