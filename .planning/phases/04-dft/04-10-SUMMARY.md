---
phase: 04-dft
plan: 10
subsystem: infra
tags: [wgpu, shader-f64, precision, d-08, dft-11, ci, libxc, fallback]

# Dependency graph
requires:
  - phase: 04-dft
    provides: "04-05 XcBackend::eval seam; 04-06 NumInt D-08 f32 seam + below-bit-exact warn + dtype_f32_smoke"
  - phase: 01-foundation
    provides: "PYSCF_BACKEND resolver (unrecognised→CPU+warn); DType::from_env (PYSCF_DTYPE escape hatch)"
provides:
  - "WGPU f64 honesty (DFT-11): default-f64 + shader-f64-less wgpu → CPU-f64 + tracing::warn!, NEVER silent f32 — by delegating the probe to xcfun_gpu::auto_backend/must_fall_back_to_cpu + reusing the Phase-1 PYSCF_BACKEND fallback"
  - "D-08 reconciliation: PYSCF_DTYPE=f32 is the explicit, warned, honest opt-in escape hatch (distinct from silent degradation), documented in docs/env-vars.md"
  - "libxc CI surface wired but DISABLED pending 04-02: dft-libxc-bitexact job (if:false) is the only --features libxc job; nightly -p libxc_rs stays excluded; wgpu-no-f64-fallback job active"
  - "libxc [patch.crates-io] re-enabled but PROVEN inert (default build = 0 libxc)"
affects: [05-mp2, 08-gpu-oracle-dist, libxc_rs]

# Tech tracking
tech-stack:
  added:
    - "xcfun-gpu as a direct (default-features=false) dep of pyscf-dft — only to name auto_backend/must_fall_back_to_cpu (already transitive via xcfun-rs; NOT a cubecl-* crate, dependency-wall still passes)"
  patterns:
    - "Delegate the shader-f64/ERF capability probe to xcfun_rs rather than re-implementing it; pipeline fallback reuses the Phase-1 PYSCF_BACKEND model"
    - "libxc CI surface ships DISABLED (if:false) while PENDING_LIBXC_RS_FEATURE_GATE — the only place the gated ~6h compile would run is kept off until 04-02 lands"

key-files:
  created:
    - crates/pyscf-dft/tests/wgpu_f64_fallback.rs
    - .planning/phases/04-dft/deferred-items.md
  modified:
    - crates/pyscf-dft/src/xc_backend.rs
    - crates/pyscf-dft/Cargo.toml
    - docs/env-vars.md
    - Cargo.toml
    - .github/workflows/ci.yml
    - .github/workflows/nightly-cross-crate.yml

key-decisions:
  - "User checkpoint approval (2026-05-22): accept WGPU honesty fallback + the inert libxc patch + the DISABLED libxc CI job. The libxc CI surface stays if:false / excluded until 04-02 (PENDING_LIBXC_RS_FEATURE_GATE) lands — activating it without per-functional features would compile all 266 kernels (~6h)."
  - "Default-f64 + shader-f64-less wgpu → CPU-f64 + warn (never silent f32); explicit PYSCF_DTYPE=f32 is the honest user opt-in (warned by 04-06 NumInt, not blocked here)."
  - "xcfun-gpu added as direct dep only to delegate the probe — it is not cubecl-*, so the ALG-06 dependency-wall lint still passes."

patterns-established:
  - "Honest precision fallback: delegate capability probe to the sibling crate that owns it; never silently change the active scalar."

requirements-completed: [DFT-11]
# DFT-03 was completed by 04-05 (xcfun bit-exact); 04-10 adds its libxc bit-exact CI job in a DISABLED
# (if:false) state pending 04-02. DFT-11's on-real-shader-f64-less-DEVICE leg is the wgpu-no-f64-fallback
# CI job (special-runner / Phase 8); the fallback-decision logic + CPU-fallback unit test are complete here.

# Metrics
duration: 11min
completed: 2026-05-22
---

# Phase 04: dft — Plan 10 Summary

**WGPU f64 honesty (DFT-11): default-f64 on a shader-f64-less wgpu adapter falls back to CPU-f64 with a `tracing::warn!` (never silent f32) by delegating the probe to `xcfun_gpu` + the Phase-1 `PYSCF_BACKEND` resolver; `PYSCF_DTYPE=f32` is the documented honest escape hatch; the libxc CI surface is wired but DISABLED pending 04-02.**

## Performance

- **Duration:** ~11 min (2 autonomous tasks) + human-verify checkpoint (approved)
- **Completed:** 2026-05-22
- **Tasks:** 2/2 autonomous + Task 3 human-verify (approved)
- **Files modified:** 8 (2 created)

## Accomplishments
- **DFT-11 (Task 1):** `xc_backend.rs` `xc_eval_substrate`/`pipeline_fallback` delegate the shader-f64/ERF probe to `xcfun_gpu::auto_backend()` + `error_routing::must_fall_back_to_cpu()` (no re-implemented probe); default-f64 + shader-f64-less wgpu → CPU-f64 + a single `tracing::warn!` (target `pyscf_dft::xc_backend`), never silent f32. `wgpu_f64_fallback.rs` asserts the CPU fallback + warn and that explicit `PYSCF_DTYPE=f32` is NOT treated as silent degradation. `docs/env-vars.md` gained a "PYSCF_DTYPE in DFT" subsection (read-only `dtype()`/`mf.precision` accessor, below-bit-exact warn, f32-escape-hatch-vs-silent-degrade distinction, no-`set_precision` deferral).
- **CI surface (Task 2):** libxc `[patch.crates-io]` re-enabled but PROVEN inert (`cargo tree -p pyscf-dft` = 0 libxc_rs; `cargo build --workspace` default = 2.9s, 0 libxc-kernel). `dft-libxc-bitexact` job added DISABLED (`if: false`, the only `--features libxc` job, heavily cached, distinct cache key) with the `PENDING_LIBXC_RS_FEATURE_GATE` comment. `nightly-cross-crate.yml` `-p libxc_rs` stays EXCLUDED with refreshed rationale. `wgpu-no-f64-fallback` job added ACTIVE (default features only, + optional `PYSCF_DTYPE=f32` run-to-completion step).

## Task Commits
1. **Task 1: WGPU f64 honesty fallback + D-08 escape-hatch docs** — `099fd2f` (feat)
2. **Task 2: re-enable inert libxc patch + DISABLED libxc CI job + active wgpu-no-f64 job + nightly comment** — `14a7d5a` (feat/ci)

**Plan metadata:** this SUMMARY + tracking finalized by the orchestrator post-approval.

## Decisions Made
- Human-verify checkpoint **approved** by user (2026-05-22): WGPU honesty + inert libxc patch + DISABLED libxc CI surface accepted.
- libxc CI surface intentionally ships DISABLED while 04-02 is PENDING (the correct state — enabling without per-functional features triggers the ~6h/266-kernel compile).

## Deviations from Plan
- The plan's Task 2 assumed 04-02 had landed (corpus-subset per-functional features). Since the user kept 04-02 PENDING, the libxc CI job + nightly `-p libxc_rs` re-enable were added DISABLED/excluded (instead of active) — a deliberate adaptation to the PENDING decision, not a defect. The wgpu-no-f64 job + the patch re-enable proceeded as planned.

## Issues Encountered
- **Pre-existing (out-of-scope):** `check-cubecl-pin` FAILs on `cubecl-hip-sys: 7.1.5280200` (a HIP `*-sys` crate version flagged as a cubecl-* family member outside the carve-out). Confirmed present in `Cargo.lock` at the phase-4 base `f4999fa` (BEFORE any Phase-4 work) and NOT changed by 04-10's commits — i.e. a pre-existing FOUND-04 lint edge case, not a Phase-4 regression. Logged to `.planning/phases/04-dft/deferred-items.md`, left unfixed per scope boundary (likely needs `cubecl-hip-sys` added to the lint carve-out allowlist — separate work).

## Verification (local)
- `cargo build --workspace` (default) — 2.9s, 0 errors, 0 libxc-kernel compilation (patch inert)
- `cargo tree -p pyscf-dft` — 0 libxc_rs
- `cargo test -p pyscf-dft wgpu_f64_fallback` — 2 tests pass
- dependency-wall lint — PASS (xcfun-gpu is not cubecl-*)
- CI yaml source assertions: `dft-libxc-bitexact` `if: false` + PENDING comment (only `--features libxc` job); `wgpu-no-f64-fallback` active; nightly `-p libxc_rs` excluded with refreshed comment
- Did NOT run `cargo test --features libxc` (the ~6h freeze guardrail)

## Next Phase Readiness
- WGPU f64 honesty fallback logic + CPU-fallback unit test complete. The on-real-shader-f64-less-DEVICE run is the `wgpu-no-f64-fallback` CI job (special-runner / Phase 8).
- The gated libxc bit-exact CI path is wired but DISABLED — flip `if: false`→on (and re-add nightly `-p libxc_rs`) once 04-02 (`PENDING_LIBXC_RS_FEATURE_GATE`) lands the corpus-subset per-functional features.

---
*Phase: 04-dft*
*Completed: 2026-05-22 (human-verify approved)*
