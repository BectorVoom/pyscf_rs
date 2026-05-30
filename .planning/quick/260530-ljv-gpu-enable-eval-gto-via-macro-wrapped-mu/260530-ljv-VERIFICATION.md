---
phase: quick-260530-ljv
verified: 2026-05-30T00:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase quick-260530-ljv Verification Report

**Phase Goal:** GPU-enable eval_gto s-shell path via a real #[cube] kernel reached through the cross-crate-exported dispatch_backend! macro, validated bit-close (TOL 1e-9) against the CPU oracle — WITHOUT changing l>=1 numerics (behavior-preserving fallback).
**Verified:** 2026-05-30
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | dispatch_backend! is callable from pyscf-kernels and expands correctly | VERIFIED | `#[macro_export]` present at dispatch.rs:53; `use pyscf_algebra::dispatch_backend;` at eval_gto.rs:85; macro invoked at eval_gto.rs:676; `cargo build -p pyscf-kernels` exits 0 |
| 2 | pyscf-algebra's 16 in-crate dispatch_backend! call sites still build | VERIFIED | `cargo build -p pyscf-algebra` exits 0; `cargo test -p pyscf-algebra --lib` — 16 passed, 0 failed |
| 3 | pure-s-shell eval_gto runs a real #[cube(launch_unchecked)] kernel matching the oracle within 1e-9 | VERIFIED | `#[cube(launch_unchecked)]` at eval_gto.rs:477; `cargo test -p pyscf-kernels --test eval_gto_oracle` — `eval_gto_s_matches_oracle_on_cpu` PASSED; SUMMARY reports observed diff = 0 (bit-identical on CpuRuntime) |
| 4 | any l>=1 basis returns byte-identical existing CPU-path result | VERIFIED | Routing at eval_gto.rs:666-690: device path gated on `!bas.is_empty() && all ANG_OF==0 && ngrids*nao>0`; everything else falls to unchanged `eval_gto_sph_cpu`; all l>=1 tests pass — `eval_gto_lge1` suite: 4/4, lib tests: 2/2 |
| 5 | differential oracle test passes on always-on CpuRuntime arm | VERIFIED | Covered by truth #3 — test name `eval_gto_s_matches_oracle_on_cpu` runs under default features; result: ok. 1 passed |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/pyscf-algebra/src/dispatch.rs` | dispatch_backend! hoisted to crate root via #[macro_export] | VERIFIED | `#[macro_export]` at line 53; doc comment at line 44 explains cross-crate export |
| `crates/pyscf-kernels/src/eval_gto.rs` | s-shell #[cube(launch_unchecked)] kernel + macro-wrapped launch fanout + all-l0 routing | VERIFIED | `#[cube(launch_unchecked)]` at line 477; `dispatch_backend!` invocation at line 676; routing guard at lines 666-690 |
| `crates/pyscf-kernels/tests/eval_gto_oracle.rs` | differential oracle test: cube s-shell vs eval_gto_sph_cpu, CpuRuntime always-on + #[cfg(feature="rocm")] arm | VERIFIED | File is 318 lines (>= 80 required); `#[cfg(feature = "rocm")]` at line 295; `eval_gto_s_matches_oracle_on_cpu` at line 279; `eval_gto_s_matches_oracle_on_rocm` at line 297 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| eval_gto.rs | pyscf_algebra::dispatch_backend! | `use pyscf_algebra::dispatch_backend;` at line 85 | WIRED | Macro invoked at line 676 inside `eval_gto_sph`; pyscf-kernels/Cargo.toml lines 26-29 forward cpu/cuda/wgpu/rocm features to pyscf-algebra so the enum variants exist per backend |
| eval_gto.rs | eval_gto_sph_kernel cube launch | all-l0 guard routes to `launch_eval_gto_s::<Rt>` which calls `eval_gto_sph_kernel::launch_unchecked::<R>` | WIRED | `launch_unchecked` at eval_gto.rs:477; launcher at ~line 555; kernel called from launcher confirmed by build |
| eval_gto_oracle.rs | eval_gto_sph_cpu ground truth | inline oracle (y00 = 0.5_f64 / PI.sqrt(), F-order write) at lines 188-231 | WIRED | Oracle uses exact pin `y00 = 0.5_f64 / std::f64::consts::PI.sqrt()` at line 190; F-order write `out[g + ao_idx * ngrids] = acc * y00` at line 227; differential test invokes `eval_gto_sph` (device) vs inline oracle |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| pyscf-algebra 16 in-crate sites build | `cargo build -p pyscf-algebra` | Finished dev profile, 0 errors | PASS |
| pyscf-algebra lib tests | `cargo test -p pyscf-algebra --lib` | 16 passed, 0 failed | PASS |
| pyscf-kernels builds without libxc | `cargo build -p pyscf-kernels` | Finished dev profile, 0 errors | PASS |
| libxc not in dep graph | `cargo tree -p pyscf-kernels -i libxc` | "did not match any packages" | PASS |
| oracle cpu arm passes | `cargo test -p pyscf-kernels --test eval_gto_oracle` | 1 passed, 0 failed | PASS |
| all pyscf-kernels tests pass | `cargo test -p pyscf-kernels` | 8 total passed (lib 2, lge1 4, oracle 1, smoke 1), 0 failed | PASS |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No TBD/FIXME/XXX markers found in phase-modified files. No unreferenced debt markers. No stub returns in production paths.

---

### Oracle Pin Verification

The inline test oracle at `tests/eval_gto_oracle.rs:190` uses `y00 = 0.5_f64 / std::f64::consts::PI.sqrt()` — exact match of the ORACLE PIN required by the plan. The F-order write `out[g + ao_idx * ngrids] = acc * y00` at line 227 byte-matches `eval_gto_sph_cpu` lines 603-614. The differential check is not self-fulfilling: the oracle is inlined independently and does not call the production function.

---

### l>=1 Behavior Preservation

`eval_gto_sph_cpu` body: zero production lines were removed or changed since baseline commit `715b918`. The only removed lines (10 total) are the old `client.kind()` warn-guard block in `eval_gto_sph` — which was the dead-code stub that preceded the new routing. The function `fn eval_gto_sph_cpu` is byte-identical to baseline. All four `eval_gto_lge1` tests pass byte-for-byte.

---

### Feature-Forwarding Deviation (Auto-Fixed, Correctly)

The SUMMARY documents that `pyscf-kernels/Cargo.toml` was modified (commit `12ea384`) to forward `cpu/cuda/wgpu/rocm` features to `pyscf-algebra`. This is verified: lines 26-29 of the current Cargo.toml each include `pyscf-algebra/{feature}` in their feature list. This was a latent bug exposed by the first cross-crate macro invocation; the fix is correct and the build is clean.

---

### rocm gfx1152 Arm

The `#[cfg(feature = "rocm")]` arm exists at test line 295, with `eval_gto_s_matches_oracle_on_rocm` at line 297. The SUMMARY claims it was exercised on real AMD gfx1152 hardware with `max_abs_diff = 1.1102230246251565e-16` (~1 ULP, within TOL=1e-9). Verification cannot re-run this without AMD hardware; the arm's existence is confirmed in source. The always-on CpuRuntime arm is the correctness gate and it PASSES.

---

### Human Verification Required

None. All correctness gates are automated (cargo build + test). The rocm arm is hardware-gated and its existence is code-verified; the SUMMARY's claim of 1.11e-16 diff is noted but not re-verifiable without hardware — this does not affect the pass decision since the CpuRuntime arm is the mandatory gate.

---

_Verified: 2026-05-30_
_Verifier: Claude (gsd-verifier)_
