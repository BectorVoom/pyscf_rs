---
phase: quick-260530-oms
plan: 01
subsystem: gpu-enable / eval_gto
tags: [cubecl, eval_gto, deriv1, gga, gpu-enable, differential-oracle, rocm]
requires: [quick-260530-mlg]
provides:
  - "eval_gto_sph_deriv1 device path (value + ∂x/∂y/∂z) for maxl<=4 on CpuRuntime + ROCm"
  - "dpow #[cube] helper + eval_gto_sph_deriv1_kernel #[cube] + launch_eval_gto_deriv1<R>"
  - "deriv1 4-component differential-oracle gate (cpu always-on + #[cfg(rocm)] gfx1152)"
affects:
  - "DFT GGA numint AO-derivative path can now run on device (no host warn! fallback for maxl<=4)"
tech-stack:
  added: []
  patterns:
    - "cubecl 0.10 idioms cloned verbatim from the shipped general kernel (usize scalar args, ABSOLUTE_POS usize, CubeDim::new_1d, 2-arg ArrayArg::from_raw_parts, launch_unchecked, i32->u32 casts, .exp() method)"
    - "#[cube] helper calls #[cube] helper (dpow -> ipow, Solution 1 from the host-fn-in-#[cube] pitfall guide)"
    - "statement-form if (let mut r = 0.0; if lq>=1 { ... }) to dodge ExpandElementTyped vs {float} mismatch + u32 underflow"
key-files:
  created: []
  modified:
    - crates/pyscf-kernels/src/eval_gto.rs
    - crates/pyscf-kernels/tests/eval_gto_oracle.rs
decisions:
  - "Kernel + oracle both sum radial/radial_2a SEQUENTIALLY (plain acc, NOT oracle_sum) per ORACLE-07 — oracle_sum == strict sequential for nprim<=128; the independent oracle matches the kernel's ordering exactly"
  - "deriv2 stays NotYetImplemented (out of scope — no CPU impl to port)"
metrics:
  duration: ~25m
  completed: 2026-05-30
---

# Phase quick-260530-oms Plan 01: GPU-enable eval_gto_sph_deriv1 (value + 3 gradients) Summary

Ported `eval_gto_sph_deriv1_cpu` (AO value + 3 analytic cartesian gradient components, `[4, ngrids, nao]`) into a real `#[cube(launch_unchecked)]` GPU kernel reusing the shipped mlg general-kernel angular infrastructure (`build_angular_tables` / `AngularTables` / `ipow`), routed via `dispatch_backend!`. This completes the `eval_gto` device surface for the DFT GGA gradient AO-derivative path (deriv2 remains out of scope — no CPU impl).

## What was built

- **`dpow` `#[cube]` helper** (eval_gto.rs ~690): statement form `let mut r = 0.0; if lq >= 1 { r = (lq as f64) * ipow(q, lq - 1); }`. Calls the `#[cube]` `ipow` (Solution 1, legal). The `if lq >= 1` gate short-circuits `lq == 0` before the `lq - 1` u32 subtraction, preventing underflow. Parallels host `q.powi(lq as i32 - 1)` (<1 ULP divergence at l>=3, inside TOL).
- **`eval_gto_sph_deriv1_kernel` `#[cube(launch_unchecked)]`** (eval_gto.rs ~806): same arg shape as `eval_gto_sph_kernel_general` plus a trailing `comp_stride: usize` scalar; `out` is `4*ngrids*nao`. One thread per `(g, shell)`, bounds-guard `if tid < ngrids*nbas`. Sequential `radial` + `radial_2a` in ONE ordered p-loop (g0 formed once), both `*fac1`; per-ci analytic gradient chain rule (operand order identical to host eval_gto.rs ~1382-1387); c2s transform applied to all 4 components; F-order write `out[k*comp_stride + off]`.
- **`launch_eval_gto_deriv1<R: Runtime>`** (eval_gto.rs ~1086): clones `launch_eval_gto_general` byte-for-byte except `out_len = 4*ngrids*nao` and the extra `comp_stride = ngrids*nao` scalar passed after the trailing scalar args.
- **Routing** in `eval_gto_sph_deriv1` (eval_gto.rs ~1234): `!bas.is_empty() && maxl <= 4 && comp_stride > 0` → `dispatch_backend!(..., launch_eval_gto_deriv1::<Rt>(...))` returning `EvalGtoBuffers { values, shape: vec![4, ngrids, nao] }`; else → UNCHANGED `eval_gto_sph_deriv1_cpu` (l>4 NotYetImplemented{phase:4} + comp_stride==0 early-return preserved). Old `client.kind()` warn-guard dropped; the now-unused `pyscf_runtime::BackendKind` import removed.
- **deriv1 4-component differential oracle** (eval_gto_oracle.rs): `oracle_eval_deriv1` (independent longhand via `lge1_reference::*` + a local host `dpow`, sequential radial+radial_2a, 4-component F-order write), `check_deriv1_case`, `DERIV1_CASES`, and `eval_gto_deriv1_matches_oracle_on_cpu` (CpuRuntime always-on) + `eval_gto_deriv1_matches_oracle_on_rocm` (`#[cfg(rocm)]` gfx1152).

## Output requested by the plan

1. **Compile fix-forward needed:** None of substance. The dpow-calls-ipow `#[cube]` pattern, the `u32` casts (`as u32` on i32 cpow entries, `lq - 1` guarded), and the `.exp()` method all compiled first try following the shipped general kernel idioms verbatim. The only edit beyond the planned additions was removing the now-unused `pyscf_runtime::BackendKind` import (it was only used by the dropped warn-guard) — required to keep the build warning-free. No genuine cubecl 0.10 blocker appeared.

2. **CpuRuntime deriv1 max_abs_diff (all 4 components, p/d/f/g + mixed):** `2.220446049250313e-16` (1 ULP — effectively bit-exact, vastly inside TOL=1e-9).

3. **ROCm gfx1152 arm RUN?** YES — `eval_gto_deriv1_matches_oracle_on_rocm` ran on real gfx1152 hardware and passed. Worst max_abs_diff = `4.440892098500626e-16` (2 ULP, inside TOL=1e-9).

4. **Full suite green + l>4 path intact + value-path unchanged:** CONFIRMED.
   - `cargo test -p pyscf-kernels` (default cpu): **0 FAILED**, 10 passed — lib (2, incl. `c2s_coeff_l5_returns_err_not_panic` + `c2s_coeff_l_le_4_unchanged`), eval_gto_lge1 (4), eval_gto_oracle (3, incl. the new deriv1 cpu), wave0_cubecl_smoke (1).
   - `cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle`: **6 passed** on gfx1152 (s-shell cpu/rocm, mixed-l cpu/rocm, deriv1 cpu/rocm).
   - `cargo clippy -p pyscf-kernels --all-targets`: clean (0 errors; only the pre-existing fma4 target-feature warning).
   - l>4 / empty basis / empty grid still route to the UNCHANGED `eval_gto_sph_deriv1_cpu` (NotYetImplemented{phase:4} preserved). Value-path tests (s-shell + mixed-l oracle) unchanged and green.
   - No `-p pyscf-gto` invoked anywhere; libxc/cintx never pulled (all logs scoped `-p pyscf-kernels`).

## Deviations from Plan

**1. [Rule 3 - Blocking] Removed the now-unused `pyscf_runtime::BackendKind` import.**
- **Found during:** Task 1 routing rewrite.
- **Issue:** The old `eval_gto_sph_deriv1` body used `client.kind()` / `BackendKind::Cpu` in its warn-guard, which the plan directed me to drop. With the warn-guard gone, the `use pyscf_runtime::BackendKind;` import became unused (would emit an `unused_imports` warning, breaking the warning-free build the done-criteria require).
- **Fix:** Removed the single import line. No other usage of `BackendKind` in eval_gto.rs.
- **Files modified:** crates/pyscf-kernels/src/eval_gto.rs
- **Commit:** f07e4c6

Otherwise the plan executed exactly as written.

## Verify status

| Task | What | Commit | Status |
| ---- | ---- | ------ | ------ |
| T1 | dpow + eval_gto_sph_deriv1_kernel + launch_eval_gto_deriv1 + routing | f07e4c6 | PASS (cpu + rocm builds, clippy clean, 3 new fns present) |
| T2 | deriv1 4-component differential oracle (cpu + rocm) | e3f5221 | PASS (cpu 2.22e-16, rocm gfx1152 4.44e-16, both << 1e-9) |
| T3 | full-suite behavior preservation | (verification-only, no commit) | PASS (cpu 10/10, rocm oracle 6/6, clippy clean) |

Note on the T1 verify `grep -c` returning 4 (not the documented 3): the 4th match is the **pre-existing host inner `fn dpow`** inside `eval_gto_sph_deriv1_cpu` (eval_gto.rs ~1558), which predates this plan. The 3 NEW functions (`dpow` ~690, `eval_gto_sph_deriv1_kernel` ~806, `launch_eval_gto_deriv1` ~1086) all exist as required.

## Deferred work

- **pyscf-dft numint device-path lock test (slice B, next)** — wiring the GGA numint grid loop to consume this device deriv1 path with a lock/regression test is the next slice. Not in scope here (would pull pyscf-dft/libxc).
- **deriv2 is NOT portable** — `eval_gto_sph_deriv2` has no CPU implementation to mirror; it remains `NotYetImplemented`. This plan COMPLETES the eval_gto device surface for everything that has a host reference.

## Self-Check: PASSED

- crates/pyscf-kernels/src/eval_gto.rs — FOUND (dpow ~690, eval_gto_sph_deriv1_kernel ~806, launch_eval_gto_deriv1 ~1086, routing in eval_gto_sph_deriv1)
- crates/pyscf-kernels/tests/eval_gto_oracle.rs — FOUND (oracle_eval_deriv1, check_deriv1_case, DERIV1_CASES, both test fns)
- Commit f07e4c6 — FOUND
- Commit e3f5221 — FOUND
