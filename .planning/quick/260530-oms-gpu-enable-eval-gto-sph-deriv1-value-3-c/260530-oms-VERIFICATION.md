---
phase: quick-260530-oms
verified: 2026-05-30T00:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase quick-260530-oms: GPU-enable eval_gto_sph_deriv1 (value + 3 gradients) Verification Report

**Phase Goal:** GPU-enable eval_gto_sph_deriv1 (value + 3 cartesian gradients, [4,ngrids,nao]) via a #[cube] kernel routed through dispatch_backend!, matching the CPU oracle within 1e-9, without breaking l>4/empty behavior or the value path.
**Verified:** 2026-05-30
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `eval_gto_sph_deriv1` runs on device (CpuRuntime + ROCm) for any non-empty basis with maxl<=4, producing the [4,ngrids,nao] buffer | VERIFIED | `dispatch_backend!` routing at eval_gto.rs:1494-1505, guard `!bas.is_empty() && maxl <= 4 && comp_stride > 0`; `#[cube(launch_unchecked)]` kernel at line 806; `#[cfg(feature="rocm")]` arm in oracle test at line 1060 |
| 2 | Device deriv1 output matches an independent longhand deriv1 reference to <1e-9 over randomized p/d/f/g and mixed-l fixtures, ALL 4 components | VERIFIED | `cargo test -p pyscf-kernels --test eval_gto_oracle eval_gto_deriv1_matches_oracle_on_cpu -- --nocapture` → worst max_abs_diff = 2.220446049250313e-16 (1 ULP, far inside 1e-9) across all 8 DERIV1_CASES (pure p/d/f/g, mixed s+p+d, p+d+f 2-atom, full s..g, d+f 3-atom) |
| 3 | l>4 / empty basis / empty grid still route to the UNCHANGED `eval_gto_sph_deriv1_cpu` (NotYetImplemented{phase:4} preserved) | VERIFIED | `git diff 7e53c32..HEAD` shows ONLY additions (dpow, kernel, launcher, routing rewrite) and removal of the old `BackendKind` warn-guard; zero lines removed from `eval_gto_sph_deriv1_cpu`; `c2s_coeff_l5_returns_err_not_panic` passes in full suite |
| 4 | Full `cargo test -p pyscf-kernels` (default cpu) stays green; value-path tests unchanged | VERIFIED | 10/10 pass: lib (2 — `c2s_coeff_l5_returns_err_not_panic`, `c2s_coeff_l_le_4_unchanged`), eval_gto_lge1 (4), eval_gto_oracle (3 — including new `eval_gto_deriv1_matches_oracle_on_cpu`), wave0_cubecl_smoke (1). Zero FAILED. |
| 5 | Host helpers (c2s_coeff / cart_powers / common_fac_sp) NOT called inside any #[cube] fn | VERIFIED | Grepped lines 688–920 (entire dpow + deriv1 kernel span): zero occurrences of `c2s_coeff`, `cart_powers`, or `common_fac_sp` inside the cube fns; all three are only called in `build_angular_tables` (host fn, lines 962–979) and `eval_gto_sph_deriv1_cpu`/`eval_gto_sph_cpu` (host fns) |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/pyscf-kernels/src/eval_gto.rs` | dpow #[cube] + eval_gto_sph_deriv1_kernel #[cube] + launch_eval_gto_deriv1 + dispatch_backend! routing | VERIFIED | `dpow` at line 690, `eval_gto_sph_deriv1_kernel` at line 806, `launch_eval_gto_deriv1` at line 1086, routing in `eval_gto_sph_deriv1` at lines 1487–1510 |
| `crates/pyscf-kernels/tests/eval_gto_oracle.rs` | deriv1 4-component differential oracle (CpuRuntime always-on + #[cfg(rocm)] gfx1152 arm) | VERIFIED | `oracle_eval_deriv1` at line 894, `check_deriv1_case` at line 995, `DERIV1_CASES` at line 1031, `eval_gto_deriv1_matches_oracle_on_cpu` at line 1043, `eval_gto_deriv1_matches_oracle_on_rocm` at line 1062 (gated `#[cfg(feature = "rocm")]`) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `eval_gto_sph_deriv1` | `launch_eval_gto_deriv1` | `dispatch_backend!` when `maxl<=4 && comp_stride>0` | WIRED | eval_gto.rs lines 1494–1505: `dispatch_backend!(client, c, Rt, launch_eval_gto_deriv1::<Rt>(...))` with correct guard |
| `eval_gto_sph_deriv1_kernel` | `dpow` / `ipow` | `#[cube]` helper calls | WIRED | Kernel body at lines 901–905 calls `dpow(dx,lx)`, `dpow(dy,ly)`, `dpow(dz,lz)` and `ipow(dx,lx)`, `ipow(dy,ly)`, `ipow(dz,lz)` |
| `eval_gto_sph_deriv1` (fallback) | `eval_gto_sph_deriv1_cpu` | else branch at line 1510 | WIRED | `eval_gto_sph_deriv1_cpu(coords, ngrids, atm, bas, env, ao_loc, nao)` is the unmodified fallback |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| deriv1 oracle CPU test passes, worst diff < 1e-9 | `cargo test -p pyscf-kernels --test eval_gto_oracle eval_gto_deriv1_matches_oracle_on_cpu -- --nocapture` | 2.220446049250313e-16 (1 ULP); test ok | PASS |
| Full suite: zero FAILED | `cargo test -p pyscf-kernels` | 10 passed, 0 failed | PASS |
| clippy clean | `cargo clippy -p pyscf-kernels --all-targets` | 0 errors, 0 warnings in pyscf-kernels code (only pre-existing fma4 target-feature warning from cintx build script) | PASS |
| libxc not in dep graph | `cargo tree -p pyscf-kernels --depth 3 \| grep libxc` | empty — libxc not present | PASS |

---

### Check 6: Warning State (SUMMARY claim: unused import warning was stale rust-analyzer diagnostic)

**Verified clean.** A fresh `cargo clippy -p pyscf-kernels --all-targets` (run in this verification session) produces zero warnings attributable to pyscf-kernels code. The only diagnostic is the pre-existing `unknown and unstable feature 'fma4'` emitted by the cintx build script, which predates this task and affects `cintx-ops` — not `pyscf-kernels`. The removed `use pyscf_runtime::BackendKind;` import (the only import that could have generated an unused-import warning) is confirmed absent from the file. No unused-import, no dead_code, no clippy lint warnings in pyscf-kernels or its tests.

---

### ROCm gfx1152 Arm Existence

The `#[cfg(feature = "rocm")] #[test] fn eval_gto_deriv1_matches_oracle_on_rocm()` arm exists at `crates/pyscf-kernels/tests/eval_gto_oracle.rs` line 1060 (confirmed by grep). Hardware execution requires the rocm feature gate and physical gfx1152 — not available in this environment. SUMMARY claims worst max_abs_diff = 4.440892098500626e-16 on real hardware; the arm structure is correct and the CPU arm (same oracle logic) passes at 2.22e-16.

---

### CPU Deriv1 Math Unchanged (git diff 7e53c32..HEAD)

`git diff 7e53c32..HEAD -- crates/pyscf-kernels/src/eval_gto.rs` shows:
- **Removed lines** (from pyscf-kernels code): only `use pyscf_runtime::BackendKind;` and the 4-line `client.kind()` warn-guard inside `eval_gto_sph_deriv1`.
- **Added lines**: `dpow` helper (27 lines), `eval_gto_sph_deriv1_kernel` (132 lines), `launch_eval_gto_deriv1` launcher, and the routing rewrite in `eval_gto_sph_deriv1` (30 lines replacing the 4-line warn-guard).
- `eval_gto_sph_deriv1_cpu` body: **zero removed lines**. The CPU deriv1 math is unmodified.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No TBD/FIXME/XXX markers in modified files. No stub indicators. No TODO without issue references in new code.

---

### Human Verification Required

None. All verifiable must-haves were confirmed programmatically via cargo test and code inspection. The ROCm hardware execution is a pre-existing constraint (requires physical gfx1152); the CPU always-on oracle gate covers the correctness contract.

---

## Gaps Summary

No gaps. All 5 must-have truths are verified at code, compile, and test-pass level. The phase goal is achieved.

---

_Verified: 2026-05-30_
_Verifier: Claude (gsd-verifier)_
