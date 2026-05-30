---
phase: quick-260530-mlg
verified: 2026-05-30T00:00:00Z
status: passed
score: 7/7
overrides_applied: 0
---

# Phase quick-260530-mlg: GPU-enable eval_gto l>=1 Verification Report

**Phase Goal:** GPU-enable eval_gto l>=1 cart->sph via a general #[cube] kernel (l 0..4)
over host-precomputed angular device tables, routed via dispatch_backend!, matching the
CPU oracle within 1e-9, WITHOUT breaking l>4/empty behavior or the validated s-shell path.

**Verified:** 2026-05-30
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | eval_gto_sph routes maxl 1..=4 to a real #[cube] kernel via dispatch_backend! matching eval_gto_sph_cpu within 1e-9 | VERIFIED | Routing arm at line 1015; dispatch_backend! call at line 1021; eval_gto_lge1_matches_oracle_on_cpu PASS (diff = 6.94e-18, far inside 1e-9) |
| 2 | Mixed-l bases (s+p+d) evaluate uniformly on device — l=0 subsumed by general kernel | VERIFIED | MIXED_CASES includes `(1, &[0,1,2], 3, 17)` (cc-pVDZ-like) and `(1, &[0,1,2,3,4], 5, 25)` (s..g); all pass eval_gto_lge1_matches_oracle_on_cpu |
| 3 | l>4 NEVER reaches the kernel — stays on eval_gto_sph_cpu, NotYetImplemented{phase:4}, never panics | VERIFIED | maxl<=4 gate at line 1015; c2s_coeff_l5_returns_err_not_panic PASS; final eval_gto_sph_cpu fallback at line 1034 unchanged |
| 4 | Empty grid or empty basis falls back to eval_gto_sph_cpu (out_len==0 early return) | VERIFIED | Guard `!bas.is_empty() && maxl<=4 && ngrids*nao>0` at line 1015; empty cases bypass kernel and fall to eval_gto_sph_cpu |
| 5 | Existing pure-s-shell fast path (launch_eval_gto_s) is unchanged | VERIFIED | all_s arm at lines 980-1003 intact; all_s check precedes general arm; git diff 8eae723..HEAD shows zero removals in that section |
| 6 | All existing pyscf-kernels tests pass (eval_gto_lge1, s-shell oracle, lib, wave0 smoke) | VERIFIED | cargo test -p pyscf-kernels: 9/9 pass, 0 FAILED — lib 2/2, eval_gto_lge1 4/4, eval_gto_oracle 2/2, wave0 smoke 1/1 |
| 7 | Differential oracle gains randomized p/d/f/g mixed-l fixtures + #[cfg(rocm)] arm | VERIFIED | MIXED_CASES (8 cases including pure p/d/f/g + mixed s+p+d + mixed p+d+f + full s..g); #[cfg(feature="rocm")] arm at line 847 in eval_gto_oracle.rs |

**Score:** 7/7 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/pyscf-kernels/src/eval_gto.rs` | #[cube] eval_gto_sph_kernel_general + ipow + build_angular_tables + launch_eval_gto_general + maxl<=4 routing | VERIFIED | All 5 components present: ipow at line 657, kernel at 679, AngularTables struct+builder at 772/793, launcher at 851, routing at 1015 |
| `crates/pyscf-kernels/tests/eval_gto_oracle.rs` | mixed-l randomized fixtures vs eval_gto_sph_cpu, CpuRuntime always-on + rocm arm | VERIFIED | lge1_reference module at line 339, build_mixed_l_fixture at 650, MIXED_CASES 8 fixtures at 820, cpu test at 832, #[cfg(rocm)] test at 847 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| eval_gto_sph | launch_eval_gto_general | dispatch_backend! when maxl<=4 && ngrids*nao>0 | VERIFIED | Lines 1015-1025: `if !bas.is_empty() && maxl <= 4 && ngrids * nao > 0 { ... dispatch_backend!(client, c, Rt, launch_eval_gto_general::<Rt>(...))` |
| eval_gto_sph_kernel_general | ipow | #[cube] helper call inside kernel body | VERIFIED | Line 762: `let mono = ipow(dx, lx) * ipow(dy, ly) * ipow(dz, lz);` inside the #[cube(launch_unchecked)] kernel |
| eval_gto_sph (l>4/empty) | eval_gto_sph_cpu | unchanged CPU fallback preserving NotYetImplemented | VERIFIED | Line 1034: `eval_gto_sph_cpu(...)` as final fallback after all device arms; c2s_coeff_l5_returns_err_not_panic confirms error path intact |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CpuRuntime mixed-l oracle: p/d/f/g within 1e-9 | cargo test -p pyscf-kernels --test eval_gto_oracle | eval_gto_s_matches_oracle_on_cpu ok, eval_gto_lge1_matches_oracle_on_cpu ok (worst diff = 6.94e-18) | PASS |
| Full pyscf-kernels suite (9 tests, 0 FAILED) | cargo test -p pyscf-kernels | 9 passed, 0 failed | PASS |
| c2s_coeff l5 error path intact | cargo test -p pyscf-kernels --lib | c2s_coeff_l5_returns_err_not_panic ok | PASS |
| No libxc in dependency graph | cargo tree -p pyscf-kernels \| grep libxc | (no output) | PASS |

---

### Host-Only Helper Isolation Check

Must-have: c2s_coeff/cart_powers/common_fac_sp NOT called inside any #[cube] fn.

- `#[cube]` functions in eval_gto.rs: `ipow` (lines 655-677) and `eval_gto_sph_kernel_general` (lines 678-770).
- grep for host helper calls in those line ranges: none found.
- `c2s_coeff` called at: 819 (build_angular_tables — host), 1207 (eval_gto_sph_cpu — host), 1399 (eval_gto_sph_deriv1_cpu — host), unit tests.
- `cart_powers` called at: 823 (build_angular_tables — host), 1158 (eval_gto_sph_cpu — host), 1334 (eval_gto_sph_deriv1_cpu — host).
- `common_fac_sp` called at: 812 (build_angular_tables — host), 1157 (eval_gto_sph_cpu — host), 1333 (eval_gto_sph_deriv1_cpu — host).

Status: VERIFIED — all three host helpers confined to host-only code paths.

---

### CPU l>=1 Branch Preservation (git diff)

`git diff 8eae723..HEAD -- crates/pyscf-kernels/src/eval_gto.rs` shows:
- Removed lines: 2 (old routing comments only, replaced by updated comments)
- eval_gto_sph_cpu function body: zero lines removed or modified
- The only `-` lines are old comments in the routing section (`// Fallback: any l>=1 shell...`)

Status: VERIFIED — CPU l>=1 math branch body unchanged from baseline.

---

### rocm Arm Existence (no hardware required)

`#[cfg(feature = "rocm")]` arms exist at:
- eval_gto_oracle.rs line 295: `eval_gto_s_matches_oracle_on_rocm`
- eval_gto_oracle.rs line 847: `eval_gto_lge1_matches_oracle_on_rocm`

The SUMMARY claims the rocm arm was run on gfx1152 hardware (worst diff 1.11e-16). This verifier did not re-run the rocm arm (AMD hardware not present in this environment), per the task specification which explicitly states "You need NOT re-run rocm — just confirm the #[cfg(feature='rocm')] arm exists." Both arms confirmed present.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| eval_gto.rs | 444 | "not yet implemented" in error string | Info | This is the REQUIRED NotYetImplemented{phase:4} error return for l>4 — not a stub, intentional design per must-have 3 |

No TBD/FIXME/XXX markers found in either modified file. The "not yet implemented" text is the intended error message inside c2s_coeff's wildcard arm, present in the baseline and preserved by this phase.

---

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| GTO-07 | eval_gto GPU path for l>=1 | SATISFIED | eval_gto_sph_kernel_general ships; maxl<=4 routing active |
| ORACLE-07 | Differential oracle gate < 1e-9 | SATISFIED | TOL=1e-9; observed worst diff 6.94e-18 on CpuRuntime |
| D-04 | dispatch_backend! used for device routing | SATISFIED | dispatch_backend! at line 1021 for launch_eval_gto_general |

---

### Human Verification Required

None. All must-haves are verifiable programmatically and cargo test gates ran successfully.

---

## Gaps Summary

No gaps. All 7 must-have truths verified, all artifacts substantive and wired, all key links confirmed, full test suite green (9/9), no libxc in dependency graph, no debt markers.

---

_Verified: 2026-05-30T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
