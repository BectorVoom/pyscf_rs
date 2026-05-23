---
phase: 04-dft
plan: 13
subsystem: dft
tags: [numint, f32-precision, overflow, error-handling, numeric-correctness]

# Dependency graph
requires:
  - phase: 04-dft
    provides: "D-08 f32/f64 precision-generic numint matmul chain (eval_rho_scalar / nr_rks_inner over Scalar S)"
provides:
  - "PyscfRsError::NumericOverflow { context } variant"
  - "Honest f32 precision path: f64→f32 overflow now returns Err instead of silently substituting inf/0.0"
  - "cast_finite / back_to_f64 fallible-cast helpers in numint.rs"
affects: [dft, scf, grad]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fallible scalar narrowing: cast_finite<S>/back_to_f64<S> detect the REAL f32 overflow mode (num-traits f32::from(1e40)=Some(inf), not None) and propagate Err(NumericOverflow); the F64 arm is the identity (bit-exact unchanged)"

key-files:
  created: []
  modified:
    - "crates/pyscf-core/src/error.rs"
    - "crates/pyscf-dft/src/numint.rs"

key-decisions:
  - "num-traits 0.2.19 f32::from(1e40) returns Some(f32::INFINITY) NOT None — the plan's prescribed S::from(x).ok_or_else(...) pattern alone would NOT have caught overflow; detection must also flag a finite f64 input mapping to a non-finite f32 (and a non-finite f32 accumulation back-cast)"
  - "Overflow detection isolated to the narrowing (F32) path via S::KIND guard so the f64 default path is bit-identical: identity cast passes finite + non-finite values through unchanged, matching the dead unwrap_or_else(S::zero) arm"
  - "Routed conversions through cast_finite/back_to_f64 helpers (ok_or, not ok_or_else — cheap struct literal, clippy-clean) instead of inline ok_or_else at each of the 9 call sites; satisfies the key_links ok_or.*NumericOverflow pattern while centralising the honest-cast logic"

patterns-established:
  - "Honest low-precision cast: never unwrap_or_else(S::zero) / unwrap_or(0.0) on a narrowing conversion in a numeric chain — surface NumericOverflow so f32-mode corruption is loud, not silent"

requirements-completed: [DFT-11]

# Metrics
duration: 11min
completed: 2026-05-23
---

# Phase 04 Plan 13: f32 Numint Overflow → Err (CR-02) Summary

**The f32 numint matmul chain (`eval_rho_scalar<f32>` / `nr_rks_inner<f32>`) now returns `Err(PyscfRsError::NumericOverflow)` on f64→f32 overflow instead of silently substituting `inf`/`0.0`, making the "honest f32 path" actually honest.**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-05-23T05:00:00Z (approx)
- **Completed:** 2026-05-23T05:11:00Z (approx)
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- Added `PyscfRsError::NumericOverflow { context: &'static str }` variant to `error.rs` with a debuggable per-site context string.
- Replaced every `S::from(x).unwrap_or_else(S::zero)` and `t.to_f64().unwrap_or(0.0)` in the f32 numeric chain (the `eval_rho_scalar` `contract` closure + the `nr_rks_inner` Vxc back-contraction loop) with fallible `cast_finite<S>` / `back_to_f64<S>` helpers that `?`-propagate `NumericOverflow`.
- Discovered and corrected a flaw in the plan's prescribed fix: with `num-traits 0.2.19`, `f32::from(1e40_f64)` returns `Some(f32::INFINITY)` (not `None`), so the plan's literal `S::from(x).ok_or_else(...)?` would have produced `Ok([inf])` — still silent corruption. The implemented helpers additionally flag a finite f64 input narrowing to a non-finite f32 (and a non-finite f32 accumulation on back-cast), which is the actual overflow mode.
- Kept the f64 default path bit-identical: `cast_finite`/`back_to_f64` are the identity for `S = f64` (a `S::KIND == F64` guard skips the finiteness rejection on back-cast), so `nr_rks_inner<f64>` / `eval_rho_scalar<f64>` behaviour is unchanged.
- New test `f32_overflow_returns_err_not_zero` proves the overflow returns `Err(NumericOverflow)` and is NOT `Ok((vec![0.0], None))`; companion test `f64_path_unchanged_no_overflow_on_large_values` proves the f64 path still computes `1e80` cleanly.

## Task Commits

Each task was committed atomically (TDD: RED → GREEN, no REFACTOR needed):

1. **Task 1 (RED): failing f32-overflow test + NumericOverflow variant + pub(crate) eval_rho_scalar** - `0e50971` (test)
2. **Task 1 (GREEN): cast_finite/back_to_f64 fallible casts propagate NumericOverflow** - `0ec3bfc` (fix)

**Plan metadata:** (this commit) (docs: complete plan)

## Files Created/Modified
- `crates/pyscf-core/src/error.rs` - Added `PyscfRsError::NumericOverflow { context: &'static str }` variant (after `EcpEngineNotAvailable`).
- `crates/pyscf-dft/src/numint.rs` - Added `cast_finite<S>`/`back_to_f64<S>` helpers; made the `eval_rho_scalar` `contract` closure fallible (`Result<Vec<f64>, PyscfRsError>`) and `?`-propagated its 3 call sites; converted the `nr_rks_inner` Vxc back-contraction (`phi_mu`, `phi_nu`, `wv`, `wvs`, `gphi_mu`, the `0.5` factor, and the f64 back-cast) to the fallible helpers; promoted `eval_rho_scalar` to `pub(crate)` for direct testing; added 2 unit tests.

## Decisions Made
- **Helper-based fallible cast over inline `ok_or_else` at every site.** Centralises the honest-cast logic in `cast_finite`/`back_to_f64`, avoids duplicating the finite-detection across 9 call sites, and uses `ok_or` (cheap struct literal — clippy-clean) while still matching the plan's `key_links` regex `ok_or.*NumericOverflow`.
- **Finiteness detection in addition to the `None` check.** Required because num-traits maps an out-of-range f64 to `Some(inf)`, not `None`; the `None` arm alone is dead for f32 overflow. See deviation below.
- **`S::KIND == F64` guard on the back-cast** keeps the f64 default path bit-for-bit identical (the must-have invariant), since a legitimately non-finite f64 (NaN/inf from a pathological XC) must pass through exactly as the old `unwrap_or(0.0)`-on-f64 (which was simply `t`) did.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan's prescribed overflow detection would not have detected the overflow**
- **Found during:** Task 1 (RED run revealed actual behaviour)
- **Issue:** The plan's interface specified replacing `S::from(x).unwrap_or_else(S::zero)` with `S::from(x).ok_or_else(|| NumericOverflow { .. })?`, on the assumption that `S::from(overflow)` returns `None`. The RED test exposed that `num-traits 0.2.19` returns `Ok(([inf], None))` — i.e. `f32::from(1e40_f64)` is `Some(f32::INFINITY)`, so `.ok_or_else(...)` never fires and the chain silently propagates `inf`. The plan's literal fix would have left CR-02 unfixed.
- **Fix:** Implemented `cast_finite<S>` which keeps the `ok_or(NumericOverflow)` for the (defensive) `None` arm AND additionally returns `Err(NumericOverflow)` when a finite f64 input narrows to a non-finite `S`; and `back_to_f64<S>` which flags a non-finite f32 accumulation (the second overflow mode) while leaving the f64 path's non-finite values untouched.
- **Files modified:** crates/pyscf-dft/src/numint.rs
- **Verification:** `f32_overflow_returns_err_not_zero` passes (Err returned, not Ok([inf]) or Ok([0.0])); `f64_path_unchanged_no_overflow_on_large_values` passes; full pyscf-dft suite (45 lib + all integration incl. `dtype_f32_smoke`, `rks_uks_bitexact`) green; clippy `-D warnings` clean; fmt clean.
- **Committed in:** `0ec3bfc` (GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug — the prescribed fix was insufficient for the actual num-traits behaviour).
**Impact on plan:** Necessary for correctness — without it CR-02 would be reported "fixed" while still silently corrupting. No scope creep; same two files, same `NumericOverflow` variant, same public surface as planned. The `key_links` pattern (`ok_or.*NumericOverflow`) and all must-have truths/artifacts are satisfied.

## Issues Encountered
- `num_traits` is not a direct dependency of `pyscf-dft`, so `num_traits::ToPrimitive::to_f64(&t)` (path form) failed to compile. Resolved by calling `t.to_f64()` as a method via the `Scalar: num_traits::Float: ToPrimitive` supertrait chain — no new dependency added (honours the threat-register `accept`/no-install disposition T-04-13-SC and the libxc-compile-avoidance constraint).
- Initial `ok_or_else(|| ...)` form tripped clippy `unnecessary_lazy_evaluations` (the error value is a cheap struct literal). Switched to `ok_or(...)`, which is both clippy-clean and still matches the `key_links` regex.

## Next Phase Readiness
- CR-02 closed. Phase 04 gap-closure remaining: Wave 7 — 04-14 (CR-01).
- No blockers introduced. f32 mode now fails loudly on overflow; downstream `nr_rks`/`nr_uks`/`nr_nlc_vxc` callers already `?`-propagate `PyscfRsError` so the new `Err` path surfaces to the SCF driver without signature changes.

---
*Phase: 04-dft*
*Completed: 2026-05-23*

## Self-Check: PASSED

- FOUND: `.planning/phases/04-dft/04-13-SUMMARY.md`
- FOUND: `crates/pyscf-core/src/error.rs` (NumericOverflow variant)
- FOUND: `crates/pyscf-dft/src/numint.rs` (cast_finite/back_to_f64 + fallible chain)
- FOUND commit: `0e50971` (RED — test + variant)
- FOUND commit: `0ec3bfc` (GREEN — fallible casts propagate NumericOverflow)
