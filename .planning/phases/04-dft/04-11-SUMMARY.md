---
phase: 04-dft
plan: 11
subsystem: dft
tags: [eval_gto, cart2sph, never-panic, pyo3-boundary, foundation-07, gap-closure]

# Dependency graph
requires:
  - phase: 04-dft (plan 04-03)
    provides: eval_gto_sph / eval_gto_sph_deriv1 l>=1 cart→sph kernel + c2s_coeff l<=4 tables
  - phase: 03-scf (PyO3 bindings)
    provides: panic→exception bridge that makes an unconditional panic in a kernel a process abort
provides:
  - "c2s_coeff returns Result<f64, PyscfRsError>; l>4 (h-shells and above) returns Err(NotYetImplemented{phase:4}) instead of panicking"
  - "eval_gto_sph / eval_gto_sph_cpu / eval_gto_sph_deriv1 / eval_gto_sph_deriv1_cpu are Result-returning end to end"
  - "FOUND-07 never-panic policy restored for user-supplied cc-pV5Z / ANO bases through the PyO3 boundary"
affects: [04-12, 04-13, 04-14, dft, grad, eval_gto consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Result-typed error propagation through the eval_gto kernel chain (c2s_coeff ? → CPU kernels ? → public surface)"
    - "Unsupported angular momentum is a typed NotYetImplemented{phase:4}, never a panic (FOUND-07)"

key-files:
  created: []
  modified:
    - crates/pyscf-kernels/src/eval_gto.rs
    - crates/pyscf-gto/src/eval_gto.rs
    - crates/pyscf-kernels/tests/eval_gto_lge1.rs

key-decisions:
  - "c2s_coeff l<=4 arms wrapped in Ok() preserve the FROZEN libcint Condon-Shortley coefficients byte-for-byte (no numeric regression); only the l>4 wildcard panic! becomes Err(NotYetImplemented{phase:4})"
  - "pyscf_gto::eval_gto was ALREADY Result-returning, so its public signature is unchanged — only two internal ? additions were needed and every downstream consumer (numint.rs eval_gto_block, etc.) compiles untouched"
  - "Integration-test call sites for l<=2 fixtures use .expect(...) on the Ok; the new lib test asserts l=5/l=6 return Err without aborting the process"

patterns-established:
  - "Kernel functions that can encounter unsupported user input return Result and propagate via ?; the panic→exception bridge therefore yields a Python exception, not a process abort"

requirements-completed: []  # DFT-01 remains [~] — this plan restores never-panic robustness of eval_gto but does NOT complete the bit-exact energy gate (still pending Phase-2 ERI rollup + live PySCF)

# Metrics
duration: 5min
completed: 2026-05-23
---

# Phase 4 Plan 11: CR-03 c2s_coeff l>4 panic → Result Summary

**`c2s_coeff` (cart→sph transform) now returns `Result<f64, PyscfRsError>` and surfaces `NotYetImplemented{phase:4}` for l>4 (h-shells: cc-pV5Z/ANO) instead of panic-aborting the Python process through the PyO3 boundary — Result threaded end-to-end through `eval_gto_sph`/`eval_gto_sph_deriv1`.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-23T04:35:26Z
- **Completed:** 2026-05-23T04:42:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- BLOCKER CR-03 closed: `c2s_coeff` no longer contains a `panic!` for any l value. l>4 returns `Err(PyscfRsError::NotYetImplemented{phase:4, ..})`, so a user supplying cc-pV5Z or an ANO basis through `from pyscf import dft` receives a Python exception instead of an aborted interpreter (FOUND-07 never-panic policy restored).
- `c2s_coeff`, `eval_gto_sph_cpu`, `eval_gto_sph_deriv1_cpu`, and the public `eval_gto_sph` / `eval_gto_sph_deriv1` are all `Result<EvalGtoBuffers, PyscfRsError>`-returning, with `?` threaded at the two c2s call sites (l>=1 value path + deriv1 path) and the early-out / final returns wrapped in `Ok(...)`.
- The l<=4 coefficient tables are byte-for-byte unchanged (FROZEN libcint provenance) — no numeric regression on the existing g/f/d/p/s corpus; the `eval_gto_lge1` 1000-point reference-grid parity test still passes to 1e-10.
- New lib test `c2s_coeff_l5_returns_err_not_panic` proves l=5 and l=6 return `Err(NotYetImplemented{phase:4})` and do not panic; companion `c2s_coeff_l_le_4_unchanged` locks the l<=4 frozen values.
- All downstream consumers compile unchanged: `pyscf_gto::eval_gto` was already `Result`-returning (only two internal `?` additions), so `pyscf-dft::numint.rs::eval_gto_block` and every other caller needed no signature change.

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): failing test for c2s_coeff l>4 returns Err** — `a67f2bf` (test)
2. **Task 1 (GREEN): c2s_coeff returns Result; propagate through eval_gto kernels** — `35bf4a1` (feat)
3. **Task 2: thread Result through eval_gto_sph downstream call sites** — `45528f8` (fix)

**Plan metadata:** _(this commit)_ (docs: complete plan)

_TDD task 1 produced the RED `test(...)` commit followed by the GREEN `feat(...)` commit; no refactor commit was needed (implementation was minimal)._

## Files Created/Modified

- `crates/pyscf-kernels/src/eval_gto.rs` — `c2s_coeff` signature → `Result<f64, PyscfRsError>` with l>4 wildcard `Err(NotYetImplemented{phase:4})`; `eval_gto_sph_cpu` / `eval_gto_sph_deriv1_cpu` / `eval_gto_sph` / `eval_gto_sph_deriv1` return `Result`; added `use pyscf_core::PyscfRsError`; added `#[cfg(test)] mod tests` with the CR-03 never-panic test + l<=4 frozen-value test.
- `crates/pyscf-gto/src/eval_gto.rs` — appended `?` to the `eval_gto_sph` and `eval_gto_sph_deriv1` calls inside `pyscf_gto::eval_gto` (already a `Result`-returning fn; public signature unchanged).
- `crates/pyscf-kernels/tests/eval_gto_lge1.rs` — 3 integration-test call sites now `.expect(...)` the `Ok` (l<=2 fixtures).

## Decisions Made

- **Preserve l<=4 byte-exact:** the FROZEN libcint Condon-Shortley coefficient tables are wrapped in `Ok(...)` with no value change, so the only behavioral delta is the l>4 arm (panic → typed Err). Confirmed by `c2s_coeff_l_le_4_unchanged` and the unchanged `eval_gto_lge1` parity tests.
- **No public-signature churn downstream:** `pyscf_gto::eval_gto` already returned `Result<EvalGtoOutput, PyscfRsError>`, so threading `?` internally was sufficient — `numint.rs` and all other consumers were left untouched, minimizing blast radius.

## Deviations from Plan

None — plan executed exactly as written. (One incidental note: the plan's Task-1 action text said the test block was "at the bottom" of `eval_gto.rs`; the file had no `#[cfg(test)]` module, so a new one was added, exactly as the plan's fallback instruction described.)

## Issues Encountered

- **Flaky `pyscf-dft` `wgpu_f64_fallback` test (out of scope):** during the Task-2 batch run `cargo test -p pyscf-kernels -p pyscf-gto -p pyscf-dft`, `tests/wgpu_f64_fallback.rs::wgpu_f64_fallback` failed once, then passed deterministically on an isolated re-run (both subtests green). `pyscf-dft` is NOT modified by this plan (`git diff --name-only` confirms); the flake is WGPU-probe + `PYSCF_BACKEND` env-var sensitivity, matching the project-memory note on flaky env-var tests. Logged to `.planning/phases/04-dft/deferred-items.md` under `## 04-11` and left unfixed per the SCOPE BOUNDARY rule.
- **gsd-sdk state/roadmap auto-handlers partially mis-parsed this gap-closure phase's STATE format:** `state.advance-plan` and `state.record-metric` errored, and `state.update-progress` / `roadmap.update-plan-progress` overwrote descriptive fields with stale/generic values. The SDK-induced doc changes were reverted (`git checkout -- STATE.md ROADMAP.md`) and STATE.md / ROADMAP.md were updated manually with correct content (plan counter 38→39, percent 89%, ROADMAP phase-04 row 10/14→11/14, 04-11 marked `[x]`).

## User Setup Required

None — no external service configuration required. Pure-Rust source change; no new dependencies (T-04-11-SC: no package installs).

## Threat Model Outcome

- **T-04-11-01 (Denial of Service, mitigate):** RESOLVED. `c2s_coeff` for l>4 now returns `Err` instead of `panic!`, preventing process abort through the PyO3 boundary. Verified by `c2s_coeff_l5_returns_err_not_panic` (l=5 and l=6 return `Err(NotYetImplemented{phase:4})`, no panic).
- **T-04-11-SC (Tampering, accept):** Honored — no new dependencies added.

No new security surface introduced beyond the plan's threat register.

## Next Phase Readiness

- Remaining Phase 04 gap-closure plans: Wave 6 — 04-12 (CR-04 cache fingerprint), 04-13 (CR-02 f32 NumericOverflow); Wave 7 — 04-14 (CR-01 UKS open-shell). 04-13 and 04-14 both touch `numint.rs` and are Wave-7-blocked accordingly; this plan does not touch `numint.rs`, so no new conflict introduced.
- DFT-01 stays `[~]`: never-panic robustness of the eval_gto kernel is restored, but the bit-exact RKS/UKS energy gate is still pending the Phase-2 arity-3/4 ERI rollup + live PySCF on CI (unchanged by this plan).

## Self-Check: PASSED

- Files verified present: `crates/pyscf-kernels/src/eval_gto.rs`, `crates/pyscf-gto/src/eval_gto.rs`, `crates/pyscf-kernels/tests/eval_gto_lge1.rs`, `.planning/phases/04-dft/04-11-SUMMARY.md`
- Commits verified in git log: `a67f2bf` (test/RED), `35bf4a1` (feat/GREEN), `45528f8` (fix/Task 2)
- `grep panic! crates/pyscf-kernels/src/eval_gto.rs` matches only a doc comment, not `c2s_coeff` code — no `panic!` remains in the transform.
- TDD gate compliance: RED `test(...)` commit `a67f2bf` precedes GREEN `feat(...)` commit `35bf4a1`; RED confirmed by compile failure against the old `f64` signature.

---
*Phase: 04-dft*
*Completed: 2026-05-23*
