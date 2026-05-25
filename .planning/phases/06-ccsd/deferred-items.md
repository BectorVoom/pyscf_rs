# Deferred Items — Phase 06 (ccsd)

Out-of-scope discoveries logged during execution (NOT fixed — they predate or
fall outside the current plan's file set).

## 06-06 execution (2026-05-25)

- **Pre-existing clippy `absurd_extreme_comparisons` in `crates/pyscf-ccsd/src/ccsd.rs:203`**
  — `Some(stack) if istep >= DIIS_START_CYCLE` where `DIIS_START_CYCLE = 0`
  (a `usize`), so the comparison is always true. Introduced by plan 06-05
  (`0225a47 feat(06-05): wire amplitude-DIIS into ccsd_kernel loop`).
  **RESOLVED in 06-08** (the executor was already editing `ccsd.rs` to wire the
  AO-direct `direct` flag): the start-cycle guard keeps its configurable `>=`
  semantics (raise `DIIS_START_CYCLE` → DIIS starts later) under a documented
  `#[allow(clippy::absurd_extreme_comparisons)]` with rationale. Confirmed gone
  via `cargo clippy -p pyscf-ccsd --tests | grep -i absurd` (no match).

## 06-09 execution (2026-05-25)

- **Pre-existing clippy `type_complexity` in `crates/pyscf-ccsd/tests/rdm.rs:39`**
  — `fn converged_lambda_state() -> (CcsdReference, ChemistsEris, Vec<f64>×4,
  WorkspacePool)` trips clippy's "very complex type used" under `-D warnings`.
  Verified verbatim on HEAD (a prior Wave-3 RDM test commit), NOT introduced by
  06-09 — my plan touched only `dfccsd.rs` + `dfccsd_spill.rs` + the ao2mo
  outcore surface. Out of scope per the SCOPE BOUNDARY rule (do not fix
  pre-existing failures in unrelated files). The 06-09 plan-touched targets
  (`pyscf-ccsd --lib`, `--test dfccsd_spill`, `pyscf-ao2mo --lib`) are
  clippy-clean under `-D warnings`. **Action for a future cleanup plan or the
  CI gate:** factor the tuple into a `struct LambdaState { .. }` or a
  `type LambdaState = (..)` alias to satisfy `type_complexity`.
