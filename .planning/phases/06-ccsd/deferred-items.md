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
