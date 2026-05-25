# Deferred Items — Phase 06 (ccsd)

Out-of-scope discoveries logged during execution (NOT fixed — they predate or
fall outside the current plan's file set).

## 06-06 execution (2026-05-25)

- **Pre-existing clippy `absurd_extreme_comparisons` in `crates/pyscf-ccsd/src/ccsd.rs:203`**
  — `Some(stack) if istep >= DIIS_START_CYCLE` where `DIIS_START_CYCLE = 0`
  (a `usize`), so the comparison is always true. Introduced by plan 06-05
  (`0225a47 feat(06-05): wire amplitude-DIIS into ccsd_kernel loop`). It is in
  `ccsd.rs`, which plan 06-06 does NOT modify (CRITICAL_PROJECT_CONSTRAINTS:
  "You do NOT modify ccsd.rs"). Out of scope — left for a 06-05 follow-up or a
  cleanup pass. `cargo test`/`cargo check` are unaffected (the lint only fires
  under `clippy -D warnings`).
