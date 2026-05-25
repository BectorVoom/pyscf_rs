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
  **RESOLVED during phase-6 verification (2026-05-25):** the verifier flagged
  this as the lone CI-blocking gap; the orchestrator added a documented
  `#[allow(clippy::type_complexity)]` on the single-use test fixture. Confirmed
  via `cargo clippy -p pyscf-ccsd -p pyscf-oracle -p pyscf-py --all-targets -- -D warnings`
  (EXIT 0, libxc never compiled).

## Phase-6 verification investigation (2026-05-25) — the "larger-nvir vvvv" finding

Plan 06-07 reported a "larger-`nvir` `vvvv` int2e shape error" on H2O/HF all-electron
runs. **This was a misdiagnosis.** Reproduced across 8 systems during phase-6
verification: CCSD handles large `nvir` fine (H2/cc-pVDZ nvir=9, vvvv=6561 converges;
H2O/STO-3G, HF/STO-3G converge). The real failure is in the SIBLING **cintx** repo:
`cintx-cubecl/src/kernels/one_electron.rs:322` panicked (out-of-bounds on a `[f64;2]`
Rys array) for `li+lj>=4` (d-functions on cc-pVDZ heavy atoms), during the **RHF SCF**
reference build — nothing to do with CCSD's vvvv/int2e path.

- **FIXED:** cintx commit `13fe9d3` (branch `fix/general-contraction-nctr-1e`) routes the
  nuclear path through the existing general `rys_roots_host` dispatcher. Panic gone;
  cintx tests green; bit-identical for s/p.
- **STILL OPEN (out of phase-6 scope — SCF/integral territory):** after the fix,
  H2O/cc-pVDZ RHF computes integrals but **SCF does not converge** (|ΔE|≈3.6e-2 after
  50 cycles). Likely d-function integral accuracy or SCF convergence-acceleration
  robustness; needs its own investigation. This (plus caffeine's compute weight) is why
  the caffeine/cc-pVDZ upstream byte-identity arm (06-HUMAN-UAT item 1) remains pending.
- **Conclusion:** Phase-6 CCSD is correct for its scope; the caffeine headline is blocked
  by upstream SCF/integral concerns, not by any CCSD defect.
