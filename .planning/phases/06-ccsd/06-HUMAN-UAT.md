---
status: partial
phase: 06-ccsd
source: [06-VERIFICATION.md]
started: 2026-05-25T05:00:00Z
updated: 2026-05-25T04:52:00Z
---

## Current Test

[testing paused — 5 items outstanding. CI dispatch on 2026-05-25 (run 26383745473)
could NOT execute any byte-identity arm: the workspace fails at manifest-load in CI
because `[patch.crates-io]` points cintx/libxc_rs/xcfun_rs at local sibling repos
(`../cintx`, `../libxc_rs`, `../xcfun_rs/...`) that CI never checks out. This is a
repo-wide CI prerequisite (the documented "restore to git pins for CI parity" item,
Cargo.toml:100-110 / Plan 01-08), NOT a Phase-6 CCSD defect. Resolve the CI build,
then re-dispatch `ccsd-oracle-upstream-manual` + `python313t-ccsd-smoke`.]

## CI Run Log

- run_id: 26383745473
  run_url: https://github.com/BectorVoom/pyscf_rs/actions/runs/26383745473
  ref: fix/ci-local-gates (HEAD 6a9d1d0)
  event: workflow_dispatch
  dispatched: 2026-05-25T04:48:00Z
  result: failure (all 12 jobs red in 3m20s; heavy jobs skipped via `needs: build-default`)
  root_cause: |
    Every cargo invocation (build, fmt, clippy, deny, xtask checks, mp2/ccsd oracle
    arms) failed at workspace-manifest load:
      error: failed to load manifest for workspace member `crates/pyscf-rs`
        … failed to read `/home/runner/work/pyscf_rs/cintx/crates/cintx-compat/Cargo.toml`
        No such file or directory (os error 2)
    Cause: root Cargo.toml `[patch.crates-io]` uses local sibling-repo path deps
    (cintx = { path = "../cintx" }, libxc_rs = ../libxc_rs, xcfun_rs = ../xcfun_rs/...)
    — a deliberate, documented local-dev shortcut (Cargo.toml:100-110) that must be
    restored to git pins for CI parity. CI checks out only pyscf_rs (no .gitmodules,
    no sibling checkout), so manifest load fails before any test runs.
  not_run: ccsd-oracle-upstream-manual (tests 1,2,3,5) failed pre-test; python313t-ccsd-smoke (test 4) skipped (needs build-default).
  libxc_safety: dft-libxc-bitexact stayed `if: false` — no 6h libxc compile occurred.

## Tests

### 1. Caffeine/cc-pVDZ RCCSD upstream byte-identity
expected: `cc.RCCSD(mf).kernel()` on caffeine/cc-pVDZ returns e_corr matching upstream PySCF to ≤1 µHartree within PYSCF_MAX_MEMORY, no OOM. Run the `ccsd-oracle-upstream-manual` CI arm (workflow_dispatch).
result: blocked
blocked_by: ci-infra
reason: "CI run 26383745473 failed at workspace-manifest load (sibling path-dep `../cintx/crates/cintx-compat/Cargo.toml` absent in CI); the ccsd-oracle arm never executed. Pre-existing risk also stands: post-cintx-13fe9d3 H2O/cc-pVDZ RHF SCF non-convergence (out of phase-6 scope) gates caffeine in-core."
risk: UPDATED 2026-05-25 after investigation. 06-07's "larger-nvir vvvv int2e shape error" was a misdiagnosis. Root cause: cintx one_electron.rs nuclear-attraction Rys-roots cap (≤2 roots) panicked on d-functions during the RHF SCF build — FIXED in cintx commit 13fe9d3 (now dispatches through the general rys_roots_host). REMAINING blocker: after the fix, H2O/cc-pVDZ RHF computes integrals but SCF does not converge (|ΔE|≈3.6e-2 / 50 cycles) — a separate SCF-robustness or d-function integral-accuracy issue (out of phase-6 scope). caffeine/cc-pVDZ in-core therefore needs that SCF-convergence issue resolved first (and is compute-heavy regardless). CCSD itself is correct: H2/cc-pVDZ (nvir=9) converges, small systems match references.

### 2. Benzene-dimer DF-CCSD constrained-budget spill proof
expected: `PYSCF_MAX_MEMORY=500` DF-CCSD on benzene-dimer/cc-pVDZ spills the vvL/Wabef intermediate to an HDF5 temp file and RAII-deletes it (no leftover scratch); converges. Run `ccsd-oracle-upstream-manual`. (In-tree: synthetic-B spill + RAII drop-delete already proven, dfccsd_spill 5/5 parallel.)
result: blocked
blocked_by: ci-infra
reason: "Same as test 1 — ccsd-oracle-upstream-manual arm could not build in CI run 26383745473 (sibling path-dep repos not checked out)."

### 3. Lambda / RDM upstream byte-identity
expected: `mycc.solve_lambda()` l1/l2 and `make_rdm1()`/`make_rdm2()` (incl. ao_repr) match upstream PySCF. Run `ccsd-oracle-upstream-manual --features python`. (In-tree: closed-shell λ/RDM numeric, Tr(rdm1)=nelec asserted; open-shell λ/RDM deferred to Phase 7.)
result: blocked
blocked_by: ci-infra
reason: "Same as test 1 — ccsd-oracle-upstream-manual arm could not build in CI run 26383745473 (sibling path-dep repos not checked out)."

### 4. python3.13t free-threaded GIL smoke
expected: `mf.CCSD().run()` under python3.13t (free-threading) completes with no GIL deadlock (Pitfall 6 — CCSD is the heaviest re-validation). Run `python313t-ccsd-smoke` (workflow_dispatch).
result: blocked
blocked_by: ci-infra
reason: "python313t-ccsd-smoke was SKIPPED in CI run 26383745473 (`needs: build-default`, which failed at manifest load for the sibling path-dep reason). Arm never executed."

### 5. Genuine UHF-reference UCCSD byte-identity
expected: `cc.UCCSD(uhf_mf).kernel()` on a true open-shell UHF reference matches upstream. (In-tree: UCCSD(α=β)==RCCSD bit-identical + asymmetric synthetic-RHF arm converges; a real α/β UHF SCF loop is plan 03-11, currently incomplete.)
result: blocked
blocked_by: ci-infra
reason: "Same as test 1 — ccsd-oracle-upstream-manual arm could not build in CI run 26383745473. Additionally a genuine α/β UHF SCF loop (plan 03-11) is still incomplete."

## Summary

total: 5
passed: 0
issues: 0
pending: 0
skipped: 0
blocked: 5

## Gaps

<!-- No code gaps. All 5 items are blocked on a CI-infrastructure prerequisite
     (restore [patch.crates-io] git pins, or have CI check out the ../cintx,
     ../libxc_rs, ../xcfun_rs sibling repos) — a repo-wide gate, not a Phase-6
     CCSD code defect. Per verify-work protocol, blocked prerequisite gates do
     not feed /gsd:plan-phase --gaps. -->
