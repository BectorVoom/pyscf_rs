---
status: partial
phase: 06-ccsd
source: [06-VERIFICATION.md]
started: 2026-05-25T05:00:00Z
updated: 2026-05-25T05:00:00Z
---

## Current Test

[awaiting human testing — all require live upstream PySCF / maturin / python3.13t, which the dev sandbox does not provide; gated as `workflow_dispatch` CI arms per 06-CONTEXT D-04]

## Tests

### 1. Caffeine/cc-pVDZ RCCSD upstream byte-identity
expected: `cc.RCCSD(mf).kernel()` on caffeine/cc-pVDZ returns e_corr matching upstream PySCF to ≤1 µHartree within PYSCF_MAX_MEMORY, no OOM. Run the `ccsd-oracle-upstream-manual` CI arm (workflow_dispatch).
result: [pending]
risk: UPDATED 2026-05-25 after investigation. 06-07's "larger-nvir vvvv int2e shape error" was a misdiagnosis. Root cause: cintx one_electron.rs nuclear-attraction Rys-roots cap (≤2 roots) panicked on d-functions during the RHF SCF build — FIXED in cintx commit 13fe9d3 (now dispatches through the general rys_roots_host). REMAINING blocker: after the fix, H2O/cc-pVDZ RHF computes integrals but SCF does not converge (|ΔE|≈3.6e-2 / 50 cycles) — a separate SCF-robustness or d-function integral-accuracy issue (out of phase-6 scope). caffeine/cc-pVDZ in-core therefore needs that SCF-convergence issue resolved first (and is compute-heavy regardless). CCSD itself is correct: H2/cc-pVDZ (nvir=9) converges, small systems match references.

### 2. Benzene-dimer DF-CCSD constrained-budget spill proof
expected: `PYSCF_MAX_MEMORY=500` DF-CCSD on benzene-dimer/cc-pVDZ spills the vvL/Wabef intermediate to an HDF5 temp file and RAII-deletes it (no leftover scratch); converges. Run `ccsd-oracle-upstream-manual`. (In-tree: synthetic-B spill + RAII drop-delete already proven, dfccsd_spill 5/5 parallel.)
result: [pending]

### 3. Lambda / RDM upstream byte-identity
expected: `mycc.solve_lambda()` l1/l2 and `make_rdm1()`/`make_rdm2()` (incl. ao_repr) match upstream PySCF. Run `ccsd-oracle-upstream-manual --features python`. (In-tree: closed-shell λ/RDM numeric, Tr(rdm1)=nelec asserted; open-shell λ/RDM deferred to Phase 7.)
result: [pending]

### 4. python3.13t free-threaded GIL smoke
expected: `mf.CCSD().run()` under python3.13t (free-threading) completes with no GIL deadlock (Pitfall 6 — CCSD is the heaviest re-validation). Run `python313t-ccsd-smoke` (workflow_dispatch).
result: [pending]

### 5. Genuine UHF-reference UCCSD byte-identity
expected: `cc.UCCSD(uhf_mf).kernel()` on a true open-shell UHF reference matches upstream. (In-tree: UCCSD(α=β)==RCCSD bit-identical + asymmetric synthetic-RHF arm converges; a real α/β UHF SCF loop is plan 03-11, currently incomplete.)
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0
blocked: 0

## Gaps
