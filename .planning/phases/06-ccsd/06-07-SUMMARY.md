---
phase: 06-ccsd
plan: 07
subsystem: ccsd
tags: [ccsd, diagnostics, t1-diagnostic, d1-diagnostic, d2-diagnostic, frozen-core, eigh, oracle_sum]

# Dependency graph
requires:
  - phase: 06-03
    provides: ccsd_kernel + CcsdResult (the converged t1/t2 the diagnostics consume) + default_ao2mo (the frozen-aware ChemistsEris build)
  - phase: 05-03
    provides: the five MP2-08 frozen helpers (get_nocc/get_nmo/get_frozen_mask/get_e_hf/mo_without_core) + the Frozen enum — the verbatim cc/ccsd.py:35 import contract
  - phase: 03-11
    provides: pyscf_algebra::eigh_gen (faer self-adjoint eigendecomp, slice API) used for D1/D2
  - phase: 01-foundation
    provides: pyscf_algebra::oracle_sum (thread-invariant pairwise reduction)
provides:
  - "get_t1_diagnostic / get_d1_diagnostic / get_d2_diagnostic in pyscf-ccsd (port ccsd.py:748-776)"
  - "tests/diagnostics.rs — hand-computed Frobenius/eigh value checks (CCSD-09)"
  - "tests/frozen.rs — CCSD frozen int/list/auto active-space contract vs MP2 helpers (CCSD-10)"
affects: [06-08, ccsd-py-bridge, ccsd-diagnostics-byte-identity]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Diagnostics as pure free functions over the converged t1/t2 slices (no instance type, mirrors the MP2 helper-over-data discipline)"
    - "D1/D2 via eigh_gen(matrix, identity, n) — reuse the SCF generalized-eigh path with S=I as a plain symmetric eigh; max sqrt(|eigenvalue|) over the ij and ab Gram blocks"
    - "Frozen-core CONTRACT test (no new logic) — assert CCSD default_ao2mo active space == MP2 helper output for int/list/auto"

key-files:
  created:
    - crates/pyscf-ccsd/tests/diagnostics.rs
    - crates/pyscf-ccsd/tests/frozen.rs
  modified:
    - crates/pyscf-ccsd/src/diagnostics.rs
    - crates/pyscf-ccsd/src/lib.rs

key-decisions:
  - "get_t1_diagnostic takes nelec explicitly (not derived from t1.shape[0]) so the caller controls normalization; the value still equals upstream ||t1||_F / sqrt(2*nocc) when nelec = 2*nocc"
  - "D1/D2 port the FULL upstream definition: max over BOTH the ij and ab Gram blocks (they share nonzero eigenvalues, so the values coincide — done for exact parity, not just t1.t1^T)"
  - "max_sqrt_abs_eig skips eigh_gen's +inf linear-dep markers so a rank-deficient Gram block stays finite"
  - "Frozen::Auto through the CCSD path is ELEMENT-BLIND (count 0 == None) — default_ao2mo routes through pyscf_mp2::get_frozen_mask (the helper surface), which carries no charges; this is the verbatim cc/ccsd.py:35 reuse contract, and the test pins that identity rather than assuming Auto freezes chemcore"

patterns-established:
  - "Eigenvalue diagnostics: build symmetric Gram block as host-loop materialize-then-oracle_sum, diagonalize with eigh_gen against identity S"
  - "Frozen contract validation: compare kernel-path active space against the shared helper output (no duplicated frozen logic to drift)"

requirements-completed: [CCSD-09, CCSD-10]

# Metrics
duration: 5min
completed: 2026-05-25
---

# Phase 6 Plan 7: CCSD Diagnostics + Frozen-Core Contract Summary

**T1/D1/D2 wavefunction diagnostics (Frobenius + eigh-of-Gram, port ccsd.py:748-776) plus the CCSD-10 frozen-core contract proving CCSD int/list/auto reuse the MP2 helpers verbatim with no new frozen logic.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-05-25T02:48:16Z
- **Completed:** 2026-05-25T02:53:15Z
- **Tasks:** 2
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments

- Ported `get_t1_diagnostic` (Lee–Taylor `||t1||_F / sqrt(nelec)`), `get_d1_diagnostic` and `get_d2_diagnostic` (Janssen / Nielsen — `sqrt(max|eigenvalue|)` over the ij and ab Gram blocks) from `ccsd.py:748-776`.
- Every reduction routes through `oracle_sum` (thread-invariant, RAYON 1==8); the Gram blocks are host-loop materialize-then-`oracle_sum` (no `gemm`/`+=`), diagonalized via `pyscf_algebra::eigh_gen` against an identity metric.
- Validated CCSD-10 end-to-end: CCSD's frozen-aware active space (from `default_ao2mo`) equals the MP2 helper output for `Frozen::Count` / `Frozen::List` / `Frozen::Auto`, and a frozen-core CCSD `e_corr` rises toward zero vs the all-electron run (LiH/STO-3G, both converge).
- Shape-validated (`ShapeMismatch`, never OOB-panics) under `#![forbid(unsafe_code)]`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Port get_t1/d1/d2_diagnostic (CCSD-09)** — `0c7e510` (feat)
2. **Task 2: Frozen-core contract test (CCSD-10)** — `196847c` (test)

**Plan metadata:** (final docs commit — this SUMMARY + STATE + ROADMAP)

_Note: Task 1 was TDD (RED via the integration test against the stub → GREEN via the implementation). The test file and implementation are combined in one feat commit (sequential-mode single deliverable); the embedded lib unit tests + the integration test together pin the RED→GREEN cycle._

## Files Created/Modified

- `crates/pyscf-ccsd/src/diagnostics.rs` — `get_t1_diagnostic`/`get_d1_diagnostic`/`get_d2_diagnostic` + private `max_sqrt_abs_eig`/`identity` helpers (was a Wave-3 stub).
- `crates/pyscf-ccsd/src/lib.rs` — re-export the three diagnostics from the crate root.
- `crates/pyscf-ccsd/tests/diagnostics.rs` — CCSD-09 value checks against hand-computed Frobenius/eigh references (T1=2.5, D1=4, D1(off-diag)=√5, D2=2) + shape-mismatch + determinism.
- `crates/pyscf-ccsd/tests/frozen.rs` — CCSD-10 contract: count/list/auto active space == MP2 helpers; `mo_without_core` column drop; frozen-core `e_corr` direction; eris sized to active space.

## Decisions Made

- **`get_t1_diagnostic(t1, nocc, nvir, nelec)` signature** — `nelec` is passed explicitly rather than derived from `t1.shape[0]` (upstream `nelectron = 2*t1.shape[0]`). The value is identical when `nelec = 2*nocc`; passing it explicitly lets the caller control normalization (and keeps the function honest about its inputs). The test passes `nelec = 2*nocc` to match upstream exactly.
- **D1/D2 port the full upstream definition** (max over BOTH ij and ab Gram blocks). The two blocks (`A·Aᵀ` and `Aᵀ·A`) share nonzero eigenvalues so the values coincide mathematically; computing both is exact-parity fidelity to `ccsd.py:758-761,772-775`, not strictly necessary numerically.
- **D1/D2 reuse `eigh_gen` with `S = I`** — rather than adding a new symmetric-eigh entry point, the existing SCF generalized-eigh slice API is called with an identity overlap, which reduces to a plain symmetric eigh. `+inf` linear-dependency markers from `eigh_gen` are skipped so a rank-deficient Gram block stays finite.
- **`Frozen::Auto` is element-blind through the CCSD path** — `default_ao2mo` calls `pyscf_mp2::get_frozen_mask` (the data-only helper surface, which carries no atomic charges), so `Auto` freezes 0 orbitals == `None`. This IS the verbatim `cc/ccsd.py:35` reuse contract; the frozen test pins that identity rather than assuming Auto applies chemcore (the MP2 numeric kernel resolves chemcore with real charges via `frozen_mask` directly — a separate path, out of scope for the CCSD helper contract).

## Deviations from Plan

None - plan executed exactly as written. Both tasks landed mechanically (diagnostics = Frobenius/eigh on the converged amplitudes; frozen = pure reuse with a contract test). No bugs, no missing critical functionality, no blocking issues, no architectural changes. `ccsd.rs` was NOT modified (the kernel already routes frozen through the helpers since 06-03).

## Issues Encountered

- **Choosing a converging multi-occ frozen test reference.** H2/STO-3G (the existing CCSD smoke molecule) has only 1 occupied orbital, so freezing it leaves nothing to correlate. A throwaway probe over candidates found **LiH/STO-3G** (nocc=2, nmo=6) converges for both the all-electron and `Frozen::Count(1)` runs with the expected `e_corr` direction (all=-0.0204491, frozen=-0.0202318). Larger systems (H2O, HF/STO-3G) error in the all-electron `vvvv` transform (a larger-`nvir` int2e shape issue, pre-existing and out of scope) but their frozen paths converge — confirming the frozen subset build is sound. The probe was deleted before commit; LiH is used in the final test.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CCSD-09 and CCSD-10 are landed in-tree and green (`cargo test -p pyscf-ccsd --test diagnostics --test frozen`).
- The diagnostics free functions are ready for the 06-08 PyO3 bridge to expose as `mycc.t1diagnostic()` / `mycc.d1diagnostic()`, and the diagnostic byte-identity-vs-upstream check is the 06-08 `workflow_dispatch` human-verify arm.
- Remaining Phase-6 plans: the DF/AO-direct CCSD paths (Wave 4 spill) and the PyO3 bridge.

## Self-Check: PASSED

- FOUND: crates/pyscf-ccsd/src/diagnostics.rs
- FOUND: crates/pyscf-ccsd/tests/diagnostics.rs
- FOUND: crates/pyscf-ccsd/tests/frozen.rs
- FOUND: .planning/phases/06-ccsd/06-07-SUMMARY.md
- FOUND commit: 0c7e510 (Task 1 — diagnostics, CCSD-09)
- FOUND commit: 196847c (Task 2 — frozen contract, CCSD-10)

---
*Phase: 06-ccsd*
*Completed: 2026-05-25*
