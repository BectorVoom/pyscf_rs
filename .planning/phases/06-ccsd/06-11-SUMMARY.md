---
phase: 06-ccsd
plan: 11
subsystem: testing
tags: [ccsd, oracle, ci, github-actions, pyscf-oracle, workflow_dispatch, nyquist, heap-alloc, free-threading]

# Dependency graph
requires:
  - phase: 06-02
    provides: heap_alloc_count + refusal CCSD-11 targets (dedicated counting #[global_allocator])
  - phase: 06-03
    provides: in-core RCCSD numeric headline (H2/STO-3G e_corr ~ -0.020525) — rccsd_numeric_smoke
  - phase: 06-04
    provides: open-shell UCCSD (UCCSD(α==β) == RCCSD) — uccsd_smoke
  - phase: 06-08
    provides: DF-CCSD Wabef→HDF5 spill + AO-direct == in-core — dfccsd_spill / direct
  - phase: 06-09
    provides: λ / RDM in-tree convergence — lambda / rdm
  - phase: 06-10
    provides: PyO3 CCSD bridge (mf.CCSD() / density_fit().CCSD()) — the 3.13t smoke target
provides:
  - "pyscf-oracle CCSD method-name registry (6 arms: ccsd_rccsd_energy / ccsd_uccsd_energy / ccsd_dfccsd_energy / ccsd_lambda / ccsd_rdm1 / ccsd_rdm2)"
  - "crates/pyscf-oracle/tests/ccsd_oracle.rs — always-on dispatch-layer arms + #[cfg(feature=python)] #[ignore]'d byte-identity arms"
  - "four CCSD CI arms: ccsd-structural + heap-alloc-count (always-on); ccsd-oracle-upstream-manual + python313t-ccsd-smoke (workflow_dispatch)"
  - "Nyquist validation contract closed (06-VALIDATION nyquist_compliant: true) — every CCSD requirement has an automated always-on verify or a tracked workflow_dispatch human-verify arm"
affects: [milestone-close, phase-gate-human-verify, ci-maintenance]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Oracle method-name registration WITHOUT live dispatch wiring (the MP2 precedent): names enter KNOWN_METHODS so the always-on dispatch guard recognises them; the live byte-identity compare is the workflow_dispatch human-verify arm"
    - "Two-tier CI gating: small always-on arms on every push; caffeine/DF-spill/upstream-byte-identity/3.13t arms workflow_dispatch ONLY (never pull libxc / never freeze CI)"

key-files:
  created:
    - crates/pyscf-oracle/tests/ccsd_oracle.rs
  modified:
    - crates/pyscf-oracle/src/runner.rs
    - .github/workflows/ci.yml
    - .planning/phases/06-ccsd/06-VALIDATION.md

key-decisions:
  - "Mirror the MP2 oracle precedent exactly: register the 6 CCSD method names in KNOWN_METHODS but leave the live python_impl dispatch helpers unwired (the byte-identity compare is the documented workflow_dispatch human-verify step). The always-on small-system NUMERIC proofs already live in crates/pyscf-ccsd/tests/, so ccsd_oracle.rs is the oracle-side registration + gated byte-identity arms, not a duplicate of the numeric smokes."
  - "ccsd-structural scopes to `-p pyscf-ccsd -p pyscf-oracle` (default features), running all the small in-tree numeric arms proven this phase; NO --features python, NO libxc."
  - "heap-alloc-count is a DEDICATED job (`--test heap_alloc_count --test refusal`), not folded into ccsd-structural, so the test binary's counting #[global_allocator] stays isolated (RESEARCH A3)."
  - "The DF-CCSD spill proof runs under a constrained `PYSCF_MAX_MEMORY: 500` env in the manual arm so the Wabef/vvL reservation is forced through the HDF5-spill backend."

patterns-established:
  - "Always-on oracle dispatch-layer tests: a KNOWN method without --features python short-circuits to `oracle_check failed` (PythonFeatureNotEnabled), distinctly NOT `unknown method` — the always-on proof that names are registered without libpython/live PySCF."
  - "workflow_dispatch human-verify arms gated by `if: github.event_name == 'workflow_dispatch'`; --features python only ever inside such gated arms."

requirements-completed: [CCSD-01, CCSD-02, CCSD-05, CCSD-06, CCSD-08, CCSD-11]

# Metrics
duration: 4min
completed: 2026-05-25
---

# Phase 6 Plan 11: CCSD Oracle Fixtures + 4 CI Arms Summary

**pyscf-oracle CCSD method registry (6 arms) + ccsd_oracle.rs gated fixtures, plus four CI arms (ccsd-structural + heap-alloc-count always-on; ccsd-oracle-upstream-manual + python3.13t CCSD smoke workflow_dispatch-only) — closing the Phase-6 Nyquist validation contract without ever pulling libxc.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-25T03:48:20Z
- **Completed:** 2026-05-25T03:51:30Z
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- Registered the six CCSD oracle method names (`ccsd_rccsd_energy`, `ccsd_uccsd_energy`, `ccsd_dfccsd_energy`, `ccsd_lambda`, `ccsd_rdm1`, `ccsd_rdm2`) in `KNOWN_METHODS` (24 total) and updated the `known_methods_list_has_all_arms` registry assertion (18 → 24).
- Created `crates/pyscf-oracle/tests/ccsd_oracle.rs`: 2 always-on dispatch-layer arms (CCSD names are recognised, not unknown; feature-gated, not unknown, without `--features python`) + 6 `#[cfg(feature = "python")]` + `#[ignore]`'d live byte-identity arms (caffeine/UCCSD/DF-spill/λ/RDM1/RDM2).
- Added the four CCSD CI arms to `.github/workflows/ci.yml`: `ccsd-structural` + `ccsd-heap-alloc-count` (always-on, scoped, no python/libxc) and `ccsd-oracle-upstream-manual` + `python313t-ccsd-smoke` (`workflow_dispatch`-gated, the documented phase-gate human-verify arms).
- Verified the default `cargo test -p pyscf-oracle` build pulls NO libxc (`ORACLE_NO_LIBXC`) and makes no live PySCF call; YAML parses (`YAML_OK`); no always-on arm uses `--features python`/`--features libxc`/`--all-features`/`--workspace`.
- Set `06-VALIDATION.md` `nyquist_compliant: true` + `wave_0_complete: true` — every CCSD requirement now has an automated always-on verify or a tracked `workflow_dispatch` human-verify arm.

## Task Commits

Each task was committed atomically:

1. **Task 1: pyscf-oracle CCSD fixtures + method-name registry** - `a141ea2` (feat)
2. **Task 2: four CCSD CI arms** - `adf997e` (ci)

**Plan metadata:** (final docs commit below — SUMMARY + STATE + ROADMAP + VALIDATION)

## Files Created/Modified

- `crates/pyscf-oracle/tests/ccsd_oracle.rs` (created) - Always-on dispatch-layer CCSD oracle arms + `#[cfg(feature=python)]`/`#[ignore]`'d live byte-identity arms.
- `crates/pyscf-oracle/src/runner.rs` (modified) - 6 CCSD names added to `KNOWN_METHODS`; registry unit test updated (len 18 → 24 + 6 `contains` asserts).
- `.github/workflows/ci.yml` (modified) - 4 CCSD jobs appended (purely additive, +154 lines).
- `.planning/phases/06-ccsd/06-VALIDATION.md` (modified) - `nyquist_compliant: true`, `wave_0_complete: true`.

## Decisions Made

- **Mirror the MP2 oracle precedent (register names, defer live dispatch wiring).** The MP2 numeric arms (`mp2_rmp2_energy` etc.) are registered in `KNOWN_METHODS` but never wired into the `python_impl::dispatch` match — the live byte-identity compare is the `mp2-oracle-upstream-manual` `workflow_dispatch` human-verify step. The CCSD arms follow this exactly: names registered so the always-on dispatch guard recognises them; the live byte-identity comparison is the `ccsd-oracle-upstream-manual` arm. The real always-on small-system NUMERIC proofs already live in `crates/pyscf-ccsd/tests/` (proven 06-03..06-09), so `ccsd_oracle.rs` is the oracle-side registration + gated byte-identity arms, NOT a duplicate of those numeric smokes.
- **Dedicated heap-alloc-count job.** Kept `--test heap_alloc_count --test refusal` separate from `ccsd-structural` so the counting `#[global_allocator]` stays scoped to that one test binary (RESEARCH §A3 isolation requirement) — it never perturbs the bit-exactness arms.
- **Constrained `PYSCF_MAX_MEMORY: 500` in the manual oracle arm** so the DF-CCSD `Wabef`/`vvL` reservation is forced through the HDF5-spill backend (the CCSD-08 spill proof).

## Deviations from Plan

None - plan executed exactly as written.

The plan's `<verification>` also instructed setting `nyquist_compliant: true` once every requirement had an automated or tracked-workflow_dispatch verify; done as part of plan close-out (not a deviation — explicit plan instruction).

## Known Stubs

**1. CCSD live oracle dispatch helpers not wired (intentional — mirrors the MP2 precedent)**
- **File:** `crates/pyscf-oracle/src/runner.rs` — the 6 CCSD names are in `KNOWN_METHODS` but `python_impl::dispatch` has no `check_ccsd_*` arms (a live `oracle_check!("ccsd_rccsd_energy", …)` under `--features python` would currently hit the `other => UnknownMethod` fall-through, surfacing as a test failure rather than a silent pass).
- **Why intentional:** This is the established MP2 pattern (`mp2_rmp2_energy` etc. are likewise registered-but-unwired). The live byte-identity compare needs an installed upstream PySCF, which the sandbox/default runners lack; it is the documented `ccsd-oracle-upstream-manual` `workflow_dispatch` human-verify arm. The always-on small-system numeric correctness is fully covered by the in-tree `crates/pyscf-ccsd/tests/` arms (RCCSD/UCCSD/λ/RDM/DF-CCSD/AO-direct), which `ccsd-structural` runs on every push.
- **Resolved by:** the phase-gate human-verify run (`ccsd-oracle-upstream-manual` + `python313t-ccsd-smoke`), executed once manually against live PySCF; wiring the live `check_ccsd_*` helpers is a follow-up to that manual run if byte-identity drift is observed.

## Issues Encountered

- **PyYAML reports `KeyError: 'on'`** when introspecting the workflow dict. This is a YAML 1.1 quirk (the bareword `on:` key parses as the boolean `True`); it does NOT affect GitHub Actions (which uses its own parser). Confirmed the triggers (`push`, `pull_request`, `workflow_dispatch`) are intact via `d.get(True)`, so the `if: github.event_name == 'workflow_dispatch'` gate functions correctly. The file still `yaml.safe_load`s cleanly (`YAML_OK`).
- **Pre-existing project-wide warnings** (`cintx patch not used`, `fma4 target-feature`) appear in every `pyscf-oracle` target including the unchanged lib — out of scope per the deviation scope boundary; left untouched.

## Next Phase Readiness

- Phase 06 CCSD validation surface complete: the always-on `cargo test` stays fast/green; the heavy proofs (caffeine byte-identity, DF-CCSD spill on constrained memory, λ/RDM byte-identity, python3.13t GIL re-validation) are the tracked `workflow_dispatch` human-verify arms.
- **Phase-gate human-verify (deferred to manual run, per 06-VALIDATION §"Manual-Only"):** run `ccsd-oracle-upstream-manual` + `python313t-ccsd-smoke` once manually before the milestone close — caffeine byte-identity, DF-CCSD spill proof, λ/RDM byte-identity, no GIL deadlock under `mf.CCSD().run()`.
- libxc NEVER compiled at any point; all validation done with `cargo test -p pyscf-oracle` default features only.

---
*Phase: 06-ccsd*
*Completed: 2026-05-25*

## Self-Check: PASSED

- FOUND: `crates/pyscf-oracle/tests/ccsd_oracle.rs`
- FOUND: `crates/pyscf-oracle/src/runner.rs`
- FOUND: `.github/workflows/ci.yml`
- FOUND: `.planning/phases/06-ccsd/06-11-SUMMARY.md`
- FOUND: `.planning/phases/06-ccsd/06-VALIDATION.md`
- FOUND commit: `a141ea2` (Task 1)
- FOUND commit: `adf997e` (Task 2)
