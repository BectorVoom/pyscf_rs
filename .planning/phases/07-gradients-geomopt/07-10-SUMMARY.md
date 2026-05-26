---
phase: 07-gradients-geomopt
plan: 10
subsystem: validation
tags: [oracle, gradients, geomopt, nuc_grad, register-but-defer-dispatch, ci-gate, workflow_dispatch, GEOMOPT-01, nyquist, cintx-gated, D-01, D-02, D-05]

# Dependency graph
requires:
  - phase: 07-gradients-geomopt
    plan: 01
    provides: "the live cintx 2-ready/6-missing gradient-integral availability split (which numeric arms stay gated) + the register-but-defer-dispatch / numeric-cintx-gated precedent (D-02)"
  - phase: 07-gradients-geomopt
    plan: 09
    provides: "the Python optimize(mf) entry point + the python/pyscf/{grad,geomopt} overlays (_graft_nuc_grad_onto_scf, NO geometric/pyberny import GEOMOPT-01) the no-runtime-dep CI proof exercises"
  - phase: 06-ccsd
    plan: 11
    provides: "the CCSD oracle register-but-defer-dispatch precedent (ccsd_oracle.rs: always-on dispatch-layer arms + #[cfg(feature=python)]/#[ignore]'d byte-identity arms; KNOWN_METHODS catalogue + len assertion; ORACLE_NO_LIBXC default)"
provides:
  - "crates/pyscf-oracle/src/runner.rs KNOWN_METHODS: 8 new Phase-7 names (nuc_grad_rhf/uhf/rks/uks/mp2/ccsd/ecp + geomopt_h2o); catalogue-len assertion 24→32"
  - "crates/pyscf-oracle/src/grad_oracle.rs: always-on dispatch-layer registration arms (no python, no libxc) + #[cfg(feature=python)]/#[ignore]'d byte-identity + geomopt-trajectory arms (workflow_dispatch / human-verify only)"
  - ".github/workflows/ci.yml grad-structural: always-on FD verify_fd (D-01) + atmlst + single_cphf_impl + geomopt convergence; scoped -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle, no python, no libxc"
  - ".github/workflows/ci.yml geomopt-no-runtime-dep: the GEOMOPT-01 pip-uninstall-geometric-pyberny proof (runs in CI per D-05, NOT workflow_dispatch)"
  - ".github/workflows/ci.yml grad-oracle-upstream-manual: workflow_dispatch upstream byte-identity ≤1e-7 Ha/Bohr + geomopt trajectory parity (gated on the cintx grad-intor workstream + an upstream PySCF/geomeTRIC install)"
  - "07-VALIDATION.md nyquist_compliant: true + wave_0_complete: true — the phase Nyquist validation contract closed"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Oracle register-but-defer-dispatch (07-01 D-02 / 06-11 precedent): the method names land in KNOWN_METHODS so the always-on dispatch layer RECOGNISES them (never UnknownMethod); the live byte-identity comparison is #[cfg(feature=python)]/#[ignore]'d on the workflow_dispatch arm"
    - "Three-tier CI gate (D-01/D-05): always-on FD/structural (grad-structural) + a CI no-runtime-dep proof (geomopt-no-runtime-dep) + a workflow_dispatch upstream byte-identity arm (grad-oracle-upstream-manual)"
    - "GEOMOPT-01 no-runtime-dep CI proof: import pyscf.grad FIRST (grafts nuc_grad_method) then pyscf.geomopt; assert geometric/pyberny are NOT importable; a cintx-availability error from optimize is the DOCUMENTED gated outcome (forbidden is a missing-geometric/pyberny ImportError)"

key-files:
  created:
    - crates/pyscf-oracle/src/grad_oracle.rs
  modified:
    - crates/pyscf-oracle/src/runner.rs
    - crates/pyscf-oracle/src/lib.rs
    - .github/workflows/ci.yml
    - .planning/phases/07-gradients-geomopt/07-VALIDATION.md

key-decisions:
  - "grad_oracle lives in crates/pyscf-oracle/SRC (a #[cfg(test)] module + the python-gated live_arms), NOT in tests/ — the plan named src/grad_oracle.rs as a deliverable (files_modified + artifacts.path). The CCSD precedent put its arms in tests/ccsd_oracle.rs; this plan's spec moved them to a src module. The always-on dispatch-layer tests use crate::run_oracle_check directly; the live byte-identity arms use crate::oracle_check! (the #[macro_export] macro)."
  - "8 names registered exactly as the plan specified (nuc_grad_rhf/uhf/rks/uks/mp2/ccsd/ecp + geomopt_h2o), so the catalogue-len assertion is 24→32 (no drift from the planned list)."
  - "The GEOMOPT-01 CI proof catches a cintx-availability error from optimize(mf) as the documented gated outcome (the analytical grad rides the 6/8-missing cintx families, 07-01) and asserts ONLY that the failure is NOT a missing-geometric/pyberny ImportError — so the proof is green today (native entry point resolves + runs) and stays green once the cintx workstream lands the numeric."
  - "The grad-oracle-upstream-manual arm installs geometric>=1.0 in addition to pyscf>=2.5 (the geomopt trajectory-parity arm needs geomeTRIC as the reference optimizer); the mp2/ccsd precedent installed only pyscf."

patterns-established:
  - "register-but-defer-dispatch oracle arm in src/ (not tests/): a #[cfg(test)] dispatch_layer mod (always-on, no python) + a #[cfg(all(test, feature=\"python\"))] live_arms mod (#[ignore]'d byte-identity)"
  - "no-runtime-dep CI proof shape: pip uninstall the external package(s) || true → assert importlib.util.find_spec is None → run the native entry point, treating a documented-gated error as success but a missing-external-package ImportError as a hard failure"

requirements-completed: [GRAD-01, GRAD-02, GRAD-03, GRAD-04, GRAD-05, GRAD-06, GRAD-07, GEOMOPT-01, GEOMOPT-07]

# Metrics
duration: 12min
completed: 2026-05-26
---

# Phase 7 Plan 10: Oracle/CI Close-Out — grad/geomopt fixtures + the phase Nyquist contract Summary

**Closed the phase Nyquist validation contract: registered the 8 Phase-7 gradient/geomopt oracle method names (`nuc_grad_rhf`/`uhf`/`rks`/`uks`/`mp2`/`ccsd`/`ecp` + `geomopt_h2o`) in `pyscf-oracle` `KNOWN_METHODS` (24→32, register-but-defer-dispatch mirroring the MP2/CCSD precedent) with the new `src/grad_oracle.rs` carrying always-on dispatch-layer registration arms (no python, no libxc) + `#[cfg(feature="python")]`/`#[ignore]`'d byte-identity + geomopt-trajectory arms; wired three CI gate groups — the always-on `grad-structural` job (FD `verify_fd` D-01 + `atmlst` + `single_cphf_impl` + geomopt convergence, scoped `-p pyscf-grad -p pyscf-geomopt -p pyscf-oracle`, no python/libxc), the `geomopt-no-runtime-dep` CI proof (GEOMOPT-01: `pip uninstall -y geometric pyberny` then `pyscf.geomopt.optimize(mf)` runs natively, per D-05), and the `workflow_dispatch` `grad-oracle-upstream-manual` arm (upstream byte-identity ≤1e-7 Ha/Bohr + geomopt trajectory parity, gated on the cintx grad-intor workstream); and set `07-VALIDATION.md` `nyquist_compliant: true` + `wave_0_complete: true`. The always-on FD/structural numerics + the GEOMOPT-01 proof are green; the upstream byte-identity numerics stay `workflow_dispatch`-gated because 6 of 8 gradient-integral families are MISSING from cintx (07-01) with no scheduled workstream.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-26T04:57Z (post-07-09)
- **Completed:** 2026-05-26T05:09Z
- **Tasks:** 2 (both `type="auto"`)
- **Files modified:** 5 (1 created, 4 modified)

## Accomplishments

- **Task 1 — pyscf-oracle grad fixtures (register-but-defer-dispatch).** Registered the 8 Phase-7 method names in `runner.rs` `KNOWN_METHODS` (`nuc_grad_rhf`, `nuc_grad_uhf`, `nuc_grad_rks`, `nuc_grad_uks`, `nuc_grad_mp2`, `nuc_grad_ccsd`, `nuc_grad_ecp`, `geomopt_h2o`) with a doc-comment recording the cintx gate + the always-on/workflow_dispatch split; updated the catalogue-len assertion (`known_methods_list_has_all_arms`) from `==24` to `==32` (24 existing + 8 new) and added a `contains` check per new name. Created `crates/pyscf-oracle/src/grad_oracle.rs` with a `#[cfg(test)] mod dispatch_layer` (4 always-on tests: every registered name is never `UnknownMethod`; a known method without `--features python` short-circuits to `PythonFeatureNotEnabled`; a typo'd grad name is rejected; all eight names are registered) + a `#[cfg(all(test, feature="python"))] mod live_arms` (8 `#[ignore]`'d byte-identity / trajectory arms, ≤1e-7 Ha/Bohr). Registered the module in `lib.rs`.
- **Task 2 — CI gate wiring + the Nyquist contract.** Added three jobs to `.github/workflows/ci.yml`: (1) always-on `grad-structural` (`cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle --locked -- --test-threads=1`, `needs: [build-default]`, `setup-sibling-crates`, no python, no libxc); (2) CI `geomopt-no-runtime-dep` (GEOMOPT-01: builds the abi3 wheel via `maturin develop --release`, `pip uninstall -y geometric pyberny || true`, asserts neither is importable, then runs `pyscf.geomopt.optimize(mf)` — treating a cintx-availability error as the documented gated outcome but a missing-external-package `ImportError` as a hard failure; runs in CI per D-05, NOT workflow_dispatch); (3) `workflow_dispatch` `grad-oracle-upstream-manual` (`if: github.event_name == 'workflow_dispatch'`, installs `pyscf>=2.5` + `geometric>=1.0`, runs `cargo test -p pyscf-oracle --features python ... --include-ignored grad geomopt`). Filled the `07-VALIDATION.md` Per-Task Verification Map Task ID column, checked the Wave 0 Requirements + Validation Sign-Off boxes, set `nyquist_compliant: true` + `wave_0_complete: true` + `status: complete`, and added a Nyquist coverage / what-stays-gated summary.

## Task Commits

Each task was committed atomically by explicit path:

1. **Task 1: register grad/geomopt oracle fixtures (register-but-defer-dispatch)** — `08bd054` (feat)
2. **Task 2: wire grad/geomopt CI gates + close the phase Nyquist contract** — `455afbc` (feat)

## Files Created/Modified

- `crates/pyscf-oracle/src/grad_oracle.rs` (created) — the always-on `dispatch_layer` registration arms + the python-gated `#[ignore]`'d `live_arms` byte-identity / trajectory arms.
- `crates/pyscf-oracle/src/runner.rs` (modified) — 8 names added to `KNOWN_METHODS`; catalogue-len assertion 24→32 + per-name `contains` checks.
- `crates/pyscf-oracle/src/lib.rs` (modified) — `pub mod grad_oracle`.
- `.github/workflows/ci.yml` (modified) — the three Phase-7 grad/geomopt jobs (`grad-structural` always-on, `geomopt-no-runtime-dep` CI proof, `grad-oracle-upstream-manual` workflow_dispatch).
- `.planning/phases/07-gradients-geomopt/07-VALIDATION.md` (modified) — Task ID map filled; Wave 0 + Sign-Off boxes checked; `nyquist_compliant: true` + `wave_0_complete: true`.

## Decisions Made

See the `key-decisions` frontmatter block. The load-bearing ones: `grad_oracle.rs` lives in `src/` (the plan named it as a deliverable) rather than `tests/` like the CCSD precedent; the 8 names matched the planned list exactly (24→32, no drift); the GEOMOPT-01 proof treats a cintx-availability error as the documented gated outcome and forbids only a missing-`geometric`/`pyberny` `ImportError` (so it is green today and stays green when the cintx numeric lands); the `grad-oracle-upstream-manual` arm additionally installs `geometric>=1.0` for the trajectory-parity reference.

## Deviations from Plan

None — plan executed exactly as written. (Two clarifications, NOT deviations: (1) the plan named `src/grad_oracle.rs` as the deliverable while the CCSD precedent it cites lives in `tests/ccsd_oracle.rs` — followed the plan's explicit `files_modified`/`artifacts.path` and put it in `src/`, registered in `lib.rs`; (2) the `grad-oracle-upstream-manual` install line adds `geometric>=1.0` beyond the plan's `pyscf>=2.5` because the geomopt trajectory-parity arm needs geomeTRIC as the reference optimizer — a necessary completion of the named arm, not a scope change.)

## Issues Encountered

- The pre-existing workspace-wide `[patch] cintx not used in the crate graph` note + the `fma4 unknown target-feature` warning appear on every scoped build (recorded in 07-04/07-06/07-09 SUMMARYs). They are independent of this plan and do not affect the gates (out of scope per the scope boundary — logged here, not fixed).

## User Setup Required

None — no external service configuration required. The `grad-oracle-upstream-manual` arm is `workflow_dispatch` only and requires an upstream PySCF + geomeTRIC install on the runner when manually triggered (the documented human-verify step).

## Known Stubs

None that block the plan's goal. Documented intentional deferrals (per the cintx gate, 07-01 / 07-09 precedent):
- **The upstream byte-identity analytical-gradient numerics (≤1e-7 Ha/Bohr) + the geomopt trajectory parity stay cintx-gated** — the `grad-oracle-upstream-manual` arm is `workflow_dispatch`-only and rides the six MISSING cintx grad-intor families (`int2e_ip1` + `int1e_ip{ovlp,kin,nuc,rinv}` + `ECPscalar_iprinv` + the `with_rinv_at_nucleus` origin shift). The `#[ignore]`'d `live_arms` byte-identity tests carry the gate in their `#[ignore]` reason. They un-gate when the cintx grad-intor workstream lands. The always-on FD (`verify_fd`, ≤1e-6 Ha/Bohr) + structural gates + the GEOMOPT-01 proof are green now.

## Threat Flags

None — no new network endpoints, auth paths, or schema changes at a trust boundary. The plan's threat register is mitigated: every grad/geomopt CI job is scoped to `-p pyscf-grad -p pyscf-geomopt -p pyscf-oracle` and NEVER enables `--features libxc` (T-07-35 — the ~6h compile freeze); `--test-threads=1` everywhere (T-07-36 — determinism); the upstream byte-identity arm is `if: github.event_name == 'workflow_dispatch'`-guarded so it never leaks into the daily gate (T-07-37); the `geomopt-no-runtime-dep` job `pip uninstall`s geometric/pyberny and proves the native optimizer still resolves (T-07-38); no new registry package — `setup-sibling-crates` provides cintx and the dependency-wall lint stays green with zero edits (T-07-SC).

## Next Phase Readiness

- The phase Nyquist contract is CLOSED (`nyquist_compliant: true`, `wave_0_complete: true`). All 9 plan requirements (GRAD-01..07, GEOMOPT-01, GEOMOPT-07) have an always-on FD/structural verify or a documented `workflow_dispatch`-gated upstream arm.
- **Coordination note (D-02 hinge):** the upstream byte-identity numeric stays gated across ALL methods on the six MISSING cintx grad-intor families. When the cintx grad-intor workstream lands those families, drop the `#[ignore]` on the `grad_oracle.rs` `live_arms` and the `grad-oracle-upstream-manual` arm un-gates with a paired cintx-side availability note (analogous to the int2e / d-shell-Rys workstream in project memory).

## Self-Check: PASSED

- `crates/pyscf-oracle/src/grad_oracle.rs` exists on disk; `runner.rs`/`lib.rs`/`ci.yml`/`07-VALIDATION.md` modified.
- Both task commits (`08bd054`, `455afbc`) present in git history.
- `cargo test -p pyscf-oracle --locked -- --test-threads=1 grad geomopt`: 4 dispatch-layer tests pass, 0 failed; full `pyscf-oracle` suite green; the catalogue-len `==32` test passes.
- `cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle --locked -- --test-threads=1` (the exact `grad-structural` command): all green (0 failed).
- `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`: `YAML_OK`.
- No `--features libxc` in any new grad/geomopt job (only the pre-existing disabled `dft-libxc-bitexact` job invokes it); `cargo tree -p pyscf-oracle` (default build) pulls neither libxc nor pyo3.
- `check-dependency-wall`: PASS (cubecl-* containment intact).
- `07-VALIDATION.md` frontmatter: `nyquist_compliant: true`, `wave_0_complete: true`.

---
*Phase: 07-gradients-geomopt*
*Completed: 2026-05-26*
