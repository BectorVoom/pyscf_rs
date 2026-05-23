---
phase: 05-mp2
plan: 01
subsystem: infra
tags: [mp2, ao2mo, workspace-scaffold, oracle, ci, algebra-wall]

# Dependency graph
requires:
  - phase: 03-scf
    provides: "pyscf-scf reference types, OverrideHooks pattern, oracle KNOWN_METHODS + python-feature gating model"
  - phase: 04-dft
    provides: "rks_energy/uks_energy cintx-gating precedent, libxc if:false disabled-CI-job pattern, From<Err> for PyscfRsError bridge precedent"
provides:
  - "pyscf-ao2mo crate (20th pyscf-* workspace member, D-01, AO→MO transform surface)"
  - "Ao2moError + Mp2Error enums with From<_> for PyscfRsError bridges"
  - "pyscf-mp2 dependency wiring (ao2mo/scf/df/gto/algebra/runtime, pyo3-free + cubecl-free)"
  - "pyscf-mp2 module skeleton (mp2/ump2/dfmp2/dfmp2_native/helpers/frozen/rdm/hooks/error)"
  - "the five MP2-08 helper signatures (CCSD cc/ccsd.py:35 import contract)"
  - "Wave-0 test scaffolds (ccsd_import_contract always-on, rmp2/ump2 structural, ao2mo transform_roundtrip)"
  - "5 MP2 numeric oracle arms in KNOWN_METHODS (mp2_rmp2_energy/mp2_ump2_energy/dfmp2_energy/dfmp2_native_energy/mp2_rdm)"
  - "CI mp2-structural (always-on) + mp2-oracle-cintx-gated (if:false) jobs"
affects: [05-02, 05-03, 05-04, 05-05, 05-06, 05-07, ccsd]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ao2mo/mp2 error enums mirror scf/error.rs From<_> for PyscfRsError bridge (route through Core(InvalidMolecule))"
    - "MP2 numeric oracle arms cintx#11-gated exactly like DF-HF/DFT-01 (structural always-on, numeric if:false)"
    - "Wave-0 always-on symbol-existence test arm + #[ignore]d numeric arm split"

key-files:
  created:
    - crates/pyscf-ao2mo/Cargo.toml
    - crates/pyscf-ao2mo/src/lib.rs
    - crates/pyscf-ao2mo/src/error.rs
    - crates/pyscf-ao2mo/src/incore.rs
    - crates/pyscf-ao2mo/src/transform.rs
    - crates/pyscf-ao2mo/tests/transform_roundtrip.rs
    - crates/pyscf-mp2/src/error.rs
    - crates/pyscf-mp2/src/helpers.rs
    - crates/pyscf-mp2/src/{mp2,ump2,dfmp2,dfmp2_native,frozen,rdm,hooks}.rs
    - crates/pyscf-mp2/tests/{ccsd_import_contract,rmp2_structural,ump2_structural}.rs
  modified:
    - Cargo.toml
    - Cargo.lock
    - crates/pyscf-mp2/Cargo.toml
    - crates/pyscf-mp2/src/lib.rs
    - crates/pyscf-oracle/src/runner.rs
    - .github/workflows/ci.yml

key-decisions:
  - "pyscf-ao2mo registered before pyscf-mp2 in workspace members so dependency layering reads top-down"
  - "MP2 python dispatch handlers deferred to 05-03..05-06 — Task 3 registers names only; catch-all dispatch arm returns UnknownMethod until then (job is if:false, never exercised)"
  - "Python _mo_without_core maps to Rust mo_without_core (no leading-underscore privacy convention) with #[doc(alias)]"

patterns-established:
  - "MP2/ao2mo error bridge: thiserror enum + From<E> for PyscfRsError via Core(InvalidMolecule(format!))"
  - "Wave-0 scaffold: always-on contract/symbol-existence arm + #[ignore = reason]d numeric arm"

requirements-completed: [MP2-01, MP2-02, MP2-04, MP2-05, MP2-08]

# Metrics
duration: 9min
completed: 2026-05-23
---

# Phase 5 Plan 01: Scaffold the MP2 Substrate Summary

**Stood up the 20th `pyscf-*` workspace member `pyscf-ao2mo`, wired `pyscf-mp2`'s pyo3-free/cubecl-free dependency block + module skeleton, scaffolded the Wave-0 test targets (incl. the always-on MP2-08 CCSD import contract), and registered the five MP2 numeric oracle arms behind a cintx#11-gated CI job — pure scaffolding, no compute.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-05-23T07:31:44Z
- **Completed:** 2026-05-23T07:40:30Z
- **Tasks:** 3
- **Files modified:** 24 (17 created, 7 modified across the 3 commits)

## Accomplishments

- `pyscf-ao2mo` is a registered, building 20th `pyscf-*` workspace member with `general`/`full` AO→MO stub surface and an `Ao2moError` type bridging to `PyscfRsError`.
- `pyscf-mp2` deps wired (ao2mo/scf/df/gto/algebra/runtime), strictly pyo3-free + cubecl-free (algebra+pyo3 wall held; `xtask check-dependency-wall` PASS), with a compiling 9-module skeleton.
- The MP2-08 CCSD import contract test (`from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core`) passes its always-on symbol-existence arm.
- All five MP2 oracle arms registered in `KNOWN_METHODS` (len 13 → 18, assertion updated); CI carries an always-on `mp2-structural` job + a `if: false` cintx#11-gated `mp2-oracle-cintx-gated` numeric job.

## Task Commits

1. **Task 1: Create pyscf-ao2mo crate skeleton + register in workspace** - `a65b982` (feat)
2. **Task 2: Wire pyscf-mp2 deps + module skeleton + Wave-0 test scaffolds** - `e6a3f8c` (feat)
3. **Task 3: Register MP2 oracle arms + CI structural/gated jobs** - `729928a` (feat)

**Plan metadata:** (final docs commit follows this SUMMARY)

## Files Created/Modified

- `crates/pyscf-ao2mo/Cargo.toml` - AO→MO crate manifest (core+algebra+gto path-deps + thiserror/tracing; no pyo3/cubecl/numpy)
- `crates/pyscf-ao2mo/src/lib.rs` - crate root: forbid(unsafe_code) + warn(clippy::unwrap_used) + pub mod incore/transform/error + re-exports
- `crates/pyscf-ao2mo/src/error.rs` - Ao2moError enum + From<Ao2moError> for PyscfRsError bridge
- `crates/pyscf-ao2mo/src/incore.rs` - general/full stubs returning NotYetImplemented (body 05-02)
- `crates/pyscf-ao2mo/src/transform.rs` - quarter-transform internals stub (body 05-02)
- `crates/pyscf-ao2mo/tests/transform_roundtrip.rs` - always-on `pyscf_ao2mo::general` smoke + #[ignore]d numeric roundtrip
- `crates/pyscf-mp2/Cargo.toml` - dep wiring (ao2mo/scf/df/gto/algebra/runtime + approx dev-dep); FORBIDDEN-pyo3 comment
- `crates/pyscf-mp2/src/lib.rs` - module skeleton + the five MP2-08 helper re-exports
- `crates/pyscf-mp2/src/error.rs` - Mp2Error enum (+ Ao2mo #[from]) + From<Mp2Error> for PyscfRsError
- `crates/pyscf-mp2/src/helpers.rs` - the five MP2-08 helper signatures (CCSD import contract)
- `crates/pyscf-mp2/src/{mp2,ump2,dfmp2,dfmp2_native,frozen,rdm,hooks}.rs` - one-line body-in-plan-05-0X stubs
- `crates/pyscf-mp2/tests/ccsd_import_contract.rs` - MP2-08 contract (always-on existence arm + #[ignore]d numeric arm)
- `crates/pyscf-mp2/tests/{rmp2_structural,ump2_structural}.rs` - always-on module-surface arms + #[ignore]d kernel-shape arms
- `crates/pyscf-oracle/src/runner.rs` - 5 MP2 arms in KNOWN_METHODS (13→18) + updated len assertion
- `.github/workflows/ci.yml` - mp2-structural (always-on) + mp2-oracle-cintx-gated (if:false) jobs
- `Cargo.toml` / `Cargo.lock` - register crates/pyscf-ao2mo member (19→20)

## Decisions Made

- **pyscf-ao2mo placed before pyscf-mp2 in the members list** so the dependency layering reads top-down (per plan).
- **MP2 python dispatch handlers deferred:** Task 3 registers the five method *names* in `KNOWN_METHODS` and updates the len-assert, but does NOT add `python_impl::dispatch` match arms — those land with the numeric bodies in 05-03..05-06. The existing catch-all `other => UnknownMethod` arm covers the names under `--features python` until then. Safe because the `mp2-oracle-cintx-gated` job is `if: false` (never runs) and the default-feature structural path only checks `KNOWN_METHODS.contains`.
- **`_mo_without_core` → `mo_without_core`:** Rust has no leading-underscore privacy convention; the Python name is recorded via `#[doc(alias = "_mo_without_core")]`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Cargo.lock pre-existing dirty version bumps isolated from the ao2mo entry**
- **Found during:** Task 1 (workspace registration / `cargo build -p pyscf-ao2mo --locked`)
- **Issue:** The working-tree `Cargo.lock` already carried ~70 unrelated lines of registry version bumps (e.g. `autocfg 1.5.0→1.5.1`, `bumpalo 3.20.2→3.20.3`) from a prior `cargo update` in this intentionally-dirty tree. Staging the lockfile as-is would have swept those unrelated bumps into the plan commit.
- **Fix:** `git checkout -- Cargo.lock` to restore the HEAD baseline, then re-ran the scoped build so the lockfile diff became exactly the new `pyscf-ao2mo` package entry (11 insertions) + the `pyscf-mp2` dependency-list update (Task 2). No registry version bumps staged.
- **Files modified:** Cargo.lock
- **Verification:** `git diff Cargo.lock` shows only the ao2mo/mp2 package nodes; `grep autocfg/bumpalo` confirms the pre-existing bumps stayed at HEAD versions (unstaged).
- **Committed in:** `a65b982` / `e6a3f8c`

**2. [Rule 1 - Formatting] rustfmt collapsed the ao2mo `general()` signature to one line**
- **Found during:** Task 2 (`cargo fmt -p pyscf-ao2mo -p pyscf-mp2 --check` before committing)
- **Issue:** `incore.rs::general` was written with a multi-line param list; rustfmt (and thus the CI `cargo fmt --check` gate) wanted it on one line.
- **Fix:** Ran `cargo fmt -p pyscf-ao2mo -p pyscf-mp2`; staged the one-line `general()` reformatting (a Task-1 file fixed during Task 2).
- **Files modified:** crates/pyscf-ao2mo/src/incore.rs
- **Verification:** `cargo fmt -p pyscf-ao2mo -p pyscf-mp2 --check` exits 0.
- **Committed in:** `e6a3f8c` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 formatting)
**Impact on plan:** Both auto-fixes were mechanical hygiene (lockfile isolation in a dirty tree + rustfmt). No scope creep, no behavior change, no architectural impact.

## Note on the `# pyo3 is FORBIDDEN here` comment

`crates/pyscf-mp2/Cargo.toml` contains the plan-mandated comment
`# pyo3 is FORBIDDEN here (D-07/D-08 algebra+pyo3 wall) — bridge lives in pyscf-py.`
A naive `grep pyo3 Cargo.toml` matches this comment, but there is NO `pyo3`/`cubecl`/`numpy`
*dependency* in either new crate. `crates/pyscf-ao2mo/Cargo.toml` has zero occurrences of any
of the three; `xtask check-dependency-wall` PASSes (no new cubecl consumer).

## Known Stubs

This plan is **pure scaffolding by design** (per the plan objective: "ships NO compute"). The
following stubs are intentional and each names the plan that fills it:

| Stub | File | Resolved by |
|------|------|-------------|
| `incore::general` / `incore::full` return `Err(NotYetImplemented)` | crates/pyscf-ao2mo/src/incore.rs | 05-02 |
| `transform.rs` (doc-comment only) | crates/pyscf-ao2mo/src/transform.rs | 05-02 |
| `helpers::{get_nocc,get_nmo,get_frozen_mask,get_e_hf,mo_without_core}` return `Err(NotYetImplemented)` | crates/pyscf-mp2/src/helpers.rs | 05-03 |
| `mp2/ump2/dfmp2/dfmp2_native/frozen/rdm/hooks` (doc-comment-only modules) | crates/pyscf-mp2/src/*.rs | 05-02..05-06 |
| MP2 oracle `python_impl::dispatch` match arms (names registered, handlers absent) | crates/pyscf-oracle/src/runner.rs | 05-03..05-06 |
| `#[ignore]`d numeric test arms (ccsd_import numeric, rmp2/ump2 kernel-shape, ao2mo roundtrip) | tests | 05-02..05-04 |

These stubs are the explicit deliverable of a Wave-0 scaffold plan; they do not block the plan's
goal (a buildable 20-crate workspace with concrete targets for the implementation waves).

## Issues Encountered

- The `--locked` build did not error on the pre-existing dirty lockfile (the dirty bumps were a no-op for resolution), so the lockfile isolation (Deviation 1) was the only handling needed.

## Next Phase Readiness

- Wave 2 (05-02 AO→MO transform) has a compiling `pyscf-ao2mo` with `general`/`full` targets + the `transform_roundtrip` test scaffold (numeric arm ready to un-`#[ignore]`).
- Waves 3–6 (RMP2/UMP2/DF-MP2/native + RDM) have the `pyscf-mp2` module skeleton, the MP2-08 helper signatures, the structural test scaffolds, and the 5 oracle arms in place.
- The `mp2-oracle-cintx-gated` numeric CI job is correctly disabled (`if: false`) pending cintx#11 (arity-3/4 `int2e`/`int3c2e_sph`), exactly like DF-HF/DFT-01. Flip the gate once cintx#11 merges.

## Self-Check: PASSED

All 12 spot-checked created files exist on disk; all 3 task commit hashes
(`a65b982`, `e6a3f8c`, `729928a`) are present in the git history.

---
*Phase: 05-mp2*
*Completed: 2026-05-23*
