---
phase: 05-mp2
plan: 03
subsystem: mp2
tags: [mp2, rmp2, frozen-core, scs-mp2, chemcore, ao2mo, oracle-sum, override-hooks]

# Dependency graph
requires:
  - phase: 05-02
    provides: "pyscf_ao2mo::general (ia|jb) AO→MO transform (bit-exact via oracle_sum)"
  - phase: 05-01
    provides: "pyscf-mp2 9-module skeleton, Mp2Error bridge, the 5 MP2-08 helper signatures + ccsd_import_contract scaffold"
provides:
  - "Frozen enum (None/Count/List/Auto/Window) + chemcore OnceLock table (verbatim elements.py:1079) + frozen_mask active-orbital resolver"
  - "The five MP2-08 helpers (get_nocc/get_nmo/get_frozen_mask/get_e_hf/mo_without_core) with upstream semantics — the always-on CCSD import contract"
  - "rmp2_kernel closed-form MP2 correlation energy (oracle_sum/oracle_dot reductions, no += accumulation)"
  - "Mp2Reference (D-07 snapshot) + default_ao2mo (int2e→ao2mo.general, ?-propagating, cintx#11-gated numeric)"
  - "scs_energy ss/os factor split (1.0/1.0 = plain MP2; 1/3,1.2 = SCS default)"
  - "Mp2OverrideHooks trait + ChemistsEris + NoMp2Overrides default (D-08, pyo3-free)"
affects: [05-04 (UMP2 + RDMs reuse helpers/hooks), 05-05 (DF-MP2 swaps ao2mo via Mp2OverrideHooks), 05-07 (pyscf-py bridge snapshots Mp2Reference), 06 (CCSD imports the MP2-08 five verbatim)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Frozen-spec → active-orbital bool mask resolver mirroring upstream get_frozen_mask (true=active)"
    - "chemcore element→orbital OnceLock<HashMap> static table (auxbasis.rs shape), verbatim transcription verified against upstream"
    - "Mp2OverrideHooks D-08 override seam delegating to crate::mp2::default_* free fns (pyscf-scf NoOverrides precedent)"
    - "ChemistsEris flat (i,a,j,b) C-order block consumed by the closed-form kernel"

key-files:
  created: []
  modified:
    - "crates/pyscf-mp2/src/frozen.rs — Frozen enum + chemcore table + frozen_mask"
    - "crates/pyscf-mp2/src/helpers.rs — the five MP2-08 helpers (real bodies)"
    - "crates/pyscf-mp2/src/mp2.rs — Mp2Reference + rmp2_kernel + default_ao2mo + scs_energy + default_energy"
    - "crates/pyscf-mp2/src/hooks.rs — Mp2OverrideHooks + ChemistsEris + NoMp2Overrides"
    - "crates/pyscf-mp2/src/lib.rs — re-exports"
    - "crates/pyscf-mp2/tests/ccsd_import_contract.rs — always-on numeric contract"
    - "crates/pyscf-mp2/tests/rmp2_structural.rs — always-on kernel/SCS/error-propagation tests"

key-decisions:
  - "chemcore table holds frozen-core ORBITALS not electrons — auto count = sum(chemcore_atm[Z]) with NO ÷2 (PLAN text said /2 but upstream chemcore() returns sum directly; followed upstream code, verified O→1/Si→5)"
  - "Frozen::Auto carries no element data; the kernel supplies refr.mol.atom_charges() to frozen_mask (helpers pass &[] so Auto resolves 0 through the data-only helper surface — documented; only None/Count/List exercised by the CCSD contract)"
  - "Mp2OverrideHooks.energy/make_rdm1/make_rdm2 given default-method bodies (energy→default_energy; rdm→NotYetImplemented{plan:4}) so a hooks impl need only override ao2mo (DF-MP2 in 05-05) and the RDM bodies land in 05-04"

patterns-established:
  - "Closed-form energy reduction discipline: materialize gi/t2i/(ib|ja) into reused scratch Vecs, reduce via oracle_dot per i, oracle_sum the per-i term Vec — bit-exact + thread-count invariant (T-05-03-FP)"
  - "int2e NotYetImplemented{phase:2} propagates with ? through default_ao2mo — never panics, never substitutes zeros (T-05-03-FFI); numeric flips on at cintx#11 with no code change (D-05)"

requirements-completed: [MP2-01, MP2-03, MP2-06, MP2-08]

# Metrics
duration: 14min
completed: 2026-05-23
---

# Phase 5 Plan 03: In-core RMP2 kernel + MP2-08 helpers + frozen-core + SCS + Mp2OverrideHooks Summary

**Closed-form RMP2 correlation-energy kernel (oracle_sum reductions, no `+=`) plus the five verbatim CCSD-import MP2-08 helpers, frozen-core (int/list/'auto'/window) resolution against a verbatim chemcore table, SCS-MP2 factor split, and the pyo3-free Mp2OverrideHooks D-08 seam — all always-on; numeric energy cintx#11-gated.**

## Performance

- **Duration:** ~14 min
- **Started:** 2026-05-23T07:50:00Z (approx)
- **Completed:** 2026-05-23T08:04:44Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- **MP2-08 CCSD import contract is now always-on** — `get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`mo_without_core` export with the exact `cc/ccsd.py:35` symbol set and upstream-matching semantics; the contract test asserts `Frozen::None → nocc=2/nmo=4` and `Frozen::Count(1) → mask [false,true,true,true]` on `mo_occ=[2,2,0,0]` (no `#[ignore]`).
- **Frozen-core resolution** — `Frozen::{None,Count,List,Auto,Window}` resolve the active-orbital mask matching upstream `get_frozen_mask` (true=active); the `Auto` chemcore element→orbital table is transcribed VERBATIM from `elements.py:1079` (all 119 Z entries verified bit-identical to upstream) and summed directly per upstream `chemcore()` (no ÷2).
- **rmp2_kernel** ports the upstream closed-form math (`mp2.py:47-76`): `t2 = (ia|jb)/(εi+εj−εa−εb)`, `edi = 2·Σ(ia|jb)·t2`, `exi = −Σ(ib|ja)·t2`, `e_ss = edi·0.5+exi`, `e_os = edi·0.5` — every reduction via `oracle_dot`/`oracle_sum` (no `+=` accumulator), verified bit-exact against a hand-computed synthetic `(ia|jb)` block (1×1 → e_corr=−0.125; 1×2 → independent longhand reference).
- **default_ao2mo** builds frozen-aware occupied/virtual MO subsets and calls `pyscf_ao2mo::general([co,cv,co,cv])` with `eri_ao = intor("int2e")`; the `int2e` `NotYetImplemented{phase:2}` error propagates with `?` (verified `Err(NotYetImplemented)` against a real H2/STO-3G Mole — no panic, no zero-buffer substitution).
- **scs_energy** factor split — `1.0/1.0` reproduces plain `e_ss+e_os`; `1/3,1.2` gives the SCS default (J. Chem. Phys. 118, 9095 (2003)).
- **Mp2OverrideHooks + ChemistsEris + NoMp2Overrides** declared pyo3-free (D-08), `NoMp2Overrides::ao2mo` delegating to `default_ao2mo` — the override seam DF-MP2 (05-05) swaps and CCSD reuses.

## Task Commits

Each task was committed atomically:

1. **Task 1: MP2-08 helpers + frozen-core resolution + CCSD import contract** — `99b9260` (feat)
2. **Task 2: RMP2 closed-form kernel + SCS energy + Mp2OverrideHooks** (TDD: RED = failing compile on missing symbols → GREEN = implementation) — `041bff4` (feat)

**Plan metadata:** (final docs commit) — SUMMARY.md + STATE.md + ROADMAP.md

## Files Created/Modified

- `crates/pyscf-mp2/src/frozen.rs` — `Frozen` enum, chemcore `OnceLock<HashMap<u32,usize>>` table (verbatim elements.py:1079), `chemcore_count`, `frozen_mask` active-orbital resolver with in-range validation (ShapeMismatch, never OOB).
- `crates/pyscf-mp2/src/helpers.rs` — the five MP2-08 helpers over `(mo_occ, &Frozen)`; `mo_without_core` does column-subset slicing on the column-major `MOCoefficients`.
- `crates/pyscf-mp2/src/mp2.rs` — `Mp2Reference` (D-07 snapshot, carries `mol`), `rmp2_kernel`, `default_ao2mo`, `scs_energy`, `default_energy`, `Mp2Result`.
- `crates/pyscf-mp2/src/hooks.rs` — `Mp2OverrideHooks` trait, `ChemistsEris` (i,a,j,b flat block), `NoMp2Overrides`.
- `crates/pyscf-mp2/src/lib.rs` — re-exports `Frozen`, the mp2 + hooks surface.
- `crates/pyscf-mp2/tests/ccsd_import_contract.rs` — always-on numeric contract (both arms).
- `crates/pyscf-mp2/tests/rmp2_structural.rs` — always-on hand-computed-energy, SCS-factor, and int2e-error-propagation tests.

## Decisions Made

- **chemcore table semantics (followed upstream over PLAN text):** The plan's prose said the auto frozen count is `sum(per-atom core electrons) / 2`. Upstream `pyscf/data/elements.py:chemcore()` actually sums `chemcore_atm[Z]` over atoms and returns it DIRECTLY (no ÷2) — `chemcore_atm` already encodes frozen-core *orbitals*, not electrons (O→1, the 1s; Si→5, the 1s2s2p shell). I implemented the upstream behaviour (`chemcore_count = Σ chemcore_atm[Z]`) and documented the discrepancy, since the threat model requires the chemcore table be "ported verbatim from upstream". Transcription verified bit-identical (119 entries) and spot-checked (O→1, Ne→1, Si→5).
- **Helper signature change (05-01 → 05-03):** The 05-01 stubs took `(mo_occ, n_frozen: usize)` / `(&[f64], &[bool])`; the plan specifies the real `(mo_occ, &Frozen)` contract (and `get_e_hf(reference_e_tot: f64)`, `mo_without_core(&MOCoefficients, &Frozen)`). I replaced the signatures and updated the always-on contract test to bind the new `fn(&[f64], &Frozen)` pointers (the compile-time CCSD interface guard).
- **Frozen::Auto element threading:** The data-only MP2-08 helpers do not carry the molecule, so through the helper surface `Auto` resolves with no element info (count 0); the kernel's `default_ao2mo` supplies the real `refr.mol.atom_charges()` to `frozen_mask`. Documented in `helpers.rs`. The CCSD contract only exercises None/Count/List.
- **Hooks default methods:** `energy` defaults to `default_energy`; `make_rdm1`/`make_rdm2` default to `NotYetImplemented{plan:4}` so a hooks impl need only override `ao2mo` (the test `SyntheticEris` and `NoMp2Overrides` both rely on this).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated structural-test scaffolds to the new (mo_occ, &Frozen) helper signature**
- **Found during:** Task 1 (helper signature change)
- **Issue:** `crates/pyscf-mp2/tests/rmp2_structural.rs:17` and `crates/pyscf-mp2/tests/ump2_structural.rs:16` bound the old `fn(&[f64], usize)` helper-pointer signatures from 05-01; changing the helpers to `(mo_occ, &Frozen)` broke their compile, blocking the whole `cargo test -p pyscf-mp2` build.
- **Fix:** Updated both one-line pointer bindings to `fn(&[f64], &pyscf_mp2::Frozen)`. (`rmp2_structural.rs` was then fully rewritten by Task 2; `ump2_structural.rs` keeps only the one-line fix and its 05-04 `#[ignore]` placeholder.)
- **Files modified:** `crates/pyscf-mp2/tests/rmp2_structural.rs`, `crates/pyscf-mp2/tests/ump2_structural.rs`
- **Verification:** `cargo test -p pyscf-mp2 --locked` green (all binaries compile + pass).
- **Committed in:** `99b9260` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking). The PLAN-vs-upstream chemcore semantics divergence is logged under Decisions (followed upstream code per the verbatim-port requirement), not counted as a deviation.
**Impact on plan:** The fix was necessary to keep the build green after the contractual helper-signature change; no scope creep. `ump2_structural.rs` was not in the plan's `files_modified` but required a one-line build fix.

## Issues Encountered

- **Edition-2024 binding-mode strictness:** `.filter(|(&occ, &active)| ...)` was rejected ("reference pattern not allowed when implicitly borrowing"); fixed to `.filter(|&(&occ, &active)| ...)`.
- **Clippy `-D warnings`:** three lints on first pass — doc-list overindent, derivable `Default` (switched to `#[derive(Default)]` + `#[default]`), and `manual checked division` (switched `if nao==0 {0} else {len/nao}` → `len.checked_div(nao).unwrap_or(0)`). All cleared; `unwrap_or` is allowed under `#![warn(clippy::unwrap_used)]`.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **05-04 (UMP2 + RDMs):** reuses `Frozen`/`frozen_mask`/the MP2-08 helpers and the `Mp2OverrideHooks` seam; the `make_rdm1`/`make_rdm2` default-method bodies are the explicit landing point.
- **05-05 (DF-MP2):** implements `Mp2OverrideHooks::ao2mo` to swap the density-fitted `(ia|jb)` source — the seam is in place.
- **05-07 (pyscf-py bridge):** snapshots `Mp2Reference` from `mf` (incl. `mf.mol`).
- **Phase 6 (CCSD):** imports the five MP2-08 helpers verbatim — the always-on contract test guards the symbol set + signatures.
- **Numeric gate (carry-over, not a blocker):** the in-core RMP2 energy on a real molecule is cintx#11-gated (arity-4 `int2e` `NotYetImplemented{phase:2}`). `default_ao2mo` propagates the error cleanly; the energy flips on with no code change once cintx#11 lands. The closed-form algorithm + structural/helper layers are complete and verified against synthetic ERIs.

---
*Phase: 05-mp2*
*Completed: 2026-05-23*

## Self-Check: PASSED

- All created/modified files present (frozen.rs, helpers.rs, mp2.rs, hooks.rs, lib.rs, both test files, SUMMARY.md).
- Both task commits present in git history (`99b9260`, `041bff4`).
- `cargo test -p pyscf-mp2 --locked` green (13 lib + 2 contract + 5 rmp2 + 1 ump2 always-on; 1 ump2 ignored = 05-04 placeholder).
- `cargo clippy -p pyscf-mp2 --all-targets -- -D warnings` clean; `cargo fmt -p pyscf-mp2 --check` clean.
- `xtask check-no-fma` PASS; `xtask check-dependency-wall` PASS (no cubecl leak).
- No `pyo3` symbol referenced in pyscf-mp2 (doc-comment mentions only).
