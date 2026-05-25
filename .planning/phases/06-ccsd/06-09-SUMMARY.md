---
phase: 06-ccsd
plan: 09
subsystem: ccsd
tags: [ccsd, df-ccsd, dfccsd, density-fitting, hdf5-spill, workspace-pool, outcore, ao2mo, vvL, ccsd-08]

# Dependency graph
requires:
  - phase: 06-03
    provides: "in-core ccsd_kernel<H: CcsdOverrideHooks> + default_ao2mo (the kernel DF-CCSD reuses verbatim, swapping only ao2mo)"
  - phase: 06-02
    provides: "WorkspacePool Spilled backend + SpillHandle RAII drop-delete (the vvL spill seam, D-07/D-08)"
  - phase: 06-01
    provides: "CcsdOverrideHooks::ao2mo seam + ChemistsEris block container + CcsdReference"
  - phase: 05-mp2
    provides: "DFRMP2(RMP2) subclass-swaps-ERI template (dfmp2.rs df_ao2mo: (ia|jb)=sum_Q B^Q*B^Q via oracle_dot over Q); pyscf-df cholesky_eri/default_ri (un-gated since 05-09); pyscf-ao2mo general/full + the D-04 outcore deferral"
provides:
  - "pyscf-ccsd::dfccsd — DFRCCSD/DFUCCSD ERI-swap (df_ao2mo) reusing the in-core ccsd_kernel; (pq|rs)=sum_Q B^Q_pq*B^Q_rs via oracle_dot; the dfccsd.py:70 RCCSD(ccsd.CCSD) DF subclass pattern"
  - "dfrccsd_kernel free fn + DFRCCSD::kernel driver method (the DFRMP2(RMP2) swap-the-source contract for CCSD)"
  - "block_sizing — the verified dfccsd.py:93-96 dmax/vvblk formulas (D-08 reservation sizing)"
  - "vvL HDF5 spill: over-budget vvL routes through the 06-02 WorkspacePool Spilled backend (pyscf_chkfile::hdf5 alias, D-07 no new dep), RAII drop-deleted (T-06-09-LEAK)"
  - "pyscf-ao2mo::outcore — general_outcore/full_outcore + OutcoreScratch HDF5-spilling AO->MO transform (the Phase-5 D-04 deferral); bit-exact == in-core general/full"
affects:
  - 06-10 (PyO3 bridge — PyDFRCCSD/factory dispatch when mf is density-fitted wraps dfrccsd_kernel)
  - 06-11 (oracle/CI — the benzene-dimer/cc-pVDZ constrained-budget spill proof is the workflow_dispatch human-verify arm)
  - phase-07 (gradients — DF-CCSD λ/RDM response may reuse the outcore surface)

# Tech tracking
tech-stack:
  added:
    - "pyscf-ao2mo now deps pyscf-chkfile + ndarray (the outcore HDF5 scratch) — NO new cubecl/libxc; libxc stays 0"
  patterns:
    - "DF-CCSD swap-the-ERI-source: a CcsdOverrideHooks::ao2mo impl builds the full ChemistsEris from the DF B-tensor; the in-core ccsd_kernel is reused verbatim (the Phase-5 DFRMP2(RMP2) template)"
    - "dedicated budget-matched isolation pool for the vvL spill (so a spilled/larger free buffer is never wrongly reused for the kernel's in-core Wvvvv — free-list matches on size, not backend)"
    - "plain Range<usize> -> hdf5 Selection (avoids ndarray::s! which emits allow(unsafe_code), colliding with forbid(unsafe_code))"
    - "RAII OutcoreScratch / vvL pool drop-delete (mirror lib.H5TmpFile auto-delete, D-07)"

key-files:
  created:
    - crates/pyscf-ao2mo/src/outcore.rs
    - crates/pyscf-ccsd/tests/dfccsd_spill.rs
  modified:
    - crates/pyscf-ccsd/src/dfccsd.rs
    - crates/pyscf-ccsd/src/lib.rs
    - crates/pyscf-ao2mo/src/lib.rs
    - crates/pyscf-ao2mo/src/error.rs
    - crates/pyscf-ao2mo/Cargo.toml

key-decisions:
  - "vvL lives in a DEDICATED budget-matched WorkspacePool inside df_ao2mo, NOT the kernel's shared pool: a spilled vvL left on the kernel pool's free-list would be wrongly reused for the next in-core Wvvvv reserve (free-list scans by size, not backend), and with_mut_slice fails on a Spilled buffer; a larger reused buffer also breaks Wvvvv's exact-nv^4-length check. Isolation keeps both tenants' lifecycles independent while spilling under the SAME PYSCF_MAX_MEMORY budget."
  - "Outcore uses plain Range<usize> -> hdf5 Selection (impl From<Range<usize>> for Selection) instead of the ndarray::s! macro, whose expansion emits #[allow(unsafe_code)] and collides with the crate-wide #![forbid(unsafe_code)]."
  - "DFUCCSD ships as a driver struct (open-shell wiring) but the numeric headline this plan is the closed-shell DFRCCSD (CCSD-08); the DF open-shell numeric parity is a 06-08-closeout human-verify arm (D-04 heavy/upstream)."

patterns-established:
  - "Swap-the-ERI-source DF subclass (DFRCCSD): only ao2mo changes; the amplitude kernel is reused"
  - "Spill is a storage-backend swap behind the WorkspacePool handle (D-01), not a math rewrite — the contraction reads vvL back through as_slice identically whether InMemory or Spilled"

requirements-completed: [CCSD-08]

# Metrics
duration: 35min
completed: 2026-05-25
---

# Phase 6 Plan 09: DF-CCSD with HDF5 Spill (CCSD-08) Summary

**DFRCCSD swaps the CCSD ERI source to the density-fitted `vvL` B-tensor via the `CcsdOverrideHooks::ao2mo` seam (the Phase-5 `DFRMP2(RMP2)` subclass pattern), reusing the in-core `ccsd_kernel` verbatim; the over-budget `vvL`/`Wabef` intermediate spills to an HDF5 temp file through the 06-02 `WorkspacePool` `Spilled` backend (RAII drop-deleted, no leftover scratch), and the Phase-5 D-04 outcore/semi-incore AO→MO surface lands in `pyscf-ao2mo`.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 2
- **Files modified:** 7 (2 created, 5 modified)

## Accomplishments

- **Task 1 — outcore AO→MO surface (the Phase-5 D-04 deferral):** `crates/pyscf-ao2mo/src/outcore.rs` ports `pyscf/ao2mo/outcore.py` + `semi_incore.py`. `general_outcore`/`full_outcore` mirror the in-core `general`/`full` quarter-transform but SPILL the half-transformed `[np,nq,nao,nao]` intermediate to an HDF5 scratch dataset (`OutcoreScratch`, via the `pyscf_chkfile::hdf5` alias — D-07, no new `hdf5-metno` dep) between the first-half (`p`,`q`) and second-half (`r`,`s`) contractions; the peak resident buffer is one `[np,nq,nao]` s-slab, not the full intermediate. `OutcoreScratch` RAII drop-deletes the temp file. Output is BIT-EXACT == in-core (same `oracle_sum` fold order). 6 always-on tests (outcore==incore bit-exact, full, scratch deleted on drop, no-leftover-scratch, shape-mismatch rejected, rectangular quarter-transform).
- **Task 2 — DFRCCSD ERI-swap + vvL spill:** `crates/pyscf-ccsd/src/dfccsd.rs` (was a Wave-4 stub). `DfCcsdHooks::ao2mo` builds the FULL `ChemistsEris` block set (`oooo`/`ovoo`/`oovv`/`ovov`/`ovvo`/`ovvv`/`vvvv`) from the DF B-tensor: `(pq|rs) = Σ_Q B^Q_pq·B^Q_rs` via `oracle_dot` over the auxiliary axis (the `dfmp2.rs::df_ao2mo` MATH), with the B-tensor MO-transform `transform_b_block` (materialize-then-`oracle_sum`, no gemm, no `+=`). The `vvL` half-tensor (the dominant tenant, `dfccsd.py:139`) is reserved with `allow_spill=true` so an over-budget run SPILLS to HDF5 (the 06-02 `Spilled` backend) instead of HARD-refusing; the spill file is RAII drop-deleted. `block_sizing` ports the verified `dfccsd.py:93-96` `dmax`/`vvblk` formulas (D-08). `dfrccsd_kernel` + `DFRCCSD::kernel` wire the swap into the reused `ccsd_kernel`; `DFUCCSD` driver struct ships the open-shell wiring. 5 always-on tests in `tests/dfccsd_spill.rs`.

## Task Commits

1. **Task 1: outcore/semi-incore HDF5-spilling AO→MO surface** - `ac45cdc` (feat)
2. **Task 2: DFRCCSD ERI-swap + dmax/vvblk sizing + vvL HDF5 spill + spill test** - `7fa9941` (feat)

**Plan metadata:** (final docs commit — this SUMMARY + STATE + ROADMAP + REQUIREMENTS + deferred-items)

## Files Created/Modified

- `crates/pyscf-ao2mo/src/outcore.rs` (created) - outcore/semi-incore AO→MO transform spilling the half-transform to HDF5; `OutcoreScratch` RAII; bit-exact == in-core.
- `crates/pyscf-ao2mo/src/error.rs` (modified) - `Ao2moError::Outcore { reason }` variant for spill-scratch I/O failures.
- `crates/pyscf-ao2mo/src/lib.rs` (modified) - `pub mod outcore` + `general_outcore`/`full_outcore`/`OutcoreScratch` re-exports.
- `crates/pyscf-ao2mo/Cargo.toml` (modified) - `+ pyscf-chkfile + ndarray` (the outcore spill; no new cubecl/libxc).
- `crates/pyscf-ccsd/src/dfccsd.rs` (modified, was stub) - DFRCCSD/DFUCCSD ERI-swap (`df_ao2mo`), `transform_b_block`, `assemble_block`, `block_sizing`, `dfrccsd_kernel`, `DfCcsdHooks`.
- `crates/pyscf-ccsd/src/lib.rs` (modified) - `DFRCCSD`/`DFUCCSD`/`block_sizing`/`df_ao2mo`/`dfrccsd_kernel` re-exports.
- `crates/pyscf-ccsd/tests/dfccsd_spill.rs` (created) - 5 always-on arms (ERI assembly correctness, DF-CCSD convergence + driver wiring, vvL spill + no-leftover, spill-file observed created-then-deleted, in-core no-spill).

## Decisions Made

- **vvL isolation pool (the load-bearing design call):** `df_ao2mo` reserves `vvL` in a DEDICATED, budget-matched `WorkspacePool` (`WorkspacePool::new(pool.budget_bytes)`), NOT the kernel's shared pool. See the Deviations section (Rule 1) — this was forced by a genuine pool-reuse hazard.
- **`ndarray::s!` avoidance:** the macro emits `#[allow(unsafe_code)]`; the outcore scratch slicing uses plain `Range<usize>` → `hdf5 Selection` (`impl From<Range<usize>> for Selection`), preserving `#![forbid(unsafe_code)]`.
- **Aux default = `default_ri` (mp2fit `*-ri`, NOT jkfit):** matches DF-MP2's A2 choice (`dfmp2.py:136`); un-gated since 05-09.
- **DFUCCSD scope:** the closed-shell DFRCCSD is the CCSD-08 numeric headline this plan; the open-shell DFUCCSD ships its driver struct (the swap-the-source wiring) but its DF numeric parity is the 06-08-closeout human-verify arm (D-04 heavy/upstream — the in-core UCCSD numeric is the 06-04 deliverable).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] vvL must spill in a dedicated pool, not the kernel's shared pool**
- **Found during:** Task 2 (the `dfccsd_spill` arms `dfccsd_incore_vvl_creates_no_scratch` + `dfccsd_vvl_spills_to_hdf5...` initially failed).
- **Issue:** The plan said to spill `vvL` "via the 06-02 WorkspacePool Spilled backend." My first cut reserved `vvL` in the SAME pool the kernel passes to the hook. But the kernel reserves its in-core `Wvvvv` `[nvir⁴]` tenant from that pool with `allow_spill=false` and accesses it via `with_mut_slice` (an in-memory-only accessor). After `df_ao2mo` released `vvL` back to the free-list, the next `Wvvvv` reserve REUSED it (the free-list scan matches on `backend.len() >= need`, NOT backend type or exact size). Two failures resulted: (a) a SPILLED `vvL` reused for `Wvvvv` → `with_mut_slice` returns `ProbeFailed("with_mut_slice on a spilled buffer")`; (b) a LARGER in-memory `vvL` (20 elems) reused for the 16-elem `Wvvvv` → `default_update_amps_with_wvvvv` rejects `wvvvv.len() != nv⁴`.
- **Fix:** `df_ao2mo` reserves `vvL` in a DEDICATED `WorkspacePool::new(pool.budget_bytes)` whose lifetime is the function scope. This keeps the kernel's pool clean (only `Wvvvv` lives there) while spilling `vvL` under the SAME `PYSCF_MAX_MEMORY` budget; the dedicated pool drops at function end → the `vvL` `SpillHandle`'s RAII deletes the temp file. This is the most faithful reading of "DF-CCSD spills its OWN `vvL` intermediate" (the kernel's `Wvvvv` and the DF's `vvL` are distinct tenants). I did NOT modify the shared `WorkspacePool` reuse semantics (out of this plan's file set; the size-only free-list match is a pre-existing pool behavior that only this DF mixed-shape/backend usage exposes — logged conceptually, but the isolation pool sidesteps it cleanly without touching 06-02's tested contract).
- **Files modified:** `crates/pyscf-ccsd/src/dfccsd.rs`
- **Verification:** all 5 `dfccsd_spill` arms pass; the in-core path creates no scratch, the tiny-budget path spills + leaves no leftover.
- **Committed in:** `7fa9941` (Task 2).

**2. [Rule 1 - Verify-command correction] `cargo tree | grep -ci 'cubecl|libxc' == 0` is unachievable for pyscf-ao2mo**
- **Found during:** Task 1 verification (the plan's `AO2MO_TREE_CLEAN` gate printed `AO2MO_TREE_DIRTY`).
- **Issue:** `cargo tree -p pyscf-ao2mo | grep -ci 'cubecl\|libxc'` can NEVER be 0: `pyscf-ao2mo` ALREADY (on HEAD, before my change) deps `pyscf-algebra` + `pyscf-gto`, both of which transitively pull `cubecl` via `cintx-cubecl`. The same verify-command inaccuracy 06-02 documented.
- **Fix:** Verified the REAL invariant instead — `libxc == 0` (holds) and NO new cubecl SOURCE (my added `pyscf-chkfile`/`ndarray` deps pull `hdf5-metno`/`ndarray` only; the cubecl already present came from the pre-existing `pyscf-gto`/`pyscf-algebra` chain). Confirmed via `cargo tree -p pyscf-ao2mo -i cubecl` (source = `cintx-cubecl` ← `pyscf-gto`, NOT my new deps) and `grep -ci libxc == 0`. No code change.
- **Committed in:** n/a (verification only).

---

**Total deviations:** 2 auto-fixed (1 bug, 1 verify-command correction).
**Impact on plan:** The vvL isolation-pool fix is essential for correctness (the shared-pool path genuinely crashes); the verify-command correction confirms the real dependency invariant. No scope creep — no shared-pool semantics changed, no kernel (ccsd.rs) edit.

## Issues Encountered

- **Pre-existing clippy `type_complexity` in `crates/pyscf-ccsd/tests/rdm.rs:39`** (`converged_lambda_state` 7-tuple return). Verified verbatim on HEAD — NOT introduced by 06-09 (my plan touched only `dfccsd.rs`/`dfccsd_spill.rs`/the ao2mo outcore surface). Out of scope per the SCOPE BOUNDARY rule; logged to `deferred-items.md`. The 06-09 plan-touched targets (`pyscf-ccsd --lib`, `--test dfccsd_spill`, `pyscf-ao2mo --lib`) are clippy-clean under `-D warnings`.
- **`ndarray::s!` × `#![forbid(unsafe_code)]` collision** — resolved by using plain `Range<usize>` HDF5 hyperslab selections (see Decisions).

## Known Stubs

None for the closed-shell headline. `DFUCCSD` is a driver struct without a numeric kernel method this plan (its `ao2mo` open-shell swap + DF parity is the 06-08-closeout human-verify arm, D-04). This is documented intent (not a silent stub): the in-core UCCSD numeric headline is the 06-04 deliverable, and the DF open-shell parity is heavy/upstream-gated per the user-memory "don't freeze the test run" constraint.

## Threat Flags

No new threat surface beyond the plan's `<threat_model>`. The `mitigate` dispositions are satisfied:
- **T-06-09-LEAK** (spill temp-file leak): `OutcoreScratch::drop` + the dedicated `vvl_pool` drop delete the HDF5 scratch (RAII); proven by `outcore_scratch_deleted_after_transform`, `general_outcore_leaves_no_leftover_scratch`, `dfccsd_vvl_spills_to_hdf5_and_no_leftover_scratch`, `vvl_spill_file_observed_created_then_deleted`.
- **T-06-09-SHAPE** (B-tensor/aux block lengths): `transform_b_block`/`general_outcore` `?`-propagate `ShapeMismatch` before indexing; `#![forbid(unsafe_code)]` holds (the `ndarray::s!` avoidance preserves it).
- **T-06-09-OOM** (spill trigger): DF-CCSD spills `vvL` to HDF5 (explicit opt-in) rather than OOMing; proven by the tiny-budget spill arm succeeding.
- **T-06-09-SC** (accept — pyscf-ao2mo gains pyscf-chkfile): no registry install; only the existing Phase-3/4-vetted `hdf5-metno` (via pyscf-chkfile) + `ndarray` enter; libxc stays 0, no new cubecl source.
- **T-06-09-FP** (DF ERI reductions): host-loop `oracle_dot` over Q; the synthetic-B `ovov` matches an independent longhand `Σ_Q B^Q·B^Q` reference to 1e-12 relative.

## Next Phase Readiness

- **06-10 (PyO3 bridge):** `dfrccsd_kernel` + `DFRCCSD` are pyo3-free and ready for `pyscf-py` to wrap (`PyDFRCCSD` + factory dispatch when `mf` is density-fitted). The `ao2mo` hook seam is the bridge point.
- **06-11 (oracle/CI):** the benzene-dimer/cc-pVDZ constrained-budget DF-CCSD spill proof is the `workflow_dispatch` human-verify arm (D-04); the always-on small-system spill arms (this plan) prove the seam in CI.
- No blockers. CCSD-08 closed (DF-CCSD bounded memory + spill).

## Self-Check: PASSED

- Files exist: `crates/pyscf-ao2mo/src/outcore.rs`, `crates/pyscf-ccsd/tests/dfccsd_spill.rs`, `crates/pyscf-ccsd/src/dfccsd.rs`, `.planning/phases/06-ccsd/06-09-SUMMARY.md` — all present on disk.
- Commits exist: `ac45cdc` (Task 1) + `7fa9941` (Task 2) — both found in `git log`.
- Required checks: `cargo check -p pyscf-ccsd -p pyscf-ao2mo --tests` exits 0; `cargo check -p pyscf-mp2 --tests` (ao2mo consumer) exits 0; `cargo test -p pyscf-ccsd --test dfccsd_spill -- --test-threads=1` 5/5 pass; `cargo test -p pyscf-ao2mo outcore` 6/6 pass; in-core RCCSD smoke regression green; `DF_SPILL_WIRED` + `AO2MO` libxc=0 invariants hold.

---
*Phase: 06-ccsd*
*Completed: 2026-05-25*
