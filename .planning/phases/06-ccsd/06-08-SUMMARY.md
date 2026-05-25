---
phase: 06-ccsd
plan: 08
subsystem: pyscf-ccsd
tags: [ccsd, ao-direct, contract-vvvv-t2, oracle-reduction, tensor-arena, memory-frugality, direct-flag]

# Dependency graph
requires:
  - phase: 06-03
    provides: "ccsd_kernel<H> loop + init_amps MP2 seed + default_energy + default_ao2mo (full ChemistsEris block set) + the in-core update_amps vvvv step (cc_Wvvvv 'abcd,ijcd->ijab') the AO-direct branch replaces"
  - phase: 06-05
    provides: "amplitude-DIIS wiring (AmplitudeSubspace + ccsd_kernel_diis(diis: bool)) the AO-direct kernel mirrors"
  - phase: 06-02
    provides: "WorkspacePool try_reserve HARD pre-flight refusal (D-01) — the AO-direct path uses the LOWER nv^3 estimate"
  - phase: 05-02
    provides: "pyscf_ao2mo::general AO→MO host-loop transform — the AO-direct path tiles it per leading-virtual index"
  - phase: 05-08
    provides: "real bit-exact int2e (cintx#11 closed) — the AO source the AO-direct contraction tiles over"
provides:
  - "direct.rs: contract_vvvv_t2_aodirect (AO int2e source, per-a slice, peak nv^3) + contract_vvvv_t2_from_eris (eris.vvvv source, per-a slice equivalence anchor) — port of ccsd.py:473-570 _contract_vvvv_t2 / _contract_s4vvvv_t2 (path-b form)"
  - "ccsd.rs: ccsd_kernel_direct / ccsd_kernel_direct_diis — the direct=True kernel routing the nv^4 vvvv step through the AO-direct branch + estimate_direct_vvvv_bytes (the lower nv^3 pre-flight peak)"
  - "update_amps.rs: default_update_amps_direct (AO-direct vvvv step) + update_amps_core (shared by in-core and direct) + vvvv_step_from_wvvvv (in-core block)"
  - "tests/direct.rs: AO-direct e_corr == in-core e_corr (LiH/STO-3G, bit-equal, ≤1e-9) + lower-reservation memory-frugality proof (CCSD-07)"
  - "clippy absurd_extreme_comparisons on ccsd.rs DIIS start-cycle guard FIXED (deferred-items.md resolved)"
affects:
  - 06-09 (DF-CCSD — the other explicit ERI-mode opt-in; mirrors the lower-reservation arena discipline)
  - 06-10/06-11 (PyO3 bridge / oracle: the direct=True flag flows through the bridge; upstream byte-identity is the 06-08-closeout workflow_dispatch arm)

# Tech tracking
tech-stack:
  added: []   # no external packages; workspace-internal only
  patterns:
    - "AO-direct vvvv contraction tiled over the leading virtual index a — peak nv^3 MO slice, never the full nv^4 MO vvvv (path-b: full AO int2e once + MO-space tiling, since pyscf-gto exposes no shell-sliced int2e)"
    - "vvvv step refactored into a precomputed [no,no,nv,nv] block fed to a shared update_amps_core — in-core (cc_Wvvvv full) and AO-direct (on-the-fly) differ ONLY in how the block is produced"
    - "memory-frugality proof = the AO-direct pre-flight reservation (nv^3·8) is strictly below the in-core one (nv^4·8); a budget in between makes in-core HARD-refuse but direct accept"

key-files:
  created:
    - crates/pyscf-ccsd/tests/direct.rs
  modified:
    - crates/pyscf-ccsd/src/direct.rs
    - crates/pyscf-ccsd/src/ccsd.rs
    - crates/pyscf-ccsd/src/update_amps.rs
    - crates/pyscf-ccsd/src/lib.rs
    - .planning/phases/06-ccsd/deferred-items.md

key-decisions:
  - "RESEARCH Open Q4 RESOLVED-AT-EXECUTION (path b): grep of pyscf-gto's intor surface (intor.rs) confirms NO shell-sliced streaming int2e primitive in-tree — only intor('int2e') returning the full arity-4 AO tensor. AO-direct v1 therefore sources the full AO int2e ONCE and tiles the AO→MO vvvv transform over the leading virtual index a (one [1,nv,nv,nv] slice at a time via ao2mo::general([&cv_a, &cv, &cv, &cv])), contracting each slice against tau and discarding it. Peak vvvv-MO buffer = nv^3, so the full nv^4 vvvv MO tensor is NEVER materialized — satisfying CCSD-07's direct=True contract even though the AO source is held (the v1 path-b cost; a shell-sliced primitive that streams the AO source is the v2 upgrade)."
  - "The vvvv step split: cc_Wvvvv[a,b,c,d] = (two ovvv·t1 t1-corrections, nv^3) + vvvv[a,c,b,d] (the nv^4 integral part). Only the heavy integral part moves to AO-direct (contract_vvvv_t2_aodirect = einsum('ijcd,acbd->ijab', tau, vvvv), upstream ccsd.py:474); the t1-corrections stay in-core (default_update_amps_direct contracts them against tau touching only the nv^3 ovvv block). Both reassembled into the same [no,no,nv,nv] block the shared update_amps_core consumes."
  - "Direct flag wired as a separate kernel entrypoint (ccsd_kernel_direct / ccsd_kernel_direct_diis) rather than a bool field threaded through ccsd_kernel_diis: the AO-direct path needs the raw AO int2e + MO coeffs which only the in-tree default ao2mo exposes (not the generic hooks.ao2mo seam), so it is naturally bound to the default path. 'direct' appears throughout ccsd.rs (DIRECT_WIRED grep passes). The in-core ccsd_kernel/ccsd_kernel_diis are UNCHANGED."
  - "Equivalence proof anchor: AO-direct is a different contraction ORDER of the same math. Proven at two levels — (1) lib test: contract_vvvv_t2_from_eris (per-a tiled) == contract_vvvv_t2_incore_full (full nv^4) bit-close across (2,2)/(2,3)/(3,4); (2) integration test: ccsd_kernel_direct e_corr == ccsd_kernel e_corr on LiH/STO-3G, BIT-IDENTICAL (-0.020449057574, same 8 iters), well within the ≤1e-9 gate."
  - "Memory-frugality proof: estimate_direct_vvvv_bytes(nv)=nv^3·8 < estimate_vvvv_bytes(nv)=nv^4·8. Integration test sizes a pool budget between them (LiH: 512 < 1280 < 2048): in-core ccsd_kernel HARD-refuses (MemoryLimitExceeded on the nv^4 try_reserve, D-01 no downgrade) but ccsd_kernel_direct accepts and converges — the on-disk witness of the lower peak reservation."

patterns-established:
  - "AO-direct vvvv: source full AO int2e once, transform ONE leading-virtual a-slice [1,nv,nv,nv] at a time, contract against tau via host-loop oracle_sum, discard. Peak nv^3, RAYON 1==8 bit-invariant, ShapeMismatch-validated (T-06-08-SHAPE)."
  - "vvvv step as a swappable precomputed [no,no,nv,nv] block → in-core and AO-direct share update_amps_core verbatim (the only divergence is the block's provenance). The in-core path is byte-unchanged (vvvv_step_from_wvvvv reproduces the old einsum exactly; rccsd_numeric_smoke + convergence + diis_amps + uccsd_smoke all still green)."

requirements-completed: [CCSD-07]

# Metrics
duration: 8min
completed: 2026-05-25
---

# Phase 6 Plan 8: AO-direct CCSD Summary

**AO-direct CCSD (`mycc.direct=True`, CCSD-07): the `_contract_vvvv_t2` vvvv step is contracted on-the-fly from AO `int2e` tiled per leading-virtual index (peak `nv^3`, never the full `nv^4` MO `vvvv`), converging BIT-IDENTICALLY to the in-core `e_corr` on LiH/STO-3G with a strictly lower pre-flight reservation.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-25T02:58:32Z
- **Completed:** 2026-05-25T03:06:34Z
- **Tasks:** 2 (Task 1 TDD)
- **Files modified:** 5 (1 created)

## Accomplishments
- Ported the AO-direct `_contract_vvvv_t2` branch (`ccsd.py:473-570`) into `direct.rs`: `contract_vvvv_t2_aodirect` sources the full AO `int2e` once and tiles the AO→MO `vvvv` transform over the leading virtual index, so the full `nv^4` `vvvv` MO tensor is never materialized (peak `nv^3`).
- Wired `ccsd_kernel_direct` / `ccsd_kernel_direct_diis`: routes the heavy vvvv contraction through the AO-direct branch and uses the LOWER `nv^3` pre-flight reservation (`estimate_direct_vvvv_bytes`), skipping the full-`nv^4` arena reservation. In-core `ccsd_kernel` is untouched.
- Proved equivalence: AO-direct `e_corr` == in-core `e_corr` on LiH/STO-3G, bit-identical `-0.020449057574` (both converge in 8 iterations, ≤1e-9 gate); and proved the memory-frugality contract (a budget the in-core path HARD-refuses but the AO-direct path accepts).
- Fixed the incidental clippy `absurd_extreme_comparisons` on the DIIS start-cycle guard (`ccsd.rs`) and marked the `deferred-items.md` entry resolved.

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD): Port the AO-direct `_contract_vvvv_t2` branch into `direct.rs`** - `d171b10` (feat) — RED tests + GREEN implementation landed in one commit (the stub had no prior test surface; the four direct tests are the RED→GREEN proof)
2. **Task 2: Wire the `direct` kernel flag + the AO-direct == in-core equivalence integration test** - `ad8df49` (feat)

**Plan metadata:** (this commit) (docs: complete plan)

## Files Created/Modified
- `crates/pyscf-ccsd/src/direct.rs` - AO-direct vvvv contraction: `contract_vvvv_t2_aodirect` (AO int2e source, per-`a` slice) + `contract_vvvv_t2_from_eris` (eris.vvvv source, equivalence anchor) + the in-core full-`nv^4` oracle (test-only) + 4 tests (equivalence, `nv^3`-not-`nv^4` peak, thread-invariant, bad-shape).
- `crates/pyscf-ccsd/src/ccsd.rs` - `ccsd_kernel_direct` / `ccsd_kernel_direct_diis` (AO-direct kernel), `estimate_direct_vvvv_bytes` (lower `nv^3` pre-flight), clippy `absurd_extreme_comparisons` fix on the DIIS start-cycle guard.
- `crates/pyscf-ccsd/src/update_amps.rs` - refactored the vvvv step into a precomputed `[no,no,nv,nv]` block + `update_amps_core` (shared) + `vvvv_step_from_wvvvv` (in-core) + `default_update_amps_direct` (AO-direct).
- `crates/pyscf-ccsd/src/lib.rs` - re-export the direct kernels + AO-direct contraction entrypoints + `default_update_amps_direct`.
- `crates/pyscf-ccsd/tests/direct.rs` - CCSD-07 integration test: AO-direct == in-core `e_corr` (LiH/STO-3G) + lower-reservation memory-frugality proof.
- `.planning/phases/06-ccsd/deferred-items.md` - marked the `absurd_extreme_comparisons` lint resolved.

## Decisions Made
See `key-decisions` frontmatter above (Open Q4 path-b resolution, the vvvv step split, the separate direct kernel entrypoint, and the two-level equivalence + memory-frugality proofs).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed clippy `absurd_extreme_comparisons` on the DIIS start-cycle guard**
- **Found during:** Task 2 (already editing `ccsd.rs` to wire the direct flag — the incidental fix the plan's CRITICAL_PROJECT_CONSTRAINTS mandated)
- **Issue:** `Some(stack) if istep >= DIIS_START_CYCLE` where `DIIS_START_CYCLE: usize = 0` is always true; CI runs `clippy -D warnings` so this would fail.
- **Fix:** Kept the configurable `>=` start-cycle semantics (raise the const → DIIS starts later) under a documented `#[allow(clippy::absurd_extreme_comparisons)]` with rationale (the plan explicitly permits this when the start-cycle is meant to stay configurable). Confirmed gone via `cargo clippy -p pyscf-ccsd --tests | grep -i absurd` (no match).
- **Files modified:** `crates/pyscf-ccsd/src/ccsd.rs`, `.planning/phases/06-ccsd/deferred-items.md`
- **Verification:** `cargo clippy -p pyscf-ccsd --tests` no longer reports `absurd`.
- **Committed in:** `ad8df49` (Task 2 commit)

**2. [Rule 3 - Blocking] vvvv-step refactor of `update_amps` to make the step swappable**
- **Found during:** Task 2 (the direct path cannot reuse `default_update_amps_with_wvvvv` as-is — it always builds the full `nv^4` `cc_Wvvvv`).
- **Issue:** The in-core `update_amps` body hard-coded the `einsum('abcd,ijcd->ijab', Wvvvv, tau)` step, so routing it through AO-direct without duplicating the entire amplitude equation required extracting the step.
- **Fix:** Refactored the vvvv step into a precomputed `[no,no,nv,nv]` block consumed by a new private `update_amps_core`; `default_update_amps_with_wvvvv` (in-core) builds the block from `cc_Wvvvv` (byte-unchanged math via `vvvv_step_from_wvvvv`), and `default_update_amps_direct` builds it on-the-fly. No behavior change to the in-core path.
- **Files modified:** `crates/pyscf-ccsd/src/update_amps.rs`
- **Verification:** `cargo test -p pyscf-ccsd --test rccsd_numeric_smoke --test uccsd_smoke --test diis_amps` + the full crate suite all green (39 lib + all integration); the AO-direct `e_corr` is bit-identical to in-core.
- **Committed in:** `ad8df49` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug-fix mandated by the plan, 1 blocking refactor)
**Impact on plan:** The clippy fix was an explicit plan requirement; the refactor was necessary to route the vvvv step through AO-direct without duplicating `update_amps`, and the in-core path is byte-unchanged. No scope creep.

## Issues Encountered
- H2/STO-3G (the rccsd_numeric_smoke system) has `nvir=1` so `nv^4 == nv^3 == 1` — too small to distinguish the AO-direct vs in-core reservation. Resolved by using LiH/STO-3G (`nocc=2, nvir=4`: `nv^4=256`, `nv^3=64`) for the integration test — the same system the frozen-core tests use, big enough that `nv^3 < nv^4` is a clean memory-frugality witness.

## Known Stubs
None — the AO-direct path is a real converging contraction, not a placeholder. The v1 path-b semantics (full AO `int2e` held, MO `vvvv` tiled per-`a`) are documented in the `direct.rs` module doc-comment as the deliberate Open-Q4 resolution; a shell-sliced streaming primitive (which would also stream the AO source) is the documented v2 upgrade once `pyscf-gto` grows a `shls_slice` API.

## Threat Flags
None — pure internal contraction. No new network/auth/session/crypto surface (per the plan's threat model). T-06-08-SHAPE (ShapeMismatch before indexing), T-06-08-OOM (lower `try_reserve` + HARD refusal preserved), and T-06-08-FP (oracle reductions + the equivalence anchor + RAYON 1==8) are all mitigated as planned.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CCSD-07 complete: `direct=True` AO-direct CCSD converges to the in-core energy with a lower peak reservation. Both explicit ERI-mode opt-ins the D-01 refusal points to are now half-shipped (AO-direct done; DF-CCSD is 06-09).
- 06-09 (DF-CCSD + HDF5 spill) is the next Wave-6 plan — it swaps the ERI source to the DF `vvL` B-tensor and adds the Spilled backend; it can reuse the lower-reservation arena discipline established here.
- Live upstream-PySCF `direct=True` byte-identity remains the 06-08-closeout `workflow_dispatch` human-verify arm (the sandbox has no PySCF), consistent with the rest of Phase 06.

## Self-Check: PASSED

- Created files verified on disk: `crates/pyscf-ccsd/src/direct.rs`, `crates/pyscf-ccsd/tests/direct.rs`, `crates/pyscf-ccsd/src/ccsd.rs`, `crates/pyscf-ccsd/src/update_amps.rs`, `.planning/phases/06-ccsd/06-08-SUMMARY.md` — all FOUND.
- Task commits verified in git log: `d171b10` (Task 1), `ad8df49` (Task 2) — both FOUND.
- `cargo test -p pyscf-ccsd --tests` green (39 lib + all integration, including `tests/direct.rs` 2/2); `cargo clippy -p pyscf-ccsd --tests | grep -i absurd` no match.

---
*Phase: 06-ccsd*
*Completed: 2026-05-25*
