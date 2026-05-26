---
phase: 07-gradients-geomopt
plan: 01
subsystem: api
tags: [gradients, cintx, intor, ecp, int2e_ip1, int1e_ecp_ipnuc, int3c2e_ip1, GRAD-07]

# Dependency graph
requires:
  - phase: 02-gto-integrals
    provides: "intor dispatcher (evaluate_arity2/arity4), layout_table, CintxEcpEngine scalar ECP path"
provides:
  - "int2e_ip1 component-leading arity-4 dispatch wired in pyscf-gto/src/intor.rs (NotYetImplemented{phase:7} guard removed)"
  - "CintxEcpEngine::ecp_int1e_ipnuc — ECP-gradient [3,nao,nao] dispatch (cintx-ready, un-gated)"
  - "Clean cintx-availability error contract for the 6 missing gradient families (never NotYetImplemented{phase:7})"
  - "grad_intor_smoke.rs round-trip for the two cintx-ready families (int3c2e_ip1 + int1e_ecp_ipnuc)"
  - "Live cintx 2-ready/6-missing gradient-integral availability split (recorded below)"
affects: [07-03-rhf-grad, 07-04-ecp-cphf, 07-05-uhf-rks-uks-grad, 07-06-mp2-grad, 07-07-ccsd-grad, 07-08-df-grad]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Component-leading arity-4 stitch: [c, nao, nao, nao, nao] F-order, component axis leading (mirrors stitch_arity2_block)"
    - "cintx-availability error (Core::InvalidMolecule) replaces NotYetImplemented{phase:7} for structurally-wired-but-unshipped families"
    - "ECP-gradient via dedicated EcpEngine::ecp_int1e_ipnuc trait method (Density carries 3*nao*nao component-leading buffer)"

key-files:
  created:
    - crates/pyscf-gto/tests/grad_intor_smoke.rs
  modified:
    - crates/pyscf-gto/src/intor.rs
    - crates/pyscf-gto/src/ecp_engine_cintx.rs
    - crates/pyscf-gto/tests/intor_smoke.rs
    - crates/pyscf-gto/tests/ecp_int1e_oracle.rs
    - crates/pyscf-gto/tests/ecp_engine_stub.rs

key-decisions:
  - "int2e_ip1 component-leading arity-4 dispatch is structurally wired but resolves to a clean cintx-availability error (int2e_ip1 MISSING from cintx); never NotYetImplemented{phase:7}"
  - "ECP-gradient ipnuc un-gated (cintx-ready); iprinv routes to a clean cintx-availability error (MISSING, no scheduled cintx workstream)"
  - "phase:7 disposition is CLOSED in pyscf-gto's intor + ECP-scalar paths; replaced with cintx-availability errors so downstream callers distinguish missing-family from not-implemented"
  - "layout_table needs NO edit — catalogue stays 23 entries (≥20 floor); ECPscalar routes through the ECP engine, not the table"

patterns-established:
  - "Structural-wiring-lands-regardless / numeric-gated-on-cintx split (D-02): the dispatch shape contract is wired now; numeric un-gating waits on cintx per-family availability"
  - "Per-family cintx-availability error message names the missing family + workstream status (downstream plans read which arms stay gated)"

requirements-completed: [GRAD-07]

# Metrics
duration: 39min
completed: 2026-05-26
---

# Phase 7 Plan 01: cintx Grad-Intor Buy-Down + GTO Guard Removal / ECP-Grad Wiring Summary

**Structurally wired the int2e_ip1 component-leading arity-4 dispatch and the ECP-gradient `int1e_ecp_ipnuc` ([3,nao,nao]) path in pyscf-gto, removing the `NotYetImplemented{phase:7}` guards; the 2 cintx-ready families (`int3c2e_ip1`, `int1e_ecp_ipnuc`) are un-gated with a round-trip smoke, and the 6 missing families route to clean cintx-availability errors.**

## Performance

- **Duration:** ~39 min (Task 1 checkpoint resolution → Tasks 2-3 execution)
- **Started:** 2026-05-26T11:16:18+09:00 (post-checkpoint resume)
- **Completed:** 2026-05-26T11:55:29+09:00
- **Tasks:** 2 executed (Task 1 was a verification-only human-verify checkpoint, resolved "approved")
- **Files modified:** 6 (1 created, 5 modified)

## cintx Gradient-Integral Availability Matrix (live, re-confirmed at execution time)

The Task 1 `checkpoint:human-verify` gate was resolved **"approved"** — gate the 6 missing families. The settled finding (re-confirmed against the live cintx manifest at `~/Documents/workspace/cintx/crates/cintx-ops/src/generated/api_manifest.rs` and `cintx-core/src/operator.rs`, matching 07-RESEARCH §"Gradient-Integral Availability Matrix" with **no drift**):

| Family | cintx symbol | Present today? | Disposition |
|--------|--------------|----------------|-------------|
| DF-grad 3-center | `int3c2e_ip1_sph` (manifest:353) | **YES** | **un-gated now** (READY) |
| ECP-grad ipnuc | `int1e_ecp_ipnuc_sph` (manifest:506) = `ECPscalar_ipnuc`; `OperatorId::INT1E_ECP_IPNUC_{CART,SPH}` (positions 28-29) | **YES** | **un-gated now** (READY) |
| 2e gradient | `int2e_ip1` (non-`cint`-prefixed) | **NO** | gated — clean cintx-availability error |
| 1e overlap grad | `int1e_ipovlp_sph` | **NO** | gated — clean cintx-availability error |
| 1e kinetic grad | `int1e_ipkin_sph` | **NO** | gated — clean cintx-availability error |
| 1e nuclear grad | `int1e_ipnuc_sph` | **NO** | gated — clean cintx-availability error |
| 1e rinv grad | `int1e_iprinv_sph` (+ no `with_rinv_at_nucleus` origin-shift param) | **NO** | gated — clean cintx-availability error |
| ECP-grad iprinv | `int1e_ecp_iprinv` / `ECPscalar_iprinv` | **NO** | gated — clean cintx-availability error |

**Split: 2 of 8 ready, 6 of 8 missing.** The 6 missing families are absent on every cintx branch checked (main, `fix/general-contraction-nctr-1e` @ c137b6e, and remote feature branches). **No cintx grad-integral workstream branch is scheduled** — record as "not yet scheduled; all 6 missing families gated (workflow_dispatch arm) until a future cintx workstream lands them."

### Which downstream numeric arms therefore stay gated

- **07-08 DF-grad (`int3c2e_ip1`)** — numeric **un-gates now** (cintx-ready).
- **07-04 ECP-grad `get_hcore` ipnuc term (`ECPscalar_ipnuc`)** — numeric **un-gates now** (cintx-ready).
- **07-03 RHF / 07-05 UHF·RKS·UKS / 07-06 MP2 / 07-07 CCSD gradient numerics** — depend on `int2e_ip1` (2e Pulay) + `int1e_ip{ovlp,kin,nuc}` (hcore/overlap Pulay) + `int1e_iprinv` (`hcore_generator`, needs the absent `with_rinv_at_nucleus` shift). **MISSING → numeric stays gated** (workflow_dispatch / byte-identity arm) until the cintx workstream lands them. The always-on **FD-structural gate (D-01, `verify_fd`)** proceeds regardless.
- **07-04 ECP-grad `hcore_deriv` iprinv term (`ECPscalar_iprinv`)** — MISSING → numeric stays gated.

## Accomplishments

- Removed the live `NotYetImplemented{phase:7}` guard in `evaluate_arity4`'s `ComponentLeadingFOrder` arm; wired the component-leading arity-4 stitch (`[c, nao, nao, nao, nao]` F-order, component axis leading — never `[..,c]`, T-07-01) mirroring `stitch_arity2_block`.
- Added `CintxEcpEngine::ecp_int1e_ipnuc` resolving `OperatorId::INT1E_ECP_IPNUC_{CART,SPH}` → component-leading `[3, nao, nao]` buffer (un-gated; the `ECPscalar_ipnuc` `get_hcore` grad term).
- Replaced the ECP-scalar-path `NotYetImplemented{phase:7}` derivative guard with a clean cintx-availability error; routed `iprinv`/`ECPscalar_iprinv` to a clean cintx-availability error in the new `ecp_int1e_ipnuc` method.
- Created `grad_intor_smoke.rs`: round-trips both cintx-ready families (`int3c2e_ip1` readiness + shell-triple eval; `int1e_ecp_ipnuc` end-to-end `[3,nao,nao]` on Cu/LANL2DZ with finite, non-zero values).
- Confirmed `layout_table::catalogue_meets_phase_2_floor` stays green (23 entries, no edit needed).
- Dependency wall (`check-dependency-wall`) passes — no `cubecl-*` leaked into pyscf-gto.

## Task Commits

1. **Task 2: Remove arity-4 guard + wire int2e_ip1 component-leading dispatch** — `df32741` (feat)
2. **Task 3: Wire ECP-gradient ipnuc + grad-intor smoke + confirm layout table** — `73e9dcc` (feat)

_Task 1 was a verification-only `checkpoint:human-verify` gate — zero commits (resolved "approved")._

## Files Created/Modified

- `crates/pyscf-gto/src/intor.rs` — removed arity-4 `NotYetImplemented{phase:7}` guard; component-leading arity-4 stitch ([c,nao,nao,nao,nao]); updated ECP-route doc comment for the cintx-availability contract.
- `crates/pyscf-gto/src/ecp_engine_cintx.rs` — added `ecp_int1e_ipnuc` ([3,nao,nao], INT1E_ECP_IPNUC); scalar-path derivative guard now a clean cintx-availability error; iprinv → cintx-availability error.
- `crates/pyscf-gto/tests/grad_intor_smoke.rs` — NEW: cintx round-trip smoke for the two ready families.
- `crates/pyscf-gto/tests/intor_smoke.rs` — added `int2e_ip1_dispatch_is_never_phase_7_not_yet_implemented`.
- `crates/pyscf-gto/tests/ecp_int1e_oracle.rs` — updated `int1e_ecp_ipnuc` rejection test for the cintx-availability contract.
- `crates/pyscf-gto/tests/ecp_engine_stub.rs` — updated `int1e_ecp_iprinv` (scalar path) test for the cintx-availability contract.

## Decisions Made

- **phase:7 disposition closed in the intor + ECP-scalar dispatch paths.** Replaced with `Core(InvalidMolecule(..))` cintx-availability errors whose message names the missing family + its (unscheduled) cintx-workstream status. This lets downstream callers distinguish "cintx hasn't shipped this family" from "pyscf-rs hasn't implemented this phase," which is the correct semantics post-structural-wiring (D-02).
- **ECP gradient routes through the dedicated `EcpEngine::ecp_int1e_ipnuc` trait method**, not the scalar `intor()` entry-point. `Density` carries the `3*nao*nao` component-leading buffer with `nao` on the AO axis. The scalar `intor()` path correctly rejects derivative names (WR-01 preserved).
- **No `layout_table.rs` edit** — ECPscalar routes through the ECP engine, not the table; the catalogue already carries the `int2e_ip1_sph`/`int3c2e_ip1_sph`/`int1e_ip*_sph` component-leading entries (23 total, ≥20 floor).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated two pre-existing tests for the contract this plan changes**
- **Found during:** Task 3 (ECP-gradient wiring)
- **Issue:** `ecp_int1e_oracle.rs::cu_lanl2dz_int1e_ecp_ipnuc_is_rejected_not_silently_scalar` and `ecp_engine_stub.rs::int1e_ecp_iprinv_returns_phase_7_not_yet_implemented` asserted `NotYetImplemented{phase:7}` for ECP derivative names routed through the scalar `intor()` path. Task 3 deliberately closes that disposition (replaces it with a cintx-availability error), so the assertions would fail against the new, correct contract.
- **Fix:** Rewrote both tests to assert the family is NEVER `NotYetImplemented{phase:7}` and instead surfaces a clean `Core(InvalidMolecule(..))` cintx-availability error. Renamed them to reflect the new contract. The stub's DEFAULT `ecp_int1e_ipnuc` (trait default, called directly) still returns `NotYetImplemented{phase:7}` — `stub_ipnuc_returns_phase_7_not_yet_implemented` left unchanged and still green.
- **Files modified:** `crates/pyscf-gto/tests/ecp_int1e_oracle.rs`, `crates/pyscf-gto/tests/ecp_engine_stub.rs`
- **Verification:** `cargo test -p pyscf-gto --locked -- --test-threads=1` — 0 failures.
- **Committed in:** `73e9dcc` (Task 3 commit)

**2. [Rule 1 - Bug] Relaxed the int3c2e_ip1 round-trip block-size assertion to match cintx's safe-API contract**
- **Found during:** Task 3 (grad_intor_smoke creation)
- **Issue:** The first draft asserted the `int3c2e_ip1_sph` shell-triple `owned_values` block carried 3 values (3 derivative components). cintx's safe API returns the inner AO block (extents `[1,1,1]`, 1 value) WITHOUT expanding the component axis into `owned_values` at this surface — the same synthetic-staging shape the scalar dispatcher already tolerates via `stitch_arity2_block`'s `expected_inner` branch. The component-leading `[3,...]` repack is owned by the pyscf-gto dispatcher, not the cintx safe-API block.
- **Fix:** Round-trip now asserts the block is non-empty + finite and its length equals the inner AO block (or its 3-component expansion if cintx later materializes it); the `[3,...]` component-leading contract is asserted at the layout-table level in `int3c2e_ip1_sph_is_cintx_ready_and_component_leading`.
- **Files modified:** `crates/pyscf-gto/tests/grad_intor_smoke.rs`
- **Verification:** `cargo test -p pyscf-gto --locked --test grad_intor_smoke -- --test-threads=1` — 5 passed.
- **Committed in:** `73e9dcc` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 — both directly caused by this plan's contract change / cintx safe-API behavior).
**Impact on plan:** Both auto-fixes keep the test suite consistent with the new (correct) cintx-availability contract and cintx's actual block shape. No scope creep — `int2e_ip1` structural shape contract is wired exactly as specified; only the (correct) error type and a test's block-size expectation changed.

## Issues Encountered

- `check_dependency_wall` binary is named `check-dependency-wall` (hyphens), not `check_dependency_wall` — invoked with the correct name; lint PASS.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **07-03 (RHF grad), 07-05 (UHF/RKS/UKS), 07-06 (MP2 grad), 07-07 (CCSD grad):** structural method-body wiring + the always-on FD `verify_fd` gate can proceed now; their upstream-byte-identity NUMERIC arms stay gated (workflow_dispatch) on the missing `int2e_ip1` + `int1e_ip{ovlp,kin,nuc,rinv}` families until a future (unscheduled) cintx grad-integral workstream lands them.
- **07-08 (DF grad):** `int3c2e_ip1` numeric is un-gated (cintx-ready) — DF-grad can wire numeric now.
- **07-04 (ECP grad):** `ECPscalar_ipnuc` (`get_hcore` term) numeric un-gated; `ECPscalar_iprinv` (`hcore_deriv` per-atom term) stays gated (MISSING).
- **Coordination note:** every downstream grad plan MUST keep its upstream-byte-identity numeric arm gated for the 6 missing families and pair any "drop the gate" with a cintx-side availability note (D-02 hinge honored).

## Self-Check: PASSED

- All 6 created/modified source+test files exist on disk.
- Both task commits (`df32741`, `73e9dcc`) exist in git history.
- `cargo test -p pyscf-gto --locked -- --test-threads=1`: 0 failures.
- `check-dependency-wall`: PASS.

---
*Phase: 07-gradients-geomopt*
*Completed: 2026-05-26*
