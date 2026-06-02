---
phase: quick-260601-rhc
plan: 01
subsystem: api
tags: [ecp, gradients, cintx, iprinv, rinv, integrals]

# Dependency graph
requires:
  - phase: 07-gradients-geomopt
    provides: "CintxEcpEngine::ecp_int1e_ipnuc + pyscf-grad::{get_hcore_ecp, hcore_deriv_ecp} (the iprinv gate this un-gates)"
  - phase: 02-gto
    provides: "EcpEngine trait + CintxEcpEngine + projection::build_cintx_basis_set_with_ecp"
provides:
  - "EcpEngine::ecp_int1e_iprinv trait method (per-atom rinv origin; default EcpEngineNotAvailable)"
  - "CintxEcpEngine::ecp_int1e_iprinv: native ecp_iprinv via Resolver::descriptor_by_symbol(..).id + ExecutionOptions.rinv_orig"
  - "pyscf-grad::hcore_deriv_ecp returns the real per-atom [3,nao,nao] iprinv buffer (no longer a hardcoded cintx-availability error)"
affects: [F-08 (analytic ECP-gradient force assembly), grad-oracle-upstream-manual]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "iprinv operator resolution via cintx manifest (Resolver::descriptor_by_symbol(symbol).id) — cintx_core has no iprinv core const"
    - "per-atom rinv origin threaded via ExecutionOptions { rinv_orig: Some(coord_bohr), ..Default::default() }"

key-files:
  created: []
  modified:
    - crates/pyscf-core/src/traits.rs
    - crates/pyscf-gto/src/ecp_engine_cintx.rs
    - crates/pyscf-grad/src/ecp.rs
    - crates/pyscf-gto/tests/grad_intor_smoke.rs
    - crates/pyscf-gto/tests/ecp_engine_stub.rs
    - crates/pyscf-grad/tests/ecp_verify_fd.rs

key-decisions:
  - "iprinv got a DEDICATED EcpEngine::ecp_int1e_iprinv method (per-atom rinv-origin semantics) rather than overloading ecp_int1e_ipnuc (all-slot accumulation)"
  - "ExecutionOptions struct literal kept on one line via #[rustfmt::skip] so the F-05 grep gate matches the LIVE constructor (rustfmt unconditionally expands ..rest spreads)"
  - "The in-tree iprinv@Cu == ipnuc compare is documented as a self-consistency / structural smoke (cintx-vs-cintx), NOT an external oracle; the upstream byte-identity lives in cintx ecp_iprinv_parity.rs"

patterns-established:
  - "F-05: ECP per-atom gradient integral un-gated; the iprinv hcore_deriv term is now a real integral, F-08 (force assembly) still out of scope"

requirements-completed: [F-05, GRAD-07]

# Metrics
duration: 18min
completed: 2026-06-01
---

# Phase quick-260601-rhc: Un-gate int1e_ecp_iprinv (ECP-gradient per-atom term) Summary

**`int1e_ecp_iprinv` / `ECPscalar_iprinv` now evaluates through `CintxEcpEngine::ecp_int1e_iprinv` (native cintx 21-07 `ecp_iprinv` kernel, per-atom rinv origin) and `pyscf-grad::hcore_deriv_ecp` returns the real per-atom `[3,nao,nao]` buffer — closing F-05.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-01
- **Completed:** 2026-06-01
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Added `EcpEngine::ecp_int1e_iprinv` default trait method (per-atom rinv origin; defaults to `EcpEngineNotAvailable` so the stub and any other impl stay valid unchanged).
- Implemented `CintxEcpEngine::ecp_int1e_iprinv`: resolves `int1e_ecp_iprinv_{cart,sph}` via `Resolver::descriptor_by_symbol(..).id`, sets `ExecutionOptions.rinv_orig = Some(rinv_origin)` (Bohr), stitches the component-leading `[3,nao,nao]` buffer (byte-identical to the ipnuc stitch). Spinor iprinv fails closed (`NotYetImplemented{phase:3}`); the legacy libcint-FFI resolver path is untouched.
- Wired `pyscf-grad::hcore_deriv_ecp` to the new method (rinv origin = `mol.atom_coord(atm_id)`), normalising the engine's component-INNER layout to RHF component-leading F-order; `EcpEngineNotAvailable` (non-ECP atom / molecule) maps to an all-zero buffer, other errors propagate.
- Flipped the two stale gated-behavior tests to un-gated assertions and refreshed the WR-01 scalar-path guard doc-comment (assertion retained).

## Task Commits

Each task was committed atomically (code only; docs handled separately by the orchestrator):

1. **Task 1: trait method + un-gate CintxEcpEngine::ecp_int1e_iprinv** - `4521323` (feat)
2. **Task 2: wire pyscf-grad::hcore_deriv_ecp to the real per-atom iprinv buffer** - `04d956d` (feat)
3. **Task 3: flip the three stale tests (2 flipped, 1 doc-only)** - `e384188` (test)

## Files Created/Modified
- `crates/pyscf-core/src/traits.rs` - new `EcpEngine::ecp_int1e_iprinv` default trait method.
- `crates/pyscf-gto/src/ecp_engine_cintx.rs` - `CintxEcpEngine::ecp_int1e_iprinv` impl (Resolver + rinv_orig); refreshed module + ipnuc/scalar gate doc-comments (iprinv no longer "missing").
- `crates/pyscf-grad/src/ecp.rs` - `hcore_deriv_ecp` now returns the real per-atom iprinv buffer; refreshed module + fn doc-comments.
- `crates/pyscf-gto/tests/grad_intor_smoke.rs` - 3 new un-gated tests (`ecp_iprinv_evaluates_real_per_atom_buffer`, `ecp_iprinv_at_cu_equals_ipnuc_single_nucleus`, `ecp_iprinv_origin_matching_no_atom_is_all_zeros`); dropped now-unused `PyscfRsError` import.
- `crates/pyscf-gto/tests/ecp_engine_stub.rs` - WR-01 scalar-path doc-comment refresh (InvalidMolecule assertion RETAINED).
- `crates/pyscf-grad/tests/ecp_verify_fd.rs` - flipped to `ecp_iprinv_per_atom_term_returns_real_buffer` + added He non-ECP zero anchor; removed now-unused helper; updated the `#[ignore]`'d end-to-end FD reason.

## Decisions Made
- **Dedicated method over overload:** iprinv carries per-atom rinv-origin semantics distinct from ipnuc's all-slot accumulation, so `ecp_int1e_iprinv` is a new trait method, not an `ecp_int1e_ipnuc` overload.
- **`#[rustfmt::skip]` on the `ExecutionOptions` literal:** rustfmt unconditionally expands struct literals containing a `..rest` spread across multiple lines, which would defeat the hardened single-line grep gate `grep -c 'ExecutionOptions { rinv_orig: Some'`. Pinning the binding with `#[rustfmt::skip]` keeps the LIVE constructor on one line (gate count = 1) while staying `cargo fmt --check` clean.
- **In-tree self-consistency, not an external oracle:** the `iprinv@Cu == ipnuc` (atol 1e-12) compare is documented honestly as a cintx-vs-cintx structural smoke (single-ECP-atom degeneracy); the external byte-identity vs upstream `nr_ecp_deriv` is owned by cintx's own `ecp_iprinv_parity.rs`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `#[rustfmt::skip]` to keep the F-05 grep gate satisfiable**
- **Found during:** Task 1
- **Issue:** rustfmt always reflows a struct literal with a `..Default::default()` spread onto multiple lines, so the plan's hardened gate `grep -c 'ExecutionOptions { rinv_orig: Some'` returned 0 after `cargo fmt` despite a correct, present constructor — the gate and fmt were mutually unsatisfiable as written.
- **Fix:** Built the options once into a `let opts = ExecutionOptions { rinv_orig: Some(rinv_origin), ..Default::default() };` binding annotated `#[rustfmt::skip]` (kept on one line), and passed `opts.clone()` into each `SessionRequest::new`. Gate count is now 1 (live constructor) and `cargo fmt --check` is clean.
- **Files modified:** crates/pyscf-gto/src/ecp_engine_cintx.rs
- **Verification:** `grep -c 'ExecutionOptions { rinv_orig: Some' ... ` = 1; `cargo +nightly fmt --check` clean.
- **Committed in:** `4521323` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking).
**Impact on plan:** No scope creep — purely a fmt/grep-gate reconciliation. All plan behavior delivered as specified.

## Issues Encountered
- **Task 2 negated-grep gate is broader than intended.** The plan's literal Task-2 verify includes `! grep -n 'engine.ecp_int1e_ipnuc' crates/pyscf-grad/src/ecp.rs`, but `get_hcore_ecp` (a SEPARATE, correct function) legitimately calls `engine.ecp_int1e_ipnuc(mol, "ECPscalar_ipnuc")` and MUST keep doing so. The plan's `<done>` clarifies the intent is that the OLD ipnuc call **inside `hcore_deriv_ecp`** is gone. Verified by a scoped check: the `hcore_deriv_ecp` body contains 0 `ecp_int1e_ipnuc` calls (now calls `ecp_int1e_iprinv`); the only remaining ipnuc call is the intended `get_hcore_ecp` one. The substantive done-criterion is met.
- **Pre-existing out-of-scope clippy warning** (`clippy::doc_lazy_continuation` in `crates/pyscf-gto/tests/spinor_intor.rs:7`) surfaced in the Task-3 clippy run — NOT caused by this task; logged to `deferred-items.md` and left untouched (SCOPE BOUNDARY). My six edited files produce no clippy lints.

## Verify Results

- **Task 1:** `cargo +nightly build -p pyscf-gto --locked` clean; `grep -c 'ExecutionOptions { rinv_orig: Some'` = 1; named smoke `ecp_iprinv_evaluates_real_per_atom_buffer` ... ok (run in Task 3).
- **Task 2:** `cargo +nightly build -p pyscf-grad --locked` clean; `grep -n 'engine.ecp_int1e_iprinv'` = the live call at ecp.rs:148; scoped check confirms 0 ipnuc calls inside `hcore_deriv_ecp` (the broader literal negated-grep is a known plan imprecision — `get_hcore_ecp`'s ipnuc call is intended).
- **Task 3:** `cargo +nightly test -p pyscf-gto -p pyscf-grad --locked` — all pass, 0 failed (the 6 new/updated ECP tests all green, including `ecp_iprinv_at_cu_equals_ipnuc_single_nucleus` at atol 1e-12). `cargo +nightly clippy -p pyscf-gto -p pyscf-grad --locked --all-targets` clean on edited code (only the pre-existing out-of-scope `spinor_intor.rs` warning). `cargo +nightly fmt --check` clean.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- F-05 closed: the per-atom ECP-gradient integral is now a real `[3,nao,nao]` buffer through `hcore_deriv_ecp`.
- **F-08 (analytic ECP-gradient force assembly) remains out of scope** — it now has a real integral to consume, but the end-to-end `ecp_verify_fd_numeric` FD arm stays `#[ignore]`'d on the still-missing base grad-intor families (`int2e_ip1` + `int1e_ip{ovlp,kin,nuc,rinv}`).
- **Caveat (carried from orientation):** the in-tree tests prove cintx *evaluates* iprinv and that iprinv self-consistently matches ipnuc for the single-ECP-atom fixture; the *derivative-correct* byte-identity vs upstream PySCF is owned by cintx's `ecp_iprinv_parity.rs`, not this in-tree suite.

## Self-Check: PASSED

All 6 modified source files + the SUMMARY exist on disk; all 3 task commits (`4521323`, `04d956d`, `e384188`) are present in git history.

---
*Phase: quick-260601-rhc*
*Completed: 2026-06-01*
