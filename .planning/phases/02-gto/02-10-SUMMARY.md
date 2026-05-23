---
phase: 02-gto
plan: 10
subsystem: api
tags: [ecp, cintx, int1e_ecp, integrals, gto, basis-set]

# Dependency graph
requires:
  - phase: 02-gto (02-07)
    provides: ECP loading (format_ecp + make_ecp_env), EcpEngine trait, EcpEngineNotAvailable stub, intor dispatcher ECP routing
  - phase: 02-gto (02-05)
    provides: intor::evaluate_arity2 cintx SessionRequest shell-pair pattern (mirrored by the ECP engine)
  - phase: external (cintx Phase 19/20)
    provides: int1e_ecp_{cart,sph} Type-1 + Type-2 projector integrals byte-identical to vendored PySCF nr_ecp (atol=1e-12)
provides:
  - "ecp_engine_cintx::CintxEcpEngine — real EcpEngine impl backed by cintx ECP evaluation"
  - "projection::build_cintx_basis_set_with_ecp — ECP-augmented cintx BasisSet builder (BasisSet::try_new_with_ecp)"
  - "Density::from_flat helper in pyscf-core"
  - "in-tree int1e_ecp gate (ecp_int1e_oracle.rs) + upstream byte-identity pytest (test_ecp_int1e.py)"
  - "GTO-05 fully closed: ECP loading + int1e_ecp evaluation"
affects: [phase-07-grad (GRAD-07 ECP gradients via int1e_ecp_ipnuc), phase-03-scf (ECP-bearing molecules in HF)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ECP-augmented BasisSet: build on demand via build_cintx_basis_set_with_ecp because the cintx safe-API ECP preflight requires basis.ecp_shells() non-empty (mol.basis_set is ECP-free)"
    - "EcpEngine swap is a single integration point (ecp_engine()) — the intor dispatcher routes through the trait, so the impl change is the only edit"
    - "ECP-less mol returns canonical EcpEngineNotAvailable via mol._ecp.is_empty() guard (preserves 02-07 user-facing contract)"

key-files:
  created:
    - crates/pyscf-gto/src/ecp_engine_cintx.rs
    - crates/pyscf-gto/tests/ecp_int1e_oracle.rs
    - tests/oracle/test_ecp_int1e.py
  modified:
    - crates/pyscf-core/src/density.rs
    - crates/pyscf-gto/src/projection.rs
    - crates/pyscf-gto/src/lib.rs
    - crates/pyscf-gto/tests/ecp_engine_stub.rs
    - crates/pyscf-gto/tests/intor_smoke.rs
    - crates/pyscf-gto/tests/dump_intor_for_oracle.rs

key-decisions:
  - "cintx is a PATH dep already pointing at the merged Phase-19 ECP tree — Cargo.lock UNCHANGED, no git-rev pin bump needed (the plan's git-rev sketch was superseded by Phase-1 D-15 path-dep topology)"
  - "Build an ECP-augmented BasisSet on demand (build_cintx_basis_set_with_ecp) rather than changing mol.basis_set: the cintx safe-API ECP preflight returns FacadeError::MissingEcpBasis unless basis.ecp_shells() is non-empty, but mol.basis_set is built ECP-free"
  - "ECP-less molecule int1e_ecp returns EcpEngineNotAvailable via a mol._ecp.is_empty() guard, keeping the 02-07 error contract"
  - "Upstream byte-identity pytest could not run in-sandbox (no numpy/upstream-pyscf) — shipped + downgraded to a Manual-Only verify item; cintx already pins atol=1e-12 vs nr_ecp at source"

patterns-established:
  - "ECP shell projection mirrors make_ecp_env: one cintx EcpShell per (atom, channel, distinct n_power); EcpShell.l sentinel (-1=local) maps to EcpChannel::Local / Projected(l)"
  - "Stub demotion: a superseded trait impl stays in-tree as documentation + a testable error path, exercised DIRECTLY (not via the default accessor)"

requirements-completed: [GTO-05]

# Metrics
duration: 38min
completed: 2026-05-23
---

# Phase 2 Plan 10: GTO-05 ECP Evaluation Gap-Closure Summary

**`int1e_ecp` now evaluates through a cintx-backed `CintxEcpEngine` — Cu/LANL2DZ returns a finite, non-zero, symmetric matrix end-to-end through `mol.intor("int1e_ecp")`, closing the evaluation half of GTO-05.**

## Performance

- **Duration:** ~38 min
- **Started:** 2026-05-23 (sequential executor, branch fix/ci-local-gates)
- **Completed:** 2026-05-23
- **Tasks:** 1 (Task 0 gate pre-resolved PROCEED by project owner)
- **Files modified:** 9 (3 created, 6 modified)

## Accomplishments

- New `ecp_engine_cintx::CintxEcpEngine` implements `EcpEngine` against the REAL cintx ECP safe API (`OperatorId::INT1E_ECP_{SPH,CART}`, `SessionRequest`), replacing `EcpEngineNotAvailable` as the default `pyscf_gto::ecp_engine()`.
- New `projection::build_cintx_basis_set_with_ecp` projects per-element `mol._ecp` `ParsedEcp` into typed cintx `EcpShell`s and attaches them via `BasisSet::try_new_with_ecp` — the missing link that lets the cintx safe-API ECP preflight (`MissingEcpBasis`) pass.
- Always-on in-tree gate `crates/pyscf-gto/tests/ecp_int1e_oracle.rs`: Cu/LANL2DZ `int1e_ecp` is finite, non-zero, and symmetric. `cargo test -p pyscf-gto --test ecp_int1e_oracle` exits 0.
- Upstream byte-identity pytest `tests/oracle/test_ecp_int1e.py` + `dump_intor_for_oracle` ECP support shipped (gated on the oracle venv; cintx already pins atol=1e-12 vs nr_ecp).
- GTO-05 fully closed: loading (02-07) + evaluation (02-10). Phase 7 GRAD-07 (ECP gradients via `int1e_ecp_ipnuc_*`) is now unblocked.

## Task Commits

Task 0 (`checkpoint:human-action` gate) was pre-resolved PROCEED by the project owner; executed Task 1 directly, split into atomic logical commits:

1. **`Density::from_flat` helper** — `f547b88` (feat)
2. **ECP-augmented cintx BasisSet builder** — `5d6db6b` (feat)
3. **Swap EcpEngine to CintxEcpEngine + lib wiring** — `f4ca0c7` (feat)
4. **int1e_ecp oracle gate + stub/smoke test updates** — `415788b` (test)
5. **Upstream byte-identity oracle harness** — `f9587c9` (test)

**Plan metadata:** (this commit) — docs: complete plan

## Files Created/Modified

- `crates/pyscf-gto/src/ecp_engine_cintx.rs` (created) — `CintxEcpEngine`: builds an ECP-augmented BasisSet, resolves the ECP operator id, iterates AO shell pairs through `SessionRequest`, stitches an F-order nao×nao matrix into `Density::from_flat`.
- `crates/pyscf-gto/src/projection.rs` (modified) — added `build_cintx_basis_set_with_ecp` + extracted shared `build_atoms_and_shells`.
- `crates/pyscf-gto/src/lib.rs` (modified) — `pub mod ecp_engine_cintx`, `pub use CintxEcpEngine`, `ecp_engine()` now returns `CintxEcpEngine`.
- `crates/pyscf-core/src/density.rs` (modified) — `Density::from_flat(nao, data)` helper.
- `crates/pyscf-gto/tests/ecp_int1e_oracle.rs` (created) — in-tree finite/non-zero/symmetric gate.
- `crates/pyscf-gto/tests/ecp_engine_stub.rs` (modified) — instantiate `EcpEngineNotAvailable` DIRECTLY; ECP-less-mol dispatcher tests still assert `EcpEngineNotAvailable` via the engine's empty-ECP guard.
- `crates/pyscf-gto/tests/intor_smoke.rs` (modified) — clarified ECP-route tests assert the ECP-less-mol contract under the cintx engine.
- `crates/pyscf-gto/tests/dump_intor_for_oracle.rs` (modified) — read optional `PYSCF_RS_ORACLE_ECP`.
- `tests/oracle/test_ecp_int1e.py` (created) — Cu/LANL2DZ `int1e_ecp` byte-identity vs upstream (auto-skips without the oracle venv).

## Decisions Made

- **No cintx pin bump.** `Cargo.toml` keeps `cintx = { path = "../cintx" }`; the path dep already resolves to the merged Phase-19/20 ECP tree (verified: `git -C ../cintx log` shows phase 20 complete; `cargo update` produced no Cargo.lock change). The plan's git-rev bump sketch was superseded by the Phase-1 D-15 path-dep topology.
- **ECP-augmented BasisSet built on demand.** `mol.basis_set` is built ECP-free (`build_cintx_basis_set` → `BasisSet::try_new`). The cintx safe-API ECP preflight returns `FacadeError::MissingEcpBasis` unless `basis.ecp_shells()` is non-empty, so the engine constructs an ECP-bearing `BasisSet` via the new `build_cintx_basis_set_with_ecp`.
- **ECP-less mol → `EcpEngineNotAvailable`.** A `mol._ecp.is_empty()` guard preserves the 02-07 user-facing error for "ask for ECP on a molecule without one".

## Deviations from Plan

The plan's Task 1 Rust/Python code blocks were explicitly labeled speculative ("hypothetical post-merge cintx API"). The real cintx ECP surface differed materially, as anticipated by the deviation protocol. These are EXPECTED deviations, implemented against the real API:

### Implemented against the real API (expected per `<plan_code_is_speculative>` + `<deviation_protocol>`)

**1. [Rule 3 - Blocking] ECP-augmented BasisSet required (`MissingEcpBasis` preflight)**
- **Found during:** Task 1 (writing the engine).
- **Issue:** The plan's sketch used `mol.cintx_basis()` (the ECP-free stored BasisSet) + `OperatorId::for_symbol(name)` (no such method) + `ShellTuple::from_indices` (no such method). The cintx safe API gates ECP operators on `basis.ecp_shells()` being non-empty (`FacadeError::MissingEcpBasis`), which the stored basis is not.
- **Fix:** Added `projection::build_cintx_basis_set_with_ecp` (projects `mol._ecp` → cintx `EcpShell`s via `BasisSet::try_new_with_ecp`); used the typed `OperatorId::INT1E_ECP_{SPH,CART}` constants; used `basis.shell_tuple_for_indices([i, j])` (the real method, same as `intor::evaluate_arity2`).
- **Files modified:** `crates/pyscf-gto/src/projection.rs`, `crates/pyscf-gto/src/ecp_engine_cintx.rs`.
- **Verification:** `cargo test -p pyscf-gto --test ecp_int1e_oracle` (2 tests pass — finite/non-zero + symmetric).
- **Committed in:** `5d6db6b`, `f4ca0c7`.

**2. [Rule 2 - Missing Critical] `Density::from_flat` helper added**
- **Found during:** Task 1.
- **Issue:** The plan assumed `Density::from_flat(values, extents)` exists; it did not. The dispatcher (`intor.rs`) reads `density.data` (flat `nao*nao`) + `density.nao`.
- **Fix:** Added `Density::from_flat(nao, data)` to `pyscf-core::density` (signature matches what the dispatcher consumes — `nao` then flat data).
- **Files modified:** `crates/pyscf-core/src/density.rs`.
- **Committed in:** `f547b88`.

**3. [Plan-directed] Stub tests rewired to instantiate `EcpEngineNotAvailable` directly**
- **Issue:** `ecp_engine_stub.rs` (02-07) called `pyscf_gto::ecp_engine()`, which now returns `CintxEcpEngine`.
- **Fix:** The direct trait-method tests instantiate `EcpEngineNotAvailable` directly (per the plan's CRITICAL UPDATE choice (i)). The dispatcher-routing tests still pass unchanged because an ECP-less H/STO-3G mol triggers the engine's `mol._ecp.is_empty()` guard, returning `EcpEngineNotAvailable`.
- **Files modified:** `crates/pyscf-gto/tests/ecp_engine_stub.rs`, `crates/pyscf-gto/tests/intor_smoke.rs`.
- **Committed in:** `415788b`.

**4. [Deviation case (c)] Upstream byte-identity downgraded to Manual-Only**
- **Issue:** `tests/oracle/test_ecp_int1e.py` requires numpy + the vendored upstream pyscf (the whole oracle suite is gated on `tests/oracle/requirements.txt`), which are NOT installed in this sandbox. The existing `test_intor_oracle.py` has the identical numpy-import requirement, confirming this is the established repo gating, not a regression.
- **Fix:** Shipped the test (auto-skips via `conftest.py`); shipped the always-on in-tree gate (`ecp_int1e_oracle.rs`) as the primary regression guard. Downgraded the upstream byte-identity to a Manual-Only verify item in `02-VALIDATION.md`. cintx itself pins atol=1e-12 byte-identity vs vendored PySCF nr_ecp in `cintx-oracle/tests/safe_api_ecp_parity.rs`.
- **Files modified:** `tests/oracle/test_ecp_int1e.py`, `.planning/phases/02-gto/02-VALIDATION.md`.
- **Committed in:** `f9587c9` (test), this commit (validation doc).

**5. [Deviation] No `#[ignore = "Pending cintx ECP"]` annotations existed to remove**
- The acceptance criterion required `! grep -rn "Pending cintx ECP" crates/ tests/` to return 0 matches. It already does (0 matches) — the placeholder annotations the plan anticipated were never added to the codebase (02-07 shipped the stub without `#[ignore]` markers).

---

**Total deviations:** 5 (4 implemented against the real API per the speculative-code protocol; 1 environment-driven downgrade per deviation case (c)).
**Impact on plan:** All necessary to deliver the acceptance criteria against the real cintx ECP surface. The primary acceptance — `int1e_ecp` returns a finite, non-zero matrix on Cu/LANL2DZ — is met and always-on. No scope creep.

## Issues Encountered

- An unrelated, pre-existing rustfmt-only edit to `crates/pyscf-dft/src/df_dft.rs` (a collapsed match arm) was present in the working tree at start. Per the sequential-execution instructions (touch only plan-declared files), it was left unstaged and untouched.

## Verification Run

- `cargo build -p pyscf-gto` — clean (compiles the cintx-cubecl ECP kernels; scoped, no libxc_rs).
- `cargo test -p pyscf-gto --test ecp_int1e_oracle` — 2 passed (finite/non-zero + symmetric).
- `cargo test -p pyscf-gto` — all suites green (lib 36; ecp_engine_stub 5; ecp_load 6; intor_smoke 8; + every other suite; 2 pre-existing `#[ignore]` deferrals unrelated to ECP).
- `cargo test -p pyscf-core` — green (lib 11 + integration).
- `cargo clippy -p pyscf-gto -p pyscf-core --all-targets` — no warnings from changed code.
- `cargo run -p xtask --bin check-dependency-wall` — PASS (cubecl-* containment intact).
- `cargo run -p xtask --bin check-cubecl-pin` — PASS (cubecl 0.10.0 lockstep preserved across the ECP path).
- `grep -rn "Pending cintx ECP" crates/ tests/` — 0 matches.

## Known Stubs

`EcpEngineNotAvailable` (`ecp_engine_stub.rs`) intentionally remains in the codebase — demoted from the default engine to a documentation + testable error path. It is NOT a goal-blocking stub: `ecp_engine()` returns the real `CintxEcpEngine`, and `int1e_ecp` on an ECP-bearing molecule evaluates real values. The stub is only reached by direct instantiation in tests.

## Next Phase Readiness

- **Phase 7 GRAD-07 (ECP gradients) UNBLOCKED.** cintx exposes `OperatorId::INT1E_ECP_IPNUC_{CART,SPH}` (manifest ids 28/29, component_rank=3, byte-identical to nr_ecp_deriv per `safe_api_ecp_parity.rs`). The trait's `EcpEngine::ecp_int1e_ipnuc` default still returns `NotYetImplemented{phase:7}`; Phase 7 wires it on `CintxEcpEngine` the same way `ecp_int1e` is wired here (build the ECP-augmented BasisSet, dispatch the ipnuc operator, stitch a 3×nao×nao component-leading buffer).
- **Manual-Only follow-up:** run `pytest tests/oracle/test_ecp_int1e.py::test_cu_lanl2dz_int1e_ecp_byte_equal -v` once the oracle venv (numpy + vendored pyscf) is installed, to confirm upstream byte-identity at atol=1e-10.

## Self-Check: PASSED

- Created files verified on disk: `ecp_engine_cintx.rs`, `ecp_int1e_oracle.rs`, `tests/oracle/test_ecp_int1e.py`, `02-10-SUMMARY.md` — all FOUND.
- Commits verified in `git log`: `f547b88`, `5d6db6b`, `f4ca0c7`, `415788b`, `f9587c9`, `4589f73` — all FOUND.
- Only plan-declared files staged; `crates/pyscf-dft/src/df_dft.rs` (pre-existing rustfmt edit) and other unrelated untracked files left untouched.

---
*Phase: 02-gto*
*Completed: 2026-05-23*
