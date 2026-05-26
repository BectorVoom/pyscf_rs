---
phase: 07-gradients-geomopt
plan: 06
subsystem: geomopt
tags: [geomopt, checkpoint, hdf5, chkfile, shims, geometric-solver, berny-solver, constraints, resume, oracle_sum]

# Dependency graph
requires:
  - phase: 07-gradients-geomopt
    plan: 04
    provides: "the ONE native BFGS+RFO redundant-internal engine — optimize(opt, scanner, mol) -> OptimizeResult + GeometryOptimizer; the run loop this plan refactors over a shared run_loop and resumes/wraps"
  - phase: 03-scf
    provides: "pyscf-chkfile: the SOLE owner of hdf5-metno (D-05) — its re-exported `hdf5` alias + the scalar/1D primitives the optimizer-state checkpoint routes through (no own hdf5-metno dep)"
  - phase: 02-gto-integrals
    provides: "pyscf_gto::set_geom_ (GTO-10) — the cache-safe geometry mutation the shims use to materialise the optimized Mole + the resume loop uses to seed the working Mole"
provides:
  - "checkpoint.rs: OptimizerState dump/load to/from an HDF5 group via the pyscf_chkfile::hdf5 alias (GEOMOPT-05) — persists geometry, trust radius, BFGS Hessian (+counter), prev-step q/g_int/E history, step counter; schema_version + shape-validation fail-clean guard (T-07-19)"
  - "optimize_resume(opt, scanner, mol, state) -> OptimizeResult: resume a partially-converged run from a loaded checkpoint to the SAME stationary point as an uninterrupted run"
  - "OptimizeResult.state: the live OptimizerState the engine now returns on every run (converged or maxsteps) so the caller can checkpoint + resume"
  - "shims.rs: geometric_solver::{kernel,optimize} + berny_solver::{kernel,optimize} (GEOMOPT-02/03) — both thin aliases over the ONE native optimize engine (D-06); ShimParams mirrors the upstream kwargs (conv_params/callback/maxsteps/constraints, D-07); kernel->(conv,mol), optimize->mol"
  - "NATIVE_ENGINE_NAME single-engine marker + engine_name() on both shims — the structural single-engine invariant forbidding a second berny optimizer (T-07-20)"
  - "constraints kwarg -> clear GeomError::ConstraintsUnsupported, never a silent no-op (T-07-17); maxsteps validated at the shim boundary (T-07-18)"
affects: [07-09-pyo3-bridge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sole-owner HDF5 checkpoint (D-05/D-07): the optimizer-state spill routes through pyscf_chkfile::hdf5 + pyscf_chkfile::primitives — NO own hdf5-metno dep; the dump/load shape mirrors pyscf-scf/src/chkfile.rs (a &hdf5::Group consumer)"
    - "fail-clean checkpoint (T-07-19): a schema_version guard + shape-validation (hessian == nint·nint, coords == 3·natm, prev-trio all-present-or-all-absent, resumed nint == this molecule's internal-coordinate count) — a corrupt/incompatible checkpoint raises GeomError::CheckpointCorrupt rather than resuming from garbage"
    - "one-engine shim alias (D-06/T-07-20): a single shared run_shim core drives the ONE native optimize(); geometric_solver and berny_solver entries both call it; a NATIVE_ENGINE_NAME constant is the single-engine marker — there is NO second optimizer implementation"
    - "shared run_loop refactor: optimize() and optimize_resume() both call one run_loop(opt, scanner, mol, init: Option<OptimizerState>) — fresh vs resumed differ only in the initial (coords, hessian, trust, prev, step) seed; the geomeTRIC outer loop body is identical"

key-files:
  created:
    - crates/pyscf-geomopt/src/checkpoint.rs
    - crates/pyscf-geomopt/src/shims.rs
    - crates/pyscf-geomopt/tests/checkpoint_resume.rs
    - crates/pyscf-geomopt/tests/shim_parity.rs
  modified:
    - crates/pyscf-geomopt/src/lib.rs
    - crates/pyscf-geomopt/src/error.rs

key-decisions:
  - "OptimizerState schema persists exactly the run_loop's cross-iteration carry (coords, trust, hessian + n_updates, prev (E,q,g_int), step, e_tot) — the minimal set to byte-resume; encoded as f64 scalars + 1D f64 datasets (counts written as f64, read back via `as usize`) so the whole state rides the pyscf-chkfile scalar/1D primitives with no new dataset type"
  - "ShimParams::default()/new() default maxsteps=100 (the upstream default) — the DERIVED Default would give maxsteps=0 (DoS-rejected), so Default is hand-implemented to mirror new() (NOT #[derive(Default)])"
  - "the shim per-step `callback` contract is preserved (a Send+Sync ShimCallback in ShimParams) but invoked ONCE on completion with the final (nsteps, coords, e_tot) — the native engine does not yet expose a per-step hook; the 07-09 bridge re-routes to per-step when the engine grows one. ShimParams::clone drops the boxed callback (closures are not Clone) — tests supply fresh params per run"
  - "the shims are pyo3-free and take (scanner, mol, params); the Python method-dispatch (GradScanner/GradientsBase/nuc_grad_method) + the GEOMOPT-01 no-runtime-dep proof land in 07-09 (the bridge adapts a Python `method` to the native GradScanner and calls these functions)"

patterns-established:
  - "OptimizerState { coords, trust, hessian, nint, n_updates, step, prev_q, prev_g_int, prev_e, e_tot } + dump(&hdf5::Group)/load(&hdf5::Group): the HDF5 optimizer-state schema the 07-09 PyO3 bridge persists (group /opt_state; schema_version=1.0)"
  - "ShimParams { conv_params: Option<ConvParams>, callback: Option<ShimCallback>, maxsteps: usize, constraints: Option<String> } + geometric_solver/berny_solver::{kernel -> (conv, Mole), optimize -> Mole}: the shim entry signatures the 07-09 bridge wires the Python pyscf.geomopt.{geometric_solver,berny_solver}.optimize entry points against"

requirements-completed: [GEOMOPT-02, GEOMOPT-03, GEOMOPT-05]

# Metrics
duration: 7min
completed: 2026-05-26
---

# Phase 7 Plan 06: Geomopt API Surface — HDF5 Checkpoint/Resume + Shims Summary

**Rounded out the native geometry-optimizer API around the 07-04 engine: an HDF5 optimizer-state checkpoint (`checkpoint.rs`) that dumps/loads the live BFGS+RFO state (geometry, trust radius, BFGS Hessian, prev-step history, step counter) through the `pyscf_chkfile::hdf5` alias — NO own `hdf5-metno` dep (D-05/D-07 sole-owner discipline) — and resumes a partially-converged run to the same stationary point as an uninterrupted run (GEOMOPT-05); plus the `geometric_solver`/`berny_solver` shims (`shims.rs`) that BOTH delegate to the ONE native `optimize` engine via a single shared `run_shim` core (D-06 — `berny` is a thin alias, no second optimizer, T-07-20), mirror the upstream `kernel(...) -> (conv, mol)` / `optimize(...) -> mol` signatures + kwargs (D-07), and reject a `constraints` kwarg with a clear `ConstraintsUnsupported` error rather than a silent no-op (T-07-17). All persistence + shim tests are always-on (no SCF, no cintx grad integral); the real-SCF arm stays `#[ignore]`'d per the 07-04 precedent.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-05-26T04:13:30Z
- **Completed:** 2026-05-26T04:20:50Z
- **Tasks:** 2 (both `type="auto" tdd="true"`)
- **Files modified:** 6 (4 created, 2 modified)

## Accomplishments

- **`checkpoint.rs` (GEOMOPT-05)** — `OptimizerState { coords, trust, hessian, nint, n_updates, step, prev_q, prev_g_int, prev_e, e_tot }` with `dump(&hdf5::Group)` / `load(&hdf5::Group)`. Routes through the `pyscf_chkfile::hdf5` re-exported alias + the `pyscf_chkfile::primitives` scalar/1D helpers (the SOLE-owner discipline, D-05/D-07 — **no `hdf5-metno` dep in `pyscf-geomopt/Cargo.toml`**), mirroring `pyscf-scf/src/chkfile.rs`'s `&hdf5::Group` dump/load shape. Schema group `/opt_state` carries `schema_version` (= 1.0) + the scalar counts (`nint`/`natm`/`n_updates`/`step`) as `f64` + `trust`/`e_tot` + the flattened `coords`/`hessian`/`prev_q`/`prev_g_int` 1D datasets + a `has_prev` flag.
- **Fail-clean corrupt-checkpoint guard (T-07-19)** — `validate()` rejects a state whose shapes are mutually inconsistent (hessian ≠ `nint·nint`, prev-trio not all-present-or-all-absent, prev-vector length ≠ `nint`) on BOTH `dump` (before any write) and `load` (after reassembly); `load` also rejects an unknown `schema_version` and a `coords` length ≠ `3·natm`; `optimize_resume` rejects a resumed `nint` that disagrees with this molecule's internal-coordinate count. A corrupt/incompatible checkpoint raises a clear `GeomError::CheckpointCorrupt` rather than resuming from garbage.
- **Resume wiring** — refactored `optimize()` + new `optimize_resume()` over a single shared `run_loop(opt, scanner, mol, init: Option<OptimizerState>)`; `OptimizeResult` now carries `state: Option<OptimizerState>` (the live state at the stopping point, whether converged or `maxsteps`-exhausted) so a partial run can be checkpointed and resumed. The resume seeds `(coords, hessian, trust, prev, step)` from the loaded state and continues the identical geomeTRIC outer loop.
- **`shims.rs` (GEOMOPT-02/03, D-06/D-07)** — `geometric_solver::{kernel, optimize}` + `berny_solver::{kernel, optimize}`. A single shared `run_shim` core builds a `GeometryOptimizer` from the `ShimParams` kwargs and drives the ONE native `optimize()` engine; both `geometric_solver` and `berny_solver` call it (`berny` delegates straight to its `geometric_solver` twin — NO second optimizer, T-07-20). `kernel` returns `(conv, Mole)`; `optimize` returns the optimized `Mole` (`= kernel(...).1`), matching the upstream `pyscf/geomopt/geometric_solver.py:96-192` shapes exactly. `engine_name()` on both shims reports the single `NATIVE_ENGINE_NAME` marker.
- **Threat mitigations** — `constraints` (non-`None`) raises a clear `GeomError::ConstraintsUnsupported` citing the deferred GEOMOPT-EXT-01, never a silent no-op (T-07-17); `maxsteps` defaults to 100 and is validated at the shim boundary via the native `validate_maxsteps` (T-07-18 / T-07-10).
- **Gates green** — 45 tests pass / 1 ignored / 0 failed under `cargo test -p pyscf-geomopt --locked -- --test-threads=1` (the new `checkpoint_resume` = 3, `shim_parity` = 6); clippy clean (scoped, `--tests`); `check-dependency-wall`: PASS (T-07-SC — no new registry package, HDF5 via the chkfile alias, no `cubecl-*`).

## Task Commits

Each task was committed atomically (TDD: test → feat):

1. **Task 1 (RED): failing HDF5 checkpoint/resume test (GEOMOPT-05)** — `bcf71bb` (test)
2. **Task 1 (GREEN): HDF5 optimizer-state checkpoint + resume** — `7647903` (feat)
3. **Task 2 (RED): failing geometric/berny shim-parity test (GEOMOPT-02/03)** — `dc79454` (test)
4. **Task 2 (GREEN): geometric_solver + berny_solver shims over one engine** — `5d8605f` (feat)

**Plan metadata:** _(this commit)_ `docs(07-06): complete geomopt shims/checkpoint plan`

## Files Created/Modified

- `crates/pyscf-geomopt/src/checkpoint.rs` (created, ≈270 lines) — `OptimizerState` + `dump`/`load` via the `pyscf_chkfile::hdf5` alias; schema + fail-clean validation (T-07-19)
- `crates/pyscf-geomopt/src/shims.rs` (created, ≈250 lines) — `ShimParams` + `geometric_solver`/`berny_solver` `{kernel, optimize}` over one shared `run_shim` (D-06/D-07); `engine_name()` single-engine marker
- `crates/pyscf-geomopt/src/lib.rs` (modified) — `pub mod checkpoint`/`shims` + re-exports; `NATIVE_ENGINE_NAME`; refactored `optimize()` + new `optimize_resume()` over a shared `run_loop`; `OptimizeResult.state`
- `crates/pyscf-geomopt/src/error.rs` (modified) — `GeomError::Chkfile` (`#[from] pyscf_chkfile::ChkfileError`) + `GeomError::CheckpointCorrupt { what }`
- `crates/pyscf-geomopt/tests/checkpoint_resume.rs` (created) — byte-exact dump→load round-trip + resume-to-same-stationary-point + corrupt-fails-cleanly (3 tests)
- `crates/pyscf-geomopt/tests/shim_parity.rs` (created) — kernel→(conv,mol) + optimize→mol shapes, berny==geometric single-engine structural assertion, constraints clear-error, maxsteps DoS cap, default maxsteps=100 (6 tests)

## Decisions Made

- **Checkpoint schema = the `run_loop` cross-iteration carry, encoded as f64 scalars + 1D datasets.** Persist exactly `(coords, trust, hessian + n_updates, prev (E,q,g_int), step, e_tot)` — the minimal set to byte-resume. Integer counts are written as `f64` and read back via `as usize` so the whole state rides the existing `pyscf-chkfile` scalar/1D primitives with no new dataset type and no new dep.
- **`ShimParams::default()` hand-implemented (NOT `#[derive(Default)]`).** The derived default would give `maxsteps=0` (which the DoS guard rejects); `Default`/`new()` set `maxsteps=100` (the upstream default).
- **The shim `callback` is invoked once on completion (not per-step) for now.** The native engine has no per-step hook yet; the `ShimCallback` contract is preserved in `ShimParams` and the 07-09 bridge re-routes it to per-step once the engine exposes one. `ShimParams::clone` drops the boxed callback (closures aren't `Clone`); tests build fresh params per run.
- **Shims stay pyo3-free, taking `(scanner, mol, params)`.** The Python `method` dispatch (`GradScanner`/`GradientsBase`/`nuc_grad_method`) + the GEOMOPT-01 no-runtime-dep proof are explicitly 07-09 (the bridge adapts a Python `method` to the native `GradScanner` and calls these entry points).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed the `coords` bit-comparison in the RED checkpoint test**
- **Found during:** Task 1 (GREEN compile of `checkpoint_resume.rs`)
- **Issue:** The round-trip test iterated `coords` (a `Vec<[f64; 3]>`) and called `.to_bits()` on each `&[f64; 3]` element — `to_bits` exists on `f64`, not on `[f64; 3]`, so the test failed to compile.
- **Fix:** Compare each of the 3 components' bits in an inner `for k in 0..3` loop.
- **Files modified:** `crates/pyscf-geomopt/tests/checkpoint_resume.rs`
- **Verification:** `cargo test -p pyscf-geomopt --test checkpoint_resume` compiles + all 3 tests pass.
- **Committed in:** `7647903` (Task 1 GREEN commit)

**2. [Rule 3 - Blocking] Removed a duplicate `Default` for `ShimParams`**
- **Found during:** Task 2 (GREEN compile of `shims.rs`)
- **Issue:** `ShimParams` had both `#[derive(Default)]` AND a hand-written `impl Default` (needed so `default()` mirrors `new()` with `maxsteps=100`) — a conflicting-implementations error (E0119) blocked the build.
- **Fix:** Dropped `#[derive(Default)]`, keeping the hand-written `impl Default for ShimParams { fn default() -> Self { Self::new() } }`.
- **Files modified:** `crates/pyscf-geomopt/src/shims.rs`
- **Verification:** `cargo test -p pyscf-geomopt --test shim_parity` compiles; `shim_default_maxsteps_is_100` asserts the correct default.
- **Committed in:** `5d8605f` (Task 2 GREEN commit)

**3. [Rule 1 - Lint] Collapsed two `if let … { if … }` blocks in `checkpoint::validate`**
- **Found during:** Task 1 (clippy `--tests`)
- **Issue:** Clippy `collapsible_if` on the two `prev_q`/`prev_g_int` length checks.
- **Fix:** Collapsed each to an `if let Some(x) = … && x.len() != nint { … }` let-chain.
- **Files modified:** `crates/pyscf-geomopt/src/checkpoint.rs`
- **Verification:** `cargo clippy -p pyscf-geomopt --locked --tests` clean.
- **Committed in:** `7647903` (Task 1 GREEN commit)

---

**Total deviations:** 3 auto-fixed (2 Rule-1 test-compile/lint, 1 Rule-3 blocking build error). **Impact on plan:** none on scope — both checkpoint + shims land exactly as specified. All three were mechanical compile/lint fixes in the new code introduced by this plan (no source-logic change beyond what the plan called for).

## Issues Encountered

- The pre-existing workspace-wide `[patch] cintx not used in the crate graph` note + the `fma4 unknown target-feature` warning appear on every scoped build (recorded in 07-04 SUMMARY); they are independent of this plan and do not affect the geomopt gates.

## Known Stubs

None that block the plan's goal. Two intentional, documented deferrals:
- **The Python entry points + the GEOMOPT-01 no-runtime-dep proof are NOT in this plan** — they are explicitly 07-09 (the PyO3 bridge adapts a Python `method` to the native `GradScanner` and calls `geometric_solver`/`berny_solver`). This plan ships the Rust-side shim surface only (the plan's own NOTE).
- **The shim `callback` fires once on completion, not per-step** — the native engine has no per-step hook yet; the contract is preserved for 07-09 (decision above).
- **The `constraints` clear-error is the intended T-07-17 behaviour, not a stub** — full constraint support is GEOMOPT-EXT-01 (deferred); the shim is required to reject, never silently ignore.

## Threat Flags

None — no new network endpoints, auth paths, or schema changes at a trust boundary. The plan's threat register is mitigated: `constraints` → clear error (T-07-17), `maxsteps` capped at the shim boundary (T-07-18), a corrupt/incompatible HDF5 checkpoint fails cleanly on load/resume (T-07-19), `berny` is a thin alias over the one engine with a structural single-engine test (T-07-20), and no new registry package is added — HDF5 routes through the `pyscf-chkfile` alias (T-07-SC).

## TDD Gate Compliance

Both tasks followed the RED → GREEN cycle with the gate commits present in git history:
- Task 1: `test(07-06)` RED `bcf71bb` → `feat(07-06)` GREEN `7647903`.
- Task 2: `test(07-06)` RED `dc79454` → `feat(07-06)` GREEN `5d8605f`.
No REFACTOR commit was needed (the GREEN code was already clean; the three deviations were folded into the GREEN commits as compile/lint fixes to the new code).

## Next Phase Readiness

- **07-09 (PyO3 bridge):** wire `pyscf.geomopt.optimize(mf)` + `geometric_solver.optimize` / `berny_solver.optimize` against the shim entry points recorded above. The bridge adapts a Python `method` (`GradScanner`/`GradientsBase`/`nuc_grad_method`) to the native `GradScanner`, builds `ShimParams` from the Python kwargs, and calls `geometric_solver::optimize` (or `berny_solver::optimize`). The HDF5 checkpoint (`OptimizerState::dump`/`load`, group `/opt_state`, schema_version 1.0) is the persistence seam for `pyscf.geomopt` checkpoint/restart. The GEOMOPT-01 no-runtime-dep proof (`pip uninstall geometric pyberny && python -c "import pyscf.geomopt; pyscf.geomopt.optimize(mf)"`) is a 07-09 CI gate (D-05).
- **Coordination note (D-02 hinge):** the real-SCF geomopt arm (`equilibrium_via_rhf_gradient` in 07-04) stays `#[ignore]`'d on the six missing cintx grad-integral families; this plan's checkpoint + shim tests are entirely always-on (model scanner) and need no cintx.

## Self-Check: PASSED

- All 4 created files (`checkpoint.rs`, `shims.rs`, `checkpoint_resume.rs`, `shim_parity.rs`) + 2 modified (`lib.rs`, `error.rs`) exist on disk (verified below).
- All 4 task commits (`bcf71bb`, `7647903`, `dc79454`, `5d8605f`) present in git history.
- `cargo test -p pyscf-geomopt --locked -- --test-threads=1`: 45 passed, 1 ignored, 0 failed.
- `cargo clippy -p pyscf-geomopt --locked --tests`: clean. `check-dependency-wall`: PASS. `pyscf-geomopt/Cargo.toml`: no `hdf5-metno`/`hdf5_metno` dependency line (routes through `pyscf-chkfile`).

---
*Phase: 07-gradients-geomopt*
*Completed: 2026-05-26*
