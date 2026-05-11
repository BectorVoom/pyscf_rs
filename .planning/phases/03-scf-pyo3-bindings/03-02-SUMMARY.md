---
phase: 03-scf-pyo3-bindings
plan: 02
subsystem: infra
tags: [maturin, pyo3, abi3, free-threading, pytest, hdf5, lint, build-rs, oracle]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: pyscf-py crate stub (cdylib+rlib), pyscf-oracle pyo3 dev-dep, xtask lint binary pattern, .gitignore base, workspace [patch.crates-io]
  - phase: 02-gto
    provides: pyscf-core::Mole (30-attribute floor) + pyscf-gto::M factory referenced by SCF test fixtures (bodies fill in plan 03-10)
provides:
  - pyproject.toml at repo root with maturin python-source overlay (BIND-01/02)
  - python/pyscf/ overlay shim (__init__.py + scf/__init__.py) re-exporting from pyscf._native
  - 19 pytest stub files under python/pyscf/tests/ each citing the REQ-ID it covers and pinned to plan 03-10 for body fill
  - conftest.py with three skip-stub fixtures (h2o_mol, benzene_mol, water_trimer_mol)
  - oracle_check! macro stub (ORACLE-02) — accepts the documented arg shape, panics with a "plan 03-08 pending" message at runtime
  - chkfile_roundtrip integration test (#[ignore]'d) exercising the macro shape at compile time (ORACLE-08 anchor)
  - crates/pyscf-py/build.rs feature-mutex guard rejecting abi3-py310 + free-threading at compile time (BIND-05; checker iter 1 WARNING 4 fix)
  - crates/pyscf-py/Cargo.toml feature scaffolding (default=abi3-py310, free-threading) — full PyO3 dep wiring lives in plan 03-07
  - xtask check-forbid-lazy-static binary (BIND-06) preventing lazy_static! under crates/pyscf-py/
  - .gitignore extended to exclude maturin's python/pyscf/_native* + .so/.dylib/.pyd artifacts
affects: [03-03, 03-04, 03-05, 03-06, 03-07, 03-08, 03-09, 03-10]

# Tech tracking
tech-stack:
  added:
    - maturin>=1.4,<2.0 (build backend wired in pyproject.toml)
    - pytest>=7.0 (declared as project optional-dependency; not installed locally)
    - h5py>=3.10 (declared; consumed by plan 03-08 chkfile oracle)
  patterns:
    - "Maturin python-source overlay (python/pyscf/) re-exports cdylib `pyscf._native` — supersedes upstream pyscf/ tree at install time per BIND-02"
    - "Cargo feature mutex via build.rs panic — Cargo features are not natively mutually exclusive; this is the standard workaround for the abi3 vs free-threaded ABI conflict (WARNING 4)"
    - "xfail-stubbed pytest scaffolding — every Wave-0 test file ships an xfail-marked function citing the REQ-ID it covers, so plan 03-10 un-xfails one at a time as real bodies land; plan-collection still discovers the stubs cleanly"
    - "ORACLE-02 macro shape locked at scaffold time: $method:literal, $fixture:expr, $tolerance:expr — body deferred to plan 03-08 but call sites compile today, with $tolerance bound to f64 so wrong types fail at the macro site"

key-files:
  created:
    - python/pyscf/__init__.py
    - python/pyscf/scf/__init__.py
    - python/pyscf/tests/__init__.py
    - python/pyscf/tests/conftest.py
    - python/pyscf/tests/test_scf_smoke.py (+ 18 sibling test_*.py stubs)
    - crates/pyscf-oracle/tests/chkfile_roundtrip.rs
    - crates/pyscf-py/build.rs
    - xtask/src/bin/check_forbid_lazy_static.rs
  modified:
    - pyproject.toml (REPLACED upstream PySCF setuptools/cmake config with maturin overlay shape)
    - .gitignore (extended with python/pyscf/_native* + *.so/*.dylib/*.pyd)
    - crates/pyscf-oracle/src/lib.rs (Phase 1 stub → ORACLE-02 macro stub)
    - crates/pyscf-py/Cargo.toml (Phase 1 stub → +build.rs +abi3-py310/free-threading features)
    - xtask/Cargo.toml (added [[bin]] check-forbid-lazy-static)

key-decisions:
  - "Replace upstream PySCF pyproject.toml rather than coexist — BIND-02 overlay supersedes the upstream tree at install time; preserving the upstream toml on disk would break the maturin install path"
  - "Plan-narrative said 19 test stub files; VALIDATION.md table listed 18 explicitly. Added test_scf_xplat_uhartree.py (Pitfall 12 / SCF-13 matrix-CI Python-side companion) so the success criterion verify (`ls test_*.py | wc -l == 19`) passes"
  - "Ship build.rs feature-mutex guard BEFORE plan 03-07 wires PyO3 deps — the guard is in place from the first build, so any accidental --features abi3-py310,free-threading CI run fails loudly instead of producing a buggy artifact"
  - "Bind the oracle_check! macro $tolerance arg to `let _tolerance: f64 = $tolerance;` so a wrong type fails at the macro site today (plan 03-08 inherits the type contract for free)"
  - "xtask check-forbid-lazy-static walks up to repo root via Cargo.toml/[workspace] sniff — works from any CWD inside the worktree; PASSes today because pyscf-py has no lazy_static! yet"

patterns-established:
  - "Pattern: Phase-3 Wave-0 scaffolding ships file structure WITHOUT production logic so subsequent waves' tests can assert green against existing files; plan 03-10 un-xfails one stub at a time as real bodies land"
  - "Pattern: build.rs feature-mutex guard for mutually exclusive Cargo features (Phase 4-7 method crates can adopt the same pattern for their respective abi3/free-threading splits)"
  - "Pattern: ORACLE-02 oracle_check!($method:literal, $fixture:expr, $tolerance:expr) — a macro shape locked at Wave 0; body fills in plan 03-08; subsequent method-phase oracles reuse the macro with method-specific suffixes"

requirements-completed: [BIND-02, BIND-05, BIND-06]

# Metrics
duration: 7min
completed: 2026-05-11
---

# Phase 03 Plan 02: Wave-0 Scaffolds Summary

**Maturin overlay packaging + 19 pytest stubs + oracle_check! macro + abi3/free-threading feature mutex guard + BIND-06 lint — Wave-0 file structure landed so plans 03-03..03-08 can write production code against existing scaffolds.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-11T12:15:29Z
- **Completed:** 2026-05-11T12:22:31Z
- **Tasks:** 3
- **Files modified:** 28 (5 modified, 23 created)

## Accomplishments

- **BIND-01/BIND-02 packaging shape locked.** pyproject.toml at repo root configures maturin with `python-source = "python"`, `module-name = "pyscf._native"`, `manifest-path = "crates/pyscf-py/Cargo.toml"`. Upstream PySCF setuptools/cmake config replaced; upstream pyscf/ tree preserved on disk per Phase 1 D-03 but no longer the active install target.
- **19 pytest stub files + 3 fixture skip-stubs** under `python/pyscf/tests/`. Each test_*.py file has one `def test_*` decorated with `@pytest.mark.xfail(reason="… plan 03-10", strict=False)`; every body is `assert False, "Phase 3 plan 03-10 must implement …"` so xfail is REPORTED rather than hidden. REQ-ID coverage: SCF-01..14, BIND-02/04/09, ORACLE-08 (via SCF-10).
- **ORACLE-02 macro stub** in `crates/pyscf-oracle/src/lib.rs` accepts the documented `($method:literal, $fixture:expr, $tolerance:expr)` shape and binds `$tolerance` to `f64` at macro-expansion time, so call sites compile today and wrong types fail at the macro site. Runtime panics with a clear "plan 03-08 pending" message.
- **chkfile_roundtrip integration test** (`crates/pyscf-oracle/tests/chkfile_roundtrip.rs`) `#[ignore]`'d pending plan 03-08, but exercises the macro shape at compile time so a syntax break is caught immediately.
- **Checker iteration 1 WARNING 4 closed.** `crates/pyscf-py/build.rs` panics at compile time if both `abi3-py310` and `free-threading` features are enabled. Verified: `cargo check -p pyscf-py --features abi3-py310,free-threading` fails with `pyscf-py: features 'abi3-py310' and 'free-threading' are mutually exclusive`. Default (abi3-py310) and `--no-default-features --features free-threading` both build cleanly.
- **BIND-06 lint** (`xtask check-forbid-lazy-static`) wired and PASSing today. When plan 03-07 lands the PyO3 caches, the lint blocks any `lazy_static!` introduction; PR-time enforcement of `pyo3::sync::PyOnceLock` discipline.
- **`.gitignore` extended** with the maturin cdylib artifact paths under `python/pyscf/`.

## Task Commits

Each task was committed atomically with `--no-verify` (parallel-executor convention; orchestrator validates hooks once after wave completion):

1. **Task 1: pyproject.toml + python/pyscf overlay (BIND-01/02)** — `6d924f7` (feat)
2. **Task 2: 19 pytest stub files + conftest.py** — `035040c` (test)
3. **Task 3: oracle_check! macro stub + chkfile_roundtrip test + build.rs feature-mutex + xtask check-forbid-lazy-static** — `8939534` (feat)

## Files Created/Modified

**Created (23):**
- `python/pyscf/__init__.py` — overlay shim re-exporting `scf` from `pyscf._native` with ImportError tolerance
- `python/pyscf/scf/__init__.py` — re-exports RHF/UHF/GHF/density_fit from `pyscf._native.scf` with import-fallback stubs
- `python/pyscf/tests/__init__.py` — empty pytest discovery marker
- `python/pyscf/tests/conftest.py` — h2o_mol/benzene_mol/water_trimer_mol fixtures (skip-stubs)
- `python/pyscf/tests/test_scf_smoke.py` — SCF-01 wave-level smoke aggregator
- `python/pyscf/tests/test_scf_rhf_h2o.py` — SCF-01 RHF on H2O/cc-pVDZ
- `python/pyscf/tests/test_scf_rhf_benzene.py` — SCF-01 RHF on benzene/6-31G*
- `python/pyscf/tests/test_scf_uhf.py` — SCF-02 UHF open-shell
- `python/pyscf/tests/test_scf_ghf.py` — SCF-03 GHF on H2
- `python/pyscf/tests/test_scf_diis.py` — SCF-04 C-DIIS iteration count
- `python/pyscf/tests/test_scf_init_guess.py` — SCF-05 5 init_guess modes
- `python/pyscf/tests/test_scf_controls.py` — SCF-06 level_shift/damp/max_cycle/conv_tol/conv_tol_grad
- `python/pyscf/tests/test_scf_df.py` — SCF-07 density-fit RHF
- `python/pyscf/tests/test_scf_override_dispatch.py` — SCF-08/BIND-07 subclass override
- `python/pyscf/tests/test_scf_analyze.py` — SCF-09 analyze/mulliken_pop/mulliken_meta/dip_moment
- `python/pyscf/tests/test_scf_chkfile.py` — SCF-10/ORACLE-08 chkfile round-trip
- `python/pyscf/tests/test_scf_cross_dispatch.py` — SCF-11 to_uhf/to_rhf/to_ghf + to_uks/to_rks NotYetImplemented stubs
- `python/pyscf/tests/test_scf_scanner.py` — SCF-12 mf.as_scanner()
- `python/pyscf/tests/test_scf_xplat_uhartree.py` — SCF-13 Linux x86_64 + macOS aarch64 µHartree consistency (Pitfall 12)
- `python/pyscf/tests/test_scf_attributes.py` — SCF-14 ≥30-attribute floor introspection
- `python/pyscf/tests/test_overlay_resolution.py` — BIND-02 overlay resolution
- `python/pyscf/tests/test_scf_stride_fuzz.py` — BIND-04 stride-fuzz
- `python/pyscf/tests/test_panic_to_exception.py` — BIND-09 panic→exception
- `crates/pyscf-oracle/tests/chkfile_roundtrip.rs` — ORACLE-08 macro invocation (#[ignore])
- `crates/pyscf-py/build.rs` — abi3-py310 + free-threading mutex guard (WARNING 4)
- `xtask/src/bin/check_forbid_lazy_static.rs` — BIND-06 lint binary

**Modified (5):**
- `pyproject.toml` — REPLACED upstream PySCF setuptools/cmake config with maturin overlay
- `.gitignore` — added Phase 3 BIND-01 cdylib artifact exclusions
- `crates/pyscf-oracle/src/lib.rs` — Phase 1 empty stub → oracle_check! macro stub
- `crates/pyscf-py/Cargo.toml` — added `build = "build.rs"` and abi3-py310/free-threading feature pair
- `xtask/Cargo.toml` — added `[[bin]] check-forbid-lazy-static`

## Decisions Made

1. **Replace upstream pyproject.toml** rather than coexist. The upstream PySCF `[build-system] requires = ["setuptools", "cmake"]` config conflicts with maturin's build flow; both pointing at "name = pyscf-rs" / "name = pyscf_rs" can't coexist in the same file. Per the plan (Task 1 step 1 NOTE), BIND-02's overlay supersedes the upstream tree at install time, so the file is REPLACED. The upstream Python tree under `pyscf/` remains untouched on disk per Phase 1 D-03.

2. **Add a 19th stub file (`test_scf_xplat_uhartree.py`)** — VALIDATION.md §"Wave 0 Requirements" lines 84-104 listed 18 unique `test_*.py` files; the plan's must_haves narrative and success criterion both said "19 pytest stub files". Rather than treat this as a pure plan-narrative typo, I shipped the cross-platform µHartree stub as the 19th file (SCF-13's Pitfall 12 Python-side companion to the matrix-CI job), so the success criterion's `ls test_*.py | wc -l == 19` passes cleanly. Documented as a plan-spec reconciliation, not scope creep — SCF-13 was already in the test-coverage map row 14.

3. **Bind `$tolerance` to `f64` at macro expansion** so wrong types (e.g., `oracle_check!("foo", "fixture", 1)` with an integer literal) fail at the macro site today rather than at plan-03-08-fill time. Plan-08 inherits this type contract for free.

4. **`xtask check-forbid-lazy-static` walks up to repo root via Cargo.toml + `[workspace]` sniff** rather than hard-coding CWD. Works from any subdirectory inside the worktree. Mirrors the pattern in `xtask/src/bin/check_dependency_wall.rs` (Phase 1).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking / Plan-Spec Reconciliation] 19th stub file added to satisfy success criterion**
- **Found during:** Task 2 verification (`ls python/pyscf/tests/test_*.py | wc -l` returned 18; plan narrative + verify regex required 19)
- **Issue:** The plan's must_haves text said "19 pytest stub files" and the verify regex was `^(19|20)$`, but VALIDATION.md §"Wave 0 Requirements" lines 84-104 listed only 18 unique `test_*.py` files. Implementing the table verbatim would fail the verify command.
- **Fix:** Added `python/pyscf/tests/test_scf_xplat_uhartree.py` covering SCF-13's "Linux x86_64 + macOS aarch64 µHartree assertion" row (already in the per-requirement map as the 14th row — just not enumerated separately in the file list).
- **Files modified:** `python/pyscf/tests/test_scf_xplat_uhartree.py` (created)
- **Verification:** `ls python/pyscf/tests/test_*.py | wc -l` now returns 19; xfail marker present; reason cites plan 03-10.
- **Committed in:** 035040c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 plan-spec reconciliation)
**Impact on plan:** No scope creep — the added stub covers a REQ-ID (SCF-13 cross-platform µHartree) that VALIDATION.md already had in its per-requirement coverage map but had not enumerated in the file list. Plan 03-10 will fill the body.

## Issues Encountered

- **`cargo test -p pyscf-oracle` cannot run in this dev environment.** The Phase 1 ORACLE-01 dev-dep is `pyo3 = "=0.28.3"` with the `auto-initialize` feature, which requires a Python install with shared-library support. The local Python on this worktree only ships static-embed support, so pyo3's build script aborts at `cargo test` time. The macro itself compiles cleanly via `cargo check -p pyscf-oracle`, and a standalone `rustc` smoke compile of an isolated copy of the macro confirmed the shape is sound. CI runners with proper python3.10+ shared-lib installs will run the test once plan 03-08 fills the body. This is a pre-existing Phase 1 environment requirement, not introduced by this plan.

## Verification Log

| Check | Result |
|-------|--------|
| `test -f pyproject.toml && grep 'python-source = "python"'` | PASS |
| `ls python/pyscf/tests/test_*.py \| wc -l` | 19 |
| `grep -lF "xfail" python/pyscf/tests/test_*.py \| wc -l` | 19 |
| `cargo run -p xtask --bin check-forbid-lazy-static` | exit 0, "PASS — no `lazy_static!` in pyscf-py (BIND-06)" |
| `cargo check -p pyscf-oracle --locked` | PASS |
| `cargo check -p pyscf-py --locked` (default abi3-py310) | PASS |
| `cargo check -p pyscf-py --no-default-features --features free-threading --locked` | PASS |
| `cargo check -p pyscf-py --features abi3-py310,free-threading --locked` | FAIL with "features `abi3-py310` and `free-threading` are mutually exclusive" (expected) |
| `grep '_native\*' .gitignore` | PASS |
| Root `Cargo.toml` `libxc_rs` line still commented out | PASS (line 94 unchanged) |

## Next Phase Readiness

**Wave 1 (plans 03-03..03-08) can now proceed.** Specifically:
- Plan 03-07 has `pyproject.toml` + `python/pyscf/__init__.py` overlay + `crates/pyscf-py/Cargo.toml` build.rs scaffolding — needs only to fill `[dependencies]` with pyo3+numpy+pyscf-scf+pyscf-runtime and write `src/lib.rs` `#[pymodule] _native(...)` body.
- Plan 03-08 has the `oracle_check!` macro stub + chkfile_roundtrip test scaffold — needs only to replace the panic body with the real Python::attach + upstream-driver + numpy-allclose comparison.
- Plan 03-10 has 19 xfail stubs — needs to un-xfail one at a time as test bodies land.
- BIND-06 lint will block any plan 03-07 PR that introduces `lazy_static!` anywhere under `crates/pyscf-py/` — forces `pyo3::sync::PyOnceLock` discipline at PR time.
- WARNING 4 mutex guard means any subsequent plan cannot accidentally enable both abi3-py310 and free-threading via a `--features` flag in CI.

**Pending downstream prerequisites (NOT blockers for this plan):**
- Phase 1 ORACLE-01 environment requirement (Python shared-lib install) needs to be on every CI runner that will execute `cargo test -p pyscf-oracle`. Local dev workaround for plans 03-03..03-06 (which don't touch pyscf-oracle): build with `cargo check -p pyscf-oracle` to validate the lib compiles; defer test runs to CI.

## Self-Check: PASSED

Verified files exist:
- pyproject.toml: FOUND
- python/pyscf/__init__.py: FOUND
- python/pyscf/scf/__init__.py: FOUND
- python/pyscf/tests/{__init__.py, conftest.py, 19 test_*.py}: ALL 21 FOUND
- crates/pyscf-oracle/src/lib.rs: FOUND (modified — macro stub)
- crates/pyscf-oracle/tests/chkfile_roundtrip.rs: FOUND
- crates/pyscf-py/build.rs: FOUND
- crates/pyscf-py/Cargo.toml: FOUND (modified — build.rs + features)
- xtask/src/bin/check_forbid_lazy_static.rs: FOUND
- xtask/Cargo.toml: FOUND (modified — [[bin]])
- .gitignore: FOUND (modified — _native* exclusions)

Verified commits exist:
- 6d924f7 (Task 1): FOUND on worktree-agent-a6a88041143a31cca
- 035040c (Task 2): FOUND on worktree-agent-a6a88041143a31cca
- 8939534 (Task 3): FOUND on worktree-agent-a6a88041143a31cca

---
*Phase: 03-scf-pyo3-bindings*
*Plan: 02 — Wave-0 Scaffolds*
*Completed: 2026-05-11*
