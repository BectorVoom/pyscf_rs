---
phase: 04-dft
plan: 09
subsystem: api
tags: [pyo3, dft, subclass-override, call_method1, precision, d-08, bindings]

# Dependency graph
requires:
  - phase: 04-dft
    provides: "04-06 RKS/UKS + KsOverrideHooks + NumInt::dtype() (the Rust DFT driver + read-only precision accessor the bridge/getter surface)"
  - phase: 03-scf
    provides: "PyOverrideBridge + call_hook + PyRHF/PyUHF pyclass pattern + to_uks/to_rks stubs (the exact analogs extended here)"
provides:
  - "PyRKS/PyUKS subclass pyclasses + _native.dft submodule + python/pyscf/dft overlay (`from pyscf import dft` surface)"
  - "PyOverrideBridge extended to impl KsOverrideHooks — get_veff (DFT-form) + define_xc_ (string form) dispatched via slf.call_method1 (Pitfall 7 re-validation on the DFT hook surface)"
  - "to_uks/to_rks wired to real KS targets (no longer NotYetImplemented{phase:4})"
  - "read-only D-08 precision #[getter] on the KS object (returns DType::name() 'f32'/'f64'); NO precision setter (deferred per D-08)"
affects: [04-10, 05-mp2, 06-ccsd, 07-grad]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "DFT PyO3 boundary mirrors the Phase 3 SCF boundary exactly (pyclass(subclass) + register + call_hook); pyscf-dft stays pyo3-free (PyO3 wall)"
    - "Read-only #[getter] without a paired #[setter] is the D-08 precision-accessor model (PYSCF_DTYPE is the single switch)"

key-files:
  created:
    - crates/pyscf-py/src/dft.rs
    - python/pyscf/dft/__init__.py
  modified:
    - crates/pyscf-py/src/bridge.rs
    - crates/pyscf-py/src/lib.rs
    - crates/pyscf-py/src/scf.rs
    - crates/pyscf-scf/src/convert.rs
    - crates/pyscf-scf/src/lib.rs
    - python/pyscf/__init__.py
    - python/tests/test_dft_override.py

key-decisions:
  - "User checkpoint approval (2026-05-22): accept the source-verified PyO3 surface; the live `from pyscf import dft` run + override pytest are CI/manual items (no maturin/numpy/pyscf in this environment) — consistent with DFT-08 being Manual-Only / PyO3-dispatch-assertion in 04-VALIDATION.md."
  - "define_xc_ string form dispatches via call_method1; the callable form stays NotYetImplemented{deferred} (D-02)."
  - "Precision surface is a read-only getter only — no set_precision/#[setter] (D-08)."

patterns-established:
  - "Every KS override hook dispatched via slf.call_method1 (MRO), never Rust dispatch — Pitfall 7 re-validated on the largest override surface."

requirements-completed: [DFT-08]
# Note: 04-09 frontmatter also lists DFT-11, but 04-09 only delivers the D-08 read-only precision getter
# (the Python-visible half of the precision surface). DFT-11's WGPU shader-f64 honest-fallback is owned by
# 04-10; DFT-11 stays [~] until 04-10 lands. The live DFT-08 pytest/script is CI/manual (Manual-Only).

# Metrics
duration: 11min
completed: 2026-05-22
---

# Phase 04: dft — Plan 09 Summary

**PyO3 DFT boundary: `PyRKS`/`PyUKS` + `_native.dft` submodule + overlay, `KsOverrideHooks` bridge dispatching `get_veff`/`define_xc_` via `call_method1` (Pitfall 7 re-validated), `to_uks`/`to_rks` wired, and a read-only D-08 precision getter (no setter).**

## Performance

- **Duration:** ~11 min (2 autonomous tasks) + human-verify checkpoint (approved)
- **Completed:** 2026-05-22
- **Tasks:** 2/2 autonomous + Task 3 human-verify (approved; live verification deferred to CI/manual)
- **Files modified:** 9 (2 created)

## Accomplishments
- `crates/pyscf-py/src/dft.rs`: `PyRKS`/`PyUKS` `#[pyclass(subclass, module="pyscf._native.dft")]` with the DFT attribute floor, `kernel`/`run`, and a read-only `dtype` `#[getter]` (returns `DType::name()` as a str; no `#[setter]`).
- `bridge.rs`: `PyOverrideBridge` extended to `impl KsOverrideHooks` — `get_veff` (DFT-form, reusing the get_veff template) + `define_xc_` (string form via `call_method1`); every hook via `call_hook`/`call_method1` (Pitfall 7), under the per-hook `Python::attach`/`detach` seam. Callable `define_xc_` stays `NotYetImplemented{deferred}` (D-02).
- `lib.rs`: `_native.dft` submodule registered (mirrors scf).
- `convert.rs`: `to_uks`/`to_rks` wired to real RHF→RKS / UHF→UKS targets (no longer `NotYetImplemented{phase:4}`).
- `python/pyscf/dft/__init__.py` + `python/pyscf/__init__.py`: overlay re-exporting RKS/UKS from `pyscf._native.dft`.
- `python/tests/test_dft_override.py`: DFT-08 override-invoked-every-cycle (`get_veff` + `define_xc_` counters) + D-08 read-only-getter (`"f64"` default, assignment raises `AttributeError`) assertions. Currently `importorskip`s here (no extension/numpy).

## Task Commits
1. **Task 1: PyRKS/PyUKS + bridge + getter + overlay + to_uks/to_rks** — `fbe1e35` (feat)
2. **Task 2: pytest override-every-cycle (DFT-08) + read-only dtype getter (D-08)** — `a7237b4` (test)

**Plan metadata:** this SUMMARY + tracking finalized by the orchestrator post-approval.

## Decisions Made
- Human-verify checkpoint **approved** by user (2026-05-22): accept source-verified PyO3 surface; live pytest/script + maturin build are CI/manual items (no pyscf/numpy/maturin in this environment). Matches DFT-08 = Manual-Only in 04-VALIDATION.md.

## Deviations from Plan
None functionally — the live pytest could not be executed here (environment lacks maturin/numpy/pyscf), so it is written-and-deferred rather than run. This is the established 04-04/04-05/04-06 convention (real implementation + CI-only live verification).

## Issues Encountered
- maturin/numpy/pyscf unavailable in this environment → the Task 2 live pytest and the Task 3 `from pyscf import dft` script are deferred to CI/the user's environment. The Rust side is fully verified (`cargo build -p pyscf-py` 0 errors, `check-forbidden-paths` PASS, `to_rks`/`to_uks` convert test passes, all source assertions confirmed).

## Verification (local, Rust side)
- `cargo build -p pyscf-py` (default) — 0 errors
- `cargo run -p xtask --bin check-forbidden-paths` — PASS (PyO3 wall intact; `pyscf-dft` names no pyo3)
- `cargo test -p pyscf-scf --lib convert` — `to_rks`/`to_uks` wiring passes
- Source assertions: `PyRKS`/`PyUKS` subclass pyclasses; `_native.dft` registered; KS bridge dispatches via `call_method1`; read-only `dtype` getter, no setter; `define_xc_` callable-form deferred.

## Deferred to CI/manual (DFT-08 Manual-Only)
1. `python -c "from pyscf import gto, dft; ... dft.RKS(m, xc='pbe').run().e_tot"` prints an energy (also needs the Phase-2 ERI closure for a real B3LYP/PBE number).
2. `cd python && python -m pytest tests/test_dft_override.py` green.
3. Subclass `get_veff` override changes the result; `mf.dtype` reads "f64" and rejects assignment.

## Next Phase Readiness
- The PyO3 DFT surface is in place for downstream phases. The live DFT-08 override re-validation runs in CI/the user's env once a maturin build + upstream pyscf are available; a real DFT energy additionally needs the Phase-2 `int2e_sph`/`int3c2e_sph` ERI rollup.

---
*Phase: 04-dft*
*Completed: 2026-05-22 (human-verify approved; live pytest CI/manual-deferred)*
