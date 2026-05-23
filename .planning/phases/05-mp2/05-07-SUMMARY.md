---
phase: 05-mp2
plan: 07
subsystem: api
tags: [pyo3, mp2, rmp2, ump2, dfmp2, bindings, scanner, numpy, override-dispatch]

# Dependency graph
requires:
  - phase: 05-mp2 (05-03 RMP2)
    provides: rmp2_kernel, Mp2Reference, Mp2Result, default_ao2mo, scs_energy, Mp2OverrideHooks, NoMp2Overrides, ChemistsEris
  - phase: 05-mp2 (05-04 UMP2 + RDMs)
    provides: ump2_kernel, UmpReference, make_rdm1/make_rdm2 (free fns)
  - phase: 05-mp2 (05-05 conventional DF-MP2)
    provides: dfrmp2_kernel, dfump2_kernel, df_ao2mo, DFRMP2/DFUMP2
  - phase: 05-mp2 (05-06 native RI-MP2)
    provides: NativeDFRMP2/NativeDFUMP2, emp2_rhf/emp2_uhf
  - phase: 03-scf-pyo3-bindings
    provides: PyRHF/PyUHF template, PyOverrideBridge call_method1 pattern, numpy_io converters, errors bridge, _native pymodule registration
provides:
  - PyRMP2/PyUMP2/PyDFMP2 PyO3 classes (the MP2 PyO3 wall, D-07/D-08)
  - Mp2PyBridge — pyo3-side Mp2OverrideHooks impl (call_method1 override dispatch + py.detach default path)
  - MP2() module-level factory (RHF->RMP2 / UHF->UMP2 / with_df->DFMP2)
  - PyMp2Scanner — Mole->energy as_scanner callable (MP2-07)
  - python/pyscf/mp/__init__.py overlay re-exporting _native.mp
  - mp submodule registered in _native
affects: [06-ccsd, 07-grad, 08-gpu-distribution, milestone-uat]

# Tech tracking
tech-stack:
  added: []  # no new external crates — pyscf-mp2 (already shipped) added as a path-dep only
  patterns:
    - "Eager SCF-reference snapshot (D-07): PyMP2::new pulls mf.mo_coeff/mo_energy/mo_occ/e_tot/mol into a plain-array Mp2Reference + holds Py<PyAny> mf"
    - "Override dispatch via slf.call_method1 (D-08 / Pitfall 7) with a __qualname__ base-class check (is_overridden) + py.detach default-hook compute (BIND-05)"
    - "MP2 scanner: mf.as_scanner() held, __call__(mol) re-runs reference -> re-snapshot -> MP2 kernel -> e_hf + scs_energy"
    - "Module-level #[pyfunction] factory registered in the submodule (MP2 dispatch on istype('UHF')/with_df)"

key-files:
  created:
    - crates/pyscf-py/src/mp.rs
    - python/pyscf/mp/__init__.py
    - crates/pyscf-py/tests/mp2_scanner.rs
  modified:
    - crates/pyscf-py/src/lib.rs
    - crates/pyscf-py/Cargo.toml
    - Cargo.lock

key-decisions:
  - "Override detection via __qualname__ base-class compare (is_overridden) rather than always dispatching: the DEFAULT (non-overridden) path runs the pure-Rust default_ao2mo/df_ao2mo under py.detach (BIND-05); a subclass override goes through call_method1 (Pitfall 7)."
  - "PyDFMP2::kernel routes through the same Mp2PyBridge with a DefaultAo2mo::Df(DfIntegrals) source (reusing rmp2_kernel + df_ao2mo — the 'swap the ERI source' contract) so a DFMP2 subclass ao2mo override is honored, not the standalone dfrmp2_kernel."
  - "DF B-tensor uses default_ri (mp2fit *-ri) aux (A2), NOT the JK-fit aux."
  - "GHF/GMP2 omitted from the factory (out of v1 MP2 scope per PROJECT.md — RMP2/UMP2/DFMP2 only)."

patterns-established:
  - "Pattern: a pyo3-free method bridge (Mp2PyBridge) impl of the crate's OverrideHooks trait, holding Py<PyAny> slf, with an enum-tagged DEFAULT compute source (InCore vs Df) — extends the SCF/KS PyOverrideBridge precedent to a method with a swappable integral source."
  - "Pattern: as_scanner over a held mf.as_scanner() — the MP2 scanner re-uses the SCF scanner's geometry re-run, then re-snapshots, keeping the MP2 method crate pyo3-free (D-07)."

requirements-completed: [MP2-01, MP2-02, MP2-04, MP2-05, MP2-06, MP2-07]

# Metrics
duration: 8min
completed: 2026-05-23
---

# Phase 5 Plan 07: PyO3 Bridge for MP2 Summary

**PyRMP2/PyUMP2/PyDFMP2 PyO3 classes that eager-snapshot the SCF reference into plain Rust arrays and call the pyo3-free pyscf-mp2 kernels, with slf.call_method1 override dispatch, the MP2() factory, an as_scanner Mole->energy callable, and the python/pyscf/mp overlay.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-23T08:43:52Z
- **Completed:** 2026-05-23T08:52:39Z
- **Tasks:** 2
- **Files modified:** 6 (3 created, 3 modified)

## Accomplishments

- `crates/pyscf-py/src/mp.rs` — the complete MP2 PyO3 surface (the PyO3 wall): `PyRMP2`/`PyUMP2`/`PyDFMP2` each eager-snapshot the Python `mf` into a plain-array `Mp2Reference` (D-07) and call the pyo3-free `rmp2_kernel`/`ump2_kernel`/`dfrmp2_kernel`.
- `Mp2PyBridge` impls `pyscf_mp2::Mp2OverrideHooks`: a subclass `ao2mo`/`make_rdm1`/`make_rdm2` override dispatches through `slf.bind(py).call_method1` (Pitfall 7 MRO resolution, via an `is_overridden` `__qualname__` base-class check); the DEFAULT path runs the pure-Rust `default_ao2mo`/`df_ao2mo`/`NoMp2Overrides` compute under `py.detach` (BIND-05). The `kernel` itself does NOT `py.detach` (hooks re-enter Python).
- `MP2()` module-level `#[pyfunction]` factory: `istype('UHF')`→UMP2, `with_df`→DFMP2, else RMP2 (MP2-01); `frozen=None/int/list/'auto'` (MP2-03).
- `PyMp2Scanner` (MP2-07): holds `mf.as_scanner()`; `__call__(mol)` re-runs the reference SCF at the new geometry, re-snapshots into a fresh `Mp2Reference`, runs the MP2 kernel, and returns `e_hf + scs_energy(...)`.
- SCS factors (MP2-06) via `emp2_ss_factor`/`emp2_os_factor` getters/setters + `set(...)` + `emp2_scs` getter; `make_rdm1`/`make_rdm2` route through the bridge (MP2-05).
- `python/pyscf/mp/__init__.py` overlay re-exports `MP2`/`RMP2`/`UMP2`/`DFMP2` from `pyscf._native.mp` (BIND-02), with `__all__` + factory-dispatch docstring.
- `crates/pyscf-py/tests/mp2_scanner.rs` — always-on structural test (4 arms): plumbing greps + a Rust-side scanner-closure-shape assertion (synthetic `Mp2Reference` + hand-supplied `(ia|jb)` block → `rmp2_kernel` → `-1.125`, no live `mf`, no `int2e` gate).

## Task Commits

Each task was committed atomically:

1. **Task 1: PyRMP2/PyUMP2/PyDFMP2 + Mp2OverrideHooks bridge + factory + as_scanner** - `f5da909` (feat)
2. **Task 2: python/pyscf/mp overlay + as_scanner structural test** - `a3d95be` (test)

**Plan metadata:** (final docs commit — this SUMMARY + STATE + ROADMAP + REQUIREMENTS)

## Files Created/Modified

- `crates/pyscf-py/src/mp.rs` (created) - PyRMP2/PyUMP2/PyDFMP2 + Mp2PyBridge + PyMp2Scanner + MP2 factory; the MP2 PyO3 wall
- `crates/pyscf-py/src/lib.rs` (modified) - `pub mod mp;` + `mp` submodule registration in `_native`
- `crates/pyscf-py/Cargo.toml` (modified) - `pyscf-mp2` path-dep (default features, no libxc)
- `Cargo.lock` (modified) - the single `pyscf-mp2` dependency line under `pyscf-py`
- `python/pyscf/mp/__init__.py` (created) - `_native.mp` re-export overlay (BIND-02)
- `crates/pyscf-py/tests/mp2_scanner.rs` (created) - always-on MP2-07 structural test

## Decisions Made

- **Override detection (`is_overridden`):** rather than blindly dispatching every hook through `call_method1`, the bridge compares the resolved bound method's `__qualname__` class component against the known base-class names (`RMP2`/`UMP2`/`DFMP2`). Only a genuine subclass override goes through Python (call_method1); the un-overridden default runs the pure-Rust compute under `py.detach`. This keeps the common (no-subclass) path GIL-released and avoids a needless Python round-trip per kernel call, while preserving Pitfall-7 MRO fidelity for subclasses.
- **PyDFMP2 routes through the bridge with `DefaultAo2mo::Df`:** instead of calling the standalone `dfrmp2_kernel` directly (which hard-wires the DF source), PyDFMP2::kernel builds the DF B-tensor then runs the reused `rmp2_kernel` with the bridge whose default `ao2mo` is `df_ao2mo` over that B-tensor — the upstream "DFRMP2 subclasses RMP2 and only swaps `ao2mo`" contract. A `DFMP2` subclass `ao2mo` override is thus honored. (The standalone `dfrmp2_kernel` is still used by the scanner's DensityFitted path.)
- **GHF/GMP2 omitted from the factory:** PROJECT.md scopes v1 MP2 to RMP2/UMP2/DFMP2; the upstream GHF→GMP2 branch is out of scope, so `MP2()` dispatches only the three in-scope variants.

## Deviations from Plan

None - plan executed exactly as written. The plan's "optionally a native variant under a `dfmp2_native` namespace" was left as the documented optional (the native `NativeDFRMP2`/`emp2_rhf` energy fast-path ships pyo3-free in 05-06; exposing it as a separate `pyscf.mp.dfmp2_native` PyO3 class was not required for the MP2-01/02/04/05/06/07 deliverable and is deferred — the conventional DFMP2 is the default factory target, matching upstream where `dfmp2_native.DFRMP2` subclasses `lib.StreamObject`, not the `mp.DFMP2` factory).

## Issues Encountered

- The synthetic scanner-closure test's hand-computed expected energy was initially wrong (`-1.0625`); the 1×1 RMP2 reference (per 05-03's SUMMARY) is `e_corr = -0.125`, so `e_tot = e_hf + e_corr = -1.0 + -0.125 = -1.125`. Corrected the comment + assertion; test passes. (Test-only arithmetic; no source change.)
- Clippy `-D warnings` flagged `collapsible_if` in `mf_is_uhf`/`parse_frozen` and a `never-constructed DefaultAo2mo::Df` variant. Fixed by flattening the nested ifs into PyResult-returning closures + `is_ok_and`, and by wiring PyDFMP2::kernel to actually use `DefaultAo2mo::Df` (the bridge-routing decision above). All resolved within Task 1 before commit.

## Manual-Only / CI Verifications (deferred to CI per 05-VALIDATION)

The live cross-module dispatch parity needs libpython + upstream pyscf + the cintx#11 arity-4 `int2e` / arity-3 `int3c2e_sph` gates (the env lacks maturin/pyscf — Phase-4 precedent). These run in the `mp2-oracle-cintx-gated` CI job, NOT in the executor sandbox:

- (a) `mf.MP2().run().e_corr == mp.RMP2(mf).kernel()[0]` — cross-module dispatch parity (live Python + fully-wired PyO3).
- (b) `mf.density_fit().MP2()` routes to DFMP2 (live + cintx#11 gate).

The Rust-side structural test (`cargo test -p pyscf-py --test mp2_scanner`, 4 arms) + the PyO3 build (`cargo build -p pyscf-py --locked`) are always-on. Numeric MP2 energies remain cintx#11-gated; the structural/contract/registration layers are the deliverable here.

## Next Phase Readiness

- The MP2 PyO3 surface is complete and building; `from pyscf import mp` resolves via the overlay once the cdylib is built by maturin.
- CCSD (Phase 6) imports the five MP2-08 helpers (`get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`mo_without_core`) from pyscf-mp2 — those are pyo3-free and already shipped; no PyO3-side dependency on this plan.
- Numeric MP2 energies (and the live dispatch-parity CI arms) flip on once cintx#11 lands arity-4 `int2e` + arity-3 `int3c2e_sph` — no code change needed in this PyO3 layer.

## Self-Check: PASSED

- Created files verified present: `crates/pyscf-py/src/mp.rs`, `python/pyscf/mp/__init__.py`, `crates/pyscf-py/tests/mp2_scanner.rs`, `.planning/phases/05-mp2/05-07-SUMMARY.md`.
- Task commits verified in git log: `f5da909` (Task 1), `a3d95be` (Task 2).

---
*Phase: 05-mp2*
*Completed: 2026-05-23*
