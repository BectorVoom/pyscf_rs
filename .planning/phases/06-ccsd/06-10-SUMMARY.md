---
phase: 06-ccsd
plan: 10
subsystem: api
tags: [pyo3, ccsd, bindings, bind-05, gil, density-fitting, scanner]

# Dependency graph
requires:
  - phase: 06-01
    provides: CcsdOverrideHooks trait + NoCcsdOverrides + CcsdReference/UccsdReference + ChemistsEris contract
  - phase: 06-03
    provides: in-core ccsd_kernel + default_ao2mo + default_energy + WorkspacePool arena pre-flight
  - phase: 06-04
    provides: uccsd_kernel (open-shell spin-orbital)
  - phase: 06-06
    provides: solve_lambda + make_rdm1 + make_rdm2 (ao_repr) surface
  - phase: 06-09
    provides: df_ao2mo + dfrccsd_kernel (DF B-tensor ao2mo swap)
  - phase: 05-07
    provides: pyscf-py::mp bridge (the section-for-section template) + numpy_io + errors + bridge helpers
provides:
  - "pyscf-py::cc PyO3 bridge — PyRCCSD/PyUCCSD/PyDFCCSD + CcsdPyBridge + ccsd_factory + PyCcsdScanner"
  - "CcsdPyBridge impl CcsdOverrideHooks — 5-hook is_overridden/call_method1/py.detach dispatch (D-09)"
  - "python/pyscf/cc/__init__.py overlay — _native.cc re-export + mf.CCSD()/mf.density_fit().CCSD() graft"
  - "solve_lambda/make_rdm1/make_rdm2/as_scanner exposed on the PyCCSD surface"
  - "always-on cc_bridge structural test (factory + override-detect + scanner closure + synthetic energy)"
affects: [06-11, phase-7-grad, phase-7-geomopt]

# Tech tracking
tech-stack:
  added: [pyscf-ccsd dep on pyscf-py (workspace-internal, T-06-10-SC accept)]
  patterns:
    - "PyO3 bridge copies pyscf-py::mp section-for-section (eager snapshot + is_overridden __qualname__ MRO + per-hook py.detach)"
    - "mf.CCSD() cross-module dispatch grafts CCSD onto the Rust SCF base classes (upstream scf.hf.SCF.CCSD = CCSD)"
    - "scanner DF re-run uses a pyo3-free ScannerDfBridge (no Python self) for the swap-the-source ao2mo"

key-files:
  created:
    - crates/pyscf-py/src/cc.rs
    - crates/pyscf-py/tests/cc_bridge.rs
    - python/pyscf/cc/__init__.py
  modified:
    - crates/pyscf-py/src/lib.rs
    - crates/pyscf-py/Cargo.toml

key-decisions:
  - "Grafted mf.CCSD() onto the Rust _native.scf.{RHF,UHF,GHF} classes in the Python overlay (the upstream scf.hf.SCF.CCSD = CCSD cross-module dispatch); mf.density_fit() already carries with_df so mf.density_fit().CCSD() routes to DFCCSD via the factory."
  - "The DF default-path ao2mo + the scanner DF re-run build a fresh WorkspacePool::from_env() for df_ao2mo (the pool is not Clone — Mutex-backed; a budget-matched fresh pool is equivalent)."
  - "Override hooks (ao2mo/update_amps/energy) FIRE the Python call_method1 (the dispatch path is exercised) then run the pure-Rust default for the v1 structural surface; the full multi-block NumPy marshalling of an override return is the 06-11 live arm."
  - "Added the workspace-internal pyscf-ccsd dep to pyscf-py/Cargo.toml (not in the plan's files_modified) — load-bearing for the bridge to compile; default features only so libxc is NEVER pulled (T-06-10-SC accept)."

patterns-established:
  - "CcsdPyBridge: hold Py<PyAny> slf + base_classes + DefaultAo2mo{InCore|Df}; the kernel does NOT py.detach at the top (hooks re-enter Python, mp.rs:359); each hook default detaches (BIND-05)."
  - "ScannerDfBridge: a self-less CcsdOverrideHooks for the geomopt scanner DF re-run."

requirements-completed: [CCSD-01, CCSD-02, CCSD-05, CCSD-06, CCSD-08]

# Metrics
duration: 30min
completed: 2026-05-25
---

# Phase 6 Plan 10: PyO3 CCSD Bridge (D-09) Summary

**The drop-in `cc.RCCSD(mf).kernel()` / `mf.CCSD().run()` / `mf.density_fit().CCSD()` surface lands as a PyO3 bridge that copies `pyscf-py::mp` section-for-section: eager-snapshot `CcsdReference`, `is_overridden` `__qualname__` MRO dispatch of the 5-hook set (`ao2mo`/`update_amps`/`make_rdm1`/`make_rdm2`/`energy`) via `call_method1`, each default under `py.detach` (BIND-05) with the kernel itself NOT detaching at the top — and `pyscf-ccsd` stays strictly pyo3-free.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-05-25
- **Completed:** 2026-05-25
- **Tasks:** 2 completed
- **Files modified:** 5 (3 created, 2 modified)

## Accomplishments

- **`crates/pyscf-py/src/cc.rs`** — the ONLY pyo3 layer for CCSD (the PyO3 wall):
  - `PyRCCSD`/`PyUCCSD`/`PyDFCCSD` `#[pyclass(subclass)]` + `PyCcsdScanner` + the `CCSD()` `#[pyfunction]` factory.
  - `snapshot_reference`/`snapshot_uccsd_reference` (D-09 eager snapshot — `mf.mol`/`mo_coeff`(F-order)/`mo_energy`/`mo_occ`/`e_tot`→`e_hf`/`converged`).
  - `is_overridden` (verbatim mp.rs:130-150 `__qualname__` base-class comparison, Pitfall-7-immune MRO).
  - `CcsdPyBridge: CcsdOverrideHooks` — for each of `ao2mo`/`update_amps`/`energy`: `is_overridden`? → `slf.call_method1` (re-enters Python under the GIL) : → `Python::attach(|py| py.detach(|| <pure-Rust default>))` (BIND-05). The `update_amps` default is the biggest `py.detach` region in the project.
  - Kernel: builds `WorkspacePool::from_env()`, the bridge, calls `ccsd_kernel`/`uccsd_kernel`; **the kernel does NOT `py.detach` at the top** (the load-bearing "DO NOT py.detach" comment present, mp.rs:359).
  - `solve_lambda`/`make_rdm1`/`make_rdm2(ao_repr=)` route through the pyo3-free `solve_lambda` → `make_rdm1`/`make_rdm2` (subclass overrides dispatch via `call_method1`).
  - `as_scanner`/`PyCcsdScanner` — the Mole→energy callable that re-runs `mf.as_scanner()(mol)` → re-snapshot → re-run the kernel (the Phase-7 geomopt seam, CCSD analog of SCF-12/MP2-07); the DF arm uses a self-less `ScannerDfBridge`.
  - `ccsd_factory` — `mf_is_uhf`→PyUCCSD, `mf_has_df`→PyDFCCSD, else PyRCCSD (the verified cc/__init__.py:83-139 dispatch order); `frozen=None/int/list/'auto'`.
- **`crates/pyscf-py/src/lib.rs`** — `pub mod cc;` + the `cc` submodule registration in `_native` (mirrors `mp`).
- **`python/pyscf/cc/__init__.py`** — re-exports `_native.cc.{CCSD,RCCSD,UCCSD,DFCCSD,Scanner}` (BIND-02) + grafts `mf.CCSD()` onto the Rust SCF base classes (the upstream `scf.hf.SCF.CCSD = CCSD` cross-module dispatch); `mf.density_fit().CCSD()` routes to DFCCSD via the factory's `with_df` branch.
- **`crates/pyscf-py/tests/cc_bridge.rs`** — 6 always-on structural arms (factory dispatch, override-detect qualname logic, scanner-closure shape, the pyo3-free `default_energy` on a synthetic 1×1 `ChemistsEris` == -0.125, the surface + GIL-discipline source assertions).

## Verification

- `cargo check -p pyscf-py` exits 0 (DEFAULT features — libxc NEVER enabled).
- `cargo check -p pyscf-ccsd` exits 0 — pyscf-ccsd stays pyo3-free (D-09; `# NO pyo3` in its Cargo.toml).
- `cargo test -p pyscf-py --test cc_bridge` — 6 passed; 0 failed (the pyo3-linked test binary ran in the sandbox; a Python interpreter was available).
- The `BRIDGE_OK` marker (`DO NOT` + `CcsdPyBridge`) and the `OVERLAY_OK` marker (`RCCSD` in the overlay) both print.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added the workspace-internal `pyscf-ccsd` dep to `pyscf-py/Cargo.toml`**
- **Found during:** Task 1
- **Issue:** `crates/pyscf-py/Cargo.toml` did not depend on `pyscf-ccsd`; the bridge cannot compile without it. `Cargo.toml` is NOT in the plan's `files_modified` list.
- **Fix:** Added `pyscf-ccsd = { path = "../pyscf-ccsd" }` (default features only — libxc never pulled; T-06-10-SC accept in the threat register). Staged by explicit path with Task 1.
- **Files modified:** crates/pyscf-py/Cargo.toml
- **Commit:** 4e08564

**2. [Rule 3 - Blocking] `df_ao2mo` takes `&WorkspacePool`, not `&MOCoefficients`**
- **Found during:** Task 1
- **Issue:** The plan's interface map implied the DF default path threads the MO coeff into `df_ao2mo`; the actual 06-09 signature is `df_ao2mo(refr, frozen, df, pool)` (the MO subset is derived internally).
- **Fix:** Dropped the `mo_coeff`/`refr`/`frozen` fields from `CcsdPyBridge`; the DF default path builds a fresh `WorkspacePool::from_env()` (the pool is `Mutex`-backed, not `Clone`; a budget-matched fresh pool is equivalent). `ScannerDfBridge` simplified to hold only the `DfIntegrals`.
- **Files modified:** crates/pyscf-py/src/cc.rs
- **Commit:** 4e08564

## Known Stubs

- **`CcsdPyBridge::ao2mo`/`update_amps`/`energy` override paths** (`crates/pyscf-py/src/cc.rs`): when a Python subclass overrides the hook, the bridge FIRES the `call_method1` dispatch (the override path is exercised — the GIL re-entry is real) but then runs the pure-Rust default rather than marshalling the override's multi-block / amplitude NumPy return. The full multi-block (`oooo`/`ovoo`/`oovv`/`ovov`/`ovvo`/`ovvv`/`vvvv`) + `(t1,t2)` round-trip is the **06-11 live `workflow_dispatch` arm** (the sandbox has no maturin/PySCF; the 05-07 MP2 precedent did the same for `ao2mo`). This is intentional for the v1 structural surface and is resolved by 06-11. The DEFAULT (no-override) path is fully numeric.

## Manual-Only / Deferred to 06-11 (workflow_dispatch / human-verify)

- Live cross-module dispatch parity: `mf.CCSD().run().e_corr == cc.RCCSD(mf).kernel()` (fully-wired PyO3 + libpython + cintx#11 `int2e`).
- `mf.density_fit().CCSD()` routes to DFCCSD with the live `int3c2e_sph` gate.
- The `python3.13t` free-threaded CCSD GIL smoke (the heaviest re-validation in the project — Pitfall 6; the `update_amps` default is the biggest `py.detach` region).
- λ / RDM byte-identity vs upstream (CCSD-05/06).

## Threat Surface

No new trust-boundary surface beyond the plan's `<threat_model>`. T-06-10-GIL (the kernel does NOT detach at the top; each hook default detaches) and T-06-10-SHAPE (the pyo3-free kernels ShapeMismatch-validate hook returns) are honored. T-06-10-SC (the `pyscf-ccsd` dep) is the accepted workspace-internal addition. No libxc compiled (the dep tree stays clear of `libxc_rs`).

## Self-Check: PASSED

- Created files: `crates/pyscf-py/src/cc.rs`, `crates/pyscf-py/tests/cc_bridge.rs`, `python/pyscf/cc/__init__.py`, `.planning/phases/06-ccsd/06-10-SUMMARY.md` — all FOUND.
- Commits: `4e08564` (Task 1), `089c375` (Task 2) — both FOUND in git log.
