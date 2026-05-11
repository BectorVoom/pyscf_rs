---
phase: 03-scf-pyo3-bindings
plan: 03
subsystem: scf
tags: [rust, scf, rhf, uhf, ghf, hartree-fock, trait-callback-bridge, pyo3-isolation]

requires:
  - phase: 01-foundation
    provides: pyscf-core types (Mole, Density, MOCoefficients, Energy, PyscfRsError), pyscf-algebra (AlgebraError), pyscf-runtime
  - phase: 02-gto
    provides: pyscf-gto (basis set + intor surface)
  - phase: 03-01
    provides: pyscf-core::canonicalize_signs (consumed by plan 03-11's eig body), pyscf-algebra::solve_linear (consumed by plan 03-04 DIIS)
  - phase: 03-02
    provides: Wave-0 SCF scaffolding stubs in pyscf-py, pyproject.toml maturin config

provides:
  - pyscf-scf crate populated with 16 modules (hooks, kernel, kernel_impl, init_guess, fock, eig, occ, rdm, energy, analyze, convert, scanner, rhf, uhf, ghf, error)
  - OverrideHooks trait (SCF-08 contract) with 11 surface methods covering all 10 logical override points
  - NoOverrides zero-cost impl (D-02) delegating to module-level default_* free fns
  - KernelConfig with upstream PySCF defaults (pyscf/scf/hf.py:1689-1759)
  - ScfResult, InitGuessMode (6 variants), parse_init_guess_mode for oracle Arm 4
  - RHF / UHF / GHF structs with the 32-field 30-attribute floor (SCF-14) and matching upstream defaults
  - generic kernel<H: OverrideHooks>(mol, hooks, cfg) signature delegating to kernel_impl::scf_loop (body in plan 03-11)
  - ScfError variant family (ConvergenceFailure, InitGuessNotYetImplemented, Algebra/#[from], Core/#[from], PythonOverrideFailed) bridged into PyscfRsError

affects:
  - 03-04 (DIIS — implements pyscf-diis::DiisAdapter against the get_fock signature declared here)
  - 03-05 (DF-HF — populates the with_df slot on RHF/UHF/GHF)
  - 03-06 (chkfile init_guess — handles InitGuessMode::Chkfile path)
  - 03-07 (PyO3 bridge — implements PyOverrideBridge against OverrideHooks)
  - 03-08 (oracle harness — consumes parse_init_guess_mode for Arm 4)
  - 03-10 (Python introspection — 30-attribute floor verification end-to-end)
  - 03-11 (kernel internals — fills all 20 unimplemented!() stubs)

tech-stack:
  added:
    - thiserror (workspace) — ScfError variant family
    - tracing (workspace) — reserved for kernel cycle logging in plan 03-11
  patterns:
    - "D-01 trait-callback bridge: pyo3 is FORBIDDEN in pyscf-scf (algebra-wall analog). PyOverrideBridge in pyscf-py implements OverrideHooks."
    - "D-02 zero-cost overrides: NoOverrides delegates to free fns so monomorphisation inlines the no-override path."
    - "30-attribute floor via Rust struct field count: missing fields fail at compile time when tests reference them (no runtime introspection needed in Rust; Python introspection lives in plan 03-10)."
    - "Unimplemented stub seam: trait impls cover every override but bodies are unimplemented!('plan 03-11') so the trait/struct surface compiles end-to-end TODAY and plan 03-11 replaces 20 stub bodies without re-touching any signature."

key-files:
  created:
    - crates/pyscf-scf/src/hooks.rs (OverrideHooks trait + NoOverrides)
    - crates/pyscf-scf/src/kernel.rs (KernelConfig, ScfResult, InitGuessMode, generic kernel<H>)
    - crates/pyscf-scf/src/kernel_impl.rs (scf_loop stub — plan 03-11 fills)
    - crates/pyscf-scf/src/init_guess.rs (5-mode dispatch + parse_init_guess_mode + init_guess_by_1e stub)
    - crates/pyscf-scf/src/fock.rs (5 default_* fock stubs — plan 03-11 fills)
    - crates/pyscf-scf/src/eig.rs (default_eig stub — plan 03-11 calls eigh+canonicalize_signs)
    - crates/pyscf-scf/src/occ.rs (default_get_occ stub)
    - crates/pyscf-scf/src/rdm.rs (default_make_rdm1 stub)
    - crates/pyscf-scf/src/energy.rs (default_energy_elec, default_energy_tot stubs)
    - crates/pyscf-scf/src/analyze.rs (analyze/mulliken_*/dip_moment stubs + MullikenResult)
    - crates/pyscf-scf/src/convert.rs (to_rhf/to_uhf/to_ghf stubs; to_rks/to_uks return NotYetImplemented{phase:4})
    - crates/pyscf-scf/src/scanner.rs (as_scanner stub)
    - crates/pyscf-scf/src/rhf.rs (RHF struct — 32 fields, manual Debug, RHF::new + RHF::kernel)
    - crates/pyscf-scf/src/uhf.rs (UHF struct — 32 fields with alpha/beta pair on MO slots)
    - crates/pyscf-scf/src/ghf.rs (GHF struct — 32 fields on 2c spinor basis)
    - crates/pyscf-scf/src/error.rs (ScfError + From<ScfError> for PyscfRsError)
    - crates/pyscf-scf/tests/hooks_kernel_types.rs (4 tests covering trait wiring + KernelConfig defaults + ScfResult derives)
    - crates/pyscf-scf/tests/attribute_floor.rs (4 tests covering 30-attribute floor + parse_init_guess_mode)
  modified:
    - crates/pyscf-scf/Cargo.toml (added pyscf-core/algebra/gto/runtime + thiserror/tracing; documented NO-pyo3 dep)
    - crates/pyscf-scf/src/lib.rs (16-module pub mod tree + re-exports)
    - Cargo.lock (pyscf-scf gained 6 deps)

key-decisions:
  - "Trait exposes 11 surface methods (not 10): energy_elec and energy_tot are sibling override points in upstream PySCF and both must be overrideable separately for fidelity with the Python class."
  - "RHF::kernel returns ScfResult by value (NOT &ScfResult via Box::leak as the plan body suggested) to avoid per-call memory leak. Converged scalar state is mirrored into the RHF struct fields before returning."
  - "ScfError → PyscfRsError bridge routes through Core(InvalidMolecule(String)) instead of the non-existent Core(Other(String)) that the plan body referenced. The pyscf-core::CoreError enum has 3 variants (InvalidMolecule, BasisParse, DimensionMismatch); InvalidMolecule is the only String-carrying catch-all today."
  - "Manual Debug impl on RHF/UHF/GHF (not derive) because Box<dyn Any> and Box<dyn Fn> are not Debug. The impl prints scalar fields verbatim and elides the opaque slots."
  - "rhf/uhf/ghf.rs ship minimal placeholder structs in Task 1 GREEN so analyze/convert/scanner (which reference crate::RHF/UHF/GHF) compile during Task 1; Task 2 GREEN replaces them with the full 32-field floor."

patterns-established:
  - "Pattern: pyo3 dep-wall pyscf-scf: Cargo.toml comment + no pyo3 anywhere in src/. Validated by grep -i pyo3 finding only the comment line."
  - "Pattern: unimplemented!('plan NN-MM') seam: every stub body names the future plan that will fill it. greppable for handoff."
  - "Pattern: 30-attribute floor in struct definition: the Rust compile-time check (test file references rhf.diis_space etc.) provides a stronger floor guarantee than runtime introspection."

requirements-completed: [SCF-01, SCF-02, SCF-03, SCF-05, SCF-06, SCF-14]

duration: 8min
completed: 2026-05-11
---

# Phase 03 Plan 03: pyscf-scf trait + RHF/UHF/GHF scaffolding Summary

**SCF trait-callback scaffolding (OverrideHooks, NoOverrides, generic kernel<H>) plus RHF/UHF/GHF structs with the 32-field 30-attribute floor — all compiles end-to-end with 20 unimplemented!('plan 03-11') stubs ready for Wave-3 body fills.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-11T12:39:44Z
- **Completed:** 2026-05-11T12:47:16Z
- **Tasks:** 2 (TDD)
- **Files modified/created:** 19 source files + 2 test files + Cargo.toml + Cargo.lock = 23

## Accomplishments

- pyscf-scf moved from Phase-1 empty stub (`#![forbid(unsafe_code)]` only) to a 16-module crate with full trait + struct scaffolding compiling end-to-end against pyscf-core + pyscf-algebra + pyscf-gto + pyscf-runtime.
- `OverrideHooks` trait declared with 11 surface methods (SCF-08 contract; plan 03-07 will implement `PyOverrideBridge` against it).
- 32-field 30-attribute floor (SCF-14) verbatim from upstream `pyscf/scf/hf.py:1716-1724 _keys` shipped on `RHF`/`UHF`/`GHF` with matching upstream defaults (diis_space=8, diis_start_cycle=1, max_cycle=50, conv_tol=1e-9, direct_scf_tol=1e-13, init_guess="minao", etc.).
- 5 init_guess modes declared (minao/atom/1e/huckel/chkfile + UserDM); "1e" routes to `init_guess_by_1e` stub (plan 03-11); minao/atom/huckel return `ScfError::InitGuessNotYetImplemented`; chkfile routes to plan 03-06.
- 8 tests covering trait wiring, default values, struct construction, and parse_init_guess_mode round-trip — all pass.
- pyo3 is documented FORBIDDEN here (D-01 trait-callback bridge — pyo3 lives only in pyscf-py).

## Task Commits

Each TDD task produced RED→GREEN commit pair:

1. **Task 1 RED: failing tests for OverrideHooks/KernelConfig/ScfResult** — `ffabc26` (test)
2. **Task 1 GREEN: scaffold pyscf-scf trait + kernel types** — `e001289` (feat)
3. **Task 2 RED: failing tests for RHF/UHF/GHF 30-attribute floor** — `72970fd` (test)
4. **Task 2 GREEN: RHF/UHF/GHF structs + 30-attribute floor** — `075cd9f` (feat)

_TDD plan: 4 atomic commits (2 RED + 2 GREEN), no REFACTOR needed._

## Files Created/Modified

### Created (19 files)
- `crates/pyscf-scf/src/hooks.rs` — `OverrideHooks` trait (11 methods) + `NoOverrides` impl
- `crates/pyscf-scf/src/kernel.rs` — `KernelConfig` (12 fields, upstream defaults), `ScfResult`, `InitGuessMode` (6 variants), generic `kernel<H: OverrideHooks>`
- `crates/pyscf-scf/src/kernel_impl.rs` — `scf_loop` stub (plan 03-11)
- `crates/pyscf-scf/src/init_guess.rs` — 5-mode dispatch + `parse_init_guess_mode` + `init_guess_by_1e` stub
- `crates/pyscf-scf/src/fock.rs` — 5 `default_*` stubs for hcore/ovlp/jk/veff/fock
- `crates/pyscf-scf/src/eig.rs` — `default_eig` stub (plan 03-11 wires eigh+canonicalize_signs SCF-13)
- `crates/pyscf-scf/src/occ.rs` — `default_get_occ` stub (Aufbau-fill)
- `crates/pyscf-scf/src/rdm.rs` — `default_make_rdm1` stub
- `crates/pyscf-scf/src/energy.rs` — `default_energy_elec` + `default_energy_tot` stubs
- `crates/pyscf-scf/src/analyze.rs` — `analyze`, `mulliken_pop`, `mulliken_meta`, `dip_moment` stubs + `MullikenResult`
- `crates/pyscf-scf/src/convert.rs` — `to_rhf`/`to_uhf`/`to_ghf` stubs; `to_rks_stub`/`to_uks_stub` returning `NotYetImplemented{phase:4}`
- `crates/pyscf-scf/src/scanner.rs` — `as_scanner` stub
- `crates/pyscf-scf/src/rhf.rs` — `RHF` struct (32 fields, manual Debug, `new`, `kernel`, `to_kernel_config`)
- `crates/pyscf-scf/src/uhf.rs` — `UHF` struct (alpha/beta-pair MO slots)
- `crates/pyscf-scf/src/ghf.rs` — `GHF` struct (2c spinor MO)
- `crates/pyscf-scf/src/error.rs` — `ScfError` enum + `From<ScfError> for PyscfRsError`
- `crates/pyscf-scf/tests/hooks_kernel_types.rs` — 4 tests
- `crates/pyscf-scf/tests/attribute_floor.rs` — 4 tests
- `.planning/phases/03-scf-pyo3-bindings/03-03-SUMMARY.md` (this file)

### Modified (3 files)
- `crates/pyscf-scf/Cargo.toml` — 6 deps + "NO pyo3" documentation
- `crates/pyscf-scf/src/lib.rs` — 16-module tree + re-exports (previously a `#![forbid(unsafe_code)]`-only stub)
- `Cargo.lock` — pyscf-scf gained 6 dep entries

## Override Hooks Trait — Method Inventory (SCF-08)

11 surface methods covering the SCF-08 10 logical override points:

| # | Method | Signature | Body source for plan 03-11 |
|---|--------|-----------|-----------------------------|
| 1 | `get_hcore` | `&Mole -> Density` | `pyscf/scf/hf.py:1356` |
| 2 | `get_ovlp` | `&Mole -> Density` | `pyscf/scf/hf.py:1360` |
| 3 | `get_init_guess` | `&Mole, &InitGuessMode -> Density` | `pyscf/scf/hf.py:1383-1462` |
| 4 | `get_jk` | `&Mole, &Density -> (J, K)` | `pyscf/scf/hf.py:1465` |
| 5 | `get_veff` | `&Mole, &Density -> Veff` | `pyscf/scf/hf.py:1471` |
| 6 | `get_fock` | h1e, s1e, vhf, dm, cycle, diis_state -> Density | `pyscf/scf/hf.py:1482` |
| 7 | `eig` | fock, s1e -> MOCoefficients | `pyscf/scf/hf.py:1349-1357` (with SCF-13 canonicalize_signs) |
| 8 | `get_occ` | mo_energy, nelec -> Vec<f64> | `pyscf/scf/hf.py:1499` |
| 9 | `make_rdm1` | mo -> Density | `pyscf/scf/hf.py:1517` |
| 10 | `energy_elec` | dm, h1e, vhf -> (E_elec, E_coul) | `pyscf/scf/hf.py:1556` |
| 11 | `energy_tot` | dm, h1e, vhf -> Energy | `pyscf/scf/hf.py:1574` |

The plan title says "10 hooks"; the trait expands to 11 because upstream `Method.energy_elec` and `Method.energy_tot` are independently overrideable in `pyscf/scf/hf.py:1556` and `pyscf/scf/hf.py:1574`. Treating them as a single hook would lose fidelity with the Python `class.kernel` MRO.

## 30-Attribute Floor Inventory (SCF-14)

32 fields on `RHF` (≥30 floor, all from upstream `pyscf/scf/hf.py:1716-1724` `_keys`):

1. `mol`
2. `mo_coeff` (None until `kernel()`)
3. `mo_energy` (None until `kernel()`)
4. `mo_occ` (None until `kernel()`)
5. `e_tot` (0.0 initial)
6. `e_elec` (0.0 initial)
7. `converged` (false initial)
8. `cycles` (0 initial — exposed for ORACLE-02 arm 3)
9. `verbose` (3)
10. `chkfile` (None)
11. `max_memory` (4000.0)
12. `direct_scf` (true)
13. `direct_scf_tol` (1e-13)
14. `init_guess` ("minao")
15. `level_shift` (0.0)
16. `damp` (0.0)
17. `diis` (true)
18. `diis_space` (8)
19. `diis_start_cycle` (1)
20. `diis_damp` (0.0)
21. `diis_file` (None)
22. `max_cycle` (50)
23. `conv_tol` (1e-9)
24. `conv_tol_grad` (None — defaults to sqrt(conv_tol) in plan 03-11 at runtime)
25. `with_df` (None — populated by plan 03-05)
26. `disp` (None)
27. `do_disp` (false)
28. `irrep_nelec` (empty HashMap — v1 is C1)
29. `nelec` (None)
30. `callback` (None — populated by plan 03-07 PyO3 bridge)
31. `scf_summary` (empty HashMap)
32. `opt` (None — populated by plan 03-04 DIIS / 03-05 DF)

`UHF` and `GHF` have the same 32-field count (alpha/beta and 2c spinor variants of mo_coeff/mo_energy/mo_occ respectively).

## No pyo3 Dependency Confirmation

```
$ grep -i pyo3 crates/pyscf-scf/Cargo.toml
description = "Self-consistent field kernels (RHF/UHF/GHF + DIIS + DF-HF). No pyo3 dep — D-01."
# NOTE: pyo3 is FORBIDDEN here (D-01 trait-callback bridge — pyo3 lives in pyscf-py).
$ grep -rn "use pyo3" crates/pyscf-scf/
(no matches)
```

D-01 algebra-wall analog confirmed: pyscf-scf depends only on pyscf-core, pyscf-algebra, pyscf-gto, pyscf-runtime, thiserror, tracing.

## Unimplemented Stubs Plan 03-11 Must Fill

20 `unimplemented!()` panics located across 11 files:

| File | Count | Bodies |
|------|-------|--------|
| `fock.rs` | 5 | `default_get_hcore`, `default_get_ovlp`, `default_get_jk`, `default_get_veff`, `default_get_fock` |
| `eig.rs` | 1 | `default_eig` (must call `pyscf_algebra::eigh` + `pyscf_core::canonicalize_signs` — SCF-13) |
| `occ.rs` | 1 | `default_get_occ` |
| `rdm.rs` | 1 | `default_make_rdm1` |
| `energy.rs` | 2 | `default_energy_elec`, `default_energy_tot` (oracle_sum/oracle_dot per Pitfall 9) |
| `analyze.rs` | 4 | `analyze`, `mulliken_pop`, `mulliken_meta`, `dip_moment` |
| `convert.rs` | 3 | `to_rhf`, `to_uhf`, `to_ghf` (to_rks/to_uks return `NotYetImplemented{phase:4}` instead of panicking) |
| `scanner.rs` | 1 | `as_scanner` |
| `init_guess.rs` | 1 | `init_guess_by_1e` |
| `kernel_impl.rs` | 1 | `scf_loop` (the SCF cycle body — verbatim port of `pyscf/scf/hf.py:48-244`) |

(Total: 20 `unimplemented!`. Documented in commit `e001289` and consumable via `grep -rn "unimplemented" crates/pyscf-scf/src/`.)

## Decisions Made

1. **11 trait methods, not 10.** Upstream `Method.energy_elec` and `Method.energy_tot` are independently overrideable, so the trait splits them. The plan title's "10 methods" is the SCF-08 *logical* override count.
2. **RHF/UHF/GHF::kernel returns by value, not by `&ScfResult` via `Box::leak`.** Plan body suggested the latter which would leak memory per call. Implementation returns `ScfResult` by value; converged scalars mirror into RHF fields before returning.
3. **ScfError→PyscfRsError bridge via `CoreError::InvalidMolecule(String)`.** Plan referenced `CoreError::Other(String)` which doesn't exist on the enum.
4. **Manual `Debug` impl on RHF/UHF/GHF.** `Box<dyn Any>` and `Box<dyn Fn>` slots don't derive `Debug`.
5. **Placeholder rhf/uhf/ghf structs land in Task 1 GREEN.** Without them, analyze/convert/scanner (which name `crate::RHF/UHF/GHF`) fail to compile. Task 2 GREEN replaces with the 30-attribute floor.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan referenced non-existent `CoreError::Other` enum variant**
- **Found during:** Task 1 (error.rs creation)
- **Issue:** Plan body specified `pyscf_core::CoreError::Other(format!("{}", e))` in `From<ScfError>` impl, but the `CoreError` enum has only 3 variants: `InvalidMolecule(String)`, `BasisParse(String)`, `DimensionMismatch { expected, actual }`. No `Other` arm exists.
- **Fix:** Routed `From<ScfError> for PyscfRsError` through `CoreError::InvalidMolecule(String)` — the only String-carrying catch-all on the enum. Same fix applied to `parse_init_guess_mode`'s error path in `init_guess.rs`.
- **Files modified:** `crates/pyscf-scf/src/error.rs`, `crates/pyscf-scf/src/init_guess.rs`
- **Verification:** `cargo build -p pyscf-scf` succeeds.
- **Committed in:** `e001289` (Task 1 GREEN)

**2. [Rule 1 - Bug] Plan's RHF::kernel body used `Box::leak` causing memory leak per call**
- **Found during:** Task 2 (rhf.rs creation)
- **Issue:** Plan body had `Ok(Box::leak(Box::new(result)))` returning a `&'static ScfResult`. Each kernel call would leak `sizeof(ScfResult)` bytes — chemistry workflows easily run 1000+ SCF computations per session.
- **Fix:** Changed `RHF::kernel` (and UHF/GHF::kernel) to return `Result<ScfResult, PyscfRsError>` (by value). The converged scalar state (mo_coeff/mo_energy/mo_occ/e_tot/converged/cycles) is mirrored into the struct fields *before* `Ok(result)`, so downstream readers can still access them via `&rhf.mo_coeff`.
- **Files modified:** `crates/pyscf-scf/src/rhf.rs`, `crates/pyscf-scf/src/uhf.rs`, `crates/pyscf-scf/src/ghf.rs`, `crates/pyscf-scf/tests/attribute_floor.rs`
- **Verification:** `cargo test -p pyscf-scf` — 8/8 tests pass.
- **Committed in:** `075cd9f` (Task 2 GREEN)

**3. [Rule 3 - Blocking] Plan's Cargo.toml-only update for tests required dev-dependency to expose pyscf_core**
- **Found during:** Task 1 RED (tests/hooks_kernel_types.rs `use pyscf_core::Energy`)
- **Issue:** Initially `cargo build --tests` failed because the integration test crate could not resolve `pyscf_core` — Rust test crates only see the test target's transitive deps via the parent crate's normal dependencies, which is fine here since pyscf-scf depends on pyscf-core directly.
- **Resolution:** No additional change needed — once Task 1 GREEN added `pyscf-core = { path = "../pyscf-core" }` to `pyscf-scf/Cargo.toml`, the integration test resolved it automatically. False alarm on first RED check.
- **Committed in:** `e001289` (Task 1 GREEN)

---

**Total deviations:** 2 critical auto-fixes (1 enum reference bug, 1 memory leak) + 1 false-alarm investigation
**Impact on plan:** Both critical fixes prevented compile errors / per-call memory leaks. No scope creep — all changes are inside the plan's named files.

## Issues Encountered

- **Worktree base mismatch on init:** The worktree HEAD was at `a05f896` (a previous worktree's branch) while the orchestrator expected `5ace1f55`. Resolved via `git reset --soft 5ace1f55` followed by `git reset` + `git checkout -- . && git clean -fd` to clear the unrelated worktree state. Clean state confirmed via `git rev-parse HEAD == 5ace1f55`.
- **Cargo.lock cannot update under `--locked`:** Initial `cargo build -p pyscf-scf --locked` fails because pyscf-scf's added deps cause a Cargo.lock delta. Resolved by dropping `--locked` for the in-plan build (lock delta is recorded in commit `e001289`). The orchestrator's wave-completion verification can re-run with `--locked` once all wave-2 plans converge.

## User Setup Required

None — pure Rust scaffolding plan, no external service config.

## Next Wave Readiness

- **Wave 3 plan 03-11 ready to start:** All 20 `unimplemented!('plan 03-11')` markers are in place; plan 03-11 replaces those bodies without re-touching any signature.
- **Wave 2 sibling plans:** 03-04 (DIIS) and 03-05 (DF-HF) and 03-06 (chkfile) can run in parallel — they extend pyscf-scf's `with_df`/`opt` slots and the `InitGuessMode::Chkfile` arm rather than modifying the trait surface.
- **Plan 03-07 (PyO3 bridge):** The `OverrideHooks` trait is the SCF-08 contract; `PyOverrideBridge` will implement it from pyscf-py.
- **Plan 03-10 (Python introspection):** The 30-attribute floor is in place on RHF/UHF/GHF; Python-side `hasattr(rhf, "diis_space")` etc. will succeed once plan 03-07 wires the `#[pyclass]`.

## Self-Check

Files claimed created/modified, verified to exist:

```
FOUND: crates/pyscf-scf/Cargo.toml
FOUND: crates/pyscf-scf/src/lib.rs
FOUND: crates/pyscf-scf/src/hooks.rs
FOUND: crates/pyscf-scf/src/kernel.rs
FOUND: crates/pyscf-scf/src/kernel_impl.rs
FOUND: crates/pyscf-scf/src/init_guess.rs
FOUND: crates/pyscf-scf/src/fock.rs
FOUND: crates/pyscf-scf/src/eig.rs
FOUND: crates/pyscf-scf/src/occ.rs
FOUND: crates/pyscf-scf/src/rdm.rs
FOUND: crates/pyscf-scf/src/energy.rs
FOUND: crates/pyscf-scf/src/analyze.rs
FOUND: crates/pyscf-scf/src/convert.rs
FOUND: crates/pyscf-scf/src/scanner.rs
FOUND: crates/pyscf-scf/src/rhf.rs
FOUND: crates/pyscf-scf/src/uhf.rs
FOUND: crates/pyscf-scf/src/ghf.rs
FOUND: crates/pyscf-scf/src/error.rs
FOUND: crates/pyscf-scf/tests/hooks_kernel_types.rs
FOUND: crates/pyscf-scf/tests/attribute_floor.rs
```

Commits claimed, verified in `git log --oneline`:

```
FOUND: ffabc26 — test(03-03) Task 1 RED
FOUND: e001289 — feat(03-03) Task 1 GREEN
FOUND: 72970fd — test(03-03) Task 2 RED
FOUND: 075cd9f — feat(03-03) Task 2 GREEN
```

## Self-Check: PASSED

---

*Phase: 03-scf-pyo3-bindings*
*Plan: 03*
*Completed: 2026-05-11*
