# Carryover — Phase 20 upstream `pyscf/pbc` suite ≥ 80 % target NOT MET (structural)

**Source:** `.planning/phases/20-pbc-python-bindings/20-VERIFICATION.md` (20-18 rollup,
2026-09-14). Nothing below is absorbed into a pass; no test was excluded to move the number.

## Measured

| run | passed / collected | ported families only | collection errors |
|---|---:|---:|---:|
| 20-18 overlay (`measurements/upstream-pbc-suite.md`, `.so` 17:42:00) | 1 / 815 | 1 / 660 | 746 |
| 20-19 A+C overlay (`measurements/upstream-pbc-suite-after-20-19.md`, `.so` 18:49:53) | **11 / 815** (1.35 %) | 11 / 660 | 684 |
| control, pure vendored 2.12.1 | 808 / 815 | 659 / 660 | 0 |
| counterfactual (scratch overlay, identity broken for `scf.hf.RHF` and `gto.Mole`; measurement only) | 49 / 815 | — | 77 |

The 20-18 final-tree re-run (after the 20-19 D `.so` 19:52) is recorded in `20-VERIFICATION.md` §3.

Denominator: 920 collected with no exclusions → −105 `pytest.ini` `-k "not _high_cost and not _skip"`
deselection → −0 from the two ignored `cc/test/test_h_*.py` scripts (no test functions) → **815**
(111 files). The other `pytest.ini` globs (`*_slow*`, `*test_kproxy*`, `*test_proxy*`, `*test_bz*`,
`*test_ks_noimport*`) match no file in the 2.12.1 `pbc` tree.

## Why it is structural

Every one of the 684 remaining collection errors is one of two mismatches that the Phase-3 /
Phase-20 identity contract creates on purpose:

| tests | files | first error | cause |
|---:|---:|---|---|
| 444 | 43 | `TypeError: type 'pyscf._native.gto.Mole' is not an acceptable base type` | upstream `pyscf/pbc/df/ft_ao.py:565` `class ExtendedMole(gto.Mole)`; the overlay's `pyscf.gto.Mole` is the native, non-subclassable class |
| 240 | 39 | `AttributeError: type object 'pyscf._native.scf.UHF' has no attribute 'get_init_guess'` | upstream `pyscf/scf/rohf.py:349/:384` `class ROHF(hf.RHF): get_init_guess = uhf.UHF.get_init_guess`; the overlay binds native `RHF`/`UHF`/`GHF` in `python/pyscf/scf/{hf,uhf,ghf}.py` |

Upstream PBC Python subclasses the molecular classes; the overlay replaces them with native
classes that Python cannot subclass (and whose class attributes are not upstream's). Lifting both
(counterfactual) reaches only 49 / 815 before the next wall: the native `Cell`/`KSCF` Python API
surface (`Cell.copy(**kw)` 83, `Cell.max_memory` 94, `Cell.nao` 51, `KSCF.mo_occ` not writable 31,
mixed AE/pseudo cell 26, `KUHF(KPoints)` 20, `convert_from_` 12, `remove_soscf` 8).

Taxonomy of the 804 non-passing tests after 20-19 (same file): unported family 155; A-mismatch at
import 529; same mismatch at runtime 15 (`scf/test_newton.py`); native API-surface gap 74; refused
input 18; **numerical 13** (own carryover: `20-upstream-suite-numerical-mismatches.md`); crash /
signal / timeout 0.

## Unblock (a design decision, not a bug fix)

1. Decide the subclassability contract: make `pyscf._native.gto.Mole` and `pyscf._native.scf.{RHF,UHF,GHF}`
   `#[pyclass(subclass)]` with upstream-compatible class attributes (`get_init_guess` etc.), OR
   serve upstream's Python classes under `pyscf.scf.hf` / `pyscf.gto.mole` for upstream callers
   while keeping native names at the package top level (breaks `pyscf.scf.hf.RHF is pyscf._native.scf.RHF`).
2. Then the native `Cell`/`KSCF` API surface list above (counterfactual next blockers).
3. Re-run with the 20-18 harness (`target/p20-19-suite/run_families.sh`, `--import-mode=prepend`).

## Harness note (must survive into CI)

`pytest.ini` sets `--import-mode=importlib`; under it upstream test modules are named
`pyscf.pbc.<fam>.test.<file>` and their parent packages are imported from the file system, i.e.
the VENDORED `pyscf/pbc/__init__.py`, even with the overlay first on `PYTHONPATH` —
0 tests collected, 111/111 files collection errors (`upstream-pbc-suite.md` §Harness finding).
Every number above uses `--import-mode=prepend` after `-c pytest.ini`. A naive
`pytest pyscf/pbc` with the repo config measures neither the overlay nor upstream.
