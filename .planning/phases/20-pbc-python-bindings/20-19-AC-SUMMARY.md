# 20-19 A + C SUMMARY — molecular overlay fallthrough, `gen_grid.gen_uniform_grids`

**Shipped:** 2026-09-14. Python-only; no Rust file edited, no `.so` built, nothing staged or committed (D-20-A). Items B (`crates/pyscf-py/src/pbc/gto.rs`) and D (repo copy) ran concurrently in other sessions; B's rebuild `.so 18:49:53` is the build every measurement below ran on.

## Result

| gate | before | after |
|---|---|---|
| upstream PBC suite (20-18 harness, same 815 denominator) | 1 / 815 (20-18) | **11 / 815** — `measurements/upstream-pbc-suite-after-20-19.md` |
| collection errors | 746 | **684** — every one is mismatch M1 or M2 below; **0** missing-name import errors remain |
| upstream `pyscf.pbc` modules that import (195, fresh interpreter) | 41 (20-17) | **73** |
| identity gate `test_pbc_identity_gate.py` | 18/18 | **18/18** |
| `test_overlay_resolution.py` | 1 passed | **1 passed** |
| full `python/pyscf/tests` | baseline (this session, same tree before the edits): 10 failed, 384 passed, 4 skipped, 3 xfailed, 9 errors | **10 failed, 411 passed, 0 skipped, 3 xfailed, 9 errors**; FAILED/ERROR id lists **identical** (`diff` of `-rfE` lines); +4 passes are `test_intor_spinor.py` (skip → pass), the rest are tests other sessions added meanwhile |

The ≥80 % target is not in reach from A/C alone: the two native re-exports below block 684 tests, and a counterfactual with both lifted reaches 49 / 815 before the native `Cell`/`KSCF` API surface stops it (details in the measurement).

## A — what changed

| file | change |
|---|---|
| `python/pyscf/_passthrough.py` (new) | `package_getattr(globals(), native_modules)`: PEP 562 `__getattr__` for a molecular overlay package. It never executes upstream `__init__` unless it must: the upstream `__init__.py` found on the extended `__path__` is **parsed with `ast`** into `from X import a as b` targets, star-import sources and body-defined names. Lookup order for a missing name: native module (`pyscf._native.<pkg>`, e.g. `pyscf.dft.NumInt`) → explicit `from X import name` → upstream submodule file → star sources → only for names upstream's `__init__` body defines (`scf.HF`, `scf.rhf`, `dft.XC`, `grad.grad_nuc`, …) the upstream `__init__` executed once as the non-package `<pkg>._upstream_init`. `module_getattr(name)` does the same for a shim module shadowing an upstream file (`scf/hf.py` → upstream `scf/hf.py` as `pyscf.scf._upstream_hf`). Re-entrancy guard per `(module, name)`; non-`AttributeError` failures become `AttributeError` (20-17 convention, `hasattr` stays safe). **Silent**: no D-PBC-35 warning for molecular packages. |
| `python/pyscf/{gto,scf,dft,mp,cc,grad,geomopt}/__init__.py` | after the native imports and `__all__`: `__path__ = pkgutil.extend_path(__path__, __name__)` + `__getattr__ = package_getattr(globals(), ("pyscf._native.<pkg>",))`. Native names stay module globals, so `__getattr__` never sees them. `geomopt` also pins `sys.modules["pyscf.geomopt.{geometric,berny}_solver"]` to the native shims so `extend_path` cannot let `import pyscf.geomopt.geometric_solver` load (and rebind) the upstream file. |
| `python/pyscf/scf/{hf,uhf,ghf}.py` | `__getattr__ = module_getattr(__name__)`; `RHF`/`UHF`/`GHF` stay native. |
| `python/pyscf/__init__.py` | `extend_path` moved first; binds the overlay **packages** `pyscf.{scf,dft,gto}` instead of the flat `pyscf._native.{scf,dft,gto}` modules (before, `from pyscf import gto` returned the native module — no `__path__`, no fallthrough — until something imported `pyscf.gto` and rebound the attribute, so upstream code saw an import-order-dependent `gto`). Adds a root `__getattr__` for upstream submodules (`pyscf.lib`, …) and `pyscf.DEBUG`; upstream's root `__init__` is never executed. |
| `python/pyscf/tests/conftest.py` | **guard, not a loosening** — see D1. |

Resolves (checked in the harness environment): `pyscf.gto.basis`, `pyscf.gto.moleintor`, `pyscf.gto.mole`, `pyscf.gto.ATOM_OF` (0), `pyscf.gto.ATM_SLOTS` (6), `gto.parse`, `gto.getints`, `pyscf.dft.radi`, `dft.Grids`, `scf.hf.SCF`, `scf.addons`/`atom_hf`/`_vhf`/`chkfile` (as modules), `pyscf.mp.mp2`, `pyscf.cc.gccsd`/`ccsd_t`/`ccsd`, `pyscf.grad.rhf`, `pyscf.geomopt.addons` — all from the vendored 2.12.1 tree, 0 site-packages leaks. Identity: `pyscf.scf.RHF`, `pyscf.scf.hf.RHF`, `pyscf.gto.Mole`, `pyscf.gto.M`, `pyscf.dft.RKS`, `pyscf.mp.MP2`, `pyscf.cc.CCSD`, `pyscf.grad.Gradients`, `pyscf.geomopt.geometric_solver` are all `is` their `pyscf._native` object.

## C — `python/pyscf/pbc/dft/gen_grid.py`

Upstream `pbc/dft/gen_grid.py` has no native counterpart for anything but `UniformGrids`/`BeckeGrids`/`AtomicGrids` (checked: `grep uniform_grids crates/pyscf-py/src` → none).
- `gen_uniform_grids` / `get_uniform_grids`: resolved lazily from `pyscf.pbc.gto.cell` (upstream `gen_grid.py:25` re-exports them from there), so `gen_grid.gen_uniform_grids is pyscf.pbc.gto.cell.gen_uniform_grids` and no `libpbc`/molecular-`dft` load is needed; cached in the module globals. Works on a native `Cell` (`(125, 3)` for mesh 5³).
- Every other upstream name importers use (`make_mask`, `BLKSIZE`, `NBINS`, `CUTOFF`, `ALIGNMENT_UNIT`, `get_becke_grids`/`gen_becke_grids`, `libpbc`, the prune functions) keeps falling through to upstream `gen_grid.py` via `_unported.fallthrough_getattr` — which now imports thanks to A.
- `__all__` unchanged (native names only), so `which_impl` is unchanged. Announced once per family as before.
- Effect: the 44 `gen_uniform_grids` tests of 20-18 no longer fail on that name (`cc/test_krccsd_gamma`, `ci/test_ci` now reach test bodies; the `cc/test_eom_*`/`test_krccsd` files stop later, at M1).

## Semantic mismatches recorded (not fixed)

| id | mismatch | measured effect |
|---|---|---|
| **M1** | `pyscf.scf.{hf,uhf,ghf}` shims bind the native `RHF`/`UHF`/`GHF`; upstream Python subclasses/reads them: `scf/rohf.py:349 class ROHF(hf.RHF)` + `:384 uhf.UHF.get_init_guess`, and 39 upstream files use `hf.RHF`. `rohf` is on the import path of `scf.addons`, `atom_hf`, `dft/__init__`, `dft.roks`, `mp.mp2`, … | 70 / 195 upstream PBC modules; 240 tests at collection + 15 `test_newton` tests at runtime |
| **M2** | `pyscf.gto.Mole` is the native class and not subclassable; `pbc/df/ft_ao.py:565 class ExtendedMole(gto.Mole)`. Also `fakemol = gto.Mole()` in `pbc/df/fft.py:117`, `pbc/dft/multigrid/multigrid.py:422`, `pbc/grad/rks_stress.py:314` would build a native `Mole` and set `_atm/_bas/_env` on it | 43 modules (101 once M1 is lifted); 444 tests at collection |
| M3 | Upstream code calling `gto.M(...)`, `scf.RHF(mol)`, `dft.RKS(mol)` through the molecular packages gets native objects; `pyscf.scf.HF`/`ROHF`/`RKS`/… fall through to upstream `scf/__init__` functions which then build native or native-based classes; upstream `SCF` methods doing `from pyscf.scf import uhf; uhf.UHF(...)` get native `UHF` | not separately measured (masked by M1/M2); no `isinstance(x, gto.Mole)` exists in non-test `pyscf/pbc` |
| M4 | Upstream `gto/__init__` rebinds `gto.eval_gto` to the function after importing the submodule; the lazy fallthrough cannot, so once `pyscf.gto.eval_gto` (module) is imported the attribute stays the module | no upstream PBC caller of `gto.eval_gto(...)` found |
| M5 | `conftest._load_upstream` executes vendored `pyscf/__init__.py`; with A it SUCCEEDS but its `gto`/`scf` are the overlay packages (native `M`/`RHF`), so every in-process oracle would compare pyscf-rs with itself — D1 | measured: the unguarded run spent > 40 min in `test_scf_rhf_benzene` running native RHF twice |
| M6 | Under plain `pytest`/`.venv`, `extend_path` falls through to site-packages **2.14.0** (pre-existing, `python-suite-triage.md`); A makes it reachable for more names. `test_intor_spinor.py` (skip → pass) now compares against upstream 2.14.0 `gto.mole` from site-packages | 4 tests |

A counterfactual run with M1/M2 lifted in a scratch overlay copy (identity broken for `pyscf.scf.hf.RHF` and `pyscf.gto.Mole`, measurement only) gives 170 / 195 importable modules and 49 / 815 passing; its next blockers are listed in the measurement.

## Deviations

- **D1 — `python/pyscf/tests/conftest.py` edited (a test file).** `_load_upstream` now refuses (raises `RuntimeError`, pops the module) any `_upstream_pyscf` whose `gto` or `scf` resolves into the overlay — on both the fresh-exec and the cache-hit path (the latter because `test_intor_spinor.py` caches a module with an upstream `gto` but overlay `scf`). Without it, A would have turned the 9 `upstream`-fixture errors and the 4 in-process-oracle failures into vacuous native-vs-native comparisons. With it they fail/error at the same ids as the triage baseline, now with an explicit message pointing at `PYSCF_RS_UPSTREAM_PYTHON` (the CI oracle, unchanged). The first after-run (without the guard) was killed at `test_scf_rhf_benzene` (`pysuite/after-noguard-killed.log`).
- **D2 — root `python/pyscf/__init__.py` changed**, beyond the listed packages (reason in the table). `pyscf.scf`/`dft`/`gto` are now the overlay packages instead of the flat native modules; every name they had is still reachable and identical (package globals, or the native-module step of `__getattr__`: `dft.NumInt`, `scf.Scanner`).
- **D3 — suite driver split per family.** `run_families.sh` replaces `run_all.sh`'s single shuffled `xargs` with one batch per family so the `.so` mtime is stamped per family (the brief's coordination rule); per-file command and every flag identical.
- **D4 — no D-PBC-35 change.** More upstream PBC modules import now, so `PbcUpstreamFallthroughWarning` fires for more families in practice (e.g. `tools`, via upstream `pbc/gto/cell.py`); still once per family. `docs/pbc-status.md` and the `_unported.FAMILIES` notes still quote the 20-17 import counts (41 / 195) — stale, left for the 20-18 rollup.
- **D5 — CubeCL manual (AGENTS.md §3) not consulted:** no kernel or Rust code.

## Verification

| command | result |
|---|---|
| `.venv/bin/pytest python/pyscf/tests/test_overlay_resolution.py test_pbc_identity_gate.py test_pbc_unported.py -q -p no:cacheprovider` | 68 passed (identity 18/18) |
| `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider -rfE` before edits (`target/p20-19-suite/pysuite/baseline.log`, `.so` 18:28:19 → 18:49:53 mid-run, B's rebuild) | 10 failed, 384 passed, 4 skipped, 3 xfailed, 9 errors |
| same, after A + C + D1 (`pysuite/after.log`, `.so` 18:49:53 start = end) | 10 failed, 411 passed, 3 xfailed, 9 errors; FAILED/ERROR ids identical to baseline |
| upstream PBC suite, `target/p20-19-suite/run_families.sh overlay 2.14.0 run-overlay 3 3` under `systemd-run MemoryMax=11G`, detached | 11 / 815; 0 crash / timeout; `.so` 18:49:53 for all 15 families |
| import probe `target/p20-19-suite/probe_imports.py` | 73 / 195 |
