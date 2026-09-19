# 20-17 SUMMARY — overlay shims, `which_impl`, announced upstream passthrough (D-PBC-35)

**Shipped:** 2026-09-14. All five tasks are done. The default policy (announced passthrough) was kept; the escalation trigger did not fire. **No Rust file was edited and no `.so` was rebuilt**; nothing was staged or committed (D-20-A). `crates/pyscf-py/src/pbc/dft.rs` and `python/pyscf/tests/test_pbc_dft.py` were not touched.

## Task 1 — shims (audit, then gap fill)

**Audit.** 20-09…20-15 had already shipped identity re-export overlays for `gto`, `df`, `scf`, `dft`, `symm`, `mp`, `cc`, `ci`, `ao2mo`. Every one re-exports the `_native` object, with no wrapping, and binds aliases as aliases (`DF is GDF`, `PWDF is AFTDF`, `RSGDF is RSDF`). They were left as they were. There were three gaps: `lib`, `tools`, and `dft.gen_grid`.

**New overlay files.** Every name in each file's `__all__` is the same `_native` object. Unported names fall through lazily, with an announcement, to the upstream module the file shadows.

| file | native names | fallthrough |
|---|---|---|
| `python/pyscf/pbc/lib/__init__.py` | — (imports `kpts_helper`, as upstream does) | `extend_path` |
| `python/pyscf/pbc/lib/kpts_helper.py` | the 13 names of `_native.pbc.lib.kpts_helper` | `loop_kkk`, `KptsHelper`, `round_to_fbz`, … → upstream `kpts_helper.py` |
| `python/pyscf/pbc/lib/kpts.py` | `KPoints` (`is symm.KPoints`), `make_kpts`, `KPT_DIFF_TOL` | upstream `kpts.py` |
| `python/pyscf/pbc/tools/__init__.py` | `fft`/`ifft`/`fftk`/`ifftk`/`get_coulG`/`madelung`/`super_cell`/`cell_plus_imgs`/`cutoff_to_mesh`/`mesh_to_cutoff`/`ExxDiv` + `get_kconserv`/`get_kconserv3`/`intersection` | named submodules; otherwise `tools.pbc`, then upstream `print_funcs` |
| `python/pyscf/pbc/tools/pbc.py` | the same 14 (so `from pyscf.pbc.tools.pbc import super_cell`, used in 5 examples, is native) | upstream `pbc.py` |
| `python/pyscf/pbc/dft/gen_grid.py` | `UniformGrids`, `BeckeGrids`, `AtomicGrids` | upstream `gen_grid.py` |

The fallthrough uses `_unported.load_upstream_module`. It executes the shadowed upstream file under the private name `<parent>._upstream_<leaf>`, so relative imports still resolve, and a failed upstream import becomes `AttributeError`, so `hasattr` stays safe.

**Edits to existing overlays.**
- `pbc/dft/__init__.py`: a comment plus `from pyscf.pbc.dft import gen_grid` appended. Upstream's `__init__` imports `gen_grid`, and example 20 spells `dft.gen_grid.BeckeGrids`.
- `pbc/gto/__init__.py` `__getattr__`: it now refuses dunder names, and turns an upstream-source `ImportError` into `AttributeError`. Before this, `hasattr(pyscf.pbc.gto, "anything")` raised `ImportError: cannot import name 'radi'` (a pre-existing 20-09 defect).

## Task 2 — Python-level dispatch (audit only)

Upstream `pbc/scf/__init__.py:40-106` and `pbc/dft/__init__.py:37-119` were compared name by name with the overlays. Every name is covered.
- The `isinstance(kpts, KPoints)` branch of `KRHF`/`KUHF`/`KGHF`/`KRKS`/`KUKS`/`KRKSpU`/`KUKSpU` lives in the native constructors. This is 20-12 D1 / 20-13 D1: a Python function under those names would break the identity gate.
- `RHF`/`UHF`/`GHF`/`ROHF`/`HF`/`KHF`/`KS`/`KKS`/`RKS`/`UKS`/`GKS`/`ROKS` are Python functions in the overlays.
- **No change was needed.** The plan truth "dispatch lives in the Python shim, never in Rust" is superseded by 20-12 D1. The native check is a type test, not MRO.

## Task 3 — registry + `which_impl` (`python/pyscf/pbc/_unported.py`)

**`FAMILIES`** has 21 entries, one per upstream `pyscf/pbc/<family>` directory in 2.12.1. Each entry carries `status`, `upstream`, `rust`, `bound_by`, `missing`, `names` and `note`.

| status | families |
|---|---|
| **upstream** (10) | `grad` (Phase 18, uncommitted, unbound), `geomopt` (stub), `tdscf`/`gw`/`adc`/`x2c`/`eph` ("Rust implementation exists (Phase 19), unbound in Python"), `tddft` (no crate), `mpicc`/`mpitools` (stub `pyscf-pbc-mpi`) |
| **partial** (6) | `scf`, `tools`, `gto`, `symm`, `ci`, `lib` |
| **native** (5) | `df`, `dft`, `mp`, `cc`, `ao2mo` |

What `missing` records for the partial families:
- **`scf`:** `newton_ah`, `stability`, `cphf`, `_response_functions`, `scfint`, `rsjk`. The note says Rust `newton_ah`/`stability`/`cphf`/`response` exist in the working tree but are unbound. `kuhf_ksymm`/`kghf_ksymm` are `refused`.
- **`tools`:** the plan's six modules plus `make_test_cell`, and the 7 `pbc.py` names that are ported in Rust but unbound (20-14 Task 6).
- **`lib`:** `arnoldi`, `linalg_helper`, `chkfile`, `ktensor`, plus 6 `kpts_helper` names.
- **`gto`:** `ecp`. **`symm`:** `pyscf_spglib`. **`ci`:** `cisd`.

**`which_impl(name)`** accepts a family or a dotted name, with or without the `pyscf.pbc.` prefix.
- It returns `native`/`partial`/`upstream` from the registry for a family.
- For a dotted name it checks the `names` overrides first. It then walks overlay modules on disk: a name in an overlay module's `__all__` is `native`; a non-overlay name is `upstream`; an overlay shim module gives its `_PYSCF_RS_IMPL` (`partial`).
- Unknown families raise `KeyError`.

## Task 4 — one-time warning

`_PassthroughAnnouncer` is inserted at the front of `sys.meta_path` by `pyscf/pbc/__init__.py`.
- For `pyscf.pbc.*` it runs `PathFinder.find_spec`. If the origin is outside the overlay, it calls `announce(family, …)`. It **always returns `None`**, so resolution is unchanged.
- The shim fallthrough announces through the same once-per-family set.
- The class is `PbcUpstreamFallthroughWarning(UserWarning)`, exported from `pyscf.pbc`.
- The message names the family, its status, the resolved file and the registry note, points at `which_impl('<family>')` and `docs/pbc-status.md`, and says it is shown once.
- `stacklevel` points at user code; the frozen importlib frames are walked but not counted, because `warnings` skips them itself.

## Task 5 — docs + decision

- `docs/pbc-status.md` covers:
  - runtime checks and how to silence the warning;
  - how to make the overlay the imported `pyscf`;
  - the 21-family table, with native names, gaps, Rust state and measured import counts;
  - the broken-passthrough measurement;
  - the example-script gaps.
- **D-PBC-35** is a new row in `PBC-MASTER-PLAN.md §3`, after D-PBC-29. D-PBC-30…34 are already used in phase documents.
- `measurements/overlay-passthrough.md` holds the per-family and per-module tables. The tool is `measurements/overlay_passthrough_probe.py`; raw data is in `overlay-passthrough-{before,after}.json`.

## Measured: upstream passthrough under the overlay (195 vendored modules, fresh interpreter each)

| | import ok | overlay/stub | upstream ok | fail | warned probes |
|---|---:|---:|---:|---:|---:|
| before 20-17 | 31 | 12 | 19 | 164 | 0 |
| after 20-17 | **41** | 18 | **23** | **154** | 177 (0 silent upstream imports) |

- `site` (2.14.0 fallthrough) and `vendored` (2.12.1 fallthrough) configs give **identical** status for every module.
- The before→after diff has only error→ok and upstream→overlay transitions, so **no regression**.
- **154 of the 177 non-shadowed modules fail to import.** The cause is the molecular overlay packages (`pyscf.gto.basis` 39, `moleintor` 35, `pyscf.grad.rhf` 15, `pyscf.mp.mp2` 9, `mole` 9, …).
- Of the 10 upstream families, only `x2c`, `eph` and `mpitools` import even at package level. For a Python user, an unported family is effectively unavailable rather than served by upstream.

## Examples for 20-18 (constructor/attribute probe, no `kernel()`)

- **`20-k_points_scf.py`.** `gto.M`, `make_kpts`, `scf.KRHF`, `dft.KRKS`, `dft.gen_grid.BeckeGrids` (new shim) and `density_fit()` are native. Two gaps:
  - `xc='m06,m06'` is a meta-GGA, refused by `pyscf-pbc-dft/src/xc.rs:109` at kernel time;
  - `.newton()` raises `AttributeError` (unbound), which aborts the script.
- **`22-k_points_mp2.py`.** `gto.Cell()` build, `KRHF` + `kmf.kpts=` (`(8,3)` and a single `(3,)`), `KMP2`, gamma `scf.RHF(cell, kpt=)`, `with_df.ao2mo`, `get_hcore` and `energy_nuc` are native. Gaps:
  - `mp.RMP2` raises `NotImplementedError`, which aborts the script;
  - after that, `scf.addons.convert_to_uhf` fails to import upstream, `UMP2`/`GMP2` are refused, and gamma results are per-k lists.

## Verification

| command | result |
|---|---|
| `PYTHONPATH=$REPO/python .venv/bin/python -c "from pyscf.pbc import scf; import pyscf._native as n; assert scf.KRHF is n.pbc.scf.KRHF"` (cwd ≠ repo root) | exit 0 (see D1 for the repo-root cwd) |
| `which_impl` | 10 `upstream`, 6 `partial`, 5 `native`, exactly the registry lists; dotted-name cases in the test file |
| unported family warns once, naming the family | `grad`, `grad`, `grad.krhf`, `eph`, `eph.eph_fd` → exactly 2 warnings (`pyscf.pbc.grad is upstream`, `pyscf.pbc.eph is upstream`); importing all 17 overlay modules/stubs → 0 warnings |
| `.venv/bin/pytest python/pyscf/tests/test_pbc_unported.py test_pbc_identity_gate.py test_overlay_resolution.py -q -p no:cacheprovider` | **68 passed** (49 new, identity gate **18/18**, overlay resolution 1) |
| `grep -n '^| \*\*D-PBC-' .planning/pbc/PBC-MASTER-PLAN.md \| tail -3` | last row `259: \| **D-PBC-35** \| **Unported pyscf.pbc families: ANNOUNCED passthrough.**` (D2) |
| full `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider` | **10 failed, 372 passed, 4 skipped, 3 xfailed, 9 errors in 521.21 s** (exit 1, `target/py-20-17-fullsuite.log`, `.so` 16:58:45 unchanged). The failing/erroring ids are exactly the known molecular set of `measurements/python-suite-triage.md` / 20-15 (`test_panic_to_exception` ×2, `test_scf_cross_dispatch` ×3, `test_scf_ghf`, `test_scf_rhf_ccpvdz` ×3, `test_scf_rhf_h2o`; 9 `moleintor` errors). **Nothing new fails**; vs 20-15 (323 passed) the +49 are `test_pbc_unported.py`. Every PBC file passes. The escalation trigger did not fire: no test asserts clean stderr or warnings (grep of `python/pyscf/tests`), and the suite shows exactly 1 warning (the passthrough warning `test_pbc_unported.py` provokes on purpose) |

## Deviations

- **D1 — the plan's first verification command fails from the repo root and without `PYTHONPATH`, independent of this plan.**
  - From the repo root, cwd `''` imports the vendored `pyscf/`.
  - `.venv/bin/python` without `PYTHONPATH` imports site-packages **pyscf 2.14.0**, because `pyscf_rs.pth` appends `python/` *after* site-packages.
  - Under pytest, the passthrough target is 2.14.0, not the 2.12.1 oracle.
  - 20-18 must run examples as `PYTHONPATH=$REPO/python .venv/bin/python examples/pbc/<x>.py`. This is recorded in `docs/pbc-status.md §2`. The venv was not changed.
- **D2 — `grep -n 'D-PBC-' … | tail -3` cannot show the new row.** D-PBC ids are cited throughout the document through §10 (last hit is line 2491). The §3-table grep above shows it as the last decision row.
- **D3 — "native for the ten bound modules" vs "partial for six" overlap** (`scf`, `gto`, `symm`, `lib`, `tools` and `ci` are both bound and partial). Resolved as follows: family-level status follows the plan's ten-upstream / six-partial split, and dotted names resolve to `native` for every bound object (`which_impl('scf.KRHF') == 'native'`). The native families are therefore `df`, `dft`, `mp`, `cc` and `ao2mo`.
- **D4 — a fourth return value, `"refused"`, for dotted names only.** It covers `kuhf_ksymm`, `kghf_ksymm`, `RMP2`/`UMP2`/`GMP2`, `RCCSD`/…, and `RCISD`/…. These are neither native nor served by upstream, so calling them `upstream` would be false. Family-level results stay within the plan's three values.
- **D5 — upstream submodules of native families also announce** (e.g. `pyscf.pbc.df.incore`), keyed by family. The plan scoped the warning to unported families, but a silent fallthrough under a native family is the same defect class. It still fires once per family.
- **D6 — the passthrough is announced, not working** (measurement above). The policy stands as the plan's default. The root cause is in the molecular overlay packages, out of scope here, and is recorded in D-PBC-35, the doc and the measurement.
- **D7 — the shims change what upstream Python sees.** Upstream modules that import `pyscf.pbc.lib.kpts_helper`/`pyscf.pbc.tools.pbc` now receive native functions for the bound names. This is how `df.fft_jk`, `df.df_jk`, `tools.k2gamma`, `lattice`, `tril` and `print_funcs` went from error to importing. Their runtime behaviour on native return types (for example, always-complex arrays) is not verified. They are upstream code and are announced.
- **D8 — CubeCL manual (AGENTS.md §3) not consulted.** No kernel or Rust code was written.
