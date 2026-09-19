# Upstream `pyscf/pbc` test suite against the overlay (20-18 Task 3)

**Measured:** 2026-09-14 18:05–18:11 JST. HEAD `5e6dd65` plus the Phase-18/20 working tree.
**`.so`:** `python/pyscf/_native.abi3.so` mtime `2026-09-14 17:42:00`. It was the same at the start and end of every pass (collection, overlay run, control run), so nothing was rebuilt mid-run.
**Harness:** all of it lives in `target/p20-18-suite/`. No repo file was edited, `.venv` was not modified, and nothing was staged or committed.

## Headline

| run | passed / collected | ported families only (unported `adc`, `grad`, `gw`, `tdscf`, `x2c` removed) |
|---|---:|---:|
| **overlay** (`pyscf` = `python/pyscf`, fallthrough = vendored 2.12.1) | **1 / 815** (0.12 %) | **1 / 660** (0.15 %) |
| **control** (pure vendored 2.12.1, no overlay) | 808 / 815 (99.1 %) | 659 / 660 |

**The ≥ 80 % target is NOT met: 1 / 815.**
- No test was excluded to reach that number. The denominator is the 815 tests the repo's `pytest.ini` selects, measured on the control.
- The overlay produced **no crash, no timeout and no numerical mismatch**. Only one test reached a Rust kernel: `dft/test/test_sampling.py::KnowValues::test_kpt_vs_supercell`, which passed.
- The other 814 tests stopped before any computation:
  - **746** at import (collection);
  - **68** in `setUp`/test bodies, while building the `Cell`.

## Denominator and exclusions (stated, not applied silently)

| | tests |
|---|---:|
| collected, no exclusions (`-o addopts=`; 113 files in 16 `test/` dirs) | 920 |
| − `pytest.ini` `-k "not _high_cost and not _skip"` deselection | −105 |
| − `pytest.ini` `--ignore-glob="*pbc/cc/test/*test_h_*.py"` (2 files) | −0 |
| **denominator** | **815** (111 files) |

- **The two ignored files.** `cc/test/test_h_2x2x1.py` and `test_h_3x3x1.py` are scripts with no test functions (`grep -c "def test"` = 0).
  - Collecting `test_h_2x2x1.py` found 0 tests, but took 38 s because the module body runs a calculation.
  - Collecting `test_h_3x3x1.py` was killed after 68 s at 3 GB RSS, for the same reason.
- **What the deselection hits.** 83 test names contain `_high_cost` and 2 contain `_skip`. The other 20 match the keyword through a parent class or module name, for example the `*_vs_spglib`, `test_t3p2_imds_*` and `test_mcol_*` tests.
  - The full id list is in `target/p20-18-suite/logs/deselected-ids.txt`.
- **Globs that match nothing here.** The remaining `pytest.ini` globs (`*_slow*`, `*test_kproxy*`, `*test_proxy*`, `*test_bz*`, `*test_ks_noimport*`) match no file in the 2.12.1 `pbc` tree.
- **Families with no test directory.** There is none for `eph`, `geomopt`, `mpicc`, `mpitools` or `tddft`, so they contribute 0 tests.

Deselected tests per family: adc 11, cc 21, ci 2, df 25, dft 14, grad 8, gw 2, mp 7, scf 6, symm 5, tdscf 1, x2c 3, and 0 for gto, lib and tools.

## Per-family results

Column meanings:
- **collected** is the control-measured, post-exclusion count.
- **coll. err** counts every test in a file whose import failed at collection.
- **errors** means errors in `setUp`/`setUpModule`.
- Skipped, xfail, timeouts and not-run were **0** in both runs, so those columns are omitted.

| family | files | collected | overlay passed | overlay failed | overlay errors | overlay coll. err | control passed | control failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `adc` (unported) | 8 | 14 | 0 | 0 | 0 | 14 | 14 | 0 |
| `cc` | 13 | 63 | 0 | 1 | 2 | 60 | 62 | 1 |
| `ci` | 2 | 5 | 0 | 0 | 0 | 5 | 5 | 0 |
| `df` | 22 | 226 | 0 | 0 | 0 | 226 | 226 | 0 |
| `dft` | 15 | 125 | **1** | 0 | 28 | 96 | 125 | 0 |
| `grad` (unported) | 10 | 57 | 0 | 0 | 0 | 57 | 51 | 6 |
| `gto` (incl. `gto/pseudo`) | 4 | 46 | 0 | 0 | 0 | 46 | 46 | 0 |
| `gw` (unported) | 2 | 2 | 0 | 0 | 0 | 2 | 2 | 0 |
| `lib` | 2 | 10 | 0 | 0 | 0 | 10 | 10 | 0 |
| `mp` | 8 | 28 | 0 | 0 | 3 | 25 | 28 | 0 |
| `scf` | 12 | 129 | 0 | 8 | 22 | 99 | 129 | 0 |
| `symm` | 2 | 9 | 0 | 0 | 0 | 9 | 9 | 0 |
| `tdscf` (unported) | 8 | 77 | 0 | 0 | 0 | 77 | 77 | 0 |
| `tools` | 2 | 19 | 0 | 0 | 4 | 15 | 19 | 0 |
| `x2c` (unported) | 1 | 5 | 0 | 0 | 0 | 5 | 5 | 0 |
| **total** | **111** | **815** | **1** | 9 | 59 | 746 | **808** | 7 |

**Wall time and memory.**
- **Overlay:** 47 s for the whole suite (3 files in parallel, 3 threads each). Max RSS 385 MB.
- **Control:** 5 min 43 s (4 files in parallel, 3 threads each). Max RSS 3.0 GB (`cc/test_rccsd_t_shift.py`).
- No family came near the 2 h budget. Every selected test ran.

**Exit codes.**
- Overlay: rc 2 (collection error) for 95 files, rc 1 for 13, rc 5 (all deselected) for 2, rc 0 for 1.
- Control: rc 0 for 103, rc 1 for 4, rc 5 for 4.
- Neither run had a signal death. Every file wrote its junit XML.

## Failure taxonomy (overlay; 814 non-passing tests)

| class | tests | what |
|---|---:|---|
| (1) unported family → expected | **155** | `adc` 14, `grad` 57, `gw` 2, `tdscf` 77, `x2c` 5 — all fail at collection |
| (2) numerical mismatch | **0** | nothing but `test_sampling` reached a kernel |
| (3) crash / segfault / abort | **0** | no signal death, no timeout (`--timeout=600` per test) |
| (4a) import resolution inside ported families — **caused by the overlay itself** | **591** | see table below; identical imports succeed in the control |
| (4b) harness / environment (2.14.0 leak, missing optional deps) | **0** after control | 0 site-packages-2.14.0 modules loaded in any of 111 processes; no optional-dependency skip in either run. The repo `pytest.ini`'s own import mode **is** a harness defect (§Harness finding) — neutralised, not counted |
| (5) API-surface gap in a bound class (not in the brief's four classes; not numerical) | **68** | native `pyscf.pbc.gto.Cell` rejects upstream input forms — see table below |

### (1) Unported families: first import error

These fail at import, so the missing Rust bindings are never even reached.

| tests | error | files |
|---:|---|---|
| 92 | `No module named 'pyscf.gto.moleintor'` | `adc` ×7 files, `gw` ×2, `tdscf` ×8 |
| 32 | `cannot import name 'intor_cross' from 'pyscf.gto'` | `grad/test_krks_stress`, `grad/test_rks_stress` |
| 11 | `cannot import name 'ATOM_OF' from 'pyscf.gto'` | `grad/test_kuks_stress`, `grad/test_uks_stress` |
| 9 | `cannot import name 'rhf' from 'pyscf.grad'` | `grad/test_krhf`, `test_krks`, `test_kuhf`, `test_kuks` |
| 5 | `No module named 'pyscf.gto.basis'` | `grad/test_krkspu`, `grad/test_kukspu` |
| 5 | `cannot import name 'mole' from 'pyscf.gto'` | `x2c/test_x2c` |
| 1 | `cannot import name 'gen_uniform_grids' from 'pyscf.pbc.dft.gen_grid'` | `adc/test_kadc` |

### (4a) Import failures in ported families

Route: **D-PBC-35 / 20-17 D6** (molecular overlay packages), plus one 20-17 shim gap.

The molecular overlay packages `python/pyscf/{gto,scf,dft,cc,mp}/__init__.py` shadow upstream without `extend_path` and without the names the upstream PBC code imports. This matches `overlay-passthrough.md` exactly. It is a product defect, not an artefact of this harness: the control imports every one of these files.

| tests | missing name | route | files |
|---:|---|---|---|
| 204 | `pyscf.gto.basis` | D-PBC-35 | `df/{test_aft,aft_jk,df_ao2mo,df_jk,fft,gdf_builder,incore,mdf_ao2mo,mdf_jk,outcore,rsdf_ao2mo,rsdf_builder,rsdf_jk}`, `dft/{test_krkspu,kukspu}`, `gto/pseudo/test_pp_velgauge`, `lib/{test_kpts,kpts_ksymm}`, `scf/{test_addons,khf,rohf,scfint,uhf}`, `tools/test_pbc` |
| 86 | `pyscf.dft.radi` | D-PBC-35 | `df/test_band`, `dft/{test_kgks,krks_ksym}`, `gto/test_cell`, `scf/test_khf_ksym` |
| 81 | `pyscf.gto.moleintor` | D-PBC-35 | `df/{test_aft_ao2mo,df,mdf,mdf_builder,rsdf,rsdf_1,rsdf_scf}`, `mp/test_kpoint_stagger` |
| 51 | `pyscf.gto.ATOM_OF` | D-PBC-35 | `dft/{test_multigrid,multigrid2}` |
| 44 | `gen_uniform_grids` from the **20-17 shim** `python/pyscf/pbc/dft/gen_grid.py` | 20-17 shim (`__all__`/fallthrough lacks it) | `cc/{test_eom_kgccsd,eom_kgccsd_diag,eom_krccsd,eom_kuccsd,eom_kuccsd_diag,krccsd,krccsd_gamma}`, `ci/test_ci` |
| 24 | `pyscf.mp.mp2` | D-PBC-35 | `mp/{test_kpoint,mask,mask_uhf,padding,scs}` |
| 24 | `pyscf.scf.atom_hf` | D-PBC-35 | `scf/test_hf` |
| 17 | `pyscf.gto.mole` | D-PBC-35 | `cc/test_kuccsd`, `symm/{test_basis,spg}` |
| 11 | `pyscf.cc.gccsd` | D-PBC-35 | `cc/test_kgccsd` |
| 11 | `pyscf.gto.ft_ao` | D-PBC-35 | `df/test_ft_ao` |
| 11 | `pyscf.dft.rks` | D-PBC-35 | `dft/test_numint` |
| 9 | `pyscf.dft.numint` | D-PBC-35 | `dft/test_gen_grid`, `gto/pseudo/test_pp` |
| 8 | `pyscf.scf._vhf` | D-PBC-35 | `scf/test_rsjk` |
| 5 | `pyscf.scf.chkfile` | D-PBC-35 | `scf/test_stability` |
| 3 | `pyscf.gto.moleintor` (as a name) | D-PBC-35 | `gto/test_rcut` |
| 1 | `pyscf.cc.ccsd_t` | D-PBC-35 | `cc/test_rccsd_t_shift` |
| 1 | `pyscf.cc.ccsd` | D-PBC-35 | `ci/test_cisd` (gamma CISD is also refused by design) |

### (5) Native `Cell` input surface

Route: **20-09** (gto binding; the last two rows are 20-09 D4). The first blocker is listed; later blockers such as the unbound `.newton()` in `test_newton` are masked by it.

The input forms were confirmed by a direct probe in the overlay environment:
- accepted: `'C 0 0 0; C …'`, a comma-and-newline atom string, a newline atom string;
- rejected: list-form entries `[['C',[0,0,0]]]` and `cell.output=`.

| tests | error | node ids |
|---:|---|---|
| 31 | `AttributeError: 'pyscf._native.pbc.gto.Cell' object has no attribute 'output' and no __dict__` | `cc/test/test_kccsd_ksymm.py::KnownValues::{test_krccsd_ksym,test_vs_krccsd}`; `mp/test/test_dm.py::KnownValues::test_kmp2_contract_eri_dm`; `scf/test/test_mulliken_meta.py::KPTvsSUPCELL_{noshift,shift}::{test_kghf,test_krhf,test_krohf,test_kuhf}` (8); `scf/test/test_newton.py::KnowValues::*` (all 20) |
| 30 | `TypeError: each cell.atom entry must be (symbol, (x, y, z))`: upstream writes entries as **lists** `[['He', (x,y,z)]]` | `dft/test/test_krks.py::KnownValues::*` (all 10); `dft/test/test_kuks.py::KnownValues::{test_klda,test_kuks_as_kuhf,test_rsh_df,test_rsh_fft,test_to_hf}`; `dft/test/test_rks.py::KnownValues::{test_chkfile_k_point,test_custom_rsh_df,test_density_fit,test_density_fit_2d,test_rsh_0d,test_rsh_0d_df,test_rsh_fft,test_rsh_mdf}`; `dft/test/test_uks.py::KnownValues::{test_pp_UKS,test_rsh_df,test_rsh_fft}`; `mp/test/test_ksym.py::KnownValues::{test_kmp2,test_rdm1}`; `scf/test/test_band.py::KnownValues::{test_band,test_band_kscf}` |
| 4 | `NotImplementedError: cell.pseudo: only a single pseudopotential NAME … is bound (plan 20-09)`: a per-element dict `{'Li': 'GTH-PBE-q3'}` | `tools/test/test_k2gamma.py::KnownValues::{test_double_translation_indices,test_k2gamma,test_k2gamma_ksymm,test_kpts_to_kmesh}` |
| 3 | `TypeError: cell.basis must be a name or a {element: name} dict`: an explicit shell list | `cc/test/test_kuccsd_openshell.py::test_kuccsd_openshell::test_kuccsd_openshell`; `dft/test/test_gks.py::KnownValues::{test_collinear_gks_gga,test_ncol_x2c_gks_lda}` |

**The one pass.** `pyscf/pbc/dft/test/test_sampling.py::KnowValues::test_kpt_vs_supercell` took 28 s. It uses native `pbcgto.Cell`, `pbcdft.RKS`/`KRKS` and `tools.super_cell`.

## Control: failures inherent to the vendored tree (not overlay-attributable)

These 7 fail identically when the control is re-run with the **2.12.1-wheel** C libraries instead of the 2.14.0 ones, digit for digit (`logs/rerun-control-libs2121/`). They are properties of the vendored upstream tree. They count against both runs' denominators.

| node id | measured |
|---|---|
| `pyscf/pbc/cc/test/test_krccsd.py::KnownValues::test_frozen_n3` | −8.648501503147841 vs −8.648503065380389 (Δ 1.56e-6, `places=6`) |
| `pyscf/pbc/grad/test/test_krks_stress.py::KnownValues::test_get_vxc_gga` | 1.34e-9 ≮ 1e-9 |
| `pyscf/pbc/grad/test/test_krks_stress.py::KnownValues::test_get_vxc_lda` | 1.10e-9 ≮ 1e-9 |
| `pyscf/pbc/grad/test/test_kuks_stress.py::KnownValues::test_get_vxc_gga` | 2.10e-8 ≮ 1e-8 |
| `pyscf/pbc/grad/test/test_rks_stress.py::KnownValues::test_get_vxc_gga` | 1.08e-9 ≮ 1e-9 |
| `pyscf/pbc/grad/test/test_rks_stress.py::KnownValues::test_get_vxc_lda` | 1.06e-9 ≮ 1e-9 |
| `pyscf/pbc/grad/test/test_rks_stress.py::KnownValues::test_get_vxc_mgga` | 1.46e-9 ≮ 1e-9 |

## Import environment: how it was established and proved

**Plugin.** `target/p20-18-suite/plugin/p2018env.py` is loaded with `-p p2018env`. It does four things:
1. It wraps `pkgutil.extend_path` for `pyscf*` packages and drops every `.venv/…/site-packages/pyscf` (2.14.0) entry. In control mode it also drops `python/pyscf`.
2. It appends a **C-library-only** directory `clibs-<ver>/pyscf`, which holds just `lib/*.so` and `lib/deps`, to `pyscf.__path__`.
   - The vendored `pyscf/lib/` has **no compiled libraries**. Vendored `load_library` falls back to `pyscf.__path__` entries.
   - Without this, the vendored tree was silently using the 2.14.0 site-packages `.so` **and** 2.14.0 Python on its `__path__`. That is also how today's repo-root oracle subprocesses run.
3. In `pytest_configure` it asserts where `pyscf.__file__` resolved.
4. At exit it writes a JSON file per process with `pyscf.__file__`, `pyscf.__path__`, `pyscf.pbc.__path__`, per-origin module counts, any site-packages-2.14.0 module (`site_2_14_0_leaks`) and the `.so` mtime.

**Proof.**

| run | processes | `pyscf.__file__` | `pyscf.pbc.__path__` | modules loaded by origin | 2.14.0 leaks |
|---|---:|---|---|---|---:|
| overlay | 111 | `python/pyscf/__init__.py` (all 111) | `[python/pyscf/pbc, pyscf/pbc]` | overlay 862, vendored 1324 | **0** |
| control | 111 | `pyscf/__init__.py`, `__version__` 2.12.1 (all 111) | `[pyscf/pbc]` | vendored only | **0** |

**The C libraries are not a variable.**
- The control used the 2.14.0 `.so` set, which is what the existing oracles load de facto.
- The 7 control failures reproduce exactly with the `pyscf-2.12.1` wheel `.so` set (uv cache `archive-v0/W6z-CtJllEiNreUa`).
- Note that the vendored Python tree is **not** byte-identical to that 2.12.1 wheel: 146 differing or extra paths, e.g. `pbc/scf/smearing.py` exists only in the vendored tree. It looks like a post-2.12.1 snapshot labelled 2.12.1.

### Harness finding: the repo `pytest.ini` bypasses the overlay for these files

`pytest.ini` sets `--import-mode=importlib`. The upstream test directories have no `__init__.py`, but `pyscf/` and `pyscf/pbc/` do. So pytest names each module `pyscf.pbc.<fam>.test.<file>` and **imports its parent packages from the file system**, which means the *vendored* `pyscf/pbc/__init__.py`, even when `PYTHONPATH` puts the overlay first.

Measured with the same plugin in importlib mode:
- **0 tests collected, 111/111 files collection errors**;
- `pyscf.pbc.__path__` = `[pyscf/pbc, python/pyscf/pbc]`, i.e. vendored first, a mixed state.

All numbers above therefore use `--import-mode=prepend`, given after `-c pytest.ini` so it overrides addopts. Everything else in `pytest.ini` (`-k`, ignore globs) stays in force. **A naive `pytest pyscf/pbc` run with the repo configuration measures neither the overlay nor upstream.**

## Exact commands

**Environment.** `target/p20-18-suite/env.sh MODE [LIBVER]`:
```bash
REPO=/home/user/Documents/workspace/pyscf_rs; S=$REPO/target/p20-18-suite
export P2018_MODE=$MODE P2018_CLIBS=$S/clibs-${LIBVER:-2.14.0}
# overlay:  PYTHONPATH=$REPO/python:$REPO:$S/plugin:$S/pydeps
# control:  PYTHONPATH=$REPO:$S/plugin:$S/pydeps
export OMP_NUM_THREADS=3 RAYON_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
unset PYSCF_EXT_PATH VIRTUAL_ENV
```

**Other setup.**
- `pydeps/` holds only `pytest-timeout 2.4.0`, installed with `uv pip install --python $REPO/.venv/bin/python --target pydeps pytest-timeout`.
- `clibs-2.14.0/pyscf/lib` is a copy of `.venv/…/site-packages/pyscf/lib/{*.so*,deps}`.
- `clibs-2.12.1/pyscf/lib` is the same set from the uv-cached 2.12.1 wheel.

**Per file.** Each file runs in its own process (`run_one.sh MODE LIBVER TAG FILE …`), with cwd a scratch dir:
```bash
/usr/bin/time -v timeout -k 60 7200 $REPO/.venv/bin/pytest -c $REPO/pytest.ini --rootdir $REPO \
  -p p2018env -p no:cacheprovider --import-mode=prepend -v -rA --junitxml=logs/$TAG/$ID.xml \
  --timeout=600 $REPO/$FILE
```

**Full runs.** Each run is detached and memory-capped. The file list is the 111 `test_*.py` under `pyscf/pbc/**/test/`, excluding `cc/test/test_h_*`.
```bash
systemd-run --user --scope -p MemoryMax=11G ./run_all.sh control 2.14.0 run-control 4 3
systemd-run --user --scope -p MemoryMax=11G ./run_all.sh overlay 2.14.0 run-overlay 3 3
```

**Collection passes.** Run through `collect_all.sh` with `--collect-only -q`:
- `collect-control`
- `collect-control-noexcl`, which adds `-o addopts=`
- `collect-overlay`
- `collect-overlay-importlib`, with `IMPORT_MODE=--import-mode=importlib`

**Deselected ids.**
```bash
… control … --collect-only -q -q -o addopts= -k "_high_cost or _skip"
```

**Control libraries re-run.** `run_one.sh control 2.12.1 rerun-control-libs2121 <the 4 failing files> --timeout=600`.

**Aggregation.** `python3 aggregate.py {collect|run} TAG` writes `logs/{collect,run}-TAG.json`, and `python3 report.py` builds the tables. All raw logs, junit XML, `/usr/bin/time` output and per-process env JSON are under `target/p20-18-suite/logs/`.
