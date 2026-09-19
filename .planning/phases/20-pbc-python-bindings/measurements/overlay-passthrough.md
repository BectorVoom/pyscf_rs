# Overlay passthrough: which upstream `pyscf.pbc.*` modules import under the overlay

**Measured:** 2026-09-14 by plan 20-17, `.so` 2026-09-14 16:58:45 (20-15 build 2).
**Tool:** `measurements/overlay_passthrough_probe.py` — every module of the
vendored 2.12.1 tree (`pyscf/pbc/**/*.py`, `test/` excluded: **195** modules)
is imported with `importlib.import_module` in a **fresh interpreter**, cwd a
scratch directory, overlay `python/` first on `sys.path`. Raw rows:
`overlay-passthrough-before.json` (tree before any 20-17 edit) and
`overlay-passthrough-after.json`.

| config | `PYTHONPATH` | passthrough target (the next `pyscf` on the path) |
|---|---|---|
| `site` | `<repo>/python` | `.venv` site-packages **pyscf 2.14.0** — what pytest sees (rootdir insertion puts `python/` first) |
| `vendored` | `<repo>/python:<repo>` | the vendored **2.12.1** tree |

**The two configs give identical ok/error results for all 195 modules**, before
and after. The failures are in the molecular overlay packages the upstream code
imports, not in which upstream copy is found.

## Headline

* **Before 20-17: 31 / 195 import** — 12 of them overlay packages or the two
  20-12 stubs, so only **19 upstream modules** worked at all (0 warnings).
* **After 20-17: 41 / 195 import** — 18 overlay/stub, **23 upstream**. Every
  change is an error→ok or upstream→overlay transition; **no module that
  imported before fails now** (diff of the two JSON files, both configs).
* **Silent passthrough is gone:** every successful import that resolved outside
  the overlay emitted a `PbcUpstreamFallthroughWarning` (0 silent). 177 / 195
  probes emitted one (a failing import warns before it fails).
* **The passthrough is BROKEN for 87 % of the modules it serves**: 154 of the 177 modules the overlay does not shadow fail to import.
  "Announced passthrough" (D-PBC-35) therefore announces a fallthrough that in
  most families raises `ImportError` next. Root cause: the MOLECULAR overlay
  packages (`python/pyscf/{gto,scf,dft,cc,mp,grad}/__init__.py`) shadow
  upstream's without `extend_path` and without the names upstream imports
  (`pyscf.gto.basis`, `moleintor`, `mole`, `ATOM_OF`, `pyscf.grad.rhf`,
  `pyscf.mp.mp2`, `pyscf.cc.rccsd`, `pyscf.dft.radi`, …). Fixing that is a
  molecular-overlay change, outside 20-17.

## Environment facts found while measuring (they matter to 20-18)

1. **`.venv/bin/python` WITHOUT `PYTHONPATH` does not use the overlay at all.**
   `pyscf_rs.pth` appends `python/` AFTER site-packages, and site-packages holds
   pyscf **2.14.0**, so `import pyscf` → `site-packages/pyscf/__init__.py` and
   `pyscf.pbc.scf` → site-packages upstream. `examples/pbc/*.py` must be run as
   `PYTHONPATH=$REPO/python .venv/bin/python examples/pbc/<x>.py`.
2. **From the repo root, `python -c` imports the VENDORED upstream `pyscf/`**
   (cwd `''` is first), so the plan's
   `python -c "from pyscf.pbc import scf; …; assert scf.KRHF is n.pbc.scf.KRHF"`
   fails there. It exits 0 from any other cwd with `PYTHONPATH=$REPO/python`.
3. With the overlay active, the passthrough target is whatever `pyscf` follows
   `python/` on `sys.path` — **2.14.0 under pytest**, not the 2.12.1 oracle.
   The announcement names the file it resolved to.

## Per family

| family | modules | import ok BEFORE 20-17 | import ok AFTER 20-17 | of which overlay | upstream modules that import |
|---|---:|---:|---:|---:|---|
| (pyscf.pbc) | 1 | 1 | 1 | 1 | — |
| __all__ | 1 | 0 | 0 | 0 | — |
| adc | 7 | 0 | 0 | 0 | — |
| ao2mo | 2 | 1 | 1 | 1 | — |
| cc | 19 | 4 | 4 | 1 | `kintermediates_rhf_ksymm`, `kintermediates_uhf`, `kuccsd_rdm` |
| ci | 3 | 1 | 1 | 1 | — |
| df | 21 | 2 | 4 | 1 | `df_jk`, `fft_jk`, `rsdf_jk` |
| dft | 25 | 1 | 2 | 2 | — |
| eph | 2 | 1 | 1 | 0 | `pyscf.pbc.eph` |
| geomopt | 2 | 0 | 0 | 0 | — |
| grad | 15 | 0 | 0 | 0 | — |
| gto | 16 | 3 | 3 | 1 | `_pbcintor`, `neighborlist` |
| gw | 7 | 0 | 0 | 0 | — |
| lib | 7 | 6 | 7 | 3 | `arnoldi`, `chkfile`, `ktensor`, `linalg_helper` |
| mp | 6 | 1 | 1 | 1 | — |
| mpicc | 4 | 0 | 0 | 0 | — |
| mpitools | 6 | 1 | 1 | 0 | `pyscf.pbc.mpitools` |
| scf | 21 | 6 | 6 | 3 | `_response_functions`, `cphf`, `scfint` |
| symm | 8 | 2 | 2 | 1 | `tables` |
| tddft | 1 | 0 | 0 | 0 | — |
| tdscf | 9 | 0 | 0 | 0 | — |
| tools | 9 | 0 | 6 | 2 | `k2gamma`, `lattice`, `print_funcs`, `tril` |
| x2c | 3 | 1 | 1 | 0 | `pyscf.pbc.x2c` |
| **total** | **195** | **31** | **41** | | |

### Failure causes (AFTER, site config; the vendored config is identical)

| count | first error line |
|---:|---|
| 39 | `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |
| 35 | `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |
| 15 | `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |
| 9 | `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |
| 9 | `ImportError: cannot import name 'mole' from 'pyscf.gto'` |
| 6 | `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` |
| 5 | `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` |
| 5 | `ModuleNotFoundError: No module named 'mpi4py'` |
| 3 | `ModuleNotFoundError: No module named 'pyscf.gto.ft_ao'` |
| 3 | `ImportError: cannot import name 'numint' from 'pyscf.dft'` |
| 2 | `ImportError: cannot import name 'gen_uniform_grids' from 'pyscf.pbc.dft.gen_grid'` |
| 2 | `ImportError: cannot import name 'eom_rccsd' from 'pyscf.cc'` |
| 2 | `ImportError: cannot import name 'gccsd' from 'pyscf.cc'` |
| 2 | `ImportError: cannot import name 'rohf' from 'pyscf.scf'` |
| 2 | `ImportError: cannot import name '_vhf' from 'pyscf.scf'` |
| 2 | `ModuleNotFoundError: No module named 'pyscf.geomopt.addons'` |
| 2 | `ImportError: cannot import name 'radi' from 'pyscf.dft'` |
| 2 | `ImportError: cannot import name 'addons' from 'pyscf.scf'` |
| 2 | `ImportError: cannot import name 'chkfile' from 'pyscf.scf'` |
| 1 | `ImportError: cannot import name 'MORotationMatrix' from 'pyscf.pbc.lib.kpts'` |
| 1 | `ImportError: cannot import name 'uccsd' from 'pyscf.cc'` |
| 1 | `ImportError: cannot import name 'ccsd' from 'pyscf.cc'` |
| 1 | `ImportError: cannot import name 'moleintor' from 'pyscf.gto'` |
| 1 | `ModuleNotFoundError: No module named 'pyscf.scf.chkfile'` |
| 1 | `ModuleNotFoundError: No module named 'spglib'` |
| 1 | `RuntimeError: ASE is not found` |

### Per-module (AFTER)

| module | site (2.14.0 passthrough) | vendored (2.12.1 passthrough) | warned family |
|---|---|---|---|
| `pyscf.pbc` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.__all__` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.adc` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.adc.dfadc` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.adc.kadc_ao2mo` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.adc.kadc_rhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.adc.kadc_rhf_amplitudes` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.adc.kadc_rhf_ea` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.adc.kadc_rhf_ip` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.ao2mo` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.ao2mo.eris` | **error** — `ImportError: cannot import name 'gen_uniform_grids' from 'pyscf.pbc.dft.gen_grid'` | **error** — `ImportError: cannot import name 'gen_uniform_grids' from 'pyscf.pbc.dft.gen_grid'` |  |
| `pyscf.pbc.cc` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.cc.ccsd` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.cc.eom_kccsd_ghf` | **error** — `ImportError: cannot import name 'eom_rccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'eom_rccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.cc.eom_kccsd_rhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.cc.eom_kccsd_rhf_ea` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |  |
| `pyscf.pbc.cc.eom_kccsd_rhf_ip` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |  |
| `pyscf.pbc.cc.eom_kccsd_uhf` | **error** — `ImportError: cannot import name 'eom_rccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'eom_rccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.cc.kccsd` | **error** — `ImportError: cannot import name 'gccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'gccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.cc.kccsd_rhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.cc.kccsd_rhf_ksymm` | **error** — `ImportError: cannot import name 'MORotationMatrix' from 'pyscf.pbc.lib.kpts'` | **error** — `ImportError: cannot import name 'MORotationMatrix' from 'pyscf.pbc.lib.kpts'` |  |
| `pyscf.pbc.cc.kccsd_t` | **error** — `ImportError: cannot import name 'gccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'gccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.cc.kccsd_t_rhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.cc.kccsd_t_rhf_slow` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.cc.kccsd_uhf` | **error** — `ImportError: cannot import name 'uccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'uccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.cc.kintermediates` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |  |
| `pyscf.pbc.cc.kintermediates_rhf` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |  |
| `pyscf.pbc.cc.kintermediates_rhf_ksymm` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.cc.kintermediates_uhf` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.cc.kuccsd_rdm` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.ci` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.ci.cisd` | **error** — `ImportError: cannot import name 'ccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'ccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.ci.kcis_rhf` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |  |
| `pyscf.pbc.df` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.df.aft` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.df.aft_ao2mo` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.df.aft_jk` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.ft_ao'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.ft_ao'` |  |
| `pyscf.pbc.df.df` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.df.df_ao2mo` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.df.df_jk` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.df.fft` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.df.fft_ao2mo` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.df.fft_jk` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.df.ft_ao` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.ft_ao'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.ft_ao'` |  |
| `pyscf.pbc.df.gdf_builder` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.df.incore` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.df.mdf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.df.mdf_ao2mo` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.df.mdf_jk` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.ft_ao'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.ft_ao'` |  |
| `pyscf.pbc.df.outcore` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.df.rsdf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.df.rsdf_builder` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.df.rsdf_helper` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.df.rsdf_jk` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.dft` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.dft.cdft` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` |  |
| `pyscf.pbc.dft.gen_grid` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.dft.gks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.kgks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.krks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.krks_ksymm` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.krkspu` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.krkspu_ksymm` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.kroks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.kuks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.kuks_ksymm` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.kukspu` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.kukspu_ksymm` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.multigrid` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` |  |
| `pyscf.pbc.dft.multigrid._backend_c` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` |  |
| `pyscf.pbc.dft.multigrid.multigrid` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` |  |
| `pyscf.pbc.dft.multigrid.multigrid_pair` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` |  |
| `pyscf.pbc.dft.multigrid.pp` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` |  |
| `pyscf.pbc.dft.multigrid.utils` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` |  |
| `pyscf.pbc.dft.numint` | **error** — `ImportError: cannot import name 'numint' from 'pyscf.dft'` | **error** — `ImportError: cannot import name 'numint' from 'pyscf.dft'` |  |
| `pyscf.pbc.dft.numint2c` | **error** — `ImportError: cannot import name 'numint' from 'pyscf.dft'` | **error** — `ImportError: cannot import name 'numint' from 'pyscf.dft'` |  |
| `pyscf.pbc.dft.rks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.dft.roks` | **error** — `ImportError: cannot import name 'rohf' from 'pyscf.scf'` | **error** — `ImportError: cannot import name 'rohf' from 'pyscf.scf'` |  |
| `pyscf.pbc.dft.uks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.eph` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.eph.eph_fd` | **error** — `ImportError: cannot import name '_vhf' from 'pyscf.scf'` | **error** — `ImportError: cannot import name '_vhf' from 'pyscf.scf'` |  |
| `pyscf.pbc.geomopt` | **error** — `ModuleNotFoundError: No module named 'pyscf.geomopt.addons'` | **error** — `ModuleNotFoundError: No module named 'pyscf.geomopt.addons'` |  |
| `pyscf.pbc.geomopt.geometric_solver` | **error** — `ModuleNotFoundError: No module named 'pyscf.geomopt.addons'` | **error** — `ModuleNotFoundError: No module named 'pyscf.geomopt.addons'` |  |
| `pyscf.pbc.grad` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.krhf` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.krks` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.krks_stress` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.krkspu` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.kuhf` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.kuks` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.kuks_stress` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.kukspu` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.rhf` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.rks` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.rks_stress` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.uhf` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.uks` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.grad.uks_stress` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` | **error** — `ImportError: cannot import name 'rhf' from 'pyscf.grad'` |  |
| `pyscf.pbc.gto` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.gto._pbcintor` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.gto.basis` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.gto.basis.split_BASIS_MOLOPT` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.gto.basis.split_GTH_BASIS_SETS` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.gto.basis.split_HFX_BASIS` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.gto.cell` | **error** — `ImportError: cannot import name 'radi' from 'pyscf.dft'` | **error** — `ImportError: cannot import name 'radi' from 'pyscf.dft'` |  |
| `pyscf.pbc.gto.ecp` | **error** — `ImportError: cannot import name 'radi' from 'pyscf.dft'` | **error** — `ImportError: cannot import name 'radi' from 'pyscf.dft'` |  |
| `pyscf.pbc.gto.eval_gto` | **error** — `ImportError: cannot import name 'moleintor' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'moleintor' from 'pyscf.gto'` |  |
| `pyscf.pbc.gto.ewald_methods` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` |  |
| `pyscf.pbc.gto.neighborlist` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.gto.pseudo` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.gto.pseudo.pp` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.gto.pseudo.pp_int` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.gto.pseudo.ppnl_velgauge` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.gto.pseudo.split_GTH_POTENTIALS` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.gw` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.gw.gw_slow` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.gw.kgw_slow` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.gw.kgw_slow_supercell` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.gw.krgw_ac` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.gw.krgw_cd` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.gw.kugw_ac` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.lib` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.lib.arnoldi` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.lib.chkfile` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.lib.kpts` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.lib.kpts_helper` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.lib.ktensor` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.lib.linalg_helper` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.mp` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.mp.kmp2` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |  |
| `pyscf.pbc.mp.kmp2_ksymm` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |  |
| `pyscf.pbc.mp.kmp2_stagger` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.mp.kump2` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |  |
| `pyscf.pbc.mp.mp2` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` | **error** — `ImportError: cannot import name 'mp2' from 'pyscf.mp'` |  |
| `pyscf.pbc.mpicc` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.mpicc.kccsd_rhf` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.mpicc.kintermediates_rhf` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.mpicc.mpi_kpoint_helper` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` | **error** — `ImportError: cannot import name 'rccsd' from 'pyscf.cc'` |  |
| `pyscf.pbc.mpitools` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.mpitools.mpi` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` |  |
| `pyscf.pbc.mpitools.mpi_blksize` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` |  |
| `pyscf.pbc.mpitools.mpi_helper` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` |  |
| `pyscf.pbc.mpitools.mpi_load_balancer` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` |  |
| `pyscf.pbc.mpitools.mpi_pool` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` | **error** — `ModuleNotFoundError: No module named 'mpi4py'` |  |
| `pyscf.pbc.scf` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.scf._response_functions` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.scf.addons` | **error** — `ImportError: cannot import name 'addons' from 'pyscf.scf'` | **error** — `ImportError: cannot import name 'addons' from 'pyscf.scf'` |  |
| `pyscf.pbc.scf.chkfile` | **error** — `ModuleNotFoundError: No module named 'pyscf.scf.chkfile'` | **error** — `ModuleNotFoundError: No module named 'pyscf.scf.chkfile'` |  |
| `pyscf.pbc.scf.cphf` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.scf.ghf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.scf.hf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.scf.kghf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.scf.kghf_ksymm` | ok — no-file (stub or native) | ok — no-file (stub or native) |  |
| `pyscf.pbc.scf.khf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.scf.khf_ksymm` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.scf.krohf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.scf.kuhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.scf.kuhf_ksymm` | ok — no-file (stub or native) | ok — no-file (stub or native) |  |
| `pyscf.pbc.scf.newton_ah` | **error** — `ImportError: cannot import name 'chkfile' from 'pyscf.scf'` | **error** — `ImportError: cannot import name 'chkfile' from 'pyscf.scf'` |  |
| `pyscf.pbc.scf.rohf` | **error** — `ImportError: cannot import name 'rohf' from 'pyscf.scf'` | **error** — `ImportError: cannot import name 'rohf' from 'pyscf.scf'` |  |
| `pyscf.pbc.scf.rsjk` | **error** — `ImportError: cannot import name '_vhf' from 'pyscf.scf'` | **error** — `ImportError: cannot import name '_vhf' from 'pyscf.scf'` |  |
| `pyscf.pbc.scf.scfint` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.scf.smearing` | **error** — `ImportError: cannot import name 'addons' from 'pyscf.scf'` | **error** — `ImportError: cannot import name 'addons' from 'pyscf.scf'` |  |
| `pyscf.pbc.scf.stability` | **error** — `ImportError: cannot import name 'chkfile' from 'pyscf.scf'` | **error** — `ImportError: cannot import name 'chkfile' from 'pyscf.scf'` |  |
| `pyscf.pbc.scf.uhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.basis'` |  |
| `pyscf.pbc.symm` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.symm.basis` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` |  |
| `pyscf.pbc.symm.geom` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` |  |
| `pyscf.pbc.symm.group` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` |  |
| `pyscf.pbc.symm.pyscf_spglib` | **error** — `ModuleNotFoundError: No module named 'spglib'` | **error** — `ModuleNotFoundError: No module named 'spglib'` |  |
| `pyscf.pbc.symm.space_group` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` |  |
| `pyscf.pbc.symm.symmetry` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` |  |
| `pyscf.pbc.symm.tables` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.tddft` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tdscf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tdscf.krhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tdscf.krks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tdscf.kuhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tdscf.kuks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tdscf.rhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tdscf.rks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tdscf.uhf` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tdscf.uks` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` | **error** — `ModuleNotFoundError: No module named 'pyscf.gto.moleintor'` |  |
| `pyscf.pbc.tools` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.tools.k2gamma` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.tools.lattice` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.tools.make_test_cell` | **error** — `ImportError: cannot import name 'gen_uniform_grids' from 'pyscf.pbc.dft.gen_grid'` | **error** — `ImportError: cannot import name 'gen_uniform_grids' from 'pyscf.pbc.dft.gen_grid'` |  |
| `pyscf.pbc.tools.pbc` | ok — overlay | ok — overlay |  |
| `pyscf.pbc.tools.print_funcs` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.tools.pyscf_ase` | **error** — `RuntimeError: ASE is not found` | **error** — `RuntimeError: ASE is not found` |  |
| `pyscf.pbc.tools.pywannier90` | **error** — `ImportError: cannot import name 'numint' from 'pyscf.dft'` | **error** — `ImportError: cannot import name 'numint' from 'pyscf.dft'` |  |
| `pyscf.pbc.tools.tril` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.x2c` | ok — site-packages 2.14.0 | ok — vendored 2.12.1 |  |
| `pyscf.pbc.x2c.sfx2c1e` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` |  |
| `pyscf.pbc.x2c.x2c1e` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` | **error** — `ImportError: cannot import name 'mole' from 'pyscf.gto'` |  |

