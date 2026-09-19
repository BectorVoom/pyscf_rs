# Upstream `pyscf/pbc` test suite against the overlay — after 20-19 A + C

**Measured:** 2026-09-14 18:55:49–18:58:30 JST. HEAD `5e6dd65` plus the Phase-18/20 working tree plus the 20-19 A/C Python edits (`20-19-AC-SUMMARY.md`).
**`.so`:** `python/pyscf/_native.abi3.so` mtime `2026-09-14 18:49:53` (item B's rebuild). Recorded before and after every family (`logs/run-overlay.families`): **unchanged for all 15 families**, so every family ran on the same build. No `maturin` process was running at start.
**Harness:** `target/p20-19-suite/` — the 20-18 harness copied verbatim with only its `S=` directory changed (`diff` = that one line in `env.sh` and `run_one.sh`); `plugin/`, `pydeps/`, `clibs-*` are symlinks to the 20-18 copies. Same file list (`files-run.txt`, 111 files), same `pytest.ini` + `--import-mode=prepend`, same `-p p2018env`, same `--timeout=600`, same overlay `PYTHONPATH`, same `clibs-2.14.0`, `P=3 NT=3`, `systemd-run --user --scope -p MemoryMax=11G`. The only change is the driver: `run_families.sh` runs the families one after another (each family's files still `xargs -P 3`) instead of one shuffled list, so the `.so` mtime could be stamped per family.
**Environment proof:** 111/111 per-process env JSONs have `pyscf.__file__` = `python/pyscf/__init__.py`, **0 site-packages-2.14.0 leaks**; no signal death; 0 timeouts.

## Headline

| run | passed / collected | ported families only | collection errors | tests that reached a test body |
|---|---:|---:|---:|---:|
| 20-18 overlay | 1 / 815 | 1 / 660 | 746 | 69 |
| **20-19 overlay (A + C, on B's `.so`)** | **11 / 815** (1.35 %) | **11 / 660** | **684** | **131** |
| control (pure vendored 2.12.1, 20-18) | 808 / 815 | 659 / 660 | 0 | 815 |

Same denominator (815, `logs/collect-collect-control.json` copied from 20-18). Nothing excluded.

**Every one of the 684 remaining collection errors is one of TWO semantic mismatches that item A was told to record, not fix** (they are native re-exports the identity contract keeps):

| tests | files | first error | cause |
|---:|---:|---|---|
| 444 | 43 | `TypeError: type 'pyscf._native.gto.Mole' is not an acceptable base type` | upstream `pyscf/pbc/df/ft_ao.py:565` `class ExtendedMole(gto.Mole)`; `pyscf.gto.Mole` is the native (non-subclassable) class |
| 240 | 39 | `AttributeError: type object 'pyscf._native.scf.UHF' has no attribute 'get_init_guess'` | upstream `pyscf/scf/rohf.py:349/384` `class ROHF(hf.RHF): get_init_guess = uhf.UHF.get_init_guess`; the overlay shims `python/pyscf/scf/{hf,uhf,ghf}.py` bind the native `RHF`/`UHF`/`GHF` |

All 17 missing-name rows of the 20-18 class-4a table (`gto.basis` 204, `dft.radi` 86, `gto.moleintor` 81, `ATOM_OF` 51, `gen_uniform_grids` 44, `mp.mp2`, `scf.atom_hf`, `gto.mole`, `cc.gccsd`, `gto.ft_ao`, `dft.rks`, `dft.numint`, `scf._vhf`, `scf.chkfile`, `moleintor`-as-name, `cc.ccsd_t`, `cc.ccsd`) and the 7 unported-family first errors are **gone**: no `ImportError`/`ModuleNotFoundError` for a molecular overlay name remains in any of the 111 logs.

## Per-family

| family | files | collected | 20-18 passed | **20-19 passed** | failed | errors (setup/test) | collection errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| `adc` (unported) | 8 | 14 | 0 | 0 | 0 | 0 | 14 |
| `cc` | 13 | 63 | 0 | 0 | 8 | 0 | 55 |
| `ci` | 2 | 5 | 0 | 0 | 0 | 4 | 1 |
| `df` | 22 | 226 | 0 | **3** | 6 | 0 | 217 |
| `dft` | 15 | 125 | 1 | **4** | 41 | 0 | 80 |
| `grad` (unported) | 10 | 57 | 0 | 0 | 0 | 0 | 57 |
| `gto` (incl. `gto/pseudo`) | 4 | 46 | 0 | 0 | 7 | 0 | 39 |
| `gw` (unported) | 2 | 2 | 0 | 0 | 0 | 0 | 2 |
| `lib` | 2 | 10 | 0 | 0 | 0 | 0 | 10 |
| `mp` | 8 | 28 | 0 | 0 | 3 | 0 | 25 |
| `scf` | 12 | 129 | 0 | **1** | 29 | 12 | 87 |
| `symm` | 2 | 9 | 0 | **2** | 7 | 0 | 0 |
| `tdscf` (unported) | 8 | 77 | 0 | 0 | 0 | 0 | 77 |
| `tools` | 2 | 19 | 0 | **1** | 3 | 0 | 15 |
| `x2c` (unported) | 1 | 5 | 0 | 0 | 0 | 0 | 5 |
| **total** | **111** | **815** | **1** | **11** | 104 | 16 | 684 |

Wall 2 min 41 s for the whole suite. Max RSS **11.3 GB** in `cc/test/test_rccsd_t_shift.py` (7 s, then failed on `rs_density_fit`; the control peaked at 3.0 GB there) — flagged, not investigated. Every other file < 1.4 GB.

**The 11 passes:** `df/test_band::{test_aft_band,test_aft_bands,test_fft_band}`, `dft/test_gks::test_collinear_gks_gga`, `dft/test_krks::test_klda`, `dft/test_kuks::test_klda`, `dft/test_sampling::test_kpt_vs_supercell` (the 20-18 pass), `scf/test_band::test_band_kscf`, `symm/test_basis::test_adjust_mesh`, `symm/test_spg::test_spg_elment_hash`, `tools/test_k2gamma::test_double_translation_indices`. Several of them (`test_krks`, `test_kuks`, `test_gks`, `test_band`) were class-5 `Cell`-input failures in 20-18, so they depend on item B's `.so` as much as on A.

## Failure taxonomy (804 non-passing)

| class | tests | what |
|---|---:|---|
| (1) import — unported family | 155 | `adc` 14, `grad` 57, `gw` 2, `tdscf` 77, `x2c` 5; first error is now one of the two mismatches below, not a missing name |
| (4a) import — **A semantic mismatch** (native re-export) | 529 | 444 `gto.Mole` not subclassable + 240 native `scf.uhf.UHF` in `rohf.py`, minus the unported-family share above (684 collection errors total) |
| (4a′) runtime import — same `rohf.py` mismatch inside a test body | 15 | `scf/test_newton.py::KnowValues::{test_nr_krhf,test_nr_krks_gga,test_nr_krks_lda,test_nr_kuhf,test_nr_kuks_gga,test_nr_kuks_lda,test_nr_rhf,test_nr_rhf_k1,test_nr_rks_gga,test_nr_rks_lda,test_nr_uhf,test_nr_uks_gga,test_nr_uks_lda,test_rks_gen_g_hop,test_uks_gen_g_hop}` |
| (5) native-input / API-surface gap | 74 | a native object lacks an attribute, kwarg or return shape upstream code or the test uses — see below |
| (5′) refused input (native raises by design) | 18 | `NotImplementedError` / typed `PyscfRsRuntimeError` — see below |
| (2) **numerical mismatch** | **13** | node ids below |
| (3) crash / signal / timeout | **0** | |

### (2) Numerical mismatches (node ids, measured value vs upstream reference)

| node id | Δ | tolerance |
|---|---:|---|
| `pyscf/pbc/dft/test/test_rks.py::KnownValues::test_density_fit` | **3.134** (−7.852143828842326 vs −4.717699891018736) | places=6 |
| `pyscf/pbc/dft/test/test_kuks.py::KnownValues::test_kuks_as_kuhf` | 9.36e-3 (−4.204045409900913 vs −4.213403459087) | places=9 |
| `pyscf/pbc/dft/test/test_rks.py::KnownValues::test_density_fit_2d` | 7.48e-3 | places=7 |
| `pyscf/pbc/dft/test/test_krks.py::KnownValues::test_klda8_cubic_kpt_222` | 4.32e-3 | places=7 |
| `pyscf/pbc/dft/test/test_krks.py::KnownValues::test_klda8_cubic_gamma` | 3.39e-3 | places=7 |
| `pyscf/pbc/dft/test/test_rks.py::KnownValues::test_rsh_0d` | 4.45e-4 | places=7 |
| `pyscf/pbc/dft/test/test_uks.py::KnownValues::test_pp_UKS` | 2.10e-4 | places=7 |
| `pyscf/pbc/dft/test/test_krks.py::KnownValues::test_klda8_primitive_kpt_222` | 6.95e-7 | places=7 |
| `pyscf/pbc/dft/test/test_krks.py::KnownValues::test_klda8_primitive_gamma` | 3.28e-7 | places=7 |
| `pyscf/pbc/df/test/test_band.py::KnownValues::test_fft_bands` | 7.86e-8 | places=7 |
| `pyscf/pbc/mp/test/test_ksym.py::KnownValues::test_kmp2` | 3.77e-10 | places=10 |
| `pyscf/pbc/mp/test/test_ksym.py::KnownValues::test_rdm1` | 1.53e-10 (vs 0.0) | places=10 |
| `pyscf/pbc/dft/test/test_gen_grid.py::KnownValues::test_becke_grids_round_error` | `assert 0.5111870535736216 < 0.1` | — |

Not triaged here. Item D (`get_ovlp` at `cell.precision` vs `precision*1e-5`) is a plausible cause of the 1e-7…1e-3 SCF rows and should be re-measured after D lands; `test_density_fit` (3.1 Ha) is too large for that and is a separate defect.

### (5) Native API-surface gaps (74), grouped

| tests | first error | node ids (class prefix `pyscf/pbc/…`) |
|---:|---|---|
| 12 | `'pyscf._native.pbc.scf.KRHF' object has no attribute 'convert_from_'` (setUp) | `scf/test/test_addons.py::KnownValues::*` (upstream `addons.convert_to_*` now imports and calls it) |
| 11 | `Cell` lacks `get_uniform_grids` / `get_lattice_Ls` / `ao_loc_nr` / `ao_loc`; `UniformGrids` lacks `non0tab`; `Cell.pbc_eval_gto(shls_slice=)` | `dft/test/test_numint.py::KnownValues::{test_1d_rho,test_2d_rho,test_3d_rho,test_eval_ao,test_eval_ao_kpt,test_eval_ao_kpts,test_eval_mat,test_eval_mat1,test_eval_rho,test_nr_rks,test_nr_uks_vxc_vv10}` |
| 8 | `'Cell' object has no attribute "set"` | `scf/test/test_mulliken_meta.py::KPTvsSUPCELL_{noshift,shift}::{test_kghf,test_krhf,test_krohf,test_kuhf}` |
| 5 | `.newton` unbound on `KRHF`/`KROHF`/`KRKS`/`KUKS` | `scf/test/test_newton.py::KnowValues::{test_exxdiv_treatment_newton,test_nr_krks_rsh,test_nr_krohf,test_nr_rohf,test_nr_uks_rsh}` |
| 5 | `GDF`/`MDF`/`RSDF` object has no attribute `set` | `df/test/test_band.py::KnownValues::{test_df_band,test_df_bands,test_mdf_band,test_rsdf_band,test_rsdf_bands}` |
| 4 | `Cell.pbc_eval_gto() got an unexpected keyword argument 'shls_slice'` | `dft/test/test_gen_grid.py::KnownValues::{test_becke_grids,test_becke_grids_1d,test_becke_grids_2d,test_becke_grids_2d_low_dim_ft_type}` |
| 4 | `'pyscf._native.pbc.ci.KCIS' object has no attribute 'ao2mo'` (setUp) | `ci/test/test_ci.py::KnownValues::*` |
| 3 | `'str' object has no attribute 'extend'` | `cc/test/test_krccsd_gamma.py::KnownValues::{test_111_n0,test_111_n1,test_311_n1}` |
| 3 | `Cell` lacks `atom_coords` / `bas_exp` / `bas_rcut` | `gto/test/test_rcut.py::KnownValues::{test_lattice_Ls_low_dim,test_loose_rcut,test_rcut}` |
| 3 | `KRKS` lacks `reset` / `to_hf`; `Cell` lacks `set_geom_` | `dft/test/test_krks.py::KnownValues::{test_reset,test_reset_ksym,test_to_hf}` |
| 2 | `KRHF` lacks `to_khf`; `numpy.ndarray` has no `todense` | `cc/test/test_kccsd_ksymm.py::KnownValues::{test_vs_krccsd,test_krccsd_ksym}` |
| 2 | `Cell` has no `magmom` / `get_scaled_atom_coords` | `symm/test/test_spg.py::KnownValues::{test_D4h_2d,test_spg_arccos_prec}` |
| 2 | `Cell` has no `nao`; `pbc_eval_gto(shls_slice=)` | `gto/pseudo/test/test_pp.py::KnowValues::{test_pp_nuc_grad,test_pp}` |
| 2 | `Cell.nimgs` read-only; `KRKS` lacks `mix_density_fit` | `dft/test/test_rks.py::KnownValues::{test_chkfile_k_point,test_rsh_mdf}` |
| 2 | `KUKS` lacks `mulliken_meta`; `kpts_to_kmesh` gets a builtin, not an array | `tools/test/test_k2gamma.py::KnownValues::{test_k2gamma,test_kpts_to_kmesh}` |
| 1 each | `KUKS.to_hf`; `KGKS.x2c`; `KRHF.rs_density_fit`; `KMP2.khelper`; `KSCF.get_hcore(kpt=)`; `Cell is not built` after an input change | `dft/test/test_kuks.py::KnownValues::test_to_hf`; `dft/test/test_gks.py::KnownValues::test_ncol_x2c_gks_lda`; `cc/test/test_rccsd_t_shift.py::KnownValues::test_water`; `mp/test/test_dm.py::KnownValues::test_kmp2_contract_eri_dm`; `scf/test/test_band.py::KnownValues::test_band`; `cc/test/test_kuccsd_openshell.py::test_kuccsd_openshell` |

### (5′) Refused input (18)

| tests | error | node ids |
|---:|---|---|
| 5 | `basis load error: unknown basis name 'gthdzvpmoloptsr'` | `symm/test/test_basis.py::KnownValues::{test_C2h_symorb,test_D3d_symorb,test_D6h_symorb,test_Oh_symorb,test_Td_symorb}` |
| 4 | `unknown XC functional token 'HSE06'` | `dft/test/test_krks.py::KnownValues::test_rsh_fft`, `dft/test/test_kuks.py::KnownValues::test_rsh_fft`, `dft/test/test_rks.py::KnownValues::test_rsh_fft`, `dft/test/test_uks.py::KnownValues::test_rsh_df` |
| 4 | `unknown XC functional token 'WB97'` | `dft/test/test_krks.py::KnownValues::test_rsh_df`, `dft/test/test_kuks.py::KnownValues::test_rsh_df`, `dft/test/test_rks.py::KnownValues::test_custom_rsh_df`, `dft/test/test_uks.py::KnownValues::test_rsh_fft` |
| 1 | `GDF.get_jk(omega > 0) on a 0-D cell` not implemented | `dft/test/test_rks.py::KnownValues::test_rsh_0d_df` |
| 1 | `cell.pseudo["He"]`: parsed GTH parameter lists not bound (plan 20-19 B) | `gto/pseudo/test/test_pp.py::KnowValues::test_pp_int` |
| 1 | mixed all-electron / pseudo cell not bound | `gto/pseudo/test/test_pp.py::KnowValues::test_pp_loc_part2` |
| 1 | symmetrized k-symmetric quadrature needs a uniform FFT grid | `tools/test/test_k2gamma.py::KnownValues::test_k2gamma_ksymm` |
| 1 | gamma `RCCSD` shim not bound | `cc/test/test_krccsd_gamma.py::KnownValues::test_111_n3` |

## Import probe (195 upstream `pyscf.pbc` modules, fresh interpreter each, overlay + vendored)

`target/p20-19-suite/probe_imports.py` (the 20-17 probe's method, `PYTHONPATH=<overlay>:<repo>`):

| tree | import ok | first-error histogram of the rest |
|---|---:|---|
| 20-17 after | 41 / 195 | 39 `gto.basis`, 35 `moleintor`, 15 `grad.rhf`, … |
| **20-19 overlay** (`logs/probe-overlay-after.json`) | **73 / 195** | **70** native `uhf.UHF.get_init_guess` (rohf.py), **43** `gto.Mole` not subclassable, 5 `mpi4py`, 1 each `geometric`/`spglib`/ASE/`libwannier90` |

## Counterfactual (supplementary — NOT a shipped configuration)

To size the two mismatches, the same suite was run against a scratch copy of the overlay (`target/p20-19-suite/cf/overlay`, `.so` symlinked) with `scf/{hf,uhf,ghf}.py` deleted and `pyscf.gto.Mole` left to upstream (`pyscf.gto.M` still native). This breaks the "every overlay name is native" rule for `pyscf.scf.hf.RHF` and `pyscf.gto.Mole`, so it is a measurement only. Harness: `cf/run_one.sh`/`cf/env.sh`/`cf/plugin` differ from the main harness only in the overlay path. Same `.so` 18:49:53 throughout (`logs/run-cf.families`).

| | passed / 815 | collection errors | import probe |
|---|---:|---:|---:|
| 20-19 overlay | 11 | 684 | 73 / 195 |
| counterfactual | **49** | 77 (`pyscf.pbc.dft` has no attribute `rks`: all 8 `tdscf` files) | 170 / 195 |

With the mismatches out of the way the next wall is the native `Cell`/`KSCF` API surface for upstream Python callers: `Cell.copy(**kwargs)` 83 (setUpModule), `Cell.max_memory` 94, `Cell.nao` 51, `KSCF.mo_occ` not writable 31, mixed AE/pseudo cell 26, `KUHF(KPoints)` 20, `convert_from_` 12, `remove_soscf` 8. One `SIGABRT` (`mp/test/test_scs.py`) and one setUpModule timeout occurred in the counterfactual only.

## Commands

```bash
cd target/p20-19-suite
stat -c '%y' python/pyscf/_native.abi3.so > logs/so-mtime-run-overlay-start.txt
systemd-run --user --scope -p MemoryMax=11G ./run_families.sh overlay 2.14.0 run-overlay 3 3
python3 aggregate.py run run-overlay && python3 report.py
python3 taxonomy.py run-overlay && python3 classify.py run-overlay      # logs/taxonomy-*.json, logs/buckets-*.json
systemd-run --user --scope -p MemoryMax=11G ./cf/run_families.sh overlay 2.14.0 run-cf 3 3   # counterfactual
.venv/bin/python target/p20-19-suite/probe_imports.py $REPO/python logs/probe-overlay-after.json
```

`report.py` prints the 20-18 control table from `logs/run-run-control.json` (copied). Raw logs, junit XML, `/usr/bin/time` and env JSON: `target/p20-19-suite/logs/run-overlay/`, `logs/run-cf/`.
