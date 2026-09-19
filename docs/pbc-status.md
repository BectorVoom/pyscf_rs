# `pyscf.pbc` in pyscf-rs — what is native, partial, upstream

**Status as of 2026-09-14** (Phase 20, plan 20-17). Policy: **D-PBC-35**
(`.planning/pbc/PBC-MASTER-PLAN.md` §3). Machine-readable source of truth:
`python/pyscf/pbc/_unported.py` (`FAMILIES`).

## 1. Checking at runtime

```python
import pyscf.pbc
pyscf.pbc.which_impl("grad")              # 'upstream'
pyscf.pbc.which_impl("scf")               # 'partial'
pyscf.pbc.which_impl("pyscf.pbc.scf.KRHF")  # 'native'
pyscf.pbc.which_impl("scf.newton_ah")     # 'upstream'
pyscf.pbc.which_impl("mp.RMP2")           # 'refused'  (bound, raises NotImplementedError)
pyscf.pbc.FAMILIES["tdscf"]["note"]       # 'Rust implementation exists (Phase 19), unbound in Python'
```

`which_impl(name)` takes a family (`"grad"`, `"pyscf.pbc.grad"`) or a dotted name
below one, and returns:

| value | meaning |
|---|---|
| `"native"` | the Rust binding: the overlay name **is** the `pyscf._native.pbc.*` object |
| `"partial"` | a family, or an overlay shim module, that mixes native names with an upstream fallthrough |
| `"upstream"` | upstream PySCF **Python** — which may fail to import (§4) |
| `"refused"` | dotted names only: bound in the overlay as a stub that raises `NotImplementedError` |

An unknown family raises `KeyError`.

**The one-time warning.** The first time per family per process that an import
resolves *outside* the overlay, `pyscf.pbc` emits
`pyscf.pbc.PbcUpstreamFallthroughWarning` (a `UserWarning` subclass) naming the
family, its status, the file it resolved to, and `which_impl`. Imports of the
overlay's own (native) modules never warn. The finder that emits it returns
`None` from `find_spec`, so it never changes what is imported. To silence it:

```python
import warnings
from pyscf.pbc import PbcUpstreamFallthroughWarning
warnings.simplefilter("ignore", PbcUpstreamFallthroughWarning)
```

## 2. Making sure the overlay is the `pyscf` you import

Measured in this repository's `.venv` (2026-09-14):

| how Python is started | `import pyscf` resolves to | overlay active? |
|---|---|---|
| `PYTHONPATH=$REPO/python .venv/bin/python script.py` (any cwd) | `python/pyscf` | **yes** |
| `.venv/bin/pytest python/pyscf/tests` | `python/pyscf` (rootdir insertion) | **yes** |
| `.venv/bin/python script.py` without `PYTHONPATH` | `.venv` site-packages pyscf **2.14.0** (`pyscf_rs.pth` appends `python/` *after* site-packages) | **no** |
| `python -c …` with cwd = repo root | vendored `pyscf/` 2.12.1 (cwd `''` is first) | **no** |

With the overlay active, an upstream passthrough resolves to the next `pyscf` on
`sys.path` — **pyscf 2.14.0 under pytest**, not the 2.12.1 oracle. The warning
names the file.

## 3. The families

21 upstream packages under `pyscf/pbc` (vendored 2.12.1). The last column counts
the family's module names (from the vendored tree) that import under the overlay,
overlay shims included (`.planning/phases/20-pbc-python-bindings/measurements/overlay-passthrough.md`,
fresh interpreter per module).

| family | `which_impl` | native in Python (overlay re-exports) | not ported / unbound | Rust | imports under overlay |
|---|---|---|---|---|---|
| `gto` | partial | `Cell`, `M`, `make_kpts`, `band_path`, `super_cell`, `cell_plus_imgs`, `get_coulG`, `get_kconserv`, `dumps`/`loads`/`pack`/`unpack` | `ecp`; explicit shell-list `basis`, per-element `pseudo` dicts raise; other names fall through to upstream `cell`/`basis`/`pseudo` | `pyscf-pbc-gto` (20-09) | 3/16 |
| `df` | native | `FFTDF`, `AFTDF`=`PWDF`, `GDF`=`DF`, `MDF`, `RSDF`=`RSGDF`, `density_fit` | submodules (`incore`, `outcore`, `ft_ao`, `*_jk`, `*_ao2mo`) are upstream | `pyscf-pbc-df` (20-10) | 4/21 |
| `scf` | partial | `KRHF`, `KUHF`, `KROHF`, `KGHF`, `KsymAdaptedKRHF`, `RHF`/`UHF`/`ROHF`/`GHF`/`HF`/`KHF`, `smearing_`, `load_scf`, `project_mo_nr2nr` | `newton_ah` (`.newton()`), `stability`, `cphf`, `_response_functions`, `scfint`, `rsjk`; `kuhf_ksymm`/`kghf_ksymm` **refused** | `pyscf-pbc-scf` (20-12); Rust `newton_ah`/`stability`/`cphf`/`response` exist in the working tree, **unbound**; `rsjk` refuses in Rust (20-06) | 6/21 |
| `dft` | native | `KRKS`, `KUKS`, `KROKS`, `KGKS`, `KRKSpU`, `KUKSpU`, `KsymAdapted*`, `UniformGrids`/`BeckeGrids` (also as `dft.gen_grid.*`), `KNumInt`/`MultiGridNumInt(2)`, `RKS`/`UKS`/`GKS`/`ROKS`/`KS`/`KKS` | `cdft`, `numint2c`; meta-GGA refused in Rust; `nlc`/VV10 refused; `KUKSpU` compute refused and ksymm GGA on s-only bases refused (20-13 D3/D4 crate defects) | `pyscf-pbc-dft` (20-13) | 2/25 (both overlay) |
| `symm` | partial | `KPoints`, `SpaceGroup`, `SPGElement`, `make_kpts`, `get_crystal_class` | `pyscf_spglib` (by design); `pyscf.pbc.symm.Symmetry` is upstream's class | `pyscf-pbc-symm` (20-14) | 2/8 |
| `lib` | partial | `kpts_helper.{get_kconserv, get_kconserv3, is_zero, is_trim, member, intersection, unique, unique_with_wrap_around, group_by_conj_pairs, kk_adapted_iter, gamma_point, KPT_DIFF_TOL}`, `kpts.{KPoints, make_kpts}` | `arnoldi`, `linalg_helper`, `chkfile`, `ktensor`; `kpts_helper.{round_to_fbz, members_with_wrap_around, loop_kkk, conj_mapping, get_kconserv_ria, KptsHelper}` fall through | `pyscf-pbc-lib`, `pyscf-pbc-symm` (20-14; shims 20-17) | 7/7 |
| `tools` | partial | `fft`, `ifft`, `fftk`, `ifftk`, `get_coulG`, `madelung`, `super_cell`, `cell_plus_imgs`, `cutoff_to_mesh`, `mesh_to_cutoff`, `ExxDiv` (also as `tools.pbc.*`) | `k2gamma`, `lattice`, `pyscf_ase`, `pywannier90`, `print_funcs`, `tril`, `make_test_cell`; `pbc.{precompute_exx, get_monkhorst_pack_size, get_lattice_Ls, check_lattice_sum_range, cutoff_to_gs, gs_to_cutoff, round_to_cell0}` ported in Rust, unbound | `pyscf-pbc-tools` (20-14; shims 20-17) | 6/9 |
| `mp` | native | `KMP2`=`KRMP2`, `KsymAdaptedKMP2`, `KUMP2`, `KMP2_stagger` | gamma `RMP2`/`MP2`/`UMP2`/`GMP2` **refused** | `pyscf-pbc-mp` (20-15) | 1/6 |
| `cc` | native | `KRCCSD`=`KCCSD`, `KUCCSD`, `KGCCSD`, `KsymAdaptedRCCSD`, `EOMIP`/`EOMEA`/`EOMEESinglet`/`EOMEE` | gamma `RCCSD`/`CCSD`/`UCCSD`/`GCCSD` **refused**; `frozen=`, EOM `partition='mp'/'full'` raise | `pyscf-pbc-cc` (20-15) | 4/19 |
| `ci` | partial | `KCIS`=`CIS` | `cisd` (`RCISD`/`CISD`/`UCISD`/`GCISD` **refused**, deferred by design) | `pyscf-pbc-ci` (20-15) | 1/3 |
| `ao2mo` | native | `general`, `get_mo_eri`, `get_mo_pairs_G`, `get_mo_pairs_invG`, `assemble_eri`, `get_ao_pairs_G`, `get_ao_eri` | — | `pyscf-pbc-ao2mo` (20-15) | 1/2 |
| `grad` | upstream | — | everything | `pyscf-pbc-grad` in progress (Phase 18, uncommitted), **unbound** | 0/15 |
| `geomopt` | upstream | — | everything | 13-line stub crate | 0/2 |
| `tdscf` | upstream | — | everything | `pyscf-pbc-tdscf` implemented (Phase 19), **unbound** | 0/9 |
| `tddft` | upstream | — | everything | no crate (upstream alias of `tdscf`) | 0/1 |
| `gw` | upstream | — | everything | `pyscf-pbc-gw` implemented (Phase 19), **unbound** | 0/7 |
| `adc` | upstream | — | everything | `pyscf-pbc-adc` implemented (Phase 19), **unbound** | 0/7 |
| `x2c` | upstream | — | everything | `pyscf-pbc-x2c` implemented (Phase 19), **unbound** | 1/3 (`__init__` only) |
| `eph` | upstream | — | everything | `pyscf-pbc-eph` implemented (Phase 19), **unbound** | 1/2 (`__init__` only) |
| `mpicc` | upstream | — | everything | `pyscf-pbc-mpi` 13-line stub | 0/4 |
| `mpitools` | upstream | — | everything | `pyscf-pbc-mpi` 13-line stub | 1/6 (`__init__` only) |

Totals: **10 upstream, 6 partial, 5 native.** "Phase 19 implemented, unbound"
means Rust code exists and is gated in Rust, but a Python user of
`pyscf.pbc.tdscf` (etc.) gets upstream Python — or, today, an `ImportError`.

## 4. The upstream passthrough is announced — and mostly broken

Measured 2026-09-14 over all 195 modules of the vendored `pyscf/pbc` tree
(`.planning/phases/20-pbc-python-bindings/measurements/overlay-passthrough.md`):

* **41 / 195 import** under the overlay: 18 are overlay packages/shims/stubs,
  **23 are upstream modules**. **154 of the 177 modules the overlay does not
  shadow fail to import** — identically whether the fallthrough target is
  site-packages 2.14.0 or the vendored 2.12.1 tree.
* The failures are caused by the **molecular** overlay packages
  (`python/pyscf/{gto,scf,dft,cc,mp,grad}/__init__.py`), which shadow upstream's
  without `extend_path` and without the names upstream imports:
  `pyscf.gto.basis` (39), `pyscf.gto.moleintor` (35), `pyscf.grad.rhf` (15),
  `pyscf.mp.mp2` (9), `pyscf.gto.mole` (9), `ATOM_OF` (6), `pyscf.cc.rccsd` (5),
  `mpi4py` (5, a genuine missing dependency), …
* Every one of the ten "upstream" families fails at `import pyscf.pbc.<family>`
  except `x2c`, `eph` and `mpitools`, whose `__init__` imports but whose
  submodules do not. **In practice an unported family is unavailable, not
  "served by upstream".** The warning fires before the `ImportError`.
* Before 20-17 the same probe gave 31 / 195 (19 upstream) and **no warning**.
  20-17's `lib`/`tools`/`dft.gen_grid` shims turned 10 errors into successes
  and broke none.

## 5. Upstream examples on the native path (for 20-18)

Attribute/constructor probe only (no `kernel()`), overlay active:

`examples/pbc/20-k_points_scf.py`
* native: `gto.M`, `cell.make_kpts`, `scf.KRHF(cell, kpts)`, `dft.KRKS`,
  `kmf.grids = dft.gen_grid.BeckeGrids(cell)`, `kmf.xc = ...`,
  `KRHF(..., exxdiv=None).density_fit()`.
* **gaps:** `kmf.xc = 'm06,m06'` is a meta-GGA, which `pyscf-pbc-dft/src/xc.rs`
  refuses at kernel time; `scf.KRHF(cell, kpts).newton()` →
  `AttributeError` (Newton solver unbound) — the script stops there.

`examples/pbc/22-k_points_mp2.py`
* native: `gto.Cell()` + attribute build, `scf.KRHF(cell)` + `kmf.kpts = kpts`
  (both `(8,3)` and a single `(3,)` k-point), `mp.KMP2(kmf)` after `kernel()`,
  `scf.RHF(cell, kpt=kpt)` (the K-driver at one k-point), `mf.with_df.ao2mo`,
  `mf.get_hcore`, `mf.energy_nuc`.
* **gaps:** `mp.RMP2(mf)` raises `NotImplementedError` (gamma MP2 refused) — the
  script stops there; `scf.addons.convert_to_uhf` (upstream `addons`, fails to
  import), `mp.UMP2`/`mp.GMP2` (refused) follow it. Gamma `RHF` results are per-k
  lists of length 1, so the later `mo_coeff.shape[1]` / `ao2mo(...).reshape`
  lines would also differ from upstream.
