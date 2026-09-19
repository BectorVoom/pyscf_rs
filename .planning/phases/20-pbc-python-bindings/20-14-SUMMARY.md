# 20-14 SUMMARY — `pbc.symm` / `pbc.lib.kpts_helper` / `pbc.tools`

**Shipped:** 2026-09-14. All six tasks are done, in TDD order. The tests were written first: 31 failed against the pre-20-14 `.so`, and all 31 pass after the rebuild. Every verification command is green except the KRHF dispatch item, which is deferred to 20-12 (see the deviations).

## Task 1 — the failing tests first

`python/pyscf/tests/test_pbc_symm.py` has 31 cases and uses the Bohr fixtures of `test_pbc_cell.py`: diamond, Si and He-fcc. The Si fixture needed a correction, see D2. Some cases compare against a vendored-2.12.1 subprocess, and the version is asserted.

RED: `pytest test_pbc_symm.py` gave **31 failed in 1.00 s** (`AttributeError: module 'pyscf._native.pbc.symm' has no attribute 'make_kpts'`, …).

## Task 2 — `KPoints` (`crates/pyscf-py/src/pbc/symm.rs`)

`KPoints` is a `#[pyclass(subclass)]` constructed as `KPoints(cell=None, kpts=None)`.

**Build**
- `build(space_group_symmetry, time_reversal_symmetry, symmorphic=True, check_mesh_symmetry=True)` calls `KPoints::build` and returns self.
- The binding keeps the Python cell, so `kpts.cell is cell`, plus the Rust cell it was built against. A Rust `KPoints` stores no `Cell` (D-PBC-25).

**Accessors**
- All upstream attributes are bound: `nkpts[_ibz]`, `kpts[_ibz]`, `kpts_scaled[_ibz]` (`None` before build), `weights[_ibz]`, `ibz2bz`, `bz2ibz`, `k2opk`, `stars[_ops]`, `stars_ops_bz`, `time_reversal_symm_bz`, `little_cogroup_ops`, `ops` (`SPGElement`), `Dmats`, `nop`, `has_inversion`, `time_reversal` and `__len__`.
- Index arrays are int64.

**Methods**
- `get_kconserv()` returns int32 and equals `kpts_helper.get_kconserv(cell, kpts.kpts)`.
- `make_gdf_kptij_lst_jk`.
- `make_ktuples_ibz(kpts_scaled=None, ntuple)`. A non-`None` `kpts_scaled` raises `NotImplementedError`, because that branch is not ported.
- `make_k4_ibz(sym, return_ops)`:
  - **`'s4'` raises** `PyscfRsRuntimeError` with kind `NotYetImplemented`.
  - Any other unknown string raises `NotImplementedError`.
- `little_cogroups()` returns sorted `(order,3,3)` int32 rotation stacks plus index arrays. A `PointGroup` object is not bound.
- `ops_outside_kmesh_subgroup()` and `little_cogroup_ops_outside_kmesh_subgroup()` are bound; they are the D-17-09-02 detector.
- `transform_mo_coeff`, `transform_single_mo_coeff`, `transform_mo_occ`, `transform_mo_energy`, `transform_dm`, `transform_1e_operator` and `transform_fock` are bound.
  - They accept upstream's `[2,]` spin form, detected by `ndim(x[0][0])`.
  - Complex blocks are read row-major by LOGICAL index (the Rust API is interleaved `Vec<Complex64>`), so non-contiguous views are read correctly (tested bitwise). The GIL is released.
- `symmetrize_wavefunction` raises like upstream (`RuntimeError`).

**The Phase-17 upstream defects are NOT "matched back".**
- D-17-07-01: `little_cogroups()` under time reversal on a cell without inversion raises kind `PbcSymm` where upstream raises `IndexError`. This is tested on diamond's symmorphic Td subset.
- D-17-09-02: the detector gives **36 of 48** ops outside the subgroup on `si [1,1,2]`. This reproduces 17-VERIFICATION §6 item 3.
- No test compares `little_cogroup_ops`, the per-irrep eig or the KCCSD guard against upstream.

**Free functions.** `make_kpts(cell, kpts, space_group_symmetry, time_reversal_symmetry)` calls `pyscf_pbc_symm::kpts::make_kpts`, the exact call Gate A gates. A `KPoints` input is rebuilt, as upstream does. `get_crystal_class` is also bound.

**`pyscf._native.pbc.lib.kpts.{KPoints, make_kpts}` are the same objects**, so upstream's `isinstance(kpts, libkpts.KPoints)` has a type to test.

## Task 3 — `Symmetry`, `SpaceGroup`, geometry

**`SpaceGroup(cell, symprec)`**
- `.build(dump_info)` gives diamond `nop=48` and `groupname['point_group_symbol']=='m-3m'`.
- `backend` is `'pyscf'`; setting `'spglib'` raises `NotImplementedError`. There is no spglib dependency.

**`Symmetry(cell)`**
- `.build(space_group_symmetry, symmorphic, check_mesh_symmetry)` binds `ops`, `nop`, `has_inversion`, `Dmats[iop][l]`, `l_max`, `spacegroup`, `_built`, `check_mesh_symmetry(..., return_mesh)`, `reset` and `dump_info`.
- The symmorphic subset of Fd-3m is 24 ops.

**`Cell.build(space_group_symmetry=True)` now builds the lattice symmetry** (`cell.py:1770-1772`). `symm::ensure_lattice_symmetry` calls `build_lattice_symmetry(cell, check = !_mesh_from_build)`. Because of this, an auto mesh is enlarged exactly as upstream does it: diamond `[48]*3` and Si `[36]*3`, both equal to upstream. Every cell entering `symm.rs` passes through the same step, upstream-shaped cells included.

## Task 4 — `pbc.lib.kpts_helper` (`crates/pyscf-py/src/pbc/lib.rs`)

These are real nested modules registered in `sys.modules`: `pyscf._native.pbc.lib.kpts_helper` and `.kpts`.

**Bound functions**
- `KPT_DIFF_TOL`
- `is_zero`; `is_gamma_point` and `gamma_point` are the same object
- `is_trim`, `member`, `intersection`, `unique`, `unique_with_wrap_around`
- `group_by_conj_pairs(cell, kpts, wrap_around, return_kpts_pairs)`
- `kk_adapted_iter(cell, kpts, kk_idx, time_reversal_symmetry=True)`. It is an iterator of `(kpt, ki int32, kj int32, self_conj)`, and `kk_idx` combined with TRS raises `NotImplementedError`.
- `get_kconserv3`
- `get_kconserv` is **20-09's `pbc.gto.get_kconserv`** (identity).

## Task 5 — `pbc.tools` (`crates/pyscf-py/src/pbc/tools.rs`)

**FFT family.** `fft`, `ifft`, `fftk` and `ifftk` read any array whose size is a multiple of `prod(mesh)`, C-order as `(nbatch, ngrids)` rows, and return complex128 in the input shape.

**`madelung`, `cutoff_to_mesh`, `mesh_to_cutoff`** are bound.

**`ExxDiv`** is bound **once**, as a `#[pyclass] enum` with `EWALD`, `VCUT_SPH`, `VCUT_WS`, `parse` and `str`. `pbc.gto.ExxDiv is pbc.tools.ExxDiv` holds: `tools::register` sets the attribute on the gto module.
- `get_coulG` now also accepts an `ExxDiv` instance.
- A garbage `exx` raises `TypeError`. Before 20-14, a non-string, non-bool value silently meant "no correction".

**Identity re-exports.** `get_coulG`, `super_cell` and `cell_plus_imgs` are **20-09's objects** (identity).

## Task 6 — unported tools and surface, handed to 20-17

**No Rust counterpart; not bound:**

| upstream module / name | note |
|---|---|
| `pbc/tools/k2gamma.py` | no Rust module (`get_kconserv` ports only `_get_kconserv_slow`) |
| `pbc/tools/lattice.py` | named-crystal builders. Not `pyscf-pbc-tools/src/lattice.rs`, which ports `pbc.py`'s lattice sums |
| `pbc/tools/pyscf_ase.py`, `pywannier90.py`, `print_funcs.py`, `tril.py` | none |
| `pbc/tools/make_test_cell.py` | partial only: feature-gated `pyscf-pbc-gto/src/test_systems.rs:89` |
| `pbc/symm/pyscf_spglib.py` | deliberately none (native detection) |
| `kpts_helper.members_with_wrap_around`, `conj_mapping`, `get_kconserv_ria`, `loop_kkk` | not ported |
| `pbc/lib/arnoldi.py`, `linalg_helper.py`, `chkfile.py` | not in `pyscf-pbc-lib` |

**Ported in Rust but NOT bound by 20-14** (candidates if 20-17 wants them):
- `pbc.tools`: `precompute_exx` (`exxdiv_vcut.rs:103`), `get_monkhorst_pack_size`, `get_lattice_Ls`, `check_lattice_sum_range`, `cutoff_to_gs`, `gs_to_cutoff`, `round_to_cell0`
- `KPoints`: `symmetrize_density`, `check_mo_occ_symmetry`, `dm_at_ref_cell`, `get_rotation_mat_for_mos`, `little_cogroup_rep`, `addition_table`/`inverse_table`, `loop_ktuples`/`ktuple_to_index`/`index_to_ktuple`, `reset`
- Classes: `KQuartets`, `MORotationMatrix`, `KsymmArray` (`ktensor.rs`), `KptsHelper` (`khelper.rs`)
- `symm.basis` (`symm_adapted_basis`), `symm.group` (`PointGroup`, `Representation`), `geom.search_*_ops`

**Overlays.**
- Only `python/pyscf/pbc/symm/__init__.py` was added. It re-exports `KPoints`, `SpaceGroup`, `SPGElement`, `make_kpts` and `get_crystal_class`, and falls through lazily to upstream submodules.
- **`Symmetry` stays upstream in the overlay** (see D4).
- `pbc/lib` and `pbc/tools` overlays were NOT added. An `__init__.py` there would shadow upstream's `pyscf.pbc.tools` (`from .pbc import *`) and `pyscf.pbc.lib.kpts`, which upstream's own modules import. That shim-and-fallthrough design is 20-17's decision.

## Verification (2026-09-14)

| command | result |
|---|---|
| `cargo check -p pyscf-py` / `cargo clippy -p pyscf-py` | exit 0. No warning in `symm.rs`/`lib.rs`/`tools.rs` (the remaining `gto.rs:1099 manual_map` is 20-09's code) |
| `maturin develop --release --skip-install` (EXECUTION-NOTES §2 env) | **maturin exit=0**, cargo `Finished release in 3.41 s` (log `target/py-20-14-build1.log`, `.so` 15:03). The 20-11 agent had just compiled the same tree, so this build was nearly a no-op. |
| `.venv/bin/pytest python/pyscf/tests/test_pbc_symm.py -q -p no:cacheprovider` | **31 passed in 4.01 s**, exit 0 |
| Gate A through Python, `[16,16,16]` | **si, diamond: `[145, 145, 245, 408, 816, 2052]` EXACT. He-fcc (Fm-3m control): `[145, 145, 145, 408, 408, 2052]` EXACT.** Star/weight/`bz2ibz` invariants hold on all 18 KPoints. `cell.make_kpts([16]*3, space_group_symmetry=True)` gives a `KPoints` with 145 IBZ points, bitwise equal to the free function. |
| vs vendored 2.12.1, `[4,4,4]` + TRS | diamond/Si: `mesh_sg` equal, `nop`, `nkpts_ibz` (8), `ibz2bz` and `bz2ibz` EXACT, `weights_ibz` Δ=0, `kpts_ibz` max\|Δ\| 0 / 1.1e-16 |
| kpts_helper vs 2.12.1 | `group_by_conj_pairs` ([3,3,3]), `kk_adapted_iter` (TRS on [2,2,2], TRS off [3,3,3]; kpt Δ=0), `get_kconserv3`, `is_trim`: all EXACT |
| tools vs 2.12.1 | `cutoff_to_mesh` EXACT. `mesh_to_cutoff` rel ≤3.9e-16. `madelung` rel 9.8e-16 (diamond) / 4.1e-15 (Si), asserted ≤1e-10. |
| `fft`/`ifft` | C-order layout vs `numpy.fft.fftn`: rel ≤1.3e-15. Repeat calls bitwise. Batch vs per-row bitwise. Real input vs `+0j` bitwise. `fftk == fft(scale_by product)` and `ifftk` bitwise. Round trip on a constant field **bitwise**. |
| `make_k4_ibz(sym="s4")` | raises `PyscfRsRuntimeError`, `args[1] == "NotYetImplemented"` |
| `pbc.gto.ExxDiv is pbc.tools.ExxDiv` | True. `get_coulG(exx=ExxDiv.EWALD)` is bitwise `exx='ewald'` and differs from no-`exx` in exactly 1 element (G+k=0). |
| `isinstance(kpts, KPoints)` | True for `symm.make_kpts`, `cell.make_kpts(..., symmetry)`, `lib.kpts.KPoints`, the overlay `pyscf.pbc.symm.KPoints`, and a Python subclass |
| `pytest test_pbc_identity_gate.py` | **8 passed, 10 failed**. `pbc.symm.KPoints` newly passes. The failures are scf ×4, dft ×4, mp, cc (20-12, 20-13, 20-15). |
| `pytest test_pbc_cell.py` | 42 passed (one test restated, D3) |
| `cargo run -p xtask --bin check-catch-unwind` | **exit 0** (PASS, 843 files) |
| `check-dependency-wall` / `check-forbid-lazy-static` / `check-orphan-modules` | exit 0 (both PASS lines) / 0 / 0 (444 files) |
| full `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider` | **20 failed, 193 passed, 5 skipped, 3 xfailed, 9 errors in 215.54 s** (exit 1, log `target/py-20-14-fullsuite.log`). The PBC files all pass: `test_pbc_symm`, `test_pbc_cell`, `test_pbc_df`, `test_complex_boundary`. The failures break down as: 10 `test_pbc_identity_gate` (scf/dft/mp/cc, not yet bound); 2 `test_panic_to_exception` (known pre-existing, 20-08 D3); and 8 failed plus 9 errors in the MOLECULAR `test_scf_*` files. Those 17 are all overlay import or signature errors: `No module named 'pyscf.gto.moleintor'` ×9, `cannot import name 'ATM_SLOTS' from 'pyscf.gto'` ×5, `M() got an unexpected keyword argument 'verbose'` ×4, `cannot import name 'radi' from 'pyscf.dft'` ×4 (20-08 D4), `cannot import name 'rccsd' from 'pyscf.cc'`, `GHF has no attribute 'run'`, `DID NOT RAISE PyscfRsError` ×3, and two 1-cycle SCF non-convergence errors in the `*_uhartree_oracle` tests. None of them touches a file this plan edited (only `pbc/*` and the `pbc.symm` overlay). They are not attributed to 20-14, but they could not be A/B'd against a pre-20-14 `.so`, because 20-11 is rebuilding the same `.so` concurrently. |

## Deviations

- **D1 — Gate A mesh is `[16,16,16]`, not `[2,2,2]`.** The plan says `si`/`diamond` `[2,2,2]`, but the six integers 145…2052 are defined at `[16,16,16]` (17-VERIFICATION §3, `kpts_ibz.rs::KMESH`). `[2,2,2]` has only 8 k-points. EXECUTION-NOTES rule: facts win.
- **D2 — `test_pbc_cell.py`'s silicon fixture is not a diamond-structure cell.** `Q_SI = 2.55555` but `a0/4 = 10.2622/4 = 2.56555`. That cell detects only 12 ops, and symmetrising its mesh blows it up from `[35]^3` to `[1795]^3`. `test_pbc_symm.py` uses `2.56555`. 20-09's own upstream comparisons are unaffected, because both sides use the same off-structure geometry. The fixture was left as is, since it is 20-09's test.
- **D3 — edits outside the owned files, all minimal:**
  - `pbc/gto.rs` has three changes:
    - `do_build` calls `ensure_lattice_symmetry` (upstream `Cell.build` parity, required for Gate A through `Cell`).
    - `Cell.make_kpts(space_group_symmetry/time_reversal_symmetry=True)` now returns a `KPoints`. It previously raised `NotImplementedError` pointing at this plan. It also applies upstream's `RuntimeError` when the cell lacks `space_group_symmetry`.
    - `get_coulG` routes `exx` through `tools::extract_exxdiv`.
  - `test_pbc_cell.py::test_make_kpts_with_symmetry_points_at_pbc_symm` was restated, not deleted.
  - `pbc/mod.rs` got three `pub mod` lines and three match arms. It was concurrently edited by 20-11, and rustfmt sorted the `pub mod` block.
- **D4 — `pyscf.pbc.symm.Symmetry` is upstream's class in the overlay.** The native `Symmetry` is bound at `pyscf._native.pbc.symm.Symmetry` only. Upstream `cell.py:1567-1579` builds `pyscf.pbc.symm.Symmetry` on an upstream cell and then `del`s `.cell`/`.spacegroup.cell`, which a native object would break. The native `Cell` never reaches that code.
- **D5 — "fft/ifft round-trip `to_bits()`-identical" does not hold for general data, and no tolerance was loosened to hide it.** Measured `max|ifft(fft(f)) − f|` for random complex batches:

  | mesh | max \|Δ\| |
  |---|---|
  | `[4,4,4]` | 9.2e-16 |
  | `[5,6,7]` | 1.4e-15 |
  | `[9,9,9]` | 3.1e-15 |
  | `[16,16,16]` | 1.2e-15 |

  74–88 % of the bits differ. This is a property of a finite-precision transform; Rust's own `fft_accuracy.rs` gates the same round trip at 1e-14. The bitwise claims that DO hold are:
  - binding-vs-Rust determinism;
  - batch/row identity;
  - `fftk`/`ifftk` = `fft`/`ifft` of `scale_by`'s product;
  - the round trip on an exactly-representable (constant) field.

  The general round trip is asserted < 1e-13.
- **D6 — `KRHF(cell, kpts=KPoints(...))` dispatch is deferred to 20-12.** `pbc.scf` does not exist natively yet. This plan provides the type, and `isinstance` works on it in every spelling.
- **D7 — gaps vs upstream:**
  - `transform_*` outputs are always complex128 lists.
  - `little_cogroups` returns rotation stacks, not `PointGroup` objects.
  - `SPGElement.rot` is int32 (upstream int64).
  - `make_ktuples_ibz(tol)` is accepted and ignored (unused on the ported branch).
  - `Symmetry.check_mesh_symmetry(ops=...)`, `get_crystal_class(ops=...)` and `SpaceGroup.dump_info(ops=...)` raise `NotImplementedError`.
