# 20-09 SUMMARY — `pbc.gto`: `PyCell`, `M`, `make_kpts`, `band_path`, `super_cell`, eval surface, `extract_cell_from_pyany`

**Shipped:** 2026-09-14. All six tasks are done, along with the two cross-cutting Step-0 fixes the orchestrator assigned. Built together with 20-10 in one source batch and two release `.so` builds.

## Step 0 — cross-cutting (orchestrator)

- **0a — `pyscf-pbc-dft` added to `crates/pyscf-py/Cargo.toml`** (default features, so libxc, per D-20-B). No bindings yet; those are 20-13's.
  - `cargo tree -p pyscf-py -e normal | grep -c libxc` gives **547 before and after**.
  - A `cargo tree -e features` diff shows exactly two added lines (`pyscf-pbc-dft feature "default"` and `"libxc"`). No other feature, including `rmath`, re-unified, so the libxc kernels did **not** rebuild.
- **0b — the seven flat submodules are now in `sys.modules`.** `lib.rs::register_flat_in_sys_modules` enters `pyscf._native.{scf,gto,dft,mp,cc,grad,geomopt}` and the two `geomopt` solver shims, using the step-3 pattern from `pbc::register`. `__name__` is unchanged.
  - `cd python && ../.venv/bin/python -c "import pyscf.dft, pyscf.mp, pyscf.cc; print('ok')"` prints `ok`, exit 0.

## Build economics (measured)

| build | what changed | cargo `Finished` | wall |
|---|---|---|---|
| r1 `maturin develop --release --skip-install` (EXECUTION-NOTES §2 env) | Cargo.toml (+pbc-dft), pyscf-py sources, plus other agents' uncommitted `pyscf-pbc-df` edits since the 06:24 `.so` (cascade pbc-df → scf → dft → mp → cc → ci → py) | **53.69 s** | 59 s, maturin exit 0 |
| r2, same command | pyscf-py source only (`df.rs` hook) | **15.98 s** | **23 s**, maturin exit 0 |

The ~65 min rebuilds in 20-07/20-08 were caused by feature re-unification (`rmath`). An incremental pyscf-py-only rebuild is well under a minute.

## Task 1 — tests first

`python/pyscf/tests/test_pbc_cell.py` has 42 cases. It uses the Bohr fixtures from `pyscf-pbc-scf/tests/common/mod.rs`: diamond and Si (`gth-szv`/`gth-pade`), and He-fcc (`sto-3g`, all-electron). The cases cover:
- the `Deref` resolution;
- GTH valence charges;
- `vol`, `b` and `make_kpts` vs vendored 2.12.1 (subprocess, version asserted) at the `oracle_phase9.rs` floors (1e-12);
- `make_kpts` bitwise;
- `get_hcore` absent;
- every refusal;
- pbc_intor/eval_gto shapes;
- supercell;
- band path;
- dumps/loads/pack/unpack/pickle/copy;
- `extract_cell_from_pyany`.

## Tasks 2–5 — the surface (`crates/pyscf-py/src/pbc/gto.rs`, `pbc/convert.rs`, `bridge.rs`)

- **`Cell`** (`#[pyclass(subclass, module="pyscf._native.pbc.gto")]`) follows upstream's mutable-then-`build()` shape.
  - `Cell(**kw)` sets inputs; `cell.build(**kw)` builds and returns the cell; `M(**kw)` does both.
  - Build-input attributes are get/set: `atom`, `basis` (a name or `{el: name}`), `pseudo` (a name), `a` (3×3 or string), `unit`, `ke_cutoff`, `precision`, `dimension`, `low_dim_ft_type`, `fractional`, `exp_to_discard`, `charge`, `spin`, `cart`, the symmetry/rcut/PME flags and `verbose`.
  - Setting a build input drops the built state. Accessors then raise `ValueError` naming `build()`. The exceptions are `mesh` and `rcut`, which are applied in place as in upstream.
- **Deref.** `cell.mol` is a `pyscf._native.gto.Mole` over the molecular half. `__getattr__` forwards `MOLE_SURFACE = nao_nr, nao_2c, natm, nbas, nelectron, atom_symbol`.
  - Periodic-meaning names (`intor`, `eval_gto`, `energy_nuc`, `dumps`, `atom_charges`) are bound periodically.
  - `intor_spinor` is deliberately not forwarded.
  - `PyMole` gained `natm`, `nbas`, `nelectron`, `atom_charges()`, `atom_coords()` and `atom_symbol()` (`gto.rs`).
- **Accessors:**
  - `lattice_vectors()`, `vol`, `reciprocal_vectors(norm_to)`, `get_abs_kpts`, `get_scaled_kpts`
  - `make_kpts(nks, wrap_around, with_gamma_point, scaled_center, space_group_symmetry, time_reversal_symmetry)`. Symmetry raises `NotImplementedError` pointing at `pbc.symm`.
  - `tot_electrons(nkpts)`, `atom_charges()` (int32), `atom_pseudo(ia)` (dict/None), `energy_nuc()`, `ewald(ew_eta, ew_cut)`, `nimgs()`
  - resolved `mesh`/`rcut` (via `try_mesh`/`try_rcut`)
- **Evaluation:**
  - `pbc_intor` / `intor(intor, comp, hermi, kpts, kpt)`
  - `pbc_eval_gto` / `eval_gto(eval_name, coords, kpts, kpt)`
  - Both return one array for `kpts=None`/`kpt`, or a list of per-k arrays (the 20-07 form). Shapes are `(a,b)` or `(comp,a,b)`, read with `BufOrder::F` over `[a,b,comp]` and permuted, with no arithmetic. Blocks are real when gamma, as upstream does.
- **Free functions:** `M`, `make_kpts`, `get_kconserv` (int32 `(nk,nk,nk)`), `band_path(cell, lattice=None, npoints)` → `KPath` (`kpts`, `scaled_kpts`, `x`, `tick_x`, `tick_labels`), `band_path_from_segments`, `detect_lattice`, `super_cell`, `cell_plus_imgs`, `get_coulG`, `dumps`, `loads`, `pack` (dict), `unpack`. `__reduce__` pickles via `loads`.
- **`bridge::extract_cell_from_pyany(py, &Bound<PyAny>) -> PyResult<pyscf_pbc_gto::Cell>`** tries, in order:
  1. a native `Cell`;
  2. a JSON string;
  3. an upstream-shaped Cell, rebuilt from its Bohr `lattice_vectors()`, `atom_coords()` and `_atom` plus basis, pseudo and flags (`rcut` is re-estimated);
  4. any `.cell` attribute.
- **Overlay:** `python/pyscf/pbc/gto/__init__.py` re-exports the native objects, keeps `extend_path`, and lazily falls through to upstream `cell`/`basis`/`pseudo`/`neighborlist` names.

## Task 6 — refusals raise (`PyscfRsRuntimeError`, `args[1] == "NotYetImplemented"`)

| refusal | reached how |
|---|---|
| `get_coulG` `dimension == 1` (`coulg.rs:180`) | **Unreachable through a built cell.** `Cell::build` refuses `dimension=1` without `inf_vacuum` first (cell.rs, "Uniform grids for dimension=1"). The test asserts that build refusal. With `inf_vacuum`, get_coulG takes the 3-D kernel. |
| `vcut_sph` `dimension < 3` | `get_coulG(slab, exx='vcut_sph')` raises NotYetImplemented |
| `vcut_ws` `dimension < 3` | `get_coulG(slab, exx='vcut_ws', kpts=…)` raises |
| `super_cell` + `space_group_symmetry` | raises NotYetImplemented |
| `pbc_intor` outside family / spinor | `int1e_rinv` and `int1e_ovlp_spinor` both raise NotYetImplemented. The spinor name is caught by the family check (`pbc_intor.rs:357`) before the spinor branch (`:381`), which is unreachable via `add_suffix`. |

## Verification (2026-09-14, rebuilt `.so` 07:57/08:00)

| command | result |
|---|---|
| `.venv/bin/pytest python/pyscf/tests/test_pbc_cell.py -q -p no:cacheprovider` | **42 passed in 25.19s**, exit 0 |
| `cell.make_kpts([2,2,2])` bitwise = Rust | yes: 3 cells × 4 option sets, `.view(uint64)` equality against `make_kpts`+`vec_mat` re-evaluated op-for-op in Python floats; the free function is bitwise the same |
| `nao_nr()` / `natm` via Deref | `'nao_nr' not in type(cell).__dict__` and `cell.nao_nr() == cell.mol.nao_nr()` |
| `hasattr(cell, "get_hcore")` | `False`; `Cell.__doc__` names `FFTDF` / `get_hcore` |
| five refusals | all raise (table above) |
| vs vendored 2.12.1 | nao, charges, nelec exact; vol rel ≤1e-12; `b`, kpts ≤1e-12 (diamond/Si/He) |
| `cargo run -p xtask --bin check-catch-unwind` | exit 0 (PASS, 825 files) |
| `check-forbid-lazy-static` / `check-dependency-wall` / `check-orphan-modules` | exit 0 / 0 (both PASS lines) / 0 (434 files) |
| `grep -rn pyo3 crates/pyscf-pbc-*/Cargo.toml` | no output, exit 1 |
| `pytest test_pbc_identity_gate.py` | **7 passed, 11 failed**. Passed: `pbc.gto.{Cell,M}` and `pbc.df.{FFTDF,AFTDF,GDF,MDF,RSDF}`. Failed: scf ×4, dft ×4, symm, mp, cc (20-12…20-15). |

## Deviations

- **D1 — "bitwise vs Rust" is established without a Rust binary.** Building a Rust test or example that prints values re-unifies features and rebuilds the libxc tree (memory `gate-target-dir-lto-spelling`). `make_kpts` is instead checked against an op-for-op re-evaluation of the Rust arithmetic, which is exact IEEE and has no FMA. Integrals are checked for exact repeat determinism and cross-route identity.
- **D2 — `make_kpts_default`** is not bound under that name. It is `make_kpts` with default kwargs, bitwise identical. **`eval_ao_kpts`** is bound as `Cell.pbc_eval_gto`/`eval_gto`, the upstream names. `pyscf.pbc.dft.numint.eval_ao_kpts` is 20-13's.
- **D3 — `get_coulG` and `get_kconserv` live in `pbc.gto`.** Upstream has them in `pbc.tools` / `pbc.lib.kpts_helper`. 20-14 should re-export the same objects, not rebind them.
- **D4 — accepted inputs are narrower than upstream's.** Explicit shell lists for `basis` and per-element `pseudo` dicts raise `NotImplementedError`. The ECP input is not bound.
- **D5 — the upstream fallthrough in the `pbc.gto` overlay is silent (lazy `__getattr__`).** Announcing it is 20-17's decision.
- **D6 — upstream parity:** `pbc_intor` with a `(nk,3)` `kpts` returns a list, not a stacked ndarray (20-07 contract).
