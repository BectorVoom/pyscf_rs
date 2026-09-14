# 20-10 SUMMARY — `pbc.df`: one `PyPeriodicDf` over FFTDF/AFTDF/GDF/MDF/RSDF

**Shipped:** 2026-09-14. All seven tasks are done. It shares 20-09's two release builds: r1 53.69 s, r2 15.98 s (pyscf-py only); see 20-09-SUMMARY.

## Task 1 — tests first

`python/pyscf/tests/test_pbc_df.py` has 23 cases. The fixture is He-fcc `sto-3g` in Bohr, 2×2×2, mesh `[15,15,15]` (the 20-CONTEXT §2 fixture). The kpts_band gate uses the default mesh, as `band_kpoints.rs` does.

## Tasks 2–3 — one wrapper, five constructors, three aliases (`crates/pyscf-py/src/pbc/df.rs`)

- **`PeriodicDf`** (`PyPeriodicDf`, `#[pyclass(subclass)]`) holds `py_cell: Py<PyAny>`, a `DfConfig` and `DfArc`. `DfArc` is `Arc<Fftdf|Aftdf|Gdf|Mdf|Rsdf>`, or `Arc<dyn PeriodicDf>` from the factory. Every method lives on this base class.
- **`FFTDF`, `AFTDF`, `GDF`, `MDF`, `RSDF`** are `#[pyclass(extends=PyPeriodicDf, subclass)]` constructors `(cell, kpts=None)`, so `isinstance(df, GDF)` discriminates routes. `cell` may be anything `extract_cell_from_pyany` accepts, and `mydf.cell is cell`.
- **Aliases.** `PWDF`, `DF` and `RSGDF` are module attributes bound to the **same type objects** (`m.add("DF", m.getattr("GDF"))`).
- **Mutability.** Every configuration setter rebuilds a fresh builder from `DfConfig` and swaps the `Arc`, so a stale `cderi` can never be served. The setters are: `kpts`, `mesh` (FFT/AFT/MDF; GDF/RSDF raise `NotImplementedError`), `max_memory`, `auxbasis`, `exp_to_discard`, `_j_only`, `_prefer_ccdf`, `_cderi`, `_cderi_to_save`. Kind-inapplicable attributes raise `AttributeError`, so `hasattr` is False.
- `name` is the `traits.rs:132` discriminator (`"RSGDF"` for RSDF) and is used in `repr()`.

### Ownership contract for 20-12 / 20-13 (drivers)

```rust
pub struct SharedDf(pub Arc<dyn PeriodicDf>);            // impl PeriodicDf, forwards all 17 methods
pub fn extract_df(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn PeriodicDf>>  // TypeError if not a PeriodicDf
impl PyPeriodicDf { pub fn shared(&self) -> Arc<dyn PeriodicDf>; pub fn boxed(&self) -> Box<dyn PeriodicDf> }
pub fn pbc_df_to_py(err: PbcDfError) -> PyErr
```

A driver stores the **Python object** (`with_df: Py<PyAny>`, settable, so `mf.with_df is mydf`). At every `kernel`/`get_jk` entry it calls `extract_df(with_df.bind(py))`, then either `Krhf::from_df(boxed)` or `krhf.with_df = boxed` (the field is `pub`). A reassignment, or a mutation after assignment, is therefore what the next call sees.
- `SharedDf::build` delegates only when it is the sole owner; otherwise it is a no-op. That is safe because every builder is lazy behind `&self` (`Gdf::cderi`, `Mdf::gdf`, `Fftdf::ao_kpts`, the Aftdf G-cache).
- The private hook `_native.pbc.df._driver_handle_get_jk(with_df, dm, exxdiv)` exercises exactly this path.

## Tasks 4–5 — trait surface, `get_hcore`, `get_jk` kwargs

- `build()` is eager and returns self: Gdf `cderi()`, Mdf `gdf()+aftdf()+resolved_mesh()`, Fftdf `ao_kpts`, Aftdf `build` when uniquely owned.
- `get_nuc` / `get_pp` / `get_hcore(kpts=None)` return one complex `(nao,nao)` array for `None` or a `(3,)` k-point, and a list for `(nk,3)`. `get_hcore` is `pyscf_pbc_df::get_hcore(&dyn PeriodicDf, kpts)` (`fftdf.rs:565`).
- `get_jk(dm, hermi=1, kpts=None, kpts_band=None, with_j=True, with_k=True, omega=None, exxdiv=None, kk_symmetry=None)`:
  - The argument order is upstream's (`df.py:459`).
  - `kk_symmetry=None` means `JkOpts::kk_symmetry_default()`. `omega=0` means `None`.
  - `dm` may be `(nao,nao)`, `(nk,nao,nao)`, `(nset,nk,nao,nao)`, a list, or a list of lists. Real input is widened to complex128 by numpy (exact) and then read with 20-07's `to_kdms(.., BufOrder::C)`.
  - Output mirrors the input: a single array, a per-k list, or a per-set list of lists. A half that was not requested is `None`.
  - The GIL is released (`py.detach`) around the Rust work.

## Task 6 — `density_fit`, persistence, ao2mo

- `density_fit(cell, kpts=None, kind='GDF', auxbasis=None, mesh=None)` calls `pyscf_pbc_df::density_fit` and returns an instance of the matching class.
- `_cderi_to_save = path` + `build()` writes HDF5. `_cderi = path` → `Gdf::load_cderi` reads it and never refits. The getter returns the written or loaded path.
- `get_naoaux()`.
- `sr_loop(kpti_kptj=None, compact=True)` returns `[(LpqR, LpqI, sign)]`.
- `get_eri(kpts=None, compact)`, `ao2mo(mo_coeffs, kpts=None, compact)`, and `ao2mo_7d(mo_coeff_kpts, factor)` → `(nk,nk,nk,n0,n1,n2,n3)` complex. All go through trait methods, so all five builders are covered.
- Overlay: `python/pyscf/pbc/df/__init__.py` re-exports the native names and falls through lazily to upstream submodules.

## Task 7 — refusals bound as raising

| refusal | test |
|---|---|
| `exp_to_discard` GDF / MDF (`gdf/mod.rs:264`, `mdf/mod.rs:128`) | setter accepted; `build()` raises NotYetImplemented |
| cartesian fused auxcell (`fuse.rs:175/422`) | `GDF(cart cell).build()` raises NotYetImplemented |
| `get_jk(omega)` GDF / MDF (`gdf/jk.rs:673`, `mdf/mdf_jk.rs:132`) — **still refusing in this tree** (20-05 is being done in a separate copy) | `get_jk(dm, omega=0.3)` raises NotYetImplemented. If 20-05 lands, this test must be restated, not deleted. |
| `incore/auxcell.rs:224` `drop_eta` | reached only through `exp_to_discard` above |

## Verification (2026-09-14)

| command | result |
|---|---|
| `.venv/bin/pytest python/pyscf/tests/test_pbc_df.py -q -p no:cacheprovider` | **23 passed in 272.26s**, exit 0. The box was loaded (load average 58). 244 s of that was `AFTDF.get_pp` on diamond, since switched to FFTDF (re-run: 1 passed, 2.56 s). |
| FFTDF `get_jk` bitwise | identical bits for: a fresh builder; `density_fit` (`Box<dyn>`); the driver handle (`extract_df`→`SharedDf`); the ndarray/tuple/complex-list spellings; the `[dm,dm]` nested and 4-D forms; `with_k=False` |
| GDF `get_jk` bitwise | repeat, driver handle, and a fresh `density_fit('GDF')` (no save) all identical to the saved builder |
| `get_hcore` bitwise | `== get_nuc(kpts)[k] + cell.pbc_intor('int1e_kin', kpts)[k]` on all 8 k-points (the Rust body's own element-wise add) |
| `PWDF is AFTDF`, `DF is GDF`, `RSGDF is RSDF` | True; overlay `pyscf.pbc.df.X is pyscf._native.pbc.df.X` |
| GDF `_cderi` HDF5 round trip | reloaded `GDF` `get_jk` **bitwise** equal; `get_naoaux` equal; `sr_loop` non-empty |
| `kpts_band` vs vendored 2.12.1 | **\|dvj\| = 1.394049e-09**, \|dvk\| = 1.981526e-11, asserted `< 2e-9`. This reproduces `band_kpoints.rs`'s 1.394e-09 through the binding. |
| `mf.with_df = AFTDF(cell,kpts)` after construction | the holder is reassigned FFTDF→AFTDF and the next handle sees AFTDF. `with_df.mesh=[11]*3` after assignment is seen. Non-DF raises TypeError. |
| `cargo run -p xtask --bin check-catch-unwind` | exit 0 |
| `check-dependency-wall` | exit 0 (both PASS). No `pyscf-pbc-*` crate touched. |

## Deviations

- **D1 — one wrapper, five thin subclasses.** Binding all five names to one class would make `FFTDF is GDF` true and break upstream's `isinstance(with_df, GDF)` dispatch. The trait-object wrapper is still single.
- **D2 — "bitwise vs Rust" is established by determinism and route identity**, not a Rust-printed reference. A Rust reference binary would rebuild the libxc tree.
- **D3 — upstream parity gaps:**
  - one-body and `get_jk` results are always complex (upstream returns `.real` at gamma);
  - `(nk,…)` results are lists (the 20-07 contract);
  - the GDF/RSDF `mesh` setter raises instead of mapping onto `rs_mesh`;
  - `MDF._prefer_ccdf` defaults to `True` in this port (`mdf/mod.rs:100`), upstream `False`.
- **D4 — `Factory` variant.** A `density_fit` object that is not reconfigured keeps the factory's `Box<dyn>`. `max_memory` and `_prefer_ccdf` read as `AttributeError` on it until a setter rebuilds it concretely.
- **D5 — no pbc-crate API gap blocked this plan.** Two refusals are unreachable from Python: `coulg.rs:180` and the `pbc_intor` spinor branch (see 20-09).
