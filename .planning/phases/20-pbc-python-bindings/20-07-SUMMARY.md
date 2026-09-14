# 20-07 SUMMARY — complex, k-resolved NumPy boundary (`numpy_io.rs`)

**Shipped:** 2026-09-14. All four tasks done, TDD order (the test failed to compile before the implementation existed). Every verification command is green, with the deviations recorded below.

## Task 1 — the failing test first

`crates/pyscf-py/tests/complex_roundtrip.rs` has 9 tests, and every comparison is `to_bits()` on both planes:
F-order and C-order round trips (with column-/row-major index checks), a C-contiguous 3-D input normalised to F,
**non-contiguous** views (`a.T`, `a[::2]`, `a[:,1:5]`, `a[::-1, ::3]`) read by logical index in both orders plus an explicit
`T[i,j] == A[j,i]` check, length/plane mismatch → `Err`, `KMats` of 8 blocks → 8 arrays, k-symmetry blocks
`(5,3)` + `(5,2)` (and proof that `ndarray::stack` refuses them), `KDms` 2×4 nested, and a byte-for-byte layout check against the `gto.rs::intor_spinor`
`IxDyn(&shape).f()` precedent (same strides).

RED: `cargo check -p pyscf-py --test complex_roundtrip` → `E0432 unresolved imports pyscf_py::numpy_io::{BufOrder, …}` + `pyscf_algebra`.

## Tasks 2–3 — the helpers (exact signatures; later plans depend on them)

```rust
pub enum BufOrder { C, F }                         // layout of the flat planes
pub type KMatsShapes = Vec<Vec<usize>>;            // shapes[k]
pub type KDmsShapes  = Vec<KMatsShapes>;           // shapes[set][k]

// pyo3-free core
pub fn ctensor_to_array(t: &CTensor, shape: &[usize], order: BufOrder) -> Result<ArrayD<Complex64>, String>
pub fn array_to_ctensor(view: ArrayViewD<'_, Complex64>, order: BufOrder) -> CTensor
pub fn kmats_to_arrays(kmats: &[CTensor], shapes: &[Vec<usize>], order: BufOrder) -> Result<Vec<ArrayD<Complex64>>, String>
pub fn arrays_to_kmats(views: &[ArrayViewD<'_, Complex64>], order: BufOrder) -> (Vec<CTensor>, KMatsShapes)
pub fn kdms_to_arrays(kdms: &[Vec<CTensor>], shapes: &[Vec<Vec<usize>>], order: BufOrder) -> Result<Vec<Vec<ArrayD<Complex64>>>, String>
pub fn arrays_to_kdms(views: &[Vec<ArrayViewD<'_, Complex64>>], order: BufOrder) -> (Vec<Vec<CTensor>>, KDmsShapes)

// PyO3 wrappers
pub fn ctensor_to_pyarray<'py>(py: Python<'py>, t: &CTensor, shape: &[usize], order: BufOrder) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>>
pub fn to_ctensor<'py>(arr: PyReadonlyArrayDyn<'py, Complex64>, order: BufOrder) -> PyResult<(CTensor, Vec<usize>)>
pub fn kmats_to_pylist<'py>(py: Python<'py>, kmats: &[CTensor], shapes: &[Vec<usize>], order: BufOrder) -> PyResult<Bound<'py, PyList>>
pub fn to_kmats<'py>(obj: &Bound<'py, PyAny>, order: BufOrder) -> PyResult<(Vec<CTensor>, KMatsShapes)>
pub fn kdms_to_pylist<'py>(py: Python<'py>, kdms: &[Vec<CTensor>], shapes: &[Vec<Vec<usize>>], order: BufOrder) -> PyResult<Bound<'py, PyList>>
pub fn to_kdms<'py>(obj: &Bound<'py, PyAny>, order: BufOrder) -> PyResult<(Vec<Vec<CTensor>>, KDmsShapes)>
pub fn register_selftest(m: &Bound<'_, PyModule>) -> PyResult<()>   // private _native._roundtrip_* hooks
```

Behaviour: the wrappers error on a mismatch and never truncate (ValueError). A dtype other than complex128 raises TypeError from extraction and is never cast.
`to_kmats` / `to_kdms` accept a `list` or `tuple` only. A stacked ndarray raises a TypeError that tells the caller to pass `list(a)`.
`KMats`/`KDms` are `pyscf_pbc_scf::types` aliases (`Vec<CTensor>`, `Vec<Vec<CTensor>>`), and the signatures take the
same types as slices. `pyscf-algebra` was added as a direct dep of `pyscf-py` for `CTensor`. It was already in the tree transitively.

## Task 4 — the wall

No `pyscf-pbc-*` crate was touched. No new workspace dep: `numpy::Complex64` is num-complex's.

## Verification (2026-09-14)

| command | result |
|---|---|
| `CARGO_TARGET_DIR=target/py CARGO_PROFILE_RELEASE_LTO=false cargo test --release -p pyscf-py --test complex_roundtrip` | **exit=0**, `test result: ok. 9 passed; 0 failed` (build `Finished release in 65m 00s`, see D3) |
| `.venv/bin/pytest python/pyscf/tests/test_complex_boundary.py -q -p no:cacheprovider` | **18 passed in 0.08s** |
| `to_bits()` (Rust) / `.view(np.uint64)` (Python) equality, no epsilon | yes, both planes |
| non-contiguous input bitwise, no silent transpose | yes (`a.T`, `a[::2]`, `a[:,1:5]`, `a[::-1,::3]`, `asfortranarray`) |
| `cargo run -p xtask --bin check-dependency-wall` | **exit=0** (`PASS — cubecl-* containment intact`, `PASS — PyO3 wall intact (D-PBC-14)`) |
| `grep -rn 'pyo3' crates/pyscf-pbc-*/Cargo.toml` | no output, exit=1 |
| `check-catch-unwind`, `check-forbid-lazy-static` | exit=0 / exit=0 |

## Deviations

- **D1 — explicit `BufOrder` + returned shape (plan: `ctensor_to_pyarray(py, &CTensor, shape)` hard-wired to `.f()`).**
  `pyscf_pbc_scf/src/types.rs:5-10` keeps `nao×nao` k-matrices **row-major** but MO coefficients **column-major**.
  A single F-only writer would silently transpose every `KMats` Fock/density block, which is the exact failure the plan
  warns about. So both directions name the layout. `BufOrder::F` is byte-identical to the `gto.rs` precedent (a test proves it).
  `CTensor` has no shape, so the `to_*` readers return it.
- **D2 — test split (orchestrator fallback).** No `crates/pyscf-py/tests/*.rs` hosts an interpreter. All of them
  are pyo3-free source/Rust-side checks, and the default `abi3-py310` feature enables `pyo3/extension-module`, so there is no libpython
  link. So the Rust test drives the pyo3-free core (it links fine because it references no Python symbols). The numpy wrappers are
  covered by the new `python/pyscf/tests/test_complex_boundary.py` through private root hooks
  `_native._roundtrip_ctensor / _roundtrip_kmats / _roundtrip_kdms / _planes` (registered by `numpy_io::register_selftest`).
- **D3 — target dir.** The plan named `$HOME/.cargo-target-gate` (debug). Used `target/py` + `LTO=false` per EXECUTION-NOTES §2.
  `cargo test` forces `panic=unwind` over the workspace's release `panic = "abort"`, so every dep (all 268 libxc
  kernels) rebuilds once for the test binary: **65 min**. It does not disturb maturin's artifacts.
