# Stack Research

**Domain:** Quantum-chemistry library — pure-Rust core + PyO3 bindings; cubecl as the sole compute primitive
**Researched:** 2026-05-09
**Confidence:** HIGH for locked choices and crates verified against crates.io today; MEDIUM for tensor-contraction recommendation and HDF5 chkfile compatibility (need empirical validation in an early phase); LOW for cubecl-linalg viability (it is stale and probably unusable as-is).

> All version numbers below were verified against the crates.io JSON API on 2026-05-09. They are not from training data. Where a sibling crate (`cintx`, `libxc_rs`, `xcfun_rs`) already pins a version, **pyscf_rs MUST match that pin** — the four crates form a single cubecl ABI surface and a divergent cubecl version means link-time/runtime incompatibility.

---

## 1. Recommended Stack at a Glance

| Layer | Recommendation | Version | Confidence |
|---|---|---|---|
| Edition / MSRV | `edition = "2024"`, `rust-version = "1.92"` | matches `xcfun_rs` workspace | HIGH |
| Compute kernels (locked) | `cubecl` + per-backend runtimes | `=0.10.0` (lockstep) | HIGH |
| Dense host linear algebra | `faer` | `0.24.0` | HIGH |
| Multi-dim arrays / tensor layout | `ndarray` | `0.17.2` | HIGH |
| Tensor contractions (post-SCF) | `cubecl` kernels (in-house) on top of `ndarray` views; **NOT** `cubecl-linalg` | — | MEDIUM (alternative `candle-core` listed) |
| Python bindings | `pyo3` + `numpy` + `pyo3-build-config` | `=0.28.3` / `=0.28.0` | HIGH |
| Python wheel builder | `maturin` | `1.13.1` | HIGH |
| HDF5 chkfile I/O | `hdf5-metno` (with `hdf5-sys/static`) | `0.12.4` | MEDIUM |
| Special functions (Boys, erf, Γ, Bessel) | `puruspe` + in-house Boys via cubecl | `0.4.4` | MEDIUM |
| Numerics traits | `num-traits`, `num-complex` | `0.2.19`, `0.4.6` | HIGH |
| Errors (library) | `thiserror` | `=2.0.18` | HIGH |
| Errors (apps/xtask/bench) | `anyhow` | `1.0.102` | HIGH |
| Logging | `tracing` (+ `tracing-subscriber` in app boundaries) | `=0.1.44` / `=0.3.23` | HIGH |
| Configuration | env vars first; `figment` only if a layered config emerges | `figment 0.10.19` | MEDIUM |
| Geometry-opt driver | `argmin` (+ `argmin-math`, `finitediff`) for BFGS/L-BFGS | `0.11.0` | MEDIUM |
| Bench harness | `criterion` | `=0.8.2` | HIGH |
| Property tests | `proptest`, `rstest` | `=1.11.0`, `=0.26.1` | HIGH |
| Snapshot / numerical-regression tests | `insta` (text), `approx` (float compare) | `1.47.2`, `=0.5.1` | HIGH |
| Test runner | `cargo-nextest` | `0.9.133` | HIGH |
| Coverage | `cargo-llvm-cov` | `0.8.6` | HIGH |
| Byte-cast / GPU ABI | `bytemuck` | `1.25.0` (sibling pin) | HIGH |
| One-shot init / static state | `once_cell` (or `OnceLock` from std) | `1.21.4` | HIGH |
| Cold orchestration parallelism | `rayon` (cold paths only — D-06 in sibling crates forbids it on hot paths) | `1.12.0` | HIGH |
| Stub-file generation for Python typing | `pyo3-stub-gen` | `0.22.2` | MEDIUM |

---

## 2. Why each block (per-question rationale)

### 2.1 Linear algebra / tensor library

**Recommendation:**
- `ndarray 0.17.2` — array container, slicing, views, the data type that crosses the PyO3 boundary into NumPy.
- `faer 0.24.0` — every dense BLAS/LAPACK-style operation that runs on the host: eigensolve (Hermitian and general), Cholesky (DIIS, density-fit metric), LU, QR, SVD, GEMM at small/medium dimensions, complex (`c64`) for GHF.
- `cubecl 0.10.0` — every operation that is rank-3+, batched, or large enough that GPU offload pays for itself: 4-index ERIs, batched GEMM in CC, AO→MO transformation, DFT grid evaluation. Hand-written `#[cube]` kernels live in `pyscf-{module}-cubecl` crates, mirroring `cintx-cubecl` and `xcfun-gpu`.

**Why faer over the alternatives**:
- **Pure Rust, no system BLAS link.** That is the entire reason this rewrite exists — `pip install` works on every platform without MKL/OpenBLAS hunting. faer ships its own `gemm` and `nano-gemm` core; verified on crates.io 2026-01-26. Performance is competitive with OpenBLAS for medium dense matrices on x86-64 (faer's design target). For SCF-sized Fock/overlap matrices (a few hundred to a few thousand AOs), this is exactly the workload faer optimizes.
- **Eigendecomposition is first-class** (Hermitian + general, including complex) — SCF needs `eigh` on every iteration; this is non-negotiable.
- **Complex scalar support** (`c32`, `c64`, `cx128`) — required for GHF.
- **Active maintenance** (1.27 M recent downloads, last release 2026-01-26).

**Why ndarray (not nalgebra) for the array container**:
- `ndarray 0.17.2` is the de facto rank-N array in Rust scientific code. It interoperates trivially with the `numpy` crate (zero-copy `PyArray<T, D>` ↔ `ArrayView<T, D>`). nalgebra is heap-of-stack-shaped, statically-typed-dim, optimized for small fixed dims and 3D-graphics math; it is the wrong shape for "AO×AO×AO×AO ERI tensor."
- faer integrates with ndarray via `faer-ext 0.7.1` (`faer::Mat` ↔ `ndarray::Array2` zero-copy where layout allows).

**Why hand-write tensor contractions in cubecl rather than reach for a library**:
- `cubecl-linalg 0.5.0` (last published 2025-04-23) is **pinned to an old cubecl version (~0.5)** and has not tracked the cubecl 0.10 release; the version-skew is over a year. It is effectively unmaintained for our purposes. **Do not use.** If the cubecl team revives it in lockstep with future cubecl releases, revisit.
- `candle-core 0.10.2` is a viable *alternative* tensor library (HuggingFace's, 2.2 M recent downloads, actively maintained). But it brings its own backend abstraction (CPU + CUDA + Metal) that overlaps and competes with cubecl. Pulling it in would create a **second** GPU runtime in the build, doubling cold-start cost and complicating runtime selection. Use only if a specific operation (e.g., a fast batched GEMM kernel) cannot be expressed in cubecl in time for v1, and only behind an opt-in feature.
- The approach the sibling crates already validate: each compute module owns its `pyscf-{name}-cubecl` crate with `#[cube]` kernels written against the cubecl 0.10 API, plus a thin `runtime` crate that does backend selection. cintx's 4-index ERI work and xcfun_rs's batched functional-evaluation work are the proof-of-concept that this scales.

**What NOT to use (linear algebra)**:

| Avoid | Why | Use instead |
|---|---|---|
| `ndarray-linalg 0.18.1` | Requires linking against a system BLAS/LAPACK (OpenBLAS, MKL, or netlib). Defeats the entire "no C deps at install time" goal that motivated the rewrite. | `faer 0.24.0` |
| `nalgebra 0.34.2` for SCF matrices | Designed for small statically-sized matrices and graphics; eigendecomposition path goes through `nalgebra-lapack` which again pulls system LAPACK. Const-generic dim is the wrong abstraction for chemistry where dim is runtime. | `faer` for dense ops; `ndarray` for the container |
| `nalgebra-lapack 0.27.0` | Same system-LAPACK dependency. | `faer` |
| `cubecl-linalg 0.5.0` | Stale (over a year behind cubecl 0.10), pinned to old cubecl. | Hand-rolled `#[cube]` kernels in `pyscf-{module}-cubecl` |
| `candle-core` as the primary tensor lib | Brings a second GPU runtime; overlaps cubecl. | cubecl. Reconsider only as an opt-in feature for specific kernels. |
| `matrixmultiply 0.3.10` directly | Pure-Rust GEMM, but it's a building block underneath faer's `gemm`. Using it directly reinvents what faer already wraps. | `faer` (which uses it transitively where appropriate) |
| `peroxide 0.41.0` | Comprehensive numerical library, but overlaps every choice above and is far less battle-tested at scale (85 K recent downloads vs. faer's 1.27 M). | Pick targeted libs (`faer`, `ndarray`, `argmin`). |

### 2.2 cubecl ecosystem

**Locked version: `=0.10.0` across all five cubecl crates.** This must move in lockstep with `cintx`, `libxc_rs`, `xcfun_rs` — they share the cubecl JIT runtime ABI, and a single divergent pin breaks everything.

| Crate | Version | When to enable |
|---|---|---|
| `cubecl` | `=0.10.0` | Always (default `cpu` feature) |
| `cubecl-cpu` | `=0.10.0` | Default in `default` feature; ships with the wheel |
| `cubecl-cuda` | `=0.10.0` | Optional, behind `cuda` feature |
| `cubecl-hip` | `=0.10.0` | Optional, behind `rocm` feature |
| `cubecl-wgpu` | `=0.10.0` | Optional, behind `wgpu` feature; also covers Apple Metal (`metal` is a feature alias for `wgpu` — verified in sibling `xcfun-gpu` and `cintx-cubecl` Cargo.tomls; `cubecl-metal` does not exist on crates.io) |
| `cubecl-runtime` | `=0.10.0` | Pulled in transitively |

**Pre-1.0 instability — be explicit**: cubecl is **alpha** per its own README, currently at `0.10.0` released 2026-05-07. Breaking changes between minor versions are routine (track record: `0.5 → 0.9 → 0.10` over 2025–2026). Pin with `=0.10.0` (exact-version operator), audit every cubecl bump against the cintx/xcfun_rs/libxc_rs ecosystem, and **never** bump cubecl in pyscf_rs alone.

**Runtime selection idiom** (mirrors `cintx-runtime` / `xcfun-gpu` `auto_backend()`):

1. `pyscf-runtime` crate exposes a `BackendKind` enum (`Cpu`, `Cuda`, `Wgpu`, `Rocm`, `Metal`) gated by feature flags.
2. `auto_backend()` picks in priority order: env var override (`PYSCF_BACKEND=cuda`) → CUDA → ROCm → WGPU/Metal → CPU.
3. Generic compute functions take `<R: cubecl::Runtime>` so backend selection happens at the dispatch site, not inside the kernel.
4. Every `pyscf-{module}-cubecl` crate exports `dispatch_kernel<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>, …)` and the `runtime` crate routes to the right `R`.

**Portable kernels** — the patterns to copy from siblings:
- `#[cube]` for the kernel body; `CubeLaunch` and `CubeType` derives for argument types.
- Use `bytemuck 1.25` for `Pod`/`Zeroable` on every kernel struct that crosses the host/device boundary.
- Use `cubecl::prelude::*` `Tensor`/`Slice` types — never raw pointers.
- Reductions via `cubecl::reduce` primitives; numerical-stability for sums must use Kahan or pairwise (chemistry energies are reported to µHartree; naive summation orders of 10⁵ ERI contributions accumulates ~10⁻⁹ error which crosses the bit-exact threshold for some assertions). The sibling crate `xcfun-eval` already has this pattern.
- **Autotuning**: cubecl 0.10 supports it (per upstream README). Defer enabling it until a concrete hot path warrants the complexity — autotune introduces nondeterminism and pyscf_rs has a strict "bit-exact PySCF agreement" contract.

**Source verification**: `cubecl 0.10.0` published 2026-05-07; sibling `xcfun_rs/Cargo.toml` line 50–54 pins all five crates `=0.10.0`; sibling `libxc_rs/Cargo.toml` line 9 uses `cubecl 0.10.0` with default-features off and `cpu` feature on, confirming the pattern.

### 2.3 PyO3 + maturin

**Versions (lockstep with `xcfun-py`):**
- `pyo3 = { version = "=0.28.3", features = ["extension-module", "abi3-py310"] }`
- `numpy = "=0.28.0"`
- `pyo3-build-config = "=0.28.3"` (build-dep, only if needed for advanced linker probing)
- `maturin 1.13.1` (build tool, not a crate dep)

**Why these versions**: `xcfun_rs/crates/xcfun-py/Cargo.toml` already pins `pyo3 =0.28.3` and `numpy =0.28.0`. Match exactly. PyO3 0.28 is the current stable line as of 2026-04-02; 0.28.3 is the latest patch.

**abi3 vs version-specific wheels — recommendation: abi3-py310**
- `abi3-py310` produces **one wheel per OS/arch** that supports CPython 3.10 + 3.11 + 3.12 + 3.13 + 3.14. PySCF's own pyproject already targets ≥3.8 but the upstream is migrating to 3.10+; aligning with `abi3-py310` matches `xcfun-py` and is the lowest-friction install story.
- Trade-off: abi3 forbids using a few unstable Python C API entry points. PyO3 0.28 papers over this transparently for normal use; the only common gotchas are (a) some `numpy` array-protocol fast paths that are not abi3-stable and (b) sub-interpreter support. Neither matters for pyscf_rs's surface.
- **Avoid**: building one wheel per CPython minor version. That triples CI time and storage and gives users no benefit.

**Conventions for exposing Python classes that hold Rust-owned arrays without copies**:

The `numpy` crate provides `PyArray<T, D>` / `PyReadonlyArray<T, D>` / `PyReadwriteArray<T, D>`, which are zero-copy views of NumPy buffers. The pattern (validated in xcfun-py and rust-numpy docs):

1. `#[pyclass]` Rust struct (`PyMole`, `PySCF`) holds the Rust-owned `ndarray::Array<f64, _>` (or a `faer::Mat<f64>`).
2. To **return** a tensor to Python without copying, allocate a `PyArray::from_owned_array(py, array)` — moves ownership to Python; the data is reused.
3. To **accept** a NumPy array from Python, take `PyReadonlyArray2<f64>`, call `.as_array()` to get an `ArrayView2<f64>` that aliases the NumPy buffer. No copy.
4. For arrays that came from cubecl GPU buffers, copy back to host (`client.read_async(handle).await`) once and wrap in a `PyArray`. There is no zero-copy path from device memory to NumPy — and there shouldn't be, because NumPy semantics are CPU-only.
5. Buffer-protocol direct access (PEP 3118) is exposed automatically by `PyArray`; no extra work.

**GIL handling for long-running compute**:
- Wrap every compute call in `py.allow_threads(|| …)`. The GIL is released for the duration; other Python threads can run. This is the single most impactful Python-bindings perf rule.
- `with_gil` is for re-acquiring the GIL inside a non-Python thread (rare in our use). The default `#[pyfunction]` entry already holds the GIL.
- **Do not** hold `PyArray` references across `allow_threads` boundaries — extract `ArrayView`s up front.

**PySCF-as-oracle from Rust tests** — answer to the explicit question: **call PySCF in-process via `pyo3::Python::with_gil`**. Subprocess launches add 1–3 s of import time per test (PySCF imports SciPy, which is heavy), and the whole point of in-process oracle is fast feedback. Pattern:

```rust
use pyo3::prelude::*;
use numpy::PyReadonlyArrayDyn;

fn pyscf_rhf_energy(atom: &str, basis: &str) -> PyResult<f64> {
    Python::with_gil(|py| {
        let pyscf = py.import("pyscf")?;
        let gto  = pyscf.getattr("gto")?;
        let scf  = pyscf.getattr("scf")?;
        let mol = gto.call_method1("M",
            ((atom,), ("basis", basis)))?;  // build kwargs properly
        let mf  = scf.call_method1("RHF", (mol,))?;
        let e: f64 = mf.call_method0("kernel")?.extract()?;
        Ok(e)
    })
}
```

In CI this requires `python3` + `pip install pyscf` + the `extension-module` feature off in test builds (otherwise PyO3 can't initialize a Python interpreter). The standard fix is a `pyscf-oracle` dev-dep crate that uses `pyo3` *without* `extension-module` and is only built under `#[cfg(test)]`.

**What NOT to use (PyO3 surface)**:

| Avoid | Why | Use instead |
|---|---|---|
| `pyo3 0.x` for x < 0.28 | Breaking-change cycle landed at 0.21+; trying to sync with a sibling crate's 0.28 across versions causes ABI conflicts | `=0.28.3` |
| `pyo3-async-runtimes 0.28` initially | Adds a Tokio/asyncio bridge; pyscf_rs is sync compute, no benefit, just complexity | Skip until a concrete async use case appears (e.g. background HDF5 streaming) |
| Copying NumPy arrays into `Vec<f64>` on every call | Defeats the bindings layer | `PyReadonlyArray::as_array()` |
| Calling Python from inside `allow_threads` | The GIL has been released — calling back into Python panics or deadlocks | Re-acquire with `Python::with_gil` |

### 2.4 HDF5 / chkfile I/O

**Recommendation: `hdf5-metno 0.12.4`** with the `hdf5-sys/static` feature enabled, **for chkfile read/write only**.

**Why hdf5-metno specifically**:
- It is the **maintained fork** of the original `aldanor/hdf5-rust` crate. The original `hdf5 0.8.1` was last published 2021-11-21 and is effectively abandoned for our purposes.
- `hdf5-metno 0.12.4` was released 2026-03-23 (current).
- Provides the same high-level API (datasets, groups, attributes), thread-safe, integrates with `ndarray` for read/write of multi-dim arrays.
- Supports static linkage (`hdf5-sys/static` feature) — bundles libhdf5, builds from source via CMake at build time. **This** is the install-time pain point we accept: maturin wheels will link a static libhdf5, end-users do not need a system HDF5.

**h5py / chkfile compatibility**:
- HDF5 is a **standardized binary format**. Files written by h5py are readable by libhdf5 (and therefore by hdf5-metno) at the binary level. There is no version drift here.
- The risks are not at the file format layer but at the **convention layer** that PySCF imposes on top:
  - PySCF chkfile stores a mix of f64 arrays, NumPy structured dtypes for some metadata, Python pickles serialized into byte arrays for things that don't have an HDF5 native form (rare but real), and group-naming conventions (`/scf/mo_coeff`, `/scf/mo_energy`, `/scf/e_tot`).
  - Variable-length strings, NumPy object arrays, and pickled blobs are the usual h5py↔non-h5py interop pain points.
- **Mitigation**: write a `pyscf-chkfile` crate that **only** speaks the subset of the chkfile schema we ship for, plus a one-pass `validate_against_pyscf()` test-fixture that round-trips a chkfile through both libraries. Treat anything outside the documented schema as out-of-scope.

**Confidence: MEDIUM** — h5py interop in particular is the kind of thing that looks fine in unit tests and breaks on a user's three-year-old chkfile. Allocate a phase early in the roadmap to validate against a corpus of real PySCF chkfiles before declaring the I/O layer stable.

**Alternatives considered and rejected**:

| Avoid | Why | Use instead |
|---|---|---|
| `hdf5 0.8.1` (the aldanor original) | Last published 2021-11-21; unmaintained | `hdf5-metno 0.12.4` |
| `netcdf 0.12.0` | NetCDF is a different format; PySCF chkfiles are HDF5 | `hdf5-metno` |
| `arrow2` / Parquet | Wrong format; would require persuading the PySCF community to migrate | `hdf5-metno` |
| Re-implementing HDF5 in Rust | The format spec is enormous; not v1-scope | `hdf5-metno` with bundled libhdf5 |

### 2.5 Numerics utilities

**Recommendation:**
- `num-traits 0.2.19` — generic number traits (`Float`, `Zero`, `One`, `Num`).
- `num-complex 0.4.6` — `Complex<f64>`. Required for GHF and any post-SCF that drops to complex amplitudes. faer also re-exports `num-complex`-compatible types.
- `puruspe 0.4.4` — Pure-Rust special functions (gamma, erf, erfc, Bessel, incomplete gamma). Last published 2026-03-17. Pure-Rust is essential. ECP integrals need incomplete gamma; the Boys function `F_n(T)` is closely related.
- `libm 0.2.16` — `no_std` implementations of `f64` math (already in `libxc_rs/dev-dependencies`). Useful inside `#[cube]` kernels where `std` math may not be available on every backend.
- **Boys function** — there is **no production-quality Rust crate** for `F_n(T) = ∫₀¹ t^(2n) exp(-T t²) dt` as of 2026-05. PySCF/libcint compute it via a hybrid (downward recursion + asymptotic series + tabulated values). Plan to implement this in `pyscf-gto-cubecl` directly, in `#[cube]` form so it's portable. Reference: PySCF's `pyscf/lib/gto/g1e.c` and Helgaker–Jørgensen–Olsen, *Molecular Electronic-Structure Theory*, §9.8. **Do not** rely on `puruspe::gammq` for the Boys function in the hot path — its accuracy is fine but it's CPU-only and the Boys function is in the inner loop of every ERI batch.

**What NOT to use:**

| Avoid | Why | Use instead |
|---|---|---|
| `scilib 1.0.0` | Last published 2023-09-13, ~12 K recent downloads; small/abandoned | `puruspe` for special funcs; in-house Boys |
| `statrs 0.18.0` | Distributions and statistics, not chemistry-relevant | (n/a) |
| `gauss-quad 0.3.1` | Tempting for quadrature, but DFT grids are Lebedev + radial schemes implemented inline (see PySCF `dft/gen_grid.py`); a generic Gauss-quadrature crate doesn't help. | In-house grid module |

### 2.6 Error handling / logging / config

**Errors**:
- **Inside library crates** (`pyscf-{module}-rs`, `pyscf-{module}-cubecl`, `pyscf-runtime`, etc.): `thiserror = "=2.0.18"`. Rich, typed errors that downstream callers can match on. Matches `xcfun_rs`/`libxc_rs` convention.
- **Outside library boundary** (xtask, examples, integration tests, benches): `anyhow = "1.0.102"`. Matches `cintx`/`xcfun_rs` convention.
- **PyO3 boundary**: convert `thiserror` errors to `PyErr` via a `From<MyError> for PyErr` impl. Use `pyo3::exceptions::PyValueError` / `PyRuntimeError` as the Python-visible types.

The "sibling crates use both" comment is correct — they use them in **different layers**. Reproduce that split exactly. There is no real tension.

**Logging**: `tracing 0.1.44` (lib boundary, `default-features = false`), `tracing-subscriber 0.3.23` (binary boundaries: xtask, tests, examples). Matches `xcfun_rs/Cargo.toml` lines 44–45.

**Configuration**: PySCF uses a `.pyscf_conf.py` file plus environment variables. Replicate the env-var subset in v1 — it's enough for parity:
- `PYSCF_MAX_MEMORY` (MB), `PYSCF_TMPDIR`, `PYSCF_BACKEND` (`cpu`|`cuda`|`wgpu`|`rocm`|`metal`), `PYSCF_PRECISION` (`f64`|`f32`).
- Read with `std::env::var` directly. No need for `figment` or `config` crates yet. If a layered file/env config emerges in a later milestone, `figment 0.10.19` is the recommended pick (the only mature, current, layered-config crate; matches the design need).
- **Avoid**: re-implementing `.pyscf_conf.py` evaluation. That requires a Python interpreter at startup, which negates the install-time goal. Document the env-var-only policy.

### 2.7 Testing & benchmarking

| Need | Crate | Version | Notes |
|---|---|---|---|
| Bench harness | `criterion` | `=0.8.2` | Sibling pin (`xcfun_rs`); use `default-features = false, features = ["html_reports"]` |
| Property tests for integral symmetries | `proptest` | `=1.11.0` | (ij\|kl) = (ji\|kl) = (kl\|ij) etc. — these are exactly the invariants proptest is good at |
| Parametric tests | `rstest` | `=0.26.1` | Sibling pin |
| Float comparisons | `approx` | `=0.5.1` | Sibling pin (NB: `0.6.0-rc2` exists but no sibling has bumped) |
| Snapshot / numerical-regression tests | `insta` | `1.47.2` | Best-in-class; use for serialized output diffs (JSON dumps of MO energies, etc.). For raw float arrays, use `approx` with a tolerance, not `insta` byte-equality. |
| Test runner | `cargo-nextest` | `0.9.133` | 2–3× faster CI, parallel-safe, used by Burn/cubecl |
| Coverage | `cargo-llvm-cov` | `0.8.6` | Released 2026-05-09, current |
| Diff rendering for assertion failures | `similar` | `3.1.0` | Used internally by `insta`; rarely needed directly |

**PySCF as in-process oracle**: see §2.3 above — pattern is `Python::with_gil` calling `pyscf` directly. Implement once in a `pyscf-oracle-dev` dev-only crate (mirrors `cintx-oracle` and `xcfun-eval` patterns) and re-export to all `pyscf-*` test suites.

**Property-test recipe for ERI symmetry**:
```rust
proptest! {
    #[test]
    fn eri_8fold_symmetry(
        i in 0usize..n_ao, j in 0usize..n_ao,
        k in 0usize..n_ao, l in 0usize..n_ao,
    ) {
        let v = eri(i, j, k, l);
        prop_assert_relative_eq!(v, eri(j, i, k, l), max_relative = 1e-12);
        prop_assert_relative_eq!(v, eri(k, l, i, j), max_relative = 1e-12);
        // ...etc, all 8 permutations
    }
}
```

### 2.8 Build / CI

**Recommended GitHub Actions matrix** (realistic — not ambitious):

| Job | Runner | Purpose |
|---|---|---|
| `lint` | `ubuntu-latest` | `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` |
| `test-cpu` | `ubuntu-latest`, `macos-latest`, `windows-latest` | `cargo nextest run --workspace --features cpu` |
| `oracle` | `ubuntu-latest` with `pip install pyscf` | Runs the PySCF-as-oracle suite. Linux only — installing PySCF on Windows runners is fragile. |
| `coverage` | `ubuntu-latest` | `cargo llvm-cov --workspace --lcov` |
| `bench-smoke` | `ubuntu-latest` | `cargo bench --no-run`, plus a tiny criterion run on a fixed input to detect 2× regressions |
| `wheels` | `ubuntu-latest`, `macos-latest`, `windows-latest` | `maturin build --release --interpreter python3` (abi3, so one wheel per OS/arch) |
| `cuda` (optional, scheduled) | self-hosted runner with NVIDIA GPU, **or** GitHub `nvidia-l4-x86-64` runner (paid) | `cargo nextest run --features cuda`. GitHub Actions GPU runners exist (`L4` class) but are paid and metered; budget for nightly only, not per-PR. |
| `wgpu` | `ubuntu-latest` (lavapipe software Vulkan) | Functional smoke test; no perf claim. cubecl-wgpu can run on lavapipe — see Burn CI. |
| `rocm` | self-hosted **only** (no GitHub Actions ROCm runners exist as of 2026-05) | Defer to v1.1; ship the feature flag, validate manually. |

**Realistic outcome**:
- CPU + WGPU + Linux PySCF oracle: every PR.
- CUDA: nightly on a self-hosted runner (or scheduled paid run).
- ROCm: manual validation per release; flag as "supported but not tested in CI" until a self-hosted AMD runner is provisioned.
- Documenting this honestly in the README is more valuable than pretending all four backends are tested every PR.

**maturin wheel build**: `maturin 1.13.1`. With `abi3-py310`, one wheel covers Python 3.10+ on each `(OS, arch)` triple — so 3 wheels per release for Linux/macOS/Windows × x86_64, plus aarch64 for macOS and (optionally) Linux ARM. Use `maturin-action` GitHub Action.

### 2.9 MSRV

**Recommendation: `rust-version = "1.92"`** — exactly matches `xcfun_rs/Cargo.toml` line 37.

Rationale:
- `edition = "2024"` is required to match sibling crates.
- cubecl 0.10 published 2026-05-07 requires a recent compiler; the cintx/xcfun_rs/libxc_rs sibling pins land on 1.92.
- Setting MSRV lower than the sibling crates buys nothing — the workspace will fail to build the cubecl path on older toolchains regardless.
- Set it equal to `xcfun_rs` so the four-crate family has a single MSRV story.

**Confidence: HIGH** — verified directly from `xcfun_rs/Cargo.toml`.

### 2.10 Versions are current — verification trail

Every version in this document was fetched from the crates.io JSON API on 2026-05-09 with the request:

```
GET https://crates.io/api/v1/crates/<name>
User-Agent: pyscf_rs-research (appservice27@gmail.com)
```

Pre-1.0 crates and instability flags (explicit per the question):

| Crate | Current | Pre-1.0? | Stability note |
|---|---|---|---|
| `cubecl` and all `cubecl-*` | `=0.10.0` | **YES, alpha** | Breaking changes between minor versions are routine. Pin `=0.10.0` exactly. Audit every bump. |
| `cubecl-linalg` | `0.5.0` (last published 2025-04-23) | **YES, stale** | Pinned to old cubecl. **Do not use.** |
| `pyo3` | `=0.28.3` | YES (0.x) | Breaking-change cycle each minor; 0.28 is current. |
| `numpy` | `=0.28.0` | YES (0.x) | Locked to PyO3 minor version. |
| `faer` | `0.24.0` | YES (0.x) | Active and well-maintained but pre-1.0; API has been stable for several minor versions. |
| `ndarray` | `0.17.2` | YES (0.x) | Pre-1.0 forever, but de facto stable. |
| `hdf5-metno` | `0.12.4` | YES (0.x) | Maintained fork; the original `hdf5` is at 0.8.1 (2021) and unmaintained. |
| `argmin` | `0.11.0` | YES (0.x) | Active; pre-1.0 |
| `puruspe` | `0.4.4` | YES (0.x) | Active; pre-1.0; small lib |
| `bincode` | `3.0.0` | NO | Recently went 1.0; major bump 2025-12-16. |
| `thiserror` | `=2.0.18` | NO (2.x) | Stable. |
| `anyhow`, `tracing`, `criterion`, `proptest`, `rstest`, `insta`, `rayon`, `bytemuck`, `num-traits`, `num-complex`, `serde`, `clap` | all current | mixed | All actively maintained. |

---

## 3. Sibling-crate alignment summary

The following pins are **identical-or-die** with the sibling workspace — divergence will break builds or runtime:

| Dep | pyscf_rs pin | Sibling pin | Source |
|---|---|---|---|
| `cubecl` | `=0.10.0` | `=0.10.0` | `xcfun_rs/Cargo.toml:50` |
| `cubecl-cpu` | `=0.10.0` | `=0.10.0` | `xcfun_rs/Cargo.toml:51` |
| `cubecl-cuda` | `=0.10.0` | `=0.10.0` | `xcfun_rs/Cargo.toml:53` |
| `cubecl-hip` | `=0.10.0` | `=0.10.0` | `xcfun_rs/Cargo.toml:52` |
| `cubecl-wgpu` | `=0.10.0` | `=0.10.0` | `xcfun_rs/Cargo.toml:54` |
| `pyo3` | `=0.28.3` | `=0.28.3` | `xcfun_rs/crates/xcfun-py/Cargo.toml:26` |
| `numpy` | `=0.28.0` | `=0.28.0` | `xcfun_rs/crates/xcfun-py/Cargo.toml:27` |
| `thiserror` | `=2.0.18` | `=2.0.18` | `xcfun_rs/Cargo.toml:41` |
| `tracing` | `=0.1.44` | `=0.1.44` | `xcfun_rs/Cargo.toml:44` |
| `bytemuck` | `1.25.0` (or ≥1.25) | `1.25.0` derive | `cintx/crates/cintx-cubecl/Cargo.toml:39` |
| `edition` | `2024` | `2024` | every sibling Cargo.toml |
| `rust-version` | `1.92` | `1.92` | `xcfun_rs/Cargo.toml:37` |

The following pins are **independent** — pyscf_rs can pick freely, but I recommend the sibling values for consistency:

| Dep | Recommended | Sibling |
|---|---|---|
| `anyhow` | `1.0.102` | `1.0.102` (cintx, xcfun_rs) |
| `criterion` | `=0.8.2` | `=0.8.2` (xcfun_rs) |
| `proptest` | `=1.11.0` | `=1.11.0` (xcfun_rs) |
| `rstest` | `=0.26.1` | `=0.26.1` (xcfun_rs) |
| `approx` | `=0.5.1` | `=0.5.1` (xcfun_rs) |
| `serde`/`serde_json` | `=1.0.228` / `=1.0.149` | match xcfun_rs |
| `cc` | `^1.2.60` | `^1.2.60` (xcfun_rs) |

---

## 4. Stack Patterns by Variant

**If a kernel must run on all 4 backends (the v1 norm):**
- Write it once in `#[cube]`, generic over `R: cubecl::Runtime`.
- Lives in `pyscf-{module}-cubecl`; the backend-routing wrapper lives in `pyscf-runtime`.
- Use `cubecl-cpu` as the always-on baseline so the test suite runs on every PR.

**If an op exists in faer but not (yet) in cubecl** (e.g., a Hermitian eigensolve at SCF time):
- Run on host via `faer`. The matrix is small (≤ a few thousand AOs); the GPU round-trip is more expensive than the host solve until matrices are huge.
- Document this in `pyscf-{module}-rs` with a comment pointing to the cubecl issue/feature gap.

**If the user has no GPU:**
- `cubecl-cpu` is the default-on backend (per sibling D-06). Everything works, just slower than CUDA.
- The 2–5× speedup target is measured against PySCF's C+OpenMP CPU baseline, so CPU-only users still see the win.

**If the user is on Apple Silicon:**
- Enable feature `metal` (which is an alias for `wgpu` — `cubecl-metal` does not exist on crates.io as of 2026-05; the sibling crates document this explicitly).
- WGPU's Metal backend covers M-series GPUs.

---

## 5. Version Compatibility

| Constraint | Reason |
|---|---|
| `pyo3 0.28.x` ↔ `numpy 0.28.x` | rust-numpy is locked to a specific PyO3 minor; mismatched minors won't compile. |
| `cubecl 0.10.x` ↔ `cubecl-{cpu,cuda,hip,wgpu,runtime} 0.10.x` | All five share an internal IR; minor mismatch breaks the JIT. |
| `cubecl-linalg 0.5.0` ↔ `cubecl ~0.5` (incompatible with 0.10) | Stale; **do not co-depend.** |
| `faer-ext 0.7.1` ↔ `faer 0.21+`, `ndarray 0.16+` | Verify the `faer-ext` Cargo.toml matches the faer version we pin (0.24.0); may need a bump. |
| `hdf5-metno 0.12.4` ↔ libhdf5 ≥ 1.8.4 (or `hdf5-sys/static` for bundled) | Needs CMake at build time when bundling. |
| `pyo3 0.28` abi3-py310 ↔ Python ≥ 3.10 | Python ≤ 3.9 users cannot install the wheel. Acceptable: PySCF's own modern minimum. |
| `cubecl 0.10` ↔ `wgpu 29.x` (when `wgpu` feature is on) | `cintx-cubecl/Cargo.toml:46` pins `wgpu = "29.0.3"`; match it. |

---

## 6. Installation snippet

```toml
# Workspace root Cargo.toml (mirrors xcfun_rs layout)
[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.92"

[workspace.dependencies]
# cubecl pivot — lockstep with cintx/xcfun_rs/libxc_rs
cubecl       = "=0.10.0"
cubecl-cpu   = "=0.10.0"
cubecl-cuda  = "=0.10.0"
cubecl-hip   = "=0.10.0"
cubecl-wgpu  = "=0.10.0"

# Sibling crates (path deps; published versions tracked in their own workspaces)
cintx     = { path = "../cintx" }
libxc_rs  = { path = "../libxc_rs" }
xcfun_rs  = { path = "../xcfun_rs" }

# Linear algebra
faer      = "0.24.0"
faer-ext  = "0.7.1"
ndarray   = "0.17.2"

# Python bindings (only in pyscf-py crate)
pyo3      = { version = "=0.28.3", features = ["extension-module", "abi3-py310"] }
numpy     = "=0.28.0"

# I/O
hdf5-metno = { version = "0.12.4", features = ["static"] }

# Numerics
num-traits  = "0.2.19"
num-complex = "0.4.6"
puruspe     = "0.4.4"
libm        = "0.2.16"

# Errors / logging / config
thiserror = "=2.0.18"
anyhow    = "1.0.102"           # app-boundary only
tracing   = { version = "=0.1.44", default-features = false }
tracing-subscriber = { version = "=0.3.23", features = ["fmt"] }

# Optimization driver
argmin     = "0.11.0"
argmin-math = "0.5.1"
finitediff = "0.2.0"

# GPU ABI
bytemuck  = { version = "1.25.0", features = ["derive"] }

# Dev / test / bench
criterion = { version = "=0.8.2", default-features = false, features = ["html_reports"] }
proptest  = "=1.11.0"
rstest    = "=0.26.1"
insta     = "1.47.2"
approx    = "=0.5.1"

# Build helpers
cc        = { version = "^1.2.60", features = ["parallel"] }
```

```bash
# Tooling
cargo install cargo-nextest --locked   # 0.9.133
cargo install cargo-llvm-cov --locked  # 0.8.6
pipx install maturin                   # 1.13.1
```

---

## 7. Sources

- crates.io JSON API (`https://crates.io/api/v1/crates/<name>`), queried 2026-05-09 — every version above.
- `~/Documents/workspace/cintx/Cargo.toml` and `cintx/crates/cintx-cubecl/Cargo.toml` — sibling cubecl pin and feature pattern (HIGH).
- `~/Documents/workspace/xcfun_rs/Cargo.toml`, `xcfun-py/Cargo.toml`, `xcfun-gpu/Cargo.toml` — workspace-deps, PyO3/numpy versions, `metal=alias-for-wgpu` pattern, MSRV 1.92, edition 2024 (HIGH).
- `~/Documents/workspace/libxc_rs/Cargo.toml` — `cubecl 0.10.0 default-features=false features=["cpu"]` confirmation (HIGH).
- `https://docs.rs/cubecl/0.10.0/cubecl/` — `Runtime` trait, `#[cube]`, `CubeLaunch`, autotuning availability (HIGH for trait names; MEDIUM for usage details — sibling code is the better reference).
- `https://github.com/tracel-ai/cubecl` — alpha-stability disclaimer; release cadence (HIGH).
- `https://docs.rs/faer/0.24.0/faer/` — pure-Rust, no system BLAS, eigendecomposition + Cholesky + LU + QR + SVD + GEMM, complex (`c64`) supported (HIGH).
- `https://github.com/metno/hdf5-rust` — fork rationale, `hdf5-sys/static` feature for bundled libhdf5, requires libhdf5 ≥ 1.8.4 (HIGH for build mechanics; MEDIUM for h5py round-trip — needs empirical validation).
- `https://pyo3.rs/v0.28.3/` — current PyO3 docs; abi3-py310 supports CPython 3.10+ in one wheel (HIGH).
- `~/Documents/workspace/pyscf_rs/.planning/codebase/STACK.md` — upstream PySCF stack (h5py for chkfile, libcint/libxc/xcfun as the C libs we replace, BLAS-as-system-dep — the thing we're escaping) (HIGH).
- `~/Documents/workspace/pyscf_rs/.planning/PROJECT.md` — locked decisions: pure Rust, cubecl as sole compute primitive, four backends, PyO3 + maturin, bit-exact PySCF agreement, 2–5× target, license Apache-2.0 (HIGH).

---
*Stack research for: pure-Rust quantum-chemistry library (pyscf_rs)*
*Researched: 2026-05-09*
