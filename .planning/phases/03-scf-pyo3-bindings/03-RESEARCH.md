# Phase 3: SCF + PyO3 bindings - Research

**Researched:** 2026-05-11
**Domain:** Self-consistent field kernels (RHF/UHF/GHF + DIIS + DF-HF + chkfile) + PyO3 0.28.3 contract surface (subclass-override, NumPy boundary, GIL release, panic→exception, abi3-py310 wheel skeleton)
**Confidence:** HIGH on PyO3/numpy/maturin/upstream-PySCF source-of-truth; MEDIUM on hdf5-metno h5py-interop edge cases (only the round-trip oracle in ORACLE-08 can give HIGH); LOW only on python3.13t-vs-abi3 interaction (PyO3 docs flag the conflict — Phase 3 must choose).

## Summary

Phase 3 lands the project's most surface-defining decisions on a deliberately tiny chemistry surface (RHF on H2O/cc-pVDZ). Every PyO3 contract every later method phase inherits — the trait-callback bridge for subclass-override dispatch, type-specific NumPy converters, per-hook `Python::detach`, `create_exception!`-based panic→exception (because `#[pyclass(extends=PyException)]` is forbidden under abi3-py310 until Python 3.12), `GILOnceCell`/`PyOnceLock` caches — is locked here on a surface small enough to debug. The chemistry side is well-understood: RHF/UHF/GHF kernel + C-DIIS (`SDF-FDS` error vector, B-matrix solve, `space=8`, `start_cycle=1`), 5 init_guess modes ported directly from `pyscf/scf/hf.py:348-700`, the inline `eig` sign-canonicalization rule at `pyscf/scf/hf.py:1349-1357`, DF-HF using `cintx.int3c2e_sph`/`int2c2e_sph` + Cholesky of `(P|Q)`, and an h5py-schema-compatible chkfile via `hdf5-metno = "0.10.0"` (verified current). Workspace grows 15 → 18 with three new crates (`pyscf-chkfile`, `pyscf-diis`, `pyscf-df`), all subject to the algebra-wall lint extension.

The single non-trivial PyO3-side surprise: **abi3-py310 forbids subclassing `PyException` at the C level until Python 3.12** [VERIFIED: PyO3 0.28 guide §building-and-distribution.md, §exception.md]. This forces the panic→exception pattern from the xcfun-py sibling (`create_exception!` for the bare class + a Python-side `__init__.py` shim grafting attributes). The second is **abi3 wheels and free-threaded Python 3.13t use incompatible ABIs** [VERIFIED: PyO3 0.28 guide §free-threading.md], so the `python3.13t` CI job in BIND-05 must build a separate non-abi3 wheel — the abi3-py310 wheel skeleton does NOT cover 3.13t.

**Primary recommendation:** Mirror the xcfun-py sibling-crate PyO3 patterns verbatim at every boundary; preserve every chemistry algorithm from upstream `pyscf/scf/hf.py` line-for-line; route every reduction through `pyscf-algebra::oracle_sum`/`oracle_dot` under `release-oracle`; build the abi3-py310 wheel skeleton (BIND-01/02) AND a separate non-abi3 Python 3.13t smoke (BIND-05) as parallel CI jobs.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### PyO3 subclass-override dispatch (Pitfall 7, BIND-07, SCF-08)

- **D-01: Trait-callback bridge.** `pyscf-scf` declares a `pub trait OverrideHooks` with one method per overrideable hook (`get_jk`, `get_veff`, `get_hcore`, `eig`, `get_occ`, `make_rdm1`, `energy_elec`, `energy_tot`, `get_init_guess`, `get_fock`). `pyscf-scf` has **zero pyo3 dependency**. `pyscf-py` provides a `PyOverrideBridge` impl that routes every call through `slf.call_method1(py, "<hook>", args)`. Python's MRO does subclass-override dispatch natively — if a Python subclass overrides, its method runs; if not, the `#[pymethods]` default in `pyscf-py` runs (which forwards to a public Rust-default function in `pyscf-scf`). Pitfall 7 immune by construction. Phase 4 DFT re-validates on the larger DFT overrideable surface (DFT-08) using the same trait shape.

- **D-02: Pub trait + pub generic kernel — Rust-only SCF API.** `OverrideHooks` is `pub`. `pyscf_scf::RHF::kernel<H: OverrideHooks>(mol, hooks) -> Result<ScfResult>` is generic over the bridge impl. A Rust-only caller implements `OverrideHooks` (no pyo3 needed — provide a `NoOverrides` zero-cost default impl) and drives SCF without Python. Aligns with DIST-01 (pyscf-rs on crates.io with workspace façade re-exporting in-scope methods).

- **D-03: Per-hook `Python::detach` GIL release seam.** Each heavy hook body wraps compute in `Python::detach` (≡ old `py.allow_threads` in PyO3 0.28): Fock build's two-electron contraction via cintx `int2e_sph`, the `eigh` call, and the DIIS extrapolation matrix solve. Override call sites stay GIL-attached by definition. The `python3.13t` free-threaded CI build runs the SCF test corpus to probe deadlock surface. Phase 4 inherits at XC-evaluation kernel; Phase 6 inherits at CCSD doubles update — minimal scope per detach so deadlock risk is contained per-hook.

- **D-04: Type-specific NumPy boundary converters.** `pyscf-py` defines `to_density(arr) -> Density`, `to_mo_coeff(arr) -> MOCoefficients`, `to_fock_matrix(arr) -> Fock`, etc. Each runs `is_standard_layout()` and calls `to_owned()` before constructing the pyscf-core Rust type (BIND-04, Pitfall 5 mitigation). Output helpers: `density_to_pyarray(d, py)`, `mo_coeff_to_pyarray(mc, py)` — always C-contiguous unless the upstream PySCF per-name convention is F-order (carried from Phase 2 D-04: planner consults `pyscf/gto/moleintor.py`). Greppable in CI. BIND-04 stride-fuzz test calls each entry with `a`, `a.T`, `a[::2]`, `a[:, 1:5]` and asserts identical answers.

#### HDF5 chkfile (SCF-10, ORACLE-08, Pitfall 11)

- **D-05: `hdf5-metno` crate.** Maintained metno fork of `aldanor/hdf5-rust`. Static linking via `hdf5-metno-sys` `static` feature satisfies DIST-05 (HDF5 ships statically linked, no system libhdf5 at install time). Has ndarray integration. STATE.md "Blockers/Concerns" already names it as the candidate needing empirical seal in Phase 3 — that seal IS the ORACLE-08 round-trip oracle.

- **D-06: New `pyscf-chkfile` workspace crate.** Sole owner of the `hdf5-metno` dependency (algebra-wall-style discipline). Exposes HDF5 primitives (`open_for_write`, `read_group`, `write_dataset`, `read_dataset`, `write_string_attr`) PLUS a `Checkpointable` trait. Per-method schema modules live in each method crate (`pyscf_scf::chkfile`, `pyscf_dft::chkfile`, `pyscf_ccsd::chkfile`, `pyscf_geomopt::chkfile`) and `impl Checkpointable for ScfResult` / `for KsResult` / `for CcsdResult` / `for OptimState`. Workspace grows 15 → 16 (this is the first of three new crates introduced in Phase 3; ROADMAP.md update required during planning).

- **D-07: Rust-driven chkfile round-trip oracle.** `pyscf-oracle::oracle_check!("chkfile_roundtrip", fixture)` macro lives in `pyscf-oracle` (dev-deps only, `pyo3 = "=0.28.3"` with `auto-initialize` already declared per Phase 1's existing Cargo.toml). Macro spawns Python via `Python::attach`, runs upstream `pyscf.scf.RHF(mol).kernel()` to produce a chkfile in a tmpdir, then pyscf-rs reads + asserts numpy-allclose at 1e-12 on `mo_coeff`/`mo_energy`/`mo_occ`/`e_tot`. Reverse direction: pyscf-rs writes chkfile, Python `from_chk(path).kernel()`, asserts converged at upstream energy. Same macro shape locks the ORACLE-02 contract for every SCF success criterion.

#### DIIS (SCF-04, Pitfall 9; Phase 6 CCSD-04 reuse)

- **D-08: New `pyscf-diis` workspace crate.** Generic over a `pub trait DiisStorable { fn as_flat(&self) -> &[f64]; fn from_flat(&mut self, slice: &[f64]); fn dot(&self, other: &Self) -> f64; }`. Depends only on `pyscf-algebra` for the small B-matrix linear solve (typically ≤ 8×8) and `axpy`/`dot` primitives. `pyscf-scf` and `pyscf-ccsd` consume the trait. Workspace 16 → 17. Pitfall 9 mitigation: all reductions inside DIIS go through Phase 1's `oracle_sum`/`oracle_dot` under `release-oracle` so the extrapolated Fock matches upstream when reduction order is held.

- **D-09: Generic `DiisStorable` trait.** `pyscf-scf::FockSubspace` impls it for Fock matrices (`nao × nao`, F-order); `pyscf-ccsd::AmpsSubspace` impls it for `(T1, T2)` tuples in Phase 6. `pyscf-diis::Diis<S: DiisStorable>` is the generic Pulay-extrapolation stack. Trait is object-safe so boxing works. Each impl picks its own algebra layout.

#### Density fitting (SCF-07; Phase 4 DFT-07, Phase 5 MP2-04, Phase 6 CCSD-08 reuse)

- **D-10: New `pyscf-df` workspace crate.** 18th workspace member. Mirrors upstream `pyscf/df/` exactly (sibling-crate fidelity hard preference inherited from Phase 1). Owns 3-center aux integrals (`mol.intor('int3c2e_sph')`), 2-center aux integrals (`mol.intor('int2c2e_sph')`), Cholesky of `(P|Q)` via `pyscf_algebra::cholesky` (host-faer per Phase 1 D-06), and B-integral assembly (`B_{μν}^Q`). Public surface: `DfIntegrals { b_uvq: Tensor, naux: usize }` consumed uniformly by SCF/DFT/MP2/CCSD. Workspace 17 → 18; **net Phase 3 growth: 15 → 18 (+pyscf-chkfile, +pyscf-diis, +pyscf-df)**.

- **D-11: In-memory B integrals in Phase 3; HDF5 spill deferred to Phase 6.** Phase 3 DF-HF ships with the full `B_{μν}^Q` array in memory. Sufficient for the test corpus (H2O/cc-pVDZ, benzene/6-31G*, water trimer) — success criterion 3 makes no `PYSCF_MAX_MEMORY` assertion at Phase 3. Phase 6 CCSD-08 + CCSD-11 (tensor-arena from CCSD day one) extends `pyscf-df` with HDF5 spill via `pyscf-chkfile` primitives. No premature optimization; aligns with ROADMAP.md's explicit Phase 6 placement of the spill machinery.

### Claude's Discretion

- **`init_guess` scope** — All five modes (`'minao'`, `'atom'`, `'1e'`, `'huckel'`, `'chkfile'`) are required by SCF-05. Planner decides whether to ship all five in the SCF core plan or split MVP (`'1e'` + `'minao'`) into early plans and `'atom'`/`'huckel'`/`'chkfile'` into later plans. `'chkfile'` mode depends on the `pyscf-chkfile` crate landing first.
- **`canonicalize_signs` (SCF-13, Pitfall 4) location** — REQUIREMENTS.md SCF-13 names it `pyscf-core::lib::canonicalize_signs`. Planner confirms: pure function in pyscf-core (no algebra deps; operates on a `Vec<f64>` view of MO coefficients), called by SCF/DFT/MP2/CCSD post-eigh. Algorithm: largest-|coefficient|-with-lowest-index sign-flip rule.
- **`python/pyscf/__init__.py` overlay strategy** — Phase 1 D-03 forbids touching upstream `pyscf/`. Phase 3 ships a NEW `python/pyscf/__init__.py` overlay (under a new `python/` directory at repo root) that re-exports from the `_native.scf` PyO3 submodule. Planner picks the overlay-vs-namespace-package mechanism that lets pyscf-rs's wheel install Python files that take precedence over the upstream tree (maturin's `python-source` config is the obvious answer).
- **`GILOnceCell` migration sites (BIND-06)** — pyscf-rs has no `lazy_static!` today (Phase 1 stub), so BIND-06 in Phase 3 is preventive: planner enforces a CI lint that forbids `lazy_static!` in any crate under the `python` feature path and uses `pyo3::sync::GILOnceCell` for the small number of PyO3-side caches (likely `type_object_id` caches for the override-dispatch fast-path detection).
- **abi3-py310 wheel skeleton scope** — ROADMAP.md says skeleton in Phase 3, full matrix in Phase 8. Planner decides whether "skeleton" means (a) just the abi3 feature flag set on `pyscf-py`'s `[lib]` config + a CI smoke test that runs `maturin develop` on Linux x86_64 only, or (b) a fuller smoke including `auditwheel`/`abi3audit` invocations (deferred to Phase 8 DIST-04 / BIND-08). Recommended: (a) — minimum viable skeleton; Phase 8 owns the full matrix.
- **Panic → Python exception conversion (BIND-09, Pitfall 14)** — Phase 1 already established `catch_unwind` at every `extern "C"` callback (lint enforced). Phase 3 specifies the conversion: planner picks between `pyo3::panic::PanicException` (standard) vs custom `PyscfRuntimeError`. Standard is the default unless a chemistry-specific error hierarchy is needed.
- **`mf.as_scanner()` shape (SCF-12)** — Planner mirrors upstream `pyscf/scf/hf.py:as_scanner` semantics: returns a callable that takes a Mole and returns the energy. Used by Phase 7 geomopt. Implementation is a closure capturing `self`; the closure handle is a `Py<PyAny>` from Python's POV.
- **Cross-module dispatch helpers (SCF-11)** — `mf.to_uhf()`, `mf.to_rhf()`, `mf.to_ghf()` ship in Phase 3. `mf.to_uks()`, `mf.to_rks()` are declared but the KS targets only exist in Phase 4 — Phase 3 ships stubs that return `NotYetImplemented { phase: 4 }`. Planner confirms this split.
- **30-attribute SCF floor (SCF-14)** — Planner enumerates the full ≥30 attribute list from upstream `pyscf/scf/hf.py:SCF` class and wires them through `#[pymethods] #[getter]`/`#[setter]` pairs in pyscf-py with the BIND-04 type-specific converters from D-04.
- **Test corpus tiering** — PR-CI: H2O/cc-pVDZ (RHF + UHF + DF-HF), benzene/6-31G* (RHF), water-trimer (RHF + chkfile round-trip). Nightly: add larger fixtures + cross-platform Linux+macOS µHartree assertion (Pitfall 12). Phase 8 ORACLE-06 nightly per-basis sweep is Phase 8's responsibility.
- **`pyscf-df` 3-center kernel home** — All three-center contractions route through `mol.intor('int3c2e_sph')` (cintx) per the existing Phase 2 dispatcher. No new cubecl kernel needed — pyscf-df only does Cholesky + matrix multiply (both via pyscf-algebra). If profiling later shows the matrix-multiply chain is the bottleneck, Phase 8 can introduce a fused cubecl kernel.
- **DIIS B-matrix linear solver** — Phase 3 uses `pyscf-algebra::solve_linear` (host-faer fallback per Phase 1 D-06) for the typically ≤ 8×8 system. Hand-rolled Gauss-Jordan is faster for these tiny sizes but the host-faer path is bit-exact and matches Phase 1 conventions.

### Deferred Ideas (OUT OF SCOPE)

- **ROHF + ROHF gradients + SOSCF (`scf.newton`) + ADIIS/EDIIS + symmetry-adapted SCF** — v1.x (REQUIREMENTS.md SCF-EXT-01..05). Phase 3 explicitly out of scope.
- **HDF5 spill for DF B integrals** — Phase 6 CCSD-08 + CCSD-11 (tensor-arena from CCSD day one). Phase 3 ships in-memory only.
- **CCSD(T) — perturbative triples** — v1.x P1 (STATE.md Deferred Items table).
- **`PYSCF_MAX_MEMORY` budget enforcement at SCF level** — Phase 6 CCSD-11 introduces the budget-aware tensor-arena. Phase 3 only logs `PYSCF_MAX_MEMORY` at kernel entry (no enforcement).
- **GPU per-backend regression for SCF** — Phase 8 (ORACLE-07). Phase 3 only ships CPU-backend correctness on the test corpus.
- **`abi3audit` CI invocation on the produced wheel** — Phase 8 (BIND-08). Phase 3 wheel skeleton is develop-mode only.
- **`auditwheel show` clean + manylinux_2_28** — Phase 8 (DIST-04).
- **Per-backend `pyscf-rs[cuda]`/`pyscf-rs[wgpu]`/`pyscf-rs[rocm]` extras** — Phase 8 (DIST-03).
- **20-idiom drop-in audit (BIND-03)** — Phase 8.
- **`mf.to_uks()`/`mf.to_rks()` cross-module dispatch** — Phase 3 ships stubs returning `NotYetImplemented { phase: 4 }`; Phase 4 wires the real KS targets.
- **Full chkfile schema port for DFT/CCSD/geomopt** — Phase 4/6/7 each impl `Checkpointable` for their own result type. Phase 3 only ships the SCF schema + the trait + the HDF5 primitives.
- **Fused cubecl 3-center contraction kernel for DF** — Phase 8 (if profiling shows it matters). Phase 3 uses cintx + `pyscf-algebra::gemm`.
- **cintx ECP gap-closure plan from Phase 2** — independent of Phase 3 SCF work; lands when cintx upstream merges Type-1+Type-2 projectors. Phase 3 SCF does not exercise ECP on the curated test corpus.
- **Wheel packaging of `pyscf/gto/basis/`** (carried from Phase 2 D-deferred) — Phase 8 DIST-02.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SCF-01 | `scf.RHF(mol).kernel()` matches upstream PySCF total energy ≤1 µHartree | RHF kernel mirrors `pyscf/scf/hf.py:48-244` (`def kernel`); `eig` at line 1349-1357 (inline sign-canon); `oracle_sum`/`oracle_dot` (Phase 1) for Fock reductions; H2O/cc-pVDZ test fixture in oracle harness |
| SCF-02 | `scf.UHF(mol).kernel()` matches upstream for open-shell | UHF class at `pyscf/scf/uhf.py:754`; alpha/beta density pair; `spin_square` re-export |
| SCF-03 | `scf.GHF(mol).kernel()` runs (correctness only, no perf parity) | GHF class at `pyscf/scf/ghf.py:378`; 2-component spinor SCF; doubled-AO Fock structure |
| SCF-04 | C-DIIS with `diis_space=8`/`diis_start_cycle=1` | `pyscf/scf/diis.py:40-66` `CDIIS`; error vector `SDF - FDS` at `diis.py:68-87`; B-matrix Pulay solve via `pyscf-algebra::solve_linear` (or cholesky+trisolve); `pyscf-diis` new workspace crate (D-08) |
| SCF-05 | 5 init_guess modes (`'minao'`,`'atom'`,`'1e'`,`'huckel'`,`'chkfile'`) + `dm0` | `pyscf/scf/hf.py:348` minao, `:495` atom, `:485` 1e, `:537` huckel, `:673` chkfile; dispatcher `get_init_guess` at `:764-789`; class-method wrappers `:1876-1973` |
| SCF-06 | `level_shift`/`damp`/`max_cycle`/`conv_tol`/`conv_tol_grad` | `pyscf/scf/hf.py:775` level_shift function; class defaults at `:1689-1712` |
| SCF-07 | `mf.density_fit(auxbasis=...)` DF-HF | `pyscf/scf/hf.py:2165-2172` density_fit method; `pyscf/df/df.py:41-191` DF class; auxbasis defaults via `pyscf/df/addons.py` DEFAULT_AUXBASIS (`weigend`/`cc-pvdz-jkfit`); new `pyscf-df` crate (D-10) |
| SCF-08 | All 10 overrideable hooks dispatch via `slf.call_method1` | `OverrideHooks` trait (D-01) — `get_jk`, `get_veff`, `get_hcore`, `eig`, `get_occ`, `make_rdm1`, `energy_elec`, `energy_tot`, `get_init_guess`, `get_fock`; PyO3 0.28 trait-bridge pattern from `pyo3.rs/.../trait-bounds.md` |
| SCF-09 | `mf.analyze()`/`mulliken_pop`/`mulliken_meta`/`dip_moment` | `pyscf/scf/hf.py:1199` analyze, `:1262` mulliken_pop, `:1301` mulliken_meta, `:1380` dip_moment |
| SCF-10 | `mf.chkfile = path` h5py-compatible HDF5 + `mf.from_chk(path)` | `pyscf/scf/chkfile.py:25-42`: groups `/scf/{e_tot,mo_energy,mo_occ,mo_coeff}` + `/mol` (mol.dumps() JSON string); `hdf5-metno = "0.10.0"` (D-05); `pyscf-chkfile` crate (D-06) |
| SCF-11 | `to_uhf()`/`to_rhf()`/`to_uks()`/`to_rks()`/`to_ghf()` | `pyscf/scf/hf.py:2272-2300` (rhf/uhf/ghf via `addons.convert_to_*`); `:2302-2318` (uks/rks via to_rhf().to_ks() — stub `NotYetImplemented{phase:4}` for KS) |
| SCF-12 | `mf.as_scanner()` callable | `pyscf/scf/hf.py:1538-1602` `as_scanner` + `SCF_Scanner.__call__`; PyO3 returns `Py<PyAny>` callable closure capturing self |
| SCF-13 | `canonicalize_signs` vendor-stable eigenvectors | Algorithm inlined at `pyscf/scf/hf.py:1349-1357` (`def eig`): `idx = argmax(abs(c.real), axis=0); c[:,c[idx,arange]<0] *= -1`. **NOT a named function upstream** — Phase 3 extracts it to `pyscf-core::lib::canonicalize_signs` as a reusable pure fn (no algebra deps) called by SCF/DFT/MP2/CCSD post-eigh. |
| SCF-14 | 30-attribute SCF floor | Class defaults at `pyscf/scf/hf.py:1689-1759`; `_keys` set at `:1716-1724` is the canonical list; 32 keys enumerated below. |
| BIND-01 | abi3-py310 cdylib wheel skeleton | `pyscf-py/Cargo.toml` `[lib] crate-type = ["cdylib","rlib"]` (already in Phase 1); add `pyo3 = "=0.28.3"` with `features = ["extension-module","abi3-py310"]` |
| BIND-02 | `from pyscf import scf` works via `_native.scf` + overlay shim | maturin `[tool.maturin] python-source = "python" module-name = "pyscf._native"` (mirrors xcfun-py pattern); new `python/pyscf/__init__.py` + `python/pyscf/scf/__init__.py` overlay |
| BIND-04 | NumPy `is_standard_layout` → `to_owned()` + stride-fuzz CI | `PyReadonlyArray2<'py, f64>::is_standard_layout()` and `is_c_contiguous()` from `PyUntypedArrayMethods` (verified in xcfun-py `numpy_io.rs:28`); type-specific converters in `pyscf-py` (D-04); pytest stride fuzzer with `a`, `a.T`, `a[::2]`, `a[:,1:5]` |
| BIND-05 | `Python::detach` + python3.13t free-threaded CI | Per-hook detach scope (D-03); `py.detach(|| compute(...))` pattern from xcfun-py `numpy_io.rs:65-72`; **abi3 wheels incompatible with python3.13t — separate non-abi3 build job needed** |
| BIND-06 | `GILOnceCell`/`PyOnceLock` replaces every `lazy_static!` | `pyo3::sync::GILOnceCell` (legacy, still works) OR `PyOnceLock` (PyO3 0.28 recommended replacement, free-thread-safe); CI lint forbids `lazy_static!` in `pyscf-py`; cache type-id-of-Python-subclass for override fast-path |
| BIND-07 | Subclass override via `slf.call_method1` (NOT Rust MRO) | Confirmed in research §1 below; trait-bridge pattern at `pyo3.rs/.../trait-bounds.md` (verified via Context7 fetch) |
| BIND-09 | Panic → Python exception preserves error chain | **abi3-py310 forbids `#[pyclass(extends=PyException)]` until Python 3.12** (verified PyO3 0.28 guide §exception.md §building-and-distribution.md); use `create_exception!(_native, PyscfRsError, PyException)` + Python `__init__.py` shim grafting `.kind`/`.source` attributes (xcfun-py pattern at `errors.rs:31-83`) |
| ORACLE-02 | `oracle_check!(method, tolerance, fixture)` macro | New macro in `pyscf-oracle/src/lib.rs`; spawns `Python::attach` (pyo3 in dev-deps only — already declared in Phase 1 `pyscf-oracle/Cargo.toml:18`); compares pyscf-rs vs upstream pyscf bit-exact under `release-oracle` |
| ORACLE-08 | chkfile round-trip oracle (h5py↔hdf5-metno empirical seal) | `oracle_check!("chkfile_roundtrip", fixture)` — drives upstream PySCF writes, pyscf-rs reads + asserts identical; reverse direction pyscf-rs writes, Python `from_chk(path).kernel()` converges to upstream energy; this IS the ORACLE-08 deliverable AND the STATE.md "Blockers/Concerns" h5py/hdf5-metno empirical seal |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

No `./CLAUDE.md` exists at the repo root. Carried constraints come from project-level memory + Phase 1/2 CONTEXT.md:

- **libxc_rs compile is ~6h — never trigger it.** Phase 3 must not pull `libxc_rs` into the dep graph. (Phase 4 owns that.)
- **Don't freeze compile.** No heavy `build.rs`, no parse-N-files macros, no libxc_rs. Phase 3 confirmed compatible — all three new crates (`pyscf-chkfile`, `pyscf-diis`, `pyscf-df`) are lightweight wrappers.
- **Algebra wall (Phase 1 D-04..06).** `pyscf-scf`, `pyscf-diis`, `pyscf-df` depend on `pyscf-algebra` only; never on `cubecl-*` directly. `pyscf-chkfile` has no algebra dep at all (pure I/O). Extended xtask `algebra-wall` lint must include these 3 new crates.
- **`#[forbid(unsafe_code)]` + `#[warn(clippy::unwrap_used)]`** — all 3 new crates adopt these.
- **Sibling-crate fidelity is a hard preference.** `pyscf-df` mirrors `pyscf/df/`; `pyscf-diis` mirrors `pyscf/scf/diis.py`; `pyscf-chkfile` mirrors `pyscf/lib/chkfile.py` + per-method modules. Deviations require explicit justification.
- **No new env vars in Phase 3.** Consume existing `PYSCF_BACKEND`, `PYSCF_DTYPE`, `PYSCF_BASIS_PATH`; surface `PYSCF_MAX_MEMORY` only at kernel-entry log (no enforcement).

## Standard Stack

### Core (Phase 3 introduces or pins)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `pyo3` | `=0.28.3` | Python bindings (cdylib + dev-deps) | [VERIFIED: github.com/PyO3/pyo3/releases, latest stable as of April 2, 2025]; Phase 1 already pinned in `pyscf-oracle/Cargo.toml:18`; Phase 3 adds to `pyscf-py` dependencies with `features=["abi3-py310","extension-module"]` |
| `numpy` (rust-numpy) | `=0.28.0` | NumPy array bindings for pyscf-py | [VERIFIED: xcfun-py `Cargo.toml:27` ships this exact pin]; aligned with pyo3 0.28.x ecosystem |
| `hdf5-metno` | `=0.10.0` | HDF5 file I/O for chkfile (sole owner: pyscf-chkfile) | [VERIFIED: github.com/metno/hdf5-rust README, "version 0.10.0"]; fork of `aldanor/hdf5-rust` published for newer crates.io; **feature `static` enables embedded libhdf5** (DIST-05 satisfied) |
| `maturin` | `>=1.12,<2.0` | Wheel build tool | [VERIFIED: xcfun-py pyproject.toml line 2]; standard for PyO3 ecosystem |

### Supporting (already in workspace)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `thiserror` | `=2.0.18` (workspace) | Error enum derivation | All new crates' error types |
| `tracing` | `=0.1.44` (workspace) | Structured logging | SCF kernel entry log (ALG-08); chkfile open events |
| `serde` + `serde_json` | `=1.0` / `=1.0.149` (workspace) | `mol.dumps()` JSON serialization | `pyscf-chkfile` writes mol.dumps() under `/mol` string dataset |
| `faer` | `=0.24.0` (workspace) | Host eigh/cholesky/QR (already used by pyscf-algebra) | DIIS B-matrix solve via `pyscf-algebra::solve_linear` (or cholesky+trisolve — extend pyscf-algebra if `solve_linear` doesn't exist yet) |
| `pyscf-core` | `0.1.0` (workspace) | Density, MOCoefficients, Energy, Mole, Scf trait | SCF result types; Phase 3 fills `Scf` trait impls |
| `pyscf-algebra` | `0.1.0` (workspace) | gemm/eigh/oracle_sum surface | Fock build, eigh, DIIS dot products under `release-oracle` |
| `pyscf-gto` | `0.1.0` (workspace) | `mol.intor('int1e_*'/'int2e_sph')` dispatcher | Integral evaluation for SCF Fock build |
| `pyscf-runtime` | `0.1.0` (workspace) | BackendKind, WorkspacePool | Kernel-entry backend logging |
| `cintx-rs` (transitive) | git pin | Integral engine | Consumed via `pyscf-gto::intor` — NO direct dep needed in pyscf-scf |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `hdf5-metno` | `aldanor/hdf5-rust` (original) | Original isn't published on crates.io anymore; hdf5-metno is the maintained fork [VERIFIED: github.com/metno/hdf5-rust README] |
| `pyo3::sync::GILOnceCell` | `pyo3::sync::PyOnceLock` | PyO3 0.28 migration guide: "PyOnceLock has been introduced for true single-initialization correct even with free-threaded Python. It offers the same API as `GILOnceCell`, but reliance on its previous racy initialization might lead to deadlocking." [VERIFIED: PyO3 0.28 migration.md] — **Recommendation: prefer `PyOnceLock` for new code; both still work in 0.28.x.** BIND-06 spec says "GILOnceCell" — planner should treat this as either-or, defaulting to `PyOnceLock`. |
| `#[pyclass(extends=PyException)]` | `create_exception!(_native, PyscfRsError, PyException)` + Python overlay shim | **abi3-py310 forbids C-level PyException subclassing until Python 3.12** [VERIFIED: PyO3 0.28 guide §exception.md, §building-and-distribution.md]. Must use the macro + overlay pattern (xcfun-py reference). |
| `pyo3::panic::PanicException` directly | Custom `create_exception!`-based `PyscfRsError` | PyO3 wraps unhandled panics as `PanicException` automatically; for richer error info (Rust source chain), we wrap in our own exception type. Recommended: hybrid — let PanicException catch genuine panics (lint-blocked in numerical modules anyway per FOUND-07), use `PyscfRsError` for all `Err(_)` returns. |
| Single PyO3 cdylib for both abi3 + 3.13t | Separate `pyscf-py` builds: abi3-py310 wheel (Linux/macOS/Windows ×x86_64+aarch64) + non-abi3 python3.13t smoke | **abi3 + 3.13t ABIs are incompatible** [VERIFIED: PyO3 0.28 free-threading.md]. The python3.13t job in BIND-05 needs `cargo build --no-default-features --features pyo3/auto-initialize` (no abi3-py310). |

**Installation (new workspace members + new deps):**

```toml
# Cargo.toml [workspace.dependencies] additions
pyo3   = { version = "=0.28.3", features = ["abi3-py310", "extension-module"] }
numpy  = { version = "=0.28.0" }
hdf5-metno = { version = "=0.10.0", features = ["static"] }

# crates/pyscf-py/Cargo.toml [dependencies]
pyo3   = { workspace = true }
numpy  = { workspace = true }
pyscf-core = { path = "../pyscf-core" }
pyscf-scf  = { path = "../pyscf-scf" }
pyscf-gto  = { path = "../pyscf-gto" }
pyscf-runtime = { path = "../pyscf-runtime" }
pyscf-chkfile = { path = "../pyscf-chkfile" }
thiserror = { workspace = true }

# crates/pyscf-chkfile/Cargo.toml [dependencies]
pyscf-core = { path = "../pyscf-core" }
hdf5-metno = { workspace = true }
serde_json = { workspace = true }
serde = { workspace = true }
ndarray = "0.16"   # hdf5-metno integrates here
thiserror = { workspace = true }
tracing = { workspace = true }

# crates/pyscf-diis/Cargo.toml [dependencies]
pyscf-core = { path = "../pyscf-core" }
pyscf-algebra = { path = "../pyscf-algebra" }
thiserror = { workspace = true }

# crates/pyscf-df/Cargo.toml [dependencies]
pyscf-core = { path = "../pyscf-core" }
pyscf-algebra = { path = "../pyscf-algebra" }
pyscf-gto = { path = "../pyscf-gto" }
thiserror = { workspace = true }
tracing = { workspace = true }
```

**Version verification:**
- `pyo3 = "=0.28.3"` — `npx ctx7@latest docs /pyo3/pyo3` confirmed Python::attach/detach/GILOnceCell/PyOnceLock APIs; GitHub releases page lists 0.28.3 (April 2, 2025) as latest stable [VERIFIED via WebFetch].
- `numpy = "=0.28.0"` — xcfun-py shipped pin [VERIFIED in `xcfun_rs/crates/xcfun-py/Cargo.toml:27`].
- `hdf5-metno = "=0.10.0"` — README.md confirms "version 0.10.0" [VERIFIED via WebFetch of github.com/metno/hdf5-rust].

## Architecture Patterns

### Recommended Project Structure

```
pyscf_rs/
├── Cargo.toml                          # Workspace: 15 → 18 members (+chkfile, +diis, +df)
├── pyproject.toml                      # NEW — maturin config for `_native` module + python-source overlay
├── python/                             # NEW overlay directory (BIND-02 + Claude's discretion)
│   └── pyscf/
│       ├── __init__.py                 # NEW — re-exports from _native + grafts XcfunError-style shim
│       ├── scf/
│       │   ├── __init__.py             # NEW — `from pyscf._native.scf import RHF, UHF, GHF, density_fit`
│       │   ├── hf.py                   # NEW thin re-export of RHF + class aliases
│       │   ├── uhf.py                  # NEW thin re-export
│       │   └── ghf.py                  # NEW thin re-export
│       └── lib/
│           └── chkfile.py              # NEW thin re-export of dump_scf/load_scf
├── crates/
│   ├── pyscf-py/                       # Phase 1 stub — Phase 3 fills cdylib
│   │   ├── Cargo.toml                  # adds pyo3, numpy, pyscf-scf, pyscf-gto, pyscf-chkfile, abi3-py310 + extension-module features
│   │   └── src/
│   │       ├── lib.rs                  # #[pymodule] fn _native — registers scf submodule
│   │       ├── scf.rs                  # #[pyclass(subclass)] PyRHF/PyUHF/PyGHF + PyOverrideBridge impl
│   │       ├── numpy_io.rs             # to_density/to_mo_coeff/to_fock_matrix + density_to_pyarray helpers (D-04)
│   │       ├── errors.rs               # create_exception!(_native, PyscfRsError, PyException) + py_err conversion
│   │       └── caches.rs               # PyOnceLock<...> type-id caches for override-dispatch fast-path
│   ├── pyscf-scf/                      # Phase 1 stub — Phase 3 fills
│   │   ├── Cargo.toml                  # NO pyo3 dep — algebra wall + D-01 trait-bridge
│   │   └── src/
│   │       ├── lib.rs                  # pub trait OverrideHooks, NoOverrides default, kernel<H>
│   │       ├── rhf.rs                  # RHF struct + impl Scf for RHF
│   │       ├── uhf.rs                  # UHF struct
│   │       ├── ghf.rs                  # GHF struct
│   │       ├── init_guess.rs           # 5 modes: minao/atom/1e/huckel/chkfile
│   │       ├── fock.rs                 # get_fock + get_jk + get_veff Rust-side defaults
│   │       ├── eig.rs                  # eig wrapper around pyscf-algebra::eigh + canonicalize_signs
│   │       ├── df_scf.rs               # DF-HF entry point — consumes pyscf-df::DfIntegrals
│   │       └── chkfile.rs              # impl Checkpointable for ScfResult — schema mirrors pyscf/scf/chkfile.py
│   ├── pyscf-chkfile/                  # NEW (D-06)
│   │   ├── Cargo.toml                  # hdf5-metno dep — sole owner
│   │   └── src/
│   │       ├── lib.rs                  # pub trait Checkpointable; primitive open_for_write/read_group/etc.
│   │       ├── primitives.rs           # write_dataset_c_order / write_dataset_f_order / write_string_attr / read_*
│   │       └── error.rs                # ChkfileError variants
│   ├── pyscf-diis/                     # NEW (D-08)
│   │   ├── Cargo.toml                  # depends on pyscf-algebra only
│   │   └── src/
│   │       ├── lib.rs                  # pub trait DiisStorable; pub struct Diis<S>
│   │       ├── cdiis.rs                # SDF-FDS error vector + B-matrix Pulay extrapolation
│   │       └── error.rs
│   ├── pyscf-df/                       # NEW (D-10)
│   │   ├── Cargo.toml                  # depends on pyscf-algebra + pyscf-gto
│   │   └── src/
│   │       ├── lib.rs                  # pub struct DfIntegrals { b_uvq, naux }
│   │       ├── auxbasis.rs             # DEFAULT_AUXBASIS table mirroring pyscf/df/addons.py
│   │       ├── cholesky_eri.rs         # 3-center int3c2e_sph + 2-center int2c2e_sph + Cholesky of (P|Q)
│   │       └── df_jk.rs                # J/K builders consuming DfIntegrals
│   ├── pyscf-oracle/                   # Phase 1 already declared pyo3 dev-dep; Phase 3 fills macro
│   │   └── src/
│   │       ├── lib.rs                  # #[macro_export] macro_rules! oracle_check { ... }
│   │       └── chkfile_roundtrip.rs    # ORACLE-08 macro arms
│   └── pyscf-core/                     # Phase 3 adds canonicalize_signs only
│       └── src/
│           └── lib_fn.rs (new module)  # pub fn canonicalize_signs(coeffs: &mut [f64], nao: usize, nmo: usize)
└── xtask/
    └── src/
        └── lints/
            └── algebra_wall.rs         # extend allowlist: pyscf-chkfile (no algebra), pyscf-diis (algebra ok), pyscf-df (algebra+gto ok)
```

### Pattern 1: PyO3 Subclass-Override Bridge (D-01, BIND-07, SCF-08, Pitfall 7)

**What:** Python subclass overrides flow back into Rust kernel via `Bound<'_, PyAny>::call_method1`. Rust kernel is GENERIC over a `H: OverrideHooks` trait; `pyscf-py` provides the bridge impl.

**When to use:** Every hook that PySCF users override on their `mf` subclass (the 10 SCF-08 hooks plus DFT-08's larger surface in Phase 4).

**Example — full pattern, ready for Phase 3 plans:**

```rust
// crates/pyscf-scf/src/lib.rs — NO pyo3 dependency (algebra wall + D-01)
// Source: PyO3 0.28 guide §trait-bounds.md, fetched via Context7.

use pyscf_core::{Density, Energy, MOCoefficients, Mole, error::PyscfRsError};

/// The 10 SCF-08 overrideable hooks. Phase 4 DFT extends with `define_xc_`,
/// `get_veff` (DFT-specific). Every method `&self` so the trait stays
/// object-safe — required to allow Box<dyn OverrideHooks> if a user passes
/// hooks dynamically.
pub trait OverrideHooks {
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError>;
    fn get_ovlp (&self, mol: &Mole) -> Result<Density, PyscfRsError>;
    fn get_init_guess(&self, mol: &Mole, key: &str) -> Result<Density, PyscfRsError>;
    fn get_jk(&self, mol: &Mole, dm: &Density) -> Result<(Density, Density), PyscfRsError>;
    fn get_veff(&self, mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError>;
    fn get_fock(&self, h1e: &Density, s1e: &Density, vhf: &Density,
                dm: &Density, cycle: i32, diis_state: Option<&Density>)
        -> Result<Density, PyscfRsError>;
    fn eig(&self, fock: &Density, s1e: &Density) -> Result<MOCoefficients, PyscfRsError>;
    fn get_occ(&self, mo_energy: &[f64]) -> Result<Vec<f64>, PyscfRsError>;
    fn make_rdm1(&self, mo: &MOCoefficients) -> Result<Density, PyscfRsError>;
    fn energy_elec(&self, dm: &Density, h1e: &Density, vhf: &Density)
        -> Result<(Energy, Energy), PyscfRsError>;
    fn energy_tot(&self, dm: &Density, h1e: &Density, vhf: &Density)
        -> Result<Energy, PyscfRsError>;
}

/// Zero-cost Rust-only impl. Used when SCF is driven from Rust (no Python).
/// Used as the public-default forwarded to by the `#[pymethods]` defaults in
/// pyscf-py (the "no override detected" path).
pub struct NoOverrides;

impl OverrideHooks for NoOverrides {
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        default_get_hcore(mol)   // pub fn in pyscf-scf
    }
    // ... 10 more default forwards
}

/// Generic kernel — works for Rust-only and Python-driven SCF identically.
/// Phase 3 ships this signature; Phase 7 geomopt consumes it via `as_scanner`.
pub fn kernel<H: OverrideHooks>(
    mol: &Mole,
    hooks: &H,
    cfg: KernelConfig,
) -> Result<ScfResult, PyscfRsError> { /* ... see Code Example #2 below */ }
```

```rust
// crates/pyscf-py/src/scf.rs — bridge impl + #[pyclass(subclass)]
// Source: xcfun-py functional.rs:148-279 pattern + PyO3 0.28 guide.
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::sync::PyOnceLock;     // PyO3 0.28 free-thread-safe replacement for GILOnceCell
use pyscf_core::{Density, Energy, MOCoefficients, Mole, error::PyscfRsError};
use pyscf_scf::{OverrideHooks, NoOverrides, kernel, ScfResult};

/// Bridge — captures the Python `slf` so call_method1 can dispatch to user
/// subclass methods. Stored on the #[pyclass(subclass)] wrapper.
struct PyOverrideBridge<'py> {
    slf: Bound<'py, PyAny>,
    py: Python<'py>,
}

impl<'py> OverrideHooks for PyOverrideBridge<'py> {
    fn get_jk(&self, mol: &Mole, dm: &Density) -> Result<(Density, Density), PyscfRsError> {
        // BIND-07: dispatch via slf.call_method1, NOT via Rust MRO.
        // The mol/dm conversion to Python is the BIND-04 boundary.
        let py_mol = mol_to_pyany(self.py, mol)?;
        let py_dm  = crate::numpy_io::density_to_pyarray(self.py, dm)?;
        let args   = PyTuple::new(self.py, &[py_mol, py_dm.into_any()])?;
        let result = self.slf
            .call_method1("get_jk", args)
            .map_err(crate::errors::py_to_pyscf)?;
        // result is (j, k) tuple of ndarrays.
        let (j, k): (Bound<'_, PyAny>, Bound<'_, PyAny>) = result.extract()
            .map_err(crate::errors::py_to_pyscf)?;
        Ok((
            crate::numpy_io::pyany_to_density(j)?,
            crate::numpy_io::pyany_to_density(k)?,
        ))
    }
    // ... 10 more hooks following the same shape
}

/// #[pyclass(subclass)] enables Python users to subclass `scf.RHF` and
/// override hooks. Subclass-override dispatch is automatic via Python MRO
/// when the bridge calls `slf.call_method1(...)` — Python looks up the method
/// on the most-derived class first.
#[pyclass(subclass, name = "RHF", module = "pyscf._native.scf")]
pub struct PyRHF {
    mol: Py<PyAny>,       // Mole, opaque from Rust's POV
    // SCF-14 30-attribute floor wired below via #[pymethods] #[getter]/#[setter]
    mo_coeff: Option<MOCoefficients>,
    mo_energy: Option<Vec<f64>>,
    mo_occ: Option<Vec<f64>>,
    e_tot: f64,
    converged: bool,
    cycles: u32,
    chkfile: Option<String>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    max_cycle: u32,
    init_guess: String,
    diis: bool,
    diis_space: u32,
    diis_start_cycle: u32,
    diis_damp: f64,
    level_shift: f64,
    damp: f64,
    direct_scf: bool,
    direct_scf_tol: f64,
    verbose: u8,
    max_memory: f64,
    // ... 30+ total per SCF-14 enumeration in §11 below
}

#[pymethods]
impl PyRHF {
    #[new]
    fn new(mol: Py<PyAny>) -> Self {
        // Pull defaults from pyscf/scf/hf.py:1689-1712 (verified line numbers).
        PyRHF { mol, mo_coeff: None, /* ... */
                conv_tol: 1e-9, max_cycle: 50, init_guess: "minao".to_string(),
                diis: true, diis_space: 8, diis_start_cycle: 1, /* ... */ }
    }

    /// The SCF driver. Called by Python: `mf.kernel()` or `mf.kernel(dm0)`.
    /// Detects override fast-path via PyOnceLock<HashSet<TypeId>> cache so the
    /// no-override case skips the call_method1 round-trip.
    fn kernel<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        dm0: Option<numpy::PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<f64> {
        // Per-hook detach happens INSIDE the kernel — at each hook call site.
        // Not whole-kernel detach (D-03 reason: override calls must re-attach).
        let mol_ref: Mole = mol_from_pyany(slf.borrow().mol.bind(py))?;
        let dm0_owned = dm0.map(|a| crate::numpy_io::to_density(a)).transpose()?;
        let hooks = PyOverrideBridge { slf: slf.into_any(), py };
        let result = pyscf_scf::kernel(&mol_ref, &hooks, /* cfg from slf */)
            .map_err(crate::errors::pyscf_to_py)?;
        Ok(result.e_tot.0)
    }

    // SCF-08: each overrideable hook has a #[pymethods] default that forwards
    // to pyscf-scf's Rust default function. Python subclasses override these
    // via standard MRO; the bridge's call_method1 dispatches to the override
    // if present, otherwise hits this default.
    fn get_jk<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        mol: Py<PyAny>,
        dm: numpy::PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, numpy::PyArray2<f64>>, Bound<'py, numpy::PyArray2<f64>>)> {
        let mol_ref = mol_from_pyany(mol.bind(py))?;
        let dm_ref = crate::numpy_io::to_density(dm)?;
        // PER-HOOK DETACH SEAM (D-03, BIND-05): release GIL during compute.
        let (j, k) = py.detach(|| pyscf_scf::default_get_jk(&mol_ref, &dm_ref))
            .map_err(crate::errors::pyscf_to_py)?;
        Ok((
            crate::numpy_io::density_to_pyarray(py, &j)?,
            crate::numpy_io::density_to_pyarray(py, &k)?,
        ))
    }
    // ... 9 more default hooks following the same shape
}

// Verified Python-side override test (Phase 3 BIND-07 CI assertion shape):
//
// ```python
// from pyscf import scf
// class MyHF(scf.RHF):
//     call_count = 0
//     def get_veff(self, mol, dm):
//         MyHF.call_count += 1
//         return super().get_veff(mol, dm)
// mf = MyHF(mol)
// mf.kernel()
// assert MyHF.call_count >= mf.cycles, \
//     f"override invoked {MyHF.call_count} times, expected at least {mf.cycles}"
// ```
```

### Pattern 2: Generic SCF Kernel (D-02)

```rust
// crates/pyscf-scf/src/lib.rs — Rust-only SCF API.
// Source: pyscf/scf/hf.py:48-244 (verbatim algorithm port).
pub struct KernelConfig {
    pub conv_tol: f64,           // default 1e-9 per pyscf/scf/hf.py:1689
    pub conv_tol_grad: Option<f64>,  // default sqrt(conv_tol) at runtime
    pub max_cycle: u32,          // default 50 per pyscf/scf/hf.py:1692
    pub diis_space: u32,         // default 8 per pyscf/scf/hf.py:1701
    pub diis_start_cycle: u32,   // default 1 per pyscf/scf/hf.py:1704
    pub level_shift: f64,
    pub damp: f64,
}

pub struct ScfResult {
    pub e_tot: Energy,
    pub mo_coeff: MOCoefficients,
    pub mo_energy: Vec<f64>,
    pub mo_occ: Vec<f64>,
    pub converged: bool,
    pub cycles: u32,
}

pub fn kernel<H: OverrideHooks>(
    mol: &Mole,
    hooks: &H,
    cfg: KernelConfig,
) -> Result<ScfResult, PyscfRsError> {
    // Phase 3 plan implements the SCF loop verbatim from pyscf/scf/hf.py:121-244:
    //   s1e = hooks.get_ovlp(mol)?
    //   dm  = dm0 OR hooks.get_init_guess(mol, &cfg.init_guess_key)?
    //   h1e = hooks.get_hcore(mol)?
    //   vhf = hooks.get_veff(mol, &dm)?
    //   e_tot = hooks.energy_tot(&dm, &h1e, &vhf)?
    //   diis = pyscf_diis::Diis::<FockSubspace>::new(cfg.diis_space)
    //   for cycle in 0..cfg.max_cycle {
    //       fock = hooks.get_fock(&h1e, &s1e, &vhf, &dm, cycle, &diis_state)?
    //       mo_e, mo_c = hooks.eig(&fock, &s1e)?
    //       mo_c = canonicalize_signs(mo_c)  // SCF-13, Pitfall 4
    //       mo_occ = hooks.get_occ(&mo_e)?
    //       dm = hooks.make_rdm1(...)?
    //       vhf = hooks.get_veff(mol, &dm)?
    //       e_new = hooks.energy_tot(&dm, &h1e, &vhf)?
    //       norm_gorb = check_grad(...) using oracle_dot/oracle_sum
    //       if abs(e_new - e_tot) < cfg.conv_tol && norm_gorb < conv_tol_grad { break }
    //       e_tot = e_new
    //   }
    todo!("plan 03-NN")
}
```

### Pattern 3: Per-Hook `Python::detach` (D-03, BIND-05, Pitfall 6)

```rust
// Inside pyscf-py's #[pymethods] default for each hook.
// Source: xcfun-py numpy_io.rs:57-72 (verified working pattern).
fn get_veff<'py>(
    slf: PyRef<'py, Self>,
    py: Python<'py>,
    mol: Py<PyAny>,
    dm: numpy::PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let mol_ref = mol_from_pyany(mol.bind(py))?;
    let dm_ref = crate::numpy_io::to_density(dm)?;
    // DETACH SCOPE: inside the per-hook body, around the heavy Rust compute.
    // The Rust closure must NOT itself need to reattach (it doesn't — Rust-side
    // hooks are pure compute on pyscf-algebra). Override path goes through
    // PyOverrideBridge which IS GIL-attached, so it never enters this detach
    // block.
    let veff = py.detach(|| pyscf_scf::default_get_veff(&mol_ref, &dm_ref))
        .map_err(crate::errors::pyscf_to_py)?;
    crate::numpy_io::density_to_pyarray(py, &veff)
}
```

**Deadlock surface (Pitfall 6):** If a `default_*` hook on the Rust side itself tries to call into Python (which it must not — Rust defaults are pure), the detach would cause a deadlock. The contract is: **Rust default hooks are pyo3-free by construction (algebra wall + D-01)** so they never need to reattach. Override hooks dispatch via `PyOverrideBridge` which is always GIL-attached (no detach in scope). The two paths cannot collide.

### Pattern 4: NumPy Boundary Discipline (D-04, BIND-04, Pitfall 5)

```rust
// crates/pyscf-py/src/numpy_io.rs
// Source: xcfun-py numpy_io.rs:9-36 + PyO3 0.28 §arrays.md.
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyscf_core::Density;

/// Type-specific Density converter — runs is_standard_layout AND
/// is_c_contiguous; calls to_owned() on any non-standard input.
/// Source: pyscf/scf/hf.py — density is C-order (row-major, [nao,nao]).
pub fn to_density<'py>(arr: PyReadonlyArray2<'py, f64>) -> PyResult<Density> {
    let shape = arr.shape();
    if shape[0] != shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("density must be square, got {:?}", shape)));
    }
    // BIND-04: if not standard C-layout, copy to owned C-order ndarray.
    // is_standard_layout() returns true for both C- and F-contiguous;
    // is_c_contiguous() narrows to C-order only.
    let data: Vec<f64> = if arr.is_c_contiguous() {
        arr.as_slice()?.to_vec()
    } else {
        // to_owned() reallocates as default-order (C-contiguous) ndarray.
        arr.as_array().to_owned().into_raw_vec_and_offset().0
    };
    Ok(Density { nao: shape[0], data })
}

/// MO coefficients are F-order per pyscf-core::MOCoefficients doc comment
/// (matches scipy.linalg.eigh + LAPACK convention; Pitfall 8 mitigation).
pub fn to_mo_coeff<'py>(arr: PyReadonlyArray2<'py, f64>) -> PyResult<MOCoefficients> {
    let shape = arr.shape();
    let (nao, nmo) = (shape[0], shape[1]);
    // For F-order output: if input is not F-contiguous, transpose-copy.
    let data: Vec<f64> = if is_f_contiguous(&arr) {
        arr.as_slice()?.to_vec()
    } else {
        // Build F-order flat from the ndarray view.
        let view = arr.as_array();
        let mut buf = Vec::with_capacity(nao * nmo);
        for j in 0..nmo { for i in 0..nao { buf.push(view[[i, j]]); } }
        buf
    };
    Ok(MOCoefficients { nao, nmo, data, energies: vec![], occupations: vec![] })
}

fn is_f_contiguous(arr: &PyReadonlyArray2<'_, f64>) -> bool {
    arr.strides()[0] == std::mem::size_of::<f64>() as isize
}

/// Density output — always C-contiguous per the SCF convention.
pub fn density_to_pyarray<'py>(py: Python<'py>, d: &Density)
    -> PyResult<Bound<'py, PyArray2<f64>>>
{
    let arr = ndarray::Array2::from_shape_vec((d.nao, d.nao), d.data.clone())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}
```

**BIND-04 stride-fuzz test (pytest contract):**
```python
# python/pyscf/tests/test_scf_stride_fuzz.py
import numpy as np
from pyscf import gto, scf

def test_get_veff_accepts_views():
    mol = gto.M(atom='O 0 0 0; H 0 1 0; H 1 0 0', basis='cc-pvdz')
    mf  = scf.RHF(mol).run()  # converged once
    nao = mol.nao_nr()
    dm = mf.make_rdm1()                                          # canonical C-contiguous
    # Build 4 stride-equivalent views:
    base    = np.asfortranarray(dm)                              # F-order
    view_a  = dm                                                 # C-contig
    view_b  = dm.T                                               # F-contig
    view_c  = np.zeros((nao*2, nao*2))                           # strided
    view_c[::2, ::2] = dm
    view_c  = view_c[::2, ::2]
    view_d  = np.zeros((nao, nao+5))[:, 1:nao+1]                 # offset C-strided
    view_d[:] = dm
    out_a = mf.get_veff(mol, view_a)
    out_b = mf.get_veff(mol, view_b)
    out_c = mf.get_veff(mol, view_c)
    out_d = mf.get_veff(mol, view_d)
    np.testing.assert_array_equal(out_a, out_b)
    np.testing.assert_array_equal(out_a, out_c)
    np.testing.assert_array_equal(out_a, out_d)
```

### Pattern 5: Panic → Python Exception (BIND-09, Pitfall 14)

```rust
// crates/pyscf-py/src/errors.rs
// Source: xcfun-py errors.rs:31-83 (verified working pattern).
// CRITICAL: abi3-py310 forbids #[pyclass(extends=PyException)] until Python 3.12.
// Must use create_exception! + Python-side __init__.py shim.

use pyo3::PyErr;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyscf_core::error::PyscfRsError;

create_exception!(_native, PyscfRsError as PyscfRsRuntimeError, PyException);

pub fn pyscf_to_py(err: PyscfRsError) -> PyErr {
    let kind = match &err {
        PyscfRsError::Core(_)             => "Core",
        PyscfRsError::NotYetImplemented{..} => "NotYetImplemented",
        PyscfRsError::ConvergenceFailure{..} => "ConvergenceFailure",
        PyscfRsError::BasisLoad(_)        => "BasisLoad",
        PyscfRsError::EcpLoad(_)          => "EcpLoad",
        PyscfRsError::EcpEngineNotAvailable => "EcpEngineNotAvailable",
        // Phase 3 will add SCF-specific variants: e.g., Diis(_), Chkfile(_).
    };
    let msg = format!("{}", err);
    // Preserve source chain through positional args; Python __init__.py shim
    // unpacks args into .kind / .source attributes.
    let source_chain = collect_source_chain(&err);
    PyscfRsRuntimeError::new_err((msg, kind, source_chain))
}

/// Walks the Rust error chain via std::error::Error::source() so the Python
/// exception's __cause__ reflects the full Rust error tree (BIND-09 contract).
fn collect_source_chain(err: &dyn std::error::Error) -> Vec<String> {
    let mut chain = vec![];
    let mut cur: Option<&dyn std::error::Error> = err.source();
    while let Some(e) = cur {
        chain.push(format!("{}", e));
        cur = e.source();
    }
    chain
}
```

```python
# python/pyscf/__init__.py — graft .kind and .source_chain attrs onto exception
from pyscf._native import PyscfRsRuntimeError as _PyscfRsBase
class PyscfRsError(_PyscfRsBase):
    """Base exception for pyscf-rs Rust-side errors.

    Attributes:
        kind: Rust error variant name (e.g., 'ConvergenceFailure').
        source_chain: list of `str(err.source())` walking the Rust error tree.
    """
    @property
    def kind(self) -> str:    return self.args[1]
    @property
    def source_chain(self) -> list[str]:  return self.args[2]
```

### Pattern 6: HDF5 Chkfile (D-05, D-06, SCF-10, ORACLE-08, Pitfall 11)

```rust
// crates/pyscf-chkfile/src/lib.rs — pure-I/O crate, NO algebra dep.
// Source schema: pyscf/scf/chkfile.py:25-42 + pyscf/lib/chkfile.py:28-191.
use hdf5_metno as hdf5;
use ndarray::{Array1, Array2};
use std::path::Path;

/// All method crates implement this on their result type.
/// Source: D-06 — schema lives per-method (`pyscf_scf::chkfile`,
/// `pyscf_dft::chkfile`, etc.) so each method owns its own keys.
pub trait Checkpointable: Sized {
    /// Write under the given HDF5 Group (e.g. group "/scf").
    /// Mole serialization is handled separately at the file root (see
    /// `write_mol` below).
    fn dump(&self, group: &hdf5::Group) -> Result<(), ChkfileError>;
    fn load(group: &hdf5::Group) -> Result<Self, ChkfileError>;
}

/// Open chkfile for write. Mirrors pyscf/lib/chkfile.py:dump
/// behaviour: if file exists, append; otherwise create.
pub fn open_for_write<P: AsRef<Path>>(path: P) -> Result<hdf5::File, ChkfileError> {
    if path.as_ref().exists() {
        hdf5::File::append(path).map_err(ChkfileError::from)
    } else {
        hdf5::File::create(path).map_err(ChkfileError::from)
    }
}

/// Write mol.dumps() JSON string under `/mol`.
/// Source: pyscf/lib/chkfile.py:179-191 save_mol — writes `mol.dumps()` (str)
/// at HDF5 key 'mol'. Read as: `gto.loads(fh5['mol'][()])`.
/// h5py reads h5py-written str datasets as bytes -> str round-trips via
/// utf8; hdf5-metno's String dataset type matches (verified via README
/// example "Write string datasets (h5py compatible)").
pub fn write_mol(file: &hdf5::File, mol_json: &str) -> Result<(), ChkfileError> {
    file.new_dataset::<hdf5::types::VarLenUnicode>()
        .create("mol")?
        .write_scalar(&hdf5::types::VarLenUnicode::from_str(mol_json)?)?;
    Ok(())
}

/// Write a 2D f64 dataset in C-order (default). Used for `mo_occ`, `mo_energy`
/// (1D), `e_tot` (scalar).
pub fn write_dataset_c_order(group: &hdf5::Group, key: &str, data: &Array2<f64>)
    -> Result<(), ChkfileError>
{
    group.new_dataset::<f64>()
        .shape(data.shape())
        .create(key)?
        .write(data)?;
    Ok(())
}

/// Write a 2D f64 dataset preserving F-order layout for `mo_coeff`.
/// Source: pyscf/scf/chkfile.py:28-42 dump_scf writes mo_coeff directly from
/// upstream SCF result, which is F-order (LAPACK convention; see Pitfall 8).
/// hdf5-metno preserves contiguity from ndarray::ArrayBase strides.
pub fn write_dataset_f_order(group: &hdf5::Group, key: &str,
                              data: ndarray::ArrayView2<f64>) -> Result<(), ChkfileError>
{
    // Ensure F-contiguous view (transpose-copy if needed) before write.
    let f_owned = data.as_standard_layout().reversed_axes().to_owned();
    group.new_dataset::<f64>()
        .shape(f_owned.shape())
        .create(key)?
        .write(&f_owned)?;
    Ok(())
}
```

```rust
// crates/pyscf-scf/src/chkfile.rs — per-method schema.
// Source: pyscf/scf/chkfile.py:28-42 dump_scf — group 'scf' with keys
// e_tot, mo_energy, mo_occ, mo_coeff.
use pyscf_chkfile::{Checkpointable, ChkfileError};

pub struct ScfResult { /* ... */ }

impl Checkpointable for ScfResult {
    fn dump(&self, group: &hdf5::Group) -> Result<(), ChkfileError> {
        // Sub-group key MUST be 'scf' to match upstream PySCF schema (verified
        // at pyscf/scf/chkfile.py:42 — `save(chkfile, 'scf', scf_dic)`).
        let scf = group.create_group("scf")?;
        scf.new_dataset::<f64>().create("e_tot")?.write_scalar(&self.e_tot.0)?;
        // mo_energy: 1D, C-order — upstream stores via h5py's default order.
        let mo_e = ndarray::Array1::from_vec(self.mo_energy.clone());
        scf.new_dataset::<f64>().create("mo_energy")?.write(&mo_e)?;
        let mo_o = ndarray::Array1::from_vec(self.mo_occ.clone());
        scf.new_dataset::<f64>().create("mo_occ")?.write(&mo_o)?;
        // mo_coeff: 2D, F-order per pyscf/scf/chkfile.py upstream convention.
        let mo_c_view = self.mo_coeff_as_f_view();  // helper on MOCoefficients
        pyscf_chkfile::write_dataset_f_order(&scf, "mo_coeff", mo_c_view)?;
        Ok(())
    }

    fn load(group: &hdf5::Group) -> Result<Self, ChkfileError> {
        let scf = group.group("scf")?;
        let e_tot: f64 = scf.dataset("e_tot")?.read_scalar()?;
        let mo_energy: Vec<f64> = scf.dataset("mo_energy")?.read_1d()?.to_vec();
        let mo_occ: Vec<f64>    = scf.dataset("mo_occ")?.read_1d()?.to_vec();
        let mo_c_raw: ndarray::Array2<f64> = scf.dataset("mo_coeff")?.read_2d()?;
        // Internal repr is F-order; upstream chkfile is F-order.
        Ok(ScfResult { /* ... */ })
    }
}
```

### Pattern 7: C-DIIS (SCF-04, D-08, D-09, Pitfall 9)

```rust
// crates/pyscf-diis/src/lib.rs — generic Pulay extrapolation.
// Source: pyscf/scf/diis.py:40-66 CDIIS + diis.py:68-87 get_err_vec.

pub trait DiisStorable {
    fn as_flat(&self) -> &[f64];
    fn from_flat(&mut self, slice: &[f64]);
    fn dot(&self, other: &Self) -> f64;
    fn len(&self) -> usize;
}

pub struct Diis<S: DiisStorable + Clone> {
    space: usize,           // 8 per SCF default
    start_cycle: usize,     // 1 per SCF default
    bookkeep: Vec<S>,       // ring buffer of past iterates
    error_vecs: Vec<Vec<f64>>,
    head: usize,
}

impl<S: DiisStorable + Clone> Diis<S> {
    pub fn new(space: usize) -> Self { /* ... */ }

    /// Extrapolated next iterate.
    /// Source: Pulay 1980 (DOI:10.1016/0009-2614(80)80396-4) +
    /// pyscf/scf/diis.py:48-58 (update method).
    /// All inner products go through pyscf-algebra::oracle_dot under
    /// release-oracle so the B-matrix is bit-exact reproducible.
    pub fn extrapolate(&mut self, current: S, error: Vec<f64>) -> Result<S, DiisError> {
        self.push(current, error);
        let n = self.bookkeep.len();
        // B[i,j] = oracle_dot(error_i, error_j) — Pitfall 9 mitigation
        let mut b = vec![0.0; (n + 1) * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                b[i * (n + 1) + j] = pyscf_algebra::oracle_dot(
                    &self.error_vecs[i], &self.error_vecs[j]);
            }
            b[i * (n + 1) + n] = -1.0;
            b[n * (n + 1) + i] = -1.0;
        }
        b[n * (n + 1) + n] = 0.0;
        let mut rhs = vec![0.0; n + 1];
        rhs[n] = -1.0;
        // Solve via pyscf-algebra. If pyscf-algebra::solve_linear doesn't
        // exist yet, Phase 3 adds it as a host-faer thin wrapper around
        // faer::solvers::Lu.
        let c = pyscf_algebra::solve_linear(&b, &rhs, n + 1)?;
        // Extrapolated iterate = sum(c[i] * bookkeep[i]).
        let mut extrap = self.bookkeep[0].clone();
        let extrap_flat: Vec<f64> = (0..extrap.len())
            .map(|k| pyscf_algebra::oracle_sum(
                (0..n).map(|i| c[i] * self.bookkeep[i].as_flat()[k])))
            .collect();
        extrap.from_flat(&extrap_flat);
        Ok(extrap)
    }
}

/// SCF Fock-matrix subspace. Phase 3 ships.
/// Source: pyscf/scf/diis.py:68-87 get_err_vec_orig — error vec = SDF - FDS.
pub struct FockSubspace {
    pub fock: Vec<f64>,   // nao × nao, F-order
    pub nao: usize,
}

impl DiisStorable for FockSubspace {
    fn as_flat(&self) -> &[f64]  { &self.fock }
    fn from_flat(&mut self, s: &[f64])  { self.fock.copy_from_slice(s); }
    fn dot(&self, other: &Self) -> f64 {
        pyscf_algebra::oracle_dot(&self.fock, &other.fock)
    }
    fn len(&self) -> usize { self.fock.len() }
}

/// SDF - FDS error vector (upstream: pyscf/scf/diis.py:69).
pub fn err_vec_scf(s: &[f64], d: &[f64], f: &[f64], nao: usize) -> Vec<f64> {
    // sdf = S @ D @ F via pyscf-algebra::gemm chain
    let sd  = matmul(s, d, nao);
    let sdf = matmul(&sd, f, nao);
    // fds = F @ D @ S
    let fd  = matmul(f, d, nao);
    let fds = matmul(&fd, s, nao);
    sdf.iter().zip(&fds).map(|(a, b)| a - b).collect()
}
```

### Pattern 8: DF-HF (D-10, D-11, SCF-07)

```rust
// crates/pyscf-df/src/lib.rs
// Source: pyscf/df/df.py:41-200 DF class + pyscf/df/incore.py cholesky_eri.
use pyscf_core::{Mole, error::PyscfRsError};

pub struct DfIntegrals {
    /// `B[μν,Q]` — three-index integrals × Cholesky-decomposed (P|Q)^{-1/2}.
    /// Layout: flattened `[nao*nao, naux]` in F-order, or `[nao*(nao+1)/2, naux]`
    /// triangular (upstream uses s2 triangular packing — see pyscf/df/incore.py).
    /// Phase 3: ships triangular packing to match upstream byte-for-byte under
    /// release-oracle.
    pub b_uvq: Vec<f64>,
    pub naux: usize,
    pub nao: usize,
}

pub fn cholesky_eri(mol: &Mole, auxbasis: &str) -> Result<DfIntegrals, PyscfRsError> {
    // 1. Build auxmol with auxbasis (see auxbasis.rs DEFAULT_AUXBASIS table).
    let auxmol = make_auxmol(mol, auxbasis)?;
    // 2. 3-center integrals via cintx int3c2e_sph.
    //    Source: pyscf/df/incore.py:cholesky_eri — int3c2e + s2 packing.
    let int3c = mol.intor_via_pyscf_gto("int3c2e_sph", Some(&auxmol))?;
    // 3. 2-center integrals (P|Q).
    let int2c = auxmol.intor("int2c2e_sph")?;
    // 4. Cholesky decomposition L L^T = (P|Q) via pyscf-algebra::cholesky.
    let l = pyscf_algebra::cholesky(&int2c)?;
    // 5. Solve L · B = int3c for B (forward triangular solve).
    //    Phase 3 plan ships pyscf-algebra::triangular_solve or extends with
    //    a backsubst kernel; faer 0.24 has `lin_solver::triangular`.
    let b_uvq = triangular_solve_forward(&l, &int3c)?;
    Ok(DfIntegrals { b_uvq, naux: int2c.shape()[0], nao: mol.nao_nr })
}
```

**Default auxbasis table (Phase 3 plan ports verbatim from `pyscf/df/addons.py:DEFAULT_AUXBASIS`):** keys are basis names (e.g., `'cc-pvdz'`), values are tuples `(jkfit_name, ri_name)`. Resolution:
- `mol.basis == 'cc-pvdz'` → JK-fit aux defaults to `'cc-pvdz-jkfit'` (Weigend); RI defaults to `'cc-pvdz-ri'`.
- `mol.basis == 'def2-svp'` → `'def2-svp-jkfit'` / `'def2-svp-ri'`.
- Fallback: `'weigend'` universal aux (Weigend, Häser, Ahlrichs 2002).

### Pattern 9: Sign Canonicalization (SCF-13, Pitfall 4 + 12)

```rust
// crates/pyscf-core/src/lib_fn.rs — pure function, no algebra deps.
// Source: pyscf/scf/hf.py:1349-1357 (def eig — inline algorithm).
//
// Note: upstream PySCF does NOT have a named `canonicalize_signs` function —
// the rule lives inline in `def eig`. Phase 3 extracts it so MP2/CCSD post-eigh
// (which also need vendor-stable eigenvectors) can call the same function.

/// For each MO column j, find the index i_max where |c[i_max, j]| is largest
/// (ties broken by lowest index per numpy.argmax semantics). If the value
/// c[i_max, j] is negative, flip the sign of the entire column j.
///
/// This makes the eigenvector sign reproducible across LAPACK vendors
/// (MKL/OpenBLAS/Accelerate may pick opposite signs for the same eigenpair).
pub fn canonicalize_signs(c: &mut [f64], nao: usize, nmo: usize) {
    // c is F-order: c[i + j*nao] = element (i, j)
    for j in 0..nmo {
        let col_start = j * nao;
        let mut i_max = 0usize;
        let mut abs_max = c[col_start].abs();
        for i in 1..nao {
            let v = c[col_start + i].abs();
            if v > abs_max {  // strict > so numpy.argmax tie-break preserved
                abs_max = v;
                i_max = i;
            }
        }
        if c[col_start + i_max] < 0.0 {
            for i in 0..nao {
                c[col_start + i] = -c[col_start + i];
            }
        }
    }
}
```

### Pattern 10: Maturin `[tool.maturin]` Config (BIND-01, BIND-02)

```toml
# pyproject.toml at repo root (new file)
# Source: xcfun-py/pyproject.toml:47-51 (verbatim mirrored).
[build-system]
requires = ["maturin>=1.12,<2.0"]
build-backend = "maturin"

[project]
name = "pyscf-rs"
version = "0.1.0"
description = "Pure-Rust rewrite of PySCF with drop-in pyscf.* import surface"
requires-python = ">=3.10"
authors = [{ name = "pyscf-rs contributors" }]
license = { text = "Apache-2.0" }
dependencies = ["numpy>=1.26"]

[tool.maturin]
python-source = "python"                # overlay packaging — `python/pyscf/__init__.py`
module-name   = "pyscf._native"         # native cdylib name matches PyO3 #[pymodule] fn _native
features      = ["pyo3/extension-module", "pyo3/abi3-py310"]
manifest-path = "crates/pyscf-py/Cargo.toml"
strip         = true
```

```python
# python/pyscf/__init__.py (new overlay)
# Re-exports from the _native cdylib so `from pyscf import scf` resolves
# to pyscf-rs's implementation.
# Source: xcfun-py/python/xcfun_rs/__init__.py pattern.
from pyscf._native import scf  # type: ignore[attr-defined]

# Re-export the panic→exception class with .kind/.source_chain grafted.
from pyscf._native import PyscfRsRuntimeError as _PyscfRsBase

class PyscfRsError(_PyscfRsBase):
    @property
    def kind(self) -> str:        return self.args[1]
    @property
    def source_chain(self) -> list[str]:  return self.args[2]

# Mole + M() factory remain at python/pyscf/__init__.py from Phase 2's import path.
# (Phase 2 ships `pyscf_gto::M(...)` accessible via Python; Phase 3 doesn't move it.)
```

### Anti-Patterns to Avoid

- **NEVER dispatch to Python-overrideable hooks via Rust trait MRO**: that bypasses the bridge. ALL overrideable hook calls inside the kernel go through `H: OverrideHooks` (Rust trait dispatch resolves to PyOverrideBridge when called from PyO3 path). The bridge then uses `slf.call_method1` to surface Python overrides. (Pitfall 7 mitigation — D-01 trait-bridge IS the answer.)
- **NEVER hold the GIL during long compute**: every `default_*` hook in `pyscf-py` must wrap its Rust call in `py.detach(|| ...)`. Forgetting this hangs python3.13t free-threaded CI. (Pitfall 6.)
- **NEVER use `as_array()` on a non-contiguous PyReadonlyArray without checking layout**: `as_array()` returns a view that respects strides, but downstream Rust code expecting `&[f64]` slices will crash or get garbage. Always `is_c_contiguous()` (or `is_standard_layout()`) check + `to_owned()` fallback. (Pitfall 5.)
- **NEVER use `#[pyclass(extends=PyException)]` under abi3-py310**: it fails to build for Python 3.10/3.11 even though it compiles. Always `create_exception!` macro + Python overlay shim. (BIND-09 trap — verified in PyO3 0.28 guide §exception.md.)
- **NEVER use `std::sync::OnceLock`/`std::sync::LazyLock`/`lazy_static!` for caches that hold PyObjects**: these deadlock under free-threaded Python. Use `pyo3::sync::PyOnceLock` (or `GILOnceCell` for legacy). (BIND-06.)
- **NEVER reach for `cubecl-*` directly** from `pyscf-scf`/`pyscf-diis`/`pyscf-df`: violates the algebra wall (ALG-06 lint). All compute primitives go through `pyscf-algebra::{gemm,eigh,oracle_sum,oracle_dot,cholesky,solve_linear,...}`.
- **NEVER use `numpy.argmax` semantics naively in `canonicalize_signs`**: must use strict-greater-than comparison so ties break to lowest index (matching `numpy.argmax` exactly). A simple `if v >= max` would give last-index tie-break, breaking Pitfall 12 cross-platform µHartree assertion.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HDF5 file I/O | Custom HDF5 writer | `hdf5-metno = "0.10.0"` with `static` feature | h5py-compatibility is non-trivial — VL strings, F-vs-C order, attribute scoping, type ABI all hard to reproduce |
| JSON serialization of Mole | Manual JSON writer | `serde_json` + Phase 2's `Mole::dumps()` | Phase 2 already shipped this surface; chkfile's `/mol` key uses it verbatim |
| Linear solver for 8×8 DIIS B-matrix | Hand-rolled Gauss-Jordan | `pyscf-algebra::solve_linear` (host-faer fallback) | Bit-exact reproducibility under release-oracle; one decision point |
| Pulay extrapolation logic | Custom DIIS | `pyscf-diis::Diis<S: DiisStorable>` (new crate D-08) | Pitfall 9 — reductions must route through `oracle_sum`/`oracle_dot`; shared with Phase 6 CCSD-04 |
| 3-center / 2-center integrals | Custom Boys-function | `mol.intor('int3c2e_sph')` / `int2c2e_sph` via cintx | Pitfall 18 — delegated to cintx (Phase 2 D-04) |
| Cholesky of `(P|Q)` | Custom Cholesky | `pyscf-algebra::cholesky` (host-faer) | Phase 1 D-06 already provides; ALG-05 host-fallback path |
| Subclass detection (whether Python user overrode hook X) | Custom Python C-API calls | `pyo3::sync::PyOnceLock<HashSet<PyTypeId>>` cache filled lazily | BIND-06; deadlock-free under free-threaded Python |
| `mol.dumps()` round-trip in chkfile | Custom JSON-in-HDF5 | `hdf5_metno::types::VarLenUnicode` dataset | h5py reads/writes VL strings; matches upstream `pyscf/lib/chkfile.py:save_mol` |
| Panic catching at FFI boundary | Manual `catch_unwind` everywhere | `pyo3::panic::PanicException` (default) + `create_exception!` for typed errors | PyO3 wraps panics into `PanicException` automatically; Phase 1 lint already enforces `catch_unwind` on `extern "C"` callbacks (FOUND-07) |
| abi3 wheel matrix | Custom build script | `maturin develop` (skeleton — Phase 3) → Phase 8 `maturin build --release` (full matrix) | Phase 8 owns the full wheel matrix per ROADMAP.md |
| Sign-flip canonicalization | Custom algorithm | `pyscf-core::lib::canonicalize_signs` (port from `pyscf/scf/hf.py:1349-1357`) | Pitfall 4 + 12 cross-platform vendor stability anchor |

**Key insight:** Phase 3 is largely composition over invention — every chemistry routine has a verbatim upstream source. The novelty is in (1) the PyO3 contract patterns (lockable on RHF before DFT explodes the surface area), and (2) the empirical seal on h5py↔hdf5-metno round-trip.

## Runtime State Inventory

**Trigger:** Phase 3 is a greenfield phase (adds new crates, new wheel module, new Python overlay). Not a rename/refactor. The Runtime State Inventory below is a defensive check anyway — when Phase 3 first lands on a developer's machine, what stateful artifacts could interfere?

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | **None.** No databases, no persistent stores in this phase. SCF chkfiles are user-controlled per-calculation artifacts (transient or named per `mf.chkfile = path`); the project itself stores no SCF state. | None |
| Live service config | **None.** No external services. (Phase 8 DIST-* introduces PyPI publishing — that's a Phase 8 concern.) | None |
| OS-registered state | **None.** No `setup.py develop` registrations to revert; the existing upstream `pyscf/` tree is read-only reference per Phase 1 D-03. **However**: if a developer ran `pip install pyscf` from PyPI in their env, the overlay test assertions in BIND-02 (verify `import pyscf` resolves to our `python/pyscf/__init__.py`) will fail. Phase 3 plan should include a CI step that creates a clean Python venv before running pytest. | CI step: `python -m venv /tmp/pyscf-rs-test && source /tmp/pyscf-rs-test/bin/activate && maturin develop` |
| Secrets/env vars | **None new** in Phase 3. Consumes existing `PYSCF_BACKEND` / `PYSCF_DTYPE` / `PYSCF_BASIS_PATH` from Phase 1/2. Surfaces `PYSCF_MAX_MEMORY` only in kernel-entry log (no enforcement). | None |
| Build artifacts | **Three new crates** (`pyscf-chkfile`, `pyscf-diis`, `pyscf-df`) — `target/` directory will grow. `maturin develop` writes to `python/pyscf/_native*.so` (Linux) / `.dylib` / `.pyd`; this artifact is `.gitignore`-required. `hdf5-metno = "0.10.0"` with `static` feature triggers a libhdf5 build from source on first compile (~5min — acceptable but noticeable; cache via CI's `actions/cache@v4` on `~/.cargo` and `target/`). | Add to `.gitignore`: `python/pyscf/_native*`, `python/pyscf/**/*.so`, `python/pyscf/**/*.dylib`, `python/pyscf/**/*.pyd`. Add CI cache step keyed on `Cargo.lock` |

**The canonical question:** *After every file in the repo is updated, what runtime systems still have the old string cached, stored, or registered?*

**Answer:** None — Phase 3 introduces new artifacts only. The only pre-existing risk vector is a stale `pip install pyscf` in the developer's venv masking the overlay; the CI clean-venv step closes this.

## Common Pitfalls

### Pitfall 7: PyO3 Subclass Override Bypass (SHOWSTOPPER — primary)

**What goes wrong:** Rust kernel dispatches to a Rust trait method directly (via Rust MRO / static dispatch). When a Python user defines `class MyHF(scf.RHF): def get_veff(self, mol, dm): return super().get_veff(mol, dm) + correction(dm)`, the kernel never invokes their override — it bypasses Python entirely. Energy is wrong by `correction(dm)` and there's no error.

**Why it happens:** Naive impl pattern: `kernel(&self) { let v = self.get_veff(...); }` where `self.get_veff` is a Rust method. The Rust compiler resolves to `<PyRHF as Scf>::get_veff` — never to the Python override.

**How to avoid:** D-01 trait-callback bridge. The kernel takes `H: OverrideHooks`; pyscf-py's `PyOverrideBridge` impl unconditionally routes through `slf.call_method1(py, "get_veff", args)`. Python MRO then dispatches to the most-derived class's method — the user override if defined, otherwise the `#[pymethods] fn get_veff` default in `PyRHF` (which forwards to `pyscf-scf::default_get_veff`).

**Warning signs:** A CI assertion: subclass with a counter on `get_veff`, run `mf.kernel()`, assert counter >= `mf.cycles`. Failure means the bridge is bypassed.

### Pitfall 5: NumPy Stride / Layout Surprise (MAJOR)

**What goes wrong:** User passes `mf.get_veff(mol, dm.T)` (transposed view — F-contiguous, not C-contiguous). Rust treats the strided view as flat C-contiguous, computes garbage Fock matrix, SCF converges to wrong energy.

**Why it happens:** `PyReadonlyArray2::as_slice()` works only on C-contiguous arrays. Calling it on a non-contiguous view returns `Err`, but if the code uses `as_array()` (returns strided `ArrayView`) and then naively reads via flat indexing, the strides are silently violated.

**How to avoid:** Every NumPy boundary converter in `pyscf-py/src/numpy_io.rs` MUST run `is_c_contiguous()` check + `to_owned()` fallback BEFORE constructing the pyscf-core type. Greppable in CI. BIND-04 stride-fuzz test (see Pattern 4 above) exercises 4 stride variants per entry point.

**Warning signs:** Stride-fuzz test fails on `a.T` or `a[::2]`. Or pytest converges to a different energy depending on whether the user transposed the input.

### Pitfall 6: GIL Deadlock Under Free-Threaded Python (MAJOR)

**What goes wrong:** `py.detach` released the GIL; the detached closure needs to call back into Python (e.g., via a user override during what was supposed to be a "pure Rust default" path); Python::attach inside the closure deadlocks because the global synchronization event blocks.

**Why it happens:** `Python::detach` lets other Python threads run. If one of them holds a lock that the closure needs to re-acquire, you get a deadlock. Under free-threaded Python 3.13t, this is more likely because there's no GIL serialization fallback.

**How to avoid:** D-03 — per-hook detach (NOT whole-kernel detach). Each `#[pymethods] fn` default body wraps ONLY its Rust compute in `py.detach`. Override path is GIL-attached by construction (PyOverrideBridge uses `Python::attach` to re-acquire if needed). Rust default hooks are pure compute on `pyscf-algebra` types — they NEVER reach back into Python. The two paths cannot collide.

**Warning signs:** `python3.13t` CI build hangs on H2O/cc-pVDZ smoke. Diagnose with `pyo3::sync::OnceExt::call_once_py_attached` if a static initializer is involved.

### Pitfall 9: DIIS Path Drift (MAJOR)

**What goes wrong:** C-DIIS B-matrix is computed slightly differently on Linux x86_64 vs macOS aarch64 (different reduction orders in the inner products); extrapolated Fock differs by ~1e-14 per iteration; over 30 cycles, the cumulative drift exceeds 1 µHartree; cross-platform assertion fails.

**Why it happens:** Naive `let b_ij = dot(err_i, err_j);` uses `pyscf-algebra::dot` which is NOT bit-exact — depends on backend's parallel reduction strategy.

**How to avoid:** Every reduction inside DIIS goes through `pyscf-algebra::oracle_dot` (Phase 1 D-06). Under `release-oracle` profile, `oracle_dot` performs strict left-to-right pairwise reduction — bit-exact across platforms (Phase 1 FOUND-05/06 mitigation). Same for the `extrapolated = sum(c_i * F_i)` reduction (use `oracle_sum`).

**Warning signs:** Linux+macOS µHartree assertion in CI (Pitfall 12 mitigation) fails by >0.5 µHartree on a corpus run.

### Pitfall 11: chkfile h5py Incompatibility (SHOWSTOPPER)

**What goes wrong:** pyscf-rs writes chkfile with hdf5-metno; user opens it with h5py; `mol.dumps()` JSON string under `/mol` reads as `bytes` instead of `str` (h5py default), or `mo_coeff` 2D array is stored as C-order when upstream writes F-order, or the group structure uses `/scf/mo_coeff` but upstream uses `/scf/mo_coeff` with subtly different attribute encoding. Downstream `mol = pyscf.lib.chkfile.load_mol(path)` crashes or returns wrong Mole.

**Why it happens:** HDF5 has many subtle encoding choices (fixed-vs-VL strings, byte order, dtype matching, attribute scoping). Reproducing h5py's behaviour byte-for-byte is non-trivial.

**How to avoid:** ORACLE-08 round-trip oracle is the empirical seal — PySCF writes, pyscf-rs reads + numpy-allclose at 1e-12 on `mo_coeff`/`mo_energy`/`mo_occ`/`e_tot`; reverse direction pyscf-rs writes, Python `from_chk(path).kernel()` converges to upstream energy. If both directions pass, the schema is empirically compatible. Critical schema points: `/mol` is VL Unicode string (`hdf5_metno::types::VarLenUnicode`); `/scf/mo_coeff` is F-order f64 2D dataset; `/scf/mo_energy` / `/scf/mo_occ` are 1D f64; `/scf/e_tot` is scalar f64.

**Warning signs:** ORACLE-08 fails in either direction on the test corpus (H2O/cc-pVDZ chkfile round-trip).

### Pitfall 4 + Pitfall 12: Eigenvector Sign + Cross-Platform µHartree (SHOWSTOPPER + MAJOR)

**What goes wrong:** LAPACK on macOS Accelerate returns the eigenpair `(λ, v)`; LAPACK on Linux MKL returns `(λ, -v)`. SCF iterates use `v @ v.T` for density, which is sign-invariant — but the AO-projection step in `'minao'` init guess uses `v` directly, so the next-iteration density matrix differs by signs in specific orbitals. Over 30 cycles, cumulative drift exceeds 1 µHartree.

**Why it happens:** Eigenvector sign convention is vendor-specific. There's no LAPACK standard.

**How to avoid:** After every `eigh` call (both in SCF and in `'minao'`/`'huckel'` init_guess projection), apply `canonicalize_signs` (SCF-13, Pattern 9 above). The rule (largest-|coefficient|-lowest-index sign-flip) is vendor-invariant.

**Warning signs:** Linux/macOS µHartree CI assertion (Phase 3 success criterion 2) fails by >0.5 µHartree on H2O/cc-pVDZ.

### Pitfall 14: Panic Across FFI (SHOWSTOPPER — Phase 3 specifies conversion)

**What goes wrong:** Rust panics inside a `#[pymethods]` body; without `catch_unwind`, the panic unwinds across the FFI boundary; undefined behaviour (could be process abort, could be silent data corruption).

**Why it happens:** Rust's panic mechanism uses stack unwinding; C ABI assumes no unwinding.

**How to avoid:** PyO3 0.28 automatically wraps panics in `PanicException` for `#[pyfunction]` / `#[pymethods]` (no manual `catch_unwind` needed at PyO3 boundaries). Phase 1's lint (`extern "C"` callbacks wrapped in `catch_unwind`) covers the non-PyO3 FFI paths. Phase 3 specifies the typed-error conversion: every `Result<T, PyscfRsError>` returned from `pyscf-scf`/`pyscf-diis`/`pyscf-df` is mapped via `crate::errors::pyscf_to_py` to a `create_exception!`-generated `PyscfRsRuntimeError` with `.kind` and `.source_chain` attributes (preserves Rust error chain per BIND-09).

**Warning signs:** A pytest test that triggers an intentional convergence failure (`max_cycle=1, conv_tol=1e-99`) should produce a Python `PyscfRsError(kind='ConvergenceFailure')` — NOT a `PanicException` and NOT a segfault.

### Pitfall 8 (re-validation): F-Order Layout Preservation

**What goes wrong:** SCF iterates write `mo_coeff` to chkfile in C-order; upstream PySCF expects F-order; h5py reads back as C-order; downstream `from_chk` projection produces transposed orbitals.

**How to avoid:** Phase 2 D-04 already established the per-name F-vs-C convention via `pyscf/gto/moleintor.py`. For SCF, `mo_coeff` is F-order both in memory (`pyscf-core::MOCoefficients.data` doc comment) and in chkfile storage (verified at `pyscf/scf/chkfile.py:dump_scf` — writes upstream `mf.mo_coeff` directly, which is the F-order LAPACK output). Phase 3 plan adds `write_dataset_f_order` primitive in `pyscf-chkfile` (Pattern 6 above) and uses it for `mo_coeff`.

### Pitfall 15 (preventive): cubecl pin lockstep (re-validation)

Phase 3 doesn't bump cubecl. But the algebra-wall lint extension to 3 new crates must NOT inadvertently allow `cubecl-*` in `pyscf-chkfile`/`pyscf-diis`/`pyscf-df`. Lint config update is a Phase 3 plan task.

### Pitfall (NEW): abi3 + Python 3.13t ABI incompatibility

**What goes wrong:** Plan ships single PyO3 cdylib with `abi3-py310` feature; CI's `python3.13t` job tries to import it; ABI mismatch → ImportError or undefined-symbol.

**Why it happens:** [VERIFIED: PyO3 0.28 guide §free-threading.md] "The free-threaded build uses a completely new ABI and there is not yet an equivalent to the limited API."

**How to avoid:** Phase 3 ships TWO build configurations of `pyscf-py`:
1. **abi3-py310 wheel** (BIND-01/02 — Linux/macOS x86_64 PR-CI smoke): `features = ["abi3-py310", "extension-module"]`.
2. **non-abi3 python3.13t** smoke (BIND-05 — separate CI job): `features = ["extension-module"]` only, no `abi3-py310`. Built fresh for 3.13t.

Document in `crates/pyscf-py/Cargo.toml` `[features]` section with explicit conditional features.

## Code Examples

(Full code examples for each pattern shown above under "Architecture Patterns". Key signatures recap:)

### SCF kernel entry
```rust
// pyscf-scf
pub fn kernel<H: OverrideHooks>(
    mol: &Mole, hooks: &H, cfg: KernelConfig
) -> Result<ScfResult, PyscfRsError>;
```

### NumPy boundary converters
```rust
// pyscf-py
pub fn to_density<'py>(arr: PyReadonlyArray2<'py, f64>) -> PyResult<Density>;
pub fn to_mo_coeff<'py>(arr: PyReadonlyArray2<'py, f64>) -> PyResult<MOCoefficients>;
pub fn density_to_pyarray<'py>(py: Python<'py>, d: &Density) -> PyResult<Bound<'py, PyArray2<f64>>>;
```

### Chkfile trait
```rust
// pyscf-chkfile
pub trait Checkpointable: Sized {
    fn dump(&self, group: &hdf5::Group) -> Result<(), ChkfileError>;
    fn load(group: &hdf5::Group) -> Result<Self, ChkfileError>;
}
```

### DIIS trait
```rust
// pyscf-diis
pub trait DiisStorable {
    fn as_flat(&self) -> &[f64];
    fn from_flat(&mut self, slice: &[f64]);
    fn dot(&self, other: &Self) -> f64;
    fn len(&self) -> usize;
}
pub struct Diis<S: DiisStorable + Clone> { /* ... */ }
impl<S: DiisStorable + Clone> Diis<S> {
    pub fn new(space: usize) -> Self;
    pub fn extrapolate(&mut self, current: S, error: Vec<f64>) -> Result<S, DiisError>;
}
```

### DF integrals
```rust
// pyscf-df
pub struct DfIntegrals { pub b_uvq: Vec<f64>, pub naux: usize, pub nao: usize }
pub fn cholesky_eri(mol: &Mole, auxbasis: &str) -> Result<DfIntegrals, PyscfRsError>;
```

### sign canonicalization
```rust
// pyscf-core
pub fn canonicalize_signs(c: &mut [f64], nao: usize, nmo: usize);
```

### oracle macro
```rust
// pyscf-oracle
/// Expand: oracle_check!("scf_rhf_energy", H2O_CC_PVDZ, 1e-6).
/// Spawns Python via Python::attach, runs upstream PySCF, compares pyscf-rs's
/// result element-wise to `tolerance`. Used by every Phase 3 success criterion.
#[macro_export]
macro_rules! oracle_check {
    ($method:literal, $fixture:expr, $tolerance:expr) => { /* ... */ };
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `Python::with_gil` | `Python::attach` | PyO3 0.26 (renamed) | Phase 3 uses `attach`; old name is alias but lints warn |
| `py.allow_threads(\|\| ...)` | `py.detach(\|\| ...)` | PyO3 0.26 (renamed) | Phase 3 uses `detach`; old name is alias |
| `lazy_static!` for static PyObjects | `pyo3::sync::GILOnceCell` or `PyOnceLock` | PyO3 0.21 (GILOnceCell), 0.28 (PyOnceLock added) | Phase 3 prefers `PyOnceLock` (free-thread-safe) |
| `&PyAny` / `Py<T>` | `Bound<'py, T>` | PyO3 0.21 (Bound API) | Phase 3 uses Bound everywhere; older Py<T> stays for stored references |
| `aldanor/hdf5-rust` crates.io publishes | `hdf5-metno` fork | aldanor/hdf5-rust unmaintained as of 2023 | Phase 3 uses `hdf5-metno = "=0.10.0"` (April 2024-era release) |
| Manual `extern "C" fn` panic wrapping | PyO3 auto-wraps panics into `PanicException` | PyO3 0.16+ | Phase 3 only needs manual `catch_unwind` on cintx-side callbacks (Phase 1 lint already covers) |
| `#[pyclass(extends=PyException)]` | `create_exception!` + Python overlay shim | PyO3 0.28 enforces abi3-py310 limitation | Phase 3 follows xcfun-py pattern verbatim |

**Deprecated/outdated:**
- `Python::with_gil` — works in PyO3 0.28 but deprecated in favor of `Python::attach`.
- `py.allow_threads` — works in PyO3 0.28 but deprecated in favor of `py.detach`.
- `lazy_static!` in PyO3 paths — works but deadlock-prone under free-threaded Python; use `PyOnceLock`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `pyscf-algebra::solve_linear` doesn't exist yet; Phase 3 must add it (host-faer wrapper) OR use `cholesky + triangular_solve` chain | Pattern 7 (DIIS) | Plan estimate off by ~1 task if Phase 1 actually shipped it; mitigation: grep confirmed `pub fn solve_linear` absent from `pyscf-algebra/src/lib.rs` re-exports (only gemm/gemv/axpy/dot/cholesky/eigh/qr/svd/reduce_sum/oracle_*) — confirmed [VERIFIED: grep of crates/pyscf-algebra/src/lib.rs:35-43] |
| A2 | h5py default reads VarLenUnicode datasets as Python `str` (not `bytes`) when the dataset's HDF5 type is `H5T_C_S1` with `H5T_STR_NULLTERM` and UTF-8 encoding | Pattern 6 (chkfile) | ORACLE-08 round-trip in either direction would fail; mitigation: the test IS the empirical seal — if it fails, planner adjusts hdf5-metno dtype |
| A3 | `pyscf/df/addons.py:DEFAULT_AUXBASIS` covers `cc-pvdz` and `def2-svp` test corpus basis names; fallback `'weigend'` aux for anything else matches upstream | Pattern 8 (DF) | DF-HF energy diverges by integral fitting error; mitigation: Phase 3 plan tests each test-corpus basis explicitly |
| A4 | PyO3 0.28's `create_exception!` macro produces a class that Python can subclass at the Python level (i.e., the overlay shim's `class PyscfRsError(_PyscfRsBase):` works) | Pattern 5 (panic→exception) | abi3-py310 wheel fails to import; mitigation: xcfun-py ships this pattern shipping on PyPI [VERIFIED — xcfun-py errors.rs:31 + __init__.py 1-50] so the pattern works in production |
| A5 | `mol.dumps()` round-trip (Phase 2 D-08) handles all 30+ `MoleBase` attributes (including `nucmod`, `nucprop`, `ecp` HashMaps) such that `Mole::loads(mol.dumps())` is semantically equivalent for chkfile reload | Pattern 6 | ORACLE-08 reverse direction (pyscf-rs writes, Python from_chk) may need fields pyscf-rs doesn't serialize; mitigation: Phase 2's verification plan should already cover JSON round-trip parity |
| A6 | Python user subclasses of `scf.RHF` go through normal Python MRO when `slf.call_method1("get_veff", args)` is invoked — i.e., a Python override of `get_veff` is preferred over the `#[pymethods] fn get_veff` default | Pattern 1 (subclass bridge) | If wrong, Pitfall 7 (override bypass) occurs; mitigation: this is `slf.call_method1`'s documented contract per PyO3 0.28 guide §class.md, equivalent to Python's `getattr(subclass, "get_veff")()` |
| A7 | The `python3.13t` free-threaded CI build can be exercised on a standard `actions/setup-python@v5` runner with `python-version: '3.13'` and a free-threaded interpreter variant flag (or via `setup-uv` + `uv python install 3.13t`) | CI section below | python3.13t job fails to provision; mitigation: this is a community-standard CI recipe and works in current PyO3 CI [reference: pyo3.rs/v0.28.3/free-threading.html mentions "free-threaded variants"] |
| A8 | `hdf5-metno 0.10.0` with the `static` feature builds libhdf5 from source on Linux x86_64 + macOS aarch64 CI runners within reasonable time (~5min) without requiring system CMake beyond what's pre-installed on `ubuntu-latest` / `macos-latest` | CI section below | First build is slow but cacheable; mitigation: `actions/cache@v4` on `~/.cargo/registry`, `~/.cargo/git`, `target/` |
| A9 | `solve_linear` for an 8×8 SPD-like matrix can be added to `pyscf-algebra` as a thin faer LU wrapper without disturbing the host-fallback discipline of ALG-05 (eigh/cholesky/qr/svd already go host-faer) | Pattern 7 + Don't Hand-Roll | Phase 1's algebra surface stays minimal; mitigation: pyscf-algebra already has the host-faer plumbing (host_fallback.rs declares eigh + cholesky) — adding solve_linear is mechanical |
| A10 | `mf.as_scanner()` returning a `Py<PyAny>` callable closure that captures `self` works under PyO3 0.28 + abi3-py310 (closure types as Python callables) | Claude's Discretion §as_scanner | Geomopt (Phase 7) needs scanner; mitigation: alternative is a tiny `#[pyclass] PyScfScanner { mf: Py<PyRHF> }` with `__call__` method — more verbose but works regardless |

**If this table is empty:** N/A — 10 explicit assumptions surface for planner confirmation. Most are mitigated by the in-plan empirical seal (ORACLE-08) or by direct fallback paths.

## Open Questions

1. **Does the `pyscf-algebra::solve_linear` surface need to expand the algebra-wall, or can DIIS use cholesky+triangular_solve?**
   - What we know: ALG-05 already routes eigh/cholesky/qr/svd via host-faer. faer 0.24 has `lin_solver::Lu` for non-SPD solves.
   - What's unclear: Whether Phase 1 reserved a slot for `solve_linear` (it didn't — confirmed by grep) and whether Phase 3 should extend the public surface or use Cholesky on `B B^T` (which is always SPD because `B[i,j] = oracle_dot(err_i, err_j)`).
   - Recommendation: Phase 3 plan ships `pyscf-algebra::solve_linear` as a thin LU wrapper (host-faer) because (a) the B-matrix Lagrange-multiplier system has a zero on the diagonal so naïve Cholesky fails; (b) faer 0.24 LU is bit-exact on host. Alternative (Gauss-Jordan in pyscf-diis directly) keeps algebra-surface minimal but duplicates work.

2. **What's the exact `python3.13t` CI runner setup recipe?**
   - What we know: PyO3 0.28+ supports free-threaded Python; abi3 wheels are NOT compatible with 3.13t.
   - What's unclear: Whether GitHub Actions' `actions/setup-python@v5` provides 3.13t directly or whether `uv python install 3.13t` is needed.
   - Recommendation: Phase 3 plan investigates both during the BIND-05 CI plan; falls back to `setup-uv` if `setup-python` doesn't support free-threaded variant flag yet.

3. **Should `mf.chkfile` default to a tempfile (upstream behaviour at `pyscf/scf/hf.py:1742-1743`) or None?**
   - What we know: Upstream creates a tempfile by default and auto-removes it; users override by `mf.chkfile = 'mycalc.chk'`.
   - What's unclear: Whether the tempfile-default matters for Phase 3 success criteria (no test explicitly asserts the default).
   - Recommendation: Mirror upstream — create tempfile on `__init__`, auto-cleanup on drop. Or default to None (simpler, may break a top-20 idiom). Planner decides.

4. **For `'chkfile'` init_guess mode: does the gap-closure plan land before SCF-05 verification?**
   - What we know: `'chkfile'` mode depends on `pyscf-chkfile` crate; Phase 3 ships both.
   - What's unclear: Plan ordering — Phase 3 plans (waves) may want to land `pyscf-chkfile` Wave 0 (or 1) before any plan touches `'chkfile'` init_guess.
   - Recommendation: Wave 1 = chkfile crate + ORACLE-08 round-trip oracle; Wave 2+ depends on it (init_guess 'chkfile' mode plan).

5. **What `auxbasis` defaults must Phase 3 ship for SCF-07?**
   - What we know: Upstream `pyscf/df/addons.py:DEFAULT_AUXBASIS` (per A3 above) covers a long list of basis pairs.
   - What's unclear: Which subset is needed for the H2O/cc-pVDZ + benzene/6-31G* test corpus.
   - Recommendation: Phase 3 plan ports the full DEFAULT_AUXBASIS table (it's small — under 200 lines of dict in upstream); avoids surprises.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `python3` | All PyO3 work, oracle harness | ✓ (system) | 3.10+ assumed | — |
| `python3.13t` | BIND-05 CI smoke (free-threaded) | TBD (CI step needed) | 3.13.x free-threaded | Skip BIND-05 fast-path; document as known limitation |
| `maturin` | BIND-01 wheel build, `maturin develop` | ✗ install via `pip install maturin>=1.12,<2.0` | — | Document install step in CONTRIBUTING.md addition |
| `cmake` | hdf5-metno-sys `static` feature builds libhdf5 from source | ✓ (`ubuntu-latest` / `macos-latest` ship it) | 3.20+ assumed | Fall back to `hdf5-metno` without static (requires system libhdf5) — but this breaks DIST-05; planner flags as blocker if missing |
| `pyscf` (upstream, Python) | ORACLE-01 + ORACLE-02 + ORACLE-08 oracle harness | ✓ (already in repo at `pyscf/`) | bundled | — |
| `h5py` | ORACLE-08 round-trip oracle (Python side validates pyscf-rs writes) | TBD (pip install in CI venv) | `>=3.10` | Required — no fallback |
| `pytest` | BIND-04 stride-fuzz, BIND-07 subclass-override CI tests | ✗ pip install in CI venv | `>=7.0` | — |
| `numpy` (Python) | All PyO3 tests | ✓ via pip | `>=1.26` | — |
| `pyo3 = "=0.28.3"` | All PyO3 entry points | ✓ on crates.io | 0.28.3 | — |
| `hdf5-metno = "=0.10.0"` | pyscf-chkfile | ✓ on crates.io | 0.10.0 | — |
| `numpy = "=0.28.0"` (rust-numpy) | pyscf-py | ✓ on crates.io | 0.28.0 | — |

**Missing dependencies with no fallback:**
- `python3.13t` CI provisioning — must be solved in Phase 3 plan (Open Question 2).
- `h5py` in CI venv — must be installed before ORACLE-08 tests.

**Missing dependencies with fallback:**
- `cmake` for hdf5-metno-sys static build — fallback (system libhdf5) breaks DIST-05; planner flags as upgrade-path issue if CI runner missing CMake.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `pytest` (Python side) + `cargo test` (Rust side) — both required |
| Config file | New `pyproject.toml` at repo root (maturin config + pytest deps); existing root `pytest.ini` is upstream PySCF's test config (don't touch per Phase 1 D-03) |
| Quick run command | `cargo test -p pyscf-scf -p pyscf-diis -p pyscf-df -p pyscf-chkfile && maturin develop && pytest python/pyscf/tests/test_scf_smoke.py -x` |
| Full suite command | `cargo build --profile release-oracle --workspace && maturin develop --release && pytest python/pyscf/tests/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SCF-01 | RHF on H2O/cc-pVDZ matches upstream ≤ 1 µHartree | oracle-integration | `pytest python/pyscf/tests/test_scf_rhf_h2o.py -x` | ❌ Wave 0 |
| SCF-01 | RHF on benzene/6-31G* matches upstream | oracle-integration | `pytest python/pyscf/tests/test_scf_rhf_benzene.py -x` | ❌ Wave 0 |
| SCF-02 | UHF on radical matches upstream | oracle-integration | `pytest python/pyscf/tests/test_scf_uhf.py -x` | ❌ Wave 0 |
| SCF-03 | GHF on H2 runs to completion | unit | `pytest python/pyscf/tests/test_scf_ghf.py -x` | ❌ Wave 0 |
| SCF-04 | C-DIIS converges in upstream iteration count ±1 | oracle-integration | `pytest python/pyscf/tests/test_scf_diis.py -x` | ❌ Wave 0 |
| SCF-05 | Each of 5 init_guess modes matches upstream first-iter density | oracle-integration | `pytest python/pyscf/tests/test_scf_init_guess.py -x` | ❌ Wave 0 |
| SCF-06 | `level_shift`, `damp` semantics match upstream | unit + oracle | `pytest python/pyscf/tests/test_scf_controls.py` | ❌ Wave 0 |
| SCF-07 | `mf.density_fit().kernel()` matches upstream | oracle-integration | `pytest python/pyscf/tests/test_scf_df.py -x` | ❌ Wave 0 |
| SCF-08 | Subclass override `get_veff` invoked once per cycle | unit (Python) | `pytest python/pyscf/tests/test_scf_override_dispatch.py -x` | ❌ Wave 0 |
| SCF-09 | `mf.analyze()`/`mulliken_pop`/`mulliken_meta`/`dip_moment` match upstream | oracle | `pytest python/pyscf/tests/test_scf_analyze.py -x` | ❌ Wave 0 |
| SCF-10 | h5py-schema chkfile round-trip works | oracle | `pytest python/pyscf/tests/test_scf_chkfile.py -x` | ❌ Wave 0 (covers ORACLE-08) |
| SCF-11 | `mf.to_uhf()` / `to_rhf()` / `to_ghf()` work; `to_uks()` raises NotYetImplemented | unit | `pytest python/pyscf/tests/test_scf_cross_dispatch.py` | ❌ Wave 0 |
| SCF-12 | `mf.as_scanner()(mol2)` returns energy | unit | `pytest python/pyscf/tests/test_scf_scanner.py` | ❌ Wave 0 |
| SCF-13 | `canonicalize_signs` is largest-|c|-lowest-index-flip | unit (Rust) | `cargo test -p pyscf-core canonicalize_signs` | ❌ Wave 0 |
| SCF-13 | Linux + macOS µHartree assertion on H2O/cc-pVDZ | matrix-CI | GitHub Actions matrix job `xplat-uhartree` | ❌ Wave 0 (covers Pitfall 12) |
| SCF-14 | 30-attribute floor introspectable from Python | unit | `pytest python/pyscf/tests/test_scf_attributes.py` | ❌ Wave 0 |
| BIND-01 | `maturin develop` produces importable `pyscf._native` | CI smoke | `maturin develop && python -c 'from pyscf._native import scf'` | ❌ Wave 0 |
| BIND-02 | `from pyscf import scf` resolves to overlay | unit (Python) | `pytest python/pyscf/tests/test_overlay_resolution.py` | ❌ Wave 0 |
| BIND-04 | Stride-fuzz: a/a.T/a[::2]/a[:,1:5] all return same get_veff | unit (Python) | `pytest python/pyscf/tests/test_scf_stride_fuzz.py` | ❌ Wave 0 |
| BIND-05 | python3.13t SCF smoke runs without deadlock | matrix-CI | GitHub Actions job `python313t-smoke` | ❌ Wave 0 |
| BIND-06 | No `lazy_static!` in `pyscf-py`; `PyOnceLock` used | lint | `xtask lint forbid-lazy-static pyscf-py` | ❌ Wave 0 |
| BIND-07 | Subclass `get_veff` override invoked >= cycles (re-shape of SCF-08) | unit (Python) | (same as SCF-08) | (same) |
| BIND-09 | Convergence failure raises PyscfRsError(kind='ConvergenceFailure') | unit (Python) | `pytest python/pyscf/tests/test_panic_to_exception.py` | ❌ Wave 0 |
| ORACLE-02 | `oracle_check!` macro invokable from `pyscf-oracle` dev-deps | unit (Rust) | `cargo test -p pyscf-oracle oracle_check` | ❌ Wave 0 |
| ORACLE-08 | chkfile round-trip both directions on H2O/cc-pVDZ | oracle | (same as SCF-10) | (same) |

### Sampling Rate

- **Per task commit:** `cargo test -p pyscf-scf -p pyscf-diis -p pyscf-df -p pyscf-chkfile -p pyscf-core` (Rust unit tests; <30s).
- **Per wave merge:** `cargo test --profile release-oracle --workspace && maturin develop && pytest python/pyscf/tests/test_scf_smoke.py` (full Rust + Python smoke; <2min).
- **Phase gate:** `cargo build --profile release-oracle --workspace && maturin develop --release && pytest python/pyscf/tests/ -x` (full oracle suite; <10min) PLUS the BIND-05 python3.13t job and Pitfall 12 cross-platform µHartree assertion.

### Wave 0 Gaps

- [ ] `python/pyscf/tests/__init__.py` — empty file
- [ ] `python/pyscf/tests/conftest.py` — shared fixtures (H2O, benzene, water-trimer Mole)
- [ ] `python/pyscf/tests/test_scf_rhf_h2o.py` — SCF-01
- [ ] `python/pyscf/tests/test_scf_rhf_benzene.py` — SCF-01
- [ ] `python/pyscf/tests/test_scf_uhf.py` — SCF-02
- [ ] `python/pyscf/tests/test_scf_ghf.py` — SCF-03
- [ ] `python/pyscf/tests/test_scf_diis.py` — SCF-04
- [ ] `python/pyscf/tests/test_scf_init_guess.py` — SCF-05 (5 modes)
- [ ] `python/pyscf/tests/test_scf_controls.py` — SCF-06
- [ ] `python/pyscf/tests/test_scf_df.py` — SCF-07
- [ ] `python/pyscf/tests/test_scf_override_dispatch.py` — SCF-08 / BIND-07
- [ ] `python/pyscf/tests/test_scf_analyze.py` — SCF-09
- [ ] `python/pyscf/tests/test_scf_chkfile.py` — SCF-10 / ORACLE-08
- [ ] `python/pyscf/tests/test_scf_cross_dispatch.py` — SCF-11
- [ ] `python/pyscf/tests/test_scf_scanner.py` — SCF-12
- [ ] `python/pyscf/tests/test_scf_attributes.py` — SCF-14
- [ ] `python/pyscf/tests/test_overlay_resolution.py` — BIND-02
- [ ] `python/pyscf/tests/test_scf_stride_fuzz.py` — BIND-04
- [ ] `python/pyscf/tests/test_panic_to_exception.py` — BIND-09
- [ ] `crates/pyscf-oracle/src/lib.rs` — `oracle_check!` macro (ORACLE-02)
- [ ] `crates/pyscf-oracle/tests/chkfile_roundtrip.rs` — ORACLE-08 macro invocation tests
- [ ] `crates/pyscf-core/src/lib_fn.rs` (or extend `lib.rs`) — `pub fn canonicalize_signs` (SCF-13)
- [ ] `.github/workflows/ci.yml` — extends with: (a) `maturin-smoke` job, (b) `stride-fuzz` job (BIND-04), (c) `xplat-uhartree` matrix job (Pitfall 12 mitigation), (d) `python313t-smoke` job (BIND-05)
- [ ] Framework install: `pip install pytest>=7.0 h5py>=3.10 numpy>=1.26 maturin>=1.12,<2.0` — required for CI Python venv

## Security Domain

**Required when `security_enforcement` is enabled (absent = enabled).** `.planning/config.json` doesn't declare `security_enforcement`, so treat as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A — pyscf-rs is a library, no auth surface |
| V3 Session Management | no | N/A — no sessions |
| V4 Access Control | no | N/A — library, no access boundaries |
| V5 Input Validation | yes | `pyscf-core::error::CoreError`/`BasisLoadError`/`EcpLoadError` (Phase 1/2 shipped); Phase 3 adds `ChkfileError`, `DiisError`, `DfError`; all `?`-propagate via `PyscfRsError`. NumPy boundary converters (D-04) validate shape + dtype + contiguity. Mole.dumps() JSON read via `serde_json` (parsing-error-safe). chkfile open() returns `Err` on missing-file rather than panic. |
| V6 Cryptography | no | N/A — no cryptography (HDF5's static linking includes no crypto primitives we expose) |
| V7 Error Handling | yes | `thiserror`-derived error types throughout; panic→exception conversion (BIND-09) hides internal panic messages from Python (sanitized via `create_exception!`); no `unwrap()` in numerical modules (FOUND-07 lint deny) |
| V8 Data Protection | yes | chkfile contains user's MO coefficients + e_tot — this is user-controlled. We don't store secrets. The xcfun-py pattern (errors.rs:78-80) of dropping payload details on certain error variants is documented as a defense-in-depth choice but not required here (no host-info leak in SCF errors). |
| V9 Communications | no | N/A — no network I/O |

### Known Threat Patterns for {pyscf-rs / PyO3 / HDF5}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed Python override raises arbitrary exception inside Rust kernel | Tampering | `PyOverrideBridge::*` catches `PyErr` and converts to `PyscfRsError::PythonOverrideFailed { cause }` — kernel reports clean failure, no Rust panic |
| NumPy view with broken strides (deliberate or accidental) | Tampering | BIND-04 — `is_c_contiguous() + to_owned()` policy; ndarray library checks bounds internally |
| HDF5 file from untrusted source contains malformed `mol.dumps()` JSON | Tampering | `serde_json` parser is panic-free (returns `Err`); pyscf-rs propagates as `ChkfileError::MalformedMol` |
| HDF5 file contains datasets exceeding memory budget (DoS) | DoS | Phase 3 reads only `e_tot` (scalar), `mo_energy`/`mo_occ` (1D, naturally bounded by nao), `mo_coeff` (2D, nao²). nao is bounded by the loaded Mole's basis. No arbitrary-size dataset reads — safe by construction. Phase 6 CCSD-08 introduces `PYSCF_MAX_MEMORY` enforcement; until then, no enforcement at chkfile-read level. |
| Rust panic across PyO3 FFI boundary | Repudiation/Tampering | PyO3 0.28 auto-wraps `#[pymethods]`/`#[pyfunction]` panics into `PanicException`; Phase 1 lint enforces `catch_unwind` on `extern "C"` callbacks (cintx side, not pyscf-rs side); Phase 3 typed-error conversion via `pyscf_to_py` preserves error chain without exposing Rust panic messages |
| Python override hook hangs (infinite loop in user code) | DoS | Out of scope — user-controlled code; PyO3 doesn't enforce timeouts. Document as a known limitation. |
| Free-threaded Python data race on shared `PyOnceLock` | Tampering/DoS | `PyOnceLock` is free-thread-safe by design [VERIFIED: PyO3 0.28 §sync.md]; `GILOnceCell` is NOT — BIND-06 lint enforces `PyOnceLock` for new caches |
| ABI mismatch between abi3-py310 wheel and Python 3.13t at install time | DoS | Two-build configuration (BIND-01+BIND-05) — installing the wrong wheel raises clean `ImportError`, not a segfault |

## Sources

### Primary (HIGH confidence)

- **Context7 `/pyo3/pyo3`** — fetched 2026-05-11. Topics covered: `trait-bounds.md` (subclass-override bridge pattern), `class.md` (`#[pyclass(subclass)]` + `PyClassInitializer` + `as_super`/`into_super`), `parallelism.md` (`Python::detach`), `migration.md` (`Python::attach` rename, `PyOnceLock` replacement for `GILOnceCell`), `building-and-distribution.md` (abi3 limitations table: text_signature 3.10+, dict/weakref 3.9+, buffer API 3.11+, PyException subclass 3.12+), `exception.md` (`create_exception!` + `#[pyclass(extends=PyException)]` abi3 restriction), `free-threading.md` (3.13t ABI incompatibility with abi3).
- **Upstream PySCF source (this repo)** — `pyscf/scf/hf.py:48-244` (kernel), `:316` (get_hcore), `:348` (init_guess_by_minao), `:485` (init_guess_by_1e), `:495` (init_guess_by_atom), `:537` (init_guess_by_huckel), `:673` (init_guess_by_chkfile), `:764` (get_init_guess dispatcher), `:957` (get_jk), `:1034` (get_veff), `:1086` (get_fock), `:1136` (get_occ), `:1199` (analyze), `:1262` (mulliken_pop), `:1301` (mulliken_meta), `:1349-1357` (eig + INLINE sign canonicalization rule — Phase 3 extracts to canonicalize_signs), `:1380` (dip_moment), `:1538-1602` (as_scanner), `:1605-2400` (SCF class + 30-attribute floor + `_keys` set), `:2272-2300` (to_rhf/to_uhf/to_ghf), `:2302-2318` (to_rks/to_uks — stub for Phase 3); `pyscf/scf/uhf.py:754` (UHF class); `pyscf/scf/ghf.py:378` (GHF class); `pyscf/scf/diis.py:40-66` (CDIIS), `:68-87` (get_err_vec_orig — error vec = SDF - FDS); `pyscf/scf/chkfile.py:25-42` (dump_scf / load_scf — schema source-of-truth); `pyscf/lib/chkfile.py:28-191` (dump / load / save_mol / load_mol — primitive layer); `pyscf/df/df.py:41-200` (DF class); `pyscf/df/df_jk.py:31-148` (density_fit / _DFHF / get_jk).
- **xcfun-py sibling crate** (`~/Documents/workspace/xcfun_rs/crates/xcfun-py/`) — `Cargo.toml:1-32` (abi3-py310 + extension-module features pattern); `pyproject.toml:47-51` (maturin `python-source` + `module-name` config); `src/lib.rs:1-52` (`#[pymodule] fn _native` skeleton); `src/numpy_io.rs:9-76` (PyReadonlyArray2 + is_c_contiguous + py.detach pattern); `src/functional.rs:148-279` (`#[pyclass]` + `#[pymethods]` + `Bound<'py, Self>` pattern); `src/errors.rs:31-83` (`create_exception!` + Python overlay shim for abi3 PyException workaround); `python/xcfun_rs/__init__.py:1-50` (Python overlay grafting `.code` / `.kind` attributes).
- **github.com/PyO3/pyo3/releases** — fetched via WebFetch 2026-05-11. PyO3 0.28.3 released April 2, 2025 (latest stable). 0.28.0 February 1, 2025 (free-threaded gil_used=false default).
- **github.com/metno/hdf5-rust README** — fetched via WebFetch 2026-05-11. hdf5-metno 0.10.0 is current; fork of aldanor/hdf5-rust; backward-compatible via `hdf5 = { package = "hdf5-metno" }`; features include `static`, `blosc`, `lzf`, `zfp`, `mpi`.
- **docs.rs/hdf5-metno/latest** — fetched via WebFetch 2026-05-11. File/Group/Dataset/Attribute API; ndarray integration; VL strings for h5py compat; `.new_dataset::<T>().shape().create(name).write(&array)?` pattern.

### Secondary (MEDIUM confidence)

- **pyo3.rs/v0.28.0/free-threading.html** — WebSearch 2026-05-11. 3.13t ABI incompatibility with abi3 (confirmed via PyO3 free-threading docs reference).
- **`PyO3/pyo3` discussion #4738 (3.13 freethreaded deadlock)** — referenced via WebSearch. Pitfall 6 mitigation pattern (`PyOnceLock`).
- **xcfun-py errors.rs comments** — internally consistent with Phase 5 D-09 / D-10 / abi3 §5 CRITICAL FINDING in xcfun-py's own RESEARCH; we trust the production pattern.

### Tertiary (LOW confidence — flagged for validation in plan)

- **`pyo3-py310 + python3.13t separate-build CI recipe** — no canonical reference; mirrors what's used in other PyO3 ecosystems but Phase 3 plan must validate empirically on CI.
- **`hdf5-metno` writing VL Unicode strings reads back as str (not bytes) on h5py 3.x** — assumed compatible per README + xcfun-py pattern, but only ORACLE-08 round-trip test confirms.

## Metadata

**Confidence breakdown:**
- PyO3 0.28 surface: HIGH — verified via Context7 against the most current 0.28 docs.
- numpy 0.28 surface: HIGH — verified via Context7 + xcfun-py production code.
- hdf5-metno 0.10.0 schema compatibility: MEDIUM — empirical seal pending ORACLE-08.
- maturin abi3-py310 build: HIGH — xcfun-py shipped pattern.
- Upstream PySCF algorithm semantics: HIGH — line-numbered citations of canonical functions.
- DIIS bit-exact reproducibility under `release-oracle`: MEDIUM — depends on Phase 1's oracle_dot/oracle_sum behaving as specified (Phase 1 success criterion 3 validates).
- Cross-platform µHartree assertion (Pitfall 12): MEDIUM — depends on Phase 1's FMA-free + canonicalize_signs combined; Phase 3 is the first phase where this is end-to-end tested on a real SCF energy.
- python3.13t CI recipe: LOW — community recipes exist but no single source-of-truth; Phase 3 plan validates.

**Research date:** 2026-05-11
**Valid until:** 2026-06-10 (30 days for stable PyO3 0.28.x ecosystem; bumps to 0.28.4+ unlikely to break this research)

---

*Phase: 03-scf-pyo3-bindings*
*Researched: 2026-05-11*
