# Phase 3: SCF + PyO3 bindings - Context

**Gathered:** 2026-05-11
**Status:** Ready for planning

<domain>
## Phase Boundary

A Python user runs `from pyscf import scf; scf.RHF(mol).kernel()` from an unmodified existing PySCF script and gets the same total energy as upstream PySCF to ≤1 µHartree, while every PyO3 contract that downstream methods inherit (subclass-override dispatch, NumPy contiguity, GIL release seam, panic-to-exception, abi3 wheel skeleton) is locked and CI-enforced on this single small surface (RHF on H2O/cc-pVDZ).

**In scope (21 REQ-IDs):**
- SCF-01..14: RHF + UHF + GHF kernel, C-DIIS (`diis_space=8`, `diis_start_cycle=1`), 5 init_guess modes (`'minao'`, `'atom'`, `'1e'`, `'huckel'`, `'chkfile'`), user-supplied `dm0`, `level_shift`/`damp`/`max_cycle`/`conv_tol`/`conv_tol_grad`, `mf.density_fit(auxbasis=...)` (DF-HF), all 10 overrideable hooks, `mf.analyze()`/`mulliken_pop`/`mulliken_meta`/`dip_moment`, `mf.chkfile` (h5py-schema-compatible), cross-module dispatch helpers (`to_uhf`/`to_rhf`/`to_uks`/`to_rks`/`to_ghf`), `mf.as_scanner()`, `canonicalize_signs` (largest-|coefficient|-lowest-index sign-flip), 30-attribute floor.
- BIND-01,02,04,05,06,07,09: single `pyscf-py` abi3-py310 cdylib (skeleton; full wheel matrix is Phase 8), `_native.scf` PyO3 submodule + `python/pyscf/__init__.py` overlay, NumPy `is_standard_layout` → `to_owned()` policy, `Python::detach` (≡ old `py.allow_threads`) GIL release seam, `pyo3::sync::GILOnceCell` replaces every `lazy_static!` in PyO3 paths, subclass-override dispatch via `slf.call_method1`, panic→exception conversion preserving error chain.
- ORACLE-02,08: `oracle_check!(method, tolerance, fixture)` macro in `pyscf-oracle` (dev-deps only), chkfile round-trip oracle (PySCF writes → pyscf-rs reads asserts identical; pyscf-rs writes → PySCF reads runs downstream calc asserts agreement).

**Cross-platform invariant:** Linux x86_64 and macOS aarch64 produce total energies that agree to within 1 µHartree under `release-oracle` (Pitfall 12 mitigation; depends on Phase 1's FMA-free profile + ordered reductions and on `canonicalize_signs` producing vendor-stable eigenvectors per Pitfall 4).

**Out of scope:**
- DFT (Phase 4), MP2 (Phase 5), CCSD (Phase 6), gradients + geomopt (Phase 7).
- GPU per-backend regression suite, drop-in audit, full wheel matrix, BIND-03, BIND-08 (Phase 8).
- ROHF, SOSCF (`scf.newton`), ADIIS/EDIIS, symmetry-adapted SCF (REQUIREMENTS.md SCF-EXT-01..05, v1.x).
- HDF5 spill for DF B integrals (Phase 6 with CCSD-08 + CCSD-11 tensor-arena).
- Real abi3-py310 wheel matrix build, manylinux_2_28 baseline, `abi3audit` CI, per-backend extras (Phase 8 DIST-*).

</domain>

<decisions>
## Implementation Decisions

### PyO3 subclass-override dispatch (Pitfall 7, BIND-07, SCF-08)

- **D-01: Trait-callback bridge.** `pyscf-scf` declares a `pub trait OverrideHooks` with one method per overrideable hook (`get_jk`, `get_veff`, `get_hcore`, `eig`, `get_occ`, `make_rdm1`, `energy_elec`, `energy_tot`, `get_init_guess`, `get_fock`). `pyscf-scf` has **zero pyo3 dependency**. `pyscf-py` provides a `PyOverrideBridge` impl that routes every call through `slf.call_method1(py, "<hook>", args)`. Python's MRO does subclass-override dispatch natively — if a Python subclass overrides, its method runs; if not, the `#[pymethods]` default in `pyscf-py` runs (which forwards to a public Rust-default function in `pyscf-scf`). Pitfall 7 immune by construction. Phase 4 DFT re-validates on the larger DFT overrideable surface (DFT-08) using the same trait shape.

- **D-02: Pub trait + pub generic kernel — Rust-only SCF API.** `OverrideHooks` is `pub`. `pyscf_scf::RHF::kernel<H: OverrideHooks>(mol, hooks) -> Result<ScfResult>` is generic over the bridge impl. A Rust-only caller implements `OverrideHooks` (no pyo3 needed — provide a `NoOverrides` zero-cost default impl) and drives SCF without Python. Aligns with DIST-01 (pyscf-rs on crates.io with workspace façade re-exporting in-scope methods).

- **D-03: Per-hook `Python::detach` GIL release seam.** Each heavy hook body wraps compute in `Python::detach` (≡ old `py.allow_threads` in PyO3 0.28): Fock build's two-electron contraction via cintx `int2e_sph`, the `eigh` call, and the DIIS extrapolation matrix solve. Override call sites stay GIL-attached by definition. The `python3.13t` free-threaded CI build runs the SCF test corpus to probe deadlock surface. Phase 4 inherits at XC-evaluation kernel; Phase 6 inherits at CCSD doubles update — minimal scope per detach so deadlock risk is contained per-hook.

- **D-04: Type-specific NumPy boundary converters.** `pyscf-py` defines `to_density(arr) -> Density`, `to_mo_coeff(arr) -> MOCoefficients`, `to_fock_matrix(arr) -> Fock`, etc. Each runs `is_standard_layout()` and calls `to_owned()` before constructing the pyscf-core Rust type (BIND-04, Pitfall 5 mitigation). Output helpers: `density_to_pyarray(d, py)`, `mo_coeff_to_pyarray(mc, py)` — always C-contiguous unless the upstream PySCF per-name convention is F-order (carried from Phase 2 D-04: planner consults `pyscf/gto/moleintor.py`). Greppable in CI. BIND-04 stride-fuzz test calls each entry with `a`, `a.T`, `a[::2]`, `a[:, 1:5]` and asserts identical answers.

### HDF5 chkfile (SCF-10, ORACLE-08, Pitfall 11)

- **D-05: `hdf5-metno` crate.** Maintained metno fork of `aldanor/hdf5-rust`. Static linking via `hdf5-metno-sys` `static` feature satisfies DIST-05 (HDF5 ships statically linked, no system libhdf5 at install time). Has ndarray integration. STATE.md "Blockers/Concerns" already names it as the candidate needing empirical seal in Phase 3 — that seal IS the ORACLE-08 round-trip oracle.

- **D-06: New `pyscf-chkfile` workspace crate.** Sole owner of the `hdf5-metno` dependency (algebra-wall-style discipline). Exposes HDF5 primitives (`open_for_write`, `read_group`, `write_dataset`, `read_dataset`, `write_string_attr`) PLUS a `Checkpointable` trait. Per-method schema modules live in each method crate (`pyscf_scf::chkfile`, `pyscf_dft::chkfile`, `pyscf_ccsd::chkfile`, `pyscf_geomopt::chkfile`) and `impl Checkpointable for ScfResult` / `for KsResult` / `for CcsdResult` / `for OptimState`. Workspace grows 15 → 16 (this is the first of three new crates introduced in Phase 3; ROADMAP.md update required during planning).

- **D-07: Rust-driven chkfile round-trip oracle.** `pyscf-oracle::oracle_check!("chkfile_roundtrip", fixture)` macro lives in `pyscf-oracle` (dev-deps only, `pyo3 = "=0.28.3"` with `auto-initialize` already declared per Phase 1's existing Cargo.toml). Macro spawns Python via `Python::attach`, runs upstream `pyscf.scf.RHF(mol).kernel()` to produce a chkfile in a tmpdir, then pyscf-rs reads + asserts numpy-allclose at 1e-12 on `mo_coeff`/`mo_energy`/`mo_occ`/`e_tot`. Reverse direction: pyscf-rs writes chkfile, Python `from_chk(path).kernel()`, asserts converged at upstream energy. Same macro shape locks the ORACLE-02 contract for every SCF success criterion.

### DIIS (SCF-04, Pitfall 9; Phase 6 CCSD-04 reuse)

- **D-08: New `pyscf-diis` workspace crate.** Generic over a `pub trait DiisStorable { fn as_flat(&self) -> &[f64]; fn from_flat(&mut self, slice: &[f64]); fn dot(&self, other: &Self) -> f64; }`. Depends only on `pyscf-algebra` for the small B-matrix linear solve (typically ≤ 8×8) and `axpy`/`dot` primitives. `pyscf-scf` and `pyscf-ccsd` consume the trait. Workspace 16 → 17. Pitfall 9 mitigation: all reductions inside DIIS go through Phase 1's `oracle_sum`/`oracle_dot` under `release-oracle` so the extrapolated Fock matches upstream when reduction order is held.

- **D-09: Generic `DiisStorable` trait.** `pyscf-scf::FockSubspace` impls it for Fock matrices (`nao × nao`, F-order); `pyscf-ccsd::AmpsSubspace` impls it for `(T1, T2)` tuples in Phase 6. `pyscf-diis::Diis<S: DiisStorable>` is the generic Pulay-extrapolation stack. Trait is object-safe so boxing works. Each impl picks its own algebra layout.

### Density fitting (SCF-07; Phase 4 DFT-07, Phase 5 MP2-04, Phase 6 CCSD-08 reuse)

- **D-10: New `pyscf-df` workspace crate.** 18th workspace member. Mirrors upstream `pyscf/df/` exactly (sibling-crate fidelity hard preference inherited from Phase 1). Owns 3-center aux integrals (`mol.intor('int3c2e_sph')`), 2-center aux integrals (`mol.intor('int2c2e_sph')`), Cholesky of `(P|Q)` via `pyscf_algebra::cholesky` (host-faer per Phase 1 D-06), and B-integral assembly (`B_{μν}^Q`). Public surface: `DfIntegrals { b_uvq: Tensor, naux: usize }` consumed uniformly by SCF/DFT/MP2/CCSD. Workspace 17 → 18; **net Phase 3 growth: 15 → 18 (+pyscf-chkfile, +pyscf-diis, +pyscf-df)**.

- **D-11: In-memory B integrals in Phase 3; HDF5 spill deferred to Phase 6.** Phase 3 DF-HF ships with the full `B_{μν}^Q` array in memory. Sufficient for the test corpus (H2O/cc-pVDZ, benzene/6-31G*, water trimer) — success criterion 3 makes no `PYSCF_MAX_MEMORY` assertion at Phase 3. Phase 6 CCSD-08 + CCSD-11 (tensor-arena from CCSD day one) extends `pyscf-df` with HDF5 spill via `pyscf-chkfile` primitives. No premature optimization; aligns with ROADMAP.md's explicit Phase 6 placement of the spill machinery.

### Claude's Discretion

The following are not user-decided — researcher/planner picks the implementation within the locked decisions above:

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

### Folded Todos

None — the cross-reference scan found 0 pending todos for Phase 3.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project specs (this repo)
- `.planning/PROJECT.md` — project vision, core value, key decisions, "out of scope" list
- `.planning/REQUIREMENTS.md` lines 41-54 (SCF-01..14), 120-148 (BIND-01..09 with BIND-03,08 in Phase 8), 152-160 (ORACLE-01..09 with Phase 3 owning 02,08) — Phase 3 owns 21 REQs total
- `.planning/ROADMAP.md` §"Phase 3: SCF + PyO3 bindings" — goal, dependencies, 6 numbered success criteria
- `.planning/ROADMAP.md` §"Cross-Cutting Concerns Threaded Through Every Phase" — algebra-responsibility wall, backend selection, bit-exact-with-PySCF, PyO3 subclass-override dispatch, NumPy contiguity, GIL-release seam, panic policy, scope-creep lint, cubecl pin lockstep (Phase 3 establishes the PyO3 subset)
- `.planning/ROADMAP.md` §"Pitfall-to-Phase Mapping" — Phase 3 owns Pitfalls 4 (eigenvector sign), 5 (NumPy zero-copy), 6 (GIL deadlock), 7 (PyO3 subclass override), 9 (DIIS path drift), 11 (chkfile compatibility); re-validates Pitfall 8 (F-order layout from Phase 2), 12 (cross-platform libm from Phase 1), 14 (panic→exception)
- `.planning/STATE.md` §"Blockers/Concerns" — h5py↔hdf5-metno chkfile round-trip robustness needs empirical seal in Phase 3 (ORACLE-08); faer-ext compatibility (transitively relevant)
- `.planning/phases/01-foundation/01-CONTEXT.md` — D-01..15 carried forward: workspace layout, AlgebraClient enum+match dispatch, opaque Tensor/BufferId, host-faer eigh per ALG-05, PYSCF_BACKEND + PYSCF_DTYPE resolver, sibling-crate sourcing under BectorVoom, cubecl 0.10.0 pin
- `.planning/phases/02-gto/02-CONTEXT.md` — D-01..07 carried forward: live basis-file reads (no compile-time codegen), PYSCF_BASIS_PATH resolver, Mole→cintx flat-array eager projection, eval_gto kernel home (pyscf-kernels), ECP loading + EcpEngine trait (gap-closure plan pending cintx ECP merge), F-order convention per `pyscf/gto/moleintor.py`

### Upstream PySCF source (this repo — the oracle)
- `pyscf/scf/__init__.py` — SCF module surface: `RHF`, `UHF`, `GHF` factories, `density_fit`, the public re-exports BIND-02 must preserve
- `pyscf/scf/hf.py` (~2500 lines) — `SCF` base class, `RHF`, `kernel()`, `get_hcore`, `get_init_guess` (5 modes), `get_jk`, `get_veff`, `get_fock`, `get_occ`, `make_rdm1`, `energy_elec`, `energy_tot`, `analyze`, `mulliken_pop`, `dip_moment`, `as_scanner`, `to_uhf`, `to_ghf`, the 30-attribute floor source-of-truth, `init_guess_by_atom`/`init_guess_by_minao`/`init_guess_by_1e`/`init_guess_by_huckel`/`init_guess_by_chkfile`, `canonicalize_signs` reference algorithm
- `pyscf/scf/uhf.py` — `UHF` class: open-shell SCF, alpha/beta density pair, spin-polarized make_rdm1, mulliken_pop_meta_lowdin
- `pyscf/scf/ghf.py` — `GHF` class: 2-component spinor SCF, doubled-AO Fock structure (correctness only — perf parity not required for v1 per SCF-03)
- `pyscf/scf/diis.py` — `CDIIS` (C-DIIS) class: error vector `[F·D·S − S·D·F]`, B-matrix construction, Pulay extrapolation; reference for `pyscf-diis::Diis<S>` semantics under `release-oracle`
- `pyscf/scf/chkfile.py` + `pyscf/lib/chkfile.py` — chkfile schema source-of-truth: `dump_scf(chkfile, mol, e_tot, mo_energy, mo_occ, mo_coeff)`, `load_scf`, `save_mol` (writes `mol.dumps()` JSON string under `/mol`), HDF5 group layout, dataset names, F-order `mo_coeff` storage
- `pyscf/df/df.py` + `pyscf/df/df_jk.py` + `pyscf/df/incore.py` — DF-HF reference: `DF` class, `auxbasis` resolution (`weigend`, `cc-pvdz-jkfit` defaults), 3-center integral assembly, Cholesky of `(P|Q)`, B-integral `(μν|P)·(P|Q)^{-1/2}` computation
- `pyscf/lib/__init__.py` — `lib.num_threads()` (consumed by ORACLE-09 reduction-order pinning)
- `pyscf/__init__.py` — module-level `M(...)` factory, `import pyscf` namespace shape (BIND-02 contract)

### Sibling-crate PyO3 precedent (read before implementing analogous surface)
- `~/Documents/workspace/cintx/crates/cintx-rs/` — closest PyO3 binding precedent; check for `#[pyclass]`/`#[pymethods]` patterns, `Bound<'_, Self>` usage, abi3 `[lib]` config
- `~/Documents/workspace/xcfun_rs/crates/xcfun-py/` — second PyO3 binding precedent; check for `Python::detach` usage, NumPy boundary helpers, GILOnceCell sites
- `~/Documents/workspace/cintx/crates/cintx-runtime/` — analog for the `Python::detach`-around-compute pattern at the runtime layer

### Phase 1 + Phase 2 shipped artifacts (this repo)
- `crates/pyscf-core/src/lib.rs` — universal types and method traits (zero compute deps per FOUND-02); Phase 3 fills `Scf` trait impls
- `crates/pyscf-core/src/density.rs` — `Density { nao, data }` skeleton; Phase 3 wires AO-basis density-matrix construction
- `crates/pyscf-core/src/mo.rs` — `MOCoefficients { nao, nmo, data, energies, occupations }` skeleton (F-order); Phase 3 fills via `pyscf-algebra::eigh` + `canonicalize_signs`
- `crates/pyscf-core/src/energy.rs` — `Energy(f64)` newtype in Hartree
- `crates/pyscf-core/src/traits.rs` — declares `Method`, `Scf`, `KohnSham`, `PostScf`, `Gradient`, `IntegralEngine`, `EcpEngine`; Phase 3 implements `Scf` for RHF/UHF/GHF
- `crates/pyscf-core/src/mole.rs` — Mole with ≥30-attribute floor (Phase 2 shipped); Phase 3 adds the callable-form atom input (5th form, NotYetImplemented in Phase 2 D-deferred) — defer if planner pushes back
- `crates/pyscf-core/src/basis_set.rs` — `BasisSet` zero-copy re-export of `cintx_core::BasisSet` (Phase 2 D-03)
- `crates/pyscf-gto/src/lib.rs` — `pyscf_gto::M(MoleBuildArgs)`, `mol.intor(name)` dispatcher (Phase 2 D-04..05), `eval_gto` user wrapper (Phase 2 D-04). Phase 3 SCF consumes these for `Fock = h_core + J - K/2`.
- `crates/pyscf-algebra/src/lib.rs` — `gemm`, `gemv`, `axpy`, `dot`, `reduce_sum`, `transpose`, `scal`, `oracle_sum`, `oracle_dot`, `oracle_einsum`, `eigh` (host-faer per Phase 1 D-06). Phase 3 consumes all of these in the Fock build + diagonalization loop.
- `crates/pyscf-runtime/src/lib.rs` — `BackendKind`, `select_backend()`, `WorkspacePool`, tracing init, `PYSCF_BACKEND` + `PYSCF_DTYPE` resolvers (Phase 1 D-07..11). Phase 3 logs `pyscf-algebra: backend=<resolved> dtype=<resolved>` at SCF kernel entry per ALG-08.
- `crates/pyscf-scf/src/lib.rs` — Phase 1 stub (empty `lib.rs`); Phase 3 fills with `RHF`/`UHF`/`GHF` structs, `OverrideHooks` trait, generic `kernel<H>`.
- `crates/pyscf-py/src/lib.rs` — Phase 1 stub with `crate-type = ["cdylib", "rlib"]`; Phase 3 wires `#[pymodule] fn _native(...)` with the `scf` submodule, `#[pyclass] PyRHF`/`PyUHF`/`PyGHF` + `PyOverrideBridge` impl.
- `crates/pyscf-oracle/Cargo.toml` — Phase 1 already declares `pyo3 = { version = "=0.28.3", features = ["auto-initialize"] }` in `[dev-dependencies]` (ORACLE-01). Phase 3 fills the macro library.
- `Cargo.toml` (workspace) — Phase 3 adds 3 new members: `crates/pyscf-chkfile`, `crates/pyscf-diis`, `crates/pyscf-df`. Adds `[patch.crates-io]` for `hdf5-metno` if needed. Adds `pyo3 = "=0.28.3"` to `[workspace.dependencies]`.
- `xtask/src/lints/algebra_wall.rs` (Phase 1) — Phase 3 extends the algebra-wall allowlist with `pyscf-chkfile`, `pyscf-diis`, `pyscf-df` if they need pyscf-algebra. `pyscf-chkfile` does NOT need cubecl (HDF5 is pure I/O); `pyscf-diis` and `pyscf-df` DO consume `pyscf-algebra`.

### Cubecl reference docs (this repo)
- `docs/manual/Cubecl/Cubecl_multi_ compute.md` — runtime/ComputeClient pattern; reference for any pyscf-df fused-kernel work (deferred to Phase 8 if needed)
- `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` — authoritative for `pyscf_algebra::gemm` calls inside Fock build

### External (Phase 3 will need to look up)
- PyO3 0.28 guide §`trait-bounds.md` — authoritative for the callback-trait + `UserModel { model: Py<PyAny> }` bridge pattern (D-01)
- PyO3 0.28 guide §`class.md` — `#[pyclass(subclass)]`, `PyClassInitializer`, `as_super`/`into_super`, `Bound<'_, Self>::call_method1` — authoritative for `PyRHF`/`PyUHF`/`PyGHF` inheritance
- PyO3 0.28 migration guide — `Python::attach` (was `Python::with_gil`), `Python::detach` (was `py.allow_threads`), `pyo3::sync::GILOnceCell` (replaces `lazy_static!` in GIL-touching paths)
- numpy 0.28 (PyO3 ecosystem) — `PyReadonlyArray2<'py, f64>`, `as_array`, `is_standard_layout`, `to_owned` semantics — authoritative for D-04 type-specific converters and BIND-04 stride-fuzz test
- maturin docs — `python-source` config for `python/pyscf/__init__.py` overlay packaging; abi3-py310 wheel build flags
- hdf5-metno crate docs — `hdf5::File::create`, `Group`, `Dataset`, attribute API, `hdf5-metno-sys` `static` feature for DIST-05
- faer 0.24 docs — `solve_linear`, `cholesky` for DF `(P|Q)^{-1/2}` (already host-fallback path per Phase 1 D-06)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`pyscf-core::Mole` + 30-attribute floor + `intor` dispatcher** (Phase 2 D-03, D-04) — Phase 3 SCF consumes `mol.intor('int1e_kin')`, `mol.intor('int1e_nuc')`, `mol.intor('int1e_ovlp_sph')`, `mol.intor('int2e_sph')` directly with no rebuild work. F-order convention preserved per Pitfall 8.
- **`pyscf-core::Density` + `MOCoefficients`** — Skeleton shipped in Phase 1; Phase 3 fills the data buffers. Direct extension, no shape rework.
- **`pyscf-algebra::{gemm,axpy,dot,reduce_sum,eigh,oracle_sum,oracle_dot}`** — Phase 1 surface is sufficient for the entire SCF compute path (Fock build is `gemm` chains + `oracle_sum` reductions; diagonalization is `eigh`; DIIS is small linear solve + axpy/dot).
- **`pyscf-runtime::select_backend()` + `PYSCF_BACKEND`/`PYSCF_DTYPE` resolvers** (Phase 1 D-07..11) — Phase 3 logs the resolved client at SCF kernel entry per ALG-08; no changes needed.
- **`pyscf-oracle/Cargo.toml` already declares `pyo3 = "=0.28.3"` in dev-deps** — Phase 1 shipped this for ORACLE-01. Phase 3 only adds the macro module to `pyscf-oracle/src/`.
- **`crates/pyscf-py/Cargo.toml` already has `crate-type = ["cdylib", "rlib"]`** — Phase 1 wired this; Phase 3 adds `pyo3 = "0.28.3"` to dependencies + the `#[pymodule]` body.
- **Phase 2 `mol.dumps()`/`Mole::loads()` JSON round-trip** — chkfile's `/mol` group stores the JSON string from `mol.dumps()`, exactly mirroring upstream `pyscf/lib/chkfile.py:save_mol`. No new serialization work needed for SCF chkfiles.
- **Sibling-crate PyO3 patterns at `cintx/crates/cintx-rs/` and `xcfun_rs/crates/xcfun-py/`** — direct templates for abi3 `[lib]` config, `#[pymodule]` shape, `Python::detach` placement, NumPy boundary helpers.

### Established Patterns

- **Algebra wall** (Phase 1 D-04..06) — `pyscf-scf`, `pyscf-diis`, `pyscf-df` depend on `pyscf-algebra` only; never on `cubecl-*` directly. `pyscf-chkfile` has no algebra dep at all (pure I/O). Enforced by the xtask `algebra-wall` lint extended to the 3 new crates.
- **Sibling-crate fidelity (hard preference)** — `pyscf-df` mirrors upstream `pyscf/df/` exactly; `pyscf-diis` mirrors `pyscf/scf/diis.py`; `pyscf-chkfile` mirrors `pyscf/lib/chkfile.py` + per-method `chkfile.py` modules.
- **"Don't freeze compile"** (user memory + Phase 2 D-01) — no heavy build.rs, no parse-N-files macros, no libxc_rs in Phase 3 dep graph. Confirmed.
- **Env-var resolver pattern** (Phase 1 D-07; Phase 2 D-02) — Phase 3 introduces no new env vars; consumes existing `PYSCF_BACKEND`, `PYSCF_DTYPE`, `PYSCF_BASIS_PATH`, and adds `PYSCF_MAX_MEMORY` is read (not introduced by Phase 3) only to surface the budget at SCF kernel entry log.
- **Bit-exact-with-upstream under `release-oracle`** — Phase 3 success criterion 1 + 2 assert ≤1 µHartree energy parity AND cross-platform Linux/macOS µHartree consistency.
- **`#[forbid(unsafe_code)]` + `#[warn(clippy::unwrap_used)]`** (Phase 1) — all 3 new crates adopt these.
- **Per-hook `Python::detach`** (D-03 — new convention established by Phase 3; inherited by Phase 4-7).

### Integration Points

- **`crates/pyscf-scf/Cargo.toml`** — Phase 3 adds deps: `pyscf-core`, `pyscf-algebra`, `pyscf-gto`, `pyscf-diis` (new), `pyscf-df` (new), `pyscf-chkfile` (new), `pyscf-runtime`, `cintx-rs` (for `int2e_sph` calls through `mol.intor`), `tracing`, `thiserror`. **No pyo3 dep** (per D-01 trait-callback bridge).
- **`crates/pyscf-py/Cargo.toml`** — Phase 3 adds deps: `pyscf-scf`, `pyscf-core`, `pyscf-gto`, `pyscf-runtime`, `pyo3 = "=0.28.3"` (with `abi3-py310` + `extension-module` features), `numpy = "0.28"`. cdylib output named `_native`.
- **`crates/pyscf-chkfile/` (new)** — depends on `pyscf-core` (for `Checkpointable` trait + result types), `hdf5-metno = "<version>"` (the only crate that pulls libhdf5), `serde_json` (for Mole roundtrip), `tracing`.
- **`crates/pyscf-diis/` (new)** — depends on `pyscf-core`, `pyscf-algebra` only.
- **`crates/pyscf-df/` (new)** — depends on `pyscf-core`, `pyscf-algebra`, `pyscf-gto` (for `mol.intor('int3c2e_sph')` + `int2c2e_sph`).
- **`python/pyscf/__init__.py` (new overlay)** — re-exports from `_native.scf`. Maturin's `python-source = "python"` config bundles it into the wheel so `from pyscf import scf` resolves to pyscf-rs's overlay rather than the upstream tree (verified at install time by import precedence).
- **`xtask/src/lints/algebra_wall.rs`** — Phase 3 extends the allowlist to include the 3 new crates: pyscf-chkfile (no algebra needed, no cubecl), pyscf-diis (algebra only), pyscf-df (algebra + gto, no direct cubecl).
- **`.github/workflows/ci.yml`** — Phase 3 adds: (a) `python3.13t` free-threaded build smoke for BIND-05; (b) `maturin develop` + `pytest` smoke for the abi3 wheel skeleton; (c) BIND-04 stride-fuzz oracle test; (d) cross-platform Linux/macOS µHartree assertion (Pitfall 12).
- **`.github/workflows/nightly-cross-crate.yml`** — Phase 3 extends to bump `hdf5-metno` alongside the cubecl pin if a sibling-crate cubecl bump warrants. No libxc_rs entries added (Phase 4 owns that).

</code_context>

<specifics>
## Specific Ideas

- **Sibling-crate fidelity is a hard preference** (carried from Phase 1, reinforced in Phase 2) — `pyscf-df` mirrors `pyscf/df/`, `pyscf-diis` mirrors `pyscf/scf/diis.py`, `pyscf-chkfile` mirrors `pyscf/lib/chkfile.py` + per-method chkfile modules. Deviation requires explicit justification.
- **Workspace growing from 15 to 18 in Phase 3** is intentional and accepted — each new crate (`pyscf-chkfile`, `pyscf-diis`, `pyscf-df`) corresponds to a single owner of a horizontal-layer concern (HDF5 binding, Pulay extrapolation, DF integrals). ROADMAP.md needs an explicit update in planning to reflect 18-member count + the new crates' positions in the dep graph.
- **`OverrideHooks` trait is the single PyO3 contract for the whole project** — Phase 4-7 method phases all inherit this shape; their respective `KsOverrideHooks`/`PostScfOverrideHooks`/`GradOverrideHooks` follow the same pattern. The trait-bridge design from D-01 means each method crate stays pyo3-free.
- **Per-hook `Python::detach` is the GIL contract** — Phase 4-7 inherit by convention; the seam is at the hook body's compute, not at the kernel-call boundary. python3.13t free-threaded CI is the validation.
- **Type-specific NumPy converters live in pyscf-py only** — pyscf-scf/diis/df NEVER see PyArray. The boundary policy (BIND-04) is enforced at the wrapper crate, single source of truth.
- **DIIS reductions go through `oracle_sum`/`oracle_dot`** (Pitfall 9 mitigation) — under `release-oracle`, DIIS extrapolation is bit-exact reproducible cross-platform. Same convention as Phase 1's algebra primitives.
- **DF B integrals stay in memory in Phase 3** — Phase 6 (CCSD-08 + CCSD-11) adds HDF5 spill. Don't pre-optimize.
- **`chkfile_roundtrip` is the canonical ORACLE-08 fixture name** — referenced by the `oracle_check!` macro; reused in Phase 4-7 with method-specific suffixes.
- **`canonicalize_signs` is the cross-platform vendor-stability anchor** (Pitfall 4 + Pitfall 12 mitigation, SCF-13) — lives in `pyscf-core::lib::canonicalize_signs` (pure function, no algebra dep), called by SCF/DFT/MP2/CCSD post-eigh.

</specifics>

<deferred>
## Deferred Ideas

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

### Reviewed Todos (not folded)
None — todo cross-reference scan returned 0 matches for Phase 3.

</deferred>

---

*Phase: 03-scf-pyo3-bindings*
*Context gathered: 2026-05-11*
