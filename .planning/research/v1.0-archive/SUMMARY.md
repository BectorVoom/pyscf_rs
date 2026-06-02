# Project Research Summary

**Project:** pyscf_rs — pure-Rust rewrite of PySCF (drop-in `pyscf.*` import surface, cubecl on hot paths, PyO3 wheel)
**Domain:** Molecular ground-state quantum chemistry library
**Researched:** 2026-05-09
**Confidence:** HIGH overall (sibling-crate Cargo.toml audit, in-tree PySCF source, crates.io versions verified 2026-05-09)

> Source files (do not duplicate, link in for detail):
> [STACK.md](./STACK.md) · [FEATURES.md](./FEATURES.md) · [ARCHITECTURE.md](./ARCHITECTURE.md) · [PITFALLS.md](./PITFALLS.md)

---

## Executive Summary

pyscf_rs is a pure-Rust quantum-chemistry library that ships as a single PyPI wheel preserving PySCF's `from pyscf import gto, scf, dft, …` import surface for seven in-scope methods (gto, scf, dft, mp2, ccsd, grad, geomopt). The architectural pattern is locked: a 14-crate workspace mirroring the `cintx`/`xcfun_rs` **horizontal layered façade** (`pyscf-{core,runtime,kernels}` shared substrate at the bottom; method crates `pyscf-{gto,scf,dft,mp2,ccsd,grad,geomopt}` above; `pyscf-py` cdylib and `pyscf-oracle` dev-only on top). Every cubecl `#[cube]` body lives in **one** `pyscf-kernels` crate (not seven `*-cubecl` crates), because vhf is shared between SCF/DFT and ao2mo is shared between MP2/CCSD/grad — splitting by method would either duplicate kernels or break the DAG. cubecl 0.10, PyO3 0.28.3, numpy 0.28.0, faer 0.24, ndarray 0.17, hdf5-metno 0.12 are pinned in lockstep with the sibling crates; MSRV 1.92, edition 2024, Apache-2.0.

The dependency graph is acyclic and dictates phase ordering almost entirely: `core → runtime → kernels → gto → scf → {dft, mp2} → ccsd → grad → geomopt → bindings → wheel`. CCSD imports MP2 helpers (`get_nocc`, `get_frozen_mask`, `_mo_without_core`) so MP2 must be solid before CCSD begins. DFT (~9000 LOC upstream) is the single largest phase by surface area. The drop-in API floor is concrete: a top-20-idiom contract from `examples/*.py` plus ≥30 attribute names per major class (Mole, SCF, MP2, CCSD, Gradients) — these are the hard requirements. v1 is explicitly molecular-only; pbc, x2c/dhf, mcscf, tdscf/tddft/eom/gw, CCSD(T)+, NAC, solvent, QM/MM are out-of-scope and `Out of Scope` lints are required to prevent creep.

The risk profile is concentrated and unusual. Five SHOWSTOPPER pitfalls cluster around two themes: **bit-exact-with-PySCF** (FMA contraction, parallel-reduction order, eigenvector sign convention, DIIS path drift, chkfile schema, cross-platform libm) and **PyO3 drop-in fidelity** (subclass-override dispatch via `slf.call_method1`, NumPy stride/contiguity at the FFI boundary, GIL deadlock in callback paths, panic-across-FFI = UB). The first cluster must be addressed in an `infra` phase before any numerical kernel lands; the second clusters in a `bindings` phase that sets the conventions every later method follows. The pre-1.0 cubecl pin (=0.10.0) is the single biggest external risk: a sibling-crate version drift breaks four crates simultaneously, and WGPU has known f64 holes (cubecl issues #1316/#1317) so consumer-GPU support is gated behind a `shader-f64` Vulkan extension check, not silently degraded to f32.

---

## Key Findings

### Recommended Stack

Pure Rust, no system BLAS/LAPACK at install time. Compute primitive is **cubecl 0.10.0** (CPU SIMD default, CUDA/WGPU/ROCm optional, all five `cubecl-*` crates pinned with `=0.10.0` in lockstep with `cintx`/`libxc_rs`/`xcfun_rs`); host dense linear algebra is **faer 0.24.0** (eigh on every SCF cycle; complex `c64` for GHF); array container is **ndarray 0.17.2** (zero-copy with `numpy 0.28`). PyO3 0.28.3 + maturin 1.13 builds an **abi3-py310** wheel covering Python 3.10–3.14 in one binary per OS/arch. Chkfile I/O via **hdf5-metno 0.12.4** with bundled libhdf5 (`hdf5-sys/static`). Geometry-opt driver via **argmin 0.11** (BFGS native, no `geomeTRIC`/`pyberny` Python dependency). MSRV `1.92`, edition `2024`, Apache-2.0 — every value matches the sibling workspace exactly.

**Core technologies:**
- **cubecl 0.10.0** — sole compute kernel framework (CPU/CUDA/WGPU/ROCm) — must move lockstep with cintx/libxc_rs/xcfun_rs; bumping pyscf_rs alone breaks the family. See [STACK §2.2](./STACK.md#22-cubecl-ecosystem).
- **faer 0.24.0** — host eigh/Cholesky/LU/QR/SVD/GEMM in pure Rust, complex scalars supported — eliminates the OpenBLAS/MKL install dance that motivates the rewrite. See [STACK §2.1](./STACK.md#21-linear-algebra--tensor-library).
- **ndarray 0.17.2** — rank-N container; zero-copy across the `numpy 0.28` PyO3 boundary. Not `nalgebra` (wrong shape) and not `ndarray-linalg` (pulls system BLAS).
- **PyO3 0.28.3 + numpy 0.28.0 + maturin 1.13.1** — single abi3-py310 wheel; `from pyscf import gto, scf, dft` preserved via PyO3 submodules in one cdylib + a thin `python/pyscf/__init__.py` re-export shim.
- **hdf5-metno 0.12.4** with `hdf5-sys/static` — bundles libhdf5 in the wheel; the chkfile schema is byte-for-byte compat with upstream PySCF (h5py-written files).
- **argmin 0.11.0** — geomopt's BFGS+RFO foundation; lets the project ship without a Python `geomeTRIC` dependency.
- **Anti-recommendations**: `cubecl-linalg 0.5.0` is **stale** (pinned to cubecl 0.5, year-old skew, do not use); `ndarray-linalg`, `nalgebra-lapack` pull system LAPACK; `candle-core` would introduce a second GPU runtime; `hdf5 0.8.1` (the aldanor original) is unmaintained since 2021.

### Expected Features

**Drop-in API floor (hard requirements):** the top-20 idioms in [FEATURES §1](./FEATURES.md#1-drop-in-api-contract--top-20-idioms) — `pyscf.M(atom=..., basis=...)`, `scf.RHF(mol).run()`, `scf.RHF(mol).density_fit().run()`, `mol.RHF().MP2().run()`, `mf.CCSD().run()`, `mf.nuc_grad_method().kernel()`, `from pyscf.geomopt.geometric_solver import optimize`, `mf.chkfile = ...`, `mf.kernel(dm0)`, `mf.analyze()`, `mol.intor('int2e')`, `mf.mo_coeff/mo_energy/mo_occ/e_tot/converged` — these are the requirements seed; existing PySCF scripts using only in-scope methods must run unchanged. Per-class attribute floors (≥30 names each on Mole / SCF / MP2 / CCSD / Gradients, listed in [FEATURES §7](./FEATURES.md#7-drop-in-api-compatibility-floor)) are non-negotiable.

**Must have (table stakes — v1 P1):**
- `gto.Mole`: 5 atom-input forms, 11 basis-input forms, ECP, all `nao_*`/`atom_*`/`ao_loc_*`/`ao_labels` accessors, `intor()` thin wrapper over cintx, `dumps`/`loads`, `set_geom_`, `copy`.
- `scf`: RHF, UHF, GHF (basic), C-DIIS, level shift, init_guess ∈ {minao, atom, 1e, chkfile, dm0}, `density_fit`, all overrideable `get_*` hooks, `analyze`/`mulliken_pop`, chkfile save/load, `to_uhf/to_rks/...` cross-module dispatch helpers.
- `dft`: RKS, UKS, libxc + xcfun XC parser (with comma-form `'pbe,pbe'` and shorthands), `Grids` with level/atom_grid/prune/radi_method/becke_scheme, range-separated hybrids (omega via `int2e_lr/sr_*`), VV10 NLC, DF-DFT.
- `mp2`: RMP2, UMP2, frozen-core (int/list/auto/window), in-core ao2mo, DF-MP2, RDMs.
- `cc`: RCCSD, UCCSD, frozen-core, `solve_lambda`, RDMs, AO-direct, DF-CCSD, T1/D1/D2 diagnostics.
- `grad`: RHF/UHF/RKS (with `grid_response=True`)/UKS/MP2/CCSD, ECP gradient, atom-list subsetting.
- `geomopt`: native Rust BFGS+RFO in redundant internals; `pyscf.geomopt.geometric_solver.optimize` and `berny_solver.optimize` are drop-in shims that delegate to the native optimizer (no Python `geomeTRIC`/`pyberny` runtime dependency).
- Cross-cutting: `lib.StreamObject`, `lib.logger`, chkfile (HDF5), `mf.max_memory`, PySCF-as-oracle CI, PyO3 bindings preserving `pyscf.*`.

**Should have (differentiators):**
- 2–5× speedup vs PySCF on the defined benchmark suite (the rewrite's raison d'être; PROJECT.md goal).
- All four cubecl backends (CPU SIMD/CUDA/WGPU/ROCm) from one source.
- Sub-second `mol.build()` for 5000-AO molecules; strict basis-name resolution with did-you-mean errors.
- Reproducible SCF (deterministic eigh tiebreak, stable across LAPACK vendors).
- Native Rust optimizer with bit-stable trajectory and full HDF5 checkpoint of optimizer state.
- Memory-aware grid streaming and bounded-memory DF-CCSD.

**Defer to v1.x (P2 — explicit user-pull pressure expected):**
- **CCSD(T)** — the most painful omission; 30–40% of CCSD users want it. Flag for v1.x P1 with explicit research-flag in roadmap.
- ROHF + ROHF gradients (v1 falls back to UHF + warning).
- `scf.newton(mf)` SOSCF for hard-converge cases.
- DFT-D3/D4 dispersion (`mf.disp = 'd3bj'`).
- Hessian (RHF/RKS) for vibrational frequencies.
- FNO-CCSD; ADIIS/EDIIS; full GHF/GMP2/GCCSD path; constrained geomopt.

**Out of scope (v2+ — anti-feature lints required):**
PBC/k-points, x2c/sfx2c1e/DHF, mcscf/casci/casscf/mcpdft/mrpt, tdscf/tddft/TDA gradients, eom-ccsd/adc/gw, CCSD(T) and beyond at v1, NAC/EPH, solvent (PCM/ddCOSMO/SMD), QM/MM, localized orbitals, multi-node MPI, conda channel, custom-Hamiltonian SCF for model systems. See [FEATURES §8](./FEATURES.md#8-anti-features-summary-consolidated).

### Architecture Approach

**14-crate horizontal layered façade workspace** mirroring cintx/xcfun_rs, NOT libxc_rs's flat-kernels pattern. Bottom-to-top: `pyscf-core` (types, traits, errors, Mole struct — no compute) → `pyscf-runtime` (BackendKind, planner, workspace pool, env-var config — no cubecl deps, just typed enum arms gated by features) → `pyscf-kernels` (every `#[cube]` body, with internal modules for vhf/ao2mo/numint/cc_tensor/grad and per-method feature gates `with-mp2`/`with-ccsd`/`with-grad` to keep small builds small) → seven method crates `pyscf-{gto,scf,dft,mp2,ccsd,grad,geomopt}` (acyclic DAG: dft and mp2 depend on scf; ccsd depends on mp2; grad depends on every method it differentiates; geomopt sits on top of grad) → `pyscf-py` (single PyO3 cdylib with submodules — one wheel; cross-method types like `Mole` are the same `pyclass` everywhere) + `pyscf-oracle` (dev-only crate; `Python::with_gil` calls upstream PySCF in-process; listed in `dev-dependencies` only, never `dependencies`, so release wheels never link Python) + `pyscf-bench` (criterion). External path-deps: `cintx`, `libxc_rs`, `xcfun_rs` from sibling workspaces.

**Major components:**
1. **`pyscf-core`** — `Mole`, `BasisSet` (re-exported from cintx_core), `AOIntegrals/MOIntegrals` handles, `Density` (RDM1/RDM2), `MOCoefficients/MOEnergies`, `Amplitudes<T1/T2>`, `Energy` (Hartree newtype), traits `Method/Scf/KohnSham/PostScf/Gradient/XcFunctional/IntegralEngine`. **Mole lives here**, not in `pyscf-gto` (anti-pattern §4 in ARCHITECTURE.md), so post-SCF tests don't artificially depend on `pyscf-gto`.
2. **`pyscf-runtime`** — `BackendKind::{Cpu,Cuda,Wgpu,Rocm,Metal}` typed enum (Metal is a feature alias for Wgpu — `cubecl-metal` does not exist on crates.io); `auto_backend()` priority chain (`PYSCF_BACKEND` env → CUDA → ROCm → Metal → WGPU → CPU); `WorkspacePool`; spill-to-HDF5 allocator when `PYSCF_MAX_MEMORY` exceeded.
3. **`pyscf-kernels`** — single crate, all `#[cube]` bodies. Backend feature flags forward to `pyscf-runtime`; per-method feature gates (`with-mp2`, `with-ccsd`, `with-grad`) so SCF-only consumers don't compile CCSD kernels. Mirrors `xcfun-kernels` role.
4. **Seven method crates** (`pyscf-gto/scf/dft/mp2/ccsd/grad/geomopt`) — the safe-Rust facades with state structs and `kernel()` drivers. Composition + traits, not inheritance (anti-pattern §6: `KohnSham<RhfState>` wraps an SCF state; doesn't inherit from it).
5. **`pyscf-py`** — one cdylib exposing `_native.{gto,scf,dft,mp,cc,grad,geomopt}`; `python/pyscf/` is a thin Python source dir that re-exports + preserves `PYSCF_EXT_PATH` plugin loader. No per-method `*-py` crates (would split `Mole` into incompatible types across submodules).
6. **`pyscf-oracle`** — `pyo3::Python::with_gil` harness that imports upstream `pyscf` and runs the same calculation; `dev-dependencies` only.

Parallelizable build waves after the critical path are: W5 = (dft, mp2) parallel after scf converges; W6 = ccsd needs mp2; W7 = grad needs all methods; W8 = (geomopt, py) parallel.

### Critical Pitfalls

The five SHOWSTOPPER pitfalls and how to address them (full table in [PITFALLS §Pitfall-to-Phase Mapping](./PITFALLS.md#pitfall-to-phase-mapping); 21 total pitfalls):

1. **PyO3 subclass-override dispatch breaks polymorphism** ([Pitfall 7](./PITFALLS.md#pitfall-7)). Rust `self.get_veff(dm)` dispatches via Rust MRO and silently bypasses the user's Python override `class MyHF(scf.RHF): def get_veff(...)`. Fix: every overridable hook (`get_jk`, `get_veff`, `get_hcore`, `get_init_guess`, `get_fock`, `eig`, `get_occ`, `make_rdm1`, `energy_elec`, `energy_tot`) dispatches via `slf.call_method1(py, "name", args)`. Audit `grep -rn "def get_jk\|def get_veff\|..." pyscf/` and enumerate. Phase: `bindings`, revalidated in every method.
2. **FMA contraction breaks bit-exact** ([Pitfall 1](./PITFALLS.md#pitfall-1)). LLVM autovectorizer emits `vfmadd*` from separate mul+add; PySCF's GCC-built C extensions don't. ~1 ulp drift per fma compounds via DIIS into mHartree-scale divergence. Fix: `[profile.release-oracle]` with `RUSTFLAGS="-C target-feature=-fma"`; named arithmetic primitives; CI grep for `llvm.fmuladd` in oracle-mode object files. Phase: `infra`.
3. **Parallel-reduction order non-determinism** ([Pitfall 2](./PITFALLS.md#pitfall-2)). `rayon::par_iter().sum()` and cubecl atomic-reduce produce different last bits per thread count / scheduler. Fix: `oracle_sum` / `oracle_dot` / `oracle_einsum` primitives with fixed chunk-256 ordered tree; pin `RAYON_NUM_THREADS=1` and `mol.lib.num_threads(1)` in oracle CI; document GPU backends as chemical-accuracy not bit-exact. Phase: `infra`.
4. **chkfile byte-for-byte schema compat** ([Pitfall 11](./PITFALLS.md#pitfall-11)). Users restart from chkfiles — broken layout breaks the drop-in promise. Round-trip oracle (PySCF writes, Rust reads, asserts; Rust writes, PySCF reads, runs downstream calc). Defer pickled-Python pieces by routing through PyO3 → CPython pickle (never reimplement). Phase: `scf` (RHF chkfile sets the schema), revalidated per method.
5. **Eigenvector sign / degenerate-subspace ordering** ([Pitfall 4](./PITFALLS.md#pitfall-4)). LAPACK `dsyev` makes no sign promise; vendors differ. Fix: deterministic post-diagonalization sign canonicalization (largest-|coefficient|-with-lowest-index sign-flip); never compare MO coefficients element-wise — compare invariants (density, energy) or `|<C_rust|C_pyscf>|² == 1` overlap. Phase: `scf`.

Two more that act SHOWSTOPPER-tier in their phases: **Rust panic across FFI = UB** (Pitfall 14 — wrap every `extern "C"` callback in `catch_unwind`, no `unwrap()` in numerical code; phase `infra`); **sibling-crate cubecl-version drift** (Pitfall 15 — workspace pin `[patch.crates-io]`, nightly cross-crate matrix; phase `infra` + `oracle`).

MAJOR pitfalls cluster at: cubecl pre-1.0 churn / WGPU f64 holes (#3 — `wgpu` feature gated on `shader-f64` Vulkan extension); NumPy zero-copy contiguity hazards (#5 — `to_owned()` on entry if `!is_standard_layout()`); GIL deadlock on Python 3.13 free-threaded (#6 — single-threaded callback model, test under `python3.13t`); F-order vs C-order mismatch (#8 — `Array2::default(shape.f())` everywhere BLAS-bound); DIIS path drift (#9 — bit-exact contract is on energy/density, not convergence path); DFT grid weights (#10 — port `gen_grid.py` byte-for-byte); cross-platform libm drift (#12 — `libm` crate not platform `f64::sin`); GPU launch overhead on small molecules (#19 — adaptive backend dispatch); CCSD memory thrash (#20 — scratchpad pattern); scope creep (#21 — forbidden-paths lint).

---

## Implications for Roadmap

The architecture's DAG and the pitfall phase-map jointly dictate phase ordering. Eleven phases below; phases 0/8/10 are cross-cutting infra/integration, the rest map 1:1 to method crates.

### Cross-cutting findings the roadmapper must see

These are emergent from combining the four research files; no single document calls them out:

- **DFT is the single largest phase by every metric simultaneously.** ~9000 LOC upstream surface; the most overrideable methods (= largest subclass-dispatch audit surface in `bindings`); the most cubecl f64 risk (numint kernels are the heaviest f64 workload — WGPU may be unusable per Pitfall 3); the only phase that integrates all three sibling crates (`cintx` + `libxc_rs` + `xcfun_rs`). **Front-load the oracle harness** before DFT begins; expect to need a dedicated phase-research subagent for DFT (grid weights, libxc string parser semantics, VV10 stability).
- **CCSD is the second largest phase and inherits MP2's correctness floor.** CCSD imports `mp.mp2.{get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core}` directly (verified at `cc/ccsd.py:35`). MP2 must be solid first. CCSD also owns the largest memory pressure (Pitfall 20 — `Wabef` is `nv^4` ≈ 4 GB at caffeine size); the scratchpad/memory-pool pattern must be in place from the start of `ccsd`, not retrofitted.
- **The bindings layer's contract surface is bigger than it looks.** Five SHOWSTOPPER-tier pitfalls (subclass dispatch, NumPy stride, GIL deadlock, panic-across-FFI, FMA) and three MAJORs (chkfile, sign canonicalization, F-order layout) all fix their conventions in either `infra` or `bindings`. Every method phase reuses these conventions; mistakes there are paid back N=7 times.
- **The `infra` phase is unusually critical.** Foundational decisions (panic policy, FMA profile, ordered-reduction primitives, cubecl pin, F-order newtype, `forbidden-paths` lint) all land here and gate every subsequent kernel. Conversely, `infra` is research-light, design-heavy — most decisions are already made in this research; the work is encoding them.
- **Geomopt is small but its scope decision is locked**: native Rust BFGS+RFO with `argmin`, with `pyscf.geomopt.geometric_solver.optimize` and `berny_solver.optimize` as drop-in Python shims that delegate to the Rust optimizer. **Do not** wrap upstream Python `geomeTRIC` — that would create a permanent Python runtime dep for a project whose value prop is "pure Rust." Default convergence thresholds match geomeTRIC bit-for-bit.
- **The cubecl pin is a four-crate-wide ABI contract.** Every cubecl bump in pyscf_rs requires synchronized bumps in cintx, libxc_rs, xcfun_rs. Document the upgrade ritual in `infra` (CONTRIBUTING.md checklist + nightly cross-crate matrix CI). pre-1.0 instability is the single biggest external risk.
- **GPU backends are deferred to a late phase, not built in per-method.** Ship CPU-correct on every method first; enable cuda/wgpu/rocm features in one dedicated phase that runs the regression suite per backend. Mirrors xcfun_rs's "GPU after substrate solid" sequencing.
- **CCSD(T) pressure is real but out-of-scope for v1.** Roadmap should note v1.x P1 explicitly so the deferral is visible and the research flag is on file.

### Suggested phase structure

| # | Phase | Rationale | Delivers | Pitfalls addressed |
|---|---|---|---|---|
| 0 | **Foundation / Infra** | Five SHOWSTOPPER conventions land here before any kernel; cubecl pin and sibling-crate ABI; `pyscf-core` types and `pyscf-runtime` dispatch are the universal substrate | Workspace builds; `pyscf-{core,runtime}` crates; `[profile.release-oracle]` profile; `oracle_sum`/`oracle_dot` ordered-reduction primitives; F-order newtype; panic policy + `catch_unwind` lint; `forbidden-paths` lint preventing scope creep; cubecl=0.10.0 lockstep; `BackendKind` enum with Metal-as-Wgpu-alias; nightly cross-crate matrix CI | 1 (FMA), 2 (reduction), 3 (cubecl pin), 12 (cross-platform), 14 (panic-across-FFI), 15 (sibling drift), 21 (scope creep) |
| 1 | **gto** | Lowest method on the DAG; thin wrapper over already-built `cintx`; basis indexing conventions (Pitfall 17) and Boys (Pitfall 18) live here but are mitigated by going through cintx exhaustively | `pyscf-gto`; `pyscf-kernels::int1e/int2e` plumbing through cintx; `Mole::build` + 5 atom-input forms + 11 basis-input forms + ECP + dumps/loads + set_geom_; `mol.intor()` thin wrapper; eval_gto for grids | 8 (F-order layout), 17 (basis indexing), 18 (Boys via cintx) |
| 2 | **scf** | First end-to-end energy ("works on H2O"); fork point — after this DFT/MP2 run in parallel; sets DIIS + canonicalization conventions everything else reuses; chkfile schema is locked here | `pyscf-scf` with RHF/UHF/GHF + C-DIIS + level shift + init_guess (minao/atom/1e/chkfile/dm0); all `get_*` overrideable hooks; `analyze`/`mulliken_pop`; chkfile save/load; `to_uhf/to_rks/...` cross-module dispatch helpers; `as_scanner`; sign canonicalization helper in `pyscf-core::lib::canonicalize_signs` | 4 (eigenvector sign), 9 (DIIS path drift), 11 (chkfile schema), 19 (small-mol GPU dispatch heuristic) |
| 3 | **bindings (PyO3 contract)** | Must land before DFT (which has the most overrideable methods); subclass-dispatch and NumPy contiguity contracts cascade through every later method; better to lock conventions early on a small surface (RHF) than retrofit | `pyscf-py` skeleton with subclass-override dispatch via `slf.call_method1` for every overridable; `Python::detach` seam map; NumPy boundary `to_owned()`-on-non-standard-layout policy; `pyo3::sync::GILOnceCell` in lieu of `lazy_static`; abi3-py310 wheel builds; `python3.13t` CI; `python/pyscf/__init__.py` shim preserving `PYSCF_EXT_PATH`; one PyO3 cdylib with submodules `_native.{gto,scf}`; abi3audit in CI | 5 (NumPy zero-copy), 6 (GIL deadlock), 7 (subclass override) |
| 4 | **dft** | Largest single phase; integrates libxc_rs + xcfun_rs; RKS + UKS + Becke grids + range-separated hybrids + VV10 NLC + DF-DFT; the `bindings` layer's overrideable surface really gets exercised here; cubecl WGPU f64 risk peaks here | `pyscf-dft` with RKS/UKS; xc-string parser parity table from `pyscf/dft/libxc.py`; `Grids` ported from `gen_grid.py` byte-for-byte (atomic radii, Lebedev tables, prune scheme); range-separated `int2e_lr/sr_*` via cintx; VV10; DF-DFT path | 10 (DFT grid weighting), revalidates 7 (subclass override), 3 (cubecl f64) |
| 5 | **mp2** | First post-SCF; canonical AO→MO transformation kernel; CCSD's prerequisite (CCSD imports MP2 helpers); mostly mechanical translation if gto/scf are right | `pyscf-mp2` with RMP2/UMP2/DF-MP2; `pyscf-kernels::ao2mo`; frozen-core (int/list/auto/window); `make_rdm1/rdm2`; `as_scanner` | (mostly clean) |
| 6 | **ccsd** | Heaviest correlated method; XL phase; depends on MP2; memory architecture is fundamental here, not retrofittable | `pyscf-ccsd` with RCCSD/UCCSD; T1/T2 amplitude solver + amplitude-DIIS; `solve_lambda`; `make_rdm1/rdm2`; AO-direct; DF-CCSD; T1/D1/D2 diagnostics; tensor-arena/scratchpad pattern in `pyscf-runtime`; `PYSCF_MAX_MEMORY` pre-flight | 20 (CCSD memory thrash), 21 (scope creep — CCSD(T) pressure point) |
| 7 | **grad** | Depends on every method it differentiates; CPHF/CPKS solver lands here; CCSD-grad needs Λ-equations from `ccsd` phase; mostly orchestration over established kernels | `pyscf-grad` with RHF/UHF/RKS (with `grid_response=True`)/UKS/MP2/CCSD; ECP gradients; atom-list subsetting; finite-difference verification mode | (mostly clean; revalidates F-order layout) |
| 8 | **geomopt** | Small; depends only on grad; native Rust BFGS+RFO with argmin; the geomeTRIC/berny shims preserve drop-in compat without a Python optimizer dep | `pyscf-geomopt`; redundant internal coordinates (Wilson B-matrix); RFO with negative-eigenvalue tracking; `pyscf.geomopt.{optimize,geometric_solver.optimize,berny_solver.optimize}` shims; HDF5 checkpoint of optimizer state | (mostly clean) |
| 9 | **GPU backends enable** | Defer until all CPU paths converged on the oracle; enables cuda/wgpu/rocm features and runs regression suite per backend | `pyscf-kernels` features `cuda/wgpu/rocm/metal` enabled; per-backend regression suite; small-molecule auto-dispatch heuristic (CPU when nao<200); cubecl autotune cache shipped at `CUBECL_CACHE_DIR`; benchmark suite proves 2–5× claim | 3 (cubecl WGPU f64), 19 (small-mol launch overhead) |
| 10 | **oracle hardening + drop-in audit** | Test isolation, fixture organization, real-world-script pass-through; the "looks done but isn't" checklist | Subprocess-per-fixture oracle pattern; `oracle_check!(method, tol)` macro; nightly per-basis bit-exact test (every basis name PySCF knows); curated upstream PySCF unit tests run against pyscf-rs as import target with ≥80% pass rate | 16 (test-oracle global state) |
| 11 | **distribution / wheel** | abi3-py310 wheels for Linux/macOS/Windows × x86_64 + macOS aarch64; per-backend extras to dodge PyPI 60 MB limit; `pyscf` import-shim package so `import pyscf` Just Works on user machines | `pyscf_rs[cuda]`, `pyscf_rs[rocm]`, `pyscf_rs[wgpu]` extras; manylinux_2_28; `auditwheel show` clean; `pip install pyscf-rs && python -c "from pyscf import gto, scf; ..."` smoke on fresh container; PyPI size exemption requested pre-emptively if needed | 13 (wheel size / CUDA distribution) |

### Phase Ordering Rationale

- **Architecture DAG dictates W0–W6**: the dependency graph in [ARCHITECTURE §4](./ARCHITECTURE.md#4-strict-dag-dependency-graph) forces `core → runtime → kernels → gto → scf → {dft, mp2} → ccsd → grad → geomopt → py`. Phases 0–8 follow this 1:1.
- **Bindings inserted between scf and dft**: the subclass-override and NumPy contracts must lock before DFT (largest overrideable surface) but after enough Rust exists to test against (RHF). PyO3 cdylib skeleton in phase 3 is then incrementally extended in 4–8.
- **GPU and oracle/wheel come last**: mirrors xcfun_rs's "GPU after substrate solid" sequencing. Validates that every CPU path is correct before introducing backend-specific drift; oracle hardening surfaces the cross-platform issues (libm, BLAS vendor) that only manifest on wider CI matrices.
- **Pitfall phase-map alignment**: every SHOWSTOPPER pitfall is addressed in phase 0–3; every MAJOR is addressed by the phase that owns its primary surface. Roadmapper can cross-reference [PITFALLS Pitfall-to-Phase Mapping](./PITFALLS.md#pitfall-to-phase-mapping) directly.
- **CCSD(T) pressure is documented but deferred**: scope-creep lint enforces it; v1.x P1 entry on the roadmap signals the deferral is intentional.

### Research Flags

Phases needing dedicated `/gsd-research-phase` before execution:

- **Phase 0 (infra)** — research-light, design-heavy; most calls already made in this research, but the FMA/reduction/panic policy needs concrete encoding (LLVM-IR grep tooling, named-arithmetic primitives, `[profile.release-oracle]` exact flags).
- **Phase 2 (scf)** — DIIS implementation, eigenvector canonicalization, chkfile schema all need PySCF source-level deep dive (`pyscf/scf/diis.py`, `pyscf/lib/chkfile.py`, `pyscf/scf/chkfile.py` byte-for-byte).
- **Phase 3 (bindings)** — overrideable-method audit (`grep -rn "def get_jk\|def get_veff\|..." pyscf/`), GIL release seam map, NumPy contiguity contract, abi3 surface boundary.
- **Phase 4 (dft)** — `gen_grid.py` byte-for-byte port, libxc/xcfun string-parser semantics edge cases (especially XC_ALIAS), VV10 stability, range-separated hybrid omega plumbing.
- **Phase 6 (ccsd)** — T2 contraction memory scheduling, DIIS+iterative_damping interplay, scratchpad/arena pattern in `pyscf-runtime`, DF-CCSD streaming.
- **Phase 7 (grad)** — Λ-equation conditioning, response-density-to-AO transformation, grid-response in DFT grad.
- **Phase 9 (GPU enable)** — cubecl per-backend f64 smoke matrix, autotune cache strategy, small-molecule dispatch heuristic.
- **Phase 10 (oracle hardening)** — test-isolation pattern (subprocess-per-fixture vs persistent worker), nightly-vs-pre-merge split, fixture corpus.
- **Phase 11 (distribution)** — wheel split strategy, PyPI size exemption process, system-lib link strategy for libhdf5.

Phases unlikely to need extra research (well-trodden):

- **Phase 1 (gto)** — cintx already does the heavy lifting; pyscf-rs gto is a thin wrapper.
- **Phase 5 (mp2)** — once gto/scf are right, MP2 is a tensor contraction with standard patterns.
- **Phase 8 (geomopt)** — `argmin` BFGS + redundant-internals BFGS+RFO is textbook; trajectory-match against geomeTRIC defaults is mechanical.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | Every version verified against crates.io 2026-05-09; sibling-crate Cargo.toml audit is direct evidence; only MEDIUM items are h5py↔hdf5-metno chkfile round-trip (needs empirical validation in `scf` phase) and Boys-function strategy (delegated to cintx, fallback to in-house only if cintx gaps appear). [STACK §2.10](./STACK.md#210-versions-are-current--verification-trail) |
| Features | **HIGH** | API names and signatures verified directly from in-tree `pyscf/*.py` source; LOC counts are exact `wc -l`; only LOW item is "30–40% of CCSD users want CCSD(T)" estimate. [FEATURES §10](./FEATURES.md#10-sources) |
| Architecture | **HIGH** | Three sibling-crate Cargo.toml audits with line citations; cintx/xcfun_rs proof-of-existence for the layered pattern; only design-decision LOW points are exact `Eri8 { … }` storage type (will firm up in `gto` phase) and tensor-arena interface (will firm up in `ccsd` phase). [ARCHITECTURE §1](./ARCHITECTURE.md#1-sibling-crate-pattern-audit-observed-not-assumed) |
| Pitfalls | **HIGH** for verified items (PyO3/cubecl/PySCF tracker links, sibling-crate evidence); **MEDIUM** for extrapolations from cintx/xcfun_rs experience; LOW items flagged inline. 21 pitfalls catalogued; phase-map is complete. [PITFALLS top](./PITFALLS.md) |

**Overall confidence: HIGH.** This is a well-bounded rewrite of a known-working system; sibling crates have already proved the cubecl/PyO3 pattern works; the upstream source is in-tree and the oracle harness is feasible.

### Gaps to Address

These are not blockers but the roadmapper should mark them for validation during planning/execution:

- **chkfile h5py round-trip** ([STACK §2.4](./STACK.md#24-hdf5--chkfile-io), [PITFALLS Pitfall 11](./PITFALLS.md#pitfall-11)): hdf5-metno reads files h5py wrote — verify on a corpus of real PySCF chkfiles in an early `scf`-phase smoke. Originally HIGH-confidence in MEDIUM area; needs empirical seal.
- **cubecl WGPU f64** ([STACK §2.2](./STACK.md#22-cubecl-ecosystem), [PITFALLS Pitfall 3](./PITFALLS.md#pitfall-3)): open issues #1316/#1317 mean WGPU on consumer GPUs may be unusable. `infra`-phase smoke test on every backend with f64 elementwise add; gate `wgpu` feature on `shader-f64` Vulkan extension; document degraded support honestly. May force per-backend test gating in the GPU-enable phase.
- **`faer-ext 0.7.1` ↔ `faer 0.24.0` compat** ([STACK §5](./STACK.md#5-version-compatibility)): faer-ext last published against faer 0.24+. Verify it builds against 0.24 in `infra`-phase; if not, either bump faer-ext upstream or drop the dependency and round-trip via `Vec<f64>`.
- **CCSD(T) v1.x deferral pressure**: 30–40% of CCSD users want it ([FEATURES §2.5 Anti-features note](./FEATURES.md#25-cc-coupled-cluster-singles-doubles)). Roadmap must include explicit v1.x P1 entry to make the deferral visible; expect a feature request to land within weeks of v1 release.
- **`hdf5-rs` vs `hdf5-metno` decision**: PITFALLS Pitfall 11 originally argued for `hdf5-rs` (the system-lib binding) over `hdf5-metno`, but STACK §2.4 chose `hdf5-metno` with `hdf5-sys/static` for the bundled-libhdf5 install story. The decision is `hdf5-metno`; the chkfile round-trip oracle test is the empirical seal. Note this resolution explicitly in `infra`-phase docs to prevent re-litigation.
- **Native geomopt vs wrap-geomeTRIC trajectory parity**: native Rust BFGS+RFO is the chosen path, but bit-stable trajectory matching against geomeTRIC defaults is a research item for `geomopt` phase. Document failures (where Rust trajectory diverges from geomeTRIC trajectory) as chemical-accuracy not bit-exact and widen tolerance accordingly.
- **PyPI wheel size for bundled GPU backends**: 60 MB ceiling vs cubecl-bundled CUDA/HIP/WGPU. Per-backend extras (`pyscf_rs[cuda]`, …) is the planned mitigation, but pre-emptive PyPI exemption request in `distribution` phase is recommended.

---

## Sources

### Primary (HIGH confidence — direct artifact inspection)

- `~/Documents/workspace/cintx/Cargo.toml` and `crates/cintx-{cubecl,runtime,oracle,rs,capi,compat,core,ops}/Cargo.toml` — sibling 8-crate horizontal-layered façade pattern.
- `~/Documents/workspace/xcfun_rs/Cargo.toml` and `crates/xcfun-{kernels,gpu,eval,rs,capi,py,core,ad}/Cargo.toml` — sibling pattern with PyO3 layer; `xcfun-py` is the canonical PyO3 wiring.
- `~/Documents/workspace/libxc_rs/Cargo.toml` — flat-kernels pattern (rejected for pyscf_rs).
- `~/Documents/workspace/pyscf_rs/pyscf/**/*.py` (in-tree upstream) — every API name and signature; LOC counts exact.
- `~/Documents/workspace/pyscf_rs/pyscf/examples/{scf,dft,mp,cc,grad,geomopt,gto}/*.py` — top-20-idiom corpus.
- `~/Documents/workspace/pyscf_rs/.planning/PROJECT.md` — locked scope, decisions, anti-features.
- `~/Documents/workspace/pyscf_rs/.planning/codebase/{ARCHITECTURE,STACK,STRUCTURE,CONCERNS,CONVENTIONS,INTEGRATIONS,TESTING}.md` — upstream PySCF map.
- crates.io JSON API queried 2026-05-09 — every Rust dep version.

### Secondary (HIGH confidence — official docs)

- https://docs.rs/cubecl/0.10.0/cubecl/ — `Runtime` trait, `#[cube]`, `CubeLaunch`, autotune availability.
- https://github.com/tracel-ai/cubecl + issues #1316/#1317/#1318 — alpha-stability disclaimer; f64 SPIR-V holes; release cadence.
- https://docs.rs/faer/0.24.0/faer/ — pure-Rust BLAS surface; complex `c64`.
- https://pyo3.rs/v0.28.3/ — current PyO3 docs; abi3-py310 mechanism.
- https://github.com/PyO3/pyo3 discussions #3045 (tokio + GIL deadlock), #4738 (Python 3.13 free-threaded), #4164 (subclass-override) and issues #492 (panic-across-FFI), #947 (subclass __new__).
- https://github.com/PyO3/rust-numpy issue #114 — slicing + `to_pyarray` non-contiguous bug.
- https://github.com/numpy/numpy issue #14627 — wrong stride/contiguous flag.
- https://www.maturin.rs/distribution.html — wheel building, abi3 support, manylinux strategy.
- https://github.com/metno/hdf5-rust — fork rationale; `hdf5-sys/static` feature.
- https://github.com/pyscf/pyscf issues #1015 (macOS-arm64), #1102/#1138 (threading), #1196 (sign convention), #1935 (ERI sign).
- https://github.com/gpuweb/gpuweb issue #2805 — WebGPU f64 status.

### Tertiary (MEDIUM confidence — community/inference)

- "30–40% of CCSD users want CCSD(T)" — arXiv-search inference, flagged in [FEATURES §2.5](./FEATURES.md#25-cc-coupled-cluster-singles-doubles).
- h5py↔hdf5-metno round-trip robustness — needs empirical validation in `scf` phase smoke.
- CCSD(T) demand pressure timeline ("within weeks of release") — extrapolation from sibling-project release patterns.
- `faer-ext 0.7.1` working with `faer 0.24.0` — needs build verification in `infra`.

---

*Research synthesized: 2026-05-09*
*Ready for roadmap: yes*
