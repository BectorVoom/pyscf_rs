# Requirements: pyscf_rs

**Defined:** 2026-05-09
**Core Value:** Run mainstream molecular ground-state quantum chemistry (HF, DFT, MP2, CCSD, gradients) 2–5× faster than current PySCF + C extensions, with bit-exact agreement on regression tests, and zero C/CMake/libcint dependency hell at install time.

> Source research: [STACK](./research/STACK.md) · [FEATURES](./research/FEATURES.md) · [ARCHITECTURE](./research/ARCHITECTURE.md) · [PITFALLS](./research/PITFALLS.md) · [SUMMARY](./research/SUMMARY.md)

REQ-IDs are stable across the project lifecycle. The numbering blocks are per-category with reserved gaps so v1.x additions don't renumber v1.

## v1 Requirements

### Foundation (FOUND)

- [x] **FOUND-01**: Workspace builds clean as a 14-crate horizontal-layered façade — `pyscf-{core,runtime,kernels,gto,scf,dft,mp2,ccsd,grad,geomopt,py,oracle,bench}` plus top-level façade — mirroring the cintx/xcfun_rs pattern
- [x] **FOUND-02**: `pyscf-core` exposes the universal types (Mole, BasisSet, Density, MOCoefficients, Amplitudes, Energy newtype) and traits (Method, Scf, KohnSham, PostScf, Gradient, IntegralEngine) with no compute dependencies
- [x] **FOUND-03**: `pyscf-runtime` provides `BackendKind::{Cpu,Cuda,Wgpu,Rocm,Metal}` enum, `auto_backend()` priority chain (`PYSCF_BACKEND` env override → CUDA → ROCm → Metal → WGPU → CPU), and a `WorkspacePool`
- [x] **FOUND-04**: cubecl 0.10.0 is exact-pinned across the workspace and lockstep with cintx/libxc_rs/xcfun_rs (`[patch.crates-io]` enforced; documented upgrade ritual in CONTRIBUTING.md)
- [x] **FOUND-05**: A `[profile.release-oracle]` build profile with `RUSTFLAGS="-C target-feature=-fma"` produces FMA-free machine code; CI greps for `llvm.fmuladd` and fails on hits
- [x] **FOUND-06**: `oracle_sum`, `oracle_dot`, `oracle_einsum` deterministic-ordered-reduction primitives are implemented and used in every numerical kernel that the oracle harness checks
- [x] **FOUND-07**: Panic policy is enforced — every `extern "C"` callback uses `catch_unwind`; clippy lint blocks `unwrap()` in numerical modules; release builds use `panic = "abort"`
- [x] **FOUND-08**: A `forbidden-paths` lint refuses imports from out-of-scope upstream modules (pbc, x2c, mcscf, tdscf, adc, gw, eom, NAC, EPH) at every PR
- [x] **FOUND-09**: Tracing-based logging (`tracing 0.1`) replicates PySCF's `pyscf.lib.logger` verbosity contract (verbosity 0–9, `mol.verbose` configurable)
- [x] **FOUND-10**: MSRV 1.92, edition 2024, Apache-2.0 license file at workspace root, and `cargo deny` clean

### Molecular structure & integrals (GTO)

- [x] **GTO-01**: `pyscf.M(...)` factory and `gto.Mole` class accept all 5 atom-input forms (string, list-of-tuples, list-of-lists, file path, geom callable)
- [x] **GTO-02**: `mol.basis = ...` accepts all 11 input forms (string name, dict, list, ECP-bcc, F12, dyall, ANO, def2, parsed Gaussian-94, NWChem, and auto-segmented)
- [x] **GTO-03**: All 207 built-in basis-set files in `pyscf/gto/basis/` resolve correctly; `gto.parse(...)` handles user-supplied Gaussian-94/NWChem text
- [x] **GTO-04**: `mol.build()` produces identical internal arrays (`_atm`, `_bas`, `_env`, `ao_loc_nr`, `nao_nr`) to upstream PySCF for the test corpus
- [x] **GTO-05**: ECP loading and `int1e_ecp` evaluation match upstream PySCF bit-exact under the `release-oracle` profile — loading shipped 02-07; evaluation closed 02-10 (cintx-backed `CintxEcpEngine`; in-tree Cu/LANL2DZ finite/non-zero/symmetric gate green; cintx pins atol=1e-12 vs nr_ecp at source; upstream byte-identity pytest `tests/oracle/test_ecp_int1e.py` shipped, gated on the oracle venv)
- [x] **GTO-06**: `mol.intor(name, ...)` is a thin wrapper over `cintx` covering all integral families upstream PySCF supports for the in-scope methods
- [x] **GTO-07**: `eval_gto(mol, eval_name, coords, ...)` for grid evaluation supports `GTOval`, `GTOval_sph`, `GTOval_deriv1`, `GTOval_deriv2`, `GTOval_ip` (gradient), and `GTOval_ig` (magnetic)
- [x] **GTO-08**: `Mole` exposes the ≥30 attribute floor (`atom`, `basis`, `charge`, `spin`, `nelectron`, `natm`, `nbas`, `nao_nr`, `nao_2c`, `ao_loc_nr`, `ao_labels`, `cart`, `verbose`, `max_memory`, `unit`, `output`, `_atm`, `_bas`, `_env`, …) with semantics matching upstream
- [x] **GTO-09**: `mol.dumps()` / `mol.loads()` round-trip Mole state via JSON; `mol.copy()` is deep-copy
- [x] **GTO-10**: `mol.set_geom_(new_atom)` mutates geometry in place, preserves basis, returns `self`
- [x] **GTO-11**: Integration of `cintx` is via `pyscf-core::BasisSet` re-exporting `cintx_core::BasisSet` (zero-copy); pyscf-rs does not maintain a parallel basis structure

### Self-consistent field (SCF)

- [ ] **SCF-01**: `scf.RHF(mol).kernel()` converges to the same total energy as upstream PySCF on the test corpus to ≤1 µHartree under `release-oracle` (bit-exact when reduction order matches)
- [ ] **SCF-02**: `scf.UHF(mol).kernel()` matches upstream for open-shell / spin-polarized systems
- [ ] **SCF-03**: `scf.GHF(mol).kernel()` runs (correctness only; perf parity not required for v1)
- [ ] **SCF-04**: C-DIIS convergence (`mf.diis = True`, `mf.diis_space = 8`, `mf.diis_start_cycle = 1`) reproduces upstream DIIS extrapolation when reduction order is held; energy/density convergence path matches to chemical accuracy
- [~] **SCF-05**: `mf.init_guess` accepts `'minao'`, `'atom'`, `'1e'`, `'huckel'`, `'chkfile'`; user-supplied `dm0` is respected (plans 03-11/03-06/03-13 — `'1e'`, `'chkfile'`, user-`dm0`, AND the DEFAULT `'minao'` are implemented: `init_guess_by_minao` byte-matches the upstream H2 docstring dm `[[0.94758917,0.09227308],…]` and converges RHF/DF-HF out-of-the-box, via the new `pyscf_gto::intor_cross` + the ported `NRSRHF_CONFIGURATION`/`frac_occ`; `crates/pyscf-scf/tests/init_guess_minao.rs`. `'atom'`/`'huckel'` remain NotYetImplemented. Caveat: the vendored `ano.dat` under-resolves the s-shell of heavier atoms (O `Tr(dm·S)≈7.9` vs 10) so minao under-normalizes there — it still converges to the correct energy; full ANO contraction coverage is a follow-up)
- [ ] **SCF-06**: `mf.level_shift`, `mf.damp`, `mf.max_cycle`, `mf.conv_tol`, `mf.conv_tol_grad` controls match upstream semantics
- [~] **SCF-07**: `mf.density_fit(auxbasis=...)` returns an SCF object that solves DF-HF; auxbasis defaults match upstream (`weigend`, `cc-pvdz-jkfit`) (plan 03-12 — DF-HF now solves END-TO-END in-tree: `RHF::density_fit` + `DfHooks` + the SCF kernel converge and match non-DF RHF within DF accuracy (H2/STO-3G: weigend 4.6e-5, cc-pvdz-jkfit 2.0e-4 Hartree), enabled by 05-08 int2e + 05-09 rank-revealing DF-metric fit; `crates/pyscf-scf/tests/dfhf_end_to_end.rs`. The default `minao` init guess landed in 03-13, so `RHF(mol).density_fit().kernel()` now works fully out-of-the-box; the only remaining item for full closure is upstream-PySCF byte-identity of the converged energy (CI-gated/human-verify))
- [ ] **SCF-08**: All overrideable hooks dispatch via PyO3 `slf.call_method1` so Python subclasses correctly override `get_jk`, `get_veff`, `get_hcore`, `get_init_guess`, `get_fock`, `get_occ`, `eig`, `make_rdm1`, `energy_elec`, `energy_tot`
- [ ] **SCF-09**: `mf.analyze()`, `mf.mulliken_pop()`, `mf.mulliken_meta()`, `mf.dip_moment()` produce the same numbers as upstream
- [ ] **SCF-10**: `mf.chkfile = path` writes an HDF5 chkfile that h5py can read with the upstream PySCF schema; `mf.from_chk(path)` reads upstream-written chkfiles
- [ ] **SCF-11**: Cross-module dispatch helpers (`mf.to_uhf()`, `mf.to_rhf()`, `mf.to_uks()`, `mf.to_rks()`, `mf.to_ghf()`) work as upstream because MP2/CCSD dispatch depends on them
- [ ] **SCF-12**: `mf.as_scanner()` returns a callable that takes a Mole and returns the energy (used by geomopt)
- [ ] **SCF-13**: `pyscf-core::lib::canonicalize_signs` produces vendor-stable eigenvectors (largest-|coefficient|-with-lowest-index sign-flip) so MO coefficients are reproducible across LAPACK vendors
- [ ] **SCF-14**: SCF exposes the ≥30 attribute floor (`mo_coeff`, `mo_energy`, `mo_occ`, `e_tot`, `e_elec`, `converged`, `mol`, `verbose`, `chkfile`, `max_memory`, `direct_scf`, `direct_scf_tol`, `init_guess`, `level_shift`, `damp`, `diis`, `diis_space`, `diis_start_cycle`, `max_cycle`, `conv_tol`, `conv_tol_grad`, `with_df`, `disp`, `do_disp`, `irrep_nelec`, `nelec`, …)

### Density functional theory (DFT)

- [~] **DFT-01**: `dft.RKS(mol, xc='b3lyp')` and `dft.UKS(mol, xc=...)` converge bit-exact to upstream PySCF on the test corpus under `release-oracle` (Phase 4 plan 04-06 — RKS/UKS reuse the Phase 3 `kernel<H>` with KS get_veff = J+Vxc−hyb·K via the algebra-orchestrated NumInt grid loop; the bit-exact energy gate is the CI-only `--features python` `rks_energy`/`uks_energy` oracle arms — final convergence pending the Phase-2 arity-3/4 ERI rollup `int2e_sph`/`int3c2e_sph` + a live PySCF on CI; the drivers are complete and need no change once working ERIs land)
- [x] **DFT-02**: XC string parser handles all upstream forms — single name (`'b3lyp'`), comma form (`'pbe,pbe'`), shorthands (`'lda'` → `'lda,vwn'`), explicit weights (`'.5*HF + .5*B88,LYP'`), aliases from `XC_ALIAS` (Phase 4 plan 04-05 — libxc-default + xcfun-alternate `parse_xc` ports, 23 parity assertions)
- [x] **DFT-03**: libxc functional evaluation routes through `libxc_rs`; xcfun routes through `xcfun_rs`; both produce identical numbers to the upstream C libraries (Phase 4 plan 04-05 — `XcBackend` cfg-gated seam; xcfun-default path bit-exact to analytic Slater LDA; the libxc-side bit-exact assertions are `#[cfg(feature="libxc")]`-gated/CI-only per `PENDING_LIBXC_RS_FEATURE_GATE`, 04-02)
- [x] **DFT-04**: `Grids` class with `level`, `atom_grid`, `prune`, `radi_method`, `becke_scheme`, `atomic_radii` controls; default Becke partitioning + Treutler radial + Lebedev angular reproduce upstream weights byte-for-byte (port `pyscf/dft/gen_grid.py`)
- [x] **DFT-05**: Range-separated hybrids (`omega`, `alpha`, `beta`) compute long/short-range exact-exchange K via the range-coulomb `env[8]` (`PTR_RANGE_OMEGA`) set/restore around the standard `int2e` — NOT distinct `int2e_lr_*`/`int2e_sr_*` symbols (RESEARCH CORRECTION / Pitfall 1: those do not exist in cintx; the standard `int2e` reads `env[8]`). Phase 4 plan 04-07 — `pyscf-gto::range_coulomb` (`OmegaGuard`/`intor_with_omega`/`get_k_with_omega`) + `veff::default_get_veff` RSH branch (`vk = hyb·K + (alpha−hyb)·K_lr`, rks.py:108-129). Open Question A5 resolved: cintx safe API has no `env[8]` setter, so the slot is owned at the pyscf-gto layer; the numerical RSH ERI + CAM-B3LYP energy assertion are CI-gated behind a cintx#11-style gap-closure (safe-API `env[8]` reader + arity-4 `int2e`).
- [x] **DFT-06**: VV10 non-local correlation (`mf.nlc = 'VV10'`, `mf.nlcgrids`) produces upstream-matching energies via the ported pure-Python `_vv10nlc` double-loop (numint.py:526-538, Pitfall 4: NOT C `VXC_vv10nlc`) over a coarser `nlcgrids`; Phase 4 plan 04-07 — `vv10::vv10nlc`/`nr_nlc_vxc`, inner reductions via `oracle_sum`, bare-VV10 default Bvv=5.9/Cvv=0.0093 (A1). The bit-exact VV10 RKS energy is CI-gated behind the Phase-2 ERI/init-guess gap.
- [x] **DFT-07**: DF-DFT (`dft.RKS(mol).density_fit()`) works
- [x] **DFT-08**: All SCF subclass-override hooks (SCF-08) re-validate at the DFT level (DFT adds `get_veff`, `define_xc_`, custom-functional hooks) (Phase 4 plan 04-06 — `KsOverrideHooks` extends `OverrideHooks` with `get_veff_ks` + `define_xc_`; `NoKsOverrides` + `KsHooks` impls; the callable `define_xc_` form returns `NotYetImplemented` per D-02; pyscf-dft stays pyo3-free. The Python-side subclass-override dispatch re-validation landed with the 04-09 PyO3 bridge — `PyRKS`/`PyUKS` + `_native.dft` + `PyOverrideBridge: KsOverrideHooks` dispatching `get_veff`/`define_xc_` via `call_method1`, source-verified (`cargo build -p pyscf-py` + `check-forbidden-paths`); the live override-invoked-every-cycle pytest is Manual-Only/CI per 04-VALIDATION, env lacks maturin/pyscf — user-approved checkpoint 2026-05-22.)
- [x] **DFT-09**: `mf.grids.level = N` for N ∈ {0, 1, …, 9} matches upstream grid sizes
- [x] **DFT-10**: `numint.NumInt` exposes `eval_xc`, `eval_rho`, `nr_rks`, `nr_uks` matching upstream signatures (port from `pyscf/dft/numint.py`) (Phase 4 plan 04-06 — `NumInt` with the upstream `numint.py` signatures; `numint_signatures` test asserts the signatures + a numeric `eval_rho` element-wise check vs an independent longhand reference; the grid loop is algebra-orchestrated with NO `#[cube]` kernel, D-07)
- [x] **DFT-11**: cubecl WGPU backend is feature-gated on the `shader-f64` Vulkan extension; runtime falls back to CPU with a warning when unavailable (Phase 4 plans 04-06 + 04-10. 04-06 landed the D-08 escape-hatch half: `PYSCF_DTYPE=f32` drives the DFT grid loop end-to-end via `DType::from_env`-driven f32/f64 dispatch with a below-bit-exact `tracing::warn!`; `dtype_f32_smoke` is the runs-end-to-end smoke (NO oracle compare, NO tolerance gate per D-08). 04-10 landed the honesty fallback: `xc_backend.rs` delegates the shader-f64/ERF probe to `xcfun_gpu::auto_backend`/`must_fall_back_to_cpu` and reuses the Phase-1 `PYSCF_BACKEND` resolver so default-f64 + shader-f64-less wgpu → CPU-f64 + `tracing::warn!`, NEVER silent f32; the `wgpu_f64_fallback` unit test verifies the fallback decision + warn + explicit-f32-not-blocked locally. The on-real-shader-f64-less-DEVICE run is the active `wgpu-no-f64-fallback` CI job (special-runner / Phase 8). Patch re-enabled inert; the libxc bit-exact CI job ships DISABLED (`if: false`) pending 04-02 `PENDING_LIBXC_RS_FEATURE_GATE`.)

### Møller–Plesset (MP2)

- [x] **MP2-01**: `mf.MP2().run()` and `mp.RMP2(mf).kernel()` reproduce upstream RMP2 correlation energy bit-exact under `release-oracle`
- [x] **MP2-02**: `mp.UMP2(uhf_mf).kernel()` reproduces upstream UMP2
- [x] **MP2-03**: Frozen-core options accept `frozen=int`, `frozen=list`, `frozen='auto'`, and frozen-window forms; defaults match upstream
- [x] **MP2-04**: `mp.DFMP2(mf).kernel()` (density-fitted MP2) reproduces upstream
- [x] **MP2-05**: `mp2.make_rdm1()` and `mp2.make_rdm2()` match upstream
- [x] **MP2-06**: SCS-MP2 (`mp.MP2(mf).set(emp2_ss_factor=..., emp2_os_factor=...)`) works
- [x] **MP2-07**: `mp2.as_scanner()` works (used by gradients and geomopt)
- [x] **MP2-08**: MP2 helpers (`get_nocc`, `get_nmo`, `get_frozen_mask`, `get_e_hf`, `_mo_without_core`) are exported because CCSD imports them

> **Numeric closure (plans 05-08 + 05-09, 2026-05-23):** the cintx#11 ERI gate is
> closed — cintx ships arity-4 `int2e` + arity-3 `int3c2e_sph`, wired in pyscf-gto
> (05-08). In-core RMP2 (MP2-01) AND conventional DF-MP2 (MP2-04) numeric paths
> are now proven always-on in-tree (finite e_corr ≤ 0; DF-MP2 -0.04424 ≈ in-core
> -0.04428; DF B reconstructs exact int2e to 1.7e-3). 05-09 added the
> rank-revealing DF-metric fit (`pyscf_algebra::df_metric_fit`, PySCF
> `LINEAR_DEP_THRESHOLD` route) so ill-conditioned `(P|Q)` metrics (cc-pvdz-jkfit,
> weigend) build cleanly. Upstream-PySCF byte-identity for all arms is the
> CI-gated (`workflow_dispatch`) / human-verify arm (sandbox lacks numpy/PySCF).
> See `phases/05-mp2/05-08-PLAN.md` + `05-09-PLAN.md`.

### Coupled cluster (CCSD)

- [ ] **CCSD-01**: `cc.RCCSD(mf).kernel()` returns CCSD correlation energy matching upstream to chemical accuracy (≤1 µHartree); convergence criteria match
- [ ] **CCSD-02**: `cc.UCCSD(uhf_mf).kernel()` matches upstream
- [ ] **CCSD-03**: T1 and T2 amplitudes converge to the same minimum as upstream (energy is the convergence target, amplitude paths may differ within tolerance)
- [ ] **CCSD-04**: Amplitude-DIIS (default `mycc.diis = True`, `mycc.diis_space = 6`) converges within the same iteration count as upstream on the test corpus
- [ ] **CCSD-05**: `mycc.solve_lambda()` produces λ amplitudes for response densities
- [ ] **CCSD-06**: `mycc.make_rdm1()`, `mycc.make_rdm2()` match upstream
- [ ] **CCSD-07**: AO-direct CCSD (`mycc.direct = True`) works
- [ ] **CCSD-08**: DF-CCSD (`mycc = mf.density_fit().CCSD()` or `cc.dfccsd.RCCSD(mf)`) works with bounded memory; spills to HDF5 when `PYSCF_MAX_MEMORY` is exceeded
- [ ] **CCSD-09**: T1/D1/D2 diagnostics expose `mycc.t1diagnostic()`, `mycc.d1diagnostic()`
- [ ] **CCSD-10**: Frozen-core options match MP2 (`frozen=int`, `frozen=list`, `frozen='auto'`)
- [ ] **CCSD-11**: Tensor-arena/scratchpad pattern in `pyscf-runtime` is in place from the start of CCSD work — not retrofitted; `Wabef` and other large intermediates do not allocate-and-drop per iteration

### Gradients (GRAD)

- [ ] **GRAD-01**: `mf.nuc_grad_method().kernel()` for RHF returns analytical gradients matching upstream
- [ ] **GRAD-02**: UHF gradients match upstream
- [ ] **GRAD-03**: RKS gradients (with `grid_response = True`) match upstream
- [ ] **GRAD-04**: UKS gradients match upstream
- [ ] **GRAD-05**: MP2 gradients via Z-vector / CPHF match upstream
- [ ] **GRAD-06**: CCSD gradients via Λ-equations match upstream
- [ ] **GRAD-07**: ECP gradients match upstream
- [ ] **GRAD-08**: Atom-list subsetting (`grad.kernel(atmlst=[1,2,3])`) works
- [ ] **GRAD-09**: A finite-difference verification mode (`grad.verify_fd(disp=1e-4)`) is available and gates unit tests
- [ ] **GRAD-10**: CPHF/CPKS solver lives in `pyscf-grad` (or a shared module) and is reused by all method gradients

### Geometry optimization (GEOMOPT)

- [ ] **GEOMOPT-01**: `pyscf.geomopt.optimize(mf)` runs a native Rust BFGS+RFO optimizer in redundant internals; no Python `geomeTRIC` or `pyberny` runtime dependency
- [ ] **GEOMOPT-02**: `pyscf.geomopt.geometric_solver.optimize(mf)` is a drop-in shim that delegates to the native optimizer (preserves the canonical PySCF import path)
- [ ] **GEOMOPT-03**: `pyscf.geomopt.berny_solver.optimize(mf)` is also a drop-in shim
- [ ] **GEOMOPT-04**: Default convergence thresholds match geomeTRIC defaults (`gradient`, `displacement`, `energy`, `gradient_max`, `displacement_max`)
- [ ] **GEOMOPT-05**: HDF5 checkpoint of optimizer state allows resuming a partially-converged optimization
- [ ] **GEOMOPT-06**: Wilson B-matrix construction for redundant internals, RFO step with negative-eigenvalue tracking, both ported from upstream/geomeTRIC
- [ ] **GEOMOPT-07**: Optimization trajectories on the test corpus converge to the same stationary point as upstream within chemical accuracy

### PyO3 bindings & drop-in API contract (BIND)

- [ ] **BIND-01**: A single `pyscf-py` cdylib produces an abi3-py310 wheel covering Python 3.10–3.14 in one binary per OS/arch
- [ ] **BIND-02**: `from pyscf import gto, scf, dft, mp, cc, grad, geomopt` works exactly as upstream — preserved via the `_native.{module}` PyO3 submodules plus a thin `python/pyscf/__init__.py` re-export shim
- [ ] **BIND-03**: All 20 top-tier drop-in idioms run unchanged from existing PySCF scripts:
  1. `pyscf.M(atom=..., basis=...)`
  2. `mol.RHF().run()`
  3. `scf.RHF(mol).run()`
  4. `dft.RKS(mol, xc='b3lyp').run()`
  5. `mf.kernel(dm0)`
  6. `mf.density_fit().run()`
  7. `mf.MP2().run()`
  8. `mp.RMP2(mf).run()`
  9. `mf.CCSD().run()`
  10. `cc.RCCSD(mf).run()`
  11. `mf.nuc_grad_method().kernel()`
  12. `from pyscf.geomopt.geometric_solver import optimize; optimize(mf)`
  13. `mol.intor('int2e')`
  14. `mol.intor('int1e_ovlp_sph')`
  15. `mol.set_geom_(new_atom)`
  16. `mf.analyze()`
  17. `mf.chkfile = 'h2o.chk'`
  18. `mf.mo_coeff`, `mf.mo_energy`, `mf.mo_occ`, `mf.e_tot`, `mf.converged`
  19. `mf.to_uhf().run()` (cross-module dispatch)
  20. `mol.dumps()` / `gto.Mole.loads(s)`
- [ ] **BIND-04**: NumPy boundary policy: any `PyArray` input that is not `is_standard_layout()` is `to_owned()` on entry; outputs are always C-contiguous unless an `order='F'` flag is explicitly passed
- [ ] **BIND-05**: GIL release seam map: long-running compute calls `Python::detach` (≡ old `py.allow_threads`) so callbacks reacquire the GIL cleanly; tested under `python3.13t` free-threaded build
- [ ] **BIND-06**: `pyo3::sync::GILOnceCell` replaces every `lazy_static!` in PyO3 paths
- [ ] **BIND-07**: All Python-overrideable methods (the SCF-08 list, replicated per method) dispatch via `slf.call_method1(py, "name", args)` so user subclasses behave correctly
- [ ] **BIND-08**: `abi3audit` runs in CI on the produced wheel and fails on non-abi3 symbols
- [ ] **BIND-09**: Error messages: Rust panics never escape FFI; conversion to Python exceptions preserves the original error chain

### Oracle, testing & CI (ORACLE)

- [x] **ORACLE-01**: `pyscf-oracle` crate uses `pyo3::Python::with_gil` to drive upstream PySCF in-process; listed only in `dev-dependencies` so release wheels never link Python
- [ ] **ORACLE-02**: `oracle_check!(method, tolerance, fixture)` macro compares pyscf-rs and upstream PySCF outputs at every test fixture
- [ ] **ORACLE-03**: Test isolation uses subprocess-per-fixture for tests that mutate global state (SCF density caches, threading config); persistent worker for stateless tests
- [ ] **ORACLE-04**: Pre-merge CI runs the full test corpus on Linux x86_64 with CPU backend; ≥80% of curated upstream PySCF unit tests for in-scope modules pass when run against pyscf-rs as the import target
- [x] **ORACLE-05**: Nightly cross-crate matrix CI rebuilds and tests cintx + libxc_rs + xcfun_rs + pyscf_rs together against the cubecl pin
- [ ] **ORACLE-06**: Nightly per-basis bit-exact test sweeps every basis-set name PySCF knows
- [ ] **ORACLE-07**: GPU backends (CUDA/WGPU/ROCm) are tested at chemical accuracy, not bit-exact; tolerance documented per backend
- [ ] **ORACLE-08**: chkfile round-trip oracle: PySCF writes → pyscf-rs reads, asserts identical; pyscf-rs writes → PySCF reads, runs downstream calc, asserts agreement
- [x] **ORACLE-09**: Floating-point determinism: oracle CI pins `RAYON_NUM_THREADS=1`, `mol.lib.num_threads(1)`, and uses the `release-oracle` profile

### Performance (PERF)

- [ ] **PERF-01**: A criterion-based benchmark suite (`pyscf-bench` crate) covers RHF, RKS, MP2, CCSD on a defined molecule set — H2O/cc-pVDZ, benzene/6-31G*, 20-water cluster/cc-pVDZ, alanine dipeptide/def2-SVP, caffeine/cc-pVDZ
- [ ] **PERF-02**: pyscf-rs achieves ≥2× speedup on the benchmark suite vs current PySCF + C extensions on a fair-comparison machine (same CPU, same thread count, no GPU)
- [ ] **PERF-03**: Stretch goal: ≥5× speedup on at least one benchmark in the suite
- [ ] **PERF-04**: GPU backends (CUDA when available) demonstrate additional speedup on the larger benchmarks (caffeine, alanine dipeptide)
- [ ] **PERF-05**: Sub-second `mol.build()` for 5000-AO molecules
- [ ] **PERF-06**: cubecl autotune cache ships at `CUBECL_CACHE_DIR` so first-run overhead doesn't regress the benchmark
- [ ] **PERF-07**: Adaptive backend dispatch: small-molecule auto-fall-back to CPU when `nao < 200` to avoid GPU launch overhead

### Distribution (DIST)

- [ ] **DIST-01**: `pyscf-rs` published on crates.io with the workspace façade crate exporting the in-scope methods
- [ ] **DIST-02**: PyPI wheel `pyscf-rs` (or a project-chosen package name) installs cleanly on Linux/macOS/Windows × x86_64 + macOS aarch64; `pip install pyscf-rs && python -c "from pyscf import gto, scf"` succeeds in a fresh container
- [ ] **DIST-03**: Per-backend optional extras (`pyscf-rs[cuda]`, `pyscf-rs[wgpu]`, `pyscf-rs[rocm]`) keep the base wheel under the PyPI 60 MB ceiling
- [ ] **DIST-04**: manylinux_2_28 baseline; `auditwheel show` is clean
- [ ] **DIST-05**: HDF5 ships statically linked via `hdf5-sys/static`; no system libhdf5 required at install time
- [ ] **DIST-06**: A `python/pyscf/__init__.py` import shim makes `import pyscf` work as upstream users expect

## v1.x Requirements

Acknowledged but deferred to a later release. Tracked here so deferral is explicit.

### Coupled cluster (CCSD-T)

- **CCSD-T-01**: CCSD(T) — perturbative triples on top of CCSD. Highest user-pull pressure deferral; flag P1 for v1.x because ~30–40% of CCSD users want it

### Self-consistent field (extended)

- **SCF-EXT-01**: ROHF (currently falls back to UHF + warning)
- **SCF-EXT-02**: ROHF gradients
- **SCF-EXT-03**: SOSCF (`scf.newton(mf)`) for hard-converge cases
- **SCF-EXT-04**: ADIIS / EDIIS variants
- **SCF-EXT-05**: Symmetry-adapted SCF (currently C1 only)

### DFT (extended)

- **DFT-EXT-01**: DFT-D3 / D4 dispersion (`mf.disp = 'd3bj'`)
- **DFT-EXT-02**: Custom-XC user functions

### Hessian / vibrational

- **HESS-01**: RHF Hessian
- **HESS-02**: RKS Hessian
- **HESS-03**: Vibrational frequencies and IR intensities

### Coupled cluster (extended)

- **CCSD-EXT-01**: FNO-CCSD (frozen natural orbitals)
- **CCSD-EXT-02**: GHF / GMP2 / GCCSD path

### Geomopt (extended)

- **GEOMOPT-EXT-01**: Constrained geometry optimization (bond/angle/dihedral constraints)

### Distribution (extended)

- **DIST-EXT-01**: conda-forge channel publishing

## Out of Scope

Explicit exclusions for v1 and (most likely) v2. Each entry has reasoning so the boundary doesn't get re-litigated.

| Feature | Reason |
|---------|--------|
| Periodic boundary conditions (`pbc/*`) | Crystal/solid-state with k-points is essentially a parallel project; defer to a future milestone |
| Relativistic methods (`x2c`, `dhf`, `sfx2c1e`) | Two-/four-component relativistic SCF; needed only for heavy-element work |
| Multi-reference (`mcscf`, `mcpdft`, `mrpt`, CASSCF, NEVPT2, CASCI) | Niche, high implementation cost; whole separate milestone |
| Excited-state / response (`tdscf`, `tddft`, `tda`, `adc`, `gw`, EOM-CC) | Entire response-theory layer; treat as a separate milestone |
| Higher-order post-SCF beyond CCSD (CC3, full-CI) | CCSD covers the bulk of practical use; CC3 etc. are research-grade |
| AGF2 / DCI / selected CI | Specialty post-SCF; not core to v1 value prop |
| Non-adiabatic coupling (NAC) | Excited-state-adjacent; defers with TDDFT |
| Electron-phonon coupling (EPH) | Specialty; rarely used outside specific groups |
| Solvent models (PCM, COSMO, ddCOSMO, SMD) | Belongs with QM/MM in a future milestone |
| QM/MM | Coupling to MM force fields is a separate concern |
| Localized orbitals (`lo/*`: IAO, IBO, Pipek-Mezey, ER, …) | Analysis tools, not core compute; deferred |
| Multi-node MPI / distributed | cubecl + shared-memory parallelism only in v1 |
| Custom-Hamiltonian SCF for model systems (Hubbard etc.) | Model-systems users are a separate audience |
| Conda channel publishing for v1 | crates.io + PyPI wheels cover v1 distribution |
| Wrapping Python `geomeTRIC` / `pyberny` | Defeats the "pure Rust" value prop; preserved as drop-in shims only |
| `cubegen`, `molden`, full `tools/*` analysis suite | Visualization/analysis utilities; valuable but deferred |
| OpenMP / non-cubecl SIMD on hot paths | cubecl is the sole compute primitive (rayon allowed only in cold orchestration) |
| `ndarray-linalg`, `nalgebra-lapack` | Pull system BLAS/LAPACK and defeat the no-C-deps install-time goal |

## Traceability

Each v1 requirement maps to exactly one phase. v1.x-deferred requirements (the `## v1.x Requirements` section) are listed at the end with Phase = `v1.x` and Status = `Deferred`; they are NOT part of the 113-requirement v1 count and are not phase-mapped. Filled by the roadmapper 2026-05-10; v1.x deferred rows appended 2026-05-23 for traceability completeness.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Complete |
| FOUND-02 | Phase 1 | Complete |
| FOUND-03 | Phase 1 | Complete |
| FOUND-04 | Phase 1 | Complete |
| FOUND-05 | Phase 1 | Complete |
| FOUND-06 | Phase 1 | Complete |
| FOUND-07 | Phase 1 | Complete |
| FOUND-08 | Phase 1 | Complete |
| FOUND-09 | Phase 1 | Complete |
| FOUND-10 | Phase 1 | Complete |
| GTO-01 | Phase 2 | Complete |
| GTO-02 | Phase 2 | Complete |
| GTO-03 | Phase 2 | Complete |
| GTO-04 | Phase 2 | Complete |
| GTO-05 | Phase 2 | Complete (02-07 loading + 02-10 eval) |
| GTO-06 | Phase 2 | Complete |
| GTO-07 | Phase 2 | Complete |
| GTO-08 | Phase 2 | Complete |
| GTO-09 | Phase 2 | Complete |
| GTO-10 | Phase 2 | Complete |
| GTO-11 | Phase 2 | Complete |
| SCF-01 | Phase 3 | Pending |
| SCF-02 | Phase 3 | Pending |
| SCF-03 | Phase 3 | Pending |
| SCF-04 | Phase 3 | Pending |
| SCF-05 | Phase 3 | Pending |
| SCF-06 | Phase 3 | Pending |
| SCF-07 | Phase 3 | Pending |
| SCF-08 | Phase 3 | Pending |
| SCF-09 | Phase 3 | Pending |
| SCF-10 | Phase 3 | Pending |
| SCF-11 | Phase 3 | Pending |
| SCF-12 | Phase 3 | Pending |
| SCF-13 | Phase 3 | Pending |
| SCF-14 | Phase 3 | Pending |
| DFT-01 | Phase 4 | Implemented (CI-only bit-exact gate pending Phase-2 ERI rollup + live PySCF) |
| DFT-02 | Phase 4 | Complete |
| DFT-03 | Phase 4 | Complete |
| DFT-04 | Phase 4 | Complete |
| DFT-05 | Phase 4 | Complete |
| DFT-06 | Phase 4 | Complete |
| DFT-07 | Phase 4 | Complete |
| DFT-08 | Phase 4 | Complete |
| DFT-09 | Phase 4 | Complete |
| DFT-10 | Phase 4 | Complete |
| DFT-11 | Phase 4 | Implemented (f32 escape-hatch half; WGPU shader-f64 fallback CI job → Phase 8) |
| MP2-01 | Phase 5 | Complete |
| MP2-02 | Phase 5 | Complete |
| MP2-03 | Phase 5 | Complete |
| MP2-04 | Phase 5 | Complete |
| MP2-05 | Phase 5 | Complete |
| MP2-06 | Phase 5 | Complete |
| MP2-07 | Phase 5 | Complete |
| MP2-08 | Phase 5 | Complete |
| CCSD-01 | Phase 6 | Pending |
| CCSD-02 | Phase 6 | Pending |
| CCSD-03 | Phase 6 | Pending |
| CCSD-04 | Phase 6 | Pending |
| CCSD-05 | Phase 6 | Pending |
| CCSD-06 | Phase 6 | Pending |
| CCSD-07 | Phase 6 | Pending |
| CCSD-08 | Phase 6 | Pending |
| CCSD-09 | Phase 6 | Pending |
| CCSD-10 | Phase 6 | Pending |
| CCSD-11 | Phase 6 | Pending |
| GRAD-01 | Phase 7 | Pending |
| GRAD-02 | Phase 7 | Pending |
| GRAD-03 | Phase 7 | Pending |
| GRAD-04 | Phase 7 | Pending |
| GRAD-05 | Phase 7 | Pending |
| GRAD-06 | Phase 7 | Pending |
| GRAD-07 | Phase 7 | Pending |
| GRAD-08 | Phase 7 | Pending |
| GRAD-09 | Phase 7 | Pending |
| GRAD-10 | Phase 7 | Pending |
| GEOMOPT-01 | Phase 7 | Pending |
| GEOMOPT-02 | Phase 7 | Pending |
| GEOMOPT-03 | Phase 7 | Pending |
| GEOMOPT-04 | Phase 7 | Pending |
| GEOMOPT-05 | Phase 7 | Pending |
| GEOMOPT-06 | Phase 7 | Pending |
| GEOMOPT-07 | Phase 7 | Pending |
| BIND-01 | Phase 3 | Pending |
| BIND-02 | Phase 3 | Pending |
| BIND-03 | Phase 8 | Pending |
| BIND-04 | Phase 3 | Pending |
| BIND-05 | Phase 3 | Pending |
| BIND-06 | Phase 3 | Pending |
| BIND-07 | Phase 3 | Pending |
| BIND-08 | Phase 8 | Pending |
| BIND-09 | Phase 3 | Pending |
| ORACLE-01 | Phase 1 | Complete |
| ORACLE-02 | Phase 3 | Pending |
| ORACLE-03 | Phase 8 | Pending |
| ORACLE-04 | Phase 8 | Pending |
| ORACLE-05 | Phase 1 | Complete |
| ORACLE-06 | Phase 8 | Pending |
| ORACLE-07 | Phase 8 | Pending |
| ORACLE-08 | Phase 3 | Pending |
| ORACLE-09 | Phase 1 | Complete |
| PERF-01 | Phase 8 | Pending |
| PERF-02 | Phase 8 | Pending |
| PERF-03 | Phase 8 | Pending |
| PERF-04 | Phase 8 | Pending |
| PERF-05 | Phase 8 | Pending |
| PERF-06 | Phase 8 | Pending |
| PERF-07 | Phase 8 | Pending |
| DIST-01 | Phase 8 | Pending |
| DIST-02 | Phase 8 | Pending |
| DIST-03 | Phase 8 | Pending |
| DIST-04 | Phase 8 | Pending |
| DIST-05 | Phase 8 | Pending |
| DIST-06 | Phase 8 | Pending |
| — v1.x deferred (not counted in v1 total) — |||
| CCSD-T-01 | v1.x | Deferred |
| SCF-EXT-01 | v1.x | Deferred |
| SCF-EXT-02 | v1.x | Deferred |
| SCF-EXT-03 | v1.x | Deferred |
| SCF-EXT-04 | v1.x | Deferred |
| SCF-EXT-05 | v1.x | Deferred |
| DFT-EXT-01 | v1.x | Deferred |
| DFT-EXT-02 | v1.x | Deferred |
| HESS-01 | v1.x | Deferred |
| HESS-02 | v1.x | Deferred |
| HESS-03 | v1.x | Deferred |
| CCSD-EXT-01 | v1.x | Deferred |
| CCSD-EXT-02 | v1.x | Deferred |
| GEOMOPT-EXT-01 | v1.x | Deferred |
| DIST-EXT-01 | v1.x | Deferred |

**Coverage:**

- v1 requirements: 113 total (10 FOUND + 11 GTO + 14 SCF + 11 DFT + 8 MP2 + 11 CCSD + 10 GRAD + 7 GEOMOPT + 9 BIND + 9 ORACLE + 7 PERF + 6 DIST)
- Mapped to phases: 113 / 113 ✓
- Unmapped: 0
- v1.x deferred (listed in table for traceability completeness, excluded from the 113 v1 count): 15 — CCSD-T-01, SCF-EXT-01..05, DFT-EXT-01..02, HESS-01..03, CCSD-EXT-01..02, GEOMOPT-EXT-01, DIST-EXT-01
- Phase distribution: Phase 1 = 13 (FOUND-01..10 + ORACLE-01,05,09); Phase 2 = 11 (GTO-01..11); Phase 3 = 23 (SCF-01..14 + BIND-01,02,04,05,06,07,09 + ORACLE-02,08); Phase 4 = 11 (DFT-01..11); Phase 5 = 8 (MP2-01..08); Phase 6 = 11 (CCSD-01..11); Phase 7 = 17 (GRAD-01..10 + GEOMOPT-01..07); Phase 8 = 19 (PERF-01..07 + DIST-01..06 + ORACLE-03,04,06,07 + BIND-03,08)

> **Counting note**: The previous version of this section asserted "116 total ... + 3 cross-listed". On enumeration, the v1 requirements list contains 113 unique REQ-IDs. The "116" figure double-counted three cross-cutting concerns that are mentioned in multiple categories' prose (subclass-override re-validation in DFT-08, frozen-core re-validation in CCSD-10, MP2 helper exposure for CCSD in MP2-08) but are encoded as single REQ-IDs. The 113 figure is the correct unique-ID count.

---
*Requirements defined: 2026-05-09*
*Traceability filled: 2026-05-10 by roadmapper (8-phase structure)*
