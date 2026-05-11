# Roadmap: pyscf_rs

**Created:** 2026-05-10
**Granularity:** standard (5-8 target; landed at 8)
**Mode:** standard (Horizontal Layers — each phase is a complete chemistry module / cross-cutting concern, not a vertical user-story slice)
**Total v1 requirements:** 121 (10 FOUND + 8 ALG + 11 GTO + 14 SCF + 11 DFT + 8 MP2 + 11 CCSD + 10 GRAD + 7 GEOMOPT + 9 BIND + 9 ORACLE + 7 PERF + 6 DIST)
**Coverage:** 121 / 121 mapped

## Overview

pyscf_rs is a pure-Rust rewrite of PySCF that ships as a `pip install`-able wheel preserving the `from pyscf import gto, scf, dft, mp, cc, grad, geomopt` import surface. The architecture is locked: a 15-crate horizontal-layered façade workspace mirroring `cintx`/`xcfun_rs` (with one new member, `pyscf-algebra`, owning all linear algebra; Phase 3 grows the workspace to 18 by adding `pyscf-chkfile`, `pyscf-diis`, `pyscf-df`), cubecl 0.10.0 as the sole compute primitive (CPU SIMD/CUDA/WGPU/ROCm), faer 0.24 used only for host eigh/Cholesky/QR/SVD behind the algebra crate's surface, PyO3 0.28 for the Python boundary, and PySCF-as-live-oracle in CI. Backend selection is runtime-driven via `PYSCF_BACKEND`; the workspace `gpu` umbrella feature is OFF by default so the standard build is CPU-only. See `docs/manual/Cubecl/` for the cubecl runtime/ComputeClient/tensor-handle pattern that `pyscf-algebra` is built on.

The dependency DAG dictates phase ordering almost entirely: `core/runtime → kernels → gto → scf → {dft, mp2} → ccsd → grad → geomopt → wheel`. Phases 1–7 walk this critical path with the PyO3 contract folded into Phase 3 (SCF) so subclass-override / NumPy-boundary / GIL-release conventions lock on a small surface (RHF) before DFT's overrideable explosion. Phase 8 is the closing "ship readiness" phase combining GPU backend enable, oracle hardening, and wheel distribution because all three gate on the same artifact (a working CPU baseline across every method) and feed the same goal (validating the 2–5× speedup claim on a real benchmark suite, in a real wheel, on a real CI machine).

Five SHOWSTOPPER pitfalls and three MAJORs are addressed in Phase 1 and Phase 3 — this is by design. Foundational decisions (FMA-free profile, ordered-reduction primitives, panic policy, cubecl pin, scope-creep lint, sibling-crate ABI matrix) all land before any kernel touches a basis function. Bit-exact-with-PySCF and PyO3 drop-in fidelity are the two themes everything else inherits from.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED) — none yet

- [ ] **Phase 1: Foundation** — Workspace (15 crates including `pyscf-algebra`), core types, runtime + env-driven backend selection (`PYSCF_BACKEND`), workspace `gpu` feature (OFF by default → CPU is the default backend), single-owner cubecl algebra crate (GEMM/reduce/AXPY/dot via `cubecl-matmul`/`cubecl-reduce`/`#[cube]`), FMA-free oracle profile, ordered-reduction primitives, panic policy, cubecl pin, scope-creep + dependency-wall lints, nightly cross-crate matrix CI
- [ ] **Phase 2: GTO** — Mole, basis-set loading (5 atom-input × 11 basis-input forms), ECP, intor wrappers via cintx, eval_gto for grids
- [ ] **Phase 3: SCF + PyO3 bindings** — RHF/UHF/GHF + DIIS + chkfile + sign canonicalization + first end-to-end energy AND lock the entire PyO3 contract (subclass-override dispatch, NumPy contiguity, GIL release seam, abi3-py310 wheel skeleton, oracle harness bootstrap)
- [ ] **Phase 4: DFT** — RKS/UKS + Becke grids ported byte-for-byte + libxc/xcfun XC parser + range-separated hybrids + VV10 NLC + DF-DFT
- [ ] **Phase 5: MP2** — RMP2/UMP2/DF-MP2 + frozen-core + AO→MO transformation kernel + helpers CCSD imports
- [ ] **Phase 6: CCSD** — RCCSD/UCCSD + amplitude DIIS + Lambda + RDMs + AO-direct + DF-CCSD with HDF5 spill + tensor-arena from day one + T1/D1/D2 diagnostics
- [ ] **Phase 7: Gradients + Geomopt** — Analytical gradients for HF/DFT/MP2/CCSD + ECP + CPHF/CPKS + native Rust BFGS+RFO in redundant internals + geomeTRIC/berny drop-in shims
- [ ] **Phase 8: GPU enable + Oracle hardening + Distribution** — Per-backend regression suite (CPU/CUDA/WGPU/ROCm), 2–5× benchmark proof, abi3-py310 wheel for Linux/macOS/Windows × x86_64+aarch64, per-backend extras, drop-in audit (≥80% upstream tests pass against pyscf-rs as import target), full top-20-idiom shakedown

## Phase Details

### Phase 1: Foundation
**Goal**: The workspace exists, builds clean as a 15-crate horizontal-layered façade, the `pyscf-algebra` crate exposes a backend-agnostic linear-algebra surface dispatching to cubecl on the active runtime, and every cross-cutting convention that gates downstream numerical correctness is in place and CI-enforced before the first kernel lands.
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, FOUND-06, FOUND-07, FOUND-08, FOUND-09, FOUND-10, ALG-01, ALG-02, ALG-03, ALG-04, ALG-05, ALG-06, ALG-07, ALG-08, ORACLE-01, ORACLE-05, ORACLE-09
**Success Criteria** (what must be TRUE):
  1. `cargo build --workspace` succeeds with no GPU features (CPU-only, the default); the workspace contains 15 members (`pyscf-{core,runtime,algebra,kernels,gto,scf,dft,mp2,ccsd,grad,geomopt,py,oracle,bench}` + top-level façade) and `pyscf-{core,runtime,algebra}` are non-stub (BackendKind enum, `select_backend()` env-driven resolver, WorkspacePool, Mole/Density/Energy types, `AlgebraClient` enum + `gemm`/`reduce_sum`/`axpy`/`dot` surface, all traits compile). `cargo build --workspace --features gpu` additionally compiles the `cuda` and `wgpu` cubecl runtimes.
  2. `cargo build --profile release-oracle --workspace` produces FMA-free machine code; CI runs `cargo-llvm-ir | grep llvm.fmuladd` over every numerical crate's object files under the oracle profile and finds **zero** matches (Pitfall 1 mitigation).
  3. A canary test using `oracle_sum`/`oracle_dot` reduction primitives produces **bit-identical** results on `RAYON_NUM_THREADS=1` and `RAYON_NUM_THREADS=8` runs of the same input vector (Pitfall 2 mitigation).
  4. `cubecl = "=0.10.0"` (and all `cubecl-*` crates including `cubecl-cpu`, `cubecl-wgpu`, `cubecl-cuda`, `cubecl-rocm`, `cubecl-matmul`, `cubecl-reduce`, `cubecl-std`) are pinned exactly via `[patch.crates-io]` in workspace `Cargo.toml`, matching cintx/libxc_rs/xcfun_rs; nightly cross-crate matrix CI rebuilds and tests cintx + libxc_rs + xcfun_rs + pyscf_rs together against the pin and reports green (Pitfall 3 + 15 mitigation, ORACLE-05).
  5. CI enforces four lints that block PR merge: (a) `unwrap()` in numerical modules → clippy deny, (b) `forbidden-paths` for upstream out-of-scope imports (pbc/x2c/mcscf/tdscf/adc/gw/eom/NAC/EPH) → custom lint deny (FOUND-08, Pitfall 21), (c) every `extern "C"` callback wrapped in `catch_unwind` → grep-based CI check (FOUND-07, Pitfall 14), (d) **algebra dependency-wall lint** (ALG-06): `cargo metadata` graph check fails the build if any crate other than `pyscf-algebra` or `pyscf-runtime` declares a `cubecl-*` dependency.
  6. **Backend resolution behaves**: a `pyscf-algebra` integration test sets `PYSCF_BACKEND` to each of `cpu`/`cuda`/`wgpu`/`rocm`/`metal`/`auto`/`unset`/`bogus` and asserts the resolved backend matches the documented FOUND-03 + ALG-04 rules — including the case where `PYSCF_BACKEND=cuda` is set but the `cuda` feature is not compiled in (must fall back to CPU + emit `tracing::warn!`). With no env var set on a CPU-only build, GEMM/reduce-sum/axpy on a 256×256 input agree with a `faer 0.24` host reference to 1e-12 (ALG-01..04, ALG-08).
**Plans**: 9 plans across 6 waves (7 shipped + 2 gap-closure)

Plans:
- [x] 01-01-PLAN.md — Workspace skeleton (root Cargo.toml, 12 stub crates, .cargo/config.toml, deny.toml; FOUND-01, FOUND-04, FOUND-10, ORACLE-01)
- [x] 01-02-PLAN.md — pyscf-core universal types and method traits (FOUND-02)
- [x] 01-03-PLAN.md — pyscf-runtime BackendKind, probes, WorkspacePool, tracing init (FOUND-03, FOUND-09, ALG-04, ALG-08-prep)
- [x] 01-04-PLAN.md — pyscf-algebra AlgebraClient + select_backend + 7 primitives + oracle_sum + 4 integration tests + façade (ALG-01..05, ALG-07, ALG-08, FOUND-06)
- [x] 01-05-PLAN.md — xtask 5 CI lint binaries (FOUND-05, FOUND-07, FOUND-08, ALG-06)
- [x] 01-06-PLAN.md — GitHub Actions ci.yml + nightly-cross-crate.yml (FOUND-05, FOUND-08, FOUND-10, ALG-06, ORACLE-05, ORACLE-09)
- [x] 01-07-PLAN.md — CONTRIBUTING.md + docs/upgrade-cubecl.md + README.md additions (FOUND-04, FOUND-09)
- [ ] 01-08-PLAN.md — GAP CLOSURE: cintx clean-SHA repin + Cargo.lock commit (closes BLOCKER 1 + 3; FOUND-01, FOUND-04, FOUND-10)
- [ ] 01-09-PLAN.md — GAP CLOSURE: check-cubecl-pin transitive version-skew reconciliation (closes BLOCKER 2; FOUND-04)

### Phase 2: GTO
**Goal**: A user can construct a molecule with any of upstream PySCF's atom-input or basis-input forms and run any 1e/2e integral upstream supports for in-scope methods, with byte-for-byte agreement on the internal `_atm`/`_bas`/`_env` arrays.
**Depends on**: Phase 1
**Requirements**: GTO-01, GTO-02, GTO-03, GTO-04, GTO-05, GTO-06, GTO-07, GTO-08, GTO-09, GTO-10, GTO-11
**Success Criteria** (what must be TRUE):
  1. `pyscf.M(atom='O 0 0 0; H 0 1 0; H 1 0 0', basis='cc-pvdz')` and the four other atom-input forms produce a `Mole` whose `_atm`, `_bas`, `_env`, `ao_loc_nr`, `nao_nr` arrays match upstream PySCF byte-for-byte on the test corpus (GTO-01, GTO-04).
  2. All 207 built-in basis-set files in `pyscf/gto/basis/` resolve correctly via `mol.basis = '<name>'`; `gto.parse(...)` accepts user-supplied Gaussian-94 and NWChem text; ECP via `mol.ecp = ...` loads and `mol.intor('int1e_ecp')` matches upstream bit-exact under `release-oracle` (GTO-02, GTO-03, GTO-05).
  3. `mol.intor('int2e')`, `mol.intor('int1e_ovlp_sph')`, and the integral families upstream PySCF supports for SCF/DFT/MP2/CCSD/grad all dispatch to `cintx` and produce arrays that match upstream within the cintx oracle tolerance; F-order layout is preserved on output where upstream returns F-order (Pitfall 8 mitigation).
  4. `eval_gto(mol, name, coords, ...)` for `GTOval`, `GTOval_sph`, `GTOval_deriv1`, `GTOval_deriv2`, `GTOval_ip`, `GTOval_ig` matches upstream values element-wise on a 1000-point grid (GTO-07).
  5. `Mole` exposes the ≥30 attribute floor (`atom`, `basis`, `charge`, `spin`, `nelectron`, `natm`, `nbas`, `nao_nr`, `nao_2c`, `ao_loc_nr`, `ao_labels`, `cart`, `verbose`, `max_memory`, `unit`, `output`, `_atm`, `_bas`, `_env`, …); `mol.dumps()`/`gto.Mole.loads()` JSON round-trip; `mol.copy()` deep-copies; `mol.set_geom_(new_atom)` mutates in place and returns self (GTO-08, GTO-09, GTO-10).
**Plans**: 10 plans across 10 waves (1 Wave 0 risk-buy-down + 8 implementation + 1 deferred gap-closure for cintx ECP)

Plans:
- [x] 02-01-PLAN.md — Wave 0 scaffolding: cintx round-trip smoke, cubecl-cpu kernel smoke, F/C-order layout table, algebra-wall allowlist update, oracle harness scaffold, PYSCF_BASIS_PATH docs (W0-T1..W0-T6)
- [x] 02-02-PLAN.md — Mole struct + ≥30-attribute floor + format_atom port (4-of-5 atom-input forms; 5th deferred to Phase 3) (GTO-01, GTO-08)
- [x] 02-03-PLAN.md — Basis loader (PYSCF_BASIS_PATH resolver + ALIAS table + NWChem/NWChem-ECP/CP2K parser dispatch + format_basis dispatcher) (GTO-02, GTO-03)
- [x] 02-04-PLAN.md — Mole↔cintx bridge (zero-copy BasisSet re-export + make_env flat-array projection) (GTO-04, GTO-11)
- [x] 02-05-PLAN.md — mol.intor(name) cintx dispatcher (with F-order layout preservation per Pitfall 8) (GTO-06)
- [x] 02-06-PLAN.md — eval_gto cubecl kernel in pyscf-kernels + algebra-wall-friendly user wrapper in pyscf-gto (GTO-07; s-shells fully implemented; l ≥ 1 deferred to Phase 4 DFT)
- [x] 02-07-PLAN.md — ECP loading parser + EcpEngine trait + EcpEngineNotAvailable stub + intor dispatcher routing (GTO-05 loading half)
- [x] 02-08-PLAN.md — mol.dumps()/Mole::loads() JSON round-trip + mol.copy() + mol.set_geom_() in-place mutation per Pattern 5 (GTO-09, GTO-10)
- [x] 02-09-PLAN.md — Phase 2 verification rollup: pytest oracle harness for byte-identity + intor + eval_gto + JSON interop + builtin basis sweep + STATE/VALIDATION updates (verifies GTO-01..11)
- [ ] 02-10-PLAN.md — DEFERRED gap-closure: cintx ECP merge → swap EcpEngineNotAvailable for cintx-backed CintxEcpEngine; closes GTO-05 evaluation half (status: PENDING_CINTX_ECP_MERGE)

### Phase 3: SCF + PyO3 bindings
**Goal**: A Python user runs `from pyscf import scf; scf.RHF(mol).kernel()` from an unmodified existing PySCF script and gets the same total energy as upstream PySCF to ≤1 µHartree, while every PyO3 contract that downstream methods inherit (subclass-override dispatch, NumPy contiguity, GIL release seam, panic-to-exception, abi3 wheel) is locked and CI-enforced on this single small surface (RHF on H2O/cc-pVDZ).
**Depends on**: Phase 2
**Requirements**: SCF-01, SCF-02, SCF-03, SCF-04, SCF-05, SCF-06, SCF-07, SCF-08, SCF-09, SCF-10, SCF-11, SCF-12, SCF-13, SCF-14, BIND-01, BIND-02, BIND-04, BIND-05, BIND-06, BIND-07, BIND-09, ORACLE-02, ORACLE-08
**Success Criteria** (what must be TRUE):
  1. `scf.RHF(mol).kernel()` on the test corpus (H2O/cc-pVDZ, benzene/6-31G*, …) converges to upstream PySCF total energy to ≤1 µHartree under `release-oracle` (bit-exact when reduction order matches); `scf.UHF(mol).kernel()` matches upstream for open-shell systems; `scf.GHF(mol).kernel()` runs to completion (correctness only, perf parity not required) (SCF-01, SCF-02, SCF-03).
  2. **Cross-platform invariant**: running `scf.RHF(H2O).kernel()` under `release-oracle` on Linux x86_64 and macOS aarch64 produces total energies that agree to within 1 µHartree of each other (Pitfall 12 mitigation, depends on FOUND-05/06 and on `pyscf-core::lib::canonicalize_signs` (SCF-13) producing vendor-stable eigenvectors via the largest-|coefficient|-with-lowest-index sign-flip rule — Pitfall 4 mitigation).
  3. C-DIIS with `mf.diis_space=8`/`mf.diis_start_cycle=1` reproduces upstream DIIS extrapolation when reduction order is held; all five `init_guess` modes (`'minao'`, `'atom'`, `'1e'`, `'huckel'`, `'chkfile'`) plus user-supplied `dm0` produce upstream-matching first-iteration densities; `mf.density_fit(auxbasis=...)` solves DF-HF with upstream-matching aux defaults; `mf.chkfile = path` writes an HDF5 file that h5py reads with the upstream PySCF schema and `mf.from_chk(path)` reads upstream-h5py-written chkfiles (SCF-04..07, SCF-10, SCF-14, ORACLE-08, Pitfall 9 + 11 mitigation).
  4. **PyO3 subclass dispatch works**: a Python user defines `class MyHF(scf.RHF): def get_veff(self, mol, dm): return super().get_veff(mol, dm) + correction(dm)` and the Rust SCF driver calls the Python override (verified by an in-CI assertion that the override is invoked at least once per cycle); the same is true for every overrideable hook (`get_jk`, `get_hcore`, `get_init_guess`, `get_fock`, `get_occ`, `eig`, `make_rdm1`, `energy_elec`, `energy_tot`) — dispatched via `slf.call_method1(py, …)`, never via Rust MRO (BIND-07, SCF-08, Pitfall 7 mitigation).
  5. **PyO3 boundary discipline locked**: every public PyO3 entry point that takes a NumPy array calls `to_owned()` on input that is not `is_standard_layout()`, and a stride-fuzz CI test that calls each entry with `a`, `a.T`, `a[::2]`, `a[:, 1:5]` produces identical answers (BIND-04, Pitfall 5); long compute calls `Python::detach` and a `python3.13t` free-threaded CI build runs the SCF test corpus without deadlocks (BIND-05, Pitfall 6); `pyo3::sync::GILOnceCell` replaces every `lazy_static!` (BIND-06); a Rust panic in any kernel called via FFI surfaces as a Python exception with the original error chain preserved, **never** as a process abort or undefined behavior (BIND-09, Pitfall 14); `from pyscf import scf` works exactly as upstream via `_native.scf` PyO3 submodule + `python/pyscf/__init__.py` re-export shim (BIND-01, BIND-02).
  6. **Oracle harness bootstrap**: the `oracle_check!(method, tolerance, fixture)` macro is implemented in `pyscf-oracle` (dev-deps only); every SCF success-criterion above is asserted via this macro on a curated H2O/benzene/water-trimer corpus; chkfile round-trip oracle (PySCF writes → pyscf-rs reads asserts identical, pyscf-rs writes → PySCF reads runs downstream calc asserts agreement) is in CI (ORACLE-02, ORACLE-08); `mf.analyze()`, `mf.mulliken_pop()`, `mf.mulliken_meta()`, `mf.dip_moment()` produce upstream-matching numbers (SCF-09); cross-module dispatch helpers `mf.to_uhf()`, `mf.to_rhf()`, `mf.to_uks()`, `mf.to_rks()`, `mf.to_ghf()` work (SCF-11) because MP2/CCSD will depend on them; `mf.as_scanner()` returns a callable used by geomopt (SCF-12).
**Plans**: 11 plans across 8 waves (split per checker iteration 1 WARNING 3)

Plans:
- [ ] 03-01-PLAN.md — Workspace scaffolding (+3 crates: pyscf-chkfile/diis/df, pyscf-algebra::solve_linear, pyscf-core::canonicalize_signs; SCF-13)
- [ ] 03-02-PLAN.md — Wave-0 test stubs (pyproject.toml maturin config, python overlay shim, 19 pytest xfail stubs, oracle macro stub, forbid-lazy-static lint; BIND-02 scaffolding, BIND-06)
- [ ] 03-03-PLAN.md — pyscf-scf trait + struct scaffolding (OverrideHooks trait, RHF/UHF/GHF + 30-attribute floor, InitGuessMode declarations, kernel signature; SCF-01..03, SCF-05, SCF-06, SCF-14) — WARNING 3 split
- [ ] 03-04-PLAN.md — pyscf-diis crate (CDIIS, SDF-FDS error vector, B-matrix via pyscf-algebra::solve_linear, FockSubspace impl DiisStorable; SCF-04, Pitfall 9 mitigation)
- [ ] 03-05-PLAN.md — pyscf-df crate (DfIntegrals, cholesky_eri, DEFAULT_AUXBASIS, get_jk_df, RHF::density_fit; SCF-07)
- [ ] 03-06-PLAN.md — pyscf-chkfile crate + pyscf-scf chkfile schema + 'chkfile' init_guess mode (D-05/D-06; SCF-10, DIST-05 baseline)
- [ ] 03-07-PLAN.md — pyscf-py PyO3 bridge (#[pymodule] _native, PyRHF/UHF/GHF, PyOverrideBridge, NumPy converters, create_exception!, abi3-py310 + free-threading features, python/pyscf overlay; BIND-01, BIND-02, BIND-04, BIND-06, BIND-07, BIND-09, SCF-08)
- [ ] 03-08-PLAN.md — pyscf-oracle macro body + chkfile round-trip oracle (ORACLE-08 empirical h5py↔hdf5-metno seal — STATE.md blocker; ORACLE-02, ORACLE-08)
- [ ] 03-09-PLAN.md — CI jobs (maturin-smoke, stride-fuzz, xplat-uhartree Linux x86_64 + macOS aarch64 matrix, python313t-smoke NON-abi3 separate build per RESEARCH Pitfall (NEW); BIND-05, Pitfall 12)
- [ ] 03-10-PLAN.md — Python test bodies (replace 19 xfail stubs with real ≤1 µHartree / element-wise / bit-identical assertions; verifies SCF-01..14 + BIND-02/04/07/09 + ORACLE-08)
- [ ] 03-11-PLAN.md — pyscf-scf kernel internals (SCF cycle loop body, Fock build, eig+canonicalize_signs, occ+rdm+energy, '1e' init_guess body, analyze/mulliken/dip, convert helpers, as_scanner; SCF-01..03, SCF-05, SCF-06, SCF-09, SCF-11, SCF-12, SCF-13) — NEW, WARNING 3 split
**UI hint**: yes

### Phase 4: DFT
**Goal**: A user runs `dft.RKS(mol, xc='b3lyp').run()` on the test corpus and gets the same total energy as upstream PySCF bit-exact under `release-oracle`; every DFT-specific overrideable hook (the largest single Python-override surface in the project) re-validates the Phase 3 PyO3 contract; and the integration of all three sibling crates (`cintx` + `libxc_rs` + `xcfun_rs`) into one consistent compute pipeline is proven on a real DFT cycle.
**Depends on**: Phase 3
**Requirements**: DFT-01, DFT-02, DFT-03, DFT-04, DFT-05, DFT-06, DFT-07, DFT-08, DFT-09, DFT-10, DFT-11
**Success Criteria** (what must be TRUE):
  1. `dft.RKS(mol, xc='b3lyp').kernel()` and `dft.UKS(mol, xc=...).kernel()` on the test corpus converge bit-exact to upstream PySCF under `release-oracle` (within 1 µHartree on every fixture in the test corpus); the XC string parser handles all upstream forms — single name (`'b3lyp'`), comma form (`'pbe,pbe'`), shorthands (`'lda'` → `'lda,vwn'`), explicit weights (`'.5*HF + .5*B88,LYP'`), and aliases from `XC_ALIAS` — with a parser-parity unit test against `pyscf/dft/libxc.py` (DFT-01, DFT-02).
  2. libxc functional evaluation routes through `libxc_rs` and produces numbers identical to upstream libxc on a 100-functional smoke; xcfun routes through `xcfun_rs` and produces numbers identical to upstream xcfun (DFT-03); `numint.NumInt` exposes `eval_xc`, `eval_rho`, `nr_rks`, `nr_uks` matching upstream signatures (DFT-10).
  3. **Grid weights bit-exact**: the `Grids` class with `level`, `atom_grid`, `prune`, `radi_method`, `becke_scheme`, `atomic_radii` controls produces grid points and weights byte-for-byte identical to upstream `pyscf/dft/gen_grid.py` for `level ∈ {0..9}` on the test corpus (DFT-04, DFT-09, Pitfall 10 mitigation); range-separated hybrids (`omega`, `alpha`, `beta`) use cintx's `int2e_lr_*`/`int2e_sr_*` integral families with a parity test on a CAM-B3LYP H2O fixture (DFT-05); VV10 non-local correlation produces upstream-matching energies via `mf.nlc='VV10'`/`mf.nlcgrids` (DFT-06); `dft.RKS(mol).density_fit()` solves DF-DFT and matches upstream (DFT-07).
  4. **Subclass-override re-validation at DFT scope**: a Python user defines `class MyKS(dft.RKS): def get_veff(...)` AND `def define_xc_(...)` and the Rust DFT driver invokes both Python overrides every cycle (DFT-08, re-asserts Pitfall 7 on the larger DFT overrideable surface).
  5. **WGPU f64 honesty**: the `wgpu` feature is gated on the `shader-f64` Vulkan extension being present at runtime; when the extension is missing, the runtime falls back to CPU with a clear warning rather than silently degrading to f32 — proven by a CI job on a `shader-f64`-less device that runs `dft.RKS(mol).run()` and prints the fallback warning while still producing CPU-correct numbers (DFT-11, Pitfall 3 mitigation).
**Plans**: TBD

### Phase 5: MP2
**Goal**: A user runs `mp.RMP2(mf).kernel()` and `mp.DFMP2(mf).kernel()` on the test corpus and gets upstream-matching correlation energies bit-exact under `release-oracle`; the AO→MO transformation kernel is general enough to be reused by CCSD; the MP2 helpers CCSD will import (`get_nocc`, `get_nmo`, `get_frozen_mask`, `get_e_hf`, `_mo_without_core`) are exposed and contract-tested.
**Depends on**: Phase 3 (SCF + PyO3 bindings); Phase 4 (DFT) is parallelizable with this phase per the architecture's wave W5.
**Requirements**: MP2-01, MP2-02, MP2-03, MP2-04, MP2-05, MP2-06, MP2-07, MP2-08
**Success Criteria** (what must be TRUE):
  1. `mp.RMP2(mf).kernel()` on RHF references and `mp.UMP2(uhf_mf).kernel()` on UHF references reproduce upstream MP2 correlation energy bit-exact under `release-oracle` on the test corpus (MP2-01, MP2-02); `mf.MP2().run()` (the cross-module dispatch idiom) returns the same numbers (MP2-01).
  2. Frozen-core options accept `frozen=int`, `frozen=list`, `frozen='auto'`, and frozen-window forms; defaults match upstream on the test corpus (MP2-03).
  3. `mp.DFMP2(mf).kernel()` reproduces upstream DF-MP2 (MP2-04); SCS-MP2 via `mp.MP2(mf).set(emp2_ss_factor=..., emp2_os_factor=...)` works (MP2-06); `mp2.make_rdm1()` and `mp2.make_rdm2()` match upstream (MP2-05).
  4. `mp2.as_scanner()` returns a callable that takes a Mole and returns the energy — exercised by a geomopt smoke test in Phase 7 (MP2-07).
  5. MP2 helpers (`get_nocc`, `get_nmo`, `get_frozen_mask`, `get_e_hf`, `_mo_without_core`) are exported with upstream-matching semantics and contract-tested via a unit test that mimics CCSD's exact import call site (MP2-08).
**Plans**: TBD

### Phase 6: CCSD
**Goal**: A user runs `cc.RCCSD(mf).kernel()` on caffeine/cc-pVDZ within `PYSCF_MAX_MEMORY` and gets upstream CCSD correlation energy to ≤1 µHartree without OOMing or thrashing the heap; the tensor-arena/scratchpad pattern in `pyscf-runtime` is in place from the start (not retrofitted) so `Wabef` and other large intermediates do not allocate-and-drop per iteration; AO-direct and DF-CCSD modes both work.
**Depends on**: Phase 5 (CCSD imports MP2 helpers `get_nocc`, `get_nmo`, `get_frozen_mask`, `get_e_hf`, `_mo_without_core` directly per `cc/ccsd.py:35`)
**Requirements**: CCSD-01, CCSD-02, CCSD-03, CCSD-04, CCSD-05, CCSD-06, CCSD-07, CCSD-08, CCSD-09, CCSD-10, CCSD-11
**Success Criteria** (what must be TRUE):
  1. `cc.RCCSD(mf).kernel()` and `cc.UCCSD(uhf_mf).kernel()` return correlation energies matching upstream to ≤1 µHartree on the test corpus; T1 and T2 amplitudes converge to the same minimum (energy is the convergence target; amplitude paths may differ within tolerance) (CCSD-01..03); amplitude-DIIS with default `diis_space=6` converges within the same iteration count as upstream on the test corpus (CCSD-04).
  2. **Tensor-arena pattern in place from day one**: a CCSD iteration on caffeine/cc-pVDZ allocates `Wabef` and other large intermediates **once** at the start of the calculation (verified by a heap-allocation count assertion in CI); a `PYSCF_MAX_MEMORY` pre-flight check refuses to start a calculation that would exceed the budget rather than OOMing mid-iteration (CCSD-11, Pitfall 20 mitigation).
  3. `mycc.solve_lambda()` produces λ amplitudes for response densities (used by CCSD gradients in Phase 7); `mycc.make_rdm1()` and `mycc.make_rdm2()` match upstream (CCSD-05, CCSD-06).
  4. `mycc.direct = True` (AO-direct CCSD) works; DF-CCSD via `mf.density_fit().CCSD()` or `cc.dfccsd.RCCSD(mf)` works with bounded memory and spills `Wabef` to HDF5 when `PYSCF_MAX_MEMORY` is exceeded — proven by a benzene-dimer/cc-pVDZ DF-CCSD run on a deliberately constrained memory budget (CCSD-07, CCSD-08).
  5. T1/D1/D2 diagnostics expose `mycc.t1diagnostic()`, `mycc.d1diagnostic()` matching upstream values; frozen-core options match MP2 (`frozen=int`, `frozen=list`, `frozen='auto'`) (CCSD-09, CCSD-10).
**Plans**: TBD

### Phase 7: Gradients + Geomopt
**Goal**: A user runs `mf.nuc_grad_method().kernel()` for any in-scope method (HF/DFT/MP2/CCSD) and gets upstream-matching analytical gradients; runs `pyscf.geomopt.optimize(mf)` (or the geomeTRIC/berny shims) and converges to the same stationary point as upstream within chemical accuracy — without a Python `geomeTRIC` or `pyberny` runtime dependency, because the optimizer is native Rust BFGS+RFO.
**Depends on**: Phase 6 (CCSD gradients need Λ-equations from `mycc.solve_lambda()`)
**Requirements**: GRAD-01, GRAD-02, GRAD-03, GRAD-04, GRAD-05, GRAD-06, GRAD-07, GRAD-08, GRAD-09, GRAD-10, GEOMOPT-01, GEOMOPT-02, GEOMOPT-03, GEOMOPT-04, GEOMOPT-05, GEOMOPT-06, GEOMOPT-07
**Success Criteria** (what must be TRUE):
  1. `mf.nuc_grad_method().kernel()` returns analytical gradients matching upstream to ≤1e-7 Hartree/Bohr for RHF, UHF, RKS (with `grid_response=True`), UKS, MP2 (Z-vector/CPHF), CCSD (Λ-equations), and ECP gradients on the test corpus (GRAD-01..07); atom-list subsetting via `grad.kernel(atmlst=[1,2,3])` returns just those rows of the gradient (GRAD-08).
  2. Every analytical gradient passes a finite-difference verification mode `grad.verify_fd(disp=1e-4)` to within 1e-6 Hartree/Bohr — this gates unit tests (GRAD-09).
  3. The CPHF/CPKS solver lives in `pyscf-grad` (or a shared module) and is reused by every method gradient that needs response equations (RKS-grad with `grid_response`, MP2-grad Z-vector, CCSD-grad Λ); a single CI test confirms there is one CPHF implementation, not N (GRAD-10).
  4. `pyscf.geomopt.optimize(mf)` runs a native Rust BFGS+RFO optimizer in redundant internal coordinates with **no** Python `geomeTRIC` or `pyberny` runtime dependency (verified by `pip uninstall geomeTRIC pyberny && python -c "import pyscf.geomopt; pyscf.geomopt.optimize(mf)"` succeeding) (GEOMOPT-01); `pyscf.geomopt.geometric_solver.optimize(mf)` and `pyscf.geomopt.berny_solver.optimize(mf)` are drop-in shims that delegate to the native optimizer, preserving the canonical PySCF import paths (GEOMOPT-02, GEOMOPT-03).
  5. Default convergence thresholds match geomeTRIC defaults (`gradient`, `displacement`, `energy`, `gradient_max`, `displacement_max`); Wilson B-matrix construction for redundant internals and RFO step with negative-eigenvalue tracking are ported from upstream/geomeTRIC; HDF5 checkpoint of optimizer state allows resuming a partially-converged optimization; optimization trajectories on the test corpus converge to the same stationary point as upstream within chemical accuracy (GEOMOPT-04..07).
**Plans**: TBD

### Phase 8: GPU enable + Oracle hardening + Distribution
**Goal**: A Python user on a fresh container runs `pip install pyscf-rs[cuda]` and `python -c "from pyscf import gto, scf, dft, mp, cc, grad, geomopt"` and every one of the top-20 drop-in idioms succeeds; the 2–5× speedup claim against current PySCF + C extensions is proven on a defined benchmark suite; ≥80% of curated upstream PySCF unit tests for in-scope modules pass when run against pyscf-rs as the import target.
**Depends on**: Phase 7 (every CPU path must be correct on the oracle before GPU drift is introduced)
**Requirements**: PERF-01, PERF-02, PERF-03, PERF-04, PERF-05, PERF-06, PERF-07, DIST-01, DIST-02, DIST-03, DIST-04, DIST-05, DIST-06, ORACLE-03, ORACLE-04, ORACLE-06, ORACLE-07, BIND-03, BIND-08
**Success Criteria** (what must be TRUE):
  1. **GPU backends enabled and correct**: per-backend regression suite runs the full SCF/DFT/MP2/CCSD test corpus on CPU SIMD, CUDA, WGPU, and ROCm by setting `PYSCF_BACKEND` (no recompile per backend on a `--features gpu` build) — exercising the Phase 1 `pyscf-algebra` dispatch end-to-end (ALG-04, ALG-07); where hardware is available in CI, GPU backends pass at chemical accuracy with documented per-backend tolerance (CPU: bit-exact under oracle profile; CUDA: 1e-10 Hartree energy / 1e-8 gradient; WGPU: chemical accuracy 1e-6; ROCm: 1e-10) (ORACLE-07); cubecl autotune cache ships at `CUBECL_CACHE_DIR` so first-run overhead does not regress the benchmark (PERF-06); adaptive backend dispatch falls back to CPU when `nao < 200` to avoid GPU launch overhead (PERF-07, Pitfall 19 mitigation).
  2. **2–5× speedup claim proven**: criterion-based `pyscf-bench` crate covers RHF, RKS, MP2, CCSD on H2O/cc-pVDZ, benzene/6-31G*, 20-water cluster/cc-pVDZ, alanine dipeptide/def2-SVP, caffeine/cc-pVDZ; pyscf-rs achieves **≥2× speedup** vs current PySCF + C extensions on this suite on a fair-comparison machine (same CPU, same thread count, no GPU); **stretch**: ≥5× on at least one benchmark; CUDA backend demonstrates additional speedup on caffeine and alanine dipeptide; `mol.build()` is sub-second for 5000-AO molecules (PERF-01..05).
  3. **Drop-in audit passes**: the top-20 idioms from BIND-03 (from `pyscf.M(...)` through `mol.dumps()`/`gto.Mole.loads(s)`) all run unchanged against pyscf-rs as the import target on a representative existing PySCF user script; ≥80% of curated upstream PySCF unit tests for in-scope modules pass against pyscf-rs (ORACLE-04); nightly per-basis bit-exact sweep covers every basis-set name PySCF knows (ORACLE-06); test isolation uses subprocess-per-fixture for fixtures that mutate global state, persistent worker for stateless ones (ORACLE-03, Pitfall 16 mitigation) (BIND-03).
  4. **Wheel ships**: `pyscf-rs` published on crates.io with the workspace façade re-exporting in-scope methods (DIST-01); abi3-py310 PyPI wheel installs cleanly on Linux/macOS/Windows × x86_64 + macOS aarch64; `pip install pyscf-rs && python -c "from pyscf import gto, scf"` succeeds in a fresh container (DIST-02); per-backend optional extras `pyscf-rs[cuda]`/`pyscf-rs[wgpu]`/`pyscf-rs[rocm]` keep the base wheel under the PyPI 60 MB ceiling (DIST-03, Pitfall 13 mitigation); manylinux_2_28 baseline; `auditwheel show` clean (DIST-04); HDF5 ships statically linked via `hdf5-sys/static`, no system libhdf5 required (DIST-05); `python/pyscf/__init__.py` import shim makes `import pyscf` Just Work (DIST-06); `abi3audit` runs in CI on the produced wheel and fails on non-abi3 symbols (BIND-08).
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 7/9 | Gap closure pending (2 plans) | - |
| 2. GTO | 0/10 | Plans created (9 active + 1 deferred gap-closure for cintx ECP) | - |
| 3. SCF + PyO3 bindings | 0/10 | Planned | - |
| 4. DFT | 0/TBD | Not started | - |
| 5. MP2 | 0/TBD | Not started | - |
| 6. CCSD | 0/TBD | Not started | - |
| 7. Gradients + Geomopt | 0/TBD | Not started | - |
| 8. GPU enable + Oracle hardening + Distribution | 0/TBD | Not started | - |

## Coverage Summary

All 121 v1 REQ-IDs are mapped to exactly one phase. No orphans. No duplicates.

| Category | Count | Phase(s) |
|----------|-------|----------|
| FOUND-01..10 | 10 | Phase 1 |
| ALG-01..08 | 8 | Phase 1 |
| GTO-01..11 | 11 | Phase 2 |
| SCF-01..14 | 14 | Phase 3 |
| DFT-01..11 | 11 | Phase 4 |
| MP2-01..08 | 8 | Phase 5 |
| CCSD-01..11 | 11 | Phase 6 |
| GRAD-01..10 | 10 | Phase 7 |
| GEOMOPT-01..07 | 7 | Phase 7 |
| BIND-01..09 | 9 | split: BIND-01,02,04,05,06,07,09 → Phase 3; BIND-03,08 → Phase 8 |
| ORACLE-01..09 | 9 | split: ORACLE-01,05,09 → Phase 1; ORACLE-02,08 → Phase 3; ORACLE-03,04,06,07 → Phase 8 |
| PERF-01..07 | 7 | Phase 8 |
| DIST-01..06 | 6 | Phase 8 |
| **Total** | **121** | **8 phases** |

## Cross-Cutting Concerns Threaded Through Every Phase

These are NOT separate phases — they are conventions established in Phase 1 + Phase 3 that every later phase re-validates as part of its success criteria:

- **Algebra-responsibility wall (`pyscf-algebra` is the only `cubecl-*` consumer)**: Phase 1 establishes (single-owner crate + dependency-graph lint); every method phase consumes algebra primitives only via `pyscf-algebra::{gemm,reduce_sum,axpy,dot,…}` and never imports `cubecl-*` directly. Reference: `docs/manual/Cubecl/`.
- **Backend selection at runtime via `PYSCF_BACKEND` (CPU-default)**: Phase 1 establishes (`select_backend()` resolver + workspace `gpu` feature OFF by default); every PyO3 entry point logs the resolved backend (ALG-08); Phase 8 GPU enable phase exercises the priority chain across CUDA/WGPU/ROCm hardware.
- **Bit-exact-with-PySCF**: Phase 1 establishes (`release-oracle` profile, `oracle_sum`/`oracle_dot`); every method phase asserts on the test corpus.
- **PyO3 subclass-override dispatch**: Phase 3 establishes (`slf.call_method1`); Phase 4 (DFT) re-asserts on the larger DFT overrideable surface; Phases 5/6/7 inherit by convention.
- **NumPy contiguity**: Phase 3 establishes (`to_owned()` on non-standard-layout); every PyO3 entry point in Phases 4–7 reuses the helper.
- **GIL-release seam**: Phase 3 establishes (`Python::detach` on long compute); Phase 6 (CCSD) is the heaviest re-validation under `python3.13t`.
- **Panic policy**: Phase 1 establishes (`catch_unwind`, no `unwrap()` in numerical code, `panic="abort"` release); every phase inherits via the workspace clippy config.
- **Scope-creep lint**: Phase 1 establishes (`forbidden-paths`); every PR in every phase passes through it.
- **cubecl pin lockstep with cintx/libxc_rs/xcfun_rs**: Phase 1 establishes; Phase 8 GPU enable phase is where a sibling-crate cubecl bump would force the most rework, so CONTRIBUTING.md upgrade ritual is documented in Phase 1.

## Pitfall-to-Phase Mapping (v1 — derived from research/PITFALLS.md)

| # | Pitfall | Severity | Primary Phase | Re-validated In |
|---|---------|----------|---------------|------------------|
| 1 | FMA contraction | SHOWSTOPPER | Phase 1 | every method phase |
| 2 | Parallel-reduction order | SHOWSTOPPER | Phase 1 | every method phase |
| 3 | cubecl pre-1.0 / WGPU f64 | MAJOR | Phase 1 | Phase 4 (DFT WGPU gating), Phase 8 |
| 4 | Eigenvector sign | SHOWSTOPPER | Phase 3 (canonicalize_signs) | Phases 5/6/7 |
| 5 | NumPy zero-copy | MAJOR | Phase 3 | every PyO3 entry point |
| 6 | GIL deadlock | MAJOR | Phase 3 | Phase 6 (CCSD heaviest), Phase 8 (3.13t CI) |
| 7 | PyO3 subclass override | SHOWSTOPPER | Phase 3 | Phase 4 (DFT large surface) |
| 8 | Loop / F-order layout | MAJOR | Phase 2 + Phase 3 | Phase 7 grad |
| 9 | DIIS path drift | MAJOR | Phase 3 | Phase 6 (amplitude DIIS) |
| 10 | DFT grid weights | MAJOR | Phase 4 | — |
| 11 | chkfile compatibility | SHOWSTOPPER | Phase 3 | every method phase via ORACLE-08 round-trip |
| 12 | Cross-platform libm | MAJOR | Phase 1 | Phase 3 (cross-platform µHartree assertion) |
| 13 | Wheel size / CUDA dist | MAJOR | Phase 8 | — |
| 14 | Panic across FFI | SHOWSTOPPER | Phase 1 (catch_unwind lint) | Phase 3 (panic→exception test) |
| 15 | Sibling-crate cubecl drift | SHOWSTOPPER | Phase 1 (nightly matrix CI) | Phase 8 |
| 16 | Test-oracle global state | MAJOR | Phase 8 (subprocess-per-fixture) | — |
| 17 | Off-by-one basis indexing | MAJOR | Phase 2 | Phase 8 nightly per-basis |
| 18 | Boys-function accuracy | MAJOR | Phase 2 (delegated to cintx) | — |
| 19 | GPU launch overhead small mol | MAJOR | Phase 8 (adaptive dispatch) | — |
| 20 | CCSD memory thrash | MAJOR | Phase 6 (tensor-arena from day one) | — |
| 21 | Scope creep | MAJOR | Phase 1 (forbidden-paths) | every PR in every phase |

## Notes on the 12→8 Phase Compression (Judgment Calls)

The research's SUMMARY.md proposed a 12-phase structure (phases 0–11). Standard granularity targets 5–8. Compressed to 8 with the following deliberate merges and one deliberate non-merge:

- **MERGED `bindings` (research phase 3) into `scf` (phase 3 here)**: the PyO3 contract surface is small enough to lock alongside RHF, and locking it earlier than DFT is the entire point. Five SHOWSTOPPER-tier conventions (subclass dispatch, NumPy stride, GIL deadlock, panic across FFI, abi3 wheel) plus three MAJORs land here on a single small surface — easier than retrofitting onto DFT. The downstream methods inherit by convention with explicit re-validation in DFT (success criterion 4).
- **MERGED `geomopt` (research phase 8) into `grad` (phase 7 here)**: geomopt is small (BFGS+RFO + shims), depends only on grad, and the integration surface (`mf.as_scanner()` + analytical gradient pipeline) is shared. Combining keeps the gradient-driven workflow as one coherent unit.
- **MERGED `GPU enable` + `oracle hardening` + `distribution` (research phases 9+10+11) into one closing phase 8**: all three gate on the same prerequisite (CPU baseline correct on every method) and feed the same exit criterion (a wheel that demonstrably ships on PyPI and proves the 2–5× claim). Splitting them creates artificial boundaries: the benchmark suite (PERF) needs the GPU enable AND the wheel build AND the oracle harness running fairly. One closing phase, three sub-deliverables, all visible in success criteria 1/2/3/4.
- **NOT MERGED `infra` + `gto`** (despite the prompt's suggestion): Phase 1 locks 7 pitfall mitigations (FMA, reduction order, panic policy, cubecl pin, F-order conventions, scope-creep lint, sibling-crate ABI matrix) and produces 2 zero-compute substrate crates whose entire job is to gate later kernels. Phase 2 (gto) is the first kernel landing — a thin wrapper over cintx that exercises those conventions end-to-end on a real basis. Merging would conflate "establish conventions" with "first compute integration" and lose CI signal on the foundation phase.
- **NOT MERGED `dft`** (kept as its own phase, despite being the project's largest): ~9000 LOC upstream surface; integrates all three sibling crates simultaneously; owns the most overrideable methods; peaks cubecl-f64 risk via WGPU. Splitting would help; merging would silently bury the largest single-phase risk.
- **NOT MERGED `ccsd`** (kept as its own phase): largest memory pressure in the project (`Wabef ≈ nv⁴ ≈ 4 GB at caffeine size`); the tensor-arena pattern in `pyscf-runtime` must land here from day one (CCSD-11 explicitly), and combining with a smaller phase would create a dilutional retrofit risk.

The compression preserves clear delivery boundaries while landing inside the standard granularity window.

---
*Roadmap created: 2026-05-10*
*Updated 2026-05-10: added the `pyscf-algebra` crate (workspace 14 → 15), added ALG-01..08 (8 new REQ-IDs, all mapped to Phase 1, total 113 → 121), formalised the `gpu` workspace feature (OFF by default → CPU is default backend) and `PYSCF_BACKEND` env-driven runtime selection. Two new cross-cutting concerns documented (algebra-responsibility wall + backend selection). Reference: `docs/manual/Cubecl/`.*
*Updated 2026-05-10 (gap closure): added Phase 1 plans 01-08 (cintx clean-SHA repin + Cargo.lock commit) and 01-09 (check-cubecl-pin transitive version-skew reconciliation) closing 3 BLOCKERs from 01-VERIFICATION.md. Plan count 7 → 9; wave count 5 → 6. Plans 01-01..01-07 marked shipped.*
*Updated 2026-05-10 (revision iteration 1): per checker feedback, reassigned Plan 01-09 from Wave 6 to Wave 5 (a plan cannot share a wave with a declared dependency; 01-08 depends_on 01-09). Wave 6 retains only 01-08; Wave 5 now contains 01-07 (already shipped) plus 01-09. Total wave count unchanged at 6.*
