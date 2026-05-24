# Phase 6: CCSD - Context

**Gathered:** 2026-05-24
**Status:** Ready for planning

<domain>
## Phase Boundary

A Python user runs `cc.RCCSD(mf).kernel()`, `cc.UCCSD(uhf_mf).kernel()`, the cross-module idiom `mf.CCSD().run()`, and `cc.dfccsd.RCCSD(mf)` on the test corpus and gets upstream CCSD correlation energy to **≤1 µHartree** without OOMing or thrashing the heap. Phase 6 builds the second post-SCF correlation layer on top of the Phase 3 SCF reference and the Phase 5 `pyscf-ao2mo` transform + MP2 helpers. It fills the `pyscf-ccsd` stub crate (currently a 5-line placeholder) and the `pyscf-runtime::WorkspacePool` / `pyscf-core::Amplitudes` skeletons reserved at Phase 1 for exactly this work.

**In scope (11 REQ-IDs):**
- **CCSD-01:** `cc.RCCSD(mf).kernel()` returns CCSD correlation energy matching upstream to ≤1 µHartree; convergence criteria match.
- **CCSD-02:** `cc.UCCSD(uhf_mf).kernel()` matches upstream (open-shell).
- **CCSD-03:** T1 and T2 amplitudes converge to the same minimum (energy is the convergence target; amplitude paths may differ within tolerance).
- **CCSD-04:** Amplitude-DIIS (default `mycc.diis = True`, `mycc.diis_space = 6`) converges within the same iteration count as upstream.
- **CCSD-05:** `mycc.solve_lambda()` produces λ amplitudes for response densities (consumed by Phase-7 CCSD gradients).
- **CCSD-06:** `mycc.make_rdm1()`, `mycc.make_rdm2()` match upstream.
- **CCSD-07:** AO-direct CCSD (`mycc.direct = True`) works.
- **CCSD-08:** DF-CCSD (`mf.density_fit().CCSD()` or `cc.dfccsd.RCCSD(mf)`) works with bounded memory; spills `Wabef` to HDF5 when `PYSCF_MAX_MEMORY` is exceeded.
- **CCSD-09:** T1/D1/D2 diagnostics expose `mycc.t1diagnostic()`, `mycc.d1diagnostic()`.
- **CCSD-10:** Frozen-core options match MP2 (`frozen=int`, `frozen=list`, `frozen='auto'`).
- **CCSD-11:** Tensor-arena/scratchpad pattern in `pyscf-runtime` is in place **from the start** of CCSD work (not retrofitted); `Wabef` and other large intermediates do not allocate-and-drop per iteration; a `PYSCF_MAX_MEMORY` pre-flight refuses to start a job that would exceed the budget rather than OOMing mid-iteration (Pitfall 20).

**Out of scope:**
- **CCSD(T) perturbative triples** (`ccsd_t.py`, `uccsd_t.py`) — deferred v1.x P1 (highest user-pull deferral; STATE.md Deferred Items). Do not implement.
- **EOM-CC / excited states** (`eom_rccsd.py`, `eom_uccsd.py`, `eom_gccsd.py`) — entire response-theory layer, separate milestone.
- **GCCSD / GHF-reference CC** (`gccsd.py`, `gintermediates.py`) — CCSD-EXT-02, v1.x; no v1 REQ maps to it (mirrors the Phase-5 GMP2 exclusion).
- **FNO-CCSD** (frozen-natural-orbital truncation) — CCSD-EXT-01, v1.x.
- **QCISD / BCCD / CCD** (`qcisd.py`, `bccd.py`, `ccd.py`) — sibling methods, no v1 REQ.
- **CCSD gradients (Λ-driven)** — Phase 7 GRAD-06; Phase 6 only *produces* λ via `solve_lambda()`, it does not build gradients.
- **Fused cubecl CCSD kernel** — Phase 8 (the 2–5× owner); Phase 6 contracts via `pyscf-algebra` `gemm`/`oracle_sum` chains (D-03 carried from Phase 5).
- **Per-backend GPU regression / benchmark proof** — Phase 8.

</domain>

<decisions>
## Implementation Decisions

### Tensor-arena & memory model (CCSD-11 — the defining decision of the phase)

- **D-01: Opaque spillable `Tensor` handles from day one + hard PYSCF_MAX_MEMORY refusal.** `t1`/`t2`/`Wabef` (and the other large CCSD intermediates) become **opaque `Tensor` handles** whose storage backend is either an in-memory buffer **or** an HDF5-backed spill file — chosen at allocation time, transparent to the contraction code. This is the most faithful reading of CCSD-11's "tensor-arena from day one, not retrofitted": spill is a **storage-backend swap behind the handle**, not a parallel rewrite of the amplitude-update math. The `PYSCF_MAX_MEMORY` pre-flight (CCSD-11) **HARD-REFUSES** an in-core job that would exceed the budget — it returns a clear `MemoryLimitExceeded`-style error (the `WorkspacePool::try_reserve` contract already shipped at Phase 1) telling the user to opt into DF (`mf.density_fit().CCSD()`) or AO-direct (`mycc.direct=True`) explicitly. **No silent auto-downgrade** — the user makes the memory-vs-accuracy tradeoff deliberately. This deliberately upgrades the current `pyscf-core::Amplitudes { t2: Vec<f64> }` placeholder to the opaque-handle shape (the field comment already anticipates this: "likely as opaque Tensor for spillability").

- **D-08: The spillable `Tensor` abstraction lives in `pyscf-runtime` next to `WorkspacePool`; `pyscf-core::Amplitudes` consumes it.** The Vec-or-HDF5 backend enum + the arena/reuse-pool body of `WorkspacePool::try_reserve`/`reserve`/`release` belong in `pyscf-runtime` (where the Phase-1 skeleton already sits and where backend/memory concerns are owned). `pyscf-core::Amplitudes` holds `Tensor` handles rather than raw `Vec<f64>`. The abstraction is designed to be **reusable by Phase-7 gradient tensors** (λ-response intermediates) and Phase-8 GPU buffers, so it should not bake in CCSD-only assumptions. Algebra wall still applies: `pyscf-runtime` may touch backend/buffer concerns; the *contraction* of these tensors goes through `pyscf-algebra` (D-03), never `cubecl-*` directly from `pyscf-ccsd`.

### ERI-mode scope & sequencing (CCSD-07 / CCSD-08)

- **D-02: In-core RCCSD/UCCSD is the un-gated numeric headline; AO-direct + DF-CCSD-with-HDF5-spill ship as sequenced follow-on plans.** In-core CCSD rides on `mol.intor('int2e')`, which is **real and bit-exact in-tree since plan 05-08** (arity-4 `int2e_{sph,cart}` landed — the cintx#11 gap that gated DF numeric in Phases 3–5 is closed). So the in-core RCCSD/UCCSD energy is achievable and oracle-validated **this phase without any external dependency**, mirroring the Phase-5 MVP sequencing (in-core RMP2 first → UMP2 → DF). The planner sequences: in-core RCCSD (headline) → in-core UCCSD → amplitude-DIIS → λ + RDMs → **AO-direct** (`mycc.direct=True`) → **DF-CCSD + HDF5 spill** as explicit later waves. All three ERI modes ship this phase; they are *ordered*, not co-equal, so the un-gated headline lands and proves out first.

- **D-04: Tiered numeric oracle corpus — small systems bit-exact always-on in-tree; caffeine/cc-pVDZ + benzene-dimer DF-CCSD spill on a CI/human-verify arm.** Always-on in-tree bit-exact gates run on small systems that fit comfortably (H2O/cc-pVDZ, and a water-dimer-sized open-shell/closed-shell pair) so `cargo test` stays fast and green. The ROADMAP's named CCSD-11 memory target — **caffeine/cc-pVDZ** (where `Wabef ≈ nv⁴ ≈ multi-GB) and the **benzene-dimer/cc-pVDZ DF-CCSD spill proof on a deliberately constrained `PYSCF_MAX_MEMORY` budget** (CCSD-08 success criterion) — run as a **CI / human-verify arm** (the 02-10 / 05-08 precedent: the sandbox can't run upstream PySCF / very large jobs, so heavy + upstream-byte-identity assertions are `workflow_dispatch`/human-verify). This honors the user-memory "don't freeze compile / don't freeze the test run" constraint while still proving the arena+spill path in CI.

### Λ-equations & reduced density matrices (CCSD-05 / CCSD-06)

- **D-03: Full numeric λ + make_rdm1/make_rdm2 this phase, INCLUDING `make_rdm2(ao_repr=True)`.** Unlike Phase 5 (where MP2 `make_rdm2(ao_repr=True)` was deferred to Phase 7), CCSD ships the complete RDM surface numerically this phase: `solve_lambda()` produces oracle-validated λ amplitudes, and `make_rdm1`/`make_rdm2` match upstream in MO basis **and** the nmo⁴ AO back-transform (`ao_repr=True`). Rationale: CCSD RDMs/λ are the heaviest consumer of the tensor-arena (D-01) — exercising the full surface here is the natural stress test for CCSD-11, and Phase-7 CCSD gradients (GRAD-06) want a complete, validated λ + RDM surface to build on rather than a half-wired one. Note this is a deliberate scope choice (more memory/compute footprint now, no Phase-7 carry-over for RDMs).

### Upstream port targets (sibling-crate fidelity — confirmed, not re-litigated)

- **D-05: Port `ccsd.CCSD` (in-core RHF real-integral), `uccsd.UCCSD`, and `dfccsd.RCCSD`/`dfuccsd.UCCSD` (DF). NOT the spin-orbital `rccsd.RCCSD`.** `cc.RCCSD(mf)` resolves (per `pyscf/cc/__init__.py:95-121`) to `ccsd.CCSD` for a standard RHF reference, `dfccsd.RCCSD` when `mf` is density-fitted, and `uccsd.UCCSD` for UHF — the separate `rccsd.RCCSD` (complex/spin-orbital module, `cc/rccsd.py`) is **not** the default factory target and is out of scope. DF-CCSD subclasses in-core CCSD and swaps the ERI/`_add_vvvv` source — the exact Phase-5 pattern where `DFRMP2` subclasses `RMP2` (D-06 there). This keeps the oracle/port mapping clean and matches the established sibling-crate-fidelity hard preference.

### PyO3 surface & DIIS (BIND inheritance — Pitfall 7/9 by convention)

- **D-06: Amplitude-DIIS reuses the `pyscf-diis` crate via a new `AmplitudeSubspace: DiisStorable`.** The t1+t2 amplitudes are packed into one error/solution vector and extrapolated through the existing Phase-3 `pyscf-diis` machinery (`solve_linear` B-matrix). This re-validates **Pitfall 9 (DIIS path drift)** on the amplitude vector — the ROADMAP explicitly maps Pitfall 9's re-validation to Phase 6. Default `diis_space=6` (CCSD-04, upstream `ccsd.py` default; note this differs from SCF's `diis_space=8`). No new DIIS implementation — one DIIS, two storables (Fock subspace from Phase 3 + amplitude subspace here).

- **D-07: HDF5 spill reuses the `pyscf-chkfile` re-exported `hdf5` alias — no new `hdf5-metno` dep.** The DF-CCSD `Wabef` spill file (and any outcore AO→MO scratch) uses the Phase-3/4 precedent: the `hdf5` alias re-exported through `pyscf-chkfile` (Phase 4 04-08 explicitly: "the re-exported hdf5 alias, NO own hdf5-metno dep"). The scratch file is the `lib.H5TmpFile()`-equivalent (a temp HDF5 file, deleted on drop). This keeps the dependency graph minimal (user-memory: don't add deps that pull heavy build chains) and reuses a proven seam.

- **D-09: `pyscf-ccsd` stays pyo3-free; `pyscf-py` owns `PyRCCSD`/`PyUCCSD` + `CcsdOverrideHooks` + factory + scanner.** Identical to the Phase-3 D-01 / Phase-5 D-07/D-08 architecture: `pyscf-py` eager-snapshots `mo_coeff`/`mo_energy`/`mo_occ`/`e_hf`/`nocc`/`nmo` from the Python `mf` into plain Rust arrays and passes them into a pyo3-free `pyscf-ccsd` kernel. A `CcsdOverrideHooks` trait (pyo3-free, declared in `pyscf-ccsd`) is bridged in `pyscf-py` via `slf.call_method1(py, "<hook>", …)` so Python subclass overrides dispatch natively (Pitfall 7 immune by construction). Hook set covers what subclasses realistically override: **`ao2mo`** (custom ERI source), **`update_amps`** (the amplitude-equation core), **`make_rdm1`**, **`make_rdm2`**, **`energy`**. `mf.CCSD()` / `mf.density_fit().CCSD()` factory dispatch (RCCSD for RHF, UCCSD for UHF, dfccsd for DF) + `as_scanner` (Phase-7 geomopt seam, mirrors SCF-12/MP2-07). Long compute calls `Python::detach`; the kernel itself does not detach (hooks re-enter Python) — the Phase-5 05-07 pattern.

### Claude's Discretion

The following are not user-decided — researcher/planner picks the implementation within the locked decisions above. Default stance: **mirror upstream** (sibling-crate fidelity).

- **CCSD intermediates** — port `pyscf/cc/rintermediates.py` (`cc_Foo`/`cc_Fvv`/`cc_Fov`/`cc_Woooo`/`cc_Wvvvv`/`cc_Wvoov`/… + `make_tau`) and `uintermediates.py` for the open-shell channels. The `Wvvvv`/`_add_vvvv` contraction (`ccsd.py:362-490`, `_contract_vvvv_t2`, the `'ijcd,acdb->ijab'` einsum) is the largest intermediate and the primary tensor-arena tenant — planner decides the blocking/tiling strategy within the D-01 arena.
- **`update_amps` body** — port `ccsd.py:104` (`update_amps`) closed-shell + `uccsd.py` open-shell; every reduction through `oracle_sum`/`oracle_dot` (no bare `+=`) for bit-exactness + thread-count invariance (the Phase-5 T-05-0x-FP discipline).
- **Init amplitudes / MP2 seed** — `get_init_guess`/`init_amps` (`ccsd.py:1048-1077`) seeds `t1=0`, `t2=(ia|jb)/Dijab` and reports `emp2`; reuses the Phase-5 in-core MP2 path / `pyscf-ao2mo` `ovov` block. The `e_hf + emp2` sanity print mirrors upstream.
- **Convergence defaults** — `max_cycle=50`, `conv_tol=1e-7` (energy), `conv_tol_normt=1e-6` (amplitude norm) per `ccsd.py` `CCSDBase` defaults; planner confirms exact upstream constants.
- **Frozen-core** — reuse the Phase-5 `Frozen` enum + the 5 MP2-08 helpers verbatim (`get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`_mo_without_core`) — already contract-tested in Phase 5 against `cc/ccsd.py:35`. No new frozen logic (CCSD-10).
- **T1/D1/D2 diagnostics** — port `get_t1_diagnostic` / `get_d1_diagnostic` / `get_d2_diagnostic` (upstream `ccsd.py`) — Frobenius-norm-based, mechanical (CCSD-09).
- **AO-direct algorithm** (`mycc.direct=True`) — port the `_contract_vvvv_t2` AO-direct branch (`ccsd.py:473`, "contract t2 to AO-integrals using AO-direct algorithm") that trades the in-memory `vvvv` tensor for on-the-fly AO contraction.
- **DF-CCSD `_add_vvvv` / blocking** — port `dfccsd.py` (`_contract_vvvv_t2` DF branch, the `max_memory`-driven `dmax`/`vvblk` block sizing at `dfccsd.py:76-95,175-185`); the block sizes become arena reservations under D-01.
- **`canonicalize_signs` reuse** — CCSD consumes the SCF reference's already-canonicalized `mo_coeff` (Pitfall 4/12); no new sign work.
- **Phase MVP wave sequencing** — scaffold (`pyscf-ccsd` fill + arena wiring) → in-core RCCSD → UCCSD → amplitude-DIIS → λ + RDMs → AO-direct → DF-CCSD+spill → PyO3 bridge → oracle/CI. Planner finalizes the wave/dependency structure.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project specs (this repo)
- `.planning/PROJECT.md` — vision, core value, key decisions, "out of scope" list; CCSD is an Active v1 requirement; CCSD(T) is the highest-pressure v1.x deferral.
- `.planning/REQUIREMENTS.md` lines 92-104 (CCSD-01..11) + 317-327 (phase mapping) + 196-198 (CCSD-T deferral) — Phase 6 owns 11 REQs.
- `.planning/ROADMAP.md` §"Phase 6: CCSD" (lines 256-269) — goal (caffeine/cc-pVDZ within `PYSCF_MAX_MEMORY`, tensor-arena from day one, AO-direct + DF-CCSD), dependency (Phase 5 — imports MP2 helpers per `cc/ccsd.py:35`), 5 numbered success criteria.
- `.planning/ROADMAP.md` §"Cross-Cutting Concerns" (lines 337-349) — algebra-responsibility wall, backend selection, bit-exact-with-PySCF, PyO3 subclass-override dispatch (Phases 5/6/7 inherit by convention), **`Python::detach` seam — "Phase 6 (CCSD) is the heaviest re-validation under `python3.13t`"**, scope-creep lint, cubecl pin lockstep.
- `.planning/ROADMAP.md` §"Pitfall-to-Phase Mapping" (lines 353-375) — Phase 6 is the **primary phase for Pitfall 20 (CCSD memory thrash → tensor-arena from day one)**; re-validates Pitfall 6 (GIL deadlock — heaviest), Pitfall 9 (DIIS path drift → amplitude DIIS), Pitfall 4 (eigenvector sign, via SCF ref), Pitfall 1/2 (FMA / reduction order in contractions).
- `.planning/STATE.md` §"Blockers/Concerns" — CCSD(T) deferral pressure; `faer`/`hdf5` compat notes; **NOTE: the cintx `int3c2e_sph`/arity-4 `int2e` gap that gated DF numeric in Phases 3-5 is now CLOSED in-tree (plans 05-08 + 05-09)** — DF-CCSD numeric is no longer externally gated, only memory-bounded.
- `.planning/phases/05-mp2/05-CONTEXT.md` — D-01 (`pyscf-ao2mo` own crate, CCSD imports it directly — clean DAG), D-02 (`general`/`full` AO→MO surface CCSD reuses day-one), D-03 (gemm-chain contractions, no new cubecl kernel), D-04 (outcore/semi-incore HDF5 spill **explicitly deferred to Phase 6** — that's now), D-07 (eager SCF snapshot, pyo3-free method crate), D-08 (`Mp2OverrideHooks` bridge — the model for `CcsdOverrideHooks`), MP2-08 helper contract.
- `.planning/phases/04-dft/04-CONTEXT.md` — algebra-orchestrated host-loop precedent (D-07), `PYSCF_DTYPE` f32/f64 seam, the re-exported `hdf5` alias / no-own-hdf5-metno-dep convention (04-08, drives D-07 here), injective `dm_fingerprint` hashing pattern (CR-04) if cache keys are needed.
- `.planning/phases/03-scf-pyo3-bindings/03-CONTEXT.md` — `OverrideHooks` trait-callback bridge (D-01, model for hooks), per-hook `Python::detach` (D-03), NumPy converters (D-04), `pyscf-diis` crate + `DiisStorable`/`FockSubspace` (the amplitude-DIIS template, D-06 here), `pyscf-chkfile` HDF5 schema + alias, `canonicalize_signs` (SCF-13), `as_scanner` shape (SCF-12), test-corpus tiering.

### Upstream PySCF source (this repo — the oracle / port reference)
- `pyscf/cc/__init__.py` (lines 83-139) — `CCSD`/`RCCSD`/`UCCSD`/`GCCSD` factories; **RCCSD resolves to `ccsd.CCSD` (RHF), `dfccsd.RCCSD` (DF), `uccsd.UCCSD` (UHF) — NOT `rccsd.RCCSD`** (D-05).
- `pyscf/cc/ccsd.py` (69 KB — the canonical RHF CCSD) — `kernel` (line 39, imports MP2 helpers at line 35), `update_amps` (104), `energy`, `_add_vvvv`/`_add_vvvv_tril`/`_add_vvvv_full`/`_contract_vvvv_t2` (362-490, the `Wabef`/`vvvv` core + AO-direct branch), `get_init_guess`/`init_amps`/`emp2` (1048-1077), `CCSDBase` (856) + `CCSD` (1261) defaults (`max_cycle`/`conv_tol`/`conv_tol_normt`/`diis_space=6`), `CCSD_Scanner` (827), `_ChemistsERIs` (1389), `t1diagnostic`/`d1diagnostic`. **Primary in-core port target.**
- `pyscf/cc/rintermediates.py` (12 KB) — `cc_Foo`/`cc_Fvv`/`cc_Fov`/`cc_Woooo`/`cc_Wvvvv`/`cc_Wvoov`/`cc_Wovvo`/`make_tau` — the closed-shell intermediates `update_amps` consumes.
- `pyscf/cc/uccsd.py` (58 KB) + `pyscf/cc/uintermediates.py` (38 KB) — `UCCSD` open-shell: α/β/αβ spin-block amplitudes, spin-resolved intermediates + frozen mask + RDMs (CCSD-02). **Open-shell port target.**
- `pyscf/cc/ccsd_lambda.py` (17 KB) + `pyscf/cc/uccsd_lambda.py` (23 KB) — `solve_lambda`, Λ-equation `update_lambda`, l1/l2 intermediates (CCSD-05). **λ port target (consumed by Phase-7 GRAD-06).**
- `pyscf/cc/ccsd_rdm.py` (21 KB) + `pyscf/cc/uccsd_rdm.py` (27 KB) — `make_rdm1`/`make_rdm2`, `_gamma1_intermediates`/`_gamma2_intermediates`, the `ao_repr=True` AO back-transform (CCSD-06, D-03 ships ao_repr this phase).
- `pyscf/cc/dfccsd.py` (9.6 KB) + `pyscf/cc/dfuccsd.py` (13 KB) — `RCCSD(ccsd.CCSD)`/`UCCSD` DF subclasses; `_add_vvvv` DF branch + the `max_memory`-driven `dmax`/`vblk`/`vvblk` block sizing (76-95, 175-185) + `lib.H5TmpFile()` spill (139) — **DF-CCSD + HDF5-spill port target (D-02 follow-on, D-07 spill seam).**
- `pyscf/cc/_ccsd.py` (948 B) — the thin Python wrapper over the C `_ccsd` extension (`_add_vvvv` packing helpers); the Rust port replaces the C calls with `pyscf-algebra` contractions.
- `pyscf/mp/mp2.py` (the 5 helpers `cc/ccsd.py:35` imports: `get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`_mo_without_core`) — **already ported + contract-tested in Phase 5 (MP2-08); CCSD imports them verbatim, no re-port.**
- `pyscf/ao2mo/outcore.py` + `pyscf/ao2mo/semi_incore.py` — the outcore/semi-incore HDF5-spilling AO→MO **deferred from Phase 5 (D-04) to here**; port for the DF-CCSD / AO-direct ERI streaming (D-02 follow-on).

### Phase 1–5 shipped artifacts (this repo)
- `crates/pyscf-ccsd/src/lib.rs` — **the 5-line stub Phase 6 fills** with `CCSD`/`UCCSD`/`DFCCSD` + `CcsdOverrideHooks` + intermediates + λ + RDMs + diagnostics.
- `crates/pyscf-runtime/src/workspace_pool.rs` — **the `WorkspacePool` skeleton (D-08 fills the body)**: `budget_bytes`, `pool: Mutex<Vec<PooledAllocation>>`, `try_reserve` (currently budget-check only, gets the real pool + the `PooledAllocation`→`BufferId` per-backend upgrade), `from_env` (reads `PYSCF_MAX_MEMORY` as MB), `MemoryLimitExceeded` error (the D-01 hard-refusal seam — already wired).
- `crates/pyscf-core/src/amplitudes.rs` — **the `Amplitudes { nocc, nvir, t1: Vec<f64>, t2: Vec<f64> }` skeleton** — D-01 upgrades `t2` (and likely `t1`/intermediates) to opaque `Tensor` handles for spillability (the field comment already anticipates this).
- `crates/pyscf-core/src/traits.rs` — `PostScf` trait (declared Phase 1, MP2 impls it Phase 5) — CCSD implements `PostScf`.
- `crates/pyscf-ao2mo/src/lib.rs` — `general`/`full` AO→MO transform (Phase 5 D-02) — CCSD's `ao2mo` builds the `oovv`/`ovov`/`ovvv`/`vvvv` MO ERI blocks from it; **the outcore/spill surface is added here (Phase 5 D-04 deferral).**
- `crates/pyscf-mp2/src/lib.rs` — the 5 MP2-08 helpers + the in-core MP2 path CCSD's init-amps seed reuses.
- `crates/pyscf-diis/src/lib.rs` — `DiisStorable` trait + `FockSubspace` (Phase 3) — **D-06 adds `AmplitudeSubspace: DiisStorable`** (one DIIS, second storable).
- `crates/pyscf-df/src/lib.rs` — `DfIntegrals`/`cholesky_eri`/`df_metric_fit` (now robust, 05-09)/`DEFAULT_AUXBASIS`/`default_ri` — DF-CCSD reuses the B-tensor assembly.
- `crates/pyscf-chkfile/src/lib.rs` — the re-exported `hdf5` alias + Checkpointable primitives (**D-07 spill reuses this**, no new hdf5 dep).
- `crates/pyscf-algebra/src/lib.rs` — `gemm`/`gemv`/`reduce_sum`/`oracle_sum`/`oracle_dot`/`solve_linear`/`eigh` — **every CCSD contraction + DIIS B-matrix + energy reduction goes through this** (D-03, algebra wall).
- `crates/pyscf-gto/src/lib.rs` — `mol.intor('int2e')` (real bit-exact since 05-08 — the un-gated in-core CCSD ERI source) + `int3c2e_sph`/`int2c2e_sph` (DF-CCSD).
- `crates/pyscf-py/src/lib.rs` — `#[pymodule] _native` + `PyOverrideBridge` + NumPy converters + `Python::detach` seam — **adds `PyRCCSD`/`PyUCCSD`/`PyDFCCSD` + the `cc` submodule + `python/pyscf/cc/__init__.py` overlay + `mf.CCSD()` dispatch** (D-09).
- `crates/pyscf-oracle/` — `oracle_check!` macro — Phase 6 adds CCSD correlation-energy fixtures (small in-tree always-on; caffeine/DF-spill CI-gated, D-04).
- `Cargo.toml` (workspace) — `pyscf-ccsd` already a member (Phase 1); wire its deps (`pyscf-core`, `pyscf-algebra`, `pyscf-ao2mo`, `pyscf-mp2`, `pyscf-scf`, `pyscf-df`, `pyscf-diis`, `pyscf-chkfile`, `pyscf-gto`, `pyscf-runtime`, `tracing`, `thiserror`) — **no pyo3 dep** (D-09).
- `xtask/src/lints/algebra_wall.rs` — extend allowlist for `pyscf-ccsd` (algebra + ao2mo + mp2 + df + diis deps, **no direct cubecl**).
- `.github/workflows/ci.yml` — add CCSD oracle jobs: small-system bit-exact always-on (`ccsd-structural`/`ccsd-oracle`); caffeine + DF-CCSD-spill numeric on a `workflow_dispatch`/human-verify arm (D-04, the 05-08 `mp2-oracle-upstream-manual` precedent); heap-allocation-count assertion for CCSD-11 (Wabef allocated once); **`python3.13t` free-threaded CCSD corpus run** (the ROADMAP "heaviest GIL re-validation" note).

### Sibling-crate / PyO3 precedent (read before implementing analogous surface)
- `~/Documents/workspace/cintx/` — `int2e` (arity-4, shipped) + `int3c2e_sph` (shipped) integral source; cintx#11 is closed for ERIs (05-08).
- `~/Documents/workspace/xcfun_rs/crates/xcfun-py/` + `~/Documents/workspace/cintx/crates/cintx-rs/` — PyO3 0.28 `#[pyclass]`/`#[pymethods]`, `Python::detach`, NumPy boundary patterns (mirror for `PyRCCSD`/`PyUCCSD`).

### Cubecl + numerics reference docs (this repo)
- `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` — authoritative for `pyscf_algebra::gemm` in the CCSD contractions (D-03).
- `docs/manual/Cubecl/Cubecl_multi_ compute.md` — runtime/ComputeClient pattern for any future fused CCSD kernel (Phase 8, deferred).

### External (Phase 6 will look up only if cross-checking the port)
- CCSD method references — the spin-adapted closed-shell CCSD amplitude equations (Hirata/Bartlett or Crawford-Schaefer tutorial), only if the `rintermediates.py` port needs a math cross-check; T1/D1/D2 diagnostic definitions (Lee-Taylor T1, Janssen-Nielsen D1/D2).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`pyscf-runtime::WorkspacePool` skeleton** — `budget_bytes` + `try_reserve` (returns `MemoryLimitExceeded` over budget) + `from_env` (reads `PYSCF_MAX_MEMORY` as MB) already ship; D-01's hard-refusal + D-08's arena body fill the reserved surface (CCSD-11, "not retrofitted").
- **`pyscf-core::Amplitudes { nocc, nvir, t1, t2 }` skeleton** — reserved at Phase 1 for exactly this; D-01 upgrades the `Vec<f64>` fields to opaque spillable `Tensor` handles.
- **`pyscf-ao2mo::general`/`full`** (Phase 5 D-02) — CCSD builds its MO ERI blocks from this; the outcore/spill surface (Phase 5 D-04 deferral) is added here.
- **The 5 MP2-08 helpers** (`pyscf-mp2`, already contract-tested vs `cc/ccsd.py:35`) — CCSD imports verbatim for frozen-core (CCSD-10); zero re-port.
- **In-core MP2 path** (`pyscf-mp2`, 05-03) — CCSD init-amps (`t2 = (ia|jb)/Dijab`, `emp2`) reuses the `ovov` transform + amplitude form.
- **`pyscf-diis` `DiisStorable`/`FockSubspace`** (Phase 3) — the amplitude-DIIS template (D-06): add `AmplitudeSubspace`, reuse `solve_linear` B-matrix; re-validates Pitfall 9.
- **`pyscf-df` `DfIntegrals`/`cholesky_eri`/`df_metric_fit`** (now robust, 05-09) — DF-CCSD reuses the B-tensor (D-02 follow-on); int3c2e is real in-tree.
- **`pyscf-chkfile` re-exported `hdf5` alias** (Phase 3/4) — DF-CCSD `Wabef` spill (D-07), no new dep.
- **`pyscf-algebra` `gemm`/`oracle_sum`/`oracle_dot`/`solve_linear`** — the entire contraction + DIIS + energy-reduction surface (D-03, bit-exact under `release-oracle`).
- **`canonicalize_signs`** (SCF-13) — already applied to the reference `mo_coeff`; CCSD inherits vendor-stable signs.
- **`oracle_check!` macro** (Phase 3) — CCSD correlation-energy fixtures reuse the shape.
- **Phase-5 05-07 PyO3 pattern** — `is_overridden` `__qualname__` base-class check + `slf.call_method1` hook dispatch + eager snapshot + `py.detach` on the kernel body; `PyRCCSD`/`PyUCCSD` follow it.

### Established Patterns
- **Algebra wall** — `pyscf-ccsd` depends on `pyscf-algebra` (+ ao2mo/mp2/df/diis/scf/gto/runtime), **never `cubecl-*` directly**; xtask `algebra_wall` allowlist extended.
- **Sibling-crate fidelity (hard preference)** — `pyscf-ccsd` mirrors `pyscf/cc/{ccsd,uccsd,dfccsd,dfuccsd,ccsd_lambda,uccsd_lambda,ccsd_rdm,uccsd_rdm,rintermediates,uintermediates}.py`; port-don't-reinvent.
- **Method crates stay pyo3-free; bridge in `pyscf-py`** (Phase 3 D-01 → MP2 D-07/D-08 → CCSD D-09).
- **Bit-exact under `release-oracle` via ordered reductions** — every CCSD contraction/energy reduction materializes-then-`oracle_sum` (no bare `+=`), the Phase-5 T-05-0x-FP discipline; thread-count invariant (Pitfall 2).
- **DF numeric is no longer externally gated** — cintx#11 closed for ERIs (05-08/05-09); CCSD numeric is **memory-bounded, not dependency-blocked** (changes the Phase-3/4/5 gating story).
- **In-memory-now-spill-when-budget-exceeded** — D-01 makes spill a backend swap behind the `Tensor` handle (vs. Phase-5 D-04's "in-memory now, spill later"); the "later" is now.
- **`PYSCF_MAX_MEMORY` HARD enforcement here** — Phase 6 is the explicit owner (CCSD-11); Phases 3/4/5 logged-only, Phase 6 pre-flight-refuses (D-01) — the single budget-check authority the Phase-5 D-04 deferred to here.
- **"Don't freeze compile / don't freeze the test run"** (user memory) — port the reference algorithm (no codegen/heavy build.rs); `libxc_rs` stays out of the CCSD dep graph; heavy caffeine/spill tests are CI/human-verify, not always-on (D-04).

### Integration Points
- **`crates/pyscf-ccsd/` (fill the stub)** — deps wired in `Cargo.toml`; ports the upstream `cc/*.py` set; pyo3-free.
- **`crates/pyscf-runtime/src/workspace_pool.rs`** — D-08 fills the arena body + the spillable `Tensor`/backend enum; `Amplitudes` (pyscf-core) consumes it.
- **`crates/pyscf-core/src/amplitudes.rs`** — `Tensor`-handle upgrade (D-01).
- **`crates/pyscf-ao2mo/`** — add the outcore/semi-incore HDF5-spilling AO→MO surface (Phase-5 D-04 deferral) for DF-CCSD / AO-direct.
- **`crates/pyscf-diis/`** — add `AmplitudeSubspace: DiisStorable` (D-06).
- **`crates/pyscf-py/`** — `PyRCCSD`/`PyUCCSD`/`PyDFCCSD` + `cc` submodule + `python/pyscf/cc/__init__.py` overlay + `mf.CCSD()`/`mf.density_fit().CCSD()` dispatch + `as_scanner`; reuse NumPy converters + `Python::detach` (D-09).
- **`Cargo.toml` workspace** — wire `pyscf-ccsd` deps (member already registered Phase 1; no member-count change).
- **`xtask/src/lints/algebra_wall.rs`** — `pyscf-ccsd` allowlist entry.
- **`.github/workflows/ci.yml`** — CCSD small-system oracle (always-on) + caffeine/DF-spill (human-verify) + heap-allocation-count assertion (CCSD-11) + `python3.13t` corpus run (heaviest GIL re-validation).

</code_context>

<specifics>
## Specific Ideas

- **CCSD-11 is the defining decision and the whole point of keeping CCSD its own phase** (ROADMAP §"Notes on the 12→8 Compression": "largest memory pressure in the project — `Wabef ≈ nv⁴ ≈ 4 GB at caffeine size`; the tensor-arena pattern must land here from day one"). D-01 (opaque spillable `Tensor` + hard refuse) takes the most-faithful reading: spill is a storage-backend swap, not a retrofit.
- **In-core RCCSD is the un-gated headline** — it rides on `int2e` (real bit-exact since 05-08). Unlike Phases 3-5, **no external cintx gate** stands between Phase 6 and a numeric CCSD energy; the only constraint is memory (D-02/D-04).
- **DF-CCSD subclasses in-core CCSD and swaps the ERI source** (`dfccsd.RCCSD(ccsd.CCSD)`) — exactly the Phase-5 `DFRMP2(RMP2)` pattern, so the DF path is "swap `_add_vvvv`/ERIs + add spill," not a parallel rewrite (D-05).
- **Full RDM surface (incl. ao_repr) ships this phase** (D-03) — a deliberate departure from Phase-5's ao_repr deferral, because CCSD RDMs are the heaviest arena tenant (best CCSD-11 stress test) and Phase-7 gradients want a complete λ+RDM surface.
- **One DIIS, two storables** — amplitude-DIIS (D-06) reuses `pyscf-diis` with an `AmplitudeSubspace`, re-validating Pitfall 9 on the amplitude vector (the ROADMAP's mapped Phase-6 DIIS re-validation).
- **Phase 6 is the heaviest `python3.13t` / GIL re-validation** (ROADMAP cross-cutting note) — the PyO3 bridge (D-09) must hold the `Python::detach` discipline under the longest compute in the project.
- **CCSD(T) deferral is real and expected** (STATE.md: ~30-40% of CCSD users want it; v1.x P1) — explicitly NOT in this phase; capture any (T) pull as a deferred idea.

</specifics>

<deferred>
## Deferred Ideas

- **CCSD(T) perturbative triples** (`cc/ccsd_t.py`, `uccsd_t.py`, `ccsd_t_rdm.py`, `ccsd_t_lambda.py`) — v1.x P1 (CCSD-T-01); highest user-pull deferral. Out of Phase 6.
- **EOM-CC excited states** (`eom_rccsd.py`, `eom_uccsd.py`, `eom_gccsd.py`) — separate milestone (response-theory layer, PROJECT.md Out of Scope).
- **GCCSD / GHF-reference CC** (`gccsd.py`, `gintermediates.py`, `gccsd_lambda.py`, `gccsd_rdm.py`) — CCSD-EXT-02, v1.x; no v1 REQ maps (mirrors Phase-5 GMP2 exclusion).
- **FNO-CCSD** (frozen-natural-orbital truncation, `cc/addons.py` `make_fno`) — CCSD-EXT-01, v1.x.
- **QCISD / BCCD / CCD** (`qcisd.py`, `bccd.py`, `ccd.py`) — sibling methods, no v1 REQ.
- **Fused cubecl CCSD kernel** — Phase 8 (the 2–5× + GPU owner), only if profiling shows the `pyscf-algebra` `gemm`/`oracle_sum` chain (D-03) is the bottleneck.
- **CCSD analytical gradients** (Λ-driven) — Phase 7 GRAD-06; Phase 6 produces λ (D-03/CCSD-05), Phase 7 consumes it.
- **Higher-order CC (CC3, CCSDT, CCSDTQ)** (`rccsdt.py`, `rccsdtq.py`, etc.) — research-grade, out of v1 entirely (PROJECT.md Out of Scope).

### Reviewed Todos (not folded)
None — the todo cross-reference scan returned 0 matches for Phase 6.

</deferred>

---

*Phase: 06-ccsd*
*Context gathered: 2026-05-24*
