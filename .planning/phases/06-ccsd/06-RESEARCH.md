# Phase 6: CCSD - Research

**Researched:** 2026-05-24
**Domain:** Coupled-cluster singles-doubles (RCCSD/UCCSD/DF-CCSD) port + spillable tensor-arena + amplitude-DIIS + PyO3 bridge in a Rust port of PySCF
**Confidence:** HIGH (codebase verified file-by-file; upstream port targets read directly in-tree; only the heap-alloc-assertion tooling and hdf5 spill-write ergonomics are MEDIUM)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 (Tensor-arena & PYSCF_MAX_MEMORY refusal):** `t1`/`t2`/`Wabef` and the other large CCSD intermediates become **opaque `Tensor` handles** whose storage backend is in-memory buffer OR HDF5 spill file, chosen at allocation time, transparent to contraction code. Spill is a **storage-backend swap behind the handle**, not a parallel rewrite. The `PYSCF_MAX_MEMORY` pre-flight **HARD-REFUSES** an in-core job that would exceed budget — returns `MemoryLimitExceeded`-style error telling the user to opt into DF (`mf.density_fit().CCSD()`) or AO-direct (`mycc.direct=True`). **No silent auto-downgrade.** Upgrades `pyscf-core::Amplitudes { t2: Vec<f64> }` to the opaque-handle shape.
- **D-08 (arena lives in `pyscf-runtime`):** The Vec-or-HDF5 backend enum + the arena/reuse-pool body of `WorkspacePool::try_reserve`/`reserve`/`release` belong in `pyscf-runtime`. `pyscf-core::Amplitudes` holds `Tensor` handles, not raw `Vec<f64>`. Designed reusable by Phase-7 gradient tensors + Phase-8 GPU buffers — no CCSD-only assumptions. Algebra wall still applies: `pyscf-runtime` may touch backend/buffer concerns; the *contraction* goes through `pyscf-algebra` (D-03), never `cubecl-*` directly from `pyscf-ccsd`.
- **D-02 (in-core headline, sequenced follow-ons):** In-core RCCSD/UCCSD is the un-gated numeric headline (rides on `mol.intor('int2e')`, real bit-exact since plan 05-08). Sequence: in-core RCCSD → in-core UCCSD → amplitude-DIIS → λ + RDMs → **AO-direct** → **DF-CCSD + HDF5 spill** as explicit later waves. All three ERI modes ship this phase; they are *ordered*, not co-equal.
- **D-04 (tiered oracle corpus):** Always-on in-tree bit-exact gates on small systems (H2O/cc-pVDZ + a water-dimer-sized open/closed-shell pair). The caffeine/cc-pVDZ memory target (`Wabef ≈ nv⁴ ≈ multi-GB`) and the benzene-dimer/cc-pVDZ DF-CCSD spill proof run as a **CI / human-verify (`workflow_dispatch`) arm** (the 02-10 / 05-08 precedent).
- **D-03 (full λ + RDMs incl. `ao_repr=True`):** CCSD ships the complete RDM surface numerically THIS phase: `solve_lambda()` produces oracle-validated λ; `make_rdm1`/`make_rdm2` match upstream in MO basis AND the nmo⁴ AO back-transform (`ao_repr=True`). Deliberate scope choice (no Phase-7 carry-over for RDMs).
- **D-05 (port targets):** Port `ccsd.CCSD` (in-core RHF real-integral), `uccsd.UCCSD`, and `dfccsd.RCCSD`/`dfuccsd.UCCSD` (DF). **NOT** the spin-orbital `rccsd.RCCSD`. DF-CCSD subclasses in-core CCSD and swaps the ERI/`_add_vvvv` source — the Phase-5 `DFRMP2(RMP2)` pattern.
- **D-06 (amplitude-DIIS):** Reuses `pyscf-diis` via a new `AmplitudeSubspace: DiisStorable`. t1+t2 packed into one error/solution vector, extrapolated through the existing `solve_linear` B-matrix. Re-validates Pitfall 9. Default `diis_space=6` (note: differs from SCF's `diis_space=8`). One DIIS, two storables.
- **D-07 (HDF5 spill reuses `pyscf-chkfile` `hdf5` alias):** No new `hdf5-metno` dep. The `Wabef` spill file (+ any outcore AO→MO scratch) uses the re-exported `hdf5` alias through `pyscf-chkfile`. Scratch file is the `lib.H5TmpFile()`-equivalent (temp HDF5, deleted on drop).
- **D-09 (`pyscf-ccsd` pyo3-free; `pyscf-py` owns the bridge):** `pyscf-py` eager-snapshots `mo_coeff`/`mo_energy`/`mo_occ`/`e_hf`/`nocc`/`nmo` into plain Rust arrays. A `CcsdOverrideHooks` trait (pyo3-free, in `pyscf-ccsd`) is bridged in `pyscf-py` via `slf.call_method1(py, "<hook>", …)`. Hooks: **`ao2mo`**, **`update_amps`**, **`make_rdm1`**, **`make_rdm2`**, **`energy`**. `mf.CCSD()` / `mf.density_fit().CCSD()` factory dispatch (RCCSD for RHF, UCCSD for UHF, dfccsd for DF) + `as_scanner`. Long compute calls `Python::detach`; the kernel itself does NOT detach (hooks re-enter Python).

### Claude's Discretion (researcher/planner picks within the locked decisions; default = mirror upstream)

- **CCSD intermediates** — port `rintermediates.py` (`cc_Foo`/`cc_Fvv`/`cc_Fov`/`cc_Woooo`/`cc_Wvvvv`/`cc_Wvoov`/`cc_Wovvo`/`make_tau`) + `uintermediates.py`. The `Wvvvv`/`_add_vvvv` `'ijcd,acdb->ijab'` contraction is the largest intermediate + primary arena tenant — planner decides blocking/tiling within the D-01 arena.
- **`update_amps` body** — port `ccsd.py:104` closed-shell + `uccsd.py` open-shell; every reduction through `oracle_sum`/`oracle_dot` (no bare `+=`).
- **Init amplitudes / MP2 seed** — `init_amps` (`ccsd.py:1048-1077`): `t1=0`, `t2=(ia|jb)/Dijab`, report `emp2`; reuse Phase-5 in-core MP2 path / `pyscf-ao2mo` `ovov` block. The `e_hf + emp2` sanity print mirrors upstream.
- **Convergence defaults** — `max_cycle=50`, `conv_tol=1e-7` (energy), `conv_tol_normt=1e-6` (amplitude norm) — planner confirms exact upstream constants (VERIFIED below).
- **Frozen-core** — reuse the Phase-5 `Frozen` enum + 5 MP2-08 helpers verbatim. No new frozen logic.
- **T1/D1/D2 diagnostics** — port `get_t1_diagnostic`/`get_d1_diagnostic`/`get_d2_diagnostic` — Frobenius-norm-based.
- **AO-direct algorithm** (`mycc.direct=True`) — port the `_contract_vvvv_t2` AO-direct branch.
- **DF-CCSD `_add_vvvv` / blocking** — port `dfccsd.py` block sizing (`dmax`/`vvblk` @76-95,175-185); block sizes become arena reservations under D-01.
- **`canonicalize_signs` reuse** — CCSD consumes the SCF reference's already-canonicalized `mo_coeff`; no new sign work.
- **Phase MVP wave sequencing** — scaffold → in-core RCCSD → UCCSD → amplitude-DIIS → λ + RDMs → AO-direct → DF-CCSD+spill → PyO3 bridge → oracle/CI. Planner finalizes.

### Deferred Ideas (OUT OF SCOPE)

- **CCSD(T) perturbative triples** (`ccsd_t.py`, `uccsd_t.py`, `ccsd_t_rdm.py`, `ccsd_t_lambda.py`) — v1.x P1; highest user-pull deferral.
- **EOM-CC excited states** (`eom_rccsd.py`, `eom_uccsd.py`, `eom_gccsd.py`) — separate milestone.
- **GCCSD / GHF-reference CC** (`gccsd.py`, `gintermediates.py`, `gccsd_lambda.py`, `gccsd_rdm.py`) — CCSD-EXT-02, v1.x.
- **FNO-CCSD** (frozen-natural-orbital truncation, `cc/addons.py:make_fno`) — CCSD-EXT-01, v1.x.
- **QCISD / BCCD / CCD** (`qcisd.py`, `bccd.py`, `ccd.py`) — sibling methods, no v1 REQ.
- **Fused cubecl CCSD kernel** — Phase 8 (the 2–5×/GPU owner).
- **CCSD analytical gradients** (Λ-driven) — Phase 7 GRAD-06; Phase 6 produces λ, Phase 7 consumes it.
- **Higher-order CC (CC3, CCSDT, CCSDTQ)** — research-grade, out of v1.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CCSD-01 | `cc.RCCSD(mf).kernel()` → CCSD corr. energy ≤1 µHartree; convergence criteria match | Upstream `ccsd.py:kernel` @44 + `update_amps` @104 + `energy` @710 ported via host-loop `oracle_sum`/`oracle_dot` discipline (gemm is a stub — §"Don't Hand-Roll"); MP2 seed via `init_amps` @1048 reuses `pyscf-ao2mo` `ovov`. |
| CCSD-02 | `cc.UCCSD(uhf_mf).kernel()` matches upstream (open-shell) | `uccsd.py`+`uintermediates.py` α/β/αβ spin blocks; mirrors the Phase-5 `UmpReference { alpha, beta }` two-channel snapshot shape. |
| CCSD-03 | T1/T2 converge to same minimum (energy is the convergence target) | Energy `conv_tol` + amplitude-norm `conv_tol_normt` dual criterion (`ccsd.py:97`); oracle on energy not amplitude paths. |
| CCSD-04 | Amplitude-DIIS (`diis=True`, `diis_space=6`) converges within same iteration count | `AmplitudeSubspace: DiisStorable` (§Pattern 5) reusing `pyscf-diis::Diis<S>`; `amplitudes_to_vector` @670 defines the flat packing. |
| CCSD-05 | `mycc.solve_lambda()` produces λ amplitudes | Port `ccsd_lambda.py:kernel`+`update_lambda`; base `CCSDBase.solve_lambda` raises NotImplementedError, the concrete `CCSD.solve_lambda` @1273 dispatches to `ccsd_lambda`. |
| CCSD-06 | `make_rdm1`/`make_rdm2` match upstream (incl. `ao_repr=True`) | Port `ccsd_rdm.py` `_gamma1`/`_gamma2_intermediates` + the nmo⁴ AO back-transform via `pyscf-ao2mo`. |
| CCSD-07 | AO-direct CCSD (`mycc.direct=True`) | Port `_add_vvvv_tril`/`_contract_s4vvvv_t2` AO-direct branch @406-570; on-the-fly `int2e` shell-pair contraction instead of in-memory vvvv. |
| CCSD-08 | DF-CCSD spills `Wabef` to HDF5 when budget exceeded | `dfccsd.py` `vvL` DF B-tensor + `dmax`/`vvblk` block sizing @76-95,139,175-185; spill via `pyscf-chkfile::hdf5` alias (D-07). |
| CCSD-09 | `t1diagnostic()`, `d1diagnostic()` | Port `get_t1_diagnostic`/`get_d1_diagnostic`/`get_d2_diagnostic` @748-776 — Frobenius norm + eigh of t1·t1ᵀ. |
| CCSD-10 | Frozen-core (`frozen=int/list/'auto'`) matches MP2 | Reuse Phase-5 `pyscf_mp2::Frozen` + the 5 MP2-08 helpers verbatim (already contract-tested vs `cc/ccsd.py:35`). |
| CCSD-11 | Tensor-arena from day one; no allocate-and-drop per iteration; pre-flight refusal | The DEFINING REQ. `WorkspacePool` arena body (D-08) + opaque `Tensor` `Amplitudes` (D-01) + `MemoryLimitExceeded` pre-flight + CI heap-alloc-count assertion (§Validation). |
</phase_requirements>

## Summary

Phase 6 is a **port-don't-reinvent** effort layered on a fully-established scaffold. Every architectural seam CCSD needs already ships and is verified in-tree: the MP2 kernel/hooks/bridge pattern (`pyscf-mp2` + `pyscf-py::mp`), the `pyscf-ao2mo` general/full transform, the `pyscf-diis::Diis<S>` generic CDIIS with its `DiisStorable` trait (whose doc-comment already names the future `pyscf-ccsd::AmpsSubspace`), the `WorkspacePool` skeleton with its `MemoryLimitExceeded` error wired, the `Amplitudes` skeleton anticipating opaque handles, the `pyscf-chkfile` `hdf5` alias, and the `Frozen` + 5 helpers. The dominant engineering work is (1) the `update_amps`/intermediates port with strict `oracle_sum`/`oracle_dot` reduction discipline, and (2) filling the `WorkspacePool` arena body so the `Wabef ≈ nv⁴` intermediate allocates once and reuses (CCSD-11).

**One finding overturns a CONTEXT.md assumption and must shape the plan:** `pyscf_algebra::gemm` is STILL a `NotYetImplemented{phase:2}` stub (verified `crates/pyscf-algebra/src/gemm.rs:16`). Both `pyscf-ao2mo::transform` and `pyscf-mp2::mp2` therefore do ALL contractions as explicit host loops with materialize-then-`oracle_sum` — never via Tensor `gemm`. CCSD's `'ijcd,acdb->ijab'` vvvv contraction (the heaviest FLOP in the project) must follow the SAME host-loop discipline, NOT a `gemm`-chain. The CONTEXT.md D-03 phrasing "contracts via `pyscf-algebra` gemm/oracle_sum chains" should be read as **`oracle_sum`/`oracle_dot` chains** for v1; a real `gemm` is Phase-8 territory. This is the single largest correctness-and-performance risk in the phase and the planner must size waves around host-loop contraction, not BLAS.

**Second finding:** the dependency wall is a **denylist** (`xtask/src/bin/check_dependency_wall.rs`), not an allowlist. Any `pyscf-*` crate that names `cubecl-*` directly fails automatically; `pyscf-ccsd` needs **no allowlist entry** (CONTEXT.md "extend allowlist for pyscf-ccsd" is inaccurate — there is nothing to add; the wall already covers it). Similarly `check-no-fma` is already scoped to `pyscf_*`-owned symbols, so faer/pulp SIMD FMA in the contraction backends is exempt.

**Primary recommendation:** Land the `WorkspacePool` arena body + opaque `Tensor`-handle `Amplitudes` FIRST (Wave 1, before any CCSD math), exactly as CCSD-11/D-01/D-08 demand ("from day one, not retrofitted"). Then port the in-core RCCSD `update_amps`/`energy`/`init_amps`/intermediates as host loops with `oracle_sum`/`oracle_dot`, validate the un-gated energy headline against a small-system oracle, and only then layer UCCSD → amplitude-DIIS → λ+RDM → AO-direct → DF+spill → PyO3 in sequenced follow-on waves.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Tensor-arena / buffer pool / spill-backend swap | `pyscf-runtime` (`WorkspacePool`) | `pyscf-chkfile` (hdf5 alias for spill) | D-08: backend/memory concerns live where the Phase-1 skeleton sits; spill file is a `H5TmpFile`-equivalent owned by the sole hdf5 owner. |
| `Amplitudes` (t1/t2/intermediate handles) | `pyscf-core` | `pyscf-runtime` (allocates via pool) | D-01: `pyscf-core::Amplitudes` *consumes* `Tensor` handles; pool *produces* them. |
| CCSD amplitude equations (`update_amps`, intermediates, λ, RDM, diagnostics) | `pyscf-ccsd` (pyo3-free) | `pyscf-algebra` (reductions) | Sibling-crate fidelity; algebra wall — contractions go through algebra's `oracle_sum`/`oracle_dot`. |
| AO→MO ERI block assembly (`oovv`/`ovov`/`ovvv`/`vvvv` + outcore) | `pyscf-ao2mo` | `pyscf-gto` (`int2e`), `pyscf-df` (B-tensor) | Phase-5 D-01/D-02 keystone; CCSD's `ao2mo` hook builds blocks from `general`/`full`. The outcore/semi-incore surface is ADDED here (Phase-5 D-04 deferral). |
| Amplitude-DIIS extrapolation | `pyscf-diis` (`AmplitudeSubspace`) | `pyscf-algebra` (`solve_linear`, `oracle_dot`) | D-06: one DIIS, second storable; B-matrix solve already shipped. |
| PyO3 surface / factory / scanner / override dispatch / `Python::detach` | `pyscf-py` (`cc` submodule) | `pyscf-ccsd` (`CcsdOverrideHooks` trait decl) | D-09: method crate pyo3-free; bridge owns marshalling + GIL discipline. |
| `mf.CCSD()` / `mf.density_fit().CCSD()` cross-module dispatch | `pyscf-py` + `python/pyscf/cc/__init__.py` overlay | — | Mirrors `cc/__init__.py:83-139` (RHF→CCSD, UHF→UCCSD, DF→dfccsd). |
| Memory pre-flight refusal (`MemoryLimitExceeded`) | `pyscf-runtime` (`try_reserve`) | `pyscf-ccsd` (calls before allocating vvvv) | CCSD-11/D-01: single budget-check authority; Phases 3/4/5 were log-only. |

## Standard Stack

This phase adds **no new external crates**. Every dependency already exists in the workspace and is verified present. CCSD-11/D-08's "don't add deps that pull heavy build chains" and the `libxc_rs` ~6h compile constraint are honored by reuse.

### Core (workspace-internal, all verified present)
| Crate | Role for CCSD | Why Standard |
|-------|---------------|--------------|
| `pyscf-ccsd` | The crate to fill (currently a 5-line stub: `#![forbid(unsafe_code)]`) | The phase deliverable. |
| `pyscf-runtime` | `WorkspacePool` arena body (D-08); `BackendError::MemoryLimitExceeded` already defined | Owns backend/memory; ALG-06 carve-out lets it touch cubecl/buffers. |
| `pyscf-core` | `Amplitudes` handle upgrade; `Mole`, `MOCoefficients`, `PostScf` trait, `PyscfRsError` | Shared types; `PostScf` declared Phase 1, CCSD impls it. |
| `pyscf-algebra` | `oracle_sum`/`oracle_dot`/`solve_linear`/`eigh` — ALL contractions + energy reductions + DIIS B-matrix | Bit-exact under `release-oracle`; **NOTE: `gemm` is a stub — use host loops.** |
| `pyscf-ao2mo` | `general`/`full` AO→MO; CCSD `ao2mo` hook builds `oovv`/`ovov`/`ovvv`/`vvvv`; outcore surface added here | Phase-5 D-01 keystone, clean DAG (no ccsd→mp2 edge). |
| `pyscf-mp2` | The 5 MP2-08 frozen helpers + `Frozen` enum + in-core MP2 path for `init_amps` seed | Already contract-tested vs `cc/ccsd.py:35`; zero re-port. |
| `pyscf-diis` | `Diis<S>` + `DiisStorable`; add `AmplitudeSubspace` | One DIIS, two storables (D-06). |
| `pyscf-df` | `DfIntegrals`/`cholesky_eri`/`df_metric_fit` (robust since 05-09)/`default_ri` for DF-CCSD `vvL` | DF numeric un-gated now (cintx#11 closed). |
| `pyscf-chkfile` | Re-exported `hdf5` alias (`pub use hdf5_metno as hdf5;`) for the `Wabef` spill temp file | Sole hdf5 owner (D-05/D-07); no new dep. |
| `pyscf-gto` | `intor("int2e")` (in-core, bit-exact since 05-08) + `int3c2e_sph`/`int2c2e_sph` (DF) | Un-gated in-core ERI source. |

### Supporting
| Crate | Role | When Used |
|-------|------|-----------|
| `pyscf-scf` | `ScfResult` reference types (CCSD snapshots from these) | Snapshot in `pyscf-py`, not a runtime dep of the kernel. |
| `pyscf-py` | `PyRCCSD`/`PyUCCSD`/`PyDFCCSD` + `cc` submodule + bridge + factory + scanner | The ONLY pyo3 layer (D-09). |
| `pyscf-oracle` | `oracle_check!` macro for correlation-energy fixtures | Small-system always-on + caffeine/DF human-verify (D-04). |
| `tracing`, `thiserror` | Logging + error enum (`CcsdError`) | Standard across all method crates. |

### Cargo.toml deps to wire for `pyscf-ccsd` (member already registered Phase 1)
```toml
[dependencies]
pyscf-core    = { workspace = true }
pyscf-algebra = { workspace = true }
pyscf-ao2mo   = { workspace = true }
pyscf-mp2     = { workspace = true }
pyscf-scf     = { workspace = true }
pyscf-df      = { workspace = true }
pyscf-diis    = { workspace = true }
pyscf-chkfile = { workspace = true }
pyscf-gto     = { workspace = true }
pyscf-runtime = { workspace = true }
tracing       = { workspace = true }
thiserror     = { workspace = true }
# NO pyo3 dep (D-09). NO cubecl-* dep (algebra wall — denylist auto-enforced).
```
**Verify after wiring:** `cargo tree -p pyscf-ccsd | grep -i libxc` must return nothing (the ~6h-compile crate stays out of the graph — `libxc_rs` is only reachable via `pyscf-dft`'s opt-in `libxc` feature, and `pyscf-ccsd` does not depend on `pyscf-dft`). [VERIFIED: Cargo.toml workspace comment lines 114-118 + dep set above]

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Host-loop `oracle_sum` vvvv contraction | Real `pyscf_algebra::gemm` (cubecl matmul) | gemm is a `phase:2` stub; wiring it is a deferred Phase-8 task and risks FMA/reduction-order drift. Host loops are bit-exact and the established Phase-5 precedent. |
| New `hdf5-metno` dep in `pyscf-ccsd` | `pyscf-chkfile::hdf5` re-export | D-07 forbids a second hdf5 owner; the alias is proven (Phase 3/4). |
| In-memory-only vvvv with OOM | Opaque `Tensor` handle + pre-flight refusal + DF/AO-direct opt-in | D-01: no silent OOM; the whole point of keeping CCSD its own phase. |

## Package Legitimacy Audit

> No external packages are installed in this phase. All dependencies are workspace-internal crates already present and built. The hdf5-metno transitive dep is owned solely by `pyscf-chkfile` and was vetted in Phase 3/4. slopcheck is not applicable (no registry install).

| Package | Registry | Disposition |
|---------|----------|-------------|
| (none — workspace-internal only) | — | N/A |

## Architecture Patterns

### System Architecture Diagram

```
  Python: cc.RCCSD(mf).kernel()  /  mf.CCSD().run()  /  mf.density_fit().CCSD()
       │
       ▼
  ┌─────────────────────────── pyscf-py :: cc submodule (the ONLY pyo3 layer) ───────────────┐
  │  factory dispatch (cc/__init__.py overlay): RHF→PyRCCSD, UHF→PyUCCSD, DFHF→PyDFCCSD       │
  │  PyRCCSD.new(mf): eager-snapshot mo_coeff/mo_energy/mo_occ/e_hf/mol → CcsdReference        │
  │  holds Py<PyAny> mf (for as_scanner re-run + hook MRO dispatch)                            │
  │                                                                                            │
  │  CcsdPyBridge (impl CcsdOverrideHooks):                                                    │
  │    is_overridden(slf,"update_amps"/"ao2mo"/"make_rdm1"/"make_rdm2"/"energy")?              │
  │       ├─ YES → slf.call_method1(py,"<hook>",args)   (Pitfall 7 MRO; re-enters Python)      │
  │       └─ NO  → py.detach(|| <pure-Rust default>)    (BIND-05; kernel does NOT detach)      │
  └─────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                             │ plain Rust arrays (pyo3-free boundary, D-09)
                                             ▼
  ┌──────────────────────────── pyscf-ccsd (pyo3-free, cubecl-free) ───────────────────────────┐
  │  ccsd_kernel(refr, frozen, hooks, pool):                                                    │
  │    1. pre-flight: pool.try_reserve(estimate_bytes(nocc,nvir)) → MemoryLimitExceeded? REFUSE │
  │    2. init_amps: t1=0, t2=(ia|jb)/Dijab via pyscf-ao2mo ovov  → emp2                        │
  │    3. for istep in 0..max_cycle:                                                            │
  │         t1new,t2new = update_amps(t1,t2,eris)   ◄── intermediates (cc_Foo/Fvv/.../Wvvvv)    │
  │            │  vvvv contraction 'ijcd,acdb->ijab' ── PRIMARY ARENA TENANT (Wabef≈nv⁴)        │
  │            │  ALL reductions: materialize→oracle_sum/oracle_dot (NO +=, NO gemm)            │
  │         normt = ‖amplitudes_to_vector(new) − vector(old)‖                                   │
  │         t1,t2 = run_diis(...)  ──► pyscf-diis::Diis<AmplitudeSubspace>                       │
  │         eccsd = energy(t1,t2,eris); converged if |dE|<conv_tol AND normt<conv_tol_normt     │
  │    4. (opt) solve_lambda → l1,l2 ;  make_rdm1/make_rdm2 (+ao_repr back-transform)           │
  └───────┬──────────────────────────────────────────┬────────────────────────┬───────────────┘
          │                                          │                        │
          ▼                                          ▼                        ▼
  pyscf-runtime::WorkspacePool          pyscf-ao2mo (oovv/ovov/        pyscf-algebra
  (arena: reserve/release/reuse;         ovvv/vvvv blocks;             (oracle_sum/oracle_dot/
   Vec-backend OR HDF5-spill-backend     in-core int2e OR DF vvL OR    solve_linear/eigh —
   behind opaque Tensor handle;          AO-direct streaming)          gemm is a STUB)
   spill via pyscf-chkfile::hdf5)               │
                                                ▼
                                  pyscf-gto::intor("int2e")  /  pyscf-df::DfIntegrals (vvL)
```
Trace the un-gated headline: Python `kernel()` → snapshot → `ccsd_kernel` pre-flight reserve → `init_amps` (MP2 seed) → iterate `update_amps`+DIIS+`energy` → converged `e_corr`. The arena/spill swap is invisible to `update_amps`; only the allocation site (pool) and the storage backend differ.

### Recommended `pyscf-ccsd` module structure (mirror upstream file split — sibling-crate fidelity)
```
crates/pyscf-ccsd/src/
├── lib.rs              # pub re-exports; CcsdError; #![forbid(unsafe_code)]
├── error.rs            # CcsdError enum + From<CcsdError> for PyscfRsError (mirror mp2/error.rs)
├── eris.rs             # ChemistsEris (oovv/ovov/ovvv/oooo/ovoo/ovvo + fock/mo_energy); _common_init_
├── hooks.rs            # CcsdOverrideHooks trait + NoCcsdOverrides default (mirror mp2/hooks.rs)
├── reference.rs        # CcsdReference snapshot (mirror Mp2Reference)
├── ccsd.rs             # kernel loop, init_amps/emp2, energy, defaults (mirror ccsd.py)
├── rintermediates.rs   # cc_Foo/Fvv/Fov/Woooo/Wvvvv/Wvoov/Wovvo/make_tau (port rintermediates.py)
├── update_amps.rs      # update_amps closed-shell (port ccsd.py:104) + _add_vvvv/_contract_vvvv_t2
├── uccsd.rs            # UCCSD α/β/αβ (port uccsd.py)
├── uintermediates.rs   # spin-resolved intermediates (port uintermediates.py)
├── diis_amps.rs        # AmplitudeSubspace: DiisStorable (D-06); amplitudes_to_vector/vector_to_amplitudes
├── lambda.rs           # solve_lambda / update_lambda (port ccsd_lambda.py)
├── ulambda.rs          # open-shell λ (port uccsd_lambda.py)
├── rdm.rs              # make_rdm1/make_rdm2 + _gamma1/_gamma2 + ao_repr (port ccsd_rdm.py)
├── urdm.rs             # open-shell RDM (port uccsd_rdm.py)
├── diagnostics.rs      # t1/d1/d2 diagnostics (port ccsd.py:748-776)
├── dfccsd.rs           # DFRCCSD/DFUCCSD ERI swap + vvL block sizing + HDF5 spill (port dfccsd.py)
└── direct.rs           # AO-direct _contract_s4vvvv_t2 branch (port ccsd.py:487-570)
```

### Pattern 1: `WorkspacePool` arena body + opaque spillable `Tensor` (CCSD-11, D-01/D-08) — the defining work
**What:** Upgrade the Phase-1 budget-check-only `try_reserve` into a real reuse pool with a backend enum.
**When:** Wave 1, BEFORE any CCSD math (the "from day one, not retrofitted" mandate).
**Current state (verified):** `crates/pyscf-runtime/src/workspace_pool.rs` has `budget_bytes`, `pool: Mutex<Vec<PooledAllocation>>` (`PooledAllocation { _bytes: Box<[u8]>, _size }` — a stub), `from_env` (reads `PYSCF_MAX_MEMORY` MB), `try_reserve` (returns `MemoryLimitExceeded` if `bytes > budget_bytes`). `BackendError::MemoryLimitExceeded { requested, limit }` exists. There is ALSO a separate `pyscf_algebra::Tensor<T=f64>` with `BufferId` (`crates/pyscf-algebra/src/tensor.rs`) — opaque handle + shape + dtype; its allocator is "later phase, inert until then." Decide whether the CCSD arena `Tensor` IS that algebra `Tensor` (preferred — single handle type) or a runtime-owned sibling; the D-08 wording places the backend enum in `pyscf-runtime`.
**Recommended shape (port discipline — keep reusable for Phase-7/8 per D-08):**
```rust
// pyscf-runtime/src/workspace_pool.rs (sketch — planner refines)
pub enum TensorBackend {
    InMemory(Box<[f64]>),                 // resident buffer
    Spilled(SpillHandle),                 // HDF5-backed (via pyscf-chkfile::hdf5)
}
pub struct PooledTensor { id: BufferId, shape: Vec<usize>, backend: TensorBackend }

impl WorkspacePool {
    /// Pre-flight (D-01 HARD refuse). Estimate before any allocation.
    pub fn try_reserve(&self, bytes: usize) -> Result<(), BackendError>; // already exists

    /// Allocate-or-reuse from the free-list. In-memory when it fits the
    /// remaining budget; otherwise the caller chose DF/AO-direct, OR (DF path)
    /// the backend spills to HDF5. Returns a handle; on release the buffer
    /// returns to the pool for reuse (NOT dropped) — satisfies the
    /// "allocate-once-reuse" / CI heap-alloc-count assertion.
    pub fn reserve(&self, shape: &[usize], allow_spill: bool) -> Result<BufferId, BackendError>;
    pub fn release(&self, id: BufferId);   // returns buffer to free-list, does not free
}
```
**Heap-alloc-count assertion (CCSD-11 success criterion):** the `Wabef` buffer is `reserve`d once before the iteration loop and `release`d after; across iterations the pool hands back the SAME `Box<[f64]>`. A CI test asserts a bounded global allocation count over N iterations (see §Validation — Wave 0 needs a counting allocator harness; recommend a `#[global_allocator]` counting shim behind a `cfg(test)`/feature, scoped to one integration test so it does not perturb the oracle determinism arms).

### Pattern 2: `update_amps` host-loop contraction (NOT gemm) — the bit-exactness discipline
**What:** Every einsum in `update_amps`/intermediates becomes an explicit loop that materializes products into a `Vec` then calls `oracle_sum`/`oracle_dot`.
**Why:** `pyscf_algebra::gemm` is a `NotYetImplemented{phase:2}` stub [VERIFIED: gemm.rs:16]. The Phase-5 `quarter_transform` (ao2mo/transform.rs) and `rmp2_kernel` (mp2.rs) are the canonical precedent — they do NOT use gemm; they loop with `oracle_sum`. CCSD must match.
**Example (the established Phase-5 pattern, mp2.rs:275-278):**
```rust
// edi = 2 · Σ_jab (ia|jb)·t2  →  materialize gi*t2i into a Vec FIRST, then reduce.
let edi = 2.0 * oracle_dot(&gi, &t2i);   // oracle_dot = oracle_sum of elementwise product
let exi = -oracle_dot(&g_jba, &t2i);     // (ib|ja) reordered to (j,a,b)
e_ss_terms.push(edi * 0.5 + exi);        // accumulate per-i terms into a Vec
// ... then once: let e_ss = oracle_sum(&e_ss_terms);   (NO running += across i)
```
Source: `crates/pyscf-mp2/src/mp2.rs:253-298`. Apply the identical discipline to `cc_Foo` (`2*einsum('kcld,ilcd->ki')` etc.) and the vvvv `'ijcd,acdb->ijab'`.

### Pattern 3: `AmplitudeSubspace: DiisStorable` (D-06) — one DIIS, second storable
**What:** Pack t1 + lower-triangular t2 into one flat vector; impl `DiisStorable`.
**Upstream packing (the exact layout, `ccsd.py:670` `amplitudes_to_vector`):**
```python
size = nov + nov*(nov+1)//2          # nov = nocc*nvir
vector[:nov] = t1.ravel()
lib.pack_tril(t2.transpose(0,2,1,3).reshape(nov,nov))   # symmetric t2[iajb]==t2[jbia]
```
`vector_to_amplitudes` (@679) unpacks with `lib.unpack_tril(filltriu=SYMMETRIC)`. The Rust `AmplitudeSubspace::as_flat`/`from_flat` reproduce this packing; `dot` routes through `oracle_dot` (the trait doc-comment mandates it — Pitfall 9). `run_diis` (@1206) only extrapolates when `istep >= diis_start_cycle && |de| < diis_start_energy_diff`. The `Diis<S>` machinery (`pyscf-diis/src/cdiis.rs`) is fully reusable — the storable doc-comment already says "Phase 6 CCSD the iterate is an `(T1, T2)` amplitude tuple (`pyscf-ccsd::AmpsSubspace` will impl this later)." [VERIFIED: storable.rs:5-6]
**When:** Wave after in-core UCCSD; default `diis_space=6` (NOT 8 — VERIFIED `ccsd.py:926`).

### Pattern 4: PyO3 bridge with override dispatch + `Python::detach` (D-09) — mirror `pyscf-py::mp`
**What:** `CcsdPyBridge` holds `Py<PyAny>` to the Python self; each hook checks `is_overridden` (`__qualname__` base-class comparison) and either `call_method1`s the override or runs the pure-Rust default under `py.detach`.
**Critical GIL discipline (Phase 6 is the HEAVIEST `python3.13t` re-validation — ROADMAP cross-cutting):** The kernel itself does NOT `py.detach` at the top (hooks re-enter Python). Each hook's DEFAULT path wraps the pure-Rust compute in `Python::attach(|py| py.detach(|| ...))`. This is verbatim the verified `Mp2PyBridge::ao2mo` pattern (`crates/pyscf-py/src/mp.rs:177-211`). For CCSD the long compute is the per-iteration `update_amps` — the `update_amps` hook's default path is the single biggest `py.detach` region in the project; under free-threaded 3.13t the test must confirm no deadlock and that concurrent calls hold the GIL correctly.
**Source:** `crates/pyscf-py/src/mp.rs` `is_overridden` (@130), `Mp2PyBridge` (@157-212), `kernel` (@346-375 — note "we DO NOT py.detach here — hooks re-enter Python"), `as_scanner` (@425-439).

### Pattern 5: DF-CCSD subclass + HDF5 spill (D-05/D-07/D-08, CCSD-08)
**What:** `DFRCCSD` reuses the in-core kernel and swaps `ao2mo`/`_add_vvvv` to the DF B-tensor (`vvL`) source; `Wabef` spills to an HDF5 temp file when the budget is exceeded.
**Upstream block sizing (port `dfccsd.py:93-96`):**
```python
dmax  = int(min((nvira+3)//4, max(BLKMIN, sqrt(max_memory*.7e6/8/nvirb**2/2))))
vvblk = int(min((nvira+3)//4, max(BLKMIN, (max_memory*1e6/8 - dmax**2*(nvirb**2*1.5+naux))/naux/naux)))
eris.feri = lib.H5TmpFile()                          # ← the spill file (D-07 H5TmpFile-equiv)
eris.vvL  = feri.create_dataset('vvL',(nvir_pair,naux),'f8',chunks=...)
```
The Rust port: `dmax`/`vvblk` become `WorkspacePool` reservation sizes (D-08); the `vvL`/`Wabef` HDF5 datasets use `pyscf_chkfile::hdf5::Group::new_dataset` (chunked/resizable — hdf5-metno supports chunked layout [CITED: docs.rs/hdf5-metno]). The DF B-tensor itself comes from `pyscf_df::cholesky_eri` + `default_ri` (the Phase-5 `build_df_integrals` helper in `pyscf-py/src/mp.rs:111-114` is the template). DF numeric is un-gated (cintx#11 closed, STATE.md) — only memory-bounded.

### Anti-Patterns to Avoid
- **Using `pyscf_algebra::gemm` for the vvvv contraction.** It is a stub; it returns `NotYetImplemented`. Use host loops + `oracle_sum`.
- **Running `+=` accumulation across the iteration or across blocks.** Breaks thread-count invariance (Pitfall 2). Materialize-then-`oracle_sum`.
- **Allocating `Wabef` inside the iteration loop.** Violates CCSD-11; allocate once via `reserve`, reuse across iterations.
- **Adding a `cubecl-*` or `hdf5-metno` dep to `pyscf-ccsd`.** The denylist auto-fails cubecl; D-07 forbids a second hdf5 owner.
- **Silent OOM or auto-downgrade to DF.** D-01: HARD-refuse with `MemoryLimitExceeded` and tell the user to opt into DF/AO-direct.
- **`py.detach` around the whole kernel.** Hooks re-enter Python; only the pure-Rust default *inside* each hook detaches (Pattern 4).
- **Adding a `pyscf-ccsd` allowlist entry to the dependency wall.** It is a denylist — there is nothing to add (see Pitfall 3 below).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pulay DIIS extrapolation for amplitudes | A new CDIIS solver | `pyscf-diis::Diis<AmplitudeSubspace>` | Generic over storable; B-matrix `solve_linear` + `oracle_dot` already bit-exact (Pitfall 9). Trait doc already anticipates `AmpsSubspace`. |
| AO→MO ERI block transform | A bespoke 4-index transform | `pyscf-ao2mo::general`/`full` | Phase-5 keystone; clean DAG; outcore surface extends it. |
| Frozen-core mask / `get_nocc`/`get_nmo` | New frozen logic | `pyscf-mp2::{Frozen, get_nocc, get_nmo, get_frozen_mask, mo_without_core}` | Contract-tested vs `cc/ccsd.py:35` in Phase 5 (CCSD-10). |
| Deterministic reductions | `+=` loops / naive `iter().sum()` | `pyscf_algebra::oracle_sum`/`oracle_dot` | Pairwise tree, chunk=128, thread-count invariant (Pitfall 1/2). |
| HDF5 temp file for spill | A second hdf5-metno dep | `pyscf_chkfile::hdf5` alias + drop-on-close temp | D-07; sole owner discipline. |
| MP2 seed for `init_amps` | A new MP2 path | `pyscf-mp2` in-core `ovov` path | `t2=(ia|jb)/Dijab`, `emp2` reuse (`ccsd.py:1048`). |
| Memory budget + refusal | Per-crate ad-hoc checks | `pyscf-runtime::WorkspacePool::try_reserve` → `MemoryLimitExceeded` | Single authority (CCSD-11); already wired. |
| Eigenvector sign canonicalization | New sign work | SCF reference's already-canonicalized `mo_coeff` | SCF-13/Pitfall 4; CCSD inherits. |
| Linear solve for DIIS B-matrix | LU by hand | `pyscf_algebra::solve_linear` | host-faer LU, Singular→DiisError mapped. |
| Symmetric eigendecomp for D1/D2 diagnostics | hand-rolled `eigh` | `pyscf_algebra::eigh` | `get_d1/d2_diagnostic` need `numpy.linalg.eigh` of t·tᵀ. |

**Key insight:** This phase is ~90% disciplined porting onto existing seams. The genuinely new engineering is the `WorkspacePool` arena body (CCSD-11) and the per-block contraction tiling for `Wabef`. Everything else is "fill the trait, mirror the upstream file, route reductions through oracle_*."

## Runtime State Inventory

> Phase 6 is greenfield code (filling a stub), NOT a rename/refactor/migration. This section is included only to record the one runtime-state concern that DOES apply: HDF5 spill temp files.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — CCSD produces no persisted keyed state. The optional `dump_chk` (`ccsd.py:1230`) writes `t1`/`t2`/`e_corr` to a chkfile, but chkfile output is not in v1 CCSD scope (no CCSD REQ maps to it). | None. |
| Live service config | None — no external services. | None — verified by scope (pure compute crate). |
| OS-registered state | None. | None. |
| Secrets/env vars | `PYSCF_MAX_MEMORY` (MB, read by `WorkspacePool::from_env`), `PYSCF_BACKEND`, `PYSCF_DTYPE` — all read-only, no rename. | None — code reads them; values unchanged. |
| Build artifacts | The HDF5 **spill temp file** created at runtime by DF-CCSD (`H5TmpFile`-equivalent, D-07). Must be deleted on drop (RAII), not left in the temp dir. | Ensure the spill `SpillHandle` deletes its file in `Drop` (mirror `lib.H5TmpFile()` auto-delete semantics). CI test asserts no leftover scratch files after a DF-CCSD-spill run. |

## Common Pitfalls

### Pitfall 20 (PRIMARY for this phase): CCSD memory thrash → tensor-arena from day one
**What goes wrong:** `Wabef ≈ nv⁴` is multi-GB at caffeine/cc-pVDZ. Allocating-and-dropping it per iteration thrashes the heap and risks OOM mid-iteration.
**Why it happens:** Naive port allocates intermediates inside `update_amps` each cycle.
**How to avoid:** D-01/D-08 arena — `reserve` once before the loop, `release` (return to free-list) after; pre-flight `try_reserve` refuses over-budget in-core jobs (no silent OOM).
**Warning signs:** Heap-alloc count grows with iteration count; RSS climbs monotonically across cycles.
**Validation:** CI heap-alloc-count assertion (Wave 0 needs the counting-allocator harness).

### Pitfall 9: DIIS path drift → amplitude DIIS
**What goes wrong:** Amplitude DIIS extrapolation diverges from upstream because the B-matrix inner products use a non-deterministic reduction.
**How to avoid:** `AmplitudeSubspace::dot` MUST route through `oracle_dot` (the `DiisStorable` trait doc-comment mandates it). `amplitudes_to_vector` packing must byte-match upstream (t1 then `pack_tril` of `t2.transpose(0,2,1,3)`).
**Warning signs:** Iteration count diverges from upstream; cross-thread-count energy mismatch.

### Pitfall 6: GIL deadlock (HEAVIEST re-validation here)
**What goes wrong:** Under free-threaded `python3.13t`, holding/releasing the GIL incorrectly around the long `update_amps` compute deadlocks or corrupts state.
**How to avoid:** Mirror the verified `Mp2PyBridge` discipline — kernel does NOT detach; each hook's pure-Rust DEFAULT path detaches (`Python::attach(|py| py.detach(|| ...))`); override path `call_method1`s under the GIL.
**Warning signs:** `python3.13t` smoke test hangs; the existing CI `python3.13t SCF smoke` job (ci.yml:274) is the template — add a CCSD analog.

### Pitfall 4: eigenvector sign (via SCF ref)
**What goes wrong:** CCSD amplitudes depend on `mo_coeff` column signs; un-canonicalized signs cause vendor-dependent results.
**How to avoid:** Consume the SCF reference's already-`canonicalize_signs`'d `mo_coeff` (SCF-13). No new sign work. The eager snapshot in `pyscf-py` already pulls the canonicalized coeffs (mp.rs:83).

### Pitfall 1/2: FMA / reduction order in contractions
**What goes wrong:** Bare `+=` or FMA fuses change rounding → cross-platform/thread bit-drift.
**How to avoid:** Materialize-then-`oracle_sum`/`oracle_dot` everywhere (Pattern 2). The `check-no-fma` guard is ALREADY scoped to `pyscf_*`-owned symbols (`symbol_is_pyscf_owned`, check_no_fma.rs:209) — faer/pulp SIMD FMA in the linalg backend is exempt, so the planner does NOT need a new exemption; just keep CCSD's own symbols FMA-free under `release-oracle`. **Note:** `check_no_fma` SCAN_TARGETS currently lists only `pyscf-algebra` + `pyscf-core` (check_no_fma.rs:82-83); to gate `pyscf-ccsd` symbols the planner should add `("pyscf-ccsd","pyscf_ccsd")` to SCAN_TARGETS.

### Pitfall 3 (process): treating the dependency wall as an allowlist
**What goes wrong:** CONTEXT.md says "extend allowlist for pyscf-ccsd" — but `check_dependency_wall.rs` is a **denylist** (`FORBIDDEN_DEPS` = cubecl-*; `ALLOWED_CRATES` = only the 3 carve-out crates). Any `pyscf-*` crate naming cubecl auto-fails.
**How to avoid:** Do NOT add a `pyscf-ccsd` entry anywhere — the wall already covers it. Just ensure `pyscf-ccsd/Cargo.toml` lists no `cubecl-*` dep. [VERIFIED: check_dependency_wall.rs:28-47]

### Pitfall: `gemm` stub mistaken for available BLAS
**What goes wrong:** Planning vvvv contraction as a `gemm` chain; it returns `NotYetImplemented{phase:2}` at runtime.
**How to avoid:** Host loops + `oracle_sum` (Pattern 2). [VERIFIED: gemm.rs:16]

## Code Examples

Verified patterns from the in-tree codebase + upstream port targets.

### Upstream CCSD kernel iteration loop (the port target — `ccsd.py:44-101`)
```python
# Source: pyscf/cc/ccsd.py:44 (in-repo oracle reference)
def kernel(mycc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6, ...):
    if t1 is None and t2 is None: t1, t2 = mycc.get_init_guess(eris)   # MP2 seed
    eccsd = mycc.energy(t1, t2, eris)
    adiis = lib.diis.DIIS(mycc); adiis.space = mycc.diis_space          # diis_space=6
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        tmpvec = mycc.amplitudes_to_vector(t1new,t2new) - mycc.amplitudes_to_vector(t1,t2)
        normt = numpy.linalg.norm(tmpvec)
        t1, t2 = t1new, t2new
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        if abs(eccsd-eold) < tol and normt < tolnormt:                  # dual criterion
            converged = True; break
    return converged, eccsd, t1, t2
```
Note: `CCSD` class default `tol=conv_tol=1e-7`, `tolnormt=conv_tol_normt=1e-5` upstream default — **but** CONTEXT.md Discretion says `conv_tol_normt=1e-6`; the upstream `CCSDBase.conv_tol_normt` default is `1e-5` (VERIFIED `ccsd.py:923`). The planner must resolve this 1e-5-vs-1e-6 discrepancy against the actual fixture (the kernel signature default is `tolnormt=1e-6` @45, but the class attribute is `1e-5` @923 — the class attribute wins at runtime). **Recommend `conv_tol=1e-7`, `conv_tol_normt=1e-5`, `max_cycle=50`, `diis_space=6` to match the CCSD class defaults; flag to discuss-phase.** [VERIFIED: ccsd.py:920-928]

### Upstream `init_amps` MP2 seed (`ccsd.py:1048`)
```python
eia = mo_e[:nocc,None] - mo_e[None,nocc:]
t1 = eris.fock[:nocc,nocc:] / eia
t2[:,:,p0:p1] = eris_ovov.transpose(0,2,1,3).conj() / direct_sum('ia,jb->ijab', eia[:,p0:p1], eia)
emp2 += 2*einsum('ijab,iajb',t2,eris_ovov) - einsum('jiab,iajb',t2,eris_ovov)
```
Reuse `pyscf-mp2`'s `ovov` path; the `t2=(ia|jb)/Dijab` form is already in `rmp2_kernel` (mp2.rs:265-273).

### In-tree reduction discipline (the contraction template — `mp2.rs:275`)
```rust
// Source: crates/pyscf-mp2/src/mp2.rs (VERIFIED present)
let edi = 2.0 * oracle_dot(&gi, &t2i);    // materialize → oracle reduce, NO +=
e_ss_terms.push(edi * 0.5 + exi);          // per-i terms collected
let e_ss = oracle_sum(&e_ss_terms);        // single final reduction
```

### In-tree PyO3 detach discipline (the bridge template — `mp.rs:204`)
```rust
// Source: crates/pyscf-py/src/mp.rs:204-210 (VERIFIED present)
// DEFAULT path: pure-Rust compute under py.detach (BIND-05). Kernel does NOT detach.
DefaultAo2mo::InCore => Python::attach(|py| py.detach(|| default_ao2mo(refr, frozen))),
```

### In-tree DIIS storable + generic Diis (the amplitude-DIIS template — `cdiis.rs:33`)
```rust
// Source: crates/pyscf-diis/src/cdiis.rs (VERIFIED present)
pub struct Diis<S: DiisStorable + Clone> { space: usize, ... }
// B[i,j] = oracle_dot(err_i, err_j); solve_linear(B, rhs, dim); extrap via oracle_sum.
// CCSD: Diis::<AmplitudeSubspace>::new(6)
```

## State of the Art

| Old Approach (prior phases) | Current Approach (Phase 6) | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `PYSCF_MAX_MEMORY` log-only (Phases 3/4/5) | HARD pre-flight refusal via `try_reserve` (D-01) | Phase 6 | Single budget authority; no silent OOM. |
| In-memory ERIs, spill deferred (Phase 5 D-04) | Spill-when-budget-exceeded = backend swap behind handle (D-01) | Phase 6 | The "later" is now; arena from day one. |
| DF numeric gated behind cintx#11 (Phases 3/4/5) | DF numeric un-gated; only memory-bounded (cintx#11 closed 05-08/09) | Plan 05-08/09 | DF-CCSD energy oracle-validatable in-tree (human-verify arm for caffeine size). |
| MP2 `make_rdm2(ao_repr=True)` deferred to Phase 7 | CCSD ships full λ + RDM incl. `ao_repr=True` THIS phase (D-03) | Phase 6 | Heaviest arena tenant exercised; no Phase-7 RDM carry-over. |

**Deprecated/outdated for this phase:**
- The spin-orbital `rccsd.RCCSD` module — NOT the factory target for a standard RHF reference (D-05). `cc.RCCSD(mf)` resolves to `ccsd.CCSD` (RHF) / `dfccsd.RCCSD` (DF) / `uccsd.UCCSD` (UHF). [VERIFIED: cc/__init__.py:95-121]
- `pyscf_algebra::gemm` as an available primitive — it is a `phase:2` stub; do not plan around it.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `conv_tol_normt` should be `1e-5` (CCSD class default) not the `1e-6` in CONTEXT.md Discretion / the kernel-signature default | Code Examples / Convergence | LOW-MED: a too-loose normt could pass the oracle at looser amplitude convergence; too-tight wastes cycles. The class attribute (`1e-5`) wins at runtime upstream. Flag to discuss-phase; resolve against the small-system fixture. |
| A2 | The CCSD arena `Tensor` handle should reuse the existing `pyscf_algebra::Tensor`/`BufferId` rather than a new runtime-only handle type | Pattern 1 | MED: D-08 places the backend enum in `pyscf-runtime`, but the algebra `Tensor` already has `BufferId`. Two handle types would be redundant; the planner must pick one. Wrong choice = a refactor mid-phase. |
| A3 | The heap-alloc-count assertion is implementable via a `cfg(test)`/feature-gated counting `#[global_allocator]` scoped to one integration test | Validation / Pattern 1 | MED: a global allocator shim can perturb other tests in the same binary; must be isolated to its own test target so the `release-oracle` determinism arms are unaffected. |
| A4 | hdf5-metno chunked/resizable datasets are sufficient for the `Wabef`/`vvL` spill (no need for a specific extendable-along-axis API beyond what Phase 3/4 used) | Pattern 5 | LOW: the spill blocks are fixed-size reservations (`dmax`/`vvblk`), so fixed chunked datasets suffice; verified hdf5-metno supports chunked layout. |
| A5 | UCCSD snapshot mirrors the Phase-5 `UmpReference { alpha, beta }` two-channel shape and the env-can't-run-libpython caveat applies equally | Pattern (UCCSD) | LOW: the structural contract is established; only live-CI UHF numeric needs the human-verify arm. |
| A6 | `solve_lambda` in base `CCSDBase` raises NotImplementedError; only the concrete `CCSD` class wires `ccsd_lambda.kernel` | CCSD-05 | LOW: VERIFIED — `CCSDBase.solve_lambda` @1118 raises; `CCSD.solve_lambda` @1273 dispatches. Port the concrete-class behavior. |

## Open Questions (RESOLVED)

> Resolution status (annotated during planning, 2026-05-24 — Phase 6 plan-phase):
> - **Q1 (`conv_tol_normt`) — RESOLVED in 06-03-PLAN.md:** use the verified class defaults `conv_tol=1e-7`, `conv_tol_normt=1e-5`, `max_cycle=50`, `diis_space=6` (`ccsd.py:920-928`); the small-system oracle fixture confirms ≤1 µHartree.
> - **Q2 (one vs two `Tensor` handle types) — RESOLVED-AT-EXECUTION (06-02-PLAN.md, Task 1):** single-handle direction (WorkspacePool owns backing storage keyed by `BufferId`; `Amplitudes`/intermediates carry the handle); executor finalizes the newtype shape in Wave 2.
> - **Q3 (counting-allocator harness) — RESOLVED in 06-02-PLAN.md, Task 2:** dedicated `tests/heap_alloc_count.rs` target with its own counting `#[global_allocator]`, NOT linked to the oracle/determinism binaries.
> - **Q4 (AO-direct ERI streaming API) — RESOLVED-AT-EXECUTION (06-08-PLAN.md, Task 1):** executor greps the `pyscf-gto` intor surface; both acceptable paths (shell-sliced streaming vs full-tensor-once + MO-space tiling) satisfy CCSD-07's "vvvv MO tensor not materialized" contract and are documented in the plan.

1. **`conv_tol_normt` = 1e-5 or 1e-6?**
   - What we know: CONTEXT.md Discretion says `1e-6`; the kernel-signature default is `1e-6` (@45); the `CCSDBase` class attribute is `1e-5` (@923); at runtime the class attribute wins.
   - What's unclear: Which value the oracle fixture must match for ≤1 µHartree CCSD-01.
   - Recommendation: Use the class defaults (`conv_tol=1e-7`, `conv_tol_normt=1e-5`, `max_cycle=50`, `diis_space=6`); flag the discrepancy to discuss-phase; confirm against the H2O/cc-pVDZ fixture vs upstream.

2. **One `Tensor` handle type or two? (A2)**
   - What we know: `pyscf_algebra::Tensor<T>` with `BufferId` exists (algebra crate); D-08 places the spill backend enum in `pyscf-runtime`.
   - What's unclear: whether the arena allocates the algebra `Tensor` directly or a runtime `PooledTensor` that the algebra `Tensor` references.
   - Recommendation: Single handle. Have `WorkspacePool` own the backing storage keyed by `BufferId`, and let `Amplitudes`/intermediates carry `pyscf_algebra::Tensor` (or a thin `pyscf-runtime` newtype) so contraction call sites stay uniform. Planner decides in Wave 1; this is the riskiest API decision.

3. **Counting-allocator harness for the CCSD-11 assertion (A3).**
   - What we know: no global counting allocator exists in the repo today (probe dir is backend probes only).
   - What's unclear: cleanest way to count allocations without perturbing the determinism arms.
   - Recommendation: a dedicated integration test target (`tests/heap_alloc_count.rs`) with its own `#[global_allocator]` counting shim, NOT linked into the oracle/determinism test binaries. Wave 0 task.

4. **AO-direct ERI streaming API on `pyscf-gto`/`pyscf-ao2mo`.**
   - What we know: upstream AO-direct uses `gto.moleintor.getints4c` with `shls_slice` for on-the-fly shell-pair `int2e` blocks (`ccsd.py:538-558`); the Rust `pyscf-gto::intor("int2e")` returns the full arity-4 tensor (verified used whole in mp2.rs:150).
   - What's unclear: whether `pyscf-gto` exposes a shell-sliced `int2e` (the streaming primitive AO-direct needs) or only the full tensor.
   - Recommendation: Investigate `pyscf-gto`'s intor surface during AO-direct wave planning; if only the full tensor is exposed, AO-direct v1 may compute the full `int2e` once and tile in MO space (correct but not memory-optimal) — flag to discuss-phase whether that satisfies CCSD-07's `direct=True` contract or if a shell-sliced primitive is needed (cintx#11 closure means the integrals are available; the question is the slicing API).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `int2e` (arity-4, in-tree) | In-core RCCSD/UCCSD ERI source | ✓ | real bit-exact since 05-08 (cintx#11 closed) | — |
| `int3c2e_sph`/`int2c2e_sph` | DF-CCSD `vvL` B-tensor | ✓ | real since 05-08/09 | — |
| hdf5-metno (via `pyscf-chkfile`) | DF-CCSD `Wabef` spill | ✓ | workspace dep, Phase 3/4 proven | — |
| `pyscf_algebra::gemm` | (NOT used — host loops instead) | ✗ STUB | `NotYetImplemented{phase:2}` | Host loops + `oracle_sum` (the actual approach) |
| upstream PySCF (oracle) | byte-identity numeric assertions | ✗ in sandbox | — | `workflow_dispatch`/human-verify arm (D-04, 05-08 precedent) |
| `python3.13t` free-threaded | GIL re-validation (heaviest here) | CI-provisioned | per ci.yml:274 SCF smoke job | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:**
- `gemm` (stub) → host-loop contraction with `oracle_sum`/`oracle_dot` (the established Phase-5 approach; not a degradation, it is the v1 design).
- upstream PySCF (sandbox can't install) → `workflow_dispatch` human-verify arm for byte-identity + caffeine/DF-spill.

## Validation Architecture

> Nyquist validation is ENABLED (`config.json` `workflow.nyquist_validation: true`). This section is REQUIRED and feeds VALIDATION.md.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `cargo test`; `pyscf-oracle` `oracle_check!` macro for energy fixtures; `--profile release-oracle` for bit-exact arms |
| Config file | none (Cargo workspace) — test targets live in `crates/pyscf-ccsd/tests/` + `crates/pyscf-oracle/tests/` |
| Quick run command | `cargo test -p pyscf-ccsd --locked` |
| Full suite command | `cargo test --workspace --locked` + (manual) `cargo test -p pyscf-oracle --features python --locked -- --test-threads=1` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CCSD-01 | in-core RCCSD energy ≤1 µHartree (small system, always-on) | oracle (in-tree, small) | `cargo test -p pyscf-ccsd --test rccsd_numeric_smoke -- --test-threads=1` | ❌ Wave 0 |
| CCSD-01 | RCCSD byte-identity vs upstream (caffeine, gated) | oracle (human-verify) | `cargo test -p pyscf-oracle --features python -- ccsd_rccsd` (`workflow_dispatch`) | ❌ Wave 0 (CI arm) |
| CCSD-02 | UCCSD energy (small open-shell, structural always-on) | structural + oracle | `cargo test -p pyscf-ccsd --test uccsd_smoke` | ❌ Wave 0 |
| CCSD-03 | T1/T2 dual convergence (energy target) | unit | `cargo test -p pyscf-ccsd convergence` | ❌ Wave 0 |
| CCSD-04 | amplitude-DIIS converges in upstream iter count; vector packing byte-matches | unit (packing) + integration (iter count) | `cargo test -p pyscf-ccsd diis_amps` | ❌ Wave 0 |
| CCSD-05 | `solve_lambda` λ amplitudes match upstream | oracle (small + human-verify) | `cargo test -p pyscf-ccsd lambda` / oracle arm | ❌ Wave 0 |
| CCSD-06 | `make_rdm1`/`make_rdm2` (incl. ao_repr) match upstream | oracle | `cargo test -p pyscf-ccsd rdm` / oracle arm | ❌ Wave 0 |
| CCSD-07 | AO-direct `direct=True` matches in-core | integration | `cargo test -p pyscf-ccsd direct` | ❌ Wave 0 |
| CCSD-08 | DF-CCSD bounded memory; spills `Wabef` to HDF5 over budget; no leftover scratch | integration (small) + human-verify (benzene-dimer spill) | `cargo test -p pyscf-ccsd dfccsd_spill` / `workflow_dispatch` | ❌ Wave 0 |
| CCSD-09 | `t1diagnostic`/`d1diagnostic` values match upstream | unit | `cargo test -p pyscf-ccsd diagnostics` | ❌ Wave 0 |
| CCSD-10 | frozen `int`/`list`/`'auto'` match MP2 | unit (reuse MP2 helpers) | `cargo test -p pyscf-ccsd frozen` | ❌ Wave 0 |
| CCSD-11 | `Wabef` allocated once across N iterations; over-budget in-core HARD-refuses | integration (alloc count) + unit (refusal) | `cargo test -p pyscf-ccsd --test heap_alloc_count` / `try_reserve` unit | ❌ Wave 0 (needs counting allocator) |

### Sampling Rate
- **Per task commit:** `cargo test -p pyscf-ccsd --locked` (fast structural + small-system arms; no upstream, no caffeine).
- **Per wave merge:** `cargo test --workspace --locked` + the `release-oracle` determinism arms (`-p pyscf-algebra --test oracle_determinism`, both rayon-1 and rayon-8) to guard Pitfall 1/2 on the new contractions.
- **Phase gate:** full workspace green + the `workflow_dispatch` human-verify arm run once manually (upstream byte-identity CCSD energy on small + caffeine, DF-CCSD spill proof on constrained `PYSCF_MAX_MEMORY`, λ/RDM byte-identity) + the `python3.13t` CCSD smoke (no GIL deadlock).

### Wave 0 Gaps
- [ ] `crates/pyscf-ccsd/tests/rccsd_numeric_smoke.rs` — covers CCSD-01 (small-system in-core, always-on)
- [ ] `crates/pyscf-ccsd/tests/uccsd_smoke.rs` — covers CCSD-02
- [ ] `crates/pyscf-ccsd/tests/diis_amps.rs` — covers CCSD-04 (vector packing + iter count)
- [ ] `crates/pyscf-ccsd/tests/diagnostics.rs` — covers CCSD-09
- [ ] `crates/pyscf-ccsd/tests/frozen.rs` — covers CCSD-10 (reuse MP2 helper fixtures)
- [ ] `crates/pyscf-ccsd/tests/dfccsd_spill.rs` — covers CCSD-08 (spill + no-leftover-scratch)
- [ ] `crates/pyscf-ccsd/tests/direct.rs` — covers CCSD-07
- [ ] `crates/pyscf-ccsd/tests/heap_alloc_count.rs` — covers CCSD-11 (DEDICATED target with its own counting `#[global_allocator]`, NOT linked to oracle/determinism binaries — A3)
- [ ] `crates/pyscf-ccsd/tests/refusal.rs` — covers CCSD-11 pre-flight `MemoryLimitExceeded`
- [ ] `crates/pyscf-oracle/tests/*` — add CCSD energy/λ/RDM byte-identity fixtures (small always-on; caffeine/DF-spill gated behind `--features python` + `workflow_dispatch`, mirroring `mp2-oracle-upstream-manual` ci.yml:445)
- [ ] `.github/workflows/ci.yml` — add: `ccsd-structural`/`ccsd-oracle` always-on small arm; `ccsd-oracle-upstream-manual` (`workflow_dispatch`, caffeine + DF-spill + λ/RDM byte-identity); `python3.13t` CCSD smoke (clone the `python3.13t SCF smoke` job @274); heap-alloc-count gate
- [ ] `xtask/src/bin/check_no_fma.rs` — add `("pyscf-ccsd","pyscf_ccsd")` to `SCAN_TARGETS` (so CCSD's own symbols are FMA-checked under release-oracle)

## Security Domain

> `security_enforcement` is not set in `config.json` (absent = enabled per agent contract). However, this is a pure numerical compute crate (post-SCF correlation energy) with NO authentication, sessions, access control, network, untrusted input, or cryptography. The standard ASVS web/application categories do not map.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | partial | Shape validation at every untrusted boundary (the established `ShapeMismatch` error pattern, mp2.rs:97/192-205; ao2mo transform.rs:67-82) — validate `nocc`/`nvir`/buffer lengths before indexing; never panic, never OOB. The PyO3 boundary (`pyscf-py`) extracts NumPy arrays — already-vetted Phase-3/5 converters. |
| V6 Cryptography | no | — (never hand-roll, but none needed) |

### Known Threat Patterns for {Rust numeric kernel + PyO3 + HDF5 spill}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| OOB index / panic on malformed shapes from a Python override (`ao2mo`/`update_amps` hook returns wrong-shaped array) | DoS / Tampering | Validate every hook-returned array length against the expected `nocc*nvir*...` before use (the `ShapeMismatch` `?`-propagation pattern); the `#![forbid(unsafe_code)]` crate attribute means no UB even on logic error. |
| Mid-iteration OOM (the CCSD-11 hazard) | DoS | `try_reserve` pre-flight HARD refusal (D-01) — never start an in-core job that would exceed the budget. |
| HDF5 spill temp-file leak / predictable path | Information Disclosure | RAII drop-delete (mirror `lib.H5TmpFile()`); use a secure temp path (the `pyscf-chkfile` temp-file convention). |
| GIL re-entrancy / data race under `python3.13t` | Tampering | The verified detach discipline (Pattern 4); the `python3.13t` smoke gate. |
| FMA/reduction-order non-determinism (reproducibility, not classic security) | Tampering (result integrity) | `oracle_sum`/`oracle_dot` + `check-no-fma` (scoped to `pyscf_*`); `release-oracle` determinism arms (rayon-1 vs rayon-8). |

## Wave / Dependency Sequencing (planner input)

Natural DAG honoring D-02's ordering and the CCSD-11 "from day one" mandate:

```
Wave 0  (test infra + scaffold — parallel seams):
   ├─ T-A: pyscf-ccsd Cargo deps + module skeleton (lib.rs, error.rs, eris.rs stubs)
   ├─ T-B: WorkspacePool arena body + opaque Tensor backend enum (D-08) ◄── CCSD-11 day-one
   ├─ T-C: pyscf-core::Amplitudes → Tensor-handle upgrade (D-01)        ◄── depends on T-B
   ├─ T-D: heap-alloc-count test harness (dedicated allocator target, A3)
   └─ T-E: small-system oracle fixtures stubs + CI arm skeleton (D-04)

Wave 1  (UN-GATED HEADLINE — depends Wave 0):
   ├─ rintermediates.rs (cc_Foo/Fvv/Fov/Woooo/Wvvvv/Wvoov/Wovvo/make_tau)
   ├─ update_amps.rs + _add_vvvv/_contract_vvvv_t2 (in-core, host-loop oracle_sum)
   ├─ ccsd.rs kernel loop + init_amps(MP2 seed) + energy + defaults + pre-flight refusal
   └─ GATE: CCSD-01 small-system oracle green (the headline proof)

Wave 2  (depends Wave 1):
   ├─ uccsd.rs + uintermediates.rs (CCSD-02)        ─┐ parallelizable with ↓
   └─ diis_amps.rs AmplitudeSubspace (CCSD-04, D-06) ─┘ (DIIS needs the kernel loop, not UCCSD)

Wave 3  (depends Wave 1; λ+RDM are the heaviest arena tenants — D-03):
   ├─ lambda.rs + ulambda.rs (CCSD-05)
   ├─ rdm.rs + urdm.rs incl. ao_repr back-transform (CCSD-06)  ◄── depends lambda (RDM needs λ)
   └─ diagnostics.rs (CCSD-09)   ─ parallelizable (only needs t1/t2)

Wave 4  (ERI-mode follow-ons — depend Wave 1; parallelizable with each other):
   ├─ direct.rs AO-direct branch (CCSD-07)
   └─ dfccsd.rs DF-CCSD + HDF5 spill + block sizing (CCSD-08, D-05/D-07)

Wave 5  (PyO3 — depends ALL kernels):
   ├─ pyscf-py::cc submodule: PyRCCSD/PyUCCSD/PyDFCCSD + CcsdPyBridge + factory + as_scanner (D-09)
   ├─ python/pyscf/cc/__init__.py overlay (mf.CCSD() / mf.density_fit().CCSD() dispatch)
   └─ python3.13t CCSD smoke (Pitfall 6, heaviest)

Wave 6  (validation close-out):
   └─ ccsd-oracle-upstream-manual workflow_dispatch arm: caffeine byte-identity, DF-spill proof,
      λ/RDM byte-identity; xtask check_no_fma SCAN_TARGETS += pyscf-ccsd
```
**Parallelizable seams:** Wave 0 T-A/T-D/T-E vs T-B/T-C; Wave 2 UCCSD vs DIIS; Wave 3 diagnostics vs lambda/rdm; Wave 4 AO-direct vs DF. **Hard serial spine:** T-B→T-C→Wave 1 kernel→(Wave 2/3/4)→Wave 5 PyO3→Wave 6 oracle.

## Sources

### Primary (HIGH confidence — read directly in-repo this session)
- `pyscf/cc/ccsd.py` — kernel @44, update_amps @104, _add_vvvv/_contract_vvvv_t2 @362-570, amplitudes_to_vector @670, energy @710, diagnostics @748-776, init_amps @1048, class defaults @920-928, run_diis @1206, CCSD class @1261, solve_lambda @1273
- `pyscf/cc/rintermediates.py` — cc_Foo/Fvv/Fov @30-60 (Hirata JCP 120, 2581 ref)
- `pyscf/cc/dfccsd.py` — vvL spill + dmax/vvblk block sizing @70-194
- `pyscf/cc/__init__.py` — factory dispatch @83-136
- `crates/pyscf-runtime/src/workspace_pool.rs` — WorkspacePool skeleton + try_reserve + from_env
- `crates/pyscf-runtime/src/error.rs` — BackendError::MemoryLimitExceeded
- `crates/pyscf-core/src/amplitudes.rs` — Amplitudes skeleton (t1/t2 Vec<f64>)
- `crates/pyscf-algebra/src/{tensor.rs, oracle.rs, gemm.rs}` — Tensor/BufferId, oracle_sum/oracle_dot, gemm STUB
- `crates/pyscf-diis/src/{storable.rs, cdiis.rs, lib.rs}` — DiisStorable, Diis<S>, amplitude-DIIS template
- `crates/pyscf-mp2/src/{mp2.rs, hooks.rs, lib.rs}` — kernel/hooks/reduction-discipline template
- `crates/pyscf-py/src/mp.rs` — PyRMP2 bridge, is_overridden, py.detach discipline, as_scanner
- `crates/pyscf-ao2mo/src/transform.rs` — quarter_transform host-loop precedent
- `crates/pyscf-chkfile/src/lib.rs` — `pub use hdf5_metno as hdf5` alias
- `crates/pyscf-df/src/lib.rs` — DfIntegrals/cholesky_eri/default_ri surface
- `xtask/src/bin/check_dependency_wall.rs` — denylist (cubecl), ALLOWED_CRATES carve-out
- `xtask/src/bin/check_no_fma.rs` — pyscf_*-scoped FMA guard, SCAN_TARGETS
- `.github/workflows/ci.yml` — mp2-oracle-upstream-manual @445, python3.13t SCF smoke @274, release-oracle arms
- `.planning/phases/06-ccsd/06-CONTEXT.md`, `.planning/phases/05-mp2/05-CONTEXT.md`, `.planning/config.json`

### Secondary (MEDIUM confidence)
- hdf5-metno chunked/resizable dataset support — [CITED: docs.rs/hdf5-metno, github.com/metno/hdf5-rust]

### Tertiary (LOW confidence)
- none — all claims grounded in in-repo files or upstream source.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every dep verified present; gemm-stub + denylist findings verified by reading source.
- Architecture/patterns: HIGH — MP2 is a complete, verified template; CCSD mirrors it file-for-file.
- Pitfalls: HIGH — all five mapped pitfalls grounded in verified guard code (oracle_sum, check_no_fma scope, detach discipline, try_reserve).
- WorkspacePool arena body / heap-alloc harness: MEDIUM — the SHAPE is clear but the exact handle-unification (A2) and counting-allocator isolation (A3) are genuine design decisions for Wave 0/1.
- AO-direct ERI-streaming API: MEDIUM — depends on whether `pyscf-gto` exposes shell-sliced int2e (Open Question 4).

**Research date:** 2026-05-24
**Valid until:** 2026-06-23 (30 days — workspace is internally stable; the only external surface is upstream PySCF source which is pinned in-repo)
