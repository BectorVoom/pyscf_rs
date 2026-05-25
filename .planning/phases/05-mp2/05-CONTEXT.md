# Phase 5: MP2 - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

A user runs `mp.RMP2(mf).kernel()`, `mp.UMP2(uhf_mf).kernel()`, `mp.DFMP2(mf).kernel()`, and the cross-module idiom `mf.MP2().run()` on the test corpus and gets upstream-matching MP2 correlation energies bit-exact under `release-oracle`. Phase 5 builds the first post-SCF correlation layer on top of the Phase 3 SCF reference (RHF/UHF `ScfResult` — converged `mo_coeff`/`mo_energy`/`mo_occ`/`e_tot`). It introduces the **AO→MO integral transformation** (a new `pyscf-ao2mo` crate, deliberately general so CCSD reuses it), frozen-core handling, in-core + density-fitted variants, SCS-MP2, MP2 1-/2-particle RDMs, `as_scanner`, and the five helper functions CCSD imports directly.

**In scope (8 REQ-IDs):**
- **MP2-01:** `mp.RMP2(mf).kernel()` + `mf.MP2().run()` reproduce upstream RMP2 correlation energy bit-exact under `release-oracle`.
- **MP2-02:** `mp.UMP2(uhf_mf).kernel()` reproduces upstream UMP2 (open-shell).
- **MP2-03:** Frozen-core options accept `frozen=int`, `frozen=list`, `frozen='auto'`, and frozen-window forms; defaults match upstream.
- **MP2-04:** `mp.DFMP2(mf).kernel()` reproduces upstream DF-MP2 (numeric oracle CI-gated behind the cintx `int3c2e_sph` merge — see D-05).
- **MP2-05:** `mp2.make_rdm1()` and `mp2.make_rdm2()` match upstream.
- **MP2-06:** SCS-MP2 via `mp.MP2(mf).set(emp2_ss_factor=..., emp2_os_factor=...)`.
- **MP2-07:** `mp2.as_scanner()` returns a callable taking a Mole → energy (exercised by a Phase 7 geomopt smoke test).
- **MP2-08:** MP2 helpers (`get_nocc`, `get_nmo`, `get_frozen_mask`, `get_e_hf`, `_mo_without_core`) exported with upstream-matching semantics, contract-tested via a unit test mimicking CCSD's exact import call site (`cc/ccsd.py:35`).

**Out of scope:**
- CCSD (Phase 6), gradients + geomopt (Phase 7), GPU per-backend regression + 2–5× benchmark proof (Phase 8).
- **Outcore / semi-incore HDF5-spilling AO→MO** — deferred to Phase 6 (CCSD-08 + CCSD-11 tensor-arena + spill from day one), per D-04 below and the Phase 3 D-11 precedent.
- **`PYSCF_MAX_MEMORY` budget-aware refusal/spill** — Phase 6 (CCSD-11). Phase 5 logs the budget only.
- **Fused cubecl AO→MO kernel** — Phase 8 (the 2–5× owner), only if profiling shows the algebra-`gemm` chain is the bottleneck.
- **GMP2 / DFGMP2 (GHF-reference MP2)** — not raised in v1 MP2 requirements (MP2-01..08 cover R/U + DF only); treat as out of scope for this phase unless a later requirement adds it. (Upstream `gmp2.py`/`dfgmp2.py` exist but no REQ-ID maps to them.)
- **MP2-F12** (`mp2f12_slow.py`) — explicitly not in v1.
- **FNO-MP2 / `make_fno` frozen-natural-orbital truncation** — not in the MP2-01..08 set; deferred.

</domain>

<decisions>
## Implementation Decisions

### AO→MO transformation crate (cross-cutting — reused by CCSD per ROADMAP Phase-5 goal)

- **D-01: New `pyscf-ao2mo` workspace crate (19 → 20 members).** The AO→MO integral transformation gets its own workspace member rather than living inside `pyscf-mp2`. This mirrors upstream's standalone `pyscf/ao2mo/` module and the established own-crate-per-shared-concern pattern (`pyscf-df` D-10, `pyscf-diis` D-08, `pyscf-grids` D-05). **Rationale (dependency-DAG correctness):** CCSD (Phase 6) imports the transform directly from `pyscf-ao2mo`, avoiding a `pyscf-ccsd → pyscf-mp2` crate dependency that would invert the natural layering. ROADMAP.md needs an explicit 19→20 member update during planning (the workspace overview currently says 19). Algebra wall applies (`pyscf-ao2mo` depends on `pyscf-algebra` for the contraction + `pyscf-gto` for `mol.intor`, never `cubecl-*` directly); xtask `algebra_wall` allowlist extended.

- **D-02: Mirror upstream `ao2mo` public surface.** Port upstream's shape — a **general** transform (`ao2mo.general(eri_or_mol, (C1,C2,C3,C4))`) + **`incore.full(mol, mo_coeff)`** producing the full MO ERI tensor — with MP2 calling it for the `(occ,vir|occ,vir)` block (the `_ao2mo_ovov` path in `mp2.py`). CCSD (Phase 6) gets the `general`/`full` kernel it needs **day one**, no retrofit. Honors the sibling-crate-fidelity hard preference (carried from Phase 1/2/3). Not a fresh non-upstream-shaped Rust API — the upstream naming preserves the oracle/parser mapping.

- **D-03: AO→MO contraction = `gemm` chains through `pyscf-algebra`; no new bespoke cubecl kernel in v1.** The `(pq|rs) → (ij|ab)` four-index transform is implemented as a sequence of `gemm`/`reduce` calls via `pyscf-algebra`, exactly like `pyscf-df` (D-10: "no new kernel — just Cholesky + matmul") and the DFT grid loop (D-07). Respects the algebra wall, gives bit-exact reductions under `release-oracle` (ordered reductions / FMA-free), and leaves the fused-kernel optimization to Phase 8 (the 2–5× owner). A fused cubecl `ao2mo` kernel is a Deferred Idea, not a v1 gate.

### ERI-storage scope (sequencing vs Phase 6)

- **D-04: Phase 5 ships in-core + DF only; outcore/semi-incore HDF5 spill is deferred to Phase 6.** In-core AO→MO (full ERIs / `B`-tensor in memory) + density-fitted MP2 cover MP2-01..06 on the test corpus (H2O/cc-pVDZ, benzene/6-31G*, water-trimer all fit in memory). The out-of-core machinery (upstream `ao2mo/outcore.py`, `semi_incore.py`) lands in Phase 6 where CCSD-08 + CCSD-11 introduce the tensor-arena + HDF5 spill from day one. Directly matches Phase 3 D-11 (DF B-integrals in-memory in Phase 3, spill deferred to Phase 6) and the ROADMAP Phase-5 "in-core and density-fitted variants" wording. **`PYSCF_MAX_MEMORY` is log-only at MP2 kernel entry** (the Phase 3 SCF / Phase 4 DFT convention) — no preflight refusal, no spill; budget-aware enforcement is exclusively Phase 6 CCSD-11's charter (avoids two divergent budget checks).

### DF-MP2 vs the open cintx `int3c2e_sph` gap (MP2-04)

- **D-05: DF-MP2 follows the DF-HF precedent — structural code lands now, the bit-exact numeric oracle is CI-gated behind the cintx `int3c2e_sph` merge.** DF-MP2 requires `mol.intor('int3c2e_sph')`, the **same** open cintx gap that left DF-HF's numeric result blocked in Phase 3 (STATE.md Blockers; `cintx#11`; PROJECT.md "cintx-ops `int3c2e_sph` upstream gap blocks numeric DF-HF result"). Phase 5 lands the full DF-MP2 structural code (DF classes, `pyscf-df::DfIntegrals` reuse, helpers, PyO3 surface, structural always-on tests), and CI-gates the `release-oracle` bit-exact assertion behind the cintx merge — exactly as Phase 3 shipped DF-HF. **The DF-MP2 code needs no change when cintx lands.** The phase is NOT timeline-coupled to the external cintx dependency.
  - **In-core RMP2/UMP2 is the un-gated headline deliverable.** It uses the full `int2e` AO integrals, which Phase 2 already ships bit-exact (Phase 2 success criterion 3: `mol.intor('int2e')` matches upstream within cintx oracle tolerance). MP2-01/02/03/05/06/07/08 are achievable and oracle-validated this phase without waiting on cintx.

- **D-06: DF-MP2 fidelity target = BOTH the conventional and native upstream implementations.** *(User chose "Both" over the recommended conventional-only — deliberate scope expansion within the in-scope MP2-04.)*
  - **Conventional (primary / drop-in contract):** port `pyscf/mp/dfmp2.py:DFRMP2` (+ `pyscf/mp/dfump2.py:DFUMP2`). This is what `mp.DFMP2(mf)` and `mf.density_fit().MP2()` actually return (`mp/__init__.py:51`; `scf.hf.RHF.DFMP2 = class_as_method(DFMP2)`), and `DFRMP2` **subclasses `mp2.RMP2`** — so it reuses the in-core RMP2 base and just swaps ERI assembly to `pyscf-df`. This is the primary oracle reference and the drop-in `mp.DFMP2` contract.
  - **Native (additional RI-MP2 fast path):** port `pyscf/mp/dfmp2_native.py` (+ `dfump2_native.py`). A distinct, faster code path exposed via its own module path (`pyscf.mp.dfmp2_native`), not the default factory.
  - Both DF paths need 3-center integrals, so **both numeric oracles gate behind the same cintx `int3c2e_sph` merge** (D-05). Planner should sequence the native path as a follow-on plan after the conventional path proves out, and may stage it behind a status marker (cintx-ECP / D-11 style) if the added surface warrants it.

### MP2 PyO3 surface (BIND inheritance — Pitfall 7 by convention)

- **D-07: Eager SCF-reference snapshot in `pyscf-py`; `pyscf-mp2` stays pyo3-free.** `pyscf-py`'s `PyRMP2`/`PyUMP2` extract `mo_coeff`/`mo_energy`/`mo_occ`/`e_hf` (+ `nocc`/`nmo`) from the Python `mf` and pass **plain Rust arrays** into a pyo3-free `pyscf-mp2` kernel — preserving the D-01 architecture (method crates have zero pyo3 dependency; the bridge lives only in `pyscf-py`), identical to how `pyscf-scf` works. The alternative (holding a live `Py<PyAny>` to `mf` inside `pyscf-mp2`) would force a pyo3 dep into `pyscf-mp2` and break the contract every prior phase established. **`as_scanner` (MP2-07)** re-runs `mf` at the new geometry then re-snapshots, mirroring the SCF-12 scanner-closure pattern (the closure handle is a `Py<PyAny>` from Python's POV).

- **D-08: Focused `Mp2OverrideHooks` trait-callback bridge (same D-01 pattern, smaller hook set).** `pyscf-mp2` declares a `pub trait Mp2OverrideHooks` (pyo3-free); `pyscf-py` provides the bridge routing through `slf.call_method1(py, "<hook>", …)` so Python subclass overrides dispatch natively (Pitfall 7 immune by construction). Hook set covers the methods subclasses realistically override: **`ao2mo`** (custom integral source — the main one), **`make_rdm1`**, **`make_rdm2`**, **`energy`**. This honors the ROADMAP cross-cutting note ("Phases 5/6/7 inherit Pitfall 7 by convention") without re-litigating the full SCF hook set. The MP2 helper functions (`get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`_mo_without_core`, MP2-08) are plain exported functions/methods, not bridged hooks (subclasses rarely override them).

### Claude's Discretion

The following are not user-decided — researcher/planner picks the implementation within the locked decisions above. Default stance: **mirror upstream** (sibling-crate fidelity).

- **Frozen-core semantics (MP2-03)** — mirror upstream `mp2.py` frozen handling: `frozen=int` (count of lowest MOs), `frozen=list` (explicit indices), `frozen='auto'` (chemical-core autodetection via upstream `set_frozen`/chemcore tables), and frozen-window forms. `get_frozen_mask`/`_mo_without_core`/`_mo_splitter` define the mask; defaults must match upstream on the corpus. Planner confirms the `'auto'` core table source.
- **SCS-MP2 (MP2-06)** — same-spin/opposite-spin energy decomposition with `emp2_ss_factor`/`emp2_os_factor` set on the MP2 object; mirror upstream's `energy()` split. Default factors reproduce plain MP2.
- **`make_rdm1`/`make_rdm2` surface (MP2-05)** — mirror upstream `make_rdm1(t2, eris, ao_repr=False, with_frozen=True)` / `make_rdm2(..., ao_repr=False)` incl. the `ao_repr` and `with_frozen` flags and `_gamma1_intermediates`.
- **MP2 helper export call site (MP2-08)** — `get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`_mo_without_core` exported so the Phase 6 CCSD import (`cc/ccsd.py:35`) works verbatim; contract-tested via a unit test that mimics that import.
- **`with_t2` / amplitude retention** — upstream `kernel(..., with_t2=WITH_T2)` optionally returns/stores `t2`; needed by RDMs and (Phase 7) gradients. Planner decides default (upstream keeps `t2` by default for small systems).
- **`mp.MP2()` cross-module factory dispatch** — `mf.MP2()` returns RMP2 for RHF refs, UMP2 for UHF refs (per `mp/__init__.py:MP2`); `mf.density_fit().MP2()` returns conventional DFMP2. Ships in Phase 5 (relates to the SCF-11 `to_uhf`/`to_rhf` dispatch helpers shipped in Phase 3).
- **`_iterative_kernel` vs closed-form** — upstream MP2 is closed-form (single pass); the `_iterative_kernel` path (for non-canonical/Brueckner refs) is likely out of v1 scope unless a fixture needs it. Planner confirms.
- **DF auxbasis defaults** — DF-MP2 reuses `pyscf-df`'s `DEFAULT_AUXBASIS` resolution (Phase 3 D-10); confirm MP2-side aux default matches upstream `dfmp2.py` (`*-ri`/`*-jkfit` per-method default).
- **`canonicalize_signs` reuse** — MP2 consumes the SCF reference's already-canonicalized `mo_coeff` (Pitfall 4/12, SCF-13); no new sign work unless MP2 re-diagonalizes (it does not for canonical MP2).
- **Phase MVP sequencing** — in-core RMP2 (bit-exact, un-gated) first → UMP2 → frozen-core/SCS/RDMs → conventional DF-MP2 (oracle gated) → native DF-MP2 follow-on. Planner finalizes wave structure.

### Folded Todos

None — the todo cross-reference scan returned 0 matches for Phase 5.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project specs (this repo)
- `.planning/PROJECT.md` — vision, core value, key decisions, "out of scope" list; MP2 is an Active v1 requirement; notes the cintx `int3c2e_sph` upstream gap blocking DF numeric results
- `.planning/REQUIREMENTS.md` lines 72-79 (MP2-01..08) + 298-305 (phase mapping) — Phase 5 owns 8 REQs
- `.planning/ROADMAP.md` §"Phase 5: MP2" — goal (incl. "AO→MO transformation kernel general enough to be reused by CCSD" + the MP2 helpers CCSD imports), dependencies (Phase 3; parallelizable with Phase 4), 5 numbered success criteria
- `.planning/ROADMAP.md` §"Cross-Cutting Concerns Threaded Through Every Phase" — algebra-responsibility wall, backend selection, bit-exact-with-PySCF, PyO3 subclass-override dispatch (Phases 5/6/7 inherit by convention), `Python::detach` seam, scope-creep lint, cubecl pin lockstep
- `.planning/ROADMAP.md` §"Pitfall-to-Phase Mapping" — Phase 5 re-validates Pitfall 4 (eigenvector sign, via the SCF reference), Pitfall 1/2 (FMA / reduction order in the AO→MO contraction), Pitfall 5/6 (NumPy boundary + GIL in the PyO3 surface)
- `.planning/STATE.md` §"Blockers/Concerns" — the cintx `int3c2e_sph` / arity-4 int2e gap (`cintx#11`) gating DF numeric parity (drives D-05); `PYSCF_MAX_MEMORY` enforcement is Phase 6's
- `.planning/phases/04-dft/04-CONTEXT.md` — D-01..08 carried: algebra-orchestrated `gemm` hot path (D-07), own-crate-per-shared-concern (D-05 grids), `PYSCF_DTYPE` f32/f64 precision seam, libxc compile-cost discipline (irrelevant to MP2 but the build-time constraint stands)
- `.planning/phases/03-scf-pyo3-bindings/03-CONTEXT.md` — D-01..11 carried: `OverrideHooks` trait-callback bridge (D-01, the model for `Mp2OverrideHooks`), per-hook `Python::detach` (D-03), type-specific NumPy converters (D-04), `pyscf-df` crate + in-memory B-integrals / HDF5 spill deferred to Phase 6 (D-10/D-11), `canonicalize_signs`, `as_scanner` shape (SCF-12), test-corpus tiering
- `.planning/phases/02-gto/02-CONTEXT.md` — F-order layout convention (Pitfall 8), `mol.intor` dispatcher (`int2e` available bit-exact; `int3c2e_sph` is the DF gap), "port the reference algorithm" precedent
- `.planning/phases/01-foundation/01-CONTEXT.md` — `AlgebraClient` enum + match dispatch, host-faer eigh, `PYSCF_BACKEND`/`PYSCF_DTYPE` resolvers, `release-oracle` FMA-free + ordered-reduction infra, cubecl `=0.10.0` pin

### Upstream PySCF source (this repo — the oracle / port reference)
- `pyscf/mp/__init__.py` — `MP2`/`RMP2`/`UMP2`/`GMP2` factories; `MP2(mf)` routes to `dfmp2.DFMP2` when `mf` is density-fitted (line 51) — the cross-module dispatch contract
- `pyscf/mp/mp2.py` (32943 B) — `RMP2`/`MP2Base`, `kernel`, `energy`, `update_amps`, `make_rdm1`/`make_rdm2`, `_gamma1_intermediates`, `_mo_splitter`, `get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`_mo_without_core`/`_mo_energy_without_core` (MP2-08 source-of-truth), `as_scanner`/`MP2_Scanner`, `_ChemistsERIs`, `_make_eris`, `_ao2mo_ovov` (the in-core AO→MO path D-02/D-03 mirrors), `make_fno` (deferred)
- `pyscf/mp/ump2.py` (32201 B) — `UMP2` open-shell: α/β spin-block MO transform, spin-resolved frozen mask + RDMs (MP2-02)
- `pyscf/mp/dfmp2.py` — `DFRMP2(mp2.RMP2)`, `_DFINCOREERIS`/`_DFOUTCOREERIS`, `MP2 = DFMP2 = DFRMP2`, `scf.hf.RHF.DFMP2 = class_as_method(DFMP2)` — **conventional DF-MP2, the default factory target + primary oracle (D-06)**
- `pyscf/mp/dfump2.py` — `DFUMP2` open-shell conventional DF-MP2 (D-06)
- `pyscf/mp/dfmp2_native.py` — **native RI-MP2 fast path, RHF reference (D-06 additional target)**; `MP2 = RMP2 = DFMP2 = DFRMP2` aliases within its own module
- `pyscf/mp/dfump2_native.py` — native RI-MP2, UHF reference (D-06)
- `pyscf/ao2mo/__init__.py` — `ao2mo.general`, `ao2mo.kernel`, `ao2mo.full` public surface (D-02 port target)
- `pyscf/ao2mo/incore.py` — in-core full AO→MO transform (`full`, `general`, `half_e1`) — **D-02/D-04 primary port target**
- `pyscf/ao2mo/_ao2mo.py` — low-level transform driver (`nr_e1`/`nr_e2`, AO-pair packing) — algorithm reference for the `gemm`-chain port (D-03)
- `pyscf/ao2mo/addons.py` — `restore` (4-fold/8-fold symmetry packing), `load` helpers — confirm symmetry-packing fidelity
- `pyscf/ao2mo/outcore.py`, `pyscf/ao2mo/semi_incore.py` — **Phase 6 reference (deferred D-04); read for the general-surface shape only, do not port in Phase 5**

### Phase 1–4 shipped artifacts (this repo)
- `crates/pyscf-mp2/src/lib.rs` — Phase 1 stub (empty, `#![forbid(unsafe_code)]`); Phase 5 fills with `RMP2`/`UMP2`/`DFMP2` + `Mp2OverrideHooks` + helpers
- `crates/pyscf-scf/src/lib.rs` — Phase 3 `RHF`/`UHF` + `OverrideHooks` + generic `kernel<H>` + `ScfResult` (the converged reference MP2 snapshots from, D-07); `as_scanner` precedent (SCF-12)
- `crates/pyscf-df/src/lib.rs` — `DfIntegrals`/`get_jk_df`/`DEFAULT_AUXBASIS`/`density_fit` (Phase 3 D-10); DF-MP2 reuses the B-integral assembly (in-memory; HDF5 spill is Phase 6)
- `crates/pyscf-gto/src/lib.rs` — `mol.intor(name)` dispatcher: `int2e` (in-core RMP2, bit-exact) + `int3c2e_sph`/`int2c2e_sph` (DF-MP2, cintx-gated)
- `crates/pyscf-algebra/src/lib.rs` — `gemm`/`gemv`/`reduce_sum`/`oracle_sum`/`oracle_dot`/`eigh` — the entire AO→MO contraction (D-03) + energy reduction go through this
- `crates/pyscf-core/src/{mo.rs,density.rs,traits.rs}` — `MOCoefficients`/`Density`; `PostScf` trait declared (Phase 1) — Phase 5 implements it for MP2; `canonicalize_signs` (SCF-13) consumed via the reference
- `crates/pyscf-py/src/lib.rs` — `#[pymodule] _native` + `PyOverrideBridge` (Phase 3 D-01); Phase 5 adds `PyRMP2`/`PyUMP2`/DF variants + the `mp` submodule + `python/pyscf/mp/__init__.py` overlay; `mf.MP2()` dispatch hook
- `crates/pyscf-oracle/` — `oracle_check!` macro (Phase 3 D-07/ORACLE-02); Phase 5 adds MP2 correlation-energy fixtures (in-core un-gated; DF cintx-gated)
- `Cargo.toml` (workspace) — Phase 5 adds member `crates/pyscf-ao2mo` (19→20, D-01); wire its deps (`pyscf-core`, `pyscf-algebra`, `pyscf-gto`)
- `xtask/src/lints/algebra_wall.rs` — extend allowlist for `pyscf-ao2mo` (algebra dep, no direct cubecl) + `pyscf-mp2` (algebra + ao2mo + df, no direct cubecl)
- `.github/workflows/ci.yml` — add MP2 oracle jobs: in-core RMP2/UMP2 always-on bit-exact; DF-MP2 numeric assertion `#[cfg]`/CI-gated behind the cintx `int3c2e_sph` merge (DF-HF precedent)

### Sibling-crate / PyO3 precedent (read before implementing analogous surface)
- `~/Documents/workspace/cintx/` — `int2e` / `int3c2e_sph` integral source; `cintx#11` tracks the open `int3c2e_sph` gap gating DF numeric parity (D-05)
- `~/Documents/workspace/xcfun_rs/crates/xcfun-py/` + `~/Documents/workspace/cintx/crates/cintx-rs/` — PyO3 0.28 `#[pyclass]`/`#[pymethods]`, `Python::detach`, NumPy boundary patterns (mirror for `PyRMP2`/`PyUMP2`)

### Cubecl + numerics reference docs (this repo)
- `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` — authoritative for `pyscf_algebra::gemm` in the AO→MO contraction (D-03)
- `docs/manual/Cubecl/Cubecl_multi_ compute.md` — runtime/ComputeClient pattern for any future fused `ao2mo` kernel (Phase 8, deferred)

### External (Phase 5 will look up)
- MP2 / RI-MP2 method references — the `t2 = ⟨ij|ab⟩ / (εi+εj−εa−εb)` amplitude form, SCS-MP2 SS/OS factors, only if cross-checking the upstream port

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase 3 SCF `ScfResult`** (`pyscf-scf`) — the converged reference (`mo_coeff`/`mo_energy`/`mo_occ`/`e_tot`); MP2 snapshots it via `pyscf-py` (D-07). No SCF re-run except in `as_scanner`.
- **`pyscf-algebra` surface** — `gemm`/`reduce_sum`/`oracle_sum`/`oracle_dot` cover the entire AO→MO contraction (D-03) + the MP2 energy reduction, bit-exact under `release-oracle`.
- **`pyscf-df::DfIntegrals` + `DEFAULT_AUXBASIS`** (Phase 3 D-10) — DF-MP2 reuses the in-memory B-integral assembly for both conventional and native paths (D-06); HDF5 spill is Phase 6.
- **`OverrideHooks` trait-callback bridge** (Phase 3 D-01) — the template for `Mp2OverrideHooks` (D-08); `pyscf-mp2` stays pyo3-free, `pyscf-py` owns the bridge.
- **`as_scanner` scanner-closure** (Phase 3 SCF-12) — the pattern MP2's `as_scanner` (MP2-07) follows (re-run `mf` → re-snapshot → MP2 kernel).
- **`canonicalize_signs`** (`pyscf-core`, SCF-13) — already applied to the reference `mo_coeff`; MP2 inherits vendor-stable signs (Pitfall 4/12).
- **`mol.intor('int2e')`** (Phase 2) — bit-exact full AO ERIs; the un-gated in-core RMP2/UMP2 path depends on it.
- **`oracle_check!` macro** (Phase 3 D-07) — MP2 correlation-energy oracle fixtures reuse the macro shape.

### Established Patterns
- **Algebra wall** — `pyscf-ao2mo` + `pyscf-mp2` depend on `pyscf-algebra` only, never `cubecl-*` directly; xtask lint extended to both.
- **Sibling-crate fidelity (hard preference)** — `pyscf-ao2mo` mirrors `pyscf/ao2mo/`; `pyscf-mp2` mirrors `pyscf/mp/mp2.py`/`ump2.py`/`dfmp2.py`/`dfmp2_native.py`. The own-crate split for `pyscf-ao2mo` is consistent with the D-05/D-08/D-10 split-out precedent (upstream also keeps `ao2mo` separate from `mp`).
- **Method crates stay pyo3-free; bridge in `pyscf-py`** (Phase 3 D-01) — drives D-07 (eager snapshot) + D-08 (`Mp2OverrideHooks` bridge).
- **DF numeric parity gated behind cintx** (DF-HF precedent, Phase 3) — DF-MP2 lands structurally now, oracle CI-gated (D-05).
- **In-memory ERIs now, HDF5 spill in Phase 6** (Phase 3 D-11) — drives D-04.
- **`PYSCF_MAX_MEMORY` log-only** (Phase 3/4) — drives D-04's no-enforcement stance.
- **Bit-exact-with-upstream under `release-oracle`** — in-core MP2 energy ≤ bit-exact (un-gated); DF-MP2 same bar (gated).
- **"Don't freeze compile"** (user memory) — no heavy build.rs / parse-N-files macros; `pyscf-ao2mo` ports the reference algorithm (D-03), no codegen. `libxc_rs` stays out of the MP2 dep graph entirely.

### Integration Points
- **`crates/pyscf-ao2mo/` (new, 20th member)** — deps `pyscf-core`, `pyscf-algebra`, `pyscf-gto`; ports `ao2mo/incore.py` + `_ao2mo.py` + `addons.py` (general/full surface, D-02).
- **`crates/pyscf-mp2/Cargo.toml`** — deps `pyscf-core`, `pyscf-algebra`, `pyscf-ao2mo` (new), `pyscf-scf` (reference types), `pyscf-df` (DF-MP2), `pyscf-gto`, `pyscf-runtime`, `tracing`, `thiserror`. **No pyo3 dep** (D-07/D-08).
- **`crates/pyscf-py/`** — adds `PyRMP2`/`PyUMP2` + conventional/native DF variants + `mp` submodule + `python/pyscf/mp/__init__.py` overlay; `mf.MP2()`/`mf.density_fit().MP2()` dispatch; reuses Phase 3 NumPy converters + `Python::detach`.
- **`Cargo.toml` workspace** — register `crates/pyscf-ao2mo` (19→20); ROADMAP.md member count update.
- **`.github/workflows/ci.yml`** — in-core MP2 oracle (always-on); DF-MP2 oracle (cintx-gated, mirrors the Phase 3 DF-HF gating); MP2-08 helper contract test (CCSD import call-site mimic).
- **`~/Documents/workspace/cintx` (`cintx#11`)** — the `int3c2e_sph` cross-repo gap that gates DF-MP2 numeric parity (D-05); no new pyscf-rs work, just the gate.

</code_context>

<specifics>
## Specific Ideas

- **In-core RMP2 is the un-gated headline of the phase** — it rides on `mol.intor('int2e')` (bit-exact since Phase 2). The whole phase ships meaningful, oracle-validated value without waiting on the cintx DF gap. DF-MP2's numeric oracle is the only gated piece (D-05).
- **`pyscf-ao2mo` as its own crate is the structural keystone** — the ROADMAP explicitly demands the AO→MO kernel be CCSD-reusable; putting it in its own crate (D-01) keeps the Phase 6 `pyscf-ccsd → pyscf-ao2mo` dependency clean and avoids a backwards `ccsd → mp2` edge.
- **`DFRMP2` subclasses `RMP2` upstream** — so in-core RMP2 is genuinely the base, and conventional DF-MP2 (D-06) is "swap the ERI source," not a parallel rewrite. This makes the conventional path the natural primary fidelity target.
- **User wants BOTH DF-MP2 variants** (D-06) — conventional (drop-in `mp.DFMP2` contract + primary oracle) AND native RI-MP2 (fast path). This is a deliberate scope expansion; planner sequences native as a follow-on and may stage it behind a status marker. Both gate on the same cintx merge.
- **MP2-08 helpers are a hard CCSD interface contract** — `get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`_mo_without_core` must export with the exact semantics `cc/ccsd.py:35` imports; the contract test mimics that call site verbatim.
- **Defer optimization** (D-03) — algebra-`gemm` chain now; the fused cubecl `ao2mo` kernel is Phase 8's, matching the Phase 3 `pyscf-df` D-10 / Phase 4 D-07 stance.

</specifics>

<deferred>
## Deferred Ideas

- **Fused cubecl AO→MO transform kernel** — Phase 8 (the 2–5× benchmark + GPU owner), only if profiling shows the algebra-`gemm` chain (D-03) is the bottleneck.
- **Outcore / semi-incore HDF5-spilling AO→MO** (`ao2mo/outcore.py`, `semi_incore.py`) — Phase 6, alongside CCSD-08 + CCSD-11's tensor-arena + spill from day one (D-04).
- **`PYSCF_MAX_MEMORY` budget-aware refusal/spill for post-SCF** — Phase 6 CCSD-11 (D-04). Phase 5 logs only.
- **GMP2 / DFGMP2 (GHF-reference MP2)** (`gmp2.py`, `dfgmp2.py`) — no v1 MP2 requirement maps to them; revisit if a later requirement adds GHF post-SCF.
- **MP2-F12** (`mp2f12_slow.py`) — explicitly out of v1.
- **FNO-MP2 / `make_fno`** (frozen-natural-orbital truncation) — not in MP2-01..08; deferred.
- **`_iterative_kernel` (non-canonical/Brueckner MP2)** — likely out of v1 scope; planner confirms no corpus fixture needs it.
- **cintx `int3c2e_sph` gap-closure** (`cintx#11`) — cross-repo dependency; lands independently. Unblocks the DF-MP2 numeric oracle (and the DF-HF / DF-DFT ones) when merged. No pyscf-rs code change required (D-05).

### Reviewed Todos (not folded)
None — the todo cross-reference scan returned 0 matches for Phase 5.

</deferred>

---

*Phase: 05-mp2*
*Context gathered: 2026-05-23*
