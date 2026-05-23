# Phase 5: MP2 - Research

**Researched:** 2026-05-23
**Domain:** Møller–Plesset 2nd-order perturbation theory (RMP2/UMP2/DF-MP2), AO→MO integral transformation, post-SCF Rust method-crate layering, PyO3 subclass-override bridge
**Confidence:** HIGH (algorithm port + existing-codebase contracts source-verified in-repo; one CONTEXT assumption corrected — see Open Question 1)

## Summary

Phase 5 ports four upstream PySCF modules (`mp/mp2.py`, `mp/ump2.py`, `mp/dfmp2.py`, `mp/dfmp2_native.py`) plus the `ao2mo` integral transformation (`ao2mo/incore.py` + `_ao2mo.py`) into two new/filled Rust crates: a new `pyscf-ao2mo` (20th workspace member) and the currently-empty `pyscf-mp2`. The architecture is fully determined by precedent already shipped in Phases 3–4: method crates stay pyo3-free and consume `pyscf-algebra` only (algebra wall); the PyO3 bridge + subclass-override dispatch lives exclusively in `pyscf-py`; all reductions go through `oracle_sum`/`oracle_dot` for bit-exactness under `release-oracle`; and the Tensor-API `gemm` is still `NotYetImplemented{phase:2}`, so the AO→MO contraction must be written as explicit host loops with `oracle_sum` reductions exactly like the Phase-4 DFT grid loop (this *is* what D-03 means in practice — there is no working `pyscf_algebra::gemm` to chain).

**The single most important correction to CONTEXT:** CONTEXT D-05 / the `<specifics>` block assert that "in-core RMP2/UMP2 is the un-gated headline deliverable" because `mol.intor('int2e')` is "bit-exact since Phase 2." This is **not true in the current codebase.** `pyscf-gto::intor` returns `NotYetImplemented{phase:2}` for *all* arity-3 and arity-4 integrals (`crates/pyscf-gto/src/intor.rs:181-185`), and `int2e_sph` is explicitly xfail-tracked behind the **same cintx safe-API arity>2 gap** that blocks DF-HF/DF-DFT (`cintx#11`). So in-core RMP2 numeric parity is gated by the *conventional* `int2e` half of `cintx#11`, and DF-MP2 by the *3-center* `int3c2e_sph` half. The planner should ship both in-core and DF MP2 structurally (full Rust algorithm + always-on shape/wiring tests) with their numeric oracle assertions CI-gated — identical to how Phase 3 shipped DF-HF and Phase 4 shipped DFT-01. The phase is still fully deliverable; the framing "in-core is un-gated" needs to flip to "in-core and DF are both numeric-gated behind `cintx#11`, but the entire algorithm + helper surface + structural tests land now."

**Primary recommendation:** Build `pyscf-ao2mo` mirroring `ao2mo/incore.py` (`general`/`full`, host-loop einsum-equivalent through `oracle_sum`, F-order in/out), fill `pyscf-mp2` with `RMP2`/`UMP2`/`DFRMP2`/native-DF as a near-verbatim port of the four upstream `.py` files (closed-form canonical kernel, the MP2-08 helpers as plain free functions, SCS factors in `energy()`, RDMs via `_gamma1_intermediates`), add the `Mp2OverrideHooks` trait + `PyRMP2`/`PyUMP2`/`PyDFMP2` bridge in `pyscf-py` modeling `PyRHF`, and CI-gate every numeric MP2 oracle behind `cintx#11` while keeping structural tests always-on.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| AO→MO 4-index transform `(pq\|rs)→(ia\|jb)` | `pyscf-ao2mo` (new) | `pyscf-algebra` (host-loop contraction + `oracle_sum`) | Standalone crate so CCSD imports it without a `ccsd→mp2` edge (D-01); contraction is bit-exact ordered reduction |
| MP2 correlation energy + amplitudes | `pyscf-mp2` | `pyscf-ao2mo` (ERIs), `pyscf-scf` (reference types), `pyscf-algebra` (reductions) | Method-crate; consumes converged SCF reference + transformed ERIs |
| Frozen-core mask / `nocc`/`nmo` helpers (MP2-08) | `pyscf-mp2` (plain pub functions) | — | CCSD imports these from the MP2 crate verbatim (`cc/ccsd.py:35`) |
| DF-MP2 ERI assembly | `pyscf-mp2` (DF classes) | `pyscf-df` (`DfIntegrals` B-tensor, in-memory) | Conventional `DFRMP2` subclasses `RMP2`, swaps ERI source to `pyscf-df` |
| `mf.MP2()` / `mp.RMP2(mf)` / DF factory dispatch | `pyscf-py` (`mp` submodule + `python/pyscf/mp/__init__.py` overlay) | `pyscf-scf` (`to_rhf`/`to_uhf` SCF-11) | Python-facing factory + cross-module dispatch is a binding concern |
| Subclass-override dispatch (`ao2mo`/`make_rdm1`/`make_rdm2`/`energy`) | `pyscf-py` (`Mp2OverrideHooks` bridge via `call_method1`) | `pyscf-mp2` (declares pyo3-free trait) | Pitfall 7; mirrors Phase-3 `OverrideHooks`/`PyOverrideBridge` |
| Eager SCF-reference snapshot + `as_scanner` | `pyscf-py` (`PyRMP2`/`PyUMP2` extract plain arrays) | `pyscf-scf` (`as_scanner` SCF-12 closure) | Method crate stays pyo3-free; bridge holds the `Py<PyAny>` (D-07) |
| Numeric ERI source (the gating dependency) | `cintx` (`int2e` + `int3c2e_sph`) | — | `cintx#11` arity>2 gap blocks both conventional and DF numeric parity |

## Standard Stack

This is a pure-Rust port phase. **No new external crates are introduced.** Every dependency is either an internal workspace path-crate or an already-pinned `[workspace.dependencies]` entry. [VERIFIED: `Cargo.toml` workspace deps + `crates/pyscf-scf/Cargo.toml` dep template, read this session]

### Core

| Crate | Version | Purpose | Why Standard |
|-------|---------|---------|--------------|
| `pyscf-ao2mo` (NEW) | 0.1.0 (workspace) | AO→MO 4-index transform (`general`/`full`); CCSD-reusable | D-01: own crate keeps `ccsd→ao2mo` clean (mirrors upstream `pyscf/ao2mo/`) |
| `pyscf-mp2` (FILL) | 0.1.0 (workspace) | RMP2/UMP2/DFRMP2 + helpers + RDMs + SCS | Phase-5 deliverable; currently an empty `#![forbid(unsafe_code)]` stub |
| `pyscf-algebra` | 0.1.0 (workspace) | `oracle_sum`/`oracle_dot` (bit-exact reductions); host `eigh` via faer | Single owner of linear algebra; **`gemm` Tensor-API is still `NotYetImplemented{phase:2}`** |
| `pyscf-scf` | 0.1.0 (workspace) | `ScfResult` reference (`mo_coeff`/`mo_energy`/`mo_occ`/`e_tot`); `OverrideHooks` model; `as_scanner` (SCF-12) | MP2 snapshots the converged reference from here |
| `pyscf-df` | 0.1.0 (workspace) | `DfIntegrals { b_uvq, naux, nao }`, `DEFAULT_AUXBASIS`, `default_ri`/`default_jkfit` | DF-MP2 reuses the in-memory B-tensor assembly |
| `pyscf-gto` | 0.1.0 (workspace) | `intor(mol, name)`; `intor_with_auxmol`; `M(MoleBuildArgs)` | ERI source — **arity-4 `int2e` + arity-3 `int3c2e_sph` both gated (`cintx#11`)** |
| `pyscf-core` | 0.1.0 (workspace) | `Mole`, `MOCoefficients` (F-order), `Density`, `Energy`, `Amplitudes`, `PostScf` trait, `PyscfRsError` | Universal types; `PostScf` declared Phase 1, MP2 implements it |

### Supporting

| Crate | Version | Purpose | When to Use |
|-------|---------|---------|-------------|
| `pyscf-py` | 0.1.0 (workspace) | PyO3 bridge: `PyRMP2`/`PyUMP2`/`PyDFMP2` + `mp` submodule + `Mp2OverrideHooks` bridge | All Python-facing surface + subclass dispatch |
| `pyscf-runtime` | 0.1.0 (workspace) | `WorkspacePool`, `DType` (f32/f64 seam), backend resolver | `PYSCF_MAX_MEMORY` log-only; precision seam if reused |
| `pyscf-oracle` | 0.1.0 (workspace) | `oracle_check!` macro + `run_oracle_check` (pyo3 in **dev-deps only**, `--features python`) | MP2 correlation-energy fixtures (CI-gated) |
| `thiserror` | 2.0.18 (pinned) | Error enums | `pyscf-mp2`/`pyscf-ao2mo` error types |
| `tracing` | 0.1.44 (pinned) | Verbosity-contract logging (FOUND-09); `PYSCF_MAX_MEMORY` budget log | Kernel-entry budget log (D-04: log-only) |
| `pyo3` / `numpy` | 0.28.3 / 0.28.0 (pinned) | **`pyscf-py` only** — FORBIDDEN in `pyscf-mp2`/`pyscf-ao2mo` (D-07/D-08 algebra/pyo3 wall) | The bridge layer |
| `approx` / `rstest` | 0.5.1 / 0.26.1 | Test assertions / fixtures | Unit + contract tests |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Host-loop `oracle_sum` contraction (D-03) | `pyscf_algebra::gemm` (cubecl matmul) | `gemm` is `NotYetImplemented{phase:2}` — not an option this phase; the host-loop precedent is Phase-4 DFT numint and Phase-3 DF/SCF (all inline loops). Fused cubecl `ao2mo` is Phase-8, deferred. |
| Port `dfmp2.py`'s C driver `libmp.MP2_contract_d` as a kernel | Re-express the contraction as a pure-Rust host loop through `oracle_sum` | Upstream calls a compiled C contraction (`libmp`); pyscf-rs has no C deps (core value). Port the *math*, not the C call. Matches the Pitfall-4 VV10 precedent (port the pure-Python loop, not the C symbol). |
| `_iterative_kernel` (DIIS amplitude iteration) | Closed-form single-pass canonical MP2 | Upstream uses closed-form for canonical SCF refs (`self._scf.converged` branch). Iterative path only for non-canonical refs — out of v1 scope unless a corpus fixture needs it (Open Question 3). |
| New external linalg crate | Existing `pyscf-algebra` surface | Out of scope per REQUIREMENTS "no `ndarray-linalg`/`nalgebra-lapack`" (pull system BLAS, defeat no-C-deps goal). |

**Installation:** None. Add `crates/pyscf-ao2mo` to workspace `members` (19→20 — note `Cargo.toml` currently lists **20** member directories already, but the *prose* in ROADMAP says 19; reconcile during planning) and wire `pyscf-mp2`/`pyscf-ao2mo` dep entries. No `cargo add` of any crates.io package.

**Version verification:** N/A — no registry packages added. All crate versions inherit `version.workspace = true` (0.1.0). External pins (`thiserror=2.0.18`, `tracing=0.1.44`, `pyo3=0.28.3`, `faer=0.24.0`) are already locked in `[workspace.dependencies]` and verified present this session.

## Package Legitimacy Audit

This phase installs **no external packages**. All dependencies are internal workspace path-crates (`pyscf-*`) and already-pinned workspace dependencies that shipped in Phases 1–4. slopcheck was available this session but there is nothing new to audit.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| (none added) | — | — | — | — | n/a | No new external packages |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*If a planner later proposes any crates.io dependency for this phase, it must run the Package Legitimacy Gate first — but the entire design (port upstream algorithm through existing crates) requires zero new packages, and the "no-C-deps / no system BLAS" core value forbids most candidates anyway.*

## Architecture Patterns

### System Architecture Diagram

```
                          Python user script
   mp.RMP2(mf).kernel()  |  mf.MP2().run()  |  mp.DFMP2(mf).kernel()  |  mp2.as_scanner()(mol)
            │                     │                      │                      │
            ▼                     ▼                      ▼                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │  pyscf-py :: mp submodule  (the ONLY pyo3 layer)                                   │
 │  • python/pyscf/mp/__init__.py overlay → MP2()/RMP2()/UMP2() factory dispatch      │
 │  • PyRMP2 / PyUMP2 / PyDFMP2  (#[pyclass(subclass)])                               │
 │  • factory: mf.istype(UHF)→UMP2 ; mf.with_df→DFMP2  (mirror mp/__init__.py)        │
 │  • EAGER SNAPSHOT (D-07): extract mo_coeff/mo_energy/mo_occ/e_hf/nocc/nmo          │
 │    from the Python mf → plain Rust arrays                                          │
 │  • Mp2OverrideHooks bridge (D-08): ao2mo / make_rdm1 / make_rdm2 / energy          │
 │    routed via slf.call_method1(py,"<hook>",..)   (Pitfall 7 immune)               │
 │  • as_scanner: hold Py<PyAny> mf-scanner → re-run mf(mol) → re-snapshot (SCF-12)   │
 └───────────────────────────────────┬───────────────────────────────────────────────┘
                  plain Rust arrays   │   (NO pyo3 below this line — algebra/pyo3 wall)
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │  pyscf-mp2  (pyo3-FREE method crate)                                               │
 │  • RMP2 / UMP2 / DFRMP2(:RMP2) / dfmp2_native path                                 │
 │  • kernel: e_hf + closed-form e_corr  (canonical: εi+εj−εa−εb denominators)        │
 │  • energy(): SCS split  e_ss/e_os, emp2_ss_factor/emp2_os_factor  (MP2-06)         │
 │  • helpers (MP2-08, plain pub fns): get_nocc get_nmo get_frozen_mask               │
 │       get_e_hf _mo_without_core (+ _mo_energy_without_core _mo_splitter)           │
 │  • make_rdm1 / make_rdm2 via _gamma1_intermediates  (MP2-05)                       │
 │  • frozen-core: int / list / 'auto'(chemcore) / window  (MP2-03)                   │
 └──────────┬───────────────────────────────┬───────────────────────┬────────────────┘
            │ (ia|jb) MO ERIs                │ converged reference   │ B-tensor (DF)
            ▼                                ▼                       ▼
 ┌────────────────────────┐   ┌──────────────────────────┐  ┌─────────────────────────┐
 │  pyscf-ao2mo (NEW)      │   │  pyscf-scf :: ScfResult   │  │  pyscf-df :: DfIntegrals │
 │  general()/full()       │   │  mo_coeff/mo_energy/      │  │  b_uvq[nao,nao,naux]     │
 │  4-index transform as   │   │  mo_occ/e_tot (canonical, │  │  DEFAULT_AUXBASIS,       │
 │  host loops + oracle_sum │   │  sign-canonicalized SCF-13)│  │  default_ri/default_jkfit │
 └───────────┬─────────────┘   └──────────────────────────┘  └────────────┬────────────┘
             │ AO ERIs                                                     │ int3c2e_sph
             ▼                                                             ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │  pyscf-gto :: intor   →   cintx                                                    │
 │  int2e (arity-4)  ── NotYetImplemented{phase:2} ── ┐                               │
 │  int3c2e_sph (arity-3) ── NotYetImplemented{phase:2} ┤  GATED by cintx#11          │
 │  int2c2e_sph (arity-2 aux) ── available             ┘  (arity>2 safe-API gap)      │
 └─────────────────────────────────────────────────────────────────────────────────┘
```
Trace the headline use case (`mp.RMP2(mf).kernel()`): Python factory → `PyRMP2` snapshots `mf` → pyo3-free `pyscf-mp2::RMP2::kernel` asks `pyscf-ao2mo` for `(ia|jb)` → ao2mo asks `pyscf-gto::intor("int2e")` for AO ERIs → **today returns `NotYetImplemented{phase:2}`** → numeric oracle CI-gated; structural test asserts the call shape + error propagation. When `cintx#11` lands, the same code returns the bit-exact energy with no change.

### Recommended Project Structure
```
crates/pyscf-ao2mo/                   # NEW — 20th member (D-01)
├── Cargo.toml                        # deps: pyscf-core, pyscf-algebra, pyscf-gto (NO pyo3, NO cubecl)
└── src/
    ├── lib.rs                        # pub use general, full, kernel
    ├── incore.rs                     # port ao2mo/incore.py: general()/full()
    ├── transform.rs                  # the 4-index host-loop contraction (_ao2mo.py nr_e1/nr_e2 math)
    └── addons.rs                     # restore() 4-fold/8-fold packing (only if a call site needs it)

crates/pyscf-mp2/src/                 # FILL — currently empty stub
├── lib.rs                            # pub RMP2/UMP2/DFRMP2 + helpers + Mp2OverrideHooks (pyo3-free)
├── mp2.rs                            # RMP2 kernel/energy/init_amps + _ChemistsERIs (port mp2.py)
├── ump2.rs                           # UMP2 spin-block kernel/energy (port ump2.py)
├── dfmp2.rs                          # DFRMP2(:RMP2) conventional — swap ERI source to pyscf-df
├── dfmp2_native.rs                   # native RI-MP2 fast path (port dfmp2_native.py) — follow-on plan
├── helpers.rs                        # MP2-08: get_nocc/get_nmo/get_frozen_mask/get_e_hf/_mo_without_core
├── frozen.rs                         # frozen=int/list/'auto'(chemcore)/window (MP2-03)
├── rdm.rs                            # make_rdm1/make_rdm2 + _gamma1_intermediates (MP2-05)
├── hooks.rs                          # Mp2OverrideHooks trait + NoMp2Overrides default impl (D-08)
└── error.rs                          # Mp2Error → From<Mp2Error> for PyscfRsError

crates/pyscf-py/src/
└── mp.rs                             # NEW: PyRMP2/PyUMP2/PyDFMP2 + mp submodule + bridge (model: scf.rs)

python/pyscf/mp/
└── __init__.py                       # NEW overlay: MP2/RMP2/UMP2 factory re-export (model: scf/__init__.py)
```

### Pattern 1: pyo3-free method crate + eager snapshot bridge (D-07)
**What:** `pyscf-mp2` takes plain Rust arrays; `pyscf-py::PyRMP2` extracts them from the Python `mf` before calling in.
**When to use:** Every MP2 entry point.
**Example:**
```rust
// pyscf-mp2 (pyo3-free) — kernel signature takes the snapshotted reference.
// Source: port of pyscf/mp/mp2.py:kernel + MP2Base.kernel (lines 33-76, 610-648)
pub struct Mp2Reference {
    pub mo_coeff: MOCoefficients,   // F-order, already sign-canonicalized (SCF-13)
    pub mo_energy: Vec<f64>,
    pub mo_occ: Vec<f64>,
    pub e_hf: f64,
    pub converged: bool,
}
pub fn rmp2_kernel<H: Mp2OverrideHooks>(
    refr: &Mp2Reference, frozen: &Frozen, hooks: &H, with_t2: bool,
) -> Result<(f64 /*e_corr*/, f64 /*e_ss*/, f64 /*e_os*/, Option<Amplitudes>), PyscfRsError> {
    let nocc = get_nocc(&refr.mo_occ, frozen)?;
    let nmo  = get_nmo(&refr.mo_occ, frozen)?;
    let eris = hooks.ao2mo(refr, frozen)?;        // (ia|jb) block from pyscf-ao2mo
    // closed-form: t2_ijab = (ia|jb) / (ei+ej-ea-eb); e = sum 2*(ia|jb)t2 - (ib|ja)t2
    // all sums via oracle_sum / oracle_dot (Pitfall 1/2 bit-exact)
    todo!("port mp2.py:kernel lines 54-76 as host loops through oracle_sum")
}
```
The `PyRMP2::new` mirrors `PyRHF::new` (`crates/pyscf-py/src/scf.rs:61`): `extract_mole_from_pyany` then snapshot the `mf` attributes. `kernel`/`run` mirror `PyRHF::kernel`/`run` (scf.rs:320,399).

### Pattern 2: AO→MO transform as host-loop einsum (D-03)
**What:** The 4-index transform with the simplest correct path — when the AO ERI is the full `nao^4` tensor (what `mol.intor("int2e")` returns), upstream does `einsum('pqrs,pi,qj,rk,sl->ijkl')` (`ao2mo/incore.py:126`). Port as a sequence of single-index host-loop transforms (each `oracle_sum`-reduced) to control memory and reduction order.
**When to use:** `pyscf-ao2mo::general`; MP2 calls it with `(co, cv, co, cv)` to get the `(ia|jb)` block (`mp2.py:_make_eris` line 793-801).
**Example:**
```rust
// Source: pyscf/ao2mo/incore.py:general (the eri_ao.size == nao**4 branch, line 125-128)
// Quarter-transform sequence (memory-bounded vs the full 4-index einsum):
//   (pq|rs) --C_p--> (iq|rs) --C_q--> (ij|rs) --C_r--> (ij|ks) --C_s--> (ij|kl)
// Each contraction index reduced with oracle_sum to stay bit-exact under release-oracle.
// F-order in/out (MOCoefficients.data is column-major — pyscf-core/src/mo.rs).
```
**Reduction-order discipline:** every contraction sum MUST go through `pyscf_algebra::oracle_sum`/`oracle_dot` (fixed pairwise chunk N=128, thread-count invariant). This is exactly the Phase-4 numint precedent (`crates/pyscf-dft/src/numint.rs` — "dense contractions are explicit host loops with `oracle_sum`").

### Pattern 3: Conventional DF-MP2 = swap ERI source (D-06)
**What:** Upstream `DFRMP2` *subclasses* `mp2.RMP2` (`dfmp2.py:124`); it reuses the RMP2 base and overrides `ao2mo` to assemble `(ia|jb)` from the DF 3-tensor instead of the full ERI. In Rust: `DFRMP2` is a thin wrapper that provides a different `Mp2OverrideHooks::ao2mo` implementation backed by `pyscf-df::DfIntegrals.b_uvq`.
**When to use:** `mp.DFMP2(mf)` / `mf.density_fit().MP2()`.
**Example:**
```rust
// (ia|jb) = sum_Q B^Q_ia · B^Q_jb   where B^Q_ia is the MO-transformed DF B-tensor.
// Upstream conventional path uses C driver libmp.MP2_contract_d; port the contraction
// as a pure-Rust oracle_sum loop (NO C dep). B^Q in MO basis = ao2mo of pyscf_df b_uvq.
// Source: pyscf/mp/dfmp2.py:kernel (lines 39-122) + _make_df_eris (215+).
```
Native path (`dfmp2_native.py`): a distinct module (`pyscf.mp.dfmp2_native`), `ints3c_cholesky` + `emp2_rhf`, plus a CPHF-based relaxed-RDM path (`solve_cphf_rhf`). Sequence as a **follow-on plan** after conventional proves out; may stage behind a status marker.

### Anti-Patterns to Avoid
- **Putting pyo3 in `pyscf-mp2` or `pyscf-ao2mo`:** breaks the D-01/D-07/D-08 wall every prior phase established. The `check-dependency-wall` lint won't catch pyo3 (it only denies cubecl), so this is a review-discipline item — model `pyscf-scf`'s Cargo.toml comment "pyo3 is FORBIDDEN here."
- **Calling `pyscf_algebra::gemm` for the contraction:** it returns `NotYetImplemented{phase:2}`. Use host loops + `oracle_sum`.
- **Plain `+=` accumulation in the energy/transform sums:** violates bit-exactness (Pitfall 1/2). Every reduction goes through `oracle_sum`/`oracle_dot`.
- **Holding a live `Py<PyAny>` to `mf` inside `pyscf-mp2`:** forces a pyo3 dep. Snapshot eagerly in `pyscf-py` (D-07).
- **Porting `libmp.MP2_contract_d` / `libmp` C symbols:** no C deps allowed. Port the math.
- **Treating in-core RMP2 as numeric-shippable today:** it depends on the gated arity-4 `int2e` (see Open Question 1). Structural now, numeric CI-gated.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deterministic reductions | A custom Kahan/naive sum | `pyscf_algebra::oracle_sum`/`oracle_dot` | Fixed pairwise N=128 tree is the project's bit-exact contract (FOUND-06); a one-off sum breaks `release-oracle` parity |
| DF B-tensor assembly | A second Cholesky/forward-subst in `pyscf-mp2` | `pyscf-df::cholesky_eri` → `DfIntegrals` | Already shipped Phase-3 D-10; DF-MP2 reuses it (CONTEXT integration point) |
| Generalized eigh / S^{-1/2} | A bespoke eigensolver | `pyscf_algebra::eigh_gen` / faer host-fallback | MP2 doesn't re-diagonalize (canonical), but native-DF CPHF may; reuse the wired faer path |
| SCF reference + sign-canonical MOs | Re-running or re-canonicalizing | `pyscf-scf::ScfResult` (mo_coeff already SCF-13 canonical) | MP2 consumes the converged, vendor-stable reference (Pitfall 4/12) |
| `as_scanner` closure | New scanner machinery | `pyscf-scf` SCF-12 scanner shape | MP2 scanner = re-run mf-scanner → re-snapshot → MP2 kernel (`mp2.py:MP2_Scanner`) |
| Subclass-override dispatch | Manual method lookup | `Mp2OverrideHooks` trait + `PyOverrideBridge`-style `call_method1` | Pitfall 7 immunity by construction (Phase-3 D-01 model) |
| Frozen-core 'auto' core counts | A new element→core table | Upstream `cc.ccsd.set_frozen` / chemcore table (port the data) | Defaults must match upstream on the corpus (MP2-03); confirm the table source (Open Question 2) |
| AO ERI / 3c integrals | A new integral engine | `pyscf-gto::intor` / `intor_with_auxmol` → cintx | GTO-06; pyscf-rs never maintains a parallel integral path |

**Key insight:** Phase 5 is almost entirely *composition of already-shipped crates plus a faithful algorithm port*. The only genuinely new compute is the AO→MO contraction loop (a quarter-transform sequence) and the closed-form MP2 energy/amplitude expression — both are short, well-specified upstream, and must be expressed through `oracle_sum`. Everything else (DF B-tensor, eigh, SCF reference, scanner, bridge) already exists.

## Runtime State Inventory

> This is a code-addition phase (new crate + filled stub + bindings), **not** a rename/refactor/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a string being renamed.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified: MP2 adds new compute paths; no datastore keys/collections renamed | none |
| Live service config | None — verified: no external service config touched | none |
| OS-registered state | None — verified: no scheduled tasks / daemons | none |
| Secrets/env vars | None new. `PYSCF_MAX_MEMORY` is *read* (log-only, D-04), `PYSCF_BACKEND`/`PYSCF_DTYPE` reused read-only — no key renamed | none |
| Build artifacts | Adding `crates/pyscf-ao2mo` requires `Cargo.lock` regeneration on first build (new member). No stale egg-info/binaries from a rename. | `cargo build` regenerates lock; ensure CI lockfile updated |

**Nothing to migrate.** The one build-side action is the lockfile/workspace-member registration for the new `pyscf-ao2mo` crate.

## Common Pitfalls

### Pitfall 1: Assuming `mol.intor("int2e")` is bit-exact today (the CONTEXT trap)
**What goes wrong:** Planning in-core RMP2 as a numeric-validated, un-gated deliverable. The kernel calls `pyscf-ao2mo` → `pyscf-gto::intor("int2e")` → `NotYetImplemented{phase:2}` (`crates/pyscf-gto/src/intor.rs:181-185`). The energy oracle cannot run on real integrals until `cintx#11` lands the arity-4 safe API.
**Why it happens:** CONTEXT D-05 and STATE both say "int2e bit-exact since Phase 2," but Phase-2 plan 02-09 marks `int2e_sph` **xfail** (arity≥3 deferred to cintx safe-API). Only arity-2 (1e) integrals are green. SCF itself gets J/K via `pyscf-df`, *bypassing* `int2e` (`crates/pyscf-scf/src/fock.rs:75-96`).
**How to avoid:** Treat in-core RMP2 numeric parity exactly like DF-HF/DFT-01 — ship the full algorithm + always-on structural/wiring tests; CI-gate the bit-exact energy assertion behind `cintx#11`. Both in-core (`int2e`) and DF (`int3c2e_sph`) are the *same* gap, different arity. The code needs no change when cintx lands.
**Warning signs:** A plan task says "assert RMP2 energy == upstream" in an always-on (non-`--features python`) test arm; or a wave is sequenced as "in-core first because it's un-gated."

### Pitfall 2: Reduction-order drift in the AO→MO contraction (Pitfall 1/2 re-validation)
**What goes wrong:** Naive `for k { acc += a[k]*b[k] }` in the 4-index transform or the energy sum gives a result that differs from upstream in the last ULPs, failing `release-oracle` bit-exactness.
**Why it happens:** Floating-point addition is non-associative; upstream's reduction order (NumPy `einsum`/BLAS) must be matched by the fixed pairwise tree.
**How to avoid:** Route every contraction/energy reduction through `oracle_sum`/`oracle_dot`. Materialize the per-element products into a `Vec` first (so the tree shape depends only on length), exactly as `oracle_dot` does internally.
**Warning signs:** Energy matches to 1e-9 but not bit-exact; results differ between `RAYON_NUM_THREADS=1` and `=8`.

### Pitfall 3: F-order vs C-order layout at every boundary (Pitfall 8)
**What goes wrong:** `MOCoefficients.data` is column-major (F-order, `pyscf-core/src/mo.rs`), `int3c2e`/`DfIntegrals.b_uvq` mix F-order (intor) and row-major (b_uvq), and the transform output must match upstream's expected layout. A silent transpose corrupts the `(ia|jb)` block.
**Why it happens:** PySCF/LAPACK is F-order; the DF crate ships `b_uvq` row-major; intor returns F-order `[nao,nao,nao,nao]`.
**How to avoid:** Pin the layout of every array at each crate boundary in the plan (the DF crate's `cholesky_eri.rs` doc-comments are the model — they spell out the exact flat-index formula for each tensor). Reuse `co = mo_coeff[:, :nocc]` slicing semantics from `mp2.py:793-794` (`order='F'`).
**Warning signs:** Transform passes a 2×2 toy test but fails on a real molecule; energy off by a permutation-symmetric factor.

### Pitfall 4: GIL / NumPy boundary in the bridge (Pitfall 5/6, BIND-04/05)
**What goes wrong:** Long MP2 compute holds the GIL (blocks Python threads) or a non-standard-layout NumPy array is consumed without `to_owned()`.
**Why it happens:** The bridge must release the GIL around the Rust kernel and own non-contiguous inputs.
**How to avoid:** Mirror `PyRHF`'s per-hook `py.detach(|| ...)` pattern (`scf.rs:461-605`) and the `to_density`/`to_mo_coeff` converters (BIND-04). Note: `PyRHF::kernel` deliberately does **not** detach at the top level because hooks re-enter Python (scf.rs:345-350); MP2's kernel has the same constraint when `Mp2OverrideHooks` dispatches to Python.
**Warning signs:** Deadlock under `python3.13t` free-threaded; corrupted results from a transposed input view.

### Pitfall 5: `make_rdm2` index gymnastics + frozen-core offsets (MP2-05)
**What goes wrong:** `make_rdm2` (`mp2.py:275-348`) has intricate `dm2[oidx[i], vidx, oidx, vidx]` fancy-indexing with frozen-core `moidx`. A direct flat-array port can mis-place blocks.
**Why it happens:** The 2-RDM is a `nmo0^4` tensor with occupied/virtual/frozen sub-block placement and Chemist-notation transposes.
**How to avoid:** Port `make_rdm1`/`_gamma1_intermediates` first (simpler — `doo`/`dvv` only), unit-test against upstream small-system RDMs, then layer `make_rdm2` on top with explicit index maps. The `ao_repr`/`with_frozen` flags route through `ccsd_rdm._make_rdm1`/`_rdm2_mo2ao` upstream — port those helpers or inline the AO back-transform.
**Warning signs:** `einsum('pq,qp', h1, rdm1)` energy check fails; trace contributions off by the diagonal `-2`/`+4` correction terms.

### Pitfall 6: Workspace member count drift (D-01 bookkeeping)
**What goes wrong:** ROADMAP prose says "19 members"; CONTEXT says "19→20"; but `Cargo.toml` already lists **20** member directories (the 19 crates + `xtask`). Mis-counting leads to a wrong ROADMAP edit.
**Why it happens:** `xtask` is a member but not a `pyscf-*` crate; counting conventions differ.
**How to avoid:** Count `pyscf-*` crates specifically. Currently 19 `pyscf-*` crates + `xtask` = 20 listed members. Adding `pyscf-ao2mo` makes 20 `pyscf-*` crates + `xtask` = 21 listed members. Update ROADMAP prose to whichever convention it uses, consistently.
**Warning signs:** A plan task "update member count 19→20" without specifying the counting basis.

## Code Examples

Verified patterns from upstream source (in-repo) and existing crates:

### MP2 closed-form energy (the headline math, MP2-01)
```python
# Source: pyscf/mp/mp2.py:kernel lines 47-76 (canonical, self._scf.converged branch)
eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]      # ε_i - ε_a
for i in range(nocc):
    gi = eris.ovov[i].reshape(nvir,nocc,nvir).transpose(1,0,2)  # (a, j, b) -> (j, a, b)
    t2i = gi.conj() / direct_sum('jb+a->jba', eia, eia[i])      # t2_ijab = (ia|jb)/(εi+εj-εa-εb)
    edi =  einsum('jab,jab', t2i, gi) * 2                       # Coulomb (direct)
    exi = -einsum('jab,jba', t2i, gi)                           # exchange
    emp2_ss += edi*0.5 + exi      # same-spin
    emp2_os += edi*0.5            # opposite-spin
# Rust port: each einsum → host loop through oracle_sum; eia denominators are exact subtraction.
```

### SCS-MP2 split (MP2-06)
```python
# Source: pyscf/mp/mp2.py:energy lines 117-126 + emp2_scs property line 597-599
# Plain MP2: e_corr = e_ss + e_os
# SCS-MP2 default factors: emp2_scs = e_ss * (1/3) + e_os * 1.2   (J.Chem.Phys.118,9095)
# User factors via mp.MP2(mf).set(emp2_ss_factor=..., emp2_os_factor=...):
#   e_corr = e_ss * ss_factor + e_os * os_factor   (default ss=os=1.0 reproduces plain MP2)
```

### MP2-08 helpers — the EXACT CCSD import contract
```python
# Source: pyscf/cc/ccsd.py:35  (the call site the contract test must mimic VERBATIM)
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
# get_nocc: count_nonzero(mo_occ>0) [- frozen | minus listed occ]   (mp2.py:351-369)
# get_nmo:  len(mo_occ) [- frozen | - len(set(frozen))]             (mp2.py:371-381)
# get_frozen_mask: bool array, False at frozen indices              (mp2.py:383-400)
# get_e_hf: mp._scf.e_tot if mo_coeff is the converged ref          (mp2.py:402-410)
# _mo_without_core(mp, mo): mo[:, get_frozen_mask(mp)]              (mp2.py:732-733)
```

### Existing PyRHF bridge (the PyRMP2 template)
```rust
// Source: crates/pyscf-py/src/scf.rs:50-62 (the eager-snapshot pattern PyRMP2 mirrors)
#[pyclass(subclass, name = "RHF", module = "pyscf._native.scf")]
pub struct PyRHF { /* ... 30-attr floor + ScfResult-ish inner ... */ }
#[pymethods]
impl PyRHF {
    #[new]
    fn new(py: Python<'_>, mol: Py<PyAny>) -> PyResult<Self> {
        let mol_inner: Mole = extract_mole_from_pyany(py, &mol)?;   // PyRMP2 extracts mf instead
        // ...
    }
    // kernel() at scf.rs:320 — does NOT py.detach at top (hooks re-enter Python)
}
```

### ao2mo.general full-tensor branch (the simplest correct transform)
```python
# Source: pyscf/ao2mo/incore.py:general lines 123-128
nao = mo_coeffs[0].shape[0]
if eri_ao.size == nao**4:   # full int2e tensor — exactly what mol.intor("int2e") gives
    return einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao.reshape([nao]*4),
                  mo_coeffs[0].conj(), mo_coeffs[1], mo_coeffs[2].conj(), mo_coeffs[3])
# Rust port: quarter-transform sequence, each step oracle_sum-reduced, F-order preserved.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| C-extension AO→MO (`libao2mo`, `_ao2mo.nr_e1`) | Pure-Rust host-loop transform through `pyscf-algebra` | This project (no-C-deps core value) | No libcint/libao2mo at install time; bit-exact via `oracle_sum` |
| C driver `libmp.MP2_contract_d` for DF-MP2 | Pure-Rust `oracle_sum` contraction over the DF B-tensor | This phase | Same energy, zero C deps |
| `lazy_static!` in PyO3 paths | `pyo3::sync::GILOnceCell` (BIND-06) | Phase 3 | Already enforced by `check_forbid_lazy_static` lint |
| Holding live `mf` reference | Eager snapshot of plain arrays (D-07) | Phase 3 (SCF), reused here | Method crates stay pyo3-free |

**Deprecated/outdated:**
- The CONTEXT/STATE phrasing "in-core MP2 is un-gated because int2e is bit-exact since Phase 2" — **superseded by source verification**: arity-4 `int2e` is `NotYetImplemented` / xfail, gated on `cintx#11` (see Open Question 1).
- Outcore/semi-incore AO→MO (`ao2mo/outcore.py`, `semi_incore.py`) — deferred to Phase 6 (D-04); read for the general-surface shape only, do not port.
- `make_fno` / FNO-MP2, GMP2/DFGMP2, MP2-F12, `_iterative_kernel` — out of v1 scope (Deferred Ideas).

## Assumptions Log

> All factual claims about the existing codebase, upstream algorithm, and CCSD import contract were source-verified in-repo this session. The items below are the residual `[ASSUMED]` claims the planner/discuss-phase should confirm.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The frozen-core `'auto'` core-electron counts come from `pyscf.cc.ccsd.set_frozen` + the chemcore element table; porting that table reproduces upstream defaults on the corpus | Frozen-core (MP2-03) | MEDIUM — wrong core counts → wrong `nocc`/`nmo` → wrong frozen RMP2 energy. Confirm table source + values (Open Question 2). |
| A2 | DF-MP2 conventional path can reuse `pyscf-df::DfIntegrals.b_uvq` (in-memory) without an MP2-specific aux default mismatch; MP2 aux default is the `*-ri` (mp2fit) basis, distinct from SCF's `*-jkfit` | DF-MP2 (MP2-04) | MEDIUM — `dfmp2.py:136` uses `make_auxbasis(mol, mp2fit=True)`; if MP2 silently uses the jkfit default, DF-MP2 energy will be slightly off. Confirm `default_ri` is the mp2fit basis. |
| A3 | The closed-form (single-pass) canonical kernel is sufficient for the entire v1 corpus; `_iterative_kernel` is not needed | Alternatives / Open Question 3 | LOW — only matters for non-canonical/Brueckner refs, none expected in the corpus. |
| A4 | No new external crate is needed; the contraction + RDMs are expressible through existing `pyscf-algebra` host-loop + `oracle_sum` surface | Standard Stack | LOW — DFT already proved this for dense contractions; native-DF CPHF may need `eigh`/`solve_linear` (both wired). |
| A5 | The `check-dependency-wall` lint needs **no change** for `pyscf-ao2mo`/`pyscf-mp2` (it's a cubecl denylist; these crates simply must not name cubecl) — contradicting CONTEXT's "extend the allowlist" wording | Anti-Patterns / Pitfall 6 | LOW — verified the lint is a denylist with a 3-crate carve-out; the new crates pass automatically as long as they avoid cubecl. The CONTEXT "extend allowlist" instruction is imprecise but harmless. |

## Open Questions (RESOLVED)

1. **In-core RMP2 numeric gating (HIGH priority — corrects CONTEXT).**
   - What we know: `pyscf-gto::intor` returns `NotYetImplemented{phase:2}` for arity-4 `int2e` (`intor.rs:181-185`); Phase-2 02-09 xfail-tracks `int2e_sph` behind cintx safe-API arity>2 (`cintx#11`). SCF gets J/K via `pyscf-df`, not `int2e`.
   - What's unclear: CONTEXT D-05 frames in-core RMP2 as the "un-gated headline." That framing is wrong against the current code.
   - Recommendation: Plan in-core RMP2/UMP2 numeric oracle **CI-gated behind `cintx#11`** (same as DF-MP2, DF-HF, DFT-01). Ship the full algorithm + always-on structural tests now. Both `int2e` and `int3c2e_sph` are the same gap. Surface this to discuss-phase so the "un-gated headline" decision is re-confirmed.
   - **RESOLVED:** All 7 plans CI-gate numeric oracle assertions behind `cintx#11`. The synthetic-ERI `ao2mo` roundtrip in 05-02 is the only always-on numeric assertion (no `intor`). Plans 05-01 T3 gate `pyscf-oracle` arms; 05-03/04/05/06 all carry `#[cfg]`-gated / feature-gated numeric oracle arms. CONTEXT "un-gated" framing corrected.

2. **Frozen-core `'auto'` core table source (MP2-03).**
   - What we know: `mp2.set_frozen` delegates to `pyscf.cc.ccsd.set_frozen(method='auto', window=...)` (`mp2.py:575-579`).
   - What's unclear: Whether the chemcore element→core-count table is in `cc/ccsd.py` or a shared `pyscf/data` module, and whether pyscf-rs already has it.
   - Recommendation: Locate `set_frozen`/`chemcore` upstream during planning; port the table verbatim. Default factors must reproduce upstream on the corpus.
   - **RESOLVED:** 05-03 T1 explicitly ports the chemcore element→core-count table from `cc/ccsd.py set_frozen` into a `static CHEMCORE: OnceLock<HashMap<u32, usize>>` (modeled on `auxbasis.rs::OnceLock` shape). Planner noted `cc/ccsd.py` as the source and the action reads it in `read_first`.

3. **`_iterative_kernel` necessity (Claude's discretion).**
   - What we know: Upstream uses closed-form for `self._scf.converged`; iterative (DIIS amplitude) path only for non-canonical refs (`mp2.py:80-115, 634-637`).
   - What's unclear: Whether any v1 corpus fixture uses a non-canonical reference.
   - Recommendation: Ship closed-form only; add `_iterative_kernel` as a deferred follow-on unless a fixture forces it. Document the `NotYetImplemented` stub for the non-canonical branch.
   - **RESOLVED:** No plan implements `_iterative_kernel`. 05-03 T2 ships closed-form canonical MP2 only. Confirmed no corpus fixture requires a non-canonical reference at v1 scope.

4. **`with_t2` default + amplitude storage shape.**
   - What we know: Upstream `WITH_T2=True` by default (`mp2.py:30`); `Amplitudes` in `pyscf-core` stores `t2` as a flat `Vec<f64>` `[nocc,nocc,nvir,nvir]`.
   - What's unclear: Whether RDMs (MP2-05) and the Phase-7 gradient need `t2` retained by default for the corpus sizes.
   - Recommendation: Default `with_t2=true` (upstream); RDMs consume stored `t2`. UMP2 needs the `(t2aa, t2ab, t2bb)` triple — the `Amplitudes` struct may need a spin-resolved variant or a separate UMP2 amplitude container.
   - **RESOLVED:** 05-03 T2 implements `with_t2=true` default; `Mp2Result.t2: Option<Amplitudes>`. 05-04 T1 uses a `UmpAmplitudes { t2aa, t2ab, t2bb }` struct for the UMP2 spin-resolved triple. RDMs (05-04 T2) consume `&UmpAmplitudes`/`&Amplitudes` respectively.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (MSRV 1.92, edition 2024) | All crates | ✓ (assumed — Phases 1–4 built) | 1.92+ | — |
| `cintx` arity-4 `int2e` (safe API) | In-core RMP2/UMP2 numeric oracle | ✗ | — | **CI-gate the numeric assertion** (`cintx#11`); ship structural now |
| `cintx` arity-3 `int3c2e_sph` (safe API) | DF-MP2 numeric oracle | ✗ | — | **CI-gate** (`cintx#11`); ship structural now |
| `cintx` arity-2 `int2c2e_sph` (aux 2-center) | DF-MP2 aux metric | ✓ (arity-2 path green) | — | — |
| libpython + importable upstream `pyscf` | `--features python` oracle arms | ✗ in executor sandbox (Phase 4 precedent) | — | Oracle tests are `--features python` CI-only; structural tests always-on |
| `pytest`/`numpy` | Python-side oracle harness | ✗ in executor sandbox | — | Write tests, defer execution to CI (Phase 2/4 precedent) |

**Missing dependencies with no fallback:** none that block the phase — every gap has the established CI-gate / structural-test fallback.
**Missing dependencies with fallback:**
- `int2e` / `int3c2e_sph` numeric → CI-gated oracle behind `cintx#11`; the full Rust algorithm + always-on structural/wiring tests ship regardless.
- Live upstream PySCF / pytest → CI-only `--features python` arms; structural Rust tests run in every `cargo test`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `pyscf-oracle` `oracle_check!` macro (subprocess-per-fixture, `Python::attach`, pyo3 dev-deps only) |
| Config file | none (cargo test discovery); oracle gated by `--features python` |
| Quick run command | `cargo test -p pyscf-mp2 -p pyscf-ao2mo` |
| Full suite command | `cargo test --workspace` then `cargo test -p pyscf-oracle --features python` (CI, with libpython + upstream pyscf) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MP2-01 | RMP2 e_corr bit-exact (structural now / numeric CI-gated) | unit + oracle | `cargo test -p pyscf-mp2 rmp2_kernel` ; oracle `mp2_rmp2_energy` (`--features python`, cintx-gated) | ❌ Wave 0 |
| MP2-02 | UMP2 e_corr (open-shell) | unit + oracle | `cargo test -p pyscf-mp2 ump2_kernel` ; oracle `mp2_ump2_energy` (cintx-gated) | ❌ Wave 0 |
| MP2-03 | frozen=int/list/'auto'/window; defaults match | unit | `cargo test -p pyscf-mp2 frozen_core` | ❌ Wave 0 |
| MP2-04 | DF-MP2 (conventional + native) e_corr | unit + oracle | `cargo test -p pyscf-mp2 dfmp2` ; oracle `dfmp2_energy` (cintx-gated) | ❌ Wave 0 |
| MP2-05 | make_rdm1/make_rdm2 match upstream | unit + oracle | `cargo test -p pyscf-mp2 make_rdm` ; oracle `mp2_rdm` (cintx-gated) | ❌ Wave 0 |
| MP2-06 | SCS-MP2 factors; default reproduces plain MP2 | unit | `cargo test -p pyscf-mp2 scs_factors` | ❌ Wave 0 |
| MP2-07 | as_scanner returns Mole→energy callable | unit + smoke | `cargo test -p pyscf-py mp2_scanner` (structural) | ❌ Wave 0 |
| MP2-08 | helpers exported; CCSD import call-site works verbatim | contract | `cargo test -p pyscf-mp2 ccsd_import_contract` | ❌ Wave 0 |
| (ao2mo) | general/full transform round-trips a toy ERI | unit | `cargo test -p pyscf-ao2mo transform` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p pyscf-mp2 -p pyscf-ao2mo` (+ `cargo clippy -D warnings`, `cargo fmt --check`, `xtask check-dependency-wall`, `xtask check-no-fma` on touched crates).
- **Per wave merge:** `cargo test --workspace` + `xtask check-forbidden-paths` + `xtask check-cubecl-pin`.
- **Phase gate:** Full workspace green + `cargo test -p pyscf-oracle --features python` on CI (cintx-gated MP2 numeric arms green once `cintx#11` lands; structural arms green always) before `/gsd:verify-work`.

### Wave 0 Gaps
- [ ] `crates/pyscf-ao2mo/Cargo.toml` + `src/lib.rs` skeleton — new crate scaffold (member registration + algebra/gto deps, no pyo3/cubecl)
- [ ] `crates/pyscf-mp2/tests/` — directory does not exist yet (currently no tests)
- [ ] `crates/pyscf-mp2/tests/ccsd_import_contract.rs` — MP2-08 verbatim-import contract test (mirror `cc/ccsd.py:35`)
- [ ] `crates/pyscf-mp2/tests/rmp2_structural.rs` / `ump2_structural.rs` — always-on shape/wiring + error-propagation (cintx-gated numeric separate)
- [ ] `crates/pyscf-ao2mo/tests/transform_roundtrip.rs` — toy-ERI transform correctness (no cintx dependency — pass in a synthetic `nao^4` ERI)
- [ ] `pyscf-oracle` new arms: `mp2_rmp2_energy`, `mp2_ump2_energy`, `dfmp2_energy`, `mp2_rdm` (extend `KNOWN_METHODS` + `--features python` driver), all cintx-gated
- [ ] `.github/workflows/ci.yml` — MP2 structural always-on job + MP2 numeric oracle job gated behind `cintx#11` (mirror the DF-HF / DFT-01 gating)

*The synthetic-ERI transform test is the one numeric assertion that can be **always-on** this phase: feed `pyscf-ao2mo::general` a hand-built `nao^4` AO ERI (not from `intor`), assert the MO transform matches a longhand reference. This proves the contraction math without waiting on cintx.*

## Security Domain

`security_enforcement` is not set to `false` in `.planning/config.json`, so this section is included. Phase 5 is a numerical/compute phase with a Python FFI boundary; the threat surface is FFI safety and untrusted-input robustness, not network/auth/crypto.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (in-process library) |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access-control surface |
| V5 Input Validation | yes | NumPy boundary: non-`is_standard_layout()` arrays `to_owned()` on entry (BIND-04); shape/dtype validated before compute; `Mole`/`mf` extraction validated (`extract_mole_from_pyany`) |
| V6 Cryptography | no | No crypto |
| (FFI safety — project FOUND-07) | yes | `#![forbid(unsafe_code)]` in `pyscf-mp2`/`pyscf-ao2mo`; panics never cross FFI (catch_unwind in `pyscf-py`); `panic="abort"` release; clippy `unwrap_used` lint in numeric modules |

### Known Threat Patterns for {Rust compute crate + PyO3 boundary}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Rust panic escaping FFI → process abort visible to Python | Denial of Service | `catch_unwind` at the pyo3 boundary (BIND-09); `?`-propagated `Result` everywhere (no `unwrap` in numeric paths — clippy-enforced) |
| Non-contiguous / aliased NumPy input corrupting the transform | Tampering | BIND-04: `to_owned()` non-standard-layout inputs; validate `nao`/`nmo` shapes against the snapshotted reference before contraction |
| GIL deadlock under free-threaded Python | Denial of Service | `py.detach` seam (BIND-05) around pure-Rust compute; tested under `python3.13t` |
| Numeric overflow in f32 precision seam (if reused) | Tampering | Phase-4 CR-02 precedent: `cast_finite`/`back_to_f64` helpers reject finite→non-finite narrowing (don't silently substitute 0.0) |
| Unbounded memory allocation from a huge `nao^4` ERI | Denial of Service | D-04: `PYSCF_MAX_MEMORY` log-only this phase (Phase-6 CCSD-11 owns enforcement); document the budget log at kernel entry |

## Sources

### Primary (HIGH confidence — source-verified in-repo this session)
- `pyscf/mp/mp2.py` — `kernel`/`energy`/`make_rdm1`/`make_rdm2`/`_gamma1_intermediates`/`get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`_mo_without_core`/`_mo_splitter`/`as_scanner`/`MP2Base`/`_ChemistsERIs`/`_make_eris` — the RMP2 + MP2-08 source-of-truth
- `pyscf/mp/__init__.py` — `MP2`/`RMP2`/`UMP2` factory + `mf.with_df → DFMP2` dispatch (the cross-module contract, MP2-01/04)
- `pyscf/mp/ump2.py` — `kernel`/`energy`/`get_nocc`/`get_frozen_mask`/`_make_eris` spin-block (MP2-02)
- `pyscf/mp/dfmp2.py` — `DFRMP2(mp2.RMP2)`, `kernel` (libmp.MP2_contract_d), `_make_df_eris`, `scf.hf.RHF.DFMP2 = class_as_method` (MP2-04 conventional)
- `pyscf/mp/dfmp2_native.py` — `DFRMP2(StreamObject)`, `ints3c_cholesky`, `emp2_rhf`, `solve_cphf_rhf` (MP2-04 native)
- `pyscf/ao2mo/incore.py` — `full`/`general` (D-02 port target; the `nao**4 einsum` branch)
- `pyscf/cc/ccsd.py:35` — `from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core` (MP2-08 contract-test call site, VERBATIM)
- `crates/pyscf-gto/src/intor.rs:181-185` — **arity 3/4 `NotYetImplemented{phase:2}`** (the gating finding; Open Question 1)
- `.planning/phases/02-gto/02-09-SUMMARY.md` — `int2e_sph`/`int3c2e_sph` xfail behind cintx safe-API arity>2 (`cintx#11`)
- `crates/pyscf-scf/src/{kernel.rs,hooks.rs,fock.rs,scanner.rs}` — `ScfResult`, `OverrideHooks`, `default_get_jk` bypasses int2e via DF, `as_scanner` (D-07/D-08 model)
- `crates/pyscf-py/src/scf.rs` — `PyRHF`/`PyUHF` eager-snapshot + per-hook `py.detach` (PyRMP2 template)
- `crates/pyscf-df/src/{lib.rs,cholesky_eri.rs}` — `DfIntegrals`/`cholesky_eri`/`DEFAULT_AUXBASIS` (DF-MP2 reuse)
- `crates/pyscf-algebra/src/{lib.rs,oracle.rs,gemm.rs,eigh_gen.rs,host_fallback.rs}` — `oracle_sum`/`oracle_dot`, **`gemm` NotYetImplemented{phase:2}**, `eigh_gen`
- `crates/pyscf-dft/src/numint.rs` — host-loop-through-`oracle_sum` contraction precedent (D-03 model)
- `crates/pyscf-core/src/{mo.rs,traits.rs,amplitudes.rs}` — `MOCoefficients` (F-order), `PostScf` trait, `Amplitudes`
- `xtask/src/bin/check_dependency_wall.rs` — cubecl denylist + 3-crate carve-out (Assumption A5)
- `Cargo.toml` (workspace) — member list (20 dirs incl. xtask), pinned external deps
- `.planning/phases/05-mp2/05-CONTEXT.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md` — phase decisions + REQ definitions + accumulated state

### Secondary (MEDIUM confidence)
- `pyscf/mp/mp2.py` SCS factor reference `J. Chem. Phys. 118, 9095 (2003)` (cited in upstream comments) — standard SCS-MP2 1/3, 1.2 factors

### Tertiary (LOW confidence)
- None — every claim in this research traces to in-repo source. No WebSearch was needed; the algorithm and contracts are fully specified by the vendored upstream and the existing crates.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all crates/deps verified present in the workspace.
- Architecture: HIGH — every pattern (pyo3-free crate, eager snapshot, hooks bridge, host-loop contraction, DF reuse) is shipped precedent verified in Phases 3–4.
- Pitfalls: HIGH — Pitfall 1 (the `int2e` gating) is source-verified against `intor.rs` + 02-09-SUMMARY; the rest are re-validations of documented Phase 3/4 pitfalls.
- Frozen-core 'auto' table (A1) + DF aux default (A2): MEDIUM — confirm exact upstream source during planning.

**Research date:** 2026-05-23
**Valid until:** ~2026-06-22 (30 days — stable; the only moving piece is the external `cintx#11` merge, which *unblocks* numeric oracles without changing the planned MP2 code)
