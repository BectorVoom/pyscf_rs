# Phase 6: CCSD - Pattern Map

**Mapped:** 2026-05-24
**Files analyzed:** 28 new/modified (19 new in `pyscf-ccsd/src` + 9 cross-crate modifications + test/CI/xtask)
**Analogs found:** 26 / 28 (2 partial — `WorkspacePool` arena body + AO-direct ERI streaming have no full in-tree analog yet)

> **Single strongest analog source: Phase 5 MP2.** CCSD mirrors `pyscf-mp2` file-for-file. The entire `pyscf-ccsd` crate copies the module-split / error / hooks / reference / kernel discipline of `pyscf-mp2`, and `pyscf-py::cc` copies `pyscf-py::mp` verbatim (snapshot + bridge + factory + scanner). **Two findings the planner must honor (from RESEARCH, verified):** (1) `pyscf_algebra::gemm` is a `NotYetImplemented{phase:2}` STUB — all contractions are host loops + `oracle_sum`/`oracle_dot`, never `gemm`; (2) the dependency wall is a DENYLIST — `pyscf-ccsd` needs NO allowlist entry, just no `cubecl-*`/`hdf5-metno` dep.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/pyscf-ccsd/src/lib.rs` | crate-root / re-exports | n/a | `crates/pyscf-mp2/src/lib.rs` | exact |
| `crates/pyscf-ccsd/src/error.rs` | error | n/a | `crates/pyscf-mp2/src/error.rs` | exact |
| `crates/pyscf-ccsd/src/eris.rs` | model (ERI container) | transform | `crates/pyscf-mp2/src/hooks.rs` `ChemistsEris` | role-match |
| `crates/pyscf-ccsd/src/hooks.rs` | trait/seam | request-response | `crates/pyscf-mp2/src/hooks.rs` | exact |
| `crates/pyscf-ccsd/src/reference.rs` | model (snapshot) | n/a | `crates/pyscf-mp2/src/mp2.rs` `Mp2Reference` | exact |
| `crates/pyscf-ccsd/src/ccsd.rs` | service (kernel) | event-driven (iterate) | `crates/pyscf-mp2/src/mp2.rs` `rmp2_kernel` | role-match |
| `crates/pyscf-ccsd/src/rintermediates.rs` | transform | batch | `crates/pyscf-ao2mo/src/transform.rs` | partial (data-flow) |
| `crates/pyscf-ccsd/src/update_amps.rs` | transform | batch | `crates/pyscf-mp2/src/mp2.rs` (host-loop reductions) | role-match |
| `crates/pyscf-ccsd/src/uccsd.rs` | service (kernel) | event-driven | `crates/pyscf-mp2/src/ump2.rs` | role-match |
| `crates/pyscf-ccsd/src/uintermediates.rs` | transform | batch | `crates/pyscf-mp2/src/ump2.rs` (spin channels) | partial |
| `crates/pyscf-ccsd/src/diis_amps.rs` | utility (DIIS storable) | transform | `crates/pyscf-diis/src/cdiis.rs` test `V` + `storable.rs` | role-match |
| `crates/pyscf-ccsd/src/lambda.rs` | transform | batch | `crates/pyscf-ccsd/src/update_amps.rs` (sibling, same phase) | role-match |
| `crates/pyscf-ccsd/src/ulambda.rs` | transform | batch | `crates/pyscf-ccsd/src/uccsd.rs` (sibling) | role-match |
| `crates/pyscf-ccsd/src/rdm.rs` | transform (RDM) | batch | `crates/pyscf-mp2/src/rdm.rs` | role-match |
| `crates/pyscf-ccsd/src/urdm.rs` | transform (RDM) | batch | `crates/pyscf-mp2/src/rdm.rs` (spin) | partial |
| `crates/pyscf-ccsd/src/diagnostics.rs` | utility | transform | `crates/pyscf-mp2/src/mp2.rs` (Frobenius/oracle reductions) | partial |
| `crates/pyscf-ccsd/src/dfccsd.rs` | service (DF subclass) | streaming + file-I/O | `crates/pyscf-mp2/src/dfmp2.rs` `DFRMP2` | role-match |
| `crates/pyscf-ccsd/src/direct.rs` | transform (AO-direct) | streaming | `crates/pyscf-ao2mo/src/transform.rs` | partial (no streaming analog) |
| `crates/pyscf-runtime/src/workspace_pool.rs` (MODIFY) | runtime (arena) | n/a | (self skeleton — no full analog) | partial |
| `crates/pyscf-core/src/amplitudes.rs` (MODIFY) | model | n/a | (self skeleton) + `pyscf-mp2` `UmpAmplitudes` shape | role-match |
| `crates/pyscf-diis/src/` (ADD `AmplitudeSubspace`) | utility | transform | `crates/pyscf-diis/src/cdiis.rs` `FockSubspace`/test `V` | exact |
| `crates/pyscf-ao2mo/src/` (ADD outcore surface) | service | file-I/O | `crates/pyscf-ao2mo/src/transform.rs` + `pyscf-chkfile` | role-match |
| `crates/pyscf-py/src/cc.rs` (NEW) | controller (PyO3) | request-response | `crates/pyscf-py/src/mp.rs` | exact |
| `python/pyscf/cc/__init__.py` (overlay) | route (factory) | request-response | `pyscf-py::mp` `mp2_factory` + `pyscf/cc/__init__.py:83-139` | exact |
| `crates/pyscf-ccsd/Cargo.toml` (MODIFY) | config | n/a | `crates/pyscf-mp2/Cargo.toml` | exact |
| `crates/pyscf-oracle/tests/*` (ADD) | test | n/a | existing `oracle_check!` fixtures (MP2 arms) | role-match |
| `xtask/src/bin/check_no_fma.rs` (MODIFY) | config (lint target) | n/a | `SCAN_TARGETS` (self) | exact |
| `.github/workflows/ci.yml` (MODIFY) | config (CI) | n/a | `mp2-structural` + `mp2-oracle-upstream-manual` + `python3.13t SCF smoke` | exact |

---

## Pattern Assignments

### `crates/pyscf-ccsd/src/lib.rs` (crate-root, exact)

**Analog:** `crates/pyscf-mp2/src/lib.rs` (lines 15-42)

Copy the shape: `#![forbid(unsafe_code)]` + `#![warn(clippy::unwrap_used)]`, then `pub mod` declarations for each module file, then flat `pub use` re-exports. The existing `pyscf-ccsd/src/lib.rs` is a 5-line stub (`#![forbid(unsafe_code)]` only) — fill it. Module list comes from RESEARCH §"Recommended module structure" (06-RESEARCH.md:195-214): `error`, `eris`, `hooks`, `reference`, `ccsd`, `rintermediates`, `update_amps`, `uccsd`, `uintermediates`, `diis_amps`, `lambda`, `ulambda`, `rdm`, `urdm`, `diagnostics`, `dfccsd`, `direct`.

```rust
// from pyscf-mp2/src/lib.rs:15-41 — copy this exact shape
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub mod hooks;
pub mod ccsd;
// ... one per module file

pub use error::CcsdError;
pub use hooks::{ChemistsEris, CcsdOverrideHooks, NoCcsdOverrides};
pub use ccsd::{CcsdReference, CcsdResult, ccsd_kernel, /* defaults, energy, init_amps */};
```

---

### `crates/pyscf-ccsd/src/error.rs` (error, exact)

**Analog:** `crates/pyscf-mp2/src/error.rs` (entire file, lines 1-35)

Copy verbatim, rename `Mp2Error`→`CcsdError`. Keep the `#[from]` bridges for `AlgebraError`, `CoreError`, `Ao2moError` — and ADD a `#[from] pyscf_diis::DiisError` arm (CCSD uses DIIS) and a `#[from] BackendError` arm (the `try_reserve` pre-flight, D-01). Keep the `ShapeMismatch { expected, got }` variant (load-bearing for the V5 shape-validation pattern at every hook boundary) and a `NotYetImplemented { wave: u8 }` variant for staged scaffolds. The `From<CcsdError> for PyscfRsError` routes through `Core(InvalidMolecule(format!("{e}")))` — copy exactly.

```rust
// pyscf-mp2/src/error.rs:28-35 — copy this bridge verbatim, rename Mp2→Ccsd
impl From<Mp2Error> for pyscf_core::PyscfRsError {
    fn from(e: Mp2Error) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
```

**Note (memory pre-flight):** the D-01 hard-refusal surfaces `pyscf_runtime::BackendError::MemoryLimitExceeded { requested, limit }` (already defined, `crates/pyscf-runtime/src/error.rs:14-15`). Add a `CcsdError::Backend(#[from] pyscf_runtime::BackendError)` arm so `try_reserve(...)?` propagates cleanly.

---

### `crates/pyscf-ccsd/src/hooks.rs` (trait/seam, exact)

**Analog:** `crates/pyscf-mp2/src/hooks.rs` (entire file, lines 1-89)

This is the **closest one-to-one analog in the phase.** Copy the structure: a `ChemistsEris` struct + an `Mp2OverrideHooks`-shaped trait + a `NoMp2Overrides` zero-sized default impl. For CCSD:
- `CcsdOverrideHooks` trait with the D-09 hook set: `ao2mo`, `update_amps`, `make_rdm1`, `make_rdm2`, `energy` (vs MP2's `ao2mo`/`energy`/`make_rdm1`/`make_rdm2`).
- `NoCcsdOverrides` default impl delegates each to a `crate::ccsd::default_*` / `crate::update_amps::default_update_amps` free fn (the exact MP2 delegate pattern, lines 81-89).
- Trait-default stubs return `Err(CcsdError::NotYetImplemented { wave }.into())` for not-yet-landed waves (MP2 pattern, lines 66-78).

```rust
// pyscf-mp2/src/hooks.rs:35-54 — trait + default-delegate shape
pub trait CcsdOverrideHooks {
    fn ao2mo(&self, refr: &CcsdReference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError>;
    fn update_amps(&self, t1: &Tensor, t2: &Tensor, eris: &ChemistsEris)
        -> Result<(Tensor, Tensor), PyscfRsError>;     // the amplitude-equation core
    fn energy(&self, t1: &Tensor, t2: &Tensor, eris: &ChemistsEris)
        -> Result<Energy, PyscfRsError> {
        Ok(Energy(crate::ccsd::default_energy(t1, t2, eris)?))   // default-delegate
    }
    fn make_rdm1(...) -> Result<Density, PyscfRsError> { /* default */ }
    fn make_rdm2(...) -> Result<Density, PyscfRsError> { /* default */ }
}
```

**Note:** `eris.rs` (the `ChemistsEris` for CCSD) carries MORE blocks than MP2's single `ovov`: `oooo`/`ovoo`/`oovv`/`ovov`/`ovvo`/`ovvv` (+ `vvvv` or its DF/AO-direct stand-in) + `fock`/`mo_energy` (upstream `ccsd._ChemistsERIs`, `ccsd.py:1389`). Use the flat C-order-with-documented-offset discipline of `pyscf-mp2/src/hooks.rs:22-30` for every block.

---

### `crates/pyscf-ccsd/src/reference.rs` (model/snapshot, exact)

**Analog:** `crates/pyscf-mp2/src/mp2.rs` `Mp2Reference` (lines 29-43)

Copy `Mp2Reference` verbatim → `CcsdReference`: `mo_coeff: MOCoefficients` (F-order, sign-canonicalized), `mo_energy: Vec<f64>`, `mo_occ: Vec<f64>`, `e_hf: f64`, `converged: bool`, `mol: Mole`. pyo3-free; populated by the `pyscf-py` snapshot (D-09). For UCCSD reuse the `UmpReference { alpha, beta }` two-channel shape (`crates/pyscf-mp2/src/ump2.rs:79-80` — two `Mp2Reference`s sharing one `mol`).

```rust
// pyscf-mp2/src/mp2.rs:29-43 — copy field-for-field, rename Mp2→Ccsd
pub struct Mp2Reference {
    pub mo_coeff: MOCoefficients,   // F-order, sign-canonicalized (SCF-13)
    pub mo_energy: Vec<f64>,
    pub mo_occ: Vec<f64>,
    pub e_hf: f64,
    pub converged: bool,
    pub mol: Mole,
}
```

---

### `crates/pyscf-ccsd/src/ccsd.rs` (kernel loop, role-match)

**Analog:** `crates/pyscf-mp2/src/mp2.rs` `rmp2_kernel` (lines 176-313) for the Rust kernel shape + the shape-validation `?`-propagation; **upstream port target** `pyscf/cc/ccsd.py:44-101` (`kernel`) for the algorithm.

**Kernel signature pattern** (copy the `rmp2_kernel` generic-over-hooks shape, mp2.rs:176-181):
```rust
pub fn ccsd_kernel<H: CcsdOverrideHooks>(
    refr: &CcsdReference, frozen: &Frozen, hooks: &H,
    pool: &WorkspacePool,            // NEW for CCSD — the arena (D-01/D-08)
) -> Result<CcsdResult, PyscfRsError> { ... }
```

**Shape-validation discipline** (copy mp2.rs:191-205 — validate every hook-returned block length BEFORE indexing, never panic, V5 mitigation):
```rust
let expected_len = nocc * nvir * nocc * nvir;        // adapt per block
if eris.ovov.len() != expected_len {
    return Err(crate::error::CcsdError::ShapeMismatch { expected: expected_len, got: eris.ovov.len() }.into());
}
```

**Upstream kernel loop to port** (`ccsd.py:44-101`, the iterate; defaults VERIFIED below):
```python
# pyscf/cc/ccsd.py:44 — port this loop into ccsd_kernel
def kernel(mycc, eris, t1=None, t2=None, max_cycle=50, tol=1e-7, tolnormt=1e-5, ...):
    if t1 is None and t2 is None: t1, t2 = mycc.get_init_guess(eris)   # MP2 seed
    eccsd = mycc.energy(t1, t2, eris)
    adiis = lib.diis.DIIS(mycc); adiis.space = mycc.diis_space          # diis_space=6
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        normt = norm(amplitudes_to_vector(t1new,t2new) - amplitudes_to_vector(t1,t2))
        t1, t2 = mycc.run_diis(t1new, t2new, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        if abs(eccsd-eold) < tol and normt < tolnormt: converged = True; break
```

**Convergence defaults — VERIFIED from `pyscf/cc/ccsd.py` `CCSDBase` class attributes (these win at runtime over the kernel-signature defaults):**
| Constant | Value | Source line |
|----------|-------|-------------|
| `max_cycle` | `50` | `ccsd.py:920` |
| `conv_tol` (energy) | `1e-7` | `ccsd.py:921` |
| `conv_tol_normt` (amplitude norm) | **`1e-5`** (NOT the `1e-6` in CONTEXT Discretion) | `ccsd.py:923` |
| `diis_space` | **`6`** (NOT SCF's 8) | `ccsd.py:926` |
| `diis_start_cycle` | `0` | `ccsd.py:928` |

> **Flag for the planner (RESEARCH Open Q1 / A1):** CONTEXT.md Discretion says `conv_tol_normt=1e-6`; the upstream class attribute is `1e-5`. Use `1e-5` to match upstream; confirm against the H2O/cc-pVDZ fixture.

**`init_amps` MP2 seed** (port `ccsd.py:1048-1077`; reuse the Phase-5 in-core path): `t1=0`, `t2=(ia|jb)/Dijab`, report `emp2`. The `t2=(ia|jb)/Dijab` form is already in `crates/pyscf-mp2/src/mp2.rs:265-273` (the `t2i[p] = gi[p] / denom` block) — copy it.

**Pre-flight refusal (CCSD-11/D-01):** before allocating `vvvv`/`Wabef`, call `pool.try_reserve(estimate_bytes)?` (the existing `crates/pyscf-runtime/src/workspace_pool.rs:53-62` contract). On `Err(MemoryLimitExceeded)` → propagate (no silent downgrade; tell user to opt into DF/AO-direct).

---

### `crates/pyscf-ccsd/src/update_amps.rs` + `rintermediates.rs` (transform, role-match)

**Reduction-discipline analog:** `crates/pyscf-mp2/src/mp2.rs:253-298` (the per-i materialize-then-`oracle_dot`/`oracle_sum` loop) and `crates/pyscf-ao2mo/src/transform.rs:9-14` (the explicit doc-note that `gemm` is a stub so contractions are host loops). **Upstream port:** `ccsd.py:104` (`update_amps`), `ccsd.py:362-490` (`_add_vvvv`/`_contract_vvvv_t2`), `rintermediates.py:30-326` (`cc_Foo`/`cc_Fvv`/`cc_Fov`/`cc_Woooo`/`cc_Wvvvv`/`cc_Wvoov`/`cc_Wovvo`/`make_tau`).

**CRITICAL — the gemm-stub finding (RESEARCH §"Don't Hand-Roll", verified `gemm.rs:16`):** EVERY einsum becomes an explicit loop that materializes products into a `Vec` then calls `oracle_sum`/`oracle_dot`. NO `gemm`, NO bare `+=`. Copy this exact template:

```rust
// crates/pyscf-mp2/src/mp2.rs:275-296 — the canonical reduction discipline
let edi = 2.0 * oracle_dot(&gi, &t2i);   // materialize gi*t2i then reduce, NO +=
let exi = -oracle_dot(&g_jba, &t2i);
e_ss_terms.push(edi * 0.5 + exi);        // collect per-index terms
// ... then ONCE at the end:
let e_ss = oracle_sum(&e_ss_terms);      // single final reduction
```

The heaviest contraction is the `vvvv` `'ijcd,acdb->ijab'` (the `Wabef ≈ nv⁴` arena tenant). It follows the SAME host-loop discipline. The flat-index offset must be doc-commented at every boundary (copy the discipline of `transform.rs:21-33`). Reductions imported from `pyscf_algebra::{oracle_dot, oracle_sum}` (mp2.rs:22).

**Arena tenancy (CCSD-11):** the `vvvv`/`Wabef` buffer is `pool.reserve`d ONCE before the iteration loop and `release`d after — NOT allocated inside `update_amps` per cycle (Pitfall 20).

---

### `crates/pyscf-ccsd/src/uccsd.rs` + `uintermediates.rs` (kernel/transform, role-match/partial)

**Analog:** `crates/pyscf-mp2/src/ump2.rs` (lines 1-69 read) — the α/β/αβ spin-channel decomposition + the `UmpAmplitudes { t2aa, t2ab, t2bb }` spin-resolved container (ump2.rs:52-69). **Upstream port:** `uccsd.py` + `uintermediates.py`.

Copy the spin-channel structure: a `UccsdAmplitudes` triple mirroring `UmpAmplitudes` (the documented flat-index layout, ump2.rs:44-51), each spin channel transforming its OWN α/β orbital energies, and `e_corr = e_aa + e_bb + e_ab` (ump2.rs:10). Same materialize-then-`oracle_*` reduction discipline per channel (ump2.rs:20-22). The `UmpReference { alpha, beta }` two-channel snapshot (ump2.rs:79-80) is the `CcsdReference`-pair shape.

---

### `crates/pyscf-ccsd/src/diis_amps.rs` (DIIS storable, role-match)

**Analog:** `crates/pyscf-diis/src/storable.rs` (the `DiisStorable` trait, lines 12-30) + the test `V` impl in `crates/pyscf-diis/src/cdiis.rs:199-214` (the minimal flat-vector storable). **Upstream packing:** `ccsd.py:670` (`amplitudes_to_vector`) / `:679` (`vector_to_amplitudes`).

The `storable.rs` doc-comment ALREADY names this work (lines 5-6): *"Phase 6 CCSD the iterate is an `(T1, T2)` amplitude tuple (`pyscf-ccsd::AmpsSubspace` will impl this later)."* Implement `AmplitudeSubspace: DiisStorable`:
- `as_flat`/`from_flat` reproduce the upstream packing: `vector[:nov] = t1.ravel()` then `pack_tril(t2.transpose(0,2,1,3).reshape(nov,nov))` where `nov = nocc*nvir` (the symmetric `t2[iajb]==t2[jbia]` lower-triangular pack).
- `dot` MUST route through `pyscf_algebra::oracle_dot` (Pitfall 9 — the trait doc-comment mandates it, storable.rs:21-23; the test `V` shows it: `cdiis.rs:208-210`).

```rust
// crates/pyscf-diis/src/cdiis.rs:201-214 — copy the storable impl shape
impl DiisStorable for AmplitudeSubspace {
    fn as_flat(&self) -> &[f64] { &self.packed }
    fn from_flat(&mut self, s: &[f64]) { self.packed.copy_from_slice(s); }
    fn dot(&self, o: &Self) -> f64 { pyscf_algebra::oracle_dot(&self.packed, &o.packed) }  // Pitfall 9
    fn len(&self) -> usize { self.packed.len() }
}
```

**Construction:** `Diis::<AmplitudeSubspace>::new(6)` — the generic `Diis<S>` machinery (`cdiis.rs:33-142`) is fully reusable, NO new DIIS body. Use `diis_space=6` (NOT `Diis::new`'s SCF-default 8). Wave: after in-core RCCSD kernel (DIIS needs the kernel loop, not UCCSD).

---

### `crates/pyscf-ccsd/src/lambda.rs` / `ulambda.rs` / `rdm.rs` / `urdm.rs` / `diagnostics.rs` (transforms, role-match/partial)

**Analogs:** intra-phase siblings (`update_amps.rs`/`uccsd.rs` for the host-loop discipline) + `crates/pyscf-mp2/src/rdm.rs` for the RDM build-from-amplitudes shape. **Upstream ports:** `ccsd_lambda.py`/`uccsd_lambda.py` (λ), `ccsd_rdm.py`/`uccsd_rdm.py` (RDM incl. `ao_repr=True` nmo⁴ back-transform via `pyscf-ao2mo`), `ccsd.py:748-776` (diagnostics).

- **lambda.rs:** port the concrete `CCSD.solve_lambda` (`ccsd.py:1273` dispatches to `ccsd_lambda.kernel`; the base `CCSDBase.solve_lambda` at `:1118` raises NotImplementedError — port the concrete-class behavior, RESEARCH A6). Same materialize-then-`oracle_*` discipline.
- **rdm.rs:** `make_rdm1`/`make_rdm2` + `_gamma1`/`_gamma2_intermediates`; the `ao_repr=True` AO back-transform routes the nmo⁴ contraction through `pyscf-ao2mo` (D-03 ships ao_repr THIS phase — heaviest arena tenant).
- **diagnostics.rs:** `get_t1_diagnostic` (Frobenius norm of t1), `get_d1_diagnostic`/`get_d2_diagnostic` (need `pyscf_algebra::eigh` of t1·t1ᵀ — `ccsd.py:748-776`). Reductions via `oracle_*`.

---

### `crates/pyscf-ccsd/src/dfccsd.rs` (DF subclass, role-match) — CCSD-08

**Analog:** `crates/pyscf-mp2/src/dfmp2.rs` `DFRMP2` (the Phase-5 `DFRMP2(RMP2)` subclass-swaps-ERI-source pattern — the exact D-05 template) + `crates/pyscf-py/src/mp.rs:111-114` `build_df_integrals` (the `default_ri` + `cholesky_eri` B-tensor assembly). **Upstream port:** `pyscf/cc/dfccsd.py`.

DF-CCSD reuses the in-core `ccsd_kernel` and swaps `ao2mo`/`_add_vvvv` to the DF B-tensor (`vvL`) source. The DF B-tensor comes from `pyscf_df::{cholesky_eri, default_ri}` (un-gated since 05-09).

**Block sizing — port `dfccsd.py:93-96` (verified):**
```python
dmax  = numpy.sqrt(max_memory*.7e6/8/nvirb**2/2)
dmax  = int(min((nvira+3)//4, max(ccsd.BLKMIN, dmax)))
vvblk = (max_memory*1e6/8 - dmax**2*(nvirb**2*1.5+naux))/naux
vvblk = int(min((nvira+3)//4, max(ccsd.BLKMIN, vvblk/naux)))
```
`dmax`/`vvblk` become `WorkspacePool` reservation sizes (D-08).

**HDF5 spill — port `dfccsd.py:139,147` (verified):**
```python
eris.feri = lib.H5TmpFile()                               # the spill temp file (D-07 H5TmpFile-equiv)
eris.vvL  = eris.feri.create_dataset('vvL', (nvir_pair, naux), 'f8', chunks=chunks)
```
The Rust port uses `pyscf_chkfile::hdf5::Group::new_dataset` (the re-exported alias, `crates/pyscf-chkfile/src/lib.rs:37` — NO new `hdf5-metno` dep, D-07). The spill `SpillHandle` MUST delete its file on `Drop` (RAII, mirror `H5TmpFile` auto-delete — RESEARCH Runtime State Inventory).

---

### `crates/pyscf-ccsd/src/direct.rs` (AO-direct, partial) — CCSD-07

**Analog:** `crates/pyscf-ao2mo/src/transform.rs` (the host-loop AO→MO contraction is the closest in-tree shape). **Upstream port:** `ccsd.py:473-570` (`_contract_vvvv_t2` AO-direct branch). **No full analog** — there is no shell-sliced streaming `int2e` primitive in-tree yet (RESEARCH Open Q4). On-the-fly AO contraction replaces the in-memory `vvvv`. Flag: if `pyscf-gto` exposes only the full `int2e` tensor, AO-direct v1 may compute the full tensor once and tile in MO space — confirm against the CCSD-07 `direct=True` contract during this wave.

---

### `crates/pyscf-runtime/src/workspace_pool.rs` (MODIFY — arena body, partial) — CCSD-11, the defining work

**Analog:** the file's OWN Phase-1 skeleton (lines 1-63) — no full external analog; this is the genuinely new engineering. The skeleton already ships `budget_bytes`, `pool: Mutex<Vec<PooledAllocation>>`, `from_env` (reads `PYSCF_MAX_MEMORY` as MB, lines 42-49), `try_reserve` (budget-check only, lines 53-62), and `PooledAllocation { _bytes, _size }` (lines 22-27). The struct/method SURFACE is reserved exactly for this — do NOT restructure the public surface (the doc-comment at lines 1-7 mandates this).

Fill the body (RESEARCH Pattern 1, 06-RESEARCH.md:216-242): a `TensorBackend` enum (`InMemory(Box<[f64]>)` | `Spilled(SpillHandle)`), `reserve(shape, allow_spill) -> BufferId`, `release(id)` (returns buffer to free-list, does NOT free — satisfies the allocate-once-reuse / heap-alloc-count assertion). The Spilled backend uses `pyscf_chkfile::hdf5` (D-07/D-08). Keep reusable for Phase-7 gradients + Phase-8 GPU buffers (no CCSD-only assumptions, D-08).

**Handle-unification decision (RESEARCH Open Q2 / A2):** `pyscf_algebra::Tensor<T>` already carries a `BufferId` (`crates/pyscf-algebra/src/tensor.rs`). Decide whether the arena allocates that `Tensor` directly (preferred — single handle type) or a runtime sibling. Riskiest Wave-1 API decision; resolve before any CCSD math.

`pyscf-runtime` is on the dependency-wall carve-out (`ALLOWED_CRATES`, `check_dependency_wall.rs:47`) so it MAY touch backend/buffer concerns; the contraction of these tensors still goes through `pyscf-algebra` (D-03/D-08).

---

### `crates/pyscf-core/src/amplitudes.rs` (MODIFY — Tensor-handle upgrade, role-match) — D-01

**Analog:** the file's own skeleton (lines 1-16) + the `pyscf-mp2` `UmpAmplitudes` multi-tensor container shape (`ump2.rs:52-69`). The skeleton's field comment ALREADY anticipates this (amplitudes.rs:13-15): *"Phase 6 wires (likely as opaque Tensor for spillability)."* Upgrade `t1: Vec<f64>`/`t2: Vec<f64>` → opaque `Tensor` handles produced by `WorkspacePool` (D-01). `Amplitudes` *consumes* handles; the pool *produces* them (D-08 split). Keep `nocc`/`nvir`. Note: `pyscf-mp2::rmp2_kernel` currently builds `Amplitudes { t1: Vec::new(), t2 }` (mp2.rs:300-305) — coordinate the field-type change so MP2 still compiles (the planner must touch the MP2 construction site or keep a Vec-compat constructor).

---

### `crates/pyscf-py/src/cc.rs` (NEW — PyO3 bridge, exact) — D-09

**Analog:** `crates/pyscf-py/src/mp.rs` (the WHOLE file — copy it section-for-section). This is the second-closest one-to-one analog after `hooks.rs`. The module doc-comment (mp.rs:1-25) is the template for the `cc` submodule doc.

Sections to copy (all verified line ranges):
- **`register`** (mp.rs:47-55): `m.add_class::<PyRCCSD>()` + `PyUCCSD` + `PyDFCCSD` + `PyCcsdScanner`; `m.add_function(wrap_pyfunction!(ccsd_factory, m)?)`.
- **Eager snapshot** (mp.rs:63-106 `snapshot_reference`): pull `mf.mol`/`mo_coeff`(F-order)/`mo_energy`/`mo_occ`/`e_tot`(→e_hf)/`converged` into a plain-array `CcsdReference`. Copy verbatim.
- **`build_df_integrals`** (mp.rs:111-114): `default_ri` + `cholesky_eri` for `PyDFCCSD`. Copy.
- **`is_overridden`** (mp.rs:130-150): `__qualname__` base-class comparison — copy verbatim, pass `&["RCCSD"]` etc.
- **`CcsdPyBridge: CcsdOverrideHooks`** (mp.rs:157-212 `Mp2PyBridge`): hold `Py<PyAny> slf` + `base_classes` + default source; each hook checks `is_overridden` then either `call_method1`s the override OR runs the pure-Rust default under `py.detach`. For CCSD the heaviest default path is the per-iteration `update_amps` hook (the biggest `py.detach` region in the project).

```rust
// crates/pyscf-py/src/mp.rs:204-209 — the GIL discipline. Copy EXACTLY for cc.rs.
// DEFAULT path: pure-Rust compute under py.detach (BIND-05). Kernel does NOT detach.
match &self.default_source {
    DefaultAo2mo::InCore => Python::attach(|py| py.detach(|| default_ao2mo(refr, frozen))),
    DefaultAo2mo::Df(df) => Python::attach(|py| py.detach(|| pyscf_mp2::df_ao2mo(refr, frozen, df))),
}
```

- **`PyRCCSD` / kernel** (mp.rs:218-375): `#[pyclass(subclass, name="RCCSD", module="pyscf._native.cc")]`; `#[new]`; `e_corr`/`e_tot` getters; `kernel` snapshots config, builds the bridge, calls `ccsd_kernel` — **NB the kernel does NOT `py.detach` at the top, hooks re-enter Python** (mp.rs:357-359, the load-bearing comment):
```rust
// crates/pyscf-py/src/mp.rs:357-365 — the kernel pattern. The comment is load-bearing.
// NB: we DO NOT py.detach here — hooks re-enter Python (scf.rs:345-350).
let bridge_slf: Py<PyAny> = slf.clone().into_any().unbind();
let bridge = CcsdPyBridge { slf: bridge_slf, base_classes: vec!["RCCSD"], default_source: ... };
let result = ccsd_kernel(&refr, &frozen, &bridge, &pool).map_err(pyscf_to_py)?;
```
- **`make_rdm1`/`make_rdm2`** via `rdm_via_bridge` (mp.rs:390-415, 736-757) — copy, add `solve_lambda` method (D-03/CCSD-05).
- **`as_scanner` / `PyCcsdScanner`** (mp.rs:425-439, 772-840): copy the Mole→energy callable that re-runs `mf.as_scanner()(mol)`, re-snapshots, re-runs the kernel. The `Mp2Kind` enum (mp.rs:766-770) → `CcsdKind { Restricted, Unrestricted, DensityFitted }`.

---

### `python/pyscf/cc/__init__.py` overlay + `ccsd_factory` (route, exact) — D-09

**Analog:** `mp2_factory` (mp.rs:847-903) + `mf_is_uhf`/`mf_has_df` (mp.rs:849-876). **Upstream port:** `pyscf/cc/__init__.py:83-139` (verified): `RCCSD` resolves to `dfccsd.RCCSD` if `with_df` else `ccsd.CCSD` (`:113-120`); `UCCSD` for UHF (`:84,101`). Copy the dispatch order: UHF→PyUCCSD, `with_df`→PyDFCCSD, else→PyRCCSD.

```rust
// crates/pyscf-py/src/mp.rs:886-903 — copy this factory shape, rename MP2→CCSD
#[pyfunction]
#[pyo3(name = "CCSD", signature = (mf, frozen=None))]
fn ccsd_factory(py: Python<'_>, mf: Py<PyAny>, frozen: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    let frozen_spec = parse_frozen(py, frozen)?;
    if mf_is_uhf(py, &mf) { /* PyUCCSD */ }
    if mf_has_df(py, &mf) { /* PyDFCCSD */ }
    /* PyRCCSD */
}
```
The overlay also wires `mf.CCSD()` / `mf.density_fit().CCSD()` cross-module dispatch (the `cc/__init__.py` overlay, mirroring how the `mp` submodule exposes `MP2()`).

---

### `crates/pyscf-ccsd/Cargo.toml` (MODIFY — config, exact)

**Analog:** `crates/pyscf-mp2/Cargo.toml`. Wire deps (member already registered Phase 1; verified dep list in 06-RESEARCH.md:115-130): `pyscf-core`, `pyscf-algebra`, `pyscf-ao2mo`, `pyscf-mp2`, `pyscf-scf`, `pyscf-df`, `pyscf-diis`, `pyscf-chkfile`, `pyscf-gto`, `pyscf-runtime`, `tracing`, `thiserror` — all `{ workspace = true }`. **NO `pyo3` dep (D-09). NO `cubecl-*`/`hdf5-metno` dep** (the denylist auto-enforces cubecl; D-07 forbids a second hdf5 owner). **Verify:** `cargo tree -p pyscf-ccsd | grep -i libxc` returns nothing.

---

### `xtask/src/bin/check_no_fma.rs` (MODIFY — config, exact)

**Analog:** the `SCAN_TARGETS` array (lines 81-84). Add `("pyscf-ccsd", "pyscf_ccsd")` so CCSD's own symbols are FMA-checked under `release-oracle` (RESEARCH Pitfall 1/2 + Wave 0 gaps). The `is_in_workspace` guard (lines 90-94) is forward-compatible so this is safe.

```rust
// xtask/src/bin/check_no_fma.rs:81-84 — add the pyscf-ccsd entry
const SCAN_TARGETS: &[(&str, &str)] = &[
    ("pyscf-algebra", "pyscf_algebra"),
    ("pyscf-core", "pyscf_core"),
    ("pyscf-ccsd", "pyscf_ccsd"),     // ADD (CCSD-own symbols FMA-free under release-oracle)
];
```

> **DO NOT touch `check_dependency_wall.rs`** — it is a DENYLIST (`FORBIDDEN_DEPS` = cubecl-*, `check_dependency_wall.rs:28-38`; `ALLOWED_CRATES` carve-out = only 3 crates, `:47`). `pyscf-ccsd` needs NO entry; the wall already covers it (RESEARCH Pitfall 3 — CONTEXT.md "extend allowlist" is inaccurate).

---

### `.github/workflows/ci.yml` (MODIFY — config, exact)

**Analogs (all verified):**
1. **`ccsd-structural`** (always-on small arm) — copy `mp2-structural` (ci.yml:416-427): `needs: [build-default]`, `cargo test -p pyscf-ccsd --locked -- --test-threads=1`. No `--features python`.
2. **`ccsd-oracle-upstream-manual`** (caffeine + DF-spill + λ/RDM byte-identity, human-verify) — copy `mp2-oracle-upstream-manual` (ci.yml:445-464): `if: github.event_name == 'workflow_dispatch'`, `pip install "pyscf>=2.5"`, `cargo test -p pyscf-oracle --features python --locked -- --test-threads=1`.
3. **`python3.13t` CCSD smoke** (Pitfall 6, HEAVIEST GIL re-validation) — clone `python3.13t SCF smoke` (ci.yml:274-323): same 3.13t venv + `maturin develop --no-default-features --features free-threading` (abi3 incompatible with 3.13t) + import-assert + a CCSD smoke replacing the SCF body.
4. **heap-alloc-count gate** (CCSD-11) — a dedicated `cargo test -p pyscf-ccsd --test heap_alloc_count` arm (the test target has its OWN counting `#[global_allocator]`, NOT linked to the oracle/determinism binaries — RESEARCH A3).

---

## Shared Patterns

### Reduction discipline (oracle_sum/oracle_dot, NEVER gemm/+=)
**Source:** `crates/pyscf-mp2/src/mp2.rs:275-296` + `crates/pyscf-ao2mo/src/transform.rs:9-14`
**Apply to:** every contraction in `ccsd.rs`, `update_amps.rs`, `rintermediates.rs`, `uccsd.rs`, `uintermediates.rs`, `lambda.rs`, `rdm.rs`, `diagnostics.rs`, `diis_amps.rs`.
```rust
let edi = 2.0 * oracle_dot(&gi, &t2i);   // materialize-then-reduce, NO +=, NO gemm (stub)
e_ss_terms.push(edi * 0.5 + exi);
let e_ss = oracle_sum(&e_ss_terms);      // single final reduction
```
`pyscf_algebra::gemm` is `NotYetImplemented{phase:2}` (verified `gemm.rs:16`) — host loops only.

### Override-hook seam + `Python::detach` GIL discipline
**Source:** `crates/pyscf-py/src/mp.rs:130-212` (`is_overridden` + `Mp2PyBridge`) + the kernel comment `mp.rs:357-359`.
**Apply to:** all of `crates/pyscf-py/src/cc.rs`.
The kernel does NOT detach (hooks re-enter Python); each hook's pure-Rust DEFAULT path wraps compute in `Python::attach(|py| py.detach(|| ...))`. Phase 6 is the heaviest 3.13t re-validation — the `update_amps` default is the biggest detach region in the project.

### Shape validation at every untrusted boundary (V5)
**Source:** `crates/pyscf-mp2/src/mp2.rs:191-205` (`ShapeMismatch` `?`-propagation).
**Apply to:** every hook-returned block in `ccsd.rs`/`uccsd.rs` + the PyO3 NumPy extraction in `cc.rs`. Validate length BEFORE indexing; `#![forbid(unsafe_code)]` means no UB on logic error.

### DIIS reuse (one DIIS, second storable)
**Source:** `crates/pyscf-diis/src/storable.rs:12-30` + `cdiis.rs:33-142,199-214`.
**Apply to:** `diis_amps.rs`. `dot` through `oracle_dot` (Pitfall 9); `Diis::<AmplitudeSubspace>::new(6)`; NO new DIIS body.

### HDF5 spill via the re-exported alias (no new dep)
**Source:** `crates/pyscf-chkfile/src/lib.rs:37` (`pub use hdf5_metno as hdf5`).
**Apply to:** `dfccsd.rs` + the `WorkspacePool` Spilled backend. RAII drop-delete the spill temp file (D-07).

### Frozen-core reuse (no re-port)
**Source:** `crates/pyscf-mp2/src/{frozen.rs, helpers.rs}` — `Frozen` enum + `get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`mo_without_core` (re-exported `pyscf-mp2/src/lib.rs:36-37`).
**Apply to:** `ccsd.rs`/`uccsd.rs` (CCSD-10). Already contract-tested vs `cc/ccsd.py:35` in Phase 5. Import verbatim.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/pyscf-runtime/src/workspace_pool.rs` (arena body) | runtime | n/a | No in-tree reuse-pool with a spill-backend enum exists. The struct/method SURFACE is the only analog (its own Phase-1 skeleton). Genuinely new engineering; RESEARCH Pattern 1 gives the recommended shape; the handle-unification (A2) is a Wave-1 design decision. |
| `crates/pyscf-ccsd/src/direct.rs` (AO-direct ERI streaming) | transform | streaming | No shell-sliced streaming `int2e` primitive exists in `pyscf-gto` (only the full arity-4 tensor, used whole in mp2.rs:150). `transform.rs` host-loop is the closest shape; v1 may compute the full tensor once + tile in MO space (RESEARCH Open Q4 — confirm against CCSD-07 `direct=True`). |

> For both, the planner uses RESEARCH §"Pattern 1" / §"Open Q4" patterns rather than a copy-from-analog approach.

---

## Metadata

**Analog search scope:** `crates/pyscf-mp2/src/`, `crates/pyscf-py/src/`, `crates/pyscf-diis/src/`, `crates/pyscf-runtime/src/`, `crates/pyscf-core/src/`, `crates/pyscf-chkfile/src/`, `crates/pyscf-ao2mo/src/`, `xtask/src/bin/`, `.github/workflows/ci.yml`; upstream port targets `pyscf/cc/{ccsd,rintermediates,dfccsd,__init__}.py`.
**Files scanned:** ~22 in-tree + 4 upstream port references.
**Verified findings carried from RESEARCH:** gemm is a stub (`gemm.rs:16`); dependency wall is a denylist (`check_dependency_wall.rs:28-47`); `conv_tol_normt=1e-5`/`diis_space=6`/`diis_start_cycle=0` (`ccsd.py:920-928`); `WorkspacePool`/`Amplitudes`/`DiisStorable` skeletons reserved exactly for this phase.
**Pattern extraction date:** 2026-05-24
