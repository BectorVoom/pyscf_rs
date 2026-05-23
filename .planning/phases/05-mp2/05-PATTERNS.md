# Phase 5: MP2 - Pattern Map

**Mapped:** 2026-05-23
**Files analyzed:** 18 new/modified (2 new crates filled/created + pyscf-py surface + python overlay + workspace/CI bookkeeping)
**Analogs found:** 16 / 18 (2 have no in-repo analog — the AO→MO 4-index transform body + the spin-resolved UMP2 amplitude container)

> Default stance for every file: **mirror upstream** (sibling-crate fidelity, the hard preference carried from Phases 1–4) AND copy the shipped Rust precedent from the analog listed. Every analog path/line below was source-verified this session.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/pyscf-ao2mo/Cargo.toml` | config | — | `crates/pyscf-df/Cargo.toml` | exact (same dep triple: core+algebra+gto) |
| `crates/pyscf-ao2mo/src/lib.rs` | lib-root | — | `crates/pyscf-df/src/lib.rs` | exact (own-crate-per-shared-concern) |
| `crates/pyscf-ao2mo/src/incore.rs` | service | transform | `crates/pyscf-df/src/cholesky_eri.rs` | role-match (port-of-incore.py + flat-index discipline) |
| `crates/pyscf-ao2mo/src/transform.rs` | service | transform | `crates/pyscf-dft/src/numint.rs` (host-loop einsum) | partial (host-loop-through-`oracle_sum` model; transform body itself has NO analog) |
| `crates/pyscf-ao2mo/src/addons.rs` | utility | transform | `crates/pyscf-df/src/cholesky_eri.rs` (packing/index helpers) | role-match (only if a call site needs `restore`) |
| `crates/pyscf-ao2mo/src/error.rs` | config | — | `crates/pyscf-scf/src/error.rs` | exact (thiserror + `From<E> for PyscfRsError`) |
| `crates/pyscf-mp2/src/lib.rs` | lib-root | — | `crates/pyscf-df/src/lib.rs` + `crates/pyscf-scf/src/lib.rs` | exact (re-export surface) |
| `crates/pyscf-mp2/src/mp2.rs` (RMP2 kernel) | service | CRUD/transform | `crates/pyscf-dft/src/numint.rs::nr_rks_inner` | role-match (closed-form energy via `oracle_sum`) |
| `crates/pyscf-mp2/src/ump2.rs` (UMP2 spin-block) | service | CRUD/transform | `crates/pyscf-dft/src/hooks.rs::UksKsHooks` | role-match (α/β spin-channel split) |
| `crates/pyscf-mp2/src/dfmp2.rs` (conventional) | service | CRUD/transform | `crates/pyscf-df/src/cholesky_eri.rs` (B-tensor source) | role-match (swap ERI source; `DFRMP2(:RMP2)`) |
| `crates/pyscf-mp2/src/dfmp2_native.rs` (native RI) | service | CRUD/transform | `crates/pyscf-df/src/cholesky_eri.rs` + `pyscf-algebra::solve_linear`/`eigh_gen` | partial (CPHF path; follow-on plan) |
| `crates/pyscf-mp2/src/helpers.rs` (MP2-08) | utility | request-response | `crates/pyscf-scf/src/occ.rs` / `rdm.rs` (plain pub fns) | role-match (plain free functions, not bridged) |
| `crates/pyscf-mp2/src/frozen.rs` (MP2-03) | utility | transform | `crates/pyscf-df/src/auxbasis.rs` (static lookup table) | role-match (chemcore table = same `OnceLock<HashMap>` shape) |
| `crates/pyscf-mp2/src/rdm.rs` (MP2-05) | service | transform | `crates/pyscf-scf/src/rdm.rs` + `pyscf-dft/src/numint.rs` (index loops) | role-match |
| `crates/pyscf-mp2/src/hooks.rs` (D-08) | trait/provider | event-driven | `crates/pyscf-dft/src/hooks.rs` (`KsOverrideHooks`+`NoKsOverrides`) | exact (trait + default-impl, pyo3-free) |
| `crates/pyscf-mp2/src/error.rs` | config | — | `crates/pyscf-scf/src/error.rs` | exact |
| `crates/pyscf-py/src/mp.rs` (PyRMP2/PyUMP2/PyDFMP2 + bridge) | controller | request-response | `crates/pyscf-py/src/scf.rs` (`PyRHF`/`PyUHF`) + `crates/pyscf-py/src/bridge.rs` | exact (eager-snapshot + `call_method1` + `py.detach`) |
| `python/pyscf/mp/__init__.py` (overlay) | route/config | request-response | `python/pyscf/scf/__init__.py` | exact (re-export from `_native`) |
| `Cargo.toml` (workspace members) | config | — | existing `members = [...]` block | exact (add `crates/pyscf-ao2mo`) |
| `.github/workflows/ci.yml` (MP2 jobs) | config | — | Phase 3 DF-HF / Phase 4 DFT-01 cintx-gated jobs | role-match (mirror gating) |

---

## Pattern Assignments

### `crates/pyscf-ao2mo/Cargo.toml` (config) — NEW CRATE

**Analog:** `crates/pyscf-df/Cargo.toml` (read this session — exact dep triple)

The new crate's deps are identical to pyscf-df: `pyscf-core`, `pyscf-algebra`, `pyscf-gto`, `thiserror`, `tracing`. **No pyo3, no cubecl** (D-01/D-03 algebra wall). Copy verbatim, change `name` and `description`:

```toml
[package]
name = "pyscf-ao2mo"
version.workspace      = true
edition.workspace      = true
rust-version.workspace = true
license.workspace      = true
description = "AO→MO integral transformation (general/full). CCSD-reusable (D-01). Phase 5."

[lib]
path = "src/lib.rs"

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-algebra = { path = "../pyscf-algebra" }
pyscf-gto     = { path = "../pyscf-gto" }
thiserror     = { workspace = true }
tracing       = { workspace = true }
```

**A5 (verified this session):** the `check_dependency_wall` lint is a **cubecl denylist** with carve-out `ALLOWED_CRATES = ["pyscf-algebra", "pyscf-runtime", "pyscf-kernels"]` (`xtask/src/bin/check_dependency_wall.rs:47`). A new crate passes automatically **as long as it never names a `cubecl-*` dep**. CONTEXT's "extend the allowlist" wording is imprecise — **no lint edit is required**; the design constraint is simply "don't add cubecl."

---

### `crates/pyscf-ao2mo/src/lib.rs` (lib-root) — NEW

**Analog:** `crates/pyscf-df/src/lib.rs` (read this session)

Copy the module-doc-+-`#![forbid(unsafe_code)]`-+-`pub mod`/`pub use` re-export shape. The df lib.rs is the template:

```rust
//! pyscf-ao2mo: AO→MO 4-index integral transformation.
//! Source: D-01 — mirrors upstream `pyscf/ao2mo/` (incore.py / _ao2mo.py / addons.py).
//! Public surface: `general(eri_or_mol, (C1,C2,C3,C4))` + `full(mol, mo_coeff)` (D-02),
//! consumed by pyscf-mp2 for the `(occ,vir|occ,vir)` block and by pyscf-ccsd (Phase 6).
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod incore;
pub mod transform;
pub mod error;

pub use error::Ao2moError;
pub use incore::{full, general};
```

Note `#![forbid(unsafe_code)]` (mp2 stub and df both use it) + `#![warn(clippy::unwrap_used)]` (df lib.rs:15). The mp2 lib.rs stub currently has only `#![forbid(unsafe_code)]` — extend it the same way.

---

### `crates/pyscf-ao2mo/src/incore.rs` + `transform.rs` (service, transform)

**Analog (flat-index + Tensor-API-not-ready discipline):** `crates/pyscf-df/src/cholesky_eri.rs` (read this session)
**Analog (host-loop-through-`oracle_sum` contraction):** `crates/pyscf-dft/src/numint.rs` `eval_rho`/`contract_rho_comp` lines 303-351 (read this session)

**The 4-index transform body itself has NO in-repo analog** — it is the one genuinely new compute in the phase. But two shipped patterns fully determine HOW to write it:

**(1) Flat-index + layout discipline — copy from `cholesky_eri.rs:135-153`:**
```rust
// int3c is F-order shape [nao, nao, naux]:
//   element (mu, nu, q) lives at int3c.values[mu + nu * nao + q * nao * nao].
// b_uvq is row-major [nao, nao, naux]:
//   element (mu, nu, q) lives at b_uvq[mu * nao * naux + nu * naux + q].
```
This is the model: spell out the exact flat-index formula for EVERY tensor at EVERY boundary in a doc-comment (Pitfall 3 mitigation). `MOCoefficients.data` is **column-major / F-order** (`pyscf-core/src/mo.rs:11-14`); `mol.intor("int2e")` returns F-order `[nao,nao,nao,nao]` (`pyscf-gto/src/intor.rs:196-198` F-order convention). Slice `co = mo_coeff[:, :nocc]` with F-order column semantics (upstream `mp2.py:793` uses `order='F'`).

**(2) Host-loop-through-`oracle_sum` — copy the shape from `numint.rs::eval_rho` (lines 305-314):**
```rust
let mut terms = vec![0.0_f64; nao * nao];   // materialize products into a Vec FIRST
for g in 0..ngrids {
    for mu in 0..nao {
        let a_mu = ao_at(ao_l, 0, g, mu);
        for nu in 0..nao {
            terms[mu * nao + nu] = a_mu * dm.data[mu * nao + nu] * ao_at(ao_r, 0, g, nu);
        }
    }
    rho[g] = oracle_sum(&terms);            // bit-exact reduction (Pitfall 1/2)
}
```
The AO→MO quarter-transform `(pq|rs) --C_p--> (iq|rs) --C_q--> (ij|rs) --C_r--> (ij|ks) --C_s--> (ij|kl)` is the same idiom: each contraction index materializes its per-element products into a `Vec`, then `oracle_sum`/`oracle_dot` reduces. **`pyscf_algebra::gemm` is NOT an option** — it returns `NotYetImplemented{phase:2}` (`pyscf-algebra/src/gemm.rs:17`). Reduction surface (verified `pyscf-algebra/src/oracle.rs`):
- `oracle_sum(xs: &[f64]) -> f64` — pairwise tree, fixed `PAIRWISE_CHUNK = 128`, thread-count invariant.
- `oracle_dot(a: &[f64], b: &[f64]) -> f64` — materializes `a[i]*b[i]` then `oracle_sum`; returns `NaN` on length mismatch.
- `oracle_einsum("ij,jk->ik", a, a_shape, b, b_shape) -> Option<Vec<f64>>` — the only general-einsum case shipped (binary contraction). Other patterns return `None`.

**Always-on test (no cintx dependency):** feed `general` a **hand-built `nao^4` AO ERI** (synthetic, not from `intor`) and assert the MO transform matches a longhand reference — this is the one numeric assertion that ships un-gated this phase (RESEARCH Validation Architecture; `cholesky_eri.rs` unit tests are the model for synthetic-input correctness tests).

---

### `crates/pyscf-ao2mo/src/error.rs` + `crates/pyscf-mp2/src/error.rs` (config)

**Analog:** `crates/pyscf-scf/src/error.rs` (read this session — full file)

Copy the `thiserror` enum + `From<E> for pyscf_core::PyscfRsError` bridge verbatim. The exact shape to mirror:
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Mp2Error {
    #[error("algebra: {0}")]
    Algebra(#[from] pyscf_algebra::AlgebraError),
    #[error("core: {0}")]
    Core(#[from] pyscf_core::CoreError),
    // ... MP2-specific variants (FrozenCoreInvalid, NonCanonicalReference, ...)
}

impl From<Mp2Error> for pyscf_core::PyscfRsError {
    fn from(e: Mp2Error) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
```
Note (scf/error.rs:22-29): the bridge routes through `Core(InvalidMolecule(String))` to avoid touching `pyscf-core::error.rs`. The `pyscf-df/src/error.rs` is the second precedent (`DfError::SingularAux` → `From<DfError>`).

---

### `crates/pyscf-mp2/src/mp2.rs` — RMP2 kernel/energy (service, CRUD/transform)

**Analog (closed-form energy + `oracle_sum` reductions):** `crates/pyscf-dft/src/numint.rs::nr_rks_inner` lines 491-566 (read this session)
**Upstream port target:** `pyscf/mp/mp2.py:kernel` (line 33), `energy` (line 117), `class RMP2(MP2Base)` (line 692), `class MP2Base` (line 463)

**Reference snapshot struct (D-07) — pyo3-free, takes plain arrays:**
```rust
pub struct Mp2Reference {
    pub mo_coeff: MOCoefficients,   // F-order, sign-canonicalized (SCF-13) — pyscf-core/src/mo.rs
    pub mo_energy: Vec<f64>,
    pub mo_occ: Vec<f64>,
    pub e_hf: f64,
    pub converged: bool,
}
```
This consumes `pyscf-scf::ScfResult` fields directly (`kernel.rs:51-59`: `e_tot`/`mo_coeff`/`mo_energy`/`mo_occ`/`converged`). MO coefficients are already sign-canonicalized (SCF-13) — MP2 does NOT re-diagonalize, so no new sign work (Pitfall 4/12).

**Energy reduction discipline (copy from numint.rs:557-566):** every einsum in the upstream closed-form (`edi = einsum('jab,jab', t2i, gi)*2`, `exi = -einsum('jab,jba', t2i, gi)`) becomes a host loop materializing products into a `Vec` then `oracle_sum`. The `eia = ε_i - ε_a` denominators are exact subtractions. Final `e_corr = oracle_sum(per_i_contributions)`. Source math: `mp2.py:47-76`.

**`PostScf` trait impl:** `pyscf-core/src/traits.rs:42-51` declares `PostScf: Method` with `type Reference: Scf`, `fn reference()`, `fn e_correlation() -> Result<Energy, _>`. Phase 5 implements it for the MP2 types. `Method::kernel(&mut self) -> Result<Energy, _>` + `fn mol()` (traits.rs:12-18) is the supertrait.

**Amplitude storage:** `pyscf-core/src/amplitudes.rs::Amplitudes { nocc, nvir, t1, t2 }` — `t2` is flat `[nocc,nocc,nvir,nvir]` (amplitudes.rs:13-15). RMP2 ignores `t1`. `with_t2=true` default (upstream `WITH_T2`, mp2.py:30).

---

### `crates/pyscf-mp2/src/ump2.rs` — UMP2 spin-block (service)

**Analog (α/β spin-channel split + per-channel handling):** `crates/pyscf-dft/src/hooks.rs::UksKsHooks` lines 383-636 (read this session — the open-shell precedent)
**Upstream port target:** `pyscf/mp/ump2.py` (`kernel`/`energy`/`get_nocc`/`get_frozen_mask`/`_make_eris` spin-block)

The UKS hooks are the closest shipped open-shell pattern: symmetric/asymmetric spin split, per-channel reductions, combined results. UMP2 needs the `(t2aa, t2ab, t2bb)` amplitude triple — the single-channel `Amplitudes` struct (amplitudes.rs) does **not** cover this. **No analog for a spin-resolved amplitude container** — Phase 5 adds either a `UmpAmplitudes { t2aa, t2ab, t2bb }` struct in `pyscf-mp2` or a spin-resolved variant. Spin-resolved frozen mask + RDMs follow the same per-channel structure as `UksKsHooks::uks_veff` (hooks.rs:435-525).

---

### `crates/pyscf-mp2/src/dfmp2.rs` — conventional DF-MP2 (service)

**Analog (DF B-tensor source):** `crates/pyscf-df/src/cholesky_eri.rs::DfIntegrals` (read this session)
**Upstream port target:** `pyscf/mp/dfmp2.py:124` `class DFRMP2(mp2.RMP2)`; `_make_df_eris` (line ~215); `scf.hf.RHF.DFMP2 = lib.class_as_method(DFMP2)` (line 192)

**Pattern: swap the ERI source, reuse the RMP2 base (D-06).** Upstream `DFRMP2` subclasses `mp2.RMP2`. In Rust this is a thin wrapper providing a different `Mp2OverrideHooks::ao2mo` impl backed by `DfIntegrals.b_uvq`:
```rust
// DfIntegrals (pyscf-df/src/cholesky_eri.rs:41-49):
//   pub struct DfIntegrals { pub b_uvq: Vec<f64>, pub naux: usize, pub nao: usize }
//   b_uvq is row-major [nao, nao, naux]: element (mu, nu, q) at b_uvq[mu*nao*naux + nu*naux + q].
// (ia|jb) = sum_Q B^Q_ia · B^Q_jb  — B^Q in MO basis = ao2mo of b_uvq, contracted via oracle_dot over Q.
// Upstream uses C driver libmp.MP2_contract_d; port the MATH as oracle_sum loop (NO C dep).
```
**A2 (verified this session):** DF-MP2 aux default IS the `*-ri`/mp2fit basis. Upstream `dfmp2.py:136` calls `make_auxbasis(mol, mp2fit=True)`. `pyscf-df::default_ri(basis)` (auxbasis.rs:76-81) already resolves the `*-ri` aux (e.g. `cc-pvdz` → `cc-pvdz-ri`) — use `default_ri`, NOT `default_jkfit`. The B-tensor assembly itself reuses `cholesky_eri(mol, default_ri(basis))`.

**D-05 gating:** `cholesky_eri` already documents that `int3c2e_sph` returns a zero-filled buffer until cintx lands (cholesky_eri.rs:104-108) — the DF-MP2 code needs NO change when cintx merges; the numeric oracle is CI-gated, structural tests are always-on.

---

### `crates/pyscf-mp2/src/dfmp2_native.rs` — native RI-MP2 (service, follow-on)

**Analog:** `crates/pyscf-df/src/cholesky_eri.rs` (B-tensor + inline Cholesky) + `pyscf-algebra::{solve_linear, eigh_gen}` (lib.rs:49,56 — for the CPHF relaxed-RDM path)
**Upstream port target:** `pyscf/mp/dfmp2_native.py` (`ints3c_cholesky`, `emp2_rhf`, `solve_cphf_rhf`)

A distinct module (own path `pyscf.mp.dfmp2_native`), sequenced as a **follow-on plan** after conventional proves out (D-06). May stage behind a status marker (cintx-ECP / D-11 style). The CPHF linear solve reuses `pyscf_algebra::solve_linear` (already wired, lib.rs:56). Both DF paths gate behind the same cintx merge.

---

### `crates/pyscf-mp2/src/helpers.rs` — MP2-08 helpers (utility)

**Analog (plain pub free functions):** `crates/pyscf-scf/src/occ.rs::default_get_occ`, `crates/pyscf-scf/src/rdm.rs::default_make_rdm1` (referenced via hooks.rs:85-90)
**Upstream port target (the EXACT CCSD import contract):** `pyscf/cc/ccsd.py:35`
```python
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
```
Source line anchors (verified this session in `pyscf/mp/mp2.py`):
- `get_nocc` — line 351 (`count_nonzero(mo_occ>0)` minus frozen)
- `get_nmo` — line 371 (`len(mo_occ)` minus frozen)
- `get_frozen_mask` — line 383 (bool array, False at frozen indices)
- `get_e_hf` — line 402 (`mp._scf.e_tot` for the converged ref)
- `_mo_without_core` — line 732 (`mo[:, get_frozen_mask(mp)]`)
- (also) `_mo_splitter` — line 206

These are **plain `pub` functions, NOT bridged hooks** (D-08 — subclasses rarely override them). They are exported so the Phase 6 CCSD import works verbatim. **Contract test:** a unit test in `crates/pyscf-mp2/tests/ccsd_import_contract.rs` that mimics the `cc/ccsd.py:35` call site (asserts the five symbols exist with the right signatures + semantics on a toy `mo_occ`).

---

### `crates/pyscf-mp2/src/frozen.rs` — frozen-core (utility, MP2-03)

**Analog (static lookup table shape):** `crates/pyscf-df/src/auxbasis.rs` (read this session — `OnceLock<HashMap>` + fallback)
**Upstream port target:** `pyscf/mp/mp2.py:set_frozen` (line ~575, delegates to `pyscf.cc.ccsd.set_frozen(method='auto', window=...)`)

`frozen=int` (count of lowest MOs), `frozen=list` (explicit indices), `frozen='auto'` (chemcore autodetection), frozen-window forms. The `'auto'` element→core-count table follows the **exact same static-table shape** as `DEFAULT_AUXBASIS` (auxbasis.rs:15-58: `static TABLE: OnceLock<HashMap<...>>` + `get_or_init` + a fallback). **A1 (open):** the chemcore table source is `pyscf.cc.ccsd.set_frozen` — locate + port the element table verbatim during planning; defaults must match upstream on the corpus.

---

### `crates/pyscf-mp2/src/rdm.rs` — make_rdm1/make_rdm2 (service, MP2-05)

**Analog:** `crates/pyscf-scf/src/rdm.rs::default_make_rdm1` + `crates/pyscf-dft/src/numint.rs` (the nested-index loop discipline)
**Upstream port target:** `pyscf/mp/mp2.py:make_rdm1` (line 151), `make_rdm2` (line 275), `_gamma1_intermediates`, `_mo_splitter` (line 206)

Mirror upstream `make_rdm1(t2, eris, ao_repr=False, with_frozen=True)` / `make_rdm2(..., ao_repr=False)` incl. the `ao_repr`/`with_frozen` flags. **Sequencing (Pitfall 5):** port `make_rdm1`/`_gamma1_intermediates` FIRST (simpler — `doo`/`dvv` only), unit-test against upstream small-system RDMs, THEN layer `make_rdm2` (the `nmo0^4` tensor with occ/vir/frozen sub-block fancy-indexing) with explicit flat-index maps (the `cholesky_eri.rs:135-153` flat-index doc-comment discipline applies).

---

### `crates/pyscf-mp2/src/hooks.rs` — Mp2OverrideHooks (trait/provider, D-08)

**Analog:** `crates/pyscf-dft/src/hooks.rs` (`KsOverrideHooks` trait + `NoKsOverrides` default-impl) AND `crates/pyscf-scf/src/hooks.rs` (`OverrideHooks` + `NoOverrides`) — both read this session

**Smaller hook set than SCF (D-08):** `ao2mo` (the main one — custom integral source), `make_rdm1`, `make_rdm2`, `energy`. Copy the trait + default-impl shape from `pyscf-scf/src/hooks.rs:13-107`:
```rust
use pyscf_core::{PyscfRsError, /* MP2 types */};

pub trait Mp2OverrideHooks {
    fn ao2mo(&self, refr: &Mp2Reference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError>;
    fn make_rdm1(&self, /* ... */) -> Result<Density, PyscfRsError>;
    fn make_rdm2(&self, /* ... */) -> Result<Density, PyscfRsError>;
    fn energy(&self, /* ... */) -> Result<Energy, PyscfRsError>;
}

pub struct NoMp2Overrides;
impl Mp2OverrideHooks for NoMp2Overrides {
    fn ao2mo(&self, refr: &Mp2Reference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
        crate::mp2::default_ao2mo(refr, frozen)   // delegates to pyscf-ao2mo
    }
    // ... each method delegates to a crate::*::default_* free fn (hooks.rs:51-107 pattern)
}
```
`pyscf-mp2` stays **pyo3-free** — the trait is declared here; the `call_method1` bridge lives in `pyscf-py` (see next). The generic-kernel signature is `rmp2_kernel<H: Mp2OverrideHooks>(...)` (mirrors `pyscf-scf::kernel<H: OverrideHooks>`, kernel.rs:63).

---

### `crates/pyscf-py/src/mp.rs` — PyRMP2/PyUMP2/PyDFMP2 + bridge (controller, request-response)

**Analog:** `crates/pyscf-py/src/scf.rs` (`PyRHF`/`PyUHF`/`PyScfScanner`) + `crates/pyscf-py/src/bridge.rs` (`PyOverrideBridge`) — both read this session

**Eager-snapshot constructor (D-07) — copy `PyRHF::new` (scf.rs:60-67):**
```rust
#[pyclass(subclass, name = "RMP2", module = "pyscf._native.mp")]
pub struct PyRMP2 {
    pub(crate) inner: Mp2Reference,   // snapshotted mo_coeff/mo_energy/mo_occ/e_hf
    pub(crate) py_mf: Py<PyAny>,      // held for as_scanner re-run + hook dispatch
}
#[pymethods]
impl PyRMP2 {
    #[new]
    fn new(py: Python<'_>, mf: Py<PyAny>) -> PyResult<Self> {
        // extract mo_coeff/mo_energy/mo_occ/e_tot/nocc/nmo from the Python mf
        // (mirror scf.rs:62 extract_mole_from_pyany, but pull mf attributes)
    }
}
```

**Bridge dispatch (D-08) — copy `bridge.rs::call_hook` (lines 57-72) + `PyOverrideBridge` (lines 35-49):**
```rust
let result = slf.bind(py).call_method1(method, args).map_err(py_to_pyscf)?;
extract(result).map_err(py_to_pyscf)
```
Each `Mp2OverrideHooks` method on the bridge does `Python::attach` → build args → `call_method1("<hook>", args)` → extract NumPy (bridge.rs:74-139 pattern). The hooks are `ao2mo`/`make_rdm1`/`make_rdm2`/`energy`.

**kernel/run + GIL discipline (scf.rs:319-402):** `kernel` does **NOT** `py.detach` at the top (hooks re-enter Python via the bridge — scf.rs:345-350); the per-hook DEFAULT methods inside the pyclass DO `py.detach(|| ...)` around the pure-Rust compute (scf.rs:472,484,498 etc.). `run` = `kernel` then return self (scf.rs:399-402).

**NumPy boundary (BIND-04, V5):** reuse `crate::numpy_io::{to_density, to_mo_coeff, density_to_pyarray, mo_coeff_to_pyarray, slice_to_pyarray1}` (scf.rs:31-33) — `to_owned()` non-standard-layout inputs.

**`as_scanner` (MP2-07) — copy the closure shape from `pyscf-scf/src/scanner.rs` (full file) + `PyRHF::as_scanner` (scf.rs:694-699):** MP2's scanner holds a `Py<PyAny>` to the mf-scanner, re-runs `mf(mol)` at the new geometry, re-snapshots, then runs the MP2 kernel. The `PyScfScanner` pyclass (scf.rs:844-882) is the wrapper template.

**Factory dispatch:** `mf.MP2()` → RMP2 (RHF ref) / UMP2 (UHF ref); `mf.density_fit().MP2()` → conventional DFMP2 (mirror `pyscf/mp/__init__.py:MP2`/`RMP2`/`UMP2`, read this session — `mf.istype('UHF')` branch + `getattr(mf, 'with_df', None)` branch at lines 28-53). Relates to SCF-11 `to_rhf`/`to_uhf`.

---

### `python/pyscf/mp/__init__.py` — overlay (route/config)

**Analog:** `python/pyscf/scf/__init__.py` (read this session — full file)

The scf overlay is a 10-line re-export from `_native`:
```python
from pyscf._native.scf import RHF, UHF, GHF  # type: ignore[attr-defined]
__all__ = ["RHF", "UHF", "GHF"]
```
The mp overlay re-exports `RMP2`/`UMP2`/`MP2`/`DFMP2` from `pyscf._native.mp` and provides the `MP2(mf)` factory dispatch (mirror upstream `pyscf/mp/__init__.py:27-65` — the `istype`/`with_df` branching). **Note:** a real `pyscf/mp/__init__.py` exists in the vendored upstream (the oracle source) — the OVERLAY at `python/pyscf/mp/` is the pyscf-rs `_native` re-export, distinct from the vendored reference.

---

### `Cargo.toml` (workspace members) + `.github/workflows/ci.yml`

**Analog:** existing `members = [...]` block (read this session) + Phase 3 DF-HF / Phase 4 DFT-01 cintx-gated CI jobs

**Pitfall 6 (verified this session):** the workspace `members` list at `Cargo.toml:2-24` currently contains `crates/pyscf-mp2` and `crates/pyscf-ccsd` but **NOT `crates/pyscf-ao2mo`** — it must be ADDED. Count basis: there are currently 19 `pyscf-*` crates + `xtask` listed; adding `pyscf-ao2mo` makes 20 `pyscf-*` crates. Update ROADMAP prose to whichever convention it uses, consistently. First build regenerates `Cargo.lock` (RESEARCH Runtime State Inventory) — ensure CI lockfile updated.

**CI:** add an always-on MP2 structural job (`cargo test -p pyscf-mp2 -p pyscf-ao2mo`) + a numeric oracle job gated behind `cintx#11` (both `int2e` in-core AND `int3c2e_sph` DF — same gap, different arity). Mirror the existing DF-HF / DFT-01 `--features python` cintx-gated arms.

---

## Shared Patterns

### Bit-exact reductions (applies to ALL numeric files: ao2mo transform, mp2/ump2 energy, dfmp2 contraction, rdm)
**Source:** `crates/pyscf-algebra/src/oracle.rs:22-38` (verified this session)
**Apply to:** every contraction/energy sum in `pyscf-ao2mo` and `pyscf-mp2`
```rust
pub fn oracle_sum(xs: &[f64]) -> f64 { /* pairwise tree, fixed PAIRWISE_CHUNK = 128 */ }
pub fn oracle_dot(a: &[f64], b: &[f64]) -> f64 { /* materialize a[i]*b[i] then oracle_sum */ }
```
**Discipline (Pitfall 1/2):** materialize per-element products into a `Vec` FIRST so the tree shape depends only on length (thread-count invariant). NEVER plain `+=` accumulation. `gemm` is `NotYetImplemented{phase:2}` (`gemm.rs:17`) — host loops only.

### pyo3-free method crate + eager snapshot (D-07/D-08)
**Source:** `crates/pyscf-scf/Cargo.toml` (no pyo3 dep) + `crates/pyscf-py/src/scf.rs:60-67` (snapshot in pyscf-py)
**Apply to:** `pyscf-mp2` + `pyscf-ao2mo` Cargo.toml (NO pyo3, NO cubecl); all PyMP2 classes snapshot eagerly
The `check-dependency-wall` lint only denies cubecl (A5) — pyo3 absence is a **review-discipline item**. Model `pyscf-scf`/`pyscf-df`/`pyscf-dft` Cargo.toml: they declare `pyscf-core`/`pyscf-algebra`/`pyscf-gto` + `thiserror`/`tracing`, never pyo3.

### Trait-callback bridge for subclass overrides (Pitfall 7)
**Source:** `crates/pyscf-py/src/bridge.rs:57-72` (`call_hook` → `call_method1`) + `crates/pyscf-dft/src/hooks.rs:34-63` (`KsOverrideHooks` trait + default impl)
**Apply to:** `crates/pyscf-mp2/src/hooks.rs` (declare `Mp2OverrideHooks` + `NoMp2Overrides`) + `crates/pyscf-py/src/mp.rs` (the `call_method1` bridge impl)
```rust
let result = slf.bind(py).call_method1(method, args).map_err(py_to_pyscf)?;
```
`call_method1` resolves the MRO so a Python subclass override is invoked transparently (Pitfall 7 immune by construction).

### GIL / NumPy boundary (Pitfall 4/5/6, V5)
**Source:** `crates/pyscf-py/src/scf.rs:472` (`py.detach(|| default_*(...))`) + scf.rs:31-33 (`to_density`/`to_mo_coeff` converters)
**Apply to:** every hook default method in `crates/pyscf-py/src/mp.rs`
The kernel itself does NOT detach (hooks re-enter Python, scf.rs:345-350); per-hook defaults DO detach around pure-Rust compute. `to_owned()` non-standard-layout NumPy inputs (BIND-04).

### Flat-index layout discipline (Pitfall 3 / Pitfall 8)
**Source:** `crates/pyscf-df/src/cholesky_eri.rs:135-153` (per-tensor flat-index doc-comments)
**Apply to:** `pyscf-ao2mo` transform + `dfmp2` B-tensor contraction + `rdm` index maps
Spell out the exact flat-index formula for every tensor at every boundary. `MOCoefficients.data` is column-major/F-order (`pyscf-core/src/mo.rs:11`); `DfIntegrals.b_uvq` is row-major (`cholesky_eri.rs:43`); `intor` returns F-order. A silent transpose corrupts the `(ia|jb)` block.

### Static lookup-table shape (frozen-core 'auto', DF aux)
**Source:** `crates/pyscf-df/src/auxbasis.rs:15-58` (`OnceLock<HashMap>` + `get_or_init` + fallback)
**Apply to:** `crates/pyscf-mp2/src/frozen.rs` chemcore element→core table
DF-MP2 aux default: use the existing `pyscf_df::default_ri(basis)` (auxbasis.rs:76, the `*-ri`/mp2fit aux — A2 confirmed), NOT `default_jkfit`.

---

## No Analog Found

Files with no close in-repo match (planner uses the upstream port + RESEARCH patterns):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/pyscf-ao2mo/src/transform.rs` (the 4-index transform BODY) | service | transform | The AO→MO quarter-transform is the one genuinely new compute. No existing crate does a 4-index transform. The HOST-LOOP idiom is fully modeled by `numint.rs::eval_rho` + the flat-index discipline by `cholesky_eri.rs`, but the transform math itself is ported fresh from `pyscf/ao2mo/incore.py:general` (the `nao**4 einsum` branch, lines 123-128). |
| `crates/pyscf-mp2/src/ump2.rs` (spin-resolved amplitude container) | model | — | `pyscf-core::Amplitudes` (amplitudes.rs:8-16) is single-channel (`t1`/`t2`). UMP2 needs the `(t2aa, t2ab, t2bb)` triple. No existing spin-resolved amplitude struct — Phase 5 adds one. The α/β handling SHAPE is modeled by `UksKsHooks` (dft/hooks.rs:383-636) but the container is new. |

---

## Metadata

**Analog search scope:** `crates/pyscf-{scf,df,dft,algebra,core,gto,py,mp2}/`, `python/pyscf/{scf,mp}/`, `pyscf/mp/` (vendored upstream oracle), `xtask/src/bin/`, workspace `Cargo.toml`
**Files scanned:** ~22 (Rust analogs + upstream `mp2.py`/`dfmp2.py`/`mp/__init__.py` anchors + workspace/lint bookkeeping)
**Key corrections folded in:** A5 (no dep-wall edit — cubecl denylist auto-passes), A2 (`default_ri` already resolves the mp2fit aux), Pitfall 6 (`pyscf-ao2mo` is genuinely absent from `Cargo.toml` members), Pitfall 1 (in-core `int2e` is `NotYetImplemented{phase:2}` — numeric CI-gated, not un-gated)
**Pattern extraction date:** 2026-05-23
