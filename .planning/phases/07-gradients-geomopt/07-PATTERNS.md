# Phase 7: Gradients + Geomopt - Pattern Map

**Mapped:** 2026-05-26
**Files analyzed:** 20 (new + modified)
**Analogs found:** 17 / 20 (3 have only an external/upstream port reference — the optimizer engine)

> Read alongside `07-CONTEXT.md` (locked decisions D-01..D-09) and `07-RESEARCH.md`
> (cintx gradient-integral availability matrix, CPHF/geomopt port targets).
>
> **Two analog tiers in this phase:**
> 1. **In-tree structural analog** — the existing pyscf-rs crate/module the new file copies
>    its *shape* from (module split, error/hooks/reference contract, pyo3 bridge, dispatch guard,
>    CI gate). This is the "sibling-crate fidelity" tier.
> 2. **Upstream / external port reference** — the PySCF `./pyscf/grad/*.py` / `./pyscf/scf/cphf.py`
>    (in-repo) or geomeTRIC (external, fetch) the *body math* is mechanically ported from. CONTEXT
>    "Claude's Discretion" + RESEARCH "Code Examples" name these line-for-line.
>
> Both are cited per file below. The planner assigns the structural analog as the file skeleton
> and the upstream reference as the body math.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest In-Tree Analog | Upstream/External Port Ref | Match Quality |
|-------------------|------|-----------|------------------------|----------------------------|---------------|
| `crates/pyscf-grad/Cargo.toml` | config | — | `crates/pyscf-ccsd/Cargo.toml` | — | exact |
| `crates/pyscf-grad/src/lib.rs` | model (re-export hub) | — | `crates/pyscf-ccsd/src/lib.rs` | `pyscf/grad/rhf.py:265` `GradientsBase` | exact |
| `crates/pyscf-grad/src/error.rs` | utility | — | `crates/pyscf-ccsd/src/error.rs` | — | exact |
| `crates/pyscf-grad/src/reference.rs` (or reuse `Ccsd`/`Mp2Reference`) | model | — | `crates/pyscf-ccsd/src/reference.rs` | — | exact |
| `crates/pyscf-grad/src/hooks.rs` (`GradOverrideHooks`) | service (trait seam) | event-driven | `crates/pyscf-ccsd/src/hooks.rs` | — | exact |
| `crates/pyscf-grad/src/rhf.rs` | service | transform (HF+Pulay assembly) | `crates/pyscf-mp2/src/mp2.rs` (kernel+default shape) | `pyscf/grad/rhf.py:38-189` | role-match |
| `crates/pyscf-grad/src/uhf.rs` | service | transform | `crates/pyscf-mp2/src/ump2.rs` | `pyscf/grad/uhf.py` | role-match |
| `crates/pyscf-grad/src/rks.rs` | service | transform (grid-weight deriv) | `crates/pyscf-dft` numint + grad rhf.rs | `pyscf/grad/rks.py` | role-match |
| `crates/pyscf-grad/src/uks.rs` | service | transform | `crates/pyscf-grad/src/rks.rs` (sibling) | `pyscf/grad/uks.py` | role-match |
| `crates/pyscf-grad/src/mp2.rs` | service | transform (Z-vector) | `crates/pyscf-mp2/src/rdm.rs` (relaxed dm shape) | `pyscf/grad/mp2.py:268-280` | role-match |
| `crates/pyscf-grad/src/ccsd.rs` | service | transform (Λ + Z-vector) | `crates/pyscf-ccsd/src/rdm.rs` + `lambda.rs` (consumed) | `pyscf/grad/ccsd.py` | role-match |
| `crates/pyscf-grad/src/ecp.rs` | service | transform | grad `rhf.rs` ECP term | `pyscf/grad/rhf.py:109-143` | role-match |
| `crates/pyscf-grad/src/cphf.rs` (D-03) | service | iterative-solve (matrix-free Krylov) | `crates/pyscf-algebra/src/solve_linear.rs` (caller pattern only — NO Krylov exists) | `pyscf/scf/cphf.py:29-148` + `pyscf/lib/linalg_helper.py:1221` | partial (novel solver) |
| `crates/pyscf-grad/src/scanner.rs` (`GradScanner`) | service | request-response | `crates/pyscf-scf/src/scanner.rs` | `pyscf/grad/rhf.py:248-262` | role-match |
| `crates/pyscf-grad/src/verify_fd.rs` (D-01, GRAD-09) | test-harness | request-response | `crates/pyscf-scf/src/scanner.rs` (consumes `as_scanner`) | — (self-validating, novel) | partial |
| `crates/pyscf-geomopt/Cargo.toml` | config | — | `crates/pyscf-ccsd/Cargo.toml` | — | exact |
| `crates/pyscf-geomopt/src/lib.rs` + internals/bmatrix/rfo/backtransform/converge/shims/checkpoint | service | iterative-solve (BFGS+RFO loop) | `crates/pyscf-chkfile` (checkpoint only) | **geomeTRIC (external, fetch — D-06)** | partial (biggest novelty) |
| `crates/pyscf-gto/src/intor.rs` (remove guard 443-447) | route (dispatch) | transform | `evaluate_arity2` component-leading path (same file, lines 216-390) | `pyscf/gto/moleintor.py` | exact |
| `crates/pyscf-gto/src/layout_table.rs` (confirm/extend) | config | — | existing `int1e_ip*`/`int2e_ip1` entries (same file) | `pyscf/gto/moleintor.py:288+` | exact |
| `crates/pyscf-gto/src/ecp_engine_cintx.rs` (wire guard 74-80) | service | transform | `ecp_int1e` scalar arity-2 stitch (same file) | `pyscf/grad/rhf.py:109-143` ECP term | exact |
| `crates/pyscf-py/src/grad.rs` (new) + `geomopt` submodule | controller (PyO3 bridge) | request-response | `crates/pyscf-py/src/cc.rs` | `pyscf/grad/rhf.py` + `pyscf/geomopt/geometric_solver.py` | exact |
| `python/pyscf/grad/__init__.py` + `python/pyscf/geomopt/__init__.py` (new) | config (overlay) | — | `python/pyscf/cc/__init__.py` | — | exact |
| `xtask/src/bin/check_dependency_wall.rs` (NO edit needed — see note) | config (lint) | — | (denylist — already covers all `pyscf-*`) | — | exact |
| `.github/workflows/ci.yml` (new grad/geomopt jobs) | config (CI) | — | `mp2-structural` + `mp2-oracle-upstream-manual` jobs (lines 463-513) | — | exact |

---

## Pattern Assignments

### `crates/pyscf-grad/Cargo.toml` (config)

**Analog:** `crates/pyscf-ccsd/Cargo.toml` (read in full).

Copy the path-dep block verbatim, then trim/add for grad. Key rules baked into the analog's comments:
- Internal crates are `{ path = "../..." }` (NOT `{ workspace = true }` — they are not registered as `[workspace.dependencies]`).
- **NO `pyo3`** (D-07/D-08 — bridge lives in `pyscf-py`). **NO `cubecl-*`** (denylist auto-enforced). **NO `hdf5-metno`** (route through `pyscf_chkfile::hdf5`).
- `thiserror` + `tracing` are `{ workspace = true }`; `approx = "0.5"` in `[dev-dependencies]`.

Per RESEARCH §Standard Stack, `pyscf-grad` deps = `pyscf-{core,algebra,ao2mo,scf,mp2,ccsd,df,grids,gto,chkfile,runtime}` + `pyscf-dft` (numint for KS-grad). The CCSD `Cargo.toml` already lists most; add `pyscf-grids` + `pyscf-dft`.

### `crates/pyscf-grad/src/lib.rs` (model — re-export hub)

**Analog:** `crates/pyscf-ccsd/src/lib.rs:1-64` (the module-skeleton + flat-re-export pattern).
**Upstream ref:** `pyscf/grad/rhf.py:265-418` `GradientsBase` (the trait shape: `mol`/`base`/`unit`/`atmlst`/`de` fields; `kernel`/`grad_elec`/`grad_nuc`/`make_rdm1e`/`hcore_generator`/`get_ovlp`/`get_veff` methods).

The CCSD lib.rs shows the exact convention to copy:
- `#![forbid(unsafe_code)]` + `#![warn(clippy::unwrap_used)]` header (CCSD lib.rs:19-20).
- One `pub mod` per file in the RESEARCH structure (rhf/uhf/rks/uks/mp2/ccsd/ecp/cphf/scanner/verify_fd/error/hooks/reference).
- A flat `pub use module::{...}` re-export block so the bridge imports stay shallow (CCSD lib.rs:40-63).

**Base trait (`Gradients` / `GradientsBase`) — from `pyscf/grad/rhf.py:265-418`:**
```python
class GradientsBase(lib.StreamObject):
    _keys = {'mol', 'base', 'unit', 'atmlst', 'de'}
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        ...
        de = self.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)   # method-specific
        self.de = de + self.grad_nuc(atmlst=atmlst)                # shared
        return self.de
    def grad_nuc(self, mol=None, atmlst=None): return grad_nuc(mol, atmlst)   # shared
    def make_rdm1e(...): raise NotImplementedError              # per-method
    hcore_generator = hcore_generator                           # shared
    def get_ovlp(self, mol=None): return get_ovlp(mol)          # shared (int1e_ipovlp)
```
`atmlst` (GRAD-08) and `verify_fd` (GRAD-09) are base-trait members from day one (D-09) so every method inherits both.

### `crates/pyscf-grad/src/error.rs` (utility)

**Analog:** `crates/pyscf-ccsd/src/error.rs:1-58` (read in full).

Copy verbatim, rename `Ccsd`→`Grad`. The exact shape to reproduce:
```rust
#[derive(Debug, Error)]
pub enum GradError {
    #[error("algebra: {0}")] Algebra(#[from] pyscf_algebra::AlgebraError),
    #[error("core: {0}")]    Core(#[from] pyscf_core::CoreError),
    #[error("ao2mo: {0}")]   Ao2mo(#[from] pyscf_ao2mo::Ao2moError),
    // add: #[from] pyscf_ccsd::CcsdError, pyscf_mp2::Mp2Error (grad consumes both)
    #[error("shape mismatch: expected {expected}, got {got}")] ShapeMismatch { expected: usize, got: usize },
    #[error("grad: not yet implemented (body lands in Wave {wave})")] NotYetImplemented { wave: u8 },
}
impl From<GradError> for pyscf_core::PyscfRsError {
    fn from(e: GradError) -> Self {
        pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!("{}", e)))
    }
}
```
The `Core(InvalidMolecule(..))` bridge arm (no dedicated `PyscfRsError::Grad`) is the locked precedent across mp2/ccsd/ao2mo/scf.

### `crates/pyscf-grad/src/reference.rs` (model) / `hooks.rs` (service seam)

**Analog (reference):** `crates/pyscf-ccsd/src/reference.rs:14-47` — the `CcsdReference { mo_coeff, mo_energy, mo_occ, e_hf, converged, mol }` snapshot + the `UccsdReference { alpha, beta }` two-channel shape. Gradients consume the *same* converged-SCF snapshot, so the planner reuses `CcsdReference`/`Mp2Reference` rather than redefining (the snapshot is identical; only the consuming math differs).

**Analog (hooks):** `crates/pyscf-ccsd/src/hooks.rs:29-97` — the `trait XOverrideHooks { ... }` + `struct NoXOverrides` default-delegation pattern. If gradients expose an override seam (e.g. `get_veff`/`extra_force` for DF-grad discretion), it copies this shape: trait method → `crate::module::default_*` free fn, `NoGradOverrides` delegates.

### `crates/pyscf-grad/src/rhf.rs` (service — HF gradient assembly)

**Analog (structure):** `crates/pyscf-mp2/src/mp2.rs` — a `default_*` free-fn + `Result<_, Error>` kernel module (the pyo3-free method-body file shape).
**Upstream ref:** `pyscf/grad/rhf.py:38-189` (the primary port target).

**Core `grad_elec` decomposition (`pyscf/grad/rhf.py:59-76`):**
```python
hcore_deriv = mf_grad.hcore_generator(mol)
s1   = mf_grad.get_ovlp(mol)            # -int1e_ipovlp  (rhf.py:146)
dm0  = mf.make_rdm1(mo_coeff, mo_occ)
vhf  = mf_grad.get_veff(mol, dm0)       # get_jk over int2e_ip1 (rhf.py:149-183)
dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)   # energy-weighted RDM (rhf.py:185-189)
for k, ia in enumerate(atmlst):
    p0, p1 = aoslices[ia, 2:]
    h1ao = hcore_deriv(ia)
    de[k] += einsum('xij,ij->x', h1ao, dm0)                       # Hellmann-Feynman
    de[k] += einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2     # 2e Pulay
    de[k] -= einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2     # overlap Pulay
```
**Bit-exact discipline (Pitfall 1/2, RESEARCH Pitfall 2):** every `einsum` → materialize-then-`pyscf_algebra::oracle_sum`/`oracle_dot`, NEVER a bare `+=`. All contractions route through `pyscf-algebra` (gemm/gemv) per the algebra wall.

**`make_rdm1e` (energy-weighted RDM, `rhf.py:185-189`)** and **`grad_nuc` (`rhf.py:92-107`)** are thin pure-linear-algebra ports over already-shipped arrays (RESEARCH "Don't Hand-Roll" — these ARE the thin port).

**`hcore_generator` (`rhf.py:121-143`)** uses `int1e_iprinv` with `mol.with_rinv_at_nucleus(atm_id)` — a per-atom origin shift **absent from cintx + pyscf-gto** (RESEARCH Open Question 2). Numeric un-gating waits on the cintx workstream; FD-structural proceeds.

### `crates/pyscf-grad/src/uhf.rs` / `rks.rs` / `uks.rs`

**Analog:** `crates/pyscf-mp2/src/ump2.rs` (spin-resolved α/β shape, file-for-file from its restricted sibling); `crates/pyscf-grad/src/rhf.rs` once written (the sibling within the same crate).
**Upstream ref:** `pyscf/grad/{uhf,rks,uks}.py`.

`rks.rs`/`uks.rs` add the XC-potential derivative + the `grid_response=True` Becke-weight-derivative term (GRAD-03/04), reusing the Phase-4 byte-exact `pyscf-grids` weights + `pyscf-dft` numint (CONTEXT canonical refs). `grid_response` defaults **off** (upstream default), fully supported on request (D-09). **NOT a CPHF consumer** (D-04 / RESEARCH Pitfall 5) — `grid_response` is a grid-weight term, not a response solve.

### `crates/pyscf-grad/src/mp2.rs` / `ccsd.rs` (Z-vector methods)

**Analog (MP2):** `crates/pyscf-mp2/src/rdm.rs` (the relaxed-density assembly shape) + `crates/pyscf-mp2/src/lib.rs` (amplitudes + `as_scanner` it consumes for the Z-vector RHS).
**Analog (CCSD):** `crates/pyscf-ccsd/src/{lambda.rs,rdm.rs}` — **GRAD-06 consumes the Phase-6 `solve_lambda` + `make_rdm1`/`make_rdm2` (incl. `ao_repr`) directly; NO λ re-derivation** (CONTEXT D-04, RESEARCH anti-pattern). The CCSD `reference.rs` doc confirms `make_rdm1`/`make_rdm2` ship with `ao_repr`.
**Upstream ref:** `pyscf/grad/mp2.py:268-280` (MP2 Z-vector) + `pyscf/grad/ccsd.py` (Λ-driven CCSD grad).

**MP2 Z-vector call (`mp2.py:268-280`) — the D-03 CPHF consumer contract:**
```python
def _response_dm1(mp, Xvo):
    def fvind(x): ...                                            # method-specific A-operator
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]   # ONE solver, max_cycle=30
```
**Pitfall 5 (RESEARCH):** MP2 grad overrides `max_cycle=30`; the bare `cphf.solve` default is `max_cycle=50`. Both MP2 and CCSD pass their own `fvind` + RHS into the single `cphf::solve` (D-04).

### `crates/pyscf-grad/src/ecp.rs` (ECP gradient)

**Analog:** the ECP term inside grad `rhf.rs` once written; the cintx dispatch in `crates/pyscf-gto/src/ecp_engine_cintx.rs`.
**Upstream ref:** `pyscf/grad/rhf.py:109-143` (`get_hcore` + `hcore_deriv` ECP branches).

```python
# get_hcore (rhf.py:116-117): + mol.intor('ECPscalar_ipnuc', comp=3)   # cintx ecp_ipnuc — READY (RESEARCH matrix)
# hcore_deriv (rhf.py:139-140): vrinv += mol.intor('ECPscalar_iprinv', comp=3)   # MISSING in cintx (workstream)
```
RESEARCH matrix: `ECPscalar_ipnuc` (=`int1e_ecp_ipnuc`) is in cintx-main today; `ECPscalar_iprinv` is a cintx workstream item. Closes the GTO-05 arc (Phase 2 wired ECP *eval*; Phase 7 wires ECP *gradient*).

### `crates/pyscf-grad/src/cphf.rs` (D-03 — the single CPHF/CPKS solver, GRAD-10)

**Analog (caller shape only):** `crates/pyscf-algebra/src/solve_linear.rs:29` (`solve_linear(a, b, n)` — but this is **dense LU**; NO matrix-free Krylov exists in `pyscf-algebra`, RESEARCH §Standard Stack). The planner BUILDS the Krylov here, routing its gemm/dot through `pyscf-algebra`.
**Upstream ref:** `pyscf/scf/cphf.py:29-148` (`solve`/`solve_nos1`/`solve_withs1`) + `pyscf/lib/linalg_helper.py:1221` (`lib.krylov`, Pople 1979).

**Exact upstream defaults to port (`cphf.py:29-31`):**
```python
def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=50, tol=1e-9, hermi=False, level_shift=0):   # ← the locked constants
    if s1 is None: return solve_nos1(...)    # MP2/CCSD Z-vector (no s1)
    else:          return solve_withs1(...)  # field-dependent (Hessian future, not Phase 7)
```
**Matrix-free `aop` (`cphf.py:73-79`):**
```python
def vind_vo(mo1):
    mo1 = mo1.reshape(-1, nvir, nocc)
    v = fvind(mo1).reshape(-1, nvir, nocc)     # caller-supplied response operator
    if level_shift != 0: v -= mo1 * level_shift
    v *= e_ai
    return v.reshape(-1, nvir*nocc)
mo1 = lib.krylov(vind_vo, mo1base.reshape(-1, nvir*nocc), tol=tol, max_cycle=max_cycle, hermi=hermi)
```
**Anti-pattern (RESEARCH):** never build the dense A-matrix (O(nocc²nvir²) memory) — port the matrix-free form. **GRAD-10 structural test:** a single-CPHF-implementation CI assertion (`single_cphf_impl`).

### `crates/pyscf-grad/src/scanner.rs` (`GradScanner` — the geomopt seam)

**Analog:** `crates/pyscf-scf/src/scanner.rs:1-54` (read in full) — the `Box<dyn Fn(&Mole) -> Result<Energy, PyscfRsError> + Send + Sync>` closure that captures scalar settings + re-runs on a new Mole. The grad scanner returns `(Energy, de)` instead of `Energy`.
**Upstream ref:** `pyscf/grad/rhf.py:248-262` `SCF_GradScanner.__call__`:
```python
def __call__(self, mol_or_geom, **kwargs):
    mol = mol_or_geom if isinstance(mol_or_geom, gto.MoleBase) else self.mol.set_geom_(mol_or_geom, inplace=False)
    self.reset(mol)
    e_tot = self.base(mol)         # as_scanner energy (SCF-12/MP2-07/CCSD-07)
    de    = self.kernel(**kwargs)  # analytical gradient
    return e_tot, de               # ← what the optimizer consumes
```
The SCF `as_scanner` captures conv_tol/diis/etc. by value and re-runs `kernel()` — the grad scanner wraps that energy scanner + appends `.kernel()`.

### `crates/pyscf-grad/src/verify_fd.rs` (D-01, GRAD-09 — the always-on numeric gate)

**Analog:** `crates/pyscf-scf/src/scanner.rs` (consumes the `as_scanner` energy closure it finite-differences).
**Novel:** central-difference, per-atom-per-component, in Bohr, `disp=1e-4`, comparing FD of the shipped `as_scanner` energy to the analytical gradient at **≤1e-6 Ha/Bohr**. No upstream PySCF needed (the first always-on *numeric* gate in the project). This is the daily `cargo test` gate for every GRAD-01..07.

---

### `crates/pyscf-geomopt/*` (the native BFGS+RFO optimizer — biggest novelty)

**In-tree structural analog:** only the checkpoint reuses one (`crates/pyscf-chkfile`'s re-exported `hdf5` alias — see Shared Patterns). The optimizer *algorithm* has **NO in-tree source** (RESEARCH §Summary / Pattern 3): upstream `geomopt` only shims external `geometric`/`pyberny`.
**External port ref (D-06):** geomeTRIC (`github.com/leeping/geomeTRIC` — fetch). Convergence + RFO + B-matrix algorithm.

**Recommended module split (RESEARCH §Recommended Project Structure):**
`lib.rs` (`optimize` entry + `GeometryOptimizer`) / `internals.rs` (bond/angle/dihedral primitives) / `bmatrix.rs` (Wilson B, G=BBᵀ, G⁻) / `rfo.rs` (RFO step + trust-radius Brent + neg-eig shift + BFGS) / `backtransform.rs` (internal→Cartesian) / `converge.rs` (5-criterion GAU check) / `shims.rs` (geometric/berny entry-point parity) / `checkpoint.rs` (HDF5).

**Cargo.toml deps (RESEARCH):** `pyscf-{algebra,scf,grad,chkfile,core,runtime}`. eigh/solve_linear route through `pyscf-algebra` (algebra wall). **NO pyo3, NO cubecl-*, NO own hdf5.**

**Convergence defaults (GEOMOPT-04 — locked by requirements, GAU preset, RESEARCH Pattern 3):**
```
convergence_energy = 1.0e-6  Eh
convergence_grms   = 3.0e-4  Eh/Bohr
convergence_gmax   = 4.5e-4  Eh/Bohr
convergence_drms   = 1.2e-3  (unit: confirm Bohr vs Angstrom at port time — Pitfall 6 / A2)
convergence_dmax   = 1.8e-3  (same unit caveat)
# trust radius: initial trust=0.1, tmax=0.3; default maxsteps=100
```
**RFO + trust step (RESEARCH Pattern 3, CITED geomeTRIC `optimize.py`):**
```
quality = ΔE_actual / ΔE_predicted
if quality > 0.75: trust = min(trust*sqrt(2), tmax)
elif quality > thre: keep
else: trust = max(tmin, min(trust, cnorm)/2)   # shrink + may reject
v0 = (epsilon - Emin) if Emin < epsilon else 0  # neg-eigenvalue shift, epsilon=1e-5
# BFGS approximate Hessian, max_updates=100
```
**Anti-patterns (RESEARCH):** no naive steepest-descent / your-own-RFO; no second optimizer for berny (the `berny_solver` shim is a thin alias over the ONE geomeTRIC engine, D-06); `constraints` kwarg raises a clear error (never silent no-op, D-07 / GEOMOPT-EXT-01 deferred).

**Shim API to mirror (D-07, `pyscf/geomopt/geometric_solver.py:96-192`):**
```python
def kernel(method, ..., constraints=None, callback=None, maxsteps=100, **kwargs):
    if   isinstance(method, GradScanner): g_scanner = method
    elif isinstance(method, GradientsBase): g_scanner = method.as_scanner()
    elif getattr(method, 'nuc_grad_method', None): g_scanner = method.nuc_grad_method().as_scanner()
    ...
    return conv, engine.mol                      # kernel returns (conv, mol)
def optimize(method, ...): return kernel(...)[1] # optimize returns just the optimized Mole
```
`pyscf.geomopt.optimize(mf)` returns the optimized `Mole`; `geometric_solver.optimize` returns the same; `kernel` returns `(conv, mol)` — match these shapes exactly so user scripts run unchanged.

---

### `crates/pyscf-gto/src/intor.rs` (remove arity-4 component-leading guard)

**Analog:** the **same file's** `evaluate_arity2` component-leading repack (lines 216-390) — the arity-2 1e gradient intors (`int1e_ipovlp`/`ipkin`/`ipnuc`/`iprinv`) already dispatch through it. The guard to remove is in `evaluate_arity4` (lines 443-447):
```rust
// int2e is scalar; component-leading arity-4 (gradients) is Phase 7.
if matches!(layout, IntorLayout::ComponentLeadingFOrder { .. }) {
    return Err(PyscfRsError::NotYetImplemented {
        phase: 7,
        what: "component-leading arity-4 integrals (int2e_ip1/ip2 gradients) — Phase 7",
    });
}
```
Wire the `int2e_ip1` component-leading arity-4 path mirroring the arity-2 component-leading stitch (`stitch_block`, lines 320-390, which already handles `block_component_leading` + `components > 1`). **D-02 gating:** `int2e_ip1` is **MISSING from cintx** (RESEARCH matrix) — the guard removal is structural; numeric un-gating waits on the cintx workstream.

> **Note:** CONTEXT cites guards at "intor.rs:94,446" and "ecp_engine_cintx.rs:78". As of this read, `intor.rs:94` is a *doc-comment* describing the ECP-derivative guard (the real guard is in `ecp_engine_cintx.rs:74-80`); `intor.rs:443-447` is the live arity-4 guard. The line numbers drifted; the planner should match on the `NotYetImplemented{phase:7}` literal, not the line number.

### `crates/pyscf-gto/src/layout_table.rs` (confirm/extend)

**Analog:** the **same file's** existing gradient entries (lines 80-128): `int1e_ipovlp_sph`, `int1e_ipkin_sph`, `int1e_ipnuc_sph`, `int1e_iprinv_sph`, `int2e_ip1_sph`, `int2e_ip2_sph`, `int3c2e_ip1_sph` — all already `ComponentLeadingFOrder { components: 3 }`. The table is **already complete** for the core gradient families; the planner confirms and adds only ECP-gradient names if needed (`ECPscalar_ipnuc`/`_iprinv` route through the ECP engine, not this table). The `catalogue_meets_phase_2_floor` test (≥20 entries) must stay green.

### `crates/pyscf-gto/src/ecp_engine_cintx.rs` (wire the ECP-gradient guard)

**Analog:** the **same file's** scalar arity-2 stitch (`ecp_int1e`, lines 51-222 — the per-shell-pair `SessionRequest` → F-order buffer loop). The guard to wire is at lines 74-80:
```rust
if core != "int1e_ecp" && core != "ECPscalar" {
    return Err(PyscfRsError::NotYetImplemented {
        phase: 7,
        what: "ECP derivative/gradient integrals (int1e_ecp_ipnuc/iprinv, ECPscalar_ip* — Phase 7 GRAD-07)",
    });
}
```
Add an `ecp_int1e_ipnuc` (component-leading `[3, nao, nao]`) path resolving `OperatorId::INT1E_ECP_IPNUC_*` (manifest positions 26-29 per the file's doc-comment). **D-02:** `ECPscalar_ipnuc` is READY in cintx; `ECPscalar_iprinv` is a workstream item (gate that arm).

---

### `crates/pyscf-py/src/grad.rs` + `geomopt` submodule (PyO3 bridge)

**Analog:** `crates/pyscf-py/src/cc.rs` (read in full — the closest one-to-one, itself a section-for-section copy of `mp.rs`).
**Upstream ref:** `pyscf/grad/rhf.py` (`Gradients` surface) + `pyscf/geomopt/geometric_solver.py` (shim signatures).

Copy these `cc.rs` mechanisms verbatim (rename `Ccsd`→`Grad`):
- **Eager SCF snapshot** (`cc.rs:82-115` `snapshot_reference`) — pull `mo_coeff`/`mo_energy`/`mo_occ`/`e_tot`/`converged`/`mol` from the Python `mf` into plain Rust arrays at construction (D-09).
- **`is_overridden` MRO check** (`cc.rs:154-174`) — `__qualname__` split against base-class names; missing attr → not overridden; no `__qualname__` → conservatively overridden:
```rust
fn is_overridden(slf: &Py<PyAny>, py: Python<'_>, method: &str, base_classes: &[&str]) -> bool {
    let bound = slf.bind(py);
    match bound.getattr(method) {
        Ok(m) => match m.getattr("__qualname__").and_then(|q| q.extract::<String>()) {
            Ok(qual) => { let class = qual.split('.').next().unwrap_or(""); !base_classes.contains(&class) }
            Err(_) => true,
        },
        Err(_) => false,
    }
}
```
- **Override dispatch** (`cc.rs:202-291`) — if overridden → `slf.bind(py).call_method1("<hook>", args)`; else pure-Rust default under `Python::attach(|py| py.detach(|| ...))` (BIND-05). The `kernel` itself does NOT `py.detach` at the top (hooks re-enter Python).
- **`as_scanner` re-run** (`cc.rs:551-822` `PyCcsdScanner`) — wrap `mf.as_scanner()`, re-run SCF at new geom, re-snapshot, re-run the method kernel under `py.detach`. The grad scanner returns `(e_tot, de)` (a tuple) instead of a scalar.
- **Factory + cross-module graft** — `cc.rs:889-906` `ccsd_factory` (RHF→/UHF→/with_df→ dispatch) + `python/pyscf/cc/__init__.py` `_graft_ccsd_onto_scf()`. For grad: graft `mf.nuc_grad_method()` onto the Rust SCF pyclasses (upstream `scf/hf.py:2484` `def nuc_grad_method`), routing to a `_native.grad` factory.
- **Module registration** (`crates/pyscf-py/src/lib.rs:85-88`) — add a `grad_mod` + `geomopt_mod` `PyModule::new` + `register` + `m.add_submodule` block, mirroring the `cc_mod` block exactly.

### `python/pyscf/grad/__init__.py` + `python/pyscf/geomopt/__init__.py` (NEW overlays)

**Analog:** `python/pyscf/cc/__init__.py` (read in full) — the `from pyscf._native.X import ...` re-export + the `_graft_*_onto_scf()` cross-module dispatch. **NOTE:** no `python/pyscf/grad` or `python/pyscf/geomopt` directory exists yet (only `cc`/`mp`/`scf`/`dft`); these are net-new overlay dirs.

The `cc/__init__.py` `_graft_ccsd_onto_scf` is the exact template for `mf.nuc_grad_method()`:
```python
def _graft_nuc_grad_onto_scf() -> None:
    def _nuc_grad_method(self): return Gradients(self)   # -> _native.grad factory
    _nuc_grad_method.__name__ = "nuc_grad_method"
    _nuc_grad_method.__qualname__ = "SCF.nuc_grad_method"
    from pyscf._native.scf import RHF, UHF, GHF
    for cls in (RHF, UHF, GHF):
        if getattr(cls, "nuc_grad_method", None) is None:
            cls.nuc_grad_method = _nuc_grad_method
```
`geomopt/__init__.py` re-exports `optimize` + the `geometric_solver`/`berny_solver` submodule shims from `_native.geomopt`.

---

### `xtask/src/bin/check_dependency_wall.rs` (NO edit needed)

**Critical correction (RESEARCH §State of the Art "Deprecated"):** CONTEXT says "extend the `algebra_wall` allowlist for `pyscf-grad`/`pyscf-geomopt`". This is **wrong on two counts**:
1. There is **no `xtask/src/lints/algebra_wall.rs`** — the lint is `xtask/src/bin/check_dependency_wall.rs` (read in full).
2. It is a **DENYLIST, not an allowlist** (lines 28-47): `FORBIDDEN_DEPS = [cubecl, cubecl-*]`; `ALLOWED_CRATES = ["pyscf-algebra", "pyscf-runtime", "pyscf-kernels"]` (the carve-out). Every `pyscf-*` crate is auto-checked; `pyscf-grad`/`pyscf-geomopt` need **NO allowlist edit** — they simply must not declare any `cubecl-*` dep. The lint will pass them automatically once they exist.

The planner should plan **zero changes** to this file (the new crates are covered by the `name.starts_with("pyscf-")` auto-scan at line 76).

### `.github/workflows/ci.yml` (new grad/geomopt gate jobs)

**Analog:** the `mp2-structural` (lines 463-475) + `mp2-oracle-upstream-manual` (lines 493-513) job pair — the established "always-on structural + `workflow_dispatch` upstream byte-identity" precedent (also the `ccsd-structural` pair at 532-569). Copy the job shape:
```yaml
  grad-structural:                                    # ← always-on (D-01 FD gate)
    name: grad-structural (always-on verify_fd + atmlst + single-CPHF; no python, no libxc)
    runs-on: ubuntu-latest
    needs: [build-default]
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/setup-sibling-crates
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle --locked -- --test-threads=1

  grad-oracle-upstream-manual:                        # ← workflow_dispatch (≤1e-7 byte-identity)
    if: github.event_name == 'workflow_dispatch'
    ...
      - run: pip install "numpy>=1.26" "pyscf>=2.5"
      - run: cargo test -p pyscf-oracle --features python --locked -- --test-threads=1
```
**Three new gate groups (D-01/D-05, RESEARCH §Wave 0 Gaps):**
1. **Always-on:** `cargo test -p pyscf-grad` (FD `verify_fd` ≤1e-6 + `atmlst` subset + `single_cphf_impl` structural) + `cargo test -p pyscf-geomopt` (H2O→equilibrium self-contained convergence + `bmatrix`/`rfo`/`conv_defaults`).
2. **`workflow_dispatch`:** upstream byte-identity grad (≤1e-7 Ha/Bohr) + geomopt trajectory parity vs `geometric_solver`.
3. **CI (special):** the `pip uninstall geometric pyberny && python -c "import pyscf.geomopt; pyscf.geomopt.optimize(mf)"` no-runtime-dep proof (GEOMOPT-01 — runs in CI per D-05, NOT workflow_dispatch).

**Constraints (user memory + the established precedent):** never `--features libxc` (6h compile freeze); `--test-threads=1` for determinism; `setup-sibling-crates` action provides cintx. Mirror the `if: github.event_name == 'workflow_dispatch'` guard on the upstream arms (default runners have no upstream PySCF).

---

## Shared Patterns

### Method-crate skeleton (apply to: `pyscf-grad`, `pyscf-geomopt`)
**Source:** `crates/pyscf-ccsd/src/lib.rs` + `Cargo.toml`
- `#![forbid(unsafe_code)]` + `#![warn(clippy::unwrap_used)]`.
- One `pub mod` per RESEARCH-structure file + a flat `pub use` re-export block.
- Path deps only (`{ path = "../..." }`); **NO pyo3, NO cubecl-*, NO own hdf5**.
- `thiserror`/`tracing` workspace deps; `approx = "0.5"` dev-dep.

### Error bridge (apply to: `pyscf-grad/src/error.rs`)
**Source:** `crates/pyscf-ccsd/src/error.rs:51-58` — `From<XError> for PyscfRsError` always routes through `Core(InvalidMolecule(format!("{}", e)))`. No dedicated `PyscfRsError::Grad` variant this phase.

### Bit-exact reduction discipline (apply to: every grad/CPHF/RFO contraction)
**Source:** the CCSD lib.rs doc (`crates/pyscf-ccsd/src/lib.rs:9-11`) + RESEARCH Pitfall 2.
Every reduction materializes then `pyscf_algebra::oracle_sum` / `oracle_dot` — **never a bare `+=`**. All matmuls/eigh/linear-solves route through `pyscf-algebra` (gemm/gemv/eigh_gen/solve_linear). Thread-count invariant under `release-oracle` (the `oracle-determinism` matrix job, ci.yml:103-125).
```rust
// crates/pyscf-algebra public surface (the only sanctioned reduction/contraction entry points):
pyscf_algebra::gemm(...)          // crates/pyscf-algebra/src/gemm.rs:11
pyscf_algebra::gemv(...)          // gemv.rs:4
pyscf_algebra::oracle_sum(&[f64]) // oracle.rs:22  — ordered reduction
pyscf_algebra::oracle_dot(a, b)   // oracle.rs:30
pyscf_algebra::eigh_gen(f, s, n)  // eigh_gen.rs:52 — RFO generalized eigen-step
pyscf_algebra::solve_linear(a, b, n) // solve_linear.rs:29 — DENSE LU (NOT the CPHF Krylov)
```

### PyO3 bridge (apply to: `pyscf-py/src/grad.rs` + `geomopt`)
**Source:** `crates/pyscf-py/src/cc.rs` (eager snapshot + `is_overridden` + `call_method1` + `py.detach` + `as_scanner` re-run + factory) and `crates/pyscf-py/src/lib.rs:85-88` (submodule registration). Method crates stay pyo3-free; ALL pyo3 lives here.

### Cross-module graft (apply to: `python/pyscf/grad/__init__.py`, `geomopt/__init__.py`)
**Source:** `python/pyscf/cc/__init__.py` `_graft_ccsd_onto_scf()` — graft `mf.nuc_grad_method()` onto the Rust `_native.scf.{RHF,UHF,GHF}` pyclasses, guarded by `if getattr(cls, attr, None) is None` (subclass override wins).

### HDF5 checkpoint (apply to: `pyscf-geomopt/src/checkpoint.rs`, GEOMOPT-05)
**Source:** `crates/pyscf-chkfile/src/lib.rs:37` — `pub use hdf5_metno as hdf5`. Name `pyscf_chkfile::hdf5::Group` (no own `hdf5-metno` dep — the "sole owner" discipline). Mirror `crates/pyscf-scf/src/chkfile.rs` for the dump/load round-trip shape.

### CI gate shape (apply to: the new ci.yml jobs)
**Source:** `mp2-structural` / `mp2-oracle-upstream-manual` (ci.yml:463-513). Always-on scoped `cargo test -p ...` (no python, no libxc) + `if: github.event_name == 'workflow_dispatch'` upstream byte-identity arm.

---

## No In-Tree Analog (planner uses upstream/external port reference)

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/pyscf-grad/src/cphf.rs` | service | iterative-solve | No matrix-free Krylov exists in `pyscf-algebra` (only dense `solve_linear`). Port `pyscf/scf/cphf.py` + `lib.krylov` (Pople 1979). Caller-shape analog = `solve_linear`. |
| `crates/pyscf-grad/src/verify_fd.rs` | test-harness | request-response | First self-validating numeric gate; no prior in-tree analog. Consumes the `as_scanner` energy closure. |
| `crates/pyscf-geomopt/src/{internals,bmatrix,rfo,backtransform,converge}.rs` | service | iterative-solve | **The biggest novelty.** No in-tree optimizer source (upstream `geomopt` only shims external packages). Port geomeTRIC's algorithm (D-06, external fetch). Only `checkpoint.rs` (chkfile alias) and `shims.rs` (cc.rs/geometric_solver.py shapes) have analogs. |

---

## Cross-cutting cintx-gating finding (RESEARCH D-02 hinge — planner MUST honor)

The structural guard removals (`intor.rs:443-447`, `ecp_engine_cintx.rs:74-80`) are mechanical, but **6 of 8 gradient-integral families are MISSING from cintx** (RESEARCH Availability Matrix): `int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`, `ECPscalar_iprinv` + the `with_rinv_at_nucleus` origin shift. Only `int3c2e_ip1` and `int1e_ecp_ipnuc` (=`ECPscalar_ipnuc`) are ready today.

**Consequence for every grad method body:** the analytical-grad **numeric** rides the FD gate (D-01, always-on) but **upstream byte-identity numeric un-gating waits on a cintx workstream** (analogous to the int2e/d-shell-Rys workstream in user memory). FD-structural + the optimizer-structural gates proceed regardless. The planner must NOT schedule "drop the guards" without a paired cintx-side dependency note.

---

## Metadata

**Analog search scope:** `crates/pyscf-{grad,geomopt,mp2,ccsd,scf,gto,algebra,chkfile,py}/src/`, `python/pyscf/{cc,mp,scf}/`, `xtask/src/bin/`, `.github/workflows/ci.yml`, `pyscf/grad/rhf.py`, `pyscf/scf/cphf.py`, `pyscf/geomopt/{geometric_solver,berny_solver}.py`
**Files scanned:** 22 in-tree + 4 upstream PySCF port targets
**Project instructions:** no root `CLAUDE.md`; no `.claude/skills` or `.agents/skills` directories present
**Pattern extraction date:** 2026-05-26
