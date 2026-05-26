# Phase 7: Gradients + Geomopt - Research

**Researched:** 2026-05-26
**Domain:** Analytical nuclear gradients (HF/DFT/MP2/CCSD/ECP) + native redundant-internal geometry optimizer (BFGS+RFO), porting PySCF `pyscf/grad/*` + `pyscf/scf/cphf.py` and geomeTRIC's algorithm, over the `pyscf-algebra` wall.
**Confidence:** HIGH (cintx availability matrix, CPHF/grad port targets, geomopt shim API all verified against in-repo source + sibling crate; geomeTRIC algorithm CITED from official docs/source)

## Summary

Phase 7 fills two 5-line stub crates (`pyscf-grad`, `pyscf-geomopt`) and wires the gradient integrals Phases 2–6 deferred as `NotYetImplemented{phase:7}`. The gradient-method *bodies* (`grad_elec`, `hcore_generator`, energy-weighted RDM, `get_veff` derivative, MP2/CCSD Z-vector) are mechanical ports of in-repo PySCF source at `./pyscf/grad/*.py` and `./pyscf/scf/cphf.py` — sibling-crate fidelity, every reduction through `oracle_sum`/`oracle_dot`. The CPHF/CPKS solver is a single matrix-free Krylov port of `cphf.py:solve` into a new `cphf` module inside `pyscf-grad` (D-03), consumed only by the two non-variational methods (MP2, CCSD — D-04). The geometry optimizer is the phase's single biggest novelty: there is **no in-tree optimizer source** (upstream `geomopt` only shims external `geometric`/`pyberny`), so its redundant-internal-coordinate engine, Wilson B-matrix, RFO/trust-radius step, and convergence defaults must be ported from geomeTRIC's published algorithm (D-06).

**The Wave-0 gating answer is the most consequential finding (D-02).** I inspected `~/Documents/workspace/cintx/` family by family. cintx exposes **only two** of the seven gradient-integral families the phase needs: `int3c2e_ip1` (DF gradient, stable, oracle-covered) and `int1e_ecp_ipnuc` = `ECPscalar_ipnuc` (ECP gradient, stable, oracle-covered, Plan 19-07). The five core first-derivative families that *every* HF/DFT/MP2/CCSD gradient depends on — **`int2e_ip1`, `int1e_ipovlp`, `int1e_ipkin`, `int1e_ipnuc`, `int1e_iprinv`** — plus the ECP **`ECPscalar_iprinv`** family, are **absent from the cintx safe-API manifest** (verified: 0 matches as `symbol_name` across the 136-entry manifest). The cubecl `gout_ip1` machinery that exists is scoped to the F12/STG/YP family only, not plain Coulomb. This means the expected "cintx ships them → un-gate like int2e" path does **NOT** hold for the core families: a cintx workstream must land `int2e_ip1` + `int1e_ip{ovlp,kin,nuc,rinv}` + `ECPscalar_iprinv` (plus a rinv-origin-shift parameter) before analytical-grad numeric can ride the FD gate. Wave 0 must front-load this cintx dependency and structure the affected method numeric as CCSD-D-04-style gated arms until cintx lands them — while FD-structural stays always-on regardless (D-01).

**Primary recommendation:** Treat Wave 0 as a **cintx gradient-integral workstream + buy-down**, not a quick guard-removal. Confirm/extend the cintx safe API for the 5 missing core families + `ECPscalar_iprinv` + rinv-origin shift; only `int3c2e_ip1` and `ECPscalar_ipnuc` are ready today. Build the base `Gradients` trait (with `atmlst` + `verify_fd` from day one — D-09), then RHF-grad → native optimizer loop early (D-08), CPHF as one Krylov module (D-03/D-04), then broaden. The optimizer is fetch-and-port from geomeTRIC; convergence defaults are the GAU preset (exact values below).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Gradient integral evaluation (`int*_ip*`) | cintx (sibling crate) | `pyscf-gto` dispatch | cintx owns all integral kernels; pyscf-gto only dispatches names + repacks layout |
| Gradient-integral name dispatch + layout | `pyscf-gto` (`intor.rs`, `layout_table.rs`, `ecp_engine_cintx.rs`) | — | Component-leading layout repack + `NotYetImplemented` guard removal lives here |
| Gradient method bodies (RHF/UHF/RKS/UKS/MP2/CCSD/ECP) | `pyscf-grad` | `pyscf-mp2`/`pyscf-ccsd`/`pyscf-grids` (consumed) | Hellmann-Feynman + Pulay assembly; pyo3-free |
| Tensor contractions (every grad einsum, CPHF aop, RFO eigen-step) | `pyscf-algebra` | — | Algebra wall: gemm/gemv/oracle_sum/oracle_dot/eigh/solve_linear; bit-exact |
| CPHF/CPKS Krylov solve (one solver) | `pyscf-grad::cphf` | `pyscf-algebra` (gemm/dot) | GRAD-10 single solver; orbital-response is a gradient concern (D-03) |
| Energy scanner (Mole→energy) | `pyscf-scf`/`pyscf-mp2`/`pyscf-ccsd` `as_scanner` | — | Already shipped (SCF-12/MP2-07/CCSD-07); optimizer drives it |
| Gradient scanner (Mole→(E, grad)) | `pyscf-grad` (`GradScanner`) | scanner above | New seam: wraps energy scanner + `.kernel()` |
| Geometry optimizer (BFGS+RFO, redundant internals, B-matrix) | `pyscf-geomopt` | `pyscf-algebra` (eigh/solve_linear) | Native engine; no in-tree source — port geomeTRIC; pyo3-free |
| HDF5 optimizer-state checkpoint | `pyscf-geomopt` | `pyscf-chkfile` (`hdf5` alias) | Reuse existing alias, no new dep (GEOMOPT-05) |
| Python `mf.nuc_grad_method()` / `geomopt.optimize` graft | `pyscf-py` (`grad.rs`, `geomopt` submodule) + `python/pyscf/{grad,geomopt}` overlay | — | PyO3 bridge; method crates stay pyo3-free (D-07) |
| Backend dispatch (CPU/GPU) | `pyscf-algebra` only | — | grad/geomopt NEVER touch `cubecl-*` directly (dependency wall) |

## Standard Stack

This is a **pure-Rust port over already-shipped workspace crates** — **no new external packages are installed.** The "stack" is the existing workspace dependency set plus the sibling cintx crate.

### Core (workspace crates consumed)
| Crate | Role in Phase 7 | Status | Notes |
|-------|-----------------|--------|-------|
| `pyscf-algebra` | All grad contractions, CPHF `aop`, RFO eigen-step | Shipped | `gemm`/`gemv`/`oracle_sum`/`oracle_dot`/`eigh_gen`/`solve_linear` present. **No Krylov/iterative solver exists — must be built in `pyscf-grad::cphf`.** [VERIFIED: grep `crates/pyscf-algebra/src/`] |
| `pyscf-gto` | Gradient intor dispatch + layout repack + ECP-grad names | Stubs to fill | `int1e_ip*`/`int2e_ip1`/`int3c2e_ip1` layout entries declared; dispatch guards at `intor.rs:94,446`; ECP-grad rejection at `ecp_engine_cintx.rs:78` [VERIFIED: read source] |
| `pyscf-scf` | `as_scanner` (SCF-12), `canonicalize_signs` (SCF-13) | Shipped | Energy seam + vendor-stable `mo_coeff` gradients inherit |
| `pyscf-mp2` | MP2 amplitudes + `as_scanner` (MP2-07) | Shipped | MP2-grad Z-vector RHS source |
| `pyscf-ccsd` | `solve_lambda`, `make_rdm1`/`make_rdm2` (incl. `ao_repr`), `as_scanner` | Shipped (Phase 6) | GRAD-06 consumes directly; **no λ re-derivation** [VERIFIED: `lambda.rs:411`, `rdm.rs:202,296`] |
| `pyscf-grids` + `pyscf-dft` numint | Byte-exact Becke weights for RKS/UKS-grad `grid_response` | Shipped (Phase 4) | weight-derivative term |
| `pyscf-chkfile` | `hdf5` re-export alias | Shipped | GEOMOPT-05 optimizer-state checkpoint, no new dep |
| `pyscf-diis` | `solve_linear` analog | Shipped | B-matrix/linear-solve reference for CPHF |
| `pyscf-py` | PyO3 bridge (`grad.rs` + `geomopt` submodule) | New surface | Mirrors `mp.rs`/`cc.rs` pattern [VERIFIED: read `cc.rs`] |

### Sibling crate (the gating dependency)
| Crate | Role | Availability for Phase 7 |
|-------|------|--------------------------|
| `cintx` (`~/Documents/workspace/cintx/`) | All gradient integral kernels | **PARTIAL — see Gradient-Integral Availability Matrix below.** Only `int3c2e_ip1` + `int1e_ecp_ipnuc` ready; 6 families missing. [VERIFIED: manifest + kernel inspection] |

### Gradient-Integral Availability Matrix (D-02 gating hinge — THE key Wave-0 finding)

| Family | PySCF call site | Needed by | cintx safe-API status | Disposition |
|--------|-----------------|-----------|----------------------|-------------|
| `int3c2e_ip1` | DF gradient (`grad/df.py`) | DF-grad (discretion follow-on) | **STABLE, oracle-covered, base profile** (manifest pos 336/353) | Ready — un-gate now |
| `int1e_ecp_ipnuc` (=`ECPscalar_ipnuc`) | `get_hcore` (rhf.py:117) | ECP-grad (GRAD-07) | **STABLE, oracle-covered, base profile, `ecp_ipnuc` op, component_rank=3** (manifest pos 488/505; kernel `ecp.rs:1009` Plan 19-07) | Ready — un-gate now |
| `int2e_ip1` | `get_jk` (rhf.py:172) | RHF/UHF/RKS/UKS/MP2/CCSD grad (ALL) | **MISSING** — 0 `symbol_name` matches; `gout_ip1` exists but is scoped to F12/STG/YP only (`f12.rs:1329`) | **cintx workstream required** |
| `int1e_ipovlp` | `get_ovlp` (rhf.py:146) | ALL grad (Pulay term) | **MISSING** — 0 matches | cintx workstream required |
| `int1e_ipkin` | `get_hcore` (rhf.py:111) | ALL grad (hcore deriv) | **MISSING** — 0 matches | cintx workstream required |
| `int1e_ipnuc` | `get_hcore` (rhf.py:115) | ALL grad (hcore deriv) | **MISSING** — 0 matches | cintx workstream required |
| `int1e_iprinv` | `hcore_generator` (rhf.py:137) | ALL grad (per-atom HF force) | **MISSING** — 0 matches; **also needs rinv-origin shift** (`with_rinv_at_nucleus`), absent from cintx safe API | cintx workstream required |
| `ECPscalar_iprinv` | `hcore_generator` (rhf.py:140) | ECP-grad (GRAD-07) | **MISSING** — only `ecp_ipnuc` in manifest, not `ecp_iprinv` | cintx workstream required |

**Interpretation for the planner:** The CONTEXT D-02 "expected case (mirrors int2e landing → un-gate)" does **not** hold for the 5 core families + `ECPscalar_iprinv`. The realistic Wave-0 shape is a **cintx gradient-integral workstream** (analogous to the int2e/d-shell-Rys workstream tracked in user memory `project_int2e_general_contraction_broken.md`) that must land in cintx-main before in-tree analytical-grad numeric can ride the FD gate. Until then:
- FD-structural (`verify_fd` wiring + harness) is always-on (D-01) and does not depend on cintx numeric.
- Each affected method's numeric correctness is a CCSD-D-04-style gated/`workflow_dispatch` arm.
- `int3c2e_ip1` and `int1e_ecp_ipnuc` numeric can un-gate immediately.

This reframes Wave 0 from "drop the guards" to "front-load the cintx dependency + drop the guards only for the two ready families." Confirm the cintx workstream's branch/timeline before locking the wave plan.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Native Rust BFGS+RFO optimizer | Python `geometric`/`pyberny` runtime | **Forbidden by GEOMOPT-01** — the no-runtime-dep proof (`pip uninstall geometric pyberny`) is a CI gate |
| Krylov CPHF in `pyscf-grad::cphf` | Dense `solve_linear` building full A-matrix | Dense A is O(nocc²nvir²) memory; PySCF uses matrix-free `lib.krylov` — port that (D-04) |
| New `pyscf-grad::cphf` module | New 21st crate for CPHF | Rejected by D-03 — no crate for one solver |
| Numeric/finite-diff gradients | Analytical | FD is the *verifier* (D-01), not the product; analytical is the deliverable |

**Installation:** None. No `cargo add`. `pyscf-grad`/`pyscf-geomopt` add intra-workspace path deps only (`pyscf-algebra`, gto, scf, mp2, ccsd, df, grids, chkfile, core, runtime). **No `cubecl-*`, no `pyo3`, no `libxc_rs`.**

**Version verification:** N/A — no external registry packages. Workspace crates are path deps pinned by the workspace `Cargo.toml`.

## Package Legitimacy Audit

> No external packages are installed in this phase. It is a pure-Rust port over existing intra-workspace path dependencies plus the sibling `cintx` crate (also a local path/sibling, not a registry package). slopcheck / npm / PyPI / crates.io registry verification is **not applicable**.

| Package | Registry | Disposition |
|---------|----------|-------------|
| (none) | — | Phase adds only intra-workspace path deps + sibling cintx; zero new registry packages |

**Packages removed due to slopcheck [SLOP] verdict:** none (no external packages)
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                         pyscf.geomopt.optimize(mf)  /  geometric_solver.optimize  /  berny_solver.optimize
                                              │  (PyO3 bridge, pyscf-py; shims mirror upstream call sigs, D-07)
                                              ▼
                       ┌──────────────────────────────────────────────────────┐
                       │  pyscf-geomopt: native BFGS+RFO engine (D-06)          │
                       │  ┌────────────────────────────────────────────────┐  │
   new geometry  ─────►│  │ 1. build redundant internals (bond/angle/dihed) │  │
   (Cartesian)         │  │ 2. Wilson B-matrix + G=BBᵀ + G⁻ (pseudo-inverse)│  │
                       │  │ 3. RFO step in internals + trust-radius (Brent) │  │
                       │  │ 4. negative-eigenvalue shift v0; BFGS H update  │  │
                       │  │ 5. internal→Cartesian back-transform (iterate)  │  │
                       │  │ 6. 5-criterion convergence check (GAU preset)   │  │
                       │  └───────────────┬─────────────────▲──────────────┘  │
                       │   HDF5 checkpoint │ (pyscf-chkfile) │ (E, grad)        │
                       └───────────────────┼─────────────────┼─────────────────┘
                              set_geom_(x)  ▼                 │ GradScanner: (E, de)
                                  ┌─────────────────────────────────────────┐
                                  │ as_scanner (SCF/MP2/CCSD) → e_tot         │  energy seam (shipped)
                                  │ + Gradients.kernel() → de                 │  gradient seam (NEW)
                                  └────────────────────┬──────────────────────┘
                                                       ▼
   ┌───────────────────────────────────────────────────────────────────────────────────┐
   │  pyscf-grad: base Gradients trait (atmlst + verify_fd from day one, D-09)           │
   │     grad_elec = hcore_deriv·dm0  +  2·veff_deriv·dm0  −  2·s1·dme0  +  grad_nuc       │
   │       │              │                    │                  │            │           │
   │  variational HF/KS (no CPHF)         non-variational (MP2, CCSD)         ECP term     │
   │       │                                   │ Z-vector RHS                  │           │
   │       │                                   ▼                               │           │
   │       │                          cphf module (ONE Krylov solver, D-03)    │           │
   │       │                          (1+A)z = b   matrix-free aop, tol 1e-9   │           │
   │       └──────────────┬────────────────────┴───────────────┬──────────────┘           │
   └──────────────────────┼────────────────────────────────────┼──────────────────────────┘
              every einsum │ (oracle_sum/oracle_dot)            │ gradient integrals
                           ▼                                    ▼
                 ┌──────────────────┐         ┌────────────────────────────────────────┐
                 │  pyscf-algebra   │         │  pyscf-gto intor dispatch → cintx        │
                 │  gemm/gemv/eigh/ │         │  int2e_ip1, int1e_ip{ovlp,kin,nuc,rinv}, │
                 │  solve_linear    │         │  int3c2e_ip1, ECPscalar_ip*  (Wave 0)    │
                 │  (algebra wall)  │         │  ⚠ 6/8 families need a cintx workstream  │
                 └──────────────────┘         └────────────────────────────────────────┘
```

### Recommended Project Structure
```
crates/pyscf-grad/src/
├── lib.rs            # Gradients base trait (mol/base/atmlst/de/unit) + re-exports
├── rhf.rs            # grad_elec, hcore_generator, get_ovlp, get_jk/get_veff, make_rdm1e, grad_nuc
├── uhf.rs            # UHF spin-resolved grad
├── rks.rs            # KS XC-potential derivative + grid_response weight-deriv term
├── uks.rs            # UKS
├── mp2.rs            # relaxed-density Lagrangian + Z-vector (calls cphf)
├── ccsd.rs           # consumes Phase-6 λ + RDMs + orbital-relaxation Z-vector (calls cphf)
├── ecp.rs            # ECP gradient term (ECPscalar_ipnuc/iprinv)
├── cphf.rs           # ONE Krylov CPHF/CPKS solver (D-03), generic aop+RHS contract (D-04)
├── scanner.rs        # GradScanner: Mole → (e_tot, de)
└── verify_fd.rs      # central-difference FD self-verification (D-01, GRAD-09)

crates/pyscf-geomopt/src/
├── lib.rs            # optimize() entry + GeometryOptimizer
├── internals.rs      # bond/angle/dihedral primitive generation + connectivity graph
├── bmatrix.rs        # Wilson B-matrix, G=BBᵀ, G⁻ pseudo-inverse
├── rfo.rs            # RFO step, trust-radius (Brent), neg-eigenvalue shift, BFGS update
├── backtransform.rs  # internal→Cartesian iteration (newCartesian analog)
├── converge.rs       # 5-criterion GAU convergence check + defaults
├── shims.rs          # geometric_solver / berny_solver entry-point parity
└── checkpoint.rs     # HDF5 optimizer-state (GEOMOPT-05, pyscf-chkfile alias)
```

### Pattern 1: RHF `grad_elec` decomposition (Hellmann-Feynman + Pulay)
**What:** The electronic gradient is `Σ hcore_deriv·dm0 + 2·veff_deriv·dm0 − 2·s1·dme0` per atom, plus `grad_nuc`.
**When to use:** Base for every variational method (RHF/UHF/RKS/UKS); MP2/CCSD reuse the structure with a relaxed density.
**Example:**
```python
# Source: ./pyscf/grad/rhf.py:59-76 (in-repo PySCF — port target, sibling-crate fidelity)
for k, ia in enumerate(atmlst):
    p0, p1 = aoslices[ia, 2:]
    h1ao = hcore_deriv(ia)                                  # int1e_ipkin + ipnuc + iprinv(shifted)
    de[k] += numpy.einsum('xij,ij->x', h1ao, dm0)
    de[k] += numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2   # int2e_ip1 contracted
    de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2   # int1e_ipovlp · dme0 (Pulay)
# every einsum → materialize-then-oracle_sum/oracle_dot (Pitfall 1/2); no bare +=
```
Key seams: `hcore_generator` (rhf.py:121-143) uses `int1e_iprinv` with **`with_rinv_at_nucleus(atm_id)`** origin shift — a capability **not yet in pyscf-gto or cintx safe API**. `make_rdm1e` (rhf.py:185-189) is the energy-weighted RDM. `aoslice_by_atom` is a pure derivation from `_bas[:,ATOM_OF]` + `ao_loc_nr` (both present in pyscf-gto) — not a missing integral.

### Pattern 2: ONE matrix-free Krylov CPHF (GRAD-10 / D-03 / D-04)
**What:** Solve `(1+A)z = b` where `A` is applied via a method-supplied `aop(x)` callback; never materialize the dense A.
**When to use:** MP2 Z-vector and CCSD orbital-relaxation Z-vector ONLY. Variational HF/KS gradients are stationary (2n+1 rule) and **never call CPHF** (D-04 corrects ROADMAP SC-3 shorthand).
**Example:**
```python
# Source: ./pyscf/scf/cphf.py:29-83 (solve / solve_nos1) + ./pyscf/lib/linalg_helper.py:1221 (krylov)
def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=50, tol=1e-9, hermi=False, level_shift=0):  # ← exact upstream defaults
    ...
    mo1 = lib.krylov(vind_vo, mo1base.reshape(-1, nvir*nocc),
                     tol=tol, max_cycle=max_cycle, hermi=hermi)   # Pople 1979 non-sym Krylov
```
**MP2 caller overrides `max_cycle=30`** (`./pyscf/grad/mp2.py:280`): `dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]`. `lib.krylov` defaults are `tol=1e-10, max_cycle=30`; the CPHF wrapper passes `tol=1e-9, max_cycle=50` for the `s1` branch. **Planner must build a matrix-free Krylov in `pyscf-grad::cphf` — no Krylov solver exists in `pyscf-algebra` today** (`solve_linear` is dense LU). The A-operator/RHS contract: each consumer (MP2, CCSD) supplies its own `fvind` (response operator) + `h1`/`Xvo` (RHS), routed through `pyscf-algebra` gemm/dot.

### Pattern 3: geomeTRIC RFO + trust-radius step (D-06, no in-tree source)
**What:** Compute full Newton/RFO step in internals; if its Cartesian norm exceeds trust radius, use Brent's method (`trust_step`) to find a reduced step matching the radius. Apply negative-eigenvalue shift `v0` to lift negative Hessian eigenvalues for minimization. Update trust radius from the energy-prediction quality factor.
**When to use:** The native optimizer's inner step.
**Example (algorithm, CITED — port to Rust over `pyscf-algebra` eigh/solve_linear):**
```
# Source: geomeTRIC optimize.py (CITED github.com/leeping/geomeTRIC)
quality = ΔE_actual / ΔE_predicted
if quality > 0.75:   trust = min(trust * sqrt(2), tmax)   # good step grow
elif quality > thre: trust = trust                         # ok, keep
else:                trust = max(tmin, min(trust, cnorm)/2) # poor, shrink + may reject
# neg-eigenvalue handling (minimization):
v0 = (epsilon - Emin) if Emin < epsilon else 0.0   # epsilon default 1e-5; shift lifts neg modes
# BFGS approximate Hessian: update_hessian(..., max_updates=100)
```
**Convergence (5 criteria, ALL must hold — GAU preset defaults):**
| Criterion | conv_params key | Default | Unit |
|-----------|-----------------|---------|------|
| Energy change | `convergence_energy` | 1.0e-6 | Hartree |
| RMS gradient | `convergence_grms` | 3.0e-4 | Hartree/Bohr |
| Max gradient | `convergence_gmax` | 4.5e-4 | Hartree/Bohr |
| RMS displacement | `convergence_drms` | 1.2e-3 | Bohr* |
| Max displacement | `convergence_dmax` | 1.8e-3 | Bohr* |

*Unit discrepancy flagged: geomeTRIC readthedocs documents displacement criteria in **Bohr**; the PySCF `geometric_solver.py` docstring (lines 109-110) labels the same `1.2e-3`/`1.8e-3` values **Angstrom**. The geomeTRIC engine internally overrides `ang2bohr`/`bohr2ang` (geometric_solver.py:44-46), so the operative numeric values are as tabled; the planner must confirm the internal unit at port time against geomeTRIC source. Trust-radius defaults: initial `trust=0.1`, `tmax=0.3` (Angstrom in CLI; converted internally). Default `maxsteps=100`. [CITED: geometric.readthedocs.io/options + geomeTRIC optimize.py]

### Anti-Patterns to Avoid
- **Calling CPHF for variational HF/KS gradients:** Wrong (D-04). RHF/UHF/RKS/UKS energy gradients are stationary; only MP2/CCSD Z-vectors need CPHF. `grid_response` is a grid-weight-derivative term, not a response solve.
- **Re-deriving CCSD λ in Phase 7:** Wrong. Phase 6 shipped `solve_lambda` + `make_rdm1`/`make_rdm2` (incl. `ao_repr`). GRAD-06 *consumes* them; only the orbital-relaxation Z-vector re-enters CPHF.
- **Building the dense CPHF A-matrix:** O(nocc²nvir²) memory; PySCF is matrix-free (`lib.krylov`). Port the matrix-free form.
- **Adding `cubecl-*` or `pyo3` to `pyscf-grad`/`pyscf-geomopt`:** Violates the dependency wall + pyo3-free-method-crate rule. Contractions route through `pyscf-algebra`; PyO3 lives in `pyscf-py`.
- **Bare `+=` in any gradient/CPHF/RFO reduction:** Violates bit-exact discipline (Pitfall 1/2). Materialize-then-`oracle_sum`/`oracle_dot`.
- **A second optimizer for berny:** `berny_solver.optimize` is a thin alias over the ONE geomeTRIC-derived engine (D-06), not a separate port.
- **Silent no-op on `constraints` kwarg:** Must raise a clear error (GEOMOPT-EXT-01 deferred; D-07).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gradient integral kernels | Custom `int2e_ip1`/`int1e_ip*` derivative recurrences | cintx (after the workstream lands them) | Boys-function accuracy + nabla recurrences are cintx's domain (Pitfall 18); hand-rolling re-opens solved numeric |
| Energy-weighted RDM, `aoslice_by_atom` | — | These ARE the thin port (pure linear algebra over existing arrays) | Trivial derivations from shipped `_bas`/`ao_loc_nr`/`mo_coeff` |
| Dense linear solve / eigendecomposition | Custom LU/Jacobi | `pyscf-algebra::solve_linear` / `eigh_gen` | Algebra wall + bit-exact |
| Geometry optimizer step logic | Naive steepest-descent / your own RFO | Port geomeTRIC's RFO+trust+BFGS (D-06) | GEOMOPT-04/06/07 require geomeTRIC-matching defaults + convergence; trajectory cross-check is meaningful only if it matches |
| HDF5 checkpoint I/O | New hdf5 dep | `pyscf-chkfile` `hdf5` alias | GEOMOPT-05 explicitly reuses it; no new dep |
| Matrix-free Krylov CPHF | Dense A-matrix CPHF | Port `lib.krylov` (Pople 1979) into `cphf.rs` | Memory + matches upstream convergence path |

**Key insight:** The *integral* layer (cintx) and the *linear-algebra* layer (`pyscf-algebra`) are solved-and-owned elsewhere. Phase 7's real work is the **assembly glue** (grad bodies, Z-vector wiring, optimizer step logic) — mechanical where a PySCF source exists, genuinely novel only for the optimizer (geomeTRIC port).

## Runtime State Inventory

> Phase 7 is **greenfield** (filling two empty stub crates + adding new method bodies/PyO3 surface). It is NOT a rename/refactor/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a renamed string. The only "state" considerations are *new* artifacts this phase creates (HDF5 optimizer checkpoints), not pre-existing state to migrate.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — phase creates new HDF5 optimizer checkpoints (GEOMOPT-05); none pre-exist | None (verified: no rename) |
| Live service config | None | None |
| OS-registered state | None | None |
| Secrets/env vars | `PYSCF_BACKEND` (read-only, existing), `PYSCF_MAX_MEMORY` (existing) — no new vars renamed | None |
| Build artifacts | `pyscf-grad`/`pyscf-geomopt` are empty stubs that compile today; filling them produces new artifacts, none stale | None |

## Common Pitfalls

### Pitfall 1: Assuming cintx ships the core gradient integrals (D-02 trap)
**What goes wrong:** Planner schedules Wave 0 as a quick `NotYetImplemented` guard-removal, then RHF-grad is dead-on-arrival because `int2e_ip1`/`int1e_ip*` return errors.
**Why it happens:** CONTEXT D-02 frames the "expected case" as mirroring the int2e landing. Verification shows 6 of 8 families are absent from the cintx safe-API manifest.
**How to avoid:** Front-load a cintx gradient-integral workstream (the `int2e_ip1` + 4× `int1e_ip*` + `ECPscalar_iprinv` + rinv-origin shift). Gate affected numeric as `workflow_dispatch` until it lands. Un-gate only `int3c2e_ip1` + `int1e_ecp_ipnuc` now.
**Warning signs:** Any wave plan that lists "remove guards" without a corresponding cintx-side task.

### Pitfall 2: Reduction-order drift in gradient contractions (Pitfall 1/2 re-validation)
**What goes wrong:** Thread-count-dependent gradient values; FD-vs-analytical comparison fails at the 7th digit non-deterministically.
**Why it happens:** Bare `+=` in the per-atom `einsum` reductions.
**How to avoid:** Every grad/CPHF/RFO reduction materializes then `oracle_sum`/`oracle_dot`. Build under `release-oracle` for the bit-exact gate.
**Warning signs:** `verify_fd` passing on 1 thread, failing on N.

### Pitfall 3: Eigenvector sign instability across geometry steps (Pitfall 4 re-validation)
**What goes wrong:** The optimizer's `mo_coeff` flips sign between steps, corrupting the gradient/Hessian-update history.
**Why it happens:** Degenerate or near-degenerate MOs.
**How to avoid:** The SCF reference already applies `canonicalize_signs` (SCF-13) — gradients inherit vendor-stable `mo_coeff` for free. Verify the gradient scanner re-uses the canonicalized coefficients each step.
**Warning signs:** BFGS Hessian update producing erratic curvature; trust radius collapsing.

### Pitfall 4: F-order / component-leading layout mismatch (Pitfall 8 — "Phase 7 grad")
**What goes wrong:** A 3-component gradient integral `(3, nao, nao)` gets indexed as `(nao, nao, 3)` or row-major, scrambling x/y/z.
**Why it happens:** Gradient intors are component-leading F-order (axis 0 = component); `layout_table.rs` declares this but the repack in `intor.rs:209-279` must honor it.
**How to avoid:** Reuse the existing `ComponentLeadingFOrder` layout path; assert shape `[3, nao, nao]` on every gradient intor return.
**Warning signs:** Gradient wrong by a permutation of components; FD agrees on one axis only.

### Pitfall 5: CPHF called for the wrong methods / wrong defaults
**What goes wrong:** HF/KS gradients waste a CPHF solve (or worse, get a non-stationary correction), or MP2 uses `max_cycle=50` instead of the upstream-overridden 30.
**Why it happens:** ROADMAP SC-3 shorthand loosely lists RKS-grad among CPHF consumers; D-04 corrects this.
**How to avoid:** CPHF consumers = {MP2, CCSD} only. Use upstream defaults: `cphf.solve` defaults `max_cycle=50, tol=1e-9, level_shift=0`; **MP2 grad overrides `max_cycle=30`** (`mp2.py:280`).
**Warning signs:** A single-CPHF structural test (GRAD-10) passing but RKS-grad slower than expected; MP2 grad off vs upstream.

### Pitfall 6: Geomopt displacement-unit confusion (Bohr vs Angstrom)
**What goes wrong:** Convergence triggers too early/late because `convergence_drms`/`dmax` are applied in the wrong unit.
**Why it happens:** geomeTRIC docs say Bohr; PySCF shim docstring says Angstrom for the same numeric `1.2e-3`/`1.8e-3`.
**How to avoid:** Confirm the operative unit against geomeTRIC source at port time. The engine internally redefines `ang2bohr`/`bohr2ang`; the numeric thresholds in the table are what the engine compares.
**Warning signs:** H2O converging to the wrong bond length, or in too few/many steps vs upstream `workflow_dispatch` arm.

## Code Examples

### ECP gradient hcore term (GRAD-07)
```python
# Source: ./pyscf/grad/rhf.py:109-143 (in-repo)
def get_hcore(mol):
    h = mol.intor('int1e_ipkin', comp=3)
    h += mol.intor('int1e_ipnuc', comp=3)
    if mol.has_ecp():
        h += mol.intor('ECPscalar_ipnuc', comp=3)   # cintx ecp_ipnuc — READY
    return -h
def hcore_deriv(atm_id):                              # per-atom Hellmann-Feynman force
    with mol.with_rinv_at_nucleus(atm_id):            # ⚠ rinv-origin shift NOT in cintx safe API
        vrinv = mol.intor('int1e_iprinv', comp=3)     # ⚠ MISSING in cintx
        vrinv *= -mol.atom_charge(atm_id)
        if with_ecp and atm_id in ecp_atoms:
            vrinv += mol.intor('ECPscalar_iprinv', comp=3)  # ⚠ MISSING (only ecp_ipnuc exists)
    vrinv[:,p0:p1] += h1[:,p0:p1]
    return vrinv + vrinv.transpose(0,2,1)
```

### MP2 Z-vector response (GRAD-05)
```python
# Source: ./pyscf/grad/mp2.py:268-280 (in-repo)
def _response_dm1(mp, Xvo):
    def fvind(x):                      # method-specific response operator (D-04 contract)
        ...
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]   # ONE solver, max_cycle=30
```

### Gradient scanner (the geomopt seam)
```python
# Source: ./pyscf/grad/rhf.py:248-262 (in-repo)
class SCF_GradScanner(lib.GradScanner):
    def __call__(self, mol_or_geom, **kwargs):
        mol = mol_or_geom if isinstance(mol_or_geom, gto.MoleBase) \
              else self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)
        e_tot = self.base(mol)         # as_scanner energy (SCF-12/MP2-07/CCSD-07)
        de = self.kernel(**kwargs)     # analytical gradient
        return e_tot, de              # ← what the optimizer consumes
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| External `geometric`/`pyberny` Python runtime dep | Native Rust BFGS+RFO engine | This phase (GEOMOPT-01) | No-runtime-dep proof is a CI gate; optimizer is self-contained |
| Upstream byte-identity as the always-on numeric gate (Phases 3–6) | FD self-verification (`verify_fd`) as the daily always-on gate; upstream byte-identity → `workflow_dispatch` | This phase (D-01) | First always-on *numeric* gate needing no upstream PySCF |
| Per-method CPHF copies | ONE generic Krylov solver in `pyscf-grad::cphf` | This phase (D-03/GRAD-10) | Structural single-implementation CI assertion |

**Deprecated/outdated:**
- The CONTEXT "extend the algebra_wall allowlist" framing: the actual lint (`xtask/src/bin/check_dependency_wall.rs`) is a **denylist** — `cubecl-*` is forbidden for all crates except 3 carve-outs (`pyscf-algebra`, `pyscf-runtime`, `pyscf-kernels`). `pyscf-grad`/`pyscf-geomopt` need **no allowlist edit** — they simply must not declare `cubecl-*`. [VERIFIED: read lint source]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The cintx workstream for the 6 missing gradient families is feasible/planned (analogous to the int2e/d-shell-Rys workstream) and will land in cintx-main | Availability Matrix | If cintx cannot land them, RHF/UHF/RKS/UKS/MP2/CCSD analytical-grad numeric stays gated indefinitely; FD-structural + the 2 ready families still ship |
| A2 | geomeTRIC's displacement convergence criteria are operatively in Bohr (table values), despite the PySCF shim docstring saying Angstrom | Pattern 3 / Pitfall 6 | Wrong unit → optimizer converges to wrong precision; must confirm against geomeTRIC source at port time |
| A3 | geomeTRIC source is Apache/BSD-compatible to port the algorithm into this repo | Pattern 3 | License conflict would block the port; geomeTRIC is BSD-licensed (commonly), but planner should confirm before vendoring code structure |
| A4 | `aoslice_by_atom` / `offset_nr_by_atom` can be derived in-tree from `_bas[:,ATOM_OF]` + `ao_loc_nr` (both present) without a cintx call | Pattern 1 | Low risk — pure index arithmetic; if a subtlety exists (ECP shells, ghost atoms), the per-atom slicing could be off |
| A5 | A matrix-free Krylov (Pople 1979 / `lib.krylov`) is the right port target for CPHF rather than reusing `pyscf-diis` machinery | Standard Stack / Pattern 2 | If DIIS-style works equally, effort estimate shifts; numeric path should still match upstream |

## Open Questions

1. **What is the cintx gradient-integral workstream's branch and timeline?**
   - What we know: `int3c2e_ip1` + `int1e_ecp_ipnuc` are in cintx-main today; the 6 core families are absent; cintx has an active general-contraction/d-shell-Rys workstream on `fix/general-contraction-nctr-1e` (user memory).
   - What's unclear: Whether the gradient families are scheduled, on which branch, and whether they require the same host+`#[cube]` byte-validation as the int2e fixes.
   - Recommendation: Treat the cintx workstream as a hard Wave-0 dependency. Confirm with the cintx maintainer/branch before locking the wave plan; structure RHF/UHF/RKS/UKS/MP2/CCSD numeric as gated arms (D-01/D-02) keyed to that landing.

2. **Does `int1e_iprinv` need a new cintx safe-API parameter for the rinv-origin shift?**
   - What we know: PySCF uses `mol.with_rinv_at_nucleus(atm_id)` to move the 1/r operator origin per atom (rhf.py:136); cintx safe API has no rinv-origin parameter (verified absent).
   - What's unclear: Whether cintx can accept a per-call origin or whether the per-atom force must be assembled differently.
   - Recommendation: Include the rinv-origin parameter in the cintx workstream scope (A1); flag as part of GRAD-07/the hcore_generator port.

3. **Krylov vs DIIS-style for the CPHF solver?**
   - What we know: PySCF uses matrix-free `lib.krylov` (Pople 1979); the workspace has `solve_linear` (dense) and `pyscf-diis` but no Krylov.
   - What's unclear: Whether porting `lib.krylov` verbatim or adapting `pyscf-diis` gives bit-exact convergence.
   - Recommendation: Port `lib.krylov` (matches upstream convergence path most closely); reserve DIIS as a fallback.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| cintx sibling crate | All gradient integrals | ✓ (partial coverage) | branch `fix/general-contraction-nctr-1e` @ c137b6e | 6/8 grad families need a cintx workstream (no fallback for those) |
| `pyscf-algebra` (gemm/eigh/solve_linear) | Contractions, CPHF, RFO | ✓ | shipped | — |
| `pyscf-ccsd` solve_lambda/RDMs | GRAD-06 | ✓ | shipped Phase 6 | — |
| `pyscf-chkfile` hdf5 alias | GEOMOPT-05 | ✓ | shipped | — |
| Upstream PySCF (oracle source) | Port reference + byte-identity arm | ✓ in repo `./pyscf/` + `.upstream-venv` | — | FD self-verification (D-01) for daily gate |
| geomeTRIC source/algorithm | Optimizer port (D-06) | external (fetch) | github.com/leeping/geomeTRIC | none — must port the algorithm |
| `libxc_rs` | NOT on grad/geomopt dep path | n/a (must stay off) | — | — (forbidden — ~6h compile) |

**Missing dependencies with no fallback:**
- cintx `int2e_ip1`, `int1e_ipovlp/ipkin/ipnuc/iprinv`, `ECPscalar_iprinv`, rinv-origin shift — blocks RHF/UHF/RKS/UKS/MP2/CCSD/ECP analytical-grad numeric until the cintx workstream lands. FD-structural + optimizer-structural still proceed.

**Missing dependencies with fallback:**
- Upstream PySCF runtime in sandbox → FD self-verification is the always-on substitute (D-01).

## Validation Architecture

> `nyquist_validation: true` in config — section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` (per-crate `tests/*.rs`) + `pyscf-oracle` fixtures (`KNOWN_METHODS`, 24 entries today) + pytest (`pytest.ini`, `--import-mode=importlib`) for the Python drop-in arm |
| Config file | `pytest.ini` (Python arm); per-crate `Cargo.toml` `[dev-dependencies]` (Rust arm); `.github/workflows/ci.yml` (gate wiring) |
| Quick run command | `cargo test -p pyscf-grad` / `cargo test -p pyscf-geomopt` (scoped — must NOT pull `libxc_rs`) |
| Full suite command | `cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle` (scoped); upstream byte-identity is `workflow_dispatch` only |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GRAD-01 | RHF analytical grad matches | unit (FD-gated always-on; upstream byte = workflow_dispatch) | `cargo test -p pyscf-grad rhf_verify_fd` | ❌ Wave 0 |
| GRAD-02 | UHF grad | unit (FD) | `cargo test -p pyscf-grad uhf_verify_fd` | ❌ Wave 0 |
| GRAD-03 | RKS grad + grid_response | unit (FD) | `cargo test -p pyscf-grad rks_verify_fd` | ❌ Wave 0 |
| GRAD-04 | UKS grad | unit (FD) | `cargo test -p pyscf-grad uks_verify_fd` | ❌ Wave 0 |
| GRAD-05 | MP2 grad (Z-vector) | unit (FD) | `cargo test -p pyscf-grad mp2_verify_fd` | ❌ Wave 0 |
| GRAD-06 | CCSD grad (Λ + Z-vector) | unit (FD) | `cargo test -p pyscf-grad ccsd_verify_fd` | ❌ Wave 0 |
| GRAD-07 | ECP grad | unit (FD; ecp_ipnuc ready, iprinv gated) | `cargo test -p pyscf-grad ecp_verify_fd` | ❌ Wave 0 |
| GRAD-08 | `atmlst` subsetting returns those rows | unit (structural) | `cargo test -p pyscf-grad atmlst_subset` | ❌ Wave 0 |
| GRAD-09 | `verify_fd(disp=1e-4)` ≤1e-6 Ha/Bohr | the gate itself (D-01) | (the harness; central difference, per-atom-per-component, Bohr) | ❌ Wave 0 |
| GRAD-10 | exactly ONE CPHF implementation | structural (single-impl assertion) | `cargo test -p pyscf-grad single_cphf_impl` | ❌ Wave 0 |
| GEOMOPT-01 | no `geometric`/`pyberny` runtime dep | CI (pip-uninstall proof) | `pip uninstall -y geometric pyberny && python -c "import pyscf.geomopt; pyscf.geomopt.optimize(mf)"` | ❌ Wave 0 |
| GEOMOPT-02/03 | geometric_solver/berny_solver shims delegate | structural (import + call-sig) | `cargo test -p pyscf-geomopt shim_parity` + python smoke | ❌ Wave 0 |
| GEOMOPT-04 | convergence defaults = geomeTRIC GAU | unit (constant assertion) | `cargo test -p pyscf-geomopt conv_defaults` | ❌ Wave 0 |
| GEOMOPT-05 | HDF5 checkpoint resume | unit (round-trip) | `cargo test -p pyscf-geomopt checkpoint_resume` | ❌ Wave 0 |
| GEOMOPT-06 | Wilson B + RFO + neg-eig tracking | unit (B-matrix vs hand-calc; RFO step) | `cargo test -p pyscf-geomopt bmatrix rfo_step` | ❌ Wave 0 |
| GEOMOPT-07 | converge to same stationary point (chem. acc.) | integration (self-contained always-on: H2O→equilibrium; upstream trajectory = workflow_dispatch) | `cargo test -p pyscf-geomopt h2o_equilibrium` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** scoped `cargo test -p pyscf-grad` / `-p pyscf-geomopt` (FD-structural + optimizer-convergence; no libxc, no upstream).
- **Per wave merge:** `cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle` + the dependency-wall + no-FMA lints.
- **Phase gate:** full scoped suite green; `workflow_dispatch` arms (upstream ≤1e-7 byte-identity, trajectory parity, `pip uninstall` proof) run as the human-verify close-out.

### Wave 0 Gaps
- [ ] `crates/pyscf-grad/tests/verify_fd.rs` — the FD harness (GRAD-09, D-01), gates all GRAD-01..07
- [ ] `crates/pyscf-grad/tests/cphf.rs` — single-CPHF structural assertion (GRAD-10)
- [ ] `crates/pyscf-grad/tests/atmlst.rs` — subsetting (GRAD-08)
- [ ] `crates/pyscf-geomopt/tests/h2o_equilibrium.rs` — self-contained convergence gate (D-05/GEOMOPT-07)
- [ ] `crates/pyscf-geomopt/tests/bmatrix.rs` + `rfo.rs` + `conv_defaults.rs` (GEOMOPT-04/06)
- [ ] `crates/pyscf-oracle` grad fixtures (register-but-defer-dispatch, mirrors MP2/CCSD precedent) — `nuc_grad_*` method names; byte-identity arms `#[ignore]`'d / `workflow_dispatch`
- [ ] `.github/workflows/ci.yml` — FD always-on grad gates + self-contained geomopt gate + `workflow_dispatch` upstream/trajectory/pip-uninstall arms
- [ ] **cintx workstream** for the 6 missing gradient families (BLOCKS numeric un-gating; not a pyscf_rs test file but a prerequisite)

## Security Domain

> This is a numerical chemistry library with no authentication, sessions, access control, network input, or secrets handling in this phase. The standard web-app ASVS categories (V2/V3/V4) do not apply. The relevant "security" surface is FFI safety + numerical integrity, already governed by workspace conventions.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | partial | PyO3 boundary: NumPy contiguity (`to_owned()` non-standard layout, BIND-04); `atmlst`/`disp`/`conv_params` bounds-check at the bridge |
| V6 Cryptography | no | — |

### Known Threat Patterns for {Rust numerical lib + PyO3}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Rust panic escaping FFI | Denial of Service | `catch_unwind` at PyO3 boundary (Pitfall 14, Phase 1 lint); `#![forbid(unsafe_code)]` already on both stub crates |
| Malformed NumPy array (non-contiguous / wrong dtype) | Tampering | `to_owned()` on non-standard-layout, C-contiguous outputs (BIND-04 helper) |
| GIL re-entrancy deadlock in scanner callbacks | Denial of Service | `Python::detach` on long compute; callbacks reacquire GIL cleanly (BIND-05) |
| Unbounded `maxsteps`/`max_cycle` from user kwargs | Denial of Service | Default `maxsteps=100`, `max_cycle=50`; cap/validate at bridge |
| `constraints` kwarg silently ignored | Tampering (wrong result) | Raise clear error (D-07, GEOMOPT-EXT-01 deferred) — never a silent no-op |

## Sources

### Primary (HIGH confidence)
- `~/Documents/workspace/cintx/crates/cintx-ops/src/generated/api_manifest.rs` — definitive 136-entry safe-API operator manifest; gradient-family availability matrix [VERIFIED: grep + read]
- `~/Documents/workspace/cintx/crates/cintx-core/src/operator.rs`, `cintx-ops/src/resolver.rs`, `cintx-cubecl/src/kernels/{f12,ecp,one_electron}.rs` — operator IDs, kernel scoping (gout_ip1 = F12 only; ecp_ipnuc = Plan 19-07)
- `./pyscf/grad/rhf.py` — RHF gradient port target (grad_elec, hcore_generator, get_jk/get_veff, make_rdm1e, grad_nuc, GradScanner, GradientsBase) [VERIFIED: read in-repo]
- `./pyscf/grad/mp2.py` — MP2 Z-vector via `cphf.solve(..., max_cycle=30)` [VERIFIED: read]
- `./pyscf/scf/cphf.py` — CPHF solver defaults (max_cycle=50, tol=1e-9, level_shift=0, lib.krylov) [VERIFIED: read]
- `./pyscf/lib/linalg_helper.py:1221` — `lib.krylov` algorithm (Pople 1979) [VERIFIED: read]
- `./pyscf/geomopt/geometric_solver.py` + `berny_solver.py` — shim API surface + convergence-default docstrings [VERIFIED: read]
- `crates/pyscf-{grad,geomopt}/src/lib.rs` + `Cargo.toml` — empty stubs, no deps [VERIFIED: read]
- `crates/pyscf-gto/src/{intor.rs,layout_table.rs,ecp_engine_cintx.rs}` — dispatch guards + layout entries [VERIFIED: read]
- `crates/pyscf-ccsd/src/{lambda.rs,rdm.rs}` — solve_lambda + make_rdm1/2 + ao_repr shipped [VERIFIED: grep]
- `crates/pyscf-algebra/src/` — present primitives; NO Krylov [VERIFIED: ls + grep]
- `xtask/src/bin/check_dependency_wall.rs` — denylist mechanism (cubecl carve-out) [VERIFIED: read]
- `.planning/config.json` — nyquist_validation: true, commit_docs: true [VERIFIED]

### Secondary (MEDIUM-HIGH, CITED official docs)
- geometric.readthedocs.io/en/latest/options.html — convergence defaults (GAU preset), trust radius (0.1/0.3), epsilon (1e-5), Hessian=never [CITED]
- github.com/leeping/geomeTRIC `optimize.py` + `internal.py` — RFO/trust_step/Brent, neg-eigenvalue shift v0, BFGS max_updates=100, primitive coordinate classes, Rotator thresholds [CITED]

### Tertiary (LOW, ecosystem context)
- WebSearch on geomeTRIC/Wang-Song 2016 (TRIC paper) — algorithm provenance only; not load-bearing

## Metadata

**Confidence breakdown:**
- Gradient-integral availability matrix (D-02): HIGH — exhaustively verified against the cintx manifest + kernel source; the central finding.
- Standard stack / port targets: HIGH — all in-repo PySCF source + shipped workspace crates read directly.
- CPHF / Z-vector: HIGH — exact upstream defaults + call sites verified; the only build-vs-port nuance (Krylov absence) is flagged.
- Geomopt algorithm: MEDIUM-HIGH — convergence defaults + trust-radius + neg-eig handling CITED from official docs/source; exact internal back-transform thresholds need confirmation at port time (A2/A3).
- Pitfalls: HIGH — derived from verified findings + ROADMAP Pitfall-to-Phase mapping.

**Research date:** 2026-05-26
**Valid until:** 2026-06-09 (14 days — cintx gradient workstream status is the fast-moving variable; re-verify the availability matrix before Wave 0 locks)
