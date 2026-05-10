# pyscf_rs

## What This Is

A pure-Rust rewrite of PySCF (the Python-based quantum chemistry package) that ships as a Rust core with PyO3 Python bindings, presenting a drop-in `pyscf.*` import surface for existing users. The core compute kernels are written in cubecl so the same source runs on CPU SIMD today and CUDA/WGPU/ROCm GPUs in the same release.

## Core Value

**Run mainstream molecular ground-state quantum chemistry (HF, DFT, MP2, CCSD, gradients) 2–5× faster than current PySCF + C extensions, with bit-exact agreement on regression tests, and zero C/CMake/libcint dependency hell at install time.**

If everything else fails, this must work: a Python user does `pip install pyscf-rs`, runs an existing PySCF script unchanged, gets the same numbers, faster.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- [x] **Molecular structure & integrals (gto)** — Mole construction (≥30 attribute floor, 5 atom-input forms, byte-equal `_atm`/`_bas`/`_env`), basis-set loading (5 BasisInput variants × 207 built-in files, Gaussian-94 + NWChem parsers, ECP loading), 1e/2e integrals via `cintx` (intor wrappers + F-order preservation), `eval_gto` AO-on-grid, JSON dumps/loads, `set_geom_`, `copy`. Validated in Phase 2: GTO (GTO-01..11; ECP *evaluation* half tracked via 02-10-PLAN.md pending cintx ECP merge).

### Active

<!-- v1 scope. Hypotheses until shipped and validated. -->

- [ ] **Self-consistent field (scf)** — RHF, UHF, GHF with DIIS convergence
- [ ] **Density functional theory (dft)** — RKS, UKS with grid integration, XC evaluation via `libxc_rs` and `xcfun_rs`
- [ ] **Møller–Plesset 2nd-order (mp2)** — RMP2 and UMP2, in-core and density-fitted variants
- [ ] **Coupled-cluster singles-doubles (ccsd)** — RCCSD and UCCSD, ground state only
- [ ] **Analytical gradients (grad)** — for HF, DFT, MP2, CCSD
- [ ] **Geometry optimization (geomopt)** — driver layer over gradients (BFGS or equivalent)
- [ ] **PyO3 bindings preserving `pyscf.*` import surface** — existing user scripts run unchanged for in-scope methods
- [ ] **PySCF-as-oracle CI** — every numerical regression test runs upstream PySCF in the same process and asserts agreement
- [ ] **cubecl backends: CPU SIMD, CUDA, WGPU, ROCm** — single kernel source, runtime backend selection
- [ ] **2–5× speedup vs PySCF on a defined benchmark suite** — RHF/RKS on representative molecule sizes (small organics → small proteins)
- [ ] **Distribution via crates.io and PyPI wheels** — maturin-built wheels for Linux/macOS/Windows

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- **Periodic boundary conditions (`pyscf/pbc/*`)** — Solid-state with k-points is essentially a parallel project; defer to a future milestone
- **Relativistic methods (`x2c`, `dhf`)** — Two-/four-component relativistic SCF; needed only for heavy-element work
- **Multi-reference methods (`mcscf`, `mcpdft`, `mrpt`)** — CASSCF, NEVPT2, etc.; niche and high implementation cost
- **Excited-state / response methods (`tdscf`, `tddft`, `adc`, `gw`, EOM-CC)** — Entire response-theory layer; treat as a separate milestone
- **Higher-order post-SCF beyond CCSD (CCSD(T), CC3, full-CI, AGF2)** — Defer; CCSD covers the bulk of practical use
- **MPI / multi-node distribution** — cubecl + shared-memory parallelism only in v1; multi-node is a separate concern
- **Conda channel publishing** — crates.io and PyPI wheels cover v1 distribution; conda-forge can come later
- **Solvent models, QM/MM, NAC, EPH, localized orbitals** — All defer with the rest of the specialty modules

## Context

- **Sister projects already complete** at `~/Documents/workspace/`:
  - `cintx` — pure-Rust libcint replacement (molecular integral engine), already cubecl-based, workspace pattern: `cintx-{rs,capi,cubecl,oracle,runtime,compat,ops,core}`
  - `libxc_rs` — pure-Rust libxc replacement (DFT exchange-correlation functionals)
  - `xcfun_rs` — pure-Rust xcfun replacement, includes `xcfun-{rs,kernels,gpu,ad,eval,py}` workspace
- **Upstream PySCF source** is checked into this repo at `pyscf/` (Python + C extensions); used as both reference implementation and oracle for tests
- **Codebase map already produced** at `.planning/codebase/` (ARCHITECTURE.md, STACK.md, STRUCTURE.md, CONCERNS.md, CONVENTIONS.md, INTEGRATIONS.md, TESTING.md) — describes the *upstream PySCF* layout, not the new Rust target
- **PySCF's hot paths** are C extensions under `pyscf/lib/{vhf,gto,dft,ao2mo,np_helper}` driven by Python; the rewrite turns these into Rust crates and inverts the relationship (Rust core + PyO3 bindings)
- **Numerical reproducibility matters**: chemistry users compare against published energies to µHartree precision; "bit-exact where possible" is the working bar

## Constraints

- **Tech stack**: Pure Rust; cubecl for SIMD/array compute (same kernels target CPU + CUDA + WGPU + ROCm); PyO3 + maturin for Python bindings; criterion for benches
- **Compute primitive**: cubecl is the *only* compute kernel framework — no rayon/std::simd hand-rolling on hot paths (rayon may appear in cold orchestration code only)
- **Reference dependency**: relies on `cintx`, `libxc_rs`, `xcfun_rs` from the sibling workspace; their pace constrains pyscf_rs
- **Workspace layout**: mirror the sibling-crate convention exactly — top-level façade crate plus `crates/pyscf-{module}-{rs,cubecl,oracle,py,runtime,compat}`
- **API surface**: PyO3 bindings must keep the `pyscf.*` import paths (e.g., `from pyscf import gto, scf, dft`) working for in-scope methods; signatures must accept the same kwargs upstream PySCF accepts
- **Numerical contract**: bit-exact agreement with upstream PySCF where feasible; where not, document the deviation and bound it to chemical accuracy (~1 µHartree in energies)
- **License**: Apache-2.0, matching upstream PySCF (allows test-fixture and reference-code reuse)
- **Team**: solo developer + Claude; phase sizes assume one human reviewer; no parallel-team coordination
- **Distribution**: crates.io + PyPI wheels at v1; conda-forge deferred

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pure Rust core + PyO3 bindings (drop-in `pyscf.*` surface) | Existing PySCF users keep their scripts; Rust users get a native crate | — Pending |
| cubecl as the sole compute kernel framework | Single kernel source targets CPU SIMD + CUDA + WGPU + ROCm; matches sibling crates (`cintx-cubecl`, `xcfun-gpu`) | — Pending |
| All four cubecl backends in v1 (CPU/CUDA/WGPU/ROCm) | Maximum portability from day one; one-time integration cost is paid up front | — Pending |
| PySCF-as-live-oracle in CI | Bit-exact regression checks; eliminates fixture-drift risk; PySCF source is already checked in | — Pending |
| Bit-exact-where-possible numerical contract | Chemistry users diff against published numbers; loose accuracy is a credibility risk | — Pending |
| v1 = molecular ground-state only (HF, DFT, MP2, CCSD, grad, geomopt) | Covers ~80% of practical PySCF use; defers all the niche/specialty modules | — Pending |
| Defer pbc, relativistic, multi-reference, excited states to later milestones | Each is essentially a parallel project; v1 must ship | — Pending |
| Mirror cintx/xcfun_rs workspace layout exactly | Consistency across the four-crate family; muscle memory transfers; tooling reusable | — Pending |
| Apache-2.0 license | Matches upstream PySCF; allows test/code reuse; standard for Rust scientific libs | — Pending |
| 2–5× speedup target vs current PySCF | Justifies the rewrite cost; achievable with cubecl + cache-friendly Rust + no Python overhead in hot paths | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-10 after Phase 2 (GTO) verification — Mole + basis-set loading + intor + eval_gto shipped; ECP-evaluation half deferred via 02-10 pending cintx upstream merge.*
