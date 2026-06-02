# pyscf_rs

## What This Is

A pure-Rust rewrite of PySCF (the Python-based quantum chemistry package) that ships as a Rust core with PyO3 Python bindings, presenting a drop-in `pyscf.*` import surface for existing users. The core compute kernels are written in cubecl so the same source runs on CPU SIMD today and CUDA/WGPU/ROCm GPUs in the same release.

## Core Value

**Run mainstream molecular ground-state quantum chemistry (HF, DFT, MP2, CCSD, gradients) 2–5× faster than current PySCF + C extensions, with bit-exact agreement on regression tests, and zero C/CMake/libcint dependency hell at install time.**

If everything else fails, this must work: a Python user does `pip install pyscf-rs`, runs an existing PySCF script unchanged, gets the same numbers, faster.

## Current Milestone: v2.0 — Periodic Boundary Conditions (PBC)

**Goal:** Deliver the full periodic stack — a drop-in `pyscf.pbc.*` import surface — so existing PySCF solid-state scripts run unchanged on the pure-Rust core: build a crystal `Cell`, sample k-points, and run periodic SCF/DFT/correlation/response with bit-exact agreement to upstream.

**Target features (full `pyscf/pbc/*` parity, per user scoping 2026-06-01):**
- **pbc.gto** — `Cell` construction, lattice vectors, real-space/reciprocal-space lattice sums, GTH pseudopotential consumption (parser already landed), k-point meshes
- **pbc.scf** — gamma-point + k-point HF: KRHF / KUHF / KROHF / KGHF (+ k-point symmetry, smearing, stability, newton)
- **pbc.dft** — KRKS / KUKS / KROKS with periodic grids + XC; multigrid
- **pbc.df** — density fitting: FFTDF, AFTDF, GDF (gdf_builder), MDF, RSDF; AO Fourier transforms (ft_ao)
- **pbc.mp / pbc.cc / pbc.ci / pbc.ao2mo** — periodic KMP2, KCCSD, k-point CI, periodic MO transforms
- **pbc.grad / pbc.geomopt** — periodic nuclear gradients + lattice/cell optimization
- **pbc.tools / pbc.lib / pbc.symm** — k-point sampling, supercell builders, Ewald, k-point symmetry
- **Periodic response & relativistic (per user "everything in pbc/")** — pbc.tdscf / pbc.tddft / pbc.gw / pbc.adc / pbc.eph / pbc.x2c
- **MPI periodic paths** — pbc.mpicc / pbc.mpitools

**Key context / constraints:**
- PBC is, per this project's own prior framing, "essentially a parallel project" — expect a large multi-phase roadmap; v1.0 Phase 8 (perf + distribution) remains incomplete and is being branched away from to start this milestone.
- Periodic integrals require new infrastructure absent from the molecular core: Bloch phase factors, real/reciprocal lattice sums, Ewald summation, and periodic density fitting (FFT/AFT/GDF) — `cintx` is currently molecular-only.
- The molecular Out-of-Scope exclusions for relativistic / response / MPI methods are scoped to *molecular* code; their **periodic variants are in-scope for v2.0** because the user chose full `pbc/` parity.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- [x] **Molecular structure & integrals (gto)** — Mole construction (≥30 attribute floor, 5 atom-input forms, byte-equal `_atm`/`_bas`/`_env`), basis-set loading (5 BasisInput variants × 207 built-in files, Gaussian-94 + NWChem parsers, ECP loading), 1e/2e integrals via `cintx` (intor wrappers + F-order preservation), `eval_gto` AO-on-grid, JSON dumps/loads, `set_geom_`, `copy`. Validated in Phase 2: GTO (GTO-01..11; ECP *evaluation* half closed 2026-05-23 by plan 02-10 — cintx-backed `CintxEcpEngine` wired, in-tree Cu/LANL2DZ `int1e_ecp` gate green; upstream byte-identity pytest is CI/venv-gated).
- [x] **Self-consistent field (scf)** — RHF/UHF/GHF kernels with DIIS Pulay extrapolation, density fitting (RHF.density_fit), chkfile dump/load, all 11 SCF hooks (get_hcore/ovlp/jk/veff/fock/eig/occ/make_rdm1/init_guess/energy_elec/energy_tot), 30-attribute floor, scanner (as_scanner), analyze/mulliken_pop/dip_moment, RHF↔UHF↔GHF conversion. PyO3 bindings (PyRHF/UHF/GHF with subclass-override dispatch via slf.call_method1, abi3-py310 + free-threading features, GIL-release seam, panic-to-exception bridge, NumPy stride policy, PyOnceLock cache). Oracle macro body + CI matrix (linux x86_64 + macos-14 aarch64, python3.13t smoke). Validated in Phase 3: SCF + PyO3 bindings (SCF-01..14, BIND-01/02/04/05/06/07/09, ORACLE-02/08). Code carry-overs now CLOSED: init_guess `minao` (03-13) + `atom`/`huckel` (03-14) ship — all 5 modes return Ok(Density); `mulliken_meta` ships (meta-Löwdin via new `orth_ao`, 03-15; SCF-09 `[~]` partial — conservation invariants hold in-tree, upstream byte-identity is a human-verify item); the `int3c2e_sph`/DF-HF numeric path was closed in the 03-12/03-13 numeric closure. Remaining: 6 CI/Python-toolchain human-verify items (µHartree parity, cross-platform agreement, python3.13t, NumPy stride-fuzz, subclass-override dispatch, h5py chkfile round-trip) tracked in 03-HUMAN-UAT.md.
- [x] **Coupled-cluster singles-doubles (ccsd)** — In-core RCCSD + open-shell UCCSD + amplitude-DIIS + λ-equations + RDMs (incl. `ao_repr`) + AO-direct + DF-CCSD with HDF5 spill + T1/D1/D2 diagnostics + frozen-core, on a `WorkspacePool` tensor-arena (allocate-once `Wabef`, hard `PYSCF_MAX_MEMORY` pre-flight refusal). `pyscf-ccsd` stays pyo3-free; the PyO3 bridge (`mf.CCSD()`/`density_fit().CCSD()`) lives only in `pyscf-py`. Validated in Phase 6 (CCSD-01..11): in-tree numeric headline RCCSD H2/STO-3G `e_corr=-0.020524500477` (≈0.5 µHartree vs reference), bit-identical across thread counts; UCCSD(α=β)=RCCSD; AO-direct=in-core; DF-CCSD converges + RAII spill. Verification `human_needed`: caffeine/cc-pVDZ + DF-spill + λ/RDM byte-identity + python3.13t GIL = workflow_dispatch arms (03-HUMAN-UAT-style) tracked in 06-HUMAN-UAT.md. Caffeine/cc-pVDZ in-core is additionally blocked by an upstream cintx/SCF d-function issue (one-electron Rys cap fixed in cintx 13fe9d3; d-function SCF convergence is a separate tracked workstream item) — not a CCSD defect.

- [x] **Analytical gradients (grad) + Geometry optimization (geomopt)** — analytical-gradient bodies for RHF/UHF/RKS/UKS/MP2/CCSD + ECP (`pyscf-grad`), the single matrix-free Krylov CPHF/CPKS solver (GRAD-10, one impl enforced by a source-scan gate, reused by MP2 & CCSD Z-vector), the always-on central-difference `verify_fd` gate (D-01, GRAD-09), atmlst subsetting (GRAD-08); the native Rust BFGS+RFO optimizer in redundant internals with Wilson B-matrix + oracle-ordered `G⁻` + 5-criterion GAU convergence + HDF5 checkpoint/resume (`pyscf-geomopt`), with `geometric_solver`/`berny_solver` thin-alias shims over the one engine and zero `geometric`/`pyberny` runtime dep (GEOMOPT-01, CI-proven); and the PyO3 bridge exposing `pyscf.grad.*` / `pyscf.geomopt.*` (method crates stay pyo3-free). Validated in Phase 7 (GRAD-01..10, GEOMOPT-01..07; 10/10 plans, verified 5/5 must-haves). **Posture:** structural surface + always-on numerics (FD gate, CPHF-vs-dense, geomopt model-scanner convergence, ECP-ipnuc/DF-grad numerics, GEOMOPT-01 no-runtime-dep proof) are green; the upstream-byte-identity analytical-grad numeric is `workflow_dispatch`-gated because 6 of 8 gradient-integral families (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`, `ECPscalar_iprinv` + `with_rinv_at_nucleus`) are MISSING from cintx (07-01) with no scheduled workstream — un-gates when the cintx grad-intor workstream lands. GRAD-01..07 + GEOMOPT-07 carry `[~]` "structural complete / numeric cintx-gated" in REQUIREMENTS.md.

### Active

<!-- v1 scope. Hypotheses until shipped and validated. -->

- [ ] **Density functional theory (dft)** — RKS, UKS with grid integration, XC evaluation via `libxc_rs` and `xcfun_rs` *(shipped Phase 4; Active→Validated move not yet recorded — pre-existing PROJECT.md drift)*
- [ ] **Møller–Plesset 2nd-order (mp2)** — RMP2 and UMP2, in-core and density-fitted variants *(shipped Phase 5; Active→Validated move not yet recorded — pre-existing PROJECT.md drift)*
- [ ] **PyO3 bindings preserving `pyscf.*` import surface** — existing user scripts run unchanged for in-scope methods *(SCF Phase 3 + grad/geomopt Phase 7 shipped; remaining surface tracked per-phase)*
- [ ] **PySCF-as-oracle CI** — every numerical regression test runs upstream PySCF in the same process and asserts agreement
- [ ] **cubecl backends: CPU SIMD, CUDA, WGPU, ROCm** — single kernel source, runtime backend selection
- [ ] **2–5× speedup vs PySCF on a defined benchmark suite** — RHF/RKS on representative molecule sizes (small organics → small proteins)
- [ ] **Distribution via crates.io and PyPI wheels** — maturin-built wheels for Linux/macOS/Windows

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- **Periodic boundary conditions (`pyscf/pbc/*`)** — ⬆ **PROMOTED to active milestone v2.0** (2026-06-01). Full periodic stack now in scope; see "Current Milestone" above.
- **Molecular relativistic methods (`x2c`, `dhf`)** — Two-/four-component relativistic SCF for *molecules*; needed only for heavy-element work. (Periodic `pbc.x2c` is in v2.0 scope.)
- **Multi-reference methods (`mcscf`, `mcpdft`, `mrpt`)** — CASSCF, NEVPT2, etc.; niche and high implementation cost
- **Molecular excited-state / response methods (`tdscf`, `tddft`, `adc`, `gw`, EOM-CC)** — Entire *molecular* response-theory layer; treat as a separate milestone. (Periodic `pbc.tdscf/tddft/gw/adc` are in v2.0 scope.)
- **Higher-order post-SCF beyond CCSD (CCSD(T), CC3, full-CI, AGF2)** — Defer; CCSD covers the bulk of practical use
- **MPI / multi-node distribution for molecular paths** — cubecl + shared-memory parallelism only for the molecular core; multi-node molecular is a separate concern. (Periodic `pbc.mpicc/mpitools` are in v2.0 scope per full-`pbc/` parity.)
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
| One matrix-free Krylov CPHF/CPKS solver, reused by every response method (07-07) | Avoids N copies of the orbital-response solve; MP2 & CCSD Z-vector + future CPKS all call `cphf::solve`; a source-scan test forbids a second impl | ✓ Phase 7 |
| geomeTRIC algorithm RE-DERIVED, never vendored (07-04) | geomeTRIC ships under a "BSD-3-clause **Non-AI** License" (clause 4 restricts AI-training use); the BFGS+RFO/redundant-internal algorithm was re-derived over the `pyscf-algebra` wall rather than copying source — keeps the repo clean of the restrictive clause | ✓ Phase 7 |
| Structural-complete + numeric-gated when a cross-repo dep lags (07-01) | When cintx ships only 2/8 gradient-integral families, land the full structural surface (dispatch → clean availability error, never `NotYetImplemented`) + all always-on numerics, and gate the upstream-byte-identity numeric behind `workflow_dispatch` until the external workstream lands | ✓ Phase 7 |

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
*Last updated: 2026-05-26 after Phase 7 (gradients + geomopt) completion — grad+geomopt moved Active → Validated (GRAD-01..10, GEOMOPT-01..07; 10/10 plans, verified 5/5 must-haves). 3 Key Decisions logged (one CPHF; geomeTRIC re-derived not vendored under its Non-AI license; structural-complete/numeric-gated posture). Upstream-byte-identity analytical-grad numeric is workflow_dispatch-gated on the unscheduled cintx grad-intor workstream (6/8 families missing, 07-01); GRAD-01..07 + GEOMOPT-07 carry `[~]` in REQUIREMENTS.md. Advisory code-review (07-REVIEW.md) reachable BLOCKERs CR-01/CR-02 fixed; WR-01..07 are cintx-gated follow-ups. NOTE: dft (Phase 4) + mp2 (Phase 5) Active→Validated move still not recorded (pre-existing drift, flagged inline). Next: Phase 8 (GPU enable + oracle hardening + distribution).*

*2026-06-01: Milestone **v2.0 — Periodic Boundary Conditions (PBC)** started. Scope = full `pyscf/pbc/*` parity (all 21 subpackages incl. periodic response/relativistic/MPI variants), per explicit user scoping. PBC moved from Out of Scope → Current Milestone; molecular relativistic/response/MPI exclusions re-scoped to molecular-only. v1.0 Phase 8 remains incomplete (branched away from to start v2.0).*
