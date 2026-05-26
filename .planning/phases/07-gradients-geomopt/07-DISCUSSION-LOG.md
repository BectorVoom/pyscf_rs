# Phase 7: Gradients + Geomopt - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-26
**Phase:** 07-gradients-geomopt
**Areas discussed:** Gradient-integral gating & oracle tiering, CPHF/CPKS single-solver seam, Native optimizer fidelity & reference source, Method coverage sequencing

---

## Gradient-integral gating & oracle tiering

### Q1 — In-tree numeric gating model

| Option | Description | Selected |
|--------|-------------|----------|
| FD self-check always-on; upstream = workflow_dispatch | Every analytical gradient gated in-tree by verify_fd against finite-difference of as_scanner energy (no PySCF needed); upstream byte-identity ≤1e-7 Ha/Bohr is a workflow_dispatch arm | ✓ |
| FD + small-system upstream both always-on | Also assert vs upstream pyscf/grad in-tree for small systems; needs PySCF importable locally (CCSD D-04 notes the sandbox can't) | |

### Q2 — cintx gradient-integral dependency (NotYetImplemented{phase:7} stubs)

| Option | Description | Selected |
|--------|-------------|----------|
| Wave-0 buy-down: wire grad intors first | First wave proves cintx round-trip + extends layout table + drops NotYetImplemented{phase:7}; numeric un-gated like CCSD's int2e; researcher confirms cintx availability | ✓ |
| Externally gated (CCSD D-04 style) if cintx lacks them | Structural/FD-synthetic always-on, numeric on a cintx-landing/human-verify arm | |
| You decide | Researcher confirms availability; planner sequences | |

**User's choice:** FD-always-on gate + Wave-0 grad-intor buy-down.
**Notes:** FD's self-validating property (no upstream dependency) is the key enabler — makes it the daily gate.

---

## CPHF/CPKS single-solver seam (GRAD-10)

### Q1 — Where the single solver lives

| Option | Description | Selected |
|--------|-------------|----------|
| New cphf module in pyscf-grad | GRAD-10 permits "pyscf-grad (or shared module)"; co-locates with consumers; no new crate (workspace already 20) | ✓ |
| Shared module in pyscf-scf | Upstream cphf.py lives under pyscf/scf/; but pushes a grad-only concern into pyscf-scf | |
| New pyscf-response crate | Future-proofs for Hessians; over-engineered for v1 (HESS deferred) | |

### Q2 — Solver shape & consumers

| Option | Description | Selected |
|--------|-------------|----------|
| Generic solve(), method-specific RHS builders | Port scf/cphf.py + ucphf.py as one generic Krylov solver; consumers = MP2 + CCSD Z-vectors; CCSD λ from Phase 6; variational HF/KS grads don't call it | ✓ |
| You decide | Planner ports + wires consumers | |

**User's choice:** New cphf module in pyscf-grad; generic solve() with method-specific RHS.
**Notes:** Corrects ROADMAP SC-3 shorthand — variational HF/KS energy gradients are stationary (2n+1 rule) and do not invoke CPHF.

---

## Native optimizer fidelity & reference source

### Q1 — Acceptance bar & always-on test shape

| Option | Description | Selected |
|--------|-------------|----------|
| Self-contained convergence always-on; upstream = workflow_dispatch | Native optimizer drives a small molecule to known equilibrium (no external geometric/pyberny); trajectory parity vs upstream + the no-runtime-dep proof are CI/workflow_dispatch | ✓ |
| Upstream stationary-point parity always-on | Assert vs upstream geometric_solver in-tree; needs the external package importable — contradicts GEOMOPT-01 | |

### Q2 — Canonical port reference (geomeTRIC source not vendored)

| Option | Description | Selected |
|--------|-------------|----------|
| geomeTRIC as canonical reference | Port internal coords + Wilson B + RFO + 5 convergence defaults from geomeTRIC's published source (researcher fetches); berny shim = thin alias over same engine | ✓ |
| Re-derive from optimization theory | Build from Pulay/Schlegel RFO + Wilson B first principles; trajectories won't track any upstream reference | |
| You decide | Researcher picks; planner targets geomeTRIC defaults for GEOMOPT-04 | |

### Q3 — Shim API fidelity

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror the external call signatures | geometric_solver/berny_solver.optimize accept same kwargs (conv_params/constraints/callback/maxsteps) + return same shapes; constraints accepted-but-unsupported (GEOMOPT-EXT-01 deferred) | ✓ |
| Minimal shim surface | Only optimize(mf) + convergence kwargs; some drop-in scripts break | |

**User's choice:** Self-contained convergence gate; geomeTRIC canonical reference; mirror external signatures.
**Notes:** Upstream PySCF geomopt confirmed to be only a shim over external geometric/pyberny — no in-tree source for the algorithm; this is the phase's biggest novelty.

---

## Method coverage sequencing (MVP wave order)

### Q1 — Sequencing strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Vertical MVP early: RHF-grad → optimizer loop, then broaden | De-risk the from-scratch optimizer soonest by proving pyscf.geomopt.optimize(mf) end-to-end on RHF early, then broaden gradient coverage | ✓ |
| Horizontal: all gradients first, then optimizer | Complete the full gradient layer before the optimizer; matches horizontal-layered convention but defers the highest-risk component | |
| You decide | Planner sets wave structure | |

### Q2 — Gradient method order & base-API features

| Option | Description | Selected |
|--------|-------------|----------|
| RHF→UHF→RKS/UKS→MP2→CCSD→ECP; atmlst + verify_fd in base API | Critical-path order; atmlst (GRAD-08) + verify_fd (GRAD-09) built into the base Gradients API from the RHF wave; grid_response off-by-default but supported | ✓ |
| You decide | Planner orders waves + places atmlst/verify_fd | |

**User's choice:** Vertical-MVP-early sequencing; RHF→UHF→RKS/UKS→MP2→CCSD→ECP with atmlst + verify_fd in base API.
**Notes:** Optimizer's novelty (no vendored source) drives the early-de-risk choice over the project's usual horizontal layering.

---

## Claude's Discretion

- RHF/UHF/RKS/UKS/MP2/CCSD/ECP gradient bodies (port the respective pyscf/grad/*.py).
- CPHF iterative solver internals (Krylov method, preconditioner, max_cycle/conv_tol per scf/cphf.py).
- Redundant-internal-coordinate machinery + Wilson B + RFO + trust-radius + Cartesian fallback (port geomeTRIC).
- Optimizer wave structure + the geomopt HDF5 checkpoint schema.
- PyO3 bridge shape (PyGradients + geomopt submodule + overlays, following mp.rs/cc.rs).
- DF-gradient (int3c2e_ip1) path as a follow-on within the locked decisions.

## Deferred Ideas

- Analytical Hessians + frequencies/IR (HESS-01..03) — v1.x.
- Constrained geometry optimization (GEOMOPT-EXT-01) — v1.x; constraints kwarg accepted-but-unsupported this phase.
- CCSD(T)/TD/CASSCF/MCSCF gradients — out of v1 / separate milestones.
- Transition-state search / IRC / NEB — no v1 REQ.
- Fused cubecl gradient/optimizer kernel — Phase 8.
- Dispersion-correction gradients — no v1 REQ.
