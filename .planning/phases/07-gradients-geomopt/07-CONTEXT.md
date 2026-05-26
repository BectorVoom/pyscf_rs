# Phase 7: Gradients + Geomopt - Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

A Python user runs `mf.nuc_grad_method().kernel()` for any in-scope method (RHF/UHF/RKS/UKS/MP2/CCSD) — plus ECP gradients — and gets analytical gradients matching upstream to **≤1e-7 Hartree/Bohr**; runs `pyscf.geomopt.optimize(mf)` (or the `geometric_solver`/`berny_solver` drop-in shims) and converges to the same stationary point as upstream within chemical accuracy — **with no Python `geomeTRIC` or `pyberny` runtime dependency**, because the optimizer is a native Rust BFGS+RFO engine in redundant internal coordinates.

Phase 7 fills the two Phase-1 stub crates (`pyscf-grad`, `pyscf-geomopt`, both 5-line placeholders today) and wires the gradient-integral dispatch that Phases 2–6 deliberately left as `NotYetImplemented{phase:7}` in `pyscf-gto`. It is the last critical-path method phase before the closing distribution/GPU phase (Phase 8).

**In scope (17 REQ-IDs):**
- **GRAD-01:** `mf.nuc_grad_method().kernel()` for RHF returns analytical gradients matching upstream.
- **GRAD-02:** UHF gradients match upstream.
- **GRAD-03:** RKS gradients (with `grid_response=True`) match upstream.
- **GRAD-04:** UKS gradients match upstream.
- **GRAD-05:** MP2 gradients via Z-vector / CPHF match upstream.
- **GRAD-06:** CCSD gradients via Λ-equations match upstream.
- **GRAD-07:** ECP gradients match upstream.
- **GRAD-08:** Atom-list subsetting (`grad.kernel(atmlst=[1,2,3])`) returns just those rows.
- **GRAD-09:** A finite-difference verification mode (`grad.verify_fd(disp=1e-4)`) is available and gates unit tests (≤1e-6 Ha/Bohr).
- **GRAD-10:** The CPHF/CPKS solver lives in `pyscf-grad` (or a shared module) and is reused by all method gradients — **one CPHF implementation, not N**.
- **GEOMOPT-01:** `pyscf.geomopt.optimize(mf)` runs a native Rust BFGS+RFO optimizer in redundant internals; no Python `geomeTRIC`/`pyberny` runtime dependency.
- **GEOMOPT-02:** `pyscf.geomopt.geometric_solver.optimize(mf)` is a drop-in shim delegating to the native optimizer.
- **GEOMOPT-03:** `pyscf.geomopt.berny_solver.optimize(mf)` is also a drop-in shim.
- **GEOMOPT-04:** Default convergence thresholds match geomeTRIC defaults (`gradient`, `displacement`, `energy`, `gradient_max`, `displacement_max`).
- **GEOMOPT-05:** HDF5 checkpoint of optimizer state allows resuming a partially-converged optimization.
- **GEOMOPT-06:** Wilson B-matrix construction for redundant internals + RFO step with negative-eigenvalue tracking, ported from upstream/geomeTRIC.
- **GEOMOPT-07:** Optimization trajectories on the test corpus converge to the same stationary point as upstream within chemical accuracy.

**Out of scope:**
- **Analytical Hessians + vibrational frequencies/IR** (`HESS-01..03`, `pyscf/hessian/*`) — v1.x deferred. The optimizer is **gradient-only** (BFGS-built approximate Hessian + RFO model Hessian); no analytical second derivatives. This bounds the optimizer scope.
- **Constrained geometry optimization** (`GEOMOPT-EXT-01`, bond/angle/dihedral constraints) — v1.x deferred. The `constraints` kwarg is **accepted-but-unsupported** (clear error), so drop-in scripts that pass it fail loudly rather than silently mis-optimize.
- **CCSD(T) / TD / CASSCF / MCSCF gradients** (`grad/ccsd_t.py`, `grad/tdrhf.py`, `grad/casscf.py`, …) — out of v1 entirely (PROJECT.md Out of Scope; CCSD(T) is v1.x P1).
- **Transition-state search / IRC / NEB** — not in any v1 REQ; the optimizer targets minima only.
- **Per-backend GPU regression / benchmark proof** — Phase 8 (the 2–5× owner). Phase 7 gradients contract via `pyscf-algebra` chains (algebra wall), not a fused cubecl kernel.
- **DF-specific gradient path as a headline** — `int3c2e_ip1` exists in the layout table; DF-gradient (`grad` over a `density_fit()` reference) is Claude's-discretion follow-on within the locked decisions, not a gated REQ of its own.

</domain>

<decisions>
## Implementation Decisions

### Numeric gating & oracle tiering (the Phase-7 analog of the CCSD D-02/D-04 sequencing)

- **D-01: Finite-difference self-verification is the PRIMARY always-on correctness gate; upstream byte-identity is a `workflow_dispatch`/human-verify arm.** GRAD-09's `verify_fd(disp=1e-4)` is **self-validating** — it finite-differences the already-shipped `as_scanner` energy (SCF-12/MP2-07/CCSD-07) and compares to the analytical gradient, needing **no upstream PySCF in the test environment**. So every analytical gradient is gated in-tree by `verify_fd` to **≤1e-6 Ha/Bohr**, keeping `cargo test` fast and green with zero external dependency. Upstream byte-identity (the **≤1e-7 Ha/Bohr** vs `pyscf/grad/*` success-criterion bar) runs as a `workflow_dispatch`/human-verify arm — the established 02-10 / 05-08 / 06-11 precedent (the sandbox can't run upstream PySCF). FD is the daily gate; upstream is the periodic cross-check.

- **D-02: Wave-0 risk-buy-down wires the gradient intors into `pyscf-gto` first; analytical-grad numeric is then un-gated like CCSD's `int2e`.** The gradient integrals (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`, `int3c2e_ip1`, and ECP `int1e_ecp_ipnuc`/`iprinv`/`ECPscalar_ip*`) are **currently `NotYetImplemented{phase:7}` stubs** in `pyscf-gto` (`intor.rs:94,446`; `ecp_engine_cintx.rs:78`), though the `layout_table.rs` already declares the component-leading layout entries. The **first wave** proves the cintx gradient-integral round-trip (the 02-01 cintx-smoke pattern), extends/confirms the layout table, and **removes the `NotYetImplemented{phase:7}` dispatch guards**. The researcher MUST confirm cintx availability of these integral families up front. If cintx ships them (the expected case — mirrors the `int2e` landing that un-gated CCSD), analytical-grad numeric is un-gated and rides the FD gate (D-01). If cintx lacks a family, that family's numeric falls back to the CCSD-D-04-style gated arm while a cintx workstream lands it — but FD-structural stays always-on regardless.

### CPHF/CPKS single-solver seam (GRAD-10)

- **D-03: ONE generic CPHF/CPKS solver lives in a new `cphf` module inside `pyscf-grad`** — not a new crate, not in `pyscf-scf`. GRAD-10 permits "`pyscf-grad` (or a shared module)"; the orbital-response/Z-vector solve is a gradient concern, every method-gradient module already depends on `pyscf-grad`, and the workspace is already at 20 crates (no 21st for one solver — respects "don't add deps/crates without need"). A single CI/structural test asserts there is exactly one CPHF implementation (GRAD-10 success criterion).

- **D-04: The solver is a generic iterative (Krylov) linear solve over `pyscf-algebra`, with method-specific RHS/A-operator builders; only the non-variational methods consume it.** Port `pyscf/scf/cphf.py:solve` (closed-shell) + `pyscf/scf/ucphf.py` (open-shell) as one generic engine. **Consumers:** MP2-grad Z-vector (GRAD-05) and CCSD-grad orbital-relaxation Z-vector (GRAD-06) pass their own RHS + response operator. **CCSD Λ is already solved in Phase 6** (`solve_lambda`, 06-06) — Phase-7 CCSD-grad *consumes* λ and only re-enters CPHF for the orbital-relaxation Z-vector, it does not re-derive λ. **Variational RHF/UHF/RKS/UKS energy gradients are stationary (the 2n+1 rule) and do NOT call CPHF** — `grid_response` (GRAD-03/04) is a grid-weight-derivative term, not a response solve. This corrects the ROADMAP success-criterion-3 shorthand that loosely listed "RKS-grad with grid_response" among CPHF consumers.

### Native geometry optimizer (the phase's biggest novelty — no vendored source to port)

- **D-05: Self-contained convergence is the always-on geomopt gate; upstream trajectory parity + the no-runtime-dep proof are CI/`workflow_dispatch`.** Upstream PySCF `geomopt` is **only a shim over the external `geometric`/`pyberny` packages** (`geometric_solver.py:24-36`, `berny_solver.py`), and geomeTRIC source is **not vendored** anywhere in the repo — so there is no checked-in PySCF source for the optimizer algorithm, unlike every prior phase. The always-on gate: the native optimizer drives a small molecule (e.g. H2O) to its known equilibrium and asserts the converged geometry + final gradient norm — fully self-contained, no external package importable. Trajectory/stationary-point parity vs upstream `geometric_solver` is a `workflow_dispatch` arm; the **`pip uninstall geometric pyberny && python -c "import pyscf.geomopt; pyscf.geomopt.optimize(mf)"` no-runtime-dep proof (GEOMOPT-01) runs in CI**. Acceptance bar is GEOMOPT-07's "same stationary point within chemical accuracy" — **not** bit-for-bit trajectory parity.

- **D-06: geomeTRIC is the canonical port reference; the `berny_solver` shim is a thin alias over the same native engine.** The researcher fetches geomeTRIC's published source (not in-tree) for the redundant-internal-coordinate setup, Wilson B-matrix, RFO/trust-radius step, negative-eigenvalue tracking (GEOMOPT-06), and the five convergence defaults (GEOMOPT-04). pyberny is a cross-reference for RFO details only. Rationale: pyscf users overwhelmingly use `geometric_solver`, so matching geomeTRIC's defaults makes the `workflow_dispatch` trajectory cross-check meaningful and lets `berny_solver.optimize` delegate to the *same* engine (not a second optimizer). GEOMOPT-04 (geomeTRIC convergence defaults) and GEOMOPT-06 (Wilson-B redundant internals + RFO) are **locked by the requirements** — not gray areas.

- **D-07: The `geometric_solver`/`berny_solver` shims mirror the external libraries' call signatures.** `pyscf.geomopt.geometric_solver.optimize(mf, **conv_params)` and `pyscf.geomopt.berny_solver.optimize(mf)` accept the same kwargs upstream's shims accept (`conv_params` dict, `callback`, `maxsteps`, `constraints`) and return the same shapes (optimized `Mole`; `(conv, mol)` where upstream does) so existing user scripts run unchanged. Both delegate to the one native engine (D-06). The `constraints` kwarg is **accepted-but-unsupported** — a clear error (GEOMOPT-EXT-01 deferred), never a silent no-op. HDF5 optimizer-state checkpoint (GEOMOPT-05) reuses the `pyscf-chkfile` `hdf5` alias (no new dep).

### Sequencing & MVP wave order

- **D-08: Vertical-MVP-early — RHF-grad → native optimizer loop early, then broaden.** The native optimizer is the phase's highest-risk novel component (no vendored source), so de-risk it the soonest: **Wave-0 grad-intor buy-down (D-02) → RHF-grad (headline, FD-gated) → native optimizer on the RHF `as_scanner` loop EARLY** (proves `pyscf.geomopt.optimize(mf)` end-to-end on the thinnest slice) → then **UHF / RKS / UKS gradients → CPHF consolidation → MP2-grad (Z-vector) → CCSD-grad (Λ + Z-vector) → ECP-grad → geomopt re-validated across methods → PyO3 bridge + oracle/CI close-out.** This intentionally departs from the project's otherwise-horizontal layering because the optimizer's from-scratch BFGS+RFO/Wilson-B engine carries the most schedule risk and benefits from earliest end-to-end exercise.

- **D-09: Gradient method order RHF→UHF→RKS/UKS→MP2→CCSD→ECP; `atmlst` subsetting and the `verify_fd` harness are base-API-from-the-start.** Critical-path order: variational HF/KS first (no CPHF), then the Z-vector methods (MP2, CCSD), ECP last. **`atmlst` subsetting (GRAD-08) and the `verify_fd` FD harness (GRAD-09) are built into the base `Gradients` trait/struct on the RHF wave**, so every subsequent method inherits both for free (and FD-gating is available from day one — D-01). `grid_response` defaults **off** (upstream default) but is fully supported (GRAD-03/04).

### Claude's Discretion

Not user-decided — researcher/planner picks within the locked decisions above. Default stance: **mirror upstream / geomeTRIC (sibling-crate fidelity)**.

- **RHF/UHF gradient bodies** — port `pyscf/grad/rhf.py` (`grad_elec`: `hcore_generator`, `get_ovlp` derivative → energy-weighted RDM/`make_rdm1e`, `get_veff` derivative, `grad_nuc`) + `uhf.py`. The Hellmann-Feynman + Pulay decomposition is mechanical; every reduction through `oracle_sum`/`oracle_dot` (no bare `+=`), the established bit-exact discipline.
- **RKS/UKS gradient + `grid_response`** — port `pyscf/grad/rks.py` + `uks.py`: the XC potential derivative on the grid + the `grid_response=True` Becke-weight-derivative term (reuses the Phase-4 `pyscf-grids` byte-exact weights + `numint`). `grid_response` off by default, on when requested.
- **MP2 Z-vector / Lagrangian** — port `pyscf/grad/mp2.py` + `ump2.py`: the relaxed-density Lagrangian, the Z-vector solve through the D-03 CPHF, reusing the Phase-5 MP2 amplitudes + `pyscf-ao2mo`.
- **CCSD gradient** — port `pyscf/grad/ccsd.py` + `uccsd.py`: builds the gradient from the Phase-6 λ + `make_rdm1`/`make_rdm2` (incl. `ao_repr`, 06-06 D-03) + the orbital-relaxation Z-vector through the D-03 CPHF. No λ re-derivation.
- **ECP gradient** — wire `int1e_ecp_ipnuc`/`iprinv` / `ECPscalar_ip*` through the Phase-2 `CintxEcpEngine` (the gradient names currently rejected at `ecp_engine_cintx.rs:78`); port the `pyscf/grad/rhf.py` ECP-gradient term.
- **CPHF iterative solver internals** — Krylov/`solve_linear` method, preconditioner, `max_cycle`/`conv_tol` constants per `scf/cphf.py` defaults; planner confirms exact upstream values.
- **Redundant-internal-coordinate machinery** — bond/angle/dihedral primitive generation, Wilson B-matrix + its pseudo-inverse, internal↔Cartesian back-transformation iteration, RFO step + trust-radius update + negative-eigenvalue handling — all ported from geomeTRIC (D-06); Cartesian fallback for pathological B-matrix conditioning is planner's call.
- **Optimizer wave structure & the geomopt HDF5 checkpoint schema** — planner finalizes (GEOMOPT-05 reuses the `pyscf-chkfile` alias).
- **PyO3 bridge shape** — `PyGradients`/per-method grad classes + the `geomopt` submodule + `python/pyscf/grad` + `python/pyscf/geomopt` overlays, following the Phase-5/6 `mp.rs`/`cc.rs` `as_scanner`-re-run + `call_method1` override-dispatch + `Python::detach` pattern. Method crates stay pyo3-free.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project specs (this repo)
- `.planning/PROJECT.md` — vision, core value (HF/DFT/MP2/CCSD/**gradients** 2–5× faster), key decisions, Out-of-Scope (pbc/relativistic/multi-ref/excited-state/CCSD(T)); `grad`/`geomopt` are Active v1 requirements.
- `.planning/REQUIREMENTS.md` lines **108-117 (GRAD-01..10)** + **121-127 (GEOMOPT-01..07)** + 328-344 (phase mapping) + 215-217/385-387 (**HESS-01..03 v1.x-deferred**) + 226/390 (**GEOMOPT-EXT-01 constrained-opt deferred**) — Phase 7 owns 17 REQs.
- `.planning/ROADMAP.md` §"Phase 7: Gradients + Geomopt" (**lines 307-320**) — goal, dependency (Phase 6 — CCSD grad needs `solve_lambda()`), 5 numbered success criteria (note SC-3's CPHF-consumer list is shorthand; see D-04).
- `.planning/ROADMAP.md` §"Cross-Cutting Concerns" (lines 373-385) — algebra wall, backend selection, bit-exact-with-PySCF, PyO3 subclass-override + NumPy contiguity + `Python::detach` (Phase 7 inherits by convention), scope-creep lint.
- `.planning/ROADMAP.md` §"Pitfall-to-Phase Mapping" (lines 387-411) — Phase 7 **re-validates Pitfall 4 (eigenvector sign, via the SCF reference's `canonicalize_signs`)** and **Pitfall 8 (loop / F-order layout — "Phase 7 grad")**; inherits Pitfall 1/2 (FMA/reduction order in gradient contractions).
- `.planning/STATE.md` — current position (Phase 6 complete, Phase 7 not started); deferred-items context.

### Prior phase context (this repo)
- `.planning/phases/06-ccsd/06-CONTEXT.md` — **CCSD `solve_lambda` + `make_rdm1`/`make_rdm2` incl. `ao_repr` shipped (D-03 there) — GRAD-06 consumes, no re-derivation**; the tensor-arena/`WorkspacePool` (D-08 there) reusable by gradient response tensors; `as_scanner` (CCSD-07) the geomopt seam; the pyo3-free-method-crate + `call_method1` bridge model (D-09 there).
- `.planning/phases/05-mp2/05-CONTEXT.md` — `as_scanner` (MP2-07) the geomopt seam; `pyscf-ao2mo` `general`/`full` transform + MP2 amplitudes (MP2-grad Z-vector RHS); `Mp2OverrideHooks` bridge pattern.
- `.planning/phases/03-scf-pyo3-bindings/03-CONTEXT.md` — **`canonicalize_signs` (SCF-13, vendor-stable `mo_coeff` the gradients inherit)**; `as_scanner` shape (SCF-12); `OverrideHooks` `call_method1` bridge + per-hook `Python::detach`; `pyscf-diis`/`solve_linear` (CPHF B-matrix analog); `pyscf-chkfile` HDF5 schema + alias (GEOMOPT-05 checkpoint); test-corpus tiering.
- `.planning/phases/04-dft/04-CONTEXT.md` — `pyscf-grids` byte-exact Becke weights + `numint` (RKS/UKS-grad `grid_response`); the re-exported `hdf5` alias / no-own-hdf5-metno-dep convention; algebra-orchestrated host-loop precedent.

### Upstream PySCF source (this repo — the oracle / port reference)
- `pyscf/grad/rhf.py` — **primary RHF gradient port target**: `Gradients`/`GradientsBase`, `grad_elec`, `hcore_generator`, `make_rdm1e` (energy-weighted RDM), `grad_nuc`, `as_scanner`/`GradScanner`, `atmlst` handling, ECP-gradient term.
- `pyscf/grad/uhf.py` — UHF gradient (GRAD-02).
- `pyscf/grad/rks.py` + `pyscf/grad/uks.py` — KS gradient + the `grid_response=True` Becke-weight-derivative term (GRAD-03/04).
- `pyscf/grad/mp2.py` + `pyscf/grad/ump2.py` — MP2 relaxed-density Lagrangian + Z-vector via CPHF (GRAD-05).
- `pyscf/grad/ccsd.py` + `pyscf/grad/uccsd.py` — Λ-driven CCSD gradient + orbital-relaxation Z-vector (GRAD-06; consumes Phase-6 λ).
- `pyscf/scf/cphf.py` + `pyscf/scf/ucphf.py` — **the generic iterative CPHF/CPKS solver — the GRAD-10 single-solver port target (D-03/D-04).**
- `pyscf/geomopt/geometric_solver.py` + `pyscf/geomopt/berny_solver.py` + `pyscf/geomopt/addons.py` — **the shim API surface to mirror (call signatures, `as_pyscf_method`, `optimize` return shapes) — NOT the optimizer algorithm** (those shims import the external packages; D-05/D-07).

### External (researcher fetches — NOT vendored in this repo)
- **geomeTRIC published source** (Wang & Song, *J. Chem. Theory Comput.* 2016 — analytic internal-coordinate geometry optimization) — **the canonical port reference for the native optimizer (D-06)**: redundant-internal-coordinate generation, Wilson B-matrix + pseudo-inverse, RFO step + trust-radius + negative-eigenvalue tracking (GEOMOPT-06), and the five convergence defaults (GEOMOPT-04).
- **pyberny** — RFO-step cross-reference only (the `berny_solver` shim is a thin alias over the geomeTRIC-derived engine, not a separate port).
- CPHF / Z-vector theory (Handy–Schaefer Z-vector method) — only if the MP2/CCSD gradient port needs a math cross-check.

### Phase 1–6 shipped artifacts (this repo — fill / extend / consume)
- `crates/pyscf-grad/src/lib.rs` — **the 5-line stub Phase 7 fills** with the `Gradients` base (RHF/UHF/RKS/UKS/MP2/CCSD/ECP), the `cphf` module (D-03), `verify_fd` (D-01), `atmlst` (D-09).
- `crates/pyscf-geomopt/src/lib.rs` — **the 5-line stub Phase 7 fills** with the native BFGS+RFO redundant-internals engine + `optimize` + the `geometric_solver`/`berny_solver` shim entry points + HDF5 checkpoint.
- `crates/pyscf-gto/src/intor.rs` — **the `NotYetImplemented{phase:7}` dispatch guards (lines 94, 446) to remove (D-02)** once the gradient intors are wired.
- `crates/pyscf-gto/src/layout_table.rs` — already declares `int1e_ip{ovlp,kin,nuc,rinv}_sph`, `int2e_ip1_sph`, `int3c2e_ip1_sph` component-leading entries (confirm/extend).
- `crates/pyscf-gto/src/ecp_engine_cintx.rs` — **the ECP-gradient rejection (line 78: `int1e_ecp_ipnuc`/`iprinv`/`ECPscalar_ip*`) to wire (D-02, GRAD-07).**
- `crates/pyscf-scf/src/scanner.rs` (`as_scanner`, SCF-12) + `crates/pyscf-core/src/lib.rs` (`canonicalize_signs`, SCF-13) — the geomopt energy seam + vendor-stable `mo_coeff`.
- `crates/pyscf-mp2/src/lib.rs` — MP2 amplitudes + `as_scanner` (MP2-07) — MP2-grad Z-vector RHS.
- `crates/pyscf-ccsd/src/lib.rs` — `solve_lambda` + `make_rdm1`/`make_rdm2` (incl. `ao_repr`, 06-06) + `as_scanner` (CCSD-07) — **GRAD-06 consumes directly.**
- `crates/pyscf-algebra/src/lib.rs` — `gemm`/`gemv`/`oracle_sum`/`oracle_dot`/`solve_linear`/`eigh` — **every gradient contraction + the CPHF iterative solve + the RFO eigen-step goes through this** (D-03/D-04, algebra wall).
- `crates/pyscf-chkfile/src/lib.rs` — the re-exported `hdf5` alias — **GEOMOPT-05 optimizer-state checkpoint reuses it** (no new dep).
- `crates/pyscf-grids/src/lib.rs` + `crates/pyscf-dft` `numint` — byte-exact Becke weights for RKS/UKS-grad `grid_response`.
- `crates/pyscf-py/src/{scf.rs,mp.rs,cc.rs}` — the `as_scanner`-re-run + `call_method1` override-dispatch + `Python::detach` patterns to mirror for `PyGradients` + the `geomopt` submodule + `python/pyscf/{grad,geomopt}` overlays.
- `xtask/src/lints/algebra_wall.rs` — extend the allowlist for `pyscf-grad` + `pyscf-geomopt` (algebra/gto/scf/mp2/ccsd/df/chkfile deps; **no direct `cubecl-*`**).
- `.github/workflows/ci.yml` — add: FD-`verify_fd` always-on grad gates + self-contained geomopt-convergence always-on gate (D-01/D-05); upstream byte-identity grad + trajectory-parity arms (`workflow_dispatch`); the `pip uninstall geometric pyberny` no-runtime-dep proof (GEOMOPT-01).

### Sibling-crate / PyO3 precedent (read before implementing analogous surface)
- `~/Documents/workspace/cintx/` — **gradient integral source: `int1e_ip{ovlp,kin,nuc,rinv}`, `int2e_ip1`, `int3c2e_ip1`, ECP `ip*` — researcher confirms availability up front (D-02 gating depends on it).**
- `~/Documents/workspace/cintx/crates/cintx-rs/` + `~/Documents/workspace/xcfun_rs/crates/xcfun-py/` — PyO3 0.28 `#[pyclass]`/`#[pymethods]` + `Python::detach` + NumPy boundary patterns.

### Cubecl + numerics reference docs (this repo)
- `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` — authoritative for `pyscf_algebra::gemm` in the gradient/CPHF contractions (algebra wall).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`as_scanner` ships for SCF (SCF-12), MP2 (MP2-07), CCSD (CCSD-07)** — `crates/pyscf-scf/src/scanner.rs`, `pyscf-py/src/{mp,cc}.rs` — the Mole→energy callable the geometry optimizer drives; gradients add the matching gradient-scanner.
- **CCSD λ + `make_rdm1`/`make_rdm2` incl. `ao_repr`** (Phase 6, 06-06 D-03) — GRAD-06 consumes the complete, validated λ+RDM surface directly; no λ re-derivation (the deliberate Phase-6 scope choice paying off here).
- **`canonicalize_signs`** (SCF-13) — the reference `mo_coeff` is already vendor-stable; gradients inherit sign-stability (Pitfall 4/12 re-validation is free).
- **`pyscf-gto` gradient-intor layout entries** — `int1e_ip*`/`int2e_ip1`/`int3c2e_ip1` already declared in `layout_table.rs` (component-leading); only the dispatch guards (`NotYetImplemented{phase:7}`) and the cintx round-trip remain (D-02).
- **`pyscf-algebra` `gemm`/`oracle_sum`/`oracle_dot`/`solve_linear`/`eigh`** — gradient contractions + the CPHF Krylov solve + the RFO eigen-step all route here (algebra wall; bit-exact under `release-oracle`).
- **`pyscf-grids` byte-exact Becke weights + `numint`** (Phase 4) — RKS/UKS-grad `grid_response` weight-derivative term.
- **`pyscf-chkfile` re-exported `hdf5` alias** (Phase 3/4) — GEOMOPT-05 optimizer-state checkpoint, no new dep.
- **`pyscf-diis`/`solve_linear`** (Phase 3) — the B-matrix/linear-solve machinery analogous to the CPHF iterative solve.
- **Phase-5/6 PyO3 pattern** (`mp.rs`/`cc.rs`) — `is_overridden` `__qualname__` MRO check + `call_method1` hook dispatch + eager SCF snapshot + `as_scanner` re-run + `py.detach`; `PyGradients`/geomopt bridge follows it.

### Established Patterns
- **Algebra wall** — `pyscf-grad`/`pyscf-geomopt` depend on `pyscf-algebra` (+ gto/scf/mp2/ccsd/df/grids/chkfile), **never `cubecl-*` directly**; xtask `algebra_wall` allowlist extended.
- **Sibling-crate fidelity (hard preference)** — `pyscf-grad` mirrors `pyscf/grad/{rhf,uhf,rks,uks,mp2,ump2,ccsd,uccsd}.py` + `pyscf/scf/{cphf,ucphf}.py`; **`pyscf-geomopt` ports geomeTRIC's algorithm (NOT PySCF's external-lib shim) + mirrors `geometric_solver.py`/`berny_solver.py` API**.
- **Method crates stay pyo3-free; bridge in `pyscf-py`** (Phase 3 D-01 → MP2 D-07 → CCSD D-09 → here).
- **Bit-exact under `release-oracle` via ordered reductions** — every gradient/CPHF/RFO reduction materializes-then-`oracle_sum`/`oracle_dot` (no bare `+=`); thread-count invariant (Pitfall 1/2).
- **FD self-verification as the always-on gate** (D-01) — a *new* gating story for this phase: unlike Phases 3–6 where upstream byte-identity was the (gated) numeric truth, `verify_fd` is self-contained and is the daily gate; upstream is the periodic `workflow_dispatch` cross-check.
- **"Don't freeze compile / don't freeze the test run"** (user memory) — port the reference algorithm (no codegen/heavy build.rs); `libxc_rs` stays out of the grad/geomopt dep graph; heavy + upstream tests are `workflow_dispatch`/human-verify (D-01/D-05).

### Integration Points
- **`crates/pyscf-grad/` (fill the stub)** — base `Gradients` + per-method grads + `cphf` module (D-03) + `verify_fd` (D-01) + `atmlst` (D-09); pyo3-free.
- **`crates/pyscf-geomopt/` (fill the stub)** — native BFGS+RFO redundant-internals engine + `geometric_solver`/`berny_solver` shim entry points + HDF5 checkpoint; pyo3-free.
- **`crates/pyscf-gto/{intor.rs,layout_table.rs,ecp_engine_cintx.rs}`** — wire the gradient intors, drop `NotYetImplemented{phase:7}`, wire ECP-grad names (D-02, Wave 0).
- **`crates/pyscf-py/`** — `PyGradients`/per-method grad classes + `geomopt` submodule + `python/pyscf/{grad,geomopt}` overlays + `mf.nuc_grad_method()` graft + `pyscf.geomopt.optimize` / `geometric_solver.optimize` / `berny_solver.optimize` entry points (D-07).
- **`Cargo.toml` workspace** — wire `pyscf-grad`/`pyscf-geomopt` deps (members already registered Phase 1; no member-count change); **no pyo3 dep on the method crates.**
- **`xtask` + `.github/workflows/ci.yml`** — algebra-wall allowlist + the D-01/D-05 CI arm structure.

</code_context>

<specifics>
## Specific Ideas

- **The native optimizer is the single biggest novelty of the whole project** — every prior phase ported from PySCF source checked into the repo; Phase 7's optimizer has **no in-tree source** because upstream `geomopt` only shims external `geometric`/`pyberny`. This is why sequencing de-risks it early (D-08) and why geomeTRIC is fetched as the canonical reference (D-06).
- **FD verification flips the gating story** — `verify_fd` (D-01) is the first always-on *numeric* gate in the project that needs no upstream PySCF, because it self-validates the analytical gradient against finite differences of the already-shipped `as_scanner` energy. This is cleaner than the Phase 3–6 "structural always-on + upstream gated" split.
- **CPHF is for the non-variational methods only** — a deliberate correction of the ROADMAP SC-3 shorthand: variational RHF/UHF/RKS/UKS energy gradients are stationary and never call CPHF; only MP2 and CCSD Z-vectors do (D-04). The "one CPHF, not N" guarantee (GRAD-10) is about those two sharing one solver, not about HF/KS grads.
- **Phase 6 pre-built GRAD-06's hardest input** — shipping the full λ + `make_rdm1`/`make_rdm2` (incl. `ao_repr`) in Phase 6 (06-06 D-03) was an explicit investment so CCSD-grad consumes a complete, validated surface rather than half-wiring it here.
- **Gradient ECP closes the GTO-05 arc** — Phase 2 wired ECP *evaluation* (`int1e_ecp`); Phase 7 wires ECP *gradients* (`int1e_ecp_ipnuc`/`iprinv`/`ECPscalar_ip*`), the names `ecp_engine_cintx.rs:78` explicitly reserves for "Phase 7 GRAD-07".
- **HESS deferral keeps the optimizer gradient-only** — no analytical second derivatives; BFGS builds the approximate Hessian and RFO uses a model Hessian. Capture any Hessian/frequency pull as a deferred idea.

</specifics>

<deferred>
## Deferred Ideas

- **Analytical Hessians + vibrational frequencies / IR intensities** (`HESS-01..03`, `pyscf/hessian/{rhf,rks}.py`, `thermo.py`) — v1.x deferred. The CPHF solver (D-03) is the natural future host for the Hessian response equations, but Phase 7 builds it for gradients only.
- **Constrained geometry optimization** (`GEOMOPT-EXT-01` — bond/angle/dihedral constraints) — v1.x. The `constraints` kwarg is accepted-but-unsupported with a clear error this phase (D-07).
- **CCSD(T) / TD / CASSCF / MCSCF gradients** (`grad/ccsd_t.py`, `grad/tdrhf.py`, `grad/casscf.py`, …) — out of v1 (CCSD(T) is v1.x P1; the rest are separate milestones).
- **Transition-state search / IRC / NEB / dimer methods** — not in any v1 REQ; the optimizer targets minima only.
- **DF-gradient as a gated headline** — `int3c2e_ip1` is in the layout table; a `density_fit()`-reference gradient path is Claude's-discretion follow-on, not a separate gated REQ.
- **Fused cubecl gradient/optimizer kernel** — Phase 8 (the 2–5× + GPU owner), only if profiling shows the `pyscf-algebra` chain is the bottleneck.
- **Dispersion-correction gradients** (`grad/dispersion.py`) — no v1 REQ maps; defer.

### Reviewed Todos (not folded)
None — the todo cross-reference scan returned 0 matches for Phase 7.

</deferred>

---

*Phase: 07-gradients-geomopt*
*Context gathered: 2026-05-26*
