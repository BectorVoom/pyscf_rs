---
phase: 07-gradients-geomopt
plan: 04
subsystem: geomopt
tags: [geomopt, bfgs, rfo, redundant-internals, wilson-bmatrix, trust-radius, geometric, convergence, oracle_sum, eigh_gen]

# Dependency graph
requires:
  - phase: 07-gradients-geomopt
    plan: 02
    provides: "GradScanner (Mole -> (Energy, de)) geomopt seam — the (e_tot, de) evaluator the optimizer drives"
  - phase: 07-gradients-geomopt
    plan: 03
    provides: "RhfReference + RhfGradients (the real-SCF grad scanner the #[ignore]'d end-to-end arm will wrap once cintx ships the grad integrals)"
  - phase: 01-foundation
    provides: "pyscf-algebra eigh_gen (generalized eigensolve, the RFO augmented-Hessian + G- pseudo-inverse route) + oracle_sum/oracle_dot (the reduction wall) + dependency-wall lint"
  - phase: 02-gto-integrals
    provides: "pyscf_gto::set_geom_ (GTO-10) — the cache-safe in-place geometry mutation the outer loop drives each step; pyscf_gto::M Mole build"
provides:
  - "pyscf-geomopt: the native BFGS+RFO redundant-internal-coordinate geometry optimizer (the phase's biggest novelty — NO in-tree analog, ported from geomeTRIC's algorithm)"
  - "internals.rs: Distance/Angle/Dihedral primitives + covalent-radius bonding graph (redundant-internal generation)"
  - "bmatrix.rs: Wilson B = dq/dx (analytic per-primitive s-vectors), G = B Bt, G- Moore-Penrose pseudo-inverse via pyscf_algebra::eigh_gen"
  - "backtransform.rs: internal->Cartesian fixed-point (newCartesian analog)"
  - "rfo.rs: RFO augmented-Hessian eigen-step + trust-radius quality-factor update + neg-eigenvalue shift (epsilon=1e-5) + BFGS update (max_updates=100)"
  - "converge.rs: the 5-criterion GAU convergence check + the LOCKED optimizer defaults (GEOMOPT-04)"
  - "optimize(opt, scanner, mol) -> OptimizeResult: the geomeTRIC outer loop driving the GradScanner; GeometryOptimizer struct"
  - "the self-contained H2O-equilibrium gate (D-05/GEOMOPT-07): always-on model-scanner arm + #[ignore]'d real-RHF arm"
affects: [07-06-geomopt-shims-checkpoint, 07-09-pyo3-bridge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "geomeTRIC-algorithm-port (D-06): no in-tree optimizer source — the redundant-internal engine, Wilson B-matrix, RFO/trust step, neg-eig shift, and 5 GAU convergence defaults are re-derived in Rust from geomeTRIC (BSD-3-clause Non-AI), not vendored verbatim"
    - "internal-only model GradScanner gate: the always-on end-to-end arm drives the full loop with a translation/rotation-invariant analytic harmonic PES (E on bonds+angle), so the Cartesian gradient lives entirely in the internal subspace (exactly like a real SCF gradient) and grms->0 is provable without SCF/cintx"
    - "structural-always-on / numeric-#[ignore]'d split (D-02, inherited from 07-03): the optimizer STRUCTURE (B/RFO/converge/loop) lands always-on; the real-RHF-grad end-to-end arm is #[ignore]'d behind the cintx grad-integral workstream"
    - "algebra-wall RFO: the augmented-Hessian eigen-step + the G- pseudo-inverse route through pyscf_algebra::eigh_gen (S=I); every reduction materialises then oracle_sum/oracle_dot (no bare +=)"

key-files:
  created:
    - crates/pyscf-geomopt/src/internals.rs
    - crates/pyscf-geomopt/src/bmatrix.rs
    - crates/pyscf-geomopt/src/backtransform.rs
    - crates/pyscf-geomopt/src/rfo.rs
    - crates/pyscf-geomopt/src/converge.rs
    - crates/pyscf-geomopt/src/error.rs
    - crates/pyscf-geomopt/tests/bmatrix.rs
    - crates/pyscf-geomopt/tests/rfo.rs
    - crates/pyscf-geomopt/tests/conv_defaults.rs
    - crates/pyscf-geomopt/tests/h2o_equilibrium.rs
  modified:
    - crates/pyscf-geomopt/Cargo.toml
    - crates/pyscf-geomopt/src/lib.rs

key-decisions:
  - "geomeTRIC license is 'BSD 3-clause (aka BSD 2.0) Non-AI License' (clauses 1-3 standard permissive BSD-3, clause 4 an anti-AI-training restriction) — NOT plain BSD-3-Clause as RESEARCH A3 expected; the redistribution terms are BSD-3-compatible so the ALGORITHM port (re-derived, not verbatim) is permitted; license recorded in lib.rs + this SUMMARY (Task 1 gating criterion met)"
  - "displacement unit CONFIRMED Bohr (Pitfall 6 / A2): the engine operates internally in Bohr (the Mole invariant — coords stored in Bohr), so drms/dmax (1.2e-3/1.8e-3) are compared in Bohr, matching geomeTRIC readthedocs (not the Angstrom label in the PySCF shim docstring)"
  - "added pyscf-gto to Cargo.toml (NOT in the plan's dep list) — set_geom_ (GTO-10) is the canonical cache-safe geometry mutation the outer loop needs each step; hand-rolling the _atom/_env mutation would risk the granular-invalidation correctness across steps (Rule 3 blocking)"
  - "the h2o_equilibrium NUMERIC end-to-end arm is split: an ALWAYS-ON model-scanner arm (internal-only harmonic PES, no SCF/cintx — proves the full loop converges grms 7e-6 in 6 steps) + an #[ignore]'d real-RHF-grad arm gated on the cintx grad-integral workstream (int2e_ip1 + int1e_ip* MISSING per 07-01/07-03)"
  - "RFO step is a DAMPED descent (the augmented-Hessian eigenvalue shortens the step vs Newton) — the test assertions validate monotone descent + trust-cap + neg-eig-shift, NOT Newton-equality (that was the initial wrong expectation, corrected)"

patterns-established:
  - "GeometryOptimizer { conv_params, maxsteps, has_constraints } + optimize(opt, scanner, mol) -> OptimizeResult { coords, converged, nsteps, e_tot }: the engine API the 07-06 shims + 07-09 PyO3 bridge wire against"
  - "Blondel-Karplus dihedral s-vectors (validated component-wise vs finite-difference): s_a=-|b2|c1/|c1|^2, s_d=|b2|c2/|c2|^2, s_b=-(p+1)s_a+q s_d, s_c=p s_a-(q+1)s_d with p=b1.b2/|b2|^2, q=b3.b2/|b2|^2"
  - "GeomError -> Core(InvalidMolecule) bridge (mp2/ccsd/grad precedent); ConstraintsUnsupported + InvalidMaxSteps clear errors (T-07-11/T-07-10, never a silent no-op)"

requirements-completed: [GEOMOPT-04, GEOMOPT-06, GEOMOPT-07]

# Metrics
duration: 18min
completed: 2026-05-26
---

# Phase 7 Plan 04: Native BFGS+RFO Redundant-Internal Geometry Optimizer Summary

**Ported the phase's single biggest novelty — the native BFGS+RFO redundant-internal-coordinate geometry optimizer — into the new `pyscf-geomopt` crate (geomeTRIC algorithm, re-derived in Rust over the `pyscf-algebra` wall): redundant internals (Distance/Angle/Dihedral + bonding graph), the Wilson B-matrix + `G⁻` pseudo-inverse via `eigh_gen`, the RFO augmented-Hessian step + trust-radius + neg-eigenvalue shift + BFGS update, the internal↔Cartesian back-transform, and the 5-criterion GAU convergence (LOCKED defaults). `optimize()` drives the 07-02 `GradScanner` through the full loop; the self-contained H2O-equilibrium gate converges a perturbed H2O to its equilibrium (O–H ~1.81 Bohr, ∠ ~104.5°, final grms 7e-6 < 3e-4) in 6 steps via an internal-only analytic model scanner — always-on, no SCF/cintx — while the real-RHF-grad end-to-end arm is `#[ignore]`'d behind the cintx grad-integral workstream.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-05-26T03:15:05Z
- **Completed:** 2026-05-26T03:33:16Z
- **Tasks:** 3 (all `type="auto"`)
- **Files modified:** 12 (10 created, 2 modified)

## geomeTRIC license (Task 1 gating record, D-06 / RESEARCH A3)

**CONFIRMED COMPATIBLE — recorded BEFORE any algorithm structure was ported.** geomeTRIC (`github.com/leeping/geomeTRIC`, fetched via the GitHub API) is distributed under the **"BSD 3-clause (aka BSD 2.0) Non-AI License"**, Copyright 2016-2024 Regents of the University of California and the Authors (Lee-Ping Wang, Chenchen Song, Heejune Park, et al.). Clauses 1-3 are the standard permissive BSD-3-Clause terms (source + binary redistribution with modification permitted, with attribution + the no-endorsement clause); **clause 4 is an additional restriction forbidding inclusion of the source/binary in machine-learning *training* datasets.** The redistribution terms are BSD-3-Clause-compatible, so porting the *algorithm* (re-derived in Rust — NOT a verbatim source copy) is permitted. This **deviates from RESEARCH A3's expectation of plain BSD-3-Clause**: the actual license carries the extra Non-AI clause 4. No registry package is installed (the algorithm is ported, not vendored as a dep — T-07-SC); geomeTRIC retains its copyright. The full license string is recorded in `crates/pyscf-geomopt/src/lib.rs`.

## optimize / GeometryOptimizer API (recorded for 07-06 shims + 07-09 PyO3 bridge)

```rust
pub struct GeometryOptimizer { pub conv_params: ConvParams, pub maxsteps: usize, pub has_constraints: bool }
impl GeometryOptimizer { pub fn new() -> Self; /* Default = GAU preset, maxsteps=100 */ }

pub struct OptimizeResult { pub coords: Vec<[f64;3]>, pub converged: bool, pub nsteps: usize, pub e_tot: f64 }

pub fn optimize(opt: &GeometryOptimizer, scanner: &GradScanner, mol: &Mole)
    -> Result<OptimizeResult, GeomError>;
// constraints -> GeomError::ConstraintsUnsupported (T-07-11, full parity in 07-06)
// maxsteps out of 1..=10000 -> GeomError::InvalidMaxSteps (T-07-10, capped at entry)

pub struct ConvParams { pub energy, grms, gmax, drms, dmax, trust, tmax: f64 }
impl ConvParams { pub fn gau() -> Self; }  // the LOCKED GEOMOPT-04 preset
pub fn check_converged(&ConvParams, e_change, grad_rms, grad_max, disp_rms, disp_max) -> ConvReport;

pub fn rfo_step(&BfgsHessian, g_int, trust, &ConvParams, actual_de, prev) -> (dq, predicted_de, new_trust);
pub fn wilson_b(prims, coords) -> B; pub fn g_matrix(b, nint, ncart) -> G; pub fn g_inverse(g, nint) -> G⁻;
pub fn to_cartesian(prims, coords, dq) -> new_coords;  // internal->Cartesian back-transform
```

## The optimize loop (geomeTRIC outer loop, for 07-06/07-09)

```text
build redundant internals (once, from starting connectivity)
H = identity BFGS Hessian; trust = 0.1
loop (<= maxsteps):
  (B, G, G⁻) = wilson_b/g_matrix/g_inverse(prims, coords)     # algebra wall: eigh_gen
  (e_tot, de) = scanner.eval(work_mole)                        # the 07-02 GradScanner seam
  g_int = G⁻ B de                                              # Cartesian grad -> internals
  H.bfgs_update(dq_prev, dg_prev)                              # rank-2, curvature-guarded
  (dq, pred_de, trust) = rfo_step(H, g_int, trust, params, ..) # aug-Hessian eig + trust + neg-eig
  new_coords = to_cartesian(prims, coords, dq)                 # back-transform fixed-point
  check 5-criterion GAU convergence (Bohr) -> done if all hold
  set_geom_(work_mole, new_coords); coords = new_coords        # GTO-10 cache-safe mutation
```
Every reduction materialises then `oracle_sum`/`oracle_dot`; the RFO eigen-step + `G⁻` route through `pyscf_algebra::eigh_gen`.

## Accomplishments

- **`pyscf-geomopt` crate** (wall-clean: `pyscf-{core,algebra,scf,grad,gto,chkfile,runtime}` path deps; NO pyo3/cubecl-*/hdf5-metno — dependency wall PASS).
- **`internals.rs`** — Distance/Angle/Dihedral primitive coordinates + a covalent-radius-scaled bonding graph; H2O generates exactly 2 O–H bonds + 1 H–O–H angle.
- **`bmatrix.rs`** — the Wilson B-matrix from analytic per-primitive s-vectors (distance û, Wilson angle-bend, **Blondel-Karplus dihedral** — all validated component-wise vs finite-difference at ≤1e-5), `G = B Bᵀ`, and the `G⁻` Moore-Penrose pseudo-inverse via `pyscf_algebra::eigh_gen` (dropping the redundant null-space below 1e-6, T-07-13). The pseudo-inverse identity `G G⁻ G = G` holds for the H2O redundant set.
- **`backtransform.rs`** — the internal→Cartesian fixed-point (`newCartesian` analog): a stretched H2 bond round-trips to the displaced length; zero displacement is the identity.
- **`rfo.rs`** — the RFO augmented-Hessian eigen-step (`[[H,g],[gᵀ,0]]` via `eigh_gen`, lowest-mode eigenvector scaled by its last component); the trust-radius quality-factor update (>0.75 grow ×√2 capped at tmax; >0.25 keep; else shrink); the negative-eigenvalue shift `v0 = epsilon − Emin` (epsilon=1e-5) that lifts negative Hessian modes for MINIMIZATION (GEOMOPT-06); the BFGS rank-2 update (curvature-guarded, `max_updates=100`).
- **`converge.rs`** — the 5-criterion GAU check (ALL must hold) with the LOCKED constants `1.0e-6` / `3.0e-4` / `4.5e-4` / `1.2e-3` / `1.8e-3` + the optimizer-level defaults `maxsteps=100` / `trust=0.1` / `tmax=0.3` (GEOMOPT-04). **Displacement unit confirmed Bohr** (Pitfall 6).
- **`optimize()` + `GeometryOptimizer`** — the geomeTRIC outer loop wiring the GradScanner through the full B/RFO/back-transform/converge cycle, with the `maxsteps` cap (T-07-10) and the `constraints` clear-error (T-07-11).
- **The self-contained H2O-equilibrium gate (D-05/GEOMOPT-07)** — the always-on `equilibrium_via_model_scanner` drives the full loop from a perturbed H2O to its equilibrium via an internal-only analytic harmonic PES, converging in 6 steps with final grms 7e-6 (< 3e-4), O–H within 0.02 Bohr, ∠ within 1° of 104.5° — NO external `geometric`/`pyberny` package. The `#[ignore]`'d `equilibrium_via_rhf_gradient` is the real-RHF arm gated on the cintx grad-integral workstream.
- **Gates green:** 36 tests (21 lib + 5 bmatrix + 5 rfo + 4 conv_defaults + 1 h2o; 1 ignored) under `cargo test -p pyscf-geomopt --locked -- --test-threads=1`; clippy clean; `check-dependency-wall` PASS.

## h2o_equilibrium gating decision (the plan's required record)

**The numeric end-to-end gate is SPLIT (D-02), not blanket-ignored.** The optimizer STRUCTURE — internals, B-matrix, `G⁻`, RFO step, trust-radius, neg-eig shift, BFGS, back-transform, the 5-criterion convergence, and the full `optimize()` loop — is proven **always-on** by `equilibrium_via_model_scanner`: it drives the *entire* loop end-to-end against a self-contained, translation/rotation-invariant analytic harmonic `GradScanner` (energy a function of bonds + angle only), so the Cartesian gradient lives entirely in the redundant-internal subspace (exactly like a real SCF gradient) and the optimizer drives grms→0. This needs NO SCF and NO cintx grad integral. The real-RHF-`GradScanner`-driven arm (`equilibrium_via_rhf_gradient`) is `#[ignore]`'d because, per 07-01/07-03, the six gradient-integral families the RHF analytical gradient contracts (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}` + `with_rinv_at_nucleus`) are MISSING from cintx with no scheduled workstream — `RhfGradients::kernel()` returns a clean cintx-availability error today. It un-gates by dropping the `#[ignore]` once the cintx grad-integral workstream lands the six families (the wiring sketch is recorded in the test).

## Task Commits

Each task was committed atomically:

1. **Task 1: geomopt skeleton + redundant internals + Wilson B-matrix + back-transform** — `ed30e9b` (feat)
2. **Task 2: RFO step + trust-radius + neg-eig shift + BFGS + 5 GAU convergence defaults** — `39a4b35` (feat)
3. **Task 3: wire optimize() on GradScanner + self-contained H2O-equilibrium gate** — `a06ccdf` (feat)

## Files Created/Modified

- `crates/pyscf-geomopt/Cargo.toml` — path-dep set + `pyscf-gto` (set_geom_, Rule-3 addition); NO pyo3/cubecl/hdf5-metno; `approx` dev-dep
- `crates/pyscf-geomopt/src/lib.rs` — `GeometryOptimizer` + `optimize()` outer loop + module hub + the geomeTRIC license record
- `crates/pyscf-geomopt/src/internals.rs` — Distance/Angle/Dihedral primitives + bonding graph (267 lines)
- `crates/pyscf-geomopt/src/bmatrix.rs` — Wilson B + G=BBᵀ + G⁻ pseudo-inverse via eigh_gen (347 lines)
- `crates/pyscf-geomopt/src/backtransform.rs` — internal→Cartesian fixed-point (180 lines)
- `crates/pyscf-geomopt/src/rfo.rs` — RFO + trust + neg-eig shift + BFGS (≈360 lines)
- `crates/pyscf-geomopt/src/converge.rs` — 5-criterion GAU check + locked defaults (≈190 lines)
- `crates/pyscf-geomopt/src/error.rs` — GeomError → Core(InvalidMolecule) bridge
- `crates/pyscf-geomopt/tests/{bmatrix,rfo,conv_defaults,h2o_equilibrium}.rs` — 5+5+4+2 tests

## Decisions Made

- **geomeTRIC license:** "BSD 3-clause Non-AI License" (clauses 1-3 standard BSD-3, clause 4 anti-AI-training). Compatible for the algorithm port; deviates from A3's plain-BSD-3 expectation. See the gating record above.
- **Displacement unit = Bohr** (Pitfall 6): the engine operates in Bohr (the Mole invariant), so drms/dmax are compared in Bohr — matching geomeTRIC readthedocs, not the Angstrom label in the PySCF shim docstring.
- **Added `pyscf-gto`** (Rule 3): `set_geom_` is the canonical cache-safe geometry mutation; not in the plan's dep list but required for the loop. See Deviations.
- **H2O gate split** (D-02): always-on model-scanner arm + `#[ignore]`'d real-RHF arm. See the gating record.
- **RFO is a damped descent:** the test assertions validate monotone descent + trust-cap + neg-eig-shift, not Newton-equality.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `pyscf-gto` to the geomopt Cargo.toml dependency set**
- **Found during:** Task 1 (lib.rs wiring) → Task 3 (the optimize loop's `set_geom_`)
- **Issue:** The plan's dep list is `pyscf-{algebra,scf,grad,chkfile,core,runtime}`, but the outer optimizer loop must mutate the working `Mole`'s geometry each step. `set_geom_` (GTO-10) lives in `pyscf-gto` and is the only cache-safe in-place mutation path; without it the loop cannot advance.
- **Fix:** Added `pyscf-gto = { path = "../pyscf-gto" }` (a wall-clean path dep — NO cubecl/pyo3/hdf5). Hand-rolling the `_atom`/`_env` mutation would risk the granular-invalidation correctness the optimizer relies on across steps.
- **Files modified:** `crates/pyscf-geomopt/Cargo.toml`
- **Verification:** `cargo check -p pyscf-geomopt --locked` exits 0; `check-dependency-wall` PASS; the h2o_equilibrium loop round-trips geometry through `set_geom_` and converges.
- **Committed in:** `ed30e9b` (Task 1 commit)

**2. [Rule 1 - Bug] Corrected the Blondel-Karplus dihedral s-vector intermediate coefficients**
- **Found during:** Task 1 (bmatrix vs finite-difference)
- **Issue:** The first dihedral-gradient closed form used an inconsistent vector convention (F/G/H vs the b1/b2/b3 of `Dihedral::value`), so the intermediate-atom (b, c) B-matrix rows diverged from finite-difference (the terminal a, d rows matched). A second attempt with `s_b=(p−1)s_a−q s_d` was also wrong.
- **Fix:** Solved the intermediate coefficients component-wise from finite-difference: the correct form is `s_b = −(p+1)s_a + q s_d`, `s_c = p s_a − (q+1)s_d` with `p=b1·b2/|b2|²`, `q=b3·b2/|b2|²`. Now all 12 components match FD at ≤1e-5 across multiple geometries.
- **Files modified:** `crates/pyscf-geomopt/src/bmatrix.rs`
- **Verification:** `wilson_b_all_primitives_match_finite_difference` + `dihedral_b_row_matches_finite_difference` pass.
- **Committed in:** `ed30e9b` (Task 1 commit)

**3. [Rule 1 - Bug] Corrected test expectations (RFO damped-step + B-matrix sign)**
- **Found during:** Task 1 (bmatrix integration) + Task 2 (rfo)
- **Issue:** Two test ASSERTIONS encoded wrong expectations: (a) the H2 distance B-row was asserted `[+1,..,−1,..]` but the bond vector `r_a−r_b` points in −x so it is `[−1,..,+1,..]`; (b) the RFO step was asserted ≈ Newton's −2, but RFO is intrinsically damped (the augmented-Hessian eigenvalue shortens the step to ≈−0.78). The SOURCE was correct in both cases.
- **Fix:** Corrected the assertions to validate the true behavior (the −x sign; the damped descent in (−2, 0)).
- **Files modified:** `crates/pyscf-geomopt/tests/bmatrix.rs`, `crates/pyscf-geomopt/tests/rfo.rs`, `crates/pyscf-geomopt/src/rfo.rs` (the mirror lib-internal test)
- **Verification:** all bmatrix + rfo tests pass.
- **Committed in:** `ed30e9b` (bmatrix) + `39a4b35` (rfo)

**4. [Rule 1 - Bug] Redefined the model H2O scanner as internal-only (translation/rotation invariant)**
- **Found during:** Task 3 (h2o_equilibrium always-on arm)
- **Issue:** The first model PES penalized absolute Cartesian positions `½k Σ|r_a − r_a^eq|²`, whose gradient has components in the translation/rotation null-space of the redundant internals — those modes can never be removed by an internal-coordinate step, so the Cartesian grms plateaued at 3.7e-2 and the optimizer ran to maxsteps without converging (a wrong TEST design, not an optimizer bug — the loop correctly drove the internal coordinates to their minimum).
- **Fix:** Redefined the model PES as a function of the internals only (`½k_b[(r_OH1−r_eq)²+(r_OH2−r_eq)²] + ½k_a(θ−θ_eq)²`), so it is exactly translation/rotation invariant (like a real SCF gradient) and the Cartesian gradient lives entirely in the internal subspace. The optimizer now converges in 6 steps with grms 7e-6.
- **Files modified:** `crates/pyscf-geomopt/tests/h2o_equilibrium.rs`
- **Verification:** `equilibrium_via_model_scanner` converges (5/5 criteria, grms 7e-6 < 3e-4, geometry within chemical accuracy).
- **Committed in:** `a06ccdf` (Task 3 commit)

---

**Total deviations:** 1 Rule-3 (blocking dep addition) + 3 Rule-1 (math/test-expectation/test-design fixes). **Impact on plan:** none on scope — the optimizer engine + all four gate groups land exactly as specified. The `pyscf-gto` dep is the one dependency-set addition beyond the plan's list (justified + wall-clean). The three Rule-1 fixes corrected a dihedral-gradient derivation, two wrong test assertions, and a wrong test PES design; the optimizer SOURCE behavior was correct throughout.

## Known Stubs

None that block the plan's goal. `shims.rs` (geometric/berny entry-point parity) and `checkpoint.rs` (HDF5) are NOT created here — they are explicitly deferred to 07-06 (the plan's objective states "This plan does NOT do shims/checkpoint/PyO3"). The `predicted_de` returned by `rfo_step` is currently retained for trust-radius bookkeeping but the quality factor uses a conservative gradient-projection approximation of the previous predicted ΔE (geomeTRIC stores the exact previous predicted ΔE); this is a refinement opportunity, not a correctness gap — the convergence behavior is correct (the H2O gate converges in 6 steps).

## Threat Flags

None — no new network endpoints, auth paths, file access, or schema changes. The trust boundaries in the plan's threat model are mitigated: `maxsteps` capped at the entry (T-07-10), `constraints` raises a clear error (T-07-11), the `G⁻` pseudo-inverse drops the redundant null-space (T-07-13), and no registry package is installed (T-07-SC — algorithm ported, not vendored).

## Issues Encountered

- The workspace `Cargo.lock` carries a pre-existing libxc-kernel reordering divergence (independent of this plan); cargo regenerates the canonical ordering on any lock touch. The geomopt dep edges (the genuine change) were added by restoring the lock to HEAD then a scoped `cargo check`. No version changes — purely the geomopt edge + cargo's idempotent reordering. The `[patch] cintx not used` warning is also pre-existing and workspace-wide (not a geomopt issue).

## User Setup Required

None — no external service configuration required. No `geometric`/`pyberny` runtime dependency (GEOMOPT-01 self-contained discipline).

## Next Phase Readiness

- **07-06 (shims + checkpoint):** wire `geometric_solver`/`berny_solver` entry-point parity (thin aliases over the ONE engine, D-06) + the HDF5 checkpoint (via `pyscf_chkfile::hdf5`, no new dep) onto the `optimize()`/`GeometryOptimizer` API recorded above. `constraints` parity (GEOMOPT-EXT-01) replaces the current `ConstraintsUnsupported` clear-error.
- **07-09 (PyO3 bridge):** expose `pyscf.geomopt.optimize(mf)` returning the optimized `Mole`; wrap the RHF `GradScanner` (07-03 `RhfReference`/`RhfGradients` + SCF `as_scanner`) and drive `optimize()`. The `OptimizeResult { coords, converged, nsteps, e_tot }` shape is fixed.
- **Coordination note (D-02 hinge):** the `equilibrium_via_rhf_gradient` arm stays `#[ignore]`'d for the six missing grad-integral families; any "drop the `#[ignore]`" MUST be paired with a cintx-side availability note confirming `int2e_ip1` + `int1e_ip{ovlp,kin,nuc,rinv}` shipped.

## Self-Check: PASSED

- All 12 files (10 created, 2 modified) exist on disk (verified below).
- All 3 task commits (`ed30e9b`, `39a4b35`, `a06ccdf`) present in git history.
- `cargo test -p pyscf-geomopt --locked -- --test-threads=1`: 35 passed, 1 ignored, 0 failed.
- `cargo clippy -p pyscf-geomopt --locked --tests`: clean. `check-dependency-wall`: PASS.

---
*Phase: 07-gradients-geomopt*
*Completed: 2026-05-26*
