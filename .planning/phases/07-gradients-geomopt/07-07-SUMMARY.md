---
phase: 07-gradients-geomopt
plan: 07
subsystem: gradients
tags: [gradients, cphf, cpks, krylov, pople-1979, mp2, z-vector, relaxed-density, lagrangian, oracle_sum, single-cphf-impl, GRAD-05, GRAD-10, D-03, cintx-gated]

# Dependency graph
requires:
  - phase: 07-gradients-geomopt
    plan: 01
    provides: "cintx grad-intor availability split (int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} MISSING → clean cintx-availability error, never NotYetImplemented{phase:7}); int2e (energy J/K) IS cintx-ready (05-08), so the MP2 Z-vector get_veff fvind runs un-gated"
  - phase: 07-gradients-geomopt
    plan: 03
    provides: "RhfReference + the grad_elec base decomposition (Hellmann-Feynman + 2e Pulay + overlap Pulay); make_rdm1e (energy-weighted RDM); get_ovlp; the structural-lands / numeric-#[ignore]'d cintx-gating precedent (D-02)"
  - phase: 05-mp2
    provides: "pyscf_mp2::gamma1_intermediates (doo/dvv from t2); Mp2Reference + Mp2Result::t2 amplitude layout; default_get_veff (energy int2e J/K)"
  - phase: 01-foundation
    provides: "pyscf_algebra::oracle_sum/oracle_dot (deterministic reductions) + solve_linear (dense LU for the small projected Krylov system)"
provides:
  - "cphf::solve — the SINGLE matrix-free Krylov CPHF/CPKS solver (D-03, GRAD-10): ports pyscf/scf/cphf.py:solve→solve_nos1 + lib.krylov (Pople 1979); exact upstream defaults max_cycle=50/tol=1e-9/level_shift=0; caller-overridable max_cycle; matrix-free aop only (NEVER materializes the dense O(nocc²·nvir²) A); solve_withs1 rejected this phase"
  - "cphf::Fvind type + DEFAULT_MAX_CYCLE/DEFAULT_TOL/DEFAULT_LEVEL_SHIFT constants — the response-operator contract CCSD-grad (07-08) consumes with its own fvind/RHS"
  - "Mp2Gradients + Mp2Reference (RHF+MP2 snapshot incl. t2): the relaxed-density Lagrangian + Z-vector consumer that routes its own fvind + Xvo through cphf::solve at max_cycle=30 (Pitfall 5)"
  - "mp2::response_dm1 (the _response_dm1 Z-vector contract) + build_xvo_base (the int2e-only Xvo RHS arm)"
  - "single_cphf_impl structural gate (GRAD-10): exactly ONE `pub fn solve(` CPHF in pyscf-grad, located in cphf.rs"
affects: [07-08-ccsd-grad, 07-09-pyo3-bridge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Matrix-free Krylov over the algebra wall: the Pople-1979 lib.krylov nroots=1 path with every inner product via oracle_dot, the projection/recombination via oracle_sum, and the small projected nd×nd system through pyscf_algebra::solve_linear — NO dense A-matrix ever materialized (T-07-22 anti-pattern)"
    - "ONE-solver consumer contract (D-03/GRAD-10): a caller-supplied Fvind callback + RHS; the single_cphf_impl source-scan test forbids a second `pub fn solve(` CPHF copy"
    - "Z-vector method gradient (the first non-variational grad, D-04): relaxed-density intermediates → dm1mo → Xvo RHS → cphf::solve(max_cycle=30) → response dm1; the numeric de-assembly rides the cintx grad-intor gate, the response machinery is always-on"
    - "non-convergence / singular-denominator / s1-Some / hermi-true → clear error, NEVER an infinite loop or wrong tensor (T-07-21)"

key-files:
  created:
    - crates/pyscf-grad/tests/cphf.rs
    - crates/pyscf-grad/tests/mp2_verify_fd.rs
  modified:
    - crates/pyscf-grad/src/cphf.rs
    - crates/pyscf-grad/src/mp2.rs
    - crates/pyscf-grad/src/lib.rs

key-decisions:
  - "The MP2 Z-vector fvind uses the ENERGY get_veff (int2e — cintx-ready as of 05-08), NOT a gradient integral, so the Z-vector solve + response_dm1 are FULLY RUNNABLE un-gated (mp2_response_dm1_shapes_or_clean_error passes end-to-end). Only the gradient `de` ASSEMBLY (which contracts int2e_ip1 + int1e_ip*) rides the cintx grad-intor gate."
  - "cphf::solve takes a `&Fvind<'_>` trait-object callback (not a generic) so the ONE solver has a single concrete entry point the single_cphf_impl scan can pin; MP2 and CCSD both pass `&|x| ...` closures."
  - "single_cphf_impl scans src/ for `pub fn solve(` (a CPHF solver declaration) and asserts exactly one, in cphf.rs — it deliberately does NOT trip on `cphf::solve(` CALL sites (path-prefixed, never start a trimmed line with `pub fn`). The pyscf-mp2 `solve_cphf_rhf` is in a DIFFERENT crate (D-03 scopes the single-impl to pyscf-grad::cphf)."
  - "The numeric MP2 FD arm is #[ignore]'d per the 07-01/07-03 precedent: the de assembly reaches the RHF get_ovlp (int1e_ipovlp — missing) first, so kernel() ?-propagates a clean cintx-availability error today; the relaxed-density + Z-vector body un-gates by dropping the #[ignore] once the cintx grad-integral workstream lands the six families."
  - "AlgebraError bridges to PyscfRsError through GradError::Algebra (the #[from] arm) via .map_err(GradError::from) — there is no direct From<AlgebraError> for PyscfRsError."

patterns-established:
  - "Fvind response-operator callback shape: `Fn(&[f64]) -> Result<Vec<f64>, PyscfRsError>` over the flat vir-major (a·nocc + i) rotation space — the CCSD-grad (07-08) consumer reuses this verbatim with its own fvind/RHS"
  - "The grad-local Mp2Reference mirrors RhfReference + t2; as_rhf() projects it onto the RHF base decomposition the relaxed density plugs into"

requirements-completed: [GRAD-05, GRAD-10]

# Metrics
duration: 9min
completed: 2026-05-26
---

# Phase 7 Plan 07: The Single CPHF/CPKS Solver + MP2 Gradient (D-03 / GRAD-05 / GRAD-10) Summary

**Built the ONE matrix-free Krylov CPHF/CPKS solver (`cphf::solve`, D-03/GRAD-10) — a faithful port of `pyscf/scf/cphf.py:solve→solve_nos1` + `lib.krylov` (Pople 1979) with the exact upstream defaults (`max_cycle=50, tol=1e-9, level_shift=0`), a caller-supplied matrix-free `Fvind` response operator (the dense `O(nocc²·nvir²)` A is NEVER materialized), every reduction routed through `oracle_dot`/`oracle_sum` and the small projected system through `pyscf_algebra::solve_linear` — then consumed it for the first non-variational gradient: MP2 via the relaxed-density Lagrangian + the Z-vector through that SAME solver at `max_cycle=30` (Pitfall 5). The CPHF solver is pure linear algebra and lands fully tested ALWAYS-ON (convergence vs the dense `solve_linear` reference, the `single_cphf_impl` GRAD-10 gate, defaults, the `max_cycle=30` override, error/edge paths); the MP2 Z-vector + `response_dm1` run un-gated against the cintx-ready energy `int2e` `get_veff`, while the MP2 numeric gradient `de` assembly is `#[ignore]`'d behind the cintx grad-intor gate per the 07-03 precedent.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-05-26T03:59Z (post context-load)
- **Completed:** 2026-05-26T04:08Z
- **Tasks:** 2 (both `type="auto" tdd="true"`)
- **Files:** 5 (2 created, 3 modified)

## The cphf::solve signature + defaults (recorded for 07-08 CCSD-grad)

```rust
pub type Fvind<'a> = dyn Fn(&[f64]) -> Result<Vec<f64>, PyscfRsError> + 'a;

pub const DEFAULT_MAX_CYCLE: usize = 50;     // cphf.py:30
pub const DEFAULT_TOL: f64 = 1e-9;           // cphf.py:30
pub const DEFAULT_LEVEL_SHIFT: f64 = 0.0;    // cphf.py:31

#[allow(clippy::too_many_arguments)]
pub fn solve(
    fvind: &Fvind<'_>,        // the caller-supplied response operator
    mo_energy: &[f64],        // full MO spectrum
    mo_occ: &[f64],           // (vir, occ) split: nocc=#{occ>0}, nvir=#{occ==0}
    h1: &[f64],               // RHS, flat vir-major (a·nocc + i)
    s1: Option<&[f64]>,       // None → solve_nos1 (the Phase-7 Z-vector path)
    max_cycle: usize,         // MP2 passes 30; default 50
    tol: f64,
    hermi: bool,              // must be false (non-symmetric response)
    level_shift: f64,
) -> Result<Vec<f64>, PyscfRsError>;          // returns z, flat vir-major (== cphf.solve(...)[0])
```

### The aop / RHS contract (port of `cphf.py:67-81`)

- `e_ai[a,i] = 1/(e_a[a] + level_shift − e_i[i])`, flat vir-major.
- RHS base: `mo1base = h1 · (−e_ai)`.
- Matrix-free aop: `vind_vo(z) = (fvind(z) [− z·level_shift]) · e_ai`, returned WITHOUT the identity term (the Krylov solver adds the `I` in `(1+a)`).
- The Krylov iteration solves `(1 + vind_vo)·z = mo1base` to `tol`/`max_cycle`; non-convergence → a clear error (never an infinite loop, T-07-21).

## The MP2 Z-vector wiring (recorded for 07-08 — CCSD consumes the SAME solver)

```text
doo,dvv = pyscf_mp2::gamma1_intermediates(t2, nocc, nvir)   # the Phase-5 amplitudes
dm1mo[:nocc,:nocc] = doo + dooᵀ ; dm1mo[nocc:,nocc:] = dvv + dvvᵀ
Xvo  = build_xvo_base(refr, dm1mo)                          # = Cvᵀ·(2·get_veff(C·dm1mo·Cᵀ))·Co   (int2e — cintx-ready)
dvo  = cphf::solve(fvind, mo_energy, mo_occ, Xvo, None, 30, 1e-9, false, 0.0)   # ← max_cycle=30
dm1mo += response_dm1(dvo)                                  # dm1[vir,occ]=dvo ; dm1[occ,vir]=dvoᵀ
de    = rhf::grad_elec(as_rhf(refr), atmlst)                # ← cintx-gated (reaches get_ovlp first)
```
`fvind(x)` (`mp2_fvind`, `mp2.py:274-279`): build AO `dm = Cv·x·Coᵀ`, apply the SCF `get_veff` to `dm+dmᵀ`, project back `Cvᵀ·v·Co`, ×2. CCSD-grad (07-08) supplies its OWN `fvind` + RHS into the identical `cphf::solve`.

## Accomplishments

- **`cphf::solve` (D-03, GRAD-10)** — the single matrix-free Krylov CPHF/CPKS solver. `solve` dispatches `s1=None → solve_nos1` (the Phase-7 Z-vector branch) and rejects `s1=Some` (the field-dependent `solve_withs1`, the Hessian future) with a clear error. The internal `krylov` is the Pople-1979 `nroots=1` path: trial vectors `xs` + their images `ax`, Gram-Schmidt-style projection, the projected `nd×nd` system `h·c=g` solved via `solve_linear`, and the recombination `x = Σ c_i x_i` — every step oracle-ordered. Singular orbital-energy denominators and non-convergence both surface clear errors (T-07-21).
- **`Mp2Gradients` + `Mp2Reference`** — the MP2 gradient driver (the first non-variational grad, D-04). `grad_elec` builds the relaxed-density `dm1mo` from `gamma1_intermediates`, forms the `Xvo` RHS, solves the Z-vector through the ONE `cphf::solve` at `max_cycle=30`, and assembles `de` on the RHF base decomposition. `make_rdm1e` reuses the RHF energy-weighted RDM (`mp2.py:169`).
- **`mp2_fvind` + `response_dm1` + `build_xvo_base`** — the `_response_dm1` Z-vector contract: the response operator uses the ENERGY `int2e` `get_veff` (cintx-ready), so the Z-vector solve runs end-to-end un-gated (the `mp2_response_dm1_shapes_or_clean_error` structural test confirms the real solve returns a transposed (vir,occ)/(occ,vir) MO response density).
- **`tests/cphf.rs` (7 tests, all always-on)** — krylov-converges-to-dense-`solve_linear`-reference (+ a residual cross-check), defaults-are-exact-upstream, the `max_cycle=30` override converges, `s1=Some` rejected, trivial-RHS→zero, aop-error-propagates, and `single_cphf_impl` (GRAD-10: exactly one `pub fn solve(`, in cphf.rs).
- **`tests/mp2_verify_fd.rs` (6 always-on + 1 `#[ignore]`'d)** — the `max_cycle=30`-not-50 override, the Z-vector runs through cphf at the MP2 cap, `response_dm1` shapes/clean-error, `make_rdm1e` shape (`dme0[0,0] = ε0·occ0 = −1.0`), `kernel()` (natm,3)-or-clean-error, atmlst subset, plus the `#[ignore]`'d numeric `verify_fd(disp=1e-4, tol=1e-6)` arm.
- **Gates green:** `cargo test -p pyscf-grad --locked -- --test-threads=1` → 42 passed / 5 ignored / 0 failed (the 5 numeric arms across rhf/uhf/rks/uks/mp2). `cargo clippy -p pyscf-grad --tests --locked` clean (no in-scope warnings). `check-dependency-wall` PASS (no `cubecl-*` in pyscf-grad, T-07-SC).

## MP2 numeric-gating decision (the plan's required record)

**The MP2 numeric FD arm is `#[ignore]`'d (cintx grad-intor workstream PENDING) — but the CPHF solver and the MP2 Z-vector are NOT.** The CPHF solver is pure linear algebra (matrix-free response solve) and is fully tested always-on against the dense `solve_linear` reference. The MP2 Z-vector `fvind` rides the ENERGY `int2e` `get_veff` (cintx-ready as of 05-08), so `response_dm1` / `build_xvo_base` run end-to-end un-gated. ONLY the gradient `de` ASSEMBLY contracts the six cintx-missing grad-intor families (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`) — it reaches the RHF `get_ovlp` (`int1e_ipovlp`) first, so `kernel()` `?`-propagates a clean `Core(InvalidMolecule(..))` cintx-availability error (never `NotYetImplemented{phase:7}`, GRAD-07 closed that). The relaxed-density + Z-vector machinery is complete and always-on; the numeric arm un-gates by dropping the `#[ignore]` once the cintx grad-integral workstream lands the six families (must be paired with a cintx-side availability note).

## Task Commits

1. **Task 1: The single matrix-free Krylov CPHF/CPKS solver (D-03, GRAD-10)** — `d3a8e40` (feat)
2. **Task 2: MP2 gradient — relaxed-density Lagrangian + Z-vector (GRAD-05)** — `c7f96d7` (feat)

## Files Created/Modified

- `crates/pyscf-grad/src/cphf.rs` — replaced the `NotYetImplemented{wave:2}` stub with the full `solve`/`solve_nos1`/`krylov` body + the `Fvind` type + the `DEFAULT_*` constants (393 lines).
- `crates/pyscf-grad/src/mp2.rs` — replaced the `default_grad_elec` stub with `Mp2Gradients`/`Mp2Reference` + `grad_elec`/`response_dm1`/`mp2_fvind`/`build_xvo_base` + `MP2_CPHF_MAX_CYCLE` (502 lines).
- `crates/pyscf-grad/src/lib.rs` — re-export the `cphf` constants + `Mp2Gradients`/`Mp2Reference`/`MP2_CPHF_MAX_CYCLE`; refresh the module-status doc (cphf/mp2 landed in 07-07).
- `crates/pyscf-grad/tests/cphf.rs` — NEW: 7 always-on CPHF tests incl. the GRAD-10 `single_cphf_impl` structural assertion (365 lines).
- `crates/pyscf-grad/tests/mp2_verify_fd.rs` — NEW: 6 always-on MP2 structural tests + 1 `#[ignore]`'d numeric FD arm (282 lines).

## Decisions Made

- **The MP2 Z-vector fvind uses the ENERGY `get_veff` (int2e — cintx-ready), so the Z-vector solve is RUNNABLE un-gated.** Only the gradient `de` assembly (int2e_ip1 + int1e_ip*) is cintx-gated. This is a finer split than the plan anticipated — the CPHF consumer contract is fully exercised end-to-end today, not just structurally.
- **`cphf::solve` takes a `&Fvind<'_>` trait-object (not a generic).** One concrete entry point the `single_cphf_impl` scan pins; both MP2 and CCSD pass `&|x| ...` closures. (Also dodges monomorphization-per-caller of a hot iterative solver.)
- **`single_cphf_impl` scans `src/` for `pub fn solve(` declarations** and asserts exactly one in cphf.rs — it does NOT trip on `cphf::solve(` CALL sites (path-prefixed). The pyscf-mp2 `solve_cphf_rhf` is in a different crate (D-03 scopes the single-impl to `pyscf-grad::cphf`).
- **Reductions go through `oracle_sum`/`oracle_dot`, the projected system through `solve_linear`, NOT `pyscf_algebra::gemm`.** `gemm`/`gemv` are still Phase-2 `NotYetImplemented` stubs (confirmed at read time); the established RHF/MP2 precedent routes contractions through the oracle primitives + the dense LU.
- **D-04 honored:** the variational HF/KS grads make no CPHF call; MP2 (this plan) is the first consumer, CCSD (07-08) the second.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] AlgebraError → PyscfRsError bridge via GradError::from**
- **Found during:** Task 1 (compile)
- **Issue:** `solve_linear(...)?` failed — there is no `From<AlgebraError> for PyscfRsError`; the bridge is `GradError::Algebra(#[from] AlgebraError)` then `From<GradError> for PyscfRsError`.
- **Fix:** `solve_linear(&h, &g, nd).map_err(GradError::from)?` (the same one-hop bridge the rest of the crate uses).
- **Files modified:** `crates/pyscf-grad/src/cphf.rs`
- **Verification:** `cargo test -p pyscf-grad --locked --test cphf` → 7 passed.
- **Committed in:** `d3a8e40` (Task 1 commit)

**2. [Rule 3 - Blocking] clippy type_complexity on the inner krylov aop + doc-list-item in tests**
- **Found during:** Task 1 (clippy gate)
- **Issue:** clippy `type_complexity` flagged `aop: &dyn Fn(...)`; `doc_list_item` flagged three module-doc continuation lines in tests/cphf.rs.
- **Fix:** reuse the public `&Fvind<'_>` alias for the inner `krylov` aop; reflow the doc paragraph (no list-like indentation).
- **Files modified:** `crates/pyscf-grad/src/cphf.rs`, `crates/pyscf-grad/tests/cphf.rs`
- **Verification:** `cargo clippy -p pyscf-grad --tests --locked` clean.
- **Committed in:** `d3a8e40` (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 — compile/clippy-gate blocking fixes, mechanical, no behavior change). **Impact on plan:** none — the CPHF solver + MP2 gradient land exactly as specified.

## Known Stubs

None that block the plan's goal. The `mp2::default_grad_elec` free fn (07-02 seam) is retained as a thin error-returning wrapper for any caller lacking an `Mp2Reference`; the real body is `grad_elec(refr, atmlst)` + the `Mp2Gradients` trait impl (the PyO3 bridge 07-09 always supplies a reference). The numeric MP2 FD arm is `#[ignore]`'d (cintx-gated, documented above) — intentional, un-gates with the cintx workstream.

## Issues Encountered

- `pyscf_algebra::gemm`/`gemv` remain Phase-2 `NotYetImplemented` stubs — all contractions route through `oracle_sum`/`oracle_dot` + `solve_linear` (the working algebra-wall primitives). No code change needed; recorded as a decision.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **07-08 (CCSD-grad):** consume the SAME `cphf::solve` (D-03/GRAD-10 — the `single_cphf_impl` gate forbids a second copy) with the CCSD Λ-driven `fvind` + RHS; reuse the `Fvind` callback shape, the relaxed-density→Xvo→cphf→response pattern, and the `as_rhf()`-onto-base assembly. CCSD overrides `max_cycle` per its own upstream (NOT necessarily 30).
- **07-09 (PyO3 bridge):** snapshot `mp.mo_*` + `mp.mol` + `mp.t2` into `Mp2Reference`; wire `mp.nuc_grad_method()` → `Mp2Gradients`; expose `kernel`. The free-fn forms (`grad_elec`, `response_dm1`, `make_rdm1e`) are bridge-reusable without the trait.
- **Coordination note (D-02 hinge):** the MP2 numeric arm stays `#[ignore]`'d for the six missing cintx grad-intor families; any "drop the `#[ignore]`" MUST be paired with a cintx-side availability note confirming `int2e_ip1` + `int1e_ip{ovlp,kin,nuc,rinv}` shipped. The CPHF solver + the Z-vector solve are NOT gated and need no such pairing.

## Self-Check: PASSED

- `crates/pyscf-grad/src/cphf.rs` exists (393 lines, contains `pub fn solve(` + `max_cycle` default 50).
- `crates/pyscf-grad/src/mp2.rs` exists (502 lines, contains `MP2_CPHF_MAX_CYCLE = 30` + `cphf::solve`).
- `crates/pyscf-grad/tests/cphf.rs` exists (365 lines, contains `single_cphf_impl`).
- `crates/pyscf-grad/tests/mp2_verify_fd.rs` exists (282 lines, calls `verify_fd`).
- Both task commits (`d3a8e40`, `c7f96d7`) present in git history.
- `cargo test -p pyscf-grad --locked -- --test-threads=1`: 42 passed, 5 ignored, 0 failed.
- `cargo clippy -p pyscf-grad --tests --locked`: clean. `check-dependency-wall`: PASS.

---
*Phase: 07-gradients-geomopt*
*Completed: 2026-05-26*
