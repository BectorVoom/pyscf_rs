---
phase: 07-gradients-geomopt
plan: 02
subsystem: gradients
tags: [grad, gradients, verify_fd, atmlst, scanner, cphf, finite-difference, oracle_sum]

# Dependency graph
requires:
  - phase: 03-scf
    provides: "as_scanner pattern (SCF-12) — the Box<dyn Fn(&Mole)->Result<Energy,_>+Send+Sync> energy closure GradScanner wraps"
  - phase: 05-mp2
    provides: "Mp2Error (#[from] bridge), ump2 spin-resolved shape, rdm relaxed-density analog"
  - phase: 06-ccsd
    provides: "CcsdError (#[from] bridge), crate-skeleton + error/hooks shape, solve_lambda + make_rdm1/2 (ao_repr) GRAD-06 will consume"
  - phase: 01-foundation
    provides: "pyscf-algebra oracle_sum/oracle_dot/gemm (the reduction wall), dependency-wall lint"
provides:
  - "pyscf-grad compiling member crate with the full intra-workspace dep set (adds pyscf-grids + pyscf-dft); NO pyo3/cubecl-*/hdf5-metno"
  - "Base Gradients trait (GradientsBase analog): mol/atmlst/de/unit accessors + kernel/grad_elec/grad_nuc/make_rdm1e/hcore_generator/get_ovlp; grad_nuc + get_ovlp shared defaults"
  - "atmlst row-subsetting (GRAD-08) built into kernel from day one (D-09), bounds-checked (T-07-06)"
  - "verify_fd central-difference FD harness (GRAD-09, D-01): disp default 1e-4 Bohr, tol 1e-6 Ha/Bohr, oracle_sum reductions"
  - "GradScanner: Mole -> (Energy, de) geomopt seam (07-04 consumer)"
  - "GradError + GradOverrideHooks/NoGradOverrides + 8 NotYetImplemented method-module stubs (rhf/uhf/rks/uks/mp2/ccsd/ecp/cphf)"
affects: [07-03-rhf-grad, 07-04-geomopt, 07-05-ks-grad, 07-06-mp2-grad, 07-07-ccsd-grad, 07-08-ecp-grad, 07-09-cphf]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Base-API-from-day-one (D-09): atmlst + verify_fd live on the base contract before any method body, so every GRAD-01..07 inherits them for free"
    - "always-on FD numeric gate (D-01): central-difference the as_scanner energy, compare to analytical grad at <=1e-6 Ha/Bohr, no upstream PySCF"
    - "atmlst row-subsetting + bounds-check via resolve_atmlst free fn (GRAD-08 / T-07-06)"

key-files:
  created:
    - crates/pyscf-grad/src/lib.rs
    - crates/pyscf-grad/src/error.rs
    - crates/pyscf-grad/src/hooks.rs
    - crates/pyscf-grad/src/scanner.rs
    - crates/pyscf-grad/src/verify_fd.rs
    - crates/pyscf-grad/src/{rhf,uhf,rks,uks,mp2,ccsd,ecp,cphf}.rs
    - crates/pyscf-grad/tests/verify_fd.rs
    - crates/pyscf-grad/tests/atmlst.rs
  modified:
    - crates/pyscf-grad/Cargo.toml

key-decisions:
  - "verify_fd signature operates over per-atom coords (Fn(&[[f64;3]])->Result<f64>) not a Mole directly — method-agnostic; the per-method wave adapts its Mole scanner into this coord closure"
  - "GradScanner takes two boxed closures (EnergyClosure + GradClosure) so the seam is fixed before any method body; mirrors pyscf-scf::as_scanner Send+Sync discipline"
  - "scanner.rs + verify_fd.rs source committed in the skeleton (Task 1) so the crate compiled; the TDD test files (Task 2) are GREEN-on-arrival"

patterns-established:
  - "resolve_atmlst(atmlst, natm) -> Result<Vec<usize>, GradError>: the single GRAD-08 subsetting + T-07-06 bounds-check helper every method routes through"
  - "GradError carries Mp2Error + CcsdError #[from] bridges (grad consumes both post-SCF crates) + InvalidDisplacement (T-07-05); routes through Core(InvalidMolecule)"

requirements-completed: [GRAD-08, GRAD-09]

# Metrics
duration: 8min
completed: 2026-05-26
---

# Phase 7 Plan 02: pyscf-grad crate skeleton + verify_fd FD gate + atmlst subsetting Summary

**Compiling `pyscf-grad` member crate with the base `Gradients` trait, the always-on central-difference `verify_fd` numeric gate (D-01/GRAD-09, ≤1e-6 Ha/Bohr), `atmlst` row-subsetting (GRAD-08), and the `GradScanner` geomopt seam — all base-API-from-day-one (D-09) before any method body lands.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-26T02:06:29Z
- **Completed:** 2026-05-26T02:13:46Z
- **Tasks:** 2
- **Files modified:** 16 (1 modified, 15 created)

## Accomplishments
- Filled the 5-line `pyscf-grad` stub into a compiling member crate carrying the full intra-workspace dep set (CCSD path-dep block + `pyscf-grids` + `pyscf-dft` for the KS-grad numint/Becke seam), with NO `pyo3` / `cubecl-*` / `hdf5-metno` — dependency wall passes.
- Defined the base `Gradients` trait (the `pyscf/grad/rhf.py:265-418` `GradientsBase` analog): `mol`/`atmlst`/`de`/`unit` accessors + `kernel`/`grad_elec`/`grad_nuc`/`make_rdm1e`/`hcore_generator`/`get_ovlp`. `grad_nuc` (a real Coulomb-force port) and `get_ovlp`/`hcore_generator` (seam stubs awaiting cintx) are shared defaults; `grad_elec`/`make_rdm1e` are per-method.
- Built `atmlst` row-subsetting into `kernel` from day one (GRAD-08, D-09): `kernel(Some(&[1,2]))` returns exactly rows [1,2] (shape `(2,3)`) in `atmlst` order; out-of-range indices return `GradError::ShapeMismatch`, never an OOB panic (T-07-06).
- Shipped the `verify_fd` central-difference harness (GRAD-09, D-01): `disp` default `1e-4` Bohr, tolerance `1e-6` Ha/Bohr, every reduction through `oracle_sum` (no bare `+=`), `disp>0` finiteness validated before any energy eval (T-07-05). Proven correct against an exact-quadratic reference and proven to *flag* a wrong gradient (not a no-op).
- Shipped `GradScanner` (the 07-04 geomopt seam): wraps an energy closure + gradient closure, returns `(Energy, de)` with `de` shape `(natm,3)`, threading `atmlst` through.
- 9 tests green under `cargo test -p pyscf-grad --locked -- --test-threads=1`; clippy clean.

## Task Commits

Each task was committed atomically:

1. **Task 1: Crate skeleton — Cargo.toml + lib.rs module hub + error.rs + hooks.rs** - `73c5c64` (feat)
2. **Task 2: GradScanner + verify_fd FD harness + atmlst subsetting tests** - `531ea58` (test)

_Note: the scanner/verify_fd SOURCE landed in Task 1 (so the crate compiled); Task 2 added the TDD test files, which were GREEN-on-arrival against the shipped harness._

## Files Created/Modified
- `crates/pyscf-grad/Cargo.toml` — intra-workspace path-dep set; adds `pyscf-grids` + `pyscf-dft`; `approx` dev-dep
- `crates/pyscf-grad/src/lib.rs` — base `Gradients` trait + `resolve_atmlst` + flat `pub use` re-export hub (12 `pub mod`)
- `crates/pyscf-grad/src/error.rs` — `GradError` (+ `Mp2Error`/`CcsdError` `#[from]`, `InvalidDisplacement`) → `Core(InvalidMolecule)`
- `crates/pyscf-grad/src/hooks.rs` — `GradOverrideHooks` (`get_veff`/`extra_force`) + `NoGradOverrides`
- `crates/pyscf-grad/src/scanner.rs` — `GradScanner` (`Mole -> (Energy, de)`), `EnergyClosure`/`GradClosure` type aliases
- `crates/pyscf-grad/src/verify_fd.rs` — `verify_fd` harness, `FdReport`, `DEFAULT_DISP`/`FD_TOL` consts
- `crates/pyscf-grad/src/{rhf,uhf,rks,uks,mp2,ccsd,ecp,cphf}.rs` — `NotYetImplemented { wave }` method-module stubs
- `crates/pyscf-grad/tests/verify_fd.rs` — 4 FD-gate tests (exact-quadratic pass, wrong-grad flag, disp/shape rejection)
- `crates/pyscf-grad/tests/atmlst.rs` — 5 subsetting/scanner tests (all-rows, subset rows [1,2], order, OOB reject, grad_nuc translational invariance, scanner tuple)

## Decisions Made
- **`verify_fd` operates over per-atom coords, not a `Mole`**: the harness signature is `Fn(&[[f64;3]]) -> Result<f64, PyscfRsError>` so it stays method-agnostic and needs no Mole build in tests; the per-method wave (07-03) adapts its `Mole`-based `as_scanner` into this coord closure (clone Mole, set displaced geometry, run energy `kernel`).
- **`GradScanner` is two boxed `Send+Sync` closures** (energy + grad) rather than holding a concrete method type — fixes the geomopt seam before any method body exists, mirroring `pyscf-scf::as_scanner`'s capture-by-value discipline.
- **`grad_nuc`/`get_ovlp`/`hcore_generator` as shared trait defaults**: `grad_nuc` is a real Coulomb-force port (translationally invariant, verified); `get_ovlp` + `hcore_generator` are `NotYetImplemented { wave: 3 }` seams because they need `int1e_ipovlp` / `int1e_iprinv` + `with_rinv_at_nucleus`, which are MISSING from cintx today (07-RESEARCH D-02 / Open Q2) — the RHF wave un-gates them once the cintx workstream lands.
- **The new trait is named `Gradients` (plural)** to avoid colliding with the pre-existing `pyscf_core::Gradient` (singular) trait.

## Deviations from Plan

None — plan executed exactly as written. The two `<acceptance_criteria>` checks and the plan `<verification>` block all passed as specified.

A minor execution note (not a deviation): the plan's Task 2 `<verify>` command `cargo test -p pyscf-grad --locked -- --test-threads=1 verify_fd atmlst` treats `verify_fd`/`atmlst` as test-NAME substring filters (cargo positional filters match function names, not file names), so it ran only the one test whose name contains "atmlst". Running the full files (`--test verify_fd --test atmlst`) and the whole suite both show all 9 tests green. No code or plan change needed.

## Issues Encountered
None — both tasks compiled and passed verification on the first scoped run.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- The always-on FD numeric gate ([`verify_fd`], ≤1e-6 Ha/Bohr) is ready for every GRAD-01..07 method to wire its real `as_scanner` energy + analytical gradient against.
- The base `Gradients` trait, `GradScanner` seam, and `atmlst` subsetting are fixed contracts downstream plans implement against without re-exploration.
- **Downstream gating note (07-RESEARCH D-02):** 6 of 8 gradient-integral families (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`, `ECPscalar_iprinv` + the `with_rinv_at_nucleus` origin shift) are still MISSING from cintx. The RHF/method bodies (07-03+) can land their FD-structural form against this gate, but upstream byte-identity numeric un-gating waits on the paired cintx workstream — `hcore_generator`/`get_ovlp` are intentionally `NotYetImplemented` seams here for that reason.

### Recorded signatures (for downstream plans)
- `pub trait Gradients { fn mol(&self) -> &Mole; fn atmlst(&self) -> Option<&[usize]>; fn de(&self) -> Option<&[[f64;3]]>; fn unit(&self) -> Unit; fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64;3]>, PyscfRsError>; fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError>; fn grad_nuc(&self, atmlst: Option<&[usize]>) -> ...; fn hcore_generator(&self) -> Result<(), PyscfRsError>; fn get_ovlp(&self) -> Result<Vec<f64>, PyscfRsError>; fn kernel(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64;3]>, PyscfRsError>; }`
- `pub fn verify_fd<F: Fn(&[[f64;3]]) -> Result<f64, PyscfRsError>>(coords: &[[f64;3]], analytical: &[[f64;3]], energy: F, disp: f64, tol: f64) -> Result<FdReport, PyscfRsError>` — `DEFAULT_DISP = 1e-4`, `FD_TOL = 1e-6`; `FdReport { max_abs_diff, fd_grad, passed }`.
- `GradScanner::new(base: EnergyClosure, grad: GradClosure)` + `GradScanner::eval(&self, mol: &Mole, atmlst: Option<&[usize]>) -> Result<(Energy, Vec<[f64;3]>), PyscfRsError>`.

## Self-Check: PASSED

All 11 spot-checked files exist on disk; both task commits (`73c5c64`, `531ea58`) are present in the git log.

---
*Phase: 07-gradients-geomopt*
*Completed: 2026-05-26*
