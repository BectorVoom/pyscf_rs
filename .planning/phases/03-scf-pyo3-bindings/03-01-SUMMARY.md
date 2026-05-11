---
phase: 03-scf-pyo3-bindings
plan: 01
subsystem: infra
tags: [workspace, faer, hdf5-metno, ndarray, scf-13, pitfall-4, pitfall-12, alg-05, alg-06, scaffolding]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: pyscf-algebra surface (host_fallback pattern, AlgebraError enum, ALG-05 host-faer discipline), pyscf-core zero-compute-deps invariant (FOUND-02), xtask check-dependency-wall lint (ALG-06)
  - phase: 02-gto
    provides: pyscf-gto crate (consumed by pyscf-df dep declaration only — no Phase 2 code touched)
provides:
  - 3 new workspace member skeletons (pyscf-chkfile, pyscf-diis, pyscf-df) — workspace grew 15 → 18 pyscf-* crates
  - pyscf-algebra::solve_linear (host-faer FullPivLu wrapper) — RESEARCH Open Question 1 resolution for DIIS B-matrix solve
  - pyscf-core::canonicalize_signs (pure function, F-order) — SCF-13 Pitfall 4 + Pitfall 12 cross-platform anchor
  - AlgebraError::Singular and ::ShapeMismatch variants
  - ROADMAP.md Phase 3 progress reflects 11 plans
affects: [03-03, 03-04, 03-05, 03-06, 03-11]

# Tech tracking
tech-stack:
  added:
    - hdf5-metno=0.10.0 (workspace dep, static feature, sole-owner pyscf-chkfile per D-06/DIST-05)
    - ndarray=0.16 (workspace dep, used by pyscf-chkfile + pyscf-df typed buffers)
    - pyo3=0.28.3 (workspace dep declaration only; pyscf-py wires in Plan 03-07)
    - numpy=0.28.0 (workspace dep declaration only; pyscf-py wires in Plan 03-07)
  patterns:
    - "Sibling-crate fidelity: pyscf-{chkfile,diis,df} mirror upstream pyscf/{lib/chkfile.py, scf/diis.py, df/} respectively"
    - "Algebra-wall extension: pyscf-chkfile has ZERO algebra dep (pure HDF5 I/O); pyscf-diis + pyscf-df go THROUGH pyscf-algebra (consumer side, not carve-out side) — ALG-06 lint passes unchanged"
    - "host_fallback wrapper pattern (Phase 1 ALG-05) extended: solve_linear copies flat slice → faer Mat → solve → flat Vec; uses faer 0.24 FullPivLu (panic-free; singular detected post-solve via is_finite())"
    - "Pure function in pyscf-core preserves FOUND-02 (zero compute deps) — canonicalize_signs operates on &mut [f64] with F-order indexing, no algebra import"

key-files:
  created:
    - crates/pyscf-chkfile/Cargo.toml
    - crates/pyscf-chkfile/src/lib.rs
    - crates/pyscf-diis/Cargo.toml
    - crates/pyscf-diis/src/lib.rs
    - crates/pyscf-df/Cargo.toml
    - crates/pyscf-df/src/lib.rs
    - crates/pyscf-algebra/src/solve_linear.rs
    - crates/pyscf-algebra/tests/solve_linear.rs
    - crates/pyscf-core/src/canonicalize.rs
    - crates/pyscf-core/tests/canonicalize_signs.rs
  modified:
    - Cargo.toml (workspace.members +3, workspace.dependencies +4 entries)
    - Cargo.lock (rebuilt for new members)
    - crates/pyscf-algebra/src/error.rs (+ Singular, + ShapeMismatch variants)
    - crates/pyscf-algebra/src/lib.rs (+ pub mod solve_linear, + pub use)
    - crates/pyscf-core/src/lib.rs (+ pub mod canonicalize, + pub use)
    - .planning/ROADMAP.md (Overview "15-crate" → "18-crate"; Progress table 0/10 → 0/11)

key-decisions:
  - "Singular detection via post-solve is_finite() rather than examining U-diagonal of LU factorization — faer 0.24 FullPivLu::new returns Self (not Result), and the `Solve` trait's solve() returns Mat<T> (not Result). Post-hoc detection is simpler, correct, and matches the plan's noted contingency."
  - "AlgebraError::ShapeMismatch is a NEW variant distinct from existing DimensionMismatch — DimensionMismatch carries Vec<usize> for tensor ops; ShapeMismatch carries simple String pairs for cheap row/col flat-slice checks (no allocation churn in solve_linear's hot path)."
  - "ROADMAP.md Phase 3 progress shows 0/11 (not 0/10 as the plan template specified) — the planning-time WARNING-3 split added 03-11 (pyscf-scf kernel internals split off from 03-03) after the 03-01 plan was authored. The §Plans block already correctly listed 11 entries; only the progress-table cell was stale."

patterns-established:
  - "Host-faer LU wrapper for tiny systems: solve_linear is the first faer-LU consumer (cholesky/eigh/qr/svd were Phase 1 stubs). Pattern reuses Mat::from_fn for row-major flat-slice → Mat conversion; sets up the convention for any future small-system solves (e.g., DIIS in Plan 03-04)."
  - "Pure-function extraction from upstream PySCF inline algorithms: canonicalize_signs is the first named-and-tested extraction of an algorithm that lived only inline in pyscf/scf/hf.py (line 1349-1357 def eig). Pattern: doc comment cites upstream path:line range; tests pin numpy.argmax tie-break-to-lowest-index semantics with a regression that would fail under `>=`."
  - "Workspace member growth ritual: new crates ship with [package] using version.workspace=true / edition.workspace=true / rust-version.workspace=true / license.workspace=true (mirrors pyscf-gto/pyscf-algebra). Both lints (#![forbid(unsafe_code)] + #![warn(clippy::unwrap_used)]) applied verbatim — these are workspace-wide invariants from Phase 1."

requirements-completed: [SCF-13]

# Metrics
duration: 13min
completed: 2026-05-11
---

# Phase 3 Plan 01: Workspace Scaffolding Summary

**3 new workspace crates (pyscf-chkfile, pyscf-diis, pyscf-df) + pyscf-algebra::solve_linear (faer 0.24 FullPivLu wrapper for DIIS B-matrix) + pyscf-core::canonicalize_signs (SCF-13 cross-platform vendor-stable eigenvector signs)**

## Performance

- **Duration:** 13 min
- **Started:** 2026-05-11T12:14:17Z
- **Completed:** 2026-05-11T12:27:51Z
- **Tasks:** 4
- **Files created:** 10
- **Files modified:** 6

## Accomplishments

- **Workspace grew 15 → 18 pyscf-* crates** (D-06/D-08/D-10): pyscf-chkfile (sole owner of hdf5-metno per DIST-05), pyscf-diis (depends on pyscf-algebra only), pyscf-df (depends on pyscf-algebra + pyscf-gto). All three skeletons are doc-only stubs with #![forbid(unsafe_code)] + #![warn(clippy::unwrap_used)] inheriting Phase 1 conventions; bodies are filled by Plans 03-04, 03-05, 03-06.
- **`pyscf-algebra::solve_linear` shipped** (RESEARCH Open Question 1): host-faer 0.24 `FullPivLu` wrapper for the DIIS B-matrix solve. Handles the Lagrange-row-and-column 0-on-diagonal pattern that breaks naive Cholesky. 4 unit tests cover identity, DIIS-shape Lagrange residual, singular detection (via post-solve is_finite), and shape mismatch.
- **`pyscf-core::canonicalize_signs` shipped** (SCF-13, Pitfall 4 + Pitfall 12 anchor): pure function with no algebra dep extracted from upstream `pyscf/scf/hf.py:1349-1357 def eig`. STRICT-greater-than tie-break preserves numpy.argmax semantics (lowest-index wins on ties). 6 unit tests pin the algorithm including a tie-break regression that would silently break under `>=`.
- **AlgebraError gained Singular + ShapeMismatch variants** — the foundation for any future LU-style routine error reporting.
- **xtask check-dependency-wall passes unchanged**: the 3 new crates are method-side consumers of pyscf-algebra (not cubecl-* carve-out members), so ALG-06 is unaffected.

## Task Commits

Each task was committed atomically (TDD where applicable):

1. **Task 1: Add three workspace member skeletons (pyscf-chkfile, pyscf-diis, pyscf-df)** — `37b4a9d` (chore)
2. **Task 2: Add pyscf-algebra::solve_linear (host-faer FullPivLu wrapper)** — `4dda4fb` (feat) — TDD: tests written first as RED, implementation made GREEN in single commit (test + impl bundled per simplicity since both must land together for `cargo test` to compile)
3. **Task 3: Implement pyscf-core::canonicalize_signs (SCF-13)** — `ce1a004` (feat) — TDD: tests written first as RED, implementation made GREEN in single commit (same bundling rationale)
4. **Task 4: Update ROADMAP.md to reflect 15 → 18 workspace members** — `e44e79f` (docs)

## Files Created/Modified

### Created
- `crates/pyscf-chkfile/Cargo.toml` — sole owner of hdf5-metno (=0.10.0, static feature) per D-06/DIST-05; depends on pyscf-core + ndarray + serde_json + tracing
- `crates/pyscf-chkfile/src/lib.rs` — empty doc-only stub citing D-06 + plan 03-06 fill plan
- `crates/pyscf-diis/Cargo.toml` — depends on pyscf-core + pyscf-algebra (D-08)
- `crates/pyscf-diis/src/lib.rs` — empty doc-only stub citing D-08 + plan 03-04 fill plan + Pitfall 9 mitigation note
- `crates/pyscf-df/Cargo.toml` — depends on pyscf-core + pyscf-algebra + pyscf-gto (D-10)
- `crates/pyscf-df/src/lib.rs` — empty doc-only stub citing D-10 + plan 03-05 fill plan + D-11 in-memory-only note
- `crates/pyscf-algebra/src/solve_linear.rs` — pub fn solve_linear(a, b, n) → Result<Vec<f64>, AlgebraError>; routes to faer FullPivLu
- `crates/pyscf-algebra/tests/solve_linear.rs` — 4 tests (identity, DIIS-shape Lagrange residual, singular, shape mismatch)
- `crates/pyscf-core/src/canonicalize.rs` — pub fn canonicalize_signs(c, nao, nmo); F-order; STRICT-greater-than tie-break
- `crates/pyscf-core/tests/canonicalize_signs.rs` — 6 tests (idempotency, flip on negative leader, no-flip on positive, tie-break-to-lowest-index regression, cross-vendor reproducibility, F-order indexing)

### Modified
- `Cargo.toml` — [workspace.members] +3 entries; [workspace.dependencies] +4 entries (hdf5-metno, ndarray, pyo3, numpy)
- `Cargo.lock` — auto-rebuilt for new members
- `crates/pyscf-algebra/src/error.rs` — +AlgebraError::ShapeMismatch { expected: String, actual: String }; +AlgebraError::Singular
- `crates/pyscf-algebra/src/lib.rs` — `pub mod solve_linear; pub use solve_linear::solve_linear;`
- `crates/pyscf-core/src/lib.rs` — `pub mod canonicalize; pub use canonicalize::canonicalize_signs;`
- `.planning/ROADMAP.md` — Overview "15-crate" → "18-crate" + parenthetical citing Phase 3 growth; Progress table Phase 3 row "0/10 | Planned" → "0/11 | Planned"

## Decisions Made

1. **Singular detection via post-solve is_finite()** rather than examining the U-diagonal: faer 0.24 `FullPivLu::new` returns `Self` (not `Result`) and the `Solve` trait's `solve()` returns `Mat<T>` (not `Result`). Post-hoc detection is the cleanest path that matches the plan's noted contingency ("if FullPivLu::new returns a Result… pattern-match…otherwise rely on is_finite check"). All 4 tests pass including the singular case.

2. **AlgebraError::ShapeMismatch is a NEW variant** distinct from existing `DimensionMismatch`: DimensionMismatch carries `Vec<usize>` for tensor ops (ALG-04); ShapeMismatch carries simple `String` pairs for cheap row/col flat-slice checks (no allocation churn in solve_linear's hot path; the Vec<usize> alloc would dwarf the validation work for tiny ≤8×8 DIIS matrices).

3. **ROADMAP.md Phase 3 progress shows 0/11 (not 0/10)**: the §Phase 3 Plans block already lists 11 entries (the planning-time WARNING-3 split added 03-11 after the 03-01 plan was authored); only the progress-table cell was stale. Updating to 11 keeps ROADMAP self-consistent.

4. **TDD test-first commit bundling**: per pure TDD, RED commit and GREEN commit should be separate. Bundled them in this plan because (a) the test file imports the function under test, so RED-only commit would leave `cargo test` in a non-compiling state across CI workflows that gate on workspace test compilation, and (b) the verifier's automated check is the GREEN result. RED-then-GREEN was demonstrated in the working tree (RED confirmed via `cargo test --no-run` failing with E0432/E0599 before the impl was added) but committed as one feat commit per task.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Stale value] ROADMAP.md Progress table value 0/10 was stale**
- **Found during:** Task 4 (Update ROADMAP.md)
- **Issue:** Plan template (authored before WARNING-3 split) said the §Plans block lists 10 plans and the Progress table should read `0/10`. Reality: the §Plans block already contained 11 entries (03-11 was added by a planning-time checker iteration), but the Progress table row was still `0/10 | Planned`.
- **Fix:** Updated Progress table cell `0/10 | Planned` → `0/11 | Planned` to match the actual §Plans block. Verified consistency between Plans block (11 entries) and Progress table (0/11) and Phase 3 §Plans line text (already says "11 plans across 8 waves").
- **Files modified:** .planning/ROADMAP.md
- **Verification:** `grep -F "0/11 | Planned" .planning/ROADMAP.md` returns the row; `grep -cE "03-(0[1-9]|1[0-1])-PLAN.md"` returns 11.
- **Committed in:** e44e79f

**2. [Rule 1 - Stale value] Plan must_haves "ROADMAP.md §Phase 3 Plans list lists 10 plans"**
- **Found during:** Task 4
- **Issue:** Same root cause as #1 — the must_haves text was authored before the WARNING-3 split.
- **Fix:** Confirmed §Plans block already lists 11 plans (acceptable per `grep -E "(10|11)"` in the verify regex's broader intent); did NOT remove 03-11 since it represents real planned work. The Plan's verify command `grep -cF "03-0"` returns 9 (because 03-10 and 03-11 don't share the "03-0" prefix); the verify regex itself is buggy. A more accurate `grep -cE "03-(0[1-9]|1[0-1])-PLAN.md"` returns 11. The intent (confirm all plan entries are present) is satisfied.
- **Files modified:** none (no fix to ROADMAP needed; only documenting the pre-existing reality)
- **Verification:** All 11 plan entries 03-01..03-11 present in §Plans block.
- **Committed in:** e44e79f (commit message documents)

---

**Total deviations:** 1 substantive auto-fix (Rule 1 - stale value in Progress table)
**Impact on plan:** Cosmetic/documentation only. All four tasks shipped exactly the artifacts the plan named. The deviation was reconciling a stale ROADMAP value with the existing 11-plan reality.

## Issues Encountered

- **Cargo --locked failed initially**: Modifying Cargo.toml requires Cargo.lock regeneration, and `cargo check --locked` blocks that. Resolved by using `cargo check --offline` (without --locked) — this is appropriate because the worktree must regenerate the lock for the 3 new members. Per the critical_compile_guard in the worktree prompt, prefer `cargo check` over `cargo build`; both run cleanly here.
- **hdf5-metno static build cost**: pyscf-chkfile pulls in hdf5-metno-sys with the `static` feature, which builds libhdf5 from source. Fortunately the build artifacts were already cached from a previous worktree session, so the build completed in ~1m. First-time builds will be substantially slower — this is the intentional cost of D-06 (sole-owner static HDF5 per DIST-05).
- No build-time concerns regarding libxc_rs: per the critical_compile_guard, the libxc_rs path patch in [patch.crates-io] remains commented out throughout this plan. Verified at start AND end. Only Phase 4 (DFT) re-enables it.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Plans 03-04, 03-05, 03-06 unblocked** (their plan headers depend on 03-01's deliverables):
- **Plan 03-04 (pyscf-diis)**: skeleton crate exists; DIIS B-matrix solve via `pyscf_algebra::solve_linear` is ready to consume.
- **Plan 03-05 (pyscf-df)**: skeleton crate exists; ready for `DfIntegrals` + `cholesky_eri` body fill.
- **Plan 03-06 (pyscf-chkfile)**: skeleton crate exists with hdf5-metno wired; ready for HDF5 primitives + Checkpointable trait + scf chkfile schema.
- **Plan 03-11 (pyscf-scf kernel internals)**: `pyscf_core::canonicalize_signs` is ready for the `eig + canonicalize_signs` post-eigh consumption point.

**No blockers introduced.** The hdf5-metno build cost is a known one-time payment (next plans that build pyscf-chkfile will reuse the cached static-libhdf5 artifact).

**Verification deferred to Phase 3 closing plans:**
- SCF-13 cross-platform µHartree assertion (Linux x86_64 + macOS aarch64 matrix CI) — Plan 03-09 ships the GitHub Actions job
- ORACLE-08 chkfile round-trip oracle — Plan 03-08 wires the empirical h5py↔hdf5-metno seal

## Self-Check: PASSED

Verified:
- `[ -f crates/pyscf-chkfile/Cargo.toml ] → FOUND`
- `[ -f crates/pyscf-chkfile/src/lib.rs ] → FOUND`
- `[ -f crates/pyscf-diis/Cargo.toml ] → FOUND`
- `[ -f crates/pyscf-diis/src/lib.rs ] → FOUND`
- `[ -f crates/pyscf-df/Cargo.toml ] → FOUND`
- `[ -f crates/pyscf-df/src/lib.rs ] → FOUND`
- `[ -f crates/pyscf-algebra/src/solve_linear.rs ] → FOUND`
- `[ -f crates/pyscf-algebra/tests/solve_linear.rs ] → FOUND`
- `[ -f crates/pyscf-core/src/canonicalize.rs ] → FOUND`
- `[ -f crates/pyscf-core/tests/canonicalize_signs.rs ] → FOUND`
- Commit `37b4a9d` (Task 1) → FOUND in `git log`
- Commit `4dda4fb` (Task 2) → FOUND in `git log`
- Commit `ce1a004` (Task 3) → FOUND in `git log`
- Commit `e44e79f` (Task 4) → FOUND in `git log`
- libxc_rs patch line still commented out in root Cargo.toml → VERIFIED

---
*Phase: 03-scf-pyo3-bindings*
*Completed: 2026-05-11*
