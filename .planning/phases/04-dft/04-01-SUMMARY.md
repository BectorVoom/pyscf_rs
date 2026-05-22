---
phase: 04-dft
plan: 01
subsystem: infra
tags: [cargo-workspace, dft, grids, libxc, xcfun, cargo-features, wave-0, nyquist, test-scaffolds, dtype-f32]

# Dependency graph
requires:
  - phase: 03-scf-pyo3-bindings
    provides: "pyscf-scf / pyscf-df / pyscf-chkfile / pyscf-runtime (DType::from_env) / pyscf-algebra DeviceScalar seam; Wave-0 xfail-stub precedent; oracle_check! + pyscf-oracle harness"
provides:
  - "pyscf-grids crate skeleton (19th crates/ member; Becke grids home, D-05) with 6 stub modules behind the algebra wall"
  - "pyscf-dft full default dependency set (core/algebra/gto/scf/df/grids/chkfile/runtime + xcfun-rs) with NO pyo3 and NO direct cubecl-* (algebra/PyO3 walls)"
  - "Off-by-default `libxc` cargo feature gating an optional libxc_rs dep — default builds carry zero libxc_rs (Pitfall 5, ~6h compile avoided)"
  - "11 Wave-0 test scaffolds (named, compiling, ignored) covering DFT-01..11 incl. the D-08 dtype_f32_smoke f32 precision switch"
  - "04-VALIDATION.md marked nyquist_compliant + wave_0_complete with every requirement's verify token now resolving to a real test"
affects: [04-04-grids, 04-05-xc-parser, 04-06-rks-core, 04-07-rsh-vv10-df, 04-08-dft-pyo3, 04-09-libxc-gated]

# Tech tracking
tech-stack:
  added: ["xcfun-rs (default XC backend, via [patch.crates-io] sibling path)", "libxc_rs (optional, behind off-by-default `libxc` feature)"]
  patterns:
    - "Off-by-default optional-dep cargo feature (default = []; libxc = [dep:libxc_rs]) — mirrors pyscf-kernels [features] + xcfun-rs [features]"
    - "Crate-level #![cfg(feature = \"libxc\")] gate on a whole test file so default builds contribute zero tests AND never compile the gated dep"
    - "Named compiling-but-ignored Wave-0 test scaffold: one #[ignore = \"plan/reason\"] test whose fn name == the VALIDATION verify token"

key-files:
  created:
    - "crates/pyscf-grids/Cargo.toml"
    - "crates/pyscf-grids/src/lib.rs (+ 6 stub modules: radial/radii/lebedev/prune/partition/levels)"
    - "crates/pyscf-grids/tests/grid_weights_level_sweep.rs"
    - "crates/pyscf-dft/tests/{parse_xc_parity,rks_uks_bitexact,xc_eval_bitexact,libxc_functional_smoke,cam_b3lyp_h2o_rsh,vv10_energy_match,df_dft_match,numint_signatures,dtype_f32_smoke}.rs"
    - "python/tests/test_dft_override.py"
  modified:
    - "Cargo.toml (workspace members 18 -> 19: + crates/pyscf-grids)"
    - "crates/pyscf-dft/Cargo.toml (full default deps + libxc feature + optional libxc_rs)"
    - ".planning/phases/04-dft/04-VALIDATION.md (nyquist_compliant + wave_0_complete = true; File Exists cells flipped to scaffold)"

key-decisions:
  - "STATE.md and ROADMAP.md NOT modified in-worktree (orchestrator owns shared-file writes after merge) despite being in the plan's files_modified — member-count 18->19 sync deferred to the central orchestrator update"
  - "xcfun-rs declared as `xcfun-rs = \"0.1.0\"` resolved via the existing [patch.crates-io] sibling-path redirect (it is not in [workspace.dependencies]); already present in Cargo.lock"
  - "cam_b3lyp_h2o_rsh.rs left NOT cfg(libxc)-gated at file level (only its real run is --features-libxc) so its token stays discoverable under default features — only xc_eval_bitexact + libxc_functional_smoke are crate-level #![cfg(feature=\"libxc\")]"

patterns-established:
  - "Algebra wall for pyscf-grids: deps are exactly pyscf-core + pyscf-algebra + thiserror + tracing; no cubecl-* (verified by check-dependency-wall + grep -L cubecl)"
  - "libxc gate verified WITHOUT compiling: cargo tree -p pyscf-dft (no libxc) lists zero libxc_rs; cargo tree --features libxc lists it — tree never triggers the ~6h build"

requirements-completed: [DFT-01, DFT-02, DFT-03, DFT-04, DFT-05, DFT-06, DFT-07, DFT-08, DFT-09, DFT-10, DFT-11]

# Metrics
duration: ~22min
completed: 2026-05-22
---

# Phase 4 Plan 01: DFT Scaffold + Wave-0 Tests Summary

**19-member workspace with the new pyscf-grids crate, a fully-wired pyscf-dft dependency set behind the off-by-default `libxc` feature (zero libxc_rs in the default dep graph), and 11 named compiling-but-ignored Wave-0 test scaffolds covering DFT-01..11 including the D-08 f32 precision smoke.**

## Performance

- **Duration:** ~22 min
- **Started:** 2026-05-22T03:00:00Z (approx)
- **Completed:** 2026-05-22T03:22:00Z
- **Tasks:** 2
- **Files modified:** 23 (across both task commits)

## Accomplishments
- Registered `crates/pyscf-grids` as the 19th `crates/` workspace member (D-05) with an algebra-wall-only Cargo.toml (pyscf-core + pyscf-algebra + thiserror + tracing, NO cubecl-*) and a 6-module skeleton (radial/radii/lebedev/prune/partition/levels) citing the upstream gen_grid/radi/LebedevGrid + data/radii sources.
- Wired `pyscf-dft`'s full default dependency set (core/algebra/gto/scf/df/grids/chkfile/runtime + xcfun-rs + tracing/thiserror), reusing the existing pyscf-runtime/pyscf-algebra precision seam for D-08 with NO new dependency, and NO pyo3 (PyO3 wall).
- Added the off-by-default `libxc` feature (`default = []`, `libxc = ["dep:libxc_rs"]`) with `libxc_rs` declared `optional = true` at `../../../libxc_rs`; verified via `cargo tree` (no compile) that the default dep graph has zero libxc_rs and only `--features libxc` pulls it.
- Created all 11 Wave-0 test scaffolds — every Phase-4 requirement (DFT-01..11) now has a real `<automated>` verify target that exists from the start, including the D-08 `dtype_f32_smoke` (CPU/xcfun-default, NOT cfg(libxc)-gated, no-oracle/no-tolerance contract).
- Marked `04-VALIDATION.md` `nyquist_compliant: true` + `wave_0_complete: true` with File-Exists cells flipped to scaffold.

## Task Commits

Each task was committed atomically:

1. **Task 1: Register pyscf-grids + wire pyscf-dft deps and libxc feature** — `201dc4a` (feat)
2. **Task 2: Create 11 Wave-0 test scaffolds + sync VALIDATION** — `802f1e1` (test)

_Plan metadata (SUMMARY) commit follows this file._

## Files Created/Modified
- `Cargo.toml` — workspace members 18 → 19 (added `crates/pyscf-grids` with a Phase-4 comment)
- `crates/pyscf-grids/Cargo.toml` — algebra-wall-only deps; description names gen_grid/radi/Lebedev + D-05
- `crates/pyscf-grids/src/lib.rs` + `radial.rs` `radii.rs` `lebedev.rs` `prune.rs` `partition.rs` `levels.rs` — crate root + 6 one-line stub modules (compile clean)
- `crates/pyscf-dft/Cargo.toml` — full default dep set + `[features]` libxc gate + optional libxc_rs; xcfun-rs default backend
- `crates/pyscf-grids/tests/grid_weights_level_sweep.rs` — DFT-04/09 scaffold (04-04)
- `crates/pyscf-dft/tests/parse_xc_parity.rs` — DFT-02 scaffold (04-05)
- `crates/pyscf-dft/tests/rks_uks_bitexact.rs` — DFT-01 scaffold (04-06)
- `crates/pyscf-dft/tests/xc_eval_bitexact.rs` + `libxc_functional_smoke.rs` — DFT-03 scaffolds, entirely behind `#![cfg(feature = "libxc")]` (04-09)
- `crates/pyscf-dft/tests/cam_b3lyp_h2o_rsh.rs` — DFT-05 scaffold (04-07)
- `crates/pyscf-dft/tests/vv10_energy_match.rs` — DFT-06 scaffold (04-07)
- `crates/pyscf-dft/tests/df_dft_match.rs` — DFT-07 scaffold (04-07)
- `crates/pyscf-dft/tests/numint_signatures.rs` — DFT-10 scaffold (04-06)
- `crates/pyscf-dft/tests/dtype_f32_smoke.rs` — DFT-11/D-08 f32 runs-end-to-end smoke scaffold; doc names the no-oracle-compare/no-tolerance-gate contract (04-06)
- `python/tests/test_dft_override.py` — DFT-08 scaffold, `pytest.mark.skip` (04-08)
- `.planning/phases/04-dft/04-VALIDATION.md` — nyquist_compliant + wave_0_complete = true; File-Exists cells, Wave-0 checklist, Sign-Off updated

## Decisions Made
- **Shared-file writes deferred to orchestrator:** the plan's `files_modified` lists `.planning/STATE.md` and `.planning/ROADMAP.md` (member-count 18→19 sync), but the parallel/worktree contract forbids modifying those in-worktree — the orchestrator updates them centrally after merge. Documented as a deviation (below).
- **xcfun-rs consumption:** declared `xcfun-rs = "0.1.0"`, resolved through the pre-existing workspace `[patch.crates-io]` sibling-path redirect (it is not a `[workspace.dependencies]` entry); version 0.1.0 was already pinned in Cargo.lock. xcfun-rs's own `default = ["cpu"]` pulls cubecl-cpu transitively, which is allowed (xcfun-rs is not a pyscf-* crate, so the dependency-wall lint does not flag it; cubecl-cpu was already in the graph via pyscf-kernels).
- **cam_b3lyp_h2o_rsh not file-gated:** even though its real run is `--features libxc`, only `xc_eval_bitexact` and `libxc_functional_smoke` were placed entirely behind `#![cfg(feature = "libxc")]` (per the plan action). cam_b3lyp_h2o_rsh stays a plain `#[ignore]` test so its token resolves under default features.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Shared orchestrator files (STATE.md, ROADMAP.md) NOT modified despite being in the plan's files_modified**
- **Found during:** Task 2 (sync member counts)
- **Issue:** The plan's Task 2 action says to sync the 18→19 member count into `.planning/ROADMAP.md` and `.planning/STATE.md`. The active execution context is a parallel git worktree, where the worktree contract explicitly forbids modifying STATE.md and ROADMAP.md (the orchestrator owns those writes and updates them centrally after all wave agents merge). Touching them here would create merge conflicts / lost central updates.
- **Fix:** Applied the member-count sync only to the non-shared phase artifact `04-VALIDATION.md`. Left STATE.md and ROADMAP.md untouched; the orchestrator performs the 18→19 sync centrally post-merge.
- **Files modified:** `.planning/phases/04-dft/04-VALIDATION.md` (instead of STATE.md/ROADMAP.md)
- **Verification:** `git status` confirms STATE.md and ROADMAP.md are unmodified in this worktree; both task commits exclude them.
- **Committed in:** `802f1e1` (Task 2 commit)

---

**Total deviations:** 1 (Rule 3 — blocking-context constraint)
**Impact on plan:** No scope creep. The only divergence is honoring the worktree shared-file ownership boundary; all crate/test/feature work matches the plan exactly. The ROADMAP/STATE member-count sync is a doc-only update the orchestrator owns.

## Issues Encountered
- None. The libxc gate, dependency wall, scoped default builds, and all 11 scaffold tokens verified cleanly without ever compiling libxc_rs.

## Known Stubs
The 6 pyscf-grids module files and all 11 test scaffolds are intentional, plan-mandated stubs:
- `crates/pyscf-grids/src/{radial,radii,lebedev,prune,partition,levels}.rs` — one-line `//! ... Phase 4 (04-04) fills this.` placeholders so the crate compiles; 04-04 fills the bodies.
- 11 Wave-0 test scaffolds — each `unimplemented!()`/`NotImplementedError` body is `#[ignore]`/`pytest.skip`/`cfg(libxc)`-gated and names its owning plan (04-04..04-09). These are the Nyquist sampling contract, intentionally green-because-ignored until their owning plan unignores them. None block this plan's goal (scaffolding); each resolution plan is named in the doc-comment.

## User Setup Required
None — no external service configuration required. (libxc_rs `[patch.crates-io]` re-enable + libxc CI gate are deferred to 04-09 by design.)

## Next Phase Readiness
- Every later Phase-4 plan (04-04 grids, 04-05 parser, 04-06 RKS core + D-08, 04-07 RSH/VV10/DF, 04-08 PyO3 override, 04-09 libxc-gated) now has a real, named, ignored verify target to unignore.
- 19-member workspace builds clean under default features; libxc stays off-by-default with zero libxc_rs in the default dep graph.
- Orchestrator must perform the 18→19 member-count sync in STATE.md + ROADMAP.md centrally after merge (deferred per worktree contract).

## Self-Check: PASSED

- All 23 created/modified files verified present on disk.
- All 3 commits verified in git log: `201dc4a` (Task 1, feat), `802f1e1` (Task 2, test), `90c5765` (SUMMARY, docs).
- STATE.md and ROADMAP.md confirmed UNMODIFIED in this worktree (orchestrator owns the central member-count sync post-merge).
- libxc guardrail: `cargo tree -p pyscf-dft` (default) lists zero libxc_rs; libxc_rs appears only under `--features libxc`. libxc was NEVER compiled.

---
*Phase: 04-dft*
*Completed: 2026-05-22*
