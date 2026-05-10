---
phase: 01-foundation
plan: 02
subsystem: foundation
tags: [pyscf-core, types, traits, thiserror, newtype, foundation, FOUND-02, ALG-06]

# Dependency graph
requires:
  - phase: 01-foundation/01
    provides: workspace Cargo.toml with [workspace.package] + [workspace.dependencies] (thiserror, serde, tracing pinned)
provides:
  - "pyscf-core crate at crates/pyscf-core/ — universal types + method traits, zero compute deps"
  - "Mole, BasisSet, Density, MOCoefficients, Amplitudes structs (Phase-1 shells; bodies filled in Phases 2/3/6)"
  - "Energy newtype (Hartree units, proper newtype not type alias per FOUND-02)"
  - "Method, Scf, KohnSham, PostScf, Gradient, IntegralEngine traits (signatures only)"
  - "PyscfRsError + CoreError thiserror enums (project-wide error type)"
  - "Re-export shape mirroring cintx-core: 1 pub mod + 1 pub use per public name"
affects: [01-03-pyscf-runtime, 01-04-pyscf-algebra, 01-05-xtask, all-method-crates-phase-2-7]

# Tech tracking
tech-stack:
  added: [thiserror 2.0.18, serde 1.0, tracing 0.1.44]
  patterns:
    - "Re-export module shape (cintx-core analog): pub mod X per submodule + pub use X::Name per public name"
    - "Crate-level #![forbid(unsafe_code)] (xcfun-core convention) — pure types/traits, no unsafe justifiable"
    - "Energy newtype with const constructors (Energy::hartree, Energy::to_hartree) + Display impl in Hartree"
    - "Two-tier error hierarchy: PyscfRsError (public) wraps CoreError (#[from]) plus project-wide variants"
    - "NotYetImplemented variant carries phase u8 + what &'static str — every Phase-1 stub method returns this so callers see exactly which Phase fills the body"

key-files:
  created:
    - "crates/pyscf-core/Cargo.toml"
    - "crates/pyscf-core/src/lib.rs"
    - "crates/pyscf-core/src/error.rs"
    - "crates/pyscf-core/src/mole.rs"
    - "crates/pyscf-core/src/basis_set.rs"
    - "crates/pyscf-core/src/density.rs"
    - "crates/pyscf-core/src/mo.rs"
    - "crates/pyscf-core/src/amplitudes.rs"
    - "crates/pyscf-core/src/energy.rs"
    - "crates/pyscf-core/src/traits.rs"
  modified: []

key-decisions:
  - "Energy newtype is `pub struct Energy(pub f64)` (tuple-struct, not named-field) — keeps the .0 access ergonomic for hot paths while still being a distinct type the compiler refuses to coerce from raw f64"
  - "Energy ships const constructors `hartree(f64)` and `to_hartree() -> f64` even though the field is pub — gives downstream code two equally idiomatic forms (Energy(x) or Energy::hartree(x)) without committing to either"
  - "PyscfRsError::NotYetImplemented carries (phase: u8, what: &'static str) so every Phase-1 stub method (Mole::build, etc.) returns a runtime error that names the responsible Phase — keeps the type compilable today without leaving silent panics for Phase 2 to discover"
  - "IntegralEngine::intor returns Density (not a fresh Tensor type) for Phase 1 — Density is the only AO-shaped buffer that exists yet; Phase 4 (ALG-04 opaque Tensor) will refactor the return type once the algebra crate lands"
  - "PostScf::Reference is bound `: Scf` (not `: Method`) — MP2/CCSD specifically need a *converged SCF* reference, not any method. The bound enforces this at trait level so a CCSD-on-CCSD nesting can't compile"

patterns-established:
  - "Re-export hub shape: lib.rs has exactly N `pub mod X;` lines (alphabetised) followed by exactly N `pub use X::Name;` lines (alphabetised). 8 mods + 8 uses in this crate."
  - "Skeleton-Phase-1 convention: structs declared with #[derive(Debug, Default, Clone)] + pub fields; one-line doc comment per field naming which Phase fills the data; method bodies return PyscfRsError::NotYetImplemented{phase, what}"
  - "Trait dispatch ladder: Method (root) → Scf : Method → KohnSham : Scf ; PostScf : Method ; Gradient (independent). IntegralEngine is independent (Phase 2 wires)."
  - "Zero-unsafe + zero-unwrap floor for pyscf-core: #![forbid(unsafe_code)] at crate root + #![warn(clippy::unwrap_used)] flagged. Plan 05's lints will promote unwrap to deny."

requirements-completed: [FOUND-02]

# Metrics
duration: 7min
completed: 2026-05-10
---

# Phase 01 Plan 02: pyscf-core Foundation Summary

**`pyscf-core` crate shipped with universal types (Mole, BasisSet, Density, MOCoefficients, Amplitudes), Energy newtype (Hartree), and 6 method-dispatch traits (Method, Scf, KohnSham, PostScf, Gradient, IntegralEngine) — zero compute deps, only thiserror+serde+tracing.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-05-10T01:41:56Z
- **Completed:** 2026-05-10T01:48:31Z
- **Tasks:** 2
- **Files created:** 10 (Cargo.toml + 9 source files)

## Accomplishments

- Locked the public surface of pyscf-core: 5 universal types (Mole/BasisSet/Density/MOCoefficients/Amplitudes), 1 Energy newtype, 6 method-dispatch traits, 2 error enums.
- Pinned the dependency profile to **only** `thiserror`, `serde`, `tracing` — Plan 05's `xtask check-dependency-wall` will refuse any future `cubecl-*`, `cintx`, `faer`, `bytemuck`, `wgpu` addition (FOUND-02 + ALG-06 method-crate side).
- Established the cintx-core re-export shape (8 `pub mod` + 8 `pub use` in `lib.rs`) so every method crate consuming pyscf-core sees a single flat namespace.
- Verified `cargo build -p pyscf-core` and `cargo test -p pyscf-core --lib` both succeed (test harness builds with zero unit tests — Phase 1 ships type shells, not behaviour).

## Task Commits

Each task was committed atomically:

1. **Task 1: pyscf-core/Cargo.toml** — `db14fa2` (feat)
2. **Task 2: 9 source modules** — `b96c4b8` (feat)

**Plan metadata:** _(this commit)_ (docs)

## Files Created/Modified

- `crates/pyscf-core/Cargo.toml` — package manifest with `version.workspace = true` shorthand and the 3 allowed workspace deps; deny-list comment block names the forbidden compute deps.
- `crates/pyscf-core/src/lib.rs` — re-export hub. 8 `pub mod` + 8 `pub use` (alphabetised). `#![forbid(unsafe_code)]` + `#![warn(clippy::unwrap_used)]`.
- `crates/pyscf-core/src/error.rs` — `PyscfRsError` (project-wide) + `CoreError` (pyscf-core-internal) thiserror enums. `Core(#[from] CoreError)` chains them. `NotYetImplemented{phase, what}` is the Phase-1 stub error.
- `crates/pyscf-core/src/mole.rs` — `Mole` struct with the bare-minimum field set (atom_coords, atom_charges, charge, spin, nelectron). `Mole::build()` returns `NotYetImplemented{phase: 2}`. Phase 2 fills the ≥30-attribute floor (GTO-08).
- `crates/pyscf-core/src/basis_set.rs` — `BasisSet` placeholder (just `name: String`). Phase 2 (GTO-11) replaces with re-export of `cintx_core::BasisSet`.
- `crates/pyscf-core/src/density.rs` — `Density{nao, data: Vec<f64>}`. Row-major. Phase 3 wires via AlgebraClient buffer.
- `crates/pyscf-core/src/mo.rs` — `MOCoefficients{nao, nmo, data, energies, occupations}`. Column-major (LAPACK/PySCF convention, Pitfall 8).
- `crates/pyscf-core/src/amplitudes.rs` — `Amplitudes{nocc, nvir, t1, t2}`. Phase 6 wires with tensor-arena.
- `crates/pyscf-core/src/energy.rs` — `pub struct Energy(pub f64)` newtype + `HARTREE_TO_KCAL_MOL` / `HARTREE_TO_EV` consts (CODATA 2018) + `hartree()`/`to_hartree()` const fns + `Display` impl printing `"{:.10} Eh"`.
- `crates/pyscf-core/src/traits.rs` — `Method` (kernel + mol), `Scf : Method` (DensityT + density), `KohnSham : Scf` (xc), `PostScf : Method` (Reference: Scf + reference + e_correlation), `Gradient` (gradient), `IntegralEngine` (intor).

## Decisions Made

- **Energy is `pub struct Energy(pub f64)` (tuple newtype with public field) instead of `pub struct Energy { value: f64 }`.** The pub field keeps `.0` ergonomic in inner loops; the distinct type defeats accidental `f64 → energy` coercion at API boundaries. Const constructors `Energy::hartree(x)` / `Energy::to_hartree()` give callers a fluent alternative.
- **`PyscfRsError::NotYetImplemented{phase: u8, what: &'static str}` as the Phase-1 stub return.** Avoids `unimplemented!()` panics in code that would compile-and-link into method crates today; lets `cargo build` complete cleanly while runtime invocation surfaces an actionable error naming the responsible Phase.
- **`IntegralEngine::intor` returns `Density` for Phase 1.** The opaque `Tensor` type lands in Plan 04 (pyscf-algebra) — using `Density` here lets the trait compile against existing types. Phase 2's GTO-06 (cintx wrapping) will refactor the return type.
- **`PostScf::Reference: Scf` (not `: Method`).** Tightens the trait so MP2/CCSD can only correlate against an SCF reference — a CCSD-on-CCSD or MP2-on-Gradient nesting fails to compile.
- **Two-tier error hierarchy.** `PyscfRsError` (public, returned from every `kernel()`) wraps `CoreError` (pyscf-core-internal). `#[from]` makes `CoreError → PyscfRsError` automatic via `?`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Workspace manifest references three crates that do not exist yet (pyscf-runtime, pyscf-algebra, xtask)**
- **Found during:** Task 2 verification (`cargo build -p pyscf-core --locked`)
- **Issue:** Plan 01-01 listed all 15 pyscf-* members + xtask in the workspace `Cargo.toml` but only created the 12 stubs that don't conflict with parallel-wave plans 01-03/01-04/01-05. Cargo refuses to load *any* package while the workspace manifest names members whose `Cargo.toml` doesn't exist. Plan 01-01-SUMMARY explicitly anticipated and accepted this state ("Cannot smoke-test individual stubs in this worktree"). The orchestrator note in the executor prompt also confirmed: "`cargo build --workspace` will fail by design".
- **Fix:** Used a **transient, in-worktree-only** workaround to verify pyscf-core compiles standalone, *without* committing any out-of-scope changes:
  1. Snapshotted root `Cargo.toml` to `/tmp/workspace_cargo_backup.toml`.
  2. Commented out the three missing `members = [...]` entries (pyscf-runtime, pyscf-algebra, xtask).
  3. Commented out the entire `[patch.crates-io]` block (the cintx git fetch hit a broken submodule reference unrelated to my plan; bypassing the patch lets resolution complete using crates.io defaults for thiserror/serde/tracing — none of which need patching).
  4. Ran `cargo build -p pyscf-core --offline` → **`Finished`**.
  5. Ran `cargo test -p pyscf-core --lib --offline` → **`test result: ok. 0 passed; 0 failed`**.
  6. Restored root `Cargo.toml` from snapshot — `git diff Cargo.toml` is empty.
  7. Deleted the spurious `Cargo.lock` generated under the modified workspace (it would record an incomplete dependency tree).
- **Files modified:** None outside `crates/pyscf-core/` are part of the commit. The transient workspace edits and `Cargo.lock` were reverted/deleted before staging.
- **Verification:** `git status --short` after revert showed only `crates/pyscf-core/src/` (the legitimate Plan 01-02 output). `git diff Cargo.toml` was empty.
- **Committed in:** N/A (transient workaround, no out-of-scope files committed).

**Note on `--locked` flag:** The plan's verify command specifies `cargo build -p pyscf-core --locked`. That flag was used in the first attempt and failed because no `Cargo.lock` exists yet (Plan 01-01 didn't ship one — it can't, since the workspace doesn't parse). The build command actually executed was `cargo build -p pyscf-core --offline` (without `--locked`). The build succeeded with the 3 allowed workspace deps (thiserror 2.0.18, serde 1.0.228, tracing 0.1.44) resolved at the workspace-pinned exact versions via inheritance. Once Plan 01-03/04/05 land and the workspace parses, `cargo build --locked` will work normally.

---

**Total deviations:** 1 auto-fixed (Rule 3 — blocking issue with anticipated upstream cause).
**Impact on plan:** None on deliverable. The pyscf-core crate compiles, links, and exposes the locked Phase-1 surface. The `--locked` flag is deferred to post-merge wave verification, which the orchestrator owns.

## Issues Encountered

- **`[patch.crates-io]` blocks resolution due to broken submodule in cintx git remote.** When cargo evaluates the workspace's `[patch.crates-io]` block, it tries to fetch `https://github.com/BectorVoom/cintx.git`, which contains a stale submodule reference to `.claude/worktrees/agent-a01e6318` (a worktree directory from a previous run that no longer exists). Resolution fails with `no URL configured for submodule '.claude/worktrees/agent-a01e6318'`. **Out of scope for Plan 01-02** — the cintx repo's submodule hygiene is an upstream issue. Plan 01-03/04/05 will surface this again when the workspace parses; logged as a known issue for the cintx maintainer.
- **No `Cargo.lock` exists in the worktree base.** Expected — Plan 01-01 couldn't generate one (workspace unparseable). The first `cargo build -p pyscf-core --locked` attempt failed with "cannot create the lock file ... because --locked was passed". Falling back to `--offline` (no `--locked`) succeeded.

## Threat Surface Notes

Plan's `<threat_model>` lists two STRIDE entries (T-1-02 dependency-profile creep, T-1-01 unsafe code in pyscf-core), both with `mitigate` disposition.

| Threat | Mitigation Verified |
|--------|---------------------|
| T-1-02 (dependency creep) | `[dependencies]` block has exactly 3 entries (thiserror, serde, tracing); deny-list comment names every forbidden compute dep. Plan 05's `xtask check-dependency-wall` will assert this in CI. |
| T-1-01 (unsafe in pyscf-core) | `#![forbid(unsafe_code)]` at `lib.rs` line 14 — the compiler now rejects any `unsafe { ... }` block anywhere in the crate. |

No new threat surface introduced beyond the plan's register.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 01-03 (pyscf-runtime):** Can `use pyscf_core::PyscfRsError` for its error type. Will create `crates/pyscf-runtime/`, which unblocks `cargo build --workspace` for one of the three missing members.
- **Plan 01-04 (pyscf-algebra):** Can `use pyscf_core::{Density, IntegralEngine, Mole}` for the AlgebraClient API surface. Will create `crates/pyscf-algebra/`, unblocking the second missing member. Will introduce the opaque `Tensor` type that Phase 2 retrofits onto `IntegralEngine::intor`.
- **Plan 01-05 (xtask):** Will create `xtask/` and ship `xtask check-dependency-wall` — the lint that enforces Plan 02's zero-compute-deps invariant going forward.
- **Plan 01-06 (CI):** Once all three Wave-2 plans land, full-workspace `cargo build --locked` and `cargo test --workspace` will succeed and become the CI baseline.
- **Phases 2-7 (method crates):** Every method crate's `Cargo.toml` will list `pyscf-core = { path = "../pyscf-core" }`. Method impl crates will provide `impl Method for RHF`, `impl Scf for RHF`, etc.

## Self-Check: PASSED

- [x] crates/pyscf-core/Cargo.toml exists
- [x] crates/pyscf-core/src/lib.rs exists
- [x] crates/pyscf-core/src/error.rs exists
- [x] crates/pyscf-core/src/mole.rs exists
- [x] crates/pyscf-core/src/basis_set.rs exists
- [x] crates/pyscf-core/src/density.rs exists
- [x] crates/pyscf-core/src/mo.rs exists
- [x] crates/pyscf-core/src/amplitudes.rs exists
- [x] crates/pyscf-core/src/energy.rs exists
- [x] crates/pyscf-core/src/traits.rs exists
- [x] Commit db14fa2 in git log
- [x] Commit b96c4b8 in git log
- [x] All 7 must_haves.truths verified

---
*Phase: 01-foundation*
*Plan: 02*
*Completed: 2026-05-10*
