---
phase: 01-foundation
plan: 08
subsystem: infra
tags: [cargo, cargo-deny, ndarray, numpy, hdf5-metno, cintx, lockfile, supply-chain]

# Dependency graph
requires:
  - phase: 01-01
    provides: workspace manifest + [patch.crates-io] + [profile.release-oracle]
  - phase: 01-04
    provides: cargo build --workspace --locked end-to-end target
  - phase: 01-09
    provides: transitive-aware check-cubecl-pin lint logic
provides:
  - Consistent, committed Cargo.lock — every --locked CI job can run on a fresh clone
  - cargo deny check exits 0 for the first time (FOUND-10) under the local path-dep topology
  - End-to-end green proof of all 5 xtask gates against a resolvable workspace
affects: [phase-02, phase-03, phase-04, phase-05, ci, dependency-policy]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Local path-dep workspace topology (cintx/libxc_rs/xcfun_rs) instead of git-dep pins"
    - "deny.toml policy reconciled to a path-dep workspace (allow-wildcard-paths, clarify, private.ignore)"

key-files:
  created:
    - .planning/phases/01-foundation/01-08-SUMMARY.md
  modified:
    - Cargo.toml (ndarray 0.16->0.17; hdf5-metno 0.10.0->0.12.4)
    - Cargo.lock (regenerated to a single unified ndarray 0.17.2 graph; committed/tracked)
    - deny.toml (FOUND-10 reconciliation — wildcards, licenses, clarify, advisory ignore)

key-decisions:
  - "Plan premise was obsolete on arrival: commit 4b9cb98 had already closed BLOCKER 1+3 by swapping cintx/libxc_rs/xcfun_rs to LOCAL PATH DEPS and committing Cargo.lock — superseding the plan's 'find a clean cintx SHA and rev-pin it' approach (Task 1/Task 3). cintx is now `path = \"../cintx\"`, so the contaminated-git-branch root cause is structurally eliminated, not patched."
  - "Task 2 human checkpoint (clean-SHA discovery) was moot — no cintx git fetch occurs anymore. Multiple human decision points were taken during close-out instead (verify-vs-skip, numpy strategy, deny.toml policy, ndarray strategy)."
  - "Verification surfaced a numpy 0.28 <-> ndarray version skew (out of the plan's cintx/Cargo.lock scope). Resolved by unifying the workspace on ndarray 0.17 (matches numpy 0.28) and bumping hdf5-metno 0.10.0->0.12.4 (first release accepting ndarray <=0.17), giving a SINGLE ndarray 0.17.2 graph. No pyscf source change was needed."
  - "FOUND-10: cargo deny had never actually passed (always masked by the un-runnable lockfile). Reconciled deny.toml to the path-dep topology rather than editing external cintx/xcfun repos."
  - "ndarray strategy chosen by user: bump to 0.17 + hdf5-metno 0.12.4 (rather than keeping 0.16 + a numpy-boundary shim). Both build; the 0.17 unification avoids a dual-version graph."

patterns-established:
  - "deny.toml clarify-from-ground-truth: first-party path-dep crates without a license field are clarified from their actual repo LICENSE (cintx=MIT) or sibling consensus (xcfun-core=MPL-2.0), not fabricated."
  - "wildcards=warn for path-dep workspaces: pathless inter-crate path deps are benign; registry-wildcard supply-chain risk stays covered by [sources] + explicit deny list."

requirements-completed: [FOUND-01, FOUND-04, FOUND-10]

# Metrics
duration: ~90min
completed: 2026-05-23
---

# Phase 1 (Foundation) — Plan 08 Summary

**Closed Phase-1 BLOCKERs 1 + 3 (and confirmed BLOCKER 2 end-to-end): the workspace now builds `--locked` on a consistent committed Cargo.lock and `cargo deny check` exits 0 — achieved by a superseding local-path-dep topology plus an ndarray 0.17 / hdf5-metno 0.12.4 unification, not by the plan's original cintx rev-pin.**

## Performance

- **Duration:** ~90 min (verification-heavy; multiple human checkpoints + a concurrent-edit reconciliation)
- **Completed:** 2026-05-23
- **Tasks:** Plan executed as a verify-and-close (premise superseded — see below), not task-by-task as written
- **Files modified:** 3 (Cargo.toml, Cargo.lock, deny.toml)

## Why this plan deviated from its written form

Plan 08 was authored 2026-05-10 to fix two blockers by **repointing the cintx `[patch.crates-io]` entry from `branch = "main"` to a clean `rev = "<sha>"`** and then generating Cargo.lock. By the time it executed, commit **`4b9cb98`** had already closed both blockers a different (cleaner) way:

| | Plan 08 prescribed | Reality at execution |
|---|---|---|
| cintx | walk git history for a clean SHA, rev-pin it | `cintx = { path = "../cintx" }` (local path dep) |
| Cargo.lock | `cargo generate-lockfile` + commit | already tracked (via 4b9cb98) |

So Task 1 (clean-SHA discovery) and Task 3 (edit `branch = "main"`) were moot, and the Task 2 human checkpoint never fired. This plan was therefore executed as a **verify-and-close**: prove the blockers are actually closed in the current state, fix anything verification surfaces, and record the outcome.

## What verification found and fixed

1. **BLOCKER 1 (Cargo.lock missing) — CLOSED.** Cargo.lock is tracked and now internally consistent. HEAD's committed lock was stale (held only ndarray 0.16.1 and was rejected by `--locked`); a clean regeneration produced a unified graph.
2. **BLOCKER 3 (cintx contamination) — CLOSED.** cintx resolves as a local path dep; dep resolution never touches the contaminated git branch. `cargo build --workspace --locked` reaches and compiles all workspace members.
3. **BLOCKER 2 (check-cubecl-pin) — CONFIRMED end-to-end** (was lint-logic-only per Plan 09): the gate now runs against a resolvable graph and PASSes.
4. **Out-of-scope build break surfaced + fixed (collaboratively):** the committed lock + numpy 0.28 produced an `ndarray` version skew (`pyscf-py`/`numpy_io.rs`, then `pyscf-chkfile`/`primitives.rs`). Root cause: numpy 0.28 needs ndarray 0.17, hdf5-metno 0.10 needs 0.16. Resolved by moving the workspace `ndarray` pin to 0.17 and **hdf5-metno to 0.12.4** (first release accepting ndarray `<=0.17`) → a single unified `ndarray 0.17.2` graph. No pyscf source change required.
5. **FOUND-10 (cargo deny) — CLOSED.** `cargo deny` had never been runnable before; once it was, it surfaced 35 policy violations rooted in the path-dep migration. `deny.toml` reconciled (see commit `c4987af`).

## Acceptance proofs (current state)

| Gate | Result |
|------|--------|
| `cargo build --workspace --locked` | ✓ Finished (exit 0) |
| `check-no-fma` | ✓ PASS — no FMA mnemonics in release-oracle asm (FOUND-05) |
| `check-forbidden-paths` | ✓ PASS — 220 .rs files (FOUND-08) |
| `check-catch-unwind` | ✓ PASS — 220 .rs files (FOUND-07) |
| `check-dependency-wall` | ✓ PASS — cubecl-* containment intact (ALG-06) |
| `check-cubecl-pin` | ✓ PASS — 6 @0.10.0, 2 @0.9.0-pre.5, 8 @0.9.0-pre.5 transitive (FOUND-04) |
| `cargo deny check` | ✓ advisories ok, bans ok, licenses ok, sources ok (FOUND-10) |
| Regression suites | ✓ oracle_determinism 5, algebra/select_backend 7, backend_matrix 2, cubecl_matmul_smoke 1, runtime/select_backend 7 |

## Task Commits

1. **Dependency fix (BLOCKER 1 + 3)** — `386085b` (build) — unify ndarray 0.17 + hdf5-metno 0.12.4; commit consistent Cargo.lock
2. **deny.toml reconciliation (FOUND-10)** — `c4987af` (build) — wildcards/licenses/clarify/advisory for the path-dep topology

_(My interim `numpy_io.rs` boundary shim was reverted once the workspace unified on ndarray 0.17 made it unnecessary — net zero pyscf source change.)_

## Files Created/Modified
- `Cargo.toml` — `ndarray` 0.16→0.17; `hdf5-metno` 0.10.0→0.12.4
- `Cargo.lock` — regenerated to a single unified ndarray 0.17.2 graph; tracked
- `deny.toml` — FOUND-10 reconciliation (allow-wildcard-paths, wildcards=warn, +HDF5/CDLA-Permissive-2.0/BSL-1.0, clarify cintx=MIT + xcfun-core=MPL-2.0, ignore RUSTSEC-2024-0436)

## Notes for future maintainers
- **cintx rev-pin vs path-dep:** the plan's D-12 rev-pin contract is moot while cintx is a path dep. If cintx ever returns to a git dep, the contaminated `.claude/worktrees/agent-*` 160000-mode gitlink issue resurfaces — re-read this plan's original `<interfaces>` block.
- **ndarray/numpy coupling:** numpy is pinned `=0.28.0` (→ ndarray 0.17) and hdf5-metno `=0.12.4` (accepts ≤0.17). Keep these moving together; a numpy bump that demands ndarray 0.18 will reopen the skew unless hdf5-metno supports it.
- **deny.toml `wildcards = "warn"`:** intentional for the path-dep workspace. The external cintx/xcfun crates are `public` (no `publish = false`), so `allow-wildcard-paths` cannot exempt their path-dep wildcards — `warn` is the pragmatic lever. Adding `publish = false` upstream would let this return to `deny`.
