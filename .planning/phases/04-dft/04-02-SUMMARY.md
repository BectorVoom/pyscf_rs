---
phase: 04-dft
plan: 02
subsystem: infra
tags: [libxc, xc-functionals, cross-crate-coordination, cargo-features, dft]

# Dependency graph
requires:
  - phase: 02-gto
    provides: cintx-ECP coordination precedent (cintx#11, status-marker tracking of sibling-repo work)
provides:
  - "Coordination decision: libxc_rs per-functional feature-gate workstream deferred (PENDING_LIBXC_RS_FEATURE_GATE) by user checkpoint"
  - "Confirmation that the xcfun-default DFT path (04-04..04-08) is independent of the libxc gate and proceeds regardless"
affects: [04-05, 04-06, 04-09, 04-10, libxc_rs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sibling-repo coordination tracked via status marker (PENDING_*) like the Phase 2 cintx-ECP precedent"

key-files:
  created:
    - .planning/phases/04-dft/04-02-SUMMARY.md
  modified:
    - .planning/STATE.md

key-decisions:
  - "User checkpoint decision (2026-05-22): KEEP PENDING — do not modify the sibling ~/Documents/workspace/libxc_rs repo this phase; ship the xcfun-default DFT path independently."
  - "The ~6h libxc compile is NEVER triggered; the libxc feature-gate is a separate cross-repo workstream (libxc_rs's own PR/issue)."

patterns-established:
  - "PENDING_LIBXC_RS_FEATURE_GATE: the --features libxc bit-exact path stays cfg-gated/CI-only until the sibling-repo feature gate lands."

requirements-completed: []  # DFT-03 (bit-identical XC eval) is delivered via the xcfun-default backend in 04-05; the libxc-side bit-exact gate remains coordination-pending here.

# Metrics
duration: <1min
completed: 2026-05-22
---

# Phase 04: dft — Plan 02 Summary

**libxc_rs per-functional feature-gate coordination deferred by user checkpoint (PENDING_LIBXC_RS_FEATURE_GATE); xcfun-default DFT path ships independently, ~6h libxc compile never triggered.**

## Performance

- **Duration:** <1 min (checkpoint resolution only — no implementation)
- **Completed:** 2026-05-22
- **Tasks:** 1/3 (checkpoint resolved to "keep pending"; Task 1 sibling-repo edits intentionally NOT executed; Task 2 status recorded)
- **Files modified:** 1 (STATE.md coordination note) + this SUMMARY

## Accomplishments
- Resolved the `checkpoint:human-verify` gate: user selected **keep pending**.
- Recorded the libxc_rs feature-gate coordination status in STATE.md (Blockers/Concerns), mirroring the Phase 2 cintx-ECP (cintx#11) precedent.
- Confirmed and documented that the xcfun-default DFT path (04-04 → 04-08) does not depend on this plan and proceeds.

## Task Commits

1. **Checkpoint: confirm libxc_rs feature-gate workstream readiness** — resolved to "approved: keep pending" (no commit; human gate).
2. **Task 1: Add [features] block + cfg-gated dispatch to libxc_rs** — INTENTIONALLY NOT EXECUTED (deferred per checkpoint). No sibling-repo edits made.
3. **Task 2: Record coordination status in STATE.md** — committed with this SUMMARY.

## Files Created/Modified
- `.planning/phases/04-dft/04-02-SUMMARY.md` — this disposition record.
- `.planning/STATE.md` — Blockers/Concerns entry recording PENDING_LIBXC_RS_FEATURE_GATE.

## Decisions Made
- **Keep PENDING** (user checkpoint, 2026-05-22): the libxc_rs `[features]`/cfg-gating workstream is a separate cross-repo task (its own PR/issue). Not implemented this phase.
- Rationale: the deliverable xcfun-default DFT path is independent; deferring avoids cross-repo churn during the parallel wave and matches the cintx-ECP precedent.

## Deviations from Plan
None — the plan explicitly provides the "keep pending" branch (resume-signal "approved: keep pending"); this is the planned deferral path, not a deviation.

## Issues Encountered
None.

## User Setup Required
None.

## Next Phase Readiness
- The `--features libxc` bit-exact assertions in 04-05 / 04-06 / 04-09 and the dedicated libxc CI job in 04-10 MUST remain `#[cfg(feature = "libxc")]`-gated and CI-only (never compiled in default/local verify) while this stays PENDING.
- Future: when the sibling libxc_rs repo grows per-functional features, revisit this plan to flip `PENDING_LIBXC_RS_FEATURE_GATE` → done and enable the gated bit-exact CI surface.

---
*Phase: 04-dft*
*Completed: 2026-05-22*
