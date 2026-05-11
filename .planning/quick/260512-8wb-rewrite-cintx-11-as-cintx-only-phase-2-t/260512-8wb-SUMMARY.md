---
quick_id: 260512-8wb
type: quick
description: "Rewrite cintx#11 as cintx-only Phase 2 task list (drop pyscf_rs framing)"
status: complete
date: 2026-05-11
---

# Quick Task 260512-8wb: Summary

## Result

Issue [BectorVoom/cintx#11](https://github.com/BectorVoom/cintx/issues/11)
edited in place — body and title rewritten as a cintx-only feature request
covering the same three tasks.

- **Old title:** *pyscf_rs Phase 2 follow-ups: int1e_ecp operators, arity ≥3 safe-API dispatch, real-integral evaluation*
- **New title:** *Safe API: extend SessionRequest with ECP operators, arity ≥3 dispatch, and real-integral evaluation*
- **State:** OPEN (unchanged), label `enhancement` (unchanged)

## What changed in the issue body

- **Removed:** every "blocks pyscf_rs plan X-Y" line, every link into
  `pyscf_rs/.planning/...`, every reference to pyscf_rs file anchors
  (`crates/pyscf-gto/...`, `crates/pyscf-core/...`), the closing
  *"Why this matters / who's blocked"* table that was a pyscf_rs gap-closure
  ledger.
- **Reframed:** title and each ask now describe what cintx is building,
  not what pyscf_rs is waiting on.
- **Added:** complexity-ordering table (smallest → largest), since the
  cross-project blocking ordering is gone. Task 3 is flagged as the smallest
  isolated PR.
- **Kept:** all three real-task anchors (`fill_staging_values` synthetic
  pattern, `cintx-ops` arity-3/4 catalog, missing `int1e_ecp` symbols),
  re-verified against cintx `master` at edit time.

## Verification

- `gh issue view 11 --repo BectorVoom/cintx --json title,state` confirms
  new title, state OPEN.
- New body file contains zero `pyscf_rs` substring matches.
- cintx `master` state re-checked before edit:
  - `gh search code --repo BectorVoom/cintx 'int1e_ecp'` → 0 matches (Ask 1 still real).
  - `crates/cintx-rs/src/api.rs::fill_staging_values` still has synthetic pattern (Ask 3 still real).

## What this task did NOT do

- Did not close cintx#11 (issue remains OPEN, just rewritten).
- Did not create a new GitHub issue (no duplicate filed).
- Did not change cintx labels or assignees.
- Did not touch pyscf_rs source code, ROADMAP.md, or phase artifacts.
- Did not bump the cintx pin in pyscf_rs `Cargo.toml`.

## Commits

This quick task does not touch source code — only the planning artifacts
(PLAN.md, SUMMARY.md) and STATE.md row are committed. The cintx#11 edit is
a remote GitHub mutation, not a local commit.
