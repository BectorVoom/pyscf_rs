---
quick_id: 260512-8jv
type: quick
description: "Create issue in cintx repository about remaining tasks from pyscf_rs Phase 2"
status: complete
date: 2026-05-11
---

# Quick Task 260512-8jv: Summary

## Result

GitHub issue created on BectorVoom/cintx:

- **URL:** https://github.com/BectorVoom/cintx/issues/11
- **Title:** pyscf_rs Phase 2 follow-ups: int1e_ecp operators, arity ≥3 safe-API dispatch, real-integral evaluation
- **Label:** `enhancement`

## What the issue covers

Three independent asks against cintx that pyscf_rs Phase 2 verification surfaced
as deferred-to-cintx-workstream (per Phase 2 design decision D-06):

1. **`int1e_ecp_*` operator family** (Type-1 + Type-2 ECP projectors) — unblocks
   pyscf_rs plan `02-10` (currently `status: PENDING_CINTX_ECP_MERGE`) and
   GTO-05 evaluation half.
2. **Arity ≥3 `SessionRequest` dispatch** in `cintx-rs` safe API — unblocks
   pyscf_rs plan 03-05 (DF-HF), Phase 7 gradients, and the arity-3/4 xfail
   tests.
3. **Real-integral evaluation in safe API** — replaces
   `cintx-rs::api.rs:465 fill_staging_values` synthetic output with real
   `cintx-compat::raw::eval_raw` dispatch, removing pyscf_rs's compat workaround.

## How the asks were grounded

Verified before drafting (2026-05-11):
- No `int1e_ecp` symbol in `cintx/crates/*/src/*.rs` (confirms Ask 1 is real).
- `cintx-ops/src/generated/api_manifest.csv:21-25` does catalog arity-3/4
  operators but `cintx-rs::SessionRequest` only dispatches arity-2 (confirms
  Ask 2 is real).
- `cintx-rs/src/api.rs:465 fn fill_staging_values` still in place with synthetic
  pattern (confirms Ask 3 is real).

## Source pointers in the issue body

- pyscf_rs Phase 2 verification: `.planning/phases/02-gto/02-VERIFICATION.md`
- Phase 2 validation table: `.planning/phases/02-gto/02-VALIDATION.md`
- Phase 2 context D-06: `.planning/phases/02-gto/02-CONTEXT.md`
- Gap-closure plan: `.planning/phases/02-gto/02-10-PLAN.md`
- pyscf_rs code anchors: `crates/pyscf-gto/src/intor.rs:86-92`, `:170-189`,
  `:22-27`, `crates/pyscf-core/src/traits.rs:86`,
  `crates/pyscf-gto/src/ecp_engine_stub.rs`

## What this task did NOT do

- Did not bump the cintx pin in pyscf_rs `Cargo.toml`
- Did not execute pyscf_rs plan `02-10` (gated on cintx merging Ask 1)
- Did not flip any `#[ignore]` or xfail test gates
- No code changes in pyscf_rs

## Commits

This quick task does not touch source code in pyscf_rs — only the planning
artifacts (PLAN.md, SUMMARY.md) and STATE.md row are committed.
