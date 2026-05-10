---
phase: 01-foundation
plan: 07
subsystem: documentation
tags: [docs, contributing, readme, cubecl-upgrade, foundation, phase-1-finale]
requires:
  - 01-01  # workspace skeleton (15 crates documented in CONTRIBUTING + README)
  - 01-04  # algebra surface — env-var contract documented
  - 01-06  # CI workflows — nightly-cross-crate.yml referenced from upgrade-cubecl.md
provides:
  - CONTRIBUTING.md-deliverable    # D-15 day-one orientation + xtask cheatsheet
  - upgrade-cubecl.md-deliverable  # FOUND-04 four-crate ABI lockstep ritual
  - readme-rust-section            # PYSCF_BACKEND/PYSCF_DTYPE quickstart (FOUND-09)
affects:
  - "developer onboarding (CONTRIBUTING.md replaces 119-byte placeholder)"
  - "cubecl bump procedure (docs/upgrade-cubecl.md is now the canonical reference)"
  - "marketing surface (README.md gains Rust-port section without disturbing upstream)"
tech-stack:
  added: []
  patterns:
    - "documentation-as-code: REQ-IDs and D-IDs are greppable in shipped docs"
    - "two-tree coexistence documented: Rust (cargo) + Python (pip) side-by-side"
    - "developer-local override via ~/.cargo/config.toml (NOT shipped)"
key-files:
  created:
    - docs/upgrade-cubecl.md
  modified:
    - CONTRIBUTING.md  # 4 lines / 119 bytes → 129 lines (replaced placeholder)
    - README.md        # 72 lines → 152 lines (appended Rust-port section)
decisions:
  - "Replaced CONTRIBUTING.md placeholder rather than appending (D-15 deliverable IS the file content)"
  - "Appended Rust section to README.md after upstream PySCF content with --- separator (D-03: don't disturb Python tree)"
  - "Documented gpu feature OFF by default — matches pyscf-algebra/Cargo.toml `default = [\"cpu\"]` (ALG-03, FOUND-03)"
  - "Cross-references between all three documents (CONTRIBUTING ↔ upgrade-cubecl ↔ README) form a navigable triangle"
metrics:
  duration: "5m 25s"
  duration_seconds: 325
  completed_date: "2026-05-10T04:25:10Z"
  tasks_completed: 3
  files_changed: 3
  files_created: 1
  files_modified: 2
  commits: 3
---

# Phase 1 Plan 07: Developer Documentation Summary

**One-liner:** Closes Phase 1 with three developer-facing documents — replaces the 119-byte CONTRIBUTING.md placeholder with the D-15 sibling-crate-development recipe + xtask cheatsheet, creates docs/upgrade-cubecl.md documenting the FOUND-04 four-crate ABI lockstep ritual, and appends a Rust-port section to README.md (preserving upstream PySCF content per D-03).

## What was built

Three documentation deliverables landed atomically, one commit per file:

1. **CONTRIBUTING.md** (`721ea7a`) — replaces the 119-byte upstream placeholder (was: 4 lines pointing at pyscf.org). The new file is 129 lines covering:
   - Project layout: 15 Rust crates + xtask, 2-tree coexistence with the upstream Python tree
   - Local sibling-crate development recipe (D-15): `[patch.crates-io]` override in `~/.cargo/config.toml` for cintx/libxc_rs/xcfun_rs
   - CI gate cheatsheet for all 5 xtask binaries with REQ-ID + Pitfall mapping
   - PYSCF_BACKEND + PYSCF_DTYPE env-var documentation (D-07, D-08, D-09)
   - shader-f64 hard-error rule
   - RAYON_NUM_THREADS oracle-determinism reproduction recipe (Roadmap criterion 3)
   - Code style rules (`clippy::unwrap_used`, `#![forbid(unsafe_code)]`, REQ-ID citation convention)
   - Cross-reference to docs/upgrade-cubecl.md

2. **docs/upgrade-cubecl.md** (`3af1407`) — new file (100 lines). Documents the FOUND-04 + D-14 cubecl bump procedure:
   - Trigger conditions (cubecl 0.11.0 ships, sibling bumps, nightly CI fails)
   - Five-step lockstep ritual: verify availability → cintx → libxc_rs+xcfun_rs in parallel → pyscf-rs → observe nightly-cross-crate.yml
   - Recovery procedure for unilateral sibling drift
   - Why the ritual exists (cubecl pre-1.0 ABI churn, RESEARCH Pitfall 3)
   - Cross-references to nightly-cross-crate.yml and xtask/src/bin/check_cubecl_pin.rs

3. **README.md** (`2a2df8c`) — appends a "pyscf-rs (Rust port)" section after the existing upstream PySCF content (separated by `---`). The new section is 80 lines covering:
   - Phase 1 status callout with a forward-looking scope teaser
   - Developer quickstart (`cargo build --workspace --locked`, `--features gpu`)
   - Backend selection table (PYSCF_BACKEND, PYSCF_DTYPE) with allowed values
   - Workspace `gpu` feature OFF by default (CPU is default backend)
   - cuda → rocm → metal → wgpu → cpu auto priority chain
   - shader-f64 hard-error rule
   - Workspace structure table (15 crates organised by phase)
   - Cubecl pin section with cross-reference to docs/upgrade-cubecl.md

## File list

| File                                 | Status   | Lines (before → after) | Commit    |
|--------------------------------------|----------|------------------------|-----------|
| `CONTRIBUTING.md`                    | replaced | 4 → 129                | `721ea7a` |
| `docs/upgrade-cubecl.md`             | created  | 0 → 100                | `3af1407` |
| `README.md`                          | appended | 72 → 152               | `2a2df8c` |

`wc -l` confirmation:
```
129 CONTRIBUTING.md
100 docs/upgrade-cubecl.md
152 README.md
```

## Verification results

All plan-level acceptance criteria pass:

- `wc -l < CONTRIBUTING.md` = 129 (≥80 required)
- `wc -l < docs/upgrade-cubecl.md` = 100 (≥60 required)
- `grep -q '\[patch.crates-io\]' CONTRIBUTING.md` ✓
- `grep -q 'path = "/home' CONTRIBUTING.md` ✓
- All 5 xtask binaries cited (check-no-fma, check-forbidden-paths, check-catch-unwind, check-dependency-wall, check-cubecl-pin) ✓
- `grep -q 'lockstep' docs/upgrade-cubecl.md` ✓
- `grep -Eq 'Step [1-5]' docs/upgrade-cubecl.md` ✓
- `grep -q 'pyscf-rs (Rust port)' README.md` ✓
- `head -1 README.md` returns `<div align="left">` (upstream first line preserved) ✓
- D-IDs cited: D-07, D-08, D-09, D-13, D-14, D-15
- REQ-IDs cited: FOUND-04, FOUND-09 (plus FOUND-03, FOUND-05, FOUND-07, FOUND-08, ALG-03, ALG-06, ALG-08 for context)

### D-03 invariant (upstream Python tree untouched)

```
$ git status --porcelain pyscf/ pyproject.toml setup.py examples/ pytest.ini | wc -l
0
```

The diff for this plan touches **only** the three declared files. No modifications to `crates/`, `xtask/`, `.github/`, or any upstream-PySCF Python file.

### Cross-reference triangle

| From                   | To                       | Verified |
|------------------------|--------------------------|----------|
| CONTRIBUTING.md        | docs/upgrade-cubecl.md   | ✓        |
| README.md              | docs/upgrade-cubecl.md   | ✓        |
| README.md              | CONTRIBUTING.md          | ✓        |

## Deviations from Plan

**None — plan executed exactly as written.**

The plan provided exact file content for all three documents. No bugs encountered, no missing functionality discovered, no architectural decisions required. CONTRIBUTING.md was 119 bytes / 4 lines as predicted; README.md was 72 lines as predicted; docs/ directory did not exist (anticipated by the plan), so it was created with `mkdir -p`.

## Authentication gates

None — purely local file edits.

## Known stubs

None — all three documents are substantive deliverables, not placeholders. The Phase 1 documentation is complete; subsequent phases will *add* method-specific documentation (e.g., docs/scf-implementation.md in Phase 2, docs/dft-functionals.md in Phase 4) rather than fill in stubs here.

## Threat flags

None — documentation-only plan touching no security-relevant surface.

## Phase 1 finale

Plan 01-07 is the **last plan of Phase 1**. With it the foundation is shipped:

- Workspace skeleton (Plan 01)
- Pyscf-core API surface (Plan 02)
- Pyscf-runtime backends + probes (Plan 03)
- Pyscf-algebra + cubecl integration (Plan 04)
- xtask gate binaries (Plan 05)
- CI workflows (Plan 06)
- **Developer documentation (Plan 07 — this one)**

Recommended next step is `/gsd-verify-work 1` to gate-check all 21 Phase 1 REQ-IDs and the 15 D-IDs from CONTEXT.md against shipped artifacts before opening Phase 2 (Hartree–Fock).

## Self-Check: PASSED

Files verified to exist with expected content:

- FOUND: `/home/user/Documents/workspace/pyscf_rs/.claude/worktrees/agent-ae0140b6bc6a17ded/CONTRIBUTING.md` (129 lines)
- FOUND: `/home/user/Documents/workspace/pyscf_rs/.claude/worktrees/agent-ae0140b6bc6a17ded/docs/upgrade-cubecl.md` (100 lines)
- FOUND: `/home/user/Documents/workspace/pyscf_rs/.claude/worktrees/agent-ae0140b6bc6a17ded/README.md` (152 lines)

Commits verified to exist on `worktree-agent-ae0140b6bc6a17ded`:

- FOUND: `721ea7a docs(01-07): replace CONTRIBUTING.md placeholder with D-15 deliverable`
- FOUND: `3af1407 docs(01-07): add docs/upgrade-cubecl.md — four-crate ABI lockstep ritual`
- FOUND: `2a2df8c docs(01-07): append Rust-port section to README.md (preserves upstream)`
