# Phase 01 Deferred Items

Items discovered during execution that are out of scope for the current plan and tracked for future work.

## 2026-05-10 — Plan 01-09 (gap closure)

### Workspace `cargo build` blocked by upstream cintx submodule data corruption

**Discovered during:** Plan 01-09 Task 1 verification step (`cargo build -p xtask --bin check-cubecl-pin --locked`).

**Symptom:** Cargo fails to resolve the workspace with:

```
error: failed to load source for dependency `cintx`
Caused by: unable to update https://github.com/BectorVoom/cintx.git?branch=main
Caused by: failed to update submodule `.claude/worktrees/agent-a01e6318`
Caused by: no URL configured for submodule '.claude/worktrees/agent-a01e6318'; class=Submodule (17)
```

**Root cause:** The upstream `cintx` repository (at `https://github.com/BectorVoom/cintx.git`, branch `main`, currently at SHA `beb56e3`) has 26 entries under `.claude/worktrees/agent-*` committed at git mode `160000` (gitlinks / submodule pointers) **without** a corresponding `.gitmodules` file declaring URLs for them. Cargo's git2-based source loader calls `Repository::submodule_init()` which fails with "no URL configured" for every such entry. This affects any workspace that pulls `cintx` via `[patch.crates-io] cintx = { git = "https://github.com/BectorVoom/cintx.git", branch = "main" }`.

**Reproducibility:** Pre-existing — same error occurs on the unmodified `b7aab14` baseline. Not introduced by Plan 09's code change.

**Workaround used in Plan 09:** Verified the new `check_cubecl_pin.rs` correctness by copying the file into a standalone Cargo project (with explicit `anyhow`, `serde`, `serde_json`, `tempfile` deps) and running `cargo test` against it. All 5 unit tests pass. End-to-end smoke (`cargo run -p xtask --bin check-cubecl-pin`) is per the plan deferred to Plan 08 Task 4.

**Fix path:** Upstream cintx must either (a) commit a `.gitmodules` file with URLs for all 26 `.claude/worktrees/agent-*` entries, or (b) remove those gitlink entries from its tree (likely the right fix — they appear to be ephemeral GSD agent worktree directories that were accidentally committed at gitlink mode rather than as untracked or .gitignored).

**Owners:** cintx repo maintainer (BectorVoom). Track on the cintx side; once fixed, re-run `cargo update -p cintx` in pyscf_rs and the workspace build will resolve cleanly.

**Impact on Plan 09 acceptance:** Source-level acceptance criteria (line count ≥ 200, all 4 new functions present, 5 `#[test]` blocks, PASS message format, tempfile dep added) ALL VERIFIED via `grep` checks and standalone harness `cargo test` (`5 passed; 0 failed`). Workspace-integrated `cargo build -p xtask --bin check-cubecl-pin --locked` and the end-to-end smoke run remain blocked by the cintx infra issue and are deferred to Plan 08 Task 4 per the plan's explicit caveat.
