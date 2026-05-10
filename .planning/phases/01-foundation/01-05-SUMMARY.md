---
phase: 01-foundation
plan: 05
subsystem: ci-tooling
tags: [xtask, ci-lints, cargo-metadata, fma-asm-scan, walkdir, rustc-demangle, scope-creep-gate, ffi-panic-safety, cubecl-pin]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: "Plan 01-01 workspace skeleton (xtask declared as workspace member, [workspace.dependencies] for anyhow/serde/serde_json shared into xtask)"
provides:
  - "5 standalone CI lint binaries that gate every PR (D-04 lint mechanism: xtask grep + cargo metadata)"
  - "check-no-fma binary scaffolding for FOUND-05 oracle-profile FMA-free contract (Pitfall 1 SHOWSTOPPER mitigation)"
  - "check-forbidden-paths gate for FOUND-08 / Pitfall 21 scope-creep prevention (12 out-of-scope upstream-PySCF module needles)"
  - "check-catch-unwind gate for FOUND-07 / Pitfall 14 unguarded-FFI prevention (forward-looking, Phase 1 has zero extern \"C\")"
  - "check-dependency-wall gate for ALG-06 cubecl-* containment (only pyscf-algebra + pyscf-runtime may consume cubecl)"
  - "check-cubecl-pin gate for FOUND-04 cubecl 0.10.0 lockstep (cubecl-{cpu,cuda,hip,wgpu,runtime}=0.10.0; cubecl-{matmul,reduce}=0.9.0-pre.5)"
affects: [01-06, 04-*, 06-*]

# Tech tracking
tech-stack:
  added: ["walkdir 2 (xtask-only)", "rustc-demangle 0.1 (xtask-only)"]
  patterns:
    - "Workspace-root discovery: walk parents until [workspace] in Cargo.toml"
    - "Forward-compatible scan lists: silently skip absent crates so future plans Just Work"
    - "Source-grep lints strip line comments before matching to avoid doc-string false positives"

key-files:
  created:
    - "xtask/Cargo.toml — 6 [[bin]] entries (1 default + 5 lints), workspace-shared deps"
    - "xtask/src/main.rs — runs all 5 checks sequentially, aggregates exit codes"
    - "xtask/src/bin/check_no_fma.rs — release-oracle asm scan for FMA mnemonics (FOUND-05; Pitfall 1)"
    - "xtask/src/bin/check_forbidden_paths.rs — source grep for upstream-PySCF imports (FOUND-08; Pitfall 21)"
    - "xtask/src/bin/check_catch_unwind.rs — source grep for unguarded extern \"C\" (FOUND-07; Pitfall 14)"
    - "xtask/src/bin/check_dependency_wall.rs — cargo metadata cubecl-* containment (ALG-06)"
    - "xtask/src/bin/check_cubecl_pin.rs — cargo metadata cubecl version lockstep (FOUND-04)"
  modified:
    - ".gitignore — added !xtask/src/bin/ unignore (upstream Python `bin/` rule was swallowing Cargo's standard multi-bin layout)"

key-decisions:
  - "Two-bucket cubecl pin model: PINNED_CRATES at 0.10.0 + PRE_PINNED_CRATES at 0.9.0-pre.5 — reflects the asymmetric crates.io publish state of cubecl-matmul/cubecl-reduce vs the rest of the cubecl 0.10.0 family (RESEARCH §\"Standard Stack\")."
  - "check-dependency-wall uses denylist (FORBIDDEN_DEPS) + ALLOWED_CRATES carve-out semantics (inverse of xcfun_rs check_boundaries' allowlist) — directly encodes ALG-06's intent: \"cubecl-* may only enter the dep graph through pyscf-algebra or pyscf-runtime.\""
  - "check-no-fma SCAN_TARGETS is forward-compatible: missing crates are silently skipped via cheap `cargo metadata --no-deps` substring check. pyscf-kernels can join the table in Phase 4 without changing this binary."
  - "check-catch-unwind is per-file pairing (extern \"C\" in file ⇒ catch_unwind in same file), not per-function. Pre-merge code review (Plan 06) catches the rare cross-file edge case."
  - "Auto-fixed inherited .gitignore bug (Rule 3): the bare `bin/` rule from upstream PySCF's Python distribution swallowed `xtask/src/bin/`. Added `!xtask/src/bin/**` un-ignore in the Rust workspace section."

patterns-established:
  - "xtask multi-binary: each lint is its own [[bin]] entry; default `cargo run -p xtask` orchestrates them all; CI calls them individually for clean job separation"
  - "Workspace-root walker: shared `workspace_root()` helper in every binary — searches parent dirs for Cargo.toml with [workspace]"
  - "Forward-compatible scan lists in lint binaries: missing targets warn or silently skip; never bail. Permits adding crates without coordinating CI updates."

requirements-completed: [FOUND-04, FOUND-05, FOUND-07, FOUND-08, ALG-06]

# Metrics
duration: ~25min
completed: 2026-05-10
---

# Phase 01 Plan 05: xtask CI lint binaries Summary

**5 standalone xtask binaries (check-no-fma, check-forbidden-paths, check-catch-unwind, check-dependency-wall, check-cubecl-pin) gate every PR — covering FMA contraction (Pitfall 1 SHOWSTOPPER), scope creep (Pitfall 21), unguarded FFI (Pitfall 14), cubecl dep-wall (ALG-06), and cubecl version lockstep (FOUND-04).**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-05-10T03:44:28Z
- **Tasks:** 3
- **Files created:** 7 (xtask/Cargo.toml + xtask/src/main.rs + 5 binaries)
- **Files modified:** 1 (.gitignore — Rule-3 deviation)

## Accomplishments

- xtask crate scaffolded with 6 `[[bin]]` entries (1 default orchestrator + 5 named lints), using workspace-shared anyhow/serde/serde_json plus xtask-only walkdir + rustc-demangle.
- 5 lint binaries written, totalling ~720 lines of safe Rust with descriptive failure output naming the violating REQ-ID and Pitfall reference.
- Two source-grep lints (check-forbidden-paths, check-catch-unwind) verified PASS on the current Phase-1 codebase (32 .rs files scanned).
- Two cargo-metadata lints (check-cubecl-pin, check-dependency-wall) compile cleanly; functional verification deferred (see Issues Encountered — environment cintx submodule problem unrelated to this plan).
- check-no-fma compiles cleanly; runtime exercise deferred to Plan 04 (needs pyscf-algebra) + Plan 06 CI execution.
- Inherited `.gitignore` bug fixed — `bin/` rule from upstream PySCF Python distribution was silently ignoring all five binary source files.

## Task Commits

Each task was committed atomically:

1. **Task 1: xtask/Cargo.toml + main.rs (orchestrator)** — `bcfb6b2` (feat)
2. **Deviation Rule 3: .gitignore unignore for xtask/src/bin/** — `ea484de` (fix)
3. **Task 2: check-cubecl-pin + check-dependency-wall (cargo metadata)** — `c679768` (feat)
4. **Task 3: check-no-fma + check-forbidden-paths + check-catch-unwind** — `eec7f42` (feat)

_Plan metadata commit (this SUMMARY.md) follows below._

## Files Created/Modified

- `xtask/Cargo.toml` — 6 [[bin]] entries; workspace-shared anyhow/serde/serde_json + xtask-only walkdir/rustc-demangle.
- `xtask/src/main.rs` — Default xtask orchestrator. Runs all 5 checks sequentially via `cargo run -p xtask --bin <name>`, aggregates exit codes (exit 2 on any failure, 0 on all-pass).
- `xtask/src/bin/check_no_fma.rs` — FOUND-05/Pitfall 1. Compiles SCAN_TARGETS (pyscf-algebra, pyscf-core) with `cargo rustc --profile release-oracle -- --emit=asm`, scans `target/release-oracle/deps/*.s` against `FORBIDDEN_MNEMONICS` table (vfmadd*, vfmsub*, vfnmadd*, vfnmsub*, fmadd, fmsub, fnmadd, fnmsub, fma213, fma231). Demangles symbol labels via rustc-demangle for context-rich failure output.
- `xtask/src/bin/check_forbidden_paths.rs` — FOUND-08/Pitfall 21. Walkdir crates/**/*.rs; per-line string-match against 12 `FORBIDDEN_IMPORT_NEEDLES` (use pyscf::pbc, ::x2c, ::mcscf, ::mcpdft, ::mrpt, ::tdscf, ::tddft, ::adc, ::gw, ::eom, ::nac, ::eph). Fails with file:line + needle on any match.
- `xtask/src/bin/check_catch_unwind.rs` — FOUND-07/Pitfall 14. Walkdir crates/**/*.rs; per-file pairing of `extern "C"` ⇔ `catch_unwind`. Strips line comments before matching to avoid doc-string false positives.
- `xtask/src/bin/check_dependency_wall.rs` — ALG-06. Walks `cargo metadata --no-deps` workspace members; fails if any `pyscf-*` crate other than the carve-out (`pyscf-algebra`, `pyscf-runtime`) declares a normal dep on any of the `FORBIDDEN_DEPS` (cubecl, cubecl-cpu, cubecl-cuda, cubecl-hip, cubecl-matmul, cubecl-reduce, cubecl-runtime, cubecl-std, cubecl-wgpu).
- `xtask/src/bin/check_cubecl_pin.rs` — FOUND-04. Walks `cargo metadata` full graph; asserts `cubecl-{cpu,cuda,hip,wgpu,runtime}` at `0.10.0` and `cubecl-{matmul,reduce}` at `0.9.0-pre.5` (the latter two have no 0.10.0 publish on crates.io as of 2026-05-10). Crates not in the resolved graph are silently skipped (forward-compatible).
- `.gitignore` — added `!xtask/src/bin/` + `!xtask/src/bin/**` un-ignore in the Rust workspace section. Rule-3 deviation (auto-fix blocking issue): the inherited bare `bin/` rule from upstream PySCF was silently swallowing Cargo's standard multi-bin layout.

## Verification Output

### Source-grep lints (verified on this worktree)

```text
$ check-forbidden-paths
check-forbidden-paths: PASS — 32 .rs file(s); no out-of-scope upstream PySCF imports (FOUND-08)

$ check-catch-unwind
check-catch-unwind: PASS — 32 .rs file(s); every `extern "C"` site pairs with `catch_unwind` (FOUND-07)
```

### Build verification

All 5 lint binaries (plus the default xtask orchestrator) compile cleanly under a standalone-Cargo build (single `cargo build` invocation, zero warnings).

### Deferred verifications

The following functional checks could not run **inside this worktree** because of environment issues unrelated to this plan; they are **not** code-quality gaps in the lint binaries themselves:

| Lint | Why deferred | Will be exercised by |
|------|--------------|----------------------|
| `check-no-fma` | Requires `cargo rustc --profile release-oracle -p pyscf-algebra --lib -- --emit=asm` to succeed; pyscf-algebra is owned by Plan 01-04 and is missing from this worktree (Wave 2 parallelism). | Plan 06 CI matrix (after Plan 04 has merged) |
| `check-cubecl-pin` | Calls `cargo metadata --format-version 1` (full graph). The cintx git checkout in `~/.cargo/git/db/cintx-c4edce1591a0822a` has stale gitlinks (Claude-Code agent-worktree submodules without a `.gitmodules` file), causing `cargo metadata` to fail with `failed to update submodule .claude/worktrees/agent-a01e6318`. This is an upstream cintx git-state issue, not a problem with the binary. | Plan 06 CI matrix (CI environment fetches cintx cleanly) |
| `check-dependency-wall` | Calls `cargo metadata --no-deps`, same submodule failure as above. Binary logic verified by code review against ALG-06 spec. | Plan 06 CI matrix |

## Decisions Made

- **Two-bucket cubecl pin model.** Phase 1 RESEARCH §"Standard Stack" notes that `cubecl-matmul` / `cubecl-reduce` have no 0.10.0 publish on crates.io as of 2026-05-10; they're held at the `0.9.0-pre.5` ABI that interoperates with cubecl-runtime 0.10.0. Reflected in `PINNED_CRATES` (0.10.0) + `PRE_PINNED_CRATES` (0.9.0-pre.5) tables.
- **Denylist semantics for check-dependency-wall.** xcfun_rs uses an allowlist (`check_boundaries`); ALG-06 inverts that intent — only the carve-out crates may consume cubecl-*, everything else is forbidden. Implemented as `FORBIDDEN_DEPS` (the cubecl family) plus `ALLOWED_CRATES` (the carve-out: pyscf-algebra, pyscf-runtime).
- **Forward-compatible scan lists.** Both check-no-fma's SCAN_TARGETS and check-cubecl-pin's PINNED_CRATES list silently skip absent crates. Future plans (Phase 4 pyscf-kernels, Phase 6 cubecl GPU plans) join the enforced set automatically by appearing in the workspace, with no coordinating change to xtask required. Mirrors xcfun_rs precedent.
- **per-file pairing for check-catch-unwind.** Per-block pairing would require a real Rust parser (syn) to handle nested blocks and macros; per-file string match catches the realistic FFI patterns (one extern "C" block per file in this codebase) at zero parser cost. Pre-merge review (Plan 06) catches edge cases.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Inherited `.gitignore` swallowing xtask/src/bin/**

- **Found during:** Task 2 (after writing `xtask/src/bin/check_cubecl_pin.rs` + `check_dependency_wall.rs`, `git status --short` showed working tree clean — files invisible to git).
- **Issue:** The inherited `.gitignore` from upstream PySCF (line 13: bare `bin/`) is intended for Python virtualenv `bin/` directories, but it silently matches any directory named `bin` anywhere in the tree — including Cargo's standard multi-binary layout under `xtask/src/bin/`. Without intervention, the 5 lint binary source files would be untracked and lost when the worktree is force-removed.
- **Fix:** Added `!xtask/src/bin/` + `!xtask/src/bin/**` un-ignore lines in the existing "Rust workspace" section of `.gitignore`. Negation pattern follows git's gitignore precedence rules (later entries win).
- **Files modified:** `.gitignore`
- **Verification:** `git check-ignore -v xtask/src/bin/check_no_fma.rs` now reports `.gitignore:99:!xtask/src/bin/**` (the un-ignore line); `git status --short` correctly shows the 5 files as untracked → addable.
- **Committed in:** `ea484de` (separate commit between Task 1 and Task 2).

---

**Total deviations:** 1 auto-fixed (Rule 3 — blocking)
**Impact on plan:** Without the .gitignore fix, the entire plan's deliverables would have been silently dropped. No scope creep; the fix is minimal (2 narrow un-ignore lines in the existing Rust section).

## Issues Encountered

- **Worktree cintx submodule failure prevented in-worktree functional verification of cargo-metadata-driven lints.** `cargo metadata` (any flavor) inside this worktree fails with `failed to update submodule .claude/worktrees/agent-a01e6318` because the cached cintx git checkout (`~/.cargo/git/db/cintx-c4edce1591a0822a`) has gitlinks (`160000 commit ...` tree entries) under `.claude/worktrees/` without a corresponding `.gitmodules` file. This is upstream cintx git-state breakage, completely outside Plan 01-05's scope. Worked around by:
  1. Verifying all 5 binaries compile cleanly via a standalone-Cargo copy at `/tmp/xtask-verify` (using literal version pins instead of `workspace = true`), confirming zero compile errors and zero warnings.
  2. Running the two source-grep lints (check-forbidden-paths, check-catch-unwind) via the standalone-built binaries with the worktree as cwd — both PASS.
  3. Functional verification of check-cubecl-pin / check-dependency-wall / check-no-fma deferred to Plan 06 CI execution where the cintx fetch will succeed.
- **Root Cargo.toml transient modification.** Per the parallel-execution guidance, I transiently commented out the `crates/pyscf-algebra` workspace member to test cargo-metadata behaviour. Restored byte-identically afterward (verified via sha256 match: `dab1db7d69914a6855c8b060e58d94e877cbac693336ace5d32826c2cee92960`). `git status` confirms no modification to root `Cargo.toml` in the final tree.

## User Setup Required

None — no external service configuration required. The 5 lint binaries are pure-Rust and use only deps already in `[workspace.dependencies]` plus two xtask-only deps (walkdir, rustc-demangle) declared inline.

## Next Phase Readiness

- **Plan 01-06 (CI workflow)** can now wire all 5 lint invocations into `.github/workflows/ci.yml` as required PR gates. Suggested job names: `check-no-fma`, `check-forbidden-paths`, `check-catch-unwind`, `check-dependency-wall`, `check-cubecl-pin`. Each is a single `cargo run --quiet -p xtask --bin <name>` invocation.
- **Plan 01-04 (pyscf-algebra)** owns the only currently-missing workspace member; once that lands, `cargo metadata` succeeds in CI and `check-cubecl-pin` + `check-dependency-wall` will gate cleanly. After Plan 04: `cargo run --quiet -p xtask --bin check-no-fma` will exercise the FMA scan against the algebra crate's hot loops.
- **Phase 3 PyO3 bindings** will need to satisfy `check-catch-unwind` — every file with `extern "C"` must also import `std::panic::catch_unwind` (or wrap the FFI body locally). The gate is now in place to enforce this from the first FFI line.
- **No blockers** for Plan 01-06 or downstream phases. The 5 lints are self-contained and forward-compatible.

## Self-Check: PASSED

All claimed artifacts verified:

- `xtask/Cargo.toml` — exists, contains 6 `[[bin]]` blocks, 5 named lints + walkdir + rustc-demangle + publish=false
- `xtask/src/main.rs` — exists, runs all 5 checks
- `xtask/src/bin/check_no_fma.rs` — exists, contains `FORBIDDEN_MNEMONICS`, `vfmadd`, `release-oracle`, `SCAN_TARGETS`
- `xtask/src/bin/check_forbidden_paths.rs` — exists, contains `FORBIDDEN_IMPORT_NEEDLES`, all 12 `use pyscf::*` needles
- `xtask/src/bin/check_catch_unwind.rs` — exists, contains `extern "C"` + `catch_unwind` needle pair
- `xtask/src/bin/check_dependency_wall.rs` — exists, contains `FORBIDDEN_DEPS`, `ALLOWED_CRATES`, `pyscf-algebra`, `pyscf-runtime`
- `xtask/src/bin/check_cubecl_pin.rs` — exists, contains `REQUIRED_VERSION = "0.10.0"`, `PRE_REQUIRED_VERSION = "0.9.0-pre.5"`, `cubecl-matmul`, `cubecl-reduce`
- `.gitignore` — modified, contains un-ignore for `xtask/src/bin/`
- All 4 commits (`bcfb6b2`, `ea484de`, `c679768`, `eec7f42`) present on branch `worktree-agent-a95cb51003acf1637`
- Root `Cargo.toml` byte-identical to pre-plan state (sha256 `dab1db7d…cee92960`)
- Source-grep lints functionally verified PASS (32 .rs files; no violations)

---
*Phase: 01-foundation*
*Completed: 2026-05-10*
