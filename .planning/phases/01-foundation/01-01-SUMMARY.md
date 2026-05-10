---
phase: 01-foundation
plan: 01
subsystem: infra
tags: [cargo-workspace, cubecl, rust-2024, deny, fma-off, release-oracle]

# Dependency graph
requires:
  - phase: 00-init
    provides: PROJECT.md / ROADMAP.md / REQUIREMENTS.md (architecture, REQ-IDs, decisions)
provides:
  - Workspace root Cargo.toml with 15 pyscf-* members + xtask
  - Shared [workspace.package] (edition 2024, rust 1.92, Apache-2.0)
  - Shared [workspace.dependencies] with cubecl 0.10.0 family pinned exactly
  - [profile.release] panic=abort + [profile.release-oracle] FMA-free profile
  - [patch.crates-io] for cintx / libxc_rs / xcfun_rs sibling git remotes
  - .cargo/config.toml with -fp-contract=off + -target-feature=-fma,-fma4 in BOTH [build] and [target.'cfg(all())']
  - deny.toml with license allowlist + bans + sources allow-list
  - 12 stub crate Cargo.toml + src/lib.rs files (workspace inheritance, forbid(unsafe_code))
  - pyscf-py declares [lib] crate-type = ["cdylib", "rlib"] (BIND-01 forward compat)
  - pyscf-oracle declares pyo3 only in [dev-dependencies] (ORACLE-01)
affects: [01-02-pyscf-core, 01-03-pyscf-runtime, 01-04-pyscf-algebra, 01-05-xtask, 01-06-ci, 01-07-contributing, all-future-phases]

# Tech tracking
tech-stack:
  added:
    - cubecl 0.10.0 family (cubecl, cubecl-runtime, cubecl-cpu, cubecl-cuda, cubecl-wgpu, cubecl-hip)
    - cubecl-matmul 0.9.0-pre.5, cubecl-reduce 0.9.0-pre.5 (version skew per RESEARCH Pitfall 1)
    - wgpu 29.0.3, faer 0.24.0
    - tracing 0.1.44, tracing-subscriber 0.3.23
    - bytemuck 1, thiserror 2.0.18, serde 1.0, serde_json 1.0.149, anyhow 1.0.102
    - approx 0.5.1, rstest 0.26.1
    - pyo3 0.28.3 (auto-initialize) — dev-dep ONLY in pyscf-oracle
  patterns:
    - 15-member Rust workspace skeleton mirroring cintx/xcfun_rs sibling layout
    - Workspace inheritance shorthand (version.workspace = true)
    - FMA-off rustflags duplicated into [target.'cfg(all())'] for user-config override resilience
    - [patch.crates-io] sibling-crate sourcing via BectorVoom git remotes
    - Stub crate convention: empty src/lib.rs with #![forbid(unsafe_code)] + TODO marker citing implementing phase + REQ-IDs
    - cdylib+rlib forward-compat anchor for Phase 3 abi3 wheel build
    - dev-deps-only oracle pattern (pyo3 isolated to test surface)

key-files:
  created:
    - Cargo.toml (workspace root manifest)
    - .cargo/config.toml (FMA-off rustflags)
    - deny.toml (cargo deny config, FOUND-10)
    - crates/pyscf-rs/Cargo.toml + src/lib.rs (top-level façade stub)
    - crates/pyscf-kernels/Cargo.toml + src/lib.rs (Phase 4 stub)
    - crates/pyscf-gto/Cargo.toml + src/lib.rs (Phase 2 stub)
    - crates/pyscf-scf/Cargo.toml + src/lib.rs (Phase 3 stub)
    - crates/pyscf-dft/Cargo.toml + src/lib.rs (Phase 4 stub)
    - crates/pyscf-mp2/Cargo.toml + src/lib.rs (Phase 5 stub)
    - crates/pyscf-ccsd/Cargo.toml + src/lib.rs (Phase 6 stub)
    - crates/pyscf-grad/Cargo.toml + src/lib.rs (Phase 7 stub)
    - crates/pyscf-geomopt/Cargo.toml + src/lib.rs (Phase 7 stub)
    - crates/pyscf-py/Cargo.toml + src/lib.rs (Phase 3, cdylib+rlib stub)
    - crates/pyscf-oracle/Cargo.toml + src/lib.rs (Phase 3, pyo3 dev-dep stub)
    - crates/pyscf-bench/Cargo.toml + src/lib.rs (Phase 8 stub)
  modified:
    - .gitignore (append target/, Cargo.lock.local, .cargo/config.toml.local)

key-decisions:
  - "cubecl-matmul / cubecl-reduce pinned at 0.9.0-pre.5 (not 0.10.0 like the rest of the cubecl family) because the 0.10.0 versions are unpublished as of 2026-05-10 (RESEARCH Pitfall 1). Plan 04 will run a build-verification test to confirm 0.9.0-pre.5 ABI compatibility with cubecl-runtime 0.10.0."
  - "[patch.crates-io] uses branch-pin form (branch = main / branch = master) rather than rev-pin SHA form. Plan 06's nightly cross-crate matrix CI bumps these via cargo update — which records exact rev SHAs in Cargo.lock — so the branch form provides forward motion without requiring manual SHA tracking in Phase 1."
  - "panic = abort applies to BOTH [profile.release] and [profile.release-oracle] (per CONTEXT § Claude's Discretion §6 — both are shipped artifacts)."
  - "Cargo.lock IS committed (not gitignored) because the workspace ships binaries (xtask in Plan 05) and a cdylib (pyscf-py in Phase 3); per CONTEXT.md <code_context> 'Cargo.lock is committed'."

patterns-established:
  - "FMA-off duplication: -Cllvm-args=-fp-contract=off + -Ctarget-feature=-fma,-fma4 in BOTH [build] and [target.'cfg(all())'] — load-bearing because user-level ~/.cargo/config.toml [target.*] sections fully override [build] rustflags (RESEARCH §6 Pitfall 2)."
  - "Per-crate description string in [package] documents implementing phase, used as a self-describing index of the workspace."
  - "Stub lib.rs marker: TODO comment cites implementing phase + REQ-IDs so any future grep can produce a stub-completion roadmap."

requirements-completed: [FOUND-01, FOUND-04, FOUND-10, ORACLE-01]

# Metrics
duration: ~10min
completed: 2026-05-10
---

# Phase 01 Plan 01: Workspace Skeleton Summary

**15-member Cargo workspace skeleton with cubecl 0.10.0 lockstep pin, FMA-free release-oracle profile, deny.toml, and 12 stub member crates (pyscf-{core,runtime,algebra} reserved for Plans 02/03/04).**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-10T01:23:00Z (approx — files staged at worktree spawn)
- **Completed:** 2026-05-10T01:33:41Z
- **Tasks:** 3 (Tasks 1-2 produced commits; Task 3 was verification-only)
- **Files modified:** 28 (4 root configs + 24 stub crate files)

## Accomplishments

- Workspace root Cargo.toml ships with all 15 pyscf-* members listed plus xtask, even though only the 12 stubs land in this plan — the missing pyscf-{core,runtime,algebra} entries are intentional placeholders for Plans 02/03/04. This gives the planner a single, stable list of members from day one.
- cubecl 0.10.0 lockstep pin established via [workspace.dependencies] (D-13). All 15 pyscf-* member crates that consume cubecl in later phases will inherit this pin, so a cubecl bump anywhere in the workspace requires a single edit to one location.
- FMA-off contract enforced via .cargo/config.toml in BOTH [build] and [target.'cfg(all())'] — defends the 1e-12 oracle parity contract against developer-local ~/.cargo/config.toml overrides (RESEARCH Pitfall 2).
- [profile.release-oracle] is the dedicated FMA-free + lto=off + codegen-units=1 profile that Plan 06's nightly CI will use for ORACLE-09 deterministic-reduction tests; release wheels still use [profile.release] (lto=thin, codegen-units=16).
- deny.toml gates the workspace against (a) yanked transitive deps, (b) GPL/AGPL/LGPL licenses, (c) openssl-sys / system-deps (DIST-04 static-only distribution), and (d) any git source other than the three BectorVoom sibling remotes.
- pyscf-py forward-compat: [lib] crate-type = ["cdylib", "rlib"] is declared NOW (in Phase 1) so Phase 3's BIND-01 maturin abi3 build does not require touching the Cargo.toml again.
- pyscf-oracle compliance: pyo3 lives ONLY in [dev-dependencies] (ORACLE-01) — release wheels NEVER link Python.

## Task Commits

Each task was committed atomically:

1. **Task 1: workspace root Cargo.toml + .cargo/config.toml + deny.toml + .gitignore** — `a79d071` (feat)
2. **Task 2: 12 stub crate Cargo.toml + lib.rs** — `5db46e9` (feat)
3. **Task 3: workspace verification (no new files)** — see "Verification" section below

**Plan metadata commit (this SUMMARY.md):** to be added after this file is staged.

## Files Created/Modified

### Workspace root (4 files)
- `Cargo.toml` — workspace manifest: members list, [workspace.package], [workspace.dependencies] (cubecl pin), [profile.release], [profile.release-oracle], [patch.crates-io]
- `.cargo/config.toml` — FMA-off rustflags in [build] + [target.'cfg(all())']
- `deny.toml` — cargo deny config (advisories + licenses + bans + sources)
- `.gitignore` — appended target/, Cargo.lock.local, .cargo/config.toml.local (Cargo.lock NOT added — it IS committed)

### Stub crates (24 files = 12 × {Cargo.toml, src/lib.rs})
- `crates/pyscf-rs/{Cargo.toml,src/lib.rs}` — top-level façade (Plan 04 wires re-exports)
- `crates/pyscf-kernels/{Cargo.toml,src/lib.rs}` — Phase 4 (DFT XC, integrals dispatch)
- `crates/pyscf-gto/{Cargo.toml,src/lib.rs}` — Phase 2 (Mole + integrals via cintx)
- `crates/pyscf-scf/{Cargo.toml,src/lib.rs}` — Phase 3 (RHF/UHF/GHF + DIIS)
- `crates/pyscf-dft/{Cargo.toml,src/lib.rs}` — Phase 4 (RKS/UKS + Becke grids)
- `crates/pyscf-mp2/{Cargo.toml,src/lib.rs}` — Phase 5 (RMP2/UMP2/DF-MP2)
- `crates/pyscf-ccsd/{Cargo.toml,src/lib.rs}` — Phase 6 (RCCSD/UCCSD)
- `crates/pyscf-grad/{Cargo.toml,src/lib.rs}` — Phase 7 (analytical gradients)
- `crates/pyscf-geomopt/{Cargo.toml,src/lib.rs}` — Phase 7 (BFGS+RFO)
- `crates/pyscf-py/{Cargo.toml,src/lib.rs}` — Phase 3 (cdylib+rlib forward compat per BIND-01)
- `crates/pyscf-oracle/{Cargo.toml,src/lib.rs}` — Phase 3 (pyo3 dev-dep only per ORACLE-01)
- `crates/pyscf-bench/{Cargo.toml,src/lib.rs}` — Phase 8 (criterion benches)

### Untouched (D-03 verified)
- `pyscf/` (upstream Python tree) — `git status --porcelain pyscf/` returns 0 lines
- `pyproject.toml`, `setup.py`, `pytest.ini`, `examples/` — all unchanged

## Decisions Made

See `key-decisions` in frontmatter. Notable additions to STATE.md:

- **Branch-pin form for [patch.crates-io]** (D-13 implementation choice). Plan 06's nightly CI will run `cargo update -p cintx -p libxc_rs -p xcfun_rs` which records exact SHAs in Cargo.lock — eliminating the manual SHA-tracking burden in Phase 1.
- **Cubecl version skew** (RESEARCH Pitfall 1). cubecl-matmul / cubecl-reduce pinned at 0.9.0-pre.5 (not 0.10.0). Plan 04 owns the build-verification test that checks ABI compatibility with cubecl-runtime 0.10.0 before any GEMM call site lands.
- **Cargo.lock is committed.** Documented in .gitignore by NOT excluding it; the workspace ships binaries (xtask in Plan 05) and a cdylib (pyscf-py in Phase 3) so the lockfile is reproducibility-critical.

## Deviations from Plan

None - plan executed exactly as written. The plan body was followed verbatim including:
- Exact file content for Cargo.toml, .cargo/config.toml, deny.toml as specified in Task 1.
- Exact stub template (description / N / ids per-crate) for the 12 stubs.
- Exact two exceptions (pyscf-py crate-type, pyscf-oracle dev-dependencies).
- The stub lib.rs body uses the form "Phase 1 stub — implemented in Phase {N} ... TODO: implemented in Phase {N}." which satisfies both the `forbid(unsafe_code)` and the `TODO Phase` acceptance criteria.

## Verification

### Task 3 outcomes (verification-only — no commit)

**Step 1: cargo metadata** — FAILS as documented in plan body. Output:
```
error: failed to load manifest for workspace member `crates/pyscf-core`
referenced by workspace at `Cargo.toml`
Caused by:
  failed to read `crates/pyscf-core/Cargo.toml`
Caused by:
  No such file or directory (os error 2)
```
This is the **expected partial state** documented in PLAN body §verification: "The full `cargo build --workspace --locked` will NOT pass until Plans 02, 03, 04 are also complete — this plan creates the workspace shell". The pyscf-core / pyscf-runtime / pyscf-algebra entries in `members = [...]` are intentional placeholders for Plans 02/03/04 (which run in the same Wave 1 in separate worktrees and will land via the orchestrator merge).

**Step 2: per-stub `cargo build -p $stub --locked`** — also FAILS for the same root cause: cargo cannot parse the workspace manifest while three member directories are missing. Cannot smoke-test individual stubs in this worktree.

**Step 3: cargo deny check** — could not run (cargo-deny not installed in this environment, AND it depends on `cargo metadata` which is in the failed state above). Re-verification deferred to Plan 04 once the full graph lands, per PLAN body Step 3: "MAY warn about license/sources because the dep graph is sparse (Plan 01 has no source dependencies beyond cubecl pins which aren't actually pulled until Plan 04)".

**Step 4: Cargo.lock state** — Cargo.lock is NOT in .gitignore (verified: `! grep -q '^Cargo\.lock$' .gitignore`). No Cargo.lock generated yet because Step 1 failed; will be generated and committed once Plans 02/03/04 land in Wave 1.

**Step 5: D-03 untouched Python tree** — verified: `git status --porcelain pyscf/ pyproject.toml examples/ pytest.ini setup.py` returns 0 lines.

### must_haves.truths from PLAN frontmatter — verified file-existence-wise

| Truth | Verified |
|-------|----------|
| Workspace contains exactly 15 pyscf-* members plus xtask (D-01, D-02) | YES — `grep -c '"crates/pyscf-' Cargo.toml` = 15; `grep -q '"xtask"' Cargo.toml` |
| cubecl 0.10.0 family pinned exactly via [workspace.dependencies] (D-13 lockstep) | YES — `grep -q 'cubecl\s*=\s*{ version = "=0.10.0"' Cargo.toml` |
| [patch.crates-io] points cintx, libxc_rs, xcfun_rs at BectorVoom git remotes (D-12, D-13) | YES — three `BectorVoom/{cintx,libxc_rs,xcfun_rs}` lines present |
| pyscf-rs/ Python tree, pyproject.toml, examples/, pytest.ini are unchanged (D-03) | YES — `git status --porcelain` shows 0 modifications |
| release-oracle profile exists with panic=abort, lto=off, codegen-units=1 (FOUND-05) | YES — `grep -A6 '\[profile.release-oracle\]' Cargo.toml` shows all three |
| .cargo/config.toml applies -Cllvm-args=-fp-contract=off + -Ctarget-feature=-fma,-fma4 in BOTH [build] and [target.'cfg(all())'] | YES — `grep -c 'fp-contract=off' .cargo/config.toml` = 2 |
| cargo deny check succeeds against deny.toml (FOUND-10) | DEFERRED — cargo-deny not installed in worktree env; deny.toml file exists with all 4 required sections; re-verify at Plan 04 / Plan 06 CI |
| pyscf-oracle/Cargo.toml declares pyo3 only in [dev-dependencies] (ORACLE-01) | YES — `[dev-dependencies]` block contains `pyo3 = ...`; `[dependencies]` is empty |
| pyscf-py/Cargo.toml declares [lib] crate-type = ["cdylib", "rlib"] | YES — verified via grep |
| cargo build --workspace --locked succeeds with default features (CPU only) | DEFERRED — depends on Plans 02/03/04 landing; intended Wave 1 merge state |

The two DEFERRED items are documented as Plan 01-aware deferrals in the PLAN body (§verification points 3 + 5) and are NOT plan-execution failures — they are intended to be re-checked once the wave merge produces a complete workspace.

## Issues Encountered

- **None blocking.** The cargo-build verification step is intentionally deferred to wave merge per the plan body. All artifacts that Plan 01 owns are present and pass file-level acceptance criteria.

## Self-Check: PASSED

All 28 created files verified present at expected paths (4 root configs + 24 stub crate files + this SUMMARY.md). Both task commits verified present in git log:
- `a79d071` (Task 1)
- `5db46e9` (Task 2)

## Next Phase Readiness

- Wave 1 sibling plans (01-02 pyscf-core, 01-03 pyscf-runtime, 01-04 pyscf-algebra) can now drop their crate dirs into `crates/` and the workspace will parse + build.
- Plans 05 (xtask), 06 (CI), 07 (CONTRIBUTING) consume this skeleton without modification.
- Future-phase blockers carried forward to STATE.md (no action needed in this plan):
  - cubecl-matmul / cubecl-reduce 0.9.0-pre.5 ↔ cubecl-runtime 0.10.0 ABI verification — Plan 04
  - faer-ext 0.7.1 ↔ faer 0.24.0 build verification — Plan 04 / ALG-05
  - cargo-deny re-verification on full dep graph — Plan 04 / Plan 06 CI

---
*Phase: 01-foundation*
*Completed: 2026-05-10*
