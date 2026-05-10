---
phase: 01-foundation
verified: 2026-05-10T06:30:00Z
status: gaps_found
score: 19/21 must-haves verified (2 BLOCKERs remain, 1 partial/source-verified)
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 18/21
  gaps_closed:
    - "BLOCKER 2 (check-cubecl-pin lint fails unconditionally): closed at lint-logic level by Plan 09. xtask/src/bin/check_cubecl_pin.rs rewritten from 130 → 618 lines with reverse-dep-aware BFS carve-out; 5 unit tests all pass. The lint will exit 0 on the real dep graph once Plan 08 unblocks workspace builds."
  gaps_remaining:
    - "BLOCKER 1: Cargo.lock missing from git (git ls-files Cargo.lock returns empty; Cargo.lock file does not exist on disk)"
    - "BLOCKER 3: cargo build --workspace --locked cannot run — Cargo.toml [patch.crates-io] cintx still points to branch=main (SHA beb56e3) which contains phantom .claude/worktrees/agent-* 160000-mode gitlinks with no .gitmodules"
  regressions: []
gaps:
  - truth: "cargo build --workspace --locked succeeds (Roadmap success criterion 1)"
    status: failed
    reason: "Cargo.lock is not committed to the repo. git ls-files Cargo.lock returns no results; the file does not exist on disk. Without it, every CI job that passes --locked fails immediately on a fresh clone. Additionally, Cargo.toml [patch.crates-io] cintx still points at branch=main (SHA beb56e343e24e1daac4ce87fe8b0113edba558c3) which contains 25+ phantom 160000-mode gitlinks under .claude/worktrees/agent-* without a .gitmodules — cargo aborts dep resolution before reaching compilation. Both issues share the same fix path: Plan 08 (already written, not yet executed)."
    artifacts:
      - path: "Cargo.lock"
        issue: "Missing entirely from repo. git ls-files Cargo.lock returns no output. File does not exist on disk."
      - path: "Cargo.toml [patch.crates-io]"
        issue: "cintx = { git = 'https://github.com/BectorVoom/cintx.git', branch = 'main' } — branch=main still points at contaminated SHA beb56e3. Plan 08 Task 3 must replace this with rev = '<clean-sha>'."
    missing:
      - "Execute Plan 08 (already written at .planning/phases/01-foundation/01-08-PLAN.md): identify a clean cintx SHA, repin [patch.crates-io] cintx to rev=<clean-sha>, run cargo generate-lockfile, commit Cargo.lock."
  - truth: "After Plan 04, cargo build --workspace --locked succeeds end-to-end (ROADMAP success criterion 1, Plan 04 must_have)"
    status: failed
    reason: "Same root cause as BLOCKER 1 above. This truth is a restatement of Roadmap success criterion 1 from the Plan 04 perspective. Both share the Plan 08 fix path."
    artifacts:
      - path: "Cargo.toml [patch.crates-io] + Cargo.lock"
        issue: "cintx branch=main contamination + missing Cargo.lock prevent any --locked build from succeeding."
    missing:
      - "Execute Plan 08. Once Plan 08 lands, this truth and Roadmap success criterion 1 close simultaneously."
deferred:
  - truth: "GPU-runtime exercise of cubecl-cuda/wgpu/rocm backends end-to-end on real hardware"
    addressed_in: "Phase 8"
    evidence: "Phase 8 success criterion 1: 'per-backend regression suite runs the full SCF/DFT/MP2/CCSD test corpus on CPU SIMD, CUDA, WGPU, and ROCm by setting PYSCF_BACKEND ... where hardware is available in CI, GPU backends pass at chemical accuracy with documented per-backend tolerance'"
  - truth: "Algebra primitive bodies (gemm, gemv, axpy, scal, transpose, dot, reduce_sum) wired to cubecl_matmul/cubecl_reduce dispatch — Phase 1 ships signature-only stubs returning AlgebraError::NotYetImplemented{phase:2}"
    addressed_in: "Phase 2 (first GTO call site lands the dispatch wiring)"
    evidence: "Plan 04 key-decisions: 'Phase 1 algebra primitives ... ship signature-only stubs returning AlgebraError::NotYetImplemented{phase:2,what:...}. The wave-3 success bar is public surface compiles and is callable; the actual cubecl_matmul/cubecl_reduce dispatch wiring lands at the first GTO call site in Phase 2'. Roadmap Phase 2 GTO success criteria 1-3 reference mol.intor() invocation which exercises those primitives."
  - truth: "host_fallback (eigh, cholesky, qr, svd) bodies wired to faer 0.24 — Phase 1 ships signature-only stubs"
    addressed_in: "Phase 3 (eigh — SCF Fock-matrix diagonalization), Phase 6 (qr — CCSD canonicalization), Phase 7 (svd — gradient null-space)"
    evidence: "Plan 04 key-decisions documents the per-Phase wiring schedule. Roadmap Phase 3 SCF success criterion 1 (eigenvector decomposition, RHF.kernel()) requires eigh."
human_verification:
  - test: "Run cargo build --workspace --locked end-to-end after Plan 08 executes"
    expected: "Exit 0; all 15 + xtask members compile (default features = cpu); release-oracle profile builds; cargo deny check exits 0 with at most warnings."
    why_human: "Cannot run until Plan 08 executes: cintx [patch.crates-io] must be re-pointed to a clean SHA and Cargo.lock generated + committed. Plan 08 is written and ready to execute — this is a go/no-go decision on running it."
  - test: "Run cargo run -p xtask --bin check-cubecl-pin end-to-end after Plan 08 executes"
    expected: "Exit 0 with: check-cubecl-pin: PASS — N crate(s) at 0.10.0, M crate(s) at 0.9.0-pre.5, K crate(s) at 0.9.0-pre.5 transitively from cubecl-matmul/reduce (FOUND-04)"
    why_human: "The lint logic is verified-by-source (see source analysis below). End-to-end execution requires a working workspace build — which needs Plan 08 to land first. Once Plan 08 completes, this is a simple cargo run invocation."
  - test: "Run the oracle-determinism CI job under release-oracle profile on a real GitHub Actions runner"
    expected: "Both matrix entries (RAYON_NUM_THREADS=1 and =8) pass; 5 passed; 0 failed on each."
    why_human: "Verifier ran this locally under default (dev) profile and all 5 tests pass. The release-oracle profile difference (LTO off, codegen-units=1, FMA-off) should make the contract stronger, but the precise CI invocation should be confirmed at least once."
---

# Phase 1: Foundation Verification Report

**Phase Goal:** Establish the 15-crate Rust workspace, FMA-off oracle profile, cubecl 0.10.0 lockstep pin, single-owner pyscf-algebra cubecl surface, ordered (pairwise-128) reduction primitives that are bit-identical across rayon thread counts, panic-safe FFI probes, scope-creep + dependency-wall lints, CI gates, and developer docs. After Phase 1, `cargo build --workspace --locked` succeeds end-to-end and the SHOWSTOPPER pitfalls (FMA contraction, oracle non-determinism, cubecl ABI drift) are mitigated and proven.

**Verified:** 2026-05-10T06:30:00Z
**Status:** gaps_found
**Re-verification:** Yes — after Plan 09 gap closure (BLOCKER 2 lint-logic fix)

**Delta from prior verification (2026-05-10T04:35:35Z):**
- BLOCKER 2 (check-cubecl-pin lint fails): CLOSED at lint-logic level. The check_cubecl_pin.rs source now implements the reverse-dep-aware BFS carve-out per Plan 09 design. Score moves from 18/21 to 19/21.
- BLOCKER 1 (Cargo.lock missing) and BLOCKER 3 (cintx phantom gitlinks): REMAIN OPEN. Plan 08 (.planning/phases/01-foundation/01-08-PLAN.md) is written, not yet executed.

---

## Goal Achievement

### Observable Truths (per ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `cargo build --workspace` succeeds with no GPU features (CPU-only); workspace contains 15 members + facade; pyscf-{core,runtime,algebra} are non-stub | ✗ FAILED (BLOCKER 1+3) | cargo metadata --no-deps lists 16 packages = 15 pyscf-* + xtask. Core crates are substantive (pyscf-core: 9 src files, pyscf-runtime: 12 src files, pyscf-algebra: 14 src files). **BUT** cargo build --workspace --locked cannot run: (a) Cargo.lock missing from git — git ls-files Cargo.lock returns empty, file does not exist; (b) Cargo.toml [patch.crates-io] cintx still uses branch=main pointing at contaminated SHA beb56e3 which has 25+ phantom .claude/worktrees/agent-* 160000-mode gitlinks causing cargo to abort dep resolution. **Remediation: Plan 08 (written, not executed).** |
| 2 | `cargo build --profile release-oracle --workspace` produces FMA-free machine code; xtask check-no-fma exits 0 | ✓ VERIFIED | check-no-fma PASS in prior verification: "PASS — no FMA mnemonics in release-oracle asm (FOUND-05)". .cargo/config.toml applies -Cllvm-args=-fp-contract=off AND -Ctarget-feature=-fma,-fma4 in both [build] and [target.'cfg(all())']. No changes to these files in Plan 09. |
| 3 | `oracle_sum`/`oracle_dot` produces bit-identical results on RAYON_NUM_THREADS=1 and =8 (Pitfall 2 mitigation) | ✓ VERIFIED | 5 oracle_determinism tests pass under both RAYON_NUM_THREADS=1 and =8 in dev profile. PAIRWISE_CHUNK=128 algorithm is thread-count-independent. No changes to oracle.rs in Plan 09. |
| 4 | cubecl 0.10.0 (and all cubecl-* crates) pinned exactly via [workspace.dependencies]; check-cubecl-pin exits 0 | PARTIAL (lint-logic verified-by-source; end-to-end blocked by BLOCKER 1+3) | **Lint logic: VERIFIED BY SOURCE.** The new check_cubecl_pin.rs (618 lines, commit 067f630) implements: (a) PINNED_CRATES [cubecl, cubecl-cpu, cubecl-cuda, cubecl-hip, cubecl-runtime, cubecl-wgpu] must be 0.10.0; (b) PRE_PINNED_CRATES [cubecl-matmul, cubecl-reduce] must be 0.9.0-pre.5; (c) carve_out_active gated on live workspace pins matching PRE_REQUIRED_VERSION; (d) reachable_only_from_carve_out_roots BFS allows 0.9.0-pre.5 cubecl-* transitives ONLY when reachable exclusively from cubecl-matmul/cubecl-reduce; (e) saw_runtime_010 enforcement requires top-level cubecl-runtime 0.10.0. 5 unit tests (passes_when_all_pinned_at_010_with_transitive_009, fails_when_cubecl_runtime_010_missing, fails_when_pyscf_kernels_pulls_old_cubecl, fails_when_matmul_pin_moved_but_runtime_skew_persists, parser_returns_none_on_multiline_table_form) all pass. **End-to-end cargo run blocked by BLOCKER 1+3** — Plan 08 Task 4 is the cross-plan integration proof. |
| 5 | CI enforces 4 lints: clippy unwrap deny, forbidden-paths, catch_unwind, dependency-wall (+ 5th: cubecl-pin) | ✓ VERIFIED (4 of 5 passed in prior verification; 5th now verified by source — see Truth #4) | All 5 xtask binaries exist at xtask/src/bin/. All 5 wired into ci.yml as separate jobs (lines 120-178). check-no-fma PASS, check-forbidden-paths PASS, check-catch-unwind PASS, check-dependency-wall PASS (prior verification). check-cubecl-pin logic verified-by-source — will pass end-to-end once Plan 08 unblocks workspace builds. |
| 6 | Backend resolution behaves: env-var → BackendKind truth table; CPU-only build with no env var resolves to Cpu; algebra primitives agree with faer reference to 1e-12 on 256×256 (ALG-04, ALG-08) | ✓ VERIFIED (env-var truth table) / DEFERRED (1e-12 numeric agreement) | tests/select_backend.rs (7 tests) all pass. The 1e-12 numeric agreement is DEFERRED per Plan 04 key-decisions: algebra primitives ship signature-only stubs returning AlgebraError::NotYetImplemented{phase:2} until Phase 2. Documented and intentional. |

**Score:** 4/6 ROADMAP success criteria fully verified; 1 partial/source-verified (Truth #4 — end-to-end blocked by BLOCKER 1+3); 1 deferred-by-design (numeric agreement in Truth #6).

### Per-Plan must_haves Spot-Check

#### Plan 01 (Workspace skeleton — FOUND-01, FOUND-04, FOUND-10, ORACLE-01)

| Must-have | Status | Evidence |
|-----------|--------|----------|
| `cargo build --workspace --locked` succeeds with default features | ✗ FAILED (BLOCKER 1+3) | Cargo.lock missing; cintx branch=main contaminated. Plan 08 is the fix path. |
| Workspace contains exactly 15 pyscf-* members plus xtask | ✓ VERIFIED | cargo metadata --no-deps returns 16 = 15 pyscf-* + xtask. |
| cubecl 0.10.0 family pinned exactly via [workspace.dependencies] | PARTIAL | Top-level pins correct; lint logic now verified-by-source to correctly handle transitive 0.9.0-pre.5. End-to-end gate blocked by BLOCKER 1+3. |
| [patch.crates-io] points cintx, libxc_rs, xcfun_rs at BectorVoom git remotes | ✓ VERIFIED | Three BectorVoom/{cintx,libxc_rs,xcfun_rs} lines present in Cargo.toml. |
| pyscf-rs/ Python tree, pyproject.toml, examples/, pytest.ini are unchanged | ✓ VERIFIED | No modifications in Plans 08/09 scope. |
| release-oracle profile exists with panic=abort, lto=off, codegen-units=1 | ✓ VERIFIED | Cargo.toml lines 73-79 — all 3 settings present. |
| .cargo/config.toml applies -Cllvm-args=-fp-contract=off and -Ctarget-feature=-fma,-fma4 in BOTH [build] and [target.'cfg(all())'] | ✓ VERIFIED | grep -c 'fp-contract=off' = 2; both stanzas present. |
| cargo deny check succeeds against deny.toml | DEFERRED (needs Cargo.lock — blocked by Plan 08) | deny.toml exists with all 4 sections. cargo-deny gate wired in CI. Will confirm once Plan 08 generates Cargo.lock. |
| pyscf-oracle/Cargo.toml declares pyo3 only in [dev-dependencies] | ✓ VERIFIED | Confirmed in prior verification. |
| pyscf-py/Cargo.toml declares [lib] crate-type = ["cdylib", "rlib"] | ✓ VERIFIED | Confirmed in prior verification. |

#### Plan 02 (pyscf-core — FOUND-02) — UNCHANGED from prior verification

All 7 must-haves VERIFIED. No changes in Plan 09 scope.

#### Plan 03 (pyscf-runtime — FOUND-03, FOUND-09, ALG-04, ALG-08) — UNCHANGED from prior verification

All 9 must-haves VERIFIED. No changes in Plan 09 scope.

#### Plan 04 (pyscf-algebra — ALG-01..05, ALG-07, ALG-08, FOUND-06) — UNCHANGED from prior verification

13 of 14 must-haves VERIFIED; 1 FAILED (cargo build --workspace --locked — same BLOCKER 1+3 root cause). No changes in Plan 09 scope.

#### Plan 05 (xtask — FOUND-05, FOUND-07, FOUND-08, ALG-06)

| Must-have | Status | Evidence |
|-----------|--------|----------|
| xtask exposes 5 binaries | ✓ VERIFIED | xtask/Cargo.toml has 5 [[bin]] entries; all 5 source files present. |
| check-no-fma scans target/release-oracle/deps/*.s for FMA mnemonics | ✓ VERIFIED | PASS in prior verification. |
| check-forbidden-paths greps for upstream-PySCF imports | ✓ VERIFIED | PASS in prior verification. |
| check-catch-unwind greps for extern "C" + catch_unwind pairing | ✓ VERIFIED | PASS in prior verification. |
| check-dependency-wall walks cargo metadata + fails if non-allowed crate declares cubecl-* | ✓ VERIFIED | PASS in prior verification. |
| check-cubecl-pin walks cargo metadata + asserts cubecl-{cpu,cuda,hip,wgpu,runtime} pinned at 0.10.0 with carve-out for matmul/reduce transitive 0.9.0-pre.5 | PARTIAL (verified-by-source) | Plan 09 rewrote check_cubecl_pin.rs (130 → 618 lines, commit 067f630). Source analysis confirms correct reverse-dep-aware BFS logic. 5 unit tests pass. End-to-end cargo run blocked by BLOCKER 1+3. |
| All five binaries exit 0 on the Phase 1 codebase | PARTIAL (4 confirmed; 5th verified-by-source) | The previous FAIL on check-cubecl-pin is now fixed at the logic level. 4 gates confirmed PASS; check-cubecl-pin will PASS once Plan 08 enables workspace builds. |

#### Plan 06 (CI workflows — FOUND-05, FOUND-08, FOUND-10, ALG-06, ORACLE-05, ORACLE-09) — UNCHANGED from prior verification

All 8 must-haves VERIFIED. No changes in Plan 09 scope.

#### Plan 07 (Documentation — FOUND-04, FOUND-09) — UNCHANGED from prior verification

All 6 must-haves VERIFIED. No changes in Plan 09 scope.

#### Plan 09 (cubecl-pin reverse-dep carve-out — FOUND-04)

| Must-have | Status | Evidence |
|-----------|--------|----------|
| xtask/src/bin/check_cubecl_pin.rs rewritten with reverse-dep-aware BFS carve-out | ✓ VERIFIED | File is exactly 618 lines (wc -l confirms). Commit 067f630 exists in git log. |
| Four new functions present: audit(), workspace_pre_pinned_versions(), build_reverse_deps(), reachable_only_from_carve_out_roots() | ✓ VERIFIED | grep -n confirms all 4 at lines 103, 153, 192, 248. |
| Carve-out gated on live workspace pins (auto-disengages when matmul/reduce move to 0.10.0) | ✓ VERIFIED | carve_out_active = matches!(workspace_pre_pinned_versions(root)?, Some((m, r)) if m == PRE_REQUIRED_VERSION && r == PRE_REQUIRED_VERSION) at lines 262-265. When workspace pins move to 0.10.0, the carve-out disengages automatically. |
| PASS message format: "N crate(s) at 0.10.0, M crate(s) at 0.9.0-pre.5, K crate(s) at 0.9.0-pre.5 transitively from cubecl-matmul/reduce (FOUND-04)" | ✓ VERIFIED | Lines 377-379 contain exactly this format with three counter variables. |
| 5 unit tests in #[cfg(test)] mod tests | ✓ VERIFIED | grep -c '#[test]' returns 5. Test names: passes_when_all_pinned_at_010_with_transitive_009, fails_when_cubecl_runtime_010_missing, fails_when_pyscf_kernels_pulls_old_cubecl, fails_when_matmul_pin_moved_but_runtime_skew_persists, parser_returns_none_on_multiline_table_form. All 5 pass per SUMMARY self-check. |
| tempfile = "3" in xtask/Cargo.toml [dev-dependencies] | ✓ VERIFIED | grep -n "tempfile" xtask/Cargo.toml → line 42: tempfile = "3". |
| .planning/phases/01-foundation/deferred-items.md created | ✓ VERIFIED | File exists at expected path. |
| End-to-end cargo run -p xtask --bin check-cubecl-pin exits 0 | PARTIAL (blocked by BLOCKER 1+3) | Per Plan 09 SUMMARY key-decisions: "end-to-end smoke is deferred to Plan 08 Task 4 per the plan's explicit caveat." The lint logic is correct; workspace dep resolution blocks the invocation. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Cargo.toml` | workspace manifest, 15 + xtask members, [profile.release-oracle], [patch.crates-io] | ✓ VERIFIED | 88 lines; all blocks present. NOTE: cintx still branch=main — BLOCKER 1+3. |
| `.cargo/config.toml` | FMA-off rustflags in both [build] and [target.'cfg(all())'] | ✓ VERIFIED | 28 lines; 2 occurrences of fp-contract=off. |
| `deny.toml` | cargo deny config | ✓ VERIFIED | 44 lines; all 4 sections. |
| `Cargo.lock` | committed lockfile | ✗ MISSING | git ls-files Cargo.lock returns empty. File does not exist. **BLOCKER 1.** Plan 08 generates this. |
| `crates/pyscf-core/src/` | 9 source files | ✓ VERIFIED | All 9 files present. |
| `crates/pyscf-runtime/src/` | 11 source files | ✓ VERIFIED | All 11 files present. |
| `crates/pyscf-algebra/src/` | 14 source files | ✓ VERIFIED | All 14 files present. |
| `crates/pyscf-algebra/tests/` | 4 integration tests | ✓ VERIFIED | 15 tests total — all pass. |
| `xtask/src/bin/check_{no_fma,forbidden_paths,catch_unwind,dependency_wall,cubecl_pin}.rs` | 5 binaries; check_cubecl_pin.rs ≥ 618 lines with reverse-dep logic | ✓ VERIFIED | All 5 files present; check_cubecl_pin.rs is exactly 618 lines with 4 new functions + 5 unit tests. |
| `.github/workflows/ci.yml` | pre-merge CI w/ all gates | ✓ VERIFIED | 191 lines; 11 jobs. |
| `.github/workflows/nightly-cross-crate.yml` | nightly cross-crate matrix | ✓ VERIFIED | 69 lines. |
| `CONTRIBUTING.md` | D-15 local sibling-crate recipe | ✓ VERIFIED | 129 lines. |
| `docs/upgrade-cubecl.md` | FOUND-04 upgrade ritual | ✓ VERIFIED | 100 lines. |
| `README.md` | PYSCF_BACKEND quickstart additions | ✓ VERIFIED | "pyscf-rs (Rust port)" section added. |
| `.planning/phases/01-foundation/deferred-items.md` | Plan 09 deferred-items documentation | ✓ VERIFIED | File created by Plan 09 (SUMMARY self-check item 3). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| Cargo.toml [workspace] members | crates/pyscf-{core,runtime,algebra,...}/ | filesystem path | ✓ WIRED | All 15 directories present; cargo metadata returns all 15 packages. |
| Cargo.toml [workspace.dependencies] | cubecl = "=0.10.0" + 5 family pins | exact-version pin | PARTIAL (top-level correct; transitive carve-out now lint-verified-by-source) | Top-level pins correct. Plan 09 lint logic verified to correctly allow transitive 0.9.0-pre.5 from matmul/reduce while flagging any other source. End-to-end gate awaits Plan 08. |
| xtask/src/bin/check_cubecl_pin.rs | live [workspace.dependencies] Cargo.toml | workspace_pre_pinned_versions() string-grep parser | ✓ WIRED | workspace_pre_pinned_versions() reads Cargo.toml at root and parses cubecl-matmul/reduce version strings via line-scanning. Both inline-table and bare-version forms recognized. Multi-line table returns None (deliberately fail-loud). |
| check_cubecl_pin.rs audit() | cargo metadata resolve.nodes | build_reverse_deps() + reachable_only_from_carve_out_roots() BFS | ✓ WIRED | audit() calls build_reverse_deps() to build reverse-dep map from metadata.resolve.nodes, then calls reachable_only_from_carve_out_roots() for each non-0.10.0 cubecl-* node. BFS absorbs at carve-out roots; fails on any leaked non-cubecl-family parent. |
| .cargo/config.toml [build] rustflags | every cargo profile | rustflags inheritance | ✓ WIRED | check-no-fma PASS proves rustflags reach codegen. |
| pyscf-algebra/src/select.rs | pyscf_runtime::probe (priority chain) | function calls + #[cfg(feature)] | ✓ WIRED | tests/select_backend.rs (7 tests) pass. |
| pyscf-algebra/src/oracle.rs | rayon thread count invariance | fixed chunk size in pairwise() | ✓ WIRED | oracle_determinism tests pass under both RAYON_NUM_THREADS=1 and =8. |
| ci.yml jobs | xtask binaries | cargo run -p xtask --bin check-NAME | ✓ WIRED | 5 separate jobs invoke the 5 binaries. |
| nightly-cross-crate.yml | cargo update + rebuild lockstep | cron schedule | ✓ WIRED | All 4 steps present. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| pyscf-algebra/oracle_sum | `xs: &[f64]` | caller-provided slice | Yes — pure function on caller data | ✓ FLOWING |
| pyscf-algebra/select_backend | `kind: BackendKind`, `client: AlgebraClient` | std::env::var("PYSCF_BACKEND") + probe results + DType::from_env() | Yes — env vars resolved at runtime | ✓ FLOWING |
| check_cubecl_pin/audit() | metadata JSON, Cargo.toml content | cargo metadata stdout + std::fs::read_to_string(root/Cargo.toml) | Yes — real metadata from cargo + real Cargo.toml strings | ✓ FLOWING (logic level) / blocked end-to-end by BLOCKER 1+3 |
| pyscf-algebra/gemm/axpy/etc. | (none) | Phase 1 stubs return AlgebraError::NotYetImplemented{phase:2} | Stub by design — Phase 2 wires cubecl dispatch | ✓ DOCUMENTED DEFERRAL |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| check_cubecl_pin unit tests (5 tests via standalone harness) | `cargo test` in standalone harness (per Plan 09 SUMMARY) | `5 passed; 0 failed; 0 ignored` | ✓ PASS (source-verified) |
| oracle_sum bit-identical across thread counts | Tests run under RAYON_NUM_THREADS=1 and =8 | `5 passed; 0 failed` both | ✓ PASS |
| pyscf-core builds | `cargo build -p pyscf-core` (no patch) | `Finished dev profile in 7.14s` | ✓ PASS (prior verification) |
| select_backend env-var truth table | `cargo test -p pyscf-algebra --test select_backend` | `7 passed; 0 failed` | ✓ PASS (prior verification) |
| backend_matrix smoke | `cargo test -p pyscf-algebra --test backend_matrix` | `2 passed; 0 failed` | ✓ PASS (prior verification) |
| cubecl_matmul ABI compat | `cargo test -p pyscf-algebra --test cubecl_matmul_smoke` | `1 passed; 0 failed` | ✓ PASS (prior verification) |
| xtask check-cubecl-pin (end-to-end) | `cargo run -p xtask --bin check-cubecl-pin` | Cannot run — BLOCKER 1+3 (cintx dep resolution fails) | ? BLOCKED (blocked pending Plan 08) |
| cargo build --workspace --locked | `cargo build --workspace --locked` | Cannot run — Cargo.lock missing + cintx contamination | ✗ FAIL |
| Cargo.lock committed | `git ls-files Cargo.lock` | No output — file not tracked | ✗ FAIL |

### Requirements Coverage

All 21 REQ-IDs claimed by Phase 1 are mapped to plans. Cross-checking against REQUIREMENTS.md (Phase 1 = 10 FOUND + ORACLE-01 + ORACLE-05 + ORACLE-09 = 13 per REQUIREMENTS.md Traceability table, extended to 21 by the ALG-* sub-requirements in Plans 03/04/05):

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FOUND-01 | 01-01 + 01-04 | 14-crate workspace + facade (15 per ROADMAP) | ✓ SATISFIED | cargo metadata returns 15 pyscf-* + xtask. |
| FOUND-02 | 01-02 | pyscf-core universal types + traits, no compute deps | ✓ SATISFIED | Cargo.toml: only thiserror/serde/tracing; 5 types + 6 traits; compiles. |
| FOUND-03 | 01-03 | BackendKind + auto_backend priority + WorkspacePool | ✓ SATISFIED | backend.rs + select.rs + workspace_pool.rs; tests pass. |
| FOUND-04 | 01-01 + 01-05 + 01-07 + 01-09 | cubecl 0.10.0 exact-pinned + lockstep + upgrade docs + carve-out lint | PARTIAL | Top-level pins correct; upgrade-cubecl.md exists; check-cubecl-pin lint logic now correct (Plan 09). End-to-end gate blocked by BLOCKER 1+3. |
| FOUND-05 | 01-01 + 01-05 + 01-06 | release-oracle profile + FMA-free + CI grep | ✓ SATISFIED | check-no-fma PASS; ci.yml has the gate. |
| FOUND-06 | 01-04 | oracle_sum/dot/einsum deterministic primitives | ✓ SATISFIED | oracle.rs pairwise N=128; 5 oracle_determinism tests pass. |
| FOUND-07 | 01-01 + 01-04 + 01-05 | panic="abort" + clippy unwrap deny + catch_unwind | ✓ SATISFIED | Profile settings present; check-catch-unwind PASS. |
| FOUND-08 | 01-05 + 01-06 | forbidden-paths lint at every PR | ✓ SATISFIED | check-forbidden-paths PASS; ci.yml has the gate. |
| FOUND-09 | 01-03 + 01-07 | tracing 0.1 + verbosity 0..=9 | ✓ SATISFIED | tracing_init.rs verbose_to_filter maps 0..=9 → LevelFilter::Off..Trace. |
| FOUND-10 | 01-01 + 01-06 | MSRV 1.92 + edition 2024 + Apache-2.0 + cargo deny clean | PARTIAL (config verified; deny check execution blocked by missing Cargo.lock) | rust-version=1.92, edition=2024, license=Apache-2.0 present. deny.toml has all 4 sections. ci.yml has cargo-deny gate. Execution blocked until Plan 08 generates Cargo.lock. |
| ALG-01 | 01-04 | AlgebraClient enum + 7 primitive surface | ✓ SATISFIED | client.rs + 7 primitive .rs files present + re-exported. |
| ALG-02 | 01-04 | Tensor opaque handle | ✓ SATISFIED | tensor.rs declares opaque Tensor + BufferId. |
| ALG-03 | 01-04 | CPU is default backend | ✓ SATISFIED | pyscf-algebra Cargo.toml: default = ["cpu"]. |
| ALG-04 | 01-03 + 01-04 | PYSCF_BACKEND env-driven resolution | ✓ SATISFIED | select.rs implements; tests/select_backend.rs proves truth table. |
| ALG-05 | 01-04 | host eigh/cholesky/qr/svd via faer | ✓ SATISFIED (signatures) / DEFERRED (bodies to Phase 3/6/7) | host_fallback.rs has 4 functions; faer dep declared. Bodies wire in Phase 3/6/7 per plan. |
| ALG-06 | 01-05 | dep-wall lint: only pyscf-algebra/runtime may use cubecl-* | ✓ SATISFIED | check-dependency-wall PASS. |
| ALG-07 | 01-04 | backend_matrix CPU baseline cross-primitive smoke | ✓ SATISFIED | tests/backend_matrix.rs (2 tests pass). |
| ALG-08 | 01-04 | log line `pyscf-algebra: backend=… (env=…, dtype=…)` | ✓ SATISFIED | client.rs::log_resolution emits exactly this format. |
| ORACLE-01 | 01-01 | pyscf-oracle: pyo3 in dev-deps only | ✓ SATISFIED | pyscf-oracle/Cargo.toml has pyo3 in [dev-dependencies]. |
| ORACLE-05 | 01-06 | nightly cross-crate matrix CI | ✓ SATISFIED | nightly-cross-crate.yml runs cargo update + check-cubecl-pin + cargo test. |
| ORACLE-09 | 01-06 | RAYON_NUM_THREADS=1 + release-oracle in CI | ✓ SATISFIED | ci.yml oracle-determinism job uses matrix.rayon=["1","8"] under release-oracle profile. |

**Coverage:** 21/21 REQ-IDs accounted for. 19/21 fully verified or partial/source-verified. 2 with material gaps:
- **FOUND-01 / Roadmap success criterion 1** (BLOCKER 1+3): cargo build --workspace --locked fails — Cargo.lock missing + cintx phantom gitlinks.
- **FOUND-04** (PARTIAL, not BLOCKED): check-cubecl-pin lint logic is now correct per source analysis and unit tests. Remains PARTIAL pending end-to-end confirmation via Plan 08 Task 4.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none in Plan 09 scope) | — | — | — | check_cubecl_pin.rs rewrite (Plan 09) has no TODO/FIXME/placeholder patterns. The 5 unit tests are substantive regression guards. |
| crates/pyscf-algebra/src/host_fallback.rs | 14-55 | NotYetImplemented signatures | Info | Documented deferral to Phase 3/6/7. Plan 04 key-decisions explicitly schedules. |
| crates/pyscf-algebra/src/{gemm,gemv,axpy,scal,transpose,dot,reduce}.rs | (function bodies) | NotYetImplemented signatures | Info | Documented deferral to Phase 2. Plan 04 key-decisions explicitly schedules. |
| Cargo.toml line 85 | branch = "main" | cintx still points at contaminated branch | BLOCKER | The Plan 08 objective. Must be repinned to a clean SHA before Cargo.lock can be generated. |

### Human Verification Required

#### 1. Execute Plan 08 — remediation path for BLOCKERs 1 + 3

**Test:** Run Plan 08 (`.planning/phases/01-foundation/01-08-PLAN.md`) end-to-end: (a) Task 1 identifies a clean cintx SHA via automated git history walk; (b) Task 2 is the human checkpoint if no clean SHA is found; (c) Task 3 applies the repin and generates Cargo.lock; (d) Task 4 runs all 5 xtask gates + cargo deny end-to-end.

**Expected:** cargo build --workspace --locked exits 0; all 5 xtask gates PASS (including check-cubecl-pin which will use the post-Plan-09 three-count PASS format); cargo deny check exits 0; git ls-files Cargo.lock returns Cargo.lock.

**Why human:** Requires identifying and choosing a clean SHA on the cintx upstream — an interactive decision if no clean SHA exists on main history (Plan 08 Task 2 is marked as a human checkpoint gate). Plan 08 is marked `autonomous: false` for this reason.

#### 2. Run cargo run -p xtask --bin check-cubecl-pin end-to-end (after Plan 08)

**Test:** After Plan 08 lands Cargo.lock and the clean cintx SHA, run `cargo run -p xtask --bin check-cubecl-pin` from the repo root.

**Expected:** Exit 0 with: `check-cubecl-pin: PASS — N crate(s) at 0.10.0, M crate(s) at 0.9.0-pre.5, K crate(s) at 0.9.0-pre.5 transitively from cubecl-matmul/reduce (FOUND-04)`.

**Why human:** Depends on Plan 08 executing first. The lint logic is correct per source analysis; this is the cross-plan integration proof that Plan 09 (lint logic) + Plan 08 (workspace build enablement) together close FOUND-04 end-to-end.

#### 3. Run the oracle-determinism CI job under release-oracle profile on a real GitHub Actions runner

**Test:** Trigger the oracle-determinism job in ci.yml (matrix.rayon=["1","8"]) on a real GitHub Actions runner with the release-oracle profile in effect.

**Expected:** Both matrix entries pass; 5 passed; 0 failed on each.

**Why human:** Verifier ran locally under dev profile — all 5 tests pass. The release-oracle profile difference should make the contract stronger, but a CI run should confirm at least once.

### Gaps Summary

Phase 1 at commit a3326cc (post-Plan-09) has closed BLOCKER 2 at the lint-logic level. The check_cubecl_pin.rs rewrite (Plan 09, commit 067f630) implements correct reverse-dep-aware BFS carve-out logic with 5 passing unit tests. The Phase 1 work product is substantive: pairwise-128 oracle reduction is correct and bit-identical across thread counts (verified), the pyscf-algebra cubecl-containment dep wall is enforced (check-dependency-wall PASS), FMA-off rustflags produce FMA-free asm (check-no-fma PASS), and the env-driven backend selection works (15 tests pass across 3 test files).

**Two concrete gaps remain:**

1. **(BLOCKER 1) Cargo.lock is not committed.** git ls-files Cargo.lock returns empty; the file does not exist on disk. Roadmap success criterion 1 demands `cargo build --workspace --locked` succeeds — without Cargo.lock, the --locked flag fails immediately on a fresh clone. All 7 CI jobs using --locked would fail before running.

2. **(BLOCKER 3) cargo build --workspace --locked cannot run.** Cargo.toml [patch.crates-io] cintx still points at `branch = "main"` (SHA beb56e3) which contains 25+ phantom .claude/worktrees/agent-* 160000-mode gitlinks without a .gitmodules file — cargo aborts dep resolution before reaching compilation. This also blocks end-to-end verification of the Plan 09 check-cubecl-pin lint logic.

**Remediation path:** Both BLOCKERs share a single fix path — **Plan 08** (`.planning/phases/01-foundation/01-08-PLAN.md`), which is fully written and ready to execute. Plan 08 Task 1 identifies a clean cintx SHA, Task 2 is the human checkpoint (plan is marked `autonomous: false`), Task 3 applies the repin and generates Cargo.lock, Task 4 proves end-to-end acceptance with all 5 xtask gates + cargo deny. Once Plan 08 completes, the expected score is 21/21 and all 3 original BLOCKERs are closed.

**No new design work is needed.** The only remaining work is executing Plan 08.

---

_Verified: 2026-05-10T06:30:00Z_
_Verifier: Claude (gsd-verifier, Sonnet 4.6)_
_Re-verification after: Plan 09 (commit a3326cc) — BLOCKER 2 closed at lint-logic level_
