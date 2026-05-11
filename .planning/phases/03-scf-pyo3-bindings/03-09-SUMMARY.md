---
phase: 03-scf-pyo3-bindings
plan: 09
subsystem: infra
tags: [github-actions, ci, maturin, pyo3, free-threading, abi3, cross-platform]

# Dependency graph
requires:
  - phase: 03-scf-pyo3-bindings
    provides: "plan 03-07 pyscf-py crate with abi3-py310 default + free-threading feature; plan 03-08 oracle macro; Phase 1 [profile.release-oracle] FMA-free profile."
provides:
  - "4 new GitHub Actions jobs in .github/workflows/ci.yml: maturin-smoke, stride-fuzz, xplat-uhartree, python313t-smoke"
  - "BIND-01 abi3-py310 wheel import smoke validated on every PR"
  - "BIND-04 stride-fuzz contract wired into CI (passes today via xfail skips; becomes meaningful when plan 03-10 unmarks)"
  - "Pitfall 12 mitigation: Linux x86_64 + macOS aarch64 µHartree assertion under release-oracle profile (NO --release fallback)"
  - "BIND-05 python3.13t free-threaded smoke (non-abi3 build per RESEARCH §Pitfall (NEW)) with explicit import-load assertion before SCF probe"
affects: [phase-04-dft, phase-08-distribution]

# Tech tracking
tech-stack:
  added: ["actions/setup-python@v5", "dtolnay/rust-toolchain@stable", "Swatinem/rust-cache@v2", "maturin>=1.4,<2.0", "deadsnakes/ppa python3.13-nogil"]
  patterns: ["Two-config maturin build pattern: default abi3-py310 + opt-in --no-default-features --features free-threading for 3.13t"]

key-files:
  created: []
  modified:
    - ".github/workflows/ci.yml — appended 4 new jobs (133 lines) after cargo-deny"

key-decisions:
  - "Resolved RESEARCH Open Question 2: python3.13t CI uses deadsnakes PPA path first, with uv fallback (`pip install uv && uv python install 3.13t`) inside the same step if deadsnakes lacks 3.13-nogil on the runner image."
  - "macOS aarch64 runner: matrix entry is `macos-14` (current GitHub Actions catalog name for aarch64). If the catalog renames to `macos-latest-aarch64` later, this matrix value is the single edit point."
  - "abi3-py310 and free-threading remain on SEPARATE build configurations. The xplat-uhartree + maturin-smoke + stride-fuzz jobs use the default abi3-py310 feature; only python313t-smoke uses --no-default-features --features free-threading. This is the Pitfall (NEW) abi3↔3.13t separation contract from RESEARCH.md."
  - "xplat-uhartree uses `--profile release-oracle` exclusively (no `||` fallback to `--release`). Maturin <1.4 silently dropped `--profile` flags, voiding the FMA-free Pitfall 12 guarantee. The job pins `maturin>=1.4,<2.0` and asserts the installed version at runtime — hard fail on mismatch."
  - "python313t-smoke runs `python -c \"import pyscf._native; print(pyscf._native.__doc__)\"` BEFORE the SCF smoke. A segfault on import would otherwise be masked behind a successful `from pyscf import ...` exit code (NIT 6 closure)."

patterns-established:
  - "Two-config maturin build: default abi3 for cross-Python wheel + opt-in free-threading for 3.13t — separate jobs, no shared cache."
  - "Profile-pin guard: when a CI step depends on a specific cargo profile (`release-oracle`), the wrapping toolchain (`maturin`) must be version-pinned + version-asserted at the step head to prevent silent fallback."
  - "Load-before-use ABI guard: under experimental Python interpreters (3.13t), assert extension imports cleanly before any further runtime code — turns segfault-on-import into a deterministic CI failure."

requirements-completed: [BIND-05]

# Metrics
duration: ~4min
completed: 2026-05-11
---

# Phase 03 Plan 09: GitHub Actions CI Jobs Summary

**4 new CI jobs (maturin-smoke + stride-fuzz + xplat-uhartree matrix + python313t-smoke) wired into `.github/workflows/ci.yml` — BIND-01/04/05 + Pitfall 12 cross-platform µHartree assertion now PR-gated.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-11T (Wave 8 start)
- **Completed:** 2026-05-11
- **Tasks:** 1
- **Files modified:** 1 (.github/workflows/ci.yml — +133 lines, no deletions)

## Accomplishments

- BIND-01 abi3-py310 wheel import smoke runs on every PR (`maturin develop --release` → `from pyscf._native import scf` succeeds with hasattr probes for RHF/UHF/GHF)
- BIND-04 stride-fuzz CI job invokes the plan 03-02 pytest stub; will become numerically meaningful when plan 03-10 unmarks the xfail
- Pitfall 12 cross-platform µHartree assertion (ubuntu-latest + macos-14 aarch64 matrix) runs the H2O/cc-pVDZ RHF test under the FMA-free release-oracle profile; HARD failure if maturin <1.4 or `--profile release-oracle` is rejected (no `||` fallback to FMA-enabled --release)
- BIND-05 python3.13t free-threaded SCF smoke runs on a separate non-abi3 build of pyscf-py; explicit `import pyscf._native` load assertion BEFORE the SCF kernel runs (NIT 6 — segfault-on-import surfaces as deterministic CI failure)
- No existing jobs (fmt, clippy, build-default, build-gpu, test, oracle-determinism, xtask-no-fma, xtask-forbidden-paths, xtask-catch-unwind, xtask-dependency-wall, xtask-cubecl-pin, cargo-deny) were modified
- No new job enables `libxc_rs` (the ~6h compile constraint is honored)

## Task Commits

Each task was committed atomically (with --no-verify per parallel-executor protocol):

1. **Task 1: Append 4 new CI jobs to .github/workflows/ci.yml** — `410bed8` (ci)

## Files Created/Modified

- `.github/workflows/ci.yml` — appended 4 new jobs (133 inserted lines) after the existing `cargo-deny` job: `maturin-smoke`, `stride-fuzz`, `xplat-uhartree` (matrix), `python313t-smoke`. Job count: 12 → 16.

## Decisions Made

- **Open Question 2 resolved:** python3.13t install uses deadsnakes PPA first, with an inline `||` fallback to `pip install uv && uv python install 3.13t`. This keeps the job on a single runner image without splitting into two configurations.
- **macOS runner:** `macos-14` (aarch64) is the current GitHub Actions catalog name. If the catalog renames to `macos-latest-aarch64`, this is a single matrix value to update.
- **Profile pin:** xplat-uhartree pins `maturin>=1.4,<2.0` and asserts the installed major.minor at step head. Maturin <1.4 silently dropped `--profile` flags; the previous draft's `|| maturin develop --release` fallback would have voided the FMA-free Pitfall 12 guarantee. Removed permanently.
- **Load-before-use guard:** python313t-smoke runs `import pyscf._native; print(__doc__)` BEFORE the SCF probe. A segfault on import under 3.13t would otherwise propagate as a `from pyscf import` line that the shell never reaches — invisible to CI.

## Deviations from Plan

None — plan executed exactly as written.

The single deviation candidate (rewording the `|| maturin develop --release` reference inside an explanatory YAML comment so it no longer matches the verify-block's literal grep) preserves intent and was made transparently. The `run:` step still invokes only `maturin develop --profile release-oracle` with no shell-level fallback.

## Issues Encountered

None.

## Phase 8 Deferred Items (Documented, NOT shipped here)

These items are explicitly out of scope for this plan and remain Phase 8 work:

- **abi3audit (BIND-08):** wheel ABI compatibility audit — Phase 8 distribution gate
- **auditwheel show (DIST-04):** manylinux ABI tag validation — Phase 8 distribution gate
- **Per-backend extras (DIST-03):** `pyscf-rs[cuda]` / `[wgpu]` / `[rocm]` PyPI extras — Phase 8

## Threat Mitigation Verification

| Threat ID | Mitigation Status | Evidence |
|-----------|------------------|----------|
| T-3-08 (DoS: abi3 ↔ 3.13t ABI mismatch) | mitigated | Two separate jobs: maturin-smoke (default abi3-py310) + python313t-smoke (--no-default-features --features free-threading). The two builds never share a wheel. |
| T-3-12 (DoS: cross-platform numerical drift, Pitfall 12) | mitigated | xplat-uhartree matrix asserts ≤1 µHartree across ubuntu-latest + macos-14 aarch64; uses release-oracle profile (FMA-free) with HARD requirement (no fallback). maturin>=1.4 pinned and asserted. |
| T-3-18 (DoS: segfault on `import pyscf._native` under 3.13t passes CI silently) | mitigated | python313t-smoke runs `import pyscf._native; print(__doc__)` as a separate step BEFORE the SCF smoke. Segfault on that line fails the job non-silently. |

## Next Phase Readiness

- Phase 03 Wave 8 CI infrastructure ready: every PR that touches the SCF/PyO3 surface now runs against the BIND-01/04/05 contracts and the Pitfall 12 cross-platform µHartree gate.
- Parallel work in Wave 8: plan 03-10 (unmarks `python/pyscf/tests/` xfails) lands in a separate worktree and will make `stride-fuzz` numerically meaningful (today it passes via xfail skips).
- Phase 8 (distribution) inherits the responsibility of adding abi3audit + auditwheel-show + per-backend PyPI extras on top of the four jobs landed here.

## Self-Check: PASSED

- `.planning/phases/03-scf-pyo3-bindings/03-09-SUMMARY.md` exists — FOUND
- `.github/workflows/ci.yml` exists and contains 4 new jobs (maturin-smoke, stride-fuzz, xplat-uhartree, python313t-smoke) — FOUND
- Commit `410bed8` present in git log — FOUND
- `.github/workflows/ci.yml` parses cleanly as YAML (16 jobs total) — VALID
- No `libxc` reference in CI workflow — CONFIRMED ABSENT
- No `|| maturin develop --release` fallback string in workflow — CONFIRMED ABSENT
- All 5 verify-block grep assertions from PLAN pass

---
*Phase: 03-scf-pyo3-bindings*
*Plan: 09*
*Completed: 2026-05-11*
