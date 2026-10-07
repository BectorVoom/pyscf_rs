---
title: Bit-exact GPU remediation — TDD implementation plan
status: draft
plan_fallback: "[UNVERIFIED: Planner Agent unavailable] — no agent named `planner` is installed in this environment (available subagents: explore, general). This PLAN.md was authored directly from SPEC.md + the research report as the skill's explicitly labeled fallback. It has NOT passed the independent Plan Checker gate (see PLAN-CHECK.md: UNVERIFIED, self-review only)."
spec: .planning/plans/bitexact-gpu/SPEC.md
updated_at: 2026-09-19T00:00:00Z
---

# PLAN — Bit-exact GPU remediation (TDD, goal-backward)

Goal (from SPEC §6): attribute → remediate cheap causes → gate per backend →
close ORACLE-07, with zero CPU/oracle-profile regression. Method order is
deliberate: **methodology (S1) before hardware claims; census (S2) before
code changes (S3); audit (S4/S5) before gates (S7).**

## Waves

- **Wave 1 (no hardware):** T1 S1-harness (KRHF) · T2 S6-null (CPU cols) ·
  T3 S2-census CPU col + scaffolding · T4 S7-scaffolding + S8 guard pin ·
  T7 S5-integral-parity (CPU-only).
- **Wave 2 (T4 required):** T5 S2 CUDA/ROCm cols · T6 S4 PTX audit +
  micro-tests · T8 S6 remaining cols.
- **Wave 3 (remediation):** T9 S3 fixed-lane reduction (only if S2 shows
  nonzero geometry ulps) + perf measure · T10 S7 gates + ORACLE-07 closure.
- **Wave 4:** T11 attribution ledger (A2) + final regression sweep.

Parallelizable: T1 ∥ T2 ∥ T3 ∥ T4 ∥ T7. T5/T6 ∥ each other after Wave 1.
T9 blocked on T5. T10 blocked on T5–T8. T11 blocked on all.

## Tasks

### T1 — S1 dm-anchored harness [specs: S1; acceptance: A1]

- Goal: one-shot energy+bands on a frozen dm, CPU/CPU control first.
- Prerequisites: none. Files: new `crates/pyscf-bench/` harness or
  `python/` script reusing `KRHF.kernel(dm0)` (`pyscf-py/src/scf.rs:319`;
  PBC `get_bands` at `pbc/scf.rs:1761`); frozen YN dm fixture + SHA-256
  digest (new file under harness dir). HF-level only — no XC functional
  anywhere in the harness.
- Red: test `frozen_dm_anchor_reproduces_tol_ordering` — CPU/CPU,
  `conv_tol` sweep 1e-8→1e-12 must show `e_tot`-≪-bands ordering and
  shrinkage; EXPECTED FAILURE before harness anchors dm (SCF-depth
  confound present).
- Green: implement frozen-dm one-shot path (no SCF loop); record
  `BackendReport` hex fields.
- Refactor: share fixture loader with S2/S6; run
  `cargo test -p pyscf-bench` + `cargo clippy --all-targets -D warnings`
  on touched crates.
- Evidence: control table in report. Rollback: delete harness dir.

### T2 — S6 eigensolver null test [specs: S6]

- Goal: 0-ulp eig on identical Fock inputs under every backend.
- Prerequisites: none (CPU cols). Files:
  `crates/pyscf-algebra/tests/` new `zeigh_backend_null.rs`; fixtures
  from YN Fock shapes.
- Red: `frozen_fock_bit_identical_across_backends` fails if any backend
  routes the solver off-host.
- Green: no production change expected (solver is host-side); test
  documents the invariant.
- Refactor: extend to `eigh_gen` real path; regression scope
  `lapack_backend_oracle` suite.
- Evidence: 0-ulp table (CPU now, CUDA/ROCm/WGPU cols when hardware lands).

### T3 — S2 census, CPU column + scaffolding [specs: S2]

- Goal: census harness + CPU partial-count/ulp baseline.
- Prerequisites: T1 fixture format. Files:
  `crates/pyscf-algebra/tests/` new `reduce_geometry_census.rs` using
  `launch_reduce_sum`/`launch_dot` on frozen YN-shape vectors.
- Red: `census_reports_partial_counts` — fails before table exists.
- Green: implement census (no kernel change); record CPU partial counts
  vs `oracle_sum` bit distance (expect 0 or explain).
- Refactor: keep census backend-parameterized for Wave 2.
- Evidence: CPU column of the S2 table.

### T4 — S7 scaffolding + S8 guard pin [specs: S7, S8]

- Goal: CI skeleton for hardware comparison jobs (skipped without
  hardware) + assertion that honesty behavior is untouched.
- Prerequisites: none. Files: `.github/workflows/ci.yml` (new job
  `gpu-bitexact-compare`, `if:` hardware label, calls S1 harness at
  documented tolerances); test that `wgpu-no-f64-fallback` still passes
  unmodified.
- Red: job file references harness CLI that T1 will provide (placeholder
  fails until T1 lands — order T1 before enabling).
- Green: skeleton merged disabled-by-default; S8 guard recorded.
- Evidence: `git diff` of workflow + passing existing fallback job.

### T5 — S2 CUDA/ROCm/WGPU columns [specs: S2; needs T4 hardware]

- Goal: complete the partial-count × ulp table per backend.
- Prerequisites: T3. No production change.
- Red→Green: table cells filled; any backend with nonzero ulp on pure
  sums becomes the entry criterion for T9.
- Evidence: full S2 table (answers research Q1).

### T6 — S4 FMA audit [specs: S4; needs T4]

- Goal: PTX/SASS verdict per audited kernel + host-vs-device micro-tests.
- Prerequisites: T5 (to separate geometry from contraction effects).
- Files: audit report (new doc) + `dot_gemm_fma_micro.rs` test.
- Red: micro-test asserts 0-ulp host-vs-device dot; EXPECTED FAILURE
  where FMA fuses (documents, not fixes).
- Green: verdicts recorded with PTX evidence; no production change.
- Evidence: answers research Q2; follow-up filed if cubecl lacks
  per-kernel FP-contract control.

### T7 — S5 integral parity, libcint vs cintx [specs: S5; CPU-only]

- Goal: per-operator `(max_abs, max_ulp)` table on the frozen YN cell for
  every integral family the KRHF band path consumes; establishes the
  integral floor before any GPU attribution.
- Prerequisites: T1 fixture format (frozen cell + digest). No hardware.
- Files: census script/test driving upstream `intor` (libcint,
  `pyscf/gto/moleintor.py:20`) vs native intor (cintx,
  `crates/pyscf-gto/Cargo.toml:13-22`); rows for `int1e_ovlp`,
  `int1e_kin`, `int1e_nuc`, GTH PP parts, and the ERI/JK route.
- Red: `integral_parity_table_complete` fails until every KRHF-consumed
  operator has a row.
- Green: table filled; nonzero rows name the owning cintx workstream
  (routed out, never fixed here).
- Refactor: share frozen-cell loader with T1; regression scope
  `pytest tests/oracle/test_intor_oracle.py`.
- Evidence: integral floor table (expected ~0 ulp where cintx pins
  byte-identity, e.g. `int2e_arity4.rs:12`). Rollback: delete census.

### T8 — S6 remaining backend columns [specs: S6; needs hardware]

- Goal: 0-ulp solver table complete. Expected green (host code).
- Prerequisites: T2.

### T9 — S3 fixed-lane reduction [specs: S3; conditional on T5]

- Goal: backend-independent partial tree, CPU results bit-identical.
- Prerequisites: T5 shows nonzero geometry ulps (else SKIP with rationale).
- Files: `crates/pyscf-algebra/src/reduce.rs`, `dot.rs`, `launch.rs`.
- Red: S2 census re-run asserts equal partial counts + 0-ulp pure sums
  (fails pre-fix on affected backends).
- Green: minimal lane-count fix; `PAIRWISE_CHUNK` untouched.
- Refactor: full CPU oracle suites (`oracle_determinism`,
  `zoracle_determinism`, `reduce_oracle`, `dot_oracle`, `gemm_oracle`)
  must pass UNMODIFIED; measure T4 throughput cost and record go/no-go
  (risk R2).
- Validation: `cargo test -p pyscf-algebra --all-features`,
  `cargo clippy --all-targets --all-features -D warnings`,
  `xtask check-dependency-wall`, `cargo build --workspace`.
- Rollback: revert commit (single code path, no migration).

### T10 — S7 gates + ORACLE-07 closure [specs: S7]

- Goal: hardware CI jobs green at documented tolerances; checkbox ticked.
- Prerequisites: T5–T8 evidence.
- Files: `ci.yml` jobs enabled; `REQUIREMENTS.md:169,360` updated with
  gate names + floors; Phase 8 `Plans: TBD` entry replaced.
- Red: gates fail until floors match measured tables.
- Green: all green; ORACLE-07 closed with evidence links.
- Invariants: no gate tighter than 1 ulp of its quantity; CPU gates untouched.

### T11 — Attribution ledger + final sweep [specs: all; acceptance: A2, A5]

- Goal: geometry + FMA + integral-floor contributions sum to the observed
  CUDA gap within 1 order of magnitude; full regression green.
- Prerequisites: T9, T10.
- Validation: `cargo build --workspace`, `cargo test` affected crates,
  `xtask check-no-fma`, dependency-wall + forbidden-paths lints,
  `cargo fmt --check`.
- Evidence: ledger table; unresolved remainder filed as follow-up (never
  silently absorbed into tolerances).

## Coverage (spec → tasks)

S1→T1 · S2→T3,T5 · S3→T9 · S4→T6 · S5→T7 · S6→T2,T8 · S7→T4,T10 · S8→T4.
Every acceptance behavior A1–A5 has ≥1 Red task (A2→T11, A3→T9/T7,
A4→T8/T10, A5→T9/T11).

## Validation commands (verified from repo)

- `cargo test -p pyscf-algebra --all-features`
- `cargo test -p pyscf-bench` (harness)
- `cargo clippy --all-targets --all-features -D warnings` (touched crates)
- `cargo build --workspace` (+ `--features gpu` compile check)
- `xtask check-no-fma`, `xtask check-dependency-wall`, `cargo fmt --check`
- Python legs: `.venv/bin/python` with `PYTHONPATH` per oracle-conftest
  convention (`tests/oracle/conftest.py`), `PYSCF_BACKEND=cpu|cuda`,
  `PYSCF_MESH`, `PYSCF_CONV_TOL` harness env.
- Intor oracle: `pytest tests/oracle/test_intor_oracle.py` (S5/T7).
