---
phase: 1
slug: foundation
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-10
validated: 2026-05-26
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Sourced from `01-RESEARCH.md` § "Validation Architecture" (lines 1553+).
> Audited 2026-05-26 against the executed phase (9 plans, `01-VERIFICATION.md` 21/21).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (built-in) + `rstest 0.26.1` for parameterized cases |
| **Config file** | none — workspace `Cargo.toml` `[dev-dependencies]` is the entire config |
| **Quick run command** | `cargo test -p pyscf-algebra --lib` |
| **Full suite command** | `cargo test --workspace --locked --no-fail-fast` |
| **Oracle-profile command** | `cargo test --profile release-oracle --workspace --locked --no-fail-fast` |
| **Estimated runtime** | ~30s cached / ~5min cold |

> ⚠️ **Build-graph caution:** `--workspace` pulls `pyscf-dft/-kernels/-ccsd`, which link `libxc_rs` (~6h cold compile). Per-requirement verification below stays scoped to the lightweight foundation crates (`pyscf-core`, `pyscf-runtime`, `pyscf-algebra`) and the `xtask` gates — none of which touch `libxc_rs`.

---

## Sampling Rate

- **After every task commit:** Run `cargo build --workspace --locked && cargo clippy --workspace --all-targets -- -D warnings` (~30s cached)
- **After every plan wave:** Full suite + four xtask gates (`check-no-fma`, `check-forbidden-paths`, `check-catch-unwind`, `check-dependency-wall`) + `cargo deny check` (~5min cold, <1min cached)
- **Before `/gsd-verify-work`:** Full suite green AND `release-oracle` profile suite green AND nightly cross-crate matrix green
- **Max feedback latency:** ~30s per task commit; ~5min per wave

---

## Per-Task Verification Map

> Audited 2026-05-26. `Plan` is the source plan(s) per `01-VERIFICATION.md` Requirements Coverage. Commands corrected to the **actual** tests that shipped (the original draft pre-dated `gsd-planner`/executor output, so several pointed at test names that were never created).

| Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01,04,08 | 1 | FOUND-01 | — | workspace builds clean | build | `cargo build --workspace --locked` | ✅ | ✅ green |
| 01-02 | 1 | FOUND-02 | — | universal types compile | unit | `cargo test -p pyscf-core --lib` (11 tests: scalar+mole) | ✅ | ✅ green |
| 01-03 | 1 | FOUND-03 | — | `select_backend()` defaults to CPU on unset env | integration | `cargo test -p pyscf-runtime --test select_backend -- test_default_is_cpu` *(also `pyscf-algebra select_backend::unset_resolves_to_cpu`)* | ✅ | ✅ green |
| 01-01,05,07,09,08 | 2 | FOUND-04 | — | cubecl 0.10.0 pinned across siblings | CI gate | `cargo run -p xtask --bin check-cubecl-pin` | ✅ | ✅ green |
| 01-01,05,06 | 2 | FOUND-05 | — | FMA-free machine code under `release-oracle` | CI gate | `cargo run -p xtask --bin check-no-fma` | ✅ | ✅ green |
| 01-04 | 2 | FOUND-06 | — | `oracle_sum` bit-identical rayon-1 vs rayon-8 | unit (parameterized) | `cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism` (6 tests, RAYON 1/8) | ✅ | ✅ green |
| 01-01,04,05 | 1 | FOUND-07 | T-1-01 | `unwrap()` denied; `extern "C"` wrapped in `catch_unwind` | clippy + CI gate | `cargo clippy --workspace --all-targets -- -D warnings` AND `cargo run -p xtask --bin check-catch-unwind` | ✅ | ✅ green |
| 01-05,06 | 2 | FOUND-08 | T-1-02 | forbidden upstream-PySCF imports rejected | CI gate | `cargo run -p xtask --bin check-forbidden-paths` | ✅ | ✅ green |
| 01-03,07 | 1 | FOUND-09 | — | verbose 0..=9 → `LevelFilter` mapping (`tracing::info!` verbosity) | unit | `cargo test -p pyscf-runtime --lib -- tracing_init` *(added 2026-05-26: `verbose_to_filter_covers_full_boundary_table`)* | ✅ | ✅ green |
| 01-01,06,08 | 2 | FOUND-10 | — | `cargo deny` clean | CI gate | `cargo deny check` | ✅ | ✅ green |
| 01-04 | 1 | ALG-01 | — | algebra public surface compiles | unit | `cargo test -p pyscf-algebra --lib` | ✅ | ✅ green |
| 01-04 | 1 | ALG-02 | — | `cubecl_matmul` symbol smoke test on CPU | integration | `cargo test -p pyscf-algebra --test cubecl_matmul_smoke` | ✅ | ✅ green |
| 01-04 | 1 | ALG-03 | — | `gpu` feature OFF by default; per-backend opt-in | build | `cargo build --workspace --locked` (default = gpu OFF; `gpu = ["cuda","wgpu"]` feature exists — opt-in compile is CI/GPU-host, see Manual-Only) | ✅ | ✅ green |
| 01-03,04 | 1 | ALG-04 | — | `PYSCF_BACKEND=auto` resolution rules | integration | `cargo test -p pyscf-algebra --test select_backend -- auto_on_cpu_only_build_resolves_to_cpu` *(+ `pyscf-runtime select_backend`)* | ✅ | ✅ green |
| 01-04 | 1 | ALG-05 | — | Eigh routes to faer via Vec<f64> round-trip | unit | Phase-1 ships signature-only stub (documented deferral); numeric body + tests landed Phase 3 — `cargo test -p pyscf-algebra --lib -- eigh_gen` (3 tests) | ✅ | ✅ green (cross-phase) |
| 01-05 | 2 | ALG-06 | T-1-03 | dependency-wall: only `pyscf-{algebra,runtime}` may depend on cubecl-* | CI gate | `cargo run -p xtask --bin check-dependency-wall` | ✅ | ✅ green |
| 01-04 | 2 | ALG-07 | — | backend-matrix smoke (CPU baseline; GPU rows Manual-Only) | integration | `cargo test --profile release-oracle -p pyscf-algebra --test backend_matrix` (2 tests, CPU) | ✅ | ✅ green |
| 01-04 | 1 | ALG-08 | — | `tracing::info!` emits backend + dtype resolution | integration | `cargo test -p pyscf-algebra --test select_backend -- alg08_log_resolution_invoked` | ✅ | ✅ green |
| 01-01 | 1 | ORACLE-01 | — | `pyscf-oracle` exists; pyo3 optional/feature-gated only | build + metadata | `cargo build -p pyscf-oracle` AND `cargo metadata` field check | ✅ | ✅ green |
| 01-06 | 2 | ORACLE-05 | — | nightly cross-crate matrix CI | scheduled | `.github/workflows/nightly-cross-crate.yml` (file present; runs on GitHub cron) | ✅ | ✅ green |
| 01-06 | 2 | ORACLE-09 | — | oracle profile pinned to `RAYON_NUM_THREADS=1` | CI gate (matrix) | env var in `.github/workflows/ci.yml` `oracle-determinism` job (matrix rayon=1,8) | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Phase 1 IS Wave 0 — every artifact is greenfield. All blockers delivered (per `01-VERIFICATION.md` Required Artifacts, 21/21):

- [x] Workspace `Cargo.toml` (15 members + `xtask`)
- [x] `crates/pyscf-{core,runtime,algebra}/Cargo.toml` + `src/lib.rs` (non-stub)
- [x] 12 stub-crate `Cargo.toml` + `src/lib.rs` (`pyscf-{kernels,gto,scf,dft,mp2,ccsd,grad,geomopt,py,oracle,bench}` + top-level façade)
- [x] `xtask/Cargo.toml` + `src/bin/check_*.rs` (`check-no-fma`, `check-forbidden-paths`, `check-catch-unwind`, `check-dependency-wall`, `check-cubecl-pin`)
- [x] `.cargo/config.toml` (rustflags `-Cllvm-args=-fp-contract=off` per xcfun precedent)
- [x] `deny.toml`
- [x] `.github/workflows/ci.yml` + `.github/workflows/nightly-cross-crate.yml`
- [x] Test infrastructure:
  - `crates/pyscf-algebra/tests/backend_matrix.rs`
  - `crates/pyscf-algebra/tests/oracle_determinism.rs`
  - `crates/pyscf-algebra/tests/cubecl_matmul_smoke.rs`
  - `crates/pyscf-runtime/tests/select_backend.rs`
- [x] `CONTRIBUTING.md` "local sibling-crate development" recipe (D-15 deliverable)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU backend matrix on CUDA hardware | ALG-07 (GPU rows) | No self-hosted GPU runner in Phase 1; full GPU regression suite is Phase 8 (ORACLE-07). Phase 1 ships only the CPU baseline. | On a CUDA host: `cargo test --profile release-oracle -p pyscf-algebra --test backend_matrix --features gpu`. Manually inspect that GEMM/AXPY/reduce_sum agree to 1e-12 with the CPU baseline. |
| `--features gpu` compile (cubecl-cuda + cubecl-wgpu) | ALG-03 (opt-in half) | The default `gpu`-OFF build is automated above; the opt-in `gpu` compile pulls cubecl-cuda/cubecl-wgpu + wgpu and needs a CUDA toolchain — runs in CI on capable runners, not in the local gate. | On a host with CUDA + Vulkan toolchains: `cargo build -p pyscf-algebra --features gpu`. Confirm it links. |
| `PYSCF_BACKEND=wgpu` + `PYSCF_DTYPE=f64` hard-error path on adapter without `shader-f64` (D-09) | ALG-04 + D-09 | Requires a wgpu adapter that reports the SHADER_F64 feature missing — varies by host/driver. CI runners may opportunistically have it. | On a host whose wgpu adapter lacks `Features::SHADER_F64`: `PYSCF_BACKEND=wgpu PYSCF_DTYPE=f64 cargo test -p pyscf-runtime --test select_backend -- shader_f64_hard_error`. Confirm `BackendError::Unsatisfiable` is returned (not silently downgraded). |

---

## Validation Audit 2026-05-26

| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 1 |
| Escalated | 0 |

- **FOUND-09 (MISSING → green):** `verbose_to_filter` (verbose 0..=9 → `LevelFilter`) had no automated test — verified in `01-VERIFICATION.md` by code inspection only. Added in-source unit test `tracing_init::tests::verbose_to_filter_covers_full_boundary_table` (covers 0,1,2,3,4,5,6,7,8,9,255). `cargo test -p pyscf-runtime --lib -- tracing_init` → `1 passed; 0 failed`.
- **Command corrections (no new tests needed — behavior already covered):**
  - FOUND-03: draft pointed at non-existent lib test `select::test_unset_falls_back_to_cpu` → corrected to integration `select_backend::test_default_is_cpu` / `unset_resolves_to_cpu`.
  - ALG-08: draft pointed at non-existent lib test `client::test_log_resolution` → corrected to integration `select_backend::alg08_log_resolution_invoked`.
  - ALG-05: draft pointed at `host_fallback::test_eigh_round_trip`; Phase 1 intentionally shipped a signature-only stub (documented deferral). Numeric body + tests landed Phase 3 (`eigh_gen.rs`) — command repointed there.
  - ALG-03: `gpu` feature confirmed present (`gpu = ["cuda","wgpu"]`); default-OFF build is the automated half, opt-in compile moved to Manual-Only.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s per task / 5min per wave
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-05-26 — 21/21 requirements have automated verification (FOUND-09 gap closed).
