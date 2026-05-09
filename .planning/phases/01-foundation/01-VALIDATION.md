---
phase: 1
slug: foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-10
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Sourced from `01-RESEARCH.md` § "Validation Architecture" (lines 1553+).

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

---

## Sampling Rate

- **After every task commit:** Run `cargo build --workspace --locked && cargo clippy --workspace --all-targets -- -D warnings` (~30s cached)
- **After every plan wave:** Full suite + four xtask gates (`check-no-fma`, `check-forbidden-paths`, `check-catch-unwind`, `check-dependency-wall`) + `cargo deny check` (~5min cold, <1min cached)
- **Before `/gsd-verify-work`:** Full suite green AND `release-oracle` profile suite green AND nightly cross-crate matrix green
- **Max feedback latency:** ~30s per task commit; ~5min per wave

---

## Per-Task Verification Map

> Plan IDs are placeholders pending `gsd-planner` output; rows below map REQ-IDs to test/command, planner fills `Plan` and `Task ID` columns.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 1 | FOUND-01 | — | workspace builds clean | build | `cargo build --workspace --locked` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | FOUND-02 | — | universal types compile | unit | `cargo test -p pyscf-core --lib` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | FOUND-03 | — | `select_backend()` defaults to CPU on unset env | unit | `cargo test -p pyscf-runtime --lib -- select::test_unset_falls_back_to_cpu` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | FOUND-04 | — | cubecl 0.10.0 pinned across siblings | CI gate | `cargo run -p xtask --bin check-cubecl-pin` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | FOUND-05 | — | FMA-free machine code under `release-oracle` | CI gate | `cargo run -p xtask --bin check-no-fma` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | FOUND-06 | — | `oracle_sum` bit-identical rayon-1 vs rayon-8 | unit (parameterized) | `cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | FOUND-07 | T-1-01 | `unwrap()` denied; `extern "C"` wrapped in `catch_unwind` | clippy + CI gate | `cargo clippy --workspace --all-targets -- -D warnings` AND `cargo run -p xtask --bin check-catch-unwind` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | FOUND-08 | T-1-02 | forbidden upstream-PySCF imports rejected | CI gate | `cargo run -p xtask --bin check-forbidden-paths` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | FOUND-09 | — | `tracing::info!` emitted on init | unit (with subscriber) | `cargo test -p pyscf-runtime --lib -- tracing_init::test_emits_info` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | FOUND-10 | — | `cargo deny` clean | CI gate | `cargo deny check` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | ALG-01 | — | algebra public surface compiles | unit | `cargo test -p pyscf-algebra --lib` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | ALG-02 | — | `cubecl_matmul::launch` smoke test on CPU | integration | `cargo test -p pyscf-algebra --test cubecl_matmul_smoke -- --nocapture` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | ALG-03 | — | `gpu` feature OFF by default; per-backend opt-in | build | `cargo build --workspace --locked` AND `cargo build --workspace --locked --features gpu` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | ALG-04 | — | `PYSCF_BACKEND=auto` resolution rules | unit | `cargo test -p pyscf-runtime --test select_backend -- --test-threads=1` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | ALG-05 | — | Eigh routes to faer 0.24 via Vec<f64> round-trip (faer-ext skew workaround) | unit | `cargo test -p pyscf-algebra --lib -- host_fallback::test_eigh_round_trip` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | ALG-06 | T-1-03 | dependency-wall: only `pyscf-{algebra,runtime}` may depend on cubecl-* | CI gate | `cargo run -p xtask --bin check-dependency-wall` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | ALG-07 | — | backend-matrix smoke (CPU baseline; GPU rows green-listed) | integration | `cargo test --profile release-oracle -p pyscf-algebra --test backend_matrix` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | ALG-08 | — | `tracing::info!` emits backend + dtype resolution | unit | `cargo test -p pyscf-algebra --lib -- client::test_log_resolution` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | ORACLE-01 | — | `pyscf-oracle` exists; pyo3 in dev-deps only | build + metadata | `cargo build -p pyscf-oracle` AND `cargo metadata` field check | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | ORACLE-05 | — | nightly cross-crate matrix CI | scheduled | `.github/workflows/nightly-cross-crate.yml` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | ORACLE-09 | — | oracle profile pinned to `RAYON_NUM_THREADS=1` | CI gate (matrix) | env var in `.github/workflows/ci.yml` `oracle-determinism` job | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Phase 1 IS Wave 0 — every artifact is greenfield. Blockers (creates the compile target for everything else):

- [ ] Workspace `Cargo.toml` (15 members + `xtask`)
- [ ] `crates/pyscf-{core,runtime,algebra}/Cargo.toml` + `src/lib.rs` (non-stub)
- [ ] 12 stub-crate `Cargo.toml` + `src/lib.rs` (`pyscf-{kernels,gto,scf,dft,mp2,ccsd,grad,geomopt,py,oracle,bench}` + top-level façade)
- [ ] `xtask/Cargo.toml` + `src/bin/check_*.rs` (`check-no-fma`, `check-forbidden-paths`, `check-catch-unwind`, `check-dependency-wall`, `check-cubecl-pin`)
- [ ] `.cargo/config.toml` (rustflags `-Cllvm-args=-fp-contract=off` per xcfun precedent)
- [ ] `deny.toml`
- [ ] `.github/workflows/ci.yml` + `.github/workflows/nightly-cross-crate.yml`
- [ ] Test infrastructure:
  - `crates/pyscf-algebra/tests/backend_matrix.rs`
  - `crates/pyscf-algebra/tests/oracle_determinism.rs`
  - `crates/pyscf-algebra/tests/cubecl_matmul_smoke.rs`
  - `crates/pyscf-runtime/tests/select_backend.rs`
- [ ] `CONTRIBUTING.md` "local sibling-crate development" recipe (D-15 deliverable)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU backend matrix on CUDA hardware | ALG-07 (GPU rows) | No self-hosted GPU runner in Phase 1; full GPU regression suite is Phase 8 (ORACLE-07). Phase 1 ships only the CPU baseline. | On a CUDA host: `cargo test --profile release-oracle -p pyscf-algebra --test backend_matrix --features gpu`. Manually inspect that GEMM/AXPY/reduce_sum agree to 1e-12 with the CPU baseline. |
| `PYSCF_BACKEND=wgpu` + `PYSCF_DTYPE=f64` hard-error path on adapter without `shader-f64` (D-09) | ALG-04 + D-09 | Requires a wgpu adapter that reports the SHADER_F64 feature missing — varies by host/driver. CI runners may opportunistically have it. | On a host whose wgpu adapter lacks `Features::SHADER_F64`: `PYSCF_BACKEND=wgpu PYSCF_DTYPE=f64 cargo test -p pyscf-runtime --test select_backend -- shader_f64_hard_error`. Confirm `BackendError::Unsatisfiable` is returned (not silently downgraded). |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s per task / 5min per wave
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
