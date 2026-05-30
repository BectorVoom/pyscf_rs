# Phase 1: Foundation - Research

**Researched:** 2026-05-10
**Domain:** Rust workspace skeleton + cubecl 0.10.0 algebra crate + CI gates for numerical correctness
**Confidence:** MEDIUM-HIGH (HIGH on sibling-crate patterns, MEDIUM on cubecl 0.10.0 ecosystem due to two confirmed registry version skews flagged below)

## Summary

Phase 1 is a 21-REQ greenfield landing of the workspace skeleton plus three non-stub crates (`pyscf-core`, `pyscf-runtime`, `pyscf-algebra`) plus four CI lints plus a `release-oracle` profile. CONTEXT.md locks 15 decisions (D-01..D-15); this research fills in the implementation specifics behind those decisions.

Two **registry-shape blockers** dominate the planning surface and must be surfaced to the user before tasks are written:

1. `cubecl-matmul` and `cubecl-reduce` are NOT published at 0.10.0 on crates.io. The latest published versions are `0.9.0-pre.5` (verified by `cargo info cubecl-matmul` / `cubecl-reduce` against the live crates.io index, 2026-05-10). The cubecl umbrella crate (`cubecl 0.10.0`), `cubecl-runtime 0.10.0`, `cubecl-cpu 0.10.0`, `cubecl-cuda 0.10.0`, `cubecl-hip 0.10.0`, `cubecl-wgpu 0.10.0`, `cubecl-core 0.10.0` ARE all at 0.10.0. cintx (sibling) confirms the pattern: it depends on `cubecl = "0.10.0"` and `cubecl-runtime = "0.10.0"` but does NOT name `cubecl-matmul` / `cubecl-reduce` — it hand-writes its own `#[cube]` kernels via `cubecl::prelude::*`.
2. `faer-ext 0.7.1` requires `faer = "0.23.0"` (verified by reading `https://docs.rs/crate/faer-ext/0.7.1/source/Cargo.toml.orig`). PROJECT.md / STATE.md target `faer 0.24.0`. The `faer-ext` ↔ `faer` skew is real and was already flagged in STATE.md "Blockers/Concerns". `faer 0.24` is at MSRV 1.84; `faer-ext 0.7.1` MSRV is 1.67 (still on faer 0.23).

These two blockers do not stop Phase 1 — they reshape the algebra-crate dependency wall (ALG-06) and the GEMM/reduce dispatch shape (ALG-02). Recommended posture: pin to `cubecl-matmul = "=0.9.0-pre.5"` and `cubecl-reduce = "=0.9.0-pre.5"` in `[patch.crates-io]`-territory if the cubecl-runtime ABI is compatible (very likely — they are Tracel-AI's own pre-releases hung off the same workspace), and drop `faer-ext` until faer-ext upstream tracks faer 0.24 (host I/O round-trips through `Vec<f64>` is the chosen fallback per STATE.md). Both decisions belong in PLAN.md, not RESEARCH.md.

**Primary recommendation:** Mirror cintx's workspace shape and dependency declarations verbatim where they overlap; introduce the new `pyscf-algebra` member with cintx-cubecl's per-backend feature shape (D-04, D-13); use **xcfun_rs's `xtask/src/bin/check_*.rs` pattern** for all four CI lints (the precedent is concrete, working, and runs under `cargo run -p xtask --bin check-foo`); bind the cubecl-matmul/cubecl-reduce 0.9.0-pre.5 vs 0.10.0 question to a Phase 1 build-verification task.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Root coexistence (cintx pattern). Top-level `Cargo.toml` + `crates/` + `xtask/` land at the repo root, sitting alongside the existing upstream `pyscf/` Python tree, `pyproject.toml`, `setup.py`, `pytest.ini`, `examples/`, and `docs/manual/Cubecl/`. Upstream PySCF source is **not** moved or renamed in Phase 1 — it stays at root as oracle reference and future maturin host.
- **D-02:** Workspace member directories live under `crates/` (e.g., `crates/pyscf-core/`, `crates/pyscf-algebra/`). Each crate is `pyscf-{name}` (hyphenated package name); the in-Rust module path is `pyscf_{name}` (underscored). Mirrors cintx (`crates/cintx-{core,ops,...}`) and xcfun_rs (`crates/xcfun-{core,kernels,...}`) exactly.
- **D-03:** No changes to `pyproject.toml`, `setup.py`, `pyscf/`, `examples/`, `pytest.ini` in Phase 1. Rust workspace and legacy Python tree must coexist without colliding. `cargo build --workspace` only sees `crates/`; existing Python tooling continues to operate on `pyscf/`.
- **D-04:** Enum + match dispatch (sibling-crate pattern). `AlgebraClient` is an enum with `#[cfg(feature = "<backend>")]`-gated arms (`Cpu`, `Cuda`, `Wgpu`, `Rocm`; `Metal` reuses the wgpu runtime per the cintx-cubecl precedent). Free functions `pyscf_algebra::gemm(&client, a, b, out)` match-dispatch internally. Method crates stay non-generic.
- **D-05:** Opaque `BufferId` + `Tensor` boundary. `pyscf-algebra` owns all device buffers. Public surface returns `Tensor { id: BufferId, shape: Vec<usize>, dtype: DType }`. Method crates hold `Tensor`s and pass by reference. Method crates **never** import or name a `cubecl::*` type.
- **D-06:** Phase 1 algebra surface: `gemm`, `gemv`, `axpy`, `dot`, `reduce_sum`, `transpose`, `scal`, plus deterministic `oracle_sum` / `oracle_dot` / `oracle_einsum`. Eigh/Cholesky/QR/SVD route to `faer 0.24` on host (ALG-05) — on a GPU `AlgebraClient`, the implementation copies down → faer → uploads back, with a documented `// host-fallback per ALG-05` comment.
- **D-07:** Auto resolver probes each compiled backend in priority order (`cuda → rocm → metal → wgpu → cpu`) and emits one `tracing::info!` per probe attempt. Picks first backend with both feature compiled AND a usable device. Final `pyscf-algebra: backend={resolved} (env={raw}, dtype={f32|f64})` line is always emitted (ALG-08).
- **D-08:** **`PYSCF_DTYPE`** ∈ {`f32`, `f64`} (case-insensitive; default `f64`). Separate axis from `PYSCF_BACKEND`. AlgebraClient carries resolved dtype as state.
- **D-09:** wgpu + f64 + missing `shader-f64` rule:
  - `PYSCF_BACKEND=wgpu` (explicit) + `PYSCF_DTYPE=f64` + adapter lacks `shader-f64` → **hard error** at resolver time. Python boundary maps to `RuntimeError`.
  - `PYSCF_BACKEND=auto` + `PYSCF_DTYPE=f64` + adapter lacks `shader-f64` → wgpu probe logs `tracing::info!` and continues priority chain.
  - Either mode + `PYSCF_DTYPE=f32` → wgpu probe satisfied without `shader-f64`.
- **D-10:** Probe implementation: wgpu via `client.properties().supports_type(ElemType::Float(FloatKind::F64))` (xcfun precedent — see `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/src/runtime/wgpu.rs` lines 60-67); cuda probes via `CudaRuntime::client(&CudaDevice::default())` then same `supports_type` gate; rocm/metal analogous. Probes use `std::panic::catch_unwind` to handle dynamic-link failures. Probe outcomes cached per-process in `OnceLock<Option<Client>>`.
- **D-11:** Phase 4 (DFT) re-asserts the wgpu/shader-f64 rule on a real DFT cycle (DFT-11) but does NOT duplicate the resolver decision.
- **D-12:** Pinned git commit SHAs in `[patch.crates-io]` for sibling crates.
- **D-13:** GitHub remotes under `BectorVoom`. Remotes verified as public, non-empty: `cintx` (default branch `main`, size 129 MB), `libxc_rs` (default `main`, size 208 MB), `xcfun_rs` (default `master`, size 5 MB) [VERIFIED: `gh api repos/BectorVoom/{name}` 2026-05-10]. **No "publish remote first" task is needed** — all three already exist publicly.
- **D-14:** Nightly cross-crate matrix CI updates pinned SHAs via `cargo update -p cintx -p libxc_rs -p xcfun_rs` and rebuilds full workspace + tests under `--features gpu`.
- **D-15:** No path-dep override is shipped. Developers set their own `~/.cargo/config.toml` `[patch.crates-io]`. Documented in `CONTRIBUTING.md` as "local sibling-crate development" recipe.

### Claude's Discretion

The following are not user-decided — researcher / planner picks:

- Specific algorithm for `oracle_sum`/`oracle_dot` (pairwise tree reduction vs Kahan-Babuska vs strict left-to-right). Constraint: bit-identical across `RAYON_NUM_THREADS=1` and `=8`. **This research recommends one in §"Recommendations on Open Questions" below.**
- Lint mechanism for `forbidden-paths` and `algebra-dependency-wall` (dylint vs xtask grep + `cargo metadata | jq` vs `cargo-deny` custom rules). **Recommendation below.**
- Stub-crate skeleton shape for the 12 method/façade crates. **Recommendation below.**
- `WorkspacePool` (FOUND-03) shape. **Recommendation below.**
- `panic = "abort"` scope. **Recommendation below.**

### Deferred Ideas (OUT OF SCOPE)

- **`python/pyscf/__init__.py` re-export shim** — Phase 3 (BIND-02).
- **Maturin wheel build** — Phase 8 (DIST-02). Phase 1 only ensures `pyscf-py` exists as an empty cdylib-capable workspace member.
- **abi3-py310 wheel skeleton** — Phase 3 (BIND-01).
- **Tensor-arena pattern in `pyscf-runtime`** — full pattern in Phase 6 (CCSD-11). Phase 1 ships only minimal `WorkspacePool` skeleton.
- **Python verbosity contract** wiring — Phase 1 establishes `tracing 0.1` infrastructure; Phase 3 wires `mol.verbose` → `tracing-subscriber`.
- **Per-backend regression suite** — Phase 8 (ORACLE-07). Phase 1 only ships `tests/backend_matrix.rs` smoke test (ALG-07).
- **`shader-f64` runtime fallback in DFT kernels** — Phase 4 (DFT-11).

## Phase Requirements

| ID | Description | Implementation answer |
|----|-------------|----------------------|
| **FOUND-01** | 15-crate workspace, only algebra+kernels touch cubecl-* | Workspace `Cargo.toml` `[workspace] members = [...]` (15 entries); ALG-06 lint enforces `cubecl-*` containment |
| **FOUND-02** | `pyscf-core` exposes universal types/traits, no compute deps | `crates/pyscf-core/src/{mole.rs,density.rs,mo_coefficients.rs,energy.rs,traits.rs,lib.rs}`; `Cargo.toml` deps = `{thiserror, serde, tracing}` only |
| **FOUND-03** | `pyscf-runtime` provides `BackendKind` + `select_backend()` + `WorkspacePool`; `gpu` umbrella feature OFF by default | `crates/pyscf-runtime/src/{backend.rs,select.rs,workspace_pool.rs,lib.rs}`; `[features] default = []`, `gpu = ["cuda", "wgpu"]`, plus per-backend features |
| **FOUND-04** | cubecl 0.10.0 exact-pinned via `[patch.crates-io]` | Workspace `Cargo.toml` `[workspace.dependencies] cubecl = "=0.10.0"` (and family); `xtask check-cubecl-pin` enforces in CI (xcfun precedent) |
| **FOUND-05** | `[profile.release-oracle]` produces FMA-free machine code; CI greps for `llvm.fmuladd` | `[profile.release-oracle]` in workspace `Cargo.toml` (inherits release, sets `lto = "thin"`, `codegen-units = 1`); `.cargo/config.toml` adds `rustflags = ["-Cllvm-args=-fp-contract=off", "-Ctarget-feature=-fma"]` for the oracle profile via `[target.'cfg(profile = "release-oracle")']` (or `[build] rustflags`); CI gate `xtask check-no-fma` ports xcfun's `check_no_fma.rs` (uses `cargo rustc --emit=asm` + demangled symbol grep over FMA mnemonics) **— `cargo-llvm-ir` does NOT exist on crates.io [VERIFIED: `cargo info cargo-llvm-ir` returns "could not find in registry" 2026-05-10]; the asm-grep approach from xcfun is the only working precedent.** |
| **FOUND-06** | `oracle_sum`, `oracle_dot`, `oracle_einsum` deterministic-ordered | `crates/pyscf-algebra/src/oracle.rs` — pairwise tree reduction with chunk-size-fixed CPU implementation (recommendation §1 below); `tests/oracle_determinism.rs` parameterizes over `RAYON_NUM_THREADS` |
| **FOUND-07** | Panic policy: `extern "C"` callbacks in `catch_unwind`; clippy denies `unwrap()` in numerical modules; release uses `panic = "abort"` | `[profile.release] panic = "abort"` AND `[profile.release-oracle] panic = "abort"` (recommendation §5); `crates/pyscf-algebra/src/lib.rs` adds `#![warn(clippy::unwrap_used)]` per crate; CI gate `xtask check-catch-unwind` greps every `extern "C"` block for surrounding `catch_unwind` |
| **FOUND-08** | `forbidden-paths` lint refuses imports from `pbc/x2c/mcscf/tdscf/adc/gw/eom/NAC/EPH` | CI gate `xtask check-forbidden-paths` greps `crates/**/*.rs` source files for forbidden module names (recommendation §2 below) |
| **FOUND-09** | Tracing 0.1 logging matching PySCF `lib.logger` verbosity | Workspace dep `tracing = "0.1.44"`; `crates/pyscf-runtime/src/lib.rs` exposes a `init_tracing(verbose: u8)` helper that maps 0..=9 → `LevelFilter`; subscriber installation deferred to binaries (Phase 3 wires `mol.verbose` from PyO3 entry point per recommendation §6 below) |
| **FOUND-10** | MSRV 1.92, edition 2024, Apache-2.0 LICENSE, `cargo deny` clean | Workspace `Cargo.toml` `[workspace.package] rust-version = "1.92" edition = "2024" license = "Apache-2.0"`; `LICENSE` already exists at root; `deny.toml` checked in at root (cargo-deny 0.19 config) |
| **ALG-01** | `pyscf-algebra` exposes `gemm`/`gemv`/`axpy`/`dot`/`reduce_sum`/`transpose`/`scal` typed against `AlgebraClient` enum | `crates/pyscf-algebra/src/{lib.rs,client.rs,tensor.rs,gemm.rs,gemv.rs,axpy.rs,dot.rs,reduce.rs,transpose.rs,scal.rs}` |
| **ALG-02** | GEMM via `cubecl_matmul::launch::<R,T>(&Strategy::Auto, &client, lhs, rhs, out)`; reductions via `cubecl_reduce::reduce::<R,_>`; element-wise via `#[cube]` + `launch_unchecked` | **BLOCKER: cubecl-matmul/cubecl-reduce 0.10.0 not published; latest is 0.9.0-pre.5.** Implementation has 2 paths: (a) pin `cubecl-matmul = "=0.9.0-pre.5"` + `cubecl-reduce = "=0.9.0-pre.5"` and verify ABI compat with cubecl-runtime 0.10.0 in a Phase 1 build-verification task, OR (b) hand-roll GEMM/reduce as `#[cube]` kernels against cubecl 0.10.0 directly (cintx's actual approach — see cintx-cubecl/src/{math,kernels}/). **Recommendation: path (a)** because the user-prescriptive `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` (CONTEXT canonical reference) names cubecl-matmul explicitly. Path (a) is contingent on the build-verification task succeeding. |
| **ALG-03** | `gpu` umbrella OFF by default; per-backend features `cuda`, `wgpu`, `rocm`, `metal`; `pyscf-{cli,py}` re-export | Workspace `Cargo.toml` `[features] default = []`, `gpu = ["pyscf-algebra/gpu", "pyscf-runtime/gpu"]`; `crates/pyscf-algebra/Cargo.toml` `[features] gpu = ["cuda", "wgpu"]`, `cuda = ["dep:cubecl-cuda", ...]` etc. (cintx-cubecl/Cargo.toml verbatim) |
| **ALG-04** | `PYSCF_BACKEND` (case-insensitive) env resolver; auto chain `cuda → rocm → metal → wgpu → cpu`; unrecognised → CPU + `tracing::warn!` | `crates/pyscf-runtime/src/select.rs` — direct port of `xcfun-gpu/src/auto_backend.rs` with the priority chain edit (cuda first per D-07; xcfun's order is rocm-first per its D-05) |
| **ALG-05** | Eigh/Cholesky/QR/SVD route to `faer 0.24` on CPU regardless of GPU backend | `crates/pyscf-algebra/src/host_fallback.rs` — `pub fn eigh(client: &AlgebraClient, m: &Tensor) -> (Tensor, Tensor)` with comment `// host-fallback per ALG-05`; uses `faer::Mat::self_adjoint_eigen()`. **NOTE: `faer-ext 0.7.1` requires faer 0.23, not 0.24** — drop faer-ext and round-trip via `Vec<f64>` per STATE.md fallback |
| **ALG-06** | Dependency-wall: only `pyscf-algebra` and `pyscf-runtime` may declare `cubecl-*` deps | CI gate `xtask check-dependency-wall` runs `cargo metadata --format-version 1 --no-deps` and asserts every workspace package's `dependencies[].name` against an allowlist (xcfun's `check_boundaries.rs` is the verbatim precedent — recommendation §2 below) |
| **ALG-07** | `tests/backend_matrix.rs` smoke test on a known input across compiled-in backends; agree with CPU reference within per-backend tolerance | `crates/pyscf-algebra/tests/backend_matrix.rs` — 256×256 GEMM + AXPY + reduce-sum, asserts each compiled backend matches CPU within `1e-12` (CPU oracle), `1e-10` (CUDA), `1e-9` (WGPU/Metal) |
| **ALG-08** | Backend resolution observable: `tracing::info!("pyscf-algebra: backend={resolved} (env={raw_env_value or unset}, dtype={f32|f64})")` at PyO3 entry-point start | `crates/pyscf-algebra/src/client.rs` — `AlgebraClient::log_resolution()` method called once per `select_backend()` invocation; gated by `mol.verbose` ≥ 4 contract (Phase 3 wires) |
| **ORACLE-01** | `pyscf-oracle` uses `pyo3::Python::with_gil` to drive upstream PySCF; only in `dev-dependencies` | `crates/pyscf-oracle/Cargo.toml` is the SHELL only in Phase 1: empty `lib.rs` with `// TODO: implemented in Phase 3` (per Claude's discretion §3 — stub crate). Phase 3 wires it. **NOTE: ORACLE-01 mapping in REQUIREMENTS.md says Phase 1 — but the *first user* of the oracle harness is Phase 3 SCF tests. Phase 1 only needs the crate exist as an empty member.** |
| **ORACLE-05** | Nightly cross-crate matrix CI rebuilds cintx + libxc_rs + xcfun_rs + pyscf_rs together against the cubecl pin | `.github/workflows/nightly-cross-crate.yml` — schedule cron `0 6 * * *`; matrix on `[ubuntu-latest, ubuntu-latest-with-gpu (self-hosted)]`; runs `cargo update -p cintx -p libxc_rs -p xcfun_rs && cargo build --workspace --features gpu && cargo test --workspace --features gpu`; failures auto-create issue. Recommendation §13 below covers shape. |
| **ORACLE-09** | Floating-point determinism: `RAYON_NUM_THREADS=1`, `mol.lib.num_threads(1)`, `release-oracle` profile | CI job `oracle-determinism` (in `.github/workflows/ci.yml`) sets these env vars and runs `cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism` |

**REQ with NO clean implementation answer (planning blocker):** None — but **ALG-02 has a registry-shape blocker** (cubecl-matmul/cubecl-reduce version skew) that needs a Phase 1 build-verification task before the implementation path can be locked.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Universal types (Mole, Density, Energy) | `pyscf-core` | — | Foundation; zero compute deps; no cubecl |
| `BackendKind` + `select_backend()` | `pyscf-runtime` | — | Runtime wiring; needs cubecl-cuda/wgpu/hip for probes BUT not matmul/reduce |
| `WorkspacePool` (Phase 1 skeleton) | `pyscf-runtime` | — | Memory orchestration; consumed by every method crate later |
| Linear algebra primitives (gemm, axpy, reduce, etc.) | `pyscf-algebra` | — | Single-owner per ALG-06; only allowed cubecl-matmul/cubecl-reduce consumer |
| Host eigh/Cholesky/QR/SVD | `pyscf-algebra` | `faer 0.24` (external) | Routes through algebra surface but executes on CPU regardless of backend (ALG-05) |
| Deterministic reductions (oracle_sum/dot/einsum) | `pyscf-algebra` | — | Living next to reduce_sum so reviewers see both algorithms side-by-side |
| `#[cube]` element-wise kernels (axpy, scal, transpose) | `pyscf-algebra` | — | Phase 1 implements via `#[cube] launch_unchecked` per cubecl_multi_compute.md |
| Tracing infrastructure | `pyscf-runtime` (helper) | binaries (subscriber install) | Library-level: `tracing::info!` calls; subscriber installation owned by `pyscf-cli` and `pyscf-py` |
| FMA-free build profile | workspace `Cargo.toml` | `.cargo/config.toml` | Cross-cutting; profile lives in workspace manifest, rustflags in `.cargo/config.toml` |
| Four CI lints | `xtask/src/bin/*.rs` | `.github/workflows/ci.yml` | xcfun precedent: each lint is a binary in `xtask`, invoked from CI as a required gate |
| Cross-crate matrix CI | `.github/workflows/nightly-cross-crate.yml` | — | Workflow file at repo `.github/workflows/`; runs the workspace build with `cargo update -p cintx ...` |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | `=0.10.0` | Compute primitive umbrella crate | CONTEXT D-12 lockstep + sibling crates already at this pin [VERIFIED: cargo info, 2026-05-10] |
| `cubecl-runtime` | `=0.10.0` | `ComputeClient<R>`, server, handle types | Cubecl 0.10.0 family member [VERIFIED: cargo info, 2026-05-10] |
| `cubecl-cpu` | `=0.10.0` | CPU runtime (always-available substrate) | Required for the CPU arm of `AlgebraClient`; xcfun precedent [VERIFIED: cargo info, 2026-05-10] |
| `cubecl-cuda` | `=0.10.0` (optional, gated `cuda` feature) | NVIDIA runtime | Phase 1 compiles, Phase 8 actually exercises [VERIFIED: cargo info, 2026-05-10] |
| `cubecl-wgpu` | `=0.10.0` (optional, gated `wgpu`/`metal` features) | Vulkan/DX12/Metal/WebGPU runtime | The `metal` feature is an alias of `wgpu` (cintx-cubecl precedent) [VERIFIED: cargo info, 2026-05-10] |
| `cubecl-hip` | `=0.10.0` (optional, gated `rocm` feature) | AMD ROCm runtime | xcfun feature flag is `hip` aliased as `rocm`; pyscf-rs adopts `rocm` per CONTEXT [VERIFIED: cargo info, 2026-05-10] |
| `cubecl-matmul` | `=0.9.0-pre.5` | GEMM dispatch (`launch`, `Strategy`, `MatmulInputHandle`) | **Version skew**: cubecl-matmul 0.10.0 is unpublished as of 2026-05-10 [VERIFIED: `cargo info cubecl-matmul`]. Build-verification task required to confirm ABI compat with cubecl-runtime 0.10.0 |
| `cubecl-reduce` | `=0.9.0-pre.5` | Reduction dispatch (`reduce`, `instructions::Sum`, `ReduceDtypes`, `ReduceStrategy`) | Same version-skew constraint as cubecl-matmul [VERIFIED: cargo info, 2026-05-10] |
| `wgpu` | `=29.0.3` | Adapter feature inspection (`Features::SHADER_F64`, but actual API is `client.properties().supports_type`) | Pinned by cubecl-wgpu transitively; cintx-cubecl/Cargo.toml line 46 confirms `wgpu = "29.0.3"` [VERIFIED: cintx + cargo info, 2026-05-10] |
| `faer` | `0.24.0` | Host eigh / Cholesky / QR / SVD | ALG-05 single intentional host-fallback path; rust-version 1.84 matches MSRV 1.92 [VERIFIED: cargo info, 2026-05-10] |
| `tracing` | `0.1.44` | Structured logging | FOUND-09 + ALG-08 mandate this; xcfun pin is `=0.1.44` [VERIFIED: cargo info, xcfun_rs/Cargo.toml] |
| `tracing-subscriber` | `0.3.23` | Subscriber install (binary-side) | xcfun precedent; `[features]` includes `fmt` |
| `bytemuck` | `1.x` (with `derive` feature) | Zero-copy `cast_slice` for cubecl host-↔-device | cubecl_matmul_gemm_example.md uses it; cintx-cubecl pulls it |
| `thiserror` | `=2.0.18` | Error type derivation | xcfun pin; standard across siblings |
| `anyhow` | `1.0.102` | App-boundary error type (xtask only) | xcfun pin; library crates do NOT use anyhow per xcfun's `check-no-anyhow` lint |
| `serde` | `=1.0` (with `derive`) | Mole serialization (Phase 2 user; Phase 1 needs the foundation) | xcfun pin |
| `serde_json` | `=1.0.149` | JSON Mole dump/load (Phase 2 user) | xcfun pin |

### Supporting (xtask / dev-only)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cargo-deny` | `0.19.5` | License + advisory + bans gating (FOUND-10 `cargo deny clean`) | Pre-merge CI |
| `cargo-machete` | `0.9.2` | Find unused dependencies | Optional; nice-to-have for ALG-06 dep-wall validation |
| `walkdir` | `2.x` | xtask filesystem walks (forbidden-paths grep, FMA grep) | xcfun precedent in `xtask` deps |
| `toml` | `0.8` | xtask `Cargo.toml` parsing for boundary lints | xcfun precedent |
| `serde_json` | `=1.0.149` | xtask `cargo metadata` JSON parsing for dep-wall | xcfun precedent |
| `rustc-demangle` | `0.1` | xtask `check-no-fma` symbol demangling | xcfun precedent in `check_no_fma.rs` |
| `rstest` | `=0.26.1` | Parameterized tests (oracle determinism over RAYON_NUM_THREADS) | xcfun precedent |
| `approx` | `=0.5.1` | Float comparison in tests | xcfun precedent; cintx-cubecl uses it |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cubecl-matmul` 0.9.0-pre.5 | Hand-roll GEMM via `#[cube]` (cintx pattern) | Avoids version skew; loses cubecl-matmul's autotune; 200-300 LOC of `#[cube]` to maintain. **Reject** — CONTEXT canonical refs name cubecl-matmul explicitly; build-verification first |
| `cubecl-reduce` 0.9.0-pre.5 | Hand-roll reductions via `#[cube]` + shared memory | Same tradeoff as above; reductions are simpler than GEMM so lower risk |
| `faer-ext 0.7.1` | `Vec<f64>` round-trip for ndarray ↔ faer | faer-ext requires faer 0.23, blocks faer 0.24 upgrade. STATE.md already chose this fallback |
| `dylint` for `forbidden-paths` | xtask grep over `crates/**/*.rs` | dylint is a 5.0 release on crates.io but requires nightly Rust feature; xtask grep is stable, already a precedent in xcfun, wins on lowest friction |
| `cargo-deny` custom rules for dep-wall | `cargo metadata` + `serde_json` in xtask | cargo-deny's `bans` table is great for "no crate X in graph", but dep-wall is "no crate X *in this specific package's dependencies*" — that's a graph-walk, easier in xtask |
| `cargo-llvm-ir` for FMA grep | `cargo rustc -- --emit=asm` + symbol grep (xcfun precedent) | **`cargo-llvm-ir` doesn't exist on crates.io** [VERIFIED]; asm-grep is the only working path |

**Installation (workspace `[workspace.dependencies]` block):**
```toml
[workspace.dependencies]
cubecl          = { version = "=0.10.0" }
cubecl-runtime  = { version = "=0.10.0" }
cubecl-cpu      = { version = "=0.10.0" }
cubecl-cuda     = { version = "=0.10.0" }
cubecl-wgpu     = { version = "=0.10.0" }
cubecl-hip      = { version = "=0.10.0" }
cubecl-matmul   = { version = "=0.9.0-pre.5" }   # version-skew, see ALG-02 note
cubecl-reduce   = { version = "=0.9.0-pre.5" }   # same
wgpu            = "29.0.3"
faer            = "0.24.0"
tracing         = "0.1.44"
tracing-subscriber = { version = "0.3.23", features = ["fmt"] }
bytemuck        = { version = "1", features = ["derive"] }
thiserror       = "=2.0.18"
serde           = { version = "=1.0", features = ["derive"] }
serde_json      = "=1.0.149"
anyhow          = "1.0.102"
approx          = "=0.5.1"
rstest          = "=0.26.1"

# Sibling crates pinned to git revs per D-12/D-13.
[patch.crates-io]
cintx     = { git = "https://github.com/BectorVoom/cintx.git",     rev = "<TBD-1>" }
libxc_rs  = { git = "https://github.com/BectorVoom/libxc_rs.git",  rev = "<TBD-2>" }
xcfun_rs  = { git = "https://github.com/BectorVoom/xcfun_rs.git",  rev = "<TBD-3>" }
```

**Version verification (2026-05-10):**
- `cubecl 0.10.0` — published, latest [VERIFIED: `cargo info cubecl`]
- `cubecl-cpu/cuda/hip/wgpu/runtime/core/ir/macros/common 0.10.0` — all published, latest [VERIFIED]
- `cubecl-matmul`, `cubecl-reduce` — latest published is `0.9.0-pre.5` (NOT 0.10.0) [VERIFIED]
- `wgpu 29.0.3` — published, latest [VERIFIED]
- `faer 0.24.0` — published, latest, MSRV 1.84 [VERIFIED]
- `faer-ext 0.7.1` — published, but DEPENDS ON faer 0.23 [VERIFIED via docs.rs source]
- `pyo3 0.28.3` — published, latest (Phase 3 will pin) [VERIFIED]
- `tracing 0.1.44` — published, latest [VERIFIED]
- `cargo-deny 0.19.5` — published, latest [VERIFIED]
- **`cargo-llvm-ir` — does NOT exist on crates.io [VERIFIED: error from `cargo info`]**

## Architecture Patterns

### System Architecture Diagram

```text
                                    ┌────────────────────────────────────────────────────┐
                                    │  Python user / pyscf-py PyO3 entry point (Phase 3) │
                                    └────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                              [Phase 1 surface starts here]
                                                          │
                                                          ▼
                          ┌──────────────────────────────────────────────────────┐
                          │  pyscf-runtime::select_backend()                     │
                          │    reads PYSCF_BACKEND, PYSCF_DTYPE                  │
                          │    probes cuda → rocm → metal → wgpu → cpu           │
                          │    emits tracing::info! per probe (D-07)             │
                          │    returns Result<AlgebraClient, BackendError>       │
                          └──────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                          ┌──────────────────────────────────────────────────────┐
                          │  pyscf-algebra::AlgebraClient { Cpu | Cuda | Wgpu    │
                          │     | Rocm | Metal-aliases-Wgpu }  (D-04)            │
                          │     dtype: DType::F32 | DType::F64                   │
                          │     OnceLock<Option<ComputeClient<R>>> per arm       │
                          └──────────────────────────────────────────────────────┘
                                                          │
                          ┌─────────────┬─────────────────┼──────────────────┬─────────────┐
                          ▼             ▼                 ▼                  ▼             ▼
                  ┌────────────┐ ┌────────────┐  ┌─────────────────┐ ┌─────────────┐ ┌────────────────┐
                  │ gemm via   │ │ reduce_sum │  │ axpy / scal /   │ │ oracle_sum /│ │ eigh/cholesky/ │
                  │ cubecl-    │ │ via cubecl-│  │ transpose via   │ │ oracle_dot /│ │ qr/svd via     │
                  │ matmul     │ │ reduce     │  │ #[cube] +       │ │ oracle_     │ │ FAER 0.24      │
                  │ launch     │ │ reduce::<R,│  │ launch_unchecked│ │ einsum on   │ │ (HOST always,  │
                  │ ::<R,T>(.. │ │ Sum>(...)  │  │ (cubecl_multi_  │ │ CPU only,   │ │ ALG-05; copies │
                  │ Strategy:: │ │ (cubecl_   │  │ compute.md)     │ │ pairwise    │ │ down/up if GPU │
                  │ Auto, ...) │ │ reduce_sum │  │                 │ │ tree-reduce │ │ client)        │
                  │ (ALG-02)   │ │ .md)       │  │                 │ │ (FOUND-06)  │ │                │
                  └────────────┘ └────────────┘  └─────────────────┘ └─────────────┘ └────────────────┘
                          │             │                 │                  │             │
                          └─────────────┴─────────────────┴──────────────────┴─────────────┘
                                                          │
                                                          ▼
                                  ┌──────────────────────────────────────────────────┐
                                  │  cubecl::server::Handle (opaque BufferId, D-05)  │
                                  │  Tensor { id, shape, dtype } returned to caller  │
                                  └──────────────────────────────────────────────────┘
                                                          │
                                                          ▼
              [Phase 2+ method crates (gto/scf/dft/...) consume Tensor only — never name a cubecl::* type]
```

### Recommended Project Structure
```
pyscf_rs/                                  # repo root (ALREADY EXISTS)
├── Cargo.toml                             # NEW: workspace manifest (15 members + xtask)
├── Cargo.lock                             # NEW: committed (workspace has binaries via xtask + cdylib via pyscf-py)
├── .cargo/
│   └── config.toml                        # NEW: rustflags for release-oracle (FMA-off)
├── deny.toml                              # NEW: cargo-deny config (FOUND-10)
├── CONTRIBUTING.md                        # MODIFIED: add "local sibling-crate development" recipe (D-15)
├── LICENSE                                # ALREADY EXISTS (Apache-2.0)
├── pyscf/                                 # UNTOUCHED (upstream Python tree)
├── pyproject.toml                         # UNTOUCHED
├── docs/manual/Cubecl/                    # UNTOUCHED (read-only reference)
├── crates/
│   ├── pyscf-core/                        # NON-STUB — universal types + traits
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── mole.rs                    # struct Mole (skeleton; Phase 2 fills)
│   │       ├── density.rs                 # struct Density
│   │       ├── mo_coefficients.rs         # struct MOCoefficients
│   │       ├── amplitudes.rs              # struct Amplitudes
│   │       ├── basis_set.rs               # re-export of cintx_core::BasisSet (zero-copy per GTO-11; Phase 2 wires)
│   │       ├── energy.rs                  # newtype Energy
│   │       ├── traits.rs                  # trait Method, Scf, KohnSham, PostScf, Gradient, IntegralEngine
│   │       └── error.rs                   # PyscfRsError thiserror enum
│   ├── pyscf-runtime/                     # NON-STUB — BackendKind, select_backend, WorkspacePool
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── backend.rs                 # enum BackendKind { Cpu, Cuda, Wgpu, Rocm, Metal } cfg-gated
│   │       ├── select.rs                  # fn select_backend() (port of xcfun-gpu/auto_backend.rs)
│   │       ├── probe/                     # per-backend probe modules (port of xcfun-gpu/runtime/)
│   │       │   ├── mod.rs
│   │       │   ├── cpu.rs
│   │       │   ├── cuda.rs                # cfg(feature = "cuda")
│   │       │   ├── wgpu.rs                # cfg(feature = "wgpu")
│   │       │   └── hip.rs                 # cfg(feature = "rocm")
│   │       ├── workspace_pool.rs          # struct WorkspacePool { budget_bytes, pool: Mutex<Vec<...>> } — minimal Phase 1 skeleton
│   │       ├── tracing_init.rs            # fn init_tracing(verbose: u8)
│   │       └── error.rs                   # BackendError thiserror enum
│   ├── pyscf-algebra/                     # NON-STUB — single owner of cubecl-* deps
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── client.rs                  # enum AlgebraClient + DType + log_resolution()
│   │       ├── tensor.rs                  # Tensor + BufferId opaque newtype
│   │       ├── gemm.rs                    # pub fn gemm(c, a, b, out) → cubecl_matmul::launch dispatch
│   │       ├── gemv.rs                    # gemv (rank-1 case of GEMM; can call gemm internally)
│   │       ├── axpy.rs                    # element-wise via #[cube]
│   │       ├── scal.rs                    # element-wise via #[cube]
│   │       ├── transpose.rs               # element-wise via #[cube]
│   │       ├── dot.rs                     # special-case reduction
│   │       ├── reduce.rs                  # reduce_sum via cubecl_reduce::reduce
│   │       ├── oracle.rs                  # oracle_sum / oracle_dot / oracle_einsum (CPU-only pairwise tree)
│   │       ├── host_fallback.rs           # eigh / cholesky / qr / svd via faer 0.24
│   │       └── error.rs                   # AlgebraError thiserror enum
│   │   └── tests/
│   │       └── backend_matrix.rs          # ALG-07 smoke test
│   ├── pyscf-kernels/                     # STUB (Phase 4 fills) — empty lib.rs + TODO comment
│   ├── pyscf-gto/                         # STUB (Phase 2 fills)
│   ├── pyscf-scf/                         # STUB (Phase 3 fills)
│   ├── pyscf-dft/                         # STUB (Phase 4 fills)
│   ├── pyscf-mp2/                         # STUB (Phase 5 fills)
│   ├── pyscf-ccsd/                        # STUB (Phase 6 fills)
│   ├── pyscf-grad/                        # STUB (Phase 7 fills)
│   ├── pyscf-geomopt/                     # STUB (Phase 7 fills)
│   ├── pyscf-py/                          # STUB (Phase 3 fills) — has [lib] crate-type = ["cdylib", "rlib"]
│   ├── pyscf-oracle/                      # STUB (Phase 3 fills) — dev-deps only
│   └── pyscf-bench/                       # STUB (Phase 8 fills)
├── crates/pyscf-rs/                       # NON-STUB — top-level façade — 15th member
│   ├── Cargo.toml                         # re-exports pyscf-{core,runtime,algebra} for `cargo add pyscf-rs`
│   └── src/
│       └── lib.rs                         # `pub use pyscf_core::*; pub use pyscf_runtime::*; ...`
├── xtask/                                 # NEW — internal build/lint helpers (xcfun pattern)
│   ├── Cargo.toml                         # not in workspace members; uses [workspace] = empty to opt out
│   └── src/
│       └── bin/
│           ├── check_no_fma.rs            # FOUND-05 grep for FMA mnemonics under release-oracle
│           ├── check_forbidden_paths.rs   # FOUND-08 grep for pbc/x2c/mcscf/...
│           ├── check_catch_unwind.rs      # FOUND-07 grep for `extern "C"` without catch_unwind
│           ├── check_dependency_wall.rs   # ALG-06 cargo-metadata graph walk
│           └── check_cubecl_pin.rs        # FOUND-04 lockstep version check
└── .github/workflows/
    ├── ci.yml                             # NEW — fmt, clippy, build, test, four lints, oracle-determinism job
    └── nightly-cross-crate.yml            # NEW — ORACLE-05 cubecl pin lockstep matrix
```

### Pattern 1: Workspace Cargo.toml shape
**What:** A `[workspace]` manifest at repo root listing all 15 member crates + `xtask`, with shared `[workspace.package]` and `[workspace.dependencies]`.

**When to use:** Always, single source of truth. xcfun_rs is the strongest precedent.

**Example sketch (workspace Cargo.toml):**
```toml
# Source: ~/Documents/workspace/xcfun_rs/Cargo.toml shape, adapted for pyscf-rs.

[workspace]
members = [
    "crates/pyscf-rs",        # top-level façade
    "crates/pyscf-core",
    "crates/pyscf-runtime",
    "crates/pyscf-algebra",
    "crates/pyscf-kernels",
    "crates/pyscf-gto",
    "crates/pyscf-scf",
    "crates/pyscf-dft",
    "crates/pyscf-mp2",
    "crates/pyscf-ccsd",
    "crates/pyscf-grad",
    "crates/pyscf-geomopt",
    "crates/pyscf-py",
    "crates/pyscf-oracle",
    "crates/pyscf-bench",
    "xtask",                  # not counted in 15-member tally; opts out via its own [workspace] = {}
]
exclude = []
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.92"
license = "Apache-2.0"
repository = "https://github.com/BectorVoom/pyscf_rs"
authors = ["pyscf-rs contributors"]

[workspace.dependencies]
# ... see "Installation" block above ...

# Workspace-wide profiles. (D-01: top-level Cargo.toml.)
[profile.release]
panic = "abort"               # FOUND-07
lto = "thin"
codegen-units = 16
strip = "symbols"

# FOUND-05: oracle profile is FMA-free; release-mode but reproducible.
# rustflags live in .cargo/config.toml because Cargo profiles can't carry them directly.
[profile.release-oracle]
inherits = "release"
panic = "abort"               # FOUND-07 (Claude's-discretion §5: enforce on both)
lto = "off"                   # avoid LTO reordering reductions across translation units
codegen-units = 1             # single CGU for fully-deterministic LLVM IR shape
opt-level = 3
debug = 1                     # enough for asm-grep symbol resolution

# Sibling-crate pin per D-12 + D-13. <TBD-N> = git rev SHA picked in PLAN.md.
[patch.crates-io]
cintx     = { git = "https://github.com/BectorVoom/cintx.git",     rev = "<TBD-1>" }
libxc_rs  = { git = "https://github.com/BectorVoom/libxc_rs.git",  rev = "<TBD-2>" }
xcfun_rs  = { git = "https://github.com/BectorVoom/xcfun_rs.git",  rev = "<TBD-3>" }
```

### Pattern 2: pyscf-algebra Cargo.toml — single cubecl-* consumer
**What:** Mirrors `cintx-cubecl/Cargo.toml` (the strongest in-family precedent). Backend feature gates each cubecl-* dep optional.

**Example sketch (`crates/pyscf-algebra/Cargo.toml`):**
```toml
# Source: ~/Documents/workspace/cintx/crates/cintx-cubecl/Cargo.toml — verbatim
# pattern, names swapped.

[package]
name = "pyscf-algebra"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
description = "Backend-agnostic linear-algebra surface for pyscf-rs (single-owner cubecl-* consumer)"

[lib]
path = "src/lib.rs"

[features]
default = ["cpu"]
cpu   = ["cubecl/cpu", "pyscf-runtime/cpu"]
wgpu  = ["dep:cubecl-wgpu", "dep:wgpu", "pyscf-runtime/wgpu"]
cuda  = ["dep:cubecl-cuda", "pyscf-runtime/cuda"]
rocm  = ["dep:cubecl-hip", "pyscf-runtime/rocm"]
# `metal = wgpu` alias — cubecl-metal does not exist on crates.io; cintx-cubecl
# precedent (Plan 16-02 D-04). Pulls in cubecl-wgpu + wgpu transitively.
metal = ["wgpu", "pyscf-runtime/metal"]
# Phase 1 (Claude's discretion) opt-in for verbose-trace probes during CI.
probe-verbose = []

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-runtime = { path = "../pyscf-runtime", default-features = false }
cubecl         = { workspace = true }
cubecl-runtime = { workspace = true }
cubecl-matmul  = { workspace = true }   # version-skew: pin =0.9.0-pre.5
cubecl-reduce  = { workspace = true }   # same
cubecl-wgpu    = { workspace = true, optional = true }
cubecl-cuda    = { workspace = true, optional = true }
cubecl-hip     = { workspace = true, optional = true }
wgpu           = { workspace = true, optional = true }
faer           = { workspace = true }   # ALG-05 host-only
bytemuck       = { workspace = true }
thiserror      = { workspace = true }
tracing        = { workspace = true }

[dev-dependencies]
approx  = { workspace = true }
rstest  = { workspace = true }
```

### Pattern 3: pyscf-algebra/src/lib.rs — AlgebraClient enum + Tensor
**What:** D-04 enum dispatch + D-05 opaque Tensor.

**Example sketch (`crates/pyscf-algebra/src/lib.rs`):**
```rust
//! pyscf-algebra: backend-agnostic linear-algebra surface for pyscf-rs.
//!
//! Phase 1 surface (FOUND-06, ALG-01..08): gemm, gemv, axpy, scal, dot,
//! reduce_sum, transpose, oracle_sum, oracle_dot, oracle_einsum, eigh,
//! cholesky, qr, svd. Eigh family routes to faer 0.24 on host (ALG-05).
//!
//! Method crates consume Tensor only and never name a cubecl::* type
//! (D-04, D-05; enforced by `xtask check-dependency-wall` per ALG-06).

#![warn(clippy::unwrap_used)]    // FOUND-07 in numerical modules
#![deny(unsafe_op_in_unsafe_fn)]

pub mod client;
pub mod tensor;
pub mod gemm;
pub mod gemv;
pub mod axpy;
pub mod scal;
pub mod transpose;
pub mod dot;
pub mod reduce;
pub mod oracle;
pub mod host_fallback;
pub mod error;

pub use client::{AlgebraClient, DType};
pub use tensor::{Tensor, BufferId};
pub use error::AlgebraError;

// Free-function re-exports — the public surface method crates call.
pub use gemm::gemm;
pub use gemv::gemv;
pub use axpy::axpy;
pub use scal::scal;
pub use transpose::transpose;
pub use dot::dot;
pub use reduce::reduce_sum;
pub use oracle::{oracle_sum, oracle_dot, oracle_einsum};
pub use host_fallback::{eigh, cholesky, qr, svd};
```

**Example sketch (`crates/pyscf-algebra/src/client.rs`):**
```rust
//! AlgebraClient — the enum-of-clients per D-04. Each arm caches a
//! ComputeClient<R> in OnceLock so repeated select_backend() calls don't
//! re-probe.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType { F32, F64 }

impl DType {
    pub fn from_env() -> Self {
        match std::env::var("PYSCF_DTYPE").as_deref().map(str::to_ascii_lowercase) {
            Ok(s) if s == "f32" => Self::F32,
            Ok(s) if s == "f64" => Self::F64,
            _ => Self::F64,    // D-08 default
        }
    }
}

/// Backend-discriminating enum. Per-arm cfg gating mirrors cintx-runtime
/// pattern (BackendKind). Each arm holds the typed compute client.
#[derive(Debug)]
pub enum AlgebraClient {
    Cpu(cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime>),
    #[cfg(feature = "cuda")]
    Cuda(cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>),
    #[cfg(feature = "wgpu")]
    Wgpu(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(cubecl::client::ComputeClient<cubecl_hip::HipRuntime>),
    // metal alias — NO separate variant. The select_backend() Metal path
    // returns AlgebraClient::Wgpu with adapter_name embedded in
    // BackendCapabilityToken (Phase 4+ extends this).
}

impl AlgebraClient {
    /// Per ALG-08 + D-08: emit one tracing::info! at every PyO3 entry-point
    /// start. `dtype` is read from env once and threaded through.
    pub fn log_resolution(&self, raw_env: Option<&str>, dtype: DType) {
        let backend_name = match self {
            Self::Cpu(_) => "cpu",
            #[cfg(feature = "cuda")] Self::Cuda(_) => "cuda",
            #[cfg(feature = "wgpu")] Self::Wgpu(_) => "wgpu",
            #[cfg(feature = "rocm")] Self::Rocm(_) => "rocm",
        };
        let env_str = raw_env.unwrap_or("unset");
        let dtype_str = match dtype { DType::F32 => "f32", DType::F64 => "f64" };
        tracing::info!(
            "pyscf-algebra: backend={backend_name} (env={env_str}, dtype={dtype_str})"
        );
    }
}
```

**Example sketch (`crates/pyscf-algebra/src/tensor.rs`):**
```rust
//! Opaque Tensor handle (D-05). Method crates name `Tensor` only, never
//! `cubecl::server::Handle`.

use crate::client::DType;

/// Opaque per-buffer ID.  Owned by the AlgebraClient arm that allocated
/// it; recovering the cubecl Handle is a per-arm match operation done
/// inside pyscf-algebra source files only.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(pub(crate) u64);

/// Public tensor newtype. Method crates pass &Tensor by reference.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: BufferId,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub fn rank(&self) -> usize { self.shape.len() }
    pub fn numel(&self) -> usize { self.shape.iter().product() }
}
```

### Pattern 4: pyscf-runtime/src/lib.rs — BackendKind + select_backend
**What:** Direct port of `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/src/{auto_backend.rs,backend.rs,runtime/wgpu.rs,runtime/cuda.rs}`.

**Example sketch (`crates/pyscf-runtime/src/lib.rs`):**
```rust
//! pyscf-runtime: BackendKind enum + select_backend() resolver +
//! WorkspacePool skeleton.
//!
//! Direct port of xcfun-gpu's auto_backend / backend / runtime modules
//! (~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/src/) with two edits:
//!   1. priority chain follows CONTEXT D-07: cuda → rocm → metal → wgpu → cpu
//!      (xcfun's order is rocm → cuda → metal → wgpu → cpu).
//!   2. PYSCF_DTYPE axis (D-08) integrated: wgpu probe gates on f64 only
//!      when PYSCF_DTYPE=f64.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod backend;
pub mod select;
pub mod probe;
pub mod workspace_pool;
pub mod tracing_init;
pub mod error;

pub use backend::BackendKind;
pub use select::{select_backend, BackendSelection};
pub use workspace_pool::WorkspacePool;
pub use error::BackendError;
pub use tracing_init::init_tracing;
```

**Example sketch (`crates/pyscf-runtime/src/select.rs`):**
```rust
//! select_backend() — env-driven resolver per D-07, D-08, D-09.
//!
//! Returns Result<AlgebraClient, BackendError>:
//!   - Ok(client) on success
//!   - Err(BackendError::Unsatisfiable {..}) ONLY for the explicit
//!     `PYSCF_BACKEND=wgpu` + `PYSCF_DTYPE=f64` + no-shader-f64 case (D-09)
//!   - All other unrecognised/uncompiled cases: tracing::warn! + fall back to CPU
//!     (ALG-04 + D-07 priority chain)

use crate::backend::BackendKind;
use crate::probe;
use pyscf_algebra::{AlgebraClient, DType};   // crate-level dependency direction OK

#[derive(Debug)]
pub struct BackendSelection {
    pub client: AlgebraClient,
    pub kind: BackendKind,
    pub raw_env: Option<String>,
    pub dtype: DType,
}

pub fn select_backend() -> Result<BackendSelection, crate::BackendError> {
    let raw_env = std::env::var("PYSCF_BACKEND").ok();
    let normalised = raw_env.as_deref().unwrap_or("cpu").to_ascii_lowercase();
    let dtype = DType::from_env();

    // Direct port of xcfun-gpu/auto_backend.rs lines 31-62, with
    // PYSCF_BACKEND name + DType axis added.
    let kind = match normalised.as_str() {
        "cpu"     => BackendKind::Cpu,
        "cuda"    => probe_cuda_or_warn(&normalised, dtype)?,
        "rocm" | "hip" => probe_rocm_or_warn(&normalised, dtype)?,
        "metal"   => probe_metal_or_warn(&normalised, dtype)?,
        "wgpu"    => probe_wgpu_or_warn(&normalised, dtype)?,   // ← D-09 hard-error path
        "auto"    => auto_resolve(dtype),
        other     => {
            tracing::warn!(
                "PYSCF_BACKEND={other:?} unrecognised; falling back to CPU. \
                 Recognised: cpu, cuda, wgpu, rocm, metal, auto."
            );
            BackendKind::Cpu
        }
    };

    let client = construct_client(kind, dtype)?;
    let sel = BackendSelection { client, kind, raw_env, dtype };
    sel.client.log_resolution(sel.raw_env.as_deref(), dtype);    // ALG-08
    Ok(sel)
}

/// Auto resolver: walks D-07 priority chain emitting one tracing::info!
/// per probe attempt. Returns CPU if every other probe fails.
fn auto_resolve(dtype: DType) -> BackendKind {
    #[cfg(feature = "cuda")]
    {
        tracing::info!("probe: cuda");
        if probe::cuda::probe(dtype) { return BackendKind::Cuda; }
        tracing::info!("probe: cuda — unavailable; skipping");
    }
    #[cfg(feature = "rocm")]
    {
        tracing::info!("probe: rocm");
        if probe::hip::probe(dtype) { return BackendKind::Rocm; }
        tracing::info!("probe: rocm — unavailable; skipping");
    }
    #[cfg(feature = "metal")]
    {
        tracing::info!("probe: metal");
        if probe::wgpu::probe_macos_metal(dtype) { return BackendKind::Metal; }
        tracing::info!("probe: metal — unavailable; skipping");
    }
    #[cfg(feature = "wgpu")]
    {
        tracing::info!("probe: wgpu");
        if probe::wgpu::probe_generic(dtype) { return BackendKind::Wgpu; }
        if dtype == DType::F64 {
            tracing::info!("probe: wgpu — adapter lacks shader-f64; skipping (f64 requested)");
        } else {
            tracing::info!("probe: wgpu — unavailable; skipping");
        }
    }
    BackendKind::Cpu
}
```

### Pattern 5: xtask runner shape — four CI lints
**What:** Each lint is a separate `[[bin]]` under `xtask/src/bin/`, invoked from CI as `cargo run -p xtask --bin check-NAME`.

**Example sketch (`xtask/Cargo.toml`):**
```toml
[package]
name = "xtask"
version = "0.0.1"
edition = "2024"
publish = false

[[bin]]
name = "check-no-fma"
path = "src/bin/check_no_fma.rs"

[[bin]]
name = "check-forbidden-paths"
path = "src/bin/check_forbidden_paths.rs"

[[bin]]
name = "check-catch-unwind"
path = "src/bin/check_catch_unwind.rs"

[[bin]]
name = "check-dependency-wall"
path = "src/bin/check_dependency_wall.rs"

[[bin]]
name = "check-cubecl-pin"
path = "src/bin/check_cubecl_pin.rs"

# Opt out of the workspace so xtask can be cargo-run from anywhere
# and so the workspace's `default-members` does not include it.
[workspace]

[dependencies]
anyhow         = "1.0.102"
serde_json     = "=1.0.149"
walkdir        = "2"
toml           = "0.8"
rustc-demangle = "0.1"
```

**Example sketch (`xtask/src/bin/check_dependency_wall.rs`):**
```rust
//! ALG-06: only `pyscf-algebra` and `pyscf-runtime` may declare cubecl-*
//! deps. CI runs `cargo metadata --format-version 1 --no-deps` and walks
//! every workspace package's normal dependencies.
//!
//! Direct port of ~/Documents/workspace/xcfun_rs/xtask/src/bin/check_boundaries.rs
//! with allowlist swapped.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::process::Command;

const CUBECL_PREFIXES: &[&str] = &[
    "cubecl",
    "cubecl-runtime", "cubecl-core", "cubecl-common", "cubecl-ir", "cubecl-macros",
    "cubecl-cpu", "cubecl-cuda", "cubecl-hip", "cubecl-wgpu",
    "cubecl-matmul", "cubecl-reduce", "cubecl-linalg", "cubecl-std",
];
const CUBECL_ALLOWED_CRATES: &[&str] = &["pyscf-algebra", "pyscf-runtime"];

fn main() -> Result<()> {
    let metadata: Value = serde_json::from_slice(
        &Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--no-deps"])
            .output()?
            .stdout
    )?;
    let mut violations: Vec<String> = Vec::new();
    let pkgs = metadata["packages"].as_array().context("packages")?;
    for pkg in pkgs {
        let name = pkg["name"].as_str().unwrap_or("");
        let deps = pkg["dependencies"].as_array().context("dependencies")?;
        for dep in deps {
            let dep_name = dep["name"].as_str().unwrap_or("");
            // `kind` is null for normal deps; skip dev-deps (oracle is allowed cubecl).
            if !dep["kind"].is_null() { continue; }
            if CUBECL_PREFIXES.contains(&dep_name) && !CUBECL_ALLOWED_CRATES.contains(&name) {
                violations.push(format!(
                    "{name}: forbidden normal dep `{dep_name}` \
                     (only {CUBECL_ALLOWED_CRATES:?} may name cubecl-* crates)"
                ));
            }
        }
    }
    if !violations.is_empty() {
        for v in &violations { eprintln!("VIOLATION: {v}"); }
        bail!("ALG-06 dependency-wall violation(s)");
    }
    println!("check-dependency-wall: PASS");
    Ok(())
}
```

**Example sketch (`xtask/src/bin/check_forbidden_paths.rs`):**
```rust
//! FOUND-08: refuse imports from upstream PySCF out-of-scope modules.
//! Greps every `crates/**/*.rs` file (skipping target/ and Cargo cache).

use anyhow::{bail, Result};
use walkdir::WalkDir;
use std::fs;

const FORBIDDEN: &[&str] = &["pbc", "x2c", "mcscf", "tdscf", "adc", "gw", "eom", "NAC", "EPH"];

fn main() -> Result<()> {
    let root = std::env::var("CARGO_MANIFEST_DIR").map(|p| {
        std::path::PathBuf::from(p).parent().unwrap().to_path_buf()
    })?;
    let crates_dir = root.join("crates");
    let mut violations = Vec::new();
    for entry in WalkDir::new(&crates_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
    {
        let txt = fs::read_to_string(entry.path())?;
        for &name in FORBIDDEN {
            // Match `use pyscf::pbc`, `pyscf_pbc::`, `from pyscf.pbc import`, etc.
            let needles = [
                format!("pyscf::{name}"), format!("pyscf_{name}"),
                format!("from pyscf.{name}"), format!("import {name}"),
            ];
            for needle in &needles {
                if txt.contains(needle) {
                    violations.push(format!("{}: contains '{}'", entry.path().display(), needle));
                }
            }
        }
    }
    if !violations.is_empty() {
        for v in &violations { eprintln!("VIOLATION: {v}"); }
        bail!("FOUND-08 forbidden-paths violation(s)");
    }
    println!("check-forbidden-paths: PASS");
    Ok(())
}
```

**Example sketch (`xtask/src/bin/check_catch_unwind.rs`):**
```rust
//! FOUND-07 / Pitfall 14: every extern "C" callback must be wrapped in
//! catch_unwind. Greps for `extern "C"` blocks and asserts a
//! `catch_unwind` literal appears within the same function body.
//! Phase 1 is mostly aspirational (no extern "C" callbacks yet); the
//! gate exists so Phase 3+ PRs are caught.

use anyhow::Result;
use walkdir::WalkDir;
use std::fs;

fn main() -> Result<()> {
    let root = std::env::var("CARGO_MANIFEST_DIR").map(|p| {
        std::path::PathBuf::from(p).parent().unwrap().to_path_buf()
    })?;
    let crates_dir = root.join("crates");
    let mut violations = Vec::new();
    for entry in WalkDir::new(&crates_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
    {
        let txt = fs::read_to_string(entry.path())?;
        // Phase-1-soft check: every line containing `extern "C" fn ` must have
        // a corresponding `catch_unwind` in the same file.
        if txt.contains("extern \"C\" fn ") && !txt.contains("catch_unwind") {
            violations.push(format!("{}: extern \"C\" fn without catch_unwind", entry.path().display()));
        }
    }
    if !violations.is_empty() {
        for v in &violations { eprintln!("VIOLATION: {v}"); }
        std::process::exit(2);
    }
    println!("check-catch-unwind: PASS");
    Ok(())
}
```

**Example sketch (`xtask/src/bin/check_no_fma.rs`):**
Direct port of `~/Documents/workspace/xcfun_rs/xtask/src/bin/check_no_fma.rs`. The only edits:
- `SCAN_TARGETS`: replace with `&[("pyscf-algebra", "pyscf_algebra", &["gemm", "axpy", "scal", "dot", "reduce", "oracle_"])]` plus pyscf-kernels when Phase 4+ adds it.
- The `cargo rustc` profile flag becomes `--profile release-oracle` (or the build is invoked with `CARGO_PROFILE=release-oracle`); the FMA-mnemonic list (`vfmadd*`, `fmadd`, etc.) is identical.

### Pattern 6: GitHub Actions CI shape
**Example sketch (`.github/workflows/ci.yml`):**
```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: ""           # CLAUDE.md: empty — no fast-math / reassociation
  RUST_BACKTRACE: 1

jobs:
  fmt:
    name: cargo fmt --check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { toolchain: stable, components: rustfmt }
      - run: cargo fmt --all -- --check

  clippy:
    name: cargo clippy -D warnings
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { toolchain: stable, components: clippy }
      - uses: Swatinem/rust-cache@v2
      - run: cargo clippy --workspace --all-targets --locked -- -D warnings

  build:
    name: cargo build --workspace
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo build --workspace --locked

  build-release-oracle:
    name: cargo build --profile release-oracle
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo build --workspace --profile release-oracle --locked

  test:
    name: cargo test --workspace
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --workspace --locked --release --no-fail-fast

  oracle-determinism:
    name: oracle reductions bit-identical (RAYON 1 vs 8)
    runs-on: ubuntu-latest
    needs: build-release-oracle
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: RAYON_NUM_THREADS=1
        env: { RAYON_NUM_THREADS: "1" }
        run: cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism -- --exact rayon_1_baseline
      - name: RAYON_NUM_THREADS=8
        env: { RAYON_NUM_THREADS: "8" }
        run: cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism -- --exact rayon_8_matches_rayon_1

  check-no-fma:
    name: FOUND-05 FMA-free release-oracle build
    runs-on: ubuntu-latest
    needs: build-release-oracle
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo run -p xtask --bin check-no-fma

  check-forbidden-paths:
    name: FOUND-08 forbidden imports
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo run -p xtask --bin check-forbidden-paths

  check-catch-unwind:
    name: FOUND-07 panic-across-FFI
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo run -p xtask --bin check-catch-unwind

  check-dependency-wall:
    name: ALG-06 cubecl-* containment
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo run -p xtask --bin check-dependency-wall

  check-cubecl-pin:
    name: FOUND-04 cubecl 0.10.0 lockstep
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo run -p xtask --bin check-cubecl-pin

  cargo-deny:
    name: FOUND-10 cargo-deny clean
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: EmbarkStudios/cargo-deny-action@v2
        with: { command: check }
```

**Example sketch (`.github/workflows/nightly-cross-crate.yml`):**
```yaml
name: Nightly cross-crate matrix (ORACLE-05)

on:
  schedule:
    - cron: "0 6 * * *"      # 06:00 UTC daily — well past US/EU evening pushes
  workflow_dispatch:

jobs:
  cross-crate-matrix:
    name: cintx + libxc_rs + xcfun_rs + pyscf_rs against cubecl 0.10.0 pin
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest]      # add gpu runner when self-hosted available
        feature: ["", "--features gpu"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Update sibling-crate revs to latest main
        run: |
          cargo update -p cintx
          cargo update -p libxc_rs
          cargo update -p xcfun_rs
      - name: Build
        run: cargo build --workspace --locked ${{ matrix.feature }}
      - name: Test
        run: cargo test --workspace --locked ${{ matrix.feature }} --no-fail-fast
      - name: Verify cubecl pin
        run: cargo run -p xtask --bin check-cubecl-pin
      - name: File issue on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner, repo: context.repo.repo,
              title: `Nightly cross-crate matrix failed (${context.sha.slice(0,7)})`,
              body: `cubecl 0.10.0 lockstep failure detected on ${{ matrix.os }} ${{ matrix.feature }}`,
              labels: ['blocker', 'cubecl-lockstep'],
            });
```

### Pattern 7: cubecl call shapes (verified against `docs/manual/Cubecl/`)

**GEMM (`pyscf_algebra::gemm`)** — direct port of `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` lines 1-67:
```rust
// Source: docs/manual/Cubecl/cubecl_matmul_gemm_example.md (canonical)
use cubecl_matmul::{launch, MatmulInputHandle, Strategy};
use cubecl_runtime::client::ComputeClient;
use cubecl_std::tensor::TensorHandle;

pub fn gemm(client: &AlgebraClient, lhs: &Tensor, rhs: &Tensor, out: &Tensor)
    -> Result<(), AlgebraError>
{
    match client {
        AlgebraClient::Cpu(c)  => gemm_inner::<cubecl::cpu::CpuRuntime>(c, lhs, rhs, out),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => gemm_inner::<cubecl_cuda::CudaRuntime>(c, lhs, rhs, out),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => gemm_inner::<cubecl_wgpu::WgpuRuntime>(c, lhs, rhs, out),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => gemm_inner::<cubecl_hip::HipRuntime>(c, lhs, rhs, out),
    }
}

fn gemm_inner<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    lhs: &Tensor, rhs: &Tensor, out: &Tensor,
) -> Result<(), AlgebraError> {
    let (m, k) = (lhs.shape[0], lhs.shape[1]);
    let n     = rhs.shape[1];
    // Reconstruct cubecl handles from BufferId. Implementation owns a
    // per-arm registry mapping BufferId → (cubecl::server::Handle, strides).
    let lhs_h = recover_handle::<R>(client, &lhs.id, &[m, k])?;
    let rhs_h = recover_handle::<R>(client, &rhs.id, &[k, n])?;
    let out_h = recover_handle::<R>(client, &out.id, &[m, n])?;
    let lhs_in = MatmulInputHandle::Normal(TensorHandle::<R, f64>::new(
        lhs_h.handle, vec![m, k], lhs_h.strides));
    let rhs_in = MatmulInputHandle::Normal(TensorHandle::<R, f64>::new(
        rhs_h.handle, vec![k, n], rhs_h.strides));
    let out_h2 = TensorHandle::<R, f64>::new(out_h.handle, vec![m, n], out_h.strides);
    launch::<R, f64>(&Strategy::Auto, client, lhs_in, rhs_in, out_h2.clone())
        .map_err(|e| AlgebraError::Cubecl(format!("matmul: {e:?}")))
}
```

**reduce_sum (`pyscf_algebra::reduce_sum`)** — direct port of `docs/manual/Cubecl/cubecl_reduce_sum.md` lines 1-83:
```rust
// Source: docs/manual/Cubecl/cubecl_reduce_sum.md (canonical)
use cubecl_reduce::{ReduceDtypes, instructions::Sum, reduce};
use cubecl_ir::{ElemType, FloatKind, StorageType};
use cubecl_core::frontend::TensorHandleRef;

fn reduce_sum_inner<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    input: &Tensor, output: &Tensor,
) -> Result<(), AlgebraError> {
    let in_alloc = recover_handle::<R>(client, &input.id, &input.shape)?;
    let out_alloc = recover_handle::<R>(client, &output.id, &output.shape)?;
    let elem_size = 8;   // f64
    let in_handle = unsafe {
        TensorHandleRef::<R>::from_raw_parts(&in_alloc.handle, &in_alloc.strides, &input.shape, elem_size)
    };
    let out_handle = unsafe {
        TensorHandleRef::<R>::from_raw_parts(&out_alloc.handle, &out_alloc.strides, &output.shape, elem_size)
    };
    reduce::<R, Sum>(
        client, in_handle, out_handle,
        0,        // reduce-axis 0
        None,     // ReduceStrategy: let CubeCL pick
        (),       // unit config for Sum
        ReduceDtypes {
            input: StorageType::Scalar(ElemType::Float(FloatKind::F64)),
            output: StorageType::Scalar(ElemType::Float(FloatKind::F64)),
            accumulation: StorageType::Scalar(ElemType::Float(FloatKind::F64)),
        },
    ).map_err(|e| AlgebraError::Cubecl(format!("reduce: {e:?}")))
}
```

**axpy / scal / transpose (`#[cube] launch_unchecked`)** — pattern from `docs/manual/Cubecl/Cubecl_multi_ compute.md` lines 1-42:
```rust
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn axpy_kernel<F: Float>(alpha: F, x: &Array<F>, y: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < y.len() {
        y[i] = alpha * x[i] + y[i];
    }
}

fn axpy_inner<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    alpha: f64, x: &Tensor, y: &Tensor,
) -> Result<(), AlgebraError> {
    let n = y.numel();
    let wg = 256usize;
    let groups = ((n + wg - 1) / wg) as u32;
    let x_h = recover_handle::<R>(client, &x.id, &x.shape)?;
    let y_h = recover_handle::<R>(client, &y.id, &y.shape)?;
    unsafe {
        axpy_kernel::launch_unchecked::<f64, R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new(wg as u32, 1, 1),
            ScalarArg::new(alpha),
            ArrayArg::from_raw_parts::<f64>(&x_h.handle, n, 1),
            ArrayArg::from_raw_parts::<f64>(&y_h.handle, n, 1),
        );
    }
    Ok(())
}
```

**wgpu f64 probe** — direct port of `xcfun_rs/crates/xcfun-gpu/src/runtime/wgpu.rs` lines 38-87:
```rust
use cubecl::Runtime;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::ComputeClient;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use std::sync::OnceLock;

pub type WgpuClient = ComputeClient<WgpuRuntime>;
static WGPU_CLIENT: OnceLock<Option<WgpuClient>> = OnceLock::new();

pub fn wgpu_with_shader_f64_available() -> bool {
    WGPU_CLIENT.get_or_init(|| {
        std::panic::catch_unwind(|| {
            let device = WgpuDevice::default();
            WgpuRuntime::client(&device)
        }).ok().and_then(|client| {
            if client.properties().supports_type(ElemType::Float(FloatKind::F64)) {
                Some(client)
            } else {
                None
            }
        })
    }).is_some()
}
```

### Anti-Patterns to Avoid
- **Generic `<R: Runtime>` leaking into method-crate APIs (D-04 violation):** every public function in `pyscf-{gto,scf,dft,mp2,ccsd,grad}` must take `&AlgebraClient` (or `&Tensor`), never `&ComputeClient<R>`.
- **Re-exporting `cubecl::*` from any crate other than `pyscf-algebra` and `pyscf-runtime`:** even a `pub use cubecl::Runtime` in `pyscf-kernels/src/lib.rs` is an ALG-06 violation.
- **`#[cube]` macro overuse on tiny helpers** (per `cubecl_macro_fanout_manual.md` §10): keep helpers free `fn` unless they're true reusable kernel stages.
- **Per-numeric-type `#[cube(launch)]`** (`cubecl_macro_fanout_manual.md` §6, §13): use `<F: Float>` and `#[define]` for f32/f64 selection.
- **`unwrap()` in any numerical module of `pyscf-algebra` or `pyscf-runtime`:** clippy `#![warn(clippy::unwrap_used)]` is per-crate (FOUND-07).
- **Importing `pyscf-py` (the cdylib) into another workspace member:** cdylibs are leaf nodes.
- **Skipping `--locked` in CI:** every `cargo` invocation in `.github/workflows/` must use `--locked` (xcfun precedent) so the cubecl pin can't drift inside CI.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GEMM dispatch on cubecl | Custom `#[cube]` matmul | `cubecl_matmul::launch::<R, T>(&Strategy::Auto, ...)` | Autotune, multi-strategy, well-tested upstream — but blocked by version skew (see ALG-02 note) |
| Reductions on cubecl | Custom `#[cube]` parallel reduction | `cubecl_reduce::reduce::<R, Sum>(...)` | Same |
| Host eigh / Cholesky / QR / SVD | Hand-roll a Lanczos/Householder | `faer 0.24` `Mat::self_adjoint_eigen()` etc. | ALG-05 explicitly delegates this; faer is the standard pure-Rust BLAS-clean choice |
| Adapter-feature querying for SHADER_F64 | Direct wgpu adapter inspection | `client.properties().supports_type(ElemType::Float(FloatKind::F64))` (xcfun pattern) | Goes through cubecl's normalized properties — survives wgpu re-exports drift |
| Determining priority chain in auto resolver | Reinvent the chain | Direct port of `xcfun-gpu/auto_backend.rs` | Already-debugged precedent; only the ordering needs editing |
| Workspace package linkage rules | dylint plugin | xtask grep + `cargo metadata | jq` | dylint requires nightly Rust + a separate `cargo dylint` step; xtask runs on stable, in-tree |
| FMA-free machine-code grep | `cargo-llvm-ir` | `cargo rustc --emit=asm` + symbol-grep (xcfun `check_no_fma.rs`) | `cargo-llvm-ir` does not exist on crates.io [VERIFIED] |
| Tensor-arena pattern | Build full arena in Phase 1 | Phase 1 ships `WorkspacePool { budget_bytes, pool: Mutex<Vec<...>> }` skeleton only; Phase 6 (CCSD-11) is the real implementation | Avoids retrofit: lock the *interface* now, defer the body |
| Bit-exact deterministic reduction | Custom Kahan-Babuška | Pairwise tree reduction with fixed chunk size (recommendation §1) | Pairwise is the standard rust-ndarray approach; Kahan is necessary only if pairwise's error bound is insufficient (it isn't for typical chemistry energies) |

**Key insight:** Phase 1 has 21 REQs but the *implementation surface* is small because the discipline is "wire the conventions, defer the bodies". The four CI lints, the FMA-free profile, and the dependency wall are all enforcement scaffolding — not chemistry.

## Recommendations on Open Questions

### §1. `oracle_sum` / `oracle_dot` algorithm: pairwise tree reduction

**Recommendation:** **Pairwise tree reduction with fixed chunk size N=128** [ASSUMED for the chunk size — N=128 is a typical default; tuning belongs in PLAN.md].

```rust
/// Source: pattern from rust-ndarray PR #577 (LukeMathWalker, 2019);
/// constraint: chunk size is FIXED, not derived from rayon thread count.
pub fn oracle_sum(xs: &[f64]) -> f64 {
    pairwise(xs, 128)    // chunk size FIXED — invariant across thread count
}

fn pairwise(xs: &[f64], chunk: usize) -> f64 {
    if xs.len() <= chunk {
        // base case: strict left-to-right
        let mut s = 0.0;
        for &x in xs { s += x; }
        s
    } else {
        let mid = xs.len() / 2;
        // associativity at this level is well-defined because the recursion
        // tree shape is determined by chunk size + input length only —
        // independent of how many threads execute it.
        pairwise(&xs[..mid], chunk) + pairwise(&xs[mid..], chunk)
    }
}
```

**Why pairwise over Kahan-Babuska:**
1. **Bit-identical across thread counts is the hard constraint** (Phase 1 success criterion 3). Pairwise's recursion-tree shape depends only on input length and chunk size — nothing thread-count-dependent. Kahan would also be deterministic *if* applied serially, but it's not parallelizable bit-identically (the running compensation term `c` depends on accumulation order).
2. **Pairwise has O(log N) error growth** vs Kahan's O(1). For the chemistry-energy sums we care about (N ≈ 1e6 elements), pairwise gives ~20 ε relative error — well below 1 µHartree at f64 precision.
3. **The cubecl-reduce primitive's CPU strategy is already pairwise-shaped.** [ASSUMED — needs verification against cubecl-reduce 0.9.0-pre.5 source. This belongs in the Phase 1 ALG-02 build-verification task.]
4. **Strict left-to-right** is the brutally-correct fallback if pairwise's deterministic-ordered guarantee turns out to be unreliable across cubecl-reduce upgrades. We surface this as the alternative in the test harness: `oracle_sum_strict_seq` is computed on rayon-1 only as a comparison oracle.

`oracle_dot(a, b)` = `oracle_sum(a.iter().zip(b).map(|(x,y)| x*y).collect())`.

`oracle_einsum` is shape-driven; for Phase 1 we ship only the binary contraction case `"ij,jk->ik"` (uses `oracle_sum` along the contracted axis) and document the path forward to general einsum at Phase 4.

**Confidence:** MEDIUM-HIGH. The pairwise approach is well-cited (rust-ndarray, LAPACK conventions, Wikipedia "Pairwise summation"). The chunk-size choice is heuristic.

### §2. `forbidden-paths` lint mechanism: xtask grep over `crates/**/*.rs` + `cargo metadata` for dep-wall

**Recommendation:** Use **xtask grep** for FOUND-08 (`check-forbidden-paths`) and **xtask `cargo metadata` graph walk** for ALG-06 (`check-dependency-wall`). Direct ports of `~/Documents/workspace/xcfun_rs/xtask/src/bin/check_boundaries.rs` (with allowlist swapped) and a new `check_forbidden_paths.rs` greppy.

**Why not dylint:** dylint requires nightly Rust + a separate `cargo dylint` step; xtask runs on stable, in-tree, no extra toolchain bootstrapping.

**Why not cargo-deny custom rules:** cargo-deny's `bans` table is great for "no crate X in the entire dep graph", but the dep-wall constraint is "crate X may appear in graph, but only as a dep of {pyscf-algebra, pyscf-runtime}". That's a per-package walk — easier in xtask via `cargo metadata --no-deps`.

**Why grep, not the syntax tree:** xcfun's check_no_fma already greps demangled symbols; check_boundaries already walks `cargo metadata` JSON. The pattern is proven, fast (sub-second), and survives cubecl upgrades. A syntax-tree approach would catch `use pyscf_pbc::bar as quux` rewrites, but the Phase 1 scope is closed (the four CI lints aren't trying to be a security boundary; they're a forcing function for code review).

**Confidence:** HIGH (sibling-crate precedent verified working).

### §3. Stub crate skeleton: empty `lib.rs` + `// TODO: implemented in Phase N` comment

**Recommendation:** Empty `lib.rs` (one line `// TODO: implemented in Phase N (REQ-IDs: ...)`) for the 12 stub crates. Phase 1 does NOT pre-emptively re-export skeleton traits from `pyscf-core`.

**Why:** PLAN.md for Phase 1 already commits to "the workspace `cargo build` passes". Skeleton trait re-exports would (a) push design decisions into Phase 1 that belong in Phase N (e.g., what's the trait shape of `Scf`?), and (b) make every Phase 1 PR review touch 12 files when it should touch 3. The `// TODO: implemented in Phase N` comment is the contract.

`pyscf-py` is the one exception: it needs `[lib] crate-type = ["cdylib", "rlib"]` so a future `cargo build --release -p pyscf-py` can produce the abi3 .so without re-architecting the workspace. Empty `lib.rs` + that one Cargo.toml setting.

`pyscf-oracle` is also slightly distinct: even though it's a Phase 3 user, its `Cargo.toml` should declare `[dev-dependencies] pyo3 = { workspace = true, features = ["auto-initialize"] }` so the dev-only restriction (ORACLE-01) is locked NOW. Empty `lib.rs` is fine.

**Confidence:** HIGH.

### §4. `WorkspacePool` shape: minimal `{ budget_bytes, pool: Mutex<Vec<Allocation>> }` skeleton

**Recommendation:** Phase 1 ships a struct with three fields and three methods, sized to be the foundation for both the Phase 6 tensor-arena AND the Phase 6 thread-pool wrapper.

```rust
//! Phase 1 minimal shape (recommendation):

pub struct WorkspacePool {
    pub budget_bytes: usize,    // PYSCF_MAX_MEMORY ceiling; default 4 GB
    pool: std::sync::Mutex<Vec<PooledAllocation>>,   // free-list of buffers; Phase 6 implements
    rayon_pool: Option<rayon::ThreadPool>,           // None in Phase 1; Phase 5+ may set
}

#[derive(Debug)]
pub(crate) struct PooledAllocation {
    /// Opaque host-side scratch buffer. Phase 6 turns this into BufferId
    /// per-backend.
    pub bytes: Box<[u8]>,
    pub size: usize,
}

impl WorkspacePool {
    pub fn new(budget_bytes: usize) -> Self {
        Self { budget_bytes, pool: Default::default(), rayon_pool: None }
    }
    pub fn from_env() -> Self {
        let budget = std::env::var("PYSCF_MAX_MEMORY").ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(4 * 1024 * 1024 * 1024);    // 4 GB default
        Self::new(budget)
    }
    /// Phase 1 stub: returns Err(MemoryLimit) if `bytes > budget_bytes`.
    /// Phase 6 implements a real pool.
    pub fn try_reserve(&self, bytes: usize) -> Result<(), BackendError> {
        if bytes > self.budget_bytes {
            return Err(BackendError::MemoryLimitExceeded { requested: bytes, limit: self.budget_bytes });
        }
        Ok(())
    }
}
```

**Why this shape:** It's a tensor-buffer pool primitive (`pool` field), a budget-aware scratchpad (`budget_bytes`), AND a thread-pool slot (`rayon_pool` — None in Phase 1; the slot exists so Phase 5+ doesn't reshape the struct). CCSD-11 in Phase 6 fills the body without changing the public surface. The four-thing combination keeps the door open without committing to any one body.

**Confidence:** MEDIUM. The exact shape is a forward-looking guess; planner may downscope `rayon_pool` to a `pub(crate)` reservation if Phase 5 turns out not to need it.

### §5. `panic = "abort"` scope: BOTH `release` AND `release-oracle`

**Recommendation:** Both. Quoted in `Pattern 1` workspace `Cargo.toml` above.

**Why:** `release-oracle` is a shipped artifact (oracle CI runs against it; PERF-01 benchmarks against it). Differing panic policy between release and release-oracle would mean the oracle build has different `extern "C"` semantics from the wheel, which is the exact bug FOUND-07 / Pitfall 14 wants to prevent. Both = symmetric; both = simpler reasoning.

### §6. `release-oracle` profile + how to make `llvm.fmuladd` impossible

**Recommendation:**
1. **`Cargo.toml` profile:** `[profile.release-oracle]` inherits release; sets `lto = "off"`, `codegen-units = 1`, `panic = "abort"`, `opt-level = 3`, `debug = 1` (line tables for asm-grep).
2. **`.cargo/config.toml`:** apply rustflags via `[build]` so they cover both library compilation and proc-macro hosts:
   ```toml
   [build]
   rustflags = [
       "-Cllvm-args=-fp-contract=off",
       "-Ctarget-feature=-fma,-fma4",
   ]
   ```
   This is xcfun's pattern (see `~/Documents/workspace/xcfun_rs/.cargo/config.toml`); applying to `[build]` is more conservative than per-profile gating but it's what cargo-config.toml actually supports cleanly. The cost is that debug builds are also FMA-free, which is *fine* for our purposes (debug builds are not perf-critical and being numerically symmetric to release is a feature, not a bug).
3. **CI grep target:** `cargo rustc --profile release-oracle -p pyscf-algebra --lib -- --emit=asm`, then walk `target/release-oracle/deps/*.s` for FMA mnemonics (the xcfun list: `vfmadd*`, `vfmsub*`, `vfnmadd*`, `vfnmsub*`, `fmadd`, `fmsub`, `fnmadd`, `fnmsub`, `fma213`, `fma231`).
4. **Belt-and-suspenders LLVM IR check (optional):** `cargo rustc --profile release-oracle -p pyscf-algebra --lib -- --emit=llvm-ir` produces `target/release-oracle/deps/pyscf_algebra-*.ll`; grep that for `llvm.fmuladd`. The roadmap names this; it's complementary to the asm grep (asm catches what survives codegen; LLVM IR catches what was emitted but might be lowered to FMA later).

**Why this pair:** `-Cllvm-args=-fp-contract=off` disables LLVM's FMA-formation pass (the source of `llvm.fmuladd` intrinsics); `-Ctarget-feature=-fma,-fma4` disables hardware FMA codegen even if some intrinsic survives. Asm grep catches what reached x86; LLVM IR grep catches what was emitted at the IR level. The two together are airtight.

**Confidence:** HIGH (xcfun precedent verified working in production CI).

### §7. Four CI lints' concrete shape

Already covered above in `Pattern 5` (xtask) and `Pattern 6` (CI yaml). Summary:

- **`unwrap()` deny in numerical modules:** `#![warn(clippy::unwrap_used)]` per-crate at the top of `crates/pyscf-algebra/src/lib.rs` and `crates/pyscf-runtime/src/lib.rs`. `cargo clippy --workspace -D warnings` upgrades the warn to deny in CI.
- **`forbidden-paths`:** `xtask check-forbidden-paths` greps all `crates/**/*.rs` for upstream module names.
- **`extern "C"` callbacks wrapped in `catch_unwind`:** `xtask check-catch-unwind` greps all `crates/**/*.rs` for `extern "C" fn` and asserts `catch_unwind` appears in the same file. (Phase 1 has zero such functions; the gate is forward-looking.)
- **Algebra dependency-wall:** `xtask check-dependency-wall` walks `cargo metadata --format-version 1 --no-deps` and asserts `cubecl-*` appears only as a normal dep of `pyscf-algebra` and `pyscf-runtime`.

### §8. Cubecl call shapes — confirmed

| Signature | Source | Verified? |
|---|---|---|
| `cubecl_matmul::launch::<R, T>(&Strategy::Auto, &client, lhs, rhs, out)` | `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` line 53 | YES (in-repo) [VERIFIED] |
| `cubecl_reduce::reduce::<R, Sum>(client, in_handle, out_handle, axis, None, (), ReduceDtypes {..})` | `docs/manual/Cubecl/cubecl_reduce_sum.md` lines 54-66 | YES (in-repo) [VERIFIED] |
| `client.create(bytes)`, `client.empty(bytes_size)`, `client.create_tensor(bytes, shape, elem_size)`, `client.empty_tensor(shape, elem_size)`, `client.read_tensor(...)`, `client.read(handles)`, `client.sync().await` | Same docs + `Cubecl_multi_ compute.md` | YES (in-repo) [VERIFIED] |
| `WgpuRuntime::client(&device)`; `client.properties().supports_type(ElemType::Float(FloatKind::F64))` | xcfun-gpu/runtime/wgpu.rs verified working in xcfun_rs CI | YES (sibling) [VERIFIED] |
| `BufferId` ownership: `cubecl::server::Handle` is the universal handle type [CITED: xcfun-gpu/pool.rs comment "verified against crates/xcfun-eval/src/functional.rs:1614-1660"] | xcfun-gpu/pool.rs | YES (sibling) [VERIFIED] |

The `MatmulInputHandle::Normal(TensorHandle::<R, T>::new(handle, shape, strides))` wrapper is the cubecl-matmul 0.9.0-pre.5 API as documented; whether it survives the unpublished 0.10.0 cubecl-matmul ABI is the subject of the Phase 1 build-verification task.

### §9. faer 0.24 host APIs — recommendation (verified via docs.rs)

```rust
use faer::Mat;

// eigh — self-adjoint (real symmetric / complex Hermitian); upstream
// PySCF's lib.eigh equivalent.
fn eigh(m: &Mat<f64>) -> (Vec<f64>, Mat<f64>) {
    let evd = m.self_adjoint_eigen(faer::Side::Lower);
    (evd.s().column_vector().iter().copied().collect(),
     evd.u().to_owned())
}

// Cholesky (LLT); positive-definite only.
fn cholesky(m: &Mat<f64>) -> Mat<f64> {
    m.llt(faer::Side::Lower).unwrap().l().to_owned()
}

// QR, no pivot.
fn qr(m: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let qr = m.qr();
    (qr.compute_q(), qr.r().to_owned())
}

// SVD, full.
fn svd(m: &Mat<f64>) -> (Mat<f64>, Vec<f64>, Mat<f64>) {
    let svd = m.svd().unwrap();
    (svd.u().to_owned(),
     svd.s_diagonal().column_vector().iter().copied().collect(),
     svd.v().to_owned())
}
```

**Upload/download glue:** since `faer-ext` is incompatible with faer 0.24, we round-trip via `Vec<f64>`:
```rust
async fn copy_down<R: cubecl::Runtime>(client: &ComputeClient<R>, t: &Tensor) -> Vec<f64> {
    let alloc = recover_handle::<R>(client, &t.id, &t.shape);
    let bytes = client.read_tensor(vec![alloc.copy_descriptor()]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

fn vec_to_mat(v: Vec<f64>, rows: usize, cols: usize) -> Mat<f64> {
    Mat::<f64>::from_fn(rows, cols, |i, j| v[i * cols + j])
}
```

**Confidence:** MEDIUM. `Mat::self_adjoint_eigen` / `Mat::llt` / `Mat::qr` / `Mat::svd` confirmed via docs.rs/faer/0.24.0; the exact return types (`Eigendecomposition`, `Llt`, `Qr`, `Svd`) and accessors may have minor name drift. PLAN.md should include a 30-min verification pass against the actual rustdoc once the workspace compiles.

### §10. wgpu probe shape — confirmed

`wgpu::Adapter::features().contains(wgpu::Features::SHADER_F64)` is the **wgpu-crate-direct** spelling but xcfun-gpu does NOT use it — it uses `client.properties().supports_type(ElemType::Float(FloatKind::F64))` because:
1. cubecl-wgpu does NOT re-export wgpu types in its public interface (verified via docs.rs/cubecl-wgpu — no `wgpu::Features` re-export).
2. cubecl normalizes the device-feature set into its own `DeviceProperties::supports_type` API. This is a more stable abstraction across cubecl pre-0.10 / 0.10 / 0.11.
3. xcfun-gpu source comments cite this explicitly (lines 30-37 of `runtime/wgpu.rs`): "the literal pattern `feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))` is the cubecl-book documentation phrasing from before the 0.10.0-pre.3 API rename. The current public API on `DeviceProperties` is `supports_type(impl Into<Type>)` — semantically identical".

**Recommendation:** copy xcfun-gpu's pattern verbatim — it's already paid the cubecl ABI-drift tax once.

### §11. Sibling-crate remotes — VERIFIED public, no publish-first task needed

All three remotes verified via `gh api repos/BectorVoom/{name}` on 2026-05-10:
- `BectorVoom/cintx` — public, default branch `main`, size 129 MB ✓
- `BectorVoom/libxc_rs` — public, default branch `main`, size 208 MB ✓
- `BectorVoom/xcfun_rs` — public, default branch `master`, size 5 MB ✓

PLAN.md needs: pick a current rev SHA for each via `git ls-remote https://github.com/BectorVoom/{name}.git refs/heads/{default}` and put in `[patch.crates-io]`. **No "publish remote first" task is needed.**

### §12. Tracing setup

**Recommendation:**
- **Library-side (Phase 1):** `pyscf-runtime/src/tracing_init.rs` exposes a helper `pub fn init_tracing(verbose: u8)` mapping 0..=9 to `LevelFilter::Off`..`LevelFilter::Trace`, but does NOT install a subscriber. Library crates emit `tracing::info!`/`warn!`/`debug!` and trust whoever holds the binary to install a subscriber.
- **Binary-side (Phase 3 `pyscf-py` PyO3 entry point + future `pyscf-cli`):** the binary calls `init_tracing(mol.verbose)` once at module init, which builds `tracing_subscriber::fmt::Subscriber::builder().with_env_filter(...).init()`.
- **`select_backend()` does NOT install a subscriber.** It only emits structured `info!` lines (per ALG-08 + D-07 probe-attempt logs). If no subscriber is installed, the events are dropped — which is correct for library-call-from-test contexts.

**Confidence:** HIGH (xcfun pattern is identical).

### §13. CI matrix design — `.github/workflows/nightly-cross-crate.yml`

Already sketched in Pattern 6 above. Summary:
- **Schedule:** `cron: "0 6 * * *"` daily — 06:00 UTC, well after US/EU evening pushes settle.
- **Matrix:** `os: [ubuntu-latest]` with a future `+ self-hosted-gpu` row gated on the runner availability; `feature: ["", "--features gpu"]`.
- **Mechanism:** `cargo update -p cintx -p libxc_rs -p xcfun_rs && cargo build --workspace --locked && cargo test --workspace --locked`.
- **GPU runner:** none in Phase 1. The `--features gpu` row builds (compiles cubecl-cuda/cubecl-wgpu) but does NOT exercise GPU at runtime — Phase 8 (PERF-04) acquires a GPU runner. Phase 1 nightly runs `cargo build --features gpu` to catch the cubecl ABI break early; runtime exercise is deferred.
- **Failure handling:** auto-issue creation via `actions/github-script@v7` with `labels: ['blocker', 'cubecl-lockstep']`.

### §14. `cargo-llvm-ir` does NOT exist; use `cargo rustc --emit=llvm-ir`

**Recommendation:** Drop `cargo-llvm-ir` from PLAN.md entirely. Use:
- `cargo rustc --profile release-oracle -p pyscf-algebra --lib -- --emit=llvm-ir` writes `target/release-oracle/deps/pyscf_algebra-*.ll`
- xtask `check-no-fma` greps both `target/release-oracle/deps/*.s` (asm) AND `target/release-oracle/deps/*.ll` (LLVM IR) for FMA mnemonics + `llvm.fmuladd` intrinsics.

This is roadmap success-criterion 2's intent rephrased to use real, working tools. The roadmap text says "`cargo-llvm-ir | grep llvm.fmuladd`" but the tool doesn't exist; the intent (ban `llvm.fmuladd`) is preserved by the rustc-direct emission.

## Project Constraints (from CLAUDE.md)

`./CLAUDE.md` does not exist in the repo. There are no project-wide CLAUDE.md directives to copy. PROJECT.md and ROADMAP.md cross-cutting-concerns section are the de-facto constraints, and they're already absorbed into CONTEXT.md decisions D-01..D-15.

## Runtime State Inventory

> Phase 1 is greenfield (per CONTEXT.md "code_context": "No Rust code exists yet. Phase 1 is greenfield"). Nothing to inventory.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified by `ls /home/user/Documents/workspace/pyscf_rs` shows no `target/`, no `.lock` files, no SQLite databases | none |
| Live service config | None — verified no Datadog/Tailscale/Cloudflare touchpoints in repo | none |
| OS-registered state | None | none |
| Secrets/env vars | None at this phase. New env vars **introduced** by Phase 1: `PYSCF_BACKEND` (D-04), `PYSCF_DTYPE` (D-08), `PYSCF_MAX_MEMORY` (read by WorkspacePool, recommendation §4), `RAYON_NUM_THREADS` (FOUND-09 / ORACLE-09 read by oracle CI) | document in CONTRIBUTING.md |
| Build artifacts | None — repo currently contains no `target/` directory | none |

## Common Pitfalls

### Pitfall 1: cubecl-matmul / cubecl-reduce 0.10.0 vs 0.9.0-pre.5 ABI break

**What goes wrong:** Pin `cubecl-matmul = "=0.9.0-pre.5"` alongside `cubecl-runtime = "=0.10.0"` and the `MatmulInputHandle` / `TensorHandle` ABI silently breaks at link time or first launch, producing inscrutable cubecl-runtime panics.
**Why it happens:** Tracel-AI's cubecl crates evolve as a single workspace; the pre-release suffix tracks an older `cubecl-runtime` ABI that may have field-reordered between 0.9.0-pre.5 and 0.10.0.
**How to avoid:** Phase 1 includes a build-verification task: write `crates/pyscf-algebra/tests/cubecl_matmul_smoke.rs` that does the docs/manual/Cubecl/cubecl_matmul_gemm_example.md GEMM verbatim, runs on `--features cpu`, asserts the output is correct. Failure means PLAN.md must reroute to hand-rolled `#[cube]` GEMM (cintx pattern).
**Warning signs:** any `cubecl-runtime` panic mentioning `Handle` field counts; any `cubecl_matmul` link error; cargo `--locked` failures.

### Pitfall 2: `release-oracle` profile rustflags ignored by `[build]`-level config

**What goes wrong:** Setting `rustflags = ["-Cllvm-args=-fp-contract=off"]` only under `[profile.release-oracle]` (which Cargo silently ignores — Cargo profiles can't carry rustflags), thinking the `release-oracle` build is FMA-free when it isn't.
**Why it happens:** Cargo's profile system supports many keys but `rustflags` isn't one of them. The ONLY places rustflags can live are `.cargo/config.toml` `[build]`, `[target.<cfg>]`, or the `RUSTFLAGS` env var.
**How to avoid:** Apply rustflags via `.cargo/config.toml` `[build] rustflags` (xcfun pattern). Both the asm grep and the LLVM IR grep validate the build *is* FMA-free; if either fails, the `[build] rustflags` was wrong or a developer's `~/.cargo/config.toml` is overriding it (xcfun's W13 deviation: also duplicate to `[target.'cfg(all())']` to win against user-level configs).
**Warning signs:** xtask `check-no-fma` finds an FMA mnemonic in `target/release-oracle/deps/*.s`. The grep found exactly what we feared.

### Pitfall 3: faer-ext incompatibility silently downgrading to faer 0.23

**What goes wrong:** Add `faer-ext = "0.7.1"` and let cargo resolve to faer 0.23 transitively, then write code calling faer 0.24-only methods (e.g., `Mat::self_adjoint_eigen` accessor name changed) → cargo build fails late or produces subtly wrong eigenvectors.
**Why it happens:** faer-ext 0.7.1 declared `faer = "0.23.0"` (verified via docs.rs source); cargo's solver picks the highest version in the union of constraints, but if the workspace pins `faer = "=0.24"` AND `faer-ext = "=0.7.1"`, cargo refuses to build (semver conflict).
**How to avoid:** Drop `faer-ext` entirely. Round-trip ndarray ↔ faer via `Vec<f64>`. Document in CONTRIBUTING.md as the chosen fallback (STATE.md already records this).
**Warning signs:** `cargo update` produces "no version of faer-ext that's compatible with faer 0.24"; or `cargo tree -p faer` shows two versions in graph.

### Pitfall 4: Method crate accidentally pulls cubecl via dev-dependency

**What goes wrong:** `crates/pyscf-scf/Cargo.toml` adds `cubecl = "=0.10.0"` to `[dev-dependencies]` (e.g., for a benchmark) — `xtask check-dependency-wall` skips dev-deps and the violation slips through. Six months later someone copy-pastes the dev-dep into the normal-dep block.
**Why it happens:** xcfun's `check_boundaries.rs` explicitly only checks normal deps (kind=null) — dev-deps (kind="dev") are unrestricted by design.
**How to avoid:** Document in PLAN.md that even dev-dependencies on cubecl are forbidden in method crates (use `pyscf-algebra` or `pyscf-oracle`'s test infrastructure instead). Optionally extend the lint to flag dev-deps with a softer warning.
**Warning signs:** an unexpected `cubecl-*` line in `cargo metadata` output for a method crate's dev-dependencies.

### Pitfall 5: WGPU probe panics on missing libVulkan

**What goes wrong:** `WgpuRuntime::client(&WgpuDevice::default())` panics on a CI box without any Vulkan/DX12/Metal driver — the probe doesn't return `false`, it kills the process.
**Why it happens:** wgpu's adapter init does `unwrap()` internally on some paths; the panic propagates unless caught.
**How to avoid:** Wrap the probe in `std::panic::catch_unwind` (xcfun-gpu/runtime/wgpu.rs lines 54-58 verbatim). The catch_unwind boundary turns a hostile panic into a polite `Option::None`.
**Warning signs:** CI logs showing "panicked at" inside cubecl-wgpu when running on a runner without GPU drivers.

### Pitfall 6: `tracing::info!` log lines getting suppressed in tests

**What goes wrong:** `select_backend()` emits `tracing::info!("probe: cuda")` inside a test, but `cargo test` shows no output — reviewer thinks the probe isn't running.
**Why it happens:** No tracing-subscriber is installed when running unit tests; tracing events are dropped silently.
**How to avoid:** In `crates/pyscf-runtime/tests/select.rs`, do `tracing_subscriber::fmt::try_init();` before the assertion (idempotent — multiple test files calling it are fine).
**Warning signs:** unit test "probe runs" assertion is "absence of panic", not "presence of log line" — the test passes vacuously even if the probe was skipped.

### Pitfall 7: `cargo build --workspace` succeeds but `cargo build --workspace --features gpu` fails

**What goes wrong:** Phase 1 ships with `default = []` (CPU only) and CI verifies that; nobody runs `--features gpu` in Phase 1, so the gpu-gated arms never get type-checked. Phase 4 lands and a method crate accidentally references `Cuda(_)` arm — the cuda feature has been broken since Phase 1.
**How to avoid:** CI matrix on `feature: ["", "--features gpu"]` for the build job from Phase 1 onward. The runtime exercise of GPU lands later (Phase 8), but the *compilation* of GPU arms is in Phase 1 CI.
**Warning signs:** `cargo build --workspace --features gpu` errors at end of Phase 1 acceptance review.

## Code Examples

All in §"Architecture Patterns" above. Three are direct from `docs/manual/Cubecl/` and one is direct from xcfun-gpu source. None are speculative.

## Validation Architecture

Phase 1 validation is dominated by **deterministic-by-construction tests** (rayon-1 vs rayon-8 oracle reductions; FMA-free asm grep; dependency-wall metadata walk). Hardware parity tests are minimal because Phase 1 has no chemistry kernels yet.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `cargo test` (built-in) + `rstest 0.26.1` for parameterized tests |
| Config file | none — `Cargo.toml` `[dev-dependencies]` is the entire config |
| Quick run command | `cargo test -p pyscf-algebra --lib` (sub-second once compiled) |
| Full suite command | `cargo test --workspace --locked --release --no-fail-fast` |
| Oracle profile command | `cargo test --profile release-oracle --workspace --locked --no-fail-fast` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01 | Workspace builds | build | `cargo build --workspace --locked` | ❌ Wave 0 (workspace not yet created) |
| FOUND-02 | Universal types compile | unit | `cargo test -p pyscf-core --lib` | ❌ Wave 0 |
| FOUND-03 | `select_backend()` returns CPU on unset env | unit | `cargo test -p pyscf-runtime --lib -- select::test_unset_falls_back_to_cpu` | ❌ Wave 0 |
| FOUND-04 | cubecl 0.10.0 lockstep | CI gate | `cargo run -p xtask --bin check-cubecl-pin` | ❌ Wave 0 |
| FOUND-05 | FMA-free machine code | CI gate | `cargo run -p xtask --bin check-no-fma` | ❌ Wave 0 |
| FOUND-06 | `oracle_sum` bit-identical across thread counts | unit (parameterized) | `cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism` | ❌ Wave 0 |
| FOUND-07 | `unwrap()` denied in numerical modules | clippy | `cargo clippy --workspace --all-targets -- -D warnings` | (existing) |
| FOUND-07 | `extern "C"` callbacks wrapped in catch_unwind | CI gate | `cargo run -p xtask --bin check-catch-unwind` | ❌ Wave 0 |
| FOUND-08 | Forbidden upstream-PySCF imports | CI gate | `cargo run -p xtask --bin check-forbidden-paths` | ❌ Wave 0 |
| FOUND-09 | `tracing::info!` emitted | unit (with subscriber) | `cargo test -p pyscf-runtime --lib -- tracing_init::test_emits_info` | ❌ Wave 0 |
| FOUND-10 | `cargo deny` clean | CI gate | `cargo deny check` | ❌ Wave 0 (`deny.toml` not yet created) |
| ALG-01 | `gemm` / `axpy` / `dot` etc. surface compiles | unit | `cargo test -p pyscf-algebra --lib` | ❌ Wave 0 |
| ALG-02 | `cubecl_matmul::launch` smoke test on CPU | integration | `cargo test -p pyscf-algebra --test cubecl_matmul_smoke -- --nocapture` | ❌ Wave 0 (build-verification task) |
| ALG-03 | `gpu` feature OFF by default; per-backend opt-in | build | `cargo build --workspace --locked` AND `cargo build --workspace --locked --features gpu` | ❌ Wave 0 |
| ALG-04 | `PYSCF_BACKEND=auto` resolution | unit | `cargo test -p pyscf-runtime --test select_backend -- --test-threads=1` | ❌ Wave 0 |
| ALG-05 | Eigh routes to faer; round-trip via Vec<f64> matches | unit | `cargo test -p pyscf-algebra --lib -- host_fallback::test_eigh_round_trip` | ❌ Wave 0 |
| ALG-06 | Dependency-wall: cubecl-* containment | CI gate | `cargo run -p xtask --bin check-dependency-wall` | ❌ Wave 0 |
| ALG-07 | Backend matrix smoke test | integration | `cargo test --profile release-oracle -p pyscf-algebra --test backend_matrix` | ❌ Wave 0 |
| ALG-08 | `tracing::info!` emits backend resolution | unit | `cargo test -p pyscf-algebra --lib -- client::test_log_resolution` | ❌ Wave 0 |
| ORACLE-01 | `pyscf-oracle` exists; pyo3 in dev-deps only | build | `cargo build -p pyscf-oracle` AND `cargo metadata` field check | ❌ Wave 0 |
| ORACLE-05 | Nightly cross-crate matrix CI | scheduled | `.github/workflows/nightly-cross-crate.yml` | ❌ Wave 0 |
| ORACLE-09 | Oracle profile pinned to RAYON_NUM_THREADS=1 | CI gate (matrix) | env var in `.github/workflows/ci.yml` `oracle-determinism` job | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo build --workspace --locked && cargo clippy -- -D warnings` (~30s once cached)
- **Per wave merge:** Full suite + four xtask gates + check-no-fma + cargo-deny (~5 min cold, <1 min cached)
- **Phase gate:** Full suite green + nightly cross-crate matrix green + manual review of `.cargo/config.toml` rustflags

### Wave 0 Gaps
Phase 1 IS Wave 0 — every test file in the table is greenfield. The blocker list:
- [ ] Workspace `Cargo.toml` (creates compile target for everything else)
- [ ] `crates/pyscf-{core,runtime,algebra}/Cargo.toml` + `src/lib.rs`
- [ ] All 12 stub crate `Cargo.toml` + `src/lib.rs` (per Claude's discretion §3)
- [ ] `xtask/Cargo.toml` + `src/bin/check_*.rs` (5 binaries)
- [ ] `.cargo/config.toml`
- [ ] `deny.toml`
- [ ] `.github/workflows/ci.yml` + `nightly-cross-crate.yml`
- [ ] Test infrastructure: `crates/pyscf-algebra/tests/{backend_matrix.rs,oracle_determinism.rs,cubecl_matmul_smoke.rs}`; `crates/pyscf-runtime/tests/select_backend.rs`
- [ ] CONTRIBUTING.md "local sibling-crate development" section (D-15 deliverable)

### Architectural split: deterministic-by-construction vs hardware-dependent vs CI-only

| Validation type | Phase 1 examples | Where it runs |
|---|---|---|
| **Deterministic-by-construction** (no hardware variance) | `oracle_sum` rayon-1 == rayon-8 (FOUND-06); `BackendKind::default()` returns `Cpu` (FOUND-03); `select_backend()` falls back to CPU on unrecognised env (ALG-04); `Mole`/`Density`/`Energy` round-trip serialization (FOUND-02) | Standard CI (any runner, no GPU) |
| **Host-vs-device parity (real hardware)** | `tests/backend_matrix.rs` GEMM/AXPY/reduce_sum across compiled backends (ALG-07) — Phase 1 ships only the CPU baseline; GPU rows green-listed but unexercised | Self-hosted GPU runner (Phase 8 introduces; Phase 1 stub-tests CPU only) |
| **Exclusively CI-driven (build-system invariants)** | FMA-free asm grep (FOUND-05); forbidden-paths grep (FOUND-08); `extern "C"` + catch_unwind grep (FOUND-07); cubecl pin lockstep (FOUND-04); algebra dependency-wall (ALG-06); `cargo deny` clean (FOUND-10); cross-crate nightly matrix (ORACLE-05) | Standard CI; some on schedule (`nightly-cross-crate.yml`) |

The CI-only category is the largest in Phase 1 — that's the discipline of "wire conventions before kernels". Eight of 21 REQs are validated entirely by CI gates without running any pyscf-rs code.

## Security Domain

`security_enforcement` is not explicitly set; treat as enabled per project default. Phase 1's compute-graph foundations have a small but real security surface:

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Phase 1 has no user auth |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | `PYSCF_BACKEND` and `PYSCF_DTYPE` env-var parsing — strictly whitelisted via `match` (D-07, D-08); unrecognised values are tracing::warn'd, not panic'd |
| V6 Cryptography | no | — |
| V7 Error Handling | yes | FOUND-07 panic-policy + thiserror taxonomy; Pitfall 14 catch_unwind boundaries; release builds `panic = "abort"` |
| V8 Data Protection | no | — |
| V9 Communication | no | — |
| V10 Malicious Code | yes | `cargo deny advisories` clean (FOUND-10); pinned cubecl/sibling revs (D-12) |
| V11 Business Logic | no | — |
| V12 Files & Resources | yes | xtask `check-forbidden-paths` greps source files (FOUND-08); `WorkspacePool::from_env()` parses `PYSCF_MAX_MEMORY` with `usize` saturation (no integer overflow into negative budget) |
| V14 Configuration | yes | `.cargo/config.toml` rustflags pinning is an integrity-of-build concern; xcfun's W13 deviation (duplicate to `[target.'cfg(all())']`) addresses user-level-config override risk |

### Known Threat Patterns for the cubecl-runtime stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Untrusted env-var injection (PYSCF_BACKEND=`<malicious>`) | Tampering | Whitelist match; unrecognised → CPU fallback + warn (ALG-04) |
| Process-abort via panic across FFI | Denial of Service | `panic = "abort"` AT release; `catch_unwind` at every `extern "C"` boundary (FOUND-07; CI gate) |
| Sibling-crate ABI drift inducing UB | Tampering | `[patch.crates-io]` git rev pinning + nightly cross-crate matrix CI (D-12 + D-14) |
| Dep-graph injection via `cargo update` of unrelated crate | Tampering | `--locked` in every CI invocation (xcfun precedent) |
| FMA-induced floating-point divergence (cross-platform µHartree drift) | Numerical Tampering | release-oracle profile + asm/IR grep CI gate (FOUND-05; Pitfall 1, 12) |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | Phase 1 build | ✓ (assumed; user has been building xcfun_rs) | ≥1.92 | none |
| `cargo` | Build | ✓ | (Cargo bundled with rustc) | none |
| `gh` CLI | Phase 1 task: verify sibling remotes | ✓ [VERIFIED 2026-05-10: `which gh` → `/usr/bin/gh`] | (any modern) | `git ls-remote https://...` |
| `git` | Sibling-rev pinning | ✓ | (any) | none |
| GitHub remotes BectorVoom/{cintx,libxc_rs,xcfun_rs} | `[patch.crates-io]` D-13 | ✓ [VERIFIED 2026-05-10] | (latest main/master) | none — all three exist publicly |
| `cargo-deny` | FOUND-10 CI gate | partial — installed via `cargo install cargo-deny` in CI step | 0.19.5 | inline `cargo install` step in CI |
| `cargo-llvm-ir` | (named in roadmap) | ✗ — does NOT exist on crates.io [VERIFIED 2026-05-10] | — | `cargo rustc --emit=llvm-ir` + grep (recommendation §6) |
| GPU device for Phase 1 actually-runs CUDA/wgpu testing | Phase 8 (deferred) | ✗ Phase 1 | — | not blocking — Phase 1 only requires `cargo build --features gpu` to succeed |
| Vulkan/DX12 driver for wgpu probe | wgpu probe in `select_backend()` | unknown on dev box; **probe MUST handle absence gracefully via catch_unwind** (Pitfall 5) | — | `OnceLock<Option<WgpuClient>>` caches the negative result; CI runners without GPU drivers report `false` — no failure |
| Python interpreter (for `pyscf-oracle` Phase 3+ wiring) | Phase 3 (deferred) | ✓ assumed (upstream PySCF tree exists at `pyscf/`) | 3.10+ | not blocking Phase 1 |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- `cargo-llvm-ir` — replace with `cargo rustc --emit=llvm-ir` + grep.
- `faer-ext 0.7.1` (incompatible with faer 0.24) — replace with `Vec<f64>` round-trip.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `cubecl ^0.x` semver-loose | Exact-pin `=0.10.0` (and `[patch.crates-io]` for siblings) | cubecl pre-1.0 era (whole project lifetime) | Breaking change of one cubecl-* crate must be lockstepped across cintx + libxc_rs + xcfun_rs + pyscf_rs (Pitfall 15) |
| `cargo-llvm-ir` | `cargo rustc --emit=llvm-ir/asm` + grep | unmaintained tool dropped | Recommendation §6 + xcfun precedent fills the gap |
| Hand-rolled `wgpu::Adapter::features` adapter probe | `client.properties().supports_type(ElemType::Float(FloatKind::F64))` | cubecl 0.10.0-pre.3 API rename | xcfun-gpu/runtime/wgpu.rs already paid this tax; copy verbatim |
| `lazy_static!` | `pyo3::sync::GILOnceCell` (Phase 3); `std::sync::OnceLock` (Phase 1 non-PyO3) | Rust 1.70+ stabilization of OnceLock | xcfun precedent uses OnceLock for cached probes |
| `panic = "unwind"` in release | `panic = "abort"` in release AND release-oracle | FOUND-07 + CONTEXT recommendation §5 | Tighter binary size + no unwind tables in shipped wheel |

**Deprecated/outdated:**
- `cargo-llvm-ir`: not on crates.io; do not use.
- `faer-ext` for ndarray ↔ faer interop in pyscf-rs Phase 1: incompatible with faer 0.24; round-trip via Vec<f64>.
- The `cubecl_book` 0.10.0-pre API spelling `feature_enabled(Feature::Type(...))`: replaced by `properties().supports_type(...)`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | cubecl-matmul 0.9.0-pre.5 ABI is compatible with cubecl-runtime 0.10.0 | §1 + ALG-02 | Phase 1 reroutes to hand-rolled `#[cube]` GEMM (cintx pattern). Adds ~200-300 LOC + autotune loss. Mitigated by build-verification task. |
| A2 | cubecl-reduce 0.9.0-pre.5 ABI is compatible with cubecl-runtime 0.10.0 | ALG-02 | Same as A1 but smaller surface (~100 LOC of `#[cube]` reduction). |
| A3 | Pairwise tree reduction with chunk size N=128 yields bit-identical results across rayon thread counts | recommendation §1 | Lower N (N=64) is safer if pairwise turns out non-deterministic at large input sizes; switch in PLAN.md. |
| A4 | `WorkspacePool::from_env` reading `PYSCF_MAX_MEMORY` as bytes (not megabytes) is the right convention | recommendation §4 | Upstream PySCF documents `mol.max_memory` in MB. Using bytes for the env var would surprise users. **Recommended fix:** parse as MB and multiply (planner: confirm). |
| A5 | The `metal` Cargo feature should NOT have its own AlgebraClient enum variant — Metal returns `AlgebraClient::Wgpu` with adapter name embedded | §"Pattern 3" | If downstream phases (DFT-11) want a typed `Backend::Metal` arm distinct from `Backend::Wgpu`, the recommendation is wrong. xcfun's BackendTag has 5 variants including separate Metal — pyscf-rs may want to mirror. **Suggest planner re-evaluate** in light of D-09 wgpu/Metal differential treatment. |
| A6 | Phase 1 stub crates may use empty `lib.rs` (no skeleton trait re-exports) without breaking later phases | recommendation §3 | If Phase 2 (gto) wants to import a `Method` trait the moment it lands, an empty `pyscf-core` stub blocks that. But Phase 1 deliberately puts non-stub `pyscf-core` (FOUND-02 owns the trait set). |
| A7 | `panic = "abort"` on `release-oracle` won't break the in-process oracle harness (Phase 3) by killing the test process on a panic | recommendation §5 | Phase 3 tests with `Python::with_gil(|py| { ...rust code that panics... })` would terminate the test binary. Mitigated by `catch_unwind` at the FFI boundary (FOUND-07); but if a Rust unit test panics outside FFI, abort is correct (test framework reports failure either way). |
| A8 | The four CI lints + `cargo-deny` together take < 5 min in CI cold, < 1 min cached | §"Validation Architecture" | If lints take 15+ min, planner may parallelize differently or drop one. xcfun's CI takes ~3 min with similar lint count → A8 is conservative. |
| A9 | `cargo build --workspace --features gpu` will compile on a CI runner without GPU drivers (compiles cubecl-cuda/cubecl-wgpu but doesn't attempt to load drivers) | Pitfall 7 | xcfun_rs CI does this successfully → A9 is well-supported. |
| A10 | The cubecl 0.10.0 umbrella crate's `cubecl::cpu::CpuRuntime`, `cubecl::client::ComputeClient`, `cubecl::Runtime` re-exports are stable as named in cintx-cubecl source | §"Pattern 7" | If cubecl 0.10.0 dropped the `cpu` re-export, every cintx-cubecl call site would break too. Lockstep risk is contained. |

If all 10 assumptions resolve in the planner's favor, Phase 1 is straightforward implementation. If A1 or A2 fails, ALG-02 reroutes to hand-rolled kernels (cintx pattern) — a real but bounded re-plan.

## Open Questions

1. **Does `cubecl 0.10.0` umbrella crate's `reduce` feature pull `cubecl-reduce 0.9.0-pre.5` or `=0.10.0`?**
   - What we know: `cargo info cubecl` shows `cubecl 0.10.0` has a `reduce` feature mapping to `dep:cubecl-reduce`; but `cubecl-reduce 0.10.0` doesn't exist on crates.io.
   - What's unclear: whether the cubecl 0.10.0 manifest pins `cubecl-reduce = "0.9.0-pre.5"` (which would auto-resolve transitively) or `=0.10.0` (which would fail to build).
   - Recommendation: Phase 1 build-verification task: `cargo build --workspace --features gpu` exposes the resolution. If it fails, the `[patch.crates-io] cubecl-reduce = "=0.9.0-pre.5"` fix is the unblock.

2. **Should there be a separate `Metal` arm of `AlgebraClient`?** (See A5 above.)
   - What we know: cintx-runtime's `BackendKind` has a `Metal` arm gated on `feature = "metal"`; xcfun-gpu's `Backend` has a `Metal` variant. Both treat Metal as type-distinct from Wgpu.
   - What's unclear: whether the algebra-level dispatch needs the same distinction, given that the underlying runtime IS WgpuRuntime in both cases.
   - Recommendation: copy the sibling pattern (separate `Metal` variant) for consistency, even though dispatch matches Wgpu. Planner decides.

3. **`PYSCF_MAX_MEMORY` units: MB or bytes?** (See A4 above.)
   - What we know: Upstream PySCF's `mol.max_memory` is documented in MB.
   - What's unclear: whether pyscf-rs adopts the MB convention for the env var.
   - Recommendation: MB. Update `WorkspacePool::from_env()` accordingly: `mb * 1024 * 1024`.

4. **Should the four xtask lints run in parallel or sequential CI jobs?**
   - What we know: xcfun runs them sequentially (clippy → build → test → lints). Total ~3 min.
   - What's unclear: whether parallelism via job-matrix gains enough to justify the extra cache overhead.
   - Recommendation: parallel separate jobs (already drafted in Pattern 6); each job compiles xtask once then runs its check, so xcfun's serial ordering isn't load-bearing.

5. **Phase 1 `pyscf-py` Cargo.toml `crate-type`: `["cdylib"]` only or `["cdylib", "rlib"]`?**
   - What we know: Phase 8's wheel wants `cdylib`. Phase 1 stub doesn't strictly need `cdylib` to compile.
   - What's unclear: whether there's any benefit to declaring it now.
   - Recommendation: declare `["cdylib", "rlib"]` now so future intra-workspace consumers can `use pyscf_py;` if needed; matches PyO3 convention.

## Sources

### Primary (HIGH confidence)
- `~/Documents/workspace/pyscf_rs/.planning/phases/01-foundation/01-CONTEXT.md` — D-01..D-15 (locked decisions)
- `~/Documents/workspace/pyscf_rs/.planning/REQUIREMENTS.md` — FOUND-01..10, ALG-01..08, ORACLE-01/05/09
- `~/Documents/workspace/pyscf_rs/.planning/ROADMAP.md` — Phase 1 success criteria + cross-cutting concerns + pitfall mapping
- `~/Documents/workspace/pyscf_rs/.planning/STATE.md` — blockers (cubecl lockstep, faer-ext skew)
- `~/Documents/workspace/pyscf_rs/.planning/PROJECT.md` — vision + constraints + key decisions
- `~/Documents/workspace/pyscf_rs/docs/manual/Cubecl/Cubecl_multi_ compute.md` — `#[cube] launch_unchecked`, `client.create`/`empty`/`sync` (canonical for axpy/scal/transpose pattern)
- `~/Documents/workspace/pyscf_rs/docs/manual/Cubecl/cubecl_matmul_gemm_example.md` — `cubecl_matmul::launch::<R, T>(&Strategy::Auto, ...)` (canonical for ALG-02 GEMM)
- `~/Documents/workspace/pyscf_rs/docs/manual/Cubecl/cubecl_reduce_sum.md` — `cubecl_reduce::reduce::<R, Sum>(...)` (canonical for ALG-02 reductions + FOUND-06 oracle_sum basis)
- `~/Documents/workspace/pyscf_rs/docs/manual/Cubecl/cubecl_macro_fanout_manual.md` — `#[cube]` discipline (anti-fanout patterns)
- `~/Documents/workspace/pyscf_rs/docs/manual/Cubecl/Cubecl_vector.md` — `Array<f32>` access pattern + `#[cube(launch)]` shape
- `~/Documents/workspace/pyscf_rs/docs/manual/Cubecl/Cubecl_shared_memory.md` — `SharedMemory<f32>::new(N)` + `sync_units()` (relevant for WorkspacePool design)
- `~/Documents/workspace/cintx/Cargo.toml` — workspace shape, edition 2024, cubecl 0.10.0 pin, member layout
- `~/Documents/workspace/cintx/crates/cintx-cubecl/Cargo.toml` — per-backend feature gating (`cpu`/`wgpu`/`cuda`/`rocm`/`metal=wgpu`) — verbatim template for pyscf-algebra/Cargo.toml
- `~/Documents/workspace/cintx/crates/cintx-runtime/Cargo.toml` + `src/{lib.rs,options.rs,workspace.rs}` — `BackendKind` enum + `BackendIntent`/`BackendCapabilityToken` patterns
- `~/Documents/workspace/cintx/crates/cintx-cubecl/src/{lib.rs,backend/mod.rs}` — `cubecl::client::ComputeClient<R>` per-arm cache pattern
- `~/Documents/workspace/xcfun_rs/Cargo.toml` — `[workspace.package]` shared block, `[workspace.dependencies]` cubecl pin pattern, member-incremental enabling
- `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/{Cargo.toml,src/lib.rs,src/auto_backend.rs,src/backend.rs,src/runtime/{cpu,cuda,hip,wgpu}.rs,src/pool.rs}` — direct templates for select_backend / probe modules / OnceLock client cache. Especially `runtime/wgpu.rs` lines 38-87 for the SHADER_F64 probe.
- `~/Documents/workspace/xcfun_rs/xtask/{Cargo.toml,src/main.rs,src/bin/check_no_fma.rs,src/bin/check_boundaries.rs,src/bin/check_cubecl_pin.rs}` — direct templates for the four CI lints
- `~/Documents/workspace/xcfun_rs/.cargo/config.toml` — `[build] rustflags = ["-Cllvm-args=-fp-contract=off"]` proven pattern
- `~/Documents/workspace/xcfun_rs/.github/workflows/ci.yml` — CI structure (fmt → clippy → build → test → checks); proven shape

### Secondary (MEDIUM confidence)
- [docs.rs/faer/0.24.0](https://docs.rs/faer/0.24.0/faer/) — `Mat::self_adjoint_eigen`, `Mat::llt`, `Mat::qr`, `Mat::svd` (verified 2026-05-10)
- [docs.rs/crate/faer-ext/0.7.1/source/Cargo.toml.orig](https://docs.rs/crate/faer-ext/0.7.1/source/Cargo.toml.orig) — confirms `faer = "0.23.0"` dep (compatibility blocker)
- [docs.rs/cubecl/0.10.0](https://docs.rs/cubecl/0.10.0/cubecl/) — confirms umbrella crate at 0.10.0 with `reduce`/`linalg` feature flags
- [docs.rs/cubecl-runtime/0.10.0](https://docs.rs/cubecl-runtime/0.10.0/cubecl_runtime/) — `ComputeClient` API surface
- `cargo info cubecl-matmul` (2026-05-10) — confirms latest 0.9.0-pre.5
- `cargo info cubecl-reduce` (2026-05-10) — confirms latest 0.9.0-pre.5
- `cargo info cargo-llvm-ir` (2026-05-10) — confirms NOT IN REGISTRY
- `gh api repos/BectorVoom/{cintx,libxc_rs,xcfun_rs}` (2026-05-10) — confirms all three remotes public + non-empty
- [orlp.net/blog/taming-float-sums](https://orlp.net/blog/taming-float-sums/) — pairwise vs Kahan tradeoffs
- [github.com/rust-ndarray/ndarray PR #577](https://github.com/rust-ndarray/ndarray/pull/577) — pairwise summation precedent in Rust ecosystem
- [Wikipedia: Pairwise summation](https://en.wikipedia.org/wiki/Pairwise_summation) — O(log N) error bound

### Tertiary (LOW confidence — flagged for validation)
- The exact name of faer 0.24 eigh accessor: `Mat::self_adjoint_eigen(faer::Side::Lower)` — needs runtime verification once workspace compiles
- Pairwise chunk size N=128: heuristic, no benchmarks run in this research session
- `WorkspacePool::from_env` MB-vs-bytes convention: see Open Question #3
- Whether `Backend::Metal` deserves its own AlgebraClient arm: see Open Question #2

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every crate version verified via `cargo info` 2026-05-10
- Architecture (D-04 enum dispatch, D-05 opaque Tensor): HIGH — sibling-crate precedent
- cubecl call shapes: HIGH for the documented examples; MEDIUM for the cubecl-matmul/cubecl-reduce 0.9.0-pre.5 ↔ 0.10.0 ABI compat (A1, A2)
- Pitfalls: HIGH — three of the seven catalogued ones (1, 2, 3) are confirmed in this research session via tool calls
- Recommendations on open questions: MEDIUM — pairwise reduction choice + WorkspacePool shape are forward-looking guesses
- Sibling-remote availability (D-13): HIGH — verified via `gh api` 2026-05-10

**Research date:** 2026-05-10
**Valid until:** 2026-06-09 (30 days; cubecl 0.10.0 stable ecosystem; pin re-verification recommended monthly)

## RESEARCH COMPLETE

Phase 1 implementation is well-understood. Two registry-shape blockers are surfaced — `cubecl-matmul`/`cubecl-reduce` published only at 0.9.0-pre.5 (not 0.10.0), and `faer-ext 0.7.1` requires faer 0.23 (forces Vec<f64> round-trip for ALG-05). Sibling remotes verified public. xcfun_rs and cintx provide direct templates for all four CI lints, the auto-backend resolver, the SHADER_F64 probe, and the `[build] rustflags` FMA-off pattern. Five Claude's-Discretion questions answered with concrete recommendations: pairwise tree reduction (chunk=128), xtask-grep + cargo-metadata for lints, empty-lib.rs stubs, three-field WorkspacePool skeleton, panic=abort on both release profiles. The 21 Phase 1 REQs each map to a concrete file/module/test — none are unmappable. Two open questions resolved as recommendations (`PYSCF_MAX_MEMORY` units, `Metal` enum-arm question) flagged for planner discretion.
