---
phase: 01-foundation
plan: 04
subsystem: foundation
tags: [pyscf-algebra, AlgebraClient, Tensor, select_backend, cubecl, oracle, pairwise, FOUND-01, FOUND-06, ALG-01, ALG-02, ALG-03, ALG-04, ALG-05, ALG-07, ALG-08]

# Dependency graph
requires:
  - phase: 01-foundation/01
    provides: workspace Cargo.toml with cubecl 0.10.0 + cubecl-matmul/cubecl-reduce 0.9.0-pre.5 + faer 0.24 pins; pyscf-algebra member declared
  - phase: 01-foundation/02
    provides: pyscf-core (Density, Energy, MOCoefficients, Mole) — re-exported through pyscf-rs façade
  - phase: 01-foundation/03
    provides: pyscf-runtime (BackendKind, DType, BackendError, probe::{cpu,cuda,wgpu,hip}, WorkspacePool) — wired by select_backend
provides:
  - "pyscf-algebra crate at crates/pyscf-algebra/ — AlgebraClient enum, Tensor opaque surface, select_backend resolver, 7 algebra primitives + 4 host_fallback signatures, FOUND-06 pairwise oracle"
  - "AlgebraClient enum-of-clients with Cpu unconditional + cfg-gated Cuda/Wgpu/Rocm arms (D-04); manual Debug impl printing only kind name"
  - "Tensor + BufferId opaque handles (D-05) — method crates never name a cubecl::* type; Tensor::placeholder helper for integration tests"
  - "select_backend() resolver: D-07 priority chain (cuda → rocm → metal → wgpu → cpu), per-probe tracing::info!, D-09 wgpu+f64 hard-error path, ALG-04 unrecognised → warn + Cpu, ALG-08 mandatory log line"
  - "7 algebra primitives (gemm/gemv/axpy/scal/transpose/dot/reduce_sum) — Phase 1 ships function signatures returning AlgebraError::NotYetImplemented{phase:2}; Phase 2 wires cubecl_matmul/cubecl_reduce dispatch"
  - "4 host_fallback signatures (eigh/cholesky/qr/svd) — Phase 1 ships signatures; Phase 3/6/7 wire faer 0.24 round-trip"
  - "FOUND-06 pairwise tree reduction: oracle_sum/oracle_dot/oracle_einsum with PAIRWISE_CHUNK=128 — bit-deterministic by construction (recursion-tree shape depends only on input length)"
  - "Top-level pyscf-rs façade: re-exports pyscf-{core,runtime,algebra} as named modules + convenience type re-exports (Density, Energy, Mole, AlgebraClient, BackendKind, ...)"
  - "4 integration tests, 15 tests total, all passing: oracle_determinism (5), cubecl_matmul_smoke (1), backend_matrix (2), select_backend (7)"
affects: [02-gto, 03-scf, 04-dft, 05-mp2, 06-ccsd, 07-grad, 07-geomopt, 06-CI]

# Tech tracking
tech-stack:
  added:
    - "cubecl-matmul 0.9.0-pre.5 (Strategy::Auto symbol verified by cubecl_matmul_smoke test — Pitfall 1 ABI gate)"
    - "cubecl-reduce 0.9.0-pre.5 (transitively wired; reduce_sum signature only in Phase 1)"
    - "faer 0.24.0 (workspace dep declared; bodies stubbed to NotYetImplemented; Phase 3 wires self_adjoint_eigen)"
    - "bytemuck 1 + thiserror 2.0.18 (regular deps in pyscf-algebra)"
  patterns:
    - "AlgebraClient enum-of-clients dispatch (D-04) with #[cfg(feature)]-gated arms — match-dispatch for primitive bodies"
    - "Opaque Tensor + BufferId surface (D-05) — method crates only see {id, shape, dtype}; never name cubecl::* types"
    - "Phase-1 NotYetImplemented stubs lock the public surface; Phase 2+ fills the bodies. Method crates can compile against the surface today"
    - "Pairwise tree reduction (FOUND-06): chunk size const = 128, recursion tree depends ONLY on input length so result is invariant under thread count, scheduler, and rayon partition"
    - "Manual Debug impl on AlgebraClient (cubecl 0.10.0 ComputeClient<R> does not implement Debug) — prints only kind name for tracing"
    - "Per-probe tracing::info! line in auto_resolve (D-07 observability)"

key-files:
  created:
    - "crates/pyscf-algebra/Cargo.toml"
    - "crates/pyscf-algebra/src/lib.rs"
    - "crates/pyscf-algebra/src/client.rs"
    - "crates/pyscf-algebra/src/tensor.rs"
    - "crates/pyscf-algebra/src/error.rs"
    - "crates/pyscf-algebra/src/select.rs"
    - "crates/pyscf-algebra/src/gemm.rs"
    - "crates/pyscf-algebra/src/gemv.rs"
    - "crates/pyscf-algebra/src/axpy.rs"
    - "crates/pyscf-algebra/src/scal.rs"
    - "crates/pyscf-algebra/src/transpose.rs"
    - "crates/pyscf-algebra/src/dot.rs"
    - "crates/pyscf-algebra/src/reduce.rs"
    - "crates/pyscf-algebra/src/oracle.rs"
    - "crates/pyscf-algebra/src/host_fallback.rs"
    - "crates/pyscf-algebra/tests/oracle_determinism.rs"
    - "crates/pyscf-algebra/tests/cubecl_matmul_smoke.rs"
    - "crates/pyscf-algebra/tests/backend_matrix.rs"
    - "crates/pyscf-algebra/tests/select_backend.rs"
  modified:
    - "crates/pyscf-rs/Cargo.toml (added 3 path deps)"
    - "crates/pyscf-rs/src/lib.rs (FOUND-01 façade body — re-exports of core/runtime/algebra)"

key-decisions:
  - "AlgebraClient drops #[derive(Debug)] in favour of a manual std::fmt::Debug impl because cubecl 0.10.0's ComputeClient<R> does not implement Debug. Manual impl prints only the backend kind name (`AlgebraClient { kind: \"cpu\" }`) — sufficient for tracing diagnostics and stable across cubecl backend internals"
  - "select_backend() lives in pyscf-algebra, NOT pyscf-runtime — adopting Plan 03's deferred decision. Resolution avoids a circular dep (pyscf-runtime would need pyscf_algebra::AlgebraClient, but pyscf-algebra already depends on pyscf-runtime)"
  - "Phase 1 algebra primitives (gemm/gemv/axpy/scal/transpose/dot/reduce_sum) ship signature-only stubs returning AlgebraError::NotYetImplemented{phase:2,what:...}. The wave-3 success bar is 'public surface compiles and is callable'; the actual cubecl_matmul/cubecl_reduce dispatch wiring lands at the first GTO call site in Phase 2 (FOUND boundary clarified by RESEARCH §8: Phase 1 is the surface lock-in; Phase 2 is the first compute load)"
  - "host_fallback (eigh/cholesky/qr/svd) ships signature-only stubs. Phase 3 (SCF Fock-matrix diagonalization) wires faer::Mat::self_adjoint_eigen for eigh; Phase 6 (CCSD intermediate canonicalization) wires qr; Phase 7 (gradient null-space projection) wires svd. The faer = { workspace = true } dep is declared so the dep-wall lint catches it but no faer:: types are imported by Phase 1 source"
  - "oracle_einsum supports only the binary contraction pattern 'ij,jk->ik' in Phase 1 (RESEARCH §1 explicit boundary). Other patterns return None; Phase 4 (DFT) extends to multi-tensor patterns when first needed"
  - "PAIRWISE_CHUNK = 128 is documented as load-bearing for bit-exact reproducibility — 'changing this breaks bit-exact compatibility with prior runs; never modify without updating chemistry-corpus regression baselines'. The number is preserved verbatim from RESEARCH §1"
  - "BackendError import in select.rs is gated #[cfg(feature = \"wgpu\")] because the only construction site (D-09 hard-error path) lives inside the wgpu arm. Avoids unused-import warning on cpu-only builds"

patterns-established:
  - "Per-task absolute-path discipline: every Edit/Write uses /home/user/Documents/workspace/pyscf_rs/.claude/worktrees/agent-{ID}/... rather than relative paths to defeat cwd-drift between Bash calls (#3097)"
  - "Per-commit `git -C $WT` discipline + HEAD-on-worktree-agent-* assertion before staging — same pattern Plan 03 documented"
  - "Transient root Cargo.toml edit pattern (comment out [patch.crates-io] block) for in-isolation build verification — restored byte-identically (sha256-verified) before any commit; spurious Cargo.lock removed. Plan 03 documented this same workaround"
  - "Phase-1 stub idiom: `pub fn foo(...) -> Result<..., AlgebraError> { Err(AlgebraError::NotYetImplemented { phase: N, what: \"...\" }) }` — locks signature, defers body to phase N, callable today (returns expected error variant)"

requirements-completed: [FOUND-01, FOUND-06, ALG-01, ALG-02, ALG-03, ALG-04, ALG-05, ALG-07, ALG-08]
# FOUND-01 (top-level façade) is now complete: pyscf-rs/src/lib.rs re-exports the three foundation
#   crates as named modules + convenience type re-exports.
# ALG-02 (Tensor opaque handle) and ALG-03 (CPU default) are also implicitly verified by the
#   compile-and-test green of pyscf-algebra default features = ["cpu"].

# Metrics
duration: ~25min
completed: 2026-05-10
---

# Phase 01 Plan 04: pyscf-algebra Algebra Surface Summary

**`pyscf-algebra` crate shipped — the single-owner cubecl-matmul/cubecl-reduce consumer (alongside pyscf-runtime) that locks the algebra public surface for the entire workspace. AlgebraClient enum-of-clients with cfg-gated arms, opaque Tensor/BufferId surface, select_backend() resolver implementing D-07 priority chain + D-09 wgpu+f64 hard-error + ALG-04 fallback + ALG-08 log line, 7 Phase-1-stubbed algebra primitives, 4 host_fallback signatures, FOUND-06 pairwise oracle reduction with PAIRWISE_CHUNK=128, top-level pyscf-rs façade (FOUND-01), and 15 passing integration tests across 4 files. `cargo build --workspace` succeeds end-to-end (Roadmap success criterion 1).**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-05-10T03:55Z (after worktree HEAD assertion)
- **Completed:** 2026-05-10T04:20Z
- **Tasks:** 3
- **Files created:** 19 (1 Cargo.toml + 14 source modules + 4 integration tests)
- **Files modified:** 2 (`crates/pyscf-rs/Cargo.toml` + `crates/pyscf-rs/src/lib.rs` for façade body)

## Accomplishments

- **Locked the algebra public surface** that Phases 2–7 will import: `AlgebraClient`, `Tensor`, `BufferId`, `DType`, `BackendKind`, `BackendSelection`, `select_backend()`, and 11 free functions (`gemm`/`gemv`/`axpy`/`scal`/`transpose`/`dot`/`reduce_sum` + `eigh`/`cholesky`/`qr`/`svd`) plus the 3 oracle reductions (`oracle_sum`/`oracle_dot`/`oracle_einsum`).
- **`select_backend()` implements the full D-07/D-08/D-09 contract** with per-probe `tracing::info!` lines, wgpu+f64+no-shader-f64 → `BackendError::Unsatisfiable` hard error, ALG-04 unrecognised-token → warn + Cpu fallback, and the ALG-08 mandatory log line `pyscf-algebra: backend={resolved} (env={raw}, dtype={f32|f64})` emitted via `AlgebraClient::log_resolution`.
- **`oracle_sum`/`oracle_dot`/`oracle_einsum`** ship the FOUND-06 pairwise tree reduction with `pub const PAIRWISE_CHUNK = 128` — bit-deterministic by construction (recursion tree depends only on input length, not thread count). Verified by `tests/oracle_determinism.rs` running under `RAYON_NUM_THREADS={1,8}` (both produce 5/5 pass; bit-pattern equality of repeated calls within process).
- **`cubecl_matmul_smoke` test** verifies the Pitfall 1 ABI gate at link time: `cubecl_matmul::Strategy::Auto` is reachable as a named symbol when cubecl-matmul 0.9.0-pre.5 is wired to cubecl-runtime 0.10.0. If the symbol path drifts in a future cubecl-matmul release, the test fails to compile, surfacing the break before any GEMM call site is added.
- **Top-level `pyscf-rs` façade** re-exports `pyscf-core`, `pyscf-runtime`, and `pyscf-algebra` as named modules (`pyscf_rs::core`, `pyscf_rs::runtime`, `pyscf_rs::algebra`) plus convenience flat re-exports of the most-commonly-used types (`Density`, `Energy`, `Mole`, `MOCoefficients`, `AlgebraClient`, `BackendSelection`, `select_backend`, `Tensor`, `BufferId`, `DType`, `BackendKind`, `WorkspacePool`).
- **`cargo build --workspace` succeeds end-to-end** (Roadmap success criterion 1) with default features = `cpu`, with `--features wgpu` (cubecl-wgpu + wgpu 29.0.3 transitive deps), and with `--features gpu` (cubecl-cuda + cubecl-wgpu — cuda symbols compile without a CUDA SDK at runtime).

## Task Commits

Each task was committed atomically on `worktree-agent-a0ca93417a36fb42d`:

1. **Task 1: pyscf-algebra Cargo.toml + foundation modules (lib.rs, client.rs, tensor.rs, error.rs)** — `a8fc7bf` (feat)
2. **Task 2: select.rs + 7 algebra primitives + host_fallback + oracle** — `29aab38` (feat)
3. **Task 3: 4 integration tests + pyscf-rs façade** — `b601040` (test)

## Files Created/Modified

### Created (19)

- `crates/pyscf-algebra/Cargo.toml` — Per-backend feature gating; `default = ["cpu"]`; `gpu = ["cuda", "wgpu"]` umbrella per Roadmap line 33; `metal = ["wgpu"]` alias; cubecl + cubecl-matmul + cubecl-reduce + faer + bytemuck + thiserror + tracing as regular deps; cubecl-{wgpu,cuda,hip} + wgpu as optional; rstest + approx + tracing-subscriber as dev-deps.
- `crates/pyscf-algebra/src/lib.rs` — Module declarations + flat re-exports of all 14 public free functions/types. Crate-level `#![deny(unsafe_op_in_unsafe_fn)]` + `#![warn(clippy::unwrap_used)]` (FOUND-07).
- `crates/pyscf-algebra/src/client.rs` — `AlgebraClient` enum with Cpu unconditional + cfg-gated Cuda/Wgpu/Rocm arms; manual `std::fmt::Debug` impl (cubecl 0.10.0 ComputeClient<R> doesn't impl Debug); `kind()` helper; `log_resolution()` ALG-08 mandatory log line.
- `crates/pyscf-algebra/src/tensor.rs` — `BufferId(pub(crate) u64)` opaque newtype; `Tensor { id, shape, dtype }` with `rank()`/`numel()`/`elem_size()`/`nbytes()` helpers and `placeholder(shape, dtype)` for integration tests (sentinel `BufferId(u64::MAX)`, never dereferenced by Phase 1 stubs).
- `crates/pyscf-algebra/src/error.rs` — `AlgebraError` thiserror enum: `Backend(#[from] BackendError)`, `DimensionMismatch`, `DtypeMismatch`, `NotYetImplemented{phase, what}`, `CubeclRuntime(String)`.
- `crates/pyscf-algebra/src/select.rs` — `select_backend()` + `BackendSelection { client, kind, raw_env, dtype }` + `auto_resolve()` (D-07 priority chain with per-probe `tracing::info!`) + `verify_explicit()` (D-09 hard-error split) + `construct_client()` (per-kind cubecl client construction).
- `crates/pyscf-algebra/src/{gemm,gemv,axpy,scal,transpose,dot,reduce}.rs` — 7 Phase-1-stub primitives. Each takes `&AlgebraClient` + `Tensor` references and returns `Err(AlgebraError::NotYetImplemented { phase: 2, what: "..." })`.
- `crates/pyscf-algebra/src/host_fallback.rs` — 4 Phase-1-stub host_fallback signatures (eigh/cholesky/qr/svd). Each returns `NotYetImplemented` with the phase that wires it (3 for eigh+cholesky, 6 for qr, 7 for svd).
- `crates/pyscf-algebra/src/oracle.rs` — `pub const PAIRWISE_CHUNK = 128`; `oracle_sum(xs)` calls `pairwise(xs, PAIRWISE_CHUNK)`; `oracle_dot(a, b)` materialises elementwise products into a Vec then `oracle_sum`s; `oracle_einsum("ij,jk->ik", ...)` does row-by-col contraction with `oracle_sum` of length-K columns; `pairwise(xs, chunk)` recurses at midpoint when `len > chunk`, base-case strict left-to-right sum otherwise.
- `crates/pyscf-algebra/tests/oracle_determinism.rs` — 5 tests: `oracle_sum_deterministic_within_process`, `oracle_sum_short_and_long`, `oracle_sum_distinguishes_orderings`, `oracle_dot_deterministic`, `oracle_sum_documented_thread_invariance` (10 repeated calls all bit-equal).
- `crates/pyscf-algebra/tests/cubecl_matmul_smoke.rs` — 1 test: `cubecl_matmul_symbol_exists` constructs a CPU client and references `cubecl_matmul::Strategy::Auto` (compile-time gate on Pitfall 1 ABI).
- `crates/pyscf-algebra/tests/backend_matrix.rs` — 2 tests: `select_default_returns_cpu`, `primitive_signatures_callable_returning_notyetimplemented` (gemm/axpy/reduce_sum each return `Err(NotYetImplemented)` — the Phase 1 contract).
- `crates/pyscf-algebra/tests/select_backend.rs` — 7 tests covering Roadmap criterion 6 truth table: `unset_resolves_to_cpu`, `cpu_explicit_resolves_to_cpu`, `bogus_resolves_to_cpu_with_warn`, `auto_on_cpu_only_build_resolves_to_cpu`, `case_insensitive_env_parsing`, `dtype_f32_honored`, `alg08_log_resolution_invoked`.

### Modified (2)

- `crates/pyscf-rs/Cargo.toml` — Added 3 path deps: `pyscf-core`, `pyscf-runtime`, `pyscf-algebra`. No version constraints; workspace declares the version.
- `crates/pyscf-rs/src/lib.rs` — Replaced empty stub with FOUND-01 façade body. Removed `#![forbid(unsafe_code)]`-only boilerplate and added 3 named-module re-exports (`pub use pyscf_core as core; pub use pyscf_runtime as runtime; pub use pyscf_algebra as algebra;`) + 9 convenience flat re-exports of common types.

## Decisions Made

See `key-decisions:` in frontmatter. Highlights:

- **Manual `Debug` impl on AlgebraClient** (cubecl 0.10.0 doesn't impl Debug for ComputeClient<R>) — prints only `kind` name.
- **Phase-1 NotYetImplemented stubs** for all 11 primitives; the public surface is locked, the wiring lands at the first call site in Phase 2+.
- **`select_backend` lives in pyscf-algebra**, not pyscf-runtime (avoids circular dep; consistent with Plan 03's deferred decision).
- **Plan-04-only addition: `Tensor::placeholder` constructor** for integration tests (`BufferId(u64::MAX)` sentinel, never dereferenced by Phase 1 stubs). Phase 2's allocator will replace it.
- **`BackendError` import in select.rs is `#[cfg(feature = "wgpu")]`** — only the wgpu arm constructs it (D-09 hard-error path).
- **`oracle_einsum` Phase 1 = binary `"ij,jk->ik"` only** — RESEARCH §1 explicit boundary; Phase 4 extends.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] `#[derive(Debug)]` on AlgebraClient fails: cubecl 0.10.0 ComputeClient<R> does not implement Debug**
- **Found during:** Task 2 build verification (`cargo build -p pyscf-algebra`).
- **Issue:** The plan's Task 1 client.rs body specified `#[derive(Debug)]` on AlgebraClient. cubecl 0.10.0's `ComputeClient<R>` type does not implement Debug (the inner Server/Channel types are not Debug). Build failed with `E0277: ComputeClient<CpuRuntime> doesn't implement Debug`.
- **Fix:** Replaced `#[derive(Debug)]` with a manual `impl std::fmt::Debug for AlgebraClient` that prints only the backend kind name (`AlgebraClient { kind: "cpu" }`) via `f.debug_struct("AlgebraClient").field("kind", &self.kind().name()).finish()`. Sufficient for tracing diagnostics; printing channel internals would be noisy and non-portable across cubecl backends.
- **Files modified:** `crates/pyscf-algebra/src/client.rs` (within Task 1's scope; rewrite committed as part of Task 2's `29aab38` because Task 1's commit had already landed).
- **Committed in:** `29aab38` (Task 2).

**2. [Rule 2 — Missing critical] `BackendError` import unused on cpu-only builds**
- **Found during:** Task 2 build verification (`cargo build -p pyscf-algebra` default features).
- **Issue:** The plan's Task 2 select.rs imports `BackendError` from pyscf-runtime, but it's only constructed inside the `#[cfg(feature = "wgpu")]` D-09 hard-error path. cpu-only builds saw a `warning: unused import: BackendError`.
- **Fix:** Gated the `use pyscf_runtime::BackendError` line behind `#[cfg(feature = "wgpu")]`. Other imports (`BackendKind`, `DType`) remain unconditional.
- **Files modified:** `crates/pyscf-algebra/src/select.rs`.
- **Committed in:** `29aab38` (Task 2).

**3. [Rule 3 — Blocking] `[patch.crates-io]` cintx git remote fails to update due to upstream submodule references**
- **Found during:** Task 2 first build attempt (`cargo build -p pyscf-algebra`).
- **Issue:** Cargo refused to load any package: `failed to update submodule .claude/worktrees/agent-a01e6318: no URL configured for submodule`. This is a pre-existing upstream issue in cintx (its `.gitmodules` references stale worktree paths). Plan 03's SUMMARY documented the same workaround.
- **Fix:** Used the **transient, in-worktree-only** workaround Plan 03 established:
  1. Snapshotted root Cargo.toml to `/tmp/wt-cargo-snapshot.toml` (`sha256 = dab1db7d…cee92960`).
  2. Commented out the `[patch.crates-io]` block in the worktree's root Cargo.toml.
  3. Ran `cargo build -p pyscf-algebra` (default features) → `Finished`.
  4. Ran `cargo build -p pyscf-algebra --features wgpu` → `Finished` (35.7s — full wgpu/wgpu-hal/wgpu-core transitive deps build).
  5. Ran `cargo build -p pyscf-algebra --features gpu` → `Finished` (cuda+wgpu both compile).
  6. Ran `cargo build --workspace` → `Finished` (all 15 members compile — Roadmap success criterion 1).
  7. Ran the 4 integration tests (15 tests total — 5 oracle_determinism + 1 cubecl_matmul_smoke + 2 backend_matrix + 7 select_backend), all passing.
  8. Ran `cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism` → 5/5 pass (Roadmap criterion 3 oracle-profile).
  9. Ran `RAYON_NUM_THREADS={1,8} cargo test ... --test oracle_determinism` → both 5/5 pass (thread invariance).
  10. Restored root Cargo.toml byte-identically from snapshot. `sha256sum` matched both before and after restore. `git diff Cargo.toml` empty.
  11. Deleted spurious Cargo.lock generated under modified workspace.
- **Files modified outside scope:** None committed. Transient workspace edits and Cargo.lock reverted/deleted before staging any task commit.
- **Verification:** `git diff 17c6c3c -- Cargo.toml` empty both before and after each task commit.
- **Committed in:** N/A (transient workaround, no out-of-scope files committed).

### Auth Gates

None — no external service authentication required. All builds and tests run offline-by-default once the cubecl/wgpu transitive deps are fetched (first build only).

---

**Total deviations:** 3 — 1 bug (Debug derive incompatibility — fixed inline), 1 critical-missing (unused-import gating), 1 blocking (workspace patch break — Plan 03 carryover, same workaround). All resolved without architectural changes (no Rule 4 escalations).

## Issues Encountered

- **Erroneous commit on master via cwd-drift (#3097)** — at the very start of Task 1, the executor's first attempt at `git commit` was executed in a Bash command that began with `cd /home/user/Documents/workspace/pyscf_rs && ...`, putting the cwd inside the **main repo** (master checked out) instead of the worktree. The result: commit `cc36476` landed on master instead of `worktree-agent-a0ca93417a36fb42d`. Per the destructive-git prohibition (#2924) the executor did NOT use `git update-ref` or `git reset --hard` to rewind master. Instead, it cherry-picked `cc36476` onto the worktree branch (creating canonical commit `a8fc7bf` for Task 1) and left the orphan `cc36476` on master. The orphan is functionally identical to `a8fc7bf` (same file content, different parent SHA). **Recommended user action at merge time:** verify the orphan; the orchestrator's `git merge worktree-agent-a0ca93417a36fb42d` may detect the duplicate and either fast-forward-skip or merge cleanly. If the merge is non-trivial, the user may choose to revert `cc36476` on master before merging the worktree. From this point onward, every Bash call avoided `cd` and used absolute worktree paths (`/home/user/Documents/workspace/pyscf_rs/.claude/worktrees/agent-a0ca93417a36fb42d/...`) and `git -C "$WT" ...` to defeat cwd-drift.
- **`pyscf-runtime` dead-code warning** — `WorkspacePool.pool` field is `pub(crate)` and not yet read by any internal code (Phase 6 fills it). Treated as expected per Plan 03 SUMMARY.
- **`pyscf-algebra` dead-code warning** — `BufferId::from_raw` is `pub(crate)` and not yet called by any internal code. Phase 2 (allocator) will use it; the `#[doc(hidden)]` attribute marks it as intentional.

## Threat Surface Notes

Plan's `<threat_model>` lists 4 STRIDE entries; verification:

| Threat | Mitigation Verified |
|--------|---------------------|
| T-1-01 (T: env-var → AlgebraClient) | `select_backend()` returns `Result<BackendSelection, AlgebraError>` with safe fallback (CPU) for unrecognised tokens (verified by `bogus_resolves_to_cpu_with_warn` test). No `unwrap()` or panic on env input. |
| T-1-01 (D: panic via probe) | Probes use `catch_unwind` (Plan 03). `select_backend()` propagates the `Option<bool>` from `probe::wgpu_available(dtype)` etc., never panics. |
| T-1-03 (E: dep wall) | `pyscf-algebra/Cargo.toml` lists cubecl + cubecl-matmul + cubecl-reduce + cubecl-{cpu,wgpu,cuda,hip} — accepted carve-out per ALG-06. Plan 05's `xtask check-dependency-wall` allowlists this crate alongside pyscf-runtime; assertion verified by Plan 05 SUMMARY. |
| T-1-01 (S: cubecl-matmul ABI vs cubecl-runtime) | `tests/cubecl_matmul_smoke.rs::cubecl_matmul_symbol_exists` references `cubecl_matmul::Strategy::Auto` at compile time. If the symbol path drifts in cubecl-matmul 0.10.0, the test fails to compile, surfacing the break before any GEMM call site. **Verified: test compiles and passes against cubecl-matmul 0.9.0-pre.5 + cubecl-runtime 0.10.0.** |

No new threat surface beyond the plan's register.

## Threat Flags

None — no new security-relevant surface beyond the threat model. Crate adds no network endpoints, no auth paths, no file access, and no schema. All env-var inputs flow through the fallible parsers in pyscf-runtime (verified by Plan 03).

## User Setup Required

None — no external service configuration required. (CUDA/wgpu hardware is OPTIONAL for this plan; `--features cuda` compiles without a CUDA SDK at runtime, `--features wgpu` builds the wgpu/wgpu-hal transitive stack but doesn't require a GPU adapter for the integration tests.)

## Next Phase Readiness

- **Phase 2 (GTO integral driver):** Will call `gemm(&client, &lhs, &rhs, &mut out)` from the first integral-contraction kernel. The signature is locked; the body is a `NotYetImplemented` stub that Phase 2 replaces with `cubecl_matmul::launch::<R, T>(&Strategy::Auto, &client, lhs_handle, rhs_handle, out_handle)`. The `cubecl_matmul_smoke` test proves the symbol path is reachable today.
- **Phase 2 (GTO):** Will also call `axpy`, `scal`, `transpose`, `reduce_sum` from the AO contraction kernel. Phase 2 lifts each NotYetImplemented stub into a real `#[cube]` kernel (RESEARCH §8 + docs/manual/Cubecl/Cubecl_multi_compute.md).
- **Phase 3 (SCF):** Will call `eigh(&client, &fock_matrix)` for the Fock-matrix diagonalization. Phase 3 wires `faer::Mat::self_adjoint_eigen(Side::Lower)` with a Vec<f64> round-trip per RESEARCH §9 + Pitfall 3 (faer-ext incompat at faer 0.24).
- **Phase 4 (DFT):** Will extend `oracle_einsum` beyond the binary `"ij,jk->ik"` pattern when grid-quadrature integration introduces multi-tensor contractions.
- **Phase 6 (CCSD):** Will call `qr` for intermediate canonicalization. Phase 6 wires `faer::Mat::qr` (full pivot variant TBD per CCSD-11 spec).
- **Phase 7 (gradient):** Will call `svd` for null-space projection. Phase 7 wires `faer::Mat::svd`.
- **Phase 1 Plan 06 (CI):** Will run `cargo test --workspace --locked` under multiple feature configurations (default, --features wgpu, --features gpu) and under `RAYON_NUM_THREADS={1,8}` for the oracle_determinism smoke. The `cubecl_matmul_smoke` test is the link-time gate that catches cubecl-matmul ABI drift in the nightly cross-crate matrix.
- **Phase 1 Plan 05 (xtask):** Already shipped — its `check-dependency-wall` allowlists `pyscf-runtime` and `pyscf-algebra` for cubecl-* deps; `check-cubecl-pin` enforces the 0.9.0-pre.5/0.10.0 lockstep.

## Self-Check: PASSED

Verified each acceptance criterion individually:

- [x] `crates/pyscf-algebra/Cargo.toml` exists with `gpu = ["cuda", "wgpu"]` (Roadmap line 33), `metal = ["wgpu"]` (D-04 cubecl-metal alias), `default = ["cpu"]` (ALG-03 default)
- [x] `crates/pyscf-algebra/src/client.rs` has `pub enum AlgebraClient`, `fn kind(&self)`, `fn log_resolution`, exact log line `"pyscf-algebra: backend="`
- [x] `crates/pyscf-algebra/src/tensor.rs` has `pub struct Tensor`, `pub struct BufferId`, `pub fn placeholder`
- [x] `crates/pyscf-algebra/src/error.rs` has `pub enum AlgebraError`
- [x] `crates/pyscf-algebra/src/select.rs` has `pub fn select_backend`, `auto_resolve`, `verify_explicit`, `BackendError::Unsatisfiable` (D-09), `tracing::info!.*"probe"` (D-07), `tracing::warn!.*PYSCF_BACKEND` (ALG-04)
- [x] All 7 algebra primitive files exist with `pub fn {gemm,gemv,axpy,scal,transpose,dot,reduce_sum}`
- [x] `crates/pyscf-algebra/src/host_fallback.rs` has `pub fn {eigh,cholesky,qr,svd}`
- [x] `crates/pyscf-algebra/src/oracle.rs` has `pub const PAIRWISE_CHUNK: usize = 128`, `pub fn oracle_sum`, `pub fn oracle_dot`, `pub fn oracle_einsum`, `fn pairwise`
- [x] `crates/pyscf-algebra/tests/oracle_determinism.rs` exists (5 tests pass; verified under default and release-oracle profiles AND under RAYON_NUM_THREADS={1,8})
- [x] `crates/pyscf-algebra/tests/cubecl_matmul_smoke.rs` exists (1 test passes — Pitfall 1 ABI gate)
- [x] `crates/pyscf-algebra/tests/backend_matrix.rs` exists (2 tests pass — ALG-07 CPU baseline)
- [x] `crates/pyscf-algebra/tests/select_backend.rs` exists (7 tests pass — ALG-04 + Roadmap criterion 6 truth table)
- [x] `crates/pyscf-rs/Cargo.toml` has 3 path deps (`pyscf-core`, `pyscf-runtime`, `pyscf-algebra`)
- [x] `crates/pyscf-rs/src/lib.rs` has `pub use pyscf_core`, `pub use pyscf_runtime`, `pub use pyscf_algebra` (FOUND-01 façade)
- [x] `cargo build --workspace` exits 0 (Roadmap success criterion 1 — all 15 members compile)
- [x] `cargo build -p pyscf-algebra --features wgpu` exits 0 (wgpu probe arm)
- [x] `cargo build -p pyscf-algebra --features gpu` exits 0 (cuda+wgpu both arms)
- [x] `cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism` exits 0 (5/5 pass — Roadmap criterion 3)
- [x] Commit `a8fc7bf` in worktree git log (Task 1 — Cargo.toml + foundation modules)
- [x] Commit `29aab38` in worktree git log (Task 2 — select_backend + 7 primitives + host_fallback + oracle)
- [x] Commit `b601040` in worktree git log (Task 3 — 4 integration tests + façade)
- [x] All 15 `must_haves.truths` from PLAN frontmatter verified
- [x] No `unwrap()` anywhere in `crates/pyscf-algebra/src/` (FOUND-07 floor)
- [x] Root `Cargo.toml` byte-identical to base `17c6c3c` (`git diff 17c6c3c -- Cargo.toml` empty; `sha256 = dab1db7d…cee92960` matches)
- [x] No files modified outside `crates/pyscf-algebra/` and `crates/pyscf-rs/` (no STATE.md / ROADMAP.md changes — orchestrator owns those)

---
*Phase: 01-foundation*
*Plan: 04*
*Completed: 2026-05-10*
