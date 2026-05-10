---
phase: 01-foundation
plan: 03
subsystem: foundation
tags: [pyscf-runtime, BackendKind, DType, probes, OnceLock, catch_unwind, cubecl, tracing, foundation, FOUND-03, FOUND-09, ALG-04]

# Dependency graph
requires:
  - phase: 01-foundation/01
    provides: workspace Cargo.toml with [workspace.dependencies] (cubecl 0.10.0 pin, tracing/thiserror, etc.) and pyscf-runtime member declared
  - phase: 01-foundation/02
    provides: pyscf-core path-dep target (PyscfRsError surface available; not yet imported by 01-03 — Plan 04 will wire)
provides:
  - "pyscf-runtime crate at crates/pyscf-runtime/ — BackendKind, per-backend probes, WorkspacePool skeleton, init_tracing"
  - "BackendKind enum (cfg-gated arms) + Default → Cpu (FOUND-03) + from_env_str/is_auto_token parsers (D-07)"
  - "DType {F32, F64} + Default → F64 (D-08) + from_env() reader (D-08)"
  - "Per-backend probes: cpu (trivial), cuda/wgpu/hip (OnceLock-cached, panic-safe via catch_unwind, D-10 supports_type gate)"
  - "WorkspacePool skeleton: PYSCF_MAX_MEMORY (MB → bytes) reader, 4 GiB default, try_reserve budget check (Phase 6 fills body)"
  - "init_tracing helper: verbose 0..=9 → LevelFilter (FOUND-09); idempotent try_init for tests (Pitfall 6)"
  - "BackendError enum: Unsatisfiable / MemoryLimitExceeded / FeatureNotEnabled / ProbeFailed (D-09 hard-error variant)"
  - "Integration test crates/pyscf-runtime/tests/select_backend.rs — 7 tests proving CPU-only env-var truth table (ALG-04 subset)"
affects: [01-04-pyscf-algebra, 01-05-xtask, 01-06-CI, 03-scf-bindings, all-method-crates-phase-2-7]

# Tech tracking
tech-stack:
  added:
    - "cubecl 0.10.0 (optional, gated per backend feature)"
    - "cubecl-cpu 0.10.0 (default)"
    - "cubecl-cuda 0.10.0 (optional, --features cuda)"
    - "cubecl-wgpu 0.10.0 (optional, --features wgpu)"
    - "cubecl-hip 0.10.0 (optional, --features rocm)"
    - "wgpu 29.0.3 (optional, --features wgpu — adapter feature inspection)"
    - "tracing-subscriber 0.3.23 (idempotent fmt::try_init pattern)"
  patterns:
    - "OnceLock<Option<Client>> probe-cache pattern (PATTERNS Shared) — negative result caches too"
    - "catch_unwind discipline around every cubecl client construction (FOUND-07 + Pitfall 5)"
    - "Per-backend feature gating with `dep:` prefix (Cargo edition 2024 namespaced features) — prevents callers from `--features pyscf-runtime/cubecl` activating cubecl unconditionally"
    - "metal = ['wgpu'] feature alias (cubecl-metal not on crates.io — cintx-cubecl precedent)"
    - "Library tracing helper that emits events but does NOT install subscriber (RESEARCH §12) + idempotent try_init wrapper for binaries"
    - "DType axis (F32/F64) carried in BackendKind probe API — wgpu probe returns false for F64 when adapter lacks shader-f64 (D-09)"
    - "PYSCF_MAX_MEMORY interpreted as MEGABYTES per upstream PySCF convention with saturating_mul guard against overflow"

key-files:
  created:
    - "crates/pyscf-runtime/Cargo.toml"
    - "crates/pyscf-runtime/src/lib.rs"
    - "crates/pyscf-runtime/src/backend.rs"
    - "crates/pyscf-runtime/src/error.rs"
    - "crates/pyscf-runtime/src/workspace_pool.rs"
    - "crates/pyscf-runtime/src/tracing_init.rs"
    - "crates/pyscf-runtime/src/probe/mod.rs"
    - "crates/pyscf-runtime/src/probe/cpu.rs"
    - "crates/pyscf-runtime/src/probe/cuda.rs"
    - "crates/pyscf-runtime/src/probe/wgpu.rs"
    - "crates/pyscf-runtime/src/probe/hip.rs"
    - "crates/pyscf-runtime/tests/select_backend.rs"
  modified: []

key-decisions:
  - "BackendKind variants outside Cpu are #[cfg(feature = '...')]-gated — guarantees a CPU-only build never references symbols from cubecl-cuda/wgpu/hip (D-04)"
  - "BackendKind::from_env_str returns None for 'auto' — auto resolution is the caller's job (D-07 priority chain) and must be a separate code path from per-backend parsing"
  - "DType::F64 is the default both at struct level (#[derive(Default)] + impl Default) AND in from_env() fallback — chemistry energies need it; explicit F32 must be opt-in per D-08"
  - "wgpu probe API takes DType (not just availability bool) so caller can distinguish 'wgpu has no GPU' from 'wgpu GPU lacks shader-f64' — Plan 04's select_backend will map F64+missing-shader-f64 to BackendError::Unsatisfiable in explicit mode and tracing::info!+skip in auto mode (D-09 split rule)"
  - "WorkspacePool ships as a struct skeleton (3 fields, 3 methods) NOT a trait — Phase 6 (CCSD-11) extends the body without breaking the public API; method crates can construct WorkspacePool::from_env() today and get a budget check for free"
  - "init_tracing returns bool indicating whether subscriber was installed by THIS call — lets test harnesses safely call it from N tests without spurious 'subscriber already installed' panics (Pitfall 6)"
  - "select_backend() does NOT live in pyscf-runtime — it's deferred to Plan 04 (pyscf-algebra) because it returns AlgebraClient, which lives there. Splitting building blocks (here) from dispatch (algebra) prevents the circular dep that PATTERNS.md sketches"

patterns-established:
  - "Per-backend feature gating in Cargo.toml uses dep: prefix to keep cubecl optional even with default = ['cpu']; metal aliases wgpu (cubecl-metal not on crates.io)"
  - "Probe modules are flat: each is a single file with one OnceLock<Option<Client>>, one init_X() helper wrapping construction in catch_unwind, one X_available(dtype) public function"
  - "Integration test convention: setup_tracing() helper at top calls fmt::try_init() (idempotent); each test calls it first; --test-threads=1 for env-var safety"
  - "Library never installs tracing subscriber — only tracing::info!/warn! macros. init_tracing() helper exists for binary callers and tests but library code does NOT call it"

requirements-completed: [FOUND-03, FOUND-09, ALG-04]
# Note: ALG-08 (per-PyO3-entry-point logging line) is partially set up here (BackendKind::name + DType::name)
# but the actual tracing::info! site lives at the AlgebraClient construction in Plan 04.
# Marking ALG-04 complete because the BackendKind parser building blocks (PYSCF_BACKEND
# {unset/cpu/CPU/CpU/bogus/empty/xxx/auto} → BackendKind::Cpu fallback chain) are proven
# in tests/select_backend.rs; Plan 04 wires the remaining {cuda,wgpu,rocm,metal} cases that
# need a live AlgebraClient.

# Metrics
duration: 12min
completed: 2026-05-10
---

# Phase 01 Plan 03: pyscf-runtime Foundation Summary

**`pyscf-runtime` crate shipped — `BackendKind` enum + 4 panic-safe per-backend probes (cpu/cuda/wgpu/hip with `OnceLock` caching and `catch_unwind` discipline) + `WorkspacePool` skeleton reading `PYSCF_MAX_MEMORY` (MB→bytes, 4 GiB default) + `init_tracing` helper mapping `verbose 0..=9` to `LevelFilter` + 7 passing integration tests proving the ALG-04 CPU-only env-var truth table.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-10T01:53:11Z
- **Completed:** 2026-05-10T02:05:30Z
- **Tasks:** 3
- **Files created:** 12 (Cargo.toml + 10 source files + 1 integration test)
- **Files modified:** 0 (root `Cargo.toml` byte-identical at end — transient workspace edits reverted)

## Accomplishments

- Locked the **runtime building-block surface** that Plan 04 (`pyscf-algebra`) will wire into `select_backend()`: `BackendKind` enum with cfg-gated arms, `DType` axis, four per-backend probes, `BackendError` thiserror enum, and a Phase-6-extensible `WorkspacePool` skeleton.
- **Adopted the OnceLock probe-cache pattern verbatim** from xcfun-gpu (PATTERNS Shared) for cuda/wgpu/hip — each probe wraps the cubecl client construction in `std::panic::catch_unwind` so a missing CUDA driver / Vulkan loader / ROCm runtime returns `false` instead of panicking through the FFI boundary (Pitfall 5 + FOUND-07).
- **Implemented the D-09 wgpu+f64 split rule's runtime gate**: `wgpu_available(DType::F64)` consults `client.properties().supports_type(ElemType::Float(FloatKind::F64))`. Plan 04's `select_backend()` will map this `bool` to either a `BackendError::Unsatisfiable` (explicit mode) or `tracing::info!`+skip (auto mode).
- **Verified `cargo check`/`build -p pyscf-runtime`** succeeds offline against the cubecl 0.10.0 pin in **both** `default = ["cpu"]` and `--features wgpu` configurations. Default build pulls cubecl-cpu + cubecl 0.10.0 transitive deps; wgpu build additionally pulls cubecl-wgpu + wgpu 29.0.3 + 50+ wgpu-hal deps.
- **Integration test** `tests/select_backend.rs` ships 7 tests, all passing with `--test-threads=1`, proving:
  - `BackendKind::default() == Cpu` (FOUND-03)
  - `from_env_str("cpu"|"CPU"|"CpU") == Some(Cpu)` (case-insensitive D-07)
  - `from_env_str("bogus"|""|"xxx"|"auto") == None` (ALG-04 fallback contract; "auto" is a separate sentinel)
  - `is_auto_token("auto"|"AUTO"|"Auto") == true` (D-07 sentinel)
  - `DType::default() == F64` (D-08)
  - `DType::from_env()` defaults to F64 when `PYSCF_DTYPE` unset
  - `BackendKind::Cpu.name() == "cpu"` (ALG-08 prep — stable display name)

## Task Commits

Each task was committed atomically on `worktree-agent-aefe13608bf17f364`:

1. **Task 1: pyscf-runtime/Cargo.toml** — `0ea3ca5` (feat)
2. **Task 2: 10 source modules (lib/backend/error/probe×5/workspace_pool/tracing_init)** — `3d9ae4b` (feat)
3. **Task 3: tests/select_backend.rs (7-test CPU-only truth table)** — `66c3839` (test)

**Note:** The plan's `<output>` section asks the executor to commit the SUMMARY.md too — that commit follows this section, see `<final_commit>` chain.

## Files Created/Modified

### Created (12)

- `crates/pyscf-runtime/Cargo.toml` — Per-backend feature gating with `dep:` prefix (`cpu`/`cuda`/`wgpu`/`rocm` each pull `cubecl` + the matching `cubecl-{cpu,cuda,wgpu,hip}`; `metal = ["wgpu"]` alias). `pyscf-core` path-dep, `tracing`/`tracing-subscriber`/`thiserror` from workspace, `cubecl-*` and `wgpu` optional. `rstest` dev-dep.
- `crates/pyscf-runtime/src/lib.rs` — Re-export hub with crate-level `#![deny(unsafe_op_in_unsafe_fn)]` + `#![warn(clippy::unwrap_used)]`. Five `pub mod` lines + four `pub use` re-exports (`BackendKind`, `DType`, `BackendError`, `init_tracing`, `WorkspacePool`).
- `crates/pyscf-runtime/src/backend.rs` — `BackendKind` enum with 1 unconditional arm (`Cpu`) + 4 cfg-gated arms; `Default → Cpu`; `name()` for ALG-08 logs; `from_env_str` (case-insensitive, returns None for "auto" and unrecognised); `is_auto_token`. `DType` enum {F32, F64}; `Default → F64`; `from_env()` reading `PYSCF_DTYPE`.
- `crates/pyscf-runtime/src/error.rs` — `BackendError` thiserror enum with 4 variants: `Unsatisfiable {backend, dtype, reason}` (D-09 hard-error path), `MemoryLimitExceeded {requested, limit}` (WorkspacePool path), `FeatureNotEnabled(name)`, `ProbeFailed {backend, reason}`.
- `crates/pyscf-runtime/src/probe/mod.rs` — Per-backend probe module declarations; `cpu` is unconditional, others are cfg-gated.
- `crates/pyscf-runtime/src/probe/cpu.rs` — `cpu_available(_dtype) -> true`. CPU is the FOUND-03 default-on backend.
- `crates/pyscf-runtime/src/probe/cuda.rs` — `static CUDA_CLIENT: OnceLock<Option<CudaClient>>` + `init_cuda` that wraps `CudaRuntime::client(&CudaDevice::default())` in `catch_unwind` and gates on `client.properties().supports_type(ElemType::Float(FloatKind::F64))`. `cuda_available(dtype)` returns `client_opt.is_some()`.
- `crates/pyscf-runtime/src/probe/wgpu.rs` — `static WGPU_CLIENT: OnceLock<Option<WgpuClient>>` + `init_wgpu` that wraps `WgpuRuntime::client(&WgpuDevice::default())` in `catch_unwind`. `wgpu_available(dtype)` matches on dtype: F32 returns `true` if adapter exists; F64 returns the `supports_type(ElemType::Float(FloatKind::F64))` result (D-09 + D-10 shader-f64 gate).
- `crates/pyscf-runtime/src/probe/hip.rs` — `static HIP_CLIENT: OnceLock<Option<HipClient>>` + `init_hip` mirroring CUDA: `catch_unwind` around `HipRuntime::client(&AmdDevice::default())` + `supports_type(F64)` gate. `rocm_available(dtype)` is the public surface (named per D-07: "rocm" is canonical, "hip" is alias).
- `crates/pyscf-runtime/src/workspace_pool.rs` — `pub struct WorkspacePool { budget_bytes: usize, pool: Mutex<Vec<PooledAllocation>> }` (Phase 1 skeleton; Phase 6 fills the inner `PooledAllocation` properly). `pub const DEFAULT_BUDGET_BYTES = 4 GiB`. `new(budget_bytes)`, `from_env()` reading `PYSCF_MAX_MEMORY` (interpreted as MEGABYTES per upstream PySCF convention) with `saturating_mul(1 MiB)` guard, `try_reserve(bytes)` returning `Err(MemoryLimitExceeded)` if `bytes > budget_bytes`.
- `crates/pyscf-runtime/src/tracing_init.rs` — `pub fn verbose_to_filter(verbose: u8) -> LevelFilter` mapping `0 → Off, 1..=2 → Error, 3..=4 → Warn, 5..=6 → Info, 7 → Debug, 8..=9+ → Trace`. `pub fn init_tracing(verbose) -> bool` calls `tracing_subscriber::registry().with(fmt::layer()).with(filter).try_init()` and returns `is_ok()` so callers know whether THIS call installed the subscriber (Pitfall 6 idempotency).
- `crates/pyscf-runtime/tests/select_backend.rs` — 7 integration tests (see Accomplishments).

### Modified
None — the root `Cargo.toml` was modified transiently for in-isolation build verification (see Deviations) and restored byte-identically (`sha256sum` verified, `git diff` empty) before any commit.

## Decisions Made

- **`select_backend()` lives in pyscf-algebra (Plan 04), NOT here.** PATTERNS.md and RESEARCH.md sketch `select_backend()` returning an `AlgebraClient`. Putting that in `pyscf-runtime` would require importing `pyscf_algebra::AlgebraClient`, but `pyscf-algebra/Cargo.toml` already declares `pyscf-runtime` as a dep — circular. Resolution: pyscf-runtime owns the building blocks (`BackendKind`, probes, `BackendError`); pyscf-algebra owns the dispatcher that wires them. Documented in `lib.rs` doc-comment so future readers don't recreate the cycle.
- **`metal = ["wgpu"]` feature alias** rather than introducing a separate `cubecl-metal` dep. cubecl-metal does not exist on crates.io; the cintx-cubecl precedent runs Metal through cubecl-wgpu's Vulkan-portability layer. Adopted verbatim for sibling-crate fidelity.
- **`dep:cubecl` prefix on every backend feature** rather than `cubecl/cpu` / `cubecl/cuda`. The `dep:` prefix (Rust edition 2024 namespaced features) prevents callers from accidentally activating an *implicit* `cubecl` feature on this crate via `--features pyscf-runtime/cubecl`. Only the named per-backend features (`cpu`/`cuda`/`wgpu`/`rocm`/`metal`) are public knobs.
- **WorkspacePool::from_env reads MEGABYTES, not bytes.** `PYSCF_MAX_MEMORY` is the upstream PySCF env var; PySCF itself interprets it as megabytes. Matching that convention keeps the user-facing contract identical even though the Rust internals are byte-counted. Multiply by 1 MiB with `saturating_mul` to defeat overflow on `PYSCF_MAX_MEMORY=99999999999`.
- **Test setup_tracing helper installs a subscriber via `try_init`.** RESEARCH Pitfall 6 explicitly notes that any second `try_init` would error if it weren't `try_init`; using `let _ = ...try_init()` discards the bool and lets seven independent test functions all "set up tracing" without panic-on-redundant-install.
- **Integration test runs with `--test-threads=1`.** Three tests touch `std::env`; serializing them is the safest contract. nextest would also work but that adds a tooling dependency this crate doesn't otherwise need.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Workspace manifest references members that don't exist yet (pyscf-algebra, xtask)**
- **Found during:** Task 2 verification (`cargo check -p pyscf-runtime`)
- **Issue:** Plan 01-01 listed all 15 pyscf-* members + xtask in the workspace `Cargo.toml`. Cargo refuses to load *any* package while the workspace manifest names members whose `Cargo.toml` doesn't exist. Plan 01-04 (pyscf-algebra) and Plan 01-05 (xtask) own those crates and run in parallel waves. The orchestrator's executor prompt explicitly anticipated and authorised this workaround.
- **Fix:** Used a **transient, in-worktree-only** workaround:
  1. Snapshotted root `Cargo.toml` to `/tmp/Cargo.toml.snapshot` (`sha256 = dab1db7d…cee92960`).
  2. Commented out `"crates/pyscf-algebra"` and `"xtask"` lines in `members = [...]`.
  3. Commented out the `[patch.crates-io]` block (the cintx git remote contains a stale submodule reference that breaks `cargo update`; bypassing the patch lets resolution complete using crates.io defaults — none of pyscf-runtime's runtime deps need patching).
  4. Ran `cargo check -p pyscf-runtime --offline` → `Finished` (default features = cpu).
  5. Ran `cargo check -p pyscf-runtime --features wgpu --offline` → `Finished`.
  6. Ran `cargo build -p pyscf-runtime --offline` → `Finished` (full build, not just check).
  7. Ran `cargo build -p pyscf-runtime --features wgpu --offline` → `Finished`.
  8. Ran `cargo test -p pyscf-runtime --lib --offline` → `0 passed; 0 failed` (Task 2 verifies the lib harness).
  9. Ran `cargo test -p pyscf-runtime --test select_backend --offline -- --test-threads=1` → `7 passed; 0 failed` (Task 3 verifies the integration test).
  10. Restored root `Cargo.toml` byte-identically from snapshot. `sha256sum` matched both before and after the restore. `git diff Cargo.toml` empty.
  11. Deleted the spurious `Cargo.lock` generated under the modified workspace (it would record an incomplete dependency tree).
- **Files modified outside scope:** None committed. The transient workspace edits and `Cargo.lock` were reverted/deleted before staging any task commit.
- **Verification:** `git diff faeda90 -- Cargo.toml` → empty, both before and after each task commit. `git status --short` after each task showed only the task's own files.
- **Committed in:** N/A (transient workaround, no out-of-scope files committed).

**2. [Rule 1 — Bug] Erroneous commit landed on master from a Bash `cd` cwd-drift incident (#3097)**
- **Found during:** Task 1 (immediately after the first commit)
- **Issue:** The agent's first attempt at writing `crates/pyscf-runtime/Cargo.toml` invoked `cd /home/user/Documents/workspace/pyscf_rs` (the **main repo** path) inside a Bash command for inspection. Subsequent Write/Edit/Bash calls inherited that path (because Claude Code's bash sessions reset cwd between calls but Write tool's absolute paths bypass cwd entirely). The result: the file was created on the main repo filesystem (master branch checked out), not on the worktree. The first `git commit` ran from the main repo and produced `4916a9d` on master — exactly the protected-branch contamination #3097 warns about.
- **Detection:** The `git status` output after the commit showed `branch: master` instead of the expected `worktree-agent-aefe13608bf17f364`, with the commit on master beyond the wave base `faeda90`.
- **Fix:** Per the destructive-git prohibition, did NOT use `git reset --hard` to rewind master. Instead, used **`git revert --no-edit 4916a9d`** which created `5c4949c` on master — a forward-moving commit that undoes the file addition. Master is now functionally back to `faeda90` (file removed) without any history rewrite. Both `4916a9d` (the goof) and `5c4949c` (the recovery) remain visible in `git log master`, providing an honest paper trail.
- **Then:** Re-created the directory structure inside the proper worktree (`mkdir -p crates/pyscf-runtime/src/probe crates/pyscf-runtime/tests`), re-Wrote the Cargo.toml using the worktree's absolute path, and committed `0ea3ca5` on the worktree branch — the canonical Task 1 commit. From this point onward, every Bash command was anchored with `WT=/home/user/Documents/workspace/pyscf_rs/.claude/worktrees/agent-aefe13608bf17f364; cd "$WT"` as the first line, and every `git commit` was preceded by both the cwd-drift assertion (#3097) and the HEAD-on-worktree-agent-* assertion (#2924) from the per-task commit protocol.
- **Files modified on master:** Two commits exist on master that should not — `4916a9d` (added Cargo.toml) and `5c4949c` (reverted it). Neither contains harm (the file was deleted by the revert), but they pollute master's history. **User attention recommended** at merge time: consider squashing both into oblivion via interactive rebase before merging the worktree branch, or accepting them as a forensic trail of the cwd-drift incident.
- **Verification:** `git log master --oneline -3` shows the revert pair on master; `git log worktree-agent-aefe13608bf17f364 --oneline -5` shows the canonical 3-task progression on the worktree branch with master's two extra commits NOT in the worktree branch's ancestry.
- **Committed in:** N/A — the canonical Task 1 commit `0ea3ca5` is on the worktree branch; the master pollution is recorded here for the orchestrator to surface to the user.

**3. [Rule 2 — Missing critical] tracing-subscriber needed in [dependencies] for tests/select_backend.rs**
- **Found during:** Task 3 (planning the test file)
- **Issue:** The plan's Task 3 test calls `tracing_subscriber::fmt::try_init()` but the plan's Task 1 Cargo.toml only lists `rstest` under `[dev-dependencies]`. If `tracing-subscriber` were ONLY a dev-dep (instead of a regular dep needed by `init_tracing`), the test would fail to compile.
- **Fix:** Verified that `tracing-subscriber` IS already in `[dependencies]` (because `lib.rs::tracing_init::init_tracing` needs it at library-build time). Regular deps are transitively available to integration tests, so no Cargo.toml change was needed. The test compiles without modification.
- **Files modified:** None.
- **Verification:** `cargo test -p pyscf-runtime --test select_backend --offline -- --test-threads=1` → `7 passed; 0 failed` on the first compile attempt.
- **Committed in:** N/A (no fix needed — investigation only).

---

**Total deviations:** 3 — 1 blocking (workspace member missing, anticipated and authorised), 1 bug (cwd-drift to master + clean revert), 1 investigated-and-no-fix (tracing-subscriber availability).
**Impact on plan:** Task 1's content delivered exactly as specified (Cargo.toml byte-for-byte matches the plan); Task 2's 10 source files committed verbatim; Task 3's 7 integration tests pass. The deliverable is unchanged. The cwd-drift incident left two commits on master that the user should review at merge time.

## Issues Encountered

- **`[patch.crates-io]` upstream submodule break** (carried over from Plan 01-02): cargo's resolution of `https://github.com/BectorVoom/cintx.git` fails because that repo's submodule references a stale path (`.claude/worktrees/agent-a01e6318`). Bypassed for in-isolation verification by commenting out the patch block transiently. Out of scope for Plan 01-03 — the cintx repo's submodule hygiene is an upstream issue. Plan 01-06 (CI) will surface this in the nightly cross-crate matrix.
- **Spurious `Cargo.lock` generated** during in-isolation build (records incomplete dep tree without pyscf-algebra/xtask). Deleted before each commit; the canonical workspace-wide `Cargo.lock` will land via Plan 01-06 once all members exist.
- **`#[derive(Default)]` warning on `WorkspacePool.pool: Mutex<Vec<…>>`**: clippy emits `field 'pool' is never read` because the field is `pub(crate)` and Phase 1 doesn't yet read it (Phase 6 fills the body). Treating as expected: the warning is documented in the source comment ("Phase 6 turns this into BufferId per-backend"); cargo build still succeeds. Plan 01-05's lints will likely flag this — Phase 6 will resolve it by actually using the pool.

## Threat Surface Notes

Plan's `<threat_model>` lists three STRIDE entries:

| Threat | Mitigation Verified |
|--------|---------------------|
| T-1-01 (T: env-var → enum) | `BackendKind::from_env_str` returns `Option<Self>` (no unwrap); `DType::from_env` returns `Self` with F64 fallback; `WorkspacePool::from_env` returns `Self` with 4 GiB fallback. **Verified by acceptance criterion `! grep -E '\bunwrap\(\)' crates/pyscf-runtime/src/...` — passes (no `.unwrap()` anywhere in source).** |
| T-1-01 (D: panic via probe) | Every probe (cuda/wgpu/hip) wraps `XRuntime::client(&XDevice::default())` in `std::panic::catch_unwind` and returns `None` on `Err`. **Verified by `grep -q catch_unwind crates/pyscf-runtime/src/probe/{cuda,wgpu,hip}.rs` — all three match.** |
| T-1-03 (E: dep wall) | `pyscf-runtime/Cargo.toml` lists `cubecl`, `cubecl-cpu`, `cubecl-cuda`, `cubecl-wgpu`, `cubecl-hip`, `wgpu` — accept disposition per ALG-06. Plan 05's `xtask check-dependency-wall` will allowlist `pyscf-runtime` + `pyscf-algebra` and assert no other crate names a `cubecl-*`. |

No new threat surface introduced beyond the plan's register.

## Threat Flags

None — no new security-relevant surface beyond what the threat model anticipates. The crate adds no network endpoints, no auth paths, no file access, and no schema. All env-var inputs flow through fallible parsers with safe defaults.

## User Setup Required

None — no external service configuration required. (CUDA/ROCm/wgpu hardware is OPTIONAL for this plan; the probes return `false` cleanly when hardware/drivers are absent. Phase 8 will exercise GPU hardware for the regression suite.)

## Next Phase Readiness

- **Plan 01-04 (pyscf-algebra):** Will `use pyscf_runtime::{BackendKind, DType, BackendError, WorkspacePool, init_tracing, probe::*}` for `select_backend()` implementation. The DType-aware probe API (`probe::wgpu::wgpu_available(dtype)`) is the load-bearing surface for the D-09 split rule (explicit-mode hard error vs auto-mode skip). All four probes return `bool`; the dispatcher in Plan 04 maps `bool` → `AlgebraClient` enum arm.
- **Plan 01-05 (xtask):** Will allowlist `pyscf-runtime` (alongside `pyscf-algebra`) in the dependency-wall check (`xtask/src/bin/check_dependency_wall.rs`). The denylist `cubecl`/`cubecl-cpu`/`cubecl-cuda`/`cubecl-wgpu`/`cubecl-hip`/`wgpu` SHOULD detect this crate's listed deps but the allowlist exemption keeps it green per ALG-06.
- **Plan 01-06 (CI):** Once all Wave-2 plans land, full-workspace `cargo build --locked --features wgpu` will exercise this crate's wgpu probe arm in CI; the `cargo test --workspace --locked --no-fail-fast` job will run the 7 integration tests under both default-feature and `--features cuda` (compile-only) configurations.
- **Phase 3 (BIND-02):** Will call `init_tracing(mol.verbose)` from the PyO3 entry point — the helper is shaped exactly for that use (returns `bool` so the binding can detect "subscriber already installed by an outer harness" and skip).
- **Phase 6 (CCSD-11):** Fills the `WorkspacePool::pool` and `PooledAllocation` bodies; the public surface (`new`, `from_env`, `try_reserve`) is locked today so no method-crate API changes are needed.

## Self-Check: PASSED

Verified each acceptance criterion individually:

- [x] `crates/pyscf-runtime/Cargo.toml` exists (FOUND: Cargo.toml)
- [x] `crates/pyscf-runtime/src/lib.rs` exists
- [x] `crates/pyscf-runtime/src/backend.rs` exists with `pub enum BackendKind`, `fn default() -> Self`, `pub enum DType`, `fn from_env_str`
- [x] `crates/pyscf-runtime/src/error.rs` exists with `Unsatisfiable` + `MemoryLimitExceeded` variants
- [x] `crates/pyscf-runtime/src/probe/mod.rs` exists
- [x] `crates/pyscf-runtime/src/probe/cpu.rs` exists with `pub fn cpu_available`
- [x] `crates/pyscf-runtime/src/probe/cuda.rs` exists with `catch_unwind` + `OnceLock`
- [x] `crates/pyscf-runtime/src/probe/wgpu.rs` exists with `catch_unwind`, `OnceLock`, `FloatKind::F64`
- [x] `crates/pyscf-runtime/src/probe/hip.rs` exists with `catch_unwind` + `OnceLock`
- [x] `crates/pyscf-runtime/src/workspace_pool.rs` exists with `pub struct WorkspacePool`, `PYSCF_MAX_MEMORY`, `pub const DEFAULT_BUDGET_BYTES`, `fn try_reserve`
- [x] `crates/pyscf-runtime/src/tracing_init.rs` exists with `fn verbose_to_filter` + `pub fn init_tracing`
- [x] `crates/pyscf-runtime/tests/select_backend.rs` exists with `tracing_subscriber::fmt::try_init`, `BackendKind::default()`, `from_env_str("bogus")`, `is_auto_token`, `DType::default()` references
- [x] Commit `0ea3ca5` in worktree git log (Task 1 — Cargo.toml)
- [x] Commit `3d9ae4b` in worktree git log (Task 2 — 10 source files)
- [x] Commit `66c3839` in worktree git log (Task 3 — integration test)
- [x] All 9 `must_haves.truths` verified
- [x] No `unwrap()` anywhere in `crates/pyscf-runtime/src/` (FOUND-07 numerical-modules floor)
- [x] Root `Cargo.toml` byte-identical to base `faeda90` (`git diff faeda90 -- Cargo.toml` empty)
- [x] No files modified outside `crates/pyscf-runtime/` (no STATE.md / ROADMAP.md changes — orchestrator owns those)

---
*Phase: 01-foundation*
*Plan: 03*
*Completed: 2026-05-10*
