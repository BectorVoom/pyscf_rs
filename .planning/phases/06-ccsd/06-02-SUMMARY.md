---
phase: 06-ccsd
plan: 02
subsystem: pyscf-runtime / pyscf-core
tags: [ccsd, tensor-arena, workspace-pool, memory-budget, hdf5-spill, amplitudes, ccsd-11]
requires:
  - pyscf-runtime::WorkspacePool (Phase-1 skeleton — budget_bytes/try_reserve/from_env surface KEPT)
  - pyscf-runtime::BackendError::MemoryLimitExceeded (the D-01 refusal seam, already wired)
  - pyscf-chkfile (sole hdf5-metno owner — the re-exported `hdf5` alias for the spill backend, D-07)
  - pyscf-core::Amplitudes (Phase-1 placeholder skeleton — the D-01 upgrade target)
provides:
  - "pyscf-runtime: a real reuse-pool WorkspacePool — reserve(shape,allow_spill)->BufferId / release(id) free-list reuse"
  - "TensorBackend { InMemory(Box<[f64]>) | Spilled(SpillHandle) } — backend-swap behind an opaque handle (D-01/D-08)"
  - "SpillHandle — HDF5 temp-file spill via pyscf_chkfile::hdf5, RAII drop-deletes the temp file (D-07, T-06-02-LEAK)"
  - "HARD PYSCF_MAX_MEMORY pre-flight refusal — over-budget in-core reserve returns MemoryLimitExceeded, no silent downgrade (D-01, T-06-02-OOM)"
  - "as_slice/write_slice/with_mut_slice — the 06-03+ kernel working-store accessors"
  - "pyscf-core::Amplitudes holds opaque AmplitudeStore { Owned(Vec<f64>) | Pooled(PooledRef) } handles (D-01)"
  - "Amplitudes::from_vec / from_pooled constructors + t1_slice/t2_slice accessors (Vec-compat for MP2)"
  - "crates/pyscf-ccsd/tests/heap_alloc_count.rs — A3 isolated counting #[global_allocator] allocate-once-reuse proof"
  - "crates/pyscf-ccsd/tests/refusal.rs — D-01 MemoryLimitExceeded no-downgrade proof"
affects:
  - 06-03..06-11 (every CCSD wave reserves intermediates from this pool + carries these amplitude handles)
  - pyscf-mp2 (Amplitudes shape change — construction site + RDM readers updated, numeric behavior unchanged)
tech-stack:
  added:
    - "pyscf-runtime now deps pyscf-chkfile + ndarray (HDF5 spill backend) — NO new cubecl/libxc; libxc stays 0"
  patterns:
    - "backend-swap-behind-opaque-handle (D-01: spill is a storage swap, not a math rewrite)"
    - "free-list reuse pool (reserve scans for a fitting released buffer before allocating fresh)"
    - "RAII drop-delete temp file (mirror lib.H5TmpFile auto-delete, D-07)"
    - "dedicated counting-#[global_allocator] integration test (A3 isolation — not linked to oracle/determinism arms)"
    - "dependency-free opaque handle enum in pyscf-core (avoids the runtime->core cycle + FOUND-02 cubecl pull)"
key-files:
  created:
    - crates/pyscf-ccsd/tests/heap_alloc_count.rs
    - crates/pyscf-ccsd/tests/refusal.rs
  modified:
    - crates/pyscf-runtime/src/workspace_pool.rs
    - crates/pyscf-runtime/src/lib.rs
    - crates/pyscf-runtime/Cargo.toml
    - crates/pyscf-core/src/amplitudes.rs
    - crates/pyscf-mp2/src/mp2.rs
    - crates/pyscf-mp2/src/rdm.rs
    - crates/pyscf-mp2/tests/rmp2_structural.rs
decisions:
  - "A2 single-handle resolution: the pool owns backing storage keyed by a pyscf-runtime BufferId (NOT pyscf_algebra::Tensor's BufferId — its inner field is pub(crate) to algebra AND pyscf-runtime must not depend on pyscf-algebra, wrong direction). The 06-03+ contraction materializes into the reserved buffer then reduces through pyscf-algebra::oracle_sum."
  - "D-01 handle home: AmplitudeStore lives IN pyscf-core, NOT pyscf-runtime — pyscf-runtime already deps pyscf-core (a reverse dep would cycle) and its default `cpu` feature pulls cubecl (would violate pyscf-core FOUND-02). PooledRef carries only { buffer_id: u64, shape } so pyscf-core stays dependency-free; the CCSD call site reads the buffer through the pool API."
  - "Amplitudes upgrade approach (b): enum { Owned(Vec<f64>) | Pooled(PooledRef) }. MP2 (no arena) uses Owned via from_vec; CCSD uses Pooled via from_pooled. t1_slice/t2_slice accessors keep MP2/RDM readers working."
  - "Live in-memory budget: a fresh in-core allocation must fit `live_inmem_bytes + need <= budget`; the single-buffer ceiling is subsumed. Spilled buffers do NOT count against the in-memory budget (they live on disk)."
metrics:
  duration: 30min
  tasks: 2
  files: 9
  completed: 2026-05-25
---

# Phase 6 Plan 02: WorkspacePool Arena Body + Opaque Amplitude Handles (CCSD-11/D-01/D-08) Summary

Filled the `WorkspacePool` arena body (CCSD-11/D-08 — "the defining decision of the phase"): a real reuse-pool (`reserve`/`release` with free-list reuse) behind opaque `BufferId` handles, a `TensorBackend { InMemory | Spilled }` enum (HDF5 spill via the `pyscf-chkfile` re-exported `hdf5` alias, RAII drop-delete), and the HARD `PYSCF_MAX_MEMORY` pre-flight refusal (D-01 — no silent downgrade). Upgraded `pyscf-core::Amplitudes` from raw `Vec<f64>` to opaque `AmplitudeStore` handles (D-01) while keeping the MP2 construction site and RDM readers compiling (numeric behavior unchanged), and landed the two dedicated CCSD-11 test targets: `heap_alloc_count.rs` (its own isolated counting `#[global_allocator]` proving allocate-once-reuse) and `refusal.rs` (proving the no-downgrade `MemoryLimitExceeded`).

## What Was Built

**Task 1 — WorkspacePool arena body (commit `cc8ca27`):**
- `workspace_pool.rs`: `reserve(shape, allow_spill) -> BufferId` first scans the free-list for a released allocation of fitting size and REUSES it (the allocate-once-reuse guarantee — CCSD-11, Pitfall 20); only on a miss does it allocate fresh. `release(id)` returns the buffer to the free-list WITHOUT dropping the `Box<[f64]>`/spill file, so the next fitting `reserve` reuses it.
- `TensorBackend { InMemory(Box<[f64]>) | Spilled(SpillHandle) }`: backend chosen at allocation time, transparent to the caller. `SpillHandle` wraps a `pyscf_chkfile::hdf5` temp file (created under `std::env::temp_dir()`, single flat `f64` dataset `"buf"`) and DELETES the file in `Drop` (RAII — mirrors `lib.H5TmpFile()` auto-delete, D-07; T-06-02-LEAK mitigation).
- HARD refusal (D-01): an over-budget in-core `reserve(allow_spill=false)` returns `MemoryLimitExceeded` BEFORE allocating anything (no buffer id consumed, no downgrade). The phase-1 `try_reserve` ceiling check + `budget_bytes`/`new`/`from_env` public surface are UNCHANGED.
- Accessors for the 06-03+ kernel: `as_slice` (read), `write_slice` (materialize products into the buffer), `with_mut_slice` (in-place resident view); `allocation_count` for diagnostics/tests.
- A2 single-handle resolution: a pool-owned `pyscf-runtime` `BufferId` (NOT `pyscf_algebra::Tensor`'s — wrong dep direction + private field). Documented in the module doc-comment.
- 7 in-crate unit tests (reuse, larger-buffer reuse, in-use non-reuse, refusal, in-mem roundtrip, spill roundtrip+drop-delete, try_reserve ceiling) — all green.
- `Cargo.toml`: added `pyscf-chkfile` + `ndarray` deps (HDF5 spill); `lib.rs`: exported `BufferId`/`SpillHandle`/`TensorBackend`/`WorkspacePool` + refreshed the status doc-comment.

**Task 2 — Amplitudes opaque handles + CCSD-11 test targets (commit `de5d60b`):**
- `amplitudes.rs`: `Amplitudes.t1`/`.t2` are now `AmplitudeStore { Owned(Vec<f64>) | Pooled(PooledRef) }` (D-01) instead of raw `Vec<f64>`. `PooledRef { buffer_id: u64, shape }` is a dependency-free handle (the CCSD call site reads the buffer through the pool). `from_vec` (MP2/owned) + `from_pooled` (CCSD/arena) constructors; `t1_slice`/`t2_slice` accessors; `Debug`/`Default`/`Clone`/`PartialEq` preserved (CcsdResult relies on them).
- `mp2.rs`: the `rmp2_kernel` construction site uses `Amplitudes::from_vec(nocc, nvir, Vec::new(), t2)`.
- `rdm.rs`: `gamma1_intermediates` + `make_rdm2` pull the resident slice via `t2.t2_slice()` (with a clear `ShapeMismatch` if a pooled store is wrongly passed); the `toy_amps` test builder uses `from_vec`.
- `rmp2_structural.rs`: the t2-amplitude assertion reads through `t2.t2_slice()`.
- `tests/heap_alloc_count.rs`: a `CountingAllocator` wrapping `System` (bumps an `AtomicUsize` on `alloc`/`realloc`) installed as `#[global_allocator]` in THIS test binary only (A3 isolation — separate integration crate, never linked to the oracle/determinism arms). Proves a `Wabef`-sized buffer reserved once and reused across N=5 iterations has bounded allocation delta + the pool never grows past one allocation under a 50-cycle reuse loop.
- `tests/refusal.rs`: 4 tests — `try_reserve`/`reserve` over budget return `MemoryLimitExceeded` (naming requested+limit in the message), one-byte-over still refuses, and spill is opt-in only (`allow_spill=true` serves the SAME request the `false` path refused — no auto-downgrade).

## Verification

- `cargo check -p pyscf-runtime -p pyscf-core -p pyscf-mp2 -p pyscf-ccsd --locked` exits 0 (scoped, default features — no libxc compile triggered).
- `cargo test -p pyscf-runtime workspace_pool --locked` — 7/7 unit tests pass.
- `cargo test -p pyscf-mp2 --locked` — all green (19 lib + integration; MP2 numeric behavior unchanged after the Amplitudes shape change).
- `cargo test -p pyscf-ccsd --test heap_alloc_count --test refusal --locked` — 2 + 4 tests pass (allocate-once-reuse + HARD refusal proven).
- `cargo clippy -p pyscf-runtime -p pyscf-core -p pyscf-mp2 -p pyscf-ccsd --locked --tests -- -D warnings` exits 0.
- Dependency wall: `cargo tree -p pyscf-runtime` libxc=0 (cubecl unchanged at 32 — already present via the cpu feature + the existing pyscf-core→cintx-compat chain; the only NEW transitive crate from `pyscf-chkfile` is `hdf5-metno`). `cargo tree -p pyscf-core` libxc=0, cubecl unchanged at 30 (no dep added to pyscf-core).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 4-avoided / architectural-constraint] D-01 handle home moved from pyscf-runtime to pyscf-core**
- **Found during:** Task 2 planning (dependency-graph inspection before editing).
- **Issue:** The plan suggested adding a `pyscf-runtime` dep to `pyscf-core` so `Amplitudes` could hold the runtime `PooledTensor` handle. This is IMPOSSIBLE: `pyscf-runtime` already depends on `pyscf-core` (verified in `pyscf-runtime/Cargo.toml`), so a reverse dep is a CYCLE; and `pyscf-runtime`'s default `cpu` feature pulls `cubecl`, which would violate `pyscf-core`'s FOUND-02 "no compute deps" rule.
- **Fix:** Defined the opaque handle enum (`AmplitudeStore` + `PooledRef { buffer_id: u64, shape }`) IN `pyscf-core` itself — dependency-free. This satisfies D-01 ("opaque Tensor handles" — the field type is no longer raw `Vec<f64>`) and the D-08 producer/consumer split (the pool in pyscf-runtime PRODUCES `BufferId`s; `Amplitudes` CONSUMES the `buffer_id` value). The CCSD call site (pyscf-ccsd, which deps BOTH crates) reads/writes the pool buffer. This is the plan's offered approach (b) ("make the handle an enum") with the home crate adjusted for the dependency wall — no architectural change to the design, just the correct crate placement. NOT a Rule-4 stop: the plan explicitly authorized choosing (a) or (b) and documenting it.
- **Files modified:** `crates/pyscf-core/src/amplitudes.rs`
- **Commit:** `de5d60b`

**2. [Rule 1 - Bug] reserve(allow_spill=true) wrongly refused at the absolute ceiling**
- **Found during:** Task 1 (the `spilled_buffer_roundtrips_and_deletes_on_drop` unit test failed).
- **Issue:** My first cut called `try_reserve(need_bytes)?` (the absolute-ceiling check) BEFORE the spill decision, so a buffer larger than the total budget was refused even when `allow_spill=true` — but spill exists precisely for over-budget buffers.
- **Fix:** Removed the early ceiling call; the in-core path is gated by the live in-memory budget (`live + need <= budget`), and the HARD refusal lives solely in the `!fits_inmem && !allow_spill` branch. Spill bypasses the in-memory budget (the buffer lives on disk).
- **Files modified:** `crates/pyscf-runtime/src/workspace_pool.rs`
- **Commit:** `cc8ca27`

**3. [Rule 1 - Verify-command correction] `cargo tree | grep -ci 'cubecl|libxc' == 0` is unachievable for pyscf-runtime**
- **Found during:** Task 1 verification.
- **Issue:** The plan's verify (`cargo tree -p pyscf-runtime | grep -ci 'cubecl\|libxc' | grep -qx 0`) can NEVER be 0: `pyscf-runtime` already pulls cubecl via its `cpu` default feature AND via the existing `pyscf-core → cintx-compat → cintx-cubecl` chain (baseline cubecl count = 32). The same verify-command inaccuracy 06-01 hit.
- **Fix:** Verified the REAL invariant instead — `libxc == 0` (unchanged) and `cubecl` count UNCHANGED at 32 (no NEW cubecl source; the only new transitive crate from `pyscf-chkfile` is `hdf5-metno`). No code change.
- **Commit:** n/a (verification only)

### Out-of-scope (NOT fixed — deferred)

**Cargo.lock NOT staged (pre-existing dirty-tree drift).** Adding `pyscf-chkfile`+`ndarray` to `pyscf-runtime` requires those two lines in the lock's `pyscf-runtime` dependency array. But the working-tree `Cargo.lock` is heavily dirty with ~100 unrelated `libxc-kernel-*` entries (the libxc_rs path members re-resolve into the lock on any `cargo` lock-touching op; HEAD's lock is already stale at 856 such entries — a known pre-existing issue per project memory "stale Cargo.lock"). A surgical 2-line edit fails `--locked` (cargo recomputes the full resolution and wants the extra libxc entries). Per the project constraint "stage ONLY the files your plan modifies, never unrelated changes", I did NOT stage `Cargo.lock` — the dirty lock-on-disk already satisfies `--locked` builds (verified: `cargo check -p pyscf-runtime --locked` exits 0 against it). The lock-unification belongs to the integration gate / a dedicated `build(...)` commit, mirroring 06-01 which also did not touch the lock. **Action for integration gate:** ensure `pyscf-runtime`'s lock entry gains `ndarray` + `pyscf-chkfile` when the lock is next regenerated cleanly.

## Known Stubs

None. The pool body, the amplitude handles, and both test targets are complete and exercised. The 06-03+ CCSD math (which will `reserve` intermediates and carry `from_pooled` amplitudes) is a future-wave concern, not a stub in this plan's deliverables.

## Threat Flags

No new threat surface beyond the plan's `<threat_model>`. The two `mitigate` dispositions are satisfied:
- **T-06-02-OOM** (DoS via over-budget): HARD `reserve(allow_spill=false)` refusal proven by `tests/refusal.rs` (4 arms incl. one-byte-over + spill-is-opt-in).
- **T-06-02-LEAK** (spill temp-file leak): `SpillHandle::drop` deletes the temp file, proven by the `spilled_buffer_roundtrips_and_deletes_on_drop` unit test (asserts the path is gone after the pool drops).
- **T-06-02-UB**: `#![forbid(unsafe_code)]` holds in the library crates; the ONLY `unsafe` is the sanctioned A3 counting-`GlobalAlloc` shim, isolated to the `heap_alloc_count.rs` test binary (forwards verbatim to `System`).
- **T-06-02-SC** (accept): no registry install; the spill backend reuses the existing `pyscf-chkfile`/`hdf5-metno` owner (vetted Phase 3/4), libxc stays 0.

## Self-Check: PASSED

- Files exist: `crates/pyscf-runtime/src/workspace_pool.rs`, `crates/pyscf-core/src/amplitudes.rs`, `crates/pyscf-ccsd/tests/heap_alloc_count.rs`, `crates/pyscf-ccsd/tests/refusal.rs` — all present on disk.
- Commits exist: `cc8ca27` (Task 1) + `de5d60b` (Task 2) — both found in `git log`.
- `cargo check -p pyscf-runtime -p pyscf-core -p pyscf-mp2 -p pyscf-ccsd --locked` exits 0; both test targets + the MP2 suite green.
