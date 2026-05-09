# Phase 1: Foundation - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning

<domain>
## Phase Boundary

The 15-crate Rust workspace exists, builds clean as a horizontal-layered façade, the `pyscf-algebra` crate exposes a backend-agnostic linear-algebra surface dispatching to cubecl on the active runtime, and every cross-cutting convention that gates downstream numerical correctness is in place and CI-enforced before the first kernel lands.

**In scope:** Workspace skeleton (15 members), `pyscf-{core,runtime,algebra}` non-stub, `BackendKind`, `select_backend()`, `WorkspacePool`, `AlgebraClient` enum + opaque `BufferId`/`Tensor` surface, `[profile.release-oracle]` FMA-free, `oracle_sum`/`oracle_dot`/`oracle_einsum`, four CI lints (unwrap deny, forbidden-paths, catch_unwind, algebra dependency-wall), nightly cross-crate matrix CI against the cubecl 0.10.0 pin, `PYSCF_BACKEND` resolver with probe-and-skip auto policy, `PYSCF_DTYPE` (f32/f64) precision axis.

**Out of scope:** Any actual chemistry kernel, GTO basis loading, integral evaluation, Python `import pyscf` shim (Phase 3 BIND-02 owns that), maturin wheel build (Phase 8), full GPU regression suite (Phase 8).

</domain>

<decisions>
## Implementation Decisions

### Workspace location & repo restructure
- **D-01:** Root coexistence (cintx pattern). Top-level `Cargo.toml` + `crates/` + `xtask/` land at the repo root, sitting alongside the existing upstream `pyscf/` Python tree, `pyproject.toml`, `setup.py`, `pytest.ini`, `examples/`, and `docs/manual/Cubecl/`. Upstream PySCF source is **not** moved or renamed in Phase 1 — it stays at root as oracle reference and future maturin host.
- **D-02:** Workspace member directories live under `crates/` (e.g., `crates/pyscf-core/`, `crates/pyscf-algebra/`). Each crate is `pyscf-{name}` (hyphenated package name); the in-Rust module path is `pyscf_{name}` (underscored). Mirrors cintx (`crates/cintx-{core,ops,...}`) and xcfun_rs (`crates/xcfun-{core,kernels,...}`) exactly.
- **D-03:** No changes to `pyproject.toml`, `setup.py`, `pyscf/`, `examples/`, `pytest.ini` in Phase 1. The Rust workspace and the legacy Python tree must coexist without colliding. `cargo build --workspace` only sees `crates/`; the existing Python tooling continues to operate on `pyscf/`.

### AlgebraClient dispatch shape
- **D-04:** Enum + match dispatch (sibling-crate pattern). `AlgebraClient` is an enum with `#[cfg(feature = "<backend>")]`-gated arms, one per compiled-in backend (`Cpu`, `Cuda`, `Wgpu`, `Rocm`; `Metal` reuses the wgpu runtime per the cintx-cubecl precedent). Free functions like `pyscf_algebra::gemm(&client, a, b, out)` match-dispatch internally. Method crates (`pyscf-{kernels,gto,scf,dft,mp2,ccsd,grad,geomopt}`) stay non-generic — no `<R: Runtime>` parameter ever leaks into their public APIs.
- **D-05:** Opaque `BufferId` + `Tensor` boundary. `pyscf-algebra` owns all device buffers. The public surface returns a `Tensor { id: BufferId, shape: Vec<usize>, dtype: DType }` newtype; method crates hold `Tensor`s and pass them by reference. Algebra primitives reconstruct the underlying `cubecl::TensorHandle<R, T>` inside the matched arm. Method crates **never** import or name a `cubecl::*` type.
- **D-06:** Algebra public surface for Phase 1: `gemm`, `gemv`, `axpy`, `dot`, `reduce_sum`, `transpose`, `scal`, plus the deterministic-ordered `oracle_sum`/`oracle_dot`/`oracle_einsum` from FOUND-06. Eigh/Cholesky/QR/SVD also live behind this surface but route to `faer 0.24` on host (ALG-05) — on a GPU `AlgebraClient`, the implementation copies down → faer → uploads back, with a documented `// host-fallback per ALG-05` comment.

### `auto` backend resolution + `PYSCF_DTYPE` axis
- **D-07:** Auto resolver probes each compiled backend in priority order (`cuda → rocm → metal → wgpu → cpu`) and emits one `tracing::info!` line per probe attempt (success or skip with reason). Picks the first backend that has both feature compiled **and** a usable device. Final `pyscf-algebra: backend={resolved} (env={raw}, dtype={f32|f64})` line is always emitted (ALG-08).
- **D-08:** **New env var introduced in Phase 1: `PYSCF_DTYPE`** ∈ {`f32`, `f64`} (case-insensitive; default `f64`). This is a separate axis from `PYSCF_BACKEND` and gates kernel precision. The `AlgebraClient` carries the resolved dtype as state; alloc / upload / kernel signatures select the matching cubecl `Numeric` parameter at the matched arm. Per-PyO3-entry-point logging (ALG-08) extends the message to include `dtype=`.
- **D-09:** **wgpu + f64 + missing `shader-f64` rule** — split by selection mode:
  - `PYSCF_BACKEND=wgpu` (explicit) + `PYSCF_DTYPE=f64` + adapter lacks `shader-f64` Vulkan extension → **hard error** at resolver time. `select_backend()` returns `Err(BackendError::Unsatisfiable { backend: Wgpu, dtype: F64, reason: "adapter lacks shader-f64" })`. The Python boundary maps this to a `RuntimeError` with the message `PYSCF_BACKEND=wgpu requested with f64, but adapter '<name>' lacks shader-f64. Set PYSCF_DTYPE=f32 or PYSCF_BACKEND=cpu/auto.` Refuses to silently downgrade.
  - `PYSCF_BACKEND=auto` + `PYSCF_DTYPE=f64` + adapter lacks `shader-f64` → wgpu probe **logs `tracing::info!` "probe wgpu — adapter lacks shader-f64; skipping (f64 requested)"** and continues the priority chain (typically falling through to CPU on a CPU-only box).
  - Either mode + `PYSCF_DTYPE=f32` → wgpu adapter check is satisfied without `shader-f64`; backend probes pass.
- **D-10:** Probe implementation: wgpu probe uses `wgpu::Adapter::features()` to test for `Features::SHADER_F64`; cuda probe uses `cubecl_cuda` runtime construction + a trivial dummy kernel launch (or `cudaGetDeviceCount` if exposed); rocm/metal analogous. Probes are bounded — each gets a short timeout (~250 ms) so pathological hardware doesn't hang `cargo run`. Probe outcomes are cached per-process (one resolver invocation per `select_backend()` call site, but the resolver itself is idempotent within a process).
- **D-11:** Phase 4 (DFT) re-asserts the wgpu/shader-f64 rule on a real DFT cycle (DFT-11) but does NOT duplicate the resolver decision — the algebra layer is the single decision point. Phase 4 gets the already-resolved `AlgebraClient` and trusts that wgpu is only present when valid.

### Sibling-crate sourcing for `[patch.crates-io]`
- **D-12:** Pinned git commit SHAs. Workspace `Cargo.toml` carries `[patch.crates-io]` entries pointing each sibling crate to a specific commit on its public GitHub remote.
- **D-13:** GitHub remotes under `BectorVoom`. Concrete URLs:
  - `cintx     = { git = "https://github.com/BectorVoom/cintx.git",     rev = "<sha>" }`
  - `libxc_rs  = { git = "https://github.com/BectorVoom/libxc_rs.git",  rev = "<sha>" }`
  - `xcfun_rs  = { git = "https://github.com/BectorVoom/xcfun_rs.git",  rev = "<sha>" }`
  If any remote does not yet exist, **publishing it is a Phase 1 prerequisite task** before pyscf_rs's `cargo build --workspace` can run in CI. Researcher / planner should verify each remote exists and surface a Phase 1 task to push it if missing.
- **D-14:** Nightly cross-crate matrix CI (ORACLE-05) updates the pinned SHAs via `cargo update -p cintx -p libxc_rs -p xcfun_rs` and rebuilds the full workspace + tests under `--features gpu`. Failures block merge of any sibling-crate cubecl bump until pyscf_rs is fixed in lockstep.
- **D-15:** No path-dep override is shipped in the repo. Developers who want path-dep iteration set their own `~/.cargo/config.toml` `[patch.crates-io]` to `path = "/home/<user>/Documents/workspace/<sibling>"`. Documented in `CONTRIBUTING.md` as the "local sibling-crate development" recipe.

### Claude's Discretion

The following are not user-decided — researcher / planner picks the implementation:

- Specific algorithm for `oracle_sum`/`oracle_dot` (pairwise tree reduction vs Kahan-Babuska vs strict left-to-right). Constraint: must be bit-identical across `RAYON_NUM_THREADS=1` and `RAYON_NUM_THREADS=8` on the same input (Phase 1 success criterion 3). Researcher should compare options against the cubecl-reduce primitives' guarantees and the cintx-cubecl/xcfun-kernels precedent.
- Lint mechanism for `forbidden-paths` (FOUND-08) and algebra-dependency-wall (ALG-06): custom dylint plugin vs xtask grep + `cargo metadata | jq` vs `cargo-deny` custom rules. Planner picks the lowest-friction option that runs in pre-merge CI.
- Stub crate skeleton: the 12 method/façade crates that are NOT non-stub in Phase 1 should compile (so `cargo build --workspace` passes) with the minimum surface needed downstream. Default: empty `lib.rs` with a single `// TODO: implemented in Phase N` comment, `[lib]` declared in `Cargo.toml`. Planner may upgrade selected crates to skeleton trait re-exports if a downstream phase's planner needs them sooner.
- `WorkspacePool` (FOUND-03) shape: it is named in the requirement but its responsibility is open. Planner should specify whether it is a tensor buffer pool, a thread pool wrapper, a `PYSCF_MAX_MEMORY`-budgeted scratchpad arena, or a combination. The CCSD tensor-arena requirement (CCSD-11) doesn't kick in until Phase 6, so Phase 1 only needs a minimal skeleton; planner picks the surface to lock now to avoid Phase 6 retrofit pressure.
- `panic = "abort"` scope: the FOUND-07 requirement is "release builds" — planner decides whether that means `[profile.release]` only, or also extends to `[profile.release-oracle]`. Reasonable default: both, since both are shipped artifacts.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project specs (this repo)
- `.planning/PROJECT.md` — project vision, core value, constraints, key decisions table
- `.planning/REQUIREMENTS.md` — full v1 REQ-IDs; Phase 1 owns FOUND-01..10, ALG-01..08, ORACLE-01/05/09 (21 REQs total)
- `.planning/ROADMAP.md` §"Phase 1: Foundation" — goal, dependencies, success criteria (5 numbered items)
- `.planning/ROADMAP.md` §"Cross-Cutting Concerns Threaded Through Every Phase" — algebra-responsibility wall, backend selection, bit-exact-with-PySCF, panic policy, scope-creep lint, cubecl pin lockstep
- `.planning/ROADMAP.md` §"Pitfall-to-Phase Mapping" — Phase 1 owns Pitfalls 1, 2, 3, 12, 14, 15, 21
- `.planning/STATE.md` §"Blockers/Concerns" — cubecl 0.10.0 lockstep, faer-ext 0.7.1 ↔ faer 0.24.0 verification

### Cubecl runtime / kernel pattern (reference docs in this repo)
- `docs/manual/Cubecl/Cubecl_multi_ compute.md` — runtime/ComputeClient pattern, `#[cube(launch_unchecked)]`, `ArrayArg::from_raw_parts`, `client.create`/`client.empty`/`client.sync`. Authoritative for `pyscf-algebra`'s low-level launch shape.
- `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` — `cubecl_matmul::launch::<R, T>(&Strategy::Auto, &client, lhs, rhs, out)` — authoritative for `pyscf_algebra::gemm` body.
- `docs/manual/Cubecl/cubecl_reduce_sum.md` — `cubecl_reduce::reduce::<R, _>` — authoritative for `pyscf_algebra::reduce_sum` and the basis for `oracle_sum`.
- `docs/manual/Cubecl/cubecl_macro_fanout_manual.md` — `#[cube]` macro fan-out for element-wise kernels (`axpy`, `scal`, `transpose`).
- `docs/manual/Cubecl/Cubecl_vector.md` — vectorisation primitives.
- `docs/manual/Cubecl/Cubecl_shared_memory.md` — shared-memory pattern (relevant for `WorkspacePool` design).

### Sibling-crate precedent (read before implementing analogous surface)
- `~/Documents/workspace/cintx/Cargo.toml` — workspace members layout, `cubecl = "0.10.0"` pin, `[patch.crates-io]` shape, edition 2024, rust 1.92.
- `~/Documents/workspace/cintx/crates/cintx-cubecl/Cargo.toml` — per-backend feature gating (`cpu`/`wgpu`/`cuda`/`rocm`/`metal` aliases), authoritative pattern for `pyscf-algebra/Cargo.toml`. **Note:** `metal = ["wgpu", ...]` because cubecl-metal is not on crates.io — pyscf-algebra adopts the same alias.
- `~/Documents/workspace/cintx/crates/cintx-runtime/` — analog for `pyscf-runtime`'s `BackendKind`, runtime construction, `WorkspacePool` reference (if any).
- `~/Documents/workspace/xcfun_rs/Cargo.toml` — per-phase incremental member-enabling pattern; xcfun-gpu's `Backend` enum and `Batch<'fun, R: cubecl::Runtime>` for kernel-call shape.
- `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/` — closest analog for `pyscf-algebra` (Backend enum, generation-counter buffer pool, auto_backend() priority chain).

### Codebase maps (this repo, .planning/codebase/)
- `.planning/codebase/STACK.md` — describes UPSTREAM PySCF stack (Python+C); reference for what's being replaced, NOT for the new Rust target.
- `.planning/codebase/STRUCTURE.md`, `ARCHITECTURE.md`, `CONVENTIONS.md`, `INTEGRATIONS.md`, `TESTING.md`, `CONCERNS.md` — same: upstream PySCF reference only.

### External (Phase 1 will need to look up)
- cubecl 0.10.0 docs / `cubecl-runtime`, `cubecl-matmul`, `cubecl-reduce`, `cubecl-cpu`, `cubecl-cuda`, `cubecl-wgpu`, `cubecl-hip` (rocm) crate docs.
- faer 0.24 docs (`eigh`, `cholesky`, `qr`, `svd` host APIs).
- `wgpu::Features::SHADER_F64` and `wgpu::Adapter::features()` for the auto-resolver wgpu probe (D-10).
- `tracing 0.1` API surface for the `tracing::info!`/`tracing::warn!` lines mandated by ALG-08, FOUND-09.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

The pyscf_rs repo currently contains **only upstream PySCF source** (Python + C extensions) at root. **No Rust code exists yet.** Phase 1 is greenfield.

The closest reusable assets live in sibling repos:
- `~/Documents/workspace/cintx/crates/cintx-cubecl/` — Direct template for `pyscf-algebra/Cargo.toml` per-backend feature shape and the `metal` ↔ `wgpu` alias workaround (`metal = ["wgpu", "cintx-runtime/metal"]`). Copy the `[features]` block structure verbatim, swap names.
- `~/Documents/workspace/cintx/crates/cintx-runtime/` — Likely contains `BackendKind` enum + a runtime construction helper analogous to what `pyscf-runtime::select_backend()` needs.
- `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/` — `Backend` enum + `auto_backend()` priority chain; closest precedent for the D-07 probe-and-skip resolver.
- `~/Documents/workspace/cintx/Cargo.toml` lines around `[workspace] members = [...]` — Direct pattern for the 15-member workspace declaration and the `[workspace.package]` shared-version block.

### Established Patterns
- **Per-backend feature naming**: cintx-cubecl uses `cpu` (default-on), `wgpu`, `cuda`, `rocm`, `metal`. pyscf-algebra adopts the same names.
- **`metal = ["wgpu", ...]` alias**: established in cintx-cubecl because `cubecl-metal` does not exist on crates.io. pyscf-algebra inherits this exact alias to keep the four-crate family consistent.
- **`edition = "2024"`, `rust-version = "1.92"`**: established in `~/Documents/workspace/xcfun_rs/Cargo.toml` `[workspace.package]`. pyscf_rs matches.
- **`crates/{name}/` directory per workspace member**: cintx + xcfun_rs both. Plus a top-level `xtask/` for build helpers.
- **`[patch.crates-io]` for cubecl pin**: cintx pins `cubecl = "0.10.0"` directly in `[dependencies]` (no patch needed because it's a registry crate). pyscf_rs adopts the same — `[patch.crates-io]` is reserved for the **sibling crates** (cintx, libxc_rs, xcfun_rs) per D-12, NOT for cubecl.

### Integration Points
- **Top-level repo root**: workspace `Cargo.toml` lands at `/Cargo.toml` (currently doesn't exist; the existing `pyproject.toml` is unrelated to Cargo).
- **Existing `pyscf/` Python tree**: untouched by Phase 1. Future Phase 3 will introduce `python/pyscf/__init__.py` re-export shim (BIND-02), but Phase 1 does NOT touch the Python side.
- **Existing `docs/manual/Cubecl/` directory**: read-only reference. Phase 1 implementation cites these docs in code comments where the implementation mirrors a documented pattern.
- **`.github/` directory**: already exists in repo root. Phase 1 adds new GitHub Actions workflows (nightly cross-crate matrix CI per ORACLE-05, FMA-free grep per FOUND-05, lint enforcement per FOUND-08 and ALG-06) without disturbing existing workflows.
- **`.gitignore`**: already exists. Phase 1 must add `target/`, `Cargo.lock` (for libraries — but pyscf_rs is a workspace with binaries via `xtask` and ultimately a cdylib via `pyscf-py`, so Cargo.lock is **committed**), `.cargo/config.toml.local` if planner introduces local override convention.

</code_context>

<specifics>
## Specific Ideas

- **Sibling-crate fidelity is a hard preference.** Where cintx and xcfun_rs have already settled a pattern (per-backend feature names, `metal = wgpu` alias, `crates/{name}/` layout, edition 2024, rust 1.92, `cubecl = "0.10.0"` exact pin), pyscf-rs adopts it verbatim. Deviation requires explicit justification in PLAN.md.
- **`PYSCF_DTYPE` (f32/f64) is a real user-facing axis**, not just an internal implementation detail. The user explicitly called out the rule "If user selects wgpu and f64, program stops." Both env vars (`PYSCF_BACKEND` and `PYSCF_DTYPE`) appear in user-facing log lines (D-08) and error messages (D-09).
- **Probe-and-skip with per-attempt logs** (D-07) is the user's vision for the auto resolver — observability over silence. The `tracing::info!` line per probe attempt is part of the contract, not a debug nicety.
- **Method crates must NEVER touch cubecl types directly.** D-04 + D-05 jointly mean a method crate's source file should be greppable for `cubecl` and return zero matches outside of the algebra dependency wall. The dependency-wall lint (ALG-06, D-claude-discretion) enforces this at build time.
- **CONTRIBUTING.md "local sibling-crate development" recipe** (D-15) is a Phase 1 deliverable, not a Phase 8. New contributors reading the repo on day one need the path-dep override instructions.

</specifics>

<deferred>
## Deferred Ideas

- **`python/pyscf/__init__.py` re-export shim** — Phase 3 (BIND-02). Phase 1 does NOT touch the upstream Python tree.
- **Maturin wheel build** — Phase 8 (DIST-02). Phase 1 only ensures `pyscf-py` exists as an empty cdylib-capable workspace member.
- **abi3-py310 wheel skeleton** — Phase 3 (BIND-01). Phase 1 sets up the workspace; Phase 3 wires the abi3 build.
- **Tensor-arena pattern in `pyscf-runtime`** — full pattern lands in Phase 6 (CCSD-11). Phase 1 establishes minimal `WorkspacePool` skeleton (Claude's discretion) to avoid Phase 6 retrofit pressure, but does not implement the spill-to-HDF5 path.
- **Python verbosity contract** (`mol.verbose` 0–9, FOUND-09) wiring — Phase 1 establishes `tracing 0.1` infrastructure; Phase 3 wires `mol.verbose` → `tracing-subscriber` filter at the PyO3 boundary.
- **Per-backend regression suite** — Phase 8 (ORACLE-07). Phase 1 only ships the `tests/backend_matrix.rs` smoke test (ALG-07) that proves GEMM/AXPY/reduce-sum agree across compiled backends on a single 256×256 input.
- **`shader-f64` runtime fallback in DFT kernels themselves** — Phase 4 (DFT-11). Phase 1 owns the resolver-side gate (D-09); Phase 4 owns kernel-side robustness if anything still slips through.

</deferred>

---

*Phase: 1-foundation*
*Context gathered: 2026-05-10*
