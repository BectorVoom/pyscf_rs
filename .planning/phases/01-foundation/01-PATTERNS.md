# Phase 1: Foundation — Pattern Map

**Mapped:** 2026-05-10
**Files analyzed:** 33 (15 workspace member Cargo.toml + 15 lib.rs + 5 xtask bins + .cargo/config.toml + deny.toml + 2 CI workflows + CONTRIBUTING.md + release-oracle profile + 4 algebra test files; deduplicated below into 15 mapping rows)
**Analogs found:** 12 / 15 strong matches; 3 NEW PATTERN deliverables

> Phase 1 is greenfield in this repo. All analog paths point at sibling repos under `~/Documents/workspace/{cintx,xcfun_rs,libxc_rs}` per CONTEXT.md `<canonical_refs>` § "Sibling-crate precedent". Every excerpt below is verbatim from those repos at the lines cited; planner / executor MUST mirror them, deviating only with explicit justification in PLAN.md per CONTEXT.md `<specifics>`.

---

## File Classification

| New file (in pyscf_rs) | Role | Data Flow | Closest Analog | Match Quality |
|------------------------|------|-----------|----------------|---------------|
| `/Cargo.toml` (workspace root) | workspace config | static-config | `~/Documents/workspace/xcfun_rs/Cargo.toml` (better than cintx — has `[workspace.package]` + `[workspace.dependencies]` shared block) | exact-shape, NEW content for `[patch.crates-io]` |
| `crates/pyscf-algebra/Cargo.toml` | per-backend feature gate | static-config | `~/Documents/workspace/cintx/crates/cintx-cubecl/Cargo.toml` | exact (CONTEXT.md `<specifics>`: "verbatim, swap names") |
| `crates/pyscf-algebra/src/lib.rs` (AlgebraClient enum + match dispatch — D-04, D-05) | runtime gateway | request-response (host→GPU dispatch) | `~/Documents/workspace/cintx/crates/cintx-cubecl/src/backend/mod.rs` (`ResolvedBackend`) + `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/src/backend.rs` (Backend enum) | exact for enum shape; **opaque BufferId/Tensor surface (D-05) is NEW** |
| `crates/pyscf-runtime/Cargo.toml` | runtime feature gate | static-config | `~/Documents/workspace/cintx/crates/cintx-runtime/Cargo.toml` | exact |
| `crates/pyscf-runtime/src/lib.rs` (BackendKind, WorkspacePool) | public surface | static-types | `~/Documents/workspace/cintx/crates/cintx-runtime/src/lib.rs` + `options.rs` (`BackendKind`) | exact for `BackendKind`; **`WorkspacePool` shape is NEW PATTERN** (CONTEXT § Claude's Discretion) |
| `crates/pyscf-runtime/src/select.rs` (auto resolver — D-07/D-08/D-09/D-10) | runtime gateway | event-driven (probe-and-skip chain) | `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/src/auto_backend.rs` + `runtime/{cpu,cuda,hip,wgpu}.rs` | exact for priority chain; **per-probe `tracing::info!` line + `PYSCF_DTYPE` axis (D-08) is NEW** |
| `crates/pyscf-core/src/lib.rs` (Mole/Density/Energy + traits — FOUND-02) | public surface | static-types | `~/Documents/workspace/cintx/crates/cintx-core/src/lib.rs` (re-export shape only — domain types are PySCF-specific) | role-match (re-export pattern); **type bodies NEW PATTERN** |
| `crates/pyscf-{kernels,gto,scf,dft,mp2,ccsd,grad,geomopt,py,oracle,bench}/Cargo.toml` (11 stub crates) + top-level façade | stub crate | static-config | xcfun_rs `[workspace] exclude/members` incremental-enabling pattern (Cargo.toml lines 7-30) — every member declared but most are minimal bodies | exact for declaration shape |
| `crates/pyscf-*/src/lib.rs` (11 stub bodies) | stub crate | none | empty/placeholder convention (Phase 1 greenfield) | NEW PATTERN (CONTEXT § Claude's Discretion §3) |
| `xtask/src/bin/check_no_fma.rs` | lint enforcement | batch (asm scan) | `~/Documents/workspace/xcfun_rs/xtask/src/bin/check_no_fma.rs` | exact (mnemonic table + `cargo rustc --emit=asm` + demangle pipeline) |
| `xtask/src/bin/check_forbidden_paths.rs` (FOUND-08) | lint enforcement | batch (source grep) | NEW (xcfun_rs's `check_no_anyhow.rs` is the closest grep-based gate but greps Cargo.toml, not source — adapt the file-walk) | role-match; pattern source: `xtask/src/bin/check_no_anyhow.rs` (walkdir + per-file grep) |
| `xtask/src/bin/check_catch_unwind.rs` | lint enforcement | batch (source grep) | xcfun_rs `xtask/src/bin/check_no_anyhow.rs` (same walkdir + grep pattern, different needle) | role-match |
| `xtask/src/bin/check_dependency_wall.rs` (ALG-06) | lint enforcement | batch (cargo metadata walk) | `~/Documents/workspace/xcfun_rs/xtask/src/bin/check_boundaries.rs` | exact (allowlist HashMap + `cargo metadata --no-deps` + per-pkg dep filter) |
| `xtask/src/bin/check_cubecl_pin.rs` | lint enforcement | batch (cargo metadata walk) | `~/Documents/workspace/xcfun_rs/xtask/src/bin/check_cubecl_pin.rs` | exact |
| `xtask/Cargo.toml` | workspace config | static-config | `~/Documents/workspace/xcfun_rs/xtask/Cargo.toml` | exact (multi-`[[bin]]` block layout) |
| `.cargo/config.toml` (rustflags `-Cllvm-args=-fp-contract=off`) | build config | static-config | `~/Documents/workspace/xcfun_rs/.cargo/config.toml` | exact (verbatim — including `[target.'cfg(all())']` duplication for user-config override resilience) |
| `[profile.release-oracle]` (FMA-free, FOUND-05) | build config | static-config | xcfun_rs CONTEXT D-21 (release profile) — pattern documented; **dedicated `release-oracle` named profile is NEW PATTERN** | role-match |
| `deny.toml` (FOUND-10) | lint enforcement | static-config | NONE in any sibling — `find` returned zero `deny.toml` files in cintx/xcfun_rs/libxc_rs | **NEW PATTERN** |
| `.github/workflows/nightly-cross-crate.yml` (ORACLE-05) | CI | scheduled | xcfun_rs `.github/workflows/{ci.yml,release.yml,validate-order3-sweep.yml}` (workflow shape only — cross-crate matrix is new) | role-match for shape; **cross-crate matrix logic NEW** |
| `.github/workflows/ci.yml` (PR CI: build/clippy/xtask gates/oracle-profile) | CI | scheduled | `~/Documents/workspace/xcfun_rs/.github/workflows/ci.yml` | exact |
| `CONTRIBUTING.md` "local sibling-crate development" recipe (D-15) | docs | static-config | NONE in any sibling — `find -name CONTRIBUTING.md` returned zero matches | **NEW PATTERN** |
| `oracle_sum`/`oracle_dot`/`oracle_einsum` (FOUND-06) | public surface | transform (deterministic reduction) | NONE — neither cintx nor xcfun_rs has an explicit ordered-reduction primitive (CONTEXT § Claude's Discretion §1; RESEARCH recommends pairwise tree reduction with chunk N=128) | **NEW PATTERN** |

---

## Pattern Assignments

### `/Cargo.toml` (workspace root) — workspace config

**Analog:** `~/Documents/workspace/xcfun_rs/Cargo.toml`

**Why xcfun_rs over cintx:** xcfun_rs Cargo.toml uses `[workspace.package]` (shared `version`/`edition`/`rust-version`) and `[workspace.dependencies]` (shared cubecl pin) — both are required for FOUND-04 (lockstep) and FOUND-10 (edition 2024 / rust 1.92 floor). cintx's root Cargo.toml is a hybrid `[package] + [workspace]` and doesn't expose `[workspace.dependencies]`.

**Workspace member declaration pattern** (xcfun_rs Cargo.toml lines 7-31):
```toml
[workspace]
members = [
    "crates/xcfun-ad",
    "crates/xcfun-core",
    "crates/xcfun-kernels",
    "crates/xcfun-eval",
    "crates/xcfun-gpu",
    "crates/xcfun-rs",
    "crates/xcfun-capi",
    "crates/xcfun-py",
    "xtask",
    "validation",
]
exclude = []
resolver = "2"
```
Pyscf-rs adapts this to **15 members**: `crates/pyscf-{core,runtime,algebra,kernels,gto,scf,dft,mp2,ccsd,grad,geomopt,py,oracle,bench}` plus the top-level façade crate plus `xtask`.

**Shared `[workspace.package]` block** (xcfun_rs Cargo.toml lines 34-37):
```toml
[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.92"
```
Pyscf-rs adopts verbatim (FOUND-10).

**Shared `[workspace.dependencies]` cubecl pin** (xcfun_rs Cargo.toml lines 47-54):
```toml
# cubecl pivot — all five must move in lockstep (CLAUDE.md risk note).
cubecl      = "=0.10.0"
cubecl-cpu  = "=0.10.0"
cubecl-hip  = "=0.10.0"
cubecl-cuda = "=0.10.0"
cubecl-wgpu = "=0.10.0"
```
Pyscf-rs adopts verbatim and adds `cubecl-matmul = "=0.9.0-pre.5"`, `cubecl-reduce = "=0.9.0-pre.5"` per RESEARCH § Open Question #1.

**`[patch.crates-io]` for sibling crates (D-12, D-13)** — **NEW** (no sibling has this block; the siblings ARE the patched-in crates):
```toml
# NEW PATTERN — no sibling precedent. Per CONTEXT D-12/D-13.
[patch.crates-io]
cintx     = { git = "https://github.com/BectorVoom/cintx.git",     rev = "<sha>" }
libxc_rs  = { git = "https://github.com/BectorVoom/libxc_rs.git",  rev = "<sha>" }
xcfun_rs  = { git = "https://github.com/BectorVoom/xcfun_rs.git",  rev = "<sha>" }
```
Note: `cubecl` is registry-pinned via `[workspace.dependencies]` above, NOT patched (CONTEXT § "Established Patterns").

---

### `crates/pyscf-algebra/Cargo.toml` (per-backend feature gating)

**Analog:** `~/Documents/workspace/cintx/crates/cintx-cubecl/Cargo.toml`

**Per-backend feature block** (cintx-cubecl Cargo.toml lines 9-32):
```toml
[features]
default = ["cpu"]
cpu = ["cubecl/cpu", "cintx-runtime/cpu"]
wgpu = ["dep:cubecl-wgpu", "dep:wgpu", "cintx-runtime/wgpu"]
cuda = ["dep:cubecl-cuda", "cintx-runtime/cuda"]
rocm = ["dep:cubecl-hip", "cintx-runtime/rocm"]
# M1 alias: `metal` reuses the wgpu runtime on Apple targets (cubecl-metal
# does not exist on crates.io). Forwarding to `wgpu` pulls in cubecl-wgpu +
# wgpu, and `cintx-runtime/metal` activates the typed `BackendKind::Metal`
# arm so the public `CINTX_BACKEND=metal` surface remains distinct from
# `=wgpu` for capability fingerprints and error diagnostics.
metal = ["wgpu", "cintx-runtime/metal"]
```

**Per-backend optional cubecl deps** (cintx-cubecl Cargo.toml lines 34-46):
```toml
[dependencies]
cintx-core = { path = "../cintx-core" }
cintx-runtime = { path = "../cintx-runtime", default-features = false }
cubecl = { version = "0.10.0" }
cubecl-wgpu = { version = "0.10.0", optional = true }
cubecl-cuda = { version = "0.10.0", optional = true }
cubecl-hip  = { version = "0.10.0", optional = true }
cubecl-runtime = "0.10.0"
tracing = "0.1"
wgpu = { version = "29.0.3", optional = true }
```
Pyscf-rs `pyscf-algebra` swaps names: `pyscf-core`/`pyscf-runtime` instead of `cintx-core`/`cintx-runtime`; same `metal = ["wgpu", "pyscf-runtime/metal"]` alias; same per-backend optional cubecl deps; ALG-03 adds workspace `gpu` umbrella feature `gpu = ["cuda", "wgpu"]` at the algebra crate level (host-portable subset per ROADMAP.md line 33).

---

### `crates/pyscf-algebra/src/lib.rs` — AlgebraClient enum + match dispatch (D-04, D-05)

**Analog (enum shape):** `~/Documents/workspace/cintx/crates/cintx-cubecl/src/backend/mod.rs`
**Analog (probe + cache pattern):** `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/src/backend.rs`

**`#[cfg(feature = "...")]`-gated enum arms** (cintx-cubecl `backend/mod.rs` lines 35-55):
```rust
pub enum ResolvedBackend {
    /// CPU backend client (default-on; `cpu` feature gates the cubecl/cpu
    /// runtime crate).
    #[cfg(feature = "cpu")]
    Cpu(cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime>),
    /// wgpu GPU backend client, paired with the adapter's feature names for
    /// capability checks (e.g. SHADER_F64 gating).
    #[cfg(feature = "wgpu")]
    Wgpu(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),
    /// CUDA backend client. Compile-only this phase ...
    #[cfg(feature = "cuda")]
    Cuda(cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>),
    /// ROCm backend client (cubecl-hip). Runtime-verifiable on the dev host.
    #[cfg(feature = "rocm")]
    Rocm(cubecl::client::ComputeClient<cubecl_hip::HipRuntime>),
    /// Metal — M1 alias: dispatches through `cubecl_wgpu::WgpuRuntime` on
    /// Apple targets. ...
    #[cfg(feature = "metal")]
    Metal(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),
}
```

**Match-dispatch shape** (cintx-cubecl `backend/mod.rs` lines 84-122):
```rust
pub fn from_intent(intent: &BackendIntent) -> Result<Self, cintxRsError> {
    match &intent.backend {
        BackendKind::Cpu => {
            #[cfg(feature = "cpu")]
            { let client = cpu_backend::resolve_cpu_client()?; Ok(ResolvedBackend::Cpu(client)) }
            #[cfg(not(feature = "cpu"))]
            Err(cintxRsError::UnsupportedApi { requested: "cpu-backend:feature-not-enabled".to_owned() })
        }
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => {
            let report = crate::runtime_bootstrap::bootstrap_wgpu_runtime(intent)?;
            let features = report.snapshot.features.clone();
            let client = wgpu_backend::resolve_wgpu_client(intent)?;
            Ok(ResolvedBackend::Wgpu(client, features))
        }
        // ... cuda, rocm, metal arms omitted for brevity ...
    }
}
```

**Per-backend client construction** (cintx-cubecl `backend/cpu_backend.rs` lines 1-18 — entire file):
```rust
#![cfg(feature = "cpu")]

use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::cpu::{CpuDevice, CpuRuntime};

/// Resolve a CPU `ComputeClient` using the default `CpuDevice`.
pub fn resolve_cpu_client() -> Result<ComputeClient<CpuRuntime>, cintxRsError> {
    Ok(CpuRuntime::client(&CpuDevice::default()))
}
```
Pyscf-rs `pyscf-algebra/src/backend/cpu.rs` mirrors this verbatim; same one-line pattern for `cuda.rs` (`CudaRuntime::client(&CudaDevice::default())`), `rocm.rs` (`HipRuntime::client(&AmdDevice::default())`), `wgpu.rs` (`WgpuRuntime::client(&WgpuDevice::default())`).

**Opaque `BufferId` + `Tensor` surface (D-05)** — **NEW PATTERN**: neither cintx-cubecl nor xcfun-gpu hides the `cubecl::client::ComputeClient<R>` from downstream method crates — they expose it directly via the enum's tuple fields. Per CONTEXT D-05 the planner must design a wrapper:
```rust
// NEW — D-05 opaque boundary. No sibling precedent.
pub struct BufferId(u64);  // monotonic counter into AlgebraClient's owned buffer arena
pub struct Tensor { id: BufferId, shape: Vec<usize>, dtype: DType }
pub enum DType { F32, F64 }  // PYSCF_DTYPE axis (D-08)
```

The match-dispatch reconstruction site (CONTEXT D-05: "algebra primitives reconstruct the underlying `cubecl::TensorHandle<R, T>` inside the matched arm") is also greenfield. Pattern source for the launch shape: `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` (`cubecl_matmul::launch::<R, T>(&Strategy::Auto, &client, lhs, rhs, out)`) and `docs/manual/Cubecl/cubecl_reduce_sum.md` per CONTEXT.md `<canonical_refs>`.

---

### `crates/pyscf-runtime/Cargo.toml` (typed-arm feature flags)

**Analog:** `~/Documents/workspace/cintx/crates/cintx-runtime/Cargo.toml` (entire file, 28 lines):
```toml
[package]
name = "cintx-runtime"
version = "0.1.0"
edition = "2024"

[lib]
path = "src/lib.rs"

[features]
default = ["cpu"]
cpu = []
wgpu = []
cuda = []
rocm = []
metal = []

[dependencies]
cintx-core = { path = "../cintx-core" }
cintx-ops  = { path = "../cintx-ops" }
tracing = "0.1"
```

**Critical pattern (cintx-runtime/Cargo.toml lines 10-22):** the runtime crate's per-backend features pull NO cubecl deps — they just gate the `BackendKind` enum arm. The `pyscf-algebra` crate's per-backend features forward to these flags (see `cintx-cubecl/Cargo.toml` `cpu = ["cubecl/cpu", "cintx-runtime/cpu"]`). This is the dependency-wall mechanism (ALG-06): `pyscf-runtime` never names a `cubecl-*` crate.

---

### `crates/pyscf-runtime/src/lib.rs` — BackendKind + WorkspacePool

**Analog (BackendKind):** `~/Documents/workspace/cintx/crates/cintx-runtime/src/options.rs` lines 16-46

**Per-variant cfg-gated enum** (cintx-runtime `options.rs` lines 16-36):
```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// CPU execution profile. Always available — `cpu` feature is default-on (D-06).
    Cpu,
    /// wgpu-backed CubeCL runtime.
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// CUDA-backed CubeCL runtime ...
    #[cfg(feature = "cuda")]
    Cuda,
    /// ROCm/HIP-backed CubeCL runtime.
    #[cfg(feature = "rocm")]
    Rocm,
    /// Metal — M1 alias for the wgpu runtime on Apple targets.
    #[cfg(feature = "metal")]
    Metal,
}

impl Default for BackendKind {
    fn default() -> Self {
        // Cpu is the typed default — always, infallibly.
        Self::Cpu
    }
}
```
Pyscf-rs adopts verbatim. Naming note: pyscf uses `BackendKind` (per FOUND-03 / ALG-04 RFC text) where xcfun-gpu uses `Backend`; the cintx variant matches pyscf's naming.

**`Backend::from_str` parser** (xcfun-gpu `src/backend.rs` lines 39-52):
```rust
#[allow(clippy::should_implement_trait)]
pub fn from_str(s: &str) -> Option<Self> {
    match s.to_ascii_lowercase().as_str() {
        "cpu"          => Some(Backend::Cpu),
        "rocm" | "hip" => Some(Backend::Rocm),
        "cuda"         => Some(Backend::Cuda),
        "metal"        => Some(Backend::Metal),
        "wgpu"         => Some(Backend::Wgpu),
        _              => None,
    }
}
```
Pyscf-rs `BackendKind::from_env_str` mirrors verbatim, with the addition that an `auto` token returns a sentinel for the resolver to handle (D-07).

**`re-export shape`** (cintx-runtime `src/lib.rs` lines 1-26):
```rust
//! Runtime planning and workspace governance for cintx.

pub mod dispatch;
pub mod metrics;
pub mod options;
// ... module declarations ...

pub use options::{BackendCapabilityToken, BackendIntent, BackendKind, ExecutionOptions};
pub use workspace::{
    ChunkInfo, ChunkPlan, ChunkPlanner, FallibleBuffer, HostWorkspaceAllocator,
    WorkspaceAllocator, WorkspaceQuery, WorkspaceRequest,
};
```
Pyscf-rs `pyscf-runtime/src/lib.rs` mirrors this re-export shape (one `pub mod foo; pub use foo::Bar;` line per public type).

**`WorkspacePool` (FOUND-03)** — **NEW PATTERN**: cintx-runtime has `WorkspaceQuery`/`WorkspaceRequest`/`ChunkPlanner` (memory-budget chunk planner), NOT a tensor buffer pool. xcfun-gpu has `pool::BatchBuffers` (fixed-size + power-of-two grow buffers — `pool.rs` lines 56-68) but it's batch-bound, not arena-shaped. CONTEXT § "Claude's Discretion" leaves the shape open; RESEARCH § Open Question #3 recommends three fields (`max_memory_bytes` from `PYSCF_MAX_MEMORY` MB env var, a `RwLock<Vec<BufferId>>` free-list, generation counter). Planner picks. Reference excerpt for the cintx memory-budget parser shape (cintx-runtime `options.rs` lines 1-3, 119-133):
```rust
pub const DEFAULT_MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

impl ExecutionOptions {
    pub fn with_default_limits() -> Self {
        Self { memory_limit_bytes: Some(DEFAULT_MEMORY_LIMIT_BYTES), ..Self::default() }
    }
    pub const fn default_memory_limit_bytes() -> usize { DEFAULT_MEMORY_LIMIT_BYTES }
    pub fn effective_memory_limit_bytes(&self, required_bytes: usize) -> usize {
        self.memory_limit_bytes.unwrap_or(required_bytes)
    }
}
```

---

### `crates/pyscf-runtime/src/select.rs` — auto resolver (D-07/D-08/D-09/D-10)

**Analog:** `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/src/auto_backend.rs`

**Priority-chain skeleton** (xcfun-gpu `auto_backend.rs` lines 31-62 — entire fn body):
```rust
pub fn auto_backend() -> Backend {
    if let Ok(force) = std::env::var("XCFUN_FORCE_BACKEND") {
        return Backend::from_str(&force).unwrap_or_else(|| {
            panic!("XCFUN_FORCE_BACKEND={force:?} unrecognised \
                    (expected one of: cpu | rocm | hip | cuda | metal | wgpu)")
        });
    }

    #[cfg(feature = "hip")]
    if crate::runtime::hip::rocm_available() { return Backend::Rocm; }

    #[cfg(feature = "cuda")]
    if crate::runtime::cuda::cuda_available() { return Backend::Cuda; }

    #[cfg(feature = "wgpu")]
    if crate::runtime::wgpu::metal_with_f64_available() { return Backend::Metal; }

    #[cfg(feature = "wgpu")]
    if crate::runtime::wgpu::wgpu_with_shader_f64_available() { return Backend::Wgpu; }

    Backend::Cpu
}
```
Pyscf-rs adopts the exact priority order from CONTEXT D-07 (`cuda → rocm → metal → wgpu → cpu`), which differs from xcfun's `hip → cuda → metal → wgpu → cpu` order. Use this skeleton with the order swapped per D-07.

**Deviation 1 — unrecognised values do NOT panic (D-07 + ALG-04):** xcfun panics on unrecognised; pyscf-rs **logs a `tracing::warn!` and falls back to CPU** per FOUND-03 ("on missing/unrecognised/uncompiled values it falls back to CPU"). Replace the `unwrap_or_else(|| panic!(...))` with:
```rust
// PYSCF DEVIATION from xcfun-gpu — ALG-04 unrecognised → CPU + warn.
return Backend::from_env_str(&force).unwrap_or_else(|| {
    tracing::warn!(env = %force, "PYSCF_BACKEND unrecognised; falling back to Cpu");
    Backend::Cpu
});
```

**Deviation 2 — per-probe `tracing::info!` line (D-07):** xcfun's chain is silent on skip. Pyscf-rs MUST emit one info line per probe attempt, success or skip:
```rust
// NEW — D-07 observability requirement.
#[cfg(feature = "cuda")]
{
    if crate::probe::cuda::cuda_available() {
        tracing::info!("probe cuda — available; selecting");
        return BackendKind::Cuda;
    } else {
        tracing::info!("probe cuda — no device or driver mismatch; skipping");
    }
}
```

**Deviation 3 — `PYSCF_DTYPE` axis (D-08):** xcfun probes only for hardware availability; pyscf-rs `PYSCF_DTYPE=f64` adds an additional gate — wgpu probe returns `false` (auto mode) or hard-error (explicit mode) when adapter lacks `shader-f64`.

**Wgpu f64 probe** — adopt `wgpu_with_shader_f64_available` verbatim (xcfun-gpu `runtime/wgpu.rs` lines 38-87):
```rust
use cubecl::Runtime;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::ComputeClient;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use std::sync::OnceLock;

pub type WgpuClient = ComputeClient<WgpuRuntime>;
static WGPU_CLIENT: OnceLock<Option<WgpuClient>> = OnceLock::new();

fn init_wgpu_with_f64() -> Option<WgpuClient> {
    let init = std::panic::catch_unwind(|| {
        let device = WgpuDevice::default();
        WgpuRuntime::client(&device)
    });
    let client = init.ok()?;
    if client.properties().supports_type(ElemType::Float(FloatKind::F64)) {
        Some(client)
    } else {
        None
    }
}

pub fn wgpu_with_shader_f64_available() -> bool {
    WGPU_CLIENT.get_or_init(init_wgpu_with_f64).is_some()
}
```
Note the `std::panic::catch_unwind` wrapper around `WgpuRuntime::client` — this is required (RESEARCH § "Pitfall 7": cubecl-wgpu can panic during dynamic-link resolution). Pyscf-rs MUST keep `catch_unwind` per FOUND-07.

**CUDA probe** — adopt verbatim (xcfun-gpu `runtime/cuda.rs` lines 73-105) including `catch_unwind` and the `supports_type(ElemType::Float(FloatKind::F64))` defensive gate.

**HIP/ROCm probe** — adopt verbatim (xcfun-gpu `runtime/hip.rs` lines 69-85).

**CPU "probe" is trivially true** (xcfun-gpu `runtime/cpu.rs` lines 15-17):
```rust
pub fn cpu_available() -> bool { true }
```

**Final resolution log line (ALG-08):** after the chain decides, emit:
```rust
// NEW — ALG-08 mandatory observability line.
tracing::info!(
    "pyscf-algebra: backend={resolved} (env={raw}, dtype={dtype})",
    resolved = resolved, raw = raw_env.as_deref().unwrap_or("unset"), dtype = dtype
);
```

---

### `crates/pyscf-core/src/lib.rs` — universal types + traits (FOUND-02)

**Analog (re-export shape):** `~/Documents/workspace/cintx/crates/cintx-core/src/lib.rs`

**Re-export pattern** (cintx-core `lib.rs` lines 1-21 — entire file):
```rust
//! Core domain primitives for cintx ...

pub mod atom;
pub mod basis;
pub mod env;
pub mod error;
pub mod operator;
pub mod shell;
pub mod tensor;

pub use atom::{Atom, NuclearModel};
pub use basis::{BasisMeta, BasisSet};
pub use env::{EnvBoundsError, EnvParams, EnvUnits};
pub use error::{CoreError, cintxRsError};
pub use operator::{OperatorId, Representation};
pub use shell::{Shell, ShellTuple, ShellTupleArityError};
pub use tensor::{TensorLayout, TensorShape};
```
Pyscf-rs `pyscf-core/src/lib.rs` mirrors the re-export shape: `pub mod mole; pub mod density; pub mod mo; pub mod amplitudes; pub mod energy; pub mod traits;` then `pub use ...`.

**Type bodies** — **NEW PATTERN**: `Mole`, `BasisSet` (re-export from cintx_core per GTO-11), `Density`, `MOCoefficients`, `Amplitudes`, `Energy` newtype, `Method`/`Scf`/`KohnSham`/`PostScf`/`Gradient`/`IntegralEngine` traits are PySCF-domain-specific. The greenfield surface is owned by FOUND-02; planner cites the Python `pyscf/gto/mole.py`, `pyscf/scf/hf.py` etc. as reference (CONTEXT.md `<canonical_refs>` § "Codebase maps") but the Rust types are new.

**xcfun-core deny convention** (xcfun-core `lib.rs` line 10):
```rust
#![forbid(unsafe_code)]
```
Pyscf-rs `pyscf-core/src/lib.rs` adopts `#![forbid(unsafe_code)]` since FOUND-02 says "no compute dependencies" — pure types and traits, no unsafe.

---

### Stub crates (11 method/façade Cargo.toml + lib.rs)

**Analog (declaration shape):** `~/Documents/workspace/xcfun_rs/Cargo.toml` lines 7-30 (incremental member-enabling)

xcfun_rs declares ALL members up-front and uses comments to mark which Phase wires which crate ("Phase 6 Plan 06-02a: xcfun-gpu promoted from exclude"). pyscf-rs adopts the same: every one of the 15 members is in `members = [...]` from day one even though only `core/runtime/algebra` are non-stub.

**Stub `Cargo.toml` shape** (xcfun-gpu `Cargo.toml` lines 1-22 — minimal block):
```toml
[package]
name = "xcfun-gpu"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
description = "..."
license = "MPL-2.0"

[features]
default = ["cpu"]
cpu  = ["dep:cubecl-cpu", "xcfun-eval/testing"]
# ...

[dependencies]
xcfun-core = { path = "../xcfun-core" }
# ...
```
Pyscf-rs stub crates use the same `version.workspace = true` shorthand with license `Apache-2.0` (per FOUND-10).

**Stub `lib.rs` body** — **NEW PATTERN** per CONTEXT § Claude's Discretion §3. RESEARCH recommendation: empty body with `#![forbid(unsafe_code)]` + a single `// TODO: implemented in Phase N` comment. No sibling precedent (every sibling crate's `lib.rs` is non-stub once it's in `members`).

```rust
// NEW — Phase 1 stub convention.
#![forbid(unsafe_code)]

// TODO: implemented in Phase {N}. Phase 1 ships this empty so
// `cargo build --workspace` compiles all 15 members (FOUND-01).
```

---

### `xtask/src/bin/check_no_fma.rs` — FMA-free CI gate (FOUND-05)

**Analog:** `~/Documents/workspace/xcfun_rs/xtask/src/bin/check_no_fma.rs` (entire file, 276 lines)

**Forbidden mnemonics table** (xcfun_rs `check_no_fma.rs` lines 38-73):
```rust
const FORBIDDEN_MNEMONICS: &[&str] = &[
    "vfmadd132pd", "vfmadd213pd", "vfmadd231pd",
    "vfmadd132sd", "vfmadd213sd", "vfmadd231sd",
    "vfmsub132pd", "vfmsub213pd", "vfmsub231pd",
    "vfmsub132sd", "vfmsub213sd", "vfmsub231sd",
    "vfnmadd132pd", "vfnmadd213pd", "vfnmadd231pd",
    "vfnmadd132sd", "vfnmadd213sd", "vfnmadd231sd",
    "vfnmsub132pd", "vfnmsub213pd", "vfnmsub231pd",
    "vfnmsub132sd", "vfnmsub213sd", "vfnmsub231sd",
    // aarch64 + generic spellings
    "fmadd", "fmsub", "fnmadd", "fnmsub",
    // LLVM-intrinsic-style (belt-and-suspenders)
    "fma213", "fma231",
];
```
Adopt verbatim.

**Pipeline shape** (xcfun_rs `check_no_fma.rs` lines 117-225):
1. `cargo rustc -p <crate> --release --lib -- --emit=asm` (line 138-148)
2. Walk `target/release/deps/*.s` (lines 158-181)
3. Per-symbol scan with `rustc_demangle::demangle` + needle match (lines 235-275)
4. Exit code 2 on FAIL, 0 on PASS, 1 on infrastructure error.

Pyscf-rs `xtask/src/bin/check_no_fma.rs` swaps the `SCAN_TARGETS` table (xcfun_rs lines 88-101) to scan **all** numerical pyscf-rs crates (`pyscf-algebra`, `pyscf-core`, `pyscf-kernels` once non-stub) under `--profile release-oracle`:
```rust
// Adapted from xcfun-eval/check_no_fma.rs lines 88-101.
const SCAN_TARGETS: &[(&str, &str, &[&str])] = &[
    ("pyscf-algebra", "pyscf_algebra", &["pyscf_algebra::"]),
    ("pyscf-core",    "pyscf_core",    &["pyscf_core::"]),
    // pyscf-kernels added once it lands as non-stub.
];
```
And replaces `--release` with `--profile release-oracle` (FOUND-05 specifies the named oracle profile, not bare `--release`).

---

### `xtask/src/bin/check_dependency_wall.rs` — algebra dependency wall (ALG-06)

**Analog:** `~/Documents/workspace/xcfun_rs/xtask/src/bin/check_boundaries.rs` (entire file, 133 lines)

**Allowlist HashMap pattern** (xcfun_rs `check_boundaries.rs` lines 39-65):
```rust
fn allowlist() -> HashMap<&'static str, &'static [&'static str]> {
    let mut m = HashMap::new();
    m.insert("xcfun-core", &["thiserror", "bitflags"][..]);
    m.insert("xcfun-ad",   &["cubecl", "cubecl-cpu", "bytemuck"][..]);
    m.insert("xcfun-kernels",
        &["xcfun-core", "xcfun-ad", "cubecl", "thiserror"][..]);
    m.insert("xcfun-eval",
        &["xcfun-core", "xcfun-ad", "xcfun-kernels", "cubecl", "cubecl-cpu", "thiserror"][..]);
    m
}
```

**Walk pattern** (xcfun_rs `check_boundaries.rs` lines 67-112):
```rust
let output = Command::new("cargo")
    .current_dir(&root)
    .args(["metadata", "--format-version", "1", "--no-deps"])
    .output()
    .context("failed to spawn `cargo metadata`")?;
// ... parse JSON ...
let packages = metadata["packages"].as_array().unwrap_or(&empty_vec);
for pkg in packages {
    let name = pkg["name"].as_str().unwrap_or("");
    let Some(allowed) = allow.get(name) else { continue; };
    let deps = pkg["dependencies"].as_array().unwrap_or(&empty_vec);
    for dep in deps {
        let dep_name = dep["name"].as_str().unwrap_or("");
        let is_normal = dep["kind"].is_null();
        if !is_normal { continue; }
        if !allowed.contains(&dep_name) {
            violations.push(format!("{}: normal dep `{}` not in allowlist {:?}",
                name, dep_name, allowed));
        }
    }
}
```
Adopt verbatim.

**Pyscf-rs allowlist** — **inverse direction** (block-list): xcfun_rs whitelists what each crate MAY depend on. ALG-06 instead specifies a **denylist**: NO crate other than `pyscf-algebra` and `pyscf-runtime` may depend on `cubecl-*`. Adapt the pattern:
```rust
// ALG-06 denylist (pyscf-rs adaptation of xcfun_rs allowlist pattern).
const FORBIDDEN_DEPS: &[&str] = &[
    "cubecl", "cubecl-cpu", "cubecl-wgpu", "cubecl-cuda", "cubecl-hip",
    "cubecl-runtime", "cubecl-matmul", "cubecl-reduce", "cubecl-std",
];
const ALLOWED_CRATES_FOR_CUBECL: &[&str] = &["pyscf-algebra", "pyscf-runtime"];
```
Then walk packages and fail if `pkg.name not in ALLOWED_CRATES_FOR_CUBECL` AND any normal dep is in `FORBIDDEN_DEPS`.

---

### `xtask/src/bin/check_cubecl_pin.rs` (FOUND-04)

**Analog:** `~/Documents/workspace/xcfun_rs/xtask/src/bin/check_cubecl_pin.rs` (entire file, 100 lines)

**Verbatim adoption** (xcfun_rs `check_cubecl_pin.rs` lines 17-31):
```rust
const REQUIRED_VERSION: &str = "0.10.0";
const PINNED_CRATES: &[&str] = &[
    "cubecl",
    "cubecl-cpu",
    "cubecl-hip",
    "cubecl-cuda",
    "cubecl-wgpu",
];
```

**Walk + assert version equality** (xcfun_rs `check_cubecl_pin.rs` lines 43-99):
```rust
let metadata: Value = serde_json::from_slice(&output.stdout).context("parse cargo metadata JSON")?;
let packages = metadata["packages"].as_array().unwrap_or(&empty_vec);
let mut violations = Vec::new();
for pkg in packages {
    let name = pkg["name"].as_str().unwrap_or("");
    if !PINNED_CRATES.contains(&name) { continue; }
    let version = pkg["version"].as_str().unwrap_or("");
    if version != REQUIRED_VERSION {
        violations.push(format!("{}: version {} (expected {})",
            name, version, REQUIRED_VERSION));
    }
}
```
Pyscf-rs adopts verbatim. Add `cubecl-matmul = "=0.9.0-pre.5"` and `cubecl-reduce = "=0.9.0-pre.5"` to a separate `PINNED_PRE_CRATES` table since their version differs (RESEARCH § Open Question #1).

---

### `xtask/src/bin/check_forbidden_paths.rs` (FOUND-08) and `check_catch_unwind.rs` (FOUND-07)

**Analog (grep/walkdir pattern):** `~/Documents/workspace/xcfun_rs/xtask/src/bin/check_no_anyhow.rs` (per `xtask/Cargo.toml` lines 30-31; uses `walkdir = "2"` per Cargo.toml line 106)

Pattern: `walkdir::WalkDir` over `crates/*/src/**/*.rs`, per-file string match, fail on first hit with `path:line` reported.

**Pyscf-rs `check_forbidden_paths.rs` needle list** (FOUND-08; ROADMAP.md line 43):
```rust
// FOUND-08 — refuse imports from out-of-scope upstream PySCF modules.
const FORBIDDEN_IMPORT_NEEDLES: &[&str] = &[
    "use pyscf::pbc",   "use pyscf::x2c",  "use pyscf::mcscf",
    "use pyscf::tdscf", "use pyscf::adc",  "use pyscf::gw",
    "use pyscf::eom",   "use pyscf::nac",  "use pyscf::eph",
];
```

**Pyscf-rs `check_catch_unwind.rs` needle pattern** (FOUND-07): grep for `extern "C"` blocks and assert each one's body contains `catch_unwind`. Pattern: file-walk + simple state machine that tracks `extern "C" {` open-brace through close-brace and checks the spanned text for `catch_unwind`. No exact sibling — adapt xcfun_rs's `walkdir` shell.

---

### `xtask/Cargo.toml`

**Analog:** `~/Documents/workspace/xcfun_rs/xtask/Cargo.toml`

**Multi-`[[bin]]` block layout** (xcfun_rs `xtask/Cargo.toml` lines 1-75):
```toml
[package]
name = "xtask"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish = false
description = "Internal build/fixture automation — not published"

[[bin]]
name = "xtask"
path = "src/main.rs"

[[bin]]
name = "check-no-fma"
path = "src/bin/check_no_fma.rs"

[[bin]]
name = "check-boundaries"
path = "src/bin/check_boundaries.rs"

[[bin]]
name = "check-cubecl-pin"
path = "src/bin/check_cubecl_pin.rs"

# ...
```

**Dependency footprint** (xcfun_rs `xtask/Cargo.toml` lines 84-108):
```toml
[dependencies]
anyhow = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
walkdir = "2"
toml = "0.8"
rustc-demangle = "0.1"
```
Pyscf-rs adopts: 5 `[[bin]]` blocks (`check-no-fma`, `check-forbidden-paths`, `check-catch-unwind`, `check-dependency-wall`, `check-cubecl-pin`); same dep set minus the xcfun-specific ones (`cbindgen`, `chrono`, `validation`).

---

### `.cargo/config.toml` (FOUND-05 build-time `-fp-contract=off`)

**Analog:** `~/Documents/workspace/xcfun_rs/.cargo/config.toml` (entire file, 30 lines)

**Verbatim adoption** (xcfun_rs `.cargo/config.toml` lines 1-30):
```toml
# `[build]` applies `-Cllvm-args=-fp-contract=off` to ALL cargo profiles
# (debug, test, bench, release). Applies stricter; no release-only behaviour
# is relaxed. Simpler than `[target.'cfg(all())']`, identical effect on
# `cargo build --release`, and keeps debug/test builds numerically consistent
# with release — a useful property for the 1e-12 parity contract when
# developers run tests locally.
#
# We also set the flag under `[target.'cfg(all())']` because cargo's
# precedence rules let any `[target.*]` rustflags in a higher-priority config
# (e.g. a developer's `~/.cargo/config.toml` with
# `[target.'cfg(target_os = "linux")']`) fully override `[build] rustflags` —
# cargo does NOT merge them. Duplicating into `[target.'cfg(all())']` here
# wins at the same precedence tier as user-level target-specific sections, so
# the 1e-12 parity guard survives on machines with a user-level cargo config.

[build]
rustflags = ["-Cllvm-args=-fp-contract=off"]

[target.'cfg(all())']
rustflags = ["-Cllvm-args=-fp-contract=off"]

[profile.dev]
incremental = false
```
The `[target.'cfg(all())']` duplication is **load-bearing** per CONTEXT.md `<security_domain>` "V14 Configuration" — without it, a developer's `~/.cargo/config.toml` `[target.'cfg(target_os = "linux")']` rustflags fully overrides `[build] rustflags`, silently re-enabling FMA contraction. Adopt verbatim.

---

### `[profile.release-oracle]` (FOUND-05)

**Analog (close):** `~/Documents/workspace/libxc_rs/Cargo.toml` lines 96-103 (`[profile.release]` shape with `debug = 0`, `incremental = false`, `codegen-units = 256`):
```toml
[profile.release]
debug = 0
incremental = false
codegen-units = 256
```

**No sibling has a named `release-oracle` profile** — **NEW PATTERN**.

Per FOUND-05 + ROADMAP.md success criterion 2 + RESEARCH.md `RUSTFLAGS="-C target-feature=-fma"`, the workspace `Cargo.toml` adds:
```toml
# NEW — FOUND-05 oracle profile, FMA-free per ROADMAP success criterion 2.
[profile.release-oracle]
inherits = "release"
debug = 0
incremental = false
codegen-units = 1   # bit-reproducible across machines (Pitfall 12 cross-platform µHartree)
panic = "abort"     # CONTEXT § Claude's Discretion: "panic=abort on both release profiles"
# Note: -C target-feature=-fma is set via RUSTFLAGS env var in the
# release-oracle CI job, NOT in this profile block (rustflags can't be set
# per-profile in stable Cargo as of 2026-05-10).
```

`[profile.release]` mirror block adopts the same `panic = "abort"` (CONTEXT § Claude's Discretion).

---

### `deny.toml` (FOUND-10) — **NEW PATTERN**

**No sibling precedent.** `find ~/Documents/workspace/{cintx,xcfun_rs,libxc_rs} -maxdepth 3 -name deny.toml` returns zero matches.

Planner designs from scratch using `cargo deny init` output as starting template. Required gates per CONTEXT.md `<security_domain>` "V10 Malicious Code":
- `[advisories]` — block known-vulnerable transitive deps.
- `[bans]` — block `openssl-sys` (pyscf-rs is statically linked-only per DIST-04/DIST-05); block `system-deps`; block `pkg-config` if any.
- `[licenses]` — Apache-2.0 + MIT + BSD-3-Clause + ISC + Unicode-DFS-2016 (cubecl ecosystem); deny GPL/AGPL/LGPL.
- `[sources]` — only crates.io and the three pinned BectorVoom git remotes per D-13.

---

### `.github/workflows/ci.yml` (per-PR CI)

**Analog:** `~/Documents/workspace/xcfun_rs/.github/workflows/ci.yml` (entire file, 129 lines)

**Skeleton structure** (xcfun_rs `ci.yml` lines 1-13):
```yaml
name: CI

on:
  push: { branches: [master] }
  pull_request:
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: ""           # CLAUDE.md: empty in CI — no fast-math / reassociation
  RUST_BACKTRACE: 1
```
Pyscf-rs adopts verbatim with `branches: [master, main]`. The `RUSTFLAGS: ""` is **load-bearing** (FOUND-05 + Pitfall 1) — explicitly empty so no inherited GitHub-Actions-default fast-math sneaks in.

**Job structure** (xcfun_rs `ci.yml` lines 14-77; four jobs: `fmt`, `clippy`, `build`, `test`):
```yaml
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
    name: cargo build --workspace --release
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { toolchain: stable }
      - uses: Swatinem/rust-cache@v2
      - run: cargo build --workspace --locked --release

  test:
    name: cargo nextest run --workspace
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: taiki-e/install-action@v2
        with: { tool: cargo-nextest }
      - run: cargo nextest run --workspace --locked --release --no-fail-fast
```
Pyscf-rs adopts verbatim. Then **adds** five additional jobs not present in xcfun_rs (xcfun has check_no_fma but only one xtask gate per workflow; pyscf-rs needs all four):
- `xtask-no-fma`: `cargo run -p xtask --bin check-no-fma` (with `--profile release-oracle` invocation)
- `xtask-forbidden-paths`: `cargo run -p xtask --bin check-forbidden-paths`
- `xtask-catch-unwind`: `cargo run -p xtask --bin check-catch-unwind`
- `xtask-dependency-wall`: `cargo run -p xtask --bin check-dependency-wall`
- `xtask-cubecl-pin`: `cargo run -p xtask --bin check-cubecl-pin`
- `oracle-determinism`: `cargo test --profile release-oracle --workspace --locked --no-fail-fast` with `env: { RAYON_NUM_THREADS: "1" }` (ORACLE-09)
- `cargo-deny`: `cargo install cargo-deny && cargo deny check` (FOUND-10; per RESEARCH § Environment Availability "partial — installed via `cargo install cargo-deny` in CI step")

Per RESEARCH § Open Question #4 recommendation: parallel separate jobs (each compiles xtask once then runs its check).

---

### `.github/workflows/nightly-cross-crate.yml` (ORACLE-05)

**Analog (workflow shape):** `~/Documents/workspace/xcfun_rs/.github/workflows/ci.yml` (general workflow shape) + `xcfun_rs/.github/workflows/regen-mpmath-full.yml` (scheduled cron pattern)

**Cron + manual dispatch trigger** (adapt from xcfun_rs `ci.yml` lines 1-7 + cintx `compat-governance-pr.yml` lines 24-25):
```yaml
on:
  schedule:
    - cron: "0 4 * * *"        # 04:00 UTC nightly
  workflow_dispatch:
```

**Cross-crate matrix logic — NEW** per D-14:
```yaml
jobs:
  cross-crate-matrix:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      # D-14: bump pinned SHAs of sibling crates and rebuild against fresh tip.
      - run: cargo update -p cintx -p libxc_rs -p xcfun_rs
      - run: cargo build --workspace --features gpu --locked
      - run: cargo test --workspace --features gpu --locked --no-fail-fast
      - run: cargo run -p xtask --bin check-cubecl-pin
```

This logic does NOT exist in any sibling — it's the lockstep-enforcement mechanism per D-14 / ROADMAP success criterion 4. Failure on this nightly blocks merge of any sibling-crate cubecl bump until pyscf-rs is fixed.

---

### `CONTRIBUTING.md` "local sibling-crate development" (D-15) — **NEW PATTERN**

**No sibling precedent.** `find ~/Documents/workspace/{cintx,xcfun_rs,libxc_rs} -name CONTRIBUTING.md` returns zero matches.

Per D-15 the recipe is path-dep override via the developer's `~/.cargo/config.toml`:
```toml
# Developer-local override, NOT shipped in pyscf_rs.
[patch.crates-io]
cintx     = { path = "/home/<user>/Documents/workspace/cintx" }
libxc_rs  = { path = "/home/<user>/Documents/workspace/libxc_rs" }
xcfun_rs  = { path = "/home/<user>/Documents/workspace/xcfun_rs" }
```
Planner writes `CONTRIBUTING.md` from scratch with this recipe, the cubecl pin upgrade ritual (FOUND-04 → "documented upgrade ritual"), and the four xtask gate invocation cheatsheet.

---

### `oracle_sum`/`oracle_dot`/`oracle_einsum` (FOUND-06) — **NEW PATTERN**

**No sibling precedent.** Neither `cintx` nor `xcfun_rs` ships an explicit ordered-reduction primitive. CONTEXT.md § Claude's Discretion defers the algorithm choice; RESEARCH §3 + § "State of the Art" recommends pairwise tree reduction with chunk size N=128 (sources: orlp.net/blog/taming-float-sums; rust-ndarray PR #577; Wikipedia Pairwise summation O(log N) error bound).

Planner picks one of three:
1. **Pairwise tree reduction (recommended)** — chunk size N=128; rayon-parallelizable yet still bit-deterministic when the chunk boundaries are content-defined (split index = N, 2N, 3N, ...) rather than thread-defined.
2. **Kahan-Babuska compensated sum** — slower but bounded relative error regardless of chunk shape.
3. **Strict left-to-right** — bit-trivial determinism; loses parallelism.

Phase 1 success criterion 3 (ROADMAP.md line 41) is the hard contract: `RAYON_NUM_THREADS=1` and `RAYON_NUM_THREADS=8` MUST produce bit-identical results on the same input vector. Pairwise + content-defined chunking + `cubecl-reduce`'s ordered-reduce variant per `docs/manual/Cubecl/cubecl_reduce_sum.md` is the recommended composition.

---

## Shared Patterns

### `tracing` Logging Convention (FOUND-09, ALG-08)

**Source:** `~/Documents/workspace/xcfun_rs/Cargo.toml` line 44 (`tracing = { version = "=0.1.44", default-features = false }`)
**Apply to:** Every `pyscf-{algebra,runtime,core}` module that does I/O or backend selection.

```rust
use tracing::{info, warn};

// FOUND-09 + ALG-08 mandatory observability lines.
tracing::info!("pyscf-runtime: probe cuda — available; selecting");
tracing::warn!(env = %raw_env_value, "PYSCF_BACKEND unrecognised; falling back to Cpu");
```
Phase 3 (BIND-02) extends this to a `tracing-subscriber` filter wired to `mol.verbose ∈ {0..9}` (deferred per CONTEXT § Deferred Ideas).

---

### `OnceLock<Option<Client>>` Probe-Cache Pattern (D-10)

**Source:** `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/src/runtime/cuda.rs` lines 49-105 (also `wgpu.rs` lines 49-87, `hip.rs` lines 47-85 — same shape three times)
**Apply to:** `pyscf-runtime/src/probe/{cuda,wgpu,rocm}.rs`

```rust
static CLIENT: OnceLock<Option<XClient>> = OnceLock::new();

pub fn x_available() -> bool {
    CLIENT.get_or_init(|| {
        let init = std::panic::catch_unwind(|| {
            let device = XDevice::default();
            XRuntime::client(&device)
        });
        match init {
            Ok(client) if client.properties().supports_type(ElemType::Float(FloatKind::F64)) => Some(client),
            _ => None,
        }
    }).is_some()
}
```
Two load-bearing properties: **(a)** `Option<XClient>` so a negative probe caches; **(b)** `catch_unwind` per FOUND-07 Pitfall 14.

---

### Panic Policy (FOUND-07)

**Source:** xcfun-gpu `runtime/{cuda,wgpu,hip}.rs` (every probe wraps init in `catch_unwind`)
**Apply to:** Every `extern "C"` callback (Phase 3+); every probe site (Phase 1).

```rust
let init = std::panic::catch_unwind(|| { /* potentially-panicking dynamic load */ });
let client = init.ok()?;
```

In `[profile.release]` and `[profile.release-oracle]`, `panic = "abort"` per CONTEXT § Claude's Discretion §6 (no sibling precedent — xcfun-gpu's `panic = "abort"` was a Phase 1 add per CONTEXT.md research note `panic = "unwind"` → `panic = "abort"` row of "State of the Art").

---

### Workspace `version.workspace = true` Inheritance

**Source:** `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/Cargo.toml` lines 3-5
**Apply to:** Every pyscf-rs crate `[package]` block.

```toml
[package]
name = "pyscf-{name}"
version.workspace      = true
edition.workspace      = true
rust-version.workspace = true
license = "Apache-2.0"
```

---

### Re-export Module Shape (Public Surface Discipline)

**Source:** `~/Documents/workspace/cintx/crates/cintx-core/src/lib.rs` lines 7-21 + `~/Documents/workspace/cintx/crates/cintx-runtime/src/lib.rs` lines 1-26
**Apply to:** Every non-stub `pyscf-*/src/lib.rs` (core, runtime, algebra).

```rust
//! Top-level docstring.
pub mod foo;
pub mod bar;
pub use foo::{Foo, FooError};
pub use bar::{Bar, BarOptions};
```
One `pub mod` line per submodule, then one `pub use` line per public type. No glob re-exports — each public name is greppable.

---

## No Analog Found

Files with no close match in any sibling repo. Planner uses RESEARCH.md recommendations + CONTEXT.md decisions in lieu of a code template:

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/pyscf-core/src/lib.rs` (`Mole`/`Density`/`Energy`/traits **bodies**) | public surface | static-types | PySCF-domain-specific; no sibling has analogous chemistry types |
| `crates/pyscf-runtime/src/workspace_pool.rs` | runtime utility | static-config | `WorkspacePool` shape open per CONTEXT § Claude's Discretion §4; cintx's `ChunkPlanner` is shape-different |
| `crates/pyscf-{stub-crates}/src/lib.rs` (11 files) | stub | none | Phase 1 stub convention is greenfield |
| `Cargo.toml` `[patch.crates-io]` block (sibling-crate sourcing D-12/D-13) | workspace config | static-config | Siblings ARE the patched crates; no precedent |
| `Cargo.toml` `[profile.release-oracle]` block | build config | static-config | Named oracle profile is new (FOUND-05) |
| `deny.toml` | lint enforcement | static-config | No sibling has `deny.toml` |
| `.github/workflows/nightly-cross-crate.yml` cross-crate matrix logic | CI | scheduled | Cross-crate update + rebuild + test logic is new (D-14) |
| `CONTRIBUTING.md` | docs | static | No sibling has CONTRIBUTING.md |
| `crates/pyscf-algebra/src/oracle.rs` (`oracle_sum`/`oracle_dot`/`oracle_einsum`) | public surface | transform | No sibling ordered-reduction primitive (FOUND-06; CONTEXT § Claude's Discretion §1) |
| `crates/pyscf-algebra/src/buffer.rs` (`BufferId`/`Tensor` opaque boundary D-05) | runtime gateway | request-response | Siblings expose `ComputeClient<R>` directly through enum tuple fields; pyscf-rs hides it |

For each of these, planner cites **RESEARCH.md** sections (and `docs/manual/Cubecl/*.md` for the kernel-launch shape) as the design source.

---

## Metadata

**Analog search scope:**
- `~/Documents/workspace/cintx/Cargo.toml`, `crates/cintx-cubecl/{Cargo.toml,src/lib.rs,src/backend/}`, `crates/cintx-runtime/{Cargo.toml,src/lib.rs,src/options.rs,src/workspace.rs}`, `crates/cintx-core/src/lib.rs`, `.github/workflows/`
- `~/Documents/workspace/xcfun_rs/Cargo.toml`, `crates/xcfun-gpu/{Cargo.toml,src/lib.rs,src/backend.rs,src/auto_backend.rs,src/batch.rs,src/pool.rs,src/runtime/{cpu,cuda,hip,wgpu,mod}.rs}`, `crates/xcfun-core/src/lib.rs`, `xtask/{Cargo.toml,src/bin/{check_no_fma,check_boundaries,check_cubecl_pin}.rs}`, `.cargo/config.toml`, `.github/workflows/ci.yml`
- `~/Documents/workspace/libxc_rs/Cargo.toml` (release profile shape)

**Files scanned:** 25 distinct sibling files (verified read; no re-reads).

**Sibling-remote verification:** D-13 GitHub remotes `BectorVoom/{cintx,libxc_rs,xcfun_rs}` confirmed public + non-empty per RESEARCH.md `<assumptions_log>` row "Sibling-remote availability (D-13): HIGH — verified via `gh api` 2026-05-10".

**Pattern extraction date:** 2026-05-10
**Valid until:** 2026-06-09 (matches RESEARCH.md research validity window).
