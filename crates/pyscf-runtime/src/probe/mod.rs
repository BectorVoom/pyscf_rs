//! Per-backend availability probes. Each module exposes a single
//! `${name}_available(dtype: DType) -> bool` that:
//!   1. Constructs a cubecl client (potentially-panicking);
//!   2. Wraps the construction in `std::panic::catch_unwind` (FOUND-07
//!      + RESEARCH Pitfall 5 — wgpu can panic on missing libVulkan);
//!   3. Returns `false` if construction fails;
//!   4. Returns `false` if the dtype is F64 and the device lacks
//!      f64 support (D-09 + D-10);
//!   5. Caches the outcome via `OnceLock<Option<Client>>`
//!      (PATTERNS Shared "OnceLock probe-cache pattern").

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "wgpu")]
pub mod wgpu;
#[cfg(feature = "rocm")]
pub mod hip;
