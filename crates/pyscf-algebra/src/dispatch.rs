//! Backend dispatch — the SINGLE cfg-gated `AlgebraClient` fanout (ALG-06).
//!
//! quick-260530-l29: every engine in this crate used to hand-write the same
//! cfg-gated 4-arm `match client { AlgebraClient::Cpu(c) => …, #[cfg] Cuda, …}`
//! block — 16 copies across 8 files — differing ONLY in the cubecl `Runtime`
//! type token each arm passes (`cubecl_cpu::CpuRuntime`,
//! `cubecl_cuda::CudaRuntime`, `cubecl_wgpu::WgpuRuntime`,
//! `cubecl_hip::HipRuntime`). That meant adding or removing a backend — or
//! fixing a per-backend dispatch bug — touched sixteen sites.
//!
//! [`dispatch_backend!`] collapses all of them into ONE macro. It is the only
//! place the cfg-gated arms live, so:
//!   * the ALG-06 wall is centralized — the cubecl `Runtime` type is selected
//!     ONLY inside this expansion and never escapes into a public signature;
//!   * adding a backend means adding ONE arm here (plus the matching variant in
//!     [`crate::client::AlgebraClient`]), not editing every engine;
//!   * the per-arm `#[cfg(feature = "…")]` attributes exist in exactly two
//!     auditable places (this macro and the enum it matches), not scattered.

/// Fan out a single body over every backend arm of an [`crate::AlgebraClient`].
///
/// This is the SINGLE cfg-gated `AlgebraClient` fanout in the crate — it
/// replaces 16 hand-written `match client { … }` blocks. It expands to a
/// `match` EXPRESSION, so it serves both value-returning call sites
/// (`let out = dispatch_backend!(…);`) and statement call sites
/// (`dispatch_backend!(…);`) identically.
///
/// # Parameters
/// * `$client` — the `&AlgebraClient` to match on.
/// * `$c` — the ident bound to the matched arm's inner `ComputeClient`.
/// * `$rt` — the ident the body uses to name that arm's concrete cubecl
///   `Runtime` type. Each arm opens with a local `type $rt = <concrete>;`
///   alias, so a body like `launch_gemm::<$rt, F>($c, …)` resolves `$rt` to the
///   right `Runtime` per arm. A body that never names `$rt` (e.g.
///   `$c.create(…)`) is equally valid — the alias is then simply unused, which
///   does not warn.
///
/// # ALG-06
/// The cubecl `Runtime` type appears ONLY in the per-arm `type $rt = …;`
/// aliases inside this expansion; it never leaks into a signature. Adding a
/// backend = adding one arm here.
///
/// # Cross-crate export (quick-260530-ljv)
/// `#[macro_export]` hoists this macro to the crate root
/// (`pyscf_algebra::dispatch_backend`) so downstream wall-allowlisted crates
/// (pyscf-kernels) can fan a launch out over every backend without re-deriving
/// the cfg-gated match. The macro body uses `$crate::AlgebraClient` and bare
/// runtime paths (`cubecl_cpu::CpuRuntime`, …) that resolve in the CALLER's
/// namespace — every downstream caller must carry the cfg-aligned cubecl-*
/// optional deps + matching features (pyscf-kernels already does). In-crate
/// call sites keep resolving it unqualified via the `#[macro_use] mod dispatch;`
/// in `lib.rs` (macro_use + macro_export coexist).
#[macro_export]
macro_rules! dispatch_backend {
    ($client:expr, $c:ident, $rt:ident, $body:expr) => {
        match $client {
            $crate::AlgebraClient::Cpu($c) => {
                #[allow(dead_code)]
                type $rt = cubecl_cpu::CpuRuntime;
                $body
            }
            #[cfg(feature = "cuda")]
            $crate::AlgebraClient::Cuda($c) => {
                #[allow(dead_code)]
                type $rt = cubecl_cuda::CudaRuntime;
                $body
            }
            #[cfg(feature = "wgpu")]
            $crate::AlgebraClient::Wgpu($c) => {
                #[allow(dead_code)]
                type $rt = cubecl_wgpu::WgpuRuntime;
                $body
            }
            #[cfg(feature = "rocm")]
            $crate::AlgebraClient::Rocm($c) => {
                #[allow(dead_code)]
                type $rt = cubecl_hip::HipRuntime;
                $body
            }
        }
    };
}
