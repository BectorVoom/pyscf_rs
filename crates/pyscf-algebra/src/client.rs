//! AlgebraClient — D-04 enum-of-clients dispatch shape.

use pyscf_runtime::{BackendKind, DType};

// NOTE: `ComputeClient<R>` does not implement `Debug` in cubecl 0.10.0
// (the inner Server/Channel types are not Debug). We provide a manual
// `Debug` impl that prints only the backend kind — sufficient for
// tracing diagnostics; printing channel internals would be noisy and
// non-portable across cubecl backends.
pub enum AlgebraClient {
    Cpu(cubecl::client::ComputeClient<cubecl_cpu::CpuRuntime>),
    #[cfg(feature = "cuda")]
    Cuda(cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>),
    #[cfg(feature = "wgpu")]
    Wgpu(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(cubecl::client::ComputeClient<cubecl_hip::HipRuntime>),
}

impl std::fmt::Debug for AlgebraClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlgebraClient")
            .field("kind", &self.kind().name())
            .finish()
    }
}

impl AlgebraClient {
    pub fn kind(&self) -> BackendKind {
        match self {
            Self::Cpu(_) => BackendKind::Cpu,
            #[cfg(feature = "cuda")] Self::Cuda(_) => BackendKind::Cuda,
            #[cfg(feature = "wgpu")] Self::Wgpu(_) => BackendKind::Wgpu,
            #[cfg(feature = "rocm")] Self::Rocm(_) => BackendKind::Rocm,
        }
    }

    /// ALG-08 + D-08 mandatory observability line.
    pub fn log_resolution(&self, raw_env: Option<&str>, dtype: DType) {
        tracing::info!(
            "pyscf-algebra: backend={} (env={}, dtype={})",
            self.kind().name(),
            raw_env.unwrap_or("unset"),
            dtype.name(),
        );
    }
}
