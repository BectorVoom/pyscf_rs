//! AlgebraClient — D-04 enum-of-clients dispatch shape.

use pyscf_runtime::{BackendKind, DType};

#[derive(Debug)]
pub enum AlgebraClient {
    Cpu(cubecl::client::ComputeClient<cubecl_cpu::CpuRuntime>),
    #[cfg(feature = "cuda")]
    Cuda(cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>),
    #[cfg(feature = "wgpu")]
    Wgpu(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(cubecl::client::ComputeClient<cubecl_hip::HipRuntime>),
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
