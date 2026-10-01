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
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => BackendKind::Cuda,
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => BackendKind::Wgpu,
            #[cfg(feature = "rocm")]
            Self::Rocm(_) => BackendKind::Rocm,
        }
    }

    /// Whether this backend has real hardware planes (a GPU-like runtime) —
    /// [`crate::launch::has_planes`] on whichever client this is. Callers
    /// outside this crate cannot expand [`crate::dispatch_backend!`] (its arms
    /// name the `cubecl_*` runtime crates, which only this crate depends on),
    /// so the probe lives here.
    pub fn has_planes(&self) -> bool {
        crate::dispatch_backend!(self, c, Rt, crate::launch::has_planes(c))
    }

    /// Block until every launch queued so far has executed.
    ///
    /// Launches are lazy on every backend, so a host span that only wraps
    /// the launch calls under-attributes the stage (the kernels then execute
    /// inside whichever later span performs the next read). A profiler that
    /// wants the execution inside its own span calls this at the span's end.
    /// It is a one-element read — `read` drains the queue and, on CUDA/HIP,
    /// synchronises the stream — so no async executor is needed.
    pub fn sync_device(&self) {
        crate::dispatch_backend!(self, c, Rt, {
            let probe = c.create_from_slice(&[0u8; 8]);
            let _ = c.read(vec![probe]);
        })
    }

    /// Ask the runtime to release pooled device memory it no longer needs
    /// (`ComputeClient::memory_cleanup`), logging usage before and after.
    ///
    /// CubeCL keeps freed buffers in a per-process pool, so a long multi-stage
    /// run (one-electron matrices, a small-basis SCF, then a large-basis SCF)
    /// otherwise carries every earlier stage's peak into the next one. Called
    /// between stages; the allocator decides what it can actually return.
    pub fn memory_cleanup(&self) {
        crate::dispatch_backend!(self, c, Rt, {
            let before = c.memory_usage().ok();
            c.memory_cleanup();
            let after = c.memory_usage().ok();
            tracing::info!(before = ?before, after = ?after, "pyscf-algebra: memory cleanup");
        })
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
