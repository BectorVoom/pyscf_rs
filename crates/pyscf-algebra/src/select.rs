//! select_backend() — env-driven resolver per D-07/D-08/D-09/D-10.

use crate::client::AlgebraClient;
use crate::error::AlgebraError;
use cubecl::Runtime;
// `BackendError` is only constructed inside the wgpu D-09 hard-error
// path; gate the import behind the wgpu feature to avoid an
// unused-import warning on cpu-only builds.
#[cfg(feature = "wgpu")]
use pyscf_runtime::BackendError;
use pyscf_runtime::{BackendKind, DType};

#[derive(Debug)]
pub struct BackendSelection {
    pub client: AlgebraClient,
    pub kind: BackendKind,
    pub raw_env: Option<String>,
    pub dtype: DType,
}

/// Resolve `PYSCF_BACKEND` + `PYSCF_DTYPE` into an `AlgebraClient`.
/// Behavior per CONTEXT D-07/D-08/D-09 + ROADMAP success criterion 6.
pub fn select_backend() -> Result<BackendSelection, AlgebraError> {
    let raw_env = std::env::var("PYSCF_BACKEND").ok();
    let dtype = DType::from_env();
    let normalised = raw_env.as_deref().unwrap_or("cpu").to_ascii_lowercase();

    let kind = if BackendKind::is_auto_token(&normalised) {
        auto_resolve(dtype)
    } else if let Some(k) = BackendKind::from_env_str(&normalised) {
        // Verify the requested backend's probe passes; D-09 hard-error
        // for explicit wgpu+f64+no-shader-f64.
        verify_explicit(k, dtype, &raw_env)?
    } else {
        tracing::warn!(
            env = %normalised,
            "PYSCF_BACKEND unrecognised; falling back to Cpu (recognised: cpu, cuda, wgpu, rocm, metal, auto)"
        );
        BackendKind::Cpu
    };

    let client = construct_client(kind)?;
    let sel = BackendSelection {
        client,
        kind,
        raw_env,
        dtype,
    };
    // ALG-08 mandatory log line.
    sel.client.log_resolution(sel.raw_env.as_deref(), dtype);
    Ok(sel)
}

/// D-07 priority chain: cuda → rocm → metal → wgpu → cpu. Per-probe
/// info! line for observability.
#[allow(unused_variables)] // dtype unused on cpu-only builds
fn auto_resolve(dtype: DType) -> BackendKind {
    #[cfg(feature = "cuda")]
    {
        tracing::info!("probe: cuda");
        if pyscf_runtime::probe::cuda::cuda_available(dtype) {
            tracing::info!("probe: cuda — available; selecting");
            return BackendKind::Cuda;
        }
        tracing::info!("probe: cuda — unavailable; skipping");
    }
    #[cfg(feature = "rocm")]
    {
        tracing::info!("probe: rocm");
        if pyscf_runtime::probe::hip::rocm_available(dtype) {
            tracing::info!("probe: rocm — available; selecting");
            return BackendKind::Rocm;
        }
        tracing::info!("probe: rocm — unavailable; skipping");
    }
    #[cfg(feature = "metal")]
    {
        tracing::info!("probe: metal");
        if pyscf_runtime::probe::wgpu::wgpu_available(dtype) {
            tracing::info!("probe: metal — available; selecting");
            return BackendKind::Metal;
        }
        tracing::info!("probe: metal — unavailable; skipping");
    }
    #[cfg(feature = "wgpu")]
    {
        tracing::info!("probe: wgpu");
        if pyscf_runtime::probe::wgpu::wgpu_available(dtype) {
            tracing::info!("probe: wgpu — available; selecting");
            return BackendKind::Wgpu;
        }
        if dtype == DType::F64 {
            tracing::info!("probe: wgpu — adapter lacks shader-f64; skipping (f64 requested)");
        } else {
            tracing::info!("probe: wgpu — unavailable; skipping");
        }
    }
    BackendKind::Cpu
}

/// D-09: explicit `PYSCF_BACKEND=wgpu` with `PYSCF_DTYPE=f64` AND adapter
/// without shader-f64 → hard error. Other explicit-but-unavailable
/// combinations log a warn and fall back to Cpu (per ALG-04).
#[allow(unused_variables)] // raw_env unused on cpu-only builds (no GPU arms)
fn verify_explicit(
    kind: BackendKind,
    dtype: DType,
    raw_env: &Option<String>,
) -> Result<BackendKind, AlgebraError> {
    match kind {
        BackendKind::Cpu => Ok(BackendKind::Cpu),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            if pyscf_runtime::probe::cuda::cuda_available(dtype) {
                Ok(kind)
            } else {
                tracing::warn!("PYSCF_BACKEND=cuda but probe failed; falling back to Cpu");
                Ok(BackendKind::Cpu)
            }
        }
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => {
            if pyscf_runtime::probe::wgpu::wgpu_available(dtype) {
                Ok(kind)
            } else if dtype == DType::F64 {
                // D-09 hard-error path: explicit wgpu+f64 with no shader-f64.
                Err(AlgebraError::Backend(BackendError::Unsatisfiable {
                    backend: "wgpu",
                    dtype: "f64",
                    reason: format!(
                        "PYSCF_BACKEND={} requested with f64, but adapter lacks shader-f64. \
                         Set PYSCF_DTYPE=f32 or PYSCF_BACKEND=cpu/auto.",
                        raw_env.as_deref().unwrap_or("wgpu")
                    ),
                }))
            } else {
                tracing::warn!("PYSCF_BACKEND=wgpu but probe failed; falling back to Cpu");
                Ok(BackendKind::Cpu)
            }
        }
        #[cfg(feature = "rocm")]
        BackendKind::Rocm => {
            if pyscf_runtime::probe::hip::rocm_available(dtype) {
                Ok(kind)
            } else {
                tracing::warn!("PYSCF_BACKEND=rocm but probe failed; falling back to Cpu");
                Ok(BackendKind::Cpu)
            }
        }
        #[cfg(feature = "metal")]
        BackendKind::Metal => {
            if pyscf_runtime::probe::wgpu::wgpu_available(dtype) {
                Ok(kind)
            } else {
                tracing::warn!("PYSCF_BACKEND=metal but probe failed; falling back to Cpu");
                Ok(BackendKind::Cpu)
            }
        }
    }
}

/// Construct the AlgebraClient for the resolved BackendKind.
fn construct_client(kind: BackendKind) -> Result<AlgebraClient, AlgebraError> {
    match kind {
        BackendKind::Cpu => {
            let device = cubecl_cpu::CpuDevice;
            Ok(AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&device)))
        }
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            let device = cubecl_cuda::CudaDevice::default();
            Ok(AlgebraClient::Cuda(cubecl_cuda::CudaRuntime::client(
                &device,
            )))
        }
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => {
            let device = cubecl_wgpu::WgpuDevice::default();
            Ok(AlgebraClient::Wgpu(cubecl_wgpu::WgpuRuntime::client(
                &device,
            )))
        }
        #[cfg(feature = "rocm")]
        BackendKind::Rocm => {
            let device = cubecl_hip::AmdDevice::default();
            Ok(AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(&device)))
        }
        #[cfg(feature = "metal")]
        BackendKind::Metal => {
            let device = cubecl_wgpu::WgpuDevice::default();
            Ok(AlgebraClient::Wgpu(cubecl_wgpu::WgpuRuntime::client(
                &device,
            )))
        }
    }
}
