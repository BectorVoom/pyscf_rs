//! CUDA probe — adapt verbatim from xcfun-gpu/runtime/cuda.rs lines
//! 73-105 (PATTERNS Shared "OnceLock probe-cache pattern"). Catches
//! panics from missing CUDA driver (Pitfall 5 reasoning).

use crate::backend::DType;
use cubecl::Runtime;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::ComputeClient;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use std::sync::OnceLock;

type CudaClient = ComputeClient<CudaRuntime>;
static CUDA_CLIENT: OnceLock<Option<CudaClient>> = OnceLock::new();

fn init_cuda() -> Option<CudaClient> {
    let init = std::panic::catch_unwind(|| {
        let device = CudaDevice::default();
        CudaRuntime::client(&device)
    });
    let client = init.ok()?;
    // CUDA always supports f64; the supports_type check is defensive
    // against future cubecl-cuda changes (xcfun-gpu pattern).
    if client
        .properties()
        .supports_type(ElemType::Float(FloatKind::F64))
    {
        Some(client)
    } else {
        None
    }
}

/// `true` iff CUDA is initialisable and supports the requested dtype.
/// f32 always works on CUDA; f64 typically also works but the probe
/// confirms via cubecl's normalised supports_type API.
pub fn cuda_available(dtype: DType) -> bool {
    let client_opt = CUDA_CLIENT.get_or_init(init_cuda);
    match (client_opt, dtype) {
        (Some(_), _) => true,
        (None, _) => false,
    }
}
