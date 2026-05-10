//! ROCm/HIP probe — adapt verbatim from xcfun-gpu/runtime/hip.rs lines
//! 69-85 (PATTERNS Shared OnceLock pattern).

use crate::backend::DType;
use cubecl::Runtime;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::ComputeClient;
use cubecl_hip::{AmdDevice, HipRuntime};
use std::sync::OnceLock;

type HipClient = ComputeClient<HipRuntime>;
static HIP_CLIENT: OnceLock<Option<HipClient>> = OnceLock::new();

fn init_hip() -> Option<HipClient> {
    let init = std::panic::catch_unwind(|| {
        let device = AmdDevice::default();
        HipRuntime::client(&device)
    });
    let client = init.ok()?;
    if client
        .properties()
        .supports_type(ElemType::Float(FloatKind::F64))
    {
        Some(client)
    } else {
        None
    }
}

pub fn rocm_available(dtype: DType) -> bool {
    let client_opt = HIP_CLIENT.get_or_init(init_hip);
    match (client_opt, dtype) {
        (Some(_), _) => true,
        (None, _) => false,
    }
}
