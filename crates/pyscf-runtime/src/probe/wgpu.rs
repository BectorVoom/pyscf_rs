//! wgpu probe — adapt verbatim from xcfun-gpu/runtime/wgpu.rs lines
//! 38-87 (PATTERNS row "select.rs"). The shader-f64 gate is the
//! load-bearing D-09 check.

use crate::backend::DType;
use cubecl::Runtime;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::ComputeClient;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use std::sync::OnceLock;

type WgpuClient = ComputeClient<WgpuRuntime>;
static WGPU_CLIENT: OnceLock<Option<WgpuClient>> = OnceLock::new();

fn init_wgpu() -> Option<WgpuClient> {
    let init = std::panic::catch_unwind(|| {
        let device = WgpuDevice::default();
        WgpuRuntime::client(&device)
    });
    init.ok()
}

/// `true` iff wgpu adapter exists AND supports the requested dtype.
/// Per D-09:
///   * dtype = F32 → returns `true` if wgpu adapter exists
///   * dtype = F64 → returns `true` only if adapter advertises shader-f64
///
/// Caller (Plan 04 select_backend) maps this to the hard-error vs
/// auto-skip rule from D-09.
pub fn wgpu_available(dtype: DType) -> bool {
    let client_opt = WGPU_CLIENT.get_or_init(init_wgpu);
    let Some(client) = client_opt.as_ref() else {
        return false;
    };
    match dtype {
        DType::F32 => true,
        DType::F64 => client
            .properties()
            .supports_type(ElemType::Float(FloatKind::F64)),
    }
}
