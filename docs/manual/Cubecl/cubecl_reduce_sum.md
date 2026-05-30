```rust
use anyhow::{Result, anyhow};
use bytemuck::cast_slice;
use cubecl_common::bytes::Bytes;
use cubecl_core::frontend::TensorHandleRef;
use cubecl_cpu::{CpuDevice, CpuRuntime, RuntimeOptions};
use cubecl_ir::{ElemType, FloatKind, StorageType};
use cubecl_reduce::{ReduceDtypes, instructions::Sum, reduce};
use cubecl_runtime::client::ComputeClient;

fn main() -> Result<()> {
    // --- 1) Set up WGPU runtime and compute client -------------------------------------------
    // Use the pure CPU runtime so it works without a GPU.
    let device = CpuDevice::default();
    let _options = RuntimeOptions::default();
    let client: ComputeClient<_> = ComputeClient::load(&device);

    // --- 2) Host data -------------------------------------------------------------------------
    let host_values: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let elem_size = core::mem::size_of::<f64>();

    // --- 3) Allocate tensors on the device ----------------------------------------------------
    // Upload the input vector.
    let input_shape = [host_values.len()];
    let output_shape = [1usize];

    let input_alloc = client.create_tensor(
        Bytes::from_elems(host_values.to_vec()),
        &input_shape,
        elem_size,
    );
    // Allocate a single-element output tensor to store the sum.
    let output_alloc = client.empty_tensor(&output_shape, elem_size);

    // Unsafe is required because we are providing raw shape/stride metadata.
    let input_handle = unsafe {
        TensorHandleRef::<CpuRuntime>::from_raw_parts(
            &input_alloc.handle,
            &input_alloc.strides,
            &input_shape,
            elem_size,
        )
    };
    let output_handle = unsafe {
        TensorHandleRef::<CpuRuntime>::from_raw_parts(
            &output_alloc.handle,
            &output_alloc.strides,
            &output_shape,
            elem_size,
        )
    };

    // --- 4) Run cubecl-reduce sum on the device -----------------------------------------------
    reduce::<CpuRuntime, Sum>(
        &client,
        input_handle,
        output_handle,
        0,    // axis to reduce (only axis 0 exists)
        None, // let CubeCL pick the best strategy
        (),   // Sum has a unit config
        ReduceDtypes {
            input: StorageType::Scalar(ElemType::Float(FloatKind::F64)),
            output: StorageType::Scalar(ElemType::Float(FloatKind::F64)),
            accumulation: StorageType::Scalar(ElemType::Float(FloatKind::F64)),
        },
    )
    .map_err(|e| anyhow!("cubecl-reduce failed: {e:?}"))?;

    // --- 5) Read back the result --------------------------------------------------------------
    let bytes = client.read_tensor(vec![output_alloc.handle.copy_descriptor(
        &output_shape,
        &output_alloc.strides,
        elem_size,
    )]);
    let mut result_host = [0.0f64; 1];
    result_host.copy_from_slice(cast_slice(&bytes[0]));

    let host_sum: f64 = host_values.iter().sum();
    println!("Host sum   : {}", host_sum);
    println!("Device sum : {}", result_host[0]);

    Ok(())
}

```