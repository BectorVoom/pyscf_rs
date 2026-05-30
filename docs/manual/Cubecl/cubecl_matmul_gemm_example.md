
```rust
use anyhow::{anyhow, Result};
use bytemuck::cast_slice;
use cubecl_matmul::{launch, MatmulInputHandle, Strategy};
use cubecl_runtime::client::ComputeClient;
use cubecl_std::tensor::TensorHandle;
use cubecl_wgpu::{init_device, init_setup, AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime};

fn main() -> Result<()> {
    // Pick the default device and set up the WGPU runtime.
    let device = WgpuDevice::DefaultDevice;
    let options = RuntimeOptions::default();

    // Initialize setup and device; this registers the compute client internally.
    let setup = init_setup::<AutoGraphicsApi>(&device, options);
    let device = init_device(setup, RuntimeOptions::default());

    // Load the compute client for the device we just initialized.
    let client: ComputeClient<_> = ComputeClient::load(&device);

    // Host-side matrices (row-major).
    let a: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3
    let b: [f32; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3 x 2

    let elem_size = core::mem::size_of::<f32>();

    // Upload A and B to device memory.
    let alloc_a = client.create_tensor(cast_slice(&a), &[2, 3], elem_size);
    let alloc_b = client.create_tensor(cast_slice(&b), &[3, 2], elem_size);

    // Allocate output C (2 x 2).
    let alloc_c = client.empty_tensor(&[2, 2], elem_size);

    // Wrap allocations into TensorHandles expected by cubecl-matmul.
    let lhs = MatmulInputHandle::Normal(TensorHandle::<WgpuRuntime, f32>::new(
        alloc_a.handle,
        vec![2, 3],
        alloc_a.strides.clone(),
    ));
    let rhs = MatmulInputHandle::Normal(TensorHandle::<WgpuRuntime, f32>::new(
        alloc_b.handle,
        vec![3, 2],
        alloc_b.strides.clone(),
    ));
    let out = TensorHandle::<WgpuRuntime, f32>::new(
        alloc_c.handle,
        vec![2, 2],
        alloc_c.strides.clone(),
    );

    // Launch GEMM with an automatic strategy selection.
    launch::<WgpuRuntime, f32>(&Strategy::Auto, &client, lhs, rhs, out.clone())
        .map_err(|e| anyhow!("{e:?}"))?;

    // Read back the result.
    let bytes = client.read_tensor(vec![out.as_copy_descriptor()]);
    let c_data: Vec<f32> = cast_slice::<u8, f32>(&bytes[0]).to_vec();

    println!("C (2x2) =");
    for row in 0..2 {
        let start = row * 2;
        let end = start + 2;
        println!("  {:?}", &c_data[start..end]);
    }

    Ok(())
}

```