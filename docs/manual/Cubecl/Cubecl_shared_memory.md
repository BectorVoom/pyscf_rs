#type/docs
#Rust/cubecl/vector
```rust
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

// CubeCL kernel demonstrating shared memory array usage
#[cube(launch)]
fn array_multiply_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let local_id = UNIT_POS;
    let global_id = ABSOLUTE_POS;
    
    // Declare shared memory array - this is the key demonstration
    // Size should match the workgroup size (CubeDim)
    let mut shared_data = SharedMemory::<f32>::new(4);
    
    // Load data from global memory into shared memory
    if local_id < input.len() && global_id < input.len() {
        shared_data[local_id] = input[global_id];
    }
    
    // Synchronize threads to ensure all data is loaded into shared memory
    sync_units();
    
    // Perform computation using shared memory data
    if local_id < output.len() && global_id < output.len() {
        // Example computation: sum of current element and next element in shared memory
        let current_val = shared_data[local_id];
        let next_val = if local_id + 1 < 4 {
            shared_data[local_id + 1]
        } else {
            f32::new(0.0)
        };
        
        // Store result back to global memory
        output[global_id] = current_val + next_val;
    }
}

fn main() {
    // Initialize CubeCL environment with WGPU backend
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    // Create a fixed-size input array - this demonstrates array declaration and usage
    let input_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    // Convert array to bytes for buffer creation
    let input_bytes = bytemuck::cast_slice(&input_data);
    let output_size = input_data.len() * std::mem::size_of::<f32>();

    // Create input and output buffers
    let input_buffer = client.create(input_bytes);
    let output_buffer = client.empty(output_size);

    // Launch the kernel
    array_multiply_kernel::launch::<WgpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(4, 1, 1),
        unsafe { ArrayArg::from_raw_parts::<f32>(&input_buffer, input_data.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&output_buffer, input_data.len(), 1) },
    );

    // Read results back from GPU
    let output_bytes = client.read(vec![output_buffer.binding()]);
    let output_data: &[f32] = bytemuck::cast_slice(&output_bytes[0]);

    println!("Input array:  {:?}", input_data);
    println!("Output array: {:?}", output_data);
    println!("Array multiplication completed successfully!");
}


```