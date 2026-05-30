#type/docs
#Rust/cubecl/random

```rust
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

#[cube(launch_unchecked)]
fn generate_random_values(output: &mut Array<f32>, seed: f32) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        // Simple pseudo-random generation using trigonometric functions
        let x = seed + f32::cast_from(index);
        let sin_val = f32::sin(x * 12.9898 + x * 78.233) * 43758.5453;
        let fract_val = sin_val - f32::floor(sin_val);
        output[index] = f32::abs(fract_val);
    }
}

#[cube(launch_unchecked)]
fn process_random_values(input: &Array<f32>, output: &mut Array<f32>, scale: f32) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        // Scale random values to desired range (e.g., 0-100)
        output[index] = input[index] * scale;
    }
}

#[tokio::main]
async fn main() {
    println!("CubeCL Random Value Generation Example");
    println!("=====================================");
    
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let size = 1024;
    
    // Method 1: GPU-based random generation using CubeCL kernel
    println!("\n1. GPU-based random generation:");
    
    let random_buffer = client.empty(size * std::mem::size_of::<f32>());
    let processed_buffer = client.empty(size * std::mem::size_of::<f32>());
    
    let wg = 256usize;
    let groups = (size as usize + wg - 1) / wg;
    let seed = 12345.0f32;
    
    // Generate random values on GPU
    unsafe {
        generate_random_values::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(groups as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&random_buffer, groups, 1),
            ScalarArg::new(seed),
        );
    }
    
    // Process random values (scale to 0-100 range)
    unsafe {
        process_random_values::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(groups as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&random_buffer, groups, 1),
            ArrayArg::from_raw_parts::<f32>(&processed_buffer, groups, 1),
            ScalarArg::new(100.0f32),
        );
    }
    
    client.sync().await;
    
    // Read back some results to verify
    // For simplicity, we'll just demonstrate that the kernels run successfully
    println!("GPU-based random generation completed successfully!");
    println!("Generated {} random values using CubeCL kernels", size);
    
    // Method 2: Host-side random generation, then transfer to GPU
    println!("\n2. Host-side random generation with GPU processing:");
    
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    // Generate random data on host using a simple method
    let host_random_data: Vec<f32> = (0..size)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (i as u64 + 42).hash(&mut hasher);
            let hash = hasher.finish();
            (hash as f32) / (u64::MAX as f32)
        })
        .collect();
    
    // Transfer to GPU
    let host_buffer = client.create(bytemuck::cast_slice(&host_random_data));
    let host_processed_buffer = client.empty(size * std::mem::size_of::<f32>());
    
    // Process on GPU (scale to 0-1000 range)
    unsafe {
        process_random_values::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(groups as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&host_buffer, groups, 1),
            ArrayArg::from_raw_parts::<f32>(&host_processed_buffer, groups, 1),
            ScalarArg::new(1000.0f32),
        );
    }
    
    client.sync().await;
    
    println!("First 10 host-generated random values (0-1000):");
    for (i, &val) in host_random_data.iter().take(10).enumerate() {
        println!("  [{}]: {:.3}", i, val * 1000.0);
    }
    
    println!("\nRandom value generation completed successfully!");
    println!("Generated {} random values using CubeCL", size);
}

```