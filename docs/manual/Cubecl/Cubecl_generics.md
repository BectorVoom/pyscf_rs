# CubeCL Generics Manual

This guide explains how to use generics in CubeCL kernels to create flexible and reusable compute operations. Generics allow you to write a single kernel that can operate on different numeric types, such as `f32` and `f64`, depending on your precision requirements.

## Defining a Generic Kernel

To define a generic kernel in CubeCL, you use standard Rust generic syntax combined with the `#[cube]` attribute. The generic type must be bounded by the `Numeric` trait (or `Float` if you specifically need floating-point operations like `sin` or `exp`).

```rust
use cubecl::prelude::*;

#[cube(launch)]
fn generic_add_kernel<N: Numeric>(input: &Array<N>, output: &mut Array<N>) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        // Basic arithmetic works across all Numeric types
        output[tid] = input[tid] + input[tid];
    }
}
```

## Launching a Generic Kernel

When you use `#[cube(launch)]` on a generic function, the generated `launch` function also becomes generic. The order of generic parameters in the generated `launch` function is:
1. The generic parameters defined on the kernel function (e.g., `N`).
2. The `Runtime` parameter (e.g., `R`).

Example of launching with `f32` and `f64`:

```rust
// Launching with f32
generic_add_kernel::launch::<f32, Runtime>(...);

// Launching with f64
generic_add_kernel::launch::<f64, Runtime>(...);
```

## Complete Example: Switching Between f32 and f64

The following example demonstrates a fully functional program that defines a generic kernel and runs it for both `f32` and `f64` using the `CpuRuntime`.

### Prerequisites

Ensure your `Cargo.toml` has the necessary dependencies:

```toml
[dependencies]
cubecl = { version = "0.10.0", features = ["cpu"] }
bytemuck = { version = "1.14", features = ["extern_crate_std"] }
```

### Code Sample

```rust
use cubecl::prelude::*;

/// A generic kernel that doubles each element in an array.
/// The trait bound `N: Numeric` allows it to work with f32, f64, i32, etc.
#[cube(launch)]
fn double_elements_kernel<N: Numeric>(input: &Array<N>, output: &mut Array<N>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * N::from_int(2);
    }
}

/// A helper function to run the kernel with a specific type N.
fn run_with_type<N, R>(device: &R::Device) 
where 
    N: Numeric + CubeElement + bytemuck::Pod,
    R: Runtime 
{
    let client = R::client(device);

    // Prepare host data
    let input_host = vec![N::from_int(1), N::from_int(2), N::from_int(3), N::from_int(4)];
    
    // Create device buffers
    // Note: Use Bytes::from_elems for type-safe data transfer
    let input_bytes = cubecl::bytes::Bytes::from_elems(input_host);
    let input_handle = client.create(input_bytes);
    let output_handle = client.empty(4 * std::mem::size_of::<N>());

    // Launch the kernel
    // The generic parameters for launch are <N, R>
    double_elements_kernel::launch::<N, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim { x: 4, y: 1, z: 1 },
        unsafe { ArrayArg::from_raw_parts(input_handle, 4) },
        unsafe { ArrayArg::from_raw_parts(output_handle.clone(), 4) },
    );

    // Read back and verify results
    let bytes = client.read(vec![output_handle]);
    let output_data: &[N] = bytemuck::cast_slice(&bytes[0]);
    
    println!("Executed with type: {}", std::any::type_name::<N>());
    println!("Results: Length {}", output_data.len());
}

fn main() {
    // Use CpuRuntime for easy verification without specific hardware
    let device = cubecl::cpu::CpuDevice::default();

    println!("Starting CubeCL Generic Demo...");

    // Demonstrate switching between f32 and f64
    run_with_type::<f32, cubecl::cpu::CpuRuntime>(&device);
    run_with_type::<f64, cubecl::cpu::CpuRuntime>(&device);

    println!("Demo completed successfully.");
}
```

## Key Considerations

1.  **Trait Bounds**: 
    -   `Numeric`: Required for basic arithmetic (+, -, *, /) and integer conversions in kernels.
    -   `Float`: Required for advanced mathematical functions (sin, exp, etc.).
    -   `CubeElement`: Required for types that are stored in `Array` or `Tensor`.
    -   `bytemuck::Pod`: Required for host-to-device data transfers using `bytemuck`.

2.  **Runtime Support**: While `f32` is universally supported, some hardware backends (like certain mobile GPUs) might have limited or no support for `f64`. `CpuRuntime` and most modern Desktop GPUs support `f64`.

3.  **Performance**: Using `f64` instead of `f32` doubles the memory bandwidth requirements and often significantly reduces computation speed on most consumer GPUs. Only use `f64` when high precision is strictly necessary.
