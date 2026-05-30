# CubeCL Conditionals Manual

This guide explains how to use `if`, `else if`, and `else` statements in CubeCL kernels. Conditional branching allows you to implement logic that depends on runtime values, such as boundary checks or value-based thresholding.

## Basic Usage

Inside a `#[cube]` kernel, you can use standard Rust `if` syntax. For generic kernels, the generic type must implement the `Numeric` trait to support comparison operators (`<`, `>`, `<=`, `>=`, `==`, `!=`).

```rust
#[cube]
fn example_kernel<N: Numeric>(val: N) {
    if val > N::from_int(0) {
        // logic for positive values
    } else if val < N::from_int(0) {
        // logic for negative values
    } else {
        // logic for zero
    }
}
```

## Key Difference: Avoid If Expressions

In standard Rust, `if` is an expression that can return a value. However, in CubeCL kernels, using `if` as an expression (e.g., `let x = if cond { a } else { b };`) can sometimes lead to IR translation errors.

**Best Practice**: Always use `if` as a statement and assign values to mutable variables initialized outside the `if` block.

### Problematic Pattern
```rust
// This might fail to compile or lower correctly
let result = if val > threshold { a } else { b };
```

### Recommended Pattern
```rust
// This is robust and recommended for CubeCL
let mut result = b; // Initialize with default
if val > threshold {
    result = a;
}
```

## Complete Example: Thresholding Kernel

The following example demonstrates a generic kernel that classifies input values into three categories (-1, 0, 1) based on two thresholds. This example is fully functional and uses the `CpuRuntime`.

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

/// A generic kernel demonstrating if, else if, and else.
/// It outputs:
/// - -1 if val < threshold_low
/// -  1 if val > threshold_high
/// -  0 otherwise
#[cube(launch)]
fn conditional_kernel<N: Numeric + CubeElement>(
    input: &Array<N>, 
    output: &mut Array<N>, 
    threshold_low: N, 
    threshold_high: N
) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        let val = input[tid];
        let mut result = N::from_int(0);

        if val < threshold_low {
            result = N::from_int(-1);
        } else if val > threshold_high {
            result = N::from_int(1);
        } else {
            result = N::from_int(0);
        }

        output[tid] = result;
    }
}

fn run_demo() {
    // Use CpuRuntime for easy verification
    let device = cubecl::cpu::CpuDevice::default();
    let client = cubecl::cpu::CpuRuntime::client(&device);

    let input_host = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
    let threshold_low = 2.5f32;
    let threshold_high = 4.5f32;
    
    let input_bytes = cubecl::bytes::Bytes::from_elems(input_host.clone());
    let input_handle = client.create(input_bytes);
    let output_handle = client.empty(input_host.len() * std::mem::size_of::<f32>());

    println!("Launching Conditional Kernel...");
    
    // Launch the kernel. 
    // In CubeCL 0.10.0, scalars are passed directly to the launch function.
    conditional_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim { x: input_host.len() as u32, y: 1, z: 1 },
        unsafe { ArrayArg::from_raw_parts(input_handle, input_host.len()) },
        unsafe { ArrayArg::from_raw_parts(output_handle.clone(), input_host.len()) },
        threshold_low,
        threshold_high,
    );

    let bytes = client.read(vec![output_handle]);
    let output_data: &[f32] = bytemuck::cast_slice(&bytes[0]);
    
    println!("Input:    {:?}", input_host);
    println!("Thresholds: low={}, high={}", threshold_low, threshold_high);
    println!("Results:  {:?}", output_data);
}

fn main() {
    run_demo();
}
```

## Key Considerations

1.  **Trait Bounds**: Use `N: Numeric` for basic comparisons. If you need more complex logic involving floating-point specific functions (like `is_nan()`), use `F: Float`.
2.  **Mutable Variables**: Use `let mut` to handle conditional assignments. This ensures the CubeCL compiler can correctly track variable state across branches.
3.  **Scalar Arguments**: When passing scalars to a kernel, ensure the type matches the generic type used in the launch call (e.g., `f32`).
4.  **Performance**: Excessive branching can lead to "thread divergence" on GPUs, which may impact performance. For simple cases, consider using functions like `clamp`, `min`, or `max` if they achieve the same result.
