# CubeCL Basic Operations Manual

This manual provides an overview of the basic supported operations in CubeCL. Building on the concepts from the [Generics Manual](./Cubecl_generics.md), it demonstrates how to perform arithmetic and mathematical operations within generic kernels.

## Trait Bounds for Operations

In CubeCL, the operations you can perform on a generic type `N` are determined by its trait bounds:

1.  **`Numeric`**: The base trait for all numeric types (integers and floats). It supports basic arithmetic: `+`, `-`, `*`, `/`, and `%`.
2.  **`Float`**: Extends `Numeric` for floating-point types (like `f32` and `f64`). It adds advanced mathematical functions such as `sin`, `cos`, `exp`, `sqrt`, and more.

## Supported Operations Reference

### Arithmetic (Numeric + Float)

These operations work on any type that implements `Numeric`.

| Operation | Syntax |
| :--- | :--- |
| Addition | `a + b` |
| Subtraction | `a - b` |
| Multiplication | `a * b` |
| Division | `a / b` |
| Remainder | `a % b` |

### Math Functions (Float Only)

These functions require the `Float` trait bound.

| Function | Syntax | Description |
| :--- | :--- | :--- |
| Absolute | `x.abs()` | Absolute value |
| Square Root | `x.sqrt()` | Square root |
| Exponential | `x.exp()` | e^x |
| Natural Log | `x.ln()` | Natural logarithm |
| Power | `x.powf(y)` | x raised to power y |
| Sine | `x.sin()` | Sine (radians) |
| Cosine | `x.cos()` | Cosine (radians) |
| Tangent | `x.tan()` | Tangent (radians) |
| Hyperbolic Tangent | `x.tanh()` | Hyperbolic tangent |
| Atan2 | `y.atan2(x)` | Arc tangent of y/x |
| Error Function | `x.erf()` | Gaussian error function |

### Comparison and Rounding (Float)

| Operation | Syntax | Description |
| :--- | :--- | :--- |
| Minimum | `x.min(y)` | Smaller of x and y |
| Maximum | `x.max(y)` | Larger of x and y |
| Clamp | `x.clamp(min, max)` | Constrain x between min and max |
| Floor | `x.floor()` | Round down |
| Ceiling | `x.ceil()` | Round up |
| Round | `x.round()` | Round to nearest integer |

## Handling Constants and Literals

Because CubeCL kernels are generic, you cannot use raw literals like `2.0` or `10` directly with generic types. Instead, use these methods provided by the traits:

*   **`N::from_int(value: i64)`**: Creates a value of type `N` from an integer. Works for both `Numeric` and `Float`.
*   **`F::cast_from(value: f64)`**: Creates a value of type `F` from a float literal. Works only for `Float`.

```rust
#[cube]
fn example_kernel<F: Float>(x: F) {
    let two = F::from_int(2);
    let half = F::cast_from(0.5);
    let result = x * two + half;
}
```

## Comprehensive Code Sample

The following example demonstrates a generic kernel that applies a variety of operations to an input array. This code is fully functional and uses the `CpuRuntime` for easy verification.

### Prerequisites

Add these to your `Cargo.toml`:
```toml
[dependencies]
cubecl = { version = "0.10.0", features = ["cpu"] }
bytemuck = { version = "1.14", features = ["extern_crate_std"] }
```

### Verified Implementation

```rust
use cubecl::prelude::*;

#[cube(launch)]
pub fn basic_ops_kernel<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    let id = ABSOLUTE_POS;
    
    if id < input.len() {
        let x = input[id];
        
        // 1. Basic Arithmetic with constants
        let a = x * F::from_int(5) + F::cast_from(1.5);
        
        // 2. Trigonometry (sin, cos)
        let s = x.sin();
        let c = x.cos();
        
        // 3. Advanced Math (exp, sqrt, powf)
        let e = x.exp();
        let p = x.powf(F::from_int(2));
        
        // 4. Comparison and Rounding
        let m = x.max(F::from_int(0));
        let r = x.round();
        
        // Combine results for demonstration
        // Note: We use intermediate variables for clarity
        output[id] = (s + c + e).sqrt() + p - m + r + a;
    }
}

fn main() {
    // 1. Setup Device and Client
    let device = cubecl::cpu::CpuDevice::default();
    let client = cubecl::cpu::CpuRuntime::client(&device);

    // 2. Prepare Data
    let input_data = vec![0.0f32, 1.0f32, 2.0f32, 3.14159f32];
    let input_bytes = cubecl::bytes::Bytes::from_elems(input_data.clone());
    let input_handle = client.create(input_bytes);
    let output_handle = client.empty(input_data.len() * std::mem::size_of::<f32>());

    // 3. Launch Kernel
    println!("Launching Basic Operations Kernel...");
    basic_ops_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim { x: input_data.len() as u32, y: 1, z: 1 },
        unsafe { ArrayArg::from_raw_parts(input_handle, input_data.len()) },
        unsafe { ArrayArg::from_raw_parts(output_handle.clone(), input_data.len()) },
    );

    // 4. Read and Verify Results
    let bytes = client.read(vec![output_handle]);
    let output: &[f32] = bytemuck::cast_slice(&bytes[0]);
    
    println!("Input:  {:?}", input_data);
    println!("Output: {:?}", output);
}
```

## Key Considerations

1.  **Precision**: When using `f32`, some operations like `sin` or `exp` might have slight precision differences across different hardware backends.
2.  **Performance**: Math functions like `powf` and `exp` are more computationally expensive than basic additions or multiplications.
3.  **Generic Reuse**: By using `F: Float`, this same kernel can be launched with `f64` on supported hardware simply by changing the type parameter in `basic_ops_kernel::launch::<f64, ...>`.
