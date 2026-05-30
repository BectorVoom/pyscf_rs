# CubeCL Algebra Operations Manual

This guide covers the algebra operations available in CubeCL kernels, from basic arithmetic to advanced mathematical functions. Like other CubeCL operations, these are designed to work within generic kernels using trait bounds.

## Trait Bounds for Algebra

To perform algebra operations in a generic kernel, you must bound your generic type with the appropriate trait:

*   **`Numeric`**: Provides basic arithmetic (`+`, `-`, `*`, `/`, `%`) and is supported by both integer and floating-point types.
*   **`Float`**: Extends `Numeric` with advanced mathematical functions like `sin`, `exp`, and `sqrt`. This trait is only supported by floating-point types (e.g., `f32`, `f64`).

## Basic Arithmetic

Basic operations are available for all types implementing `Numeric`.

| Operation | Syntax | Description |
| :--- | :--- | :--- |
| Addition | `a + b` | Sum of two numbers |
| Subtraction | `a - b` | Difference between two numbers |
| Multiplication | `a * b` | Product of two numbers |
| Division | `a / b` | Quotient of two numbers |
| Remainder | `a % b` | Modulo of two numbers |

## Advanced Mathematical Functions

These functions require the `Float` trait bound. In CubeCL, these are typically available as methods on the floating-point type.

| Function | Syntax | Description |
| :--- | :--- | :--- |
| Absolute | `x.abs()` | Absolute value of x |
| Square Root | `x.sqrt()` | Square root of x |
| Exponential | `x.exp()` | e raised to the power of x |
| Natural Log | `x.ln()` | Natural logarithm (base e) |
| Power | `x.powf(y)` | x raised to the power of y |
| Sine | `x.sin()` | Sine of x (radians) |
| Cosine | `x.cos()` | Cosine of x (radians) |
| Tangent | `x.tan()` | Tangent of x (radians) |

## Comparisons and Rounding

These operations help manage ranges and precision. They are available for `Float` types.

| Operation | Syntax | Description |
| :--- | :--- | :--- |
| Minimum | `x.min(y)` | Returns the smaller of x and y |
| Maximum | `x.max(y)` | Returns the larger of x and y |
| Clamp | `x.clamp(min, max)` | Constrains x between min and max |
| Floor | `x.floor()` | Largest integer ≤ x |
| Ceiling | `x.ceil()` | Smallest integer ≥ x |
| Round | `x.round()` | Rounds to the nearest integer |

## Working with Constants

In generic kernels, you cannot use literals like `2.0` directly if the type is generic. Instead, use these trait methods:

*   `N::from_int(i)`: Creates a constant from an integer (works for any `Numeric`).
*   `F::cast_from(f)`: Creates a constant from a literal (works for `Float`).

```rust
#[cube]
fn example<F: Float>(x: F) {
    let two = F::from_int(2);
    let pi = F::cast_from(3.14159);
    let result = x * pi + two;
}
```

## Complete Example: Algebra Kernel

The following example demonstrates a functional CubeCL program with a generic kernel performing various algebra operations.

### Prerequisites

Ensure your `Cargo.toml` includes:
```toml
[dependencies]
cubecl = { version = "0.10.0", features = ["cpu"] }
bytemuck = { version = "1.14", features = ["extern_crate_std"] }
```

### Code Sample

```rust
use cubecl::prelude::*;

#[cube(launch)]
fn algebra_kernel<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    let id = ABSOLUTE_POS;
    if id < input.len() {
        let x = input[id];
        
        // 1. Basic Arithmetic and Constants
        let pi = F::cast_from(3.14159265);
        let linear = x * F::from_int(2) + pi;
        
        // 2. Trigonometry and Exponentials
        let s = x.sin();
        let e = x.exp();
        
        // 3. Logarithms and Power
        let l = (x.abs() + F::from_int(1)).ln();
        let p = x.powf(F::from_int(2));
        
        // 4. Comparison and Rounding
        let m = x.min(F::from_int(10));
        let c = x.clamp(F::from_int(0), F::from_int(100));
        let r = x.round();
        
        // Combine all operations
        output[id] = (s + l + p + e).sqrt() + m - c + r + linear;
    }
}

fn main() {
    let device = cubecl::cpu::CpuDevice::default();
    let client = cubecl::cpu::CpuRuntime::client(&device);

    let input_data = vec![1.0f32, 2.5f32, 5.0f32, 10.0f32];
    let input_bytes = cubecl::bytes::Bytes::from_elems(input_data.clone());
    let input_handle = client.create(input_bytes);
    let output_handle = client.empty(input_data.len() * std::mem::size_of::<f32>());

    println!("Launching Algebra Kernel...");
    algebra_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim { x: input_data.len() as u32, y: 1, z: 1 },
        unsafe { ArrayArg::from_raw_parts(input_handle, input_data.len()) },
        unsafe { ArrayArg::from_raw_parts(output_handle.clone(), input_data.len()) },
    );

    let bytes = client.read(vec![output_handle]);
    let output: &[f32] = bytemuck::cast_slice(&bytes[0]);
    
    println!("Input:  {:?}", input_data);
    println!("Output: {:?}", output);
}
```

## Key Considerations

1.  **Trait Bounds**: Always use `Float` if you need more than basic arithmetic. Using `Numeric` will limit you to `+`, `-`, `*`, `/`, and `%`.
2.  **Generic Parameters**: When launching, ensure the generic type (e.g., `f32`) matches the data provided in the host buffers.
3.  **Performance**: Transcendental functions (`sin`, `exp`, etc.) are hardware-dependent and generally more expensive than basic arithmetic.
4.  **Hardware Support**: While `f32` is universal, `f64` support varies by GPU. `CpuRuntime` supports both.
