```markdown
Below is an error-resolution document describing why calling a “normal” Rust function from inside a `#[cube]` function fails in CubeCL, and how to resolve it.

---

# Error Resolution: Calling Normal Rust Functions from `#[cube]` Functions in CubeCL

## 1. Problem Description

### 1.1 Typical Error Message

When you call a regular Rust function from inside a `#[cube]` function, you may see an error such as:

```text
failed to resolve: function double_factorial_f64 is not a crate or module
function double_factorial_f64 is not a crate or module (rustc E0433)
```

Example code:

```rust
#[inline(always)]
fn double_factorial_f64(n: u32) -> f64 {
    if n == 0 {
        1.0
    } else if n == 1 {
        1.0
    } else if n == 2 {
        3.0
    } else if n == 3 {
        15.0
    } else if n == 4 {
        105.0
    } else {
        945.0
    }
}

#[cube]
#[inline(always)]
fn boys_asymptotic_f64(n: u32, x: f64) -> f64 {
    let coeff = double_factorial_f64(n); // <- error here

    let two_power = (1u64 << (n + 1)) as f64;
    let sqrt_x = f64::sqrt(x);

    let mut x_pow = 1.0;
    let mut m: u32 = 0;
    while m < n {
        x_pow *= x;
        m += 1;
    }

    coeff * SQRT_PI_F64 / (two_power * x_pow * sqrt_x)
}
```

The compiler complains that `double_factorial_f64` “is not a crate or module,” even though it is clearly defined as a normal Rust function in the same file.

---

## 2. Root Cause

### 2.1 How `#[cube]` Functions Are Transformed

CubeCL does not execute `#[cube]` functions as plain Rust at runtime. Instead, it:

1. Parses the `#[cube]` function.
    
2. Translates it into an internal IR.
    
3. Generates GPU kernels from that IR.
    

During this translation, **function calls inside a `#[cube]` function are not treated as normal Rust calls**. The macro expansion expects called functions to participate in CubeCL’s IR system (typically by being `#[cube]` functions themselves or by being provided as part of the CubeCL frontend).

When you call a regular Rust function like:

```rust
let coeff = double_factorial_f64(n);
```

inside a `#[cube]` function, CubeCL’s macro tries to resolve that call as if `double_factorial_f64` were a CubeCL “expandable” function or module. Since it is just a plain Rust function, the macro cannot resolve it correctly and the compiler effectively sees something like:

- “Trying to access `double_factorial_f64` as a module or crate, but it isn’t one.”
    

Hence the error:

```text
function double_factorial_f64 is not a crate or module (E0433)
```

### 2.2 General Rule

**Any function you call from inside a `#[cube]` function must be compatible with CubeCL’s IR transformation.** In practice, this means:

- It must itself be a `#[cube]` function (or a CubeCL-provided intrinsic).
    
- Or, its logic must be inlined directly into the `#[cube]` function.
    

Plain Rust helper functions are not automatically usable from GPU kernels.

---

## 3. Recommended Solutions

There are two main patterns to fix this issue.

### Solution 1: Make the Helper Function a `#[cube]` Function

If the helper function is purely numeric and should be usable in GPU kernels, the simplest fix is to mark it as `#[cube]` and ensure it uses only CubeCL-compatible types and operations.

**Before (error-prone):**

```rust
#[inline(always)]
fn double_factorial_f64(n: u32) -> f64 {
    if n == 0 {
        1.0
    } else if n == 1 {
        1.0
    } else if n == 2 {
        3.0
    } else if n == 3 {
        15.0
    } else if n == 4 {
        105.0
    } else {
        945.0
    }
}

#[cube]
#[inline(always)]
fn boys_asymptotic_f64(n: u32, x: f64) -> f64 {
    let coeff = double_factorial_f64(n); // ERROR
    // ...
}
```

**After (CubeCL-compatible):**

```rust
use cubecl::prelude::*;

#[cube]
#[inline(always)]
fn double_factorial_f64(n: u32) -> f64 {
    if n == 0 {
        1.0
    } else if n == 1 {
        1.0
    } else if n == 2 {
        3.0
    } else if n == 3 {
        15.0
    } else if n == 4 {
        105.0
    } else {
        945.0
    }
}

#[cube]
#[inline(always)]
fn boys_asymptotic_f64(n: u32, x: f64) -> f64 {
    let coeff = double_factorial_f64(n);

    // Prefer u32/i32 in kernels instead of u64
    let two_power = (1u32 << (n + 1)) as f64;

    let sqrt_x = f64::sqrt(x);

    let mut x_pow = 1.0;
    let mut m: u32 = 0;
    while m < n {
        x_pow *= x;
        m += 1;
    }

    coeff * SQRT_PI_F64 / (two_power * x_pow * sqrt_x)
}
```

Key points:

- Add `#[cube]` to the helper function.
    
- Ensure all types inside both functions are supported by CubeCL (e.g., `f32`/`f64`, `u32`/`i32`).
    
- Avoid unsupported types like `u64` inside `#[cube]` functions.
    

This approach allows the CubeCL macro to transform the helper function into its IR, just like the main kernel.

---

### Solution 2: Inline the Logic Into the `#[cube]` Function

If the helper function is simple and only used from a single kernel, it may be clearer to inline its logic directly into the `#[cube]` function, avoiding cross-function calls entirely.

**Refactored example:**

```rust
use cubecl::prelude::*;

#[cube]
#[inline(always)]
fn boys_asymptotic_f64(n: u32, x: f64) -> f64 {
    // Inline double_factorial_f64(n)
    let coeff = if n == 0 {
        1.0
    } else if n == 1 {
        1.0
    } else if n == 2 {
        3.0
    } else if n == 3 {
        15.0
    } else if n == 4 {
        105.0
    } else {
        945.0
    };

    let two_power = (1u32 << (n + 1)) as f64;
    let sqrt_x = f64::sqrt(x);

    let mut x_pow = 1.0;
    let mut m: u32 = 0;
    while m < n {
        x_pow *= x;
        m += 1;
    }

    coeff * SQRT_PI_F64 / (two_power * x_pow * sqrt_x)
}
```

Advantages of inlining:

- No need to mark the helper as `#[cube]`.
    
- Eliminates one level of function-call complexity in the IR.
    
- Often easier to debug, especially for small, fixed logic (like a small double-factorial table).
    

Disadvantages:

- Code duplication if you need the same helper elsewhere (host code or other kernels).
    
- Less modular if the logic becomes more complex.
    

---

## 4. Best Practices for Helpers in `#[cube]` Code

To avoid similar issues in the future, follow these guidelines when writing CubeCL kernels:

1. **All helpers called from `#[cube]` must be CubeCL-aware**
    
    - Either mark them as `#[cube]` and ensure they use only CubeCL-supported types and operations.
        
    - Or inline their logic directly into the kernel.
        
2. **Respect CubeCL’s type restrictions**
    
    - Use `f32` / `f64` for floats.
        
    - Use `u32` / `i32` for indices and counters.
        
    - Avoid `usize`, `u64`, and other host-only or non-`CubeType` types inside `#[cube]` functions.
        
3. **Keep host-only logic outside `#[cube]`**
    
    - If a function uses features not supported in kernels (allocations, I/O, complex borrowing, etc.), it must remain host-only and **must not** be called from `#[cube]` functions.
        
4. **When in doubt, test your design with a minimal kernel**
    
    - Write a small `#[cube]` function that calls your helper.
        
    - Run `cargo build` and fix any translation errors early.
        

---

## 5. Summary

- **Symptom**: Calling a plain Rust function from a `#[cube]` function leads to an error like  
    `function <name> is not a crate or module (E0433)`.
    
- **Root Cause**: The `#[cube]` macro expects called functions to be part of CubeCL’s IR pipeline, not plain Rust functions.
    
- **Primary Fixes**:
    
    - Mark helper functions as `#[cube]` and keep them CubeCL-compatible; or
        
    - Inline the helper logic directly into the `#[cube]` function.
        

By ensuring all functions used in kernels are either `#[cube]` functions or directly inlined, you can avoid this error and keep your CubeCL kernels compiling cleanly.
```