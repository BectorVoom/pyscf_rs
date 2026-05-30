version 0.8.1
```markdown
Below is a consolidated “error resolution” document for the two CubeCL-related errors you encountered.

---

# CubeCL Error Resolution Guide

## 1. `mismatched types: expected struct ExpandElementTyped<_> found type {float}` (E0308)

### 1.1 Symptom

You see an error similar to:

```text
mismatched types
expected struct `ExpandElementTyped<_>`
found type `{float}`
```

with code of the form:

```rust
#[cube]
#[inline(always)]
pub fn erf_approx_f32(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let y    = if x >= 0.0 { x   } else { -x   };

    // ...
}
```

### 1.2 Root Cause

Inside a `#[cube]` function, CubeCL does not treat all expressions as plain `f32`/`f64` values. After macro expansion, expressions are wrapped in internal IR types such as `ExpandElementTyped<T>`.

When you use an `if` expression directly as a value (e.g., `let sign = if cond { a } else { b };`), the CubeCL macro sometimes fails to unify the IR types of both branches and the binding target. This leads to a mismatch:

- “expected `ExpandElementTyped<_>`”
    
- “found `{float}`” (a plain float literal)
    

This is a known limitation/fragility in the current `#[cube]` translation of `if`-expressions.

### 1.3 Recommended Fix

Rewrite `if` _expressions_ as `if` _statements_ that assign to mutable variables. That is, avoid:

```rust
let var = if cond { expr1 } else { expr2 };
```

and instead:

```rust
let mut var = initial_value;
if cond {
    var = expr1;
} else {
    var = expr2;
}
```

#### Before (problematic)

```rust
#[cube]
#[inline(always)]
pub fn erf_approx_f32(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let y    = if x >= 0.0 { x   } else { -x   };

    // ...
}
```

#### After (CubeCL-friendly)

```rust
use cubecl::prelude::*;

#[cube]
#[inline(always)]
pub fn erf_approx_f32(x: f32) -> f32 {
    let mut sign: f32 = 1.0;
    let mut y:    f32 = x;

    if x < 0.0 {
        sign = -1.0;
        y    = -x;
    }

    // ... rest of the implementation
    sign
}
```

Key points:

- Initialize variables with a default value.
    
- Use `if` as a statement to update `sign` and `y`.
    
- Do not rely on `if` expressions returning values in `#[cube]` code.
    

This pattern avoids the `ExpandElementTyped` vs `{float}` mismatch.

---

## 2. `no method named __expand_exp_method found for struct ExpandElementTyped<T>` (E0599)

### 2.1 Symptom

You see an error similar to:

```text
no method named `__expand_exp_method` found for struct `ExpandElementTyped<T>` in the current scope
```

with code of the form:

```rust
#[cube]
#[inline(always)]
pub fn erf_approx_f64(x: f64) -> f64 {
    let mut sign: f64 = 1.0;
    let mut y: f64 = x;
    if x < 0.0 {
        sign = -1.0;
        y = -x;
    }

    // ...

    let exp_term = (-y * y).exp(); // <- error here
    let result = sign * (1.0 - poly * exp_term);
    result
}
```

### 2.2 Root Cause

CubeCL provides exponentials via an `Exp` trait implementation (and related IR machinery):

- The trait exposes `fn exp(x: Self) -> Self` as a _static/associated function_, not as an instance method `x.exp()`.
    
- For `f32` and `f64`, CubeCL implements `Exp`, and the macro knows how to lower calls like `f64::exp(value)` or `Exp::exp(value)` into its IR.
    

When you write `(-y * y).exp()`, the `#[cube]` macro tries to find an internal method like `__expand_exp_method` on `ExpandElementTyped<T>` to support a method-style call, but no such method exists. Hence the error.

### 2.3 Recommended Fix

Do **not** use `.exp()` as a method in `#[cube]` functions. Instead, call the associated function via the type or trait:

- `f64::exp(arg)` or `f32::exp(arg)`
    
- or `Exp::exp(arg)` (after importing the trait)
    

#### Before (problematic)

```rust
#[cube]
#[inline(always)]
pub fn erf_approx_f64(x: f64) -> f64 {
    // ...
    let exp_term = (-y * y).exp();
    let result = sign * (1.0 - poly * exp_term);
    result
}
```

#### After (CubeCL-friendly)

```rust
use cubecl::frontend::Exp; // or use cubecl::prelude::Exp;

#[cube]
#[inline(always)]
pub fn erf_approx_f64(x: f64) -> f64 {
    let mut sign: f64 = 1.0;
    let mut y: f64 = x;
    if x < 0.0 {
        sign = -1.0;
        y = -x;
    }

    let t  = 1.0 / (1.0 + ERF_P_F64 * y);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly =
        ERF_A1_F64 * t
        + ERF_A2_F64 * t2
        + ERF_A3_F64 * t3
        + ERF_A4_F64 * t4
        + ERF_A5_F64 * t5;

    // Use type/trait associated function instead of .exp()
    let exp_term = f64::exp(-y * y);
    // or:
    // let exp_term = Exp::exp(-y * y);

    let result = sign * (1.0 - poly * exp_term);
    result
}
```

Key points:

- Import the `Exp` trait (e.g., `use cubecl::frontend::Exp;` or via the prelude).
    
- Call `f64::exp(arg)` or `Exp::exp(arg)` rather than `arg.exp()`.
    

### 2.4 Generic Variant (Optional)

If you use a generic `F: Float` pattern (as in some CubeCL/Burn examples), use the type’s associated functions:

```rust
#[cube]
fn some_kernel<F: Float>(x: F) -> F {
    let e = F::exp(x);
    // ...
    e
}
```

This style is fully compatible with CubeCL’s IR translation.

---

## 3. Summary of Best Practices for CubeCL Kernels

When writing `#[cube]` functions in Rust:

1. **Avoid `if` expressions that produce values**
    
    - Do not use `let v = if cond { a } else { b };` inside `#[cube]`.
        
    - Prefer:
        
        ```rust
        let mut v = default;
        if cond { v = a; } else { v = b; }
        ```
        
2. **Use associated functions for math (exp, sqrt, etc.)**
    
    - Do not call `x.exp()`, `x.sqrt()`, etc. on CubeCL values.
        
    - Instead, use:
        
        - `f64::exp(x)`, `f64::sqrt(x)` (or `f32::…`)
            
        - or `Exp::exp(x)`, `Sqrt::sqrt(x)` via the relevant traits.
            
3. **Stay within CubeCL-supported types**
    
    - Use `f32` / `f64` and `u32`/`i32` in kernels.
        
    - Avoid `usize` and host-only types in device code.
        

Following these patterns will prevent the two errors above and generally make your CubeCL kernels compile and lower correctly.
```