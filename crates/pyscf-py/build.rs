//! Compile-time feature-mutex guard.
//!
//! Cargo features are not natively mutually exclusive. Without this guard a
//! developer or CI run can pass `--features abi3-py310,free-threading` and
//! produce a wheel whose C ABI is undefined: abi3 promises a stable
//! limited-API surface, but the free-threaded interpreter (3.13t) drops the
//! GIL semantics that abi3 callers rely on.
//!
//! Source: RESEARCH §"CRITICAL abi3 vs free-threaded ABI conflict" + checker
//! iteration 1 WARNING 4. Phase 3 plan 03-02 ships this guard BEFORE plan
//! 03-07 wires the PyO3 deps so the guard is in place from the first build.
fn main() {
    let abi3 = std::env::var("CARGO_FEATURE_ABI3_PY310").is_ok();
    let ft = std::env::var("CARGO_FEATURE_FREE_THREADING").is_ok();
    if abi3 && ft {
        panic!(
            "pyscf-py: features `abi3-py310` and `free-threading` are mutually exclusive \
             — choose one. See .planning/phases/03-scf-pyo3-bindings/03-RESEARCH.md \
             §'CRITICAL abi3 vs free-threaded ABI conflict'"
        );
    }
    println!("cargo:rerun-if-changed=build.rs");
}
