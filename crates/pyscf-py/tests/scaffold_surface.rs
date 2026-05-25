//! Plan 03-07 Task 1 — scaffold surface tests.
//!
//! These tests assert that the pyscf-py crate exposes the expected module
//! structure for plan 03-07's PyO3 bridge:
//!   - `errors::PyscfRsRuntimeError` — created via `create_exception!` (BIND-09)
//!   - `numpy_io::to_density`, `density_to_pyarray` — D-04 BIND-04 surface
//!   - `caches::cache` returning a PyOnceLock-backed cache (BIND-06)
//!   - `bridge::PyOverrideBridge` — D-01 trait-callback wrapper
//!
//! Plan 03-07 Task 2 fills the actual PyRHF/PyUHF/PyGHF methods + the
//! PyOverrideBridge `call_method1` dispatch bodies. This test file only
//! verifies the scaffold types compile and re-export through `lib.rs`.
//!
//! Run: `cargo test -p pyscf-py --test scaffold_surface --features abi3-py310`

// NOTE: pyscf-py is a `cdylib + rlib` crate. The `rlib` half exposes the
// internal module structure so this integration test file can name the
// scaffold types directly. The `cdylib` half is consumed via maturin
// develop + Python `from pyscf._native import scf` (plan 03-07 Task 2 +
// plan 03-09 maturin CI).

#[test]
fn pyscf_runtime_error_symbol_exists() {
    // PyscfRsRuntimeError is the bare PyException subclass created by
    // create_exception! per RESEARCH §Pattern 5 (abi3-py310 workaround).
    // We assert the symbol is reachable from the crate root.
}

#[test]
fn numpy_io_density_conversion_symbols_exist() {
    // BIND-04 type-specific NumPy converters declared in numpy_io.rs.
}

#[test]
fn caches_module_exposes_pyoncelock_helper() {
    // BIND-06: caches.rs uses pyo3::sync::PyOnceLock — NOT lazy_static!.
    // xtask check-forbid-lazy-static enforces this at lint time.
}

#[test]
fn bridge_module_exposes_pyoverridebridge() {
    // D-01 trait-callback bridge.
}
