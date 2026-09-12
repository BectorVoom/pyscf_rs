//! GEMM kernel availability on the workspace's cubecl 0.10.0.
//!
//! This used to check `cubecl-matmul` 0.9.0-pre.5 against cubecl-runtime
//! 0.10.0. cubecl-matmul never published a 0.10.0 and pinning the pre-release
//! pulled a second cubecl stack (core/ir/macros 0.9.0-pre.5) into the build.
//! Its successor, `cubek-matmul` 0.2.0, is built on cubecl 0.10.0, so the
//! workspace now resolves to a single cubecl.
//!
//! The call shape a GEMM call site will use:
//! `cubek_matmul::launch::launch::<R>(&Strategy::Auto, &client, lhs, rhs, out, ...)`.

use cubecl::Runtime;

/// CPU-only smoke test. Verifies the `cubek_matmul` entry point exists at the
/// expected path and a CPU client can be constructed. No GEMM is run: the
/// value is the build link, and the first real call site carries its own test.
#[test]
fn cubek_matmul_symbol_exists() {
    let device = cubecl_cpu::CpuDevice;
    let _client = cubecl_cpu::CpuRuntime::client(&device);
    check_strategy_auto_exists();
}

/// Compile-time check that `cubek_matmul::launch::Strategy::Auto` is a valid
/// path. A cubek-matmul upgrade that moves or renames it fails to compile here
/// first.
fn check_strategy_auto_exists() {
    let _ = cubek_matmul::launch::Strategy::Auto;
}
