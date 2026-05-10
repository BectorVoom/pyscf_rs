//! Pitfall 1 mitigation: cubecl-matmul 0.9.0-pre.5 + cubecl-runtime
//! 0.10.0 ABI compatibility verification. If this test fails to
//! compile or panics at runtime, PLAN.md must reroute to a hand-rolled
//! `#[cube]` GEMM (cintx pattern) — but as of RESEARCH 2026-05-10 the
//! ABI was assumed compatible.
//!
//! Per docs/manual/Cubecl/cubecl_matmul_gemm_example.md:
//! `cubecl_matmul::launch::<R, T>(&Strategy::Auto, &client, lhs, rhs, out)`.

use cubecl::Runtime;

/// CPU-only smoke test. Just verifies the `cubecl_matmul` symbols
/// exist at the expected paths. We do NOT run a real GEMM in Phase 1
/// because the value is the build link (RESEARCH Pitfall 1: "Failure
/// means PLAN.md must reroute to hand-rolled GEMM"); the actual GEMM
/// call site lands in Phase 2.
#[test]
fn cubecl_matmul_symbol_exists() {
    // Construct a CPU client to confirm runtime construction works.
    let device = cubecl_cpu::CpuDevice::default();
    let _client = cubecl_cpu::CpuRuntime::client(&device);
    // The fact that this test compiles AND links is the proof: the
    // cubecl-matmul crate's symbols and types are visible. Rustdoc-
    // checking helpers below confirm `cubecl_matmul::Strategy` is
    // reachable as named.
    check_strategy_auto_exists();
}

/// Compile-time check that `cubecl_matmul::Strategy::Auto` is a valid
/// path. If cubecl-matmul 0.10.0 lands and renames `Strategy::Auto`,
/// this test fails to compile and surfaces the ABI break early.
fn check_strategy_auto_exists() {
    // Reference the symbol; does not invoke (Phase 1 doesn't exercise
    // a real GEMM — Phase 2 does).
    let _ = cubecl_matmul::Strategy::Auto;
}
