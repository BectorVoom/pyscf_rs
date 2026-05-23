//! UMP2 structural / wiring scaffold (always-on layer for the `mp2-structural`
//! CI job). For THIS plan (05-01) the spin-block kernel is not yet wired, so the
//! kernel-shape + error-propagation assertions are `#[ignore]`d (they land in
//! plan 05-04). The always-on arm asserts the module/type surface exists so the
//! file compiles and is discovered by the test harness.

/// Always-on: the UMP2 path's crate surface is reachable (error type +
/// helper re-exports shared with the restricted path). Proves the wiring
/// compiles independent of the (not-yet-written) spin-block kernel.
#[test]
fn ump2_module_surface_exists() {
    fn assert_error_type(_e: Option<pyscf_mp2::Mp2Error>) {}
    assert_error_type(None);

    // get_frozen_mask is exercised by both restricted and unrestricted paths.
    let _: fn(&[f64], usize) -> _ = pyscf_mp2::get_frozen_mask;
}

/// UMP2 spin-block kernel SHAPE + error-propagation. Wired in plan 05-04
/// (needs the not-yet-implemented UMP2 alpha/beta spin-block kernel).
#[test]
#[ignore = "wired in plan 05-04"]
fn ump2_kernel_shape_and_error_propagation() {
    // Placeholder: once ump2::UMP2 lands, build a toy unrestricted reference and
    // assert the per-spin-block kernel call shape + error propagation.
}
