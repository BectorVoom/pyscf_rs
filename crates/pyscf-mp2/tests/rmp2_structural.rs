//! RMP2 structural / wiring scaffold (always-on layer for the `mp2-structural`
//! CI job). For THIS plan (05-01) the kernel is not yet wired, so the
//! kernel-shape + error-propagation assertions are `#[ignore]`d (they land in
//! plan 05-03). The always-on arm asserts the module/type surface exists so the
//! file compiles and is discovered by the test harness.

/// Always-on: the RMP2 module surface + the error type + the MP2-08 helper
/// re-exports are reachable. This proves the wiring (deps + module tree)
/// compiles and is discovered, independent of the (not-yet-written) kernel.
#[test]
fn rmp2_module_surface_exists() {
    // The crate-level error type is exported.
    fn assert_error_type(_e: Option<pyscf_mp2::Mp2Error>) {}
    assert_error_type(None);

    // The MP2-08 helper re-exports are reachable from the crate root.
    let _: fn(&[f64], &pyscf_mp2::Frozen) -> _ = pyscf_mp2::get_nocc;
}

/// RMP2 kernel SHAPE + error-propagation. Wired in plan 05-03 (needs the
/// not-yet-implemented RMP2 kernel + `_ChemistsERIs` ao2mo path).
#[test]
#[ignore = "wired in plan 05-03"]
fn rmp2_kernel_shape_and_error_propagation() {
    // Placeholder: once mp2::RMP2 lands, build a toy Mp2Reference-shaped input
    // and assert the kernel call shape + that an ao2mo error propagates as an
    // Mp2Error (NOT the numeric energy, which is the cintx-gated oracle's job).
}
