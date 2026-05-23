//! AO→MO synthetic-ERI transform correctness scaffold.
//!
//! This is the ONE numeric assertion that ships un-gated this phase (per
//! RESEARCH Validation Architecture): a synthetic ERI tensor transformed with
//! an identity MO coefficient block must round-trip to itself. For THIS plan
//! (05-01) the transform body is a stub returning `NotYetImplemented`, so the
//! numeric assertion is `#[ignore]`d (it lands in plan 05-02). The always-on
//! compile-smoke references `pyscf_ao2mo::general` to prove the public symbol
//! surface exists and is callable.

/// Always-on compile-smoke: `pyscf_ao2mo::general` is exported and callable.
/// The stub currently returns `Err(NotYetImplemented)`; this arm asserts the
/// CONTRACT (symbol exists + callable), not the numeric result.
#[test]
fn ao2mo_general_symbol_exists() {
    // 1-AO toy: eri_ao is a single (00|00) element, identity MO block.
    let eri_ao = [1.0_f64];
    let mo = [1.0_f64];
    let mo_coeffs: [&[f64]; 4] = [&mo, &mo, &mo, &mo];
    let _ = pyscf_ao2mo::general(&eri_ao, &mo_coeffs, 1);

    // `full` (the symmetric special case) is also exported.
    let _: fn(&[f64], &[f64], usize) -> _ = pyscf_ao2mo::full;
}

/// Numeric roundtrip: a synthetic ERI transformed with an identity MO block
/// equals itself. Wired in plan 05-02 (needs the real `general` body).
#[test]
#[ignore = "transform body lands in plan 05-02"]
fn ao2mo_identity_transform_roundtrips() {
    // Placeholder: with an identity MO coefficient matrix, general(eri_ao) must
    // equal eri_ao element-wise (the un-gated synthetic-ERI correctness check).
}
