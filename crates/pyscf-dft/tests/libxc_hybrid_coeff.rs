#![cfg(feature = "libxc")]
//! Hybrid / range-separation coefficients under the libxc backend.
//!
//! # Why this test is load-bearing for the libxc default
//!
//! The two parsers put the hybrid mixing in different places, and both match
//! their upstream counterpart:
//!
//! * `parser::xcfun` expands `b3lyp` into four primitives and reports
//!   `hyb = (0.2, 0, 0)` in the triple;
//! * `parser::libxc` resolves `b3lyp` to the single COMPOUND id `402` and
//!   reports `hyb = (0, 0, 0)` — exactly as upstream's `libxc.parse_xc` does.
//!   The 0.2 lives INSIDE functional 402.
//!
//! So a libxc-backed `rsh_coeff` that read `spec.hyb()` would report every
//! hybrid as a pure functional and drop the exact exchange entirely — B3LYP
//! would run with no HF exchange and still converge, to the wrong answer.
//! `XcBackend::rsh_coeff` asks the library instead. This test is what says so.

use pyscf_dft::XcBackend;

/// Reference values from upstream PySCF 2.12.1:
/// ```python
/// from pyscf.dft import libxc
/// libxc.rsh_coeff(xc)      # -> (omega, alpha, beta)
/// libxc.hybrid_coeff(xc)   # -> hyb
/// ```
#[test]
fn libxc_hybrid_and_rsh_coefficients_match_upstream() {
    let b = XcBackend::Libxc;
    let cases: &[(&str, f64, f64, f64)] = &[
        ("lda,vwn", 0.0, 0.0, 0.0),
        ("pbe", 0.0, 0.0, 0.0),
        ("blyp", 0.0, 0.0, 0.0),
        ("b3lyp", 0.0, 0.2, 0.2),
        ("pbe0", 0.0, 0.25, 0.25),
    ];
    for &(xc, w, a, h) in cases {
        let (gw, ga, gh) = b
            .rsh_and_hybrid_coeff(xc)
            .unwrap_or_else(|e| panic!("{xc}: {e}"));
        println!("{xc:>10}: omega {gw:.6} alpha {ga:.6} hyb {gh:.6}  (want {w} {a} {h})");
        assert!((gw - w).abs() < 1e-12, "{xc}: omega {gw} != {w}");
        assert!((ga - a).abs() < 1e-12, "{xc}: alpha {ga} != {a}");
        assert!((gh - h).abs() < 1e-12, "{xc}: hyb {gh} != {h}");
    }
}

/// A pure functional is reported as pure, and a hybrid as a hybrid — the
/// distinction the J/K dispatch branches on.
#[test]
fn hybrid_detection_separates_pure_from_hybrid() {
    let b = XcBackend::Libxc;
    for xc in ["lda,vwn", "pbe", "blyp"] {
        assert_eq!(
            b.hybrid_coeff(xc).expect(xc),
            0.0,
            "{xc} is a PURE functional"
        );
    }
    for xc in ["b3lyp", "pbe0"] {
        assert!(b.hybrid_coeff(xc).expect(xc) > 0.0, "{xc} is a HYBRID");
    }
}

/// **CAM-B3LYP — a range-separated hybrid, now working.**
///
/// This replaces `cam_b3lyp_is_blocked_by_a_libxc_rs_defect`, which asserted
/// that constructing functional 433 failed inside libxc_rs with
/// `PropagationConflict { parent_name: "_omega", aux_slot: 1 }`. That defect
/// (and the eight sibling range-separated hybrids it also broke) was fixed on
/// 2026-08-28, and the trip-wire fired as designed.
///
/// CAM-B3LYP is the strongest case in this file: three DISTINCT coefficients,
/// none of them reachable from `spec.hyb()` — the libxc parser resolves it to
/// the single compound id 433 and reports `(0, 0, 0)`, exactly as upstream
/// does. Getting `(0.33, 0.65, 0.19)` out requires the library's own
/// `xc_hyb_cam_coef` path plus upstream's `hyb = alpha + beta` rule for
/// `omega != 0`. Note `beta` is NEGATIVE here (-0.46), so a sign error would
/// still produce a plausible-looking hybrid fraction.
///
/// Reference: upstream PySCF 2.12.1 `libxc.rsh_coeff('cam-b3lyp')` ->
/// `(0.33, 0.65, -0.46)`, `libxc.hybrid_coeff('cam-b3lyp')` -> 0.65.
#[test]
fn cam_b3lyp_range_separated_coefficients_match_upstream() {
    let (omega, alpha, hyb) = XcBackend::Libxc
        .rsh_and_hybrid_coeff("cam-b3lyp")
        .expect("cam-b3lyp must resolve");
    println!("cam-b3lyp: omega {omega} alpha {alpha} hyb {hyb}");
    assert!((omega - 0.33).abs() < 1e-12, "omega {omega} != 0.33");
    assert!((alpha - 0.65).abs() < 1e-12, "alpha {alpha} != 0.65");
    // hyb = alpha + beta = 0.65 + (-0.46); upstream's rule for omega != 0.
    assert!((hyb - 0.19).abs() < 1e-12, "hyb {hyb} != 0.19");
}
