//! DFT-02: XC-string parser parity vs `pyscf/dft/libxc.py:parse_xc` (the
//! default resolver) and `pyscf/dft/xcfun.py:parse_xc` (the alternate).
//!
//! Upstream reference: `pyscf/dft/libxc.py:parse_xc` (lines 491-718) +
//! `XC_CODES` (154) + `XC_ALIAS` (217); `pyscf/dft/xcfun.py:parse_xc`
//! (416-569) + its `XC_CODES`/`XC_ALIAS`. Owning plan: 04-05 (D-01/DFT-02).
//! License: Apache-2.0 (matching upstream PySCF).
//!
//! Parity oracle (mirrors the 04-04 `grid_weights_level_sweep` count-sweep
//! precedent): each XC string's expected `((hyb,alpha,omega), [(id,fac),...])`
//! is transcribed *directly from the upstream `parse_xc` algorithm* — the
//! libxc IDs are the canonical libxc 7.0.0 numbers (verified against
//! `libxc_rs/src/registry/by_name.rs`), the xcfun IDs are the xcfun 0..77
//! numbers (`xcfun-core/src/functional_id.rs`). Because `parse_xc` is a pure
//! string→spec mapping with NO floating-point transcendentals (only exact
//! `+`/`*` on the literal weights in the string), this hand-transcribed table
//! IS the authoritative DFT-02 oracle — bit-exact equality is the contract.
//!
//! This avoids a direct pyo3 dependency in `pyscf-dft` (PyO3 wall — the live
//! libpython surface lives in `pyscf-py` / the `pyscf-oracle` dev-harness, not
//! in a method crate). The same rationale the grids count-sweep uses: the
//! independent algorithm replica is the oracle, not a live-Python round-trip.
//!
//! Run:
//!   `cargo test -p pyscf-dft parse_xc_parity`   (or `--test parse_xc_parity`)

use pyscf_dft::error::DftError;
use pyscf_dft::parser::{libxc, xcfun, XcSpec};

/// Compare a parsed spec against the expected `(hyb, components)`, with an
/// exact float match on every factor (the parser does only exact arithmetic —
/// no FMA, no transcendentals — so bit-exact equality is the contract).
fn assert_spec(spec: &XcSpec, hyb: (f64, f64, f64), components: &[(u32, f64)]) {
    assert_eq!(spec.hyb, hyb, "hyb (hybrid,alpha,omega) mismatch");
    assert_eq!(
        spec.components.as_slice(),
        components,
        "component (id,fac) list mismatch"
    );
}

// ---------------------------------------------------------------------------
// libxc-default parser parity (D-01 — b3lyp routes here for upstream numbers)
// ---------------------------------------------------------------------------

#[test]
fn parse_xc_parity_libxc_single_compound_b3lyp() {
    // Compound libxc functional id 402 (HYB_GGA_XC_B3LYP), factor 1. The
    // hybrid coefficient is folded inside the libxc functional, so hyb=(0,0,0).
    let spec = libxc::parse_xc("b3lyp").expect("b3lyp parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(402, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_comma_form_pbe_pbe() {
    // X part PBE -> GGA_X_PBE(101); C part PBE -> GGA_C_PBE(130).
    let spec = libxc::parse_xc("pbe,pbe").expect("pbe,pbe parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(101, 1.0), (130, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_shorthand_pbe_alias() {
    // No-comma `pbe` -> XC_ALIAS "PBE,PBE" -> GGA_X_PBE(101), GGA_C_PBE(130).
    let spec = libxc::parse_xc("pbe").expect("pbe parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(101, 1.0), (130, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_shorthand_blyp_alias() {
    // BLYP -> XC_ALIAS "B88,LYP" -> GGA_X_B88(106), GGA_C_LYP(131).
    let spec = libxc::parse_xc("blyp").expect("blyp parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(106, 1.0), (131, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_shorthand_svwn_alias() {
    // SVWN -> XC_ALIAS "SLATER,VWN" -> LDA_X(1), LDA_C_VWN(7).
    let spec = libxc::parse_xc("svwn").expect("svwn parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(1, 1.0), (7, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_explicit_weights_hf_b88_lyp() {
    // '.5*HF + .5*B88,LYP': HF mixes both SR (hyb) and LR (alpha); B88 X, LYP C.
    let spec = libxc::parse_xc(".5*HF + .5*B88,LYP").expect("weighted form parses");
    assert_spec(&spec, (0.5, 0.5, 0.0), &[(106, 0.5), (131, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_factor_either_order() {
    // 'B88*0.5,LYP' — factor after the name; same result as '0.5*B88,LYP'.
    let spec = libxc::parse_xc("B88*0.5,LYP").expect("factor-after-name parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(106, 0.5), (131, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_raw_integer_id() {
    // A bare integer is a raw libxc ID (libxc.py:600-601).
    let spec = libxc::parse_xc("106,131").expect("raw ids parse");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(106, 1.0), (131, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_x_only_trailing_comma() {
    // 'slater,' — X functional only (LDA_X = 1), empty C part.
    let spec = libxc::parse_xc("slater,").expect("X-only parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(1, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_c_only_leading_comma() {
    // ',vwn' — C functional only.
    let spec = libxc::parse_xc(",vwn").expect("C-only parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(7, 1.0)]);
}

#[test]
fn parse_xc_parity_libxc_unit_scaled_compound() {
    // '0.5*b3lyp5' scales the expanded compound as a unit
    // ('.2*HF + .08*SLATER + .72*B88, .81*LYP + .19*VWN').
    let spec = libxc::parse_xc("0.5*b3lyp5").expect("unit-scaled compound parses");
    assert_spec(
        &spec,
        (0.1, 0.1, 0.0),
        &[(1, 0.04), (106, 0.36), (131, 0.405), (7, 0.095)],
    );
}

#[test]
fn parse_xc_parity_libxc_dedup_sums_factors() {
    // Same functional twice -> factors summed in encounter order (remove_dup).
    let spec = libxc::parse_xc("0.3*B88 + 0.7*B88,LYP").expect("dedup parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(106, 1.0), (131, 1.0)]);
}

// --- malformed inputs: clean Err, never a panic (T-04-05a) ---

#[test]
fn parse_xc_parity_libxc_malformed_unknown_functional() {
    assert!(matches!(
        libxc::parse_xc("totally_bogus_xc_name"),
        Err(DftError::UnknownFunctional { .. })
    ));
}

#[test]
fn parse_xc_parity_libxc_malformed_too_many_commas() {
    assert!(matches!(
        libxc::parse_xc("b88,lyp,vwn"),
        Err(DftError::TooManyCommas { .. })
    ));
}

#[test]
fn parse_xc_parity_libxc_malformed_bad_rsh() {
    // RSH with a non-numeric parameter must not panic.
    assert!(matches!(
        libxc::parse_xc("RSH(a;b;c) + HF"),
        Err(DftError::MalformedToken { .. })
    ));
}

#[test]
fn parse_xc_parity_libxc_malformed_does_not_panic_on_adversarial() {
    // A pile of operators / empty tokens / stray punctuation must return a
    // clean result or Err — never panic, never hang.
    for s in ["+++", "*", "0.5*", ",", "---,---", "()", "RSH()", "1e999*HF"] {
        let _ = libxc::parse_xc(s); // must simply not panic
    }
}

// ---------------------------------------------------------------------------
// xcfun-alternate parser parity
// ---------------------------------------------------------------------------

#[test]
fn parse_xc_parity_xcfun_comma_form_pbe_pbe() {
    // suffix fallback: X part PBE -> PBEX(5); C part PBE -> PBEC(4).
    let spec = xcfun::parse_xc("pbe,pbe").expect("xcfun pbe,pbe parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(5, 1.0), (4, 1.0)]);
}

#[test]
fn parse_xc_parity_xcfun_shorthand_blyp_alias() {
    // BLYP -> "B88,LYP" -> BECKEX(6), LYPC(16).
    let spec = xcfun::parse_xc("blyp").expect("xcfun blyp parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(6, 1.0), (16, 1.0)]);
}

#[test]
fn parse_xc_parity_xcfun_b3lyp_expands_to_primitives() {
    // xcfun b3lyp -> B3LYPG expansion (NOT a single id like libxc):
    // .2*HF + .08*SLATER + .72*B88 + .81*LYP + .19*VWN3C.
    let spec = xcfun::parse_xc("b3lyp").expect("xcfun b3lyp parses");
    assert_spec(
        &spec,
        (0.2, 0.0, 0.0),
        &[(0, 0.08), (6, 0.72), (16, 0.81), (2, 0.19)],
    );
}

#[test]
fn parse_xc_parity_xcfun_explicit_weights_hf_b88_lyp() {
    // xcfun zeroes LR_HF (alpha) when no omega is assigned (xcfun.py:567-568).
    let spec = xcfun::parse_xc(".5*HF + .5*B88,LYP").expect("xcfun weighted parses");
    assert_spec(&spec, (0.5, 0.0, 0.0), &[(6, 0.5), (16, 1.0)]);
}

#[test]
fn parse_xc_parity_xcfun_svwn_compound() {
    // SVWN -> XC_ALIAS "SLATER,VWN" -> SLATERX(0), VWN5C(3).
    let spec = xcfun::parse_xc("svwn").expect("xcfun svwn parses");
    assert_spec(&spec, (0.0, 0.0, 0.0), &[(0, 1.0), (3, 1.0)]);
}

#[test]
fn parse_xc_parity_xcfun_malformed_unknown_functional() {
    assert!(matches!(
        xcfun::parse_xc("not_a_functional_xyz"),
        Err(DftError::UnknownFunctional { .. })
    ));
}

#[test]
fn parse_xc_parity_xcfun_malformed_too_many_commas() {
    assert!(matches!(
        xcfun::parse_xc("b88,lyp,vwn"),
        Err(DftError::TooManyCommas { .. })
    ));
}
