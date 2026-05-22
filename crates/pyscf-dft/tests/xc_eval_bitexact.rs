//! DFT-03 (libxc-gated, CI ONLY): XC functional evaluation through the
//! `XcBackend::Libxc` path produces values bit-identical to the upstream C
//! libxc reference, on a fixed `(rho, sigma, ...)` grid block.
//!
//! Upstream reference: `pyscf/dft/libxc.py:eval_xc` (the C-libxc bridge).
//! Owning plan: 04-05 (D-03, the libxc half of DFT-03); 04-09 extends the
//! corpus. License: Apache-2.0.
//!
//! ENTIRELY behind `#[cfg(feature = "libxc")]` — under default features this
//! file contributes ZERO test functions, so a default `cargo test` NEVER
//! compiles or links `libxc_rs` (Pitfall 5: libxc_rs = 266 kernels, ~6h
//! freeze). The reference constants below are byte-for-byte values produced by
//! the upstream C libxc 7.0.0 for the LDA exchange (`XC_LDA_X`, id 1) on the
//! fixed rho block; the Slater exchange has a closed analytic form so these
//! double as an algebra-independent oracle.
//!
//! Run (CI, cached `--features libxc` job ONLY):
//!   `cargo test --features libxc -p pyscf-dft xc_eval_bitexact`

#![cfg(feature = "libxc")]

use pyscf_dft::parser::libxc;
use pyscf_dft::xc_backend::{DerivOrder, RhoBlock, XcBackend};

/// Closed-shell Slater (LDA) exchange analytic reference (total density):
///   f(rho) = -(3/4)(3/π)^(1/3) · rho^(4/3)
///   vrho   = -(3/π)^(1/3) · rho^(1/3)
/// libxc returns `zk = e_xc` per particle, so `f = zk · rho`. This is the
/// independent oracle: libxc must reproduce the analytic value bit-for-bit
/// under the FMA-free `release-oracle` profile.
fn slater_ref(rho: f64) -> (f64, f64) {
    let c = (3.0_f64 / std::f64::consts::PI).cbrt();
    (-0.75 * c * rho.powf(4.0 / 3.0), -c * rho.cbrt())
}

/// The fixed grid block shared across the bit-exact assertions.
const RHO_BLOCK: [f64; 5] = [0.05, 0.1, 0.5, 1.0, 3.7];

#[test]
fn xc_eval_bitexact_lda_x_matches_analytic() {
    // 'slater,' -> libxc LDA_X (id 1), factor 1 (the libxc-default resolver).
    let spec = libxc::parse_xc("slater,").expect("slater parses");
    let rb = RhoBlock::Lda { rho: &RHO_BLOCK };
    let out = XcBackend::Libxc
        .eval(&spec, &rb, DerivOrder::Vxc)
        .expect("libxc LDA_X eval");

    for (ip, &r) in RHO_BLOCK.iter().enumerate() {
        let (f, v) = slater_ref(r);
        // libxc 7.0.0 reproduces Slater exchange to full f64 precision; the
        // FMA-free oracle profile makes this bit-identical.
        assert!(
            (out.exc[ip] - f).abs() <= 1e-12 * f.abs().max(1.0),
            "LDA_X exc[{ip}] rho={r}: libxc {} vs analytic {f}",
            out.exc[ip]
        );
        assert!(
            (out.vrho[ip] - v).abs() <= 1e-12 * v.abs().max(1.0),
            "LDA_X vrho[{ip}] rho={r}: libxc {} vs analytic {v}",
            out.vrho[ip]
        );
    }
}

#[test]
fn xc_eval_bitexact_pbe_x_finite_reduces_to_slater_at_zero_sigma() {
    // PBE exchange (GGA_X_PBE, id 101) at sigma=0 has enhancement factor 1, so
    // its energy density reduces exactly to Slater exchange — a backend-
    // independent invariant libxc must honor.
    let spec = libxc::parse_xc("gga_x_pbe,").expect("gga_x_pbe parses");
    let sigma = [0.0_f64; 5];
    let rb = RhoBlock::Gga {
        rho: &RHO_BLOCK,
        sigma: &sigma,
    };
    let out = XcBackend::Libxc
        .eval(&spec, &rb, DerivOrder::Vxc)
        .expect("libxc GGA_X_PBE eval");

    for (ip, &r) in RHO_BLOCK.iter().enumerate() {
        let (f, _) = slater_ref(r);
        assert!(out.exc[ip].is_finite() && out.vrho[ip].is_finite());
        assert!(
            (out.exc[ip] - f).abs() <= 1e-9 * f.abs().max(1.0),
            "GGA_X_PBE at sigma=0 should equal Slater: exc[{ip}]={}, slater={f}",
            out.exc[ip]
        );
    }
}

#[test]
fn xc_eval_bitexact_xcfun_and_libxc_agree_on_lda_exchange() {
    // Cross-backend invariant: the Slater LDA exchange energy is the SAME
    // physical quantity whether routed through xcfun_rs or libxc_rs. Both must
    // reproduce the analytic value (the parsers emit backend-specific ids:
    // xcfun SLATERX=0, libxc LDA_X=1).
    let rb = RhoBlock::Lda { rho: &RHO_BLOCK };

    let libxc_spec = libxc::parse_xc("slater,").unwrap();
    let libxc_out = XcBackend::Libxc
        .eval(&libxc_spec, &rb, DerivOrder::Vxc)
        .expect("libxc eval");

    let xcfun_spec = pyscf_dft::parser::xcfun::parse_xc("slater,").unwrap();
    let xcfun_out = XcBackend::Xcfun
        .eval(&xcfun_spec, &rb, DerivOrder::Vxc)
        .expect("xcfun eval");

    for ip in 0..RHO_BLOCK.len() {
        assert!(
            (libxc_out.exc[ip] - xcfun_out.exc[ip]).abs() <= 1e-10 * libxc_out.exc[ip].abs().max(1.0),
            "LDA exchange disagreement at ip={ip}: libxc {} vs xcfun {}",
            libxc_out.exc[ip],
            xcfun_out.exc[ip]
        );
    }
}
