//! DFT-03 (libxc-gated, CI ONLY): a libxc functional-corpus smoke — each
//! corpus functional evaluates without error and returns finite values over a
//! small grid block (smoke — no oracle compare; the bit-exact compare lives in
//! `xc_eval_bitexact.rs`).
//!
//! Upstream reference: `pyscf/dft/libxc.py` functional registry. Owning plan:
//! 04-05 (a starter corpus); 04-09 expands to the ~100-functional sweep.
//! License: Apache-2.0.
//!
//! ENTIRELY behind `#[cfg(feature = "libxc")]` — under default features this
//! file contributes ZERO test functions, so a default `cargo test` NEVER
//! compiles or links `libxc_rs` (Pitfall 5: libxc_rs = 266 kernels, ~6h
//! freeze).
//!
//! Run (CI, cached `--features libxc` job ONLY):
//!   `cargo test --features libxc -p pyscf-dft libxc_functional_smoke`

#![cfg(feature = "libxc")]

use pyscf_dft::parser::libxc;
use pyscf_dft::xc_backend::{DerivOrder, Family, RhoBlock, XcBackend};

/// A small grid block; nonzero rho + nonzero sigma/tau so every family path
/// exercises a real kernel.
const RHO: [f64; 4] = [0.05, 0.2, 1.0, 2.5];
const SIGMA: [f64; 4] = [0.001, 0.01, 0.1, 0.5];
const LAPL: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
const TAU: [f64; 4] = [0.01, 0.05, 0.2, 0.6];

/// Corpus subset: (XC string for the libxc-default parser, family). Each entry
/// is parsed via `parser::libxc` (emitting libxc ids) and evaluated through
/// `XcBackend::Libxc`.
const CORPUS: &[(&str, Family)] = &[
    // LDA
    ("slater,", Family::Lda),
    (",vwn", Family::Lda),
    // GGA exchange / correlation
    ("gga_x_b88,", Family::Gga),
    (",gga_c_lyp", Family::Gga),
    ("gga_x_pbe,", Family::Gga),
    (",gga_c_pbe", Family::Gga),
    ("blyp", Family::Gga),
    ("pbe,pbe", Family::Gga),
];

#[test]
fn libxc_functional_smoke() {
    for &(xc, family) in CORPUS {
        let spec = libxc::parse_xc(xc).unwrap_or_else(|e| panic!("parse '{xc}': {e}"));

        let rb = match family {
            Family::Lda => RhoBlock::Lda { rho: &RHO },
            Family::Gga => RhoBlock::Gga {
                rho: &RHO,
                sigma: &SIGMA,
            },
            Family::Mgga => RhoBlock::Mgga {
                rho: &RHO,
                sigma: &SIGMA,
                lapl: &LAPL,
                tau: &TAU,
            },
        };

        let out = XcBackend::Libxc
            .eval(&spec, &rb, DerivOrder::Vxc)
            .unwrap_or_else(|e| panic!("libxc eval '{xc}': {e}"));

        assert_eq!(out.exc.len(), RHO.len(), "exc length for '{xc}'");
        for (ip, &e) in out.exc.iter().enumerate() {
            assert!(e.is_finite(), "non-finite exc[{ip}] for '{xc}': {e}");
        }
        for (ip, &v) in out.vrho.iter().enumerate() {
            assert!(v.is_finite(), "non-finite vrho[{ip}] for '{xc}': {v}");
        }
        if family != Family::Lda {
            for (ip, &vs) in out.vsigma.iter().enumerate() {
                assert!(vs.is_finite(), "non-finite vsigma[{ip}] for '{xc}': {vs}");
            }
        }
    }
}
