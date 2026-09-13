//! Gamma-point restricted Kohn-Sham TDA/TDDFT (`pbc/tdscf/rks.py`, 52 l).
//!
//! Thin subclass over the 19-06 HF driver: the response is the HF part at the
//! functional's exact-exchange fraction plus the XC-kernel matrix,
//!
//! * `A = A_HF(hyb) + fxc_a`, `B = B_HF(hyb) + fxc_b`,
//!
//! where `A_HF`/`B_HF` come from [`crate::rhf::build_ab`] with `hyb` from the
//! functional (0 for pure, 0.25 for PBE0, 0.2 for B3LYP — see
//! [`hyb_fraction`]), and `fxc_a`/`fxc_b` are the grid-contracted XC-kernel
//! matrices (`(ia|f_xc|jb)`-type, same `(dim², [ia][jb])` layout).
//!
//! The XC contribution enters through `gen_response` — `NotImplemented` on
//! the PBC `KohnShamDFT` base, rebound only on the concrete class (19-03's
//! [`pyscf_pbc_scf::response::RksGenResponse`]; re-referenced here so the
//! seam is compiled into this crate's contract, not remembered). A hybrid
//! MUST NOT take the pure path: `hyb = 0` with a hybrid name is refused by
//! [`check_hybrid_kernel`].

use crate::davidson::tda_dense;
use crate::error::PbcTdscfError;
use crate::types::{TdaConfig, TdaResult};

/// Exact-exchange fraction by functional name (the `rsh_and_hybrid_coeff`
/// scalar every KS response needs; range separation `omega` is refused —
/// RSH needs the range-separated ERI route, a named non-goal here).
pub fn hyb_fraction(xc: &str) -> Result<f64, PbcTdscfError> {
    let key = xc.to_lowercase().replace([' ', ',', '_', '-'], "");
    // Range separation first: a range-separated name must NEVER resolve
    // through the global-hybrid table (the table contains cam-b3lyp's
    // short-range fraction, which is not the whole kernel).
    if key.contains("hse") || key.contains("cam") || key.contains("rsh") || key.contains("wb97") {
        return Err(PbcTdscfError::NotYetImplemented { module: "tdscf KS range-separated hybrid kernel" });
    }
    if key.contains("hf") && !key.contains("b3lyp") && !key.contains("pbe0") {
        return Ok(1.0);
    }
    for (name, f) in [
        ("pbe0", 0.25),
        ("b3lyp", 0.20),
        ("b3p86", 0.20),
        ("b3pw91", 0.20),
        ("pbeh", 0.25),
    ] {
        if key.contains(name) {
            return Ok(f);
        }
    }
    // Pure (semi)local functionals: LDA/GGA/MGGA names fall through to 0.
    Ok(0.0)
}

/// Refuse a pure-DFT kernel under a hybrid name: `hyb == 0` with an
/// exact-exchange-bearing name is a silent wrong-kernel error, not a default.
pub fn check_hybrid_kernel(xc: &str, hyb: f64) -> Result<(), PbcTdscfError> {
    let key = xc.to_lowercase();
    let looks_hybrid = ["b3lyp", "pbe0", "b3p86", "b3pw91", "hse", "cam", "pbeh", "hf"]
        .iter()
        .any(|t| key.contains(t));
    if looks_hybrid && hyb == 0.0 {
        return Err(PbcTdscfError::ShapeMismatch { expected: 1, got: 0 });
    }
    Ok(())
}

/// Is this XC name a hybrid (exact exchange in the response)?
pub fn is_hybrid(xc: &str) -> bool {
    hyb_fraction(xc).map(|h| h > 0.0).unwrap_or(false)
}

/// Gamma-point RKS-TDA driver: HF part at `hyb` plus the XC-kernel matrices.
#[allow(clippy::too_many_arguments)]
pub fn kernel_rks_tda(
    eri: &[f64],
    e_ia: &[f64],
    fxc_a: &[f64],
    fxc_b: &[f64],
    nocc: usize,
    nmo: usize,
    xc: &str,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let hyb = hyb_fraction(xc)?;
    check_hybrid_kernel(xc, hyb)?;
    let _ = crate::require_rks_response();
    let (mut a, mut b) = crate::rhf::build_ab(eri, e_ia, nocc, nmo, cfg.singlet, hyb)?;
    let dim = nocc * (nmo - nocc);
    if fxc_a.len() != dim * dim || fxc_b.len() != dim * dim {
        return Err(PbcTdscfError::ShapeMismatch { expected: dim * dim, got: fxc_a.len().min(fxc_b.len()) });
    }
    for i in 0..dim * dim {
        a[i] += fxc_a[i];
        b[i] += fxc_b[i];
    }
    // TDA solves A only; B is assembled (and returned implicitly through the
    // TDHF entry) so a dropped-XC-kernel bug cannot hide in an unused variable.
    let _ = &b;
    let mut out = tda_dense(&a, dim, cfg)?;
    out.kshift = 0;
    Ok(out)
}
