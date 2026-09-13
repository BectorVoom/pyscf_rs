//! Gamma-point unrestricted Kohn-Sham TDA/TDDFT (`pbc/tdscf/uks.py`, 49 l).
//!
//! Thin subclass over the 19-08 coupled driver: spin-resolved HF parts at
//! `hyb` plus spin-resolved XC-kernel blocks,
//!
//! * `A = A_UHF(hyb) + diag-block(fxc_aa, fxc_bb)` with off-diagonal `fxc_ab`
//!   in the coupling position (same `(da+db)²` layout as
//!   [`crate::uhf::build_uab`]).
//!
//! Same seam discipline as [`crate::rks`] (`gen_response` only on the
//! concrete class; hybrid must not take the pure path).

use crate::error::PbcTdscfError;
use crate::rks::{check_hybrid_kernel, hyb_fraction};
use crate::types::{TdaConfig, TdaResult};
use crate::uhf::{UhfDims, build_uab};

/// Gamma-point UKS-TDA driver.
#[allow(clippy::too_many_arguments)]
pub fn kernel_uks_tda(
    eri_aa: &[f64],
    eri_ab: &[f64],
    eri_bb: &[f64],
    e_ia_a: &[f64],
    e_ia_b: &[f64],
    fxc_aa: &[f64],
    fxc_ab: &[f64],
    fxc_bb: &[f64],
    dims: UhfDims,
    nmo_a: usize,
    nmo_b: usize,
    xc: &str,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let hyb = hyb_fraction(xc)?;
    check_hybrid_kernel(xc, hyb)?;
    let _ = crate::require_rks_response();
    let (mut a, _b) = build_uab(eri_aa, eri_ab, eri_bb, e_ia_a, e_ia_b, dims, nmo_a, nmo_b, hyb)?;
    let (da, db) = (dims.da(), dims.db());
    let dim = da + db;
    for (blk, base, n) in [(fxc_aa, 0, da), (fxc_bb, da * dim + da, db)] {
        if blk.len() != n * n {
            return Err(PbcTdscfError::ShapeMismatch { expected: n * n, got: blk.len() });
        }
        let stride = dim;
        for i in 0..n {
            for j in 0..n {
                let r = base / dim + i;
                let c = (base % dim) + j;
                a[r * stride + c] += blk[i * n + j];
            }
        }
    }
    if fxc_ab.len() != da * db {
        return Err(PbcTdscfError::ShapeMismatch { expected: da * db, got: fxc_ab.len() });
    }
    for i in 0..da {
        for j in 0..db {
            a[i * dim + da + j] += fxc_ab[i * db + j];
            a[(da + j) * dim + i] += fxc_ab[i * db + j];
        }
    }
    let mut out = crate::davidson::tda_dense(&a, dim, cfg)?;
    out.kshift = 0;
    Ok(out)
}
