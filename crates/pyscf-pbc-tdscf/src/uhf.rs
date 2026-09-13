//! Gamma-point unrestricted TDA/TDHF (`pbc/tdscf/uhf.py`, 268 l).
//!
//! The alpha and beta rotation spaces COUPLE through the Coulomb kernel —
//! this is one `(da+db)` problem, never two independent restricted ones:
//!
//! * `A = [[A_aa, A_ab],[A_abᵀ, A_bb]]` with `A_aa = diag(e_ia_a) + J_aa −
//!   hyb·K_aa`, `A_bb` likewise, `A_ab = J_ab` (Coulomb only, no exchange);
//! * `B` likewise with the `'jaib'`-layout exchange.
//!
//! (No singlet doubling: the factor 2 is a restricted-singlet feature.)
//! Spin contamination is REPORTED (`<S²>` diagnostic in the result), never
//! filtered — upstream does not filter either.

use crate::davidson::tda_dense;
use crate::error::PbcTdscfError;
use crate::types::{TdaConfig, TdaResult};

/// Unrestricted dimensions.
#[derive(Debug, Clone, Copy)]
pub struct UhfDims {
    /// Alpha occupied count.
    pub nocc_a: usize,
    /// Alpha virtual count.
    pub nvir_a: usize,
    /// Beta occupied count.
    pub nocc_b: usize,
    /// Beta virtual count.
    pub nvir_b: usize,
}

impl UhfDims {
    /// Alpha rotation dimension.
    pub fn da(&self) -> usize {
        self.nocc_a * self.nvir_a
    }
    /// Beta rotation dimension.
    pub fn db(&self) -> usize {
        self.nocc_b * self.nvir_b
    }
}

/// Build the coupled `(da+db)²` UHF `A`/`B` (row-major, alpha block first).
///
/// `eri_aa` is `(nocc_a,nmo_a,nmo_a,nmo_a)` `[i][p][q][r]`
/// (`ab`: `(nocc_a,nmo_a,nmo_b,nmo_b)`, `bb`: `(nocc_b,nmo_b,nmo_b,nmo_b)`):
/// `A_aa[ia][jb] = δ·e + (ia|bj) − hyb·(ij|ba)`,
/// `B_aa[ia][jb] = (ia|jb) − hyb·(ja|ib)` (`'jaib'` layout),
/// `A_ab[ia][jb] = B_ab[ia][jb] = (ia|bj)` (Coulomb only).
/// `e_ia_a`/`e_ia_b` are `(i,a)`-major gap lists.
#[allow(clippy::too_many_arguments)]
pub fn build_uab(
    eri_aa: &[f64],
    eri_ab: &[f64],
    eri_bb: &[f64],
    e_ia_a: &[f64],
    e_ia_b: &[f64],
    dims: UhfDims,
    nmo_a: usize,
    nmo_b: usize,
    hyb: f64,
) -> Result<(Vec<f64>, Vec<f64>), PbcTdscfError> {
    let (noa, nva, nob, nvb) = (dims.nocc_a, dims.nvir_a, dims.nocc_b, dims.nvir_b);
    let (da, db) = (dims.da(), dims.db());
    if eri_aa.len() != noa * nmo_a * nmo_a * nmo_a
        || eri_ab.len() != noa * nmo_a * nmo_b * nmo_b
        || eri_bb.len() != nob * nmo_b * nmo_b * nmo_b
        || e_ia_a.len() != da
        || e_ia_b.len() != db
    {
        return Err(PbcTdscfError::ShapeMismatch { expected: da + db, got: e_ia_a.len() + e_ia_b.len() });
    }
    let eaa = |i: usize, p: usize, q: usize, r: usize| -> f64 {
        eri_aa[((i * nmo_a + p) * nmo_a + q) * nmo_a + r]
    };
    let eab = |i: usize, p: usize, q: usize, r: usize| -> f64 {
        eri_ab[((i * nmo_a + p) * nmo_b + q) * nmo_b + r]
    };
    let ebb = |i: usize, p: usize, q: usize, r: usize| -> f64 {
        eri_bb[((i * nmo_b + p) * nmo_b + q) * nmo_b + r]
    };
    let dim = da + db;
    let mut a = vec![0.0f64; dim * dim];
    let mut b = vec![0.0f64; dim * dim];
    // Alpha-alpha block: J is 'iabj' ([i][a][b][j]) − hyb·K 'ijba'; B exchange is 'jaib'.
    for i in 0..noa {
        for aj in 0..nva {
            let ia = i * nva + aj;
            for j in 0..noa {
                for bj in 0..nva {
                    let jb = j * nva + bj;
                    a[ia * dim + jb] = eaa(i, noa + aj, noa + bj, j)
                        - hyb * eaa(i, j, noa + bj, noa + aj);
                    b[ia * dim + jb] = eaa(i, noa + aj, j, noa + bj)
                        - hyb * eaa(j, noa + aj, i, noa + bj);
                }
            }
            a[ia * dim + ia] += e_ia_a[ia];
        }
    }
    // Alpha-beta coupling (Coulomb only, 'iabj' over the mixed block:
    // [i][a][b][j]) + symmetric transpose.
    for i in 0..noa {
        for aj in 0..nva {
            let ia = i * nva + aj;
            for j in 0..nob {
                for bj in 0..nvb {
                    let jb = da + j * nvb + bj;
                    let v = eab(i, noa + aj, nob + bj, j);
                    a[ia * dim + jb] = v;
                    a[jb * dim + ia] = v;
                    b[ia * dim + jb] = v;
                    b[jb * dim + ia] = v;
                }
            }
        }
    }
    // Beta-beta block ('iabj' J layout as alpha).
    for i in 0..nob {
        for aj in 0..nvb {
            let ia = da + i * nvb + aj;
            for j in 0..nob {
                for bj in 0..nvb {
                    let jb = da + j * nvb + bj;
                    a[ia * dim + jb] = ebb(i, nob + aj, nob + bj, j)
                        - hyb * ebb(i, j, nob + bj, nob + aj);
                    b[ia * dim + jb] = ebb(i, nob + aj, j, nob + bj)
                        - hyb * ebb(j, nob + aj, i, nob + bj);
                }
            }
            a[ia * dim + ia] += e_ia_b[ia - da];
        }
    }
    Ok((a, b))
}

/// Gamma-point UHF-TDA driver over the coupled matrix.
#[allow(clippy::too_many_arguments)]
pub fn kernel_uhf_tda(
    eri_aa: &[f64],
    eri_ab: &[f64],
    eri_bb: &[f64],
    e_ia_a: &[f64],
    e_ia_b: &[f64],
    dims: UhfDims,
    nmo_a: usize,
    nmo_b: usize,
    hyb: f64,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let (a, _b) =
        build_uab(eri_aa, eri_ab, eri_bb, e_ia_a, e_ia_b, dims, nmo_a, nmo_b, hyb)?;
    let mut out = tda_dense(&a, dims.da() + dims.db(), cfg)?;
    out.kshift = 0;
    Ok(out)
}

/// Gamma-point UHF-TDHF driver (shared symmetric core, distinct entry point).
#[allow(clippy::too_many_arguments)]
pub fn kernel_uhf_tdhf(
    eri_aa: &[f64],
    eri_ab: &[f64],
    eri_bb: &[f64],
    e_ia_a: &[f64],
    e_ia_b: &[f64],
    dims: UhfDims,
    nmo_a: usize,
    nmo_b: usize,
    hyb: f64,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let (a, b) =
        build_uab(eri_aa, eri_ab, eri_bb, e_ia_a, e_ia_b, dims, nmo_a, nmo_b, hyb)?;
    crate::rhf::symm_tdhf(&a, &b, dims.da() + dims.db(), cfg.nroots)
}
