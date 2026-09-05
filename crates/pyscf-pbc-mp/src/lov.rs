//! Density-fitted occupied-virtual three-index tensor (`kmp2.py:156-227`).

use pyscf_algebra::CTensor;
use pyscf_pbc_df::{MoCoeff, PeriodicDf, df_ao2mo::r_e2};
use rayon::prelude::*;

use crate::{PbcMpError, mo_slice};

/// `Lov[ki,kj]`; each block is row-major `(nocc*nvir, naux)`, so `L` is the
/// fastest axis used by KMP2's ordered inner products.
#[derive(Debug, Clone)]
pub struct LovTable {
    pub nkpts: usize,
    pub nocc: usize,
    pub nvir: usize,
    pub blocks: Vec<(usize, CTensor)>,
}

impl LovTable {
    pub fn block(&self, ki: usize, kj: usize) -> &(usize, CTensor) {
        &self.blocks[ki * self.nkpts + kj]
    }

    pub fn aux_slice(&self, ki: usize, kj: usize, i: usize, a: usize) -> &[f64] {
        let (naux, block) = self.block(ki, kj);
        let lo = (i * self.nvir + a) * *naux;
        &block.re[lo..lo + *naux]
    }
}

/// `_init_mp_df_eris` (`pyscf/pbc/mp/kmp2.py:156-227`).
///
/// DEVIATION from line 190: storage is transposed once from `(L,i,a)` to
/// `(i,a,L)`, keeping every later auxiliary contraction contiguous.
pub fn build_lov(
    df: &dyn PeriodicDf,
    mo_coeff: &[MoCoeff],
    nocc: usize,
) -> Result<LovTable, PbcMpError> {
    if !df.has_cderi() {
        return Err(PbcMpError::Shape {
            what: format!("{} is not a cderi-backed builder", df.name()),
        });
    }
    if df.cell().dimension == 2 {
        return Err(PbcMpError::Shape {
            what: "_init_mp_df_eris is unavailable for dimension == 2 (upstream kmp2.py:145-153)"
                .into(),
        });
    }
    let nk = df.kpts().len();
    if mo_coeff.len() != nk || mo_coeff.is_empty() {
        return Err(PbcMpError::Shape {
            what: "MO blocks do not match sampling k-points".into(),
        });
    }
    let nmo = mo_coeff[0].nmo;
    if nocc == 0 || nocc >= nmo || mo_coeff.iter().any(|m| m.nmo != nmo) {
        return Err(PbcMpError::Shape {
            what: "Lov needs uniform non-empty occupied and virtual MO spaces".into(),
        });
    }
    let nvir = nmo - nocc;
    let blocks: Result<Vec<_>, PbcMpError> = (0..nk * nk)
        .into_par_iter()
        .map(|flat| {
            let (ki, kj) = (flat / nk, flat % nk);
            let occ = mo_slice(&mo_coeff[ki], 0, nocc)?;
            let vir = mo_slice(&mo_coeff[kj], nocc, nmo)?;
            let sr = df.sr_loop(ki, kj, false)?;
            if sr.iter().any(|b| b.sign != 1) {
                return Err(PbcMpError::Shape {
                    what: "negative cderi block is unsupported in KMP2".into(),
                });
            }
            let naux: usize = sr.iter().map(|b| b.naux).sum();
            let npair = nocc * nvir;
            let mut out = CTensor::zeros(npair * naux);
            let mut l0 = 0;
            for blk in &sr {
                // DEVIATION from upstream's hstack: r_e2 accepts the two MO
                // slices directly and performs identical arithmetic.
                let z = r_e2(blk, mo_coeff[ki].nao, &occ, &vir);
                for l in 0..blk.naux {
                    for ia in 0..npair {
                        out.re[ia * naux + l0 + l] = z.re[l * npair + ia];
                        out.im[ia * naux + l0 + l] = z.im[l * npair + ia];
                    }
                }
                l0 += blk.naux;
            }
            Ok((naux, out))
        })
        .collect();
    Ok(LovTable {
        nkpts: nk,
        nocc,
        nvir,
        blocks: blocks?,
    })
}
