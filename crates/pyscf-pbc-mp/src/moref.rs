//! The single SCF-column-major to DF-row-major MO coefficient seam.

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_ao2mo::MoCoeff;
use pyscf_pbc_scf::KScfResult;

use crate::PbcMpError;

/// `KScfResult.mo_coeff` (column-major `nao x nmo`) to [`MoCoeff`] (row-major).
pub fn mo_coeff_from_kscf(c: &CTensor, nao: usize, nmo: usize) -> Result<MoCoeff, PbcMpError> {
    if c.len() != nao * nmo {
        return Err(PbcMpError::Shape {
            what: format!("MO coefficient length {} != {nao}*{nmo}", c.len()),
        });
    }
    let mut out = CTensor::zeros(c.len());
    for p in 0..nao {
        for i in 0..nmo {
            out.re[p * nmo + i] = c.re[i * nao + p];
            out.im[p * nmo + i] = c.im[i * nao + p];
        }
    }
    Ok(MoCoeff::new(nao, nmo, out))
}

/// MO-column slice `[lo, hi)` of a row-major coefficient block.
pub fn mo_slice(m: &MoCoeff, lo: usize, hi: usize) -> Result<MoCoeff, PbcMpError> {
    if lo >= hi || hi > m.nmo {
        return Err(PbcMpError::Shape {
            what: format!("invalid MO slice [{lo},{hi}) for nmo={}", m.nmo),
        });
    }
    let n = hi - lo;
    let mut out = CTensor::zeros(m.nao * n);
    for p in 0..m.nao {
        out.re[p * n..(p + 1) * n].copy_from_slice(&m.c.re[p * m.nmo + lo..p * m.nmo + hi]);
        out.im[p * n..(p + 1) * n].copy_from_slice(&m.c.im[p * m.nmo + lo..p * m.nmo + hi]);
    }
    Ok(MoCoeff::new(m.nao, n, out))
}

#[derive(Debug, Clone, Copy)]
pub struct KMoRef<'a> {
    pub mo_energy: &'a [Vec<f64>],
    pub mo_coeff: &'a [CTensor],
    pub mo_occ: &'a [Vec<f64>],
}

/// One spin block; KUHF stores blocks at `set * nkpts + k`.
pub fn spin_block(r: &KScfResult, set: usize) -> Result<KMoRef<'_>, PbcMpError> {
    if set >= r.nset {
        return Err(PbcMpError::Shape {
            what: format!("spin set {set} >= {}", r.nset),
        });
    }
    let lo = set * r.nkpts;
    let hi = lo + r.nkpts;
    Ok(KMoRef {
        mo_energy: &r.mo_energy[lo..hi],
        mo_coeff: &r.mo_coeff[lo..hi],
        mo_occ: &r.mo_occ[lo..hi],
    })
}
