//! Backward-compatible `pyscf.pbc.ao2mo.eris` convenience wrappers.

use pyscf_algebra::{CTensor, oracle_zdotu};
use pyscf_pbc_df::pbc_ao2mo::{fft_ao_pairs_g, fft_general_mo_first, fft_get_eri};
use pyscf_pbc_df::{Eri, Fftdf, MoCoeff};
use pyscf_pbc_gto::Cell;

use crate::PbcAo2moError;

#[derive(Debug, Clone, PartialEq)]
pub struct PairG {
    pub ngrids: usize,
    pub npair: usize,
    /// Row-major `(G,pair)`.
    pub data: CTensor,
}

/// `compact` is intentionally ignored, exactly as `eris.py:34-39`; upstream's
/// alternative direct `FFTDF.ao2mo` line is commented out.
pub fn general(
    cell: &Cell,
    mos: [&MoCoeff; 4],
    kpts: Option<[[f64; 3]; 4]>,
    _compact: bool,
) -> Result<Eri, PbcAo2moError> {
    get_mo_eri(cell, mos, kpts)
}

pub fn get_mo_eri(
    cell: &Cell,
    mos: [&MoCoeff; 4],
    kpts: Option<[[f64; 3]; 4]>,
) -> Result<Eri, PbcAo2moError> {
    let k = kpts.unwrap_or([[0.0; 3]; 4]);
    let df = Fftdf::new(cell.clone(), &k)?;
    Ok(fft_general_mo_first(&df, mos, k, None)?)
}

pub fn get_mo_pairs_g(
    cell: &Cell,
    mos: [&MoCoeff; 2],
    kpts: Option<[[f64; 3]; 2]>,
    q: Option<[f64; 3]>,
) -> Result<PairG, PbcAo2moError> {
    let k = kpts.unwrap_or([[0.0; 3]; 2]);
    let q = q.unwrap_or([k[1][0] - k[0][0], k[1][1] - k[0][1], k[1][2] - k[0][2]]);
    let df = Fftdf::new(cell.clone(), &k)?;
    let ao = fft_ao_pairs_g(&df, k, q)?;
    let data = pyscf_pbc_df::pbc_ao2mo::get_mo_pairs_g(&ao, cell.mol.nao_nr, mos[0], mos[1]);
    Ok(PairG {
        ngrids: data.len() / (mos[0].nmo * mos[1].nmo),
        npair: mos[0].nmo * mos[1].nmo,
        data,
    })
}

pub fn get_mo_pairs_invg(
    cell: &Cell,
    mos: [&MoCoeff; 2],
    kpts: Option<[[f64; 3]; 2]>,
    q: Option<[f64; 3]>,
) -> Result<PairG, PbcAo2moError> {
    let k = kpts.unwrap_or([[0.0; 3]; 2]);
    let forward = get_mo_pairs_g(
        cell,
        [mos[1], mos[0]],
        Some([k[1], k[0]]),
        q.map(|x| [-x[0], -x[1], -x[2]]),
    )?;
    let mut data = CTensor::zeros(forward.data.len());
    for g in 0..forward.ngrids {
        for i in 0..mos[0].nmo {
            for j in 0..mos[1].nmo {
                let dst = g * (mos[0].nmo * mos[1].nmo) + i * mos[1].nmo + j;
                let src = g * (mos[1].nmo * mos[0].nmo) + j * mos[0].nmo + i;
                data.re[dst] = forward.data.re[src];
                data.im[dst] = -forward.data.im[src];
            }
        }
    }
    Ok(PairG {
        ngrids: forward.ngrids,
        npair: mos[0].nmo * mos[1].nmo,
        data,
    })
}

pub fn assemble_eri(
    cell: &Cell,
    left: &PairG,
    right: &PairG,
    q: Option<[f64; 3]>,
) -> Result<CTensor, PbcAo2moError> {
    if left.ngrids != right.ngrids {
        return Err(PbcAo2moError::Shape("pair grids differ".into()));
    }
    let q = q.unwrap_or([0.0; 3]);
    let coulg = pyscf_pbc_gto::get_coulg(
        cell,
        pyscf_pbc_gto::CoulGArgs {
            k: [-q[0], -q[1], -q[2]],
            ..Default::default()
        },
    )?;
    let scale = cell.vol() / (left.ngrids * left.ngrids) as f64;
    let mut out = CTensor::zeros(left.npair * right.npair);
    for i in 0..left.npair {
        for j in 0..right.npair {
            let mut x = CTensor::zeros(left.ngrids);
            let mut y = CTensor::zeros(left.ngrids);
            for (g, &coulg_g) in coulg.iter().enumerate().take(left.ngrids) {
                let a = g * left.npair + i;
                let b = g * right.npair + j;
                x.re[g] = left.data.re[a];
                x.im[g] = left.data.im[a];
                y.re[g] = right.data.re[b] * coulg_g * scale;
                y.im[g] = right.data.im[b] * coulg_g * scale;
            }
            let (re, im) = oracle_zdotu(&x, &y);
            out.re[i * right.npair + j] = re;
            out.im[i * right.npair + j] = im;
        }
    }
    Ok(out)
}

pub fn get_ao_pairs_g(
    cell: &Cell,
    kpts: Option<[[f64; 3]; 2]>,
) -> Result<(Vec<f64>, Vec<f64>), PbcAo2moError> {
    let k = kpts.unwrap_or([[0.0; 3]; 2]);
    let q = [k[1][0] - k[0][0], k[1][1] - k[0][1], k[1][2] - k[0][2]];
    let df = Fftdf::new(cell.clone(), &k)?;
    Ok(fft_ao_pairs_g(&df, k, q)?)
}

pub fn get_ao_eri(cell: &Cell, kpts: Option<[[f64; 3]; 4]>) -> Result<CTensor, PbcAo2moError> {
    let k = kpts.unwrap_or([[0.0; 3]; 4]);
    let df = Fftdf::new(cell.clone(), &k)?;
    Ok(fft_get_eri(&df, k)?)
}
