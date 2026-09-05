//! K-dependent occupied/virtual padding (`pyscf/pbc/mp/kmp2.py:228-374`).

use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_ao2mo::MoCoeff;

use crate::{FrozenK, KCount, PbcMpError, get_frozen_mask, get_nmo, get_nocc};

#[derive(Debug, Clone)]
pub struct PaddedMos {
    pub mo_coeff: Vec<MoCoeff>,
    pub mo_energy: Vec<Vec<f64>>,
    pub nmo_per_kpt: Vec<usize>,
    pub nocc_per_kpt: Vec<usize>,
    pub nmo: usize,
    pub nocc: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingKind {
    Split,
    Joint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingIdx {
    Split {
        occupied: Vec<Vec<usize>>,
        virtuals: Vec<Vec<usize>>,
    },
    Joint(Vec<Vec<usize>>),
}

/// Occupied states are bottom-aligned; virtual states are top-aligned. Padding
/// therefore sits at the Fermi level (`kmp2.py:276-315`).
pub fn padding_k_idx(
    nmo: &[usize],
    nocc: &[usize],
    kind: PaddingKind,
) -> Result<PaddingIdx, PbcMpError> {
    if nmo.len() != nocc.len() || nmo.iter().zip(nocc).any(|(m, o)| o > m) {
        return Err(PbcMpError::Shape {
            what: "ragged or invalid nmo/nocc".into(),
        });
    }
    let dense_o = nocc.iter().copied().max().unwrap_or(0);
    let nvir: Vec<_> = nmo.iter().zip(nocc).map(|(m, o)| m - o).collect();
    let dense_v = nvir.iter().copied().max().unwrap_or(0);
    let mut occupied = Vec::with_capacity(nmo.len());
    let mut virtuals = Vec::with_capacity(nmo.len());
    let mut joint = Vec::with_capacity(nmo.len());
    for (&o, &v) in nocc.iter().zip(&nvir) {
        let oi: Vec<_> = (0..o).collect();
        let vi: Vec<_> = (dense_v - v..dense_v).collect();
        let mut ji = oi.clone();
        ji.extend(dense_o + dense_v - v..dense_o + dense_v);
        occupied.push(oi);
        virtuals.push(vi);
        joint.push(ji);
    }
    Ok(match kind {
        PaddingKind::Split => PaddingIdx::Split { occupied, virtuals },
        PaddingKind::Joint => PaddingIdx::Joint(joint),
    })
}

pub fn padded_mo_energy(
    mo_energy: &[Vec<f64>],
    masks: &[Vec<bool>],
    joint: &[Vec<usize>],
    nmo: usize,
) -> Result<Vec<Vec<f64>>, PbcMpError> {
    if mo_energy.len() != masks.len() || masks.len() != joint.len() {
        return Err(PbcMpError::Shape {
            what: "energy/mask/padding k-point counts differ".into(),
        });
    }
    let mut out = vec![vec![0.0; nmo]; mo_energy.len()];
    for k in 0..out.len() {
        let active: Vec<_> = mo_energy[k]
            .iter()
            .zip(&masks[k])
            .filter_map(|(&e, &keep)| keep.then_some(e))
            .collect();
        if active.len() != joint[k].len() {
            return Err(PbcMpError::Shape {
                what: format!("active energies and padding differ at k={k}"),
            });
        }
        for (&dst, &e) in joint[k].iter().zip(&active) {
            out[k][dst] = e;
        }
    }
    Ok(out)
}

pub fn padded_mo_coeff(
    mo_coeff: &[MoCoeff],
    masks: &[Vec<bool>],
    joint: &[Vec<usize>],
    nmo: usize,
) -> Result<Vec<MoCoeff>, PbcMpError> {
    if mo_coeff.len() != masks.len() || masks.len() != joint.len() {
        return Err(PbcMpError::Shape {
            what: "coefficient/mask/padding k-point counts differ".into(),
        });
    }
    mo_coeff
        .iter()
        .enumerate()
        .map(|(k, m)| {
            if m.nmo != masks[k].len() {
                return Err(PbcMpError::Shape {
                    what: format!("MO mask length differs at k={k}"),
                });
            }
            let src: Vec<_> = masks[k]
                .iter()
                .enumerate()
                .filter_map(|(i, &keep)| keep.then_some(i))
                .collect();
            if src.len() != joint[k].len() {
                return Err(PbcMpError::Shape {
                    what: format!("active coefficients and padding differ at k={k}"),
                });
            }
            let mut c = CTensor::zeros(m.nao * nmo);
            for p in 0..m.nao {
                for (&i, &dst) in src.iter().zip(&joint[k]) {
                    c.re[p * nmo + dst] = m.c.re[p * m.nmo + i];
                    c.im[p * nmo + dst] = m.c.im[p * m.nmo + i];
                }
            }
            Ok(MoCoeff::new(m.nao, nmo, c))
        })
        .collect()
}

/// Upstream `_add_padding`: apply frozen masks and place virtual padding next
/// to the Fermi level before the energy denominator is formed.
pub fn add_padding(
    mo_coeff: &[MoCoeff],
    mo_energy: &[Vec<f64>],
    mo_occ: &[Vec<f64>],
    frozen: &FrozenK,
) -> Result<PaddedMos, PbcMpError> {
    if mo_coeff.len() != mo_energy.len() || mo_energy.len() != mo_occ.len() {
        return Err(PbcMpError::Shape {
            what: "MO coefficient/energy/occupation k-point counts differ".into(),
        });
    }
    let nmo_per_kpt = match get_nmo(mo_occ, frozen, true)? {
        KCount::PerKpoint(v) => v,
        KCount::Dense(_) => unreachable!(),
    };
    let nocc_per_kpt = match get_nocc(mo_occ, frozen, true)? {
        KCount::PerKpoint(v) => v,
        KCount::Dense(_) => unreachable!(),
    };
    let nmo = match get_nmo(mo_occ, frozen, false)? {
        KCount::Dense(v) => v,
        KCount::PerKpoint(_) => unreachable!(),
    };
    let nocc = match get_nocc(mo_occ, frozen, false)? {
        KCount::Dense(v) => v,
        KCount::PerKpoint(_) => unreachable!(),
    };
    let masks = get_frozen_mask(mo_occ, frozen)?;
    let joint = match padding_k_idx(&nmo_per_kpt, &nocc_per_kpt, PaddingKind::Joint)? {
        PaddingIdx::Joint(v) => v,
        PaddingIdx::Split { .. } => unreachable!(),
    };
    Ok(PaddedMos {
        mo_coeff: padded_mo_coeff(mo_coeff, &masks, &joint, nmo)?,
        mo_energy: padded_mo_energy(mo_energy, &masks, &joint, nmo)?,
        nmo_per_kpt,
        nocc_per_kpt,
        nmo,
        nocc,
    })
}
