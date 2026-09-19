//! Periodic SCF add-ons — plan 11-11, port of `pyscf/pbc/scf/addons.py`.
//!
//! What is here:
//!
//! * `smearing_` — attach Fermi-Dirac or Gaussian smearing to an existing
//!   method object (`addons.py` re-exports `pbc/scf/smearing.py:165-190`);
//! * `convert_to_uhf` / `convert_to_rhf` / `convert_to_ghf` — the density-matrix
//!   and orbital conversions between periodic references
//!   (`addons.py:convert_to_*`);
//! * `canonical_occ_` — the "one electron per k-point band" occupancy variant.
//!
//! * `project_mo_nr2nr` — project periodic orbitals from one cell's basis onto
//!   another's (`addons.py:39-57`), `C2 = S22^-1 <AO2|AO1> C1` per k-point
//!   (plan 20-12).

use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::{Cell, PbcIntorOpts};

use crate::smearing::{Smearing, SmearingMethod};
use crate::types::{KDms, KMats};
use crate::{Krhf, Kuhf};

/// `mf.smearing_(sigma, method)` — attach smearing in place.
pub fn smearing_krhf(mf: &mut Krhf, sigma: f64, method: SmearingMethod) {
    mf.smearing = Some(Smearing {
        sigma,
        method,
        mu0: None,
    });
}

/// `mf.smearing_(sigma, method)` for an unrestricted reference.
pub fn smearing_kuhf(mf: &mut Kuhf, sigma: f64, method: SmearingMethod) {
    mf.smearing = Some(Smearing {
        sigma,
        method,
        mu0: None,
    });
}

/// `convert_to_uhf`'s density half: a restricted `D` becomes `(D/2, D/2)`.
pub fn rhf_dm_to_uhf(dms: &KDms) -> KDms {
    let half: KMats = dms[0]
        .iter()
        .map(|m| {
            CTensor::from_planes(
                m.re.iter().map(|v| v * 0.5).collect(),
                m.im.iter().map(|v| v * 0.5).collect(),
            )
        })
        .collect();
    vec![half.clone(), half]
}

/// `convert_to_rhf`'s density half: `(Da, Db)` becomes `Da + Db`.
pub fn uhf_dm_to_rhf(dms: &KDms) -> KDms {
    vec![
        dms[0]
            .iter()
            .zip(dms[1].iter())
            .map(|(a, b)| {
                let mut m = a.clone();
                for i in 0..m.len() {
                    m.re[i] += b.re[i];
                    m.im[i] += b.im[i];
                }
                m
            })
            .collect(),
    ]
}

/// `convert_to_ghf`'s density half: `(Da, Db)` becomes the block-diagonal
/// spin-orbital density `[[Da, 0], [0, Db]]`.
pub fn uhf_dm_to_ghf(dms: &KDms, nao: usize) -> KDms {
    let nso = 2 * nao;
    let out = dms[0]
        .iter()
        .zip(dms[1].iter())
        .map(|(a, b)| {
            let mut m = CTensor::zeros(nso * nso);
            for i in 0..nao {
                for j in 0..nao {
                    m.re[i * nso + j] = a.re[i * nao + j];
                    m.im[i * nso + j] = a.im[i * nao + j];
                    m.re[(nao + i) * nso + nao + j] = b.re[i * nao + j];
                    m.im[(nao + i) * nso + nao + j] = b.im[i * nao + j];
                }
            }
            m
        })
        .collect();
    vec![out]
}

/// `canonical_occ_(mf)` — `addons.py`'s "fill each k-point independently"
/// occupancy.
///
/// This is deliberately NOT the default: `get_occ` uses ONE Fermi level over
/// the whole Brillouin zone (see `crate::kocc`). `canonical_occ_` restores the
/// per-k filling, which upstream offers for the special case of a system whose
/// band occupations are known to be k-independent — an insulator with an
/// integer band count.
pub fn canonical_occ(mo_energy_kpts: &[Vec<f64>], nocc_per_k: usize) -> Vec<Vec<f64>> {
    mo_energy_kpts
        .iter()
        .map(|e| {
            let mut idx: Vec<usize> = (0..e.len()).collect();
            idx.sort_by(|a, b| {
                e[*a]
                    .partial_cmp(&e[*b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut occ = vec![0.0_f64; e.len()];
            for i in idx.into_iter().take(nocc_per_k) {
                occ[i] = 2.0;
            }
            occ
        })
        .collect()
}

/// `project_mo_nr2nr(cell1, mo1, cell2, kpts)` — `pbc/scf/addons.py:39-57`.
///
/// ```text
/// |psi2> = P |psi1> = |AO2> S22^-1 <AO2|AO1> C1 = |AO2> C2
/// C2[k]  = solve(S22[k], S21[k] . C1[k])
/// ```
///
/// `S22[k]` is `cell2`'s overlap (`pbc_intor('int1e_ovlp', hermi=1)`, i.e.
/// [`pyscf_pbc_gto::get_ovlp`]) and `S21[k]` the cross overlap
/// `intor_cross('int1e_ovlp', cell2, cell1)` (hermi 0, unscreened — upstream's
/// defaults). `mo1[k]` is COLUMN-MAJOR `nao1 x nmo`; the result is COLUMN-MAJOR
/// `nao2 x nmo`, one block per k-point.
///
/// Upstream solves with `scipy.linalg.solve(assume_a='pos')` (Cholesky); this
/// port solves the same system column by column with
/// [`pyscf_algebra::zsolve_linear`] (pivoted LU). The two agree to the
/// conditioning of `S22`, not bitwise.
///
/// # Errors
/// * [`CoreError::InvalidMolecule`] when `mo1.len() != kpts.len()` or a block
///   is not a multiple of `cell1`'s AO count long, or the solve fails;
/// * propagates both overlap integrals.
pub fn project_mo_nr2nr(
    cell1: &Cell,
    mo1: &[CTensor],
    cell2: &Cell,
    kpts: &[[f64; 3]],
) -> Result<Vec<CTensor>, PyscfRsError> {
    let invalid = |msg: String| PyscfRsError::Core(CoreError::InvalidMolecule(msg));
    if mo1.len() != kpts.len() {
        return Err(invalid(format!(
            "project_mo_nr2nr: {} MO blocks for {} k-points",
            mo1.len(),
            kpts.len()
        )));
    }
    let nao1 = cell1.mol.nao_nr;
    let nao2 = cell2.mol.nao_nr;
    // Both integrals come back F-order per k-point.
    let s22 = pyscf_pbc_gto::get_ovlp(cell2, kpts)?;
    let s21 = pyscf_pbc_gto::intor_cross(
        "int1e_ovlp",
        cell2,
        cell1,
        kpts,
        PbcIntorOpts {
            hermi: 0,
            ..Default::default()
        },
    )?;
    if s21.ni != nao2 || s21.nj != nao1 || s21.comp != 1 {
        return Err(invalid(format!(
            "project_mo_nr2nr: cross overlap is {}x{}x{}, expected 1x{nao2}x{nao1}",
            s21.comp, s21.ni, s21.nj
        )));
    }

    let mut out = Vec::with_capacity(kpts.len());
    for (k, c1) in mo1.iter().enumerate() {
        if nao1 == 0 || c1.len() % nao1 != 0 {
            return Err(invalid(format!(
                "project_mo_nr2nr: block {k} has {} elements, not (nao1={nao1}, nmo)",
                c1.len()
            )));
        }
        let nmo = c1.len() / nao1;
        let a = pyscf_pbc_df::zlinalg::forder_to_c(&s22[k], nao2, nao2);
        let x = s21.at(k);
        let mut c2 = CTensor::zeros(nao2 * nmo);
        for m in 0..nmo {
            // b = S21 . C1[:, m]   (S21 F-order: x[i + j * nao2])
            let mut b = CTensor::zeros(nao2);
            for i in 0..nao2 {
                let (mut sr, mut si) = (0.0_f64, 0.0_f64);
                for j in 0..nao1 {
                    let (xr, xi) = (x.re[i + j * nao2], x.im[i + j * nao2]);
                    let (cr, ci) = (c1.re[j + m * nao1], c1.im[j + m * nao1]);
                    sr += xr * cr - xi * ci;
                    si += xr * ci + xi * cr;
                }
                b.re[i] = sr;
                b.im[i] = si;
            }
            let z = pyscf_algebra::zsolve_linear(&a, &b, nao2).map_err(|e| {
                invalid(format!(
                    "project_mo_nr2nr: overlap solve failed at k = {k}, mo = {m}: {e}"
                ))
            })?;
            c2.re[m * nao2..(m + 1) * nao2].copy_from_slice(&z.re);
            c2.im[m * nao2..(m + 1) * nao2].copy_from_slice(&z.im);
        }
        out.push(c2);
    }
    Ok(out)
}
