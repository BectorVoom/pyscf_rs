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
//! What is NOT here, and why: `project_mo_nr2nr` (basis projection of periodic
//! orbitals) needs `intor_cross` between two DIFFERENT cells, which
//! `pyscf_pbc_gto::intor_cross` supports, but its only consumer is
//! `init_guess_by_chkfile` across a basis change — a Phase 20 drop-in concern
//! rather than a Phase 11 one. It returns `NotYetImplemented { phase: 20 }`.

use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;

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
                e[*a].partial_cmp(&e[*b]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut occ = vec![0.0_f64; e.len()];
            for i in idx.into_iter().take(nocc_per_k) {
                occ[i] = 2.0;
            }
            occ
        })
        .collect()
}

/// `project_mo_nr2nr` for periodic orbitals — deferred.
///
/// # Errors
/// ALWAYS [`PyscfRsError::NotYetImplemented`] `{ phase: 20 }`; see the module
/// docs.
pub fn project_mo_nr2nr() -> Result<(), PyscfRsError> {
    Err(PyscfRsError::NotYetImplemented {
        phase: 20,
        what: "project_mo_nr2nr for periodic orbitals (basis-changing chkfile \
               restart) — pbc/scf/addons.py",
    })
}
