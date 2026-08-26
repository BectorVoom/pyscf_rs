//! Periodic initial guesses — `khf.py:345-386` (`_cast_mol_init_guess`,
//! `init_guess_by_minao/atom/chkfile`) and `khf.py:832-852` (`KRHF.get_init_guess`).
//!
//! # `_cast_mol_init_guess` (khf.py:345-362)
//!
//! The periodic MINAO and atomic guesses are NOT new code: upstream builds the
//! MOLECULAR guess on the cell-as-a-molecule and replicates the resulting REAL
//! density matrix to every k-point. This port does exactly that, over
//! `pyscf_scf::default_get_init_guess`, so the two guesses stay bit-identical to
//! the molecular ones the Phase-3 gates already pin.

use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, Density, PyscfRsError};
use pyscf_pbc_gto::Cell;
use pyscf_scf::InitGuessMode;

use crate::krdm::{electron_count, make_rdm1_one};
use crate::types::{KDms, KInitGuess, KMats};

/// `_cast_mol_init_guess(fn)(mf, cell, kpts)` — replicate a real molecular
/// density matrix to every k-point.
pub fn cast_mol_init_guess(dm: &Density, nkpts: usize) -> KMats {
    let c = CTensor::from_real(&dm.data);
    vec![c; nkpts]
}

/// The molecular guess for `mode`, cast to `nkpts` k-points and split over
/// `nset` channels.
///
/// For `nset = 2` (UHF) the molecular restricted density is halved into each
/// spin channel, which is what `mol_hf.SCF.get_init_guess` returns for a UHF
/// object (`init_guess_by_minao` produces `(2, nao, nao)` there); this port
/// takes the restricted density and splits it, which is the same matrix.
///
/// # Errors
/// Propagates the molecular guess; `KInitGuess::Chkfile` propagates the HDF5
/// read.
pub fn get_init_guess(
    cell: &Cell,
    nkpts: usize,
    nset: usize,
    mode: &KInitGuess,
    s1e: &KMats,
    nelectron: f64,
) -> Result<KDms, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let mut dms: KDms = match mode {
        KInitGuess::UserDm(d) => d.clone(),
        KInitGuess::Chkfile(path) => {
            let prior = pyscf_scf::load_scf_from_file(path).map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "periodic init_guess_by_chkfile: {e}"
                )))
            })?;
            // `scf/mo_coeff` in a chkfile written by this crate is the GAMMA
            // block; upstream's `khf.init_guess_by_chkfile` projects the stored
            // k-resolved orbitals when they are there and falls back to the
            // molecular route otherwise. The molecular route is what a
            // molecular chkfile supports, so that is what is implemented.
            let occ = &prior.mo_occ;
            let c = CTensor::from_real(&prior.mo_coeff.data);
            let dm = make_rdm1_one(&c, occ, nao);
            let per_k = vec![dm; nkpts];
            if nset == 1 {
                vec![per_k]
            } else {
                let half: KMats = per_k
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
        }
        other => {
            let mol_mode = match other {
                KInitGuess::Minao => InitGuessMode::Minao,
                KInitGuess::Atom => InitGuessMode::Atom,
                KInitGuess::OneElectron => InitGuessMode::OneElectron,
                // handled above
                KInitGuess::Chkfile(_) | KInitGuess::UserDm(_) => unreachable!(),
            };
            let dm = pyscf_scf::default_get_init_guess(&cell.mol, &mol_mode)?;
            let per_k = cast_mol_init_guess(&dm, nkpts);
            if nset == 1 {
                vec![per_k]
            } else {
                let half: KMats = per_k
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
        }
    };

    // khf.py:838-852 — renormalise when the guess has badly wrong electron
    // count. Upstream's threshold is `0.01 * nkpts` on `Ne` summed over the BZ.
    let ne = electron_count(&dms, s1e, nao);
    if ne.abs() > 1e-12 && (ne - nelectron).abs() > 0.01 * nkpts as f64 {
        tracing::debug!(
            ne_per_cell = ne / nkpts as f64,
            want = nelectron / nkpts as f64,
            "periodic init guess has the wrong electron count; renormalising"
        );
        let s = nelectron / ne;
        for set in dms.iter_mut() {
            for m in set.iter_mut() {
                for v in m.re.iter_mut() {
                    *v *= s;
                }
                for v in m.im.iter_mut() {
                    *v *= s;
                }
            }
        }
    }
    Ok(dms)
}
