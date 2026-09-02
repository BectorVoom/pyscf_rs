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

use crate::krdm::{electron_count_per_set, make_rdm1_one};
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
/// For `nset = 2` the molecular restricted density is halved into each spin
/// channel and then handed to the symmetry break — `uhf.py:855-863` does
/// exactly that (`dma = dmb = dm*.5`, then `_break_dm_spin_symm`), and
/// `kuhf.py:421-425` inherits those methods verbatim.
///
/// `nelec` carries one target electron count per channel, over the WHOLE BZ
/// (the same scale `s1e` traces on): `[Ne]` for `nset = 1`, `[Nalpha, Nbeta]`
/// for `nset = 2`. `breaksym` is the driver's `init_guess_breaksym`
/// (`uhf.py:778`, `kuhf.py:417` — upstream's default is `1`); pass `0` for a
/// restricted or restricted-open driver, which never breaks the symmetry.
///
/// # Errors
/// Propagates the molecular guess; `KInitGuess::Chkfile` propagates the HDF5
/// read; the `breaksym` branches propagate `aoslice_by_atom` / `int1e_ovlp`.
pub fn get_init_guess(
    cell: &Cell,
    nkpts: usize,
    nset: usize,
    mode: &KInitGuess,
    s1e: &KMats,
    nelec: &[f64],
    breaksym: i32,
) -> Result<KDms, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    if nelec.len() != nset {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "periodic init guess: nelec has {} entries but nset = {nset}",
            nelec.len()
        ))));
    }
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
            //
            // No symmetry break here, deliberately: upstream's
            // `kuhf.init_guess_by_chkfile` (kuhf.py:100-114) reads a stored
            // UNRESTRICTED pair and never calls `_break_dm_spin_symm` on it.
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
            if nset == 1 {
                vec![cast_mol_init_guess(&dm, nkpts)]
            } else {
                // uhf.py:859 — `dma = dmb = dm*.5`, THEN break.
                let half: Vec<f64> = dm.data.iter().map(|v| v * 0.5).collect();
                let (dma, dmb) = break_for_mode(cell, &mol_mode, &half, breaksym)?;
                vec![
                    cast_mol_init_guess(&Density { nao, data: dma }, nkpts),
                    cast_mol_init_guess(&Density { nao, data: dmb }, nkpts),
                ]
            }
        }
    };

    // `kuhf.py:476-486` (and `khf.py:838-852` for the restricted case) —
    // renormalise when the guess has a badly wrong electron count.
    //
    // U-02: the count and the scale are BOTH per channel. `electron_count`
    // used to sum the two spin channels into one `f64` and apply one factor
    // `Ne / ne_total` to both, which
    //   * cannot restore `(nalpha, nbeta)` on a `cell.spin != 0` cell — and
    //     since `_break_dm_spin_symm` short-circuits on `spin == 0`, this
    //     renormalisation is the ONLY thing that polarises the minao guess
    //     for an open-shell cell; and
    //   * fired at half upstream's threshold on a closed-shell one, because
    //     `ne_total - Ne = 2 (ne_a - nalpha)` when the channels are equal.
    // Upstream's `np.any` semantics are kept literally: if ANY channel is off
    // by more than `0.01 * nkpts`, EVERY channel is rescaled by its own factor.
    let ne = electron_count_per_set(&dms, s1e, nao);
    let fire = ne
        .iter()
        .zip(nelec)
        .any(|(got, want)| got.abs() > 1e-12 && (got - want).abs() > 0.01 * nkpts as f64);
    if fire {
        tracing::debug!(
            ne_per_cell = ?ne.iter().map(|v| v / nkpts as f64).collect::<Vec<_>>(),
            want = ?nelec.iter().map(|v| v / nkpts as f64).collect::<Vec<_>>(),
            "periodic init guess has the wrong electron count; renormalising"
        );
        for (set, (got, want)) in dms.iter_mut().zip(ne.iter().zip(nelec)) {
            if got.abs() <= 1e-12 {
                continue;
            }
            let s = want / got;
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

/// Dispatch the spin-symmetry break for the guess `mode`.
///
/// Upstream uses TWO different schemes and picks by mode, so this does too
/// (RULE 2 — port, do not invent):
///
/// * `minao` (`uhf.py:855-863`) and `1e` (`uhf.py:906-920`) →
///   [`pyscf_scf::break_dm_spin_symm`].
/// * `atom` (`uhf.py:864-877`) → [`pyscf_scf::break_atom_guess_spin_symm`],
///   which breaks the ALPHA channel against `1e-2 * S` instead.
///
/// One knowing divergence, recorded rather than hidden: upstream's UHF `1e`
/// guess diagonalises `hcore` with the UNRESTRICTED occupation
/// `(nalpha, nbeta)`, so on a `spin != 0` cell its two channels already differ
/// before any break; this port halves the RESTRICTED `hcore` density. The break
/// itself is unaffected — it is a no-op at `spin != 0` on both sides.
fn break_for_mode(
    cell: &Cell,
    mode: &InitGuessMode,
    half: &[f64],
    breaksym: i32,
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    match mode {
        InitGuessMode::Atom => {
            pyscf_scf::break_atom_guess_spin_symm(&cell.mol, half, half, breaksym)
        }
        _ => pyscf_scf::break_dm_spin_symm(&cell.mol, half, half, breaksym),
    }
}
