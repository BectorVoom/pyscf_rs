//! `Mole.aoslice_by_atom()` — the per-atom AO range.
//!
//! Upstream `pyscf/gto/mole.py:aoslice_by_atom` returns an `(natm, 4)` integer
//! table whose last two columns are the AO range `[p0, p1)` of each atom. Every
//! consumer in this workspace wants exactly those two columns, so this returns
//! them directly.
//!
//! # Why it lives here
//!
//! It is a property of the `Mole` layout and nothing else — it reads `_bas`'s
//! `ATOM_OF` slot and `ao_loc_nr`, neither of which any downstream crate owns.
//! Three independent copies had accumulated (`pyscf-scf::init_guess`,
//! `pyscf-grad::rhf`, and a `Cell`-shaped one in `pyscf-pbc-symm`); the first
//! two are now thin re-exports of this function, which is what lets
//! `pyscf-pbc-scf` reach it without depending on `pyscf-grad` (the SCF crates
//! sit BELOW gradients — see `KUKS-OPTIMISATION-PLAN.md` U-02 step 5).

use pyscf_core::raw_layout::{ATOM_OF, BAS_SLOTS};
use pyscf_core::{CoreError, Mole, PyscfRsError};

/// Per-atom AO range `[lo, hi)` — `mol.aoslice_by_atom()[ia, 2:]`.
///
/// For each atom walks the `mol._bas` rows whose `ATOM_OF` slot names it and
/// unions their `ao_loc_nr[shell]..ao_loc_nr[shell+1]` ranges. Shells are
/// atom-ordered after `build`, so each atom's AO block is contiguous. An atom
/// carrying no basis function gets an empty `[lo, lo)` anchored at the running
/// offset, matching upstream's zero-width slice.
///
/// # Errors
/// An unbuilt `Mole` (`ao_loc_nr` shorter than `nbas + 1`), an `ATOM_OF` slot
/// out of range, or an AO range outside `[0, nao]` — never a panic
/// (T-03-14-PANIC).
pub fn aoslice_by_atom(mol: &Mole) -> Result<Vec<(usize, usize)>, PyscfRsError> {
    let natm = mol.natm;
    let nbas = mol.nbas;
    let nao = mol.nao_nr;
    if mol.ao_loc_nr.len() <= nbas {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "aoslice_by_atom: ao_loc_nr len {} <= nbas {} (Mole not built?)",
            mol.ao_loc_nr.len(),
            nbas
        ))));
    }
    // Per-atom [lo, hi), seeded with the sentinel (nao, 0) so an untouched atom
    // is distinguishable from a genuine [0, 0).
    let mut slices = vec![(nao, 0usize); natm];
    for shell in 0..nbas {
        let atom = mol._bas[shell * BAS_SLOTS + ATOM_OF] as usize;
        if atom >= natm {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "aoslice_by_atom: _bas[{shell}, ATOM_OF] = {atom} but natm = {natm}"
            ))));
        }
        let lo = mol.ao_loc_nr[shell] as usize;
        let hi = mol.ao_loc_nr[shell + 1] as usize;
        if hi > nao || lo > hi {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "aoslice_by_atom: shell {shell} AO range [{lo},{hi}) invalid (nao={nao})"
            ))));
        }
        let (cur_lo, cur_hi) = slices[atom];
        slices[atom] = (cur_lo.min(lo), cur_hi.max(hi));
    }
    let mut next = 0usize;
    for slot in slices.iter_mut() {
        if slot.0 == nao && slot.1 == 0 {
            *slot = (next, next);
        } else {
            next = slot.1;
        }
    }
    Ok(slices)
}
