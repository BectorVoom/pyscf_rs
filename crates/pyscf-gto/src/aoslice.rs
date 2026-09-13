//! `Mole.aoslice_by_atom()` — the per-atom AO range.
//!
//! Upstream `pyscf/gto/mole.py:aoslice_by_atom` returns an `(natm, 4)` integer
//! table `(shl0, shl1, p0, p1)` whose last two columns are the AO range
//! `[p0, p1)` of each atom.  Periodic gradients also require the shell bounds,
//! so this is deliberately the upstream-shaped four-tuple rather than a
//! lossy AO-only projection.
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

/// Validated shell-to-atom mapping, in the atom-sorted order produced by build.
pub fn shell_atoms(mol: &Mole) -> Result<Vec<usize>, PyscfRsError> {
    let expected = mol
        .nbas
        .checked_mul(BAS_SLOTS)
        .ok_or_else(|| CoreError::InvalidMolecule("shell_atoms: shell count overflow".into()))?;
    if mol._bas.len() < expected {
        return Err(CoreError::InvalidMolecule("shell_atoms: truncated _bas".into()).into());
    }
    let atoms: Vec<usize> = (0..mol.nbas)
        .map(|s| mol._bas[s * BAS_SLOTS + ATOM_OF] as usize)
        .collect();
    if atoms.iter().any(|&a| a >= mol.natm) || atoms.windows(2).any(|w| w[0] > w[1]) {
        return Err(CoreError::InvalidMolecule(
            "shell_atoms: invalid or non-atom-sorted shell ownership".into(),
        )
        .into());
    }
    Ok(atoms)
}

/// Per-atom `(shl0, shl1, p0, p1)` slice — `mol.aoslice_by_atom()[ia]`.
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
pub fn aoslice_by_atom(mol: &Mole) -> Result<Vec<(usize, usize, usize, usize)>, PyscfRsError> {
    let natm = mol.natm;
    let nbas = mol.nbas;
    let nao = mol.nao_nr;
    let atoms = shell_atoms(mol)?;
    if mol.ao_loc_nr.len() <= nbas {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "aoslice_by_atom: ao_loc_nr len {} <= nbas {} (Mole not built?)",
            mol.ao_loc_nr.len(),
            nbas
        ))));
    }
    // Per-atom shell/AO bounds, seeded with sentinels so an untouched atom is
    // distinguishable from a genuine zero-width range.
    let mut slices = vec![(nbas, 0usize, nao, 0usize); natm];
    for shell in 0..nbas {
        let atom = atoms[shell];
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
        let (cur_shl_lo, cur_shl_hi, cur_lo, cur_hi) = slices[atom];
        slices[atom] = (
            cur_shl_lo.min(shell),
            cur_shl_hi.max(shell + 1),
            cur_lo.min(lo),
            cur_hi.max(hi),
        );
    }
    let mut next_shl = 0usize;
    let mut next_ao = 0usize;
    for slot in slices.iter_mut() {
        if slot.0 == nbas && slot.1 == 0 {
            *slot = (next_shl, next_shl, next_ao, next_ao);
        } else {
            next_shl = slot.1;
            next_ao = slot.3;
        }
    }
    Ok(slices)
}
