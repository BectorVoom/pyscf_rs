//! MP2-08: the five helper functions CCSD imports verbatim from
//! `pyscf.mp.mp2` (`cc/ccsd.py:35`):
//!
//! ```python
//! from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
//! ```
//!
//! Upstream `pyscf/mp/mp2.py` defines these on the `mp` instance
//! (`get_nocc(mp)` etc.). In Rust we expose them as free functions over the
//! underlying data (the MO occupation vector + frozen spec) so both MP2 and
//! CCSD can call them without a shared instance type. The Python `_mo_without_core`
//! maps to the Rust `mo_without_core` (Rust has no leading-underscore privacy
//! convention; the doc records the mapping).
//!
//! The helpers take a [`crate::frozen::Frozen`] spec (mirroring upstream
//! `mp.frozen`). NOTE on `Frozen::Auto`: chemcore needs the molecule's per-atom
//! atomic numbers, which these data-only helpers do not carry — so through this
//! helper surface `Frozen::Auto` resolves with NO element info (count 0). The
//! RMP2 kernel's `default_ao2mo` resolves `Auto` with the real `Mole` charges
//! (`refr.mol.atom_charges()`) via [`crate::frozen::frozen_mask`] directly. The
//! CCSD import contract only exercises `None` / `Count` / `List` here.

use crate::error::Mp2Error;
use crate::frozen::{self, Frozen};
use pyscf_core::MOCoefficients;

/// Number of occupied orbitals (counting doubly-occupied as 1 each), minus the
/// frozen-occupied count. Upstream: `pyscf/mp/mp2.py:get_nocc` (line 351) —
/// `count_nonzero(mo_occ > 0)` minus the frozen-occupied orbitals.
///
/// Computed as `count(mo_occ[i] > 0 AND active[i])`: an occupied orbital counts
/// only if it is also active (not frozen). This matches upstream's
/// `occ_idx = mo_occ > 0; occ_idx[frozen] = False; count_nonzero(occ_idx)`.
///
/// # Errors
/// Propagates [`Mp2Error::ShapeMismatch`] from the mask resolution when a
/// frozen index is out of range.
pub fn get_nocc(mo_occ: &[f64], frozen: &Frozen) -> Result<usize, Mp2Error> {
    let mask = frozen::frozen_mask(frozen, mo_occ, &[], &[])?;
    let nocc = mo_occ
        .iter()
        .zip(mask.iter())
        .filter(|&(&occ, &active)| occ > 0.0 && active)
        .count();
    Ok(nocc)
}

/// Number of active MOs (total MOs minus frozen). Upstream:
/// `pyscf/mp/mp2.py:get_nmo` (line 371) — `len(mo_occ)` minus the frozen count.
///
/// Computed as the count of active entries in the frozen mask.
///
/// # Errors
/// Propagates [`Mp2Error::ShapeMismatch`] from the mask resolution.
pub fn get_nmo(mo_occ: &[f64], frozen: &Frozen) -> Result<usize, Mp2Error> {
    let mask = frozen::frozen_mask(frozen, mo_occ, &[], &[])?;
    Ok(mask.iter().filter(|&&active| active).count())
}

/// Boolean mask over MOs marking which are active (true) vs frozen (false).
/// Upstream: `pyscf/mp/mp2.py:get_frozen_mask` (line 383) — "the element is
/// False if it corresponds to the frozen orbital". Delegates to
/// [`crate::frozen::frozen_mask`].
///
/// # Errors
/// Propagates [`Mp2Error::ShapeMismatch`] from the mask resolution.
pub fn get_frozen_mask(mo_occ: &[f64], frozen: &Frozen) -> Result<Vec<bool>, Mp2Error> {
    frozen::frozen_mask(frozen, mo_occ, &[], &[])
}

/// Hartree–Fock reference energy used as the MP2 baseline. Upstream:
/// `pyscf/mp/mp2.py:get_e_hf` (line 402) — for the converged-reference case it
/// returns `mp._scf.e_tot` directly. We take the already-computed reference
/// `e_tot` (the converged SCF total energy) and return it verbatim.
pub fn get_e_hf(reference_e_tot: f64) -> f64 {
    reference_e_tot
}

/// MO coefficient block with the frozen-core columns removed. Maps to the
/// Python `_mo_without_core(mp, mo)` — `mo[:, get_frozen_mask(mp)]`. Upstream:
/// `pyscf/mp/mp2.py:_mo_without_core` (line 732).
///
/// `mo.data` is column-major `[nao, nmo]` (element `C[ao, mo]` at
/// `ao + mo*nao`), so dropping a frozen column means dropping its `nao`-length
/// contiguous slice. Returns a new [`MOCoefficients`] holding only the active
/// columns (same `nao`, `nmo == active_count`).
///
/// # Errors
/// Propagates [`Mp2Error::ShapeMismatch`] from the mask resolution.
#[doc(alias = "_mo_without_core")]
pub fn mo_without_core(mo: &MOCoefficients, frozen: &Frozen) -> Result<MOCoefficients, Mp2Error> {
    // The frozen mask is over the MO axis; mo.occupations is the per-MO occ
    // vector (same length as nmo).
    let mask = frozen::frozen_mask(frozen, &mo.occupations, &[], &[])?;
    let nao = mo.nao;

    let mut data = Vec::with_capacity(nao * mask.iter().filter(|&&a| a).count());
    let mut energies = Vec::new();
    let mut occupations = Vec::new();
    for (col, &active) in mask.iter().enumerate() {
        if !active {
            continue;
        }
        let start = col * nao;
        let end = start + nao;
        // Defensive: the column slice must be in range.
        if end > mo.data.len() {
            return Err(Mp2Error::ShapeMismatch {
                expected: mo.data.len(),
                got: end,
            });
        }
        data.extend_from_slice(&mo.data[start..end]);
        if col < mo.energies.len() {
            energies.push(mo.energies[col]);
        }
        if col < mo.occupations.len() {
            occupations.push(mo.occupations[col]);
        }
    }
    let nmo = data.len() / nao.max(1);
    Ok(MOCoefficients {
        nao,
        nmo,
        data,
        energies,
        occupations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nocc_nmo_no_frozen() {
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        assert_eq!(get_nocc(&mo_occ, &Frozen::None).expect("nocc"), 2);
        assert_eq!(get_nmo(&mo_occ, &Frozen::None).expect("nmo"), 4);
    }

    #[test]
    fn nocc_nmo_count_one_frozen() {
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        // Freezing the inner-most occupied orbital drops nocc by 1, nmo by 1.
        assert_eq!(get_nocc(&mo_occ, &Frozen::Count(1)).expect("nocc"), 1);
        assert_eq!(get_nmo(&mo_occ, &Frozen::Count(1)).expect("nmo"), 3);
    }

    #[test]
    fn frozen_mask_count_one() {
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        let mask = get_frozen_mask(&mo_occ, &Frozen::Count(1)).expect("mask");
        assert_eq!(mask, vec![false, true, true, true]);
    }

    #[test]
    fn e_hf_passthrough() {
        assert_eq!(get_e_hf(-76.026_760_737), -76.026_760_737);
    }

    #[test]
    fn mo_without_core_drops_frozen_column() {
        // nao=2, nmo=3; column-major data [c00,c10, c01,c11, c02,c12].
        let mo = MOCoefficients {
            nao: 2,
            nmo: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            energies: vec![-1.0, 0.1, 0.2],
            occupations: vec![2.0, 0.0, 0.0],
        };
        let active = mo_without_core(&mo, &Frozen::Count(1)).expect("split");
        // Column 0 frozen → keep columns 1 and 2.
        assert_eq!(active.nao, 2);
        assert_eq!(active.nmo, 2);
        assert_eq!(active.data, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(active.energies, vec![0.1, 0.2]);
        assert_eq!(active.occupations, vec![0.0, 0.0]);
    }
}
