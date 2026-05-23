//! Frozen-core spec resolution: `frozen=int` / `frozen=list` / `frozen='auto'`
//! (chemcore) / frozen-window forms (MP2-03).
//!
//! Ports the four upstream frozen-spec branches:
//!   - `frozen=None`    → no orbitals frozen (`pyscf/mp/mp2.py:get_frozen_mask`)
//!   - `frozen=int`     → the inner-most `n` orbitals frozen (`moidx[:n] = False`)
//!   - `frozen=list`    → the listed orbital indices frozen (`moidx[list] = False`)
//!   - `frozen='auto'`  → chemcore element→core-count table
//!     (`pyscf/cc/ccsd.py:set_frozen` → `pyscf/data/elements.py:chemcore`)
//!   - `frozen='window'`→ energy-window form (`mo_e < emin` or `mo_e > emax`)
//!
//! The resolved mask is the active-orbital mask: `true` = active, `false` =
//! frozen, matching upstream `get_frozen_mask` (line 383) semantics exactly
//! ("the element is False if it corresponds to the frozen orbital").

use crate::error::Mp2Error;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Frozen-core specification. Mirrors upstream `mp.frozen`:
/// `None` / `int` / `list[int]` / `'auto'` / `'window'`.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Frozen {
    /// No orbitals frozen (`frozen=None`). The default.
    #[default]
    None,
    /// Freeze the inner-most `n` orbitals (`frozen=int`; `moidx[:n] = False`).
    Count(usize),
    /// Freeze the listed orbital indices (`frozen=list`; `moidx[list] = False`).
    List(Vec<usize>),
    /// Auto chemcore (`frozen='auto'`): the per-atom frozen-core ORBITAL count
    /// is looked up in the chemcore table and summed (no ÷2; the table encodes
    /// orbitals, not electrons — see [`chemcore_count`]). The per-atom atomic
    /// numbers `Z` are supplied to [`frozen_mask`] by the caller (the kernel
    /// extracts them from `refr.mol.atom_charges()`).
    Auto,
    /// Energy-window form (`frozen='window'`): freeze MOs whose energy is below
    /// `emin` or above `emax`. Upstream default window is `(-1000.0, 1000.0)`
    /// (`pyscf/cc/ccsd.py:set_frozen`).
    Window { emin: f64, emax: f64 },
}

/// Per-element chemical-core ORBITAL counts, indexed by atomic number `Z`
/// (`Z = 0` is the ghost atom). Ported VERBATIM from upstream
/// `pyscf/data/elements.py:chemcore_atm` (line 1079).
///
/// IMPORTANT: each value is the number of FROZEN-CORE ORBITALS for that
/// element (NOT core electrons). Upstream `chemcore()` (elements.py:1095) sums
/// `chemcore_atm[Z]` over atoms and returns the total DIRECTLY (no ÷2): for the
/// non-ECP `spinorb=False` case it computes `core += chemcore_atm[Z]`. So the
/// auto frozen-orbital count is simply `sum(chemcore_atm[Z])` over atoms — the
/// table already encodes orbitals, not electrons (e.g. O→1 frozen orbital, the
/// 1s; Si→5, the 1s2s2p shell).
static CHEMCORE: OnceLock<HashMap<u32, usize>> = OnceLock::new();

/// The chemcore element→core-orbital table (`pyscf/data/elements.py:1079`).
/// Index = atomic number `Z`; value = frozen-core orbital count.
fn chemcore_table() -> &'static HashMap<u32, usize> {
    CHEMCORE.get_or_init(|| {
        // Verbatim transcription of `chemcore_atm` (elements.py:1079-1091),
        // a flat list indexed by Z (Z=0 ghost). Reproduced in Z order.
        const CHEMCORE_ATM: &[usize] = &[
            0, // 0: ghost
            0, 0, // 1-2:  H  He
            0, 0, 1, 1, 1, 1, 1, 1, // 3-10:  Li Be B C N O F Ne
            1, 1, 5, 5, 5, 5, 5, 5, // 11-18: Na Mg Al Si P S Cl Ar
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, // 19-36: K..Kr
            9, 9, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 18, 18, 18, 18, 18,
            18, // 37-54: Rb..Xe
            18, 18, // 55-56: Cs Ba
            // lanthanides (57-71) + 5d start
            18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 23, // 57-71
            23, 23, 23, 23, 23, 23, 23, 23, 23, 34, 34, 34, 34, 34, 34, // 72-86
            34, 34, // 87-88: Fr Ra
            // actinides (89-103) + 6d start
            34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, // 89-103
            50, 50, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55, 55, 55, 55, // 104-118
        ];
        CHEMCORE_ATM
            .iter()
            .enumerate()
            .map(|(z, &core)| (z as u32, core))
            .collect()
    })
}

/// Auto frozen-orbital count for a molecule given its per-atom atomic numbers.
///
/// Sums the per-atom frozen-core ORBITAL counts from the chemcore table and
/// returns the total directly — mirroring `pyscf/data/elements.py:chemcore`
/// (line 1095) for the non-ECP, `spinorb=False` (R/U) case where it computes
/// `core += chemcore_atm[Z]` and returns `core` (NO ÷2; the table already
/// encodes orbitals). Elements beyond the table (Z > 118) contribute 0
/// (upstream would index-error; we treat unknown super-heavy Z as no extra
/// core, the conservative choice).
pub fn chemcore_count(elements: &[u32]) -> usize {
    let table = chemcore_table();
    elements
        .iter()
        .map(|z| table.get(z).copied().unwrap_or(0))
        .sum()
}

/// Resolve the active-orbital boolean mask for a frozen spec.
///
/// Returns a `Vec<bool>` of length `mo_occ.len()` where `true` = active,
/// `false` = frozen — matching upstream `get_frozen_mask` (line 383): "the
/// element is False if it corresponds to the frozen orbital".
///
/// - `mo_occ`: MO occupation vector (used for length + window orbital count).
/// - `mo_energy`: MO energies (only consulted for `Frozen::Window`; may be
///   empty for the non-window variants).
/// - `elements`: per-atom atomic numbers `Z` (only consulted for
///   `Frozen::Auto`; may be empty for the others).
///
/// # Errors
/// Returns [`Mp2Error::ShapeMismatch`] when a `Frozen::List` / `Frozen::Count`
/// index is out of range of `mo_occ`, or when `Frozen::Window` is requested
/// but `mo_energy.len() != mo_occ.len()` — never panics / never indexes OOB
/// (T-05-03-FROZEN).
pub fn frozen_mask(
    frozen: &Frozen,
    mo_occ: &[f64],
    mo_energy: &[f64],
    elements: &[u32],
) -> Result<Vec<bool>, Mp2Error> {
    let nmo = mo_occ.len();
    // Start fully active (moidx = ones(bool), upstream get_frozen_mask:383).
    let mut mask = vec![true; nmo];

    match frozen {
        Frozen::None => {}
        Frozen::Count(n) => {
            // `moidx[:n] = False` — freeze the inner-most n orbitals.
            if *n > nmo {
                return Err(Mp2Error::ShapeMismatch {
                    expected: nmo,
                    got: *n,
                });
            }
            for m in mask.iter_mut().take(*n) {
                *m = false;
            }
        }
        Frozen::List(indices) => {
            // `moidx[list] = False` — freeze the listed indices.
            for &idx in indices {
                if idx >= nmo {
                    return Err(Mp2Error::ShapeMismatch {
                        expected: nmo,
                        got: idx,
                    });
                }
                mask[idx] = false;
            }
        }
        Frozen::Auto => {
            // chemcore: freeze the inner-most `chemcore_count(elements)`
            // orbitals (equivalent to `frozen=int` with the auto count;
            // upstream `set_frozen('auto')` sets `mycc.frozen = chemcore(mol)`,
            // an int, which `get_frozen_mask` then treats as `moidx[:n]=False`).
            let n = chemcore_count(elements);
            if n > nmo {
                return Err(Mp2Error::ShapeMismatch {
                    expected: nmo,
                    got: n,
                });
            }
            for m in mask.iter_mut().take(n) {
                *m = false;
            }
        }
        Frozen::Window { emin, emax } => {
            // `frozen = flatnonzero(mo_e < emin) + flatnonzero(mo_e > emax)`,
            // then `moidx[frozen] = False` (ccsd.py:set_frozen 'window').
            if mo_energy.len() != nmo {
                return Err(Mp2Error::ShapeMismatch {
                    expected: nmo,
                    got: mo_energy.len(),
                });
            }
            for (i, &e) in mo_energy.iter().enumerate() {
                if e < *emin || e > *emax {
                    mask[i] = false;
                }
            }
        }
    }

    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_none_all_active() {
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        let mask = frozen_mask(&Frozen::None, &mo_occ, &[], &[]).expect("none");
        assert_eq!(mask, vec![true, true, true, true]);
    }

    #[test]
    fn frozen_count_freezes_innermost() {
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        let mask = frozen_mask(&Frozen::Count(1), &mo_occ, &[], &[]).expect("count");
        assert_eq!(mask, vec![false, true, true, true]);
    }

    #[test]
    fn frozen_list_freezes_named() {
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        let mask = frozen_mask(&Frozen::List(vec![0, 3]), &mo_occ, &[], &[]).expect("list");
        assert_eq!(mask, vec![false, true, true, false]);
    }

    #[test]
    fn frozen_count_out_of_range_errors_not_panics() {
        let mo_occ = [2.0, 2.0];
        let r = frozen_mask(&Frozen::Count(5), &mo_occ, &[], &[]);
        assert!(matches!(r, Err(Mp2Error::ShapeMismatch { .. })));
    }

    #[test]
    fn frozen_list_out_of_range_errors_not_panics() {
        let mo_occ = [2.0, 2.0];
        let r = frozen_mask(&Frozen::List(vec![9]), &mo_occ, &[], &[]);
        assert!(matches!(r, Err(Mp2Error::ShapeMismatch { .. })));
    }

    #[test]
    fn chemcore_water_freezes_one() {
        // H2O: O(Z=8 → chemcore_atm[8]=1 frozen orbital, the 1s) + 2*H(Z=1 → 0).
        // Upstream chemcore(H2O) == 1 (freeze the oxygen 1s). No ÷2.
        let elements = [8_u32, 1, 1];
        assert_eq!(chemcore_count(&elements), 1);
    }

    #[test]
    fn chemcore_silicon_freezes_five() {
        // Si (Z=14 → chemcore_atm[14]=5: 1s 2s 2p_x 2p_y 2p_z). No ÷2.
        let elements = [14_u32];
        assert_eq!(chemcore_count(&elements), 5);

        // Ne (Z=10 → chemcore_atm[10]=1: the 1s).
        assert_eq!(chemcore_count(&[10_u32]), 1);
    }

    #[test]
    fn frozen_window_freezes_outside_band() {
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        let mo_energy = [-1500.0, -0.4, 0.2, 1500.0];
        let mask = frozen_mask(
            &Frozen::Window {
                emin: -1000.0,
                emax: 1000.0,
            },
            &mo_occ,
            &mo_energy,
            &[],
        )
        .expect("window");
        assert_eq!(mask, vec![false, true, true, false]);
    }
}
