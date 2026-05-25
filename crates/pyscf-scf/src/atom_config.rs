//! Per-element nonrelativistic spherical-RHF electron configuration table
//! and `frac_occ` — the occupation source for the minao init guess (plan 03-13).
//!
//! `NRSRHF_CONFIGURATION[Z]` is `[n_s, n_p, n_d, n_f]`, the number of electrons
//! in each angular-momentum shell for the neutral atom's spherically-averaged
//! RHF ground configuration. Ported VERBATIM from
//! `pyscf/data/elements.py:NRSRHF_CONFIGURATION` (Apache-2.0) — index 0 is the
//! GHOST atom (all zeros); valid Z range is 0..=118.

/// `[n_s, n_p, n_d, n_f]` electrons per l-shell, indexed by atomic number Z.
/// Source: `pyscf/data/elements.py:NRSRHF_CONFIGURATION`.
pub(crate) const NRSRHF_CONFIGURATION: [[u32; 4]; 119] = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [2, 0, 0, 0],
    [3, 0, 0, 0],
    [4, 0, 0, 0],
    [4, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 3, 0, 0],
    [4, 4, 0, 0],
    [4, 5, 0, 0],
    [4, 6, 0, 0],
    [5, 6, 0, 0],
    [6, 6, 0, 0],
    [6, 7, 0, 0],
    [6, 8, 0, 0],
    [6, 9, 0, 0],
    [6, 10, 0, 0],
    [6, 11, 0, 0],
    [6, 12, 0, 0],
    [7, 12, 0, 0],
    [8, 12, 0, 0],
    [8, 13, 0, 0],
    [8, 12, 2, 0],
    [8, 12, 3, 0],
    [8, 12, 4, 0],
    [6, 12, 7, 0],
    [6, 12, 8, 0],
    [6, 12, 9, 0],
    [6, 12, 10, 0],
    [7, 12, 10, 0],
    [8, 12, 10, 0],
    [8, 13, 10, 0],
    [8, 14, 10, 0],
    [8, 15, 10, 0],
    [8, 16, 10, 0],
    [8, 17, 10, 0],
    [8, 18, 10, 0],
    [9, 18, 10, 0],
    [10, 18, 10, 0],
    [10, 19, 10, 0],
    [10, 18, 12, 0],
    [10, 18, 13, 0],
    [8, 18, 16, 0],
    [8, 18, 17, 0],
    [8, 18, 18, 0],
    [8, 18, 19, 0],
    [8, 18, 20, 0],
    [9, 18, 20, 0],
    [10, 18, 20, 0],
    [10, 19, 20, 0],
    [10, 20, 20, 0],
    [10, 21, 20, 0],
    [10, 22, 20, 0],
    [10, 23, 20, 0],
    [10, 24, 20, 0],
    [11, 24, 20, 0],
    [12, 24, 20, 0],
    [12, 24, 21, 0],
    [12, 24, 22, 0],
    [12, 24, 21, 2],
    [12, 24, 20, 4],
    [12, 24, 20, 5],
    [12, 24, 20, 6],
    [12, 24, 20, 7],
    [11, 24, 20, 9],
    [10, 24, 20, 11],
    [10, 24, 20, 12],
    [10, 24, 20, 13],
    [10, 24, 20, 14],
    [11, 24, 20, 14],
    [12, 24, 20, 14],
    [12, 25, 20, 14],
    [12, 24, 22, 14],
    [12, 24, 23, 14],
    [10, 24, 26, 14],
    [10, 24, 27, 14],
    [10, 24, 28, 14],
    [10, 24, 29, 14],
    [10, 24, 30, 14],
    [11, 24, 30, 14],
    [12, 24, 30, 14],
    [12, 25, 30, 14],
    [12, 26, 30, 14],
    [12, 27, 30, 14],
    [12, 28, 30, 14],
    [12, 29, 30, 14],
    [12, 30, 30, 14],
    [13, 30, 30, 14],
    [14, 30, 30, 14],
    [14, 30, 31, 14],
    [14, 30, 32, 14],
    [14, 30, 30, 17],
    [14, 30, 30, 18],
    [14, 30, 30, 19],
    [13, 30, 30, 21],
    [12, 30, 30, 23],
    [12, 30, 30, 24],
    [12, 30, 30, 25],
    [12, 30, 30, 26],
    [12, 30, 30, 27],
    [12, 30, 30, 28],
    [13, 30, 30, 28],
    [14, 30, 30, 28],
    [14, 30, 31, 28],
    [14, 30, 32, 28],
    [14, 30, 33, 28],
    [12, 30, 36, 28],
    [12, 30, 37, 28],
    [12, 30, 38, 28],
    [12, 30, 39, 28],
    [12, 30, 40, 28],
    [13, 30, 40, 28],
    [14, 30, 40, 28],
    [14, 31, 40, 28],
    [14, 32, 40, 28],
    [14, 33, 40, 28],
    [14, 34, 40, 28],
    [14, 35, 40, 28],
    [14, 36, 40, 28],
];

/// Fractional occupation for angular momentum `l` of element `z`, mirroring
/// `pyscf/scf/atom_hf.py:frac_occ`:
///   `ne = config[z][l]; nd = (2l+1)*2; ndocc = ne / nd; frac = (ne/nd - ndocc)*2`.
/// Returns `(ndocc, frac)` — `ndocc` fully (doubly) occupied shells of this `l`
/// and a fractional `frac ∈ [0, 2)` for the frontier shell. `(0, 0.0)` if `l >= 4`
/// or the shell is empty.
pub(crate) fn frac_occ(z: usize, l: usize) -> (u32, f64) {
    if l >= 4 || z >= NRSRHF_CONFIGURATION.len() {
        return (0, 0.0);
    }
    let ne = NRSRHF_CONFIGURATION[z][l];
    if ne == 0 {
        return (0, 0.0);
    }
    let nd = ((2 * l + 1) * 2) as u32;
    let ndocc = ne / nd;
    let frac = (ne as f64 / nd as f64 - ndocc as f64) * 2.0;
    (ndocc, frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frac_occ_matches_hand_computed() {
        // H (Z=1): 1 s electron → ndocc=0, frac=1.0.
        assert_eq!(frac_occ(1, 0), (0, 1.0));
        // He (Z=2): 2 s → ndocc=1, frac=0.0.
        assert_eq!(frac_occ(2, 0), (1, 0.0));
        // C (Z=6): s=4 → ndocc=2,frac=0; p=2 → nd=6, ndocc=0, frac=2/6*2=2/3.
        assert_eq!(frac_occ(6, 0), (2, 0.0));
        let (nd_c_p, frac_c_p) = frac_occ(6, 1);
        assert_eq!(nd_c_p, 0);
        assert!(
            (frac_c_p - 2.0 / 3.0).abs() < 1e-12,
            "C p frac = {frac_c_p}"
        );
        // O (Z=8): p=4 → nd=6, ndocc=0, frac=4/6*2=4/3.
        let (_, frac_o_p) = frac_occ(8, 1);
        assert!(
            (frac_o_p - 4.0 / 3.0).abs() < 1e-12,
            "O p frac = {frac_o_p}"
        );
        // l>=4 and empty shells → (0,0).
        assert_eq!(frac_occ(1, 1), (0, 0.0));
        assert_eq!(frac_occ(6, 3), (0, 0.0));
    }

    #[test]
    fn table_covers_z0_to_118() {
        assert_eq!(NRSRHF_CONFIGURATION.len(), 119);
        assert_eq!(NRSRHF_CONFIGURATION[0], [0, 0, 0, 0]); // GHOST
        assert_eq!(NRSRHF_CONFIGURATION[1], [1, 0, 0, 0]); // H
    }
}
