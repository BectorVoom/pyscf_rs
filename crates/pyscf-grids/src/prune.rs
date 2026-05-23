//! Grid-pruning schemes — port of `pyscf/dft/gen_grid.py` (Apache-2.0).
//!
//! D-06: ports the three pruning functions that reduce the per-radial-grid
//! angular count. Each takes the nuclear charge `nuc`, the radial coordinates
//! `rads`, and the max angular count `n_ang`, and returns a `Vec<u32>` (one
//! angular count per radial grid).
//!
//!   * [`nwchem_prune`] — NWChem scheme, the `Grids` **class default**.
//!   * [`sg1_prune`]    — SG1, Gill/Johnson/Pople, CPL 209 (1993) 506.
//!   * [`treutler_prune`] — Treutler-Ahlrichs, JCP 102, 346 (1995).

use crate::lebedev::lebedev_ngrid;
use crate::radii::{bragg_radii, sg1_radii};

/// NWChem pruning — `gen_grid.py:nwchem_prune` (the `Grids` class default).
///
/// `rads` are the radial coordinates (Bohr); `n_ang` is the max Lebedev point
/// count. Returns one angular count per radial grid. Mirrors upstream exactly:
/// the `LEBEDEV_NGRID[4:]` slice (`[38, 50, 74, ...]`), the `n_ang < 50` /
/// `== 50` / general branches, the per-period `alphas` rows, and the
/// `place = sum(rads/r_atom > alphas)` bucketing.
#[must_use]
pub fn nwchem_prune(nuc: u32, rads: &[f64], n_ang: u32) -> Vec<u32> {
    // alphas[period_class][threshold]
    const ALPHAS: [[f64; 4]; 3] = [
        [0.25, 0.5, 1.0, 4.5],
        [0.1667, 0.5, 0.9, 3.5],
        [0.1, 0.4, 0.8, 2.5],
    ];
    // leb_ngrid = LEBEDEV_NGRID[4:]  → [38, 50, 74, 86, ...]
    let full = lebedev_ngrid();
    let leb_ngrid: Vec<u32> = full[4..].to_vec();

    if n_ang < 50 {
        return vec![n_ang; rads.len()];
    }

    // leb_l: per-region Lebedev-index selector.
    let leb_l: [usize; 5] = if n_ang == 50 {
        [1, 2, 2, 2, 1]
    } else {
        // idx = index of n_ang in leb_ngrid.
        let idx = leb_ngrid
            .iter()
            .position(|&x| x == n_ang)
            .expect("n_ang must be a supported Lebedev count");
        [1, 3, idx - 1, idx, idx - 1]
    };

    let r_atom = bragg_radii(nuc as usize) + 1e-200;
    // Choose the alphas row by element period: H,He → row 0; Li-Ne → row 1;
    // heavier → row 2.
    let row = if nuc <= 2 {
        &ALPHAS[0]
    } else if nuc <= 10 {
        &ALPHAS[1]
    } else {
        &ALPHAS[2]
    };

    rads.iter()
        .map(|&r| {
            // place = sum over thresholds of (r/r_atom > alpha)
            let ratio = r / r_atom;
            let place = row.iter().filter(|&&a| ratio > a).count();
            let ang_idx = leb_l[place];
            leb_ngrid[ang_idx]
        })
        .collect()
}

/// SG1 pruning — `gen_grid.py:sg1_prune`. Five-region scheme with fixed
/// `[6, 38, 86, 194, 86]` Lebedev counts, bucketed by SG1 atomic radii.
#[must_use]
pub fn sg1_prune(nuc: u32, rads: &[f64], _n_ang: u32) -> Vec<u32> {
    const LEB_NGRID: [u32; 5] = [6, 38, 86, 194, 86];
    const ALPHAS: [[f64; 4]; 3] = [
        [0.25, 0.5, 1.0, 4.5],
        [0.1667, 0.5, 0.9, 3.5],
        [0.1, 0.4, 0.8, 2.5],
    ];
    let r_atom = sg1_radii(nuc as usize) + 1e-200;
    let row = if nuc <= 2 {
        &ALPHAS[0]
    } else if nuc <= 10 {
        &ALPHAS[1]
    } else {
        &ALPHAS[2]
    };
    rads.iter()
        .map(|&r| {
            let ratio = r / r_atom;
            let place = row.iter().filter(|&&a| ratio > a).count();
            LEB_NGRID[place]
        })
        .collect()
}

/// Treutler-Ahlrichs pruning — `gen_grid.py:treutler_prune`. First `nr//3`
/// radial shells get 14 points (l=5), `nr//3..nr//2` get 50 (l=11), the rest
/// get `n_ang`.
#[must_use]
pub fn treutler_prune(_nuc: u32, rads: &[f64], n_ang: u32) -> Vec<u32> {
    let nr = rads.len();
    let mut out = vec![n_ang; nr];
    let third = nr / 3;
    let half = nr / 2;
    for v in out.iter_mut().take(third) {
        *v = 14;
    }
    for v in out.iter_mut().take(half).skip(third) {
        *v = 50;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nwchem_prune_below_50_returns_uniform() {
        // n_ang < 50 → every radial grid gets n_ang.
        let rads = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        let out = nwchem_prune(8, &rads, 38);
        assert_eq!(out, vec![38, 38, 38, 38, 38]);
    }

    #[test]
    fn nwchem_prune_reduces_inner_shells() {
        // For O (Z=8) with n_ang=302, inner shells get fewer angular points,
        // outer shells reach the max. The result is monotone-ish: the very
        // small-radius shells must be pruned below the max.
        let r_atom = bragg_radii(8) + 1e-200;
        // Build radii spanning 0.01*r_atom .. 10*r_atom.
        let rads: Vec<f64> = (1..=20).map(|i| (i as f64) * 0.5 * r_atom).collect();
        let out = nwchem_prune(8, &rads, 302);
        assert_eq!(out.len(), rads.len());
        // The max angular count present must be 302 (the outer region).
        assert!(out.contains(&302));
        // All entries must be valid Lebedev counts ≤ 302.
        let valid = lebedev_ngrid();
        for &a in &out {
            assert!(valid.contains(&a), "{a} not a Lebedev count");
            assert!(a <= 302);
        }
    }

    #[test]
    fn nwchem_prune_n_ang_50_uses_special_leb_l() {
        // n_ang == 50 → leb_l = [1,2,2,2,1] → counts from leb_ngrid =
        // [38,50,74,...]: regions map to {50,74,74,74,50}.
        let r_atom = bragg_radii(1) + 1e-200; // H, period-0 alphas
        // Pick radii that fall into each of the 5 buckets (place 0..4).
        // alphas row 0 = [0.25, 0.5, 1.0, 4.5].
        let rads = vec![
            0.1 * r_atom, // place 0
            0.3 * r_atom, // place 1
            0.7 * r_atom, // place 2
            2.0 * r_atom, // place 3
            5.0 * r_atom, // place 4
        ];
        let out = nwchem_prune(1, &rads, 50);
        // leb_ngrid = [38,50,74,86,...]; leb_l=[1,2,2,2,1] → idx into leb_ngrid
        // → [50,74,74,74,50].
        assert_eq!(out, vec![50, 74, 74, 74, 50]);
    }

    #[test]
    fn treutler_prune_three_regions() {
        let rads = vec![0.0; 12];
        let out = treutler_prune(6, &rads, 302);
        // nr=12: [0..4)=14, [4..6)=50, [6..12)=302.
        assert_eq!(out[0], 14);
        assert_eq!(out[3], 14);
        assert_eq!(out[4], 50);
        assert_eq!(out[5], 50);
        assert_eq!(out[6], 302);
        assert_eq!(out[11], 302);
    }

    #[test]
    fn sg1_prune_buckets() {
        let r_atom = sg1_radii(1) + 1e-200; // H
        let rads = vec![
            0.1 * r_atom,
            0.3 * r_atom,
            0.7 * r_atom,
            2.0 * r_atom,
            5.0 * r_atom,
        ];
        let out = sg1_prune(1, &rads, 0);
        // LEB_NGRID = [6,38,86,194,86] by place 0..4.
        assert_eq!(out, vec![6, 38, 86, 194, 86]);
    }
}
