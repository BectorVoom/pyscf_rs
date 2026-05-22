//! Grid-level presets — port of `pyscf/dft/gen_grid.py` RAD_GRIDS / ANG_ORDER
//! tables + `_default_rad` / `_default_ang` period lookup (Apache-2.0).
//!
//! D-06 / DFT-09: inline `const` tables, no codegen. For `level ∈ 0..=9` and
//! a nuclear charge `Z`, [`default_rad`] returns the number of radial grids
//! and [`default_ang`] returns the number of angular (Lebedev) grids — these
//! determine the per-atom grid sizes that must match upstream point counts.

use crate::lebedev::lebedev_order;

/// `RAD_GRIDS[level][period]` — number of radial grids. Port of
/// `gen_grid.py:RAD_GRIDS` (10 levels × 7 periods). Period index 0..6 maps
/// to periods 1..7.
#[rustfmt::skip]
const RAD_GRIDS: [[u32; 7]; 10] = [
    [ 10,  15,  20,  30,  35,  40,  50], // 0
    [ 30,  40,  50,  60,  65,  70,  75], // 1
    [ 40,  60,  65,  75,  80,  85,  90], // 2
    [ 50,  75,  80,  90,  95, 100, 105], // 3
    [ 60,  90,  95, 105, 110, 115, 120], // 4
    [ 70, 105, 110, 120, 125, 130, 135], // 5
    [ 80, 120, 125, 135, 140, 145, 150], // 6
    [ 90, 135, 140, 150, 155, 160, 165], // 7
    [100, 150, 155, 165, 170, 175, 180], // 8
    [200, 200, 200, 200, 200, 200, 200], // 9
];

/// `ANG_ORDER[level][period]` — Lebedev *order* (not point count). Port of
/// `gen_grid.py:ANG_ORDER`. The order is mapped to a point count via
/// `LEBEDEV_ORDER` (see [`default_ang`]).
#[rustfmt::skip]
const ANG_ORDER: [[u32; 7]; 10] = [
    [11, 15, 17, 17, 17, 17, 17], // 0
    [17, 23, 23, 23, 23, 23, 23], // 1
    [23, 29, 29, 29, 29, 29, 29], // 2
    [29, 29, 35, 35, 35, 35, 35], // 3
    [35, 41, 41, 41, 41, 41, 41], // 4
    [41, 47, 47, 47, 47, 47, 47], // 5
    [47, 53, 53, 53, 53, 53, 53], // 6
    [53, 59, 59, 59, 59, 59, 59], // 7
    [59, 59, 59, 59, 59, 59, 59], // 8
    [65, 65, 65, 65, 65, 65, 65], // 9
];

/// Period boundaries — `gen_grid.py:_default_rad/_default_ang` use
/// `tab = (2, 10, 18, 36, 54, 86, 118)` and `period = (nuc > tab).sum()`.
const PERIOD_TAB: [u32; 7] = [2, 10, 18, 36, 54, 86, 118];

/// Period index `(nuc > tab).sum()` for nuclear charge `nuc` (0..=6,
/// corresponding to periods 1..7 — capped at index 6).
#[must_use]
pub fn period_index(nuc: u32) -> usize {
    PERIOD_TAB.iter().filter(|&&t| nuc > t).count()
}

/// Number of radial grids for `(charge, level)` — `_default_rad`.
/// `level` is clamped to 0..=9 (matching the supported preset range).
#[must_use]
pub fn default_rad(nuc: u32, level: usize) -> u32 {
    let period = period_index(nuc);
    RAD_GRIDS[level][period]
}

/// Number of angular (Lebedev) grids for `(charge, level)` — `_default_ang`.
/// Looks up the Lebedev *order* in `ANG_ORDER`, then maps it to the point
/// count via `LEBEDEV_ORDER` (`gen_grid.py:_default_ang`).
#[must_use]
pub fn default_ang(nuc: u32, level: usize) -> u32 {
    let period = period_index(nuc);
    let order = ANG_ORDER[level][period];
    lebedev_order(order).expect("ANG_ORDER value must be a valid Lebedev order")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn period_lookup_matches_upstream() {
        // H (Z=1) → period 0; C (Z=6) → period 1 (>2); Na (Z=11) → period 2;
        // K (Z=19) → period 3; Z=37 → 4; Z=55 → 5; Z=87 → 6.
        assert_eq!(period_index(1), 0);
        assert_eq!(period_index(2), 0);
        assert_eq!(period_index(3), 1);
        assert_eq!(period_index(10), 1);
        assert_eq!(period_index(11), 2);
        assert_eq!(period_index(19), 3);
        assert_eq!(period_index(37), 4);
        assert_eq!(period_index(55), 5);
        assert_eq!(period_index(87), 6);
    }

    #[test]
    fn default_rad_class_default_level3() {
        // gen_grid.py docstring: level 3 → (50,302) for H,He; (75,302) for
        // 2nd row; (80~105, 434) for the rest. Radial part:
        assert_eq!(default_rad(1, 3), 50); // H
        assert_eq!(default_rad(6, 3), 75); // C (2nd row)
        assert_eq!(default_rad(8, 3), 75); // O (2nd row)
    }

    #[test]
    fn default_ang_class_default_level3() {
        // level 3: H,He → order 29 → 302 points; 2nd row → order 29 → 302.
        assert_eq!(default_ang(1, 3), 302); // H, order 29
        assert_eq!(default_ang(6, 3), 302); // C, order 29
        // 3rd row (e.g. S, Z=16, period 2) → order 35 → 434 points.
        assert_eq!(default_ang(16, 3), 434);
    }

    #[test]
    fn rad_grids_level_sweep_for_h_and_c() {
        // Spot-check the full level 0..9 column for H (period 0) and C (period 1).
        let h_expected = [10, 30, 40, 50, 60, 70, 80, 90, 100, 200];
        let c_expected = [15, 40, 60, 75, 90, 105, 120, 135, 150, 200];
        for level in 0..=9 {
            assert_eq!(default_rad(1, level), h_expected[level]);
            assert_eq!(default_rad(6, level), c_expected[level]);
        }
    }

    #[test]
    fn ang_order_level_sweep_maps_through_lebedev() {
        // H angular orders per level 0..9 → point counts.
        // orders: 11,17,23,29,35,41,47,53,59,65
        let h_order_pts = [50u32, 110, 194, 302, 434, 590, 770, 974, 1202, 1454];
        for level in 0..=9 {
            assert_eq!(default_ang(1, level), h_order_pts[level]);
        }
    }
}
