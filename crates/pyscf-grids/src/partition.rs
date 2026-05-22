//! Becke partitioning weights — port of `pyscf/dft/gen_grid.py` (Apache-2.0).
//!
//! D-05 / Pitfall 10 (the byte-exactness hot spot this crate owns).
//!
//! Ports the **pure-Python `get_partition` fallback** (gen_grid.py:314-329),
//! NOT the C cell-function kernel (Pitfall 4: that C path lives in PySCF's own
//! `libdft` extension, not in any sibling crate). Implements the
//! `original_becke` partition scheme (the `Grids` class default) and the
//! per-atom Becke weight product.
//!
//! The normalization `weights = vol * pbecke[ia] * (1 / pbecke.sum(axis=0))`
//! (gen_grid.py:337) is THE byte-exactness hot spot: the `pbecke.sum(axis=0)`
//! reduction over atoms is routed through [`pyscf_algebra::oracle_sum`] (an
//! ordered, FMA-free reduction). For the molecule sizes in scope (natm ≤ 128),
//! `oracle_sum` performs a strict left-to-right sequential sum that matches
//! numpy's `sum(axis=0)` byte-for-byte under the `release-oracle` profile.

use crate::radial::RadiiAdjust;

/// `original_becke(g)` — Becke, JCP 88, 2547 (1988). The `Grids` class
/// default partition scheme. Applies `g = (3 - g²) · g · 0.5` three times
/// (the iteration the optimized C cell-function kernel unrolls — see the
/// commented reference at gen_grid.py:177-181).
#[must_use]
#[inline]
pub fn original_becke(mut g: f64) -> f64 {
    g = (3.0 - g * g) * g * 0.5;
    g = (3.0 - g * g) * g * 0.5;
    g = (3.0 - g * g) * g * 0.5;
    g
}

/// `stratmann(g)` — Stratmann/Scuseria/Frisch, CPL 257, 213 (1996). Provided
/// for completeness (not the class default). `a = 0.64`; clamps to ±1 outside.
#[must_use]
pub fn stratmann(g: f64) -> f64 {
    let a = 0.64_f64;
    if g <= -a {
        return -1.0;
    }
    if g >= a {
        return 1.0;
    }
    let ma = g / a;
    let ma2 = ma * ma;
    (1.0 / 16.0) * (ma * (35.0 + ma2 * (-35.0 + ma2 * (21.0 - 5.0 * ma2))))
}

/// Euclidean distance matrix between atoms — port of `gto.inter_distance(mol)`.
/// Returns a flat `natm × natm` row-major matrix (diagonal = 0).
#[must_use]
pub fn inter_distance(atm_coords: &[[f64; 3]]) -> Vec<f64> {
    let natm = atm_coords.len();
    let mut d = vec![0.0_f64; natm * natm];
    for i in 0..natm {
        for j in 0..natm {
            let dx = atm_coords[i][0] - atm_coords[j][0];
            let dy = atm_coords[i][1] - atm_coords[j][1];
            let dz = atm_coords[i][2] - atm_coords[j][2];
            d[i * natm + j] = (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }
    d
}

/// Compute the Becke partition matrix `pbecke[natm][ngrids]` for a block of
/// grid coordinates, mirroring the pure-Python `gen_grid_partition`
/// (gen_grid.py:314-329).
///
/// `coords` is row-major `ngrids × 3` (already shifted by the owning atom's
/// center, as upstream does at gen_grid.py:335). `atm_coords` are the atom
/// centers; `atm_dist` is the flat distance matrix from [`inter_distance`];
/// `f_radii_adjust` is the optional radii-adjust closure.
///
/// Returns `pbecke` as `natm` rows of `ngrids` weights each.
#[must_use]
pub fn gen_grid_partition(
    coords: &[[f64; 3]],
    atm_coords: &[[f64; 3]],
    atm_dist: &[f64],
    f_radii_adjust: Option<&RadiiAdjust>,
    becke_scheme: fn(f64) -> f64,
) -> Vec<Vec<f64>> {
    let natm = atm_coords.len();
    let ngrids = coords.len();

    // grid_dist[ia][g] = |coords[g] - atm_coords[ia]|
    let mut grid_dist = vec![vec![0.0_f64; ngrids]; natm];
    for (ia, ga) in atm_coords.iter().enumerate() {
        for (g, c) in coords.iter().enumerate() {
            let dx = c[0] - ga[0];
            let dy = c[1] - ga[1];
            let dz = c[2] - ga[2];
            grid_dist[ia][g] = (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }

    // pbecke starts as ones, then accumulates the pairwise cell functions.
    let mut pbecke = vec![vec![1.0_f64; ngrids]; natm];
    for i in 0..natm {
        for j in 0..i {
            let inv_dist = 1.0 / atm_dist[i * natm + j];
            for g in 0..ngrids {
                // g = 1/atm_dist[i,j] * (grid_dist[i] - grid_dist[j])
                let mut gg = inv_dist * (grid_dist[i][g] - grid_dist[j][g]);
                if let Some(fadj) = f_radii_adjust {
                    gg = fadj(i, j, gg);
                }
                gg = becke_scheme(gg);
                pbecke[i][g] *= 0.5 * (1.0 - gg);
                pbecke[j][g] *= 0.5 * (1.0 + gg);
            }
        }
    }
    pbecke
}

/// Normalize a single atom's grid weights — the byte-exactness hot spot.
/// `weights[g] = vol[g] * pbecke[ia][g] * (1 / Σ_a pbecke[a][g])`, where the
/// atom-axis sum `Σ_a pbecke[a][g]` (= `pbecke.sum(axis=0)`) is computed via
/// [`pyscf_algebra::oracle_sum`] over a per-grid terms buffer.
#[must_use]
pub fn normalize_atom_weights(pbecke: &[Vec<f64>], vol: &[f64], ia: usize) -> Vec<f64> {
    let natm = pbecke.len();
    let ngrids = vol.len();
    let mut weights = vec![0.0_f64; ngrids];
    let mut col_buf = vec![0.0_f64; natm];
    for g in 0..ngrids {
        // Build the per-grid atom-axis column, then ordered-sum it.
        for (a, item) in col_buf.iter_mut().enumerate() {
            *item = pbecke[a][g];
        }
        let denom = pyscf_algebra::oracle_sum(&col_buf);
        weights[g] = vol[g] * pbecke[ia][g] * (1.0 / denom);
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_becke_fixed_points() {
        // g=0 → 0; g=1 → 1; g=-1 → -1 (fixed points of the iteration).
        assert!((original_becke(0.0)).abs() < 1e-15);
        assert!((original_becke(1.0) - 1.0).abs() < 1e-12);
        assert!((original_becke(-1.0) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn original_becke_monotone_smoothstep() {
        // The Becke step is monotone increasing and odd on [-1, 1].
        let mut prev = original_becke(-1.0);
        for k in -9..=10 {
            let g = (k as f64) / 10.0;
            let v = original_becke(g);
            assert!(v >= prev - 1e-12, "becke not monotone at g={g}");
            prev = v;
            // odd symmetry: f(-g) = -f(g).
            assert!((original_becke(-g) + v).abs() < 1e-12);
        }
    }

    #[test]
    fn inter_distance_symmetric_zero_diagonal() {
        let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [1.0, 0.0, 0.0]];
        let d = inter_distance(&coords);
        let n = 3;
        for i in 0..n {
            assert_eq!(d[i * n + i], 0.0);
            for j in 0..n {
                assert_eq!(d[i * n + j], d[j * n + i]);
            }
        }
        assert!((d[0 * n + 1] - 1.5).abs() < 1e-15);
        assert!((d[0 * n + 2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn partition_sums_to_one_over_atoms() {
        // For any grid point, Σ_ia (pbecke[ia] / pbecke.sum) = 1 by construction.
        let atm_coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let atm_dist = inter_distance(&atm_coords);
        // A few test grid points around the two atoms.
        let coords = vec![[0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.5], [1.0, 1.0, 1.0]];
        let pbecke = gen_grid_partition(&coords, &atm_coords, &atm_dist, None, original_becke);
        let natm = 2;
        for g in 0..coords.len() {
            let mut col = vec![0.0; natm];
            for a in 0..natm {
                col[a] = pbecke[a][g];
            }
            let denom = pyscf_algebra::oracle_sum(&col);
            let mut frac_sum = 0.0;
            for a in 0..natm {
                frac_sum += pbecke[a][g] / denom;
            }
            assert!((frac_sum - 1.0).abs() < 1e-12, "partition fractions sum != 1 at g={g}");
        }
    }

    #[test]
    fn normalize_uses_oracle_sum_path() {
        // Single-atom normalization with a hand-checked 2-atom pbecke.
        let pbecke = vec![vec![0.8, 0.5], vec![0.2, 0.5]];
        let vol = vec![1.0, 2.0];
        // atom 0: w[g] = vol[g] * pbecke[0][g] / (pbecke[0][g]+pbecke[1][g])
        let w0 = normalize_atom_weights(&pbecke, &vol, 0);
        assert!((w0[0] - (1.0 * 0.8 / 1.0)).abs() < 1e-15);
        assert!((w0[1] - (2.0 * 0.5 / 1.0)).abs() < 1e-15);
        let w1 = normalize_atom_weights(&pbecke, &vol, 1);
        assert!((w1[0] - (1.0 * 0.2 / 1.0)).abs() < 1e-15);
    }

    #[test]
    fn radii_adjust_changes_partition() {
        use crate::radial::treutler_atomic_radii_adjust;
        use crate::radii::bragg_radii;
        let atm_coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let atm_dist = inter_distance(&atm_coords);
        let coords = vec![[0.0, 0.0, 1.0]];
        let charges = vec![8usize, 1]; // O-H, different radii → adjustment active
        let fadj = treutler_atomic_radii_adjust(&charges, &bragg_radii);
        let with = gen_grid_partition(&coords, &atm_coords, &atm_dist, Some(&fadj), original_becke);
        let without = gen_grid_partition(&coords, &atm_coords, &atm_dist, None, original_becke);
        assert_ne!(with[0][0], without[0][0]);
    }
}
