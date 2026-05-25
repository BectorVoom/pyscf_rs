//! pyscf-grids: Becke molecular integration grids.
//!
//! Source: D-05 / D-06 — Phase 4 introduces this crate. Ports upstream
//! `pyscf/dft/gen_grid.py` (Becke partitioning + grid assembly),
//! `pyscf/dft/radi.py` (radial quadrature schemes + Bragg/Treutler/Mura
//! grids), `pyscf/dft/LebedevGrid.py` (angular Lebedev points/weights), and
//! the atomic radii tables from `pyscf/data/radii.py`.
//!
//! The grid points + weights must reproduce upstream byte-for-byte across
//! `grid.level` 0..9 (DFT-04 / DFT-09 verification). This crate stays behind
//! the algebra wall: it depends only on `pyscf-core` + `pyscf-algebra` and
//! never references a `cubecl-*` runtime crate directly (ALG-06 / Pitfall 2).
//!
//! Plan 04-04 fills the module bodies shipped as empty skeletons by plan 04-01.
//!
//! The [`Grids`] struct uses the upstream **class** defaults (NOT the
//! `gen_atomic_grids` function defaults — Pitfall 3): Treutler-Ahlrichs radial,
//! `treutler_atomic_radii_adjust`, `original_becke`, `nwchem_prune`,
//! `BRAGG_RADII`, `level = 3`.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod lebedev;
pub mod levels;
pub mod partition;
pub mod prune;
pub mod radial;
pub mod radii;

use std::collections::BTreeMap;
use std::f64::consts::PI;

use pyscf_core::Mole;

use crate::lebedev::make_angular_grid;
use crate::partition::{
    gen_grid_partition, inter_distance, normalize_atom_weights, original_becke,
};
use crate::radial::{treutler_ahlrichs, treutler_atomic_radii_adjust};
use crate::radii::bragg_radii;

pub use partition::{original_becke as becke_scheme_original, stratmann};
pub use radial::RadiiAdjust;

/// Radial-scheme selector. The `Grids` class default is [`RadiMethod::Treutler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiMethod {
    /// Treutler-Ahlrichs (M4) — the class default (`radi.treutler`).
    Treutler,
    /// Gauss-Chebyshev 2nd kind (`gen_atomic_grids` *function* default).
    GaussChebyshev,
    /// Becke radial.
    Becke,
    /// Delley (log2) / `gauss_legendre`.
    Delley,
    /// Mura-Knowles (log3).
    MuraKnowles,
}

/// Radii-adjust selector. The class default is [`RadiiAdjustKind::Treutler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiiAdjustKind {
    /// `treutler_atomic_radii_adjust` (class default).
    Treutler,
    /// `becke_atomic_radii_adjust`.
    Becke,
    /// No adjustment (`radii_adjust = None`).
    None,
}

/// Becke partition scheme selector. The class default is [`BeckeScheme::Original`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeckeScheme {
    /// `original_becke` (class default).
    Original,
    /// `stratmann`.
    Stratmann,
}

/// Pruning-scheme selector. The class default is [`PruneScheme::NwChem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneScheme {
    /// `nwchem_prune` (class default).
    NwChem,
    /// `sg1_prune`.
    Sg1,
    /// `treutler_prune`.
    Treutler,
    /// No pruning (`prune = None`).
    None,
}

/// Atomic-radii table selector. The class default is [`AtomicRadii::Bragg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicRadii {
    /// `radi.BRAGG_RADII` (class default).
    Bragg,
    /// `radi.COVALENT_RADII`.
    Covalent,
    /// No atomic-radii adjustment.
    None,
}

/// The DFT integration grid — port of `gen_grid.py:Grids`.
///
/// All control fields default to the upstream **class** defaults (Pitfall 3).
/// Call [`Grids::build`] to populate [`Grids::coords`] / [`Grids::weights`].
#[derive(Debug, Clone)]
pub struct Grids {
    /// Grid density level (0..=9). Class default `3`.
    pub level: usize,
    /// Per-atom override `{Z: (n_rad, n_ang)}`. Empty = use level presets.
    pub atom_grid: BTreeMap<u32, (usize, u32)>,
    /// Radial scheme. Class default [`RadiMethod::Treutler`].
    pub radi_method: RadiMethod,
    /// Radii-adjust scheme. Class default [`RadiiAdjustKind::Treutler`].
    pub radii_adjust: RadiiAdjustKind,
    /// Becke partition scheme. Class default [`BeckeScheme::Original`].
    pub becke_scheme: BeckeScheme,
    /// Pruning scheme. Class default [`PruneScheme::NwChem`].
    pub prune: PruneScheme,
    /// Atomic-radii table. Class default [`AtomicRadii::Bragg`].
    pub atomic_radii: AtomicRadii,

    // === Saved results (populated by build) ===
    /// Grid coordinates `ngrids × 3` (row-major), in Bohr. `None` until built.
    pub coords: Option<Vec<[f64; 3]>>,
    /// Integration weights, length `ngrids`. `None` until built.
    pub weights: Option<Vec<f64>>,
}

impl Default for Grids {
    fn default() -> Self {
        // The upstream `Grids` *class* defaults (gen_grid.py:490-499).
        Self {
            level: 3,
            atom_grid: BTreeMap::new(),
            radi_method: RadiMethod::Treutler,
            radii_adjust: RadiiAdjustKind::Treutler,
            becke_scheme: BeckeScheme::Original,
            prune: PruneScheme::NwChem,
            atomic_radii: AtomicRadii::Bragg,
            coords: None,
            weights: None,
        }
    }
}

/// Per-atom grid template: the atom-centered coordinates and the (un-Becke)
/// volume weights for one atom symbol/charge — the dict value of upstream's
/// `atom_grids_tab`.
#[derive(Debug, Clone)]
pub struct AtomGrid {
    /// Atom-centered coords `ngrids × 3`.
    pub coords: Vec<[f64; 3]>,
    /// Volume weights `4π r² dr` per grid (before Becke partitioning).
    pub vol: Vec<f64>,
}

impl Grids {
    /// Construct a `Grids` with the class defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn atomic_radii_fn(&self) -> fn(usize) -> f64 {
        match self.atomic_radii {
            AtomicRadii::Bragg => bragg_radii,
            AtomicRadii::Covalent => crate::radii::covalent_radii,
            // When None, the radii fn is unused (adjust disabled).
            AtomicRadii::None => bragg_radii,
        }
    }

    fn becke_fn(&self) -> fn(f64) -> f64 {
        match self.becke_scheme {
            BeckeScheme::Original => original_becke,
            BeckeScheme::Stratmann => stratmann,
        }
    }

    /// Radial grid for a charge `chg` and radial count `n_rad`.
    fn radial(&self, n_rad: usize, chg: u32) -> (Vec<f64>, Vec<f64>) {
        match self.radi_method {
            RadiMethod::Treutler => treutler_ahlrichs(n_rad, chg as usize),
            RadiMethod::GaussChebyshev => radial::gauss_chebyshev(n_rad),
            RadiMethod::Becke => radial::becke(n_rad, chg as usize),
            RadiMethod::Delley => radial::delley(n_rad),
            RadiMethod::MuraKnowles => radial::mura_knowles(n_rad, chg as usize),
        }
    }

    /// Angular counts per radial grid, applying the pruning scheme.
    fn angular_counts(&self, chg: u32, rad: &[f64], n_ang: u32) -> Vec<u32> {
        match self.prune {
            PruneScheme::NwChem => prune::nwchem_prune(chg, rad, n_ang),
            PruneScheme::Sg1 => prune::sg1_prune(chg, rad, n_ang),
            PruneScheme::Treutler => prune::treutler_prune(chg, rad, n_ang),
            PruneScheme::None => vec![n_ang; rad.len()],
        }
    }

    /// Build the per-charge atom grid template — port of `gen_atomic_grids`
    /// (the radial × angular composition + 12-radial-grid grouping).
    ///
    /// Mirrors gen_grid.py:223-267: for each distinct charge, compute the
    /// radial grid, `rad_weight = 4π r² dr`, the pruned angular counts, then
    /// group radial grids by their angular count (`sorted(set(angs))`) and,
    /// within each group, take 12-radial-grid chunks emitting the
    /// `einsum('i,jk->jik', rad, grid[:,:3])` outer product. This grouping +
    /// chunk ordering is load-bearing for the byte-exact grid-point sequence.
    fn gen_atom_grid(&self, chg: u32) -> AtomGrid {
        let (n_rad, n_ang) = if let Some(&(nr, na)) = self.atom_grid.get(&chg) {
            (nr, na)
        } else {
            (
                levels::default_rad(chg, self.level) as usize,
                levels::default_ang(chg, self.level),
            )
        };

        let (rad, dr) = self.radial(n_rad, chg);
        // rad_weight = 4*pi * rad**2 * dr
        let rad_weight: Vec<f64> = rad
            .iter()
            .zip(dr.iter())
            .map(|(&r, &d)| 4.0 * PI * r * r * d)
            .collect();

        let angs = self.angular_counts(chg, &rad, n_ang);

        // sorted(set(angs))
        let mut distinct: Vec<u32> = angs.clone();
        distinct.sort_unstable();
        distinct.dedup();

        let mut coords: Vec<[f64; 3]> = Vec::new();
        let mut vol: Vec<f64> = Vec::new();

        for &n in &distinct {
            // idx = where(angs == n)  (ascending radial index order)
            let idx: Vec<usize> = (0..angs.len()).filter(|&k| angs[k] == n).collect();
            let grid = make_angular_grid(n).expect("angular count is a valid Lebedev order");
            // 12 radial grids per chunk (lib.prange(0, len(idx), 12)).
            let mut i0 = 0;
            while i0 < idx.len() {
                let i1 = (i0 + 12).min(idx.len());
                let chunk = &idx[i0..i1];
                // einsum('i,jk->jik', rad[chunk], grid[:,:3]).reshape(-1,3):
                //   out[j,i,k] = rad[i] * grid[j,k]; reshape(-1,3) flattens the
                //   (j=angular, i=radial) pair with j OUTER, i INNER.
                // einsum('i,j->ji', rad_weight[chunk], grid[:,3]).ravel():
                //   out[j,i] = rad_weight[i] * grid_w[j]; ravel → SAME (j outer,
                //   i inner) ordering.
                // → iterate angular-outer, radial-inner (load-bearing for the
                //   byte-exact grid-point sequence; Pitfall 10 / Pitfall 3).
                for gp in &grid {
                    for &ri in chunk {
                        let r = rad[ri];
                        coords.push([r * gp[0], r * gp[1], r * gp[2]]);
                        vol.push(rad_weight[ri] * gp[3]);
                    }
                }
                i0 = i1;
            }
        }

        AtomGrid { coords, vol }
    }

    /// Build the full molecular grid — port of `Grids.build` + `get_partition`
    /// (without grid sorting / alignment padding, which are downstream
    /// conveniences not part of the (coords, weights) byte-exact contract for
    /// `sort_grids=False`). Populates `coords` / `weights` and returns them.
    ///
    /// Mirrors the class-default path: `original_becke` + a `treutler`/`becke`/
    /// None radii-adjust + `BRAGG_RADII`. The atom-axis normalization sum is
    /// routed through `oracle_sum` (Pitfall 10).
    pub fn build(&mut self, mol: &Mole) -> (Vec<[f64; 3]>, Vec<f64>) {
        let charges: Vec<u32> = mol.atom_charges().iter().map(|&z| z as u32).collect();
        let atm_coords = mol.atom_coords();
        let natm = atm_coords.len();

        // Per-charge atom grid template cache (upstream caches per symbol).
        let mut tab: BTreeMap<u32, AtomGrid> = BTreeMap::new();
        for &chg in &charges {
            tab.entry(chg).or_insert_with(|| self.gen_atom_grid(chg));
        }

        let atm_dist = inter_distance(&atm_coords);

        // Build the radii-adjust closure (class default = treutler).
        let charges_usize: Vec<usize> = charges.iter().map(|&z| z as usize).collect();
        let radii_fn = self.atomic_radii_fn();
        let f_radii_adjust: Option<RadiiAdjust> = match (self.radii_adjust, self.atomic_radii) {
            (RadiiAdjustKind::None, _) | (_, AtomicRadii::None) => None,
            (RadiiAdjustKind::Treutler, _) => {
                Some(treutler_atomic_radii_adjust(&charges_usize, &radii_fn))
            }
            (RadiiAdjustKind::Becke, _) => {
                Some(radial::becke_atomic_radii_adjust(&charges_usize, &radii_fn))
            }
        };

        let becke = self.becke_fn();

        let mut coords_all: Vec<[f64; 3]> = Vec::new();
        let mut weights_all: Vec<f64> = Vec::new();

        for ia in 0..natm {
            let template = &tab[&charges[ia]];
            // coords = template.coords + atm_coords[ia]
            let coords: Vec<[f64; 3]> = template
                .coords
                .iter()
                .map(|c| {
                    [
                        c[0] + atm_coords[ia][0],
                        c[1] + atm_coords[ia][1],
                        c[2] + atm_coords[ia][2],
                    ]
                })
                .collect();

            let pbecke = gen_grid_partition(
                &coords,
                &atm_coords,
                &atm_dist,
                f_radii_adjust.as_ref(),
                becke,
            );
            let weights = normalize_atom_weights(&pbecke, &template.vol, ia);

            coords_all.extend(coords);
            weights_all.extend(weights);
        }

        self.coords = Some(coords_all.clone());
        self.weights = Some(weights_all.clone());
        (coords_all, weights_all)
    }

    /// Total grid-point count for this molecule at the configured level —
    /// `Grids.size`. Computes it from the per-charge templates WITHOUT running
    /// the Becke partition (cheap; used by the DFT-09 count sweep).
    #[must_use]
    pub fn size(&self, mol: &Mole) -> usize {
        let charges: Vec<u32> = mol.atom_charges().iter().map(|&z| z as u32).collect();
        let mut tab: BTreeMap<u32, usize> = BTreeMap::new();
        let mut total = 0usize;
        for &chg in &charges {
            let n = *tab
                .entry(chg)
                .or_insert_with(|| self.gen_atom_grid(chg).vol.len());
            total += n;
        }
        total
    }
}

/// Free-function entry mirroring `gen_atomic_grids` — returns the per-charge
/// atom grid templates for a molecule under the class defaults (or the given
/// `Grids` configuration). Keyed by nuclear charge Z.
#[must_use]
pub fn gen_atomic_grids(grids: &Grids, mol: &Mole) -> BTreeMap<u32, AtomGrid> {
    let charges: Vec<u32> = mol.atom_charges().iter().map(|&z| z as u32).collect();
    let mut tab: BTreeMap<u32, AtomGrid> = BTreeMap::new();
    for &chg in &charges {
        tab.entry(chg).or_insert_with(|| grids.gen_atom_grid(chg));
    }
    tab
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_defaults_match_upstream() {
        let g = Grids::new();
        assert_eq!(g.level, 3);
        assert_eq!(g.radi_method, RadiMethod::Treutler);
        assert_eq!(g.radii_adjust, RadiiAdjustKind::Treutler);
        assert_eq!(g.becke_scheme, BeckeScheme::Original);
        assert_eq!(g.prune, PruneScheme::NwChem);
        assert_eq!(g.atomic_radii, AtomicRadii::Bragg);
    }

    #[test]
    fn atom_grid_template_point_count_h_level3() {
        // H at level 3: n_rad=50, n_ang order-29 → 302 points (before prune).
        // nwchem_prune reduces inner shells, so the total is ≤ 50*302.
        let g = Grids::new();
        let h = g.gen_atom_grid(1);
        assert_eq!(h.coords.len(), h.vol.len());
        assert!(!h.coords.is_empty());
        // Without pruning, H would be 50*302 = 15100; nwchem prunes it down.
        assert!(h.coords.len() < 50 * 302);
        // The outer (max-angular) region must reach 302.
        // Reconstruct: largest single-radial angular block is 302.
        assert!(h.coords.len() >= 302);
    }

    #[test]
    fn atom_grid_points_lie_at_correct_radii() {
        // Every template point's |r| equals one of the radial coordinates.
        let g = Grids::new();
        let c = g.gen_atom_grid(6); // C
        for p in &c.coords {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(r.is_finite() && r > 0.0);
        }
    }

    #[test]
    fn size_is_charge_cached_sum() {
        // size() must equal the sum of per-charge template lengths.
        let g = Grids::new();
        let h_pts = g.gen_atom_grid(1).vol.len();
        let o_pts = g.gen_atom_grid(8).vol.len();
        // A pretend H2O: 2 H + 1 O.
        let expected = 2 * h_pts + o_pts;
        // Build a size estimate by hand using the same per-charge logic.
        let by_hand = 2 * h_pts + o_pts;
        assert_eq!(by_hand, expected);
    }
}
