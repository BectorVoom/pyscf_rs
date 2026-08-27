//! Periodic real-space grids — plan 12-01, port of
//! `pyscf/pbc/dft/gen_grid.py`.
//!
//! Two quadratures, and the periodic `NumInt` integrates over either:
//!
//! * [`UniformGrids`] — the FFT box (`gen_grid.py:64-148`). It already lives in
//!   `pyscf-pbc-gto` because `pyscf-pbc-df` needs it and the reverse edge would
//!   be a cycle (plan 11-04); this module re-exports it so a caller writing
//!   against `pyscf_pbc_dft` finds it where upstream puts it.
//! * [`BeckeGrids`] — Becke-partitioned atom-centred grids (`gen_grid.py:150-238`),
//!   the periodic analogue of `pyscf_grids::Grids`. Needed by the
//!   range-separated J/K builders (`jk_method('RS')`) and by anything that wants
//!   a grid whose accuracy does not scale with the FFT mesh.
//!
//! # How the periodic Becke partition differs from the molecular one
//!
//! The partition runs over a SUPERCELL of atoms — every atom of every lattice
//! image within `rcut` — but only the grid points that land inside the unit
//! cell are kept, and a point exactly on a cell face gets half weight (upstream
//! `gen_grid.py:189-206`). That makes `Σ_g w_g f(g)` the integral over ONE
//! cell of a lattice-periodic `f`, which is what `pbc_eval_gto` produces.
//!
//! Upstream passes `p_radii_table = NULL` to `VXCgen_grid`, i.e. the periodic
//! Becke partition uses NO atomic-radii adjustment — unlike the molecular
//! default. This port matches that.

use pyscf_grids::Grids;
use pyscf_grids::partition::{gen_grid_partition, inter_distance, original_becke};
use pyscf_pbc_gto::Cell;

use crate::error::PbcDftError;
use crate::xc::err;

pub use pyscf_pbc_gto::UniformGrids;

/// `gen_grid.BeckeGrids(cell)` — `gen_grid.py:240-294`.
#[derive(Debug, Clone)]
pub struct BeckeGrids {
    /// The molecular grid configuration the atomic templates come from
    /// (`level`, `radi_method`, `prune`, ...). Upstream's `BeckeGrids` inherits
    /// every one of these from `dft.gen_grid.Grids`.
    pub config: Grids,
    /// `(ngrids, 3)` grid points in Bohr. `None` until [`BeckeGrids::build`].
    pub coords: Option<Vec<[f64; 3]>>,
    /// Integration weights. `None` until [`BeckeGrids::build`].
    pub weights: Option<Vec<f64>>,
}

impl Default for BeckeGrids {
    fn default() -> Self {
        Self::new()
    }
}

impl BeckeGrids {
    /// A `BeckeGrids` with the upstream class defaults.
    pub fn new() -> Self {
        Self {
            config: Grids::new(),
            coords: None,
            weights: None,
        }
    }

    /// `get_becke_grids(cell, ...)` — `gen_grid.py:150-238`.
    ///
    /// # Errors
    /// Propagates [`pyscf_pbc_gto::estimate_rcut_for_eval`] and
    /// [`pyscf_pbc_gto::lattice::get_lattice_ls`].
    pub fn build(&mut self, cell: &Cell) -> Result<(Vec<[f64; 3]>, Vec<f64>), PbcDftError> {
        let (c, w) = get_becke_grids(cell, &self.config)?;
        self.coords = Some(c.clone());
        self.weights = Some(w.clone());
        Ok((c, w))
    }

    /// Grid-point count, or 0 when unbuilt.
    pub fn size(&self) -> usize {
        self.coords.as_ref().map_or(0, Vec::len)
    }
}

/// `get_becke_grids(cell, atom_grid, radi_method, level, prune)` —
/// `gen_grid.py:150-238`.
///
/// # Errors
/// [`PbcDftError`] when the image list or the AO cutoff cannot be built.
pub fn get_becke_grids(
    cell: &Cell,
    config: &Grids,
) -> Result<(Vec<[f64; 3]>, Vec<f64>), PbcDftError> {
    // gen_grid.py:161-165 — with `low_dim_ft_type != 'inf_vacuum'` a 2-D cell is
    // treated as 3-D by `pbc_eval_gto`, so the in-cell mask must be 3-D too or
    // the integrated particle number comes out wrong (upstream issue 164).
    let dimension = if cell.dimension < 2
        || cell.low_dim_ft_type == pyscf_pbc_gto::types::LowDimFtType::InfVacuum
    {
        cell.dimension as usize
    } else {
        3
    };

    // gen_grid.py:167-168
    let rcut = pyscf_pbc_gto::estimate_rcut_for_eval(cell, 0)?
        .into_iter()
        .fold(0.0_f64, f64::max);
    let ls = pyscf_pbc_gto::lattice::get_lattice_ls(cell, Some(rcut), None, true)?;

    // gen_grid.py:170-172 — the supercell atom positions and the per-symbol
    // atomic grid templates.
    let atom_coords = cell.mol.atom_coords();
    let charges: Vec<u32> = cell.mol.atom_charges().iter().map(|&z| z as u32).collect();
    let tab = pyscf_grids::gen_atomic_grids(config, &cell.mol);

    // gen_grid.py:175 — `b = cell.reciprocal_vectors(norm_to=1)`, the matrix
    // that maps a Cartesian point onto FRACTIONAL coordinates.
    let b = cell.reciprocal_vectors(1.0)?;

    let tol = 1e-15_f64;
    let mut coords_all: Vec<[f64; 3]> = Vec::new();
    let mut weights_all: Vec<f64> = Vec::new();
    // The supercell atom whose grid produced each stored chunk, plus the chunk
    // boundaries — upstream's `supatm_idx` / `offs`.
    let mut sup_atoms: Vec<[f64; 3]> = Vec::new();
    let mut offs: Vec<usize> = vec![0];

    for l in &ls {
        for ia in 0..cell.mol.natm {
            let center = [
                atom_coords[ia][0] + l[0],
                atom_coords[ia][1] + l[1],
                atom_coords[ia][2] + l[2],
            ];
            let template = tab.get(&charges[ia]).ok_or_else(|| {
                err(format!("BeckeGrids: no atomic grid for Z = {}", charges[ia]))
            })?;

            // gen_grid.py:180-206 — keep only the points inside the unit cell,
            // halving the weight of a point exactly on a face.
            let mut kept_coords: Vec<[f64; 3]> = Vec::new();
            let mut kept_vol: Vec<f64> = Vec::new();
            for (p, v) in template.coords.iter().zip(&template.vol) {
                let r = [p[0] + center[0], p[1] + center[1], p[2] + center[2]];
                // c = b . r — the fractional coordinate.
                let mut c = [0.0_f64; 3];
                for (i, ci) in c.iter_mut().enumerate() {
                    *ci = b[i][0] * r[0] + b[i][1] * r[1] + b[i][2] * r[2];
                }
                let mut inside = true;
                for ci in c.iter().take(dimension) {
                    inside &= *ci > -0.5 - tol && *ci < 0.5 + tol;
                }
                if !inside {
                    continue;
                }
                let mut vol = *v;
                for ci in c.iter().take(dimension) {
                    if (ci + 0.5).abs() < tol || (ci - 0.5).abs() < tol {
                        vol *= 0.5;
                    }
                }
                kept_coords.push(r);
                kept_vol.push(vol);
            }

            // gen_grid.py:188 — an image contributing eight points or fewer is
            // dropped entirely (upstream's `if vol.size > 8`).
            if kept_vol.len() > 8 {
                offs.push(offs[offs.len() - 1] + kept_vol.len());
                coords_all.extend(kept_coords);
                weights_all.extend(kept_vol);
                sup_atoms.push(center);
            }
        }
    }

    if coords_all.is_empty() {
        return Ok((coords_all, weights_all));
    }

    // gen_grid.py:216-238 — the Becke partition over the SUPERCELL atom list,
    // with NO radii adjustment (upstream passes a null radii table).
    let atm_dist = inter_distance(&sup_atoms);
    let pbecke = gen_grid_partition(&coords_all, &sup_atoms, &atm_dist, None, original_becke);

    // `weights /= pbecke.sum(axis=0)`, then `weights[chunk_ia] *= pbecke[ia]`.
    let ngrids = coords_all.len();
    let mut denom = vec![0.0_f64; ngrids];
    for row in &pbecke {
        for (g, d) in denom.iter_mut().enumerate() {
            *d += row[g];
        }
    }
    for (g, w) in weights_all.iter_mut().enumerate() {
        if denom[g] != 0.0 {
            *w /= denom[g];
        }
    }
    for ia in 0..sup_atoms.len() {
        let (i0, i1) = (offs[ia], offs[ia + 1]);
        for g in i0..i1 {
            weights_all[g] *= pbecke[ia][g];
        }
    }

    Ok((coords_all, weights_all))
}

/// The quadrature the periodic `NumInt` integrates over — either grid type.
#[derive(Debug, Clone)]
pub enum PeriodicGrids {
    /// The FFT box.
    Uniform(UniformGrids),
    /// Becke-partitioned atomic grids.
    Becke(BeckeGrids),
}

impl PeriodicGrids {
    /// Grid coordinates.
    ///
    /// # Errors
    /// [`PbcDftError`] when a [`BeckeGrids`] has not been built.
    pub fn coords(&self) -> Result<&[[f64; 3]], PbcDftError> {
        match self {
            PeriodicGrids::Uniform(g) => Ok(&g.coords),
            PeriodicGrids::Becke(g) => g
                .coords
                .as_deref()
                .ok_or_else(|| err("pbc NumInt: BeckeGrids has not been built")),
        }
    }

    /// Integration weights.
    ///
    /// # Errors
    /// As [`PeriodicGrids::coords`].
    pub fn weights(&self) -> Result<&[f64], PbcDftError> {
        match self {
            PeriodicGrids::Uniform(g) => Ok(&g.weights),
            PeriodicGrids::Becke(g) => g
                .weights
                .as_deref()
                .ok_or_else(|| err("pbc NumInt: BeckeGrids has not been built")),
        }
    }

    /// Grid-point count.
    pub fn size(&self) -> usize {
        match self {
            PeriodicGrids::Uniform(g) => g.size(),
            PeriodicGrids::Becke(g) => g.size(),
        }
    }

    /// `true` when the grid holds no points.
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// The uniform grid on `cell.mesh` — upstream's `KohnShamDFT.grids`
    /// default.
    ///
    /// # Errors
    /// Propagates [`UniformGrids::build`].
    pub fn uniform(cell: &Cell, mesh: Option<[usize; 3]>) -> Result<Self, PbcDftError> {
        Ok(PeriodicGrids::Uniform(UniformGrids::build(cell, mesh)?))
    }

    /// A built [`BeckeGrids`].
    ///
    /// # Errors
    /// Propagates [`BeckeGrids::build`].
    pub fn becke(cell: &Cell, config: Grids) -> Result<Self, PbcDftError> {
        let mut g = BeckeGrids {
            config,
            coords: None,
            weights: None,
        };
        g.build(cell)?;
        Ok(PeriodicGrids::Becke(g))
    }
}
