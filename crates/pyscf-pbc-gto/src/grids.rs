//! Periodic real-space grids — plan 11-04, port of
//! `pyscf/pbc/dft/gen_grid.py:64-148` (`UniformGrids`).
//!
//! # Why this lives in `pyscf-pbc-gto` and not `pyscf-pbc-dft`
//!
//! Upstream puts `UniformGrids` in `pbc/dft/gen_grid.py`, and `pbc/df/fft.py`
//! imports it from there. That edge would make `pyscf-pbc-df` depend on
//! `pyscf-pbc-dft`, which depends on `pyscf-pbc-df` — a cycle. PBC-MASTER-PLAN
//! plan 11-04 resolves it by putting the type here, at the bottom of the
//! periodic DAG, and having `pyscf-pbc-dft` re-export it.
//!
//! The Becke-partitioned atomic grids (`gen_grid.py:150-238`, `BeckeGrids`) are
//! a DFT quantity with no FFTDF consumer; they land with periodic `NumInt` in
//! Phase 12.

use crate::cell::Cell;
use pyscf_core::PyscfRsError;

/// `gen_grid.UniformGrids` — the FFT box as a quadrature grid.
///
/// The weights are the uniform `vol / ngrids`; the coordinates are
/// [`crate::gv::get_uniform_grids`] with upstream's `wrap_around = True`
/// default, so the grid is centred on the origin rather than filling `[0, 1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformGrids {
    /// The FFT mesh this grid samples.
    pub mesh: [usize; 3],
    /// `(ngrids, 3)` Cartesian grid points in Bohr, C-order over `(x, y, z)`.
    pub coords: Vec<[f64; 3]>,
    /// `vol / ngrids`, repeated. Kept as a vector so that a caller written
    /// against a general quadrature grid needs no special case.
    pub weights: Vec<f64>,
}

impl UniformGrids {
    /// Build the grid for `cell` at `mesh` (`None` uses `cell.mesh`).
    ///
    /// # Errors
    /// Propagates [`crate::gv::get_uniform_grids`] — an unset mesh or a zero
    /// axis.
    pub fn build(cell: &Cell, mesh: Option<[usize; 3]>) -> Result<Self, PyscfRsError> {
        let mesh = match mesh {
            Some(m) => m,
            None => cell.try_mesh()?,
        };
        let coords = crate::gv::get_uniform_grids(cell, Some(mesh), true)?;
        let ngrids = coords.len();
        let w = cell.vol() / ngrids as f64;
        Ok(Self {
            mesh,
            coords,
            weights: vec![w; ngrids],
        })
    }

    /// `np.prod(mesh)`.
    pub fn size(&self) -> usize {
        self.coords.len()
    }

    /// `true` when the grid holds no points.
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// The single quadrature weight `vol / ngrids`, which is what the FFT J/K
    /// builders multiply by rather than walking [`UniformGrids::weights`].
    pub fn weight(&self) -> f64 {
        self.weights.first().copied().unwrap_or(0.0)
    }
}

impl Cell {
    /// `gen_grid.UniformGrids(cell)` — see [`UniformGrids::build`].
    ///
    /// # Errors
    /// As [`UniformGrids::build`].
    pub fn uniform_grids(&self, mesh: Option<[usize; 3]>) -> Result<UniformGrids, PyscfRsError> {
        UniformGrids::build(self, mesh)
    }
}
