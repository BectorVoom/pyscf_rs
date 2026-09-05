//! `RSDF` — the user-facing range-separated density-fitting class
//! (`pyscf/pbc/df/rsdf.py`), plan 14-08.
//!
//! # STATUS: shipped (plan 14-07 sub-tasks 7b/7c, on D-PBC-24's cintx `range_omega`).
//!
//! `rsdf.RSGDF` subclasses `df.GDF` and selects `rsdf_builder._RSGDFBuilder`.
//! This port expresses the same relation: [`Rsdf`] IS a [`crate::Gdf`] with
//! [`crate::Gdf::prefer_ccdf`] set to `false`, which routes `make_j3c` through
//! [`crate::gdf_builder::j3c::Scheme::RangeSeparated`]. There is one fitting
//! pipeline and one place a scheme can drift, which is the same reason
//! `_CCMDFBuilder` is a subclass rather than a copy upstream.
//!
//! Measured against upstream's own `df.GDF()` (its default is the RS route) on
//! He-fcc `sto-3g` 2x2x2, `conv_tol = 1e-12`:
//!
//! | quantity | upstream | this port | error |
//! |---|---|---|---|
//! | RSDF `KRHF` | -2.80842508717097 | -2.80842508693849 | **2.32e-10** |
//! | GDF (CC) `KRHF` | -2.80842508664874 | -2.80842508692377 | **2.75e-10** |
//!
//! The two routes agree with upstream to the same order, i.e. the residual is
//! this port's ordinary fitting accuracy and not something range separation
//! introduced. The port's OWN `|CC - RS|` is 1.47e-11 against upstream's
//! 5.222e-10; see [`crate::rsdf_builder`] for why the port's two routes agree
//! with each other more closely than upstream's two do.
//!
//! [`get_aux_chg`] is independent of all of that and ships: it is
//! `ft_ao(auxcell, G = 0).real`, the same monopole convention plan 14-01 Task 3
//! fixed, and it is what identifies the CHARGED auxiliary functions that range
//! separation has to treat specially.

use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;

/// `get_aux_chg(auxcell)` — `rsdf.py:65-73`.
///
/// `\int_Omega chi_P(r) dr` for every auxiliary AO, evaluated as the `G = 0`
/// Fourier transform. Identical to
/// [`crate::rsdf_builder::gaussian_int`]; both names exist upstream and both
/// are kept so a reader of either file finds what they expect.
///
/// # Errors
/// Propagates the single-centre FT.
pub fn get_aux_chg(auxcell: &Cell) -> Result<Vec<f64>, PbcDfError> {
    crate::rsdf_builder::gaussian_int(auxcell)
}

/// `RSGDF` — `rsdf.py:75-322`.
///
/// A [`crate::Gdf`] whose fitting route is the range-separated one. Every
/// method is the `GDF` one; the only difference is
/// [`crate::Gdf::prefer_ccdf`], which upstream also expresses as a flag
/// (`_prefer_ccdf`) rather than as a separate code path.
#[derive(Debug)]
pub struct Rsdf {
    /// The underlying `GDF`, with `prefer_ccdf = false`.
    pub gdf: crate::Gdf,
}

impl Rsdf {
    /// An `RSDF` on `cell` at `kpts`.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Self {
        let mut gdf = crate::Gdf::new(cell, kpts);
        gdf.prefer_ccdf = false;
        Self { gdf }
    }

    /// The cell.
    pub fn cell(&self) -> &Cell {
        &self.gdf.cell
    }

    /// The sampling k-points.
    pub fn kpts(&self) -> &[[f64; 3]] {
        &self.gdf.kpts
    }

    /// The `(omega, mesh, ke_cutoff)` this builder runs at.
    ///
    /// # Errors
    /// Propagates [`crate::rsdf_builder::guess_omega`].
    pub fn guess_omega(&self) -> Result<(f64, [usize; 3], f64), PbcDfError> {
        crate::rsdf_builder::guess_omega(&self.gdf.cell, &self.gdf.kpts, None)
    }

    /// `RSGDF.build()` — builds the fitted tensor.
    ///
    /// # Errors
    /// Propagates the builder.
    pub fn build(&mut self) -> Result<(), PbcDfError> {
        self.gdf.build()
    }
}

impl crate::traits::PeriodicDf for Rsdf {
    fn cell(&self) -> &Cell {
        self.gdf.cell()
    }
    fn mesh(&self) -> [usize; 3] {
        self.gdf.mesh()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        crate::traits::PeriodicDf::kpts(&self.gdf)
    }
    fn build(&mut self) -> Result<(), PbcDfError> {
        crate::traits::PeriodicDf::build(&mut self.gdf)
    }
    fn get_nuc(&self, kpts: &[[f64; 3]]) -> Result<Vec<pyscf_algebra::CTensor>, PbcDfError> {
        self.gdf.get_nuc(kpts)
    }
    fn get_pp(&self, kpts: &[[f64; 3]]) -> Result<Vec<pyscf_algebra::CTensor>, PbcDfError> {
        self.gdf.get_pp(kpts)
    }
    fn name(&self) -> &'static str {
        "RSGDF"
    }
    fn get_jk(
        &self,
        dms: &[crate::df_jk::KMats],
        kpts: &[[f64; 3]],
        opts: crate::traits::JkOpts<'_>,
    ) -> Result<crate::traits::JkResult, PbcDfError> {
        self.gdf.get_jk(dms, kpts, opts)
    }
    fn ao2mo(
        &self,
        mos: [&crate::MoCoeff; 4],
        kidx: [usize; 4],
        compact: bool,
    ) -> Result<crate::Eri, PbcDfError> {
        crate::traits::PeriodicDf::ao2mo(&self.gdf, mos, kidx, compact)
    }
    fn get_ao_eri(&self, kidx: [usize; 4], compact: bool) -> Result<crate::Eri, PbcDfError> {
        crate::traits::PeriodicDf::get_ao_eri(&self.gdf, kidx, compact)
    }
    fn ao2mo_7d(&self, mos: crate::MoKpts<'_>, factor: f64) -> Result<crate::Eri7d, PbcDfError> {
        crate::traits::PeriodicDf::ao2mo_7d(&self.gdf, mos, factor)
    }
    fn has_cderi(&self) -> bool {
        true
    }
    fn sr_loop(
        &self,
        ki: usize,
        kj: usize,
        compact: bool,
    ) -> Result<Vec<crate::SrBlock>, PbcDfError> {
        self.gdf.sr_loop(ki, kj, compact)
    }
    fn get_naoaux(&self) -> Result<usize, PbcDfError> {
        self.gdf.get_naoaux()
    }
}
