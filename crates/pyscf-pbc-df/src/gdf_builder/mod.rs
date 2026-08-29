//! `_CCGDFBuilder` — the compensated-charge Gaussian density-fitting builder
//! (`pyscf/pbc/df/gdf_builder.py`, plan 14-02).
//!
//! # The scheme in one paragraph
//!
//! A 3-centre integral against a *charged* auxiliary function decays as `1/R`,
//! so its lattice sum does not converge (plan 14-01 measured exactly that). GDF
//! subtracts a smooth Gaussian of the same charge from every auxiliary
//! function: what is left is short-ranged and sums in real space, and the piece
//! that was subtracted has an analytic Fourier transform and is added back in
//! reciprocal space through `ft_ao`. [`fuse`] builds the fused
//! (auxiliary + model-charge) cell and performs the subtraction; `j3c` adds the
//! long-range half back.
//!
//! # `exclude_dd_block` is FALSE here (D-PBC-23)
//!
//! Upstream defaults it to `true` and diverts the smooth-smooth block of
//! `(ij|L)` into an FFT. That is not screening — it re-routes, and it is worth
//! a measured **1.835e-8 Ha** on diamond and **exactly 0** on the all-electron
//! He-fcc control. This port takes the `false` branch and prices the deferral;
//! setting the flag `true` returns `NotYetImplemented { phase: 17 }` rather
//! than being silently ignored.

pub mod eta;
pub mod fuse;
pub mod j2c;
pub mod j3c;

pub use eta::{
    ETA_MIN, EtaChoice, estimate_eta_for_ke_cutoff, estimate_eta_min, estimate_ke_cutoff_for_eta,
    estimate_rcut, guess_eta,
};
pub use fuse::{FusedCell, auxbar, compensate_nuccell, fuse_auxcell, make_modchg_basis};
pub use j2c::{CdJ2c, J2cTag, LINEAR_DEP_THRESHOLD, decompose_j2c, get_2c2e, weighted_coulg};
pub use j3c::{Cderi, CderiBlock, make_j3c, outcore_auxe2, weighted_ft_ao};

pub use self::CcGdfBuilder as Builder;

use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;
use crate::incore::Aosym;

/// `_CCGDFBuilder` — the state upstream carries on the builder object
/// (`gdf_builder.py:48-137`).
#[derive(Debug, Clone)]
pub struct CcGdfBuilder {
    /// The orbital cell.
    pub cell: Cell,
    /// The sampling k-points.
    pub kpts: Vec<[f64; 3]>,
    /// The auxiliary basis name; `None` runs `make_auxbasis`.
    pub auxbasis: Option<String>,
    /// The compensating-charge exponent and the mesh it implies. `None` until
    /// [`CcGdfBuilder::build`].
    pub eta: Option<EtaChoice>,
    /// The fused auxiliary cell. `None` until [`CcGdfBuilder::build`].
    pub fused: Option<FusedCell>,
    /// **D-PBC-23.** Upstream defaults this to `true`; this port ships `false`
    /// and prices the deferral (1.835e-8 Ha on diamond, exactly 0 on He-fcc —
    /// `measurements/ddblock.py`). Setting it `true` is refused rather than
    /// ignored.
    pub exclude_dd_block: bool,
    /// `j2c_eig_always` — force the eigenvalue route. Upstream default `false`.
    pub j2c_eig_always: bool,
    /// `linear_dep_threshold`.
    pub linear_dep_threshold: f64,
    /// The 3-centre image radius. `None` uses `gdf_builder::estimate_rcut`.
    pub rcut: Option<f64>,
}

impl CcGdfBuilder {
    /// A builder on `cell` at `kpts`, matching upstream's defaults.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Self {
        Self {
            cell,
            kpts: if kpts.is_empty() {
                vec![[0.0; 3]]
            } else {
                kpts.to_vec()
            },
            auxbasis: None,
            eta: None,
            fused: None,
            exclude_dd_block: false,
            j2c_eig_always: false,
            linear_dep_threshold: j2c::LINEAR_DEP_THRESHOLD,
            rcut: None,
        }
    }

    /// `_CCGDFBuilder.build()` — `gdf_builder.py:80-137`, minus the
    /// `_RangeSeparatedCell` / `ExtendedMole` supermole (D-PBC-23).
    ///
    /// # Errors
    /// [`PyscfRsError::NotYetImplemented`] `{ phase: 17 }` when
    /// [`Self::exclude_dd_block`] is set, and propagates the auxiliary-cell
    /// build.
    pub fn build(&mut self) -> Result<(), PbcDfError> {
        if self.exclude_dd_block {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 17,
                what: "exclude_dd_block — ft_ao._RangeSeparatedCell + _int_dd_block \
                       (D-PBC-23; worth 1.835e-8 Ha on diamond, 0 on He-fcc)",
            }));
        }
        let aux = crate::incore::make_modrho_basis(
            &self.cell,
            self.auxbasis.as_deref(),
            None,
        )?;
        // `_guess_eta(auxcell, kpts, self.mesh)` — the AUXCELL, not the cell.
        let eta = eta::guess_eta(&aux.cell, &self.kpts, self.eta.map(|e| e.mesh))?;
        self.fused = Some(fuse::fuse_auxcell(
            &self.cell,
            self.auxbasis.as_deref(),
            eta.eta,
        )?);
        self.eta = Some(eta);
        Ok(())
    }

    /// `make_j3c(...)` on this builder's state.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] when [`Self::build`] has not run, and propagates
    /// every stage of the 3-centre build.
    pub fn make_j3c(&self, aosym: Aosym, j_only: bool) -> Result<j3c::Cderi, PbcDfError> {
        let (Some(fused), Some(eta)) = (self.fused.as_ref(), self.eta) else {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(
                    "CcGdfBuilder::make_j3c: call build() first".into(),
                ),
            )));
        };
        let rcut = self
            .rcut
            .or_else(|| Some(eta::estimate_rcut(&self.cell, &fused.fused.cell, None)));
        j3c::make_j3c(
            &self.cell,
            fused,
            &self.kpts,
            aosym,
            eta.mesh,
            j_only,
            self.j2c_eig_always,
            rcut,
        )
    }
}
