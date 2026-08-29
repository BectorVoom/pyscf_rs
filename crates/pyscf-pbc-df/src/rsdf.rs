//! `RSDF` — the user-facing range-separated density-fitting class
//! (`pyscf/pbc/df/rsdf.py`), plan 14-08.
//!
//! # STATUS: `get_aux_chg` ships. `RSGDF` itself is BLOCKED.
//!
//! `rsdf.RSGDF` subclasses `df.GDF` and selects `rsdf_builder._RSGDFBuilder`,
//! which plan 14-07 could not ship: cintx's safe API has no `range_omega`
//! (libcint `env[8]`) knob, so there is no short-range `int3c2e`/`int2c2e` to
//! build the short-range half of the fit from. The full evidence is in
//! [`crate::rsdf_builder`]'s module docs; the one-line reason is
//! [`crate::rsdf_builder::CINTX_SR_GAP`].
//!
//! Consequences, all recorded in `14-VERIFICATION.md`:
//!
//! * **Gate 3 is unreachable this phase.** It compares `|E(GDF) − E(RSDF)|`
//!   against upstream's own floor (1.353e-08 on diamond 2×2×2, 4.566e-09 at
//!   gamma, 1.113e-10 on He-fcc). With one of the two builders missing there is
//!   nothing to compare.
//! * Plan 14-07 Task 7d's flip of `Gdf::prefer_ccdf` to `false` does not
//!   happen, and the committed `df_swap` baseline therefore does not move.
//!
//! [`get_aux_chg`] is independent of all of that and ships: it is
//! `ft_ao(auxcell, G = 0).real`, the same monopole convention plan 14-01 Task 3
//! fixed, and it is what identifies the CHARGED auxiliary functions that range
//! separation has to treat specially.

use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;
use crate::rsdf_builder::CINTX_SR_GAP;

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

/// `RSGDF` — `rsdf.py:75-322`. **Refused**; see the module docs.
#[derive(Debug, Clone)]
pub struct Rsdf {
    /// The cell.
    pub cell: Cell,
    /// The sampling k-points.
    pub kpts: Vec<[f64; 3]>,
    /// Auxiliary basis name.
    pub auxbasis: Option<String>,
}

impl Rsdf {
    /// An `RSDF` on `cell` at `kpts`.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Self {
        Self {
            cell,
            kpts: if kpts.is_empty() {
                vec![[0.0; 3]]
            } else {
                kpts.to_vec()
            },
            auxbasis: None,
        }
    }

    /// The `(omega, mesh, ke_cutoff)` this builder WOULD run at — the ω half
    /// ships (plan 14-07 sub-task 7a).
    ///
    /// # Errors
    /// Propagates [`crate::rsdf_builder::guess_omega`].
    pub fn guess_omega(&self) -> Result<(f64, [usize; 3], f64), PbcDfError> {
        crate::rsdf_builder::guess_omega(&self.cell, &self.kpts, None)
    }

    /// `RSGDF.build()` — **refused**.
    ///
    /// # Errors
    /// Always [`PyscfRsError::NotYetImplemented`], naming the cintx gap.
    pub fn build(&mut self) -> Result<(), PbcDfError> {
        Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 14,
                what: CINTX_SR_GAP,
            },
        ))
    }
}
