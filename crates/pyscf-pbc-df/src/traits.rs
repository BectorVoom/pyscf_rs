//! The density-fitting seam every periodic builder plugs into — plan 11-05.
//!
//! FFTDF is the first implementor; AFTDF (Phase 13) and GDF/MDF/RSDF (Phase 14)
//! are the rest (D-PBC-09). The trait is what `pyscf-pbc-scf` programs against,
//! so a later builder is a drop-in swap with no driver change.

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{Cell, ExxDiv};

use crate::df_jk::KMats;
use crate::error::PbcDfError;

/// The `(vj, vk)` pair a J/K build returns. Either half is `None` when the
/// caller asked for the other only.
#[derive(Debug, Clone, Default)]
pub struct JkResult {
    /// `vj[iset][kband]`, `nao x nao` row-major.
    pub vj: Option<Vec<KMats>>,
    /// `vk[iset][kband]`, `nao x nao` row-major.
    pub vk: Option<Vec<KMats>>,
}

/// Optional extras a caller can hand a J/K build.
#[derive(Debug, Clone, Copy, Default)]
pub struct JkOpts<'a> {
    /// `hermi` — 1 when the density matrices are Hermitian (the SCF case).
    pub hermi: i32,
    /// Arbitrary "band" k-points at which to evaluate the matrices. `None`
    /// evaluates at the sampling k-points.
    pub kpts_band: Option<&'a [[f64; 3]]>,
    /// Compute `vj`.
    pub with_j: bool,
    /// Compute `vk`.
    pub with_k: bool,
    /// How to treat the exchange divergence at `G + k = 0`.
    pub exxdiv: Option<ExxDiv>,
}

impl JkOpts<'_> {
    /// `hermi = 1`, both matrices, no band k-points, no exxdiv.
    pub fn hermitian() -> Self {
        Self {
            hermi: 1,
            kpts_band: None,
            with_j: true,
            with_k: true,
            exxdiv: None,
        }
    }
}

/// A periodic density-fitting builder.
pub trait PeriodicDf {
    /// The cell the builder is bound to.
    fn cell(&self) -> &Cell;
    /// The FFT mesh (or, for a Gaussian builder, the auxiliary mesh).
    fn mesh(&self) -> [usize; 3];
    /// The sampling k-points.
    fn kpts(&self) -> &[[f64; 3]];
    /// Precompute whatever the builder caches. Idempotent.
    ///
    /// # Errors
    /// Builder-specific; FFTDF's builds the uniform grid and the AO table.
    fn build(&mut self) -> Result<(), PbcDfError>;
    /// Nuclear attraction `V_ne` at each k-point, `nao x nao` row-major.
    ///
    /// # Errors
    /// Builder-specific.
    fn get_nuc(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError>;
    /// GTH pseudopotential `V_pp = V_loc + V_nl` at each k-point.
    ///
    /// # Errors
    /// Builder-specific.
    fn get_pp(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError>;
    /// Coulomb and exchange matrices for `nset` density-matrix channels.
    ///
    /// # Errors
    /// Builder-specific.
    fn get_jk(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        opts: JkOpts<'_>,
    ) -> Result<JkResult, PbcDfError>;
}
