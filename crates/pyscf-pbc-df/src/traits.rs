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
    /// Range-separation parameter for an RSH functional, mirroring upstream's
    /// `FFTDF.get_jk(..., omega=...)` (`pbc/df/fft.py`), which forwards it by
    /// setting `cell.omega` inside `range_coulomb`. This port has no mutable
    /// `cell.omega`, so the value is threaded explicitly into `get_coulG`
    /// instead: `Some(w > 0)` selects the LONG-range kernel, `Some(w < 0)` the
    /// SHORT-range one, `None` the full Coulomb kernel.
    pub omega: Option<f64>,
    /// W-08 (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`) — exploit the
    /// conjugate relation between the `(k1, k2)` and `(k2, k1)` members of the
    /// exchange pair loop, halving the number of 3-D transforms.
    ///
    /// **Opt-in, and it CHANGES THE RESULT in the last bits** (it changes which
    /// terms are summed into `vk[k]` and in what order), so it is `false`
    /// everywhere by default and any gate run with it on must be re-baselined
    /// rather than inherit the existing tolerances. Only `fftdf` honours it;
    /// every other builder ignores it. See
    /// [`crate::fft_jk::get_k_kpts_opts`] for the identity it rests on and the
    /// preconditions it checks.
    pub kk_symmetry: bool,
}

impl JkOpts<'_> {
    /// `hermi = 1`, both matrices, no band k-points, no exxdiv, no k-pair
    /// symmetry.
    pub fn hermitian() -> Self {
        Self {
            hermi: 1,
            kpts_band: None,
            with_j: true,
            with_k: true,
            exxdiv: None,
            omega: None,
            kk_symmetry: false,
        }
    }

    /// `PYSCF_PBC_KK_SYMMETRY`, read once — whether an SCF driver should turn
    /// W-08's k-pair symmetry on.
    ///
    /// Default `false`. This exists so the accuracy gate can be RE-BASELINED
    /// with the flag on (W-08's own TEST demands "a separate gate run with the
    /// flag on whose tolerance is re-baselined and recorded") without every
    /// driver growing a new argument. `1`/`true`/`yes`/`on` enable it; anything
    /// else, including unset, does not.
    pub fn kk_symmetry_default() -> bool {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("PYSCF_PBC_KK_SYMMETRY").is_ok_and(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
        })
    }
}

/// A periodic density-fitting builder.
///
/// Object-safe by construction (every method takes `&self`/`&mut self`, none is
/// generic, none returns `Self`), which is what lets plan 13-07 store it as
/// `Box<dyn PeriodicDf>` in every k-point driver (D-PBC-22). The `Debug`
/// supertrait is required because those drivers `#[derive(Debug)]`.
pub trait PeriodicDf: std::fmt::Debug {
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
    /// A short name for `dump_flags` and chkfile provenance — `"FFTDF"`,
    /// `"AFTDF"`, … (plan 13-07 STEP 4).
    fn name(&self) -> &'static str;
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
