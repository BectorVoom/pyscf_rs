//! `rsjk` — range-separated J/K with NO density fitting
//! (`pyscf/pbc/scf/rsjk.py`), plan 14-08 Task 4.
//!
//! # STATUS: NOT PORTED. The blocker that stopped 14-07 is gone.
//!
//! `rsjk` is a different animal from every other builder in Phase 14: it has no
//! auxiliary basis and no `cderi`. It splits the Coulomb operator itself and
//! builds `vj`/`vk` *exactly* —
//!
//! ```text
//! 1/r = erfc(w r)/r  +  erf(w r)/r
//!       real-space         reciprocal-space
//!       int2e over a       ft_aopair over a
//!       supermole          coarse plane-wave grid
//! ```
//!
//! — which is why `14-08-PLAN.md` Task 5.3 insists it be gated against **FFTDF
//! and not GDF**: gating an exact builder against a fitted one would hide a
//! real error behind the 1.2e-3 fitting gap.
//!
//! **Its short-range half is a short-range `int2e`.** `rsjk.py:136-187` builds
//! `supmol_sr` and sets `supmol_sr.omega = -self.omega` before evaluating
//! `int2e` over it. That is libcint's `PTR_RANGE_OMEGA` (`env[8]`) toggle
//! around the standard symbol, and cintx's safe API **now exposes it**:
//! `ExecutionOptions::range_omega` / `SessionBuilder::with_range_omega`
//! (D-PBC-24), on the same sign convention [`JkOpts::omega`] already uses.
//! Phase 4's Open Question A5 / cintx#11 in
//! `crates/pyscf-gto/src/range_coulomb.rs` is answered.
//!
//! **cintx was necessary but NOT sufficient, and the remaining blocker is
//! Phase 17.** Two independent things are missing, neither of them an integral
//! flag:
//!
//! 1. **The supermole.** `rsjk.py:150-200` builds the short-range half over
//!    `ft_ao._RangeSeparatedCell.from_cell` + `ft_ao.ExtendedMole.from_cell`
//!    followed by `strip_basis`, and `_get_jk_sr` (`:267-436`) indexes the
//!    result by `supmol.bas_mask` with shape `(bvk_ncells, rs_nbas, nimgs)`.
//!    This port has neither type — D-PBC-21 / D-PBC-23 defer both to Phase 17,
//!    which is also what `exclude_dd_block` and `strip_basis` wait on.
//! 2. **A periodic 4-centre `int2e` driver.** `_get_jk_sr` drives
//!    `PBCVHF_direct_drv1`, a SCREENED direct sweep. `grep int2e` across every
//!    `pyscf-pbc-*` crate finds one doc comment and no implementation:
//!    [`pyscf_pbc_df::incore::aux_e2`] is 3-centre.
//!
//! Point 2 is why the trick that unblocked RSDF does not transfer.
//! `_RSGDFBuilder` could be ported by treating every basis function as compact
//! and paying for the missing compact/smooth split in grid points — a
//! degenerate case that is merely slower. Here the screening **is** the
//! algorithm: an unscreened 4-centre sweep over the BvK images is not slower
//! but infeasible, so there is no correct-but-slow version to fall back on.
//!
//! [`RangeSeparatedJkBuilder::build`] therefore still refuses (D-PBC-20), and
//! **must not** be finished by substituting the full-range kernel: because
//! `rsjk` is EXACT, a wrong answer would land within the DF fitting error of a
//! correct GDF and look entirely plausible. Sequence it after Phase 17's
//! supermole and size it as its own plan.
//!
//! # What ships anyway
//!
//! The ω machinery `rsjk.build` uses is **the same `_guess_omega` /
//! `estimate_ke_cutoff_for_omega` / `estimate_rcut` family plan 14-07 shipped**
//! (`rsjk.py:145-186` imports them from `rsdf_builder`), and this type exposes
//! it through [`RangeSeparatedJkBuilder::guess_omega`] — so the parameters
//! `rsjk` would run at are computed and gated today, and only the integral is
//! missing.
//!
//! # It is NOT a `PeriodicDf`, and that is deliberate
//!
//! `14-08-PLAN.md`: "it must not be given a `PeriodicDf` impl whose
//! `sr_loop`/`get_naoaux` half is a lie." It has no `cderi` to loop over and no
//! auxiliary count to report. It gets its own narrow surface — `build` and
//! `get_jk` — and a driver would take it as an alternative `get_veff` source,
//! not as a density-fitting builder.

use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::error::PbcDfError;
use pyscf_pbc_df::traits::{JkOpts, JkResult};
use pyscf_pbc_gto::Cell;

/// The one-line reason `rsjk` is refused, re-exported from
/// [`pyscf_pbc_df::rsdf_builder::RS_BUILDER_GAP`].
///
/// That constant used to cover BOTH this and `_RSGDFBuilder`. Plan 14-07
/// sub-tasks 7b/7c shipped the latter (Phase 14 Gate 3 is MET), so the text now
/// names only this — the last consumer of D-PBC-24 still unwritten.
pub const RS_BUILDER_GAP: &str = pyscf_pbc_df::rsdf_builder::RS_BUILDER_GAP;

/// `RangeSeparatedJKBuilder` — `rsjk.py:47-…`.
#[derive(Debug, Clone)]
pub struct RangeSeparatedJkBuilder {
    /// The cell.
    pub cell: Cell,
    /// The sampling k-points.
    pub kpts: Vec<[f64; 3]>,
    /// The range-separation parameter. `None` lets `_guess_omega` choose.
    pub omega: Option<f64>,
    /// The long-range plane-wave mesh. `None` lets `_guess_omega` choose.
    pub mesh: Option<[usize; 3]>,
    /// **D-PBC-23.** `false` here as everywhere in this phase.
    pub exclude_dd_block: bool,
}

impl RangeSeparatedJkBuilder {
    /// A builder on `cell` at `kpts`.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Self {
        Self {
            cell,
            kpts: if kpts.is_empty() {
                vec![[0.0; 3]]
            } else {
                kpts.to_vec()
            },
            omega: None,
            mesh: None,
            exclude_dd_block: false,
        }
    }

    /// The `(omega, mesh, ke_cutoff)` `rsjk.build` would run at —
    /// `rsjk.py:145-151`, which is plan 14-07's `_guess_omega`.
    ///
    /// # Errors
    /// Propagates [`pyscf_pbc_df::rsdf_builder::guess_omega`].
    pub fn guess_omega(&self) -> Result<(f64, [usize; 3], f64), PbcDfError> {
        if let Some(w) = self.omega {
            let mesh = match self.mesh {
                Some(m) => m,
                None => {
                    let ke = pyscf_pbc_df::rsdf_builder::estimate_ke_cutoff_for_omega(
                        &self.cell, w, None,
                    );
                    self.cell.cutoff_to_mesh(ke)?
                }
            };
            let ke = pyscf_pbc_df::rsdf_builder::estimate_ke_cutoff_for_omega(&self.cell, w, None);
            return Ok((w, mesh, ke));
        }
        pyscf_pbc_df::rsdf_builder::guess_omega(&self.cell, &self.kpts, self.mesh)
    }

    /// `build(omega, intor='int2e')` — **refused**; see the module docs.
    ///
    /// # Errors
    /// Always [`PyscfRsError::NotYetImplemented`], naming the cintx gap.
    pub fn build(&mut self) -> Result<(), PbcDfError> {
        Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 14,
                what: RS_BUILDER_GAP,
            },
        ))
    }

    /// `get_jk(dm, hermi, kpts, kpts_band, with_j, with_k, omega, exxdiv)` —
    /// **refused**; the short-range half cannot be evaluated.
    ///
    /// # Errors
    /// Always [`PyscfRsError::NotYetImplemented`].
    pub fn get_jk(
        &self,
        _dms: &[KMats],
        _kpts: &[[f64; 3]],
        _opts: JkOpts<'_>,
    ) -> Result<JkResult, PbcDfError> {
        Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 14,
                what: RS_BUILDER_GAP,
            },
        ))
    }

    /// The MPI and multi-threaded partitioning variants — a NON-GOAL of this
    /// phase (`14-CONTEXT.md`: "one correct serial path").
    ///
    /// # Errors
    /// Always [`PyscfRsError::NotYetImplemented`] `{ phase: 19 }`.
    pub fn get_jk_mpi(&self) -> Result<JkResult, PbcDfError> {
        Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 19,
                what: "rsjk's MPI / multi-threaded partitioning variants — \
                       14-CONTEXT.md makes them a non-goal: one correct serial path",
            },
        ))
    }
}
