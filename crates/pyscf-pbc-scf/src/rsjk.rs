//! `rsjk` — range-separated J/K with NO density fitting
//! (`pyscf/pbc/scf/rsjk.py`), plan 14-08 Task 4.
//!
//! # STATUS: BLOCKED, and the blocker is the same one that stopped 14-07
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
//! around the standard symbol — the exact capability cintx's safe API does not
//! expose. `ExecutionOptions` (`cintx-runtime/src/options.rs:96`) carries
//! `f12_zeta` (`env[9]`), `rinv_orig` and `common_orig`, and nothing reads
//! `env[8]`. This repository already records the gap as Phase 4's Open
//! Question A5 / cintx#11 in `crates/pyscf-gto/src/range_coulomb.rs`.
//!
//! So `rsjk` cannot be built without substituting the full-range kernel for the
//! short-range one, which would produce a J/K builder that runs, converges, and
//! is silently not `rsjk`. [`RangeSeparatedJkBuilder::build`] therefore refuses
//! (D-PBC-20).
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

/// The one-line reason `rsjk` is refused. Deliberately the same text
/// [`pyscf_pbc_df::rsdf_builder::CINTX_SR_GAP`] carries, because it is the same
/// missing capability — a reader who hits one should recognise the other.
pub const CINTX_SR_GAP: &str = pyscf_pbc_df::rsdf_builder::CINTX_SR_GAP;

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
                what: CINTX_SR_GAP,
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
                what: CINTX_SR_GAP,
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
