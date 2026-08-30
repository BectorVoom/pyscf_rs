//! `rsdf_builder` — range-separated Gaussian density fitting
//! (`pyscf/pbc/df/rsdf_builder.py`), plan 14-07.
//!
//! # STATUS: sub-task 7a ships. 7b, 7c and 7d are UNBLOCKED but NOT DONE.
//!
//! ## The blocker that stopped 14-07 is gone
//!
//! Plan 14-07 Task 7b said, in its own words:
//!
//! > **Check first that the cintx resolver exposes a short-range `int3c2e`**;
//! > if it does not, this is the plan's one real blocker and it must be
//! > reported as such, not worked around with a numerically different kernel.
//!
//! It did not, and this module recorded five pieces of evidence for that. All
//! five have since been answered by **D-PBC-24** in cintx:
//!
//! 1. `cintx_runtime::ExecutionOptions` carries `range_omega` (libcint's
//!    `PTR_RANGE_OMEGA`, `env[8]`), and `SessionBuilder::with_range_omega` sets
//!    it. It is part of the WORKSPACE query, because short range doubles the
//!    Rys roots.
//! 2. The raw path reads `env[8]`, and the `CINTg0_2e` omega branch is ported
//!    once in `cintx-cubecl::math::range_separation`, called from
//!    `two_electron.rs`, `center_3c2e.rs` and `center_2c2e.rs`.
//! 3. Range separation is still not a distinct symbol, as upstream confirms:
//!    it is an `env[8]` toggle around the STANDARD `int3c2e`/`int2c2e`/`int2e`.
//! 4. `pyscf-gto`'s Open Question A5 / cintx#11 is answered on the safe path,
//!    not only through `mol._env[8]`.
//! 5. The second obstruction named here — that [`crate::incore::aux_e2`] reaches
//!    cintx through `build_image_expanded_with_aux` and never materialises an
//!    `_env`, so `OmegaGuard`'s trick has nothing to write into — turned out
//!    never to matter: ω travels in the OPTIONS, not in the basis.
//!    [`crate::incore::aux_e2`], [`crate::incore::fill_2c2e`] and
//!    [`pyscf_pbc_gto::pbc_intor::PbcIntorOpts::omega`] all take it now, and
//!    `tests/incore.rs` gates `SR(ω) + LR(ω) == full` on the assembled 3-centre
//!    tensor and 2-centre metric.
//!
//! ## What is still missing is this file
//!
//! `_RSGDFBuilder` itself has not been ported. That is sub-tasks 7b and 7c —
//! `get_2c2e` (the short-range metric plus its long-range plane-wave
//! correction), `outcore_auxe2` (the short-range 3-centre half),
//! `add_ft_j3c`, `solve_cderi`, and `_RSNucBuilder` — plus 7d's flip of
//! `Gdf::prefer_ccdf` to `false`. The integrals they need are available; the
//! several hundred lines of `rsdf_builder.py` that assemble them are not
//! written.
//!
//! [`RsGdfBuilder::build`] therefore still refuses, and still for a reason
//! rather than a shrug — but the reason is now [`RS_BUILDER_GAP`], this port's
//! own unfinished work, and no longer a missing capability in another
//! repository. **Do not substitute the full-range kernel** while finishing it:
//! a builder that looks like RSDF, runs and converges is silently a different
//! method, which is the one outcome D-PBC-20 forbids.
//!
//! # What DOES ship, and why it was worth shipping alone
//!
//! [`omega`] — all twelve estimators, `weighted_coulG_LR` / `_SR`, and
//! `_gaussian_int`. They are pure functions of the cell, they are gated
//! against `measurements/omega.out`, and three separate downstream consumers
//! need them regardless: `rsjk` (14-08), RSH functionals (`JkOpts::omega`,
//! already threaded through `get_coulG`), and Phase 17. Plan 14-07 sequenced
//! 7a first precisely so that it could land on its own: "Do 7a completely, with
//! its tests green, before writing a line of 7b — the whole scheme's accuracy
//! is one ω away."
//!
//! # What a caller gets meanwhile
//!
//! [`RsGdfBuilder::build`] returns `NotYetImplemented { phase: 14 }` naming
//! [`RS_BUILDER_GAP`] (D-PBC-20: a deferred branch never returns a silently
//! wrong answer). `Gdf::prefer_ccdf` therefore stays `true` — Task 7d's flip
//! cannot happen yet — and Gate 3 (`|E(GDF) − E(RSDF)|` against upstream's
//! 1.353e-08 floor) is still unreachable. Both are recorded in
//! `14-VERIFICATION.md`.

pub mod omega;

pub use omega::{
    OMEGA_MIN, RCUT_THRESHOLD, estimate_ft_rcut, estimate_ke_cutoff_for_omega, estimate_meshz,
    estimate_omega_for_ke_cutoff, estimate_omega_min, estimate_rcut, estimate_rs_2c2e_rcut,
    gaussian_int, guess_omega, round_off_to_odd_mesh, weighted_coulg_lr, weighted_coulg_sr,
};

use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;

/// The one-line reason every range-separated 3-centre path is still refused.
/// Kept as a constant so the tests can assert on it and so the message cannot
/// drift between call sites.
///
/// It changed meaning once already: it used to name a missing cintx capability,
/// and D-PBC-24 supplied that capability. What it names now is this port's own
/// unfinished `_RSGDFBuilder`. Read the module docs before assuming the old
/// reason still applies.
pub const RS_BUILDER_GAP: &str = "range-separated density fitting — the cintx side is DONE \
     (ExecutionOptions::range_omega, libcint env[8]; incore::aux_e2 and \
     incore::fill_2c2e both take an omega, gated by SR + LR == full in \
     tests/incore.rs), but _RSGDFBuilder itself is not ported: plan 14-07 \
     sub-tasks 7b/7c (get_2c2e's short-range metric and its long-range \
     plane-wave correction, outcore_auxe2's short-range half, add_ft_j3c, \
     solve_cderi, _RSNucBuilder) and 7d. Finish those rather than substituting \
     the full-range kernel, which runs, converges, and is a different method";

/// `_RSGDFBuilder` — `rsdf_builder.py:59-1096`.
///
/// The state is carried so that [`omega`]'s estimators have a natural home and
/// so a later phase can fill in `build` without moving the type; the 3-centre
/// half is refused. See the module docs.
#[derive(Debug, Clone)]
pub struct RsGdfBuilder {
    /// The orbital cell.
    pub cell: Cell,
    /// The sampling k-points.
    pub kpts: Vec<[f64; 3]>,
    /// Auxiliary basis name; `None` runs `make_auxbasis`.
    pub auxbasis: Option<String>,
    /// The range-separation parameter. `None` until `build`.
    pub omega: Option<f64>,
    /// The long-range plane-wave mesh. `None` lets [`guess_omega`] choose.
    pub mesh: Option<[usize; 3]>,
    /// The kinetic-energy cutoff `mesh` implies.
    pub ke_cutoff: Option<f64>,
    /// **D-PBC-23.** `false` here as everywhere in this phase.
    pub exclude_dd_block: bool,
}

impl RsGdfBuilder {
    /// A builder on `cell` at `kpts`, with upstream's defaults.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Self {
        Self {
            cell,
            kpts: if kpts.is_empty() {
                vec![[0.0; 3]]
            } else {
                kpts.to_vec()
            },
            auxbasis: None,
            omega: None,
            mesh: None,
            ke_cutoff: None,
            exclude_dd_block: false,
        }
    }

    /// The `(omega, mesh, ke_cutoff)` this builder WOULD run at.
    ///
    /// Shippable and gated (`tests/rsdf_builder.rs`) even though the builder
    /// itself is not: it is a pure function of the cell and the k-points.
    ///
    /// # Errors
    /// Propagates [`guess_omega`].
    pub fn guess(&self) -> Result<(f64, [usize; 3], f64), PbcDfError> {
        guess_omega(&self.cell, &self.kpts, self.mesh)
    }

    /// `_RSGDFBuilder.build()` — **refused**; see the module docs.
    ///
    /// # Errors
    /// Always [`PyscfRsError::NotYetImplemented`], naming [`RS_BUILDER_GAP`].
    pub fn build(&mut self) -> Result<(), PbcDfError> {
        Err(PbcDfError::Core(
            pyscf_core::PyscfRsError::NotYetImplemented {
                phase: 14,
                what: RS_BUILDER_GAP,
            },
        ))
    }
}
