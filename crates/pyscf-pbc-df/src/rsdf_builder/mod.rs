//! `rsdf_builder` — range-separated Gaussian density fitting
//! (`pyscf/pbc/df/rsdf_builder.py`), plan 14-07.
//!
//! # STATUS: sub-task 7a ships. 7b, 7c and 7d are BLOCKED, and the blocker is
//! # named, measured and outside this repository.
//!
//! Plan 14-07 Task 7b says, in its own words:
//!
//! > **Check first that the cintx resolver exposes a short-range `int3c2e`**;
//! > if it does not, this is the plan's one real blocker and it must be
//! > reported as such, not worked around with a numerically different kernel.
//!
//! **It does not.** The evidence, gathered before any code was written:
//!
//! 1. `cintx_runtime::ExecutionOptions` (`cintx/crates/cintx-runtime/src/options.rs:96`)
//!    carries `f12_zeta` (libcint `env[9]`), `rinv_orig` (`env[4..6]`) and
//!    `common_orig` (`env[1..3]`). There is **no `range_omega`** field —
//!    libcint's `PTR_RANGE_OMEGA` is `env[8]`, and the safe API has no setter
//!    for it.
//! 2. `cintx-compat/src/raw.rs:35-41` names `PTR_RANGE_OMEGA = 8` only in a
//!    warning not to overwrite the slot. No kernel reads it: neither
//!    `cintx-cubecl/src/kernels/center_3c2e.rs` nor `two_electron.rs` mentions
//!    `omega` at all.
//! 3. `cintx-ops`'s resolver knows `int3c2e` and `int3c2e_ip1` and no
//!    range-separated variant, and upstream PySCF confirms none should exist:
//!    range separation is an `env[8]` toggle around the STANDARD symbol, never
//!    a distinct `int2e_sr_*` name.
//! 4. **This gap is already on this repository's record.**
//!    `crates/pyscf-gto/src/range_coulomb.rs` documents it as Phase 4's Open
//!    Question A5 / cintx#11: "cintx *reads* `env[8]`, but its safe API […]
//!    exposes only `f12_zeta` (env[9]) […] there is **no** `range_omega`
//!    (env[8]) setter on the safe path". Phase 4 shipped the set/restore
//!    semantics and CI-gated the numerical RSH assertion behind the same gap.
//! 5. There is a second, independent obstruction on this particular path:
//!    [`crate::incore::aux_e2`] reaches cintx through
//!    `pyscf_gto::build_image_expanded_with_aux`, which builds its `BasisSet`
//!    from `cell.mol._atom` / `_basis` — the per-element parsed basis — and not
//!    from a `_env` array. So even the `pyscf-gto` workaround of writing
//!    `mol._env[8]` directly (which `range_coulomb.rs` uses for the molecular
//!    path) is not reachable from the periodic 3-centre driver.
//!
//! **The work needed to lift this is planned**, in
//! `.planning/carryovers/D-PBC-24-cintx-range-omega-PLAN.md`: five stages, of
//! which stage 2 is enough to unblock everything this phase lost. The finding
//! that sizes it is that `rys_order = (sum l_ceil)/2 + 1` is `<= 3` on every
//! system this milestone gates, and libcint computes the short-range integral
//! in that regime as `full - LR` with DOUBLED Rys roots
//! (`libcint/src/g2e.c:4477-4491`) using only the STANDARD root finder — so
//! `CINTsr_rys_roots`, the genuinely hard part, is a later stage rather than a
//! prerequisite.
//!
//! `_RSGDFBuilder`'s `get_2c2e` needs a short-range `int2c2e` and its
//! `outcore_auxe2` a short-range `int3c2e`. Both are the same missing
//! capability. Substituting the full-range kernel would produce a builder that
//! looks like RSDF, runs, converges, and is silently a different method — the
//! one outcome the plan explicitly forbids.
//!
//! # What DOES ship, and why it is worth shipping alone
//!
//! [`omega`] — all twelve estimators, `weighted_coulG_LR` / `_SR`, and
//! `_gaussian_int`. They are pure functions of the cell, they are gated
//! against `measurements/omega.out`, and three separate downstream consumers
//! need them regardless of the blocker: `rsjk` (14-08), RSH functionals
//! (`JkOpts::omega`, already threaded through `get_coulG`), and Phase 17.
//! Plan 14-07 sequenced 7a first precisely so that it could land on its own:
//! "Do 7a completely, with its tests green, before writing a line of 7b — the
//! whole scheme's accuracy is one `ω` away."
//!
//! # What a caller gets instead
//!
//! [`RsGdfBuilder::build`] returns `NotYetImplemented { phase: 14 }` naming the
//! cintx gap (D-PBC-20: a deferred branch never returns a silently wrong
//! answer). `Gdf::prefer_ccdf` therefore stays `true` — plan 14-07's Task 7d
//! flip cannot happen — and Gate 3 (`|E(GDF) − E(RSDF)|` against upstream's
//! 1.353e-08 floor) is unreachable this phase. Both are recorded in
//! `14-VERIFICATION.md`.

pub mod omega;

pub use omega::{
    OMEGA_MIN, RCUT_THRESHOLD, estimate_ft_rcut, estimate_ke_cutoff_for_omega, estimate_meshz,
    estimate_omega_for_ke_cutoff, estimate_omega_min, estimate_rcut, estimate_rs_2c2e_rcut,
    gaussian_int, guess_omega, round_off_to_odd_mesh, weighted_coulg_lr, weighted_coulg_sr,
};

use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;

/// The one-line reason every range-separated 3-centre path is refused. Kept as
/// a constant so the tests can assert on it and so the message cannot drift
/// between call sites.
pub const CINTX_SR_GAP: &str = "range-separated int3c2e/int2c2e — cintx's safe API has no \
     range_omega (libcint env[8]) knob: ExecutionOptions carries f12_zeta \
     (env[9]), rinv_orig and common_orig only, and no kernel reads env[8]. \
     Same gap as pyscf-gto's range_coulomb.rs Open Question A5 / cintx#11. \
     Plan 14-07 requires this to be reported, NOT worked around with a \
     numerically different kernel";

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
