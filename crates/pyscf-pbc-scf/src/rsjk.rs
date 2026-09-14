//! `rsjk` — range-separated J/K with NO density fitting
//! (`pyscf/pbc/scf/rsjk.py`), plan 14-08 Task 4.
//!
//! # STATUS: NOT PORTED. Re-assessed by plan 20-06 (2026-09-13).
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
//! ## What is no longer the blocker
//!
//! * **The integral.** `rsjk.py:186` sets `supmol_sr.omega = -self.omega`
//!   around the STANDARD `int2e`. cintx's safe API honours that as
//!   `ExecutionOptions::range_omega` (libcint `env[8]`, D-PBC-24) on its SCALAR
//!   route, including the lower-bounded `sr_rys_roots_host` for `rys_order > 3`.
//!   (Its quartet-BATCH route refuses `range_omega` outright —
//!   `cintx-rs/src/api.rs:3405` — so a port would pay scalar per-quartet cost.)
//! * **The supermole types.** `ft_ao._RangeSeparatedCell` →
//!   [`pyscf_pbc_df::ft_ao::rs_cell::RsCell`] (`in_rsjk` honoured) and
//!   `ft_ao.ExtendedMole` + `strip_basis` →
//!   [`pyscf_pbc_df::ft_ao::ExtendedMole`] (plan 17-10).
//!
//! ## What IS the blocker — `rsjk.py`'s own body, none of which exists
//!
//! Plan 20-06 Task 1 listed every construct `build` (`rsjk.py:136-238`) and
//! `_get_jk_sr` (`:267-435`) consume; the full table is in
//! `.planning/phases/20-pbc-python-bindings/20-06-SUMMARY.md`. Absent:
//!
//! 1. `rsjk.estimate_rcut` (`rsjk.py:1182-1261`) — the 4-centre SR radius
//!    `strip_basis` is called with (NOT `rsdf_builder::estimate_rcut`, which is
//!    the 3-centre one).
//! 2. A libcint-shaped supermole. [`pyscf_pbc_df::ft_ao::ExtendedMole`] is
//!    compact (`bas_mask`/`seg_loc`/`seg2sh` + image lists); it never
//!    materialises the `_atm/_bas/_env` (or cintx `BasisSet`) of the translated
//!    shells that `ft_ao.py:614-628` builds and every C driver below indexes.
//! 3. The Schwarz/overlap prescreen: `PBCVHFnr_int2e_q_cond`
//!    (`pyscf/lib/pbc/nr_direct.c:1037`), `PBCVHFnr_sindex` (`:1120`), the
//!    `qindex` assembly (`rsjk.py:212-235`), `_qcond_cell0_abstract`
//!    (`rsjk.py:1336`) and `_sort_qcond_cell0` (`:255`).
//! 4. The screened periodic 4-centre driver: `PBCVHF_direct_drv` /
//!    `_nodddd` (`nr_direct.c:721`, `:855`), the six
//!    `PBCVHF_contract_{j,k,jk}_{s1,s2kl}` kernels (`:38-530`), the
//!    `approx_bvk_rcond0` / `PBCapprox_bvk_rcond` distance screens
//!    (`:531`, `:591`) and `PBCint2e_sph` (`pyscf/lib/pbc/cint2e.c:330`).
//!    Here the screening IS the algorithm: an unscreened sweep over the BvK
//!    images is infeasible, so there is no correct-but-slow fallback.
//! 5. The BvK density plumbing: `k2gamma.kpts_to_kmesh` /
//!    `double_translation_indices` (`k2gamma.py:39`, `:104`), the `sc_dm`
//!    transform and the `NP_absmax` `dmindex` (`rsjk.py:330-374`).
//! 6. The long-range half as `rsjk` composes it: `coulG − coulG_SR` with the
//!    analytic `π/ω²` G=0 term (`rsjk.py:596-612`), `_ExtendedMoleFT`
//!    (`rsdf_builder.py:1312`), the `dm_factor` exchange path
//!    (`rsjk.py:817-1170`) and `_purify` (`:1172`).
//!    [`pyscf_pbc_df::aft_jk::get_j_kpts`] / `get_k_kpts` take an `omega` but
//!    over the full image list with a single-kernel `coulG` — a different
//!    computation.
//!
//! [`RangeSeparatedJkBuilder::build`] therefore still refuses (D-PBC-20), and
//! **must not** be finished by substituting the full-range kernel: because
//! `rsjk` is EXACT, a wrong answer would land within the DF fitting error of a
//! correct GDF and look entirely plausible.
//!
//! # What ships anyway: the ω `rsjk.build` runs at
//!
//! **Corrected by plan 20-06.** This type used to return
//! `rsdf_builder._guess_omega`'s ω, on the belief that `rsjk.py` imports it.
//! It does not: `rsjk.py` defines its OWN module-level `_guess_omega`
//! (`:1263`), `estimate_ke_cutoff_for_omega` (`:1293`) and
//! `estimate_omega_for_ke_cutoff` (`:1306`), with different formulas, and
//! `build` resolves those names. Upstream 2.12.1 on He-fcc `sto-3g` 2×2×2 runs
//! at ω = 1.312754030266949 (mesh 15), not RSDF's 0.7393586378665364 (mesh 11).
//! [`guess_omega`], [`estimate_ke_cutoff_for_omega`] and
//! [`estimate_omega_for_ke_cutoff`] below are ports of the `rsjk.py` versions.
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
use pyscf_pbc_df::rsdf_builder::OMEGA_MIN;
use pyscf_pbc_df::traits::{JkOpts, JkResult};
use pyscf_pbc_gto::{Cell, LowDimFtType};

/// The one-line reason `rsjk` is refused, re-exported from
/// [`pyscf_pbc_df::rsdf_builder::RS_BUILDER_GAP`].
///
/// That constant used to cover BOTH this and `_RSGDFBuilder`. Plan 14-07
/// sub-tasks 7b/7c shipped the latter (Phase 14 Gate 3 is MET), so the text now
/// names only this; plan 20-06 rewrote it to name `rsjk`'s actual missing body.
pub const RS_BUILDER_GAP: &str = pyscf_pbc_df::rsdf_builder::RS_BUILDER_GAP;

/// Every primitive exponent of every shell — upstream's
/// `np.hstack(cell.bas_exps())`.
fn all_exps(cell: &Cell) -> impl Iterator<Item = f64> + '_ {
    (0..cell.mol.nbas).flat_map(move |i| pyscf_pbc_gto::cutoff::bas_exp(cell, i))
}

/// `rsjk.estimate_ke_cutoff_for_omega(cell, omega, precision)` —
/// `rsjk.py:1293-1304`.
///
/// **Not** [`pyscf_pbc_df::rsdf_builder::estimate_ke_cutoff_for_omega`]
/// (`rsdf_builder.py:1595`), which runs `aft._estimate_ke_cutoff` per shell.
/// This one uses only the steepest exponent and a fixed two-step iteration.
pub fn estimate_ke_cutoff_for_omega(cell: &Cell, omega: f64, precision: Option<f64>) -> f64 {
    let precision = precision.unwrap_or(cell.precision);
    let ai = all_exps(cell).fold(f64::NEG_INFINITY, f64::max);
    let theta = 1.0 / (1.0 / ai + omega.powf(-2.0));
    let fac = 32.0 * std::f64::consts::PI.powi(2) * theta / precision;
    let mut ecut = 20.0_f64;
    ecut = (fac / (2.0 * ecut) + 1.0).ln() * 2.0 * theta;
    ecut = (fac / (2.0 * ecut) + 1.0).ln() * 2.0 * theta;
    ecut
}

/// `rsjk.estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision)` —
/// `rsjk.py:1306-1334`, including the clamp to `OMEGA_MIN` (upstream logs a
/// warning there; this port clamps silently).
///
/// **Not** [`pyscf_pbc_df::rsdf_builder::estimate_omega_for_ke_cutoff`]
/// (`rsdf_builder.py:1606`): `rsjk.py` keeps that formula only as a comment.
pub fn estimate_omega_for_ke_cutoff(cell: &Cell, ke_cutoff: f64, precision: Option<f64>) -> f64 {
    let precision = precision.unwrap_or(cell.precision);
    let ai = all_exps(cell).fold(f64::NEG_INFINITY, f64::max);
    let aij = ai * 2.0;
    let fac = 32.0 * std::f64::consts::PI.powi(2) / precision;
    let omega = 0.3_f64;
    let mut theta = 1.0 / (1.0 / ai + omega.powf(-2.0));
    let mut omega2 =
        1.0 / ((fac * theta / (2.0 * ke_cutoff) + 1.0).ln() * 2.0 / ke_cutoff - 1.0 / aij);
    if omega2 > 0.0 {
        theta = 1.0 / (1.0 / ai + 1.0 / omega2);
        omega2 = 1.0 / ((fac * theta / (2.0 * ke_cutoff) + 1.0).ln() * 2.0 / ke_cutoff - 1.0 / aij);
    }
    let omega = omega2.max(0.0).sqrt();
    if omega < OMEGA_MIN { OMEGA_MIN } else { omega }
}

/// `rsjk._guess_omega(cell, kpts, mesh)` — `rsjk.py:1263-1291`.
///
/// **Not** [`pyscf_pbc_df::rsdf_builder::guess_omega`] (`rsdf_builder.py:1330`):
/// the default `ke_cutoff` here is `50 / (.7 + .25 nk + .05 nk³)` rather than
/// `20 nk⁻¹`, and ω comes from [`estimate_omega_for_ke_cutoff`] above.
///
/// # Errors
/// Propagates `cutoff_to_mesh` / `mesh_to_cutoff`.
pub fn guess_omega(
    cell: &Cell,
    kpts: &[[f64; 3]],
    mesh: Option<[usize; 3]>,
) -> Result<(f64, [usize; 3], f64), PbcDfError> {
    let a = cell.lattice_vectors();
    if cell.dimension == 0 {
        let m = match mesh {
            Some(m) => m,
            None => cell.try_mesh()?,
        };
        let ke = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, m)?
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        return Ok((0.0, m, ke));
    }

    let mesh = match mesh {
        Some(m) => m,
        None => {
            let nkpts = kpts.len().max(1) as f64;
            let ke_min = estimate_ke_cutoff_for_omega(cell, OMEGA_MIN, None);
            let nk = (cell.mol.nao_nr as f64 / 25.0 * nkpts).powf(1.0 / 3.0);
            let mut ke_cutoff = 50.0 / (0.7 + 0.25 * nk + 0.05 * nk.powf(3.0));
            ke_cutoff = ke_cutoff.max(ke_min);
            // `exps = [e for l, e in zip(ls, bas_exps()) if l != 0]`
            let mut exp_min = f64::INFINITY;
            for i in 0..cell.mol.nbas {
                if pyscf_pbc_gto::cutoff::bas_angular(cell, i) != 0 {
                    for e in pyscf_pbc_gto::cutoff::bas_exp(cell, i) {
                        exp_min = exp_min.min(e);
                    }
                }
            }
            if exp_min.is_finite() {
                let omega_max = exp_min.sqrt() * 2.0;
                let ke_max = estimate_ke_cutoff_for_omega(cell, omega_max, None);
                ke_cutoff = ke_cutoff.min(ke_max);
            }
            cell.cutoff_to_mesh(ke_cutoff)?
        }
    };
    let ke_cutoff = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, mesh)?
        .into_iter()
        .take(cell.dimension as usize)
        .fold(f64::INFINITY, f64::min);
    let omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff, None);
    Ok((omega, mesh, ke_cutoff))
}

/// `RangeSeparatedJKBuilder` — `rsjk.py:52-…`.
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
    /// `rsjk.py:142-156`.
    ///
    /// With no (or a zero) `omega`, [`guess_omega`] over `self.mesh`. With an
    /// explicit `omega`, upstream DISCARDS any preset mesh and takes
    /// `cutoff_to_mesh(estimate_ke_cutoff_for_omega(cell, omega))`
    /// (`:148-149`); this does the same. A 2-D cell with a non-`inf_vacuum`
    /// `low_dim_ft_type` then has `mesh[2]` replaced by `_estimate_meshz`
    /// (`:151-153`).
    ///
    /// # Errors
    /// Propagates [`guess_omega`], `cutoff_to_mesh` and `estimate_meshz`.
    pub fn guess_omega(&self) -> Result<(f64, [usize; 3], f64), PbcDfError> {
        let (omega, mut mesh, ke) = match self.omega {
            Some(w) if w != 0.0 => {
                let ke = estimate_ke_cutoff_for_omega(&self.cell, w, None);
                (w, self.cell.cutoff_to_mesh(ke)?, ke)
            }
            _ => guess_omega(&self.cell, &self.kpts, self.mesh)?,
        };
        if self.cell.dimension == 2 && self.cell.low_dim_ft_type != LowDimFtType::InfVacuum {
            mesh[2] = pyscf_pbc_df::rsdf_builder::estimate_meshz(&self.cell, None)?;
        }
        Ok((omega, mesh, ke))
    }

    /// `build(omega, intor='int2e')` — **refused**; see the module docs.
    ///
    /// # Errors
    /// Always [`PyscfRsError::NotYetImplemented`], naming what is missing.
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
