//! `rsdf_builder` — range-separated Gaussian density fitting
//! (`pyscf/pbc/df/rsdf_builder.py`), plan 14-07.
//!
//! # STATUS: sub-tasks 7a, 7b and 7c ship. 7d (the default flip) does not.
//!
//! ## What D-PBC-24 unblocked, and what this module then had to write
//!
//! Plan 14-07 Task 7b said, in its own words:
//!
//! > **Check first that the cintx resolver exposes a short-range `int3c2e`**;
//! > if it does not, this is the plan's one real blocker and it must be
//! > reported as such, not worked around with a numerically different kernel.
//!
//! It did not, and Phase 14 closed with Gate 3 unreachable for that reason.
//! **D-PBC-24 supplied the capability** — `ExecutionOptions::range_omega`
//! (libcint `env[8]`), part of the WORKSPACE query because short range doubles
//! the Rys roots — and ω now rides in the OPTIONS rather than the basis, which
//! is why [`crate::incore::aux_e2`] reaches it at all without an `_env`.
//!
//! On top of that, this module ships `_RSGDFBuilder`: [`RsGdfBuilder::build`]
//! picks `(omega, mesh, ke_cutoff)`, [`j2c::get_2c2e`] builds the metric,
//! [`j2c::weighted_ft_ao`] the long-range plane-wave half, and
//! [`RsGdfBuilder::make_j3c`] drives
//! [`crate::gdf_builder::j3c::Scheme::RangeSeparated`] — one pipeline, three
//! schemes, as `_CCMDFBuilder` is a subclass rather than a copy upstream.
//!
//! ## Measured, against upstream's own RSDF
//!
//! He-fcc `sto-3g` 2x2x2, `conv_tol = 1e-12`, `KRHF`:
//!
//! | route | upstream | this port | error |
//! |---|---|---|---|
//! | RSDF (upstream's DEFAULT) | -2.80842508717097 | -2.80842508693849 | **2.32e-10** |
//! | GDF, compensated charge | -2.80842508664874 | -2.80842508692377 | **2.75e-10** |
//!
//! Both land at the same order, so range separation added nothing to the
//! port's residual. The port's own `|CC - RS|` is **1.47e-11** against
//! upstream's **5.222e-10** — the port's two routes agree with each other more
//! closely than upstream's two do, because upstream's two differ partly through
//! the `exclude_d_aux` / `exclude_dd_block` splits that this port has in
//! NEITHER route. Gate 3's "within a factor of 2 of upstream's gap" criterion
//! is therefore the wrong shape for this port; agreement with upstream's RSDF
//! itself is the meaningful statement and is what the tests assert.
//!
//! ## The one deliberate divergence: `_guess_omega` takes the ORBITAL cell
//!
//! Upstream passes the AUXCELL (`rsdf_builder.py:145`). This port passes the
//! cell, giving a finer `(omega, mesh)` — `[11,11,11]` where upstream uses
//! `[7,7,7]` on He-fcc 2x2x2. It is the price of having no
//! `_RangeSeparatedCell`: upstream routes what a coarse grid cannot resolve
//! around the grid, and this port cannot, so it resolves it. At upstream's own
//! mesh the error is **8.67e-7**; at this one, **1.97e-10**. See
//! [`RsGdfBuilder::build`] and [`j2c`]'s module docs.
//!
//! ## What is still NOT ported
//!
//! * `_RSNucBuilder` (`rsdf_builder.py:1098-1311`) — sub-task 7c's other half,
//!   and **a performance carry-over rather than a fidelity gap**. This port
//!   uses NEITHER split nuclear builder: [`crate::gdf::nuc::get_nuc`] goes
//!   straight to AFTDF at the cell's converged mesh, which is oracle-gated at
//!   2.755e-12 and strictly more accurate than either split. What the split
//!   buys is speed (a `[9,9,9]` mesh instead of `[43,43,43]`), and 14-04
//!   measured that evaluating the WHOLE nuclear attraction on the small mesh is
//!   worth 0.0743 Ha — so the split is not optional, it is the only way to use
//!   that mesh at all.
//! * `rsdf_helper.py`'s prescreening (`get_q_cond`, the Schwarz bound). Its
//!   absence keeps MORE primitives than upstream — conservative, and the same
//!   posture 14-05 took toward `ExtendedMole.strip_basis`.
//! * Task 7d's flip of [`crate::Gdf::prefer_ccdf`] to `false`. It moves a
//!   committed reference energy and must be its own cited edit.
//! * `pyscf_pbc_scf::rsjk` (14-08 Task 4) — unblocked by D-PBC-24 for `s`/`p`
//!   bases, still unwritten. [`RS_BUILDER_GAP`] is what it still names.

pub mod j2c;
pub mod omega;

pub use omega::{
    OMEGA_MIN, RCUT_THRESHOLD, estimate_ft_rcut, estimate_ke_cutoff_for_omega, estimate_meshz,
    estimate_omega_for_ke_cutoff, estimate_omega_min, estimate_rcut, estimate_rs_2c2e_rcut,
    gaussian_int, guess_omega, round_off_to_odd_mesh, weighted_coulg_lr, weighted_coulg_sr,
};

use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;

/// The one-line reason range-separated EXCHANGE is still refused.
/// Kept as a constant so the tests can assert on it and so the message cannot
/// drift between call sites.
///
/// It has changed meaning twice: it named a missing cintx capability, then
/// this port's unported `_RSGDFBuilder`, and now — both of those being done —
/// only `pyscf_pbc_scf::rsjk`. Read the module docs before assuming an older
/// reason still applies.
pub const RS_BUILDER_GAP: &str = "range-separated EXCHANGE (rsjk) — the cintx side is DONE \
     (ExecutionOptions::range_omega, libcint env[8]; incore::aux_e2 and \
     incore::fill_2c2e both take an omega, gated by SR + LR == full in \
     tests/incore.rs) and rsdf_builder::_RSGDFBuilder is ported on top of it \
     (plan 14-07 7b/7c), but pyscf_pbc_scf::rsjk itself is not: plan 14-08 \
     Task 4. Finish it rather than substituting the full-range kernel, which \
     runs, converges, and is a different method — and rsjk is EXACT, so a \
     wrong answer there lands inside GDF's fitting error and looks plausible";

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
    /// **D-PBC-23, plan 17-10 Task 3.** `false` is this port's OWN default
    /// (deliberately not upstream's `true` — see
    /// `crate::gdf_builder`'s module docs for why). `true` re-routes the
    /// smooth-smooth block through
    /// [`crate::gdf_builder::dd_block::fft_dd_block`], as for
    /// [`crate::gdf_builder::CcGdfBuilder`], and is a fully-working, gated
    /// opt-in.
    pub exclude_dd_block: bool,
    /// The short-range 3-centre image radius — upstream's `Int3cBuilder.rcut`.
    /// `None` uses [`omega::estimate_rcut`].
    pub rcut: Option<f64>,
    /// Drive `_RSMDFBuilder` (`mdf.py:238-353`) instead of `_RSGDFBuilder`.
    ///
    /// Upstream makes this a subclass overriding three methods; this port makes
    /// it a flag on one builder, for the same reason 14-02 made `Scheme` a tag
    /// rather than two copies of `make_j3c`. See
    /// [`crate::gdf_builder::j3c::Scheme::RangeSeparated`] for what it changes.
    pub mixed: bool,
    /// The decontracted cell — built only when [`Self::exclude_dd_block`] is
    /// set. `None` until [`Self::build`].
    pub rs_cell: Option<crate::ft_ao::rs_cell::RsCell>,
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
            rcut: None,
            mixed: false,
            rs_cell: None,
        }
    }

    /// `_RSMDFBuilder` on `cell` at `kpts` — [`RsGdfBuilder::new`] with
    /// [`RsGdfBuilder::mixed`] set.
    pub fn new_mixed(cell: Cell, kpts: &[[f64; 3]]) -> Self {
        Self {
            mixed: true,
            ..Self::new(cell, kpts)
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

    /// `_RSGDFBuilder.build(omega)` — `rsdf_builder.py:127-193`, minus the
    /// `_RangeSeparatedCell` / `ExtendedMole` supermole (D-PBC-21 / D-PBC-23;
    /// see [`j2c`]'s module docs for what treating every function as compact
    /// costs and why the direction is safe).
    ///
    /// Picks `omega`, the long-range mesh and the kinetic-energy cutoff. What
    /// upstream does after that — building `rs_cell`, `rs_auxcell`, `supmol`
    /// and `supmol_ft` — is the compact/smooth partition and the stripped
    /// Born–von-Kármán supercell, neither of which this port has; the SR radius
    /// they exist to tighten is instead taken whole from
    /// [`omega::estimate_rcut`].
    ///
    /// # Errors
    /// Propagates [`guess_omega`] and the auxiliary-cell build.
    pub fn build(&mut self) -> Result<(), PbcDfError> {
        // `rsdf_builder.py:137-152`. An omega set by the caller keeps its mesh
        // from `estimate_ke_cutoff_for_omega`; an unset one lets `_guess_omega`
        // balance the real-space and reciprocal-space halves against each other.
        match self.omega {
            None | Some(0.0) => {
                // **The ORBITAL cell, where upstream passes the AUXCELL**
                // (`rsdf_builder.py:145`), and this is a deliberate, measured
                // divergence — the price of not having `_RangeSeparatedCell`.
                //
                // Upstream can afford the auxcell's coarser answer because
                // `exclude_d_aux` and `exclude_dd_block` route the pieces a
                // coarse grid cannot resolve AROUND the grid: the smooth
                // auxiliary functions get the full kernel through AFT and the
                // smooth-smooth orbital block goes to an FFT. This port has
                // neither split (see [`j2c`]'s module docs), so every function
                // rides the same plane-wave grid and the grid has to be good
                // enough for the compact ones too.
                //
                // Measured on He-fcc `sto-3g` 2x2x2 against upstream's own
                // RSDF energy (**-2.80842508717097**):
                //
                // | mesh | source | this port's error |
                // |---|---|---|
                // | [7,7,7] | `_guess_omega(auxcell)` — upstream's | **8.67e-7** |
                // | [11,11,11] | `_guess_omega(cell)` — this line | **1.97e-10** |
                //
                // 1.97e-10 is the same order as this port's compensated route
                // manages against upstream's GDF (2.75e-10), i.e. the residual
                // is the port's ordinary fitting accuracy rather than anything
                // range separation added. `[11,11,11]` is also exactly what
                // `measurements/omega.out` records and what
                // `tests/rsdf_builder.rs::guess_omega_matches_upstream` pins.
                //
                // The `(omega, mesh)` pair moves TOGETHER — a larger omega puts
                // more of the kernel in real space and demands a finer grid for
                // what is left — so taking one from the orbital cell and the
                // other from the auxcell would be worse than either.
                //
                // This applies to `_RSMDFBuilder` too, and the evidence is the
                // same shape. At MATCHED meshes the port reproduces upstream's
                // RSMDF exactly as it should — He-fcc 2x2x2, `KRHF`:
                //
                // | mesh | error vs upstream at the same mesh |
                // |---|---|
                // | [7,7,7] (upstream's default, from the auxcell) | 1.160e-6 |
                // | [11,11,11] (this port's, from the cell) | **3.049e-10** |
                // | [15,15,15] | 1.900e-11 |
                // | [21,21,21] | 7.809e-12 |
                //
                // — so the algebra is right and the coarse-grid gap is the
                // missing splits, not a defect. **For MDF the mesh is
                // definitional**, not merely a convergence knob: its plane-wave
                // set is part of the BASIS (`<g|g> - <g|G><G|g>`, with `aft_jk`
                // adding the residual back over the same `{G}`), so two meshes
                // are two different — equally valid — MDF approximations, and
                // an MDF number is only comparable against another at the SAME
                // mesh. `tests/gate3_rsdf.rs` therefore gates RSMDF at matched
                // meshes and records the default-mesh gap rather than gating it.
                let (omega, mesh, ke) = guess_omega(&self.cell, &self.kpts, self.mesh)?;
                self.omega = Some(omega);
                self.mesh = Some(mesh);
                self.ke_cutoff = Some(ke);
            }
            // `rsdf_builder.py:147-152` — an omega the caller set keeps its own
            // mesh; only the missing half is derived.
            Some(omega) => match (self.mesh, self.ke_cutoff) {
                (None, _) => {
                    let ke = estimate_ke_cutoff_for_omega(&self.cell, omega, None);
                    self.mesh = Some(self.cell.cutoff_to_mesh(ke)?);
                    self.ke_cutoff = Some(ke);
                }
                (Some(mesh), None) => {
                    // `min(mesh_to_cutoff(a, mesh)[:cell.dimension])`.
                    let ke = pyscf_pbc_tools::mesh::mesh_to_cutoff(&self.cell.a, mesh)?
                        .into_iter()
                        .take(self.cell.dimension as usize)
                        .fold(f64::INFINITY, f64::min);
                    self.ke_cutoff = Some(ke);
                }
                (Some(_), Some(_)) => {}
            },
        }
        if self.exclude_dd_block {
            let ke_cutoff = self.ke_cutoff.expect("just set above");
            self.rs_cell = Some(crate::ft_ao::rs_cell::RsCell::from_cell(
                &self.cell,
                Some(ke_cutoff),
                Some(RCUT_THRESHOLD),
                false,
            )?);
        }
        Ok(())
    }

    /// `make_j3c(...)` on this builder's state — the range-separated arm of
    /// [`crate::gdf_builder::j3c::make_j3c_scheme`].
    ///
    /// # Errors
    /// [`PbcDfError::Core`] when [`Self::build`] has not run, and propagates
    /// every stage of the 3-centre build.
    pub fn make_j3c(
        &self,
        aosym: crate::incore::Aosym,
        j_only: bool,
    ) -> Result<crate::gdf_builder::j3c::Cderi, PbcDfError> {
        let (Some(omega), Some(mesh)) = (self.omega, self.mesh) else {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(
                    "RsGdfBuilder::make_j3c: call build() first".into(),
                ),
            )));
        };
        // The unfused auxiliary cell — range separation needs no compensating
        // charge (see [`crate::gdf_builder::fuse::unfused_auxcell`]).
        let fused =
            crate::gdf_builder::fuse::unfused_auxcell(&self.cell, self.auxbasis.as_deref())?;
        // `estimate_rcut(rs_cell, rs_auxcell, omega, ...)` with no compact/
        // smooth split, so the radius must cover the most diffuse auxiliary
        // function. `.max()` is upstream's `rcut_sr.max()` (`:174`).
        let rcut = self.rcut.unwrap_or_else(|| {
            omega::estimate_rcut(&self.cell, &fused.auxcell.cell, omega, None)
                .into_iter()
                .fold(0.0_f64, f64::max)
        });
        crate::gdf_builder::j3c::make_j3c_scheme_dd(
            &self.cell,
            &fused,
            &self.kpts,
            aosym,
            mesh,
            j_only,
            // Cholesky first for RSGDF; RSMDF forces the eigen route
            // (`mdf.py:245-250`, upstream's own words: "large difference may be
            // found in results between the CD/ED treatments ... Abandon CD
            // treatment for better numerical stability"), because subtracting
            // a plane-wave projection can push the metric indefinite and
            // Cholesky on an indefinite matrix does not merely lose accuracy —
            // it fails or returns nonsense.
            self.mixed,
            Some(rcut),
            crate::gdf_builder::j3c::Scheme::RangeSeparated {
                omega,
                mixed: self.mixed,
            },
            self.rs_cell.as_ref(),
        )
    }
}

/// The mesh [`j2c::get_2c2e`] evaluates its plane-wave part on —
/// `rsdf_builder.py:288-297`.
///
/// Tighter than the builder's own `mesh`, at `precision^1.5`, because the
/// metric is more sensitive than the tensor. The compensated route makes the
/// same distinction with `precision^2` (`gdf_builder.py:150-158`).
///
/// # Errors
/// Propagates `cutoff_to_mesh`.
pub fn j2c_mesh(cell: &Cell, auxcell: &Cell, omega: f64) -> Result<[usize; 3], PbcDfError> {
    let precision = auxcell.precision.powf(1.5);
    let ke = estimate_ke_cutoff_for_omega(auxcell, omega, Some(precision));
    let mesh = cell.cutoff_to_mesh(ke)?;
    Ok(mesh)
}
