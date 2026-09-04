//! `MultiGridNumInt` — plan 17-11 Task 4.
//!
//! A selectable alternative to `crate::numint::KNumInt` for `get_nuc`,
//! `get_pp`, `get_j`, `get_veff`/`nr_rks`/`nr_uks`, built on the collocation
//! engine in `crate::multigrid::colloc`.
//!
//! # Scope: GAMMA POINT ONLY
//!
//! Upstream's `multigrid.py` is k-point general (Bloch-phase-weighted
//! collocation, complex density matrices). This port implements the
//! **gamma-point** case only — real, single-k density matrices and
//! matrices — which is what every Gate E number in
//! `.planning/phases/17-ksymm-multigrid/measurements/gate_multigrid.py`
//! actually measures (`MultiGridNumInt(cell)` with no `kpts`, and the
//! converged `KRKS` runs are gamma-only). k-point-resolved multigrid
//! (Bloch-phase collocation, matching `crate::numint::KNumInt`'s k-point
//! generality) is NOT ported; a caller asking for `kpts.len() > 1` gets
//! [`crate::error::PbcDftError`]. Stated here rather than left implicit, per
//! the plan's "judgment call, stated out loud" convention.
//!
//! # `get_nuc` / `get_pp`: delegated to FFTDF, not re-derived
//!
//! Upstream's OWN `multigrid.py::get_nuc`/`get_pp` (`:365-515`) reuse the
//! SAME analytic machinery `pbc.df.fft.FFTDF` does for every term except the
//! local-part "pass2" (G-space potential -> AO matrix): `get_gth_vlocG_part1`,
//! `pp_int.get_pp_loc_part2` and `ft_ao`-based `vppnl` are ALL called
//! unchanged from `pyscf.pbc.gto.pseudo.pp_int`, identical to what
//! `crates/pyscf-pbc-df/src/fftdf.rs::{get_nuc, get_pp}` already compute.
//! 17-01's own measurement (`measurements/gate_multigrid.out`) found the
//! *only* observable difference between multigrid's pass2 and FFTDF's own
//! `eval_mat`-based pass2 to be 1e-12..1e-13 (`get_pp v1 vs FFTDF`), i.e.
//! floating-point noise, not a physical effect. Re-deriving that pass2 here
//! (which the plan's `colloc` module could do — see [`get_j`]) would buy
//! nothing this Gate needs and would duplicate an already-shipped,
//! already-tested code path for a result the measurement shows is
//! indistinguishable. `get_nuc`/`get_pp` therefore delegate to
//! `pyscf_pbc_df::fftdf::{get_nuc, get_pp}` directly. `get_j`/`get_veff` do
//! NOT delegate — they are the actual point of this plan, and go through
//! [`crate::multigrid::colloc`] end to end.

use std::sync::{Arc, Mutex};

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_pbc_df::fftdf::Fftdf;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::{get_coulg_at_gv, get_gv};

use crate::error::PbcDftError;
use crate::multigrid::colloc::{self, LevelValues};
use crate::multigrid::tasks::{self, Decontracted, GridLevel};
use crate::multigrid::utils::cell_fingerprint;
use crate::xc::{RhoEff, VxcEff, XcType, eval_xc_eff_rks, eval_xc_eff_uks};

const GAMMA: [f64; 3] = [0.0, 0.0, 0.0];

/// The cell-independent geometry one multigrid density evaluation needs:
/// the decontracted primitive basis and the grid-level task list.
///
/// M-02. Both are pure functions of the cell, and both were rebuilt on every
/// call — i.e. on every SCF cycle — before this cache existed.
type V1Tasks = (Decontracted, Vec<GridLevel>);

/// `pbc.dft.multigrid.MultiGridNumInt` — the v1 multigrid driver (gamma
/// point; see the module doc).
///
/// # M-02 — this used to be a unit struct
///
/// It now carries a one-entry geometry cache keyed by
/// [`crate::multigrid::utils::cell_fingerprint`]. `new()` and `default()` are
/// unchanged from a caller's point of view; the cache fills on first use and
/// is dropped whenever a different cell is passed (or on [`Self::reset`]).
///
/// The stored value is small — the pshell records plus the `nao_p x nao`
/// expansion matrix and the level index lists — so there is no memory
/// trade-off to make here. The LARGE per-level collocation tables are NOT
/// cached across calls; they are shared between the forward and reverse
/// passes of ONE call instead, which is where the duplication actually was
/// (see [`MultiGridNumInt::nr_rks`]).
#[derive(Debug, Default)]
pub struct MultiGridNumInt {
    prepared: Mutex<Option<(u64, Arc<V1Tasks>)>>,
}

/// `(exc, nelec, veff)` — [`MultiGridNumInt::nr_rks`]'s return, mirroring
/// upstream's `nr_rks` tuple (`multigrid.py:1059-1155`) minus `vj`/`ecoul`
/// tagging (returned alongside instead of attached to the array).
#[derive(Debug, Clone)]
pub struct MgNrRksResult {
    pub nelec: f64,
    pub exc: f64,
    pub ecoul: f64,
    /// `nao x nao` row-major.
    pub veff: Vec<f64>,
}

/// [`MultiGridNumInt::nr_uks`]'s return — the open-shell twin of
/// [`MgNrRksResult`], with a per-spin electron count and two potentials.
#[derive(Debug, Clone)]
pub struct MgNrUksResult {
    /// `(n_alpha, n_beta)` from the numerical integration.
    pub nelec: (f64, f64),
    pub exc: f64,
    pub ecoul: f64,
    /// `[alpha, beta]`, each `nao x nao` row-major.
    pub veff: [Vec<f64>; 2],
}

impl MultiGridNumInt {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the cached geometry — the `KNumInt::reset` idiom. Only needed if a
    /// caller mutates a `Cell` in place in a way
    /// [`crate::multigrid::utils::cell_fingerprint`] would not see; a
    /// different cell is detected automatically.
    pub fn reset(&self) {
        if let Ok(mut g) = self.prepared.lock() {
            *g = None;
        }
    }

    /// The decontraction and the level task list, memoised on the cell — M-02.
    ///
    /// # Errors
    /// Propagates [`tasks::build_pshells`] / [`tasks::multi_grids_tasks_for_ke_cut`].
    fn build_tasks(&self, cell: &Cell) -> Result<Arc<V1Tasks>, PbcDftError> {
        let key = cell_fingerprint(cell);
        if let Ok(g) = self.prepared.lock()
            && let Some((k, v)) = g.as_ref()
            && *k == key
        {
            let _span = tracing::info_span!("pbc_mg_build_tasks_hit").entered();
            return Ok(Arc::clone(v));
        }
        let _span = tracing::info_span!("pbc_mg_build_tasks_miss").entered();
        let decon = tasks::build_pshells(cell)?;
        let levels = tasks::multi_grids_tasks_for_ke_cut(cell, &decon, cell.mesh)?;
        let out = Arc::new((decon, levels));
        if let Ok(mut g) = self.prepared.lock() {
            *g = Some((key, Arc::clone(&out)));
        }
        Ok(out)
    }

    /// `get_nuc(mydf)` — delegated, see the module doc.
    ///
    /// # Errors
    /// Propagates `Fftdf::new`/`get_nuc`.
    pub fn get_nuc(&self, cell: &Cell) -> Result<Vec<f64>, PbcDftError> {
        let df = Fftdf::new(cell.clone(), &[GAMMA]).map_err(wrap_df)?;
        let v = pyscf_pbc_df::fftdf::get_nuc(&df, &[GAMMA]).map_err(wrap_df)?;
        Ok(v[0].re.clone())
    }

    /// `get_pp(mydf)` — delegated, see the module doc.
    ///
    /// # Errors
    /// Propagates `Fftdf::new`/`get_pp`.
    pub fn get_pp(&self, cell: &Cell) -> Result<Vec<f64>, PbcDftError> {
        let df = Fftdf::new(cell.clone(), &[GAMMA]).map_err(wrap_df)?;
        let v = pyscf_pbc_df::fftdf::get_pp(&df, &[GAMMA]).map_err(wrap_df)?;
        Ok(v[0].re.clone())
    }

    /// `rho(G)` on `cell.mesh`, combined from every grid level via the
    /// G-space window insertion `_eval_rhoG`/`_takebak_4d` use
    /// (`multigrid.py:546-679`). `dm` is the CONTRACTED (`nao x nao`,
    /// row-major) density matrix.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT construction.
    pub fn eval_rho_g(&self, cell: &Cell, dm: &[f64]) -> Result<CTensor, PbcDftError> {
        let prep = self.build_tasks(cell)?;
        let (decon, levels) = (&prep.0, &prep.1);
        let dm_p = colloc::expand_dm(decon, dm);
        let lvs = collocate_all_levels(cell, decon, levels)?;
        rho_g_from_level_values(cell, decon, &lvs, &dm_p)
    }

    /// `get_j_kpts` at gamma — `multigrid.py:515-544`, `_get_j_pass2`
    /// (`:850-940`)'s `grids_sparse is None` branch generalised to every
    /// level via [`colloc::level_pass2`].
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT construction.
    #[allow(clippy::needless_range_loop)]
    pub fn get_j(&self, cell: &Cell, dm: &[f64]) -> Result<Vec<f64>, PbcDftError> {
        let prep = self.build_tasks(cell)?;
        let (decon, levels) = (&prep.0, &prep.1);
        let dm_p = colloc::expand_dm(decon, dm);
        // M-02: collocate each level ONCE and share it between the forward
        // (density) and reverse (`pass2`) directions. They used to be two
        // independent `collocate_level` sweeps over the same levels with the
        // same inputs, i.e. the collocation — the dominant cost of a v1 call —
        // was done twice. Bit-exact: the same table, used twice.
        let lvs = collocate_all_levels(cell, decon, levels)?;
        let rho_g = rho_g_from_level_values(cell, decon, &lvs, &dm_p)?;

        let mesh = cell.mesh;
        let gv = get_gv(cell, Some(mesh))?;
        let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
        let mut vg = rho_g;
        for g in 0..vg.re.len() {
            vg.re[g] *= coulg[g];
            vg.im[g] *= coulg[g];
        }
        let v_p = pass2_from_level_values(cell, decon, &lvs, &vg)?;
        Ok(colloc::contract_v(decon, &v_p))
    }

    /// `nr_rks(mydf, xc_code, dm, with_j=True)` at gamma — `multigrid.py:1059-1155`.
    ///
    /// GGA support uses upstream's DEFAULT `RHOG_HIGH_ORDER=False` route:
    /// `grad rho` is obtained in G-space (`i*Gv*rhoG`, never a real-space AO
    /// gradient), and the GGA weight is folded back to a single LDA-style
    /// scalar field via `wv[0] -= i*Gv . wv[1:4]` before pass2 — see
    /// `multigrid.py:1137-1141`. This is why [`colloc::level_pass2`] never
    /// needs a GGA-typed kernel (17-11-PLAN.md Task 2's kernel is LDA-only by
    /// design, not by omission).
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT / XC evaluation.
    #[allow(clippy::needless_range_loop)]
    pub fn nr_rks(
        &self,
        cell: &Cell,
        xc_code: &str,
        dm: &[f64],
    ) -> Result<MgNrRksResult, PbcDftError> {
        let prep = self.build_tasks(cell)?;
        let (decon, levels) = (&prep.0, &prep.1);
        let dm_p = colloc::expand_dm(decon, dm);
        let keep = keep_level_values(decon, levels);
        tracing::debug!(
            target: "pyscf_pbc_dft::multigrid",
            keep_all_levels = keep,
            estimated_bytes = estimated_level_value_bytes(decon, levels),
            "v1 level-value memory decision"
        );
        let lvs = keep
            .then(|| collocate_all_levels(cell, decon, levels))
            .transpose()?;
        let rho_g = if let Some(lvs) = &lvs {
            rho_g_from_level_values(cell, decon, lvs, &dm_p)?
        } else {
            rho_g_streaming(cell, decon, levels, &[dm_p.as_slice()])?.remove(0)
        };

        // M-00: the middle is shared with `nr_uks` and with the v2 driver.
        let parts = mg_xc_parts(cell, xc_code, std::slice::from_ref(&rho_g))?;
        let v_p = if let Some(lvs) = &lvs {
            pass2_from_level_values(cell, decon, lvs, &parts.wv_freq0[0])?
        } else {
            pass2_streaming(cell, decon, levels, &[&parts.wv_freq0[0]])?.remove(0)
        };
        let veff = colloc::contract_v(decon, &v_p);

        Ok(MgNrRksResult {
            nelec: parts.nelec[0],
            exc: parts.exc,
            ecoul: parts.ecoul,
            veff,
        })
    }

    /// `nr_uks(mydf, xc_code, [dm_a, dm_b], with_j=True)` at gamma —
    /// `multigrid.py:1166-1270`. **M-00.**
    ///
    /// # Why this did not exist before
    ///
    /// 17-11 shipped `nr_rks` only, and 17-12 the same for v2, so "KUKS on
    /// multigrid" was a phrase and not a code path — which also meant no
    /// multigrid optimisation could be validated on an open-shell density
    /// (RULE U). The Coulomb half is built from the SPIN-SUMMED density and
    /// both channels receive the same `vG`; only the XC evaluation and the
    /// two `pass2` sweeps are per spin.
    ///
    /// The collocation is done ONCE per spin for the density and then REUSED
    /// for both `pass2` sweeps (M-02), so an open-shell evaluation costs two
    /// density sweeps and two `pass2` sweeps over ONE set of level tables —
    /// not two independent `nr_rks` calls.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT / XC evaluation.
    pub fn nr_uks(
        &self,
        cell: &Cell,
        xc_code: &str,
        dm: &[&[f64]; 2],
    ) -> Result<MgNrUksResult, PbcDftError> {
        let prep = self.build_tasks(cell)?;
        let (decon, levels) = (&prep.0, &prep.1);
        let dm_p = dm.map(|d| colloc::expand_dm(decon, d));
        let keep = keep_level_values(decon, levels);
        tracing::debug!(
            target: "pyscf_pbc_dft::multigrid",
            keep_all_levels = keep,
            estimated_bytes = estimated_level_value_bytes(decon, levels),
            "v1 level-value memory decision"
        );
        let lvs = keep
            .then(|| collocate_all_levels(cell, decon, levels))
            .transpose()?;
        let rho_g = if let Some(lvs) = &lvs {
            vec![
                rho_g_from_level_values(cell, decon, lvs, &dm_p[0])?,
                rho_g_from_level_values(cell, decon, lvs, &dm_p[1])?,
            ]
        } else {
            rho_g_streaming(cell, decon, levels, &[&dm_p[0], &dm_p[1]])?
        };

        let parts = mg_xc_parts(cell, xc_code, &rho_g)?;
        let vps = if let Some(lvs) = &lvs {
            vec![
                pass2_from_level_values(cell, decon, lvs, &parts.wv_freq0[0])?,
                pass2_from_level_values(cell, decon, lvs, &parts.wv_freq0[1])?,
            ]
        } else {
            pass2_streaming(
                cell,
                decon,
                levels,
                &[&parts.wv_freq0[0], &parts.wv_freq0[1]],
            )?
        };
        let mut veff: Vec<Vec<f64>> = vps
            .iter()
            .map(|v_p| colloc::contract_v(decon, v_p))
            .collect();
        let vb = veff.pop().expect("two channels");
        let va = veff.pop().expect("two channels");

        Ok(MgNrUksResult {
            nelec: (parts.nelec[0], parts.nelec[1]),
            exc: parts.exc,
            ecoul: parts.ecoul,
            veff: [va, vb],
        })
    }
}

/// What [`mg_xc_parts`] returns: everything a multigrid `nr_rks`/`nr_uks`
/// needs between the density and the `pass2` sweep.
///
/// M-00. The two drivers (v1 `MultiGridNumInt`, v2 `MultiGridNumInt2`) and the
/// two spin cases (RKS, UKS) share this middle entirely — upstream's
/// `multigrid.py:1059` `nr_rks` and `:1166` `nr_uks` differ only in producing
/// `rhoG` per spin, summing it for the Coulomb term, and calling the
/// unrestricted XC evaluator. Writing it once is what makes `nr_uks` a small
/// addition to each driver instead of a second copy of this arithmetic.
pub(crate) struct MgXcParts {
    /// Integrated electron count, per spin channel.
    pub nelec: Vec<f64>,
    /// `Σ_s Σ_g rho_s(g) · eps_xc(g) · w` — upstream's `excsum`.
    pub exc: f64,
    /// `0.5 · Re<rho_sf|v_G> / vol` on the SPIN-SUMMED density.
    pub ecoul: f64,
    /// The G-space field each spin's `pass2` sweep contracts, `wv_freq[s][0]`
    /// with the GGA divergence folded in and the Coulomb potential added.
    pub wv_freq0: Vec<CTensor>,
}

/// The spin-generic middle of a multigrid `nr_rks` / `nr_uks` — M-00.
///
/// `rho_g[s]` is spin channel `s`'s density on `cell.mesh` in G-space, already
/// carrying the `weight` scaling `_eval_rhoG` applies. One entry means RKS,
/// two means UKS; nothing else is accepted.
///
/// Ported from `multigrid.py:1059-1160` (`nr_rks`) and `:1166-1270`
/// (`nr_uks`), which agree line for line once the spin axis is factored out:
///
/// * the Coulomb term uses the SPIN-SUMMED density
///   (`rhoG_sf = rhoG[0,0] + rhoG[1,0]`, `:1223`), and both channels then
///   receive the SAME `vG` (`wv_freq[:,0] += vG`, `:1246`);
/// * `excsum` is `rhoR[:,0].dot(exc).sum()` (`:1238`) — a per-spin dot
///   followed by a sum over spins, which is the association reproduced here
///   rather than one flat reduction over `2 * ngrids` terms (the U-03
///   discipline: upstream's association is part of the answer);
/// * the GGA route is upstream's DEFAULT `RHOG_HIGH_ORDER = False` one —
///   `grad rho` from G-space, then `wv[0] -= i·Gv·wv[1:4]` (`:1256`) — so
///   `pass2` is always an LDA-shaped contraction.
///
/// # Bit-parity against the pre-M-00 `nr_rks`
///
/// EXACT for the one-channel case: with `rho_g.len() == 1` the spin sum is
/// `0 + x`, the `excsum` composition is a one-element pairwise sum, and every
/// other statement is the same arithmetic in the same order.
///
/// # Errors
/// Propagates the FFT and the XC evaluation, and rejects a channel count
/// other than 1 or 2.
pub(crate) fn mg_xc_parts(
    cell: &Cell,
    xc_code: &str,
    rho_g: &[CTensor],
) -> Result<MgXcParts, PbcDftError> {
    let nspin = rho_g.len();
    if nspin != 1 && nspin != 2 {
        return Err(PbcDftError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!(
                "multigrid: expected 1 (RKS) or 2 (UKS) density channels, got {nspin}"
            )),
        )));
    }
    let ty = XcType::of(xc_code)?;
    let mesh = cell.mesh;
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    let vol = cell.vol();
    let weight = vol / ngrids as f64;
    let gv = get_gv(cell, Some(mesh))?;
    let coulg = get_coulg_at_gv(cell, mesh, &gv)?;

    // `multigrid.py:1223` — the Coulomb term sees the SPIN-SUMMED density.
    // For `nspin == 1` this is `0 + rho`, i.e. bit-identical to using it
    // directly.
    let mut rho_sf = CTensor::zeros(ngrids);
    for r in rho_g {
        for g in 0..ngrids {
            rho_sf.re[g] += r.re[g];
            rho_sf.im[g] += r.im[g];
        }
    }
    let mut vg = rho_sf.clone();
    for g in 0..ngrids {
        vg.re[g] *= coulg[g];
        vg.im[g] *= coulg[g];
    }
    // P-03: ordered, not `Iterator::sum` — see the note this replaced.
    let ecoul_terms: Vec<f64> = (0..ngrids)
        .map(|g| rho_sf.re[g] * vg.re[g] + rho_sf.im[g] * vg.im[g])
        .collect();
    let ecoul = 0.5 * oracle_sum(&ecoul_terms) / vol;

    // Real-space density (and, for GGA, its gradient) per spin.
    let mut rho_eff: Vec<RhoEff> = Vec::with_capacity(nspin);
    let mut rho_scalar: Vec<Vec<f64>> = Vec::with_capacity(nspin);
    let mut nelec: Vec<f64> = Vec::with_capacity(nspin);
    for r in rho_g {
        let rho_r = pyscf_pbc_tools::ifft(r, mesh).map_err(wrap_tools)?;
        let scalar: Vec<f64> = rho_r.re.iter().map(|x| x / weight).collect();
        nelec.push(oracle_sum(&scalar) * weight);
        let mut eff = RhoEff::zeros(ty, ngrids);
        eff.row_mut(0).copy_from_slice(&scalar);
        if ty == XcType::Gga {
            for axis in 0..3 {
                let mut gcomp = CTensor::zeros(ngrids);
                for g in 0..ngrids {
                    // d(rho)/dx_axis in G-space: i * Gv[g,axis] * rhoG[g].
                    let gk = gv[g][axis];
                    gcomp.re[g] = -gk * r.im[g];
                    gcomp.im[g] = gk * r.re[g];
                }
                let grad_r = pyscf_pbc_tools::ifft(&gcomp, mesh).map_err(wrap_tools)?;
                let row = eff.row_mut(1 + axis);
                for g in 0..ngrids {
                    row[g] = grad_r.re[g] / weight;
                }
            }
        }
        rho_eff.push(eff);
        rho_scalar.push(scalar);
    }

    let xc_out: VxcEff = if nspin == 1 {
        eval_xc_eff_rks(xc_code, &rho_eff[0])?
    } else {
        eval_xc_eff_uks(xc_code, &rho_eff[0], &rho_eff[1])?
    };

    // `multigrid.py:1238` — a per-spin dot, then a sum over spins.
    let exc_row = &xc_out.exc;
    let per_spin: Vec<f64> = rho_scalar
        .iter()
        .map(|scalar| {
            let terms: Vec<f64> = (0..ngrids).map(|g| scalar[g] * exc_row[g]).collect();
            oracle_sum(&terms)
        })
        .collect();
    let exc = oracle_sum(&per_spin) * weight;

    // `wv = weight * vxc`, FFT'd per row; the GGA rows fold into row 0 via the
    // G-space divergence, then the Coulomb potential is added to row 0.
    let mut wv_freq0 = Vec::with_capacity(nspin);
    for s in 0..nspin {
        let mut rows: Vec<CTensor> = Vec::with_capacity(ty.nvar());
        for v in 0..ty.nvar() {
            let row = xc_out.row(s, v);
            let wv: Vec<f64> = row.iter().map(|x| x * weight).collect();
            let wv_c = CTensor::from_planes(wv, vec![0.0; ngrids]);
            rows.push(pyscf_pbc_tools::fft(&wv_c, mesh).map_err(wrap_tools)?);
        }
        if ty == XcType::Gga {
            for g in 0..ngrids {
                let mut dot_re = 0.0f64;
                let mut dot_im = 0.0f64;
                for axis in 0..3 {
                    let gk = gv[g][axis];
                    // i * Gv . wv_freq[1:4]
                    dot_re += -gk * rows[1 + axis].im[g];
                    dot_im += gk * rows[1 + axis].re[g];
                }
                rows[0].re[g] -= dot_re;
                rows[0].im[g] -= dot_im;
            }
        }
        // with_j: fold the Coulomb potential into the same G-space field. Both
        // channels get the SAME `vG` (`multigrid.py:1246`).
        for g in 0..ngrids {
            rows[0].re[g] += vg.re[g];
            rows[0].im[g] += vg.im[g];
        }
        wv_freq0.push(rows.swap_remove(0));
    }

    Ok(MgXcParts {
        nelec,
        exc,
        ecoul,
        wv_freq0,
    })
}

pub(crate) fn wrap_tools(e: pyscf_pbc_tools::PbcToolsError) -> PbcDftError {
    PbcDftError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(format!("{e}")),
    ))
}

pub(crate) fn wrap_df(e: pyscf_pbc_df::PbcDfError) -> PbcDftError {
    PbcDftError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(format!("{e}")),
    ))
}

/// Collocate every level ONCE — M-02.
///
/// The forward (`rho`) and reverse (`pass2`) directions of one v1 call read
/// the identical per-level Cartesian primitive values, and used to build them
/// independently: `rho_g_from_levels` called `collocate_level` per level and
/// `pass2_from_full_vg` called it again, per level, a few statements later.
/// Collocation is the dominant cost of a v1 evaluation (a `PeriodicGrids`
/// build, a per-pshell lattice image list, and one kernel launch per level),
/// so that was a factor of two on the whole driver.
///
/// The tables are held for the duration of ONE call and dropped at its end —
/// they are `n_slots * ngrids` doubles per level and caching them across SCF
/// cycles is a memory trade-off this item deliberately does not make.
///
/// # Errors
/// Propagates [`colloc::collocate_level`].
fn estimated_level_value_bytes(decon: &Decontracted, levels: &[GridLevel]) -> usize {
    levels
        .iter()
        .map(|level| {
            let nslots: usize = level
                .dense
                .iter()
                .chain(&level.sparse)
                .map(|&p| crate::multigrid::tasks::pshell_cart_powers(decon.pshells[p].l).len())
                .sum();
            nslots
                .saturating_mul(level.mesh.iter().product::<usize>())
                .saturating_mul(core::mem::size_of::<f64>())
        })
        .sum()
}

fn keep_level_values(decon: &Decontracted, levels: &[GridLevel]) -> bool {
    let max_mb = std::env::var("PYSCF_MAX_MEMORY")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|x| x.is_finite() && *x >= 0.0)
        .unwrap_or(4000.0);
    estimated_level_value_bytes(decon, levels) as f64 <= 0.25 * max_mb * 1.0e6
}

fn collocate_all_levels(
    cell: &Cell,
    decon: &Decontracted,
    levels: &[GridLevel],
) -> Result<Vec<LevelValues>, PbcDftError> {
    levels
        .iter()
        .map(|level| colloc::collocate_level(cell, decon, level))
        .collect()
}

/// Bounded-memory forward phase. Since the XC field is known only after all
/// levels contribute, the reverse phase necessarily re-collocates levels.
fn rho_g_streaming(
    cell: &Cell,
    decon: &Decontracted,
    levels: &[GridLevel],
    dm_p: &[&[f64]],
) -> Result<Vec<CTensor>, PbcDftError> {
    let mesh = cell.mesh;
    let ngrids: usize = mesh.iter().product();
    let mut out: Vec<CTensor> = dm_p.iter().map(|_| CTensor::zeros(ngrids)).collect();
    let vol = cell.vol();
    for (level_idx, level) in levels.iter().enumerate() {
        let _level_span = tracing::info_span!(
            "pbc_mg_forward_level",
            level = level_idx as u64,
            launches = 1_u64,
            mesh = ?level.mesh,
        )
        .entered();
        let lv = colloc::collocate_level(cell, decon, level)?;
        let scale = vol / lv.ngrids as f64;
        for (dst, dm) in out.iter_mut().zip(dm_p) {
            let rr = CTensor::from_planes(colloc::level_rho(&lv, decon, dm), vec![0.0; lv.ngrids]);
            let mut freq = {
                let _fft_span = tracing::info_span!(
                    "pbc_mg_fft",
                    level = level_idx as u64,
                    direction = "forward",
                )
                .entered();
                pyscf_pbc_tools::fft(&rr, lv.mesh).map_err(wrap_tools)?
            };
            for x in freq.re.iter_mut().chain(freq.im.iter_mut()) {
                *x *= scale;
            }
            insert_gspace_window(dst, mesh, &freq, lv.mesh);
        }
    }
    Ok(out)
}

fn pass2_streaming(
    cell: &Cell,
    decon: &Decontracted,
    levels: &[GridLevel],
    vg_full: &[&CTensor],
) -> Result<Vec<Vec<f64>>, PbcDftError> {
    let mesh = cell.mesh;
    let mut out = vec![vec![0.0f64; decon.nao_p * decon.nao_p]; vg_full.len()];
    for (level_idx, level) in levels.iter().enumerate() {
        let _level_span = tracing::info_span!(
            "pbc_mg_reverse_level",
            level = level_idx as u64,
            launches = 0_u64,
            mesh = ?level.mesh,
        )
        .entered();
        let lv = colloc::collocate_level(cell, decon, level)?;
        for (dst, vg) in out.iter_mut().zip(vg_full) {
            let sub = extract_gspace_window(vg, mesh, lv.mesh);
            let vr = {
                let _fft_span = tracing::info_span!(
                    "pbc_mg_fft",
                    level = level_idx as u64,
                    direction = "reverse",
                )
                .entered();
                pyscf_pbc_tools::ifft(&sub, lv.mesh).map_err(wrap_tools)?
            };
            colloc::level_pass2(&lv, decon, &vr.re, dst);
        }
    }
    Ok(out)
}

/// Combine every level's real-space `rho` into `rho(G)` on `cell.mesh`,
/// via `fft(rho_level) * weight_level` inserted at that level's own
/// G-window into the big mesh's G array (`_eval_rhoG`/`_takebak_4d`,
/// `multigrid.py:664-679`).
///
/// M-02: takes the already-collocated level tables rather than building them.
fn rho_g_from_level_values(
    cell: &Cell,
    decon: &Decontracted,
    lvs: &[LevelValues],
    dm_p: &[f64],
) -> Result<CTensor, PbcDftError> {
    let mesh = cell.mesh;
    let ngrids_full = mesh[0] * mesh[1] * mesh[2];
    let mut rho_g = CTensor::zeros(ngrids_full);
    let vol = cell.vol();
    for (level, lv) in lvs.iter().enumerate() {
        let _level_span = tracing::info_span!(
            "pbc_mg_forward_level",
            level = level as u64,
            launches = 1_u64,
            mesh = ?lv.mesh,
        )
        .entered();
        let rho_r = colloc::level_rho(lv, decon, dm_p);
        let ngrids_level = lv.ngrids;
        let weight = vol / ngrids_level as f64;
        let rr = CTensor::from_planes(rho_r, vec![0.0; ngrids_level]);
        let mut freq = {
            let _fft_span =
                tracing::info_span!("pbc_mg_fft", level = level as u64, direction = "forward",)
                    .entered();
            pyscf_pbc_tools::fft(&rr, lv.mesh).map_err(wrap_tools)?
        };
        for x in freq.re.iter_mut().chain(freq.im.iter_mut()) {
            *x *= weight;
        }
        insert_gspace_window(&mut rho_g, mesh, &freq, lv.mesh);
    }
    Ok(rho_g)
}

/// Contract a G-space weight field on the FULL mesh, level by level: extract
/// each level's own G-window, `ifft` it to that level's real-space mesh, and
/// accumulate that level's `pass2` contribution into a decontracted
/// potential matrix.
///
/// # Errors
/// Propagates the FFT / collocation.
///
/// M-02: takes the already-collocated level tables rather than rebuilding
/// them, which is the half of the duplication this driver used to pay.
fn pass2_from_level_values(
    cell: &Cell,
    decon: &Decontracted,
    lvs: &[LevelValues],
    vg_full: &CTensor,
) -> Result<Vec<f64>, PbcDftError> {
    let mesh = cell.mesh;
    let mut v_p = vec![0.0f64; decon.nao_p * decon.nao_p];
    for (level, lv) in lvs.iter().enumerate() {
        let _level_span = tracing::info_span!(
            "pbc_mg_reverse_level",
            level = level as u64,
            launches = 0_u64,
            mesh = ?lv.mesh,
        )
        .entered();
        let sub = extract_gspace_window(vg_full, mesh, lv.mesh);
        let v_r = {
            let _fft_span =
                tracing::info_span!("pbc_mg_fft", level = level as u64, direction = "reverse",)
                    .entered();
            pyscf_pbc_tools::ifft(&sub, lv.mesh).map_err(wrap_tools)?
        };
        colloc::level_pass2(lv, decon, &v_r.re, &mut v_p);
    }
    Ok(v_p)
}

/// The integer FFT frequency `numpy.fft.fftfreq(n, 1/n)` assigns index `i`.
pub(crate) fn fftfreq_int(n: usize, i: usize) -> i64 {
    let cutoff = ((n as i64) - 1) / 2;
    if i as i64 <= cutoff {
        i as i64
    } else {
        i as i64 - n as i64
    }
}

/// Map a level-mesh flat index to the corresponding full-mesh flat index,
/// via the shared integer-frequency identity both meshes carry (level mesh
/// axes are always `<= ` the full mesh's, so every level frequency has a
/// unique home in the full mesh, no aliasing).
pub(crate) fn window_index_map(mesh_small: [usize; 3], mesh_big: [usize; 3]) -> Vec<usize> {
    let mut map = Vec::with_capacity(mesh_small[0] * mesh_small[1] * mesh_small[2]);
    for ix in 0..mesh_small[0] {
        let fx = fftfreq_int(mesh_small[0], ix);
        let bx = if fx >= 0 {
            fx as usize
        } else {
            (mesh_big[0] as i64 + fx) as usize
        };
        for iy in 0..mesh_small[1] {
            let fy = fftfreq_int(mesh_small[1], iy);
            let by = if fy >= 0 {
                fy as usize
            } else {
                (mesh_big[1] as i64 + fy) as usize
            };
            for iz in 0..mesh_small[2] {
                let fz = fftfreq_int(mesh_small[2], iz);
                let bz = if fz >= 0 {
                    fz as usize
                } else {
                    (mesh_big[2] as i64 + fz) as usize
                };
                map.push((bx * mesh_big[1] + by) * mesh_big[2] + bz);
            }
        }
    }
    map
}

pub(crate) fn insert_gspace_window(
    big: &mut CTensor,
    mesh_big: [usize; 3],
    small: &CTensor,
    mesh_small: [usize; 3],
) {
    let map = window_index_map(mesh_small, mesh_big);
    for (i, &bi) in map.iter().enumerate() {
        big.re[bi] += small.re[i];
        big.im[bi] += small.im[i];
    }
}

pub(crate) fn extract_gspace_window(
    big: &CTensor,
    mesh_big: [usize; 3],
    mesh_small: [usize; 3],
) -> CTensor {
    let map = window_index_map(mesh_small, mesh_big);
    let mut small = CTensor::zeros(map.len());
    for (i, &bi) in map.iter().enumerate() {
        small.re[i] = big.re[bi];
        small.im[i] = big.im[bi];
    }
    small
}
