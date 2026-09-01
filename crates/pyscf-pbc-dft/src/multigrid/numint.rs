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

use pyscf_algebra::CTensor;
use pyscf_pbc_df::fftdf::Fftdf;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::{get_coulg_at_gv, get_gv};

use crate::error::PbcDftError;
use crate::multigrid::colloc::{self, LevelValues};
use crate::multigrid::tasks::{self, Decontracted, GridLevel};
use crate::xc::{RhoEff, VxcEff, XcType, eval_xc_eff_rks};

const GAMMA: [f64; 3] = [0.0, 0.0, 0.0];

/// `pbc.dft.multigrid.MultiGridNumInt` — the v1 multigrid driver (gamma
/// point; see the module doc).
#[derive(Debug, Default)]
pub struct MultiGridNumInt;

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

impl MultiGridNumInt {
    pub fn new() -> Self {
        Self
    }

    fn build_tasks(&self, cell: &Cell) -> Result<(Decontracted, Vec<GridLevel>), PbcDftError> {
        let decon = tasks::build_pshells(cell)?;
        let levels = tasks::multi_grids_tasks_for_ke_cut(cell, &decon, cell.mesh)?;
        Ok((decon, levels))
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
        let (decon, levels) = self.build_tasks(cell)?;
        let dm_p = colloc::expand_dm(&decon, dm);
        rho_g_from_levels(cell, &decon, &levels, &dm_p)
    }

    /// `get_j_kpts` at gamma — `multigrid.py:515-544`, `_get_j_pass2`
    /// (`:850-940`)'s `grids_sparse is None` branch generalised to every
    /// level via [`colloc::level_pass2`].
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT construction.
    #[allow(clippy::needless_range_loop)]
    pub fn get_j(&self, cell: &Cell, dm: &[f64]) -> Result<Vec<f64>, PbcDftError> {
        let (decon, levels) = self.build_tasks(cell)?;
        let dm_p = colloc::expand_dm(&decon, dm);
        let rho_g = rho_g_from_levels(cell, &decon, &levels, &dm_p)?;

        let mesh = cell.mesh;
        let gv = get_gv(cell, Some(mesh))?;
        let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
        let mut vg = rho_g;
        for g in 0..vg.re.len() {
            vg.re[g] *= coulg[g];
            vg.im[g] *= coulg[g];
        }
        let v_p = pass2_from_full_vg(cell, &decon, &levels, &vg)?;
        Ok(colloc::contract_v(&decon, &v_p))
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
        let ty = XcType::of(xc_code)?;
        let (decon, levels) = self.build_tasks(cell)?;
        let dm_p = colloc::expand_dm(&decon, dm);
        let rho_g = rho_g_from_levels(cell, &decon, &levels, &dm_p)?;

        let mesh = cell.mesh;
        let ngrids = mesh[0] * mesh[1] * mesh[2];
        let vol = cell.vol();
        let gv = get_gv(cell, Some(mesh))?;
        let coulg = get_coulg_at_gv(cell, mesh, &gv)?;

        let mut vg = rho_g.clone();
        for g in 0..ngrids {
            vg.re[g] *= coulg[g];
            vg.im[g] *= coulg[g];
        }
        // ecoul = 0.5 * Re(<rhoG|vG>) / vol   (multigrid.py:1112-1114)
        let ecoul = 0.5
            * (0..ngrids)
                .map(|g| rho_g.re[g] * vg.re[g] + rho_g.im[g] * vg.im[g])
                .sum::<f64>()
            / vol;

        let weight = vol / ngrids as f64;
        let rho_r = pyscf_pbc_tools::ifft(&rho_g, mesh).map_err(wrap_tools)?;
        let rho_scalar: Vec<f64> = rho_r.re.iter().map(|x| x / weight).collect();
        let nelec = rho_scalar.iter().sum::<f64>() * weight;

        let mut rho_eff = RhoEff::zeros(ty, ngrids);
        rho_eff.row_mut(0).copy_from_slice(&rho_scalar);
        if ty == XcType::Gga {
            for (axis, gv_axis) in [0usize, 1, 2].into_iter().zip([0usize, 1, 2]) {
                let mut gcomp = CTensor::zeros(ngrids);
                for g in 0..ngrids {
                    // d(rho)/dx_axis in G-space: i * Gv[g,axis] * rhoG[g].
                    let gk = gv[g][gv_axis];
                    gcomp.re[g] = -gk * rho_g.im[g];
                    gcomp.im[g] = gk * rho_g.re[g];
                }
                let grad_r = pyscf_pbc_tools::ifft(&gcomp, mesh).map_err(wrap_tools)?;
                let row = rho_eff.row_mut(1 + axis);
                for g in 0..ngrids {
                    row[g] = grad_r.re[g] / weight;
                }
            }
        }

        let xc_out: VxcEff = eval_xc_eff_rks(xc_code, &rho_eff)?;
        let exc_row = &xc_out.exc;
        let exc = (0..ngrids)
            .map(|g| rho_scalar[g] * exc_row[g])
            .sum::<f64>()
            * weight;

        // wv = weight * vxc, FFT'd per row; GGA folds rows 1..4 into row 0
        // via the G-space divergence trick (see the fn doc).
        let mut wv_freq: Vec<CTensor> = Vec::with_capacity(ty.nvar());
        for v in 0..ty.nvar() {
            let row = xc_out.row(0, v);
            let wv: Vec<f64> = row.iter().map(|x| x * weight).collect();
            let wv_c = CTensor::from_planes(wv, vec![0.0; ngrids]);
            wv_freq.push(pyscf_pbc_tools::fft(&wv_c, mesh).map_err(wrap_tools)?);
        }
        if ty == XcType::Gga {
            for g in 0..ngrids {
                let mut dot_re = 0.0f64;
                let mut dot_im = 0.0f64;
                for axis in 0..3 {
                    let gk = gv[g][axis];
                    // i * Gv . wv_freq[1:4]
                    dot_re += -gk * wv_freq[1 + axis].im[g];
                    dot_im += gk * wv_freq[1 + axis].re[g];
                }
                wv_freq[0].re[g] -= dot_re;
                wv_freq[0].im[g] -= dot_im;
            }
        }
        // with_j: fold the Coulomb potential into the same G-space field.
        for g in 0..ngrids {
            wv_freq[0].re[g] += vg.re[g];
            wv_freq[0].im[g] += vg.im[g];
        }

        let v_p = pass2_from_full_vg(cell, &decon, &levels, &wv_freq[0])?;
        let veff = colloc::contract_v(&decon, &v_p);

        Ok(MgNrRksResult {
            nelec,
            exc,
            ecoul,
            veff,
        })
    }
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

/// Combine every level's real-space `rho` into `rho(G)` on `cell.mesh`,
/// via `fft(rho_level) * weight_level` inserted at that level's own
/// G-window into the big mesh's G array (`_eval_rhoG`/`_takebak_4d`,
/// `multigrid.py:664-679`).
fn rho_g_from_levels(
    cell: &Cell,
    decon: &Decontracted,
    levels: &[GridLevel],
    dm_p: &[f64],
) -> Result<CTensor, PbcDftError> {
    let mesh = cell.mesh;
    let ngrids_full = mesh[0] * mesh[1] * mesh[2];
    let mut rho_g = CTensor::zeros(ngrids_full);
    let vol = cell.vol();
    for level in levels {
        let lv: LevelValues = colloc::collocate_level(cell, decon, level)?;
        let rho_r = colloc::level_rho(&lv, decon, dm_p);
        let ngrids_level = level.mesh[0] * level.mesh[1] * level.mesh[2];
        let weight = vol / ngrids_level as f64;
        let rr = CTensor::from_planes(rho_r, vec![0.0; ngrids_level]);
        let mut freq = pyscf_pbc_tools::fft(&rr, level.mesh).map_err(wrap_tools)?;
        for x in freq.re.iter_mut().chain(freq.im.iter_mut()) {
            *x *= weight;
        }
        insert_gspace_window(&mut rho_g, mesh, &freq, level.mesh);
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
fn pass2_from_full_vg(
    cell: &Cell,
    decon: &Decontracted,
    levels: &[GridLevel],
    vg_full: &CTensor,
) -> Result<Vec<f64>, PbcDftError> {
    let mesh = cell.mesh;
    let mut v_p = vec![0.0f64; decon.nao_p * decon.nao_p];
    for level in levels {
        let sub = extract_gspace_window(vg_full, mesh, level.mesh);
        let v_r = pyscf_pbc_tools::ifft(&sub, level.mesh).map_err(wrap_tools)?;
        let lv = colloc::collocate_level(cell, decon, level)?;
        colloc::level_pass2(&lv, decon, &v_r.re, &mut v_p);
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

pub(crate) fn extract_gspace_window(big: &CTensor, mesh_big: [usize; 3], mesh_small: [usize; 3]) -> CTensor {
    let map = window_index_map(mesh_small, mesh_big);
    let mut small = CTensor::zeros(map.len());
    for (i, &bi) in map.iter().enumerate() {
        small.re[i] = big.re[bi];
        small.im[i] = big.im[bi];
    }
    small
}
