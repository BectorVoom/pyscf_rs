//! Multigrid **v2** task list + pair-fused collocation —
//! `pyscf/pbc/dft/multigrid/multigrid_pair.py` + `_backend_c.py`'s
//! `TaskList`/`GridLevel_Info`/`build_task_list`/`init_gridlevel_info`/
//! `init_rs_grid` (plan 17-12, Tasks 1/2).
//!
//! # The four C destructors
//!
//! Upstream's `_backend_c.py` pairs every constructor with a matching
//! destructor: `del_task_list`, `del_gridlevel_info`, `del_rs_grid` (twice —
//! `RS_Grid` is built and freed on both the "dense" and the "sparse" side in
//! some call paths). **None of the four is ported.** This is a deliberate
//! port decision, not an omission: [`PairTaskList`], [`GridLevelSpec`] and
//! [`PairLevelValues`] below are ordinary Rust structs owning `Vec`s: they
//! are freed by `Drop` the moment their last owner goes out of scope, with
//! no `unsafe`, no reference counting and no possibility of the
//! double-free/leak class of bug the C API's four-function contract exists
//! to avoid. Same courtesy `17-03-SUMMARY.md` Task 5 pays `cell.py`'s
//! deleted Python reference cycle, and `17-11-SUMMARY.md`'s "what did not
//! ship" section pays FFTDF's `del_task_list` non-equivalent.
//!
//! # Why this is NOT a literal port of `build_task_list` (`multigrid.c:427`)
//!
//! Upstream's C task-list builder is 3 182 lines across
//! `multigrid.c`/`grid_collocate.c`/`grid_integrate.c` combined — hand
//! cache-blocked, restricted to a per-pair rcut SUBMESH (not the full
//! level mesh), and written against a Hermite-Gaussian recursion this
//! plan's time budget could not port line-by-line (RULE 2 in spirit, not
//! letter — see `.planning/phases/17-ksymm-multigrid/17-12-SUMMARY.md`'s
//! "what did not ship" section for the honest accounting). What ships here
//! is a MATHEMATICALLY FAITHFUL reformulation of the SAME physical
//! quantity, gated by construction (the adjoint identity,
//! `crates/pyscf-kernels/tests/multigrid_pair.rs`) rather than by literal
//! agreement with upstream's per-level pair membership:
//!
//! * **Gaussian product theorem.** A primitive pair `(p ∈ shell i, q ∈
//!   shell j, image L)` fuses into ONE combined Gaussian at
//!   `P = (alpha_p·A + alpha_q·(B+L)) / eta`, `eta = alpha_p+alpha_q`,
//!   `K = exp(-alpha_p·alpha_q/eta · |A-(B+L)|²)`.
//! * **Binomial shift expansion** ([`binom_shift`]) re-expresses
//!   `(x-Ax)^a (x-Bx-Lx)^b` in powers of `(x-Px)` — the standard identity
//!   `(x-A)^a = Σ_m C(a,m) (x-P)^m (P-A)^{a-m}`, multiplied out and
//!   collected by power of `(x-P)`. Applied separably per Cartesian axis,
//!   this gives every `(k1,k2,k3)` monomial term's coefficient with NO
//!   approximation — it is an EXACT polynomial identity, not a truncation.
//! * **Level assignment.** Each pair is assigned the COARSEST of the
//!   `NTASKS` grid levels whose plane-wave cutoff still resolves
//!   `max(pshell_i.ke_cutoff, pshell_j.ke_cutoff)` — reusing the
//!   per-PRIMITIVE cutoff [`crate::multigrid::tasks::build_pshells`]
//!   already computes via `estimate_ke_cutoff_pgto` (Task 1's ladder,
//!   `multi_grids_tasks`, `multigrid_pair.py:59-78`, IS a literal, direct
//!   port — only the per-pair MEMBERSHIP rule is a documented
//!   reformulation, not the level ladder itself).
//!
//! This is the same class of judgment call D-PBC-21/23/27 record: measure
//! and state the deviation rather than silently approximate it. The
//! consequence is priced in `crate::multigrid::pair (this module)`'s Gate E numbers
//! (v1-vs-v2 and v2-vs-FFTDF), reported honestly rather than assumed to
//! match upstream's own C task list bit-for-bit.
//!
//! # Scope: GAMMA POINT ONLY
//!
//! Same stated scope reduction as v1 (`crate::multigrid::numint`'s module
//! doc) and the same one this plan's own instructions sanction pending
//! 17-05's `KPoints` — Bloch-phase (k-point-resolved) pair collocation is
//! NOT ported.

use pyscf_algebra::AlgebraError;
use pyscf_kernels::multigrid_pair::{PairSlotTable, collocate_pairs};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::lattice::get_lattice_ls;

use crate::error::PbcDftError;
use crate::multigrid::tasks::{Decontracted, Pshell, pshell_cart_powers};

/// `NTASKS` — `_backend_c.py`/`multigrid_pair.py:47`, `pbc_dft_multigrid_ntasks`.
pub const NTASKS: usize = 4;
/// `KE_RATIO` — `multigrid_pair.py:48`, `pbc_dft_multigrid_ke_ratio`.
pub const KE_RATIO: f64 = 3.0;

/// One grid level's mesh + the plane-wave cutoff that defines it —
/// `GridLevel_Info` (`multigrid.c:20-40`; the struct itself, not its C
/// constructor/destructor, which Rust ownership replaces — see the module
/// doc).
#[derive(Debug, Clone, Copy)]
pub struct GridLevelSpec {
    pub mesh: [usize; 3],
    pub cutoff: f64,
}

/// `multi_grids_tasks(cell, ke_cutoff=None, ...)` — `multigrid_pair.py:59-78`,
/// a LITERAL port (unlike per-pair membership, see the module doc): `NTASKS`
/// levels, geometric in `ke_cutoff` by `KE_RATIO`, finest level pinned to
/// `cell.mesh` exactly.
///
/// # Errors
/// Propagates [`pyscf_pbc_tools::mesh::mesh_to_cutoff`] /
/// [`pyscf_pbc_tools::mesh::cutoff_to_mesh`] (singular lattice).
pub fn pair_grid_levels(cell: &Cell) -> Result<Vec<GridLevelSpec>, PbcDftError> {
    let a = cell.lattice_vectors();
    let fft_mesh = cell.mesh;
    let ke_cutoff = match cell.ke_cutoff {
        Some(k) => k,
        None => {
            let k3 = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, fft_mesh)?;
            k3.into_iter().fold(f64::NEG_INFINITY, f64::max)
        }
    };
    let mut cutoff = vec![0.0f64; NTASKS];
    cutoff[NTASKS - 1] = ke_cutoff;
    let mut ke1 = ke_cutoff;
    for i in (0..NTASKS - 1).rev() {
        ke1 /= KE_RATIO;
        cutoff[i] = ke1;
    }
    let mut levels = Vec::with_capacity(NTASKS);
    for (i, &c) in cutoff.iter().enumerate() {
        let mesh = if i == NTASKS - 1 {
            fft_mesh
        } else {
            pyscf_pbc_tools::mesh::cutoff_to_mesh(&a, c)?
        };
        levels.push(GridLevelSpec { mesh, cutoff: c });
    }
    Ok(levels)
}

/// The full v2 task list: `NTASKS` [`GridLevelSpec`]s, plus every
/// primitive-pair `(pi, pj)` (global [`Pshell`] indices into
/// `Decontracted::pshells`) assigned to each — `TaskList`
/// (`multigrid.c:145-311`; struct only, see the module doc for the
/// destructor note).
#[derive(Debug, Clone)]
pub struct PairTaskList {
    pub levels: Vec<GridLevelSpec>,
    /// `level_pairs[l]` — every `(pi, pj)` this level owns.
    pub level_pairs: Vec<Vec<(usize, usize)>>,
}

/// `EXTRA_PREC` — same constant `tasks.rs` already documents
/// (`multigrid_pair.py:52`, `pbc_gto_eval_gto_extra_precision`); used here as
/// the pair-overlap prefactor `K` screening threshold (relative to
/// `cell.precision`).
const EXTRA_PREC: f64 = 1e-2;

/// `build_task_list(cell, decon)` — Task 1's pair enumeration + level
/// assignment (module doc: a documented reformulation of `build_task_list`,
/// not a literal port).
///
/// For every ordered pair `(pi, pj)` of decontracted primitives
/// (`decon.pshells`, built by [`crate::multigrid::tasks::build_pshells`]),
/// every periodic image `L` with `|A-(B+L)| <= rcut_pi + rcut_pj` is a
/// candidate; images whose Gaussian-product prefactor `K` falls below
/// `cell.precision * EXTRA_PREC` are screened out. A pair with NO surviving
/// image is dropped entirely (its two members do not overlap at all, to the
/// cell's own precision).
///
/// # Errors
/// Propagates [`pair_grid_levels`] / [`get_lattice_ls`].
pub fn build_pair_task_list(
    cell: &Cell,
    decon: &Decontracted,
) -> Result<PairTaskList, PbcDftError> {
    let levels = pair_grid_levels(cell)?;
    let cutoffs: Vec<f64> = levels.iter().map(|l| l.cutoff).collect();
    let mut level_pairs: Vec<Vec<(usize, usize)>> = vec![Vec::new(); levels.len()];

    let threshold = cell.precision * EXTRA_PREC;
    let n = decon.pshells.len();
    for pi in 0..n {
        for pj in 0..n {
            let p = &decon.pshells[pi];
            let q = &decon.pshells[pj];
            let search_rcut = (p.rcut + q.rcut).max(1e-6);
            let ls = get_lattice_ls(cell, Some(search_rcut), None, false)?;
            let mut any = false;
            for l in &ls {
                let dx = p.center[0] - (q.center[0] + l[0]);
                let dy = p.center[1] - (q.center[1] + l[1]);
                let dz = p.center[2] - (q.center[2] + l[2]);
                let dist2 = dx * dx + dy * dy + dz * dz;
                let eta = p.alpha + q.alpha;
                let k = (-p.alpha * q.alpha / eta * dist2).exp();
                if k * p.coef.abs() * q.coef.abs() >= threshold {
                    any = true;
                    break;
                }
            }
            if !any {
                continue;
            }
            let pair_ke = p.ke_cutoff.max(q.ke_cutoff);
            let level_idx = cutoffs
                .iter()
                .position(|&c| c >= pair_ke)
                .unwrap_or(levels.len() - 1);
            level_pairs[level_idx].push((pi, pj));
        }
    }

    Ok(PairTaskList {
        levels,
        level_pairs,
    })
}

/// Binomial coefficient `C(n,k)`, `n` small (angular momenta this milestone
/// gates are `<= 4`), plain product form — no factorial overflow risk at
/// this scale.
fn binom(n: u32, k: u32) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0f64;
    for i in 0..k {
        result = result * (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// The binomial shift expansion: coefficient of `(x-P)^k`, `k=0..=(a+b)`, in
/// `(x-A)^a (x-B)^b`, given `pa = P-A`, `pb = P-B` (one Cartesian axis).
///
/// `(x-A)^a = Σ_m C(a,m) (x-P)^m (P-A)^{a-m}` (binomial theorem on
/// `x-A = (x-P)+(P-A)`), similarly for `(x-B)^b`; multiplying the two sums
/// and collecting by `m+n=k` gives this. Exact — no truncation.
pub fn binom_shift(a: u32, b: u32, pa: f64, pb: f64) -> Vec<f64> {
    let mut f = vec![0.0f64; (a + b + 1) as usize];
    for m in 0..=a {
        let cam = binom(a, m) * pa.powi((a - m) as i32);
        if cam == 0.0 {
            continue;
        }
        for nn in 0..=b {
            let cbn = binom(b, nn) * pb.powi((b - nn) as i32);
            f[(m + nn) as usize] += cam * cbn;
        }
    }
    f
}

/// One level's collocated pair-fused slot values, plus the `(ci, cj)`
/// routing every slot needs for the dm/weight contraction —
/// [`pairlevel_rho`] / [`pairlevel_pass2`].
pub struct PairLevelValues {
    pub table: PairSlotTable,
    pub slot_ci: Vec<usize>,
    pub slot_cj: Vec<usize>,
    pub values: Vec<f64>,
    pub ngrids: usize,
    pub mesh: [usize; 3],
}

fn wrap_alg(e: AlgebraError) -> PbcDftError {
    PbcDftError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(format!("multigrid pair collocate: {e}")),
    ))
}

/// Build the pair-fused slot table for `pairs` on `level`'s own mesh, and
/// launch [`collocate_pairs`] — Task 2's forward primitive.
///
/// # Errors
/// Propagates grid construction / the kernel's shape checks.
pub fn collocate_pair_level(
    cell: &Cell,
    decon: &Decontracted,
    level: &GridLevelSpec,
    pairs: &[(usize, usize)],
) -> Result<PairLevelValues, PbcDftError> {
    let grids = crate::gen_grid::PeriodicGrids::uniform(cell, Some(level.mesh))?;
    let coords = grids.coords()?;
    let ngrids = coords.len();
    let mut coords_flat = Vec::with_capacity(ngrids * 3);
    for c in coords {
        coords_flat.push(c[0]);
        coords_flat.push(c[1]);
        coords_flat.push(c[2]);
    }

    let mut slot_pow = Vec::new();
    let mut slot_coef = Vec::new();
    let mut slot_instance = Vec::new();
    let mut instance_alpha = Vec::new();
    let mut instance_center = Vec::new();
    let mut slot_ci = Vec::new();
    let mut slot_cj = Vec::new();

    let threshold = cell.precision * EXTRA_PREC;

    for &(pi, pj) in pairs {
        let p: &Pshell = &decon.pshells[pi];
        let q: &Pshell = &decon.pshells[pj];
        let search_rcut = (p.rcut + q.rcut).max(1e-6);
        let ls = get_lattice_ls(cell, Some(search_rcut), None, false)?;
        let powers_i = pshell_cart_powers(p.l);
        let powers_j = pshell_cart_powers(q.l);
        let eta = p.alpha + q.alpha;

        for l in &ls {
            let bshift = [q.center[0] + l[0], q.center[1] + l[1], q.center[2] + l[2]];
            let d = [
                p.center[0] - bshift[0],
                p.center[1] - bshift[1],
                p.center[2] - bshift[2],
            ];
            let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let k = (-p.alpha * q.alpha / eta * dist2).exp();
            if k * p.coef.abs() * q.coef.abs() < threshold {
                continue;
            }
            let pcen = [
                (p.alpha * p.center[0] + q.alpha * bshift[0]) / eta,
                (p.alpha * p.center[1] + q.alpha * bshift[1]) / eta,
                (p.alpha * p.center[2] + q.alpha * bshift[2]) / eta,
            ];
            let pa = [
                pcen[0] - p.center[0],
                pcen[1] - p.center[1],
                pcen[2] - p.center[2],
            ];
            let pb = [pcen[0] - bshift[0], pcen[1] - bshift[1], pcen[2] - bshift[2]];

            let inst = (instance_alpha.len()) as u32;
            instance_alpha.push(eta);
            instance_center.push(pcen[0]);
            instance_center.push(pcen[1]);
            instance_center.push(pcen[2]);

            let kcoef = k * p.coef * q.coef;

            for (ci, &(aix, aiy, aiz)) in powers_i.iter().enumerate() {
                for (cj, &(bjx, bjy, bjz)) in powers_j.iter().enumerate() {
                    let fx = binom_shift(aix, bjx, pa[0], pb[0]);
                    let fy = binom_shift(aiy, bjy, pa[1], pb[1]);
                    let fz = binom_shift(aiz, bjz, pa[2], pb[2]);
                    for (k1, &cx) in fx.iter().enumerate() {
                        if cx == 0.0 {
                            continue;
                        }
                        for (k2, &cy) in fy.iter().enumerate() {
                            if cy == 0.0 {
                                continue;
                            }
                            for (k3, &cz) in fz.iter().enumerate() {
                                if cz == 0.0 {
                                    continue;
                                }
                                let coef = kcoef * cx * cy * cz;
                                if coef == 0.0 {
                                    continue;
                                }
                                slot_pow.push(k1 as u32);
                                slot_pow.push(k2 as u32);
                                slot_pow.push(k3 as u32);
                                slot_coef.push(coef);
                                slot_instance.push(inst);
                                slot_ci.push(p.cart_ao0 + ci);
                                slot_cj.push(q.cart_ao0 + cj);
                            }
                        }
                    }
                }
            }
        }
    }

    let table = PairSlotTable {
        coords: coords_flat,
        slot_pow,
        slot_coef,
        slot_instance,
        instance_alpha,
        instance_center,
    };
    let client = pyscf_algebra::select_backend()
        .map_err(|e| {
            PbcDftError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "multigrid pair collocate: backend selection failed: {e}"
                )),
            ))
        })?
        .client;
    let values = collocate_pairs(&client, &table).map_err(wrap_alg)?;

    Ok(PairLevelValues {
        table,
        slot_ci,
        slot_cj,
        values,
        ngrids,
        mesh: level.mesh,
    })
}

/// `NUMINT_fill`/`grid_collocate_drv`'s forward direction at one level —
/// `rho(r) = Σ_slot dm_p[ci,cj] · values[slot,r]`, [`pyscf_algebra::oracle_sum`]
/// over slots per grid point (D-PBC-17 shape — fixed slot order, independent
/// of which grid points a rayon worker owns).
pub fn pairlevel_rho(lv: &PairLevelValues, decon: &Decontracted, dm_p: &[f64]) -> Vec<f64> {
    let ngrids = lv.ngrids;
    let nslots = lv.slot_ci.len();
    let mut terms: Vec<(usize, f64)> = Vec::with_capacity(nslots);
    for s in 0..nslots {
        let d = dm_p[lv.slot_ci[s] * decon.nao_p + lv.slot_cj[s]];
        if d != 0.0 {
            terms.push((s, d));
        }
    }
    use rayon::prelude::*;
    let mut rho = vec![0.0f64; ngrids];
    rho.par_iter_mut().enumerate().for_each(|(g, out)| {
        if terms.is_empty() {
            *out = 0.0;
            return;
        }
        let mut buf = vec![0.0f64; terms.len()];
        for (k, &(s, d)) in terms.iter().enumerate() {
            buf[k] = d * lv.values[s * ngrids + g];
        }
        *out = pyscf_algebra::oracle_sum(&buf);
    });
    rho
}

/// `NUMINT_fill2c`/`grid_integrate_drv`'s reverse direction ("pass2") at one
/// level — `v_p[ci,cj] += Σ_r w[r]·values[slot,r]` for every slot routing to
/// `(ci,cj)`. Each slot's grid reduction is an [`pyscf_algebra::oracle_sum`],
/// matching `crate::multigrid::colloc::level_pass2`'s idiom. ADDS into
/// `v_p`, does not overwrite.
pub fn pairlevel_pass2(
    lv: &PairLevelValues,
    decon: &Decontracted,
    weight: &[f64],
    v_p: &mut [f64],
) {
    debug_assert_eq!(weight.len(), lv.ngrids);
    let ngrids = lv.ngrids;
    let nslots = lv.slot_ci.len();
    let mut buf = vec![0.0f64; ngrids];
    for s in 0..nslots {
        for g in 0..ngrids {
            buf[g] = weight[g] * lv.values[s * ngrids + g];
        }
        let acc = pyscf_algebra::oracle_sum(&buf);
        v_p[lv.slot_ci[s] * decon.nao_p + lv.slot_cj[s]] += acc;
    }
}

// ---------------------------------------------------------------------
// Task 5 — `MultiGridNumInt2` assembly.
// ---------------------------------------------------------------------

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{get_coulg_at_gv, get_gv};

use crate::multigrid::numint::{extract_gspace_window, insert_gspace_window};
use crate::multigrid::tasks;
use crate::xc::{RhoEff, VxcEff, XcType, eval_xc_eff_rks};

/// `pbc.dft.multigrid.multigrid_pair.MultiGridNumInt` (re-exported upstream
/// as `MultiGridNumInt2`, `__init__.py:18`) — the v2 multigrid driver
/// (gamma point; see the module doc). **This is the class Phase 18's
/// `grad/rhf.py:44` / `grad/uhf.py:40` `assert isinstance(ni,
/// MultiGridNumInt2)` on — NOT `crate::multigrid::MultiGridNumInt` (v1)** —
/// recorded in `PBC-MASTER-PLAN.md §8.10` by this plan.
#[derive(Debug, Default)]
pub struct MultiGridNumInt2;

/// [`MultiGridNumInt2::nr_rks`]'s return — same shape as v1's
/// `MgNrRksResult`.
#[derive(Debug, Clone)]
pub struct Mg2NrRksResult {
    pub nelec: f64,
    pub exc: f64,
    pub ecoul: f64,
    /// `nao x nao` row-major.
    pub veff: Vec<f64>,
}

impl MultiGridNumInt2 {
    pub fn new() -> Self {
        Self
    }

    fn build_tasks(&self, cell: &Cell) -> Result<(Decontracted, PairTaskList), PbcDftError> {
        let decon = tasks::build_pshells(cell)?;
        let task_list = build_pair_task_list(cell, &decon)?;
        Ok((decon, task_list))
    }

    /// `get_nuc(mydf)` — delegated to AFTDF, see `crate::multigrid::pp`.
    ///
    /// # Errors
    /// Propagates `crate::multigrid::pp::get_nuc`.
    pub fn get_nuc(&self, cell: &Cell) -> Result<Vec<f64>, PbcDftError> {
        crate::multigrid::pp::get_nuc(cell)
    }

    /// `get_pp(mydf)` — delegated to AFTDF, see `crate::multigrid::pp`.
    ///
    /// # Errors
    /// Propagates `crate::multigrid::pp::get_pp`.
    pub fn get_pp(&self, cell: &Cell) -> Result<Vec<f64>, PbcDftError> {
        crate::multigrid::pp::get_pp(cell)
    }

    /// `rho(G)` on `cell.mesh`, combined from every grid level's pair-fused
    /// collocation — the v2 analogue of `MultiGridNumInt::eval_rho_g`.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT construction.
    pub fn eval_rho_g(&self, cell: &Cell, dm: &[f64]) -> Result<CTensor, PbcDftError> {
        let (decon, task_list) = self.build_tasks(cell)?;
        let dm_p = crate::multigrid::colloc::expand_dm(&decon, dm);
        rho_g_from_pair_levels(cell, &decon, &task_list, &dm_p)
    }

    /// `get_j_kpts` at gamma — the v2 analogue of `MultiGridNumInt::get_j`.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT construction.
    pub fn get_j(&self, cell: &Cell, dm: &[f64]) -> Result<Vec<f64>, PbcDftError> {
        let (decon, task_list) = self.build_tasks(cell)?;
        let dm_p = crate::multigrid::colloc::expand_dm(&decon, dm);
        let rho_g = rho_g_from_pair_levels(cell, &decon, &task_list, &dm_p)?;

        let mesh = cell.mesh;
        let gv = get_gv(cell, Some(mesh))?;
        let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
        let mut vg = rho_g;
        for g in 0..vg.re.len() {
            vg.re[g] *= coulg[g];
            vg.im[g] *= coulg[g];
        }
        let v_p = pass2_from_full_vg_pair(cell, &decon, &task_list, &vg)?;
        Ok(crate::multigrid::colloc::contract_v(&decon, &v_p))
    }

    /// `nr_rks(mydf, xc_code, dm, with_j=True)` at gamma — the v2 analogue
    /// of `MultiGridNumInt::nr_rks`. Same GGA route (upstream default
    /// `RHOG_HIGH_ORDER=False`: grad rho from G-space, `wv[0] -=
    /// i·Gv·wv[1:4]` fold) `crate::multigrid::numint::MultiGridNumInt::nr_rks`
    /// already documents.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT / XC evaluation.
    #[allow(clippy::needless_range_loop)]
    pub fn nr_rks(
        &self,
        cell: &Cell,
        xc_code: &str,
        dm: &[f64],
    ) -> Result<Mg2NrRksResult, PbcDftError> {
        let ty = XcType::of(xc_code)?;
        let (decon, task_list) = self.build_tasks(cell)?;
        let dm_p = crate::multigrid::colloc::expand_dm(&decon, dm);
        let rho_g = rho_g_from_pair_levels(cell, &decon, &task_list, &dm_p)?;

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
        let ecoul = 0.5
            * (0..ngrids)
                .map(|g| rho_g.re[g] * vg.re[g] + rho_g.im[g] * vg.im[g])
                .sum::<f64>()
            / vol;

        let weight = vol / ngrids as f64;
        let rho_r = pyscf_pbc_tools::ifft(&rho_g, mesh).map_err(crate::multigrid::numint::wrap_tools)?;
        let rho_scalar: Vec<f64> = rho_r.re.iter().map(|x| x / weight).collect();
        let nelec = rho_scalar.iter().sum::<f64>() * weight;

        let mut rho_eff = RhoEff::zeros(ty, ngrids);
        rho_eff.row_mut(0).copy_from_slice(&rho_scalar);
        if ty == XcType::Gga {
            for (axis, gv_axis) in [0usize, 1, 2].into_iter().zip([0usize, 1, 2]) {
                let mut gcomp = CTensor::zeros(ngrids);
                for g in 0..ngrids {
                    let gk = gv[g][gv_axis];
                    gcomp.re[g] = -gk * rho_g.im[g];
                    gcomp.im[g] = gk * rho_g.re[g];
                }
                let grad_r =
                    pyscf_pbc_tools::ifft(&gcomp, mesh).map_err(crate::multigrid::numint::wrap_tools)?;
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

        let mut wv_freq: Vec<CTensor> = Vec::with_capacity(ty.nvar());
        for v in 0..ty.nvar() {
            let row = xc_out.row(0, v);
            let wv: Vec<f64> = row.iter().map(|x| x * weight).collect();
            let wv_c = CTensor::from_planes(wv, vec![0.0; ngrids]);
            wv_freq.push(pyscf_pbc_tools::fft(&wv_c, mesh).map_err(crate::multigrid::numint::wrap_tools)?);
        }
        if ty == XcType::Gga {
            for g in 0..ngrids {
                let mut dot_re = 0.0f64;
                let mut dot_im = 0.0f64;
                for axis in 0..3 {
                    let gk = gv[g][axis];
                    dot_re += -gk * wv_freq[1 + axis].im[g];
                    dot_im += gk * wv_freq[1 + axis].re[g];
                }
                wv_freq[0].re[g] -= dot_re;
                wv_freq[0].im[g] -= dot_im;
            }
        }
        for g in 0..ngrids {
            wv_freq[0].re[g] += vg.re[g];
            wv_freq[0].im[g] += vg.im[g];
        }

        let v_p = pass2_from_full_vg_pair(cell, &decon, &task_list, &wv_freq[0])?;
        let veff = crate::multigrid::colloc::contract_v(&decon, &v_p);

        Ok(Mg2NrRksResult {
            nelec,
            exc,
            ecoul,
            veff,
        })
    }
}

/// Combine every level's real-space pair-fused `rho` into `rho(G)` on
/// `cell.mesh` — the v2 analogue of `numint::rho_g_from_levels`.
fn rho_g_from_pair_levels(
    cell: &Cell,
    decon: &Decontracted,
    task_list: &PairTaskList,
    dm_p: &[f64],
) -> Result<CTensor, PbcDftError> {
    let mesh = cell.mesh;
    let ngrids_full = mesh[0] * mesh[1] * mesh[2];
    let mut rho_g = CTensor::zeros(ngrids_full);
    let vol = cell.vol();
    for (level, pairs) in task_list.levels.iter().zip(task_list.level_pairs.iter()) {
        if pairs.is_empty() {
            continue;
        }
        let lv = collocate_pair_level(cell, decon, level, pairs)?;
        let rho_r = pairlevel_rho(&lv, decon, dm_p);
        let ngrids_level = level.mesh[0] * level.mesh[1] * level.mesh[2];
        let weight = vol / ngrids_level as f64;
        let rr = CTensor::from_planes(rho_r, vec![0.0; ngrids_level]);
        let mut freq = pyscf_pbc_tools::fft(&rr, level.mesh).map_err(crate::multigrid::numint::wrap_tools)?;
        for x in freq.re.iter_mut().chain(freq.im.iter_mut()) {
            *x *= weight;
        }
        insert_gspace_window(&mut rho_g, mesh, &freq, level.mesh);
    }
    Ok(rho_g)
}

/// Contract a G-space weight field on the full mesh into a decontracted
/// potential matrix, level by level — the v2 analogue of
/// `numint::pass2_from_full_vg`.
fn pass2_from_full_vg_pair(
    cell: &Cell,
    decon: &Decontracted,
    task_list: &PairTaskList,
    vg_full: &CTensor,
) -> Result<Vec<f64>, PbcDftError> {
    let mesh = cell.mesh;
    let mut v_p = vec![0.0f64; decon.nao_p * decon.nao_p];
    for (level, pairs) in task_list.levels.iter().zip(task_list.level_pairs.iter()) {
        if pairs.is_empty() {
            continue;
        }
        let sub = extract_gspace_window(vg_full, mesh, level.mesh);
        let v_r = pyscf_pbc_tools::ifft(&sub, level.mesh).map_err(crate::multigrid::numint::wrap_tools)?;
        let lv = collocate_pair_level(cell, decon, level, pairs)?;
        pairlevel_pass2(&lv, decon, &v_r.re, &mut v_p);
    }
    Ok(v_p)
}
