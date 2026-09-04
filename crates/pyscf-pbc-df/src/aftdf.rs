//! `AFTDF` — the analytic-Fourier-transform density-fitting builder
//! (`pyscf/pbc/df/aft.py`, plans 13-03 and 13-04).
//!
//! # Consolidation with plan 13-03, recorded deliberately
//!
//! PBC-MASTER-PLAN §8.5 plan 13-03 asks for a standalone `FtKernel` object that
//! caches the screened record table between G-blocks. That object is folded into
//! [`Aftdf::ft_loop`] here: the record table is a function of `(cell, kpt, rcut)`
//! only, and every consumer in this phase walks G in blocks at a FIXED k-set, so
//! a separate type would carry exactly one field the loop already owns. The
//! properties 13-03 gates — G-block invariance, the `q` shift, `s2` packing —
//! are tested in `tests/aftdf.rs` against this loop instead.
//!
//! # AFTDF and FFTDF differ in ONE place, and it is not `get_pp`
//!
//! `get_pp` = analytic part 1 + `get_pp_loc_part2` + `get_pp_nl`, and the last
//! two are Phase 10's real-space routines, shared verbatim with FFTDF. So any
//! `get_pp` difference between the two builders isolates `ft_aopair`. The real
//! divergence is `exxdiv` — see `aft_jk`.

use std::collections::HashMap;
use std::sync::Mutex;

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{Cell, CoulGArgs, ExxDiv, get_coulg};

use crate::df_jk::KMats;
use crate::error::PbcDfError;
use crate::ft_ao::{FtKernel, FtScreen, RcutChoice, fake_nuc, ft_ao_mol};
use crate::traits::{JkOpts, JkResult, PeriodicDf};

/// `KE_SCALING` — `aft.py`'s guard on an under-resolved mesh.
pub const KE_SCALING: f64 = 0.75;

/// One block of the `ft_loop`: the AO-pair Fourier transform for every k-point
/// over a contiguous slice of G.
#[derive(Debug, Clone)]
pub struct FtBlock {
    /// `[k]` → dense `(nblock, nao, nao)` real plane.
    pub re: Vec<Vec<f64>>,
    /// `[k]` → dense `(nblock, nao, nao)` imaginary plane.
    pub im: Vec<Vec<f64>>,
    /// First G index of this block.
    pub p0: usize,
    /// One past the last G index.
    pub p1: usize,
    /// AO count.
    pub nao: usize,
}

/// `AFTDF(cell, kpts)` — `aft.py:585-770`.
#[derive(Debug)]
pub struct Aftdf {
    /// The cell.
    pub cell: Cell,
    /// Sampling k-points; empty means the single gamma point.
    pub kpts: Vec<[f64; 3]>,
    /// The planewave mesh the G-sum runs over.
    pub mesh: [usize; 3],
    /// Memory budget in MB — sizes the `ft_loop` G-block.
    pub max_memory: f64,
    /// Which lattice-sum radius `ft_aopair` uses. `Upstream` reproduces
    /// upstream's screening (and is the default, as `AFTDF` is defined by it);
    /// `Scaled(1.5)` converges the sum instead.
    pub rcut: RcutChoice,
    /// **MDF only.** Zero the plane waves at `+/-Gmax +/- 0.5` when the
    /// k-point sits on a half-integer of the reciprocal lattice — the screen
    /// upstream applies exclusively inside MDF (`mdf.py:143-172`; the screen
    /// was removed from `tools.pbc.get_coulG` because it broke supercell /
    /// k-point consistency, and re-added there). `mdf_jk` and `mdf_ao2mo` reach
    /// `aft_jk` / `aft_ao2mo` through an `Aftdf` with this set, which is how
    /// upstream reaches it too: its `mydf` IS the `MDF` object, so
    /// `mydf.weighted_coulG` is MDF's.
    pub mdf_pw_edge_screen: bool,
    gv_cache: Mutex<HashMap<[usize; 3], std::sync::Arc<GvCache>>>,
}

#[derive(Debug)]
struct GvCache {
    gv: Vec<[f64; 3]>,
    weights: Vec<f64>,
}

fn default_max_memory() -> f64 {
    std::env::var("PYSCF_MAX_MEMORY")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(4000.0)
}

impl Aftdf {
    /// Build an `AFTDF` for `cell` at `kpts` (empty = gamma), using `cell.mesh`.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] when the cell has no mesh.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Result<Self, PbcDfError> {
        let mesh = cell.try_mesh()?;
        Self::with_mesh(cell, kpts, mesh)
    }

    /// [`Aftdf::new`] with an explicit mesh — upstream's `mydf.mesh = ...`.
    ///
    /// # Errors
    /// As [`Aftdf::new`].
    pub fn with_mesh(cell: Cell, kpts: &[[f64; 3]], mesh: [usize; 3]) -> Result<Self, PbcDfError> {
        let kpts = if kpts.is_empty() {
            vec![[0.0; 3]]
        } else {
            kpts.to_vec()
        };
        Ok(Self {
            cell,
            kpts,
            mesh,
            max_memory: default_max_memory(),
            rcut: RcutChoice::Upstream,
            mdf_pw_edge_screen: false,
            gv_cache: Mutex::new(HashMap::new()),
        })
    }

    fn gv_weights(&self, mesh: [usize; 3]) -> Result<std::sync::Arc<GvCache>, PbcDfError> {
        if let Ok(c) = self.gv_cache.lock()
            && let Some(v) = c.get(&mesh)
        {
            return Ok(v.clone());
        }
        let gw = pyscf_pbc_gto::gv::get_gv_weights(&self.cell, Some(mesh))?;
        let gv = pyscf_pbc_gto::gv::get_gv(&self.cell, Some(mesh))?;
        let weights: Vec<f64> = (0..gv.len()).map(|g| gw.weight(g)).collect();
        let arc = std::sync::Arc::new(GvCache { gv, weights });
        if let Ok(mut c) = self.gv_cache.lock() {
            c.insert(mesh, arc.clone());
        }
        Ok(arc)
    }

    /// `weighted_coulG(kpt, exx, mesh, omega)` — `aft.py:236-245`.
    ///
    /// `get_coulG(...) * kws`. **`exx` is threaded through**, which is the whole
    /// AFTDF/FFTDF `exxdiv` divergence: for a 3-D cell `get_coulG` folds
    /// `Nk·vol·madelung` into `coulG[G+k=0]` instead of applying
    /// `_ewald_exxdiv_for_G0` afterwards.
    ///
    /// # Errors
    /// Propagates `get_coulG`.
    pub fn weighted_coulg(
        &self,
        kpt: [f64; 3],
        exx: Option<ExxDiv>,
        mesh: [usize; 3],
        omega: Option<f64>,
    ) -> Result<Vec<f64>, PbcDfError> {
        let c = self.gv_weights(mesh)?;
        let mut coulg = get_coulg(
            &self.cell,
            CoulGArgs {
                k: kpt,
                exxdiv: exx,
                kpts: Some(&self.kpts),
                mesh: Some(mesh),
                gv: Some(&c.gv),
                omega,
                ..Default::default()
            },
        )?;
        for (g, v) in coulg.iter_mut().enumerate() {
            *v *= c.weights[g];
        }
        if self.mdf_pw_edge_screen {
            crate::mdf::builder::screen_pw_edges(&self.cell, kpt, mesh, &mut coulg);
        }
        Ok(coulg)
    }

    /// The G-vectors this builder sums over, at `mesh`.
    ///
    /// # Errors
    /// Propagates the G-vector build.
    pub fn gv(&self, mesh: [usize; 3]) -> Result<Vec<[f64; 3]>, PbcDfError> {
        Ok(self.gv_weights(mesh)?.gv.clone())
    }

    /// G-block size for `ft_loop`, from `max_memory`.
    fn gblksize(&self, nao: usize, nkpts: usize, ngrids: usize) -> usize {
        let per_g = nao * nao * nkpts * 16; // complex f64 per G, all k
        let n = ((self.max_memory * 1e6 * 0.5) / per_g.max(1) as f64) as usize;
        n.clamp(16, ngrids.max(1)).min(16384)
    }

    /// Build one screened [`FtKernel`] per k-point, ready for repeated
    /// `eval` at different `q` and different G-blocks.
    ///
    /// The table depends on `(cell, kpt, rcut)` and NOT on `q` or `G`, which is
    /// what lets `get_k_kpts` build `nkpts` of them instead of `nkpts²`.
    ///
    /// # Errors
    /// Propagates the lattice-image build and the table build.
    pub fn ft_kernels(&self, kpts: &[[f64; 3]]) -> Result<Vec<FtKernel>, PbcDfError> {
        let radius = self.rcut.resolve_for(&self.cell)?;
        let ls = pyscf_pbc_gto::lattice::get_lattice_ls(&self.cell, Some(radius), None, true)?;
        let screen = if self.rcut == RcutChoice::Upstream {
            FtScreen::Upstream
        } else {
            FtScreen::None
        };
        kpts.iter()
            .map(|k| FtKernel::build(&self.cell, *k, &ls, screen))
            .collect()
    }

    /// G-block bounds for `mesh`, matching [`Self::ft_loop`]'s blocking.
    ///
    /// # Errors
    /// Propagates the G-vector build.
    pub fn g_blocks(
        &self,
        mesh: [usize; 3],
        nkpts: usize,
    ) -> Result<Vec<(usize, usize)>, PbcDfError> {
        let c = self.gv_weights(mesh)?;
        let ngrids = c.gv.len();
        let blk = self.gblksize(self.cell.mol.nao_nr, nkpts.max(1), ngrids);
        let mut v = Vec::new();
        let mut p0 = 0usize;
        while p0 < ngrids {
            let p1 = (p0 + blk).min(ngrids);
            v.push((p0, p1));
            p0 = p1;
        }
        Ok(v)
    }

    /// `ft_loop(mesh, q, kpts, ...)` — `aft.py:495-551`.
    ///
    /// Walks the mesh in G-blocks, yielding the AO-pair transform at every
    /// k-point. The record table inside `ft_aopair` is rebuilt per block, which
    /// is the one thing plan 13-03's `FtKernel` would have cached; see the
    /// module docs.
    ///
    /// # Errors
    /// Propagates `ft_aopair`.
    pub fn ft_loop<F>(
        &self,
        mesh: [usize; 3],
        q: [f64; 3],
        kpts: &[[f64; 3]],
        mut f: F,
    ) -> Result<(), PbcDfError>
    where
        F: FnMut(&FtBlock) -> Result<(), PbcDfError>,
    {
        let c = self.gv_weights(mesh)?;
        let ngrids = c.gv.len();
        let nao = self.cell.mol.nao_nr;
        let blk = self.gblksize(nao, kpts.len().max(1), ngrids);

        // Build the screened record table ONCE per k-point, outside the G loop.
        // It is `O(nimgs·nprim²·npairs)` McMurchie–Davidson recursions and does
        // not depend on `G`; rebuilding it per block made `get_pp` at mesh 15
        // take minutes.
        let kernels = self.ft_kernels(kpts)?;

        let mut p0 = 0usize;
        while p0 < ngrids {
            let p1 = (p0 + blk).min(ngrids);
            let slice = &c.gv[p0..p1];
            let mut re = Vec::with_capacity(kpts.len());
            let mut im = Vec::with_capacity(kpts.len());
            for kern in &kernels {
                let o = kern.eval(&self.cell, slice, q)?;
                re.push(o.re);
                im.push(o.im);
            }
            f(&FtBlock {
                re,
                im,
                p0,
                p1,
                nao,
            })?;
            p0 = p1;
        }
        Ok(())
    }

    /// `_get_pp_loc_part1(mydf, kpts, with_pseudo)` — `aft.py:104-165`.
    ///
    /// # Errors
    /// Propagates the G-space factors and `ft_loop`.
    pub fn get_pp_loc_part1(
        &self,
        kpts: &[[f64; 3]],
        with_pseudo: bool,
    ) -> Result<Vec<CTensor>, PbcDfError> {
        let cell = &self.cell;
        let mesh = self.mesh;
        let nao = cell.mol.nao_nr;
        let nkpts = kpts.len();
        let c = self.gv_weights(mesh)?;
        let ngrids = c.gv.len();

        // vpplocG[G], the nuclear/pseudopotential factor.
        let (mut vg_re, mut vg_im) = (vec![0.0f64; ngrids], vec![0.0f64; ngrids]);
        let natm = cell.mol.natm;
        if with_pseudo {
            // −Σ_i SI[i,G]·get_gth_vlocG_part1[i,G]
            let si = pyscf_pbc_gto::gv::get_si(cell, Some(&c.gv), None, None)?;
            let vloc1 = pyscf_pbc_gto::pseudo::vloc::get_gth_vlocg_part1(cell, &c.gv)?;
            for ia in 0..natm {
                for g in 0..ngrids {
                    let v = vloc1[ia * ngrids + g];
                    vg_re[g] -= si.re[ia * ngrids + g] * v;
                    vg_im[g] -= si.im[ia * ngrids + g] * v;
                }
            }
        } else {
            // Σ_i (−Z_i)·ft_ao(fakenuc)[G,i]·coulG[G]
            let fk = fake_nuc(cell, false)?;
            let (fre, fim) = ft_ao_mol(&fk, &c.gv)?;
            let charges = cell.atom_charges();
            let coulg = get_coulg(
                cell,
                CoulGArgs {
                    mesh: Some(mesh),
                    gv: Some(&c.gv),
                    ..Default::default()
                },
            )?;
            for ia in 0..natm {
                let z = -(charges[ia] as f64);
                for g in 0..ngrids {
                    vg_re[g] += z * fre[g * natm + ia] * coulg[g];
                    vg_im[g] += z * fim[g * natm + ia] * coulg[g];
                }
            }
        }
        // `vpplocG *= kws`
        for g in 0..ngrids {
            vg_re[g] *= c.weights[g];
            vg_im[g] *= c.weights[g];
        }

        // Contract with the AO-pair transform. Upstream packs `s2` and unpacks
        // at the end; this port stays dense — the triangle saves half the
        // arithmetic on a quantity that is not the bottleneck, and staying dense
        // keeps the real/imaginary bookkeeping below literally upstream's.
        let mut out: Vec<CTensor> = (0..nkpts)
            .map(|_| CTensor {
                re: vec![0.0; nao * nao],
                im: vec![0.0; nao * nao],
            })
            .collect();
        let gamma: Vec<bool> = kpts
            .iter()
            .map(|k| k[0].abs() + k[1].abs() + k[2].abs() < 1e-9)
            .collect();

        self.ft_loop(mesh, [0.0; 3], kpts, |b| {
            for k in 0..nkpts {
                let (br, bi) = (&b.re[k], &b.im[k]);
                for (gi, g) in (b.p0..b.p1).enumerate() {
                    let (vr, vi) = (vg_re[g], vg_im[g]);
                    let base = gi * nao * nao;
                    for p in 0..nao * nao {
                        // aft.py:145-151 — vjR += vGR·GpqR + vGI·GpqI;
                        //                  vjI += vGR·GpqI − vGI·GpqR (k ≠ 0).
                        out[k].re[p] += vr * br[base + p] + vi * bi[base + p];
                        if !gamma[k] {
                            out[k].im[p] += vr * bi[base + p] - vi * br[base + p];
                        }
                    }
                }
            }
            Ok(())
        })?;
        Ok(out)
    }
}

/// `get_nuc(mydf, kpts)` — `aft.py:216-234`.
///
/// # Errors
/// Propagates [`Aftdf::get_pp_loc_part1`].
pub fn get_nuc(df: &Aftdf, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    df.get_pp_loc_part1(kpts, false)
}

/// `get_pp(mydf, kpts)` — `aft.py:186-214`.
///
/// Part 1 is the analytic Fourier transform above; parts 2 and the non-local
/// projector come from Phase 10's real-space routines, **shared verbatim with
/// FFTDF**. That is what makes a `get_pp` comparison between the two builders a
/// clean measurement of `ft_aopair`.
///
/// # Errors
/// Propagates part 1, `get_pp_loc_part2` and `get_pp_nl`.
pub fn get_pp(df: &Aftdf, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let cell = &df.cell;
    let nao = cell.mol.nao_nr;
    let mut vpp = df.get_pp_loc_part1(kpts, true)?;
    // Plan 14-03 found that Phase 10's `get_pp_loc_part2` is GAMMA-ONLY, which
    // silently limited this whole builder to gamma. `pp_int::get_pp_loc_part2_kpts`
    // is upstream's `aft._IntPPBuilder`, ported in Phase 14 on top of
    // `incore::aux_e2_intor` — the same double lattice sum with the
    // `int3c1e_r{2,4,6}_origk` operators. It reproduces the gamma route exactly
    // (asserted in `tests/pp_int.rs`) and extends it to every k-point.
    let part2 = crate::pp_int::get_pp_loc_part2_kpts(cell, kpts)?;
    let nl = pyscf_pbc_gto::pseudo::vnl::get_pp_nl(cell, kpts)?;
    for (k, m) in vpp.iter_mut().enumerate() {
        // BOTH Phase-10 outputs are F-ORDER (`zlinalg::forder_to_c`), the same
        // convention `fftdf::get_pp` converts. Adding them raw transposes the
        // non-local block and breaks Hermiticity — which is exactly what the
        // Hermiticity test catches, and the only thing it caught.
        // `get_pp_loc_part2_kpts` is already ROW-MAJOR (it comes out of
        // `aux_e2`, not Phase 10's F-order routines); `get_pp_nl` is still
        // F-order. Phase-13 defect #4 was adding an F-order block raw.
        let nlk = crate::zlinalg::forder_to_c(&nl[k], nao, nao);
        for p in 0..nao * nao {
            m.re[p] += part2[k].re[p] + nlk.re[p];
            m.im[p] += part2[k].im[p] + nlk.im[p];
        }
        // `aft.py` leaves a gamma block real by construction.
        if pyscf_pbc_gto::is_zero(&kpts[k]) {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    Ok(vpp)
}

impl PeriodicDf for Aftdf {
    fn cell(&self) -> &Cell {
        &self.cell
    }
    fn mesh(&self) -> [usize; 3] {
        self.mesh
    }
    fn name(&self) -> &'static str {
        "AFTDF"
    }
    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }
    fn build(&mut self) -> Result<(), PbcDfError> {
        let mesh = self.mesh;
        self.gv_weights(mesh)?;
        Ok(())
    }
    fn get_nuc(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        get_nuc(self, kpts)
    }
    fn get_pp(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        get_pp(self, kpts)
    }
    fn get_jk(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        opts: JkOpts<'_>,
    ) -> Result<JkResult, PbcDfError> {
        crate::aft_jk::get_jk(self, dms, kpts, opts)
    }
}
