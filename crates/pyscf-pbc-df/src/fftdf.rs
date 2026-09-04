//! `FFTDF` — plane-wave density fitting on the FFT box (plans 11-05 / 11-08).
//!
//! Ports `pyscf/pbc/df/fft.py:40-80` (`get_nuc`), `:82-178` (`get_pp`) and
//! `:185-405` (the `FFTDF` class).
//!
//! # Deviation from upstream's `get_pp` (documented, deliberate)
//!
//! `fft.py`'s `get_pp` evaluates the NON-LOCAL half in reciprocal space through
//! `ft_ao.ft_ao`, the McMurchie-Davidson planewave AO transform that
//! PBC-MASTER-PLAN schedules for Phase 13. Phase 10 already shipped the SAME
//! quantity in real space — [`pyscf_pbc_gto::pseudo::get_pp_nl`], via
//! `intor_cross` against the projector fake-cell — and gated it against
//! upstream at 1.9e-15 on diamond. This port therefore assembles
//!
//! ```text
//! V_pp(k) = ifft(-sum_a SI[a] * vlocG[a])  +  V_nl(k)
//! ```
//!
//! using the FFT for the local half (identical to upstream) and the Phase-10
//! real-space route for the non-local half. Both are the same operator; the
//! only difference is which quadrature evaluates it, and the real-space one is
//! the more accurate of the two (it is exact in the basis, with no planewave
//! truncation). `tests/fftdf.rs` pins the assembled `V_pp` against upstream.
//!
//! # The AO cache
//!
//! `eval_ao_kpts` over `ngrids = mesh.product()` points is the single most
//! expensive non-FFT step, and neither the grid nor the k-points move during an
//! SCF. Upstream re-evaluates on every `aoR_loop`; this port memoises the AO
//! table per k-point list, bounded by [`Fftdf::max_memory`]. Cached values are
//! bit-identical to a fresh evaluation — it is the same function of the same
//! inputs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyscf_algebra::{AlgebraClient, CTensor, select_backend};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::{
    Cell, CoulGArgs, ExxDiv, UniformGrids, eval_ao_kpts, get_coulg, get_coulg_at_gv, get_gv,
    get_si, is_zero,
};
use pyscf_pbc_tools::ifft;

use crate::df_jk::KMats;
use crate::error::PbcDfError;
use crate::traits::{JkOpts, JkResult, PeriodicDf};
use crate::zlinalg::{forder_to_c, zadd_assign};

/// AO values on the uniform grid, one `(nao, ngrids)` ROW-MAJOR block per
/// k-point.
///
/// That layout is `eval_ao_kpts`'s native one (its `(ngrids, nao)` F-order per
/// component is the same buffer) and it is also upstream's `ao2T`/`ao1T`, so no
/// transpose happens anywhere on the J/K path.
#[derive(Debug, Clone, PartialEq)]
pub struct AoKpts {
    /// AO count.
    pub nao: usize,
    /// Grid-point count.
    pub ngrids: usize,
    /// `aot[k][mu * ngrids + g]`.
    pub aot: Vec<CTensor>,
}

impl AoKpts {
    /// The `(nao, ngrids)` block at k-point `k`.
    pub fn at(&self, k: usize) -> &CTensor {
        &self.aot[k]
    }
}

/// `FFTDF(cell, kpts)` — `fft.py:185-405`.
#[derive(Debug)]
pub struct Fftdf {
    /// The cell.
    pub cell: Cell,
    /// Sampling k-points. Empty means the single gamma point.
    pub kpts: Vec<[f64; 3]>,
    /// FFT mesh; defaults to `cell.mesh`. Assigning to it directly leaves
    /// [`Fftdf::grids`] and the AO cache describing the OLD mesh — use
    /// [`Fftdf::set_mesh`], which is upstream's `mydf.mesh = ...` (its `grids`
    /// is a property recomputed on every read).
    pub mesh: [usize; 3],
    /// The uniform quadrature grid on `mesh`.
    pub grids: UniformGrids,
    /// Memory budget in MB, used to size the `get_k_kpts` AO block and to cap
    /// the AO cache. Upstream's `mydf.max_memory`.
    pub max_memory: f64,
    /// Cached `(nao, ngrids)` AO tables, keyed by the k-point list.
    ao_cache: Mutex<HashMap<Vec<[u64; 3]>, Arc<AoKpts>>>,
    /// W-01: `get_coulG(dk)` and `expmikr(dk)`, keyed on the wrapped `dk =
    /// kpt2 - kpt1` (bit pattern), `omega` and the exxdiv actually applied
    /// INSIDE the k-pair loop (never `Ewald` — see [`Fftdf::coulg_and_expmikr`]).
    /// Both quantities are invariant across the whole SCF for a fixed
    /// `(dk, omega, exxdiv)`, and a Monkhorst-Pack mesh has only `Nk` distinct
    /// `dk` values (not `Nk^2`) because `kpts[i] - kpts[j]` is bit-identical
    /// for every pair sharing the same `(i - j) mod Nk`.
    coulg_expmikr_cache: Mutex<HashMap<CoulgKey, Arc<(Vec<f64>, Option<CTensor>)>>>,
}

/// `(dk.to_bits(), omega.to_bits(), exxdiv)` — see
/// [`Fftdf::coulg_expmikr_cache`].
type CoulgKey = ([u64; 3], Option<u64>, Option<ExxDiv>);

/// Upstream's `lib.param.MAX_MEMORY` default, in MB, overridable through
/// `PYSCF_MAX_MEMORY` (the same variable the molecular crates read).
fn default_max_memory() -> f64 {
    std::env::var("PYSCF_MAX_MEMORY")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(4000.0)
}

fn kpt_key(kpts: &[[f64; 3]]) -> Vec<[u64; 3]> {
    kpts.iter()
        .map(|k| [k[0].to_bits(), k[1].to_bits(), k[2].to_bits()])
        .collect()
}

impl Fftdf {
    /// Build an `FFTDF` for `cell` at `kpts` (empty = gamma), using
    /// `cell.mesh`.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] when the cell has no mesh, or the grid cannot be
    /// built.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Result<Self, PbcDfError> {
        let mesh = cell.try_mesh()?;
        Self::with_mesh(cell, kpts, mesh)
    }

    /// [`Fftdf::new`] with an explicit mesh — upstream's `mydf.mesh = ...`.
    ///
    /// # Errors
    /// As [`Fftdf::new`].
    pub fn with_mesh(cell: Cell, kpts: &[[f64; 3]], mesh: [usize; 3]) -> Result<Self, PbcDfError> {
        let grids = UniformGrids::build(&cell, Some(mesh))?;
        let kpts = if kpts.is_empty() {
            vec![[0.0; 3]]
        } else {
            kpts.to_vec()
        };
        Ok(Self {
            cell,
            kpts,
            mesh,
            grids,
            max_memory: default_max_memory(),
            ao_cache: Mutex::new(HashMap::new()),
            coulg_expmikr_cache: Mutex::new(HashMap::new()),
        })
    }

    /// `mydf.mesh = ...` — rebuild the grid and drop the AO cache.
    ///
    /// # Errors
    /// Propagates the grid construction.
    pub fn set_mesh(&mut self, mesh: [usize; 3]) -> Result<(), PbcDfError> {
        self.grids = UniformGrids::build(&self.cell, Some(mesh))?;
        self.mesh = mesh;
        self.reset();
        Ok(())
    }

    /// `ngrids = mesh.product()`.
    pub fn ngrids(&self) -> usize {
        self.grids.size()
    }

    /// The quadrature weight `vol / ngrids`.
    pub fn weight(&self) -> f64 {
        self.grids.weight()
    }

    /// The `(nao, ngrids)` AO table at `kpts`, from the cache when possible.
    ///
    /// # Errors
    /// Propagates [`eval_ao_kpts`].
    pub fn ao_kpts(&self, kpts: &[[f64; 3]]) -> Result<Arc<AoKpts>, PbcDfError> {
        let key = kpt_key(kpts);
        if let Ok(c) = self.ao_cache.lock() {
            if let Some(v) = c.get(&key) {
                return Ok(Arc::clone(v));
            }
        }
        let out = eval_ao_kpts(&self.cell, "GTOval_sph", &self.grids.coords, kpts)?;
        debug_assert_eq!(out.comp, 1, "the LDA AO path evaluates one component");
        let block = Arc::new(AoKpts {
            nao: out.nao,
            ngrids: out.ngrids,
            aot: out.kaos,
        });
        // 16 bytes per complex entry; keep the cache under a quarter of the
        // memory budget so the J/K scratch still fits.
        let bytes = 16.0 * (block.nao * block.ngrids * kpts.len()) as f64;
        if bytes < 0.25 * self.max_memory * 1e6 {
            if let Ok(mut c) = self.ao_cache.lock() {
                c.insert(key, Arc::clone(&block));
            }
        }
        Ok(block)
    }

    /// Drop the AO cache and the `get_coulG`/`expmikr` cache (W-01) — call
    /// after mutating `cell` or `mesh`.
    pub fn reset(&self) {
        if let Ok(mut c) = self.ao_cache.lock() {
            c.clear();
        }
        if let Ok(mut c) = self.coulg_expmikr_cache.lock() {
            c.clear();
        }
    }

    /// `get_coulG(dk)` and the phase table `expmikr(dk) = exp(-i dk.r)` on
    /// `self.grids.coords`, memoised on `(dk, omega, exxdiv)` — `fft_jk.py`'s
    /// `get_k_kpts` rebuilds both from scratch on every one of the `Nk^2`
    /// `(k1, k2)` pairs even though neither depends on the density matrix or
    /// on which pair produced this particular `dk` (W-01, §2.4 of
    /// KRKS-OPTIMISATION-PLAN.md). `exxdiv` here is the value ACTUALLY passed
    /// to `get_coulG` inside the pair loop — `fft_jk::get_k_kpts` already maps
    /// `Some(ExxDiv::Ewald) | None` to `None` before calling this (the Ewald
    /// probe-charge correction is applied once, after the loop, at `G+k = 0`),
    /// so `exxdiv` here is never `Some(ExxDiv::Ewald)`.
    ///
    /// # Errors
    /// Propagates [`get_coulg`].
    pub fn coulg_and_expmikr(
        &self,
        dk: [f64; 3],
        omega: Option<f64>,
        exxdiv: Option<ExxDiv>,
        kpts: &[[f64; 3]],
        gv: &[[f64; 3]],
    ) -> Result<Arc<(Vec<f64>, Option<CTensor>)>, PbcDfError> {
        let key: CoulgKey = (
            [dk[0].to_bits(), dk[1].to_bits(), dk[2].to_bits()],
            omega.map(f64::to_bits),
            exxdiv,
        );
        if let Ok(c) = self.coulg_expmikr_cache.lock() {
            if let Some(v) = c.get(&key) {
                return Ok(Arc::clone(v));
            }
        }
        let coulg = get_coulg(
            &self.cell,
            CoulGArgs {
                k: dk,
                exxdiv,
                kpts: Some(kpts),
                mesh: Some(self.mesh),
                gv: Some(gv),
                wrap_around: true,
                omega,
            },
        )?;
        let expmikr = if is_zero(&dk) {
            None
        } else {
            let ngrids = self.grids.coords.len();
            let mut re = vec![0.0_f64; ngrids];
            let mut im = vec![0.0_f64; ngrids];
            for (g, r) in self.grids.coords.iter().enumerate() {
                let ph = -(r[0] * dk[0] + r[1] * dk[1] + r[2] * dk[2]);
                re[g] = ph.cos();
                im[g] = ph.sin();
            }
            Some(CTensor::from_planes(re, im))
        };
        let entry = Arc::new((coulg, expmikr));
        if let Ok(mut c) = self.coulg_expmikr_cache.lock() {
            c.insert(key, Arc::clone(&entry));
        }
        Ok(entry)
    }

    pub(crate) fn client(&self) -> Result<AlgebraClient, PbcDfError> {
        Ok(select_backend()
            .map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "FFTDF: backend selection failed: {e}"
                )))
            })?
            .client)
    }

    /// Contract a REAL local potential on the grid into `nao x nao` matrices:
    /// `v[k][p, q] = sum_g conj(ao_k[p, g]) vR[g] ao_k[q, g]`.
    ///
    /// This is the `lib.dot(ao.T.conj() * vR, ao)` of `fft.py:71` and `:110`.
    /// There is NO quadrature weight: `ifft` already carries `1/ngrids` and the
    /// `1/vol` of the inverse Fourier transform cancels the `vol/ngrids` of the
    /// quadrature (see the module docs of `get_nuc`).
    fn contract_local_potential(&self, ao: &AoKpts, vr: &[f64], nkpts: usize) -> Vec<CTensor> {
        let (nao, ngrids) = (ao.nao, ao.ngrids);
        let mut out = Vec::with_capacity(nkpts);
        for k in 0..nkpts {
            let a = &ao.aot[k];
            let mut re = vec![0.0_f64; nao * nao];
            let mut im = vec![0.0_f64; nao * nao];
            for p in 0..nao {
                for q in 0..nao {
                    let mut sr = 0.0_f64;
                    let mut si = 0.0_f64;
                    let (pb, qb) = (p * ngrids, q * ngrids);
                    for g in 0..ngrids {
                        // conj(ao[p,g]) * ao[q,g] * vR[g]
                        let (pr, pi) = (a.re[pb + g], -a.im[pb + g]);
                        let (qr, qi) = (a.re[qb + g], a.im[qb + g]);
                        let w = vr[g];
                        sr += (pr * qr - pi * qi) * w;
                        si += (pr * qi + pi * qr) * w;
                    }
                    re[p * nao + q] = sr;
                    im[p * nao + q] = si;
                }
            }
            out.push(CTensor::from_planes(re, im));
        }
        out
    }
}

/// `get_nuc(mydf, kpts)` — `fft.py:40-80`.
///
/// ```text
/// rhoG  = -sum_a Z_a SI[a, G]          (nuclear charge density in G space)
/// vneG  = rhoG * coulG
/// vneR  = ifft(vneG).real
/// vne_k = sum_g conj(ao_k) vneR ao_k
/// ```
///
/// # Why there is no `vol/ngrids` factor
///
/// The real-space potential is `V(r) = (1/vol) sum_G vneG e^{iGr}` while `ifft`
/// computes `(1/ngrids) sum_G ...`, and the quadrature that follows carries
/// `vol/ngrids`. The two cancel exactly, which is why upstream's line 71 has no
/// weight in it. Adding one is the classic factor-of-`vol` bug here.
///
/// # Errors
/// Propagates the G-vector, structure-factor, `coulG` and AO evaluations.
pub fn get_nuc(df: &Fftdf, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let cell = &df.cell;
    let mesh = df.mesh;
    let gv = get_gv(cell, Some(mesh))?;
    let ngrids = gv.len();
    let si = get_si(cell, Some(&gv), None, None)?;
    let charges = cell.atom_charges();
    let natm = cell.mol.natm;

    // fft.py:60-63 — charge = -atom_charges; rhoG = charge . SI.
    let mut rho_re = vec![0.0_f64; ngrids];
    let mut rho_im = vec![0.0_f64; ngrids];
    for ia in 0..natm {
        let z = -(charges[ia] as f64);
        let base = ia * ngrids;
        for g in 0..ngrids {
            rho_re[g] += z * si.re[base + g];
            rho_im[g] += z * si.im[base + g];
        }
    }

    // fft.py:65-67
    let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
    for g in 0..ngrids {
        rho_re[g] *= coulg[g];
        rho_im[g] *= coulg[g];
    }
    let vner = ifft(&CTensor::from_planes(rho_re, rho_im), mesh)?.re;

    let ao = df.ao_kpts(kpts)?;
    Ok(df.contract_local_potential(&ao, &vner, kpts.len()))
}

/// `get_pp(mydf, kpts)` — `fft.py:82-178`, with the non-local half taken from
/// Phase 10's real-space route (see the module docs).
///
/// # Errors
/// Propagates the G-space local factors, the FFT, `get_pp_nl` and the AO
/// evaluation.
pub fn get_pp(df: &Fftdf, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let cell = &df.cell;
    let mesh = df.mesh;
    let gv = get_gv(cell, Some(mesh))?;
    let ngrids = gv.len();
    let si = get_si(cell, Some(&gv), None, None)?;
    let natm = cell.mol.natm;

    // fft.py:101-103 — vpplocG = -einsum('ij,ij->j', SI, get_vlocG(cell, Gv)).
    let vlocg = pyscf_pbc_gto::pseudo::get_vlocg(cell, &gv)?;
    let mut re = vec![0.0_f64; ngrids];
    let mut im = vec![0.0_f64; ngrids];
    for ia in 0..natm {
        let base = ia * ngrids;
        for g in 0..ngrids {
            re[g] -= si.re[base + g] * vlocg[base + g];
            im[g] -= si.im[base + g] * vlocg[base + g];
        }
    }

    // fft.py:106-112 — the local part, evaluated in real space.
    let vpplocr = ifft(&CTensor::from_planes(re, im), mesh)?.re;
    let ao = df.ao_kpts(kpts)?;
    let mut vpp = df.contract_local_potential(&ao, &vpplocr, kpts.len());

    // fft.py:114-176 — the non-local part. Phase 10 owns it in real space.
    let vnl = pyscf_pbc_gto::pseudo::get_pp_nl(cell, kpts)?;
    let nao = cell.mol.nao_nr;
    for (k, v) in vpp.iter_mut().enumerate() {
        // Phase-10 output is F-order (see `zlinalg::forder_to_c`).
        let nl = forder_to_c(&vnl[k], nao, nao);
        zadd_assign(v, &nl);
        // fft.py:172-175 — a gamma-point block is real by construction.
        if pyscf_pbc_gto::is_zero(&kpts[k]) {
            for t in v.im.iter_mut() {
                *t = 0.0;
            }
        }
    }
    Ok(vpp)
}

/// `get_hcore` for a periodic cell — `khf.py:66-90`.
///
/// `T + V_pp` for a pseudopotential cell, `T + V_ne` for an all-electron one.
/// This is the function `pyscf_pbc_gto::hcore::get_hcore` deferred to Phase 11.
///
/// # Errors
/// Propagates [`get_pp`] / [`get_nuc`] and `pbc_intor('int1e_kin')`.
///
/// Takes `&dyn PeriodicDf` since plan 13-07 (D-PBC-22): the body is builder
/// agnostic — it picks `get_pp` vs `get_nuc` from `cell.pseudo` and adds
/// `int1e_kin` — so binding it to `Fftdf` was the only thing stopping a driver
/// from running on AFTDF.
pub fn get_hcore(df: &dyn PeriodicDf, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let cell = df.cell();
    let nao = cell.mol.nao_nr;
    let mut nuc = if cell.pseudo.is_some() {
        df.get_pp(kpts)?
    } else {
        df.get_nuc(kpts)?
    };
    let t = pyscf_pbc_gto::get_t(cell, kpts)?;
    for (k, h) in nuc.iter_mut().enumerate() {
        zadd_assign(h, &forder_to_c(&t[k], nao, nao));
    }
    Ok(nuc)
}

impl PeriodicDf for Fftdf {
    fn cell(&self) -> &Cell {
        &self.cell
    }
    fn mesh(&self) -> [usize; 3] {
        self.mesh
    }
    fn name(&self) -> &'static str {
        "FFTDF"
    }
    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }
    fn build(&mut self) -> Result<(), PbcDfError> {
        let kpts = self.kpts.clone();
        self.ao_kpts(&kpts)?;
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
        let vj = if opts.with_j {
            Some(crate::fft_jk::get_j_kpts(
                self,
                dms,
                opts.hermi,
                kpts,
                opts.kpts_band,
                opts.omega,
            )?)
        } else {
            None
        };
        let vk = if opts.with_k {
            Some(crate::fft_jk::get_k_kpts_opts(
                self,
                dms,
                opts.hermi,
                kpts,
                opts.kpts_band,
                opts.exxdiv,
                opts.omega,
                opts.kk_symmetry,
            )?)
        } else {
            None
        };
        Ok(JkResult { vj, vk })
    }
}
