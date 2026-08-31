//! `KNumInt` — the periodic numerical-integration grid loop (plans 12-01, 12-02).
//!
//! Port of `pyscf/pbc/dft/numint.py`:
//!
//! | this file | upstream |
//! |---|---|
//! | [`KNumInt::block_ranges`] | `numint.py:1253-1310` |
//! | [`KNumInt::eval_rho`] | `numint.py:1150-1172` (k-average of `:96-186`) |
//! | [`KNumInt::nr_rks`] | `numint.py:284-386` |
//! | [`KNumInt::nr_uks`] | `numint.py:387-505` |
//! | `KNumInt::accumulate_vxc` / `vxc_mat_one` | `numint.py:1223-1240` + `:828-850` |
//! | [`KNumInt::get_rho`] | `numint.py:951-971` |
//! | [`KNumInt::cache_xc_kernel`] | `numint.py:852-900` |
//! | [`KNumInt::cache_xc_kernel1`] | `numint.py:901-950` |
//! | [`KNumInt::nr_rks_fxc`] | `numint.py:593-686` |
//! | [`KNumInt::nr_uks_fxc`] | `numint.py:719-827` |
//!
//! # The one formula
//!
//! ```text
//! rho(r)   = (1/N_k) Σ_k Σ_{μν} ao_k[r,μ] D^k[μν] conj(ao_k[r,ν])      (REAL)
//! V^k[μν]  = Σ_r conj(ao_k[r,μ]) (Σ_n wv[n][r] ao_k^{(n)}[r,ν])  + h.c.
//! ```
//!
//! `rho` is real by Hermiticity of `D^k`; the imaginary residue is a
//! convergence diagnostic, not a quantity ([`KNumInt::last_rho_imag`]).
//! There is NO `1/N_k` on `V^k` — the average lives in `rho` alone, exactly as
//! `numint.py:1168` puts it.
//!
//! # Bloch phases and derivatives
//!
//! `∇[e^{i k·L} φ(r−L)] = e^{i k·L} ∇φ(r−L)`: the phase is r-independent inside
//! a cell, so the GGA gradient block is the ORDINARY `deriv1` AO block summed
//! with the same phases. `pyscf_pbc_gto::eval_ao_kpts` already produces it —
//! there is no extra `i k` term anywhere in this file.
//!
//! # Layout
//!
//! AO blocks come from [`pyscf_pbc_gto::eval_ao_kpts`] in ITS layout,
//! `value[c * ngrids * nao + g + mu * ngrids]` (F-order per component). Density
//! matrices are ROW-MAJOR `nao x nao` [`CTensor`]s, the Phase-11
//! [`pyscf_pbc_scf::types::KMats`] convention, and so are the returned Vxc
//! matrices.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_pbc_gto::{Cell, EvalAoKptsOutput, eval_ao_kpts};
use pyscf_pbc_scf::types::{KDms, KMats};

use crate::error::PbcDftError;
use crate::gen_grid::PeriodicGrids;
use crate::xc::{
    FxcEff, RhoEff, VxcEff, XcType, err, eval_fxc_eff_rks, eval_fxc_eff_uks, eval_xc_eff_rks,
    eval_xc_eff_uks,
};

/// Upstream's `BLKSIZE` grid-block granularity (`dft/numint.py:44`).
pub const BLKSIZE: usize = 128;

/// Grid points one rayon worker owns where the split has to be over the GRID
/// rather than over an AO index (W-06 — [`eval_rho_one`]'s `_contract_rho`
/// stage, whose output IS indexed by the grid point).
///
/// One grid point per worker would be pure dispatch overhead; this is large
/// enough that a chunk is real work and small enough that a `mesh = 21` block
/// still spreads over every core.
const RHO_CHUNK: usize = 512;

/// `PYSCF_PBC_NUMINT_BLKSIZE`, read once — the W-07 grid-block override.
///
/// Rounded DOWN to a whole number of [`BLKSIZE`] blocks so the partition stays
/// on the same lattice the memory-derived default uses; a value below one
/// block, or an unparseable one, is ignored (and warned about) rather than
/// silently producing a one-point block.
fn numint_blksize_override() -> Option<usize> {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        let raw = std::env::var("PYSCF_PBC_NUMINT_BLKSIZE").ok()?;
        match raw.trim().parse::<usize>() {
            Ok(v) if v >= BLKSIZE => Some(v / BLKSIZE * BLKSIZE),
            _ => {
                tracing::warn!(
                    value = raw,
                    minimum = BLKSIZE,
                    "PYSCF_PBC_NUMINT_BLKSIZE is not an integer >= BLKSIZE; ignoring"
                );
                None
            }
        }
    })
}

/// Upstream's block-loop cap, `BLKSIZE * 2400` (`numint.py:1290`).
const MAX_BLOCK: usize = BLKSIZE * 2400;

/// `(nelec, excsum, vmat)` — what `nr_rks` returns.
#[derive(Debug, Clone)]
pub struct NrKResult {
    /// `∫ ρ` per density set.
    pub nelec: Vec<f64>,
    /// `E_xc` per density set.
    pub excsum: Vec<f64>,
    /// `vmat[iset][kband]`, `nao x nao` ROW-MAJOR.
    pub vmat: Vec<KMats>,
}

/// `(nelec[2], excsum, vmat[2])` — what `nr_uks` returns.
#[derive(Debug, Clone)]
pub struct NrKUksResult {
    /// `(∫ ρ_a, ∫ ρ_b)` per density set.
    pub nelec: Vec<(f64, f64)>,
    /// `E_xc` per density set.
    pub excsum: Vec<f64>,
    /// `vmat[spin][iset][kband]`.
    pub vmat: [Vec<KMats>; 2],
}

/// The 0th-order density plus the XC kernel — `cache_xc_kernel`'s return.
#[derive(Debug, Clone)]
pub struct XcKernelCache {
    /// `rho0`: one block for RKS (`spin = 0`), two for UKS (`spin = 1`).
    pub rho0: Vec<RhoEff>,
    /// The transformed first derivative on `rho0`.
    pub vxc: VxcEff,
    /// The transformed second derivative on `rho0`.
    pub fxc: FxcEff,
}

/// Cache key for an AO table: the k-points, the derivative order, the grid
/// size and a content hash of the coordinates.
type AoKey = (Vec<[u64; 3]>, u32, usize, u64);

/// `pbc/dft/numint.py:KNumInt` — the k-point numerical-integration driver.
///
/// Holds the sampling k-points and an AO cache, nothing else; the functional is
/// passed per call, exactly as upstream's `xc_code` argument is.
#[derive(Debug)]
pub struct KNumInt {
    /// Sampling k-points. Empty is normalised to the single gamma point.
    pub kpts: Vec<[f64; 3]>,
    /// Memory budget in MB — sizes the grid block and caps the AO cache.
    pub max_memory: f64,
    /// Largest `|Im ρ|` seen by the last [`KNumInt::eval_rho`]. Upstream drops
    /// the imaginary part silently (`numint.py:361`, `.real`); this port keeps
    /// the residue so a caller can assert on it.
    last_imag: std::cell::Cell<f64>,
    ao_cache: Mutex<HashMap<AoKey, Arc<EvalAoKptsOutput>>>,
}

impl KNumInt {
    /// A `KNumInt` over `kpts` (empty = gamma).
    pub fn new(kpts: &[[f64; 3]]) -> Self {
        let kpts = if kpts.is_empty() {
            vec![[0.0; 3]]
        } else {
            kpts.to_vec()
        };
        Self {
            kpts,
            max_memory: default_max_memory(),
            last_imag: std::cell::Cell::new(0.0),
            ao_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Number of sampling k-points.
    pub fn nkpts(&self) -> usize {
        self.kpts.len()
    }

    /// The imaginary residue of the last density evaluation.
    pub fn last_rho_imag(&self) -> f64 {
        self.last_imag.get()
    }

    /// Drop the AO cache — call after the cell or the grid changes.
    pub fn reset(&self) {
        if let Ok(mut c) = self.ao_cache.lock() {
            c.clear();
        }
    }

    // -----------------------------------------------------------------
    // AO evaluation
    // -----------------------------------------------------------------

    /// `eval_ao_kpts(cell, coords, kpts, deriv)` — `numint.py:70-93`, memoised.
    ///
    /// # Errors
    /// Propagates [`eval_ao_kpts`].
    pub fn eval_ao(
        &self,
        cell: &Cell,
        coords: &[[f64; 3]],
        kpts: &[[f64; 3]],
        ty: XcType,
    ) -> Result<Arc<EvalAoKptsOutput>, PbcDftError> {
        let key: AoKey = (
            kpts.iter()
                .map(|k| [k[0].to_bits(), k[1].to_bits(), k[2].to_bits()])
                .collect(),
            ty.ao_deriv(),
            coords.len(),
            coord_hash(coords),
        );
        if let Ok(c) = self.ao_cache.lock()
            && let Some(v) = c.get(&key)
        {
            return Ok(Arc::clone(v));
        }
        let out = Arc::new(eval_ao_kpts(cell, ty.eval_gto_name(), coords, kpts)?);
        // 16 bytes per complex entry; keep the table under a quarter of the
        // budget so the Vxc scratch still fits (the same rule `Fftdf` uses).
        let bytes = 16.0 * (out.comp * out.ngrids * out.nao * kpts.len()) as f64;
        if bytes < 0.25 * self.max_memory * 1e6
            && let Ok(mut c) = self.ao_cache.lock()
        {
            c.insert(key, Arc::clone(&out));
        }
        Ok(out)
    }

    /// The grid-block partition — `numint.py:1286-1291`.
    ///
    /// Returns the `[p0, p1)` half-open ranges the loop walks.
    pub fn block_ranges(&self, ngrids: usize, ty: XcType, nkpts: usize) -> Vec<(usize, usize)> {
        let comp = ty.ncomp();
        let denom = (comp * 2 * nkpts.max(1) * 16 * BLKSIZE) as f64;
        // `nao` is folded in by the caller through `max_memory`; upstream
        // divides by it here. Use a conservative unit so the block never
        // exceeds the cap.
        let raw = ((self.max_memory * 1e6 / denom) as usize) * BLKSIZE;
        // W-07: `PYSCF_PBC_NUMINT_BLKSIZE` overrides the memory-derived block,
        // rounded down to a whole number of `BLKSIZE` blocks. The DEFAULT is
        // unchanged — for a 4000 MB budget and a small cell it is still one
        // block covering the whole grid — so this adds a tuning knob without
        // moving a single energy. It is a knob and not a new default because a
        // different block partition changes `oracle_sum`'s input lengths, hence
        // the pairwise-tree shape, hence the last bits of `nelec`/`excsum`;
        // `nr_rks` mitigates that by summing per-block PARTIALS through
        // `oracle_sum` rather than with a running `+=`, but the partition still
        // shows. See `tests/numint_blocking.rs`.
        let raw = numint_blksize_override().unwrap_or(raw);
        let blksize = raw.clamp(BLKSIZE, MAX_BLOCK).min(ngrids.max(1));
        let mut out = Vec::new();
        let mut p0 = 0usize;
        while p0 < ngrids {
            let p1 = (p0 + blksize).min(ngrids);
            out.push((p0, p1));
            p0 = p1;
        }
        out
    }

    // -----------------------------------------------------------------
    // eval_rho
    // -----------------------------------------------------------------

    /// `KNumInt.eval_rho(cell, ao_kpts, dm_kpts, xctype, hermi=1)` —
    /// `numint.py:1150-1172`.
    ///
    /// The BZ average `ρ = (1/N_k) Σ_k ρ_k` over the block `ao` covers.
    /// Returns a real [`RhoEff`]; the imaginary residue lands in
    /// [`KNumInt::last_rho_imag`].
    ///
    /// # Only `hermi = 1`
    ///
    /// This is upstream's `hermi = 1` branch: the GGA rows get the `+ c.c.`
    /// factor 2 (`numint.py:141`) and the result is real. Upstream's `hermi = 0`
    /// branch builds a second `c1 = ao·D^H` contraction and returns a COMPLEX
    /// density, which every consumer downstream of here would have to carry.
    /// A caller with a non-Hermitian density must not route through this
    /// function; [`KNumInt::nr_rks_fxc`] and [`KNumInt::nr_uks_fxc`] refuse such
    /// input rather than silently returning the Hermitian answer.
    ///
    /// # Errors
    /// [`PbcDftError`] when `dms` and `ao` disagree in shape.
    // `k` indexes BOTH the AO table and the density-matrix list.
    #[allow(clippy::needless_range_loop)]
    pub fn eval_rho(
        &self,
        ao: &EvalAoKptsOutput,
        dms: &KMats,
        ty: XcType,
    ) -> Result<RhoEff, PbcDftError> {
        let nkpts = ao.nkpts();
        if dms.len() != nkpts {
            return Err(err(format!(
                "pbc eval_rho: {} density matrices for {nkpts} k-points",
                dms.len()
            )));
        }
        let ngrids = ao.ngrids;
        let nao = ao.nao;
        let mut rho = RhoEff::zeros(ty, ngrids);
        let mut imag = 0.0_f64;
        for k in 0..nkpts {
            let (block, im) = eval_rho_one(ao.at(k), &dms[k], ngrids, nao, ty)?;
            rho.add_assign(&block);
            imag = imag.max(im);
        }
        rho.scale(1.0 / nkpts as f64);
        self.last_imag
            .set(self.last_imag.get().max(imag / nkpts as f64));
        Ok(rho)
    }

    // -----------------------------------------------------------------
    // nr_rks / nr_uks
    // -----------------------------------------------------------------

    /// `nr_rks(ni, cell, grids, xc_code, dms, hermi, kpts, kpts_band)` —
    /// `numint.py:284-386`.
    ///
    /// `dms[iset][k]` is one closed-shell density-matrix set per k-point.
    ///
    /// # Errors
    /// Propagates the AO evaluation and the XC backend.
    pub fn nr_rks(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &KDms,
        hermi: i32,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<NrKResult, PbcDftError> {
        require_hermitian(hermi, "nr_rks")?;
        let ty = XcType::of(xc_code)?;
        let nset = dms.len();
        let coords = grids.coords()?;
        let weights = grids.weights()?;
        let ngrids = coords.len();
        let nao = cell.mol.nao_nr;
        let band = kpts_band.unwrap_or(&self.kpts);

        self.last_imag.set(0.0);
        // W-07: `nelec`/`excsum` used to be accumulated with a running `+=`
        // over the grid blocks — a naive sequential sum on the two quantities
        // that land straight in the total energy, which is exactly what
        // D-PBC-17 exists to forbid. Collect one partial per block and reduce
        // THOSE through `oracle_sum` instead. With the default single-block
        // partition this is bit-identical to the old code (a one-element
        // pairwise sum is the element); with several blocks it replaces a
        // sequential fold with the ordered tree.
        let mut nelec_parts: Vec<Vec<f64>> = vec![Vec::new(); nset];
        let mut excsum_parts: Vec<Vec<f64>> = vec![Vec::new(); nset];
        let mut vmat: Vec<KMats> =
            vec![vec![CTensor::zeros(nao * nao); band.len()]; nset];

        for (p0, p1) in self.block_ranges(ngrids, ty, self.nkpts()) {
            let chunk = &coords[p0..p1];
            let w = &weights[p0..p1];
            let ao2 = self.eval_ao(cell, chunk, &self.kpts, ty)?;
            let ao1 = if kpts_band.is_none() {
                Arc::clone(&ao2)
            } else {
                self.eval_ao(cell, chunk, band, ty)?
            };
            for i in 0..nset {
                let rho = self.eval_rho(&ao2, &dms[i], ty)?;
                let out = eval_xc_eff_rks(xc_code, &rho)?;
                // numint.py:363-368 — den = rho[0]*weight.
                let den: Vec<f64> = rho.row(0).iter().zip(w).map(|(r, wg)| r * wg).collect();
                nelec_parts[i].push(oracle_sum(&den));
                let terms: Vec<f64> =
                    den.iter().zip(&out.exc).map(|(d, e)| d * e).collect();
                excsum_parts[i].push(oracle_sum(&terms));
                // numint.py:369 — wv = weight * vxc.
                let wv = weighted(&out, 0, w);
                self.accumulate_vxc(&mut vmat[i], &ao1, &wv, ty);
            }
        }

        // numint.py:373-375 — vmat = vmat + vmat^H.
        for set in vmat.iter_mut() {
            for m in set.iter_mut() {
                add_conj_transpose(m, nao);
            }
        }
        let nelec: Vec<f64> = nelec_parts.iter().map(|p| oracle_sum(p)).collect();
        let excsum: Vec<f64> = excsum_parts.iter().map(|p| oracle_sum(p)).collect();
        Ok(NrKResult {
            nelec,
            excsum,
            vmat,
        })
    }

    /// `nr_uks(...)` — `numint.py:387-505`.
    ///
    /// `dms[0]` is the alpha channel, `dms[1]` the beta one; each is
    /// `[iset][k]`.
    ///
    /// # Errors
    /// As [`KNumInt::nr_rks`].
    pub fn nr_uks(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &[KDms; 2],
        hermi: i32,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<NrKUksResult, PbcDftError> {
        require_hermitian(hermi, "nr_uks")?;
        let ty = XcType::of(xc_code)?;
        let nset = dms[0].len();
        if dms[1].len() != nset {
            return Err(err("pbc nr_uks: alpha and beta carry different set counts"));
        }
        let coords = grids.coords()?;
        let weights = grids.weights()?;
        let ngrids = coords.len();
        let nao = cell.mol.nao_nr;
        let band = kpts_band.unwrap_or(&self.kpts);

        self.last_imag.set(0.0);
        let mut nelec = vec![(0.0_f64, 0.0_f64); nset];
        let mut excsum = vec![0.0_f64; nset];
        let mut vmat: [Vec<KMats>; 2] = [
            vec![vec![CTensor::zeros(nao * nao); band.len()]; nset],
            vec![vec![CTensor::zeros(nao * nao); band.len()]; nset],
        ];

        for (p0, p1) in self.block_ranges(ngrids, ty, self.nkpts()) {
            let chunk = &coords[p0..p1];
            let w = &weights[p0..p1];
            let ao2 = self.eval_ao(cell, chunk, &self.kpts, ty)?;
            let ao1 = if kpts_band.is_none() {
                Arc::clone(&ao2)
            } else {
                self.eval_ao(cell, chunk, band, ty)?
            };
            for i in 0..nset {
                let rho_a = self.eval_rho(&ao2, &dms[0][i], ty)?;
                let rho_b = self.eval_rho(&ao2, &dms[1][i], ty)?;
                let out = eval_xc_eff_uks(xc_code, &rho_a, &rho_b)?;
                let dena: Vec<f64> =
                    rho_a.row(0).iter().zip(w).map(|(r, wg)| r * wg).collect();
                let denb: Vec<f64> =
                    rho_b.row(0).iter().zip(w).map(|(r, wg)| r * wg).collect();
                nelec[i].0 += oracle_sum(&dena);
                nelec[i].1 += oracle_sum(&denb);
                let ta: Vec<f64> = dena.iter().zip(&out.exc).map(|(d, e)| d * e).collect();
                let tb: Vec<f64> = denb.iter().zip(&out.exc).map(|(d, e)| d * e).collect();
                excsum[i] += oracle_sum(&ta) + oracle_sum(&tb);
                for (s, vm) in vmat.iter_mut().enumerate() {
                    let wv = weighted(&out, s, w);
                    self.accumulate_vxc(&mut vm[i], &ao1, &wv, ty);
                }
            }
        }

        for vm in vmat.iter_mut() {
            for set in vm.iter_mut() {
                for m in set.iter_mut() {
                    add_conj_transpose(m, nao);
                }
            }
        }
        Ok(NrKUksResult {
            nelec,
            excsum,
            vmat,
        })
    }

    /// `KNumInt._vxc_mat` — `numint.py:1223-1240`, accumulated into `out`.
    ///
    /// `wv` is `[var][grid]` and ALREADY weight-scaled. With `hermi = 1` — the
    /// only mode the SCF drivers use — `wv[0]` carries the `*0.5` that pairs
    /// with the `V + V†` symmetrisation the caller applies.
    fn accumulate_vxc(
        &self,
        out: &mut KMats,
        ao: &EvalAoKptsOutput,
        wv: &[Vec<f64>],
        ty: XcType,
    ) {
        let nao = ao.nao;
        let ngrids = ao.ngrids;
        let nvar = ty.nvar();
        for (k, m) in out.iter_mut().enumerate() {
            vxc_mat_one(m, ao.at(k), wv, nao, ngrids, nvar);
        }
    }

    // -----------------------------------------------------------------
    // get_rho / cache_xc_kernel
    // -----------------------------------------------------------------

    /// `get_rho(ni, cell, dm, grids, kpts)` — `numint.py:951-971`.
    ///
    /// The real-space density on the grid, one value per grid point.
    ///
    /// # Errors
    /// As [`KNumInt::nr_rks`].
    pub fn get_rho(
        &self,
        cell: &Cell,
        dms: &KMats,
        grids: &PeriodicGrids,
    ) -> Result<Vec<f64>, PbcDftError> {
        let coords = grids.coords()?;
        let ngrids = coords.len();
        let mut rho = vec![0.0_f64; ngrids];
        for (p0, p1) in self.block_ranges(ngrids, XcType::Lda, self.nkpts()) {
            let ao = self.eval_ao(cell, &coords[p0..p1], &self.kpts, XcType::Lda)?;
            let block = self.eval_rho(&ao, dms, XcType::Lda)?;
            rho[p0..p1].copy_from_slice(block.row(0));
        }
        Ok(rho)
    }

    /// `cache_xc_kernel1(ni, cell, grids, xc_code, dm, spin, kpts)` —
    /// `numint.py:901-950`.
    ///
    /// `dms` carries one channel for `spin = 0` and two for `spin = 1`.
    ///
    /// # Errors
    /// As [`KNumInt::nr_rks`].
    pub fn cache_xc_kernel1(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &[KMats],
        spin: i32,
    ) -> Result<XcKernelCache, PbcDftError> {
        let ty = XcType::of(xc_code)?;
        let coords = grids.coords()?;
        let ngrids = coords.len();

        let nchan = if spin == 0 { 1 } else { 2 };
        if dms.len() < nchan {
            return Err(err(format!(
                "pbc cache_xc_kernel1: spin = {spin} needs {nchan} density channels, got {}",
                dms.len()
            )));
        }
        let mut rho: Vec<RhoEff> = (0..dms.len().min(nchan))
            .map(|_| RhoEff {
                nvar: ty.nvar(),
                ngrids: 0,
                data: Vec::new(),
            })
            .collect();
        for (p0, p1) in self.block_ranges(ngrids, ty, self.nkpts()) {
            let ao = self.eval_ao(cell, &coords[p0..p1], &self.kpts, ty)?;
            for (c, acc) in rho.iter_mut().enumerate() {
                let block = self.eval_rho(&ao, &dms[c], ty)?;
                acc.append(&block);
            }
        }

        // numint.py:934-936 — a closed-shell density asked for at spin = 1 is
        // halved and duplicated.
        if spin == 1 && rho.len() == 1 {
            let mut half = rho[0].clone();
            half.scale(0.5);
            rho = vec![half.clone(), half];
        }

        if spin == 0 {
            let vxc = eval_xc_eff_rks(xc_code, &rho[0])?;
            let fxc = eval_fxc_eff_rks(xc_code, &rho[0])?;
            Ok(XcKernelCache {
                rho0: rho,
                vxc,
                fxc,
            })
        } else {
            let vxc = eval_xc_eff_uks(xc_code, &rho[0], &rho[1])?;
            let fxc = eval_fxc_eff_uks(xc_code, &rho[0], &rho[1])?;
            Ok(XcKernelCache {
                rho0: rho,
                vxc,
                fxc,
            })
        }
    }

    /// `cache_xc_kernel(ni, cell, grids, xc_code, mo_coeff, mo_occ, spin, kpts)`
    /// — `numint.py:852-900`.
    ///
    /// Builds the 0th-order density from ORBITALS rather than from a density
    /// matrix. The two differ only in how `ρ0` is formed, so this assembles the
    /// density matrices and defers to [`KNumInt::cache_xc_kernel1`] — upstream
    /// takes the `eval_rho2` route for the same result.
    ///
    /// # Errors
    /// As [`KNumInt::cache_xc_kernel1`].
    pub fn cache_xc_kernel(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        mo_coeff: &[Vec<CTensor>],
        mo_occ: &[Vec<Vec<f64>>],
        spin: i32,
    ) -> Result<XcKernelCache, PbcDftError> {
        let nao = cell.mol.nao_nr;
        let dms: Vec<KMats> = mo_coeff
            .iter()
            .zip(mo_occ)
            .map(|(c, o)| pyscf_pbc_scf::krdm::make_rdm1(c, o, nao))
            .collect();
        self.cache_xc_kernel1(cell, grids, xc_code, &dms, spin)
    }

    // -----------------------------------------------------------------
    // fxc contraction
    // -----------------------------------------------------------------

    /// `nr_rks_fxc(ni, cell, grids, xc_code, dm0, dms, hermi, fxc, kpts)` —
    /// `numint.py:593-686`.
    ///
    /// Contracts the XC kernel with the RESPONSE density matrices `dms`.
    /// `fxc` is the cached kernel from [`KNumInt::cache_xc_kernel1`]; passing
    /// `None` recomputes it from `dm0`.
    ///
    /// The returned matrices are NOT symmetrised — upstream applies
    /// `v + v^H` only when `kpts` is gamma and the input is real
    /// (`numint.py:653-658`, `v_hermi`), and the response drivers that consume
    /// this expect the unsymmetrised form otherwise.
    ///
    /// # Errors
    /// As [`KNumInt::nr_rks`].
    #[allow(clippy::too_many_arguments)]
    pub fn nr_rks_fxc(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dm0: Option<&KMats>,
        dms: &KDms,
        hermi: i32,
        fxc: Option<&FxcEff>,
        v_hermi: bool,
    ) -> Result<Vec<KMats>, PbcDftError> {
        require_hermitian(hermi, "nr_rks_fxc")?;
        let ty = XcType::of(xc_code)?;
        let owned;
        let fxc = match fxc {
            Some(f) => f,
            None => {
                let d0 = dm0.ok_or_else(|| {
                    err("pbc nr_rks_fxc: neither a cached fxc nor a dm0 was supplied")
                })?;
                owned = self
                    .cache_xc_kernel1(cell, grids, xc_code, std::slice::from_ref(d0), 0)?
                    .fxc;
                &owned
            }
        };
        let coords = grids.coords()?;
        let weights = grids.weights()?;
        let ngrids = coords.len();
        let nao = cell.mol.nao_nr;
        let nset = dms.len();
        let nvar = ty.nvar();
        let mut vmat: Vec<KMats> =
            vec![vec![CTensor::zeros(nao * nao); self.nkpts()]; nset];

        for (p0, p1) in self.block_ranges(ngrids, ty, self.nkpts()) {
            let ao = self.eval_ao(cell, &coords[p0..p1], &self.kpts, ty)?;
            let w = &weights[p0..p1];
            let block = fxc.slice(p0, p1);
            for i in 0..nset {
                let rho1 = self.eval_rho(&ao, &dms[i], ty)?;
                // numint.py:667-670 — vxc1[y] = Σ_x rho1[x] fxc[x, y].
                let mut wv: Vec<Vec<f64>> = vec![vec![0.0; p1 - p0]; nvar];
                for (y, row) in wv.iter_mut().enumerate() {
                    for (g, item) in row.iter_mut().enumerate() {
                        let mut acc = 0.0_f64;
                        for x in 0..nvar {
                            acc += rho1.row(x)[g] * block.at(0, x, 0, y, g);
                        }
                        *item = acc * w[g];
                    }
                }
                if v_hermi {
                    for x in wv[0].iter_mut() {
                        *x *= 0.5;
                    }
                }
                self.accumulate_vxc(&mut vmat[i], &ao, &wv, ty);
            }
        }
        if v_hermi {
            for set in vmat.iter_mut() {
                for m in set.iter_mut() {
                    add_conj_transpose(m, nao);
                }
            }
        }
        Ok(vmat)
    }

    /// `nr_uks_fxc(...)` — `numint.py:719-827`.
    ///
    /// # Errors
    /// As [`KNumInt::nr_rks_fxc`].
    // `b` indexes the spin channel of BOTH `fxc` and `vmat`.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn nr_uks_fxc(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dm0: Option<&[KMats; 2]>,
        dms: &[KDms; 2],
        hermi: i32,
        fxc: Option<&FxcEff>,
        v_hermi: bool,
    ) -> Result<[Vec<KMats>; 2], PbcDftError> {
        require_hermitian(hermi, "nr_uks_fxc")?;
        let ty = XcType::of(xc_code)?;
        let owned;
        let fxc = match fxc {
            Some(f) => f,
            None => {
                let d0 = dm0.ok_or_else(|| {
                    err("pbc nr_uks_fxc: neither a cached fxc nor a dm0 was supplied")
                })?;
                owned = self
                    .cache_xc_kernel1(cell, grids, xc_code, &[d0[0].clone(), d0[1].clone()], 1)?
                    .fxc;
                &owned
            }
        };
        let coords = grids.coords()?;
        let weights = grids.weights()?;
        let ngrids = coords.len();
        let nao = cell.mol.nao_nr;
        let nset = dms[0].len();
        let nvar = ty.nvar();
        let mut vmat: [Vec<KMats>; 2] = [
            vec![vec![CTensor::zeros(nao * nao); self.nkpts()]; nset],
            vec![vec![CTensor::zeros(nao * nao); self.nkpts()]; nset],
        ];

        for (p0, p1) in self.block_ranges(ngrids, ty, self.nkpts()) {
            let ao = self.eval_ao(cell, &coords[p0..p1], &self.kpts, ty)?;
            let w = &weights[p0..p1];
            let block = fxc.slice(p0, p1);
            for i in 0..nset {
                let r1a = self.eval_rho(&ao, &dms[0][i], ty)?;
                let r1b = self.eval_rho(&ao, &dms[1][i], ty)?;
                for b in 0..2 {
                    // numint.py:806-809
                    let mut wv: Vec<Vec<f64>> = vec![vec![0.0; p1 - p0]; nvar];
                    for (y, row) in wv.iter_mut().enumerate() {
                        for (g, item) in row.iter_mut().enumerate() {
                            let mut acc = 0.0_f64;
                            for x in 0..nvar {
                                acc += r1a.row(x)[g] * block.at(0, x, b, y, g);
                                acc += r1b.row(x)[g] * block.at(1, x, b, y, g);
                            }
                            *item = acc * w[g];
                        }
                    }
                    if v_hermi {
                        for x in wv[0].iter_mut() {
                            *x *= 0.5;
                        }
                    }
                    self.accumulate_vxc(&mut vmat[b][i], &ao, &wv, ty);
                }
            }
        }
        if v_hermi {
            for vm in vmat.iter_mut() {
                for set in vm.iter_mut() {
                    for m in set.iter_mut() {
                        add_conj_transpose(m, nao);
                    }
                }
            }
        }
        Ok(vmat)
    }
}

// ---------------------------------------------------------------------------
// free helpers
// ---------------------------------------------------------------------------

/// `wv = weight * vxc[spin]`, with the `hermi = 1` half-factor on row 0
/// (`numint.py:1234-1237`).
fn weighted(out: &VxcEff, spin: usize, w: &[f64]) -> Vec<Vec<f64>> {
    (0..out.nvar)
        .map(|v| {
            let row = out.row(spin, v);
            let scale = if v == 0 { 0.5 } else { 1.0 };
            row.iter().zip(w).map(|(x, wg)| x * wg * scale).collect()
        })
        .collect()
}

/// `m += m^H` in place (`numint.py:374`).
fn add_conj_transpose(m: &mut CTensor, nao: usize) {
    let re = m.re.clone();
    let im = m.im.clone();
    for i in 0..nao {
        for j in 0..nao {
            m.re[i * nao + j] = re[i * nao + j] + re[j * nao + i];
            m.im[i * nao + j] = im[i * nao + j] - im[j * nao + i];
        }
    }
}

/// One k-point's `_vxc_mat`, accumulated into `out` — `numint.py:828-850`.
///
/// ```text
/// aow[g, ν] = Σ_{n<nvar} wv[n][g] · ao^{(n)}[g, ν]
/// out[μ, ν] += Σ_g conj(ao^{(0)}[g, μ]) · aow[g, ν]
/// ```
// `n` indexes the AO component AND the matching `wv` row.
#[allow(clippy::needless_range_loop)]
fn vxc_mat_one(
    out: &mut CTensor,
    ao: &CTensor,
    wv: &[Vec<f64>],
    nao: usize,
    ngrids: usize,
    nvar: usize,
) {
    if ngrids == 0 {
        return;
    }
    // aow, in the same F-order-per-component layout as `ao`'s component 0.
    //
    // W-06: `nu` indexes DISJOINT output rows of `aow`, so it is the axis split
    // across workers; the component sum over `n` stays serial and ascending
    // inside each row, which is what makes this bit-identical to the pre-W-06
    // `n`-outer nest. The `if s == 0.0 { continue; }` skip is kept deliberately
    // — see the module note on it.
    let mut aow_re = vec![0.0_f64; ngrids * nao];
    let mut aow_im = vec![0.0_f64; ngrids * nao];
    aow_re
        .par_chunks_mut(ngrids)
        .zip(aow_im.par_chunks_mut(ngrids))
        .enumerate()
        .for_each(|(nu, (wre, wim))| {
            let b = nu * ngrids;
            for n in 0..nvar {
                let base = n * ngrids * nao;
                let wvn = &wv[n];
                for g in 0..ngrids {
                    let s = wvn[g];
                    if s == 0.0 {
                        continue;
                    }
                    wre[g] += s * ao.re[base + b + g];
                    wim[g] += s * ao.im[base + b + g];
                }
            }
        });
    // W-06: one worker per output ROW `mu` of `out`. `oracle_sum`'s pairwise
    // tree shape depends only on `ngrids` and the fixed `PAIRWISE_CHUNK`, never
    // on which thread evaluates it, so D-PBC-17's thread-count invariance is
    // preserved exactly. The `terms` scratch becomes per-worker; it used to be
    // one buffer reused across `(mu, nu)`.
    out.re
        .par_chunks_mut(nao)
        .zip(out.im.par_chunks_mut(nao))
        .enumerate()
        .for_each(|(mu, (orow, oirow))| {
            let mut terms_re = vec![0.0_f64; ngrids];
            let mut terms_im = vec![0.0_f64; ngrids];
            let mb = mu * ngrids;
            for nu in 0..nao {
                let nb = nu * ngrids;
                for g in 0..ngrids {
                    let (ar, ai) = (ao.re[mb + g], -ao.im[mb + g]);
                    let (br, bi) = (aow_re[nb + g], aow_im[nb + g]);
                    terms_re[g] = ar * br - ai * bi;
                    terms_im[g] = ar * bi + ai * br;
                }
                orow[nu] += oracle_sum(&terms_re);
                oirow[nu] += oracle_sum(&terms_im);
            }
        });
}

/// `eval_rho(cell, ao, dm, xctype, hermi=1)` at ONE k-point — `numint.py:96-186`.
///
/// Returns the real density block and the largest imaginary residue.
fn eval_rho_one(
    ao: &CTensor,
    dm: &CTensor,
    ngrids: usize,
    nao: usize,
    ty: XcType,
) -> Result<(RhoEff, f64), PbcDftError> {
    let want = ty.ncomp() * ngrids * nao;
    if ao.len() != want {
        return Err(err(format!(
            "pbc eval_rho: AO block has {} entries, expected {want}",
            ao.len()
        )));
    }
    if dm.len() != nao * nao {
        return Err(err(format!(
            "pbc eval_rho: density matrix has {} entries, expected {}",
            dm.len(),
            nao * nao
        )));
    }
    // c0[g, j] = Σ_i ao0[g, i] dm[i, j]   (`_dot_ao_dm`)
    //
    // W-06: `j` indexes disjoint output rows of `c0`; the reduction over `i`
    // stays serial and ascending inside each of them, so the same terms reach
    // each `c0[j, g]` in the same order as the pre-W-06 `i`-outer nest.
    let mut c0_re = vec![0.0_f64; ngrids * nao];
    let mut c0_im = vec![0.0_f64; ngrids * nao];
    c0_re
        .par_chunks_mut(ngrids)
        .zip(c0_im.par_chunks_mut(ngrids))
        .enumerate()
        .for_each(|(j, (crow, cirow))| {
            for i in 0..nao {
                let (dr, di) = (dm.re[i * nao + j], dm.im[i * nao + j]);
                if dr == 0.0 && di == 0.0 {
                    continue;
                }
                let ib = i * ngrids;
                for g in 0..ngrids {
                    let (ar, ai) = (ao.re[ib + g], ao.im[ib + g]);
                    crow[g] += ar * dr - ai * di;
                    cirow[g] += ar * di + ai * dr;
                }
            }
        });

    let mut rho = RhoEff::zeros(ty, ngrids);
    let mut imag = 0.0_f64;
    // rho[c] = Σ_j conj(ao_c[g, j]) c0[g, j]   (`_contract_rho`)
    let ncomp = ty.ncomp();
    for c in 0..ncomp {
        let base = c * ngrids * nao;
        // W-06: `g` is the OUTPUT index here and `j` is the reduction axis, so
        // the split is over disjoint grid chunks with `j` serial and ascending
        // inside each — the pre-W-06 order, term for term.
        let mut acc_re = vec![0.0_f64; ngrids];
        let mut acc_im = vec![0.0_f64; ngrids];
        acc_re
            .par_chunks_mut(RHO_CHUNK)
            .zip(acc_im.par_chunks_mut(RHO_CHUNK))
            .enumerate()
            .for_each(|(c, (are, aim))| {
                let g0 = c * RHO_CHUNK;
                for j in 0..nao {
                    let jb = j * ngrids;
                    for t in 0..are.len() {
                        let g = g0 + t;
                        let (ar, ai) = (ao.re[base + jb + g], -ao.im[base + jb + g]);
                        let (br, bi) = (c0_re[jb + g], c0_im[jb + g]);
                        are[t] += ar * br - ai * bi;
                        aim[t] += ar * bi + ai * br;
                    }
                }
            });
        // `hermi = 1` — the gradient rows carry the `+ c.c.` factor 2
        // (`numint.py:141`).
        let scale = if c == 0 { 1.0 } else { 2.0 };
        let row = rho.row_mut(c);
        for g in 0..ngrids {
            row[g] = scale * acc_re[g];
        }
        for v in &acc_im {
            imag = imag.max(v.abs());
        }
    }
    Ok((rho, imag))
}

/// Reject a non-Hermitian input density rather than silently applying the
/// `hermi = 1` shortcut. See the note on [`KNumInt::eval_rho`].
fn require_hermitian(hermi: i32, who: &str) -> Result<(), PbcDftError> {
    if hermi == 1 {
        return Ok(());
    }
    Err(err(format!(
        "pbc {who}: hermi = {hermi}. The periodic NumInt implements upstream's \
         hermi = 1 branch only; a non-Hermitian density needs the complex \
         `eval_rho` of numint.py:118-121 and a complex fxc contraction with it."
    )))
}

/// Upstream's `lib.param.MAX_MEMORY`, overridable through `PYSCF_MAX_MEMORY`.
fn default_max_memory() -> f64 {
    std::env::var("PYSCF_MAX_MEMORY")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(4000.0)
}

/// FNV-1a over the raw coordinate bits — the AO cache's grid identity.
fn coord_hash(coords: &[[f64; 3]]) -> u64 {
    // W-07 (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`): this used to be
    // byte-at-a-time FNV-1a — EIGHT rounds of xor/multiply/shift per f64, i.e.
    // `24 * ngrids` rounds on every single `eval_ao` lookup, purely to decide a
    // cache hit. At the gate mesh that is 715 000 rounds per call, a visible
    // share of a warm `nr_rks`.
    //
    // The plan's own suggestion was to key on a grid GENERATION COUNTER instead.
    // That is not available here: `eval_ao` takes a bare `&[[f64; 3]]` slice
    // with no stable identity — keying on its address would hand a stale AO
    // table to a caller whose grid was freed and whose replacement landed at the
    // same address with the same length, which is a wrong-answer bug, not a
    // cache miss. So the key stays a full hash of every coordinate bit (same
    // collision semantics as before — nothing is sampled or skipped) and only
    // the mixing gets cheaper: two multiplies per 64-bit WORD instead of eight
    // rounds per byte.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in coords {
        for x in c {
            // Multiply by an odd constant (invertible, so no information is
            // lost), then rotate before folding in — the rotate is what carries
            // the high-bit avalanche down into the low bits that a following
            // multiply would otherwise leave under-mixed.
            let z = x.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
            h = (h ^ z).rotate_left(27).wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}
