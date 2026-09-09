//! K-point-resolved multigrid — `pyscf/pbc/dft/multigrid/multigrid_pair.py`'s
//! Bloch-phase half, which 17-11/17-12 deliberately left at the gamma point
//! (`17-VERIFICATION.md` §10.12: "multigrid v1/v2 are gamma-only").
//!
//! # The whole k-point generalisation is a host-side change of variable
//!
//! A fused pair term in [`crate::multigrid::pair`] is one primitive pair
//! `(p at A, q at B + L)` at one lattice image `L`, and every Cartesian slot
//! feeding it shares that `L` ([`PairLevelTable::term_img`]). The
//! collocation kernel only ever sees `term_coef[t]` — a real number per term
//! — and gives back `I[t]`, a real grid integral per term. **Neither knows
//! what a k-point is.** So the k-resolved density and potential are the
//! gamma-point machinery with two different real vectors:
//!
//! * forward, `multigrid.py`'s `_dm_translation`:
//!   `D_R[μν, L] = Σ_k w_k D_k[μν] e^{-i k·L}` — real for a Hermitian,
//!   translation-complete k-set, and the number the slot contraction feeds
//!   into `term_coef` in place of the gamma point's `D[μν]`.
//! * reverse: `V_k[μν] = Σ_L I[μν, L] e^{+i k·L}` — the same integrals
//!   scattered with the conjugate phase.
//!
//! **No kernel changes, no new launches, and the gamma point is bit-exact**:
//! at `nkpts = 1`, `k = 0`, the phase is `cos = 1` / `sin = 0` and `w = 1`, so
//! `d = re·1.0 + im·0.0` reproduces the scalar the gamma path multiplies,
//! exactly. `tests/multigrid_kpts.rs` holds that to `to_bits()` equality
//! rather than to a tolerance.
//!
//! # Why the phase sign is `e^{-i k·L}` forward and `e^{+i k·L}` reverse
//!
//! This port's periodic AO convention is upstream's:
//! `S_k[μν] = Σ_L e^{+i k·L} ⟨φ_μ(r) | φ_ν(r - L)⟩`, i.e. the phase belongs
//! to the image of the **column** function — which is exactly the `bshift =
//! q.center + l` the pair table is built on. Every one-electron matrix in
//! that basis therefore carries `e^{+i k·L}`, and the density matrix
//! conjugate to it carries `e^{-i k·L}`.
//!
//! The forward direction cannot detect a sign error on its own — the pair
//! list runs over ALL ordered `(pi, pj)` and its image list is closed under
//! `L -> -L`, so `(μ, ν, L)` and `(ν, μ, -L)` are both present and their
//! contributions are complex conjugates whose sum is the same real number
//! under either convention. The REVERSE direction can: the wrong sign
//! returns `V_{-k}` for `V_k`. That is what
//! `tests/multigrid_kpts.rs::vj_matches_fftdf_per_k` is
//! for, and why it compares per k-point rather than comparing an energy.
//!
//! # `kpts_band` falls out, and with it k-point symmetry
//!
//! `I[t]` is a real grid integral that knows nothing about k. Evaluating the
//! potential at a DIFFERENT k-list from the density is therefore free — one
//! more phase table — which is the whole of `kpts_band` support, and
//! `kpts_band` support is the whole of what the k-symmetric drivers need:
//! `krks_ksymm` hands `nr_rks` a full-BZ density and asks for an IBZ-length
//! potential (`krks_ksymm.rs:21-27`). The gamma-only v2 driver refused both.
//!
//! # Cost
//!
//! The collocation — the entire GPU cost — is over IMAGES, not k-points, so
//! it does not grow with `nkpts` at all. What grows is the host contraction,
//! `O(nslots · nkpts)` reals in each direction, against the
//! `O(ninstances · ngrids)` exponentials the kernel evaluates. This is the
//! structural reason multigrid is the right engine for a dense k-mesh, and
//! the reason k-point symmetry buys multigrid far less than it buys the
//! reference `KNumInt` (whose AO evaluation IS per k).

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::Cell;

use crate::error::PbcDftError;
use crate::multigrid::pair::{
    MultiGridNumInt2, PairLevelTable, each_level, insert_level_rho_g, level_v_r, pair_level_span,
    pairlevel_integrals, pairlevel_rho_from_terms, pairlevel_rho2_from_terms,
};
use crate::multigrid::tasks::Decontracted;

/// One k-point set's Bloch phases against one level's lattice images:
/// `cos[k * nimg + i]` / `sin[k * nimg + i]` for `k · L_i`.
///
/// Built per level per call. It is `nkpts · nimages` reals — a few thousand
/// for any cell this engine is used on — against the level's millions of
/// slots, so it is never worth caching across calls and never worth
/// threading through the geometry cache (which is keyed on the CELL, and
/// would then have to be invalidated by a k-mesh change).
#[derive(Debug)]
pub struct PhaseTable {
    nimg: usize,
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl PhaseTable {
    /// `k · L` over every `(k, image)` pair of one level.
    pub fn new(kpts: &[[f64; 3]], images: &[f64]) -> Self {
        let nimg = images.len() / 3;
        let mut cos = Vec::with_capacity(kpts.len() * nimg);
        let mut sin = Vec::with_capacity(kpts.len() * nimg);
        for k in kpts {
            for i in 0..nimg {
                let kl = k[0] * images[i * 3] + k[1] * images[i * 3 + 1] + k[2] * images[i * 3 + 2];
                // The gamma point must be EXACT, not merely accurate: it is
                // what makes the k-general route bit-identical to the
                // gamma-only route it replaces. `f64::cos(0.0)` and
                // `f64::sin(0.0)` do return 1.0 and 0.0, but they do so as a
                // libm promise rather than as this module's, and a `-0.0`
                // image coordinate is enough to make the promise interesting.
                if kl == 0.0 {
                    cos.push(1.0);
                    sin.push(0.0);
                } else {
                    cos.push(kl.cos());
                    sin.push(kl.sin());
                }
            }
        }
        Self { nimg, cos, sin }
    }

    #[inline]
    fn at(&self, k: usize, img: u32) -> (f64, f64) {
        let i = k * self.nimg + img as usize;
        (self.cos[i], self.sin[i])
    }
}

/// A density-matrix stack decontracted onto the primitive Cartesian basis,
/// real and imaginary planes separately — `E · D_k · E^T` per k-point.
///
/// `E` is real ([`Decontracted::expand`]), so the two planes decontract
/// independently through the SAME already-gated
/// [`crate::multigrid::colloc::expand_dm`]; nothing here re-derives it.
#[derive(Debug)]
pub struct KDmP {
    /// Per k-point, `nao_p x nao_p` row-major.
    pub re: Vec<Vec<f64>>,
    pub im: Vec<Vec<f64>>,
}

impl KDmP {
    /// Decontract every k-point's density matrix.
    pub fn expand(decon: &Decontracted, dms: &[CTensor]) -> Self {
        let re = dms
            .iter()
            .map(|d| crate::multigrid::colloc::expand_dm(decon, &d.re))
            .collect();
        let im = dms
            .iter()
            .map(|d| crate::multigrid::colloc::expand_dm(decon, &d.im))
            .collect();
        Self { re, im }
    }

    pub fn nkpts(&self) -> usize {
        self.re.len()
    }
}

/// `D_R[μν, L]` contracted into this level's fused terms — the k-resolved
/// replacement for [`crate::multigrid::pair::pairlevel_rho_with`]'s
/// gamma-point slot contraction.
///
/// `term_coef[t] = Σ_{s→t} coef_s · Σ_k w_k [ Re D_k(ci_s,cj_s) cos(k·L_t)
/// + Im D_k(ci_s,cj_s) sin(k·L_t) ]`.
///
/// Slot order is the table's, k order is the caller's, and both are fixed —
/// the D-PBC-17 property the gamma path already has, unchanged.
fn term_coef_kpts(
    lv: &PairLevelTable,
    decon: &Decontracted,
    dm: &KDmP,
    wk: &[f64],
    ph: &PhaseTable,
) -> Vec<f64> {
    let mut term_coef = vec![0.0f64; lv.nterms];
    for s in 0..lv.nslots() {
        let idx = lv.slot_ci[s] as usize * decon.nao_p + lv.slot_cj[s] as usize;
        let img = lv.term_img[lv.slot_term[s] as usize];
        let mut d = 0.0f64;
        // Fixed k order — the caller's, which is the sampling list's — so the
        // partial sum is the same however the level's slots are visited
        // (D-PBC-17, applied to the axis this contraction added).
        for (k, &w) in wk.iter().enumerate() {
            let (c, sn) = ph.at(k, img);
            d += w * (dm.re[k][idx] * c + dm.im[k][idx] * sn);
        }
        term_coef[lv.slot_term[s] as usize] += d * lv.slot_coef[s];
    }
    term_coef
}

/// One decontracted potential matrix per band k-point, real and imaginary
/// planes apart — what the reverse sweep accumulates into before
/// [`contract_v_kpts`] folds it back onto the contracted basis.
type KVp = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// One level's real-space density from a k-resolved density-matrix stack.
///
/// # Errors
/// As [`crate::multigrid::pair::pairlevel_rho_from_terms`].
pub fn pairlevel_rho_kpts(
    lv: &PairLevelTable,
    decon: &Decontracted,
    dm: &KDmP,
    wk: &[f64],
    ph: &PhaseTable,
) -> Result<Vec<f64>, PbcDftError> {
    if lv.nkslots() == 0 || lv.ngrids == 0 {
        return Ok(vec![0.0f64; lv.ngrids]);
    }
    let term_coef = term_coef_kpts(lv, decon, dm, wk, ph);
    pairlevel_rho_from_terms(lv, &term_coef, true)
}

/// One level's contribution to the k-resolved decontracted potential:
/// `V_k[ci,cj] += coef_s · I_t · e^{+i k·L_t}`. ADDS, does not overwrite.
///
/// # Errors
/// As [`crate::multigrid::pair::pairlevel_integrals`].
pub fn pairlevel_pass2_kpts(
    lv: &PairLevelTable,
    decon: &Decontracted,
    weight: &[f64],
    ph: &PhaseTable,
    v_re: &mut [Vec<f64>],
    v_im: &mut [Vec<f64>],
) -> Result<(), PbcDftError> {
    if lv.nkslots() == 0 || lv.ngrids == 0 {
        return Ok(());
    }
    let integrals = pairlevel_integrals(lv, weight, true)?;
    let nk = v_re.len();
    for s in 0..lv.nslots() {
        let t = lv.slot_term[s] as usize;
        let c = lv.slot_coef[s] * integrals[t];
        if c == 0.0 {
            continue;
        }
        let idx = lv.slot_ci[s] as usize * decon.nao_p + lv.slot_cj[s] as usize;
        let img = lv.term_img[t];
        for k in 0..nk {
            let (co, sn) = ph.at(k, img);
            v_re[k][idx] += c * co;
            v_im[k][idx] += c * sn;
        }
    }
    Ok(())
}

/// Both spin channels' k-resolved densities at one level, through ONE
/// geometry traversal — the k-general twin of
/// [`crate::multigrid::pair::pairlevel_rho2`].
///
/// # Errors
/// As [`crate::multigrid::pair::pairlevel_rho2_from_terms`].
pub fn pairlevel_rho2_kpts(
    lv: &PairLevelTable,
    decon: &Decontracted,
    dm: [&KDmP; 2],
    wk: &[f64],
    ph: &PhaseTable,
) -> Result<[Vec<f64>; 2], PbcDftError> {
    if lv.nkslots() == 0 || lv.ngrids == 0 {
        return Ok([vec![0.0f64; lv.ngrids], vec![0.0f64; lv.ngrids]]);
    }
    let a = term_coef_kpts(lv, decon, dm[0], wk, ph);
    let b = term_coef_kpts(lv, decon, dm[1], wk, ph);
    pairlevel_rho2_from_terms(lv, [&a, &b])
}

/// Every level's k-resolved real-space density, combined into `rho(G)` on
/// `cell.mesh` — the k-general twin of `rho_g_from_pair_levels`.
///
/// # Errors
/// Propagates the collocation and the FFT.
pub fn rho_g_from_pair_levels_kpts(
    cell: &Cell,
    decon: &Decontracted,
    tables: &[Option<PairLevelTable>],
    dm: &KDmP,
    wk: &[f64],
    kpts: &[[f64; 3]],
) -> Result<CTensor, PbcDftError> {
    let mesh = cell.mesh;
    let ngrids_full = mesh[0] * mesh[1] * mesh[2];
    let mut rho_g = CTensor::zeros(ngrids_full);
    let vol = cell.vol();
    for (level, lv) in each_level(tables) {
        let _level_span = pair_level_span("forward", level, 1, lv);
        let ph = PhaseTable::new(kpts, &lv.images);
        let rho_r = pairlevel_rho_kpts(lv, decon, dm, wk, &ph)?;
        insert_level_rho_g(rho_r, lv, level, vol, mesh, &mut rho_g)?;
    }
    Ok(rho_g)
}

/// A G-space weight field contracted back into one decontracted potential
/// matrix PER BAND k-POINT — the k-general twin of `pass2_from_full_vg_pair`.
///
/// # Errors
/// Propagates the FFT and the reverse collocation.
pub fn pass2_from_full_vg_pair_kpts(
    cell: &Cell,
    decon: &Decontracted,
    tables: &[Option<PairLevelTable>],
    vg_full: &CTensor,
    kpts_band: &[[f64; 3]],
) -> Result<KVp, PbcDftError> {
    let mesh = cell.mesh;
    let n = decon.nao_p * decon.nao_p;
    let nk = kpts_band.len();
    let mut v_re = vec![vec![0.0f64; n]; nk];
    let mut v_im = vec![vec![0.0f64; n]; nk];
    for (level, lv) in each_level(tables) {
        let _level_span = pair_level_span("reverse", level, 1, lv);
        let v_r = level_v_r(vg_full, mesh, lv, level)?;
        let ph = PhaseTable::new(kpts_band, &lv.images);
        pairlevel_pass2_kpts(lv, decon, &v_r.re, &ph, &mut v_re, &mut v_im)?;
    }
    Ok((v_re, v_im))
}

/// [`MultiGridNumInt2::nr_rks_kpts`]'s return — [`Mg2NrRksResult`]'s
/// k-resolved twin.
///
/// [`Mg2NrRksResult`]: crate::multigrid::pair::Mg2NrRksResult
#[derive(Debug, Clone)]
pub struct Mg2NrRksKResult {
    pub nelec: f64,
    pub exc: f64,
    pub ecoul: f64,
    /// Per band k-point, `nao x nao` row-major complex.
    pub veff: Vec<CTensor>,
}

/// [`MultiGridNumInt2::nr_uks_kpts`]'s return.
#[derive(Debug, Clone)]
pub struct Mg2NrUksKResult {
    pub nelec: (f64, f64),
    pub exc: f64,
    pub ecoul: f64,
    /// `[alpha, beta]`, each one `CTensor` per band k-point.
    pub veff: [Vec<CTensor>; 2],
}

/// Uniform Monkhorst-Pack weights, `1/nkpts` — what `KNumInt::eval_rho`'s
/// `rho.scale(1.0 / nkpts)` applies and therefore what the multigrid route
/// must apply to agree with it.
fn uniform_weights(nkpts: usize) -> Vec<f64> {
    vec![1.0 / nkpts as f64; nkpts]
}

impl MultiGridNumInt2 {
    /// `get_nuc(mydf, kpts)` — the k-general twin of
    /// [`MultiGridNumInt2::get_nuc`], delegated as that one is.
    ///
    /// # Errors
    /// Propagates [`crate::multigrid::pp::get_nuc_kpts`].
    pub fn get_nuc_kpts(
        &self,
        cell: &Cell,
        kpts: &[[f64; 3]],
    ) -> Result<Vec<CTensor>, PbcDftError> {
        crate::multigrid::pp::get_nuc_kpts(cell, kpts)
    }

    /// `get_pp(mydf, kpts)` — the k-general twin of
    /// [`MultiGridNumInt2::get_pp`].
    ///
    /// # Errors
    /// Propagates [`crate::multigrid::pp::get_pp_kpts`].
    pub fn get_pp_kpts(&self, cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDftError> {
        crate::multigrid::pp::get_pp_kpts(cell, kpts)
    }

    /// `rho(G)` on `cell.mesh` from a k-resolved density-matrix stack.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT construction.
    pub fn eval_rho_g_kpts(
        &self,
        cell: &Cell,
        dms: &[CTensor],
        kpts: &[[f64; 3]],
    ) -> Result<CTensor, PbcDftError> {
        let prep = self.tasks(cell)?;
        let (decon, tables) = (&prep.0, &prep.1);
        let dm = KDmP::expand(decon, dms);
        let wk = uniform_weights(kpts.len());
        rho_g_from_pair_levels_kpts(cell, decon, tables, &dm, &wk, kpts)
    }

    /// `get_j_kpts(mydf, dm_kpts, kpts, kpts_band)` — the Coulomb matrix at
    /// every band k-point from a k-resolved density.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT construction.
    pub fn get_j_kpts(
        &self,
        cell: &Cell,
        dms: &[CTensor],
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<Vec<CTensor>, PbcDftError> {
        let prep = self.tasks(cell)?;
        let (decon, tables) = (&prep.0, &prep.1);
        let dm = KDmP::expand(decon, dms);
        let wk = uniform_weights(kpts.len());
        let rho_g = rho_g_from_pair_levels_kpts(cell, decon, tables, &dm, &wk, kpts)?;

        let mesh = cell.mesh;
        let gv = pyscf_pbc_gto::get_gv(cell, Some(mesh))?;
        let coulg = pyscf_pbc_gto::get_coulg_at_gv(cell, mesh, &gv)?;
        let mut vg = rho_g;
        for ((re, im), c) in vg.re.iter_mut().zip(vg.im.iter_mut()).zip(&coulg) {
            *re *= c;
            *im *= c;
        }
        let band = kpts_band.unwrap_or(kpts);
        let (v_re, v_im) = pass2_from_full_vg_pair_kpts(cell, decon, tables, &vg, band)?;
        Ok(contract_v_kpts(decon, &v_re, &v_im))
    }

    /// `nr_rks(mydf, xc_code, dm_kpts, kpts, kpts_band)` with `with_j = True`
    /// — the k-resolved v2 driver.
    ///
    /// `dms` is the density at `kpts` (the FULL sampling set, already
    /// unfolded by a k-symmetric caller); `kpts_band`, when given, is the
    /// k-list the potential is returned at — `krks_ksymm`'s IBZ subset.
    ///
    /// # Errors
    /// Propagates task-list / collocation / FFT / XC evaluation.
    pub fn nr_rks_kpts(
        &self,
        cell: &Cell,
        xc_code: &str,
        dms: &[CTensor],
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<Mg2NrRksKResult, PbcDftError> {
        let prep = self.tasks(cell)?;
        let (decon, tables) = (&prep.0, &prep.1);
        let dm = KDmP::expand(decon, dms);
        let wk = uniform_weights(kpts.len());
        let rho_g = rho_g_from_pair_levels_kpts(cell, decon, tables, &dm, &wk, kpts)?;

        // M-00: the middle — Coulomb, XC, the GGA G-space fold, the ordered
        // energy reductions — is the SAME shared function the gamma path
        // uses. `rho(G)` is one scalar field however many k-points made it.
        let parts =
            crate::multigrid::numint::mg_xc_parts(cell, xc_code, std::slice::from_ref(&rho_g))?;
        let band = kpts_band.unwrap_or(kpts);
        let (v_re, v_im) =
            pass2_from_full_vg_pair_kpts(cell, decon, tables, &parts.wv_freq0[0], band)?;

        Ok(Mg2NrRksKResult {
            nelec: parts.nelec[0],
            exc: parts.exc,
            ecoul: parts.ecoul,
            veff: contract_v_kpts(decon, &v_re, &v_im),
        })
    }

    /// [`MultiGridNumInt2::nr_rks_kpts`]'s open-shell twin.
    ///
    /// # Errors
    /// As [`MultiGridNumInt2::nr_rks_kpts`].
    pub fn nr_uks_kpts(
        &self,
        cell: &Cell,
        xc_code: &str,
        dms: [&[CTensor]; 2],
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<Mg2NrUksKResult, PbcDftError> {
        let prep = self.tasks(cell)?;
        let (decon, tables) = (&prep.0, &prep.1);
        let wk = uniform_weights(kpts.len());
        // ONE geometry traversal for both channels, level by level — M-17's
        // two-channel forward kernel, which a per-channel loop would waste.
        let dm = [KDmP::expand(decon, dms[0]), KDmP::expand(decon, dms[1])];
        let mesh = cell.mesh;
        let ngrids: usize = mesh.iter().product();
        let mut rho_g = [CTensor::zeros(ngrids), CTensor::zeros(ngrids)];
        let vol = cell.vol();
        for (level, lv) in each_level(tables) {
            let _level_span = pair_level_span("forward", level, 2, lv);
            let ph = PhaseTable::new(kpts, &lv.images);
            let mut rho_r = pairlevel_rho2_kpts(lv, decon, [&dm[0], &dm[1]], &wk, &ph)?;
            for (spin, target) in rho_g.iter_mut().enumerate() {
                let r = core::mem::take(&mut rho_r[spin]);
                insert_level_rho_g(r, lv, level, vol, mesh, target)?;
            }
        }
        let rho_g = rho_g.to_vec();

        let parts = crate::multigrid::numint::mg_xc_parts(cell, xc_code, &rho_g)?;
        let band = kpts_band.unwrap_or(kpts);
        let mut veff = Vec::with_capacity(2);
        for wv in &parts.wv_freq0 {
            let (v_re, v_im) = pass2_from_full_vg_pair_kpts(cell, decon, tables, wv, band)?;
            veff.push(contract_v_kpts(decon, &v_re, &v_im));
        }
        let vb = veff.pop().expect("mg_xc_parts returns two spin channels");
        let va = veff.pop().expect("mg_xc_parts returns two spin channels");

        Ok(Mg2NrUksKResult {
            nelec: (parts.nelec[0], parts.nelec[1]),
            exc: parts.exc,
            ecoul: parts.ecoul,
            veff: [va, vb],
        })
    }
}

/// Contract every band k-point's decontracted potential back onto the
/// contracted AO basis — `E^T · V_p · E` on each plane, through the same
/// real [`crate::multigrid::colloc::contract_v`] the gamma path uses.
fn contract_v_kpts(decon: &Decontracted, v_re: &[Vec<f64>], v_im: &[Vec<f64>]) -> Vec<CTensor> {
    v_re.iter()
        .zip(v_im)
        .map(|(re, im)| {
            CTensor::from_planes(
                crate::multigrid::colloc::contract_v(decon, re),
                crate::multigrid::colloc::contract_v(decon, im),
            )
        })
        .collect()
}
