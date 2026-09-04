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
use pyscf_pbc_symm::kpts::KPoints;

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

/// Numerical-integration implementation selected by a periodic KS driver.
/// The ordinary grid path remains the default; both multigrid implementations
/// are explicit, gamma-only alternatives.
#[derive(Debug)]
pub enum KsNumInt {
    Grid(KNumInt),
    MultiGrid(crate::multigrid::MultiGridNumInt),
    MultiGrid2(crate::multigrid::MultiGridNumInt2),
}

/// Spin-restricted result at the common KS numerical-integration seam.
#[derive(Debug, Clone)]
pub struct KsNrRksResult {
    pub nelec: f64,
    pub exc: f64,
    pub vmat: KDms,
    /// Present when multigrid fused Coulomb into `vmat`.
    pub ecoul: Option<f64>,
}

/// Spin-unrestricted result at the common KS numerical-integration seam.
#[derive(Debug, Clone)]
pub struct KsNrUksResult {
    pub nelec: (f64, f64),
    pub exc: f64,
    pub vmat: [KMats; 2],
    /// Present when multigrid fused Coulomb into both spin potentials.
    pub ecoul: Option<f64>,
}

impl KsNumInt {
    pub fn grid(kpts: &[[f64; 3]]) -> Self {
        Self::Grid(KNumInt::new(kpts))
    }

    pub fn multigrid() -> Self {
        Self::MultiGrid(crate::multigrid::MultiGridNumInt::new())
    }

    pub fn multigrid2() -> Self {
        Self::MultiGrid2(crate::multigrid::MultiGridNumInt2::new())
    }

    pub fn is_multigrid(&self) -> bool {
        !matches!(self, Self::Grid(_))
    }

    pub fn reset(&self) {
        match self {
            Self::Grid(ni) => ni.reset(),
            Self::MultiGrid(ni) => ni.reset(),
            Self::MultiGrid2(ni) => ni.reset(),
        }
    }

    fn require_multigrid_inputs(
        grids: &PeriodicGrids,
        xc_code: &str,
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<(), PbcDftError> {
        if kpts.len() != 1 || kpts[0].iter().any(|&x| x != 0.0) {
            return Err(PbcDftError::MultiGridRequiresGamma { nkpts: kpts.len() });
        }
        if kpts_band.is_some() {
            return Err(PbcDftError::MultiGridBandUnsupported);
        }
        if !matches!(grids, PeriodicGrids::Uniform(_)) {
            return Err(PbcDftError::MultiGridRequiresUniformGrid);
        }
        if crate::xc::is_hybrid_xc(xc_code)? {
            return Err(PbcDftError::MultiGridHybridUnsupported(xc_code.to_string()));
        }
        Ok(())
    }

    pub fn nr_rks(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &KDms,
        hermi: i32,
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<KsNrRksResult, PbcDftError> {
        match self {
            Self::Grid(ni) => {
                let out = ni.nr_rks(cell, grids, xc_code, dms, hermi, kpts_band)?;
                Ok(KsNrRksResult {
                    nelec: out.nelec[0],
                    exc: out.excsum[0],
                    vmat: out.vmat,
                    ecoul: None,
                })
            }
            Self::MultiGrid(ni) => {
                Self::require_multigrid_inputs(grids, xc_code, kpts, kpts_band)?;
                let out = ni.nr_rks(cell, xc_code, &dms[0][0].re)?;
                Ok(Self::wrap_mg_rks(
                    cell, out.nelec, out.exc, out.ecoul, out.veff,
                ))
            }
            Self::MultiGrid2(ni) => {
                Self::require_multigrid_inputs(grids, xc_code, kpts, kpts_band)?;
                let out = ni.nr_rks(cell, xc_code, &dms[0][0].re)?;
                Ok(Self::wrap_mg_rks(
                    cell, out.nelec, out.exc, out.ecoul, out.veff,
                ))
            }
        }
    }

    /// The multigrid `nr_rks` return, in `KsNrRksResult` shape — the gamma
    /// point's single real `veff` promoted to the k-resolved `vmat` the KS
    /// seam expects. Shared by the v1 and v2 arms, whose results differ only
    /// in the type that carries them.
    fn wrap_mg_rks(
        cell: &Cell,
        nelec: f64,
        exc: f64,
        ecoul: f64,
        veff: Vec<f64>,
    ) -> KsNrRksResult {
        let n = cell.mol.nao_nr * cell.mol.nao_nr;
        KsNrRksResult {
            nelec,
            exc,
            vmat: vec![vec![CTensor::from_planes(veff, vec![0.0; n])]],
            ecoul: Some(ecoul),
        }
    }

    /// [`Self::wrap_mg_rks`]'s open-shell twin.
    fn wrap_mg_uks(
        cell: &Cell,
        nelec: (f64, f64),
        exc: f64,
        ecoul: f64,
        veff: [Vec<f64>; 2],
    ) -> KsNrUksResult {
        let n = cell.mol.nao_nr * cell.mol.nao_nr;
        let [va, vb] = veff;
        KsNrUksResult {
            nelec,
            exc,
            vmat: [
                vec![CTensor::from_planes(va, vec![0.0; n])],
                vec![CTensor::from_planes(vb, vec![0.0; n])],
            ],
            ecoul: Some(ecoul),
        }
    }

    pub fn nr_uks(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &[KDms; 2],
        hermi: i32,
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<KsNrUksResult, PbcDftError> {
        match self {
            Self::Grid(ni) => {
                let mut out = ni.nr_uks(cell, grids, xc_code, dms, hermi, kpts_band)?;
                Ok(KsNrUksResult {
                    nelec: out.nelec[0],
                    exc: out.excsum[0],
                    vmat: [out.vmat[0].swap_remove(0), out.vmat[1].swap_remove(0)],
                    ecoul: None,
                })
            }
            Self::MultiGrid(ni) => {
                Self::require_multigrid_inputs(grids, xc_code, kpts, kpts_band)?;
                let out = ni.nr_uks(cell, xc_code, &[&dms[0][0][0].re, &dms[1][0][0].re])?;
                Ok(Self::wrap_mg_uks(
                    cell, out.nelec, out.exc, out.ecoul, out.veff,
                ))
            }
            Self::MultiGrid2(ni) => {
                Self::require_multigrid_inputs(grids, xc_code, kpts, kpts_band)?;
                let out = ni.nr_uks(cell, xc_code, &[&dms[0][0][0].re, &dms[1][0][0].re])?;
                Ok(Self::wrap_mg_uks(
                    cell, out.nelec, out.exc, out.ecoul, out.veff,
                ))
            }
        }
    }

    pub fn get_rho(
        &self,
        cell: &Cell,
        dms: &KMats,
        grids: &PeriodicGrids,
    ) -> Result<Vec<f64>, PbcDftError> {
        match self {
            Self::Grid(ni) => ni.get_rho(cell, dms, grids),
            Self::MultiGrid(_) | Self::MultiGrid2(_) => {
                Err(err("get_rho is not exposed by the multigrid KS seam"))
            }
        }
    }
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
type AoKey = (u8, Vec<[u64; 3]>, u32, usize, u64);

fn symmetry_rho_enabled() -> bool {
    std::env::var("PYSCF_PBC_KSYMM_RHO").is_ok_and(|v| v.eq_ignore_ascii_case("symmetrize"))
}

/// Which k-set [`KNumInt`] integrates over — plan 17-08 Task 1.
///
/// Upstream branches on `isinstance(kpts, KPoints)` at **seven** sites
/// (`pbc/dft/numint.py:328, 431, 647, 779, 859, 908, 956`). They do **two**
/// different things, and neither is a symmetrization:
///
/// * **Five** (`:328, :431, :859, :908, :956`) unfold to the **full BZ** —
///   `dms = kpts.transform_dm(dms)` (or `transform_mo_coeff`/`transform_mo_occ`
///   at `:859`), then `kpts = kpts.kpts` — and run the ordinary full-BZ path
///   unchanged.
/// * **Two** (`:647` `nr_rks_fxc`, `:779` `nr_uks_fxc`) take
///   `kpts = kpts.kpts_ibz` directly, with no transform.
///
/// # 17-08-PLAN.md Task 1's premise was wrong — see `17-08-FINDING-numint.md`
///
/// The plan asserted that all seven "evaluate the density at the IBZ points,
/// then symmetrize the real-space density through `kpts.symmetrize_density`".
/// None of them do, and `symmetrize_density` has **no caller anywhere in
/// `pyscf/pbc/` except its own unit test**. Implementing the plan literally
/// also fights this file's block loop, because `symmetrize_density` rotates
/// grid *indices* across the whole mesh while the density is built per
/// block — a wall upstream never hits, because upstream never does this.
///
/// The consequence is worth knowing: under symmetry, `numint` does the
/// full-BZ amount of work **plus** an unfold. It is a convenience interface,
/// not an optimisation; the IBZ saving in a ksymm DFT run comes from the SCF
/// side (D-PBC-26), not the XC quadrature.
///
/// # Why a field rather than the plan's threaded parameter
///
/// The plan asked for `KSet<'a>` as a parameter on every builder. As a field
/// defaulted to [`KSet::Full`] by [`KNumInt::new`], the `Full` path is not
/// merely unedited — both arms reach the **same** code, since `Ibz` unfolds
/// and then joins it. The required bit-identity therefore holds by
/// construction rather than by assertion. The plan's real requirement, "one
/// enum, one match, not a `KPoints` field bolted onto each function", is met.
#[derive(Debug, Default)]
pub enum KSet {
    /// Every sampling k-point; `KNumInt::kpts` is the full BZ. The
    /// pre-17-08 behaviour, unchanged.
    #[default]
    Full,
    /// Built from a `KPoints`. `KNumInt::kpts` is still the **full BZ**
    /// (upstream's Group-A `kpts = kpts.kpts`); the folding is reached through
    /// [`KNumInt::unfold_dms`] and [`KNumInt::kpts_ibz`].
    Ibz(Box<KPoints>),
}

/// `pbc/dft/numint.py:KNumInt` — the k-point numerical-integration driver.
///
/// Holds the sampling k-points and an AO cache, nothing else; the functional is
/// passed per call, exactly as upstream's `xc_code` argument is.
#[derive(Debug)]
pub struct KNumInt {
    /// Sampling k-points. Empty is normalised to the single gamma point.
    ///
    /// Under [`KSet::Ibz`] these are the **IBZ** points, not the full BZ.
    pub kpts: Vec<[f64; 3]>,
    /// Full BZ, or an IBZ set plus the symmetry needed to unfold it.
    pub kset: KSet,
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
            kset: KSet::Full,
            max_memory: default_max_memory(),
            last_imag: std::cell::Cell::new(0.0),
            ao_cache: Mutex::new(HashMap::new()),
        }
    }

    /// A `KNumInt` over an IBZ k-set — plan 17-08 Task 1.
    ///
    /// **`kpts` here is the FULL BZ**, matching upstream's five Group-A sites,
    /// which set `kpts = kpts.kpts` after unfolding the density. The IBZ
    /// points are reached through [`KNumInt::kpts_ibz`] for the two Group-B
    /// (`*_fxc`) sites that want them. See `KSet`'s doc and
    /// `17-08-FINDING-numint.md`.
    pub fn with_symmetry(kpts: &KPoints) -> Self {
        let mut ni = Self::new(&kpts.kpts);
        ni.kset = KSet::Ibz(Box::new(kpts.clone()));
        ni
    }

    /// The `KPoints` when built over a symmetric k-set, else `None`.
    pub fn ksymm(&self) -> Option<&KPoints> {
        match &self.kset {
            KSet::Full => None,
            KSet::Ibz(k) => Some(k),
        }
    }

    /// The IBZ k-points — upstream's Group-B branches (`numint.py:647` in
    /// `nr_rks_fxc`, `:779` in `nr_uks_fxc`), which use `kpts.kpts_ibz`
    /// directly with no transform.
    ///
    /// Falls back to the full k-set under [`KSet::Full`], so a caller written
    /// against it needs no branch.
    pub fn kpts_ibz(&self) -> &[[f64; 3]] {
        match &self.kset {
            KSet::Full => &self.kpts,
            KSet::Ibz(k) => &k.kpts_ibz,
        }
    }

    /// [`KNumInt::unfold_dms`] over every density-matrix set — the `KDms`
    /// shape `nr_rks` / `nr_uks` / `nr_*_fxc` carry.
    ///
    /// # Errors
    /// As [`KNumInt::unfold_dms`].
    ///
    /// # U-06 — returns a [`Cow`], and the borrow is the common case
    ///
    /// Under [`KSet::Full`] — every non-symmetric driver, i.e. the default —
    /// this used to `clone()` the entire `(nset, nkpts, nao^2)` density stack
    /// on EVERY `nr_rks`/`nr_uks` call, purely to hand back an owned value that
    /// the symmetric branch needs and the common branch does not. For the
    /// `MESH_GATE` reference cell that is `2 x 8 x 64` complex entries per SCF
    /// cycle in KUKS, twice what KRKS pays, and it is pure allocate-and-copy.
    /// `Cow::Borrowed` removes it with no change to any number.
    pub fn unfold_kdms<'a>(
        &self,
        cell: &Cell,
        dms: &'a KDms,
        nao: usize,
    ) -> Result<std::borrow::Cow<'a, KDms>, PbcDftError> {
        let Some(kpts) = self.ksymm() else {
            return Ok(std::borrow::Cow::Borrowed(dms));
        };
        // S-01: already full-BZ — upstream's `if kpts.kpts.size > 3` guard,
        // which `unfold_dms` also carries per set. Borrowing here rather than
        // letting `unfold_dms` clone matters because S-01 makes the ksymm
        // drivers unfold ONCE and then hand the full-BZ stack to `nr_rks` /
        // `nr_uks`, whose own unfold must therefore cost nothing at all rather
        // than a full k-stack copy. A needless round trip would also perturb
        // the density, which is why upstream guards it too.
        if dms.iter().all(|s| s.len() == kpts.nkpts()) {
            return Ok(std::borrow::Cow::Borrowed(dms));
        }
        let out: KDms = dms
            .iter()
            .map(|s| self.unfold_dms(cell, s, nao))
            .collect::<Result<_, _>>()?;
        Ok(std::borrow::Cow::Owned(out))
    }

    /// Unfold IBZ-length MO coefficients and occupations to the full BZ —
    /// upstream's `numint.py:859-863`, the one Group-A site that transforms
    /// the orbitals rather than the density.
    ///
    /// Faithful to upstream (RULE 2) rather than the equivalent
    /// "build the DM, then unfold it": `make_rdm1 ∘ transform_mo_coeff` and
    /// `transform_dm ∘ make_rdm1` agree mathematically, but upstream
    /// transforms the orbitals here and this port follows it, so a future
    /// reader diffing the two files finds the same call.
    ///
    /// # Errors
    /// Propagates [`KPoints::transform_mo_coeff`] / [`KPoints::transform_mo_occ`].
    #[allow(clippy::type_complexity)]
    pub fn unfold_mos(
        &self,
        cell: &Cell,
        mo_coeff: &[Vec<CTensor>],
        mo_occ: &[Vec<Vec<f64>>],
        nao: usize,
    ) -> Result<(Vec<Vec<CTensor>>, Vec<Vec<Vec<f64>>>), PbcDftError> {
        let Some(kpts) = self.ksymm() else {
            return Ok((mo_coeff.to_vec(), mo_occ.to_vec()));
        };
        let mut cs = Vec::with_capacity(mo_coeff.len());
        let mut os = Vec::with_capacity(mo_occ.len());
        for (c, o) in mo_coeff.iter().zip(mo_occ.iter()) {
            if c.len() == kpts.nkpts() {
                cs.push(c.clone());
                os.push(o.clone());
                continue;
            }
            // `mo_coeff[k]` is `nao x nmo`, so `nmo` comes from its length.
            let nmo = c.first().map_or(0, |t| t.re.len() / nao.max(1));
            let c_c: Vec<Vec<num_complex::Complex64>> = c
                .iter()
                .map(|t| {
                    t.re.iter()
                        .zip(t.im.iter())
                        .map(|(&re, &im)| num_complex::Complex64::new(re, im))
                        .collect()
                })
                .collect();
            let bz = kpts
                .transform_mo_coeff(cell, &c_c, nao, nmo)
                .map_err(|e| err(&format!("pbc NumInt: transform_mo_coeff: {e}")))?;
            cs.push(
                bz.iter()
                    .map(|m| CTensor {
                        re: m.iter().map(|z| z.re).collect(),
                        im: m.iter().map(|z| z.im).collect(),
                    })
                    .collect(),
            );
            os.push(
                kpts.transform_mo_occ(o)
                    .map_err(|e| err(&format!("pbc NumInt: transform_mo_occ: {e}")))?,
            );
        }
        Ok((cs, os))
    }

    /// Unfold IBZ-length density matrices to the full BZ — upstream's
    /// Group-A branches (`dms = kpts.transform_dm(dms)`, `numint.py:330` and
    /// four siblings).
    ///
    /// A no-op under [`KSet::Full`], and a no-op under [`KSet::Ibz`] when the
    /// input is already full-BZ length, so callers need no branch of their
    /// own. Upstream guards the same way with `if kpts.kpts.size > 3`.
    ///
    /// # Errors
    /// Propagates [`KPoints::transform_dm`] (17-05 Task 3, gated at 1e-12).
    pub fn unfold_dms(&self, cell: &Cell, dms: &KMats, nao: usize) -> Result<KMats, PbcDftError> {
        let Some(kpts) = self.ksymm() else {
            return Ok(dms.clone());
        };
        if dms.len() == kpts.nkpts() {
            return Ok(dms.clone());
        }
        let as_c: Vec<Vec<num_complex::Complex64>> = dms
            .iter()
            .map(|t| {
                t.re.iter()
                    .zip(t.im.iter())
                    .map(|(&re, &im)| num_complex::Complex64::new(re, im))
                    .collect()
            })
            .collect();
        let bz = kpts
            .transform_dm(cell, &as_c, nao)
            .map_err(|e| err(&format!("pbc NumInt: transform_dm: {e}")))?;
        Ok(bz
            .iter()
            .map(|m| CTensor {
                re: m.iter().map(|c| c.re).collect(),
                im: m.iter().map(|c| c.im).collect(),
            })
            .collect())
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
        self.eval_ao_route(cell, coords, kpts, ty, 0)
    }

    fn eval_ao_route(
        &self,
        cell: &Cell,
        coords: &[[f64; 3]],
        kpts: &[[f64; 3]],
        ty: XcType,
        route: u8,
    ) -> Result<Arc<EvalAoKptsOutput>, PbcDftError> {
        let key: AoKey = (
            route,
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
        self.eval_rho_into(ao, dms, ty, &mut Scratch::default())
    }

    /// [`KNumInt::eval_rho`] against a caller-owned scratch — U-10.
    ///
    /// The public entry point above allocates one [`Scratch`] per call, which
    /// already removes the per-k-point allocation it used to do. `nr_rks` and
    /// `nr_uks` take this variant instead and hold ONE scratch for the whole
    /// grid-block loop, so a KUKS cycle allocates these buffers once rather
    /// than `nblocks * nset * nkpts * 2` times.
    ///
    /// # Errors
    /// As [`KNumInt::eval_rho`].
    pub(crate) fn eval_rho_into(
        &self,
        ao: &EvalAoKptsOutput,
        dms: &KMats,
        ty: XcType,
        sc: &mut Scratch,
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
            let (block, im) = eval_rho_one(ao.at(k), &dms[k], ngrids, nao, ty, sc)?;
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
        if symmetry_rho_enabled() && self.ksymm().is_some() {
            return self.nr_rks_symmetrized(cell, grids, xc_code, dms, ty, kpts_band);
        }
        // Group A (`numint.py:328-331`): a symmetric k-set unfolds the density
        // to the full BZ and then runs this path unchanged. No-op under
        // `KSet::Full`. See `KSet`'s doc / `17-08-FINDING-numint.md`.
        let dms = &*self.unfold_kdms(cell, dms, cell.mol.nao_nr)?;
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
        let mut vmat: Vec<KMats> = vec![vec![CTensor::zeros(nao * nao); band.len()]; nset];

        // U-10: ONE scratch for the whole grid-block loop instead of a fresh
        // `vec![0.0; ..]` inside `eval_rho_one` / `vxc_mat_one` per k-point per
        // set per block. Bit-exact — see `Scratch`.
        let mut sc = Scratch::default();
        let mut den: Vec<f64> = Vec::new();
        let mut terms: Vec<f64> = Vec::new();

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
                let rho = self.eval_rho_into(&ao2, &dms[i], ty, &mut sc)?;
                let out = eval_xc_eff_rks(xc_code, &rho)?;
                // numint.py:363-368 — den = rho[0]*weight.
                //
                // U-10: `clear` + `extend` reuses the allocation and produces
                // the identical values in the identical order.
                den.clear();
                den.extend(rho.row(0).iter().zip(w).map(|(r, wg)| r * wg));
                nelec_parts[i].push(oracle_sum(&den));
                terms.clear();
                terms.extend(den.iter().zip(&out.exc).map(|(d, e)| d * e));
                excsum_parts[i].push(oracle_sum(&terms));
                // numint.py:369 — wv = weight * vxc.
                let wv = weighted(&out, 0, w);
                self.accumulate_vxc_into(&mut vmat[i], &ao1, &wv, ty, &mut sc);
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
        if symmetry_rho_enabled() && self.ksymm().is_some() {
            return self.nr_uks_symmetrized(cell, grids, xc_code, dms, ty, kpts_band);
        }
        // Group A (`numint.py:431-435`), per spin channel.
        let dms = &[
            self.unfold_kdms(cell, &dms[0], cell.mol.nao_nr)?,
            self.unfold_kdms(cell, &dms[1], cell.mol.nao_nr)?,
        ];
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
        // W-07 / U-03 step 3: `nelec` and `excsum` used to be accumulated with
        // a running `+=` over the grid blocks — a naive sequential sum on the
        // two quantities that land straight in the total energy, which is what
        // D-PBC-17 forbids. `nr_rks` was converted by W-07 and `nr_uks` was
        // not; this is the open-shell half. Collect one partial per block and
        // reduce THOSE through `oracle_sum`.
        //
        // It also makes `excsum`/`nelec` BIT-STABLE across `max_memory`: the
        // tree shape now depends only on the block COUNT through a single
        // ordered reduction rather than on the fold order of a running sum.
        // (`block_ranges` derives `blksize` from `max_memory`, `ty.ncomp()` and
        // `nkpts` — never from `nset` — so KRKS and KUKS partition identically.)
        let mut nelec_a_parts: Vec<Vec<f64>> = vec![Vec::new(); nset];
        let mut nelec_b_parts: Vec<Vec<f64>> = vec![Vec::new(); nset];
        let mut excsum_parts: Vec<Vec<f64>> = vec![Vec::new(); nset];
        let mut vmat: [Vec<KMats>; 2] = [
            vec![vec![CTensor::zeros(nao * nao); band.len()]; nset],
            vec![vec![CTensor::zeros(nao * nao); band.len()]; nset],
        ];

        // U-10: ONE scratch for the whole block loop, shared by BOTH spin
        // channels — the open-shell path allocated these buffers twice as
        // often as the closed-shell one did, which is what made this the
        // larger of the two wins. Bit-exact; see `Scratch`.
        let mut sc = Scratch::default();
        let mut dena: Vec<f64> = Vec::new();
        let mut denb: Vec<f64> = Vec::new();
        let mut ta: Vec<f64> = Vec::new();
        let mut tb: Vec<f64> = Vec::new();

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
                let rho_a = self.eval_rho_into(&ao2, &dms[0][i], ty, &mut sc)?;
                let rho_b = self.eval_rho_into(&ao2, &dms[1][i], ty, &mut sc)?;
                let out = eval_xc_eff_uks(xc_code, &rho_a, &rho_b)?;
                // U-10: `clear` + `extend` reuses each allocation and produces
                // the identical values in the identical order.
                dena.clear();
                dena.extend(rho_a.row(0).iter().zip(w).map(|(r, wg)| r * wg));
                denb.clear();
                denb.extend(rho_b.row(0).iter().zip(w).map(|(r, wg)| r * wg));
                nelec_a_parts[i].push(oracle_sum(&dena));
                nelec_b_parts[i].push(oracle_sum(&denb));
                ta.clear();
                ta.extend(dena.iter().zip(&out.exc).map(|(d, e)| d * e));
                tb.clear();
                tb.extend(denb.iter().zip(&out.exc).map(|(d, e)| d * e));
                // U-03 step 4: `numint.py:485-486` performs TWO separate
                // accumulations, `excsum[i] += dena.dot(exc)` then
                // `excsum[i] += denb.dot(exc)`. Folding them as
                // `E + (Sa + Sb)` instead of `(E + Sa) + Sb` is ~1 ulp per
                // block — harmless against a 1e-11 gate, but a real KUKS-only
                // bit-parity divergence from the oracle. Pushing them as two
                // partials keeps upstream's association AND the ordered tree.
                excsum_parts[i].push(oracle_sum(&ta));
                excsum_parts[i].push(oracle_sum(&tb));
                for (s, vm) in vmat.iter_mut().enumerate() {
                    let wv = weighted(&out, s, w);
                    self.accumulate_vxc_into(&mut vm[i], &ao1, &wv, ty, &mut sc);
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
        let nelec: Vec<(f64, f64)> = nelec_a_parts
            .iter()
            .zip(&nelec_b_parts)
            .map(|(a, b)| (oracle_sum(a), oracle_sum(b)))
            .collect();
        let excsum: Vec<f64> = excsum_parts.iter().map(|p| oracle_sum(p)).collect();
        Ok(NrKUksResult {
            nelec,
            excsum,
            vmat,
        })
    }

    fn symmetrized_rhos(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        dms: &[&KMats],
        ty: XcType,
    ) -> Result<Vec<RhoEff>, PbcDftError> {
        let mesh = match grids {
            PeriodicGrids::Uniform(g) => g.mesh,
            PeriodicGrids::Becke(_) => {
                return Err(err(
                    "PYSCF_PBC_KSYMM_RHO=symmetrize requires a uniform FFT grid",
                ));
            }
        };
        let kp = self.ksymm().expect("checked by caller");
        let coords = grids.coords()?;
        let ngrids = coords.len();
        let nibz = kp.kpts_ibz.len();
        let mut per_k: Vec<Vec<RhoEff>> = (0..dms.len())
            .map(|_| (0..nibz).map(|_| RhoEff::zeros(ty, ngrids)).collect())
            .collect();
        let mut sc = Scratch::default();
        let mut imag = 0.0_f64;
        for (p0, p1) in self.block_ranges(ngrids, ty, nibz) {
            let ao = self.eval_ao_route(cell, &coords[p0..p1], &kp.kpts_ibz, ty, 1)?;
            for (set, matrices) in dms.iter().enumerate() {
                for ik in 0..nibz {
                    let bz = kp.ibz2bz[ik];
                    let dm = if matrices.len() == nibz {
                        &matrices[ik]
                    } else {
                        matrices.get(bz).ok_or_else(|| {
                            err(format!(
                                "symmetrized density: {} matrices do not cover IBZ point {ik}",
                                matrices.len()
                            ))
                        })?
                    };
                    let (block, im) =
                        eval_rho_one(ao.at(ik), dm, p1 - p0, cell.mol.nao_nr, ty, &mut sc)?;
                    imag = imag.max(im);
                    for v in 0..ty.nvar() {
                        per_k[set][ik].row_mut(v)[p0..p1].copy_from_slice(block.row(v));
                    }
                }
            }
        }

        let mut out: Vec<RhoEff> = dms.iter().map(|_| RhoEff::zeros(ty, ngrids)).collect();
        for set in 0..dms.len() {
            for ik in 0..nibz {
                let scalar = kp
                    .symmetrize_density(per_k[set][ik].row(0), ik, mesh)
                    .map_err(|e| err(format!("symmetrize_density: {e}")))?;
                for (dst, value) in out[set].row_mut(0).iter_mut().zip(scalar) {
                    *dst += value;
                }
                if ty == XcType::Gga {
                    let grad = kp
                        .symmetrize_density_vec(
                            [
                                per_k[set][ik].row(1),
                                per_k[set][ik].row(2),
                                per_k[set][ik].row(3),
                            ],
                            ik,
                            mesh,
                        )
                        .map_err(|e| err(format!("symmetrize_density_vec: {e}")))?;
                    for v in 0..3 {
                        for (dst, &value) in out[set].row_mut(v + 1).iter_mut().zip(&grad[v]) {
                            *dst += value;
                        }
                    }
                }
            }
            out[set].scale(1.0 / kp.nkpts() as f64);
        }
        self.last_imag.set(imag / kp.nkpts() as f64);
        Ok(out)
    }

    fn nr_rks_symmetrized(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &KDms,
        ty: XcType,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<NrKResult, PbcDftError> {
        let unfolded = self.unfold_kdms(cell, dms, cell.mol.nao_nr)?;
        let refs: Vec<&KMats> = unfolded.iter().collect();
        let rho = self.symmetrized_rhos(cell, grids, &refs, ty)?;
        let coords = grids.coords()?;
        let weights = grids.weights()?;
        let nao = cell.mol.nao_nr;
        let band = kpts_band.unwrap_or(self.kpts_ibz());
        let mut nelec_parts = vec![Vec::new(); rho.len()];
        let mut excsum_parts = vec![Vec::new(); rho.len()];
        let mut vmat = vec![vec![CTensor::zeros(nao * nao); band.len()]; rho.len()];
        let mut sc = Scratch::default();
        for (p0, p1) in self.block_ranges(coords.len(), ty, band.len()) {
            let w = &weights[p0..p1];
            let ao = self.eval_ao_route(cell, &coords[p0..p1], band, ty, 1)?;
            for set in 0..rho.len() {
                let rb = rho[set].slice(p0, p1);
                let xc = eval_xc_eff_rks(xc_code, &rb)?;
                let den: Vec<f64> = rb.row(0).iter().zip(w).map(|(r, x)| r * x).collect();
                nelec_parts[set].push(oracle_sum(&den));
                let terms: Vec<f64> = den.iter().zip(&xc.exc).map(|(d, e)| d * e).collect();
                excsum_parts[set].push(oracle_sum(&terms));
                let wv = weighted(&xc, 0, w);
                self.accumulate_vxc_into(&mut vmat[set], &ao, &wv, ty, &mut sc);
            }
        }
        for set in &mut vmat {
            for m in set {
                add_conj_transpose(m, nao);
            }
        }
        Ok(NrKResult {
            nelec: nelec_parts.iter().map(|x| oracle_sum(x)).collect(),
            excsum: excsum_parts.iter().map(|x| oracle_sum(x)).collect(),
            vmat,
        })
    }

    fn nr_uks_symmetrized(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &[KDms; 2],
        ty: XcType,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<NrKUksResult, PbcDftError> {
        let unfolded = [
            self.unfold_kdms(cell, &dms[0], cell.mol.nao_nr)?,
            self.unfold_kdms(cell, &dms[1], cell.mol.nao_nr)?,
        ];
        let nset = unfolded[0].len();
        if unfolded[1].len() != nset {
            return Err(err("pbc nr_uks: alpha and beta carry different set counts"));
        }
        let mut refs = Vec::with_capacity(2 * nset);
        refs.extend(unfolded[0].iter());
        refs.extend(unfolded[1].iter());
        let rho = self.symmetrized_rhos(cell, grids, &refs, ty)?;
        let coords = grids.coords()?;
        let weights = grids.weights()?;
        let nao = cell.mol.nao_nr;
        let band = kpts_band.unwrap_or(self.kpts_ibz());
        let mut nelec_a = vec![Vec::new(); nset];
        let mut nelec_b = vec![Vec::new(); nset];
        let mut excsum = vec![Vec::new(); nset];
        let mut vmat: [Vec<KMats>; 2] = [
            vec![vec![CTensor::zeros(nao * nao); band.len()]; nset],
            vec![vec![CTensor::zeros(nao * nao); band.len()]; nset],
        ];
        let mut sc = Scratch::default();
        for (p0, p1) in self.block_ranges(coords.len(), ty, band.len()) {
            let w = &weights[p0..p1];
            let ao = self.eval_ao_route(cell, &coords[p0..p1], band, ty, 1)?;
            for set in 0..nset {
                let ra = rho[set].slice(p0, p1);
                let rb = rho[nset + set].slice(p0, p1);
                let xc = eval_xc_eff_uks(xc_code, &ra, &rb)?;
                let da: Vec<f64> = ra.row(0).iter().zip(w).map(|(r, x)| r * x).collect();
                let db: Vec<f64> = rb.row(0).iter().zip(w).map(|(r, x)| r * x).collect();
                nelec_a[set].push(oracle_sum(&da));
                nelec_b[set].push(oracle_sum(&db));
                excsum[set].push(oracle_sum(
                    &da.iter()
                        .zip(&xc.exc)
                        .map(|(d, e)| d * e)
                        .collect::<Vec<_>>(),
                ));
                excsum[set].push(oracle_sum(
                    &db.iter()
                        .zip(&xc.exc)
                        .map(|(d, e)| d * e)
                        .collect::<Vec<_>>(),
                ));
                for spin in 0..2 {
                    let wv = weighted(&xc, spin, w);
                    self.accumulate_vxc_into(&mut vmat[spin][set], &ao, &wv, ty, &mut sc);
                }
            }
        }
        for spin in &mut vmat {
            for set in spin {
                for m in set {
                    add_conj_transpose(m, nao);
                }
            }
        }
        Ok(NrKUksResult {
            nelec: nelec_a
                .iter()
                .zip(&nelec_b)
                .map(|(a, b)| (oracle_sum(a), oracle_sum(b)))
                .collect(),
            excsum: excsum.iter().map(|x| oracle_sum(x)).collect(),
            vmat,
        })
    }

    /// `KNumInt._vxc_mat` — `numint.py:1223-1240`, accumulated into `out`.
    ///
    /// `wv` is `[var][grid]` and ALREADY weight-scaled. With `hermi = 1` — the
    /// only mode the SCF drivers use — `wv[0]` carries the `*0.5` that pairs
    /// with the `V + V†` symmetrisation the caller applies.
    fn accumulate_vxc(&self, out: &mut KMats, ao: &EvalAoKptsOutput, wv: &[Vec<f64>], ty: XcType) {
        self.accumulate_vxc_into(out, ao, wv, ty, &mut Scratch::default());
    }

    /// [`KNumInt::accumulate_vxc`] against a caller-owned scratch — U-10, and
    /// the same reasoning as [`KNumInt::eval_rho_into`].
    fn accumulate_vxc_into(
        &self,
        out: &mut KMats,
        ao: &EvalAoKptsOutput,
        wv: &[Vec<f64>],
        ty: XcType,
        sc: &mut Scratch,
    ) {
        let nao = ao.nao;
        let ngrids = ao.ngrids;
        let nvar = ty.nvar();
        for (k, m) in out.iter_mut().enumerate() {
            vxc_mat_one(m, ao.at(k), wv, nao, ngrids, nvar, sc);
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
        // Group A (`numint.py:956-959`).
        let dms = &self.unfold_dms(cell, dms, cell.mol.nao_nr)?;
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
        // Group A (`numint.py:908-911`).
        let dms = &dms
            .iter()
            .map(|d| self.unfold_dms(cell, d, cell.mol.nao_nr))
            .collect::<Result<Vec<_>, _>>()?;
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
        // Group A (`numint.py:859-863`) — the one site that unfolds the
        // ORBITALS rather than the density. The resulting `dms` are then
        // full-BZ length, so `cache_xc_kernel1`'s own unfold is a no-op.
        let (mo_coeff, mo_occ) = self.unfold_mos(cell, mo_coeff, mo_occ, nao)?;
        let dms: Vec<KMats> = mo_coeff
            .iter()
            .zip(&mo_occ)
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
        // Group B (`numint.py:647-649` / `:779-781`): a symmetric k-set uses
        // `kpts.kpts_ibz` DIRECTLY here — no unfold, unlike the five Group-A
        // sites. Falls back to the full set under `KSet::Full`.
        let kset = self.kpts_ibz();
        let nk = kset.len();
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
        let mut vmat: Vec<KMats> = vec![vec![CTensor::zeros(nao * nao); nk]; nset];

        for (p0, p1) in self.block_ranges(ngrids, ty, nk) {
            let ao = self.eval_ao(cell, &coords[p0..p1], kset, ty)?;
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
        // Group B (`numint.py:647-649` / `:779-781`): a symmetric k-set uses
        // `kpts.kpts_ibz` DIRECTLY here — no unfold, unlike the five Group-A
        // sites. Falls back to the full set under `KSet::Full`.
        let kset = self.kpts_ibz();
        let nk = kset.len();
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
            vec![vec![CTensor::zeros(nao * nao); nk]; nset],
            vec![vec![CTensor::zeros(nao * nao); nk]; nset],
        ];

        for (p0, p1) in self.block_ranges(ngrids, ty, nk) {
            let ao = self.eval_ao(cell, &coords[p0..p1], kset, ty)?;
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

/// Reusable per-call scratch for [`eval_rho_one`] and [`vxc_mat_one`] —
/// plan item U-10 (`.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`),
/// which is `KUKS-OPTIMISATION-PLAN` U-06 step 6 and U-05 step 2 closed.
///
/// # What this replaces, and why it was worth closing
///
/// Every one of these buffers used to be a fresh `vec![0.0; n]` INSIDE the
/// grid-block loop. Per block, `nr_uks` calls `eval_rho` twice (once per
/// spin), each of which calls [`eval_rho_one`] once per k-point, each of which
/// allocated and zeroed `2 * ngrids * nao` doubles for `c0` plus
/// `2 * ngrids` per density component; `accumulate_vxc` then does the same
/// again per spin per k for `aow`, and [`vxc_mat_one`]'s per-row `terms`
/// buffers were allocated `nao` times per call on top.
///
/// At the reference cell (`si`, `mesh = 31`, `ngrids = 29 791`, `nao = 8`,
/// `nkpts = 8`) `c0` alone is 3.8 MiB, taken 16 times per block per cycle in
/// KUKS — about 61 MiB of pure allocate-and-zero for `c0`, and as much again
/// for `aow`. `U-06` left this open because it read the fix as needing
/// interior-mutable state on `KNumInt` "whose aliasing story is not free".
/// It does not: the lifetime that matters is one CALL, not one `KNumInt`, so
/// the scratch is an ordinary `&mut` argument threaded down from `nr_rks` /
/// `nr_uks` and there is no aliasing story at all.
///
/// # Bit-parity: EXACT
///
/// Every buffer this replaces was zero-filled at allocation and then either
/// fully overwritten or accumulated into from zero. [`Scratch::zeroed`]
/// reproduces that starting state exactly, and [`Scratch::raw`] is used only
/// where every element is assigned before it is read. No arithmetic, no
/// ordering and no partition changes.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    /// `eval_rho_one`'s `c0 = ao . dm` — two `ngrids * nao` planes.
    c0_re: Vec<f64>,
    c0_im: Vec<f64>,
    /// `eval_rho_one`'s per-component `_contract_rho` accumulator —
    /// two `ngrids` planes.
    acc_re: Vec<f64>,
    acc_im: Vec<f64>,
    /// `vxc_mat_one`'s `aow = ao * wv` — two `ngrids * nao` planes.
    aow_re: Vec<f64>,
    aow_im: Vec<f64>,
    /// `vxc_mat_one`'s per-output-row product buffer — two `nao * ngrids`
    /// planes, one `ngrids` slice per row, handed to the rayon workers by
    /// `par_chunks_mut` so each worker owns a disjoint slice.
    terms_re: Vec<f64>,
    terms_im: Vec<f64>,
}

impl Scratch {
    /// A zero-filled `&mut [f64]` of length `n` — the state a fresh
    /// `vec![0.0; n]` had.
    fn zeroed(buf: &mut Vec<f64>, n: usize) -> &mut [f64] {
        if buf.len() < n {
            buf.resize(n, 0.0);
        }
        let out = &mut buf[..n];
        out.fill(0.0);
        out
    }

    /// A `&mut [f64]` of length `n` whose contents are UNSPECIFIED. Only for
    /// buffers every element of which is assigned before it is read.
    fn raw(buf: &mut Vec<f64>, n: usize) -> &mut [f64] {
        if buf.len() < n {
            buf.resize(n, 0.0);
        }
        &mut buf[..n]
    }

    /// [`eval_rho_one`]'s four buffers, all zeroed, borrowed together.
    ///
    /// They are split in ONE call rather than two because `c0` stays borrowed
    /// across the whole component loop that `acc` is used inside; taking them
    /// separately would be two overlapping `&mut self` borrows.
    fn split_rho(
        &mut self,
        n_c0: usize,
        n_acc: usize,
    ) -> (&mut [f64], &mut [f64], &mut [f64], &mut [f64]) {
        (
            Self::zeroed(&mut self.c0_re, n_c0),
            Self::zeroed(&mut self.c0_im, n_c0),
            Self::zeroed(&mut self.acc_re, n_acc),
            Self::zeroed(&mut self.acc_im, n_acc),
        )
    }

    /// [`vxc_mat_one`]'s four buffers. `aow` is ACCUMULATED into and so must
    /// start at zero; `terms` is fully assigned before it is read on every
    /// `(row, nu)` pass, so it is handed out raw — the same contract the
    /// `vec![0.0; ngrids]` it replaces satisfied by accident.
    fn split_vxc(
        &mut self,
        n_aow: usize,
        n_terms: usize,
    ) -> (&mut [f64], &mut [f64], &mut [f64], &mut [f64]) {
        (
            Self::zeroed(&mut self.aow_re, n_aow),
            Self::zeroed(&mut self.aow_im, n_aow),
            Self::raw(&mut self.terms_re, n_terms),
            Self::raw(&mut self.terms_im, n_terms),
        )
    }
}

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
    sc: &mut Scratch,
) {
    if ngrids == 0 {
        return;
    }
    // U-10: `aow` (zeroed, accumulated into) and the per-row `terms` planes
    // (fully assigned before read) come from reused scratch instead of a
    // `vec![0.0; ..]` per call and a `vec![0.0; ngrids]` per output row.
    let (aow_re, aow_im, terms_re_all, terms_im_all) = sc.split_vxc(ngrids * nao, nao * ngrids);
    // aow, in the same F-order-per-component layout as `ao`'s component 0.
    //
    // W-06: `nu` indexes DISJOINT output rows of `aow`, so it is the axis split
    // across workers; the component sum over `n` stays serial and ascending
    // inside each row, which is what makes this bit-identical to the pre-W-06
    // `n`-outer nest. The `if s == 0.0 { continue; }` skip is kept deliberately
    // — see the module note on it.
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
    // U-10: each output row's `terms` planes are a DISJOINT `ngrids` slice of
    // one preallocated buffer, handed to the worker that owns that row by the
    // same `par_chunks_mut` split. The partition, the ownership and the
    // arithmetic are unchanged; only the per-row allocation is gone.
    out.re
        .par_chunks_mut(nao)
        .zip(out.im.par_chunks_mut(nao))
        .zip(
            terms_re_all
                .par_chunks_mut(ngrids)
                .zip(terms_im_all.par_chunks_mut(ngrids)),
        )
        .enumerate()
        .for_each(|(mu, ((orow, oirow), (terms_re, terms_im)))| {
            let mb = mu * ngrids;
            for nu in 0..nao {
                let nb = nu * ngrids;
                for g in 0..ngrids {
                    let (ar, ai) = (ao.re[mb + g], -ao.im[mb + g]);
                    let (br, bi) = (aow_re[nb + g], aow_im[nb + g]);
                    terms_re[g] = ar * br - ai * bi;
                    terms_im[g] = ar * bi + ai * br;
                }
                orow[nu] += oracle_sum(terms_re);
                oirow[nu] += oracle_sum(terms_im);
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
    sc: &mut Scratch,
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
    // U-10: reused, zero-filled scratch instead of a fresh `vec![0.0; ..]`
    // per k-point per spin per block. Same starting state, same partition,
    // same arithmetic — bit-exact.
    let (c0_re, c0_im, acc_re, acc_im) = sc.split_rho(ngrids * nao, ngrids);
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
        // U-10: the per-component accumulators are re-zeroed here, which is
        // exactly the state the per-component `vec![0.0; ngrids]` allocations
        // they replace started in.
        acc_re.fill(0.0);
        acc_im.fill(0.0);
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
        for v in acc_im.iter() {
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
