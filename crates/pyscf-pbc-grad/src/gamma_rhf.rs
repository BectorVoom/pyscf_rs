//! Gamma-point restricted gradient over `MultiGridNumInt2` — `pbc/grad/rhf.py` (188 l).
//!
//! Upstream's `grad_elec` (`rhf.py:33-87`) is a multigrid-v2 program with NO
//! non-multigrid branch: `rhf.py:42-47` asserts
//! `isinstance(ni, MultiGridNumInt2)` and the `else` is
//! `raise NotImplementedError` (`18-CONTEXT §1.1`). That assertion is
//! [`GammaCoulombEngine`]: callers hand the engine in, and anything that is
//! not `MultiGridNumInt2` is a named [`PbcGradError::NonMultigridCoulomb`]
//! refusal, never a fallback. A non-gamma `kpt` raises
//! [`PbcGradError::NonGammaKpt`] (`rhf.py:78-79`), exactly as upstream.
//!
//! Assembly order follows upstream's lines:
//!
//! * pseudo branch (`:64-70`): `de = vpploc_part1_nuc_grad + vpploc_part2 +
//!   vppnl`, and `h1ao -= get_vpploc_part1_ip1` ONLY when the part-1
//!   potential was not precomputed (`ni.vpplocG_part1 is None`). Ported
//!   unconditionally the subtraction double-counts — hence
//!   [`GammaRhfGradients::with_vpploc_part1_precomputed`]. This port never
//!   populates that cache (`pair_grad` documents the no-op), so the default
//!   is "not precomputed" and the subtraction applies.
//! * all-electron branch (`:71-73`): `de = get_nuc_nuc_grad`,
//!   `h1ao -= get_nuc_ip1`. The two branches share no prefix.
//! * `de += contract(h1ao + vhf, dm0) * 2` (`:74`),
//!   `de += contract(s1, dme0) * -2` (`:75`). The `*2`/`*-2` scalings are
//!   exact (powers of two); the three per-atom contributions are combined
//!   through `oracle_sum` (D-PBC-17 fixed-order discipline).
//! * `get_veff` (`:138-142`) returns `-get_veff_ip1`, with `xc_code =
//!   getattr(mf, 'xc', None)` — `None` for HF, which is how one function
//!   serves both the HF and the KS gamma gradient. `None` maps to the
//!   `"HF"` (Coulomb-only) route here.
//! * `get_ovlp` (`:134-135`) is `-pbc_intor('int1e_ipovlp')`; `grad_nuc`
//!   (`:145-155`) is 18-03's `ewald_nuc_grad`.
//!
//! At gamma every quantity here is real: the `.real`-inside rule (D-PBC-31
//! clause 4) is trivially satisfied and no complex accumulator is
//! introduced. All AO matrices below are symmetric at gamma, so every trace
//! and every `_contract_vhf_dm` call (which contracts matching indices —
//! `contract.rs`, mirroring the C worker, NOT the transposed `dm[j,i]` of
//! the `krhf` einsum in `18-CONTEXT` trap 9) is layout-independent.
//!
//! The cached-density handle is threaded explicitly, as 18-09 Task 1
//! requires: `get_veff_ip1` returns `rhoG`, and the pseudo branch passes it
//! to `vpploc_part1_nuc_grad` instead of rebuilding it. The all-electron
//! branch passes `None` to `get_nuc_nuc_grad` — upstream does NOT read the
//! cache there.
//!
//! Screening follows [`SCREEN_VHF_DM_CONTRACT`](crate::contract::SCREEN_VHF_DM_CONTRACT)
//! (18-17 Task 1: screened, exact). No cubecl here (ALG-06 — orchestration
//! only; the one device kernel consumed is the clause-10
//! `contract_atom_grid` inside the multigrid entries).

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_dft::multigrid::pair::MultiGridNumInt2;
use pyscf_pbc_gto::Cell;

use crate::contract::{SCREEN_VHF_DM_CONTRACT, contract_vhf_dm};
use crate::error::PbcGradError;
use crate::gradients::{Gradient, Gradients};

fn wrap_dft(context: &str, error: impl std::fmt::Display) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "gamma RHF gradient ({context}): {error}"
    )))
}

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(message.into()))
}

/// Upstream `gamma_point(kpt)` (`pbc/lib/kpts_helper.py`): the gamma test
/// this port's multigrid entries already use (`|k|^2 < 1e-18`).
pub(crate) fn is_gamma_kpt(kpt: &[f64; 3]) -> bool {
    kpt[0] * kpt[0] + kpt[1] * kpt[1] + kpt[2] * kpt[2] < 1e-18
}

/// Refuse a non-gamma k-point (`pbc/grad/rhf.py:78-79`, shared with
/// `uhf.py:78-79`), before any work.
pub(crate) fn require_gamma_kpt(kpt: &[f64; 3]) -> Result<(), PyscfRsError> {
    if !is_gamma_kpt(kpt) {
        return Err(PbcGradError::NonGammaKpt(kpt[0], kpt[1], kpt[2]).into());
    }
    Ok(())
}

/// `-pbc_intor(intor)` at a single k-point as `(3, nao, nao)` planes in
/// `[x, mu, nu]` order (shared with the unrestricted body). `intor` must be
/// a 3-component derivative family (`int1e_ipkin`, `int1e_ipovlp`).
pub(crate) fn negated_ip_intor(
    cell: &Cell,
    kpts: &[[f64; 3]],
    intor: &str,
) -> Result<Vec<f64>, PyscfRsError> {
    let out = pyscf_pbc_gto::pbc_intor(cell, intor, kpts, Default::default())?;
    if out.kmats.len() != 1 || out.comp != 3 {
        return Err(invalid(format!(
            "gamma gradient: {intor} returned {} k-point planes with comp = {}",
            out.kmats.len(),
            out.comp
        )));
    }
    let nao = cell.mol.nao_nr;
    let size = nao * nao;
    if out.ni != nao || out.nj != nao || out.kmats[0].re.len() != 3 * size {
        return Err(invalid(format!(
            "gamma gradient: {intor} has the wrong AO shape"
        )));
    }
    Ok(out.kmats[0].re.iter().map(|v| -v).collect())
}

/// The Coulomb engine seam — `rhf.py:42-47`.
///
/// `MultiGridV2` is the only working route. `Other` names what was supplied
/// instead (e.g. `"FFTDF"`, `"GDF"`, `"AFTDF"`) and resolves to the
/// [`PbcGradError::NonMultigridCoulomb`] refusal, mirroring the
/// `else: raise NotImplementedError`.
#[derive(Debug, Clone, Copy)]
pub enum GammaCoulombEngine<'a> {
    /// `isinstance(ni, MultiGridNumInt2)` — the assertion passes.
    MultiGridV2(&'a MultiGridNumInt2),
    /// Any other numint — refused with its name attached.
    Other(&'static str),
}

impl<'a> GammaCoulombEngine<'a> {
    pub(crate) fn resolve(self) -> Result<&'a MultiGridNumInt2, PyscfRsError> {
        match self {
            GammaCoulombEngine::MultiGridV2(ni) => Ok(ni),
            GammaCoulombEngine::Other(detail) => Err(PbcGradError::NonMultigridCoulomb {
                detail: detail.into(),
            }
            .into()),
        }
    }
}

/// Gamma-point restricted (Hartree-Fock or Kohn-Sham) nuclear gradient —
/// `pbc/grad/rhf.py`'s `Gradients` class.
///
/// The density `dm0` and the energy-weighted density `dme0` are caller-held
/// `nao × nao` symmetric real matrices (use [`gamma_make_rdm1e`] to build
/// the latter from orbitals); the SCF itself lives outside this crate, so
/// there is no convergence path here and no second-solution noise.
pub struct GammaRhfGradients<'a> {
    cell: &'a Cell,
    ni: &'a MultiGridNumInt2,
    dm0: Vec<f64>,
    dme0: Vec<f64>,
    kpts: [[f64; 3]; 1],
    xc_code: Option<String>,
    atmlst: Option<Vec<usize>>,
    screened: bool,
    vpploc_part1_precomputed: bool,
}

impl<'a> GammaRhfGradients<'a> {
    /// Resolve the Coulomb engine (`rhf.py:42-47`) and validate the density
    /// shapes. The k-point defaults to gamma; a non-gamma k-point is refused
    /// later, inside [`GammaRhfGradients::electronic_gradient`]
    /// (`rhf.py:78-79`).
    pub fn new(
        cell: &'a Cell,
        engine: GammaCoulombEngine<'a>,
        dm0: Vec<f64>,
        dme0: Vec<f64>,
    ) -> Result<Self, PyscfRsError> {
        let ni = engine.resolve()?;
        let nao = cell.mol.nao_nr;
        let size = nao
            .checked_mul(nao)
            .ok_or_else(|| invalid("gamma RHF gradient: AO count overflow"))?;
        if dm0.len() != size {
            return Err(invalid(format!(
                "gamma RHF gradient: dm0 has {} elements, expected nao x nao = {size}",
                dm0.len()
            )));
        }
        if dme0.len() != size {
            return Err(invalid(format!(
                "gamma RHF gradient: dme0 has {} elements, expected nao x nao = {size}",
                dme0.len()
            )));
        }
        if dm0.iter().chain(&dme0).any(|x| !x.is_finite()) {
            return Err(invalid("gamma RHF gradient: dm0/dme0 must be finite"));
        }
        Ok(Self {
            cell,
            ni,
            dm0,
            dme0,
            kpts: [[0.0; 3]],
            xc_code: None,
            atmlst: None,
            screened: SCREEN_VHF_DM_CONTRACT,
            vpploc_part1_precomputed: false,
        })
    }

    /// K-point for the gradient (`rhf.py:34`, default gamma).
    pub fn with_kpt(mut self, kpt: [f64; 3]) -> Self {
        self.kpts = [kpt];
        self
    }

    /// Exchange-correlation code (`getattr(mf, 'xc', None)` at `rhf.py:140`).
    /// `None` is HF — the Coulomb-only route — which is how one `get_veff`
    /// serves both the HF and the KS gamma gradient.
    pub fn with_xc(mut self, xc: Option<&str>) -> Self {
        self.xc_code = xc.map(str::to_string);
        self
    }

    /// Atom subset (`de = de[atmlst]`, `rhf.py:77`).
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Self {
        self.atmlst = Some(atmlst);
        self
    }

    /// Screening switch for both `_contract_vhf_dm` calls. Defaults to
    /// [`SCREEN_VHF_DM_CONTRACT`]; set `false` for the unscreened branch.
    pub fn with_screened(mut self, screened: bool) -> Self {
        self.screened = screened;
        self
    }

    /// Whether the part-1 local potential was precomputed
    /// (`ni.vpplocG_part1 is not None`, `rhf.py:68-70`). When `false` (this
    /// port's only populated state — the cache field does not exist here)
    /// `h1ao -= get_vpploc_part1_ip1` applies. Setting `true` while the
    /// potential was NOT precomputed silently drops a term; setting `false`
    /// while it was double-counts it.
    pub fn with_vpploc_part1_precomputed(mut self, precomputed: bool) -> Self {
        self.vpploc_part1_precomputed = precomputed;
        self
    }

    fn xc_str(&self) -> &str {
        self.xc_code.as_deref().unwrap_or("HF")
    }

    fn has_pseudo(&self) -> bool {
        (0..self.cell.natm).any(|ia| self.cell.atom_pseudo(ia).is_some())
    }

    /// `-pbc_intor(intor)` at the (single) k-point, as `(3, nao, nao)`
    /// planes in `[x, mu, nu]` order. `intor` must be a 3-component
    /// derivative family (`int1e_ipkin`, `int1e_ipovlp`).
    fn negated_ip(&self, intor: &str) -> Result<Vec<f64>, PyscfRsError> {
        negated_ip_intor(self.cell, &self.kpts, intor)
    }

    /// `get_ovlp` (`rhf.py:134-135`): `-cell.pbc_intor('int1e_ipovlp', kpt)`.
    pub fn overlap_ip(&self) -> Result<Vec<f64>, PyscfRsError> {
        self.negated_ip("int1e_ipovlp")
    }

    /// `get_veff` (`rhf.py:138-142`): `-mf._numint.get_veff_ip1(...)`.
    /// The minus is here. Returns `(3, nao, nao)` in `[x, mu, nu]` order.
    pub fn veff_ip(&self) -> Result<Vec<f64>, PyscfRsError> {
        let result = self
            .ni
            .get_veff_ip1(self.cell, self.xc_str(), &self.dm0, &self.kpts)
            .map_err(|e| wrap_dft("get_veff_ip1", e))?;
        Ok(result.veff_ip1.into_iter().map(|v| -v).collect())
    }

    /// `grad_nuc` (`rhf.py:145-155`): 18-03's `ewald_nuc_grad`.
    pub fn nuclear_gradient(&self) -> Result<Gradient, PyscfRsError> {
        pyscf_pbc_gto::ewald_nuc_grad(self.cell, None, None)
    }

    fn neighbor_list(&self) -> Result<Option<pyscf_pbc_gto::NeighborList>, PyscfRsError> {
        if !self.screened {
            return Ok(None);
        }
        let ls = pyscf_pbc_gto::get_lattice_ls_default(self.cell)?;
        Ok(Some(pyscf_pbc_gto::build_neighbor_list_for_shlpairs(
            self.cell, &ls,
        )?))
    }

    fn contract_planes(&self, planes: &[f64], dm: &[f64]) -> Result<Gradient, PyscfRsError> {
        let nao = self.cell.mol.nao_nr;
        let size = nao * nao;
        if planes.len() != 3 * size || dm.len() != size {
            return Err(invalid(
                "gamma RHF gradient: contraction planes are not (3, nao, nao)",
            ));
        }
        let vhf: [CTensor; 3] = std::array::from_fn(|c| CTensor {
            re: planes[c * size..(c + 1) * size].to_vec(),
            im: vec![0.0; size],
        });
        let dm_tensor = CTensor {
            re: dm.to_vec(),
            im: vec![0.0; size],
        };
        let neighbor = self.neighbor_list()?;
        contract_vhf_dm(self.cell, &vhf, &dm_tensor, neighbor.as_ref())
    }

    /// `grad_elec` (`rhf.py:33-87`), in upstream's order — including the
    /// early unbinding hint (`rhf.py:76` frees the full-grid arrays before
    /// the atom loop; the scopes below drop each plane as soon as its last
    /// consumer runs).
    ///
    /// `extra_force` (`rhf.py:81-82`) contributes nothing here: the shared
    /// [`Gradients::extra_force`] default is zero, and DFT+U / dispersion
    /// extras are outside this plan's scope.
    pub fn electronic_gradient(&self) -> Result<Gradient, PyscfRsError> {
        // rhf.py:78-79 — the non-gamma refusal, before any work.
        require_gamma_kpt(&self.kpts[0])?;
        let natm = self.cell.natm;
        let nao = self.cell.mol.nao_nr;
        let size = nao * nao;

        let s1 = self.overlap_ip()?;
        // One `get_veff_ip1` call serves both halves: the raw matrix (whose
        // negation is `vhf = get_veff`, rhf.py:138-142) and the cached
        // `rhoG` handle the pseudo branch threads into
        // `vpploc_part1_nuc_grad` (18-09 Task 1 — returned, never stashed).
        let veff_result = self
            .ni
            .get_veff_ip1(self.cell, self.xc_str(), &self.dm0, &self.kpts)
            .map_err(|e| wrap_dft("get_veff_ip1", e))?;
        debug_assert_eq!(veff_result.veff_ip1.len(), 3 * size);
        let mut h1ao = self.negated_ip("int1e_ipkin")?;

        // rhf.py:62-76. The two branches take different paths for BOTH `de`
        // and `h1ao` — no shared prefix.
        let mut de = if self.has_pseudo() {
            let mut de = self
                .ni
                .vpploc_part1_nuc_grad(
                    self.cell,
                    &self.dm0,
                    &self.kpts,
                    Some(&veff_result.rho_g),
                    None,
                )
                .map_err(|e| wrap_dft("vpploc_part1_nuc_grad", e))?;
            let part2 =
                pyscf_pbc_gto::pseudo::vpploc_part2_nuc_grad(self.cell, &self.dm0, &self.kpts)?;
            let nonloc = pyscf_pbc_gto::pseudo::vppnl_nuc_grad(self.cell, &self.dm0, &self.kpts)?;
            for ia in 0..natm {
                for c in 0..3 {
                    de[ia][c] = oracle_sum(&[de[ia][c], part2[ia][c], nonloc[ia][c]]);
                }
            }
            // rhf.py:68-70 — conditional on `vpplocG_part1 is None`.
            if !self.vpploc_part1_precomputed {
                let correction = self
                    .ni
                    .get_vpploc_part1_ip1(self.cell, &self.kpts)
                    .map_err(|e| wrap_dft("get_vpploc_part1_ip1", e))?;
                for (h, v) in h1ao.iter_mut().zip(&correction) {
                    *h -= v;
                }
            }
            de
        } else {
            let de = self
                .ni
                .get_nuc_nuc_grad(self.cell, &self.dm0, &self.kpts, None)
                .map_err(|e| wrap_dft("get_nuc_nuc_grad", e))?;
            let nuc_ip1 = self
                .ni
                .get_nuc_ip1(self.cell, &self.kpts)
                .map_err(|e| wrap_dft("get_nuc_ip1", e))?;
            for (h, v) in h1ao.iter_mut().zip(&nuc_ip1) {
                *h -= v;
            }
            de
        };

        // rhf.py:74 (`np.add(h1ao, vhf)` with `vhf = get_veff = -veff_ip1`,
        // so the raw matrix subtracts). `h1ao`/`s1`/the veff result drop
        // after this block (:76).
        let mut h_plus_v = h1ao;
        for (h, v) in h_plus_v.iter_mut().zip(&veff_result.veff_ip1) {
            *h -= v;
        }
        let first = self.contract_planes(&h_plus_v, &self.dm0)?;
        let second = self.contract_planes(&s1, &self.dme0)?;
        for ia in 0..natm {
            for c in 0..3 {
                // `* 2` / `* -2` are exact (powers of two); the three-way
                // sum is fixed-order through `oracle_sum`.
                de[ia][c] = oracle_sum(&[de[ia][c], 2.0 * first[ia][c], -2.0 * second[ia][c]]);
            }
        }

        // rhf.py:77.
        if let Some(atmlst) = &self.atmlst {
            let mut selected = Vec::with_capacity(atmlst.len());
            for &ia in atmlst {
                if ia >= natm {
                    return Err(invalid(format!(
                        "gamma RHF gradient: atmlst id {ia} out of range for {natm} atoms"
                    )));
                }
                selected.push(de[ia]);
            }
            return Ok(selected);
        }
        Ok(de)
    }

    /// `kernel`: electronic plus nuclear parts, mirroring upstream's
    /// `Gradients.kernel()` (`pyscf/grad/rhf.py`, via `pbc/grad/rhf.py`'s
    /// `grad_nuc` override).
    pub fn kernel(&self) -> Result<Gradient, PyscfRsError> {
        let elec = self.electronic_gradient()?;
        let nuc = self.nuclear_gradient()?;
        if elec.len() != nuc.len() {
            return Err(invalid(
                "gamma RHF gradient: electronic and nuclear parts disagree on atom count",
            ));
        }
        Ok(elec
            .into_iter()
            .zip(nuc)
            .map(|(e, n)| {
                [
                    oracle_sum(&[e[0], n[0]]),
                    oracle_sum(&[e[1], n[1]]),
                    oracle_sum(&[e[2], n[2]]),
                ]
            })
            .collect())
    }
}

impl<'a> Gradients for GammaRhfGradients<'a> {
    fn cell(&self) -> &Cell {
        self.cell
    }

    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }

    fn grad_elec(&self) -> Result<Gradient, PyscfRsError> {
        self.electronic_gradient()
    }

    fn grad_nuc(&self) -> Result<Gradient, PyscfRsError> {
        self.nuclear_gradient()
    }

    fn make_rdm1e(&self) -> Result<pyscf_pbc_scf::types::KDms, PyscfRsError> {
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "gamma RHF make_rdm1e from SCF orbitals (pass dme0 explicitly; see gamma_make_rdm1e)",
        }
        .into())
    }
}

/// Energy-weighted density at gamma — the `mol_rhf.Gradients.make_rdm1e`
/// line `rhf.py:187` delegates to.
///
/// `mo_coeff` is column-major `nao × nmo` (this port's convention,
/// `pyscf-pbc-scf/src/types.rs:119`); `mo_energy`/`mo_occ` are per-orbital.
/// `dme[i,j] = Σ_m C[i,m]·e[m]·n[m]·C[j,m]`, symmetric at gamma — the index
/// order upstream's `einsum('xkij,kji->x', …)` (`18-CONTEXT` trap 9) folds
/// into, documented here so the contraction convention cannot drift.
pub fn gamma_make_rdm1e(
    mo_coeff_col_major: &[f64],
    mo_energy: &[f64],
    mo_occ: &[f64],
    nao: usize,
) -> Result<Vec<f64>, PyscfRsError> {
    let nmo = mo_occ.len();
    if mo_energy.len() != nmo
        || mo_coeff_col_major.len()
            != nao
                .checked_mul(nmo)
                .ok_or_else(|| invalid("gamma_make_rdm1e: orbital count overflow"))?
    {
        return Err(invalid(
            "gamma_make_rdm1e: mo_coeff/mo_energy/mo_occ shapes disagree",
        ));
    }
    if mo_energy.iter().chain(mo_occ).any(|x| !x.is_finite())
        || mo_coeff_col_major.iter().any(|x| !x.is_finite())
    {
        return Err(invalid("gamma_make_rdm1e: orbitals must be finite"));
    }
    let at = |i: usize, m: usize| mo_coeff_col_major[i + m * nao];
    let mut dme = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            let terms: Vec<f64> = (0..nmo)
                .map(|m| at(i, m) * mo_energy[m] * mo_occ[m] * at(j, m))
                .collect();
            dme[i * nao + j] = oracle_sum(&terms);
        }
    }
    Ok(dme)
}

/// `grad/rks.py:22` — `class Gradients(rhf.Gradients)` with a
/// `pass`-equivalent body (29 l upstream).
///
/// Shipped as a re-export, not a body: the KS gamma gradient IS the RHF
/// assembly with an `xc_code` carried through `get_veff` (`rhf.py:140`).
/// (`get_stress` defers to 18-12/18-20's `rks_stress`.)
pub type GammaRksGradients<'a> = GammaRhfGradients<'a>;
