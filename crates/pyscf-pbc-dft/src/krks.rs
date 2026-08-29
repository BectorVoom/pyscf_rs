//! `KRKS` — restricted Kohn-Sham with k-point sampling (plan 12-03).
//!
//! Port of `pyscf/pbc/dft/krks.py:37-140` (`get_veff`, `get_rho`,
//! `energy_elec`) and `:249-292` (the class).
//!
//! ```text
//! nset      = 1
//! get_veff  = Vxc + J − 0.5·hyb·K                        (krks.py:88-100)
//! ecoul     = 0.5 · (1/N_k) Σ_k Tr(D^k J^k)              (krks.py:96)
//! exc       = ∫ ε_xc ρ  −  0.25 · (1/N_k) Σ_k Tr(D^k K^k)  (krks.py:99)
//! energy    = e1 + ecoul + exc                           (krks.py:115-128)
//! ```
//!
//! Everything else — the occupation rule, the density build, DIIS, the
//! eigenproblem — is `KRHF`'s and reaches the driver through the same
//! [`KOverrideHooks`] surface.
//!
//! # Where the KS energy components live
//!
//! Upstream returns `veff` as a `lib.tag_array` carrying `ecoul` and `exc`, and
//! `energy_elec` reads them off it. Rust has no attribute-tagged arrays, so the
//! pair is cached in an interior-mutable cell that `get_veff` writes and
//! `energy_elec` reads. The SCF driver calls the two back-to-back on the same
//! density ([`pyscf_pbc_scf::kernel`]), which is exactly upstream's contract.

use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_df::{Fftdf, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::khooks::KOverrideHooks;
use pyscf_pbc_scf::kocc::get_occ_restricted;
use pyscf_pbc_scf::krdm::make_rdm1;
use pyscf_pbc_scf::krhf::{df_err, eig_channel, to_row_major};
use pyscf_pbc_scf::kscf::kernel;
use pyscf_pbc_scf::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};

use crate::error::PbcDftError;
use crate::gen_grid::PeriodicGrids;
use crate::numint::KNumInt;
use crate::veff::{get_jk, sub_scaled, trace_dm_v};
use crate::xc::{err, is_hybrid_xc};

/// The energy components `get_veff` hands to `energy_elec` — upstream's
/// `lib.tag_array(vxc, ecoul=..., exc=...)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct KsEnergyTags {
    /// `E_coul` — the Coulomb energy.
    pub ecoul: f64,
    /// `E_xc` — the exchange-correlation energy, INCLUDING the exact-exchange
    /// share of a hybrid.
    pub exc: f64,
    /// `∫ ρ` from the numerical integration, for the electron-count diagnostic.
    pub nelec: f64,
}

/// Restricted periodic Kohn-Sham.
#[derive(Debug)]
pub struct Krks {
    /// The density-fitting object; it owns the cell and the k-points.
    pub with_df: Box<dyn PeriodicDf>,
    /// The XC functional string. Upstream's class default is `'LDA,VWN'`.
    pub xc: String,
    /// Exchange divergence treatment (`pbc_scf_SCF_exxdiv`, default
    /// `ExxDiv::Ewald`).
    pub exxdiv: Option<ExxDiv>,
    /// The integration grid. Upstream's default is the uniform FFT box.
    pub grids: PeriodicGrids,
    /// The numerical-integration driver.
    pub ni: KNumInt,
    /// Smearing, when enabled.
    pub smearing: Option<pyscf_pbc_scf::smearing::Smearing>,
    tags: std::cell::Cell<Option<KsEnergyTags>>,
    entropy: std::cell::Cell<Option<f64>>,
}

impl Krks {
    /// Build a `KRKS` for `cell` at `kpts` (empty = gamma) with the default
    /// `FFTDF` and the uniform grid on `cell.mesh`.
    ///
    /// # Errors
    /// Propagates the `FFTDF` and grid construction.
    pub fn new(cell: Cell, kpts: &[[f64; 3]], xc: &str) -> Result<Self, PbcDftError> {
        let with_df = Fftdf::new(cell, kpts)
            .map_err(|e| err(format!("KRKS: FFTDF construction failed: {e}")))?;
        Self::from_df(Box::new(with_df), xc)
    }

    /// `KRKS` over an explicitly configured density-fitting object.
    ///
    /// # Errors
    /// Propagates the grid construction.
    pub fn from_df(with_df: Box<dyn PeriodicDf>, xc: &str) -> Result<Self, PbcDftError> {
        let grids = PeriodicGrids::uniform(with_df.cell(), Some(with_df.mesh()))?;
        let ni = KNumInt::new(with_df.kpts());
        Ok(Self {
            with_df,
            xc: xc.to_string(),
            exxdiv: Some(ExxDiv::Ewald),
            grids,
            ni,
            smearing: None,
            tags: std::cell::Cell::new(None),
            entropy: std::cell::Cell::new(None),
        })
    }

    /// The cell.
    pub fn cell(&self) -> &Cell {
        self.with_df.cell()
    }

    /// The k-points.
    pub fn kpts(&self) -> &[[f64; 3]] {
        self.with_df.kpts()
    }

    /// Electrons in the whole BZ supercell.
    pub fn nelectron(&self) -> usize {
        self.cell().tot_electrons(self.kpts().len())
    }

    /// The energy components of the last `get_veff`.
    pub fn last_tags(&self) -> Option<KsEnergyTags> {
        self.tags.get()
    }

    /// Run the SCF.
    ///
    /// # Errors
    /// Propagates every hook and the driver.
    pub fn kernel(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        self.tags.set(None);
        self.entropy.set(None);
        kernel(self, cfg)
    }

    /// Run with the cell-derived default settings.
    ///
    /// # Errors
    /// As [`Krks::kernel`].
    pub fn run(&self) -> Result<KScfResult, PyscfRsError> {
        self.kernel(&KScfConfig::for_cell(self.cell()))
    }

    /// `get_rho(mf, dm, grids, kpts)` — `krks.py:105-112`.
    ///
    /// # Errors
    /// Propagates the AO evaluation.
    pub fn get_rho(&self, dms: &KMats) -> Result<Vec<f64>, PbcDftError> {
        self.ni.get_rho(self.cell(), dms, &self.grids)
    }

    /// `get_veff` proper, returning BOTH the potential and the energy
    /// components — `krks.py:37-103`.
    ///
    /// # Errors
    /// Propagates the grid loop and the J/K build.
    pub fn get_veff_tagged(
        &self,
        dms: &KDms,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<(KDms, KsEnergyTags), PbcDftError> {
        let cell = self.cell();
        let nao = cell.mol.nao_nr;
        let nkpts = self.kpts().len();
        let weight = 1.0 / nkpts as f64;
        let ground_state = kpts_band.is_none();

        // krks.py:71-72 — the XC half.
        let nr = self
            .ni
            .nr_rks(cell, &self.grids, &self.xc, dms, 1, kpts_band)?;
        let mut vxc = nr.vmat;
        let mut exc = nr.excsum[0];

        // krks.py:89 — J (and K for a hybrid).
        let jk = get_jk(
            self.with_df.as_ref(),
            &self.xc,
            dms,
            1,
            self.kpts(),
            kpts_band,
            self.exxdiv,
            true,
        )?;
        let vj = jk
            .vj
            .ok_or_else(|| err("KRKS: the density-fitting object returned no vj"))?;
        crate::veff::add_assign(&mut vxc, &vj);
        // krks.py:96 — ecoul = einsum('Kij,Kji', dm, vj) * .5 * weight
        let ecoul = if ground_state {
            0.5 * weight * trace_dm_v(dms, &vj, nao).0
        } else {
            0.0
        };

        if let Some(vk) = jk.vk.as_ref() {
            // krks.py:98 — vxc -= .5 * vk
            sub_scaled(&mut vxc, 0.5, vk);
            if ground_state {
                // krks.py:100 — exc -= einsum('Kij,Kji', dm, vk).real * .25 * weight
                exc -= 0.25 * weight * trace_dm_v(dms, vk, nao).0;
            }
        }

        Ok((
            vxc,
            KsEnergyTags {
                ecoul,
                exc,
                nelec: nr.nelec[0],
            },
        ))
    }

    /// `get_bands(kpts_band, dm_kpts)` — the KS analogue of
    /// `pyscf_pbc_scf::Krhf::get_bands`.
    ///
    /// # Errors
    /// Propagates the integrals and the generalized eigensolve.
    pub fn get_bands(
        &self,
        kpts_band: &[[f64; 3]],
        dms: &KDms,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        let mut fock =
            pyscf_pbc_df::get_hcore(self.with_df.as_ref(), kpts_band).map_err(df_err)?;
        let (veff, _) = self
            .get_veff_tagged(dms, Some(kpts_band))
            .map_err(unwrap_err)?;
        for (k, f) in fock.iter_mut().enumerate() {
            for i in 0..f.len() {
                f.re[i] += veff[0][k].re[i];
                f.im[i] += veff[0][k].im[i];
            }
        }
        let s1e = to_row_major(pyscf_pbc_gto::get_ovlp(self.cell(), kpts_band)?, nao);
        eig_channel(&fock, &s1e, nao)
    }
}

/// Flatten a [`PbcDftError`] back into the core error type the hook surface
/// speaks.
pub(crate) fn unwrap_err(e: PbcDftError) -> PyscfRsError {
    match e {
        PbcDftError::Core(c) => c,
    }
}

impl KOverrideHooks for Krks {
    fn cell(&self) -> &Cell {
        self.with_df.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.with_df.kpts()
    }
    fn nset(&self) -> usize {
        1
    }

    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        Ok(to_row_major(
            pyscf_pbc_gto::get_ovlp(self.cell(), self.kpts())?,
            nao,
        ))
    }

    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        pyscf_pbc_df::get_hcore(self.with_df.as_ref(), self.kpts()).map_err(df_err)
    }

    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        pyscf_pbc_scf::init_guess::get_init_guess(
            self.cell(),
            self.kpts().len(),
            1,
            mode,
            s1e,
            self.nelectron() as f64,
        )
    }

    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let (v, tags) = self.get_veff_tagged(dms, None).map_err(unwrap_err)?;
        self.tags.set(Some(tags));
        Ok(v)
    }

    fn eig(
        &self,
        fock: &KDms,
        s1e: &KMats,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        eig_channel(&fock[0], s1e, self.cell().mol.nao_nr)
    }

    fn get_occ(
        &self,
        mo_energy: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        if let Some(sm) = self.smearing.as_ref() {
            let (occ, fermi, entropy) = sm.occupations(mo_energy, self.nelectron() as f64, 2.0)?;
            self.entropy.set(Some(entropy));
            return Ok((occ, vec![fermi]));
        }
        let (occ, fermi) = get_occ_restricted(mo_energy, self.nelectron() / 2)?;
        Ok((occ, vec![fermi]))
    }

    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        Ok(vec![make_rdm1(mo_coeff, mo_occ, self.cell().mol.nao_nr)])
    }

    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        // krks.py:115-128 — e1 + ecoul + exc, with the components taken from
        // the `get_veff` that produced `vhf`.
        let tags = match self.tags.get() {
            Some(t) => t,
            None => {
                let (_, t) = self.get_veff_tagged(dms, None).map_err(unwrap_err)?;
                self.tags.set(Some(t));
                t
            }
        };
        let _ = vhf;
        let nao = self.cell().mol.nao_nr;
        let nkpts = h1e.len();
        let weight = 1.0 / nkpts as f64;
        let mut e1 = 0.0_f64;
        for (k, h) in h1e.iter().enumerate() {
            e1 += pyscf_pbc_scf::krdm::trace_ab(&dms[0][k], h, nao).0;
        }
        e1 *= weight;
        Ok((e1 + tags.ecoul + tags.exc, tags.ecoul + tags.exc))
    }

    fn free_energy(&self) -> Option<f64> {
        self.entropy
            .get()
            .map(|s| -self.smearing.as_ref().map_or(0.0, |sm| sm.sigma) * s)
    }
}

/// `is_hybrid_xc(xc)` re-exported at the driver level — a caller often wants it
/// before deciding whether `_j_only` density fitting is admissible.
///
/// # Errors
/// Propagates the XC-string parse.
pub fn hybrid(xc_code: &str) -> Result<bool, PbcDftError> {
    is_hybrid_xc(xc_code)
}
