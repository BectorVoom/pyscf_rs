//! `KROKS` — restricted OPEN-shell Kohn-Sham with k-point sampling (plan 12-04).
//!
//! Port of `pyscf/pbc/dft/kroks.py`. Upstream's whole file is
//!
//! ```text
//! KROKS.get_veff     = kuks.get_veff   (on the ROHF alpha/beta densities)
//! KROKS.energy_elec  = kuks.energy_elec
//! everything else    = KROHF's
//! ```
//!
//! so this port COMPOSES rather than copies: it owns a
//! [`pyscf_pbc_scf::Krohf`] and forwards every hook except `get_veff` and
//! `energy_elec` to it. The Roothaan effective Fock, the `nfock = 1` vs
//! `nset = 2` split, the ROHF occupation rule and the three-block ROHF gradient
//! are therefore literally the tested Phase-11 code, not a second copy of it.

use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_df::{Fftdf, PeriodicDf};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_scf::Krohf;
use pyscf_pbc_scf::khooks::KOverrideHooks;
use pyscf_pbc_scf::kscf::kernel;
use pyscf_pbc_scf::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};

use crate::error::PbcDftError;
use crate::gen_grid::PeriodicGrids;
use crate::krks::{KsEnergyTags, unwrap_err};
use crate::kuks::Kuks;
use crate::numint::KNumInt;

/// Restricted open-shell periodic Kohn-Sham.
#[derive(Debug)]
pub struct Kroks {
    /// The Hartree-Fock half — owns the density fitting, the cell and the
    /// k-points, and supplies every non-KS hook.
    pub hf: Krohf,
    /// The XC functional string.
    pub xc: String,
    /// The integration grid.
    pub grids: PeriodicGrids,
    /// The numerical-integration driver.
    pub ni: KNumInt,
    tags: std::cell::Cell<Option<KsEnergyTags>>,
}

impl Kroks {
    /// Build a `KROKS` with the default `FFTDF` and the uniform grid.
    ///
    /// # Errors
    /// Propagates the `FFTDF` and grid construction.
    pub fn new(cell: Cell, kpts: &[[f64; 3]], xc: &str) -> Result<Self, PbcDftError> {
        let with_df = Fftdf::new(cell, kpts).map_err(|e| {
            crate::xc::err(format!("KROKS: FFTDF construction failed: {e}"))
        })?;
        Self::from_df(Box::new(with_df), xc)
    }

    /// `KROKS` over an explicit density-fitting object.
    ///
    /// # Errors
    /// Propagates the grid construction.
    pub fn from_df(with_df: Box<dyn PeriodicDf>, xc: &str) -> Result<Self, PbcDftError> {
        let grids = PeriodicGrids::uniform(with_df.cell(), Some(with_df.mesh()))?;
        let ni = KNumInt::new(with_df.kpts());
        Ok(Self {
            hf: Krohf::from_df(with_df),
            xc: xc.to_string(),
            grids,
            ni,
            tags: std::cell::Cell::new(None),
        })
    }

    /// The cell.
    pub fn cell(&self) -> &Cell {
        self.hf.cell()
    }

    /// The k-points.
    pub fn kpts(&self) -> &[[f64; 3]] {
        self.hf.kpts()
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
        kernel(self, cfg)
    }

    /// Run with cell-derived defaults.
    ///
    /// # Errors
    /// As [`Kroks::kernel`].
    pub fn run(&self) -> Result<KScfResult, PyscfRsError> {
        self.kernel(&KScfConfig::for_cell(self.cell()))
    }

    /// `get_veff` — `kroks.py:32-41` delegating to `kuks.get_veff`.
    ///
    /// # Errors
    /// Propagates the grid loop and the J/K build.
    pub fn get_veff_tagged(
        &self,
        dms: &KDms,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<(KDms, KsEnergyTags), PbcDftError> {
        // The unrestricted KS `get_veff` IS `KROKS`'s; borrowing it here rather
        // than duplicating it keeps the two in lockstep. `Kuks` is constructed
        // as a thin view over the same density-fitting object.
        Kuks::veff_from_parts(
            self.hf.with_df.as_ref(),
            &self.xc,
            self.hf.exxdiv,
            &self.grids,
            &self.ni,
            dms,
            kpts_band,
        )
    }
}

impl KOverrideHooks for Kroks {
    fn cell(&self) -> &Cell {
        self.hf.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.hf.kpts()
    }
    fn nset(&self) -> usize {
        2
    }
    fn nfock(&self) -> usize {
        1
    }

    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        self.hf.get_ovlp()
    }

    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        self.hf.get_hcore()
    }

    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        self.hf.get_init_guess(mode, s1e)
    }

    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let (v, tags) = self.get_veff_tagged(dms, None).map_err(unwrap_err)?;
        self.tags.set(Some(tags));
        Ok(v)
    }

    fn get_fock(&self, h1e: &KMats, vhf: &KDms, dms: &KDms) -> Result<KDms, PyscfRsError> {
        self.hf.get_fock(h1e, vhf, dms)
    }

    fn diis_dms(&self, dms: &KDms) -> KDms {
        self.hf.diis_dms(dms)
    }

    fn eig(
        &self,
        fock: &KDms,
        s1e: &KMats,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        self.hf.eig(fock, s1e)
    }

    fn get_occ(
        &self,
        mo_energy: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        self.hf.get_occ(mo_energy)
    }

    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        self.hf.make_rdm1(mo_coeff, mo_occ)
    }

    fn get_grad(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
        h1e: &KMats,
        vhf: &KDms,
    ) -> Vec<f64> {
        self.hf.get_grad(mo_coeff, mo_occ, h1e, vhf)
    }

    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        // kroks.py imports `energy_elec` straight from `kuks`.
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
        let weight = 1.0 / h1e.len() as f64;
        let mut e1 = 0.0_f64;
        for set in dms.iter() {
            for (k, h) in h1e.iter().enumerate() {
                e1 += pyscf_pbc_scf::krdm::trace_ab(&set[k], h, nao).0;
            }
        }
        e1 *= weight;
        Ok((e1 + tags.ecoul + tags.exc, tags.ecoul + tags.exc))
    }
}
