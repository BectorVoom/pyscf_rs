//! `KUHF` — unrestricted Hartree-Fock with k-point sampling (`kuhf.py`).
//!
//! ```text
//! nset      = 2
//! get_veff  = vj[a] + vj[b] - vk[s]         (kuhf.py:488-495)
//! get_occ   = TWO global Fermi levels        (kuhf.py:136-204)
//! nelec     = ((ne + spin)/2, that - spin)   (kuhf.py:442-458)
//! ```
//!
//! Note the electron count: `ne` is the BZ-SUPERCELL total
//! (`cell.tot_electrons(nkpts)`) while `cell.spin` is PER CELL — upstream mixes
//! the two scales deliberately (`kuhf.py:447-450`), because the spin
//! polarisation is a property of the cell and is not multiplied by the k-mesh.

use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_df::{Fftdf, JkOpts, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};

use crate::khooks::KOverrideHooks;
use crate::kocc::get_occ_unrestricted;
use crate::krdm::{energy_elec, make_rdm1};
use crate::krhf::{df_err, eig_channel, to_row_major};
use crate::kscf::kernel;
use crate::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};

/// Unrestricted periodic Hartree-Fock.
#[derive(Debug)]
pub struct Kuhf {
    /// The density-fitting object.
    pub with_df: Box<dyn PeriodicDf>,
    /// Exchange divergence treatment.
    pub exxdiv: Option<ExxDiv>,
    /// Override the `(nalpha, nbeta)` derived from `cell.spin`.
    pub nelec: Option<(usize, usize)>,
    /// Smearing, when enabled.
    pub smearing: Option<crate::smearing::Smearing>,
    entropy: std::cell::Cell<Option<f64>>,
}

impl Kuhf {
    /// Build a `KUHF` with the default `FFTDF`.
    ///
    /// # Errors
    /// Propagates the `FFTDF` construction.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Result<Self, PyscfRsError> {
        Ok(Self::from_df(Box::new(Fftdf::new(cell, kpts).map_err(df_err)?)))
    }

    /// `KUHF` over an explicit density-fitting object.
    pub fn from_df(with_df: Box<dyn PeriodicDf>) -> Self {
        Self {
            with_df,
            exxdiv: Some(ExxDiv::Ewald),
            nelec: None,
            smearing: None,
            entropy: std::cell::Cell::new(None),
        }
    }

    /// The cell.
    pub fn cell(&self) -> &Cell {
        self.with_df.cell()
    }

    /// The k-points.
    pub fn kpts(&self) -> &[[f64; 3]] {
        self.with_df.kpts()
    }

    /// `(nalpha, nbeta)` over the Brillouin-zone supercell — `kuhf.py:442-458`.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] when the electron count and `cell.spin`
    /// are inconsistent — upstream raises the same `RuntimeError`.
    pub fn nelec(&self) -> Result<(usize, usize), PyscfRsError> {
        if let Some(n) = self.nelec {
            return Ok(n);
        }
        let cell = self.cell();
        let ne = cell.tot_electrons(self.kpts().len()) as i64;
        let spin = cell.mol.spin as i64;
        let nalpha = (ne + spin) / 2;
        let nbeta = nalpha - spin;
        if nalpha + nbeta != ne || nalpha < 0 || nbeta < 0 {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "KUHF: electron number {ne} and spin {spin} are not consistent \
                 (cell.spin = 2S = Nalpha - Nbeta, not 2S+1)"
            ))));
        }
        Ok((nalpha as usize, nbeta as usize))
    }

    /// Run the SCF.
    ///
    /// # Errors
    /// Propagates every hook and the driver.
    pub fn kernel(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        self.entropy.set(None);
        kernel(self, cfg)
    }

    /// Run with cell-derived defaults.
    ///
    /// # Errors
    /// As [`Kuhf::kernel`].
    pub fn run(&self) -> Result<KScfResult, PyscfRsError> {
        self.kernel(&KScfConfig::for_cell(self.cell()))
    }
}

impl KOverrideHooks for Kuhf {
    fn cell(&self) -> &Cell {
        self.with_df.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.with_df.kpts()
    }
    fn nset(&self) -> usize {
        2
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
        let (na, nb) = self.nelec()?;
        crate::init_guess::get_init_guess(
            self.cell(),
            self.kpts().len(),
            2,
            mode,
            s1e,
            (na + nb) as f64,
        )
    }

    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let r = self
            .with_df
            .get_jk(
                dms,
                self.kpts(),
                JkOpts {
                    hermi: 1,
                    kpts_band: None,
                    with_j: true,
                    with_k: true,
                    exxdiv: self.exxdiv,
                    omega: None,
                },
            )
            .map_err(df_err)?;
        let vj = r.vj.ok_or_else(|| missing("vj"))?;
        let vk = r.vk.ok_or_else(|| missing("vk"))?;
        // kuhf.py:494 — vhf[s] = vj[a] + vj[b] - vk[s]
        let nkpts = self.kpts().len();
        let mut out: KDms = Vec::with_capacity(2);
        for s in 0..2 {
            let mut set = Vec::with_capacity(nkpts);
            for k in 0..nkpts {
                let mut m = vj[0][k].clone();
                for i in 0..m.len() {
                    m.re[i] += vj[1][k].re[i] - vk[s][k].re[i];
                    m.im[i] += vj[1][k].im[i] - vk[s][k].im[i];
                }
                set.push(m);
            }
            out.push(set);
        }
        Ok(out)
    }

    fn eig(
        &self,
        fock: &KDms,
        s1e: &KMats,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        let (mut e, mut c) = eig_channel(&fock[0], s1e, nao)?;
        let (eb, cb) = eig_channel(&fock[1], s1e, nao)?;
        e.extend(eb);
        c.extend(cb);
        Ok((e, c))
    }

    fn get_occ(
        &self,
        mo_energy: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        let nkpts = self.kpts().len();
        let (na, nb) = self.nelec()?;
        let (ea, eb) = mo_energy.split_at(nkpts);
        if let Some(sm) = self.smearing.as_ref() {
            // pbc/scf/smearing.py:105-125 without `fix_spin`: BOTH channels
            // share one chemical potential, so they are pooled.
            let pooled: Vec<Vec<f64>> = mo_energy.to_vec();
            let (occ, fermi, entropy) = sm.occupations(&pooled, (na + nb) as f64, 1.0)?;
            self.entropy.set(Some(entropy));
            return Ok((occ, vec![fermi, fermi]));
        }
        let (occ_a, occ_b, fermi) = get_occ_unrestricted(ea, eb, na, nb)?;
        let mut occ = occ_a;
        occ.extend(occ_b);
        Ok((occ, fermi.to_vec()))
    }

    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        let nkpts = self.kpts().len();
        Ok(vec![
            make_rdm1(&mo_coeff[..nkpts], &mo_occ[..nkpts], nao),
            make_rdm1(&mo_coeff[nkpts..], &mo_occ[nkpts..], nao),
        ])
    }

    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        Ok(energy_elec(dms, h1e, vhf, self.cell().mol.nao_nr))
    }

    fn free_energy(&self) -> Option<f64> {
        self.entropy
            .get()
            .map(|s| -self.smearing.as_ref().map_or(0.0, |sm| sm.sigma) * s)
    }
}

fn missing(what: &str) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "KUHF: the density-fitting object returned no {what}"
    )))
}
