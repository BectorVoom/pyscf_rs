//! `KRHF` — restricted Hartree-Fock with k-point sampling (`khf.py:789-864`).
//!
//! A thin struct over [`crate::kscf::kernel`]: the loop is shared, only the
//! eleven hooks are RHF-specific.
//!
//! ```text
//! nset      = 1
//! get_veff  = vj - vk/2                     (khf.py:624-633)
//! get_occ   = global aufbau, occupancy 2    (khf.py:184-225)
//! energy    = e1 + e_coul                   (khf.py:249-268)
//! ```

use pyscf_algebra::{CTensor, zeigh_gen};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_df::zlinalg::forder_to_c;
use pyscf_pbc_df::{Fftdf, JkOpts, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};

use crate::khooks::KOverrideHooks;
use crate::kocc::get_occ_restricted;
use crate::krdm::{energy_elec, make_rdm1};
use crate::kscf::kernel;
use crate::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};

/// Restricted periodic Hartree-Fock.
#[derive(Debug)]
pub struct Krhf {
    /// The density-fitting object; it owns the cell and the k-points.
    pub with_df: Box<dyn PeriodicDf>,
    /// Exchange divergence treatment. Upstream's default is
    /// `ExxDiv::Ewald` (`pbc_scf_SCF_exxdiv`).
    pub exxdiv: Option<ExxDiv>,
    /// Smearing, when enabled by [`crate::smearing::Smearing`].
    pub smearing: Option<crate::smearing::Smearing>,
    /// Filled by [`Krhf::kernel`] when smearing is on.
    entropy: std::cell::Cell<Option<f64>>,
}

impl Krhf {
    /// Build a `KRHF` for `cell` at `kpts` (empty = gamma) with the default
    /// `FFTDF` on `cell.mesh`.
    ///
    /// # Errors
    /// Propagates the `FFTDF` construction.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Result<Self, PyscfRsError> {
        let with_df = Fftdf::new(cell, kpts).map_err(df_err)?;
        Ok(Self::from_df(Box::new(with_df)))
    }

    /// `KRHF` over an explicitly configured density-fitting object — the seam
    /// AFTDF/GDF plug into in Phases 13/14.
    pub fn from_df(with_df: Box<dyn PeriodicDf>) -> Self {
        Self {
            with_df,
            exxdiv: Some(ExxDiv::Ewald),
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

    /// Electrons in the whole BZ supercell: `cell.tot_electrons(nkpts)`.
    pub fn nelectron(&self) -> usize {
        self.cell().tot_electrons(self.kpts().len())
    }

    /// Run the SCF.
    ///
    /// # Errors
    /// Propagates every hook and the driver.
    pub fn kernel(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        self.entropy.set(None);
        kernel(self, cfg)
    }

    /// Run with the cell-derived default settings (`conv_tol = max(precision*10, 1e-8)`).
    ///
    /// # Errors
    /// As [`Krhf::kernel`].
    pub fn run(&self) -> Result<KScfResult, PyscfRsError> {
        let cfg = KScfConfig::for_cell(self.cell());
        self.kernel(&cfg)
    }

    /// `get_bands(kpts_band, dm_kpts)` — `khf.py:670-695`.
    ///
    /// Band energies and orbitals at ARBITRARY k-points, from a converged
    /// density. The Fock matrix at a band point is `hcore(k_band) + veff` with
    /// `veff` built from the SAMPLING k-points' density but evaluated at the
    /// band points — which is what `JkOpts::kpts_band` selects.
    ///
    /// Returns `(mo_energy, mo_coeff)`, one block per band k-point;
    /// `mo_coeff` is COLUMN-MAJOR.
    ///
    /// # Errors
    /// Propagates the integrals and the generalized eigensolve.
    pub fn get_bands(
        &self,
        kpts_band: &[[f64; 3]],
        dm_kpts: &KDms,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        let mut fock = pyscf_pbc_df::get_hcore(self.with_df.as_ref(), kpts_band).map_err(df_err)?;
        let r = self
            .with_df
            .get_jk(
                dm_kpts,
                self.kpts(),
                JkOpts {
                    hermi: 1,
                    kpts_band: Some(kpts_band),
                    with_j: true,
                    with_k: true,
                    exxdiv: self.exxdiv,
                    omega: None,
                    // W-08 — opt-in, off unless `PYSCF_PBC_KK_SYMMETRY` says
                    // otherwise. See `JkOpts::kk_symmetry_default`.
                    kk_symmetry: JkOpts::kk_symmetry_default(),
                },
            )
            .map_err(df_err)?;
        let vj = r.vj.ok_or_else(|| missing("vj"))?;
        let vk = r.vk.ok_or_else(|| missing("vk"))?;
        for (k, f) in fock.iter_mut().enumerate() {
            for i in 0..f.len() {
                f.re[i] += vj[0][k].re[i] - 0.5 * vk[0][k].re[i];
                f.im[i] += vj[0][k].im[i] - 0.5 * vk[0][k].im[i];
            }
        }
        let s1e = to_row_major(
            pyscf_pbc_gto::get_ovlp(self.cell(), kpts_band)?,
            nao,
        );
        eig_channel(&fock, &s1e, nao)
    }
}

pub fn df_err(e: pyscf_pbc_df::PbcDfError) -> PyscfRsError {
    match e {
        pyscf_pbc_df::PbcDfError::Core(c) => c,
        other => PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "periodic SCF: density fitting failed: {other}"
        ))),
    }
}

/// Convert Phase 10's F-order k-matrices to the row-major convention Phase 11
/// works in.
pub fn to_row_major(mats: Vec<CTensor>, nao: usize) -> KMats {
    mats.iter().map(|m| forder_to_c(m, nao, nao)).collect()
}

/// `eig(h_kpts, s_kpts)` for one channel — `khf.py:645-654`.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when the generalized eigenproblem fails.
pub fn eig_channel(
    fock: &KMats,
    s1e: &KMats,
    nao: usize,
) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
    let mut es = Vec::with_capacity(fock.len());
    let mut cs = Vec::with_capacity(fock.len());
    for (k, f) in fock.iter().enumerate() {
        let (e, c) = zeigh_gen(f, &s1e[k], nao).map_err(|err| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "periodic SCF: zeigh_gen failed at k = {k}: {err}"
            )))
        })?;
        es.push(e);
        cs.push(c);
    }
    Ok((es, cs))
}

impl KOverrideHooks for Krhf {
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
        crate::init_guess::get_init_guess(
            self.cell(),
            self.kpts().len(),
            1,
            mode,
            s1e,
            &[self.nelectron() as f64],
            0,
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
                    // W-08 — opt-in, off unless `PYSCF_PBC_KK_SYMMETRY` says
                    // otherwise. See `JkOpts::kk_symmetry_default`.
                    kk_symmetry: JkOpts::kk_symmetry_default(),
                },
            )
            .map_err(df_err)?;
        let vj = r.vj.ok_or_else(|| missing("vj"))?;
        let vk = r.vk.ok_or_else(|| missing("vk"))?;
        // khf.py:632 — vhf = vj - vk * .5
        let mut out = vj;
        for (s, set) in out.iter_mut().enumerate() {
            for (k, m) in set.iter_mut().enumerate() {
                for i in 0..m.len() {
                    m.re[i] -= 0.5 * vk[s][k].re[i];
                    m.im[i] -= 0.5 * vk[s][k].im[i];
                }
            }
        }
        Ok(out)
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
            let (occ, fermi, entropy) =
                sm.occupations(mo_energy, self.nelectron() as f64, 2.0)?;
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
        Ok(energy_elec(dms, h1e, vhf, self.cell().mol.nao_nr))
    }

    fn get_grad(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
        h1e: &KMats,
        vhf: &KDms,
    ) -> Vec<f64> {
        let nao = self.cell().mol.nao_nr;
        let fock = crate::kscf::bare_fock(h1e, vhf);
        if self.smearing.is_none() {
            let mut g = Vec::new();
            for (k, f) in fock[0].iter().enumerate() {
                g.extend_from_slice(&crate::kocc::get_grad(&mo_coeff[k], &mo_occ[k], f, nao));
            }
            return g;
        }
        // Smeared occupations: the strict lower triangle of the full MO Fock,
        // because the occupied-virtual split no longer separates the
        // stationary conditions (pbc/scf/smearing.py:25-31).
        let mut g = Vec::new();
        for (k, f) in fock[0].iter().enumerate() {
            g.extend_from_slice(&crate::smearing::grad_tril(
                &mo_coeff[k],
                f,
                nao,
                mo_occ[k].len(),
            ));
        }
        g
    }

    fn free_energy(&self) -> Option<f64> {
        self.entropy.get().map(|s| {
            -self.smearing.as_ref().map_or(0.0, |sm| sm.sigma) * s
        })
    }
}

fn missing(what: &str) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "periodic SCF: the density-fitting object returned no {what}"
    )))
}
