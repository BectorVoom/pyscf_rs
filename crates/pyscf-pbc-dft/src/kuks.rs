//! `KUKS` — unrestricted Kohn-Sham with k-point sampling (plan 12-04).
//!
//! Port of `pyscf/pbc/dft/kuks.py:38-113` (`get_veff`, `energy_elec`) and
//! `:153-200` (the class).
//!
//! ```text
//! nset      = 2
//! J         = vj[a] + vj[b]                              (kuks.py:82)
//! get_veff  = Vxc[s] + J − hyb·K[s]                      (kuks.py:83, :89)
//! ecoul     = 0.5 · (1/N_k) Σ_{s,k} Tr(D^{s,k} J^k)      (kuks.py:87)
//! exc       = ∫ ε_xc ρ − 0.5 · (1/N_k) Σ_{s,k} Tr(D^{s,k} K^{s,k})  (kuks.py:91)
//! ```
//!
//! Note the factors: `KUKS` subtracts the FULL `vk` (not `0.5·vk`) and takes
//! `0.5` of the exchange trace, because its two channels are not
//! spin-degenerate.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::PyscfRsError;
use pyscf_pbc_df::{Fftdf, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::khooks::KOverrideHooks;
use pyscf_pbc_scf::kocc::get_occ_unrestricted;
use pyscf_pbc_scf::krdm::make_rdm1;
use pyscf_pbc_scf::krhf::{df_err, eig_channel, to_row_major};
use pyscf_pbc_scf::kscf::kernel;
use pyscf_pbc_scf::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};

use crate::error::PbcDftError;
use crate::gen_grid::PeriodicGrids;
use crate::krks::{KsEnergyTags, unwrap_err};
use crate::numint::KsNumInt;
use crate::veff::{get_jk, sub_scaled, trace_dm_v, trace_dm_v_shared};
use crate::xc::err;

/// Unrestricted periodic Kohn-Sham.
#[derive(Debug)]
pub struct Kuks {
    /// The density-fitting object.
    pub with_df: Box<dyn PeriodicDf>,
    /// The XC functional string.
    pub xc: String,
    /// Exchange divergence treatment.
    pub exxdiv: Option<ExxDiv>,
    /// The integration grid.
    pub grids: PeriodicGrids,
    /// The numerical-integration driver.
    pub ni: KsNumInt,
    /// Override the `(nalpha, nbeta)` derived from `cell.spin`.
    pub nelec: Option<(usize, usize)>,
    /// Smearing, when enabled.
    pub smearing: Option<pyscf_pbc_scf::smearing::Smearing>,
    /// `init_guess_breaksym` — `uhf.py:778`, re-declared at `kuhf.py:417`.
    /// Upstream's default is **1**. `0` disables the break; `1` keeps only the
    /// intra-atomic blocks of the beta guess; `2` rescales the two channels to
    /// the doublet counts. Without it `dm_a == dm_b` is an exact fixed point of
    /// the SCF map at `cell.spin == 0` and no spin-broken solution is reachable
    /// (KUKS-OPTIMISATION-PLAN §2.2.1).
    pub init_guess_breaksym: i32,
    tags: std::cell::Cell<Option<KsEnergyTags>>,
    entropy: std::cell::Cell<Option<f64>>,
}

impl Kuks {
    /// Build a `KUKS` with the default `FFTDF` and the uniform grid.
    ///
    /// # Errors
    /// Propagates the `FFTDF` and grid construction.
    pub fn new(cell: Cell, kpts: &[[f64; 3]], xc: &str) -> Result<Self, PbcDftError> {
        let with_df = Fftdf::new(cell, kpts)
            .map_err(|e| err(format!("KUKS: FFTDF construction failed: {e}")))?;
        Self::from_df(Box::new(with_df), xc)
    }

    /// `KUKS` over an explicit density-fitting object.
    ///
    /// # Errors
    /// Propagates the grid construction.
    pub fn from_df(with_df: Box<dyn PeriodicDf>, xc: &str) -> Result<Self, PbcDftError> {
        let grids = PeriodicGrids::uniform(with_df.cell(), Some(with_df.mesh()))?;
        let ni = KsNumInt::grid(with_df.kpts());
        Ok(Self {
            with_df,
            xc: xc.to_string(),
            exxdiv: Some(ExxDiv::Ewald),
            grids,
            ni,
            nelec: None,
            smearing: None,
            init_guess_breaksym: 1,
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

    /// `(nalpha, nbeta)` over the Brillouin-zone supercell — `kuhf.py:442-458`.
    ///
    /// # Errors
    /// [`PyscfRsError`] when the electron count and `cell.spin` disagree.
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
            return Err(unwrap_err(err(format!(
                "KUKS: electron number {ne} and spin {spin} are not consistent"
            ))));
        }
        Ok((nalpha as usize, nbeta as usize))
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

    /// Run with the cell-derived defaults.
    ///
    /// # Errors
    /// As [`Kuks::kernel`].
    pub fn run(&self) -> Result<KScfResult, PyscfRsError> {
        self.kernel(&KScfConfig::for_cell(self.cell()))
    }

    /// `get_rho(mf, dm, grids, kpts)` — `kuks.py` reuses `krks.get_rho` on the
    /// SPIN-SUMMED density (`krks.py:105-112`).
    ///
    /// # Errors
    /// Propagates the AO evaluation.
    pub fn get_rho(&self, dms: &KDms) -> Result<Vec<f64>, PbcDftError> {
        let total: KMats = dms[0]
            .iter()
            .zip(&dms[1])
            .map(|(a, b)| {
                let mut m = a.clone();
                for i in 0..m.len() {
                    m.re[i] += b.re[i];
                    m.im[i] += b.im[i];
                }
                m
            })
            .collect();
        self.ni.get_rho(self.cell(), &total, &self.grids)
    }

    /// `get_bands(kpts_band, dms)` — the KS analogue of
    /// [`pyscf_pbc_scf::Kuhf::get_bands`], and the unrestricted counterpart of
    /// [`crate::Krks::get_bands`].
    ///
    /// The exchange-correlation potential is rebuilt at the band k-points from
    /// the SCF density: [`Kuks::get_veff_tagged`] passes `kpts_band` down to
    /// `nr_uks` and to the J/K build, and seeing `Some(..)` there also switches
    /// off the ground-state energy tags, which are meaningless off the mesh.
    ///
    /// # Layout
    /// Alpha for every band k-point, then beta — the split point is
    /// `kpts_band.len()`, matching this class's `eig` / `get_occ`.
    ///
    /// # Errors
    /// Propagates the grid loop, the J/K build and the generalized eigensolve.
    pub fn get_bands(
        &self,
        kpts_band: &[[f64; 3]],
        dms: &KDms,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        let hcore = pyscf_pbc_df::get_hcore(self.with_df.as_ref(), kpts_band).map_err(df_err)?;
        let (veff, _) = self
            .get_veff_tagged(dms, Some(kpts_band))
            .map_err(unwrap_err)?;

        let s1e = to_row_major(pyscf_pbc_gto::get_ovlp(self.cell(), kpts_band)?, nao);
        let mut es = Vec::with_capacity(2 * kpts_band.len());
        let mut cs = Vec::with_capacity(2 * kpts_band.len());
        for (s, channel) in veff.iter().enumerate().take(2) {
            let mut fock = hcore.clone();
            for (k, f) in fock.iter_mut().enumerate() {
                for i in 0..f.len() {
                    f.re[i] += channel[k].re[i];
                    f.im[i] += channel[k].im[i];
                }
            }
            let (e, c) = eig_channel(&fock, &s1e, nao)?;
            debug_assert_eq!(e.len(), kpts_band.len(), "channel {s} band count");
            es.extend(e);
            cs.extend(c);
        }
        Ok((es, cs))
    }

    /// `get_veff` proper — `kuks.py:38-101`.
    ///
    /// # Errors
    /// Propagates the grid loop and the J/K build.
    pub fn get_veff_tagged(
        &self,
        dms: &KDms,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<(KDms, KsEnergyTags), PbcDftError> {
        Self::veff_from_parts(
            self.with_df.as_ref(),
            &self.xc,
            self.exxdiv,
            &self.grids,
            &self.ni,
            dms,
            kpts_band,
        )
    }

    /// The body of [`Kuks::get_veff_tagged`], taken apart from `self` so that
    /// [`crate::Kroks`] — whose `get_veff` upstream imports verbatim from
    /// `kuks` (`kroks.py:32-41`) — can call the SAME code rather than carry a
    /// second copy of it.
    ///
    /// # Errors
    /// Propagates the grid loop and the J/K build.
    #[allow(clippy::too_many_arguments)]
    pub fn veff_from_parts(
        with_df: &dyn PeriodicDf,
        xc: &str,
        exxdiv: Option<ExxDiv>,
        grids: &PeriodicGrids,
        ni: &KsNumInt,
        dms: &KDms,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<(KDms, KsEnergyTags), PbcDftError> {
        let cell = with_df.cell();
        let nao = cell.mol.nao_nr;
        let nkpts = with_df.kpts().len();
        let weight = 1.0 / nkpts as f64;
        let ground_state = kpts_band.is_none();

        // kuks.py:59-60 — one density SET per spin, each carrying every k-point.
        let sets: [KDms; 2] = [vec![dms[0].clone()], vec![dms[1].clone()]];
        let nr = ni.nr_uks(cell, grids, xc, &sets, 1, with_df.kpts(), kpts_band)?;
        let mut exc = nr.exc;
        let mut vxc: KDms = vec![nr.vmat[0].clone(), nr.vmat[1].clone()];

        let ecoul = if let Some(ecoul) = nr.ecoul {
            ecoul
        } else {
            let jk = get_jk(with_df, xc, dms, 1, with_df.kpts(), kpts_band, exxdiv, true)?;
            let vj = jk
                .vj
                .ok_or_else(|| err("KUKS: the density-fitting object returned no vj"))?;
            // kuks.py:82 — vj = vj[0] + vj[1], ONE Coulomb matrix for both spins.
            let nband = vj[0].len();
            let jtot: KMats = (0..nband)
                .map(|k| {
                    let mut m = vj[0][k].clone();
                    for i in 0..m.len() {
                        m.re[i] += vj[1][k].re[i];
                        m.im[i] += vj[1][k].im[i];
                    }
                    m
                })
                .collect();
            for set in vxc.iter_mut() {
                for (k, m) in set.iter_mut().enumerate() {
                    for i in 0..m.len() {
                        m.re[i] += jtot[k].re[i];
                        m.im[i] += jtot[k].im[i];
                    }
                }
            }
            let ecoul = if ground_state {
                0.5 * weight * trace_dm_v_shared(dms, &jtot, nao).0
            } else {
                0.0
            };

            if let Some(vk) = jk.vk.as_ref() {
                sub_scaled(&mut vxc, 1.0, vk);
                if ground_state {
                    exc -= 0.5 * weight * trace_dm_v(dms, vk, nao).0;
                }
            }
            ecoul
        };

        Ok((
            vxc,
            KsEnergyTags {
                ecoul,
                exc,
                nelec: nr.nelec.0 + nr.nelec.1,
            },
        ))
    }
}

impl KOverrideHooks for Kuks {
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
        pyscf_pbc_scf::init_guess::get_init_guess(
            self.cell(),
            self.kpts().len(),
            2,
            mode,
            s1e,
            &[na as f64, nb as f64],
            self.init_guess_breaksym,
        )
    }

    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let (v, tags) = self.get_veff_tagged(dms, None).map_err(unwrap_err)?;
        self.tags.set(Some(tags));
        Ok(v)
    }

    fn eig(&self, fock: &KDms, s1e: &KMats) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        let (mut e, mut c) = eig_channel(&fock[0], s1e, nao)?;
        let (eb, cb) = eig_channel(&fock[1], s1e, nao)?;
        e.extend(eb);
        c.extend(cb);
        Ok((e, c))
    }

    fn get_occ(&self, mo_energy: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        let nkpts = self.kpts().len();
        let (na, nb) = self.nelec()?;
        let (ea, eb) = mo_energy.split_at(nkpts);
        if let Some(sm) = self.smearing.as_ref() {
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

    fn make_rdm1(&self, mo_coeff: &[CTensor], mo_occ: &[Vec<f64>]) -> Result<KDms, PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        let nkpts = self.kpts().len();
        Ok(vec![
            make_rdm1(&mo_coeff[..nkpts], &mo_occ[..nkpts], nao),
            make_rdm1(&mo_coeff[nkpts..], &mo_occ[nkpts..], nao),
        ])
    }

    fn energy_elec(&self, dms: &KDms, h1e: &KMats, vhf: &KDms) -> Result<(f64, f64), PyscfRsError> {
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
        // U-09 step 2: `e1` is a term of every KUKS total energy and was a
        // naive `(nset * nkpts)`-long running sum. U-03 ordered the KUHF copy
        // in `krdm::energy_elec` and the traces in `veff.rs`, and did not
        // reach this one — the same defect class, one file over. Collect the
        // per-`(set, k)` partials (each an ordered `trace_ab`) and reduce THOSE
        // through the pairwise tree, so the composition is ordered-inside-
        // ordered.
        //
        // BIT-EXACT at every cell this repository gates on: for
        // `nset * nkpts <= PAIRWISE_CHUNK` (128) `oracle_sum`'s base case is
        // the strict left-to-right fold this loop performed.
        let mut e1_parts = Vec::with_capacity(dms.len() * h1e.len());
        for set in dms.iter() {
            for (k, h) in h1e.iter().enumerate() {
                e1_parts.push(pyscf_pbc_scf::krdm::trace_ab(&set[k], h, nao).0);
            }
        }
        let e1 = weight * oracle_sum(&e1_parts);
        Ok((e1 + tags.ecoul + tags.exc, tags.ecoul + tags.exc))
    }

    fn free_energy(&self) -> Option<f64> {
        self.entropy
            .get()
            .map(|s| -self.smearing.as_ref().map_or(0.0, |sm| sm.sigma) * s)
    }
}
