//! `KGKS` — generalized (2-component) Kohn-Sham with k-point sampling
//! (plan 12-04). Port of `pyscf/pbc/dft/kgks.py:42-135`.
//!
//! ```text
//! nset = 1, but every matrix is 2·nao x 2·nao
//! get_veff = Vxc(2c) + J                             (kgks.py:105-107)
//! ecoul    = 0.5 · (1/N_k) Σ_k Tr(D^k J^k)           (kgks.py:130)
//! ```
//!
//! `J` is built from the two DIAGONAL spin blocks and is itself block-diagonal
//! with `J_aa = J_bb = J[D_aa] + J[D_bb]` (`kghf.py:80-83`).
//!
//! Upstream raises `NotImplementedError` for a hybrid `KGKS` (`kgks.py:66-68`);
//! this port raises the same rather than silently dropping the exchange.

use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_df::{Fftdf, JkOpts, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::Kghf;
use pyscf_pbc_scf::khooks::KOverrideHooks;
use pyscf_pbc_scf::kscf::kernel;
use pyscf_pbc_scf::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};

use crate::error::PbcDftError;
use crate::gen_grid::PeriodicGrids;
use crate::krks::{KsEnergyTags, unwrap_err};
use crate::numint2c::KNumInt2C;
use crate::veff::trace_ab;
use crate::xc::{err, is_hybrid_xc};

/// Generalized periodic Kohn-Sham.
#[derive(Debug)]
pub struct Kgks {
    /// The Hartree-Fock half — owns the density fitting and supplies every
    /// non-KS hook.
    pub hf: Kghf,
    /// The XC functional string.
    pub xc: String,
    /// The integration grid.
    pub grids: PeriodicGrids,
    /// The 2-component numerical-integration driver.
    pub ni: KNumInt2C,
    tags: std::cell::Cell<Option<KsEnergyTags>>,
}

impl Kgks {
    /// Build a `KGKS` with the default `FFTDF` and the uniform grid.
    ///
    /// # Errors
    /// Propagates the `FFTDF` and grid construction.
    pub fn new(cell: Cell, kpts: &[[f64; 3]], xc: &str) -> Result<Self, PbcDftError> {
        let with_df = Fftdf::new(cell, kpts)
            .map_err(|e| err(format!("KGKS: FFTDF construction failed: {e}")))?;
        Self::from_df(Box::new(with_df), xc)
    }

    /// `KGKS` over an explicit density-fitting object.
    ///
    /// # Errors
    /// Propagates the grid construction.
    pub fn from_df(with_df: Box<dyn PeriodicDf>, xc: &str) -> Result<Self, PbcDftError> {
        let grids = PeriodicGrids::uniform(with_df.cell(), Some(with_df.mesh()))?;
        let ni = KNumInt2C::new(with_df.kpts());
        Ok(Self {
            hf: Kghf::from_df(with_df),
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

    /// The exchange divergence treatment.
    pub fn exxdiv(&self) -> Option<ExxDiv> {
        self.hf.exxdiv
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
    /// As [`Kgks::kernel`].
    pub fn run(&self) -> Result<KScfResult, PyscfRsError> {
        self.kernel(&KScfConfig::for_cell(self.cell()))
    }

    /// `get_veff` — `kgks.py:42-135`.
    ///
    /// # Errors
    /// [`PbcDftError`] for a hybrid functional (upstream raises too), or from
    /// the grid loop / J build.
    // `k` indexes `vxc`, both `vj` channels and `jfull` together.
    #[allow(clippy::needless_range_loop)]
    pub fn get_veff_tagged(
        &self,
        dms: &KDms,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<(KDms, KsEnergyTags), PbcDftError> {
        if is_hybrid_xc(&self.xc)? {
            return Err(PbcDftError::Core(PyscfRsError::NotYetImplemented {
                phase: 12,
                what: "KGKS with a hybrid functional — upstream raises \
                       NotImplementedError at kgks.py:66-68 as well",
            }));
        }
        let cell = self.cell();
        let nao = cell.mol.nao_nr;
        let n2 = 2 * nao;
        let nkpts = self.kpts().len();
        let weight = 1.0 / nkpts as f64;
        let ground_state = kpts_band.is_none();

        // kgks.py:97-98 — the 2-component XC.
        let nr = self
            .ni
            .nr_vxc(cell, &self.grids, &self.xc, &dms[0], kpts_band)?;
        let mut vxc = nr.vmat;

        // kgks.py:105-107 — J on the two diagonal spin blocks.
        let blocks: KDms = vec![
            dms[0].iter().map(|d| sub_block(d, nao, 0, 0)).collect(),
            dms[0].iter().map(|d| sub_block(d, nao, nao, nao)).collect(),
        ];
        let r = self
            .hf
            .with_df
            .get_jk(
                &blocks,
                self.kpts(),
                JkOpts {
                    hermi: 1,
                    kpts_band,
                    with_j: true,
                    with_k: false,
                    exxdiv: None,
                    omega: None,
                    kk_symmetry: false,
                },
            )
            .map_err(|e| err(format!("KGKS: density fitting failed: {e}")))?;
        let vj = r
            .vj
            .ok_or_else(|| err("KGKS: the density-fitting object returned no vj"))?;

        let nband = vxc.len();
        let mut jfull: KMats = Vec::with_capacity(nband);
        for k in 0..nband {
            let mut jsum = vj[0][k].clone();
            for i in 0..jsum.len() {
                jsum.re[i] += vj[1][k].re[i];
                jsum.im[i] += vj[1][k].im[i];
            }
            let mut m = CTensor::zeros(n2 * n2);
            write_block(&mut m, &jsum, nao, 0, 0);
            write_block(&mut m, &jsum, nao, nao, nao);
            jfull.push(m);
        }
        for (k, m) in vxc.iter_mut().enumerate() {
            for i in 0..m.len() {
                m.re[i] += jfull[k].re[i];
                m.im[i] += jfull[k].im[i];
            }
        }

        // kgks.py:130
        let ecoul = if ground_state {
            let mut acc = 0.0_f64;
            for (k, d) in dms[0].iter().enumerate() {
                acc += trace_ab(d, &jfull[k], n2).0;
            }
            0.5 * weight * acc
        } else {
            0.0
        };

        Ok((
            vec![vxc],
            KsEnergyTags {
                ecoul,
                exc: nr.excsum,
                nelec: nr.nelec,
            },
        ))
    }
}

impl KOverrideHooks for Kgks {
    fn cell(&self) -> &Cell {
        self.hf.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.hf.kpts()
    }
    fn nset(&self) -> usize {
        1
    }
    fn nao(&self) -> usize {
        2 * self.cell().mol.nao_nr
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

    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        // kgks.py:137 — `energy_elec = krks.energy_elec`.
        let tags = match self.tags.get() {
            Some(t) => t,
            None => {
                let (_, t) = self.get_veff_tagged(dms, None).map_err(unwrap_err)?;
                self.tags.set(Some(t));
                t
            }
        };
        let _ = vhf;
        let n2 = 2 * self.cell().mol.nao_nr;
        let weight = 1.0 / h1e.len() as f64;
        let mut e1 = 0.0_f64;
        for (k, h) in h1e.iter().enumerate() {
            e1 += trace_ab(&dms[0][k], h, n2).0;
        }
        e1 *= weight;
        Ok((e1 + tags.ecoul + tags.exc, tags.ecoul + tags.exc))
    }
}

/// The `nao x nao` sub-block of a `2·nao x 2·nao` matrix at `(r0, c0)`.
fn sub_block(m: &CTensor, nao: usize, r0: usize, c0: usize) -> CTensor {
    let n2 = 2 * nao;
    let mut re = vec![0.0_f64; nao * nao];
    let mut im = vec![0.0_f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            re[i * nao + j] = m.re[(r0 + i) * n2 + c0 + j];
            im[i * nao + j] = m.im[(r0 + i) * n2 + c0 + j];
        }
    }
    CTensor::from_planes(re, im)
}

fn write_block(dst: &mut CTensor, src: &CTensor, nao: usize, r0: usize, c0: usize) {
    let n2 = 2 * nao;
    for i in 0..nao {
        for j in 0..nao {
            dst.re[(r0 + i) * n2 + c0 + j] = src.re[i * nao + j];
            dst.im[(r0 + i) * n2 + c0 + j] = src.im[i * nao + j];
        }
    }
}
