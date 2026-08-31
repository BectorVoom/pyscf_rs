//! `KGHF` — generalised (two-component) Hartree-Fock with k-point sampling
//! (`kghf.py`).
//!
//! Every matrix lives in the `2 nao` SPIN-ORBITAL basis. `S` and `H` are
//! block-diagonal copies of the scalar ones (`kghf.py:216-231`); the J/K build
//! decomposes the spin-orbital density into `nao`-sized blocks, hands them to
//! the SAME `PeriodicDf::get_jk` as every other method, and reassembles
//! (`kghf.py:40-99`):
//!
//! ```text
//! dm  = [[dm_aa, .    ], [dm_ab, dm_bb]]      (dm_ab is the LOWER-left block)
//! J   = diag(J[aa] + J[bb],  J[aa] + J[bb])
//! K   = [[K[aa], K[ab]], [K[ab]^H, K[bb]]]
//! vhf = J - K
//! ```

use pyscf_algebra::{CTensor, zeigh_gen};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_df::{Fftdf, JkOpts, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};

use crate::khooks::KOverrideHooks;
use crate::kocc::fermi_level;
use crate::krdm::{energy_elec, make_rdm1};
use crate::krhf::{df_err, to_row_major};
use crate::kscf::kernel;
use crate::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};

/// Generalised periodic Hartree-Fock.
#[derive(Debug)]
pub struct Kghf {
    /// The density-fitting object.
    pub with_df: Box<dyn PeriodicDf>,
    /// Exchange divergence treatment.
    pub exxdiv: Option<ExxDiv>,
}

impl Kghf {
    /// Build a `KGHF` with the default `FFTDF`.
    ///
    /// # Errors
    /// Propagates the `FFTDF` construction.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Result<Self, PyscfRsError> {
        Ok(Self::from_df(Box::new(Fftdf::new(cell, kpts).map_err(df_err)?)))
    }

    /// `KGHF` over an explicit density-fitting object.
    pub fn from_df(with_df: Box<dyn PeriodicDf>) -> Self {
        Self {
            with_df,
            exxdiv: Some(ExxDiv::Ewald),
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
    /// Scalar AO count (half the spin-orbital dimension).
    pub fn nao_scalar(&self) -> usize {
        self.cell().mol.nao_nr
    }

    /// Run the SCF.
    ///
    /// # Errors
    /// Propagates every hook and the driver.
    pub fn kernel(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        kernel(self, cfg)
    }

    /// Run with cell-derived defaults.
    ///
    /// # Errors
    /// As [`Kghf::kernel`].
    pub fn run(&self) -> Result<KScfResult, PyscfRsError> {
        self.kernel(&KScfConfig::for_cell(self.cell()))
    }
}

/// `scipy.linalg.block_diag(x, x)` — `kghf.py:218` / `:231`.
fn block_diag(x: &CTensor, nao: usize) -> CTensor {
    let nso = 2 * nao;
    let mut re = vec![0.0_f64; nso * nso];
    let mut im = vec![0.0_f64; nso * nso];
    for i in 0..nao {
        for j in 0..nao {
            let v = (x.re[i * nao + j], x.im[i * nao + j]);
            re[i * nso + j] = v.0;
            im[i * nso + j] = v.1;
            re[(nao + i) * nso + (nao + j)] = v.0;
            im[(nao + i) * nso + (nao + j)] = v.1;
        }
    }
    CTensor::from_planes(re, im)
}

/// Extract the `(row_off, col_off)` `nao x nao` block of a `2nao x 2nao` matrix.
fn block(x: &CTensor, nao: usize, roff: usize, coff: usize) -> CTensor {
    let nso = 2 * nao;
    let mut re = vec![0.0_f64; nao * nao];
    let mut im = vec![0.0_f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            re[i * nao + j] = x.re[(roff + i) * nso + coff + j];
            im[i * nao + j] = x.im[(roff + i) * nso + coff + j];
        }
    }
    CTensor::from_planes(re, im)
}

fn set_block(out: &mut CTensor, b: &CTensor, nao: usize, roff: usize, coff: usize) {
    let nso = 2 * nao;
    for i in 0..nao {
        for j in 0..nao {
            out.re[(roff + i) * nso + coff + j] = b.re[i * nao + j];
            out.im[(roff + i) * nso + coff + j] = b.im[i * nao + j];
        }
    }
}

impl KOverrideHooks for Kghf {
    fn cell(&self) -> &Cell {
        self.with_df.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.with_df.kpts()
    }
    fn nset(&self) -> usize {
        1
    }
    fn nao(&self) -> usize {
        2 * self.cell().mol.nao_nr
    }

    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        let nao = self.nao_scalar();
        let s = to_row_major(
            pyscf_pbc_gto::get_ovlp(self.cell(), self.kpts())?,
            nao,
        );
        Ok(s.iter().map(|m| block_diag(m, nao)).collect())
    }

    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        let nao = self.nao_scalar();
        let h = pyscf_pbc_df::get_hcore(self.with_df.as_ref(), self.kpts()).map_err(df_err)?;
        Ok(h.iter().map(|m| block_diag(m, nao)).collect())
    }

    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        // kghf.py:180-190 — `_from_rhf_init_dm`: the restricted guess is split
        // evenly between the two diagonal spin blocks.
        let nao = self.nao_scalar();
        let nkpts = self.kpts().len();
        let scalar_s = to_row_major(
            pyscf_pbc_gto::get_ovlp(self.cell(), self.kpts())?,
            nao,
        );
        let scalar = crate::init_guess::get_init_guess(
            self.cell(),
            nkpts,
            1,
            mode,
            &scalar_s,
            self.cell().tot_electrons(nkpts) as f64,
        )?;
        let _ = s1e;
        let mut out = Vec::with_capacity(nkpts);
        for k in 0..nkpts {
            let mut m = CTensor::zeros(4 * nao * nao);
            let half = CTensor::from_planes(
                scalar[0][k].re.iter().map(|v| v * 0.5).collect(),
                scalar[0][k].im.iter().map(|v| v * 0.5).collect(),
            );
            set_block(&mut m, &half, nao, 0, 0);
            set_block(&mut m, &half, nao, nao, nao);
            out.push(m);
        }
        Ok(vec![out])
    }

    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let nao = self.nao_scalar();
        let nkpts = self.kpts().len();
        // kghf.py:56-62 — three nao-sized channels: aa, bb and the LOWER-left
        // ab block. `hermi = 0` for the K build, as upstream sets `_hermi`.
        let mut blocks: KDms = vec![Vec::with_capacity(nkpts); 3];
        for k in 0..nkpts {
            blocks[0].push(block(&dms[0][k], nao, 0, 0));
            blocks[1].push(block(&dms[0][k], nao, nao, nao));
            blocks[2].push(block(&dms[0][k], nao, nao, 0));
        }
        let r = self
            .with_df
            .get_jk(
                &blocks,
                self.kpts(),
                JkOpts {
                    hermi: 0,
                    kpts_band: None,
                    with_j: true,
                    with_k: true,
                    exxdiv: self.exxdiv,
                    omega: None,
                    kk_symmetry: false,
                },
            )
            .map_err(df_err)?;
        let j1 = r.vj.ok_or_else(|| missing("vj"))?;
        let k1 = r.vk.ok_or_else(|| missing("vk"))?;

        let mut out = Vec::with_capacity(nkpts);
        for k in 0..nkpts {
            let mut v = CTensor::zeros(4 * nao * nao);
            // kghf.py:80-83 — J is block diagonal with J[aa] + J[bb].
            let mut jsum = j1[0][k].clone();
            for i in 0..jsum.len() {
                jsum.re[i] += j1[1][k].re[i];
                jsum.im[i] += j1[1][k].im[i];
            }
            set_block(&mut v, &jsum, nao, 0, 0);
            set_block(&mut v, &jsum, nao, nao, nao);
            // kghf.py:86-95 — subtract K.
            let kaa = &k1[0][k];
            let kbb = &k1[1][k];
            let kab = &k1[2][k];
            let mut kfull = CTensor::zeros(4 * nao * nao);
            set_block(&mut kfull, kaa, nao, 0, 0);
            set_block(&mut kfull, kbb, nao, nao, nao);
            set_block(&mut kfull, kab, nao, 0, nao);
            // hermi branch: vk[nao:, :nao] = k1[2].conj().T
            let mut kba = CTensor::zeros(nao * nao);
            for i in 0..nao {
                for j in 0..nao {
                    kba.re[i * nao + j] = kab.re[j * nao + i];
                    kba.im[i * nao + j] = -kab.im[j * nao + i];
                }
            }
            set_block(&mut kfull, &kba, nao, nao, 0);
            for i in 0..v.len() {
                v.re[i] -= kfull.re[i];
                v.im[i] -= kfull.im[i];
            }
            out.push(v);
        }
        Ok(vec![out])
    }

    fn eig(
        &self,
        fock: &KDms,
        s1e: &KMats,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        let nso = self.nao();
        let mut es = Vec::with_capacity(fock[0].len());
        let mut cs = Vec::with_capacity(fock[0].len());
        for (k, f) in fock[0].iter().enumerate() {
            let (e, c) = zeigh_gen(f, &s1e[k], nso).map_err(|err| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "KGHF: zeigh_gen failed at k = {k}: {err}"
                )))
            })?;
            es.push(e);
            cs.push(c);
        }
        Ok((es, cs))
    }

    fn get_occ(
        &self,
        mo_energy: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        // kghf.py:105-120 — nocc = cell.nelectron * nkpts, occupancy 1.
        let nocc = self.cell().tot_electrons(self.kpts().len());
        let (fermi, _) = fermi_level(mo_energy, nocc)?;
        let occ = mo_energy
            .iter()
            .map(|e| {
                e.iter()
                    .map(|v| if *v <= fermi { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        Ok((occ, vec![fermi]))
    }

    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        Ok(vec![make_rdm1(mo_coeff, mo_occ, self.nao())])
    }

    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        Ok(energy_elec(dms, h1e, vhf, self.nao()))
    }
}

fn missing(what: &str) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "KGHF: the density-fitting object returned no {what}"
    )))
}
