//! `KROHF` — restricted OPEN-shell Hartree-Fock with k-point sampling
//! (`krohf.py`).
//!
//! ROHF is the one method whose Fock channel count differs from its density
//! channel count: the two spin Fock matrices are combined into ONE Roothaan
//! effective Fock (`krohf.py:85-120`) whose eigenvectors are the single set of
//! orbitals, while the density still has an alpha and a beta channel. That is
//! why [`crate::khooks::KOverrideHooks::nfock`] exists.
//!
//! ```text
//! nset  = 2   (dma from occ > 0, dmb from occ == 2)
//! nfock = 1   (the Roothaan effective Fock)
//! ```

use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_df::{Fftdf, JkOpts, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};

use crate::khooks::KOverrideHooks;
use crate::kocc::fermi_level;
use crate::krdm::{energy_elec, make_rdm1_one};
use crate::krhf::{df_err, eig_channel, to_row_major};
use crate::kscf::{bare_fock, kernel};
use crate::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};

/// Restricted open-shell periodic Hartree-Fock.
#[derive(Debug)]
pub struct Krohf {
    /// The density-fitting object.
    pub with_df: Fftdf,
    /// Exchange divergence treatment.
    pub exxdiv: Option<ExxDiv>,
    /// Override the `(nalpha, nbeta)` derived from `cell.spin`.
    pub nelec: Option<(usize, usize)>,
}

impl Krohf {
    /// Build a `KROHF` with the default `FFTDF`.
    ///
    /// # Errors
    /// Propagates the `FFTDF` construction.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Result<Self, PyscfRsError> {
        Ok(Self::from_df(Fftdf::new(cell, kpts).map_err(df_err)?))
    }

    /// `KROHF` over an explicit density-fitting object.
    pub fn from_df(with_df: Fftdf) -> Self {
        Self {
            with_df,
            exxdiv: Some(ExxDiv::Ewald),
            nelec: None,
        }
    }

    /// The cell.
    pub fn cell(&self) -> &Cell {
        &self.with_df.cell
    }

    /// The k-points.
    pub fn kpts(&self) -> &[[f64; 3]] {
        &self.with_df.kpts
    }

    /// `(nalpha, nbeta)` — same rule as [`crate::Kuhf::nelec`].
    ///
    /// # Errors
    /// As [`crate::Kuhf::nelec`].
    pub fn nelec(&self) -> Result<(usize, usize), PyscfRsError> {
        if let Some(n) = self.nelec {
            return Ok(n);
        }
        let ne = self.cell().tot_electrons(self.kpts().len()) as i64;
        let spin = self.cell().mol.spin as i64;
        let na = (ne + spin) / 2;
        let nb = na - spin;
        if na + nb != ne || na < 0 || nb < 0 {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "KROHF: electron number {ne} and spin {spin} are not consistent"
            ))));
        }
        Ok((na as usize, nb as usize))
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
    /// As [`Krohf::kernel`].
    pub fn run(&self) -> Result<KScfResult, PyscfRsError> {
        self.kernel(&KScfConfig::for_cell(self.cell()))
    }
}

/// `get_roothaan_fock((focka, fockb), (dma, dmb), s)` — `krohf.py:85-120`.
///
/// ```text
///           closed   open   virtual
/// closed      Fc      Fb      Fc
/// open        Fb      Fc      Fa
/// virtual     Fc      Fa      Fc      with Fc = (Fa + Fb)/2
/// ```
///
/// Built as upstream builds it — six projected products, then `F + F^H` — so
/// the result is exactly Hermitian by construction.
pub fn roothaan_fock(
    focka: &CTensor,
    fockb: &CTensor,
    dma: &CTensor,
    dmb: &CTensor,
    s: &CTensor,
    nao: usize,
) -> CTensor {
    let fc = {
        let mut m = focka.clone();
        for i in 0..m.len() {
            m.re[i] = 0.5 * (m.re[i] + fockb.re[i]);
            m.im[i] = 0.5 * (m.im[i] + fockb.im[i]);
        }
        m
    };
    // pc = dmb.S ; po = (dma - dmb).S ; pv = I - dma.S
    let pc = mm(dmb, s, nao);
    let dab = {
        let mut m = dma.clone();
        for i in 0..m.len() {
            m.re[i] -= dmb.re[i];
            m.im[i] -= dmb.im[i];
        }
        m
    };
    let po = mm(&dab, s, nao);
    let mut pv = mm(dma, s, nao);
    for i in 0..nao * nao {
        pv.re[i] = -pv.re[i];
        pv.im[i] = -pv.im[i];
    }
    for i in 0..nao {
        pv.re[i * nao + i] += 1.0;
    }

    let term = |l: &CTensor, m0: &CTensor, r: &CTensor| -> CTensor {
        let lh = conj_t(l, nao);
        mm(&mm(&lh, m0, nao), r, nao)
    };
    let mut f = CTensor::zeros(nao * nao);
    let add = |f: &mut CTensor, t: &CTensor, w: f64| {
        for i in 0..f.len() {
            f.re[i] += w * t.re[i];
            f.im[i] += w * t.im[i];
        }
    };
    add(&mut f, &term(&pc, &fc, &pc), 0.5);
    add(&mut f, &term(&po, &fc, &po), 0.5);
    add(&mut f, &term(&pv, &fc, &pv), 0.5);
    add(&mut f, &term(&po, fockb, &pc), 1.0);
    add(&mut f, &term(&po, focka, &pv), 1.0);
    add(&mut f, &term(&pv, &fc, &pc), 1.0);

    // krohf.py:119 — fock + fock.conj().T
    let fh = conj_t(&f, nao);
    for i in 0..f.len() {
        f.re[i] += fh.re[i];
        f.im[i] += fh.im[i];
    }
    f
}

fn conj_t(a: &CTensor, n: usize) -> CTensor {
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            re[i * n + j] = a.re[j * n + i];
            im[i * n + j] = -a.im[j * n + i];
        }
    }
    CTensor::from_planes(re, im)
}

fn mm(a: &CTensor, b: &CTensor, n: usize) -> CTensor {
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sr = 0.0_f64;
            let mut si = 0.0_f64;
            for t in 0..n {
                let (ar, ai) = (a.re[i * n + t], a.im[i * n + t]);
                let (br, bi) = (b.re[t * n + j], b.im[t * n + j]);
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }
            re[i * n + j] = sr;
            im[i * n + j] = si;
        }
    }
    CTensor::from_planes(re, im)
}

impl KOverrideHooks for Krohf {
    fn cell(&self) -> &Cell {
        &self.with_df.cell
    }
    fn kpts(&self) -> &[[f64; 3]] {
        &self.with_df.kpts
    }
    fn nset(&self) -> usize {
        2
    }
    fn nfock(&self) -> usize {
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
        pyscf_pbc_df::get_hcore(&self.with_df, self.kpts()).map_err(df_err)
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
        // krohf.py:311-323 — identical to KUHF's.
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
                },
            )
            .map_err(df_err)?;
        let vj = r.vj.ok_or_else(|| missing("vj"))?;
        let vk = r.vk.ok_or_else(|| missing("vk"))?;
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

    fn get_fock(&self, h1e: &KMats, vhf: &KDms, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        let spin_fock = bare_fock(h1e, vhf);
        let s1e = self.get_ovlp()?;
        let f: KMats = (0..h1e.len())
            .map(|k| {
                roothaan_fock(
                    &spin_fock[0][k],
                    &spin_fock[1][k],
                    &dms[0][k],
                    &dms[1][k],
                    &s1e[k],
                    nao,
                )
            })
            .collect();
        Ok(vec![f])
    }

    fn diis_dms(&self, dms: &KDms) -> KDms {
        // krohf.py:74 — dm_sf = dma + dmb
        let sum: KMats = dms[0]
            .iter()
            .zip(dms[1].iter())
            .map(|(a, b)| {
                let mut m = a.clone();
                for i in 0..m.len() {
                    m.re[i] += b.re[i];
                    m.im[i] += b.im[i];
                }
                m
            })
            .collect();
        vec![sum]
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
        // krohf.py:121-160. Without the `mo_ea` tag (which upstream only has
        // when the caller kept the alpha Fock around) `mo_ea == mo_energy`,
        // which is the branch this port implements: the core level fills
        // doubly, the next `na - nb` Roothaan levels singly.
        let (na, nb) = self.nelec()?;
        let core_level = if nb > 0 {
            fermi_level(mo_energy, nb)?.0
        } else {
            -1e9
        };
        let fermi = if na == nb {
            core_level
        } else {
            let above: Vec<Vec<f64>> = mo_energy
                .iter()
                .map(|e| e.iter().copied().filter(|v| *v > core_level).collect())
                .collect();
            fermi_level(&above, na - nb)?.0
        };
        let occ = mo_energy
            .iter()
            .map(|e| {
                e.iter()
                    .map(|v| {
                        if *v <= core_level {
                            2.0
                        } else if na != nb && *v <= fermi {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        Ok((occ, vec![fermi, core_level]))
    }

    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        // krohf.py:38-51 — alpha from `occ > 0`, beta from `occ == 2`, each
        // with unit weight.
        let nao = self.cell().mol.nao_nr;
        let mut dma = Vec::with_capacity(mo_coeff.len());
        let mut dmb = Vec::with_capacity(mo_coeff.len());
        for (c, occ) in mo_coeff.iter().zip(mo_occ.iter()) {
            let occ_a: Vec<f64> = occ.iter().map(|o| f64::from(*o > 0.0)).collect();
            let occ_b: Vec<f64> = occ.iter().map(|o| f64::from(*o == 2.0)).collect();
            dma.push(make_rdm1_one(c, &occ_a, nao));
            dmb.push(make_rdm1_one(c, &occ_b, nao));
        }
        Ok(vec![dma, dmb])
    }

    fn get_grad(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
        h1e: &KMats,
        vhf: &KDms,
    ) -> Vec<f64> {
        // rohf.get_grad: three blocks (open-virtual with Fa, closed-open with
        // Fb, closed-virtual with Fc).
        let nao = self.cell().mol.nao_nr;
        let spin_fock = bare_fock(h1e, vhf);
        let mut g = Vec::new();
        for (k, occ) in mo_occ.iter().enumerate() {
            let fa = &spin_fock[0][k];
            let fb = &spin_fock[1][k];
            let nmo = occ.len();
            let ea = mo_diag_pair(fa, &mo_coeff[k], nao, nmo);
            let eb = mo_diag_pair(fb, &mo_coeff[k], nao, nmo);
            // `rohf.get_grad`: two masks over the MO-basis Fock matrices,
            //   uniq_var_a = (occ_i == 0)          & (occ_j >  0)   -> F_alpha
            //   uniq_var_b = (occ_i != 2)          & (occ_j == 2)   -> F_beta
            // and the entries where they OVERLAP (a virtual row against a
            // doubly-occupied column) take the SUM of the two. Upstream builds
            // that as `g[a] = focka; g[b] += fockb` and returns `g[a | b]`.
            for i in 0..nmo {
                for j in 0..nmo {
                    let (oi, oj) = (occ[i], occ[j]);
                    let in_a = oi == 0.0 && oj > 0.0;
                    let in_b = oi != 2.0 && oj == 2.0;
                    if !in_a && !in_b {
                        continue;
                    }
                    let mut re = 0.0_f64;
                    let mut im = 0.0_f64;
                    if in_a {
                        re += ea[i * nmo + j].0;
                        im += ea[i * nmo + j].1;
                    }
                    if in_b {
                        re += eb[i * nmo + j].0;
                        im += eb[i * nmo + j].1;
                    }
                    g.push(re);
                    g.push(im);
                }
            }
        }
        g
    }

    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        Ok(energy_elec(dms, h1e, vhf, self.cell().mol.nao_nr))
    }
}

/// The full MO-basis Fock matrix `C^H F C` as `(re, im)` pairs, row-major
/// `nmo x nmo`.
fn mo_diag_pair(fock: &CTensor, c: &CTensor, nao: usize, nmo: usize) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0_f64, 0.0_f64); nmo * nmo];
    for i in 0..nmo {
        for j in 0..nmo {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for mu in 0..nao {
                let mut fr = 0.0_f64;
                let mut fi = 0.0_f64;
                for nu in 0..nao {
                    let (x, y) = (fock.re[mu * nao + nu], fock.im[mu * nao + nu]);
                    let (u, v) = (c.re[nu + j * nao], c.im[nu + j * nao]);
                    fr += x * u - y * v;
                    fi += x * v + y * u;
                }
                let (cr, ci) = (c.re[mu + i * nao], -c.im[mu + i * nao]);
                re += cr * fr - ci * fi;
                im += cr * fi + ci * fr;
            }
            out[i * nmo + j] = (re, im);
        }
    }
    out
}

fn missing(what: &str) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "KROHF: the density-fitting object returned no {what}"
    )))
}
