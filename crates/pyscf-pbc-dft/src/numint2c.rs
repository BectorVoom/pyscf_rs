//! `KNumInt2C` — the 2-component (spinor / generalized) numerical integrator
//! (plan 12-07). Port of `pyscf/pbc/dft/numint2c.py:328-642`.
//!
//! A 2-component density matrix is `2·nao x 2·nao`, blocked as
//!
//! ```text
//! D = [[D_aa, D_ab],
//!      [D_ba, D_bb]]
//! ```
//!
//! and the density it produces is a FOUR-vector `(ρ, m_x, m_y, m_z)`:
//!
//! ```text
//! ρ   = ρ_aa + ρ_bb          m_x =  2 Re ρ_ab
//! m_z = ρ_aa − ρ_bb          m_y = −2 Im ρ_ab      (Hermitian D)
//! ```
//!
//! # The three spin treatments
//!
//! | [`Collinear`] | what it does | status |
//! |---|---|---|
//! | [`Collinear::Col`] | ignores `m_x`/`m_y`; runs the ordinary UKS grid loop on `(D_aa, D_bb)` and puts the two `V_xc` into the diagonal blocks (`numint2c.py:618-629`) | full |
//! | [`Collinear::Ncol`] | LDA only: the functional is evaluated at `ρ_{a,b} = (ρ ± \|m\|)/2` and `V_xc` acquires off-diagonal blocks along `m̂` (`dft/numint2c.py:_ncol_lda_vxc_mat`) | LDA only, as upstream |
//! | [`Collinear::Mcol`] | multi-collinear: the functional is integrated over spin-angular samples by `mcfun` | [`PyscfRsError::NotYetImplemented`] |
//!
//! `Col` is the upstream default (`dft_numint2c_NumInt2C_collinear = 'col'`)
//! and is what `KGKS` uses unless a caller changes it.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::PyscfRsError;
use pyscf_pbc_gto::{Cell, EvalAoKptsOutput};
use pyscf_pbc_scf::types::{KDms, KMats};

use crate::error::PbcDftError;
use crate::gen_grid::PeriodicGrids;
use crate::numint::KNumInt;
use crate::xc::{XcType, err, eval_xc_eff_uks};

/// How the two spin components are coupled to the functional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Collinear {
    /// `'col'` — collinear. The upstream default.
    #[default]
    Col,
    /// `'ncol'` — non-collinear (LDA only).
    Ncol,
    /// `'mcol'` — multi-collinear (needs `mcfun`).
    Mcol,
}

/// `(nelec, excsum, vmat)` for a 2-component build.
#[derive(Debug, Clone)]
pub struct Nr2cResult {
    /// `∫ ρ` (both spin components together).
    pub nelec: f64,
    /// `E_xc`.
    pub excsum: f64,
    /// `vmat[k]`, `2·nao x 2·nao` ROW-MAJOR.
    pub vmat: KMats,
}

/// `pbc/dft/numint2c.py:KNumInt2C`.
#[derive(Debug)]
pub struct KNumInt2C {
    /// The scalar (1-component) integrator every branch delegates to.
    pub ni: KNumInt,
    /// The spin treatment.
    pub collinear: Collinear,
}

impl KNumInt2C {
    /// A 2-component integrator over `kpts`, collinear by default.
    pub fn new(kpts: &[[f64; 3]]) -> Self {
        Self {
            ni: KNumInt::new(kpts),
            collinear: Collinear::default(),
        }
    }

    /// The sampling k-points.
    pub fn kpts(&self) -> &[[f64; 3]] {
        &self.ni.kpts
    }

    /// `nr_vxc(cell, grids, xc_code, dms, hermi, kpts, kpts_band)` —
    /// `numint2c.py:611-630`.
    ///
    /// `dms[k]` is one `2·nao x 2·nao` matrix per k-point.
    ///
    /// # Errors
    /// [`PbcDftError`] for `Collinear::Mcol`, for a non-LDA `Collinear::Ncol`,
    /// or from the grid loop.
    pub fn nr_vxc(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &KMats,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<Nr2cResult, PbcDftError> {
        match self.collinear {
            Collinear::Col => self.nr_vxc_collinear(cell, grids, xc_code, dms, kpts_band),
            Collinear::Ncol => self.nr_vxc_ncol(cell, grids, xc_code, dms, kpts_band),
            Collinear::Mcol => Err(PbcDftError::Core(PyscfRsError::NotYetImplemented {
                phase: 12,
                what: "KNumInt2C with collinear = 'mcol' — the multi-collinear XC \
                       evaluation needs the `mcfun` spin-angular quadrature \
                       (numint2c.py:502-560)",
            })),
        }
    }

    /// The collinear branch — `numint2c.py:618-629`.
    fn nr_vxc_collinear(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &KMats,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<Nr2cResult, PbcDftError> {
        let nao = cell.mol.nao_nr;
        let (dm_a, dm_b) = split_diagonal(dms, nao)?;
        let sets: [KDms; 2] = [vec![dm_a], vec![dm_b]];
        let nr = self.ni.nr_uks(cell, grids, xc_code, &sets, 1, kpts_band)?;
        let mut vmat = Vec::with_capacity(nr.vmat[0][0].len());
        for k in 0..nr.vmat[0][0].len() {
            let mut m = CTensor::zeros(4 * nao * nao);
            set_block(&mut m, &nr.vmat[0][0][k], nao, 0, 0);
            set_block(&mut m, &nr.vmat[1][0][k], nao, nao, nao);
            vmat.push(m);
        }
        Ok(Nr2cResult {
            nelec: nr.nelec[0].0 + nr.nelec[0].1,
            excsum: nr.excsum[0],
            vmat,
        })
    }

    /// The non-collinear LDA branch — `dft/numint2c.py:_ncol_lda_vxc_mat`
    /// driven by `pbc/dft/numint2c.py:502-556`.
    fn nr_vxc_ncol(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dms: &KMats,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<Nr2cResult, PbcDftError> {
        let ty = XcType::of(xc_code)?;
        if ty != XcType::Lda {
            return Err(PbcDftError::Core(PyscfRsError::NotYetImplemented {
                phase: 12,
                what: "KNumInt2C with collinear = 'ncol' supports LDA only — upstream \
                       maps only ('LDA','n') in its f_eval_mat table \
                       (numint2c.py:521-527)",
            }));
        }
        let nao = cell.mol.nao_nr;
        let coords = grids.coords()?;
        let weights = grids.weights()?;
        let ngrids = coords.len();
        let band = kpts_band.unwrap_or(self.kpts());
        let nkpts = self.kpts().len();

        let mut nelec = 0.0_f64;
        let mut excsum = 0.0_f64;
        let mut vmat: KMats = vec![CTensor::zeros(4 * nao * nao); band.len()];

        for (p0, p1) in self.ni.block_ranges(ngrids, ty, nkpts) {
            let chunk = &coords[p0..p1];
            let w = &weights[p0..p1];
            let ao2 = self.ni.eval_ao(cell, chunk, self.kpts(), ty)?;
            let ao1 = if kpts_band.is_none() {
                std::sync::Arc::clone(&ao2)
            } else {
                self.ni.eval_ao(cell, chunk, band, ty)?
            };
            let rho_m = eval_rho_m(&ao2, dms, nao)?;
            let n = p1 - p0;

            // `_eval_xc_eff`, ncol branch: ρ_{a,b} = (ρ ± |m|)/2.
            let s: Vec<f64> = (0..n)
                .map(|g| {
                    // `std` on the host — see the note in `kspu.rs`: cube-math
                    // is a DEVICE libm and panics outside a `#[cube]` expansion.
                    // `f64::sqrt` is IEEE-exact anyway, so there is nothing to
                    // gain here even where cube-math would run.
                    (rho_m[1][g].powi(2) + rho_m[2][g].powi(2) + rho_m[3][g].powi(2)).sqrt()
                })
                .collect();
            let ra = crate::xc::RhoEff {
                nvar: 1,
                ngrids: n,
                data: (0..n).map(|g| (rho_m[0][g] + s[g]) * 0.5).collect(),
            };
            let rb = crate::xc::RhoEff {
                nvar: 1,
                ngrids: n,
                data: (0..n).map(|g| (rho_m[0][g] - s[g]) * 0.5).collect(),
            };
            let out = eval_xc_eff_uks(xc_code, &ra, &rb)?;

            let den: Vec<f64> = (0..n).map(|g| rho_m[0][g] * w[g]).collect();
            nelec += oracle_sum(&den);
            let terms: Vec<f64> = den.iter().zip(&out.exc).map(|(d, e)| d * e).collect();
            excsum += oracle_sum(&terms);

            // `ud2ts`: v_t = (v_u + v_d)/2, v_s = (v_u − v_d)/2.
            // `hermi = 1` folds the usual 0.5 into both.
            let wv: Vec<f64> = (0..n)
                .map(|g| 0.5 * w[g] * 0.5 * (out.row(0, 0)[g] + out.row(1, 0)[g]))
                .collect();
            let ws: Vec<f64> = (0..n)
                .map(|g| {
                    if s[g] < 1e-20 {
                        0.0
                    } else {
                        0.5 * w[g] * 0.5 * (out.row(0, 0)[g] - out.row(1, 0)[g]) / s[g]
                    }
                })
                .collect();

            for (k, m) in vmat.iter_mut().enumerate() {
                ncol_lda_vxc_mat(m, ao1.at(k), &wv, &ws, &rho_m, nao, n);
            }
        }

        // `mat + mat^H` over the FULL 2nao block (r_vxc's symmetrisation).
        for m in vmat.iter_mut() {
            add_conj_transpose(m, 2 * nao);
        }
        Ok(Nr2cResult {
            nelec,
            excsum,
            vmat,
        })
    }

    /// `nr_fxc` — the collinear branch of `numint2c.py:288-322`.
    ///
    /// # Errors
    /// [`PbcDftError`] for a non-collinear treatment, or from the grid loop.
    // `k` indexes the two spin blocks of `v` together.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn nr_fxc(
        &self,
        cell: &Cell,
        grids: &PeriodicGrids,
        xc_code: &str,
        dm0: Option<&KMats>,
        dms: &KMats,
        v_hermi: bool,
    ) -> Result<KMats, PbcDftError> {
        if self.collinear != Collinear::Col {
            return Err(PbcDftError::Core(PyscfRsError::NotYetImplemented {
                phase: 12,
                what: "KNumInt2C.nr_fxc for a non-collinear treatment \
                       (numint2c.py:290-298 requires mcfun)",
            }));
        }
        let nao = cell.mol.nao_nr;
        let (d1a, d1b) = split_diagonal(dms, nao)?;
        let d0 = match dm0 {
            Some(d) => {
                let (a, b) = split_diagonal(d, nao)?;
                Some([a, b])
            }
            None => None,
        };
        let sets: [KDms; 2] = [vec![d1a], vec![d1b]];
        let v = self
            .ni
            .nr_uks_fxc(cell, grids, xc_code, d0.as_ref(), &sets, 0, None, v_hermi)?;
        let nkpts = v[0][0].len();
        let mut out = Vec::with_capacity(nkpts);
        for k in 0..nkpts {
            let mut m = CTensor::zeros(4 * nao * nao);
            set_block(&mut m, &v[0][0][k], nao, 0, 0);
            set_block(&mut m, &v[1][0][k], nao, nao, nao);
            out.push(m);
        }
        Ok(out)
    }
}

/// `(ρ, m_x, m_y, m_z)` on a grid block, BZ-averaged — `_contract_rho_m` with
/// `hermi = 1` and `bra_eq_ket = True` (`dft/numint2c.py:_contract_rho_m`).
// `k` indexes the AO table and the density-matrix list together.
#[allow(clippy::needless_range_loop)]
fn eval_rho_m(
    ao: &EvalAoKptsOutput,
    dms: &KMats,
    nao: usize,
) -> Result<[Vec<f64>; 4], PbcDftError> {
    let nkpts = ao.nkpts();
    if dms.len() != nkpts {
        return Err(err(format!(
            "KNumInt2C: {} density matrices for {nkpts} k-points",
            dms.len()
        )));
    }
    let ng = ao.ngrids;
    let n2 = 2 * nao;
    let mut out = [
        vec![0.0_f64; ng],
        vec![0.0_f64; ng],
        vec![0.0_f64; ng],
        vec![0.0_f64; ng],
    ];
    for k in 0..nkpts {
        let a = ao.at(k);
        let d = &dms[k];
        if d.len() != n2 * n2 {
            return Err(err(
                "KNumInt2C: the density matrix is not 2*nao x 2*nao".to_string()
            ));
        }
        // c_ab[g, j] = Σ_i ao[g, i] D[block_a + i, block_b + j]
        let cx = |ra: usize, rb: usize| -> (Vec<f64>, Vec<f64>) {
            let mut cre = vec![0.0_f64; ng * nao];
            let mut cim = vec![0.0_f64; ng * nao];
            for i in 0..nao {
                let ib = i * ng;
                for j in 0..nao {
                    let (dr, di) = (d.re[(ra + i) * n2 + rb + j], d.im[(ra + i) * n2 + rb + j]);
                    if dr == 0.0 && di == 0.0 {
                        continue;
                    }
                    let jb = j * ng;
                    for g in 0..ng {
                        let (ar, ai) = (a.re[ib + g], a.im[ib + g]);
                        cre[jb + g] += ar * dr - ai * di;
                        cim[jb + g] += ar * di + ai * dr;
                    }
                }
            }
            (cre, cim)
        };
        let (caa_re, caa_im) = cx(0, 0);
        let (cbb_re, cbb_im) = cx(nao, nao);
        let (cab_re, cab_im) = cx(0, nao);

        // r_xy[g] = Σ_j conj(ao[g, j]) c_xy[g, j]
        let contract = |cre: &[f64], cim: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let mut re = vec![0.0_f64; ng];
            let mut im = vec![0.0_f64; ng];
            for j in 0..nao {
                let jb = j * ng;
                for g in 0..ng {
                    let (ar, ai) = (a.re[jb + g], -a.im[jb + g]);
                    re[g] += ar * cre[jb + g] - ai * cim[jb + g];
                    im[g] += ar * cim[jb + g] + ai * cre[jb + g];
                }
            }
            (re, im)
        };
        let (raa, _) = contract(&caa_re, &caa_im);
        let (rbb, _) = contract(&cbb_re, &cbb_im);
        let (rab_re, rab_im) = contract(&cab_re, &cab_im);
        for g in 0..ng {
            out[0][g] += raa[g] + rbb[g];
            out[1][g] += 2.0 * rab_re[g];
            out[2][g] += 2.0 * rab_im[g];
            out[3][g] += raa[g] - rbb[g];
        }
    }
    let inv = 1.0 / nkpts as f64;
    for row in out.iter_mut() {
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
    Ok(out)
}

/// `_ncol_lda_vxc_mat` at one k-point, accumulated into `out`.
fn ncol_lda_vxc_mat(
    out: &mut CTensor,
    ao: &CTensor,
    wv: &[f64],
    ws: &[f64],
    rho_m: &[Vec<f64>; 4],
    nao: usize,
    ng: usize,
) {
    let n2 = 2 * nao;
    // `tmp(f)[μν] = Σ_g conj(ao[g,μ]) f[g] ao[g,ν]`.
    let tmp = |f: &dyn Fn(usize) -> f64| -> CTensor {
        let mut re = vec![0.0_f64; nao * nao];
        let mut im = vec![0.0_f64; nao * nao];
        let mut tr = vec![0.0_f64; ng];
        let mut ti = vec![0.0_f64; ng];
        for mu in 0..nao {
            let mb = mu * ng;
            for nu in 0..nao {
                let nb = nu * ng;
                for g in 0..ng {
                    let (ar, ai) = (ao.re[mb + g], -ao.im[mb + g]);
                    let s = f(g);
                    let (br, bi) = (s * ao.re[nb + g], s * ao.im[nb + g]);
                    tr[g] = ar * br - ai * bi;
                    ti[g] = ar * bi + ai * br;
                }
                re[mu * nao + nu] = oracle_sum(&tr);
                im[mu * nao + nu] = oracle_sum(&ti);
            }
        }
        CTensor::from_planes(re, im)
    };

    let tx = tmp(&|g| ws[g] * rho_m[1][g]);
    let ty = tmp(&|g| ws[g] * rho_m[2][g]);
    let taa = tmp(&|g| wv[g] + ws[g] * rho_m[3][g]);
    let tbb = tmp(&|g| wv[g] - ws[g] * rho_m[3][g]);

    // mat_ba = tx + i·ty ; mat_ab = 0 (the Hermitian branch, `hermi = 1`).
    for i in 0..nao {
        for j in 0..nao {
            let p = i * nao + j;
            out.re[i * n2 + j] += taa.re[p];
            out.im[i * n2 + j] += taa.im[p];
            out.re[(nao + i) * n2 + nao + j] += tbb.re[p];
            out.im[(nao + i) * n2 + nao + j] += tbb.im[p];
            out.re[(nao + i) * n2 + j] += tx.re[p] - ty.im[p];
            out.im[(nao + i) * n2 + j] += tx.im[p] + ty.re[p];
        }
    }
}

/// Split a `2·nao x 2·nao` stack into its two DIAGONAL blocks.
fn split_diagonal(dms: &KMats, nao: usize) -> Result<(KMats, KMats), PbcDftError> {
    let n2 = 2 * nao;
    let mut a = Vec::with_capacity(dms.len());
    let mut b = Vec::with_capacity(dms.len());
    for d in dms {
        if d.len() != n2 * n2 {
            return Err(err(format!(
                "KNumInt2C: density matrix has {} entries, expected {}",
                d.len(),
                n2 * n2
            )));
        }
        a.push(block(d, nao, 0, 0));
        b.push(block(d, nao, nao, nao));
    }
    Ok((a, b))
}

/// The `nao x nao` sub-block of a `2·nao x 2·nao` matrix at `(r0, c0)`.
fn block(m: &CTensor, nao: usize, r0: usize, c0: usize) -> CTensor {
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

/// Write an `nao x nao` block into a `2·nao x 2·nao` matrix at `(r0, c0)`.
fn set_block(dst: &mut CTensor, src: &CTensor, nao: usize, r0: usize, c0: usize) {
    let n2 = 2 * nao;
    for i in 0..nao {
        for j in 0..nao {
            dst.re[(r0 + i) * n2 + c0 + j] = src.re[i * nao + j];
            dst.im[(r0 + i) * n2 + c0 + j] = src.im[i * nao + j];
        }
    }
}

fn add_conj_transpose(m: &mut CTensor, n: usize) {
    let re = m.re.clone();
    let im = m.im.clone();
    for i in 0..n {
        for j in 0..n {
            m.re[i * n + j] = re[i * n + j] + re[j * n + i];
            m.im[i * n + j] = im[i * n + j] - im[j * n + i];
        }
    }
}
