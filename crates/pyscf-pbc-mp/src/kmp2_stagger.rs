//! Staggered-submesh KMP2 (`pyscf/pbc/mp/kmp2_stagger.py`).

use pyscf_algebra::{CTensor, oracle_sum, oracle_zdotu_re};
use pyscf_pbc_df::{Fftdf, Gdf, PeriodicDf};
use pyscf_pbc_lib::round_to_fbz;
use pyscf_pbc_scf::{KOverrideHooks, KScfResult, Krhf};
use pyscf_pbc_tools::ExxDiv;

use crate::kmp2_kernel::{LARGE_DENOM, ao2mo_oovv, df_oovv};
use crate::{
    FrozenK, Kmp2, PaddedMos, PaddingIdx, PaddingKind, PbcMpError, add_padding, build_lov,
    mo_coeff_from_kscf, padding_k_idx,
};

#[derive(Debug, Clone, PartialEq)]
pub struct StaggerMesh {
    pub kpts_occ: Vec<[f64; 3]>,
    pub kpts_vir: Vec<[f64; 3]>,
    pub kpts_idx_occ: Vec<usize>,
    pub kpts_idx_vir: Vec<usize>,
}

fn locate(
    cell: &pyscf_pbc_gto::Cell,
    all: &[[f64; 3]],
    subset: &[[f64; 3]],
) -> Result<Vec<usize>, PbcMpError> {
    let all = round_to_fbz(&cell.get_scaled_kpts(all), true, 1e-8);
    let sub = round_to_fbz(&cell.get_scaled_kpts(subset), true, 1e-8);
    let mut out = Vec::with_capacity(sub.len());
    for q in sub {
        let hits: Vec<_> = all
            .iter()
            .enumerate()
            .filter_map(|(i, k)| {
                let d = (k[0] - q[0]).powi(2) + (k[1] - q[1]).powi(2) + (k[2] - q[2]).powi(2);
                (d < 1e-10).then_some(i)
            })
            .collect();
        if hits.len() != 1 {
            return Err(PbcMpError::Shape {
                what: "cannot uniquely locate occupied/virtual stagger submesh".into(),
            });
        }
        out.push(hits[0]);
    }
    if out
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != out.len()
    {
        return Err(PbcMpError::Shape {
            what: "stagger submesh indices are not a bijection".into(),
        });
    }
    Ok(out)
}

pub fn staggered_submesh(
    cell: &pyscf_pbc_gto::Cell,
    kpts: &[[f64; 3]],
    mesh: [usize; 3],
) -> Result<StaggerMesh, PbcMpError> {
    if mesh.iter().any(|n| n % 2 != 0) {
        return Err(PbcMpError::OddStaggerMesh { mesh });
    }
    let half = mesh.map(|n| n / 2);
    let shift = *kpts
        .iter()
        .min_by(|a, b| {
            let aa = a.iter().map(|x| x * x).sum::<f64>();
            let bb = b.iter().map(|x| x * x).sum::<f64>();
            aa.total_cmp(&bb)
        })
        .ok_or_else(|| PbcMpError::Shape {
            what: "empty k-point mesh".into(),
        })?;
    let mut kpts_vir = cell.make_kpts(half)?;
    for k in &mut kpts_vir {
        for d in 0..3 {
            k[d] += shift[d];
        }
    }
    let hs = cell.get_abs_kpts(&[[
        0.5 / half[0] as f64,
        0.5 / half[1] as f64,
        0.5 / half[2] as f64,
    ]])?[0];
    let kpts_occ = kpts_vir
        .iter()
        .map(|k| [k[0] + hs[0], k[1] + hs[1], k[2] + hs[2]])
        .collect::<Vec<_>>();
    Ok(StaggerMesh {
        kpts_idx_occ: locate(cell, kpts, &kpts_occ)?,
        kpts_idx_vir: locate(cell, kpts, &kpts_vir)?,
        kpts_occ,
        kpts_vir,
    })
}

enum StaggerDf<'a> {
    Borrowed(&'a dyn PeriodicDf),
    Owned(Box<dyn PeriodicDf>),
}

impl StaggerDf<'_> {
    fn get(&self) -> &dyn PeriodicDf {
        match self {
            Self::Borrowed(df) => *df,
            Self::Owned(df) => df.as_ref(),
        }
    }
}

pub struct Kmp2Stagger<'a> {
    pub mesh: StaggerMesh,
    pub kpts: Vec<[f64; 3]>,
    pub padded: PaddedMos,
    pub khelper: pyscf_pbc_lib::KptsHelper,
    pub with_df_ints: bool,
    /// Upstream's `flag_submesh` (`kmp2_stagger.py:216`). It is not decoration:
    /// `_init_mp_df_eris_stagger` reuses the mean field's `_cderi` only when it
    /// is `True` (`:165`), and rebuilds a GDF over `mp.kpts` otherwise.
    pub flag_submesh: bool,
    df: StaggerDf<'a>,
}

impl<'a> Kmp2Stagger<'a> {
    pub fn new(mp: Kmp2<'a>, kmesh: [usize; 3]) -> Result<Self, PbcMpError> {
        let mesh = staggered_submesh(mp.cell, mp.kpts, kmesh)?;
        let padded = mp.padded_mos()?;
        Ok(Self {
            mesh,
            kpts: mp.kpts.to_vec(),
            padded,
            khelper: mp.khelper,
            with_df_ints: mp.with_df_ints,
            flag_submesh: true,
            df: StaggerDf::Borrowed(mp.with_df),
        })
    }

    /// Upstream's `flag_submesh = false` constructor (`kmp2_stagger.py:255-277`):
    /// evaluate bands on a half-shifted mesh with a temporary FFTDF/`vcut_sph`
    /// reference, then run MP2 on a builder bound to the combined
    /// occupied+virtual mesh — see [`Kmp2Stagger::integral_df`] for which one.
    pub fn new_full_mesh(
        mf: &Krhf,
        result: &KScfResult,
    ) -> Result<Kmp2Stagger<'static>, PbcMpError> {
        if mf.cell().dimension < 3 {
            return Err(PbcMpError::Shape {
                what: "KMP2 stagger non-submesh get_bands is valid only for dimension == 3".into(),
            });
        }
        let nks = pyscf_pbc_gto::get_monkhorst_pack_size_default(mf.cell(), mf.kpts())?;
        let half = mf.cell().get_abs_kpts(&[[
            0.5 / nks[0] as f64,
            0.5 / nks[1] as f64,
            0.5 / nks[2] as f64,
        ]])?[0];
        let kpts_vir = mf.kpts().to_vec();
        let kpts_occ = kpts_vir
            .iter()
            .map(|k| [k[0] + half[0], k[1] + half[1], k[2] + half[2]])
            .collect::<Vec<_>>();
        let mut kpts = kpts_occ.clone();
        kpts.extend_from_slice(&kpts_vir);

        let temp_df = Fftdf::with_mesh(mf.cell().clone(), mf.kpts(), mf.with_df.mesh())?;
        let mut temp_mf = Krhf::from_df(Box::new(temp_df));
        temp_mf.exxdiv = Some(ExxDiv::VcutSph);
        let (mo_energy, mo_coeff) = temp_mf.get_bands(&kpts, &result.dm)?;
        let (mo_occ, _) = KOverrideHooks::get_occ(&temp_mf, &mo_energy)?;
        let nao = mf.cell().mol.nao_nr;
        let mos = mo_coeff
            .iter()
            .zip(&mo_occ)
            .map(|(c, o)| mo_coeff_from_kscf(c, nao, o.len()))
            .collect::<Result<Vec<_>, _>>()?;
        let padded = add_padding(&mos, &mo_energy, &mo_occ, &FrozenK::default())?;
        let mesh = StaggerMesh {
            kpts_idx_occ: locate(mf.cell(), &kpts, &kpts_occ)?,
            kpts_idx_vir: locate(mf.cell(), &kpts, &kpts_vir)?,
            kpts_occ,
            kpts_vir,
        };
        // `kmp2_stagger.py:279-282` sets `with_df_ints` from the MEAN FIELD's
        // builder regardless of `flag_submesh`; only the *reuse* of its
        // `_cderi` is gated on `flag_submesh` (`:165`). The builder itself is
        // then rebuilt over the combined mesh by `kernel`.
        let with_df_ints = mf.with_df.has_cderi();
        let df: Box<dyn PeriodicDf> = if with_df_ints {
            Box::new(Gdf::new(mf.cell().clone(), &kpts))
        } else {
            Box::new(Fftdf::new(mf.cell().clone(), &kpts)?)
        };
        Ok(Kmp2Stagger {
            mesh,
            khelper: pyscf_pbc_lib::KptsHelper::without_symm_map(&mf.cell().a, &kpts),
            kpts,
            padded,
            with_df_ints,
            flag_submesh: false,
            df: StaggerDf::Owned(df),
        })
    }

    /// The builder the kernel actually reads integrals from.
    ///
    /// Upstream chooses it INSIDE the kernel, not at construction, and the two
    /// branches do not agree with `self.df`:
    ///
    /// * `with_df_ints == false` uses a fresh `df.FFTDF(mp.cell, mp.kpts)`
    ///   (`kmp2_stagger.py:74`) — **even on a GDF mean field**, unlike plain
    ///   `KMP2`, which uses `mp._scf.with_df.ao2mo` (`kmp2.py:92`).
    /// * `with_df_ints == true` reuses the mean field's `_cderi` only when this
    ///   is a submesh run; otherwise it rebuilds `df.GDF(mp.cell, mp.kpts)`
    ///   (`kmp2_stagger.py:165-169`).
    fn integral_df(&self) -> Result<Option<Box<dyn PeriodicDf>>, PbcMpError> {
        let df = self.df.get();
        let same_kpts = df.kpts() == self.kpts.as_slice();
        if self.with_df_ints {
            if self.flag_submesh && df.has_cderi() && same_kpts {
                return Ok(None);
            }
            return Ok(Some(Box::new(Gdf::new(df.cell().clone(), &self.kpts))));
        }
        if df.name() == "FFTDF" && same_kpts {
            return Ok(None);
        }
        Ok(Some(Box::new(Fftdf::new(df.cell().clone(), &self.kpts)?)))
    }

    pub fn kernel(&self) -> Result<f64, PbcMpError> {
        let p = &self.padded;
        let (nk, no, nm) = (self.kpts.len(), p.nocc, p.nmo);
        let nv = nm - no;
        let nov2 = (no * nv).pow(2);
        let nkov = self.mesh.kpts_idx_vir.len();
        let split = match padding_k_idx(&p.nmo_per_kpt, &p.nocc_per_kpt, PaddingKind::Split)? {
            PaddingIdx::Split { occupied, virtuals } => (occupied, virtuals),
            _ => unreachable!(),
        };
        let rebuilt = self.integral_df()?;
        let df = rebuilt.as_deref().unwrap_or_else(|| self.df.get());
        let lov = if self.with_df_ints {
            Some(build_lov(df, &p.mo_coeff, no)?)
        } else {
            None
        };
        let mut terms = Vec::new();
        for &ki in &self.mesh.kpts_idx_occ {
            for &kj in &self.mesh.kpts_idx_occ {
                let mut oovv = Vec::with_capacity(nkov);
                for &ka in &self.mesh.kpts_idx_vir {
                    let kb = self.khelper.kconserv.get(ki, ka, kj) as usize;
                    let mut z = match &lov {
                        Some(l) => df_oovv(l, ki, ka, kj, kb)?,
                        None => ao2mo_oovv(df, &p.mo_coeff, ki, ka, kj, kb, no, nm, nk, None)?,
                    };
                    let rescale = nk as f64 / nkov as f64;
                    z.re.iter_mut().for_each(|x| *x *= rescale);
                    z.im.iter_mut().for_each(|x| *x *= rescale);
                    oovv.push(z);
                }
                for (ika, &ka) in self.mesh.kpts_idx_vir.iter().enumerate() {
                    let kb = self.khelper.kconserv.get(ki, ka, kj) as usize;
                    let ikb = self
                        .mesh
                        .kpts_idx_vir
                        .iter()
                        .position(|&x| x == kb)
                        .ok_or_else(|| PbcMpError::Shape {
                            what: "kconserv leaves virtual stagger submesh".into(),
                        })?;
                    let mut amp = CTensor::zeros(nov2);
                    let mut weighted = CTensor::zeros(nov2);
                    for i in 0..no {
                        for j in 0..no {
                            for a in 0..nv {
                                for b in 0..nv {
                                    let z = ((i * no + j) * nv + a) * nv + b;
                                    let eia = if split.0[ki].contains(&i)
                                        && split.1[ka].iter().take(nv).any(|&x| x == a)
                                    {
                                        p.mo_energy[ki][i] - p.mo_energy[ka][no + a]
                                    } else {
                                        LARGE_DENOM
                                    };
                                    let ejb = if split.0[kj].contains(&j)
                                        && split.1[kb].iter().take(nv).any(|&x| x == b)
                                    {
                                        p.mo_energy[kj][j] - p.mo_energy[kb][no + b]
                                    } else {
                                        LARGE_DENOM
                                    };
                                    let d = eia + ejb;
                                    amp.re[z] = oovv[ika].re[z] / d;
                                    amp.im[z] = -oovv[ika].im[z] / d;
                                    let q = ((i * no + j) * nv + b) * nv + a;
                                    weighted.re[z] = 2.0 * oovv[ika].re[z] - oovv[ikb].re[q];
                                    weighted.im[z] = 2.0 * oovv[ika].im[z] - oovv[ikb].im[q];
                                }
                            }
                        }
                    }
                    terms.push(oracle_zdotu_re(&amp, &weighted));
                }
            }
        }
        Ok(oracle_sum(&terms) / nkov as f64)
    }
}
