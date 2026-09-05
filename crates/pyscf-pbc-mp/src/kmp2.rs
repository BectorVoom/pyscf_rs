//! Restricted k-point MP2 object (`pyscf/pbc/mp/kmp2.py:692-786`).

use pyscf_pbc_df::{MoCoeff, PeriodicDf};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_scf::KScfResult;

use crate::{
    FrozenK, KMoRef, PaddedMos, PbcMpError, Rdm2, RdmKind, add_padding, kmp2_kernel,
    mo_coeff_from_kscf,
};

#[derive(Debug, Clone)]
pub struct T2 {
    pub nkpts: usize,
    pub nocc: usize,
    pub nvir: usize,
    pub blocks: Vec<pyscf_algebra::CTensor>,
}

impl T2 {
    pub fn block(&self, ki: usize, kj: usize, ka: usize) -> &pyscf_algebra::CTensor {
        &self.blocks[(ki * self.nkpts + kj) * self.nkpts + ka]
    }
}

#[derive(Debug, Clone)]
pub struct Kmp2Result {
    pub e_corr: f64,
    pub e_corr_ss: f64,
    pub e_corr_os: f64,
    pub e_hf: f64,
    pub e_tot: f64,
    pub t2: Option<T2>,
}

pub struct Kmp2<'a> {
    pub mf: KMoRef<'a>,
    pub with_df: &'a dyn PeriodicDf,
    pub cell: &'a Cell,
    pub kpts: &'a [[f64; 3]],
    pub khelper: KptsHelper,
    pub frozen: FrozenK,
    pub with_df_ints: bool,
    pub with_t2: bool,
    pub max_memory: f64,
    e_hf: f64,
    converged: bool,
}

impl<'a> Kmp2<'a> {
    pub fn new(result: &'a KScfResult, with_df: &'a dyn PeriodicDf) -> Result<Self, PbcMpError> {
        if result.nset != 1 || result.nkpts != with_df.kpts().len() {
            return Err(PbcMpError::Shape {
                what: "KMP2 requires one restricted SCF channel matching with_df.kpts()".into(),
            });
        }
        let mf = crate::spin_block(result, 0)?;
        let cell = with_df.cell();
        let kpts = with_df.kpts();
        Ok(Self {
            mf,
            with_df,
            cell,
            kpts,
            // Upstream builds the O(nk^3) symmetry map eagerly, but its KMP2
            // kernel reads only kconserv. Only two of four operations preserve
            // the (ov|ov) pattern, so the <=2x assembly saving is left to KCCSD.
            khelper: KptsHelper::without_symm_map(&cell.a, kpts),
            frozen: FrozenK::default(),
            with_df_ints: with_df.has_cderi(),
            with_t2: true,
            max_memory: 4_000.0,
            e_hf: result.e_tot,
            converged: result.converged,
        })
    }

    pub fn kernel(&self) -> Result<Kmp2Result, PbcMpError> {
        if !self.converged {
            return Err(PbcMpError::UnconvergedReference);
        }
        let padded = self.padded_mos()?;
        let (ss, os, t2) = kmp2_kernel(self, &padded)?;
        let e_corr = ss + os;
        Ok(Kmp2Result {
            e_corr,
            e_corr_ss: ss,
            e_corr_os: os,
            e_hf: self.e_hf,
            e_tot: self.e_hf + e_corr,
            t2,
        })
    }

    pub fn padded_mos(&self) -> Result<PaddedMos, PbcMpError> {
        let nao = self.cell.nao_nr;
        let raw: Result<Vec<MoCoeff>, _> = self
            .mf
            .mo_coeff
            .iter()
            .zip(self.mf.mo_occ)
            .map(|(c, occ)| mo_coeff_from_kscf(c, nao, occ.len()))
            .collect();
        add_padding(&raw?, self.mf.mo_energy, self.mf.mo_occ, &self.frozen)
    }

    pub fn make_rdm1(
        &self,
        t2: &T2,
        kind: RdmKind,
    ) -> Result<Vec<pyscf_algebra::CTensor>, PbcMpError> {
        let padded = self.padded_mos()?;
        crate::make_rdm1(
            t2,
            &self.khelper.kconserv,
            &padded.nmo_per_kpt,
            &padded.nocc_per_kpt,
            kind,
        )
    }

    pub fn make_rdm2(&self, t2: &T2, kind: RdmKind) -> Result<Rdm2, PbcMpError> {
        let padded = self.padded_mos()?;
        crate::make_rdm2(
            t2,
            &self.khelper.kconserv,
            &padded.nmo_per_kpt,
            &padded.nocc_per_kpt,
            kind,
        )
    }
}

pub type Krmp2<'a> = Kmp2<'a>;
