//! Unrestricted k-point MP2 surface. Upstream exposes bookkeeping but refuses
//! its energy kernel (`kump2.py:38,384,402`), which this port mirrors.

use pyscf_pbc_scf::KScfResult;

use crate::{
    FrozenK, KCount, KMoRef, PaddingIdx, PaddingKind, PbcMpError, get_frozen_mask, get_nmo,
    get_nocc, padding_k_idx, spin_block,
};

/// Rust's type system replaces upstream `_is_arraylike`: unrestricted frozen
/// specifications are either shared by both spins or supplied independently.
#[derive(Debug, Clone, PartialEq)]
pub enum FrozenU {
    Both(FrozenK),
    PerSpin(FrozenK, FrozenK),
}

impl Default for FrozenU {
    fn default() -> Self {
        Self::Both(FrozenK::default())
    }
}

pub struct Kump2<'a> {
    pub alpha: KMoRef<'a>,
    pub beta: KMoRef<'a>,
    pub frozen: FrozenU,
}

impl<'a> Kump2<'a> {
    pub fn new(result: &'a KScfResult) -> Result<Self, PbcMpError> {
        if result.nset != 2 {
            return Err(PbcMpError::Shape {
                what: "KUMP2 requires a two-channel KUHF result".into(),
            });
        }
        Ok(Self {
            alpha: spin_block(result, 0)?,
            beta: spin_block(result, 1)?,
            frozen: FrozenU::default(),
        })
    }

    fn frozen_for_spin(&self, spin: usize) -> &FrozenK {
        match &self.frozen {
            FrozenU::Both(frozen) => frozen,
            FrozenU::PerSpin(alpha, beta) => [alpha, beta][spin],
        }
    }

    pub fn get_nocc(&self, per_kpoint: bool) -> Result<[KCount; 2], PbcMpError> {
        Ok([
            get_nocc(self.alpha.mo_occ, self.frozen_for_spin(0), per_kpoint)?,
            get_nocc(self.beta.mo_occ, self.frozen_for_spin(1), per_kpoint)?,
        ])
    }

    pub fn get_nmo(&self, per_kpoint: bool) -> Result<[KCount; 2], PbcMpError> {
        Ok([
            get_nmo(self.alpha.mo_occ, self.frozen_for_spin(0), per_kpoint)?,
            get_nmo(self.beta.mo_occ, self.frozen_for_spin(1), per_kpoint)?,
        ])
    }

    pub fn get_frozen_mask(&self) -> Result<[Vec<Vec<bool>>; 2], PbcMpError> {
        Ok([
            get_frozen_mask(self.alpha.mo_occ, self.frozen_for_spin(0))?,
            get_frozen_mask(self.beta.mo_occ, self.frozen_for_spin(1))?,
        ])
    }

    pub fn padding_k_idx(&self, kind: PaddingKind) -> Result<[PaddingIdx; 2], PbcMpError> {
        let counts = |spin: usize| -> Result<(Vec<usize>, Vec<usize>), PbcMpError> {
            let block = [self.alpha, self.beta][spin];
            let frozen = self.frozen_for_spin(spin);
            let KCount::PerKpoint(nmo) = get_nmo(block.mo_occ, frozen, true)? else {
                unreachable!()
            };
            let KCount::PerKpoint(nocc) = get_nocc(block.mo_occ, frozen, true)? else {
                unreachable!()
            };
            Ok((nmo, nocc))
        };
        let (nmo_a, nocc_a) = counts(0)?;
        let (nmo_b, nocc_b) = counts(1)?;
        Ok([
            padding_k_idx(&nmo_a, &nocc_a, kind)?,
            padding_k_idx(&nmo_b, &nocc_b, kind)?,
        ])
    }

    pub fn dump_flags(&self) -> Result<String, PbcMpError> {
        Ok(format!(
            "KUMP2 nkpts={} nocc={:?} nmo={:?} frozen={:?}",
            self.alpha.mo_occ.len(),
            self.get_nocc(false)?,
            self.get_nmo(false)?,
            self.frozen
        ))
    }

    pub fn add_padding(&self) -> Result<(), PbcMpError> {
        Err(PbcMpError::Kump2NotImplemented)
    }

    pub fn kernel(&self) -> Result<(), PbcMpError> {
        Err(PbcMpError::Kump2NotImplemented)
    }
}

pub type Kump2Alias<'a> = Kump2<'a>;
