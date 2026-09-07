//! The seam between `pyscf-pbc-symm`'s `KsymmArray` world (`Vec<Complex64>`,
//! IBZ block stores) and this crate's [`ZArr`] world (planar `CTensor`, dense
//! full-BZ arrays) — plan 17-09, the CC half.
//!
//! # Why the k-symmetric CC keeps DENSE reads and SPARSE writes
//!
//! `kccsd_rhf_ksymm.py`'s default is `ktensor_direct = False`
//! (`:395`, and the docstring at `:389-394`): every tensor is **built** on the
//! `kqrts_ibz` quartets only and then `todense()`d whenever it fits in memory,
//! because the amplitude equations read `t2[kk, kj, kc]` at arbitrary BZ
//! triples and re-deriving each read through `transform_4d` would cost more
//! than the storage saves. The saving that survives is the one that matters:
//! the **integral transforms** and the **intermediate builds** run over
//! `len(kqrts_ibz)` quartets instead of `nkpts^3` — 120 of 512 on upstream's
//! own He `[2,2,2]` fixture (`measurements/gate_kccsd_ksymm.out`).
//!
//! This port ships that default path. `ktensor_direct = True` — block-sparse
//! reads, every non-IBZ block recomputed on the fly — is a MEMORY option with
//! no effect on any number, and it is deferred rather than shipped untested:
//! see [`crate::kccsd_rhf_ksymm::KsymRccsdOpts`].
//!
//! # The two index maps, and why they are functions and not inline arithmetic
//!
//! A rank-2 tensor is stored per IBZ **k-point** (`kpts.bz2ibz`); a rank-4 one
//! per IBZ **k-quartet** (`kqrts.bz2ibz`, keyed by the flat `(ki, kj, ka)`
//! triple). Upstream's `set_2d`/`set_4d` silently DISCARD a write whose key is
//! not a representative, with a warning (`ktensor.py:245-247`, `:257-261`).
//! Every write in the k-symmetric CC is supposed to land on a representative,
//! so a discard is a defect and not a normal event — [`ibz2_index`] and
//! [`ibz4_index`] return `None` there and every caller turns that into a loud
//! error instead of losing the block.

use num_complex::Complex64;
use pyscf_algebra::CTensor;
use pyscf_pbc_symm::kpts::{KPoints, KQuartets, MORotationMatrix};
use pyscf_pbc_symm::ktensor::{Conj, KsymmArray, KsymmMeta, OrbSpace, SubarrayOrder};

use crate::error::PbcCcError;
use crate::zarr::ZArr;

pub(crate) fn symm_err(e: pyscf_pbc_symm::PbcSymmError) -> PbcCcError {
    PbcCcError::Shape(format!("k-point symmetry: {e}"))
}

/// `ZArr` -> the interleaved `Vec<Complex64>` `KsymmArray` speaks.
pub fn zarr_to_cvec(a: &ZArr) -> Vec<Complex64> {
    a.data()
        .re
        .iter()
        .zip(a.data().im.iter())
        .map(|(&re, &im)| Complex64::new(re, im))
        .collect()
}

/// The inverse of [`zarr_to_cvec`].
///
/// # Errors
/// [`PbcCcError::Shape`] when `v` does not have `shape.product()` elements.
pub fn cvec_to_zarr(shape: &[usize], v: &[Complex64]) -> Result<ZArr, PbcCcError> {
    ZArr::from_ctensor(
        shape,
        CTensor {
            re: v.iter().map(|c| c.re).collect(),
            im: v.iter().map(|c| c.im).collect(),
        },
    )
}

/// A row-major `n x n` rotation block of [`MORotationMatrix`] as a [`ZArr`].
///
/// # Errors
/// [`PbcCcError::Shape`] when `build` was never called, or `k`/`iop` is out of
/// range. **The `iop` case is reachable** with `time_reversal = true`:
/// `stars_ops` indexes `k2opk`'s `2 * nop` column space while `rmat.oo[k]` has
/// `nop` entries (D-17-07-01). Naming it beats folding the index with `% nop`,
/// which would silently apply a different symmetry operation.
pub fn rot(
    rmat: &MORotationMatrix,
    space: OrbSpace,
    k: usize,
    iop: usize,
) -> Result<ZArr, PbcCcError> {
    let (table, n) = match space {
        OrbSpace::Occ => (rmat.oo.as_ref(), rmat.nocc),
        OrbSpace::Vir => (rmat.vv.as_ref(), rmat.nmo - rmat.nocc),
    };
    let table = table
        .ok_or_else(|| PbcCcError::Shape("MORotationMatrix::build was never called".into()))?;
    let row = table
        .get(k)
        .ok_or_else(|| PbcCcError::Shape(format!("MORotationMatrix has no k-point {k}")))?;
    let m = row.get(iop).ok_or_else(|| {
        PbcCcError::Shape(format!(
            "MORotationMatrix at k = {k} has {} operations, but stars_ops asked for {iop} \
             (see D-17-07-01: time_reversal doubles k2opk's column space)",
            row.len()
        ))
    })?;
    cvec_to_zarr(&[n, n], m)
}

/// `label` / `trans` tables for every tensor shape the k-symmetric CC builds,
/// parsed once.
///
/// They are owned here because [`KsymmMeta`] borrows them, and a `&[OrbSpace]`
/// built inside a function that returns a `KsymmArray` would not outlive it.
#[derive(Debug)]
pub struct Labels {
    pub oo: Vec<OrbSpace>,
    pub ov: Vec<OrbSpace>,
    pub vv: Vec<OrbSpace>,
    pub oooo: Vec<OrbSpace>,
    pub ooov: Vec<OrbSpace>,
    pub oovv: Vec<OrbSpace>,
    pub ovov: Vec<OrbSpace>,
    pub voov: Vec<OrbSpace>,
    pub vovo: Vec<OrbSpace>,
    pub vovv: Vec<OrbSpace>,
    pub vvvv: Vec<OrbSpace>,
    /// `'cn'` — the rank-2 intermediates (`kintermediates_rhf_ksymm.py:29`).
    pub cn: Vec<Conj>,
    /// `'nc'` — `t1` (`kccsd_rhf_ksymm.py:434`).
    pub nc: Vec<Conj>,
    /// `'nncc'` — `t2` (`kccsd_rhf_ksymm.py:438`).
    pub nncc: Vec<Conj>,
    /// `'ccnn'` — every `_ERIS` block and every rank-4 intermediate.
    pub ccnn: Vec<Conj>,
}

impl Default for Labels {
    fn default() -> Self {
        use Conj::{C, N};
        use OrbSpace::{Occ as O, Vir as V};
        Self {
            oo: vec![O, O],
            ov: vec![O, V],
            vv: vec![V, V],
            oooo: vec![O, O, O, O],
            ooov: vec![O, O, O, V],
            oovv: vec![O, O, V, V],
            ovov: vec![O, V, O, V],
            voov: vec![V, O, O, V],
            vovo: vec![V, O, V, O],
            vovv: vec![V, O, V, V],
            vvvv: vec![V, V, V, V],
            cn: vec![C, N],
            nc: vec![N, C],
            nncc: vec![N, N, C, C],
            ccnn: vec![C, C, N, N],
        }
    }
}

/// Everything the k-symmetric CC equations index by, bundled so the
/// intermediates do not each take six arguments.
///
/// Every field is BORROWED (17-06-PLAN Task 1): a cloned `KPoints` would
/// silently desynchronise from the one the SCF converged against.
pub struct KsymCtx<'a> {
    pub kpts: &'a KPoints,
    pub kqrts: &'a KQuartets,
    pub rmat: &'a MORotationMatrix,
    pub kconserv: &'a pyscf_pbc_lib::Kconserv,
    pub nocc: usize,
    pub nvir: usize,
    pub labels: Labels,
}

impl<'a> KsymCtx<'a> {
    pub fn new(
        kpts: &'a KPoints,
        kqrts: &'a KQuartets,
        rmat: &'a MORotationMatrix,
        kconserv: &'a pyscf_pbc_lib::Kconserv,
        nocc: usize,
        nvir: usize,
    ) -> Self {
        Self {
            kpts,
            kqrts,
            rmat,
            kconserv,
            nocc,
            nvir,
            labels: Labels::default(),
        }
    }

    /// The `label` table for one `_ERIS` block.
    pub fn label_for(&self, b: crate::keris::Blk) -> &[OrbSpace] {
        use crate::keris::Blk;
        match b {
            Blk::Oooo => &self.labels.oooo,
            Blk::Ooov => &self.labels.ooov,
            Blk::Oovv => &self.labels.oovv,
            Blk::Ovov => &self.labels.ovov,
            Blk::Voov => &self.labels.voov,
            Blk::Vovv => &self.labels.vovv,
            Blk::Vvvv => &self.labels.vvvv,
        }
    }

    pub fn nkpts(&self) -> usize {
        self.kpts.nkpts()
    }

    pub fn nkpts_ibz(&self) -> usize {
        self.kpts.nkpts_ibz()
    }

    /// The IBZ k-quartets, `[ki, kj, ka, kb]`.
    pub fn kqrts_ibz(&self) -> &[[usize; 4]] {
        &self.kqrts.kqrts_ibz
    }

    fn meta(&'a self, label: &'a [OrbSpace], trans: &'a [Conj], rank4: bool) -> KsymmMeta<'a> {
        KsymmMeta {
            kpts: self.kpts,
            kqrts: rank4.then_some(self.kqrts),
            rmat: Some(self.rmat),
            label: Some(label),
            trans: Some(trans),
            incore: true,
        }
    }

    /// A zero-filled rank-2 `KsymmArray` — `nkpts_ibz` blocks of `dims`.
    ///
    /// # Errors
    /// As [`KsymmArray::zeros`].
    pub fn empty2(
        &'a self,
        dims: [usize; 2],
        label: &'a [OrbSpace],
        trans: &'a [Conj],
    ) -> Result<KsymmArray<'a>, PbcCcError> {
        KsymmArray::zeros(&dims, SubarrayOrder::C, self.meta(label, trans, false)).map_err(symm_err)
    }

    /// A zero-filled rank-4 `KsymmArray` — `len(kqrts_ibz)` blocks of `dims`.
    ///
    /// # Errors
    /// As [`KsymmArray::zeros`].
    pub fn empty4(
        &'a self,
        dims: [usize; 4],
        label: &'a [OrbSpace],
        trans: &'a [Conj],
    ) -> Result<KsymmArray<'a>, PbcCcError> {
        KsymmArray::zeros(&dims, SubarrayOrder::C, self.meta(label, trans, true)).map_err(symm_err)
    }

    /// Rebuild a rank-2 `KsymmArray` from an IBZ block store, then unfold it
    /// to a dense `[nkpts, d0, d1]` array — `ktensor.fromraw(...).todense()`.
    ///
    /// # Errors
    /// As [`KsymmArray::from_raw`] / [`KsymmArray::to_dense`].
    pub fn dense2(
        &'a self,
        ibz: &ZArr,
        dims: [usize; 2],
        label: &'a [OrbSpace],
        trans: &'a [Conj],
    ) -> Result<ZArr, PbcCcError> {
        let raw = zarr_to_cvec(ibz);
        let arr = KsymmArray::from_raw(
            &raw,
            &dims,
            SubarrayOrder::C,
            self.meta(label, trans, false),
        )
        .map_err(symm_err)?;
        to_dense2(&arr, self.nkpts(), dims)
    }

    /// The rank-4 counterpart of [`KsymCtx::dense2`].
    ///
    /// # Errors
    /// As [`KsymmArray::from_raw`] / [`KsymmArray::to_dense`].
    pub fn dense4(
        &'a self,
        ibz: &ZArr,
        dims: [usize; 4],
        label: &'a [OrbSpace],
        trans: &'a [Conj],
    ) -> Result<ZArr, PbcCcError> {
        let raw = zarr_to_cvec(ibz);
        let arr =
            KsymmArray::from_raw(&raw, &dims, SubarrayOrder::C, self.meta(label, trans, true))
                .map_err(symm_err)?;
        to_dense4(&arr, self.nkpts(), dims)
    }
}

/// `arr.todense()` for a rank-2 array — shape `[nkpts, d0, d1]`.
///
/// # Errors
/// As [`KsymmArray::to_dense`].
pub fn to_dense2(arr: &KsymmArray<'_>, nkpts: usize, dims: [usize; 2]) -> Result<ZArr, PbcCcError> {
    let flat = arr.to_dense().map_err(symm_err)?;
    cvec_to_zarr(&[nkpts, dims[0], dims[1]], &flat)
}

/// `arr.todense()` for a rank-4 array — shape `[nkpts, nkpts, nkpts, d0..d3]`.
///
/// # Errors
/// As [`KsymmArray::to_dense`].
pub fn to_dense4(arr: &KsymmArray<'_>, nkpts: usize, dims: [usize; 4]) -> Result<ZArr, PbcCcError> {
    let flat = arr.to_dense().map_err(symm_err)?;
    cvec_to_zarr(
        &[nkpts, nkpts, nkpts, dims[0], dims[1], dims[2], dims[3]],
        &flat,
    )
}

/// The IBZ block index of BZ k-point `ki`, or `None` when `ki` is not a
/// representative.
pub fn ibz2_index(kpts: &KPoints, ki: usize) -> Option<usize> {
    let i = *kpts.bz2ibz.get(ki)?;
    (kpts.ibz2bz.get(i) == Some(&ki)).then_some(i)
}

/// The IBZ block index of the k-triple `(ki, kj, ka)`, or `None` when it is
/// not a `kqrts` representative.
pub fn ibz4_index(kpts: &KPoints, kqrts: &KQuartets, klc: [usize; 3]) -> Option<usize> {
    let flat = kpts.ktuple_to_index(&klc);
    let i = *kqrts.bz2ibz.get(flat)?;
    let q = kqrts.kqrts_ibz.get(i)?;
    ([q[0], q[1], q[2]] == klc).then_some(i)
}

/// Read the STORED block at BZ k-point `ki`.
///
/// # Errors
/// [`PbcCcError::Shape`] when `ki` is not an IBZ representative — upstream
/// would silently drop the corresponding write, so this port refuses instead.
pub fn get_stored2(
    arr: &KsymmArray<'_>,
    kpts: &KPoints,
    ki: usize,
    dims: [usize; 2],
) -> Result<ZArr, PbcCcError> {
    let i = ibz2_index(kpts, ki)
        .ok_or_else(|| PbcCcError::Shape(format!("k-point {ki} is not an IBZ representative")))?;
    cvec_to_zarr(&dims, &arr.stored_block(i).map_err(symm_err)?)
}

/// `arr[ki] = value` at an IBZ representative.
///
/// # Errors
/// As [`get_stored2`].
pub fn set_stored2(
    arr: &mut KsymmArray<'_>,
    kpts: &KPoints,
    ki: usize,
    value: &ZArr,
) -> Result<(), PbcCcError> {
    let i = ibz2_index(kpts, ki)
        .ok_or_else(|| PbcCcError::Shape(format!("k-point {ki} is not an IBZ representative")))?;
    arr.set_stored_block(i, &zarr_to_cvec(value))
        .map_err(symm_err)
}

/// `arr[ki] += value`.
///
/// # Errors
/// As [`get_stored2`].
pub fn add_stored2(
    arr: &mut KsymmArray<'_>,
    kpts: &KPoints,
    ki: usize,
    value: &ZArr,
) -> Result<(), PbcCcError> {
    let dims = [arr.subarray_shape()[0], arr.subarray_shape()[1]];
    let mut cur = get_stored2(arr, kpts, ki, dims)?;
    cur.add_assign(value)?;
    set_stored2(arr, kpts, ki, &cur)
}

/// The rank-4 counterpart of [`get_stored2`].
///
/// # Errors
/// [`PbcCcError::Shape`] when `klc` is not a `kqrts` representative.
pub fn get_stored4(
    arr: &KsymmArray<'_>,
    kpts: &KPoints,
    kqrts: &KQuartets,
    klc: [usize; 3],
    dims: [usize; 4],
) -> Result<ZArr, PbcCcError> {
    let i = ibz4_index(kpts, kqrts, klc).ok_or_else(|| {
        PbcCcError::Shape(format!("k-triple {klc:?} is not a kqrts representative"))
    })?;
    cvec_to_zarr(&dims, &arr.stored_block(i).map_err(symm_err)?)
}

/// `arr[ki, kj, ka] = value` at a `kqrts` representative.
///
/// # Errors
/// As [`get_stored4`].
pub fn set_stored4(
    arr: &mut KsymmArray<'_>,
    kpts: &KPoints,
    kqrts: &KQuartets,
    klc: [usize; 3],
    value: &ZArr,
) -> Result<(), PbcCcError> {
    let i = ibz4_index(kpts, kqrts, klc).ok_or_else(|| {
        PbcCcError::Shape(format!("k-triple {klc:?} is not a kqrts representative"))
    })?;
    arr.set_stored_block(i, &zarr_to_cvec(value))
        .map_err(symm_err)
}

/// `arr[ki, kj, ka] += value`.
///
/// # Errors
/// As [`get_stored4`].
pub fn add_stored4(
    arr: &mut KsymmArray<'_>,
    kpts: &KPoints,
    kqrts: &KQuartets,
    klc: [usize; 3],
    value: &ZArr,
) -> Result<(), PbcCcError> {
    let s = arr.subarray_shape();
    let dims = [s[0], s[1], s[2], s[3]];
    let mut cur = get_stored4(arr, kpts, kqrts, klc, dims)?;
    cur.add_assign(value)?;
    set_stored4(arr, kpts, kqrts, klc, &cur)
}
