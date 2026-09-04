//! `pyscf/pbc/lib/ktensor.py` (386 l) — [`KsymmArray`], the container that
//! stores only the IBZ-irreducible blocks of a k-indexed tensor and
//! materialises every other block on demand through the MO rotation
//! matrices ([`crate::kpts::MORotationMatrix`]).
//!
//! Plan `.planning/phases/17-ksymm-multigrid/17-06-PLAN.md`. `15-CONTEXT §1.1`
//! moved this file out of Phase 15 and into Phase 17 because its only
//! consumers are `kmp2_ksymm` / `khf_ksymm` / `kccsd_rhf_ksymm`.
//!
//! # What the container is for
//!
//! A k-symmetry CCSD `Wvvvv` is `nkpts^3 x nvir^4`. Storing only the
//! `len(kqrts_ibz)` irreducible quartets is the entire point of the class,
//! and the out-of-core branch ([`Store::Outcore`]) is what keeps even the
//! irreducible part off the heap. **That branch goes through
//! `pyscf_chkfile::hdf5` (decision D-07), never a direct `hdf5-metno`
//! dependency** — `pyscf-pbc-df` had exactly that dependency removed for
//! this reason in plan 14-03 Task 0.
//!
//! # Element type: `Complex64`, not the planar `CTensor` split
//!
//! RULE 8 forbids `Complex<f64>` ACROSS THE ALGEBRA WALL. This crate never
//! crosses it — it declares no `cubecl-*` dependency and calls no
//! `pyscf-algebra` device entry point — and [`crate::kpts`] already stores
//! every row-major complex matrix as `Vec<Complex64>` for the same reason
//! (`kpts.rs`'s `cmatmul` doc). `MORotationMatrix`, whose matrices this
//! module contracts against, is `Vec<Complex64>`; a planar split here would
//! mean converting at every call.
//!
//! Upstream's `dtype` argument (`ktensor.py:26`, default `float`) is
//! therefore not modelled: every `KsymmArray` upstream ever constructs is
//! given `t1.dtype` / `eris.fock.dtype` / `t2.dtype`, all `complex128`, and
//! `empty(shape, dtype, order, metadata=None)` — the branch that could be
//! real — returns a plain `np.empty`, i.e. a dense array, not this type.
//!
//! # `subarray_order` is layout-only, and it round-trips
//!
//! `17-06-PLAN.md` requires the C/F subarray order to be an enum on the
//! struct rather than a runtime string, and to survive [`KsymmArray::from_raw`].
//! It is [`SubarrayOrder`], stored on the struct and returned by
//! [`KsymmArray::subarray_order`].
//!
//! **It never changes a VALUE.** Upstream's `fromraw` reads its input as
//! `arr.reshape(-1, *subarray_shape)` (`ktensor.py:217`) — a C-logical
//! reshape — and then only asks NumPy for a particular MEMORY layout
//! (`np.asarray(a, dtype=dtype, order=order)`, `:219`); `amplitudes_to_vector`
//! reads the result back with `np.concatenate(..., axis=None)`
//! (`kccsd_rhf_ksymm.py:475`), which ravels in C order regardless. So the
//! logical block contents are order-independent upstream, and this port
//! stores every block C-order (row-major) internally while carrying the
//! declared order as metadata. `from_raw(to_raw(x)) == x` holds for both.
//!
//! # `empty` allocates zeros
//!
//! `np.empty` returns uninitialised memory (`ktensor.py:70`). Rust cannot
//! hand out uninitialised `Complex64` safely, so [`KsymmArray::empty`] and
//! [`KsymmArray::zeros`] both zero-fill. The only observable difference is
//! that reading a never-written block yields `0` instead of garbage.

use std::borrow::Cow;

use num_complex::Complex64;
use rayon::prelude::*;

use pyscf_chkfile::H5Complex;
use pyscf_chkfile::hdf5;

use crate::error::PbcSymmError;
use crate::kpts::{KPoints, KQuartets, MORotationMatrix};

// =====================================================================
// Metadata vocabulary
// =====================================================================

/// `ktensor.py:43`'s `subarray_order` — the memory layout each subarray was
/// declared with. See the module doc: it is layout metadata only, and it
/// round-trips through [`KsymmArray::from_raw`] / [`KsymmArray::to_raw`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SubarrayOrder {
    /// Row-major (`order='C'`), upstream's default.
    #[default]
    C,
    /// Column-major (`order='F'`).
    F,
}

/// One character of a `label` string (`'oovv'`, `'ov'`, `'vvvv'`, ...):
/// which MO space that tensor axis lives in, and therefore which block of
/// [`MORotationMatrix`] rotates it (`ktensor.py:272-274`'s
/// `getattr(rmat, pi * 2)` — `'o'` -> `rmat.oo`, `'v'` -> `rmat.vv`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrbSpace {
    /// `'o'` — occupied; rotated by [`MORotationMatrix::oo`].
    Occ,
    /// `'v'` — virtual; rotated by [`MORotationMatrix::vv`].
    Vir,
}

/// One character of a `trans` string (`'nc'`, `'nncc'`, `'ccnn'`, ...):
/// whether that axis's rotation matrix is CONJUGATED before the contraction
/// (`ktensor.py:280-283`, `:309-316`).
///
/// **This is the antiunitary half of the transform and it is the trap the
/// plan names.** `14-VERIFICATION` recorded the same defect class twice in
/// `gen_uniq_kpts_groups`. Every `(label, trans)` combination is tested
/// individually in `tests/ktensor.rs`; none is inferred from another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conj {
    /// `'n'` — use the rotation as is.
    N,
    /// `'c'` — conjugate the rotation (time reversal is antiunitary).
    C,
}

/// Parse a `label` string such as `"ov"` or `"oovv"` (`ktensor.py:272`,
/// `:296`).
///
/// # Errors
/// [`PbcSymmError::KsymmBadMetadataString`] for any character other than
/// `o` / `v`, or a length other than `rank`.
pub fn parse_label(s: &str, rank: usize) -> Result<Vec<OrbSpace>, PbcSymmError> {
    if s.len() != rank {
        return Err(PbcSymmError::KsymmBadMetadataString {
            kind: "label",
            value: s.to_string(),
            reason: "length must equal the subarray rank",
        });
    }
    s.chars()
        .map(|c| match c {
            'o' => Ok(OrbSpace::Occ),
            'v' => Ok(OrbSpace::Vir),
            _ => Err(PbcSymmError::KsymmBadMetadataString {
                kind: "label",
                value: s.to_string(),
                reason: "every character must be 'o' or 'v'",
            }),
        })
        .collect()
}

/// Parse a `trans` string such as `"nc"` or `"nncc"` (`ktensor.py:279`,
/// `:308`).
///
/// # Errors
/// [`PbcSymmError::KsymmBadMetadataString`] for any character other than
/// `n` / `c`, or a length other than `rank`.
pub fn parse_trans(s: &str, rank: usize) -> Result<Vec<Conj>, PbcSymmError> {
    if s.len() != rank {
        return Err(PbcSymmError::KsymmBadMetadataString {
            kind: "trans",
            value: s.to_string(),
            reason: "length must equal the subarray rank",
        });
    }
    s.chars()
        .map(|c| match c {
            'n' => Ok(Conj::N),
            'c' => Ok(Conj::C),
            _ => Err(PbcSymmError::KsymmBadMetadataString {
                kind: "trans",
                value: s.to_string(),
                reason: "every character must be 'n' or 'c'",
            }),
        })
        .collect()
}

/// `ktensor.py:43`'s `metadata` dict, as a typed struct.
///
/// **Every reference is BORROWED, never cloned** (17-06-PLAN.md Task 1):
/// the same argument `17-CONTEXT §3.9` makes for `Symmetry` not owning a
/// `Cell`, and which 17-03 and 17-05 already follow. A cloned `KPoints`
/// would silently desynchronise from the one the SCF is using.
#[derive(Clone, Copy, Debug)]
pub struct KsymmMeta<'a> {
    /// `metadata['kpts']` — required by every path.
    pub kpts: &'a KPoints,
    /// `metadata['kqrts']` — required for rank-4 arrays only
    /// (`ktensor.py:59`, `:142`, `:171`).
    pub kqrts: Option<&'a KQuartets>,
    /// `metadata['rmat']` — required by the `transform_*` paths only. A
    /// `KsymmArray` can be filled and read at its stored keys without one.
    pub rmat: Option<&'a MORotationMatrix>,
    /// `metadata['label']`, parsed. `None` is legal only when no
    /// `transform_*` is ever requested.
    pub label: Option<&'a [OrbSpace]>,
    /// `metadata['trans']`, parsed. Same rule as `label`.
    pub trans: Option<&'a [Conj]>,
    /// `metadata['incore']` (`ktensor.py:50`, default `True`).
    pub incore: bool,
}

impl<'a> KsymmMeta<'a> {
    /// The minimum viable metadata: a `KPoints` and nothing else. Enough for
    /// a rank-2 array that is only ever written and read at its IBZ keys.
    pub fn new(kpts: &'a KPoints) -> Self {
        Self {
            kpts,
            kqrts: None,
            rmat: None,
            label: None,
            trans: None,
            incore: true,
        }
    }

    fn need_kqrts(&self) -> Result<&'a KQuartets, PbcSymmError> {
        self.kqrts
            .ok_or(PbcSymmError::KsymmMissingMetadata("kqrts"))
    }

    fn need_rmat(&self) -> Result<&'a MORotationMatrix, PbcSymmError> {
        self.rmat.ok_or(PbcSymmError::KsymmMissingMetadata("rmat"))
    }

    fn need_label(&self, rank: usize) -> Result<&'a [OrbSpace], PbcSymmError> {
        let l = self
            .label
            .ok_or(PbcSymmError::KsymmMissingMetadata("label"))?;
        if l.len() != rank {
            return Err(PbcSymmError::KsymmBadMetadataString {
                kind: "label",
                value: format!("{l:?}"),
                reason: "length must equal the subarray rank",
            });
        }
        Ok(l)
    }

    fn need_trans(&self, rank: usize) -> Result<&'a [Conj], PbcSymmError> {
        let t = self
            .trans
            .ok_or(PbcSymmError::KsymmMissingMetadata("trans"))?;
        if t.len() != rank {
            return Err(PbcSymmError::KsymmBadMetadataString {
                kind: "trans",
                value: format!("{t:?}"),
                reason: "length must equal the subarray rank",
            });
        }
        Ok(t)
    }
}

// =====================================================================
// Blocks — the read-only view the free `transform_*` functions take
// =====================================================================

/// A flat `n_blocks x block_len` C-order buffer, viewed block by block.
/// This is upstream's `arr` argument (`self.data`) to [`transform_2d`] /
/// [`transform_4d`] / [`set_2d`] / [`set_4d`].
///
/// It is a plain borrowed slice rather than a trait object so the unfold
/// loops can be `rayon` `par_iter`s over a `Sync` view with no locking.
#[derive(Clone, Copy, Debug)]
pub struct Blocks<'d> {
    data: &'d [Complex64],
    block_len: usize,
}

impl<'d> Blocks<'d> {
    /// Wrap `data` as `data.len() / block_len` blocks.
    ///
    /// # Errors
    /// [`PbcSymmError::KsymmShapeMismatch`] when `block_len` is zero or does
    /// not divide `data.len()`.
    pub fn new(data: &'d [Complex64], block_len: usize) -> Result<Self, PbcSymmError> {
        if block_len == 0 || !data.len().is_multiple_of(block_len) {
            return Err(PbcSymmError::KsymmShapeMismatch {
                what: "Blocks buffer",
                expected: block_len,
                got: data.len(),
            });
        }
        Ok(Self { data, block_len })
    }

    /// The number of blocks.
    pub fn n_blocks(&self) -> usize {
        self.data.len() / self.block_len
    }

    /// Block `i`, C-order.
    ///
    /// # Errors
    /// [`PbcSymmError::KsymmIndexOutOfRange`] when `i >= n_blocks()`.
    pub fn block(&self, i: usize) -> Result<&'d [Complex64], PbcSymmError> {
        if i >= self.n_blocks() {
            return Err(PbcSymmError::KsymmIndexOutOfRange(
                i as i64,
                self.n_blocks(),
            ));
        }
        Ok(&self.data[i * self.block_len..(i + 1) * self.block_len])
    }
}

// =====================================================================
// Task 2 — the index algebra (ktensor.py:339-381)
// =====================================================================

/// One component of a NumPy-style key (`ktensor.py:344-353`).
#[derive(Clone, Debug)]
pub enum Key {
    /// `arr[3]` — a single integer.
    Index(i64),
    /// `arr[1:5:2]` — a slice, with upstream's `None` semantics.
    Slice(SliceSpec),
    /// `arr[np.array([0, 3, 1])]` — an explicit 1-d index array.
    Array(Vec<i64>),
}

/// A Python slice, with `None` for the omitted components
/// (`ktensor.py:370-381`).
#[derive(Clone, Copy, Debug, Default)]
pub struct SliceSpec {
    /// `k.start` — `None` means 0; a negative value is folded by `+= n`.
    pub start: Option<i64>,
    /// `k.stop` — `None` means `n`; a negative value is folded by `+= n`.
    pub stop: Option<i64>,
    /// `k.step` — `None` means 1.
    pub step: Option<i64>,
}

impl SliceSpec {
    /// `[:]` — upstream's `self[:]` in `todense` (`ktensor.py:182`).
    pub fn full() -> Self {
        Self::default()
    }
}

/// `ktensor.py:369-381` — `slice_to_coords`. A LITERAL port, including the
/// facts that upstream does NOT clamp `start`/`stop` to `[0, n]` and does
/// NOT special-case a negative `step`: it hands both to `np.arange`.
///
/// # Errors
/// [`PbcSymmError::KsymmZeroStep`] for `step == 0` (NumPy's own
/// `ZeroDivisionError`).
pub fn slice_to_coords(k: SliceSpec, n: usize) -> Result<Vec<i64>, PbcSymmError> {
    let n = n as i64;
    let start = match k.start {
        None => 0,
        Some(s) if s < 0 => s + n,
        Some(s) => s,
    };
    let stop = match k.stop {
        None => n,
        Some(s) if s < 0 => s + n,
        Some(s) => s,
    };
    let step = k.step.unwrap_or(1);
    if step == 0 {
        return Err(PbcSymmError::KsymmZeroStep);
    }
    // np.arange(start, stop, step)
    let mut out = Vec::new();
    let mut v = start;
    if step > 0 {
        while v < stop {
            out.push(v);
            v += step;
        }
    } else {
        while v > stop {
            out.push(v);
            v += step;
        }
    }
    Ok(out)
}

/// The return of [`index_to_coords`] — upstream's `coords`, whose `ndim`
/// the callers branch on (`ktensor.py:149`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coords {
    /// `coords.ndim == 1`: every key component was an integer AND the key
    /// covered every axis (`ktensor.py:365-366`).
    Single(Vec<i64>),
    /// `coords.ndim == 2`: a list of coordinate tuples, LAST AXIS VARYING
    /// FASTEST (`lib.cartesian_prod`).
    Many(Vec<Vec<i64>>),
}

impl Coords {
    /// Every coordinate tuple, whichever variant this is.
    pub fn rows(&self) -> Vec<&[i64]> {
        match self {
            Coords::Single(c) => vec![c.as_slice()],
            Coords::Many(v) => v.iter().map(|r| r.as_slice()).collect(),
        }
    }
}

/// `ktensor.py:339-367` — `index_to_coords`. Expands a partial NumPy key
/// into the full cartesian product of per-axis index lists, padding the
/// omitted trailing axes with `np.arange(n)` (`:359-362`).
///
/// # Errors
/// * [`PbcSymmError::KsymmShapeMismatch`] when `key` is longer than `shape`
///   (upstream's bare `raise RuntimeError`, `:358`).
/// * [`PbcSymmError::KsymmZeroStep`] from [`slice_to_coords`].
pub fn index_to_coords(key: &[Key], shape: &[usize]) -> Result<Coords, PbcSymmError> {
    let ndim = shape.len();
    if key.len() > ndim {
        return Err(PbcSymmError::KsymmShapeMismatch {
            what: "index_to_coords key",
            expected: ndim,
            got: key.len(),
        });
    }

    let mut idxs: Vec<Vec<i64>> = Vec::with_capacity(ndim);
    for (i, k) in key.iter().enumerate() {
        let n = shape[i];
        let idx = match k {
            Key::Slice(s) => slice_to_coords(*s, n)?,
            Key::Index(v) => vec![*v],
            Key::Array(v) => v.clone(),
        };
        idxs.push(idx);
    }
    for item in shape.iter().take(ndim).skip(key.len()) {
        idxs.push((0..*item as i64).collect());
    }

    // lib.cartesian_prod(idxs) — last axis varies fastest.
    let total: usize = idxs.iter().map(|v| v.len()).product();
    let mut coords: Vec<Vec<i64>> = Vec::with_capacity(total);
    let mut row = vec![0i64; ndim];
    fn rec(d: usize, idxs: &[Vec<i64>], row: &mut Vec<i64>, out: &mut Vec<Vec<i64>>) {
        if d == idxs.len() {
            out.push(row.clone());
            return;
        }
        for &v in &idxs[d] {
            row[d] = v;
            rec(d + 1, idxs, row, out);
        }
    }
    rec(0, &idxs, &mut row, &mut coords);

    // `:365-366` — all-integer, full-rank keys collapse to a 1-d coordinate.
    let all_int = key.iter().all(|k| matches!(k, Key::Index(_)));
    if all_int && key.len() == ndim {
        let first = coords.into_iter().next().unwrap_or_default();
        return Ok(Coords::Single(first));
    }
    Ok(Coords::Many(coords))
}

fn checked_index(v: i64, n: usize) -> Result<usize, PbcSymmError> {
    if v < 0 || v as usize >= n {
        return Err(PbcSymmError::KsymmIndexOutOfRange(v, n));
    }
    Ok(v as usize)
}

// =====================================================================
// Task 2 — set_2d / set_4d (ktensor.py:240-264)
// =====================================================================

/// The write half of the block store — upstream's `arr` argument to
/// `set_2d` / `set_4d` (`ktensor.py:240`, `:252`), i.e. `self.data`.
///
/// A trait rather than a slice because the destination may be the HDF5
/// dataset of an out-of-core [`KsymmArray`], which cannot be handed out as
/// a `&mut [Complex64]`. [`FlatBlocks`] is the plain in-memory
/// implementation, and is what the tests drive `set_2d`/`set_4d` with
/// directly.
pub trait BlockSink {
    /// Elements per block.
    fn block_len(&self) -> usize;
    /// Number of blocks.
    fn n_blocks(&self) -> usize;
    /// Overwrite block `i`.
    ///
    /// # Errors
    /// [`PbcSymmError::KsymmIndexOutOfRange`] /
    /// [`PbcSymmError::KsymmShapeMismatch`], and
    /// [`PbcSymmError::KsymmOutcore`] for an out-of-core store.
    fn put_block(&mut self, i: usize, v: &[Complex64]) -> Result<(), PbcSymmError>;
}

/// A plain `n_blocks x block_len` in-memory [`BlockSink`].
#[derive(Debug)]
pub struct FlatBlocks<'d> {
    data: &'d mut [Complex64],
    block_len: usize,
}

impl<'d> FlatBlocks<'d> {
    /// Wrap `data` as `data.len() / block_len` writable blocks.
    ///
    /// # Errors
    /// [`PbcSymmError::KsymmShapeMismatch`] when `block_len` is zero or does
    /// not divide `data.len()`.
    pub fn new(data: &'d mut [Complex64], block_len: usize) -> Result<Self, PbcSymmError> {
        if block_len == 0 || !data.len().is_multiple_of(block_len) {
            return Err(PbcSymmError::KsymmShapeMismatch {
                what: "FlatBlocks buffer",
                expected: block_len,
                got: data.len(),
            });
        }
        Ok(Self { data, block_len })
    }
}

impl BlockSink for FlatBlocks<'_> {
    fn block_len(&self) -> usize {
        self.block_len
    }
    fn n_blocks(&self) -> usize {
        self.data.len() / self.block_len
    }
    fn put_block(&mut self, i: usize, v: &[Complex64]) -> Result<(), PbcSymmError> {
        if i >= self.n_blocks() {
            return Err(PbcSymmError::KsymmIndexOutOfRange(
                i as i64,
                self.n_blocks(),
            ));
        }
        if v.len() != self.block_len {
            return Err(PbcSymmError::KsymmShapeMismatch {
                what: "block",
                expected: self.block_len,
                got: v.len(),
            });
        }
        self.data[i * self.block_len..(i + 1) * self.block_len].copy_from_slice(v);
        Ok(())
    }
}

/// `ktensor.py:240-250` — `set_2d`. `ki` are FULL-BZ k-indices; the values
/// whose key is not an IBZ representative are DISCARDED with a warning,
/// exactly as upstream (`:245-247`).
///
/// # Errors
/// [`PbcSymmError::KsymmShapeMismatch`] on a `value` / `ki` length mismatch
/// or a block of the wrong size; [`PbcSymmError::KsymmIndexOutOfRange`] for
/// a `ki` outside `[0, nkpts)`.
pub fn set_2d(
    arr: &mut dyn BlockSink,
    value: &[&[Complex64]],
    kpts: &KPoints,
    ki: &[usize],
) -> Result<(), PbcSymmError> {
    if value.len() != ki.len() {
        return Err(PbcSymmError::KsymmShapeMismatch {
            what: "set_2d value count",
            expected: ki.len(),
            got: value.len(),
        });
    }
    let mut discarded: Vec<usize> = Vec::new();
    for (m, &k) in ki.iter().enumerate() {
        let _ = checked_index(k as i64, kpts.nkpts())?;
        // mask = np.isin(ki, kpts.ibz2bz)
        if !kpts.ibz2bz.contains(&k) {
            discarded.push(k);
            continue;
        }
        // ki_ibz = kpts.bz2ibz[ki[mask]];  arr[ki_ibz] = value[mask]
        arr.put_block(kpts.bz2ibz[k], value[m])?;
    }
    if !discarded.is_empty() {
        tracing::warn!(
            "Indices {discarded:?} are not in the irreducible wedge. \
             The corresponding data will be discarded."
        );
    }
    Ok(())
}

/// `ktensor.py:252-264` — `set_4d`. `klc[m]` is a `(ki, kj, ka)` triple in
/// the FULL BZ; blocks whose flat tuple index is not a `kqrts` IBZ
/// representative are discarded with a warning (`:257-261`).
///
/// **This is the index map 17-06-PLAN.md Task 2 calls out.** A round-trip
/// test cannot catch a wrong one, because reading uses the same map;
/// `tests/ktensor.rs` therefore compares against an INDEPENDENTLY built
/// dense tensor.
///
/// # Errors
/// As [`set_2d`], plus [`PbcSymmError::KsymmIndexOutOfRange`] for a triple
/// component outside `[0, nkpts)`.
pub fn set_4d(
    arr: &mut dyn BlockSink,
    value: &[&[Complex64]],
    kpts: &KPoints,
    kqrts: &KQuartets,
    klc: &[[usize; 3]],
) -> Result<(), PbcSymmError> {
    if value.len() != klc.len() {
        return Err(PbcSymmError::KsymmShapeMismatch {
            what: "set_4d value count",
            expected: klc.len(),
            got: value.len(),
        });
    }
    let mut discarded: Vec<[usize; 3]> = Vec::new();
    for (m, s) in klc.iter().enumerate() {
        for &c in s.iter() {
            let _ = checked_index(c as i64, kpts.nkpts())?;
        }
        // kk_bz = [kpts.ktuple_to_index(s) for s in klc]
        let kk_bz = kpts.ktuple_to_index(s);
        // mask = np.isin(kk_bz, kqrts.ibz2bz)
        if !kqrts.ibz2bz.contains(&kk_bz) {
            discarded.push(*s);
            continue;
        }
        // kk_ibz = kqrts.bz2ibz[kk_bz[mask]];  arr[kk_ibz] = value[mask]
        arr.put_block(kqrts.bz2ibz[kk_bz], value[m])?;
    }
    if !discarded.is_empty() {
        tracing::warn!(
            "Indices {discarded:?} are not in the irreducible wedge. \
             The corresponding data will be discarded."
        );
    }
    Ok(())
}

// =====================================================================
// Task 3 — transform_2d / transform_4d (ktensor.py:266-337)
// =====================================================================

/// `getattr(rmat, pi * 2)[k][iop]` (`ktensor.py:273-274`, `:297-306`) — the
/// rotation matrix for MO space `sp` at BZ k-point `k` under operation
/// `iop`, plus its dimension.
///
/// # Errors
/// [`PbcSymmError::KsymmMissingMetadata`] if [`MORotationMatrix::build`] was
/// never called; [`PbcSymmError::KptsSymmInputMismatch`] if `k` or `iop` is
/// out of range. The `iop` case is REACHABLE: `stars_ops_bz` indexes the
/// `k2opk` column space, which has `2 * nop` columns when
/// `time_reversal = true`, while `rmat.oo[k]` has `nop` entries. Upstream
/// raises `IndexError` on the same input (`ktensor.py:277`); this port names
/// it rather than folding the index with `% nop`, which would invent a
/// different operation.
fn rot_of(
    rmat: &MORotationMatrix,
    sp: OrbSpace,
    k: usize,
    iop: usize,
) -> Result<(&[Complex64], usize), PbcSymmError> {
    let (blocks, dim, which) = match sp {
        OrbSpace::Occ => (rmat.oo.as_ref(), rmat.nocc, "oo"),
        OrbSpace::Vir => (rmat.vv.as_ref(), rmat.nmo - rmat.nocc, "vv"),
    };
    let blocks = blocks.ok_or(PbcSymmError::KsymmMissingMetadata("rmat (not built)"))?;
    let per_k = blocks.get(k).ok_or_else(|| {
        PbcSymmError::KptsSymmInputMismatch(format!(
            "rmat.{which} has {} k-points, asked for {k}",
            blocks.len()
        ))
    })?;
    let m = per_k.get(iop).ok_or_else(|| {
        PbcSymmError::KptsSymmInputMismatch(format!(
            "rmat.{which}[{k}] has {} operations, asked for {iop} \
             (with time_reversal = true the k2opk column space is twice as \
              wide as `ops`; upstream raises IndexError here)",
            per_k.len()
        ))
    })?;
    if m.len() != dim * dim {
        return Err(PbcSymmError::KsymmShapeMismatch {
            what: "rotation matrix",
            expected: dim * dim,
            got: m.len(),
        });
    }
    Ok((m.as_slice(), dim))
}

/// Row-major `(m x k) @ (k x n)`. Host-only, like `kpts.rs`'s `cmatmul`:
/// this crate never crosses the algebra wall (ALG-06 / RULE 8) and these
/// matrices are `nocc`/`nvir`-sized.
fn zgemm(a: &[Complex64], m: usize, k: usize, b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == Complex64::new(0.0, 0.0) {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    out
}

/// Row-major `m x n` -> `n x m` (NO conjugation — upstream writes
/// `rot_i.T`, never `.conj().T`; the conjugation is the `trans` flag's job).
fn transpose(a: &[Complex64], m: usize, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

fn maybe_conj(a: &[Complex64], t: Conj) -> Cow<'_, [Complex64]> {
    match t {
        Conj::N => Cow::Borrowed(a),
        Conj::C => Cow::Owned(a.iter().map(|z| z.conj()).collect()),
    }
}

/// `(n0, n1, n2)` -> `(n1, n0, n2)`, i.e. NumPy's `transpose(1, 0, 2)`.
fn transpose_102(a: &[Complex64], n0: usize, n1: usize, n2: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n0 * n1 * n2];
    for i in 0..n0 {
        for j in 0..n1 {
            let src = (i * n1 + j) * n2;
            let dst = (j * n0 + i) * n2;
            out[dst..dst + n2].copy_from_slice(&a[src..src + n2]);
        }
    }
    out
}

/// `(nr, n0, n1)` -> `(nr, n1, n0)`, i.e. NumPy's `transpose(0, 2, 1)`.
fn transpose_021(a: &[Complex64], nr: usize, n0: usize, n1: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); nr * n0 * n1];
    for r in 0..nr {
        for i in 0..n0 {
            for j in 0..n1 {
                out[(r * n1 + j) * n0 + i] = a[(r * n0 + i) * n1 + j];
            }
        }
    }
    out
}

/// `ktensor.py:266-287` — `transform_2d`. Materialise the block at FULL-BZ
/// k-point `ki` from the stored block of its IBZ representative.
///
/// `out = rot_i.T @ arr[ki_ibz] @ rot_j`, with each rotation conjugated iff
/// its `trans` character is `'c'`.
///
/// When `ki` IS its own IBZ representative the stored block is returned
/// UNCHANGED (`:269-270`) — this is what makes
/// `transform_2d(block, identity) == block` hold BIT-exactly.
///
/// # Errors
/// As [`rot_of`], plus a block/shape mismatch.
pub fn transform_2d(
    arr: &Blocks<'_>,
    kpts: &KPoints,
    ki: usize,
    rmat: &MORotationMatrix,
    label: &[OrbSpace],
    trans: &[Conj],
    subarray_shape: [usize; 2],
) -> Result<Vec<Complex64>, PbcSymmError> {
    let ki = checked_index(ki as i64, kpts.nkpts())?;
    let ki_ibz = kpts.bz2ibz[ki];
    let ki_ibz_bz = kpts.ibz2bz[ki_ibz];
    let stored = arr.block(ki_ibz)?;
    if ki == ki_ibz_bz {
        return Ok(stored.to_vec());
    }

    let (di, dj) = (subarray_shape[0], subarray_shape[1]);
    if stored.len() != di * dj {
        return Err(PbcSymmError::KsymmShapeMismatch {
            what: "transform_2d block",
            expected: di * dj,
            got: stored.len(),
        });
    }

    let iop = kpts.stars_ops_bz[ki];
    let (rot_i, ni) = rot_of(rmat, label[0], ki_ibz_bz, iop)?;
    let (rot_j, nj) = rot_of(rmat, label[1], ki_ibz_bz, iop)?;
    if ni != di || nj != dj {
        return Err(PbcSymmError::KsymmShapeMismatch {
            what: "transform_2d rotation dimension",
            expected: di * dj,
            got: ni * nj,
        });
    }
    let rot_i = maybe_conj(rot_i, trans[0]);
    let rot_j = maybe_conj(rot_j, trans[1]);

    // reduce(np.dot, (rot_i.T, arr[ki_ibz], rot_j))
    let rit = transpose(&rot_i, di, di);
    let tmp = zgemm(&rit, di, di, stored, dj);
    Ok(zgemm(&tmp, di, dj, &rot_j, dj))
}

/// `ktensor.py:289-337` — `transform_4d`. Materialise the block at the
/// FULL-BZ triple `klc = (ki, kj, ka)` from the stored block of its
/// `kqrts` IBZ representative.
///
/// `out[k,l,c,d] = sum_{i,j,a,b} arr[i,j,a,b] rot_i[i,k] rot_j[j,l]
/// rot_a[a,c] rot_b[b,d]`, each rotation conjugated iff its `trans`
/// character is `'c'`. The four contractions are performed in UPSTREAM'S
/// OWN ORDER (`:322-336`) rather than as one `einsum`, so the operation
/// count and the summation order match line for line.
///
/// When `(i, j, a)` already equals `klc` the stored block is returned
/// unchanged (`:293-294`).
///
/// # Errors
/// As [`rot_of`], plus a block/shape mismatch.
#[allow(clippy::too_many_arguments)]
pub fn transform_4d(
    arr: &Blocks<'_>,
    kpts: &KPoints,
    kqrts: &KQuartets,
    klc: [usize; 3],
    rmat: &MORotationMatrix,
    label: &[OrbSpace],
    trans: &[Conj],
    subarray_shape: [usize; 4],
) -> Result<Vec<Complex64>, PbcSymmError> {
    for &c in klc.iter() {
        let _ = checked_index(c as i64, kpts.nkpts())?;
    }
    let kk_bz = kpts.ktuple_to_index(&klc);
    let kk_ibz = kqrts.bz2ibz[kk_bz];
    let q = kqrts.kqrts_ibz[kk_ibz];
    let stored = arr.block(kk_ibz)?;
    if [q[0], q[1], q[2]] == klc {
        return Ok(stored.to_vec());
    }

    let [di, dj, da, db] = subarray_shape;
    if stored.len() != di * dj * da * db {
        return Err(PbcSymmError::KsymmShapeMismatch {
            what: "transform_4d block",
            expected: di * dj * da * db,
            got: stored.len(),
        });
    }

    let iop = kqrts.stars_ops_bz[kk_bz];
    let (rot_i, ni) = rot_of(rmat, label[0], q[0], iop)?;
    let (rot_j, nj) = rot_of(rmat, label[1], q[1], iop)?;
    let (rot_a, na) = rot_of(rmat, label[2], q[2], iop)?;
    let (rot_b, nb) = rot_of(rmat, label[3], q[3], iop)?;
    if (ni, nj, na, nb) != (di, dj, da, db) {
        return Err(PbcSymmError::KsymmShapeMismatch {
            what: "transform_4d rotation dimensions",
            expected: di * dj * da * db,
            got: ni * nj * na * nb,
        });
    }
    let rot_i = maybe_conj(rot_i, trans[0]);
    let rot_j = maybe_conj(rot_j, trans[1]);
    let rot_a = maybe_conj(rot_a, trans[2]);
    let rot_b = maybe_conj(rot_b, trans[3]);

    // tmp = np.dot(rot_i.T, arr[kk_ibz].reshape(di,-1))          # k,jab
    let rit = transpose(&rot_i, di, di);
    let tmp = zgemm(&rit, di, di, stored, dj * da * db);
    // tmp = tmp.reshape(di,dj,-1).transpose(1,0,2)               # j,k,ab
    let tmp = transpose_102(&tmp, di, dj, da * db);

    // tmp = np.dot(rot_j.T, tmp.reshape(dj,-1))                  # l,kab
    let rjt = transpose(&rot_j, dj, dj);
    let tmp = zgemm(&rjt, dj, dj, &tmp, di * da * db);
    // tmp = tmp.reshape(dj,di,-1).transpose(1,0,2)               # k,l,ab
    let tmp = transpose_102(&tmp, dj, di, da * db);

    // tmp = tmp.reshape(-1,da,db).transpose(0,2,1).reshape(-1,da) # klb,a
    let tmp = transpose_021(&tmp, di * dj, da, db);
    // tmp = np.dot(tmp, rot_a)                                    # klb,c
    let tmp = zgemm(&tmp, di * dj * db, da, &rot_a, da);

    // tmp = tmp.reshape(-1,db,da).transpose(0,2,1).reshape(-1,db) # klc,b
    let tmp = transpose_021(&tmp, di * dj, db, da);
    // out = np.dot(tmp, rot_b).reshape(di,dj,da,db)               # k,l,c,d
    Ok(zgemm(&tmp, di * dj * da, db, &rot_b, db))
}

// =====================================================================
// Task 1 — the container (ktensor.py:26-227)
// =====================================================================

/// The backing store — `ktensor.py:64-81`'s `incore` / `H5TmpFile` branch.
#[derive(Debug)]
enum Store {
    /// `np.zeros/np.empty(shape, dtype, order)` (`:71-77`).
    Incore(Vec<Complex64>),
    /// `lib.H5TmpFile().create_dataset('data', shape, dtype)` (`:79-80`).
    Outcore(OutcoreStore),
}

/// The out-of-core scratch. D-07: the HDF5 handle comes from
/// `pyscf_chkfile::hdf5`, the workspace's sole `hdf5-metno` owner, exactly
/// as `pyscf-ao2mo`'s `OutcoreScratch` does. The temp file is REMOVED on
/// drop (RAII), mirroring upstream's `H5TmpFile()` auto-delete.
#[derive(Debug)]
struct OutcoreStore {
    file: Option<hdf5::File>,
    path: std::path::PathBuf,
    len: usize,
}

impl OutcoreStore {
    const DATASET: &'static str = "data";

    fn next_uid() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static UID: AtomicU64 = AtomicU64::new(0);
        UID.fetch_add(1, Ordering::Relaxed)
    }

    fn create(len: usize) -> Result<Self, PbcSymmError> {
        let uid = Self::next_uid();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "pyscf_ksymm_ktensor_{}_{}.h5",
            std::process::id(),
            uid
        ));

        let file = hdf5::File::create(&path)
            .map_err(|e| PbcSymmError::KsymmOutcore(format!("create {}: {e}", path.display())))?;
        let zeros = vec![H5Complex::default(); len];
        let arr = ndarray::Array1::from_vec(zeros);
        file.new_dataset::<H5Complex>()
            .shape([len])
            .create(Self::DATASET)
            .and_then(|ds| ds.write(&arr))
            .map_err(|e| PbcSymmError::KsymmOutcore(format!("create dataset: {e}")))?;
        Ok(Self {
            file: Some(file),
            path,
            len,
        })
    }

    fn handle(&self) -> Result<&hdf5::File, PbcSymmError> {
        self.file
            .as_ref()
            .ok_or_else(|| PbcSymmError::KsymmOutcore("scratch already closed".to_string()))
    }

    fn read_range(&self, offset: usize, len: usize) -> Result<Vec<Complex64>, PbcSymmError> {
        let end = offset + len;
        if end > self.len {
            return Err(PbcSymmError::KsymmOutcore(format!(
                "read [{offset},{end}) exceeds scratch len {}",
                self.len
            )));
        }
        // Plain `Range<usize>` -> HDF5 hyperslab. We avoid `ndarray::s![..]`
        // for the same reason `pyscf-ao2mo::outcore` does.
        let arr: ndarray::Array1<H5Complex> = self
            .handle()?
            .dataset(Self::DATASET)
            .and_then(|ds| ds.read_slice_1d(offset..end))
            .map_err(|e| PbcSymmError::KsymmOutcore(format!("read: {e}")))?;
        Ok(arr.iter().map(|z| Complex64::new(z.r, z.i)).collect())
    }

    fn write_range(&self, offset: usize, data: &[Complex64]) -> Result<(), PbcSymmError> {
        let end = offset + data.len();
        if end > self.len {
            return Err(PbcSymmError::KsymmOutcore(format!(
                "write [{offset},{end}) exceeds scratch len {}",
                self.len
            )));
        }
        let arr = ndarray::Array1::from_vec(
            data.iter()
                .map(|z| H5Complex { r: z.re, i: z.im })
                .collect::<Vec<_>>(),
        );
        self.handle()?
            .dataset(Self::DATASET)
            .and_then(|ds| ds.write_slice(&arr, offset..end))
            .map_err(|e| PbcSymmError::KsymmOutcore(format!("write: {e}")))?;
        Ok(())
    }
}

impl Drop for OutcoreStore {
    fn drop(&mut self) {
        // Close the handle FIRST so the OS releases the file, then remove it.
        self.file = None;
        let _ = std::fs::remove_file(&self.path);
    }
}

/// `ktensor.py:42-226` — `KsymmArray`.
///
/// Stores only `nkpts_ibz` (rank 2) or `len(kqrts_ibz)` (rank 4) blocks and
/// materialises every other block through [`transform_2d`] /
/// [`transform_4d`].
///
/// The metadata is BORROWED for `'a` — see [`KsymmMeta`].
#[derive(Debug)]
pub struct KsymmArray<'a> {
    meta: KsymmMeta<'a>,
    subarray_shape: Vec<usize>,
    subarray_order: SubarrayOrder,
    block_len: usize,
    n_blocks: usize,
    store: Store,
}

impl BlockSink for KsymmArray<'_> {
    fn block_len(&self) -> usize {
        self.block_len
    }
    fn n_blocks(&self) -> usize {
        self.n_blocks
    }
    fn put_block(&mut self, i: usize, v: &[Complex64]) -> Result<(), PbcSymmError> {
        self.set_stored_block(i, v)
    }
}

impl<'a> KsymmArray<'a> {
    /// `ktensor.py:43-52` + `:54-81` — `__init__` / `_init`, with
    /// `init_with_zeros = True`. See the module doc for why `empty` also
    /// zero-fills.
    ///
    /// # Errors
    /// [`PbcSymmError::KsymmUnsupportedRank`] for a rank other than 2 or 4
    /// (`:62`); [`PbcSymmError::KsymmMissingMetadata`] when a rank-4 array
    /// is built without `kqrts` (`:59`); [`PbcSymmError::KsymmOutcore`] on
    /// an HDF5 failure.
    pub fn empty(
        subarray_shape: &[usize],
        order: SubarrayOrder,
        meta: KsymmMeta<'a>,
    ) -> Result<Self, PbcSymmError> {
        let rank = subarray_shape.len();
        // _init (`:55-62`)
        let n_blocks = match rank {
            2 => meta.kpts.nkpts_ibz(),
            4 => meta.need_kqrts()?.kqrts_ibz.len(),
            _ => return Err(PbcSymmError::KsymmUnsupportedRank(rank)),
        };
        let block_len: usize = subarray_shape.iter().product();
        let total = n_blocks * block_len;
        let store = if meta.incore {
            Store::Incore(vec![Complex64::new(0.0, 0.0); total])
        } else {
            Store::Outcore(OutcoreStore::create(total)?)
        };
        Ok(Self {
            meta,
            subarray_shape: subarray_shape.to_vec(),
            subarray_order: order,
            block_len,
            n_blocks,
            store,
        })
    }

    /// `ktensor.py:222-225` — `zeros`. Identical to [`KsymmArray::empty`] in
    /// this port (see the module doc).
    ///
    /// # Errors
    /// As [`KsymmArray::empty`].
    pub fn zeros(
        subarray_shape: &[usize],
        order: SubarrayOrder,
        meta: KsymmMeta<'a>,
    ) -> Result<Self, PbcSymmError> {
        Self::empty(subarray_shape, order, meta)
    }

    /// `ktensor.py:32-39` — `empty_like`. A NEW array with `a`'s subarray
    /// shape, order and metadata, zero-filled.
    ///
    /// # Errors
    /// As [`KsymmArray::empty`].
    pub fn empty_like(a: &KsymmArray<'a>) -> Result<Self, PbcSymmError> {
        Self::empty(&a.subarray_shape, a.subarray_order, a.meta)
    }

    // ---- accessors (`ktensor.py:83-107`) ----------------------------

    /// `ktensor.py:83-87` — `shape`: `[nkpts] * (rank - 1) ++ subarray_shape`.
    pub fn shape(&self) -> Vec<usize> {
        let nkpts = self.meta.kpts.nkpts();
        let mut s = vec![nkpts; self.subarray_ndim() - 1];
        s.extend_from_slice(&self.subarray_shape);
        s
    }

    /// `ktensor.py:89-91` — `ndim`: `(rank - 1) + rank`.
    pub fn ndim(&self) -> usize {
        self.subarray_ndim() - 1 + self.subarray_ndim()
    }

    /// `ktensor.py:93-95` — `subarray_ndim`.
    pub fn subarray_ndim(&self) -> usize {
        self.subarray_shape.len()
    }

    /// `ktensor.py:97-99` — `subarray_shape`.
    pub fn subarray_shape(&self) -> &[usize] {
        &self.subarray_shape
    }

    /// `ktensor.py:101-103` — `subarray_order`.
    pub fn subarray_order(&self) -> SubarrayOrder {
        self.subarray_order
    }

    /// The number of STORED blocks — `nkpts_ibz` or `len(kqrts_ibz)`.
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// The element count of one subarray.
    pub fn block_len(&self) -> usize {
        self.block_len
    }

    /// Whether the store is in memory (`metadata['incore']`).
    pub fn is_incore(&self) -> bool {
        matches!(self.store, Store::Incore(_))
    }

    /// The metadata this array was built with.
    pub fn meta(&self) -> KsymmMeta<'a> {
        self.meta
    }

    // ---- raw store access -------------------------------------------

    /// The whole flat store, C-order per block. `Cow::Borrowed` for an
    /// incore array (no copy); an incore materialisation of the whole HDF5
    /// dataset for an out-of-core one.
    ///
    /// # Errors
    /// [`PbcSymmError::KsymmOutcore`] on an HDF5 read failure.
    pub fn raw(&self) -> Result<Cow<'_, [Complex64]>, PbcSymmError> {
        match &self.store {
            Store::Incore(v) => Ok(Cow::Borrowed(v.as_slice())),
            Store::Outcore(o) => Ok(Cow::Owned(o.read_range(0, o.len)?)),
        }
    }

    /// `kccsd_rhf_ksymm.py:473-476`'s `np.asarray(t1.data).ravel()` — the
    /// flat buffer `from_raw` reads back. See the module doc: the raveling
    /// is C-logical whatever `subarray_order` says, so this is
    /// [`KsymmArray::raw`] as an owned `Vec`.
    ///
    /// # Errors
    /// As [`KsymmArray::raw`].
    pub fn to_raw(&self) -> Result<Vec<Complex64>, PbcSymmError> {
        Ok(self.raw()?.into_owned())
    }

    /// Stored block `i`, C-order.
    ///
    /// # Errors
    /// [`PbcSymmError::KsymmIndexOutOfRange`] for `i >= n_blocks()`;
    /// [`PbcSymmError::KsymmOutcore`] on an HDF5 read failure.
    pub fn stored_block(&self, i: usize) -> Result<Vec<Complex64>, PbcSymmError> {
        if i >= self.n_blocks {
            return Err(PbcSymmError::KsymmIndexOutOfRange(i as i64, self.n_blocks));
        }
        match &self.store {
            Store::Incore(v) => Ok(v[i * self.block_len..(i + 1) * self.block_len].to_vec()),
            Store::Outcore(o) => o.read_range(i * self.block_len, self.block_len),
        }
    }

    /// Overwrite stored block `i` directly, bypassing the IBZ key mapping.
    /// This is upstream's `out.data[i] = ...` (`ktensor.py:219`), used by
    /// `from_raw`.
    ///
    /// # Errors
    /// As [`KsymmArray::stored_block`], plus
    /// [`PbcSymmError::KsymmShapeMismatch`].
    pub fn set_stored_block(&mut self, i: usize, v: &[Complex64]) -> Result<(), PbcSymmError> {
        if i >= self.n_blocks {
            return Err(PbcSymmError::KsymmIndexOutOfRange(i as i64, self.n_blocks));
        }
        if v.len() != self.block_len {
            return Err(PbcSymmError::KsymmShapeMismatch {
                what: "stored block",
                expected: self.block_len,
                got: v.len(),
            });
        }
        match &mut self.store {
            Store::Incore(d) => {
                d[i * self.block_len..(i + 1) * self.block_len].copy_from_slice(v);
                Ok(())
            }
            Store::Outcore(o) => o.write_range(i * self.block_len, v),
        }
    }

    /// A [`Blocks`] view of the whole store. For an out-of-core array the
    /// dataset is read ONCE into the returned buffer, so the caller pays one
    /// sequential read for a whole unfold instead of one per block (and the
    /// rayon unfold below never touches HDF5 from a worker thread —
    /// `hdf5-metno` is not built thread-safe here).
    fn view(&self) -> Result<(Cow<'_, [Complex64]>, usize), PbcSymmError> {
        Ok((self.raw()?, self.block_len))
    }

    // ---- __setitem__ (`ktensor.py:117-178`) -------------------------

    /// `ktensor.py:158-168` — `_setitem_2d` for a single FULL-BZ key.
    ///
    /// # Errors
    /// As [`set_2d`], plus [`PbcSymmError::KsymmUnsupportedRank`].
    pub fn set_2d_at(&mut self, ki: usize, value: &[Complex64]) -> Result<(), PbcSymmError> {
        self.set_2d_many(&[ki], &[value])
    }

    /// `ktensor.py:158-168` — `_setitem_2d` for a list of FULL-BZ keys.
    ///
    /// # Errors
    /// As [`set_2d`].
    pub fn set_2d_many(
        &mut self,
        ki: &[usize],
        value: &[&[Complex64]],
    ) -> Result<(), PbcSymmError> {
        if self.subarray_ndim() != 2 {
            return Err(PbcSymmError::KsymmUnsupportedRank(self.subarray_ndim()));
        }
        // `KsymmMeta` is `Copy` and its `kpts` borrow lives for `'a`, so this
        // does NOT alias the `&mut self` below.
        let kpts = self.meta.kpts;
        set_2d(self, value, kpts, ki)
    }

    /// `ktensor.py:170-178` — `_setitem_4d` for a single FULL-BZ triple.
    ///
    /// # Errors
    /// As [`set_4d`], plus [`PbcSymmError::KsymmUnsupportedRank`] and
    /// [`PbcSymmError::KsymmMissingMetadata`].
    pub fn set_4d_at(&mut self, klc: [usize; 3], value: &[Complex64]) -> Result<(), PbcSymmError> {
        self.set_4d_many(&[klc], &[value])
    }

    /// `ktensor.py:170-178` — `_setitem_4d` for a list of FULL-BZ triples.
    ///
    /// # Errors
    /// As [`set_4d`].
    pub fn set_4d_many(
        &mut self,
        klc: &[[usize; 3]],
        value: &[&[Complex64]],
    ) -> Result<(), PbcSymmError> {
        if self.subarray_ndim() != 4 {
            return Err(PbcSymmError::KsymmUnsupportedRank(self.subarray_ndim()));
        }
        let kqrts = self.meta.need_kqrts()?;
        let kpts = self.meta.kpts;
        set_4d(self, value, kpts, kqrts, klc)
    }

    // ---- __getitem__ (`ktensor.py:109-156`) -------------------------

    /// `ktensor.py:125-138` — `_getitem_2d` for one FULL-BZ key.
    ///
    /// # Errors
    /// As [`transform_2d`].
    pub fn get_2d(&self, ki: usize) -> Result<Vec<Complex64>, PbcSymmError> {
        Ok(self.get_2d_many(&[ki])?.remove(0))
    }

    /// `ktensor.py:132-136` — `_getitem_2d` for a list of FULL-BZ keys.
    ///
    /// **SPEED (17-06-PLAN.md Task 3):** each key writes exactly one output
    /// slot and reads a shared immutable view, so the loop is a `rayon`
    /// `par_iter().map().collect()` — disjoint by construction, no
    /// reduction, hence no `oracle_sum` ordering to protect (the same
    /// argument 17-05's star unfolds make). `collect()` restores key order,
    /// so the result is bit-identical at any worker count; `tests/ktensor.rs`
    /// proves that with explicit 1- and 8-worker pools.
    ///
    /// # Errors
    /// As [`transform_2d`], plus [`PbcSymmError::KsymmUnsupportedRank`] and
    /// [`PbcSymmError::KsymmMissingMetadata`].
    pub fn get_2d_many(&self, ki: &[usize]) -> Result<Vec<Vec<Complex64>>, PbcSymmError> {
        if self.subarray_ndim() != 2 {
            return Err(PbcSymmError::KsymmUnsupportedRank(self.subarray_ndim()));
        }
        let rmat = self.meta.need_rmat()?;
        let label = self.meta.need_label(2)?;
        let trans = self.meta.need_trans(2)?;
        let shape = [self.subarray_shape[0], self.subarray_shape[1]];
        let (buf, block_len) = self.view()?;
        let blocks = Blocks::new(buf.as_ref(), block_len)?;
        ki.par_iter()
            .map(|&k| transform_2d(&blocks, self.meta.kpts, k, rmat, label, trans, shape))
            .collect()
    }

    /// `ktensor.py:140-156` — `_getitem_4d` for one FULL-BZ triple.
    ///
    /// # Errors
    /// As [`transform_4d`].
    pub fn get_4d(&self, klc: [usize; 3]) -> Result<Vec<Complex64>, PbcSymmError> {
        Ok(self.get_4d_many(&[klc])?.remove(0))
    }

    /// `ktensor.py:152-156` — `_getitem_4d` for a list of FULL-BZ triples.
    /// Parallelised exactly as [`KsymmArray::get_2d_many`].
    ///
    /// # Errors
    /// As [`transform_4d`], plus [`PbcSymmError::KsymmUnsupportedRank`] and
    /// [`PbcSymmError::KsymmMissingMetadata`].
    pub fn get_4d_many(&self, klc: &[[usize; 3]]) -> Result<Vec<Vec<Complex64>>, PbcSymmError> {
        if self.subarray_ndim() != 4 {
            return Err(PbcSymmError::KsymmUnsupportedRank(self.subarray_ndim()));
        }
        let kqrts = self.meta.need_kqrts()?;
        let rmat = self.meta.need_rmat()?;
        let label = self.meta.need_label(4)?;
        let trans = self.meta.need_trans(4)?;
        let shape = [
            self.subarray_shape[0],
            self.subarray_shape[1],
            self.subarray_shape[2],
            self.subarray_shape[3],
        ];
        let (buf, block_len) = self.view()?;
        let blocks = Blocks::new(buf.as_ref(), block_len)?;
        klc.par_iter()
            .map(|&c| transform_4d(&blocks, self.meta.kpts, kqrts, c, rmat, label, trans, shape))
            .collect()
    }

    /// `ktensor.py:109-123` — `__getitem__` driven by a NumPy-style key,
    /// through [`index_to_coords`]. `self[:]` is `key = [Key::Slice(full)]`.
    ///
    /// Returns one block per coordinate row, in `index_to_coords` order.
    ///
    /// # Errors
    /// As [`index_to_coords`] and the rank-specific getters.
    pub fn get(&self, key: &[Key]) -> Result<Vec<Vec<Complex64>>, PbcSymmError> {
        let nkpts = self.meta.kpts.nkpts();
        match self.subarray_ndim() {
            2 => {
                let coords = index_to_coords(key, &[nkpts])?;
                let ki: Result<Vec<usize>, PbcSymmError> = coords
                    .rows()
                    .iter()
                    .map(|r| checked_index(r[0], nkpts))
                    .collect();
                self.get_2d_many(&ki?)
            }
            4 => {
                let coords = index_to_coords(key, &[nkpts, nkpts, nkpts])?;
                let mut klc = Vec::new();
                for r in coords.rows() {
                    klc.push([
                        checked_index(r[0], nkpts)?,
                        checked_index(r[1], nkpts)?,
                        checked_index(r[2], nkpts)?,
                    ]);
                }
                self.get_4d_many(&klc)
            }
            r => Err(PbcSymmError::KsymmUnsupportedRank(r)),
        }
    }

    // ---- todense / fromdense / fromraw (`ktensor.py:180-220`) -------

    /// `ktensor.py:180-182` — `todense`: `self[:].reshape(self.shape)`, as a
    /// flat C-order buffer of `shape().iter().product()` elements.
    ///
    /// # Errors
    /// As [`KsymmArray::get`].
    pub fn to_dense(&self) -> Result<Vec<Complex64>, PbcSymmError> {
        let blocks = self.get(&[Key::Slice(SliceSpec::full())])?;
        let mut out = Vec::with_capacity(blocks.len() * self.block_len);
        for b in blocks {
            out.extend_from_slice(&b);
        }
        Ok(out)
    }

    /// `ktensor.py:184-206` — `fromdense`. `arr` is the flat C-order dense
    /// array of [`KsymmArray::shape`].
    ///
    /// # DEVIATION D-17-06-01 — upstream's `fromdense` writes to the wrong keys
    ///
    /// Upstream's rank-2 branch (`:194-198`) is
    /// ```text
    /// for ki in kpts.ibz2bz:
    ///     ki_ibz = kpts.bz2ibz[ki]
    ///     out[ki_ibz] = arr[ki]
    /// ```
    /// but `__setitem__` -> `set_2d` treats its key as a FULL-BZ index
    /// (`:244-250`: `np.isin(ki, kpts.ibz2bz)` then `kpts.bz2ibz[ki]`). So
    /// upstream passes an already-mapped IBZ index where a BZ index is
    /// expected: every block whose IBZ index is not COINCIDENTALLY also a BZ
    /// index inside the wedge is silently dropped with the "not in the
    /// irreducible wedge" warning.
    ///
    /// The rank-4 branch (`:199-203`) is broken the same way and more
    /// visibly: `out[m] = arr[ki, kj, ka]` with an integer `m` makes
    /// `index_to_coords(m, [nkpts]*3)` pad the two missing axes with
    /// `arange(nkpts)`, producing `nkpts**2` coordinates for ONE value block.
    ///
    /// `fromdense` has NO caller and NO test in the vendored tree
    /// (`grep -rn 'fromdense' pyscf/` finds only its definition and the
    /// module-level alias, `:386`), which is why neither bug is visible
    /// upstream. This port writes at the keys `set_2d`/`set_4d` actually
    /// expect — `out[ki] = arr[ki]` and `out[ki,kj,ka] = arr[ki,kj,ka]` over
    /// the IBZ representatives — so that
    /// `from_dense(to_dense(x)) == x` holds bit-exactly, which is what
    /// 17-06-PLAN.md Task 1 asks for and what upstream's version cannot
    /// satisfy.
    ///
    /// # Errors
    /// [`PbcSymmError::KsymmShapeMismatch`] when `arr` is not
    /// `shape().product()` long; otherwise as [`KsymmArray::empty`].
    pub fn from_dense(
        arr: &[Complex64],
        subarray_shape: &[usize],
        order: SubarrayOrder,
        meta: KsymmMeta<'a>,
    ) -> Result<Self, PbcSymmError> {
        let mut out = Self::empty(subarray_shape, order, meta)?;
        let expected: usize = out.shape().iter().product();
        if arr.len() != expected {
            return Err(PbcSymmError::KsymmShapeMismatch {
                what: "from_dense input",
                expected,
                got: arr.len(),
            });
        }
        let nkpts = meta.kpts.nkpts();
        let bl = out.block_len;
        match out.subarray_ndim() {
            2 => {
                let keys: Vec<usize> = meta.kpts.ibz2bz.clone();
                let vals: Vec<&[Complex64]> = keys
                    .iter()
                    .map(|&ki| &arr[ki * bl..(ki + 1) * bl])
                    .collect();
                out.set_2d_many(&keys, &vals)?;
            }
            4 => {
                let kqrts = meta.need_kqrts()?;
                let mut keys: Vec<[usize; 3]> = Vec::with_capacity(kqrts.kqrts_ibz.len());
                let mut offs: Vec<usize> = Vec::with_capacity(kqrts.kqrts_ibz.len());
                for kq in kqrts.kqrts_ibz.iter() {
                    let (ki, kj, ka) = (kq[0], kq[1], kq[2]);
                    keys.push([ki, kj, ka]);
                    offs.push(((ki * nkpts + kj) * nkpts + ka) * bl);
                }
                let vals: Vec<&[Complex64]> = offs.iter().map(|&o| &arr[o..o + bl]).collect();
                out.set_4d_many(&keys, &vals)?;
            }
            r => return Err(PbcSymmError::KsymmUnsupportedRank(r)),
        }
        Ok(out)
    }

    /// `ktensor.py:208-220` — `fromraw`. `arr` is the flat
    /// `n_blocks * block_len` buffer of ALREADY-IRREDUCIBLE blocks (what
    /// `vector_to_amplitudes` hands back, `kccsd_rhf_ksymm.py:488-496`), and
    /// `order` is the declared subarray order it round-trips.
    ///
    /// # Errors
    /// [`PbcSymmError::KsymmShapeMismatch`] on a length mismatch; otherwise
    /// as [`KsymmArray::empty`].
    pub fn from_raw(
        arr: &[Complex64],
        subarray_shape: &[usize],
        order: SubarrayOrder,
        meta: KsymmMeta<'a>,
    ) -> Result<Self, PbcSymmError> {
        let mut out = Self::empty(subarray_shape, order, meta)?;
        let expected = out.n_blocks * out.block_len;
        if arr.len() != expected {
            return Err(PbcSymmError::KsymmShapeMismatch {
                what: "from_raw input",
                expected,
                got: arr.len(),
            });
        }
        let bl = out.block_len;
        for i in 0..out.n_blocks {
            out.set_stored_block(i, &arr[i * bl..(i + 1) * bl])?;
        }
        Ok(out)
    }
}
