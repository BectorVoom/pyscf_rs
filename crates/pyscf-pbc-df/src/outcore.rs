//! `outcore` — the blocked, on-disk 3-centre drivers
//! (`pyscf/pbc/df/outcore.py:29-250`), plan 14-05.
//!
//! # Why these exist
//!
//! [`crate::incore::aux_e2`] returns one `(nao_pair, naux)` tensor per `(ki,
//! kj)` pair, all of them in core. For GDF that array is `O(nkpts² · naux ·
//! nao_pair)` — the very quantity `measurements/memory.py` sizes at 3.86 MiB
//! for diamond 2×2×2 and 44.20 MiB at 3×3×3. `aux_e1` and [`aux_e2`] evaluate
//! it in blocks and stream each block to HDF5, so a large cell never holds the
//! whole tensor.
//!
//! # The blocking axis is the k-POINT PAIR here, not the auxiliary shell
//!
//! **Stated deviation.** Upstream blocks over auxiliary shell ranges
//! (`balance_segs(auxdims, buflen)`, `outcore.py:112-115` / `:203-206`) and
//! calls `wrap_int3c` with a `shls_slice` that names them. This port's
//! [`crate::incore::aux_e2`] builds its cintx `BasisSet` from `cell.mol._atom`
//! and `cell.mol._basis` (`pyscf_gto::build_image_expanded_with_aux`), i.e.
//! from the per-ELEMENT parsed basis rather than from a sliceable `_bas` array,
//! so an arbitrary auxiliary shell range is not expressible without
//! synthesising per-atom basis entries. Plan 14-02 found what synthesising
//! per-atom basis data costs when it goes wrong: libcint deduplicates identical
//! basis blocks across atoms of the same element, and the resulting double-scale
//! took `‖j2c‖` to 4495 against upstream's 251.96.
//!
//! The `kptij_lst` axis is already a first-class parameter of `aux_e2`, it is
//! the axis that dominates GDF's footprint (`nkpts²` against the auxiliary
//! basis's `naux`), and blocking on it needs no new integral machinery. So that
//! is what these drivers block on. [`balance_segs`] is ported faithfully and is
//! what computes the partition, so switching the axis later is a one-line
//! change at the call site.
//!
//! # Layout
//!
//! Upstream's, with this port's planar-complex convention (the same one
//! [`crate::gdf::CderiFile`] uses, so one reader serves both):
//!
//! ```text
//! /<dataname>-kptij   (nkptij, 2, 3) f8
//! /<dataname>/<k>     (2, naux, nao_pair)   — aux_e1
//! /<dataname>/<k>     (2, nao_pair, naux)   — aux_e2
//! ```
//!
//! The leading `2` is `[re, im]`. A gamma pair's imaginary plane is written and
//! is all-zero, where upstream writes an `f8` dataset with no imaginary part at
//! all; keeping one shape makes the reader total.

use std::path::{Path, PathBuf};

use pyscf_algebra::CTensor;
use pyscf_chkfile::hdf5;

use crate::error::PbcDfError;
use crate::incore::int3c::KptPair;
use crate::incore::{Aosym, AuxCell, aux_e2_intor};
use pyscf_pbc_gto::Cell;

fn h5err(what: &str, e: impl std::fmt::Display) -> PbcDfError {
    PbcDfError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(format!("outcore: {what}: {e}")),
    ))
}

/// `lib.misc._blocksize_partition(cum, blocksize)` — `pyscf/lib/misc.py:268`.
///
/// Greedy: extend the current segment while the cumulative width stays within
/// `blocksize`, and always emit at least one segment.
fn blocksize_partition(cum: &[usize], blocksize: usize) -> Vec<usize> {
    let n = cum.len() - 1;
    let mut displs = vec![0usize];
    if n == 0 {
        return displs;
    }
    let mut p0 = 0usize;
    for i in 1..n {
        if cum[i + 1] - cum[p0] > blocksize {
            displs.push(i);
            p0 = i;
        }
    }
    displs.push(n);
    displs
}

/// `ao2mo.outcore.balance_segs(segs_lst, blksize, start_id, stop_id)` —
/// `pyscf/ao2mo/outcore.py:764-777`.
///
/// Returns `(i0, i1, width)` triples over the segment list. `width` is the
/// summed size of segments `i0..i1`, which is what a caller sizes its buffer
/// from.
///
/// # Panics
/// Never; an empty `segs` returns an empty task list.
pub fn balance_segs(
    segs: &[usize],
    blksize: usize,
    start_id: usize,
    stop_id: Option<usize>,
) -> Vec<(usize, usize, usize)> {
    let mut loc = Vec::with_capacity(segs.len() + 1);
    loc.push(0usize);
    for s in segs {
        loc.push(loc[loc.len() - 1] + s);
    }
    let stop = stop_id.map_or(loc.len() - 1, |s| s.min(start_id + loc.len() - 1));
    if start_id >= stop {
        return Vec::new();
    }
    let displs = blocksize_partition(&loc[start_id..=stop], blksize.max(1));
    let displs: Vec<usize> = displs.into_iter().map(|i| i + start_id).collect();
    displs
        .windows(2)
        .map(|w| (w[0], w[1], loc[w[1]] - loc[w[0]]))
        .collect()
}

/// Which of the two on-disk orientations a driver writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// `aux_e1` — `(naux, nao_pair)`, the auxiliary index leading.
    AuxLeading,
    /// `aux_e2` — `(nao_pair, naux)`, the AO pair leading. This is the
    /// orientation `_make_j3c` consumes.
    PairLeading,
}

/// The blocking parameters, held explicitly rather than as hidden constants —
/// the plan's requirement, and the thing that makes an out-of-core driver
/// tunable at all.
#[derive(Debug, Clone, Copy)]
pub struct Blocking {
    /// `max_memory` in MB. Upstream's default is 2000.
    pub max_memory: f64,
    /// An explicit block size in k-point pairs, overriding the `max_memory`
    /// estimate. `None` derives it, as upstream does.
    pub blksize: Option<usize>,
}

impl Default for Blocking {
    fn default() -> Self {
        Self {
            max_memory: 2000.0,
            blksize: None,
        }
    }
}

impl Blocking {
    /// How many k-point pairs fit in one block.
    ///
    /// Upstream's `buflen = max(8, int(max_memory*1e6/16/(nkptij*ni*nj*comp)))`
    /// (`outcore.py:112`) sizes an AUXILIARY-shell block against the whole
    /// `kptij` axis; here the roles are swapped, so the same budget divided by
    /// one pair's footprint `16 · nao_pair · naux` bytes gives the number of
    /// pairs. The `max(1, …)` floor is upstream's `max(8, …)` adapted to the
    /// coarser axis: one pair must always be evaluable.
    pub fn pairs_per_block(&self, nao_pair: usize, naux: usize) -> usize {
        if let Some(b) = self.blksize {
            return b.max(1);
        }
        let per = 16.0 * (nao_pair.max(1) * naux.max(1)) as f64;
        let n = (self.max_memory * 1e6 / per).floor();
        if n.is_finite() && n >= 1.0 {
            n as usize
        } else {
            1
        }
    }
}

/// An HDF5 file holding a blocked 3-centre tensor. Deleted on drop unless
/// [`Aux3cFile::keep`] was called — the same RAII-spill contract
/// [`crate::gdf::CderiFile`] uses.
#[derive(Debug)]
pub struct Aux3cFile {
    path: PathBuf,
    dataname: String,
    keep: bool,
    /// `(nao_pair, naux)` of each stored block, in `kptij_lst` order.
    shapes: Vec<(usize, usize)>,
    orientation: Orientation,
}

impl Aux3cFile {
    /// The file's path.
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// The HDF5 group the blocks live under.
    pub fn dataname(&self) -> &str {
        &self.dataname
    }
    /// Which orientation the blocks were written in.
    pub fn orientation(&self) -> Orientation {
        self.orientation
    }
    /// Keep the file when this handle drops.
    pub fn keep(&mut self) {
        self.keep = true;
    }
    /// Number of `(ki, kj)` pairs stored.
    pub fn nkptij(&self) -> usize {
        self.shapes.len()
    }

    /// Read block `k` back, in the orientation it was written.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] on any HDF5 failure or a shape surprise.
    pub fn read(&self, k: usize) -> Result<CTensor, PbcDfError> {
        let f = hdf5::File::open(&self.path).map_err(|e| h5err("open", e))?;
        let g = f
            .group(&self.dataname)
            .map_err(|e| h5err("open group", e))?;
        let d = g
            .dataset(&k.to_string())
            .map_err(|e| h5err("open block", e))?;
        let raw: Vec<f64> = d.read_raw::<f64>().map_err(|e| h5err("read block", e))?;
        if !raw.len().is_multiple_of(2) {
            return Err(h5err("block size", "odd length"));
        }
        let n = raw.len() / 2;
        Ok(CTensor {
            re: raw[..n].to_vec(),
            im: raw[n..].to_vec(),
        })
    }
}

impl Drop for Aux3cFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// `outcore.aux_e1(cell, auxcell, erifile, intor, aosym, kptij_lst, …)` —
/// `outcore.py:29-143`.
///
/// `(L|ij)` with the double lattice sum, written `(naux, nao_pair)` per pair.
///
/// # Errors
/// Propagates [`crate::incore::aux_e2`] and any HDF5 failure.
#[allow(clippy::too_many_arguments)]
pub fn aux_e1(
    cell: &Cell,
    aux: &AuxCell,
    erifile: impl AsRef<Path>,
    intor: &str,
    aosym: Aosym,
    kptij_lst: &[KptPair],
    dataname: &str,
    blocking: Blocking,
    rcut: Option<f64>,
) -> Result<Aux3cFile, PbcDfError> {
    run(
        cell,
        aux,
        erifile,
        intor,
        aosym,
        kptij_lst,
        dataname,
        blocking,
        rcut,
        Orientation::AuxLeading,
    )
}

/// `outcore._aux_e2(cell, auxcell, erifile, intor, aosym, kptij_lst, …)` —
/// `outcore.py:147-250`.
///
/// `(ij|L)` with the double lattice sum, written `(nao_pair, naux)` per pair —
/// the orientation `_make_j3c` consumes, which is why upstream's docstring says
/// this function "should be only used by df and mdf initialization".
///
/// # Errors
/// As [`aux_e1`].
#[allow(clippy::too_many_arguments)]
pub fn aux_e2(
    cell: &Cell,
    aux: &AuxCell,
    erifile: impl AsRef<Path>,
    intor: &str,
    aosym: Aosym,
    kptij_lst: &[KptPair],
    dataname: &str,
    blocking: Blocking,
    rcut: Option<f64>,
) -> Result<Aux3cFile, PbcDfError> {
    run(
        cell,
        aux,
        erifile,
        intor,
        aosym,
        kptij_lst,
        dataname,
        blocking,
        rcut,
        Orientation::PairLeading,
    )
}

#[allow(clippy::too_many_arguments)]
fn run(
    cell: &Cell,
    aux: &AuxCell,
    erifile: impl AsRef<Path>,
    intor: &str,
    aosym: Aosym,
    kptij_lst: &[KptPair],
    dataname: &str,
    blocking: Blocking,
    rcut: Option<f64>,
    orientation: Orientation,
) -> Result<Aux3cFile, PbcDfError> {
    let path = erifile.as_ref().to_path_buf();
    let nao = cell.mol.nao_nr;
    let naux = aux.naux();
    let nao_pair = aosym.nao_pair(nao);

    let f = hdf5::File::create(&path).map_err(|e| h5err("create", e))?;
    // `feri[dataname+'-kptij'] = kptij_lst` — `outcore.py:63`.
    let flat: Vec<f64> = kptij_lst
        .iter()
        .flat_map(|p| p.ki.into_iter().chain(p.kj))
        .collect();
    f.new_dataset::<f64>()
        .shape([kptij_lst.len(), 2, 3])
        .create(format!("{dataname}-kptij").as_str())
        .map_err(|e| h5err("create kptij", e))?
        .write_raw(&flat)
        .map_err(|e| h5err("write kptij", e))?;
    let g = f
        .create_group(dataname)
        .map_err(|e| h5err("create group", e))?;

    let per_block = blocking.pairs_per_block(nao_pair, naux);
    let segs = vec![1usize; kptij_lst.len()];
    let tasks = balance_segs(&segs, per_block, 0, None);

    let mut shapes = Vec::with_capacity(kptij_lst.len());
    for (i0, i1, _) in tasks {
        let mats = aux_e2_intor(cell, aux, intor, aosym, &kptij_lst[i0..i1], rcut)?;
        for (n, m) in mats.iter().enumerate() {
            let k = i0 + n;
            // `aux_e2` returns `(nao_pair, naux)` row-major.
            let (rows, cols, t) = match orientation {
                Orientation::PairLeading => (nao_pair, naux, m.clone()),
                Orientation::AuxLeading => {
                    let mut t = CTensor::zeros(naux * nao_pair);
                    for p in 0..nao_pair {
                        for l in 0..naux {
                            t.re[l * nao_pair + p] = m.re[p * naux + l];
                            t.im[l * nao_pair + p] = m.im[p * naux + l];
                        }
                    }
                    (naux, nao_pair, t)
                }
            };
            let mut buf = Vec::with_capacity(t.re.len() * 2);
            buf.extend_from_slice(&t.re);
            buf.extend_from_slice(&t.im);
            g.new_dataset::<f64>()
                .shape([2, rows, cols])
                .create(k.to_string().as_str())
                .map_err(|e| h5err("create block", e))?
                .write_raw(&buf)
                .map_err(|e| h5err("write block", e))?;
            shapes.push((rows, cols));
        }
    }

    Ok(Aux3cFile {
        path,
        dataname: dataname.to_string(),
        keep: false,
        shapes,
        orientation,
    })
}
