//! The on-disk `_cderi` store — `pyscf/pbc/df/df.py:338-400` (`sr_loop`),
//! `:568-611` (`get_naoaux`), `:613-746` (`CDERIArray`, `_load3c`).
//! Plan 14-03, Task 1.
//!
//! # Why GDF is worth having
//!
//! FFTDF holds an AO table of `nkpts · ngrids · nao` complex numbers — 62.5 MiB
//! for diamond 2×2×2 at mesh `[40,40,40]`. GDF holds `cderi`:
//! `nkpts² · naux · nao_pair`, **3.86 MiB** for the same system, and it lives on
//! disk rather than in core. That is the 6.17 % this phase's Gate 4 measures
//! (`measurements/memory.py`).
//!
//! The ratio is `O(nkpts)` though — 20.95 % at 3×3×3 — which is why Gate 4
//! names its k-mesh.
//!
//! # Layout
//!
//! Upstream's, so a file this port writes is one Phase 15/16 (and PySCF) can
//! read:
//!
//! ```text
//! /kpts                    (nkpts, 3) f8
//! /aosym                   's1' | 's2'
//! /j3c/<ki*nkpts+kj>/<step>   (rank, ncol) — real at gamma, else interleaved
//! /j3c-/<ki*nkpts+kj>/<step>  the NEGATIVE branch (2-D truncated Coulomb only)
//! ```
//!
//! This port writes ONE step per k-pair; upstream splits by `_guess_shell_ranges`
//! when memory demands it, and the reader concatenates either way.
//!
//! # D-07
//!
//! HDF5 is reached through `pyscf_chkfile::hdf5`, never a direct `hdf5-metno`
//! dependency — `pyscf-chkfile` is the sole owner. Plan 14-03 Task 0 removed the
//! unused direct dep this crate had carried since the Phase-9 scaffolding.

use std::path::{Path, PathBuf};

use pyscf_algebra::CTensor;
use pyscf_chkfile::hdf5;

use crate::error::PbcDfError;
use crate::gdf_builder::j3c::{Cderi, CderiBlock};
use crate::incore::Aosym;

fn h5err(what: &str, e: impl std::fmt::Display) -> PbcDfError {
    PbcDfError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(format!("_cderi store: {what}: {e}")),
    ))
}

/// One `(LpqR, LpqI, sign)` yield of [`sr_loop`] — upstream's generator item.
#[derive(Debug, Clone)]
pub struct SrBlock {
    /// `Lpq` real part, `(naux_slice, ncol)` row-major.
    pub re: Vec<f64>,
    /// `Lpq` imaginary part. All-zero (but present) when the pair is real.
    pub im: Vec<f64>,
    /// Number of auxiliary rows in this slice.
    pub naux: usize,
    /// Number of `(mu, nu)` columns.
    pub ncol: usize,
    /// `+1` for the positive-definite branch, `-1` for the 2-D negative one.
    pub sign: i32,
}

/// `sr_loop(kpti_kptj, max_memory, compact, blksize, aux_slice)` —
/// `df.py:338-400`.
///
/// `compact = true` returns the `s2`-packed columns, `false` the full `nao²`
/// square. The conversion is where upstream's one genuinely subtle line lives:
/// unpacking the IMAGINARY part uses `lib.ANTIHERMI`, i.e. the upper triangle is
/// the NEGATED lower one. It is invisible at gamma, where the imaginary part is
/// zero, and wrong everywhere else — hence the `k != 0` test in
/// `tests/gdf.rs`.
///
/// # Errors
/// [`PbcDfError::Core`] when the pair was never computed.
pub fn sr_loop(
    cderi: &Cderi,
    ki: usize,
    kj: usize,
    nao: usize,
    compact: bool,
) -> Result<Vec<SrBlock>, PbcDfError> {
    // PySCF's v1 cderi store contains the lower-triangular k-pairs.  `_load3c`
    // serves an upper-triangular request from the reverse pair by applying
    // L(ki,kj)[mu,nu] = conj(L(kj,ki)[nu,mu]).  Builders in this port retain
    // both directions for convenience, but using the independently evaluated
    // upper block loses that exact identity and makes the KMP2 Lov route differ
    // from `df_ao2mo` at self-inverse momentum transfers.
    let reverse = ki < kj;
    let (bi, bj) = if reverse { (kj, ki) } else { (ki, kj) };
    let b = cderi.get(bi, bj).ok_or_else(|| {
        PbcDfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!(
                "sr_loop: no cderi block for the k-pair ({ki}, {kj}); build() with \
                 j_only = false to get every pair"
            )),
        ))
    })?;
    let mut first = reshape_block(b, cderi.aosym, nao, compact)?;
    if reverse {
        reverse_pair_in_place(&mut first, nao, compact);
    }
    let mut out = vec![first];
    if let Some(neg) = &b.negative {
        let nb = CderiBlock {
            data: neg.clone(),
            rank: neg.re.len() / b.nao_pair,
            nao_pair: b.nao_pair,
            negative: None,
        };
        let mut m = reshape_block(&nb, cderi.aosym, nao, compact)?;
        if reverse {
            reverse_pair_in_place(&mut m, nao, compact);
        }
        m.sign = -1;
        out.push(m);
    }
    Ok(out)
}

fn reverse_pair_in_place(block: &mut SrBlock, nao: usize, compact: bool) {
    if compact {
        // A packed Hermitian AO pair is unchanged by transpose; conjugation
        // remains load-bearing away from gamma.
        block.im.iter_mut().for_each(|v| *v = -*v);
        return;
    }
    let mut re = vec![0.0; block.re.len()];
    let mut im = vec![0.0; block.im.len()];
    for l in 0..block.naux {
        let base = l * nao * nao;
        for mu in 0..nao {
            for nu in 0..nao {
                let dst = base + mu * nao + nu;
                let src = base + nu * nao + mu;
                re[dst] = block.re[src];
                im[dst] = -block.im[src];
            }
        }
    }
    block.re = re;
    block.im = im;
}

/// `pack_tril` / `unpack_tril(..., ANTIHERMI)` on one block.
fn reshape_block(
    b: &CderiBlock,
    stored: Aosym,
    nao: usize,
    compact: bool,
) -> Result<SrBlock, PbcDfError> {
    let want = if compact { Aosym::S2 } else { Aosym::S1 };
    let ncol = want.nao_pair(nao);
    if stored == want {
        return Ok(SrBlock {
            re: b.data.re.clone(),
            im: b.data.im.clone(),
            naux: b.rank,
            ncol,
            sign: 1,
        });
    }
    let mut re = vec![0.0_f64; b.rank * ncol];
    let mut im = vec![0.0_f64; b.rank * ncol];
    for l in 0..b.rank {
        for mu in 0..nao {
            for nu in 0..nao {
                let (lo, hi) = if mu >= nu { (nu, mu) } else { (mu, nu) };
                let tri = hi * (hi + 1) / 2 + lo;
                let sq = mu * nao + nu;
                match (stored, want) {
                    (Aosym::S1, Aosym::S2) => {
                        if mu >= nu {
                            re[l * ncol + tri] = b.data.re[l * b.nao_pair + sq];
                            im[l * ncol + tri] = b.data.im[l * b.nao_pair + sq];
                        }
                    }
                    (Aosym::S2, Aosym::S1) => {
                        re[l * ncol + sq] = b.data.re[l * b.nao_pair + tri];
                        // `lib.ANTIHERMI`: the upper triangle is the NEGATED
                        // lower one. Zero at gamma, load-bearing elsewhere.
                        im[l * ncol + sq] = if mu >= nu {
                            b.data.im[l * b.nao_pair + tri]
                        } else {
                            -b.data.im[l * b.nao_pair + tri]
                        };
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(SrBlock {
        re,
        im,
        naux: b.rank,
        ncol,
        sign: 1,
    })
}

/// `get_naoaux()` — `df.py:568-611`.
///
/// # Errors
/// As [`Cderi::naoaux`].
pub fn get_naoaux(cderi: &Cderi) -> Result<usize, PbcDfError> {
    cderi.naoaux()
}

/// An HDF5-backed `_cderi` file. Deleted on drop unless the caller asked for it
/// to be kept — the same RAII-spill contract `pyscf-runtime`'s `SpillHandle`
/// uses.
#[derive(Debug)]
pub struct CderiFile {
    path: PathBuf,
    keep: bool,
}

impl CderiFile {
    /// The file's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Keep the file when this handle drops — upstream's `_cderi_to_save`.
    pub fn keep(&mut self) {
        self.keep = true;
    }

    /// Write `cderi` to `path` in upstream's layout.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] on any HDF5 failure.
    pub fn save(cderi: &Cderi, path: impl AsRef<Path>, keep: bool) -> Result<Self, PbcDfError> {
        let path = path.as_ref().to_path_buf();
        let f = hdf5::File::create(&path).map_err(|e| h5err("create", e))?;

        let nkpts = cderi.kpts.len();
        let flat: Vec<f64> = cderi.kpts.iter().flatten().copied().collect();
        f.new_dataset::<f64>()
            .shape([nkpts, 3])
            .create("kpts")
            .map_err(|e| h5err("create kpts", e))?
            .write_raw(&flat)
            .map_err(|e| h5err("write kpts", e))?;
        let aosym = match cderi.aosym {
            Aosym::S1 => 1_i32,
            Aosym::S2 => 2,
        };
        f.new_dataset::<i32>()
            .shape([1])
            .create("aosym")
            .map_err(|e| h5err("create aosym", e))?
            .write_raw(&[aosym])
            .map_err(|e| h5err("write aosym", e))?;

        let g = f.create_group("j3c").map_err(|e| h5err("group j3c", e))?;
        // Deterministic order — an HDF5 file whose contents depend on hash
        // iteration order is not reproducible.
        let mut keys: Vec<usize> = cderi.blocks.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            let b = &cderi.blocks[&k];
            let kg = g
                .create_group(&k.to_string())
                .map_err(|e| h5err("group pair", e))?;
            write_block(&kg, "0", &b.data, b.rank, b.nao_pair)?;
            if let Some(n) = &b.negative {
                write_block(&kg, "0-", n, n.re.len() / b.nao_pair, b.nao_pair)?;
            }
        }
        Ok(Self { path, keep })
    }

    /// Read a `_cderi` file back.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] on any HDF5 failure or a shape surprise.
    pub fn load(path: impl AsRef<Path>) -> Result<Cderi, PbcDfError> {
        let f = hdf5::File::open(path.as_ref()).map_err(|e| h5err("open", e))?;
        let kflat: Vec<f64> = f
            .dataset("kpts")
            .and_then(|d| d.read_raw::<f64>())
            .map_err(|e| h5err("read kpts", e))?;
        let kpts: Vec<[f64; 3]> = kflat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let a: Vec<i32> = f
            .dataset("aosym")
            .and_then(|d| d.read_raw::<i32>())
            .map_err(|e| h5err("read aosym", e))?;
        let aosym = if a.first() == Some(&1) {
            Aosym::S1
        } else {
            Aosym::S2
        };

        let g = f.group("j3c").map_err(|e| h5err("group j3c", e))?;
        let mut blocks = std::collections::HashMap::new();
        for name in g.member_names().map_err(|e| h5err("member_names", e))? {
            let key: usize = name
                .parse()
                .map_err(|e| h5err(&format!("k-pair key '{name}'"), e))?;
            let kg = g.group(&name).map_err(|e| h5err("open pair", e))?;
            let (data, rank, nao_pair) = read_block(&kg, "0")?;
            let negative = if kg.member_names().is_ok_and(|m| m.iter().any(|s| s == "0-")) {
                Some(read_block(&kg, "0-")?.0)
            } else {
                None
            };
            blocks.insert(
                key,
                CderiBlock {
                    data,
                    rank,
                    nao_pair,
                    negative,
                },
            );
        }
        Ok(Cderi {
            blocks,
            kpts,
            aosym,
        })
    }
}

impl Drop for CderiFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// One `(rank, ncol)` complex block, stored as `[..re.., ..im..]`.
fn write_block(
    g: &hdf5::Group,
    name: &str,
    t: &CTensor,
    rank: usize,
    ncol: usize,
) -> Result<(), PbcDfError> {
    let mut buf = Vec::with_capacity(t.re.len() * 2);
    buf.extend_from_slice(&t.re);
    buf.extend_from_slice(&t.im);
    g.new_dataset::<f64>()
        .shape([2, rank, ncol])
        .create(name)
        .map_err(|e| h5err("create block", e))?
        .write_raw(&buf)
        .map_err(|e| h5err("write block", e))?;
    Ok(())
}

fn read_block(g: &hdf5::Group, name: &str) -> Result<(CTensor, usize, usize), PbcDfError> {
    let d = g.dataset(name).map_err(|e| h5err("open block", e))?;
    let shape = d.shape();
    if shape.len() != 3 || shape[0] != 2 {
        return Err(h5err(
            "block shape",
            format!("expected [2, rank, ncol], got {shape:?}"),
        ));
    }
    let (rank, ncol) = (shape[1], shape[2]);
    let raw: Vec<f64> = d.read_raw::<f64>().map_err(|e| h5err("read block", e))?;
    let n = rank * ncol;
    if raw.len() != 2 * n {
        return Err(h5err(
            "block size",
            format!("expected {} values, got {}", 2 * n, raw.len()),
        ));
    }
    Ok((
        CTensor {
            re: raw[..n].to_vec(),
            im: raw[n..].to_vec(),
        },
        rank,
        ncol,
    ))
}
