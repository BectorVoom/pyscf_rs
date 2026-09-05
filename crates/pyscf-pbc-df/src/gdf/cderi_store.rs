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
/// `df.py:338-400`, through `_load3c.getitem` (`:807-836`) and
/// `_KPair3CLoader.__getitem__` (`:990-1009`).
///
/// `compact = true` returns the `s2`-packed columns, `false` the full `nao²`
/// square.
///
/// # An `s2` store holds HALF of each off-diagonal k-pair
///
/// `(L | mu^{ki} nu^{kj})` is Hermitian in `(mu, nu)` only when `ki == kj`.
/// Away from the diagonal the packed store keeps the lower triangle `mu >= nu`
/// of the pair `(ki, kj)` and the lower triangle of `(kj, ki)`, and the two
/// halves are joined by
///
/// ```text
/// L(ki,kj)[nu, mu] = conj(L(kj,ki)[mu, nu])          mu > nu
/// ```
///
/// — `PBCunpack_tril_triu` (`pyscf/lib/pbc/fill_ints.c:1460-1483`), which
/// upstream calls with `tril` from the `(ki, kj)` block and `triu` from the
/// `(kj, ki)` one. So unpacking to `s1` needs BOTH blocks; `lib.ANTIHERMI` on
/// one block is the `ki == kj` special case of the same formula, not the
/// general rule.
///
/// A `compact` request is the stored `(ki, kj)` block verbatim, whatever the
/// k-pair: upstream reaches the same place by unpacking and then re-packing
/// the lower triangle (`df.py:359-375`).
///
/// # Errors
/// [`PbcDfError::Core`] when the pair was never computed, and — for an `s1`
/// request off the k-diagonal — when its conjugate pair was not computed
/// either, since the upper triangle is unreachable without it.
pub fn sr_loop(
    cderi: &Cderi,
    ki: usize,
    kj: usize,
    nao: usize,
    compact: bool,
) -> Result<Vec<SrBlock>, PbcDfError> {
    let b = cderi.get(ki, kj).ok_or_else(|| missing_pair(ki, kj))?;
    // `(k, k)` is Hermitian in `(mu, nu)`, so it is its own conjugate pair.
    let conj_pair = if ki == kj { Some(b) } else { cderi.get(kj, ki) };
    let mut out = vec![reshape_block(b, conj_pair, cderi.aosym, nao, compact, ki, kj)?];
    if let Some(neg) = &b.negative {
        let nb = negative_block(neg, b.nao_pair);
        let nc = if ki == kj {
            Some(nb.clone())
        } else {
            conj_pair
                .and_then(|u| u.negative.as_ref().map(|n| negative_block(n, u.nao_pair)))
        };
        let mut m = reshape_block(&nb, nc.as_ref(), cderi.aosym, nao, compact, ki, kj)?;
        m.sign = -1;
        out.push(m);
    }
    Ok(out)
}

fn missing_pair(ki: usize, kj: usize) -> PbcDfError {
    PbcDfError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(format!(
            "sr_loop: no cderi block for the k-pair ({ki}, {kj}); build() with \
             j_only = false to get every pair"
        )),
    ))
}

/// The `2-D` negative branch as a standalone block, so it reshapes by the same
/// code path as the positive one.
fn negative_block(neg: &CTensor, nao_pair: usize) -> CderiBlock {
    CderiBlock {
        data: neg.clone(),
        rank: neg.re.len() / nao_pair,
        nao_pair,
        negative: None,
    }
}

/// `pack_tril` / `PBCunpack_tril_triu` on one block.
///
/// `conj_pair` is the `(kj, ki)` block, and it is consulted only by the
/// `s2 -> s1` unpack, where it supplies the upper triangle. See [`sr_loop`].
fn reshape_block(
    b: &CderiBlock,
    conj_pair: Option<&CderiBlock>,
    stored: Aosym,
    nao: usize,
    compact: bool,
    ki: usize,
    kj: usize,
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
    match (stored, want) {
        (Aosym::S1, Aosym::S2) => {
            for l in 0..b.rank {
                for mu in 0..nao {
                    for nu in 0..=mu {
                        let tri = mu * (mu + 1) / 2 + nu;
                        let sq = mu * nao + nu;
                        re[l * ncol + tri] = b.data.re[l * b.nao_pair + sq];
                        im[l * ncol + tri] = b.data.im[l * b.nao_pair + sq];
                    }
                }
            }
        }
        (Aosym::S2, Aosym::S1) => {
            let u = conj_pair.ok_or_else(|| {
                PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                    pyscf_core::CoreError::InvalidMolecule(format!(
                        "sr_loop: the s2 block for ({ki}, {kj}) carries only mu >= nu; \
                         unpacking it to the full square needs the conjugate pair \
                         ({kj}, {ki}), which was never computed"
                    )),
                ))
            })?;
            if u.rank != b.rank || u.nao_pair != b.nao_pair {
                return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                    pyscf_core::CoreError::InvalidMolecule(format!(
                        "sr_loop: the k-pair ({ki}, {kj}) has rank {}/{} columns and its \
                         conjugate ({kj}, {ki}) has {}/{}; the two halves of one square \
                         cannot come from different decompositions",
                        b.rank, b.nao_pair, u.rank, u.nao_pair
                    )),
                )));
            }
            for l in 0..b.rank {
                for mu in 0..nao {
                    for nu in 0..nao {
                        let sq = l * ncol + mu * nao + nu;
                        if mu >= nu {
                            // `pout[i*nao+j] = ptril[ij]`.
                            let tri = l * b.nao_pair + mu * (mu + 1) / 2 + nu;
                            re[sq] = b.data.re[tri];
                            im[sq] = b.data.im[tri];
                        } else {
                            // `pout[j*nao+i] = conj(ptriu[ij])`, `ptriu` being
                            // the CONJUGATE pair's lower triangle. Reduces to
                            // `lib.ANTIHERMI` when `u` is `b`, i.e. at `ki == kj`.
                            let tri = l * u.nao_pair + nu * (nu + 1) / 2 + mu;
                            re[sq] = u.data.re[tri];
                            im[sq] = -u.data.im[tri];
                        }
                    }
                }
            }
        }
        _ => {}
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
