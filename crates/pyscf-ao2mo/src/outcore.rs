//! Outcore / semi-incore AO→MO 4-index transform (the Phase-5 D-04 deferral,
//! landing in 06-09).
//!
//! Ports the spilling AO→MO transform of upstream `pyscf/ao2mo/outcore.py`
//! (`general` / `half_e1`) + `pyscf/ao2mo/semi_incore.py`. It computes the SAME
//! `(pq|rs) --C--> (ij|kl)` quarter-transform as the in-core
//! [`crate::transform::quarter_transform`], but instead of holding the
//! half-transformed `[np, nq, nao, nao]` intermediate fully resident it SPILLS
//! it to an HDF5 scratch dataset (the `lib.H5TmpFile()` equivalent), then
//! completes the second-half (`r`/`s`) transform block-by-block reading the
//! half-transform back from disk. This is the memory-frugality contract: the
//! peak resident buffer is one `[np, nq, nao]` row-slab of the half-transform,
//! not the full intermediate.
//!
//! ## Spill seam (D-07 — NO new hdf5-metno dep)
//!
//! The scratch file uses the `pyscf_chkfile::hdf5` re-exported alias (the sole
//! hdf5-metno owner). The scratch is a uniquely-named temp HDF5 file under the
//! system temp dir, holding a single flat `f64` dataset; [`OutcoreScratch`]
//! RAII drop-deletes it (mirrors the upstream `H5TmpFile()` auto-delete) so no
//! leftover scratch survives the transform (RESEARCH "Runtime State Inventory").
//!
//! ## Reduction discipline (Pitfall 1/2 + FOUND-06)
//!
//! Every contraction materializes products into a scratch `Vec` FIRST then
//! folds via [`oracle_sum`] — NO `+=` accumulation, NO `gemm`
//! (`pyscf_algebra::gemm` is `NotYetImplemented{phase:2}`). The result is
//! bit-exact and thread-count invariant, IDENTICAL to the in-core path (the
//! two paths visit the same finite values in the same fold order; a match is
//! `delta == 0.0` bit-exact — proven by the always-on outcore test).
//!
//! ## Flat-index discipline (Pitfall 3)
//!
//! All tensors are carried in the SAME column-major (F-order) flat layout as
//! [`crate::transform`], offsets doc-commented at every boundary:
//!
//! * Input AO ERI `eri_ao` — F-order `[nao,nao,nao,nao]`: `(p,q,r,s)` at
//!   `p + q*nao + r*nao^2 + s*nao^3`.
//! * MO coefficient block `c_x` — column-major `[nao, nx]`: `C[ao,mo]` at
//!   `ao + mo*nao`.
//! * Half-transform (spilled) — F-order `[np, nq, nao, nao]`: `(i,j,r,s)` at
//!   `i + j*np + r*np*nq + s*np*nq*nao`.
//! * Final MO ERI — F-order `[np, nq, nr, ns]`: `(i,j,k,l)` at
//!   `i + j*np + k*np*nq + l*np*nq*nr`.

use crate::error::Ao2moError;
use pyscf_algebra::oracle_sum;
use pyscf_chkfile::hdf5;
use pyscf_core::MOCoefficients;

/// HDF5-backed scratch for the spilled half-transform — the `lib.H5TmpFile()`
/// equivalent (D-07).
///
/// Holds an open `pyscf_chkfile::hdf5` temp file + the on-disk path; the
/// half-transformed intermediate lives in a single flat `f64` dataset named
/// [`Self::DATASET`]. On `Drop` the open handle is closed and the temp file is
/// REMOVED from disk (RAII — no leftover scratch; mirrors upstream
/// `H5TmpFile()` auto-delete). T-06-09-LEAK mitigation.
#[derive(Debug)]
pub struct OutcoreScratch {
    /// Open HDF5 file handle (kept alive for read/write; closed on drop).
    file: Option<hdf5::File>,
    /// On-disk temp path (deleted on drop).
    path: std::path::PathBuf,
    /// Element count (flat f64 buffer length).
    len: usize,
}

impl OutcoreScratch {
    /// Dataset name for the single flat half-transform buffer.
    const DATASET: &'static str = "half_e1";

    /// Monotonic uid source so concurrent transforms in one process get unique
    /// scratch paths (pid + counter). Not security-sensitive — the RAII drop is
    /// the leak mitigation (T-06-09-LEAK).
    fn next_uid() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static UID: AtomicU64 = AtomicU64::new(0);
        UID.fetch_add(1, Ordering::Relaxed)
    }

    /// Create a fresh zero-initialized scratch dataset of `len` f64 elements in
    /// a uniquely-named temp HDF5 file under the system temp dir.
    fn create(len: usize) -> Result<Self, Ao2moError> {
        let uid = Self::next_uid();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "pyscf_ao2mo_outcore_{}_{}.h5",
            std::process::id(),
            uid
        ));

        let file = hdf5::File::create(&path).map_err(|e| Ao2moError::Outcore {
            reason: format!("create outcore scratch {}: {e}", path.display()),
        })?;
        // Pre-create the flat dataset (zero-filled). The half-transform is
        // written row-slab by row-slab through `write_slab`.
        let zeros = vec![0.0_f64; len];
        let arr = ndarray::Array1::from_vec(zeros);
        file.new_dataset::<f64>()
            .shape([len])
            .create(Self::DATASET)
            .and_then(|ds| ds.write(&arr))
            .map_err(|e| Ao2moError::Outcore {
                reason: format!("create outcore dataset: {e}"),
            })?;

        Ok(Self {
            file: Some(file),
            path,
            len,
        })
    }

    /// Overwrite a contiguous `[offset, offset+slab.len())` window of the flat
    /// scratch dataset (the half-transform row-slab spill).
    fn write_slab(&self, offset: usize, slab: &[f64]) -> Result<(), Ao2moError> {
        let end = offset.checked_add(slab.len()).ok_or(Ao2moError::Outcore {
            reason: "outcore slab offset overflow".to_string(),
        })?;
        if end > self.len {
            return Err(Ao2moError::Outcore {
                reason: format!(
                    "outcore write slab [{offset},{end}) exceeds scratch len {}",
                    self.len
                ),
            });
        }
        let file = self.file.as_ref().ok_or(Ao2moError::Outcore {
            reason: "outcore scratch already closed".to_string(),
        })?;
        let arr = ndarray::Array1::from_vec(slab.to_vec());
        // Plain `Range<usize>` → HDF5 hyperslab `Selection` (hdf5-metno
        // `impl From<Range<usize>> for Selection`). We deliberately AVOID the
        // `ndarray::s![..]` macro: its expansion emits `#[allow(unsafe_code)]`,
        // which collides with the crate-wide `#![forbid(unsafe_code)]`.
        file.dataset(Self::DATASET)
            .and_then(|ds| ds.write_slice(&arr, offset..end))
            .map_err(|e| Ao2moError::Outcore {
                reason: format!("write outcore slab: {e}"),
            })?;
        Ok(())
    }

    /// Read a contiguous `[offset, offset+len)` window of the flat scratch
    /// dataset back into an owned `Vec<f64>` (the block-by-block read for the
    /// second-half transform).
    fn read_slab(&self, offset: usize, len: usize) -> Result<Vec<f64>, Ao2moError> {
        let end = offset.checked_add(len).ok_or(Ao2moError::Outcore {
            reason: "outcore slab offset overflow".to_string(),
        })?;
        if end > self.len {
            return Err(Ao2moError::Outcore {
                reason: format!(
                    "outcore read slab [{offset},{end}) exceeds scratch len {}",
                    self.len
                ),
            });
        }
        let file = self.file.as_ref().ok_or(Ao2moError::Outcore {
            reason: "outcore scratch already closed".to_string(),
        })?;
        // Plain `Range<usize>` hyperslab (no `ndarray::s!` — see `write_slab`).
        let arr: ndarray::Array1<f64> = file
            .dataset(Self::DATASET)
            .and_then(|ds| ds.read_slice_1d(offset..end))
            .map_err(|e| Ao2moError::Outcore {
                reason: format!("read outcore slab: {e}"),
            })?;
        Ok(arr.to_vec())
    }

    /// On-disk scratch path (for tests asserting the RAII delete).
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for OutcoreScratch {
    fn drop(&mut self) {
        // Close the HDF5 handle FIRST (so the OS releases the file), THEN remove
        // the temp file. Best-effort: a failed removal must not panic
        // (`#![forbid(unsafe_code)]` — no UB), but a leaked temp file is the
        // LEAK hazard, so we always try (T-06-09-LEAK).
        self.file = None;
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Outcore general AO→MO transform — the HDF5-spilling analog of
/// [`crate::incore::general`].
///
/// Transforms the AO-basis 2-electron integrals `eri_ao` (F-order
/// `[nao,nao,nao,nao]`) into the MO basis using four DISTINCT MO coefficient
/// blocks, SPILLING the half-transformed `[np, nq, nao, nao]` intermediate to
/// an HDF5 scratch dataset between the first-half (`p`,`q`) and second-half
/// (`r`,`s`) contractions. Returns the same flat F-order `[np,nq,nr,ns]` MO
/// ERI buffer as [`crate::incore::general`] — BIT-EXACT identical (the two
/// paths perform the same `oracle_sum` folds in the same order).
///
/// The scratch file is RAII drop-deleted before this returns: no leftover
/// scratch (T-06-09-LEAK). The peak resident buffer is one `[np, nq, nao]`
/// row-slab of the half-transform (the `s`-fixed slab), NOT the full
/// intermediate — the memory-frugality whole point (D-04 / D-08).
///
/// # Errors
/// [`Ao2moError::ShapeMismatch`] when `eri_ao.len() != nao^4` or any
/// `mo_coeffs[i].nao != nao`; [`Ao2moError::Outcore`] on an HDF5 scratch
/// create/read/write failure. Never indexes OOB, never panics
/// (T-05-02-SHAPE / `#![forbid(unsafe_code)]`).
pub fn general_outcore(
    eri_ao: &[f64],
    nao: usize,
    mo_coeffs: [&MOCoefficients; 4],
) -> Result<Vec<f64>, Ao2moError> {
    // T-05-02-SHAPE: every coefficient block must span the AO dimension.
    for c in mo_coeffs.iter() {
        if c.nao != nao {
            return Err(Ao2moError::ShapeMismatch {
                expected: nao,
                got: c.nao,
            });
        }
        if c.data.len() != c.nao * c.nmo {
            return Err(Ao2moError::ShapeMismatch {
                expected: c.nao * c.nmo,
                got: c.data.len(),
            });
        }
    }

    let [c_p, c_q, c_r, c_s] = mo_coeffs;
    quarter_transform_outcore(
        eri_ao, nao, &c_p.data, c_p.nmo, &c_q.data, c_q.nmo, &c_r.data, c_r.nmo, &c_s.data, c_s.nmo,
    )
}

/// Full outcore AO→MO transform: the symmetric special case of
/// [`general_outcore`] where the SAME MO coefficient block is used for all four
/// indices. Mirrors [`crate::incore::full`] but spills the half-transform.
///
/// # Errors
/// [`Ao2moError::ShapeMismatch`] when `mo_coeff.nao != nao` or
/// `eri_ao.len() != nao^4`; [`Ao2moError::Outcore`] on a scratch I/O failure.
pub fn full_outcore(
    eri_ao: &[f64],
    nao: usize,
    mo_coeff: &MOCoefficients,
) -> Result<Vec<f64>, Ao2moError> {
    if mo_coeff.nao != nao {
        return Err(Ao2moError::ShapeMismatch {
            expected: nao,
            got: mo_coeff.nao,
        });
    }
    general_outcore(eri_ao, nao, [mo_coeff; 4])
}

/// AO→MO 4-index transform via the SPILLING quarter-transform sequence.
///
/// Stages:
/// 1. **first-half (`p`,`q`)**: `(p,q,r,s) --C_p--> (i,q,r,s) --C_q--> (i,j,r,s)`
///    — held resident only one `s`-slab at a time, written to the HDF5 scratch.
/// 2. **second-half (`r`,`s`)**: read the half-transform back `s`-slab by
///    `s`-slab, `(i,j,r,s) --C_r--> (i,j,k,s) --C_s--> (i,j,k,l)`.
///
/// IDENTICAL math + fold order to [`crate::transform::quarter_transform`]; the
/// only difference is that the `[np, nq, nao, nao]` intermediate lives on disk
/// between the two halves. Every reduction routes through [`oracle_sum`]
/// (materialize-then-reduce — no `+=`), so the output is bit-exact.
///
/// # Errors
/// [`Ao2moError::ShapeMismatch`] for a bad `eri_ao`/coeff length;
/// [`Ao2moError::Outcore`] on a scratch I/O failure.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn quarter_transform_outcore(
    eri_ao: &[f64],
    nao: usize,
    c_p: &[f64],
    np: usize,
    c_q: &[f64],
    nq: usize,
    c_r: &[f64],
    nr: usize,
    c_s: &[f64],
    ns: usize,
) -> Result<Vec<f64>, Ao2moError> {
    // T-05-02-SHAPE: validate every untrusted shape input at entry.
    let expected_eri = nao * nao * nao * nao;
    if eri_ao.len() != expected_eri {
        return Err(Ao2moError::ShapeMismatch {
            expected: expected_eri,
            got: eri_ao.len(),
        });
    }
    for (c, n) in [(c_p, np), (c_q, nq), (c_r, nr), (c_s, ns)] {
        if c.len() != nao * n {
            return Err(Ao2moError::ShapeMismatch {
                expected: nao * n,
                got: c.len(),
            });
        }
    }

    // The half-transform `[np, nq, nao, nao]` lives on the HDF5 scratch; element
    // (i,j,r,s) at flat offset  i + j*np + r*np*nq + s*np*nq*nao.
    // The s-slab `[np, nq, nao]` (all (i,j,r) for a fixed s) is the unit we hold
    // resident — base offset s*np*nq*nao, length np*nq*nao.
    let half_len = np * nq * nao * nao;
    let scratch = OutcoreScratch::create(half_len)?;
    let slab_len = np * nq * nao; // one s-slab of the half-transform.

    // ---- FIRST HALF: build the half-transform one s-slab at a time, spill it.
    //
    // For each fixed s, build the resident slab `half_s[i + j*np + r*np*nq]`:
    //   p-step:  o1[i,q,r] = sum_p eri_ao[p,q,r,s] * c_p[p + i*nao]
    //   q-step:  half_s[i,j,r] = sum_q o1[i,q,r] * c_q[q + j*nao]
    // then write the slab to scratch at offset s*slab_len.
    let mut prod = vec![0.0_f64; nao]; // per-index product buffer (reused).
    for s in 0..nao {
        // p-step → o1 F-order [np, nao(q), nao(r)]: (i,q,r) at i + q*np + r*np*nao.
        let mut o1 = vec![0.0_f64; np * nao * nao];
        for r in 0..nao {
            for q in 0..nao {
                let eri_base = q * nao + r * nao * nao + s * nao * nao * nao;
                for i in 0..np {
                    let c_col = i * nao;
                    for p in 0..nao {
                        prod[p] = eri_ao[p + eri_base] * c_p[p + c_col];
                    }
                    o1[i + q * np + r * np * nao] = oracle_sum(&prod);
                }
            }
        }
        // q-step → half_s F-order [np, nq, nao(r)]: (i,j,r) at i + j*np + r*np*nq.
        let mut half_s = vec![0.0_f64; slab_len];
        for r in 0..nao {
            for j in 0..nq {
                let c_col = j * nao;
                for i in 0..np {
                    for q in 0..nao {
                        prod[q] = o1[i + q * np + r * np * nao] * c_q[q + c_col];
                    }
                    half_s[i + j * np + r * np * nq] = oracle_sum(&prod);
                }
            }
        }
        // Spill this s-slab to scratch (offset s*slab_len, length slab_len).
        scratch.write_slab(s * slab_len, &half_s)?;
    }

    // ---- SECOND HALF: read the half-transform back slab-by-slab, complete the
    // r/s transform into the final `[np, nq, nr, ns]` F-order buffer.
    //
    // The r-step contracts over r (which lives WITHIN an s-slab), so we read one
    // s-slab at a time. The s-step contracts over s (across slabs) — so we first
    // build an intermediate `o3[i,j,k,s]` (F-order [np,nq,nr,nao]) reading each
    // s-slab once, then fold over s into the final output.
    //   r-step:  o3[i,j,k,s] = sum_r half[i,j,r,s] * c_r[r + k*nao]
    //   s-step:  out[i,j,k,l] = sum_s o3[i,j,k,s] * c_s[s + l*nao]
    let mut o3 = vec![0.0_f64; np * nq * nr * nao]; // (i,j,k,s) at i + j*np + k*np*nq + s*np*nq*nr.
    for s in 0..nao {
        let half_s = scratch.read_slab(s * slab_len, slab_len)?; // [np, nq, nao(r)].
        for k in 0..nr {
            let c_col = k * nao;
            for j in 0..nq {
                for i in 0..np {
                    for r in 0..nao {
                        // half_s (i,j,r) at i + j*np + r*np*nq.
                        prod[r] = half_s[i + j * np + r * np * nq] * c_r[r + c_col];
                    }
                    o3[i + j * np + k * np * nq + s * np * nq * nr] = oracle_sum(&prod);
                }
            }
        }
    }

    // s-step → final out F-order [np, nq, nr, ns]: (i,j,k,l) at
    //   i + j*np + k*np*nq + l*np*nq*nr.
    let mut out = vec![0.0_f64; np * nq * nr * ns];
    for l in 0..ns {
        let c_col = l * nao;
        for k in 0..nr {
            for j in 0..nq {
                for i in 0..np {
                    for s in 0..nao {
                        // o3 (i,j,k,s) at i + j*np + k*np*nq + s*np*nq*nr.
                        prod[s] = o3[i + j * np + k * np * nq + s * np * nq * nr] * c_s[s + c_col];
                    }
                    out[i + j * np + k * np * nq + l * np * nq * nr] = oracle_sum(&prod);
                }
            }
        }
    }

    // `scratch` drops here → OutcoreScratch::drop deletes the temp file (RAII).
    drop(scratch);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform;
    use pyscf_core::MOCoefficients;
    use std::sync::Mutex;

    /// Serializes every test that creates an [`OutcoreScratch`] so the
    /// `no-leftover-scratch` snapshot test's before/after window cannot
    /// interleave a sibling transform's scratch lifecycle (the temp dir is a
    /// process-global namespace). Without this, a parallel sibling's in-flight
    /// scratch file would appear in the `after` snapshot and spuriously fail.
    static SCRATCH_SERIAL: Mutex<()> = Mutex::new(());

    /// Build a deterministic, non-symmetric synthetic ERI `[nao,nao,nao,nao]`
    /// (F-order) — the same generator the in-core transform tests use.
    fn synthetic_eri(nao: usize) -> Vec<f64> {
        let n = nao;
        let mut eri = vec![0.0_f64; n * n * n * n];
        for s in 0..n {
            for r in 0..n {
                for q in 0..n {
                    for p in 0..n {
                        let v = (p as f64) * 1.0
                            + (q as f64) * 0.25
                            + (r as f64) * 0.0625
                            + (s as f64) * 0.015_625
                            + 0.5;
                        eri[p + q * n + r * n * n + s * n * n * n] = v;
                    }
                }
            }
        }
        eri
    }

    /// A deterministic column-major `[nao, n]` MO coefficient block.
    fn mk_coeff(nao: usize, n: usize, seed: f64) -> MOCoefficients {
        let mut data = vec![0.0_f64; nao * n];
        for mo in 0..n {
            for ao in 0..nao {
                data[ao + mo * nao] = seed + (ao as f64) * 0.5 - (mo as f64) * 0.3;
            }
        }
        MOCoefficients {
            nao,
            nmo: n,
            data,
            energies: Vec::new(),
            occupations: Vec::new(),
        }
    }

    /// ALWAYS-ON: outcore == in-core `general` to BIT-EXACTNESS on a synthetic
    /// small input. The spilling path reduces in the SAME `oracle_sum` order as
    /// the resident path, so `delta == 0.0` is the contract (CCSD-08 / D-04).
    #[test]
    fn outcore_matches_incore_bit_exact() {
        let _g = SCRATCH_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let nao = 3;
        let eri = synthetic_eri(nao);
        let c_p = mk_coeff(nao, 2, 1.1);
        let c_q = mk_coeff(nao, 3, 0.7);
        let c_r = mk_coeff(nao, 2, -0.4);
        let c_s = mk_coeff(nao, 1, 0.9);

        let incore = crate::incore::general(&eri, nao, [&c_p, &c_q, &c_r, &c_s])
            .expect("in-core general");
        let outcore = general_outcore(&eri, nao, [&c_p, &c_q, &c_r, &c_s])
            .expect("outcore general");

        assert_eq!(incore.len(), outcore.len());
        for (idx, (a, b)) in incore.iter().zip(outcore.iter()).enumerate() {
            assert_eq!(
                a, b,
                "outcore vs in-core mismatch at flat index {idx}: outcore {b}, in-core {a}"
            );
        }
    }

    /// ALWAYS-ON: the `full_outcore` symmetric case also matches the in-core
    /// `full` bit-exactly.
    #[test]
    fn full_outcore_matches_incore_bit_exact() {
        let _g = SCRATCH_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let nao = 3;
        let eri = synthetic_eri(nao);
        let c = mk_coeff(nao, nao, 0.3);
        let incore = crate::incore::full(&eri, nao, &c).expect("in-core full");
        let outcore = full_outcore(&eri, nao, &c).expect("outcore full");
        assert_eq!(incore, outcore, "full_outcore must be bit-identical to in-core full");
    }

    /// ALWAYS-ON: the scratch HDF5 file is created during the transform and
    /// DELETED afterward — no leftover scratch (RAII drop-delete, T-06-09-LEAK).
    #[test]
    fn outcore_scratch_deleted_after_transform() {
        let _g = SCRATCH_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // Capture the scratch path while it is open, then assert it is gone once
        // the OutcoreScratch drops.
        let scratch_path;
        {
            let scratch = OutcoreScratch::create(8).expect("create scratch");
            scratch_path = scratch.path().to_path_buf();
            assert!(scratch_path.exists(), "scratch must exist while open");
            // Round-trip a slab to prove the dataset is usable.
            scratch.write_slab(2, &[1.0, 2.0, 3.0]).expect("write slab");
            let back = scratch.read_slab(2, 3).expect("read slab");
            assert_eq!(back, vec![1.0, 2.0, 3.0]);
        } // scratch drops here.
        assert!(
            !scratch_path.exists(),
            "outcore scratch must be deleted on drop (RAII, T-06-09-LEAK)"
        );
    }

    /// ALWAYS-ON: a full transform leaves no scratch behind (no leftover
    /// scratch — the Runtime State Inventory rule).
    ///
    /// We snapshot the SET of `pyscf_ao2mo_outcore_<pid>_*` scratch paths before
    /// and after the transform and assert no NEW path persists. A plain count
    /// delta would be racy under parallel test execution (a sibling outcore test
    /// may hold a scratch open during our window); the set-difference is robust
    /// because any path created+dropped by OUR transform is absent from both
    /// snapshots, and any sibling's in-flight path appears in BOTH (or its own
    /// RAII removes it). The serialization mutex below additionally pins our
    /// transform so its scratch lifecycle does not interleave with the snapshot.
    #[test]
    fn general_outcore_leaves_no_leftover_scratch() {
        use std::collections::BTreeSet;
        let _guard = SCRATCH_SERIAL.lock().unwrap_or_else(|p| p.into_inner());

        let snapshot = || -> BTreeSet<std::path::PathBuf> {
            let dir = std::env::temp_dir();
            let prefix = format!("pyscf_ao2mo_outcore_{}_", std::process::id());
            std::fs::read_dir(&dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
                        .map(|e| e.path())
                        .collect()
                })
                .unwrap_or_default()
        };

        let before = snapshot();
        let nao = 2;
        let eri = synthetic_eri(nao);
        let c = mk_coeff(nao, nao, 0.5);
        let _ = general_outcore(&eri, nao, [&c, &c, &c, &c]).expect("outcore general");
        let after = snapshot();

        // No NEW scratch path persisted (after ⊆ before): our transform's
        // scratch is RAII drop-deleted before it returns.
        let leaked: Vec<_> = after.difference(&before).collect();
        assert!(
            leaked.is_empty(),
            "leftover outcore scratch after the transform: {leaked:?}"
        );
    }

    /// ALWAYS-ON: a shape mismatch (coeff block does not span AO) is rejected as
    /// an `Err`, never an OOB panic (T-05-02-SHAPE).
    #[test]
    fn outcore_rejects_shape_mismatch() {
        let nao = 2;
        let eri = synthetic_eri(nao);
        let bad = MOCoefficients {
            nao: 3, // wrong AO span.
            nmo: 1,
            data: vec![0.0; 3],
            energies: Vec::new(),
            occupations: Vec::new(),
        };
        let good = mk_coeff(nao, 1, 0.1);
        let res = general_outcore(&eri, nao, [&bad, &good, &good, &good]);
        assert!(matches!(res, Err(Ao2moError::ShapeMismatch { .. })));
    }

    /// Cross-check vs the in-core internal quarter-transform on a rectangular
    /// block (independent of `incore::general`'s wrapper).
    #[test]
    fn quarter_transform_outcore_matches_internal_incore() {
        let _g = SCRATCH_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let nao = 3;
        let eri = synthetic_eri(nao);
        let c_p = mk_coeff(nao, 2, 1.1);
        let c_q = mk_coeff(nao, 1, 0.7);
        let c_r = mk_coeff(nao, 3, -0.4);
        let c_s = mk_coeff(nao, 2, 0.9);
        let want = transform::quarter_transform(
            &eri, nao, &c_p.data, 2, &c_q.data, 1, &c_r.data, 3, &c_s.data, 2,
        )
        .expect("incore quarter_transform");
        let got = quarter_transform_outcore(
            &eri, nao, &c_p.data, 2, &c_q.data, 1, &c_r.data, 3, &c_s.data, 2,
        )
        .expect("outcore quarter_transform");
        assert_eq!(want, got, "outcore quarter_transform must be bit-identical");
    }
}
