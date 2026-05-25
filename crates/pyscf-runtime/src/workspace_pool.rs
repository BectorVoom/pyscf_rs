//! WorkspacePool — the CCSD-11 / D-08 tensor-arena (the defining decision of
//! Phase 6).
//!
//! Phase 1 shipped a struct with three fields and three methods so the Phase-6
//! tensor-arena could land WITHOUT restructuring the public surface. Phase 6
//! fills the body: a real reuse-pool (`reserve`/`release` with free-list reuse)
//! behind opaque [`BufferId`] handles, a [`TensorBackend`] enum
//! (`InMemory` | `Spilled`), and the HARD `PYSCF_MAX_MEMORY` pre-flight refusal
//! (D-01 — NO silent auto-downgrade).
//!
//! The PUBLIC SURFACE of `budget_bytes` / `new` / `from_env` / `try_reserve` is
//! UNCHANGED (the Phase-1 doc-comment mandates not restructuring it); the body
//! gains `reserve` / `release` / `as_slice` / `as_mut_slice` and the backend
//! types.
//!
//! ## Handle-unification decision (RESEARCH Open Q2 / A2 — resolved here)
//!
//! `pyscf_algebra::Tensor<T>` also carries a `BufferId`, but that `BufferId`'s
//! inner field is `pub(crate)` TO `pyscf-algebra` and `pyscf-runtime` must NOT
//! depend on `pyscf-algebra` (wrong dependency direction — algebra depends on
//! runtime). So the pool OWNS its backing storage keyed by a `pyscf-runtime`
//! [`BufferId`] and hands callers that lightweight handle. The 06-03+
//! contraction call sites materialize products INTO the reserved buffer (via
//! [`WorkspacePool::as_mut_slice`]) and then reduce through
//! `pyscf-algebra::oracle_sum` on the materialized buffer — so a runtime-owned
//! handle is sufficient and keeps the dependency direction clean.
//!
//! ## Spill backend (D-07)
//!
//! [`SpillHandle`] wraps a `pyscf_chkfile::hdf5` temp file + dataset (the
//! `lib.H5TmpFile()`-equivalent) and DELETES its file in `Drop` (RAII — mirror
//! the upstream auto-delete; RESEARCH "Runtime State Inventory" requires no
//! leftover scratch). The spill goes through `pyscf-chkfile` (the sole
//! hdf5-metno owner — D-07), so no new hdf5 dep enters the graph.

use std::sync::Mutex;

use crate::error::BackendError;
use pyscf_chkfile::hdf5;

/// Opaque pool buffer handle. The pool OWNS the backing storage; callers hold
/// only this id (RESEARCH A2 — single runtime-owned handle).
///
/// Distinct from `pyscf_algebra::BufferId` (whose inner field is private to
/// algebra). Cheap `Copy` so the kernel can pass it around freely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub(crate) u64);

impl BufferId {
    /// Raw id (for diagnostics / tests).
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Storage backend behind an opaque buffer handle (D-01/D-08).
///
/// `InMemory` is a resident `Box<[f64]>`; `Spilled` is an HDF5-backed temp file
/// (via [`SpillHandle`]) chosen at allocation time when an in-core buffer would
/// exceed the budget AND the caller opted into spilling (`allow_spill==true`,
/// the DF/AO-direct path). The choice is transparent to the contraction code —
/// access always goes through [`WorkspacePool::as_slice`] /
/// [`WorkspacePool::as_mut_slice`].
#[derive(Debug)]
pub enum TensorBackend {
    /// Resident in-memory buffer.
    InMemory(Box<[f64]>),
    /// HDF5-spilled buffer (RAII drop-deletes the temp file).
    Spilled(SpillHandle),
}

impl TensorBackend {
    /// Element capacity of this backend's buffer.
    fn len(&self) -> usize {
        match self {
            TensorBackend::InMemory(b) => b.len(),
            TensorBackend::Spilled(h) => h.len,
        }
    }
}

/// HDF5-backed spill buffer — the `lib.H5TmpFile()` equivalent (D-07).
///
/// Holds an open `pyscf_chkfile::hdf5` temp file and the on-disk path; the data
/// lives in a single flat `f64` dataset named `"buf"`. On `Drop` the open file
/// handle is closed and the temp file is REMOVED from disk (RAII — no leftover
/// scratch; mirrors upstream `H5TmpFile()` auto-delete). T-06-02-LEAK
/// mitigation.
#[derive(Debug)]
pub struct SpillHandle {
    /// Open HDF5 file handle (kept alive for read/write; closed on drop).
    file: Option<hdf5::File>,
    /// On-disk temp path (deleted on drop).
    path: std::path::PathBuf,
    /// Element count (flat f64 buffer length).
    len: usize,
}

impl SpillHandle {
    /// Dataset name for the single flat spill buffer.
    const DATASET: &'static str = "buf";

    /// Create a fresh zero-initialized spill dataset of `len` f64 elements in a
    /// uniquely-named temp HDF5 file under the system temp dir.
    fn create(len: usize, uid: u64) -> Result<Self, BackendError> {
        let mut path = std::env::temp_dir();
        // Unique name: pid + pool-assigned uid avoids collisions across
        // processes/threads. Not security-sensitive content; the RAII drop is
        // the leak mitigation (T-06-02-LEAK).
        path.push(format!(
            "pyscf_ccsd_spill_{}_{}.h5",
            std::process::id(),
            uid
        ));

        let file = hdf5::File::create(&path).map_err(|e| BackendError::ProbeFailed {
            backend: "hdf5-spill",
            reason: format!("create temp spill file {}: {e}", path.display()),
        })?;
        let zeros = vec![0.0_f64; len];
        let arr = ndarray::Array1::from_vec(zeros);
        file.new_dataset::<f64>()
            .shape([len])
            .create(Self::DATASET)
            .and_then(|ds| ds.write(&arr))
            .map_err(|e| BackendError::ProbeFailed {
                backend: "hdf5-spill",
                reason: format!("create spill dataset: {e}"),
            })?;

        Ok(Self {
            file: Some(file),
            path,
            len,
        })
    }

    /// Read the full spill buffer into an owned `Vec<f64>`.
    fn read(&self) -> Result<Vec<f64>, BackendError> {
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| BackendError::ProbeFailed {
                backend: "hdf5-spill",
                reason: "spill file already closed".to_string(),
            })?;
        let arr: ndarray::Array1<f64> = file
            .dataset(Self::DATASET)
            .and_then(|ds| ds.read_1d())
            .map_err(|e| BackendError::ProbeFailed {
                backend: "hdf5-spill",
                reason: format!("read spill dataset: {e}"),
            })?;
        Ok(arr.to_vec())
    }

    /// Overwrite the full spill buffer from a flat slice.
    fn write(&self, data: &[f64]) -> Result<(), BackendError> {
        if data.len() != self.len {
            return Err(BackendError::ProbeFailed {
                backend: "hdf5-spill",
                reason: format!(
                    "spill write length mismatch: got {}, expected {}",
                    data.len(),
                    self.len
                ),
            });
        }
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| BackendError::ProbeFailed {
                backend: "hdf5-spill",
                reason: "spill file already closed".to_string(),
            })?;
        let arr = ndarray::Array1::from_vec(data.to_vec());
        file.dataset(Self::DATASET)
            .and_then(|ds| ds.write(&arr))
            .map_err(|e| BackendError::ProbeFailed {
                backend: "hdf5-spill",
                reason: format!("write spill dataset: {e}"),
            })?;
        Ok(())
    }

    /// On-disk path (for tests asserting the RAII delete).
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for SpillHandle {
    fn drop(&mut self) {
        // Close the HDF5 handle FIRST (so the OS releases the file), then
        // remove the temp file. Best-effort: a failed removal must not panic
        // (`#![forbid(unsafe_code)]` workspace-wide — no UB), but a leaked
        // temp file is the LEAK hazard so we always try.
        self.file = None;
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A single pooled allocation: its handle, its backend storage, and whether it
/// is currently checked out (`in_use`) or available for reuse on the free-list.
#[derive(Debug)]
pub(crate) struct PooledAllocation {
    /// Stable handle id.
    pub id: BufferId,
    /// Backing storage (in-memory or spilled).
    pub backend: TensorBackend,
    /// `true` while a caller holds the handle; `false` after `release` (back on
    /// the free-list, NOT dropped — this is what makes the heap-alloc count
    /// bounded across iterations).
    pub in_use: bool,
}

/// The CCSD-11 tensor-arena: a budget-checked reuse pool of f64 buffers behind
/// opaque [`BufferId`] handles.
#[derive(Debug, Default)]
pub struct WorkspacePool {
    /// PYSCF_MAX_MEMORY ceiling in bytes. Default 4 GB.
    pub budget_bytes: usize,
    /// Free-list of buffer allocations (reserve scans for a reusable released
    /// allocation before allocating fresh).
    pub(crate) pool: Mutex<Vec<PooledAllocation>>,
    /// Monotonic id counter for fresh allocations.
    pub(crate) next_id: Mutex<u64>,
    /// Sum of bytes of currently-IN-MEMORY allocations (in_use OR free-listed,
    /// since a free-listed in-memory buffer still holds its `Box<[f64]>`). Used
    /// for the live-budget check on fresh allocations. Spilled buffers do NOT
    /// count against the in-memory budget.
    pub(crate) live_inmem_bytes: Mutex<usize>,
}

impl WorkspacePool {
    /// 4 GiB default (PERF reasonable upper bound for v1 single-node).
    pub const DEFAULT_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;

    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            pool: Mutex::new(Vec::new()),
            next_id: Mutex::new(0),
            live_inmem_bytes: Mutex::new(0),
        }
    }

    /// Read PYSCF_MAX_MEMORY (interpreted as MEGABYTES per upstream PySCF
    /// convention). Default 4 GB if unset or unparseable.
    pub fn from_env() -> Self {
        let budget = std::env::var("PYSCF_MAX_MEMORY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(Self::DEFAULT_BUDGET_BYTES);
        Self::new(budget)
    }

    /// HARD pre-flight refusal (D-01). Returns `Err(MemoryLimitExceeded)` if a
    /// single in-core reservation of `bytes` would exceed the total budget.
    ///
    /// This is the budget ceiling check the Phase-1 skeleton shipped; the body
    /// is unchanged so existing callers keep their contract. `reserve` calls it
    /// before any fresh in-memory allocation; there is NO silent auto-downgrade
    /// — over-budget in-core simply refuses (the user opts into DF/AO-direct).
    pub fn try_reserve(&self, bytes: usize) -> Result<(), BackendError> {
        if bytes > self.budget_bytes {
            Err(BackendError::MemoryLimitExceeded {
                requested: bytes,
                limit: self.budget_bytes,
            })
        } else {
            Ok(())
        }
    }

    /// Bytes a buffer of `shape` requires (`product * 8`).
    fn shape_bytes(shape: &[usize]) -> usize {
        shape.iter().product::<usize>().saturating_mul(8)
    }

    /// Reserve a buffer of `shape` (f64 elements), reusing a free-listed buffer
    /// of sufficient size when one exists — otherwise allocating fresh.
    ///
    /// Free-list reuse is what makes the allocate-once-reuse / heap-alloc-count
    /// guarantee hold (CCSD-11, Pitfall 20): a buffer `reserve`d once before the
    /// iteration loop and `release`d after is handed BACK on the next `reserve`
    /// of a fitting shape, NOT freshly heap-allocated.
    ///
    /// When no free buffer fits and a fresh in-memory allocation would exceed
    /// the budget:
    ///   * `allow_spill == false` → `Err(MemoryLimitExceeded)` (D-01 HARD
    ///     refusal — no downgrade). The DoS mitigation (T-06-02-OOM).
    ///   * `allow_spill == true`  → allocate a `Spilled` (HDF5-backed) buffer
    ///     (the DF/AO-direct path).
    ///
    /// # Errors
    /// [`BackendError::MemoryLimitExceeded`] when over budget and spill is not
    /// permitted; [`BackendError::ProbeFailed`] if an HDF5 spill file cannot be
    /// created. Never panics.
    pub fn reserve(&self, shape: &[usize], allow_spill: bool) -> Result<BufferId, BackendError> {
        let need_elems = shape.iter().product::<usize>();
        let need_bytes = Self::shape_bytes(shape);

        // 1. Free-list scan: reuse the smallest released allocation that fits.
        {
            let mut pool = self
                .pool
                .lock()
                .map_err(|_| Self::poisoned("workspace pool"))?;
            let mut best: Option<usize> = None;
            for (idx, alloc) in pool.iter().enumerate() {
                if alloc.in_use {
                    continue;
                }
                if alloc.backend.len() >= need_elems {
                    let better = match best {
                        None => true,
                        Some(b) => alloc.backend.len() < pool[b].backend.len(),
                    };
                    if better {
                        best = Some(idx);
                    }
                }
            }
            if let Some(idx) = best {
                pool[idx].in_use = true;
                return Ok(pool[idx].id);
            }
        }

        // 2. No reusable buffer. Decide in-memory vs spill vs refuse.
        //    Live in-memory budget: does THIS fresh in-memory buffer (on top of
        //    already-resident buffers) still fit the ceiling? The single-buffer
        //    ceiling (`need_bytes <= budget_bytes`) is subsumed by this check
        //    when no other buffers are resident.
        let fits_inmem = {
            let live = self
                .live_inmem_bytes
                .lock()
                .map_err(|_| Self::poisoned("live-bytes"))?;
            live.saturating_add(need_bytes) <= self.budget_bytes
        };

        // Over budget AND spill not permitted: HARD refuse (D-01) BEFORE
        // consuming a buffer id or allocating anything. NO silent downgrade.
        if !fits_inmem && !allow_spill {
            return Err(BackendError::MemoryLimitExceeded {
                requested: need_bytes,
                limit: self.budget_bytes,
            });
        }

        let id = self.next_buffer_id()?;

        if fits_inmem {
            let backend = TensorBackend::InMemory(vec![0.0_f64; need_elems].into_boxed_slice());
            {
                let mut live = self
                    .live_inmem_bytes
                    .lock()
                    .map_err(|_| Self::poisoned("live-bytes"))?;
                *live = live.saturating_add(need_bytes);
            }
            self.push_alloc(id, backend)?;
            Ok(id)
        } else {
            // `allow_spill == true` and over the in-memory budget: spill.
            let backend = TensorBackend::Spilled(SpillHandle::create(need_elems, id.0)?);
            self.push_alloc(id, backend)?;
            Ok(id)
        }
    }

    /// Release a buffer back to the free-list for reuse. Does NOT drop the
    /// backing `Box<[f64]>` / spill file — the next `reserve` of a fitting shape
    /// reuses it (the allocate-once-reuse guarantee).
    ///
    /// A no-op for an unknown or already-released id.
    pub fn release(&self, id: BufferId) {
        if let Ok(mut pool) = self.pool.lock()
            && let Some(alloc) = pool.iter_mut().find(|a| a.id == id)
        {
            alloc.in_use = false;
        }
    }

    /// Read the full contents of a reserved buffer into an owned `Vec<f64>`.
    ///
    /// For an `InMemory` backend this copies the resident slice; for a `Spilled`
    /// backend it reads the HDF5 dataset. Reductions/contractions over a
    /// reserved buffer (a 06-03+ concern) route the materialized data through
    /// `pyscf-algebra::oracle_sum`.
    ///
    /// # Errors
    /// [`BackendError::ProbeFailed`] for an unknown id or a spill read failure.
    pub fn as_slice(&self, id: &BufferId) -> Result<Vec<f64>, BackendError> {
        let pool = self
            .pool
            .lock()
            .map_err(|_| Self::poisoned("workspace pool"))?;
        let alloc = pool
            .iter()
            .find(|a| a.id == *id)
            .ok_or_else(|| Self::unknown_id(id))?;
        match &alloc.backend {
            TensorBackend::InMemory(b) => Ok(b.to_vec()),
            TensorBackend::Spilled(h) => h.read(),
        }
    }

    /// Write the full contents of a reserved buffer from a flat slice.
    ///
    /// The kernel materializes a product INTO the reserved buffer via this
    /// write path (the buffer is the working store); for an `InMemory` backend
    /// the resident slice is overwritten in place, for a `Spilled` backend the
    /// HDF5 dataset is rewritten. Length must match the reserved capacity.
    ///
    /// # Errors
    /// [`BackendError::ProbeFailed`] for an unknown id, a length mismatch, or a
    /// spill write failure.
    pub fn write_slice(&self, id: &BufferId, data: &[f64]) -> Result<(), BackendError> {
        let mut pool = self
            .pool
            .lock()
            .map_err(|_| Self::poisoned("workspace pool"))?;
        let alloc = pool
            .iter_mut()
            .find(|a| a.id == *id)
            .ok_or_else(|| Self::unknown_id(id))?;
        match &mut alloc.backend {
            TensorBackend::InMemory(b) => {
                if data.len() > b.len() {
                    return Err(BackendError::ProbeFailed {
                        backend: "workspace-pool",
                        reason: format!(
                            "write length {} exceeds buffer capacity {}",
                            data.len(),
                            b.len()
                        ),
                    });
                }
                b[..data.len()].copy_from_slice(data);
                Ok(())
            }
            TensorBackend::Spilled(h) => h.write(data),
        }
    }

    /// Mutable in-memory view of a reserved buffer (the working store the
    /// 06-03+ kernel materializes products into).
    ///
    /// Returns the resident `Box<[f64]>` slice for an `InMemory` backend. A
    /// `Spilled` backend has no resident slice, so this returns
    /// [`BackendError::ProbeFailed`] — spilled buffers are accessed through
    /// [`WorkspacePool::as_slice`] / [`WorkspacePool::write_slice`]. The closure
    /// runs while the pool lock is held.
    ///
    /// # Errors
    /// [`BackendError::ProbeFailed`] for an unknown id or a spilled backend.
    pub fn with_mut_slice<R>(
        &self,
        id: &BufferId,
        f: impl FnOnce(&mut [f64]) -> R,
    ) -> Result<R, BackendError> {
        let mut pool = self
            .pool
            .lock()
            .map_err(|_| Self::poisoned("workspace pool"))?;
        let alloc = pool
            .iter_mut()
            .find(|a| a.id == *id)
            .ok_or_else(|| Self::unknown_id(id))?;
        match &mut alloc.backend {
            TensorBackend::InMemory(b) => Ok(f(b)),
            TensorBackend::Spilled(_) => Err(BackendError::ProbeFailed {
                backend: "workspace-pool",
                reason: "with_mut_slice on a spilled buffer; use write_slice/as_slice".to_string(),
            }),
        }
    }

    /// Number of allocations currently on the pool (in_use + free-listed).
    /// Diagnostics / tests.
    pub fn allocation_count(&self) -> usize {
        self.pool.lock().map(|p| p.len()).unwrap_or(0)
    }

    /// Allocate the next monotonic buffer id.
    fn next_buffer_id(&self) -> Result<BufferId, BackendError> {
        let mut n = self
            .next_id
            .lock()
            .map_err(|_| Self::poisoned("id counter"))?;
        let id = BufferId(*n);
        *n = n.saturating_add(1);
        Ok(id)
    }

    /// Push a fresh allocation onto the pool (marked in_use).
    fn push_alloc(&self, id: BufferId, backend: TensorBackend) -> Result<(), BackendError> {
        let mut pool = self
            .pool
            .lock()
            .map_err(|_| Self::poisoned("workspace pool"))?;
        pool.push(PooledAllocation {
            id,
            backend,
            in_use: true,
        });
        Ok(())
    }

    fn poisoned(what: &'static str) -> BackendError {
        BackendError::ProbeFailed {
            backend: "workspace-pool",
            reason: format!("{what} mutex poisoned"),
        }
    }

    fn unknown_id(id: &BufferId) -> BackendError {
        BackendError::ProbeFailed {
            backend: "workspace-pool",
            reason: format!("unknown buffer id {}", id.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// reserve → release → reserve(same shape) hands back the SAME BufferId
    /// (free-list reuse — the allocate-once-reuse guarantee, CCSD-11).
    #[test]
    fn release_then_reserve_reuses_same_buffer() {
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let shape = [8usize, 8, 4, 4]; // a small Wabef-like 4-tensor.

        let id1 = pool.reserve(&shape, false).expect("first reserve fits");
        pool.release(id1);
        let id2 = pool.reserve(&shape, false).expect("second reserve reuses");

        assert_eq!(id1, id2, "released buffer must be reused, not reallocated");
        // Only ONE allocation ever made.
        assert_eq!(pool.allocation_count(), 1);
    }

    /// A larger free buffer is reused for a smaller request (the "fits" rule).
    #[test]
    fn reserve_reuses_larger_free_buffer() {
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let big = pool.reserve(&[100], false).unwrap();
        pool.release(big);
        let small = pool.reserve(&[10], false).unwrap();
        assert_eq!(big, small, "smaller request reuses the larger free buffer");
        assert_eq!(pool.allocation_count(), 1);
    }

    /// While a buffer is still in_use, a second reserve allocates fresh.
    #[test]
    fn in_use_buffer_is_not_reused() {
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let a = pool.reserve(&[16], false).unwrap();
        let b = pool.reserve(&[16], false).unwrap();
        assert_ne!(a, b, "an in-use buffer must not be handed out twice");
        assert_eq!(pool.allocation_count(), 2);
    }

    /// Over-budget in-core reserve(allow_spill=false) HARD-refuses (D-01) — no
    /// downgrade, no allocation.
    #[test]
    fn over_budget_no_spill_refuses() {
        // 64-byte budget; a [100] f64 buffer needs 800 bytes.
        let pool = WorkspacePool::new(64);
        let err = pool
            .reserve(&[100], false)
            .expect_err("over-budget in-core must refuse");
        match err {
            BackendError::MemoryLimitExceeded { requested, limit } => {
                assert_eq!(requested, 800);
                assert_eq!(limit, 64);
            }
            other => panic!("expected MemoryLimitExceeded, got {other:?}"),
        }
        // No allocation occurred (no silent downgrade).
        assert_eq!(pool.allocation_count(), 0);
    }

    /// write_slice then as_slice round-trips through an in-memory buffer.
    #[test]
    fn inmemory_write_read_roundtrip() {
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let id = pool.reserve(&[4], false).unwrap();
        pool.write_slice(&id, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(pool.as_slice(&id).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// A spilled buffer (allow_spill=true over the in-memory budget) round-trips
    /// through HDF5 and deletes its temp file on Drop (RAII — T-06-02-LEAK).
    #[test]
    fn spilled_buffer_roundtrips_and_deletes_on_drop() {
        // Budget too small for an in-memory [50] f64 (400 bytes) but spill ok.
        let spill_path;
        {
            let pool = WorkspacePool::new(64);
            let id = pool
                .reserve(&[50], true)
                .expect("spill allowed over in-mem budget");
            // Capture the on-disk path via the pool internals for the
            // post-drop assertion.
            {
                let guard = pool.pool.lock().unwrap();
                let alloc = guard.iter().find(|a| a.id == id).unwrap();
                match &alloc.backend {
                    TensorBackend::Spilled(h) => spill_path = h.path().to_path_buf(),
                    TensorBackend::InMemory(_) => panic!("expected a spilled backend"),
                }
            }
            assert!(spill_path.exists(), "spill file must exist while reserved");

            let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
            pool.write_slice(&id, &data).unwrap();
            assert_eq!(pool.as_slice(&id).unwrap(), data);
        } // pool drops here → SpillHandle::drop deletes the temp file.
        assert!(
            !spill_path.exists(),
            "spill temp file must be deleted on drop (RAII, T-06-02-LEAK)"
        );
    }

    /// try_reserve (the Phase-1 ceiling check) keeps its exact contract.
    #[test]
    fn try_reserve_ceiling_unchanged() {
        let pool = WorkspacePool::new(1000);
        assert!(pool.try_reserve(1000).is_ok());
        assert!(matches!(
            pool.try_reserve(1001),
            Err(BackendError::MemoryLimitExceeded { .. })
        ));
    }
}
