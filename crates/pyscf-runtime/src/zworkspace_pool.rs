//! `ZWorkspacePool` — the COMPLEX tensor arena (Phase 16, D-PBC-29 clause 1).
//!
//! # Why this is a new type and not a change to [`WorkspacePool`]
//!
//! `PBC-MASTER-PLAN §8.8`'s Reuse note tells Phase 16 to reuse the molecular
//! arena. `16-CONTEXT §1.3` shows that instruction cannot be followed
//! literally: [`crate::WorkspacePool`] is `f64` all the way down —
//! `shape_bytes` is `product * 8` (`workspace_pool.rs:278-280`),
//! `TensorBackend::InMemory(Box<[f64]>)`, `as_slice -> Vec<f64>` (`:397`),
//! `write_slice(&[f64])` (`:422`) — and every k-point CC tensor is
//! `complex128` (`kccsd_rhf.py:553-554`).
//!
//! **Reinterpreting a `Box<[f64]>` as complex pairs is forbidden.**
//! `shape_bytes`'s `* 8` feeds `try_reserve`, which IS the HARD
//! `MemoryLimitExceeded` refusal (`workspace_pool.rs:266-274`). A complex
//! tensor sized with `* 8` reports HALF its footprint to the one mechanism
//! whose job is to refuse before an OOM — and on the machine this project runs
//! on (17-12's whole host suite SIGKILLed, exit 137) a 2× under-report is the
//! failure mode, not a rounding error.
//!
//! **The f64 pool is left byte-for-byte unchanged.** It has shipped callers
//! (`pyscf-ccsd`'s `rdm.rs:32`, `lambda.rs:37`) and shipped tests
//! (`heap_alloc_count.rs`); 16-14 is not the place to discover a regression in
//! molecular CCSD's `rdm`/`lambda`. What is shared is the *shape*: the budget
//! ceiling, the free-list, the `InMemory | Spilled` split, and the HARD refusal
//! with no silent downgrade.
//!
//! # The two f64-pool properties this type deliberately does NOT copy
//!
//! `16-REVIEW.md §2.2` records both, harmless for molecular CCSD and
//! pathological at `nkpts³`:
//!
//! * **`WorkspacePool::as_slice` copies the whole buffer on every access**
//!   (`:397-410`, `Ok(b.to_vec())`). A k-point `update_amps` touches ERI blocks
//!   inside an `nkpts³` loop; copying `Wvvvv` per access is a different
//!   complexity class, not a constant factor. Here
//!   [`ZWorkspacePool::with_slices`] BORROWS.
//! * **`WorkspacePool::with_mut_slice` holds the pool's single mutex across the
//!   caller's closure** (`:461-483`), so two rayon threads working on two
//!   *different* buffers serialise — which would cap the whole phase at one
//!   core. Here the registry lock is released before the closure runs and each
//!   allocation carries its OWN `Mutex`.
//!
//! # Planar storage (RULE 8 / D-PBC-02)
//!
//! Complex numbers never cross the ALG-06 wall as a `Complex<f64>` element
//! type: they are carried as two equal-length `f64` planes
//! (`pyscf_algebra::CTensor`). This arena stores the same way — a `re` plane
//! and an `im` plane per buffer — so a reserved block hands straight to the
//! existing real cubecl primitives. The byte accounting is unaffected: a
//! complex element is 16 bytes either way.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::BackendError;
use pyscf_chkfile::hdf5;

/// Opaque complex-pool buffer handle.
///
/// Deliberately a DIFFERENT type from [`crate::BufferId`]: an f64 handle and a
/// complex handle must not be interchangeable, because the whole point of this
/// module is that the two arenas size their buffers differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZBufferId(pub(crate) u64);

impl ZBufferId {
    /// Raw id (diagnostics / tests).
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// A resident planar complex buffer: two equal-length `f64` planes.
#[derive(Debug)]
pub struct ZBuffer {
    re: Box<[f64]>,
    im: Box<[f64]>,
}

impl ZBuffer {
    fn zeros(n: usize) -> Self {
        Self {
            re: vec![0.0_f64; n].into_boxed_slice(),
            im: vec![0.0_f64; n].into_boxed_slice(),
        }
    }

    /// Complex element count.
    pub fn len(&self) -> usize {
        self.re.len()
    }

    /// `true` when the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.re.is_empty()
    }
}

/// Storage backend behind a [`ZBufferId`] — the same `InMemory | Spilled`
/// split the f64 arena uses (D-01/D-08).
#[derive(Debug)]
pub enum ZTensorBackend {
    /// Resident planar complex buffer.
    InMemory(ZBuffer),
    /// HDF5-spilled buffer (RAII drop-deletes the temp file).
    Spilled(ZSpillHandle),
}

impl ZTensorBackend {
    /// Complex element capacity.
    ///
    /// The free-list scan does NOT call this — it reads
    /// `ZAllocation::capacity`, which needs no lock — so this exists for
    /// callers that already hold the backend.
    pub fn len(&self) -> usize {
        match self {
            ZTensorBackend::InMemory(b) => b.len(),
            ZTensorBackend::Spilled(h) => h.len,
        }
    }

    /// `true` when the backend holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// HDF5-backed complex spill buffer — the `lib.H5TmpFile()` equivalent (D-07).
///
/// Two datasets, `"re"` and `"im"`, keeping the planar convention on disk as
/// well as in memory. On `Drop` the handle is closed and the temp file removed
/// (RAII — no leftover scratch; the T-06-02-LEAK mitigation).
#[derive(Debug)]
pub struct ZSpillHandle {
    file: Option<hdf5::File>,
    path: std::path::PathBuf,
    len: usize,
}

impl ZSpillHandle {
    const RE: &'static str = "re";
    const IM: &'static str = "im";

    fn create(len: usize, uid: u64) -> Result<Self, BackendError> {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "pyscf_kcc_zspill_{}_{}.h5",
            std::process::id(),
            uid
        ));
        let file = hdf5::File::create(&path).map_err(|e| BackendError::ProbeFailed {
            backend: "hdf5-zspill",
            reason: format!("create temp spill file {}: {e}", path.display()),
        })?;
        for name in [Self::RE, Self::IM] {
            let arr = ndarray::Array1::from_vec(vec![0.0_f64; len]);
            file.new_dataset::<f64>()
                .shape([len])
                .create(name)
                .and_then(|ds| ds.write(&arr))
                .map_err(|e| BackendError::ProbeFailed {
                    backend: "hdf5-zspill",
                    reason: format!("create spill dataset {name}: {e}"),
                })?;
        }
        Ok(Self {
            file: Some(file),
            path,
            len,
        })
    }

    fn read_plane(&self, name: &str) -> Result<Vec<f64>, BackendError> {
        let file = self.file.as_ref().ok_or_else(|| BackendError::ProbeFailed {
            backend: "hdf5-zspill",
            reason: "spill file already closed".to_string(),
        })?;
        let arr: ndarray::Array1<f64> = file
            .dataset(name)
            .and_then(|ds| ds.read_1d())
            .map_err(|e| BackendError::ProbeFailed {
                backend: "hdf5-zspill",
                reason: format!("read spill dataset {name}: {e}"),
            })?;
        Ok(arr.to_vec())
    }

    fn write_plane(&self, name: &str, data: &[f64]) -> Result<(), BackendError> {
        if data.len() != self.len {
            return Err(BackendError::ProbeFailed {
                backend: "hdf5-zspill",
                reason: format!(
                    "spill write length mismatch: got {}, expected {}",
                    data.len(),
                    self.len
                ),
            });
        }
        let file = self.file.as_ref().ok_or_else(|| BackendError::ProbeFailed {
            backend: "hdf5-zspill",
            reason: "spill file already closed".to_string(),
        })?;
        let arr = ndarray::Array1::from_vec(data.to_vec());
        file.dataset(name)
            .and_then(|ds| ds.write(&arr))
            .map_err(|e| BackendError::ProbeFailed {
                backend: "hdf5-zspill",
                reason: format!("write spill dataset {name}: {e}"),
            })?;
        Ok(())
    }

    /// On-disk path (for tests asserting the RAII delete).
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for ZSpillHandle {
    fn drop(&mut self) {
        self.file = None;
        let _ = std::fs::remove_file(&self.path);
    }
}

/// One pooled complex allocation.
///
/// The backend sits behind its OWN `Mutex`, not the pool's: that is what lets
/// two rayon threads write two different blocks concurrently
/// (`16-REVIEW.md §2.2`).
#[derive(Debug)]
struct ZAllocation {
    id: ZBufferId,
    backend: Mutex<ZTensorBackend>,
    /// Complex element capacity, readable WITHOUT taking the backend lock —
    /// the free-list scan must not block on a buffer another thread is writing.
    capacity: usize,
    /// Bytes this allocation charges against the in-memory budget (0 when
    /// spilled), also readable without the backend lock.
    inmem_bytes: usize,
    /// Checked-out flag. An `AtomicBool` rather than a plain `bool` because a
    /// caller may hold an `Arc` to this allocation (inside `with_slices`) while
    /// another thread releases a different one: `Arc::get_mut` would spuriously
    /// fail in that window and silently leak the buffer off the free-list.
    in_use: AtomicBool,
}

/// The Phase-16 complex tensor arena: a budget-checked reuse pool of planar
/// complex buffers behind opaque [`ZBufferId`] handles.
#[derive(Debug, Default)]
pub struct ZWorkspacePool {
    /// `PYSCF_MAX_MEMORY` ceiling in bytes. Default 4 GiB.
    pub budget_bytes: usize,
    /// Registry of allocations. This lock guards only the REGISTRY, and is
    /// never held across a caller closure.
    pool: Mutex<Vec<Arc<ZAllocation>>>,
    next_id: Mutex<u64>,
    live_inmem_bytes: Mutex<usize>,
}

impl ZWorkspacePool {
    /// 4 GiB default — the same ceiling [`crate::WorkspacePool`] uses.
    pub const DEFAULT_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;

    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            pool: Mutex::new(Vec::new()),
            next_id: Mutex::new(0),
            live_inmem_bytes: Mutex::new(0),
        }
    }

    /// Read `PYSCF_MAX_MEMORY` (MEGABYTES, the upstream PySCF convention).
    pub fn from_env() -> Self {
        let budget = std::env::var("PYSCF_MAX_MEMORY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(Self::DEFAULT_BUDGET_BYTES);
        Self::new(budget)
    }

    /// Bytes a COMPLEX buffer of `shape` requires: `product * 16`.
    ///
    /// **The `16` is the whole reason this type exists** (D-PBC-29 clause 1).
    /// `WorkspacePool::shape_bytes` is `product * 8`; using it for a complex
    /// tensor halves the number that reaches the HARD refusal below.
    pub fn shape_bytes(shape: &[usize]) -> usize {
        shape.iter().product::<usize>().saturating_mul(16)
    }

    /// HARD pre-flight refusal (D-01), unchanged in contract from the f64 pool:
    /// a single in-core reservation larger than the whole budget refuses.
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

    /// `true` while the handle is checked out (diagnostics / tests).
    pub fn is_in_use(&self, id: &ZBufferId) -> Result<bool, BackendError> {
        Ok(self.alloc(id)?.in_use.load(Ordering::Acquire))
    }

    /// Reserve a complex buffer of `shape`, reusing a free-listed buffer of
    /// sufficient capacity when one exists — otherwise allocating fresh.
    ///
    /// * `allow_spill == false` and over budget → `Err(MemoryLimitExceeded)`
    ///   (D-01 HARD refusal, NO silent downgrade).
    /// * `allow_spill == true` and over budget → an HDF5-backed `Spilled`
    ///   buffer.
    ///
    /// # Errors
    /// [`BackendError::MemoryLimitExceeded`] when over budget and spill is not
    /// permitted; [`BackendError::ProbeFailed`] if the HDF5 spill file cannot be
    /// created or a lock is poisoned. Never panics.
    pub fn reserve(&self, shape: &[usize], allow_spill: bool) -> Result<ZBufferId, BackendError> {
        let need_elems = shape.iter().product::<usize>();
        let need_bytes = Self::shape_bytes(shape);

        // 1. Free-list scan — capacity is read from the allocation record, not
        //    from behind the backend lock, so a buffer being written by another
        //    thread does not stall the scan.
        {
            let pool = self
                .pool
                .lock()
                .map_err(|_| Self::poisoned("zworkspace pool"))?;
            let mut best: Option<usize> = None;
            for (idx, alloc) in pool.iter().enumerate() {
                if alloc.in_use.load(Ordering::Acquire) || alloc.capacity < need_elems {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some(b) => alloc.capacity < pool[b].capacity,
                };
                if better {
                    best = Some(idx);
                }
            }
            if let Some(idx) = best {
                pool[idx].in_use.store(true, Ordering::Release);
                return Ok(pool[idx].id);
            }
        }

        self.fresh(need_elems, need_bytes, allow_spill)
    }

    fn fresh(
        &self,
        need_elems: usize,
        need_bytes: usize,
        allow_spill: bool,
    ) -> Result<ZBufferId, BackendError> {
        let fits_inmem = {
            let live = self
                .live_inmem_bytes
                .lock()
                .map_err(|_| Self::poisoned("live-bytes"))?;
            live.saturating_add(need_bytes) <= self.budget_bytes
        };

        // Over budget AND spill not permitted: HARD refuse (D-01) BEFORE
        // consuming an id or allocating anything. NO silent downgrade.
        if !fits_inmem && !allow_spill {
            return Err(BackendError::MemoryLimitExceeded {
                requested: need_bytes,
                limit: self.budget_bytes,
            });
        }

        let id = self.next_buffer_id()?;
        let (backend, charged) = if fits_inmem {
            {
                let mut live = self
                    .live_inmem_bytes
                    .lock()
                    .map_err(|_| Self::poisoned("live-bytes"))?;
                *live = live.saturating_add(need_bytes);
            }
            (ZTensorBackend::InMemory(ZBuffer::zeros(need_elems)), need_bytes)
        } else {
            (
                ZTensorBackend::Spilled(ZSpillHandle::create(need_elems, id.0)?),
                0,
            )
        };

        let mut pool = self
            .pool
            .lock()
            .map_err(|_| Self::poisoned("zworkspace pool"))?;
        pool.push(Arc::new(ZAllocation {
            id,
            backend: Mutex::new(backend),
            capacity: need_elems,
            inmem_bytes: charged,
            in_use: AtomicBool::new(true),
        }));
        Ok(id)
    }

    /// Release a buffer back to the free-list. Does NOT drop the storage — the
    /// next `reserve` of a fitting shape reuses it (the allocate-once-reuse
    /// guarantee). A no-op for an unknown or already-released id.
    pub fn release(&self, id: ZBufferId) {
        if let Ok(pool) = self.pool.lock()
            && let Some(alloc) = pool.iter().find(|a| a.id == id)
        {
            alloc.in_use.store(false, Ordering::Release);
        }
    }

    /// Whether a reserved buffer is spilled (tests / tier assertions).
    pub fn is_spilled(&self, id: &ZBufferId) -> Result<bool, BackendError> {
        let alloc = self.alloc(id)?;
        let backend = alloc
            .backend
            .lock()
            .map_err(|_| Self::poisoned("zbuffer"))?;
        Ok(matches!(&*backend, ZTensorBackend::Spilled(_)))
    }

    /// Complex element capacity of a reserved buffer.
    pub fn capacity(&self, id: &ZBufferId) -> Result<usize, BackendError> {
        Ok(self.alloc(id)?.capacity)
    }

    /// Run `f` over the buffer's `(re, im)` planes WITHOUT copying them.
    ///
    /// This is the access path the `nkpts³` contraction loops use. Only the
    /// buffer's own lock is held; the registry lock is released before `f`
    /// runs, so two threads working on two different buffers do not serialise
    /// (`16-REVIEW.md §2.2`).
    ///
    /// A `Spilled` buffer has no resident planes, so its contents are read into
    /// a temporary and `f` sees that; the borrow-not-copy guarantee is for the
    /// in-memory tier, which is where the hot loops live.
    ///
    /// # Errors
    /// [`BackendError::ProbeFailed`] for an unknown id, a poisoned lock, or a
    /// spill read failure.
    pub fn with_slices<R>(
        &self,
        id: &ZBufferId,
        f: impl FnOnce(&[f64], &[f64]) -> R,
    ) -> Result<R, BackendError> {
        let alloc = self.alloc(id)?;
        let backend = alloc
            .backend
            .lock()
            .map_err(|_| Self::poisoned("zbuffer"))?;
        match &*backend {
            ZTensorBackend::InMemory(b) => Ok(f(&b.re, &b.im)),
            ZTensorBackend::Spilled(h) => {
                let re = h.read_plane(ZSpillHandle::RE)?;
                let im = h.read_plane(ZSpillHandle::IM)?;
                Ok(f(&re, &im))
            }
        }
    }

    /// Mutable planar access. In-memory buffers are edited in place; a spilled
    /// buffer is read, edited in a temporary, and written back.
    ///
    /// # Errors
    /// As [`ZWorkspacePool::with_slices`].
    pub fn with_mut_slices<R>(
        &self,
        id: &ZBufferId,
        f: impl FnOnce(&mut [f64], &mut [f64]) -> R,
    ) -> Result<R, BackendError> {
        let alloc = self.alloc(id)?;
        let mut backend = alloc
            .backend
            .lock()
            .map_err(|_| Self::poisoned("zbuffer"))?;
        match &mut *backend {
            ZTensorBackend::InMemory(b) => Ok(f(&mut b.re, &mut b.im)),
            ZTensorBackend::Spilled(h) => {
                let mut re = h.read_plane(ZSpillHandle::RE)?;
                let mut im = h.read_plane(ZSpillHandle::IM)?;
                let out = f(&mut re, &mut im);
                h.write_plane(ZSpillHandle::RE, &re)?;
                h.write_plane(ZSpillHandle::IM, &im)?;
                Ok(out)
            }
        }
    }

    /// Overwrite a reserved buffer from planar slices. Lengths must not exceed
    /// the reserved capacity and must match each other.
    ///
    /// # Errors
    /// [`BackendError::ProbeFailed`] for an unknown id, a plane-length
    /// mismatch, an over-capacity write, or a spill write failure.
    pub fn write_planes(
        &self,
        id: &ZBufferId,
        re: &[f64],
        im: &[f64],
    ) -> Result<(), BackendError> {
        if re.len() != im.len() {
            return Err(BackendError::ProbeFailed {
                backend: "zworkspace-pool",
                reason: format!("plane length mismatch: re {} im {}", re.len(), im.len()),
            });
        }
        let alloc = self.alloc(id)?;
        let mut backend = alloc
            .backend
            .lock()
            .map_err(|_| Self::poisoned("zbuffer"))?;
        match &mut *backend {
            ZTensorBackend::InMemory(b) => {
                if re.len() > b.len() {
                    return Err(BackendError::ProbeFailed {
                        backend: "zworkspace-pool",
                        reason: format!(
                            "write length {} exceeds buffer capacity {}",
                            re.len(),
                            b.len()
                        ),
                    });
                }
                b.re[..re.len()].copy_from_slice(re);
                b.im[..im.len()].copy_from_slice(im);
                Ok(())
            }
            ZTensorBackend::Spilled(h) => {
                h.write_plane(ZSpillHandle::RE, re)?;
                h.write_plane(ZSpillHandle::IM, im)?;
                Ok(())
            }
        }
    }

    /// Read a reserved buffer into owned planes. Prefer
    /// [`ZWorkspacePool::with_slices`] in a hot loop — this one copies, exactly
    /// as `WorkspacePool::as_slice` does, and is here for the boundaries where
    /// an owned value is genuinely wanted.
    ///
    /// # Errors
    /// As [`ZWorkspacePool::with_slices`].
    pub fn read_planes(&self, id: &ZBufferId) -> Result<(Vec<f64>, Vec<f64>), BackendError> {
        self.with_slices(id, |re, im| (re.to_vec(), im.to_vec()))
    }

    /// Number of allocations on the pool (in_use + free-listed).
    pub fn allocation_count(&self) -> usize {
        self.pool.lock().map(|p| p.len()).unwrap_or(0)
    }

    /// Bytes currently charged against the in-memory budget. Spilled buffers
    /// charge nothing. This is the quantity Phase-16 peak-memory assertions
    /// read (`16-REVIEW.md §4`).
    pub fn live_inmem_bytes(&self) -> usize {
        self.live_inmem_bytes.lock().map(|v| *v).unwrap_or(0)
    }

    /// Bytes charged by ONE reserved buffer (0 when it is spilled).
    ///
    /// # Errors
    /// [`BackendError::ProbeFailed`] for an unknown id.
    pub fn charged_bytes(&self, id: &ZBufferId) -> Result<usize, BackendError> {
        Ok(self.alloc(id)?.charged_bytes())
    }

    fn alloc(&self, id: &ZBufferId) -> Result<Arc<ZAllocation>, BackendError> {
        let pool = self
            .pool
            .lock()
            .map_err(|_| Self::poisoned("zworkspace pool"))?;
        pool.iter()
            .find(|a| a.id == *id)
            .cloned()
            .ok_or_else(|| Self::unknown_id(id))
    }

    fn next_buffer_id(&self) -> Result<ZBufferId, BackendError> {
        let mut n = self
            .next_id
            .lock()
            .map_err(|_| Self::poisoned("id counter"))?;
        let id = ZBufferId(*n);
        *n = n.saturating_add(1);
        Ok(id)
    }

    fn poisoned(what: &'static str) -> BackendError {
        BackendError::ProbeFailed {
            backend: "zworkspace-pool",
            reason: format!("{what} mutex poisoned"),
        }
    }

    fn unknown_id(id: &ZBufferId) -> BackendError {
        BackendError::ProbeFailed {
            backend: "zworkspace-pool",
            reason: format!("unknown complex buffer id {}", id.0),
        }
    }
}

impl ZAllocation {
    /// Bytes this allocation charges against the in-memory budget. A spilled
    /// allocation charges nothing, which is what makes the spill tier a way to
    /// stay under the ceiling rather than a way to hide from it.
    fn charged_bytes(&self) -> usize {
        self.inmem_bytes
    }
}
