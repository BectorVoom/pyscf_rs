//! `KTensor` — the k-indexed complex block container every Phase-16 tensor is
//! expressed in (plan 16-02 Task 3, D-PBC-29 clauses 1 and 4).
//!
//! # What upstream does, and what this replaces
//!
//! `pyscf/pbc/cc/kccsd_rhf.py` selects between **three** storage tiers in four
//! separate places — `:132-137` (`Woooo`), `:179-192` (`Wvoov`/`Wvovo`),
//! `:423-455` (`Wvvvv`) and `:777-832` (the seven `_ERIS` blocks): skip the
//! tensor entirely / an incore `np.empty` / an HDF5 `create_dataset`. The tier
//! is chosen from `mem_now` against `cc.max_memory`.
//!
//! **The tier here is chosen from an EXACT per-tensor byte count** —
//! `nkpts^k_rank * prod(block_shape) * 16` — and never from upstream's
//! `_mem_usage` (`kccsd_rhf.py:1100-1107`), which returns `nkpts³·nmo⁴·4·16`
//! and carries its own `# TODO: Improve incore estimate`. Measured against the
//! seven blocks actually allocated it over-estimates **9.1×** on diamond
//! `gth-szv` 2×2×2 and **6.2×** on `gth-dzvp` 2×2×2 (`16-REVIEW.md §2.4`).
//! Porting it literally would import that factor into this port's HARD
//! `MemoryLimitExceeded` refusal — i.e. it would refuse jobs that fit.
//!
//! # Why this type lives in `pyscf-pbc-cc` and not in `pyscf-runtime`
//!
//! The k-indexing is CC-specific and `pyscf-runtime` sits below
//! `pyscf-pbc-lib`. What comes from `pyscf-runtime` is the arena
//! ([`ZWorkspacePool`]); the `(nkpts, nkpts, nkpts)` addressing is this crate's.
//!
//! # The two access properties this container guarantees
//!
//! `16-REVIEW.md §2.2` records the two `WorkspacePool` behaviours that are
//! harmless for molecular CCSD and pathological at `nkpts³`, and
//! [`ZWorkspacePool`] already fixes both. `KTensor` preserves them:
//!
//! * [`KTensor::with_block`] borrows the block's planes; it does not copy the
//!   whole tensor per access;
//! * the pool's registry lock is not held across the caller's closure, and each
//!   allocation carries its own lock, so two rayon threads writing two
//!   different blocks make concurrent progress.

use std::sync::Arc;

use pyscf_algebra::CTensor;
use pyscf_runtime::{BackendError, ZBufferId, ZWorkspacePool};

use crate::error::PbcCcError;
use crate::zarr::ZArr;

/// How many k-indices address a block.
///
/// Upstream's `_ERIS` blocks are `[nkpts, nkpts, nkpts, ...]`-shaped
/// (`kccsd_rhf.py:789-794`) because the fourth k is fixed by momentum
/// conservation; the amplitudes `t1`/`t2` are rank 1 and rank 3 respectively
/// (`kccsd_rhf.py:553-554`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KRank {
    /// One k-index — `t1[ki]`, the Fock blocks.
    One,
    /// Two k-indices — the EOM `r2`-shaped quantities.
    Two,
    /// Three k-indices — `t2[ki,kj,ka]`, every `_ERIS` block, every `W`.
    Three,
}

impl KRank {
    /// Number of k-indices.
    pub fn n(&self) -> usize {
        match self {
            KRank::One => 1,
            KRank::Two => 2,
            KRank::Three => 3,
        }
    }

    /// Number of blocks at `nkpts`.
    pub fn blocks(&self, nkpts: usize) -> usize {
        nkpts.pow(self.n() as u32)
    }
}

/// The storage tier a [`KTensor`] landed in — the port's analogue of upstream's
/// three-way branch. Asserted directly by 16-02 test 10 and 16-05 test 4, so a
/// fixture that silently stayed incore fails rather than passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// The tensor is not built at all — upstream's first branch, taken when the
    /// consumer can recompute the block instead of storing it.
    Absent,
    /// Resident in the complex arena.
    InMemory,
    /// HDF5-spilled through the arena's spill backend.
    Spilled,
}

/// A `(nkpts^rank)`-indexed tensor of equal-shaped complex blocks, allocated
/// from a [`ZWorkspacePool`].
///
/// One pool buffer per block, so a block can be borrowed, written and released
/// independently — which is what makes the `nkpts³` rayon loops parallel.
#[derive(Debug)]
pub struct KTensor {
    nkpts: usize,
    rank: KRank,
    block_shape: Vec<usize>,
    block_len: usize,
    tier: Tier,
    /// One buffer per k-address, in row-major k order. Empty when
    /// `tier == Tier::Absent`.
    buffers: Vec<ZBufferId>,
}

impl KTensor {
    /// Exact byte requirement of the whole tensor:
    /// `nkpts^rank * prod(block_shape) * 16`.
    ///
    /// **This is the number the tier is chosen from** (D-PBC-29 clause 4), and
    /// the `16` is why the complex arena exists at all: an `f64`-sized count
    /// would halve it, and it feeds a HARD refusal.
    pub fn exact_bytes(nkpts: usize, rank: KRank, block_shape: &[usize]) -> usize {
        rank.blocks(nkpts)
            .saturating_mul(block_shape.iter().product::<usize>())
            .saturating_mul(16)
    }

    /// Allocate a zeroed tensor, choosing its tier from [`KTensor::exact_bytes`].
    ///
    /// `allow_spill` is the caller's opt-in to the HDF5 tier: with it `false`
    /// an over-budget tensor HARD-refuses (D-01, no silent downgrade); with it
    /// `true` the arena spills. The choice of tier is reported by
    /// [`KTensor::tier`] rather than inferred.
    ///
    /// # Errors
    /// [`BackendError::MemoryLimitExceeded`] when the tensor does not fit and
    /// spilling was not permitted; [`BackendError::ProbeFailed`] on a spill
    /// failure.
    pub fn zeros(
        pool: &ZWorkspacePool,
        nkpts: usize,
        rank: KRank,
        block_shape: &[usize],
        allow_spill: bool,
    ) -> Result<Self, BackendError> {
        let block_len: usize = block_shape.iter().product();
        let nblocks = rank.blocks(nkpts);

        // The whole-tensor pre-flight, on the exact count. A tensor that cannot
        // fit must refuse HERE, before nblocks partial allocations have already
        // been made and the process is deep in the loop.
        let total = Self::exact_bytes(nkpts, rank, block_shape);
        if !allow_spill {
            pool.try_reserve(total)?;
        }

        let mut buffers = Vec::with_capacity(nblocks);
        let mut spilled_any = false;
        for _ in 0..nblocks {
            let id = pool.reserve(block_shape, allow_spill)?;
            spilled_any |= pool.is_spilled(&id)?;
            buffers.push(id);
        }
        Ok(Self {
            nkpts,
            rank,
            block_shape: block_shape.to_vec(),
            block_len,
            tier: if spilled_any {
                Tier::Spilled
            } else {
                Tier::InMemory
            },
            buffers,
        })
    }

    /// The `Tier::Absent` branch — upstream's "do not build this tensor at all"
    /// (`kccsd_rhf.py:132-137`, `:423-455`). Carries its shape so a consumer can
    /// still ask what it WOULD have cost.
    pub fn absent(nkpts: usize, rank: KRank, block_shape: &[usize]) -> Self {
        Self {
            nkpts,
            rank,
            block_shape: block_shape.to_vec(),
            block_len: block_shape.iter().product(),
            tier: Tier::Absent,
            buffers: Vec::new(),
        }
    }

    /// Which of upstream's three tiers this tensor landed in.
    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// Number of k-points this tensor is indexed over.
    pub fn nkpts(&self) -> usize {
        self.nkpts
    }

    /// The k-rank.
    pub fn rank(&self) -> KRank {
        self.rank
    }

    /// Per-block shape (no k-indices).
    pub fn block_shape(&self) -> &[usize] {
        &self.block_shape
    }

    /// Complex elements per block.
    pub fn block_len(&self) -> usize {
        self.block_len
    }

    /// Exact bytes this tensor occupies.
    pub fn bytes(&self) -> usize {
        Self::exact_bytes(self.nkpts, self.rank, &self.block_shape)
    }

    /// Row-major flat index of a k-address. `k` must have `rank.n()` entries.
    fn kflat(&self, k: &[usize]) -> Result<usize, BackendError> {
        if k.len() != self.rank.n() {
            return Err(BackendError::ProbeFailed {
                backend: "ktensor",
                reason: format!(
                    "k-address of length {} for a rank-{} tensor",
                    k.len(),
                    self.rank.n()
                ),
            });
        }
        let mut idx = 0_usize;
        for &ki in k {
            if ki >= self.nkpts {
                return Err(BackendError::ProbeFailed {
                    backend: "ktensor",
                    reason: format!("k index {ki} out of range for nkpts {}", self.nkpts),
                });
            }
            idx = idx * self.nkpts + ki;
        }
        Ok(idx)
    }

    fn buffer(&self, k: &[usize]) -> Result<ZBufferId, BackendError> {
        if self.tier == Tier::Absent {
            return Err(BackendError::ProbeFailed {
                backend: "ktensor",
                reason: "block access on a Tier::Absent tensor".to_string(),
            });
        }
        Ok(self.buffers[self.kflat(k)?])
    }

    /// Borrow one block's `(re, im)` planes. **Does not copy the tensor.**
    ///
    /// # Errors
    /// [`BackendError::ProbeFailed`] for a bad k-address, a `Tier::Absent`
    /// tensor, or a spill read failure.
    pub fn with_block<R>(
        &self,
        pool: &ZWorkspacePool,
        k: &[usize],
        f: impl FnOnce(&[f64], &[f64]) -> R,
    ) -> Result<R, BackendError> {
        let id = self.buffer(k)?;
        pool.with_slices(&id, f)
    }

    /// Mutably borrow one block's planes. Only THIS block's lock is taken, so
    /// two threads writing two different blocks do not serialise
    /// (`16-REVIEW.md §2.2`).
    ///
    /// # Errors
    /// As [`KTensor::with_block`].
    pub fn with_block_mut<R>(
        &self,
        pool: &ZWorkspacePool,
        k: &[usize],
        f: impl FnOnce(&mut [f64], &mut [f64]) -> R,
    ) -> Result<R, BackendError> {
        let id = self.buffer(k)?;
        pool.with_mut_slices(&id, f)
    }

    /// Overwrite one block from a [`CTensor`].
    ///
    /// # Errors
    /// [`BackendError::ProbeFailed`] on a length mismatch or a bad k-address.
    pub fn set_block(
        &self,
        pool: &ZWorkspacePool,
        k: &[usize],
        value: &CTensor,
    ) -> Result<(), BackendError> {
        if value.re.len() != self.block_len {
            return Err(BackendError::ProbeFailed {
                backend: "ktensor",
                reason: format!(
                    "block length {} does not match the tensor's {}",
                    value.re.len(),
                    self.block_len
                ),
            });
        }
        let id = self.buffer(k)?;
        pool.write_planes(&id, &value.re, &value.im)
    }

    /// Read one block into an owned [`CTensor`]. Prefer [`KTensor::with_block`]
    /// inside an `nkpts³` loop — this one copies.
    ///
    /// # Errors
    /// As [`KTensor::with_block`].
    pub fn block(&self, pool: &ZWorkspacePool, k: &[usize]) -> Result<CTensor, BackendError> {
        self.with_block(pool, k, |re, im| CTensor {
            re: re.to_vec(),
            im: im.to_vec(),
        })
    }

    /// Return every block buffer to the arena's free-list. Call when the tensor
    /// goes out of scope in a loop that will allocate the same shape again.
    pub fn release(&self, pool: &ZWorkspacePool) {
        for id in &self.buffers {
            pool.release(*id);
        }
    }
}

/// A [`KTensor`] carried together with the arena it lives in, so a caller can
/// read and write blocks as shaped [`ZArr`]s without threading the pool.
///
/// This is what the `W` intermediates and every `nkpts³` CC block are returned
/// as: the tier decision (`kccsd_rhf.py:132-137`, `:179-192`, `:423-455`) rides
/// with the data rather than being re-derived by each consumer.
#[derive(Debug, Clone)]
pub struct KBlocks {
    pool: Arc<ZWorkspacePool>,
    inner: Arc<KTensor>,
    shape: Vec<usize>,
}

impl KBlocks {
    /// Allocate a zeroed rank-3 k-indexed tensor of `block_shape` blocks.
    ///
    /// # Errors
    /// The arena's HARD refusal, or a spill failure.
    pub fn zeros(
        pool: &Arc<ZWorkspacePool>,
        nkpts: usize,
        block_shape: &[usize],
        allow_spill: bool,
    ) -> Result<Self, PbcCcError> {
        let inner = KTensor::zeros(pool, nkpts, KRank::Three, block_shape, allow_spill)?;
        Ok(Self {
            pool: Arc::clone(pool),
            inner: Arc::new(inner),
            shape: block_shape.to_vec(),
        })
    }

    /// Allocate, choosing the tier from the exact byte count against
    /// `max_memory_bytes`: in memory when it fits, spilled when it does not.
    ///
    /// This is the port of upstream's three-way branch, with the estimate
    /// replaced by an exact count (D-PBC-29 clause 4).
    ///
    /// # Errors
    /// A spill failure.
    pub fn with_budget(
        pool: &Arc<ZWorkspacePool>,
        nkpts: usize,
        block_shape: &[usize],
        max_memory_bytes: usize,
    ) -> Result<Self, PbcCcError> {
        let need = KTensor::exact_bytes(nkpts, KRank::Three, block_shape);
        let fits = pool.live_inmem_bytes().saturating_add(need) <= max_memory_bytes;
        Self::zeros(pool, nkpts, block_shape, !fits)
    }

    /// The block at `[k0, k1, k2]`.
    ///
    /// # Errors
    /// A bad k-address, or a spill read failure.
    pub fn get(&self, k: [usize; 3]) -> Result<ZArr, PbcCcError> {
        let c = self.inner.block(&self.pool, &k)?;
        Ok(ZArr::from_ctensor(&self.shape, c)?)
    }

    /// Overwrite the block at `[k0, k1, k2]`.
    ///
    /// # Errors
    /// A shape mismatch, a bad k-address, or a spill write failure.
    pub fn set(&self, k: [usize; 3], v: &ZArr) -> Result<(), PbcCcError> {
        if v.shape() != self.shape.as_slice() {
            return Err(PbcCcError::Shape(format!(
                "KBlocks::set: block shape {:?} does not match {:?}",
                v.shape(),
                self.shape
            )));
        }
        self.inner.set_block(&self.pool, &k, v.data())?;
        Ok(())
    }

    /// Which storage tier this tensor landed in.
    pub fn tier(&self) -> Tier {
        self.inner.tier()
    }

    /// Exact bytes.
    pub fn bytes(&self) -> usize {
        self.inner.bytes()
    }

    /// Per-block shape.
    pub fn block_shape(&self) -> &[usize] {
        &self.shape
    }

    /// k-point count.
    pub fn nkpts(&self) -> usize {
        self.inner.nkpts()
    }

    /// Return every buffer to the arena's free-list.
    pub fn release(&self) {
        self.inner.release(&self.pool);
    }
}
