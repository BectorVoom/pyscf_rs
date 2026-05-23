//! WorkspacePool — Phase 1 skeleton (RESEARCH §4 recommendation).
//!
//! Phase 1 ships a struct with three fields and three methods so the
//! Phase 6 (CCSD-11) tensor-arena lands without restructuring the
//! public surface. The body of `try_reserve` is currently a budget
//! check only; Phase 6 implements the actual buffer pool.

use crate::error::BackendError;

/// Phase-1 skeleton workspace pool. Body filled in Phase 6.
#[derive(Debug, Default)]
pub struct WorkspacePool {
    /// PYSCF_MAX_MEMORY ceiling in bytes. Default 4 GB.
    pub budget_bytes: usize,
    /// Free-list of buffer allocations. Phase 6 fills the inner
    /// PooledAllocation type. Written by `new()` today; Phase 6 (CCSD-11)
    /// adds the read path (reserve/release), so the field is dead until then.
    #[allow(dead_code)]
    pub(crate) pool: std::sync::Mutex<Vec<PooledAllocation>>,
}

#[derive(Debug)]
pub(crate) struct PooledAllocation {
    /// Phase 6 turns this into BufferId per-backend. Phase 1 stub.
    pub _bytes: Box<[u8]>,
    pub _size: usize,
}

impl WorkspacePool {
    /// 4 GiB default (PERF reasonable upper bound for v1 single-node).
    pub const DEFAULT_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;

    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            pool: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Read PYSCF_MAX_MEMORY (interpreted as MEGABYTES per upstream
    /// PySCF convention). Default 4 GB if unset or unparseable.
    pub fn from_env() -> Self {
        let budget = std::env::var("PYSCF_MAX_MEMORY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(Self::DEFAULT_BUDGET_BYTES);
        Self::new(budget)
    }

    /// Phase 1 stub: returns Err(MemoryLimitExceeded) if `bytes >
    /// budget_bytes`. Phase 6 (CCSD-11) implements an actual pool.
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
}
