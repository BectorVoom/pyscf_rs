//! Coupled-cluster amplitude tensors (D-01).
//!
//! Phase 1 declared `Amplitudes { t1: Vec<f64>, t2: Vec<f64> }` as a placeholder.
//! Phase 6 (CCSD-11/D-01) upgrades `t1`/`t2` to OPAQUE handles
//! ([`AmplitudeStore`]) so the storage backend can be either an owned in-memory
//! buffer OR a `pyscf-runtime::WorkspacePool`-pooled / HDF5-spillable buffer,
//! transparently to the consumer — so `Wabef` (and friends) don't
//! allocate-and-drop per iteration (CCSD-11, Pitfall 20).
//!
//! ## Dependency-direction constraint (why the handle lives HERE, not in
//! pyscf-runtime)
//!
//! D-08 places the `WorkspacePool` + `TensorBackend` (the PRODUCER) in
//! `pyscf-runtime`. `Amplitudes` is the CONSUMER. But `pyscf-runtime` already
//! depends on `pyscf-core`, so `pyscf-core` CANNOT depend on `pyscf-runtime`
//! (cycle) — and `pyscf-runtime`'s default feature pulls `cubecl`, which would
//! also violate `pyscf-core`'s FOUND-02 "no compute deps" rule. So the opaque
//! handle ([`AmplitudeStore`]) is defined HERE as a dependency-free enum:
//!   * `Owned(Vec<f64>)` — the MP2 path (MP2 has no arena; it builds amplitudes
//!     directly and stores them owned).
//!   * `Pooled(PooledRef)` — the CCSD path (a `{ buffer_id, shape }` reference
//!     into a `WorkspacePool`). The pool buffer is read/written through the
//!     pool API at the call site (pyscf-ccsd depends on BOTH crates), so
//!     `pyscf-core` needs no runtime types — just the plain id + shape.
//!
//! This satisfies D-01's "opaque Tensor handles" (the field type is no longer a
//! raw `Vec<f64>`) while keeping `pyscf-core` dependency-clean.

/// A reference to a buffer living in a `pyscf-runtime::WorkspacePool` (the CCSD
/// arena path).
///
/// Dependency-free by design: it carries only the pool-assigned `buffer_id`
/// (the raw `u64` from `pyscf_runtime::BufferId::raw()`) and the logical
/// `shape`. The actual buffer bytes are read/written through the pool API at
/// the call site that owns the pool handle (pyscf-ccsd). `pyscf-core` therefore
/// needs no `pyscf-runtime` dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PooledRef {
    /// The pool-assigned buffer id (`pyscf_runtime::BufferId::raw()`).
    pub buffer_id: u64,
    /// Logical shape of the amplitude tensor.
    pub shape: Vec<usize>,
}

/// Opaque amplitude storage backend (D-01).
///
/// Either an owned in-memory buffer (MP2 / small in-core) or a reference into a
/// `WorkspacePool` (the CCSD arena, spillable). Consumers access the data via
/// the [`Amplitudes`] accessors; the variant is transparent to amplitude math.
#[derive(Debug, Clone, PartialEq)]
pub enum AmplitudeStore {
    /// Owned resident buffer (the MP2 path — no arena).
    Owned(Vec<f64>),
    /// Reference into a `pyscf-runtime::WorkspacePool` buffer (the CCSD path).
    Pooled(PooledRef),
}

impl Default for AmplitudeStore {
    fn default() -> Self {
        AmplitudeStore::Owned(Vec::new())
    }
}

impl AmplitudeStore {
    /// Resident slice for an `Owned` store; `None` for a `Pooled` reference
    /// (whose bytes live in the pool and are read through the pool API).
    pub fn as_owned_slice(&self) -> Option<&[f64]> {
        match self {
            AmplitudeStore::Owned(v) => Some(v),
            AmplitudeStore::Pooled(_) => None,
        }
    }

    /// The pooled reference, if this is a `Pooled` store.
    pub fn as_pooled(&self) -> Option<&PooledRef> {
        match self {
            AmplitudeStore::Owned(_) => None,
            AmplitudeStore::Pooled(r) => Some(r),
        }
    }
}

/// CCSD amplitude container (D-01).
///
/// `t1`/`t2` are now opaque [`AmplitudeStore`] handles rather than raw
/// `Vec<f64>`. The MP2 construction path uses [`Amplitudes::from_vec`] (owned
/// storage); the CCSD arena path uses [`Amplitudes::from_pooled`] (a pool
/// reference). Accessors [`Amplitudes::t1_slice`] / [`Amplitudes::t2_slice`]
/// return the resident data for owned stores (MP2/RDM readers), and `None` for
/// pooled stores (CCSD reads through the pool).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Amplitudes {
    pub nocc: usize,
    pub nvir: usize,
    /// t1 amplitudes `[nocc, nvir]` flattened (opaque store).
    pub t1: AmplitudeStore,
    /// t2 amplitudes `[nocc, nocc, nvir, nvir]` flattened (opaque store).
    pub t2: AmplitudeStore,
}

impl Amplitudes {
    /// Construct from owned `Vec<f64>` buffers (the MP2 path — no arena).
    ///
    /// This is the Vec-compat constructor the MP2 `rmp2_kernel` site uses so it
    /// keeps compiling unchanged after the D-01 handle upgrade.
    pub fn from_vec(nocc: usize, nvir: usize, t1: Vec<f64>, t2: Vec<f64>) -> Self {
        Self {
            nocc,
            nvir,
            t1: AmplitudeStore::Owned(t1),
            t2: AmplitudeStore::Owned(t2),
        }
    }

    /// Construct from `WorkspacePool` buffer references (the CCSD arena path).
    ///
    /// Each `PooledRef` carries the pool-assigned buffer id + logical shape; the
    /// buffer bytes are read/written through the pool API by the owner of the
    /// pool handle (pyscf-ccsd).
    pub fn from_pooled(nocc: usize, nvir: usize, t1: PooledRef, t2: PooledRef) -> Self {
        Self {
            nocc,
            nvir,
            t1: AmplitudeStore::Pooled(t1),
            t2: AmplitudeStore::Pooled(t2),
        }
    }

    /// Resident `t1` slice for an owned store; `None` for a pooled reference.
    pub fn t1_slice(&self) -> Option<&[f64]> {
        self.t1.as_owned_slice()
    }

    /// Resident `t2` slice for an owned store; `None` for a pooled reference.
    pub fn t2_slice(&self) -> Option<&[f64]> {
        self.t2.as_owned_slice()
    }
}
