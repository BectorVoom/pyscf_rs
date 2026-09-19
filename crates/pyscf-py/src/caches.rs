//! Free-thread-safe `PyOnceLock` caches (BIND-06).
//!
//! Replaces `lazy_static!` for any cache that holds Python object handles
//! or is touched by multiple Python threads. Under free-threaded Python
//! 3.13t, `lazy_static!`-style `std::sync::Once` deadlocks because it lacks
//! PyO3-aware coordination; `pyo3::sync::PyOnceLock` is the supported
//! alternative.
//!
//! Lint enforcement: `xtask check-forbid-lazy-static` (plan 03-02) blocks
//! any `lazy_static!` invocation under `crates/pyscf-py/`.
//!
//! Phase 3 use case: cache the type-id of Python subclasses that override
//! SCF hooks. The override-detection fast path looks up the type-id in this
//! cache instead of calling `hasattr()` once per cycle per hook. Lazy init
//! on first kernel call; thread-safe across free-threaded interpreters.
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Set of Python type-id raw pointers known to have at least one SCF hook
/// override. The pointer is the address of the Python type object; it is
/// stable across the lifetime of the type, so caching it is safe.
///
/// The `Mutex<HashSet>` shape provides interior mutability under the
/// `PyOnceLock` initialisation barrier. Reads + writes go through the
/// mutex; the set is small (typically ≤ a few entries per session) so
/// contention is negligible.
static OVERRIDE_TYPE_CACHE: PyOnceLock<Mutex<HashSet<usize>>> = PyOnceLock::new();

/// Get the override-detection cache, initialising it on first call.
///
/// Returns a `&'static Mutex<HashSet<usize>>` so callers can lock + mutate.
pub fn override_cache(py: pyo3::Python<'_>) -> &'static Mutex<HashSet<usize>> {
    OVERRIDE_TYPE_CACHE.get_or_init(py, || Mutex::new(HashSet::new()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Plan 20-11 — the k-point override probe (`pbc/kbridge.rs`).
// ─────────────────────────────────────────────────────────────────────────────

/// One cached probe: which of the eleven `KOverrideHooks` a Python type
/// overrides relative to a native base class, as a bitmask over
/// `pbc::kbridge::K_HOOKS`.
///
/// The entry holds STRONG references to both type objects. The key is their
/// addresses, and an address is only a stable identity while the object is
/// alive: without the reference a class defined inside a function could be
/// collected and a NEW class allocated at the same address would inherit a
/// stale mask. The set of distinct driver subclasses in a session is small,
/// so keeping them alive is the cheap side of that trade.
pub struct KOverrideProbe {
    _ty: Py<PyType>,
    _base: Py<PyType>,
    mask: u16,
}

static K_OVERRIDE_PROBE_CACHE: PyOnceLock<Mutex<HashMap<(usize, usize), KOverrideProbe>>> =
    PyOnceLock::new();

/// How many probes have actually run (cache misses). Read by the private
/// `_native.pbc.scf._kbridge_probe_count()` test hook.
static K_OVERRIDE_PROBES_RUN: AtomicUsize = AtomicUsize::new(0);

/// The override mask of `ty` relative to `base`, probing at most once per
/// `(ty, base)` pair for the life of the interpreter.
///
/// `probe` runs with the cache lock RELEASED: it calls into Python
/// (`__mro__`, `__dict__`), and holding a Rust mutex across a call that may
/// release the GIL is how a second thread deadlocks against it. Two threads
/// racing on a cold type both probe and the second insert wins — both compute
/// the same mask, so the race is benign.
///
/// A class mutated AFTER its first probe (`Cls.get_veff = f` on the class
/// object) keeps its old mask; an INSTANCE attribute (`mf.get_veff = f`) is
/// not cached and is always seen (`pbc/kbridge.rs`).
pub fn k_override_mask<F>(
    py: Python<'_>,
    ty: &Bound<'_, PyType>,
    base: &Bound<'_, PyType>,
    probe: F,
) -> PyResult<u16>
where
    F: FnOnce() -> PyResult<u16>,
{
    let key = (ty.as_ptr() as usize, base.as_ptr() as usize);
    let cache = K_OVERRIDE_PROBE_CACHE.get_or_init(py, || Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(hit.mask);
    }
    let mask = probe()?;
    K_OVERRIDE_PROBES_RUN.fetch_add(1, Ordering::Relaxed);
    cache.lock().unwrap_or_else(|p| p.into_inner()).insert(
        key,
        KOverrideProbe {
            _ty: ty.clone().unbind(),
            _base: base.clone().unbind(),
            mask,
        },
    );
    Ok(mask)
}

/// Number of k-point override probes run so far (cache misses).
pub fn k_override_probes_run() -> usize {
    K_OVERRIDE_PROBES_RUN.load(Ordering::Relaxed)
}
