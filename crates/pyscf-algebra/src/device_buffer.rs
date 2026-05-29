//! Phase-2 device-buffer registry backing the opaque `Tensor`/`BufferId` surface.
//!
//! Phase 1 left every `Tensor`-based primitive (`axpy`, `scal`, `dot`,
//! `reduce_sum`, …) returning `NotYetImplemented { phase: 2 }` because a
//! `Tensor` only carried a *sentinel* `BufferId` (`BufferId::SENTINEL`, set by
//! `Tensor::placeholder`) with no backing storage — there was nowhere to read
//! the operand data from or write a result back to.
//!
//! This module is that backing store. It is a process-global registry mapping
//! each live `BufferId` to its host bytes + element count + dtype:
//!
//!   * [`upload`] parks a host slice, allocates a fresh `BufferId` (via
//!     `BufferId::from_raw`), and returns a `Tensor` referencing it.
//!   * [`download`] reads a tensor's bytes back to a host `Vec`.
//!   * [`release`] frees a tensor's buffer.
//!   * `write_back` (crate-private) overwrites an existing buffer in place — the
//!     mutation path the in-place ops (`axpy`, `scal`) and the reduction sink
//!     (`reduce_sum`'s `out`) use.
//!
//! The `Tensor`-based ops are thin wrappers: they [`download`] their operands,
//! run the SAME already-verified `*_dense` device launcher (so numeric
//! correctness is inherited, byte-for-byte, from the oracle-tested slice path),
//! and `write_back` the result.
//!
//! **Why a registry of host bytes (re-uploaded per op) rather than cached
//! device handles.** Storing a cubecl `Handle` would keep buffers device-resident
//! across ops (the eventual optimization), but a `Handle` is backend-bound and
//! lifetime-tied to the `ComputeClient` that created it — parking one in a
//! process-global outliving its client is unsound. Host-byte staging makes the
//! `Tensor` surface correct and usable end-to-end *today* by reusing the
//! verified `*_dense` kernels verbatim, and the public surface (`Tensor` in,
//! `Tensor` out) does not change when device-resident caching later lands.
//!
//! **Lifetime.** `Tensor` is `Clone`, so the registry cannot free on drop
//! without refcounting; buffers persist until explicitly [`release`]d (or
//! process exit). Callers that churn many transient tensors should `release`
//! them. The sentinel id is never allocated, so a `placeholder` tensor always
//! misses the lookup and yields `AlgebraError::UnallocatedBuffer`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use pyscf_runtime::DType;

use crate::error::AlgebraError;
use crate::scalar::{DeviceScalar, dtype_of};
use crate::tensor::{BufferId, Tensor};

/// One registry entry: the raw host bytes plus the element count and dtype tag
/// needed to reinterpret them. `len * dtype.size()` always equals `bytes.len()`.
struct StoredBuffer {
    bytes: Vec<u8>,
    len: usize,
    dtype: DType,
}

/// Process-global buffer table. `OnceLock<Mutex<…>>` so the map is lazily
/// created and `Send + Sync` without a runtime init step.
static REGISTRY: OnceLock<Mutex<HashMap<u64, StoredBuffer>>> = OnceLock::new();

/// Monotonic id source. Starts at 0 and only ever increments, so ids are never
/// reused and never collide with `BufferId::SENTINEL` (`u64::MAX`) in any
/// realistic run.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn registry() -> &'static Mutex<HashMap<u64, StoredBuffer>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Lock the registry, recovering from a poisoned mutex. A panic in one test
/// while holding the lock must not cascade into unrelated tests — the map's
/// invariant (`len * dtype.size() == bytes.len()` per entry) is upheld by the
/// `upload`/`write_back` writers, not by lock continuity, so recovering the
/// inner guard is safe.
fn lock_registry() -> std::sync::MutexGuard<'static, HashMap<u64, StoredBuffer>> {
    registry().lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Upload a host slice into the registry and return a `Tensor<F>` whose
/// `BufferId` references the stored bytes. The tensor's element count
/// (`shape.iter().product()`) must equal `data.len()`, otherwise
/// `DimensionMismatch` is returned and nothing is stored.
///
/// Generic over `F: DeviceScalar` (f32/f64); the stored dtype is derived from
/// `F` so it can never disagree with the tensor's compile-time element type.
pub fn upload<F: DeviceScalar>(data: &[F], shape: Vec<usize>) -> Result<Tensor<F>, AlgebraError> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(AlgebraError::DimensionMismatch {
            op: "device_buffer::upload",
            lhs: shape,
            rhs: vec![data.len()],
        });
    }

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let stored = StoredBuffer {
        bytes: bytemuck::cast_slice::<F, u8>(data).to_vec(),
        len: data.len(),
        dtype: dtype_of::<F>(),
    };
    lock_registry().insert(id, stored);
    Ok(Tensor::from_buffer(BufferId::from_raw(id), shape))
}

/// Read a tensor's buffer back to a host `Vec<F>`.
///
/// Errors with `UnallocatedBuffer` if the tensor's `BufferId` is not in the
/// registry (a `placeholder`, or already `release`d), and with `DtypeMismatch`
/// if `F` does not match the dtype the buffer was uploaded with.
pub fn download<F: DeviceScalar>(t: &Tensor<F>) -> Result<Vec<F>, AlgebraError> {
    read_raw::<F>(t.id.raw(), "device_buffer::download")
}

/// Free a tensor's buffer. Idempotent: releasing an unregistered id (placeholder
/// or already-released) is a no-op, so double-release never panics.
pub fn release<F: DeviceScalar>(t: &Tensor<F>) {
    lock_registry().remove(&t.id.raw());
}

/// Shared read path used by `download` and by every `Tensor` op that needs an
/// operand's host bytes. `op` names the caller for the error messages.
pub(crate) fn read_raw<F: DeviceScalar>(id: u64, op: &'static str) -> Result<Vec<F>, AlgebraError> {
    let guard = lock_registry();
    let stored = guard
        .get(&id)
        .ok_or(AlgebraError::UnallocatedBuffer { op, id })?;
    if stored.dtype != dtype_of::<F>() {
        return Err(AlgebraError::DtypeMismatch {
            op,
            lhs: stored.dtype,
            rhs: dtype_of::<F>(),
        });
    }
    Ok(bytemuck::cast_slice::<u8, F>(&stored.bytes).to_vec())
}

/// Overwrite an existing buffer's bytes in place (same `BufferId`). The new
/// element count must equal the old one — an op must not silently resize a
/// caller's tensor. Errors with `UnallocatedBuffer` if the id is unregistered.
pub(crate) fn write_back<F: DeviceScalar>(
    id: u64,
    data: &[F],
    op: &'static str,
) -> Result<(), AlgebraError> {
    let mut guard = lock_registry();
    let stored = guard
        .get_mut(&id)
        .ok_or(AlgebraError::UnallocatedBuffer { op, id })?;
    if stored.len != data.len() {
        return Err(AlgebraError::DimensionMismatch {
            op,
            lhs: vec![stored.len],
            rhs: vec![data.len()],
        });
    }
    stored.bytes = bytemuck::cast_slice::<F, u8>(data).to_vec();
    stored.dtype = dtype_of::<F>();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_download_roundtrips() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = upload::<f64>(&data, vec![2, 2]).expect("upload");
        assert_ne!(t.id.raw(), BufferId::SENTINEL, "must get a real id");
        assert_eq!(download::<f64>(&t).expect("download"), data);
        release::<f64>(&t);
        // After release the buffer is gone.
        assert!(matches!(
            download::<f64>(&t),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ));
    }

    #[test]
    fn placeholder_is_unallocated() {
        let t = Tensor::<f64>::placeholder(vec![3]);
        assert!(matches!(
            download::<f64>(&t),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ));
    }

    #[test]
    fn upload_rejects_shape_count_mismatch() {
        let data = vec![1.0_f64, 2.0, 3.0];
        assert!(matches!(
            upload::<f64>(&data, vec![2, 2]),
            Err(AlgebraError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn write_back_rejects_length_change() {
        let t = upload::<f64>(&[1.0_f64, 2.0], vec![2]).expect("upload");
        assert!(matches!(
            write_back::<f64>(t.id.raw(), &[1.0_f64, 2.0, 3.0], "test"),
            Err(AlgebraError::DimensionMismatch { .. })
        ));
        release::<f64>(&t);
    }
}
