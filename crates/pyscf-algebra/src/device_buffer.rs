//! Phase-2 device-buffer registry backing the opaque `Tensor`/`BufferId` surface.
//!
//! Phase 1 left every `Tensor`-based primitive (`axpy`, `scal`, `dot`,
//! `reduce_sum`, gemm/gemv/transpose, …) returning `NotYetImplemented { phase: 2 }`
//! because a `Tensor` only carried a *sentinel* `BufferId` (`BufferId::SENTINEL`,
//! set by `Tensor::placeholder`) with no backing storage.
//!
//! This module is that backing store. It is a process-global registry mapping
//! each live `BufferId` to a **resident cubecl device handle** plus its element
//! count, dtype, and owning backend:
//!
//!   * [`upload`] copies a host slice onto the device (via the active client's
//!     allocator), allocates a fresh `BufferId`, and returns a `Tensor`.
//!   * [`download`] reads a tensor's device buffer back to a host `Vec`.
//!   * [`release`] drops the handle (freeing the device allocation).
//!   * `handle_of` (crate-private) hands an op a clone of an operand's resident
//!     handle, checked for existence, backend, and dtype.
//!
//! **Device residency (quick-260529-mtx-2): no per-op re-upload.** Earlier this
//! registry stored host *bytes* and every `Tensor` op round-tripped
//! host→device→host through a `*_dense` call. Now the bytes live on the device
//! between ops: an op clones the operand handles and launches the kernel
//! directly on them. cubecl handle clones share one memory binding, so an
//! in-place kernel (axpy/scal) mutates the resident buffer the registry still
//! holds — no write-back — and an output-producing op (reduce/gemm/gemv/
//! transpose) writes straight into the caller's pre-allocated `out` handle. Data
//! is copied to the host only on an explicit [`download`]. A chain like
//! `axpy → scal → dot` therefore uploads each operand once, not once per op.
//!
//! **Backend binding.** A device handle is valid only with the `ComputeClient`
//! (backend) that created it, so each entry records its [`BackendKind`]; any op
//! or `download` given a client for a different backend fails with
//! `AlgebraError::BackendMismatch` rather than dereferencing a foreign handle.
//!
//! **Lifetime.** `Tensor` is `Clone` and holds only a `BufferId`, so the
//! registry cannot free on drop without refcounting; buffers persist until
//! explicitly [`release`]d (or process exit). The sentinel id is never
//! allocated, so a `placeholder` tensor always misses the lookup and yields
//! `AlgebraError::UnallocatedBuffer`. Handles must not outlive their backend's
//! client; in practice one backend is selected for the program's lifetime.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use pyscf_runtime::{BackendKind, DType};

use crate::client::AlgebraClient;
use crate::error::AlgebraError;
use crate::scalar::{DeviceScalar, dtype_of};
use crate::tensor::{BufferId, Tensor};

/// One registry entry: the resident device handle plus the metadata needed to
/// reinterpret and validate it. `len` is the element count; `dtype` and
/// `backend` guard against type- and backend-confused reuse.
struct StoredBuffer {
    handle: Handle,
    len: usize,
    dtype: DType,
    backend: BackendKind,
}

/// A clone of a resident handle handed to an op, with its element count. The
/// clone shares the underlying device memory with the registry's stored handle,
/// so an in-place kernel launched on it mutates the buffer the registry holds.
pub(crate) struct BufferRef {
    pub handle: Handle,
    pub len: usize,
}

/// Process-global buffer table. `OnceLock<Mutex<…>>` so the map is lazily
/// created and `Send + Sync` (cubecl `Handle` is `Send + Sync`) without a
/// runtime init step.
static REGISTRY: OnceLock<Mutex<HashMap<u64, StoredBuffer>>> = OnceLock::new();

/// Monotonic id source. Starts at 0 and only ever increments, so ids are never
/// reused and never collide with `BufferId::SENTINEL` (`u64::MAX`).
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn registry() -> &'static Mutex<HashMap<u64, StoredBuffer>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Lock the registry, recovering from a poisoned mutex. A panic in one test
/// while holding the lock must not cascade into unrelated tests.
fn lock_registry() -> std::sync::MutexGuard<'static, HashMap<u64, StoredBuffer>> {
    registry().lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Allocate a device handle on `client`'s backend holding `data`. The cubecl
/// `Handle` type is runtime-agnostic, so a single return type serves every arm;
/// the `Runtime` generic stays inside this match (ALG-06 wall).
fn create_handle<F: DeviceScalar>(client: &AlgebraClient, data: &[F]) -> Handle {
    match client {
        AlgebraClient::Cpu(c) => c.create(Bytes::from_elems(data.to_vec())),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => c.create(Bytes::from_elems(data.to_vec())),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => c.create(Bytes::from_elems(data.to_vec())),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => c.create(Bytes::from_elems(data.to_vec())),
    }
}

/// Read a resident handle back to host bytes on `client`'s backend.
fn read_handle(client: &AlgebraClient, handle: &Handle) -> Vec<u8> {
    let mut bytes = match client {
        AlgebraClient::Cpu(c) => c.read(vec![handle.clone()]),
        #[cfg(feature = "cuda")]
        AlgebraClient::Cuda(c) => c.read(vec![handle.clone()]),
        #[cfg(feature = "wgpu")]
        AlgebraClient::Wgpu(c) => c.read(vec![handle.clone()]),
        #[cfg(feature = "rocm")]
        AlgebraClient::Rocm(c) => c.read(vec![handle.clone()]),
    };
    bytes.remove(0).to_vec()
}

/// Upload a host slice onto `client`'s device and return a `Tensor<F>` whose
/// `BufferId` references the resident handle. The tensor's element count
/// (`shape.iter().product()`) must equal `data.len()`, otherwise
/// `DimensionMismatch` is returned and nothing is stored.
///
/// Generic over `F: DeviceScalar` (f32/f64); the stored dtype is derived from
/// `F` and the backend from `client`, so neither can disagree with the buffer.
pub fn upload<F: DeviceScalar>(
    client: &AlgebraClient,
    data: &[F],
    shape: Vec<usize>,
) -> Result<Tensor<F>, AlgebraError> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(AlgebraError::DimensionMismatch {
            op: "device_buffer::upload",
            lhs: shape,
            rhs: vec![data.len()],
        });
    }

    let handle = create_handle::<F>(client, data);
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    lock_registry().insert(
        id,
        StoredBuffer {
            handle,
            len: data.len(),
            dtype: dtype_of::<F>(),
            backend: client.kind(),
        },
    );
    Ok(Tensor::from_buffer(BufferId::from_raw(id), shape))
}

/// Read a tensor's resident device buffer back to a host `Vec<F>`.
///
/// Errors with `UnallocatedBuffer` if the tensor's `BufferId` is not in the
/// registry (a `placeholder`, or already `release`d), `BackendMismatch` if
/// `client`'s backend differs from the one the buffer is resident on, and
/// `DtypeMismatch` if `F` does not match the upload dtype.
pub fn download<F: DeviceScalar>(
    client: &AlgebraClient,
    t: &Tensor<F>,
) -> Result<Vec<F>, AlgebraError> {
    let bref = handle_of::<F>(t.id.raw(), client, "device_buffer::download")?;
    let bytes = read_handle(client, &bref.handle);
    Ok(bytemuck::cast_slice::<u8, F>(&bytes).to_vec())
}

/// Free a tensor's resident device buffer (drops the handle → cubecl frees the
/// allocation when the last clone is gone). Idempotent: releasing an
/// unregistered id (placeholder or already-released) is a no-op.
pub fn release<F: DeviceScalar>(t: &Tensor<F>) {
    lock_registry().remove(&t.id.raw());
}

/// Hand an op a clone of an operand's resident handle (sharing device memory),
/// checked for existence, backend, and dtype. `op` names the caller for errors.
pub(crate) fn handle_of<F: DeviceScalar>(
    id: u64,
    client: &AlgebraClient,
    op: &'static str,
) -> Result<BufferRef, AlgebraError> {
    let guard = lock_registry();
    let stored = guard
        .get(&id)
        .ok_or(AlgebraError::UnallocatedBuffer { op, id })?;
    if stored.backend != client.kind() {
        return Err(AlgebraError::BackendMismatch {
            op,
            expected: client.kind().name(),
            actual: stored.backend.name(),
        });
    }
    if stored.dtype != dtype_of::<F>() {
        return Err(AlgebraError::DtypeMismatch {
            op,
            lhs: stored.dtype,
            rhs: dtype_of::<F>(),
        });
    }
    Ok(BufferRef {
        handle: stored.handle.clone(),
        len: stored.len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::Runtime; // `CpuRuntime::client` needs the trait in scope (tests only).

    fn cpu() -> AlgebraClient {
        AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
    }

    #[test]
    fn upload_download_roundtrips() {
        let client = cpu();
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = upload::<f64>(&client, &data, vec![2, 2]).expect("upload");
        assert_ne!(t.id.raw(), BufferId::SENTINEL, "must get a real id");
        assert_eq!(download::<f64>(&client, &t).expect("download"), data);
        release::<f64>(&t);
        // After release the buffer is gone.
        assert!(matches!(
            download::<f64>(&client, &t),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ));
    }

    #[test]
    fn placeholder_is_unallocated() {
        let client = cpu();
        let t = Tensor::<f64>::placeholder(vec![3]);
        assert!(matches!(
            download::<f64>(&client, &t),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ));
    }

    #[test]
    fn upload_rejects_shape_count_mismatch() {
        let client = cpu();
        let data = vec![1.0_f64, 2.0, 3.0];
        assert!(matches!(
            upload::<f64>(&client, &data, vec![2, 2]),
            Err(AlgebraError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn handle_of_rejects_dtype_mismatch() {
        let client = cpu();
        let t = upload::<f64>(&client, &[1.0_f64, 2.0], vec![2]).expect("upload");
        // Same buffer id, read as f32 → dtype guard fires.
        assert!(matches!(
            handle_of::<f32>(t.id.raw(), &client, "test"),
            Err(AlgebraError::DtypeMismatch { .. })
        ));
        release::<f64>(&t);
    }
}
