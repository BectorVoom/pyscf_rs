//! K-04 — element-wise complex multiply on the device (PBC-MASTER-PLAN §6).
//!
//! `c[i] = a[i] * b[i]` for planar-complex operands: the ONLY complex operation
//! in the §5.2 table that cannot be expressed in the existing real primitives
//! (there is no real element-wise-multiply engine in `pyscf-algebra`), so it
//! gets its own cubecl kernel here, inside the ALG-06 wall.
//!
//! Layout is PLANAR (D-PBC-02 / RULE 8): six flat `Array<F>` operands, never an
//! interleaved `[re, im, re, im, …]` buffer. The planar split is what lets every
//! other complex op reduce to real `gemm`/`axpy`/`dot` calls, and mixing layouts
//! per-op would force a repack at every boundary.
//!
//! The kernel is generic over the device float (`F: Float`, AGENTS.md §3 /
//! RULE 5), so it monomorphizes for f32 (GPU speed path) and f64 (chemistry
//! precision path) alike. The arithmetic is the schoolbook 4-multiply form
//!
//! ```text
//! cr[i] = ar[i]*br[i] - ai[i]*bi[i]
//! ci[i] = ar[i]*bi[i] + ai[i]*br[i]
//! ```
//!
//! matching D-PBC-03's mandate for `zgemm_dense`: exact cancellation when one
//! plane is zero, and no three-term Karatsuba rounding.

use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::{AlgebraClient, AlgebraError};

use crate::scalar::DeviceScalar;

/// Element-wise complex multiply, planar layout, one thread per element.
///
/// The `i < n` guard is required: the launch rounds the thread count up to a
/// whole number of cubes, so tail threads must not write out of range.
#[cube(launch_unchecked)]
fn zhadamard_kernel<F: Float>(
    ar: &Array<F>,
    ai: &Array<F>,
    br: &Array<F>,
    bi: &Array<F>,
    cr: &mut Array<F>,
    ci: &mut Array<F>,
    n: usize,
) {
    let i = ABSOLUTE_POS;
    if i < n {
        cr[i] = ar[i] * br[i] - ai[i] * bi[i];
        ci[i] = ar[i] * bi[i] + ai[i] * br[i];
    }
}

/// Threads per cube for the zhadamard launch.
const BLOCK: u32 = 256;

/// Core launch on resident device handles. `R` stays generic so one body serves
/// every backend; the `Runtime` type never escapes into a public signature.
#[allow(clippy::too_many_arguments)]
fn launch_zhadamard_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    ar: &Handle,
    ai: &Handle,
    br: &Handle,
    bi: &Handle,
    cr: &Handle,
    ci: &Handle,
    n: usize,
) {
    let groups = (n as u32).div_ceil(BLOCK);
    unsafe {
        zhadamard_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(BLOCK),
            // SAFETY: every handle is `n` elements of `F`; the kernel guards `i < n`.
            ArrayArg::from_raw_parts(ar.clone(), n),
            ArrayArg::from_raw_parts(ai.clone(), n),
            ArrayArg::from_raw_parts(br.clone(), n),
            ArrayArg::from_raw_parts(bi.clone(), n),
            ArrayArg::from_raw_parts(cr.clone(), n),
            ArrayArg::from_raw_parts(ci.clone(), n),
            // Scalar dimension arg is passed as a bare value (LaunchArg for usize).
            n,
        );
    }
}

/// Host-slice launcher: upload the four input planes, allocate the two output
/// planes, run the kernel, read both back in ONE batched `client.read`.
fn launch_zhadamard<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    ar: &[F],
    ai: &[F],
    br: &[F],
    bi: &[F],
) -> (Vec<F>, Vec<F>) {
    let n = ar.len();
    let ar_h = client.create(Bytes::from_elems(ar.to_vec()));
    let ai_h = client.create(Bytes::from_elems(ai.to_vec()));
    let br_h = client.create(Bytes::from_elems(br.to_vec()));
    let bi_h = client.create(Bytes::from_elems(bi.to_vec()));
    let cr_h = client.empty(core::mem::size_of_val(ar));
    let ci_h = client.empty(core::mem::size_of_val(ar));
    launch_zhadamard_on_handles::<R, F>(client, &ar_h, &ai_h, &br_h, &bi_h, &cr_h, &ci_h, n);
    let bytes = client.read(vec![cr_h, ci_h]);
    (
        bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec(),
        bytemuck::cast_slice::<u8, F>(&bytes[1]).to_vec(),
    )
}

/// K-04 public entry point: element-wise complex multiply over planar operands.
///
/// Takes and returns bare planes rather than `pyscf_algebra::CTensor` so that
/// `pyscf-algebra` itself can host the mirrored in-crate copy of this kernel
/// (it cannot call upward into `pyscf-kernels` — that would be a dependency
/// cycle). PBC method crates, which depend on `pyscf-kernels`, use THIS entry
/// point; see `pyscf_algebra::zblas::zhadamard_dense` for the in-wall sibling.
///
/// All four input planes must have the same length; the two returned planes have
/// that same length. Empty input returns two empty vectors without launching.
pub fn zhadamard(
    client: &AlgebraClient,
    ar: &[f64],
    ai: &[f64],
    br: &[f64],
    bi: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    let n = ar.len();
    if ai.len() != n || br.len() != n || bi.len() != n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("all four planes of length {n}"),
            actual: format!("ai {} br {} bi {}", ai.len(), br.len(), bi.len()),
        });
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        launch_zhadamard::<Rt, f64>(c, ar, ai, br, bi)
    );
    Ok(out)
}
