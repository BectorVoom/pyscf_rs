//! K-01 — reciprocal-lattice G-vectors on the device (PBC-MASTER-PLAN §6).
//!
//! One thread per grid point of the FFT mesh:
//!
//! ```text
//! Gv[x,y,z] = rx[x]*b[0] + ry[y]*b[1] + rz[z]*b[2]
//! ```
//!
//! The body is the nine-line inner loop of `pyscf/lib/pbc/cell.c:122-146`
//! transcribed verbatim, including its accumulation ORDER (`rx` term first,
//! then `+= ry`, then `+= rz`) — floating-point addition is not associative, so
//! reordering would move the last bits of every G-vector and therefore of every
//! planewave integral downstream.
//!
//! Upstream flattens `(mesh[0], mesh[1], mesh[2], 3)` C-order and the caller
//! reshapes to `(ngrids, 3)`; the flat grid index is
//! `g = x*my*mz + y*mz + z`, which this kernel inverts per thread rather than
//! launching a 3D grid (PBC-MASTER-PLAN §8.1 plan 09-05 step 2 mandates the 1D
//! `CubeCount::Static(ceil(ngrids/256), 1, 1)` / `CubeDim { x: 256, .. }`
//! geometry).
//!
//! Generic over the device float (`F: Float`, AGENTS.md §3 / RULE 5).

use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::{AlgebraClient, AlgebraError};

use crate::scalar::DeviceScalar;

/// `Gv[g, :] = rx[x]*b[0, :] + ry[y]*b[1, :] + rz[z]*b[2, :]`, one thread per `g`.
///
/// `b` is the row-major 3x3 reciprocal-lattice matrix flattened to nine
/// elements, matching the `double* b` upstream passes to `libpbc.get_Gv`.
/// The `g < ngrids` guard is required: the launch rounds the thread count up to
/// a whole number of cubes.
#[cube(launch_unchecked)]
fn gv_kernel<F: Float>(
    rx: &Array<F>,
    ry: &Array<F>,
    rz: &Array<F>,
    b: &Array<F>,
    gv: &mut Array<F>,
    mx: usize,
    my: usize,
    mz: usize,
) {
    let g = ABSOLUTE_POS;
    let ngrids = mx * my * mz;
    if g < ngrids {
        // Invert the C-order flattening of the (mx, my, mz) loop nest.
        let x = g / (my * mz);
        let y = (g / mz) % my;
        let z = g % mz;
        let p = g * 3;

        // cell.c:133-141, verbatim including the accumulation order.
        let mut v = rx[x] * b[0];
        v += ry[y] * b[3];
        v += rz[z] * b[6];
        gv[p] = v;

        let mut v1 = rx[x] * b[1];
        v1 += ry[y] * b[4];
        v1 += rz[z] * b[7];
        gv[p + 1] = v1;

        let mut v2 = rx[x] * b[2];
        v2 += ry[y] * b[5];
        v2 += rz[z] * b[8];
        gv[p + 2] = v2;
    }
}

/// Threads per cube (PBC-MASTER-PLAN §8.1 plan 09-05 step 2).
const BLOCK: u32 = 256;

/// Core launch on resident device handles. `R` stays generic so one body serves
/// every backend; the `Runtime` type never escapes into a public signature.
#[allow(clippy::too_many_arguments)]
fn launch_gv_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    rx: &Handle,
    ry: &Handle,
    rz: &Handle,
    b: &Handle,
    gv: &Handle,
    mx: usize,
    my: usize,
    mz: usize,
) {
    let ngrids = mx * my * mz;
    let groups = (ngrids as u32).div_ceil(BLOCK);
    unsafe {
        gv_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(BLOCK),
            // SAFETY: the handles hold exactly mx / my / mz / 9 / 3*ngrids
            // elements of `F`; the kernel guards `g < ngrids`.
            ArrayArg::from_raw_parts(rx.clone(), mx),
            ArrayArg::from_raw_parts(ry.clone(), my),
            ArrayArg::from_raw_parts(rz.clone(), mz),
            ArrayArg::from_raw_parts(b.clone(), 9),
            ArrayArg::from_raw_parts(gv.clone(), ngrids * 3),
            mx,
            my,
            mz,
        );
    }
}

/// Host-slice launcher: upload the three frequency axes and the reciprocal
/// matrix, allocate the output, run the kernel, read it back.
fn launch_gv<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    rx: &[F],
    ry: &[F],
    rz: &[F],
    b: &[F],
) -> Vec<F> {
    let (mx, my, mz) = (rx.len(), ry.len(), rz.len());
    let ngrids = mx * my * mz;
    let rx_h = client.create(Bytes::from_elems(rx.to_vec()));
    let ry_h = client.create(Bytes::from_elems(ry.to_vec()));
    let rz_h = client.create(Bytes::from_elems(rz.to_vec()));
    let b_h = client.create(Bytes::from_elems(b.to_vec()));
    let gv_h = client.empty(ngrids * 3 * core::mem::size_of::<F>());
    launch_gv_on_handles::<R, F>(client, &rx_h, &ry_h, &rz_h, &b_h, &gv_h, mx, my, mz);
    let bytes = client.read(vec![gv_h]);
    bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec()
}

/// K-01 public entry point: the `(ngrids, 3)` G-vector table, flattened
/// row-major (`out[g*3 + c]`).
///
/// `rx` / `ry` / `rz` are the per-axis integer frequencies
/// (`np.fft.fftfreq(n, 1./n)`); `b` is the row-major 3x3 reciprocal lattice
/// flattened to nine elements. `ngrids = rx.len() * ry.len() * rz.len()`.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] if `b` is not exactly nine elements.
/// An empty mesh (any axis of length 0) returns an empty vector without
/// launching.
pub fn gv(
    client: &AlgebraClient,
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    b: &[f64],
) -> Result<Vec<f64>, AlgebraError> {
    if b.len() != 9 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "reciprocal matrix b of 9 elements (row-major 3x3)".to_string(),
            actual: format!("{} elements", b.len()),
        });
    }
    if rx.is_empty() || ry.is_empty() || rz.is_empty() {
        return Ok(Vec::new());
    }
    let out = dispatch_backend!(client, c, Rt, launch_gv::<Rt, f64>(c, rx, ry, rz, b));
    Ok(out)
}
