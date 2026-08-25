//! K-02 — structure factors on the device (PBC-MASTER-PLAN §6).
//!
//! `SI[a, g] = exp(-i * Gv[g] . R_a)`, i.e. MH (3.34), for every atom `a` and
//! every G-vector `g`. Ports the `else` branch of
//! `pyscf/pbc/gto/cell.py:635-646` (`SI = np.exp(-1j*np.dot(coords, Gv.T))`).
//!
//! Layout is PLANAR (D-PBC-02 / RULE 8): two flat `Array<F>` outputs, never an
//! interleaved `[re, im, re, im, …]` buffer. `si_re` and `si_im` are row-major
//! `(natm, ngrids)`.
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

/// `theta = -(Gv[g] . R_a)`; `si_re = cos(theta)`, `si_im = sin(theta)`.
/// One thread per `(a, g)` pair, flattened as `i = a*ngrids + g`.
///
/// The `i < natm*ngrids` guard is required: the launch rounds the thread count
/// up to a whole number of cubes.
#[cube(launch_unchecked)]
fn struct_factor_kernel<F: Float>(
    coords: &Array<F>,
    gv: &Array<F>,
    si_re: &mut Array<F>,
    si_im: &mut Array<F>,
    natm: usize,
    ngrids: usize,
) {
    let i = ABSOLUTE_POS;
    if i < natm * ngrids {
        let a = i / ngrids;
        let g = i % ngrids;
        let ac = a * 3;
        let gc = g * 3;
        // Gv[g] . R_a, then negated for the exp(-i ...) convention.
        let mut rg = gv[gc] * coords[ac];
        rg += gv[gc + 1] * coords[ac + 1];
        rg += gv[gc + 2] * coords[ac + 2];
        let theta = F::from_int(0) - rg;
        si_re[i] = F::cos(theta);
        si_im[i] = F::sin(theta);
    }
}

/// Threads per cube.
const BLOCK: u32 = 256;

/// Core launch on resident device handles. `R` stays generic so one body serves
/// every backend; the `Runtime` type never escapes into a public signature.
#[allow(clippy::too_many_arguments)]
fn launch_struct_factor_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    coords: &Handle,
    gv: &Handle,
    si_re: &Handle,
    si_im: &Handle,
    natm: usize,
    ngrids: usize,
) {
    let n = natm * ngrids;
    let groups = (n as u32).div_ceil(BLOCK);
    unsafe {
        struct_factor_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(BLOCK),
            // SAFETY: coords holds 3*natm and gv 3*ngrids elements of `F`; the
            // two outputs hold natm*ngrids each; the kernel guards `i < n`.
            ArrayArg::from_raw_parts(coords.clone(), natm * 3),
            ArrayArg::from_raw_parts(gv.clone(), ngrids * 3),
            ArrayArg::from_raw_parts(si_re.clone(), n),
            ArrayArg::from_raw_parts(si_im.clone(), n),
            natm,
            ngrids,
        );
    }
}

/// Host-slice launcher: upload the coordinates and G-vectors, allocate the two
/// output planes, run the kernel, read both back in ONE batched `client.read`.
fn launch_struct_factor<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    coords: &[F],
    gv: &[F],
    natm: usize,
    ngrids: usize,
) -> (Vec<F>, Vec<F>) {
    let n = natm * ngrids;
    let coords_h = client.create(Bytes::from_elems(coords.to_vec()));
    let gv_h = client.create(Bytes::from_elems(gv.to_vec()));
    let re_h = client.empty(n * core::mem::size_of::<F>());
    let im_h = client.empty(n * core::mem::size_of::<F>());
    launch_struct_factor_on_handles::<R, F>(client, &coords_h, &gv_h, &re_h, &im_h, natm, ngrids);
    let bytes = client.read(vec![re_h, im_h]);
    (
        bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec(),
        bytemuck::cast_slice::<u8, F>(&bytes[1]).to_vec(),
    )
}

/// K-02 public entry point: `(si_re, si_im)`, each row-major `(natm, ngrids)`.
///
/// `coords` is the flat `(natm, 3)` atom-coordinate table in Bohr; `gv` is the
/// flat `(ngrids, 3)` G-vector table from [`crate::pbc::gv::gv`].
///
/// Returns the PLANAR split rather than a `CTensor` so that this crate does not
/// have to agree with `pyscf-algebra` about a host container type; the caller
/// wraps the pair (see `pyscf_pbc_gto::gv::get_si`).
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] if either buffer is not a multiple of 3
/// long. An empty atom list or mesh returns two empty vectors without
/// launching.
pub fn struct_factor(
    client: &AlgebraClient,
    coords: &[f64],
    gv: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    if !coords.len().is_multiple_of(3) || !gv.len().is_multiple_of(3) {
        return Err(AlgebraError::ShapeMismatch {
            expected: "coords and gv lengths to be multiples of 3".to_string(),
            actual: format!("coords {} gv {}", coords.len(), gv.len()),
        });
    }
    let natm = coords.len() / 3;
    let ngrids = gv.len() / 3;
    if natm == 0 || ngrids == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        launch_struct_factor::<Rt, f64>(c, coords, gv, natm, ngrids)
    );
    Ok(out)
}
