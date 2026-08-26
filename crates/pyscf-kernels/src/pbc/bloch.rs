//! K-07 — Bloch phase table on the device (PBC-MASTER-PLAN §6).
//!
//! `expkL[k, L] = exp(+i * kpts[k] . Ls[L])`, the factor that turns a lattice
//! sum of molecular shell-pair blocks into a k-resolved periodic matrix
//! (D-PBC-07):
//!
//! ```text
//! S^k_{ij} = Σ_L expkL[k, L] * <φ_i(r − R_i) | O | φ_j(r − R_j − L)>
//! ```
//!
//! # Sign convention
//!
//! `+i`, matching upstream `cell.py:224`
//! (`expkL = np.exp(1j*np.dot(kpts_lst, Ls.T))`) and the C driver, which shifts
//! the KET atom by `+L` (`fill_ints.c:1281-1286,1371`). The opposite sign
//! yields a matrix that is still Hermitian and still positive definite — it is
//! simply the wrong k-point, which is why the convention is pinned here rather
//! than left to each call site.
//!
//! Layout is PLANAR (D-PBC-02 / RULE 8): two flat `Array<F>` outputs, both
//! row-major `(nkpts, nimgs)`. This is deliberately the SAME shape upstream
//! feeds to its two `dgemm_` calls (`fill_ints.c:1382-1385`), so the contraction
//! step is a pair of ordinary real GEMMs over the existing `pyscf-algebra`
//! primitive and no complex device type is introduced.
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

/// `theta = kpts[k] . Ls[l]`; `re = cos(theta)`, `im = sin(theta)`.
/// One thread per `(k, l)` pair, flattened as `i = k*nimgs + l`.
///
/// The `i < nkpts*nimgs` guard is required: the launch rounds the thread count
/// up to a whole number of cubes, so tail threads must not write out of range.
#[cube(launch_unchecked)]
fn bloch_phase_kernel<F: Float>(
    kpts: &Array<F>,
    ls: &Array<F>,
    re: &mut Array<F>,
    im: &mut Array<F>,
    nkpts: usize,
    nimgs: usize,
) {
    let i = ABSOLUTE_POS;
    if i < nkpts * nimgs {
        let k = i / nimgs;
        let l = i % nimgs;
        let kc = k * 3;
        let lc = l * 3;
        let mut theta = kpts[kc] * ls[lc];
        theta += kpts[kc + 1] * ls[lc + 1];
        theta += kpts[kc + 2] * ls[lc + 2];
        re[i] = F::cos(theta);
        im[i] = F::sin(theta);
    }
}

/// Threads per cube for the bloch-phase launch.
const BLOCK: u32 = 256;

/// Core launch on resident device handles. `R` stays generic so one body serves
/// every backend; the `Runtime` type never escapes into a public signature.
#[allow(clippy::too_many_arguments)]
fn launch_bloch_phase_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    kpts: &Handle,
    ls: &Handle,
    re: &Handle,
    im: &Handle,
    nkpts: usize,
    nimgs: usize,
) {
    let n = nkpts * nimgs;
    let groups = (n as u32).div_ceil(BLOCK);
    unsafe {
        bloch_phase_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(BLOCK),
            // SAFETY: kpts holds 3*nkpts and ls 3*nimgs elements of `F`; the two
            // outputs hold nkpts*nimgs each; the kernel guards `i < n`.
            ArrayArg::from_raw_parts(kpts.clone(), nkpts * 3),
            ArrayArg::from_raw_parts(ls.clone(), nimgs * 3),
            ArrayArg::from_raw_parts(re.clone(), n),
            ArrayArg::from_raw_parts(im.clone(), n),
            nkpts,
            nimgs,
        );
    }
}

/// Host-slice launcher: upload k-points and lattice vectors, allocate the two
/// output planes, run the kernel, read both back in ONE batched `client.read`.
fn launch_bloch_phase<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    kpts: &[F],
    ls: &[F],
    nkpts: usize,
    nimgs: usize,
) -> (Vec<F>, Vec<F>) {
    let n = nkpts * nimgs;
    let kpts_h = client.create(Bytes::from_elems(kpts.to_vec()));
    let ls_h = client.create(Bytes::from_elems(ls.to_vec()));
    let re_h = client.empty(n * core::mem::size_of::<F>());
    let im_h = client.empty(n * core::mem::size_of::<F>());
    launch_bloch_phase_on_handles::<R, F>(client, &kpts_h, &ls_h, &re_h, &im_h, nkpts, nimgs);
    let bytes = client.read(vec![re_h, im_h]);
    (
        bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec(),
        bytemuck::cast_slice::<u8, F>(&bytes[1]).to_vec(),
    )
}

/// K-07 public entry point: `(re, im)`, each row-major `(nkpts, nimgs)`.
///
/// `kpts` is the flat `(nkpts, 3)` absolute k-point table in 1/Bohr; `ls` is the
/// flat `(nimgs, 3)` lattice-translation table in Bohr, as returned by
/// `pyscf_pbc_gto::lattice::get_lattice_ls`.
///
/// Returns the PLANAR split rather than a `CTensor` for the same reason
/// [`crate::pbc::struct_factor`] does: this crate does not have to agree with
/// `pyscf-algebra` about a host container type, and the two planes are what the
/// two real GEMMs of the contraction step consume directly.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] if either buffer is not a multiple of 3 long.
/// An empty k-point list or image list returns two empty vectors without
/// launching.
pub fn bloch_phase(
    client: &AlgebraClient,
    kpts: &[f64],
    ls: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    if !kpts.len().is_multiple_of(3) || !ls.len().is_multiple_of(3) {
        return Err(AlgebraError::ShapeMismatch {
            expected: "kpts and Ls lengths to be multiples of 3".to_string(),
            actual: format!("kpts {} Ls {}", kpts.len(), ls.len()),
        });
    }
    let nkpts = kpts.len() / 3;
    let nimgs = ls.len() / 3;
    if nkpts == 0 || nimgs == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        launch_bloch_phase::<Rt, f64>(c, kpts, ls, nkpts, nimgs)
    );
    Ok(out)
}
