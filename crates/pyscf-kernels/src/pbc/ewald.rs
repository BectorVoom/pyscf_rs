//! K-05 / K-06 — Ewald summation kernels (PBC-MASTER-PLAN §6, plan 09-08).
//!
//! Two device kernels back `pyscf_pbc_gto::ewald`:
//!
//! * **K-05** [`ewald_rlij`] — the real-space pair distances
//!   `r[L,i,j] = |R_i - R_j + L|` of `cell.py:729-732`:
//!   ```text
//!   rLij = coords[:,None,:] - coords[None,:,:] + Lall[:,None,None,:]
//!   r    = sqrt(einsum('Lijx,Lijx->Lij', rLij, rLij))
//!   ```
//!   Only the DISTANCES are computed on the device. `erfc` is evaluated on the
//!   host (PBC-MASTER-PLAN §8.1 plan 09-08 step 2 explicitly prefers this:
//!   there is no correctly-rounded `erfc` in the cubecl `Float` surface, and the
//!   1e-9 Ha acceptance gate needs the full double-precision function). The
//!   `r < 1e-16 -> 1e200` masking of `cell.py:733` also stays on the host so the
//!   sentinel is exact in `f64`.
//!
//! * **K-06** [`ewald_gs_terms`] — the per-G-vector reciprocal-space terms of
//!   `cell.py:753-770`:
//!   ```text
//!   absG2 = einsum('gi,gi->g', Gv, Gv);  absG2[absG2==0] = 1e200
//!   coulG = 4*pi / absG2 * weights
//!   term  = |ZSI[g]|^2 * exp(-absG2/(4*eta^2)) * coulG[g]
//!   ```
//!   The caller reduces `term` with `oracle_sum` and multiplies by `0.5`, so the
//!   summation ORDER stays deterministic on the host (§9.3) rather than
//!   depending on a device reduction tree.
//!
//! Both kernels are generic over the device float (`F: Float`, AGENTS.md §3 /
//! RULE 5) and launched through `dispatch_backend!`.
//!
//! # Why the scalars ride in an `Array<F>`
//!
//! `ScalarArg::new` is not part of the cubecl 0.10.0 public surface (see the
//! note in `crates/pyscf-kernels/src/eval_gto.rs`), and `F::new` only accepts an
//! `f32` literal — which would turn the `1e200` sentinel into `inf`. Both
//! kernels therefore take their scalars as a small `Array<F>` uploaded by the
//! host, exactly as `gv_kernel` already does for the 3x3 `b` matrix.

use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::{AlgebraClient, AlgebraError};

use crate::scalar::DeviceScalar;

/// Threads per cube. Same geometry as K-01 / K-02.
const BLOCK: u32 = 256;

// ---------------------------------------------------------------------------
// K-05 — real-space pair distances.
// ---------------------------------------------------------------------------

/// `r[t] = |coords[i] - coords[j] + ls[l]|` for `t = (l*natm + i)*natm + j`.
///
/// The flattening is C-order over `(L, i, j)`, matching upstream's
/// `rLij` array so the host reduction walks the same elements in the same
/// order. The `t < nl*natm*natm` guard is required: the launch rounds the
/// thread count up to a whole number of cubes.
#[cube(launch_unchecked)]
fn ewald_rlij_kernel<F: Float>(
    coords: &Array<F>,
    ls: &Array<F>,
    r: &mut Array<F>,
    natm: usize,
    nl: usize,
) {
    let t = ABSOLUTE_POS;
    if t < nl * natm * natm {
        // Invert the C-order flattening of the (L, i, j) loop nest.
        let l = t / (natm * natm);
        let i = (t / natm) % natm;
        let j = t % natm;
        let ic = i * 3;
        let jc = j * 3;
        let lc = l * 3;

        // cell.py:729 — rLij = coords[i] - coords[j] + L.
        let dx = coords[ic] - coords[jc] + ls[lc];
        let dy = coords[ic + 1] - coords[jc + 1] + ls[lc + 1];
        let dz = coords[ic + 2] - coords[jc + 2] + ls[lc + 2];

        // cell.py:730 — r = sqrt(einsum('Lijx,Lijx->Lij', rLij, rLij)).
        let mut r2 = dx * dx;
        r2 += dy * dy;
        r2 += dz * dz;
        r[t] = F::sqrt(r2);
    }
}

/// Core launch on resident device handles.
fn launch_ewald_rlij_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    coords: &Handle,
    ls: &Handle,
    r: &Handle,
    natm: usize,
    nl: usize,
) {
    let n = nl * natm * natm;
    let groups = (n as u32).div_ceil(BLOCK);
    unsafe {
        ewald_rlij_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(BLOCK),
            // SAFETY: coords holds 3*natm and ls 3*nl elements of `F`; the
            // output holds nl*natm*natm; the kernel guards `t < n`.
            ArrayArg::from_raw_parts(coords.clone(), natm * 3),
            ArrayArg::from_raw_parts(ls.clone(), nl * 3),
            ArrayArg::from_raw_parts(r.clone(), n),
            natm,
            nl,
        );
    }
}

/// Host-slice launcher for K-05.
fn launch_ewald_rlij<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    coords: &[F],
    ls: &[F],
    natm: usize,
    nl: usize,
) -> Vec<F> {
    let n = nl * natm * natm;
    let coords_h = client.create(Bytes::from_elems(coords.to_vec()));
    let ls_h = client.create(Bytes::from_elems(ls.to_vec()));
    let r_h = client.empty(n * core::mem::size_of::<F>());
    launch_ewald_rlij_on_handles::<R, F>(client, &coords_h, &ls_h, &r_h, natm, nl);
    let bytes = client.read(vec![r_h]);
    bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec()
}

/// K-05 public entry point: the `(nL, natm, natm)` C-order table of pair
/// distances `|R_i - R_j + L|` in Bohr.
///
/// `coords` is the flat `(natm, 3)` atom-coordinate table; `ls` is the flat
/// `(nL, 3)` lattice-translation table from
/// `pyscf_pbc_gto::lattice::get_lattice_ls`.
///
/// The caller still owes the `r < 1e-16 -> 1e200` mask and the
/// `erfc(eta*r)/r` evaluation — both stay on the host on purpose (see the
/// module docs).
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] if either buffer length is not a multiple of
/// 3. An empty atom list or lattice-image list returns an empty vector without
/// launching.
pub fn ewald_rlij(
    client: &AlgebraClient,
    coords: &[f64],
    ls: &[f64],
) -> Result<Vec<f64>, AlgebraError> {
    if !coords.len().is_multiple_of(3) || !ls.len().is_multiple_of(3) {
        return Err(AlgebraError::ShapeMismatch {
            expected: "coords and ls lengths to be multiples of 3".to_string(),
            actual: format!("coords {} ls {}", coords.len(), ls.len()),
        });
    }
    let natm = coords.len() / 3;
    let nl = ls.len() / 3;
    if natm == 0 || nl == 0 {
        return Ok(Vec::new());
    }
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        launch_ewald_rlij::<Rt, f64>(c, coords, ls, natm, nl)
    );
    Ok(out)
}

// ---------------------------------------------------------------------------
// K-06 — reciprocal-space terms.
// ---------------------------------------------------------------------------

/// `term[g] = |ZSI[g]|^2 * exp(-absG2/(4*eta^2)) * (4*pi/absG2) * weights`.
///
/// `params` carries the four host scalars, in order:
/// `[eta, weights, big, four_pi]`. `big` is upstream's `1e200` G=0 sentinel
/// (`cell.py:755`) and `four_pi` is `4*pi`. Both ride in the buffer rather than
/// being written as literals because `F::new` takes an `f32`: `1e200` would
/// become `inf` and `4*pi` would lose 29 bits of mantissa — enough on its own to
/// blow the 1e-9 Ha acceptance gate.
///
/// The `g < ngrids` guard is required: the launch rounds the thread count up to
/// a whole number of cubes.
#[cube(launch_unchecked)]
fn ewald_gs_kernel<F: Float>(
    gv: &Array<F>,
    zsi_re: &Array<F>,
    zsi_im: &Array<F>,
    params: &Array<F>,
    term: &mut Array<F>,
    ngrids: usize,
) {
    let g = ABSOLUTE_POS;
    if g < ngrids {
        let gc = g * 3;
        // cell.py:754 — absG2 = einsum('gi,gi->g', Gv, Gv).
        let mut absg2 = gv[gc] * gv[gc];
        absg2 += gv[gc + 1] * gv[gc + 1];
        absg2 += gv[gc + 2] * gv[gc + 2];
        // cell.py:755 — absG2[absG2 == 0] = 1e200.
        if absg2 == F::from_int(0) {
            absg2 = params[2];
        }
        let eta = params[0];
        let weights = params[1];
        // cell.py:756-757 — coulG = 4*pi/absG2; coulG *= weights.
        let coulg = params[3] / absg2 * weights;
        // cell.py:767 — ZexpG2 = ZSI * exp(-absG2/(4*eta^2)).
        let four_eta2 = F::from_int(4) * eta * eta;
        let expfac = F::exp(F::from_int(0) - absg2 / four_eta2);
        // cell.py:768 — real part of conj(ZSI) * ZexpG2 * coulG.
        let mut z2 = zsi_re[g] * zsi_re[g];
        z2 += zsi_im[g] * zsi_im[g];
        term[g] = z2 * expfac * coulg;
    }
}

/// Core launch on resident device handles.
#[allow(clippy::too_many_arguments)]
fn launch_ewald_gs_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    gv: &Handle,
    zsi_re: &Handle,
    zsi_im: &Handle,
    params: &Handle,
    term: &Handle,
    ngrids: usize,
) {
    let groups = (ngrids as u32).div_ceil(BLOCK);
    unsafe {
        ewald_gs_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(BLOCK),
            // SAFETY: gv holds 3*ngrids elements of `F`, both ZSI planes hold
            // ngrids, params holds 4, and the output holds ngrids; the kernel
            // guards `g < ngrids`.
            ArrayArg::from_raw_parts(gv.clone(), ngrids * 3),
            ArrayArg::from_raw_parts(zsi_re.clone(), ngrids),
            ArrayArg::from_raw_parts(zsi_im.clone(), ngrids),
            ArrayArg::from_raw_parts(params.clone(), 4),
            ArrayArg::from_raw_parts(term.clone(), ngrids),
            ngrids,
        );
    }
}

/// Host-slice launcher for K-06.
fn launch_ewald_gs<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    gv: &[F],
    zsi_re: &[F],
    zsi_im: &[F],
    params: &[F],
    ngrids: usize,
) -> Vec<F> {
    let gv_h = client.create(Bytes::from_elems(gv.to_vec()));
    let re_h = client.create(Bytes::from_elems(zsi_re.to_vec()));
    let im_h = client.create(Bytes::from_elems(zsi_im.to_vec()));
    let par_h = client.create(Bytes::from_elems(params.to_vec()));
    let term_h = client.empty(ngrids * core::mem::size_of::<F>());
    launch_ewald_gs_on_handles::<R, F>(client, &gv_h, &re_h, &im_h, &par_h, &term_h, ngrids);
    let bytes = client.read(vec![term_h]);
    bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec()
}

/// Upstream's G=0 sentinel (`cell.py:755`). Public so the host side and the
/// device side cannot drift apart.
pub const EWALD_G0_SENTINEL: f64 = 1e200;

/// K-06 public entry point: the per-G-vector reciprocal-space Ewald terms.
///
/// The caller finishes with `0.5 * oracle_sum(&terms)` — the reduction stays on
/// the host so it is bit-deterministic (§9.3).
///
/// * `gv` — flat `(ngrids, 3)` G-vector table;
/// * `zsi_re` / `zsi_im` — the charge-weighted structure factor
///   `ZSI[g] = sum_a q_a * SI[a, g]`, planar (D-PBC-02 / RULE 8);
/// * `eta` — the Ewald screening parameter `ew_eta`;
/// * `weights` — `get_Gv_weights`'s scalar `1/vol`.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] if `gv` is not `3 * ngrids` long or the two
/// `ZSI` planes disagree. An empty grid returns an empty vector.
pub fn ewald_gs_terms(
    client: &AlgebraClient,
    gv: &[f64],
    zsi_re: &[f64],
    zsi_im: &[f64],
    eta: f64,
    weights: f64,
) -> Result<Vec<f64>, AlgebraError> {
    if !gv.len().is_multiple_of(3) {
        return Err(AlgebraError::ShapeMismatch {
            expected: "gv length to be a multiple of 3".to_string(),
            actual: format!("gv {}", gv.len()),
        });
    }
    let ngrids = gv.len() / 3;
    if zsi_re.len() != ngrids || zsi_im.len() != ngrids {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("both ZSI planes to hold {ngrids} elements"),
            actual: format!("re {} im {}", zsi_re.len(), zsi_im.len()),
        });
    }
    if ngrids == 0 {
        return Ok(Vec::new());
    }
    let params = [eta, weights, EWALD_G0_SENTINEL, 4.0 * std::f64::consts::PI];
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        launch_ewald_gs::<Rt, f64>(c, gv, zsi_re, zsi_im, &params, ngrids)
    );
    Ok(out)
}
