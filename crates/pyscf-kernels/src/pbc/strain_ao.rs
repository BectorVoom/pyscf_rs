//! Plan 18-11 — the strain-tensor AO derivative collocation kernel.
//!
//! # Upstream
//!
//! All four stress modules obtain their AO strain derivatives from
//! `cell.pbc_eval_gto('GTOval_sph_deriv%d_strain_tensor' % deriv, ...)`
//! (`pyscf/pbc/grad/rks_stress.py:127-147`), which resolves to
//! `pyscf/lib/pbc/grid_ao.c:431`, `PBCeval_cart_for_strain_tensor_iter`
//! (sph twin: `PBCeval_sph_for_strain_tensor_iter`). The output shape is
//! (`grid_ao.c:442`, `rks_stress.py:130-131`, `143-146`)
//!
//! ```text
//! (nkpts, 3, 3, comp, ngrids, nao), comp = (deriv+1)(deriv+2)(deriv+3)/6
//! ```
//!
//! so `comp = 1` at `deriv = 0` and `comp = 4` at `deriv = 1`, with the `3x3`
//! axis the strain tensor `eps_{xy}`. Both `sph` and `cart` are shipped:
//! upstream tests both separately (`test_rks_stress.py:125`
//! `test_eval_ao_cart`, `:142` `test_eval_ao_sph`, `:160`
//! `test_eval_ao_deriv1_cart`, `:178` `test_eval_ao_deriv1_sph`).
//!
//! # What the kernel computes
//!
//! Per lattice image `L`, per shell on atom `A` at `ri`, with
//! `R = -(L + ri)` (`grid_ao.c`, `Rx/Ry/Rz` assignment) and the Cartesian AO
//! derivative table `buf` (`[D, ngrids, nao]`, `D = 4` value+gradient at
//! `deriv = 0`, `D = 10` value+gradient+hessian at `deriv = 1`):
//!
//! * `deriv = 0` (`ncomp = 1`, 9 strain blocks `b = x*3+y`):
//!   `strain[x,y] = dphi/dx_x * R_y`.
//! * `deriv = 1` (`ncomp = 4`, 36 strain blocks `b = (x*3+y)*4+c`):
//!   `c = 0` is the value's strain (`grad * R`, as at `deriv = 0`) and
//!   `c = 1,2,3` are the x/y/z-gradient strains (`hess_row * R`), with the
//!   hessian source rows `xx,xy,xz / xy,yy,yz / xz,yz,zz` per the `case 4`
//!   switch. [`strain_block_map`] is that switch transcribed entry by entry.
//!
//! The per-image strain blocks are Bloch-summed with `exp(+i k.L)`
//! (`eval_gto.py:139`, same sign as K-07) over the caller's image list.
//! The grid-response addition (`rks_stress.py:215-216`,
//! `ao_strain[:,:,0] += einsum('xig,yg->xyig', ao[1:4], coordsT)`, ...) is
//! deliberately NOT part of this kernel: it is the response of the grid
//! points to the strain, not of the basis function, and shipping it inside
//! would double-count it for any caller that does its own grid response
//! (18-12). Task 2.
//!
//! # Why this kernel is not redundant (18-REVIEW Part I §4.3, closed by Part II §9)
//!
//! The *grid* response is exactly the `ao_deriv (x) coords` contraction, but
//! the *basis* response carries the per-image weight `(R_A + L)_y` INSIDE the
//! lattice sum, and `pbc_eval_gto` returns a table that has already summed
//! over images — the weight cannot be recovered from it. That is why upstream
//! ships a dedicated C kernel here (and an ordinary integral for the overlap,
//! where the image sum CAN be re-run weighted; D-PBC-31 clause 2).
//!
//! # Fusion decision (D-PBC-30 clause 5): DO NOT FUSE
//!
//! Upstream evaluates the AO table twice over the same grid block (`block_loop`
//! at `deriv+1` plus `_eval_ao_strain_derivatives`). Plan 18-17 measured the
//! split on one GGA grid block (`blk = 97608`;
//! `measurements/sizings.md` Task 3): strain call **22414 ms**, `block_loop`
//! AO eval **10148 ms**, dtype-matched no-op alloc+write of the same
//! `(8,3,3,4,blk,8)` buffer **839 ms** — the arithmetic share of the strain
//! call is **21575 ms (96 %)** and traffic ≈ **4 %**. The repo's prior
//! ("buffer traffic dominates collocation kernels") is refuted for this
//! shape: fusing would save at most one ~0.8 s traffic pass over a ~32 s
//! combined cost. **18-11 ships the separate strain kernel.** (A fusion case
//! built on shared-exponential savings is not measured here.)
//!
//! `measurements/parseval.md` (18-16 Task 1.2) concerns the 18-05 fused-`hcore`
//! G-space contraction, not this kernel; it sets no number here.
//!
//! # Gate
//!
//! Gate A tier **A1 = 1e-9** (`measurements/gate-a-tiers.md`; upstream
//! `test_rks_stress.py:140` cart, `:158` sph, `:176`/`:194` deriv1,
//! `:214`/`:240` grid response): nine `(x,y)` components, each checked,
//! at `deriv = 0` and `1`, sph and cart, against the central difference of
//! the ordinary AO over `_finite_diff_cells` at `disp = 1e-5`.
//!
//! # Layering and precision
//!
//! ALG-06 corollary (17-CONTEXT): the collocation kernel lives in
//! `pyscf-kernels`, NOT in `pbc-grad`/`pbc-dft`; `pyscf-pbc-gto` drives it
//! through this module's public surface without naming cubecl. The device
//! kernel is generic over the device float (`F: Float`, AGENTS.md §3; CubeCL
//! manual `Cubecl_generics.md`); the public driver is f64-only, like `gv`.
//! The scale kernel is a single multiply per lane — FMA-free by construction
//! (`check-no-fma` has nothing to fuse). Host accumulation is sequential
//! (no rayon), so output is deterministic across thread counts.
//!
//! # Screening
//!
//! Per `(image, shell)`, the shell is skipped when the squared distance from
//! its image-shifted centre to the grid bounding box exceeds its `rcut^2`
//! (the box contains every grid point, so the box distance lower-bounds the
//! true min-distance: sound, may keep extra, never drops needed). The radii
//! are the caller's per-shell `rcut` (the wrapper passes
//! `estimate_rcut_for_eval(cell, deriv+1)` — one order above the output, so
//! conservative). Dropped mass is bounded by the same `precision` that sized
//! the image list; `PYSCF_PBC_STRAIN_SCREEN=0` (read per call by the
//! `pyscf-pbc-gto` wrapper) disables it for bisection.

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::launch_1d;
use pyscf_algebra::{AlgebraClient, AlgebraError};
use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP,
};

use crate::scalar::DeviceScalar;

// ---------------------------------------------------------------------------
// Block maps: the grid_ao.c strain switch, transcribed.
// ---------------------------------------------------------------------------

/// Number of AO-derivative components at strain order `deriv`:
/// `comp = (deriv+1)(deriv+2)(deriv+3)/6` (`rks_stress.py:133`).
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] — the named refusal for `deriv >= 2`,
/// which has no upstream caller and no oracle (18-11 Task 4).
pub fn strain_comp(deriv: u32) -> Result<usize, AlgebraError> {
    match deriv {
        0 => Ok(1),
        1 => Ok(4),
        other => Err(AlgebraError::ShapeMismatch {
            expected: "strain_tensor deriv 0 or 1 (the only orders with an upstream caller)".into(),
            actual: format!(
                "deriv={other} has no upstream caller and no oracle; refusing rather than falling through (18-11 Task 4)"
            ),
        }),
    }
}

/// Number of strain blocks: `9 * comp` (`grid_ao.c:442`,
/// `comp_strain_tensor = ncomp * 3 * 3`).
///
/// # Errors
/// As [`strain_comp`].
pub fn strain_nblocks(deriv: u32) -> Result<usize, AlgebraError> {
    Ok(9 * strain_comp(deriv)?)
}

/// (`src_of`, `raxis_of`): for strain block `b`, which derivative-table
/// component to take and which `R` axis to multiply by.
///
/// * `deriv = 0`: 9 blocks, `b = x*3+y`, `src = 1+x` (the gradient), axis `y`.
/// * `deriv = 1`: 36 blocks, `b = (x*3+y)*4+c`; `c = 0` reads the gradient
///   (`1+x`), `c = 1` reads `xx,xy,xz` (`4+x`), `c = 2` reads `xy,yy,yz`
///   (`[5,7,8][x]`), `c = 3` reads `xz,yz,zz` (`[6,8,9][x]`); axis is `y` in
///   all cases. This is `grid_ao.c`'s `case 1` / `case 4` switch, entry by
///   entry (derivative-table order is value, gx, gy, gz, hxx, hxy, hxz, hyy,
///   hyz, hzz).
///
/// # Errors
/// As [`strain_comp`].
pub fn strain_block_map(deriv: u32) -> Result<(Vec<u32>, Vec<u32>), AlgebraError> {
    let comp = strain_comp(deriv)?;
    let mut src_of = Vec::with_capacity(9 * comp);
    let mut raxis_of = Vec::with_capacity(9 * comp);
    for x in 0..3usize {
        for y in 0..3usize {
            if comp == 1 {
                src_of.push((1 + x) as u32);
                raxis_of.push(y as u32);
            } else {
                // c = 0..4 in order: value-strain, x/y/z-gradient strain.
                src_of.push((1 + x) as u32);
                src_of.push((4 + x) as u32);
                src_of.push([5u32, 7, 8][x]);
                src_of.push([6u32, 8, 9][x]);
                for _ in 0..4 {
                    raxis_of.push(y as u32);
                }
            }
        }
    }
    Ok((src_of, raxis_of))
}

// ---------------------------------------------------------------------------
// Device kernel: deriv-block x R -> strain-block.
// ---------------------------------------------------------------------------

/// `out[b*n + q] = deriv[src_of[b]*n + q] * rvec[a*3 + raxis_of[b]]`,
/// one lane per strain element.
///
/// `n = ngrids*nao`, `q = g + a*ngrids` (F-order, grid fastest — the same
/// per-component layout the `eval_gto` family uses). `src_of`/`raxis_of` are
/// [`strain_block_map`]; `rvec` is the per-AO `R = -(L + ri)`. A single
/// multiply per lane: FMA-free by construction, and generic over the device
/// float (`F: Float`, AGENTS.md §3).
///
/// The `i < nblocks*n` guard is required: the launch rounds the lane count
/// up to a whole number of cubes, so tail lanes must not write out of range.
#[cube(launch_unchecked)]
fn strain_scale_kernel<F: Float>(
    deriv: &Array<F>,
    src_of: &Array<u32>,
    raxis_of: &Array<u32>,
    rvec: &Array<F>,
    out: &mut Array<F>,
    n: usize,
    nblocks: usize,
    ngrids: usize,
) {
    let i = ABSOLUTE_POS;
    if i < nblocks * n {
        let b = i / n;
        let q = i % n;
        let a = q / ngrids;
        let src = src_of[b] as usize;
        let axis = raxis_of[b] as usize;
        out[i] = deriv[src * n + q] * rvec[a * 3 + axis];
    }
}

/// Launch [`strain_scale_kernel`] on resident handles. `R` stays generic so
/// one body serves every backend; the `Runtime` type never escapes into a
/// public signature.
fn launch_strain_scale_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    deriv: &cubecl::server::Handle,
    src_of: &cubecl::server::Handle,
    raxis_of: &cubecl::server::Handle,
    rvec: &cubecl::server::Handle,
    out: &cubecl::server::Handle,
    nblocks: usize,
    n: usize,
    ngrids: usize,
    dlen: usize,
) {
    let lanes = nblocks * n;
    // One multiply plus index arithmetic per lane.
    let (count, dim) = launch_1d(client, lanes, 8);
    unsafe {
        strain_scale_kernel::launch_unchecked::<F, R>(
            client,
            count,
            dim,
            // SAFETY: `deriv` holds exactly `dlen` elements (`D*n`,
            // `D = 4`/`10`), the maps hold `nblocks`, `rvec` holds
            // `nao*3`, `out` holds `nblocks*n`; the kernel guards
            // `i < nblocks*n` and every index is range-checked by construction.
            ArrayArg::from_raw_parts(deriv.clone(), dlen),
            ArrayArg::from_raw_parts(src_of.clone(), nblocks),
            ArrayArg::from_raw_parts(raxis_of.clone(), nblocks),
            ArrayArg::from_raw_parts(rvec.clone(), rvec_len(n, ngrids)),
            ArrayArg::from_raw_parts(out.clone(), lanes),
            n,
            nblocks,
            ngrids,
        );
    }
}

/// Length bound for the `rvec` raw part: `nao*3` where `nao = n/ngrids`.
fn rvec_len(n: usize, ngrids: usize) -> usize {
    if ngrids == 0 { 0 } else { 3 * (n / ngrids) }
}

/// Host-slice entry point: scale one image's Cartesian derivative table by
/// its per-AO `R` into the Cartesian strain table.
///
/// * `deriv_block` — `[D, ngrids, nao]`, F-order per component
///   (`D = 4` at `deriv = 0`, `D = 10` at `deriv = 1`).
/// * `rvec` — `nao*3` reals, `R = -(L + ri)` per AO.
/// * Returns `[S, ngrids, nao]`, `S = 9*comp` per [`strain_nblocks`].
///
/// An empty table (`nblocks*n == 0`) returns empty without launching.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] for `deriv >= 2` (via [`strain_comp`])
/// or a short input slice.
pub fn strain_scale(
    client: &AlgebraClient,
    deriv_block: &[f64],
    rvec: &[f64],
    ngrids: usize,
    nao: usize,
    deriv: u32,
) -> Result<Vec<f64>, AlgebraError> {
    let comp = strain_comp(deriv)?;
    let nblocks = 9 * comp;
    let d = if deriv == 0 { 4 } else { 10 };
    let n = ngrids
        .checked_mul(nao)
        .ok_or_else(|| AlgebraError::ShapeMismatch {
            expected: "ngrids*nao addressable".into(),
            actual: format!("ngrids={ngrids} nao={nao} overflow"),
        })?;
    if deriv_block.len() < d * n || rvec.len() < 3 * nao {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("deriv_block of {d}*{n} and rvec of {}", 3 * nao),
            actual: format!("deriv_block {} rvec {}", deriv_block.len(), rvec.len()),
        });
    }
    if nblocks * n == 0 {
        return Ok(Vec::new());
    }
    let (src_of, raxis_of) = strain_block_map(deriv)?;
    let out = dispatch_backend!(client, c, Rt, {
        let deriv_h = pyscf_algebra::launch::upload(c, &deriv_block[..d * n]);
        let src_h = c.create_from_slice(bytemuck::cast_slice(&src_of));
        let raxis_h = c.create_from_slice(bytemuck::cast_slice(&raxis_of));
        let rvec_h = pyscf_algebra::launch::upload(c, &rvec[..3 * nao]);
        let out_h = c.empty(nblocks * n * core::mem::size_of::<f64>());
        launch_strain_scale_on_handles::<Rt, f64>(
            c,
            &deriv_h,
            &src_h,
            &raxis_h,
            &rvec_h,
            &out_h,
            nblocks,
            n,
            ngrids,
            d * n,
        );
        let bytes = c.read(vec![out_h]);
        bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
    });
    Ok(out)
}

// ---------------------------------------------------------------------------
// Host Cartesian derivative tables (arbitrary l, normalised env coeffs).
// ---------------------------------------------------------------------------

/// One decoded libcint shell row.
struct ShellInfo {
    l: usize,
    nprim: usize,
    nctr: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    atom: usize,
    cart_off: usize,
    sph_off: usize,
}

/// Decode `_bas` into shell infos plus the Cartesian and spherical AO counts.
/// Widths are `ncart(l)*nctr` / `nsph(l)*nctr`, mirroring what `make_env`
/// stores in `ao_loc_nr` for a cart/sph `Mole` respectively.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] on ragged tables or out-of-range indices;
/// [`AlgebraError::NotYetImplemented`] for `l > 6` (the `c2s` ceiling, same
/// as [`crate::cart2sph_l_matrix`]).
fn decode_shells(
    bas: &[i32],
    atm_len: usize,
) -> Result<(Vec<ShellInfo>, usize, usize), AlgebraError> {
    if bas.len() % BAS_SLOTS != 0 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "bas a multiple of BAS_SLOTS".into(),
            actual: format!("len {}", bas.len()),
        });
    }
    if atm_len % ATM_SLOTS != 0 {
        return Err(AlgebraError::ShapeMismatch {
            expected: "atm a multiple of ATM_SLOTS".into(),
            actual: format!("len {atm_len}"),
        });
    }
    let natm = atm_len / ATM_SLOTS;
    let mut shells = Vec::with_capacity(bas.len() / BAS_SLOTS);
    let mut cart_off = 0usize;
    let mut sph_off = 0usize;
    for row in bas.chunks_exact(BAS_SLOTS) {
        let l = row[ANG_OF];
        if l < 0 {
            return Err(AlgebraError::ShapeMismatch {
                expected: "non-negative ANG_OF".into(),
                actual: format!("l={l}"),
            });
        }
        let l = l as usize;
        if l > 6 {
            return Err(AlgebraError::NotYetImplemented {
                phase: 18,
                what: "strain AO for l > 6 (k-shells): no c2s transform available",
            });
        }
        let nprim = row[NPRIM_OF] as usize;
        let nctr = row[NCTR_OF] as usize;
        let atom = row[ATOM_OF];
        if atom < 0 || (atom as usize) >= natm {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("ATOM_OF in [0, {natm})"),
                actual: format!("atom={atom}"),
            });
        }
        let ncart = (l + 1) * (l + 2) / 2;
        let nsph = 2 * l + 1;
        shells.push(ShellInfo {
            l,
            nprim,
            nctr,
            ptr_exp: row[PTR_EXP] as usize,
            ptr_coeff: row[PTR_COEFF] as usize,
            atom: atom as usize,
            cart_off,
            sph_off,
        });
        cart_off += ncart * nctr;
        sph_off += nsph * nctr;
    }
    Ok((shells, cart_off, sph_off))
}

/// `cart2sph` matrix with the error mapped into [`AlgebraError`]
/// (`decode_shells` already refuses `l > 6`, so this is unreachable in
/// practice — but `c2s` errors must still not panic, FOUND-07).
fn c2s_matrix(l: usize) -> Result<Vec<f64>, AlgebraError> {
    crate::cart2sph_l_matrix(l as u32).map_err(|e| AlgebraError::ShapeMismatch {
        expected: format!("c2s matrix for l={l}"),
        actual: format!("{e}"),
    })
}

/// Cartesian monomial powers for one axis, `p[k] = d^k`, `k = 0..=l`.
/// `p[0] = 1` even at `d = 0` (`0^0 = 1`), so no division ever appears and
/// no NaN can form at coincident grid points.
fn axis_powers(d: f64, l: usize, p: &mut [f64; 7]) {
    p[0] = 1.0;
    for k in 1..=l {
        p[k] = p[k - 1] * d;
    }
}

/// One image's Cartesian derivative table, `[D, ngrids, nao_cart]` F-order
/// per component (`D = 4` value+gradient, `D = 10` with the six hessian
/// components `xx,xy,xz,yy,yz,zz` when `need_hess`).
///
/// Per primitive with exponent `e` and (normalised — `make_env` already
/// applied `gto_norm` and the contraction normalisation) coefficient `c`,
/// with `M = X*Y*Z` the monomial and `E = c*exp(-e*r2)`:
///
/// ```text
/// v  = M*E
/// gx = (X1*Y*Z - 2e*dx*M)*E,            X1 = lx*px[lx-1] (0 if lx = 0)
/// hxx = (X2*Y*Z - 2e*(2lx+1)*M + 4e^2*dx^2*M)*E
/// hxy = (X1*Y1*Z - 2e*dx*X*Y1*Z - 2e*dy*X1*Y*Z + 4e^2*dx*dy*M)*E
/// ```
///
/// (and cyclic permutations), i.e. the analytic derivatives of
/// `x^lx*y^ly*z^lz*exp(-e*r^2)` in expanded polynomial form. The shell sum
/// over primitives is in `p_idx` order and the shell block is scaled by
/// `CINTcommon_fac_sp(l)` afterwards — the same factor `grid_ao.c` threads
/// through `feval` (`fac = CINTcommon_fac_sp(l)`).
///
/// `keep_shell[s]` (the bounding-box screen) skips whole shells; `centre`
/// is the image-shifted atom positions `ri + L`, one per atom.
#[allow(clippy::too_many_arguments)]
fn cart_deriv_image(
    shells: &[ShellInfo],
    env: &[f64],
    centre: &[[f64; 3]],
    coords: &[[f64; 3]],
    need_hess: bool,
    keep_shell: &[bool],
    out: &mut [f64],
    nao_cart: usize,
) -> Result<(), AlgebraError> {
    let ngrids = coords.len();
    let d = if need_hess { 10 } else { 4 };
    if out.len() < d * ngrids * nao_cart {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("deriv image block of {d}*{ngrids}*{nao_cart}"),
            actual: format!("len {}", out.len()),
        });
    }
    // Caller zeroes once per driver call would be wrong (per-image); zero here.
    out[..d * ngrids * nao_cart].fill(0.0);
    let mut px = [0.0f64; 7];
    let mut py = [0.0f64; 7];
    let mut pz = [0.0f64; 7];
    for (s, sh) in shells.iter().enumerate() {
        if !keep_shell[s] {
            continue;
        }
        let c0 = centre
            .get(sh.atom)
            .ok_or_else(|| AlgebraError::ShapeMismatch {
                expected: "centre per atom".into(),
                actual: format!("atom {} of {}", sh.atom, centre.len()),
            })?;
        let fac1 = crate::common_fac_sp(sh.l as u32);
        let powers = crate::cart_powers(sh.l as u32);
        for g in 0..ngrids {
            let r = coords[g];
            let dx = r[0] - c0[0];
            let dy = r[1] - c0[1];
            let dz = r[2] - c0[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            axis_powers(dx, sh.l, &mut px);
            axis_powers(dy, sh.l, &mut py);
            axis_powers(dz, sh.l, &mut pz);
            for c in 0..sh.nctr {
                for (ci, &(lx, ly, lz)) in powers.iter().enumerate() {
                    let (lx, ly, lz) = (lx as usize, ly as usize, lz as usize);
                    let x = px[lx];
                    let y = py[ly];
                    let z = pz[lz];
                    let m = x * y * z;
                    let x1 = if lx > 0 { lx as f64 * px[lx - 1] } else { 0.0 };
                    let y1 = if ly > 0 { ly as f64 * py[ly - 1] } else { 0.0 };
                    let z1 = if lz > 0 { lz as f64 * pz[lz - 1] } else { 0.0 };
                    let a = sh.cart_off + c * powers.len() + ci;
                    // Ordered primitive sum (p_idx order), then fac1 — the
                    // same association the host eval_gto kernel uses.
                    let mut v = 0.0;
                    let mut gx = 0.0;
                    let mut gy = 0.0;
                    let mut gz = 0.0;
                    let mut hxx = 0.0;
                    let mut hyy = 0.0;
                    let mut hzz = 0.0;
                    let mut hxy = 0.0;
                    let mut hxz = 0.0;
                    let mut hyz = 0.0;
                    for p in 0..sh.nprim {
                        let e = *env.get(sh.ptr_exp + p).ok_or_else(|| {
                            AlgebraError::ShapeMismatch {
                                expected: "exponent in env".into(),
                                actual: format!("ptr {}", sh.ptr_exp + p),
                            }
                        })?;
                        let coef = *env.get(sh.ptr_coeff + c * sh.nprim + p).ok_or_else(|| {
                            AlgebraError::ShapeMismatch {
                                expected: "coefficient in env".into(),
                                actual: format!("ptr {}", sh.ptr_coeff + c * sh.nprim + p),
                            }
                        })?;
                        let ep = coef * (-e * r2).exp();
                        v += m * ep;
                        let t = 2.0 * e;
                        gx += (x1 * y * z - t * dx * m) * ep;
                        gy += (x * y1 * z - t * dy * m) * ep;
                        gz += (x * y * z1 - t * dz * m) * ep;
                        if need_hess {
                            let t2 = t * t;
                            let x2 = if lx > 1 {
                                (lx * (lx - 1)) as f64 * px[lx - 2]
                            } else {
                                0.0
                            };
                            let y2 = if ly > 1 {
                                (ly * (ly - 1)) as f64 * py[ly - 2]
                            } else {
                                0.0
                            };
                            let z2 = if lz > 1 {
                                (lz * (lz - 1)) as f64 * pz[lz - 2]
                            } else {
                                0.0
                            };
                            hxx += (x2 * y * z - t * (2.0 * lx as f64 + 1.0) * m
                                + t2 * dx * dx * m)
                                * ep;
                            hyy += (x * y2 * z - t * (2.0 * ly as f64 + 1.0) * m
                                + t2 * dy * dy * m)
                                * ep;
                            hzz += (x * y * z2 - t * (2.0 * lz as f64 + 1.0) * m
                                + t2 * dz * dz * m)
                                * ep;
                            hxy += (x1 * y1 * z - t * dx * x * y1 * z - t * dy * x1 * y * z
                                + t2 * dx * dy * m)
                                * ep;
                            hxz += (x1 * y * z1 - t * dx * x * y * z1 - t * dz * x1 * y * z
                                + t2 * dx * dz * m)
                                * ep;
                            hyz += (x * y1 * z1 - t * dy * x * y * z1 - t * dz * x * y1 * z
                                + t2 * dy * dz * m)
                                * ep;
                        }
                    }
                    let n = ngrids * nao_cart;
                    let q = g + a * ngrids;
                    out[q] += v * fac1;
                    out[n + q] += gx * fac1;
                    out[2 * n + q] += gy * fac1;
                    out[3 * n + q] += gz * fac1;
                    if need_hess {
                        out[4 * n + q] += hxx * fac1;
                        out[5 * n + q] += hxy * fac1;
                        out[6 * n + q] += hxz * fac1;
                        out[7 * n + q] += hyy * fac1;
                        out[8 * n + q] += hyz * fac1;
                        out[9 * n + q] += hzz * fac1;
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public drivers.
// ---------------------------------------------------------------------------

/// Per-k-point strain-tensor AO tables.
///
/// `re[k]` / `im[k]` hold `S*ngrids*nao` reals, `S = 9*comp`
/// ([`strain_nblocks`]), F-order per block: block `b = (x*3+y)*comp+c` starts
/// at `b*ngrids*nao`, element `(g, mu)` at `+ g + mu*ngrids`. Reshaping the
/// flat block run to `(3, 3, comp, ngrids, nao)` C-order reproduces
/// `rks_stress.py:143-146`'s `out.reshape(3,3,comp,ngrids,-1)` exactly.
#[derive(Debug, Clone)]
pub struct StrainAoOutput {
    /// Real planes, one per k-point.
    pub re: Vec<Vec<f64>>,
    /// Imaginary planes, one per k-point.
    pub im: Vec<Vec<f64>>,
    /// Grid-point count.
    pub ngrids: usize,
    /// AO count (spherical or Cartesian per `spherical`).
    pub nao: usize,
    /// Strain order (0 or 1).
    pub deriv: u32,
    /// `true` for the spherical transform, `false` for raw Cartesian.
    pub spherical: bool,
}

/// The periodic strain-tensor AO collocation driver (18-11 Task 1+2).
///
/// Per image `L`: shift the grid (`coords - L`, i.e. upstream's atom `+L`),
/// evaluate the Cartesian derivative table on the host, run it through the
/// [`strain_scale`] device kernel with `R = -(L + ri)` per AO, transform to
/// spherical per shell when `spherical`, and Bloch-sum with `exp(+i k.L)`
/// (K-07's sign) into the per-k planes.
///
/// An empty `kpts` means the single gamma point. `rcut` is the per-shell
/// lattice radius (`None` disables the bounding-box screen); `screen`
/// additionally gates it (the wrapper's `PYSCF_PBC_STRAIN_SCREEN` switch).
/// `ls` may be empty (all-screened or degenerate cell): the planes are then
/// correctly-shaped zeros.
///
/// # Errors
/// * [`AlgebraError::ShapeMismatch`] — ragged libcint tables, short slices,
///   or `deriv >= 2` (the Task-4 named refusal).
/// * [`AlgebraError::NotYetImplemented`] — `l > 6` shells.
#[allow(clippy::too_many_arguments)]
pub fn eval_strain_ao(
    client: &AlgebraClient,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    ls: &[[f64; 3]],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    atom_coords: &[[f64; 3]],
    deriv: u32,
    spherical: bool,
    rcut: Option<&[f64]>,
    screen: bool,
) -> Result<StrainAoOutput, AlgebraError> {
    let comp = strain_comp(deriv)?;
    let sblocks = 9 * comp;
    let need_hess = deriv == 1;
    let (shells, nao_cart, nao_sph) = decode_shells(bas, atm.len())?;
    let nao = if spherical { nao_sph } else { nao_cart };
    let ngrids = coords.len();
    if let Some(r) = rcut {
        if r.len() != shells.len() {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("rcut per shell ({})", shells.len()),
                actual: format!("len {}", r.len()),
            });
        }
    }
    let natm = if atm.len() % ATM_SLOTS == 0 {
        atm.len() / ATM_SLOTS
    } else {
        0
    };
    if atom_coords.len() != natm {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("atom_coords per atom ({natm})"),
            actual: format!("len {}", atom_coords.len()),
        });
    }

    let owned_gamma = [[0.0f64; 3]];
    let kpts: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    let nkpts = kpts.len();
    let n = sblocks * ngrids * nao;

    // Grid bounding box for the O(1) per-(image, shell) screen.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for r in coords {
        for axis in 0..3 {
            if r[axis] < lo[axis] {
                lo[axis] = r[axis];
            }
            if r[axis] > hi[axis] {
                hi[axis] = r[axis];
            }
        }
    }
    if ngrids == 0 {
        lo = [0.0; 3];
        hi = [0.0; 3];
    }

    let mut acc_re = vec![vec![0.0f64; n]; nkpts];
    let mut acc_im = vec![vec![0.0f64; n]; nkpts];
    if n == 0 {
        return Ok(StrainAoOutput {
            re: acc_re,
            im: acc_im,
            ngrids,
            nao,
            deriv,
            spherical,
        });
    }

    let d = if need_hess { 10 } else { 4 };
    let mut deriv_block = vec![0.0f64; d * ngrids * nao_cart];
    let mut rvec = vec![0.0f64; 3 * nao_cart];
    let mut centre = vec![[0.0f64; 3]; natm];
    let mut keep = vec![true; shells.len()];
    let mut cart_strain = vec![0.0f64; sblocks * ngrids * nao_cart];
    // Spherical assembly buffer (identity copy when cart output).
    let mut strain = vec![0.0f64; n];

    for l_img in ls {
        // Image-shifted centres + per-AO R, and the box screen.
        for (i, c) in centre.iter_mut().enumerate() {
            let ri = atom_coords[i];
            c[0] = ri[0] + l_img[0];
            c[1] = ri[1] + l_img[1];
            c[2] = ri[2] + l_img[2];
        }
        for (s, sh) in shells.iter().enumerate() {
            let c = centre[sh.atom];
            let use_screen = screen && rcut.is_some() && ngrids > 0;
            keep[s] = if use_screen {
                let r = rcut.map_or(f64::INFINITY, |r| r[s]);
                let r2 = r * r;
                let mut d2 = 0.0;
                for axis in 0..3 {
                    let dd = if c[axis] < lo[axis] {
                        lo[axis] - c[axis]
                    } else if c[axis] > hi[axis] {
                        c[axis] - hi[axis]
                    } else {
                        0.0
                    };
                    d2 += dd * dd;
                }
                d2 <= r2
            } else {
                true
            };
            // R = -(L + ri) per grid_ao.c (same for every AO of the shell).
            let rx = -(l_img[0] + atom_coords[sh.atom][0]);
            let ry = -(l_img[1] + atom_coords[sh.atom][1]);
            let rz = -(l_img[2] + atom_coords[sh.atom][2]);
            let ncart = (sh.l + 1) * (sh.l + 2) / 2;
            for c_idx in 0..sh.nctr {
                for ci in 0..ncart {
                    let a = sh.cart_off + c_idx * ncart + ci;
                    rvec[a * 3] = rx;
                    rvec[a * 3 + 1] = ry;
                    rvec[a * 3 + 2] = rz;
                }
            }
        }

        cart_deriv_image(
            &shells,
            env,
            &centre,
            coords,
            need_hess,
            &keep,
            &mut deriv_block,
            nao_cart,
        )?;
        cart_strain = strain_scale(client, &deriv_block, &rvec, ngrids, nao_cart, deriv)?;

        // Cart -> sph per shell (linear: commutes with the R scale, matching
        // upstream's transform-then-scale order exactly).
        if spherical {
            strain.fill(0.0);
            for sh in &shells {
                let ncart = (sh.l + 1) * (sh.l + 2) / 2;
                let nsph = 2 * sh.l + 1;
                let cmat = c2s_matrix(sh.l)?;
                for b in 0..sblocks {
                    for g in 0..ngrids {
                        for c_idx in 0..sh.nctr {
                            for m in 0..nsph {
                                let mut v = 0.0;
                                for ci in 0..ncart {
                                    let ca = sh.cart_off + c_idx * ncart + ci;
                                    v += cmat[m * ncart + ci]
                                        * cart_strain[b * ngrids * nao_cart + g + ca * ngrids];
                                }
                                let sa = sh.sph_off + c_idx * nsph + m;
                                strain[b * ngrids * nao + g + sa * ngrids] = v;
                            }
                        }
                    }
                }
            }
        } else {
            strain.copy_from_slice(&cart_strain);
        }

        // Bloch sum: acc[k] += exp(i k.L) * strain, K-07's theta order
        // (kx*Lx, then += ky*Ly, then += kz*Lz).
        for (k, kpt) in kpts.iter().enumerate() {
            let mut theta = kpt[0] * l_img[0];
            theta += kpt[1] * l_img[1];
            theta += kpt[2] * l_img[2];
            let (s, c) = theta.sin_cos();
            let (re, im) = (&mut acc_re[k], &mut acc_im[k]);
            for (p, v) in strain.iter().enumerate() {
                re[p] += c * v;
                im[p] += s * v;
            }
        }
    }

    Ok(StrainAoOutput {
        re: acc_re,
        im: acc_im,
        ngrids,
        nao,
        deriv,
        spherical,
    })
}

/// Per-k-point periodic second-derivative AO tables (`[10, ngrids, nao]`,
/// value + gradient + `xx,xy,xz,yy,yz,zz`), Bloch-summed over `ls` with
/// `exp(+i k.L)`.
///
/// Reference collocation for the 18-12 grid-response caller: upstream's
/// `test_eval_ao_grid_response` (`test_rks_stress.py:196`, `:240`) builds
/// the deriv-1 grid response from the undisplaced cell's deriv-2 AO, which
/// no other Rust surface provides. Same primitives, image list and phases
/// as [`eval_strain_ao`], without the strain weight.
///
/// # Errors
/// As [`eval_strain_ao`].
#[allow(clippy::too_many_arguments)]
pub fn eval_ao_deriv2(
    client: &AlgebraClient,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    ls: &[[f64; 3]],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    atom_coords: &[[f64; 3]],
    spherical: bool,
    rcut: Option<&[f64]>,
    screen: bool,
) -> Result<Deriv2AoOutput, AlgebraError> {
    // Reuse the strain driver's per-image machinery through a deriv=1-shaped
    // assembly: implement directly here to keep the strain block layout out
    // of this function's contract.
    let (shells, nao_cart, nao_sph) = decode_shells(bas, atm.len())?;
    let nao = if spherical { nao_sph } else { nao_cart };
    let ngrids = coords.len();
    if let Some(r) = rcut {
        if r.len() != shells.len() {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("rcut per shell ({})", shells.len()),
                actual: format!("len {}", r.len()),
            });
        }
    }
    let natm = atm.len() / ATM_SLOTS;
    if atom_coords.len() != natm {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("atom_coords per atom ({natm})"),
            actual: format!("len {}", atom_coords.len()),
        });
    }
    let _ = client;

    let owned_gamma = [[0.0f64; 3]];
    let kpts: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    let nkpts = kpts.len();
    const D2: usize = 10;
    let n = D2 * ngrids * nao;

    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for r in coords {
        for axis in 0..3 {
            if r[axis] < lo[axis] {
                lo[axis] = r[axis];
            }
            if r[axis] > hi[axis] {
                hi[axis] = r[axis];
            }
        }
    }

    let mut acc_re = vec![vec![0.0f64; n]; nkpts];
    let mut acc_im = vec![vec![0.0f64; n]; nkpts];
    if n == 0 {
        return Ok(Deriv2AoOutput {
            re: acc_re,
            im: acc_im,
            ngrids,
            nao,
            spherical,
        });
    }

    let mut cart_block = vec![0.0f64; D2 * ngrids * nao_cart];
    let mut centre = vec![[0.0f64; 3]; natm];
    let mut keep = vec![true; shells.len()];
    let mut shell_block = vec![0.0f64; n];

    for l_img in ls {
        for (i, c) in centre.iter_mut().enumerate() {
            let ri = atom_coords[i];
            c[0] = ri[0] + l_img[0];
            c[1] = ri[1] + l_img[1];
            c[2] = ri[2] + l_img[2];
        }
        for (s, sh) in shells.iter().enumerate() {
            let c = centre[sh.atom];
            keep[s] = if screen && rcut.is_some() && ngrids > 0 {
                let r = rcut.map_or(f64::INFINITY, |r| r[s]);
                let r2 = r * r;
                let mut d2 = 0.0;
                for axis in 0..3 {
                    let dd = if c[axis] < lo[axis] {
                        lo[axis] - c[axis]
                    } else if c[axis] > hi[axis] {
                        c[axis] - hi[axis]
                    } else {
                        0.0
                    };
                    d2 += dd * dd;
                }
                d2 <= r2
            } else {
                true
            };
        }

        cart_deriv_image(
            &shells,
            env,
            &centre,
            coords,
            true,
            &keep,
            &mut cart_block,
            nao_cart,
        )?;

        if spherical {
            shell_block.fill(0.0);
            for sh in &shells {
                let ncart = (sh.l + 1) * (sh.l + 2) / 2;
                let nsph = 2 * sh.l + 1;
                let cmat = c2s_matrix(sh.l)?;
                for b in 0..D2 {
                    for g in 0..ngrids {
                        for c_idx in 0..sh.nctr {
                            for m in 0..nsph {
                                let mut v = 0.0;
                                for ci in 0..ncart {
                                    let ca = sh.cart_off + c_idx * ncart + ci;
                                    v += cmat[m * ncart + ci]
                                        * cart_block[b * ngrids * nao_cart + g + ca * ngrids];
                                }
                                let sa = sh.sph_off + c_idx * nsph + m;
                                shell_block[b * ngrids * nao + g + sa * ngrids] = v;
                            }
                        }
                    }
                }
            }
        } else {
            shell_block.copy_from_slice(&cart_block);
        }

        for (k, kpt) in kpts.iter().enumerate() {
            let mut theta = kpt[0] * l_img[0];
            theta += kpt[1] * l_img[1];
            theta += kpt[2] * l_img[2];
            let (s, c) = theta.sin_cos();
            let (re, im) = (&mut acc_re[k], &mut acc_im[k]);
            for (p, v) in shell_block.iter().enumerate() {
                re[p] += c * v;
                im[p] += s * v;
            }
        }
    }

    Ok(Deriv2AoOutput {
        re: acc_re,
        im: acc_im,
        ngrids,
        nao,
        spherical,
    })
}

/// Per-k-point periodic second-derivative AO tables; see [`eval_ao_deriv2`].
/// `re[k]` / `im[k]` hold `10*ngrids*nao` reals, component-leading
/// (`value, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz`), F-order within.
#[derive(Debug, Clone)]
pub struct Deriv2AoOutput {
    /// Real planes, one per k-point.
    pub re: Vec<Vec<f64>>,
    /// Imaginary planes, one per k-point.
    pub im: Vec<Vec<f64>>,
    /// Grid-point count.
    pub ngrids: usize,
    /// AO count (spherical or Cartesian per `spherical`).
    pub nao: usize,
    /// `true` for the spherical transform, `false` for raw Cartesian.
    pub spherical: bool,
}
