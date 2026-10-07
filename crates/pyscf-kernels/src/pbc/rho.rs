//! SCF-01 — the periodic density on the device, one k-point per call.
//!
//! ```text
//! c0[j, g] = Σ_i ao⁰[i, g] · dm[i, j]              (`_dot_ao_dm`)
//! ρ_c[g]   = s_c · Re Σ_j conj(aoᶜ[j, g]) · c0[j, g] (`_contract_rho`, hermi = 1)
//! ```
//!
//! with `s_0 = 1` and `s_c = 2` for the gradient rows. This is
//! `eval_rho_one` in `pyscf-pbc-dft/src/numint.rs` (`numint.py:96-186`). On
//! the host it runs as two rayon passes over a materialised `c0` table; here
//! one lane owns one grid point and never materialises `c0`: for each `j` it
//! forms `c0[j, g]` (the `i` sum in the host's order, zero `dm` entries
//! skipped as the host skips them) and immediately adds `conj(aoᶜ) · c0` into
//! the `ncomp` complex accumulators. Per output the additions are the host's,
//! term for term, so on the CPU runtime the result matches it bitwise; on a
//! GPU the compiler's FMA contraction may move the last bits.
//!
//! Why: measured 2026-09-24 on a Kaggle T4 VM (4 vCPU), `eval_rho` was 5.7 s
//! of a 26 s SCF effective-potential total — host work the GPU never saw.
//!
//! # SCF-03 — the tiled route
//!
//! That one-lane-per-point kernel walks the whole density matrix per lane:
//! `4 · nao²` loads per grid point, the AO ones strided by `ngrids`. At
//! `nao = 910` that is 26 MB of traffic per lane, and on a Kaggle T4 the
//! density of one 4 992-point block cost as much as its AO table. The tiled
//! route materialises `c0` after all, as a GEMM: one lane owns
//! [`RHO_TILE`] columns `j` at one grid point (adjacent lanes are adjacent
//! grid points, so every AO load is coalesced and the density-matrix loads
//! are the same eight values for a whole plane), then a second kernel folds
//! `conj(aoᶜ) · c0` over `j`. Each `c0[j, g]` still adds the `i` terms in
//! ascending order and each accumulator the `j` terms in ascending order —
//! the operations of [`rho_k_kernel`] in the same order — so the two routes
//! are bitwise identical on the CPU runtime. `c0` costs one component of one
//! k-point of the table (`16 · nao · ngrids` bytes).
//!
//! MEASURED 2026-10-07 on a Kaggle T4 (`grid_contract_bench`, `nao = 910`,
//! 9 k-points): 6.1 s per-output against 2.8 s tiled for a 4 992-point
//! deriv-1 block, 14.9 s against 9.0 s for a 16 003-point value block; the
//! two routes agree to 8e-16 relative there (not bitwise: the GPU compiler
//! contracts the two differently).
//!
//! Generic over the device float (`F: Float`, AGENTS.md §3 / RULE 5).

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::{launch_1d, launch_1d_chunked, upload};
use pyscf_algebra::{AlgebraClient, AlgebraError};

/// The most density components a lane accumulates (value + gradient).
const RHO_NCOMP_MAX: usize = 4;

/// One lane per grid point `g`. `ao_*` hold `ncomp · nao · ngrids` values
/// (`e = c · nao · ngrids + j · ngrids + g`), `dm_*` `nao · nao` row-major
/// (`dm[i · nao + j]`). Writes the raw accumulators `out_*[c · ngrids + g]`
/// (unscaled); the host applies `s_c` and takes `Re`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn rho_k_kernel<F: Float>(
    ao_re: &Array<F>,
    ao_im: &Array<F>,
    dm_re: &Array<F>,
    dm_im: &Array<F>,
    out_re: &mut Array<F>,
    out_im: &mut Array<F>,
    nao: usize,
    ngrids: usize,
    ncomp: usize,
    base: usize,
) {
    let g = ABSOLUTE_POS;
    if g < ngrids {
        let zero = F::from_int(0);
        let cstride = nao * ngrids;
        // `ncomp <= 4`: the accumulators are written back per component
        // after the `j` loop, so keep them in a small array.
        let mut acc_re = Array::<F>::new(RHO_NCOMP_MAX);
        let mut acc_im = Array::<F>::new(RHO_NCOMP_MAX);
        for c in 0..RHO_NCOMP_MAX {
            acc_re[c] = zero;
            acc_im[c] = zero;
        }
        for j in 0..nao {
            let mut cr = zero;
            let mut ci = zero;
            for i in 0..nao {
                let dr = dm_re[i * nao + j];
                let di = dm_im[i * nao + j];
                if dr != zero || di != zero {
                    let ar = ao_re[base + i * ngrids + g];
                    let ai = ao_im[base + i * ngrids + g];
                    cr += ar * dr - ai * di;
                    ci += ar * di + ai * dr;
                }
            }
            for c in 0..ncomp {
                let e = base + c * cstride + j * ngrids + g;
                let ar = ao_re[e];
                let ai = -ao_im[e];
                acc_re[c] += ar * cr - ai * ci;
                acc_im[c] += ar * ci + ai * cr;
            }
        }
        for c in 0..ncomp {
            out_re[c * ngrids + g] = acc_re[c];
            out_im[c * ngrids + g] = acc_im[c];
        }
    }
}

/// Density-matrix columns one [`rho_c0_kernel`] lane carries.
const RHO_TILE: usize = 8;

/// SCF-03 — `c0[j, g] = Σ_i ao⁰[i, g] · dm[i, j]`, one lane per
/// `(j-tile, g)`; `c0_*[j · ngrids + g]`. Zero `dm` entries are skipped as
/// [`rho_k_kernel`] skips them.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn rho_c0_kernel<F: Float>(
    ao_re: &Array<F>,
    ao_im: &Array<F>,
    dm_re: &Array<F>,
    dm_im: &Array<F>,
    c0_re: &mut Array<F>,
    c0_im: &mut Array<F>,
    nao: usize,
    ngrids: usize,
    base: usize,
    lane0: usize,
) {
    // `lane0`: chunked on the CPU runtime — the accumulators are stack per
    // iteration there (`launch_1d_chunked`).
    let tid = ABSOLUTE_POS + lane0;
    let ntile = (nao + RHO_TILE - 1) / RHO_TILE;
    if tid < ntile * ngrids {
        let g = tid % ngrids;
        let j0 = (tid / ngrids) * RHO_TILE;
        let zero = F::from_int(0);
        let mut cr = Array::<F>::new(RHO_TILE);
        let mut ci = Array::<F>::new(RHO_TILE);
        #[unroll]
        for b in 0..RHO_TILE {
            cr[b] = zero;
            ci[b] = zero;
        }
        for i in 0..nao {
            let ar = ao_re[base + i * ngrids + g];
            let ai = ao_im[base + i * ngrids + g];
            let row = i * nao + j0;
            #[unroll]
            for b in 0..RHO_TILE {
                if j0 + b < nao {
                    let dr = dm_re[row + b];
                    let di = dm_im[row + b];
                    if dr != zero || di != zero {
                        cr[b] += ar * dr - ai * di;
                        ci[b] += ar * di + ai * dr;
                    }
                }
            }
        }
        #[unroll]
        for b in 0..RHO_TILE {
            if j0 + b < nao {
                c0_re[(j0 + b) * ngrids + g] = cr[b];
                c0_im[(j0 + b) * ngrids + g] = ci[b];
            }
        }
    }
}

/// SCF-03 — `out_c[g] = Σ_j conj(aoᶜ[j, g]) · c0[j, g]`, one lane per grid
/// point: the second half of [`rho_k_kernel`] over a materialised `c0`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn rho_fold_kernel<F: Float>(
    ao_re: &Array<F>,
    ao_im: &Array<F>,
    c0_re: &Array<F>,
    c0_im: &Array<F>,
    out_re: &mut Array<F>,
    out_im: &mut Array<F>,
    nao: usize,
    ngrids: usize,
    ncomp: usize,
    base: usize,
) {
    let g = ABSOLUTE_POS;
    if g < ngrids {
        let zero = F::from_int(0);
        let cstride = nao * ngrids;
        let mut acc_re = Array::<F>::new(RHO_NCOMP_MAX);
        let mut acc_im = Array::<F>::new(RHO_NCOMP_MAX);
        for c in 0..RHO_NCOMP_MAX {
            acc_re[c] = zero;
            acc_im[c] = zero;
        }
        for j in 0..nao {
            let cr = c0_re[j * ngrids + g];
            let ci = c0_im[j * ngrids + g];
            for c in 0..ncomp {
                let e = base + c * cstride + j * ngrids + g;
                let ar = ao_re[e];
                let ai = -ao_im[e];
                acc_re[c] += ar * cr - ai * ci;
                acc_im[c] += ar * ci + ai * cr;
            }
        }
        for c in 0..ncomp {
            out_re[c * ngrids + g] = acc_re[c];
            out_im[c * ngrids + g] = acc_im[c];
        }
    }
}

/// Queue one k-point's density launches over AO planes already on the device
/// (`ao_len` values each, this k-point's block starting at `base`) and return
/// the two `ncomp · ngrids` output handles.
#[allow(clippy::too_many_arguments)]
fn launch_rho<R: Runtime>(
    c: &ComputeClient<R>,
    are: &Handle,
    aim: &Handle,
    ao_len: usize,
    base: usize,
    dm_re: &[f64],
    dm_im: &[f64],
    ncomp: usize,
    nao: usize,
    ngrids: usize,
) -> (Handle, Handle) {
    let dre = upload::<R, f64>(c, dm_re);
    let dim_ = upload::<R, f64>(c, dm_im);
    let ore = c.empty(ncomp * ngrids * core::mem::size_of::<f64>());
    let oim = c.empty(ncomp * ngrids * core::mem::size_of::<f64>());
    if crate::pbc::band_vmat::tiled_contractions(c) {
        let plane = nao * ngrids;
        let c0_re = c.empty(plane * core::mem::size_of::<f64>());
        let c0_im = c.empty(plane * core::mem::size_of::<f64>());
        let lanes = nao.div_ceil(RHO_TILE) * ngrids;
        let local_bytes = 2 * RHO_TILE * core::mem::size_of::<f64>();
        for chunk in launch_1d_chunked(c, lanes, 8 * nao * RHO_TILE, local_bytes) {
            unsafe {
                rho_c0_kernel::launch_unchecked::<f64, R>(
                    c,
                    CubeCount::Static(chunk.count_x, 1, 1),
                    chunk.dim,
                    // SAFETY: the planes hold `ao_len >= base + nao · ngrids`
                    // values, the density matrix `nao²` and `c0` `nao ·
                    // ngrids`; the kernel guards the lane and `j < nao`.
                    ArrayArg::from_raw_parts(are.clone(), ao_len),
                    ArrayArg::from_raw_parts(aim.clone(), ao_len),
                    ArrayArg::from_raw_parts(dre.clone(), nao * nao),
                    ArrayArg::from_raw_parts(dim_.clone(), nao * nao),
                    ArrayArg::from_raw_parts(c0_re.clone(), plane),
                    ArrayArg::from_raw_parts(c0_im.clone(), plane),
                    nao,
                    ngrids,
                    base,
                    chunk.lane0,
                );
            }
        }
        let (count, dim) = launch_1d(c, ngrids, 8 * nao * ncomp);
        unsafe {
            rho_fold_kernel::launch_unchecked::<f64, R>(
                c,
                count,
                dim,
                // SAFETY: as above, with `base + ncomp · nao · ngrids <=
                // ao_len`; outputs hold `ncomp · ngrids`; the kernel guards
                // `g < ngrids` and only indexes `c < ncomp <= 4`.
                ArrayArg::from_raw_parts(are.clone(), ao_len),
                ArrayArg::from_raw_parts(aim.clone(), ao_len),
                ArrayArg::from_raw_parts(c0_re, plane),
                ArrayArg::from_raw_parts(c0_im, plane),
                ArrayArg::from_raw_parts(ore.clone(), ncomp * ngrids),
                ArrayArg::from_raw_parts(oim.clone(), ncomp * ngrids),
                nao,
                ngrids,
                ncomp,
                base,
            );
        }
        return (ore, oim);
    }
    // `8 · nao · (nao + ncomp)` flops per lane.
    let (count, dim) = launch_1d(c, ngrids, 8 * nao * (nao + ncomp));
    unsafe {
        rho_k_kernel::launch_unchecked::<f64, R>(
            c,
            count,
            dim,
            // SAFETY: the planes hold `ao_len >= base + ncomp · nao · ngrids`
            // values, the density matrix `nao²`, the outputs `ncomp ·
            // ngrids`; the kernel guards `g < ngrids` and only indexes
            // `c < ncomp <= 4`.
            ArrayArg::from_raw_parts(are.clone(), ao_len),
            ArrayArg::from_raw_parts(aim.clone(), ao_len),
            ArrayArg::from_raw_parts(dre, nao * nao),
            ArrayArg::from_raw_parts(dim_, nao * nao),
            ArrayArg::from_raw_parts(ore.clone(), ncomp * ngrids),
            ArrayArg::from_raw_parts(oim.clone(), ncomp * ngrids),
            nao,
            ngrids,
            ncomp,
            base,
        );
    }
    (ore, oim)
}

/// The raw density accumulators at one k-point: `(re, im)`, each
/// `ncomp · ngrids` (`[c · ngrids + g]`), BEFORE the `s_c` factor and `Re`.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when the planes or the density matrix do
/// not match `(ncomp, nao, ngrids)`, or `ncomp` is not 1..=4.
#[allow(clippy::too_many_arguments)]
pub fn rho_k(
    client: &AlgebraClient,
    ao_re: &[f64],
    ao_im: &[f64],
    dm_re: &[f64],
    dm_im: &[f64],
    ncomp: usize,
    nao: usize,
    ngrids: usize,
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    let want = ncomp * nao * ngrids;
    if !(1..=4).contains(&ncomp) {
        return Err(AlgebraError::ShapeMismatch {
            expected: "ncomp in 1..=4".into(),
            actual: ncomp.to_string(),
        });
    }
    if ao_re.len() != want || ao_im.len() != want {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("AO planes of ncomp·nao·ngrids = {want}"),
            actual: format!("re {} im {}", ao_re.len(), ao_im.len()),
        });
    }
    if dm_re.len() != nao * nao || dm_im.len() != nao * nao {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("density matrix of nao² = {}", nao * nao),
            actual: format!("re {} im {}", dm_re.len(), dm_im.len()),
        });
    }
    if ngrids == 0 || nao == 0 {
        return Ok((vec![0.0; ncomp * ngrids], vec![0.0; ncomp * ngrids]));
    }
    let bytes = dispatch_backend!(client, c, Rt, {
        let are = upload::<Rt, f64>(c, ao_re);
        let aim = upload::<Rt, f64>(c, ao_im);
        let (ore, oim) = launch_rho::<Rt>(c, &are, &aim, want, 0, dm_re, dm_im, ncomp, nao, ngrids);
        c.read(vec![ore, oim])
    });
    let re: &[f64] = bytemuck::cast_slice(&bytes[0]);
    let im: &[f64] = bytemuck::cast_slice(&bytes[1]);
    Ok((re.to_vec(), im.to_vec()))
}

/// SCF-02 — [`rho_k`] at k-point `k` of a device-resident AO table: only the
/// density matrix goes up and the `ncomp · ngrids` accumulators come back.
///
/// # Errors
/// As [`rho_k`], plus `k` out of range.
pub fn rho_k_table(
    client: &AlgebraClient,
    table: &crate::pbc::DeviceAoTable,
    k: usize,
    dm_re: &[f64],
    dm_im: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    let (nkpts, ncomp, nao, ngrids) = table.dims();
    if k >= nkpts {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("k < nkpts = {nkpts}"),
            actual: k.to_string(),
        });
    }
    if dm_re.len() != nao * nao || dm_im.len() != nao * nao {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("density matrix of nao² = {}", nao * nao),
            actual: format!("re {} im {}", dm_re.len(), dm_im.len()),
        });
    }
    if ngrids == 0 || nao == 0 {
        return Ok((vec![0.0; ncomp * ngrids], vec![0.0; ncomp * ngrids]));
    }
    let n = ncomp * nao * ngrids;
    let (are, aim) = table.planes();
    let bytes = dispatch_backend!(client, c, Rt, {
        let (ore, oim) = launch_rho::<Rt>(
            c,
            are,
            aim,
            nkpts * n,
            k * n,
            dm_re,
            dm_im,
            ncomp,
            nao,
            ngrids,
        );
        c.read(vec![ore, oim])
    });
    let re: &[f64] = bytemuck::cast_slice(&bytes[0]);
    let im: &[f64] = bytemuck::cast_slice(&bytes[1]);
    Ok((re.to_vec(), im.to_vec()))
}
