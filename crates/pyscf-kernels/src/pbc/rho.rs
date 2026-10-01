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
//! Generic over the device float (`F: Float`, AGENTS.md §3 / RULE 5).

use cubecl::prelude::*;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::{launch_1d, upload};
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
        let dre = upload::<Rt, f64>(c, dm_re);
        let dim_ = upload::<Rt, f64>(c, dm_im);
        let ore = c.empty(ncomp * ngrids * core::mem::size_of::<f64>());
        let oim = c.empty(ncomp * ngrids * core::mem::size_of::<f64>());
        // `8 · nao · (nao + ncomp)` flops per lane.
        let (count, dim) = launch_1d(c, ngrids, 8 * nao * (nao + ncomp));
        unsafe {
            rho_k_kernel::launch_unchecked::<f64, Rt>(
                c,
                count,
                dim,
                // SAFETY: every handle length is its slice's length (or the
                // `ncomp · ngrids` outputs); the kernel guards `g < ngrids`
                // and only indexes `c < ncomp <= 4`.
                ArrayArg::from_raw_parts(are.clone(), want),
                ArrayArg::from_raw_parts(aim.clone(), want),
                ArrayArg::from_raw_parts(dre.clone(), nao * nao),
                ArrayArg::from_raw_parts(dim_.clone(), nao * nao),
                ArrayArg::from_raw_parts(ore.clone(), ncomp * ngrids),
                ArrayArg::from_raw_parts(oim.clone(), ncomp * ngrids),
                nao,
                ngrids,
                ncomp,
                0,
            );
        }
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
        let dre = upload::<Rt, f64>(c, dm_re);
        let dim_ = upload::<Rt, f64>(c, dm_im);
        let ore = c.empty(ncomp * ngrids * core::mem::size_of::<f64>());
        let oim = c.empty(ncomp * ngrids * core::mem::size_of::<f64>());
        let (count, dim) = launch_1d(c, ngrids, 8 * nao * (nao + ncomp));
        unsafe {
            rho_k_kernel::launch_unchecked::<f64, Rt>(
                c,
                count,
                dim,
                // SAFETY: the table holds `nkpts · n` values per plane and
                // `base = k · n` with `k < nkpts` keeps every index in range;
                // outputs hold `ncomp · ngrids`; the kernel guards `g < ngrids`.
                ArrayArg::from_raw_parts(are.clone(), nkpts * n),
                ArrayArg::from_raw_parts(aim.clone(), nkpts * n),
                ArrayArg::from_raw_parts(dre.clone(), nao * nao),
                ArrayArg::from_raw_parts(dim_.clone(), nao * nao),
                ArrayArg::from_raw_parts(ore.clone(), ncomp * ngrids),
                ArrayArg::from_raw_parts(oim.clone(), ncomp * ngrids),
                nao,
                ngrids,
                ncomp,
                k * n,
            );
        }
        c.read(vec![ore, oim])
    });
    let re: &[f64] = bytemuck::cast_slice(&bytes[0]);
    let im: &[f64] = bytemuck::cast_slice(&bytes[1]);
    Ok((re.to_vec(), im.to_vec()))
}
