//! BAND-08 — the band-k XC/local contraction, fused onto the device so the
//! band AO table (with gradients) never reaches the host.
//!
//! ```text
//! aow[k][q, g]  = Σ_{n<nvar} wv[n][g] · ao_k^{(n)}[q, g]
//! v[k][p, q]    = Σ_g conj(ao_k^{(0)}[p, g]) · aow[k][q, g]
//! ```
//!
//! This is `_vxc_mat` of `pbc/dft/numint.py:828-850` for one k-point — the
//! only thing the fused band Fock (`fused_band_fock`) does with the AO table
//! `band_vmats` evaluates at the band k-points. It generalises the K-14f
//! kernel ([`crate::pbc::local_vmat`]) in two ways: the table may carry
//! `comp` components (`1` for LDA, `4` for a GGA's value + gradient), and the
//! weight is a `nvar × ngrids` table rather than one real potential.
//!
//! # Why this kernel exists
//!
//! Measured 2026-09-23 on KTaO3 (`gth-szv-molopt-sr`, 27 AOs, mesh 35³,
//! 33 band k-points, 16-core CPU runtime): one `get_bands` was 12.8 s, of
//! which the band AO kernel itself was ~1.2 s and the READ-BACK of the
//! `nk · 4 · ngrids · nao` complex table (2.4 GB per call) plus the host
//! contraction was the rest. On a discrete GPU the same table would cross
//! PCIe and then be reduced by a host with two cores. Contracting on the
//! device brings home `nk · nao²` — the answer, not the intermediate.
//!
//! # Summation order
//!
//! One lane owns one `(k, p, q)` element and walks `g` upwards serially; per
//! grid point it forms `aow` from the `nvar` components in increasing `n`
//! and adds `conj(ao⁰) · aow` to its running complex sum. The host route
//! (`vxc_mat_one`) forms `aow` in the same `n` order but reduces over `g` with
//! `oracle_sum`'s pairwise tree, so the two routes are NOT bit-identical —
//! they agree to the rounding of a 40 000-term sum (~1e-13 relative on the
//! Fock matrix, ~1e-12 Ha on the band energies). Rust-vs-Rust gates on this
//! path therefore compare at `1e-9`, the same band-chain floor the upstream
//! oracle uses ([`band_energies_are_never_bitwise_identical`]).
//!
//! Generic over the device float (`F: Float`, AGENTS.md §3 / RULE 5).

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::{launch_1d, upload};
use pyscf_algebra::{AlgebraClient, AlgebraError};

use crate::pbc::AoKAccumulator;
use crate::pbc::local_vmat::{AoPlanes, KMatPlanes, gather_k};
use crate::scalar::DeviceScalar;

/// `v[k][p, q] = Σ_g conj(ao⁰[k, p, g]) · Σ_n wv[n, g] · aoⁿ[k, q, g]`, one
/// lane per `(k, p, q)`.
///
/// The planes are addressed through two strides so one kernel serves both
/// accumulator layouts (`stride_k = n, stride_e = 1` k-major;
/// `stride_k = 1, stride_e = nkpts` point-major — see
/// [`crate::pbc::local_vmat`]). The element index of component `c`, AO `mu`,
/// grid point `g` is `e = c · ngrids · nao + mu · ngrids + g`, the layout
/// `eval_ao_kpts` writes.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn band_vmat_kernel<F: Float>(
    ao_re: &Array<F>,
    ao_im: &Array<F>,
    wv: &Array<F>,
    out_re: &mut Array<F>,
    out_im: &mut Array<F>,
    nao: usize,
    ngrids: usize,
    nvar: usize,
    stride_k: usize,
    stride_e: usize,
    total: usize,
    lane0: usize,
    lanes: usize,
    accumulate: u32,
    nsplit: usize,
    #[comptime] im_zero: bool,
    #[comptime] split: bool,
) {
    // `split` (BAND-09, GPU only): `nsplit` consecutive lanes share one
    // output element, lane `s` summing grid points `s, s + nsplit, ...` into
    // partial `s` (`out[s · total + i]`), so adjacent lanes load adjacent
    // grid points. Without it one lane owns the element and walks every `g`
    // — the CPU shape, unchanged.
    let local = ABSOLUTE_POS;
    let mut limit = lanes;
    if comptime!(split) {
        limit = lanes * nsplit;
    }
    if local < limit {
        let mut i = local + lane0;
        let mut s0 = 0usize;
        let mut gstep = 1usize;
        if comptime!(split) {
            i = local / nsplit + lane0;
            s0 = local % nsplit;
            gstep = nsplit;
        }
        let npair = nao * nao;
        let k = i / npair;
        let pq = i % npair;
        let p = pq / nao;
        let q = pq % nao;
        let cstride = ngrids * nao * stride_e;
        let pb = k * stride_k + p * ngrids * stride_e;
        let qb = k * stride_k + q * ngrids * stride_e;
        let zero = F::from_int(0);
        let mut sr = F::from_int(0);
        let mut si = F::from_int(0);
        if accumulate == 1 {
            sr = out_re[i];
            si = out_im[i];
        }
        for g in range_stepped(s0, ngrids, gstep) {
            let off = g * stride_e;
            let pr = ao_re[pb + off];
            // Gamma: the host route reads a literally zeroed imaginary plane
            // (`eval_gto.py:157-158`); this variant reads the same `+0.0`
            // without touching the accumulator, so every sign of zero matches.
            let mut pim = zero;
            if comptime!(!im_zero) {
                pim = ao_im[pb + off];
            }
            let pi = -pim;
            let mut ar = F::from_int(0);
            let mut ai = F::from_int(0);
            for n in 0..nvar {
                let w = wv[n * ngrids + g];
                ar += w * ao_re[qb + n * cstride + off];
                let mut qim = zero;
                if comptime!(!im_zero) {
                    qim = ao_im[qb + n * cstride + off];
                }
                ai += w * qim;
            }
            sr += pr * ar - pi * ai;
            si += pr * ai + pi * ar;
        }
        let mut o = i;
        if comptime!(split) {
            o = s0 * total + i;
        }
        out_re[o] = sr;
        out_im[o] = si;
    }
}

/// BAND-09 — fold the `nsplit` partials of every output element, `s`
/// ascending (a fixed order, so the GPU result is deterministic).
#[cube(launch_unchecked)]
fn band_vmat_reduce_kernel<F: Float>(
    part_re: &Array<F>,
    part_im: &Array<F>,
    out_re: &mut Array<F>,
    out_im: &mut Array<F>,
    total: usize,
    nsplit: usize,
) {
    let i = ABSOLUTE_POS;
    if i < total {
        let mut sr = F::from_int(0);
        let mut si = F::from_int(0);
        for s in 0..nsplit {
            sr += part_re[s * total + i];
            si += part_im[s * total + i];
        }
        out_re[i] = sr;
        out_im[i] = si;
    }
}

/// BAND-09 — POINT-MAJOR planes (`src[e · nkpts + k]`) to K-MAJOR
/// (`dst[k · n + e]`) in one launch, so the split contraction can cover every
/// k-point at once instead of one 729-lane launch per k. Pure data movement.
#[cube(launch_unchecked)]
pub(crate) fn point_to_k_major_kernel<F: Float>(src: &Array<F>, dst: &mut Array<F>, nkpts: usize, n: usize) {
    let j = ABSOLUTE_POS;
    if j < nkpts * n {
        let k = j / n;
        let e = j % n;
        dst[j] = src[e * nkpts + k];
    }
}

/// Lanes per launch — see [`crate::pbc::local_vmat`].
const LANES_PER_LAUNCH: usize = 1 << 20;

/// One contraction launch over the lane range `[range0, range0 + range_len)`.
#[allow(clippy::too_many_arguments)]
fn launch_range<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    ao_re: &Handle,
    ao_im: &Handle,
    wv: &Handle,
    out_re: &Handle,
    out_im: &Handle,
    ao_len: usize,
    total: usize,
    nao: usize,
    ngrids: usize,
    nvar: usize,
    stride_k: usize,
    stride_e: usize,
    range0: usize,
    range_len: usize,
    accumulate: u32,
    im_zero: bool,
    nsplit: usize,
) {
    let end = range0 + range_len;
    let mut lane0 = range0;
    let split = nsplit > 1;
    let per_launch = (LANES_PER_LAUNCH / nsplit.max(1)).max(1);
    while lane0 < end {
        let lanes = (end - lane0).min(per_launch);
        // `(2·nvar + 8)` flops per grid point per lane sizes the CPU thread
        // count; passing the lane count alone would run a grid-sized
        // reduction on one thread.
        let (count, dim) = launch_1d(
            client,
            lanes * nsplit.max(1),
            (2 * nvar + 8) * ngrids.div_ceil(nsplit.max(1)),
        );
        unsafe {
            band_vmat_kernel::launch_unchecked::<F, R>(
                client,
                count,
                dim,
                // SAFETY: both planes hold `ao_len` elements and the strides
                // keep every `(k, c, q, g)` with `c < nvar <= comp` in range;
                // `wv` holds `nvar · ngrids`; each output plane holds `total`,
                // and `local < lanes` with `range0 + range_len <= total`
                // bounds every write.
                ArrayArg::from_raw_parts(ao_re.clone(), ao_len),
                ArrayArg::from_raw_parts(ao_im.clone(), ao_len),
                ArrayArg::from_raw_parts(wv.clone(), nvar * ngrids),
                ArrayArg::from_raw_parts(out_re.clone(), total * nsplit.max(1)),
                ArrayArg::from_raw_parts(out_im.clone(), total * nsplit.max(1)),
                nao,
                ngrids,
                nvar,
                stride_k,
                stride_e,
                total,
                lane0,
                lanes,
                accumulate,
                nsplit.max(1),
                im_zero,
                split,
            );
        }
        lane0 += lanes;
    }
}

/// Upload the weights, zero the gamma planes, launch, read the answer back.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<R: Runtime>(
    client: &ComputeClient<R>,
    ao_re: &Handle,
    ao_im: &Handle,
    ao_len: usize,
    stride_k: usize,
    stride_e: usize,
    wv: &[f64],
    nvar: usize,
    comp: usize,
    nkpts: usize,
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
) -> KMatPlanes {
    let npair = nao * nao;
    let total = nkpts * npair;
    let n = comp * nao * ngrids;
    let out_re = client.empty(total * core::mem::size_of::<f64>());
    let out_im = client.empty(total * core::mem::size_of::<f64>());
    let wv_h = upload::<R, f64>(client, wv);
    let has_gamma = gamma.iter().any(|&g| g);
    // Lanes per launch the split is sized against: the whole output, or one
    // k-point's block when a Γ point forces per-k launches.
    let launch_outputs = if has_gamma { npair } else { total };
    let nsplit = split_factor(client, launch_outputs, total, ngrids);
    if nsplit > 1 {
        // BAND-09: k-major planes (transposed once if point-major), every
        // output's grid sum split over `nsplit` coalesced lanes, then a
        // fixed-order fold of the partials.
        let (kre, kim, klen) = if stride_e == 1 && stride_k == n {
            (ao_re.clone(), ao_im.clone(), ao_len)
        } else {
            let len = nkpts * n;
            let dre = client.empty(len * core::mem::size_of::<f64>());
            let dim_ = client.empty(len * core::mem::size_of::<f64>());
            let (count, dim) = launch_1d(client, len, 1);
            for (src, dst) in [(ao_re, &dre), (ao_im, &dim_)] {
                unsafe {
                    point_to_k_major_kernel::launch_unchecked::<f64, R>(
                        client,
                        count.clone(),
                        dim,
                        // SAFETY: `src` holds `ao_len = nkpts · n` point-major
                        // values, `dst` `len` k-major; the kernel guards `j < len`.
                        ArrayArg::from_raw_parts(src.clone(), ao_len),
                        ArrayArg::from_raw_parts(dst.clone(), len),
                        nkpts,
                        n,
                    );
                }
            }
            (dre, dim_, len)
        };
        let part_bytes = nsplit * total * core::mem::size_of::<f64>();
        let part_re = client.empty(part_bytes);
        let part_im = client.empty(part_bytes);
        if has_gamma {
            for (k, &is_gamma) in gamma.iter().enumerate() {
                launch_range::<R, f64>(
                    client,
                    &kre,
                    &kim,
                    &wv_h,
                    &part_re,
                    &part_im,
                    klen,
                    total,
                    nao,
                    ngrids,
                    nvar,
                    n,
                    1,
                    k * npair,
                    npair,
                    0,
                    is_gamma,
                    nsplit,
                );
            }
        } else {
            launch_range::<R, f64>(
                client, &kre, &kim, &wv_h, &part_re, &part_im, klen, total, nao, ngrids, nvar, n,
                1, 0, total, 0, false, nsplit,
            );
        }
        let (count, dim) = launch_1d(client, total, nsplit);
        unsafe {
            band_vmat_reduce_kernel::launch_unchecked::<f64, R>(
                client,
                count,
                dim,
                // SAFETY: partials hold `nsplit · total`, outputs `total`; the
                // kernel guards `i < total`.
                ArrayArg::from_raw_parts(part_re.clone(), nsplit * total),
                ArrayArg::from_raw_parts(part_im.clone(), nsplit * total),
                ArrayArg::from_raw_parts(out_re.clone(), total),
                ArrayArg::from_raw_parts(out_im.clone(), total),
                total,
                nsplit,
            );
        }
    } else if stride_e == 1 {
        // K-major planes: a lane's `g` walk is already contiguous. The planes
        // are never written (the accumulator is borrowed): a gamma k-point
        // runs the `im_zero` variant over its own lane range instead.
        if gamma.iter().all(|&g| !g) {
            launch_range::<R, f64>(
                client, ao_re, ao_im, &wv_h, &out_re, &out_im, ao_len, total, nao, ngrids, nvar,
                stride_k, 1, 0, total, 0, false, 1,
            );
        } else {
            for (k, &is_gamma) in gamma.iter().enumerate() {
                launch_range::<R, f64>(
                    client,
                    ao_re,
                    ao_im,
                    &wv_h,
                    &out_re,
                    &out_im,
                    ao_len,
                    total,
                    nao,
                    ngrids,
                    nvar,
                    stride_k,
                    1,
                    k * npair,
                    npair,
                    0,
                    is_gamma,
                    1,
                );
            }
        }
    } else {
        // Point-major planes: gather each k into a contiguous scratch first —
        // the stride, not the kernel, was the whole cost in K-14f's
        // measurement (see `local_vmat`'s module docs).
        let scratch_re = client.empty(n * core::mem::size_of::<f64>());
        let scratch_im = client.empty(n * core::mem::size_of::<f64>());
        for (k, &is_gamma) in gamma.iter().enumerate() {
            gather_k::<R, f64>(client, ao_re, &scratch_re, ao_len, k, nkpts, n);
            if !is_gamma {
                gather_k::<R, f64>(client, ao_im, &scratch_im, ao_len, k, nkpts, n);
            }
            launch_range::<R, f64>(
                client,
                &scratch_re,
                &scratch_im,
                &wv_h,
                &out_re,
                &out_im,
                n,
                total,
                nao,
                ngrids,
                nvar,
                0,
                1,
                k * npair,
                npair,
                0,
                is_gamma,
                1,
            );
        }
    }
    // The read is where the lazily launched kernels execute; the span holds
    // them too.
    let _span = tracing::info_span!(
        "pbc_band_vmat",
        nkpts = nkpts as u64,
        nao = nao as u64,
        ngrids = ngrids as u64,
        nvar = nvar as u64
    )
    .entered();
    let bytes = client.read(vec![out_re, out_im]);
    let re: &[f64] = bytemuck::cast_slice(&bytes[0]);
    let im: &[f64] = bytemuck::cast_slice(&bytes[1]);
    (0..nkpts)
        .map(|k| {
            (
                re[k * npair..(k + 1) * npair].to_vec(),
                im[k * npair..(k + 1) * npair].to_vec(),
            )
        })
        .collect()
}

/// Target concurrent lanes per contraction launch on a GPU (BAND-09).
///
/// MEASURED 2026-09-24 on a Kaggle T4, KTaO3 (729 outputs per Γ-forced
/// per-k launch, 30 grid blocks per call): contraction total 6 376 ms
/// unsplit, 1 339 ms at split 256, 1 128 ms at 32, 1 024 ms at 64. `1 << 16`
/// picks 64 for that shape; [`SPLIT_MIN`] keeps a full warp per element.
const SPLIT_TARGET_LANES: usize = 1 << 16;
/// Smallest GPU split: one 32-lane warp reading 32 consecutive grid points.
const SPLIT_MIN: usize = 32;
/// Largest grid split; wider only adds partials to fold.
const SPLIT_MAX: usize = 256;
/// Device bytes the partial planes may take.
const SPLIT_PARTIAL_BUDGET: usize = 512 << 20;

/// BAND-09 — how many lanes share one output element's grid sum.
///
/// `PYSCF_PBC_BAND_VMAT_SPLIT=<n>` pins it (`1` is the unsplit kernel, the
/// CPU shape). Unset: `1` on the CPU runtime (a lane there is a thread, and
/// the serial walk is already the fast shape), and on a runtime with hardware
/// planes the power of two that brings `launch_outputs · split` to
/// [`SPLIT_TARGET_LANES`], capped by [`SPLIT_MAX`], `ngrids` and the partial
/// budget. The split changes the summation order (partials folded in fixed
/// order), so the GPU result is deterministic but not bitwise the CPU's.
fn split_factor<R: Runtime>(
    client: &ComputeClient<R>,
    launch_outputs: usize,
    total: usize,
    ngrids: usize,
) -> usize {
    let cap = |s: usize| {
        let by_mem = (SPLIT_PARTIAL_BUDGET / (16 * total.max(1))).max(1);
        s.min(ngrids.max(1)).min(by_mem).max(1)
    };
    if let Some(n) = std::env::var("PYSCF_PBC_BAND_VMAT_SPLIT")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return cap(n);
    }
    if !pyscf_algebra::launch::has_planes(client) {
        return 1;
    }
    let want = (SPLIT_TARGET_LANES / launch_outputs.max(1)).clamp(SPLIT_MIN, SPLIT_MAX);
    // Round down to a power of two so a plane's lanes share one element.
    let pow2 = 1usize << (usize::BITS - 1 - want.leading_zeros());
    cap(pow2)
}

/// The shape checks both entry points share. `Ok(Some(..))` answers the
/// degenerate (empty) case without a launch.
fn check_shapes(
    wv: &[f64],
    nvar: usize,
    comp: usize,
    nkpts: usize,
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
) -> Result<Option<KMatPlanes>, AlgebraError> {
    if nvar == 0 || nvar > comp {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("1 <= nvar <= comp = {comp}"),
            actual: nvar.to_string(),
        });
    }
    if wv.len() != nvar * ngrids {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("wv of length nvar·ngrids = {}", nvar * ngrids),
            actual: wv.len().to_string(),
        });
    }
    if gamma.len() != nkpts {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("one gamma flag per k-point, nkpts = {nkpts}"),
            actual: gamma.len().to_string(),
        });
    }
    if nkpts == 0 || nao == 0 || ngrids == 0 {
        return Ok(Some((0..nkpts).map(|_| (Vec::new(), Vec::new())).collect()));
    }
    Ok(None)
}

/// BAND-08 from HOST planes: `(nkpts, comp · nao · ngrids)` row-major per
/// plane (component-major inside a k-point, `e = c·ngrids·nao + mu·ngrids +
/// g`). `wv` is `nvar · ngrids`, `wv[n · ngrids + g]`. Returns `(re, im)` per
/// k-point, `nao · nao` reals each, ROW-MAJOR `[p · nao + q]`.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when the planes, `wv`, `gamma` or the
/// declared shape disagree, or `nvar` exceeds `comp`.
#[allow(clippy::too_many_arguments)]
pub fn band_vmat(
    client: &AlgebraClient,
    ao: &AoPlanes<'_>,
    wv: &[f64],
    nvar: usize,
    comp: usize,
    nkpts: usize,
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
) -> Result<KMatPlanes, AlgebraError> {
    let want = nkpts * comp * nao * ngrids;
    let AoPlanes { re, im } = ao;
    if re.len() != want || im.len() != want {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("both AO planes of length nkpts·comp·nao·ngrids = {want}"),
            actual: format!("re {} im {}", re.len(), im.len()),
        });
    }
    if let Some(out) = check_shapes(wv, nvar, comp, nkpts, nao, ngrids, gamma)? {
        return Ok(out);
    }
    let out = dispatch_backend!(client, c, Rt, {
        let ao_re = upload::<Rt, f64>(c, re);
        let ao_im = upload::<Rt, f64>(c, im);
        run::<Rt>(
            c,
            &ao_re,
            &ao_im,
            want,
            comp * nao * ngrids,
            1,
            wv,
            nvar,
            comp,
            nkpts,
            nao,
            ngrids,
            gamma,
        )
    });
    Ok(out)
}

/// BAND-08 on an [`AoKAccumulator`] still resident on the device — the
/// route `band_vmats` takes. The accumulator is BORROWED and never written
/// (several weight sets contract the same table; a gamma k-point's imaginary
/// plane is ignored by the kernel rather than zeroed in place); none of the
/// table is read back, only the `nkpts · nao²` answer.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when the accumulator's `n` is not
/// `comp · nao · ngrids`, or as [`band_vmat`].
#[allow(clippy::too_many_arguments)]
pub fn band_vmat_resident(
    client: &AlgebraClient,
    acc: &AoKAccumulator,
    wv: &[f64],
    nvar: usize,
    comp: usize,
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
) -> Result<KMatPlanes, AlgebraError> {
    let (nkpts, n) = acc.shape();
    if n != comp * nao * ngrids {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("accumulator n = comp·nao·ngrids = {}", comp * nao * ngrids),
            actual: n.to_string(),
        });
    }
    if let Some(out) = check_shapes(wv, nvar, comp, nkpts, nao, ngrids, gamma)? {
        return Ok(out);
    }
    let (stride_k, stride_e) = if acc.is_point_major() {
        (1, nkpts)
    } else {
        (n, 1)
    };
    let (re, im) = acc.planes();
    let (re, im) = (re.clone(), im.clone());
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        run::<Rt>(
            c,
            &re,
            &im,
            nkpts * n,
            stride_k,
            stride_e,
            wv,
            nvar,
            comp,
            nkpts,
            nao,
            ngrids,
            gamma,
        )
    );
    Ok(out)
}

/// SCF-02 — BAND-08's contraction on a device-resident, k-major AO table
/// ([`crate::pbc::DeviceAoTable`]): every k-point of the table at once, no
/// upload of the table. Γ imaginary planes were zeroed when the table was
/// built, so no k is special here.
///
/// # Errors
/// As [`band_vmat`].
pub fn band_vmat_table(
    client: &AlgebraClient,
    table: &crate::pbc::DeviceAoTable,
    wv: &[f64],
    nvar: usize,
) -> Result<KMatPlanes, AlgebraError> {
    let (nkpts, ncomp, nao, ngrids) = table.dims();
    let gamma = vec![false; nkpts];
    if let Some(out) = check_shapes(wv, nvar, ncomp, nkpts, nao, ngrids, &gamma)? {
        return Ok(out);
    }
    let n = ncomp * nao * ngrids;
    let (re, im) = table.planes();
    let (re, im) = (re.clone(), im.clone());
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        run::<Rt>(c, &re, &im, nkpts * n, n, 1, wv, nvar, ncomp, nkpts, nao, ngrids, &gamma)
    );
    Ok(out)
}
