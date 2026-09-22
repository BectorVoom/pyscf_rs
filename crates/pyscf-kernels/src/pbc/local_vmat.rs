//! K-14f — the local-potential contraction, fused onto the device so the AO
//! table never reaches the host.
//!
//! ```text
//! v[k][p, q] = Σ_g conj(ao_k[p, g]) · vR[g] · ao_k[q, g]
//! ```
//!
//! This is the `lib.dot(ao.T.conj() * vR, ao)` of `pyscf/pbc/df/fft.py:71`
//! (`get_nuc`) and `:110` (`get_pp`) — the one thing `get_hcore`'s local half
//! does with the AO table it evaluates.
//!
//! # Why this kernel exists at all
//!
//! The contraction is a REDUCTION: it consumes `nkpts · nao · ngrids` complex
//! AO values and produces `nkpts · nao²` complex matrix elements. On a 2×2×2
//! diamond cell at its default mesh that is 8 · 26 · 29 791 complex numbers in
//! and 8 · 676 out — a 1 100× shrink. Evaluating the table on the device,
//! reading all of it back, and reducing it on the host therefore pays the
//! table's bytes THREE times over at peak:
//!
//! | where | bytes |
//! |---|---|
//! | [`crate::pbc::AoKAccumulator`]'s two device planes | `16 · nkpts · nao · ngrids` |
//! | the `client.read` staging buffer | `16 · nkpts · nao · ngrids` |
//! | the per-k host planes it is split into | `16 · nkpts · nao · ngrids` |
//!
//! All three are live at once inside
//! [`crate::pbc::AoKAccumulator::into_k_planes`], and the third then survives
//! the call (`FFTDF` caches it). Contracting on the device instead reads the
//! planes where they already are and brings home `16 · nkpts · nao²` — the
//! answer, not the intermediate — so only the first line of that table is ever
//! allocated.
//!
//! On a discrete backend that removes the table from host memory outright. On
//! CubeCL's CPU runtime — the default here (ALG-03) — "device" memory is the
//! same RAM, so the accumulator's one copy is the floor and the saving is the
//! other two: a ~3× cut in the call's peak, not an unbounded one. Driving the
//! grid in blocks cuts the remaining copy too: [`CarriedVmat`] (B-03) carries
//! this contraction across grid blocks, so the device holds one block's planes
//! at a time, at the cost of re-running the lattice-image loop per block.
//!
//! # Why not a GEMM
//!
//! `Σ_g conj(ao_p) w ao_q` is formally `zgemm('N', 'T')`, and `pyscf-algebra`
//! has a dense complex GEMM. It is the wrong engine for this shape: the
//! reduction axis (`ngrids`, tens of thousands) is enormous next to the output
//! (`nao`, tens), which is the regime where the dense route loses badly to a
//! per-output-element reduction — and it would want `conj(ao) · vR`
//! materialised as a second full-size table first, which is exactly the
//! allocation this module exists to avoid.
//!
//! # Bit-exactness
//!
//! One lane owns one `(k, p, q)` element and walks `g` upwards in a serial
//! loop, accumulating
//!
//! ```text
//! sr += (pr·qr − pi·qi) · w
//! si += (pr·qi + pi·qr) · w
//! ```
//!
//! — the same operations, in the same order, with the same associativity as
//! the host loop in `pyscf_pbc_df::fftdf::Fftdf::contract_local_potential`
//! that it replaces. No partial sums are merged and no lane sees another
//! lane's grid points, so the result is bit-identical on every backend with
//! IEEE-754 doubles.
//!
//! The one place the two routes could still diverge is the gamma point, where
//! `eval_ao_kpts` drops the imaginary plane (`eval_gto.py:157-158`) by handing
//! the host a freshly zeroed `Vec` — so the host contraction reads `+0.0`
//! there, while the accumulator still holds the lattice sum's roundoff
//! residue. [`crate::pbc::fill::fill_zero_kernel`] writes the same literal `0.0` over that plane
//! on the device before the contraction runs, so the sign of every zero
//! matches too. Multiplying the plane by `0.0` instead would NOT: that
//! preserves the residue's sign bit, and `pr·(−0.0) + (−0.0)·qr` is `−0.0`
//! where the host gets `+0.0`.
//!
//! # The point-major detour
//!
//! [`crate::pbc::AoKAccumulator`] has two layouts, and the fast one for the AO
//! evaluation is the slow one for this reduction. The K-10v fused AO path
//! accumulates POINT-MAJOR (`plane[e·nkpts + k]`, so its k-loop vectorises),
//! which puts successive `g` of one AO `nkpts` apart — a lane walking `g` then
//! touches a new cache line on every load. Measured on diamond gth-dzvp 3×3×3
//! at mesh 41, contracting those planes in place took 12.3 s against the host
//! route's 6.7 s, while the same kernel over k-major planes took 13.1 s
//! against a 13.9 s host route: the stride was the entire difference.
//!
//! So a point-major table is contracted one k at a time, each k first gathered
//! into a contiguous `n`-element scratch ([`gather_k_kernel`]). The scratch is
//! one k-point's worth — `1/nkpts` of the table — and a gather copies values
//! without combining them, so the numbers are untouched.
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
use crate::scalar::DeviceScalar;

/// One `nao x nao` complex matrix per k-point: `(re, im)` planes, each
/// `nao · nao` reals in ROW-MAJOR `[p · nao + q]` order — the layout
/// `pyscf_pbc_df` builds its `CTensor`s from.
pub type KMatPlanes = Vec<(Vec<f64>, Vec<f64>)>;

/// `v[k][p, q] = Σ_g conj(ao[k, p, g]) · vr[g] · ao[k, q, g]`, one lane per
/// `(k, p, q)`.
///
/// The AO planes are addressed through two strides rather than a fixed layout,
/// so one kernel serves both accumulator shapes: `stride_k = n, stride_e = 1`
/// for the k-major planes (`plane[k·n + e]`) and `stride_k = 1,
/// stride_e = nkpts` for the point-major ones the fused K-10 path writes
/// (`plane[e·nkpts + k]`), where `e = mu · ngrids + g`.
///
/// The `local < lanes` guard is required: the launch rounds the lane count up
/// to a whole number of cubes, so tail lanes must not write out of range.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn local_vmat_kernel<F: Float>(
    ao_re: &Array<F>,
    ao_im: &Array<F>,
    vr: &Array<F>,
    out_re: &mut Array<F>,
    out_im: &mut Array<F>,
    nao: usize,
    ngrids: usize,
    stride_k: usize,
    stride_e: usize,
    lane0: usize,
    lanes: usize,
    accumulate: u32,
) {
    let local = ABSOLUTE_POS;
    if local < lanes {
        let i = local + lane0;
        let npair = nao * nao;
        let k = i / npair;
        let pq = i % npair;
        let p = pq / nao;
        let q = pq % nao;
        // First element of AO `p` (resp. `q`) at k-point `k`.
        let pb = k * stride_k + p * ngrids * stride_e;
        let qb = k * stride_k + q * ngrids * stride_e;
        // `accumulate == 1` carries the sum across grid blocks (B-03): the lane
        // LOADS the running sum, extends the SAME serial chain over this
        // block's grid points, and STORES it back. §2.4 requires one carried
        // accumulator — adding a from-zero block sum into the output instead
        // reassociates and moves the last bits (verified: serial vs block-+=
        // differ in the last hex digit on 1000 random terms, load-store is
        // exact). The plan's B-03b sketch shows `+=`; this is the §2.4 form.
        let mut sr = F::from_int(0);
        let mut si = F::from_int(0);
        if accumulate == 1 {
            sr = out_re[i];
            si = out_im[i];
        }
        for g in 0..ngrids {
            let off = g * stride_e;
            // conj(ao[p, g]) · ao[q, g] · vR[g]
            let pr = ao_re[pb + off];
            let pi = -ao_im[pb + off];
            let qr = ao_re[qb + off];
            let qi = ao_im[qb + off];
            let w = vr[g];
            sr += (pr * qr - pi * qi) * w;
            si += (pr * qi + pi * qr) * w;
        }
        out_re[i] = sr;
        out_im[i] = si;
    }
}

/// Copy one k-point's `n` elements out of a POINT-MAJOR plane
/// (`plane[e·nkpts + k]`) into a contiguous scratch buffer.
///
/// Pure data movement — every value is copied, none combined — so it cannot
/// change a bit of the contraction that follows.
#[cube(launch_unchecked)]
fn gather_k_kernel<F: Float>(src: &Array<F>, dst: &mut Array<F>, k: usize, nkpts: usize, n: usize) {
    let i = ABSOLUTE_POS;
    if i < n {
        dst[i] = src[i * nkpts + k];
    }
}

/// Lanes per contraction launch.
///
/// The kernel holds no `Array::new` locals, so there is no per-iteration stack
/// growth to chunk around — [`pyscf_algebra::launch::launch_1d_chunked`]'s
/// reason for existing does not apply here. The range is still cut into
/// launches so that a very large `nkpts · nao²` cannot ask a backend for a
/// cube count it will not take.
const LANES_PER_LAUNCH: usize = 1 << 20;

#[allow(clippy::too_many_arguments)]
fn launch_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    ao_re: &Handle,
    ao_im: &Handle,
    vr: &Handle,
    out_re: &Handle,
    out_im: &Handle,
    ao_len: usize,
    nkpts: usize,
    nao: usize,
    ngrids: usize,
    stride_k: usize,
    stride_e: usize,
    accumulate: u32,
) {
    let total = nkpts * nao * nao;
    launch_range::<R, F>(
        client, ao_re, ao_im, vr, out_re, out_im, ao_len, total, nao, ngrids, stride_k, stride_e,
        0, total, accumulate,
    );
}

/// [`launch_on_handles`] over ONE lane range `[range0, range0 + range_len)`,
/// cut into launches of at most [`LANES_PER_LAUNCH`].
#[allow(clippy::too_many_arguments)]
fn launch_range<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    ao_re: &Handle,
    ao_im: &Handle,
    vr: &Handle,
    out_re: &Handle,
    out_im: &Handle,
    ao_len: usize,
    total: usize,
    nao: usize,
    ngrids: usize,
    stride_k: usize,
    stride_e: usize,
    range0: usize,
    range_len: usize,
    accumulate: u32,
) {
    let end = range0 + range_len;
    let mut lane0 = range0;
    while lane0 < end {
        let lanes = (end - lane0).min(LANES_PER_LAUNCH);
        // Eight flops and five loads per grid point, per lane — the per-lane
        // work `launch_1d` sizes the CPU thread count from. Passing the lane
        // count alone would under-count it by `ngrids` and run a grid-sized
        // reduction on a single thread.
        let (count, dim) = launch_1d(client, lanes, 8 * ngrids);
        unsafe {
            local_vmat_kernel::launch_unchecked::<F, R>(
                client,
                count,
                dim,
                // SAFETY: both AO planes hold `ao_len` elements, and `run`
                // pairs `ao_len` with strides that keep every `(k, p, g)` this
                // forms in range — `nkpts · nao · ngrids` with `(n, 1)` or
                // `(1, nkpts)` for the whole table, `nao · ngrids` with
                // `(0, 1)` for one k's gathered scratch; `vr` holds `ngrids`;
                // each output plane holds `total`, and `local < lanes` with
                // `range0 + range_len <= total` bounds every write.
                ArrayArg::from_raw_parts(ao_re.clone(), ao_len),
                ArrayArg::from_raw_parts(ao_im.clone(), ao_len),
                ArrayArg::from_raw_parts(vr.clone(), ngrids),
                ArrayArg::from_raw_parts(out_re.clone(), total),
                ArrayArg::from_raw_parts(out_im.clone(), total),
                nao,
                ngrids,
                stride_k,
                stride_e,
                lane0,
                lanes,
                accumulate,
            );
        }
        lane0 += lanes;
    }
}

fn gather_k<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    src: &Handle,
    dst: &Handle,
    src_len: usize,
    k: usize,
    nkpts: usize,
    n: usize,
) {
    let (count, dim) = launch_1d(client, n, 1);
    unsafe {
        gather_k_kernel::launch_unchecked::<F, R>(
            client,
            count,
            dim,
            // SAFETY: `(n - 1) * nkpts + k < src_len = nkpts * n` for `k < nkpts`,
            // `dst` holds `n`, and the kernel guards `i < n`.
            ArrayArg::from_raw_parts(src.clone(), src_len),
            ArrayArg::from_raw_parts(dst.clone(), n),
            k,
            nkpts,
            n,
        );
    }
}

fn zero_gamma_plane<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    im: &Handle,
    ao_len: usize,
    base: usize,
    stride: usize,
    n: usize,
) {
    crate::pbc::fill::fill_range::<R, F>(client, im, ao_len, base, stride, n);
}

/// A HOST-resident AO table, `(nkpts, nao · ngrids)` row-major per plane.
///
/// [`local_vmat`] uploads it and reduces it on the device — for a caller that
/// already has a materialised table (`FFTDF`'s AO cache, or a test comparing
/// the two routes) and wants the reduction on the device anyway. The
/// device-resident route, which is the point of this module, is
/// [`local_vmat_resident`]: it never has a host table to pass here.
pub struct AoPlanes<'a> {
    /// Real plane, `nkpts · nao · ngrids` reals.
    pub re: &'a [f64],
    /// Imaginary plane, same length.
    pub im: &'a [f64],
}

/// Contract a REAL local potential on the grid into one `nao × nao` complex
/// matrix per k-point, on the device, from HOST AO planes.
///
/// Returns `(re, im)` per k-point, each `nao · nao` reals in ROW-MAJOR
/// `[p · nao + q]` order — the layout `pyscf_pbc_df` builds its `CTensor`s
/// from. `gamma` flags the k-points whose imaginary plane is dropped
/// (`eval_gto.py:157-158`); see the module docs.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when the planes, `vr`, `gamma` or the
/// declared `(nkpts, nao, ngrids)` shape disagree.
pub fn local_vmat(
    client: &AlgebraClient,
    ao: &AoPlanes<'_>,
    vr: &[f64],
    nkpts: usize,
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
) -> Result<KMatPlanes, AlgebraError> {
    let want = nkpts * nao * ngrids;
    let AoPlanes { re, im } = ao;
    if re.len() != want || im.len() != want {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("both AO planes of length nkpts·nao·ngrids = {want}"),
            actual: format!("re {} im {}", re.len(), im.len()),
        });
    }
    if let Some(out) = check_shapes(vr, nkpts, nao, ngrids, gamma)? {
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
            nao * ngrids,
            1,
            vr,
            nkpts,
            nao,
            ngrids,
            gamma,
            0, // overwrite mode — B-03c passes 1 for carried blocks
        )
    });
    Ok(out)
}

/// K-14f on a [`crate::pbc::AoKAccumulator`] still resident on the device — the
/// memory-efficient route `get_hcore`'s local half takes.
///
/// The accumulator is CONSUMED and its planes are contracted where they lie:
/// none of the `nkpts · nao · ngrids` AO table is read back, and the only
/// transfer home is the `nkpts · nao²` answer. Compare
/// [`crate::pbc::AoKAccumulator::into_k_planes`], which brings the whole table
/// across — and allocates two further copies of it on the way — before the
/// caller reduces it.
///
/// Both accumulator layouts are accepted; see [`local_vmat_kernel`] for how
/// the strides encode them.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when the accumulator's `n` is not
/// `nao · ngrids` (a `deriv1` table carries four components and has no
/// contraction of this shape), or when `vr` or `gamma` disagree with it.
pub fn local_vmat_resident(
    client: &AlgebraClient,
    acc: AoKAccumulator,
    vr: &[f64],
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
) -> Result<KMatPlanes, AlgebraError> {
    let (nkpts, n) = acc.shape();
    if n != nao * ngrids {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("accumulator n = nao·ngrids = {}", nao * ngrids),
            actual: n.to_string(),
        });
    }
    if let Some(out) = check_shapes(vr, nkpts, nao, ngrids, gamma)? {
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
            vr,
            nkpts,
            nao,
            ngrids,
            gamma,
            0, // overwrite mode — B-03c passes 1 for carried blocks
        )
    );
    Ok(out)
}

/// The shape checks both entry points share.
///
/// `Ok(Some(..))` is the degenerate case — an empty grid, basis or k-list —
/// answered without a launch.
fn check_shapes(
    vr: &[f64],
    nkpts: usize,
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
) -> Result<Option<KMatPlanes>, AlgebraError> {
    if vr.len() != ngrids {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("vr of length ngrids = {ngrids}"),
            actual: vr.len().to_string(),
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

/// Upload `vr`, zero the gamma planes, and launch the contraction into
/// `out_re`/`out_im` — the launch half of [`run`], shared with
/// [`CarriedVmat::accumulate_block`] (B-03c).
///
/// `blk_len` is the grid count this launch covers: `ngrids` for a whole-table
/// call, the block width for a blocked one. `vr` holds exactly those entries;
/// `out_*` hold `nkpts · nao²` in both cases.
#[allow(clippy::too_many_arguments)]
fn contract_into<R: Runtime>(
    client: &ComputeClient<R>,
    ao_re: &Handle,
    ao_im: &Handle,
    ao_len: usize,
    stride_k: usize,
    stride_e: usize,
    vr: &[f64],
    out_re: &Handle,
    out_im: &Handle,
    nkpts: usize,
    nao: usize,
    blk_len: usize,
    gamma: &[bool],
    accumulate: u32,
) {
    let n = nao * blk_len;
    let vr_h = upload::<R, f64>(client, vr);
    let npair = nao * nao;
    let total = nkpts * npair;

    if stride_e == 1 {
        // K-major planes: a lane's `g` walk is already contiguous.
        for (k, &is_gamma) in gamma.iter().enumerate() {
            if is_gamma {
                zero_gamma_plane::<R, f64>(client, ao_im, ao_len, k * stride_k, 1, n);
            }
        }
        launch_on_handles::<R, f64>(
            client, ao_re, ao_im, &vr_h, out_re, out_im, ao_len, nkpts, nao, blk_len, stride_k, 1,
            accumulate,
        );
    } else {
        // POINT-MAJOR planes (the K-10v fused AO path): successive `g` of one
        // AO are `nkpts` apart, so a lane that walks `g` touches a new cache
        // line on every single load. MEASURED on diamond gth-dzvp 3x3x3 at
        // mesh 41: contracting in place cost 12.3 s against the host route's
        // 6.7 s, while the same contraction over k-major planes ran in 13.1 s
        // against a 13.9 s host route — i.e. the stride, not the kernel, was
        // the whole difference.
        //
        // So each k is gathered into a contiguous `n`-element scratch first
        // and contracted with `stride_e = 1`. The scratch is ONE k-point's
        // worth (`2 · n` reals, `1/nkpts` of the table), and the gather is
        // pure data movement, so nothing about the result changes — the
        // `stride_k = 0` below is what makes the kernel read the scratch as
        // that single k while still writing to `out[k·npair ..]`.
        let scratch_re = client.empty(n * core::mem::size_of::<f64>());
        let scratch_im = client.empty(n * core::mem::size_of::<f64>());
        for (k, &is_gamma) in gamma.iter().enumerate() {
            gather_k::<R, f64>(client, ao_re, &scratch_re, ao_len, k, nkpts, n);
            gather_k::<R, f64>(client, ao_im, &scratch_im, ao_len, k, nkpts, n);
            if is_gamma {
                zero_gamma_plane::<R, f64>(client, &scratch_im, n, 0, 1, n);
            }
            launch_range::<R, f64>(
                client,
                &scratch_re,
                &scratch_im,
                &vr_h,
                out_re,
                out_im,
                n,
                total,
                nao,
                blk_len,
                0,
                1,
                k * npair,
                npair,
                accumulate,
            );
        }
    }
}

/// Zero the gamma planes, launch the contraction, read the answer back.
#[allow(clippy::too_many_arguments)]
fn run<R: Runtime>(
    client: &ComputeClient<R>,
    ao_re: &Handle,
    ao_im: &Handle,
    ao_len: usize,
    stride_k: usize,
    stride_e: usize,
    vr: &[f64],
    nkpts: usize,
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
    accumulate: u32,
) -> KMatPlanes {
    let npair = nao * nao;
    let total = nkpts * npair;
    // Every one of the `total` slots is written by exactly one lane, so an
    // uninitialised allocation is sound here — unlike the accumulating
    // kernels, which must start from a zeroed buffer.
    let out_re = client.empty(total * core::mem::size_of::<f64>());
    let out_im = client.empty(total * core::mem::size_of::<f64>());
    // B-03b: in carry mode the first block LOADS these outputs as its running
    // sums, so they must start zeroed — `fill_zero`, not unwritten `empty` (T4).
    if accumulate == 1 {
        crate::pbc::fill::fill_zero::<R, f64>(client, &out_re, total);
        crate::pbc::fill::fill_zero::<R, f64>(client, &out_im, total);
    }
    contract_into::<R>(
        client, ao_re, ao_im, ao_len, stride_k, stride_e, vr, &out_re, &out_im, nkpts, nao, ngrids,
        gamma, accumulate,
    );
    // The read is where the lazily launched kernels actually execute, so this
    // span holds them too.
    let _span = tracing::info_span!(
        "pbc_local_vmat",
        nkpts = nkpts as u64,
        nao = nao as u64,
        ngrids = ngrids as u64
    )
    .entered();
    let bytes = client.read(vec![out_re, out_im]);
    let re: &[f64] = bytemuck::cast_slice(&bytes[0]);
    let im: &[f64] = bytemuck::cast_slice(&bytes[1]);
    let npair = nao * nao;
    (0..nkpts)
        .map(|k| {
            (
                re[k * npair..(k + 1) * npair].to_vec(),
                im[k * npair..(k + 1) * npair].to_vec(),
            )
        })
        .collect()
}

/// B-03c — the device-carried `nkpts · nao²` output of a grid-blocked
/// `local_vmat`.
///
/// One lane owns one `(k, p, q)` element, and every block's kernel extends the
/// SAME serial chain: it LOADS the running sum, adds this block's grid points
/// in increasing `g` order, and STORES it back (§2.4 — one carried
/// accumulator, never per-block-then-merged). `finish` therefore returns the
/// bit-identical whole-grid contraction while peak device memory holds one
/// block's AO planes, never the whole table.
///
/// The buffers are cubecl `Handle`s, but only as PRIVATE fields — the
/// `AoKAccumulator` precedent, so the algebra wall (ALG-06) holds.
///
/// Blocks must arrive in increasing `g` order through [`accumulate_block`],
/// serially — the `&mut self` receiver makes a parallel block loop a compile
/// error. See `eval_ao_kpts_local_vmat_blocked`.
pub struct CarriedVmat {
    re: Handle,
    im: Handle,
    nkpts: usize,
    nao: usize,
    ngrids: usize,
}

impl CarriedVmat {
    /// Zeroed output planes for `(nkpts, nao)` over `ngrids` grid points.
    pub fn new(client: &AlgebraClient, nkpts: usize, nao: usize, ngrids: usize) -> Self {
        // `max(1)`: the same degenerate-guard rationale as
        // `AoKAccumulator::zeros`. T4 — `empty` recycles dirty buffers, so the
        // fill is mandatory, not an optimisation: the first block LOADS these
        // zeros as its running sums.
        let total = (nkpts * nao * nao).max(1);
        let bytes = total * core::mem::size_of::<f64>();
        let (re, im) = dispatch_backend!(client, c, Rt, {
            let re = c.empty(bytes);
            let im = c.empty(bytes);
            crate::pbc::fill::fill_zero::<Rt, f64>(c, &re, total);
            crate::pbc::fill::fill_zero::<Rt, f64>(c, &im, total);
            (re, im)
        });
        Self {
            re,
            im,
            nkpts,
            nao,
            ngrids,
        }
    }

    /// Fold one grid block's resident accumulator into the carried sums.
    /// `vr_block` is that block's `vr[g0..g0+len]`; the block accumulator is
    /// CONSUMED, so peak holds one block's planes at a time.
    ///
    /// # Errors
    /// [`AlgebraError::ShapeMismatch`] when the accumulator's width is not
    /// `nao · vr_block.len()`, the block range leaves `ngrids`, or `gamma`
    /// disagrees with `nkpts`.
    pub fn accumulate_block(
        &mut self,
        client: &AlgebraClient,
        acc: AoKAccumulator,
        g0: usize,
        vr_block: &[f64],
        gamma: &[bool],
    ) -> Result<(), AlgebraError> {
        let blk_len = vr_block.len();
        let (ankpts, n) = acc.shape();
        if ankpts != self.nkpts || n != self.nao * blk_len || blk_len == 0 {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!(
                    "block accumulator (nkpts, nao·blk>0) = ({}, {}·{blk_len})",
                    self.nkpts, self.nao,
                ),
                actual: format!("({ankpts}, {n})"),
            });
        }
        if gamma.len() != self.nkpts {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("one gamma flag per k-point, nkpts = {}", self.nkpts),
                actual: gamma.len().to_string(),
            });
        }
        if g0.saturating_add(blk_len) > self.ngrids {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("block [g0, g0+blk) within ngrids = {}", self.ngrids),
                actual: format!("[{g0}, {})", g0.saturating_add(blk_len)),
            });
        }
        let (stride_k, stride_e) = if acc.is_point_major() {
            (1, self.nkpts)
        } else {
            (n, 1)
        };
        let (re, im) = acc.planes();
        let (re, im) = (re.clone(), im.clone());
        // Peak intent: the block's planes are queued into the launches below
        // and released here; the barrier at the end of the dispatch returns
        // them to the pool before the next block allocates.
        drop(acc);
        dispatch_backend!(client, c, Rt, {
            contract_into::<Rt>(
                c,
                &re,
                &im,
                self.nkpts * n,
                stride_k,
                stride_e,
                vr_block,
                &self.re,
                &self.im,
                self.nkpts,
                self.nao,
                blk_len,
                gamma,
                1, // carry mode — §2.4, one carried accumulator
            );
            // Execution barrier: `read` syncs the stream, so this block's
            // kernels complete and its AO planes return to the pool BEFORE the
            // next block allocates. Without it every block's planes stay
            // referenced until `finish` and blocking would save no memory. The
            // payload is the `nkpts·nao²` running sums — read for the sync and
            // discarded; `finish` reads them for real.
            let _ = c.read(vec![self.re.clone(), self.im.clone()]);
        });
        Ok(())
    }

    /// Read the carried sums home: one `(re, im)` pair of `nao · nao` reals
    /// per k-point, the same layout [`local_vmat`] returns.
    pub fn finish(self, client: &AlgebraClient) -> KMatPlanes {
        let npair = self.nao * self.nao;
        let bytes = dispatch_backend!(
            client,
            c,
            Rt,
            c.read(vec![self.re.clone(), self.im.clone()])
        );
        // The read is where the lazily launched kernels actually execute, so
        // this span holds them too.
        let _span = tracing::info_span!(
            "pbc_local_vmat",
            nkpts = self.nkpts as u64,
            nao = self.nao as u64,
            ngrids = self.ngrids as u64
        )
        .entered();
        let re: &[f64] = bytemuck::cast_slice(&bytes[0]);
        let im: &[f64] = bytemuck::cast_slice(&bytes[1]);
        (0..self.nkpts)
            .map(|k| {
                (
                    re[k * npair..(k + 1) * npair].to_vec(),
                    im[k * npair..(k + 1) * npair].to_vec(),
                )
            })
            .collect()
    }
}
