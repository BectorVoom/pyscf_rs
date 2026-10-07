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
//! # SCF-03 — the tiled route
//!
//! One lane per output re-reads its two AO rows for every element: `2 + 3 ·
//! nvar` loads per grid point per `(p, q)`. At `nao = 910` and `nvar = 4`
//! that is 4 TB of device memory traffic for one 4 992-point block, and it —
//! not the AO evaluation — was the SCF cycle on a Kaggle T4 (2026-10-01:
//! 44 s per XC block, scaling with `nao²`). The tiled route is a GEMM:
//! [`aow_kernel`] materialises `aow[k]` once (one component of one k-point
//! of the table), and a [`band_vmat_tile_kernel`] lane owns a
//! [`VMAT_TILE`]`×`[`VMAT_TILE`] block of outputs, so each loaded value feeds
//! four outputs — one load per output per grid point instead of fourteen.
//! The `nsplit` lanes of a tile read consecutive grid points of the same
//! rows, so every load is coalesced.
//!
//! MEASURED 2026-10-07 on a Kaggle T4 (`grid_contract_bench`, `nao = 910`,
//! 4 992 points, 9 k-points, `nvar = 4` — one XC block of that run): 34.9 s
//! per-output, 2.45 s tiled; the Coulomb block (16 003 points, `nvar = 1`)
//! 13.7 s against 7.0 s. On the 16-core CPU runtime the XC shape at 3
//! k-points went 17.6 s to 2.0 s.
//!
//! Each output still adds `conj(ao⁰) · aow` over its grid points in the order
//! of [`band_vmat_kernel`] at the same split, and `aow` is formed by the same
//! expression — the same operations in the same order, so at one split the
//! two routes are bitwise identical on the CPU runtime. The partials are
//! folded launch by launch, so their size no longer caps the split.
//!
//! Generic over the device float (`F: Float`, AGENTS.md §3 / RULE 5).

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::{launch_1d, launch_1d_chunked, upload};
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

/// AO rows per side of one [`band_vmat_tile_kernel`] lane's output tile.
const VMAT_TILE: usize = 4;

/// SCF-03 — `aow[kk][q, g] = Σ_{n<nvar} wv[n][g] · aoⁿ[kk, q, g]` for `kn`
/// k-points of K-MAJOR planes (k-point `kk`'s block starts at `base + kk ·
/// kstride`), one lane per `(kk, q, g)`; `aow_*[kk · nao · ngrids + q ·
/// ngrids + g]`. The expression is [`band_vmat_kernel`]'s.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn aow_kernel<F: Float>(
    ao_re: &Array<F>,
    ao_im: &Array<F>,
    wv: &Array<F>,
    aow_re: &mut Array<F>,
    aow_im: &mut Array<F>,
    nao: usize,
    ngrids: usize,
    nvar: usize,
    base: usize,
    kstride: usize,
    kn: usize,
    #[comptime] im_zero: bool,
) {
    let i = ABSOLUTE_POS;
    let cstride = nao * ngrids;
    if i < kn * cstride {
        let r = i % cstride;
        let g = r % ngrids;
        let kb = base + (i / cstride) * kstride + r;
        let zero = F::from_int(0);
        let mut ar = F::from_int(0);
        let mut ai = F::from_int(0);
        for n in 0..nvar {
            let w = wv[n * ngrids + g];
            ar += w * ao_re[kb + n * cstride];
            let mut qim = zero;
            if comptime!(!im_zero) {
                qim = ao_im[kb + n * cstride];
            }
            ai += w * qim;
        }
        aow_re[i] = ar;
        aow_im[i] = ai;
    }
}

/// SCF-03 — `v[kk][p, q] = Σ_g conj(ao⁰[kk, p, g]) · aow[kk][q, g]` for the
/// tile rows `[row0, row0 + rows)` of `kn` k-points, one lane per
/// `(kk, tile, s)`: lane `s` sums grid points `s, s + nsplit, ...` of its
/// tile's sixteen outputs into `part[s · kn · kout + kk · kout + (p − row0 ·
/// VMAT_TILE) · nao + q]`, `kout = rows · VMAT_TILE · nao`. A tile hanging
/// over the edge reads row `nao − 1` again and stores nothing for it.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn band_vmat_tile_kernel<F: Float>(
    ao_re: &Array<F>,
    ao_im: &Array<F>,
    aow_re: &Array<F>,
    aow_im: &Array<F>,
    part_re: &mut Array<F>,
    part_im: &mut Array<F>,
    nao: usize,
    ngrids: usize,
    base: usize,
    kstride: usize,
    kn: usize,
    row0: usize,
    rows: usize,
    nsplit: usize,
    lane0: usize,
    #[comptime] im_zero: bool,
) {
    // `lane0`: chunked on the CPU runtime — the tile is stack per iteration
    // there (`launch_1d_chunked`).
    let tid = ABSOLUTE_POS + lane0;
    let ntq = (nao + VMAT_TILE - 1) / VMAT_TILE;
    let ktiles = rows * ntq;
    if tid < kn * ktiles * nsplit {
        let s = tid % nsplit;
        let t = tid / nsplit;
        let kk = t / ktiles;
        let tt = t % ktiles;
        let p0 = (row0 + tt / ntq) * VMAT_TILE;
        let q0 = (tt % ntq) * VMAT_TILE;
        let pbase = base + kk * kstride;
        let wbase = kk * nao * ngrids;
        let zero = F::from_int(0);
        let mut sr = Array::<F>::new(VMAT_TILE * VMAT_TILE);
        let mut si = Array::<F>::new(VMAT_TILE * VMAT_TILE);
        let mut pr = Array::<F>::new(VMAT_TILE);
        let mut pi = Array::<F>::new(VMAT_TILE);
        let mut ar = Array::<F>::new(VMAT_TILE);
        let mut ai = Array::<F>::new(VMAT_TILE);
        #[unroll]
        for i in 0..VMAT_TILE * VMAT_TILE {
            sr[i] = zero;
            si[i] = zero;
        }
        for g in range_stepped(s, ngrids, nsplit) {
            #[unroll]
            for a in 0..VMAT_TILE {
                let mut p = p0 + a;
                if p >= nao {
                    p = nao - 1;
                }
                pr[a] = ao_re[pbase + p * ngrids + g];
                // Gamma: `+0.0` without touching the plane, as
                // `band_vmat_kernel` reads it.
                let mut pim = zero;
                if comptime!(!im_zero) {
                    pim = ao_im[pbase + p * ngrids + g];
                }
                pi[a] = -pim;
                let mut q = q0 + a;
                if q >= nao {
                    q = nao - 1;
                }
                ar[a] = aow_re[wbase + q * ngrids + g];
                ai[a] = aow_im[wbase + q * ngrids + g];
            }
            #[unroll]
            for a in 0..VMAT_TILE {
                #[unroll]
                for b in 0..VMAT_TILE {
                    sr[a * VMAT_TILE + b] += pr[a] * ar[b] - pi[a] * ai[b];
                    si[a * VMAT_TILE + b] += pr[a] * ai[b] + pi[a] * ar[b];
                }
            }
        }
        let kout = rows * VMAT_TILE * nao;
        #[unroll]
        for a in 0..VMAT_TILE {
            #[unroll]
            for b in 0..VMAT_TILE {
                let p = p0 + a;
                let q = q0 + b;
                if p < nao {
                    if q < nao {
                        let o = s * kn * kout + kk * kout + (p - row0 * VMAT_TILE) * nao + q;
                        part_re[o] = sr[a * VMAT_TILE + b];
                        part_im[o] = si[a * VMAT_TILE + b];
                    }
                }
            }
        }
    }
}

/// SCF-03 — fold the `nsplit` partials of one tiled launch, `s` ascending:
/// the first `valid` elements of each of the `kn` k-points' `kout` go to
/// `out[out0 + kk · npair + i]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn band_vmat_tile_reduce_kernel<F: Float>(
    part_re: &Array<F>,
    part_im: &Array<F>,
    out_re: &mut Array<F>,
    out_im: &mut Array<F>,
    kout: usize,
    valid: usize,
    kn: usize,
    nsplit: usize,
    out0: usize,
    npair: usize,
) {
    let j = ABSOLUTE_POS;
    if j < kn * valid {
        let kk = j / valid;
        let i = j % valid;
        let src = kk * kout + i;
        let mut sr = F::from_int(0);
        let mut si = F::from_int(0);
        for s in 0..nsplit {
            sr += part_re[s * kn * kout + src];
            si += part_im[s * kn * kout + src];
        }
        out_re[out0 + kk * npair + i] = sr;
        out_im[out0 + kk * npair + i] = si;
    }
}

/// SCF-03 — k-points `[k0, k0 + kn)` of POINT-MAJOR planes (`src[e · nkpts +
/// k]`) into a K-MAJOR scratch (`dst[kk · n + e]`). Pure data movement.
#[cube(launch_unchecked)]
fn gather_k_range_kernel<F: Float>(
    src: &Array<F>,
    dst: &mut Array<F>,
    nkpts: usize,
    n: usize,
    k0: usize,
    kn: usize,
) {
    let j = ABSOLUTE_POS;
    if j < kn * n {
        dst[j] = src[(j % n) * nkpts + k0 + j / n];
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
    let tiled = tiled_contractions(client);
    let nsplit = if tiled {
        1
    } else {
        split_factor(client, launch_outputs, total, ngrids)
    };
    if tiled {
        run_tiled(
            client, ao_re, ao_im, ao_len, stride_k, stride_e, &wv_h, nvar, comp, nkpts, nao,
            ngrids, gamma, &out_re, &out_im,
        );
    } else if nsplit > 1 {
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

/// Device bytes one tiled launch's partial planes may take.
const TILE_PARTIAL_BUDGET: usize = 128 << 20;
/// Device bytes the tiled route's `aow` planes (and, for point-major planes,
/// its gathered k-points) may take — what bounds the k-points per launch.
const TILE_SCRATCH_BUDGET: usize = 256 << 20;

/// `PYSCF_PBC_GRID_TILE_BUDGET=<bytes>` pins both tiled-route budgets
/// ([`TILE_PARTIAL_BUDGET`], [`TILE_SCRATCH_BUDGET`]) — how many k-points and
/// tile rows share a launch. It moves launch boundaries only, never a sum's
/// order, so every value gives the same bits (the gates' dial).
fn tile_budgets() -> (usize, usize) {
    match std::env::var("PYSCF_PBC_GRID_TILE_BUDGET")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        Some(n) => (n, n),
        None => (TILE_PARTIAL_BUDGET, TILE_SCRATCH_BUDGET),
    }
}

/// `PYSCF_PBC_GRID_TILED` — whether the grid contractions (this module's and
/// [`crate::pbc::rho`]'s) take the SCF-03 tiled route. `0` pins the
/// one-lane-per-output kernels (the reference the gates compare against),
/// `1` forces the tiled ones; unset, the tiled route is taken on every
/// backend.
pub(crate) fn tiled_contractions<R: Runtime>(_client: &ComputeClient<R>) -> bool {
    !std::env::var("PYSCF_PBC_GRID_TILED").is_ok_and(|v| v.trim() == "0")
}

/// SCF-03 — how many lanes share one tile's grid sum.
///
/// `PYSCF_PBC_BAND_VMAT_SPLIT=<n>` pins it, as it pins [`split_factor`].
/// Unset: `1` on the CPU runtime, and with hardware planes the power of two
/// that brings one launch's `tiles · split` to [`SPLIT_TARGET_LANES`],
/// between [`SPLIT_MIN`] (a plane of consecutive grid points per tile) and
/// [`SPLIT_MAX`]. The partials are folded per launch, so no memory cap.
fn tile_split_factor<R: Runtime>(client: &ComputeClient<R>, tiles: usize, ngrids: usize) -> usize {
    let cap = |s: usize| s.min(ngrids.max(1)).max(1);
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
    let want = (SPLIT_TARGET_LANES / tiles.max(1)).clamp(SPLIT_MIN, SPLIT_MAX);
    cap(1usize << (usize::BITS - 1 - want.leading_zeros()))
}

/// SCF-03 — queue the tiled contraction of every k-point into `out_*`
/// (`nkpts · nao²`, row-major per k). See the module docs.
///
/// As many k-points share a launch as [`TILE_SCRATCH_BUDGET`] lets `aow`
/// hold (a run of k-points with one Γ flag — the flag is compiled in), so a
/// small basis at many k-points still fills the device; a large one goes a
/// few tile rows of one k-point at a time.
#[allow(clippy::too_many_arguments)]
fn run_tiled<R: Runtime>(
    client: &ComputeClient<R>,
    ao_re: &Handle,
    ao_im: &Handle,
    ao_len: usize,
    stride_k: usize,
    stride_e: usize,
    wv: &Handle,
    nvar: usize,
    comp: usize,
    nkpts: usize,
    nao: usize,
    ngrids: usize,
    gamma: &[bool],
    out_re: &Handle,
    out_im: &Handle,
) {
    let f64s = core::mem::size_of::<f64>();
    let npair = nao * nao;
    let total = nkpts * npair;
    let n = comp * nao * ngrids;
    let plane = nao * ngrids;
    let ntq = nao.div_ceil(VMAT_TILE);
    let point_major = stride_e != 1;
    let (partial_budget, scratch_budget) = tile_budgets();
    // K-points per launch, by scratch memory first.
    let per_k = if point_major { n } else { plane };
    let mut kb = (scratch_budget / (2 * f64s * per_k)).clamp(1, nkpts);
    let nsplit = tile_split_factor(client, kb * ntq * ntq, ngrids);
    // ... then so that one tile row of every k-point fits a launch's lanes
    // and its partial planes.
    let row_out = VMAT_TILE * nao;
    kb = kb
        .min((LANES_PER_LAUNCH / (ntq * nsplit)).max(1))
        .min((partial_budget / (2 * f64s * nsplit * row_out)).max(1));
    let rows_max = (partial_budget / (2 * f64s * nsplit * kb * row_out))
        .min(LANES_PER_LAUNCH / (kb * ntq * nsplit))
        .clamp(1, ntq);
    let part_len = nsplit * kb * rows_max * row_out;
    let aow_re = client.empty(kb * plane * f64s);
    let aow_im = client.empty(kb * plane * f64s);
    let part_re = client.empty(part_len * f64s);
    let part_im = client.empty(part_len * f64s);
    // Point-major planes: the launch's k-points gathered into a contiguous
    // scratch (the stride, not the kernel, is the cost — see `local_vmat`'s
    // module docs).
    let scratch = point_major.then(|| (client.empty(kb * n * f64s), client.empty(kb * n * f64s)));
    let local_bytes = (2 * VMAT_TILE * VMAT_TILE + 4 * VMAT_TILE) * f64s;
    let mut k0 = 0usize;
    while k0 < nkpts {
        let im_zero = gamma[k0];
        let mut kn = 1usize;
        while kn < kb && k0 + kn < nkpts && gamma[k0 + kn] == im_zero {
            kn += 1;
        }
        let (src_re, src_im, src_len, base, kstride) = match &scratch {
            None => (ao_re, ao_im, ao_len, k0 * stride_k, stride_k),
            Some((sre, sim)) => {
                let (count, dim) = launch_1d(client, kn * n, 1);
                let planes = [(ao_re, sre, true), (ao_im, sim, !im_zero)];
                for (src, dst, wanted) in planes {
                    if wanted {
                        unsafe {
                            gather_k_range_kernel::launch_unchecked::<f64, R>(
                                client,
                                count.clone(),
                                dim,
                                // SAFETY: `src` holds `ao_len = nkpts · n`
                                // point-major values and `k0 + kn <= nkpts`;
                                // `dst` holds `kb · n >= kn · n`.
                                ArrayArg::from_raw_parts(src.clone(), ao_len),
                                ArrayArg::from_raw_parts(dst.clone(), kb * n),
                                nkpts,
                                n,
                                k0,
                                kn,
                            );
                        }
                    }
                }
                (sre, sim, kb * n, 0, n)
            }
        };
        let (count, dim) = launch_1d(client, kn * plane, 4 * nvar);
        unsafe {
            aow_kernel::launch_unchecked::<f64, R>(
                client,
                count,
                dim,
                // SAFETY: the planes hold `src_len >= base + (kn − 1) ·
                // kstride + comp · nao · ngrids` values with `nvar <= comp`,
                // `wv` `nvar · ngrids`, `aow` `kb · nao · ngrids`; the kernel
                // guards `i < kn · nao · ngrids`.
                ArrayArg::from_raw_parts(src_re.clone(), src_len),
                ArrayArg::from_raw_parts(src_im.clone(), src_len),
                ArrayArg::from_raw_parts(wv.clone(), nvar * ngrids),
                ArrayArg::from_raw_parts(aow_re.clone(), kb * plane),
                ArrayArg::from_raw_parts(aow_im.clone(), kb * plane),
                nao,
                ngrids,
                nvar,
                base,
                kstride,
                kn,
                im_zero,
            );
        }
        let mut row0 = 0usize;
        while row0 < ntq {
            let rows = (ntq - row0).min(rows_max);
            let lanes = kn * rows * ntq * nsplit;
            let per_lane = 8 * VMAT_TILE * VMAT_TILE * ngrids.div_ceil(nsplit);
            for chunk in launch_1d_chunked(client, lanes, per_lane, local_bytes) {
                unsafe {
                    band_vmat_tile_kernel::launch_unchecked::<f64, R>(
                        client,
                        CubeCount::Static(chunk.count_x, 1, 1),
                        chunk.dim,
                        // SAFETY: the AO planes hold every `(kk, p, g)` of
                        // the launch's k-points, `aow` `kb · nao · ngrids`,
                        // the partials `part_len >= nsplit · kn · rows ·
                        // VMAT_TILE · nao`; the kernel guards the lane and
                        // clamps every row to `nao − 1`.
                        ArrayArg::from_raw_parts(src_re.clone(), src_len),
                        ArrayArg::from_raw_parts(src_im.clone(), src_len),
                        ArrayArg::from_raw_parts(aow_re.clone(), kb * plane),
                        ArrayArg::from_raw_parts(aow_im.clone(), kb * plane),
                        ArrayArg::from_raw_parts(part_re.clone(), part_len),
                        ArrayArg::from_raw_parts(part_im.clone(), part_len),
                        nao,
                        ngrids,
                        base,
                        kstride,
                        kn,
                        row0,
                        rows,
                        nsplit,
                        chunk.lane0,
                        im_zero,
                    );
                }
            }
            // This launch's rows `[row0 · T, min((row0 + rows) · T, nao))`.
            let p_first = row0 * VMAT_TILE;
            let valid = (((row0 + rows) * VMAT_TILE).min(nao) - p_first) * nao;
            let (count, dim) = launch_1d(client, kn * valid, nsplit);
            unsafe {
                band_vmat_tile_reduce_kernel::launch_unchecked::<f64, R>(
                    client,
                    count,
                    dim,
                    // SAFETY: partials hold `part_len >= nsplit · kn · rows ·
                    // VMAT_TILE · nao`, outputs `total`, and `(k0 + kn − 1) ·
                    // npair + p_first · nao + valid <= total`.
                    ArrayArg::from_raw_parts(part_re.clone(), part_len),
                    ArrayArg::from_raw_parts(part_im.clone(), part_len),
                    ArrayArg::from_raw_parts(out_re.clone(), total),
                    ArrayArg::from_raw_parts(out_im.clone(), total),
                    rows * row_out,
                    valid,
                    kn,
                    nsplit,
                    k0 * npair + p_first * nao,
                    npair,
                );
            }
            row0 += rows;
        }
        k0 += kn;
    }
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
