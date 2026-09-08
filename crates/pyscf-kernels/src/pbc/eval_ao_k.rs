//! K-08 — Bloch phase accumulation for periodic AO evaluation (PBC-MASTER-PLAN §6).
//!
//! ```text
//! ao_k[k][p] += exp(i·k·L) · ao_L[p]      for every k, for one lattice image L
//! ```
//!
//! `ao_L` is the REAL molecular AO block that
//! `pyscf_kernels::eval_gto` already produces on `coords − L`; this kernel is
//! ONLY the phase-accumulate step. PBC-MASTER-PLAN plan 10-04 is explicit about
//! that: "Do **not** write a new AO evaluator" — `crates/pyscf-kernels/src/eval_gto.rs`
//! (2 564 lines) handles s/p/d + deriv1, sph + cart, and periodicity adds
//! nothing to the radial part.
//!
//! Layout is PLANAR (D-PBC-02 / RULE 8): `out_re` / `out_im` are
//! `(nkpts, n)` row-major, where `n = comp * ngrids * nao` is however many
//! reals one image's AO block holds — this kernel never interprets it.
//!
//! One lane per `(k, p)` pair, ACCUMULATING (`+=`) rather than assigning, so
//! the caller drives the image loop and the output survives across images.
//!
//! # Why not a GEMM
//!
//! `Σ_L expkL[k,L] · ao_L[p]` is formally `expkL @ AO`, and the 1-electron
//! driver's contraction has exactly that shape. It is NOT expressed as a
//! `gemm_dense` call because that would require materialising every image's AO
//! block at once (`nimgs · comp · ngrids · nao` reals — gigabytes on a real
//! grid). Streaming one image at a time keeps peak memory at ONE AO block and
//! the summation order fixed to `Ls` order, which is what upstream's
//! `PBCGTOval_*` driver does too.
//!
//! # Why the accumulators stay on the device
//!
//! This kernel is memory-bound to the point of being nothing else: it reads
//! `2·nkpts·n` reals and writes `2·nkpts·n` to do `4·nkpts·n` flops. The
//! original slice API took the accumulators by value and returned fresh `Vec`s,
//! so the driver's image loop uploaded and read back both `(nkpts, n)` planes
//! ONCE PER IMAGE — `4·nkpts·n` reals of round-trip traffic per image, to move
//! `n` reals of new data in. With a few hundred images that is the entire cost
//! of periodic AO evaluation, and none of it is arithmetic.
//!
//! [`AoKAccumulator`] holds the two planes in device buffers for the life of the
//! image loop: zeros uploaded once, one launch per image writing in place, one
//! read-back at the end. Per image the transfer drops to just the new AO block
//! (`n` reals) and the `2·nkpts` phase factors — independent of how many images
//! came before. The buffers are opaque (the cubecl `Handle`s are private
//! fields), so `pyscf-pbc-gto` drives the loop without naming a cubecl type and
//! the ALG-06 wall holds.
//!
//! Generic over the device float (`F: Float`, AGENTS.md §3 / RULE 5).

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::{launch_1d, line_size_for, upload};
use pyscf_algebra::{AlgebraClient, AlgebraError};

use crate::scalar::DeviceScalar;

/// `out_re[k*n + p] += pr[k]*ao[p]`, `out_im[k*n + p] += pi[k]*ao[p]`.
///
/// One lane per `(k, p_line)` pair, flattened as `i = k*n_lines + p_line`, where
/// a line is `N` adjacent `p`. Every operand is indexed by `p` alone except the
/// two phase factors, which are constant across the whole `p` axis for a given
/// `k` — so they broadcast into the vector and the AO block, both accumulators,
/// and the arithmetic all widen together.
///
/// `line_size_for` guarantees the width divides `n` exactly, so a lane's vector
/// never straddles two k rows. The `i < nkpts*n_lines` guard is still required:
/// the launch rounds the lane count up to a whole number of cubes, so tail lanes
/// must not write out of range.
#[cube(launch_unchecked)]
fn eval_ao_k_accumulate_kernel<F: Float + CubeElement, N: Size>(
    ao: &Array<Vector<F, N>>,
    pr: &Array<F>,
    pi: &Array<F>,
    out_re: &mut Array<Vector<F, N>>,
    out_im: &mut Array<Vector<F, N>>,
    nkpts: usize,
    n_lines: usize,
) {
    let i = ABSOLUTE_POS;
    if i < nkpts * n_lines {
        let k = i / n_lines;
        let p = i % n_lines;
        let v = ao[p];
        out_re[i] += Vector::<F, N>::new(pr[k]) * v;
        out_im[i] += Vector::<F, N>::new(pi[k]) * v;
    }
}

#[cube(launch_unchecked)]
fn eval_ao_k_accumulate_scatter_kernel<F: Float + CubeElement>(
    ao: &Array<F>,
    index: &Array<u32>,
    pr: &Array<F>,
    pi: &Array<F>,
    out_re: &mut Array<F>,
    out_im: &mut Array<F>,
    nkpts: usize,
    nkeep: usize,
    ngrids: usize,
    nao: usize,
    comp: usize,
) {
    let sub_n = comp * nkeep * nao;
    let i = ABSOLUTE_POS;
    if i < nkpts * sub_n {
        let k = i / sub_n;
        let q = i % sub_n;
        let c = q / (nkeep * nao);
        let rem = q % (nkeep * nao);
        let a = rem / nkeep;
        let j = rem % nkeep;
        let p = c * ngrids * nao + a * ngrids + index[j] as usize;
        out_re[k * comp * ngrids * nao + p] += pr[k] * ao[q];
        out_im[k * comp * ngrids * nao + p] += pi[k] * ao[q];
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    ao: &Handle,
    pr: &Handle,
    pi: &Handle,
    out_re: &Handle,
    out_im: &Handle,
    nkpts: usize,
    n: usize,
) {
    let total = nkpts * n;
    // The width must divide `n`, not `total`: a lane's vector spans adjacent `p`
    // within ONE k row, and `n` is what bounds that row.
    let line = line_size_for::<R, F>(client, n);
    let n_lines = n / line;
    let lanes = nkpts * n_lines;
    // Two multiply-adds per element, so the per-lane work is twice the width.
    let (count, dim) = launch_1d(client, lanes, 2 * line);

    unsafe {
        eval_ao_k_accumulate_kernel::launch_unchecked::<F, R>(
            client,
            count,
            dim,
            line,
            // SAFETY: ao holds n elements, pr/pi hold nkpts, the two outputs
            // hold nkpts*n each; the kernel guards `i < nkpts*n_lines`.
            ArrayArg::from_raw_parts(ao.clone(), n),
            ArrayArg::from_raw_parts(pr.clone(), nkpts),
            ArrayArg::from_raw_parts(pi.clone(), nkpts),
            ArrayArg::from_raw_parts(out_re.clone(), total),
            ArrayArg::from_raw_parts(out_im.clone(), total),
            nkpts,
            n_lines,
        );
    }
}

/// K-08b (session 3): the scatter accumulate with the `k` loop INSIDE the
/// lane — one lane per kept AO element instead of one per `(k, element)`.
///
/// The session-3 instrument measured the per-`(k, element)` form at the same
/// 2.1 s whether the W-09 screen kept 24 % or 100 % of the grid, i.e. ~4×
/// less efficient per element than the dense vectorised kernel: every lane
/// re-did the five-division index decode and re-read `ao[q]` / `index[j]`
/// for each of the `nkpts` k-points. Here the decode and the two loads happen
/// once per element and the `nkpts` accumulations follow.
///
/// Bit-exact: each `(k, p)` accumulator still receives exactly one addition
/// per image, `pr[k] * ao[q]`, the same product in the same place — only
/// which lane performs it changes.
#[cube(launch_unchecked)]
fn eval_ao_k_accumulate_scatter_kloop_kernel<F: Float + CubeElement>(
    ao: &Array<F>,
    index: &Array<u32>,
    pr: &Array<F>,
    pi: &Array<F>,
    out_re: &mut Array<F>,
    out_im: &mut Array<F>,
    nkpts: usize,
    nkeep: usize,
    ngrids: usize,
    nao: usize,
    comp: usize,
) {
    let sub_n = comp * nkeep * nao;
    let q = ABSOLUTE_POS;
    if q < sub_n {
        let c = q / (nkeep * nao);
        let rem = q % (nkeep * nao);
        let a = rem / nkeep;
        let j = rem % nkeep;
        let p = c * ngrids * nao + a * ngrids + index[j] as usize;
        let n = comp * ngrids * nao;
        let v = ao[q];
        for k in 0..nkpts {
            out_re[k * n + p] += pr[k] * v;
            out_im[k * n + p] += pi[k] * v;
        }
    }
}

/// `PYSCF_PBC_K08_SCATTER=legacy` pins the per-`(k, element)` kernel so the
/// profiler can measure both forms with one binary.
fn legacy_scatter_kernel() -> bool {
    std::env::var("PYSCF_PBC_K08_SCATTER").is_ok_and(|v| v.eq_ignore_ascii_case("legacy"))
}

#[allow(clippy::too_many_arguments)]
fn launch_scatter_on_handles<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    ao: &Handle,
    index: &Handle,
    pr: &Handle,
    pi: &Handle,
    out_re: &Handle,
    out_im: &Handle,
    nkpts: usize,
    nkeep: usize,
    ngrids: usize,
    nao: usize,
    comp: usize,
) {
    let sub_n = comp * nkeep * nao;
    if !legacy_scatter_kernel() {
        // K-08b: one lane per element, `2 * nkpts` multiply-adds each.
        let (count, dim) = launch_1d(client, sub_n, 2 * nkpts);
        unsafe {
            eval_ao_k_accumulate_scatter_kloop_kernel::launch_unchecked::<F, R>(
                client,
                count,
                dim,
                ArrayArg::from_raw_parts(ao.clone(), sub_n),
                ArrayArg::from_raw_parts(index.clone(), nkeep),
                ArrayArg::from_raw_parts(pr.clone(), nkpts),
                ArrayArg::from_raw_parts(pi.clone(), nkpts),
                ArrayArg::from_raw_parts(out_re.clone(), nkpts * comp * ngrids * nao),
                ArrayArg::from_raw_parts(out_im.clone(), nkpts * comp * ngrids * nao),
                nkpts,
                nkeep,
                ngrids,
                nao,
                comp,
            );
        }
        return;
    }
    let lanes = nkpts * sub_n;
    let (count, dim) = launch_1d(client, lanes, 2);
    unsafe {
        eval_ao_k_accumulate_scatter_kernel::launch_unchecked::<F, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(ao.clone(), sub_n),
            ArrayArg::from_raw_parts(index.clone(), nkeep),
            ArrayArg::from_raw_parts(pr.clone(), nkpts),
            ArrayArg::from_raw_parts(pi.clone(), nkpts),
            ArrayArg::from_raw_parts(out_re.clone(), nkpts * comp * ngrids * nao),
            ArrayArg::from_raw_parts(out_im.clone(), nkpts * comp * ngrids * nao),
            nkpts,
            nkeep,
            ngrids,
            nao,
            comp,
        );
    }
}

/// The most images one K-09 batch may hold — the width of the kernel's
/// per-lane gather registers. 16 → 32 in session 5: at 16 the accumulate's
/// remaining traffic per image was one block read plus `4·nkpts·n/16` of
/// plane read-modify-write, i.e. the planes were still half of it.
pub const AO_IMAGE_BATCH_MAX: usize = 32;

/// K-09 — fold a BATCH of lattice images into every k-point in one launch.
///
/// Session 4 measured the cold periodic AO pass with K-08 switched off
/// (`PYSCF_PBC_AO_SKIP_K08=1`): 540 ms against 1 946 ms at `deriv 0` and
/// 826 ms against 5 217 ms at `deriv 1` (si gth-szv 2×2×2, mesh 31). The
/// accumulate was 72-84 % of the pass, not the 20-23 % its host span showed
/// (lazy launches), and the AO kernel's arithmetic was ~0 — a lane doing
/// nothing but its zero-fill store cost the same. What K-08 pays for is the
/// read-modify-write of BOTH `(nkpts, n)` planes on EVERY image: `4·nkpts·n`
/// reals of traffic to fold `n` reals in.
///
/// This kernel reads and writes each `(k, p)` accumulator ONCE per `nimg`
/// images. Per lane (one element `p`): gather the element's value from each
/// image of the batch — a dense image at `p` itself, a screened image through
/// its inverse index `pos[m·ngrids + g]` (`>= nkeep[m]` when the point was
/// not kept) — then, per k, `acc = out[k,p]; acc += pr[m,k]·v_m` over the
/// images IN IMAGE ORDER; store. That is exactly the sequence of additions the
/// per-image launches performed on the same accumulator (an un-kept element
/// received no addition from that image, and receives none here), so the
/// planes are bit-identical to the one-image-per-launch path; RULE-T traffic
/// per image drops from `4·nkpts·n` to `4·nkpts·n / nimg + n + ngrids/2`.
///
/// `ao` holds the batch's AO blocks in fixed-stride slots of `block_len`
/// reals; `nkeep[m]` is the image's kept-point count (`ngrids` when dense),
/// `dense[m] != 0` marks a full-grid image whose block is in grid order.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn eval_ao_k_accumulate_batch_kernel<F: Float + CubeElement>(
    ao: &Array<F>,
    pos: &Array<u32>,
    dense: &Array<u32>,
    nkeep: &Array<u32>,
    pr: &Array<F>,
    pi: &Array<F>,
    out_re: &mut Array<F>,
    out_im: &mut Array<F>,
    nkpts: usize,
    n: usize,
    ngrids: usize,
    nao: usize,
    nimg: usize,
    block_len: usize,
    lane0: usize,
) {
    // `lane0`: the launch is chunked on the CPU runtime (`launch_1d_chunked`)
    // because this kernel's two local arrays cost stack per iteration there.
    let p = ABSOLUTE_POS + lane0;
    if p < n {
        let c = p / (ngrids * nao);
        let rem = p % (ngrids * nao);
        let a = rem / ngrids;
        let g = rem % ngrids;
        let mut vals = Array::<F>::new(AO_IMAGE_BATCH_MAX);
        let mut present = Array::<u32>::new(AO_IMAGE_BATCH_MAX);
        for m in 0..nimg {
            let mut v = F::from_int(0);
            let mut hit = 0u32;
            if dense[m] != 0u32 {
                v = ao[m * block_len + p];
                hit = 1u32;
            } else {
                let j = pos[m * ngrids + g];
                let nk = nkeep[m];
                if j < nk {
                    let nk_us = nk as usize;
                    v = ao[m * block_len + c * nk_us * nao + a * nk_us + j as usize];
                    hit = 1u32;
                }
            }
            vals[m] = v;
            present[m] = hit;
        }
        for k in 0..nkpts {
            let mut re = out_re[k * n + p];
            let mut im = out_im[k * n + p];
            for m in 0..nimg {
                if present[m] != 0u32 {
                    let v = vals[m];
                    re += pr[m * nkpts + k] * v;
                    im += pi[m * nkpts + k] * v;
                }
            }
            out_re[k * n + p] = re;
            out_im[k * n + p] = im;
        }
    }
}

/// K-09: a device buffer of `capacity` fixed-stride slots, each able to hold
/// one image's AO block (`block_len` reals), plus the per-image bookkeeping the
/// batched kernel needs. The driver fills a slot with
/// [`AoImageBatch::slot`] + `eval_gto_*_into_target`, registers the image with
/// [`AoImageBatch::push`], and hands the batch to
/// [`AoKAccumulator::accumulate_batch`] when it is full or the image list ends.
///
/// Handles stay private (ALG-06); the driver never names a cubecl type.
pub struct AoImageBatch {
    buf: Handle,
    block_len: usize,
    capacity: usize,
    ngrids: usize,
    /// Per registered image: kept-point count (`ngrids` when dense).
    nkeep: Vec<u32>,
    /// Per registered image: `1` when the slot holds a full-grid block.
    dense: Vec<u32>,
    /// `len · ngrids` inverse indices — `pos[m·ngrids + g]` is the kept
    /// position of grid point `g` in image `m`, or `u32::MAX`. Dense images'
    /// rows are left as `u32::MAX` and never read.
    pos: Vec<u32>,
    /// `len · nkpts` phases, image-major.
    pr: Vec<f64>,
    pi: Vec<f64>,
    nkpts: usize,
}

impl core::fmt::Debug for AoImageBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AoImageBatch")
            .field("block_len", &self.block_len)
            .field("capacity", &self.capacity)
            .field("len", &self.nkeep.len())
            .finish_non_exhaustive()
    }
}

impl AoImageBatch {
    /// Allocate `capacity` slots of `block_len` reals. `capacity` is clamped
    /// to `1..=AO_IMAGE_BATCH_MAX`.
    pub fn new(
        client: &AlgebraClient,
        capacity: usize,
        block_len: usize,
        ngrids: usize,
        nkpts: usize,
    ) -> Self {
        let capacity = capacity.clamp(1, AO_IMAGE_BATCH_MAX);
        let bytes = (capacity * block_len).max(1) * core::mem::size_of::<f64>();
        let buf = dispatch_backend!(client, c, Rt, c.empty(bytes));
        Self {
            buf,
            block_len,
            capacity,
            ngrids,
            nkeep: Vec::with_capacity(capacity),
            dense: Vec::with_capacity(capacity),
            pos: Vec::new(),
            pr: Vec::with_capacity(capacity * nkpts),
            pi: Vec::with_capacity(capacity * nkpts),
            nkpts,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Reals per slot.
    pub fn block_len(&self) -> usize {
        self.block_len
    }
    /// The whole slot buffer — for the batched AO kernels (A-06), which write
    /// every image of a batch into its slot in one launch. Crate-private.
    pub(crate) fn buffer(&self) -> &Handle {
        &self.buf
    }
    /// Images registered so far.
    pub fn len(&self) -> usize {
        self.nkeep.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nkeep.is_empty()
    }
    pub fn is_full(&self) -> bool {
        self.nkeep.len() == self.capacity
    }

    /// A view of the NEXT free slot as a device block of `len <= block_len`
    /// reals with the given logical shape — the target `eval_gto_*_into_target`
    /// writes into. Call [`Self::push`] afterwards to register the image.
    ///
    /// # Errors
    /// [`AlgebraError::ShapeMismatch`] when the batch is full or `len` exceeds
    /// the slot.
    pub fn slot(
        &self,
        len: usize,
        shape: Vec<usize>,
    ) -> Result<crate::AoBlockDevice, AlgebraError> {
        if self.is_full() || len > self.block_len {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!(
                    "a free slot (len {} < capacity {}) of {} reals",
                    self.len(),
                    self.capacity,
                    self.block_len
                ),
                actual: format!("len {} images, {len} reals requested", self.len()),
            });
        }
        let offset = (self.len() * self.block_len * core::mem::size_of::<f64>()) as u64;
        Ok(crate::AoBlockDevice::from_handle(
            self.buf.clone().offset_start(offset),
            len,
            shape,
        ))
    }

    /// Register the image whose block was just written into the slot
    /// [`Self::slot`] handed out: `index` is its kept-point list (`None` for a
    /// dense, full-grid block), `pr`/`pi` its `nkpts` phases.
    ///
    /// # Errors
    /// [`AlgebraError::ShapeMismatch`] when the batch is full, the phases are
    /// the wrong length, or an index is out of range.
    pub fn push(
        &mut self,
        index: Option<&[usize]>,
        pr: &[f64],
        pi: &[f64],
    ) -> Result<(), AlgebraError> {
        if self.is_full() || pr.len() != self.nkpts || pi.len() != self.nkpts {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!(
                    "a free slot and pr/pi of len {} (batch {}/{})",
                    self.nkpts,
                    self.len(),
                    self.capacity
                ),
                actual: format!("pr {} pi {}", pr.len(), pi.len()),
            });
        }
        let row = self.pos.len();
        self.pos.resize(row + self.ngrids, u32::MAX);
        match index {
            None => {
                self.dense.push(1);
                self.nkeep.push(self.ngrids as u32);
            }
            Some(index) => {
                if index.iter().any(|&g| g >= self.ngrids) {
                    self.pos.truncate(row);
                    return Err(AlgebraError::ShapeMismatch {
                        expected: format!("kept indices < ngrids = {}", self.ngrids),
                        actual: format!("max index {:?}", index.iter().max()),
                    });
                }
                for (j, &g) in index.iter().enumerate() {
                    self.pos[row + g] = j as u32;
                }
                self.dense.push(0);
                self.nkeep.push(index.len() as u32);
            }
        }
        self.pr.extend_from_slice(pr);
        self.pi.extend_from_slice(pi);
        Ok(())
    }

    /// Forget the registered images; the slots are reused from the start.
    pub fn clear(&mut self) {
        self.nkeep.clear();
        self.dense.clear();
        self.pos.clear();
        self.pr.clear();
        self.pi.clear();
    }
}

/// The two `(nkpts, n)` accumulator planes, resident on the device for the whole
/// lattice-image loop.
///
/// Build one with [`AoKAccumulator::zeros`], fold each image in with
/// [`AoKAccumulator::accumulate`], and take the result with
/// [`AoKAccumulator::into_planes`]. See the module docs for why this exists
/// rather than passing the planes through the host on every image.
///
/// The buffers are cubecl `Handle`s, but only as PRIVATE fields — nothing in
/// this type's public surface names a cubecl type, so callers outside the ALG-06
/// allowlist can drive the loop. Handles are reference-counted by the runtime
/// and released when this value drops.
pub struct AoKAccumulator {
    re: Handle,
    im: Handle,
    nkpts: usize,
    n: usize,
    /// K-10v: `true` when the planes are stored `out[p·nkpts + k]` (k fastest,
    /// so the fused kernel's k-loop is a vector), `false` for the k-major
    /// `out[k·n + p]` every other kernel uses. Read-back transposes either to
    /// per-k planes.
    point_major: bool,
}

impl core::fmt::Debug for AoKAccumulator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AoKAccumulator")
            .field("nkpts", &self.nkpts)
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

impl AoKAccumulator {
    /// Allocate both `(nkpts, n)` planes on the device, zero-filled.
    ///
    /// The zeros are uploaded rather than produced by a fill kernel: it is one
    /// transfer for the whole loop either way, and a `create_from_slice` needs
    /// no launch, no extra kernel variant, and no guarantee about what
    /// `client.empty` leaves in the buffer.
    ///
    /// A degenerate `nkpts * n == 0` allocates nothing; [`Self::accumulate`] is
    /// then a no-op and [`Self::into_planes`] returns empty vectors.
    pub fn zeros(client: &AlgebraClient, nkpts: usize, n: usize) -> Self {
        // `max(1)`: a zero-length allocation is not something every cubecl
        // backend is obliged to handle, and this path is reachable from public
        // API (an empty grid or an empty basis). The extra element is never
        // read — `accumulate` and `into_planes` both short-circuit on an empty
        // shape — so one wasted f64 buys the degenerate case out entirely.
        let zeros = vec![0.0f64; (nkpts * n).max(1)];
        let (re, im) = dispatch_backend!(
            client,
            c,
            Rt,
            (upload(c, zeros.as_slice()), upload(c, zeros.as_slice()))
        );
        Self {
            re,
            im,
            nkpts,
            n,
            point_major: false,
        }
    }

    /// [`Self::zeros`] in the point-major layout the fused K-10 kernel
    /// accumulates into. Only `eval_ao_k_fused_batch` and
    /// [`Self::into_k_planes`] understand this layout; the per-image and
    /// batched accumulates refuse it.
    pub fn zeros_point_major(client: &AlgebraClient, nkpts: usize, n: usize) -> Self {
        let mut acc = Self::zeros(client, nkpts, n);
        acc.point_major = true;
        acc
    }

    /// Whether the planes are point-major (see [`Self::zeros_point_major`]).
    pub fn is_point_major(&self) -> bool {
        self.point_major
    }

    /// The `(nkpts, n)` shape these planes were built for.
    pub fn shape(&self) -> (usize, usize) {
        (self.nkpts, self.n)
    }

    /// The two resident planes — for the fused K-10 kernel, which adds into
    /// them directly. Crate-private.
    pub(crate) fn planes(&self) -> (&Handle, &Handle) {
        (&self.re, &self.im)
    }

    /// Fold ONE image's real AO block into every k-point, in place on the
    /// resident planes.
    ///
    /// * `ao` — the image's AO block, `n` reals, whatever internal layout the
    ///   caller uses (it is combined elementwise).
    /// * `pr` / `pi` — `exp(i·k·L)` for this image, one entry per k-point
    ///   (a column of the [`crate::pbc::bloch_phase`] table).
    ///
    /// Only `ao` and the two phase vectors cross to the device; the accumulators
    /// stay where they are.
    ///
    /// # Errors
    /// [`AlgebraError::ShapeMismatch`] when `ao` or either phase vector
    /// disagrees with the shape this accumulator was built for.
    pub fn accumulate(
        &mut self,
        client: &AlgebraClient,
        ao: &[f64],
        pr: &[f64],
        pi: &[f64],
    ) -> Result<(), AlgebraError> {
        if ao.len() != self.n {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("ao len {}", self.n),
                actual: ao.len().to_string(),
            });
        }
        if pr.len() != self.nkpts || pi.len() != self.nkpts {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("pr/pi len {}", self.nkpts),
                actual: format!("pr {} pi {}", pr.len(), pi.len()),
            });
        }
        if self.nkpts == 0 || self.n == 0 {
            return Ok(());
        }
        self.require_k_major()?;
        let (nkpts, n) = (self.nkpts, self.n);
        dispatch_backend!(client, c, Rt, {
            let ao_h = upload(c, ao);
            let pr_h = upload(c, pr);
            let pi_h = upload(c, pi);
            launch_on_handles::<Rt, f64>(c, &ao_h, &pr_h, &pi_h, &self.re, &self.im, nkpts, n)
        });
        Ok(())
    }

    /// Fold a resident, full-size AO block without a device→host→device round trip.
    pub fn accumulate_device(
        &mut self,
        client: &AlgebraClient,
        ao: &crate::AoBlockDevice,
        pr: &[f64],
        pi: &[f64],
    ) -> Result<(), AlgebraError> {
        if ao.len() != self.n {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("resident ao len {}", self.n),
                actual: ao.len().to_string(),
            });
        }
        if pr.len() != self.nkpts || pi.len() != self.nkpts {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("pr/pi len {}", self.nkpts),
                actual: format!("pr {} pi {}", pr.len(), pi.len()),
            });
        }
        if self.nkpts == 0 || self.n == 0 {
            return Ok(());
        }
        self.require_k_major()?;
        dispatch_backend!(client, c, Rt, {
            let pr_h = upload(c, pr);
            let pi_h = upload(c, pi);
            launch_on_handles::<Rt, f64>(
                c,
                ao.handle(),
                &pr_h,
                &pi_h,
                &self.re,
                &self.im,
                self.nkpts,
                self.n,
            )
        });
        Ok(())
    }

    /// Fold a resident AO sub-grid directly into full-grid accumulator positions.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_device_scatter(
        &mut self,
        client: &AlgebraClient,
        ao: &crate::AoBlockDevice,
        index: &[usize],
        ngrids: usize,
        nao: usize,
        comp: usize,
        pr: &[f64],
        pi: &[f64],
    ) -> Result<(), AlgebraError> {
        let expected_sub = comp * index.len() * nao;
        if ao.len() != expected_sub
            || self.n != comp * ngrids * nao
            || index.iter().any(|&g| g >= ngrids)
        {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!(
                    "resident sub-AO {expected_sub}, accumulator {}, indices < {ngrids}",
                    comp * ngrids * nao
                ),
                actual: format!(
                    "sub-AO {}, accumulator {}, max index {:?}",
                    ao.len(),
                    self.n,
                    index.iter().max()
                ),
            });
        }
        if pr.len() != self.nkpts || pi.len() != self.nkpts {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("pr/pi len {}", self.nkpts),
                actual: format!("pr {} pi {}", pr.len(), pi.len()),
            });
        }
        if self.nkpts == 0 || expected_sub == 0 {
            return Ok(());
        }
        self.require_k_major()?;
        let index_u32: Vec<u32> = index.iter().map(|&g| g as u32).collect();
        dispatch_backend!(client, c, Rt, {
            let index_h = c.create_from_slice(bytemuck::cast_slice(&index_u32));
            let pr_h = upload(c, pr);
            let pi_h = upload(c, pi);
            launch_scatter_on_handles::<Rt, f64>(
                c,
                ao.handle(),
                &index_h,
                &pr_h,
                &pi_h,
                &self.re,
                &self.im,
                self.nkpts,
                index.len(),
                ngrids,
                nao,
                comp,
            )
        });
        Ok(())
    }

    /// K-09: fold every image registered in `batch` into the planes in ONE
    /// launch — see [`eval_ao_k_accumulate_batch_kernel`] for why this is
    /// bit-identical to folding them one launch at a time.
    ///
    /// `ngrids`, `nao`, `comp` describe the accumulator's `n = comp · ngrids ·
    /// nao` layout; the batch's `block_len` must be `n`.
    ///
    /// # Errors
    /// [`AlgebraError::ShapeMismatch`] on a layout disagreement. An empty
    /// batch is a no-op.
    pub fn accumulate_batch(
        &mut self,
        client: &AlgebraClient,
        batch: &AoImageBatch,
        ngrids: usize,
        nao: usize,
        comp: usize,
    ) -> Result<(), AlgebraError> {
        let nimg = batch.len();
        if nimg == 0 || self.nkpts == 0 || self.n == 0 {
            return Ok(());
        }
        self.require_k_major()?;
        if batch.block_len != self.n
            || self.n != comp * ngrids * nao
            || batch.nkpts != self.nkpts
            || batch.ngrids != ngrids
        {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!(
                    "batch block_len {} == n {} == comp·ngrids·nao {}, nkpts {}, ngrids {}",
                    batch.block_len,
                    self.n,
                    comp * ngrids * nao,
                    self.nkpts,
                    ngrids
                ),
                actual: format!("batch nkpts {}, batch ngrids {}", batch.nkpts, batch.ngrids),
            });
        }
        let (nkpts, n) = (self.nkpts, self.n);
        dispatch_backend!(client, c, Rt, {
            let pos_h = c.create_from_slice(bytemuck::cast_slice(&batch.pos));
            let dense_h = c.create_from_slice(bytemuck::cast_slice(&batch.dense));
            let nkeep_h = c.create_from_slice(bytemuck::cast_slice(&batch.nkeep));
            let pr_h = upload(c, &batch.pr);
            let pi_h = upload(c, &batch.pi);
            // Per lane: `nimg` gathers, then `2·nkpts·nimg` multiply-adds.
            // Chunked: the lane's `vals` + `present` locals are stack per
            // iteration on the CPU runtime (see `launch_1d_chunked`).
            let local_bytes = AO_IMAGE_BATCH_MAX * (core::mem::size_of::<f64>() + 4);
            for chunk in
                pyscf_algebra::launch::launch_1d_chunked(c, n, nimg * (2 * nkpts + 1), local_bytes)
            {
                unsafe {
                    eval_ao_k_accumulate_batch_kernel::launch_unchecked::<f64, Rt>(
                        c,
                        CubeCount::Static(chunk.count_x, 1, 1),
                        chunk.dim,
                        ArrayArg::from_raw_parts(
                            batch.buf.clone(),
                            batch.capacity * batch.block_len,
                        ),
                        ArrayArg::from_raw_parts(pos_h.clone(), batch.pos.len()),
                        ArrayArg::from_raw_parts(dense_h.clone(), nimg),
                        ArrayArg::from_raw_parts(nkeep_h.clone(), nimg),
                        ArrayArg::from_raw_parts(pr_h.clone(), nimg * nkpts),
                        ArrayArg::from_raw_parts(pi_h.clone(), nimg * nkpts),
                        ArrayArg::from_raw_parts(self.re.clone(), nkpts * n),
                        ArrayArg::from_raw_parts(self.im.clone(), nkpts * n),
                        nkpts,
                        n,
                        ngrids,
                        nao,
                        nimg,
                        batch.block_len,
                        chunk.lane0,
                    );
                }
            }
        });
        Ok(())
    }

    fn require_k_major(&self) -> Result<(), AlgebraError> {
        if self.point_major {
            return Err(AlgebraError::ShapeMismatch {
                expected: "a k-major accumulator (`AoKAccumulator::zeros`)".to_string(),
                actual: "a point-major accumulator (only the fused K-10 kernel writes it)"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Read both planes back to the host, consuming the accumulator.
    ///
    /// One transfer for the whole image loop. Returns `(out_re, out_im)`, each
    /// `nkpts * n` reals in row-major `(nkpts, n)` order — a point-major
    /// accumulator is transposed on the way.
    pub fn into_planes(self, client: &AlgebraClient) -> (Vec<f64>, Vec<f64>) {
        let (nkpts, n) = (self.nkpts, self.n);
        let planes = self.into_k_planes(client);
        let mut re = Vec::with_capacity(nkpts * n);
        let mut im = Vec::with_capacity(nkpts * n);
        for (r, i) in planes {
            re.extend_from_slice(&r);
            im.extend_from_slice(&i);
        }
        (re, im)
    }

    /// Read the planes back as one `(re, im)` pair of `n` reals per k-point —
    /// the shape the periodic driver splits into per-k tensors anyway, so a
    /// point-major accumulator is gathered straight into them (one strided
    /// pass per k, k-points in parallel) and never materialised k-major.
    pub fn into_k_planes(self, client: &AlgebraClient) -> Vec<(Vec<f64>, Vec<f64>)> {
        use rayon::prelude::*;
        let (nkpts, n) = (self.nkpts, self.n);
        if nkpts * n == 0 {
            return Vec::new();
        }
        let bytes = dispatch_backend!(
            client,
            c,
            Rt,
            c.read(vec![self.re.clone(), self.im.clone()])
        );
        let re: &[f64] = bytemuck::cast_slice(&bytes[0]);
        let im: &[f64] = bytemuck::cast_slice(&bytes[1]);
        if self.point_major {
            (0..nkpts)
                .into_par_iter()
                .map(|k| {
                    let rk: Vec<f64> = (0..n).map(|p| re[p * nkpts + k]).collect();
                    let ik: Vec<f64> = (0..n).map(|p| im[p * nkpts + k]).collect();
                    (rk, ik)
                })
                .collect()
        } else {
            (0..nkpts)
                .into_par_iter()
                .map(|k| {
                    (
                        re[k * n..(k + 1) * n].to_vec(),
                        im[k * n..(k + 1) * n].to_vec(),
                    )
                })
                .collect()
        }
    }
}

/// K-08 single-shot entry point: fold ONE image's real AO block into the
/// k-resolved accumulators, taking and returning them on the host.
///
/// Prefer [`AoKAccumulator`] when driving a lattice-image loop: this signature
/// forces both `(nkpts, n)` planes through a host round-trip on every call,
/// which is the dominant cost of periodic AO evaluation (see the module docs).
/// This form is kept for callers that genuinely fold a single image.
///
/// * `ao` — the image's AO block, `n` reals.
/// * `pr` / `pi` — `exp(i·k·L)` for this image, one entry per k-point.
/// * `out_re` / `out_im` — the `(nkpts, n)` row-major accumulators; the returned
///   pair replaces them.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when the phase vectors disagree with each
/// other or the accumulators are not `nkpts * ao.len()` long.
pub fn eval_ao_k_accumulate(
    client: &AlgebraClient,
    ao: &[f64],
    pr: &[f64],
    pi: &[f64],
    out_re: &[f64],
    out_im: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    let nkpts = pr.len();
    let n = ao.len();
    if pi.len() != nkpts {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("pi len {nkpts}"),
            actual: pi.len().to_string(),
        });
    }
    if out_re.len() != nkpts * n || out_im.len() != nkpts * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("accumulators of length nkpts*n = {}", nkpts * n),
            actual: format!("re {} im {}", out_re.len(), out_im.len()),
        });
    }
    if nkpts == 0 || n == 0 {
        return Ok((out_re.to_vec(), out_im.to_vec()));
    }

    // Seed the device planes from the caller's accumulators rather than from
    // zeros, then run the one image and read back — the same launch the resident
    // path uses, wrapped in the transfers this signature requires.
    let (re_h, im_h) = dispatch_backend!(client, c, Rt, (upload(c, out_re), upload(c, out_im)));
    let mut acc = AoKAccumulator {
        re: re_h,
        im: im_h,
        nkpts,
        n,
        point_major: false,
    };
    acc.accumulate(client, ao, pr, pi)?;
    Ok(acc.into_planes(client))
}
