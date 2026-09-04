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
        Self { re, im, nkpts, n }
    }

    /// The `(nkpts, n)` shape these planes were built for.
    pub fn shape(&self) -> (usize, usize) {
        (self.nkpts, self.n)
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

    /// Read both planes back to the host, consuming the accumulator.
    ///
    /// One transfer for the whole image loop. Returns `(out_re, out_im)`, each
    /// `nkpts * n` reals in row-major `(nkpts, n)` order.
    pub fn into_planes(self, client: &AlgebraClient) -> (Vec<f64>, Vec<f64>) {
        let total = self.nkpts * self.n;
        if total == 0 {
            return (Vec::new(), Vec::new());
        }
        dispatch_backend!(client, c, Rt, {
            let bytes = c.read(vec![self.re.clone(), self.im.clone()]);
            (
                bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec(),
                bytemuck::cast_slice::<u8, f64>(&bytes[1]).to_vec(),
            )
        })
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
    };
    acc.accumulate(client, ao, pr, pi)?;
    Ok(acc.into_planes(client))
}
