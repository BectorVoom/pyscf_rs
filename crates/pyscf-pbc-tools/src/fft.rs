//! Complex 3-D FFT — `pyscf/pbc/tools/pbc.py:30-236` (plans 11-01 and 11-03).
//!
//! # Two engines (D-PBC-05, D-PBC-06)
//!
//! | engine | `PYSCF_PBC_FFT_ENGINE` | what it is |
//! |---|---|---|
//! | [`fft_blas`] | `blas` | statement-for-statement port of upstream's `_fftn_blas` — three batched complex GEMMs against explicit DFT matrices, every product through `pyscf_algebra::zgemm_dense` (D-PBC-03) |
//! | [`fft_stockham`] | `stockham` (default) | host mixed radix-2 / direct / Bluestein transform, [`crate::fft_kernel`] |
//!
//! `_fftn_blas` is not a fallback upstream: with the default
//! `FFT_ENGINE = 'NUMPY+BLAS'`, `pbc.py:128-140` routes to it whenever ALL
//! THREE mesh axes are in its `_EXCLUDE` prime list — which is exactly the
//! `[47, 47, 47]` default mesh of the reference diamond cell. So the GEMM
//! engine is the one that has to agree with upstream, and it does so through
//! the same algebra.
//!
//! The default is `stockham` because [`crate::fft_kernel`] is `O(n log n)`
//! where the GEMM engine is `O(n^{4/3})` (risk R-04) and because the CPU
//! runtime that backs `zgemm_dense` by default sustains only ~5 GFLOP/s here.
//! `tests/fft.rs` pins the two engines against each other to 1e-13 over 200
//! random `(mesh, n_batch)` combinations, which is the D-PBC-06 condition for
//! this default.
//!
//! # Layout
//!
//! Every entry point takes and returns a planar [`CTensor`] of `n_batch *
//! ngrids` elements, row-major over `(batch, x, y, z)` — the flattening
//! upstream's `f.reshape(-1, *mesh)` produces. `n_batch` is inferred from the
//! buffer length.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use pyscf_algebra::{AlgebraClient, CTensor, select_backend, zgemm_dense, ztranspose_dense};
use pyscf_core::{CoreError, PyscfRsError};

use crate::error::PbcToolsError;
use crate::fft_kernel::transform_axis;

/// Which transform engine [`fft`] / [`ifft`] dispatch to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftEngine {
    /// Upstream's `_fftn_blas` — three batched complex GEMMs.
    Blas,
    /// The host `O(n log n)` transform.
    Stockham,
}

/// `PYSCF_PBC_FFT_ENGINE`, read once. `blas` or `stockham`; anything else (and
/// unset) is [`FftEngine::Stockham`], with a warning for an unrecognised value.
pub fn fft_engine() -> FftEngine {
    static ENGINE: OnceLock<FftEngine> = OnceLock::new();
    *ENGINE.get_or_init(|| match std::env::var("PYSCF_PBC_FFT_ENGINE") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "blas" => FftEngine::Blas,
            "stockham" | "" => FftEngine::Stockham,
            other => {
                tracing::warn!(
                    engine = other,
                    "PYSCF_PBC_FFT_ENGINE is not one of blas|stockham; using stockham"
                );
                FftEngine::Stockham
            }
        },
        Err(_) => FftEngine::Stockham,
    })
}

fn ngrids_of(mesh: [usize; 3]) -> usize {
    mesh[0] * mesh[1] * mesh[2]
}

fn check_shape(f: &CTensor, mesh: [usize; 3]) -> Result<usize, PbcToolsError> {
    let ngrids = ngrids_of(mesh);
    if ngrids == 0 {
        return Err(PbcToolsError::Core(PyscfRsError::Core(
            CoreError::InvalidMolecule(format!("fft: mesh {mesh:?} has a zero axis")),
        )));
    }
    if f.len() % ngrids != 0 {
        return Err(PbcToolsError::Core(PyscfRsError::Core(
            CoreError::InvalidMolecule(format!(
                "fft: buffer length {} is not a multiple of ngrids {ngrids} (mesh {mesh:?})",
                f.len()
            )),
        )));
    }
    Ok(f.len() / ngrids)
}

/// `tools.fft(f, mesh)` — forward 3-D transform, normalisation factor 1.
///
/// # Errors
/// [`PbcToolsError::Core`] when `mesh` has a zero axis, when the buffer length
/// is not a multiple of `ngrids`, or when a device launch fails.
pub fn fft(f: &CTensor, mesh: [usize; 3]) -> Result<CTensor, PbcToolsError> {
    match fft_engine() {
        FftEngine::Blas => fft_blas(f, mesh),
        FftEngine::Stockham => fft_stockham(f, mesh, false),
    }
}

/// `tools.ifft(g, mesh)` — inverse 3-D transform, normalisation factor
/// `1/ngrids` applied as `1/mx * 1/my * 1/mz` across the three stages (upstream
/// stages the scale the same way; folding it into one final multiply would
/// change the rounding).
///
/// # Errors
/// As [`fft`].
pub fn ifft(g: &CTensor, mesh: [usize; 3]) -> Result<CTensor, PbcToolsError> {
    match fft_engine() {
        FftEngine::Blas => ifft_blas(g, mesh),
        FftEngine::Stockham => fft_stockham(g, mesh, true),
    }
}

/// `tools.fftk(f, mesh, expmikr)` — the transform of a Bloch-periodic function
/// `f(r) e^{-i k r}` (`pbc.py:222-227`).
///
/// `expmikr` is either `ngrids` long (broadcast across every batch row) or the
/// full `n_batch * ngrids`.
///
/// # Errors
/// As [`fft`], plus a shape error when `expmikr` is neither length.
pub fn fftk(f: &CTensor, mesh: [usize; 3], expmikr: &CTensor) -> Result<CTensor, PbcToolsError> {
    fft(&scale_by(f, mesh, expmikr)?, mesh)
}

/// `tools.ifftk(g, mesh, expikr)` — inverse transform followed by the
/// `e^{i k r}` phase (`pbc.py:229-236`).
///
/// # Errors
/// As [`fftk`].
pub fn ifftk(g: &CTensor, mesh: [usize; 3], expikr: &CTensor) -> Result<CTensor, PbcToolsError> {
    let out = ifft(g, mesh)?;
    scale_by(&out, mesh, expikr)
}

/// Row-broadcast complex Hadamard product: `out[b, r] = f[b, r] * w[r]`.
fn scale_by(f: &CTensor, mesh: [usize; 3], w: &CTensor) -> Result<CTensor, PbcToolsError> {
    let ngrids = ngrids_of(mesh);
    let nb = check_shape(f, mesh)?;
    if w.len() != ngrids && w.len() != f.len() {
        return Err(PbcToolsError::Core(PyscfRsError::Core(
            CoreError::InvalidMolecule(format!(
                "fftk: phase length {} is neither ngrids {ngrids} nor the full buffer {}",
                w.len(),
                f.len()
            )),
        )));
    }
    let broadcast = w.len() == ngrids;
    let mut re = vec![0.0_f64; f.len()];
    let mut im = vec![0.0_f64; f.len()];
    for b in 0..nb {
        for r in 0..ngrids {
            let p = b * ngrids + r;
            let q = if broadcast { r } else { p };
            re[p] = f.re[p] * w.re[q] - f.im[p] * w.im[q];
            im[p] = f.re[p] * w.im[q] + f.im[p] * w.re[q];
        }
    }
    Ok(CTensor::from_planes(re, im))
}

// ---------------------------------------------------------------------------
// The host (`stockham`) engine
// ---------------------------------------------------------------------------

/// The `O(n log n)` host transform. `backward` selects `ifft` (conjugate kernel
/// plus the staged `1/mx`, `1/my`, `1/mz` scaling).
///
/// # Errors
/// As [`fft`].
pub fn fft_stockham(
    f: &CTensor,
    mesh: [usize; 3],
    backward: bool,
) -> Result<CTensor, PbcToolsError> {
    let nb = check_shape(f, mesh)?;
    let [mx, my, mz] = mesh;
    let mut re = f.re.clone();
    let mut im = f.im.clone();

    // Layout is (nb, mx, my, mz) row-major throughout — no reshaping needed,
    // only the (outer, n, inner) triple changes per axis.
    transform_axis(&mut re, &mut im, nb, mx, my * mz, backward);
    if backward {
        scale_planes(&mut re, &mut im, 1.0 / mx as f64);
    }
    transform_axis(&mut re, &mut im, nb * mx, my, mz, backward);
    if backward {
        scale_planes(&mut re, &mut im, 1.0 / my as f64);
    }
    transform_axis(&mut re, &mut im, nb * mx * my, mz, 1, backward);
    if backward {
        scale_planes(&mut re, &mut im, 1.0 / mz as f64);
    }
    Ok(CTensor::from_planes(re, im))
}

fn scale_planes(re: &mut [f64], im: &mut [f64], s: f64) {
    for v in re.iter_mut() {
        *v *= s;
    }
    for v in im.iter_mut() {
        *v *= s;
    }
}

// ---------------------------------------------------------------------------
// The GEMM (`blas`) engine — upstream `_fftn_blas` / `_ifftn_blas`
// ---------------------------------------------------------------------------

/// Cached per-axis DFT matrices. Key is `(n, backward)`.
type DftMatCache = Mutex<HashMap<(usize, bool), Arc<CTensor>>>;

fn dft_mats() -> &'static DftMatCache {
    static CACHE: OnceLock<DftMatCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `expRG` for one axis.
///
/// Forward (`pbc.py:32-34`): `exp(-2j pi arange(n)[:,None] * fftfreq(n))`, i.e.
/// element `[r, g] = exp(-2 pi i r k_g / n)` with `k_g` the INTEGER frequency.
///
/// Backward (`pbc.py:52-54`): `exp(+2j pi fftfreq(n)[:,None] * arange(n))`,
/// i.e. element `[g, r] = exp(+2 pi i k_g r / n)`. Both are the same table with
/// the sign flipped, because the product `r * k_g` is symmetric.
fn dft_matrix(n: usize, backward: bool) -> Arc<CTensor> {
    let mut cache = match dft_mats().lock() {
        Ok(c) => c,
        Err(p) => p.into_inner(),
    };
    if let Some(m) = cache.get(&(n, backward)) {
        return Arc::clone(m);
    }
    let sign = if backward { 1.0 } else { -1.0 };
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    for a in 0..n {
        for b in 0..n {
            // `fftfreq_scaled`: [0, 1, .., (n-1)/2, -(n/2), .., -1].
            let k = if b <= (n - 1) / 2 {
                b as f64
            } else {
                b as f64 - n as f64
            };
            let ang = sign * 2.0 * std::f64::consts::PI * a as f64 * k / n as f64;
            re[a * n + b] = ang.cos();
            im[a * n + b] = ang.sin();
        }
    }
    let m = Arc::new(CTensor::from_planes(re, im));
    cache.insert((n, backward), Arc::clone(&m));
    m
}

/// One `_fftn_blas` stage: `g` is `(m, rest)` row-major; transpose it to
/// `(rest, m)` and multiply by the `(m, m)` DFT matrix.
fn contract_axis(
    client: &AlgebraClient,
    g: &CTensor,
    m: usize,
    rest: usize,
    mat: &CTensor,
) -> Result<CTensor, PbcToolsError> {
    let gt = ztranspose_dense(client, g, m, rest).map_err(algebra_err)?;
    zgemm_dense(client, &gt, mat, rest, m, m).map_err(algebra_err)
}

fn algebra_err(e: pyscf_algebra::AlgebraError) -> PbcToolsError {
    PbcToolsError::Core(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "fft: algebra failure: {e}"
    ))))
}

fn client() -> Result<AlgebraClient, PbcToolsError> {
    Ok(select_backend()
        .map_err(|e| {
            PbcToolsError::Core(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "fft: backend selection failed: {e}"
            ))))
        })?
        .client)
}

/// Upstream `_fftn_blas` (`pbc.py:30-48`), unbatched.
///
/// Upstream chunks the batch at `max(1e5/ngrids, 8) * 4` rows purely to bound
/// its two scratch arrays; the arithmetic per output element is independent of
/// the chunking (each is one length-`m` dot product), so this port transforms
/// the whole batch in one pass and is numerically identical.
///
/// # Errors
/// As [`fft`].
pub fn fft_blas(f: &CTensor, mesh: [usize; 3]) -> Result<CTensor, PbcToolsError> {
    blas_engine(f, mesh, false)
}

/// Upstream `_ifftn_blas` (`pbc.py:50-68`).
///
/// # Errors
/// As [`fft`].
pub fn ifft_blas(g: &CTensor, mesh: [usize; 3]) -> Result<CTensor, PbcToolsError> {
    blas_engine(g, mesh, true)
}

fn blas_engine(f: &CTensor, mesh: [usize; 3], backward: bool) -> Result<CTensor, PbcToolsError> {
    let nb = check_shape(f, mesh)?;
    let [mx, my, mz] = mesh;
    let ngrids = ngrids_of(mesh);
    if nb == 0 {
        return Ok(CTensor::zeros(0));
    }
    let client = client()?;

    // `g = lib.transpose(f.reshape(n, -1))` -> (ngrids, n).
    let mut g = ztranspose_dense(&client, f, nb, ngrids).map_err(algebra_err)?;

    for (m, rest) in [
        (mx, my * mz * nb),
        (my, mz * nb * mx),
        (mz, nb * mx * my),
    ] {
        let mat = dft_matrix(m, backward);
        g = contract_axis(&client, &g, m, rest, &mat)?;
        if backward {
            scale_planes(&mut g.re, &mut g.im, 1.0 / m as f64);
        }
    }
    Ok(g)
}
