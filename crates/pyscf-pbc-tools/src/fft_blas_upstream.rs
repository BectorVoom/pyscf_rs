//! Upstream's `_ifftn_blas` GEMM FFT (`pyscf/pbc/tools/pbc.py:50-68`) — the route
//! upstream's default `NUMPY+BLAS` engine takes when ALL three mesh axes are in
//! `_EXCLUDE`.
//!
//! # What was probed (all in `target/probes-task4/`)
//!
//! * `expRG = np.exp(2j*pi*fftfreq(n)[:,None] * arange(n))`: the complex
//!   scalar is `(0, 2*pi)` (exact), each column is `(2*pi)*ff[a]` (`2*pi`
//!   left), each element scales that by the integer index, and numpy's complex
//!   `exp` of the resulting pure-imaginary array is `(exp(0)*cos, exp(0)*sin)`
//!   with scalar-libm rounding — i.e. `(cos(theta), sin(theta))` with
//!   `theta = ((2*pi)*ff[a])*j`. Exhaustively verified bit-exact for every
//!   `n` in `1..300` plus all 106 `_EXCLUDE` members (12.2M complex
//!   exponentials, 0 mismatches: `exprg_exhaust.py`).
//! * The GEMM shape, by logging `_zgemm`'s arguments on the real call
//!   (`gemm_shape_probe.py`): `zdot('T','N', rest, L, L, f_view, expRG,
//!   alpha=1/L, beta=0)`, i.e. at BLAS level `zgemm_('N','T', L, rest, L,
//!   alpha, expRG, f, 0, c)` with `rest` the product of the other axes (times
//!   the batch chunk).
//! * The Barcelona kernel for that shape (`zgemm_alpha_probe.py`,
//!   `zgemm_kbig_probe.py`): [`pyscf_algebra::openblas_emu::zgemm`]'s model —
//!   the four real sub-sums in `seq`/`two_lanes` corner order with the real
//!   `alpha` scaling each K-block's combined sums, `GEMM_Q = 224` K-blocking
//!   included (0 mismatches).
//!
//! `lib.transpose` is an exact copy (no arithmetic); reshapes and `.T` views
//! are index arithmetic. The batch loop (`blksize`, `prange`) is reproduced
//! exactly; `get_nuc` runs it with `n = 1`.

use pyscf_algebra::{CTensor, openblas_emu};
use pyscf_core::{CoreError, PyscfRsError};

use crate::error::PbcToolsError;

/// `np.fft.fftfreq(n)` — the integer table times the ROUNDED reciprocal `1/n`
/// (a multiply, not a divide — same formula as
/// `pyscf_pbc_gto::gv::fftfreq`, which the grid-coords gate pins bit-exact).
fn fftfreq(n: usize) -> Vec<f64> {
    let val = 1.0 / n as f64;
    (0..n)
        .map(|i| {
            let f = if i <= (n - 1) / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            f * val
        })
        .collect()
}

/// `expRG(n) = np.exp(2j*pi*fftfreq(n)[:,None] * arange(n))` as an `(n, n)`
/// C-order table: `re = cos(theta)`, `im = sin(theta)` with
/// `theta = ((2*pi)*ff[a]) * j` (glibc trig, matching numpy's scalar-rounded
/// complex `exp` — see the module docs).
pub fn exp_rg(n: usize) -> CTensor {
    let ff = fftfreq(n);
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    for a in 0..n {
        let base = two_pi * ff[a];
        for j in 0..n {
            let theta = base * j as f64;
            re[a * n + j] = theta.cos();
            im[a * n + j] = theta.sin();
        }
    }
    CTensor::from_planes(re, im)
}

/// `max(int(1e5/(mx*my*mz)), 8) * 4` — `pbc.py:56`.
fn ifftn_blas_blksize(mesh: [usize; 3]) -> usize {
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    ((1e5 / ngrids as f64) as usize).max(8) * 4
}

/// One `_ifftn_blas` contraction stage along the axis of length `L`:
/// `C = (1/L) * F * expRG` where `F` is the `(L, rest)` C-order input in
/// `input` and the `(rest, L)` C-order output goes into `output` — i.e.
/// BLAS-level `zgemm_('N','T', L, rest, L, 1/L, exp, F, 0, C)`.
fn blas_stage(
    l: usize,
    rest: usize,
    exp: &CTensor,
    input_re: &[f64],
    input_im: &[f64],
    output_re: &mut [f64],
    output_im: &mut [f64],
) {
    openblas_emu::zgemm(
        'N',
        'T',
        l,
        rest,
        l,
        (1.0 / l as f64, 0.0),
        &exp.re,
        &exp.im,
        l,
        input_re,
        input_im,
        rest,
        (0.0, 0.0),
        output_re,
        output_im,
        l,
    );
}

/// `_ifftn_blas(g, mesh)` (`pbc.py:50-68`): the inverse 3-D transform as three
/// GEMM contractions, for a batch of `n = g.len()/ngrids` flat `(ngrids,)` rows.
///
/// # Errors
/// [`PbcToolsError`] when the buffer length is not a multiple of `ngrids`.
pub fn ifftn_blas(g: &CTensor, mesh: [usize; 3]) -> Result<CTensor, PbcToolsError> {
    let [mx, my, mz] = mesh;
    let ngrids = mx * my * mz;
    if ngrids == 0 {
        return Err(PbcToolsError::Core(PyscfRsError::Core(
            CoreError::InvalidMolecule(format!("ifftn_blas: mesh {mesh:?} has a zero axis")),
        )));
    }
    if !g.len().is_multiple_of(ngrids) {
        return Err(PbcToolsError::Core(PyscfRsError::Core(
            CoreError::InvalidMolecule(format!(
                "ifftn_blas: buffer length {} is not a multiple of ngrids {ngrids} (mesh {mesh:?})",
                g.len()
            )),
        )));
    }
    let nb = g.len() / ngrids;
    if nb == 0 {
        return Ok(CTensor::zeros(0));
    }
    let exp_x = exp_rg(mx);
    let exp_y = exp_rg(my);
    let exp_z = exp_rg(mz);
    let blksize = ifftn_blas_blksize(mesh);
    let mut out_re = vec![0.0_f64; nb * ngrids];
    let mut out_im = vec![0.0_f64; nb * ngrids];
    let mut buf_re = vec![0.0_f64; blksize * ngrids];
    let mut buf_im = vec![0.0_f64; blksize * ngrids];
    let mut b0 = 0;
    while b0 < nb {
        let ni = (b0 + blksize).min(nb) - b0;
        let (g_re, g_im) = (
            &g.re[b0 * ngrids..(b0 + ni) * ngrids],
            &g.im[b0 * ngrids..(b0 + ni) * ngrids],
        );
        let (buf1_re, buf1_im) = (&mut buf_re[..ni * ngrids], &mut buf_im[..ni * ngrids]);
        let (out1_re, out1_im) = (
            &mut out_re[b0 * ngrids..(b0 + ni) * ngrids],
            &mut out_im[b0 * ngrids..(b0 + ni) * ngrids],
        );
        // `f = lib.transpose(g[i0:i1].reshape(ni,-1))`: exact copy into the
        // `(ngrids, ni)` view.
        for b in 0..ni {
            for gg in 0..ngrids {
                buf1_re[gg * ni + b] = g_re[b * ngrids + gg];
                buf1_im[gg * ni + b] = g_im[b * ngrids + gg];
            }
        }
        // `f = lib.dot(f.reshape(mx,-1).T, expRGx, 1/mx, c=out1.reshape(-1,mx))`
        blas_stage(mx, my * mz * ni, &exp_x, buf1_re, buf1_im, out1_re, out1_im);
        // `f = lib.dot(f.reshape(my,-1).T, expRGy, 1/my, c=buf1.reshape(-1,my))`
        blas_stage(my, mz * ni * mx, &exp_y, out1_re, out1_im, buf1_re, buf1_im);
        // `f = lib.dot(f.reshape(mz,-1).T, expRGz, 1/mz, c=out1.reshape(-1,mz))`
        blas_stage(mz, ni * mx * my, &exp_z, buf1_re, buf1_im, out1_re, out1_im);
        b0 += ni;
    }
    Ok(CTensor::from_planes(out_re, out_im))
}
