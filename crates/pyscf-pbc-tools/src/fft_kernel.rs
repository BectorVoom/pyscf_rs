//! Host complex 1-D FFT kernels — the `stockham` engine's core (plan 11-03).
//!
//! # Why this is host code and not a `#[cube]` kernel
//!
//! PBC-MASTER-PLAN plan 11-03 sketches the fast FFT as a cubecl radix-2/3/5
//! Stockham kernel. That shape is a poor fit for the numbers this milestone
//! actually runs on, and this module is the measured deviation (the same kind
//! of deviation plan 10-03 recorded for its Bloch contraction):
//!
//! * the default diamond mesh is `[47, 47, 47]` — 47 is PRIME, so a radix-2/3/5
//!   decomposition never applies and the kernel would fall through to its
//!   Bluestein path on every axis anyway;
//! * `pyscf-algebra`'s CPU runtime — the default backend — sustains ~5 GFLOP/s
//!   on the `(141376, 47) x (47, 47)` products the GEMM engine issues, so the
//!   device route is not the fast route here;
//! * the transform is a strict `O(n log n)` reshuffle of scalar work with no
//!   reduction across units, so there is no ordered-reduction hazard
//!   (Pitfall 2) in doing it on the host: every output element is produced by
//!   ONE fixed sequence of butterflies, independent of thread count.
//!
//! The GEMM engine ([`crate::fft::fft_blas`]) remains the reference, is a
//! statement-for-statement port of upstream's own default engine, and gates
//! this one to 1e-13 (D-PBC-06).
//!
//! # Algorithms
//!
//! One plan per length, cached:
//!
//! | length | plan | cost |
//! |---|---|---|
//! | power of two | iterative radix-2 Cooley-Tukey with bit-reversal | `n/2 log2 n` butterflies |
//! | `n <= 40`, not a power of two | direct `O(n^2)` DFT off a shared twiddle ring | `n^2` |
//! | anything else | Bluestein chirp-z over a padded power of two `m >= 2n-1` | `~m log2 m` |
//!
//! Only the FORWARD (`e^{-2 pi i jk/n}`) transform is planned. The inverse is
//! `conj(forward(conj(x)))`, which is exact — sign flips introduce no rounding.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

/// Non-power-of-two lengths at or below this use the direct `O(n^2)` DFT.
/// Above it Bluestein wins: at `n = 47` the direct route is 2209 complex
/// multiplies against Bluestein's ~1150.
const DIRECT_MAX: usize = 40;

/// A planned length-`n` complex FFT.
#[derive(Debug)]
pub struct Fft1d {
    n: usize,
    plan: Plan,
}

#[derive(Debug)]
enum Plan {
    /// Length 0 or 1 — the identity.
    Identity,
    /// Direct `O(n^2)` DFT. `wr`/`wi` are `exp(-2 pi i j / n)` for `j in 0..n`.
    Direct { wr: Vec<f64>, wi: Vec<f64> },
    /// Iterative radix-2. `wr`/`wi` are `exp(-2 pi i j / n)` for `j in 0..n/2`.
    Radix2 { wr: Vec<f64>, wi: Vec<f64> },
    /// Bluestein chirp-z over a padded power of two.
    Bluestein {
        m: usize,
        /// `exp(-i pi k^2 / n)`, `k in 0..n`.
        chirp_re: Vec<f64>,
        chirp_im: Vec<f64>,
        /// Forward transform of the zero-padded, symmetric conjugate chirp.
        ker_re: Vec<f64>,
        ker_im: Vec<f64>,
        inner: Box<Fft1d>,
    },
}

impl Fft1d {
    /// Plan a length-`n` forward transform.
    pub fn new(n: usize) -> Self {
        if n <= 1 {
            return Self {
                n,
                plan: Plan::Identity,
            };
        }
        if n.is_power_of_two() {
            let half = n / 2;
            let mut wr = Vec::with_capacity(half);
            let mut wi = Vec::with_capacity(half);
            for j in 0..half {
                let a = -2.0 * PI * j as f64 / n as f64;
                wr.push(a.cos());
                wi.push(a.sin());
            }
            return Self {
                n,
                plan: Plan::Radix2 { wr, wi },
            };
        }
        if n <= DIRECT_MAX {
            let mut wr = Vec::with_capacity(n);
            let mut wi = Vec::with_capacity(n);
            for j in 0..n {
                let a = -2.0 * PI * j as f64 / n as f64;
                wr.push(a.cos());
                wi.push(a.sin());
            }
            return Self {
                n,
                plan: Plan::Direct { wr, wi },
            };
        }

        // Bluestein. `m` is the smallest power of two with `m >= 2n - 1`, so the
        // linear convolution of two length-`n` sequences fits without wrap-around.
        let mut m = 1usize;
        while m < 2 * n - 1 {
            m <<= 1;
        }
        let inner = Box::new(Fft1d::new(m));

        // chirp[k] = exp(-i pi k^2 / n). `k^2` is reduced mod `2n` first: the
        // angle is periodic with period `2n` in `k^2`, and reducing keeps the
        // argument of cos/sin small enough that its own rounding stays at 1 ulp
        // of pi rather than of `k^2 pi`.
        let mut chirp_re = Vec::with_capacity(n);
        let mut chirp_im = Vec::with_capacity(n);
        for k in 0..n {
            let idx = (k * k) % (2 * n);
            let a = -PI * idx as f64 / n as f64;
            chirp_re.push(a.cos());
            chirp_im.push(a.sin());
        }

        // The convolution kernel is the CONJUGATE chirp, laid out symmetrically
        // over the padded window: b[0..n] and b[m-n+1..m] mirror each other.
        let mut ker_re = vec![0.0_f64; m];
        let mut ker_im = vec![0.0_f64; m];
        for k in 0..n {
            ker_re[k] = chirp_re[k];
            ker_im[k] = -chirp_im[k];
            if k > 0 {
                ker_re[m - k] = chirp_re[k];
                ker_im[m - k] = -chirp_im[k];
            }
        }
        inner.forward(&mut ker_re, &mut ker_im);

        Self {
            n,
            plan: Plan::Bluestein {
                m,
                chirp_re,
                chirp_im,
                ker_re,
                ker_im,
                inner,
            },
        }
    }

    /// Planned length.
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` for the degenerate length-0 plan.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward transform in place: `X[k] = sum_j x[j] exp(-2 pi i jk/n)`.
    ///
    /// # Panics
    /// Debug-asserts `re.len() == im.len() == n`.
    pub fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        debug_assert_eq!(re.len(), self.n);
        debug_assert_eq!(im.len(), self.n);
        match &self.plan {
            Plan::Identity => {}
            Plan::Direct { wr, wi } => direct_dft(re, im, wr, wi),
            Plan::Radix2 { wr, wi } => radix2(re, im, wr, wi),
            Plan::Bluestein {
                m,
                chirp_re,
                chirp_im,
                ker_re,
                ker_im,
                inner,
            } => bluestein(re, im, *m, chirp_re, chirp_im, ker_re, ker_im, inner),
        }
    }

    /// Backward (unnormalised) transform in place:
    /// `x[j] = sum_k X[k] exp(+2 pi i jk/n)`.
    ///
    /// Implemented as `conj(forward(conj(X)))`. The two conjugations are pure
    /// sign flips, so this adds no rounding of its own.
    pub fn backward(&self, re: &mut [f64], im: &mut [f64]) {
        for v in im.iter_mut() {
            *v = -*v;
        }
        self.forward(re, im);
        for v in im.iter_mut() {
            *v = -*v;
        }
    }
}

/// `O(n^2)` DFT against a length-`n` twiddle ring.
fn direct_dft(re: &mut [f64], im: &mut [f64], wr: &[f64], wi: &[f64]) {
    let n = re.len();
    let xr = re.to_vec();
    let xi = im.to_vec();
    for k in 0..n {
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        let mut idx = 0usize;
        for j in 0..n {
            let (cr, ci) = (wr[idx], wi[idx]);
            sr += xr[j] * cr - xi[j] * ci;
            si += xr[j] * ci + xi[j] * cr;
            idx += k;
            if idx >= n {
                idx -= n;
            }
        }
        re[k] = sr;
        im[k] = si;
    }
}

/// Iterative radix-2 Cooley-Tukey, decimation in time.
fn radix2(re: &mut [f64], im: &mut [f64], wr: &[f64], wi: &[f64]) {
    let n = re.len();
    // Bit-reversal permutation.
    let shift = usize::BITS - n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let step = n / size;
        let mut i = 0usize;
        while i < n {
            let mut k = 0usize;
            for j in i..i + half {
                let l = j + half;
                let (cr, ci) = (wr[k], wi[k]);
                let tr = re[l] * cr - im[l] * ci;
                let ti = re[l] * ci + im[l] * cr;
                re[l] = re[j] - tr;
                im[l] = im[j] - ti;
                re[j] += tr;
                im[j] += ti;
                k += step;
            }
            i += size;
        }
        size <<= 1;
    }
}

/// Bluestein chirp-z transform.
#[allow(clippy::too_many_arguments)]
fn bluestein(
    re: &mut [f64],
    im: &mut [f64],
    m: usize,
    chirp_re: &[f64],
    chirp_im: &[f64],
    ker_re: &[f64],
    ker_im: &[f64],
    inner: &Fft1d,
) {
    let n = re.len();
    let mut ar = vec![0.0_f64; m];
    let mut ai = vec![0.0_f64; m];
    for k in 0..n {
        let (cr, ci) = (chirp_re[k], chirp_im[k]);
        ar[k] = re[k] * cr - im[k] * ci;
        ai[k] = re[k] * ci + im[k] * cr;
    }
    inner.forward(&mut ar, &mut ai);
    for k in 0..m {
        let (xr, xi) = (ar[k], ai[k]);
        let (yr, yi) = (ker_re[k], ker_im[k]);
        ar[k] = xr * yr - xi * yi;
        ai[k] = xr * yi + xi * yr;
    }
    inner.backward(&mut ar, &mut ai);
    let inv = 1.0 / m as f64;
    for k in 0..n {
        let (cr, ci) = (chirp_re[k], chirp_im[k]);
        let (xr, xi) = (ar[k] * inv, ai[k] * inv);
        re[k] = xr * cr - xi * ci;
        im[k] = xr * ci + xi * cr;
    }
}

/// Process-wide plan cache. Planning a length is `O(m log m)` (Bluestein has to
/// transform its kernel once), and the periodic code path transforms the same
/// three lengths millions of times, so the cache is load-bearing rather than a
/// convenience.
fn plan_cache() -> &'static Mutex<HashMap<usize, &'static Fft1d>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, &'static Fft1d>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The cached plan for length `n`.
///
/// Plans are leaked deliberately: there is one per distinct FFT axis length in
/// a process (three, typically), they are immutable, and handing out `&'static`
/// removes an `Arc` clone from the inner loop.
pub fn plan(n: usize) -> &'static Fft1d {
    let mut cache = match plan_cache().lock() {
        Ok(c) => c,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some(p) = cache.get(&n) {
        return p;
    }
    let leaked: &'static Fft1d = Box::leak(Box::new(Fft1d::new(n)));
    cache.insert(n, leaked);
    leaked
}

/// Transform along the middle axis of an `(outer, n, inner)` row-major array.
///
/// `inner == 1` is the contiguous case and runs in place on sub-slices; larger
/// strides gather into a scratch vector and scatter back.
pub fn transform_axis(
    re: &mut [f64],
    im: &mut [f64],
    outer: usize,
    n: usize,
    inner: usize,
    backward: bool,
) {
    debug_assert_eq!(re.len(), outer * n * inner);
    debug_assert_eq!(im.len(), outer * n * inner);
    if n <= 1 || outer == 0 || inner == 0 {
        return;
    }
    let p = plan(n);

    if inner == 1 {
        for o in 0..outer {
            let s = o * n;
            let (r, i) = (&mut re[s..s + n], &mut im[s..s + n]);
            if backward {
                p.backward(r, i);
            } else {
                p.forward(r, i);
            }
        }
        return;
    }

    let mut br = vec![0.0_f64; n];
    let mut bi = vec![0.0_f64; n];
    for o in 0..outer {
        let base = o * n * inner;
        for t in 0..inner {
            for j in 0..n {
                let p0 = base + j * inner + t;
                br[j] = re[p0];
                bi[j] = im[p0];
            }
            if backward {
                p.backward(&mut br, &mut bi);
            } else {
                p.forward(&mut br, &mut bi);
            }
            for j in 0..n {
                let p0 = base + j * inner + t;
                re[p0] = br[j];
                im[p0] = bi[j];
            }
        }
    }
}
