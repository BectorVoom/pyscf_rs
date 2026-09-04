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
//! One plan per length, cached (W-02, `KRKS-OPTIMISATION-PLAN.md`):
//!
//! | length | plan | cost |
//! |---|---|---|
//! | power of two | iterative radix-2 Cooley-Tukey with bit-reversal | `n/2 log2 n` butterflies |
//! | `n <= 7`, not a power of two | direct `O(n^2)` DFT off a shared twiddle ring — the base-case codelet both for a length that small on its own AND for one radix stage of [`Plan::MixedRadix`] below | `n^2` (`<= 49`) |
//! | composite, `n = n1 * n2` with `n1` a proper factor (`n1` preferred from `{7,5,4,3,2}`, else the smallest general factor) | [`Plan::MixedRadix`]: transpose to `(n1, n2)`, recurse on each length-`n2` row, twiddle, then a length-`n1` direct-DFT codelet on each column | `O(n (n1 + log n))`-ish, recursively |
//! | prime, `n > 7` | [`Plan::Rader`]: permute by a primitive root mod `n` into a length-`(n-1)` cyclic convolution, done as two length-`(n-1)` transforms (recursing into `MixedRadix`/`Radix2`/`Bluestein` as `n-1` factors) plus a pointwise multiply | `O(n log n)`-ish when `n-1` factors well |
//! | fallback (should not trigger once Rader is wired to every prime factor a `MixedRadix` stage can hand it) | Bluestein chirp-z over a padded power of two `m >= 2n-1` | `~m log2 m` |
//!
//! `n <= 7` deliberately covers `{3, 5, 6, 7}` (`{2, 4}` are already powers of
//! two), not just `{2, 3, 4}`: `5` and `7` are exactly the two prime radices
//! the mixed-radix codelet set names, and routing a bare length-5 or length-7
//! transform through Bluestein instead of the trivial `O(25)`/`O(49)` direct
//! DFT would be a regression, not an optimisation — Bluestein pads to the
//! next power of two `>= 2n-1` (16 for both), which is MORE arithmetic than
//! the direct codelet for a size this small.
//!
//! Only the FORWARD (`e^{-2 pi i jk/n}`) transform is planned. The inverse is
//! `conj(forward(conj(x)))`, which is exact — sign flips introduce no rounding.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

use rayon::prelude::*;

/// Non-power-of-two lengths at or below this use the direct `O(n^2)` DFT —
/// see the module-level doc comment for why `7` (not `4`) is the cutoff.
const DIRECT_MAX: usize = 7;

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
    /// W-02: mixed-radix Cooley-Tukey, `n = n1 * n2`. See
    /// [`mixed_radix_forward`] for the transpose/recurse/twiddle/codelet
    /// steps this runs.
    MixedRadix {
        n1: usize,
        n2: usize,
        /// The length-`n1` direct-DFT codelet's twiddle ring,
        /// `exp(-2 pi i j / n1)` for `j in 0..n1`.
        wr1: Vec<f64>,
        wi1: Vec<f64>,
        /// Cross-stage twiddles, `tw[j1*n2+k2] = exp(-2 pi i j1 k2 / n)`.
        tw_re: Vec<f64>,
        tw_im: Vec<f64>,
        /// The length-`n2` sub-plan each of the `n1` rows recurses into.
        inner: Box<Fft1d>,
    },
    /// W-02: Rader's algorithm for prime `n > 7`. See [`rader_forward`].
    Rader {
        /// `idx_in[a] = g^a mod n`, `a in 0..n-1` — the input permutation by
        /// powers of the primitive root `g`.
        idx_in: Vec<usize>,
        /// `idx_out[r] = g^{-r} mod n`, `r in 0..n-1` — the output
        /// permutation by powers of `g^{-1} mod n`.
        idx_out: Vec<usize>,
        /// The length-`(n-1)` forward transform of the REVERSED kernel
        /// `B[c] = exp(-2 pi i (g^c mod n) / n)`, i.e. `FFT(Brev)` where
        /// `Brev[m] = B[(n-1-m) mod (n-1)]` — precomputed once so the
        /// per-call cost is one forward transform, one pointwise multiply,
        /// one inverse transform (see [`rader_forward`] for the derivation).
        ker_re: Vec<f64>,
        ker_im: Vec<f64>,
        /// The length-`(n-1)` sub-plan the convolution runs through.
        inner: Box<Fft1d>,
    },
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
            let (wr, wi) = direct_twiddles(n);
            return Self {
                n,
                plan: Plan::Direct { wr, wi },
            };
        }

        if let Some(n1) = choose_radix(n) {
            let n2 = n / n1;
            let (wr1, wi1) = direct_twiddles(n1);
            let mut tw_re = vec![0.0_f64; n1 * n2];
            let mut tw_im = vec![0.0_f64; n1 * n2];
            for j1 in 0..n1 {
                for k2 in 0..n2 {
                    let a = -2.0 * PI * (j1 * k2) as f64 / n as f64;
                    tw_re[j1 * n2 + k2] = a.cos();
                    tw_im[j1 * n2 + k2] = a.sin();
                }
            }
            let inner = Box::new(Fft1d::new(n2));
            return Self {
                n,
                plan: Plan::MixedRadix {
                    n1,
                    n2,
                    wr1,
                    wi1,
                    tw_re,
                    tw_im,
                    inner,
                },
            };
        }

        // `n` is prime and `> DIRECT_MAX` — Rader's algorithm, §"Algorithms"
        // above, and (having exhausted `n<=1`, power-of-two, `n<=DIRECT_MAX`
        // and composite above) the last case, so this is the function's tail
        // expression. `n - 1` is even (n is an odd prime here: n=2 is a power
        // of two, handled above), so its own plan recurses into `MixedRadix`
        // / `Radix2` rather than needing Bluestein itself — see
        // [`choose_radix`]. `build_bluestein` below is kept, unreferenced, as
        // the risk-mitigation fallback `KRKS-OPTIMISATION-PLAN.md` W-02 names
        // ("keep Bluestein as the prime fallback").
        let p = n as u64;
        let g = primitive_root(p);
        let g_inv = mod_inverse(g, p);
        let m = n - 1; // the cyclic-convolution length
        let mut idx_in = vec![0usize; m];
        let mut idx_out = vec![0usize; m];
        let mut cur_in = 1u64;
        let mut cur_out = 1u64;
        // `ker[c] = exp(-2 pi i (g^c mod n) / n)` for `c in 0..m` — the
        // "B" kernel of the derivation, NOT yet reversed.
        let mut ker_re = vec![0.0_f64; m];
        let mut ker_im = vec![0.0_f64; m];
        for c in 0..m {
            idx_in[c] = cur_in as usize;
            idx_out[c] = cur_out as usize;
            let a = -2.0 * PI * cur_in as f64 / n as f64;
            ker_re[c] = a.cos();
            ker_im[c] = a.sin();
            cur_in = cur_in * g % p;
            cur_out = cur_out * g_inv % p;
        }
        // `Brev[m'] = B[(m - m') mod m]` (see [`rader_forward`]).
        let mut brev_re = vec![0.0_f64; m];
        let mut brev_im = vec![0.0_f64; m];
        brev_re[0] = ker_re[0];
        brev_im[0] = ker_im[0];
        for c in 1..m {
            brev_re[c] = ker_re[m - c];
            brev_im[c] = ker_im[m - c];
        }
        let inner = Box::new(Fft1d::new(m));
        inner.forward(&mut brev_re, &mut brev_im);
        Self {
            n,
            plan: Plan::Rader {
                idx_in,
                idx_out,
                ker_re: brev_re,
                ker_im: brev_im,
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
            Plan::MixedRadix {
                n1,
                n2,
                wr1,
                wi1,
                tw_re,
                tw_im,
                inner,
            } => mixed_radix_forward(re, im, *n1, *n2, wr1, wi1, tw_re, tw_im, inner),
            Plan::Rader {
                idx_in,
                idx_out,
                ker_re,
                ker_im,
                inner,
            } => rader_forward(re, im, idx_in, idx_out, ker_re, ker_im, inner),
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

/// `exp(-2 pi i j / n)` for `j in 0..n` — the direct-DFT twiddle ring shared
/// by [`Plan::Direct`] and the length-`n1` codelet stage of
/// [`Plan::MixedRadix`].
fn direct_twiddles(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut wr = Vec::with_capacity(n);
    let mut wi = Vec::with_capacity(n);
    for j in 0..n {
        let a = -2.0 * PI * j as f64 / n as f64;
        wr.push(a.cos());
        wi.push(a.sin());
    }
    (wr, wi)
}

/// W-02: the Bluestein construction that used to be `Fft1d::new`'s universal
/// fallback for every non-power-of-two length. `Fft1d::new` no longer calls
/// this — every composite length now routes through [`Plan::MixedRadix`] and
/// every prime length `> DIRECT_MAX` through [`Plan::Rader`] — but the
/// function is kept, working and untouched, as the risk-mitigation fallback
/// `KRKS-OPTIMISATION-PLAN.md` W-02 names ("keep Bluestein as the prime
/// fallback"): if a future defect narrows what `primitive_root`/
/// `choose_radix` can handle, wiring their failure path back to this
/// function is a one-line change, not a rewrite.
#[allow(dead_code)]
fn build_bluestein(n: usize) -> Plan {
    let mut m = 1usize;
    while m < 2 * n - 1 {
        m <<= 1;
    }
    let inner = Box::new(Fft1d::new(m));

    let mut chirp_re = Vec::with_capacity(n);
    let mut chirp_im = Vec::with_capacity(n);
    for k in 0..n {
        let idx = (k * k) % (2 * n);
        let a = -PI * idx as f64 / n as f64;
        chirp_re.push(a.cos());
        chirp_im.push(a.sin());
    }

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

    Plan::Bluestein {
        m,
        chirp_re,
        chirp_im,
        ker_re,
        ker_im,
        inner,
    }
}

/// The proper factor `Fft1d::new` peels off next: `r` from the preferred
/// codelet set `{7, 5,4, 3, 2}` if one divides `n` (largest first, to
/// minimise recursion depth), else the smallest general factor up to
/// `sqrt(n)` (the "generic radix-r stage" `KRKS-OPTIMISATION-PLAN.md` W-02
/// step 1 describes). `None` means `n` is prime.
fn choose_radix(n: usize) -> Option<usize> {
    for r in [7usize, 5, 4, 3, 2] {
        if r < n && n % r == 0 {
            return Some(r);
        }
    }
    let mut f = 8usize;
    while f * f <= n {
        if n % f == 0 {
            return Some(f);
        }
        f += 1;
    }
    None
}

/// `base^exp mod modu`, using `u128` intermediates so the squaring never
/// overflows for any `modu` that fits a PBC mesh axis.
fn mod_pow(mut base: u64, mut exp: u64, modu: u64) -> u64 {
    let mut result = 1u64 % modu;
    base %= modu;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modu as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modu as u128) as u64;
    }
    result
}

/// Distinct prime factors of `n`, by trial division. `n` here is always
/// `p - 1` for a prime `p` no larger than a realistic FFT mesh axis, so trial
/// division to `sqrt(n)` is cheap and this never runs on the hot path (it is
/// plan-build-time only, cached by [`plan`]).
fn prime_factors_distinct(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

/// The smallest primitive root mod the prime `p`: the smallest `g` whose
/// order in `(Z/pZ)*` is exactly `p - 1`, found by checking
/// `g^{(p-1)/q} != 1 (mod p)` for every distinct prime factor `q` of `p - 1`
/// (a candidate failing this for some `q` has order dividing `(p-1)/q`, not
/// the full group).
fn primitive_root(p: u64) -> u64 {
    let phi = p - 1;
    let factors = prime_factors_distinct(phi);
    let mut g = 2u64;
    loop {
        if factors.iter().all(|&f| mod_pow(g, phi / f, p) != 1) {
            return g;
        }
        g += 1;
    }
}

/// `a^{-1} mod p` for prime `p`, via Fermat's little theorem
/// (`a^{p-2} = a^{-1} mod p`) — avoids a separate extended-Euclid
/// implementation for a value only ever needed once per [`Plan::Rader`] plan.
fn mod_inverse(a: u64, p: u64) -> u64 {
    mod_pow(a, p - 2, p)
}

/// W-02: mixed-radix Cooley-Tukey, `n = n1 * n2`.
///
/// Derivation (`j = j1 + n1*j2`, `k = k2 + n2*k1`, `j1,k1 in 0..n1`,
/// `j2,k2 in 0..n2`):
///
/// ```text
/// X[k2 + n2*k1] = sum_{j1} W_n1^{j1*k1} * ( W_n^{j1*k2} * sum_{j2} x[j1+n1*j2] * W_n2^{j2*k2} )
/// ```
///
/// `x` viewed as `(n2, n1)` row-major is `x` itself (no data movement); the
/// inner sum over `j2` is therefore a length-`n2` transform of COLUMN `j1` of
/// that view. Step by step:
///
/// 1. Transpose `(n2, n1)` into `M`, `(n1, n2)` row-major (`M[j1][j2] =
///    x[j2*n1+j1]`).
/// 2. Recurse: forward-transform each length-`n2` row of `M` in place
///    (`inner`).
/// 3. Twiddle: `M[j1][k2] *= exp(-2 pi i j1 k2 / n)`.
/// 4. For each column `k2`, a length-`n1` DIRECT DFT over `j1` (the `wr1`/
///    `wi1` codelet — `n1 <= 7` by construction, so `O(n1^2) <= 49` ops) gives
///    the `n1` output values `Z[k1]`; scatter `Z[k1]` to output index
///    `k1*n2+k2`, which is exactly `(n1, n2)` row-major — the FINAL layout,
///    no further rearrangement needed.
#[allow(clippy::too_many_arguments)]
fn mixed_radix_forward(
    re: &mut [f64],
    im: &mut [f64],
    n1: usize,
    n2: usize,
    wr1: &[f64],
    wi1: &[f64],
    tw_re: &[f64],
    tw_im: &[f64],
    inner: &Fft1d,
) {
    let n = n1 * n2;
    debug_assert_eq!(re.len(), n);

    // Step 1: transpose (n2, n1) -> (n1, n2).
    let mut mre = vec![0.0_f64; n];
    let mut mim = vec![0.0_f64; n];
    for j2 in 0..n2 {
        for j1 in 0..n1 {
            mre[j1 * n2 + j2] = re[j2 * n1 + j1];
            mim[j1 * n2 + j2] = im[j2 * n1 + j1];
        }
    }

    // Step 2: recurse on each length-n2 row.
    for j1 in 0..n1 {
        let b = j1 * n2;
        inner.forward(&mut mre[b..b + n2], &mut mim[b..b + n2]);
    }

    // Step 3: cross-stage twiddle.
    for j1 in 1..n1 {
        let b = j1 * n2;
        let tb = j1 * n2;
        for k2 in 0..n2 {
            let (xr, xi) = (mre[b + k2], mim[b + k2]);
            let (cr, ci) = (tw_re[tb + k2], tw_im[tb + k2]);
            mre[b + k2] = xr * cr - xi * ci;
            mim[b + k2] = xr * ci + xi * cr;
        }
    }

    // Step 4: length-n1 direct codelet per column, scattered straight into
    // final (n1, n2) row-major position.
    let mut cr = vec![0.0_f64; n1];
    let mut ci = vec![0.0_f64; n1];
    for k2 in 0..n2 {
        for j1 in 0..n1 {
            cr[j1] = mre[j1 * n2 + k2];
            ci[j1] = mim[j1 * n2 + k2];
        }
        direct_dft(&mut cr, &mut ci, wr1, wi1);
        for k1 in 0..n1 {
            re[k1 * n2 + k2] = cr[k1];
            im[k1 * n2 + k2] = ci[k1];
        }
    }
}

/// W-02: Rader's algorithm for prime `n`.
///
/// `X[0] = sum_j x[j]` (plain accumulation; no ordered-reduction hazard per
/// the module doc — this is one fixed-length sequential sum, same as every
/// other scalar in this file, and is never split across threads).
///
/// For `k != 0`: writing every nonzero residue as a power of the primitive
/// root `g`, `j = g^a mod n` and using `k = g^{-r} mod n` (inverse powers,
/// NOT `g^{+r}`, for the output index) turns the `O(n^2)` sum into the
/// length-`(n-1)` cyclic convolution `conv(A, Brev)`, where `A[a] =
/// x[g^a mod n]`, `B[c] = exp(-2 pi i (g^c mod n) / n)` and `Brev[m] =
/// B[(-m) mod (n-1)]` (derivation: `X[g^{-r}] = x[0] + sum_a A[a] B[(a-r) mod
/// (n-1)] = x[0] + conv(A, Brev)[r]` — `x[0]` here is the SINGLE input sample
/// at index 0, not `X[0]`; the two are easy to conflate and only differ once
/// `n > 1`, which is why a bug here survives a spot check at small `n` and is
/// worth this explicit callout). `Brev`'s forward transform is precomputed
/// once at plan-build time ([`Fft1d::new`]), so the per-call cost is one
/// forward transform of `A`, one pointwise multiply, one inverse transform,
/// and the `idx_in`/`idx_out` gather/scatter.
fn rader_forward(
    re: &mut [f64],
    im: &mut [f64],
    idx_in: &[usize],
    idx_out: &[usize],
    ker_re: &[f64],
    ker_im: &[f64],
    inner: &Fft1d,
) {
    let m = idx_in.len(); // n - 1
    // x[0]: the single sample, the additive constant for every k != 0 output.
    let x_first_re = re[0];
    let x_first_im = im[0];
    // X[0] = sum_j x[j]: the DC output, goes to output position 0 ONLY.
    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;
    for i in 0..re.len() {
        sum_re += re[i];
        sum_im += im[i];
    }

    let mut ar = vec![0.0_f64; m];
    let mut ai = vec![0.0_f64; m];
    for a in 0..m {
        ar[a] = re[idx_in[a]];
        ai[a] = im[idx_in[a]];
    }
    inner.forward(&mut ar, &mut ai);
    for c in 0..m {
        let (xr, xi) = (ar[c], ai[c]);
        let (yr, yi) = (ker_re[c], ker_im[c]);
        ar[c] = xr * yr - xi * yi;
        ai[c] = xr * yi + xi * yr;
    }
    inner.backward(&mut ar, &mut ai);

    let inv = 1.0 / m as f64;
    re[0] = sum_re;
    im[0] = sum_im;
    for r in 0..m {
        let k = idx_out[r];
        re[k] = x_first_re + ar[r] * inv;
        im[k] = x_first_im + ai[r] * inv;
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
///
/// # W-02b — parallel over `outer`, bit-exact
///
/// The `outer` blocks are disjoint length-`n*inner` slices, and each one's
/// `n`-point transform is a fixed sequence of butterflies/DFT sums that does
/// not depend on how many OTHER blocks run alongside it — there is no
/// reduction shared across blocks. Splitting the batch across `rayon`
/// workers therefore changes nothing about any individual output value: the
/// result is bit-identical to the serial loop for any `RAYON_NUM_THREADS`
/// (`10_grid_stride_occupancy.md` §6 — disjoint writes are bit-exact on every
/// backend). Do NOT parallelise *inside* one length-`n` transform.
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
        re.par_chunks_mut(n)
            .zip(im.par_chunks_mut(n))
            .for_each(|(r, i)| {
                if backward {
                    p.backward(r, i);
                } else {
                    p.forward(r, i);
                }
            });
        return;
    }

    let block = n * inner;
    re.par_chunks_mut(block)
        .zip(im.par_chunks_mut(block))
        .for_each(|(re_o, im_o)| {
            let mut br = vec![0.0_f64; n];
            let mut bi = vec![0.0_f64; n];
            for t in 0..inner {
                for j in 0..n {
                    let p0 = j * inner + t;
                    br[j] = re_o[p0];
                    bi[j] = im_o[p0];
                }
                if backward {
                    p.backward(&mut br, &mut bi);
                } else {
                    p.forward(&mut br, &mut bi);
                }
                for j in 0..n {
                    let p0 = j * inner + t;
                    re_o[p0] = br[j];
                    im_o[p0] = bi[j];
                }
            }
        });
}
