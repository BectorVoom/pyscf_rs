//! Bit-faithful port of pocketfft's complex FFTPACK transform — the engine
//! behind upstream's `scipy.fft.fftn`/`ifftn`, which `tools.fft`/`tools.ifft`
//! call for every mesh that is NOT routed to `_fftn_blas` (`pbc.py:128-140`).
//!
//! # Source
//!
//! scipy 1.17.1 vendors pocketfft as the `scipy/_lib/pocketfft` submodule at
//! commit `9367142748fcc9696a1c9e5a99b76ed9897c9daa`; everything here follows
//! that commit's `pocketfft_hdronly.h` (`cfftp`, `sincos_2pibyn`,
//! `pocketfft_c`, `general_nd`/`ExecC2C`) and `pypocketfft.cxx`'s `norm_fct`
//! statement for statement, so that the rounding sequence — not just the
//! mathematics — is upstream's. [`crate::fft::fft_stockham`] computes the same
//! transform to ~1e-16 but in a different order, which is enough to move
//! `get_nuc` off the upstream bits.
//!
//! Three long-double details are reproduced exactly:
//!
//! * the twiddle angle `Thigh(0.25L*pi/n)` is an 80-bit quotient rounded
//!   twice (to 64 then 53 bits) — [`ld_quotient_to_f64`];
//! * the normalisation `T(1/ldbl_t(N))` likewise;
//! * the pass constants are `T0(<long double literal>)`; every one of them
//!   rounds to the same double as its decimal literal parsed directly (checked
//!   with exact rational arithmetic), so they are plain `f64` literals here.
//!
//! # Scope
//!
//! The `cfftp` (mixed-radix) plan and the `fftblue` (Bluestein chirp-Z) plan,
//! selected per length by [`uses_bluestein`] exactly as `pocketfft_c` selects
//! them — every length upstream can plan is covered.

// The pass constants are pocketfft's long-double literals, digit for digit.
#![allow(clippy::excessive_precision)]

/// One complex value, laid out as pocketfft's `cmplx<double>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Cmplx {
    pub r: f64,
    pub i: f64,
}

impl Cmplx {
    #[inline]
    fn new(r: f64, i: f64) -> Self {
        Self { r, i }
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.r + o.r, self.i + o.i)
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.r - o.r, self.i - o.i)
    }
    /// `cmplx * T0` — `{r*other, i*other}`.
    #[inline]
    fn scale(self, x: f64) -> Self {
        Self::new(self.r * x, self.i * x)
    }
}

/// `PM(a, b, c, d)`: `a = c + d; b = c - d`.
#[inline]
fn pm(c: Cmplx, d: Cmplx) -> (Cmplx, Cmplx) {
    (c.add(d), c.sub(d))
}

/// `PMINPLACE(a, b)`: `t = a; a += b; b = t - b`.
#[inline]
fn pm_inplace(a: &mut Cmplx, b: &mut Cmplx) {
    let t = *a;
    *a = a.add(*b);
    *b = t.sub(*b);
}

/// `special_mul<fwd>(v1, v2, res)`.
#[inline]
fn special_mul(fwd: bool, v1: Cmplx, v2: Cmplx) -> Cmplx {
    if fwd {
        Cmplx::new(v1.r * v2.r + v1.i * v2.i, v1.i * v2.r - v1.r * v2.i)
    } else {
        Cmplx::new(v1.r * v2.r - v1.i * v2.i, v1.r * v2.i + v1.i * v2.r)
    }
}

/// `ROTX90<fwd>(a)`.
#[inline]
fn rotx90(fwd: bool, a: &mut Cmplx) {
    let tmp = if fwd { -a.r } else { a.r };
    a.r = if fwd { a.i } else { -a.i };
    a.i = tmp;
}

/// `T0(0.707106781186547524400844362104849L)` — rounds to exactly `1/sqrt(2)`.
const HSQT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// `ROTX45<fwd>(a)`.
#[inline]
fn rotx45(fwd: bool, a: &mut Cmplx) {
    let tmp = a.r;
    if fwd {
        a.r = HSQT2 * (a.r + a.i);
        a.i = HSQT2 * (a.i - tmp);
    } else {
        a.r = HSQT2 * (a.r - a.i);
        a.i = HSQT2 * (a.i + tmp);
    }
}

/// `ROTX135<fwd>(a)`.
#[inline]
fn rotx135(fwd: bool, a: &mut Cmplx) {
    let tmp = a.r;
    if fwd {
        a.r = HSQT2 * (a.i - a.r);
        a.i = HSQT2 * (-tmp - a.i);
    } else {
        a.r = HSQT2 * (-a.r - a.i);
        a.i = HSQT2 * (tmp - a.i);
    }
}

/// A signed sum of products written with the C preprocessor's token pasting,
/// e.g. `cb.i = +w1*a -w2*b +w3*c`: the first term is always `+`, every later
/// one is added or subtracted as written.
#[inline]
fn signed_sum(terms: &[(bool, f64, f64)]) -> f64 {
    let mut acc = terms[0].1 * terms[0].2;
    for &(plus, w, t) in &terms[1..] {
        if plus {
            acc += w * t;
        } else {
            acc -= w * t;
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Long-double arithmetic
// ---------------------------------------------------------------------------

/// The x87 80-bit value of pocketfft's `pi` literal: `0xC90FDAA22168C235 * 2^-62`.
const PI_LD_MANTISSA: u64 = 0xC90F_DAA2_2168_C235;

/// `double(long double(mant * 2^exp) / long double(den))` with x87 semantics:
/// the quotient is rounded to a 64-bit mantissa (nearest-even), then to 53.
/// `mant` must be normalised (top bit set); `den` must be non-zero.
pub fn ld_quotient_to_f64(mant: u64, exp: i32, den: u64) -> f64 {
    debug_assert!(mant >> 63 == 1 && den > 0);
    let num = (mant as u128) << 64;
    let q = num / den as u128;
    let rem = num % den as u128;
    // `mant >= 2^63` and `den < 2^64` put `q` in [2^63, 2^128).
    let bits = 128 - q.leading_zeros() as i32;
    let shift = bits - 64;
    let mut m64: u128;
    let mut e = exp - 64 + shift.max(0);
    if shift > 0 {
        m64 = q >> shift;
        let dropped = q & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);
        let sticky = rem != 0;
        if dropped > half || (dropped == half && (sticky || m64 & 1 == 1)) {
            m64 += 1;
        }
    } else {
        m64 = q;
        // bits == 64: the remainder alone decides.
        if rem * 2 > den as u128 || (rem * 2 == den as u128 && m64 & 1 == 1) {
            m64 += 1;
        }
    }
    if m64 >> 64 == 1 {
        m64 >>= 1;
        e += 1;
    }
    // Round the 64-bit mantissa to 53 bits — no sticky beyond it: this is the
    // second rounding of a value that already IS the long double.
    let mut m53 = (m64 >> 11) as u64;
    let dropped = (m64 & 0x7ff) as u64;
    if dropped > 0x400 || (dropped == 0x400 && m53 & 1 == 1) {
        m53 += 1;
    }
    let mut e53 = e + 11;
    if m53 >> 53 == 1 {
        m53 >>= 1;
        e53 += 1;
    }
    (m53 as f64) * 2f64.powi(e53)
}

/// `T(1/ldbl_t(N))` — `pypocketfft.cxx`'s `norm_fct` for `inorm == 2`.
pub fn norm_fct_inv(n: u64) -> f64 {
    ld_quotient_to_f64(1u64 << 63, -63, n)
}

// ---------------------------------------------------------------------------
// sincos_2pibyn
// ---------------------------------------------------------------------------

struct SinCos2PiByN {
    n: usize,
    mask: usize,
    shift: u32,
    v1: Vec<Cmplx>,
    v2: Vec<Cmplx>,
}

impl SinCos2PiByN {
    fn calc(mut x: usize, n: usize, ang: f64) -> Cmplx {
        x <<= 3;
        let c = |v: usize| (v as f64 * ang).cos();
        let s = |v: usize| (v as f64 * ang).sin();
        if x < 4 * n {
            if x < 2 * n {
                if x < n {
                    return Cmplx::new(c(x), s(x));
                }
                return Cmplx::new(s(2 * n - x), c(2 * n - x));
            }
            x -= 2 * n;
            if x < n {
                return Cmplx::new(-s(x), c(x));
            }
            Cmplx::new(-c(2 * n - x), s(2 * n - x))
        } else {
            x = 8 * n - x;
            if x < 2 * n {
                if x < n {
                    return Cmplx::new(c(x), -s(x));
                }
                return Cmplx::new(s(2 * n - x), -c(2 * n - x));
            }
            x -= 2 * n;
            if x < n {
                return Cmplx::new(-s(x), -c(x));
            }
            Cmplx::new(-c(2 * n - x), -s(2 * n - x))
        }
    }

    fn new(n: usize) -> Self {
        // Thigh ang = Thigh(0.25L*pi/n): 0.25L*pi is exact (exponent shift).
        let ang = ld_quotient_to_f64(PI_LD_MANTISSA, -64, n as u64);
        let nval = (n + 2) / 2;
        let mut shift = 1u32;
        while (1usize << shift) * (1usize << shift) < nval {
            shift += 1;
        }
        let mask = (1usize << shift) - 1;
        let mut v1 = vec![Cmplx::default(); mask + 1];
        v1[0] = Cmplx::new(1.0, 0.0);
        for (i, v) in v1.iter_mut().enumerate().skip(1) {
            *v = Self::calc(i, n, ang);
        }
        let mut v2 = vec![Cmplx::default(); nval.div_ceil(mask + 1)];
        v2[0] = Cmplx::new(1.0, 0.0);
        for (i, v) in v2.iter_mut().enumerate().skip(1) {
            *v = Self::calc(i * (mask + 1), n, ang);
        }
        Self {
            n,
            mask,
            shift,
            v1,
            v2,
        }
    }

    fn get(&self, idx: usize) -> Cmplx {
        if 2 * idx <= self.n {
            let x1 = self.v1[idx & self.mask];
            let x2 = self.v2[idx >> self.shift];
            return Cmplx::new(x1.r * x2.r - x1.i * x2.i, x1.r * x2.i + x1.i * x2.r);
        }
        let idx = self.n - idx;
        let x1 = self.v1[idx & self.mask];
        let x2 = self.v2[idx >> self.shift];
        Cmplx::new(x1.r * x2.r - x1.i * x2.i, -(x1.r * x2.i + x1.i * x2.r))
    }
}

// ---------------------------------------------------------------------------
// Plan selection (`pocketfft_c`) and Bluestein (`fftblue`)
// ---------------------------------------------------------------------------

fn largest_prime_factor(mut n: usize) -> usize {
    let mut res = 1;
    while n & 1 == 0 {
        res = 2;
        n >>= 1;
    }
    let mut x = 3;
    while x * x <= n {
        while n.is_multiple_of(x) {
            res = x;
            n /= x;
        }
        x += 2;
    }
    if n > 1 {
        res = n;
    }
    res
}

fn cost_guess(mut n: usize) -> f64 {
    const LFP: f64 = 1.1;
    let ni = n;
    let mut result = 0.0;
    while n & 1 == 0 {
        result += 2.0;
        n >>= 1;
    }
    let mut x = 3;
    while x * x <= n {
        while n.is_multiple_of(x) {
            result += if x <= 5 { x as f64 } else { LFP * x as f64 };
            n /= x;
        }
        x += 2;
    }
    if n > 1 {
        result += if n <= 5 { n as f64 } else { LFP * n as f64 };
    }
    result * ni as f64
}

/// The smallest composite of 2, 3, 5, 7 and 11 that is `>= n`.
fn good_size_cmplx(n: usize) -> usize {
    if n <= 12 {
        return n;
    }
    let mut bestfac = 2 * n;
    let mut f11 = 1;
    while f11 < bestfac {
        let mut f117 = f11;
        while f117 < bestfac {
            let mut f1175 = f117;
            while f1175 < bestfac {
                let mut x = f1175;
                while x < n {
                    x *= 2;
                }
                loop {
                    if x < n {
                        x *= 3;
                    } else if x > n {
                        if x < bestfac {
                            bestfac = x;
                        }
                        if x & 1 == 1 {
                            break;
                        }
                        x >>= 1;
                    } else {
                        return n;
                    }
                }
                f1175 *= 5;
            }
            f117 *= 7;
        }
        f11 *= 11;
    }
    bestfac
}

/// `true` when pocketfft plans `length` with Bluestein (`fftblue`) instead of
/// `cfftp` — the case this port does not cover.
pub fn uses_bluestein(length: usize) -> bool {
    let tmp = if length < 50 {
        0
    } else {
        largest_prime_factor(length)
    };
    if tmp * tmp <= length {
        return false;
    }
    let comp1 = cost_guess(length);
    let comp2 = 2.0 * cost_guess(good_size_cmplx(2 * length - 1)) * 1.5;
    comp2 < comp1
}

/// pocketfft's `fftblue<double>` plan (Bluestein chirp-Z for lengths with a
/// large prime factor).
///
/// Follows `pocketfft_hdronly.h`'s `fftblue` statement for statement: the `bk`
/// table from `sincos_2pibyn tmp(2*n)` with the `coeff = (coeff + 2m-1) mod 2n`
/// recurrence, `bkf` from the forward inner FFT of the zero-padded `b_k`
/// scaled by the PLAIN-double `xn2 = 1/n2` (not the long-double quotient of
/// [`norm_fct_inv`]), and `fft<fwd>`'s `special_mul` convolutions with the
/// inner `cfftp` of length `n2`. `fct` is applied once, on the final
/// multiply — the inner `exec` calls all use `1.0`.
pub struct Fftblue {
    n: usize,
    n2: usize,
    plan: Cfftp,
    bk: Vec<Cmplx>,
    /// Only `0..=n2/2` are stored, like upstream's `bkf` (`mem(n+n2/2+1)`);
    /// the convolution loop never reads past `n2/2`.
    bkf: Vec<Cmplx>,
}

impl Fftblue {
    /// Build the plan. `length` must be non-zero.
    pub fn new(length: usize) -> Self {
        assert!(length > 0, "zero-length FFT requested");
        let n = length;
        let n2 = good_size_cmplx(n * 2 - 1);
        let plan = Cfftp::new(n2);
        // `sincos_2pibyn<T0> tmp(2*n)`, `bk[0] = (1, 0)`.
        let tmp = SinCos2PiByN::new(2 * n);
        let mut bk = vec![Cmplx::default(); n];
        bk[0] = Cmplx::new(1.0, 0.0);
        let mut coeff = 0usize;
        for (m, slot) in bk.iter_mut().enumerate().take(n).skip(1) {
            coeff += 2 * m - 1;
            if coeff >= 2 * n {
                coeff -= 2 * n;
            }
            *slot = tmp.get(coeff);
        }
        // The zero-padded, Fourier-transformed `b_k`, normalised by the
        // plain-double `T0 xn2 = T0(1)/T0(n2)`.
        let xn2 = 1.0 / n2 as f64;
        let mut tbkf = vec![Cmplx::default(); n2];
        let b0 = bk[0].scale(xn2);
        tbkf[0] = b0;
        for m in 1..n {
            let v = bk[m].scale(xn2);
            tbkf[m] = v;
            tbkf[n2 - m] = v;
        }
        plan.exec(&mut tbkf, 1.0, true);
        let bkf = tbkf[..n2 / 2 + 1].to_vec();
        Self {
            n,
            n2,
            plan,
            bk,
            bkf,
        }
    }

    fn fft(&self, c: &mut [Cmplx], fct: f64, fwd: bool) {
        let (n, n2) = (self.n, self.n2);
        let mut akf = vec![Cmplx::default(); n2];
        for m in 0..n {
            akf[m] = special_mul(fwd, c[m], self.bk[m]);
        }
        // `auto zero = akf[0]*T0(0)`: sign-preserving zeros, not `(0, 0)`.
        let zero = Cmplx::new(akf[0].r * 0.0, akf[0].i * 0.0);
        for slot in akf.iter_mut().take(n2).skip(n) {
            *slot = zero;
        }
        self.plan.exec(&mut akf, 1.0, true);
        // `special_mul<!fwd>`: the convolution undoes the forward chirp.
        let back = !fwd;
        akf[0] = special_mul(back, akf[0], self.bkf[0]);
        for m in 1..n2.div_ceil(2) {
            akf[m] = special_mul(back, akf[m], self.bkf[m]);
            akf[n2 - m] = special_mul(back, akf[n2 - m], self.bkf[m]);
        }
        if n2 & 1 == 0 {
            akf[n2 / 2] = special_mul(back, akf[n2 / 2], self.bkf[n2 / 2]);
        }
        self.plan.exec(&mut akf, 1.0, false);
        for m in 0..n {
            c[m] = special_mul(fwd, akf[m], self.bk[m]).scale(fct);
        }
    }

    /// `exec(c, fct, fwd)` — `fft<true>` vs `fft<false>`.
    pub fn exec(&self, c: &mut [Cmplx], fct: f64, fwd: bool) {
        if fwd {
            self.fft(c, fct, true);
        } else {
            self.fft(c, fct, false);
        }
    }
}

/// One flexible 1-D plan: `pocketfft_c`'s `packplan` vs `blueplan` choice,
/// which is exactly [`uses_bluestein`].
pub enum Plan1d {
    /// Mixed-radix `cfftp` path.
    Pack(Cfftp),
    /// Bluestein `fftblue` path.
    Blue(Fftblue),
}

impl Plan1d {
    /// Build the plan upstream would build for `length` (non-zero).
    pub fn new(length: usize) -> Self {
        if uses_bluestein(length) {
            Self::Blue(Fftblue::new(length))
        } else {
            Self::Pack(Cfftp::new(length))
        }
    }

    /// Transform `c` in place, scaling by `fct` exactly where upstream does.
    pub fn exec(&self, c: &mut [Cmplx], fct: f64, fwd: bool) {
        match self {
            Self::Pack(p) => p.exec(c, fct, fwd),
            Self::Blue(b) => b.exec(c, fct, fwd),
        }
    }
}

// ---------------------------------------------------------------------------
// cfftp
// ---------------------------------------------------------------------------

struct Factor {
    fct: usize,
    tw: Vec<Cmplx>,
    tws: Vec<Cmplx>,
}

/// pocketfft's `cfftp<double>` plan.
pub struct Cfftp {
    length: usize,
    fact: Vec<Factor>,
}

impl Cfftp {
    /// Build the plan. `length` must be non-zero.
    pub fn new(length: usize) -> Self {
        assert!(length > 0, "zero-length FFT requested");
        let mut plan = Self {
            length,
            fact: Vec::new(),
        };
        if length == 1 {
            return plan;
        }
        plan.factorize();
        plan.comp_twiddle();
        plan
    }

    fn add_factor(&mut self, f: usize) {
        self.fact.push(Factor {
            fct: f,
            tw: Vec::new(),
            tws: Vec::new(),
        });
    }

    fn factorize(&mut self) {
        let mut len = self.length;
        while len & 7 == 0 {
            self.add_factor(8);
            len >>= 3;
        }
        while len & 3 == 0 {
            self.add_factor(4);
            len >>= 2;
        }
        if len & 1 == 0 {
            len >>= 1;
            self.add_factor(2);
            let last = self.fact.len() - 1;
            let (f0, fl) = (self.fact[0].fct, self.fact[last].fct);
            self.fact[0].fct = fl;
            self.fact[last].fct = f0;
        }
        let mut divisor = 3;
        while divisor * divisor <= len {
            while len.is_multiple_of(divisor) {
                self.add_factor(divisor);
                len /= divisor;
            }
            divisor += 2;
        }
        if len > 1 {
            self.add_factor(len);
        }
    }

    fn comp_twiddle(&mut self) {
        let twiddle = SinCos2PiByN::new(self.length);
        let mut l1 = 1;
        let length = self.length;
        for f in &mut self.fact {
            let ip = f.fct;
            let ido = length / (l1 * ip);
            f.tw = vec![Cmplx::default(); (ip - 1) * (ido - 1)];
            for j in 1..ip {
                for i in 1..ido {
                    f.tw[(j - 1) * (ido - 1) + i - 1] = twiddle.get(j * l1 * i);
                }
            }
            if ip > 11 {
                f.tws = (0..ip).map(|j| twiddle.get(j * l1 * ido)).collect();
            }
            l1 *= ip;
        }
    }

    /// `pass_all<fwd>(c, fct)` — transform `c` in place, then scale by `fct`
    /// exactly where upstream does.
    pub fn exec(&self, c: &mut [Cmplx], fct: f64, fwd: bool) {
        let length = self.length;
        if length == 1 {
            c[0] = c[0].scale(fct);
            return;
        }
        let mut ch = vec![Cmplx::default(); length];
        // `p1_is_c` tracks upstream's `p1 == c` after the pointer swaps.
        let mut p1_is_c = true;
        let mut l1 = 1;
        for f in &self.fact {
            let ip = f.fct;
            let l2 = ip * l1;
            let ido = length / l2;
            let (src, dst): (&mut [Cmplx], &mut [Cmplx]) = if p1_is_c {
                (&mut *c, &mut ch[..])
            } else {
                (&mut ch[..], &mut *c)
            };
            match ip {
                4 => pass4(fwd, ido, l1, src, dst, &f.tw),
                8 => pass8(fwd, ido, l1, src, dst, &f.tw),
                2 => pass2(fwd, ido, l1, src, dst, &f.tw),
                3 => pass3(fwd, ido, l1, src, dst, &f.tw),
                5 => pass5(fwd, ido, l1, src, dst, &f.tw),
                7 => pass7(fwd, ido, l1, src, dst, &f.tw),
                11 => pass11(fwd, ido, l1, src, dst, &f.tw),
                _ => {
                    passg(fwd, ido, ip, l1, src, dst, &f.tw, &f.tws);
                    // passg leaves its result in `cc`: upstream swaps twice.
                    p1_is_c = !p1_is_c;
                }
            }
            p1_is_c = !p1_is_c;
            l1 = l2;
        }
        if !p1_is_c {
            if fct != 1.0 {
                for (ci, hi) in c.iter_mut().zip(&ch) {
                    *ci = hi.scale(fct);
                }
            } else {
                c.copy_from_slice(&ch);
            }
        } else if fct != 1.0 {
            for ci in c.iter_mut() {
                *ci = ci.scale(fct);
            }
        }
    }
}

/// One `POCKETFFT_PARTSTEP` line: output slots `(u1, u2)`, the real twiddles
/// and the signed imaginary ones.
type Step<const N: usize> = (usize, usize, [f64; N], [(bool, f64); N]);

// Index helpers shared by the passes: CH(a,b,c) = ch[a+ido*(b+l1*c)],
// CC(a,b,c) = cc[a+ido*(b+cdim*c)], WA(x,i) = wa[i-1+x*(ido-1)].
#[inline]
fn ich(ido: usize, l1: usize, a: usize, b: usize, c: usize) -> usize {
    a + ido * (b + l1 * c)
}
#[inline]
fn icc(ido: usize, cdim: usize, a: usize, b: usize, c: usize) -> usize {
    a + ido * (b + cdim * c)
}
#[inline]
fn wa(w: &[Cmplx], ido: usize, x: usize, i: usize) -> Cmplx {
    w[i - 1 + x * (ido - 1)]
}

fn pass2(fwd: bool, ido: usize, l1: usize, cc: &[Cmplx], ch: &mut [Cmplx], w: &[Cmplx]) {
    for k in 0..l1 {
        let (a, b) = (cc[icc(ido, 2, 0, 0, k)], cc[icc(ido, 2, 0, 1, k)]);
        ch[ich(ido, l1, 0, k, 0)] = a.add(b);
        ch[ich(ido, l1, 0, k, 1)] = a.sub(b);
        for i in 1..ido {
            let (a, b) = (cc[icc(ido, 2, i, 0, k)], cc[icc(ido, 2, i, 1, k)]);
            ch[ich(ido, l1, i, k, 0)] = a.add(b);
            ch[ich(ido, l1, i, k, 1)] = special_mul(fwd, a.sub(b), wa(w, ido, 0, i));
        }
    }
}

fn pass3(fwd: bool, ido: usize, l1: usize, cc: &[Cmplx], ch: &mut [Cmplx], w: &[Cmplx]) {
    let tw1r = -0.5;
    let tw1i = if fwd { -1.0 } else { 1.0 } * 0.8660254037844386467637231707529362;
    for k in 0..l1 {
        for i in 0..ido {
            let t0 = cc[icc(ido, 3, i, 0, k)];
            let (t1, t2) = pm(cc[icc(ido, 3, i, 1, k)], cc[icc(ido, 3, i, 2, k)]);
            ch[ich(ido, l1, i, k, 0)] = t0.add(t1);
            let ca = t0.add(t1.scale(tw1r));
            let cb = Cmplx::new(-t2.i * tw1i, t2.r * tw1i);
            if i == 0 {
                let (x, y) = pm(ca, cb);
                ch[ich(ido, l1, 0, k, 1)] = x;
                ch[ich(ido, l1, 0, k, 2)] = y;
            } else {
                ch[ich(ido, l1, i, k, 1)] = special_mul(fwd, ca.add(cb), wa(w, ido, 0, i));
                ch[ich(ido, l1, i, k, 2)] = special_mul(fwd, ca.sub(cb), wa(w, ido, 1, i));
            }
        }
    }
}

fn pass4(fwd: bool, ido: usize, l1: usize, cc: &[Cmplx], ch: &mut [Cmplx], w: &[Cmplx]) {
    for k in 0..l1 {
        {
            let (t2, t1) = pm(cc[icc(ido, 4, 0, 0, k)], cc[icc(ido, 4, 0, 2, k)]);
            let (t3, mut t4) = pm(cc[icc(ido, 4, 0, 1, k)], cc[icc(ido, 4, 0, 3, k)]);
            rotx90(fwd, &mut t4);
            let (a, b) = pm(t2, t3);
            ch[ich(ido, l1, 0, k, 0)] = a;
            ch[ich(ido, l1, 0, k, 2)] = b;
            let (a, b) = pm(t1, t4);
            ch[ich(ido, l1, 0, k, 1)] = a;
            ch[ich(ido, l1, 0, k, 3)] = b;
        }
        for i in 1..ido {
            let cc0 = cc[icc(ido, 4, i, 0, k)];
            let cc1 = cc[icc(ido, 4, i, 1, k)];
            let cc2 = cc[icc(ido, 4, i, 2, k)];
            let cc3 = cc[icc(ido, 4, i, 3, k)];
            let (t2, t1) = pm(cc0, cc2);
            let (t3, mut t4) = pm(cc1, cc3);
            rotx90(fwd, &mut t4);
            ch[ich(ido, l1, i, k, 0)] = t2.add(t3);
            ch[ich(ido, l1, i, k, 1)] = special_mul(fwd, t1.add(t4), wa(w, ido, 0, i));
            ch[ich(ido, l1, i, k, 2)] = special_mul(fwd, t2.sub(t3), wa(w, ido, 1, i));
            ch[ich(ido, l1, i, k, 3)] = special_mul(fwd, t1.sub(t4), wa(w, ido, 2, i));
        }
    }
}

fn pass5(fwd: bool, ido: usize, l1: usize, cc: &[Cmplx], ch: &mut [Cmplx], w: &[Cmplx]) {
    let s = if fwd { -1.0 } else { 1.0 };
    let tw1r = 0.3090169943749474241022934171828191;
    let tw1i = s * 0.9510565162951535721164393333793821;
    let tw2r = -0.8090169943749474241022934171828191;
    let tw2i = s * 0.5877852522924731291687059546390728;
    for k in 0..l1 {
        for i in 0..ido {
            let t0 = cc[icc(ido, 5, i, 0, k)];
            let (t1, t4) = pm(cc[icc(ido, 5, i, 1, k)], cc[icc(ido, 5, i, 4, k)]);
            let (t2, t3) = pm(cc[icc(ido, 5, i, 2, k)], cc[icc(ido, 5, i, 3, k)]);
            ch[ich(ido, l1, i, k, 0)] = Cmplx::new(t0.r + t1.r + t2.r, t0.i + t1.i + t2.i);
            // (u1, u2, twar, twbr, (sign, twai), (sign, twbi))
            let steps = [
                (1, 4, tw1r, tw2r, (true, tw1i), (true, tw2i)),
                (2, 3, tw2r, tw1r, (true, tw2i), (false, tw1i)),
            ];
            for (u1, u2, twar, twbr, (_, twai), (sb, twbi)) in steps {
                let ca = Cmplx::new(
                    t0.r + twar * t1.r + twbr * t2.r,
                    t0.i + twar * t1.i + twbr * t2.i,
                );
                let cb = Cmplx::new(
                    -signed_sum(&[(true, twai, t4.i), (sb, twbi, t3.i)]),
                    signed_sum(&[(true, twai, t4.r), (sb, twbi, t3.r)]),
                );
                if i == 0 {
                    let (x, y) = pm(ca, cb);
                    ch[ich(ido, l1, 0, k, u1)] = x;
                    ch[ich(ido, l1, 0, k, u2)] = y;
                } else {
                    ch[ich(ido, l1, i, k, u1)] =
                        special_mul(fwd, ca.add(cb), wa(w, ido, u1 - 1, i));
                    ch[ich(ido, l1, i, k, u2)] =
                        special_mul(fwd, ca.sub(cb), wa(w, ido, u2 - 1, i));
                }
            }
        }
    }
}

fn pass7(fwd: bool, ido: usize, l1: usize, cc: &[Cmplx], ch: &mut [Cmplx], w: &[Cmplx]) {
    let s = if fwd { -1.0 } else { 1.0 };
    let tw1r = 0.6234898018587335305250048840042398;
    let tw1i = s * 0.7818314824680298087084445266740578;
    let tw2r = -0.2225209339563144042889025644967948;
    let tw2i = s * 0.9749279121818236070181316829939312;
    let tw3r = -0.9009688679024191262361023195074451;
    let tw3i = s * 0.433883739117558120475768332848359;
    // (u1, u2, [x1, x2, x3], [(sign, y); 3])
    let steps: [Step<3>; 3] = [
        (
            1,
            6,
            [tw1r, tw2r, tw3r],
            [(true, tw1i), (true, tw2i), (true, tw3i)],
        ),
        (
            2,
            5,
            [tw2r, tw3r, tw1r],
            [(true, tw2i), (false, tw3i), (false, tw1i)],
        ),
        (
            3,
            4,
            [tw3r, tw1r, tw2r],
            [(true, tw3i), (false, tw1i), (true, tw2i)],
        ),
    ];
    for k in 0..l1 {
        for i in 0..ido {
            let t1 = cc[icc(ido, 7, i, 0, k)];
            let (t2, t7) = pm(cc[icc(ido, 7, i, 1, k)], cc[icc(ido, 7, i, 6, k)]);
            let (t3, t6) = pm(cc[icc(ido, 7, i, 2, k)], cc[icc(ido, 7, i, 5, k)]);
            let (t4, t5) = pm(cc[icc(ido, 7, i, 3, k)], cc[icc(ido, 7, i, 4, k)]);
            ch[ich(ido, l1, i, k, 0)] =
                Cmplx::new(t1.r + t2.r + t3.r + t4.r, t1.i + t2.i + t3.i + t4.i);
            for (u1, u2, x, y) in steps {
                let ca = Cmplx::new(
                    t1.r + x[0] * t2.r + x[1] * t3.r + x[2] * t4.r,
                    t1.i + x[0] * t2.i + x[1] * t3.i + x[2] * t4.i,
                );
                let cb = Cmplx::new(
                    -signed_sum(&[
                        (y[0].0, y[0].1, t7.i),
                        (y[1].0, y[1].1, t6.i),
                        (y[2].0, y[2].1, t5.i),
                    ]),
                    signed_sum(&[
                        (y[0].0, y[0].1, t7.r),
                        (y[1].0, y[1].1, t6.r),
                        (y[2].0, y[2].1, t5.r),
                    ]),
                );
                let (da, db) = pm(ca, cb);
                if i == 0 {
                    ch[ich(ido, l1, 0, k, u1)] = da;
                    ch[ich(ido, l1, 0, k, u2)] = db;
                } else {
                    ch[ich(ido, l1, i, k, u1)] = special_mul(fwd, da, wa(w, ido, u1 - 1, i));
                    ch[ich(ido, l1, i, k, u2)] = special_mul(fwd, db, wa(w, ido, u2 - 1, i));
                }
            }
        }
    }
}

fn pass8(fwd: bool, ido: usize, l1: usize, cc: &[Cmplx], ch: &mut [Cmplx], w: &[Cmplx]) {
    let c = |i: usize, j: usize, k: usize| cc[icc(ido, 8, i, j, k)];
    for k in 0..l1 {
        {
            let (mut a1, mut a5) = pm(c(0, 1, k), c(0, 5, k));
            let (mut a3, mut a7) = pm(c(0, 3, k), c(0, 7, k));
            pm_inplace(&mut a1, &mut a3);
            rotx90(fwd, &mut a3);

            rotx90(fwd, &mut a7);
            pm_inplace(&mut a5, &mut a7);
            rotx45(fwd, &mut a5);
            rotx135(fwd, &mut a7);

            let (a0, a4) = pm(c(0, 0, k), c(0, 4, k));
            let (a2, mut a6) = pm(c(0, 2, k), c(0, 6, k));
            let (x, y) = pm(a0.add(a2), a1);
            ch[ich(ido, l1, 0, k, 0)] = x;
            ch[ich(ido, l1, 0, k, 4)] = y;
            let (x, y) = pm(a0.sub(a2), a3);
            ch[ich(ido, l1, 0, k, 2)] = x;
            ch[ich(ido, l1, 0, k, 6)] = y;
            rotx90(fwd, &mut a6);
            let (x, y) = pm(a4.add(a6), a5);
            ch[ich(ido, l1, 0, k, 1)] = x;
            ch[ich(ido, l1, 0, k, 5)] = y;
            let (x, y) = pm(a4.sub(a6), a7);
            ch[ich(ido, l1, 0, k, 3)] = x;
            ch[ich(ido, l1, 0, k, 7)] = y;
        }
        for i in 1..ido {
            let (mut a1, mut a5) = pm(c(i, 1, k), c(i, 5, k));
            let (mut a3, mut a7) = pm(c(i, 3, k), c(i, 7, k));
            rotx90(fwd, &mut a7);
            pm_inplace(&mut a1, &mut a3);
            rotx90(fwd, &mut a3);
            pm_inplace(&mut a5, &mut a7);
            rotx45(fwd, &mut a5);
            rotx135(fwd, &mut a7);
            let (mut a0, mut a4) = pm(c(i, 0, k), c(i, 4, k));
            let (mut a2, mut a6) = pm(c(i, 2, k), c(i, 6, k));
            pm_inplace(&mut a0, &mut a2);
            ch[ich(ido, l1, i, k, 0)] = a0.add(a1);
            ch[ich(ido, l1, i, k, 4)] = special_mul(fwd, a0.sub(a1), wa(w, ido, 3, i));
            ch[ich(ido, l1, i, k, 2)] = special_mul(fwd, a2.add(a3), wa(w, ido, 1, i));
            ch[ich(ido, l1, i, k, 6)] = special_mul(fwd, a2.sub(a3), wa(w, ido, 5, i));
            rotx90(fwd, &mut a6);
            pm_inplace(&mut a4, &mut a6);
            ch[ich(ido, l1, i, k, 1)] = special_mul(fwd, a4.add(a5), wa(w, ido, 0, i));
            ch[ich(ido, l1, i, k, 5)] = special_mul(fwd, a4.sub(a5), wa(w, ido, 4, i));
            ch[ich(ido, l1, i, k, 3)] = special_mul(fwd, a6.add(a7), wa(w, ido, 2, i));
            ch[ich(ido, l1, i, k, 7)] = special_mul(fwd, a6.sub(a7), wa(w, ido, 6, i));
        }
    }
}

fn pass11(fwd: bool, ido: usize, l1: usize, cc: &[Cmplx], ch: &mut [Cmplx], w: &[Cmplx]) {
    let s = if fwd { -1.0 } else { 1.0 };
    let tw1r = 0.8412535328311811688618116489193677;
    let tw1i = s * 0.5406408174555975821076359543186917;
    let tw2r = 0.4154150130018864255292741492296232;
    let tw2i = s * 0.9096319953545183714117153830790285;
    let tw3r = -0.1423148382732851404437926686163697;
    let tw3i = s * 0.9898214418809327323760920377767188;
    let tw4r = -0.6548607339452850640569250724662936;
    let tw4i = s * 0.7557495743542582837740358439723444;
    let tw5r = -0.9594929736144973898903680570663277;
    let tw5i = s * 0.2817325568414296977114179153466169;
    let (p, m) = (true, false);
    // (u1, u2, [x1..x5], [(sign, y1)..(sign, y5)]) — `pass11`'s five
    // POCKETFFT_PARTSTEP11 lines, verbatim.
    let steps: [Step<5>; 5] = [
        (
            1,
            10,
            [tw1r, tw2r, tw3r, tw4r, tw5r],
            [(p, tw1i), (p, tw2i), (p, tw3i), (p, tw4i), (p, tw5i)],
        ),
        (
            2,
            9,
            [tw2r, tw4r, tw5r, tw3r, tw1r],
            [(p, tw2i), (p, tw4i), (m, tw5i), (m, tw3i), (m, tw1i)],
        ),
        (
            3,
            8,
            [tw3r, tw5r, tw2r, tw1r, tw4r],
            [(p, tw3i), (m, tw5i), (m, tw2i), (p, tw1i), (p, tw4i)],
        ),
        (
            4,
            7,
            [tw4r, tw3r, tw1r, tw5r, tw2r],
            [(p, tw4i), (m, tw3i), (p, tw1i), (p, tw5i), (m, tw2i)],
        ),
        (
            5,
            6,
            [tw5r, tw1r, tw4r, tw2r, tw3r],
            [(p, tw5i), (m, tw1i), (p, tw4i), (m, tw2i), (p, tw3i)],
        ),
    ];
    for k in 0..l1 {
        for i in 0..ido {
            let q = |j: usize| cc[icc(ido, 11, i, j, k)];
            let t1 = q(0);
            let (t2, t11) = pm(q(1), q(10));
            let (t3, t10) = pm(q(2), q(9));
            let (t4, t9) = pm(q(3), q(8));
            let (t5, t8) = pm(q(4), q(7));
            let (t6, t7) = pm(q(5), q(6));
            ch[ich(ido, l1, i, k, 0)] = Cmplx::new(
                t1.r + t2.r + t3.r + t4.r + t5.r + t6.r,
                t1.i + t2.i + t3.i + t4.i + t5.i + t6.i,
            );
            let tb = [t11, t10, t9, t8, t7];
            for (u1, u2, x, y) in steps {
                // T ca = t1 + t2*x1 + t3*x2 + t4*x3 + t5*x4 + t6*x5
                let ca = t1
                    .add(t2.scale(x[0]))
                    .add(t3.scale(x[1]))
                    .add(t4.scale(x[2]))
                    .add(t5.scale(x[3]))
                    .add(t6.scale(x[4]));
                let im: [(bool, f64, f64); 5] = std::array::from_fn(|j| (y[j].0, y[j].1, tb[j].r));
                let re: [(bool, f64, f64); 5] = std::array::from_fn(|j| (y[j].0, y[j].1, tb[j].i));
                let cb = Cmplx::new(-signed_sum(&re), signed_sum(&im));
                let (da, db) = pm(ca, cb);
                if i == 0 {
                    ch[ich(ido, l1, 0, k, u1)] = da;
                    ch[ich(ido, l1, 0, k, u2)] = db;
                } else {
                    ch[ich(ido, l1, i, k, u1)] = special_mul(fwd, da, wa(w, ido, u1 - 1, i));
                    ch[ich(ido, l1, i, k, u2)] = special_mul(fwd, db, wa(w, ido, u2 - 1, i));
                }
            }
        }
    }
}

/// `passg` — the generic odd-prime pass. Its result lands back in `cc`.
#[allow(clippy::too_many_arguments)]
fn passg(
    fwd: bool,
    ido: usize,
    ip: usize,
    l1: usize,
    cc: &mut [Cmplx],
    ch: &mut [Cmplx],
    w: &[Cmplx],
    csarr: &[Cmplx],
) {
    let cdim = ip;
    let ipph = ip.div_ceil(2);
    let idl1 = ido * l1;
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let cci = |a: usize, b: usize, c: usize| a + ido * (b + cdim * c);
    let cxi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let x2i = |a: usize, b: usize| a + idl1 * b;

    let mut wal = vec![Cmplx::default(); ip];
    wal[0] = Cmplx::new(1.0, 0.0);
    for i in 1..ip {
        wal[i] = Cmplx::new(csarr[i].r, if fwd { -csarr[i].i } else { csarr[i].i });
    }

    for k in 0..l1 {
        for i in 0..ido {
            ch[chi(i, k, 0)] = cc[cci(i, 0, k)];
        }
    }
    let (mut j, mut jc) = (1, ip - 1);
    while j < ipph {
        for k in 0..l1 {
            for i in 0..ido {
                let (a, b) = pm(cc[cci(i, j, k)], cc[cci(i, jc, k)]);
                ch[chi(i, k, j)] = a;
                ch[chi(i, k, jc)] = b;
            }
        }
        j += 1;
        jc -= 1;
    }
    for k in 0..l1 {
        for i in 0..ido {
            let mut tmp = ch[chi(i, k, 0)];
            for j in 1..ipph {
                tmp = tmp.add(ch[chi(i, k, j)]);
            }
            cc[cxi(i, k, 0)] = tmp;
        }
    }
    let (mut l, mut lc) = (1, ip - 1);
    while l < ipph {
        for ik in 0..idl1 {
            let h = |b: usize| ch[x2i(ik, b)];
            cc[x2i(ik, l)] = Cmplx::new(
                h(0).r + wal[l].r * h(1).r + wal[2 * l].r * h(2).r,
                h(0).i + wal[l].r * h(1).i + wal[2 * l].r * h(2).i,
            );
            cc[x2i(ik, lc)] = Cmplx::new(
                -wal[l].i * h(ip - 1).i - wal[2 * l].i * h(ip - 2).i,
                wal[l].i * h(ip - 1).r + wal[2 * l].i * h(ip - 2).r,
            );
        }
        let mut iwal = 2 * l;
        let (mut j, mut jc) = (3, ip - 3);
        while j + 1 < ipph {
            iwal += l;
            if iwal > ip {
                iwal -= ip;
            }
            let xwal = wal[iwal];
            iwal += l;
            if iwal > ip {
                iwal -= ip;
            }
            let xwal2 = wal[iwal];
            for ik in 0..idl1 {
                let h = |b: usize| ch[x2i(ik, b)];
                let (hj, hj1, hjc, hjc1) = (h(j), h(j + 1), h(jc), h(jc - 1));
                let o = &mut cc[x2i(ik, l)];
                o.r += hj.r * xwal.r + hj1.r * xwal2.r;
                o.i += hj.i * xwal.r + hj1.i * xwal2.r;
                let o = &mut cc[x2i(ik, lc)];
                o.r -= hjc.i * xwal.i + hjc1.i * xwal2.i;
                o.i += hjc.r * xwal.i + hjc1.r * xwal2.i;
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            iwal += l;
            if iwal > ip {
                iwal -= ip;
            }
            let xwal = wal[iwal];
            for ik in 0..idl1 {
                let (hj, hjc) = (ch[x2i(ik, j)], ch[x2i(ik, jc)]);
                let o = &mut cc[x2i(ik, l)];
                o.r += hj.r * xwal.r;
                o.i += hj.i * xwal.r;
                let o = &mut cc[x2i(ik, lc)];
                o.r -= hjc.i * xwal.i;
                o.i += hjc.r * xwal.i;
            }
            j += 1;
            jc -= 1;
        }
        l += 1;
        lc -= 1;
    }

    // shuffling and twiddling
    let (mut j, mut jc) = (1, ip - 1);
    while j < ipph {
        if ido == 1 {
            for ik in 0..idl1 {
                let (t1, t2) = (cc[x2i(ik, j)], cc[x2i(ik, jc)]);
                let (a, b) = pm(t1, t2);
                cc[x2i(ik, j)] = a;
                cc[x2i(ik, jc)] = b;
            }
        } else {
            for k in 0..l1 {
                let (t1, t2) = (cc[cxi(0, k, j)], cc[cxi(0, k, jc)]);
                let (a, b) = pm(t1, t2);
                cc[cxi(0, k, j)] = a;
                cc[cxi(0, k, jc)] = b;
                for i in 1..ido {
                    let (x1, x2) = pm(cc[cxi(i, k, j)], cc[cxi(i, k, jc)]);
                    let idij = (j - 1) * (ido - 1) + i - 1;
                    cc[cxi(i, k, j)] = special_mul(fwd, x1, w[idij]);
                    let idij = (jc - 1) * (ido - 1) + i - 1;
                    cc[cxi(i, k, jc)] = special_mul(fwd, x2, w[idij]);
                }
            }
        }
        j += 1;
        jc -= 1;
    }
}

// ---------------------------------------------------------------------------
// N-D driver
// ---------------------------------------------------------------------------

/// `pocketfft::c2c` over every axis of a batch of row-major 3-D arrays, as
/// `scipy.fft.fftn/ifftn(a, axes=(1, 2, 3))` calls it: the axes are transformed
/// in the order x, y, z, and `fct` is applied by the FIRST axis only
/// (`general_nd`: "factor has been applied, use 1 for remaining axes").
///
/// Each axis uses the `cfftp` or Bluestein plan upstream would pick
/// ([`Plan1d`]).
///
/// # Panics
/// When any axis length is zero.
pub fn c2c_3d(
    re: &[f64],
    im: &[f64],
    mesh: [usize; 3],
    forward: bool,
    fct: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert!(!mesh.contains(&0), "c2c_3d: mesh {mesh:?} has a zero axis");
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    let n_batch = re.len() / ngrids;
    let mut data: Vec<Cmplx> = re.iter().zip(im).map(|(&r, &i)| Cmplx::new(r, i)).collect();
    let strides = [mesh[1] * mesh[2], mesh[2], 1];
    let mut fct = fct;
    for axis in 0..3 {
        let len = mesh[axis];
        let plan = Plan1d::new(len);
        let stride = strides[axis];
        let (oa, ob) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let mut line = vec![Cmplx::default(); len];
        for b in 0..n_batch {
            let base = b * ngrids;
            for p in 0..mesh[oa] {
                for q in 0..mesh[ob] {
                    let start = base + p * strides[oa] + q * strides[ob];
                    for (t, v) in line.iter_mut().enumerate() {
                        *v = data[start + t * stride];
                    }
                    plan.exec(&mut line, fct, forward);
                    for (t, v) in line.iter().enumerate() {
                        data[start + t * stride] = *v;
                    }
                }
            }
        }
        fct = 1.0;
    }
    data.iter().map(|c| (c.r, c.i)).unzip()
}
