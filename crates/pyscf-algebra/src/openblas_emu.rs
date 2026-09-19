//! Summation-order emulation of the `dgemm_`/`zgemm_` that upstream PySCF's
//! C extensions call — for the few places where a pyscf-rs result has to
//! reproduce upstream's bits, not just its value.
//!
//! # Which BLAS, and why its order is knowable
//!
//! The PySCF 2.12.1 wheel bundles its own OpenBLAS **0.3.3**
//! (`pyscf/lib/libopenblas-r0-f650aae0.3.3.so`, `DYNAMIC_ARCH`,
//! `SINGLE_THREADED`); `libnp_helper` (`lib.dot`) and `libpbc` (`PBCeval_*`)
//! link against it. That release predates Zen 3+, so on the development
//! machine (AMD Ryzen AI 7 350) it selects the **Barcelona** core: SSE2
//! kernels with no FMA. Every element is therefore a plain sum of
//! individually-rounded products, and only the ORDER of the additions varies —
//! which is what this module encodes. An Intel host would pick a Haswell or
//! SkylakeX kernel with FMA, and upstream's bits would change with it: these
//! emulators are specific to the Barcelona core, i.e. to the reference
//! machine the oracle runs on.
//!
//! # The model (measured, not read from the assembly)
//!
//! Fitted against the real `dgemm_`/`zgemm_` through ctypes over every
//! `M, N <= 10` and `K` up to 1331 (48 400 `dgemm` and 8 624 `zgemm` elements,
//! zero mismatches):
//!
//! * the level-3 driver splits `K` into blocks of `GEMM_Q = 224` (the last two
//!   blocks halved and rounded up to the M-unroll), and adds each block's
//!   kernel sum to `C` in turn: `C = (C + s_1) + s_2 ...`;
//! * `dgemm` (4x4 register tile): an element in a full 4-row block or the
//!   2-row tail, crossed with a full 4-column block or the 2-column tail, is a
//!   left-to-right sum. The 1-row and 1-column tails, and the 2x2 tail corner,
//!   use split accumulators —
//!   see [`d_kernel_sum`];
//! * `zgemm` (2x2 tile): `Re = S(ar*br) - S(ai*bi)` and
//!   `Im = S(ar*bi) + S(ai*br)`, four independent real sums, each
//!   left-to-right except in the 1x1 corner.

/// `GEMM_Q` of the Barcelona `dgemm` and `zgemm` drivers.
const GEMM_Q: usize = 224;

/// The `K` blocks the level-3 driver hands the kernel (`level3.c`):
/// `min_l = GEMM_Q` while `>= 2*GEMM_Q` remains, then the remainder is halved
/// (rounded up to `unroll_m`) once if it still exceeds `GEMM_Q`.
pub fn k_blocks(k: usize, unroll_m: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut ls = 0;
    while ls < k {
        let mut m = k - ls;
        if m >= 2 * GEMM_Q {
            m = GEMM_Q;
        } else if m > GEMM_Q {
            m = (m / 2).div_ceil(unroll_m) * unroll_m;
        }
        out.push(m);
        ls += m;
    }
    out
}

/// Position of a row (or column) relative to the register tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tile {
    /// Inside a full 4-wide block.
    Full,
    /// In the 2-wide tail.
    Tail2,
    /// The single trailing row/column.
    Tail1,
}

fn d_tile(i: usize, m: usize) -> Tile {
    let m4 = m / 4 * 4;
    if i < m4 {
        Tile::Full
    } else if m - m4 >= 2 && i < m4 + 2 {
        Tile::Tail2
    } else {
        Tile::Tail1
    }
}

fn seq(p: &[f64]) -> f64 {
    let mut s = 0.0;
    for x in p {
        s += x;
    }
    s
}

/// Two SSE lanes over a K loop unrolled by 4: even `k` into lane 0, odd into
/// lane 1; the `K % 4` leftovers go into lane 0; then `lane0 + lane1`.
fn two_lanes(p: &[f64]) -> f64 {
    let ku = p.len() / 4 * 4;
    let (mut a0, mut a1) = (0.0, 0.0);
    for pair in p[..ku].as_chunks::<2>().0 {
        a0 += pair[0];
        a1 += pair[1];
    }
    for x in &p[ku..] {
        a0 += x;
    }
    a0 + a1
}

/// Four accumulators over a K loop unrolled by 4 (`k % 4`), leftovers into
/// accumulator 0, combined as `(a0+a2)+(a1+a3)` (`pair_02`) or
/// `(a0+a1)+(a2+a3)`.
fn four_accs(p: &[f64], pair_02: bool) -> f64 {
    let ku = p.len() / 4 * 4;
    let mut a = [0.0; 4];
    for quad in p[..ku].as_chunks::<4>().0 {
        for (acc, x) in a.iter_mut().zip(quad) {
            *acc += x;
        }
    }
    for x in &p[ku..] {
        a[0] += x;
    }
    if pair_02 {
        (a[0] + a[2]) + (a[1] + a[3])
    } else {
        (a[0] + a[1]) + (a[2] + a[3])
    }
}

/// The Barcelona `dgemm` kernel's sum of one K block for an element whose row
/// and column sit at tile positions `(ri, cj)`.
fn d_kernel_sum(p: &[f64], ri: Tile, cj: Tile) -> f64 {
    use Tile::*;
    match (ri, cj) {
        (Full, Full) | (Full, Tail2) | (Tail2, Full) => seq(p),
        (Full, Tail1) | (Tail1, Full) | (Tail2, Tail2) => two_lanes(p),
        (Tail1, Tail1) => four_accs(p, true),
        (Tail1, Tail2) | (Tail2, Tail1) => four_accs(p, false),
    }
}

/// `dgemm_('N', 'T', m, n, k, 1.0, a, m, b, n, 1.0, c, m)` — `c += a * b^T`
/// in upstream's summation order. All three are column-major: `a` is `m x k`,
/// `b` is `n x k`, `c` is `m x n`.
///
/// With `beta = 0` upstream zeroes `c` first; pass a zeroed `c` for that.
pub fn dgemm_nt(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    assert!(a.len() >= m * k && b.len() >= n * k && c.len() >= m * n);
    let blocks = k_blocks(k, 4);
    let mut p = Vec::with_capacity(2 * GEMM_Q);
    for j in 0..n {
        let cj = d_tile(j, n);
        for i in 0..m {
            let ri = d_tile(i, m);
            let mut acc = c[i + j * m];
            let mut k0 = 0;
            for &kb in &blocks {
                p.clear();
                p.extend((k0..k0 + kb).map(|kk| a[i + kk * m] * b[j + kk * n]));
                acc += d_kernel_sum(&p, ri, cj);
                k0 += kb;
            }
            c[i + j * m] = acc;
        }
    }
}

/// `zgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)` —
/// `c = alpha * a * b^T + beta * c` (no conjugation) in upstream's summation
/// order, on planar complex column-major arrays.
///
/// Verified configurations (probed against the wheel's OpenBLAS, Barcelona
/// core — `target/probes-task4/`, 0 mismatches over ~150k elements plus
/// `K > GEMM_Q` cases): `('N', 'T')` with real `alpha` and `beta` 0 or 1, any
/// `m, n, k`. The model is the [`zgemm_nt`] one: `Re = S(ar*br) - S(ai*bi)`,
/// `Im = S(ar*bi) + S(ai*br)`, four independent real sums, each left-to-right
/// except in the 1x1 corner — with the real `alpha` scaling each K-block's
/// combined sums (`acc += alpha * block`, not a single scale at the end;
/// `K = 225..303` probes decide this). Anything else panics: an unverified
/// BLAS shape must refuse, not silently approximate.
#[allow(clippy::too_many_arguments)]
pub fn zgemm(
    transa: char,
    transb: char,
    m: usize,
    n: usize,
    k: usize,
    alpha: (f64, f64),
    a_re: &[f64],
    a_im: &[f64],
    lda: usize,
    b_re: &[f64],
    b_im: &[f64],
    ldb: usize,
    beta: (f64, f64),
    c_re: &mut [f64],
    c_im: &mut [f64],
    ldc: usize,
) {
    assert!(
        (transa, transb) == ('N', 'T'),
        "openblas_emu::zgemm: unverified (transa, transb) = ({transa}, {transb}); only ('N', 'T') is probed"
    );
    assert!(
        alpha.1 == 0.0,
        "openblas_emu::zgemm: unverified complex alpha = {alpha:?}; only real alpha is probed"
    );
    assert!(
        beta == (0.0, 0.0) || beta == (1.0, 0.0),
        "openblas_emu::zgemm: unverified beta = {beta:?}; only 0 and 1 are probed"
    );
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if beta == (0.0, 0.0) {
            for v in c_re.iter_mut().chain(c_im.iter_mut()) {
                *v = 0.0;
            }
        }
        return;
    }
    assert!(a_re.len() >= (k - 1) * lda + m && a_im.len() >= (k - 1) * lda + m);
    assert!(b_re.len() >= (k - 1) * ldb + n && b_im.len() >= (k - 1) * ldb + n);
    assert!(c_re.len() >= (n - 1) * ldc + m && c_im.len() >= (n - 1) * ldc + m);
    let ar = alpha.0;
    let blocks = k_blocks(k, 2);
    let (m2, n2) = (m / 2 * 2, n / 2 * 2);
    let mut rr = Vec::with_capacity(2 * GEMM_Q);
    let mut ii = Vec::with_capacity(2 * GEMM_Q);
    let mut ri = Vec::with_capacity(2 * GEMM_Q);
    let mut ir = Vec::with_capacity(2 * GEMM_Q);
    for j in 0..n {
        for i in 0..m {
            let corner = i >= m2 && j >= n2;
            let sum = |v: &[f64]| if corner { two_lanes(v) } else { seq(v) };
            let (mut cr, mut ci) = if beta == (0.0, 0.0) {
                (0.0, 0.0)
            } else {
                (c_re[i + j * ldc], c_im[i + j * ldc])
            };
            let mut k0 = 0;
            for &kb in &blocks {
                rr.clear();
                ii.clear();
                ri.clear();
                ir.clear();
                for kk in k0..k0 + kb {
                    let (arv, aiv) = (a_re[i + kk * lda], a_im[i + kk * lda]);
                    let (brv, biv) = (b_re[j + kk * ldb], b_im[j + kk * ldb]);
                    rr.push(arv * brv);
                    ii.push(aiv * biv);
                    ri.push(arv * biv);
                    ir.push(aiv * brv);
                }
                cr += ar * (sum(&rr) - sum(&ii));
                ci += ar * (sum(&ri) + sum(&ir));
                k0 += kb;
            }
            c_re[i + j * ldc] = cr;
            c_im[i + j * ldc] = ci;
        }
    }
}

/// `zgemm_('N', 'T', m, n, k, 1+0i, a, m, b, n, beta, c, m)` — `c += a * b^T`
/// (no conjugation) in upstream's summation order, on planar complex
/// column-major arrays. `beta = 0` is a zeroed `c`, as for [`dgemm_nt`].
///
/// Thin wrapper over [`zgemm`] (`alpha = 1`, `beta = 1` accumulate); the op
/// sequence is identical (`1.0 * x` is exact), so the Barcelona fit transfers.
#[allow(clippy::too_many_arguments)]
pub fn zgemm_nt(
    m: usize,
    n: usize,
    k: usize,
    a_re: &[f64],
    a_im: &[f64],
    b_re: &[f64],
    b_im: &[f64],
    c_re: &mut [f64],
    c_im: &mut [f64],
) {
    zgemm(
        'N',
        'T',
        m,
        n,
        k,
        (1.0, 0.0),
        a_re,
        a_im,
        m,
        b_re,
        b_im,
        n,
        (1.0, 0.0),
        c_re,
        c_im,
        m,
    );
}
