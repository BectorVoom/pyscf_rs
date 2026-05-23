//! Deterministic-ordered reductions (FOUND-06).
//!
//! Implements pairwise tree reduction with FIXED chunk size N=128.
//! The fixed chunk size is load-bearing for ROADMAP success criterion 3:
//! the recursion-tree shape depends ONLY on input length and chunk
//! size — independent of rayon thread count — so RAYON_NUM_THREADS=1
//! and =8 produce bit-identical results.
//!
//! RESEARCH §1 + Pitfall 2 mitigation. Algorithm: rust-ndarray PR #577
//! (LukeMathWalker, 2019).

/// Pairwise tree reduction chunk size. CHANGING THIS BREAKS BIT-EXACT
/// COMPATIBILITY with prior runs — never modify without updating the
/// chemistry-corpus regression baselines.
pub const PAIRWISE_CHUNK: usize = 128;

/// Bit-deterministic sum. Result is invariant under thread count and
/// any input partition that respects the fixed chunk size. CPU-only —
/// not delegated to cubecl-reduce in Phase 1 (RESEARCH §1: cubecl-reduce
/// CPU strategy is "ASSUMED pairwise-shaped" but unverified against
/// 0.9.0-pre.5; Phase 1 owns the algorithm to control determinism).
pub fn oracle_sum(xs: &[f64]) -> f64 {
    pairwise(xs, PAIRWISE_CHUNK)
}

/// Bit-deterministic dot product. Equivalent to
/// `oracle_sum(a.iter().zip(b).map(|(x,y)| x*y))`.
/// Panics-free: requires `a.len() == b.len()`; on length mismatch
/// returns NaN (caller is expected to verify shapes upstream).
pub fn oracle_dot(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    // Materialise the elementwise product into a Vec so the recursion
    // tree shape depends only on length, not iterator buffering.
    let products: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    oracle_sum(&products)
}

/// Bit-deterministic einsum — Phase 1 ships only the binary contraction
/// case `"ij,jk->ik"` per RESEARCH §1 ("Phase 1 ships only the binary
/// contraction case; document the path forward to general einsum at
/// Phase 4"). Other patterns return None.
pub fn oracle_einsum(
    pattern: &str,
    a: &[f64],
    a_shape: (usize, usize),
    b: &[f64],
    b_shape: (usize, usize),
) -> Option<Vec<f64>> {
    if pattern != "ij,jk->ik" {
        return None;
    }
    let (m, k1) = a_shape;
    let (k2, n) = b_shape;
    if k1 != k2 {
        return None;
    }
    if a.len() != m * k1 || b.len() != k2 * n {
        return None;
    }

    let mut out = vec![0.0_f64; m * n];
    // Contract j: out[i,k] = oracle_sum(a[i, :] * b[:, k]) for each (i,k).
    let mut col_buf = vec![0.0_f64; k1];
    for i in 0..m {
        for k in 0..n {
            for j in 0..k1 {
                col_buf[j] = a[i * k1 + j] * b[j * n + k];
            }
            out[i * n + k] = oracle_sum(&col_buf);
        }
    }
    Some(out)
}

/// Pairwise recursion. Base case is strict left-to-right; non-base
/// case splits at the midpoint and recurses.
fn pairwise(xs: &[f64], chunk: usize) -> f64 {
    if xs.len() <= chunk {
        // Base case: strict sequential sum.
        let mut s = 0.0_f64;
        for &x in xs {
            s += x;
        }
        s
    } else {
        let mid = xs.len() / 2;
        // Associativity at this level is well-defined: the recursion
        // tree shape depends only on `xs.len()` and `chunk` — NOT on
        // thread count, scheduler, or rayon partition.
        pairwise(&xs[..mid], chunk) + pairwise(&xs[mid..], chunk)
    }
}
