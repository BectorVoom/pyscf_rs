//! Ordered complex reductions — D-PBC-17.
//!
//! These are the ONLY reductions numerical PBC code may use. The device
//! reductions in [`crate::zblas`] (`zdotc_dense`, `zreduce_sum_dense`) sum in
//! whatever order the launch geometry produces; that is fine for diagnostics
//! and norms, but any quantity that lands in an energy, a density matrix or a
//! convergence test must be thread-count-invariant.
//!
//! Both functions delegate to the FOUND-06 pairwise reducers
//! ([`crate::oracle_sum`] / [`crate::oracle_dot`]), whose recursion-tree shape
//! depends only on input length and the fixed chunk size — never on rayon
//! thread count, scheduler or partition. Splitting the complex reduction into
//! independent per-plane real reductions preserves that property exactly: each
//! plane is its own deterministic tree, and the two results are never mixed.

use crate::complex::CTensor;
use crate::{oracle_dot, oracle_sum};

/// Bit-deterministic complex sum `Σ x[i]`, as two [`oracle_sum`] calls
/// (one per plane). Invariant under `RAYON_NUM_THREADS`.
pub fn oracle_zsum(x: &CTensor) -> (f64, f64) {
    (oracle_sum(&x.re), oracle_sum(&x.im))
}

/// Bit-deterministic conjugated inner product `xᴴ · y = Σ conj(x[i]) * y[i]`,
/// as four [`oracle_dot`] calls combined in the `zdotc` pattern:
///
/// ```text
/// re = dot(xr, yr) + dot(xi, yi)
/// im = dot(xr, yi) − dot(xi, yr)
/// ```
///
/// On a length mismatch each `oracle_dot` returns NaN, so the result is
/// `(NaN, NaN)` — matching [`oracle_dot`]'s panic-free contract (callers verify
/// shapes upstream).
///
/// This is `zdotc`. KMP2's already-conjugated contractions require
/// [`oracle_zdotu`] instead (Phase 15 context §3.10).
pub fn oracle_zdot(x: &CTensor, y: &CTensor) -> (f64, f64) {
    let rr = oracle_dot(&x.re, &y.re);
    let ii = oracle_dot(&x.im, &y.im);
    let ri = oracle_dot(&x.re, &y.im);
    let ir = oracle_dot(&x.im, &y.re);
    (rr + ii, ri - ir)
}

/// Bit-deterministic unconjugated inner product `xᵀ · y = Σ x[i] * y[i]`.
pub fn oracle_zdotu(x: &CTensor, y: &CTensor) -> (f64, f64) {
    let rr = oracle_dot(&x.re, &y.re);
    let ii = oracle_dot(&x.im, &y.im);
    let ri = oracle_dot(&x.re, &y.im);
    let ir = oracle_dot(&x.im, &y.re);
    (rr - ii, ri + ir)
}

/// Real part of [`oracle_zdotu`], avoiding the two imaginary-output products.
pub fn oracle_zdotu_re(x: &CTensor, y: &CTensor) -> f64 {
    oracle_dot(&x.re, &y.re) - oracle_dot(&x.im, &y.im)
}

/// Real part of the conjugated [`oracle_zdot`] product.
pub fn oracle_zdot_re(x: &CTensor, y: &CTensor) -> f64 {
    oracle_dot(&x.re, &y.re) + oracle_dot(&x.im, &y.im)
}
