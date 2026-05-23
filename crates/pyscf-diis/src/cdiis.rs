//! C-DIIS body — Pulay extrapolation (CDIIS).
//!
//! Source:
//!   - `pyscf/scf/diis.py:40-87` — CDIIS class (`update` + `get_err_vec_orig`)
//!   - Pulay 1980, DOI:10.1016/0009-2614(80)80396-4
//!   - RESEARCH §"Pattern 7" lines 808-893
//!
//! Algorithm:
//!   1. Push (current iterate, error vector) into the ring buffer.
//!   2. Build Pulay's Lagrange-multiplier system:
//!      B has shape (n+1, n+1); B[i,j] = <err_i, err_j> for i,j<n,
//!      B[i,n] = B[n,i] = -1, B[n,n] = 0.
//!      RHS b has shape (n+1); b[i] = 0 for i<n, b[n] = -1.
//!   3. Solve B·c = b via `pyscf_algebra::solve_linear` (host-faer LU).
//!   4. Extrapolated iterate = Σ_{i<n} c[i] · bookkeep[i].
//!
//! Pitfall 9 mitigation: all reductions go through
//! `pyscf_algebra::oracle_dot` (B-matrix inner products) and
//! `pyscf_algebra::oracle_sum` (extrapolated-iterate cross-iterate sums).
//! Threat T-3-09: bit-identical across thread counts.
//!
//! Threat T-3-13 mitigation: `solve_linear` returns
//! `AlgebraError::Singular` on rank-deficient B; we re-package that as
//! `DiisError::Singular` so the caller can fall back to a damped Fock.

use crate::{DiisError, DiisStorable};

/// CDIIS Pulay extrapolation stack.
///
/// Generic over the iterate type `S: DiisStorable + Clone`. SCF uses
/// `Diis<FockSubspace>`; Phase 6 CCSD will use `Diis<AmpsSubspace>` with
/// the same machinery and a different `DiisStorable` impl.
pub struct Diis<S: DiisStorable + Clone> {
    /// Maximum subspace size (default 8 per upstream
    /// `pyscf/scf/hf.py:1701` `diis_space = 8`).
    pub space: usize,
    /// Cycle index at which to start extrapolating (default 1 per
    /// `pyscf/scf/hf.py:1704` `diis_start_cycle = 1`). The caller (SCF
    /// kernel loop) decides when to call `extrapolate`; this field is
    /// documentation only on the Diis struct.
    pub start_cycle: usize,
    /// Ring buffer of past iterates. Length grows to `space` then wraps.
    bookkeep: Vec<S>,
    /// Ring buffer of past error vectors (parallel to `bookkeep`).
    error_vecs: Vec<Vec<f64>>,
    /// Ring-buffer head: index of the oldest entry (where the next push
    /// overwrites once the buffer is full).
    head: usize,
}

impl<S: DiisStorable + Clone> Diis<S> {
    /// New extrapolation stack with subspace size `space`.
    /// Panics if `space == 0` (caller bug — no buffer to extrapolate in).
    pub fn new(space: usize) -> Self {
        assert!(space > 0, "DIIS subspace must be > 0");
        Self {
            space,
            start_cycle: 1,
            bookkeep: Vec::with_capacity(space),
            error_vecs: Vec::with_capacity(space),
            head: 0,
        }
    }

    /// Push `(current, error)` into the ring buffer, maintaining a
    /// `space`-sized window. When full, overwrite the oldest entry and
    /// advance the head.
    fn push(&mut self, current: S, error: Vec<f64>) {
        if self.bookkeep.len() < self.space {
            self.bookkeep.push(current);
            self.error_vecs.push(error);
        } else {
            self.bookkeep[self.head] = current;
            self.error_vecs[self.head] = error;
            self.head = (self.head + 1) % self.space;
        }
    }

    /// Pulay extrapolation: push the new iterate, build the B-matrix +
    /// solve, return `Σ_i c[i] · bookkeep[i]` as the extrapolated iterate.
    ///
    /// Source: `pyscf/scf/diis.py:48-58` (`update` method).
    pub fn extrapolate(&mut self, current: S, error: Vec<f64>) -> Result<S, DiisError> {
        self.push(current, error);
        let n = self.bookkeep.len();
        let dim = n + 1;

        // Build B in row-major flat layout.
        //   B[i,j] = oracle_dot(err_i, err_j)   for i,j < n   (Pitfall 9)
        //   B[i,n] = B[n,i] = -1                 for i < n
        //   B[n,n] = 0
        let mut b = vec![0.0_f64; dim * dim];
        for i in 0..n {
            for j in 0..n {
                b[i * dim + j] =
                    pyscf_algebra::oracle_dot(&self.error_vecs[i], &self.error_vecs[j]);
            }
            b[i * dim + n] = -1.0;
            b[n * dim + i] = -1.0;
        }
        b[n * dim + n] = 0.0;

        // RHS = [0, 0, …, 0, -1].
        let mut rhs = vec![0.0_f64; dim];
        rhs[n] = -1.0;

        // Solve via pyscf-algebra::solve_linear (plan 03-01 host-faer LU).
        // Threat T-3-13: AlgebraError::Singular → DiisError::Singular.
        let c = pyscf_algebra::solve_linear(&b, &rhs, dim).map_err(|e| match e {
            pyscf_algebra::AlgebraError::Singular => DiisError::Singular,
            other => DiisError::Algebra(other),
        })?;

        // Extrapolated iterate: F_new[k] = Σ_{i<n} c[i] · bookkeep[i].as_flat()[k].
        // Cross-iterate reduction must go through oracle_sum (Pitfall 9).
        let flat_len = self.bookkeep[0].len();
        let mut extrap_flat = vec![0.0_f64; flat_len];
        // Scratch buffer reused per element to keep allocation cost low.
        // oracle_sum takes &[f64], so we materialise the per-iterate
        // contributions for each `k`.
        let mut terms = vec![0.0_f64; n];
        for (k, extrap) in extrap_flat.iter_mut().enumerate() {
            for i in 0..n {
                terms[i] = c[i] * self.bookkeep[i].as_flat()[k];
            }
            *extrap = pyscf_algebra::oracle_sum(&terms);
        }
        let mut extrap = self.bookkeep[0].clone();
        extrap.from_flat(&extrap_flat);
        Ok(extrap)
    }

    /// Number of currently stored iterates (≤ `space`).
    pub fn len(&self) -> usize {
        self.bookkeep.len()
    }

    /// `true` when no iterates have been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.bookkeep.is_empty()
    }
}

/// SDF − FDS error vector for SCF use.
///
/// Source: `pyscf/scf/diis.py:68-87` (`get_err_vec_orig`).
///
/// Inputs: `s`, `d`, `f` are `nao × nao` row-major matrices (matching
/// `pyscf-core::Density.data`'s documented row-major convention).
/// Output: `nao*nao` row-major flat error vector `SDF − FDS`.
///
/// Implementation note: `pyscf_algebra::gemm` is Tensor-based and currently
/// `NotYetImplemented{phase:2}`. We therefore use an O(nao³) explicit
/// matmul via `oracle_dot` per row × column — mirroring the pattern in
/// `pyscf-scf::rdm::default_make_rdm1`. When Tensor gemm lands, callers
/// can swap in a faster route without changing this fn's signature.
pub fn err_vec_scf(s: &[f64], d: &[f64], f: &[f64], nao: usize) -> Vec<f64> {
    debug_assert_eq!(s.len(), nao * nao);
    debug_assert_eq!(d.len(), nao * nao);
    debug_assert_eq!(f.len(), nao * nao);

    // sdf = S · D · F   (row-major chain)
    let sd = matmul_row_major(s, d, nao);
    let sdf = matmul_row_major(&sd, f, nao);
    // fds = F · D · S
    let fd = matmul_row_major(f, d, nao);
    let fds = matmul_row_major(&fd, s, nao);
    // err = sdf - fds elementwise.
    sdf.iter().zip(&fds).map(|(a, b)| a - b).collect()
}

/// Row-major nao×nao matmul: `out[i,j] = Σ_k a[i,k] · b[k,j]`.
///
/// Implemented as an explicit triple loop over (i, j) with an inner
/// reduction. Uses standard `f64` multiply-add accumulation (NOT oracle_*)
/// since this fn is a host-fallback for `gemm` — Tensor gemm will be the
/// determinism-preserving path once it lands. For DIIS use, the error
/// vector is only used to build B-matrix inner products (and those DO go
/// through `oracle_dot`), so the determinism guarantee at the err_vec
/// layer is not load-bearing.
fn matmul_row_major(a: &[f64], b: &[f64], nao: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            let mut s = 0.0_f64;
            for k in 0..nao {
                s += a[i * nao + k] * b[k * nao + j];
            }
            out[i * nao + j] = s;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct V(Vec<f64>);
    impl DiisStorable for V {
        fn as_flat(&self) -> &[f64] {
            &self.0
        }
        fn from_flat(&mut self, s: &[f64]) {
            self.0.copy_from_slice(s);
        }
        fn dot(&self, o: &Self) -> f64 {
            pyscf_algebra::oracle_dot(&self.0, &o.0)
        }
        fn len(&self) -> usize {
            self.0.len()
        }
    }

    #[test]
    fn extrapolate_smoke() {
        let mut diis = Diis::<V>::new(8);
        let out = diis
            .extrapolate(V(vec![1.0, 2.0]), vec![0.1, 0.2])
            .expect("smoke");
        // n=1: c = [1, _]; extrapolated = current iterate.
        assert!((out.0[0] - 1.0).abs() < 1e-12);
        assert!((out.0[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn err_vec_zero_when_dm_commutes() {
        // If F·D = D·F = D·F·S (handpicked), the err vector is zero.
        // Simplest: S = I, D = F → SDF = D·F = F·F = FDS, so err = 0.
        let nao = 2;
        let s = vec![1.0, 0.0, 0.0, 1.0]; // I
        let d = vec![1.0, 0.5, 0.5, 1.0]; // arbitrary symmetric
        let f = d.clone(); // F = D
        let err = err_vec_scf(&s, &d, &f, nao);
        for (i, &e) in err.iter().enumerate() {
            assert!(
                e.abs() < 1e-12,
                "err[{}] = {} (expected ~0 when F = D, S = I)",
                i,
                e
            );
        }
    }
}
