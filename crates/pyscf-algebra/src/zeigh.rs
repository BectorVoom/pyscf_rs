//! Complex Hermitian linear algebra — PBC-MASTER-PLAN §5.3 / D-PBC-04.
//!
//! Three routines, each shipping TWO independent routes:
//!
//! | routine | primary route (`FAER_C64 == true`) | mandated second route |
//! |---|---|---|
//! | [`zeigh_gen`] | faer `c64` Löwdin transform | real `2n × 2n` embedding through [`crate::eigh_gen`] |
//! | [`zcholesky`] | faer `c64` `Llt` | explicit complex Crout factorization in real arithmetic |
//! | [`zsolve_linear`] | faer `c64` `PartialPivLu` | real `2n × 2n` embedding through [`crate::solve_linear`] |
//!
//! The second route is written and tested even when the first one works — it IS
//! the CI cross-check (§5.3). `zeigh_gen` additionally cross-checks itself
//! against the embedding on every debug-build call for `n <= 16`.
//!
//! # D-PBC-04 — the probe result
//!
//! Plan 09-02 Task 0 probed faer 0.24 with a throwaway example
//! (`examples/faer_c64_probe.rs`, deleted afterwards). Verbatim output:
//!
//! ```text
//! eigenvalues: Complex { re: 0.9999999999999998, im: 0.0 } Complex { re: 4.0, im: 0.0 }
//! EIGH_OK = true
//! LLT_OK = true
//! LU_RESID = 2.7755575615628914e-17
//! LU_OK = true
//! FAER_C64 = true
//! ```
//!
//! faer 0.24 therefore has a working native complex `SelfAdjointEigen`, `Llt`
//! and `PartialPivLu`, and [`FAER_C64`] is `true`.
//!
//! # Layout
//!
//! Inputs (`f`, `s`, `a`) are ROW-MAJOR `n × n` Hermitian [`CTensor`]s. The
//! returned eigenvector matrix is COLUMN-MAJOR (F-order) `n × n`, matching
//! [`crate::eigh_gen`] and `pyscf_core::MOCoefficients`. `zcholesky` returns a
//! ROW-MAJOR lower-triangular factor; `zsolve_linear` returns a length-`n`
//! vector.
//!
//! # Phase convention (Pitfall 4)
//!
//! A complex eigenvector is defined only up to a global phase `e^{iθ}`, and the
//! two routes pick different ones. Both routes therefore run the same fix
//! before returning: rotate each column so that its largest-modulus component is
//! real and non-negative, then apply `pyscf_core::canonicalize_signs` to the
//! real part (mirroring any sign flip it makes onto the imaginary part, so the
//! column stays an eigenvector). After the phase rotation the largest-|re|
//! entry IS the largest-modulus entry and is already positive, so
//! `canonicalize_signs` is a no-op in practice — it is retained because §5.3
//! mandates it and it is the same vendor-stability hook the molecular path uses.

use crate::complex::CTensor;
use crate::{AlgebraError, eigh_gen, solve_linear};
use faer::linalg::solvers::{Llt, PartialPivLu, SelfAdjointEigen, Solve};
use faer::{Mat, Side, c64};

/// D-PBC-04 outcome: faer 0.24 ships a working native `c64` eigensolver,
/// Cholesky and LU (see the module docs for the probe transcript), so the
/// `c64` routes are the primary implementation and the real-embedding /
/// Crout routes are the cross-check.
pub(crate) const FAER_C64: bool = true;

/// Largest `n` for which [`zeigh_gen`] debug-cross-checks its result against
/// [`zeigh_gen_embedding`]. The embedding route costs an `O((2n)^3)` real eigh,
/// so the guard is capped rather than unconditional.
const CROSS_CHECK_MAX_N: usize = 16;

/// Eigenvalue agreement demanded by the debug cross-check.
const CROSS_CHECK_TOL: f64 = 1e-11;

/// One accepted embedding eigenvector: `(re, im, (S·v).re, (S·v).im)`. Caching
/// `S·v` alongside the vector turns the S-metric overlap test into one O(n) dot
/// instead of a matvec per candidate.
type AcceptedVec = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Below this `S`-norm an extracted embedding eigenvector is treated as
/// linearly dependent on the ones already accepted (degenerate-pair collision).
const EMBED_ORTHO_TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Small host helpers. This module is on the ALG-05 host-fallback path (like
// `eigh_gen` / `solve_linear`), so the O(n^2) glue runs on the host in plain
// real arithmetic rather than round-tripping through the device.
// ---------------------------------------------------------------------------

fn check_square(name: &'static str, x: &CTensor, n: usize) -> Result<(), AlgebraError> {
    if x.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("{name} len {} (= {n}*{n})", n * n),
            actual: x.len().to_string(),
        });
    }
    Ok(())
}

/// `y = A · v` for a ROW-MAJOR `n × n` complex `A` and a length-`n` complex `v`.
fn zmatvec(a: &CTensor, vr: &[f64], vi: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut yr = vec![0.0_f64; n];
    let mut yi = vec![0.0_f64; n];
    for i in 0..n {
        let (mut acc_r, mut acc_i) = (0.0_f64, 0.0_f64);
        for j in 0..n {
            let (ar, ai) = (a.re[i * n + j], a.im[i * n + j]);
            acc_r += ar * vr[j] - ai * vi[j];
            acc_i += ar * vi[j] + ai * vr[j];
        }
        yr[i] = acc_r;
        yi[i] = acc_i;
    }
    (yr, yi)
}

/// `uᴴ · v` for length-`n` complex vectors.
fn zdot_h(ur: &[f64], ui: &[f64], vr: &[f64], vi: &[f64]) -> (f64, f64) {
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for k in 0..ur.len() {
        re += ur[k] * vr[k] + ui[k] * vi[k];
        im += ur[k] * vi[k] - ui[k] * vr[k];
    }
    debug_assert_eq!(ur.len(), vr.len(), "zdot_h: length mismatch");
    debug_assert_eq!(ui.len(), vi.len(), "zdot_h: length mismatch");
    (re, im)
}

/// Rotate one column so its largest-modulus entry becomes real and
/// non-negative, killing the arbitrary global phase both routes would
/// otherwise return (Pitfall 4).
fn fix_phase(cr: &mut [f64], ci: &mut [f64]) {
    let mut p = 0usize;
    let mut best = -1.0_f64;
    for k in 0..cr.len() {
        // Compare squared moduli — same ordering, no sqrt. STRICT `>` so ties
        // resolve to the lowest index (matching `canonicalize_signs`).
        let m = cr[k] * cr[k] + ci[k] * ci[k];
        if m > best {
            best = m;
            p = k;
        }
    }
    let modulus = best.sqrt();
    if modulus == 0.0 {
        return;
    }
    // Multiply by conj(c[p]) / |c[p]|.
    let (ur, ui) = (cr[p] / modulus, -ci[p] / modulus);
    for k in 0..cr.len() {
        let (r, i) = (cr[k], ci[k]);
        cr[k] = r * ur - i * ui;
        ci[k] = r * ui + i * ur;
    }
}

/// Apply `pyscf_core::canonicalize_signs` to the real part of an F-order
/// eigenvector buffer and MIRROR every sign flip it makes onto the imaginary
/// part, so each column remains the same eigenvector (up to the intended sign)
/// rather than an unrelated mixture.
fn canonicalize_complex_signs(c: &mut CTensor, n: usize) {
    let before = c.re.clone();
    pyscf_core::canonicalize_signs(&mut c.re, n, n);
    for j in 0..n {
        let col = j * n;
        // A flipped column has its first non-zero entry negated; comparing one
        // representative entry is enough because the flip is column-global.
        let flipped = (0..n).any(|i| before[col + i] != 0.0 && before[col + i] == -c.re[col + i]);
        if flipped {
            for i in 0..n {
                c.im[col + i] = -c.im[col + i];
            }
        }
    }
}

/// Normalize an F-order eigenvector buffer so that `Cᴴ S C = I` column-wise,
/// then apply the phase convention.
fn normalize_and_phase(c: &mut CTensor, s: &CTensor, n: usize) {
    for j in 0..n {
        let col = j * n;
        let (vr, vi) = (c.re[col..col + n].to_vec(), c.im[col..col + n].to_vec());
        let (sr, si) = zmatvec(s, &vr, &vi, n);
        // vᴴ S v is real for Hermitian S; take the real part.
        let (norm_sq, _) = zdot_h(&vr, &vi, &sr, &si);
        if norm_sq > 0.0 {
            let inv = 1.0 / norm_sq.sqrt();
            for i in 0..n {
                c.re[col + i] *= inv;
                c.im[col + i] *= inv;
            }
        }
        fix_phase(&mut c.re[col..col + n], &mut c.im[col..col + n]);
    }
    canonicalize_complex_signs(c, n);
}

// ---------------------------------------------------------------------------
// zeigh_gen — the real 2n x 2n embedding route (ALWAYS built, §5.3).
// ---------------------------------------------------------------------------

/// Embed a Hermitian `H = Hr + i·Hi` as the real SYMMETRIC `2n × 2n`
/// ```text
/// M = [ Hr  -Hi ]
///     [ Hi   Hr ]
/// ```
/// (symmetric because `Hr` is symmetric and `Hi` is antisymmetric). Returned
/// row-major, `2n × 2n`. The eigenvalues of `M` are those of `H`, each twice.
fn embed_hermitian(h: &CTensor, n: usize) -> Vec<f64> {
    let m = 2 * n;
    let mut out = vec![0.0_f64; m * m];
    for i in 0..n {
        for j in 0..n {
            let (hr, hi) = (h.re[i * n + j], h.im[i * n + j]);
            out[i * m + j] = hr;
            out[i * m + (n + j)] = -hi;
            out[(n + i) * m + j] = hi;
            out[(n + i) * m + (n + j)] = hr;
        }
    }
    out
}

/// Extract complex eigenvector `k` from the F-order real embedding buffer
/// `c_embed` (`2n × 2n`), column `col`: top half is the real part, bottom half
/// the imaginary part.
fn extract_embedded_column(c_embed: &[f64], col: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let m = 2 * n;
    let base = col * m;
    (
        c_embed[base..base + n].to_vec(),
        c_embed[base + n..base + m].to_vec(),
    )
}

/// The MANDATED real `2n × 2n` route of PBC-MASTER-PLAN §5.3 — solve the complex
/// generalized Hermitian problem `F·C = S·C·diag(ε)` using only the existing
/// REAL [`crate::eigh_gen`].
///
/// Steps, in the order §5.3 fixes them:
/// 1. build `M` and `S_M` (see [`embed_hermitian`]) and call `eigh_gen`;
/// 2. eigenvalues come back ascending, each of `H`'s appearing twice — take
///    indices `0, 2, 4, …, 2n−2`;
/// 3. eigenvector `k` is `C[0..n] + i·C[n..2n]` from column `2k`;
/// 4. normalize so `Cᴴ S C = I`, fix the global phase, then apply
///    `pyscf_core::canonicalize_signs` on the real part.
///
/// # Degenerate eigenvalues
/// Step 2's fixed stride assumes each of `H`'s eigenvalues has multiplicity one
/// *in `H`* (multiplicity two in `M`). When `H` itself is degenerate — routine
/// at high-symmetry k-points — `M`'s eigenvectors for that value span a 4-real-
/// dimensional space and columns `2k`, `2k+2` can land on the SAME complex
/// direction. This function detects that (the newly extracted vector has no
/// `S`-component orthogonal to the ones already accepted) and falls back to
/// scanning ALL `2n` columns, greedily accepting the next one that is
/// `S`-orthogonal to the accepted set. Non-degenerate inputs never reach the
/// fallback and get exactly the `0, 2, 4, …` columns the plan mandates.
pub fn zeigh_gen_embedding(
    f: &CTensor,
    s: &CTensor,
    n: usize,
) -> Result<(Vec<f64>, CTensor), AlgebraError> {
    check_square("f", f, n)?;
    check_square("s", s, n)?;
    if n == 0 {
        return Ok((Vec::new(), CTensor::zeros(0)));
    }

    let m = 2 * n;
    let f_embed = embed_hermitian(f, n);
    let s_embed = embed_hermitian(s, n);
    let (evals_embed, c_embed) = eigh_gen(&f_embed, &s_embed, m)?;

    let mut eigenvalues = Vec::with_capacity(n);
    let mut c = CTensor::zeros(n * n);
    // Accepted columns, kept as (v, S·v) so the orthogonality test is O(n) each.
    let mut accepted: Vec<AcceptedVec> = Vec::with_capacity(n);

    // Candidate order: the mandated even columns first, then the odd ones as the
    // degeneracy fallback.
    let candidates = (0..n)
        .map(|k| 2 * k)
        .chain((0..n).map(|k| 2 * k + 1))
        .collect::<Vec<_>>();

    for col in candidates {
        if accepted.len() == n {
            break;
        }
        // `eigh_gen` pads dropped linearly-dependent directions with +inf.
        if !evals_embed[col].is_finite() {
            continue;
        }
        let (mut vr, mut vi) = extract_embedded_column(&c_embed, col, n);
        // Project out the already-accepted directions in the S metric.
        for (ur, ui, sur, sui) in &accepted {
            // `(S·u)ᴴ v == uᴴ S v` because S is Hermitian — so the stored `S·u`
            // gives the S-metric overlap with one O(n) dot instead of a matvec.
            let (qr, qi) = zdot_h(sur, sui, &vr, &vi);
            for i in 0..n {
                vr[i] -= qr * ur[i] - qi * ui[i];
                vi[i] -= qr * ui[i] + qi * ur[i];
            }
        }
        let (svr, svi) = zmatvec(s, &vr, &vi, n);
        let (norm_sq, _) = zdot_h(&vr, &vi, &svr, &svi);
        if norm_sq <= EMBED_ORTHO_TOL {
            // Degenerate collision — this column adds nothing new.
            continue;
        }
        let inv = 1.0 / norm_sq.sqrt();
        for i in 0..n {
            vr[i] *= inv;
            vi[i] *= inv;
        }
        let (svr, svi) = zmatvec(s, &vr, &vi, n);
        let k = accepted.len();
        eigenvalues.push(evals_embed[col]);
        let base = k * n;
        c.re[base..base + n].copy_from_slice(&vr);
        c.im[base..base + n].copy_from_slice(&vi);
        accepted.push((vr, vi, svr, svi));
    }

    if accepted.len() != n {
        return Err(AlgebraError::CubeclRuntime(format!(
            "zeigh_gen_embedding: recovered only {} of {n} S-orthogonal eigenvectors \
             from the 2n x 2n embedding (S may be singular)",
            accepted.len()
        )));
    }

    // Eigenvalues are picked in embedding order, which is ascending; the
    // degeneracy fallback can visit an odd column out of order, so re-sort the
    // (eigenvalue, column) pairs together.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    if order.iter().enumerate().any(|(i, &j)| i != j) {
        let sorted_vals: Vec<f64> = order.iter().map(|&j| eigenvalues[j]).collect();
        let mut sorted_c = CTensor::zeros(n * n);
        for (i, &j) in order.iter().enumerate() {
            sorted_c.re[i * n..i * n + n].copy_from_slice(&c.re[j * n..j * n + n]);
            sorted_c.im[i * n..i * n + n].copy_from_slice(&c.im[j * n..j * n + n]);
        }
        eigenvalues = sorted_vals;
        c = sorted_c;
    }

    normalize_and_phase(&mut c, s, n);
    Ok((eigenvalues, c))
}

// ---------------------------------------------------------------------------
// zeigh_gen — the faer c64 route (primary, FAER_C64 == true).
// ---------------------------------------------------------------------------

/// Build a faer `c64` matrix from a ROW-MAJOR `n × n` [`CTensor`].
fn to_faer(x: &CTensor, n: usize) -> Mat<c64> {
    Mat::<c64>::from_fn(n, n, |i, j| c64::new(x.re[i * n + j], x.im[i * n + j]))
}

/// Native complex Löwdin route: the exact algorithm [`crate::eigh_gen`] runs,
/// lifted to `c64` (`S = U·diag(s)·Uᴴ`, `X = U·diag(s^{-1/2})` with
/// linear-dependency removal, `F' = Xᴴ·F·X`, `C = X·V`).
///
/// Only compiled/reachable while [`FAER_C64`] is `true`.
pub fn zeigh_gen_faer(
    f: &CTensor,
    s: &CTensor,
    n: usize,
) -> Result<(Vec<f64>, CTensor), AlgebraError> {
    check_square("f", f, n)?;
    check_square("s", s, n)?;
    if n == 0 {
        return Ok((Vec::new(), CTensor::zeros(0)));
    }

    // === Step 1: eigh(S) ===
    let s_mat = to_faer(s, n);
    let s_evd = SelfAdjointEigen::new(s_mat.as_ref(), Side::Lower)
        .map_err(|e| AlgebraError::CubeclRuntime(format!("zeigh: eigh(S) failed: {e:?}")))?;
    let s_evals = s_evd.S();
    let s_evecs = s_evd.U();

    // === Step 2: X = U · diag(s^{-1/2}) with linear-dependency removal ===
    let mut valid_cols: Vec<(usize, f64)> = Vec::with_capacity(n);
    for j in 0..n {
        let lam = s_evals[j].re;
        if lam > crate::eigh_gen::S_LINEAR_DEP_TOL {
            valid_cols.push((j, 1.0 / lam.sqrt()));
        }
    }
    if valid_cols.is_empty() {
        return Err(AlgebraError::Singular);
    }
    let n_lin = valid_cols.len();
    let x = Mat::<c64>::from_fn(n, n_lin, |i, k| {
        let (j, inv_sqrt) = valid_cols[k];
        s_evecs[(i, j)] * c64::new(inv_sqrt, 0.0)
    });

    // === Step 3: F' = Xᴴ · F · X (n_lin × n_lin, Hermitian) ===
    let f_mat = to_faer(f, n);
    let xh_f = x.adjoint() * &f_mat;
    let fp = &xh_f * &x;

    // === Step 4: eigh(F') ===
    let fp_evd = SelfAdjointEigen::new(fp.as_ref(), Side::Lower)
        .map_err(|e| AlgebraError::CubeclRuntime(format!("zeigh: eigh(F') failed: {e:?}")))?;
    let fp_evals = fp_evd.S();
    let v = fp_evd.U();

    // === Step 5: C = X · V (n × n_lin) ===
    let c_lin = &x * v;

    // Pack: eigenvalues padded with +inf for the dropped linearly-dependent
    // directions; C is F-order with the dropped columns left at zero — exactly
    // the convention `eigh_gen` documents.
    let mut eigenvalues = vec![f64::INFINITY; n];
    for (i, e) in eigenvalues.iter_mut().enumerate().take(n_lin) {
        *e = fp_evals[i].re;
    }
    let mut c = CTensor::zeros(n * n);
    for j in 0..n_lin {
        for i in 0..n {
            c.re[i + j * n] = c_lin[(i, j)].re;
            c.im[i + j * n] = c_lin[(i, j)].im;
        }
    }

    normalize_and_phase(&mut c, s, n);
    Ok((eigenvalues, c))
}

/// Solve the complex generalized Hermitian eigenproblem `F·C = S·C·diag(ε)`.
///
/// `f` and `s` are ROW-MAJOR `n × n` Hermitian [`CTensor`]s; the returned
/// eigenvalues are ascending and the returned `C` is COLUMN-MAJOR (F-order)
/// `n × n` normalized so that `Cᴴ S C = I`, with the phase convention described
/// in the module docs.
///
/// Because [`FAER_C64`] is `true` this dispatches to [`zeigh_gen_faer`]. In
/// debug builds and for `n <= 16` the eigenvalues are cross-checked against
/// [`zeigh_gen_embedding`] to [`CROSS_CHECK_TOL`]. Only the EIGENVALUES are
/// compared: an eigenvector is defined up to a global phase, and while both
/// routes apply the same phase convention, a degenerate eigenvalue leaves the
/// eigen*space* basis genuinely route-dependent.
pub fn zeigh_gen(f: &CTensor, s: &CTensor, n: usize) -> Result<(Vec<f64>, CTensor), AlgebraError> {
    if !FAER_C64 {
        return zeigh_gen_embedding(f, s, n);
    }
    let out = zeigh_gen_faer(f, s, n)?;
    #[cfg(debug_assertions)]
    if n <= CROSS_CHECK_MAX_N
        && n > 0
        && let Ok((embed_vals, _)) = zeigh_gen_embedding(f, s, n)
    {
        for (i, (a, b)) in out.0.iter().zip(embed_vals.iter()).enumerate() {
            if a.is_finite() && b.is_finite() {
                debug_assert!(
                    (a - b).abs() <= CROSS_CHECK_TOL * (1.0 + a.abs()),
                    "D-PBC-04 cross-check failed at eigenvalue {i}: \
                     faer c64 route {a} vs 2n x 2n embedding {b}"
                );
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// zcholesky
// ---------------------------------------------------------------------------

/// Lower-triangular complex Cholesky factor of a Hermitian positive-definite
/// `A`, computed by an explicit Crout recurrence in REAL arithmetic — the
/// second route mandated by §5.3.
///
/// NOTE the deviation from the eigh/solve pattern: the real `2n × 2n` embedding
/// is NOT usable for Cholesky. `M = [[Ar,−Ai],[Ai,Ar]]` factors as
/// `M_L · M_Lᵀ` with `M_L = [[Lr,−Li],[Li,Lr]]`, which is block-lower-triangular
/// but NOT element-wise lower triangular (the `−Li` block sits above the
/// diagonal), so a real Cholesky of `M` returns a different factor that does not
/// unpack into `L`. The independent cross-check is therefore this direct
/// factorization, which shares no code with faer.
///
/// Returns `L` ROW-MAJOR `n × n`, strictly-upper entries zero, with `A = L·Lᴴ`.
pub fn zcholesky_crout(a: &CTensor, n: usize) -> Result<CTensor, AlgebraError> {
    check_square("a", a, n)?;
    let mut l = CTensor::zeros(n * n);
    for i in 0..n {
        for j in 0..=i {
            // acc = A[i][j] - Σ_{k<j} L[i][k] * conj(L[j][k])
            let mut acc_r = a.re[i * n + j];
            let mut acc_i = a.im[i * n + j];
            for k in 0..j {
                let (ar, ai) = (l.re[i * n + k], l.im[i * n + k]);
                let (br, bi) = (l.re[j * n + k], -l.im[j * n + k]);
                acc_r -= ar * br - ai * bi;
                acc_i -= ar * bi + ai * br;
            }
            if i == j {
                if acc_r <= 0.0 {
                    return Err(AlgebraError::Singular);
                }
                l.re[i * n + j] = acc_r.sqrt();
                l.im[i * n + j] = 0.0;
            } else {
                // Divide by the REAL diagonal L[j][j].
                let d = l.re[j * n + j];
                if d == 0.0 {
                    return Err(AlgebraError::Singular);
                }
                l.re[i * n + j] = acc_r / d;
                l.im[i * n + j] = acc_i / d;
            }
        }
    }
    Ok(l)
}

/// Native faer `c64` `Llt`. Only reachable while [`FAER_C64`] is `true`.
pub fn zcholesky_faer(a: &CTensor, n: usize) -> Result<CTensor, AlgebraError> {
    check_square("a", a, n)?;
    if n == 0 {
        return Ok(CTensor::zeros(0));
    }
    let mat = to_faer(a, n);
    let llt = Llt::new(mat.as_ref(), Side::Lower).map_err(|_| AlgebraError::Singular)?;
    let lref = llt.L();
    let mut l = CTensor::zeros(n * n);
    for i in 0..n {
        for j in 0..=i {
            // faer's `L()` exposes the whole storage; only the lower triangle is
            // meaningful, so the strict upper triangle is left at zero.
            l.re[i * n + j] = lref[(i, j)].re;
            l.im[i * n + j] = lref[(i, j)].im;
        }
    }
    Ok(l)
}

/// Cholesky factorization `A = L·Lᴴ` of a Hermitian positive-definite `A`
/// (ROW-MAJOR `n × n`). Returns the ROW-MAJOR lower-triangular `L`.
///
/// Dispatches to [`zcholesky_faer`] because [`FAER_C64`] is `true`; the
/// mandated second route is [`zcholesky_crout`].
pub fn zcholesky(a: &CTensor, n: usize) -> Result<CTensor, AlgebraError> {
    if FAER_C64 {
        zcholesky_faer(a, n)
    } else {
        zcholesky_crout(a, n)
    }
}

// ---------------------------------------------------------------------------
// zsolve_linear
// ---------------------------------------------------------------------------

/// The real `2n × 2n` embedding route for `A·z = b`:
/// ```text
/// [ Ar  -Ai ] [ zr ]   [ br ]
/// [ Ai   Ar ] [ zi ] = [ bi ]
/// ```
/// Unlike the Cholesky case this embedding IS exact for a general solve — the
/// block system is literally the real form of the complex equation — so it goes
/// straight through the existing real [`crate::solve_linear`].
///
/// `a` is ROW-MAJOR `n × n`, `b` is a length-`n` vector.
pub fn zsolve_linear_embedding(
    a: &CTensor,
    b: &CTensor,
    n: usize,
) -> Result<CTensor, AlgebraError> {
    check_square("a", a, n)?;
    if b.len() != n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("b len {n}"),
            actual: b.len().to_string(),
        });
    }
    if n == 0 {
        return Ok(CTensor::zeros(0));
    }
    // `embed_hermitian` builds exactly [[Ar,-Ai],[Ai,Ar]]; it does not require
    // `a` to actually be Hermitian.
    let a_embed = embed_hermitian(a, n);
    let mut rhs = Vec::with_capacity(2 * n);
    rhs.extend_from_slice(&b.re);
    rhs.extend_from_slice(&b.im);
    let z = solve_linear(&a_embed, &rhs, 2 * n)?;
    Ok(CTensor::from_planes(z[..n].to_vec(), z[n..].to_vec()))
}

/// Native faer `c64` `PartialPivLu` solve. Only reachable while [`FAER_C64`]
/// is `true`.
pub fn zsolve_linear_faer(a: &CTensor, b: &CTensor, n: usize) -> Result<CTensor, AlgebraError> {
    check_square("a", a, n)?;
    if b.len() != n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("b len {n}"),
            actual: b.len().to_string(),
        });
    }
    if n == 0 {
        return Ok(CTensor::zeros(0));
    }
    let mat = to_faer(a, n);
    let rhs = Mat::<c64>::from_fn(n, 1, |i, _| c64::new(b.re[i], b.im[i]));
    let lu = PartialPivLu::new(mat.as_ref());
    let x = lu.solve(rhs.as_ref());
    let mut out = CTensor::zeros(n);
    for i in 0..n {
        out.re[i] = x[(i, 0)].re;
        out.im[i] = x[(i, 0)].im;
        // faer's LU does not signal singularity; non-finite output does.
        if !out.re[i].is_finite() || !out.im[i].is_finite() {
            return Err(AlgebraError::Singular);
        }
    }
    Ok(out)
}

/// Solve the complex linear system `A·z = b` (`a` ROW-MAJOR `n × n`, `b` a
/// length-`n` vector).
///
/// Dispatches to [`zsolve_linear_faer`] because [`FAER_C64`] is `true`; the
/// mandated second route is [`zsolve_linear_embedding`].
pub fn zsolve_linear(a: &CTensor, b: &CTensor, n: usize) -> Result<CTensor, AlgebraError> {
    if FAER_C64 {
        zsolve_linear_faer(a, b, n)
    } else {
        zsolve_linear_embedding(a, b, n)
    }
}
